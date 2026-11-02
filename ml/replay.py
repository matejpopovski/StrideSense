"""Feed held-out recordings through the whole device path and into the shipped LiteRT model.

Scoring pre-cut windows says nothing about the code the phone runs. This replays each test recording
as jittered, unsynchronized sensor events, so resampling, filtering, windowing, standardizing and
smoothing all sit between the signal and the score. The recordings were filtered by the dataset
authors, so the 20 Hz stage runs a second time here. A phone's raw signal passes through it once.

python -m ml.replay
"""

import json
from pathlib import Path

import numpy as np

from ml import data
from ml.export_litert import ASSETS, MODEL_FILE, SPEC_FILE
from ml.metrics import summarize
from ml.preprocess import model_input
from ml.runtime import LiteRtModel
from ml.streams import contiguous_segments, simulate_sensors

RESULTS = Path("results")
SMOOTHING_WINDOWS = 3

CONDITIONS = {
    "50 Hz, 1 ms jitter": {"rate_hz": 50.0, "jitter_s": 0.001, "gyro_offset_s": 0.007},
    "50 Hz, 4 ms jitter": {"rate_hz": 50.0, "jitter_s": 0.004, "gyro_offset_s": 0.007},
    "40 Hz, 2 ms jitter": {"rate_hz": 40.0, "jitter_s": 0.002, "gyro_offset_s": 0.007},
    "100 Hz, 1 ms jitter": {"rate_hz": 100.0, "jitter_s": 0.001, "gyro_offset_s": 0.003},
}


def smoothed(probabilities: np.ndarray) -> np.ndarray:
    """ProbabilitySmoother in core/: the mean of each window and the ones just before it."""
    return np.stack(
        [
            probabilities[max(0, i - SMOOTHING_WINDOWS + 1) : i + 1].mean(axis=0)
            for i in range(len(probabilities))
        ]
    )


def replay(segments, model, mean, std, condition: dict, rng) -> dict:
    labels, raw, smooth, subjects = [], [], [], []
    for segment in segments:
        accel, gyro = simulate_sensors(segment.signal, rng=rng, **condition)
        windows = model_input(accel, gyro, mean, std)
        if len(windows) == 0:
            continue
        probabilities = model.predict(windows)
        labels += [segment.label] * len(windows)
        subjects += [segment.subject] * len(windows)
        raw.append(probabilities.argmax(axis=1))
        smooth.append(smoothed(probabilities).argmax(axis=1))
    y, s = np.array(labels), np.array(subjects)
    return {
        "window": summarize(y, np.concatenate(raw), s),
        "smoothed": summarize(y, np.concatenate(smooth), s),
    }


def brief(summary: dict) -> dict:
    keys = ("windows", "accuracy", "macro_f1", "subject_accuracy_min", "subject_accuracy_max")
    return {key: summary[key] for key in keys}


def main() -> None:
    spec = json.loads((ASSETS / SPEC_FILE).read_text())
    mean, std = np.array(spec["mean"]), np.array(spec["std"])
    model = LiteRtModel(ASSETS / MODEL_FILE)
    test = data.load_part("test", data.DEVICE_CHANNELS)
    segments = contiguous_segments(test)

    x = ((test.x - mean) / std).astype(np.float32)
    direct = summarize(test.y, model.predict(x).argmax(axis=1), test.subject)
    report = {
        "note": (
            "Simulated sensor events built from held-out recordings. No phone or emulator was "
            "involved. The 20 Hz filter runs a second time on data the authors already filtered."
        ),
        "recordings": len(segments),
        "direct_windows": brief(direct),
        "conditions": {},
    }
    for name, condition in CONDITIONS.items():
        result = replay(segments, model, mean, std, condition, np.random.default_rng(20))
        report["conditions"][name] = {
            "window": brief(result["window"]),
            "smoothed": brief(result["smoothed"]),
        }
        print(name, f"{result['window']['accuracy']:.4f}", f"{result['smoothed']['accuracy']:.4f}")
    (RESULTS / "replay.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
