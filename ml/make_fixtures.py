"""Write the fixtures that hold Kotlin to the Python reference, from a few real test windows.

fixtures/parity/ holds raw sensor events and the windows ml/preprocess.py makes of them.
app/src/androidTest/assets/ holds windows and the outputs the Python LiteRT runtime gave for them.

python -m ml.make_fixtures
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ml import data, preprocess
from ml.export_litert import ASSETS, MODEL_FILE, SPEC_FILE
from ml.preprocess import SensorStream
from ml.runtime import LiteRtModel
from ml.streams import contiguous_segments, simulate_sensors

PARITY = Path("fixtures/parity")
GOLDEN = Path("app/src/androidTest/assets/golden_windows.json")
FIXTURE_SAMPLES = 352  # four windows, with room for what resampling trims off the ends
GOLDEN_PER_CLASS = 2


def expected(accel: SensorStream, gyro: SensorStream, mean, std) -> tuple[np.ndarray, np.ndarray]:
    per_run = [preprocess.sliding_windows(r) for r in preprocess.filtered_runs(accel, gyro)]
    windows = np.concatenate(per_run)
    return windows, preprocess.standardize(windows, mean, std)


def stream_json(stream: SensorStream) -> dict:
    return {"t": stream.timestamps_ns.tolist(), "v": stream.values.tolist()}


def write_case(name: str, source: str, accel, gyro, mean, std) -> None:
    windows, inputs = expected(accel, gyro, mean, std)
    payload = {
        "name": name,
        "source": source,
        "accelerometer": stream_json(accel),
        "gyroscope": stream_json(gyro),
        "windows": windows.reshape(len(windows), -1).tolist(),
        "model_input": inputs.reshape(len(inputs), -1).tolist(),
    }
    (PARITY / f"{name}.json").write_text(json.dumps(payload) + "\n")
    print(name, "events", len(accel.timestamps_ns), "windows", len(windows))


def without(stream: SensorStream, first_ns: int, last_ns: int) -> SensorStream:
    keep = (stream.timestamps_ns < first_ns) | (stream.timestamps_ns > last_ns)
    return SensorStream(stream.timestamps_ns[keep], stream.values[keep])


def with_stalled_clock(stream: SensorStream, every: int) -> SensorStream:
    times = stream.timestamps_ns.copy()
    times[every::every] = times[every - 1 :: every][: len(times[every::every])]
    return replace(stream, timestamps_ns=times)


def longest(segments, label: int):
    return max((s for s in segments if s.label == label), key=lambda s: len(s.signal))


def main() -> None:
    spec = json.loads((ASSETS / SPEC_FILE).read_text())
    mean, std = np.array(spec["mean"]), np.array(spec["std"])
    test = data.load_part("test", data.DEVICE_CHANNELS)
    segments = contiguous_segments(test)
    PARITY.mkdir(parents=True, exist_ok=True)

    def source(segment) -> str:
        activity = data.ACTIVITIES[segment.label]
        return f"UCI HAR test subject {segment.subject}, {activity}, window {segment.first_window}"

    walking = longest(segments, 0)
    signal = walking.signal[:FIXTURE_SAMPLES]
    rng = np.random.default_rng(20)
    accel, gyro = simulate_sensors(signal, 50.0, 0.002, 0.007, rng)
    write_case("jitter_walking", source(walking), accel, gyro, mean, std)

    stairs = longest(segments, 2)
    accel, gyro = simulate_sensors(stairs.signal[:FIXTURE_SAMPLES], 100.0, 0.001, 0.003, rng)
    write_case("fast_sensors_downstairs", source(stairs), accel, gyro, mean, std)

    sitting = longest(segments, 3)
    accel, gyro = simulate_sensors(sitting.signal[:FIXTURE_SAMPLES], 50.0, 0.001, 0.150, rng)
    write_case("late_gyroscope_sitting", source(sitting), accel, gyro, mean, std)

    upstairs = longest(segments, 1)
    long_signal = upstairs.signal[: FIXTURE_SAMPLES * 2]
    accel, gyro = simulate_sensors(long_signal, 50.0, 0.001, 0.007, rng)
    start = int(accel.timestamps_ns[0])
    gapped = without(gyro, start + 2_400_000_000, start + 2_800_000_000)
    write_case("gyroscope_dropout_upstairs", source(upstairs), accel, gapped, mean, std)

    standing = longest(segments, 4)
    accel, gyro = simulate_sensors(standing.signal[:FIXTURE_SAMPLES], 50.0, 0.001, 0.007, rng)
    stalled = with_stalled_clock(accel, every=17)
    write_case("repeated_timestamps_standing", source(standing), stalled, gyro, mean, std)

    model = LiteRtModel(ASSETS / MODEL_FILE)
    x = ((test.x - mean) / std).astype(np.float32)
    picks = np.concatenate(
        [np.flatnonzero(test.y == label)[::150][:GOLDEN_PER_CLASS] for label in range(6)]
    )
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(
        json.dumps(
            {
                "source": "UCI HAR held-out windows, scaled for the model",
                "model_sha256": spec["model_sha256"],
                "labels": [int(test.y[i]) for i in picks],
                "windows": x[picks].reshape(len(picks), -1).tolist(),
                "probabilities": model.predict(x[picks]).tolist(),
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
