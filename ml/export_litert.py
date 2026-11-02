"""Export the best seed to LiteRT, picking by validation score so the test set has no say.

python -m ml.export_litert --experiment shipped
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from ml import data, preprocess
from ml.model import ActivityLSTM, Probabilities

ASSETS = Path("app/src/main/assets")
MODEL_FILE = "stridesense.tflite"
SPEC_FILE = "preprocessing.json"
RUNS = Path("runs")
EXPORT_RECORD = Path("results/export.json")


def best_run(experiment: str) -> Path:
    results = [
        (json.loads((run / "result.json").read_text())["val_macro_f1"], run)
        for run in sorted((RUNS / experiment).glob("seed*"))
        if (run / "result.json").exists()
    ]
    if not results:
        raise SystemExit(f"no finished runs under {RUNS / experiment}")
    return max(results, key=lambda item: item[0])[1]


def exported_run() -> Path:
    """The run behind the file on disk. best_run() can change as seeds finish, so it cannot say."""
    record = json.loads(EXPORT_RECORD.read_text())
    digest = hashlib.sha256((ASSETS / MODEL_FILE).read_bytes()).hexdigest()
    if digest != record["model_sha256"]:
        raise SystemExit(f"{ASSETS / MODEL_FILE} is not the model {EXPORT_RECORD} describes")
    return Path(record["run"])


def load_model(run: Path) -> tuple[ActivityLSTM, dict]:
    checkpoint = torch.load(run / "best.pt", weights_only=False)
    config = checkpoint["config"]
    model = ActivityLSTM(
        checkpoint["n_channels"], config["hidden"], config["layers"], config["dropout"]
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def pipeline_spec(mean: np.ndarray, std: np.ndarray, model_sha256: str) -> dict:
    b, a, zi = preprocess.butterworth()
    return {
        "sample_rate_hz": preprocess.SAMPLE_RATE_HZ,
        "sample_period_ns": preprocess.SAMPLE_PERIOD_NS,
        "standard_gravity": preprocess.STANDARD_GRAVITY,
        "window_length": preprocess.WINDOW_LENGTH,
        "window_stride": preprocess.WINDOW_STRIDE,
        "max_gap_ns": preprocess.MAX_GAP_NS,
        "filter_b": b.tolist(),
        "filter_a": a.tolist(),
        "filter_zi": zi.tolist(),
        "channels": list(data.DEVICE_CHANNELS),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "labels": list(data.ACTIVITIES),
        "model_file": MODEL_FILE,
        "model_sha256": model_sha256,
    }


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main() -> None:
    import litert_torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    run = best_run(args.experiment)
    model, checkpoint = load_model(run)
    if checkpoint["n_channels"] != len(data.DEVICE_CHANNELS):
        raise SystemExit("only a six-channel model can run on the phone")

    example = torch.zeros(1, preprocess.WINDOW_LENGTH, checkpoint["n_channels"])
    converted = litert_torch.convert(Probabilities(model).eval(), (example,))
    ASSETS.mkdir(parents=True, exist_ok=True)
    EXPORT_RECORD.parent.mkdir(exist_ok=True)
    temporary = ASSETS / f"{MODEL_FILE}.tmp"
    converted.export(str(temporary))
    os.replace(temporary, ASSETS / MODEL_FILE)

    digest = hashlib.sha256((ASSETS / MODEL_FILE).read_bytes()).hexdigest()
    spec = pipeline_spec(checkpoint["mean"], checkpoint["std"], digest)
    write_atomic(ASSETS / SPEC_FILE, (json.dumps(spec, indent=2) + "\n").encode())
    record = {"run": str(run), "seed": checkpoint["seed"], "model_sha256": digest}
    write_atomic(EXPORT_RECORD, (json.dumps(record, indent=2) + "\n").encode())
    print(f"exported {run} to {ASSETS / MODEL_FILE} ({digest[:12]})")


if __name__ == "__main__":
    main()
