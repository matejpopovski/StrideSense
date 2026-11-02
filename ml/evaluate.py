"""Score the exported model on the held-out subjects, then aggregate every experiment.

Run after export. The headline numbers come from the LiteRT file the app ships, not from PyTorch.

python -m ml.evaluate
"""

import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ml import data
from ml.export_litert import ASSETS, MODEL_FILE, RUNS, SPEC_FILE, exported_run, load_model
from ml.metrics import summarize
from ml.runtime import LiteRtModel

RESULTS = Path("results")
LATENCY_REPEATS = 2000


def hardware() -> str:
    chip = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return f"{chip or platform.machine()}, {platform.system()}, single LiteRT thread, Python host"


def write_json(name: str, payload: dict) -> None:
    (RESULTS / name).write_text(json.dumps(payload, indent=2) + "\n")


def seed_table(experiment: str) -> dict:
    runs = [
        json.loads(path.read_text())
        for path in sorted((RUNS / experiment).glob("seed*/result.json"))
    ]
    entry = {
        "seeds": [run["seed"] for run in runs],
        "val_macro_f1": [run["val_macro_f1"] for run in runs],
        "best_epoch": [run["best_epoch"] for run in runs],
        "train_seconds": [run["train_seconds"] for run in runs],
    }
    scored = [run["test"] for run in runs if "test" in run]
    if scored:
        accuracy = np.array([test["accuracy"] for test in scored])
        macro_f1 = np.array([test["macro_f1"] for test in scored])
        entry.update(
            test_accuracy=accuracy.tolist(),
            test_macro_f1=macro_f1.tolist(),
            test_accuracy_mean=float(accuracy.mean()),
            test_accuracy_std=float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0,
            test_macro_f1_mean=float(macro_f1.mean()),
        )
    return entry


def main() -> None:
    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)

    model, checkpoint = load_model(exported_run())
    test = data.load_part("test", data.DEVICE_CHANNELS)
    x = ((test.x - checkpoint["mean"]) / checkpoint["std"]).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.from_numpy(x))
    torch_probabilities = torch.softmax(logits, dim=-1).numpy()

    path = ASSETS / MODEL_FILE
    runtime = LiteRtModel(path)
    litert_probabilities = runtime.predict(x)

    torch_labels = torch_probabilities.argmax(axis=1)
    litert_labels = litert_probabilities.argmax(axis=1)
    conversion = {
        "windows": len(x),
        "label_agreement": float((torch_labels == litert_labels).mean()),
        "max_absolute_probability_difference": float(
            np.abs(torch_probabilities - litert_probabilities).max()
        ),
    }

    timings = []
    for index in range(LATENCY_REPEATS):
        started = time.perf_counter()
        runtime(x[index % len(x)])
        timings.append(time.perf_counter() - started)
    ops = runtime.builtin_ops()

    metrics = {
        "shipped_seed": checkpoint["seed"],
        "split": "UCI HAR official split: 21 training subjects, 9 test subjects, no overlap",
        "selection": "seed and epoch chosen on 5 validation subjects from the training 21",
        "test_subjects": sorted(np.unique(test.subject).tolist()),
        "litert": summarize(test.y, litert_labels, test.subject),
        "pytorch": summarize(test.y, torch_labels, test.subject),
        "conversion": conversion,
        "model": {
            "file_bytes": path.stat().st_size,
            "parameters": sum(p.numel() for p in model.parameters()),
            "graph_ops": sum(ops.values()),
            "input_shape": [1, 128, 6],
            "sha256": json.loads((ASSETS / SPEC_FILE).read_text())["model_sha256"],
        },
        "latency_ms": {
            "note": "Measured on the development Mac. On-device latency is unmeasured.",
            "hardware": hardware(),
            "repeats": LATENCY_REPEATS,
            "p50": float(np.percentile(timings, 50) * 1000),
            "p95": float(np.percentile(timings, 95) * 1000),
        },
    }
    if conversion["label_agreement"] < 0.999:
        raise SystemExit(f"LiteRT disagrees with the model it came from: {conversion}")
    write_json("test_metrics.json", metrics)

    with (RESULTS / "per_subject.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["subject", "windows", "litert_accuracy", "litert_macro_f1", "pytorch_accuracy"]
        )
        for subject, entry in metrics["litert"]["per_subject"].items():
            writer.writerow(
                [
                    subject,
                    entry["windows"],
                    f"{entry['accuracy']:.4f}",
                    f"{entry['macro_f1']:.4f}",
                    f"{metrics['pytorch']['per_subject'][subject]['accuracy']:.4f}",
                ]
            )

    experiments = {
        path.name: seed_table(path.name)
        for path in sorted(RUNS.iterdir())
        if path.is_dir() and any(path.glob("seed*/result.json"))
    }
    write_json("experiments.json", experiments)
    litert, overall = metrics["litert"], metrics["conversion"]
    print(
        f"litert accuracy={litert['accuracy']:.4f} macro_f1={litert['macro_f1']:.4f} "
        f"subjects {litert['subject_accuracy_min']:.3f}-{litert['subject_accuracy_max']:.3f} "
        f"agreement={overall['label_agreement']:.4f} size={path.stat().st_size}"
    )


if __name__ == "__main__":
    main()
