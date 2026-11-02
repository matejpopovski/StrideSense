"""Score the pose-independent model against a rule fixed before any test score existed.

The rotations are simulated: held-out windows are turned in software and scored again. Nothing here
measures a phone in a pocket. The channels cancel a fixed rotation by construction, so a rotated
score matching the waist score shows the arithmetic is right. It is not evidence about pockets.

Only variants without the gravity level can decide the rule. The leak that rules the others out was
found on training subjects, before any test score from this experiment existed, and it is measured
here again. The ineligible model is scored afterwards purely to show what the leak is worth.

python -m ml.gravity_experiment
"""

import json
from pathlib import Path

import numpy as np

from ml import data
from ml.export_litert import load_model
from ml.gravity import FEATURES
from ml.metrics import summarize
from ml.orientation import NAMED_PLACEMENTS
from ml.rotation import axis_rotation, random_rotations, rotate_windows
from ml.runtime import LiteRtModel

RESULTS = Path("results")
RUNS = Path("runs")
FINAL = "gravity_final"
LEAKY = "gravity_leaky_comparison"
ELIGIBLE = ("gravity_tune_dynamic3", "gravity_tune_dynamic3_log")
INELIGIBLE = (
    "gravity_tune_aligned4",
    "gravity_tune_aligned3",
    "gravity_tune_log",
    "gravity_tune_log_dropout3",
)

RULE = {
    "ship_if": "worst simulated rotation accuracy >= 0.75 and waist accuracy >= 0.80",
    "otherwise": "keep the waist model and record this as a negative result",
    "eligible": "variants whose channels hold no gravity level, so calibration cannot carry pose",
    "selection": (
        "eligible variant by highest mean validation macro F1 over seeds 0 and 1, test scoring "
        "off; then seed by highest validation macro F1 over seeds 0 to 4"
    ),
    "scored_on": "exported LiteRT file of that seed, on the 9 held-out subjects",
    "worst_rotation_means": "lowest of the named poses and the per-window uniform random case",
}
MIN_ROTATED = 0.75
MIN_WAIST = 0.80


def tuning_table() -> dict:
    table = {}
    for names, eligible in ((ELIGIBLE, True), (INELIGIBLE, False)):
        for name in names:
            paths = sorted((RUNS / name).glob("seed*/result.json"))
            runs = [json.loads(path.read_text()) for path in paths]
            if any("test" in run for run in runs):
                raise SystemExit(f"{name} was scored on test, which tuning must not do")
            scores = [run["val_macro_f1"] for run in runs]
            table[name] = {
                "eligible": eligible,
                "val_macro_f1": scores,
                "mean": float(np.mean(scores)),
            }
    return table


def gravity_level_leak() -> dict:
    """On training subjects: measured gravity changes with posture, which real gravity cannot."""
    train = data.load_part("train", data.DEVICE_CHANNELS)
    level = np.linalg.norm(train.x[..., :3].astype(np.float64).mean(axis=1), axis=-1)
    laying = level[train.y == data.ACTIVITIES.index("LAYING")]
    upright = level[np.isin(train.y, [data.ACTIVITIES.index(a) for a in ("SITTING", "STANDING")])]
    threshold = (laying.mean() + upright.mean()) / 2
    correct = (laying < threshold).sum() + (upright >= threshold).sum()
    return {
        "note": "Training subjects only. Differences are sensor calibration, not physics.",
        "mean_measured_gravity_g": {
            activity: float(level[train.y == index].mean())
            for index, activity in enumerate(data.ACTIVITIES)
        },
        "laying_vs_upright_accuracy_from_this_number_alone": float(
            correct / (len(laying) + len(upright))
        ),
    }


def best_run(experiment: str) -> tuple[Path, list[dict]]:
    paths = sorted((RUNS / experiment).glob("seed*/result.json"))
    runs = [json.loads(path.read_text()) for path in paths]
    best = max(range(len(runs)), key=lambda index: runs[index]["val_macro_f1"])
    return paths[best].parent, runs


def export(run: Path, path: Path) -> dict:
    import litert_torch
    import torch

    from ml.model import Probabilities

    model, checkpoint = load_model(run)
    example = torch.zeros(1, 128, checkpoint["n_channels"])
    litert_torch.convert(Probabilities(model).eval(), (example,)).export(str(path))
    return checkpoint


def still_merged_accuracy(confusion: list[list[int]]) -> float:
    """Accuracy if sitting, standing and laying count as one class. Decided after seeing test."""
    matrix = np.array(confusion)
    still = [data.ACTIVITIES.index(a) for a in ("SITTING", "STANDING", "LAYING")]
    moving = [index for index in range(len(data.ACTIVITIES)) if index not in still]
    correct = matrix[moving, moving].sum() + matrix[np.ix_(still, still)].sum()
    return float(correct / matrix.sum())


def score(experiment: str, test: data.Split, leaky: bool = False) -> dict:
    run, finals = best_run(experiment)
    path = RUNS / experiment / "model.tflite"
    checkpoint = export(run, path)
    convert = FEATURES[checkpoint["config"]["features"]]
    model = LiteRtModel(path)

    def labels(windows: np.ndarray) -> np.ndarray:
        scaled = (convert(windows) - checkpoint["mean"]) / checkpoint["std"]
        return model.predict(scaled.astype(np.float32)).argmax(axis=1)

    def accuracy(windows: np.ndarray) -> float:
        return float((labels(windows) == test.y).mean())

    waist = summarize(test.y, labels(test.x), test.subject)
    poses = {
        pose: accuracy(rotate_windows(test.x, axis_rotation(index, angle)))
        for pose, (index, angle) in NAMED_PLACEMENTS.items()
    }
    poses["uniformly random per window"] = accuracy(
        rotate_windows(test.x, random_rotations(len(test), np.random.default_rng(20)))
    )
    rotated = {pose: value for pose, value in poses.items() if not pose.startswith("as trained")}
    accuracies = [entry["test"]["accuracy"] for entry in finals]

    def level_held_constant() -> float:
        """Replace each window's gravity level with the training mean, as a recalibrated phone."""
        channels = convert(test.x).astype(np.float64)
        channels[..., 0] += checkpoint["mean"][0] - channels[..., 0].mean(axis=1, keepdims=True)
        scaled = (channels - checkpoint["mean"]) / checkpoint["std"]
        return float((model.predict(scaled.astype(np.float32)).argmax(axis=1) == test.y).mean())

    extra = {"accuracy_with_gravity_level_held_constant": level_held_constant()} if leaky else {}
    return {
        **extra,
        "accuracy_with_still_classes_merged_post_hoc": still_merged_accuracy(waist["confusion"]),
        "variant": checkpoint["config"]["features"],
        "selected_seed": checkpoint["seed"],
        "final_seeds": {
            "seeds": [entry["seed"] for entry in finals],
            "val_macro_f1": [entry["val_macro_f1"] for entry in finals],
            "test_accuracy": accuracies,
            "test_accuracy_mean": float(np.mean(accuracies)),
            "test_accuracy_std": float(np.std(accuracies, ddof=1)),
        },
        "model_file_bytes": path.stat().st_size,
        "waist": waist,
        "poses": poses,
        "worst_rotation_accuracy": min(rotated.values()),
    }


def main() -> None:
    table = tuning_table()
    test = data.load_part("test", data.DEVICE_CHANNELS)
    eligible = score(FINAL, test)
    waist = eligible["waist"]
    worst = eligible["worst_rotation_accuracy"]
    ship = worst >= MIN_ROTATED and waist["accuracy"] >= MIN_WAIST

    shipped = json.loads((RESULTS / "test_metrics.json").read_text())["litert"]
    report = {
        "note": (
            "Rotations are simulated by turning held-out windows in software. This is not a "
            "measurement of a phone in a pocket. The channels cancel a fixed rotation by "
            "construction, so matching scores under rotation are expected and prove only that."
        ),
        "rule": RULE,
        "decision": "ship" if ship else "keep the waist model",
        "gravity_level_leak": gravity_level_leak(),
        "waist_model_with_still_classes_merged_post_hoc": still_merged_accuracy(
            shipped["confusion"]
        ),
        "tuning": table,
        "eligible_model": eligible,
        "per_class_against_waist_model": {
            activity: {
                "waist_model_recall": shipped["per_class"][activity]["recall"],
                "gravity_recall": waist["per_class"][activity]["recall"],
                "waist_model_f1": shipped["per_class"][activity]["f1"],
                "gravity_f1": waist["per_class"][activity]["f1"],
            }
            for activity in data.ACTIVITIES
        },
    }
    if (RUNS / LEAKY).exists():
        report["ineligible_comparison"] = {
            "note": (
                "Keeps the gravity level, so it cannot decide the rule. Its rotated scores are "
                "unreliable: software rotation preserves the level, a real phone would not."
            ),
            **score(LEAKY, test, leaky=True),
        }
    (RESULTS / "gravity_aligned.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["decision"], f"waist={waist['accuracy']:.4f} worst_rotation={worst:.4f}")


if __name__ == "__main__":
    main()
