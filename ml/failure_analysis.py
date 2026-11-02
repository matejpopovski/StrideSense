"""Why some people score far below others: sitting and standing differ only by a slight tilt.

Standing and sitting are both still, so all the model has is the direction of gravity in the phone.
This measures how far apart the two postures are per person and sets that against the errors.

python -m ml.failure_analysis
"""

import json
from pathlib import Path

import numpy as np

from ml import data
from ml.export_litert import ASSETS, MODEL_FILE, SPEC_FILE, exported_run
from ml.runtime import LiteRtModel

RESULTS = Path("results")
SITTING = data.ACTIVITIES.index("SITTING")
STANDING = data.ACTIVITIES.index("STANDING")


def mean_gravity(x: np.ndarray) -> np.ndarray:
    return x[..., :3].mean(axis=(0, 1), dtype=np.float64)


def separation_degrees(sitting: np.ndarray, standing: np.ndarray) -> float:
    cosine = sitting @ standing / (np.linalg.norm(sitting) * np.linalg.norm(standing))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def posture(split: data.Split, mask: np.ndarray) -> dict:
    sitting = mean_gravity(split.x[mask & (split.y == SITTING)])
    standing = mean_gravity(split.x[mask & (split.y == STANDING)])
    return {
        "sitting_gravity_g": sitting.tolist(),
        "standing_gravity_g": standing.tolist(),
        "tilt_between_postures_degrees": separation_degrees(sitting, standing),
    }


def main() -> None:
    exported_run()
    spec = json.loads((ASSETS / SPEC_FILE).read_text())
    mean, std = np.array(spec["mean"]), np.array(spec["std"])
    test = data.load_part("test", data.DEVICE_CHANNELS)
    train = data.load_part("train", data.DEVICE_CHANNELS)
    predicted = (
        LiteRtModel(ASSETS / MODEL_FILE)
        .predict(((test.x - mean) / std).astype(np.float32))
        .argmax(axis=1)
    )

    subjects = {}
    for subject in np.unique(test.subject).tolist():
        mask = test.subject == subject
        still = mask & np.isin(test.y, (SITTING, STANDING))
        confused = (test.y == SITTING) & (predicted == STANDING)
        confused |= (test.y == STANDING) & (predicted == SITTING)
        subjects[str(subject)] = {
            "accuracy": float((predicted[mask] == test.y[mask]).mean()),
            "sit_stand_confusions": int((confused & mask).sum()),
            "sit_stand_windows": int(still.sum()),
            **posture(test, mask),
        }

    errors = (predicted != test.y).sum()
    confused = ((test.y == SITTING) & (predicted == STANDING)) | (
        (test.y == STANDING) & (predicted == SITTING)
    )
    tilts = [entry["tilt_between_postures_degrees"] for entry in subjects.values()]
    rates = [
        entry["sit_stand_confusions"] / entry["sit_stand_windows"] for entry in subjects.values()
    ]
    report = {
        "all_errors": int(errors),
        "sit_stand_errors": int(confused.sum()),
        "sit_stand_share_of_errors": float(confused.sum() / errors),
        "correlation_tilt_vs_confusion_rate": float(np.corrcoef(tilts, rates)[0, 1]),
        "training_subjects": posture(train, np.ones(len(train), dtype=bool)),
        "test_subjects": subjects,
    }
    (RESULTS / "failure_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
