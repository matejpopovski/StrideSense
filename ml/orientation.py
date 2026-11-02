"""How much accuracy depends on the phone sitting the way it sat at the waist.

Turning the sensor frame is only a proxy for a pocket: it changes orientation, not the way a thigh
moves compared with a hip. It bounds one failure, and says nothing about real pocket accuracy.

python -m ml.orientation
"""

import json
from pathlib import Path

import numpy as np
import torch

from ml import data
from ml.export_litert import RUNS, best_run, exported_run, load_model
from ml.rotation import axis_rotation, random_rotations, rotate_windows

RESULTS = Path("results")
ANGLES = (0, 15, 30, 45, 60, 90, 120, 150, 180)
AXES = {"x": 0, "y": 1, "z": 2}
# In UCI HAR gravity loads +x when the wearer is upright: the phones sat sideways at the waist. An
# upright phone reads gravity on y instead, which is a quarter turn about z from the training pose.
NAMED_PLACEMENTS = {
    "as trained, sideways at the waist": (2, 0.0),
    "upright in portrait": (2, 90.0),
    "upside down in portrait": (2, -90.0),
    "sideways, flipped end for end": (2, 180.0),
}


def accuracy(model, checkpoint, windows: np.ndarray, labels: np.ndarray) -> float:
    x = ((windows - checkpoint["mean"]) / checkpoint["std"]).astype(np.float32)
    with torch.no_grad():
        predictions = torch.cat([model(torch.from_numpy(b)) for b in np.array_split(x, 12)])
    return float((predictions.argmax(dim=-1).numpy() == labels).mean())


def gravity_axis(test: data.Split) -> list[float]:
    standing = test.x[test.y == data.ACTIVITIES.index("STANDING")]
    return standing[..., :3].mean(axis=(0, 1)).astype(float).tolist()


def sweep(run: Path, test: data.Split) -> dict:
    model, checkpoint = load_model(run)
    return {
        "gravity_axis_when_standing": gravity_axis(test),
        "named_placements": {
            name: accuracy(
                model, checkpoint, rotate_windows(test.x, axis_rotation(index, angle)), test.y
            )
            for name, (index, angle) in NAMED_PLACEMENTS.items()
        },
        "per_axis": {
            axis: {
                str(angle): accuracy(
                    model, checkpoint, rotate_windows(test.x, axis_rotation(index, angle)), test.y
                )
                for angle in ANGLES
            }
            for axis, index in AXES.items()
        },
        "uniform_random": accuracy(
            model,
            checkpoint,
            rotate_windows(test.x, random_rotations(len(test), np.random.default_rng(20))),
            test.y,
        ),
    }


def main() -> None:
    torch.set_num_threads(4)
    test = data.load_part("test", data.DEVICE_CHANNELS)
    report = {
        "note": (
            "Rotated copies of the held-out windows, scored with PyTorch. A proxy for orientation "
            "only, not a measurement of a phone in a pocket."
        ),
        "models": {"shipped": {"run": str(exported_run()), **sweep(exported_run(), test)}},
    }
    if (RUNS / "rotation_augmented").exists():
        run = best_run("rotation_augmented")
        report["models"]["any_orientation"] = {"run": str(run), **sweep(run, test)}
    (RESULTS / "orientation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
