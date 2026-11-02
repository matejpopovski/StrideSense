"""UCI HAR loading and the subject-wise splits every experiment shares."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASET_DIR = Path("data") / "UCI HAR Dataset"
WINDOW_LENGTH = 128

ACTIVITIES = ("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING")

TOTAL_ACC = ("total_acc_x", "total_acc_y", "total_acc_z")
BODY_GYRO = ("body_gyro_x", "body_gyro_y", "body_gyro_z")
BODY_ACC = ("body_acc_x", "body_acc_y", "body_acc_z")

# The phone can reproduce these six live. body_acc comes from a 0.3 Hz filter the dataset authors
# ran before windowing, without saying whether it was causal, so a live version could not be shown
# to match it. It is only used in the ablation that measures what it is worth.
DEVICE_CHANNELS = TOTAL_ACC + BODY_GYRO
ALL_CHANNELS = TOTAL_ACC + BODY_GYRO + BODY_ACC

# Taken from the official training subjects, so the official test subjects stay untouched by
# early stopping and model selection. From default_rng(20).choice(train_subjects, 5, False).
VALIDATION_SUBJECTS = (7, 8, 16, 25, 28)


@dataclass(frozen=True)
class Split:
    x: np.ndarray  # (windows, 128, channels), acceleration in g, angular rate in rad/s
    y: np.ndarray  # class index into ACTIVITIES
    subject: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def where(self, mask: np.ndarray) -> "Split":
        return Split(self.x[mask], self.y[mask], self.subject[mask])


def load_part(part: str, channels: tuple[str, ...], root: Path = DATASET_DIR) -> Split:
    signals = root / part / "Inertial Signals"
    stacked = [
        np.loadtxt(signals / f"{name}_{part}.txt", dtype=np.float64).astype(np.float32)
        for name in channels
    ]
    x = np.stack(stacked, axis=-1)
    y = np.loadtxt(root / part / f"y_{part}.txt", dtype=np.int64) - 1
    subject = np.loadtxt(root / part / f"subject_{part}.txt", dtype=np.int64)
    if x.shape[1] != WINDOW_LENGTH or not len(x) == len(y) == len(subject):
        raise ValueError(f"unexpected shape in {part}: {x.shape}, {y.shape}, {subject.shape}")
    return Split(x, y, subject)


def subject_split(train_part: Split) -> tuple[Split, Split]:
    is_validation = np.isin(train_part.subject, VALIDATION_SUBJECTS)
    return train_part.where(~is_validation), train_part.where(is_validation)


def random_window_split(
    train_part: Split, test_part: Split, seed: int
) -> tuple[Split, Split, Split]:
    """The leaky split: windows of one person, overlapping by 50%, land on both sides."""
    x = np.concatenate([train_part.x, test_part.x])
    y = np.concatenate([train_part.y, test_part.y])
    subject = np.concatenate([train_part.subject, test_part.subject])
    order = np.random.default_rng(seed).permutation(len(y))
    x, y, subject = x[order], y[order], subject[order]
    n_test, n_val = len(test_part), round(0.2 * len(train_part))
    return (
        Split(x[n_test + n_val :], y[n_test + n_val :], subject[n_test + n_val :]),
        Split(
            x[n_test : n_test + n_val], y[n_test : n_test + n_val], subject[n_test : n_test + n_val]
        ),
        Split(x[:n_test], y[:n_test], subject[:n_test]),
    )


def channel_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x.mean(axis=(0, 1), dtype=np.float64), x.std(axis=(0, 1), dtype=np.float64)
