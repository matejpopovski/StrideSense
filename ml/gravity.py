"""Channels that do not depend on how the phone is held, built from gravity in each window.

Gravity is the mean acceleration over the window, which is a heavy low-pass that a phone can compute
from the same 128 samples. Every channel is a dot product or a length measured against it, so a
fixed rotation of the phone cancels exactly. The cost is the same fact seen from the other side: the
direction of gravity in the phone is what told still postures apart, and it is gone.

Cancelling a rotation in software is not the same as being independent of pose on a real phone.
Measured gravity depends on which axis it falls on, because each axis has its own calibration error:
in UCI HAR it reads 1.033 g standing and 1.007 g laying. A channel holding that level still carries
the pose, and rotating the numbers cannot reveal it. The aligned variants keep the level and are
recorded as ineligible. The dynamic variants subtract it, which is what makes them eligible.
"""

import numpy as np

LOG_FLOOR = 1e-3


def gravity_direction(accel: np.ndarray) -> np.ndarray:
    mean = accel.mean(axis=1, keepdims=True)
    return mean / np.linalg.norm(mean, axis=-1, keepdims=True)


def _along_and_across(vectors: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    along = (vectors * up).sum(axis=-1)
    across = np.linalg.norm(vectors - along[..., np.newaxis] * up, axis=-1)
    return along, across


def aligned4(windows: np.ndarray) -> np.ndarray:
    """Acceleration and rotation, each split into the part along gravity and the part across it."""
    data = windows.astype(np.float64)
    up = gravity_direction(data[..., :3])
    vertical, horizontal = _along_and_across(data[..., :3], up)
    yaw, tilt_rate = _along_and_across(data[..., 3:6], up)
    return np.stack([vertical, horizontal, yaw, tilt_rate], axis=-1).astype(np.float32)


def aligned3(windows: np.ndarray) -> np.ndarray:
    """As aligned4, but gyroscope magnitude only, which also needs no gravity estimate."""
    data = windows.astype(np.float64)
    up = gravity_direction(data[..., :3])
    vertical, horizontal = _along_and_across(data[..., :3], up)
    spin = np.linalg.norm(data[..., 3:6], axis=-1)
    return np.stack([vertical, horizontal, spin], axis=-1).astype(np.float32)


def aligned4_log(windows: np.ndarray) -> np.ndarray:
    """As aligned4, with magnitudes on a log scale so a still person's small sway is not lost."""
    channels = aligned4(windows).astype(np.float64)
    channels[..., 1] = np.log(channels[..., 1] + LOG_FLOOR)
    channels[..., 3] = np.log(channels[..., 3] + LOG_FLOOR)
    return channels.astype(np.float32)


def dynamic3(windows: np.ndarray) -> np.ndarray:
    """Vertical motion with the gravity level removed, horizontal motion and gyroscope magnitude."""
    channels = aligned3(windows).astype(np.float64)
    channels[..., 0] -= channels[..., 0].mean(axis=1, keepdims=True)
    return channels.astype(np.float32)


def dynamic3_log(windows: np.ndarray) -> np.ndarray:
    """As dynamic3, with magnitudes on a log scale so a still person's small sway is not lost."""
    channels = dynamic3(windows).astype(np.float64)
    channels[..., 1] = np.log(channels[..., 1] + LOG_FLOOR)
    channels[..., 2] = np.log(channels[..., 2] + LOG_FLOOR)
    return channels.astype(np.float32)


FEATURES = {
    "aligned4": aligned4,
    "aligned3": aligned3,
    "aligned4_log": aligned4_log,
    "dynamic3": dynamic3,
    "dynamic3_log": dynamic3_log,
}
