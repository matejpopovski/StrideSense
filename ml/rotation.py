"""Rigid rotations of the sensor frame, for augmentation and for sensitivity sweeps."""

import numpy as np


def axis_rotation(axis: int, degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    i, j = (axis + 1) % 3, (axis + 2) % 3
    matrix = np.eye(3)
    matrix[i, i], matrix[i, j], matrix[j, i], matrix[j, j] = c, -s, s, c
    return matrix


def _quaternion_matrices(w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


def random_rotations(count: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrices, from normalized quaternions."""
    q = rng.standard_normal((4, count))
    return _quaternion_matrices(*(q / np.linalg.norm(q, axis=0)))


def small_rotations(count: int, max_degrees: float, rng: np.random.Generator) -> np.ndarray:
    """Random axis, angle uniform in +/-max_degrees: the spread of a phone on different belts."""
    axes = rng.standard_normal((count, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    half = np.radians(rng.uniform(-max_degrees, max_degrees, count)) / 2
    w, x, y, z = np.cos(half), *(axes.T * np.sin(half))
    return _quaternion_matrices(w, x, y, z)


def rotate_windows(windows: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate (n, t, channels) windows, whose channels are consecutive xyz triples.

    A rigid frame change turns every triaxial signal alike. `rotation` is one matrix or one per
    window.
    """
    matrices = rotation if rotation.ndim == 3 else rotation[np.newaxis]
    triples = [
        np.einsum("nij,ntj->nti", matrices, windows[..., first : first + 3])
        for first in range(0, windows.shape[-1], 3)
    ]
    return np.concatenate(triples, axis=-1).astype(windows.dtype)
