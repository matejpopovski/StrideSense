"""Rebuild continuous recordings from UCI HAR's overlapping windows, and fake a phone's sensors."""

from dataclasses import dataclass

import numpy as np

from ml import preprocess
from ml.data import Split
from ml.preprocess import SensorStream

OVERLAP = 64


@dataclass(frozen=True)
class Segment:
    subject: int
    label: int
    first_window: int
    window_count: int
    signal: np.ndarray  # (samples, 6) at 50 Hz, acceleration in g, angular rate in rad/s


def contiguous_segments(split: Split) -> list[Segment]:
    """Runs of windows that really do share their overlap, which is how one recording looks."""
    joined = np.isclose(split.x[:-1, OVERLAP:], split.x[1:, :OVERLAP], atol=1e-6).all(axis=(1, 2))
    same = (split.y[:-1] == split.y[1:]) & (split.subject[:-1] == split.subject[1:])
    breaks = np.flatnonzero(~(joined & same)) + 1
    segments = []
    for start, stop in zip([0, *breaks.tolist()], [*breaks.tolist(), len(split)], strict=True):
        windows = split.x[start:stop]
        signal = np.concatenate([windows[0], *windows[1:, OVERLAP:]], axis=0, dtype=np.float64)
        segments.append(
            Segment(int(split.subject[start]), int(split.y[start]), start, stop - start, signal)
        )
    return segments


def sample_at(signal: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    grid = np.arange(len(signal)) / preprocess.SAMPLE_RATE_HZ
    return np.stack(
        [np.interp(times_s, grid, signal[:, c]) for c in range(signal.shape[1])], axis=1
    )


def _event_times(duration_s: float, rate_hz: float, jitter_s: float, offset_s: float, rng):
    count = int((duration_s - offset_s) * rate_hz)
    times = offset_s + np.arange(count) / rate_hz + rng.normal(0, jitter_s, count)
    return np.sort(np.clip(times, 0, duration_s))


def simulate_sensors(
    signal: np.ndarray,
    rate_hz: float,
    jitter_s: float,
    gyro_offset_s: float,
    rng: np.random.Generator,
    start_ns: int = 1_000_000_000,
) -> tuple[SensorStream, SensorStream]:
    """What the two sensors would report: independent clocks, jitter, SI units for acceleration."""
    duration = (len(signal) - 1) / preprocess.SAMPLE_RATE_HZ
    accel_times = _event_times(duration, rate_hz, jitter_s, 0.0, rng)
    gyro_times = _event_times(duration, rate_hz, jitter_s, gyro_offset_s, rng)

    def to_ns(times_s: np.ndarray) -> np.ndarray:
        return start_ns + np.round(times_s * 1e9).astype(np.int64)

    return (
        SensorStream(
            to_ns(accel_times),
            sample_at(signal[:, :3], accel_times) * preprocess.STANDARD_GRAVITY,
        ),
        SensorStream(to_ns(gyro_times), sample_at(signal[:, 3:6], gyro_times)),
    )
