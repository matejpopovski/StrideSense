"""The reference for what the phone does to raw sensor events before inference.

core/ ports this to Kotlin and the parity fixtures hold the two together, so a change here has to
be made there too. Everything runs in float64: the model input is cast to float32 once, at the end.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal

SAMPLE_RATE_HZ = 50
SAMPLE_PERIOD_NS = 20_000_000
STANDARD_GRAVITY = 9.80665
WINDOW_LENGTH = 128
WINDOW_STRIDE = 64
FILTER_CUTOFF_HZ = 20.0
FILTER_ORDER = 3
MEDIAN_LENGTH = 3
# A sensor that goes silent for longer than this has stopped, and a straight line across the gap
# would invent motion. Ten samples is the longest gap treated as ordinary jitter.
MAX_GAP_NS = 10 * SAMPLE_PERIOD_NS


def butterworth() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coefficients plus the state that starts the filter settled on a unit input."""
    b, a = signal.butter(FILTER_ORDER, FILTER_CUTOFF_HZ / (SAMPLE_RATE_HZ / 2))
    return b, a, signal.lfilter_zi(b, a)


@dataclass(frozen=True)
class SensorStream:
    """One sensor's events: timestamps in nanoseconds and (n, 3) values in SI units."""

    timestamps_ns: np.ndarray
    values: np.ndarray


def strictly_increasing(stream: SensorStream) -> SensorStream:
    """Drop events that do not move time forward: interpolation is undefined across them."""
    keep = np.ones(len(stream.timestamps_ns), dtype=bool)
    latest = None
    for i, timestamp in enumerate(stream.timestamps_ns.tolist()):
        if latest is not None and timestamp <= latest:
            keep[i] = False
        else:
            latest = timestamp
    return SensorStream(stream.timestamps_ns[keep], stream.values[keep])


def _resample_segment(
    stream: SensorStream, first: int, last: int, grid_start: int, start_index: int, stop_index: int
) -> np.ndarray:
    grid = grid_start + np.arange(start_index, stop_index, dtype=np.int64) * SAMPLE_PERIOD_NS
    times = stream.timestamps_ns[first : last + 1]
    return np.stack(
        [
            np.interp(grid - times[0], times - times[0], stream.values[first : last + 1, axis])
            for axis in range(3)
        ],
        axis=1,
    )


def _segments(timestamps_ns: np.ndarray) -> list[tuple[int, int]]:
    breaks = np.flatnonzero(np.diff(timestamps_ns) > MAX_GAP_NS)
    starts = [0, *(breaks + 1).tolist()]
    stops = [*breaks.tolist(), len(timestamps_ns) - 1]
    return list(zip(starts, stops, strict=True))


def _grid_range(times: np.ndarray, grid_start: int) -> tuple[int, int]:
    """Half-open grid indices lying inside [times[0], times[-1]]."""
    first = -((grid_start - int(times[0])) // SAMPLE_PERIOD_NS)
    stop = (int(times[-1]) - grid_start) // SAMPLE_PERIOD_NS + 1
    return first, stop


def resample(accel: SensorStream, gyro: SensorStream) -> list[np.ndarray]:
    """Interpolate both sensors onto one 50 Hz grid.

    Returns one (n, 6) run per stretch of signal where both sensors were live. Each run is a
    contiguous grid, so filters and windows start over at a run boundary.
    """
    grid_start = max(int(accel.timestamps_ns[0]), int(gyro.timestamps_ns[0]))
    runs = []
    for a_first, a_last in _segments(accel.timestamps_ns):
        a_range = _grid_range(accel.timestamps_ns[a_first : a_last + 1], grid_start)
        for g_first, g_last in _segments(gyro.timestamps_ns):
            g_range = _grid_range(gyro.timestamps_ns[g_first : g_last + 1], grid_start)
            start, stop = max(a_range[0], g_range[0]), min(a_range[1], g_range[1])
            if stop <= start:
                continue
            runs.append(
                np.hstack(
                    [
                        _resample_segment(accel, a_first, a_last, grid_start, start, stop),
                        _resample_segment(gyro, g_first, g_last, grid_start, start, stop),
                    ]
                )
            )
    return runs


def median3(run: np.ndarray) -> np.ndarray:
    """Causal median of the current and two previous samples, passing the first two through."""
    out = run.copy()
    if len(run) >= MEDIAN_LENGTH:
        stacked = np.stack([run[:-2], run[1:-1], run[2:]])
        out[2:] = np.median(stacked, axis=0)
    return out


def lowpass(run: np.ndarray) -> np.ndarray:
    """Causal Butterworth, settled on the first sample so gravity causes no start-up transient."""
    b, a, zi = butterworth()
    out = np.empty_like(run)
    for channel in range(run.shape[1]):
        out[:, channel], _ = signal.lfilter(b, a, run[:, channel], zi=zi * run[0, channel])
    return out


def to_standard_units(accel_si: np.ndarray) -> np.ndarray:
    return accel_si / STANDARD_GRAVITY


def filtered_runs(accel: SensorStream, gyro: SensorStream) -> list[np.ndarray]:
    """Raw events to filtered 50 Hz runs: acceleration in g, angular rate in rad/s."""
    in_g = SensorStream(accel.timestamps_ns, to_standard_units(accel.values))
    runs = resample(strictly_increasing(in_g), strictly_increasing(gyro))
    return [lowpass(median3(run)) for run in runs]


def sliding_windows(run: np.ndarray) -> np.ndarray:
    count = max(0, (len(run) - WINDOW_LENGTH) // WINDOW_STRIDE + 1)
    if count == 0:
        return np.empty((0, WINDOW_LENGTH, run.shape[1]))
    return np.stack(
        [run[i * WINDOW_STRIDE : i * WINDOW_STRIDE + WINDOW_LENGTH] for i in range(count)]
    )


def standardize(windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((windows - mean) / std).astype(np.float32)


def model_input(
    accel: SensorStream, gyro: SensorStream, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """The full device path: (windows, 128, 6) float32, ready for the model."""
    per_run = [sliding_windows(run) for run in filtered_runs(accel, gyro)]
    windows = np.concatenate(per_run) if per_run else np.empty((0, WINDOW_LENGTH, 6))
    return standardize(windows, mean, std)
