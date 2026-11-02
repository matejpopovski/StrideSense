import numpy as np
import pytest

from ml import data
from ml.streams import contiguous_segments, simulate_sensors


def recording(samples=320, windows=3, subject=2, label=0, offset=0.0):
    signal = np.arange(samples * 6, dtype=np.float32).reshape(samples, 6) + offset
    x = np.stack([signal[i * 64 : i * 64 + 128] for i in range(windows)])
    return x, np.full(windows, label), np.full(windows, subject)


def test_overlapping_windows_rebuild_one_recording():
    x, y, subject = recording()
    segments = contiguous_segments(data.Split(x, y, subject))
    assert len(segments) == 1
    assert segments[0].window_count == 3
    assert segments[0].signal.shape == (256, 6)


def test_a_change_of_activity_or_person_starts_a_new_recording():
    first = recording()
    second = recording(label=1, offset=9999.0)
    third = recording(subject=4, label=1, offset=19999.0)
    parts = [first, second, third]
    split = data.Split(*(np.concatenate(items) for items in zip(*parts, strict=True)))
    assert [s.label for s in contiguous_segments(split)] == [0, 1, 1]


def test_windows_that_do_not_overlap_are_not_joined():
    x, y, subject = recording(windows=2)
    x[1] += 500.0
    assert len(contiguous_segments(data.Split(x, y, subject))) == 2


def test_simulated_sensors_report_si_units_on_increasing_clocks():
    signal = np.zeros((300, 6))
    signal[:, 0] = 1.0
    accel, gyro = simulate_sensors(signal, 50.0, 0.002, 0.007, np.random.default_rng(0))
    assert np.all(np.diff(accel.timestamps_ns) >= 0)
    assert accel.values[:, 0] == pytest.approx(9.80665)
    assert gyro.timestamps_ns[0] > accel.timestamps_ns[0]
