import numpy as np
import pytest

from ml import preprocess
from ml.preprocess import SensorStream


def steady(values, period_ns=preprocess.SAMPLE_PERIOD_NS, start_ns=0) -> SensorStream:
    values = np.asarray(values, dtype=np.float64)
    return SensorStream(start_ns + np.arange(len(values)) * period_ns, values)


def test_acceleration_converts_from_si_to_g():
    assert preprocess.to_standard_units(np.array([9.80665])) == pytest.approx(1.0)


def test_events_on_the_grid_pass_through_resampling():
    accel = steady(np.arange(30).reshape(10, 3))
    gyro = steady(np.arange(30).reshape(10, 3) + 100)
    runs = preprocess.resample(accel, gyro)
    assert len(runs) == 1
    np.testing.assert_allclose(runs[0][:, 0], accel.values[:, 0])
    np.testing.assert_allclose(runs[0][:, 3], gyro.values[:, 0])


def test_grid_starts_at_the_later_sensor():
    accel = steady(np.zeros((10, 3)))
    gyro = steady(np.zeros((10, 3)), start_ns=30_000_000)
    assert preprocess.resample(accel, gyro)[0].shape[0] == 8


def test_between_events_values_are_interpolated():
    accel = SensorStream(np.array([0, 40_000_000]), np.array([[0.0, 0, 0], [4.0, 0, 0]]))
    gyro = SensorStream(np.array([0, 40_000_000]), np.zeros((2, 3)))
    assert preprocess.resample(accel, gyro)[0][:, 0].tolist() == [0.0, 2.0, 4.0]


def test_a_gap_splits_the_recording_into_runs():
    times = np.concatenate([np.arange(20), np.arange(40, 60)]) * preprocess.SAMPLE_PERIOD_NS
    accel = SensorStream(times, np.zeros((40, 3)))
    gyro = steady(np.zeros((60, 3)))
    assert [len(run) for run in preprocess.resample(accel, gyro)] == [20, 20]


def test_events_that_repeat_a_timestamp_are_dropped():
    stream = SensorStream(np.array([0, 10, 10, 5, 20]), np.arange(15.0).reshape(5, 3))
    assert preprocess.strictly_increasing(stream).timestamps_ns.tolist() == [0, 10, 20]


def test_median_removes_a_lone_spike():
    run = np.ones((6, 6))
    run[3, 0] = 50.0
    assert preprocess.median3(run)[:, 0].tolist() == [1.0] * 6


def test_median_leaves_the_first_two_samples_alone():
    run = np.array([[1.0], [9.0], [2.0]])
    assert preprocess.median3(run).ravel().tolist() == [1.0, 9.0, 2.0]


def test_lowpass_has_no_startup_transient_on_gravity():
    run = np.full((200, 6), 0.98)
    np.testing.assert_allclose(preprocess.lowpass(run), run, atol=1e-12)


def test_lowpass_attenuates_above_the_cutoff_and_passes_below():
    time = np.arange(1000) / preprocess.SAMPLE_RATE_HZ

    def gain(frequency):
        tone = np.sin(2 * np.pi * frequency * time)
        run = np.tile(tone[:, np.newaxis], (1, 6))
        return preprocess.lowpass(run)[200:, 0].std() / tone[200:].std()

    assert gain(2.0) == pytest.approx(1.0, abs=0.01)
    assert gain(24.0) < 0.1


def test_window_count_and_overlap():
    run = np.zeros((100, 6))
    short = preprocess.sliding_windows(run)
    assert short.shape == (0, 128, 6)
    run = np.arange(320 * 6.0).reshape(320, 6)
    windows = preprocess.sliding_windows(run)
    assert windows.shape == (4, 128, 6)
    np.testing.assert_array_equal(windows[0, 64:], windows[1, :64])


def test_standardize_is_float32_and_channelwise():
    windows = np.array([[[3.0, 8.0]]])
    out = preprocess.standardize(windows, np.array([1.0, 2.0]), np.array([2.0, 3.0]))
    assert out.dtype == np.float32
    assert out.ravel().tolist() == [1.0, 2.0]


def test_too_little_data_gives_no_windows_rather_than_failing():
    accel, gyro = steady(np.zeros((5, 3))), steady(np.zeros((5, 3)))
    out = preprocess.model_input(accel, gyro, np.zeros(6), np.ones(6))
    assert out.shape == (0, 128, 6)
