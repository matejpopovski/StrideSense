import numpy as np
import pytest

from ml.gravity import FEATURES, aligned4, dynamic3, dynamic3_log, gravity_direction
from ml.rotation import axis_rotation, random_rotations, rotate_windows


def walking_like(count=3, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(128) / 50
    accel = np.stack([np.ones(128), 0.2 * np.sin(5 * time), 0.1 * np.cos(9 * time)], axis=-1)
    gyro = 0.3 * rng.standard_normal((128, 3))
    window = np.concatenate([accel, gyro], axis=-1)
    return np.stack([window + 0.02 * rng.standard_normal((128, 6)) for _ in range(count)]).astype(
        np.float32
    )


def test_gravity_direction_is_a_unit_vector_along_the_mean():
    windows = np.array([[[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]])
    np.testing.assert_allclose(gravity_direction(windows), [[[0.0, 0.0, 1.0]]])


def test_a_still_phone_reads_one_g_up_and_nothing_across():
    windows = np.zeros((1, 128, 6), np.float32)
    windows[..., 2] = 1.0
    channels = aligned4(windows)
    np.testing.assert_allclose(channels[..., 0], 1.0)
    np.testing.assert_allclose(channels[..., 1:], 0.0, atol=1e-6)


def test_vertical_and_horizontal_recover_the_acceleration_length():
    windows = walking_like()
    channels = aligned4(windows)
    length = np.linalg.norm(windows[..., :3], axis=-1)
    np.testing.assert_allclose(np.hypot(channels[..., 0], channels[..., 1]), length, atol=1e-5)


@pytest.mark.parametrize("name", sorted(FEATURES))
@pytest.mark.parametrize(("axis", "degrees"), [(2, 90.0), (2, -90.0), (2, 180.0), (0, 37.0)])
def test_a_fixed_rotation_of_the_phone_changes_nothing(name, axis, degrees):
    windows = walking_like()
    turned = rotate_windows(windows, axis_rotation(axis, degrees))
    np.testing.assert_allclose(FEATURES[name](turned), FEATURES[name](windows), atol=2e-4)


@pytest.mark.parametrize("name", sorted(FEATURES))
def test_a_different_rotation_per_window_changes_nothing(name):
    windows = walking_like(count=8)
    turned = rotate_windows(windows, random_rotations(8, np.random.default_rng(1)))
    np.testing.assert_allclose(FEATURES[name](turned), FEATURES[name](windows), atol=2e-4)


def test_the_channels_really_do_differ_between_motion_and_stillness():
    still = np.zeros((1, 128, 6), np.float32)
    still[..., 0] = 1.0
    assert aligned4(walking_like(1))[..., 1].std() > aligned4(still)[..., 1].std()


@pytest.mark.parametrize("convert", [dynamic3, dynamic3_log])
def test_a_calibration_offset_in_gravity_does_not_reach_the_eligible_channels(convert):
    windows = walking_like().astype(np.float64)
    up = gravity_direction(windows[..., :3])
    biased = windows.copy()
    biased[..., :3] += 0.03 * up
    np.testing.assert_allclose(
        convert(biased.astype(np.float32)), convert(windows.astype(np.float32)), atol=2e-4
    )


def test_the_raw_vertical_channel_does_carry_that_offset():
    windows = walking_like().astype(np.float64)
    up = gravity_direction(windows[..., :3])
    biased = windows.copy()
    biased[..., :3] += 0.03 * up
    difference = aligned4(biased.astype(np.float32)) - aligned4(windows.astype(np.float32))
    np.testing.assert_allclose(difference[..., 0], 0.03, atol=1e-4)


def test_dynamic_vertical_averages_to_zero_in_every_window():
    channels = dynamic3(walking_like().astype(np.float32))
    np.testing.assert_allclose(channels[..., 0].mean(axis=1), 0.0, atol=1e-5)
