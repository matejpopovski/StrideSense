import numpy as np

from ml import data
from ml.metrics import confusion, per_class, summarize
from ml.rotation import axis_rotation, random_rotations, rotate_windows, small_rotations

OFFICIAL_TEST_SUBJECTS = {2, 4, 9, 10, 12, 13, 18, 20, 24}


def fake_split(subjects):
    subject = np.array(subjects)
    n = len(subject)
    return data.Split(np.zeros((n, 128, 6), np.float32), np.zeros(n, np.int64), subject)


def test_validation_subjects_never_touch_the_test_subjects():
    assert not set(data.VALIDATION_SUBJECTS) & OFFICIAL_TEST_SUBJECTS


def test_subject_split_keeps_each_person_on_one_side():
    fit, validation = data.subject_split(fake_split([1, 7, 7, 8, 3, 16, 3, 25, 28]))
    assert set(fit.subject) == {1, 3}
    assert set(validation.subject) == {7, 8, 16, 25, 28}


def test_the_device_channels_exclude_the_offline_body_acceleration():
    assert not set(data.DEVICE_CHANNELS) & set(data.BODY_ACC)
    assert len(data.DEVICE_CHANNELS) == 6


def test_channel_stats_are_per_channel():
    x = np.array([[[1.0, 10.0], [3.0, 30.0]]])
    mean, std = data.channel_stats(x)
    assert mean.tolist() == [2.0, 20.0]
    assert std.tolist() == [1.0, 10.0]


def test_confusion_counts_true_rows_and_predicted_columns():
    matrix = confusion(np.array([0, 0, 1]), np.array([0, 1, 1]))
    assert matrix[0].tolist()[:2] == [1, 1]
    assert matrix[1].tolist()[:2] == [0, 1]


def test_per_class_handles_a_class_with_no_predictions():
    report = per_class(confusion(np.array([0, 0]), np.array([1, 1])))
    assert report["WALKING"]["recall"] == 0.0
    assert report["WALKING_UPSTAIRS"]["precision"] == 0.0


def test_summary_reports_each_subject():
    y = np.array([0, 0, 1, 1])
    summary = summarize(y, np.array([0, 0, 1, 0]), np.array([5, 5, 6, 6]))
    assert summary["per_subject"]["5"]["accuracy"] == 1.0
    assert summary["per_subject"]["6"]["accuracy"] == 0.5
    assert summary["accuracy"] == 0.75


def test_rotations_are_proper_rotations():
    rng = np.random.default_rng(0)
    for matrices in (random_rotations(20, rng), small_rotations(20, 30.0, rng)):
        identity = np.einsum("nij,nkj->nik", matrices, matrices)
        np.testing.assert_allclose(identity, np.tile(np.eye(3), (20, 1, 1)), atol=1e-9)
        np.testing.assert_allclose(np.linalg.det(matrices), 1.0, atol=1e-9)


def test_small_rotations_stay_within_the_limit():
    rng = np.random.default_rng(0)
    matrices = small_rotations(500, 15.0, rng)
    angle = np.degrees(np.arccos((np.trace(matrices, axis1=1, axis2=2) - 1) / 2))
    assert angle.max() <= 15.0 + 1e-6


def test_axis_rotation_turns_gravity_the_expected_way():
    window = np.array([[[1.0, 0, 0, 0, 0, 0]]], np.float32)
    turned = rotate_windows(window, axis_rotation(2, 90.0))
    np.testing.assert_allclose(turned[0, 0, :3], [0, 1, 0], atol=1e-6)


def test_rotation_moves_both_sensors_alike():
    window = np.array([[[1.0, 0, 0, 1.0, 0, 0]]], np.float32)
    turned = rotate_windows(window, axis_rotation(2, 90.0))
    np.testing.assert_allclose(turned[0, 0, :3], turned[0, 0, 3:], atol=1e-6)
