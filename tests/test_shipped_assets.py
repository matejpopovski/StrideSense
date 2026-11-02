"""Checks on the committed model and fixtures. They need no dataset and no torch."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ml import data, preprocess
from ml.preprocess import SensorStream
from ml.runtime import LiteRtModel

ASSETS = Path("app/src/main/assets")
SPEC = json.loads((ASSETS / "preprocessing.json").read_text())
MODEL = ASSETS / SPEC["model_file"]
GOLDEN = json.loads(Path("app/src/androidTest/assets/golden_windows.json").read_text())
FIXTURES = sorted(Path("fixtures/parity").glob("*.json"))


def test_the_spec_names_the_model_it_was_exported_with():
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == SPEC["model_sha256"]


def test_the_spec_agrees_with_the_reference_constants():
    assert SPEC["sample_period_ns"] == preprocess.SAMPLE_PERIOD_NS
    assert SPEC["standard_gravity"] == preprocess.STANDARD_GRAVITY
    assert SPEC["window_length"] == preprocess.WINDOW_LENGTH
    assert SPEC["window_stride"] == preprocess.WINDOW_STRIDE
    assert SPEC["max_gap_ns"] == preprocess.MAX_GAP_NS
    assert SPEC["labels"] == list(data.ACTIVITIES)
    assert SPEC["channels"] == list(data.DEVICE_CHANNELS)
    b, a, zi = preprocess.butterworth()
    np.testing.assert_array_equal(SPEC["filter_b"], b)
    np.testing.assert_array_equal(SPEC["filter_a"], a)
    np.testing.assert_array_equal(SPEC["filter_zi"], zi)


def test_the_model_takes_one_window_and_gives_six_probabilities():
    model = LiteRtModel(MODEL)
    assert model.input["shape"].tolist() == [1, 128, 6]
    assert model.input["dtype"] == np.float32
    probabilities = model(np.zeros((128, 6), np.float32))
    assert probabilities.shape == (6,)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)


def test_the_model_uses_builtin_operators_only():
    assert not [name for name in LiteRtModel(MODEL).builtin_ops() if name.startswith("Flex")]


def test_the_runtime_reproduces_the_golden_outputs():
    model = LiteRtModel(MODEL)
    windows = np.array(GOLDEN["windows"], dtype=np.float32).reshape(-1, 128, 6)
    np.testing.assert_allclose(model.predict(windows), np.array(GOLDEN["probabilities"]), atol=1e-5)


def test_the_golden_windows_were_recorded_from_this_model():
    assert GOLDEN["model_sha256"] == SPEC["model_sha256"]


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_the_reference_still_produces_the_committed_parity_windows(path):
    fixture = json.loads(path.read_text())
    streams = [
        SensorStream(np.array(fixture[name]["t"], dtype=np.int64), np.array(fixture[name]["v"]))
        for name in ("accelerometer", "gyroscope")
    ]
    runs = preprocess.filtered_runs(*streams)
    windows = np.concatenate([preprocess.sliding_windows(run) for run in runs])
    np.testing.assert_allclose(
        windows.reshape(len(windows), -1), np.array(fixture["windows"]), atol=1e-12
    )
    scaled = preprocess.standardize(windows, np.array(SPEC["mean"]), np.array(SPEC["std"]))
    np.testing.assert_allclose(
        scaled.reshape(len(scaled), -1), np.array(fixture["model_input"]), atol=1e-6
    )


def test_there_are_parity_fixtures_to_check():
    assert len(FIXTURES) == 5
