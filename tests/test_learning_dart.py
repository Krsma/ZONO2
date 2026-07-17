import numpy as np
import pandas as pd
import pytest

from pzr.learning.dart import DartCalibration, calibrate_dart
from pzr.learning.dataset import ReducerCostDataset


class _Policy:
    candidate_names = ("girard", "scott", "pca")
    feature_schema = type("Schema", (), {"feature_names": ("prediction",)})()

    def predict_scores(self, features):
        predicted = np.asarray(features)[:, 0].astype(int)
        scores = np.ones((len(predicted), 3), dtype=np.float64)
        scores[np.arange(len(predicted)), predicted] = 0.0
        return scores


def _dataset():
    return ReducerCostDataset(
        features=np.asarray([[2.0], [2.0], [1.0], [0.0]], dtype=np.float32),
        teacher_costs=np.asarray([
            [0.0, 0.4, 1.0], [0.0, 0.4, 1.0],
            [0.0, 0.4, 1.0], [0.0, 0.4, 1.0],
        ]),
        feasible=np.ones((4, 3), dtype=np.bool_),
        candidate_names=_Policy.candidate_names,
        feature_names=("prediction",),
        splits=("train", "validation", "validation", "validation"),
        sample_ids=("a", "b", "c", "d"),
    )


def _metadata():
    return pd.DataFrame({
        "split": ("train", "validation", "validation", "validation"),
        "budget": (40, 40, 40, 40),
        "teacher_action": ("girard", "girard", "girard", "girard"),
    })


def test_dart_kernel_uses_only_held_out_confusion_and_identity_backoff():
    calibration, diagnostics = calibrate_dart(
        _Policy(), _dataset(), _metadata(), split="validation", context={"source": "test"},
    )
    np.testing.assert_allclose(calibration.probabilities[0, 0], [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_array_equal(calibration.probabilities[0, 1], [0.0, 1.0, 0.0])
    assert calibration.row_counts[0, 0] == 3
    assert calibration.row_counts[0, 1] == 0
    girard = diagnostics[diagnostics["teacher_action"] == "girard"].iloc[0]
    assert girard["disagreement_rate"] == pytest.approx(2 / 3)


def test_dart_infeasible_mass_is_redirected_to_teacher():
    calibration = DartCalibration(
        candidate_names=("girard", "scott", "pca"), budgets=(40,),
        probabilities=np.asarray([[
            [0.2, 0.3, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ]]),
        row_counts=np.asarray([[10, 0, 0]]), context={},
    )
    distribution = calibration.collection_distribution(
        40, "girard", np.asarray([True, True, False]),
    )
    np.testing.assert_allclose(distribution, [0.7, 0.3, 0.0])


def test_dart_calibration_round_trip_is_exact(tmp_path):
    calibration, diagnostics = calibrate_dart(
        _Policy(), _dataset(), _metadata(), split="validation", context={"source": "test"},
    )
    calibration.save(tmp_path, diagnostics)
    loaded = DartCalibration.load(tmp_path)
    np.testing.assert_array_equal(loaded.probabilities, calibration.probabilities)
    np.testing.assert_array_equal(loaded.row_counts, calibration.row_counts)
    assert loaded.context == {"source": "test"}
