import numpy as np
import pandas as pd
import pytest

from pzr.learning.dart import (
    DartCalibration,
    DartCalibrationConfig,
    calibrate_dart,
)
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
        "trace_id": ("train", "validation", "validation", "validation"),
        "step": (0, 0, 1, 2),
    })


def test_dart_calibrates_global_error_radius_direction_and_recovery_scale():
    calibration, budgets, directions = calibrate_dart(
        _Policy(),
        _dataset(),
        _metadata(),
        split="validation",
        context={"source": "test"},
    )

    assert calibration.target_disturbance_rates[0] == pytest.approx(2 / 3)
    assert calibration.regret_caps[0] == pytest.approx(0.94)
    assert calibration.injection_probabilities[0] == pytest.approx(1.0)
    assert calibration.expected_disturbance_rates[0] == pytest.approx(2 / 3)
    np.testing.assert_allclose(calibration.direction_probabilities[0, 0], [0.0, 0.5, 0.5])
    np.testing.assert_allclose(calibration.direction_probabilities[0, 1], [0.5, 0.0, 0.5])
    assert calibration.row_counts[0, 0] == 3
    assert calibration.error_counts[0, 0] == 2
    assert budgets.iloc[0]["meaningful_novice_error_count"] == 2
    assert len(directions) == 9


def test_dart_alternative_distribution_masks_teacher_infeasible_and_regret():
    calibration, _, _ = calibrate_dart(
        _Policy(), _dataset(), _metadata(), split="validation", context={},
    )
    distribution = calibration.alternative_distribution(
        40,
        "girard",
        np.asarray([True, True, False]),
        np.asarray([0.0, 0.4, np.nan]),
    )
    np.testing.assert_array_equal(distribution, [0.0, 1.0, 0.0])


def test_dart_zero_error_calibration_injects_no_noise():
    dataset = _dataset()
    zero_policy = _Policy()
    dataset = ReducerCostDataset(
        features=np.zeros_like(dataset.features),
        teacher_costs=dataset.teacher_costs,
        feasible=dataset.feasible,
        candidate_names=dataset.candidate_names,
        feature_names=dataset.feature_names,
        splits=dataset.splits,
        sample_ids=dataset.sample_ids,
    )
    calibration, budgets, _ = calibrate_dart(
        zero_policy, dataset, _metadata(), split="validation", context={},
    )
    assert calibration.target_disturbance_rates[0] == 0.0
    assert calibration.injection_probabilities[0] == 0.0
    assert calibration.regret_caps[0] == 0.0
    assert not budgets.iloc[0]["saturated"]


def test_dart_calibration_round_trip_is_exact(tmp_path):
    calibration, budget_diagnostics, direction_diagnostics = calibrate_dart(
        _Policy(),
        _dataset(),
        _metadata(),
        split="validation",
        context={"source": "test"},
        config=DartCalibrationConfig(
            regret_cap_quantile=0.9,
            direction_pseudocount=1.0,
            recovery_decisions=1,
        ),
    )
    calibration.save(tmp_path, budget_diagnostics, direction_diagnostics)
    loaded = DartCalibration.load(tmp_path)
    np.testing.assert_array_equal(
        loaded.direction_probabilities, calibration.direction_probabilities,
    )
    np.testing.assert_array_equal(loaded.regret_caps, calibration.regret_caps)
    assert loaded.config == calibration.config
    assert loaded.context == {"source": "test"}
