import numpy as np

from pzr.rtlola.features import (
    RTL_RANKING_FEATURE_NAMES,
    ranking_features_from_matrices,
)


def test_ranking_features_preserve_dense_active_zero_and_constant_distinctions():
    dynamic = np.asarray([
        [1.0, -2.0, 0.0, 0.0],
        [2.0, 0.0, 3.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
    ])
    total = np.column_stack((dynamic, np.asarray([0.0, 0.0, 4.0])))
    features = ranking_features_from_matrices(dynamic, total, budget=2)
    by_name = dict(zip(RTL_RANKING_FEATURE_NAMES, features))

    assert by_name["dynamic_generator_count"] == 3
    assert by_name["active_dynamic_generator_count"] == 2
    assert by_name["zero_dynamic_fraction"] == np.float32(1 / 3)
    assert by_name["compact_dimension"] == 3
    assert by_name["logical_dynamic_dimension"] == 3
    assert by_name["state_width"] == 10.0
    assert by_name["mean_active_generator_norm"] == 2.5


def test_ranking_features_are_generator_permutation_and_sign_invariant():
    dynamic = np.asarray([
        [0.0, -1.0, 2.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
    ])
    total = dynamic.copy()
    expected = ranking_features_from_matrices(dynamic, total, budget=2)
    permuted = dynamic[:, [0, 2, 3, 1]].copy()
    permuted[:, 1:] *= -1.0
    actual = ranking_features_from_matrices(permuted, permuted, budget=2)
    np.testing.assert_allclose(actual, expected)
