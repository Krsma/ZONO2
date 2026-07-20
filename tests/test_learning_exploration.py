import pandas as pd

from pzr.learning.exploration import screen_challengers
from pzr.rtlola.policy_evaluation import PolicyComparison


COMPARISONS = (
    PolicyComparison(
        "data_scale", "pairwise_ranking_policy_clean36", "pairwise_ranking_policy_clean20",
    ),
    PolicyComparison(
        "dart_effect", "pairwise_ranking_policy_dart36", "pairwise_ranking_policy_clean36",
    ),
    PolicyComparison(
        "objective", "expected_regret_clean20", "pairwise_ranking_policy_clean20",
    ),
)


def _summary(
    *,
    clean20: float = 100.0,
    clean36: float = 97.0,
    dart36: float = 94.0,
    expected20: float = 96.6,
    expected_false_positives: int = 0,
) -> pd.DataFrame:
    values = {
        "pairwise_ranking_policy_clean20": (clean20, 10.0, 0),
        "pairwise_ranking_policy_clean36": (clean36, 9.7, 0),
        "pairwise_ranking_policy_dart36": (dart36, 9.4, 0),
        "expected_regret_clean20": (expected20, 9.6, expected_false_positives),
        "girard": (120.0, 12.0, 0),
    }
    return pd.DataFrame([
        {
            "trace_kind": "figure8", "budget": 40, "method": method,
            "sum_approx_loss": summed_loss, "mean_approx_loss": mean_loss,
            "false_positive_count": false_positives,
            "false_negative_count": 0, "infeasible_candidate_count": 0,
            "fallback_count": 0,
        }
        for method, (summed_loss, mean_loss, false_positives) in values.items()
    ])


def _validation() -> dict[str, float]:
    return {
        "pairwise_ranking_policy_clean20": 0.10,
        "pairwise_ranking_policy_clean36": 0.09,
        "pairwise_ranking_policy_dart36": 0.08,
        "expected_regret_clean20": 0.10,
    }


def test_near_tied_passing_challengers_prefer_clean36():
    assessments, selection = screen_challengers(
        _summary(), COMPARISONS, _validation(),
    )
    assert assessments["passed"].all()
    assert selection["winner"] == {
        "comparison": "data_scale",
        "challenger": "pairwise_ranking_policy_clean36",
        "reference": "pairwise_ranking_policy_clean20",
    }
    assert not selection["stop_method_expansion"]


def test_largest_reduction_wins_outside_near_tie_band():
    _, selection = screen_challengers(
        _summary(expected20=95.0), COMPARISONS, _validation(),
    )
    assert selection["winner"]["challenger"] == "expected_regret_clean20"


def test_safety_regression_blocks_challenger_and_empty_pass_set_stops():
    validation = _validation()
    validation["pairwise_ranking_policy_clean36"] = 0.11
    validation["pairwise_ranking_policy_dart36"] = 0.12
    assessments, selection = screen_challengers(
        _summary(expected20=95.0, expected_false_positives=1),
        COMPARISONS,
        validation,
    )
    assert not assessments["passed"].any()
    assert selection["winner"] is None
    assert selection["stop_method_expansion"]


def test_safety_gate_is_cellwise_not_net_counted():
    first = _summary(expected20=95.0)
    second = _summary(expected20=95.0)
    second["trace_kind"] = "random"
    first.loc[first["method"] == "expected_regret_clean20", "false_positive_count"] = 1
    second.loc[
        second["method"] == "pairwise_ranking_policy_clean20", "false_positive_count"
    ] = 1
    assessments, _ = screen_challengers(
        pd.concat([first, second], ignore_index=True), COMPARISONS, _validation(),
    )
    objective = assessments[assessments["comparison"] == "objective"].iloc[0]
    assert objective["false_positive_count_delta"] == 0
    assert not objective["no_added_safety_events"]
    assert not objective["passed"]
