import json

import pandas as pd

from pzr.experiments.cli import _selected_methods, main


def test_benchmark_cli_writes_paper_style_artifacts(tmp_path) -> None:
    exit_code = main(
        [
            "robot",
            "--length",
            "8",
            "--budget",
            "6",
            "--horizon",
            "2",
            "--seeds",
            "2",
            "--bootstrap-samples",
            "20",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    for filename in (
        "raw_runs.csv",
        "summary.csv",
        "comparisons.csv",
        "predictor_comparisons.csv",
        "timeseries.csv",
        "bounds_timeseries.csv",
        "decision_features.csv",
        "selection_summary.csv",
        "predicted_sequence_summary.csv",
        "config.json",
        "report.json",
    ):
        assert (tmp_path / filename).exists()

    raw = pd.read_csv(tmp_path / "raw_runs.csv")
    assert set(raw["method"]) == {
        "reference",
        "box",
        "girard",
        "girard7",
        "combastel",
        "methA",
        "scott",
        "pca",
        "adaptive",
        "keep_norm",
        "keep_calibration_aware",
        "mpc",
        "mpc_sequence",
        "mpc_rollout_girard",
        "mpc_rollout_wide",
    }
    assert set(raw["seed"]) == {0, 1}

    budgeted = raw[raw["method"] != "reference"]
    assert (budgeted["budget_violation_count"] == 0).all()
    assert (budgeted["unsound_certificate_count"] == 0).all()
    assert (budgeted["reduction_failure_count"] == 0).all()
    assert (budgeted["unsafe_disagreement_count"] == 0).all()
    assert {"inconclusive_count", "extra_inconclusive_count", "false_alarm_count"} <= set(
        raw.columns
    )
    assert {"no_op_count", "chosen_no_reduction_count"} <= set(raw.columns)
    assert raw["inconclusive_count"].notna().all()
    assert raw["false_alarm_rate"].notna().all()

    mpc = raw[
        raw["method"].isin(
            {"mpc", "mpc_sequence", "mpc_rollout_girard", "mpc_rollout_wide"}
        )
    ]
    chosen_total = (
        mpc["chosen_box_count"]
        + mpc["chosen_girard_count"]
        + mpc["chosen_combastel_count"]
        + mpc["chosen_methA_count"]
        + mpc["chosen_scott_count"]
        + mpc["chosen_pca_count"]
        + mpc["chosen_adaptive_count"]
        + mpc["chosen_keep_norm_count"]
        + mpc["chosen_keep_calibration_aware_count"]
        + mpc["chosen_other_count"]
    )
    assert (chosen_total == mpc["reduction_count"]).all()
    assert (mpc["chosen_no_reduction_count"] == mpc["no_op_count"]).all()
    assert (mpc["no_op_count"] == 0).all()

    summary = pd.read_csv(tmp_path / "summary.csv")
    assert {
        "scenario",
        "predictor_mode",
        "method",
        "metric",
        "mean",
        "ci95_low",
        "ci95_high",
    } <= set(summary.columns)
    assert "inconclusive_rate" in set(summary["metric"])

    timeseries = pd.read_csv(tmp_path / "timeseries.csv")
    assert {
        "interval_hull_mse",
        "trigger_interval_hull_mse",
        "inconclusive_count",
        "false_alarm_rate",
        "verdict_disagreement_count",
        "generator_count",
        "reduction_applied",
        "no_op_selected",
        "predicted_sequence",
    } <= set(timeseries.columns)
    assert not (timeseries["reducer_name"] == "no_reduction").any()
    assert len(timeseries) == len(raw) * 8

    decisions = pd.read_csv(tmp_path / "decision_features.csv")
    assert not decisions.empty
    assert (decisions["generator_count"] > decisions["budget"]).all()
    assert "no_reduction" not in set(decisions["chosen_reducer_label"])

    selection = pd.read_csv(tmp_path / "selection_summary.csv")
    assert {"selected_reducer", "selection_count", "selection_fraction"} <= set(
        selection.columns
    )
    assert not selection.empty

    sequences = pd.read_csv(tmp_path / "predicted_sequence_summary.csv")
    assert {"first_action_box_count", "future_box_count"} <= set(sequences.columns)
    wide_sequences = sequences[sequences["method"] == "mpc_rollout_wide"]
    assert not wide_sequences.empty
    assert (wide_sequences["first_action_box_count"] == 0).all()

    bounds = pd.read_csv(tmp_path / "bounds_timeseries.csv")
    assert {
        "state_name",
        "lower",
        "upper",
        "reference_lower",
        "reference_upper",
    } <= set(bounds.columns)
    assert {"position_x", "position_y"} <= set(bounds["state_name"])

    comparisons = pd.read_csv(tmp_path / "comparisons.csv")
    assert {"method", "baseline", "metric", "wilcoxon_p_value"} <= set(comparisons.columns)
    assert set(comparisons["baseline"]) == {"mpc_rollout_wide"}

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["length"] == 8
    assert config["budget"] == 6
    assert config["horizon"] == 2
    assert config["seeds"] == [0, 1]


def test_paper_plus_wide_method_set_adds_both_rollout_mpc_methods() -> None:
    methods = _selected_methods("paper_plus_wide")

    assert {method.name for method in methods} == {
        "box",
        "girard",
        "combastel",
        "methA",
        "scott",
        "pca",
        "adaptive",
        "mpc_rollout_girard",
        "mpc_rollout_wide",
    }


def test_paper_plus_ours_method_set_keeps_focused_rollout_only() -> None:
    methods = _selected_methods("paper_plus_ours")

    assert "mpc_rollout_girard" in {method.name for method in methods}
    assert "mpc_rollout_wide" not in {method.name for method in methods}


def test_benchmark_cli_can_skip_reference(tmp_path) -> None:
    exit_code = main(
        [
            "robot",
            "--length",
            "7",
            "--budget",
            "6",
            "--horizon",
            "2",
            "--seeds",
            "1",
            "--no-reference",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    raw = pd.read_csv(tmp_path / "raw_runs.csv")
    assert "reference" not in set(raw["method"])


def test_benchmark_cli_can_run_online_and_oracle_modes(tmp_path) -> None:
    exit_code = main(
        [
            "robot",
            "--length",
            "8",
            "--budget",
            "6",
            "--horizon",
            "2",
            "--seeds",
            "1",
            "--predictor-mode",
            "both",
            "--bootstrap-samples",
            "10",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    raw = pd.read_csv(tmp_path / "raw_runs.csv")
    assert set(raw["predictor_mode"]) == {"online", "oracle"}

    summary = pd.read_csv(tmp_path / "summary.csv")
    assert set(summary["predictor_mode"]) == {"online", "oracle"}

    predictor_comparisons = pd.read_csv(tmp_path / "predictor_comparisons.csv")
    assert {
        "method",
        "metric",
        "mean_delta_online_minus_oracle",
        "wilcoxon_p_value",
    } <= set(predictor_comparisons.columns)

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["predictor_mode"] == "both"
