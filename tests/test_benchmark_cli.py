import json

import pandas as pd

from pzr.experiments.cli import main


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
    }
    assert set(raw["seed"]) == {0, 1}

    budgeted = raw[raw["method"] != "reference"]
    assert (budgeted["budget_violation_count"] == 0).all()
    assert (budgeted["unsound_certificate_count"] == 0).all()
    assert (budgeted["reduction_failure_count"] == 0).all()
    assert (budgeted["unsafe_disagreement_count"] == 0).all()

    mpc = raw[raw["method"].isin({"mpc", "mpc_sequence", "mpc_rollout_girard"})]
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

    comparisons = pd.read_csv(tmp_path / "comparisons.csv")
    assert {"method", "baseline", "metric", "wilcoxon_p_value"} <= set(comparisons.columns)
    assert set(comparisons["baseline"]) == {"mpc_rollout_girard"}

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["length"] == 8
    assert config["budget"] == 6
    assert config["horizon"] == 2
    assert config["seeds"] == [0, 1]


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
