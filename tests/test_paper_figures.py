import pandas as pd

from pzr.experiments.paper_figures import METHOD_LABELS, _selected_methods, main


def test_paper_figures_smoke_writes_summaries_and_figures(tmp_path) -> None:
    exit_code = main(
        [
            "--out",
            str(tmp_path),
            "--method-set",
            "paper",
            "--seeds",
            "1",
            "--length",
            "8",
            "--budget",
            "6",
            "--budgets",
            "6,8",
            "--fig4-length",
            "4",
            "--fig4-seed",
            "0",
            "--fpr-length",
            "8",
            "--formats",
            "png,pdf",
            "--bootstrap-samples",
            "10",
        ]
    )

    assert exit_code == 0
    for name in (
        "fig3a_simple_error_over_time",
        "fig3b_simple_error_by_budget",
        "fig4a_omni_position_x_trace",
        "fig4b_omni_false_alarm_rates",
        "fig5a_omni_error_over_time",
        "fig5b_omni_error_by_budget",
        "selection_robot_simple",
        "selection_robot",
        "selection_thermostat",
        "fallback_box_usage",
    ):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 0
        assert (tmp_path / "figures" / f"{name}.pdf").stat().st_size > 0

    summary = pd.read_csv(tmp_path / "data" / "figure_summaries.csv")
    assert {
        "figure",
        "scenario",
        "budget",
        "method",
        "metric",
        "mean",
        "ci95_low",
        "ci95_high",
    } <= set(summary.columns)

    bounds = pd.read_csv(tmp_path / "data" / "fig4a_omni_position_x_trace.csv")
    assert {"position_x", "position_y"} <= set(bounds["state_name"])

    selection = pd.read_csv(tmp_path / "data" / "selection_summary.csv")
    assert {"robot", "robot_simple", "thermostat"} <= set(selection["scenario"])
    assert (tmp_path / "data" / "predicted_sequence_summary.csv").exists()
    assert (tmp_path / "data" / "fallback_box_usage.csv").exists()
    assert (tmp_path / "data" / "analysis_notes.json").exists()


def test_paper_figures_paper_plus_wide_includes_both_rollout_methods() -> None:
    methods = _selected_methods("paper_plus_wide")

    assert "mpc_rollout_girard" in {method.name for method in methods}
    assert "mpc_rollout_wide" in {method.name for method in methods}
    assert METHOD_LABELS["mpc_rollout_wide"] == "MPC rollout wide (ours)"
