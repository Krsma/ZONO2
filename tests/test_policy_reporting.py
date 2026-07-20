import pandas as pd

from pzr.rtlola.policy_reporting import write_policy_plots


def test_policy_plots_are_non_empty(tmp_path):
    summary = pd.DataFrame([
        {
            "trace_kind": trace,
            "method": method,
            "budget": budget,
            "mean_approx_loss": float(budget + index),
            "mean_state_width": float(index + 1),
            "fpr": 0.1 * index,
            "fnr": 0.05 * index,
        }
        for index, (trace, method, budget) in enumerate((
            ("figure8", "girard", 40),
            ("figure8", "pairwise_ranking_policy", 40),
            ("square", "girard", 80),
            ("square", "pairwise_ranking_policy", 80),
        ))
    ])
    timeseries = pd.DataFrame([
        {
            "trace_kind": trace,
            "method": "pairwise_ranking_policy",
            "budget": budget,
            "step": step,
            "approx_loss": float(step + 1),
            "reducer_used": reducer,
        }
        for trace, budget, reducer in (
            ("figure8", 40, "girard"),
            ("square", 80, "scott"),
        )
        for step in range(2)
    ])

    write_policy_plots(
        timeseries, summary, tmp_path, policy_methods=("pairwise_ranking_policy",),
    )

    assert {path.name for path in tmp_path.glob("*.png")} == {
        "metrics_vs_budget.png",
        "generalization_by_trace.png",
        "candidate_composition_pairwise_ranking_policy.png",
        "loss_over_time_pairwise_ranking_policy.png",
    }
    assert all(path.stat().st_size > 0 for path in tmp_path.glob("*.png"))
