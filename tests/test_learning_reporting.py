import pandas as pd

from pzr.learning.reporting import write_learning_plots


def test_learning_plots_are_non_empty(tmp_path):
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
            ("figure8", "learned_direct", 40),
            ("square", "girard", 80),
            ("square", "learned_direct", 80),
        ))
    ])
    timeseries = pd.DataFrame([
        {
            "trace_kind": trace,
            "method": "learned_direct",
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

    write_learning_plots(timeseries, summary, tmp_path)

    assert {path.name for path in tmp_path.glob("*.png")} == {
        "metrics_vs_budget.png",
        "generalization_by_trace.png",
        "candidate_selection.png",
        "learned_loss_over_time.png",
    }
    assert all(path.stat().st_size > 0 for path in tmp_path.glob("*.png"))
