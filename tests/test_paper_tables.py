import json

import pandas as pd

from pzr.experiments.paper_tables import collect_experiment_tables, write_paper_tables


def test_collect_experiment_tables_normalizes_benchmark_and_robotics(tmp_path):
    omni = tmp_path / "omni" / "budget_8"
    (omni / "omni_robot").mkdir(parents=True)
    (omni / "config.yaml").write_text(
        "length: 12\nseeds: 2\nbudget: 8\nhorizon: 4\n",
        encoding="utf-8",
    )
    pd.DataFrame([
        {
            "method": "girard",
            "mean_trigger_width_mean": 10.0,
            "max_trigger_width_mean": 20.0,
            "false_positive_rate_mean": 0.1,
            "mean_approx_error_mean": 1.0,
            "total_time_ms_mean": 2.0,
            "budget_violations_mean": 0.0,
            "unsound_certificates_mean": 0.0,
        },
        {
            "method": "mpc_beam3",
            "mean_trigger_width_mean": 8.0,
            "max_trigger_width_mean": 18.0,
            "false_positive_rate_mean": 0.05,
            "mean_approx_error_mean": 0.7,
            "total_time_ms_mean": 40.0,
            "budget_violations_mean": 0.0,
            "unsound_certificates_mean": 0.0,
        },
    ]).to_csv(omni / "omni_robot" / "aggregate.csv", index=False)

    robotics = tmp_path / "robotics"
    robotics.mkdir()
    (robotics / "budget_sweep_metadata.json").write_text(json.dumps({
        "trace_source": "procedural",
        "monitor_model": "dynamics_physical_v2,dynamics_physical_v3",
        "length": 12,
        "horizon": 4,
    }), encoding="utf-8")
    pd.DataFrame([
        {
            "candidate": "drone",
            "budget": 8,
            "seed": 0,
            "method": "girard",
            "mean_trigger_width": 11.0,
            "max_trigger_width": 22.0,
            "false_positive_rate": 0.2,
            "mean_approx_error": 1.5,
            "total_time_ms": 3.0,
            "budget_violations": 0,
            "unsound_certificates": 0,
        },
        {
            "candidate": "drone",
            "budget": 8,
            "seed": 0,
            "method": "mpc_beam3",
            "mean_trigger_width": 9.0,
            "max_trigger_width": 19.0,
            "false_positive_rate": 0.15,
            "mean_approx_error": 1.0,
            "total_time_ms": 50.0,
            "budget_violations": 0,
            "unsound_certificates": 0,
        },
    ]).to_csv(robotics / "budget_sweep_summary.csv", index=False)

    combined = collect_experiment_tables([tmp_path / "omni", robotics])

    assert {"omni_robot", "drone"} <= set(combined["environment"])
    assert {"static", "mpc"} <= set(combined["method_family"])
    assert combined["false_positive_rate"].notna().all()


def test_write_paper_tables_emits_latex_fragments(tmp_path):
    robotics = tmp_path / "robotics"
    robotics.mkdir()
    (robotics / "budget_sweep_metadata.json").write_text(json.dumps({
        "trace_source": "procedural",
        "length": 10,
        "horizon": 4,
    }), encoding="utf-8")
    pd.DataFrame([
        {
            "candidate": "drone",
            "budget": 8,
            "seed": 0,
            "method": "girard",
            "mean_trigger_width": 10.0,
            "max_trigger_width": 20.0,
            "false_positive_rate": 0.1,
            "mean_approx_error": 1.0,
            "total_time_ms": 2.0,
            "budget_violations": 0,
            "unsound_certificates": 0,
        },
        {
            "candidate": "drone",
            "budget": 8,
            "seed": 0,
            "method": "mpc_beam3",
            "mean_trigger_width": 7.0,
            "max_trigger_width": 17.0,
            "false_positive_rate": 0.05,
            "mean_approx_error": 0.5,
            "total_time_ms": 30.0,
            "budget_violations": 0,
            "unsound_certificates": 0,
        },
    ]).to_csv(robotics / "budget_sweep_summary.csv", index=False)

    artifacts = write_paper_tables([robotics], tmp_path / "tables")

    assert artifacts["combined_summary"].stat().st_size > 0
    assert "\\begin{longtable}" in artifacts["main_k_sweep"].read_text(encoding="utf-8")
    assert "\\input{main_k_sweep.tex}" in artifacts["overview"].read_text(encoding="utf-8")


def test_collect_experiment_tables_scans_split_budget_cells(tmp_path):
    root = tmp_path / "split"
    for budget, width in [(8, 10.0), (16, 8.0)]:
        cell = root / f"k{budget}"
        cell.mkdir(parents=True)
        (cell / "budget_sweep_metadata.json").write_text(json.dumps({
            "trace_source": "procedural",
            "monitor_model": "dynamics_physical_v2",
            "length": 250,
            "horizon": 4,
        }), encoding="utf-8")
        pd.DataFrame([
            {
                "candidate": "drone",
                "budget": budget,
                "seed": 0,
                "method": "girard",
                "mean_trigger_width": width,
                "max_trigger_width": width * 2.0,
                "false_positive_rate": 0.1,
                "mean_approx_error": 1.0,
                "total_time_ms": 2.0,
                "budget_violations": 0,
                "unsound_certificates": 0,
            },
            {
                "candidate": "drone",
                "budget": budget,
                "seed": 0,
                "method": "mpc_beam3",
                "mean_trigger_width": width - 1.0,
                "max_trigger_width": width * 2.0 - 1.0,
                "false_positive_rate": 0.05,
                "mean_approx_error": 0.5,
                "total_time_ms": 30.0,
                "budget_violations": 0,
                "unsound_certificates": 0,
            },
        ]).to_csv(cell / "budget_sweep_summary.csv", index=False)

    combined = collect_experiment_tables([root])

    assert set(combined["budget"]) == {8, 16}
    assert set(combined["method"]) == {"girard", "mpc_beam3"}
