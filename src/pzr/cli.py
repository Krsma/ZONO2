"""Command-line interface for benchmark experiments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from pzr.experiments.benchmark import run_benchmark, save_benchmark_results
from pzr.experiments.config import BenchmarkConfig, from_profile
from pzr.experiments.tables import (
    format_comparison_table,
    format_soundness_report,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run zonotope reduction benchmark",
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "standard", "paper"],
        default="standard",
    )
    parser.add_argument(
        "--scenario",
        default="all",
        help="Scenario(s) to run: all, omni_robot, simple_robot, point_mass, robot_arm",
    )
    parser.add_argument(
        "--method-set",
        default="all",
        choices=["all", "static", "standard"],
    )
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel seed workers for benchmark runs. Use 1 for serial execution.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--no-dagger",
        action="store_true",
        help="Skip the DAgger pipeline (DAgger runs by default)",
    )
    parser.add_argument("--dagger-iterations", type=int, default=3)
    parser.add_argument("--dagger-epochs", type=int, default=100)
    parser.add_argument("--dagger-expert", type=str, default="mpc_sequence3")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    parser.add_argument(
        "--budget-sweep",
        type=str,
        default=None,
        help="Comma-separated list of budgets to sweep (e.g. '6,8,10,12'). Disables DAgger.",
    )

    args = parser.parse_args(argv)

    overrides: dict = {
        "scenario": args.scenario,
        "method_set": args.method_set,
        "output_dir": str(args.output),
        "jobs": args.jobs,
    }
    if args.budget is not None:
        overrides["budget"] = args.budget
    if args.horizon is not None:
        overrides["horizon"] = args.horizon
    if args.beam_width is not None:
        overrides["beam_width"] = args.beam_width
    if args.seeds is not None:
        overrides["seeds"] = args.seeds

    config = from_profile(args.profile, **overrides)

    if args.budget_sweep:
        budgets = [int(b.strip()) for b in args.budget_sweep.split(",")]
        print(f"Budget sweep over {budgets} (DAgger disabled).")
        _run_budget_sweep(config, budgets, args)
        return

    print(f"Profile: {args.profile} | Scenario: {config.scenario} | "
          f"Methods: {config.method_set} | Seeds: {config.seeds} | "
          f"Length: {config.length} | Budget: {config.budget} | "
          f"Horizon: {config.horizon} | Beam: {config.beam_width} | "
          f"Jobs: {config.jobs}")
    print()

    t0 = time.perf_counter()
    results = run_benchmark(config, show_progress=not args.no_progress)
    elapsed = time.perf_counter() - t0

    save_benchmark_results(results, args.output)

    for name, r in results.items():
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        print(format_comparison_table(r.aggregate))
        print()
        print(format_soundness_report(r.summary))

    print(f"\nBenchmark completed in {elapsed:.1f}s")
    print(f"Results saved to {args.output}/")

    if not args.no_dagger:
        _run_dagger(config, results, args)

    _generate_figures(results, args.output)


def _run_budget_sweep(
    base_config: BenchmarkConfig,
    budgets: list[int],
    args: argparse.Namespace,
) -> None:
    """Run the benchmark across a list of budgets and emit trade-off figures."""
    from dataclasses import replace
    from pzr.experiments.figures import plot_budget_sweep

    aggregates_per_scenario: dict[str, dict[int, pd.DataFrame]] = {}

    for budget in budgets:
        cfg = replace(base_config, budget=budget)
        budget_output = args.output / f"budget_{budget}"
        print(f"\n{'=' * 60}\n  Budget = {budget}\n{'=' * 60}")

        t0 = time.perf_counter()
        results = run_benchmark(cfg, show_progress=not args.no_progress)
        elapsed = time.perf_counter() - t0

        save_benchmark_results(results, budget_output)
        print(f"  Budget {budget} completed in {elapsed:.1f}s")

        for name, r in results.items():
            aggregates_per_scenario.setdefault(name, {})[budget] = r.aggregate

        _generate_figures(results, budget_output)

    sweep_dir = args.output / "budget_sweep"
    for scenario, agg_by_budget in aggregates_per_scenario.items():
        for metric in (
            "mean_trigger_width",
            "false_positive_rate",
            "mean_approx_error",
            "total_time_ms",
        ):
            plot_budget_sweep(
                agg_by_budget, metric=metric,
                title=f"{scenario} — {metric} vs budget",
                out_path=sweep_dir / f"{scenario}_{metric}_vs_budget.pdf",
            )

    print(f"\nBudget sweep figures saved to {sweep_dir}/")


def _run_dagger(
    config: BenchmarkConfig,
    results: dict,
    args: argparse.Namespace,
) -> None:
    from pzr.experiments.benchmark import (
        TOP3_REDUCER_NAMES,
        default_methods,
        default_scenarios,
    )
    from pzr.experiments.dagger_eval import train_and_evaluate_dagger
    from pzr.experiments.evaluation import aggregate_summary
    from pzr.experiments.runner import summarize_results

    scenarios = default_scenarios()
    if config.scenario != "all":
        scenarios = [s for s in scenarios if s.name == config.scenario]

    for scenario in scenarios:
        methods = default_methods(
            scenario.monitor, config.budget, config.horizon, config.cost_weights,
            config.beam_width,
        )
        mpc_methods = [m for m in methods if m.name.startswith("mpc")]
        if not mpc_methods:
            print(f"\nSkipping DAgger for {scenario.name}: no MPC methods")
            continue

        method_by_name = {m.name: m for m in mpc_methods}
        if args.dagger_expert not in method_by_name:
            available = ", ".join(sorted(method_by_name))
            raise ValueError(
                f"unknown DAgger expert {args.dagger_expert!r}; available: {available}"
            )
        expert = method_by_name[args.dagger_expert].policy
        train_seeds = range(config.seeds)
        eval_seeds = range(config.seeds, config.seeds + max(config.seeds // 3, 3))

        print(f"\n{'=' * 60}")
        print(f"  DAgger: {scenario.name} (expert={args.dagger_expert})")
        print(f"{'=' * 60}")

        t0 = time.perf_counter()
        result = train_and_evaluate_dagger(
            monitor=scenario.monitor,
            trace_fn=scenario.trace_fn,
            expert_policy=expert,
            budget=config.budget,
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            length=config.length,
            dagger_iterations=args.dagger_iterations,
            epochs_per_iteration=args.dagger_epochs,
            candidate_names=TOP3_REDUCER_NAMES,
            show_progress=not args.no_progress,
        )
        elapsed = time.perf_counter() - t0

        print(f"  Traces collected: {result.total_traces}")
        print(f"  Eval runs: {len(result.eval_results)}")
        print(f"  Avg inference: {result.inference_time_ms:.3f} ms")
        violations = sum(r.budget_violations for r in result.eval_results)
        unsound = sum(r.unsound_certificates for r in result.eval_results)
        print(f"  Budget violations: {violations}")
        print(f"  Unsound certificates: {unsound}")
        if result.training_results:
            last = result.training_results[-1]
            print(f"  Final train acc: {last.train_accuracy:.3f}")
            print(f"  Final val acc: {last.val_accuracy:.3f}")
        print(f"  DAgger completed in {elapsed:.1f}s")

        if scenario.name in results and result.eval_results:
            from pzr.experiments.runner import results_to_dataframe
            dagger_summary = summarize_results(result.eval_results)
            dagger_agg = aggregate_summary(dagger_summary)
            dagger_ts = results_to_dataframe(result.eval_results)

            br = results[scenario.name]
            br.aggregate = pd.concat([br.aggregate, dagger_agg], ignore_index=True)
            br.summary = pd.concat([br.summary, dagger_summary], ignore_index=True)
            br.timeseries = pd.concat([br.timeseries, dagger_ts], ignore_index=True)

            print(f"\n  Updated comparison table ({scenario.name}):")
            print(format_comparison_table(br.aggregate))


def _generate_figures(results: dict, output: Path) -> None:
    from pzr.experiments.figures import (
        plot_approximation_error_timeseries,
        plot_combined_timeseries,
        plot_fig4_panel,
        plot_method_comparison_bars,
        plot_reducer_selection_bars,
    )

    fig_dir = output / "figures"

    for name, r in results.items():
        plot_combined_timeseries(
            r.timeseries, title=name,
            out_path=fig_dir / f"{name}_timeseries.pdf",
        )
        plot_method_comparison_bars(
            r.aggregate, metric="mean_trigger_width",
            title=f"{name} — Mean Trigger Width",
            out_path=fig_dir / f"{name}_trigger_width_bars.pdf",
        )
        plot_method_comparison_bars(
            r.aggregate, metric="total_time_ms",
            title=f"{name} — Total Reduction Time",
            out_path=fig_dir / f"{name}_time_bars.pdf",
        )
        plot_approximation_error_timeseries(
            r.timeseries, title=f"{name} — Approximation Error",
            out_path=fig_dir / f"{name}_approx_error_timeseries.pdf",
        )
        plot_fig4_panel(
            r.aggregate, title=f"{name} — FPR & Absolute Error Range",
            out_path=fig_dir / f"{name}_fig4_panel.pdf",
        )
        mpc_methods = [m for m in r.timeseries["method"].unique()
                       if m.startswith("mpc") or m.startswith("learned")]
        if mpc_methods:
            plot_reducer_selection_bars(
                r.timeseries, methods=mpc_methods,
                title=f"{name} — Reducer Selection",
                out_path=fig_dir / f"{name}_reducer_selection.pdf",
            )

    print(f"Figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
