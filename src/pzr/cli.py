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
        help="Scenario(s) to run: all, omni_robot, simple_robot, point_mass",
    )
    parser.add_argument(
        "--method-set",
        default="all",
        choices=["all", "static", "standard"],
    )
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--dagger",
        action="store_true",
        help="Run DAgger pipeline after benchmark",
    )
    parser.add_argument("--dagger-iterations", type=int, default=3)
    parser.add_argument("--dagger-epochs", type=int, default=100)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )

    args = parser.parse_args(argv)

    overrides: dict = {
        "scenario": args.scenario,
        "method_set": args.method_set,
        "output_dir": str(args.output),
    }
    if args.budget is not None:
        overrides["budget"] = args.budget
    if args.horizon is not None:
        overrides["horizon"] = args.horizon
    if args.seeds is not None:
        overrides["seeds"] = args.seeds

    config = from_profile(args.profile, **overrides)

    print(f"Profile: {args.profile} | Scenario: {config.scenario} | "
          f"Methods: {config.method_set} | Seeds: {config.seeds} | "
          f"Length: {config.length} | Budget: {config.budget} | "
          f"Horizon: {config.horizon}")
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

    if args.dagger:
        _run_dagger(config, results, args)

    _generate_figures(results, args.output)


def _run_dagger(
    config: BenchmarkConfig,
    results: dict,
    args: argparse.Namespace,
) -> None:
    from pzr.experiments.benchmark import default_scenarios, default_methods
    from pzr.experiments.dagger_eval import train_and_evaluate_dagger
    from pzr.experiments.evaluation import aggregate_summary
    from pzr.experiments.runner import summarize_results

    scenarios = default_scenarios()
    if config.scenario != "all":
        scenarios = [s for s in scenarios if s.name == config.scenario]

    for scenario in scenarios:
        methods = default_methods(
            scenario.monitor, config.budget, config.horizon, config.cost_weights,
        )
        mpc_methods = [m for m in methods if m.name.startswith("mpc")]
        if not mpc_methods:
            print(f"\nSkipping DAgger for {scenario.name}: no MPC methods")
            continue

        expert = mpc_methods[0].policy
        train_seeds = range(config.seeds)
        eval_seeds = range(config.seeds, config.seeds + max(config.seeds // 3, 3))

        print(f"\n{'=' * 60}")
        print(f"  DAgger: {scenario.name} (expert={mpc_methods[0].name})")
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
        plot_combined_timeseries,
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
