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
        help=(
            "Scenario(s) to run: all, omni_robot, robot_arm, simple_robot, "
            "point_mass. all excludes deprecated simple_robot/point_mass."
        ),
    )
    parser.add_argument(
        "--method-set",
        default="all",
        choices=["all", "static", "standard", "headline", "paper_core"],
    )
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)
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
        "--learned-mode",
        choices=["none", "regret"],
        default="none",
        help="Optional learned reducer selector to train/evaluate after baselines.",
    )
    parser.add_argument(
        "--regret-oracle",
        choices=["beam3", "sequence3", "pair_rollout3", "rollout_wide", "sequence_wide"],
        default="beam3",
        help="MPC teacher used for regret/ranking distillation.",
    )
    parser.add_argument("--regret-iterations", type=int, default=3)
    parser.add_argument("--regret-epochs", type=int, default=100)
    parser.add_argument("--regret-train-seeds", type=int, default=None)
    parser.add_argument("--regret-eval-seeds", type=int, default=None)
    parser.add_argument(
        "--regret-loss",
        choices=["pairwise", "mse"],
        default="pairwise",
        help="Training loss for regret/ranking distillation.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    parser.add_argument(
        "--budget-sweep",
        type=str,
        default=None,
        help="Comma-separated list of budgets to sweep (e.g. '6,8,10,12').",
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
    if args.length is not None:
        overrides["length"] = args.length
    if args.horizon is not None:
        overrides["horizon"] = args.horizon
    if args.beam_width is not None:
        overrides["beam_width"] = args.beam_width
    if args.seeds is not None:
        overrides["seeds"] = args.seeds

    config = from_profile(args.profile, **overrides)

    if args.budget_sweep:
        budgets = [int(b.strip()) for b in args.budget_sweep.split(",")]
        print(f"Budget sweep over {budgets} (learned={args.learned_mode}).")
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

    if args.learned_mode == "regret":
        _run_regret_distillation(config, results, args)

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
        if args.learned_mode == "regret":
            _run_regret_distillation(cfg, results, args, output_dir=budget_output)
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


def _run_regret_distillation(
    config: BenchmarkConfig,
    results: dict,
    args: argparse.Namespace,
    output_dir: Path | None = None,
) -> None:
    from pzr.experiments.benchmark import (
        default_scenarios,
        registered_scenarios,
    )
    from pzr.experiments.evaluation import aggregate_summary
    from pzr.experiments.regret_eval import (
        RegretOracleConfig,
        train_and_evaluate_regret,
        write_regret_artifacts,
    )
    from pzr.experiments.runner import summarize_results

    scenarios = (
        default_scenarios()
        if config.scenario == "all"
        else [s for s in registered_scenarios() if s.name == config.scenario]
    )

    for scenario in scenarios:
        train_count = args.regret_train_seeds or config.seeds
        eval_count = args.regret_eval_seeds or max(config.seeds // 3, 3)
        train_seeds = range(train_count)
        eval_seeds = range(train_count, train_count + eval_count)
        oracle_config = RegretOracleConfig(
            mode=args.regret_oracle,
            horizon=config.horizon,
            beam_width=config.beam_width,
            cost_weights=config.cost_weights,
        )

        print(f"\n{'=' * 60}")
        print(
            f"  Regret distillation: {scenario.name} "
            f"(oracle={args.regret_oracle})"
        )
        print(f"{'=' * 60}")

        t0 = time.perf_counter()
        result = train_and_evaluate_regret(
            monitor=scenario.monitor,
            trace_fn=scenario.trace_fn,
            budget=config.budget,
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            length=config.length,
            oracle_config=oracle_config,
            iterations=args.regret_iterations,
            epochs_per_iteration=args.regret_epochs,
            regret_loss=args.regret_loss,
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
            print(f"  Final train top-1: {last.train_top1_accuracy:.3f}")
            print(f"  Final val top-1: {last.val_top1_accuracy:.3f}")
            print(f"  Final val chosen regret: {last.val_mean_chosen_regret:.6f}")
        print(f"  Regret distillation completed in {elapsed:.1f}s")

        if scenario.name in results and result.eval_results:
            from pzr.experiments.runner import results_to_dataframe
            regret_summary = summarize_results(result.eval_results)
            regret_agg = aggregate_summary(regret_summary)
            regret_ts = results_to_dataframe(result.eval_results)

            br = results[scenario.name]
            br.aggregate = pd.concat([br.aggregate, regret_agg], ignore_index=True)
            br.summary = pd.concat([br.summary, regret_summary], ignore_index=True)
            br.timeseries = pd.concat([br.timeseries, regret_ts], ignore_index=True)
            br.raw_results.extend(result.eval_results)

            write_regret_artifacts(
                result,
                (output_dir or args.output) / "learning" / scenario.name,
                metadata={
                    "scenario": scenario.name,
                    "length": config.length,
                    "budget": config.budget,
                    "horizon": config.horizon,
                    "beam_width": config.beam_width,
                    "train_seeds": list(train_seeds),
                    "eval_seeds": list(eval_seeds),
                    "cost_weights": config.cost_weights.__dict__,
                    "candidate_names": list(oracle_config.candidate_names),
                    "regret_loss": args.regret_loss,
                },
            )

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
