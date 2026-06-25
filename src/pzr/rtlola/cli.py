"""CLI for RTLola-native PZR benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pzr.experiments.evaluation import aggregate_summary
from pzr.rtlola.learning import train_and_evaluate_regret, write_regret_artifacts
from pzr.rtlola.runner import (
    RTLOLA_AGGREGATE_METRICS,
    RtlolaBenchmarkConfig,
    run_benchmark,
    save_benchmark_results,
)


PROFILE_DEFAULTS = {
    "smoke": {"length": 30, "seeds": 3, "horizon": 2},
    "standard": {"length": 200, "seeds": 10, "horizon": 4},
    "paper": {"length": 200, "seeds": 30, "horizon": 4},
}


def _parse_methods(value: str) -> list[str]:
    methods = [part.strip() for part in value.split(",") if part.strip()]
    if not methods:
        raise argparse.ArgumentTypeError("--methods must contain at least one method name")
    return methods


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run RTLola-native PZR benchmark")
    parser.add_argument("--profile", choices=PROFILE_DEFAULTS, default="smoke")
    parser.add_argument("--scenario", default="omni_robot")
    parser.add_argument(
        "--trace-kind",
        default="default",
        help="RTLola trace kind; robot_arm supports figure8_violated, figure8, square_violated, square",
    )
    parser.add_argument("--method-set", choices=["static", "mpc", "all"], default="all")
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=None,
        help="Comma-separated RTLola methods to run; overrides --method-set",
    )
    parser.add_argument("--reference-mode", choices=["exact", "off"], default="exact")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/rtlola"))
    parser.add_argument("--learned-mode", choices=["none", "regret"], default="none")
    parser.add_argument("--regret-iterations", type=int, default=3)
    parser.add_argument("--regret-epochs", type=int, default=100)
    parser.add_argument("--regret-train-seeds", type=int, default=None)
    parser.add_argument("--regret-eval-seeds", type=int, default=None)
    parser.add_argument("--regret-loss", choices=["pairwise", "mse"], default="pairwise")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    params = {
        **PROFILE_DEFAULTS[args.profile],
        "scenario": args.scenario,
        "trace_kind": args.trace_kind,
        "method_set": args.method_set,
        "methods": args.methods,
        "reference_mode": args.reference_mode,
        "output_dir": str(args.output),
        "learned_mode": args.learned_mode,
        "regret_iterations": args.regret_iterations,
        "regret_epochs": args.regret_epochs,
        "regret_train_seeds": args.regret_train_seeds,
        "regret_eval_seeds": args.regret_eval_seeds,
        "regret_loss": args.regret_loss,
    }
    for name in ("budget", "length", "horizon", "beam_width", "seeds"):
        value = getattr(args, name)
        if value is not None:
            params[name] = value
    config = RtlolaBenchmarkConfig(**params)

    result = run_benchmark(config)
    if args.learned_mode == "regret":
        learned = train_and_evaluate_regret(config, show_progress=not args.no_progress)
        learned_summary = summarize_learned(learned.eval_results)
        learned_timeseries = learned_timeseries_df(learned.eval_results)
        result.summary = pd.concat([result.summary, learned_summary], ignore_index=True)
        result.timeseries = pd.concat([result.timeseries, learned_timeseries], ignore_index=True)
        result.aggregate = aggregate_summary(
            result.summary,
            metric_columns=RTLOLA_AGGREGATE_METRICS,
        )
        write_regret_artifacts(
            learned,
            args.output / "learning" / config.scenario,
            metadata={
                "scenario": config.scenario,
                "trace_kind": config.trace_kind,
                "budget": config.budget,
            },
        )

    save_benchmark_results(result, args.output)
    print(f"RTLola benchmark complete: {args.output}")


def summarize_learned(results):
    from pzr.rtlola.runner import summarize_results

    return summarize_results(results)


def learned_timeseries_df(results):
    from pzr.rtlola.runner import results_to_dataframe

    return results_to_dataframe(results)


if __name__ == "__main__":
    main()
