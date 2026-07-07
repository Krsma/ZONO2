"""CLI for RTLola-native PZR benchmark runs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from pzr.rtlola.benchmark import (
    RTLOLA_AGGREGATE_METRICS,
    RtlolaBenchmarkConfig,
    aggregate_summary,
    prepare_reference_cache,
    results_to_dataframe,
    run_benchmark,
    save_benchmark_results,
    summarize_results,
)
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.learning import train_and_evaluate_regret, write_regret_artifacts
from pzr.rtlola.scenarios import scenario_by_name


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


def _parse_csv(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("comma-separated value must not be empty")
    return values


def _parse_budgets(value: str) -> list[int]:
    try:
        budgets = [int(part) for part in _parse_csv(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("budgets must be comma-separated integers") from exc
    if any(budget < 0 for budget in budgets):
        raise argparse.ArgumentTypeError("budgets must be non-negative")
    return budgets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run RTLola-native PZR benchmark")
    parser.add_argument("--profile", choices=PROFILE_DEFAULTS, default="smoke")
    parser.add_argument("--scenario", default="omni_robot")
    parser.add_argument(
        "--trace-kind",
        default="default",
        help=(
            "RTLola trace kind; robot_arm supports figure8, figure8_drift, "
            "random, random_violated, square, square_drift"
        ),
    )
    parser.add_argument(
        "--method-set",
        choices=["core", "static", "mpc", "all"],
        default="core",
    )
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=None,
        help="Comma-separated RTLola methods to run; overrides --method-set",
    )
    parser.add_argument(
        "--reference-mode",
        choices=["exact", "verdict", "off"],
        default="exact",
        help=(
            "exact caches compact unreduced references for approximation loss "
            "and FPR/FNR; verdict caches only exact trigger outcomes"
        ),
    )
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=None,
        help="Optional JSON cache for exact trigger and approximation references",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Generate or validate --reference-cache and exit without a benchmark run",
    )
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--mpc-tail-horizon", type=int, default=8)
    parser.add_argument("--mpc-root-beam-width", type=int, default=1)
    parser.add_argument(
        "--mpc-candidates",
        type=_parse_csv,
        default=None,
        help=(
            "Comma-separated subset of the default MPC transform catalog; "
            "defaults to the full configured catalog"
        ),
    )
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/rtlola"))
    parser.add_argument("--learned-mode", choices=["none", "regret"], default="none")
    parser.add_argument("--regret-iterations", type=int, default=3)
    parser.add_argument("--regret-epochs", type=int, default=100)
    parser.add_argument("--regret-train-seeds", type=int, default=None)
    parser.add_argument("--regret-eval-seeds", type=int, default=None)
    parser.add_argument("--regret-train-seed-start", type=int, default=10_000)
    parser.add_argument("--regret-eval-seed-start", type=int, default=0)
    parser.add_argument("--regret-loss", choices=["pairwise", "mse"], default="pairwise")
    parser.add_argument(
        "--regret-budgets",
        type=_parse_budgets,
        default=None,
        help="Budgets pooled into one learned policy and evaluated separately",
    )
    parser.add_argument(
        "--regret-train-traces",
        type=_parse_csv,
        default=None,
        help="Trace kinds pooled for policy training",
    )
    parser.add_argument(
        "--regret-eval-traces",
        type=_parse_csv,
        default=None,
        help="Held-out trace kinds used for learned-policy evaluation",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    params = {
        **PROFILE_DEFAULTS[args.profile],
        "scenario": args.scenario,
        "trace_kind": args.trace_kind,
        "method_set": args.method_set,
        "methods": args.methods,
        "reference_mode": args.reference_mode,
        "reference_cache": (
            str(args.reference_cache)
            if args.reference_cache is not None else None
        ),
        "output_dir": str(args.output),
        "learned_mode": args.learned_mode,
        "regret_iterations": args.regret_iterations,
        "regret_epochs": args.regret_epochs,
        "regret_train_seeds": args.regret_train_seeds,
        "regret_eval_seeds": args.regret_eval_seeds,
        "regret_train_seed_start": args.regret_train_seed_start,
        "regret_eval_seed_start": args.regret_eval_seed_start,
        "regret_loss": args.regret_loss,
        "regret_budgets": args.regret_budgets,
        "regret_train_trace_kinds": args.regret_train_traces,
        "regret_eval_trace_kinds": args.regret_eval_traces,
        "mpc_tail_horizon": args.mpc_tail_horizon,
        "mpc_root_beam_width": args.mpc_root_beam_width,
    }
    if args.mpc_candidates is not None:
        params["mpc_candidate_names"] = args.mpc_candidates
    for name in ("budget", "length", "horizon", "beam_width", "seeds"):
        value = getattr(args, name)
        if value is not None:
            params[name] = value
    config = RtlolaBenchmarkConfig(**params)

    if args.reference_only:
        prepare_reference_cache(config)
        return

    result = run_benchmark(config)
    if args.learned_mode == "regret":
        scenario = scenario_by_name(config.scenario)
        learned = train_and_evaluate_regret(
            config,
            show_progress=not args.no_progress,
            reference_cache_dir=(
                args.reference_cache.parent
                if args.reference_cache is not None
                else args.output / "references"
            ),
        )
        learned_summary = summarize_results(learned.eval_results)
        learned_timeseries = results_to_dataframe(learned.eval_results)
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
                "train_trace_kinds": (
                    config.regret_train_trace_kinds or [config.trace_kind]
                ),
                "eval_trace_kinds": (
                    config.regret_eval_trace_kinds or [config.trace_kind]
                ),
                "budgets": config.regret_budgets or [config.budget],
                "horizon": config.horizon,
                "mpc_objective": config.mpc_objective,
                "train_seed_start": config.regret_train_seed_start,
                "eval_seed_start": config.regret_eval_seed_start,
                "binding_revision": BINDING_REVISION,
                "interpreter_revision": INTERPRETER_REVISION,
                "binding_build_profile": BINDING_BUILD_PROFILE,
                "spec_sha256": hashlib.sha256(
                    scenario.spec.encode("utf-8"),
                ).hexdigest(),
            },
        )

    save_benchmark_results(result, args.output)
    print(f"RTLola benchmark complete: {args.output}")
if __name__ == "__main__":
    main()
