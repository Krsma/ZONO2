"""Command-line entry point for paper-style benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

from pzr.experiments.benchmark import (
    BenchmarkConfig,
    combine_reports,
    default_methods,
    format_terminal_summary,
    learned_distilled_method,
    paper_baseline_methods,
    run_benchmark,
)
from pzr.experiments.scenarios import SCENARIOS


def main(argv: Sequence[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    scenario_factory = SCENARIOS[args.scenario]
    seeds = tuple(range(args.seed_start, args.seed_start + args.seeds))
    config = BenchmarkConfig(
        length=args.length,
        budget=args.budget,
        horizon=args.horizon,
        seeds=seeds,
        predictor_mode=args.predictor_mode,
        include_reference=not args.no_reference,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    out_dir = Path(args.out) if args.out else _default_out_dir(args.scenario)
    scenario = scenario_factory()
    methods = _selected_methods(args.method_set)
    if args.learned_policy:
        learned = learned_distilled_method(args.learned_policy)
        methods = (*default_methods(), learned) if methods is None else (*methods, learned)
    if args.predictor_mode == "both":
        reports = tuple(
            run_benchmark(scenario, replace(config, predictor_mode=mode), methods=methods)
            for mode in ("online", "oracle")
        )
        report = combine_reports(config, reports)
    else:
        report = run_benchmark(scenario, config, methods=methods)
    report.write_artifacts(out_dir)
    if not args.quiet:
        print(format_terminal_summary(report))
        print(f"\nWrote benchmark artifacts to {out_dir}")
    return 0


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pzr-benchmark",
        description="Run paper-style benchmarks for predictive zonotope reduction.",
    )
    subparsers = parser.add_subparsers(dest="scenario", required=True)
    for scenario_name in sorted(SCENARIOS):
        scenario = subparsers.add_parser(scenario_name)
        scenario.add_argument("--length", type=int, default=200)
        scenario.add_argument("--budget", type=int, default=8)
        scenario.add_argument("--horizon", type=int, default=4)
        scenario.add_argument("--seeds", type=int, default=30)
        scenario.add_argument("--seed-start", type=int, default=0)
        scenario.add_argument("--out", type=str, default=None)
        scenario.add_argument(
            "--predictor-mode",
            choices=("online", "oracle", "both"),
            default="online",
        )
        scenario.add_argument("--bootstrap-samples", type=int, default=1000)
        scenario.add_argument("--bootstrap-seed", type=int, default=0)
        scenario.add_argument(
            "--method-set",
            choices=(
                "extended",
                "paper",
                "paper_plus_focused",
                "paper_plus_mpc_ablation",
                "paper_plus_ours",
                "paper_plus_wide",
            ),
            default="extended",
        )
        scenario.add_argument("--no-reference", action="store_true")
        scenario.add_argument("--learned-policy", type=str, default=None)
        scenario.add_argument("--quiet", action="store_true")
    return parser


def _default_out_dir(scenario: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("results") / f"{scenario}-{timestamp}"


def _selected_methods(method_set: str):
    if method_set == "paper_plus_ours":
        method_set = "paper_plus_focused"
    elif method_set == "paper_plus_wide":
        method_set = "paper_plus_mpc_ablation"
    if method_set == "extended":
        return None
    if method_set == "paper":
        return paper_baseline_methods()
    if method_set == "paper_plus_focused":
        ours = tuple(
            method
            for method in default_methods()
            if method.name in {"mpc_focused_fixed_girard", "mpc_focused_sequence"}
        )
        return (*paper_baseline_methods(), *ours)
    if method_set == "paper_plus_mpc_ablation":
        ours = tuple(
            method
            for method in default_methods()
            if method.name
            in {
                "mpc_focused_fixed_girard",
                "mpc_wide_fixed_girard",
                "mpc_focused_sequence",
            }
        )
        return (*paper_baseline_methods(), *ours)
    raise ValueError(f"unknown method set: {method_set}")


if __name__ == "__main__":
    raise SystemExit(main())
