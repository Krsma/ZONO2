"""CLI for RTLola-native PZR benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from pzr.rtlola.benchmark import (
    METHOD_SET_CHOICES,
    RtlolaBenchmarkConfig,
    prepare_reference_cache,
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


def _parse_csv(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("comma-separated value must not be empty")
    return values


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run RTLola-native PZR benchmark")
    parser.add_argument("--profile", choices=PROFILE_DEFAULTS, default="smoke")
    parser.add_argument("--scenario", default="omni_robot")
    parser.add_argument(
        "--trace-kind",
        default="default",
        help=(
            "RTLola trace kind; robot_arm supports figure8, figure8_drift, "
            "random, random_drift, square, square_drift"
        ),
    )
    parser.add_argument(
        "--method-set",
        choices=METHOD_SET_CHOICES,
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
            "exact caches logical-row unreduced references for approximation loss "
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
    save_benchmark_results(result, args.output)
    print(f"RTLola benchmark complete: {args.output}")
if __name__ == "__main__":
    main()
