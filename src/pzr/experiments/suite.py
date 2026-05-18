"""One-command orchestration for paper experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import tarfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from pzr.experiments.benchmark import (
    BenchmarkConfig,
    BenchmarkReport,
    KEY_METRICS,
    combine_reports,
    default_methods,
    learned_distilled_method,
    run_benchmark,
)
from pzr.experiments.cli import _selected_methods as benchmark_methods
from pzr.experiments.paper_figures import main as paper_figures_main
from pzr.experiments.scenarios import SCENARIOS
from pzr.learning.distill_cli import train_policy


@dataclass(frozen=True)
class SuiteProfile:
    """Configuration knobs for an experiment-suite profile."""

    length: int
    budget: int
    horizon: int
    seeds: int
    method_set: str
    predictor_mode: str
    bootstrap_samples: int
    figure_seeds: int
    figure_length: int
    figure_budgets: str
    figure_fig4_length: int
    figure_fpr_length: int
    distill_epochs: int
    distill_batch_size: int


PROFILES = {
    "smoke": SuiteProfile(
        length=8,
        budget=8,
        horizon=2,
        seeds=1,
        method_set="paper_plus_wide",
        predictor_mode="both",
        bootstrap_samples=10,
        figure_seeds=1,
        figure_length=8,
        figure_budgets="8,10",
        figure_fig4_length=4,
        figure_fpr_length=8,
        distill_epochs=5,
        distill_batch_size=4,
    ),
    "standard": SuiteProfile(
        length=80,
        budget=8,
        horizon=4,
        seeds=5,
        method_set="paper_plus_wide",
        predictor_mode="both",
        bootstrap_samples=200,
        figure_seeds=3,
        figure_length=80,
        figure_budgets="6,8,10,12",
        figure_fig4_length=20,
        figure_fpr_length=200,
        distill_epochs=50,
        distill_batch_size=32,
    ),
    "paper": SuiteProfile(
        length=200,
        budget=8,
        horizon=4,
        seeds=30,
        method_set="paper_plus_wide",
        predictor_mode="both",
        bootstrap_samples=1000,
        figure_seeds=10,
        figure_length=200,
        figure_budgets="6,8,10,12,16,20",
        figure_fig4_length=20,
        figure_fpr_length=1000,
        distill_epochs=200,
        distill_batch_size=64,
    ),
}

SCENARIO_NAMES = ("robot", "robot_simple", "thermostat")


def main(argv: Sequence[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    run_suite(args)
    return 0


def run_suite(args: argparse.Namespace) -> Path:
    """Run the configured experiment suite and return the output directory."""

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required because the experiment suite always includes "
            "learned policy distillation. Install the learning extra with "
            "`python -m pip install -e .[learning]`."
        ) from exc

    profile = _profile_from_args(args)
    out_dir = _resolve_out_dir(args.out)
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"output directory already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "suite": "pzr_experiment_suite",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "profile_config": asdict(profile),
        "scenarios": list(SCENARIO_NAMES),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "steps": [],
    }

    baseline_reports: dict[str, BenchmarkReport] = {}
    for scenario_name in SCENARIO_NAMES:
        report = _run_scenario_benchmark(
            scenario_name,
            profile,
            learned_policy=None,
            out_dir=out_dir / "runs" / scenario_name / "baseline",
        )
        baseline_reports[scenario_name] = report
        manifest["steps"].append(
            {
                "kind": "baseline_benchmark",
                "scenario": scenario_name,
                "out": _relative(out_dir, out_dir / "runs" / scenario_name / "baseline"),
            }
        )

    learning_dir = out_dir / "learning"
    learning_dir.mkdir(parents=True, exist_ok=True)
    training_data = learning_dir / "training_decision_features.csv"
    pd.concat(
        [report.decision_features for report in baseline_reports.values()],
        ignore_index=True,
    ).to_csv(training_data, index=False)
    checkpoint = learning_dir / "learned_distilled.pt"
    _train_learned_policy(training_data, checkpoint, profile)
    manifest["steps"].append(
        {
            "kind": "learned_policy_distillation",
            "data": _relative(out_dir, training_data),
            "checkpoint": _relative(out_dir, checkpoint),
        }
    )

    learned_reports: dict[str, BenchmarkReport] = {}
    for scenario_name in SCENARIO_NAMES:
        report = _run_scenario_benchmark(
            scenario_name,
            profile,
            learned_policy=checkpoint,
            out_dir=out_dir / "runs" / scenario_name / "learned",
        )
        learned_reports[scenario_name] = report
        manifest["steps"].append(
            {
                "kind": "learned_evaluation",
                "scenario": scenario_name,
                "out": _relative(out_dir, out_dir / "runs" / scenario_name / "learned"),
            }
        )

    figures_dir = out_dir / "figures"
    paper_figures_main(
        [
            "--out",
            str(figures_dir),
            "--method-set",
            profile.method_set,
            "--seeds",
            str(profile.figure_seeds),
            "--length",
            str(profile.figure_length),
            "--budget",
            str(profile.budget),
            "--budgets",
            profile.figure_budgets,
            "--horizon",
            str(profile.horizon),
            "--fig4-length",
            str(profile.figure_fig4_length),
            "--fig4-seed",
            "0",
            "--fpr-length",
            str(profile.figure_fpr_length),
            "--formats",
            args.formats,
            "--bootstrap-samples",
            str(profile.bootstrap_samples),
            "--bootstrap-seed",
            str(args.bootstrap_seed),
            "--learned-policy",
            str(checkpoint),
        ]
    )
    manifest["steps"].append(
        {
            "kind": "paper_figures",
            "out": _relative(out_dir, figures_dir),
            "learned_policy": _relative(out_dir, checkpoint),
        }
    )

    _write_aggregate_outputs(out_dir, baseline_reports, learned_reports)
    manifest["steps"].append({"kind": "aggregate_outputs", "out": "aggregate"})

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    index_path = out_dir / "artifact_index.csv"
    _write_artifact_index(out_dir, index_path)

    if not args.no_archive:
        archive_path = out_dir.with_suffix(".tar.gz")
        _write_archive(out_dir, archive_path)
    return out_dir


def _run_scenario_benchmark(
    scenario_name: str,
    profile: SuiteProfile,
    *,
    learned_policy: Path | None,
    out_dir: Path,
) -> BenchmarkReport:
    scenario = SCENARIOS[scenario_name]()
    config = BenchmarkConfig(
        length=profile.length,
        budget=profile.budget,
        horizon=profile.horizon,
        seeds=tuple(range(profile.seeds)),
        predictor_mode=profile.predictor_mode,  # type: ignore[arg-type]
        include_reference=True,
        bootstrap_samples=profile.bootstrap_samples,
        bootstrap_seed=0,
    )
    methods = benchmark_methods(profile.method_set)
    if learned_policy is not None:
        learned = learned_distilled_method(learned_policy)
        methods = (*default_methods(), learned) if methods is None else (*methods, learned)
    if profile.predictor_mode == "both":
        reports = tuple(
            run_benchmark(
                scenario,
                replace(config, predictor_mode=mode),
                methods=methods,
            )
            for mode in ("online", "oracle")
        )
        report = combine_reports(config, reports)
    else:
        report = run_benchmark(scenario, config, methods=methods)
    report.write_artifacts(out_dir)
    return report


def _train_learned_policy(
    training_data: Path,
    checkpoint: Path,
    profile: SuiteProfile,
) -> None:
    train_policy(
        argparse.Namespace(
            data=training_data,
            expert_method="mpc_rollout_wide",
            predictor_mode="online",
            out=checkpoint,
            seed=0,
            epochs=profile.distill_epochs,
            batch_size=profile.distill_batch_size,
            lr=1e-3,
            weight_decay=0.0,
            validation_fraction=0.2,
            hidden_sizes=(64, 64),
        )
    )


def _write_aggregate_outputs(
    out_dir: Path,
    baseline_reports: dict[str, BenchmarkReport],
    learned_reports: dict[str, BenchmarkReport],
) -> None:
    aggregate_dir = out_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    reports = _aggregate_reports(baseline_reports, learned_reports)
    _concat_attr(reports, "raw_runs").to_csv(aggregate_dir / "raw_runs.csv", index=False)
    _concat_attr(reports, "summary").to_csv(aggregate_dir / "summary.csv", index=False)
    _concat_attr(reports, "timeseries").to_csv(aggregate_dir / "timeseries.csv", index=False)
    _concat_attr(reports, "bounds_timeseries").to_csv(
        aggregate_dir / "bounds_timeseries.csv",
        index=False,
    )
    _concat_attr(reports, "decision_features").to_csv(
        aggregate_dir / "decision_features.csv",
        index=False,
    )
    _concat_attr(reports, "selection_summary").to_csv(
        aggregate_dir / "selection_summary.csv",
        index=False,
    )
    _concat_attr(reports, "predicted_sequence_summary").to_csv(
        aggregate_dir / "predicted_sequence_summary.csv",
        index=False,
    )
    _write_analysis_notes(aggregate_dir)


def _concat_attr(reports: Sequence[BenchmarkReport], attr: str) -> pd.DataFrame:
    frames = [getattr(report, attr) for report in reports]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _aggregate_reports(
    baseline_reports: dict[str, BenchmarkReport],
    learned_reports: dict[str, BenchmarkReport],
) -> list[BenchmarkReport]:
    reports = list(baseline_reports.values())
    reports.extend(_learned_only_report(report) for report in learned_reports.values())
    return reports


def _learned_only_report(report: BenchmarkReport) -> BenchmarkReport:
    return BenchmarkReport(
        report.config,
        _filter_learned(report.raw_runs),
        _filter_learned(report.summary),
        _filter_learned(report.comparisons),
        _filter_learned(report.predictor_comparisons),
        _filter_learned(report.timeseries),
        _filter_learned(report.bounds_timeseries),
        _filter_learned(report.decision_features),
        _filter_learned(report.selection_summary),
        _filter_learned(report.predicted_sequence_summary),
    )


def _filter_learned(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "method" not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame[frame["method"] == "learned_distilled"].copy()


def _write_analysis_notes(aggregate_dir: Path) -> None:
    raw = pd.read_csv(aggregate_dir / "raw_runs.csv")
    summary = pd.read_csv(aggregate_dir / "summary.csv")
    predicted_path = aggregate_dir / "predicted_sequence_summary.csv"
    predicted = pd.read_csv(predicted_path) if predicted_path.exists() else pd.DataFrame()
    notes = {
        "top_winners": _top_winners(summary),
        "soundness_checks": _soundness_checks(raw),
        "warning_flags": _warning_flags(raw, predicted),
    }
    (aggregate_dir / "analysis_notes.json").write_text(
        json.dumps(_json_safe(notes), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _top_winners(summary: pd.DataFrame) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    metrics = [metric for metric in KEY_METRICS if metric in set(summary["metric"])]
    rows: list[dict[str, Any]] = []
    for (scenario, predictor_mode, metric), group in summary[
        summary["metric"].isin(metrics)
    ].groupby(["scenario", "predictor_mode", "metric"], sort=True):
        ranked = group.sort_values(["mean", "method"], ascending=[True, True])
        winner = ranked.iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "predictor_mode": predictor_mode,
                "metric": metric,
                "method": winner["method"],
                "mean": float(winner["mean"]),
            }
        )
    return rows


def _soundness_checks(raw: pd.DataFrame) -> dict[str, Any]:
    budgeted = raw[raw["method"] != "reference"] if "method" in raw else raw
    return {
        "row_count": int(raw.shape[0]),
        "budgeted_row_count": int(budgeted.shape[0]),
        "budget_violation_count": _sum_column(budgeted, "budget_violation_count"),
        "unsound_certificate_count": int(
            _sum_column(budgeted, "unsound_certificate_count")
        ),
        "reduction_failure_count": _sum_column(budgeted, "reduction_failure_count"),
        "no_op_count": _sum_column(budgeted, "no_op_count"),
    }


def _warning_flags(raw: pd.DataFrame, predicted: pd.DataFrame) -> list[str]:
    flags: list[str] = []
    checks = _soundness_checks(raw)
    for key in (
        "budget_violation_count",
        "unsound_certificate_count",
        "reduction_failure_count",
        "no_op_count",
    ):
        if checks[key]:
            flags.append(f"{key}={checks[key]}")
    if not predicted.empty:
        wide = predicted[predicted["method"] == "mpc_rollout_wide"]
        first_box = _sum_column(wide, "first_action_box_count")
        if first_box:
            flags.append(f"mpc_rollout_wide_first_action_box_count={first_box}")
    return flags


def _sum_column(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].fillna(0).sum())


def _write_artifact_index(out_dir: Path, index_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in out_dir.rglob("*") if item.is_file()):
        if path == index_path:
            continue
        rows.append(
            {
                "path": _relative(out_dir, path),
                "kind": _artifact_kind(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    pd.DataFrame(rows, columns=("path", "kind", "bytes", "sha256")).to_csv(
        index_path,
        index=False,
    )


def _write_archive(out_dir: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)


def _artifact_kind(path: Path) -> str:
    if path.suffix == ".csv":
        return "csv"
    if path.suffix == ".json":
        return "json"
    if path.suffix == ".png":
        return "figure_png"
    if path.suffix == ".pdf":
        return "figure_pdf"
    if path.suffix == ".pt":
        return "checkpoint"
    return path.suffix.lstrip(".") or "file"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_from_args(args: argparse.Namespace) -> SuiteProfile:
    profile = PROFILES[args.profile]
    updates = {
        "length": args.length,
        "budget": args.budget,
        "horizon": args.horizon,
        "seeds": args.seeds,
        "method_set": args.method_set,
        "bootstrap_samples": args.bootstrap_samples,
        "figure_seeds": args.figure_seeds,
        "figure_length": args.figure_length,
        "figure_budgets": args.figure_budgets,
        "figure_fig4_length": args.figure_fig4_length,
        "figure_fpr_length": args.figure_fpr_length,
        "distill_epochs": args.distill_epochs,
    }
    clean = {key: value for key, value in updates.items() if value is not None}
    return replace(profile, **clean)


def _resolve_out_dir(value: str | None) -> Path:
    if value:
        return Path(value)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("results") / f"experiment-suite-{timestamp}"


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pzr-run-experiments",
        description="Run the full paper experiment suite and package artifacts.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="paper",
        help="Preset experiment scale.",
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--formats", type=str, default="png,pdf")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument(
        "--method-set",
        choices=("paper", "paper_plus_ours", "paper_plus_wide", "extended"),
        default=None,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--figure-seeds", type=int, default=None)
    parser.add_argument("--figure-length", type=int, default=None)
    parser.add_argument("--figure-budgets", type=str, default=None)
    parser.add_argument("--figure-fig4-length", type=int, default=None)
    parser.add_argument("--figure-fpr-length", type=int, default=None)
    parser.add_argument("--distill-epochs", type=int, default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
