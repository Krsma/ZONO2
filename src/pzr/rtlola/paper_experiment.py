"""Versioned contracts and statistics for the paper evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from pzr.learning.provenance import payload_sha256, pzr_source_sha256, sha256_files
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.reference import REFERENCE_CACHE_SCHEMA
from pzr.rtlola.robot_arm import RLOLAEVAL_REVISION, ROBOT_ARM_SPEC_SHA256


PAPER_CONFIG_SCHEMA = "pzr.paper-evaluation-config.v1"
PAPER_CELL_SCHEMA = "pzr.paper-evaluation-cell.v1"
PAPER_STAGE_SCHEMA = "pzr.paper-evaluation-stage.v1"
PAPER_RUN_SCHEMA = "pzr.paper-evaluation-run.v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260721
ORDINARY_REDUCERS = ("girard", "scott", "pca", "combastel")
HEADLINE_METHODS = (
    *ORDINARY_REDUCERS,
    "mpc_terminal_beam",
    "mpc_terminal_beam_predictive_linear",
    "mpc_terminal_full_width",
    "pairwise_ranking_policy",
)
PILOT_METHODS = (*HEADLINE_METHODS, "pairwise_ranking_policy_budget80")
GENERALIZATION_METHODS = PILOT_METHODS
OBJECTIVE_METHODS = ("mpc_terminal_beam", "mpc_cumulative_beam")
COMPOSITION_METHODS = (
    "mpc_terminal_beam",
    "pairwise_ranking_policy",
    "mpc_terminal_beam_predictive_linear",
)
STAGES = (
    "prepare",
    "train",
    "pilot",
    "objective-comparison",
    "headline",
    "generalization",
    "ablation",
    "timing",
    "report",
    "validate",
)


class ExecutionRegime(str, Enum):
    STATIC_ONLINE = "static_online"
    OFFLINE_RECORDED = "offline_recorded"
    ONLINE_PREDICTIVE = "online_predictive"
    OFFLINE_TEACHER = "offline_teacher"
    OFFLINE_COMPARISON = "offline_comparison"
    LEARNED_ONLINE = "learned_online"


class Predictor(str, Enum):
    NONE = "none"
    RECORDED_FUTURE = "recorded_future"
    CAUSAL_LINEAR = "causal_linear"


class Objective(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    TERMINAL = "terminal"
    CUMULATIVE = "cumulative"
    LEARNED_TERMINAL_TEACHER = "learned_terminal_teacher"
    LEARNED_TERMINAL_TEACHER_BUDGET80 = "learned_terminal_teacher_budget80"


class RunState(str, Enum):
    COMPLETED = "completed"
    FALLBACK_FAILED = "fallback_failed"
    NATIVE_FAILED = "native_failed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"


@dataclass(frozen=True)
class MethodConfig:
    name: str
    execution_regime: ExecutionRegime
    predictor: Predictor
    horizon: int
    beam_width: int
    objective: Objective
    candidate_names: tuple[str, ...]
    selection_reference: str = "native_unreduced_rollout"
    metric_reference: str = "exact_cache_dynamic_and_total_radius"

    def __post_init__(self) -> None:
        if self.horizon < 0 or self.beam_width < 0:
            raise ValueError("method horizon and beam width must be non-negative")
        if self.execution_regime in {
            ExecutionRegime.OFFLINE_RECORDED,
            ExecutionRegime.OFFLINE_TEACHER,
            ExecutionRegime.OFFLINE_COMPARISON,
        } and self.predictor is not Predictor.RECORDED_FUTURE:
            raise ValueError(f"offline method {self.name} must use recorded future inputs")
        if self.execution_regime is ExecutionRegime.ONLINE_PREDICTIVE:
            if self.predictor is not Predictor.CAUSAL_LINEAR:
                raise ValueError(f"online predictive method {self.name} must be causal linear")
        if self.execution_regime is ExecutionRegime.LEARNED_ONLINE and self.horizon != 0:
            raise ValueError(f"learned policy {self.name} must be pre-event and horizon-free")
        if self.objective is Objective.CUMULATIVE and self.name != "mpc_cumulative_beam":
            raise ValueError("cumulative objective is reserved for mpc_cumulative_beam")


@dataclass(frozen=True)
class PaperExperimentConfig:
    source: Path
    schema: str
    experiment_id: str
    output_root: Path
    paper_artifact_dir: Path
    event_count: int
    budgets: tuple[int, ...]
    candidate_names: tuple[str, ...]
    conditions: tuple[str, ...]
    figure8_conditions: tuple[str, ...]
    teacher_workers: int
    evaluation_workers: int
    ablation_workers: int
    training_epochs: int
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    reserved_exploration_seeds: tuple[int, ...]
    pilot_seeds: tuple[int, ...]
    pilot_budgets: tuple[int, ...]
    maximum_projected_wall_hours: float
    generalization_seeds: tuple[int, ...]
    ablation_seeds: tuple[int, ...]
    ablation_budget: int
    ablation_horizons: tuple[int, ...]
    ablation_widths: tuple[int, ...]
    timing_warmup_events: int
    timing_repetitions: int
    timing_workers: int
    timing_native_threads: int
    methods: tuple[MethodConfig, ...]
    config_sha256: str
    enforce_canonical_scope: bool = True

    def __post_init__(self) -> None:
        if self.schema != PAPER_CONFIG_SCHEMA:
            raise ValueError(f"unsupported paper config schema: {self.schema}")
        if self.event_count < 2 or not self.budgets:
            raise ValueError("paper config needs at least two events and one budget")
        _require_unique("budgets", self.budgets)
        _require_unique("candidate names", self.candidate_names)
        _require_unique("conditions", self.conditions)
        _require_unique("figure8 conditions", self.figure8_conditions)
        _require_unique("method names", tuple(method.name for method in self.methods))
        seed_groups = {
            "training": set(self.train_seeds),
            "validation": set(self.validation_seeds),
            "exploration": set(self.reserved_exploration_seeds),
            "pilot": set(self.pilot_seeds),
            "generalization": set(self.generalization_seeds),
            "ablation": set(self.ablation_seeds),
        }
        for left, left_values in seed_groups.items():
            for right, right_values in seed_groups.items():
                if left < right and left_values & right_values:
                    raise ValueError(f"paper seed groups overlap: {left} and {right}")
        if self.enforce_canonical_scope and tuple(self.budgets) != (
            40, 80, 120, 150, 200, 250, 500,
        ):
            raise ValueError("canonical paper budgets differ")
        expected_methods = {*HEADLINE_METHODS, *PILOT_METHODS, *OBJECTIVE_METHODS}
        if set(self.method_by_name) != expected_methods:
            raise ValueError("paper method catalog differs from the stable identities")
        if self.timing_workers != 1 or self.timing_native_threads != 1:
            raise ValueError("paper timing must be sequential with one native thread")
        if self.ablation_workers != 1:
            raise ValueError("paper ablation timing must use one experiment worker")
        if self.training_epochs < 1:
            raise ValueError("training epochs must be positive")
        if self.enforce_canonical_scope:
            expected_counts = {
                "pilot": 216,
                "generalization": 5_040,
                "headline": 224,
                "objective-comparison": 56,
                "ablation": 320,
            }
            actual_counts = {stage: self.expected_cells(stage) for stage in expected_counts}
            if actual_counts != expected_counts:
                raise ValueError(
                    f"canonical paper cell counts differ: {actual_counts} != {expected_counts}"
                )

    @property
    def method_by_name(self) -> dict[str, MethodConfig]:
        return {method.name: method for method in self.methods}

    def expected_cells(self, stage: str) -> int:
        if stage == "pilot":
            return len(self.pilot_seeds) * len(self.conditions) * len(
                self.pilot_budgets,
            ) * len(PILOT_METHODS)
        if stage == "generalization":
            return len(self.generalization_seeds) * len(self.conditions) * len(
                self.budgets,
            ) * len(GENERALIZATION_METHODS)
        if stage == "headline":
            return len(self.figure8_conditions) * len(self.budgets) * len(
                HEADLINE_METHODS,
            )
        if stage == "objective-comparison":
            return len(self.figure8_conditions) * len(self.budgets) * len(
                OBJECTIVE_METHODS,
            )
        if stage == "ablation":
            return (
                len(self.ablation_seeds)
                * len(self.conditions)
                * len(self.ablation_horizons)
                * len(self.ablation_widths)
            )
        raise ValueError(f"stage has no scientific cell matrix: {stage}")


def load_paper_experiment_config(path: Path) -> PaperExperimentConfig:
    """Load the checked-in YAML and reject incomplete or reinterpreted schemas."""
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes)
    if not isinstance(raw, dict):
        raise ValueError("paper experiment config must be a mapping")
    methods = tuple(
        MethodConfig(
            name=str(name),
            execution_regime=ExecutionRegime(str(values["execution_regime"])),
            predictor=Predictor(str(values["predictor"])),
            horizon=int(values["horizon"]),
            beam_width=int(values["beam_width"]),
            objective=Objective(str(values["objective"])),
            candidate_names=tuple(str(value) for value in raw["candidate_names"]),
        )
        for name, values in raw["methods"].items()
    )
    return PaperExperimentConfig(
        source=path.resolve(),
        schema=str(raw["schema"]),
        experiment_id=str(raw["experiment_id"]),
        output_root=Path(raw["output_root"]),
        paper_artifact_dir=Path(raw["paper_artifact_dir"]),
        event_count=int(raw["event_count"]),
        budgets=tuple(int(value) for value in raw["budgets"]),
        candidate_names=tuple(str(value) for value in raw["candidate_names"]),
        conditions=tuple(str(value) for value in raw["conditions"]),
        figure8_conditions=tuple(str(value) for value in raw["figure8_conditions"]),
        teacher_workers=int(raw["workers"]["teacher"]),
        evaluation_workers=int(raw["workers"]["evaluation"]),
        ablation_workers=int(raw["workers"]["ablation"]),
        training_epochs=int(raw["training"]["epochs"]),
        train_seeds=tuple(int(value) for value in raw["training"]["train_seeds"]),
        validation_seeds=tuple(
            int(value) for value in raw["training"]["validation_seeds"]
        ),
        reserved_exploration_seeds=tuple(
            int(value) for value in raw["reserved_exploration_seeds"]
        ),
        pilot_seeds=tuple(int(value) for value in raw["pilot"]["seeds"]),
        pilot_budgets=tuple(int(value) for value in raw["pilot"]["budgets"]),
        maximum_projected_wall_hours=float(
            raw["pilot"]["maximum_projected_wall_hours"]
        ),
        generalization_seeds=tuple(int(value) for value in raw["generalization_seeds"]),
        ablation_seeds=tuple(int(value) for value in raw["ablation"]["seeds"]),
        ablation_budget=int(raw["ablation"]["budget"]),
        ablation_horizons=tuple(int(value) for value in raw["ablation"]["horizons"]),
        ablation_widths=tuple(int(value) for value in raw["ablation"]["widths"]),
        timing_warmup_events=int(raw["timing"]["warmup_events"]),
        timing_repetitions=int(raw["timing"]["repetitions"]),
        timing_workers=int(raw["timing"]["workers"]),
        timing_native_threads=int(raw["timing"]["native_threads"]),
        methods=methods,
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def method_identity(method: MethodConfig) -> dict[str, object]:
    """Serialize every method choice that can alter a scientific result."""
    payload = asdict(method)
    payload["execution_regime"] = method.execution_regime.value
    payload["predictor"] = method.predictor.value
    payload["objective"] = method.objective.value
    payload["candidate_names"] = list(method.candidate_names)
    return payload


def cell_identity(
    config: PaperExperimentConfig,
    *,
    stage: str,
    trace_id: str,
    trace_sha256: str,
    condition: str,
    seed: int,
    event_count: int,
    budget: int,
    method: MethodConfig,
    reference_path: Path,
    model_sha256: str | None,
    source_sha256: str | None = None,
) -> dict[str, object]:
    """Build a complete source-aware identity for a resumable paper cell."""
    if stage not in STAGES:
        raise ValueError(f"unknown paper stage: {stage}")
    payload: dict[str, object] = {
        "schema": PAPER_CELL_SCHEMA,
        "experiment_id": config.experiment_id,
        "stage": stage,
        "trace_id": trace_id,
        "trace_sha256": trace_sha256,
        "condition": condition,
        "seed": seed,
        "event_count": event_count,
        "budget": budget,
        "method": method_identity(method),
        "spec_sha256": ROBOT_ARM_SPEC_SHA256,
        "rlolaeval_revision": RLOLAEVAL_REVISION,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "reference_cache_schema": REFERENCE_CACHE_SCHEMA,
        "reference_cache_sha256": sha256_files((reference_path,)),
        "reference_semantics": {
            "selection": method.selection_reference,
            "metrics": method.metric_reference,
        },
        "model_sha256": model_sha256,
        "seed_lists": {
            "train": list(config.train_seeds),
            "validation": list(config.validation_seeds),
            "exploration": list(config.reserved_exploration_seeds),
            "pilot": list(config.pilot_seeds),
            "generalization": list(config.generalization_seeds),
            "ablation": list(config.ablation_seeds),
        },
        "conditions": list(config.conditions),
        "config_sha256": config.config_sha256,
        "pzr_source_sha256": source_sha256 or pzr_source_sha256(),
    }
    return {**payload, "fingerprint": payload_sha256(payload)}


def validate_cell_manifest(
    manifest: Mapping[str, object],
    expected_identity: Mapping[str, object],
) -> None:
    """Reject old cell schemas and any source/config mismatch."""
    if manifest.get("schema") != PAPER_CELL_SCHEMA:
        raise ValueError("unsupported paper-evaluation cell schema")
    identity = manifest.get("identity")
    if identity != dict(expected_identity):
        raise ValueError("stale paper-evaluation cell identity")
    try:
        RunState(str(manifest["status"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("invalid paper-evaluation run state") from exc


def trace_level_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct trace-level rates and reject inconsistent denominators."""
    required = {
        "condition", "seed", "budget", "method", "status",
        "false_positive_count", "false_negative_count",
        "reference_negative_count", "reference_positive_count",
        "mean_approx_loss", "total_time_ms", "event_count",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"paper summary lacks columns: {sorted(missing)}")
    data = summary.copy()
    completed = data["status"] == RunState.COMPLETED.value
    negative = data["reference_negative_count"].astype(float)
    positive = data["reference_positive_count"].astype(float)
    if bool((negative < 0).any() or (positive < 0).any()):
        raise ValueError("trigger denominators must be non-negative")
    if bool((data["false_positive_count"].astype(float) > negative).any()):
        raise ValueError("false-positive count exceeds its exact-negative denominator")
    if bool((data["false_negative_count"].astype(float) > positive).any()):
        raise ValueError("false-negative count exceeds its exact-positive denominator")
    data["fpr"] = np.where(
        completed & (negative > 0),
        data["false_positive_count"].astype(float) / negative,
        np.nan,
    )
    data["fnr"] = np.where(
        completed & (positive > 0),
        data["false_negative_count"].astype(float) / positive,
        np.nan,
    )
    runtime_column = (
        "event_loop_time_ms" if "event_loop_time_ms" in data else "total_time_ms"
    )
    runtime_ms = data[runtime_column].astype(float)
    data["throughput_events_per_second"] = np.where(
        completed & (runtime_ms > 0),
        data["event_count"].astype(float) * 1000.0
        / runtime_ms,
        np.nan,
    )
    return data


def aggregate_trace_metrics(
    summary: pd.DataFrame,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Compute failure-aware macro/pooled summaries and paired bootstrap CIs."""
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap replicate count must be positive")
    data = trace_level_metrics(summary)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(bootstrap_seed)
    group_keys = ["condition", "budget"]
    for (condition, budget), point in data.groupby(group_keys, sort=True):
        methods = sorted(point["method"].astype(str).unique())
        seed_sets = {
            method: tuple(sorted(
                point.loc[point["method"] == method, "seed"].astype(int)
            ))
            for method in methods
        }
        aligned = len(set(seed_sets.values())) == 1
        common_seeds = seed_sets[methods[0]] if aligned and methods else ()
        bootstrap_indices = (
            rng.integers(0, len(common_seeds), size=(bootstrap_replicates, len(common_seeds)))
            if common_seeds else np.empty((bootstrap_replicates, 0), dtype=np.int64)
        )
        for method in methods:
            frame = point[point["method"] == method].sort_values("seed")
            valid = frame[frame["status"] == RunState.COMPLETED.value]
            failed_count = len(frame) - len(valid)
            fpr_values = valid["fpr"].dropna().to_numpy(dtype=np.float64)
            losses = valid["mean_approx_loss"].dropna().to_numpy(dtype=np.float64)
            throughput = valid["throughput_events_per_second"].dropna().to_numpy(
                dtype=np.float64,
            )
            macro_fpr = float(np.mean(fpr_values)) if len(fpr_values) else float("nan")
            ci_low = ci_high = float("nan")
            if failed_count == 0 and aligned and len(common_seeds) and len(fpr_values) == len(frame):
                draws = fpr_values[bootstrap_indices].mean(axis=1)
                ci_low, ci_high = np.quantile(draws, (0.025, 0.975)).tolist()
            fp_count = float(valid["false_positive_count"].sum())
            negative_count = float(valid["reference_negative_count"].sum())
            rows.append({
                "condition": condition,
                "budget": int(budget),
                "method": method,
                "available": failed_count == 0,
                "valid_count": len(valid),
                "failed_count": failed_count,
                "fallback_rate": float(
                    (frame["status"] == RunState.FALLBACK_FAILED.value).mean()
                ),
                "macro_fpr": macro_fpr if failed_count == 0 else float("nan"),
                "macro_fpr_ci_low": ci_low if failed_count == 0 else float("nan"),
                "macro_fpr_ci_high": ci_high if failed_count == 0 else float("nan"),
                "valid_only_macro_fpr": macro_fpr,
                "pooled_fpr": (
                    fp_count / negative_count
                    if failed_count == 0 and negative_count > 0 else float("nan")
                ),
                "valid_only_pooled_fpr": (
                    fp_count / negative_count if negative_count > 0 else float("nan")
                ),
                "median_fpr": _median(fpr_values, failed_count == 0),
                "fpr_iqr_low": _quantile(fpr_values, 0.25, failed_count == 0),
                "fpr_iqr_high": _quantile(fpr_values, 0.75, failed_count == 0),
                "macro_mean_approx_loss": _mean(losses, failed_count == 0),
                "valid_only_macro_mean_approx_loss": _mean(losses, True),
                "median_mean_approx_loss": _median(losses, failed_count == 0),
                "loss_iqr_low": _quantile(losses, 0.25, failed_count == 0),
                "loss_iqr_high": _quantile(losses, 0.75, failed_count == 0),
                "median_throughput_events_per_second": _median(
                    throughput, failed_count == 0,
                ),
                "paired_seed_alignment": aligned,
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_seed": bootstrap_seed,
            })
    return pd.DataFrame(rows)


def reducer_composition(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Count ordinary choices, excluding none, fallback, and infeasible events."""
    required = {
        "condition", "budget", "method", "reducer_used", "fallback_used",
        "infeasible_candidate_count",
    }
    missing = required - set(timeseries.columns)
    if missing:
        raise ValueError(f"composition data lacks columns: {sorted(missing)}")
    selected = timeseries[
        timeseries["method"].isin(COMPOSITION_METHODS)
        & timeseries["reducer_used"].isin(ORDINARY_REDUCERS)
        & ~timeseries["fallback_used"].astype(bool)
        & (timeseries["infeasible_candidate_count"].astype(int) == 0)
    ].copy()
    groups = ["condition", "budget", "method", "reducer_used"]
    result = selected.groupby(groups, dropna=False).size().rename("count").reset_index()
    if result.empty:
        return pd.DataFrame(columns=(*groups, "count", "percentage"))
    totals = result.groupby(groups[:-1])["count"].transform("sum")
    result["percentage"] = result["count"] * 100.0 / totals
    return result


def pilot_projection(
    summary: pd.DataFrame,
    *,
    target_cell_count: int,
    worker_count: int,
    disk_bytes: int,
    threshold_hours: float,
) -> dict[str, object]:
    """Project the held-out run from observed pilot cells and expose its gate."""
    if target_cell_count < 1 or worker_count < 1 or disk_bytes < 0:
        raise ValueError("invalid pilot projection inputs")
    data = trace_level_metrics(summary)
    completed = data[data["status"] == RunState.COMPLETED.value]
    if completed.empty:
        raise ValueError("pilot has no completed cells")
    timing_column = (
        "event_loop_time_ms" if "event_loop_time_ms" in completed
        else "total_time_ms"
    )
    seconds = completed[timing_column].astype(float) / 1000.0
    mean_seconds = float(seconds.mean())
    cpu_hours = mean_seconds * target_cell_count / 3600.0
    wall_hours = cpu_hours / worker_count
    per_method = {
        str(method): {
            "completed_cells": len(frame),
            "mean_seconds_per_cell": float(
                frame[timing_column].astype(float).mean() / 1000.0
            ),
        }
        for method, frame in completed.groupby("method")
    }
    return {
        "pilot_cell_count": len(data),
        "completed_pilot_cell_count": len(completed),
        "target_cell_count": target_cell_count,
        "worker_count": worker_count,
        "projected_cpu_hours": cpu_hours,
        "projected_four_worker_wall_hours": wall_hours,
        "projected_disk_bytes": int(round(disk_bytes * target_cell_count / len(data))),
        "per_method_scaling": per_method,
        "maximum_wall_hours": threshold_hours,
        "approval_required": wall_hours > threshold_hours,
    }


def validate_summary_matrix(
    config: PaperExperimentConfig,
    stage: str,
    summary: pd.DataFrame,
) -> None:
    """Validate uniqueness, exact scope, statuses, and trigger denominators."""
    expected = config.expected_cells(stage)
    if len(summary) != expected:
        raise ValueError(f"{stage} has {len(summary)} cells, expected {expected}")
    keys = ["condition", "seed", "budget", "method"]
    if stage == "ablation":
        keys.extend(("horizon", "beam_width"))
    missing = set(keys) - set(summary.columns)
    if missing:
        raise ValueError(f"{stage} summary lacks identity columns: {sorted(missing)}")
    if bool(summary.duplicated(keys).any()):
        raise ValueError(f"{stage} contains duplicate scientific cells")
    actual = set(tuple(row) for row in summary[keys].itertuples(index=False, name=None))
    expected_keys = _expected_matrix_keys(config, stage)
    if actual != expected_keys:
        raise ValueError(f"{stage} scientific cell identities differ from the config")
    states = set(summary["status"].astype(str))
    if not states <= {state.value for state in RunState}:
        raise ValueError(f"{stage} contains an invalid run state")
    trace_level_metrics(summary)


def _expected_matrix_keys(
    config: PaperExperimentConfig,
    stage: str,
) -> set[tuple[object, ...]]:
    if stage == "pilot":
        return {
            (condition, seed, budget, method)
            for seed in config.pilot_seeds
            for condition in config.conditions
            for budget in config.pilot_budgets
            for method in PILOT_METHODS
        }
    if stage == "generalization":
        return {
            (condition, seed, budget, method)
            for seed in config.generalization_seeds
            for condition in config.conditions
            for budget in config.budgets
            for method in GENERALIZATION_METHODS
        }
    if stage == "headline":
        return {
            (condition, 0, budget, method)
            for condition in config.figure8_conditions
            for budget in config.budgets
            for method in HEADLINE_METHODS
        }
    if stage == "objective-comparison":
        return {
            (condition, 0, budget, method)
            for condition in config.figure8_conditions
            for budget in config.budgets
            for method in OBJECTIVE_METHODS
        }
    if stage == "ablation":
        return {
            (
                condition, seed, config.ablation_budget,
                f"mpc_terminal_beam_h{horizon}_w{width}", horizon, width,
            )
            for seed in config.ablation_seeds
            for condition in config.conditions
            for horizon in config.ablation_horizons
            for width in config.ablation_widths
        }
    raise ValueError(f"stage has no scientific matrix: {stage}")


def stage_manifest(
    config: PaperExperimentConfig,
    *,
    stage: str,
    status: str,
    cell_count: int | None = None,
    failure_count: int | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": PAPER_STAGE_SCHEMA,
        "experiment_id": config.experiment_id,
        "stage": stage,
        "status": status,
        "config_path": str(config.source),
        "config_sha256": config.config_sha256,
        "pzr_source_sha256": pzr_source_sha256(),
        "spec_sha256": ROBOT_ARM_SPEC_SHA256,
        "rlolaeval_revision": RLOLAEVAL_REVISION,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "reference_cache_schema": REFERENCE_CACHE_SCHEMA,
    }
    if cell_count is not None:
        payload["cell_count"] = cell_count
    if failure_count is not None:
        payload["failure_count"] = failure_count
    if extra:
        payload.update(dict(extra))
    return payload


def artifact_hash_manifest(directory: Path) -> dict[str, object]:
    """Hash every generated paper artifact except the hash manifest itself."""
    files = tuple(sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    ))
    return {
        "schema": "pzr.paper-generated-artifact-hashes.v1",
        "files": [
            {
                "path": str(path.relative_to(directory)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }


def _require_unique(name: str, values: Sequence[object]) -> None:
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be non-empty and unique")


def _mean(values: np.ndarray, available: bool) -> float:
    return float(np.mean(values)) if available and len(values) else float("nan")


def _median(values: np.ndarray, available: bool) -> float:
    return float(np.median(values)) if available and len(values) else float("nan")


def _quantile(values: np.ndarray, q: float, available: bool) -> float:
    return float(np.quantile(values, q)) if available and len(values) else float("nan")


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload
