#!/usr/bin/env python3
"""Frozen paper-level confirmation sweep for the selected vote-3 PRP guard.

The sweep is intentionally separate from ``paper-evaluation-v3`` and from the
completed vote/guard selection artifact.  It evaluates the frozen selection on
untouched nominal seeds 328--347 and then reports the four fixed figure-eight
traces as descriptive controlled cases.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from pzr.artifact_io import write_csv_atomic, write_json_atomic
from pzr.learning.provenance import model_sha256, payload_sha256, pzr_source_sha256, sha256_files
from pzr.learning.ranker import ReducerPolicy
from pzr.rtlola.benchmark import RtlolaBenchmarkConfig, run_event_trace_benchmark
from pzr.rtlola.binding import BINDING_BUILD_PROFILE, BINDING_REVISION, INTERPRETER_REVISION
from pzr.rtlola.engine import RtlolaEvent
from pzr.rtlola.learning_traces import (
    RandomWaypointTraceStoreConfig,
    generate_random_waypoint_trace_store,
    load_random_waypoint_trace_store,
)
from pzr.rtlola.reference import load_or_compute_reference
from pzr.rtlola.robot_arm import (
    ROBOT_ARM_SPEC_SHA256,
    ROBOT_ARM_TRACE_ROWS,
    ROBOT_ARM_TRACE_SHA256,
    generate_robot_arm_events,
    load_robot_arm_trace,
)
from pzr.rtlola.scenarios import scenario_by_name

import prp_tail_vote_guard_exploratory as selection
import prp_v4_exploratory as v4


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/prp-vote3-guarded-paper-sweep-v1"
SCHEMA = "pzr.prp-vote3-guarded-paper-sweep.v1"
CELL_SCHEMA = "pzr.prp-vote3-guarded-paper-sweep-cell.v1"
FREEZE_SCHEMA = "pzr.prp-vote3-guarded-paper-sweep-freeze.v1"

G15_METHOD = selection.G15_METHOD
PURE_METHOD = "dagger05_vote3"
GUARDED_METHOD = "dagger05_vote3_guarded"
PREDICTIVE_METHOD = selection.PREDICTIVE_METHOD
METHODS = (G15_METHOD, PURE_METHOD, GUARDED_METHOD, PREDICTIVE_METHOD)
CHALLENGERS = (PURE_METHOD, GUARDED_METHOD)
BUDGETS = selection.BUDGETS
CONFIRMATION_SEEDS = selection.CONFIRMATION_SEEDS
FIXED_TRACE_KINDS = v4.FIXED_TRACE_KINDS
EVENT_COUNT = selection.EVENT_COUNT
WORKERS = 10
TAIL_MULTIPLIER = selection.parent.TAIL_MULTIPLIER
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260802
PILOT_BUDGETS = (40, 80, 120, 500)
STAGES = (
    "preflight",
    "prepare",
    "pilot",
    "evaluate-nominal",
    "evaluate-fixed",
    "report",
    "validate",
)

Scope = Literal["nominal", "fixed"]
MethodKind = Literal["g15", "vote", "predictive"]


@dataclass(frozen=True)
class SweepConfig:
    output: Path = DEFAULT_OUTPUT
    budgets: tuple[int, ...] = BUDGETS
    confirmation_seeds: tuple[int, ...] = CONFIRMATION_SEEDS
    fixed_trace_kinds: tuple[str, ...] = FIXED_TRACE_KINDS
    event_count: int = EVENT_COUNT
    workers: int = WORKERS
    smoke: bool = False

    def __post_init__(self) -> None:
        if not self.budgets or tuple(sorted(self.budgets)) != self.budgets:
            raise ValueError("budgets must be non-empty and sorted")
        if not self.confirmation_seeds or len(set(self.confirmation_seeds)) != len(
            self.confirmation_seeds
        ):
            raise ValueError("confirmation seeds must be non-empty and unique")
        if set(self.confirmation_seeds) & set(selection.SELECTION_SEEDS):
            raise ValueError("selection and confirmation seeds overlap")
        if not self.fixed_trace_kinds or len(set(self.fixed_trace_kinds)) != len(
            self.fixed_trace_kinds
        ):
            raise ValueError("fixed trace kinds must be non-empty and unique")
        if self.event_count < 3 or self.workers < 1:
            raise ValueError("event count and workers must be positive")

    @property
    def identity(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "output": str(self.output.resolve()),
            "budgets": list(self.budgets),
            "confirmation_seeds": list(self.confirmation_seeds),
            "fixed_trace_kinds": list(self.fixed_trace_kinds),
            "event_count": self.event_count,
            "workers": self.workers,
            "smoke": self.smoke,
            "methods": list(METHODS),
            "tail_multiplier": TAIL_MULTIPLIER,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "tool_sha256": tool_sha256(),
            "pzr_source_sha256": pzr_source_sha256(),
            "binding_revision": BINDING_REVISION,
            "interpreter_revision": INTERPRETER_REVISION,
            "binding_build_profile": BINDING_BUILD_PROFILE,
        }

    @property
    def fingerprint(self) -> str:
        return payload_sha256(self.identity)

    @property
    def expected_nominal_cells(self) -> int:
        return len(self.confirmation_seeds) * len(self.budgets) * len(METHODS)

    @property
    def expected_fixed_cells(self) -> int:
        return len(self.fixed_trace_kinds) * len(self.budgets) * len(METHODS)


def smoke_config(output: Path, *, workers: int = 1) -> SweepConfig:
    return SweepConfig(
        output=output,
        budgets=(40,),
        confirmation_seeds=(CONFIRMATION_SEEDS[0],),
        fixed_trace_kinds=(FIXED_TRACE_KINDS[0],),
        event_count=30,
        workers=workers,
        smoke=True,
    )


@dataclass(frozen=True)
class TraceRecord:
    trace_id: str
    trace_kind: str
    seed: int
    events: tuple[RtlolaEvent, ...]
    trace_sha256: str
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: MethodKind
    model_paths: tuple[Path, ...] = ()
    guarded: bool = False


@dataclass(frozen=True)
class EvaluationJob:
    config: SweepConfig
    scope: Scope
    trace: TraceRecord
    budget: int
    spec: MethodSpec
    reference_path: Path
    directory: Path


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tool_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        ROOT / "tools/run_prp_vote3_guarded_paper_sweep.sh",
        ROOT / "tools/prp_tail_vote_guard_exploratory.py",
        ROOT / "tools/prp_v4_exploratory.py",
    )
    return sha256_files(tuple(path for path in paths if path.is_file()), relative_to=ROOT)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _stage_path(config: SweepConfig, stage: str) -> Path:
    return config.output / stage / "manifest.json"


def _write_stage(config: SweepConfig, stage: str, extra: Mapping[str, object]) -> Path:
    path = _stage_path(config, stage)
    write_json_atomic(
        {
            **config.identity,
            "experiment_fingerprint": config.fingerprint,
            "stage": stage,
            "status": "completed",
            **dict(extra),
        },
        path,
    )
    return path


def _load_stage(config: SweepConfig, stage: str) -> dict[str, object]:
    path = _stage_path(config, stage)
    manifest = _load_json(path)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported paper-sweep manifest: {path}")
    if manifest.get("experiment_fingerprint") != config.fingerprint:
        raise ValueError(f"stale paper-sweep manifest: {path}")
    return manifest


def _selection_config() -> selection.ExploreConfig:
    return selection.ExploreConfig()


def _selected_member_records(budgets: Sequence[int]) -> dict[int, tuple[dict[str, object], ...]]:
    source_config = _selection_config()
    records: dict[int, tuple[dict[str, object], ...]] = {}
    for budget in budgets:
        spec = next(
            item
            for item in selection._method_specs(source_config, budget)
            if item.name == GUARDED_METHOD
        )
        if not spec.guarded or len(spec.member_paths) != 3:
            raise ValueError(f"selected vote-3 guard identity differs at budget {budget}")
        members = []
        for optimizer_seed, path in zip(selection.REUSED_OPTIMIZER_SEEDS, spec.member_paths, strict=True):
            members.append(
                {
                    "optimizer_seed": optimizer_seed,
                    "budget": budget,
                    "path": str(path),
                    "sha256": model_sha256(path),
                }
            )
        records[budget] = tuple(members)
    return records


def _g15_records(budgets: Sequence[int]) -> dict[int, dict[str, object]]:
    source = v4._v3_model_records(v4.ExploreConfig())
    return {budget: source[budget] for budget in budgets}


def _source_snapshot(config: SweepConfig) -> dict[str, str]:
    source_config = _selection_config()
    paths = [
        selection._stage_path(source_config, stage)
        for stage in selection.STAGES
    ]
    paths.extend(
        (
            selection.DEFAULT_OUTPUT / "report/artifacts/decision.json",
            selection.DEFAULT_OUTPUT / "report/artifacts/artifact_hashes.json",
            v4.PAPER_CONFIG,
            v4.PAPER_ROOT / "train/manifest.json",
            v4.PAPER_ROOT / "science-validate/manifest.json",
        )
    )
    snapshot = {str(path.relative_to(ROOT)): _raw_sha256(path) for path in paths}
    for budget, members in _selected_member_records(config.budgets).items():
        for member in members:
            snapshot[f"vote3:model:{member['optimizer_seed']}:{budget}"] = str(member["sha256"])
    for budget, record in _g15_records(config.budgets).items():
        snapshot[f"g15:model:42:{budget}"] = str(record["sha256"])
    return snapshot


def _deterministic_checks() -> dict[str, bool]:
    scores = np.asarray(
        (
            (0.0, 1.0, 2.0, 3.0),
            (1.0, 0.0, 2.0, 3.0),
            (0.0, 2.0, 1.0, 3.0),
        )
    )
    vote = selection.plurality_order(scores)
    checks = {
        "plurality_winner": int(vote.order[0]) == 0,
        "two_one_guard": selection.strong_disagreement(np.asarray((2, 1, 0, 0))),
        "three_zero_direct": not selection.strong_disagreement(np.asarray((3, 0, 0, 0))),
        "native_tolerance_tie": selection.tolerance_aware_minimum(
            {0: 1.0, 1: 1.0 + 5e-10}, np.asarray((1, 0, 2, 3))
        ) == 1,
    }
    if not all(checks.values()):
        raise AssertionError(f"paper-sweep deterministic checks failed: {checks}")
    return checks


def run_preflight(config: SweepConfig) -> Path:
    if BINDING_BUILD_PROFILE != "release":
        raise ValueError("vote-3 guarded paper sweep requires a release binding")
    source_config = _selection_config()
    selection_validate = selection._load_stage(source_config, "validate")
    selection_report = selection._load_stage(source_config, "report")
    if selection_report.get("selected_method") != GUARDED_METHOD:
        raise ValueError("completed selection did not choose vote3_guarded")
    if not bool(selection_validate.get("confirmation_seeds_untouched")):
        raise ValueError("selection artifact did not preserve confirmation seeds")
    if not bool(selection_validate.get("fixed_traces_untouched")):
        raise ValueError("selection artifact did not preserve fixed traces")
    snapshot = _source_snapshot(config)
    return _write_stage(
        config,
        "preflight",
        {
            "scientific_role": "frozen paper-level PRP confirmation and exploration closure",
            "selection_method": GUARDED_METHOD,
            "selection_manifest": str(selection._stage_path(source_config, "report")),
            "source_snapshot": snapshot,
            "source_snapshot_sha256": payload_sha256(snapshot),
            "deterministic_checks": _deterministic_checks(),
            "fixed_cases_are_descriptive": True,
        },
    )


def _trace_record(item: object) -> TraceRecord:
    return TraceRecord(
        trace_id=item.trace_id,
        trace_kind=item.condition,
        seed=item.seed,
        events=tuple(item.trace.events),
        trace_sha256=item.trace.metadata.trace_sha256,
        provenance={
            "trace_store_relative_path": str(item.relative_path),
            "generator_config": item.trace.metadata.generator_config,
        },
    )


def _fixed_traces(config: SweepConfig) -> tuple[TraceRecord, ...]:
    traces = []
    for trace_kind in config.fixed_trace_kinds:
        rows = load_robot_arm_trace(trace_kind)
        if len(rows) != ROBOT_ARM_TRACE_ROWS[trace_kind]:
            raise ValueError(f"fixed trace length differs: {trace_kind}")
        events = tuple(generate_robot_arm_events(0, trace_kind=trace_kind))
        if config.smoke:
            events = events[: config.event_count]
        traces.append(
            TraceRecord(
                trace_id=trace_kind,
                trace_kind=trace_kind,
                seed=0,
                events=events,
                trace_sha256=ROBOT_ARM_TRACE_SHA256[trace_kind],
                provenance={
                    "fixed": True,
                    "spec_sha256": ROBOT_ARM_SPEC_SHA256,
                    "authoritative_event_count": ROBOT_ARM_TRACE_ROWS[trace_kind],
                },
            )
        )
    return tuple(traces)


def run_prepare(config: SweepConfig) -> Path:
    preflight = _load_stage(config, "preflight")
    trace_store = generate_random_waypoint_trace_store(
        RandomWaypointTraceStoreConfig(
            output=config.output / "prepare/nominal-traces",
            event_count=config.event_count,
            conditions=("random_waypoint",),
            seed_start=min(config.confirmation_seeds),
            seed_count=len(config.confirmation_seeds),
        )
    )
    actual_seeds = tuple(item.seed for item in trace_store.traces)
    if actual_seeds != config.confirmation_seeds:
        raise ValueError("confirmation trace-store seed coverage differs")
    members = _selected_member_records(config.budgets)
    g15 = _g15_records(config.budgets)
    freeze = {
        "schema": FREEZE_SCHEMA,
        "experiment_fingerprint": config.fingerprint,
        "selected_method": GUARDED_METHOD,
        "selection_manifest_sha256": _raw_sha256(Path(str(preflight["selection_manifest"]))),
        "selection_decision_sha256": _raw_sha256(
            selection.DEFAULT_OUTPUT / "report/artifacts/decision.json"
        ),
        "vote3_members_by_budget": {
            str(budget): list(records) for budget, records in members.items()
        },
        "g15_models_by_budget": {
            str(budget): {
                "optimizer_seed": 42,
                "path": str(record["path"]),
                "sha256": str(record["sha256"]),
            }
            for budget, record in g15.items()
        },
    }
    freeze_path = config.output / "prepare/freeze.json"
    if freeze_path.is_file():
        if _load_json(freeze_path) != freeze:
            raise ValueError("paper-sweep freeze differs")
    else:
        write_json_atomic(freeze, freeze_path)
    nominal_records = [
        {
            "trace_id": item.trace_id,
            "seed": item.seed,
            "event_count": len(item.trace.events),
            "trace_sha256": item.trace.metadata.trace_sha256,
        }
        for item in trace_store.traces
    ]
    fixed_records = [
        {
            "trace_id": trace.trace_id,
            "event_count": len(trace.events),
            "trace_sha256": trace.trace_sha256,
            "provenance": dict(trace.provenance),
        }
        for trace in _fixed_traces(config)
    ]
    return _write_stage(
        config,
        "prepare",
        {
            "freeze": str(freeze_path),
            "freeze_sha256": _raw_sha256(freeze_path),
            "nominal_trace_store": str(trace_store.root),
            "nominal_trace_store_manifest_sha256": trace_store.manifest_sha256,
            "nominal_traces": nominal_records,
            "fixed_traces": fixed_records,
        },
    )


def _nominal_traces(config: SweepConfig) -> tuple[TraceRecord, ...]:
    prepare = _load_stage(config, "prepare")
    store = load_random_waypoint_trace_store(Path(str(prepare["nominal_trace_store"])))
    traces = tuple(_trace_record(store.traces_for_seed(seed)[0]) for seed in config.confirmation_seeds)
    if tuple(trace.seed for trace in traces) != config.confirmation_seeds:
        raise ValueError("loaded confirmation traces differ")
    return traces


def _reference_path(config: SweepConfig, scope: Scope, trace: TraceRecord) -> Path:
    return config.output / (
        f"evaluate/{scope}/references/{trace.trace_kind}_seed-{trace.seed}_"
        f"events-{len(trace.events)}.json"
    )


def _prepare_references(
    config: SweepConfig,
    scope: Scope,
    traces: Sequence[TraceRecord],
) -> None:
    for trace in traces:
        load_or_compute_reference(
            trace.events,
            scenario=scenario_by_name("robot_arm"),
            trace_kind=trace.trace_id,
            seed=trace.seed,
            cache_path=_reference_path(config, scope, trace),
            include_approximation=True,
        )


def _method_specs(config: SweepConfig, budget: int) -> tuple[MethodSpec, ...]:
    freeze = _load_json(Path(str(_load_stage(config, "prepare")["freeze"])))
    members = tuple(
        Path(str(item["path"]))
        for item in freeze["vote3_members_by_budget"][str(budget)]
    )
    if len(members) != 3:
        raise ValueError(f"paper sweep requires three vote members at budget {budget}")
    g15_path = Path(str(freeze["g15_models_by_budget"][str(budget)]["path"]))
    return (
        MethodSpec(G15_METHOD, "g15", (g15_path,)),
        MethodSpec(PURE_METHOD, "vote", members, guarded=False),
        MethodSpec(GUARDED_METHOD, "vote", members, guarded=True),
        MethodSpec(PREDICTIVE_METHOD, "predictive"),
    )


def _jobs(
    config: SweepConfig,
    scope: Scope,
    traces: Sequence[TraceRecord],
    *,
    allowed_pairs: set[tuple[str, int]] | None = None,
) -> list[EvaluationJob]:
    jobs = []
    for trace in traces:
        reference_path = _reference_path(config, scope, trace)
        for budget in config.budgets:
            if allowed_pairs is not None and (trace.trace_id, budget) not in allowed_pairs:
                continue
            for spec in _method_specs(config, budget):
                jobs.append(
                    EvaluationJob(
                        config=config,
                        scope=scope,
                        trace=trace,
                        budget=budget,
                        spec=spec,
                        reference_path=reference_path,
                        directory=config.output / (
                            f"evaluate/{scope}/cells/{trace.trace_kind}/seed-{trace.seed}/"
                            f"budget-{budget}/{spec.name}"
                        ),
                    )
                )
    return jobs


def _failure_row(job: EvaluationJob, failure: object) -> dict[str, object]:
    failure_type = str(failure.failure_type)
    return {
        "mean_approx_loss": np.nan,
        "fpr": np.nan,
        "fnr": np.nan,
        "event_count": len(job.trace.events),
        "failure_type": failure_type,
        "failure_message": str(failure.message),
        "status": "fallback_failed" if failure_type == "RtlolaNoFeasibleAction" else "native_failed",
    }


def _execute_evaluation(job: EvaluationJob) -> dict[str, object]:
    manifest_path = job.directory / "manifest.json"
    summary_path = job.directory / "summary.csv"
    model_hashes = [model_sha256(path) for path in job.spec.model_paths]
    identity = {
        "schema": CELL_SCHEMA,
        "experiment_fingerprint": job.config.fingerprint,
        "scope": job.scope,
        "trace_id": job.trace.trace_id,
        "trace_sha256": job.trace.trace_sha256,
        "event_count": len(job.trace.events),
        "budget": job.budget,
        "method": job.spec.name,
        "method_kind": job.spec.kind,
        "guarded": job.spec.guarded,
        "model_sha256": model_hashes,
        "reference_sha256": sha256_files((job.reference_path,)),
    }
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        if manifest.get("identity") != identity:
            raise ValueError(f"stale paper-sweep cell: {job.directory}")
        if _raw_sha256(summary_path) != manifest.get("summary_sha256"):
            raise ValueError(f"paper-sweep summary hash differs: {job.directory}")
        return pd.read_csv(summary_path).iloc[0].to_dict()

    job.directory.mkdir(parents=True, exist_ok=True)
    direct_policy = None
    if job.spec.kind == "g15":
        direct_policy = v4.ExploratoryPolicy(
            ReducerPolicy.load(job.spec.model_paths[0]),
            v4.FeatureVariant.G15,
            events=job.trace.events,
            challenger=False,
        )
    elif job.spec.kind == "vote":
        direct_policy = selection.VoteGuardPolicy(
            tuple(ReducerPolicy.load(path) for path in job.spec.model_paths),
            job.trace.events,
            guarded=job.spec.guarded,
        )

    reference = load_or_compute_reference(
        job.trace.events,
        scenario=scenario_by_name("robot_arm"),
        trace_kind=job.trace.trace_id,
        seed=job.trace.seed,
        cache_path=job.reference_path,
        include_approximation=True,
    )
    benchmark_config = RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind=job.trace.trace_kind,
        length=len(job.trace.events),
        budget=job.budget,
        horizon=4 if job.spec.kind == "predictive" else 0,
        beam_width=4,
        prediction_step_seconds=0.1,
        seeds=1,
        methods=[job.spec.name],
        reference_mode="exact",
        mpc_reference="rollout",
        output_dir=str(job.directory),
        mpc_candidate_names=list(selection.CANDIDATES),
    )
    started = perf_counter()
    result = run_event_trace_benchmark(
        benchmark_config,
        job.trace.events,
        trace_kind=job.trace.trace_kind,
        seed=job.trace.seed,
        method=job.spec.name,
        policy=direct_policy,
        reference_steps=reference,
    )
    elapsed = perf_counter() - started
    if len(result.summary) == 1:
        row = result.summary.iloc[0].to_dict()
        row["status"] = "completed"
    elif result.failures:
        row = _failure_row(job, result.failures[0])
    else:
        raise ValueError("benchmark produced neither a result nor a failure")
    row.update(
        {
            "scope": job.scope,
            "trace_id": job.trace.trace_id,
            "trace_sha256": job.trace.trace_sha256,
            "condition": job.trace.trace_kind,
            "seed": job.trace.seed,
            "budget": job.budget,
            "event_count": len(job.trace.events),
            "method": job.spec.name,
            "guarded": job.spec.guarded,
            "member_count": len(job.spec.model_paths),
            "cell_elapsed_seconds": elapsed,
            "model_sha256": json.dumps(model_hashes),
        }
    )
    write_csv_atomic(pd.DataFrame([row]), summary_path)
    if not result.timeseries.empty:
        write_csv_atomic(result.timeseries, job.directory / "timeseries.csv")
    if not result.failed_timeseries.empty:
        write_csv_atomic(result.failed_timeseries, job.directory / "failed_timeseries.csv")
    if direct_policy is not None and hasattr(direct_policy, "diagnostics"):
        write_csv_atomic(pd.DataFrame(direct_policy.diagnostics), job.directory / "decisions.csv")
    write_json_atomic(
        {
            "schema": CELL_SCHEMA,
            "identity": identity,
            "status": row["status"],
            "summary_sha256": _raw_sha256(summary_path),
        },
        manifest_path,
    )
    return row


def _run_jobs(config: SweepConfig, jobs: Sequence[EvaluationJob]) -> pd.DataFrame:
    if config.workers == 1:
        rows = [_execute_evaluation(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=get_context("spawn"),
            max_tasks_per_child=1,
        ) as executor:
            rows = list(executor.map(_execute_evaluation, jobs))
    return pd.DataFrame(rows)


def run_pilot(config: SweepConfig) -> Path:
    _load_stage(config, "prepare")
    trace = _nominal_traces(config)[0]
    budgets = tuple(budget for budget in PILOT_BUDGETS if budget in config.budgets)
    if not budgets:
        budgets = (config.budgets[0],)
    pairs = {(trace.trace_id, budget) for budget in budgets}
    _prepare_references(config, "nominal", (trace,))
    started = perf_counter()
    summary = _run_jobs(config, _jobs(config, "nominal", (trace,), allowed_pairs=pairs))
    expected = len(budgets) * len(METHODS)
    if len(summary) != expected:
        raise ValueError(f"paper-sweep pilot has {len(summary)} cells, expected {expected}")
    path = config.output / "pilot/summary.csv"
    write_csv_atomic(summary, path)
    cpu_seconds = float(summary["cell_elapsed_seconds"].sum())
    projected_nominal_wall_hours = (
        cpu_seconds / len(summary) * config.expected_nominal_cells / config.workers / 3600.0
    )
    return _write_stage(
        config,
        "pilot",
        {
            "cell_count": len(summary),
            "seed": trace.seed,
            "budgets": list(budgets),
            "reused_by_nominal_sweep": True,
            "wall_seconds": perf_counter() - started,
            "cpu_seconds": cpu_seconds,
            "projected_nominal_wall_hours": projected_nominal_wall_hours,
            "fixed_projection_deferred_because_authoritative_lengths_differ": True,
            "summary": str(path),
        },
    )


def run_evaluate_nominal(config: SweepConfig) -> Path:
    _load_stage(config, "pilot")
    traces = _nominal_traces(config)
    _prepare_references(config, "nominal", traces)
    started = perf_counter()
    summary = _run_jobs(config, _jobs(config, "nominal", traces))
    if len(summary) != config.expected_nominal_cells:
        raise ValueError(
            f"nominal sweep has {len(summary)} cells, expected {config.expected_nominal_cells}"
        )
    keys = summary[["trace_id", "budget", "method"]]
    if keys.duplicated().any():
        raise ValueError("nominal sweep contains duplicate cells")
    path = config.output / "evaluate-nominal/summary.csv"
    write_csv_atomic(summary, path)
    return _write_stage(
        config,
        "evaluate-nominal",
        {
            "cell_count": len(summary),
            "trace_count": len(traces),
            "summary": str(path),
            "summary_sha256": _raw_sha256(path),
            "wall_seconds": perf_counter() - started,
        },
    )


def run_evaluate_fixed(config: SweepConfig) -> Path:
    _load_stage(config, "evaluate-nominal")
    traces = _fixed_traces(config)
    _prepare_references(config, "fixed", traces)
    started = perf_counter()
    summary = _run_jobs(config, _jobs(config, "fixed", traces))
    if len(summary) != config.expected_fixed_cells:
        raise ValueError(
            f"fixed sweep has {len(summary)} cells, expected {config.expected_fixed_cells}"
        )
    keys = summary[["trace_id", "budget", "method"]]
    if keys.duplicated().any():
        raise ValueError("fixed sweep contains duplicate cells")
    path = config.output / "evaluate-fixed/summary.csv"
    write_csv_atomic(summary, path)
    return _write_stage(
        config,
        "evaluate-fixed",
        {
            "cell_count": len(summary),
            "trace_count": len(traces),
            "summary": str(path),
            "summary_sha256": _raw_sha256(path),
            "wall_seconds": perf_counter() - started,
            "descriptive_only": True,
        },
    )


def _paired(
    summary: pd.DataFrame,
    method: str,
    reference: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["scope", "trace_id", "budget"]
    candidate = summary[summary["method"] == method].set_index(keys).sort_index()
    baseline = summary[summary["method"] == reference].set_index(keys).sort_index()
    if set(candidate.index) != set(baseline.index):
        raise ValueError(f"{method} and {reference} cells do not align")
    return candidate, baseline.loc[candidate.index]


def _bootstrap_mean(values: np.ndarray) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = data[rng.integers(0, len(data), size=(BOOTSTRAP_REPLICATES, len(data)))]
    means = samples.mean(axis=1)
    return float(data.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _throughput(frame: pd.DataFrame) -> np.ndarray:
    return frame["event_count"].to_numpy(float) / (
        frame["event_loop_time_ms"].to_numpy(float) / 1000.0
    )


def _ratios(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    left = np.asarray(numerator, dtype=np.float64)
    right = np.asarray(denominator, dtype=np.float64)
    result = np.full_like(left, np.inf)
    np.divide(left, right, out=result, where=right != 0.0)
    result[(left == 0.0) & (right == 0.0)] = 1.0
    return result


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if len(values) else float("nan")


def _maximum(values: np.ndarray) -> float:
    return float(np.max(values)) if len(values) else float("nan")


def _scope_metrics(
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired_rows = []
    overall_rows = []
    budget_rows = []
    severe_rows = []
    for method in CHALLENGERS:
        candidate, g15 = _paired(summary, method, G15_METHOD)
        _, predictive = _paired(summary, method, PREDICTIVE_METHOD)
        valid = (
            (candidate["status"].astype(str) == "completed")
            & (g15["status"].astype(str) == "completed")
            & (predictive["status"].astype(str) == "completed")
        )
        numeric_columns = (
            "mean_approx_loss",
            "fpr",
            "event_count",
            "event_loop_time_ms",
        )
        for column in numeric_columns:
            valid &= np.isfinite(pd.to_numeric(candidate[column], errors="coerce"))
            valid &= np.isfinite(pd.to_numeric(g15[column], errors="coerce"))
            valid &= np.isfinite(pd.to_numeric(predictive[column], errors="coerce"))
        candidate_valid = candidate.loc[valid]
        g15_valid = g15.loc[valid]
        predictive_valid = predictive.loc[valid]
        loss = candidate_valid["mean_approx_loss"].to_numpy(float)
        g15_loss = g15_valid["mean_approx_loss"].to_numpy(float)
        predictive_loss = predictive_valid["mean_approx_loss"].to_numpy(float)
        ratio_g15 = _ratios(loss, g15_loss)
        ratio_predictive = _ratios(loss, predictive_loss)
        fpr_delta = candidate_valid["fpr"].to_numpy(float) - g15_valid["fpr"].to_numpy(float)
        throughput = _throughput(candidate_valid)
        g15_throughput = _throughput(g15_valid)
        fallback_count = int(
            pd.to_numeric(candidate["fallback_count"], errors="coerce").fillna(0).sum()
        )
        fpr_mean, fpr_low, fpr_high = _bootstrap_mean(fpr_delta)
        for key, row, rg15, rpred, fpr, retention in zip(
            candidate_valid.index,
            candidate_valid.itertuples(),
            ratio_g15,
            ratio_predictive,
            fpr_delta,
            throughput / g15_throughput,
            strict=True,
        ):
            paired_rows.append(
                {
                    "scope": key[0],
                    "trace_id": key[1],
                    "seed": int(row.seed),
                    "budget": int(key[2]),
                    "method": method,
                    "loss_ratio_vs_g15": float(rg15),
                    "loss_ratio_vs_predictive": float(rpred),
                    "fpr_difference_vs_g15": float(fpr),
                    "throughput_retention_vs_g15": float(retention),
                }
            )
            if rpred > TAIL_MULTIPLIER:
                severe_rows.append(
                    {
                        "scope": key[0],
                        "trace_id": key[1],
                        "seed": int(row.seed),
                        "budget": int(key[2]),
                        "method": method,
                        "loss_ratio_vs_predictive": float(rpred),
                    }
                )
        overall_rows.append(
            {
                "scope": str(summary["scope"].iloc[0]),
                "method": method,
                "cell_count": len(candidate),
                "valid_count": int(valid.sum()),
                "failure_count": int((candidate["status"].astype(str) != "completed").sum()),
                "unavailable_count": int((~valid).sum()),
                "fallback_count": fallback_count,
                "severe_tail_count": int(np.count_nonzero(ratio_predictive > TAIL_MULTIPLIER)),
                "worst_loss_ratio_vs_predictive": _maximum(ratio_predictive),
                "p95_loss_ratio_vs_predictive": _quantile(ratio_predictive, 0.95),
                "median_loss_ratio_vs_g15": _quantile(ratio_g15, 0.5),
                "mean_fpr_difference_vs_g15": fpr_mean,
                "mean_fpr_difference_ci_low": fpr_low,
                "mean_fpr_difference_ci_high": fpr_high,
                "max_fpr_difference_vs_g15": _maximum(fpr_delta[np.isfinite(fpr_delta)]),
                "median_paired_throughput_retention": float(
                    _quantile(_ratios(throughput, g15_throughput), 0.5)
                ),
            }
        )
        for budget in sorted(summary["budget"].astype(int).unique()):
            budget_candidate = candidate_valid[
                candidate_valid.index.get_level_values("budget").astype(int) == budget
            ]
            budget_g15 = g15_valid.loc[budget_candidate.index]
            budget_predictive = predictive_valid.loc[budget_candidate.index]
            budget_loss = budget_candidate["mean_approx_loss"].to_numpy(float)
            budget_g15_loss = budget_g15["mean_approx_loss"].to_numpy(float)
            budget_predictive_loss = budget_predictive["mean_approx_loss"].to_numpy(float)
            budget_rows.append(
                {
                    "scope": str(summary["scope"].iloc[0]),
                    "method": method,
                    "budget": budget,
                    "cell_count": len(candidate[candidate.index.get_level_values("budget") == budget]),
                    "valid_count": len(budget_candidate),
                    "median_loss_ratio_vs_g15": _quantile(
                        _ratios(budget_loss, budget_g15_loss), 0.5
                    ),
                    "p95_loss_ratio_vs_predictive": _quantile(
                        _ratios(budget_loss, budget_predictive_loss), 0.95
                    ),
                    "worst_loss_ratio_vs_predictive": _maximum(
                        _ratios(budget_loss, budget_predictive_loss)
                    ),
                    "severe_tail_count": int(
                        np.count_nonzero(
                            _ratios(budget_loss, budget_predictive_loss) > TAIL_MULTIPLIER
                        )
                    ),
                }
            )
    per_budget = pd.DataFrame(budget_rows)
    overall = pd.DataFrame(overall_rows)
    maxima = per_budget.groupby("method")["median_loss_ratio_vs_g15"].max()
    overall["max_budget_median_loss_ratio_vs_g15"] = overall["method"].map(maxima)
    return pd.DataFrame(paired_rows), overall, per_budget, pd.DataFrame(severe_rows)


def _eligibility(per_budget: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in overall.iterrows():
        method_budgets = per_budget[per_budget["method"] == row["method"]]
        checks = {
            "zero_severe_tail": int(row["severe_tail_count"]) == 0,
            "zero_failures": int(row["failure_count"]) == 0,
            "all_cells_available": int(row["unavailable_count"]) == 0,
            "zero_fallbacks": int(row["fallback_count"]) == 0,
            "all_budget_medians_within_1_25_g15": bool(
                (method_budgets["median_loss_ratio_vs_g15"] <= 1.25).all()
            ),
            "throughput_at_least_half_g15": float(
                row["median_paired_throughput_retention"]
            ) >= 0.50,
            "mean_fpr_regression_at_most_0_005": float(
                row["mean_fpr_difference_vs_g15"]
            ) <= 0.005,
            "individual_fpr_regression_at_most_0_05": float(
                row["max_fpr_difference_vs_g15"]
            ) <= 0.05,
        }
        rows.append({**row.to_dict(), **checks, "eligible": all(checks.values())})
    return pd.DataFrame(rows)


def _guard_benefit(summary: pd.DataFrame) -> pd.DataFrame:
    guarded, pure = _paired(summary, GUARDED_METHOD, PURE_METHOD)
    valid = (
        (guarded["status"].astype(str) == "completed")
        & (pure["status"].astype(str) == "completed")
    )
    guarded = guarded.loc[valid]
    pure = pure.loc[valid]
    rows = []
    for key in guarded.index:
        guard_row = guarded.loc[key]
        pure_row = pure.loc[key]
        rows.append(
            {
                "scope": key[0],
                "trace_id": key[1],
                "seed": int(guard_row["seed"]),
                "budget": int(key[2]),
                "mean_loss_ratio_guarded_vs_pure": _ratio(
                    float(guard_row["mean_approx_loss"]), float(pure_row["mean_approx_loss"])
                ),
                "final_loss_ratio_guarded_vs_pure": _ratio(
                    float(guard_row["final_approx_loss"]), float(pure_row["final_approx_loss"])
                ),
                "max_loss_ratio_guarded_vs_pure": _ratio(
                    float(guard_row["max_approx_loss"]), float(pure_row["max_approx_loss"])
                ),
                "fpr_difference_guarded_vs_pure": float(guard_row["fpr"] - pure_row["fpr"]),
                "throughput_retention_guarded_vs_pure": _ratio(
                    float(_throughput(guarded.loc[[key]])[0]),
                    float(_throughput(pure.loc[[key]])[0]),
                ),
            }
        )
    return pd.DataFrame(rows)


def _decision_tables(config: SweepConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for scope in ("nominal", "fixed"):
        root = config.output / f"evaluate/{scope}/cells"
        for path in sorted(root.rglob("decisions.csv")):
            if path.parent.name not in (PURE_METHOD, GUARDED_METHOD):
                continue
            frame = pd.read_csv(path)
            frame.insert(0, "scope", scope)
            frame.insert(1, "method", path.parent.name)
            frame.insert(2, "trace_kind", next(part for part in path.parts if part in (*config.fixed_trace_kinds, "random_waypoint")))
            frame.insert(3, "seed", int(next(part.split("-", 1)[1] for part in path.parts if part.startswith("seed-"))))
            frame["budget"] = int(next(part.split("-", 1)[1] for part in path.parts if part.startswith("budget-")))
            frames.append(frame)
    decisions = pd.concat(frames, ignore_index=True)
    guarded = decisions[
        (decisions["method"] == GUARDED_METHOD) & decisions["over_bound"].astype(bool)
    ]
    rates = guarded.groupby(["scope", "budget"], sort=True).agg(
        decision_count=("step", "size"),
        guard_count=("guard_invoked", "sum"),
        guard_rate=("guard_invoked", "mean"),
        override_count=("guard_override", "sum"),
        mean_contenders=("contender_count", lambda values: float(values[values > 1].mean())),
        mean_added_decision_time_ms=("guard_added_decision_time_ms", lambda values: float(values[values > 0].mean())),
    ).reset_index()
    rates["conditional_override_rate"] = rates["override_count"] / rates["guard_count"].replace(0, np.nan)
    return decisions, rates


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _save_figure(fig: object, path: Path) -> tuple[Path, Path]:
    pdf = path.with_suffix(".pdf")
    png = path.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    return pdf, png


def _write_figures(
    nominal_paired: pd.DataFrame,
    nominal_benefit: pd.DataFrame,
    fixed_paired: pd.DataFrame,
    output: Path,
) -> list[Path]:
    plt = _pyplot()
    palette = {PURE_METHOD: "#0072B2", GUARDED_METHOD: "#D55E00"}
    labels = {PURE_METHOD: "Vote-3", GUARDED_METHOD: "Vote-3 guarded"}
    artifacts: list[Path] = []

    fig, axis = plt.subplots(figsize=(7.1, 3.0), constrained_layout=True)
    for method, linestyle in ((PURE_METHOD, "--"), (GUARDED_METHOD, "-")):
        values = np.sort(
            nominal_paired.loc[
                nominal_paired["method"] == method,
                "loss_ratio_vs_predictive",
            ].to_numpy(float)
        )
        axis.step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            color=palette[method],
            linestyle=linestyle,
            linewidth=1.4,
            label=f"{labels[method]} (n={len(values)} cells)",
        )
    axis.axvline(1.0, color="0.35", linewidth=0.8, label="Predictive MPC")
    axis.axvline(TAIL_MULTIPLIER, color="0.55", linewidth=0.8, linestyle=":", label="Severe-tail threshold")
    axis.set_xscale("log")
    axis.set_xlabel("Paired mean-loss ratio to predictive MPC (log scale; lower is better)")
    axis.set_ylabel("Empirical cumulative fraction of cells")
    axis.grid(axis="both", color="0.9", linewidth=0.5)
    axis.legend(frameon=False, ncol=2, loc="lower right")
    artifacts.extend(_save_figure(fig, output / "nominal_loss_ratio_ecdf"))
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.1, 3.0), constrained_layout=True)
    budgets = sorted(nominal_benefit["budget"].astype(int).unique())
    for position, budget in enumerate(budgets):
        values = nominal_benefit.loc[
            nominal_benefit["budget"].astype(int) == budget,
            "mean_loss_ratio_guarded_vs_pure",
        ].to_numpy(float)
        offsets = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.zeros(1)
        axis.scatter(
            np.full(len(values), position) + offsets,
            values,
            s=10,
            facecolors="none",
            edgecolors=palette[GUARDED_METHOD],
            linewidths=0.7,
            alpha=0.8,
        )
        q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
        axis.vlines(position, q25, q75, color="black", linewidth=1.8)
        axis.scatter(position, median, color="black", marker="_", s=75, linewidths=1.5, zorder=3)
    axis.axhline(1.0, color="0.35", linewidth=0.8)
    axis.set_yscale("log")
    axis.set_xticks(range(len(budgets)), budgets)
    axis.set_xlabel("Reducer budget")
    axis.set_ylabel("Guarded/pure mean-loss ratio (log scale; lower is better)")
    axis.grid(axis="y", color="0.9", linewidth=0.5)
    artifacts.extend(_save_figure(fig, output / "nominal_guard_benefit_by_budget"))
    plt.close(fig)

    guard_fixed = fixed_paired[fixed_paired["method"] == GUARDED_METHOD].copy()
    trace_kinds = list(dict.fromkeys(guard_fixed["trace_id"].astype(str)))
    matrix = guard_fixed.pivot(index="trace_id", columns="budget", values="loss_ratio_vs_predictive").loc[
        trace_kinds, budgets
    ]
    log_matrix = np.log10(matrix.to_numpy(float))
    bound = max(1.0, float(np.nanmax(np.abs(log_matrix))))
    fig, axis = plt.subplots(figsize=(7.1, 2.8), constrained_layout=True)
    image = axis.imshow(log_matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    axis.set_xticks(range(len(budgets)), budgets)
    axis.set_yticks(range(len(trace_kinds)), [name.replace("figure8", "figure-8") for name in trace_kinds])
    axis.set_xlabel("Reducer budget")
    axis.set_ylabel("Fixed controlled trace")
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("log10 mean-loss ratio to predictive MPC")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix.iloc[row, column])
            axis.text(column, row, f"{value:.2g}", ha="center", va="center", fontsize=6)
    artifacts.extend(_save_figure(fig, output / "fixed_guarded_loss_ratio"))
    plt.close(fig)
    return artifacts


def run_report(config: SweepConfig) -> Path:
    nominal_stage = _load_stage(config, "evaluate-nominal")
    fixed_stage = _load_stage(config, "evaluate-fixed")
    nominal = pd.read_csv(str(nominal_stage["summary"]))
    fixed = pd.read_csv(str(fixed_stage["summary"]))
    nominal_paired, nominal_overall, nominal_budget, nominal_severe = _scope_metrics(nominal)
    fixed_paired, fixed_overall, fixed_budget, fixed_severe = _scope_metrics(fixed)
    eligibility = _eligibility(nominal_budget, nominal_overall)
    nominal_benefit = _guard_benefit(nominal)
    fixed_benefit = _guard_benefit(fixed)
    decisions, guard_rates = _decision_tables(config)
    artifact_dir = config.output / "report/artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "trace_cells.csv": pd.concat((nominal, fixed), ignore_index=True),
        "nominal_paired_cells.csv": nominal_paired,
        "fixed_paired_cells.csv": fixed_paired,
        "nominal_overall_metrics.csv": nominal_overall,
        "fixed_overall_metrics.csv": fixed_overall,
        "nominal_per_budget_metrics.csv": nominal_budget,
        "fixed_per_budget_metrics.csv": fixed_budget,
        "nominal_severe_tail_cells.csv": nominal_severe,
        "fixed_severe_tail_cells.csv": fixed_severe,
        "confirmation_eligibility.csv": eligibility,
        "nominal_guard_benefit.csv": nominal_benefit,
        "fixed_guard_benefit.csv": fixed_benefit,
        "guard_rates.csv": guard_rates,
        "vote_decisions.csv": decisions,
    }
    artifacts: list[Path] = []
    for name, frame in tables.items():
        path = artifact_dir / name
        write_csv_atomic(frame, path)
        artifacts.append(path)
    artifacts.extend(_write_figures(nominal_paired, nominal_benefit, fixed_paired, artifact_dir))
    guarded_row = eligibility[eligibility["method"] == GUARDED_METHOD]
    confirmed = len(guarded_row) == 1 and bool(guarded_row.iloc[0]["eligible"])
    decision = {
        "selected_method": GUARDED_METHOD if confirmed else G15_METHOD,
        "vote3_guarded_confirmed": confirmed,
        "fallback_if_not_confirmed": G15_METHOD,
        "selection_was_frozen_before_confirmation": True,
        "nominal_confirmation_is_decision_evidence": True,
        "fixed_cases_are_descriptive": True,
        "fixed_severe_tail_count": int(
            fixed_overall.loc[
                fixed_overall["method"] == GUARDED_METHOD, "severe_tail_count"
            ].iloc[0]
        ),
        "exploration_concluded": True,
        "decision_rule": "unchanged selection eligibility gates on untouched nominal confirmation seeds",
    }
    decision_path = artifact_dir / "decision.json"
    write_json_atomic(decision, decision_path)
    artifacts.append(decision_path)
    figure_metadata = {
        "nominal_observational_unit": "one 500-event seed-budget cell",
        "fixed_observational_unit": "one full controlled trace-budget cell",
        "nominal_seed_count": len(config.confirmation_seeds),
        "fixed_trace_count": len(config.fixed_trace_kinds),
        "loss_ratio_baseline": PREDICTIVE_METHOD,
        "guard_benefit_baseline": PURE_METHOD,
        "ecdf_aggregation": "trace-cell empirical distribution; no time-step pseudoreplication",
        "fixed_heatmap_transform": "base-10 logarithm of paired positive loss ratios",
        "exports": "double-column provisional 7.1-inch PDF with embedded TrueType fonts and 300-dpi PNG preview",
    }
    metadata_path = artifact_dir / "figure_metadata.json"
    write_json_atomic(figure_metadata, metadata_path)
    artifacts.append(metadata_path)
    hashes = {str(path.relative_to(artifact_dir)): _raw_sha256(path) for path in artifacts}
    hash_path = artifact_dir / "artifact_hashes.json"
    write_json_atomic(hashes, hash_path)
    return _write_stage(
        config,
        "report",
        {
            **decision,
            "artifact_directory": str(artifact_dir),
            "artifact_count": len(artifacts) + 1,
            "artifact_hashes": str(hash_path),
        },
    )


def run_validate(config: SweepConfig) -> Path:
    report = _load_stage(config, "report")
    for stage in STAGES[:-1]:
        _load_stage(config, stage)
    preflight = _load_stage(config, "preflight")
    if _source_snapshot(config) != preflight["source_snapshot"]:
        raise ValueError("frozen selection or paper references changed during the sweep")
    prepare = _load_stage(config, "prepare")
    if _raw_sha256(Path(str(prepare["freeze"]))) != prepare["freeze_sha256"]:
        raise ValueError("paper-sweep freeze hash differs")
    nominal = _load_stage(config, "evaluate-nominal")
    fixed = _load_stage(config, "evaluate-fixed")
    if int(nominal["cell_count"]) != config.expected_nominal_cells:
        raise ValueError("nominal confirmation cell count differs")
    if int(fixed["cell_count"]) != config.expected_fixed_cells:
        raise ValueError("fixed controlled-case cell count differs")
    hashes = _load_json(Path(str(report["artifact_hashes"])))
    artifact_root = Path(str(report["artifact_hashes"])).parent
    for relative, expected in hashes.items():
        if _raw_sha256(artifact_root / relative) != expected:
            raise ValueError(f"paper-sweep report artifact hash differs: {relative}")
    trace_cells = pd.read_csv(artifact_root / "trace_cells.csv")
    expected_total = config.expected_nominal_cells + config.expected_fixed_cells
    if len(trace_cells) != expected_total:
        raise ValueError("joined paper-sweep report cell count differs")
    nominal_seeds = tuple(
        sorted(trace_cells.loc[trace_cells["scope"] == "nominal", "seed"].astype(int).unique())
    )
    if nominal_seeds != config.confirmation_seeds:
        raise ValueError("reported confirmation seed coverage differs")
    fixed_kinds = tuple(
        trace_cells.loc[trace_cells["scope"] == "fixed", "trace_id"].astype(str).drop_duplicates()
    )
    if fixed_kinds != config.fixed_trace_kinds:
        raise ValueError("reported fixed-trace coverage differs")
    return _write_stage(
        config,
        "validate",
        {
            "source_snapshot_unchanged": True,
            "freeze_unchanged": True,
            "nominal_cell_count": config.expected_nominal_cells,
            "fixed_cell_count": config.expected_fixed_cells,
            "report_cell_count": expected_total,
            "confirmation_seed_coverage_exact": True,
            "fixed_trace_coverage_exact": True,
            "fixed_cases_are_descriptive": True,
            "vote3_guarded_confirmed": bool(report["vote3_guarded_confirmed"]),
            "selected_method": str(report["selected_method"]),
            "exploration_concluded": True,
            "artifact_hash_count": len(hashes),
        },
    )


def run_stage(config: SweepConfig, stage: str) -> Path:
    path = _stage_path(config, stage)
    if path.is_file():
        _load_stage(config, stage)
        print(f"skip completed vote3_guarded paper-sweep stage: {stage}", flush=True)
        return path
    functions = {
        "preflight": run_preflight,
        "prepare": run_prepare,
        "pilot": run_pilot,
        "evaluate-nominal": run_evaluate_nominal,
        "evaluate-fixed": run_evaluate_fixed,
        "report": run_report,
        "validate": run_validate,
    }
    print(f"start vote3_guarded paper-sweep stage: {stage}", flush=True)
    result = functions[stage](config)
    print(f"complete vote3_guarded paper-sweep stage: {stage}", flush=True)
    return result


def run_all(config: SweepConfig) -> Path:
    print(json.dumps(config.identity, indent=2), flush=True)
    for stage in STAGES:
        run_stage(config, stage)
    return _stage_path(config, "validate")


def status(config: SweepConfig) -> dict[str, object]:
    stages = {}
    for stage in STAGES:
        path = _stage_path(config, stage)
        if not path.is_file():
            stages[stage] = "missing"
            continue
        try:
            _load_stage(config, stage)
            stages[stage] = "completed"
        except ValueError as exc:
            stages[stage] = f"stale: {exc}"
    return {
        "output": str(config.output),
        "smoke": config.smoke,
        "methods": list(METHODS),
        "expected_nominal_cells": config.expected_nominal_cells,
        "expected_fixed_cells": config.expected_fixed_cells,
        "expected_total_cells": config.expected_nominal_cells + config.expected_fixed_cells,
        "stages": stages,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(*STAGES, "run", "status"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = (
        smoke_config(args.output, workers=args.workers)
        if args.smoke
        else SweepConfig(output=args.output, workers=args.workers)
    )
    if args.command == "status":
        print(json.dumps(status(config), indent=2))
        return 0
    path = run_all(config) if args.command == "run" else run_stage(config, args.command)
    print(f"vote3_guarded paper sweep stage complete: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
