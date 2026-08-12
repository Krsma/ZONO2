#!/usr/bin/env python3
"""Standalone final paper-evaluation v4 orchestration.

V4 evaluates every paper method on the same untouched nominal cohort.  It
imports immutable v3 design/fixed artifacts by hash and does not modify the
scientific package or any earlier result directory.  The default ``run``
command executes only the workstation scientific stages; timing is an explicit
diagnostic command because the paper-facing timing is collected independently
on Raspberry Pi 5.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Literal, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import yaml

import pzr.rtlola.benchmark as benchmark_module
from pzr.artifact_io import write_csv_atomic, write_json_atomic
from pzr.learning.provenance import model_sha256, payload_sha256, pzr_source_sha256
from pzr.learning.ranker import ReducerPolicy
from pzr.rtlola.actions import default_action_catalog
from pzr.rtlola.benchmark import RtlolaBenchmarkConfig, run_event_trace_benchmark
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.engine import RtlolaEvent
from pzr.rtlola.learned_policy import RtlolaReducerPolicy
from pzr.rtlola.learning_traces import (
    RandomWaypointTraceStoreConfig,
    generate_random_waypoint_trace_store,
    load_random_waypoint_trace_store,
)
from pzr.rtlola.reference import load_or_compute_reference
from pzr.rtlola.robot_arm import ROBOT_ARM_SPEC_SHA256, ROBOT_ARM_TRACE_SHA256
from pzr.rtlola.scenarios import scenario_by_name

import prp_tail_vote_guard_exploratory as vote_selection


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/paper_evaluation_v4.yaml"
SCHEMA = "pzr.paper-evaluation-config.v4"
STAGE_SCHEMA = "pzr.paper-evaluation-stage.v4"
CELL_SCHEMA = "pzr.paper-evaluation-cell.v4"
STAGES = (
    "preflight",
    "prepare",
    "pilot",
    "nominal",
    "prediction-ablation",
    "runtime",
    "report",
    "validate",
)
RUN_STAGES = STAGES[:5]
METHOD_NAMES = (
    "girard",
    "scott",
    "pca",
    "combastel",
    "mpc_terminal_beam",
    "mpc_terminal_full_width",
    "mpc_terminal_beam_predictive_linear",
    "pairwise_ranking_policy",
    "dagger05_vote3",
    "dagger05_vote3_guarded",
)
PREDICTOR_METHODS = {
    "hold": "mpc_terminal_beam_predictive_hold",
    "linear": "mpc_terminal_beam_predictive_linear",
    "quadratic": "mpc_terminal_beam_predictive_quadratic",
}
FIXED_TRACE_KINDS = (
    "figure8",
    "figure8_drift",
    "figure8_geofence",
    "figure8_drift_geofence",
)
EXPECTED_V3_SOURCE_HASH = "27ce18877eb911bf130e7d38982288bd89900238c6cf757c127385f9fb1ca23c"
SEVERE_TAIL_MULTIPLIER = 1_000.0
# The packaged robot-arm recurrence appends 35 dynamic generators per event.
# The binding transform bound applies before that fresh contribution.  Some
# native reducers retain at most one calibration row per logical dimension, so
# the sound post-event check is b + 35 + logical dimension (never b itself).
ROBOT_ARM_FRESH_DYNAMIC_ROWS = 35

MethodKind = Literal["static", "mpc", "g15", "vote3", "vote3_guarded"]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    kind: MethodKind
    runtime_method: str
    horizon: int
    beam_width: int


@dataclass(frozen=True)
class ImportSpec:
    name: str
    manifest: Path
    manifest_sha256: str
    summary: Path
    summary_sha256: str


@dataclass(frozen=True)
class V4Config:
    source: Path
    output: Path
    event_count: int
    budgets: tuple[int, ...]
    workers: int
    candidates: tuple[str, ...]
    nominal_seeds: tuple[int, ...]
    pilot_seeds: tuple[int, ...]
    prediction_seeds: tuple[int, ...]
    prediction_budget: int
    prediction_horizon: int
    prediction_width: int
    predictors: tuple[str, ...]
    timing_seeds: tuple[int, ...]
    warmup_start: int
    warmup_stop: int
    measured_start: int
    measured_stop: int
    methods: tuple[MethodSpec, ...]
    imports: tuple[ImportSpec, ...]
    native: Mapping[str, str]
    fixed_hashes: Mapping[str, str]
    selection_decision: Path
    selection_decision_sha256: str
    selection_freeze: Path
    selection_freeze_sha256: str
    config_sha256: str
    smoke: bool = False

    def __post_init__(self) -> None:
        if tuple(method.name for method in self.methods) != METHOD_NAMES:
            raise ValueError("v4 method matrix or order differs")
        if tuple(sorted(self.budgets)) != self.budgets or len(set(self.budgets)) != len(self.budgets):
            raise ValueError("v4 budgets must be unique and sorted")
        seed_sets = {
            "nominal": set(self.nominal_seeds),
            "prediction": set(self.prediction_seeds),
            "v3_train": set(range(0, 20)) | set(range(26, 42)) | set(range(200, 312)),
            "v3_validation": set(range(20, 26)),
            "v3_exploration": set(range(312, 328)),
            "vote_selection": set(range(328, 348)),
        }
        for left, left_values in seed_sets.items():
            for right, right_values in seed_sets.items():
                if left < right and left_values & right_values:
                    raise ValueError(f"v4 seed groups overlap: {left} and {right}")
        if not set(self.pilot_seeds) <= set(self.nominal_seeds):
            raise ValueError("pilot must reuse v4 nominal cells")
        if not set(self.timing_seeds) <= set(self.nominal_seeds):
            raise ValueError("timing must use v4 nominal traces")
        if self.predictors != tuple(PREDICTOR_METHODS):
            raise ValueError("predictor identities differ")
        if not (self.warmup_start == 0 < self.warmup_stop == self.measured_start < self.measured_stop):
            raise ValueError("timing windows must be contiguous and non-empty")
        if self.measured_stop > self.event_count:
            raise ValueError("timing window exceeds trace length")

    @property
    def fingerprint(self) -> str:
        return payload_sha256({
            "schema": SCHEMA,
            "config_sha256": self.config_sha256,
            "output": str(self.output.resolve()),
            "smoke": self.smoke,
            "tool_sha256": raw_sha256(Path(__file__)),
        })

    @property
    def expected_nominal_cells(self) -> int:
        return len(self.nominal_seeds) * len(self.budgets) * len(self.methods)

    @property
    def expected_prediction_cells(self) -> int:
        return len(self.prediction_seeds) * len(self.predictors)

    @property
    def expected_timing_cells(self) -> int:
        return len(self.timing_seeds) * len(self.budgets) * len(self.methods)


@dataclass(frozen=True)
class TraceRecord:
    trace_id: str
    seed: int
    events: tuple[RtlolaEvent, ...]
    sha256: str


@dataclass(frozen=True)
class CellJob:
    config: V4Config
    stage: str
    trace: TraceRecord
    budget: int
    method: MethodSpec
    reference: Path
    directory: Path


def raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path = DEFAULT_CONFIG, *, output: Path | None = None, smoke: bool = False, workers: int | None = None) -> V4Config:
    payload = yaml.safe_load(path.read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported v4 config schema: {payload.get('schema')}")
    methods = tuple(MethodSpec(**item) for item in payload["methods"])
    imports = tuple(
        ImportSpec(name=name, manifest=_path(item["manifest"]), manifest_sha256=str(item["manifest_sha256"]), summary=_path(item["summary"]), summary_sha256=str(item["summary_sha256"]))
        for name, item in payload["imports"].items()
    )
    prediction = payload["prediction_ablation"]
    timing = payload["timing"]
    config = V4Config(
        source=path.resolve(), output=(output or _path(payload["output_root"])).resolve(),
        event_count=int(payload["event_count"]), budgets=tuple(map(int, payload["budgets"])),
        workers=int(workers or payload["workers"]), candidates=tuple(payload["candidate_names"]),
        nominal_seeds=tuple(map(int, payload["nominal_seeds"])), pilot_seeds=tuple(map(int, payload["pilot_seeds"])),
        prediction_seeds=tuple(map(int, prediction["seeds"])), prediction_budget=int(prediction["budget"]),
        prediction_horizon=int(prediction["horizon"]), prediction_width=int(prediction["beam_width"]),
        predictors=tuple(prediction["predictors"]), timing_seeds=tuple(map(int, timing["seeds"])),
        warmup_start=int(timing["warmup_start"]), warmup_stop=int(timing["warmup_stop"]),
        measured_start=int(timing["measured_start"]), measured_stop=int(timing["measured_stop"]),
        methods=methods, imports=imports, native={str(k): str(v) for k, v in payload["native"].items()},
        fixed_hashes={str(k): str(v) for k, v in payload["fixed_trace_sha256"].items()},
        selection_decision=_path(payload["selection"]["decision"]),
        selection_decision_sha256=str(payload["selection"]["decision_sha256"]),
        selection_freeze=_path(payload["selection"]["freeze"]),
        selection_freeze_sha256=str(payload["selection"]["freeze_sha256"]),
        config_sha256=raw_sha256(path), smoke=smoke,
    )
    if not smoke:
        return config
    return replace(
        config, event_count=30, budgets=(config.budgets[0],), workers=min(config.workers, 1),
        nominal_seeds=(config.nominal_seeds[0],), pilot_seeds=(config.nominal_seeds[0],),
        prediction_seeds=(config.prediction_seeds[0],), timing_seeds=(config.nominal_seeds[0],),
        warmup_stop=5, measured_start=5, measured_stop=15,
    )


def timing_window_indices(config: V4Config) -> tuple[range, range]:
    return range(config.warmup_start, config.warmup_stop), range(config.measured_start, config.measured_stop)


def rotate_method_order(methods: Sequence[str], seed_index: int, budget_index: int) -> tuple[str, ...]:
    if not methods:
        raise ValueError("cannot rotate an empty method order")
    offset = (seed_index + budget_index) % len(methods)
    return tuple(methods[offset:]) + tuple(methods[:offset])


def _stage_path(config: V4Config, stage: str) -> Path:
    return config.output / stage / "manifest.json"


def _write_stage(config: V4Config, stage: str, extra: Mapping[str, object]) -> Path:
    path = _stage_path(config, stage)
    write_json_atomic({
        "schema": STAGE_SCHEMA, "experiment_id": "paper-evaluation-v4",
        "experiment_fingerprint": config.fingerprint, "config_sha256": config.config_sha256,
        "stage": stage, "status": "completed", **dict(extra),
    }, path)
    return path


def _load_stage(config: V4Config, stage: str) -> dict[str, object]:
    path = _stage_path(config, stage)
    payload = json.loads(path.read_text())
    if payload.get("schema") != STAGE_SCHEMA or payload.get("experiment_fingerprint") != config.fingerprint:
        raise ValueError(f"stale v4 stage: {path}")
    return payload


def _verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing pinned {label}: {path}")
    actual = raw_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}")


def _verify_native(config: V4Config) -> None:
    actual = {
        "binding_revision": BINDING_REVISION, "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE, "specification_sha256": ROBOT_ARM_SPEC_SHA256,
        "pzr_source_sha256": pzr_source_sha256(),
    }
    if actual != dict(config.native):
        raise ValueError(f"v4 native/source identity differs: {actual} != {dict(config.native)}")
    if dict(config.fixed_hashes) != {kind: ROBOT_ARM_TRACE_SHA256[kind] for kind in FIXED_TRACE_KINDS}:
        raise ValueError("fixed trace hashes differ")


def _verify_models(config: V4Config) -> dict[str, object]:
    _verify_file(config.selection_decision, config.selection_decision_sha256, "Vote3 selection decision")
    _verify_file(config.selection_freeze, config.selection_freeze_sha256, "Vote3/model freeze")
    freeze = json.loads(config.selection_freeze.read_text())
    model_count = 0
    for budget in config.budgets:
        members = freeze["vote3_members_by_budget"][str(budget)]
        if len(members) != 3 or tuple(int(item["optimizer_seed"]) for item in members) != (42, 1042, 2042):
            raise ValueError(f"Vote3 member identity differs at bound {budget}")
        for item in members:
            actual = model_sha256(Path(str(item["path"])))
            if actual != item["sha256"]:
                raise ValueError(f"Vote3 model hash mismatch at bound {budget}")
            model_count += 1
        g15 = freeze["g15_models_by_budget"][str(budget)]
        if int(g15["optimizer_seed"]) != 42 or model_sha256(Path(str(g15["path"]))) != g15["sha256"]:
            raise ValueError(f"G15 model identity differs at bound {budget}")
        model_count += 1
    return {"freeze": freeze, "verified_model_count": model_count}


def _run_release_preflight(config: V4Config) -> dict[str, object]:
    directory = config.output / "preflight"
    directory.mkdir(parents=True, exist_ok=True)
    junit = directory / "pytest.xml"
    command = [sys.executable, "-m", "pytest", "-m", "not rlola_parity", f"--junitxml={junit}"]
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (directory / "pytest.log").write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"release preflight failed; see {directory / 'pytest.log'}")
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError("preflight JUnit has no testsuite")
    counts = {key: int(suite.attrib.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")}
    if counts["failures"] or counts["errors"] or counts["skipped"]:
        raise ValueError(f"release preflight outcomes differ: {counts}")
    return {"command": command, **counts, "parity_marker_exclusion": "not rlola_parity"}


def run_preflight(config: V4Config) -> Path:
    _verify_native(config)
    models = _verify_models(config)
    imported = {}
    for spec in config.imports:
        _verify_file(spec.manifest, spec.manifest_sha256, f"{spec.name} manifest")
        _verify_file(spec.summary, spec.summary_sha256, f"{spec.name} summary")
        imported[spec.name] = {"manifest": str(spec.manifest), "manifest_sha256": spec.manifest_sha256, "summary": str(spec.summary), "summary_sha256": spec.summary_sha256}
    release = _run_release_preflight(config)
    return _write_stage(config, "preflight", {
        "native_identity_verified": True, "model_matrix_verified": True,
        "verified_model_count": models["verified_model_count"], "imports": imported,
        "release_preflight": release, "nominal_seeds_untouched": list(config.nominal_seeds),
    })


def _generate_store(output: Path, seeds: Sequence[int], event_count: int) -> object:
    if tuple(seeds) != tuple(range(min(seeds), min(seeds) + len(seeds))):
        raise ValueError("trace-store seeds must be contiguous")
    return generate_random_waypoint_trace_store(RandomWaypointTraceStoreConfig(
        output=output, event_count=event_count, conditions=("random_waypoint",),
        seed_start=min(seeds), seed_count=len(seeds),
    ))


def run_prepare(config: V4Config) -> Path:
    _load_stage(config, "preflight")
    nominal = _generate_store(config.output / "prepare/nominal-traces", config.nominal_seeds, config.event_count)
    prediction = _generate_store(config.output / "prepare/prediction-traces", config.prediction_seeds, config.event_count)
    imported_root = config.output / "prepare/imported"
    records = {}
    for spec in config.imports:
        target = imported_root / spec.name
        target.mkdir(parents=True, exist_ok=True)
        manifest_target, summary_target = target / "manifest.json", target / "summary.csv"
        shutil.copy2(spec.manifest, manifest_target)
        shutil.copy2(spec.summary, summary_target)
        _verify_file(manifest_target, spec.manifest_sha256, f"copied {spec.name} manifest")
        _verify_file(summary_target, spec.summary_sha256, f"copied {spec.name} summary")
        records[spec.name] = {"source_manifest": str(spec.manifest), "source_summary": str(spec.summary), "manifest": str(manifest_target), "summary": str(summary_target), "manifest_sha256": spec.manifest_sha256, "summary_sha256": spec.summary_sha256, "execution_origin": "verified_import"}
    freeze_target = config.output / "prepare/freeze.json"
    shutil.copy2(config.selection_freeze, freeze_target)
    return _write_stage(config, "prepare", {
        "nominal_trace_store": str(nominal.root), "nominal_trace_store_sha256": nominal.manifest_sha256,
        "prediction_trace_store": str(prediction.root), "prediction_trace_store_sha256": prediction.manifest_sha256,
        "freeze": str(freeze_target), "freeze_sha256": raw_sha256(freeze_target), "imports": records,
        "h_w_ablation_executed": False,
    })


def _traces(config: V4Config, which: Literal["nominal", "prediction"]) -> tuple[TraceRecord, ...]:
    prepare = _load_stage(config, "prepare")
    seeds = config.nominal_seeds if which == "nominal" else config.prediction_seeds
    store = load_random_waypoint_trace_store(Path(str(prepare[f"{which}_trace_store"])))
    records = []
    for seed in seeds:
        item = store.traces_for_seed(seed)[0]
        records.append(TraceRecord(item.trace_id, seed, tuple(item.trace.events), item.trace.metadata.trace_sha256))
    return tuple(records)


def _freeze(config: V4Config) -> dict[str, object]:
    return json.loads(Path(str(_load_stage(config, "prepare")["freeze"])).read_text())


def _model_paths(config: V4Config, method: MethodSpec, budget: int) -> tuple[Path, ...]:
    freeze = _freeze(config)
    if method.kind == "g15":
        return (Path(str(freeze["g15_models_by_budget"][str(budget)]["path"])),)
    if method.kind in ("vote3", "vote3_guarded"):
        return tuple(Path(str(item["path"])) for item in freeze["vote3_members_by_budget"][str(budget)])
    return ()


def _policy(config: V4Config, method: MethodSpec, budget: int, events: Sequence[RtlolaEvent]) -> object | None:
    paths = _model_paths(config, method, budget)
    if method.kind == "g15":
        return RtlolaReducerPolicy(ReducerPolicy.load(paths[0]), default_action_catalog(config.candidates))
    if method.kind in ("vote3", "vote3_guarded"):
        return vote_selection.VoteGuardPolicy(
            tuple(ReducerPolicy.load(path) for path in paths), events,
            guarded=method.kind == "vote3_guarded",
        )
    return None


def _reference_path(config: V4Config, trace: TraceRecord) -> Path:
    return config.output / f"prepare/references/{trace.trace_id}.json"


def _prepare_reference(config: V4Config, trace: TraceRecord) -> Path:
    path = _reference_path(config, trace)
    load_or_compute_reference(trace.events, scenario=scenario_by_name("robot_arm"), trace_kind=trace.trace_id, seed=trace.seed, cache_path=path, include_approximation=True)
    return path


def _job_identity(job: CellJob) -> dict[str, object]:
    return {
        "schema": CELL_SCHEMA, "experiment_fingerprint": job.config.fingerprint,
        "stage": job.stage, "trace_id": job.trace.trace_id, "trace_sha256": job.trace.sha256,
        "seed": job.trace.seed, "event_count": len(job.trace.events), "budget": job.budget,
        "method": job.method.name, "runtime_method": job.method.runtime_method,
        "model_sha256": [model_sha256(path) for path in _model_paths(job.config, job.method, job.budget)],
        "reference_sha256": raw_sha256(job.reference),
    }


def _execute_cell(job: CellJob) -> dict[str, object]:
    manifest_path, summary_path = job.directory / "manifest.json", job.directory / "summary.csv"
    identity = _job_identity(job)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity") != identity or manifest.get("summary_sha256") != raw_sha256(summary_path):
            raise ValueError(f"stale v4 cell: {job.directory}")
        return pd.read_csv(summary_path).iloc[0].to_dict()
    policy = _policy(job.config, job.method, job.budget, job.trace.events)
    benchmark = RtlolaBenchmarkConfig(
        scenario="robot_arm", trace_kind="random_waypoint", length=len(job.trace.events), budget=job.budget,
        horizon=job.method.horizon, beam_width=max(1, job.method.beam_width), prediction_step_seconds=0.1,
        seeds=1, methods=[job.method.runtime_method], reference_mode="exact", mpc_reference="rollout",
        reference_cache=str(job.reference), mpc_candidate_names=list(job.config.candidates), output_dir=str(job.directory),
    )
    started = perf_counter()
    result = run_event_trace_benchmark(benchmark, job.trace.events, trace_kind=job.trace.trace_id, seed=job.trace.seed, method=job.method.runtime_method, policy=policy)
    if result.failures:
        failure = result.failures[0]
        if str(failure.failure_type) != "RtlolaNoFeasibleAction":
            raise RuntimeError(f"native/infrastructure failure in {job.directory}: {failure.failure_type}: {failure.message}")
        row = {"status": "fallback_failed", "failure_type": failure.failure_type, "failure_message": failure.message}
    elif len(result.summary) == 1:
        row = result.summary.iloc[0].to_dict()
        if int(row.get("fallback_count", 0)) > 0:
            row.update({
                "status": "fallback_failed",
                "failure_type": "IntervalFallback",
                "failure_message": "ordinary run used interval fallback",
            })
        else:
            row["status"] = "completed"
    else:
        raise RuntimeError(f"empty benchmark result: {job.directory}")
    row.update({
        "method": job.method.name, "method_label": job.method.label, "runtime_method": job.method.runtime_method,
        "seed": job.trace.seed, "budget": job.budget, "trace_id": job.trace.trace_id,
        "trace_sha256": job.trace.sha256, "event_count": len(job.trace.events),
        "cell_elapsed_seconds": perf_counter() - started,
    })
    job.directory.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(pd.DataFrame([row]), summary_path)
    if not result.timeseries.empty:
        series = result.timeseries.copy(); series["method"] = job.method.name
        write_csv_atomic(series, job.directory / "timeseries.csv")
    if not result.failed_timeseries.empty:
        write_csv_atomic(result.failed_timeseries, job.directory / "failed_timeseries.csv")
    if policy is not None and hasattr(policy, "diagnostics"):
        write_csv_atomic(pd.DataFrame(policy.diagnostics), job.directory / "decisions.csv")
    write_json_atomic({"schema": CELL_SCHEMA, "identity": identity, "status": row["status"], "summary_sha256": raw_sha256(summary_path)}, manifest_path)
    return row


def _jobs(config: V4Config, stage: str, traces: Sequence[TraceRecord], budgets: Sequence[int], methods: Sequence[MethodSpec]) -> list[CellJob]:
    jobs = []
    for trace in traces:
        reference = _prepare_reference(config, trace)
        for budget in budgets:
            for method in methods:
                jobs.append(CellJob(config, stage, trace, int(budget), method, reference, config.output / f"cells/{trace.trace_id}/budget-{budget}/{method.name}"))
    return jobs


def _run_jobs(config: V4Config, jobs: Sequence[CellJob]) -> pd.DataFrame:
    if config.workers == 1:
        rows = [_execute_cell(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=config.workers, mp_context=get_context("spawn"), max_tasks_per_child=1) as pool:
            rows = list(pool.map(_execute_cell, jobs))
    return pd.DataFrame(rows)


def _write_matrix_stage(config: V4Config, stage: str, summary: pd.DataFrame, expected: int, started: float) -> Path:
    if len(summary) != expected:
        raise ValueError(f"{stage} has {len(summary)} cells, expected {expected}")
    if summary[["seed", "budget", "method"]].duplicated().any():
        raise ValueError(f"{stage} contains duplicate cells")
    path = config.output / stage / "summary.csv"
    write_csv_atomic(summary, path)
    return _write_stage(config, stage, {"cell_count": len(summary), "expected_cell_count": expected, "failure_count": int((summary["status"] != "completed").sum()), "summary": str(path), "summary_sha256": raw_sha256(path), "workers": config.workers, "wall_seconds": perf_counter() - started})


def run_pilot(config: V4Config) -> Path:
    _load_stage(config, "prepare")
    traces = tuple(trace for trace in _traces(config, "nominal") if trace.seed in config.pilot_seeds)
    started = perf_counter()
    summary = _run_jobs(config, _jobs(config, "nominal", traces, config.budgets, config.methods))
    path = config.output / "pilot/summary.csv"; write_csv_atomic(summary, path)
    cpu = float(summary["cell_elapsed_seconds"].sum())
    return _write_stage(config, "pilot", {"cell_count": len(summary), "summary": str(path), "reused_by_nominal": True, "projected_nominal_wall_hours": cpu / len(summary) * config.expected_nominal_cells / config.workers / 3600.0, "wall_seconds": perf_counter() - started})


def run_nominal(config: V4Config) -> Path:
    _load_stage(config, "pilot")
    started = perf_counter()
    summary = _run_jobs(config, _jobs(config, "nominal", _traces(config, "nominal"), config.budgets, config.methods))
    return _write_matrix_stage(config, "nominal", summary, config.expected_nominal_cells, started)


def run_prediction_ablation(config: V4Config) -> Path:
    _load_stage(config, "nominal")
    methods = tuple(MethodSpec(PREDICTOR_METHODS[name], name.title(), "mpc", PREDICTOR_METHODS[name], config.prediction_horizon, config.prediction_width) for name in config.predictors)
    started = perf_counter()
    summary = _run_jobs(config, _jobs(config, "prediction-ablation", _traces(config, "prediction"), (config.prediction_budget,), methods))
    summary["predictor"] = summary["runtime_method"].map({value: key for key, value in PREDICTOR_METHODS.items()})
    return _write_matrix_stage(config, "prediction-ablation", summary, config.expected_prediction_cells, started)


def _timing_cell(config: V4Config, trace: TraceRecord, budget: int, method: MethodSpec, order_index: int) -> dict[str, object]:
    # Exact-reference construction and artifact I/O occur before/after this call.
    # The benchmark runs the warm-up and measured prefix in one monitor.  The
    # deployment measure is the exact per-event selection/commit clock.  The
    # instrumented event-loop measure additionally apportions benchmark record
    # construction overhead to the fixed steady-state window.
    events = trace.events[:config.measured_stop]
    policy = _policy(config, method, budget, events)
    benchmark = RtlolaBenchmarkConfig(
        scenario="robot_arm", trace_kind="random_waypoint", length=len(events), budget=budget,
        horizon=method.horizon, beam_width=max(1, method.beam_width), prediction_step_seconds=0.1,
        seeds=1, methods=[method.runtime_method], reference_mode="off", mpc_reference="rollout",
        mpc_candidate_names=list(config.candidates),
    )
    diagnostic_function = benchmark_module.prediction_diagnostics
    benchmark_module.prediction_diagnostics = lambda *_args, **_kwargs: ()
    try:
        result = run_event_trace_benchmark(benchmark, events, trace_kind=trace.trace_id, seed=trace.seed, method=method.runtime_method, policy=policy)
    finally:
        benchmark_module.prediction_diagnostics = diagnostic_function
    if result.failures or len(result.summary) != 1:
        failure = result.failures[0] if result.failures else None
        raise RuntimeError(f"timing failure for {trace.seed}/{budget}/{method.name}: {failure}")
    series = result.timeseries.sort_values("step")
    measured = series[(series["step"] >= config.measured_start) & (series["step"] < config.measured_stop)]
    expected_events = config.measured_stop - config.measured_start
    if len(measured) != expected_events:
        raise ValueError("timing measured window differs")
    deployment_ms = float(measured["decision_time_ms"].sum())
    event_loop_ms = float(result.summary.iloc[0]["event_loop_time_ms"])
    all_instrumented_ms = float(series["decision_time_ms"].sum())
    overhead_per_event_ms = max(0.0, event_loop_ms - all_instrumented_ms) / len(events)
    instrumented_ms = deployment_ms + overhead_per_event_ms * expected_events
    return {
        "seed": trace.seed, "budget": budget, "method": method.name, "method_label": method.label,
        "order_index": order_index, "warmup_start": config.warmup_start, "warmup_stop": config.warmup_stop,
        "measured_start": config.measured_start, "measured_stop": config.measured_stop,
        "warmup_event_count": config.warmup_stop - config.warmup_start, "measured_event_count": expected_events,
        "instrumented_event_loop_latency_ms": instrumented_ms / expected_events,
        "deployment_path_latency_ms": deployment_ms / expected_events,
        "deployment_throughput_events_per_second": expected_events * 1000.0 / deployment_ms,
        "deployment_measurement": "steady-state reducer selection and native commit",
        "reference_metrics_included": False, "prediction_diagnostics_included": False, "artifact_io_included": False,
        "status": "completed",
    }


def run_runtime(config: V4Config) -> Path:
    _load_stage(config, "prediction-ablation")
    nominal = pd.read_csv(str(_load_stage(config, "nominal")["summary"]))
    prediction = pd.read_csv(str(_load_stage(config, "prediction-ablation")["summary"]))
    if len(nominal) != config.expected_nominal_cells or len(prediction) != config.expected_prediction_cells:
        raise ValueError("scientific matrices are incomplete before timing")
    completed = nominal[nominal["status"] == "completed"]
    if not (pd.to_numeric(completed["false_negative_count"]) == 0).all():
        raise ValueError("scientific validation found a false negative before timing")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    traces = tuple(trace for trace in _traces(config, "nominal") if trace.seed in config.timing_seeds)
    by_name = {method.name: method for method in config.methods}
    rows = []
    started = perf_counter()
    for seed_index, trace in enumerate(traces):
        for budget_index, budget in enumerate(config.budgets):
            order = rotate_method_order(METHOD_NAMES, seed_index, budget_index)
            for order_index, name in enumerate(order):
                rows.append(_timing_cell(config, trace, budget, by_name[name], order_index))
    raw = pd.DataFrame(rows)
    if len(raw) != config.expected_timing_cells:
        raise ValueError("timing cell count differs")
    summary = raw.groupby(["budget", "method", "method_label"], sort=True).agg(
        cell_count=("seed", "size"), median_deployment_path_latency_ms=("deployment_path_latency_ms", "median"),
        p05_deployment_path_latency_ms=("deployment_path_latency_ms", lambda values: float(np.quantile(values, 0.05))),
        p95_deployment_path_latency_ms=("deployment_path_latency_ms", lambda values: float(np.quantile(values, 0.95))),
        median_instrumented_event_loop_latency_ms=("instrumented_event_loop_latency_ms", "median"),
        median_deployment_throughput_events_per_second=("deployment_throughput_events_per_second", "median"),
    ).reset_index()
    directory = config.output / "runtime"; write_csv_atomic(raw, directory / "timing_cells.csv"); write_csv_atomic(summary, directory / "summary.csv")
    return _write_stage(config, "runtime", {"cell_count": len(raw), "expected_cell_count": config.expected_timing_cells, "summary_point_count": len(summary), "timing_cells": str(directory / "timing_cells.csv"), "summary": str(directory / "summary.csv"), "workers": 1, "native_threads": 1, "scientific_validation_completed_before_timing": True, "method_order_rotation": "(seed_index + budget_index) mod 10", "wall_seconds": perf_counter() - started})


def _availability(nominal: pd.DataFrame) -> pd.DataFrame:
    return nominal.groupby(["method", "method_label"], sort=False).agg(cell_count=("status", "size"), completed_count=("status", lambda values: int((values == "completed").sum())), unavailable_count=("status", lambda values: int((values != "completed").sum()))).reset_index()


def _metric_aggregates(nominal: pd.DataFrame) -> pd.DataFrame:
    completed = nominal[nominal["status"] == "completed"].copy()
    return completed.groupby(["method", "method_label", "budget"], sort=False).agg(
        valid_count=("seed", "size"), mean_fpr=("fpr", "mean"), max_fnr=("fnr", "max"), false_negative_count=("false_negative_count", "sum"),
        median_mean_approx_loss=("mean_approx_loss", "median"), mean_mean_approx_loss=("mean_approx_loss", "mean"),
        max_generator_count=("max_generator_count", "max"), max_active_dynamic_generator_count=("max_active_dynamic_generator_count", "max"),
        fallback_count=("fallback_count", "sum"), infeasible_candidate_count=("infeasible_candidate_count", "sum"),
    ).reset_index()


def _severe_tails(nominal: pd.DataFrame) -> pd.DataFrame:
    complete = nominal[nominal["status"] == "completed"]
    pivot = complete.pivot(index=["seed", "budget"], columns="method", values="mean_approx_loss")
    reference = pivot["mpc_terminal_beam_predictive_linear"]
    rows = []
    for method in METHOD_NAMES:
        if method not in pivot:
            continue
        ratio = pivot[method] / reference
        for (seed, budget), value in ratio[ratio > SEVERE_TAIL_MULTIPLIER].items():
            rows.append({"seed": seed, "budget": budget, "method": method, "loss_ratio_vs_mpc_l": value})
    return pd.DataFrame(rows, columns=("seed", "budget", "method", "loss_ratio_vs_mpc_l"))


def _guard_decisions(config: V4Config) -> pd.DataFrame:
    frames = []
    for path in sorted((config.output / "cells").rglob("dagger05_vote3_guarded/decisions.csv")):
        frame = pd.read_csv(path)
        trace_part = next(part for part in path.parts if part.startswith("random_waypoint:seed-"))
        frame["seed"] = int(trace_part.rsplit("-", 1)[1])
        frame["budget"] = int(next(part.split("-", 1)[1] for part in path.parts if part.startswith("budget-")))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _reducer_composition(config: V4Config) -> pd.DataFrame:
    counts: dict[tuple[str, int, str], int] = {}
    totals: dict[tuple[str, int], int] = {}
    for path in sorted((config.output / "cells").rglob("timeseries.csv")):
        if "random_waypoint:seed-" not in path.as_posix():
            continue
        trace_part = next(part for part in path.parts if part.startswith("random_waypoint:seed-"))
        if int(trace_part.rsplit("-", 1)[1]) not in config.nominal_seeds:
            continue
        frame = pd.read_csv(path, usecols=("method", "budget", "pre_generator_count", "reducer_used"))
        over_bound = frame[frame["pre_generator_count"].astype(int) > frame["budget"].astype(int)]
        if over_bound.empty:
            continue
        method = str(over_bound["method"].iloc[0]); budget = int(over_bound["budget"].iloc[0])
        totals[(method, budget)] = totals.get((method, budget), 0) + len(over_bound)
        for reducer, count in over_bound["reducer_used"].astype(str).value_counts().items():
            key = (method, budget, str(reducer)); counts[key] = counts.get(key, 0) + int(count)
    return pd.DataFrame([
        {"method": method, "budget": budget, "reducer_used": reducer, "decision_count": count,
         "decision_share": count / totals[(method, budget)]}
        for (method, budget, reducer), count in sorted(counts.items())
    ])


def _imported_scientific_tables(prepare: Mapping[str, object]) -> dict[str, pd.DataFrame]:
    records = prepare["imports"]
    v3_fixed = pd.read_csv(str(records["v3_fixed"]["summary"]))
    vote_fixed = pd.read_csv(str(records["vote3_fixed"]["summary"]))
    vote_fixed = vote_fixed[vote_fixed["method"].isin(("dagger05_vote3", "dagger05_vote3_guarded"))]
    fixed = pd.concat((v3_fixed, vote_fixed), ignore_index=True)
    if len(fixed) != 280 or set(fixed["method"]) != set(METHOD_NAMES):
        raise ValueError("imported fixed-method matrix differs")
    return {
        "fixed_cells.csv": fixed,
        "h_w_ablation.csv": pd.read_csv(str(records["v3_ablation"]["summary"])),
        "objective_comparison.csv": pd.read_csv(str(records["v3_objective_comparison"]["summary"])),
    }


def run_report(config: V4Config) -> Path:
    nominal_manifest = _load_stage(config, "nominal"); prediction_manifest = _load_stage(config, "prediction-ablation"); runtime_manifest = _load_stage(config, "runtime"); prepare = _load_stage(config, "prepare")
    nominal = pd.read_csv(str(nominal_manifest["summary"])); prediction = pd.read_csv(str(prediction_manifest["summary"])); timing = pd.read_csv(str(runtime_manifest["timing_cells"])); timing_summary = pd.read_csv(str(runtime_manifest["summary"]))
    directory = config.output / "report/artifacts"; directory.mkdir(parents=True, exist_ok=True)
    availability = _availability(nominal); metrics = _metric_aggregates(nominal); severe = _severe_tails(nominal); decisions = _guard_decisions(config); composition = _reducer_composition(config)
    guard_flow = pd.DataFrame()
    if not decisions.empty:
        guard_flow = decisions.groupby(["budget", "selected_action"], sort=True).agg(decision_count=("step", "size"), guard_count=("guard_invoked", "sum"), override_count=("guard_override", "sum")).reset_index()
    tables = {
        "nominal_cells.csv": nominal, "availability.csv": availability, "nominal_metrics.csv": metrics,
        "severe_tail_cells.csv": severe, "guard_decisions.csv": decisions, "guard_flow.csv": guard_flow,
        "reducer_composition.csv": composition,
        "predictor_ablation.csv": prediction, "timing_cells.csv": timing, "timing_distributions.csv": timing_summary,
        **_imported_scientific_tables(prepare),
    }
    for name, frame in tables.items(): write_csv_atomic(frame, directory / name)
    imported = {name: record for name, record in dict(prepare["imports"]).items()}
    write_json_atomic({"schema": "pzr.paper-evaluation-import-provenance.v4", "artifacts": imported, "fixed_cases_are_controlled_descriptive_traces": True}, directory / "imported_provenance.json")
    complete = nominal[nominal["status"] == "completed"]
    claims = {
        "schema": "pzr.paper-evaluation-claims.v4", "nominal_seed_first": min(config.nominal_seeds), "nominal_seed_last": max(config.nominal_seeds),
        "nominal_cell_count": len(nominal), "nominal_unavailable_count": int((nominal["status"] != "completed").sum()),
        "false_negative_count": int(pd.to_numeric(complete["false_negative_count"], errors="raise").sum()),
        "all_completed_cells_zero_false_negatives": bool((pd.to_numeric(complete["false_negative_count"]) == 0).all()),
        "maximum_dynamic_generator_count": int(pd.to_numeric(complete["max_generator_count"], errors="raise").max()),
        "maximum_post_event_excess_over_transform_bound": int((pd.to_numeric(complete["max_generator_count"]) - pd.to_numeric(complete["budget"])).max()),
        "generator_counts_bounded_independent_of_trace_length": bool((pd.to_numeric(complete["max_generator_count"]) <= pd.to_numeric(complete["budget"]) + ROBOT_ARM_FRESH_DYNAMIC_ROWS + pd.to_numeric(complete["max_logical_dynamic_dimension"])).all()),
        "prediction_ablation_cell_count": len(prediction), "timing_cell_count": len(timing), "severe_tail_cell_count": len(severe),
        "mpc_f_contract": "two-event full-width terminal-loss teacher (current event plus one recorded future event)",
        "learned_teacher_contract": "full-width terminal-loss teacher; not MPC-L",
        "runtime_primary_measure": "deployment_path_latency_ms over events 100--299",
    }
    write_json_atomic(claims, directory / "claim_values.json")
    hashes = {path.name: raw_sha256(path) for path in sorted(directory.iterdir()) if path.is_file()}
    write_json_atomic(hashes, directory / "artifact_hashes.json")
    return _write_stage(config, "report", {"artifact_directory": str(directory), "artifact_count": len(hashes) + 1, "artifact_hashes": str(directory / "artifact_hashes.json"), "claim_values": str(directory / "claim_values.json"), "no_placeholders": True})


def run_validate(config: V4Config) -> Path:
    for stage in STAGES[:-1]: _load_stage(config, stage)
    _verify_native(config); _verify_models(config)
    prepare = _load_stage(config, "prepare")
    for spec in config.imports:
        record = prepare["imports"][spec.name]
        _verify_file(Path(str(record["manifest"])), spec.manifest_sha256, f"imported {spec.name} manifest")
        _verify_file(Path(str(record["summary"])), spec.summary_sha256, f"imported {spec.name} summary")
    nominal = pd.read_csv(str(_load_stage(config, "nominal")["summary"])); prediction = pd.read_csv(str(_load_stage(config, "prediction-ablation")["summary"])); timing = pd.read_csv(str(_load_stage(config, "runtime")["timing_cells"])); claims = json.loads(Path(str(_load_stage(config, "report")["claim_values"])).read_text())
    if len(nominal) != config.expected_nominal_cells or set(nominal["method"]) != set(METHOD_NAMES) or set(nominal["seed"].astype(int)) != set(config.nominal_seeds):
        raise ValueError("nominal matrix contract differs")
    if len(prediction) != config.expected_prediction_cells or set(prediction["predictor"]) != set(config.predictors):
        raise ValueError("prediction matrix contract differs")
    if len(timing) != config.expected_timing_cells or timing[["seed", "budget", "method"]].duplicated().any():
        raise ValueError("timing matrix contract differs")
    completed = nominal[nominal["status"] == "completed"]
    if not (pd.to_numeric(completed["false_negative_count"]) == 0).all():
        raise ValueError("a completed nominal cell has a false negative")
    if not (pd.to_numeric(completed["max_generator_count"]) <= pd.to_numeric(completed["budget"]) + ROBOT_ARM_FRESH_DYNAMIC_ROWS + pd.to_numeric(completed["max_logical_dynamic_dimension"])).all():
        raise ValueError("a completed nominal cell exceeds the post-event generator bound")
    if not claims["all_completed_cells_zero_false_negatives"] or not claims["generator_counts_bounded_independent_of_trace_length"]:
        raise ValueError("machine-readable claims differ")
    report = _load_stage(config, "report"); hashes = json.loads(Path(str(report["artifact_hashes"])).read_text()); root = Path(str(report["artifact_hashes"])).parent
    for name, expected in hashes.items(): _verify_file(root / name, expected, f"report artifact {name}")
    return _write_stage(config, "validate", {"nominal_cell_count": len(nominal), "prediction_cell_count": len(prediction), "timing_cell_count": len(timing), "zero_false_negatives": True, "bounded_generator_counts_independent_of_trace_length": True, "post_event_bound": "transform bound + 35 fresh dynamic generators + logical dimension", "imports_unchanged": True, "report_hashes_verified": True})


def run_stage(config: V4Config, stage: str) -> Path:
    path = _stage_path(config, stage)
    if path.is_file():
        _load_stage(config, stage); print(f"skip completed v4 stage: {stage}", flush=True); return path
    functions = {"preflight": run_preflight, "prepare": run_prepare, "pilot": run_pilot, "nominal": run_nominal, "prediction-ablation": run_prediction_ablation, "runtime": run_runtime, "report": run_report, "validate": run_validate}
    print(f"start v4 stage: {stage}", flush=True); result = functions[stage](config); print(f"complete v4 stage: {stage}", flush=True); return result


def run_all(config: V4Config) -> Path:
    for stage in RUN_STAGES:
        run_stage(config, stage)
    return _stage_path(config, RUN_STAGES[-1])


def status(config: V4Config) -> dict[str, object]:
    stages = {}
    for stage in STAGES:
        try: stages[stage] = "completed" if _stage_path(config, stage).is_file() and _load_stage(config, stage) else "missing"
        except ValueError as exc: stages[stage] = f"stale: {exc}"
    return {
        "output": str(config.output),
        "smoke": config.smoke,
        "expected_nominal_cells": config.expected_nominal_cells,
        "expected_prediction_cells": config.expected_prediction_cells,
        "expected_timing_cells": config.expected_timing_cells,
        "default_run_stages": list(RUN_STAGES),
        "workstation_timing_is_explicit_only": True,
        "stages": stages,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=(*STAGES, "run", "status")); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); parser.add_argument("--output", type=Path); parser.add_argument("--workers", type=int); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args(argv)
    config = load_config(args.config, output=args.output, smoke=args.smoke, workers=args.workers)
    if args.command == "status": print(json.dumps(status(config), indent=2)); return 0
    path = run_all(config) if args.command == "run" else run_stage(config, args.command); print(f"v4 stage complete: {path}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
