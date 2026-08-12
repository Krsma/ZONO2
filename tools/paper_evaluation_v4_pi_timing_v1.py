#!/usr/bin/env python3
"""Safety-isolated Raspberry Pi timing add-on for paper evaluation v4.

The parent v4 experiment is a read-only scientific input.  This tool copies a
verified subset into a portable bundle, invokes the existing benchmark and
policy implementations unchanged, and writes only to new versioned roots.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import importlib.util
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from time import perf_counter_ns
from typing import Any, Mapping, Sequence
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
    require_binding,
)
from pzr.rtlola.learned_policy import RtlolaReducerPolicy
from pzr.rtlola.learning_traces import load_random_waypoint_trace_store
from pzr.rtlola.robot_arm import ROBOT_ARM_SPEC_SHA256

import prp_tail_vote_guard_exploratory as vote_selection


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/paper_evaluation_v4_pi_timing_v1.yaml"
CONFIG_SCHEMA = "pzr.paper-evaluation-v4-pi-timing-config.v1"
BUNDLE_SCHEMA = "pzr.paper-evaluation-v4-pi-timing-bundle.v1"
CELL_SCHEMA = "pzr.paper-evaluation-v4-pi-timing-cell.v1"
RESULT_SCHEMA = "pzr.paper-evaluation-v4-pi-timing-result.v1"
PROFILE_SCHEMA = "pzr.paper-evaluation-v4-pi-profile.v1"
COMBINED_SCHEMA = "pzr.paper-evaluation-v4-final-report.v1"

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
ISOLATED_RESULT_ROOTS = (
    "paper-evaluation-v4-pi-timing-v1",
    "paper-evaluation-v4-final-report-v1",
)
PROTECTED_RESULT_ROOTS = (
    "paper-evaluation-v2",
    "paper-evaluation-v3",
    "paper-evaluation-v4",
    "dart-rescue-v1",
)
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TORCH_NUM_THREADS",
)
_POLICY_UNSET = object()


@dataclass(frozen=True)
class PiTimingConfig:
    source: Path
    output: Path
    combined_output: Path
    parent_config: Path
    parent_config_sha256: str
    parent_runner: Path
    parent_runner_sha256: str
    parent_output: Path
    parent_source_sha256: str
    dependency_hashes: Mapping[Path, str]
    seeds: tuple[int, ...]
    budgets: tuple[int, ...]
    methods: tuple[str, ...]
    warmup_start: int
    warmup_stop: int
    measured_start: int
    measured_stop: int
    cpu_core: int
    profile_seed: int
    profile_budget: int
    on_time_targets: tuple[float, ...]
    native: Mapping[str, str]
    environment: Mapping[str, str]
    config_sha256: str
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.methods != METHOD_NAMES:
            raise ValueError("Pi timing method matrix differs from v4")
        if not self.smoke:
            if self.seeds != tuple(range(348, 353)):
                raise ValueError("Pi timing seeds must be 348--352")
            if self.budgets != (40, 80, 120, 150, 200, 250, 500):
                raise ValueError("Pi timing bounds differ from v4")
        if not (
            self.warmup_start == 0
            < self.warmup_stop == self.measured_start
            < self.measured_stop
        ):
            raise ValueError("Pi timing windows must be contiguous and non-empty")
        if self.profile_seed not in self.seeds or self.profile_budget not in self.budgets:
            raise ValueError("profile cell must belong to the timing matrix")
        if self.cpu_core < 0:
            raise ValueError("Pi timing CPU core must be non-negative")
        if self.on_time_targets != (0.95, 0.99, 1.0):
            raise ValueError("on-time targets differ from the paper contract")
        _assert_isolated_output(self.output)
        _assert_isolated_output(self.combined_output)

    @property
    def expected_cells(self) -> int:
        return len(self.seeds) * len(self.budgets) * len(self.methods)

    @property
    def measured_event_count(self) -> int:
        return self.measured_stop - self.measured_start

    @property
    def identity(self) -> dict[str, object]:
        return {
            "schema": CONFIG_SCHEMA,
            "config_sha256": self.config_sha256,
            "parent_config_sha256": self.parent_config_sha256,
            "parent_runner_sha256": self.parent_runner_sha256,
            "parent_source_sha256": self.parent_source_sha256,
            "dependency_hashes": {
                str(path.relative_to(ROOT)): value
                for path, value in sorted(
                    self.dependency_hashes.items(), key=lambda item: str(item[0])
                )
            },
            "seeds": list(self.seeds),
            "budgets": list(self.budgets),
            "methods": list(self.methods),
            "warmup": [self.warmup_start, self.warmup_stop],
            "measured": [self.measured_start, self.measured_stop],
            "profile": [self.profile_seed, self.profile_budget],
            "cpu_core": self.cpu_core,
            "workers": 1,
            "native_threads": 1,
            "method_order_rotation": "(seed_index + budget_index) mod 10",
            "on_time_targets": list(self.on_time_targets),
            "native": dict(self.native),
            "environment": dict(self.environment),
            "smoke": self.smoke,
        }

    @property
    def identity_sha256(self) -> str:
        return payload_sha256(self.identity)


def raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: object, *, root: Path = ROOT) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _assert_isolated_output(path: Path) -> None:
    resolved = path.resolve()
    results = (ROOT / "results").resolve()
    if resolved.parent != results or resolved.name not in ISOLATED_RESULT_ROOTS:
        raise ValueError(
            "Pi timing outputs must use one of the two isolated versioned roots"
        )
    if resolved.name in PROTECTED_RESULT_ROOTS:
        raise ValueError("Pi timing cannot write to a protected result root")


def _assert_runtime_output(path: Path) -> None:
    """Reject accidental execution into a parent experiment directory."""
    resolved = path.resolve()
    if resolved.name != "paper-evaluation-v4-pi-timing-v1":
        raise ValueError("timing run output must use the isolated add-on directory name")
    for protected in PROTECTED_RESULT_ROOTS:
        protected_root = (ROOT / "results" / protected).resolve()
        if resolved == protected_root or protected_root in resolved.parents:
            raise ValueError("timing run output overlaps a protected result root")


def load_config(
    path: Path = DEFAULT_CONFIG,
    *,
    smoke: bool = False,
) -> PiTimingConfig:
    payload = yaml.safe_load(path.read_text())
    if payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"unsupported Pi timing config: {payload.get('schema')}")
    parent = payload["parent"]
    timing = payload["timing"]
    config = PiTimingConfig(
        source=path.resolve(),
        output=_resolve(payload["output_root"]),
        combined_output=_resolve(payload["combined_report_root"]),
        parent_config=_resolve(parent["config"]),
        parent_config_sha256=str(parent["config_sha256"]),
        parent_runner=_resolve(parent["runner"]),
        parent_runner_sha256=str(parent["runner_sha256"]),
        parent_output=_resolve(parent["output_root"]),
        parent_source_sha256=str(parent["pzr_source_sha256"]),
        dependency_hashes={
            _resolve(name): str(value)
            for name, value in parent["tool_dependencies"].items()
        },
        seeds=tuple(map(int, timing["seeds"])),
        budgets=tuple(map(int, timing["budgets"])),
        methods=tuple(map(str, timing["methods"])),
        warmup_start=int(timing["warmup_start"]),
        warmup_stop=int(timing["warmup_stop"]),
        measured_start=int(timing["measured_start"]),
        measured_stop=int(timing["measured_stop"]),
        cpu_core=int(timing["cpu_core"]),
        profile_seed=int(timing["profile_seed"]),
        profile_budget=int(timing["profile_budget"]),
        on_time_targets=tuple(map(float, timing["on_time_targets"])),
        native={str(key): str(value) for key, value in payload["native"].items()},
        environment={
            str(key): str(value) for key, value in payload["environment"].items()
        },
        config_sha256=raw_sha256(path),
        smoke=smoke,
    )
    if not smoke:
        return config
    return replace(
        config,
        seeds=(config.seeds[0],),
        budgets=(config.budgets[0],),
        warmup_stop=5,
        measured_start=5,
        measured_stop=15,
        profile_seed=config.seeds[0],
        profile_budget=config.budgets[0],
        smoke=True,
    )


def verify_parent_pins(config: PiTimingConfig) -> dict[str, str]:
    checks = {
        config.parent_config: config.parent_config_sha256,
        config.parent_runner: config.parent_runner_sha256,
        **config.dependency_hashes,
    }
    verified: dict[str, str] = {}
    for path, expected in checks.items():
        if not path.is_file():
            raise ValueError(f"pinned parent input is missing: {path}")
        actual = raw_sha256(path)
        if actual != expected:
            raise ValueError(
                f"read-only parent hash mismatch: {path} ({actual} != {expected})"
            )
        verified[str(path.relative_to(ROOT))] = actual
    source_hash = pzr_source_sha256()
    if source_hash != config.parent_source_sha256:
        raise ValueError(
            "read-only scientific source hash mismatch: "
            f"{source_hash} != {config.parent_source_sha256}"
        )
    if ROBOT_ARM_SPEC_SHA256 != config.native["specification_sha256"]:
        raise ValueError("packaged robot-arm specification hash differs")
    return verified


def _import_parent(config: PiTimingConfig) -> Any:
    module_name = "paper_evaluation_v4_pi_parent_read_only"
    spec = importlib.util.spec_from_file_location(module_name, config.parent_runner)
    if spec is None or spec.loader is None:
        raise ValueError("could not import the pinned v4 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _tree_hashes(root: Path, *, exclude: Sequence[Path] = ()) -> dict[str, str]:
    excluded = {path.resolve() for path in exclude}
    return {
        str(path.relative_to(root)): raw_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.resolve() not in excluded
    }


def _verify_tree(root: Path, files: Mapping[str, str]) -> None:
    expected = dict(files)
    actual_paths = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != (root / "manifest.json").resolve()
    }
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        extra = sorted(actual_paths - set(expected))
        raise ValueError(f"bundle file set differs (missing={missing}, extra={extra})")
    for relative, expected_hash in expected.items():
        actual_hash = raw_sha256(root / relative)
        if actual_hash != expected_hash:
            raise ValueError(f"bundle file hash mismatch: {relative}")


def _seal_identity(payload: Mapping[str, object], field: str) -> dict[str, object]:
    sealed = dict(payload)
    sealed.pop(field, None)
    sealed[field] = payload_sha256(sealed)
    return sealed


def _verify_identity(payload: Mapping[str, object], field: str) -> None:
    expected = str(payload.get(field, ""))
    unsealed = dict(payload)
    unsealed.pop(field, None)
    if not expected or payload_sha256(unsealed) != expected:
        raise ValueError(f"artifact identity hash differs: {field}")


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"unsafe archive path: {member.name}")
        handle.extractall(destination, filter="data")


def _copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(".git", "target", "__pycache__", "*.pyc"),
    )


def build_bundle(config: PiTimingConfig, archive: Path) -> Path:
    """Create a portable archive without changing any parent artifact."""
    before = verify_parent_pins(config)
    parent = _import_parent(config)
    parent_config = parent.load_config(config.parent_config)
    prepare = parent._load_stage(parent_config, "prepare")
    nominal_manifest = parent._load_stage(parent_config, "nominal")
    prediction_manifest = parent._load_stage(parent_config, "prediction-ablation")
    nominal = pd.read_csv(str(nominal_manifest["summary"]))
    prediction = pd.read_csv(str(prediction_manifest["summary"]))
    if len(nominal) != parent_config.expected_nominal_cells:
        raise ValueError("parent nominal matrix is incomplete")
    if len(prediction) != parent_config.expected_prediction_cells:
        raise ValueError("parent predictor matrix is incomplete")
    completed = nominal[nominal["status"] == "completed"]
    if not (pd.to_numeric(completed["false_negative_count"]) == 0).all():
        raise ValueError("parent nominal matrix contains a false negative")

    archive = archive.resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pzr-pi-bundle-") as temporary:
        bundle = Path(temporary) / "paper-evaluation-v4-pi-timing-v1-bundle"
        bundle.mkdir()
        (bundle / "experiments").mkdir()
        (bundle / "tools").mkdir()
        (bundle / "tests").mkdir()
        shutil.copy2(config.source, bundle / "experiments" / config.source.name)
        shutil.copy2(config.parent_config, bundle / "experiments" / config.parent_config.name)
        tool_paths = (
            Path(__file__).resolve(),
            ROOT / "tools/run_paper_evaluation_v4_pi_timing_v1.sh",
            ROOT / "tools/setup_paper_evaluation_v4_pi_timing_v1.sh",
            config.parent_runner,
            *config.dependency_hashes,
        )
        for source in tool_paths:
            shutil.copy2(source, bundle / "tools" / source.name)
        for name in (
            "test_rtlola_binding.py",
            "test_rtlola_features.py",
            "test_rtlola_learning.py",
            "test_input_prediction.py",
            "test_paper_evaluation_v4_pi_timing_v1.py",
        ):
            source = ROOT / "tests" / name
            if source.is_file():
                shutil.copy2(source, bundle / "tests" / name)
        shutil.copy2(ROOT / "pyproject.toml", bundle / "pyproject.toml")
        _copy_tree(ROOT / "src", bundle / "src")
        _copy_tree(ROOT / "rlolapythonbinding", bundle / "rlolapythonbinding")

        trace_store = Path(str(prepare["nominal_trace_store"]))
        _copy_tree(trace_store, bundle / "traces" / "nominal")
        traces = parent._traces(parent_config, "nominal")
        trace_records = [
            {
                "seed": trace.seed,
                "trace_id": trace.trace_id,
                "trace_sha256": trace.sha256,
            }
            for trace in traces
            if trace.seed in config.seeds
        ]
        if tuple(record["seed"] for record in trace_records) != config.seeds:
            raise ValueError("parent trace store lacks the Pi timing cohort")

        method_specs = {method.name: method for method in parent_config.methods}
        model_records: dict[str, list[dict[str, str]]] = {}
        copied_models: dict[str, str] = {}
        for method_name in config.methods:
            method = method_specs[method_name]
            for budget in config.budgets:
                records = []
                for model_path in parent._model_paths(parent_config, method, budget):
                    digest = model_sha256(model_path)
                    relative = f"models/{digest}"
                    if digest not in copied_models:
                        _copy_tree(model_path, bundle / relative)
                        copied_models[digest] = relative
                    records.append({"path": relative, "model_sha256": digest})
                model_records[f"{method_name}:{budget}"] = records

        parent_inputs = {
            "prepare_manifest": str(parent._stage_path(parent_config, "prepare")),
            "prepare_manifest_sha256": raw_sha256(parent._stage_path(parent_config, "prepare")),
            "nominal_manifest": str(parent._stage_path(parent_config, "nominal")),
            "nominal_manifest_sha256": raw_sha256(parent._stage_path(parent_config, "nominal")),
            "nominal_summary_sha256": raw_sha256(Path(str(nominal_manifest["summary"]))),
            "prediction_manifest": str(parent._stage_path(parent_config, "prediction-ablation")),
            "prediction_manifest_sha256": raw_sha256(parent._stage_path(parent_config, "prediction-ablation")),
            "prediction_summary_sha256": raw_sha256(Path(str(prediction_manifest["summary"]))),
        }
        shutil.copy2(Path(str(nominal_manifest["summary"])), bundle / "parent-nominal-summary.csv")
        shutil.copy2(Path(str(prediction_manifest["summary"])), bundle / "parent-prediction-summary.csv")

        base_payload = yaml.safe_load(config.parent_config.read_text())
        methods = [item for item in base_payload["methods"] if item["name"] in config.methods]
        manifest: dict[str, object] = {
            "schema": BUNDLE_SCHEMA,
            "config_identity": config.identity,
            "config_identity_sha256": config.identity_sha256,
            "parent_pins": before,
            "parent_inputs": parent_inputs,
            "method_specs": methods,
            "trace_records": trace_records,
            "model_records": model_records,
            "native": dict(config.native),
        }
        manifest["files"] = _tree_hashes(bundle)
        manifest["bundle_identity_sha256"] = payload_sha256(manifest)
        write_json_atomic(manifest, bundle / "manifest.json")
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(bundle, arcname=bundle.name)

        current_parent_inputs = {
            "prepare_manifest": str(parent._stage_path(parent_config, "prepare")),
            "prepare_manifest_sha256": raw_sha256(parent._stage_path(parent_config, "prepare")),
            "nominal_manifest": str(parent._stage_path(parent_config, "nominal")),
            "nominal_manifest_sha256": raw_sha256(parent._stage_path(parent_config, "nominal")),
            "nominal_summary_sha256": raw_sha256(Path(str(nominal_manifest["summary"]))),
            "prediction_manifest": str(parent._stage_path(parent_config, "prediction-ablation")),
            "prediction_manifest_sha256": raw_sha256(parent._stage_path(parent_config, "prediction-ablation")),
            "prediction_summary_sha256": raw_sha256(Path(str(prediction_manifest["summary"]))),
        }
        if current_parent_inputs != parent_inputs:
            raise ValueError("read-only parent artifacts changed while building the bundle")
    after = verify_parent_pins(config)
    if after != before:
        raise ValueError("read-only parent inputs changed while building the bundle")
    print(json.dumps({"archive": str(archive), "sha256": raw_sha256(archive)}, indent=2))
    return archive


def verify_bundle(bundle: Path) -> dict[str, object]:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"bundle manifest is missing: {bundle}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("unsupported Pi timing bundle schema")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("bundle file manifest is missing")
    _verify_tree(bundle, {str(key): str(value) for key, value in files.items()})
    identity = dict(manifest)
    actual_identity = str(identity.pop("bundle_identity_sha256"))
    if payload_sha256(identity) != actual_identity:
        raise ValueError("bundle identity hash differs")
    return manifest


def _memory_snapshot() -> dict[str, int]:
    values = {
        "rss_kib": -1,
        "pss_kib": -1,
        "uss_kib": -1,
        "peak_rss_kib": -1,
    }
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                values["rss_kib"] = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                values["peak_rss_kib"] = int(line.split()[1])
    rollup = Path("/proc/self/smaps_rollup")
    if rollup.is_file():
        private = 0
        for line in rollup.read_text().splitlines():
            if line.startswith("Pss:"):
                values["pss_kib"] = int(line.split()[1])
            elif line.startswith(("Private_Clean:", "Private_Dirty:")):
                private += int(line.split()[1])
        values["uss_kib"] = private
    return values


def latency_statistics(milliseconds: Sequence[float]) -> dict[str, float]:
    values = np.asarray(milliseconds, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("latency samples must be a finite non-empty vector")
    if np.any(values <= 0.0):
        raise ValueError("latency samples must be positive")
    p50, p90, p95, p99 = np.quantile(values, (0.50, 0.90, 0.95, 0.99))
    median = float(p50)
    mad = float(np.median(np.abs(values - median)))
    return {
        "p50_selection_commit_latency_ms": median,
        "p90_selection_commit_latency_ms": float(p90),
        "p95_selection_commit_latency_ms": float(p95),
        "p99_selection_commit_latency_ms": float(p99),
        "max_selection_commit_latency_ms": float(np.max(values)),
        "mad_selection_commit_latency_ms": mad,
        "iqr_selection_commit_latency_ms": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "p99_p50_tail_ratio": float(p99 / p50),
        "saturation_throughput_events_per_second": float(values.size * 1000.0 / values.sum()),
    }


def replay_service_times(
    milliseconds: Sequence[float],
    rate_hz: float,
) -> tuple[float, float]:
    """Return on-time share and final backlog for a sequential server."""
    values = np.asarray(milliseconds, dtype=np.float64) / 1000.0
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("service times must be finite and non-empty")
    if np.any(values <= 0.0) or not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("service times and rate must be positive")
    period = 1.0 / rate_hz
    finish = 0.0
    on_time = 0
    for index, service in enumerate(values):
        arrival = index * period
        finish = max(arrival, finish) + float(service)
        tolerance = 1e-12 * max(1.0, abs(finish), abs(arrival + period))
        on_time += int(finish <= arrival + period + tolerance)
    final_deadline = values.size * period
    return on_time / values.size, max(0.0, finish - final_deadline)


def maximum_empirical_rate(
    milliseconds: Sequence[float],
    target_on_time: float,
    *,
    iterations: int = 80,
) -> float:
    values = np.asarray(milliseconds, dtype=np.float64)
    if not 0.0 < target_on_time <= 1.0:
        raise ValueError("on-time target must be in (0, 1]")
    saturation = float(values.size * 1000.0 / values.sum())
    low, high = 0.0, saturation * (1.0 + 1e-12)
    for _ in range(iterations):
        middle = (low + high) / 2.0
        share, backlog = replay_service_times(values, middle)
        if share + 1e-15 >= target_on_time and backlog <= 1e-12:
            low = middle
        else:
            high = middle
    return low


def _thread_controls() -> None:
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:
        pass


def _method_specs(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    records = manifest.get("method_specs")
    if not isinstance(records, list):
        raise ValueError("bundle method specifications are missing")
    result = {str(record["name"]): dict(record) for record in records}
    if tuple(result) != METHOD_NAMES:
        raise ValueError("bundle method order differs")
    return result


def _trace(bundle: Path, seed: int) -> tuple[str, str, tuple[Any, ...]]:
    store = load_random_waypoint_trace_store(bundle / "traces" / "nominal")
    item = store.traces_for_seed(seed)
    if len(item) != 1:
        raise ValueError(f"bundle trace coverage differs for seed {seed}")
    stored = item[0]
    return stored.trace_id, stored.trace.metadata.trace_sha256, tuple(stored.trace.events)


def _models_for(
    bundle: Path,
    manifest: Mapping[str, object],
    method: str,
    budget: int,
) -> tuple[Path, ...]:
    matrix = manifest.get("model_records")
    if not isinstance(matrix, dict):
        raise ValueError("bundle model matrix is missing")
    records = matrix.get(f"{method}:{budget}")
    if not isinstance(records, list):
        raise ValueError(f"bundle model mapping is missing for {method}/{budget}")
    paths = []
    for record in records:
        path = bundle / str(record["path"])
        if model_sha256(path) != str(record["model_sha256"]):
            raise ValueError(f"bundle model hash differs: {method}/{budget}")
        paths.append(path)
    return tuple(paths)


def _policy(
    bundle: Path,
    manifest: Mapping[str, object],
    method: str,
    budget: int,
    events: Sequence[Any],
) -> object | None:
    specs = _method_specs(manifest)
    kind = str(specs[method]["kind"])
    paths = _models_for(bundle, manifest, method, budget)
    candidates = default_action_catalog(("girard", "scott", "pca", "combastel"))
    if kind == "g15":
        if len(paths) != 1:
            raise ValueError("G15 timing requires exactly one frozen model")
        return RtlolaReducerPolicy(ReducerPolicy.load(paths[0]), candidates)
    if kind in ("vote3", "vote3_guarded"):
        if len(paths) != 3:
            raise ValueError("Vote3 timing requires exactly three frozen models")
        return vote_selection.VoteGuardPolicy(
            tuple(ReducerPolicy.load(path) for path in paths),
            events,
            guarded=kind == "vote3_guarded",
        )
    if paths:
        raise ValueError(f"non-learned method unexpectedly has models: {method}")
    return None


def _benchmark_config(
    method: Mapping[str, object],
    budget: int,
    event_count: int,
) -> RtlolaBenchmarkConfig:
    return RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind="random_waypoint",
        length=event_count,
        budget=budget,
        horizon=int(method["horizon"]),
        beam_width=max(1, int(method["beam_width"])),
        prediction_step_seconds=0.1,
        seeds=1,
        methods=[str(method["runtime_method"])],
        reference_mode="off",
        mpc_reference="rollout",
        mpc_candidate_names=["girard", "scott", "pca", "combastel"],
    )


def _cell_identity(
    manifest: Mapping[str, object],
    seed: int,
    budget: int,
    method: str,
    trace_sha256: str,
    order_index: int,
) -> dict[str, object]:
    models = manifest["model_records"][f"{method}:{budget}"]  # type: ignore[index]
    return {
        "schema": CELL_SCHEMA,
        "bundle_identity_sha256": manifest["bundle_identity_sha256"],
        "seed": seed,
        "budget": budget,
        "method": method,
        "order_index": order_index,
        "trace_sha256": trace_sha256,
        "model_sha256": [record["model_sha256"] for record in models],
    }


def _run_benchmark(
    bundle: Path,
    manifest: Mapping[str, object],
    seed: int,
    budget: int,
    method_name: str,
    *,
    policy_override: object = _POLICY_UNSET,
) -> tuple[Any, object | None, str, str]:
    trace_id, trace_hash, full_events = _trace(bundle, seed)
    identity = manifest["config_identity"]
    measured_stop = int(identity["measured"][1])  # type: ignore[index]
    events = full_events[:measured_stop]
    method = _method_specs(manifest)[method_name]
    policy = (
        _policy(bundle, manifest, method_name, budget, events)
        if policy_override is _POLICY_UNSET
        else policy_override
    )
    config = _benchmark_config(method, budget, len(events))
    diagnostic_function = benchmark_module.prediction_diagnostics
    benchmark_module.prediction_diagnostics = lambda *_args, **_kwargs: ()
    try:
        result = run_event_trace_benchmark(
            config,
            events,
            trace_kind=trace_id,
            seed=seed,
            method=str(method["runtime_method"]),
            policy=policy,
        )
    finally:
        benchmark_module.prediction_diagnostics = diagnostic_function
    return result, policy, trace_id, trace_hash


def execute_cell(
    bundle: Path,
    output: Path,
    seed: int,
    budget: int,
    method: str,
    order_index: int,
) -> Path:
    _thread_controls()
    _assert_runtime_output(output)
    if order_index not in range(len(METHOD_NAMES)):
        raise ValueError("cell method order index is out of range")
    manifest = verify_bundle(bundle)
    trace_id, trace_hash, _ = _trace(bundle, seed)
    identity = _cell_identity(
        manifest, seed, budget, method, trace_hash, order_index
    )
    directory = output / "cells" / f"seed-{seed}" / f"budget-{budget}" / method
    manifest_path = directory / "manifest.json"
    summary_path = directory / "summary.csv"
    events_path = directory / "events.csv"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity") != identity:
            raise ValueError(f"stale Pi timing cell: {directory}")
        if prior.get("summary_sha256") != raw_sha256(summary_path):
            raise ValueError(f"Pi timing cell summary hash differs: {directory}")
        if prior.get("events_sha256") != raw_sha256(events_path):
            raise ValueError(f"Pi timing cell event hash differs: {directory}")
        return manifest_path

    baseline_memory = _memory_snapshot()
    model_paths = _models_for(bundle, manifest, method, budget)
    model_bytes = sum(
        path.stat().st_size
        for model in model_paths
        for path in model.rglob("*")
        if path.is_file()
    )
    measured_stop = int(manifest["config_identity"]["measured"][1])  # type: ignore[index]
    _, _, full_events = _trace(bundle, seed)
    policy_for_run = _policy(
        bundle, manifest, method, budget, full_events[:measured_stop]
    )
    policy_loaded_memory = _memory_snapshot()
    # Policy creation is intentionally outside the unchanged decision_time_ms.
    result, policy, _trace_id, actual_trace_hash = _run_benchmark(
        bundle,
        manifest,
        seed,
        budget,
        method,
        policy_override=policy_for_run,
    )
    if actual_trace_hash != trace_hash:
        raise ValueError("cell trace hash changed during execution")
    post_memory = _memory_snapshot()

    status = "completed"
    failure_type = ""
    failure_message = ""
    if result.failures:
        failure = result.failures[0]
        if str(failure.failure_type) != "RtlolaNoFeasibleAction":
            raise RuntimeError(
                f"Pi timing infrastructure/native failure: {failure.failure_type}: "
                f"{failure.message}"
            )
        status = "fallback_failed"
        failure_type = str(failure.failure_type)
        failure_message = str(failure.message)

    series = (
        result.timeseries
        if not result.timeseries.empty
        else result.failed_timeseries
    ).copy()
    measured_start, measured_stop = manifest["config_identity"]["measured"]  # type: ignore[index]
    if "step" in series:
        series = series.sort_values("step")
        measured = series[
            (series["step"] >= int(measured_start))
            & (series["step"] < int(measured_stop))
        ].copy()
    else:
        measured = pd.DataFrame(columns=(
            "seed", "budget", "method", "trace_id", "step",
            "decision_time_ms", "reducer_used", "generator_count",
            "active_dynamic_generator_count", "evaluated_leaves",
            "pruned_branches", "fallback_used", "reducer_failure_count",
            "infeasible_candidate_count",
        ))
    if status == "completed" and len(measured) != int(measured_stop) - int(measured_start):
        raise ValueError("completed Pi timing cell has the wrong measured window")
    if status == "completed" and len(result.summary) != 1:
        raise ValueError("completed Pi timing cell lacks one benchmark summary")
    fallback_count = (
        int(result.summary.iloc[0].get("fallback_count", 0))
        if len(result.summary) == 1
        else 0
    )
    if status == "completed" and fallback_count > 0:
        status = "fallback_failed"
        failure_type = "IntervalFallback"
        failure_message = "ordinary timing run used interval fallback"

    measured["method"] = method
    measured["trace_id"] = trace_id
    measured["selection_commit_latency_ms"] = measured["decision_time_ms"]
    keep = [
        "seed",
        "budget",
        "method",
        "trace_id",
        "step",
        "selection_commit_latency_ms",
        "reducer_used",
        "generator_count",
        "active_dynamic_generator_count",
        "evaluated_leaves",
        "pruned_branches",
        "fallback_used",
        "reducer_failure_count",
        "infeasible_candidate_count",
    ]
    event_frame = measured[[column for column in keep if column in measured]].copy()
    if status == "completed":
        statistics = latency_statistics(event_frame["selection_commit_latency_ms"])
        capacities = {
            f"max_rate_{int(round(target * 100))}pct_on_time_hz": maximum_empirical_rate(
                event_frame["selection_commit_latency_ms"], target
            )
            for target in (0.95, 0.99, 1.0)
        }
    else:
        statistics = {
            key: float("nan")
            for key in latency_statistics([1.0]).keys()
        }
        capacities = {
            "max_rate_95pct_on_time_hz": float("nan"),
            "max_rate_99pct_on_time_hz": float("nan"),
            "max_rate_100pct_on_time_hz": float("nan"),
        }
    row = {
        "seed": seed,
        "budget": budget,
        "method": method,
        "order_index": order_index,
        "trace_id": trace_id,
        "trace_sha256": trace_hash,
        "status": status,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "measured_event_count": len(event_frame),
        "model_bytes": model_bytes,
        **{f"baseline_{key}": value for key, value in baseline_memory.items()},
        **{
            f"policy_loaded_{key}": value
            for key, value in policy_loaded_memory.items()
        },
        **{f"post_run_{key}": value for key, value in post_memory.items()},
        "policy_loaded_rss_delta_kib": (
            policy_loaded_memory["rss_kib"] - baseline_memory["rss_kib"]
        ),
        "post_run_rss_delta_kib": (
            post_memory["rss_kib"] - baseline_memory["rss_kib"]
        ),
        **statistics,
        **capacities,
    }
    directory.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(event_frame, events_path)
    write_csv_atomic(pd.DataFrame([row]), summary_path)
    if policy is not None and hasattr(policy, "diagnostics"):
        # This is deliberately written after the timing run.  Diagnostic record
        # construction remains unchanged inside the frozen policy.
        write_csv_atomic(pd.DataFrame(policy.diagnostics), directory / "policy-diagnostics.csv")
    cell_manifest = {
        "schema": CELL_SCHEMA,
        "identity": identity,
        "status": status,
        "summary_sha256": raw_sha256(summary_path),
        "events_sha256": raw_sha256(events_path),
    }
    write_json_atomic(cell_manifest, manifest_path)
    return manifest_path


def rotate_method_order(
    methods: Sequence[str], seed_index: int, budget_index: int
) -> tuple[str, ...]:
    offset = (seed_index + budget_index) % len(methods)
    return tuple(methods[offset:]) + tuple(methods[:offset])


def _child_environment(bundle: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for variable in THREAD_VARIABLES:
        environment[variable] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    pythonpath = (bundle / "src", bundle / "tools")
    environment["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in pythonpath), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return environment


def _cell_command(
    bundle: Path,
    output: Path,
    seed: int,
    budget: int,
    method: str,
    order_index: int,
    cpu_core: int,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_cell",
        "--bundle",
        str(bundle),
        "--run-output",
        str(output),
        "--seed",
        str(seed),
        "--budget",
        str(budget),
        "--method",
        method,
        "--order-index",
        str(order_index),
    ]
    if platform.system() == "Linux" and hasattr(os, "sched_setaffinity"):
        return ["taskset", "-c", str(cpu_core), *command]
    return command


def _aggregate_run(
    bundle_manifest: Mapping[str, object],
    output: Path,
    config: PiTimingConfig,
) -> Path:
    summaries = []
    events = []
    for seed in config.seeds:
        for budget in config.budgets:
            for method in config.methods:
                directory = output / "cells" / f"seed-{seed}" / f"budget-{budget}" / method
                cell_manifest = json.loads((directory / "manifest.json").read_text())
                if cell_manifest.get("schema") != CELL_SCHEMA:
                    raise ValueError(f"unsupported cell manifest: {directory}")
                if raw_sha256(directory / "summary.csv") != cell_manifest["summary_sha256"]:
                    raise ValueError(f"cell summary changed: {directory}")
                if raw_sha256(directory / "events.csv") != cell_manifest["events_sha256"]:
                    raise ValueError(f"cell events changed: {directory}")
                summaries.append(pd.read_csv(directory / "summary.csv"))
                events.append(pd.read_csv(directory / "events.csv"))
    cells = pd.concat(summaries, ignore_index=True)
    event_samples = pd.concat(events, ignore_index=True)
    if len(cells) != config.expected_cells or cells[["seed", "budget", "method"]].duplicated().any():
        raise ValueError("Pi timing aggregate cell matrix differs")
    for seed_index, seed in enumerate(config.seeds):
        for budget_index, budget in enumerate(config.budgets):
            frame = cells[(cells["seed"] == seed) & (cells["budget"] == budget)]
            actual = tuple(
                frame.sort_values("order_index")["method"].astype(str)
            )
            expected = rotate_method_order(
                config.methods, seed_index, budget_index
            )
            if actual != expected:
                raise ValueError(
                    f"Pi timing method rotation differs for {seed}/{budget}"
                )
    completed = cells[cells["status"] == "completed"].copy()
    summary_rows = []
    metric_columns = [
        "p50_selection_commit_latency_ms",
        "p90_selection_commit_latency_ms",
        "p95_selection_commit_latency_ms",
        "p99_selection_commit_latency_ms",
        "max_selection_commit_latency_ms",
        "mad_selection_commit_latency_ms",
        "iqr_selection_commit_latency_ms",
        "p99_p50_tail_ratio",
        "saturation_throughput_events_per_second",
        "max_rate_95pct_on_time_hz",
        "max_rate_99pct_on_time_hz",
        "max_rate_100pct_on_time_hz",
        "post_run_rss_kib",
        "post_run_pss_kib",
        "post_run_uss_kib",
        "post_run_peak_rss_kib",
        "policy_loaded_rss_kib",
        "policy_loaded_pss_kib",
        "policy_loaded_uss_kib",
        "policy_loaded_rss_delta_kib",
        "post_run_rss_delta_kib",
        "model_bytes",
    ]
    for (budget, method), frame in completed.groupby(["budget", "method"], sort=True):
        row: dict[str, object] = {
            "budget": int(budget),
            "method": str(method),
            "valid_seed_count": len(frame),
            "available": len(frame) == len(config.seeds),
        }
        for column in metric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            row[f"median_{column}"] = float(values.median()) if len(values) else np.nan
            row[f"min_{column}"] = float(values.min()) if len(values) else np.nan
            row[f"max_{column}"] = float(values.max()) if len(values) else np.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    write_csv_atomic(cells, output / "timing_cells.csv")
    write_csv_atomic(event_samples, output / "event_samples.csv")
    write_csv_atomic(summary, output / "summary.csv")
    manifest = {
        "schema": RESULT_SCHEMA,
        "bundle_identity_sha256": bundle_manifest["bundle_identity_sha256"],
        "config_identity_sha256": config.identity_sha256,
        "cell_count": len(cells),
        "completed_cell_count": int((cells["status"] == "completed").sum()),
        "unavailable_cell_count": int((cells["status"] != "completed").sum()),
        "event_sample_count": len(event_samples),
        "primary_measure": "existing decision_time_ms: reducer selection plus native live commit",
        "unpaced": True,
        "workers": 1,
        "native_threads": 1,
        "files": {
            "timing_cells.csv": raw_sha256(output / "timing_cells.csv"),
            "event_samples.csv": raw_sha256(output / "event_samples.csv"),
            "summary.csv": raw_sha256(output / "summary.csv"),
        },
    }
    write_json_atomic(
        _seal_identity(manifest, "result_identity_sha256"),
        output / "manifest.json",
    )
    return output / "manifest.json"


def run_matrix(
    config: PiTimingConfig,
    bundle: Path,
    output: Path,
    *,
    allow_non_pi: bool = False,
) -> Path:
    _assert_runtime_output(output)
    manifest = verify_bundle(bundle)
    if manifest["config_identity_sha256"] != config.identity_sha256:
        raise ValueError("bundle/config identity differs for this timing run")
    before = preflight(bundle, allow_non_pi=allow_non_pi)
    swap_before = _swap_counts()
    output.mkdir(parents=True, exist_ok=True)
    for seed_index, seed in enumerate(config.seeds):
        for budget_index, budget in enumerate(config.budgets):
            order = rotate_method_order(config.methods, seed_index, budget_index)
            for order_index, method in enumerate(order):
                command = _cell_command(
                    bundle,
                    output,
                    seed,
                    budget,
                    method,
                    order_index,
                    config.cpu_core,
                )
                subprocess.run(
                    command,
                    check=True,
                    env=_child_environment(bundle),
                    cwd=bundle,
                )
    result = _aggregate_run(manifest, output, config)
    swap_after = _swap_counts()
    if swap_after != swap_before:
        raise ValueError(
            f"swap activity invalidates Pi timing: {swap_before} -> {swap_after}"
        )
    after = preflight(bundle, allow_non_pi=allow_non_pi, check_idle=False)
    platform_path = output / "platform-validation.json"
    write_json_atomic(
        {
            "preflight_before": before,
            "preflight_after": after,
            "swap_before": swap_before,
            "swap_after": swap_after,
        },
        platform_path,
    )
    result_payload = json.loads(result.read_text())
    result_payload["files"] = _tree_hashes(output, exclude=(result,))
    write_json_atomic(
        _seal_identity(result_payload, "result_identity_sha256"), result
    )
    return result


class _TimedPolicy:
    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.selection_ns: list[int] = []

    def choose(self, *args: Any, **kwargs: Any) -> Any:
        started = perf_counter_ns()
        result = self.policy.choose(*args, **kwargs)
        self.selection_ns.append(perf_counter_ns() - started)
        return result


def _semantic_frame(result: Any) -> pd.DataFrame:
    excluded = {"decision_time_ms", "binding_runtime_ns"}
    return result.timeseries.drop(
        columns=[column for column in excluded if column in result.timeseries],
    ).reset_index(drop=True)


def execute_profile_cell(
    bundle: Path,
    manifest: Mapping[str, object],
    seed: int,
    budget: int,
    method_name: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    uninstrumented, _, _, _ = _run_benchmark(bundle, manifest, seed, budget, method_name)
    method = _method_specs(manifest)[method_name]
    trace_id, _, full_events = _trace(bundle, seed)
    measured_stop = int(manifest["config_identity"]["measured"][1])  # type: ignore[index]
    events = full_events[:measured_stop]
    direct = _policy(bundle, manifest, method_name, budget, events)
    timed_direct = _TimedPolicy(direct) if direct is not None else None
    config = _benchmark_config(method, budget, len(events))

    selection_ns: list[int] = []
    commit_ns: list[int] = []
    predictor_ns: list[int] = []
    original_select = benchmark_module._select_method_decision
    original_commit = benchmark_module._commit_decision
    original_predict = benchmark_module.predict_future_events
    original_diagnostics = benchmark_module.prediction_diagnostics

    def timed_select(*args: Any, **kwargs: Any) -> Any:
        started = perf_counter_ns()
        result = original_select(*args, **kwargs)
        selection_ns.append(perf_counter_ns() - started)
        return result

    def timed_commit(*args: Any, **kwargs: Any) -> Any:
        started = perf_counter_ns()
        result = original_commit(*args, **kwargs)
        commit_ns.append(perf_counter_ns() - started)
        return result

    def timed_predict(*args: Any, **kwargs: Any) -> Any:
        started = perf_counter_ns()
        result = original_predict(*args, **kwargs)
        predictor_ns.append(perf_counter_ns() - started)
        return result

    benchmark_module._select_method_decision = timed_select
    benchmark_module._commit_decision = timed_commit
    benchmark_module.predict_future_events = timed_predict
    benchmark_module.prediction_diagnostics = lambda *_args, **_kwargs: ()
    try:
        instrumented = run_event_trace_benchmark(
            config,
            events,
            trace_kind=trace_id,
            seed=seed,
            method=str(method["runtime_method"]),
            policy=timed_direct,
        )
    finally:
        benchmark_module._select_method_decision = original_select
        benchmark_module._commit_decision = original_commit
        benchmark_module.predict_future_events = original_predict
        benchmark_module.prediction_diagnostics = original_diagnostics

    pd.testing.assert_frame_equal(
        _semantic_frame(uninstrumented),
        _semantic_frame(instrumented),
        check_dtype=False,
        check_exact=True,
    )
    if uninstrumented.failures != instrumented.failures:
        raise ValueError("instrumented profile changed failure semantics")
    direct_selection = timed_direct.selection_ns if timed_direct is not None else selection_ns
    event_count = len(events)
    if len(direct_selection) != event_count or len(commit_ns) != event_count:
        raise ValueError("profile phase call counts differ from event count")
    if predictor_ns and len(predictor_ns) != event_count:
        raise ValueError("predictor phase call count differs from event count")
    if not predictor_ns:
        predictor_ns = [0] * event_count
    measured_start = int(manifest["config_identity"]["measured"][0])  # type: ignore[index]
    rows = pd.DataFrame({
        "seed": seed,
        "budget": budget,
        "method": method_name,
        "step": np.arange(event_count),
        "predictor_latency_ms": np.asarray(predictor_ns) / 1_000_000.0,
        "selection_latency_ms": np.asarray(direct_selection) / 1_000_000.0,
        "commit_latency_ms": np.asarray(commit_ns) / 1_000_000.0,
    })
    rows = rows[(rows["step"] >= measured_start) & (rows["step"] < measured_stop)].copy()
    return rows, {
        "method": method_name,
        "seed": seed,
        "budget": budget,
        "semantic_parity": True,
        "measured_event_count": len(rows),
    }


def run_profile(config: PiTimingConfig, bundle: Path, output: Path) -> Path:
    _assert_runtime_output(output)
    manifest = verify_bundle(bundle)
    if manifest["config_identity_sha256"] != config.identity_sha256:
        raise ValueError("bundle/config identity differs for this profile run")
    rows = []
    cells = []
    for method in config.methods:
        frame, cell = execute_profile_cell(
            bundle,
            manifest,
            config.profile_seed,
            config.profile_budget,
            method,
        )
        rows.append(frame)
        cells.append(cell)
    events = pd.concat(rows, ignore_index=True)
    cell_frame = pd.DataFrame(cells)
    profile_dir = output / "profile"
    write_csv_atomic(events, profile_dir / "event_samples.csv")
    write_csv_atomic(cell_frame, profile_dir / "cells.csv")
    profile_manifest = {
        "schema": PROFILE_SCHEMA,
        "bundle_identity_sha256": manifest["bundle_identity_sha256"],
        "cell_count": len(cell_frame),
        "event_sample_count": len(events),
        "semantic_parity": bool(cell_frame["semantic_parity"].all()),
        "files": {
            "event_samples.csv": raw_sha256(profile_dir / "event_samples.csv"),
            "cells.csv": raw_sha256(profile_dir / "cells.csv"),
        },
    }
    write_json_atomic(profile_manifest, profile_dir / "manifest.json")
    result_manifest_path = output / "manifest.json"
    if result_manifest_path.is_file():
        result_manifest = json.loads(result_manifest_path.read_text())
        result_manifest["profile_cell_count"] = len(cell_frame)
        result_manifest["profile_event_sample_count"] = len(events)
        result_manifest["profile_semantic_parity"] = True
        result_manifest["files"] = _tree_hashes(
            output, exclude=(result_manifest_path,)
        )
        write_json_atomic(
            _seal_identity(result_manifest, "result_identity_sha256"),
            result_manifest_path,
        )
    return profile_dir / "manifest.json"


def _hardware_provenance() -> dict[str, object]:
    model_path = Path("/proc/device-tree/model")
    model = model_path.read_bytes().replace(b"\0", b"").decode() if model_path.is_file() else "unknown"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "model": model,
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }


def _swap_counts() -> dict[str, int]:
    counts = {"pswpin": 0, "pswpout": 0}
    path = Path("/proc/vmstat")
    if not path.is_file():
        return counts
    for line in path.read_text().splitlines():
        name, *values = line.split()
        if name in counts and values:
            counts[name] = int(values[0])
    return counts


def _verify_environment(expected: Mapping[str, object]) -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": metadata.version("PyYAML"),
        "torch": metadata.version("torch"),
        "maturin": metadata.version("maturin"),
        "pytest": metadata.version("pytest"),
    }
    for name, actual in versions.items():
        wanted = str(expected[name])
        if actual != wanted:
            raise ValueError(f"Pi environment version differs for {name}: {actual} != {wanted}")
    rustc = subprocess.run(
        ["rustc", "--version"], check=True, capture_output=True, text=True
    ).stdout.split()
    versions["rust"] = rustc[1]
    if versions["rust"] != str(expected["rust"]):
        raise ValueError(
            f"Pi Rust version differs: {versions['rust']} != {expected['rust']}"
        )
    return versions


def preflight(
    bundle: Path,
    *,
    allow_non_pi: bool = False,
    check_idle: bool = True,
) -> dict[str, object]:
    manifest = verify_bundle(bundle)
    provenance = _hardware_provenance()
    if not allow_non_pi:
        if provenance["machine"] != "aarch64" or "Raspberry Pi 5" not in str(provenance["model"]):
            raise ValueError("canonical Pi timing requires a Raspberry Pi 5/aarch64")
        cpu_core = int(manifest["config_identity"]["cpu_core"])  # type: ignore[index]
        if cpu_core >= (os.cpu_count() or 0):
            raise ValueError(f"configured Pi timing CPU core is unavailable: {cpu_core}")
        governor_paths = sorted(Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"))
        governors = {path.read_text().strip() for path in governor_paths}
        if governors != {"performance"}:
            raise ValueError(f"CPU governors are not uniformly performance: {governors}")
        throttled = subprocess.run(
            ["vcgencmd", "get_throttled"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if throttled != "throttled=0x0":
            raise ValueError(f"Raspberry Pi throttle history invalidates timing: {throttled}")
        provenance["get_throttled"] = throttled
        if check_idle:
            load_average = os.getloadavg()[0]
            if load_average >= 0.5:
                raise ValueError(f"one-minute load average is too high for timing: {load_average}")
            provenance["one_minute_load_average"] = load_average
        provenance["environment_versions"] = _verify_environment(
            manifest["config_identity"]["environment"]  # type: ignore[index]
        )
    require_binding()
    if (
        BINDING_REVISION != manifest["native"]["binding_revision"]  # type: ignore[index]
        or INTERPRETER_REVISION != manifest["native"]["interpreter_revision"]  # type: ignore[index]
        or BINDING_BUILD_PROFILE != manifest["native"]["binding_build_profile"]  # type: ignore[index]
    ):
        raise ValueError("installed native stack differs from the Pi bundle")
    return {
        "bundle_identity_sha256": manifest["bundle_identity_sha256"],
        "native_verified": True,
        "hardware": provenance,
        "allow_non_pi": allow_non_pi,
    }


def run_contract_tests(bundle: Path) -> dict[str, int]:
    """Run binding-backed Pi semantic contracts without mutating the bundle."""
    verify_bundle(bundle)
    test_paths = [
        bundle / "tests/test_rtlola_binding.py",
        bundle / "tests/test_rtlola_features.py",
        bundle / "tests/test_rtlola_learning.py",
        bundle / "tests/test_input_prediction.py",
    ]
    if not all(path.is_file() for path in test_paths):
        raise ValueError("Pi contract-test files are missing from the bundle")
    with tempfile.TemporaryDirectory(prefix="pzr-pi-contract-tests-") as temporary:
        report = Path(temporary) / "junit.xml"
        environment = _child_environment(bundle)
        environment["PYTEST_ADDOPTS"] = ""
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *(str(path) for path in test_paths),
                "-o",
                f"cache_dir={temporary}/cache",
                f"--junitxml={report}",
            ],
            check=True,
            cwd=bundle,
            env=environment,
        )
        root = ET.parse(report).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        counts = {
            "tests": sum(int(suite.attrib.get("tests", "0")) for suite in suites),
            "failures": sum(int(suite.attrib.get("failures", "0")) for suite in suites),
            "errors": sum(int(suite.attrib.get("errors", "0")) for suite in suites),
            "skipped": sum(int(suite.attrib.get("skipped", "0")) for suite in suites),
        }
    if counts["tests"] < 1 or any(counts[name] for name in ("failures", "errors", "skipped")):
        raise ValueError(f"Pi binding-backed contract outcomes differ: {counts}")
    return counts


def pack_results(output: Path, archive: Path) -> Path:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("completed timing manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != RESULT_SCHEMA:
        raise ValueError("unsupported timing result schema")
    _verify_identity(manifest, "result_identity_sha256")
    if int(manifest.get("profile_cell_count", -1)) != len(METHOD_NAMES):
        raise ValueError("complete ten-method phase profile is required before packing")
    if not bool(manifest.get("profile_semantic_parity", False)):
        raise ValueError("phase profile semantic parity is not established")
    for relative, expected in manifest["files"].items():
        if raw_sha256(output / relative) != expected:
            raise ValueError(f"timing result hash differs: {relative}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output, arcname=output.name)
    print(json.dumps({"archive": str(archive), "sha256": raw_sha256(archive)}, indent=2))
    return archive


def import_results(config: PiTimingConfig, archive: Path) -> Path:
    """Install a verified Pi archive into the new add-on root only."""
    _assert_isolated_output(config.output)
    if config.output.exists():
        raise ValueError(f"isolated Pi result root already exists: {config.output}")
    with tempfile.TemporaryDirectory(prefix="pzr-pi-results-") as temporary:
        extracted = Path(temporary)
        _safe_extract(archive.resolve(), extracted)
        candidates = [
            path for path in extracted.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        ]
        if len(candidates) != 1:
            raise ValueError("Pi result archive must contain exactly one result root")
        source = candidates[0]
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest.get("schema") != RESULT_SCHEMA:
            raise ValueError("unsupported Pi result archive schema")
        _verify_identity(manifest, "result_identity_sha256")
        for relative, expected in manifest["files"].items():
            if raw_sha256(source / relative) != expected:
                raise ValueError(f"Pi result archive hash differs: {relative}")
        if int(manifest.get("cell_count", -1)) != config.expected_cells:
            raise ValueError("Pi result archive cell count differs")
        _copy_tree(source, config.output)
    return config.output / "manifest.json"


def _pareto_flags(frame: pd.DataFrame, x: str, y: str) -> pd.Series:
    values = frame[[x, y]].to_numpy(dtype=np.float64)
    flags = np.ones(len(frame), dtype=bool)
    for index, point in enumerate(values):
        dominates = np.all(values <= point, axis=1) & np.any(values < point, axis=1)
        dominates[index] = False
        flags[index] = not bool(np.any(dominates))
    return pd.Series(flags, index=frame.index)


def combine_report(config: PiTimingConfig, pi_output: Path) -> Path:
    verify_parent_pins(config)
    result_manifest = json.loads((pi_output / "manifest.json").read_text())
    if result_manifest.get("schema") != RESULT_SCHEMA:
        raise ValueError("unsupported Pi result for combined reporting")
    _verify_identity(result_manifest, "result_identity_sha256")
    for relative, expected in result_manifest["files"].items():
        if raw_sha256(pi_output / relative) != expected:
            raise ValueError(f"Pi result changed before reporting: {relative}")
    parent = _import_parent(config)
    parent_config = parent.load_config(config.parent_config)
    nominal_manifest = parent._load_stage(parent_config, "nominal")
    nominal = pd.read_csv(str(nominal_manifest["summary"]))
    timing = pd.read_csv(pi_output / "timing_cells.csv")
    profile_manifest = json.loads((pi_output / "profile/manifest.json").read_text())
    if profile_manifest.get("schema") != PROFILE_SCHEMA or not profile_manifest.get("semantic_parity"):
        raise ValueError("validated Pi phase profile is missing")
    profile = pd.read_csv(pi_output / "profile/event_samples.csv")
    selected_nominal = nominal[nominal["seed"].isin(config.seeds)].copy()
    joined = timing.merge(
        selected_nominal,
        on=["seed", "budget", "method"],
        how="inner",
        suffixes=("_timing", "_nominal"),
        validate="one_to_one",
    )
    expected_join = config.expected_cells
    if len(joined) != expected_join:
        raise ValueError("Pi timing and parent nominal identities do not pair exactly")
    available = joined[
        (joined["status_timing"] == "completed")
        & (joined["status_nominal"] == "completed")
        & (pd.to_numeric(joined["false_negative_count"]) == 0)
    ].copy()
    rows = []
    for (budget, method), frame in available.groupby(["budget", "method"], sort=True):
        rows.append({
            "budget": int(budget),
            "method": str(method),
            "valid_seed_count": len(frame),
            "available": len(frame) == len(config.seeds),
            "median_p99_selection_commit_latency_ms": float(frame["p99_selection_commit_latency_ms"].median()),
            "median_mean_approx_loss": float(frame["mean_approx_loss"].median()),
            "false_positive_count": int(frame["false_positive_count"].sum()),
            "reference_negative_count": int(frame["reference_negative_count"].sum()),
            "pooled_fpr": float(frame["false_positive_count"].sum() / frame["reference_negative_count"].sum()),
        })
    pareto = pd.DataFrame(rows)
    pareto["pareto_latency_loss_global"] = False
    pareto["pareto_latency_fpr_global"] = False
    pareto["pareto_latency_loss_within_bound"] = False
    pareto["pareto_latency_fpr_within_bound"] = False
    eligible = pareto[pareto["available"]].copy()
    if len(eligible):
        pareto.loc[eligible.index, "pareto_latency_loss_global"] = _pareto_flags(
            eligible,
            "median_p99_selection_commit_latency_ms",
            "median_mean_approx_loss",
        )
        pareto.loc[eligible.index, "pareto_latency_fpr_global"] = _pareto_flags(
            eligible, "median_p99_selection_commit_latency_ms", "pooled_fpr"
        )
    for _, indices in eligible.groupby("budget").groups.items():
        subset = pareto.loc[indices]
        pareto.loc[indices, "pareto_latency_loss_within_bound"] = _pareto_flags(
            subset, "median_p99_selection_commit_latency_ms", "median_mean_approx_loss"
        )
        pareto.loc[indices, "pareto_latency_fpr_within_bound"] = _pareto_flags(
            subset, "median_p99_selection_commit_latency_ms", "pooled_fpr"
        )

    phase = profile.groupby("method", sort=False).agg(
        median_predictor_latency_ms=("predictor_latency_ms", "median"),
        median_selection_latency_ms=("selection_latency_ms", "median"),
        median_commit_latency_ms=("commit_latency_ms", "median"),
        p95_predictor_latency_ms=("predictor_latency_ms", lambda values: float(np.quantile(values, 0.95))),
        p95_selection_latency_ms=("selection_latency_ms", lambda values: float(np.quantile(values, 0.95))),
        p95_commit_latency_ms=("commit_latency_ms", lambda values: float(np.quantile(values, 0.95))),
    ).reset_index()

    output = config.combined_output
    if output.exists():
        raise ValueError(f"combined report root already exists: {output}")
    output.mkdir(parents=True)
    write_csv_atomic(pareto, output / "latency_quality_pareto.csv")
    write_csv_atomic(pd.read_csv(pi_output / "summary.csv"), output / "pi_timing_summary.csv")
    write_csv_atomic(phase, output / "pi_phase_profile_summary.csv")
    claims = {
        "schema": COMBINED_SCHEMA,
        "parent_config_sha256": config.parent_config_sha256,
        "parent_nominal_manifest_sha256": raw_sha256(parent._stage_path(parent_config, "nominal")),
        "pi_result_manifest_sha256": raw_sha256(pi_output / "manifest.json"),
        "timing_cell_count": len(timing),
        "paired_cell_count": len(joined),
        "pareto_point_count": len(pareto),
        "primary_measure": "selection plus native live commit (existing decision_time_ms)",
        "predictor_primary_measure_included": False,
    }
    write_json_atomic(claims, output / "claim_values.json")
    report_manifest = {
        "schema": COMBINED_SCHEMA,
        "parents_unchanged": True,
        "files": {
            "latency_quality_pareto.csv": raw_sha256(output / "latency_quality_pareto.csv"),
            "pi_timing_summary.csv": raw_sha256(output / "pi_timing_summary.csv"),
            "pi_phase_profile_summary.csv": raw_sha256(output / "pi_phase_profile_summary.csv"),
            "claim_values.json": raw_sha256(output / "claim_values.json"),
        },
    }
    write_json_atomic(report_manifest, output / "manifest.json")
    verify_parent_pins(config)
    return output / "manifest.json"


def status(config: PiTimingConfig) -> dict[str, object]:
    parent_error = None
    try:
        verify_parent_pins(config)
    except ValueError as error:
        parent_error = str(error)
    return {
        "config_identity_sha256": config.identity_sha256,
        "expected_cells": config.expected_cells,
        "expected_measured_samples_if_all_completed": config.expected_cells * config.measured_event_count,
        "parent_pins_valid": parent_error is None,
        "parent_error": parent_error,
        "output": str(config.output),
        "combined_output": str(config.combined_output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "bundle",
            "preflight",
            "contract-tests",
            "smoke",
            "run",
            "profile",
            "pack",
            "import-results",
            "combine-report",
            "status",
            "_cell",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--pi-output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--method", choices=METHOD_NAMES)
    parser.add_argument("--order-index", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-non-pi", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config, smoke=args.smoke or args.command == "smoke")

    if args.command == "status":
        print(json.dumps(status(config), indent=2))
        return 0
    if args.command == "bundle":
        if args.archive is None:
            parser.error("bundle requires --archive")
        build_bundle(config, args.archive)
        return 0
    if args.bundle is None and args.command in {"preflight", "contract-tests", "smoke", "run", "profile", "_cell"}:
        parser.error(f"{args.command} requires --bundle")
    if args.command == "preflight":
        print(json.dumps(preflight(args.bundle, allow_non_pi=args.allow_non_pi), indent=2))
        return 0
    if args.command == "contract-tests":
        print(json.dumps(run_contract_tests(args.bundle), indent=2))
        return 0
    if args.command == "_cell":
        if None in (
            args.run_output,
            args.seed,
            args.budget,
            args.method,
            args.order_index,
        ):
            parser.error(
                "_cell requires --run-output, --seed, --budget, --method, "
                "and --order-index"
            )
        execute_cell(
            args.bundle,
            args.run_output,
            args.seed,
            args.budget,
            args.method,
            args.order_index,
        )
        return 0
    if args.command in {"smoke", "run", "profile"}:
        output = args.run_output or (
            args.bundle.parent
            / f"{args.bundle.name}-run"
            / "paper-evaluation-v4-pi-timing-v1"
        )
        if args.command in {"smoke", "run"}:
            print(run_matrix(
                config,
                args.bundle,
                output,
                allow_non_pi=args.allow_non_pi,
            ))
            if args.command == "smoke":
                print(run_profile(config, args.bundle, output))
        else:
            print(run_profile(config, args.bundle, output))
        return 0
    if args.command == "pack":
        if args.run_output is None or args.archive is None:
            parser.error("pack requires --run-output and --archive")
        pack_results(args.run_output, args.archive)
        return 0
    if args.command == "import-results":
        if args.archive is None:
            parser.error("import-results requires --archive")
        print(import_results(config, args.archive))
        return 0
    if args.command == "combine-report":
        if args.pi_output is None:
            parser.error("combine-report requires --pi-output")
        print(combine_report(config, args.pi_output))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
