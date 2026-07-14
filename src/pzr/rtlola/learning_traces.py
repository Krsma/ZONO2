"""Resumable random-waypoint trace stores for RTLola learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from pzr.learning.provenance import pzr_source_sha256, sha256_files
from pzr.rtlola.robot_arm_random import (
    RANDOM_WAYPOINT_CONDITIONS,
    RANDOM_WAYPOINT_SOURCE_REVISION,
    RandomWaypointConfig,
    RandomWaypointTrace,
    generate_random_waypoint_trace,
    load_random_waypoint_trace,
    write_random_waypoint_trace,
)


RANDOM_WAYPOINT_TRACE_STORE_SCHEMA = "pzr.random-waypoint-trace-store.v1"


@dataclass(frozen=True)
class RandomWaypointTraceStoreConfig:
    output: Path
    event_count: int
    conditions: tuple[str, ...]
    seed_start: int
    seed_count: int

    def __post_init__(self) -> None:
        if self.event_count < 2:
            raise ValueError("event count must be at least two")
        if not self.conditions:
            raise ValueError("at least one random-waypoint condition is required")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("random-waypoint conditions must be unique")
        unknown = set(self.conditions) - set(RANDOM_WAYPOINT_CONDITIONS)
        if unknown:
            raise ValueError(f"unknown random-waypoint conditions: {sorted(unknown)}")
        if self.seed_start < 0 or self.seed_count < 1:
            raise ValueError("seed start must be non-negative and seed count positive")

    @property
    def seeds(self) -> range:
        return range(self.seed_start, self.seed_start + self.seed_count)


@dataclass(frozen=True)
class StoredRandomWaypointTrace:
    trace_id: str
    condition: str
    seed: int
    relative_path: Path
    trace: RandomWaypointTrace


@dataclass(frozen=True)
class RandomWaypointTraceStore:
    root: Path
    event_count: int
    conditions: tuple[str, ...]
    seed_start: int
    seed_count: int
    traces: tuple[StoredRandomWaypointTrace, ...]
    manifest_sha256: str

    def traces_for_seed(self, seed: int) -> tuple[StoredRandomWaypointTrace, ...]:
        selected = tuple(item for item in self.traces if item.seed == seed)
        if tuple(item.condition for item in selected) != self.conditions:
            raise ValueError(f"trace store does not contain every condition for seed {seed}")
        return selected


def generate_random_waypoint_trace_store(
    config: RandomWaypointTraceStoreConfig,
) -> RandomWaypointTraceStore:
    """Generate missing traces and atomically publish a complete manifest."""
    config.output.mkdir(parents=True, exist_ok=True)
    identity = _store_identity(config)
    manifest_path = config.output / "manifest.json"
    existing = None
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        _validate_manifest_identity(existing, identity)

    records = []
    for seed in config.seeds:
        for condition in config.conditions:
            trace_id = f"{condition}:seed-{seed}"
            relative_path = Path(trace_id)
            trace_config = RandomWaypointConfig(
                seed=seed,
                condition=condition,
                event_count=config.event_count,
            )
            trace = _load_or_generate_trace(
                trace_config,
                config.output / relative_path,
            )
            records.append(_trace_record(trace_id, relative_path, trace))

    manifest = {**identity, "traces": records}
    if existing is not None and existing != manifest:
        raise ValueError("random-waypoint trace-store manifest contents differ")
    if existing is None:
        _write_json_atomic(manifest, manifest_path)
    return load_random_waypoint_trace_store(config.output)


def load_random_waypoint_trace_store(directory: Path) -> RandomWaypointTraceStore:
    """Load a complete trace store and validate every persisted trace."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"random-waypoint trace-store manifest is missing: {directory}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != RANDOM_WAYPOINT_TRACE_STORE_SCHEMA:
        raise ValueError("unsupported random-waypoint trace-store schema")
    config = RandomWaypointTraceStoreConfig(
        output=directory,
        event_count=int(manifest["event_count"]),
        conditions=tuple(str(value) for value in manifest["conditions"]),
        seed_start=int(manifest["seed_start"]),
        seed_count=int(manifest["seed_count"]),
    )
    _validate_manifest_identity(manifest, _store_identity(config))
    event_count = config.event_count
    conditions = config.conditions
    seed_start = config.seed_start
    seed_count = config.seed_count
    expected_ids = tuple(
        f"{condition}:seed-{seed}"
        for seed in range(seed_start, seed_start + seed_count)
        for condition in conditions
    )
    raw_records = manifest.get("traces")
    if not isinstance(raw_records, list) or not all(
        isinstance(record, dict) for record in raw_records
    ):
        raise ValueError("random-waypoint trace-store records are missing")
    actual_ids = tuple(str(record.get("trace_id")) for record in raw_records)
    if actual_ids != expected_ids:
        raise ValueError("random-waypoint trace-store seed coverage differs")

    stored = []
    for record in raw_records:
        trace_id = str(record["trace_id"])
        relative_path = Path(str(record["relative_path"]))
        if relative_path != Path(trace_id):
            raise ValueError(f"random-waypoint trace path differs for {trace_id}")
        trace = load_random_waypoint_trace(directory / relative_path)
        condition = str(record["condition"])
        seed = int(record["seed"])
        if trace.metadata.condition != condition or trace.metadata.seed != seed:
            raise ValueError(f"random-waypoint trace identity differs for {trace_id}")
        if trace.metadata.event_count != event_count:
            raise ValueError(f"random-waypoint trace length differs for {trace_id}")
        if trace.metadata.trace_sha256 != str(record["trace_sha256"]):
            raise ValueError(f"random-waypoint trace hash differs for {trace_id}")
        if trace.metadata.generator_config != record.get("generator_config"):
            raise ValueError(f"random-waypoint generator configuration differs for {trace_id}")
        stored.append(StoredRandomWaypointTrace(
            trace_id=trace_id,
            condition=condition,
            seed=seed,
            relative_path=relative_path,
            trace=trace,
        ))

    return RandomWaypointTraceStore(
        root=directory,
        event_count=event_count,
        conditions=conditions,
        seed_start=seed_start,
        seed_count=seed_count,
        traces=tuple(stored),
        manifest_sha256=sha256_files((manifest_path,)),
    )


def _store_identity(config: RandomWaypointTraceStoreConfig) -> dict[str, object]:
    return {
        "schema": RANDOM_WAYPOINT_TRACE_STORE_SCHEMA,
        "scenario": "robot_arm",
        "event_count": config.event_count,
        "conditions": list(config.conditions),
        "seed_start": config.seed_start,
        "seed_count": config.seed_count,
        "seeds": list(config.seeds),
        "random_waypoint_source_revision": RANDOM_WAYPOINT_SOURCE_REVISION,
        "pzr_source_sha256": pzr_source_sha256(),
    }


def _validate_manifest_identity(
    manifest: dict[str, object],
    expected: dict[str, object],
) -> None:
    mismatched = [
        name for name, value in expected.items()
        if manifest.get(name) != value
    ]
    if mismatched:
        raise ValueError(
            "random-waypoint trace-store identity differs for: "
            + ", ".join(sorted(mismatched))
        )


def _load_or_generate_trace(
    config: RandomWaypointConfig,
    directory: Path,
) -> RandomWaypointTrace:
    trace_path = directory / "trace.csv"
    metadata_path = directory / "metadata.json"
    if trace_path.exists() != metadata_path.exists():
        raise ValueError(f"incomplete random-waypoint trace artifact: {directory}")
    if trace_path.exists():
        trace = load_random_waypoint_trace(directory)
        if trace.metadata.generator_config != asdict(config):
            raise ValueError(f"random-waypoint trace configuration differs: {directory}")
        return trace
    trace = generate_random_waypoint_trace(config)
    write_random_waypoint_trace(trace, directory)
    return trace


def _trace_record(
    trace_id: str,
    relative_path: Path,
    trace: RandomWaypointTrace,
) -> dict[str, object]:
    return {
        "trace_id": trace_id,
        "condition": trace.metadata.condition,
        "seed": trace.metadata.seed,
        "event_count": trace.metadata.event_count,
        "relative_path": str(relative_path),
        "trace_sha256": trace.metadata.trace_sha256,
        "attempts": trace.metadata.attempts,
        "max_tracking_error": trace.metadata.max_tracking_error,
        "mujoco_version": trace.metadata.mujoco_version,
        "generator_config": trace.metadata.generator_config,
    }


def _write_json_atomic(payload: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)
