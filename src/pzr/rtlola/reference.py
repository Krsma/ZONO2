"""Offline exact-reference cache handling for RTLola benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from pzr.artifact_io import write_text_atomic
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
    require_binding,
)
from pzr.rtlola.engine import RtlolaApproximationReference, RtlolaEvent
from pzr.rtlola.scenarios import RtlolaScenario


REFERENCE_CACHE_SCHEMA = 2


@dataclass(frozen=True)
class RtlolaReferenceStep:
    """Exact verdicts and optional logical-row native-loss reference."""

    verdicts: dict[str, bool]
    approximation: RtlolaApproximationReference | None = None


def load_or_compute_reference(
    trace: Sequence[RtlolaEvent],
    *,
    scenario: RtlolaScenario,
    trace_kind: str,
    seed: int,
    cache_path: Path | None,
    include_approximation: bool,
) -> tuple[RtlolaReferenceStep, ...]:
    """Load or compute exact trigger and logical-row approximation references."""
    base_metadata = _reference_metadata(
        trace,
        scenario=scenario,
        trace_kind=trace_kind,
        seed=seed,
    )
    if cache_path is not None and cache_path.exists():
        return _load_reference_cache(
            cache_path,
            trace=trace,
            scenario=scenario,
            base_metadata=base_metadata,
            include_approximation=include_approximation,
        )

    result = _compute_reference(
        trace,
        scenario=scenario,
        base_metadata=base_metadata,
        include_approximation=include_approximation,
    )
    if cache_path is not None:
        _write_reference_cache(
            cache_path,
            steps=result,
            base_metadata=base_metadata,
            include_approximation=include_approximation,
        )
    return result


def reference_cache_path(
    value: str | None,
    seed: int,
    seed_count: int,
) -> Path | None:
    """Return the cache path for one seed, preserving historical naming."""
    if value is None:
        return None
    path = Path(value)
    if seed_count == 1:
        return path
    return path.with_name(f"{path.stem}.seed_{seed}{path.suffix}")


def trace_sha256(trace: Sequence[RtlolaEvent]) -> str:
    payload = [
        [float(event.time), [
            None if value is None else float(value)
            for value in event.values
        ]]
        for event in trace
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()


def _reference_metadata(
    trace: Sequence[RtlolaEvent],
    *,
    scenario: RtlolaScenario,
    trace_kind: str,
    seed: int,
) -> dict[str, object]:
    selected_trace = (
        scenario.default_trace_kind
        if trace_kind == "default" else trace_kind
    )
    return {
        "schema": REFERENCE_CACHE_SCHEMA,
        "scenario": scenario.name,
        "trace_kind": selected_trace,
        "seed": int(seed),
        "length": len(trace),
        "trace_sha256": trace_sha256(trace),
        "spec_sha256": hashlib.sha256(scenario.spec.encode("utf-8")).hexdigest(),
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "trigger_keys": list(scenario.trigger_keys),
    }


def _load_reference_cache(
    cache_path: Path,
    *,
    trace: Sequence[RtlolaEvent],
    scenario: RtlolaScenario,
    base_metadata: dict[str, object],
    include_approximation: bool,
) -> tuple[RtlolaReferenceStep, ...]:
    try:
        payload = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid RTLola reference cache: {cache_path}"
        ) from exc
    metadata = payload.get("metadata")
    capabilities = (
        metadata.get("capabilities")
        if isinstance(metadata, dict) else None
    )
    actual_base = (
        {key: value for key, value in metadata.items() if key != "capabilities"}
        if isinstance(metadata, dict) else None
    )
    if actual_base != base_metadata:
        raise ValueError(
            f"RTLola reference metadata mismatch: {cache_path}"
        )
    if (
        not isinstance(capabilities, list)
        or "trigger_verdicts" not in capabilities
    ):
        raise ValueError(
            f"RTLola reference capabilities are invalid: {cache_path}"
        )
    if include_approximation and "approx_loss" not in capabilities:
        raise ValueError(
            f"RTLola reference cache lacks approximation data: {cache_path}"
        )
    rows = payload.get("steps")
    if not isinstance(rows, list) or len(rows) != len(trace):
        raise ValueError(
            f"RTLola reference step count mismatch: {cache_path}"
        )
    try:
        parsed: list[RtlolaReferenceStep] = []
        for index, row in enumerate(rows):
            verdict_row = row["verdicts"]
            if not isinstance(verdict_row, dict):
                raise TypeError("verdict row is not a mapping")
            verdicts = {
                key: verdict_row[key]
                for key in scenario.trigger_keys
            }
            if not all(isinstance(value, bool) for value in verdicts.values()):
                raise TypeError("trigger verdict is not boolean")
            approximation = None
            if include_approximation:
                approximation = RtlolaApproximationReference(
                    center=np.asarray(row["center"], dtype=np.float64),
                    radius=np.asarray(row["radius"], dtype=np.float64),
                    spec_id=str(base_metadata["spec_sha256"]),
                    step=index + 1,
                )
            parsed.append(RtlolaReferenceStep(
                verdicts=verdicts,
                approximation=approximation,
            ))
        return tuple(parsed)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid RTLola reference rows: {cache_path}"
        ) from exc


def _compute_reference(
    trace: Sequence[RtlolaEvent],
    *,
    scenario: RtlolaScenario,
    base_metadata: dict[str, object],
    include_approximation: bool,
) -> tuple[RtlolaReferenceStep, ...]:
    _, RLolaMonitor, ZonotopeConfig = require_binding()
    monitor = RLolaMonitor(scenario.spec)
    none = ZonotopeConfig.none()
    steps: list[RtlolaReferenceStep] = []
    for index, event in enumerate(trace):
        verdict = monitor.accept_event(
            list(event.values),
            float(event.time),
            none,
        )
        approximation = None
        if include_approximation:
            matrix = np.asarray(monitor.current_zonotope(True), dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[1] < 1:
                raise RuntimeError(
                    f"invalid exact RTLola zonotope shape at step {index}: {matrix.shape}"
                )
            approximation = RtlolaApproximationReference(
                center=matrix[:, 0],
                radius=np.abs(matrix[:, 1:]).sum(axis=1),
                spec_id=str(base_metadata["spec_sha256"]),
                step=index + 1,
            )
        steps.append(RtlolaReferenceStep(
            verdicts={
                key: bool(verdict.get(key, False))
                for key in scenario.trigger_keys
            },
            approximation=approximation,
        ))
    return tuple(steps)


def _write_reference_cache(
    cache_path: Path,
    *,
    steps: tuple[RtlolaReferenceStep, ...],
    base_metadata: dict[str, object],
    include_approximation: bool,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({
        "metadata": {
            **base_metadata,
            "capabilities": [
                "trigger_verdicts",
                *(["approx_loss"] if include_approximation else []),
            ],
        },
        "steps": [
            {
                "verdicts": step.verdicts,
                **(
                    {
                        "center": step.approximation.center.tolist(),
                        "radius": step.approximation.radius.tolist(),
                    }
                    if step.approximation is not None else {}
                ),
            }
            for step in steps
        ],
    }, indent=2, sort_keys=True)
    write_text_atomic(payload, cache_path)
