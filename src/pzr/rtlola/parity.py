"""Independent RLolaEval notebook parity and throughput validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from pzr.artifact_io import write_csv_atomic, write_json_atomic
from pzr.rtlola.actions import default_action_catalog
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
    require_binding,
)
from pzr.rtlola.engine import (
    RtlolaApproximationReference,
    RtlolaEngine,
    RtlolaEvent,
)
from pzr.rtlola.reference import (
    RtlolaReferenceStep,
    load_or_compute_reference,
)
from pzr.rtlola.robot_arm import (
    ARM_SPEC,
    RLOLAEVAL_REVISION,
    ROBOT_ARM_SPEC_SHA256,
    ROBOT_ARM_TRACE_SHA256,
    TRACE_KINDS,
)
from pzr.rtlola.scenarios import scenario_by_name
from pzr.rtlola.search import MPC_VARIANTS, search_mpc_variant


PARITY_SCHEMA = "pzr.rlolaeval-parity.v2"
PARITY_BOUNDS = (15, 30, 50, 100, 200, 500)
STATIC_METHODS = (
    "girard", "scott", "interval_hull", "pca", "combastel", "interval",
)
BEAM_METHODS = ("girard", "scott", "pca", "combastel")
BEAM_HORIZONS = (1, 5)
BEAM_WIDTH = 5
FLOAT_RTOL = 1e-12
FLOAT_ATOL = 1e-15
MIN_MEDIAN_THROUGHPUT_RATIO = 0.80
MIN_CELL_THROUGHPUT_RATIO = 0.50

RL_TRACE_LABELS = {
    "figure8": ("Figure-8", "compliant"),
    "figure8_drift": ("Figure-8", "drift"),
    "figure8_geofence": ("Figure-8", "geofence"),
    "figure8_drift_geofence": ("Figure-8", "drift & geo."),
    "random": ("Random", "compliant"),
    "random_drift": ("Random", "drift"),
    "random_geofence": ("Random", "geofence"),
    "random_drift_geofence": ("Random", "drift & geo."),
    "square": ("Square", "compliant"),
    "square_drift": ("Square", "drift"),
    "square_geofence": ("Square", "geofence"),
    "square_drift_geofence": ("Square", "drift & geo."),
}


@dataclass(frozen=True)
class ParityConfig:
    rlola_eval: Path
    output: Path
    trace_kinds: tuple[str, ...] = TRACE_KINDS
    bounds: tuple[int, ...] = PARITY_BOUNDS
    run_speed_gate: bool = True

    def __post_init__(self) -> None:
        if not self.trace_kinds or not self.bounds:
            raise ValueError("parity traces and bounds must be non-empty")
        if len(set(self.trace_kinds)) != len(self.trace_kinds):
            raise ValueError("parity traces must be unique")
        if set(self.trace_kinds) - set(TRACE_KINDS):
            raise ValueError("parity contains a non-canonical trace")
        if len(set(self.bounds)) != len(self.bounds) or min(self.bounds) < 0:
            raise ValueError("parity bounds must be unique and non-negative")


@dataclass
class _Series:
    failed: int | None
    losses: list[float]
    triggers: list[bool]
    choices: list[str]
    reduction_required: list[bool]
    state_hashes: list[str]
    elapsed: float

    def payload(self, reference_triggers: Sequence[bool]) -> dict[str, object]:
        count = len(self.losses)
        accumulated = float(sum(self.losses))
        return {
            "failed": self.failed,
            "acc_loss": accumulated,
            "avg_loss": accumulated / count if count else 0.0,
            "final_loss": self.losses[-1] if self.losses else 0.0,
            "ev_per_sec": count / self.elapsed if self.elapsed > 0.0 else 0.0,
            "elapsed": self.elapsed,
            "triggers": self.triggers,
            "ref_triggers": list(reference_triggers[:len(self.triggers)]),
        }


def run_parity(config: ParityConfig) -> pd.DataFrame:
    """Run or resume all notebook-faithful correctness and speed cells."""
    upstream = _validate_upstream(config.rlola_eval)
    scenario = scenario_by_name("robot_arm")
    fingerprint_payload = {
        "schema": PARITY_SCHEMA,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "rlolaeval_revision": upstream["revision"],
        "spec_sha256": ROBOT_ARM_SPEC_SHA256,
        "trace_sha256": {
            name: ROBOT_ARM_TRACE_SHA256[name] for name in config.trace_kinds
        },
        "trace_kinds": list(config.trace_kinds),
        "bounds": list(config.bounds),
        "static_methods": list(STATIC_METHODS),
        "beam_methods": list(BEAM_METHODS),
        "beam_horizons": list(BEAM_HORIZONS),
        "beam_width": BEAM_WIDTH,
        "float_rtol": FLOAT_RTOL,
        "float_atol": FLOAT_ATOL,
    }
    fingerprint = _payload_sha256(fingerprint_payload)
    manifest_path = config.output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("fingerprint") != fingerprint:
            raise ValueError(f"stale parity output directory: {config.output}")
    config.output.mkdir(parents=True, exist_ok=True)
    write_json_atomic({
        **fingerprint_payload,
        "fingerprint": fingerprint,
        "status": "running",
    }, manifest_path)

    golden_static = json.loads((
        config.rlola_eval / "result_cache/robot_arm/all_results.json"
    ).read_text())
    golden_beam = {
        horizon: json.loads((
            config.rlola_eval
            / f"result_cache/robot_arm/bs_results_h{horizon}.json"
        ).read_text())
        for horizon in BEAM_HORIZONS
    }
    _validate_golden_matrix(golden_static, golden_beam)

    rows: list[dict[str, object]] = []
    cell_index = 0
    for trace_kind in config.trace_kinds:
        print(f"parity: preparing exact reference for {trace_kind}", flush=True)
        generated = scenario.generate_trace(
            0, 0, trace_kind=trace_kind,
        )
        trace = generated.events
        reference = load_or_compute_reference(
            trace,
            scenario=scenario,
            trace_kind=trace_kind,
            seed=0,
            cache_path=(config.output / "references" / f"{trace_kind}.json"),
            include_approximation=True,
        )
        _warm_binding(trace, reference)
        ref_triggers = [_has_trigger(step.verdicts) for step in reference]
        print(
            f"parity: reference ready for {trace_kind} ({len(trace)} events)",
            flush=True,
        )
        for bound in config.bounds:
            for method in STATIC_METHODS:
                cell_path = (
                    config.output / "cells" / "static" / trace_kind
                    / f"bound-{bound}" / f"{method}.json"
                )
                golden = golden_static[str(bound)][
                    _static_golden_key(method, trace_kind)
                ]
                row = _load_or_run_cell(
                    cell_path,
                    fingerprint=fingerprint,
                    run=lambda reverse=bool(cell_index % 2): _run_static_cell(
                        trace,
                        reference,
                        ref_triggers,
                        method=method,
                        bound=bound,
                        golden=golden,
                        reverse=reverse,
                    ),
                )
                rows.append({
                    "kind": "static", "trace_kind": trace_kind,
                    "bound": bound, "method": method, "horizon": 0,
                    **row,
                })
                cell_index += 1
            for horizon in BEAM_HORIZONS:
                cell_path = (
                    config.output / "cells" / "beam" / trace_kind
                    / f"bound-{bound}" / f"horizon-{horizon}.json"
                )
                golden = golden_beam[horizon][
                    _beam_golden_key(bound, trace_kind)
                ]
                row = _load_or_run_cell(
                    cell_path,
                    fingerprint=fingerprint,
                    run=lambda reverse=bool(cell_index % 2): _run_beam_cell(
                        trace,
                        reference,
                        ref_triggers,
                        bound=bound,
                        horizon=horizon,
                        golden=golden,
                        reverse=reverse,
                    ),
                )
                rows.append({
                    "kind": "beam", "trace_kind": trace_kind,
                    "bound": bound, "method": "mpc_cumulative_beam",
                    "horizon": horizon,
                    **row,
                })
                cell_index += 1
            print(
                f"parity: completed {trace_kind} at bound {bound}",
                flush=True,
            )

    summary = pd.DataFrame(rows)
    write_csv_atomic(summary, config.output / "parity_cells.csv")
    correctness_passed = bool(summary["correctness_passed"].all())
    speed = _speed_panel(summary)
    write_csv_atomic(speed, config.output / "speed_panel.csv")
    speed_passed = (
        _speed_gate_passed(speed, trace_count=len(config.trace_kinds))
        if config.run_speed_gate else True
    )
    write_json_atomic({
        **fingerprint_payload,
        "fingerprint": fingerprint,
        "status": "complete" if correctness_passed and speed_passed else "failed",
        "correctness_passed": correctness_passed,
        "archived_golden_match_count": int(
            summary["production_matches_golden"].sum()
        ),
        "archived_golden_cell_count": len(summary),
        "speed_gate_enabled": config.run_speed_gate,
        "speed_passed": speed_passed,
        "expected_cell_count": (
            len(config.trace_kinds)
            * len(config.bounds)
            * (len(STATIC_METHODS) + len(BEAM_HORIZONS))
        ),
        "actual_cell_count": len(summary),
        "speed_median_vs_notebook": _speed_ratio_stat(speed, "median"),
        "speed_min_vs_notebook": _speed_ratio_stat(speed, "min"),
    }, manifest_path)
    if not correctness_passed:
        failures = summary.loc[
            ~summary["correctness_passed"],
            ["kind", "trace_kind", "bound", "method", "horizon", "message"],
        ]
        raise AssertionError(
            "RLolaEval correctness parity failed:\n" + failures.to_string(index=False)
        )
    if not speed_passed:
        raise AssertionError(
            "RLolaEval throughput gate failed: median production/notebook must be "
            f">={MIN_MEDIAN_THROUGHPUT_RATIO:.2f} and every cell must be "
            f">={MIN_CELL_THROUGHPUT_RATIO:.2f}"
        )
    return summary


def _run_static_cell(
    trace: Sequence[RtlolaEvent],
    reference: Sequence[RtlolaReferenceStep],
    reference_triggers: Sequence[bool],
    *,
    method: str,
    bound: int,
    golden: dict[str, object],
    reverse: bool,
) -> dict[str, object]:
    functions: tuple[Callable[[], _Series], Callable[[], _Series]] = (
        lambda: _run_static_oracle(trace, reference, method=method, bound=bound),
        lambda: _run_static_production(trace, reference, method=method, bound=bound),
    )
    if reverse:
        production, oracle = functions[1](), functions[0]()
    else:
        oracle, production = functions[0](), functions[1]()
    return _compare_cell(
        oracle,
        production,
        reference_triggers,
        golden,
        compare_choices=False,
    )


def _run_beam_cell(
    trace: Sequence[RtlolaEvent],
    reference: Sequence[RtlolaReferenceStep],
    reference_triggers: Sequence[bool],
    *,
    bound: int,
    horizon: int,
    golden: dict[str, object],
    reverse: bool,
) -> dict[str, object]:
    functions: tuple[Callable[[], _Series], Callable[[], _Series]] = (
        lambda: _run_beam_oracle(trace, reference, bound=bound, horizon=horizon),
        lambda: _run_beam_production(trace, reference, bound=bound, horizon=horizon),
    )
    if reverse:
        production, oracle = functions[1](), functions[0]()
    else:
        oracle, production = functions[0](), functions[1]()
    return _compare_cell(
        oracle,
        production,
        reference_triggers,
        golden,
        compare_choices=True,
    )


def _run_static_oracle(
    trace: Sequence[RtlolaEvent],
    reference: Sequence[RtlolaReferenceStep],
    *,
    method: str,
    bound: int,
) -> _Series:
    _, RLolaMonitor, ZonotopeConfig = require_binding()
    monitor = RLolaMonitor(ARM_SPEC)
    losses: list[float] = []
    triggers: list[bool] = []
    choices: list[str] = []
    reductions: list[bool] = []
    hashes: list[str] = []
    failed = None
    start = time.perf_counter()
    for index, (event, exact) in enumerate(zip(trace, reference)):
        before = np.asarray(monitor.current_zonotope(False), dtype=np.float64)
        reduce = before.shape[1] - 1 > bound
        try:
            config = _notebook_config(ZonotopeConfig, method, bound)
            verdict = monitor.accept_event(
                list(event.values), float(event.time), config,
            )
            matrix = np.asarray(monitor.current_zonotope(False), dtype=np.float64)
            loss = float(monitor.approx_loss(_exact_interval(exact), False))
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            failed = index
            break
        losses.append(loss)
        triggers.append(_has_trigger(verdict))
        choices.append(method if reduce else "none")
        reductions.append(reduce)
        hashes.append(_matrix_sha256(matrix))
    return _Series(
        failed, losses, triggers, choices, reductions, hashes,
        time.perf_counter() - start,
    )


def _run_static_production(
    trace: Sequence[RtlolaEvent],
    reference: Sequence[RtlolaReferenceStep],
    *,
    method: str,
    bound: int,
) -> _Series:
    scenario = scenario_by_name("robot_arm")
    engine = RtlolaEngine(
        scenario.spec,
        event_arity=scenario.event_arity,
        expected_verdict_keys=scenario.expected_verdict_keys,
    )
    catalog = default_action_catalog()
    action = catalog.by_name[method]
    losses: list[float] = []
    triggers: list[bool] = []
    choices: list[str] = []
    reductions: list[bool] = []
    hashes: list[str] = []
    failed = None
    start = time.perf_counter()
    for index, (event, exact) in enumerate(zip(trace, reference)):
        state = engine.snapshot(step=index, time=event.time)
        dynamic, _ = engine.matrices(state)
        reduce = dynamic.shape[1] - 1 > bound
        selected = action
        try:
            committed = engine.live_step(
                event, selected, bound, step=index + 1,
            )
            loss = engine.approx_loss_reference(
                _require_approximation(exact),
                committed.state,
                include_constant_slack=False,
            )
            matrix, _ = engine.matrices(committed.state)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            failed = index
            break
        losses.append(loss)
        triggers.append(_has_trigger(committed.verdict))
        choices.append(selected.name)
        reductions.append(reduce)
        hashes.append(_matrix_sha256(matrix))
    return _Series(
        failed, losses, triggers, choices, reductions, hashes,
        time.perf_counter() - start,
    )


def _run_beam_oracle(
    trace: Sequence[RtlolaEvent],
    reference: Sequence[RtlolaReferenceStep],
    *,
    bound: int,
    horizon: int,
) -> _Series:
    _, RLolaMonitor, ZonotopeConfig = require_binding()
    monitor = RLolaMonitor(ARM_SPEC)
    current_state = monitor.state()
    losses: list[float] = []
    triggers: list[bool] = []
    choices: list[str] = []
    reductions: list[bool] = []
    hashes: list[str] = []
    failed = None
    start = time.perf_counter()
    for index, event in enumerate(trace):
        before = np.asarray(
            monitor.state_zonotope(current_state, False), dtype=np.float64,
        )
        reduce = before.shape[1] - 1 > bound
        depth = min(horizon, len(trace) - index)
        beam: list[tuple[float, Any, tuple[str, ...]]] = [
            (0.0, current_state, ()),
        ]
        try:
            for offset in range(depth):
                future = trace[index + offset]
                exact = reference[index + offset]
                candidates: list[tuple[float, Any, tuple[str, ...]]] = []
                for cumulative, parent, sequence in beam:
                    for method in BEAM_METHODS:
                        try:
                            verdict, state = monitor.accept_event_from_state(
                                parent,
                                list(future.values),
                                float(future.time),
                                _notebook_config(ZonotopeConfig, method, bound),
                            )
                            _ = verdict
                            loss = float(monitor.approx_loss(
                                _exact_interval(exact), False,
                            ))
                            candidates.append((
                                cumulative + loss, state, (*sequence, method),
                            ))
                        except BaseException as exc:
                            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                                raise
                candidates.sort(key=lambda item: item[0])
                beam = candidates[:BEAM_WIDTH]
            if not beam or not beam[0][2]:
                raise RuntimeError("notebook beam produced no complete sequence")
            selected = beam[0][2][0]
            verdict, current_state = monitor.accept_event_from_state(
                current_state,
                list(event.values),
                float(event.time),
                _notebook_config(ZonotopeConfig, selected, bound),
            )
            loss = float(monitor.approx_loss(
                _exact_interval(reference[index]), False,
            ))
            matrix = np.asarray(
                monitor.state_zonotope(current_state, False), dtype=np.float64,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            failed = index
            break
        losses.append(loss)
        triggers.append(_has_trigger(verdict))
        choices.append(selected)
        reductions.append(reduce)
        hashes.append(_matrix_sha256(matrix))
    return _Series(
        failed, losses, triggers, choices, reductions, hashes,
        time.perf_counter() - start,
    )


def _run_beam_production(
    trace: Sequence[RtlolaEvent],
    reference: Sequence[RtlolaReferenceStep],
    *,
    bound: int,
    horizon: int,
) -> _Series:
    scenario = scenario_by_name("robot_arm")
    engine = RtlolaEngine(
        scenario.spec,
        event_arity=scenario.event_arity,
        expected_verdict_keys=scenario.expected_verdict_keys,
    )
    catalog = default_action_catalog(BEAM_METHODS)
    losses: list[float] = []
    triggers: list[bool] = []
    choices: list[str] = []
    reductions: list[bool] = []
    hashes: list[str] = []
    failed = None
    start = time.perf_counter()
    for index, event in enumerate(trace):
        state = engine.snapshot(step=index, time=event.time)
        dynamic, _ = engine.matrices(state)
        reduce = dynamic.shape[1] - 1 > bound
        depth = min(horizon, len(trace) - index)
        future = tuple(trace[index + 1:index + depth])
        exact = tuple(
            _require_approximation(item)
            for item in reference[index:index + depth]
        )
        try:
            decision = search_mpc_variant(
                engine,
                state,
                event,
                future,
                (),
                catalog.mpc_candidates,
                bound,
                BEAM_WIDTH,
                variant=MPC_VARIANTS["mpc_cumulative_beam"],
                root_beam_width=BEAM_WIDTH,
                fallback=catalog.fallback,
                none_action=catalog.no_op,
                tail_action=catalog.by_name["girard"],
                configured_horizon=horizon - 1,
                configured_tail_horizon=0,
                reference_steps=exact,
                include_constant_slack=False,
                automatic_none=False,
            )
            if decision.fallback_used:
                raise RuntimeError("production parity beam used interval fallback")
            committed = engine.live_step(
                event,
                decision.first_action,
                decision.first_action_budget,
                step=index + 1,
            )
            loss = engine.approx_loss_reference(
                exact[0], committed.state, include_constant_slack=False,
            )
            matrix, _ = engine.matrices(committed.state)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            failed = index
            break
        losses.append(loss)
        triggers.append(_has_trigger(committed.verdict))
        choices.append(decision.first_action.name)
        reductions.append(reduce)
        hashes.append(_matrix_sha256(matrix))
    return _Series(
        failed, losses, triggers, choices, reductions, hashes,
        time.perf_counter() - start,
    )


def _compare_cell(
    oracle: _Series,
    production: _Series,
    reference_triggers: Sequence[bool],
    golden: dict[str, object],
    *,
    compare_choices: bool,
) -> dict[str, object]:
    oracle_payload = oracle.payload(reference_triggers)
    production_payload = production.payload(reference_triggers)
    implementation_oracle = _payloads_match(
        production_payload, oracle_payload,
    )
    oracle_golden = _payload_matches_golden(oracle_payload, golden)
    production_golden = _payload_matches_golden(production_payload, golden)
    state_mismatches = [
        index for index, (left, right) in enumerate(zip(
            oracle.state_hashes, production.state_hashes,
        )) if left != right
    ]
    choice_mismatches = []
    if compare_choices:
        choice_mismatches = [
            index for index, (left, right, required) in enumerate(zip(
                oracle.choices,
                production.choices,
                oracle.reduction_required,
            )) if required and left != right
        ]
    lengths_match = (
        len(oracle.state_hashes) == len(production.state_hashes)
        and oracle.failed == production.failed
    )
    passed = bool(
        implementation_oracle
        and lengths_match
        and not state_mismatches
        and not choice_mismatches
    )
    messages = []
    if not implementation_oracle:
        messages.append("production metrics differ from current notebook oracle")
    if not oracle_golden or not production_golden:
        messages.append("archived upstream JSON differs from current runtime")
    if not lengths_match:
        messages.append("oracle/production completion differs")
    if state_mismatches:
        messages.append(f"state mismatches at {state_mismatches[:8]}")
    if choice_mismatches:
        messages.append(f"reducer mismatches at {choice_mismatches[:8]}")
    return {
        "correctness_passed": passed,
        "implementation_matches_oracle": implementation_oracle,
        "oracle_matches_golden": oracle_golden,
        "production_matches_golden": production_golden,
        "oracle_failed": oracle.failed,
        "production_failed": production.failed,
        "oracle_ev_per_sec": oracle_payload["ev_per_sec"],
        "production_ev_per_sec": production_payload["ev_per_sec"],
        "golden_ev_per_sec": golden["ev_per_sec"],
        "oracle_acc_loss": oracle_payload["acc_loss"],
        "production_acc_loss": production_payload["acc_loss"],
        "golden_acc_loss": golden["acc_loss"],
        "state_mismatch_count": len(state_mismatches),
        "choice_mismatch_count": len(choice_mismatches),
        "message": "; ".join(messages),
    }


def _payload_matches_golden(
    actual: dict[str, object],
    golden: dict[str, object],
) -> bool:
    if actual["failed"] != golden["failed"]:
        return False
    if actual["triggers"] != golden["triggers"]:
        return False
    if actual["ref_triggers"] != golden["ref_triggers"]:
        return False
    return all(np.isclose(
        float(actual[key]),
        float(golden[key]),
        rtol=FLOAT_RTOL,
        atol=FLOAT_ATOL,
    ) for key in ("acc_loss", "avg_loss", "final_loss"))


def _payloads_match(
    actual: dict[str, object],
    expected: dict[str, object],
) -> bool:
    """Compare two runs while deliberately excluding measured throughput."""
    if actual["failed"] != expected["failed"]:
        return False
    if actual["triggers"] != expected["triggers"]:
        return False
    if actual["ref_triggers"] != expected["ref_triggers"]:
        return False
    return all(np.isclose(
        float(actual[key]),
        float(expected[key]),
        rtol=FLOAT_RTOL,
        atol=FLOAT_ATOL,
    ) for key in ("acc_loss", "avg_loss", "final_loss"))


def _load_or_run_cell(
    path: Path,
    *,
    fingerprint: str,
    run: Callable[[], dict[str, object]],
) -> dict[str, object]:
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("fingerprint") != fingerprint:
            raise ValueError(f"stale parity cell: {path}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"invalid parity cell: {path}")
        return result
    result = run()
    write_json_atomic({"fingerprint": fingerprint, "result": result}, path)
    return result


def _validate_upstream(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise ValueError(f"RLolaEval checkout does not exist: {root}")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != RLOLAEVAL_REVISION:
        raise ValueError(
            f"RLolaEval revision mismatch: {revision} != {RLOLAEVAL_REVISION}"
        )
    spec = root / "specs/robot_arm/ternary.lola"
    if _file_sha256(spec) != ROBOT_ARM_SPEC_SHA256:
        raise ValueError("RLolaEval robot-arm specification hash differs")
    if (root / "beam_search.ipynb").read_bytes().startswith(
        b"version https://git-lfs.github.com/spec/v1\n",
    ):
        raise ValueError("beam_search.ipynb is still an LFS pointer")
    for trace_kind, expected in ROBOT_ARM_TRACE_SHA256.items():
        actual = _file_sha256(root / f"traces/robot_arm/{trace_kind}.csv")
        if actual != expected:
            raise ValueError(f"RLolaEval trace hash differs: {trace_kind}")
    require_binding()
    return {"revision": revision}


def _validate_golden_matrix(
    static: dict[str, object],
    beam: dict[int, dict[str, object]],
) -> None:
    if set(static) != {str(bound) for bound in PARITY_BOUNDS}:
        raise ValueError("RLolaEval static golden bounds differ")
    if sum(len(value) for value in static.values()) != 432:  # type: ignore[arg-type]
        raise ValueError("RLolaEval static golden matrix must contain 432 cells")
    if set(beam) != set(BEAM_HORIZONS) or any(
        len(value) != 72 for value in beam.values()
    ):
        raise ValueError("RLolaEval beam golden matrix must contain 144 cells")


def _speed_panel(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary[summary["bound"] == 50].copy()
    if selected.empty:
        return pd.DataFrame(columns=(
            "kind", "trace_kind", "method", "horizon",
            "oracle_ev_per_sec", "production_ev_per_sec", "production_failed",
            "production_over_oracle", "production_over_golden",
        ))
    selected["production_over_oracle"] = (
        selected["production_ev_per_sec"] / selected["oracle_ev_per_sec"]
    )
    selected["production_over_golden"] = (
        selected["production_ev_per_sec"] / selected["golden_ev_per_sec"]
    )
    return selected[[
        "kind", "trace_kind", "method", "horizon",
        "oracle_ev_per_sec", "production_ev_per_sec", "production_failed",
        "production_over_oracle", "production_over_golden",
    ]].reset_index(drop=True)


def _speed_gate_passed(speed: pd.DataFrame, *, trace_count: int) -> bool:
    if len(speed) != trace_count * (len(STATIC_METHODS) + len(BEAM_HORIZONS)):
        return False
    completed = speed.loc[speed["production_failed"].isna()]
    if completed.empty:
        return False
    ratios = completed["production_over_golden"].to_numpy(dtype=np.float64)
    return bool(
        np.all(np.isfinite(ratios))
        and np.median(ratios) >= MIN_MEDIAN_THROUGHPUT_RATIO
        and np.min(ratios) >= MIN_CELL_THROUGHPUT_RATIO
    )


def _speed_ratio_stat(speed: pd.DataFrame, operation: str) -> float:
    completed = speed.loc[speed["production_failed"].isna()]
    if completed.empty:
        return float("nan")
    values = completed["production_over_golden"]
    if operation == "median":
        return float(values.median())
    if operation == "min":
        return float(values.min())
    raise ValueError(f"unsupported speed statistic: {operation}")


def _warm_binding(
    trace: Sequence[RtlolaEvent],
    reference: Sequence[RtlolaReferenceStep],
) -> None:
    """Warm native dispatch and loss paths outside timed cells."""
    if not trace:
        return
    _, RLolaMonitor, ZonotopeConfig = require_binding()
    monitor = RLolaMonitor(ARM_SPEC)
    event = trace[0]
    monitor.accept_event(
        list(event.values), float(event.time), ZonotopeConfig.girard(50),
    )
    monitor.approx_loss(_exact_interval(reference[0]), False)


def _exact_interval(step: RtlolaReferenceStep) -> np.ndarray:
    exact = _require_approximation(step)
    dimension = exact.center.size
    matrix = np.zeros((dimension, dimension + 1), dtype=np.float64)
    matrix[:, 0] = exact.center
    matrix[np.arange(dimension), np.arange(dimension) + 1] = (
        exact.dynamic_radius
    )
    return matrix


def _require_approximation(
    step: RtlolaReferenceStep,
) -> RtlolaApproximationReference:
    if step.approximation is None:
        raise ValueError("parity requires dynamic exact-reference intervals")
    return step.approximation


def _notebook_config(config_type: Any, method: str, bound: int) -> Any:
    """Call the notebook's bound-passing factory convention exactly."""
    if method == "interval":
        return config_type.interval()
    return getattr(config_type, method)(bound)


def _has_trigger(verdict: dict[str, object]) -> bool:
    return any(
        key.startswith("Trigger") and value is not False
        for key, value in verdict.items()
    )


def _matrix_sha256(matrix: np.ndarray) -> str:
    value = np.ascontiguousarray(matrix, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _static_golden_key(method: str, trace_kind: str) -> str:
    shape, violation = RL_TRACE_LABELS[trace_kind]
    return f"{method}|{shape}|{violation}"


def _beam_golden_key(bound: int, trace_kind: str) -> str:
    shape, violation = RL_TRACE_LABELS[trace_kind]
    return f"{bound}|{shape}|{violation}"


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("value must not be empty")
    return result


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bounds must be integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("bounds must not be empty")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate PZR against the pinned RLolaEval notebook results",
    )
    parser.add_argument("--rlola-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trace-kinds", type=_parse_csv_strings, default=TRACE_KINDS,
    )
    parser.add_argument(
        "--bounds", type=_parse_csv_ints, default=PARITY_BOUNDS,
    )
    parser.add_argument("--skip-speed-gate", action="store_true")
    args = parser.parse_args(argv)
    summary = run_parity(ParityConfig(
        rlola_eval=args.rlola_eval,
        output=args.output,
        trace_kinds=tuple(args.trace_kinds),
        bounds=tuple(args.bounds),
        run_speed_gate=not args.skip_speed_gate,
    ))
    print(f"RLolaEval parity complete: {len(summary)} cells at {args.output}")


if __name__ == "__main__":
    main()
