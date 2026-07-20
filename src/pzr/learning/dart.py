"""Calibrated discrete DART for guarded supervisor-noise collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from pzr.artifact_io import write_csv_atomic, write_json_atomic
from pzr.learning.artifacts import load_reducer_cost_dataset
from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.objectives import normalized_regrets, tolerant_best_mask
from pzr.learning.provenance import model_sha256, pzr_source_sha256
from pzr.learning.ranker import ReducerPolicy
from pzr.learning.training import NamedDataset, dataset_sha256
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA


DART_CALIBRATION_SCHEMA = "pzr.dart-calibration.v2"


@dataclass(frozen=True)
class DartCalibrationConfig:
    """Hyperparameters of the guarded categorical DART adaptation."""

    regret_cap_quantile: float = 0.9
    direction_pseudocount: float = 1.0
    recovery_decisions: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.regret_cap_quantile <= 1.0:
            raise ValueError("DART regret-cap quantile must lie in (0, 1]")
        if not np.isfinite(self.direction_pseudocount) or self.direction_pseudocount <= 0.0:
            raise ValueError("DART direction pseudocount must be finite and positive")
        if self.recovery_decisions < 0:
            raise ValueError("DART recovery decisions must be non-negative")


@dataclass(frozen=True)
class DartCalibrationArtifactConfig:
    model: Path
    dataset: NamedDataset
    output: Path
    split: str = "validation"
    calibration: DartCalibrationConfig = DartCalibrationConfig()


@dataclass(frozen=True)
class DartCalibration:
    """Budget-scaled disturbance rates and teacher-conditioned directions."""

    candidate_names: tuple[str, ...]
    budgets: tuple[int, ...]
    direction_probabilities: NDArray[np.float64]
    row_counts: NDArray[np.int64]
    error_counts: NDArray[np.int64]
    target_disturbance_rates: NDArray[np.float64]
    injection_probabilities: NDArray[np.float64]
    regret_caps: NDArray[np.float64]
    expected_disturbance_rates: NDArray[np.float64]
    eligible_fractions: NDArray[np.float64]
    saturated: NDArray[np.bool_]
    config: DartCalibrationConfig
    context: Mapping[str, object]

    def __post_init__(self) -> None:
        candidate_count = len(self.candidate_names)
        budget_count = len(self.budgets)
        expected_directions = (budget_count, candidate_count, candidate_count)
        directions = np.asarray(self.direction_probabilities, dtype=np.float64).copy()
        row_counts = np.asarray(self.row_counts, dtype=np.int64).copy()
        error_counts = np.asarray(self.error_counts, dtype=np.int64).copy()
        vectors = {
            "target disturbance rates": np.asarray(self.target_disturbance_rates, dtype=np.float64).copy(),
            "injection probabilities": np.asarray(self.injection_probabilities, dtype=np.float64).copy(),
            "regret caps": np.asarray(self.regret_caps, dtype=np.float64).copy(),
            "expected disturbance rates": np.asarray(self.expected_disturbance_rates, dtype=np.float64).copy(),
            "eligible fractions": np.asarray(self.eligible_fractions, dtype=np.float64).copy(),
        }
        saturated = np.asarray(self.saturated, dtype=np.bool_).copy()
        if directions.shape != expected_directions:
            raise ValueError("DART directions do not match budget/candidate axes")
        if row_counts.shape != expected_directions[:2] or error_counts.shape != row_counts.shape:
            raise ValueError("DART counts do not match budget/teacher axes")
        if any(values.shape != (budget_count,) for values in vectors.values()):
            raise ValueError("DART budget vectors do not match the budget axis")
        if saturated.shape != (budget_count,):
            raise ValueError("DART saturation flags do not match the budget axis")
        if len(set(self.budgets)) != budget_count or tuple(sorted(self.budgets)) != self.budgets:
            raise ValueError("DART budgets must be sorted and unique")
        if candidate_count < 2 or len(set(self.candidate_names)) != candidate_count:
            raise ValueError("DART needs at least two unique candidates")
        if np.any(directions < 0.0) or not np.all(np.isfinite(directions)):
            raise ValueError("DART direction probabilities must be finite and non-negative")
        if not np.allclose(np.diagonal(directions, axis1=1, axis2=2), 0.0, atol=0.0, rtol=0.0):
            raise ValueError("DART direction kernels must have zero diagonal")
        if not np.allclose(np.sum(directions, axis=2), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("every DART direction row must sum to one")
        if np.any(row_counts < 0) or np.any(error_counts < 0) or np.any(error_counts > row_counts):
            raise ValueError("DART counts must be non-negative and aligned")
        for name, values in vectors.items():
            if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError(f"DART {name} must be finite and lie in [0, 1]")
        for values in (directions, row_counts, error_counts, saturated, *vectors.values()):
            values.setflags(write=False)
        object.__setattr__(self, "direction_probabilities", directions)
        object.__setattr__(self, "row_counts", row_counts)
        object.__setattr__(self, "error_counts", error_counts)
        object.__setattr__(self, "target_disturbance_rates", vectors["target disturbance rates"])
        object.__setattr__(self, "injection_probabilities", vectors["injection probabilities"])
        object.__setattr__(self, "regret_caps", vectors["regret caps"])
        object.__setattr__(self, "expected_disturbance_rates", vectors["expected disturbance rates"])
        object.__setattr__(self, "eligible_fractions", vectors["eligible fractions"])
        object.__setattr__(self, "saturated", saturated)
        object.__setattr__(self, "context", dict(self.context))

    def budget_index(self, budget: int) -> int:
        try:
            return self.budgets.index(int(budget))
        except ValueError as exc:
            raise ValueError("DART calibration does not cover this budget") from exc

    def alternative_distribution(
        self,
        budget: int,
        teacher_action: str,
        feasible: NDArray[np.bool_],
        regrets: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Mask a direction row to feasible, non-teacher, in-radius actions."""
        budget_index = self.budget_index(budget)
        try:
            teacher_index = self.candidate_names.index(teacher_action)
        except ValueError as exc:
            raise ValueError("DART calibration does not cover this teacher action") from exc
        mask = np.asarray(feasible, dtype=np.bool_)
        values = np.asarray(regrets, dtype=np.float64)
        if mask.shape != (len(self.candidate_names),) or values.shape != mask.shape:
            raise ValueError("DART feasibility and regret vectors must match candidates")
        if not mask[teacher_index]:
            raise ValueError("teacher action must be feasible")
        allowed = mask & np.isfinite(values) & (values <= self.regret_caps[budget_index])
        allowed[teacher_index] = False
        distribution = self.direction_probabilities[budget_index, teacher_index].copy()
        distribution[~allowed] = 0.0
        total = float(np.sum(distribution))
        if total > 0.0:
            distribution /= total
        return distribution

    def contract(self) -> dict[str, object]:
        """Return collection semantics suitable for dataset provenance."""
        return {
            "schema": DART_CALIBRATION_SCHEMA,
            "target": "budget_tolerant_novice_error_rate",
            "direction": "teacher_conditioned_meaningful_error_with_dirichlet_pseudocount",
            "scaling": "ordered_validation_expected_rate_with_recovery_cooldown",
            "regret": "tolerance_aware_normalized_regret",
            "config": asdict(self.config),
            "budgets": list(self.budgets),
            "target_disturbance_rates": self.target_disturbance_rates.tolist(),
            "injection_probabilities": self.injection_probabilities.tolist(),
            "regret_caps": self.regret_caps.tolist(),
            "expected_disturbance_rates": self.expected_disturbance_rates.tolist(),
            "eligible_fractions": self.eligible_fractions.tolist(),
            "saturated": self.saturated.tolist(),
        }

    def save(
        self,
        directory: Path,
        budget_diagnostics: pd.DataFrame,
        direction_diagnostics: pd.DataFrame,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": DART_CALIBRATION_SCHEMA,
            "candidate_names": list(self.candidate_names),
            "budgets": list(self.budgets),
            "direction_probabilities": self.direction_probabilities.tolist(),
            "row_counts": self.row_counts.tolist(),
            "error_counts": self.error_counts.tolist(),
            "target_disturbance_rates": self.target_disturbance_rates.tolist(),
            "injection_probabilities": self.injection_probabilities.tolist(),
            "regret_caps": self.regret_caps.tolist(),
            "expected_disturbance_rates": self.expected_disturbance_rates.tolist(),
            "eligible_fractions": self.eligible_fractions.tolist(),
            "saturated": self.saturated.tolist(),
            "config": asdict(self.config),
            "context": dict(self.context),
        }
        write_json_atomic(payload, directory / "calibration.json")
        write_csv_atomic(budget_diagnostics, directory / "dart_budget_calibration.csv")
        write_csv_atomic(direction_diagnostics, directory / "dart_direction_kernel.csv")

    @classmethod
    def load(cls, directory: Path) -> "DartCalibration":
        payload = json.loads((directory / "calibration.json").read_text())
        if payload.get("schema") != DART_CALIBRATION_SCHEMA:
            raise ValueError("unsupported DART calibration schema")
        return cls(
            candidate_names=tuple(payload["candidate_names"]),
            budgets=tuple(int(value) for value in payload["budgets"]),
            direction_probabilities=np.asarray(payload["direction_probabilities"], dtype=np.float64),
            row_counts=np.asarray(payload["row_counts"], dtype=np.int64),
            error_counts=np.asarray(payload["error_counts"], dtype=np.int64),
            target_disturbance_rates=np.asarray(payload["target_disturbance_rates"], dtype=np.float64),
            injection_probabilities=np.asarray(payload["injection_probabilities"], dtype=np.float64),
            regret_caps=np.asarray(payload["regret_caps"], dtype=np.float64),
            expected_disturbance_rates=np.asarray(payload["expected_disturbance_rates"], dtype=np.float64),
            eligible_fractions=np.asarray(payload["eligible_fractions"], dtype=np.float64),
            saturated=np.asarray(payload["saturated"], dtype=np.bool_),
            config=DartCalibrationConfig(**payload["config"]),
            context=dict(payload["context"]),
        )


def calibrate_dart(
    policy: ReducerPolicy,
    dataset: ReducerCostDataset,
    metadata: pd.DataFrame,
    *,
    split: str,
    context: Mapping[str, object],
    config: DartCalibrationConfig = DartCalibrationConfig(),
) -> tuple[DartCalibration, pd.DataFrame, pd.DataFrame]:
    """Fit direction, radius, and DART error scale on ordered held-out states."""
    required = {"split", "budget", "teacher_action", "trace_id", "step"}
    if not required <= set(metadata.columns):
        raise ValueError(f"DART metadata lacks columns: {sorted(required - set(metadata))}")
    if len(metadata) != dataset.num_samples:
        raise ValueError("DART metadata and dataset are not aligned")
    if policy.candidate_names != dataset.candidate_names:
        raise ValueError("DART model and dataset candidate catalogs differ")
    if policy.feature_schema.feature_names != dataset.feature_names:
        raise ValueError("DART model and dataset feature schemas differ")
    split_rows = np.flatnonzero(metadata["split"].astype(str).to_numpy() == split)
    if split_rows.size == 0:
        raise ValueError("DART calibration split is empty")
    candidate_names = dataset.candidate_names
    candidate_index = {name: index for index, name in enumerate(candidate_names)}
    split_teacher = metadata.iloc[split_rows]["teacher_action"].astype(str).to_numpy()
    usable = np.any(dataset.feasible[split_rows], axis=1) & np.isin(split_teacher, candidate_names)
    selected_rows = split_rows[usable]
    if selected_rows.size == 0:
        raise ValueError("DART calibration split has no state with a feasible catalog teacher action")
    selected_metadata = metadata.iloc[selected_rows].reset_index(drop=True)
    budgets = tuple(sorted(int(value) for value in selected_metadata["budget"].unique()))
    budget_index = {budget: index for index, budget in enumerate(budgets)}
    scores = np.asarray(policy.predict_scores(dataset.features[selected_rows]), dtype=np.float64)
    novice = np.argmin(scores, axis=1)
    tie_mask = tolerant_best_mask(dataset.teacher_costs[selected_rows], dataset.feasible[selected_rows])
    regrets = normalized_regrets(dataset.teacher_costs[selected_rows], dataset.feasible[selected_rows])
    teacher = np.asarray([
        candidate_index[str(name)] for name in selected_metadata["teacher_action"]
    ], dtype=np.int64)
    meaningful_error = ~tie_mask[np.arange(selected_rows.size), novice]
    candidate_count = len(candidate_names)
    counts = np.zeros((len(budgets), candidate_count, candidate_count), dtype=np.int64)
    row_counts = np.zeros((len(budgets), candidate_count), dtype=np.int64)
    error_counts = np.zeros_like(row_counts)
    for local in range(selected_rows.size):
        b_index = budget_index[int(selected_metadata.iloc[local]["budget"])]
        teacher_index = int(teacher[local])
        row_counts[b_index, teacher_index] += 1
        if meaningful_error[local]:
            counts[b_index, teacher_index, novice[local]] += 1
            error_counts[b_index, teacher_index] += 1
    directions = np.zeros_like(counts, dtype=np.float64)
    for b_index in range(len(budgets)):
        for teacher_index in range(candidate_count):
            weights = counts[b_index, teacher_index].astype(np.float64)
            weights += config.direction_pseudocount
            weights[teacher_index] = 0.0
            directions[b_index, teacher_index] = weights / np.sum(weights)

    target_rates = np.zeros(len(budgets), dtype=np.float64)
    injection_probabilities = np.zeros(len(budgets), dtype=np.float64)
    regret_caps = np.zeros(len(budgets), dtype=np.float64)
    expected_rates = np.zeros(len(budgets), dtype=np.float64)
    eligible_fractions = np.zeros(len(budgets), dtype=np.float64)
    saturated = np.zeros(len(budgets), dtype=np.bool_)
    budget_rows: list[dict[str, object]] = []
    for b_index, budget in enumerate(budgets):
        local_rows = np.flatnonzero(selected_metadata["budget"].to_numpy() == budget)
        errors = local_rows[meaningful_error[local_rows]]
        selected_regrets = regrets[errors, novice[errors]]
        finite_error_regrets = selected_regrets[np.isfinite(selected_regrets)]
        regret_cap = (
            float(np.quantile(finite_error_regrets, config.regret_cap_quantile, method="linear"))
            if finite_error_regrets.size else 0.0
        )
        eligible = np.zeros(local_rows.size, dtype=np.bool_)
        for position, local in enumerate(local_rows):
            allowed = (
                dataset.feasible[selected_rows[local]]
                & np.isfinite(regrets[local])
                & (regrets[local] <= regret_cap)
            )
            allowed[teacher[local]] = False
            eligible[position] = bool(np.any(allowed))
        sequences = _ordered_local_sequences(selected_metadata.iloc[local_rows])
        target_rate = float(np.mean(meaningful_error[local_rows]))
        maximum_rate = _expected_disturbance_rate(
            1.0, eligible, sequences, config.recovery_decisions,
        )
        is_saturated = target_rate > maximum_rate + 1e-12
        injection_probability = (
            1.0 if is_saturated else _solve_injection_probability(
                target_rate, eligible, sequences, config.recovery_decisions,
            )
        )
        expected_rate = _expected_disturbance_rate(
            injection_probability, eligible, sequences, config.recovery_decisions,
        )
        target_rates[b_index] = target_rate
        injection_probabilities[b_index] = injection_probability
        regret_caps[b_index] = regret_cap
        expected_rates[b_index] = expected_rate
        eligible_fractions[b_index] = float(np.mean(eligible))
        saturated[b_index] = is_saturated
        budget_rows.append({
            "budget": budget,
            "sample_count": int(local_rows.size),
            "meaningful_novice_error_count": int(np.count_nonzero(meaningful_error[local_rows])),
            "infeasible_novice_error_count": int(np.count_nonzero(~np.isfinite(selected_regrets))),
            "target_disturbance_rate": target_rate,
            "regret_cap_quantile": config.regret_cap_quantile,
            "regret_cap": regret_cap,
            "eligible_fraction": eligible_fractions[b_index],
            "injection_probability": injection_probability,
            "expected_disturbance_rate": expected_rate,
            "maximum_expected_disturbance_rate": maximum_rate,
            "saturated": is_saturated,
            "mean_meaningful_novice_regret": (
                float(np.mean(finite_error_regrets)) if finite_error_regrets.size else float("nan")
            ),
            "q50_meaningful_novice_regret": (
                float(np.quantile(finite_error_regrets, 0.5)) if finite_error_regrets.size else float("nan")
            ),
            "q90_meaningful_novice_regret": (
                float(np.quantile(finite_error_regrets, 0.9)) if finite_error_regrets.size else float("nan")
            ),
        })

    calibration = DartCalibration(
        candidate_names=candidate_names,
        budgets=budgets,
        direction_probabilities=directions,
        row_counts=row_counts,
        error_counts=error_counts,
        target_disturbance_rates=target_rates,
        injection_probabilities=injection_probabilities,
        regret_caps=regret_caps,
        expected_disturbance_rates=expected_rates,
        eligible_fractions=eligible_fractions,
        saturated=saturated,
        config=config,
        context=context,
    )
    direction_rows = []
    for b_index, budget in enumerate(budgets):
        for teacher_index, teacher_name in enumerate(candidate_names):
            for action_index, action_name in enumerate(candidate_names):
                direction_rows.append({
                    "budget": budget,
                    "teacher_action": teacher_name,
                    "direction_action": action_name,
                    "teacher_row_count": int(row_counts[b_index, teacher_index]),
                    "meaningful_error_count": int(error_counts[b_index, teacher_index]),
                    "raw_direction_count": int(counts[b_index, teacher_index, action_index]),
                    "direction_probability": float(directions[b_index, teacher_index, action_index]),
                })
    return calibration, pd.DataFrame(budget_rows), pd.DataFrame(direction_rows)


def run_dart_calibration_artifact(config: DartCalibrationArtifactConfig) -> Path:
    """Fit held-out DART calibration and write its complete diagnostics."""
    policy = ReducerPolicy.load(config.model)
    dataset, metadata, manifest = load_reducer_cost_dataset(config.dataset.path)
    context = {
        "model_sha256": model_sha256(config.model),
        "dataset_name": config.dataset.name,
        "dataset_sha256": dataset_sha256(config.dataset.path),
        "split": config.split,
        "candidate_names": list(dataset.candidate_names),
        "feature_schema": {
            "name": RTL_RANKING_FEATURE_SCHEMA.name,
            "version": RTL_RANKING_FEATURE_SCHEMA.version,
            "feature_names": list(RTL_RANKING_FEATURE_SCHEMA.feature_names),
            "log1p_features": list(RTL_RANKING_FEATURE_SCHEMA.log1p_features),
        },
        "cost_contract": manifest["cost_contract"],
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "pzr_source_sha256": pzr_source_sha256(),
    }
    calibration, budget_diagnostics, direction_diagnostics = calibrate_dart(
        policy,
        dataset,
        metadata,
        split=config.split,
        context=context,
        config=config.calibration,
    )
    calibration.save(config.output, budget_diagnostics, direction_diagnostics)
    from pzr.learning.reporting import write_dart_calibration_plot

    write_dart_calibration_plot(
        budget_diagnostics,
        direction_diagnostics,
        config.output / "dart_calibration.png",
    )
    return config.output


def _ordered_local_sequences(metadata: pd.DataFrame) -> tuple[NDArray[np.int64], ...]:
    working = metadata.reset_index(drop=True).copy()
    working["_local_index"] = np.arange(len(working), dtype=np.int64)
    return tuple(
        frame.sort_values("step")["_local_index"].to_numpy(dtype=np.int64)
        for _, frame in working.groupby("trace_id", sort=False)
    )


def _expected_disturbance_rate(
    injection_probability: float,
    eligible: NDArray[np.bool_],
    sequences: tuple[NDArray[np.int64], ...],
    recovery_decisions: int,
) -> float:
    """Expected disturbed fraction under a finite recovery-state Markov chain."""
    if not 0.0 <= injection_probability <= 1.0:
        raise ValueError("DART injection probability must lie in [0, 1]")
    if eligible.size == 0:
        return 0.0
    disturbed_total = 0.0
    for sequence in sequences:
        recovery = np.zeros(recovery_decisions + 1, dtype=np.float64)
        recovery[0] = 1.0
        for local in sequence:
            disturbance = recovery[0] * injection_probability * float(eligible[local])
            disturbed_total += disturbance
            updated = np.zeros_like(recovery)
            updated[0] += recovery[0] - disturbance
            for remaining in range(1, recovery_decisions + 1):
                updated[remaining - 1] += recovery[remaining]
            if recovery_decisions:
                updated[recovery_decisions] += disturbance
            else:
                updated[0] += disturbance
            recovery = updated
    return disturbed_total / eligible.size


def _solve_injection_probability(
    target_rate: float,
    eligible: NDArray[np.bool_],
    sequences: tuple[NDArray[np.int64], ...],
    recovery_decisions: int,
) -> float:
    if target_rate <= 0.0:
        return 0.0
    lower, upper = 0.0, 1.0
    for _ in range(64):
        midpoint = (lower + upper) / 2.0
        rate = _expected_disturbance_rate(
            midpoint, eligible, sequences, recovery_decisions,
        )
        if rate < target_rate:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0
