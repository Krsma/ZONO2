"""Validated multi-dataset training and model-artifact ownership."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from pzr.artifact_io import write_csv_atomic, write_json_atomic
from pzr.learning.artifacts import load_reducer_cost_dataset
from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.diagnostics import (
    candidate_diagnostics,
    dataset_diagnostics,
    validation_metrics,
)
from pzr.learning.provenance import model_sha256, pzr_source_sha256, sha256_files
from pzr.learning.objectives import ObjectiveName
from pzr.learning.ranker import (
    ReducerPolicy,
    ReducerTrainingResult,
    train_reducer_policy,
)
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA


@dataclass(frozen=True)
class NamedDataset:
    name: str
    path: Path


@dataclass(frozen=True)
class ReducerTrainingConfig:
    datasets: tuple[NamedDataset, ...]
    output: Path
    objective: ObjectiveName = "pairwise"
    temperature_grid: tuple[float, ...] | None = None
    temperature_from: Path | None = None
    feasibility_penalty: float = 1.0
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("at least one named training dataset is required")
        if len({item.name for item in self.datasets}) != len(self.datasets):
            raise ValueError("named training datasets must have unique names")


def run_reducer_training(config: ReducerTrainingConfig) -> Path:
    """Train, select a checkpoint, and write a complete model artifact."""
    loaded = [load_reducer_cost_dataset(item.path) for item in config.datasets]
    validate_named_datasets(config.datasets, loaded)
    dataset = ReducerCostDataset.concatenate([item[0] for item in loaded])
    metadata_frames = []
    for named_input, (_, metadata, _) in zip(config.datasets, loaded):
        frame = metadata.copy()
        frame.insert(0, "dataset_label", named_input.name)
        metadata_frames.append(frame)
    sample_metadata = pd.concat(metadata_frames, ignore_index=True)
    temperatures, temperature_source_hash = _training_temperatures(config, dataset)
    candidates: list[tuple[float | None, ReducerPolicy, ReducerTrainingResult]] = []
    for temperature in temperatures:
        policy, result = train_reducer_policy(
            dataset,
            RTL_RANKING_FEATURE_SCHEMA,
            objective=config.objective,
            temperature=temperature,
            feasibility_penalty=config.feasibility_penalty,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            patience=config.patience,
            seed=config.seed,
        )
        candidates.append((temperature, policy, result))
    selected_index = min(
        range(len(candidates)), key=lambda index: _temperature_selection_key(candidates[index]),
    )
    selected_temperature, policy, result = candidates[selected_index]
    config.output.mkdir(parents=True, exist_ok=True)
    policy.save(config.output)
    temperature_frame = pd.DataFrame([
        {
            "temperature": temperature,
            "selected": index == selected_index,
            "infeasible_selection_count": candidate_result.val_metrics.infeasible_selection_count,
            "mean_selected_normalized_regret": candidate_result.val_metrics.mean_chosen_normalized_regret,
            "max_selected_normalized_regret": candidate_result.val_metrics.max_chosen_normalized_regret,
            "validation_kl": candidate_result.val_metrics.kl_divergence,
            "validation_loss": candidate_result.val_loss_history[candidate_result.best_epoch],
        }
        for index, (temperature, _, candidate_result) in enumerate(candidates)
    ])
    write_csv_atomic(temperature_frame, config.output / "temperature_selection.csv")
    # Plotting is deliberately lazy so importing the CLI does not initialize Matplotlib.
    from pzr.learning.reporting import write_training_plots

    write_training_plots(temperature_frame, result, config.output)
    payload = {
        "schema": "pzr.reducer-training.v4",
        "datasets": [
            {
                "name": item.name,
                "path": str(item.path),
                "sha256": dataset_sha256(item.path),
            }
            for item in config.datasets
        ],
        "candidate_names": list(policy.candidate_names),
        "feature_schema": asdict(policy.feature_schema),
        "objective_contract": policy.objective_contract,
        "selected_temperature": selected_temperature,
        "temperature_candidates": [value for value, _, _ in candidates],
        "temperature_source_model_sha256": temperature_source_hash,
        "checkpoint_selection": "minimum_clean_validation_objective_loss",
        "temperature_selection": (
            "infeasible_count_mean_regret_max_regret_kl_lower_temperature"
            if config.objective == "soft-kl" else None
        ),
        "validation_dataset": next(
            item.name
            for item, (_, metadata, _) in zip(config.datasets, loaded)
            if bool((metadata["split"].astype(str) == "validation").any())
        ),
        "validation_seeds": sorted({
            int(seed)
            for _, metadata, _ in loaded
            for seed in metadata.loc[
                metadata["split"].astype(str) == "validation", "seed"
            ]
        }),
        "training": asdict(result),
        "seed": config.seed,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "pzr_source_sha256": pzr_source_sha256(),
    }
    write_json_atomic(payload, config.output / "training.json")
    write_csv_atomic(
        validation_metrics(policy, dataset, sample_metadata),
        config.output / "validation_metrics.csv",
    )
    write_csv_atomic(
        dataset_diagnostics(dataset, sample_metadata),
        config.output / "dataset_diagnostics.csv",
    )
    write_csv_atomic(
        candidate_diagnostics(policy, dataset, sample_metadata),
        config.output / "candidate_diagnostics.csv",
    )
    return config.output


def dataset_sha256(path: Path) -> str:
    return sha256_files((
        path / "manifest.json",
        path / "samples.npz",
        path / "samples.csv",
        path / "candidate_costs.csv",
    ))


def validate_named_datasets(
    inputs: Sequence[NamedDataset],
    loaded: Sequence[tuple[ReducerCostDataset, pd.DataFrame, dict[str, object]]],
) -> None:
    if not loaded:
        raise ValueError("at least one named training dataset is required")
    if len(inputs) != len(loaded):
        raise ValueError("named dataset paths and loaded datasets must align")
    first_dataset, _, first_manifest = loaded[0]
    validation_sources = []
    validation_seeds: set[int] = set()
    nonvalidation_seeds: set[int] = set()
    for named_input, (dataset, metadata, manifest) in zip(inputs, loaded):
        if dataset.candidate_names != first_dataset.candidate_names:
            raise ValueError(f"dataset {named_input.name!r} candidate catalog differs")
        if dataset.feature_names != first_dataset.feature_names:
            raise ValueError(f"dataset {named_input.name!r} feature schema differs")
        if manifest.get("cost_contract") != first_manifest.get("cost_contract"):
            raise ValueError(f"dataset {named_input.name!r} cost schema differs")
        if dataset.indices_for_split("train").size == 0:
            raise ValueError(f"dataset {named_input.name!r} has no training samples")
        if not {"split", "seed"} <= set(metadata.columns):
            raise ValueError(f"dataset {named_input.name!r} lacks split/seed provenance")
        if tuple(metadata["split"].astype(str)) != dataset.splits:
            raise ValueError(f"dataset {named_input.name!r} split metadata differs")
        validation = metadata["split"].astype(str) == "validation"
        if bool(validation.any()):
            validation_sources.append(named_input.name)
            validation_seeds.update(metadata.loc[validation, "seed"].astype(int))
        nonvalidation_seeds.update(metadata.loc[~validation, "seed"].astype(int))
    if not validation_sources:
        raise ValueError("aggregate training dataset has no validation samples")
    if len(validation_sources) != 1:
        raise ValueError("validation samples must come exclusively from one primary dataset")
    overlap = validation_seeds & nonvalidation_seeds
    if overlap:
        raise ValueError(f"validation seeds overlap other splits: {sorted(overlap)}")


def _training_temperatures(
    config: ReducerTrainingConfig,
    dataset: ReducerCostDataset,
) -> tuple[tuple[float | None, ...], str | None]:
    if config.objective in ("pairwise", "expected-regret"):
        if config.temperature_grid is not None or config.temperature_from is not None:
            raise ValueError(f"{config.objective} training does not accept temperature options")
        return (None,), None
    if (config.temperature_grid is None) == (config.temperature_from is None):
        raise ValueError("soft-KL training needs exactly one temperature source")
    if config.temperature_grid is not None:
        if len(set(config.temperature_grid)) != len(config.temperature_grid):
            raise ValueError("temperature grid values must be unique")
        return tuple(config.temperature_grid), None
    assert config.temperature_from is not None
    source = ReducerPolicy.load(config.temperature_from)
    if source.objective_contract.get("schema") != "pzr.reducer-objective.soft-kl-v1":
        raise ValueError("temperature source must reference a soft-KL model")
    if source.candidate_names != dataset.candidate_names:
        raise ValueError("temperature source candidate catalog differs")
    if source.feature_schema.feature_names != dataset.feature_names:
        raise ValueError("temperature source feature schema differs")
    return (
        (float(source.objective_contract["temperature"]),),
        model_sha256(config.temperature_from),
    )


def _temperature_selection_key(
    candidate: tuple[float | None, ReducerPolicy, ReducerTrainingResult],
) -> tuple[float, ...]:
    temperature, _, result = candidate
    metrics = result.val_metrics
    return (
        float(metrics.infeasible_selection_count),
        metrics.mean_chosen_normalized_regret,
        metrics.max_chosen_normalized_regret,
        metrics.kl_divergence if pd.notna(metrics.kl_divergence) else 0.0,
        float(temperature or 0.0),
    )
