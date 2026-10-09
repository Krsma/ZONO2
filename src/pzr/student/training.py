"""Small PyTorch interface to the existing scientific training contract."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from pzr.artifact_io import write_json_atomic
from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.objectives import cost_sensitive_pairwise_loss
from pzr.learning.provenance import pzr_source_sha256
from pzr.learning.ranker import (
    FeatureNormalizer, ReducerPolicy, ReducerScorer, ReducerTrainingResult,
    evaluate_reducer, train_reducer_policy,
)
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA

ModelFactory = Callable[[int, int], nn.Module]
if TYPE_CHECKING:
    from pzr.student.data import Task


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError("epochs, batch_size and patience must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


@contextmanager
def seeded_training(seed: int) -> Iterator[None]:
    """Restore caller RNG and deterministic settings, including on failure."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=[]):
        try:
            random.seed(seed)
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@dataclass(frozen=True)
class TrainingRun:
    policy: ReducerPolicy
    history: ReducerTrainingResult
    config: TrainingConfig
    provenance: dict[str, object]

    def save(self, directory: str | Path) -> Path:
        path = Path(directory)
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Choose an empty output directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.model.state_dict(), path / "weights.pt")
        write_json_atomic({
            "schema": "pzr.student-selector.v1",
            "custom_network": self.policy.model.custom_network,
            "feature_schema": asdict(self.policy.feature_schema),
            "candidate_names": list(self.policy.candidate_names),
            "normalizer_mean": self.policy.normalizer.mean.tolist(),
            "normalizer_std": self.policy.normalizer.std.tolist(),
            "objective_contract": self.policy.objective_contract,
            "training_config": asdict(self.config),
            "history": asdict(self.history),
            "provenance": self.provenance,
        }, path / "student.json")
        return path


def train_selector(task: Task, *, model_factory: ModelFactory | None = None,
                   config: TrainingConfig = TrainingConfig()) -> TrainingRun:
    """Train a CPU model: normalized [B,15] inputs, lower-is-better [B,4] scores."""
    with seeded_training(config.seed):
        policy, history = train_reducer_policy(
            task.dataset, RTL_RANKING_FEATURE_SCHEMA, objective="pairwise",
            model_factory=model_factory, **asdict(config),
        )
    return TrainingRun(policy, history, config, {
        "bundle_sha256": task.bundle_sha256, "budget": task.budget,
        "pzr_source_sha256": pzr_source_sha256(), "torch_version": torch.__version__,
    })


def load_selector(directory: str | Path, *, model_factory: ModelFactory | None = None) -> ReducerPolicy:
    """Load weights without pickled model code; custom networks need their factory."""
    path = Path(directory)
    payload = json.loads((path / "student.json").read_text())
    if payload.get("schema") != "pzr.student-selector.v1":
        raise ValueError("unsupported student selector schema")
    if payload["feature_schema"] != json.loads(json.dumps(asdict(RTL_RANKING_FEATURE_SCHEMA))):
        raise ValueError("selector feature schema differs")
    from pzr.rtlola.actions import MPC_ACTION_NAMES
    if tuple(payload["candidate_names"]) != MPC_ACTION_NAMES:
        raise ValueError("selector candidate order differs")
    custom = payload["custom_network"]
    if custom != (model_factory is not None):
        raise ValueError("Supply model_factory exactly when loading a custom network")
    with seeded_training(0):
        network = model_factory(15, 4) if custom else None
        model = ReducerScorer(RTL_RANKING_FEATURE_SCHEMA, MPC_ACTION_NAMES, network=network)
    model.load_state_dict(torch.load(path / "weights.pt", map_location="cpu", weights_only=True))
    return ReducerPolicy(model, FeatureNormalizer(
        np.asarray(payload["normalizer_mean"], dtype=np.float32),
        np.asarray(payload["normalizer_std"], dtype=np.float32),
    ), payload["objective_contract"])


def prepared_samples(dataset: ReducerCostDataset, normalizer: FeatureNormalizer) -> TensorDataset:
    """Return (normalized features, float64 costs, bool feasible) for custom loops.

    Fit the normalizer on training data only. Infeasible costs are NaN; pass
    their explicit mask to cost_sensitive_pairwise_loss.
    """
    values = normalizer.transform(RTL_RANKING_FEATURE_SCHEMA.transform(dataset.features))
    return TensorDataset(torch.tensor(values), torch.tensor(dataset.teacher_costs),
                         torch.tensor(dataset.feasible))


def evaluate_selector(policy: ReducerPolicy, dataset: ReducerCostDataset) -> dict[str, float | int]:
    scores = policy.predict_scores(dataset.features)
    if scores.shape != dataset.teacher_costs.shape or not np.isfinite(scores).all():
        raise ValueError("selector returned invalid scores")
    metrics = evaluate_reducer(policy, dataset)
    loss = cost_sensitive_pairwise_loss(torch.tensor(scores, dtype=torch.float64),
                                       torch.tensor(dataset.teacher_costs), torch.tensor(dataset.feasible))
    return {
        "ranking_loss": float(loss), "pairwise_accuracy": metrics.pairwise_accuracy,
        "teacher_agreement": metrics.top1_accuracy,
        "mean_normalized_regret": metrics.mean_chosen_normalized_regret,
        "max_normalized_regret": metrics.max_chosen_normalized_regret,
        "infeasible_selection_count": metrics.infeasible_selection_count,
        "all_infeasible_states": metrics.all_infeasible_states,
    }
