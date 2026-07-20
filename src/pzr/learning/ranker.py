"""PyTorch reducer scoring with ranking, distillation, and regret objectives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from pzr.artifact_io import write_json_atomic
from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.objectives import (
    ABSOLUTE_TOLERANCE,
    EXPECTED_REGRET_OBJECTIVE_CONTRACT,
    ObjectiveName,
    RELATIVE_TOLERANCE,
    cost_sensitive_pairwise_loss,
    expected_regret_loss,
    expected_regret_targets,
    normalized_regrets,
    objective_contract,
    rankable_state_mask,
    soft_distillation_loss,
    soft_teacher_distribution,
    tolerant_best_mask,
    validate_objective_contract,
)


MODEL_SCHEMA = "pzr.reducer-scorer.v3"


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    version: int
    feature_names: tuple[str, ...]
    log1p_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature names must be non-empty and unique")
        unknown = set(self.log1p_features) - set(self.feature_names)
        if unknown:
            raise ValueError(f"unknown log1p features: {sorted(unknown)}")

    def transform(self, features: NDArray[np.floating]) -> NDArray[np.float32]:
        values = np.asarray(features, dtype=np.float64).copy()
        if values.shape[-1] != len(self.feature_names):
            raise ValueError("feature array does not match feature schema")
        for name in self.log1p_features:
            column = self.feature_names.index(name)
            if np.any(values[..., column] < 0.0):
                raise ValueError(f"log1p feature {name!r} contains negative values")
            values[..., column] = np.log1p(values[..., column])
        if not np.all(np.isfinite(values)):
            raise ValueError("transformed features contain non-finite values")
        return values.astype(np.float32)


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: NDArray[np.float32]
    std: NDArray[np.float32]

    @classmethod
    def fit(cls, features: NDArray[np.float32]) -> "FeatureNormalizer":
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("normalization requires a non-empty feature matrix")
        return cls(
            mean=np.mean(values, axis=0, dtype=np.float64).astype(np.float32),
            std=np.std(values, axis=0, dtype=np.float64).astype(np.float32),
        )

    def transform(self, features: NDArray[np.floating]) -> NDArray[np.float32]:
        values = np.asarray(features, dtype=np.float32)
        if values.shape[-1] != self.mean.size or self.std.shape != self.mean.shape:
            raise ValueError("normalizer and feature dimensions differ")
        return ((values - self.mean) / np.maximum(self.std, 1e-8)).astype(np.float32)


class ReducerScorer(nn.Module):
    """Fixed-catalog MLP returning one lower-is-better score per candidate."""

    def __init__(
        self,
        feature_schema: FeatureSchema,
        candidate_names: tuple[str, ...],
        hidden_sizes: tuple[int, ...] = (32, 32),
    ) -> None:
        super().__init__()
        if not candidate_names or len(set(candidate_names)) != len(candidate_names):
            raise ValueError("candidate names must be non-empty and unique")
        if any(size < 1 for size in hidden_sizes):
            raise ValueError("hidden layer sizes must be positive")
        self.feature_schema = feature_schema
        self.candidate_names = candidate_names
        self.hidden_sizes = hidden_sizes
        sizes = [len(feature_schema.feature_names), *hidden_sizes, len(candidate_names)]
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != len(self.feature_schema.feature_names):
            raise ValueError("model input does not match feature schema")
        scores = self.network(features)
        if scores.shape[-1] != len(self.candidate_names):
            raise RuntimeError("model output does not match candidate catalog")
        return scores


@dataclass(frozen=True)
class ReducerMetrics:
    pairwise_accuracy: float
    top1_accuracy: float
    mean_chosen_normalized_regret: float
    max_chosen_normalized_regret: float
    feasible_selection_rate: float
    infeasible_selection_count: int
    valid_states: int
    all_infeasible_states: int
    rankable_states: int
    skipped_tie_states: int
    target_entropy: float
    predicted_entropy: float
    kl_divergence: float
    infeasible_probability: float
    regression_rmse: float
    regression_mae: float
    prediction_below_zero_count: int
    prediction_above_two_count: int
    prediction_outside_target_range_count: int
    prediction_outside_target_range_rate: float


@dataclass(frozen=True)
class ReducerTrainingResult:
    objective: str
    epochs: int
    best_epoch: int
    train_loss_history: tuple[float, ...]
    val_loss_history: tuple[float, ...]
    train_kl_history: tuple[float, ...]
    val_kl_history: tuple[float, ...]
    train_feasibility_history: tuple[float, ...]
    val_feasibility_history: tuple[float, ...]
    train_metrics: ReducerMetrics
    val_metrics: ReducerMetrics


class ReducerPolicy:
    """Validated direct-inference wrapper for a reducer scorer."""

    def __init__(
        self,
        model: ReducerScorer,
        normalizer: FeatureNormalizer,
        objective_contract: dict[str, object],
    ) -> None:
        if normalizer.mean.size != len(model.feature_schema.feature_names):
            raise ValueError("model and normalizer feature dimensions differ")
        validate_objective_contract(objective_contract)
        self.model = model.cpu().eval()
        self.normalizer = normalizer
        self.objective_contract = dict(objective_contract)

    @property
    def candidate_names(self) -> tuple[str, ...]:
        return self.model.candidate_names

    @property
    def feature_schema(self) -> FeatureSchema:
        return self.model.feature_schema

    def predict_scores(self, raw_features: NDArray[np.floating]) -> NDArray[np.float32]:
        transformed = self.feature_schema.transform(raw_features)
        normalized = self.normalizer.transform(transformed)
        tensor = torch.as_tensor(normalized, dtype=torch.float32)
        with torch.no_grad():
            return self.model(tensor).cpu().numpy().astype(np.float32)

    def predict_probabilities(
        self,
        raw_features: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        scores = np.asarray(self.predict_scores(raw_features), dtype=np.float64)
        shifted = -scores - np.max(-scores, axis=-1, keepdims=True)
        weights = np.exp(shifted)
        return weights / np.sum(weights, axis=-1, keepdims=True)

    def rank_candidates(self, raw_features: NDArray[np.floating]) -> list[str]:
        scores = self.predict_scores(raw_features)
        if scores.ndim != 1 or scores.size != len(self.candidate_names):
            raise ValueError("ranking requires one feature vector")
        order = np.argsort(scores, kind="stable")
        return [self.candidate_names[index] for index in order]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "weights.pt")
        payload = {
            "schema": MODEL_SCHEMA,
            "feature_schema": asdict(self.feature_schema),
            "candidate_names": list(self.candidate_names),
            "hidden_sizes": list(self.model.hidden_sizes),
            "normalizer_mean": self.normalizer.mean.tolist(),
            "normalizer_std": self.normalizer.std.tolist(),
            "torch_version": torch.__version__,
            "objective_contract": self.objective_contract,
        }
        write_json_atomic(payload, directory / "model.json")

    @classmethod
    def load(cls, directory: Path) -> "ReducerPolicy":
        payload = json.loads((directory / "model.json").read_text())
        if payload.get("schema") != MODEL_SCHEMA:
            raise ValueError("unsupported reducer scorer schema")
        objective_contract = dict(payload["objective_contract"])
        validate_objective_contract(objective_contract)
        schema_payload = payload["feature_schema"]
        schema = FeatureSchema(
            name=str(schema_payload["name"]),
            version=int(schema_payload["version"]),
            feature_names=tuple(schema_payload["feature_names"]),
            log1p_features=tuple(schema_payload["log1p_features"]),
        )
        model = ReducerScorer(
            schema,
            tuple(payload["candidate_names"]),
            tuple(int(value) for value in payload["hidden_sizes"]),
        )
        state = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return cls(
            model,
            FeatureNormalizer(
                mean=np.asarray(payload["normalizer_mean"], dtype=np.float32),
                std=np.asarray(payload["normalizer_std"], dtype=np.float32),
            ),
            objective_contract,
        )


def train_reducer_policy(
    dataset: ReducerCostDataset,
    feature_schema: FeatureSchema,
    *,
    objective: ObjectiveName,
    temperature: float | None = None,
    feasibility_penalty: float = 1.0,
    hidden_sizes: tuple[int, ...] = (32, 32),
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    patience: int = 10,
    seed: int = 42,
) -> tuple[ReducerPolicy, ReducerTrainingResult]:
    if dataset.feature_names != feature_schema.feature_names:
        raise ValueError("dataset does not match feature schema")
    contract = objective_contract(
        objective,
        temperature=temperature,
        feasibility_penalty=feasibility_penalty,
    )
    train_indices = dataset.indices_for_split("train")
    val_indices = dataset.indices_for_split("validation")
    if train_indices.size == 0 or val_indices.size == 0:
        raise ValueError("training and validation splits must both be non-empty")
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch size, and patience must be positive")
    if objective in ("soft-kl", "expected-regret") and (
        not np.any(np.any(dataset.feasible[train_indices], axis=1))
        or not np.any(np.any(dataset.feasible[val_indices], axis=1))
    ):
        raise ValueError(f"{objective} training and validation each need a feasible state")
    if objective == "pairwise" and (
        not np.any(rankable_state_mask(dataset.teacher_costs[train_indices], dataset.feasible[train_indices]))
        or not np.any(rankable_state_mask(dataset.teacher_costs[val_indices], dataset.feasible[val_indices]))
    ):
        raise ValueError("pairwise training and validation each need a rankable state")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    transformed = feature_schema.transform(dataset.features)
    normalizer = FeatureNormalizer.fit(transformed[train_indices])
    features = torch.as_tensor(normalizer.transform(transformed), dtype=torch.float32)
    costs = torch.tensor(dataset.teacher_costs, dtype=torch.float64)
    feasible = torch.tensor(dataset.feasible, dtype=torch.bool)
    teacher_probabilities = None
    regression_targets = None
    if objective == "soft-kl":
        teacher_probabilities = torch.tensor(
            soft_teacher_distribution(dataset.teacher_costs, dataset.feasible, float(temperature)),
            dtype=torch.float64,
        )
    elif objective == "expected-regret":
        regression_targets = torch.tensor(
            expected_regret_targets(dataset.teacher_costs, dataset.feasible),
            dtype=torch.float64,
        )
    model = ReducerScorer(feature_schema, dataset.candidate_names, hidden_sizes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(train_indices, dtype=torch.int64)),
        batch_size=min(batch_size, train_indices.size),
        shuffle=True,
        generator=generator,
    )
    best_state: dict[str, Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    remaining_patience = patience
    train_history: list[float] = []
    val_history: list[float] = []
    train_kl_history: list[float] = []
    val_kl_history: list[float] = []
    train_feasibility_history: list[float] = []
    val_feasibility_history: list[float] = []

    def loss_for(indices: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        scores = model(features[indices]).to(torch.float64)
        if objective == "pairwise":
            total = cost_sensitive_pairwise_loss(scores, costs[indices], feasible[indices])
            zero = total.detach() * 0.0
            return total, zero, zero
        if objective == "expected-regret":
            assert regression_targets is not None
            total = expected_regret_loss(
                scores, regression_targets[indices], feasible[indices],
            )
            zero = total.detach() * 0.0
            return total, zero, zero
        assert teacher_probabilities is not None
        return soft_distillation_loss(
            scores,
            teacher_probabilities[indices],
            feasible[indices],
            feasibility_penalty=feasibility_penalty,
        )

    for epoch in range(epochs):
        model.train()
        batch_values: list[tuple[float, float, float]] = []
        for (indices,) in loader:
            optimizer.zero_grad(set_to_none=True)
            total, kl, infeasible_mass = loss_for(indices)
            total.backward()
            optimizer.step()
            batch_values.append((float(total.detach()), float(kl.detach()), float(infeasible_mass.detach())))
        train_history.append(float(np.mean([value[0] for value in batch_values])))
        train_kl_history.append(float(np.mean([value[1] for value in batch_values])))
        train_feasibility_history.append(float(np.mean([value[2] for value in batch_values])))
        model.eval()
        with torch.no_grad():
            val_total, val_kl, val_feasibility = loss_for(torch.as_tensor(val_indices))
        val_loss = float(val_total)
        val_history.append(val_loss)
        val_kl_history.append(float(val_kl))
        val_feasibility_history.append(float(val_feasibility))
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    policy = ReducerPolicy(model, normalizer, contract)
    return policy, ReducerTrainingResult(
        objective=objective,
        epochs=len(train_history),
        best_epoch=best_epoch,
        train_loss_history=tuple(train_history),
        val_loss_history=tuple(val_history),
        train_kl_history=tuple(train_kl_history),
        val_kl_history=tuple(val_kl_history),
        train_feasibility_history=tuple(train_feasibility_history),
        val_feasibility_history=tuple(val_feasibility_history),
        train_metrics=evaluate_reducer(policy, dataset.subset(train_indices)),
        val_metrics=evaluate_reducer(policy, dataset.subset(val_indices)),
    )


def evaluate_reducer(policy: ReducerPolicy, dataset: ReducerCostDataset) -> ReducerMetrics:
    if dataset.candidate_names != policy.candidate_names:
        raise ValueError("dataset and policy candidate catalogs differ")
    if dataset.feature_names != policy.feature_schema.feature_names:
        raise ValueError("dataset and policy feature schemas differ")
    scores = np.asarray(policy.predict_scores(dataset.features), dtype=np.float64)
    probabilities = np.asarray(policy.predict_probabilities(dataset.features), dtype=np.float64)
    chosen = np.argmin(scores, axis=1)
    rows = np.arange(dataset.num_samples)
    valid = np.any(dataset.feasible, axis=1)
    selected_feasible = dataset.feasible[rows, chosen]
    regrets = normalized_regrets(dataset.teacher_costs, dataset.feasible)
    selected_regret = np.where(selected_feasible, regrets[rows, chosen], 1.0)
    top1_mask = tolerant_best_mask(dataset.teacher_costs, dataset.feasible)
    top1 = top1_mask[rows, chosen]
    correct = total = 0
    for row in range(dataset.num_samples):
        for left in range(dataset.num_candidates):
            for right in range(left + 1, dataset.num_candidates):
                if dataset.feasible[row, left] and dataset.feasible[row, right]:
                    gap = dataset.teacher_costs[row, right] - dataset.teacher_costs[row, left]
                    tolerance = max(
                        ABSOLUTE_TOLERANCE,
                        RELATIVE_TOLERANCE * max(
                            abs(dataset.teacher_costs[row, left]),
                            abs(dataset.teacher_costs[row, right]),
                        ),
                    )
                    if abs(gap) <= tolerance:
                        continue
                    predicted = scores[row, right] - scores[row, left]
                    correct += int(np.sign(gap) == np.sign(predicted))
                    total += 1
                elif dataset.feasible[row, left] != dataset.feasible[row, right]:
                    better = left if dataset.feasible[row, left] else right
                    worse = right if dataset.feasible[row, left] else left
                    correct += int(scores[row, better] < scores[row, worse])
                    total += 1
    rankable = rankable_state_mask(dataset.teacher_costs, dataset.feasible)
    target_entropy = predicted_entropy = kl_divergence = float("nan")
    objective_schema = str(policy.objective_contract["schema"])
    if objective_schema == "pzr.reducer-objective.soft-kl-v1" and np.any(valid):
        teacher = soft_teacher_distribution(
            dataset.teacher_costs,
            dataset.feasible,
            float(policy.objective_contract["temperature"]),
        )
        positive = teacher > 0.0
        target_entropy = float(np.mean(-np.sum(np.where(positive, teacher * np.log(np.maximum(teacher, 1e-300)), 0.0), axis=1)[valid]))
        kl_divergence = float(np.mean(np.sum(np.where(positive, teacher * (np.log(np.maximum(teacher, 1e-300)) - np.log(np.maximum(probabilities, 1e-300))), 0.0), axis=1)[valid]))
    if np.any(valid):
        predicted_entropy = float(np.mean(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1)[valid]))
    infeasible_probability = np.sum(np.where(~dataset.feasible, probabilities, 0.0), axis=1)
    regression_rmse = regression_mae = float("nan")
    below_zero_count = above_two_count = outside_count = 0
    outside_rate = float("nan")
    if objective_schema == EXPECTED_REGRET_OBJECTIVE_CONTRACT["schema"] and np.any(valid):
        targets = expected_regret_targets(dataset.teacher_costs, dataset.feasible)
        errors = scores[valid] - targets[valid]
        regression_rmse = float(np.sqrt(np.mean(np.square(errors))))
        regression_mae = float(np.mean(np.abs(errors)))
        valid_scores = scores[valid]
        below_zero_count = int(np.count_nonzero(valid_scores < 0.0))
        above_two_count = int(np.count_nonzero(valid_scores > 2.0))
        outside_count = below_zero_count + above_two_count
        outside_rate = float(outside_count / valid_scores.size)
    return ReducerMetrics(
        pairwise_accuracy=float(correct / total) if total else 1.0,
        top1_accuracy=float(np.mean(top1[valid])) if np.any(valid) else float("nan"),
        mean_chosen_normalized_regret=float(np.mean(selected_regret[valid])) if np.any(valid) else float("nan"),
        max_chosen_normalized_regret=float(np.max(selected_regret[valid])) if np.any(valid) else float("nan"),
        feasible_selection_rate=float(np.mean(selected_feasible[valid])) if np.any(valid) else float("nan"),
        infeasible_selection_count=int(np.count_nonzero(valid & ~selected_feasible)),
        valid_states=int(np.count_nonzero(valid)),
        all_infeasible_states=int(np.count_nonzero(~valid)),
        rankable_states=int(np.count_nonzero(rankable)),
        skipped_tie_states=int(np.count_nonzero(valid & ~rankable)),
        target_entropy=target_entropy,
        predicted_entropy=predicted_entropy,
        kl_divergence=kl_divergence,
        infeasible_probability=float(np.mean(infeasible_probability[valid])) if np.any(valid) else float("nan"),
        regression_rmse=regression_rmse,
        regression_mae=regression_mae,
        prediction_below_zero_count=below_zero_count,
        prediction_above_two_count=above_two_count,
        prediction_outside_target_range_count=outside_count,
        prediction_outside_target_range_rate=outside_rate,
    )
