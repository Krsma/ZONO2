"""PyTorch cost-sensitive reducer ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from pzr.learning.dataset import RankingDataset


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


class ReducerRanker(nn.Module):
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
class RankingMetrics:
    pairwise_accuracy: float
    top1_accuracy: float
    mean_chosen_regret: float
    max_chosen_regret: float
    feasible_selection_rate: float


@dataclass(frozen=True)
class RankingTrainingResult:
    epochs: int
    best_epoch: int
    train_loss_history: tuple[float, ...]
    val_loss_history: tuple[float, ...]
    train_metrics: RankingMetrics
    val_metrics: RankingMetrics


class RankingPolicy:
    """Validated inference wrapper for a trained reducer ranker."""

    def __init__(self, model: ReducerRanker, normalizer: FeatureNormalizer) -> None:
        if normalizer.mean.size != len(model.feature_schema.feature_names):
            raise ValueError("model and normalizer feature dimensions differ")
        self.model = model.cpu().eval()
        self.normalizer = normalizer

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
            result = self.model(tensor).cpu().numpy().astype(np.float32)
        return result

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
            "schema": "pzr.reducer-ranker.v1",
            "feature_schema": asdict(self.feature_schema),
            "candidate_names": list(self.candidate_names),
            "hidden_sizes": list(self.model.hidden_sizes),
            "normalizer_mean": self.normalizer.mean.tolist(),
            "normalizer_std": self.normalizer.std.tolist(),
            "torch_version": torch.__version__,
        }
        (directory / "model.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, directory: Path) -> "RankingPolicy":
        payload = json.loads((directory / "model.json").read_text())
        if payload.get("schema") != "pzr.reducer-ranker.v1":
            raise ValueError("unsupported reducer ranker schema")
        schema_payload = payload["feature_schema"]
        schema = FeatureSchema(
            name=str(schema_payload["name"]),
            version=int(schema_payload["version"]),
            feature_names=tuple(schema_payload["feature_names"]),
            log1p_features=tuple(schema_payload["log1p_features"]),
        )
        model = ReducerRanker(
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
        )


def cost_sensitive_pairwise_loss(
    scores: Tensor,
    teacher_costs: Tensor,
    feasible: Tensor,
    *,
    gap_cap: float = 10.0,
) -> Tensor:
    """Return weighted pairwise loss with explicit infeasibility ordering."""
    if scores.shape != teacher_costs.shape or feasible.shape != scores.shape:
        raise ValueError("score, cost, and feasibility tensors must align")
    safe_costs = torch.where(feasible, teacher_costs, torch.zeros_like(teacher_costs))
    positive_inf = torch.full_like(safe_costs, float("inf"))
    best = torch.where(feasible, safe_costs, positive_inf).amin(dim=1)
    scale = torch.maximum(best.abs(), torch.ones_like(best))
    tolerance = torch.maximum(
        torch.full_like(best, 1e-9),
        best.abs() * 1e-9,
    )
    cost_i = safe_costs.unsqueeze(2)
    cost_j = safe_costs.unsqueeze(1)
    feasible_i = feasible.unsqueeze(2)
    feasible_j = feasible.unsqueeze(1)
    gap = cost_j - cost_i
    ranked = feasible_i & feasible_j & (gap > tolerance[:, None, None])
    feasible_over_infeasible = feasible_i & ~feasible_j
    weights = torch.where(
        ranked,
        torch.clamp(gap / scale[:, None, None], max=gap_cap),
        torch.zeros_like(gap),
    )
    weights = torch.where(
        feasible_over_infeasible,
        torch.ones_like(weights),
        weights,
    )
    score_margin = scores.unsqueeze(2) - scores.unsqueeze(1)
    total_weight = weights.sum()
    if not bool(total_weight > 0):
        return scores.sum() * 0.0
    return (weights * F.softplus(score_margin)).sum() / total_weight


def train_ranking_policy(
    dataset: RankingDataset,
    feature_schema: FeatureSchema,
    *,
    hidden_sizes: tuple[int, ...] = (32, 32),
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    patience: int = 10,
    seed: int = 42,
) -> tuple[RankingPolicy, RankingTrainingResult]:
    if dataset.feature_names != feature_schema.feature_names:
        raise ValueError("dataset does not match feature schema")
    train_indices = dataset.indices_for_split("train")
    val_indices = dataset.indices_for_split("validation")
    if train_indices.size == 0 or val_indices.size == 0:
        raise ValueError("training and validation splits must both be non-empty")
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch size, and patience must be positive")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    transformed = feature_schema.transform(dataset.features)
    normalizer = FeatureNormalizer.fit(transformed[train_indices])
    normalized = normalizer.transform(transformed)
    features = torch.as_tensor(normalized, dtype=torch.float32)
    costs = torch.tensor(dataset.teacher_costs, dtype=torch.float64)
    feasible = torch.tensor(dataset.feasible, dtype=torch.bool)
    model = ReducerRanker(feature_schema, dataset.candidate_names, hidden_sizes)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
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
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for (indices,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = cost_sensitive_pairwise_loss(
                model(features[indices]).to(torch.float64),
                costs[indices],
                feasible[indices],
            )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        train_history.append(float(np.mean(batch_losses)))
        model.eval()
        with torch.no_grad():
            val_loss = float(cost_sensitive_pairwise_loss(
                model(features[val_indices]).to(torch.float64),
                costs[val_indices],
                feasible[val_indices],
            ))
        val_history.append(val_loss)
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    policy = RankingPolicy(model, normalizer)
    return policy, RankingTrainingResult(
        epochs=len(train_history),
        best_epoch=best_epoch,
        train_loss_history=tuple(train_history),
        val_loss_history=tuple(val_history),
        train_metrics=evaluate_ranking(policy, dataset.subset(train_indices)),
        val_metrics=evaluate_ranking(policy, dataset.subset(val_indices)),
    )


def evaluate_ranking(policy: RankingPolicy, dataset: RankingDataset) -> RankingMetrics:
    if dataset.candidate_names != policy.candidate_names:
        raise ValueError("dataset and policy candidate catalogs differ")
    if dataset.feature_names != policy.feature_schema.feature_names:
        raise ValueError("dataset and policy feature schemas differ")
    scores = np.asarray(policy.predict_scores(dataset.features), dtype=np.float64)
    chosen = np.argmin(scores, axis=1)
    rows = np.arange(dataset.num_samples)
    selected_feasible = dataset.feasible[rows, chosen]
    best = np.nanmin(dataset.teacher_costs, axis=1)
    chosen_cost = dataset.teacher_costs[rows, chosen]
    scale = np.maximum(np.abs(best), 1.0)
    regret = np.where(selected_feasible, np.maximum((chosen_cost - best) / scale, 0.0), 10.0)
    tolerance = np.maximum(1e-9, np.abs(best) * 1e-9)
    top1 = selected_feasible & (chosen_cost <= best + tolerance)
    correct = total = 0
    for row in range(dataset.num_samples):
        for left in range(dataset.num_candidates):
            for right in range(left + 1, dataset.num_candidates):
                if not dataset.feasible[row, left] or not dataset.feasible[row, right]:
                    continue
                gap = dataset.teacher_costs[row, right] - dataset.teacher_costs[row, left]
                if abs(gap) <= tolerance[row]:
                    continue
                predicted = scores[row, right] - scores[row, left]
                correct += int(np.sign(gap) == np.sign(predicted))
                total += 1
    return RankingMetrics(
        pairwise_accuracy=float(correct / total) if total else 1.0,
        top1_accuracy=float(np.mean(top1)),
        mean_chosen_regret=float(np.mean(regret)),
        max_chosen_regret=float(np.max(regret)),
        feasible_selection_rate=float(np.mean(selected_feasible)),
    )
