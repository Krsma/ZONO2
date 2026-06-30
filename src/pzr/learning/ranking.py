"""Scenario-neutral regret-ranking model and training data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

@dataclass
class RegretDataset:
    """Feature and per-candidate regret targets for ranking distillation."""

    features: NDArray[np.float64]
    regrets: NDArray[np.float64]
    candidate_names: tuple[str, ...]
    feature_names: tuple[str, ...] = ()

    @property
    def num_samples(self) -> int:
        return self.features.shape[0]

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    @property
    def num_candidates(self) -> int:
        return len(self.candidate_names)

    def train_val_split(
        self, val_fraction: float = 0.2, seed: int = 42,
    ) -> tuple["RegretDataset", "RegretDataset"]:
        rng = np.random.default_rng(seed)
        n = self.num_samples
        indices = rng.permutation(n)
        split = int(n * (1.0 - val_fraction))
        train_idx = indices[:split]
        val_idx = indices[split:]
        return (
            RegretDataset(
                self.features[train_idx], self.regrets[train_idx],
                self.candidate_names, self.feature_names,
            ),
            RegretDataset(
                self.features[val_idx], self.regrets[val_idx],
                self.candidate_names, self.feature_names,
            ),
        )


@dataclass
class RegretTrainingResult:
    """Training diagnostics for the regret ranker."""

    train_loss: float
    val_loss: float
    train_top1_accuracy: float
    val_top1_accuracy: float
    train_mean_chosen_regret: float
    val_mean_chosen_regret: float
    epochs: int
    train_loss_history: list[float]


class RegretRankingPolicy:
    """MLP that ranks reducers by predicted normalized regret."""

    def __init__(
        self,
        candidate_names: tuple[str, ...],
        feature_mean: NDArray[np.float64],
        feature_std: NDArray[np.float64],
        weights: list[NDArray[np.float64]],
        biases: list[NDArray[np.float64]],
        feature_names: tuple[str, ...] = (),
    ) -> None:
        self.candidate_names = candidate_names
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.weights = weights
        self.biases = biases
        self.feature_names = feature_names

    def predict_regret(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward pass returning one predicted regret per reducer."""
        x = (features - self.feature_mean) / np.maximum(self.feature_std, 1e-8)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(0.0, x @ w + b)
        return x @ self.weights[-1] + self.biases[-1]

    def rank_reducers(self, features: NDArray[np.float64]) -> list[str]:
        """Rank reducer names by predicted regret, lowest first."""
        regrets = self.predict_regret(features)
        order = np.argsort(regrets, kind="stable")
        return [self.candidate_names[i] for i in order]

    def save(self, path: Path) -> None:
        """Save policy weights to a NumPy archive."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            candidate_names=np.array(self.candidate_names),
            feature_names=np.array(self.feature_names),
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            num_layers=np.array(len(self.weights)),
            **{f"w{i}": w for i, w in enumerate(self.weights)},
            **{f"b{i}": b for i, b in enumerate(self.biases)},
        )

    @classmethod
    def load(cls, path: Path) -> "RegretRankingPolicy":
        """Load a saved regret-ranking policy."""
        data = np.load(path, allow_pickle=True)
        num_layers = int(data["num_layers"])
        return cls(
            candidate_names=tuple(str(v) for v in data["candidate_names"]),
            feature_mean=data["feature_mean"],
            feature_std=data["feature_std"],
            weights=[data[f"w{i}"] for i in range(num_layers)],
            biases=[data[f"b{i}"] for i in range(num_layers)],
            feature_names=(
                tuple(str(v) for v in data["feature_names"])
                if "feature_names" in data else ()
            ),
        )


def train_regret_policy(
    dataset: RegretDataset,
    hidden_sizes: tuple[int, ...] = (64, 64),
    epochs: int = 200,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 42,
    show_progress: bool = False,
    loss: Literal["mse", "pairwise"] = "pairwise",
    pairwise_margin_tol: float = 1e-3,
    pairwise_mse_weight: float = 0.05,
) -> tuple[RegretRankingPolicy, RegretTrainingResult]:
    """Train a numpy MLP to rank reducers by normalized regret."""
    if loss not in {"mse", "pairwise"}:
        raise ValueError("loss must be 'mse' or 'pairwise'")
    rng = np.random.default_rng(seed)
    train_ds, val_ds = dataset.train_val_split(val_fraction, seed)

    feature_mean = np.mean(train_ds.features, axis=0)
    feature_std = np.std(train_ds.features, axis=0)
    safe_std = np.maximum(feature_std, 1e-8)

    X_train = (train_ds.features - feature_mean) / safe_std
    y_train = train_ds.regrets
    X_val = (val_ds.features - feature_mean) / safe_std
    y_val = val_ds.regrets

    layer_sizes = [dataset.num_features, *hidden_sizes, dataset.num_candidates]
    weights: list[NDArray[np.float64]] = []
    biases: list[NDArray[np.float64]] = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        weights.append(rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in))
        biases.append(np.zeros(fan_out, dtype=np.float64))

    loss_history: list[float] = []
    iterator = tqdm(
        range(epochs), desc="regret epochs", disable=not show_progress,
        unit="epoch", leave=False,
    )
    for _ in iterator:
        activations = [X_train]
        for w, b in zip(weights[:-1], biases[:-1]):
            activations.append(np.maximum(0.0, activations[-1] @ w + b))
        pred = activations[-1] @ weights[-1] + biases[-1]
        if loss == "mse":
            err = pred - y_train
            train_loss = float(np.mean(err * err))
            n = max(X_train.shape[0], 1)
            delta = 2.0 * err / n
        else:
            train_loss, delta = _pairwise_loss_and_delta(
                pred, y_train,
                margin_tol=pairwise_margin_tol,
                mse_weight=pairwise_mse_weight,
            )
        loss_history.append(train_loss)
        grad_weights: list[NDArray[np.float64]] = []
        grad_biases: list[NDArray[np.float64]] = []
        for i in range(len(weights) - 1, -1, -1):
            grad_weights.insert(0, activations[i].T @ delta)
            grad_biases.insert(0, np.sum(delta, axis=0))
            if i > 0:
                delta = (delta @ weights[i].T) * (activations[i] > 0.0)

        for i in range(len(weights)):
            weights[i] -= learning_rate * grad_weights[i]
            biases[i] -= learning_rate * grad_biases[i]

    policy = RegretRankingPolicy(
        candidate_names=dataset.candidate_names,
        feature_mean=feature_mean,
        feature_std=feature_std,
        weights=[w.copy() for w in weights],
        biases=[b.copy() for b in biases],
        feature_names=dataset.feature_names,
    )

    train_metrics = _ranking_metrics(policy, train_ds)
    val_metrics = _ranking_metrics(policy, val_ds)
    return policy, RegretTrainingResult(
        train_loss=train_metrics["loss"],
        val_loss=val_metrics["loss"],
        train_top1_accuracy=train_metrics["top1_accuracy"],
        val_top1_accuracy=val_metrics["top1_accuracy"],
        train_mean_chosen_regret=train_metrics["mean_chosen_regret"],
        val_mean_chosen_regret=val_metrics["mean_chosen_regret"],
        epochs=epochs,
        train_loss_history=loss_history,
    )


def _ranking_metrics(
    policy: RegretRankingPolicy,
    dataset: RegretDataset,
) -> dict[str, float]:
    if dataset.num_samples == 0:
        return {"loss": 0.0, "top1_accuracy": 0.0, "mean_chosen_regret": 0.0}
    preds = np.stack([policy.predict_regret(x) for x in dataset.features])
    loss = float(np.mean((preds - dataset.regrets) ** 2))
    pred_best = np.argmin(preds, axis=1)
    true_best = np.argmin(dataset.regrets, axis=1)
    chosen = dataset.regrets[np.arange(dataset.num_samples), pred_best]
    return {
        "loss": loss,
        "top1_accuracy": float(np.mean(pred_best == true_best)),
        "mean_chosen_regret": float(np.mean(chosen)),
    }


def _pairwise_loss_and_delta(
    pred: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    margin_tol: float,
    mse_weight: float,
) -> tuple[float, NDArray[np.float64]]:
    """Return pairwise ranking loss and d(loss)/d(pred)."""
    delta = np.zeros_like(pred)
    total = 0.0
    pair_count = 0
    for row in range(pred.shape[0]):
        for better in range(pred.shape[1]):
            for worse in range(pred.shape[1]):
                gap = float(target[row, worse] - target[row, better])
                if gap <= margin_tol:
                    continue
                weight = min(gap, 10.0)
                margin = float(pred[row, better] - pred[row, worse])
                total += weight * float(np.logaddexp(0.0, margin))
                sig = 1.0 / (1.0 + np.exp(-np.clip(margin, -60.0, 60.0)))
                delta[row, better] += weight * sig
                delta[row, worse] -= weight * sig
                pair_count += 1
    if pair_count > 0:
        total /= pair_count
        delta /= pair_count
    mse = float(np.mean((pred - target) ** 2))
    n = max(pred.shape[0], 1)
    total += mse_weight * mse
    delta += mse_weight * 2.0 * (pred - target) / n
    return total, delta
