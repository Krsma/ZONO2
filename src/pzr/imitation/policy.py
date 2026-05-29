"""Neural policy for learned reduction selection.

A small MLP classifier that maps zonotope state features to a softmax
distribution over reducer names. Inference is a single forward pass —
orders of magnitude faster than MPC tree search.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from pzr.imitation.dataset import ReductionDataset
from pzr.zonotope.reduction import Reducer, ReductionResult
from pzr.zonotope.core import Zonotope


@dataclass
class TrainingResult:
    train_accuracy: float
    val_accuracy: float
    epochs: int
    train_loss_history: list[float]


class LearnedPolicy:
    """MLP-based reduction policy."""

    def __init__(
        self,
        class_names: tuple[str, ...],
        feature_mean: NDArray[np.float64],
        feature_std: NDArray[np.float64],
        weights: list[NDArray[np.float64]],
        biases: list[NDArray[np.float64]],
    ) -> None:
        self.class_names = class_names
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.weights = weights
        self.biases = biases

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward pass returning softmax probabilities."""
        x = (features - self.feature_mean) / np.maximum(self.feature_std, 1e-8)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(0, x @ w + b)
        logits = x @ self.weights[-1] + self.biases[-1]
        exp = np.exp(logits - np.max(logits))
        return exp / np.sum(exp)

    def rank_reducers(self, features: NDArray[np.float64]) -> list[str]:
        """Rank reducer names by predicted probability (highest first)."""
        proba = self.predict_proba(features)
        order = np.argsort(-proba)
        return [self.class_names[i] for i in order]

    def select_reducer(
        self,
        features: NDArray[np.float64],
        candidates: dict[str, Reducer],
        z: Zonotope,
        budget: int,
    ) -> tuple[str, ReductionResult] | None:
        """Try reducers in order of predicted probability until one succeeds."""
        for name in self.rank_reducers(features):
            reducer = candidates.get(name)
            if reducer is None:
                continue
            try:
                result = reducer.reduce(z, budget)
                if result.certificate.is_sound:
                    return name, result
            except ValueError:
                continue
        return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            class_names=np.array(self.class_names),
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            **{f"w{i}": w for i, w in enumerate(self.weights)},
            **{f"b{i}": b for i, b in enumerate(self.biases)},
            num_layers=np.array(len(self.weights)),
        )

    @classmethod
    def load(cls, path: Path) -> "LearnedPolicy":
        data = np.load(path, allow_pickle=True)
        num_layers = int(data["num_layers"])
        return cls(
            class_names=tuple(data["class_names"]),
            feature_mean=data["feature_mean"],
            feature_std=data["feature_std"],
            weights=[data[f"w{i}"] for i in range(num_layers)],
            biases=[data[f"b{i}"] for i in range(num_layers)],
        )


def _compute_class_weights(labels: NDArray[np.int64], num_classes: int) -> NDArray[np.float64]:
    """Inverse-frequency class weights: w[c] = N / (K * count[c])."""
    n = len(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return n / (num_classes * counts)


def train_policy(
    dataset: ReductionDataset,
    hidden_sizes: tuple[int, ...] = (64, 64),
    epochs: int = 200,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 42,
    balanced: bool = True,
    show_progress: bool = False,
) -> tuple[LearnedPolicy, TrainingResult]:
    """Train an MLP policy using numpy-only gradient descent."""
    rng = np.random.default_rng(seed)
    train_ds, val_ds = dataset.train_val_split(val_fraction, seed)

    feature_mean = np.mean(train_ds.features, axis=0)
    feature_std = np.std(train_ds.features, axis=0)
    safe_std = np.maximum(feature_std, 1e-8)

    X_train = (train_ds.features - feature_mean) / safe_std
    y_train = train_ds.labels
    X_val = (val_ds.features - feature_mean) / safe_std
    y_val = val_ds.labels

    sample_weights = np.ones(len(y_train), dtype=np.float64)
    if balanced:
        class_w = _compute_class_weights(y_train, dataset.num_classes)
        sample_weights = class_w[y_train]

    # Initialize weights
    layer_sizes = [dataset.num_features, *hidden_sizes, dataset.num_classes]
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        w = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
        b = np.zeros(fan_out)
        weights.append(w)
        biases.append(b)

    loss_history = []

    epoch_iter = tqdm(
        range(epochs), desc="train epochs", disable=not show_progress,
        unit="epoch", leave=False,
    )
    for epoch in epoch_iter:
        # Forward pass
        activations = [X_train]
        for w, b in zip(weights[:-1], biases[:-1]):
            z = activations[-1] @ w + b
            activations.append(np.maximum(0, z))
        logits = activations[-1] @ weights[-1] + biases[-1]
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        # Weighted cross-entropy loss
        n = X_train.shape[0]
        per_sample = -np.log(probs[np.arange(n), y_train] + 1e-12) * sample_weights
        loss = float(np.mean(per_sample))
        loss_history.append(loss)

        # Backward pass
        grad_logits = probs.copy()
        grad_logits[np.arange(n), y_train] -= 1
        grad_logits *= sample_weights[:, np.newaxis]
        grad_logits /= n

        grad_weights = []
        grad_biases = []
        delta = grad_logits

        for i in range(len(weights) - 1, -1, -1):
            gw = activations[i].T @ delta
            gb = np.sum(delta, axis=0)
            grad_weights.insert(0, gw)
            grad_biases.insert(0, gb)
            if i > 0:
                delta = (delta @ weights[i].T) * (activations[i] > 0)

        # Update
        for i in range(len(weights)):
            weights[i] -= learning_rate * grad_weights[i]
            biases[i] -= learning_rate * grad_biases[i]

    # Evaluate
    def accuracy(X, y):
        a = X
        for w, b in zip(weights[:-1], biases[:-1]):
            a = np.maximum(0, a @ w + b)
        logits = a @ weights[-1] + biases[-1]
        preds = np.argmax(logits, axis=1)
        return float(np.mean(preds == y))

    policy = LearnedPolicy(
        class_names=dataset.class_names,
        feature_mean=feature_mean,
        feature_std=feature_std,
        weights=[w.copy() for w in weights],
        biases=[b.copy() for b in biases],
    )

    return policy, TrainingResult(
        train_accuracy=accuracy(X_train, y_train),
        val_accuracy=accuracy(X_val, y_val),
        epochs=epochs,
        train_loss_history=loss_history,
    )
