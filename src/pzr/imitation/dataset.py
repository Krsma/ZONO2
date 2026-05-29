"""Dataset construction from expert traces for policy training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pzr.imitation.traces import TraceCollector


@dataclass
class ReductionDataset:
    """Features, labels, and metadata for training a reduction policy."""

    features: NDArray[np.float64]
    labels: NDArray[np.int64]
    class_names: tuple[str, ...]

    @property
    def num_samples(self) -> int:
        return self.features.shape[0]

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def label_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self.labels:
            name = self.class_names[label]
            counts[name] = counts.get(name, 0) + 1
        return counts

    def train_val_split(
        self, val_fraction: float = 0.2, seed: int = 42,
    ) -> tuple["ReductionDataset", "ReductionDataset"]:
        rng = np.random.default_rng(seed)
        n = self.num_samples
        indices = rng.permutation(n)
        split = int(n * (1.0 - val_fraction))
        train_idx = indices[:split]
        val_idx = indices[split:]
        return (
            ReductionDataset(self.features[train_idx], self.labels[train_idx], self.class_names),
            ReductionDataset(self.features[val_idx], self.labels[val_idx], self.class_names),
        )


def build_dataset(collector: TraceCollector) -> ReductionDataset:
    """Build a dataset from collected traces."""
    traces = collector.traces
    if not traces:
        raise ValueError("no traces to build dataset from")

    actions = sorted(set(t.action for t in traces))
    action_to_idx = {a: i for i, a in enumerate(actions)}

    features = np.stack([t.features for t in traces])
    labels = np.array([action_to_idx[t.action] for t in traces], dtype=np.int64)
    return ReductionDataset(
        features=features,
        labels=labels,
        class_names=tuple(actions),
    )


def class_balanced_indices(
    labels: NDArray[np.int64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Oversample minority classes to match the largest class count."""
    if rng is None:
        rng = np.random.default_rng(42)
    unique, counts = np.unique(labels, return_counts=True)
    max_count = int(np.max(counts))
    indices = []
    for cls, count in zip(unique, counts):
        cls_indices = np.where(labels == cls)[0]
        if count < max_count:
            extra = rng.choice(cls_indices, size=max_count - count, replace=True)
            indices.append(np.concatenate([cls_indices, extra]))
        else:
            indices.append(cls_indices)
    return rng.permutation(np.concatenate(indices))
