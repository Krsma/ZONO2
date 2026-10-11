"""Explicit pre-event observation encoders for Gym and custom policies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from pzr.rtlola.features import RTL_RANKING_FEATURE_NAMES

if TYPE_CHECKING:
    from pzr.learning.monitor import Observation


@dataclass(frozen=True)
class FeatureEncoder:
    """Map inspection data to a fixed-shape, finite float32 model input.

    ``function`` receives only a pre-event Observation, never a monitor, future
    events, teacher labels, or references. Request geometry explicitly when
    using dynamic/total zonotopes. Functions should be deterministic and must
    handle empty geometry at reset. Keep trainable embedding modules inside
    the policy: the NumPy Gym boundary does not preserve autograd graphs.

    ``name`` identifies the feature definition in experiment configurations;
    version it when changing the computation. It is not a code hash.
    """

    name: str
    shape: tuple[int, ...]
    function: Callable[[Observation], np.ndarray]
    requires_geometry: bool = False
    feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.shape or any(type(n) is not int or n < 1 for n in self.shape):
            raise ValueError('encoder requires a name and a positive fixed shape')
        if not callable(self.function):
            raise TypeError('encoder function must be callable')
        if self.feature_names and (len(self.shape) != 1 or len(self.feature_names) != self.shape[0]):
            raise ValueError('feature_names must match a vector encoder shape')

    def __call__(self, observation: Observation) -> np.ndarray:
        if self.requires_geometry and (observation.dynamic is None or observation.total is None):
            raise ValueError('encoder requires observe(geometry=True)')
        values = np.array(self.function(observation), dtype=np.float32, copy=True)
        if values.shape != self.shape or not np.isfinite(values).all():
            raise ValueError(f'encoder {self.name!r} must return finite values of shape {self.shape}, got {values.shape}')
        return values


def geometry15_features(observation: Observation) -> np.ndarray:
    """Raw paper features, before schema log1p and train-fitted normalization.

    The computation is ``pzr.rtlola.features.ranking_features_from_matrices``.
    Observation.features retains its native Geometry15 meaning with any encoder.
    """
    return observation.features


GEOMETRY15_ENCODER = FeatureEncoder(
    name='rtlola.current-zonotope.v2', shape=(15,), function=geometry15_features,
    feature_names=RTL_RANKING_FEATURE_NAMES,
)


@dataclass(frozen=True)
class ObservationPolicy:
    """Use the same encoder for live evaluation and Gym experiments.

    score_function receives one encoded tensor (without a batch axis) and
    returns four finite lower-is-better scores in native candidate order:
    girard, scott, pca, combastel. It owns any normalization or embedding.
    """

    encoder: FeatureEncoder
    score_function: Callable[[np.ndarray], np.ndarray]

    def __call__(self, observation: Observation) -> np.ndarray:
        scores = np.asarray(self.score_function(self.encoder(observation)), dtype=np.float64)
        if scores.shape != (4,) or not np.isfinite(scores).all():
            raise ValueError('observation policy must return four finite lower-is-better scores')
        return scores
