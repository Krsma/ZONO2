"""Runtime policy for distilled reducer-selection models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np

from pzr.control.policies import ReductionDecision, _reduction_context
from pzr.core.certificates import ReductionResult
from pzr.monitoring.base import MonitorAdapter, MonitorState
from pzr.reduction.base import Reducer, ReductionContext
from pzr.reduction.reducers import IdentityReducer
from pzr.learning.features import (
    DECISION_FEATURE_SCHEMA_VERSION,
    decision_feature_values,
)

InputT = TypeVar("InputT")


@dataclass
class LearnedReductionPolicy(Generic[InputT]):
    """Apply reducers in the probability order proposed by a distilled model."""

    checkpoint_path: str | Path
    reducers: tuple[Reducer, ...]
    budget: int
    horizon: int
    fallback_reducer: Reducer | None = None
    _torch: Any = field(init=False, repr=False)
    _model: Any = field(init=False, repr=False)
    _feature_names: tuple[str, ...] = field(init=False, repr=False)
    _class_names: tuple[str, ...] = field(init=False, repr=False)
    _candidate_names: tuple[str, ...] = field(init=False, repr=False)
    _mean: np.ndarray = field(init=False, repr=False)
    _std: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for learned policy evaluation. "
                "Install the learning extra with `python -m pip install -e .[learning]`."
            ) from exc

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        schema_version = checkpoint.get("schema_version", "")
        if schema_version != DECISION_FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"learned policy schema {schema_version!r} does not match "
                f"{DECISION_FEATURE_SCHEMA_VERSION!r}"
            )
        feature_names = tuple(checkpoint["feature_names"])
        class_names = tuple(checkpoint["class_names"])
        hidden_sizes = tuple(checkpoint.get("hidden_sizes", (64, 64)))
        model = _build_mlp(len(feature_names), len(class_names), hidden_sizes, torch=torch)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        object.__setattr__(self, "_torch", torch)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_feature_names", feature_names)
        object.__setattr__(self, "_class_names", class_names)
        object.__setattr__(
            self,
            "_candidate_names",
            tuple(checkpoint.get("candidate_reducer_names", class_names)),
        )
        object.__setattr__(
            self,
            "_mean",
            np.asarray(checkpoint["normalizer_mean"], dtype=float),
        )
        object.__setattr__(
            self,
            "_std",
            np.asarray(checkpoint["normalizer_std"], dtype=float),
        )

    def reduce_state(
        self,
        monitor: MonitorAdapter[InputT],
        state: MonitorState,
        context: ReductionContext | None = None,
    ) -> ReductionDecision:
        ctx = context or _reduction_context(monitor, state)
        if state.zonotope.generator_count <= self.budget:
            no_op = IdentityReducer()
            result = no_op.reduce(state.zonotope, self.budget, ctx)
            return ReductionDecision(
                state=state,
                result=result,
                reducer_name=no_op.name,
                is_no_op=True,
                predicted_cost=0.0,
                predicted_sequence=(no_op.name,),
                evaluated_sequences=1,
            )
        features = decision_feature_values(
            monitor,
            state,
            budget=self.budget,
            horizon=self.horizon,
            required_generators=ctx.required_generators,
        )
        vector = np.asarray([features[name] for name in self._feature_names], dtype=float)
        normalized = (vector - self._mean) / self._std
        with self._torch.no_grad():
            logits = self._model(
                self._torch.as_tensor(normalized, dtype=self._torch.float32).reshape(1, -1)
            )
            probabilities = self._torch.softmax(logits, dim=1).cpu().numpy()[0]

        ranked_classes = [
            name
            for _, name in sorted(
                zip(probabilities, self._class_names),
                key=lambda item: (-float(item[0]), item[1]),
            )
        ]
        ordered_names = _unique_names(
            (
                *ranked_classes,
                *self._candidate_names,
                *(reducer.name for reducer in self.reducers),
            )
        )
        reducer_by_name = {reducer.name: reducer for reducer in self.reducers}
        if self.fallback_reducer is not None:
            reducer_by_name.setdefault(self.fallback_reducer.name, self.fallback_reducer)
            ordered_names = _unique_names((*ordered_names, self.fallback_reducer.name))

        tried = 0
        failed = 0
        for name in ordered_names:
            reducer = reducer_by_name.get(name)
            if reducer is None:
                failed += 1
                continue
            tried += 1
            reduced = _try_certified_budgeted_reduce(monitor, reducer, state, self.budget, ctx)
            if reduced is None:
                failed += 1
                continue
            reduced_state, result = reduced
            probability = (
                float(probabilities[self._class_names.index(name)])
                if name in self._class_names
                else 0.0
            )
            return ReductionDecision(
                state=reduced_state,
                result=result,
                reducer_name=name,
                is_no_op=name == "no_reduction",
                predicted_cost=float(1.0 - probability),
                predicted_sequence=tuple(ordered_names),
                evaluated_sequences=tried,
                pruned_sequences=failed,
            )
        raise ValueError(
            "no learned-policy candidate reducer could produce a certified budgeted state"
        )


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: tuple[int, ...],
    *,
    torch: Any,
) -> Any:
    layers: list[Any] = []
    previous = input_dim
    for hidden in hidden_sizes:
        layers.append(torch.nn.Linear(previous, hidden))
        layers.append(torch.nn.ReLU())
        previous = hidden
    layers.append(torch.nn.Linear(previous, output_dim))
    return torch.nn.Sequential(*layers)


def _try_certified_budgeted_reduce(
    monitor: MonitorAdapter[InputT],
    reducer: Reducer,
    state: MonitorState,
    budget: int,
    context: ReductionContext,
) -> tuple[MonitorState, ReductionResult] | None:
    try:
        result = reducer.reduce(state.zonotope, budget, context)
    except ValueError:
        return None
    if not result.certificate.is_sound or result.reduced.generator_count > budget:
        return None
    return monitor.replace_zonotope(state, result.reduced), result


def _unique_names(names: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)
