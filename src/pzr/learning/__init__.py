"""Learning interface and reducer-cost models for neural reducer experiments."""

from pzr.learning.artifacts import (
    REDUCER_COST_DATASET_SCHEMA,
    load_reducer_cost_dataset,
    write_reducer_cost_dataset,
)
from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.objectives import (
    cost_sensitive_pairwise_loss,
    expected_regret_loss,
    soft_distillation_loss,
)
from pzr.learning.ranker import (
    FeatureNormalizer,
    FeatureSchema,
    ReducerMetrics,
    ReducerPolicy,
    ReducerScorer,
    ReducerTrainingResult,
    evaluate_reducer,
    train_reducer_policy,
)

__all__ = [
    "FeatureNormalizer",
    "FeatureSchema",
    "REDUCER_COST_DATASET_SCHEMA",
    "ReducerCostDataset",
    "ReducerMetrics",
    "ReducerPolicy",
    "ReducerScorer",
    "ReducerTrainingResult",
    "cost_sensitive_pairwise_loss",
    "expected_regret_loss",
    "evaluate_reducer",
    "load_reducer_cost_dataset",
    "soft_distillation_loss",
    "train_reducer_policy",
    "write_reducer_cost_dataset",
]

# Lazy exports avoid importing the RTLola stack when low-level training modules
# are imported, and keep Gymnasium optional.
_INTERFACE_MODULES = {
    **dict.fromkeys(("FeatureEncoder", "GEOMETRY15_ENCODER", "geometry15_features", "ObservationPolicy"), "encoding"),
    **dict.fromkeys(("Task", "TraceSet", "load_task"), "tasks"),
    **dict.fromkeys(("TrainingConfig", "TrainingRun", "train_selector",
                     "load_selector", "evaluate_selector", "prepared_samples"), "selector"),
    **dict.fromkeys(("MonitorSession", "Observation", "Zonotope", "evaluate_monitor"), "monitor"),
}
__all__ += list(_INTERFACE_MODULES)


def __getattr__(name: str) -> object:
    if name not in _INTERFACE_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f"{__name__}.{_INTERFACE_MODULES[name]}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
