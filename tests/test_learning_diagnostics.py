import numpy as np
import pandas as pd

from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.diagnostics import dataset_diagnostics


def test_equal_named_inputs_have_equal_soft_objective_contribution():
    labels = ("clean", "dart")
    costs = np.asarray([[0.0, 1.0], [1.0, 1.0]] * len(labels))
    dataset = ReducerCostDataset(
        features=np.zeros((4, 1), dtype=np.float32), teacher_costs=costs,
        feasible=np.ones_like(costs, dtype=np.bool_), candidate_names=("girard", "scott"),
        feature_names=("feature",), splits=("train",) * 4,
        sample_ids=tuple(f"sample-{index}" for index in range(4)),
    )
    metadata = pd.DataFrame({
        "dataset_label": tuple(label for label in labels for _ in range(2)),
        "split": ("train",) * 4, "budget": (40,) * 4,
        "teacher_action": ("girard",) * 4, "executed_action": ("girard",) * 4,
    })
    diagnostics = dataset_diagnostics(dataset, metadata)
    np.testing.assert_allclose(diagnostics["soft_objective_fraction"], 0.5)
    assert set(diagnostics["rankable_states"]) == {1}
    assert set(diagnostics["skipped_tie_states"]) == {1}
