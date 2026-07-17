import numpy as np
import pandas as pd

from pzr.learning.dataset import RankingDataset
from pzr.learning.diagnostics import dataset_diagnostics


def test_equal_stages_have_equal_objective_contribution():
    labels = ("base", "dagger1", "dagger2")
    costs = np.asarray([[0.0, 1.0], [1.0, 1.0]] * len(labels))
    dataset = RankingDataset(
        features=np.zeros((6, 1), dtype=np.float32),
        teacher_costs=costs,
        feasible=np.ones_like(costs, dtype=np.bool_),
        tie_mask=np.asarray([[True, False], [True, True]] * len(labels)),
        candidate_names=("girard", "scott"),
        feature_names=("feature",),
        splits=("train",) * 6,
        sample_ids=tuple(f"sample-{index}" for index in range(6)),
    )
    metadata = pd.DataFrame({
        "dataset_label": tuple(label for label in labels for _ in range(2)),
        "split": ("train",) * 6,
        "budget": (40,) * 6,
    })

    diagnostics = dataset_diagnostics(dataset, metadata)

    np.testing.assert_allclose(diagnostics["objective_fraction"], 1.0 / 3.0)
    assert set(diagnostics["rankable_states"]) == {1}
    assert set(diagnostics["skipped_tie_states"]) == {1}
