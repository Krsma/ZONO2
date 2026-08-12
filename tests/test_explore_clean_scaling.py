from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


TOOLS = Path(__file__).parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from explore_clean_scaling import (  # noqa: E402
    ExploreConfig,
    method_name,
    select_training_rows,
    smoke_config,
    training_seeds,
)
from pzr.learning.dataset import ReducerCostDataset  # noqa: E402


def test_exploratory_scope_is_nominal_and_small():
    config = ExploreConfig()

    assert config.all_sizes == (20, 36, 52, 68, 84)
    assert training_seeds(config, 36) == (
        tuple(range(20)) + tuple(range(26, 42))
    )
    assert training_seeds(config, 52)[-1] == 215
    assert training_seeds(config, 68)[-1] == 231
    assert training_seeds(config, 84)[-1] == 247
    assert config.expected_new_shards == 336
    assert config.expected_new_models == 21
    assert config.expected_reported_cells == 1400
    assert config.expected_new_cells == 840
    assert tuple(method_name(size) for size in config.all_sizes) == (
        "clean20",
        "clean36",
        "clean52",
        "clean68",
        "clean84",
    )


def test_smoke_scope_runs_one_new_model_and_cell():
    config = smoke_config()

    assert config.event_count == 100
    assert config.budgets == (40,)
    assert config.new_training_sizes == (3,)
    assert config.expected_new_shards == 1
    assert config.expected_new_models == 1
    assert config.expected_reported_cells == 1
    assert config.expected_new_cells == 1


def test_training_selection_is_seed_budget_and_split_aligned():
    rows = [
        (seed, budget)
        for seed in (0, 1, 20, 21)
        for budget in (40, 80)
    ]
    sample_ids = tuple(f"{seed}:{budget}" for seed, budget in rows)
    splits = tuple("validation" if seed >= 20 else "train" for seed, _ in rows)
    dataset = ReducerCostDataset(
        features=np.ones((len(rows), 2), dtype=np.float32),
        teacher_costs=np.ones((len(rows), 2), dtype=np.float64),
        feasible=np.ones((len(rows), 2), dtype=np.bool_),
        candidate_names=("girard", "scott"),
        feature_names=("a", "b"),
        splits=splits,
        sample_ids=sample_ids,
    )
    metadata = pd.DataFrame({
        "sample_id": sample_ids,
        "split": splits,
        "seed": [seed for seed, _ in rows],
        "budget": [budget for _, budget in rows],
    })

    selected, frame = select_training_rows(
        dataset,
        metadata,
        seeds=(0,),
        validation_seeds=(20, 21),
        budget=40,
    )

    assert tuple(frame["sample_id"]) == selected.sample_ids
    assert set(frame.loc[frame["split"] == "train", "seed"]) == {0}
    assert set(frame.loc[frame["split"] == "validation", "seed"]) == {20, 21}
    assert set(frame["budget"]) == {40}
