import numpy as np

from pzr.learning.ranking import RegretDataset, RegretRankingPolicy, train_regret_policy


def test_regret_ranker_trains_and_round_trips(tmp_path):
    dataset = RegretDataset(
        features=np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ]),
        regrets=np.array([
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]),
        candidate_names=("girard", "scott"),
        feature_names=("left", "right"),
    )

    policy, result = train_regret_policy(dataset, epochs=3, seed=1)
    path = tmp_path / "ranker.npz"
    policy.save(path)
    loaded = RegretRankingPolicy.load(path)

    np.testing.assert_allclose(
        loaded.predict_regret(dataset.features[0]),
        policy.predict_regret(dataset.features[0]),
    )
    assert loaded.candidate_names == dataset.candidate_names
    assert loaded.feature_names == dataset.feature_names
    assert result.epochs == 3
