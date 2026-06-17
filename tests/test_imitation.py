"""Tests for imitation learning: features, traces, datasets, and policies."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pzr.imitation.dataset import ReductionDataset, build_dataset, class_balanced_indices
from pzr.imitation.features import FEATURE_NAMES, extract_features
from pzr.imitation.policy import LearnedPolicy, train_policy
from pzr.imitation.regret import RegretDataset, RegretRankingPolicy, train_regret_policy
from pzr.imitation.traces import ReductionTrace, TraceCollector
from pzr.monitoring.base import MonitorState, TriggerSpec
from pzr.systems.omni_robot import OmniRobotMonitor, generate_omni_robot_trace
from pzr.zonotope.core import Zonotope
from pzr.zonotope.protected import ProtectedReducer
from pzr.zonotope.reduction import BoxReducer, GirardReducer


class TestFeatures:
    def test_feature_count(self):
        z = Zonotope([0.0, 0.0], np.eye(2) * 0.5)
        state = MonitorState(zonotope=z, step=0, calibration_indices=(0,))
        features = extract_features(state, budget=4)
        assert features.shape == (len(FEATURE_NAMES),)

    def test_features_finite(self):
        monitor = OmniRobotMonitor()
        state = monitor.initial_state()
        trace = generate_omni_robot_trace(10, seed=0)
        for m in trace:
            result = monitor.step(state, m)
            state = result.state
        features = extract_features(state, budget=8, triggers=monitor.triggers)
        assert np.all(np.isfinite(features))

    def test_empty_zonotope_features(self):
        z = Zonotope([1.0, 2.0])
        state = MonitorState(zonotope=z, step=0)
        features = extract_features(state, budget=5)
        assert np.all(np.isfinite(features))

    def test_budget_headroom(self):
        z = Zonotope([0.0], [[1.0, 0.5, 0.3]])
        state = MonitorState(zonotope=z, step=0)
        features = extract_features(state, budget=8)
        idx = FEATURE_NAMES.index("budget_headroom")
        assert features[idx] == 5.0  # 8 - 3

    def test_trigger_features_use_trigger_zonotope(self):
        raw_z = Zonotope([0.0, 0.0, 0.0], np.diag([10.0, 1.0, 1.0]))
        trigger_z = Zonotope([0.0, 0.0], np.diag([0.5, 0.25]))
        state = MonitorState(zonotope=raw_z, step=0)
        triggers = (
            TriggerSpec("x", 0, 2.0),
            TriggerSpec("y", 1, 2.0),
        )

        features = extract_features(
            state, budget=8, triggers=triggers,
            trigger_zonotope=trigger_z,
        )

        assert features[FEATURE_NAMES.index("width_sum")] == pytest.approx(24.0)
        assert features[FEATURE_NAMES.index("trigger_width_sum")] == pytest.approx(1.5)
        assert features[FEATURE_NAMES.index("trigger_width_mean")] == pytest.approx(0.75)


class TestTraces:
    def test_record_and_retrieve(self):
        collector = TraceCollector()
        collector.record(ReductionTrace(
            features=np.array([1.0, 2.0]),
            action="girard",
            cost=0.5,
            step=3,
            episode_id=0,
        ))
        assert len(collector) == 1
        assert collector.traces[0].action == "girard"

    def test_save_and_load(self, tmp_path):
        collector = TraceCollector()
        collector.record(ReductionTrace(
            features=np.array([1.0, 2.0, 3.0]),
            action="box",
            cost=1.2,
            step=5,
            episode_id=1,
        ))
        path = tmp_path / "traces.json"
        collector.save(path)
        loaded = TraceCollector.load(path)
        assert len(loaded) == 1
        np.testing.assert_allclose(loaded.traces[0].features, [1.0, 2.0, 3.0])
        assert loaded.traces[0].action == "box"


class TestDataset:
    def test_build_from_traces(self):
        collector = TraceCollector()
        for i in range(20):
            collector.record(ReductionTrace(
                features=np.random.randn(5),
                action="girard" if i % 2 == 0 else "box",
                cost=float(i),
                step=i,
                episode_id=0,
            ))
        ds = build_dataset(collector)
        assert ds.num_samples == 20
        assert ds.num_features == 5
        assert ds.num_classes == 2
        assert set(ds.class_names) == {"box", "girard"}

    def test_train_val_split(self):
        collector = TraceCollector()
        for i in range(50):
            collector.record(ReductionTrace(
                features=np.random.randn(3),
                action="a" if i < 25 else "b",
                cost=0.0,
                step=i,
                episode_id=0,
            ))
        ds = build_dataset(collector)
        train, val = ds.train_val_split(val_fraction=0.2)
        assert train.num_samples + val.num_samples == 50
        assert train.num_samples > val.num_samples

    def test_label_distribution(self):
        collector = TraceCollector()
        for i in range(30):
            collector.record(ReductionTrace(
                features=np.random.randn(2),
                action=["a", "b", "c"][i % 3],
                cost=0.0,
                step=i,
                episode_id=0,
            ))
        ds = build_dataset(collector)
        dist = ds.label_distribution()
        assert dist["a"] == 10
        assert dist["b"] == 10
        assert dist["c"] == 10

    def test_class_balancing(self):
        labels = np.array([0, 0, 0, 0, 0, 1], dtype=np.int64)
        balanced = class_balanced_indices(labels)
        _, counts = np.unique(labels[balanced], return_counts=True)
        assert counts[0] == counts[1]


class TestPolicy:
    def _make_dataset(self, n=200, seed=42):
        """Create a linearly separable dataset for testing."""
        rng = np.random.default_rng(seed)
        features = rng.standard_normal((n, 5))
        labels = (features[:, 0] > 0).astype(np.int64)
        return ReductionDataset(features, labels, ("left", "right"))

    def test_train_converges(self):
        ds = self._make_dataset()
        policy, result = train_policy(ds, hidden_sizes=(32,), epochs=300, learning_rate=3e-3)
        assert result.train_accuracy > 0.7
        assert result.val_accuracy > 0.6

    def test_predict_proba_sums_to_one(self):
        ds = self._make_dataset()
        policy, _ = train_policy(ds, hidden_sizes=(16,), epochs=50)
        proba = policy.predict_proba(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        assert proba.shape == (2,)
        assert abs(np.sum(proba) - 1.0) < 1e-6

    def test_rank_reducers(self):
        ds = self._make_dataset()
        policy, _ = train_policy(ds, hidden_sizes=(16,), epochs=50)
        ranked = policy.rank_reducers(np.array([2.0, 0.0, 0.0, 0.0, 0.0]))
        assert set(ranked) == {"left", "right"}
        assert len(ranked) == 2

    def test_save_and_load(self, tmp_path):
        ds = self._make_dataset()
        policy, _ = train_policy(ds, hidden_sizes=(16,), epochs=50)
        path = tmp_path / "policy.npz"
        policy.save(path)
        loaded = LearnedPolicy.load(path)
        test_input = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            policy.predict_proba(test_input),
            loaded.predict_proba(test_input),
        )

    def test_select_reducer_picks_valid(self):
        ds = self._make_dataset()
        policy, _ = train_policy(ds, hidden_sizes=(16,), epochs=50)
        z = Zonotope([0.0, 0.0, 0.0], np.random.randn(3, 6))
        candidates = {
            "left": GirardReducer(name="left"),
            "right": GirardReducer(name="right"),
        }
        result = policy.select_reducer(
            np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            candidates, z, budget=4,
        )
        assert result is not None
        name, red_result = result
        assert name in ("left", "right")
        assert red_result.certificate.is_sound

    def test_select_reducer_preserves_calibration_indices(self):
        policy = LearnedPolicy(
            class_names=("box",),
            feature_mean=np.zeros(1),
            feature_std=np.ones(1),
            weights=[np.zeros((1, 1))],
            biases=[np.zeros(1)],
        )
        z = Zonotope(
            np.zeros(2),
            np.array([
                [1e-6, 10.0, 0.0, 9.0, 0.0],
                [0.0, 0.0, 10.0, 0.0, 9.0],
            ]),
        )
        candidates = {"box": ProtectedReducer(base=BoxReducer())}

        result = policy.select_reducer(
            np.array([0.0]), candidates, z, budget=3,
            protected_indices=(0,),
        )

        assert result is not None
        _, red_result = result
        np.testing.assert_allclose(red_result.reduced.generators[:, 0], z.generators[:, 0])


class TestRegretPolicy:
    def _make_regret_dataset(self, n=200, seed=42):
        rng = np.random.default_rng(seed)
        features = rng.standard_normal((n, 4))
        regrets = np.zeros((n, 3), dtype=np.float64)
        regrets[:, 0] = np.maximum(0.0, features[:, 0])
        regrets[:, 1] = np.maximum(0.0, -features[:, 0])
        regrets[:, 2] = 0.5 + 0.1 * np.abs(features[:, 1])
        return RegretDataset(features, regrets, ("left", "right", "backup"))

    def test_train_regret_policy_learns_ranking(self):
        ds = self._make_regret_dataset()
        policy, result = train_regret_policy(
            ds, hidden_sizes=(32,), epochs=250, learning_rate=3e-3, loss="mse",
        )
        assert result.train_loss < 0.2
        assert result.val_mean_chosen_regret < 0.4
        assert set(policy.rank_reducers(np.array([2.0, 0.0, 0.0, 0.0]))) == {
            "left", "right", "backup",
        }

    def test_pairwise_regret_policy_learns_ranking(self):
        ds = self._make_regret_dataset()
        policy, result = train_regret_policy(
            ds, hidden_sizes=(32,), epochs=250, learning_rate=3e-3, loss="pairwise",
        )
        assert result.val_mean_chosen_regret < 0.35
        assert policy.rank_reducers(np.array([2.0, 0.0, 0.0, 0.0]))[0] == "right"

    def test_regret_policy_save_and_load(self, tmp_path):
        ds = self._make_regret_dataset()
        policy, _ = train_regret_policy(ds, hidden_sizes=(16,), epochs=50)
        path = tmp_path / "regret_policy.npz"
        policy.save(path)
        loaded = RegretRankingPolicy.load(path)

        x = np.array([0.2, -0.3, 0.1, 0.0])
        np.testing.assert_allclose(policy.predict_regret(x), loaded.predict_regret(x))
        assert loaded.candidate_names == policy.candidate_names
        assert loaded.feature_names == policy.feature_names

    def test_regret_select_reducer_preserves_calibration_indices(self):
        policy = RegretRankingPolicy(
            candidate_names=("box",),
            feature_mean=np.zeros(1),
            feature_std=np.ones(1),
            weights=[np.zeros((1, 1))],
            biases=[np.zeros(1)],
        )
        z = Zonotope(
            np.zeros(2),
            np.array([
                [1e-6, 10.0, 0.0, 9.0, 0.0],
                [0.0, 0.0, 10.0, 0.0, 9.0],
            ]),
        )
        candidates = {"box": ProtectedReducer(base=BoxReducer())}

        result = policy.select_reducer(
            np.array([0.0]), candidates, z, budget=3,
            protected_indices=(0,),
        )

        assert result is not None
        _, red_result = result
        np.testing.assert_allclose(red_result.reduced.generators[:, 0], z.generators[:, 0])
