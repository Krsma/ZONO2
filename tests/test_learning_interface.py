"""Learning workflow contracts: scientific parity, custom models, and isolation."""
from dataclasses import replace
from pathlib import Path
import random
import shutil

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import torch
from torch import nn

from pzr import learning
from pzr.learning.ranker import train_reducer_policy
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA
from pzr.rtlola.scenarios import RtlolaTrace


@pytest.fixture(scope='module')
def task():
    return learning.load_task()


def factory(inputs, actions):
    return nn.Sequential(nn.Linear(inputs, 8), nn.Tanh(), nn.Linear(8, actions))


def test_bundle_and_splits(task):
    assert task.dataset.num_samples == 2982
    assert set(task.sample_metadata.loc[task.sample_metadata.split == 'train', 'seed']) == {0, 1, 2, 3}
    assert set(task.sample_metadata.loc[task.sample_metadata.split == 'validation', 'seed']) == {20, 21}
    assert len(task.training_traces.traces) == 4
    assert len(task.validation_traces.traces) == 2
    assert task.frozen_baseline().candidate_names == task.dataset.candidate_names
    with pytest.raises(ValueError, match='unavailable'):
        learning.load_task(budget=80)


def test_bundle_hash_tampering(task, tmp_path):
    root = tmp_path / 'bundle'
    shutil.copytree(task.bundle_root, root)
    with (root / 'traces/seed-0.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='integrity'):
        learning.load_task(bundle=root)


def test_default_training_matches_existing(task):
    config = learning.TrainingConfig(epochs=2)
    run = learning.train_selector(task, config=config)
    original, history = train_reducer_policy(task.dataset, RTL_RANKING_FEATURE_SCHEMA,
                                            objective='pairwise', epochs=2)
    assert_array_equal(run.policy.predict_scores(task.validation.features),
                       original.predict_scores(task.validation.features))
    assert run.history.val_loss_history == history.val_loss_history


def test_custom_network_training_roundtrip_and_rng(task, tmp_path):
    before_torch = torch.get_rng_state().clone()
    before_numpy = np.random.get_state()
    before_python = random.getstate()
    config = learning.TrainingConfig(epochs=2)
    run = learning.train_selector(task, model_factory=factory, config=config)
    assert_array_equal(torch.get_rng_state(), before_torch)
    assert_array_equal(np.random.get_state()[1], before_numpy[1])
    assert random.getstate() == before_python
    again = learning.train_selector(task, model_factory=factory, config=config)
    assert_array_equal(run.policy.predict_scores(task.validation.features),
                       again.policy.predict_scores(task.validation.features))
    assert run.history.train_loss_history[0] != run.history.train_loss_history[1]
    run.save(tmp_path / 'model')
    with pytest.raises(ValueError, match='model_factory'):
        learning.load_selector(tmp_path / 'model')
    loaded = learning.load_selector(tmp_path / 'model', model_factory=factory)
    assert_array_equal(run.policy.predict_scores(task.validation.features),
                       loaded.predict_scores(task.validation.features))
    with pytest.raises(FileExistsError):
        run.save(tmp_path / 'model')
    metrics = learning.evaluate_selector(loaded, task.validation)
    assert 0 <= metrics['teacher_agreement'] <= 1


def test_default_roundtrip_and_train_only_normalizer(task, tmp_path):
    run = learning.train_selector(task, config=learning.TrainingConfig(epochs=1))
    expected = RTL_RANKING_FEATURE_SCHEMA.transform(task.training.features).mean(axis=0, dtype=np.float64)
    assert_allclose(run.policy.normalizer.mean, expected, rtol=1e-6)
    run.save(tmp_path / 'model')
    loaded = learning.load_selector(tmp_path / 'model')
    assert_array_equal(loaded.predict_scores(task.validation.features), run.policy.predict_scores(task.validation.features))
    batch = learning.prepared_samples(task.validation, run.policy.normalizer)
    assert batch.tensors[0].shape == (994, 15)
    assert batch.tensors[1].dtype == torch.float64


def test_invalid_network_fails(task):
    with pytest.raises((ValueError, RuntimeError), match='output|scores'):
        learning.train_selector(task, model_factory=lambda n, a: nn.Linear(n, a + 1),
                               config=learning.TrainingConfig(epochs=1))


def test_zonotope_preserves_signed_and_zero_geometry():
    z = learning.Zonotope(np.array([1., 0.]), np.array([[-2., 0.], [0., 0.]]))
    assert z.generators[0, 0] == -2
    assert z.generators.shape == (2, 2)
    assert not z.generators.flags.writeable
    with pytest.raises(ValueError):
        learning.Zonotope(np.zeros(2), np.zeros((3, 4)))


@pytest.fixture
def native_trace(task):
    from pzr.rtlola.binding import require_binding
    try:
        require_binding()
    except (ImportError, RuntimeError) as error:
        pytest.skip(str(error))
    t = task.validation_traces.traces[0]
    return RtlolaTrace(t.scenario, t.trace_kind, t.seed, t.events[:12])


def test_session_replay_geometry_and_evaluation(native_trace):
    left = learning.MonitorSession(native_trace, budget=40)
    right = learning.MonitorSession(native_trace, budget=40)
    while not left.done:
        obs = left.observe(geometry=True)
        assert obs.dynamic.generators.shape[0] == obs.dynamic.center.size
        assert not obs.features.flags.writeable
        a, b = left.step(0), right.step(np.arange(4, dtype=float))
        assert a['executed_action'] == b['executed_action']
        assert a['approx_loss'] == b['approx_loss']
    assert left.records[0]['executed_action'] == 'none'
    result = learning.evaluate_monitor('girard', learning.TraceSet((native_trace,), 40, 'validation'))
    assert result.iloc[0].final_approx_loss == left.records[-1]['approx_loss']
    assert result.iloc[0].sum_approx_loss == sum(r['approx_loss'] for r in left.records)
    with pytest.raises(RuntimeError):
        left.step(0)


def test_gym_checker_and_terminal_reward(native_trace):
    gym = pytest.importorskip('gymnasium')
    from gymnasium.utils.env_checker import check_env
    from pzr.learning.gym import ReducerEnv
    traces = learning.TraceSet((native_trace,), 40, 'validation')
    env = ReducerEnv(traces)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=42)
    while True:
        obs, reward, terminated, truncated, info = env.step(0)
        assert 'approx_loss' not in info and 'reference' not in obs
        if terminated:
            assert reward == -env.session.records[-1]['approx_loss']
            break
        assert reward == 0 and not truncated
    env = ReducerEnv(traces, max_steps=2)
    env.reset(seed=42)
    env.step(0)
    _, reward, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated and reward == 0
    with pytest.raises(RuntimeError):
        env.step(0)


def test_session_fallback_for_infeasible_bound(native_trace):
    session = learning.MonitorSession(native_trace, budget=0)
    for _ in range(5):
        session.step(0)
    assert any(r['fallback_used'] for r in session.records)
    assert any(r['infeasible_candidate_count'] == 4 for r in session.records)


def test_native_evaluator_matches_existing_benchmark(native_trace, task):
    from pzr.rtlola.actions import default_action_catalog
    from pzr.rtlola.benchmark import RtlolaBenchmarkConfig, run_event_trace_benchmark
    from pzr.rtlola.learned_policy import RtlolaReducerPolicy
    policy = task.frozen_baseline()
    existing = run_event_trace_benchmark(
        RtlolaBenchmarkConfig(scenario='robot_arm', length=len(native_trace.events), budget=40,
                             reference_mode='exact', mpc_candidate_names=policy.candidate_names),
        native_trace.events, trace_kind=native_trace.trace_kind, seed=native_trace.seed,
        method='learning', policy=RtlolaReducerPolicy(policy, default_action_catalog()),
    )
    assert not existing.failures
    new = learning.evaluate_monitor(policy, learning.TraceSet((native_trace,), 40, 'validation')).iloc[0]
    original = existing.summary.iloc[0]
    for name in ('final_approx_loss', 'sum_approx_loss', 'fallback_count', 'infeasible_candidate_count', 'fpr', 'fnr'):
        assert_allclose(new[name], original[name], equal_nan=True)


def test_custom_network_receives_batch_dimension_for_single_state(task):
    def batched_network(inputs, actions):
        return nn.Sequential(nn.Linear(inputs, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, actions))
    run = learning.train_selector(task, model_factory=batched_network,
                                 config=learning.TrainingConfig(epochs=1))
    single = run.policy.predict_scores(task.validation.features[0])
    batch = run.policy.predict_scores(task.validation.features[:1])
    assert single.shape == (4,)
    assert_array_equal(single, batch[0])


def test_feature_encoder_validation_and_default():
    from types import SimpleNamespace
    obs = SimpleNamespace(features=np.arange(15, dtype=np.float32), dynamic=None, total=None)
    encoded = learning.GEOMETRY15_ENCODER(obs)
    assert_array_equal(encoded, obs.features)
    encoded[0] = -1
    assert obs.features[0] == 0
    for result in (np.zeros(2), np.array([np.nan]), np.array([np.inf])):
        encoder = learning.FeatureEncoder('invalid.v1', (1,), lambda o: result)
        with pytest.raises(ValueError, match='finite values of shape'):
            encoder(obs)
    geometry = learning.FeatureEncoder('geometry.v1', (1,), lambda o: np.zeros(1), True)
    with pytest.raises(ValueError, match='geometry=True'):
        geometry(obs)
    tensor = learning.FeatureEncoder('tensor.v1', (2, 3), lambda o: np.ones((2, 3)))
    assert tensor(obs).shape == (2, 3)
    with pytest.raises(ValueError, match='positive fixed shape'):
        learning.FeatureEncoder('empty.v1', (0,), lambda o: np.zeros(0))
    with pytest.raises(ValueError, match='four finite'):
        learning.ObservationPolicy(tensor, lambda x: np.zeros(3))(obs)


def test_custom_geometry_encoder_native_parity(native_trace):
    from gymnasium.utils.env_checker import check_env
    from pzr.learning.gym import ReducerEnv

    def properties(obs):
        radius = np.abs(obs.dynamic.generators).sum(axis=1)
        return np.array([obs.budget, radius.sum(), radius.max(initial=0)])

    encoder = learning.FeatureEncoder('radius.v1', (3,), properties, requires_geometry=True)
    traces = learning.TraceSet((native_trace,), 40, 'validation')
    custom = ReducerEnv(traces, feature_encoder=encoder)
    check_env(custom, skip_render_check=True)
    default = ReducerEnv(traces)
    custom.reset(seed=42)
    observation, _ = default.reset(seed=42)
    assert_array_equal(observation['features'], default.session.observe().features)
    while True:
        observation, reward, done, truncated, info = custom.step(0)
        reference, expected_reward, expected_done, expected_truncated, expected_info = default.step(0)
        raw = custom.session.observe(geometry=True)
        assert_allclose(observation['features'], properties(raw))
        assert observation['features'].dtype == np.float32
        assert custom.observation_space.contains(observation)
        assert_array_equal(observation['action_mask'], reference['action_mask'])
        assert (reward, done, truncated, info) == (expected_reward, expected_done, expected_truncated, expected_info)
        if done:
            break
    policy = learning.ObservationPolicy(encoder, lambda features: np.arange(4))
    report = learning.evaluate_monitor(policy, traces).iloc[0]
    assert report.final_approx_loss == custom.session.records[-1]['approx_loss']
    assert report.fallback_count == sum(r['fallback_used'] for r in custom.session.records)
    custom.close()
    default.close()
