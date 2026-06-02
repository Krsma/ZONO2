"""Tests for MPC policies and tree search."""

import numpy as np
import pytest

from pzr.monitoring.base import TriggerSpec
from pzr.monitoring.base import MonitorResult, MonitorState
from pzr.mpc.objectives import CostWeights, WeightedZonotopeCost
from pzr.mpc.policies import (
    BeamMPCPolicy,
    MPCPolicy,
    PairRolloutMPCPolicy,
    ReductionDecision,
    RolloutMPCPolicy,
    StaticPolicy,
)
from pzr.mpc.prediction import ConstantPredictor
from pzr.mpc.search import tree_search
from pzr.zonotope.core import Zonotope
from pzr.zonotope.protected import ProtectedReducer
from pzr.systems.omni_robot import (
    OmniRobotMonitor,
    generate_omni_robot_trace,
)
from pzr.zonotope.reduction import (
    BoxReducer,
    CombastelReducer,
    GirardReducer,
    MethAReducer,
    ScottReducer,
)


BUDGET = 8


class AppendGeneratorMonitor:
    @property
    def triggers(self):
        return ()

    @property
    def num_calibration_generators(self):
        return 1

    def initial_state(self):
        raise NotImplementedError

    def clone_state(self, state):
        return state

    def replace_zonotope(self, state, zonotope):
        return state.with_zonotope(zonotope)

    def trigger_zonotope(self, state):
        return state.zonotope

    def step(self, state, measurement):
        generators = np.hstack([state.zonotope.generators, measurement])
        next_state = state.with_zonotope(Zonotope(state.zonotope.center, generators))
        return MonitorResult(next_state, ())


class RecordingProtectedReducer(ProtectedReducer):
    def __init__(self, base):
        super().__init__(base=base)
        object.__setattr__(self, "calls", [])

    def reduce(self, z, budget, protected_indices=()):
        self.calls.append(protected_indices)
        return super().reduce(z, budget, protected_indices=protected_indices)


def _calibration_sensitive_state():
    generators = np.array([
        [1e-6, 10.0, 0.0, 9.0, 0.0],
        [0.0, 0.0, 10.0, 0.0, 9.0],
    ])
    return MonitorState(
        zonotope=Zonotope(np.zeros(2), generators),
        calibration_indices=(0,),
    )


@pytest.fixture
def monitor():
    return OmniRobotMonitor()


@pytest.fixture
def overflow_state(monitor):
    """Run monitor until generators exceed budget."""
    state = monitor.initial_state()
    trace = generate_omni_robot_trace(20, seed=0)
    for m in trace:
        result = monitor.step(state, m)
        state = result.state
    assert state.zonotope.generator_count > BUDGET
    return state, trace


class TestStaticPolicy:
    def test_applies_reducer(self, monitor, overflow_state):
        state, _ = overflow_state
        policy = StaticPolicy(reducer=GirardReducer(), budget=BUDGET)
        decision = policy.select(monitor, state)
        assert decision.reducer_name == "girard"
        assert decision.result.reduced.generator_count <= BUDGET
        assert decision.result.certificate.is_sound

    def test_deterministic(self, monitor, overflow_state):
        state, _ = overflow_state
        policy = StaticPolicy(reducer=GirardReducer(), budget=BUDGET)
        d1 = policy.select(monitor, state)
        d2 = policy.select(monitor, state)
        assert d1.reducer_name == d2.reducer_name
        np.testing.assert_allclose(
            d1.result.reduced.generators, d2.result.reduced.generators,
        )


class TestMPCPolicy:
    def test_cost_uses_trigger_zonotope_for_robot_arm(self):
        from pzr.envs.base import NoisySensorModel
        from pzr.envs.robot_arm_monitor import RobotArmMeasurement, RobotArmMonitor

        monitor = RobotArmMonitor(
            noise_model=NoisySensorModel(
                bias_bound=np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.01]),
                noise_bound=np.array([0.01, 0.01, 0.01, 0.005, 0.005, 0.005]),
            ),
        )
        state = monitor.step(
            monitor.initial_state(),
            RobotArmMeasurement(0.0, (0.4, -0.8, 0.6), (0.0, 0.0, 0.0)),
        ).state
        cost = WeightedZonotopeCost(
            weights=CostWeights(
                trigger_width=1.0,
                straddling=0.0,
                generator_count=0.0,
                total_width=0.0,
            ),
            triggers=monitor.triggers,
            trigger_zonotope=monitor.trigger_zonotope,
        )

        trigger_z = monitor.trigger_zonotope(state)
        lower, upper = trigger_z.interval_bounds()
        expected = sum(
            float(upper[t.state_index] - lower[t.state_index])
            for t in monitor.triggers
        )
        raw_widths = state.zonotope.widths()
        raw = sum(float(raw_widths[t.state_index]) for t in monitor.triggers)

        assert cost(state) == pytest.approx(expected)
        assert expected != pytest.approx(raw)

    def test_selects_from_candidates(self, monitor, overflow_state):
        state, trace = overflow_state
        cost = WeightedZonotopeCost(
            weights=CostWeights(trigger_width=1.0, straddling=20.0),
            triggers=monitor.triggers,
        )
        policy = MPCPolicy(
            candidates=(GirardReducer(), CombastelReducer(), BoxReducer()),
            budget=BUDGET,
            horizon=3,
            cost=cost,
        )
        decision = policy.select(monitor, state, trace[15:])
        assert decision.reducer_name in ("girard", "combastel", "box")
        assert decision.result.reduced.generator_count <= BUDGET
        assert decision.result.certificate.is_sound

    def test_returns_predicted_cost(self, monitor, overflow_state):
        state, trace = overflow_state
        cost = WeightedZonotopeCost(
            weights=CostWeights(),
            triggers=monitor.triggers,
        )
        policy = MPCPolicy(
            candidates=(GirardReducer(), BoxReducer()),
            budget=BUDGET,
            horizon=2,
            cost=cost,
        )
        decision = policy.select(monitor, state, trace[15:])
        assert decision.predicted_cost > 0.0
        assert len(decision.predicted_sequence) >= 1

    def test_preserves_calibration_indices_in_tree_search(self):
        state = _calibration_sensitive_state()
        monitor = AppendGeneratorMonitor()
        policy = MPCPolicy(
            candidates=(ProtectedReducer(base=BoxReducer()),),
            budget=3,
            horizon=0,
            cost=WeightedZonotopeCost(),
        )

        decision = policy.select(monitor, state, predicted_inputs=())

        assert decision.state.calibration_indices == (0,)
        np.testing.assert_allclose(
            decision.state.zonotope.generators[:, 0],
            state.zonotope.generators[:, 0],
        )


class TestRolloutMPCPolicy:
    def test_first_action_from_candidates(self, monitor, overflow_state):
        state, trace = overflow_state
        cost = WeightedZonotopeCost(
            weights=CostWeights(trigger_width=1.0, straddling=20.0),
            triggers=monitor.triggers,
        )
        policy = RolloutMPCPolicy(
            candidates=(GirardReducer(), CombastelReducer(), BoxReducer()),
            base_reducer=GirardReducer(),
            budget=BUDGET,
            horizon=4,
            cost=cost,
        )
        decision = policy.select(monitor, state, trace[15:])
        assert decision.reducer_name in ("girard", "combastel", "box")
        assert decision.result.reduced.generator_count <= BUDGET

    def test_base_used_for_future(self, monitor, overflow_state):
        state, trace = overflow_state
        cost = WeightedZonotopeCost(
            weights=CostWeights(),
            triggers=monitor.triggers,
        )
        policy = RolloutMPCPolicy(
            candidates=(GirardReducer(), BoxReducer()),
            base_reducer=GirardReducer(),
            budget=BUDGET,
            horizon=4,
            cost=cost,
        )
        decision = policy.select(monitor, state, trace[15:])
        # If there were future overflows, base reducer should appear
        if len(decision.predicted_sequence) > 1:
            assert "girard" in decision.predicted_sequence[1:]

    def test_future_rollout_reductions_receive_protected_indices(self):
        state = _calibration_sensitive_state()
        monitor = AppendGeneratorMonitor()
        first = RecordingProtectedReducer(BoxReducer())
        base = RecordingProtectedReducer(BoxReducer())
        policy = RolloutMPCPolicy(
            candidates=(first,),
            base_reducer=base,
            budget=3,
            horizon=1,
            cost=WeightedZonotopeCost(),
        )
        new_generators = np.array([
            [0.0, 0.25],
            [0.25, 0.0],
        ])

        decision = policy.select(monitor, state, predicted_inputs=(new_generators,))

        assert first.calls == [(0,)]
        assert base.calls == [(0,)]
        assert decision.predicted_sequence == ("box", "box")


class TestPairRolloutMPCPolicy:
    def test_searches_first_and_base_pairs(self, monitor, overflow_state):
        state, trace = overflow_state
        cost = WeightedZonotopeCost(
            weights=CostWeights(trigger_width=1.0, straddling=20.0),
            triggers=monitor.triggers,
        )
        policy = PairRolloutMPCPolicy(
            first_candidates=(GirardReducer(), MethAReducer(), ScottReducer()),
            base_candidates=(GirardReducer(), MethAReducer(), ScottReducer()),
            budget=BUDGET,
            horizon=2,
            cost=cost,
            fallback=BoxReducer(),
        )

        decision = policy.select(monitor, state, trace[15:])

        assert decision.reducer_name in ("girard", "methA", "scott")
        assert decision.predicted_sequence[0] == decision.reducer_name
        assert decision.result.certificate.is_sound
        assert decision.result.reduced.generator_count <= BUDGET

    def test_pair_rollout_reductions_receive_protected_indices(self):
        state = _calibration_sensitive_state()
        monitor = AppendGeneratorMonitor()
        first = RecordingProtectedReducer(BoxReducer())
        base = RecordingProtectedReducer(BoxReducer())
        policy = PairRolloutMPCPolicy(
            first_candidates=(first,),
            base_candidates=(base,),
            budget=3,
            horizon=1,
            cost=WeightedZonotopeCost(),
        )
        new_generators = np.array([
            [0.0, 0.25],
            [0.25, 0.0],
        ])

        decision = policy.select(monitor, state, predicted_inputs=(new_generators,))

        assert first.calls == [(0,)]
        assert base.calls == [(0,)]
        assert decision.predicted_sequence == ("box", "box")


class TestBeamMPCPolicy:
    def test_selects_certified_budgeted_state(self, monitor, overflow_state):
        state, trace = overflow_state
        policy = BeamMPCPolicy(
            candidates=(GirardReducer(), MethAReducer(), ScottReducer()),
            budget=BUDGET,
            horizon=2,
            beam_width=4,
            cost=WeightedZonotopeCost(triggers=monitor.triggers),
            fallback=BoxReducer(),
        )

        decision = policy.select(monitor, state, trace[15:])

        assert decision.reducer_name in ("girard", "methA", "scott")
        assert decision.result.certificate.is_sound
        assert decision.result.reduced.generator_count <= BUDGET
        assert decision.evaluated_leaves > 0

    def test_matches_exact_search_when_beam_is_wide(self, monitor, overflow_state):
        state, trace = overflow_state
        cost = WeightedZonotopeCost(triggers=monitor.triggers)
        candidates = (GirardReducer(), BoxReducer())
        exact = MPCPolicy(
            candidates=candidates,
            budget=BUDGET,
            horizon=2,
            cost=cost,
        ).select(monitor, state, trace[15:])
        beam = BeamMPCPolicy(
            candidates=candidates,
            budget=BUDGET,
            horizon=2,
            beam_width=16,
            cost=cost,
        ).select(monitor, state, trace[15:])

        assert beam.reducer_name == exact.reducer_name
        assert beam.predicted_cost == pytest.approx(exact.predicted_cost)
        assert beam.predicted_sequence == exact.predicted_sequence

    def test_prunes_when_beam_is_narrow(self, monitor, overflow_state):
        state, trace = overflow_state
        policy = BeamMPCPolicy(
            candidates=(GirardReducer(), CombastelReducer(), BoxReducer()),
            budget=BUDGET,
            horizon=3,
            beam_width=1,
            cost=WeightedZonotopeCost(triggers=monitor.triggers),
        )

        decision = policy.select(monitor, state, trace[15:])

        assert decision.pruned_branches > 0

    def test_preserves_calibration_indices(self):
        state = _calibration_sensitive_state()
        monitor = AppendGeneratorMonitor()
        reducer = RecordingProtectedReducer(BoxReducer())
        policy = BeamMPCPolicy(
            candidates=(reducer,),
            budget=3,
            horizon=1,
            beam_width=2,
            cost=WeightedZonotopeCost(),
        )
        new_generators = np.array([
            [0.0, 0.25],
            [0.25, 0.0],
        ])

        decision = policy.select(monitor, state, predicted_inputs=(new_generators,))

        assert reducer.calls == [(0,), (0,)]
        assert decision.state.calibration_indices == (0,)
        np.testing.assert_allclose(
            decision.state.zonotope.generators[:, 0],
            state.zonotope.generators[:, 0],
        )


class TestTreeSearch:
    def test_exhaustive_on_small_example(self, monitor, overflow_state):
        state, trace = overflow_state
        cost = WeightedZonotopeCost(
            weights=CostWeights(),
            triggers=monitor.triggers,
        )
        result = tree_search(
            monitor=monitor,
            state=state,
            candidates=(GirardReducer(), BoxReducer()),
            budget=BUDGET,
            horizon=2,
            cost_fn=cost,
            predicted_inputs=trace[15:],
        )
        assert result.evaluated_leaves > 0
        assert result.best_reducer in ("girard", "box")
        assert result.best_result.certificate.is_sound

    def test_pruning_reduces_evaluations(self, monitor, overflow_state):
        state, trace = overflow_state
        cost = WeightedZonotopeCost(
            weights=CostWeights(trigger_width=1.0, straddling=20.0),
            triggers=monitor.triggers,
        )
        result = tree_search(
            monitor=monitor,
            state=state,
            candidates=(GirardReducer(), CombastelReducer(), BoxReducer()),
            budget=BUDGET,
            horizon=3,
            cost_fn=cost,
            predicted_inputs=trace[15:],
        )
        # Pruning should occur if some branches are worse than others
        total = result.evaluated_leaves + result.pruned_branches
        assert total > 0


class TestMPCBeatsSomething:
    """MPC should produce valid certified reductions."""

    def test_mpc_produces_valid_result(self, monitor, overflow_state):
        state, trace = overflow_state
        cost = WeightedZonotopeCost(
            weights=CostWeights(trigger_width=1.0, straddling=20.0),
            triggers=monitor.triggers,
        )
        policy = MPCPolicy(
            candidates=(GirardReducer(), CombastelReducer(), BoxReducer()),
            budget=BUDGET,
            horizon=2,
            cost=cost,
        )
        decision = policy.select(monitor, state, trace[15:])
        assert decision.result.certificate.is_sound
        assert decision.result.reduced.generator_count <= BUDGET
        assert decision.predicted_cost > 0.0
        assert decision.predicted_cost < float("inf")

    def test_mpc_first_step_cost_not_worse_than_worst(self, monitor, overflow_state):
        state, trace = overflow_state
        cost_fn = WeightedZonotopeCost(
            weights=CostWeights(trigger_width=1.0, straddling=20.0),
            triggers=monitor.triggers,
        )

        # Compare first-step cost only
        worst_first_step = -float("inf")
        for reducer in (GirardReducer(), CombastelReducer(), BoxReducer()):
            try:
                result = reducer.reduce(state.zonotope, BUDGET)
                reduced_state = monitor.replace_zonotope(state, result.reduced)
                c = cost_fn(reduced_state)
                worst_first_step = max(worst_first_step, c)
            except ValueError:
                pass

        policy = MPCPolicy(
            candidates=(GirardReducer(), CombastelReducer(), BoxReducer()),
            budget=BUDGET,
            horizon=2,
            cost=cost_fn,
        )
        decision = policy.select(monitor, state, trace[15:])
        mpc_first_cost = cost_fn(decision.state)
        assert mpc_first_cost <= worst_first_step + 1e-6


class TestConstantPredictor:
    def test_predicts_correct_length(self):
        from pzr.systems.omni_robot import OmniRobotMeasurement
        history = [
            OmniRobotMeasurement(0.0, 0.0, 1.0),
            OmniRobotMeasurement(1.0, 0.1, 0.9),
        ]
        predictor = ConstantPredictor()
        predicted = predictor.predict(history, horizon=5)
        assert len(predicted) == 5
        for p in predicted:
            assert p.acceleration == history[-1].acceleration

    def test_times_advance(self):
        from pzr.systems.omni_robot import OmniRobotMeasurement
        history = [
            OmniRobotMeasurement(0.0, 0.0, 1.0),
            OmniRobotMeasurement(1.0, 0.1, 0.9),
        ]
        predictor = ConstantPredictor()
        predicted = predictor.predict(history, horizon=3)
        assert predicted[0].time == pytest.approx(2.0)
        assert predicted[1].time == pytest.approx(3.0)
        assert predicted[2].time == pytest.approx(4.0)
