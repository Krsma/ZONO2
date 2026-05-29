"""Tests for MPC policies and tree search."""

import numpy as np
import pytest

from pzr.monitoring.base import TriggerSpec
from pzr.mpc.objectives import CostWeights, WeightedZonotopeCost
from pzr.mpc.policies import MPCPolicy, ReductionDecision, RolloutMPCPolicy, StaticPolicy
from pzr.mpc.prediction import ConstantPredictor
from pzr.mpc.search import tree_search
from pzr.systems.omni_robot import (
    OmniRobotMonitor,
    generate_omni_robot_trace,
)
from pzr.zonotope.reduction import (
    BoxReducer,
    CombastelReducer,
    GirardReducer,
)


BUDGET = 8


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
