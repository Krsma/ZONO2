from types import SimpleNamespace

import numpy as np
import pytest

from pzr.rtlola.actions import RtlolaAction
from pzr.rtlola.engine import RtlolaEngine, RtlolaEvent, RtlolaStateRef
from pzr.rtlola.metrics import (
    active_generator_count,
    generator_count,
    matrix_metrics,
)
from pzr.rtlola.omni import OMNI_SPEC, generate_omni_events
from pzr.rtlola.robot_arm import (
    ARM_SPEC,
    DEFAULT_TRACE_KIND,
    generate_robot_arm_events,
    validate_trace_tcp_against_fk,
)
from pzr.rtlola.scenarios import scenario_by_name
from pzr.rtlola.search import beam_search


def test_matrix_metrics_distinguish_dense_active_and_constant_generators():
    dynamic = np.array([
        [1.0, 0.5, -0.25, 0.0],
        [2.0, 0.0, 0.75, 0.0],
    ])
    total = np.column_stack([dynamic, np.array([0.1, 0.2])])

    metrics = matrix_metrics(dynamic, total)

    assert generator_count(dynamic) == 3
    assert active_generator_count(dynamic) == 2
    assert metrics.dynamic_generator_count == 3
    assert metrics.active_dynamic_generator_count == 2
    assert metrics.zero_dynamic_generator_count == 1
    assert metrics.total_generator_count == 4
    assert metrics.full_width_sum == pytest.approx(3.0)


def test_packaged_specs_and_registered_scenarios_are_authoritative():
    assert "constant delta: Variable" in OMNI_SPEC
    assert "constant a5H: Variable" in ARM_SPEC
    assert scenario_by_name("omni_robot").spec == OMNI_SPEC
    assert scenario_by_name("robot_arm").spec == ARM_SPEC


def test_omni_trace_is_seeded_and_deterministic():
    left = generate_omni_events(5, seed=42)
    right = generate_omni_events(5, seed=42)

    assert left == right
    assert left[0].time == 0.0
    assert all(a.time < b.time for a, b in zip(left, left[1:]))


def test_robot_arm_trace_matches_packaged_forward_kinematics():
    events = generate_robot_arm_events(4, trace_kind=DEFAULT_TRACE_KIND)

    assert len(events) == 4
    assert len(events[0].values) == 6
    assert validate_trace_tcp_against_fk(
        DEFAULT_TRACE_KIND,
        max_rows=4,
    ) < 3e-4


def test_beam_search_supports_forced_root_with_full_continuation_pool():
    first = RtlolaAction("first", lambda _budget: object(), explicit_budget=False)
    future = RtlolaAction("future", lambda _budget: object(), explicit_budget=False)

    class FakeEngine:
        def metrics(self, state):
            return SimpleNamespace(dynamic_generator_count=99, dimension=1)

        def branch_step(self, state, event, action, config_budget):
            del event, config_budget
            depth = state.depth + 1
            cost = 5.0 if action.name == "first" else 1.0
            return SimpleNamespace(
                verdict={},
                state=SimpleNamespace(depth=depth),
                action_name=action.name,
                metrics=SimpleNamespace(full_width_sum=cost),
            )

    result = beam_search(
        FakeEngine(),
        SimpleNamespace(depth=0),
        object(),
        (object(),),
        (first, future),
        budget=10,
        beam_width=2,
        fallback=future,
        forced_first_action=first,
        cost_fn=lambda _engine, step: step.metrics.full_width_sum,
    )

    assert result.predicted_sequence == ("first", "future")
    assert result.predicted_cost == pytest.approx(1.0)


def test_engine_wraps_binding_panics_so_search_can_fallback():
    class BindingPanic(BaseException):
        pass

    class PanickingPlanner:
        def accept_event_from_state(self, *args):
            raise BindingPanic("native transform panic")

    engine = RtlolaEngine.__new__(RtlolaEngine)
    engine.spec_id = "spec"
    engine.event_arity = 1
    engine.planner = PanickingPlanner()
    state = RtlolaStateRef(object(), "spec", 0, 0.0)
    action = RtlolaAction("panic", lambda _budget: object(), explicit_budget=False)

    with pytest.raises(RuntimeError, match="planner branch failed") as captured:
        engine.branch_step(
            state,
            RtlolaEvent(1.0, (1.0,)),
            action,
            budget=0,
        )

    assert isinstance(captured.value.__cause__, BindingPanic)
