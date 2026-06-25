from types import SimpleNamespace

import numpy as np
import pytest

from pzr.rtlola.actions import RtlolaAction
from pzr.rtlola.metrics import (
    active_generator_count,
    generator_count,
    matrix_metrics,
    relevant_row_cost,
)
from pzr.rtlola.omni import OMNI_EXPECTED_VERDICT_KEYS, OMNI_SPEC, generate_omni_events
from pzr.rtlola.robot_arm import (
    ARM_SPEC,
    DEFAULT_TRACE_KIND,
    Q,
    J,
    I,
    H,
    generate_robot_arm_events,
    validate_trace_tcp_against_fk,
)
from pzr.rtlola.search import beam_search


def test_matrix_metrics_count_dynamic_and_total_generators():
    dynamic = np.array([
        [1.0, 0.5, -0.25, 0.0],
        [2.0, 0.0, 0.75, 0.0],
    ])
    total = np.array([
        [1.0, 0.5, -0.25, 0.0, 0.1],
        [2.0, 0.0, 0.75, 0.0, 0.2],
    ])

    metrics = matrix_metrics(dynamic, total)

    assert generator_count(dynamic) == 3
    assert active_generator_count(dynamic) == 2
    assert metrics.dynamic_generator_count == 3
    assert metrics.total_generator_count == 4
    assert metrics.active_dynamic_generator_count == 2
    assert metrics.active_total_generator_count == 3
    assert metrics.zero_dynamic_generator_count == 1
    assert metrics.zero_total_generator_count == 1
    assert metrics.dimension == 2
    assert metrics.full_width_sum == pytest.approx(3.0)
    assert metrics.cost() == pytest.approx(metrics.full_width_sum)


def test_omni_spec_documents_required_uncertainty_and_triggers():
    assert "constant delta: Variable" in OMNI_SPEC
    assert "output epsilon: Variable @true" in OMNI_SPEC
    for key in OMNI_EXPECTED_VERDICT_KEYS:
        assert key in OMNI_SPEC


def test_omni_event_conversion_matches_existing_trace_shape():
    events = generate_omni_events(3, seed=0)
    assert len(events) == 3
    assert events[0].time == 0.0
    assert len(events[0].values) == 3
    assert events[0].values[0] == events[0].time
    assert events[1].time > events[0].time


def test_robot_arm_spec_uses_float64_uncertainty_constants_and_public_streams():
    assert "Fixed64_32" not in ARM_SPEC
    assert ".abs()" not in ARM_SPEC
    for name, value in {"Q": Q, "J": J, "I": I, "H": H}.items():
        assert f"constant {name}: Float64" in ARM_SPEC
        assert f"{value:.6f}" in ARM_SPEC
    assert "#[public]\noutput dist_to_expected" in ARM_SPEC
    assert "#[public]\noutput tpl" in ARM_SPEC


def test_robot_arm_trace_kind_loads_default_events_and_matches_fk():
    events = generate_robot_arm_events(4, trace_kind=DEFAULT_TRACE_KIND)
    assert len(events) == 4
    assert len(events[0].values) == 6
    assert events[0].values[0] == events[0].time
    assert events[1].time > events[0].time
    assert validate_trace_tcp_against_fk(DEFAULT_TRACE_KIND, max_rows=4) < 3e-4


def test_relevant_row_cost_uses_only_requested_rows():
    matrix = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0],
        [0.0, 4.0, 8.0],
    ])

    cost = relevant_row_cost(matrix, rows=(0, 1))

    assert cost == pytest.approx(2.0 + 4.0)


def test_relevant_row_cost_ignores_generator_count():
    narrow_many = np.array([
        [0.0, 0.25, 0.25, 0.25, 0.25],
    ])
    wide_few = np.array([
        [0.0, 1.1],
    ])

    assert relevant_row_cost(narrow_many, rows=(0,)) < relevant_row_cost(
        wide_few, rows=(0,),
    )


def test_rtlola_beam_search_uses_terminal_cost_not_cumulative_sum():
    action = RtlolaAction("reduce", lambda _budget: object(), explicit_budget=False)

    class FakeEngine:
        def metrics(self, state):
            _ = state
            return SimpleNamespace(dynamic_generator_count=99, dimension=1)

        def branch_step(self, state, event, action, config_budget):
            _ = event, action, config_budget
            depth = state.depth + 1
            width = {1: 10.0, 2: 3.0}[depth]
            return SimpleNamespace(
                verdict={},
                state=SimpleNamespace(depth=depth),
                action_name="reduce",
                metrics=SimpleNamespace(full_width_sum=width),
            )

    result = beam_search(
        FakeEngine(),
        SimpleNamespace(depth=0),
        object(),
        (object(),),
        (action,),
        budget=10,
        beam_width=1,
        fallback=action,
        cost_fn=lambda _engine, step: step.metrics.full_width_sum,
    )

    assert result.predicted_sequence == ("reduce", "reduce")
    assert result.predicted_cost == pytest.approx(3.0)
