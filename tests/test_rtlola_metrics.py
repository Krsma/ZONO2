import numpy as np
import pytest

from pzr.rtlola.metrics import generator_count, matrix_metrics
from pzr.rtlola.omni import OMNI_EXPECTED_VERDICT_KEYS, OMNI_SPEC, generate_omni_events


def test_matrix_metrics_count_dynamic_and_total_generators():
    dynamic = np.array([
        [1.0, 0.5, -0.25],
        [2.0, 0.0, 0.75],
    ])
    total = np.array([
        [1.0, 0.5, -0.25, 0.1],
        [2.0, 0.0, 0.75, 0.2],
    ])

    metrics = matrix_metrics(dynamic, total)

    assert generator_count(dynamic) == 2
    assert metrics.dynamic_generator_count == 2
    assert metrics.total_generator_count == 3
    assert metrics.dimension == 2
    assert metrics.full_width_sum == pytest.approx(3.0)
    assert metrics.cost() > metrics.full_width_sum


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
