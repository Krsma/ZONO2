import numpy as np

import pytest

from pzr.core.zonotope import (
    GeneratorKind,
    GeneratorMetadata,
    GeneratorRequirement,
    Zonotope,
)
from pzr.reduction.paper_reducers import (
    AdaptiveReducer,
    CombastelReducer,
    GirardReducer,
    MethAReducer,
    PcaReducer,
    ScottReducer,
)
from pzr.reduction.base import ReductionContext
from pzr.reduction.reducers import BoxReducer, ProtectedReducer, ScoredKeepReducer


def _sample_points(zonotope: Zonotope) -> list[np.ndarray]:
    samples = [
        np.zeros(zonotope.generator_count),
        np.ones(zonotope.generator_count),
        -np.ones(zonotope.generator_count),
    ]
    rng = np.random.default_rng(1)
    samples.extend(rng.uniform(-1.0, 1.0, zonotope.generator_count) for _ in range(10))
    return [zonotope.sample(sample) for sample in samples]


def test_box_reducer_contains_sampled_original_points() -> None:
    zonotope = Zonotope([0.0, 1.0], [[1.0, -0.5, 0.25], [0.2, 1.0, -0.4]])

    result = BoxReducer().reduce(zonotope, budget=2)

    assert result.certificate.is_sound
    assert result.reduced.generator_count <= 2
    for point in _sample_points(zonotope):
        assert result.reduced.contains_in_interval_hull(point)


def test_scored_keep_reducer_respects_budget_and_preserves_calibration() -> None:
    metadata = (
        GeneratorMetadata(GeneratorKind.CALIBRATION, "delta"),
        GeneratorMetadata(GeneratorKind.MEASUREMENT, "epsilon@1"),
        GeneratorMetadata(GeneratorKind.MEASUREMENT, "epsilon@2"),
        GeneratorMetadata(GeneratorKind.MEASUREMENT, "epsilon@3"),
    )
    zonotope = Zonotope(
        [0.0, 0.0],
        [[0.1, 2.0, 0.4, -0.2], [0.1, 0.1, 1.0, 0.7]],
        metadata,
    )

    result = ScoredKeepReducer.calibration_aware().reduce(
        zonotope,
        budget=3,
        context=ReductionContext(preserve_calibration=True),
    )

    assert result.certificate.is_sound
    assert result.reduced.generator_count <= 3
    assert any(meta.kind == GeneratorKind.CALIBRATION for meta in result.reduced.metadata)
    for point in _sample_points(zonotope):
        assert result.reduced.contains_in_interval_hull(point)


def test_protected_reducer_preserves_required_generator_metadata() -> None:
    metadata = (
        GeneratorMetadata(GeneratorKind.CALIBRATION, "delta"),
        GeneratorMetadata(GeneratorKind.MEASUREMENT, "epsilon@1"),
        GeneratorMetadata(GeneratorKind.MEASUREMENT, "epsilon@2"),
        GeneratorMetadata(GeneratorKind.MEASUREMENT, "epsilon@3"),
    )
    zonotope = Zonotope(
        [0.0, 0.0],
        [[0.1, 2.0, 0.4, -0.2], [0.1, 0.1, 1.0, 0.7]],
        metadata,
    )
    context = ReductionContext(
        required_generators=(
            GeneratorRequirement(GeneratorKind.CALIBRATION, "delta"),
        )
    )

    result = ProtectedReducer(BoxReducer()).reduce(zonotope, budget=3, context=context)

    assert result.certificate.is_sound
    assert result.reduced.generator_count <= 3
    assert any(meta.source == "delta" for meta in result.reduced.metadata)
    for point in _sample_points(zonotope):
        assert result.reduced.contains_in_interval_hull(point)


def test_protected_reducer_fails_when_required_generators_exceed_budget() -> None:
    metadata = (
        GeneratorMetadata(GeneratorKind.CALIBRATION, "delta_x"),
        GeneratorMetadata(GeneratorKind.CALIBRATION, "delta_y"),
    )
    zonotope = Zonotope([0.0], [[0.1, 0.2]], metadata)
    context = ReductionContext(
        required_generators=(GeneratorRequirement(GeneratorKind.CALIBRATION),)
    )

    with pytest.raises(ValueError, match="cannot preserve"):
        ProtectedReducer(BoxReducer()).reduce(zonotope, budget=1, context=context)


def test_paper_reducers_respect_budget_and_contain_samples() -> None:
    zonotope = Zonotope(
        [0.0, 1.0, -0.5],
        [
            [1.0, -0.5, 0.25, 0.2, -0.1, 0.4],
            [0.2, 1.0, -0.4, 0.3, 0.5, -0.2],
            [0.3, -0.1, 0.7, 0.6, -0.4, 0.1],
        ],
    )
    reducers = (
        GirardReducer(),
        CombastelReducer(),
        MethAReducer(),
        ScottReducer(),
        PcaReducer(),
        AdaptiveReducer(),
    )

    for reducer in reducers:
        result = reducer.reduce(zonotope, budget=4)
        assert result.certificate.is_sound
        assert result.reduced.generator_count <= 4
        for point in _sample_points(zonotope):
            assert result.reduced.contains_in_interval_hull(point)
