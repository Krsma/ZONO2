from pzr.control.costs import CostWeights, WeightedZonotopeCost
from pzr.core.zonotope import GeneratorKind, GeneratorMetadata, Zonotope
from pzr.monitoring.base import MonitorState


def test_weighted_zonotope_cost_defaults_ignore_metadata_terms() -> None:
    zonotope = Zonotope(
        [0.0, 0.0],
        [[1.0, 0.5], [0.25, -0.25]],
        (
            GeneratorMetadata(GeneratorKind.MEASUREMENT, "epsilon@1"),
            GeneratorMetadata(GeneratorKind.SYNTHETIC, "box_axis_0"),
        ),
    )
    state = MonitorState(zonotope)

    cost = WeightedZonotopeCost(CostWeights(trigger_width=0.0, generator_count=0.0))

    assert cost(state) == 0.0


def test_weighted_zonotope_cost_rewards_measurements_and_penalizes_synthetic() -> None:
    measurement_state = MonitorState(
        Zonotope(
            [0.0],
            [[1.0, 0.5]],
            (
                GeneratorMetadata(GeneratorKind.MEASUREMENT, "epsilon@1"),
                GeneratorMetadata(GeneratorKind.MEASUREMENT, "epsilon@2"),
            ),
        )
    )
    synthetic_state = MonitorState(
        Zonotope(
            [0.0],
            [[1.0, 0.5]],
            (
                GeneratorMetadata(GeneratorKind.MEASUREMENT, "epsilon@1"),
                GeneratorMetadata(GeneratorKind.SYNTHETIC, "box_axis_0"),
            ),
        )
    )
    cost = WeightedZonotopeCost(
        CostWeights(
            trigger_width=0.0,
            generator_count=0.0,
            synthetic_generator=1.0,
            measurement_generator_reward=1.0,
        )
    )

    assert cost(measurement_state) < cost(synthetic_state)
