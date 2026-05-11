import numpy as np

from pzr.core.zonotope import GeneratorKind, GeneratorMetadata, Zonotope


def test_interval_bounds_and_sampling() -> None:
    zonotope = Zonotope(
        [1.0, -2.0],
        [[2.0, -0.5], [0.25, 1.0]],
    )

    lower, upper = zonotope.interval_bounds()

    np.testing.assert_allclose(lower, [-1.5, -3.25])
    np.testing.assert_allclose(upper, [3.5, -0.75])
    np.testing.assert_allclose(zonotope.sample([1.0, -1.0]), [3.5, -2.75])


def test_affine_map_preserves_generator_metadata() -> None:
    metadata = (
        GeneratorMetadata(GeneratorKind.CALIBRATION, "delta"),
        GeneratorMetadata(GeneratorKind.MEASUREMENT, "epsilon@1"),
    )
    zonotope = Zonotope([1.0, 2.0], [[1.0, 0.0], [0.0, 2.0]], metadata)

    mapped = zonotope.affine_map([[2.0, 0.0], [0.0, -1.0]], [1.0, 0.5])

    np.testing.assert_allclose(mapped.center, [3.0, -1.5])
    np.testing.assert_allclose(mapped.generators, [[2.0, 0.0], [0.0, -2.0]])
    assert mapped.metadata == metadata
