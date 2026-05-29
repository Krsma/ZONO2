"""Tests for zonotope core and metrics."""

import numpy as np
import pytest

from pzr.zonotope.core import Zonotope
from pzr.zonotope.metrics import containment_check, interval_hull_mse, width_inflation


class TestZonotopeConstruction:
    def test_center_only(self):
        z = Zonotope([1.0, 2.0, 3.0])
        assert z.dimension == 3
        assert z.generator_count == 0
        assert z.order == 0.0

    def test_center_and_generators(self):
        z = Zonotope([0.0, 0.0], [[1.0, 0.5], [0.0, 1.0]])
        assert z.dimension == 2
        assert z.generator_count == 2
        assert z.order == 1.0

    def test_immutable_center(self):
        z = Zonotope([1.0, 2.0])
        with pytest.raises(ValueError):
            z.center[0] = 99.0

    def test_immutable_generators(self):
        z = Zonotope([0.0], [[1.0, 2.0]])
        with pytest.raises(ValueError):
            z.generators[0, 0] = 99.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            Zonotope([1.0, 2.0], [[1.0], [2.0], [3.0]])

    def test_input_arrays_not_aliased(self):
        c = np.array([1.0, 2.0])
        g = np.array([[1.0], [0.0]])
        z = Zonotope(c, g)
        c[0] = 99.0
        g[0, 0] = 99.0
        assert z.center[0] == 1.0
        assert z.generators[0, 0] == 1.0


class TestZonotopeProperties:
    def test_interval_bounds(self):
        z = Zonotope([1.0], [[0.5, 0.3]])
        lo, hi = z.interval_bounds()
        np.testing.assert_allclose(lo, [0.2])
        np.testing.assert_allclose(hi, [1.8])

    def test_widths(self):
        z = Zonotope([0.0, 0.0], [[1.0, 0.0], [0.0, 2.0]])
        np.testing.assert_allclose(z.widths(), [2.0, 4.0])

    def test_interval_radius(self):
        z = Zonotope([0.0], [[-0.5, 0.3]])
        np.testing.assert_allclose(z.interval_radius(), [0.8])

    def test_volume_proxy(self):
        z = Zonotope([0.0, 0.0], np.eye(2))
        assert z.volume_proxy() == pytest.approx(4.0)

    def test_empty_generators(self):
        z = Zonotope([1.0, 2.0])
        lo, hi = z.interval_bounds()
        np.testing.assert_allclose(lo, [1.0, 2.0])
        np.testing.assert_allclose(hi, [1.0, 2.0])
        np.testing.assert_allclose(z.widths(), [0.0, 0.0])


class TestZonotopeOperations:
    def test_affine_map(self):
        z = Zonotope([1.0, 0.0], [[1.0], [0.0]])
        a = np.array([[2.0, 0.0], [0.0, 1.0]])
        mapped = z.affine_map(a, bias=[1.0, 1.0])
        np.testing.assert_allclose(mapped.center, [3.0, 1.0])
        np.testing.assert_allclose(mapped.generators, [[2.0], [0.0]])

    def test_affine_map_dimension_check(self):
        z = Zonotope([1.0, 2.0])
        with pytest.raises(ValueError):
            z.affine_map(np.array([[1.0, 2.0, 3.0]]))

    def test_minkowski_sum(self):
        z1 = Zonotope([1.0], [[0.5]])
        z2 = Zonotope([2.0], [[0.3]])
        s = z1.minkowski_sum(z2)
        np.testing.assert_allclose(s.center, [3.0])
        assert s.generator_count == 2
        np.testing.assert_allclose(s.generators, [[0.5, 0.3]])

    def test_append_generators(self):
        z = Zonotope([0.0, 0.0], [[1.0], [0.0]])
        z2 = z.append_generators([[0.0], [1.0]])
        assert z2.generator_count == 2
        assert z.generator_count == 1  # original unchanged

    def test_take_generators(self):
        z = Zonotope([0.0], [[1.0, 2.0, 3.0]])
        z2 = z.take_generators([0, 2])
        assert z2.generator_count == 2
        np.testing.assert_allclose(z2.generators, [[1.0, 3.0]])

    def test_take_empty(self):
        z = Zonotope([0.0], [[1.0, 2.0]])
        z2 = z.take_generators([])
        assert z2.generator_count == 0

    def test_with_center(self):
        z = Zonotope([0.0], [[1.0]])
        z2 = z.with_center([5.0])
        np.testing.assert_allclose(z2.center, [5.0])
        np.testing.assert_allclose(z2.generators, [[1.0]])

    def test_with_generators(self):
        z = Zonotope([1.0], [[0.5]])
        z2 = z.with_generators([[0.3, 0.4]])
        assert z2.generator_count == 2
        np.testing.assert_allclose(z2.center, [1.0])


class TestZonotopeSampling:
    def test_sample_center(self):
        z = Zonotope([1.0, 2.0], [[0.5, 0.0], [0.0, 0.3]])
        point = z.sample([0.0, 0.0])
        np.testing.assert_allclose(point, [1.0, 2.0])

    def test_sample_vertex(self):
        z = Zonotope([0.0], [[1.0, 0.5]])
        point = z.sample([1.0, 1.0])
        np.testing.assert_allclose(point, [1.5])

    def test_sample_wrong_count(self):
        z = Zonotope([0.0], [[1.0, 0.5]])
        with pytest.raises(ValueError):
            z.sample([1.0])

    def test_sample_out_of_range(self):
        z = Zonotope([0.0], [[1.0]])
        with pytest.raises(ValueError):
            z.sample([1.5])

    def test_contains_in_interval_hull(self):
        z = Zonotope([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        assert z.contains_in_interval_hull([0.5, 0.5])
        assert z.contains_in_interval_hull([1.0, 1.0])
        assert not z.contains_in_interval_hull([1.5, 0.0])

    def test_sampled_points_in_hull(self):
        z = Zonotope(
            [0.2, -0.3, 0.5],
            [[1.0, -0.4, 0.2], [0.1, 0.8, -0.5], [-0.2, 0.1, 0.7]],
        )
        rng = np.random.default_rng(123)
        for _ in range(200):
            xi = rng.uniform(-1.0, 1.0, size=z.generator_count)
            point = z.sample(xi)
            assert z.contains_in_interval_hull(point)


class TestMetrics:
    def test_interval_hull_mse_identical(self):
        z = Zonotope([0.0], [[1.0]])
        assert interval_hull_mse(z, z) == pytest.approx(0.0)

    def test_interval_hull_mse_different(self):
        z1 = Zonotope([0.0], [[1.0]])
        z2 = Zonotope([0.0], [[2.0]])
        mse = interval_hull_mse(z1, z2)
        assert mse > 0.0

    def test_width_inflation_identity(self):
        z = Zonotope([0.0, 0.0], np.eye(2))
        infl = width_inflation(z, z)
        np.testing.assert_allclose(infl, [1.0, 1.0])

    def test_containment_check_self(self):
        z = Zonotope([0.0, 0.0], [[1.0, 0.5], [0.3, 1.0]])
        assert containment_check(z, z)

    def test_containment_check_overapprox(self):
        z_small = Zonotope([0.0], [[0.5]])
        z_big = Zonotope([0.0], [[1.0]])
        assert containment_check(z_small, z_big)

    def test_containment_check_fails_underapprox(self):
        z_big = Zonotope([0.0], [[1.0]])
        z_small = Zonotope([0.0], [[0.3]])
        assert not containment_check(z_big, z_small)


class TestRepr:
    def test_repr(self):
        z = Zonotope([0.0, 0.0], np.eye(2))
        assert "dim=2" in repr(z)
        assert "generators=2" in repr(z)
