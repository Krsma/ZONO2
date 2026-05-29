"""Tests for zonotope reduction methods.

Validates:
1. Soundness: sampled points from original lie in interval hull of reduced
2. Budget compliance: reduced generator count <= budget
3. CORA parity: results match CORA MATLAB reference fixture
4. Protected generators survive exactly
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pzr.zonotope.core import Zonotope
from pzr.zonotope.metrics import containment_check
from pzr.zonotope.reduction import (
    ALL_REDUCERS,
    BoxReducer,
    CombastelReducer,
    GirardReducer,
    IdentityReducer,
    MethAReducer,
    PcaReducer,
    ReductionResult,
    ScottReducer,
)
from pzr.zonotope.protected import ProtectedReducer


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cora_reference.json"


@pytest.fixture
def reference_fixture():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture
def reference_zonotope(reference_fixture):
    return Zonotope(
        center=reference_fixture["center"],
        generators=reference_fixture["generators"],
    )


@pytest.fixture
def reference_budget(reference_fixture):
    return reference_fixture["budget"]


# -----------------------------------------------------------------------
# Soundness: sample-based containment check
# -----------------------------------------------------------------------

class TestSoundness:
    """Every reducer must produce a result whose interval hull contains the original."""

    @pytest.mark.parametrize("reducer_name", ["box", "girard", "combastel", "pca", "methA", "scott"])
    def test_containment(self, reducer_name, reference_zonotope, reference_budget):
        reducer = ALL_REDUCERS[reducer_name]
        result = reducer.reduce(reference_zonotope, reference_budget)
        assert result.certificate.is_sound
        assert containment_check(reference_zonotope, result.reduced, n_samples=500)

    @pytest.mark.parametrize("reducer_name", ["box", "girard", "combastel", "pca", "methA", "scott"])
    def test_budget_compliance(self, reducer_name, reference_zonotope, reference_budget):
        reducer = ALL_REDUCERS[reducer_name]
        result = reducer.reduce(reference_zonotope, reference_budget)
        assert result.reduced.generator_count <= reference_budget

    @pytest.mark.parametrize("reducer_name", ["box", "girard", "combastel", "pca", "methA", "scott"])
    def test_center_preserved(self, reducer_name, reference_zonotope, reference_budget):
        reducer = ALL_REDUCERS[reducer_name]
        result = reducer.reduce(reference_zonotope, reference_budget)
        np.testing.assert_allclose(result.reduced.center, reference_zonotope.center)

    def test_no_reduction_needed(self):
        z = Zonotope([0.0, 0.0], np.eye(2))
        result = GirardReducer().reduce(z, budget=5)
        assert result.reduced is z

    def test_larger_zonotope_containment(self):
        rng = np.random.default_rng(42)
        z = Zonotope(rng.standard_normal(5), rng.standard_normal((5, 12)))
        for reducer in ALL_REDUCERS.values():
            if reducer.name == "identity":
                continue
            result = reducer.reduce(z, budget=6)
            assert result.certificate.is_sound
            assert result.reduced.generator_count <= 6
            assert containment_check(z, result.reduced, n_samples=200)


# -----------------------------------------------------------------------
# CORA parity
# -----------------------------------------------------------------------

class TestCORAParity:
    """Verify that our reducers match CORA MATLAB reference outputs."""

    @pytest.mark.parametrize("method", ["girard", "combastel", "pca", "methA", "scott"])
    def test_interval_bounds_match(self, method, reference_fixture, reference_zonotope, reference_budget):
        ref = reference_fixture["methods"][method]
        reducer = ALL_REDUCERS[method]
        result = reducer.reduce(reference_zonotope, reference_budget)

        lo, hi = result.reduced.interval_bounds()
        np.testing.assert_allclose(lo, ref["lower"], atol=1e-10)
        np.testing.assert_allclose(hi, ref["upper"], atol=1e-10)

    @pytest.mark.parametrize("method", ["girard", "combastel", "pca", "methA", "scott"])
    def test_widths_match(self, method, reference_fixture, reference_zonotope, reference_budget):
        ref = reference_fixture["methods"][method]
        reducer = ALL_REDUCERS[method]
        result = reducer.reduce(reference_zonotope, reference_budget)

        np.testing.assert_allclose(result.reduced.widths(), ref["widths"], atol=1e-10)

    @pytest.mark.parametrize("method", ["girard", "combastel", "pca", "methA", "scott"])
    def test_generator_count_match(self, method, reference_fixture, reference_zonotope, reference_budget):
        ref = reference_fixture["methods"][method]
        reducer = ALL_REDUCERS[method]
        result = reducer.reduce(reference_zonotope, reference_budget)

        assert result.reduced.generator_count == ref["generator_count"]

    @pytest.mark.parametrize("method", ["girard", "combastel", "pca", "methA", "scott"])
    def test_center_match(self, method, reference_fixture, reference_zonotope, reference_budget):
        ref = reference_fixture["methods"][method]
        reducer = ALL_REDUCERS[method]
        result = reducer.reduce(reference_zonotope, reference_budget)

        np.testing.assert_allclose(result.reduced.center, ref["center"], atol=1e-10)


# -----------------------------------------------------------------------
# Box reducer specifics
# -----------------------------------------------------------------------

class TestBoxReducer:
    def test_produces_diagonal(self, reference_zonotope, reference_budget):
        result = BoxReducer().reduce(reference_zonotope, reference_budget)
        G = result.reduced.generators
        for col in range(G.shape[1]):
            nonzero = np.count_nonzero(np.abs(G[:, col]) > 1e-12)
            assert nonzero == 1, "box generator should have exactly one nonzero entry"

    def test_interval_hull_matches(self, reference_zonotope, reference_budget):
        result = BoxReducer().reduce(reference_zonotope, reference_budget)
        lo_orig, hi_orig = reference_zonotope.interval_bounds()
        lo_red, hi_red = result.reduced.interval_bounds()
        np.testing.assert_allclose(lo_red, lo_orig, atol=1e-12)
        np.testing.assert_allclose(hi_red, hi_orig, atol=1e-12)

    def test_budget_too_small_raises(self):
        z = Zonotope([0.0, 0.0, 0.0], np.eye(3) * 0.5)
        with pytest.raises(ValueError):
            BoxReducer().reduce(z, budget=1)


# -----------------------------------------------------------------------
# Identity reducer
# -----------------------------------------------------------------------

class TestIdentityReducer:
    def test_within_budget_succeeds(self):
        z = Zonotope([0.0], [[1.0, 0.5]])
        result = IdentityReducer().reduce(z, budget=3)
        assert result.reduced is z

    def test_over_budget_raises(self):
        z = Zonotope([0.0], [[1.0, 0.5, 0.3]])
        with pytest.raises(ValueError):
            IdentityReducer().reduce(z, budget=2)


# -----------------------------------------------------------------------
# Protected reducer
# -----------------------------------------------------------------------

class TestProtectedReducer:
    def test_protected_generators_survive(self):
        z = Zonotope(
            [0.0, 0.0],
            [[1.0, 0.5, 0.3, 0.1, 0.2],
             [0.0, 0.8, 0.2, 0.4, 0.1]],
        )
        protected = ProtectedReducer(base=GirardReducer())
        result = protected.reduce(z, budget=3, protected_indices=(0,))
        assert result.reduced.generator_count <= 3
        assert result.certificate.is_sound
        # Column 0 should be preserved exactly
        np.testing.assert_allclose(result.reduced.generators[:, 0], z.generators[:, 0])
        assert containment_check(z, result.reduced, n_samples=200)

    def test_no_protected_delegates_to_base(self):
        z = Zonotope([0.0], [[1.0, 0.5, 0.3]])
        protected = ProtectedReducer(base=GirardReducer())
        result_protected = protected.reduce(z, budget=2)
        result_base = GirardReducer().reduce(z, budget=2)
        np.testing.assert_allclose(result_protected.reduced.generators, result_base.reduced.generators)

    def test_too_many_protected_raises(self):
        z = Zonotope([0.0, 0.0], np.eye(2) * 0.5)
        protected = ProtectedReducer(base=BoxReducer())
        with pytest.raises(ValueError):
            protected.reduce(z, budget=1, protected_indices=(0, 1))

    def test_multiple_protected(self):
        z = Zonotope(
            [0.0, 0.0, 0.0],
            np.hstack([np.eye(3), np.ones((3, 4)) * 0.1]),
        )
        protected = ProtectedReducer(base=GirardReducer())
        result = protected.reduce(z, budget=5, protected_indices=(0, 1))
        assert result.reduced.generator_count <= 5
        np.testing.assert_allclose(result.reduced.generators[:, 0], z.generators[:, 0])
        np.testing.assert_allclose(result.reduced.generators[:, 1], z.generators[:, 1])
        assert containment_check(z, result.reduced, n_samples=200)


# -----------------------------------------------------------------------
# All reducers: stress test on random zonotopes
# -----------------------------------------------------------------------

class TestStress:
    @pytest.mark.parametrize("dim,gen,budget", [
        (2, 5, 3),
        (3, 8, 4),
        (4, 10, 5),
        (5, 15, 7),
        (3, 6, 4),
    ])
    def test_random_zonotopes(self, dim, gen, budget):
        rng = np.random.default_rng(123)
        z = Zonotope(rng.standard_normal(dim), rng.standard_normal((dim, gen)))
        for name, reducer in ALL_REDUCERS.items():
            if name == "identity":
                continue
            result = reducer.reduce(z, budget)
            assert result.reduced.generator_count <= budget, f"{name} violated budget"
            assert result.certificate.is_sound, f"{name} returned unsound"
            assert containment_check(z, result.reduced, n_samples=100), f"{name} failed containment"
