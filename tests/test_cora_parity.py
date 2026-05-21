import importlib.util
import json
from pathlib import Path

import numpy as np

from pzr.core.zonotope import Zonotope


def _load_tool():
    path = Path("tools/validate_cora_reducers.py")
    spec = importlib.util.spec_from_file_location("validate_cora_reducers", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cora_fixture_matches_python_reducer_outputs() -> None:
    tool = _load_tool()
    fixture = json.loads(Path("tests/fixtures/cora_reducers_reference.json").read_text())
    zonotope = Zonotope(fixture["center"], fixture["generators"])

    for method, expected in fixture["methods"].items():
        actual = tool.python_reduce(zonotope, method, fixture["budget"])
        comparison = tool.compare_outputs(method, actual, expected, "fixture")
        assert comparison["matched"], comparison
        assert actual["generator_count"] <= fixture["budget"]


def test_cora_budget_translation_uses_order_not_absolute_budget() -> None:
    tool = _load_tool()

    assert np.isclose(tool.cora_order(4, 3), 4.0 / 3.0)
