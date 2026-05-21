#!/usr/bin/env python3
"""Validate Python zonotope reducers against saved fixtures and optional CORA."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pzr.core.zonotope import Zonotope
from pzr.reduction.paper_reducers import (
    CombastelReducer,
    GirardReducer,
    MethAReducer,
    PcaReducer,
    ScottReducer,
)

PYTHON_REDUCERS = {
    "girard": GirardReducer,
    "combastel": CombastelReducer,
    "pca": PcaReducer,
    "methA": MethAReducer,
    "scott": ScottReducer,
}

DEFAULT_METHODS = ("girard", "combastel", "pca", "methA", "scott", "sadraddini")


def main(argv: Sequence[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    zonotope = Zonotope(fixture["center"], fixture["generators"])
    budget = int(args.budget or fixture["budget"])
    methods = tuple(args.methods or DEFAULT_METHODS)

    rows: list[dict[str, Any]] = []
    python_outputs: dict[str, Any] = {}
    cora_outputs: dict[str, Any] = {}
    for method in methods:
        if method in PYTHON_REDUCERS:
            python_outputs[method] = python_reduce(zonotope, method, budget)
            if method in fixture["methods"]:
                rows.append(compare_outputs(method, python_outputs[method], fixture["methods"][method], "fixture"))
        else:
            rows.append(
                {
                    "method": method,
                    "target": "python",
                    "python_available": False,
                    "matched": False,
                    "note": "No Python reducer is registered for this optional CORA method.",
                }
            )

    if args.cora_root is not None:
        cora_outputs = run_cora(
            args.cora_root,
            zonotope,
            methods,
            budget,
            runner=args.runner,
        )
        for method, output in cora_outputs.items():
            if method in python_outputs:
                rows.append(compare_outputs(method, python_outputs[method], output, "cora"))

    report = {
        "schema": "pzr_cora_validation_v1",
        "budget": budget,
        "dimension": zonotope.dimension,
        "cora_order": cora_order(budget, zonotope.dimension),
        "methods": list(methods),
        "comparisons": rows,
        "python_outputs": python_outputs,
        "cora_outputs": cora_outputs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
    failed = [row for row in rows if row.get("matched") is False and row.get("python_available", True)]
    return 1 if failed and args.fail_on_mismatch else 0


def python_reduce(zonotope: Zonotope, method: str, budget: int) -> dict[str, Any]:
    reducer = PYTHON_REDUCERS[method]()
    result = reducer.reduce(zonotope, budget)
    lower, upper = result.reduced.interval_bounds()
    widths = result.reduced.widths()
    return {
        "center": result.reduced.center.tolist(),
        "generators": result.reduced.generators.tolist(),
        "generator_count": result.reduced.generator_count,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "widths": widths.tolist(),
        "volume_proxy": float(np.prod(np.maximum(widths, 0.0))),
    }


def compare_outputs(
    method: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
    target: str,
    *,
    atol: float = 1e-9,
) -> dict[str, Any]:
    width_delta = _max_abs_delta(actual["widths"], expected["widths"])
    lower_delta = _max_abs_delta(actual["lower"], expected["lower"])
    upper_delta = _max_abs_delta(actual["upper"], expected["upper"])
    volume_delta = abs(float(actual["volume_proxy"]) - float(expected["volume_proxy"]))
    generator_count_match = int(actual["generator_count"]) == int(expected["generator_count"])
    matched = (
        generator_count_match
        and width_delta <= atol
        and lower_delta <= atol
        and upper_delta <= atol
        and volume_delta <= atol
    )
    return {
        "method": method,
        "target": target,
        "matched": matched,
        "generator_count_match": generator_count_match,
        "max_width_delta": width_delta,
        "max_lower_delta": lower_delta,
        "max_upper_delta": upper_delta,
        "volume_proxy_delta": volume_delta,
    }


def cora_order(budget: int, dimension: int) -> float:
    """Translate this repo's absolute generator budget to CORA order."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return float(budget) / float(dimension)


def run_cora(
    cora_root: Path,
    zonotope: Zonotope,
    methods: tuple[str, ...],
    budget: int,
    *,
    runner: str,
) -> dict[str, Any]:
    """Call CORA through MATLAB or Octave and parse JSON reducer outputs."""

    root = cora_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"CORA root does not exist: {root}")
    script = _matlab_script(root, zonotope, methods, cora_order(budget, zonotope.dimension))
    with tempfile.TemporaryDirectory(prefix="pzr-cora-") as tmp:
        script_path = Path(tmp) / "validate_cora_reducers.m"
        out_path = Path(tmp) / "cora_outputs.json"
        script_path.write_text(script.replace("__OUT_PATH__", str(out_path)), encoding="utf-8")
        if runner == "matlab":
            command = [
                "matlab",
                "-batch",
                f"run('{script_path.as_posix()}')",
            ]
        else:
            command = ["octave", "--quiet", "--no-gui", script_path.as_posix()]
        subprocess.run(command, check=True)
        return json.loads(out_path.read_text(encoding="utf-8"))


def _matlab_script(
    cora_root: Path,
    zonotope: Zonotope,
    methods: tuple[str, ...],
    order: float,
) -> str:
    center_matrix = _matlab_matrix(zonotope.center.reshape(-1, 1))
    generator_matrix = _matlab_matrix(zonotope.generators)
    method_cells = "{" + ",".join(f"'{method}'" for method in methods) + "}"
    return f"""
addpath(genpath('{cora_root.as_posix()}'));
Z = zonotope({center_matrix}, {generator_matrix});
methods = {method_cells};
out = struct();
for i = 1:numel(methods)
    method = methods{{i}};
    try
        Zred = reduce(Z, method, {order:.17g});
        c = center(Zred);
        G = generators(Zred);
        widths = 2 * sum(abs(G), 2);
        entry = struct();
        entry.center = c(:)';
        entry.generators = G;
        entry.generator_count = size(G, 2);
        entry.lower = (c(:) - sum(abs(G), 2))';
        entry.upper = (c(:) + sum(abs(G), 2))';
        entry.widths = widths(:)';
        entry.volume_proxy = prod(widths);
        out.(method) = entry;
    catch err
        entry = struct();
        entry.error = err.message;
        out.(method) = entry;
    end
end
fid = fopen('__OUT_PATH__', 'w');
fprintf(fid, '%s', jsonencode(out));
fclose(fid);
"""


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Python CORA-style reducers against fixtures and optional CORA outputs.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/cora_reducers_reference.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/cora-validation/report.json"))
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--cora-root", type=Path, default=None)
    parser.add_argument("--runner", choices=("matlab", "octave"), default="matlab")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser


def _matlab_matrix(array: np.ndarray) -> str:
    rows = []
    for row in np.asarray(array, dtype=float):
        rows.append(" ".join(f"{float(value):.17g}" for value in row))
    return "[" + "; ".join(rows) + "]"


def _max_abs_delta(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
