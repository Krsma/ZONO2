# Repository Guidelines

## Project Structure & Module Organization

This is a small Python 3.11 research package for predictive zonotope reduction.
Source code lives under `src/pzr/`:

- `core/`: zonotope primitives and certificates.
- `reduction/`: reducer interfaces, scoring, and baseline reducers.
- `monitoring/`: monitor adapter boundary.
- `control/`: static and receding-horizon policies.
- `benchmarks/`: benchmark systems, currently the robot monitor.
- `experiments/`: CLI, scenarios, and benchmark orchestration.

Tests are in `tests/` and follow the same domain split as the package. Research notes are in `science/SCIENCE.md`. Generated benchmark outputs are written under `results/`, for example `results/robot/`.

## Build, Test, and Development Commands

- `python -m pip install -e ".[dev]"`: install the package in editable mode with pytest.
- `pytest`: run the full test suite configured by `pyproject.toml`.
- `pzr-benchmark robot --length 200 --budget 8 --horizon 4 --seeds 30 --out results/robot`: run the default paper-style robot benchmark and write CSV/JSON artifacts.
- `python -m pzr.experiments.cli robot --quiet --seeds 1`: useful when testing the CLI without relying on the console script.

## Coding Style & Naming Conventions

Use 4-space indentation, Python type hints, and small dataclasses or immutable value objects where appropriate. Keep modules and functions in `snake_case`; classes and enums use `PascalCase`; constants use `UPPER_SNAKE_CASE`. Prefer explicit `numpy` array conversions and validation at API boundaries, matching `src/pzr/core/zonotope.py`. Add short docstrings for public modules and domain objects. No formatter or linter is configured, so preserve the existing style and keep imports grouped as standard library, third-party, then local.

## Testing Guidelines

Tests use `pytest` and live in files named `tests/test_*.py`. Name tests by behavior, such as `test_box_reducer_contains_sampled_original_points`. Use `tmp_path` for generated artifacts and `numpy.testing` for numeric comparisons. New reducers, policies, or benchmark outputs should include soundness, budget, and artifact-shape assertions where relevant.

## Commit & Pull Request Guidelines

This checkout has no usable Git history to infer project-specific commit conventions. Use concise, imperative commit messages, for example `Add calibration-aware reducer test`. Pull requests should describe the research or behavior change, list commands run, mention any changed benchmark outputs, and link related notes or issues. Include screenshots only for notebook or report-rendering changes.

## Agent-Specific Instructions

Do not hand-edit generated files in `results/` unless the task explicitly concerns saved benchmark artifacts. Prefer changing source, tests, or benchmark configuration and regenerating outputs with the CLI.
