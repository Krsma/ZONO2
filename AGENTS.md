# Repository Guidelines

## Project Structure & Module Organization

This is a small Python 3.11 research package for predictive zonotope reduction.
Source code lives under `src/pzr/`:

- `core/`: zonotope primitives and certificates.
- `reduction/`: reducer interfaces, scoring, and baseline reducers.
- `monitoring/`: monitor adapter boundary.
- `control/`: static and receding-horizon policies.
- `benchmarks/`: benchmark systems, currently the omnidirectional robot, simple robot, and thermostat monitors.
- `experiments/`: CLI, scenarios, benchmark orchestration, suite aggregation, diagnostics, and paper-style figures.
- `learning/`: policy distillation utilities trained from benchmark decision features.

Tests are in `tests/` and follow the same domain split as the package. Research notes are in `science/SCIENCE.md`. Generated benchmark outputs are written under `results/`, for example `results/robot/`.

## Build, Test, and Development Commands

- `python -m pip install -e ".[dev]"`: install the package in editable mode with pytest.
- `pytest`: run the full test suite configured by `pyproject.toml`.
- `pzr-benchmark robot --length 200 --budget 8 --horizon 4 --seeds 30 --out results/robot`: run the default paper-style robot benchmark and write CSV/JSON artifacts.
- `pzr-benchmark robot --length 200 --budget 8 --horizon 4 --seeds 30 --predictor-mode both --out results/robot`: run both online and oracle predictor modes for ablation.
- `pzr-benchmark robot_simple --length 200 --budget 8 --horizon 4 --seeds 30 --method-set paper_plus_ours --out results/robot-simple`: run the two-axis motivating robot with paper baselines plus the focused rollout MPC method.
- `pzr-benchmark thermostat --length 200 --budget 8 --horizon 4 --seeds 30 --out results/thermostat`: run the non-robot thermostat benchmark.
- `pzr-paper-figures --method-set paper_plus_wide --seeds 10 --length 200 --budget 8 --out results/paper-figures`: regenerate Figures 3--5 style CSV summaries and PNG/PDF plots.
- `pzr-run-experiments --profile smoke --out results/experiment-suite-smoke --force`: run the full suite, learned-policy distillation, aggregate diagnostics, and figure smoke path.
- `python -m pzr.experiments.cli robot --quiet --length 8 --budget 6 --horizon 2 --seeds 1 --bootstrap-samples 20 --out /tmp/pzr-smoke`: useful when smoke-testing the CLI without relying on the console script or writing into `results/`.

## Coding Style & Naming Conventions

Use 4-space indentation, Python type hints, and small dataclasses or immutable value objects where appropriate. Keep modules and functions in `snake_case`; classes and enums use `PascalCase`; constants use `UPPER_SNAKE_CASE`. Prefer explicit `numpy` array conversions and validation at API boundaries, matching `src/pzr/core/zonotope.py`. Add short docstrings for public modules and domain objects. No formatter or linter is configured, so preserve the existing style and keep imports grouped as standard library, third-party, then local.

## Testing Guidelines

Tests use `pytest` and live in files named `tests/test_*.py`. Name tests by behavior, such as `test_box_reducer_contains_sampled_original_points`. Use `tmp_path` for generated artifacts and `numpy.testing` for numeric comparisons. New reducers, policies, or benchmark outputs should include soundness, budget, and artifact-shape assertions where relevant.

MPC-related tests should also assert metadata preservation, certified budgeted
states, chosen reducer accounting, predictor-mode coverage, and
`evaluated_sequence_count` / `pruned_sequence_count` behavior when sequence or
rollout search changes. Normal paper runs should keep `no_op_count` and
`chosen_no_reduction_count` at zero; explicit no-op selection is deferred
future work even though `IdentityReducer` remains available.

Figure or artifact pipeline changes should smoke-test `pzr-paper-figures` with
small lengths/seeds and assert that plotting CSVs plus PNG/PDF outputs are
non-empty. Diagnostic output changes should also check
`selection_summary.csv`, `predicted_sequence_summary.csv`, fallback-box usage
tables/plots, and `analysis_notes.json` where those artifacts are produced.

## Benchmark Methods & Artifacts

The default benchmark suite currently includes optional `reference`, static
reducers (`box`, `girard`, `girard7`, `combastel`, `methA`, `scott`, `pca`,
`adaptive`, `keep_norm`, `keep_calibration_aware`), and four MPC methods built
from three selector styles:

- `mpc`: chooses one reducer and reuses it across predicted horizon overflows.
- `mpc_sequence`: searches reducer choices at each predicted overflow.
- `mpc_rollout_girard`: searches focused first reductions, then rolls out future
  overflows with protected Girard and protected box as fallback.
- `mpc_rollout_wide`: searches broad protected precision reducers as first
  reductions, excluding box as a first action, then uses the same protected
  Girard rollout and protected box fallback.

Benchmark runs write `raw_runs.csv`, `summary.csv`, `comparisons.csv`,
`predictor_comparisons.csv`, `timeseries.csv`, `bounds_timeseries.csv`,
`decision_features.csv`, `selection_summary.csv`,
`predicted_sequence_summary.csv`, `config.json`, and `report.json`. Summaries
are grouped by scenario, predictor mode, method, and metric; comparisons use
`mpc_rollout_wide` as the preferred MPC baseline when it is present, then fall
back to `mpc_rollout_girard`, `mpc_sequence`, and `mpc`.

Selection diagnostics are data-driven from existing run artifacts:
`selection_summary.csv` counts first selected reducer labels, while
`predicted_sequence_summary.csv` separates first-action box usage from future
fallback-box usage. Suite aggregation writes learned-policy rows without
double-counting the baseline reruns and emits `analysis_notes.json` with metric
winners, soundness checks, and warning flags.

The benchmark CLI and figure generator support method sets `paper`,
`paper_plus_ours`, `paper_plus_wide`, and `extended`. Use `paper` for exact
static-baseline replication, `paper_plus_ours` for the focused TACAS story, and
`paper_plus_wide` for the broader rollout ablation.

## Commit & Pull Request Guidelines

This checkout has no usable Git history to infer project-specific commit conventions. Use concise, imperative commit messages, for example `Add calibration-aware reducer test`. Pull requests should describe the research or behavior change, list commands run, mention any changed benchmark outputs, and link related notes or issues. Include screenshots only for notebook or report-rendering changes.

## Agent-Specific Instructions

Do not hand-edit generated files in `results/` unless the task explicitly concerns saved benchmark artifacts. Prefer changing source, tests, or benchmark configuration and regenerating outputs with the CLI.

When changing reducer or MPC behavior, keep the soundness boundary explicit:
policies may optimize approximate predicted cost, but only certified reducers
may change monitor state. Preserve required generator metadata through
`ProtectedReducer` unless the task deliberately changes that contract.

Do not add `IdentityReducer` or `no_reduction` back into default experiment or
learned-policy candidate sets unless the task explicitly reopens no-op
experiments. Keep protected box out of `mpc_rollout_wide` first-action
candidates; it is currently an emergency fallback only.
