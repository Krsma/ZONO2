# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.11 research package for predictive zonotope reduction:
sound, bounded-memory reducer selection for runtime monitors that track
uncertainty as zonotopes. Source code lives under `src/pzr/`:

- `zonotope/`: immutable zonotope primitive, interval metrics, certified
  reducers, scoring heuristics, and `ProtectedReducer`.
- `monitoring/`: monitor adapter protocol, monitor state, trigger specs, and
  trigger evaluation.
- `mpc/`: receding-horizon reducer selection over certified reduction actions.
- `imitation/`: DAgger traces, feature extraction, datasets, and learned
  reducer-selection policies.
- `systems/`: math-only benchmark monitors (`omni_robot`, `simple_robot`).
- `envs/`: MuJoCo-backed environments and monitors (`point_mass`,
  `robot_arm`), requiring the optional `sim` dependencies.
- `experiments/`: benchmark orchestration, aggregation, tables, figures,
  robot-arm animation, DAgger evaluation, and profiles.
- `utils/`: seeding, timing, and serialization helpers.

Tests are in `tests/`. The CORA comparison fixture is
`tests/fixtures/cora_reference.json`. Generated benchmark outputs are written
under `results/`; do not hand-edit them unless the task explicitly concerns
saved benchmark artifacts.

## Build, Test, and Development Commands

- `python -m pip install -e ".[dev]"`: install the package in editable mode
  with pytest.
- `python -m pip install -e ".[dev,learning,sim]"`: include PyTorch and MuJoCo
  extras for DAgger and simulator-backed scenarios.
- `pytest`: run the full test suite configured by `pyproject.toml`.
- `pytest tests/test_full_eval.py -x -q`: focused full-evaluation smoke path.
- `pytest tests/test_robot_arm.py -k trigger_zonotope`: focused robot-arm
  trigger projection checks.
- `pzr-benchmark --profile smoke --scenario omni_robot --output /tmp/pzr-smoke`:
  quick CLI smoke run.
- `pzr-benchmark --profile standard --output results/my-run`: standard suite.
- `pzr-benchmark --profile paper --scenario all --method-set all --budget 10 --horizon 4 --beam-width 4 --seeds 30 --jobs 4 --no-dagger --output results/paper-full`:
  benchmark-only paper-style suite; expect a multi-hour run when exhaustive
  `mpc_sequence` is included.
- `pzr-benchmark --profile smoke --budget-sweep "6,8,10,12" --output results/sweep`:
  budget trade-off run; DAgger is disabled automatically.
- `pzr-robot-arm-animation --trace benchmark --seed 0 --length 200 --method scott --output results/robot-arm-animation`:
  render a benchmark-distribution MuJoCo robot-arm replay as GIF, paper
  stills, storyboard, and metadata.
- `pzr-robot-arm-animation --trace paper --length 200 --method scott --output results/robot-arm-paper-viz`:
  render a deterministic explanatory robot-arm trace for paper motivation.
- `python -m pzr.cli --profile smoke --scenario simple_robot --no-dagger --no-progress --output /tmp/pzr-smoke`:
  smoke-test without relying on the installed console script.
- `python -m pzr.experiments.robot_arm_animation --trace benchmark --seed 0 --length 50 --method scott --output /tmp/pzr-arm-viz`:
  smoke-test the robot-arm animation module without relying on the installed
  console script.
- `python -m pzr.experiments.robotics_probe --candidate all --length 60 --budget 10 --output /tmp/pzr-robotics-probe`:
  audit candidate drone/F1TENTH robotics environments before promoting either
  into benchmark defaults.
- `tools/run_pzr_smoke_parallel.sh`: scripted parallel smoke run with bounded
  BLAS/OpenMP threads and log capture.
- `tools/run_pzr_paper_static.sh`: static-baseline paper run, useful for fast
  sanity checks without MPC or DAgger.
- `tools/run_pzr_paper_full.sh`: full paper run wrapper. It disables DAgger by
  default; set `PZR_WITH_DAGGER=1` to include the learned-policy pass.

`pyproject.toml` currently declares `pzr-benchmark = pzr.cli:main` and
`pzr-robot-arm-animation = pzr.experiments.robot_arm_animation:main`. CLI
profiles are `smoke`, `standard`, and `paper`; scenarios are `all`,
`omni_robot`, `simple_robot`, plus
`point_mass` and `robot_arm` when MuJoCo imports successfully. Method sets are
`all`, `static`, and `standard`; `all` includes every registered method,
`static` excludes MPC, and `standard` preserves the old static plus legacy
`mpc_rollout`/`mpc_sequence` set. DAgger runs by default; use `--no-dagger`
to skip it. `--beam-width N` controls the bounded-width beam MPC method and
defaults to 4. DAgger uses `mpc_sequence3` as its default expert; use
`--dagger-expert NAME` to override it. `--jobs N` parallelizes
default benchmark runs across seeds when no custom scenarios, custom methods,
or trace collector are provided. Keep BLAS/OpenMP thread counts at 1 for long
`--jobs` runs; the scripts in `tools/` already export these limits and set
`MPLCONFIGDIR=results/matplotlib-cache`.

## Coding Style & Naming Conventions

Use 4-space indentation, Python type hints, and small dataclasses or immutable
value objects where appropriate. Keep modules and functions in `snake_case`;
classes and enums use `PascalCase`; constants use `UPPER_SNAKE_CASE`. Prefer
explicit `numpy` array conversions and validation at API boundaries, matching
`src/pzr/zonotope/core.py`. Add short docstrings for public modules and domain
objects. No formatter or linter is configured, so preserve the existing style
and keep imports grouped as standard library, third-party, then local.

## Testing Guidelines

Tests use `pytest` and live in files named `tests/test_*.py`. Name tests by
behavior, such as `test_box_reducer_contains_sampled_original_points`. Use
`tmp_path` for generated artifacts and `numpy.testing` for numeric
comparisons. New reducers, policies, monitor outputs, or benchmark artifacts
should include soundness, budget, metadata, and artifact-shape assertions where
relevant.

MPC-related changes should assert certified budgeted states, calibration
metadata preservation, chosen reducer accounting, predictor/search metadata
(`evaluated_leaves` and `pruned_branches`), fallback behavior, and trigger
projection behavior. Beam-search changes should cover wide-beam agreement
with exact search, narrow-beam pruning, deterministic tie behavior, and
protected-index preservation through first and future reductions. Learned-policy
and DAgger changes should keep the candidate set aligned with benchmark
reducers and should not include `IdentityReducer` unless no-op experiments are
explicitly reopened.

Figure or artifact pipeline changes should smoke-test `pzr-benchmark` with
small profiles/seeds and assert that expected CSV and PDF outputs are
non-empty. Robot-arm animation changes should smoke-test
`pzr-robot-arm-animation` or `python -m pzr.experiments.robot_arm_animation`
with a short trace and assert non-empty GIF/PNG/PDF/metadata outputs.
Diagnostic or aggregation changes should check `summary.csv`, `aggregate.csv`,
`timeseries.csv`, `config.yaml`, generated figure files, and learned-policy
rows when DAgger is enabled.

Parallel benchmark changes should compare serial and parallel runs on stable
seed-level metrics. Avoid asserting raw wall-clock timings exactly.

## Soundness Boundary

The project relies on policy-independent soundness: selectors may optimize
approximate predicted cost, but only certified reducers may change monitor
state. Every reducer must return a `ReductionResult` whose certificate is
sound and whose reduced zonotope stays within the requested generator budget.

Use `ProtectedReducer` for monitors with persistent calibration generators.
It preserves columns named by `state.calibration_indices`, reduces only the
residual columns, and recombines protected columns first. After a protected
reduction, calibration indices should be renumbered to
`tuple(range(len(old_calibration_indices)))`. Do not bypass this path in
static, MPC, rollout, or learned policies unless the task deliberately changes
the calibration contract.

Use `reduce_with_protection` when applying a reducer from policy code. It
centralizes the "plain reducer vs protected reducer" branch and keeps static,
MPC, rollout, and learned policies consistent.

Do not add `IdentityReducer` to default benchmark, MPC, or learned-policy
candidate sets unless the task explicitly reopens no-op experiments.

## Trigger Zonotope Contract

Monitors expose `trigger_zonotope(state) -> Zonotope`. For most monitors this
returns `state.zonotope`, but `RobotArmMonitor` projects the 6D joint-space
state into a 2D Cartesian end-effector zonotope before evaluating triggers.
Trigger `state_index` values index into the trigger zonotope, not necessarily
the raw monitor state.

Any code that evaluates trigger bounds, trigger widths, trigger straddling, or
trigger-derived features must use `monitor.trigger_zonotope(state)`. This is
especially important in MPC costs and imitation features, where using
`state.zonotope` directly makes robot-arm decisions optimize joint-space axes
instead of Cartesian end-effector axes.

`RobotArmMonitor` uses a conservative coupled joint uncertainty basis for its
three persistent calibration generators and three fresh noise generators per
step. This is intentionally not independent axis-only joint noise: it makes the
Cartesian trigger projection reducer-discriminative while still containing the
original independent joint-error box. Robot-arm changes should preserve that
containment property, the coupled generator structure, Cartesian trigger
projection, and reducer-discrimination regression coverage.

Robot-arm trace visualization supports `--trace benchmark` and `--trace paper`.
Benchmark trace rendering uses `generate_robot_arm_trace_records`, which keeps
the existing measurement-only trace API intact while also exposing true MuJoCo
state, target, action, end-effector position, and episode metadata. Paper trace
rendering is a deterministic scripted-kinematic explanatory path with reachable
joint-space waypoints whose end-effector path passes near the trigger region;
it is not the quantitative benchmark distribution. The animation overlay should
use separate physical and trigger-space panels: the physical panel stays at
workspace scale, while the trigger-space panel shows the Cartesian
end-effector zonotope polygon and interval hull as the rigorous uncertainty
view. Do not reintroduce ghost-arm diagnostics; the visualization should focus
on the true arm, measured arm/point, end-effector trail, trigger zonotope,
interval hull, and end-effector trigger region.

## Benchmark Methods & Artifacts

Default benchmark methods include static protected reducers (`girard`,
`combastel`, `pca`, `methA`, `scott`, `box`) plus legacy and focused MPC
methods:

- `mpc_rollout`: broad first-action search, then fixed protected Girard rollout
  for future predicted overflows, with protected box fallback.
- `mpc_rollout_methA`: top-3 first-action search, then fixed protected MethA
  rollout for future predicted overflows, with protected box fallback.
- `mpc_rollout_scott`: top-3 first-action search, then fixed protected Scott
  rollout for future predicted overflows, with protected box fallback.
- `mpc_pair_rollout3`: evaluates first-action and future-base reducer pairs
  over the top-3 set.
- `mpc_sequence`: exhaustive broad search over `girard`, `combastel`, `pca`,
  `methA`, and `scott`.
- `mpc_sequence3`: exhaustive focused search over the top-3 set
  `girard`, `methA`, and `scott`.
- `mpc_beam3`: deterministic beam search over the same top-3 set, using
  `BenchmarkConfig.beam_width` / `--beam-width` (default 4).

The top-3 set is intentionally `girard`, `methA`, and `scott`; it removes PCA
and Combastel from focused MPC searches because current robot-arm diagnostics
show PCA is harmful and Combastel is usually redundant with Girard. `all`
includes every method above. `standard` preserves the old static plus legacy
`mpc_rollout`/`mpc_sequence` set so existing programmatic smoke runs do not
become unexpectedly larger.

Benchmark runs save per-scenario `timeseries.csv`, `summary.csv`, and
`aggregate.csv`, plus top-level `config.yaml`. The CLI also writes figure PDFs
under `figures/`. `summary.csv` is per method and seed; `aggregate.csv`
bootstraps or aggregates those seed-level rows.

`timeseries.csv` includes trigger width, exact unreduced trigger width,
approximation error, false-positive flags, reducer selections, and reduction
timings. `summary.csv` and `aggregate.csv` include `mean_approx_error`,
`max_approx_error`, `abs_error_range`, and `false_positive_rate` in addition
to width, generator-count, timing, and soundness metrics. Ground truth for
these columns is computed by running the same monitor trace without reductions
and comparing trigger-zonotope bounds.

Figure generation currently includes combined trigger-width timeseries,
approximation-error timeseries, method comparison bars, FPR/error-range panels,
and reducer-selection bars for MPC or learned methods.

Robot-arm animation writes per-run GIFs, first/middle/last PNG and PDF stills,
storyboard PNG/PDF files, and metadata JSON under the requested output
directory. The storyboard is the primary paper-facing artifact. The CLI
requires an explicit `--method`; valid methods are any method registered by
`default_methods`, plus `none` for unreduced visualization. Use `--trace
benchmark --seed N` for sanity-checking quantitative benchmark seeds and
`--trace paper` for explanatory paper motivation.

The current robot-arm visualization remains a sanity/debug artifact, not the
main paper visualization. Use `pzr.experiments.robotics_probe` for the next
environment-selection pass. The probe is intentionally outside benchmark
defaults and scores candidate drone/F1TENTH derived-stream traces for reducer
differentiation, near-threshold behavior, reductions, and budget/soundness.

DAgger evaluation defaults to exact `mpc_sequence3` as the expert and trains
over the top-3 reducer labels, with protected box kept as learned-policy
fallback. It appends learned-policy rows to the in-memory benchmark result used
for printed tables and post-run figures. Saved per-scenario benchmark CSVs are
written before DAgger unless the persistence behavior is explicitly changed.

## Commit & Pull Request Guidelines

This checkout has no usable Git history to infer project-specific commit
conventions. Use concise, imperative commit messages, for example
`Add calibration-aware reducer test`. Pull requests should describe the
research or behavior change, list commands run, mention any changed benchmark
outputs, and link related notes or issues. Include screenshots only for
notebook or report-rendering changes.

## Agent-Specific Instructions

Do not hand-edit generated files in `results/`; change source, tests, or
benchmark configuration and regenerate artifacts through the CLI.

When changing reducer, MPC, monitor, or learned-policy behavior, keep the
soundness boundary explicit and preserve required generator metadata through
`ProtectedReducer`. Keep protected box as an emergency fallback for MPC rather
than a first-action candidate unless the task explicitly changes the method
definition.

Do not change `WeightedZonotopeCost` or trigger straddling weights as part of
top-3, beam, or rollout search work unless the task explicitly reopens cost
ablations. Those ablations were intentionally deferred.

Treat `AUDIT.md` as a list of claims to verify against code and tests, not as
ground truth. Some findings may be research-metric choices rather than bugs.
