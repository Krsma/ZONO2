# Predictive Zonotope Reduction

This repository evaluates static, predictive, and learned reducer selection for
RTLola monitors whose uncertainty state is represented as a zonotope. RTLola
owns monitor evaluation and every state-changing reduction. Python provides
scenario traces, bounded search, learning, metrics, and artifact generation.

## Architecture

- `src/pzr/rtlola/`: binding adapter, packaged RTLola specifications, scenario
  traces, native transform catalog, MPC search, benchmark runner, and CLI.
- `../rlola-eval/`: upstream source of truth for the packaged robot-arm
  specification and recorded traces.
- `src/pzr/learning/`: scenario-neutral regret-ranking model.
- `rlolapythonbinding/`: pinned RTLola Python binding submodule.
- `tests/`: pure unit tests and binding-backed semantic tests.
- `tools/`: binding/MuJoCo environment setup and robot-arm smoke wrapper.

The installed package has no Python-native monitor or reducer implementation.

## Setup

Initialize the pinned submodule and build the dedicated environment:

```bash
git submodule update --init --recursive
tools/setup_robot_arm_env.sh
```

The wrapper builds the binding from the exact superproject revision and
installs the package plus optional MuJoCo validation support. The current
binding dependency on `kmeans` uses a nightly feature; setup scopes
`RUSTC_BOOTSTRAP` to that crate only.

## Commands

Run tests in the binding environment:

```bash
LD_PRELOAD="$PWD/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so" \
PYTHONPATH=src external/miniconda3/envs/pzr-robot-arm/bin/python -m pytest
```

Run focused benchmarks:

```bash
pzr-benchmark --profile smoke --scenario omni_robot \
  --trace-kind x_violated --method-set core --output /tmp/pzr-omni

tools/run_rtlola_robot_arm.sh --length 40 --seeds 1 \
  --method-set core --output /tmp/pzr-arm
```

The Omni scenario provides the historically compatible `canonical` trace and
calibrated `safe`, `x_violated`, and `y_violated` traces. Numeric public
`position_x` and `position_y` bounds are included in its artifacts.

Run regret/ranking distillation:

```bash
pzr-benchmark --profile smoke --scenario robot_arm \
  --trace-kind figure8_drift --budget 80 --method-set core \
  --learned-mode regret --regret-iterations 1 --regret-epochs 10 \
  --regret-train-seeds 1 --regret-eval-seeds 1 \
  --output /tmp/pzr-arm-learned
```

Prepare or resume the full FPR-first robot-arm sweep:

```bash
PZR_OUT_DIR=results/rtlola-arm-big-a143dd6-f587a0e-release \
  tools/run_rtlola_robot_arm_fpr_overnight.sh
```

The overnight wrapper evaluates all six packaged RLolaEval traces at their full
authoritative lengths and at budgets `40,80,120,180`, with Girard, Scott,
interval hull, PCA, Combastel, and beam MPC. MPC and learning choose among
those five reducers plus deterministic clustering. Set `PZR_LENGTH` only when
an intentional common truncation is required. Cells have command- and
revision-aware completion markers and logs, so identical configurations resume
while stale binding, specification, method, or candidate configurations rerun.
Learned selection is deferred and skipped by default; set
`PZR_SKIP_LEARNING=0` to run the pooled ranker explicitly.

Run the 10-seed state-fidelity Omni pilot:

```bash
PZR_OUT_DIR=results/rtlola-omni-a143dd6-release \
  tools/run_rtlola_omni_fidelity_overnight.sh
```

The Omni wrapper evaluates budgets `8,12,16,20`, all four trace kinds, the five
static bounded comparators used by the primary experiments (including
Combastel), the binding-native terminal objective, a horizon scan, and a
held-out learned policy. Completion markers make the run resumable.

Method sets are:

- `core`: exact no-reduction baseline, Girard, Scott, interval hull, PCA, and
  binding-loss beam MPC.
- `static`: exact baseline plus every bounded native binding transform.
- `mpc`: beam MPC only.
- `all`: `static` plus beam MPC.

The binding also exposes Althoff A, colinear scale, and three clustering
reducers. Althoff A, colinear scale, and deterministic clustering remain
opt-in through `--methods`; current robot-arm screening found the first two too
slow for the full sweep and deterministic clustering frequently falls back on
rank-deficient states. Random and diverse clustering are not wired into the
benchmark because both fail immediately on the robot-arm state.

The MPC and learned candidate set remains `girard`, `scott`,
`interval_hull`, and `pca`. `none` is automatic only while the pre-event
state is within the transform bound. `interval` is an emergency fallback.

## Semantic Contract

- `budget` is passed directly to `ZonotopeConfig.<method>(budget)`. It is an
  RTLola pre-event transform bound, not a post-event dense-column cap.
- Fresh event slack may make the committed state exceed that number. This is
  reported as `post_event_over_bound`.
- Dense dynamic slots, active nonzero dynamic generators, zero dynamic slots,
  and total generators including constant slack are reported separately.
- MPC and teacher costs default to binding-native terminal
  `approx_loss_state` against an unreduced rollout over the same horizon.
- The benchmark reference mode controls offline metrics and caching only;
  binding-loss MPC always constructs its own unreduced horizon rollout.
- Learned inference ranks native transforms and directly tries them through
  the binding. It never writes a Python-reduced matrix into RTLola.
- `none` and fallback actions are excluded from learned targets.
- Robot-arm and Omni constant calibration generators are kept outside dynamic
  reduction and protected by binding-backed regression tests.
- Robot-arm trigger labels are the sparse binding keys `Trigger#0` through
  `Trigger#4`; an absent key is false and an emitted message is true.
- Numeric public streams remain part of the specification contract, but their
  symbolic affine strings are not parsed into Python-side bounds.
- Use `--reference-mode verdict` for full-length FPR/FNR evaluation. It
  streams the unreduced monitor and caches only exact trigger booleans;
  `--reference-mode exact` additionally retains every unreduced matrix and is
  intended only for short approximation-loss studies.
- FPR is false positives divided by exact negative steps; FNR is false
  negatives divided by exact positive steps. Saved timeseries contain trigger
  booleans and numeric public bounds, not raw symbolic binding objects.

## Artifacts

Each run writes `timeseries.csv`, `summary.csv`, `aggregate.csv`, and
`run_failures.csv` below the scenario directory, plus `config.yaml`,
trigger-confusion data, and runtime/loss figures. Incomplete runs are recorded
in the failure table and excluded from metrics. Learned runs also write policy,
candidate-cost, training, ranking, metadata, and held-out evaluation files
under `learning/<scenario>/`.
The overnight wrapper additionally writes `combined_summary.csv`,
`combined_trigger_confusion.csv`, `combined_reducer_counts.csv`,
`combined_run_failures.csv`, `method_comparison.csv`,
`mpc_action_composition.csv`, `mpc_vs_static_fpr.csv`, and
`mpc_vs_static_fidelity.csv` at its output root. The composition table reports
both all-step and reduction-only MPC action shares.

The packaged robot-arm assets come from RLolaEval commit
`f587a0ecb783dbc88f2feb6621c5278a10cf781d`. Supported traces are `figure8`,
`figure8_drift`, `random`, `random_violated`, `square`, and `square_drift`;
`figure8_drift` is the default.

Generated files under `results/` must be regenerated through the CLI rather
than edited manually.
