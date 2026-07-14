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
- `src/pzr/learning/`: scenario-neutral cost-sensitive ranking model.
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

Run a small staged current-state ranking smoke:

```bash
pzr-learning collect --output /tmp/pzr-learning/base --event-count 30 \
  --budgets 40,80 --train-seeds 2 --validation-seeds 1 --test-seeds 1
pzr-learning train --dataset /tmp/pzr-learning/base/dataset \
  --output /tmp/pzr-learning/model
```

Run the resumable two-round Geometry15 experiment with:

```bash
PZR_OUT_DIR=results/rtlola-learning-geometry15-drift4cm-7371495-b4cfbf4-e6ecd0b \
  tools/run_rtlola_learning_full.sh
```

See `science/LEARNING_PIPELINE.md` for the feature contract, seed schedule,
teacher semantics, aggregation rounds, exact evaluation, and artifact schemas.
Learning runs are intentionally separate from `pzr-benchmark`.

Prepare or resume the full FPR-first robot-arm sweep:

```bash
PZR_OUT_DIR=results/rtlola-arm-mpc-variants-a143dd6-e6ecd0b-exact-metrics \
  tools/run_rtlola_robot_arm_fpr_overnight.sh
```

The overnight wrapper evaluates all six packaged RLolaEval traces at their full
authoritative lengths and at budgets `40,80,120,180`, with Girard, Scott,
PCA, Combastel, legacy beam MPC, root-tail MPC, endpoint-tail
MPC, and integrated-tail MPC. Tail variants default to an eight-event Girard
tail and one beam continuation per first-action root. MPC and learning choose
among Girard, Scott, PCA, and Combastel. Set
`PZR_LENGTH` only when an intentional common truncation is required. Every
trace/budget/method has its own command- and source-aware completion marker and
log, so interrupted runs resume without repeating other methods. Successful
stages are validated for complete rows; native method failures are accepted
only when recorded explicitly. Before those cells, one resumable reference
stage per trace caches exact trigger verdicts and compact state-loss data.
Learned selection is deferred and skipped by default; set
`PZR_SKIP_LEARNING=0` to run the pooled ranker explicitly.

The emitted MPC method identifiers are:

- `mpc_terminal_beam`: multi-action beam search with terminal loss only;
- `mpc_terminal_girard_tail`: beam search scored after a fixed Girard tail;
- `mpc_cumulative_girard_tail`: cumulative explicit and Girard-tail loss;
- `mpc_one_step_girard_rollout`: one optimized reducer followed by Girard rollout.

Prepare or resume the short exact-reference MPC objective study:

```bash
PZR_OUT_DIR=results/rtlola-arm-mpc-variants \
  tools/run_rtlola_mpc_variant_study.sh
```

This compares the legacy terminal-loss beam with extended-endpoint,
integrated Girard-tail, and root-only Girard-tail variants. The default tail
scan is `0,4,8,16`; `PZR_TAIL_HORIZONS`, `PZR_ROOT_BEAM_WIDTH`, and the usual
trace, budget, and length variables can override it.

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

- `core`: exact no-reduction baseline, Girard, Scott, PCA, Combastel, and
  binding-loss beam MPC.
- `static`: exact baseline plus the default bounded comparator set; excluded
  transforms remain available through explicit `--methods` overrides.
- `mpc`: the legacy beam and all three experimental tail variants.
- `all`: `static` plus all MPC variants.

The binding also exposes interval hull, Althoff A, colinear scale, and three
clustering reducers. They remain opt-in through `--methods`; current robot-arm
screening found interval hull consistently poor, Althoff A and colinear scale
too slow, and deterministic clustering both unreliable and harmful to ranking
training because of its extreme loss scale. Random and diverse clustering are
not wired into the benchmark because both fail immediately on the robot-arm
state.

The MPC and learned candidate set is `girard`, `scott`, `pca`, and
`combastel`. `none` is automatic only while the pre-event state is within the
transform bound. `interval` is an emergency fallback.

## Semantic Contract

- `budget` is passed directly to `ZonotopeConfig.<method>(budget)`. It is an
  RTLola pre-event transform bound, not a post-event dense-column cap.
- Fresh event slack may make the committed state exceed that number. This is
  reported as `post_event_over_bound`.
- Dense dynamic slots, active nonzero dynamic generators, zero dynamic slots,
  and total generators including constant slack are reported separately.
- Legacy MPC and teacher costs use binding-native terminal
  `approx_loss_state`. Experimental tail variants use either the extended
  endpoint loss or the undiscounted sum of binding-native state losses.
- Tail variants evaluate a static Girard auxiliary policy after the optimized
  horizon; tail actions are diagnostics and are not reported as committed
  predicted actions.
- The benchmark reference mode controls offline metrics and caching only;
  binding-loss MPC always constructs its own unreduced horizon rollout.
- Learned inference uses 15 aggregate current-zonotope/budget features, ranks
  native transforms once, and directly tries them through the binding. It has
  no future-event input or inference-time rollout and never writes a matrix.
- `none` and fallback actions are excluded from learned targets.
- Robot-arm and Omni constant calibration generators are kept outside dynamic
  reduction and protected by binding-backed regression tests.
- Robot-arm trigger labels are the sparse binding keys `Trigger#0` through
  `Trigger#3`; an absent key is false and an emitted message is true.
- Numeric public streams remain part of the specification contract, but their
  symbolic affine strings are not parsed into Python-side bounds.
- Use `--reference-mode exact` for full-length evaluation. It runs the
  unreduced monitor once and caches exact trigger booleans plus each state
  coordinate's center and interval radius. Reduced runs reconstruct a compact
  interval matrix and call the binding's native `approx_loss`; opaque states
  and full generator matrices are not persisted. `verdict` caches only trigger
  booleans, and `off` disables reference metrics.
- FPR is false positives divided by exact negative steps; FNR is false
  negatives divided by exact positive steps. `state_width` is the sum of
  coordinate-wise interval widths over the dynamic state and excludes constant
  slack. Saved timeseries use the compact metric names `approx_loss` and
  `state_width`. Summaries report mean, final, maximum, and summed native
  approximation loss plus mean and maximum state width. Final loss is the
  binding result after the last event; summed loss is an unweighted per-event
  sum and is therefore trace-length dependent.

## Artifacts

Each benchmark run writes `timeseries.csv`, `summary.csv`, `aggregate.csv`, and
`run_failures.csv` below the scenario directory, plus `config.yaml`,
trigger-confusion data, and runtime/loss figures. Incomplete runs are recorded
in the failure table and excluded from metrics. Learned runs also write policy,
candidate-cost, training, ranking, metadata, and held-out evaluation files
under `learning/<scenario>/`. The staged pipeline instead writes versioned
datasets, explicit PyTorch model directories, and generalization evaluation
artifacts at the user-provided paths.
The overnight wrapper additionally writes `combined_summary.csv`,
`combined_trigger_confusion.csv`, `combined_reducer_counts.csv`,
`combined_run_failures.csv`, `method_comparison.csv`,
`mpc_action_composition.csv`, `mpc_vs_static_metrics.csv`, and the compact
`primary_metrics.csv` at its output root. The compact primary table is printed
when consolidation finishes. The metric comparison selects the best static
method independently for FPR, FNR, approximation loss, and state width. The
composition table reports both all-step and reduction-only MPC action shares.

The packaged robot-arm assets come from RLolaEval commit
`e6ecd0b2f60263e0a4270bd76a71cd9c90e685e5`. Supported traces are `figure8`,
`figure8_drift`, `random`, `random_drift`, `square`, and `square_drift`;
`figure8_drift` is the default.

Generated files under `results/` must be regenerated through the CLI rather
than edited manually.
