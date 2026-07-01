# Predictive Zonotope Reduction

This repository evaluates static, predictive, and learned reducer selection for
RTLola monitors whose uncertainty state is represented as a zonotope. RTLola
owns monitor evaluation and every state-changing reduction. Python provides
scenario traces, bounded search, learning, metrics, and artifact generation.

## Architecture

- `src/pzr/rtlola/`: binding adapter, packaged RTLola specifications, scenario
  traces, native transform catalog, MPC search, benchmark runner, and CLI.
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
  --method-set core --output /tmp/pzr-omni

tools/run_rtlola_robot_arm.sh --length 40 --seeds 1 \
  --method-set core --output /tmp/pzr-arm
```

Run regret/ranking distillation:

```bash
pzr-benchmark --profile smoke --scenario robot_arm \
  --trace-kind figure8_violated --budget 80 --method-set core \
  --learned-mode regret --regret-iterations 1 --regret-epochs 10 \
  --regret-train-seeds 1 --regret-eval-seeds 1 \
  --output /tmp/pzr-arm-learned
```

Prepare or resume the full FPR-first robot-arm sweep:

```bash
PZR_OUT_DIR=results/rtlola-arm-binding-loss \
  tools/run_rtlola_robot_arm_fpr_overnight.sh
```

The overnight wrapper evaluates full available held-out traces at budgets
`40,80,120,180` with Girard, Scott, interval hull, PCA, Combastel, and beam
MPC. Cells have completion markers and logs, so rerunning the same output
directory resumes. Learned selection is deferred and skipped by default while
its regret scaling is audited; set `PZR_SKIP_LEARNING=0` only to run the
existing pooled ranker explicitly.

Method sets are:

- `core`: exact no-reduction baseline, Girard, Scott, interval hull, PCA, and
  binding-loss beam MPC.
- `static`: exact baseline plus every bounded native binding transform.
- `mpc`: beam MPC only.
- `all`: `static` plus beam MPC.

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
- MPC and teacher costs use binding-native terminal `approx_loss_state`
  against an unreduced rollout over the same horizon.
- The benchmark reference mode controls offline metrics and caching only;
  binding-loss MPC always constructs its own unreduced horizon rollout.
- Learned inference ranks native transforms and directly tries them through
  the binding. It never writes a Python-reduced matrix into RTLola.
- `none` and fallback actions are excluded from learned targets.
- Robot-arm constant calibration generators are kept outside dynamic
  reduction and protected by binding-backed regression tests.
- Trigger labels come from RTLola `#[public]` outputs. State-zonotope widths
  and symbolic public bounds are diagnostics, not substitute trigger
  semantics.
- Use `--reference-mode verdict` for full-length FPR/FNR evaluation. It
  streams the unreduced monitor and caches only exact trigger booleans;
  `--reference-mode exact` additionally retains every unreduced matrix and is
  intended only for short approximation-loss studies.
- FPR is false positives divided by exact negative steps; FNR is false
  negatives divided by exact positive steps. Saved timeseries contain trigger
  booleans and numeric public bounds, not raw symbolic binding objects.

## Artifacts

Each run writes `timeseries.csv`, `summary.csv`, and `aggregate.csv` below the
scenario directory, plus `config.yaml`, trigger-confusion data, runtime/loss
figures, and public-stream range figures. Learned runs also write policy,
candidate-cost, training, ranking, metadata, and held-out evaluation files
under `learning/<scenario>/`.
The overnight wrapper additionally writes `combined_summary.csv`,
`combined_trigger_confusion.csv`, `combined_reducer_counts.csv`, and
`mpc_vs_static_fpr.csv` at its output root.

Generated files under `results/` must be regenerated through the CLI rather
than edited manually.
