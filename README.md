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
- `src/pzr/learning/`: scenario-neutral pairwise reducer learning, secondary
  distillation/DART objectives, and bounded challenger screening.
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

Run a small Pairwise Ranking Policy smoke:

```bash
pzr-learning generate --output /tmp/pzr-learning/traces --event-count 30 \
  --conditions random_waypoint --seed-count 3
pzr-learning collect --output /tmp/pzr-learning/clean \
  --trace-store /tmp/pzr-learning/traces --budgets 40,80 \
  --train-seeds 2 --validation-seeds 1 --test-seeds 0 \
  --collection-mode teacher
pzr-learning train --dataset clean=/tmp/pzr-learning/clean/dataset \
  --output /tmp/pzr-learning/model-pairwise-ranking-policy --objective pairwise --epochs 2
```

The current paper evaluation is defined by `experiments/paper_evaluation_v2.yaml`.
Run or resume only the release-checked teacher/training/formal-pilot bundle with:

```bash
tools/run_paper_evaluation.sh explore
```

`explore` includes preflight, verified reuse of the nominal teacher dataset,
seven fixed-hyperparameter budget-specialist trainings, and the 112-cell
nominal pilot (two seeds, seven budgets, eight methods). It
does not run objective comparison, headline, held-out generalization, ablation,
timing, reporting, validation, or the historical bounded-exploration study. Its
outputs are source-compatible with a later complete `run` and will be validated
and skipped when that run resumes.

Run or resume the paper's scientific evaluation, without dedicated timing, with:

```bash
tools/run_paper_evaluation.sh evaluate
```

The command runs the release/binding preflight excluding explicitly marked
RLolaEval parity tests, teacher reuse, specialist training, pilot, headline,
matched objective comparison, held-out generalization, H/W ablation, interim
reporting, and integrity validation. Dedicated contention-free timing is
deferred; run `tools/run_paper_evaluation.sh timing` and then `run` when it is
needed for final throughput artifacts. Inspect a running or
interrupted evaluation with `tools/run_paper_evaluation.sh status`. Individual
`pzr-paper` stages remain available for focused diagnostics.

Notebook parity is a standalone development diagnostic rather than a paper-run
prerequisite. Run it explicitly after substantial binding, monitor, or search
changes, for example:

```bash
pzr-rtlola-parity --rlola-eval ../rlola-eval --output /tmp/pzr-parity \
  --trace-kinds figure8 --bounds 15 --length 20 --skip-speed-gate
```

The ordinary unfiltered pytest command still includes all parity-marked tests.

The 112-cell pilot projects only the 1,120-cell nominal held-out sweep. If it
exceeds 72 four-worker hours, the run exits after recording the projection.
After reviewing it, resume with:

```bash
tools/run_paper_evaluation.sh evaluate --approve-long-run
```

Supplying approval before the pilot exists is rejected. Valid completed stages
are skipped, partial cell matrices resume, and stale manifests require a fresh
output directory. Raw cells and stage logs remain under
`results/paper-evaluation-v2`; compact interim source tables, TeX tables, PDF/PNG
figures, and their hashes are written under
`results/paper-evaluation-v2/science-report/artifacts`. Final timing-inclusive
artifacts use `paper/corl2026/Zonotopes_at_CoRL/generated/paper_evaluation_v2`.

Learning, pilot, held-out generalization, and H/W ablation use independently
generated nominal random-waypoint traces only. The four full-length imported
figure-eight variants provide the controlled nominal/drift/geofence/combined
headline, objective, and timing comparisons. Those fixed patterned traces are
case studies, not multi-seed estimates of randomized fault generalization; the
pipeline makes no randomized drift/geofence generalization claim.

For an unattended run, start the checked-in command in a detached session:

```bash
tmux new-session -d -s paper-evaluation \
  'cd /home/vlkr/Faks/phd/ZONO2 && tools/run_paper_evaluation.sh evaluate'
```

For the exploratory bundle only:

```bash
tmux new-session -d -s paper-exploration \
  'cd /home/vlkr/Faks/phd/ZONO2 && tools/run_paper_evaluation.sh explore'
```

Exit status `0` means every required point completed, `2` means the artifact
bundle validated with explicitly unavailable scientific or timing points, `75`
means pilot approval is required, and `1` indicates an execution or integrity
failure.

`pairwise_ranking_policy` denotes a family of seven specialists. Specialist
`b` is trained only from budget-`b` rows of the one verified terminal full-width
teacher dataset and is evaluated only at budget `b`; all specialists share the
same fixed training hyperparameters. Cross-budget and joint-budget learned
policies are deferred follow-up work. Offline terminal beam uses recorded future inputs, while the linear
predictive beam is causal and deployable. Exact caches are used only for
offline metrics; selection and teaching retain native unreduced rollouts.

Soft-KL and guarded one-round DART remain available as completed secondary
ablations. Their historical result artifact is no longer active. The observed
DART improvement was marginal and is confounded by additional training data,
so neither method appears in the default pipeline.

Predictive MPC sees the exact arrived current event and causal input history;
PRP remains strictly pre-event. Forecasts use scheduled times at 0.1-second
increments and never inspect recorded future inputs during action selection.

See `science/LEARNING_PIPELINE.md` for the feature contract, pairwise and
expected-regret objectives, secondary DART calibration, seed schedules,
promotion gate, exact evaluation, and artifact schemas.
Learning runs are intentionally separate from `pzr-benchmark`.

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
- Paper MPC and teacher costs use binding-native `approx_loss_state`.
  `mpc_terminal_beam` and `mpc_terminal_full_width` use endpoint loss;
  `mpc_cumulative_beam` uses the undiscounted explicit-horizon sum only in the
  matched objective comparison.
- The benchmark reference mode controls offline metrics and caching only;
  binding-loss MPC always constructs its own unreduced horizon rollout.
- Learned inference uses 15 aggregate current-zonotope/budget features, scores
  native transforms once, and directly tries them through the binding. It has
  no future-event input or inference-time rollout and never writes a matrix.
- Soft targets are normalized from the complete feasible teacher-cost vector;
  `none` and fallback actions are excluded.
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
candidate-cost, target diagnostics, optional DART calibration, metadata, and held-out evaluation files
under `learning/<scenario>/`. The staged pipeline instead writes versioned
datasets, explicit PyTorch model directories, and generalization evaluation
artifacts at the user-provided paths.
The paper pipeline writes source-aware cells, diagnostic time series, stage
summaries, a top-level run manifest, and per-stage logs below
`results/paper-evaluation-v2`. Its report stage writes compact CSV and TeX
tables plus PDF/PNG figures and a complete hash manifest into the paper's
generated-artifact directory.

The packaged robot-arm assets come from RLolaEval commit
`2257d074173a6dd475c042ef9a82cd8755a81ac3`. Each of `figure8`, `random`, and
`square` has compliant, drift, geofence, and drift-geofence variants;
`figure8_drift` is the default. The standalone `pzr-rtlola-parity` command
retains the full notebook-cell and production/oracle throughput validation; it
is not invoked by the paper pipeline.

Generated files under `results/` must be regenerated through the CLI rather
than edited manually.
