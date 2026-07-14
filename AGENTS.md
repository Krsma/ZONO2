# Repository Guidelines

## Project Structure

This Python 3.11 research package is RTLola-centered:

- `src/pzr/rtlola/`: specifications, trace adapters, binding wrapper, native
  transform catalog, search, benchmark execution, reporting, and CLI.
- `src/pzr/learning/`: generic cost-sensitive ranking data/model/training code.
- `rlolapythonbinding/`: pinned binding submodule.
- `tests/`: pure tests plus binding-backed semantic contracts.
- `tools/`: reproducible environment setup and robot-arm smoke execution.

Robot-arm trace CSVs and the vendored MuJoCo model are data/validation assets,
not an alternative Python monitor.

## Setup and Tests

```bash
git submodule update --init --recursive
tools/setup_robot_arm_env.sh

LD_PRELOAD="$PWD/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so" \
PYTHONPATH=src external/miniconda3/envs/pzr-robot-arm/bin/python -m pytest
```

The normal Python environment runs pure tests and skips binding integration
tests. Release validation must use the binding environment with no skips.

Useful smokes:

```bash
pzr-benchmark --profile smoke --scenario omni_robot --method-set core \
  --output /tmp/pzr-omni

tools/run_rtlola_robot_arm.sh --length 20 --seeds 1 --method-set core \
  --output /tmp/pzr-arm

PZR_OUT_DIR=results/rtlola-arm-mpc-variants-b4cfbf4-e6ecd0b-exact-metrics \
  tools/run_rtlola_robot_arm_fpr_overnight.sh

pzr-learning generate --output /tmp/pzr-learning/traces --event-count 10 \
  --conditions random_waypoint --seed-count 3
pzr-learning collect --output /tmp/pzr-learning \
  --trace-store /tmp/pzr-learning/traces \
  --budgets 10 --candidates girard,scott --train-seeds 1 \
  --validation-seeds 1 --test-seeds 1
pzr-learning train --dataset /tmp/pzr-learning/dataset \
  --output /tmp/pzr-learning-model --epochs 2
```

## Current RTLola Experiment Configuration

The packaged robot-arm specification is
`src/pzr/rtlola/specs/robot_arm.lola`. It and the six trace CSVs were imported
from RLolaEval revision `e6ecd0b2f60263e0a4270bd76a71cd9c90e685e5`;
the expected specification SHA-256 is
`aab5b768d872bc4f5b6dc11b96805c2d451cc5c91eb573225f6b0e246cee6acc`.
Do not substitute an older local robot-arm specification.

The required native stack is:

- binding revision `7371495a113694ebb9958061f93910e7f65e84f3`;
- interpreter revision `b4cfbf4680e6641f131a64d6d9e9ef57ec228976`;
- a `maturin build --release`/release-profile binding.

`src/pzr/rtlola/binding.py` rejects a mismatched interpreter or debug build.
The current interpreter exposes logical all-zero dynamic rows for state export,
while binding-native transformations still reduce compact nonzero rows.
Negative coefficients are not zero and must remain represented. Python code
must not depend on stable generator row positions or dense matrix shapes; use
the binding-native transforms, counters, and approximation loss. PZR reports
the compact reducer dimension separately from the exported logical row count,
and budget checks must use the compact reducer dimension.

The last recorded full release-binding validation after the parallel learning
integration was 95 passing tests with no skips.

The authoritative trace kinds and full lengths are:

- `figure8` and `figure8_drift`: 2,340 events each;
- `random`: 1,495 events;
- `random_drift`: 1,433 events;
- `square` and `square_drift`: 1,983 events each.

`figure8` and `square` are nominal structured paths, their `_drift` variants
add progressive tool-center drift, `random` explores the geofence broadly, and
`random_drift` combines random exploration with progressive drift. Do not pool
them without preserving `trace_kind`.

The emitted MPC methods are:

- `mpc_terminal_beam`: multi-action beam search, terminal loss only;
- `mpc_terminal_girard_tail`: beam search scored at the end of a fixed Girard
  tail;
- `mpc_cumulative_girard_tail`: cumulative explicit-horizon and Girard-tail
  loss;
- `mpc_one_step_girard_rollout`: optimize the current reducer, then score a
  Girard rollout.

The overnight defaults are horizon 4, beam width 4, Girard tail horizon 8,
and one retained continuation per first-action root for stratified variants.
The wrapper first prepares one exact reference cache per trace, then runs each
trace/budget/method as a separately validated, source-aware resumable stage.
Combined tables are built only after all stages finish.

## Current Robot-Arm Results

The previous robot-arm artifacts used the obsolete verdict-only reference and
long metric-column schema and were removed on 2026-07-05. There is currently
no active full-suite artifact. Do not quote the earlier partial sweep as a
completed six-trace evaluation.

The focused Girard-versus-MPC run started on 2026-07-06 was deliberately
terminated before completing the square traces. Its partial artifact is not a
completed evaluation. The next run must use a fresh output directory. Exact
reference stages cache trigger verdicts and logical-row center/radius data once per
trace. Method summaries report `fpr`, `fnr`, mean/final/max/summed native
approximation loss, and mean/max `state_width`; `primary_metrics.csv` contains
the compact completion table. `mpc_vs_static_metrics.csv` selects the best
static method independently for each metric.

## Coding and Testing

Use 4-space indentation, type hints, immutable dataclasses where appropriate,
and grouped standard-library/third-party/local imports. Tests use pytest,
`tmp_path`, and `numpy.testing`.

Changes to scenarios, actions, search, or learning require tests for:

- deterministic state branching and tie behavior;
- exact RTLola transform-bound semantics;
- dense versus active generator accounting;
- outer-bound soundness against an unreduced branch;
- constant calibration generator preservation;
- trigger/public-stream keys from the packaged specification;
- fallback and infeasible-candidate accounting;
- learned candidate alignment and direct-inference behavior;
- non-empty benchmark and learning artifacts.

## Trusted Boundary

Selectors may inspect states and choose actions, but only
`rlola_python_binding.ZonotopeConfig` transforms may mutate monitor state.
Do not add matrix writeback or Python-side reducers.

The current robot-arm MPC/learning candidates are `girard`, `scott`, `pca`,
and `combastel`. Interval hull and deterministic clustering remain available
only as explicit binding diagnostics; short learning screens found interval
hull consistently poor and clustering's extreme losses dominated the ranking
objective. Do not add them, `none`, `interval`, unbounded transforms,
random/diverse clustering, Althoff A, or colinear scale to ordinary candidate
catalogs without a new explicit experiment change. `none` is the exact
baseline and automatic under-bound action; `interval` is fallback-only.

The current learned policy uses the version-2, 15-scalar Geometry15 schema:
the original 12 budget/current-zonotope aggregates plus row-width
concentration, active-generator norm variation, and mean normalized off-axis
generator mass. It is strictly pre-event and does not use stream values,
history, spectral statistics, or an inference-time preview rollout. The
current experiment pre-generates forty independent 500-event nominal random-
waypoint traces: twelve base-training seeds, four validation seeds, and two
fresh twelve-seed DAgger rounds. Its fixed-trace comparison is Girard versus
`learned_geometry15` versus the two-event `mpc_terminal_full_width` teacher.
The six fixed evaluation traces always retain their full authoritative lengths.
Teacher shards and post-reference evaluation cells use spawned worker
processes, defaulting to `PZR_WORKERS=8`; each worker owns its monitor and
planner. BLAS, OpenMP, MKL, and NumExpr remain limited to one thread per worker.

`budget` is the binding transform bound. Never subtract a fresh-generator
reserve or interpret post-event dense slots as a violation. Preserve the
distinction between dynamic, active, zero, and constant generators.

MPC and teacher costs use binding-native approximation loss. Terminal beam
uses terminal loss; the experimental tail variants use either the extended
endpoint loss or an undiscounted sum of binding-native state losses. Do not
replace these with width, trigger-straddling, or a Python proxy during
unrelated cleanup.

Benchmark reference mode controls offline metrics and caching only. MPC and
teacher searches construct their own unreduced horizon rollouts.

Offline exact references remain specification-independent. Each cache row
contains exact trigger booleans and total-state logical-row center/radius
vectors. The engine reconstructs an interval matrix and invokes the existing native
`approx_loss` while the candidate is applied only to the planner monitor. It
must restore the planner in `finally` and must never mutate the live monitor.
Do not edit the RTLola binding to implement these metrics.

Robot-arm trigger labels and public metrics come from
`rtlola/specs/robot_arm.lola`. Constant encoder-calibration slack must remain
unchanged by dynamic reduction.

For full-length metrics use `--reference-mode exact`. Exact caches are reusable
across methods and budgets and do not retain opaque states or full generator
matrices. `verdict` remains available for trigger-only runs. FPR uses exact
negative steps as its denominator; FNR uses exact positive steps. `state_width`
is the existing dynamic-state interval-width sum and excludes constant slack.
`final_approx_loss` is the last event's binding result and
`sum_approx_loss` is the unweighted sum across events, so summed loss is only
comparable between methods evaluated on the same trace.

## Repository Safety

Do not discard uncommitted work. Use `git pull --ff-only`, pin submodules
through the superproject, and avoid setup scripts that silently fetch or
checkout another binding revision. Do not hand-edit generated files in
`results/`.

Use concise imperative commits. Report commands run, changed experiment
semantics, binding revision changes, and generated-artifact impact.
