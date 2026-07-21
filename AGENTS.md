# Repository Guidelines

## Project Structure

This Python 3.11 research package is RTLola-centered:

- `src/pzr/rtlola/`: specifications, trace adapters, binding wrapper, native
  transform catalog, search, benchmark execution, reporting, and CLI.
- `src/pzr/learning/`: reducer-cost datasets, Pairwise Ranking Policy training,
  secondary objectives, DART calibration, and bounded challenger screening.
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

PZR_OUT_DIR=results/rtlola-arm-mpc-variants-01c92a2-2257d07-exact-metrics \
  tools/run_rtlola_robot_arm_fpr_overnight.sh

pzr-learning generate --output /tmp/pzr-learning/traces --event-count 10 \
  --conditions random_waypoint --seed-count 3
pzr-learning collect --output /tmp/pzr-learning \
  --trace-store /tmp/pzr-learning/traces \
  --budgets 10 --candidates girard,scott --train-seeds 1 \
  --validation-seeds 1 --test-seeds 0 --collection-mode teacher
pzr-learning train --dataset clean=/tmp/pzr-learning/dataset \
  --output /tmp/pzr-learning-model --objective pairwise --epochs 2
```

## Current RTLola Experiment Configuration

The packaged robot-arm specification is
`src/pzr/rtlola/specs/robot_arm.lola`. It and the twelve trace CSVs were imported
from RLolaEval revision `2257d074173a6dd475c042ef9a82cd8755a81ac3`;
the expected specification SHA-256 is
`aab5b768d872bc4f5b6dc11b96805c2d451cc5c91eb573225f6b0e246cee6acc`.
Do not substitute an older local robot-arm specification.

The required native stack is:

- binding revision `01c92a2bfac58755e3b832bb0094816f3f36e1d1`;
- interpreter revision `2724b05ae6c62ed0df14f1401ed8db89472725a6`;
- a `maturin build --release`/release-profile binding.

`src/pzr/rtlola/binding.py` rejects a mismatched interpreter or debug build.
The current interpreter exposes logical all-zero dynamic rows for state export,
while binding-native transformations still reduce compact nonzero rows.
Negative coefficients are not zero and must remain represented. Python code
must not depend on stable generator row positions or dense matrix shapes; use
the binding-native transforms, counters, and approximation loss. PZR reports
the compact reducer dimension separately from the exported logical row count,
and budget checks must use the compact reducer dimension.

The last recorded full release-binding validation after the Pairwise Ranking Policy
cleanup and bounded-exploration integration was 129 passing tests with no skips on
2026-07-19.

The authoritative trace kinds and full lengths are:

- all four `figure8` variants: 2,340 events each;
- `random`: 1,495 events;
- `random_drift`: 1,433 events;
- `random_geofence`: 1,063 events;
- `random_drift_geofence`: 1,105 events;
- all four `square` variants: 1,983 events each.

Each path family has compliant, `_drift`, `_geofence`, and `_drift_geofence`
conditions. Drift adds progressive tool-center drift; geofence conditions add
progressive path rotation against waypoint-derived walls. Do not pool them
without preserving `trace_kind`.

The emitted MPC methods are:

- `mpc_terminal_beam`: multi-action beam search, terminal loss only;
- `mpc_cumulative_beam`: global beam search with undiscounted cumulative
  explicit-horizon loss and no tail;
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
completed twelve-trace evaluation.

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
history, spectral statistics, or an inference-time preview rollout.

Pairwise Ranking Policy is the primary paper-facing learned method. The
versioned experiment in `experiments/terminal_loss_paper_v1.yaml` pre-generates
26 independent 500-event nominal random-waypoint traces and collects one
terminal full-width teacher dataset at budgets `40,80,120,150,200,250,500`.
Clean teacher train/validation seeds are 0--19/20--25. The primary model uses
all seven budgets; `pairwise_ranking_policy_budget80` is separately trained by
filtering that dataset to recorded budget-80 samples and is only an
extrapolation diagnostic.

The Phase 1 cleanup reset all prior learning result directories. There is no
active primary, secondary, exploratory, or terminal-loss paper result artifact.
New claims require the versioned 224-cell figure-8 headline and 5,040-cell
held-out manifests, with every failed point explicitly unavailable.

Soft-KL and guarded DART remain completed secondary ablations and are not part
of the default wrapper. Their historical result artifact was removed during the
schema reset; the observed improvement was marginal and is confounded by
additional data.
DART calibration uses the frozen Pairwise Ranking Policy model's tolerance-aware clean-
validation errors, fits a smoothed teacher-action-conditioned direction kernel,
targets the global per-budget novice-error rate, restricts alternatives to the
Q90 normalized-regret radius, and forces one teacher recovery decision after
every disturbance.

The bounded exploration generates extra seeds 26--41 and collects those same 16
traces as clean teacher and guarded-DART train-only datasets. It screens
`pairwise_ranking_policy_clean20`, `pairwise_ranking_policy_clean36`, `pairwise_ranking_policy_dart36`, and
`expected_regret_clean20` with Girard over all twelve fixed traces: 240 cells.
Explicit comparisons are `data_scale`, `dart_effect`, and `objective`. At most
one passing challenger receives a 144-cell full
evaluation with its matched reference and Girard. Do not claim exploratory
results until the screen, selection, and any required promotion manifests
validate.

The four figure-8 headline traces always retain their full authoritative lengths.
Held-out generalization uses seeds 100--119 under all four random-waypoint
conditions at 500 events. Pilot seeds are 90--91, ablation seeds are 60--64,
and reserved exploration/model-selection seeds remain 26--41.
Teacher shards use ten spawned workers. Post-reference evaluation cells use
four spawned workers with `max_tasks_per_child=1`; each worker owns its monitor
and planner. BLAS, OpenMP, MKL, and NumExpr remain limited to one thread per worker.

The primary objective is tolerance-aware state-balanced pairwise ranking. Feasible
cost gaps within `max(1e-15, 1e-9 * max(abs(cost_i), abs(cost_j)))` are ties;
meaningful pair weights are divided by the largest meaningful gap in the state,
and every feasible candidate ranks above every infeasible candidate. Scores are
lower-is-better and uncalibrated.

The `expected-regret-v1` challenger uses feasible normalized regret targets in
`[0,1]`, an infeasible target of `2.0`, candidate-mean MSE within each state, and
an equal mean over states with a feasible action. It uses no hyperparameter
grid, selects checkpoints by clean-validation regression loss, and does not
clamp predictions. Preserve stable ranking, native feasibility retries, and
interval fallback accounting.

`budget` is the binding transform bound. Never subtract a fresh-generator
reserve or interpret post-event dense slots as a violation. Preserve the
distinction between dynamic, active, zero, and constant generators.

MPC and teacher costs use binding-native approximation loss. Terminal beam and
the two-event full-width teacher use terminal loss. Cumulative beam is a
matched offline comparison and never the primary method; experimental tail variants use either the extended
endpoint loss or an undiscounted sum of binding-native state losses. Do not
replace these with width, trigger-straddling, or a Python proxy during
unrelated cleanup.

Benchmark reference mode controls offline metrics and caching only. Paper MPC
and teacher searches construct their own unreduced horizon rollouts. Offline
terminal beam uses recorded future inputs; predictive linear beam uses causal
history only.

Offline exact references remain specification-independent. Each cache row
contains exact trigger booleans, a shared logical-row center, and separate
dynamic and total-state radii. The engine reconstructs an interval matrix and
invokes the existing native `approx_loss` while the candidate is applied only to the planner monitor. It
must restore the planner in `finally` and must never mutate the live monitor.
Do not edit the RTLola binding to implement these metrics.

The binding exposes affine verdict intervals and volume-ratio methods. Affine
verdict intervals are supported. Volume methods remain available only as an
upstream diagnostic and must not be used in objectives, reports, caches, or
learning targets.

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
