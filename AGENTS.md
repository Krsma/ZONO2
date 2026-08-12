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

tools/run_paper_evaluation.sh evaluate --smoke

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

The ordinary release-validation command must pass with no skips. Paper
evaluation preflight records its own pass count and zero skips after excluding
tests marked `rlola_parity`; notebook parity remains a standalone development
diagnostic and is not a paper prerequisite.

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

The paper-facing MPC methods are:

- `mpc_terminal_beam`: multi-action beam search, terminal loss only;
- `mpc_terminal_beam_predictive_linear`: causal linear prediction with the
  terminal objective;
- `mpc_cumulative_beam`: global beam search with undiscounted cumulative
  explicit-horizon loss, used only for objective comparison;
- `mpc_terminal_full_width`: exhaustive two-event terminal-loss teacher.

The canonical wrapper is `tools/run_paper_evaluation.sh`. It performs the
non-parity release preflight, training, pilot gating, all scientific matrices,
reporting, and validation. It prepares one exact reference per trace and uses
source-aware resumable cells. The H/W ablation uses one experiment worker so
its diagnostic throughput is contention-free; headline and generalization use ten
workers. Dedicated timing is deferred from the main `evaluate` command. Full
RLolaEval notebook parity remains available through
`pzr-rtlola-parity` but is never invoked or required by paper stages.
`tools/run_paper_evaluation.sh explore` is the preliminary entrypoint: it runs
only release preflight, verified Clean148 specialist import, and the formal
pilot. It does not run paper-scale matrices or the historical
bounded-exploration study.

The v4 finalization is a standalone orchestration layer defined by
`experiments/paper_evaluation_v4.yaml` and
`tools/run_paper_evaluation_v4.sh`; it must not modify `src/pzr` or any existing
result directory. It uses nominal seeds 348--367 for all ten methods and writes
only below `results/paper-evaluation-v4`. Its default `run` command ends after
`prediction-ablation` and must never start workstation timing. The explicit
`runtime`, `report`, and `validate` commands remain diagnostic-only because
they retain their workstation-timing dependency.

The independently versioned Pi add-on is
`experiments/paper_evaluation_v4_pi_timing_v1.yaml`, with wrapper
`tools/run_paper_evaluation_v4_pi_timing_v1.sh`. It consumes a frozen,
hash-verified v4 parent and writes only to
`results/paper-evaluation-v4-pi-timing-v1` and
`results/paper-evaluation-v4-final-report-v1`. Workstation v4 timing is a
separately labelled diagnostic once the Pi is the paper-facing runtime source;
never pool the two timing distributions or copy Pi measurements into the v4
parent.

## Current Robot-Arm Results

The current completed canonical paper artifact is
`results/paper-evaluation-v3`; `results/paper-evaluation-v4` becomes the
scientific parent only after its required matrices validate. The completed
Clean20 artifact at `results/paper-evaluation-v2` remains a preserved, loadable
historical contract and must never be reinterpreted as v3 or v4. Earlier
robot-arm, four-budget learning, MPC-tail, and bounded-exploration outputs are
historical and must not be quoted as the current evaluation. The separate DART
rescue study belongs under `results/dart-rescue-v1` and cannot silently replace
the canonical paper artifact.

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
versioned experiment in `experiments/paper_evaluation_v3.yaml` uses the frozen
optimizer-seed-42 Clean148 matrix selected by the completed scaling study.
Training trajectories are nominal random waypoints at seeds 0--19, 26--41, and
200--311; validation remains exclusively seeds 20--25. Every trace has 500
events and terminal full-width teacher rows at budgets
`40,80,120,150,200,250,500`. One fixed-hyperparameter specialist is trained
from only its matching budget rows and dispatched only at that budget. The
paper pipeline verifies and copies the seven frozen models by hash rather than
retraining or recollecting them. Joint- and cross-budget learned policies are
deferred follow-up work.

The canonical paper result requires the versioned 224-cell fixed figure-8
headline and 1,120-cell nominal held-out manifests, with every failed point
explicitly unavailable.

Soft-KL and guarded DART remain completed secondary ablations and are not part
of the default wrapper. Their historical result artifact was removed during the
schema reset; the observed improvement was marginal and is confounded by
additional data.
DART calibration uses the frozen Pairwise Ranking Policy model's tolerance-aware clean-
validation errors, fits a smoothed teacher-action-conditioned direction kernel,
targets the global per-budget novice-error rate, restricts alternatives to the
Q90 normalized-regret radius, and forces one teacher recovery decision after
every disturbance.

The former Clean20/Clean36/DART36/expected-regret promotion workflow remains
historical. The new `dart-rescue-v1` study is narrower: it compares exact-budget
Clean20, Clean36, and guarded-DART36 specialists without expected-regret or an
automatic promotion gate. It uses seeds 26--41 for paired additional clean and
DART training trajectories, replays seeds 100--119 retrospectively, and
reserves untouched seeds 120--139 for confirmation.

The canonical DART wrapper is `tools/run_dart_rescue.sh`. It validates the
existing teacher dataset and Clean20 models, fits seven budget-specific DART
calibrations, collects paired Clean36/DART36 data, trains fourteen new
specialists, evaluates 924 reported cells, and writes failure-aware tables and
figures. Its orchestration stays outside `src/pzr` so it does not invalidate the
completed paper cells merely by adding follow-up reporting logic.

The four fixed figure-8 headline traces always retain their full authoritative
lengths and pinned RLolaEval hashes. They are controlled patterned case studies,
not multi-seed fault-population estimates. Held-out generalization uses generated
nominal random-waypoint seeds 100--119 at 500 events; pilot seeds 90--91 and
ablation seeds 60--64 are nominal-only. No randomized drift/geofence
generalization claim is made. Reserved exploration/model-selection seeds are
312--327 for the v3 contract. Historical studies retain their recorded seed
roles. Headline, pilot, objective comparison, and generalization use ten spawned
workers with `max_tasks_per_child=1`; each
worker owns its monitor and planner. Ablation and timing are sequential. BLAS,
OpenMP, MKL, and NumExpr remain limited to one thread per worker.

The primary objective is tolerance-aware state-balanced pairwise ranking. Feasible
cost gaps within `max(1e-15, 1e-9 * max(abs(cost_i), abs(cost_j)))` are ties;
meaningful pair weights are divided by the largest meaningful gap in the state,
and every feasible candidate ranks above every infeasible candidate. Scores are
lower-is-better and uncalibrated.

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
