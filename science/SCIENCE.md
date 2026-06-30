# Predictive Zonotope Reduction: Science Notes

## Motivation

Runtime monitors for uncertain sensor streams can represent affine uncertainty
as a zonotope

```text
Z = c + G[-1, 1]^m
```

where `c` stores nominal stream values and the columns of `G` store symbolic
slack-variable coefficients. Persistent calibration uncertainty reuses
generators across time; independent measurement noise introduces fresh
generators. Fresh generators make exact monitor state grow without bound, so a
long-running monitor needs sound order reduction.

The project treats reduction as a policy decision over certified operators:
static, predictive, and learned selectors may choose different reducers, but
the reducer certificate is what preserves soundness.

## Soundness Contract

For every reduction action:

```text
Z subseteq Reduce(Z, action)
generator_count(Reduce(Z, action)) <= K
```

Prediction quality, MPC search width, feature quality, and learned rankings
can affect precision and runtime. They do not justify soundness.

The executable boundary in the current code is:

- `src/pzr/zonotope/`: `Zonotope`, certified reducers, metrics, and
  `ProtectedReducer`;
- `src/pzr/monitoring/`: monitor state and trigger contracts;
- `src/pzr/mpc/`: search policies over certified reduction actions;
- `src/pzr/imitation/`: learned ranking policies that still select certified
  reducers;
- `src/pzr/experiments/`: benchmark execution, summaries, figures, robotics
  replay, and tables;
- `src/pzr/rtlola/`: optional RTLola-native monitor integration.

### Policy-Independent Soundness

Assume a monitor step is sound before reduction. If every chosen reducer returns
a certified enclosure `Z'` with `Z subseteq Z'`, then the reduced execution
over-approximates the unreduced execution no matter how the policy selected the
reducer. This remains true for heuristic, predictive, learned, or adversarial
selectors.

### Bounded-Memory Invariant

If the monitor reduces whenever the generator count exceeds `K`, and every
admissible reducer returns at most `K` generators, then every stored
post-reduction state satisfies the memory bound.

The RTLola-native path uses a different but explicit invariant. There,
`budget` is the bound passed to `ZonotopeConfig.<method>(budget)`. RTLola
applies the transform before accepting the next event; event evaluation may
then allocate fresh slack, so the post-event dense slot count can exceed
`budget`. This is not a budget violation in RTLola-native experiments. It is
reported separately as `post_event_over_bound`.

### Protected Generators

Monitors with persistent calibration generators use `ProtectedReducer` or
`reduce_with_protection`. Protected columns are copied exactly before the
residual zonotope is reduced, and calibration indices are renumbered to the
front after reduction. If protected columns alone exceed the budget, the
reduction must fail rather than silently drop required state.

## Trigger Semantics

Monitors expose `trigger_zonotope(state) -> Zonotope`. Most monitors return
`state.zonotope`; `RobotArmMonitor` projects the 6D joint state into a 2D
Cartesian end-effector zonotope.

Any trigger-bound, trigger-width, trigger-straddling, or trigger-feature code
must use `monitor.trigger_zonotope(state)`. Generator-count and memory costs
remain costs on the raw monitor state.

Ground-truth comparison runs the same trace without reductions and compares
reduced trigger-zonotope bounds against unreduced trigger-zonotope bounds.

## Current Reducer And Policy Surface

Static protected reducers:

- `girard`
- `combastel`
- `pca`
- `methA`
- `scott`
- `box`

Focused predictive methods include:

- `mpc_rollout`
- `mpc_rollout_methA`
- `mpc_rollout_scott`
- `mpc_pair_rollout3`
- `mpc_sequence`
- `mpc_sequence3`
- `mpc_beam3`

The top-3 focused MPC set is intentionally `girard`, `methA`, and `scott` in
the main benchmark path. Robotics replay currently records its focused
candidate set as the same `girard`, `methA`, and `scott` top-3 set.
Robotics replay method sets are scoped separately from the main benchmark:
`focused` includes focused static reducers plus `mpc_rollout_scott`,
`mpc_beam3`, and `mpc_sequence3`; `sweep` is the budget-sweep default and
uses only `mpc_beam3` as the predictive method; `headline` includes exact
`mpc_sequence3`; and `paper_core` omits that exact audit for practical runs.

`IdentityReducer` is a theory/test primitive and should not enter default
benchmark, MPC, or learned candidate sets unless no-op experiments are
explicitly reopened.

## Benchmark Outputs

Main benchmark runs write per-scenario:

- `timeseries.csv`
- `summary.csv`
- `aggregate.csv`

and top-level:

- `config.yaml`
- figure PDFs under `figures/`
- learning artifacts under `learning/<scenario>/` when regret/ranking
  distillation is enabled.

Tracked metrics include trigger width, exact unreduced trigger width,
approximation error, false-positive rate, generator count, reduction timing,
chosen reducers, MPC search leaves, pruned branches, budget violations, and
unsound certificates.

Robotics replay is outside `default_scenarios()` and writes additional replay
artifacts documented in `science/EXPERIMENT_READINESS.md`.

Paper-table generation is implemented in `pzr.experiments.paper_tables` /
`pzr-paper-tables`. It scans benchmark `aggregate.csv` files and robotics
`budget_sweep_summary.csv` files, normalizes them into
`combined_summary.csv`, and writes LaTeX fragments:

- `main_k_sweep.tex`
- `horizon_sweep.tex`
- `full_methods_h4.tex`
- `distillation.tex`
- `overview.tex`

The current ICRA matrix wrapper runs this table builder after the primary
budget sweeps, horizon sweeps, optional exact-sequence audit, and optional
regret/ranking stage.

## RTLola Integration

The optional RTLola path uses `rlolapythonbinding` snapshots to support branch
planning:

```text
live RTLola monitor       -> committed execution
planner RTLola monitor    -> accept_event_from_state branches
EvaluatorState snapshots  -> reusable branch roots
```

PZR currently selects RTLola-native `ZonotopeConfig` transforms. The binding
does not currently expose arbitrary reduced matrix writeback, so native PZR
reducers are not injected into the RTLola evaluator.

In RTLola result artifacts, `generator_count` is the dense post-event dynamic
slot count from `state_zonotope(False)`. Active mathematical generator count is
reported separately as `active_dynamic_generator_count`; zero dense columns are
reported as `zero_dynamic_generator_count`. Use these fields to distinguish
RTLola memory-slot behavior from nonzero zonotope support.

RTLola `mpc_beam` now uses the binding's approximation-loss metric as its
optimization signal. At each branch root it rolls out an unreduced `none`
reference over the same finite horizon and scores each candidate terminal state
with `approx_loss_state(reference_terminal, candidate_terminal)` through the
binding. This keeps the online MPC objective aligned with the offline exact
metric: `--reference-mode exact` reports binding-native `approx_loss` against
the unreduced ground-truth state, while `--reference-mode off` leaves that
exact-run column unset even though MPC still uses finite-horizon reference loss
internally. Relevant-width columns remain diagnostics, not the current MPC
objective.

Previous robot-arm RTLola diagnostics showed that Scott-heavy `mpc_beam`
behavior was not caused by Girard infeasibility after the transform-bound
cleanup: at budget 160, Girard, Scott, interval hull, and PCA all ran at every
reduction point. That diagnosis was made under the older short-horizon
relevant-width objective. The binding-loss objective should be evaluated before
carrying over conclusions about Scott dominance, objective quality, or
trajectory dependence.

See `science/RTLOLA_INTEGRATION_NOTES.md` for setup and validation.

For the RTLola robot-arm path, use `tools/setup_robot_arm_env.sh` and
`tools/run_rtlola_robot_arm.sh`. This dedicated environment installs the
RTLola binding and MuJoCO without `safety-gymnasium`, avoiding the older
pygame/Gymnasium dependency conflicts that can downgrade the Python 3.11
robot-arm stack.

## Learning

Regret/ranking distillation trains a policy from per-candidate MPC cost tables
rather than hard expert labels. The learned policy ranks reducers and remains
outside the trusted boundary; the selected certified reducer still performs the
state mutation.

Learned rows should be treated as paper evidence only when held-out metrics,
chosen-action regret, second-best margins, and ranking-collapse diagnostics are
healthy.

For future RTLola-native learning, teacher costs should be aligned with the
binding-native finite-horizon reference-loss objective rather than the older
hand-rolled relevant-width proxy.

## Paper Framing

The concise claim:

> Predictive certified reducer selection improves bounded-memory monitor
> precision, or downstream intervention quality, while preserving formal
> soundness because all monitor state mutations are certified reductions.

The work is closest to a trusted-boundary pattern: arbitrary policy proposes,
certified operator preserves an invariant. Related-work positioning is tracked
in `paper/related_work_foundation.md`.

## Current Scope

Implemented now:

- Python zonotope reducers and protected reduction.
- Math-only and MuJoCo-backed monitor adapters.
- Static, rollout, sequence, pair-rollout, and beam MPC selectors.
- Regret/ranking learned selector paths.
- Robotics probe/replay diagnostics.
- Reproducible ICRA-style robotics/omni budget and horizon matrix wrappers.
- Paper-table normalization from benchmark and robotics replay artifacts.
- Optional RTLola-native benchmark path.
- Dedicated RTLola/MuJoCO robot-arm environment wrapper.

Not yet current active infrastructure:

- Arbitrary native PZR reducer writeback into RTLola.
- Old `pzr-run-corl` / `pzr.experiments.corl_*` commands.
- A final accepted paper-critical robotics result; current candidates should
  still be judged through the ICRA matrix and readiness criteria.
