# RTLola Pairwise Ranking Policy and Bounded Method Exploration

## Proposed Policy and Teacher

We use *Pairwise Ranking Policy* as the sole paper-facing learned method. The policy
selects Girard, Scott, PCA, or Combastel; `none` remains automatic while the
state is within the binding transform bound, and `interval` is fallback-only.
The model and dataset artifacts fix the candidate order.

Geometry15 version 2 contains fifteen pre-event aggregates: the original twelve
budget/current-zonotope features, row-width concentration, active-generator
norm variation, and mean normalized off-axis generator mass. The 15-32-32-4
ReLU scorer emits one raw lower-is-better score per candidate. Inference performs
one forward pass and then tries binding reducers in stable score order.

The privileged `mpc_terminal_full_width` teacher evaluates every first reducer
and every required second reducer over the current and next events. Its target
is the binding-native terminal approximation loss against an ephemeral
unreduced rollout. Incomplete roots are infeasible. Neither collection nor
inference writes zonotope matrices from Python.

## Pairwise Ranking Policy Objective

For two feasible actions with costs (C_i) and (C_j), we treat their costs as
tied when their gap is at most

```text
max(1e-15, 1e-9 * max(abs(C_i), abs(C_j))).
```

Every meaningful feasible pair contributes a softplus ranking loss weighted by
its gap divided by the largest meaningful feasible gap in that state. Every
feasible action ranks above every infeasible action with unit weight. We first
normalize the weighted pair loss within each rankable state and then average
states equally. Raw scores remain uncalibrated; only their stable ordering is
used at inference time.

## Primary Data and Evaluation

The primary trace store contains 26 independent 500-event nominal
random-waypoint trajectories:

| Dataset | Train | Validation | Collection |
|---|---:|---:|---|
| Primary clean | 0--19 | 20--25 | teacher |

Validation trajectories never contribute gradients. The wrapper trains only
`pairwise_ranking_policy` across all seven paper budgets and separately trains
`pairwise_ranking_policy_budget80` from the recorded budget-80 subset. The
all-budget policy is the only primary learned method; the budget-80 model is an
extrapolation diagnostic.

Exact references are prepared once per trace. Teacher collection uses ten
spawned workers; pilot, headline, objective comparison, and generalization use
four spawned workers with `max_tasks_per_child=1`. Ablation and timing are
sequential so their reported throughput is not contaminated by experiment-worker
contention. Every worker owns its binding monitor, while BLAS, OpenMP, MKL, and
NumExpr remain limited to one native thread.

Run or resume the complete paper evaluation with:

```bash
tools/run_paper_evaluation.sh run
```

The output must be fresh and source-aware. We do not claim a revised result
until the 224-cell headline and 5,040-cell held-out manifests validate, with
every failed point explicitly unavailable.

## Secondary Soft-KL and Guarded DART Ablations

Soft action-value distillation and guarded one-round discrete DART remain in
the codebase as completed secondary ablations. They are not stages of the
primary wrapper. Their historical result artifact was removed in the schema
reset and is not an active result.

Soft-KL converts tolerance-aware feasible regrets to
`softmax(-regret / temperature)`, assigns infeasible candidates zero target
probability, and penalizes student probability on infeasible actions. Guarded
DART calibrates the global per-budget disturbance rate from the frozen Pairwise
Clean model's clean-validation errors. It separately fits a smoothed teacher-
action-conditioned direction kernel, restricts alternatives to the Q90
normalized-regret radius, and forces one teacher recovery decision after every
disturbance.

The guarded calibration corrected a harmful feedback mechanism in the earlier
adaptation: shifted corrective states could otherwise trigger repeated
disturbances. However, the observed improvement from the corrected DART run was
marginal and is confounded by its additional training traces. We therefore keep
it as an ablation rather than presenting it as the proposed method.

## Expected-Regret Challenger

The exploratory *expected-regret-v1* objective uses the same Geometry15 scorer.
For each state with at least one feasible action, we assign feasible candidates
their tolerance-aware normalized regret in ([0,1]) and infeasible candidates a
fixed target of (2.0). We compute mean squared error across candidates within
each state and then average valid states equally. States with no feasible action
are excluded and reported.

This regression has no hyperparameter grid and uses clean-validation regression
loss for checkpoint selection. Outputs are not clamped: they estimate the
conditional mean penalized regret when repeated Geometry15 observations alias
different teacher states. Diagnostics report RMSE, MAE, per-candidate errors,
and predictions outside ([0,2]), alongside selection regret, feasibility, and
ranking metrics.

## Historical Bounded Exploration

The earlier proposed exploration would have reused the primary clean dataset
and generated seeds 26--41 for Clean36, guarded DART36, and expected-regret
challengers. That workflow is no longer an active entry point: the learning
method decision is settled for the current paper and its wrapper was removed.
The seeds remain reserved for a future explicitly versioned study.

We screen four models:

| Model | Training data | Objective |
|---|---|---|
| `pairwise_ranking_policy_clean20` | frozen primary clean | pairwise |
| `pairwise_ranking_policy_clean36` | primary clean + 16 extra clean | pairwise |
| `pairwise_ranking_policy_dart36` | primary clean + 16 extra DART | pairwise |
| `expected_regret_clean20` | primary clean | expected regret |

The old promotion thresholds and model definitions remain here only as
scientific history; they do not authorize claims or a new run.

## Artifacts

Version-5 reducer-cost datasets store Geometry15 features, complete native cost
and feasibility vectors, split/sample identity, teacher summaries, native
revisions, trace hashes, and source fingerprints. Clean datasets contain no
synthetic disturbance columns. DART datasets additionally store optional
per-decision disturbance metadata and `dart_collection_summary.csv`.
Model directories store the objective contract, dataset hashes, validation
provenance, histories, seed, weights, normalizer, and source fingerprint.

Evaluation reports retain complete time series and summaries, macro
loss/width/runtime, micro FPR/FNR, fallback and infeasible accounting, candidate
composition, `method_comparisons.csv`, and independently selected
`best_static_metrics.csv`. The low-level evaluator uses schema
`pzr.policy-evaluation.v2`. Version 2 records the configured predictor, fixed
schedule, horizon, beam width, and exact-reference identity. The paper pipeline
uses its stricter source-aware cell schema. Predictive MPC is causal but
current-event-aware; the learned policy remains strictly pre-event.

Exploratory runs additionally write validated
`policy_comparisons.csv`, `challenger_assessments.csv`, and `selection.json`.
The primary run omits empty policy-comparison tables and plots.
