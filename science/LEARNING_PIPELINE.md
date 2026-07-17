# RTLola Soft-Distillation and DART Pipeline

## Policy and Teacher

The learned policy selects Girard, Scott, PCA, or Combastel. `none` remains
automatic while the state is within the transform bound, and `interval` is
fallback-only. The model and dataset artifacts fix the candidate order.

Geometry15 version 2 contains the original twelve budget/current-zonotope
aggregates plus row-width concentration, active-generator norm variation, and
mean normalized off-axis generator mass. The 15-32-32-4 ReLU scorer is strictly
pre-event and emits uncalibrated lower-is-better scores. Inference performs one
forward pass and then tries reducers through the binding in stable score order.

The privileged `mpc_terminal_full_width` teacher evaluates every first reducer
and every required second reducer over the current and next events. Its target
is binding-native terminal approximation loss against an ephemeral unreduced
rollout. Incomplete roots are infeasible. Neither collection nor inference
writes zonotope matrices from Python.

## Soft Action-Value Distillation

For feasible action (a), let (d_a=C_a-\min_b C_b). We set (d_a=0) when
it is within

```text
max(1e-15, 1e-9 * max(abs(C_a), abs(C_min)))
```

of the minimum. We divide the remaining gaps by the largest meaningful gap in
the state. If no meaningful gap exists, every feasible action has zero regret.
The teacher distribution is

```text
q_tau(a | s) = softmax(-normalized_regret(a) / tau)
```

over feasible actions; infeasible actions receive probability zero. The
student distribution is `softmax(-score)`. Each valid state's loss is
`KL(q || p) + lambda * sum(p[infeasible])`, with `lambda=1`, and valid states
are averaged equally. States with no feasible reducer are excluded from the
objective and reported. The same tolerance helper defines tolerant-best masks,
normalized regrets, diagnostics, and the hard pairwise ablation.

We train the soft-clean model at temperatures `0.05,0.1,0.2,0.5` from identical
initialization and data order. Clean validation selects one temperature by
infeasible selections, mean selected normalized regret, maximum regret, KL,
and finally the lower temperature. The soft-DART model reuses this temperature
and trains from scratch. The pairwise-clean model uses the corrected per-state
pairwise objective only as an ablation.

## One-Round Discrete DART

The teacher sees the next event while Geometry15 does not. This asymmetric
information can make exact teacher actions unlearnable from the student input,
as discussed by [Warrington et al.
(2021)](https://proceedings.mlr.press/v139/warrington21a.html) and, in a more
general theoretical setting, [Cai et al.
(2024)](https://papers.nips.cc/paper_files/paper/2024/hash/74d188c51d97fcfbc0269f584d6a53b7-Abstract-Conference.html).
We therefore do not use learner-controlled collection.

Following the supervisor-noise principle of [DART (Laskey et al.,
2017)](https://proceedings.mlr.press/v78/laskey17a.html), we fit
`P(student_action | budget, teacher_action)` by categorical maximum likelihood
on the frozen soft-clean model's clean validation predictions. Unobserved rows
remain the identity distribution. During DART collection, probability assigned
to currently infeasible alternatives is redirected to the teacher action.
Sampling is deterministic in disturbance seed, trace seed, budget, and event
index. At the next reduction decision, the teacher recomputes the complete
cost vector and retakes control. Thus errors never compound through a student
roll-in.

## Data Schedule and Artifacts

The fresh trace store contains 48 independent 500-event nominal
random-waypoint trajectories:

| Dataset | Train | Validation | Collection |
|---|---:|---:|---|
| Clean | 0--19 | 20--25 | teacher |
| DART | 26--41 | 42--47 | frozen disturbed teacher |

The clean validation split is development data: It selects the temperature and
calibrates the confusion kernel, but never contributes gradients. DART
validation is collected with the frozen kernel and also remains outside
training. The six packaged fixed traces are the final out-of-distribution
evaluation.

Version-3 reducer-cost datasets store features, complete cost and feasibility
vectors, split/sample identity, teacher and executed actions, disturbance
metadata, native revisions, trace hashes, and source fingerprints. Targets are
derived during training rather than persisted as tie masks. Model artifacts
store the objective contract, dataset hashes, temperature, feasibility weight,
histories, seed, and source fingerprint. DART calibration stores the empirical
confusion matrix, row counts, diagnostics, and model/dataset hashes.

## Commands and Evaluation

Run the full resumable experiment with:

```bash
PZR_OUT_DIR=results/rtlola-learning-geometry15-random500-soft-dart-v3-7371495-b4cfbf4-e6ecd0b \
PZR_COLLECTION_WORKERS=10 PZR_EVALUATION_WORKERS=4 \
tools/run_rtlola_learning_full.sh
```

The wrapper trains `learned_pairwise_clean`, `learned_soft_clean`, and
`learned_soft_dart`. Repeated named `--dataset NAME=PATH` and
`--model NAME=PATH` inputs require exact schema and catalog alignment. The
`calibrate-dart` command freezes the clean model/dataset identity before any
disturbed shard is collected.

Evaluation compares the three learned models with the four static reducers and
the two-event full-width MPC teacher: six traces by four budgets by eight
methods gives 192 cells. Exact references are prepared once per trace before
parallel cells. Collection uses ten spawned workers; evaluation uses four
spawned workers and `max_tasks_per_child=1`. Every worker owns its binding
monitor, while BLAS, OpenMP, MKL, and NumExpr use one thread.

Reports retain complete time series and summaries, macro loss/width/runtime,
micro FPR/FNR, fallback and infeasible accounting, candidate composition,
`method_comparisons.csv`, independently selected `best_static_metrics.csv`,
and `objective_data_ablation.csv`. Finite extreme static results remain in the
tables. Native failures, non-finite losses, incomplete rows, or missing cells
leave the experiment incomplete. A smoke or prefix run is not a completed
scientific result.
