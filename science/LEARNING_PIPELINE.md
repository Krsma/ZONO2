# RTLola Learning Pipeline

## Scope

The learned policy selects one binding-native reducer from a fixed catalog.
The catalog is part of the model artifact and must match exactly at collection,
training, aggregation, and inference. `none` is automatic while the current
state is within the transform bound; `interval` is fallback-only. The current
ordinary catalog is Girard, Scott, PCA, and Combastel. Interval hull and
deterministic clustering are explicit diagnostics rather than learning
candidates.

Inference is intentionally non-predictive. It reads the current zonotope and
the configured binding transform bound, evaluates one PyTorch MLP, stably ranks
the configured candidates, and tries them through the binding in that order.
It does not read future events, run MPC, mutate matrices, or use exact caches.

## Features And Targets

Feature schema `rtlola.current-zonotope` version 2 contains 15 scalars:

1. budget;
2. dense dynamic generator count;
3. active dynamic generator count;
4. compact reducer dimension;
5. logical dynamic dimension;
6. generator overflow ratio;
7. zero dynamic fraction;
8. dynamic state width;
9. maximum row width;
10. mean active generator norm;
11. maximum-to-mean active generator norm;
12. mean absolute coupling between active generators;
13. maximum-row-width share of total dynamic width;
14. coefficient of variation of active generator L2 norms;
15. mean normalized off-axis generator mass `(L1 - Linf) / L1`.

Features use aggregate quantities only. They do not assume stable dense row or
column positions, and state width excludes constant calibration slack. The
policy is strictly pre-event: it does not use raw stream values, history,
centers, spectral statistics, or a post-event preview branch.

The teacher exhaustively evaluates every configured first action and every
required second action over the current and next event. Its terminal target is
binding-native approximation loss against an ephemeral two-event `none`
rollout. The full exact trajectory is never precomputed for teacher labels.
Incomplete roots are masked infeasible; equal-cost best candidates are retained
in an explicit tie mask. The 15-32-32-K ReLU model is trained with a
cost-weighted pairwise ranking loss, where lower output scores are better.

## Data And Splits

Robot-arm learning traces adapt RLolaEval revision
`e6ecd0b2f60263e0a4270bd76a71cd9c90e685e5` random-waypoint generation. The
primary experiment uses nominal random-waypoint motion only. Drift and
geofence generators remain available for explicit diagnostics, but they are
not part of this training distribution. Each trace records its seed,
condition, generator configuration, source revision, trace hash, and MuJoCo
diagnostics.

Splits are made by trajectory seed before budgets are expanded. All budgets for
a trajectory remain in the same split. The Geometry15 experiment uses forty
independent 500-event traces: seeds 0--11 train, seeds 12--15 validate early
stopping, seeds 16--27 form the first DAgger round, and seeds 28--39 form the
second. DAgger shards are training-only. The six packaged fixed traces retain
their full authoritative lengths and form the final out-of-distribution
evaluation rather than a generated test split.

The complete trace store is generated and hash-validated before any teacher
rollout begins. Collection only reads this store and checkpoints each
split/condition/seed/budget shard. Reuse requires matching trace-store and
trace hashes, candidates, features, behavior-model hash, PZR source, and native
revisions. Explicitly empty under-bound shards are valid, while the
consolidated training dataset must be non-empty.

## Staged Commands

The complete resumable run is:

```bash
PZR_OUT_DIR=results/rtlola-learning-geometry15-random500-7371495-b4cfbf4-e6ecd0b \
  PZR_WORKERS=8 \
  tools/run_rtlola_learning_full.sh
```

Trace generation writes one shared, resumable store of inspectable CSVs,
per-trace metadata, hashes, and a versioned manifest. Collection consumes that
store and writes aligned compressed arrays, sample rows, long-form candidate
costs, and a versioned dataset manifest.
Missing teacher shards execute in isolated worker processes and are reloaded
in deterministic split/seed/budget order. Exact reference caches are prepared
before missing evaluation cells execute in isolated workers. Worker count is
an execution setting, not part of either scientific artifact identity; each
worker constructs its own binding monitor and planner state. Native BLAS and
OpenMP thread counts remain one to avoid oversubscription.
Training writes `weights.pt`, `model.json`, `training.json`, and grouped
validation metrics. Evaluation runs Girard, `learned_geometry15`, and the
teacher-matched `mpc_terminal_full_width` in fingerprinted trace/budget/method
cells. It writes exact-metric time series, macro loss/width summaries,
micro-pooled FPR/FNR, reduction-conditioned candidate composition, comparison
tables, plots, reference caches, and a manifest. ONNX is not exported
automatically; an explicit export command may be added after the model contract
stabilizes.

## Evaluation Contract

Girard and full-width MPC use the same budget and trace as the learned policy.
`none` is an exact diagnostic reference, not a learned candidate. Full-length
tables report FPR, FNR, mean/final/max/summed native loss, state width, runtime,
fallback and infeasible rates, and reducer selection. Ranking accuracy,
feasible selection, and chosen regret are evaluated on the held-out validation
seed. Automatic `none` steps are separated from action composition on
reduction-required steps. No prefix or timing smoke is a completed six-trace
result.
