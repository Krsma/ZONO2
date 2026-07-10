# RTLola Learning Pipeline

## Scope

The learned policy selects one binding-native reducer from a fixed catalog.
The catalog is part of the model artifact and must match exactly at collection,
training, aggregation, and inference. `none` is automatic while the current
state is within the transform bound; `interval` is fallback-only.

Inference is intentionally non-predictive. It reads the current zonotope and
the configured binding transform bound, evaluates one PyTorch MLP, stably ranks
the configured candidates, and tries them through the binding in that order.
It does not read future events, run MPC, mutate matrices, or use exact caches.

## Features And Targets

Feature schema `rtlola.current-zonotope` version 1 contains 12 scalars:

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
12. mean absolute coupling between active generators.

Features use aggregate quantities only. They do not assume stable dense row or
column positions, and state width excludes constant calibration slack.

The teacher exhaustively evaluates every configured first action and every
required second action over the current and next event. Its terminal target is
binding-native approximation loss against an ephemeral two-event `none`
rollout. The full exact trajectory is never precomputed for teacher labels.
Incomplete roots are masked infeasible; equal-cost best candidates are retained
in an explicit tie mask. The 12-32-32-K ReLU model is trained with a
cost-weighted pairwise ranking loss, where lower output scores are better.

## Data And Splits

Robot-arm learning traces adapt RLolaEval revision
`e6ecd0b2f60263e0a4270bd76a71cd9c90e685e5` random-waypoint generation. The
four fixed conditions are nominal, drift, geofence interaction, and combined
drift/geofence interaction. Each trace records its seed, condition, generator
configuration, source revision, trace hash, and MuJoCo diagnostics.

Splits are made by trajectory seed before budgets are expanded. All budgets for
a trajectory remain in the same split. Base teacher collection requires
non-empty train, validation, and test splits. A learned-behavior aggregation
round contains training trajectories only; validation and test data remain
teacher-collected and held out.

## Staged Commands

Base collection:

```bash
pzr-learning collect --output /tmp/pzr-learning/base --event-count 200 \
  --budgets 40,80,120,180 --train-seeds 8 --validation-seeds 2 --test-seeds 2
```

Initial training:

```bash
pzr-learning train --dataset /tmp/pzr-learning/base/dataset \
  --output /tmp/pzr-learning/model-initial
```

One learned-behavior aggregation round and final training:

```bash
pzr-learning collect --output /tmp/pzr-learning/dagger --event-count 200 \
  --budgets 40,80,120,180 --train-seeds 8 --validation-seeds 0 --test-seeds 0 \
  --behavior-model /tmp/pzr-learning/model-initial

pzr-learning train --dataset /tmp/pzr-learning/base/dataset \
  --dataset /tmp/pzr-learning/dagger/dataset \
  --output /tmp/pzr-learning/model-final
```

Full-length generalization evaluation always uses exact references and the six
packaged fixed traces:

```bash
pzr-learning evaluate --model /tmp/pzr-learning/model-final \
  --budgets 40,80,120,180 --output /tmp/pzr-learning/evaluation
```

Collection writes inspectable trace CSVs and metadata, aligned compressed
arrays, sample rows, long-form candidate costs, and a versioned manifest.
Training writes `weights.pt`, `model.json`, and `training.json`. Evaluation
writes exact-metric time series and summaries, candidate-selection counts,
failures, reference caches, and a manifest. ONNX is not exported automatically;
an explicit export command may be added after the model contract stabilizes.

## Evaluation Contract

Static and MPC baselines use the same budget and trace as the learned policy.
`none` is an exact diagnostic reference, not a learned candidate. Full-length
tables report FPR, FNR, mean/final/max/summed native loss, state width, runtime,
fallback and infeasible rates, and reducer selection. Ranking accuracy,
teacher agreement, and chosen regret are evaluated on held-out generated data.
No current smoke artifact is a completed six-trace result.
