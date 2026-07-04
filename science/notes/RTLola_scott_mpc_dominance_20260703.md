# Scott dominance in RTLola MPC: current evidence and hypotheses

Date: 2026-07-03

## Scope

This note records the current investigation into why `scott` dominates the
actions selected by RTLola MPC even though a static Scott policy becomes
catastrophically wide at higher generator budgets.

The completed classification sweep used:

- robot-arm `figure8_drift`, 2,340 events, one seed;
- budgets 40, 80, 120, and 180;
- MPC horizon 4 (current event plus four future events), beam width 4;
- binding-native terminal approximation loss;
- verdict reference mode.

Verdict mode retains exact trigger booleans but not the unreduced state history,
so it cannot report full-trace exact approximation loss.

## Observed classification and width behavior

MPC and Girard both had zero FPR and zero FNR at every completed budget.
Static PCA had 90.52--96.37% FPR. Static Scott had zero FPR at budget 40 and
53.13% FPR at budgets 80, 120, and 180.

The temporal mean of the aggregate internal-state width was:

| Budget | MPC | Girard | Scott | PCA |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 25.353 | 25.555 | 25.330 | 89.786 |
| 80 | 25.325 | 25.539 | 1.743e13 | 58.961 |
| 120 | 25.316 | 25.542 | 1.132e13 | 55.020 |
| 180 | 25.314 | 25.523 | 3.815e12 | 47.584 |

Here "aggregate width" is

```text
sum_i 2 * sum_j abs(G[i, j])
```

over all scalar rows of the dynamic RTLola state zonotope. It is not a
distance in metres, the width of one robot coordinate, or a generator-budget
violation.

For static Scott, the instantaneous aggregate width first exceeded 100 at:

| Budget | Step | Scott width | Approximate exact width |
| ---: | ---: | ---: | ---: |
| 80 | 132 | 101.95 | 2.94 |
| 120 | 148 | 102.37 | not retained in verdict mode |
| 180 | 201 | 102.44 | not retained in verdict mode |

This divergence precedes Scott's first fallback at these budgets by roughly
one thousand steps. Fallback is therefore not the cause of the initial
widening.

## MPC action composition

Executed reducer counts over 2,340 events:

| Budget | None | Girard | Scott | Interval hull | PCA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 2 | 26 | 2,177 | 0 | 135 |
| 80 | 3 | 40 | 2,192 | 0 | 105 |
| 120 | 7 | 44 | 2,258 | 1 | 30 |
| 180 | 6 | 51 | 2,161 | 0 | 122 |

Thus MPC executes Scott on 92.35--96.50% of all events, but the small number
of Girard/PCA switches is sufficient to avoid the static Scott trajectory in
this trace.

## Receding-horizon deferral is present

Scott is more common at the first planned position than at the terminal
position. Girard shows the opposite pattern:

| Budget | Scott at position 0 | Scott at position 4 | Girard at position 0 | Girard at position 4 |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 93.10% | 81.83% | 1.11% | 13.67% |
| 80 | 93.78% | 84.70% | 1.71% | 11.06% |
| 120 | 96.78% | 79.13% | 1.89% | 20.22% |
| 180 | 92.58% | 69.66% | 2.19% | 24.68% |

At budget 180, 641 Scott-first plans scheduled a non-Scott action at position
4. After replanning on each intervening event, 584 of those position-4 events
actually executed Scott. The previously suspected "Scott now, Girard later"
procrastination pattern therefore still exists.

This observation does not by itself show that procrastination causes the good
or bad results. The horizon-sensitivity diagnostic below contradicts that
simple explanation.

## Short exact-reference diagnostic

A 300-event budget-80 run retained the global unreduced state and compared it
with each committed reduced state:

| Policy | Mean exact loss | Mean aggregate absolute error | Scott uses |
| --- | ---: | ---: | ---: |
| Girard | 3.502e-4 | 0.17095 | 0 |
| Static Scott | 7.499e4 | 989.14 | 297 |
| MPC, horizon 0 | 4.704e-7 | 0.00645 | 256 |
| MPC, horizon 1 | 5.240e-7 | 0.00704 | 260 |
| MPC, horizon 2 | 8.180e-7 | 0.00888 | 262 |
| MPC, horizon 4 | 9.138e-7 | 0.00903 | 269 |
| MPC, horizon 8 | 8.832e-7 | 0.00929 | 264 |

The greedy horizon-0 selector was best on this short exact study and still
interleaved 41 Girard actions with 256 Scott actions (plus three initial
no-op events). This demonstrates that:

1. a mostly-Scott adaptive mixture can have genuinely low global exact loss;
2. static Scott's failure does not imply that every individual Scott action
   is poor;
3. the horizon-4 result is not currently better than a much cheaper one-step
   selector.

These are single-trace, single-seed, 300-event results and are not sufficient
for a general conclusion.

## What the Scott transform does

For `G` with `n` rows and `k` active generators, the pinned implementation
uses pivoted Gauss--Jordan elimination to write the reordered support as

```text
G = T [I R].
```

For each redundant column `r` selected for removal, it minimizes the estimated
added-volume cost

```text
product_i (1 + abs(r_i)) - (1 + sum_i abs(r_i))
```

and updates

```text
T <- T (I + diag(abs(r)))
R <- diag(1 / (1 + abs(r))) R_remaining.
```

It finally returns `[T, T R]`. The retained generators remain represented,
while the selected shared uncertainty is absorbed as independent enlargement
along the basis directions in `T`. The transform is an outer approximation.
It fails if the active support does not have full row rank.

## Confirmed properties and ruled-out explanations

### Confirmed

- Reference and candidate states are compared at the same logical rollout
  depth. No temporal row-shift was found.
- The binding-native loss is the mean squared difference between row-wise
  interval-hull endpoints.
- The loss is insensitive to generator-column alignment and cross-row
  dependency when two zonotopes have the same row-wise interval hull.
- MPC constructs its unreduced horizon reference from the current, already
  approximated state. It scores incremental future approximation, not all
  historic error from the beginning of the trace.
- Offline exact mode separately compares the committed trajectory against a
  globally unreduced trajectory. The short exact results above therefore rule
  out a reporting-only explanation.
- Receding-horizon action deferral occurs, but horizon 0 performed better than
  longer horizons in the current short study.

### Ruled out as the initial cause

- Scott fallback does not initiate the widening.
- The large Scott width is not a physical robot-arm distance.
- Static Scott's poor result is not evidence that MPC secretly executed
  Girard most of the time; MPC genuinely executed Scott on most events.

## Current hypotheses

1. **Sparse corrective switching.** Scott has low incremental loss on most
   states, while occasional Girard or PCA actions prevent the geometric state
   from entering the unstable static-Scott trajectory. This is strongly
   supported by the short exact run, but the mechanism of the correction has
   not yet been isolated.

2. **Correlation-blind loss.** Scott can replace one shared generator by
   independent enlargement in its selected basis. The native interval-hull
   loss may rate that step highly even when it loses dependencies that later
   computations amplify. This is a plausible mechanism, not yet a measured
   causal result for this trace.

3. **Local-reference blindness.** Because every MPC reference rollout starts
   from the current approximate state, accumulated historic error is sunk and
   cannot influence the action score. This can permit locally cheap actions
   that would be poor as a stationary policy.

4. **Receding-horizon procrastination.** The planner often places Girard later
   in a sequence and then replans to Scott before that action is reached. This
   is directly observed. However, horizon 0's better exact result means it is
   not a sufficient explanation for Scott dominance or MPC stability.

5. **Budget-dependent Scott geometry.** Scott's pivot basis, removal sequence,
   and future generator geometry change with the target bound. Its sequential
   behavior is not guaranteed to be monotone in the budget. This may explain
   why budget 40 is stable while larger budgets diverge, but it remains to be
   tested.

## Required follow-up experiments

1. Compare horizon 0 and horizon 4 over all full robot-arm traces in verdict
   mode, recording FPR/FNR, actions, failures, and runtime.
2. Run short exact-reference sweeps over multiple traces, budgets, and seeds
   for horizons 0, 1, 2, 4, and 8.
3. At selected states, force each first action and record both its predicted
   local terminal loss and its subsequently realized global exact loss.
4. Add read-only diagnostics for generator coupling, singular values,
   conditioning of Scott's selected basis, and before/after interval widths.
   Do not add Python-side state mutation or reducers.
5. Compare normal replanning against an explicit committed-action-sequence
   experiment to measure procrastination directly.
6. Treat cumulative native stage loss or a switching penalty as separate
   experimental objectives. Do not silently replace the canonical
   binding-native terminal objective.

## Current conclusion

Static Scott is not the best policy. A predominantly Scott adaptive mixture is
excellent on the short exact study and stable on the completed drift
classification sweep, but there is no demonstrated advantage for horizon-4
MPC over greedy horizon-0 selection. The strongest unresolved question is
whether sparse switches repair dependency damage that the native interval
loss cannot observe directly.
