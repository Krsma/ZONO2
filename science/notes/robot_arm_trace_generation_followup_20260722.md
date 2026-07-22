# Robot-Arm Trace Generation Follow-Up

Status: the preferred trace split was adopted by the versioned paper experiment
on 2026-07-22. Randomized fault-generator redesign remains deferred.

## Preferred evaluation split

The adopted split is:

- train the Pairwise Ranking Policy only on independently generated, nominal
  random-waypoint traces;
- use independently generated nominal random-waypoint traces for the
  non-patterned statistical evaluation;
- use the imported, fixed RLolaEval figure-eight variants for the controlled
  paper running example under nominal, drift, geofence, and combined conditions;
- retain fixed square and canonical-random variants as historical/diagnostic
  assets without expanding the paper matrix.

This separates trajectory diversity from controlled fault presentation. It
also avoids presenting condition-dependent rejection samples as though they
were paired fault variants of one random trajectory. This is now the contract
in `experiments/paper_evaluation_v2.yaml`: pilot, held-out generalization, and
horizon/width ablation generate nominal traces only. The earlier
four-generated-condition design is retained here as superseded history and its
artifacts must not be reinterpreted.

## What matches the pinned RLolaEval source

The implementation in `src/pzr/rtlola/robot_arm_random.py` is an adaptation of
the generator at RLolaEval revision
`2257d074173a6dd475c042ef9a82cd8755a81ac3`. It uses the same MuJoCo model and
substantially the same forward/inverse kinematics, waypoint ordering,
arc-length interpolation, singular-value filtering, tracking-error check, and
drift/geofence construction.

The twelve packaged recorded traces are not regenerated approximations. Their
CSVs are byte-for-byte identical to the files imported from the pinned
RLolaEval revision. They therefore remain the canonical fixed examples.

## Material differences from the notebook

The RLolaEval notebook is a curated-example workflow rather than a demonstrated
multi-seed dataset generator:

- it selects individual example seeds instead of evaluating predetermined seed
  banks;
- its random examples use ten laps and therefore have variable event counts,
  while PZR requests exactly 500 events;
- it uses wall margins of `0.1` for nominal/drift and `0.05` for geofence
  examples, while the current PZR generator uses `0.025` uniformly;
- it allows 100 retries for its nominal example and 1,000 retries for faulted
  examples, while the current PZR configuration allows 100 for every
  condition;
- its condition-specific acceptance filters mean that reusing a seed does not
  generally produce paired nominal and faulted versions of the same accepted
  base trajectory.

The retry difference explains the failed exploratory generation. The pinned
notebook's random-drift example was accepted only on zero-based attempt 504
(the 505th attempt). The failed PZR drift seed exhausted its 100-attempt limit;
the same deterministic sequence can obtain an accepted trace if allowed to
continue. This was a trace-generation policy failure, not an MPC, monitor, or
binding failure.

## Requirements for a future general-purpose generator

If randomized drift and geofence populations become scientifically necessary,
do not merely raise the retry limit and call the result identical to the
notebook. A revised generator should:

1. State that it is adapted from the pinned RLolaEval implementation and record
   every intentional deviation in trace metadata.
2. Separate nominal trajectory generation from fault injection when paired
   comparisons are intended. Generate and accept the base path once, then
   apply each fault condition to that same path and report any condition that
   cannot track it.
3. Define condition-specific, explicit retry limits at least as permissive as
   the notebook where rejection sampling remains necessary. Preflight every
   configured seed before launching an evaluation matrix.
4. Record attempted candidates, rejection reason counts, accepted attempt,
   tracking error, singular-value ratio, wall geometry, fault parameters, and
   generator/source revisions.
5. Fail preparation before evaluation if a required trace is unavailable;
   never substitute another seed or silently relax an acceptance threshold.
6. Test determinism, exact event count, seed/condition identity, paired-path
   identity when requested, and resumability from validated metadata.
7. Characterize acceptance rate and generation cost across the complete seed
   bank so a hand-selected easy seed cannot hide a brittle sampler.

Increasing faulted retries to 1,000 is a reasonable immediate alignment with
the notebook, but by itself it fixes only retry exhaustion. It does not resolve
condition confounding, paired-trajectory semantics, or unpredictable
preparation cost.

## Adopted paper decision and future alternative

The paper uses the preferred split. A future, separately versioned study may
instead choose randomized fault generalization, but only after implementing a
redesigned provenance-rich generator satisfying the requirements above and
publishing fresh experiment artifacts.

Do not reinterpret artifacts produced under the superseded four-condition
generation contract. Regenerate affected manifests under the schema-v3
specification/configuration hash.
