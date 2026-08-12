# DART Rescue Experiment

## Question

The paper evaluation exposes unstable Pairwise Ranking Policy trajectories at
budgets 40, 80, 120, and 500. We test whether guarded DART reduces these tails,
and whether any improvement comes from DART rather than from adding sixteen
training trajectories.

## Controlled Comparison

We train three exact-budget policy families:

| Policy | Training trajectories |
| --- | --- |
| Clean20 | clean seeds 0--19 |
| Clean36 | clean seeds 0--19 and clean seeds 26--41 |
| DART36 | clean seeds 0--19 and guarded-DART seeds 26--41 |

All models use clean validation seeds 20--25, Geometry15, the pairwise
objective, and the paper's fixed hyperparameters. We calibrate DART separately
for each budget from the corresponding frozen Clean20 specialist. The
calibration targets its meaningful validation error rate, restricts
disturbances to the Q90 normalized-regret radius, and forces one teacher
recovery decision.

The primary comparison is DART36 against Clean36. Clean36 against Clean20
isolates data scale, whereas DART36 against Clean20 records the total rescue.
We do not tune a promotion threshold after observing the replay.

## Evaluation

We keep the evaluation roles explicit:

- *Retrospective replay* uses nominal random-waypoint seeds 100--119. These
  trajectories motivated DART and cannot provide fresh confirmatory evidence.
- *Untouched confirmation* uses nominal random-waypoint seeds 120--139.
- *Controlled patterned cases* use the four pinned full-length figure-eight
  traces. They are not fault-population replicates.

The report contains 924 policy cells: 420 replay cells, 420 confirmation cells,
and 84 fixed-case cells. It executes 756 new cells because it imports verified
Clean20 replay and fixed-case results. Any failed run makes its main aggregate
unavailable.

Nominal effects use seed-aligned 10,000-replicate paired bootstrap intervals.
We report FPR differences, geometric loss ratios, mean and quantile loss,
complete ECDFs, fallback rates, and reducer composition. Fixed cases remain
descriptive and are never pooled with generated nominal traces.

## Execution

Run the disposable smoke workflow with:

```bash
tools/run_dart_rescue.sh run --smoke
```

Run or resume the complete study with:

```bash
tools/run_dart_rescue.sh run
```

The canonical output namespace is `results/dart-rescue-v1`. Every stage records
the experiment config, runner, PZR source, native revisions, input artifacts,
models, traces, and exact-reference hashes. Stale artifacts are rejected rather
than reinterpreted.
