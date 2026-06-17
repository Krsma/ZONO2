# Robotics Scenario Spike

## Goal

Find a robotics-flavored scenario that can carry the paper motivation better
than the current robot-arm animation, while still producing nontrivial
zonotope-reduction behavior: frequent reductions, visible trigger geometry,
distinct reducer choices, and clean soundness/budget accounting.

## Current Robot-Arm Visualization Status

The robot-arm visualization should remain a sanity/debug artifact. After the
split-panel renderer and paper-trace repair, the result is still not good enough
for the paper motivation. Keep a note to revisit it later, but do not treat the
current robot-arm animation as the main explanatory figure.

Its current failure is mostly visual and trace-level:

- the MuJoCo controller barely moves on the existing benchmark and paper traces;
- the trigger zonotope can grow much larger than the arm workspace;
- plotting the arm and full trigger hull on one axis makes the arm unreadable.

Any future repaired artifact should use split coordinates:

- physical arm panel: workspace-scale true arm, measured arm/point,
  end-effector trail, and end-effector trigger region;
- trigger-space panel: Cartesian trigger zonotope and interval hull, zoomed
  around the trigger region with numeric annotation when the full hull is
  clipped;
- timeline panel: trigger width, generator count, budget, and reduction points.

The paper-oriented robot-arm trace is scripted kinematics, not a quantitative
benchmark distribution. It uses the same sensor model, monitor, reducer policy,
budget, and horizon, but it exists only to explain monitor-side uncertainty.
Use `--trace benchmark` for MuJoCo sanity checks.

## Candidate A: safe-control-gym / Crazyflie

This is the best-aligned candidate for the main paper story. The existing
framing notes already identify safe-control-gym IROS gate flying as the CoRL
headline setting: a Crazyflie-like quadrotor, gate/corridor/obstacle geometry,
bounded noisy observations, and monitor-triggered fallback.

Why it fits:

- drone/gate-flying is clearly robotics-native;
- derived streams can be semantically rich: obstacle clearance, gate deviation,
  corridor deviation, altitude margins, speed, and safety margin;
- these streams are coupled functions of pose/velocity, so reducers should
  differ more than on simple position-only examples;
- intervention metrics are paper-facing: spurious interventions, fallback
  duration, missed violations, gates passed, completion.

Risks:

- the current `src/pzr` tree does not contain the `pzr.robotics.*` modules
  referenced by the notes, so this is not a small animation edit;
- safe-control-gym and firmware support require sidecar infrastructure;
- calibration must avoid saturated triggers where every reducer looks the same.

Spike checks before implementation:

- verify or recover the sidecar interface described in `science/CORL_RUNBOOK.md`;
- run one short fixed-controller trace and record pose, velocity, gate target,
  obstacle geometry, and fallback labels;
- build an offline monitor prototype over saved traces before closing the loop;
- compare static reducers on trigger width and intervention labels at one
  budget and length.

Recommendation: make this the preferred main-body scenario if the sidecar can
be restored without major dependency churn.

## Candidate B: F1TENTH / RoboRacer

F1TENTH is the strongest car-flavored alternative. The official Gym interface
exposes racing maps, pose, velocity, yaw rate, LiDAR scans, and collision
flags. That gives a natural monitor story around wall clearance and speed near
track boundaries.

Candidate streams:

- lateral corridor deviation from centerline;
- heading error relative to the local track tangent;
- minimum LiDAR clearance in front/side sectors;
- time-to-collision proxy from speed and front clearance;
- speed envelope conditioned on curvature;
- collision-risk margin.

Why it fits:

- racing is visually intuitive;
- LiDAR sector reductions and track geometry should produce coupled,
  trigger-relevant uncertainty;
- false positives have an understandable downstream effect: unnecessary
  slowing or fallback braking.

Risks:

- this is a new dependency and monitor design from scratch;
- LiDAR uncertainty is high-dimensional, so the first monitor should aggregate
  sectors before constructing zonotopes;
- centerline/map tooling must be deterministic and testable.

Spike checks before implementation:

- install/run a minimal F1TENTH Gym rollout outside benchmark defaults;
- save trace records with pose, velocity, yaw rate, action, scan sectors,
  collision, and map name;
- prototype a low-dimensional monitor over sector minima plus pose/velocity;
- check reducer differentiation on trigger width and false-positive labels.

Recommendation: keep as a backup or parallel exploratory track if
safe-control-gym recovery stalls.

## Candidate C: Improved MuJoCo Point-Mass

The existing point-mass is useful for renderer prototyping but too simple as a
main paper setting unless the monitor is made richer.

Possible upgrade:

- use obstacle-clearance and time-to-collision streams, not only boundary
  x/y triggers;
- add corridor or waypoint-following constraints;
- visualize true point, measured point, trigger zonotope, obstacle/corridor
  regions, and fallback decisions.

Recommendation: deprecated from default/headline sweeps. It remains explicitly
runnable for regression checks, but long rollout checks tie best static across
tested budgets and therefore crowd out more informative evaluations.

## Deprecated Simple Baseline Status

`simple_robot` is also deprecated from default/headline sweeps. It remains
explicitly runnable for tests and historical comparisons, but long rollout
checks tie best static across tested budgets. Treat it as a regression fixture,
not a paper evaluation environment.

## Robot Arm Rescue Status

Robot arm is not deprecated. Current evidence says the legacy Girard-base
`mpc_rollout` policy is mismatched to the robot-arm monitor: long traces make
it worse than the best fixed Scott/MethA static reducers, with higher
false-positive rates. However, short high-K all-method sweeps did show small
positive gains from broad exact `mpc_sequence` at some budgets. Keep robot arm
as a diagnostic and rescue candidate, but do not use legacy rollout as its
headline policy without additional tuning.

## Decision Criteria

Commit to a new main-body scenario only after a short diagnostic run shows:

- reductions occur often but do not saturate every trigger;
- at least three static reducers have measurably different trigger widths or
  intervention labels;
- predictive methods improve a paper-facing metric, not only an internal width;
- all methods have zero budget violations and zero unsound certificates;
- the visualization is legible without special pleading.

## Implemented Probe Path

`python -m pzr.experiments.robotics_probe` is an audit path, not a benchmark.
It writes `probe_metadata.json`, `method_scores.csv`,
`method_score_summary.csv`, `candidate_scores.csv`,
`candidate_score_summary.csv`, `trace_summary.csv`, candidate raw JSONL
traces, derived-stream CSVs, timeseries CSVs, and `candidate_report.md`.
Single-seed runs write candidate artifacts at the output root. Multi-seed runs
write per-seed artifacts under `seed_N/` and top-level aggregate CSVs.

The probe scores a candidate by:

- trigger-width spread across static reducers;
- number of differentiated reducers;
- reduction frequency;
- near-threshold trace fraction;
- oracle violation balance when available;
- budget and soundness failures.

`drone` currently performs a sidecar preflight and then runs a
safe-control-gym Level0 rollout when `--trace-source live` or `auto` succeeds.
The collector runs outside the main package in `tools/collect_safe_control_drone_trace.py`.
If live collection is unavailable in `auto`, the probe falls back to the
Level0 geometry-derived stream proxy.

`f1tenth` uses an isolated sidecar environment created by
`tools/setup_f1tenth_sidecar.sh`. The collector in
`tools/collect_f1tenth_trace.py` runs the official Gym package and creates a
small deterministic corridor map when packaged maps are unavailable.

Initial live results at length 120, budget 10:

- drone: live safe-control-gym run succeeds and is reducer-discriminative
  (`relative_width_spread ~= 1.91`) with zero budget/soundness failures, but is
  currently marked `revise` because oracle violations are too frequent
  (`oracle_violation_fraction ~= 0.875`);
- F1TENTH: live sidecar run succeeds on the generated corridor map and is
  reducer-discriminative (`relative_width_spread ~= 0.91`) with zero
  budget/soundness failures, but is currently marked `revise` because oracle
  violations are saturated (`oracle_violation_fraction ~= 0.975`).

Current recommendation: use the probe to decide between safe-control-gym drone
and F1TENTH before adding either candidate to `default_scenarios()`.

## Tuned Live Probe Results

The tuned command used for the first more serious exploratory pass was:

```bash
python -m pzr.experiments.robotics_probe --candidate all --trace-source live --length 300 --seeds 5 --warmup-steps 30 --budget 10 --f1tenth-sidecar-python external/f1tenth-py38-venv/bin/python --output /tmp/pzr-robotics-tuned-eval
```

Both candidates promoted on all five seeds after trimming the first 30
measurements:

- drone: `relative_width_spread ~= 1.43`,
  `near_threshold_fraction ~= 0.56`, `oracle_violation_fraction ~= 0.36`,
  zero budget violations, and zero unsound certificates;
- F1TENTH: `relative_width_spread ~= 0.86`,
  `near_threshold_fraction = 1.00`, `oracle_violation_fraction ~= 0.14`,
  zero budget violations, and zero unsound certificates.

The static reducers are clearly separated in both probes. Girard, Combastel,
and box currently tie on these low-dimensional stream profiles, while MethA,
Scott, and PCA produce larger mean trigger widths and nonzero false-positive
rates.

Important caveat: the live sidecar trajectories are still effectively
deterministic across seeds. The seed currently changes measurement noise but
not enough trajectory-level behavior to produce different aggregate metrics.
These results support continuing both candidates, but they are not yet a
statistical benchmark distribution.

Best next steps:

- drone: randomize gate/obstacle layouts or controller disturbances, then add
  intervention/fallback metrics tied to gate passage;
- F1TENTH: replace the generated corridor with real or procedurally varied
  maps and compute centerline-relative heading/curvature streams;
- both: add MPC/beam policies after the trace distributions vary enough for
  seed sweeps to mean something.

## Replay Evaluation And Visualization Path

`python -m pzr.experiments.robotics_replay` is the follow-on path for focused
static-vs-MPC evaluation and visualization. It remains outside
`default_scenarios()` and should not be treated as the quantitative paper
benchmark until the trace distributions and trigger projections are stronger.

The `eval` subcommand runs focused static reducers (`girard`, `combastel`,
`methA`, `scott`, `box`) plus MPC methods (`mpc_rollout_scott`, `mpc_beam3`,
`mpc_sequence3`) through the normal runner. It writes benchmark-style
`timeseries.csv`, `summary.csv`, and `aggregate.csv`, plus
`policy_gain.csv`, `winner_by_step.csv`, `trace_summary.csv`,
`trace_metadata.json`, and per-seed derived streams/payload JSONL files.

For F1TENTH, `--monitor physical` is now the preferred replay mode. It tracks
a physical state zonotope over `x, y, theta, speed, yaw_rate`, projects that
state into trigger margins for left/right boundary, heading, time to collision,
curvature-speed, and yaw-rate risk, and varies fresh uncertainty
deterministically with local racing phase. This is more faithful to the
problem we want the paper visualization to motivate than the older stream
monitor, while still staying outside the default quantitative benchmark.

The `render` subcommand creates paper-style artifacts from an eval directory:
first/middle/last stills, storyboard PNG/PDF, optional GIF, and metadata JSON.
The intended first comparison is `scott` versus `mpc_beam3` on the same trace,
with panels for physical path/map, safety margins, trigger width, generator
count, and reduction events.

The `sweep` subcommand is the preferred next ICRA-facing robotics workflow. It
supports `--candidate drone|f1tenth|all`, runs physical stress replay across
generator budgets, uses static reducers plus `mpc_beam3` only by default, and
writes top-level budget summaries:
`budget_sweep_summary.csv`, `budget_policy_gain.csv`,
`budget_reducer_counts.csv`, `budget_runtime.csv`,
`budget_scenario_summary.csv`, `budget_intervention_summary.csv`,
`budget_sweep_metadata.json`, and budget plots under `figures/`. Exact
`mpc_sequence3` remains an optional diagnostic but is intentionally excluded
from the sweep default because current high-K probes show it is much slower
without improving over beam search. Replay metadata records the focused
robotics MPC candidate reducers (`girard`, `combastel`, `scott`) explicitly.

The default procedural replay family is now `--scenario-family stress`.
F1TENTH stress traces strengthen the chicane/bottleneck, speed, heading, and
front-clearance phases. Drone stress traces use a physical
`x, y, z, vx, vy, vz` monitor with seeded gate, obstacle, corridor, altitude,
and speed-risk phases. Use `--scenario-family legacy` only for comparison
against the earlier replay traces.

Current modeling caveat: the physical F1TENTH monitor produces stable
Scott-reference gains and small best-static gains at budget 12, but the
best-static improvement is still not large enough to be the final paper story.
The older stream monitor is even simpler: Girard/box can tie MPC on
interval-width metrics over the derived stream state. Use Scott-reference
visualizations for the current explanatory storyboard and treat
`best_static,mpc_beam3` as a diagnostic until the scenario dynamics are richer.

Example commands:

```bash
python -m pzr.experiments.robotics_replay eval --candidate all --trace-source procedural --monitor physical --scenario-family stress --length 160 --seeds 3 --budget 12 --horizon 4 --beam-width 4 --output /tmp/pzr-robotics-replay-eval
python -m pzr.experiments.robotics_replay sweep --candidate all --trace-source procedural --monitor physical --scenario-family stress --budgets 8,10,12,16,20,24 --length 80 --seeds 2 --horizon 4 --beam-width 4 --output /tmp/pzr-robotics-high-k-sweep
python -m pzr.experiments.robotics_replay render --eval-dir /tmp/pzr-robotics-replay-eval --candidate all --methods scott,mpc_beam3 --output /tmp/pzr-robotics-replay-viz
tools/run_pzr_icra_table_matrix.sh
```

`tools/run_pzr_icra_table_matrix.sh` is the preferred full-table runner. It is
staged and resumable: each robotics/omni budget and horizon cell writes its own
directory plus a `.complete` marker, and the final table exporter scans those
split outputs recursively. The main matrix uses `paper_core` by default so
exact `mpc_sequence3` and regret/ranking distillation do not dominate the
overnight run; those are separate opt-in audit stages.

The current high-K success criterion is `mpc_beam3` beating the best fixed
static reducer, not only a named weak static reference. Preliminary non-paper
probes show the best-static gain is effectively zero around `K=8`, becomes
consistently positive around `K=12`, and improves further for `K=16..24`.
