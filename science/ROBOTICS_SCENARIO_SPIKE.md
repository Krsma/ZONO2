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

Recommendation: use only as a fallback visualization/debug scaffold.

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
It writes `probe_metadata.json`, `method_scores.csv`, `trace_summary.csv`,
candidate timeseries CSVs, and `candidate_report.md`.

The probe scores a candidate by:

- trigger-width spread across static reducers;
- number of differentiated reducers;
- reduction frequency;
- near-threshold trace fraction;
- oracle violation balance when available;
- budget and soundness failures.

`drone` currently performs a sidecar preflight and then runs a
safe-control-gym Level0 geometry-derived stream proxy. This checks whether the
monitor framing is reducer-discriminative before rebuilding the old missing
CoRL modules.

`f1tenth` currently records dependency status if `f110_gym` is unavailable. If
the dependency is present, it runs the same low-dimensional derived-stream probe
for the proposed F1TENTH monitor design. A promotion step should replace that
proxy with live Gym rollout traces.

Current recommendation: use the probe to decide between safe-control-gym drone
and F1TENTH before adding either candidate to `default_scenarios()`.
