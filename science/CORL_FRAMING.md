# CoRL Framing: Robotics Interventions, Regret Distillation, and CORA Parity

## Claim Scope

The CoRL-facing claim should be operational before it is algorithmic: predictive
zonotope reduction reduces avoidable monitor-triggered fallback interventions in
robotics tasks while preserving certified safety-relevant monitor state. The
preliminary experiments can use a Python monitor that mirrors the intended
RTLola stream semantics. Direct RTLola and ROS integration are follow-up
engineering milestones once the intervention metric is strong enough to justify
the integration cost.

## IROS Gate-Flying Experiment

The headline robotics benchmark is the safe-control-gym IROS gate-flying task.
The initial adapter is intentionally optional and should only load the external
repository when `PZR_SAFE_CONTROL_GYM_ROOT` points at a checkout of the
`beta-iros-competition` branch. The internal experiment boundary is:

- true simulator state and exact obstacle/gate geometry define oracle safety;
- noisy bounded observations feed the monitor and reducer policy;
- monitor triggers switch the controller from nominal gate-following to fallback
  control;
- all monitor state reductions remain certified reducers.

Report fallback activation count, fallback duration, spurious interventions,
justified interventions, missed violations, collisions, constraint violations,
gates passed, task completion, time-to-target, reducer latency, reducer choices,
budget violations, and unsound certificates. Keep trigger width, interval-hull
MSE, width inflation, generator count, and MPC sequence counts as diagnostics.

## Simulated RTLola Monitor

The v1 monitor is a Python implementation of the RTLola-equivalent stream
semantics. Inputs are noisy pose, velocity, target gate, obstacle geometry, and
command/reference state. Derived streams are obstacle clearance, gate deviation,
corridor deviation, altitude low/high margins, speed, and safety margin.
Triggers cover collision risk, obstacle clearance, corridor deviation, altitude
band, and speed envelope violations.

The monitor should remain easy to compare with, or export to, an RTLola
specification later. That means deterministic stream names, explicit threshold
parameters, and no hidden controller-specific side effects in predicate logic.

## Learning Story

One-shot distillation is a useful baseline but should not be the main learning
story. Use regret/ranking distillation:

1. Roll out the current learned selector on training seeds.
2. Query the MPC/oracle teacher for all candidate first-action costs at
   learner-induced over-budget states.
3. Aggregate normalized per-candidate regrets with previous data.
4. Retrain a ranker that predicts one regret score per reducer.
5. Validate on held-out closed-loop rollouts, not only row-wise accuracy.

The trusted boundary stays unchanged: learned policies only rank reducers;
certified reducers still gate all monitor state changes.

## CORA Parity

Reducer baselines must be cross-checked against CORA before they are used for a
paper-critical claim. CORA dispatches `girard`, `combastel`, `pca`, `methA`,
`scott`, and `sadraddini` through `contSet/@zonotope/reduce.m`; its budget
argument is an order, while this repository uses an absolute generator budget.
Validation therefore translates `budget` to `budget / dimension` before calling
CORA, then compares enclosure, generator count, interval widths, hull volume
proxy, and deterministic generator matrices up to column permutation where
appropriate.

If differences are material, either revise the Python implementation to match
CORA exactly for paper baselines or relabel the reducers as CORA-style and use
CORA-generated artifacts for the paper-critical comparison.
