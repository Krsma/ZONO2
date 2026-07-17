# Scientific Contract

## Research Question

At a fixed RTLola transform bound, can predictive or learned selection among
certified native zonotope transforms reduce monitor approximation loss relative
to a fixed transform?

## Execution Semantics

RTLola owns the monitor state and applies a selected `ZonotopeConfig` before
accepting the next event. The configured `budget` is passed unchanged as that
transform bound. Event evaluation may subsequently allocate fresh slack, so a
post-event dense dynamic count above the bound is expected and is recorded
separately.

`state_zonotope(False)` is interpreted as dynamic state uncertainty.
`state_zonotope(True)` also contains constant slack such as robot-arm
calibration uncertainty. Dense columns include zero holes; active and zero
counts are therefore reported alongside the dense count.

## Trusted Boundary

Search and learned policies are untrusted selectors. Every committed state
change is performed by an RTLola binding transform. There is no Python-side
reducer or matrix writeback path.

The no-op action is used for exact reference execution and while the pre-event
state is already within the bound. It is not an optimization candidate. The
unbounded interval transform is fallback-only.

## Objective

Beam MPC first rolls out an unreduced reference over the prediction horizon.
Candidate terminal states are scored by the binding's
`approx_loss_state(reference, candidate)`. Widths, generator counts, public
stream bounds, and trigger outcomes remain diagnostics.

The learning teacher exhaustively evaluates the current and next events for
every feasible first reducer. It labels each visited state with the complete
binding-native cost vector. Geometry15 inference remains strictly pre-event:
the scorer observes only current-zonotope aggregates, emits one lower-is-better
score per reducer, and tries binding actions in stable score order.

The primary target is a temperature-controlled distribution derived from
tolerance-aware normalized regret. Training minimizes state-balanced KL
divergence and explicitly penalizes student probability on infeasible actions.
The scores are not calibrated regrets. The corrected hard pairwise loss remains
only as a clean-data ablation.

## Asymmetric Learning and DART

The two-event teacher has privileged information that Geometry15 cannot
observe. Warrington et al., [*Robust Asymmetric Learning in
POMDPs*](https://proceedings.mlr.press/v139/warrington21a.html), show why an
asymmetric expert can prescribe actions that are unsuitable for a partially
observed student. Cai et al., [*Provable Partially Observable Reinforcement
Learning with Privileged
Information*](https://papers.nips.cc/paper_files/paper/2024/hash/74d188c51d97fcfbc0269f584d6a53b7-Abstract-Conference.html),
formalize related failure modes for expert distillation in general POMDPs.
These results motivate our design, but they do not constitute a theorem about
this RTLola experiment.

Student-controlled DAgger rollouts compound this information mismatch.
Instead, we adapt Laskey et al.'s [*DART: Noise Injection for Robust Imitation
Learning*](https://proceedings.mlr.press/v78/laskey17a.html) to the discrete
reducer catalog: We estimate a categorical novice-confusion kernel on held-out
clean trajectories, occasionally execute one feasible disturbed teacher
action, and return control to the teacher at the next decision. This is a
one-round discrete DART adaptation, not a claim that the original continuous
noise model applies unchanged.

## Scenarios

- `omni_robot`: stochastic seeded acceleration/direction traces with one
  persistent constant calibration variable and fresh measurement uncertainty.
- `robot_arm`: six deterministic recorded 5-DOF RLolaEval traces evaluated by
  the packaged RTLola forward kinematics, running-average drift detector,
  alpha-beta observer, and safe-stopping geofence. Five constant calibration
  variables remain outside dynamic reduction.

Robot-arm trigger decisions are sparse `Trigger#0` through `Trigger#4`
verdicts. Python treats absence as false and does not approximate trigger
logic.
