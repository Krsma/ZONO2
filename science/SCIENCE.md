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
state is already within the bound. It is not an optimization candidate.
The unbounded interval transform is fallback-only.

## Objective

Beam MPC first rolls out an unreduced reference over the prediction horizon.
Candidate terminal states are scored by the binding’s
`approx_loss_state(reference, candidate)`. Widths, generator counts, public
stream bounds, and trigger outcomes remain diagnostics.

The benchmark reference mode is independent of this search reference.
`verdict` mode avoids retaining full offline state histories, but MPC still
constructs an unreduced terminal reference for each horizon search.

Ranking distillation uses a two-event, full-width teacher. Each training row
forces one candidate first action and exhaustively evaluates required second
actions against an ephemeral unreduced rollout using binding-native terminal
loss. Direct learned inference uses aggregate current-state features only,
ranks candidates once, and tries binding actions in stable score order.

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
