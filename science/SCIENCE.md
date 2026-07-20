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

Our proposed learned method is Pairwise Ranking Policy. Its tolerance-aware objective
ranks feasible reducers by binding-native cost, ranks every feasible reducer
above every infeasible reducer, and averages normalized pair losses equally
across rankable states. The primary experiment uses twenty clean teacher traces
for training and six seed-disjoint clean traces for validation.

We retain two secondary objectives for controlled ablations. Soft-KL distills a
temperature-controlled action distribution. Expected-regret regression assigns
feasible candidates normalized regret in `[0,1]`, assigns infeasible candidates
the target `2.0`, and estimates conditional mean penalized regret with
state-balanced mean squared error. Only Pairwise Ranking Policy appears in the primary
evaluation.

## Secondary Asymmetric Learning and DART Ablation

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
reducer catalog. We calibrate a global per-budget noise magnitude from the
pairwise novice's held-out meaningful error, while a smoothed categorical
kernel models the error direction. We restrict perturbations to the Q90
novice-regret radius and force the next reduction decision to use the teacher.
This design preserves DART's supervisor-noise and recovery mechanism without a
learner roll-in. However, it is a guarded one-round discrete adaptation, not a
claim that the original continuous noise model applies unchanged.

The preceding v3 implementation coupled magnitude and direction in each
teacher-action row. The resulting feedback was harmful: shifted states made
rare corrective teacher actions likely to be overwritten again, so nominally
one-step disturbances formed long runs. The corrected calibration explicitly
prevents this behavior and records target, expected, and realized rates.

The completed guarded-DART evaluation remains a secondary ablation. Its
observed improvement was marginal and is confounded by additional training
data. Consequently, the default pipeline does not train Soft-KL or DART models.
Its historical artifact, together with every prior learning result directory,
was removed by the Phase 1 schema reset and is not an active result.

## Evaluation Claims

The primary completion criterion is 144 validated cells: six authoritative
traces, four transform bounds, Pairwise Ranking Policy, four static reducers, and the
two-event MPC teacher. A separate 60-cell screen compares data scale, guarded
DART, and expected-regret challengers under explicit matched references. At
most one challenger can proceed to a 72-cell full evaluation.

These matrices define experimental contracts, not results. We do not claim new
primary or exploratory findings before the corresponding source-aware manifests
validate every required cell without native failures or non-finite artifacts.
In particular, no active learning result artifact exists after the Phase 1
cleanup; a new Pairwise Ranking Policy claim requires the canonical 144 cells.

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
