# Predictive Zonotope Reduction: Science Notes

## Paper Motivation

The project builds on "Cutting Corners on Uncertainty: Zonotope
Abstractions for Stream-based Runtime Monitoring" (arXiv:2601.11358v1).
The paper observes that runtime monitors for uncertain sensor streams can
track calibration and measurement noise with affine arithmetic. A monitor
state then becomes a vector of affine forms. The affine coefficients form a
zonotope:

```text
Z = c + G[-1, 1]^m
```

The center `c` stores nominal stream values and generator columns in `G`
store symbolic slack-variable coefficients. Constant calibration error is
represented by a reused symbolic variable, while per-measurement noise creates
fresh variables. Fresh variables make `m` grow with trace length, so any
trace-length independent monitor must reduce or unify generators.

## Core Soundness Contract

The monitor remains sound if every reduction satisfies:

```text
Z subseteq Reduce(Z, action)
generator_count(Reduce(Z, action)) <= K
```

Prediction, scoring, and optimization are allowed to be approximate. They
choose among certified reductions; they do not justify soundness. This is the
main separation used in the codebase:

- `pzr.core` stores zonotopes and reduction certificates.
- `pzr.reduction` implements reducers that must return sound certificates.
- `pzr.control` selects certified reducers using static or receding-horizon
  policies.
- `pzr.monitoring` defines a black-box monitor adapter boundary.

Generator metadata is part of this contract. Calibration, measurement,
synthetic, and unknown generators are tagged in `pzr.core.zonotope`, and a
monitor can declare `GeneratorRequirement` patterns for generators that must
survive reduction exactly. `ProtectedReducer` enforces those requirements by
splitting required generators from the residual zonotope before delegating to a
certified reducer.

The executable theory obligations can be stated as the following proof notes.

**Policy-independent soundness theorem.** Fix a monitor transition relation
`Step` that is sound over zonotope states. Let a policy choose, at each
reduction point, any candidate reducer whose certificate establishes
`Z subseteq Z'`. Then the reduced execution over-approximates the unreduced
execution regardless of how the policy predicted future inputs or scored the
candidates. Prediction quality can affect precision and runtime, but it is not
part of the soundness proof.

**Bounded-memory invariant.** If the monitor applies a certified reduction
whenever `generator_count(Z) > K`, and every admissible reducer returns
`generator_count(Z') <= K`, then every stored post-reduction monitor state has
at most `K` generators. The explicit `no_reduction` reducer is kept as a
future-work primitive and succeeds only when `generator_count(Z) <= K`, but it
is not part of the current paper experiment candidate sets.

**Protected-generator preservation lemma.** Suppose the monitor declares a set
of generator metadata requirements and `ProtectedReducer` is used with
`require_existing=True`. If the number of matching required generator columns
is at most `K`, then those columns are copied exactly into the reduced state
before the residual zonotope is reduced. If the required columns alone exceed
`K`, the protected reducer fails rather than silently dropping monitor-required
state.

**Finite-horizon optimality limit.** `SequenceMPCPolicy` is optimal only over
the finite tree induced by the supplied candidate reducers, the supplied
predicted inputs, the configured horizon, and the configured cost. It is not a
global optimality result for the real trace. `RolloutMPCPolicy` is intentionally
more approximate: it optimizes the first action over a candidate set, then uses
a fixed protected rollout reducer and fallback for future predicted overflows.
Both policies replan after each real monitor step, and both inherit soundness
only from certified first actions.

**Robust prediction gate.** Robust or tube-aware predictors should be added only
if the existing online-vs-oracle artifacts show a meaningful gap on precision
or false-alarm metrics. The current TACAS-main artifacts do not yet show a large
or consistent oracle advantage, so the constant-input online predictor and
oracle ablation are enough for now to separate prediction quality from certified
reduction soundness.

## Control-Theoretic Project Idea

The paper identifies two limitations of existing zonotope approximations for
monitoring:

1. They optimize the current enclosure and ignore how future monitor updates
   propagate approximation error.
2. They often treat calibration and per-sample measurement error similarly,
   even though calibration error reappears in future steps.

This project treats reduction as an abstraction-control action. At a reduction
point, a policy predicts short-horizon monitor evolution, scores candidate
certified reductions, applies only the first chosen action, and repeats at the
next step. This is model-predictive control in structure, but the "control
input" is a compression decision rather than a physical action.

The current benchmark supports both the deployable online predictor and an
oracle predictor used for ablations. Online prediction extrapolates from the
observed history, while oracle prediction uses the held-out future trace. The
CLI can run either mode or both modes in one artifact set with
`--predictor-mode online`, `--predictor-mode oracle`, or
`--predictor-mode both`. Dynamic policy calls and decision-feature rows are
emitted only at real over-budget reduction points; budgeted no-op decisions
are not part of the current experimental design.

The latest full artifact examined in this checkout is `results/tacas-main`.
Its manifest records a paper-profile run overridden to `length=300`,
`budget=8`, `horizon=6`, `seeds=50`, `predictor_mode=both`, and
`method_set=paper_plus_wide` over the `robot`, `robot_simple`, and
`thermostat` scenarios. The aggregate diagnostics are clean:
`budget_violation_count = 0`, `unsound_certificate_count = 0`,
`reduction_failure_count = 0`, and `no_op_count = 0`. The strongest empirical
claim from this artifact is focused rollout precision on the hard robot
monitor; broad rollout is a useful ablation, not the headline method, and the
learned selector is best described as a cheap certified selector approximation.

## Code Mapping

The benchmark suite contains two Python robot monitors inspired by the paper.
The harder `robot` scenario is the omnidirectional example. The black-box
adapter in `pzr.benchmarks.robot` tracks:

- filtered acceleration,
- velocity,
- current distance increment,
- x position,
- y position.

The `robot_simple` scenario is the paper's two-axis motivating example. It
tracks measured velocity, filtered velocity, and position for x/y, including
end-stop resets that zero one coordinate and its generator row. The simple
monitor introduces two persistent calibration generators (`delta_x`,
`delta_y`) and two fresh measurement generators per step.

It exposes only the `MonitorAdapter` methods and trigger metadata. The
controller can step, clone, and replace the zonotope component of the state,
but it does not inspect equations or depend on monitor internals.

The `iros` scenario is the CoRL headline benchmark. It tracks obstacle
clearance, gate deviation, corridor deviation, altitude margins, speed, and
safety margin for a Crazyflie quadrotor flying through gates under bounded
sensor noise. It has one persistent calibration generator and fresh measurement
generators per step. The `InterventionManager` closes the loop between
monitor verdicts and controller fallback. Code: `pzr.robotics.iros`,
`pzr.robotics.safe_control_gym`.

The learning pipeline (`pzr.learning`) provides DAgger-based policy
distillation against MPC experts. `pzr.learning.features` extracts 19 numeric
decision features at each reduction point. `pzr.learning.dagger` implements
the on-policy aggregation loop. `pzr.learning.policy` wraps trained
checkpoints as `LearnedReductionPolicy`.

The CoRL experiment suite (`pzr.experiments.corl_suite`) orchestrates
preflight, calibration, DAgger training, and heldout evaluation for the IROS
gate-flying task. It runs safe-control-gym through a sidecar subprocess
interface (`pzr.robotics.safe_control_gym.SidecarSafeControlGymClient`).

The `thermostat` scenario is the first non-robot family. It tracks room
temperature, filtered temperature, heating/cooling effort, and comfort
deviation from the setpoint. It has one persistent thermal-bias calibration
generator, one fresh temperature-noise generator per step, and axis-aligned
comfort and safety triggers. Its predictor uses the same constant-input
interface as the robot scenarios.

Reducers currently include:

- `BoxReducer`: replace a zonotope by its interval hull.
- `GirardReducer`, `CombastelReducer`, `MethAReducer`, `ScottReducer`,
  `PcaReducer`, and `AdaptiveReducer`: Python baselines corresponding to the
  main reducer families compared in Kohn et al.
- `TargetBudgetReducer`: wrap another reducer and spend at most a fixed target
  budget; the default suite uses this to include `girard7`.
- `ScoredKeepReducer.by_norm`: keep large generators and box-merge the rest.
- `ScoredKeepReducer.calibration_aware`: keep important generators while giving
  calibration generators a strong preservation bonus and including
  near-threshold trigger influence in the score.
- `IdentityReducer`: the explicit certified `no_reduction` action. It preserves
  the zonotope exactly and succeeds only while the state is already within the
  generator budget. Current paper experiments defer explicit no-op selection,
  so the compatibility columns for no-op accounting should remain zero.
- `ProtectedReducer`: wrap a reducer so monitor-required generators, such as
  the robot calibration generator, are preserved exactly before reducing the
  residual zonotope.

The benchmark now compares three smarter MPC selectors:

- `MPCPolicy` is the original one-action MPC baseline. It chooses the first
  reducer at the current overflow and reuses that same reducer for any
  predicted future overflow in the horizon.
- `SequenceMPCPolicy` searches reducer choices at each predicted overflow. It
  records the chosen reducer sequence, counts evaluated leaves, and prunes
  branches whose partial cost already exceeds the best complete sequence. The
  method `mpc_focused_sequence` uses a geometry-regularized candidate set
  consisting of protected Girard, `girard_slack1`, trigger-influence keep, and
  norm keep, with protected box only as fallback when all normal candidates fail.
- `RolloutMPCPolicy` keeps the first-action search small. It evaluates
  candidate first reductions, then rolls future overflows forward with a fixed
  certified base reducer and an optional certified fallback. The default
  method `mpc_focused_fixed_girard` tests the same focused geometry candidates
  as first actions, uses protected Girard for future overflows, and uses
  protected box reduction only as a last-resort fallback.
- `mpc_wide_fixed_girard` is the broader first-action ablation. It uses the same
  future protected Girard policy and protected box fallback as
  `mpc_focused_fixed_girard`,
  but admits broad protected precision reducers as first-action candidates.
  Protected box is deliberately excluded from the first-action candidate set
  and appears only as an emergency fallback in predicted future overflows.

Artifact and figure outputs may also use report-facing labels for these rollout
methods. In the latest TACAS-main aggregate, `mpc_focused_fixed_girard` appears
as `mpc_rollout_girard`, and `mpc_wide_fixed_girard` appears as
`mpc_rollout_wide`. These are naming aliases for interpretation; the code-level
method IDs above remain the source of truth for CLI method selection and tests.

All three selectors use `WeightedZonotopeCost`. The default MPC and sequence
objectives penalize predicted trigger width, threshold straddling, and a small
generator-count term. The default rollout objective focuses on trigger width
and straddling, with no generator-count or terminal cost term. The cost type
also exposes total-width, synthetic-generator, measurement-generator, and
calibration-generator terms for later sweeps.

The `pzr-paper-figures` entry point builds paper-replication plots and plotting
CSVs for Figures 3--5 style experiments. It also writes diagnostic selection
tables and plots for robot, simple robot, and thermostat runs, including a
fallback-box view that separates destructive box first actions from protected
box emergency fallback use. It supports the same method-set choices as the
benchmark CLI:

- `paper`: only the static reducer baselines.
- `paper_plus_focused`: static baselines plus `mpc_focused_fixed_girard` and
  `mpc_focused_sequence`.
- `paper_plus_mpc_ablation`: static baselines plus focused fixed-Girard, wide
  fixed-Girard, and focused sequence MPC methods.
- `paper_plus_ours`: legacy alias for `paper_plus_focused`.
- `paper_plus_wide`: legacy alias for `paper_plus_mpc_ablation`.
- `extended`: the full default benchmark suite.

The `pzr-run-experiments` suite orchestrates baseline benchmarks, learned
policy distillation from `decision_features.csv`, learned-policy evaluation,
paper figures, aggregate CSVs, and packaging metadata. Aggregate outputs filter
the learned evaluation down to the learned method so baseline methods are not
double-counted across baseline and learned reruns. The suite and figure paths
write `analysis_notes.json` with metric winners, soundness counters, and
warning flags for unexpected no-op use, reduction failures, budget violations,
unsound certificates, or box-first choices in `mpc_wide_fixed_girard`.

Learned-policy results should be interpreted with care. The current code trains
the suite policy against `mpc_focused_sequence` labels, while the latest
TACAS-main artifact's `learned_distilled.metrics.json` records
`expert_method = mpc_rollout_wide`; treat the artifact metadata as authoritative
when reporting that run. In that artifact, the label distribution is highly
Girard-heavy (`girard` 37224 of 40862 rows), with validation accuracy about
0.906 and top-3 accuracy about 0.988. This supports the claim that learned
selection can be kept outside the trusted soundness boundary, but not yet that
learning discovers a qualitatively better selector than the focused rollout.

## Trigger Semantics

The prototype follows the paper's overlap-aware RLola trigger predicates. For an
interval hull component `[l, u]`, an above-threshold trigger `expr >_p c` is a
violation exactly when the fraction of the interval above `c` is strictly
greater than `p`; below-threshold triggers use the symmetric fraction below
`c`. Degenerate intervals use strict point semantics (`x > c` or `x < c`).
Thus paper-style trigger evaluation reports either `violation` or `safe`; the
older `inconclusive` artifact fields remain for schema compatibility and for
separate precision metrics.

Predictive reduction does not change monitor equations or trigger semantics. It
only chooses among certified zonotope approximations before the monitor state
exceeds the generator budget. Straddling and trigger-width costs are precision
heuristics over the interval hull, separate from the paper predicate above.

## Benchmark Outputs

The paper-style robot benchmark includes the optional unreduced `reference`,
static reducers (`box`, `girard`, `girard7`, `combastel`, `methA`, `scott`,
`pca`, `adaptive`, `keep_norm`, `keep_calibration_aware`), and five MPC
methods (`mpc`, `mpc_sequence`, `mpc_focused_sequence`,
`mpc_focused_fixed_girard`, `mpc_wide_fixed_girard`). It writes:

In suite aggregates that include learned evaluation and figure-facing method
labels, expect the method set to be narrower and renamed. For example,
`results/tacas-main/aggregate/raw_runs.csv` contains static baselines plus
`mpc_rollout_girard`, `mpc_rollout_wide`, and `learned_distilled`, but not all
extended CLI method IDs.

- `raw_runs.csv`: one row per scenario, predictor mode, method, and seed.
- `summary.csv`: bootstrapped aggregate statistics with 95% confidence
  intervals.
- `comparisons.csv`: paired method-vs-MPC deltas, effect sizes, and Wilcoxon
  p-values. The baseline priority is `mpc_focused_sequence`, then
  `mpc_focused_fixed_girard`, then `mpc_wide_fixed_girard`, then
  `mpc_sequence`, then `mpc`.
- `predictor_comparisons.csv`: paired online-vs-oracle deltas when both modes
  are run.
- `timeseries.csv`: per-step precision, verdict, reducer, and MPC search
  traces.
- `bounds_timeseries.csv`: per-step state-coordinate interval bounds and
  reference bounds, used by the figure pipeline.
- `decision_features.csv`: per-decision feature rows for policy distillation,
  emitted only at actual over-budget reduction points. Explicit
  `no_reduction` labels are deferred future work and should not appear in
  normal paper runs.
- `selection_summary.csv`: selected first-reducer counts and fractions, grouped
  by scenario, predictor mode, method, and reducer, with reduction failure and
  MPC search totals.
- `predicted_sequence_summary.csv`: MPC predicted-sequence diagnostics,
  including first-action box and future fallback-box counts so destructive box
  first actions can be separated from emergency fallback usage.
- `analysis_notes.json`: suite and figure diagnostic notes with top methods by
  selected metrics, soundness checks, and warning flags.
- `config.json` and `report.json`: machine-readable configuration and full
  report payloads.

Tracked metrics include inconclusive and straddling counts, trigger widths,
width inflation relative to the unreduced reference, verdict disagreement,
unsafe disagreement, false alarms/false violations relative to reference
verdicts, interval-hull MSE, trigger interval-hull MSE, generator counts,
reduction timing, chosen reducer counts, evaluated sequence counts, and pruned
sequence counts. No-op accounting is split from real compression:
`no_op_count` and `chosen_no_reduction_count` count explicit no-op decisions,
while `reduction_count` counts only true certified reductions.

## TACAS Research Roadmap

The target framing is a TACAS research paper rather than a tool paper. The
main claim should be algorithmic: bounded-memory runtime monitoring under
sensor uncertainty benefits from treating zonotope reduction as a predictive
abstraction-control problem. The prototype and artifact support the claim, but
the scientific contribution is the monitor-aware receding-horizon policy and
its soundness boundary.

The current baseline paper makes the bounded-memory problem precise for
stream-based monitoring: affine arithmetic tracks calibration and measurement
uncertainty exactly, but fresh measurement slack variables make monitor state
grow without bound, so runtime monitors need sound online unification or
over-approximation. Classical zonotope order-reduction work in control and
reachability focuses on one-step set enclosure quality and computational cost.
Our gap is semantic and temporal: a reducer should care which generators drive
future trigger verdicts, whether uncertainty is persistent calibration or
one-shot measurement noise, and how approximation error propagates through the
monitor.

Near-term research hypotheses:

- **Predictive abstraction control:** choose the first certified reducer by
  minimizing predicted future verdict imprecision, then replan at the next
  overflow. The current evidence supports a precision claim most strongly:
  focused rollout reduces robot trigger width and interval-hull MSE at the same
  generator budget compared with strong static order-reduction heuristics.
  False-alarm improvements should be stated cautiously because rates are already
  low for the strongest baselines.
- **Information-aware preservation:** calibration generators and
  near-threshold trigger directions should be preserved preferentially because
  their future influence differs from old independent measurement noise.
- **Prediction robustness:** the online-vs-oracle gap should quantify how much
  value comes from better trace prediction and where robust or tube-style
  predictors would matter. The latest evidence suggests this is future work
  rather than the current headline.
- **Artifact-ready evaluation:** TACAS artifact expectations favor
  reproducible scripts, stable data artifacts, documented smoke tests, and
  representative subsets. The current CLI, figure generator, and small smoke
  configurations are the right skeleton for this.

Potential extensions for a TACAS submission:

- Add one more monitor family with different dynamics or trigger geometry to
  show the policy is not robot-specific and not already saturated by the static
  reducers.
- Add a predictor ablation beyond constant-input extrapolation, such as a
  conservative tube predictor, while preserving the rule that predictions never
  justify soundness.
- Compare against a learned or tuned static scoring policy, and train a learned
  selector against the focused rollout labels, to separate the benefit of
  semantics-aware scoring from the benefit of receding-horizon planning.
- Formalize a policy interface theorem: if every candidate reducer returns a
  sound enclosure under budget and protected generator requirements are
  enforced, then any predictive selector over those candidates preserves
  monitor soundness independently of prediction quality.

Useful sources for the paper trail:

- Finkbeiner, Fränzle, Kohn, Kröger, "Cutting Corners on Uncertainty:
  Zonotope Abstractions for Stream-based Runtime Monitoring,"
  <https://arxiv.org/abs/2601.11358>.
- Yang and Scott, "A comparison of zonotope order reduction techniques,"
  Automatica 95, 2018, DOI `10.1016/j.automatica.2018.06.006`.
- Kopetzki, Schürmann, Althoff, "Methods for Order Reduction of Zonotopes,"
  IEEE CDC 2017, DOI `10.1109/CDC.2017.8264508`.
- Combastel, "A state bounding observer based on zonotopes," ECC 2003.
- Scott et al., "Set operations and order reductions for constrained
  zonotopes," Automatica 2022.
- TACAS 2026 artifact guidance, <https://etaps.org/2026/conferences/tacas/>.
- RTLola framework overview, "Stream-based monitoring with RTLola,"
  Science of Computer Programming 253, 2026, DOI
  `10.1016/j.scico.2026.103495`.

## CoRL Venue Framing

The CoRL submission repositions the contribution from TACAS (algorithmic,
formal-methods-first) to CoRL (operational, robot-deployment-first). The
operational claim leads: predictive zonotope reduction reduces avoidable
monitor-triggered fallback interventions on a quadrotor while preserving
certified safety-relevant monitor state. The algorithmic claim (predictive
abstraction control with policy-independent soundness) follows as the
mechanism. See `paper/CORL_2026_PROJECT_NOTES.md` for venue strategy and
`paper/related_work_foundation.md` for the trusted-boundary positioning.

The trusted-boundary architecture unifies the positioning. In classical
shielding, a policy proposes actions and a certified operator preserves system
safety. In predictive safety filters, the operator preserves recursive
feasibility. In PZR, the policy proposes a reduction action and the certified
reducer preserves `Z subseteq rho(Z)` with `gen(rho(Z)) <= K`. Soundness is
policy-independent: any selector over certified candidates inherits the
contract, including learned selectors.

## CoRL Monitor-First Benchmark

The CoRL headline benchmark is monitor-first: certified bounded-memory
uncertainty monitoring on the safe-control-gym IROS gate-flying Level0 task
with a Crazyflie 2.x quadrotor, firmware-in-the-loop via pycffirmware. The
drone navigates through gates while avoiding obstacles under bounded sensor
noise. Level1 is a stress/appendix setting unless it independently passes the
same calibration gates. Code: `pzr.robotics.iros`,
`pzr.robotics.safe_control_gym`, `pzr.experiments.corl_suite`.

The `IrosGateMonitor` tracks 7 derived streams from noisy bounded observations:

- `obstacle_clearance`: minimum clearance to nearest obstacle
- `gate_deviation`: position error relative to current gate center
- `corridor_deviation`: lateral deviation from the gate-to-gate corridor
- `altitude_low_margin`: margin above the altitude floor
- `altitude_high_margin`: margin below the altitude ceiling
- `speed`: scalar speed
- `safety_margin`: combined clearance metric

Five triggers fire on safety envelope violations:

- `collision_risk`: safety_margin < 0
- `obstacle_clearance_violation`: obstacle_clearance < min_obstacle_clearance
- `altitude_low_violation`: altitude_low_margin < 0
- `altitude_high_violation`: altitude_high_margin < 0
- `speed_envelope_violation`: speed > speed_max

The `NoisySensorModel` generates persistent calibration bias (sampled once
per episode from bounded support) plus fresh bounded measurement noise per
step, applied to all 6 state components (3 pose + 3 velocity). Default
configuration: `sensor_bias_bound = 0.015`, `sensor_noise_bound = 0.03`.
The monitor regime also exposes `monitor_overlap`, `stream_memory_decay`, and
`generator_memory_decay` so calibration can avoid saturated trigger behavior
without changing the certified reducer contract.

The `InterventionManager` implements closed-loop fallback control. When any
monitor trigger fires, it switches from nominal gate-following to a fallback
hover command and holds for `fallback_hold_steps` (default 2). It classifies
each intervention step as:

- spurious: monitor triggered but oracle (true state) says safe
- justified: both monitor and oracle triggered
- missed: oracle triggered but monitor did not

Oracle safety is defined by exact simulator state and obstacle/gate geometry,
separate from the bounded noisy observations the monitor sees.

## CoRL Experiment Infrastructure

The CoRL suite (`pzr-run-corl` entry point) orchestrates preflight validation,
optional calibration sweeps, DAgger training, and heldout evaluation. It runs
safe-control-gym in a sidecar Python 3.8 conda environment while the main PZR
package runs on Python 3.11+.

Three profiles control scale:

- `smoke`: 1 seed, 30 steps, horizon 2 — end-to-end test
- `overnight`: 20 train seeds, 50 eval seeds, 1000 steps, horizon 6
- `paper`: 40 train seeds, 100 eval seeds, 1000 steps, horizon 6

Method sets:

- `core`: box, girard, keep_calibration_aware, mpc_focused_fixed_girard,
  mpc_wide_fixed_girard
- `extended`: core + combastel, pca, keep_norm, mpc_focused_sequence

Headline metrics: task_completed, gates_passed, collision_episode,
constraint_violation_episode, fallback_activation_count,
fallback_duration_fraction, spurious_intervention_rate,
justified_intervention_rate, missed_violation_rate, time_to_target,
mean_reducer_latency_ms.

Quality gates enforce `budget_violation_count = 0`,
`unsound_certificate_count = 0`, `reduction_failure_count = 0`.
`analysis_notes.json` records a `paper_usable` flag; runs failing this gate
should not produce headline evidence even if files completed.
Calibration additionally requires nonempty `paper_candidate_config_ids`,
nominal completion at least 0.8, non-saturated fallback, zero missed violations
for the headline MPC method, and bounded methods that differ from Girard on
intervention metrics.

CoRL-specific outputs include `headline_table.csv`, `headline_table.md`,
`headline_quality.md`, `intervention_timeseries.csv`, `monitor_timeseries.csv`,
`failure_events.csv`, `selection_summary.csv`, `predicted_sequence_summary.csv`,
and `analysis_notes.json`, alongside the standard benchmark outputs.

## DAgger Learning Pipeline

DAgger is secondary deployability evidence, not the headline controller claim.
The DAgger loop (implemented in `pzr.experiments.corl_suite._run_dagger_training`)
iterates:

1. Roll out the current learned selector on training seeds in the closed-loop
   Crazyflie environment.
2. At each learner-induced over-budget state, query the MPC expert (default
   `mpc_wide_fixed_girard`) for its certified reducer choice.
3. Aggregate new decision-feature rows with previous rounds.
4. Retrain a small MLP classifier on the accumulated feature rows using
   `pzr.learning.distill_cli.train_policy`.
5. Repeat for `dagger_iterations` rounds (default 3 for overnight).

Decision features (19 numeric features from `pzr.learning.features`): generator
count, state dimension, budget headroom, zonotope order, generator-count growth
rate, largest and median generator norms, trigger proximity metrics, dominant
stream directions, previous reducer choice, success/failure history, and
horizon count estimate.

The trusted boundary is preserved: the learned policy only ranks candidate
reducers from a fixed certified set. The chosen reducer still produces a
certificate `Z subseteq Z'`, `gen(Z') <= K`. If the learned policy picks a
suboptimal reducer, precision degrades but soundness is preserved.

A label-diversity gate checks that training data contains at least 3 distinct
reducer labels and no single label exceeds 90% of rows. If this gate fails, the
learned selector is excluded from headline results. The gate catches degenerate
scenarios where the expert always selects the same reducer, making the learning
problem trivial.

## CoRL Notation Conventions

Paper and SCIENCE.md use consistent notation:

- `Z` for zonotope (`c + G[-1,1]^m`)
- `c` for center, `G` for generator matrix, `m` for generator count, `n` for
  state dimension
- `rho` (or `rho_a`) for a certified reducer applied with action `a`
- `pi` for the reduction policy
- `phi` for trigger predicates
- `K` for the generator budget
- `h` for the prediction horizon
- `S_t` for the monitor state at step `t`

## Current Scope

Version 1 intentionally does not implement an RLola parser, gray-box
sensitivity hooks, learned predictors, or CORA/MATLAB integration. Those can
be added behind the existing adapter, reducer, and policy interfaces without
changing the soundness boundary.
