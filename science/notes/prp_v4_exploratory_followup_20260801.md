# Exploratory PRP v4 Follow-Up

**Status:** Exploratory design record, 2026-08-01.

**Current paper decision:** The completed v3 evaluation remains the leading
candidate for the paper. This follow-up should determine whether a small,
well-motivated change can remove the rare Pairwise Ranking Policy (PRP) tails
before we freeze that decision. It must not delay writing the MPC-centered
paper or silently replace the v3 artifact.

## Implementation Boundary

This is a disposable exploratory study. Implement it with small scripts and a
thin wrapper outside the canonical paper pipeline, using a separate output
namespace such as `results/prp-v4-exploratory-v1/`.

In particular:

- do not redesign `src/pzr` or the canonical paper pipeline for this study;
- do not modify or reinterpret `results/paper-evaluation-v3`;
- do not add a general framework for feature plugins, guards, DAgger rounds, or
  arbitrary scenarios;
- do not update `experiments/paper_evaluation_v3.yaml` unless a challenger is
  later selected for a genuine paper v4;
- reuse existing trace, model, teacher-cost, prediction, and evaluation helpers
  where practical, but keep exploratory orchestration in `tools/`;
- prefer a direct implementation for the robot-arm experiment over abstractions
  that are not required by the hypotheses below;
- preserve explicit hashes, seeds, model identities, feature identities, and
  parent artifacts even in the disposable outputs;
- run a small pilot and project runtime before starting a larger sweep.

The purpose of the scripts is to answer the scientific questions quickly. We
should only promote the required mechanisms into core code if the exploratory
results justify a paper-facing method change.

## Motivation

The v3 PRP is fast and usually follows the MPC-quality regime, but it exhibits
rare, severe loss and false-positive tails. The problematic nominal cells occur
at budgets 80 and 120, while the fixed figure-eight evaluation also exposes
weak regimes at budgets 120 and 500. All affected runs complete without native,
fallback, or infrastructure failures. Their approximation loss grows over many
events, indicating accumulated closed-loop degradation rather than a single
invalid reducer application.

The current policy is deliberately information-poor. It ranks reducers from
the pre-event Geometry15 zonotope summary and the budget. In contrast, its
two-event full-width teacher evaluates the current event and the next recorded
event. The policy therefore omits both causally available current inputs and
the teacher's future input. This information mismatch is separate from the
distribution mismatch produced when policy errors move the monitor into states
that are absent from teacher-controlled training trajectories.

We should test these explanations separately:

1. **Missing causal input information.** Does a minimal summary of the current
   arm measurements and their motion resolve ambiguous reducer rankings?
2. **Learner-induced state distribution.** Does supervision on states visited
   by the autonomous policy remove the remaining closed-loop tails?
3. **Residual ranking errors.** If the policy remains imperfect, can a small
   causal rollout challenger verify its proposal more reliably than a neural
   confidence threshold?

## Evidence Against an Immediate Confidence Guard

The v3 time series retains the chosen reducer and its winning score, but not the
complete score vector, Geometry15 vector, or teacher costs on the PRP-visited
state. We therefore cannot reconstruct score-margin confidence on the exact v3
failure events.

As a proxy, we retrospectively applied the frozen Clean148 models to the
teacher-labelled perturbed states preserved by `results/dart-rescue-v1`. We
used tolerance-aware normalized regret, so tied teacher actions did not count as
errors. Deferring the 10% of states with the smallest policy-score margins gave
the following results:

| Budget | Initial error rate | Error after 10% deferral | Total regret caught | Worst remaining regret |
| ---: | ---: | ---: | ---: | ---: |
| 80 | 8.85% | 7.17% | 20.7% | 1.0 |
| 120 | 9.37% | 6.83% | 19.3% | 1.0 |
| 500 | 15.02% | 10.01% | 54.1% | 1.0 |

The corresponding Spearman correlations between score margin and regret were
-0.116, -0.149, and -0.380. A simple feature-distance guard was no better.
Moreover, Clean148 error rates on forced DART recovery states rose to 19.0%,
19.0%, and 18.6% at budgets 80, 120, and 500, respectively. These observations
support covariate shift, but they do not support treating the current
uncalibrated ranking margin as a reliable risk estimate.

Consequently, a score-margin guard should not be the default v4 direction. We
should first measure confidence and teacher regret on the exact learner-visited
states. If we later retain a guard, it should preferably verify actions through
small binding-native causal rollouts.

## Minimal Causal Feature Variants

We should preserve the current fixed-hyperparameter, exact-budget specialist
setup and the existing `32 x 32` ranking MLP. Only the input schema changes.

### Geometry15

This is the existing baseline: fifteen pre-event zonotope and budget summaries,
with no event values or history.

### Geometry20

Append the five current robot joint measurements
`a1m, ..., a5m` to Geometry15. These inputs are already available when the
reducer action is selected and directly affect the nonlinear sine, cosine, and
forward-kinematics propagation.

### Geometry25

Append both the five current joint measurements and their five one-step causal
linear predictions. The prediction should use only the previous and current
arrived events and should degrade deterministically to a hold prediction when
there is insufficient history. This representation exposes a compact version
of the motion information relevant to the two-event teacher without adding an
inference-time rollout or recurrent network:

```text
[Geometry15(Z_t, budget), a_t, linear_predict(a_{t+1} | a_{t-1}, a_t)]
```

Do not add triggers, public monitor streams, exact-reference metrics, recorded
future events, raw absolute time, or every sparse robot-arm input merely because
it is available. The expected-center and geofence constants are held inside the
monitor and are not the primary missing information in the nominal learning
study.

Geometry20 distinguishes the value of the current event from the value of
short motion history. Geometry25 is the preferred minimal candidate if the
prediction adds measurable benefit. Both remain small direct-inference models.

## Stage A: Shadow-Teacher Diagnosis

Replay a deliberately small set without changing the executed PRP trajectory:

- the five known bad nominal seed/budget cells;
- five successful nominal cells matched by budget;
- optionally the problematic fixed figure-eight budget-120 and budget-500
  cells.

At every over-bound decision, record:

- all four PRP scores, ranking, and top-two margin;
- Geometry15, current joint measurements, and predicted next measurements;
- the action selected by PRP;
- full-width root costs using the recorded next event;
- causal full-width root costs using the linearly predicted next event;
- tolerance-aware normalized regret under both teachers;
- reducer feasibility and native failures;
- current approximation loss and the subsequent loss trajectory.

This replay should answer:

- whether regret appears before approximation loss begins to grow;
- whether the important mistakes are low-margin or confidently wrong;
- how frequently the offline and causal teacher disagree;
- whether the current event or predicted motion separates otherwise similar
  Geometry15 states;
- whether locally low-regret decisions can still produce long-term degradation.

The previously inspected seeds are valid for diagnosis only. Do not reuse them
as fresh confirmation evidence.

## Stage B: Clean Information Ablation

Train the following exact-budget specialists with the same Clean148 trajectory
set, candidate catalog, pairwise objective, optimizer seed 42, architecture,
epochs, and fixed hyperparameters:

- `g15_clean148`: the frozen v3 reference;
- `g20_clean148`: current joint measurements added;
- `g25_clean148`: current and predicted next joint measurements added.

Reuse the preserved teacher costs by joining each sample's trace identity and
step to its hashed trace when possible. Reject an incomplete or mismatched join.
Recollect only data that cannot be reconstructed safely from preserved
artifacts.

Evaluate the three variants on reserved model-selection seeds 320--327 at all
seven budgets. This stage tests information content without mixing in a change
to the training-state distribution.

## Stage C: True Learner-Visited DAgger

The historical DART experiment injected a bounded alternative action and then
forced teacher recovery. The new diagnostic should instead perform one genuine
learner-visited aggregation round:

1. Select the best clean representation from Stage B.
2. Roll its exact-budget specialists autonomously on nominal seeds 312--319.
3. Let the learned policy remain in control for the complete trajectory.
4. Query the two-event terminal full-width teacher on every visited over-bound
   state without changing the executed action.
5. Aggregate these rows with Clean148 and retrain using the same optimizer and
   architecture.
6. Balance clean and learner-visited states or trajectories so the much larger
   clean dataset cannot erase the corrective samples.
7. Evaluate the retrained specialists on seeds 320--327.

Collect a Geometry15 DAgger baseline if runtime permits. It separates the effect
of distribution correction from the effect of causal event information:

| Comparison | Question answered |
| --- | --- |
| Geometry20/25 clean vs. Geometry15 clean | Does causal event information help? |
| Geometry15 DAgger vs. Geometry15 clean | Does distribution correction alone help? |
| Geometry25 DAgger vs. Geometry25 clean | Does distribution correction still help after adding information? |
| Geometry25 DAgger vs. Geometry15 DAgger | Does the representation remain important on learner-visited states? |

Start with one DAgger round. Run a second round only if the first clearly reduces
the tail while leaving a measurable residual learner-state error. Do not build a
general iterative DAgger framework for this exploratory study.

## Stage D: PRP-Guided Causal Challenger

Test a deterministic rollout challenger on the best unguarded policy:

1. Rank the four reducers with PRP.
2. Use the PRP proposal and Scott as root actions.
3. If PRP already proposes Scott, use Scott and the second-ranked reducer.
4. Expand both roots with all four candidates at the second event.
5. Use the arrived current event and one linearly predicted future event.
6. Select the root with the smallest two-event binding-native terminal loss.

This evaluates eight terminal leaves. It is more expensive than direct PRP but
smaller than the sixteen-leaf two-event full-width teacher and considerably
smaller than the four-event predictive beam. It tests whether learned
shortlisting can retain most PRP throughput while actual causal rollout loss
prevents severe choices.

Use a descriptive identity such as `prp_causal_challenger_h2`; avoid a generic
"guarded" name that suggests a calibrated safety guarantee. All selected
reducers remain binding-native, so this experiment concerns tightness and
false-positive behavior rather than outer-bound soundness.

A margin-triggered variant is optional and should be tested only if Stage A
shows that a predeclared useful operating point exists, for example at least 80%
recall of high-regret actions while deferring no more than 10--15% of decisions.

## Selection and Confirmation

Use seeds according to the following roles:

- seeds 100--119: retrospective v3 diagnosis only;
- seeds 312--319: learner-visited DAgger collection;
- seeds 320--327: representation, DAgger, and challenger selection;
- seeds 328--347: untouched confirmation, only after freezing a challenger.

The exploratory selection sweep should cover all seven budgets. A likely upper
bound is eight selection seeds times seven budgets times five or six methods,
or 280--336 nominal cells. Select on nominal trajectories only. Evaluate the
four fixed figure-eight traces after freezing the selected variant; they remain
descriptive controlled cases and must not become model-selection data.

If a challenger proceeds to confirmation, compare:

- frozen v3 Clean148 PRP;
- the selected unguarded exploratory PRP;
- `prp_causal_challenger_h2`;
- predictive terminal MPC.

Run all four on seeds 328--347 and all seven budgets. Add the fixed figure-eight
comparison afterward. Do not rerun the complete v3 paper matrix merely to
answer this follow-up question.

## Metrics and Decision Rule

The primary diagnostic is tail removal rather than a small mean improvement.
Report, per budget:

- trace-level loss and FPR distributions;
- median, interquartile range, 95th percentile, and worst trace;
- paired loss ratios and FPR differences against v3 PRP and predictive MPC;
- count of cells exceeding 1,000 times the paired predictive-MPC mean loss;
- reducer composition and action-transition counts;
- learner/teacher disagreement and normalized regret;
- fallback, infeasible-candidate, native-failure, and completion counts;
- diagnostic event-loop throughput and, for the challenger, evaluated leaves.

Promote a v4 challenger only if it removes the severe nominal tails on untouched
confirmation seeds, avoids a meaningful regression at every budget, remains
sound and failure-free, and retains a useful throughput advantage over H=4,
W=4 predictive MPC. Do not promote a variant because it repairs only the
already observed failures.

If no challenger clearly satisfies these criteria, retain v3 and frame PRP as a
fast but imperfect secondary component. That is an acceptable and currently
well-supported paper direction.

## Suggested Disposable Interface

The precise filenames are not binding, but a thin exploratory wrapper could
offer:

```bash
tools/run_prp_v4_exploratory.sh diagnose
tools/run_prp_v4_exploratory.sh feature-screen --workers 10
tools/run_prp_v4_exploratory.sh collect-dagger --workers 10
tools/run_prp_v4_exploratory.sh train
tools/run_prp_v4_exploratory.sh select --workers 10
tools/run_prp_v4_exploratory.sh confirm --workers 10
tools/run_prp_v4_exploratory.sh report
```

The default exploratory command should stop after selection. Confirmation must
require an explicit frozen-variant choice. Each stage should be resumable at a
coarse artifact level, but we should not reproduce the full paper pipeline's
general configuration, cell-planning, and release-reporting machinery.

## Minimal Verification

The disposable implementation still needs focused correctness checks:

- Geometry20/25 use the current event and causal history only;
- changing any future event cannot change extracted features or PRP scores;
- the first linear prediction degrades deterministically to hold;
- sample-to-trace joins preserve trace hash, seed, budget, and step alignment;
- tolerance-aware ties are not counted as policy errors;
- shadow teacher evaluation does not mutate the live PRP monitor;
- autonomous DAgger collection does not force teacher recovery;
- candidate ordering, infeasibility, and fallback behavior remain deterministic;
- selection and confirmation seeds are disjoint;
- v3 manifests and artifacts remain untouched.

These checks can live next to the disposable scripts or in narrowly scoped pure
tests. Broader core tests and documentation changes are warranted only if the
study produces a method that we decide to promote.
