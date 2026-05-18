# TACAS Roadmap

This note records the more detailed TACAS direction discussed after the initial
prototype and benchmark work. The intended framing is a TACAS research paper,
not primarily a tool paper: bounded-memory runtime monitoring under uncertainty
benefits from treating zonotope reduction as a policy-guided abstraction-control
problem.

## Core Framing

Stay close to the paper's semantic stack:

```text
noisy measurement trace
  -> RLola relational semantics
  -> symbolic affine semantics
  -> resolved affine monitor state
  -> zonotope representation
  -> bounded-memory zonotope over-approximation
  -> overlap-aware trigger verdicts
```

Our contribution changes only the approximation-selection step. Instead of a
fixed approximation operator:

```text
if generator_count(Z_t) > K:
    Z_t := reduce_K(Z_t)
```

we use a policy-selected certified reducer:

```text
if generator_count(Z_t) > K:
    a_t := policy(history, Z_t, predictions)
    Z_t := reduce_K^{a_t}(Z_t)
```

The key theorem should be policy-independent:

> If every candidate reducer is a sound bounded over-approximation, then any
> policy selecting among those reducers preserves the same monitor soundness
> guarantee as the paper's generic approximation step.

Prediction quality, MPC optimality, learned selectors, and oracle predictors
can affect precision and runtime, but not soundness.

## Research Directions

1. **Theory-first formalization**

   Define the reducer action contract and policy-guided approximation precisely.
   A reducer action `rho_a,K` is a partial function that, when it succeeds,
   satisfies:

   ```text
   gamma(Z) subseteq gamma(rho_a,K(Z))
   gen(rho_a,K(Z)) <= K
   ```

   The policy is an untrusted selector over certified reducers:

   ```text
   pi_t : H_t x Z_t x Pred_t -> A
   ```

   where `H_t` is observed history, `Z_t` is the current zonotope, `Pred_t` is
   an optional future trace or tube, and `A` is the finite reducer action set.

   Formal results to state:

   - Policy-independent soundness: approximate states always enclose the exact
     symbolic monitor state.
   - Bounded-memory invariant: after reduction points, `gen(Z_t) <= K`, assuming
     a certified reducer succeeds.
   - Protected calibration preservation: required semantic generators are kept
     exactly when they fit within budget.

2. **Finite-horizon optimality**

   State what the existing MPC selectors optimize under fixed predictions:

   - `SequenceMPCPolicy` returns a minimum-cost reducer sequence over the finite
     predicted overflow tree, given deterministic monitor transitions, fixed
     candidate reducers, additive cost, and admissible pruning.
   - `RolloutMPCPolicy` returns a minimum-cost first reducer within the
     restricted class "choose first action, then use fixed base reducer and
     fallback for future overflows."

   This is not needed for soundness, but it explains why the MPC machinery is a
   principled selector rather than just a heuristic.

3. **Optional reduction / no-op actions**

   Extend the action model so a policy can explicitly choose not to reduce when
   the state is within budget, or can defer reduction until the next predicted
   overflow. This makes the selector closer to a control policy over both
   reduction timing and reduction type.

   Key implementation idea:

   - Add an explicit `no_reduction` / identity action only when
     `generator_count <= K`.
   - Keep the certified reducer boundary unchanged: no-op is sound because it
     preserves the current zonotope exactly and does not violate the budget.
   - Track no-op choices separately from reducer choices in benchmark artifacts.

4. **Learned policy distillation**

   Use expensive MPC, sequence, or rollout decisions as offline labels for a
   cheap selector. Start with simple artifact-friendly models such as a decision
   tree, random forest, or nearest-centroid classifier before considering neural
   models.

   Candidate features:

   - trigger widths and threshold distances;
   - straddling counts;
   - generator count and budget slack;
   - generator kind counts, ages, and norms;
   - recent verdicts and reduction history.

   Theorem stays simple: the learned policy is sound because it only selects
   certified reducer actions.

5. **Robust / tube-aware prediction**

   Replace point future traces with bounded prediction sets, ensembles, or
   zonotopic input tubes. Choose reducers by minimizing worst-case or average
   predicted future imprecision.

   Implementation options:

   - Easy first version: generate multiple plausible future traces and score a
     reducer by average or max rollout cost.
   - Stronger version: represent future input uncertainty as zonotopic tubes and
     propagate them conservatively through the monitor.

   This is the strongest control-theory connection, via robust/tube MPC and
   set-membership prediction. Prioritize it after checking whether the current
   oracle-vs-online predictor gap is large enough to make prediction quality a
   meaningful factor.

6. **Broader evaluation beyond robot**

   Add a second monitor family with different dynamics and trigger geometry to
   avoid a robot-specific story.

   Good candidates:

   - thermostat/building control monitor;
   - battery or energy monitor;
   - network latency/SLA monitor;
   - signal threshold monitor;
   - drone altitude-rate envelope.

   The first non-robot benchmark should be simple, artifact-friendly, and still
   include persistent calibration uncertainty plus fresh measurement noise.

## Recommended Implementation Sequence

1. Add explicit theory notes in `science/SCIENCE.md` for policy-independent
   soundness, bounded memory, protected generators, and finite-horizon selector
   optimality.
2. Add optional/no-op reduction actions and benchmark accounting for no-op
   choices.
3. Add training-data logging for MPC decisions and a simple distilled selector
   baseline.
4. Add one non-robot monitor family and include it in smoke tests and paper
   figure generation where appropriate.
5. Add robust/tube-aware prediction only if oracle-vs-online results show that
   prediction quality materially affects precision.

## Near-term Acceptance Criteria

- The formal notes clearly separate monitor soundness from selector quality.
- Every method that mutates monitor state still goes through certified reducers
  or exact no-op preservation.
- Benchmarks report chosen reducer counts, no-op counts, evaluated/pruned
  sequence counts, verdict metrics, false alarms, and trigger precision metrics.
- Generated outputs remain reproducible through small smoke commands and larger
  paper-style CLI runs.
