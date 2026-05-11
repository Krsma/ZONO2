# Predictive Zonotope Reduction for Bounded-Memory Runtime Monitoring

## Project start note

This note compresses the current project discussion into a starting point for a possible RV 2026 paper. The central idea is to apply control-theoretic ideas, especially receding-horizon / model predictive control (MPC), to the problem of zonotope approximation in runtime verification.

---

## 1. Core motivation

Runtime monitors for uncertain cyber-physical streams can use zonotopes to represent uncertainty compactly. A zonotope has the form

\[
Z_t = c_t + G_t[-1,1]^{m_t},
\]

where \(c_t\) is the center and \(G_t\) contains the generators. As new uncertain measurements arrive, new generators are introduced. Over long traces, the number of generators grows, so bounded-memory monitoring requires reducing a large zonotope to a smaller one.

The standard problem is therefore:

> Given a zonotope with many generators, replace it by a sound over-approximation with at most \(K\) generators.

The proposed project reframes this as a control problem:

> Use predictive control to decide, online, how to spend a bounded generator budget in a sound uncertainty-aware runtime monitor.

This is more RV-relevant than simply saying “use MPC to approximate zonotopes.” The monitor must remain sound; the control-theoretic part should improve precision, not replace the soundness argument.

---

## 2. Main research idea

The proposed formalism is:

> **Receding-horizon abstraction control for bounded-memory runtime monitoring.**

The monitor state contains a zonotope-valued abstract state. At each time step, the monitor propagates uncertainty through the stream specification. If the generator count exceeds a budget \(K\), a reduction must be applied.

The control input is not a physical action. It is a **compression decision**:

\[
a_t = \text{choice of certified zonotope reduction parameters}.
\]

The resulting reduced zonotope must satisfy:

\[
Z_t \subseteq \operatorname{Reduce}(Z_t, a_t),
\qquad
\operatorname{order}(\operatorname{Reduce}(Z_t,a_t)) \le K.
\]

The key separation is:

- **Soundness** comes from the certified reduction operator.
- **Precision** is improved by the MPC-style optimizer.

Thus, the optimizer may be imperfect, the predictions may be wrong, and the horizon may be short; the monitor remains sound as long as every chosen reduction is a valid over-approximation.

---

## 3. Why MPC is not impossible despite the infinite action space

Naively, the action space is infinite: one could ask the controller to choose any lower-dimensional zonotope containing the current one. That is too broad and likely intractable.

The solution is to restrict the action space to a family of certified reduction operators. The MPC controller does not choose an arbitrary smaller zonotope. It chooses parameters of a reduction scheme whose soundness is already known.

Examples of admissible actions:

1. Choose which generators to preserve.
2. Choose how many generators to allocate to different streams or subexpressions.
3. Choose among several reduction operators.
4. Choose scoring weights used to rank generators.
5. Choose template directions or approximation modes, provided containment is guaranteed.

For example, the controller may preserve a subset of generators and absorb the discarded generators into a box over-approximation. This is sound because the discarded generator contribution is contained in its interval hull.

A useful low-dimensional continuous action is:

\[
a_t = (\alpha, \beta, \gamma, \delta),
\]

where the coefficients define a generator score

\[
s_i =
\alpha \|g_i\|
+ \beta \cdot \operatorname{sensitivity}_i
+ \gamma \cdot \operatorname{thresholdRisk}_i
+ \delta \cdot \operatorname{age}_i.
\]

The reduction then keeps the top-ranked generators and safely merges the rest.

---

## 4. MPC objective

The objective should be monitor-aware, not merely geometric. A generic objective could be:

\[
J = \sum_{\tau=t}^{t+H}
\Big[
\alpha \cdot \operatorname{width}_{\varphi}(Z_\tau)
+ \beta \cdot \operatorname{risk}_{\text{threshold}}(Z_\tau)
+ \gamma \cdot \operatorname{runtime}(a_\tau)
+ \lambda \cdot \operatorname{memory}(Z_\tau)
\Big].
\]

Important cost terms:

- output uncertainty width,
- whether the zonotope straddles a specification threshold,
- predicted false-alarm or inconclusive-verdict risk,
- runtime cost,
- memory/generator budget.

The most RV-relevant objective is **specification-sensitive precision**, not raw geometric volume. Two zonotopes with similar volume may have very different effects on a monitor verdict. The controller should preserve uncertainty directions that affect future verdicts and merge directions that are irrelevant to the specification.

---

## 5. Prediction models

MPC requires a prediction model for future monitor evolution. This does not need to be a perfect model of the physical system. It only needs to estimate how current reduction choices affect future monitor precision.

Possible models:

### 5.1 Trace replay or extrapolation

Use recent observations to predict short-horizon future inputs, for example via constant extrapolation, moving average, or last-period replay. This is simple and likely sufficient for a first version.

### 5.2 Specification-local sensitivity model

Estimate how much each generator affects future verdict-relevant streams. For generator \(g_i\):

\[
s_i = \sum_{\tau=t}^{t+H} \|J_\tau g_i\|,
\]

where \(J_\tau\) is a linearized sensitivity of future monitored outputs with respect to the current abstract state.

This is probably the cleanest technical direction because it connects control, prediction, and RV semantics.

### 5.3 Learned trace model

A learned model can predict future inputs or future abstraction growth. This is more ambitious and should not be the first version unless clearly separated from soundness. Learning may guide reduction choices, but every reduction must remain certified.

Important claim:

> Prediction affects precision and runtime, not soundness.

---

## 6. Candidate algorithm

```text
Input:
  stream monitor M
  generator budget K
  horizon H
  certified reduction family R(·, a)
  cost function J

At each time t:
  1. Read new uncertain input.
  2. Propagate the zonotope through the monitor update.
  3. If generator count <= K, continue.
  4. Otherwise:
       a. Generate candidate reduction actions or action sequences.
       b. For each candidate:
            i. simulate the abstract monitor H steps forward,
           ii. apply certified reductions during rollout,
          iii. estimate future imprecision / verdict risk.
       c. Select the candidate with lowest predicted cost.
  5. Apply only the first certified reduction action.
  6. Emit the monitor verdict.
```

This is standard receding-horizon control: optimize over a horizon, apply the first action, observe the next state, and repeat.

---

## 7. Possible action spaces

### 7.1 Minimal discrete action space

Choose among a small library of certified reductions:

\[
a_t \in \{
\text{box},
\text{norm-based},
\text{sensitivity-preserving},
\text{threshold-preserving},
\text{template/PCA-style}
\}.
\]

This makes the MPC component easy and robust. The contribution becomes adaptive online algorithm selection.

### 7.2 Budget-allocation action space

Allocate the generator budget across monitor streams or subexpressions:

\[
a_t = (K_1, K_2, \dots, K_p),
\qquad
\sum_i K_i \le K.
\]

This is likely very RV-relevant: the monitor has many internal streams, and the controller decides where precision matters most.

### 7.3 Continuous scoring-weight action space

Choose weights for a generator scoring function:

\[
s_i =
\alpha \|g_i\|
+ \beta \cdot \operatorname{sensitivity}_i
+ \gamma \cdot \operatorname{thresholdRisk}_i
+ \delta \cdot \operatorname{age}_i.
\]

Then keep the top \(K\) generators and soundly merge the rest. The action space is continuous but low-dimensional.

---

## 8. Main soundness theorem

A central theorem should be simple and strong.

### Theorem sketch

Assume:

1. the abstract monitor update is sound with respect to the concrete stream semantics;
2. every reduction operator \(R(\cdot,a)\) satisfies

\[
Z \subseteq R(Z,a)
\]

for every admissible action \(a\);

3. the monitor verdict semantics is conservative with respect to the abstract state.

Then the MPC-guided monitor is sound for all choices of:

- prediction model,
- horizon length,
- optimizer,
- cost function,
- admissible action sequence.

The reason is that the MPC component only chooses among sound over-approximations. It can affect precision and runtime, but it cannot make the monitor unsound.

This theorem should be one of the main RV-facing contributions.

---

## 9. Implementation plan

### Stage 1: Baseline zonotope monitor

Implement or reuse a zonotope-based stream monitor:

- affine stream updates,
- uncertain inputs,
- zonotope state,
- generator growth,
- bounded generator budget,
- baseline reduction schemes.

Baselines should include:

- box abstraction,
- norm-based generator selection,
- static truncation/reduction,
- Girard-style reduction if appropriate,
- the reduction strategies from the motivating paper, if reproducible.

### Stage 2: Specification-sensitive generator scoring

Implement generator priority scores based on verdict relevance.

Simple score:

\[
s_i = \|C_t g_i\|,
\]

where \(C_t\) projects the monitor state onto verdict-relevant output streams.

More expressive score:

\[
s_i =
\alpha \|g_i\|
+ \beta \cdot \operatorname{distToThreshold}^{-1}(g_i)
+ \gamma \cdot \operatorname{futureSensitivity}_H(g_i).
\]

Generators that strongly affect near-threshold outputs should be preserved; irrelevant generators can be merged.

### Stage 3: Receding-horizon controller

At each reduction point:

1. sample or enumerate candidate reduction parameters;
2. roll out the abstract monitor for horizon \(H\);
3. apply certified reductions during rollout;
4. compute predicted cost;
5. choose the best first action;
6. apply that action to the real monitor state.

Possible optimizers:

- enumeration over a small action library,
- random shooting,
- cross-entropy method,
- beam search,
- simple gradient-free optimization.

For RV, keep the optimizer simple. The novelty should be the sound predictive abstraction-control formulation, not a complicated MPC solver.

---

## 10. Evaluation plan

The main empirical question:

> Under the same generator budget, does predictive reduction produce more precise verdicts than static or geometry-only reduction?

Metrics:

- runtime per tick,
- memory usage,
- generator count,
- zonotope width at output streams,
- number of inconclusive verdicts,
- number of spurious alarms / false positives,
- distance to exact affine arithmetic when exact tracking is feasible,
- degradation as the generator budget \(K\) decreases.

Benchmarks:

- RLola-style stream specifications,
- uncertain sensor streams,
- threshold monitors,
- moving average / integration / derivative-style monitors,
- CPS-inspired traces such as temperature, speed, battery, altitude, distance-to-obstacle.

Important ablations:

- horizon length \(H\),
- generator budget \(K\),
- cost function variants,
- prediction model variants,
- action-space variants.

Most convincing expected result:

> MPC-guided reduction reduces inconclusive verdicts or spurious alarms under the same memory budget, especially near specification thresholds.

---

## 11. RV 2026 paper framing

The paper should be framed as an RV paper about online precision management under resource bounds, not as a generic control-theory paper.

Possible title:

> **Predictive Zonotope Reduction for Bounded-Memory Runtime Monitoring**

Alternative title:

> **Receding-Horizon Abstraction Control for Uncertainty-Aware Runtime Verification**

Possible abstract-level framing:

> Runtime monitors for uncertain cyber-physical streams must reason about measurement error. Zonotopes provide a compact symbolic representation of uncertainty, but online monitoring introduces fresh uncertainty symbols over time, making reduction unavoidable under bounded memory. Existing reductions are largely local and geometry-driven: they compress the current zonotope without considering how approximation error will affect future verdicts. We propose receding-horizon abstraction control, a sound MPC-inspired framework that treats zonotope reduction as an online control problem. The controller predicts the effect of reduction choices on future monitor precision and selects certified reductions under a fixed generator budget. Since all actions are sound over-approximations, the resulting monitor remains sound regardless of prediction accuracy or optimizer suboptimality. Experiments on stream-monitoring benchmarks show improved verdict precision under the same memory budget.

---

## 12. Things to avoid

Avoid claiming:

> We solve optimal zonotope approximation with MPC.

That is too broad and likely not defensible.

Avoid making reinforcement learning the first version. RL creates avoidable reviewer concerns about training distributions, reproducibility, interpretability, and whether the learned component compromises soundness.

A safer and stronger claim is:

> We use receding-horizon planning to choose among certified abstraction reductions.

This keeps soundness clean and makes the contribution plausible for RV.

---

## 13. Recommended first version

The most realistic first version is:

- **Formalism:** receding-horizon abstraction control.
- **State:** zonotope-valued monitor state.
- **Action:** generator-retention scores or precision-budget allocation.
- **Dynamics:** abstract monitor update plus certified reduction.
- **Cost:** predicted output uncertainty and threshold-straddling risk.
- **Constraint:** at most \(K\) generators.
- **Soundness:** every reduction must over-approximate.
- **Optimizer:** random shooting or enumeration over a small action set.
- **Evaluation:** compare against static/geometric reductions on uncertain stream-monitoring benchmarks.

This gives a coherent RV paper: control theory is used to guide approximation, but runtime-verification soundness remains the central invariant.
