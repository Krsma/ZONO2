# PZR Related-Work Foundation

**Project:** Predictive Zonotope Reduction (PZR)
**Target venue:** CoRL
**Document purpose:** Reference document for the CoRL submission. Maps PZR onto the safe-learning-for-robotics and runtime-monitoring landscape, records the framing decisions taken for the submission, and provides the source material for the related-work section. All framing in this document assumes a CoRL audience.

---

## 1. What PZR Is and Where It Fits

### 1.1 The safety stack on a real robot under sensor uncertainty

A safety stack for a robot operating under sensor uncertainty needs three coupled components:

1. **Safety filter / shield** — decides which actions to allow (CBF, predictive safety filter, classical shield, HJ filter). Provides the active guarantee that unsafe actions are blocked.
2. **Uncertainty representation** — what the filter consumes. The filter must be conservative across the set of true states consistent with the noisy observation, so it needs that set tracked and propagated forward.
3. **Runtime monitor** — verifies that the specification is actually being satisfied over the running trace. Necessary because every safety filter's guarantee is conditional on its assumptions holding, and you need to detect when they don't.

In benchmark experiments the third component is often omitted: with clean sensors and short horizons the filter's assumptions hold and the monitor is unnecessary. In real deployment the monitor is load-bearing — it is what catches assumption violations before they propagate into the controlled system, and it is what produces the verdicts a safety case can be built on.

### 1.2 The problem PZR addresses

Sound runtime monitoring under independent measurement noise hits a structural problem. The standard symbolic representation — affine arithmetic with zonotopes `Z = c + G[-1,1]^m` (Finkbeiner, Fränzle, Kohn, Kröger 2026, "Cutting Corners on Uncertainty: Zonotope Abstractions for Stream-based Runtime Monitoring", arXiv 2601.11358) — reuses generators for persistent calibration errors but must add a *fresh* generator for each independent measurement reading. Exact monitor state therefore grows without bound, so any long-running deployment either runs out of memory, falls back to an unsound approximation, or pays unbounded compute per step.

This is not a theoretical concern. It blocks deployment of zonotope-based monitors on any robot that runs for more than minutes at a time with noisy sensors. Some form of order reduction is mandatory.

The existing answer in the zonotope literature is to apply a fixed local order-reduction operator (Girard, Combastel, etc.) whenever the generator budget is exceeded. This works but is myopic: the operator is chosen without regard to which generators matter for upcoming trigger evaluations, so monitor precision degrades faster than necessary, increasing spurious or missed verdicts and undermining the monitor's usefulness in the stack.

### 1.3 PZR's contribution

PZR treats zonotope order reduction as a **policy decision over certified operators** rather than a fixed local operation. At each over-budget state, a policy chooses an action from a finite set of certified reducers (Girard, Combastel, MethA, Scott, PCA, adaptive, scored-keep, target-budget, protected wrappers). The policy may be:

- **Static** (fixed choice of reducer),
- **Predictive** (one-step / sequence / rollout MPC over future trigger imprecision), or
- **Learned** (neural policy distilled from the MPC via imitation learning + expert iteration, AlphaZero-style).

The trusted boundary: every certified reducer guarantees `Z ⊆ ρ(Z)` and `gen(ρ(Z)) ≤ K`. Soundness is policy-independent — only the certified reducer mutates monitor state, and the policy merely chooses among certified options. Whatever the policy chooses (or fails to learn well, or is adversarially manipulated to choose), the monitor remains sound.

The result: a runtime monitor that stays sound under bounded memory across unbounded traces, even when sensors are noisy. This makes the monitor deployable on the robot itself rather than only post-hoc on offline logs, which in turn lets the rest of the safety stack — filters, controllers, supervisors — depend on the monitor's verdicts in deployment.

### 1.4 Scope

PZR is the monitor-side uncertainty-tracking component of the safety stack. It is not itself a safety filter: it does not block actuator commands, it does not project unsafe actions away, it does not prevent crashes on its own. A deployment that needs to block unsafe actions still needs a CBF, predictive safety filter, or similar — and PZR sits alongside those, providing the bounded-memory sound uncertainty representation that a monitor over the same uncertainty model requires.

This scoping matters for honest positioning but is not a limitation in the deployment-relevance sense. A safety stack without a sound long-running monitor has a hole in it; PZR is what fills that hole when sensors are noisy.

---

## 2. Background: Safe Learning for Robotics

A condensed taxonomy of methods for keeping robots safe during learning, included here so PZR's positioning is legible.

### 2.1 Safety filters / shielding (Layer A)

A learned policy proposes an action and a certified module overrides if unsafe.

- **Control Barrier Functions (CBFs).** Forward-invariant safe set defined by `h(x) ≥ 0`; unsafe actions projected onto the boundary via a QP. Requires control-affine dynamics model.
- **Hamilton-Jacobi reachability.** Computes guaranteed safe sets by solving a PDE over the state space. Robust HJ handles bounded disturbances; stochastic HJ handles probabilistic ones.
- **Classical (formal-methods) shielding.** Alshiekh et al. "Safe Reinforcement Learning via Shielding" (AAAI 2018) is the canonical reference. Precomputed shield enumerates safe actions per state.

### 2.2 Constrained RL

Optimisation-side approach: reformulate the problem as a CMDP and constrain expected cost.

- **CPO** (Achiam et al.).
- **Lagrangian PPO/SAC.** Practical baselines.
- Provides statistical guarantees (constraint violation in expectation), not per-trajectory.

### 2.3 Safe exploration under uncertainty

- **SafeOpt** (Berkenkamp, Krause et al.). GP-based; only act where worst-case bound is safe.
- **Lyapunov-based safe RL** (Berkenkamp, Turchetta, Schoellig, Krause 2017). Expand safe set as model improves.
- **GoSafe / GoSafeOpt** (Sukhija, Krause et al.). Extensions handling both model and measurement uncertainty.

### 2.4 Sim-to-real

- **Domain randomisation** (Tobin, Peng).
- **System identification + fine-tuning.** Used in most quadruped locomotion (ANYmal, ETH Zurich line; Unitree research).

### 2.5 Recovery and reset-free learning

- **Recovery RL** (Thananjeyan, Goldberg et al.). Learn a backup policy invoked when safety critic predicts danger.
- Reset-free learning forces the agent to recover from arbitrary states without human resets.

### 2.6 Risk-sensitive RL

- CVaR-constrained and distributional RL methods. Optimise worst-case quantiles rather than expectation.

### 2.7 Physical safeguards

Torque/velocity limits, geofences, tethers, kill switches. Always present in real deployments alongside algorithmic safety.

---

## 3. Methods Under Uncertainty

When measurements are noisy or the state is partially observable, every method above has a corresponding extension. Guarantees weaken and conservatism grows as uncertainty sources are stacked.

### 3.1 Robust CBFs under measurement noise

Assume bounded error `‖x − x̂‖ ≤ ε`; tighten the barrier by a margin absorbing the worst case.

- Dean, Taylor, Cosner, Recht, Ames 2020, "Guaranteeing Safety of Learned Perception Modules via Measurement-Robust Control Barrier Functions."
- Cosner, Singletary, Taylor, Molnar, Bouman, Ames 2021, "Measurement-Robust Control Barrier Functions: Certainty in Safety with Uncertainty in State."

Guarantee: **deterministic worst-case**, conditional on bounded-error assumption.

### 3.2 Stochastic / probabilistic CBFs

`P(h(x) ≥ 0) ≥ 1 − δ`.

- Clark 2019, "Control Barrier Functions for Stochastic Systems."
- Santoyo, Dutreix, Coogan 2021.

Guarantee: **probabilistic**, conditional on distributional assumption.

### 3.3 Conformal prediction for safe control

Distribution-free probabilistic guarantees on the perception module, composed with a downstream safety filter that handles every state in the conformal prediction set.

- Lindemann, Qin, Deshmukh, Pappas 2023, "Safe Planning in Dynamic Environments using Conformal Prediction" (RA-L).
- Dixit, Lindemann, Wei, Cleaveland, Pappas, Burdick 2023, "Adaptive Conformal Prediction for Motion Planning Among Dynamic Agents."
- Cleaveland, Lindemann, Ivanov, Pappas 2024 — robust conformal prediction for time series.

Guarantee: **probabilistic, distribution-free, finite-sample**. Caveat: exchangeability assumption breaks under distribution shift.

### 3.4 HJ reachability with disturbances

- Robust HJ: min-max game over bounded disturbances.
- Stochastic HJ: expected-value / chance-constrained.
- Belief-space HJ for POMDPs (limited scalability).
- Neural HJ (DeepReach, Bansal & Tomlin): higher dimensions but guarantee becomes "guarantee modulo verification of the learned value function."

### 3.5 Tube MPC and robust MPC

Plan a nominal trajectory; prove all realisations stay in a tube around it under bounded disturbances.

- Mayne, Seron, Raković 2005 — foundational tube MPC.
- Hewing, Wabersich, Zeilinger — learning-based extensions.
- **Wabersich & Zeilinger "predictive safety filter"** — particularly clean formulation, sits on top of an RL policy and proves recursive feasibility / constraint satisfaction.

Guarantee: **deterministic worst-case**, conditional on bounded-disturbance assumption.

### 3.6 Formal verification of NN controllers

Verify a trained policy directly against a dynamics-plus-uncertainty model.

- Verisig (Ivanov, Carpenter, Weimer, Alur, Pappas, Lee 2021).
- α,β-CROWN, Marabou (Katz et al.), POLAR, NNV.

Guarantee: **deterministic formal proof** for a specific policy. Bottleneck: scalability.

### 3.7 Hierarchy of guarantees

From strongest (most robust to deployment reality) to weakest assumptions:

1. **Bounded disturbance / bounded error** — easy to state, sometimes hard to verify. Fails silently if real noise exceeds bound. *(Robust CBFs, tube MPC, robust HJ — and PZR.)*
2. **Distributional** — sharper guarantees, requires knowing/bounding the distribution. *(Stochastic CBFs, standard probabilistic methods.)*
3. **Exchangeability** — distribution-free but breaks under shift. *(Conformal prediction.)*
4. **i.i.d. training data** — standard ML assumption, often broken in deployment. *(PAC-Bayes safe RL.)*

The strongest *real-world* guarantees come from stacking methods (e.g., conformal prediction on perception + robust CBF using the conformal sets + tube-MPC backup + hardware limits). Any single layer can fail; the composition is harder to break.

---

## 4. Online Shielding

The relevant subthread of the shielding literature for PZR, both for genuine prior work (on the bounded-memory monitoring side) and for the architectural-pattern analogy.

### 4.1 Online shielding proper (Bloem, Könighofer et al.)

Motivation: precomputed shields require enumerating safe actions per state, which doesn't scale. Online shielding computes the shield on the fly for the current state's local subgraph.

- Pranger, Könighofer, Tappler, Deixelberger, Jovanović, Bloem, **"Adaptive Shielding under Uncertainty"** (2021). Directly addresses uncertain transition probabilities.
- Könighofer, Rudolf, Palmisano, Tappler, Bloem, **"Online Shielding for Reinforcement Learning"** (Innovations in Systems and Software Engineering, 2023). Canonical RL statement.
- Pranger, Könighofer, Posch, Bloem, **"TEMPEST — Synthesis Tool for Reactive Systems and Shields in Probabilistic Environments"** (ATVA 2021). Tooling.

The shield is constructed by bounded-horizon probabilistic model checking from the current state, returning actions whose worst-case violation probability stays below δ. Structurally close to PZR's MPC-over-reducers (search forward from current state at every step rather than enumerate a fixed shield).

> **Verify before citing:** confirm venues and years for the above; group is correct, exact bibtex needs cross-checking.

### 4.2 Shielding under partial observability

Once observation uncertainty is in play, you are in POMDP territory.

- Carr, Jansen, Topcu, **"Safe Reinforcement Learning via Shielding for POMDPs"** (AAAI 2023). Shields constructed over belief states.
- Junges, Jansen, Topcu, **"Enforcing Almost-Sure Reachability in POMDPs"** (CAV 2021).
- Jansen, Könighofer, Junges, Serban, Bloem, **"Safe Reinforcement Learning Using Probabilistic Shields"** (CONCUR 2020). Tolerates δ violation probability.

Contrast with PZR: these methods handle uncertainty by *expanding the state representation* (beliefs, distributions). PZR handles uncertainty by *over-approximation in a fixed symbolic structure* (zonotopes with bounded generators). The probabilistic-shields work tolerates δ violation; PZR preserves soundness deterministically. Different points on the same tradeoff curve.

### 4.3 Predictive safety filters as online shields

- Wabersich, Hewing, Carron, Zeilinger 2023, **"Probabilistic Model Predictive Safety Certification for Learning-Based Control."**
- Wabersich, Taylor, Choi, Sreenath, Tomlin, Ames, Zeilinger 2023, **"Data-Driven Safety Filters: Hamilton-Jacobi Reachability, Control Barrier Functions, and Predictive Methods for Uncertain Systems"** (IEEE CSM). Consolidating reference that explicitly groups CBF filters, HJ filters, and predictive filters as variants of the same online-shielding idea.

Closest control-theoretic analog to PZR's MPC architecture: receding-horizon optimisation decides the chosen action at every step, guarantee preserved by structure of the optimisation rather than by the specific choice.

### 4.4 Bounded memory specifically

PZR is genuinely distinctive here. The online-shielding literature focuses on **bounded computation time** (compute locally on demand) but not on **bounded memory of the runtime state itself**. Most online shields are stateless or near-stateless.

Adjacent threads:

- **Stream-based runtime monitoring with bounded memory.** RTLola line (Faymonville, Finkbeiner, Schwenger, Torfah) and earlier Lola work. Bounded memory is first-class; uncertainty handling is recent.
- **Bounded-memory automaton-based monitoring.** Classical, no numerical uncertainty handling.
- **Approximate model checking with bounded representations.** Operator-precision literature in abstract interpretation.

The Finkbeiner/Fränzle/Kohn/Kröger paper PZR builds on is essentially the first to bring sound bounded-memory representations of *numerical* uncertainty into stream-based monitoring. PZR's contribution — making the reduction choice itself a learned policy — is a natural next step, with no direct prior work doing exactly this as far as our search has revealed.

### 4.5 Learned shield policies

A few threads where the *policy controlling the shield* is itself learned:

- **Latent shielding / learned shield approximations.** He, Jansen, Topcu, Bharadhwaj et al. have papers on conservative learned approximations of intractable exact shields.
- **Learning-based MPC** with safety guarantees. Hewing/Wabersich/Zeilinger broader programme: learn parts of MPC (model, cost, terminal set) while preserving safety structurally.
- **Differentiable shields.** Yang/Topcu and others — backprop through the shield to train the policy end-to-end.

PZR's learning angle is closest to the third bucket structurally (certified reducer is the "shield"; learned MPC distillation is the "policy"), but distinct in that PZR learns to *select among certified options* rather than learning the shield itself or learning around it.

---

## 5. The Architectural Pattern: Trusted Boundary

A unifying lens that connects shielding, predictive safety filters, tube MPC, and PZR:

> **Trusted boundary pattern.** An arbitrary policy proposes; a certified operator preserves an invariant. The invariant holds independent of the policy. The policy can be heuristic, learned, or even adversarial without breaking the guarantee.

This pattern recurs throughout the safety stack at every component where a learned or otherwise unverified module must coexist with a guarantee. PZR is the instantiation of the pattern at the monitor-side uncertainty-tracking component:

| Stack component | Policy proposes | Certified operator preserves | Invariant |
|---|---|---|---|
| Action selection (classical shielding, Alshiekh) | Action | Shield (precomputed) | System never enters unsafe state |
| Action selection (predictive safety filter) | Action | MPC with terminal invariant | Recursive feasibility, constraint satisfaction |
| Action selection (robust CBF) | Action | QP projection | `h(x) ≥ 0` |
| Trajectory planning (tube MPC) | Trajectory | Tube construction | Realisations stay in tube |
| **Monitor uncertainty tracking (PZR)** | **Reduction action** | **Certified reducer** | **`Z ⊆ ρ(Z)`, `gen(ρ(Z)) ≤ K`** |

Read as a stack, this table says: the trusted-boundary pattern is already accepted as the right design for action selection and for trajectory planning. PZR extends it to the monitor-side uncertainty representation, which is the component of the stack that previously had no policy-independent way to handle bounded memory under noisy sensors.

This framing turns the related work into a story of *completing the pattern across the stack* rather than *applying a pattern in a new domain*. The former is a stronger argument: it identifies a gap in current safety stacks and fills it with the same design principle that justified the rest of the stack.

---

## 6. Positioning for CoRL

### 6.1 Lead framing

PZR is the **bounded-memory sound uncertainty-tracking component** that a robot's safety stack needs at its monitor layer when sensors are noisy and deployments are long-running. The framing for the CoRL submission:

> "A safety stack on a real robot under sensor noise has three components: a safety filter, an uncertainty representation, and a runtime monitor that verifies the filter's assumptions hold. The first two have well-studied solutions. The third has an unsolved structural problem: under independent measurement noise the monitor's symbolic state grows without bound, blocking long-running deployment. PZR is the component that resolves this — by treating zonotope reduction as a policy decision over certified operators, monitor soundness is preserved under bounded memory across unbounded traces, even for learned policies."

This positions PZR as a necessary component of a deployable safety stack under realistic conditions, not as a parallel artefact. The contribution is concrete: the monitor in the stack diagram now has a sound bounded-memory implementation that did not exist before.

CoRL-specific emphases that follow from this framing:

- **Robotics deployment is the motivation, not an application.** The structural memory problem only becomes blocking on real robots with noisy sensors and long horizons. Frame it as a deployment problem first; the formal-methods content (certified reducers, soundness) appears in service of solving the deployment problem.
- **The learned policy is the headline contribution alongside the MPC formulation.** CoRL audiences expect a learning contribution; "MPC distilled into a neural policy via expert iteration" is both the natural framing for that audience and an honest description of the method.
- **Empirical evaluation must demonstrate end-to-end deployment relevance.** Monitor-internal precision metrics alone read as formal-methods content. At least one experiment should show downstream effect on the safety stack — e.g., monitor verdicts feeding into an adaptive safety filter, with PZR's predictive policy reducing average filter conservatism while preserving worst-case guarantees relative to static reducers.

### 6.2 Candidate framings for the introduction

Three candidate frames for the lead pitch, ordered by strength for CoRL:

1. **"Completing the trusted-boundary pattern across the safety stack."** Lead claim: the design principle that produced shielding, predictive safety filters, and tube MPC (policy proposes, certified operator preserves an invariant) had no instantiation at the monitor-side uncertainty representation. PZR is that instantiation. The contribution is the gap-fill, which is legible to any reviewer who already accepts the pattern's value for the other components.
2. **"Sound runtime monitoring as deployment infrastructure under sensor noise."** Lead claim: existing safety filters and verification methods assume a monitor; the monitor itself has been the implicit silent-success assumption. Under realistic sensor noise that assumption fails (memory grows without bound), and PZR is the component that makes the assumption true again.
3. **"Learning to allocate uncertainty budget."** The learned policy solves a resource-allocation problem (where to spend generator budget) that classical reducers solve myopically. Connects to learning-augmented-algorithms literature; weaker on the safety-relevance pitch but stronger on the ML-methods novelty.

**Recommendation:** lead with framing 1 in the introduction. It makes the contribution legible in one sentence to a safe-RL reviewer ("PZR extends the trusted-boundary pattern to the monitor-side uncertainty representation, where it was previously absent") and positions PZR within an accepted design tradition rather than alongside it. Framing 2 supplies the deployment-relevance argument; framing 3 supplies the learning-methods angle.

### 6.3 Suggested related-work section structure

1. **The safety stack and the trusted-boundary pattern at the action-selection layer** — Alshiekh et al. (classical shielding); Wabersich/Zeilinger (predictive safety filter); Cheng et al. (CBF-based). Establishes the pattern and its deployment role.
2. **The pattern at trajectory planning and reachability layers** — tube MPC (Mayne et al., Hewing et al.); robust HJ. Establishes the pattern's generality across the stack.
3. **Online shielding under uncertainty** — Bloem/Könighofer line; Pranger et al.; Jansen/Carr/Junges/Topcu POMDP work. Establishes that computing the certified operator at runtime is the right move when uncertainty is structured.
4. **Bounded-memory monitoring under uncertainty** — Finkbeiner et al. (motivating paper); RTLola line; set-based estimation (Combastel). Establishes the monitor-side problem and the existing static solutions.
5. **PZR's contribution** — fills the gap. First instantiation (to our knowledge) of the trusted-boundary pattern at the monitor-side uncertainty representation, enabling sound bounded-memory monitoring under noisy sensors and long horizons.

This gives the reviewer a story of pattern → stack-wide application → identified gap → contribution that closes the gap.

### 6.4 Scoping (what PZR does and does not handle in the stack)

The honest scoping that complements the framing:

- **PZR handles:** sound bounded-memory representation of monitor-side uncertainty under noisy sensors and long horizons. Policy-independent soundness across the certified reducer family.
- **PZR does not handle:** blocking unsafe actions (that is the safety filter's role — CBF, predictive safety filter); choosing the safety specification itself (that is the user's role); guaranteeing the underlying noise model is accurate (that is the system identification / sensor characterisation step).
- **PZR depends on:** an accurate bounded-noise model encoded in the affine-arithmetic representation. If real noise exceeds the encoded bounds, soundness degrades the same way it would for any other zonotope-based monitor.
- **PZR is most useful when:** the stack already includes a safety filter using zonotope-based or otherwise set-based uncertainty representation, sensors are noisy enough that fresh generators accumulate quickly, and the deployment horizon is long enough that exact monitor state would exhaust memory.

Phrased this way, the scoping reads as a component datasheet — *what this part does, what it depends on, what other parts you need* — rather than as a defensive list of non-claims.

---

## 7. Key References (Curated)

### 7.1 Direct prior work (problem domain)

- Finkbeiner, Fränzle, Kohn, Kröger 2026, **"Cutting Corners on Uncertainty: Zonotope Abstractions for Stream-based Runtime Monitoring"** (arXiv 2601.11358). Motivating paper.
- RTLola line: Faymonville, Finkbeiner, Schwenger, Torfah and related authors. Stream-based runtime monitoring with bounded memory.
- Lola (Sankaranarayanan et al.) — earlier stream-based monitoring.

### 7.2 Architectural-pattern precedent

- Alshiekh, Bloem, Ehlers, Könighofer, Niekum, Topcu 2018, **"Safe Reinforcement Learning via Shielding"** (AAAI).
- Wabersich, Zeilinger — predictive safety filter series.
- Wabersich, Taylor, Choi, Sreenath, Tomlin, Ames, Zeilinger 2023, **"Data-Driven Safety Filters..."** (IEEE CSM). Consolidating reference across CBF / HJ / predictive filters.
- Cheng, Orosz, Murray, Burdick 2019, **"End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks"** (AAAI).

### 7.3 Online shielding

- Pranger, Könighofer, Tappler, Deixelberger, Jovanović, Bloem 2021, **"Adaptive Shielding under Uncertainty."** *(Verify exact venue.)*
- Könighofer, Rudolf, Palmisano, Tappler, Bloem 2023, **"Online Shielding for Reinforcement Learning"** (ISSE).
- Pranger, Könighofer, Posch, Bloem 2021, **"TEMPEST"** (ATVA).

### 7.4 Shielding under partial observability / uncertainty

- Carr, Jansen, Topcu 2023, **"Safe Reinforcement Learning via Shielding for POMDPs"** (AAAI). *(Verify year.)*
- Junges, Jansen, Topcu 2021, **"Enforcing Almost-Sure Reachability in POMDPs"** (CAV).
- Jansen, Könighofer, Junges, Serban, Bloem 2020, **"Safe Reinforcement Learning Using Probabilistic Shields"** (CONCUR).

### 7.5 Uncertainty + safety filters / CBFs

- Dean, Taylor, Cosner, Recht, Ames 2020, **"Guaranteeing Safety of Learned Perception Modules via Measurement-Robust Control Barrier Functions."**
- Cosner, Singletary, Taylor, Molnar, Bouman, Ames 2021, **"Measurement-Robust Control Barrier Functions."**
- Lindemann, Qin, Deshmukh, Pappas 2023, **"Safe Planning in Dynamic Environments using Conformal Prediction"** (RA-L).
- Dixit et al. 2023, **"Adaptive Conformal Prediction for Motion Planning Among Dynamic Agents."**
- Berkenkamp, Turchetta, Schoellig, Krause 2017, **"Safe Model-based RL with Stability Guarantees."**

### 7.6 Surveys and entry points

- Brunke, Greeff, Hall, Yuan, Zhou, Panerati, Schoellig 2022, **"Safe Learning in Robotics"** (Annual Review of Control). Standard reference.
- Hewing, Wabersich, Menner, Zeilinger 2020, **"Learning-Based Model Predictive Control: Toward Safe Learning in Control."** Heavy on uncertainty handling.

> **All references above should be cross-checked against the bibtex before final submission** — venue/year tags in particular need verification.

---

## 8. Glossary

- **Safety stack.** The set of safety-related components on a deployed robot: safety filter (decides which actions to allow), uncertainty representation (what the filter consumes), and runtime monitor (verifies specifications hold over the running trace). PZR is the monitor-side uncertainty-tracking component.
- **Trusted boundary.** Architectural pattern in which an arbitrary (possibly learned) policy chooses among operations performed by a certified module, with the relevant invariant guaranteed by the certified module independent of the policy's choice. Instantiated in classical shielding (action selection), predictive safety filter (action selection), tube MPC (trajectory planning), and PZR (monitor uncertainty tracking).
- **Certified reducer.** A zonotope order-reduction operator `ρ` satisfying `Z ⊆ ρ(Z)` and `gen(ρ(Z)) ≤ K`. Examples: Girard, Combastel, MethA, Scott, PCA, adaptive, scored-keep, target-budget, protected wrappers.
- **Soundness (of monitor).** The monitor's symbolic state always over-approximates the true uncertainty set, so any specification violation in the true system is also flagged by the monitor (no false negatives). PZR preserves soundness by design.
- **Online shielding.** Constructing the shield on the fly per current state rather than precomputing the full state-action shield. Architecturally similar to PZR's MPC-over-reducers, applied at the action-selection layer of the stack rather than the monitor layer.
- **Expert iteration.** AlphaZero-style training loop: an MPC expert generates trajectories, a neural policy is trained to imitate them, the trained policy is used to seed the next round of MPC search, and so on.

---

*Document maintained alongside the project repository. Update as related-work surveys complete and as positioning decisions firm up.*
