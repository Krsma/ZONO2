# CoRL 2026 Submission Notes: Venue, Framing, Core Message

**Project:** Certified Predictive Reduction for Bounded-Memory Robot Uncertainty Monitors
**Target venue:** Conference on Robot Learning (CoRL) 2026
**Document purpose:** Keep the venue context, the framing arguments, and the core message in one place. This is a framing document, not a project plan.

---

## 1. What CoRL Is

CoRL is the field-defining venue for work at the intersection of machine learning and robotics. Founded in 2017, single-track, annual. Sits alongside ICRA, IROS (broad IEEE flagships), and RSS (small, selective, prestigious) as the four conferences that matter for robotics. CoRL is the youngest and the most ML-flavoured.

Key things to internalize before writing:

- **Selectivity.** Overall acceptance around 30–40%. Oral track much tighter (6–7% recently). Real but not crushing.
- **Audience.** Smaller and more focused than ICRA/IROS. Same researchers across years. Senior figures in safe learning (Schoellig, Krause, Fisac, Zeilinger, Tomlin) and large-scale robot learning (Levine, Finn, Abbeel, Sadigh, Song) actually attend. Heavy industry presence.
- **Atmosphere.** Single-track, intimate, demo-heavy. Energetic and grad-student-heavy. Learning-first applied to robotics.
- **Paper tone.** Closer to NeurIPS than to classical robotics. Crisp method sections, ablation tables, video supplements, learned components doing the heavy lifting. Less mechanism design, kinematics, control theory; more policy architectures, data scaling, benchmarks.

### Hard scope rule (enforced)

From the CoRL call: *"Submissions should focus on a core robotics problem and demonstrate the relevance of proposed models, algorithms, datasets, and benchmarks to robotics. Authors are encouraged to report real-robot experiments or provide convincing evidence that simulation experiments are transferable to real robots. **Submissions without a robotics focus will be returned without review.**"*

Mandatory Limitations section. Honesty is expected — but Limitations sentences like "this is not yet a complete CoRL robotics story" are area-chair gifts for desk rejection. Frame residual limitations precisely (e.g., "1D/2D quadrotor models, full 3D deferred to future work"), not categorically.

### What survives review

The robotics part is the *entry ticket*. The ML/method part is where the contribution *lives*. A paper that is 80% method novelty with a credible 20% robot demo can land. A paper that is 80% formal/hardware engineering with off-the-shelf learning belongs at RSS or ICRA. Pure simulation results without a transfer story are increasingly viewed with suspicion.

---

## 2. The Core Message of the Paper

> The math chapter is good. The soundness boundary is elegant. The empirical core (predictive reducer beats fixed geometric rule at fixed memory) is real. What needs to land at CoRL is the *robotics envelope* around the formal contribution: a robot that visibly benefits, metrics a roboticist recognizes, and a learning story that pays its rent.

The contribution itself — **policy-guided certified zonotope reduction with policy-independent soundness** — stays. The reframing is in the experiments and the narrative.

### Target abstract (the shape we are aiming for)

> *Quadrotors deployed under sensor uncertainty rely on runtime monitors to gate aggressive maneuvers, but uncertainty-aware monitors based on affine arithmetic grow unboundedly with flight duration. We show that the choice of bounded-memory zonotope reducer changes the rate of spurious safety interventions on a Crazyflie quadrotor by X% at fixed monitor memory, and that a predictive selector reduces interventions further while preserving formal soundness via certified reducers. Integrated with RTLola's ROS adapter, the method is a drop-in for existing aerial monitoring stacks.*

This is a CoRL abstract. The original "policy-guided certified reduction of zonotopes" abstract was an HSCC abstract. Same contribution, different audience.

### What needs to be true in the paper for that abstract to be honest

| Component | HSCC framing | CoRL framing |
|---|---|---|
| Robot | Synthetic traces shaped like robot data | Crazyflie 2.x in safe-control-gym (PyBullet) |
| Loop | Open-loop monitor on traces | Closed-loop: monitor → safety filter → controller → dynamics |
| Headline metrics | Trigger width, hull MSE, false-alarm rate | Spurious interventions/episode, missed violations, mission success rate |
| RTLola | Cited as background | Used in the loop (or honestly: semantically reimplemented with RTLola as the reference spec) |
| Learning contribution | Distillation that fails to beat the expert | DAgger against the focused expert: matches/beats expert precision AND removes the runtime overhead blocking real-time deployment |

---

## 3. Why the Repositioning Works

Three framing arguments do the structural work.

### 3.1 RTLola is already robotics-native

The Baumeister et al. autonomous-aircraft monitoring work and the published ROS adapter establish that the *underlying monitoring framework* is robotics-native. Citing this transfers credibility through the dependency chain. It does **not** by itself establish that *our contribution* is robotics-native — that is what the closed-loop Crazyflie experiment is for. The framing argument and the experimental work have to do their jobs together; neither suffices alone.

### 3.2 This is not reachability

The relationship to CORA / JuliaReach / SpaceEx is where a knowledgeable reviewer will push hardest. Three distinct objections to be ready for:

**1. "Use CORA's reducer implementations."** Easy. Either use them directly, or validate our implementations against CORA's outputs on a fixed test suite and say so in an appendix. Implementation-credibility objection defused.

**2. "Compare against newer reducers like Sadraddini–Tedrake."** Medium. Add it to the baseline set. Beating it strengthens the claim from "beats classical reducers" to "beats the current strongest static reducer."

**3. "Why not just use online reachability instead of stream monitoring?"** The deep one. Substantive defense:

- *Reachability assumes a known dynamics model. Stream monitoring doesn't.* RTLola specs include sensor fusion expressions, threshold checks, temporal aggregates — many have no dynamical-system semantics ("did IMU temperature exceed bounds for >500ms in the last 5s" is not a reachability problem). We cover the full stream-monitoring case; reachability covers the subset castable as state-space propagation.
- *Different uncertainty sources, different composition.* Reachability propagates uncertainty *in system state* under bounded inputs. Our monitor tracks uncertainty *in sensor observations* — calibration bias, measurement noise, possibly clock skew. These compose differently in the affine arithmetic. Reachability tools handling noisy measurements typically use Kalman-style updates (throws away symbolic correlation) or constrained-zonotope measurement intersection (engineering-heavy, not the standard pipeline).
- *Our contribution is the reduction decision, not the set representation.* Even with CORA as runtime, the question of which reducer to apply when, under a monitor-aware future cost, is still ours. The method is compatible with any zonotope-using runtime, including CORA-based ones. This makes the contribution broader, not narrower.

### 3.3 Avoiding the reviewer trap

If we push the "not reachability" argument too far, a reviewer can flip it: *"if this isn't reachability, then it's just signal processing with intervals — why is it interesting?"*

The right framing is in the middle: **set-based estimation in spirit, but applied to monitor stream semantics rather than dynamical state, with reduction reframed as a control problem under a precision objective.** Connection to reachability is the source of tooling and baselines; disconnection is in what's being tracked and why generators accumulate.

Anchor citation for cross-community credibility: **Yang & Scott, Automatica 2018** ("A comparison of zonotope order reduction techniques"). Both reachability and monitoring communities accept this as the standard benchmark reference.

---

## 4. The Learning Story (Two-Sided)

CoRL is a learning-in-robotics venue. The learning contribution has to pay rent. With DAgger now implemented against the focused MPC-G expert, the learning story is two-sided and both halves matter equally.

**Side 1: Precision.** DAgger collects on-policy data against MPC-G specifically. The hope and the claim to verify empirically: the learned selector matches or exceeds the rollout expert on the precision metrics (trigger width, hull MSE, intervention rate at fixed memory) because it sees states the expert's offline data did not cover. This converts the learning component from "distillation almost as good" to "learned selection that meets or beats the expert it imitates."

**Side 2: Deployability.** Crazyflie inner loops run at ~500 Hz with tight deadlines. MPC rollout costs roughly an order of magnitude more than a static reducer per decision; there is a control-rate ceiling above which rollout-based selection becomes undeployable. The learned selector is a single forward pass and removes that overhead. Measure the control rate at which MPC-G's decision latency exceeds the period and how much further the learned selector extends the deployable regime.

Together these say: **the learned selector matches or improves the expert's precision while removing the runtime overhead that prevents the expert from being deployed at robotic control rates.** That is a real CoRL learning contribution — it is not a negative ablation, and it is not just acceleration. It is acceleration *plus* on-policy improvement, both grounded in a deployment constraint the audience cares about.

The distilled baseline from the original draft stays as an ablation showing that pure distillation from the broader expert was not enough; DAgger against the focused expert is what gets us both halves of the story.

---

## 5. Reviewer Reactions to Anticipate

Useful to keep in mind while writing each section.

- **Robotics-leaning reviewer.** "Where is the robot? Why should the CoRL audience care?" Answered by the closed-loop Crazyflie scenario and robotics metrics. If they cannot find a robot in the experiments section, the rest of the paper does not matter.
- **Safe-learning subcommunity reviewer (Schoellig / Krause / Fisac orbit).** Will recognize safe-control-gym instantly. Will ask about safety filter baselines, conformal prediction comparisons, and whether the safety filter is held fixed across conditions. Hold the filter fixed; reference CP in related work as complementary, not competing.
- **Formal-methods-leaning reviewer (rare at CoRL but possible).** Will probe the CORA relationship and the soundness proof. The proposition itself is solid; the positioning paragraph (Section 3.2 of this doc) is what defuses pushback.
- **ML-leaning reviewer.** Will focus on the learning contribution. Must not be a negative ablation — see Section 4.
- **Area chair.** Reads the abstract and the Limitations section first. Both must read as a CoRL paper, not as an HSCC paper accidentally submitted to CoRL.

---

## 6. The One-Paragraph Summary

We are repositioning a formally clean bounded-memory monitor paper from a monitoring/formal-methods narrative into a robot-learning narrative. The math (policy-independent soundness over certified reducers) and the empirical core (predictive selection beats fixed reduction at fixed memory) are intact. The CoRL-shaped paper makes the monitor *do something for a robot*: a closed-loop Crazyflie in safe-control-gym where the monitor's verdict gates a safety filter, where reducer choice changes mission-level outcomes a roboticist recognizes, and where a DAgger-trained selector both matches or improves on the MPC expert's precision and removes the runtime overhead that prevents the expert from being deployed at robotic control rates. The framing distinguishes our setting from reachability (different generator-growth source, broader specification language) while using the reachability community's tooling and baselines to defuse credibility objections.
