# PZR Related-Work Foundation

This is the active paper-facing framing note. It intentionally avoids old
module names and removed experiment commands.

## Problem Position

PZR addresses monitor-side uncertainty tracking in long-running robotic or
cyber-physical systems. A safety stack under sensor uncertainty usually needs:

1. a safety filter or shield that decides which actions are allowed;
2. an uncertainty representation consumed by safety and monitoring logic;
3. a runtime monitor that checks whether specifications hold over the actual
   execution trace.

Zonotope-based stream monitoring handles bounded sensor uncertainty, but fresh
measurement noise creates new symbolic generators over time. Exact affine state
therefore grows without bound. PZR makes this monitor layer bounded-memory by
choosing certified order-reduction actions online.

## Contribution Framing

The trusted-boundary pattern is the central framing:

| Stack component | Policy proposes | Certified operator preserves |
| --- | --- | --- |
| Classical shielding | Action | Safe-action shield |
| Predictive safety filter | Action | Feasibility/safety certificate |
| Robust CBF | Action | Barrier-respecting QP projection |
| Tube MPC | Trajectory | Disturbance tube |
| PZR | Reduction action | `Z subseteq rho(Z)`, `gen(rho(Z)) <= K` |

PZR's policy can be static, predictive, or learned. The certified reducer is
the trusted operator, so soundness is independent of policy quality.

## Distinction From Reachability

Reachability propagates sets through a dynamical system model. PZR propagates
uncertainty through monitor stream semantics: filters, temporal offsets,
derived trigger streams, and specification predicates. Some monitors can be
embedded into state-space reachability, but not all stream-monitoring logic is
naturally a dynamics model.

The relationship is complementary:

- reachability provides set representations and reducer baselines;
- PZR studies online reducer selection under monitor-specific precision costs;
- native reducers should be validated against reachability tooling such as CORA
  when paper claims depend on exact baseline parity.

## Robotics Position

The robotics story should be operational before it is algorithmic:

> A robot with noisy sensors needs a long-running sound monitor. Exact
> zonotope monitor state grows without bound. Predictive certified reduction
> keeps the monitor bounded while reducing avoidable trigger imprecision or
> downstream intervention load.

This is not a claim that PZR replaces safety filters. PZR provides the
bounded-memory monitor-side uncertainty state that can feed or audit such
filters.

## Related Areas To Cover

Direct prior work:

- Finkbeiner, Fränzle, Kohn, Kröger, "Cutting Corners on Uncertainty:
  Zonotope Abstractions for Stream-based Runtime Monitoring."
- RTLola and Lola stream-based runtime monitoring.
- Zonotope order reduction: Girard, Combastel, MethA, Scott, PCA-style and
  related reachability literature.

Trusted-boundary and safety-stack context:

- Classical shielding for reinforcement learning.
- Online shielding under uncertainty.
- Predictive safety filters and robust MPC.
- Robust and measurement-aware control barrier functions.
- Tube MPC and Hamilton-Jacobi reachability under disturbances.
- Conformal prediction and probabilistic safety filters as contrasting
  probabilistic uncertainty layers.

Learning context:

- Learning-augmented MPC and predictive safety filters.
- Learned shield approximations.
- Imitation or regret/ranking distillation where a learned component proposes
  actions but certified logic preserves guarantees.

## Scope Statement

PZR handles:

- bounded-memory monitor-side uncertainty tracking;
- policy-independent soundness over certified reducers;
- reducer selection that can optimize monitor precision or intervention
  quality.

PZR does not handle:

- choosing the safety specification;
- guaranteeing the sensor noise model is correct;
- blocking unsafe actuator commands by itself;
- replacing CBFs, predictive safety filters, or reachability tools.

PZR is most useful when sensor noise is bounded, fresh uncertainty accumulates
quickly, and the deployment horizon is long enough that exact monitor state is
not viable.

## Reference Hygiene

Before final paper writing, verify venue/year/bibtex details for all cited
shielding, CBF, predictive safety filter, and RTLola references. This file is a
positioning scaffold, not a bibliography source of truth.
