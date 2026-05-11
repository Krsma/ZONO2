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

## Code Mapping

The initial benchmark is the paper's harder omnidirectional robot example. The
black-box adapter in `pzr.benchmarks.robot` tracks:

- filtered acceleration,
- velocity,
- current distance increment,
- x position,
- y position.

It exposes only the `MonitorAdapter` methods and trigger metadata. The
controller can step, clone, and replace the zonotope component of the state,
but it does not inspect equations or depend on monitor internals.

Reducers currently include:

- `BoxReducer`: replace a zonotope by its interval hull.
- `ScoredKeepReducer.by_norm`: keep large generators and box-merge the rest.
- `ScoredKeepReducer.calibration_aware`: keep important generators while giving
  calibration generators a strong preservation bonus.

The MPC policy compares these certified actions by predicted trigger width,
threshold straddling, and generator count.

## Current Scope

Version 1 intentionally does not implement an RLola parser, gray-box
sensitivity hooks, learned predictors, or CORA/MATLAB integration. Those can
be added behind the existing adapter, reducer, and policy interfaces without
changing the soundness boundary.
