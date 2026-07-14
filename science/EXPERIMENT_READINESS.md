# Experiment Readiness

An experiment is ready only when:

- the pinned binding builds and all binding-backed tests pass;
- the installed binding reports the pinned interpreter and release build profile;
- every bounded native transform outer-bounds the unreduced branch in the
  soundness regression;
- robot-arm constant calibration columns are unchanged by reduction;
- the configured bound is passed unchanged to every bounded transform;
- dense, active, zero, and constant generators are interpreted separately;
- exact-reference approximation loss and trigger outcomes are available;
- learned candidates exactly match the MPC candidate catalog;
- learning splits are disjoint by trajectory seed and preserve trace kind;
- teacher labels use short online unreduced rollouts, not offline exact caches;
- direct inference reads no future events and performs no planner rollout;
- held-out learned rows record real decision time and fallback metadata;
- all generated CSV, YAML, PDF, PNG, policy, and metadata artifacts are
  non-empty.
- incomplete transform runs are recorded and excluded from aggregates.

The primary overnight method list contains Girard, Scott, PCA, Combastel, and
beam MPC. The MPC and learning candidate catalog contains the same four
bounded reducers. Interval hull is excluded because it was consistently poor
in short exact-reference screens. Deterministic clustering is excluded because
its extreme losses dominated cost-sensitive ranking and its frequent interval
fallback obscures standalone behavior. Althoff A, colinear scale, and the
randomized/diverse clustering reducers are excluded because they are not
tractable or robust at robot-arm sweep length.

Use `/tmp` for smoke outputs. Serious outputs belong under a new `results/`
directory and must be generated through `pzr-benchmark`.

The retired Python monitors, robotics replay/probe paths, drone/F1TENTH
sidecars, and old paper wrappers are not valid experiment entry points.
