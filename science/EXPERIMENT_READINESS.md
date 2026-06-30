# Experiment Readiness

An experiment is ready only when:

- the pinned binding builds and all binding-backed tests pass;
- every bounded native transform outer-bounds the unreduced branch in the
  soundness regression;
- robot-arm constant calibration columns are unchanged by reduction;
- the configured bound is passed unchanged to every bounded transform;
- dense, active, zero, and constant generators are interpreted separately;
- exact-reference approximation loss and trigger outcomes are available;
- learned candidates exactly match the MPC candidate catalog;
- held-out learned rows record real decision time and fallback metadata;
- all generated CSV, YAML, PDF, PNG, policy, and metadata artifacts are
  non-empty.

Use `/tmp` for smoke outputs. Serious outputs belong under a new `results/`
directory and must be generated through `pzr-benchmark`.

The retired Python monitors, robotics replay/probe paths, drone/F1TENTH
sidecars, and old paper wrappers are not valid experiment entry points.
