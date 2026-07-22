# Experiment Readiness

An experiment is ready only when:

- the pinned binding builds and all non-parity binding-backed paper-preflight
  tests pass; the unfiltered release suite and standalone parity diagnostic
  remain development checks outside the paper run;
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
- every paper cell identity includes trace/config/source/model/cache hashes and
  the complete typed method configuration;
- interval fallback changes the run state to `fallback_failed`, makes headline
  FPR and completed-run throughput unavailable, and retains pre-fallback
  diagnostics;
- every learned evaluation row records a model trained only at the same exact
  budget, and all seven specialist trainings use the shared fixed hyperparameters;
- the 112-cell nominal pilot projection is at most 72 hours with four one-thread
  workers, or explicit approval is recorded before the 1,120-cell nominal
  held-out sweep; the gate excludes fixed headline/objective/timing work;
- headline aggregation is trace-level, paired bootstrap intervals use 10,000
  deterministic seed-level replicates, and failed points are not connected;
- the timing stage uses one worker and one native thread, with warm-up,
  reference preparation, trace generation, and artifact I/O excluded.
- generated training, pilot, held-out, and H/W traces are nominal-only, while
  the four imported figure-eight variants retain their pinned hashes and lengths;
- nominal and fixed patterned aggregates and reducer composition remain
  separate, with no randomized drift/geofence generalization claim.

The primary overnight method list contains Girard, Scott, PCA, Combastel, and
beam MPC. The MPC and learning candidate catalog contains the same four
bounded reducers. Interval hull is excluded because it was consistently poor
in short exact-reference screens. Deterministic clustering is excluded because
its extreme losses dominated cost-sensitive ranking and its frequent interval
fallback obscures standalone behavior. Althoff A, colinear scale, and the
randomized/diverse clustering reducers are excluded because they are not
tractable or robust at robot-arm sweep length.

Use `/tmp` for smoke outputs. Serious paper outputs belong under
`results/paper-evaluation-v2` and must be generated through
`tools/run_paper_evaluation.sh`; ordinary benchmark diagnostics still use
`pzr-benchmark`.

The retired Python monitors, robotics replay/probe paths, drone/F1TENTH
sidecars, and old paper wrappers are not valid experiment entry points.
