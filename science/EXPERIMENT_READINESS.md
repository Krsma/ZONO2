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
- the v3 run verifies the frozen Clean148 matrix, optimizer seed 42, 148 nominal
  training seeds, six validation seeds, source-dataset hashes, and seven model
  hashes before copying any specialist into the paper namespace;
- the 112-cell nominal pilot projection is at most 72 hours with ten one-thread
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

The v4 finalization is ready for independent Raspberry Pi timing only when:

- `preflight`, `prepare`, `pilot`, `nominal`, and `prediction-ablation` are
  complete under `results/paper-evaluation-v4`, with exactly 1,400 nominal and
  15 predictor cells;
- nominal seeds 348--367 cover all ten stable method identities and all seven
  bounds exactly once, and every completed nominal cell has zero false
  negatives and respects the trace-length-independent generator bound;
- every imported v3 fixed-trace, objective, and H/W artifact, every frozen
  specialist and selection decision, and every binding, interpreter,
  specification, trace, source, and tool pin matches the v4 contract;
- the Pi bundle command verifies the frozen parent and copies only the inputs
  required for timing; the extracted bundle remains byte-for-byte immutable,
  while its environment and run output live in separate directories;
- Pi preflight records a 64-bit AArch64 Raspberry Pi 5, the pinned release
  binding/interpreter, one native thread, the selected isolated CPU, and the
  requested host controls before measurement;
- all 350 Pi timing cells use events 0--99 only as warm-up and events 100--299
  as the measured window, with one sequential pass and the prescribed rotated
  method order;
- the primary latency is the unchanged `decision_time_ms` path -- reducer
  selection plus live native commit -- while exact-reference computation,
  predictor diagnostics, and artifact I/O are excluded; predictor, selection,
  and commit phases are reported separately only after semantic parity passes;
- native or infrastructure failures abort, expected reducer infeasibility is
  explicitly unavailable, and packing requires all ten phase-profile cells;
- the imported Pi manifest and every contained file pass hash validation, and
  the combined report pairs exactly 350 Pi cells with the corresponding v4
  nominal identities without modifying either parent.

Once Pi timing is the paper-facing runtime source, a workstation v4 `runtime`
stage is diagnostic only. We do not mix its distributions with Pi measurements,
and we do not treat the original v4 `report` or `validate` stage as validation
of the Pi-timed final artifact. The final release gate is the immutable v4
scientific parent, the validated Pi timing archive, and the separately
versioned combined-report manifest.

The optional DART rescue study is ready only when:

- its PZR source hash, release binding, teacher dataset, seven Clean20 models,
  replay traces, fixed traces, and imported exact references match the validated
  `paper-evaluation-v2` manifests;
- each DART calibration uses only validation rows from the matching exact
  budget, and every Clean36/DART36 model is dispatched only at that budget;
- seeds 26--41 are paired additional training paths, seeds 100--119 are labelled
  retrospective replay, and untouched seeds 120--139 remain the confirmation
  bank;
- the 112 clean and 112 DART collection shards, fourteen new specialists, 924
  reported cells, and 756 new evaluation cells validate exactly;
- DART disturbances never exceed their normalized-regret cap and a forced
  recovery prevents consecutive disturbances;
- paired bootstrap intervals use nominal trajectories as the independent unit,
  while fixed figure-eight effects remain descriptive and separate.

The primary overnight method list contains Girard, Scott, PCA, Combastel, and
beam MPC. The MPC and learning candidate catalog contains the same four
bounded reducers. Interval hull is excluded because it was consistently poor
in short exact-reference screens. Deterministic clustering is excluded because
its extreme losses dominated cost-sensitive ranking and its frequent interval
fallback obscures standalone behavior. Althoff A, colinear scale, and the
randomized/diverse clustering reducers are excluded because they are not
tractable or robust at robot-arm sweep length.

Use `/tmp` for smoke outputs. Serious paper outputs belong under
`results/paper-evaluation-v3` and must be generated through
`tools/run_paper_evaluation.sh`; ordinary benchmark diagnostics still use
`pzr-benchmark`. DART rescue outputs belong under `results/dart-rescue-v1` and
must be generated through `tools/run_dart_rescue.sh`.

V4 scientific outputs belong under `results/paper-evaluation-v4`. Pi timing and
the joined final report belong exclusively under
`results/paper-evaluation-v4-pi-timing-v1` and
`results/paper-evaluation-v4-final-report-v1`, respectively.

The retired Python monitors, robotics replay/probe paths, drone/F1TENTH
sidecars, and old paper wrappers are not valid experiment entry points.
