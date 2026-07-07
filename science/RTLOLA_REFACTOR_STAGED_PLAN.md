# RTLola Refactor Staged Plan

This plan was prepared for the `refactor/rtlola-learning-cleanup-plan` branch
after inspecting the current repository state, RTLola modules, learning code,
tests, benchmark tooling, and recent robot-arm orchestration scripts.

The refactor is correctness-first. Public behavior, RTLola semantics, binding
behavior, exact-reference metrics, and experiment interpretation must remain
unchanged unless a later reviewed stage explicitly approves a behavior change.

## Repository Responsibility Map

### Trace Loading And Adaptation

- `src/pzr/rtlola/scenarios.py`
  - Defines `RtlolaScenario` and `RtlolaTrace`.
  - Owns scenario registration and trace iteration.
  - Keeps trace identity, trace kind, length, and event dictionaries separate.
- `src/pzr/rtlola/robot_arm.py`
  - Packages the RLolaEval robot-arm spec and six trace CSVs.
  - Validates expected specification hash and trace lengths.
  - Preserves trace kinds: `figure8`, `figure8_drift`, `random`,
    `random_violated`, `square`, `square_drift`.
- `src/pzr/rtlola/specs/`
  - Contains packaged RTLola specs. `robot_arm.lola` is the public source of
    trigger labels and public output metrics.

### RTLola Specification Handling

- `src/pzr/rtlola/binding.py`
  - Pins the binding revision, interpreter revision, and release-build
    requirement.
  - Validates that the active binding matches expected native assumptions.
- `src/pzr/rtlola/engine.py`
  - Creates live and planner monitors from the spec and input stream names.
  - Owns monitor lifecycle, branch execution, live execution, and planner
    restore behavior.
  - Exposes native state matrices, native metrics, trigger/public outputs, and
    binding-native approximation loss.

### Binding Wrapper

- `src/pzr/rtlola/engine.py`
  - Wraps `rlola_python_binding.ZonotopeConfig` and monitor APIs.
  - Applies candidate actions only through binding-native transforms.
  - Reconstructs compact exact-reference interval states only for offline loss
    evaluation.
  - Restores planner state in `finally` blocks around speculative execution and
    exact-reference loss probes.

### Native Transform Catalog

- `src/pzr/rtlola/actions.py`
  - Defines `RtlolaAction`.
  - Maps method names to native `ZonotopeConfig` transforms.
  - Defines default bounded static candidates and MPC candidates.
  - Keeps `budget` as the native binding transform bound.
  - Distinguishes exact baseline `none`, fallback-only `interval`, and current
    bounded static/MPC candidates.

### Action And Candidate Definitions

- `src/pzr/rtlola/actions.py`
  - Owns candidate action construction and compatibility aliases.
- `src/pzr/rtlola/search.py`
  - Consumes the action catalog for static, beam, and MPC policies.
- `src/pzr/rtlola/benchmark.py`
  - Builds method sets and benchmark configurations from CLI/options.
- `src/pzr/rtlola/learning.py`
  - Reuses action names and catalogs for teacher generation and learned direct
    policy execution.

### Monitor Execution

- `src/pzr/rtlola/engine.py`
  - Primary monitor execution boundary.
  - `live_step` advances the live monitor.
  - `branch_step` speculatively advances the planner monitor and restores it.
  - `approx_loss` and `approx_loss_reference` defer to native loss semantics.
- `src/pzr/rtlola/benchmark.py`
  - Drives event loops, applies selected actions, and records per-step metrics.
- `src/pzr/rtlola/search.py`
  - Calls the engine for speculative rollout and action selection.

### MPC And Teacher Search

- `src/pzr/rtlola/search.py`
  - Implements static action selection, beam search, MPC terminal search, and
    Girard-tail variants.
  - Owns horizon/beam/tail options, root-candidate diagnostics, fallback and
    infeasible-candidate accounting.
- `src/pzr/rtlola/learning.py`
  - Uses forced-root beam search to build teacher labels/regrets.
  - Evaluates direct learned policies through native candidate actions.

### Exact Reference Evaluation And Caching

- `src/pzr/rtlola/benchmark.py`
  - Defines reference modes and exact-reference cache behavior.
  - Computes trigger verdicts and compact total-state center/radius references
    offline.
  - Maintains cache schema and metadata.
- `src/pzr/rtlola/engine.py`
  - Converts exact-reference center/radius data to the compact interval form
    used by native approximation loss.
  - Ensures the live planner is restored after offline loss probes.
- `tests/test_rtlola_binding.py`
  - Exercises exact cache schema, exact loss, and non-mutating behavior.

### Metric Aggregation And Reporting

- `src/pzr/rtlola/metrics.py`
  - Computes generator counts, active/zero/dynamic/constant distinctions, and
    `state_width`.
- `src/pzr/rtlola/benchmark.py`
  - Builds step records, run summaries, method summaries, artifact files, and
    benchmark data frames.
- `src/pzr/rtlola/sweep_report.py`
  - Consolidates result directories.
  - Produces completion tables, primary metrics, best-static comparisons, MPC
    composition/follow-through/deferral summaries, and LaTeX/table outputs.

### CLI And Benchmark Orchestration

- `src/pzr/rtlola/cli.py`
  - Defines `pzr-benchmark` CLI arguments and dispatches benchmark execution.
- `tools/run_rtlola_robot_arm_fpr_overnight.sh`
  - Source-aware, resumable full robot-arm benchmark orchestration.
  - Prepares exact references once per trace, then runs trace/budget/method
    stages.
- `tools/run_rtlola_mpc_variant_study.sh`
  - Focused MPC variant orchestration.
- `tools/run_rtlola_mpc_vs_girard_quick.sh`
  - Quick Girard/MPC smoke comparison.

### Learning Data, Model, Training, And Inference

- `src/pzr/rtlola/learning.py`
  - RTLola-specific teacher generation, feature extraction, artifact writing,
    training/evaluation orchestration, and learned direct-policy inference.
- `src/pzr/learning/ranking.py`
  - Generic numpy regret-ranking MLP, dataset helpers, training loop,
    prediction, save/load, and deterministic behavior checks.
- `tests/test_rtlola_learning.py`
  - Covers teacher behavior, direct learned policy execution, seed overlap, and
    robot-arm pipeline smoke behavior.
- `tests/test_ranking.py`
  - Covers generic ranker roundtrip and deterministic model behavior.

## Correctness Invariants

- Only `rlola_python_binding.ZonotopeConfig` transforms may mutate monitor
  state.
- Selectors may inspect monitor state and choose actions, but Python must not
  perform matrix writeback or implement reducers.
- `budget` remains the native binding transform bound.
- Python must not subtract a fresh-generator reserve or treat post-event dense
  slots as a violation.
- Binding-native approximation loss is authoritative for MPC, teacher search,
  learned evaluation, and reported approximation-loss metrics.
- No width, trigger-straddling, or Python proxy metric may replace native loss
  during cleanup.
- Exact references remain offline-only and must not mutate the live planner
  monitor.
- Exact-reference probes must restore planner state in `finally`.
- Exact-reference caches retain exact trigger booleans and compact center/radius
  data, not opaque monitor states or full generator matrices.
- Benchmark reference mode controls offline metrics and caching only. MPC and
  teacher searches must continue constructing their own unreduced rollouts.
- Trigger labels and public metrics come from
  `src/pzr/rtlola/specs/robot_arm.lola`.
- Robot-arm spec and trace assets must remain packaged at the current RLolaEval
  revision and validated by current hashes/lengths.
- Trace kinds must remain separate. Do not pool drift, random, violated, and
  structured traces without preserving `trace_kind`.
- Dynamic, active, zero, and constant generator counts must remain distinct.
- Constant encoder-calibration slack must remain preserved by dynamic
  reduction.
- `state_width` remains the existing dynamic-state interval-width sum and
  excludes constant slack.
- FPR uses exact negative steps as denominator.
- FNR uses exact positive steps as denominator.
- `final_approx_loss` remains the final event binding loss.
- `sum_approx_loss` remains the unweighted sum of per-event binding losses and
  is comparable only within the same trace.
- `verdict` reference mode remains available for trigger-only runs.
- Candidate method defaults remain unchanged: current robot-arm MPC/learning
  candidates are `girard`, `scott`, `interval_hull`, `pca`, `combastel`, and
  deterministic `clustering`; `none` remains exact baseline/automatic
  under-bound action; `interval` remains fallback-only.
- Existing method names and emitted MPC semantics remain unchanged:
  `mpc_terminal_beam`, `mpc_terminal_girard_tail`,
  `mpc_cumulative_girard_tail`, and `mpc_one_step_girard_rollout`.
- Completed and partial experiment artifacts must not be reinterpreted by a
  cleanup patch.

## Complexity Hotspots

- `src/pzr/rtlola/benchmark.py`
  - Large mixed-responsibility module.
  - Combines configuration, method-set expansion, execution, exact-reference
    caching, metric collection, artifact generation, and summary tables.
  - Exact-reference cache logic is tightly coupled to benchmark execution.
  - Per-step record construction is central but broad and fragile.
- `src/pzr/rtlola/search.py`
  - Search policy code mixes action selection, MPC rollout mechanics,
    diagnostics, fallback accounting, and variant-specific scoring.
  - Method/candidate assumptions are partly mirrored in benchmark and learning.
- `src/pzr/rtlola/learning.py`
  - RTLola-specific learning module currently spans teacher generation,
    feature construction, model training, direct inference, evaluation, and
    artifact output.
  - Candidate alignment and teacher-forced root behavior are important but not
    isolated as contracts.
- `src/pzr/rtlola/sweep_report.py`
  - Reporting logic is broad and schema-sensitive.
  - It mixes loading, validation, summary computation, comparison tables, and
    optional table formatting.
- Configuration boundaries
  - CLI, benchmark configs, search options, learning options, and tools encode
    related defaults in separate places.
  - This increases risk when adding learning-pipeline variants.
- Compatibility and naming
  - Some paths exist for compatibility or fallback behavior and should not be
    removed until tests prove they are unused.
  - Several names are operational rather than semantic, which makes it easier
    to confuse native loss, width metrics, exact references, and live planner
    behavior.

## Staged Implementation Plan

Each stage should be reviewed and implemented as a small patch. Later stages
may be adjusted based on findings from earlier tests, but semantic invariants
above remain fixed.

### Stage 0: Persist Plan And Establish Baseline

**Objective**

- Save this plan in the repository.
- Confirm the branch and working tree state.
- Establish current test behavior before code changes.
- Capture any environment limitation, especially binding availability.

**Likely affected files**

- `science/RTLOLA_REFACTOR_STAGED_PLAN.md`
- No source code changes.

**Behavior that must remain unchanged**

- All runtime behavior.
- All generated artifacts under `results/`.
- All benchmark, cache, CLI, and learning semantics.

**Tests to run or record**

- `git status --short --branch`
- Fast pure tests:
  - `PYTHONPATH=src pytest tests/test_rtlola_units.py tests/test_ranking.py`
- If binding environment is available:
  - `LD_PRELOAD="$PWD/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so" PYTHONPATH=src external/miniconda3/envs/pzr-robot-arm/bin/python -m pytest tests/test_rtlola_binding.py tests/test_rtlola_learning.py`
- Optional smoke after binding tests pass:
  - `pzr-benchmark --profile smoke --scenario omni_robot --method-set core --output /tmp/pzr-omni-refactor-baseline`
  - `tools/run_rtlola_robot_arm.sh --length 20 --seeds 1 --method-set core --output /tmp/pzr-arm-refactor-baseline`

**Rollback risk**

- Very low. Documentation-only plus test observation.

**Ordering rationale**

- This must happen first so later changes can be compared against a recorded
  baseline and the staged plan survives compaction.

### Stage 1: Clarify Method And Candidate Configuration

**Objective**

- Centralize method-set and candidate-definition contracts without changing
  defaults.
- Make it clear which actions are static candidates, MPC candidates, exact
  baseline, fallback-only, and learning candidates.
- Reduce duplicated method-name assumptions before changing execution,
  reporting, or learning boundaries.

**Likely affected files**

- `src/pzr/rtlola/actions.py`
- `src/pzr/rtlola/search.py`
- `src/pzr/rtlola/benchmark.py`
- `src/pzr/rtlola/cli.py`
- Possibly `tests/test_rtlola_units.py` and `tests/test_rtlola_binding.py`

**Behavior that must remain unchanged**

- Public method names.
- Method-set expansion.
- CLI defaults and accepted values.
- Bounded static candidate defaults.
- MPC candidate defaults.
- Learning candidate alignment.
- `none` exact baseline behavior.
- `interval` fallback-only behavior.
- `budget` passed unchanged to native transforms.

**Tests to run or add**

- Pure tests for method-set expansion and candidate membership.
- Pure tests that unsupported candidate names still fail the same way.
- Binding candidate tests in `tests/test_rtlola_binding.py`.
- Existing learning candidate-alignment tests.

**Rollback risk**

- Low to medium. Most risk is accidental candidate/default drift.

**Ordering rationale**

- Configuration clarity should precede execution and learning refactors because
  both depend on the same method/candidate contracts.

### Stage 2: Separate Execution And Search From Step Measurement

**Objective**

- Keep monitor execution and search decisions separate from per-step metric row
  construction.
- Introduce a narrow internal result contract for "selected action plus
  diagnostics" if useful.
- Preserve `make_step_record` as the single place that maps an executed step to
  public row fields.
- Make learned evaluation reuse the same measurement path where practical.

**Likely affected files**

- `src/pzr/rtlola/benchmark.py`
- `src/pzr/rtlola/search.py`
- `src/pzr/rtlola/learning.py`
- `tests/test_rtlola_units.py`
- `tests/test_rtlola_binding.py`
- `tests/test_rtlola_learning.py`

**Behavior that must remain unchanged**

- Per-step row fields and meanings.
- Step ordering and event indices.
- Trigger confusion accounting.
- `post_event_over_bound` semantics.
- Fallback and infeasible-candidate accounting.
- MPC root diagnostics.
- Native approximation-loss source.
- Live monitor mutation order.
- Planner restore behavior.

**Tests to run or add**

- Pure tests for fake-engine MPC fallback/diagnostic semantics.
- Binding tests for deterministic branching and outer-bound soundness.
- Binding benchmark artifact tests.
- Learning direct-inference artifact tests.

**Rollback risk**

- Medium. This touches event-loop structure and can subtly affect metrics.

**Ordering rationale**

- This follows Stage 1 so method/candidate contracts are stable.
- It should precede exact-reference extraction because exact metrics attach to
  the same step records.

### Stage 3: Extract Exact-Reference Cache And Offline Metric Logic

**Objective**

- Move exact-reference cache schema and offline metric helpers out of the main
  benchmark loop into a focused module.
- Make "offline reference data" explicit and separate from "live planner
  monitor state."
- Keep cache loading, validation, writing, and native loss comparison behavior
  unchanged.

**Likely affected files**

- `src/pzr/rtlola/benchmark.py`
- New module such as `src/pzr/rtlola/reference.py`
- `src/pzr/rtlola/engine.py` only if names/contracts need clarification
- `tests/test_rtlola_binding.py`
- Possibly `tests/test_rtlola_units.py`

**Behavior that must remain unchanged**

- Exact cache schema version and metadata meaning.
- Cache paths and reuse behavior.
- Trace hash validation.
- Reference modes `none`, `verdict`, and `exact`.
- Exact trigger booleans.
- Compact total-state center/radius data.
- No opaque states or full generator matrices in caches.
- Exact-reference loss computation through native `approx_loss`.
- Candidate applied only to planner monitor during exact loss probe.
- Planner restored in `finally`.
- Live monitor never mutated by exact-reference metrics.

**Tests to run or add**

- Binding tests for exact cache creation, reuse, and schema validation.
- Binding tests for exact loss and planner restoration.
- Pure tests for cache metadata validation if feasible.
- Smoke benchmark with exact reference mode on a short trace.

**Rollback risk**

- Medium. Cache compatibility and non-mutating behavior are high-value
  correctness contracts.

**Ordering rationale**

- This depends on Stage 2's cleaner step-measurement boundary.
- It should precede reporting cleanup because reporting consumes exact metric
  outputs and cache-derived columns.

### Stage 4: Simplify Reporting And Table Generation

**Objective**

- Separate benchmark artifact writing from result-directory aggregation and
  table generation.
- Keep schema-sensitive reporting helpers focused and testable.
- Preserve generated filenames, column names, column meanings, and table
  semantics.

**Likely affected files**

- `src/pzr/rtlola/benchmark.py`
- `src/pzr/rtlola/sweep_report.py`
- New module such as `src/pzr/rtlola/tables.py` or focused helpers if useful
- `tests/test_rtlola_units.py`
- `tests/test_rtlola_binding.py`

**Behavior that must remain unchanged**

- Artifact filenames.
- CSV column names and meanings.
- Completion table semantics.
- `primary_metrics.csv` semantics.
- `mpc_vs_static_metrics.csv` best-static selection independently per metric.
- FPR/FNR definitions.
- Mean/final/max/summed native approximation-loss definitions.
- Mean/max `state_width` definitions.
- MPC composition, follow-through, and deferral summary meanings.

**Tests to run or add**

- Pure sweep-report tests for known small data frames.
- Pure trigger-confusion and metric aggregation tests.
- Binding benchmark artifact smoke to verify non-empty generated artifacts and
  expected columns.

**Rollback risk**

- Low to medium. The main risk is schema drift or column-order surprises.

**Ordering rationale**

- Reporting cleanup should happen after exact-reference extraction so table
  inputs have stable ownership.
- It should happen before broader learning cleanup because learning artifacts
  can then depend on clearer reporting contracts.

### Stage 5: Clarify Learning Interfaces Without Changing Model Behavior

**Objective**

- Split RTLola-specific learning responsibilities into clearer internal
  contracts: teacher generation, feature extraction, model training,
  direct-policy inference, evaluation, and artifact writing.
- Keep the generic ranker independent.
- Make candidate alignment and forced-root teacher behavior explicit.

**Likely affected files**

- `src/pzr/rtlola/learning.py`
- `src/pzr/learning/ranking.py` only for type/contract clarification if needed
- Possibly new focused helpers under `src/pzr/rtlola/`
- `tests/test_rtlola_learning.py`
- `tests/test_ranking.py`

**Behavior that must remain unchanged**

- Feature order and feature values.
- Regret target definitions.
- Forced-root teacher search semantics.
- Native direct-inference policy execution.
- Candidate alignment across teacher, training, and inference.
- Random seed handling and train/eval split behavior.
- Pooled budget and trace-kind handling.
- Artifact filenames and expected schemas.
- Model save/load compatibility unless a reviewed migration is introduced.

**Tests to run or add**

- Existing learning tests.
- Ranker save/load roundtrip tests.
- Tests for teacher root behavior and learned candidate alignment.
- Smoke learned benchmark:
  - `pzr-benchmark --profile smoke --scenario omni_robot --budget 10 --methods girard,mpc_terminal_beam --learned-mode regret --regret-iterations 1 --regret-epochs 2 --regret-train-seeds 1 --regret-eval-seeds 1 --output /tmp/pzr-learned-refactor`

**Rollback risk**

- Medium to high. Learning behavior is easy to perturb through feature order,
  labels, or candidate alignment.

**Ordering rationale**

- Learning cleanup comes after candidate, execution, exact-reference, and
  reporting boundaries are clearer.
- This minimizes the chance of mixing model-behavior changes with supporting
  infrastructure cleanup.

### Stage 6: Remove Proven-Dead Or Obsolete Paths

**Objective**

- Remove compatibility paths only after earlier stages and tests prove they are
  unused.
- Keep removals small and separately reviewable.

**Likely affected files**

- To be determined from coverage, search results, and Stage 1 through Stage 5
  findings.

**Behavior that must remain unchanged**

- Public CLI compatibility required by README, tools, and tests.
- Artifact schema compatibility for current active outputs.
- Exact-reference cache meaning.
- Candidate defaults and method names.
- Experiment interpretation for completed and partial runs.

**Tests to run or add**

- Full pure tests.
- Full binding-backed semantic tests.
- Robot-arm short smoke.
- Learned smoke if learning paths are touched.
- Grep-based check for removed public names in README, tools, and tests.

**Rollback risk**

- Medium. Deleting compatibility paths can break workflows that tests do not
  currently cover.

**Ordering rationale**

- This must be last because the earlier stages expose which paths are actually
  obsolete and which are still part of the public workflow.

## Test Matrix

### Fast Pure Tests

Use these for Stage 0 through Stage 6 when binding is not required:

```bash
PYTHONPATH=src pytest tests/test_rtlola_units.py tests/test_ranking.py
```

Expected coverage:

- Method-set and candidate configuration.
- Sweep/report table logic.
- Trigger confusion accounting.
- Generic ranker behavior.
- Fake-engine search invariants.

Stages that can initially rely on pure tests:

- Stage 0 documentation and baseline recording.
- Stage 1 configuration contracts, followed by binding confirmation.
- Stage 4 reporting logic, followed by artifact smoke.
- Stage 6 deletion candidates, only after binding validation for touched paths.

### Binding-Backed Semantic Tests

Use the pinned robot-arm environment:

```bash
LD_PRELOAD="$PWD/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so" \
PYTHONPATH=src external/miniconda3/envs/pzr-robot-arm/bin/python -m pytest \
  tests/test_rtlola_binding.py tests/test_rtlola_learning.py
```

Expected coverage:

- Binding revision and release-build assumptions.
- Deterministic state branching and tie behavior.
- Native transform-bound semantics.
- Dense versus active generator accounting.
- Outer-bound soundness against unreduced branches.
- Constant calibration generator preservation.
- Trigger/public-stream keys from packaged spec.
- Fallback and infeasible-candidate accounting.
- Exact cache schema and native exact-loss behavior.
- Learned candidate alignment and direct-inference behavior.

Stages requiring binding validation before acceptance:

- Stage 1, because candidate/default drift must be ruled out.
- Stage 2, because event-loop and planner behavior can change.
- Stage 3, because exact-reference metrics are binding-native and
  non-mutating.
- Stage 5, because learned direct inference and teacher labels depend on
  binding behavior.
- Stage 6, if any touched path reaches binding/search/learning execution.

### Smoke Benchmarks

Short omni-robot smoke:

```bash
pzr-benchmark --profile smoke --scenario omni_robot --method-set core \
  --output /tmp/pzr-omni-refactor-smoke
```

Short robot-arm smoke:

```bash
tools/run_rtlola_robot_arm.sh --length 20 --seeds 1 --method-set core \
  --output /tmp/pzr-arm-refactor-smoke
```

Learned smoke:

```bash
pzr-benchmark --profile smoke --scenario omni_robot --budget 10 \
  --methods girard,mpc_terminal_beam --learned-mode regret \
  --regret-iterations 1 --regret-epochs 2 \
  --regret-train-seeds 1 --regret-eval-seeds 1 \
  --output /tmp/pzr-learned-refactor-smoke
```

Expected coverage:

- CLI argument wiring.
- Non-empty benchmark artifacts.
- Method summaries.
- Learning artifact production.
- Basic search and learned-policy integration.

### Artifact And Schema Checks

Check after Stages 2 through 5:

- Non-empty `steps.csv`.
- Non-empty method summary outputs.
- Expected `trace_kind`, `method`, `budget`, FPR/FNR, state-width, and loss
  columns.
- `primary_metrics.csv` compact completion table semantics.
- `mpc_vs_static_metrics.csv` best-static comparison semantics.
- Exact-reference cache metadata, schema version, and trace hash behavior.
- No generated files under `results/` modified by cleanup commits.

### Learning-Pipeline Checks

Check after Stage 5 and any Stage 6 learning deletion:

- Teacher root behavior is unchanged.
- Candidate order and model output alignment are unchanged.
- Direct learned inference still applies native actions only.
- Regret labels and feature vectors are stable.
- Seed overlap checks continue to catch invalid configurations.
- Ranker save/load roundtrip remains deterministic.

## Non-Goals

- Do not change packaged robot-arm spec or traces.
- Do not change binding revision, interpreter revision, or release-build
  assumption.
- Do not change transform semantics.
- Do not add or remove robot-arm experiment candidates during cleanup.
- Do not change candidate method defaults.
- Do not reinterpret `budget`.
- Do not introduce Python-side reducers or matrix writeback.
- Do not replace native approximation loss.
- Do not change exact-reference cache meaning.
- Do not edit generated files under `results/`.
- Do not reinterpret completed or partial experiment artifacts.
- Do not quote the terminated partial Girard-versus-MPC run as a completed
  evaluation.
- Do not merge trace kinds without preserving `trace_kind`.
- Do not alter FPR/FNR/state-width/final-loss/sum-loss definitions.
- Do not change learning model behavior, feature order, label semantics, or
  artifact schemas during infrastructure cleanup.

## Open Questions And Risks

- The current plan assumes all active public workflows are represented in
  README, tools, tests, and current CLI entry points. Stage 6 must verify this
  before deleting compatibility paths.
- Binding-backed validation may be environment-sensitive. If the pinned
  environment is unavailable, code stages that touch execution, search, exact
  references, or learning should not be accepted as complete.
- Reporting cleanup must be conservative because downstream scripts may depend
  on exact column names and filenames.
- Exact-reference extraction is high leverage but risky because cache reuse,
  native loss, and non-mutating planner behavior meet there.
- Learning cleanup should avoid changing model results unless a separate
  reviewed experiment intentionally changes features, candidates, labels, or
  training defaults.

## Readiness Checklist Before Implementation

- [ ] Plan file committed or explicitly accepted as uncommitted documentation.
- [ ] Working tree checked.
- [ ] Baseline pure tests run or environment limitation recorded.
- [ ] Binding test availability checked before any semantic stage.
- [ ] Stage 1 restated with exact changes, non-changes, and tests before
      editing source files.
