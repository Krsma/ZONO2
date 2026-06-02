# Codebase Audit — May 2026

Findings from a deep audit of `src/pzr/` after the eval-suite extension
(Cutting-Corners alignment + DAgger default + budget sweep + `trigger_zonotope`
protocol method). The suite now has 215 tests after adding regressions for the
fixed issues. Severity reflects research impact, not surface count.

**Implementation status:** H1, H2, M1, and M2 were implemented after this
audit update using trigger-zonotope callbacks, a shared protected-reduction
helper, and pre-reduction DAgger trace features. Remaining M/L items are still
open unless noted otherwise.

---

## H1 — `trigger_zonotope` not propagated to MPC cost (HIGH, robot_arm only; fixed)

**File:** `src/pzr/mpc/objectives.py:38`

```python
def __call__(self, state: MonitorState, verdicts=None) -> float:
    z = state.zonotope        # <-- should be monitor.trigger_zonotope(state)
    widths = z.widths()
    ...
    for trigger in triggers:
        w = float(widths[trigger.state_index])   # joint-space widths,
        ...                                       # Cartesian indices
```

`WeightedZonotopeCost` evaluates trigger width and straddling on the raw state
zonotope. For `omni_robot`, `simple_robot`, `point_mass` this happens to be
correct because their `trigger_zonotope` returns `state.zonotope`. For
**`robot_arm`** the state is 6D joint space (θ₁,θ₂,θ₃,ω₁,ω₂,ω₃) while triggers
use `state_index=0`/`1` meaning Cartesian `EE_X`/`EE_Y`. Result: the MPC cost
function reads `widths[0]` (a joint-angle width) and treats it as an
end-effector width. MPC decisions on `robot_arm` are optimizing the wrong
quantity.

**Why it slipped:** `monitor.trigger_zonotope()` was added recently and
`runner._trigger_metrics()` was updated to use it (`runner.py:144-156`), but
`WeightedZonotopeCost` was not. `WeightedZonotopeCost.__call__` doesn't even
receive a `monitor` parameter, so the fix requires a signature change OR
having the cost hold a reference to the monitor at construction time.

**Implemented fix:** Add a trigger-zonotope callback to `WeightedZonotopeCost`
construction; in `__call__` evaluate trigger width/straddling on
`trigger_zonotope(state)` while keeping generator-count cost on the raw monitor
state. `benchmark.default_methods` already has the monitor in scope
(`benchmark.py:135`).

**Why it's not "critical":** `robot_arm` MPC results are still *sound* — the
chosen reducer is still certified — but their *quality claims* are not
defensible. Don't include `robot_arm` MPC numbers in the paper until this is
fixed and re-run.

---

## H2 — `trigger_zonotope` not propagated to DAgger features (HIGH, robot_arm only; fixed)

**File:** `src/pzr/imitation/features.py:55, 67-76`

```python
z = state.zonotope        # <-- should be monitor.trigger_zonotope(state)
...
widths = z.widths()
...
if triggers:
    lower, upper = z.interval_bounds()
    for t in triggers:
        w = float(widths[t.state_index])    # same joint-vs-Cartesian
        ...                                  # mismatch as H1
```

Same pattern, different consumer. The trigger-proximity features
(`trigger_width_sum`, `trigger_width_mean`, `trigger_straddle_count` —
positions 12, 13, 14 of the 24-dim feature vector) are computed on joint-space
widths with Cartesian indices for `robot_arm`. Features become meaningless on
that scenario; on the others they're correct.

`extract_features` does not currently receive trigger-zonotope context. The
call sites (`runner.py:220`, `dagger_eval.py:54`, `dagger_eval.py:146`) have
`monitor` in scope, so threading `monitor.trigger_zonotope` through is a
one-parameter change.

**Practical impact:** smaller than H1 — the feature vector is 24-dim and the
learned policy may compensate via the other 21 features. But you cannot claim
"learned policy uses trigger geometry on robot_arm." It currently does not.

**Post-fix verification (2026-05-30):** `dagger_eval.py:65` looks like a fourth
`extract_features` site at first glance but is actually the fallback
`BoxReducer.reduce(...)` invocation when the learned policy fails to pick a
candidate. The three real feature-extraction call sites
(`runner.py:206-209`, `dagger_eval.py:54-57`, `dagger_eval.py:151-154`) all
pass `trigger_zonotope=monitor.trigger_zonotope`. No remaining caveat.

---

## M1 — `ProtectedReducer` extends the `Reducer` Protocol without satisfying it (MEDIUM; fixed)

**File:** `src/pzr/zonotope/protected.py:30-35`

```python
def reduce(self, z: Zonotope, budget: int,
           protected_indices: tuple[int, ...] = ()) -> ReductionResult:
```

The third positional parameter breaks the Reducer Protocol contract
(`reduce(z, budget) -> ReductionResult`). When `ProtectedReducer` is passed
into `tree_search`, `MPCPolicy`, or learned-policy candidate selection as a
`Reducer`, the search code calls `reducer.reduce(s.zonotope, budget)` —
silently passing no `protected_indices`, so it degenerates to the base reducer
with zero protection. Calibration generators are *not* protected during MPC
tree search or learned-policy candidate attempts.

**Evidence:** `mpc/search.py:49` calls `result = reducer.reduce(s.zonotope, budget)`.
`mpc/policies.py:141,182` do the same for rollout first actions and future
rollout reductions. `imitation/policy.py:76` does the same inside
`LearnedPolicy.select_reducer`. `StaticReductionPolicy.decide`
(`runner.py:62-65`) correctly handles this by checking
`isinstance(self.reducer, ProtectedReducer)` and threading
`protected_indices=cal`. MPC and learned selection have no equivalent.

**Practical impact:** `mpc_rollout` and `mpc_sequence` on `omni_robot`
(1 cal gen), `simple_robot` (2), `point_mass` (2), `robot_arm` (3) may
silently lose calibration generators across MPC-induced reductions. This may
be why MPC's trigger-width advantage over Girard is small in some configs —
they're not actually preserving the same persistent state. Learned-policy
evaluation has the same risk for all non-fallback candidate attempts. Verify by
checking whether MPC and learned-policy runs maintain calibration_indices
length and exact protected columns across long traces.

**Implemented fix:** Keep the `Reducer` protocol stable and introduce one helper
that applies `ProtectedReducer` with `state.calibration_indices`. Use it in
static policy, MPC tree search, rollout MPC, and learned-policy selection.

---

## M2 — DAgger expert traces record post-reduction features with pre-reduction labels (MEDIUM; fixed)

**File:** `src/pzr/experiments/runner.py:208-220`

```python
decision = policy.decide(...)
state = decision.state
...
features = extract_features(state, budget, monitor.triggers)
trace_collector.record(ReductionTrace(features=features, action=reducer_used, ...))
```

Expert trace collection records the feature vector after the expert reduction
has already been applied, but labels it with the reducer chosen for the
pre-reduction overflow state. This produces supervised rows whose state/action
pair does not match the decision the expert actually made.

**Practical impact:** DAgger training quality can degrade because examples are
shifted one reduction later in feature space. The issue is independent from
H2, but the same call site should pass trigger-zonotope context once it records
the correct pre-reduction state.

**Implemented fix:** In `run_single`, compute `decision_features` before
`policy.decide(...)` mutates the run state, then record those pre-reduction
features after timing/decision metadata is available.

---

## M3 — `runner.summarize_results` computes `abs_error_range` per seed, not per method (MEDIUM/LOW)

**File:** `src/pzr/experiments/runner.py:317`

```python
"abs_error_range": float(np.max(approx_errors) - np.min(approx_errors)),
```

The Cutting-Corners paper defines absolute error range as `max - min` of
per-step approximation error *across the trajectory*. The code computes it
per-seed, which matches that definition. But then `aggregate_summary`
averages this per-seed value across seeds. This produces "mean of per-seed
ranges" rather than the cleaner "range across the pooled trajectory" the
paper plots.

**Practical impact:** Low — the metric is still a valid seed-level summary and
the paper text does not force the exact aggregation implemented here. Document
this choice or refactor to pool first, then range if the paper figure needs
that specific convention.

---

## M4 — `extract_features` includes `gen_condition` which can spike under near-singular generators (MEDIUM, potentially-explains-PCA-blowup)

**File:** `src/pzr/imitation/features.py:97-101`

```python
if n_gen >= n_dim and n_dim > 0:
    sv = np.linalg.svd(G, compute_uv=False)
    if sv[-1] > 1e-12:
        gen_condition = float(sv[0] / sv[-1])
```

When `sv[-1]` is barely above `1e-12`, `gen_condition` can reach 10¹⁰+. The MLP
training path standardizes features, but extreme outliers can still dominate
the feature mean/std and may interact badly with PCA's known instability on
near-singular matrices. This is a candidate explanation for the **deferred PCA
blowup investigation** (50× trigger width with CI [607, 5350] on omni_robot,
per `results/first_run/`). Worth checking whether PCA's reduced generator
matrices are ill-conditioned in the failing seeds.

---

## L1 — Documentation drift (LOW, but pervasive)

Three top-level docs describe a layout that no longer exists in source.

- `README.md` references `src/pzr/{core,reduction,control,benchmarks}` —
  none exist. Current: `zonotope/`, `monitoring/`, `mpc/`, `imitation/`,
  `envs/`, `systems/`, `experiments/`, `utils/`.
- `README.md` lists CLI scripts `pzr-paper-figures`, `pzr-run-experiments`,
  `pzr-run-corl` — `pyproject.toml` only declares `pzr-benchmark`. The other
  scripts have no source.
- `README.md` references scenarios `robot`, `thermostat`, `iros` — none
  exist. Current: `omni_robot`, `simple_robot`, `point_mass`, `robot_arm`.
- `AGENTS.md` was updated during this audit pass and now matches the current
  package layout, CLI, artifacts, and soundness/trigger contracts.
- `science/SCIENCE.md` mixes still-correct theory with stale code mapping
  (e.g., references to `pzr.robotics.iros`, `pzr.experiments.corl_suite`,
  `pzr.learning.distill_cli`).
- `src/predictive_zonotope_reduction.egg-info/PKG-INFO` is stale; next
  editable install regenerates.

`AGENTS.md` and `CLAUDE.md` are the current architecture docs; README and
science notes should be updated or marked as historical research narrative.

---

## L2 — `simple_robot` zero-rate calibration check is absent (LOW)

**File:** `src/pzr/systems/simple_robot.py` — `generate_simple_robot_trace`

The trace generator initializes a persistent bias once per seed
(deterministic) but does not test that the calibration-bias channel is
*actually used* (i.e., that its zonotope generator coefficient does not
collapse to zero after several reductions). No regression test catches a
case where, e.g., the calibration generator is silently filtered by a
non-protected reducer. The H2 → L2 chain matters: if M1 is real, this is one
of the symptoms we'd expect.

---

## L3 — `ScottReducer` and `MethAReducer` claim CORA parity through `cora_reference.json` but tests check only derived bounds, not exact generator matrices (LOW)

**File:** `tests/test_reduction.py` + `tests/fixtures/cora_reference.json`

Tests assert per-axis interval bounds and total widths match CORA's reference
output. They do NOT check exact `generators` matrix equality (e.g., column
order, sign). Two reducers could produce identical interval hulls from
different generators and pass; only the bound-based downstream metrics would
be correct. Probably fine for the paper (downstream metrics are what we
report), but flagging because the SCIENCE.md "CORA parity" claim is stronger
than what the tests enforce.

---

## L4 — DAgger label-diversity gate mentioned in docs is not implemented (LOW)

`science/SCIENCE.md` describes a label-diversity gate: "training data
contains at least 3 distinct reducer labels and no single label exceeds 90% of
rows." Current code (`imitation/dataset.py` + `dagger.py`) only requires
`num_classes >= 2` (`dagger_eval.py:167`). No 90% cap, no headline-exclusion
flag. Either add the gate or remove the claim from SCIENCE.md.

---

## L5 — `pzr.systems/` vs `pzr.envs/` split is unexplained (LOW)

Two monitor families with different conventions:

| | `pzr/systems/`              | `pzr/envs/`                       |
|--|------------------------------|-----------------------------------|
| Members  | `omni_robot`, `simple_robot` | `point_mass`, `robot_arm`         |
| Backend  | Pure math, hand-tuned scales | MuJoCo simulator                  |
| Noise    | Hardcoded `_noise_scale`     | `NoisySensorModel(bias, noise)`   |
| Trace    | Synthetic measurements       | `env.reset() + env.step()`        |

This appears intentional (the simulator path requires the optional `[sim]`
extra), but there is no shared base class, no comment explaining the split,
and the convention isn't documented. New contributors will assume one is the
"canonical" pattern. CLAUDE.md now explains the split.

---

## Deferred investigations (carried from prior `results/first_run/` analysis)

These were noted before the eval-suite extension and remain open.

1. **PCA blowup on omni_robot.** 50× trigger width with CI [607, 5350].
   Hypothesis: a single seed with an ill-conditioned generator matrix is
   driving the mean. Action: dump per-seed `summary.csv` and look for
   outliers; cross-reference with M4.

2. **Exact-tie pattern on point_mass and robot_arm static methods.** All six
   static reducers report identical `mean_trigger_width` on certain configs.
   Hypothesis: reductions always fire at `generator_count = budget + 1` with
   a rank-1 discarded set, so every "keep-and-box" variant produces the same
   interval hull. Action: print the discarded-set rank at each reduction
   point on a smoke run; if uniformly 1, document the structural reason.

3. **MPC vs static approx-error inversion.** MPC methods show *larger*
   approx error (≈0.35–0.37) than static methods (≈1e-16, i.e. floating
   noise) while maintaining zero FPR. Hypothesis: MPC's cost function
   trades per-axis approximation for trigger width, so it tolerates more
   absolute error on non-trigger axes. Action: split `approx_error_sum` by
   trigger vs non-trigger axes. **Also**: re-evaluate after H1 (MPC cost
   bug) and M1 (ProtectedReducer signature drift) are fixed — this finding
   may shift.

---

## Suggested ordering for fixes

1. **H1 + H2 + M1 + M2** — **DONE (2026-05-30).** Implemented via a
   `trigger_zonotope` callback parameter on `WeightedZonotopeCost` and
   `extract_features`, and a shared `reduce_with_protection` helper in
   `zonotope/protected.py` used by all four policy families. New regression
   tests added in `test_mpc.py` (+129 lines), `test_imitation.py` (+49),
   `test_full_eval.py` (+68), `test_robot_arm.py` (+12). Suite size now
   215+ tests. Re-run an overnight benchmark to refresh the corrected
   baselines.

2. **L1 doc updates** in a separate PR (low-risk, high signal-to-noise).
   `AGENTS.md` was refreshed alongside the fixes; `README.md` and
   `science/SCIENCE.md` are still stale and need either a rewrite or a
   "historical research narrative" header.

3. **M3/M4 + deferred investigations** against the corrected baselines from
   the post-fix overnight run. The MPC-vs-static approx-error inversion and
   the PCA blowup are the top candidates to re-examine first since the cost
   and protection fixes may shift their results.

4. **L3/L4/L5** as cleanup.

---

## Test-coverage gaps to add alongside fixes

- `tests/test_mpc.py`: add `test_mpc_uses_trigger_zonotope_for_cost_on_robot_arm` (will fail before H1 fix). Construct a `RobotArmMonitor`, build a synthetic state where joint-space and Cartesian widths differ obviously, assert the cost reflects Cartesian widths.
- `tests/test_imitation.py`: same shape for features.
- `tests/test_mpc.py`: add `test_mpc_search_preserves_calibration_indices` — run a 50-step trace, assert `len(state.calibration_indices)` is constant and protected columns survive.
- `tests/test_imitation.py`: add learned-policy candidate selection coverage
  showing `ProtectedReducer` receives calibration indices.
- `tests/test_full_eval.py` or `tests/test_imitation.py`: add DAgger trace
  coverage showing collected features describe the overflowing pre-reduction
  state, not the budgeted post-reduction state.
- `tests/test_reduction.py`: add rank-deficient-generator edge cases (linearly dependent columns, all-zero columns, generators with norm < 1e-12).
