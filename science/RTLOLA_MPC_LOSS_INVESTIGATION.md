# Investigation: MPC-vs-Girard Native-Loss Discrepancy (robot arm)

Date: 2026-07-06.
Artifact under investigation:
`results/rtlola-arm-mpc-vs-girard-four-actions-b60-80-100-20260706`.
Comparison environment: `~/Faks/phd/rlola-eval` at `1fb9382` ("Added
Beamsearch"), whose `specs/robot_arm/ternary.lola` and shared traces are
hash-identical to the packaged ZONO2 spec (`ec1cb912…`) and traces.

Everything below was verified by independent replay against the pinned stack
(binding `abe3dab`, interpreter `a143dd6`, release build). Replay scripts are
reproduced in the appendix. No repository files or generated results were
modified.

## Verdict in one paragraph

The stored ZONO2 numbers are correct end to end: every `approx_loss` and
`state_width` value in the focused run reproduces from scratch to ≤1e-16 /
≤3e-14, the exact-reference cache is bitwise identical to a fresh unreduced
monitor, and the summary tables reproduce bit-exactly from the raw
timeseries. The huge Girard/MPC loss ratios (176–2845×) are the *square* of a
modest excess-width ratio (~37× on figure8/b80) on a metric that measures
only per-stream interval excess, which is ~1% of total state width — so a
1.2% width improvement and a 1379× loss ratio are the same fact expressed on
two different scales. The orders-of-magnitude disagreement with the
rlola-eval `beam_search.ipynb` numbers is a defect in *that* notebook's
reference cache: it was generated with `current_zonotope()`, whose Python
default **excludes** the five constant calibration-slack columns, while
`approx_loss` always evaluates the monitor state **with** them. Its reported
losses are therefore dominated by the calibration slack itself, not by
reduction error.

## 1. Precise definitions of every loss in play

**Binding-native loss** (`rlolapythonbinding/src/lib.rs:158-175`, unchanged
in every revision of the binding):

```
r_i(Z)   = Σ_j |G_ij|                    (row-wise interval radius)
upper_i  = c_i + r_i ;  lower_i = c_i − r_i
L(Z1,Z2) = [ Σ_i (u2_i−u1_i)² + Σ_i (l2_i−l1_i)² ] / (2d)
```

evaluated on the **total** state zonotope (`include_constant_slack=true` for
z1, hardcoded). With equal centers this reduces exactly to
`mean_i (r2_i − r1_i)²` — an **interval-radius MSE**. It is blind to
generator directions and correlations. The `debug_assert_eq!` checks on
dimension and center equality (lib.rs:163-164) are compiled out in the
required release build.

**Offline `approx_loss` column** (`benchmark.py:603`,
`engine.py:196-262`): the committed live state after event *i* (reduction is
applied *before* event processing; `evaluator.rs::eval_event`) is compared
against the global exact reference at the same step. The reference is a
separate monitor run with `ZonotopeConfig.none()` from trace start
(`benchmark.py:766-795`); `none` is a literal no-op (`noop.rs`), so "exact"
means *unreduced RTLola affine monitor*, not a physical reachable set. Each
cache row stores the total-state center and row radii; the engine
reconstructs a diagonal interval matrix and calls the native loss. Because
the metric reads only row radii, this reconstruction is lossless (verified:
manual full-matrix recomputation matches stored values to 1e-16).

**Summary statistics** (`benchmark.py:1023-1068`): `mean/final/max/sum` are
nan-aware mean/last/max/sum of the per-step column. All were reproduced
bit-exactly from the raw timeseries CSVs. `sum_approx_loss` is unweighted
and only comparable within a trace (documented in AGENTS.md).

**`state_width`** (`metrics.py:53-61,108`): `2·Σ_rows Σ_gens |G|` of the
**dynamic-only** matrix (constant slack excluded). This is *not* the loss's
operand. The constant-slack contribution to total width is method-invariant
and grows from 0.01 to 15.69 over figure8 (mean 7.86), so mean total width is
~33.3 while mean dynamic width is ~25.8.

**MPC search cost** (`search.py:776-802`): terminal binding loss of a
candidate rollout against a *local* reference — a `none` continuation started
from the **current committed state** (it inherits all prior approximation
damage). Horizon 4 = current event + 4 recorded future events (oracle
lookahead, `benchmark.py:427`). Intermediate losses only prune the beam.

**rlola-eval `beam_search.ipynb` loss**: same native formula, same spec,
cumulative over the horizon and evaluated against a *global* stored
reference — but the reference npz was generated with
`monitor.current_zonotope()` (slack **excluded** by the PyO3 default,
`python.rs:406`), while `approx_loss` compares against the state **with**
slack. Its "Total loss" is the per-step sum, i.e. our `sum_approx_loss`
semantics *plus the artifact*.

## 2. Reproducible minimal example (figure8, budget 80, step 10)

Replaying the committed action prefixes from the stored timeseries
(`scratchpad/minimal_example.py`, appendix):

```
girard   actions: none×3, girard×8         mpc: none×3, girard×2, combastel, girard×5
d=7; candidate gens=120, exact gens=390
centers == cache centers exactly: True;  fresh exact radii == cache radii exactly: True
exact radii : [0.041408 0.030088 0.086909 0.001129 0.003737 0.000916 0.002915]
girard per-row excess: [~1e-17 ~1e-17 ~1e-17 4.716e-06 2.329e-06 8.778e-06 3.014e-06]
girard manual mean-sq excess = 1.6255597702e-11 = stored approx_loss (exact)
mpc    manual mean-sq excess = 1.6049937729e-11 = stored approx_loss (exact)
state_width reproduces to 10 decimals for both.
```

## 3. Hypotheses: evidence for and against

**Arithmetic/numerical error — ruled out.** Full-trace replays of both
methods (figure8/b80): max |manual loss − stored| = 1.0e-16 (girard),
1.7e-17 (MPC); max width error 3.6e-14. Zero-loss accounting closes exactly:
48 automatic `none` steps + 2 zero reduction rows = 50 zero rows per method.

**Reference/candidate self-comparison or state aliasing — ruled out.** The
offline reference is a JSON-loaded numpy array, not an evaluator state; there
is no object to alias. On the search path, `apply_state` and
`accept_event_from_state` deep-clone `EvaluatorState`
(`python.rs:369,392`; commit `72622a3` "Clone state in python when
applied"; `EvaluatorState` is `derive(Clone)` over owned stores). Losses are
nonzero on 39,049 of 39,099 MPC rows, so no systematic self-comparison is
occurring. Planner restoration in `finally` was re-verified.

**Event/state misalignment — ruled out.** Reference rows and committed states
carry step tags that must match (`engine.py:207`), both counted post-event.
The engine additionally requires *exact* center equality per step. Measured
over figure8: min over steps of max |center(i) − center(i+1)| = 0.157, so a
one-step shift can never satisfy the bitwise center check. Full-trace replay
confirmed center equality at every one of 2×2340 steps.

**Cache-reconstruction error — ruled out.** Fresh unreduced replay vs cache:
max center error 0.0, max radius error 0.0 (bitwise) over all 2340 rows.
JSON float round-trip is exact (repr shortest-roundtrip).

**Row-order/dimension mismatch — ruled out.** Rows are emitted in
deterministic IR order (`zonotope.rs::row_map`); rows whose affine form has
all-zero coefficients are dropped, which is why step 0 has d=5 and later
steps d=7 (expected warm-up, one row per retained affine memory position).
Any mismatch would trip the dimension or center check; none did.

**Mean/sum/final reporting error — ruled out.** All summary numbers reproduce
bit-exactly from the raw timeseries; `primary_metrics.csv` and
`fpr_native_loss_comparison.csv` agree with each other and with the raw data.

**Semantic mismatch with the other environment — CONFIRMED (root cause of
the cross-environment gap).** See §4/§5.

**Metric blindness to correlations — CONFIRMED and quantified.** Support
functions in 128 fixed random unit directions at 48 sampled steps
(candidate-vs-exact, slack columns cancel):

- mean directed support excess: girard 0.875, MPC 0.863 — ratio **1.01**;
- axis-aligned (per-stream) excess ratio at the same steps: ~37×;
- soundness holds in every sampled direction for both (min excess ≥ 0).

So the MPC advantage is essentially confined to axis-aligned directions —
i.e. to per-stream interval bounds. That is precisely what the specification's
triggers read (`pAbove(stream, const)`), so the metric is aligned with
trigger fidelity, but the 37× does **not** transfer to a general set
distance (e.g. directed Hausdorff), where both reducers are ~equally lossy.
Scott exploits this: it absorbs discarded generators into a basis of dominant
directions (`scott.rs`, `T ← T·(I+diag|r|)`), keeping axis radii nearly exact
while the set in oblique directions stays inflated. Girard instead boxes the
smallest generators onto the axes (`girard.rs`), which the low-gain
alpha-beta observer in the spec was *deliberately designed* to punish (see
the spec comment: "LOW gains … this is what makes boxing compound").

**Oracle lookahead — present, but symmetric.** Both our MPC
(`benchmark.py:427`) and the rlola-eval beam search (`events[step+d]`) use
recorded future inputs. It inflates both relative to a causal controller and
is not a differentiator, but it must be stated in any writeup.

**Width/loss relationship — verified quantitatively.** Full-trace figure8/b80:

```
mean excess TOTAL width vs exact:  girard 0.3077   mpc 0.008355   ratio 36.83
sqrt(mean-loss ratio) = sqrt(1378.9) = 37.13   (consistent)
girard row-wise excess: ~0 on rows 0-2, concentrated on 4 observer rows
   [0, 0, 0, 0.0537, 0.1032, 0.0519, 0.0989]
```

Mean total width is ~33.3 (25.8 dynamic + 7.86 slack), so Girard's excess is
0.9% of total width and MPC removes ~97% of it. A ~1.2% dynamic-width
improvement and a ~1379× squared-loss ratio are therefore the *same*
finding. Same structure holds at b60/b100 and on random (excess ratio ~82×,
loss ratio 6742×) and square_drift (~26×, 669×).

**Independent behavioral corroboration.** On the `random` trace the geofence
triggers false-fire on 95.4% of exact-negative steps under static Girard vs
1.1% under MPC (b80; per-trigger and `__any__` confusion rows are separately
tabulated, absent sparse keys normalized to False, FNR = 0 for both as
soundness requires). The loss improvement is behaviorally real at the
trigger level.

## 4. Comparison with rlola-eval `beam_search.ipynb`

| Field | ZONO2 focused run | rlola-eval beam_search (cell 1 / cell 2) |
|---|---|---|
| Spec | `robot_arm.lola` = `ternary.lola`, sha `ec1cb912…` | identical file |
| Trace | six packaged traces | `random_drift` (1433 events; not in ZONO2 set) |
| Binding | pinned `abe3dab`, release, enforced at import | unpinned venv; `approx_loss` formula identical in all revisions |
| Reducer candidates | girard, scott, pca, combastel | identical |
| Budget | 60/80/100, transform bound | 500 fixed / per-step sweep 10–100 |
| Action timing | transform before event | identical (same binding API) |
| Lookahead | recorded future, 4 events | recorded future, 4 / 9 events |
| Objective | terminal loss vs local `none` continuation | cumulative loss vs global stored reference |
| Reference | global exact, **with** constant slack, bitwise-validated | global exact, **without** constant slack (defect) |
| `none`/fallback | auto-`none` under budget; `interval` fallback | `none` only in cell 2 when under bound; no fallback |
| Headline number | mean per-step loss | "Total loss" = per-step sum |
| Scott share | 91.9% of reductions | 85.8% of steps (corroborates our composition) |

## 5. The rlola-eval reference defect, proven four ways

1. **API asymmetry.** `current_zonotope()` defaults to
   `include_constant_slack=false` in every binding revision
   (`python.rs:406`), while `approx_loss` hardcodes `true` for its own state
   (`lib.rs:159`). The notebook's reference-generation cell calls
   `monitor.current_zonotope()` bare.
2. **Column arithmetic from their own output.** The generation cell printed
   `last zonotope shape: (7, 50156)` = 50155 generators = 35·1433 — exactly
   the no-slack count for 1433 events (35 fresh slack vars per event); with
   slack it would be 50160.
3. **Parabola signature.** Their per-step losses fit
   `0.74·(step/1433)²` to ~2% at every logged step regardless of the chosen
   method (e.g. step 600: 0.124 vs predicted 0.130; step 900: 0.278 vs
   0.292), and their total 343.5 matches the parabola integral (≈353). A
   squared linearly-growing offset is the signature of the accumulating
   calibration-slack radius (measured: slack width grows 0.01 → 15.7 over a
   trace); genuine reduction error fluctuates and is orders smaller
   (≤ ~5e-3 per step even for static Girard).
4. **Perfect-candidate floor (decisive).** Replaying their exact trace and
   spec with a pure `none` monitor — zero reduction error by construction —
   and scoring it exactly as the notebook does against their actual LFS npz:

   ```
   final loss = 7.408824e-01   (their beam run: 7.408826e-01)
   total loss = 3.435484e+02   (their beam run: 3.435486e+02)
   ```

   The zero-error candidate reproduces their reported numbers to seven
   significant figures; the difference (2e-4 of 343.5) is the entire genuine
   reduction error of their run. Their stored matrices are missing exactly 5
   columns versus the slack-included state at every sampled step, and match
   our fresh no-slack matrices up to generator column order and last-bit
   summation noise (row radii agree to 8e-17).

Consequence: 99.99994% of the notebook's loss magnitude is artifact. After fixing
the reference (`current_zonotope(True)`), its numbers should land in the
same regime as ZONO2's (per-step means of 1e-7…1e-3 depending on method),
with residual differences from the cumulative-vs-terminal objective, budget,
and trace.

## 6. Concrete defects found (with locations)

ZONO2 pipeline: **none**. All stored values reproduce; alignment, soundness,
and aggregation verified.

Elsewhere:

- `rlola-eval` `robot_arm.ipynb` (reference-generation cell) and everything
  downstream in `beam_search.ipynb`: reference built with slack-less
  `current_zonotope()` but scored against slack-included states. Fix:
  `monitor.current_zonotope(True)` and regenerate `result_cache/*.npz`.
- `rlolapythonbinding/src/python.rs:406,421`: asymmetric defaults
  (`include_constant_slack=false`) vs `lib.rs:154,159` (`true`). This is the
  footgun that produced the above. Recommend defaulting to `true` or making
  the argument mandatory.
- `rlolapythonbinding/src/lib.rs:163-164`: dimension/center checks are
  `debug_assert_eq!`, compiled out in the required release build; a
  center-mismatched comparison silently returns a plausible number.
  Recommend runtime errors. (ZONO2's engine independently enforces both on
  the offline path, `engine.py:229-238`.)

## 7. Recommendations (no changes made)

1. Ask the collaborator to regenerate the npz caches with
   `current_zonotope(True)`; expect the beam-search conclusions about
   *method ranking* (Scott-dominant) to survive, and the loss magnitudes to
   drop by ~5–6 orders.
2. In the paper, frame the MPC gain as **per-stream interval fidelity /
   trigger fidelity** (37× excess-radius reduction, 95%→1% FPR), not as a
   general set-distance improvement — the support-function test shows the
   gain does not transfer to random directions (ratio 1.01).
3. Report `sqrt(mean loss)` or mean excess width alongside the squared
   metric so headline ratios are not read as set-volume factors.
4. Document the oracle lookahead; consider a causal variant (e.g.
   constant-hold or learned event prediction) as an ablation.
5. Optionally add a direction-sampled support-excess diagnostic as a
   secondary offline metric (cheap at sampled steps; needs retained exact
   matrices only at those steps).

## Appendix: verification runs

Scripts (session scratchpad, reproducible from this description):
`width_loss_check.py` (stored-artifact consistency),
`replay_verify.py` (full-trace replay + support functions),
`minimal_example.py` (§2), `rlola_eval_floor_npz.py` (§5.4).

Key replay output (figure8/b80):

```
cache-vs-fresh-exact: max|center err|=0.0  max|radius err|=0.0
min over steps of max|center[i]-center[i+1]| = 1.573e-01
[girard] max|manual−stored loss|=9.996e-17; min per-row excess=-1.208e-13
         mean excess total width=0.3077; const-slack width 0.0100→15.6854
[mpc]    max|manual−stored loss|=1.658e-17; min per-row excess=-8.882e-16
         mean excess total width=0.008355; const-slack width identical
excess-width ratio 36.83; support-function ratio in random directions 1.01
```

Floor check against the collaborator's actual npz (random_drift, 1433 events):

```
perfect none candidate, notebook scoring: final=7.408824e-01 total=3.435484e+02
their beam_search.ipynb (bound 500):      final=7.408826e-01 total=3.435486e+02
missing columns vs slack-included state: exactly 5 at every sampled step
```
