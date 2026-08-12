# Final Runs for the ICRA 2027 Evaluation

Status as of 2026-08-11. The evaluation section of
`paper/Zonotopes_at_ICRA2027/root.tex` is structurally complete and every number
in it is derived from frozen artifacts by
`paper/Zonotopes_at_ICRA2027/generate_experimental_artifacts.py`. This document
lists what still has to run before those numbers are final, and what does not.

**Prose status (2026-08-11).** The analysis text in Sec. VI has been cut back to
placeholder density: each result is stated, the digits live in the floats, and
the preregistration/gate discussion has been removed from the paper entirely. A
`\todo[inline]` at the head of Sec. VI records this and points here. The full
analysis gets written after the runs below land — see §7 for what that involves.

Cost figures are measured, not estimated: they come from `cell_elapsed_ms` in the
existing summaries. "core-h" is summed single-core cell time; divide by the
worker count for wall clock.

**V4 execution update (2026-08-11).** The standalone v4 finalization supersedes
the separate Runs A--C below: It evaluates all ten methods on the fresh paired
cohort 348--367, adds the H=4/W=4 predictor ablation, and imports unchanged v3
fixed/objective/H-W artifacts by hash. We will collect the paper-facing runtime
independently on a Raspberry Pi 5. Thus, the workstation v4 run should stop
after `prediction-ablation`; the append-only Pi add-on then bundles the frozen
parent, validates 350 timing cells, and writes a separate combined report. Runs
A--C remain useful as the historical rationale for v4, but they are no longer
independent execution items.

The default `tools/run_paper_evaluation_v4.sh run` command now enforces this
boundary: It executes only the five scientific stages and cannot start
workstation timing. The old timing-dependent `runtime`, `report`, and `validate`
stages require explicit invocation and remain diagnostic-only.

The manuscript must keep the provenance split explicit: Scientific quality and
soundness claims come from `results/paper-evaluation-v4`, whereas latency,
memory, and empirical on-time capacity come from
`results/paper-evaluation-v4-pi-timing-v1`. The final source of cross-metric
claims is `results/paper-evaluation-v4-final-report-v1`; no Pi value is copied
into or used to reinterpret the v4 parent.

| Per-cell cost (median, measured) | Nominal, 500 ev | Figure-8, 2,340 ev |
| --- | --- | --- |
| Fixed reducer | 5–7 s | — |
| Learned policy (G15, Vote3, Vote3-Guarded) | 9 s | 37 s |
| MPC-F | 61 s | — |
| MPC-L / MPC-B | 231–241 s | ~1,172 s |

---

## 0. Answerable now, with no new run

Two claims the paper makes are already supported by data on disk and are missing
only because nothing reads those columns yet. Do these before scheduling
anything.

### 0.1 Bounded memory is never shown, and the data proves it

The abstract and introduction both claim bounded memory. The evaluation never
demonstrates it. `generalization/summary.csv` and `headline/summary.csv` carry
`max_generator_count`, and the result is clean:

- Peak generator count is **`b + 35` or `b + 36` on every one of the 1,296
  completed cells** — 1,100 nominal plus 196 figure-eight.
- The same constant holds on 500-event and 2,340-event traces, so the bound is
  independent of trace length.
- The same constant holds for all eight methods, so it is a property of the
  transform bound and not of the policy.

That is the bounded-memory claim as a measured invariant, and it costs one
sentence plus possibly one column. **This is the largest gap in the section.**

### 0.2 Soundness is never checked empirically

The setup paragraph argues no method can produce a false negative *by
construction*. `false_negative_count` is **0 across all 1,296 completed cells**.
Stating the construction argument and then confirming it held over roughly
1.0 M monitored events is strictly stronger than the argument alone, and a
reviewer will look for exactly this. Also free.

---

## P0 — required for the numbers the paper currently prints

### Run A. Canonical cohort for Vote3 and Vote3-Guarded

**Why.** Vote3 and Vote3-Guarded exist only on the confirmation cohort (seeds
328–347). Table I and Fig. 1 therefore cannot show the policy the paper promotes
as its policy of record. Both artifacts already reserve the slots: Table I prints
red `\placeholder` rows and Fig. 1(a) prints a standing "pending canonical run"
note. **No generator change is needed** — the rows and trails fill themselves in
when the cells appear.

**Scope.** 2 methods × 20 seeds (100–119) × 7 bounds = **280 cells**.

**Cost.** 280 × ~9.2 s ≈ **0.7 core-h** (~5 min at 10 workers). This is the
cheapest item on the list and it closes the most visible hole.

**Legitimacy.** Verified: `train_seeds` (n=148, range 0–311) has **zero overlap**
with seeds 100–119, so the cohort is genuinely held out for the voting policies.
Note the distinction — these seeds are held out for *training*, but they were
already used to report every other method, so Run A supports **reporting** in
Table I, not a fresh **selection** decision. That is what Run B is for.

**Done when.** `pending_canonical_methods` in
`generated/generation_manifest.json` is `[]`, the red note disappears from
Fig. 1(a), and the two `\placeholder` rows carry numbers. Re-tune the two
reserved label offsets in `_plot_tradeoff` against the real positions — they are
marked provisional in the code and were only validated against synthetic cells.

### Run B. Fresh confirmation cohort for Vote3-Guarded

**Why.** The frozen selection rule picked G15/Clean148. Promoting Vote3-Guarded
to policy of record is a decision made *after* seeing the confirmation cells,
which turns that cohort from a confirmation set into a selection set — the
policy of record has no cohort on which it was not also chosen.

**This got more important, not less, on 2026-08-11.** The paper used to carry a
paragraph disclosing exactly this and stating that a clean promotion needed a
fresh cohort. That paragraph is gone with the rest of the preregistration
language, so Sec. VI now asserts Vote3-Guarded as policy of record with no
in-text caveat. Either this run happens and the assertion is earned, or the
caveat comes back. Do not ship the assertion with neither.

**Scope.** 4 methods × 20 new seeds × 7 bounds = 560 nominal cells. Seeds must
avoid `train_seeds`, 100–119, 312–327 (reserved exploration), and 328–347. Seeds
**348–367** are clean.

**Cost.** MPC-L dominates: 140 × 241 s ≈ 9.4 core-h, plus 420 learned cells ×
9.2 s ≈ 1.1 core-h → **≈ 10.5 core-h** (~1 h at 10 workers). Adding the 112
figure-eight cells roughly doubles it to ~23 core-h (~2.3 h); those are
descriptive case studies, so they are optional here.

**How.** `tools/prp_vote3_guarded_paper_sweep.py` already parameterizes
`confirmation_seeds` and asserts selection/confirmation seeds do not overlap, so
this is a config change plus a rerun, not new code.

### Run C. v3 prediction ablation with input-predictor variation

**Why.** `tab:prediction-ablation` is now prose, but its numbers still trace to
the **superseded v2 run** (`results/rtlola-learning-paper-v2-full-20260719/`) at
horizon 3 / width 4, while every other number in the section comes from v3 at
horizon 4 / width 4. A `\todo` in `root.tex` marks this.

**Blocker.** Confirmed: the v3 `ablation` stage writes `input_predictor` and
`prediction_step_seconds` as **all-NaN** — the stage varies horizon and beam
width only. A v3 replacement does not exist and cannot be recovered by
re-analysis; the stage has to be run with predictor variation.

**Cost.** At the v3 operating point (H=4, W=4, ~81 s/cell), 4 predictor variants
× 5 seeds (60–64) × 1 bound ≈ 20 cells ≈ **0.5 core-h**. Cheap once the stage
emits the column.

**Fallback if it does not run.** Keep the prose and state the v2 provenance
explicitly in the text rather than only in a `\todo`. Do not leave v2 numbers
sitting unmarked among v3 numbers.

---

## P1 — closes known weaknesses, not required for correctness

### Run D. Objective comparison on more than one seed

The terminal-vs-cumulative comparison behind "tighter on 16 of 28 with a median
loss ratio of 0.867" is **56 cells at seed 0 only** (2 objectives × 7 bounds ×
4 figure-eight conditions). The paper reads this as no meaningful separation,
which is the conservative direction, so a single seed does not invalidate the
conclusion — but it is thin for a stated design justification.

**Cost.** 10.2 core-h per additional seed (215 s/cell × 56). Five seeds ≈ 51
core-h (~5 h at 10 workers). Given it supports a negative result, one or two
extra seeds is probably the right amount of evidence to buy.

### Run E. Promote the ensemble-size evidence

A reviewer will ask why three specialists and not five. There is already an
answer in `results/prp-tail-vote-guard-exploratory-v1/` — 392 cells over
exploration seeds 320–327, covering `dagger05_ensemble3`, `vote3`,
`vote3_guarded`, `vote5`, `vote5_guarded`, `g15_clean148`, and MPC-L. Measured
loss ratio to MPC-L:

| Policy | p50 | p95 | severe (>10³) |
| --- | --- | --- | --- |
| G15/Clean148 | 1.15 | 348,615 | 3/56 |
| Vote3 | 1.28 | 6.16 | 2/56 |
| **Vote3-Guarded** | **1.11** | **2.10** | **0/56** |
| Vote5 | 1.16 | 2,533 | 3/56 |
| Vote5-Guarded | 1.19 | 5.79 | 2/56 |
| Ensemble3 (mean, no vote) | 1.30 | 6.23 | 2/56 |

Two things worth saying: **five specialists are worse than three**, and
Vote3-Guarded is best on all three axes on a **third independent cohort**. Seeds
320–327 are declared `reserved_exploration_seeds` in the v3 config and have zero
overlap with training, so using them for exactly this is methodologically clean —
provided the text says these seeds were explored rather than held out. One
sentence naming the cohort does it; the paper no longer has the
exploratory/confirmatory vocabulary to lean on.

**Cost.** Zero. This is a reporting decision, not a run.

### Run F. Guard margin threshold

The symbolic override fires at winner margin ≤ 1, a fixed hyperparameter that is
never varied. With three specialists the only alternative is margin ≤ 0 (never
fire) or ≤ 2 (always fire), so the sweep is nearly degenerate and the flow
diagram in Fig. 2 already shows what happens at each margin. **Recommend
skipping** unless a reviewer raises it.

---

## Not planned, and why

- **Second system or second specification.** Everything runs on the 5-DOF MuJoCo
  arm with one packaged RTLola monitor. This is the section's real external
  validity limit and it belongs in Limitations, not in a run — a second system is
  a paper of its own, not a final-numbers item.
- **Scott at higher bounds to make it complete.** Already answered by the data:
  Scott dies at 998–1,088 of 2,340 events at *every* bound above 40, and a 12.5×
  increase in *b* buys 90 events. Raising the bound cannot rescue it, and the
  failure is a result rather than a gap.
- **Re-running fixed reducers or MPC variants.** Nothing about them has changed;
  29.5 core-h of canonical generalization data stands.

---

## 7. After the runs: rewriting Sec. VI

The prose was thinned on 2026-08-11 so that no sentence has to be re-verified
twice. What that leaves to do once the cells land:

- **Re-check every number still quoted in the text.** The survivors are the ones
  no rerun should move — availability counts (120/140, 0/28), the 94 % agreement
  rate, 69,120 decisions, 85–89 % Scott selection, the 91-of-140 and 115-of-140
  win counts, `2.62 %` paired throughput retention, `2.58 %` FPR ceiling, the
  `1.46 ms` amortized guard cost, 16-of-28 on the objective comparison. Run B
  replaces the cohort behind the guard and win-count figures, so those move.
- **Put the digits back where they earn their place.** Cut material worth
  restoring selectively: Scott's 998–1,088 survival window and its `e17`–`e19`
  loss inflation, the per-reducer pooled FPR spread, the quantile/severe columns
  read out in §VI-D, and the guard's 728/416/311 net flows.
- **Add §0.1 and §0.2.** Bounded memory and the zero-false-negative result are
  not in the section at all yet.
- **Decide the promotion sentence.** See Run B: earned by a fresh cohort, or
  hedged in text. Not neither.
- **Do not restore the gate scorecard.** The `\pass`/`\fail` macros were deleted
  from the preamble along with the prose; `\checkmark` stays for the headline
  table's real-time column.

## Suggested order

1. Run `tools/run_paper_evaluation_v4.sh run` on the workstation. It executes
   `preflight`, `prepare`, `pilot`, `nominal`, and `prediction-ablation`, then
   stops before workstation timing.
2. Build both Pi bundles. Execute the one-seed/one-bound smoke first and inspect
   all ten method types before transferring authority to the full bundle.
3. Run the 350-cell Pi matrix sequentially, then run the ten-method semantic
   phase profile and pack the result archive.
4. Import the archive on the workstation and generate
   `results/paper-evaluation-v4-final-report-v1`. Treat successful exact pairing
   and hash validation as the final cross-platform gate.
5. Regenerate the ICRA artifacts and rewrite Sec. VI from the v4 scientific
   claims plus the Pi/combined timing claims. Recheck every manually written
   number against the corresponding manifest.
6. Run the larger objective-comparison extension only if time remains; it is
   additional evidence for a conservative design choice and is not part of the
   v4/Pi acceptance contract.

Runs A--C and the older ordering remain above as design history. They must not
be executed into the v3 namespace or reported alongside v4 as if they were a
second confirmation cohort.

## Invariants for every run

- Do not hand-edit anything under `results/`; the generator reads it and
  `tests/test_paper_evaluation.py` checks it is untouched.
- Require exactly 1,400 v4 nominal cells, 15 predictor cells, 350 Pi timing
  cells, and an exact seed--bound--method join. Expected reducer infeasibility
  remains explicitly unavailable; native or infrastructure failures abort.
- Freeze the v4 scientific parent before building either Pi bundle. Any later
  change to a pinned config, runner, source, model, trace, or helper requires an
  explicit reviewed version/pin change, not reuse of the old archive.
- Update the manuscript generator to consume the v4 and combined manifests,
  then regenerate with
  `external/miniconda3/envs/pzr-robot-arm/bin/python paper/Zonotopes_at_ICRA2027/generate_experimental_artifacts.py`
  and rebuild with `latexmk -pdf root.tex`. Every `_require` provenance assertion
  must pass.
- Re-tune the Vote3 and Vote3-Guarded label offsets against the actual v4
  positions; the existing offsets were validated against synthetic cells only.
- Numbers typed into Sec. VI prose are not regenerated and not checked by any
  assertion. Every one of them has to be read back against the regenerated
  floats by hand — that is the whole reason the prose was thinned.
