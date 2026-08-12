# Paper Evaluation

## V4 finalization contract

The authoritative finalization contract is
`experiments/paper_evaluation_v4.yaml`. It preserves the v3 result directory,
uses nominal seeds 348--367 as one paired cohort for all ten methods, and writes
new scientific results only below `results/paper-evaluation-v4`. The stages are
`preflight`, `prepare`, `pilot`, `nominal`, `prediction-ablation`, `runtime`,
`report`, and `validate`. The 1,400-cell nominal matrix and 15-cell predictor
ablation remain workstation experiments; unchanged v3 fixed-trace, objective,
and H/W artifacts are imported by verified hash.

We collect the paper-facing deployment timing independently on a Raspberry Pi
5 through `experiments/paper_evaluation_v4_pi_timing_v1.yaml`. This changes the
finalization workflow, but not the v4 scientific contract: The add-on consumes
a frozen, hash-verified v4 snapshot and writes only to its own timing and
combined-report roots. Consequently, the v4 nominal quality, availability,
FPR/FNR, approximation-loss, tail, reducer-composition, guard-flow, generator,
and predictor claims remain workstation-derived. Latency, empirical rate
capacity, model footprint, process-memory, and phase-profile claims are
Pi-derived and carry separate machine and artifact provenance.

For the canonical Pi-timed artifact, the default v4 `run` command executes the
scientific stages only and stops after `prediction-ablation`:

```bash
tools/run_paper_evaluation_v4.sh run
```

The v4 `runtime` stage remains a valid workstation diagnostic. However, its
measurements are not the primary paper runtime once the Pi experiment is
adopted, and the original v4 `report` and `validate` stages still depend on that
local runtime. They execute only when requested by their exact stage names and
are never reached by `run`. Final acceptance is therefore split across three
immutable artifacts: the v4 scientific parent, the validated Pi timing result,
and `results/paper-evaluation-v4-final-report-v1`, which pairs the two by seed,
bound, and method. We do not copy Pi measurements into
`results/paper-evaluation-v4` or relabel a workstation v4 report as Pi-timed.

The primary Pi measure retains the existing benchmark semantics: reducer
selection plus the live binding-native commit over events 100--299, after
events 0--99 warm the process. Predictor computation is reported separately in
the phase profile and is not included in the primary measure. Exact-reference
metrics, prediction-diagnostic construction, and artifact I/O remain excluded.
All ten methods, seven bounds, and five seeds produce 350 method--seed--bound
cells in one sequential pass with rotated method order and one native thread.

See `RASPBERRY_PI_TIMING_V1.md` for the transfer, setup, execution, import, and
combined-report procedure.

## Preserved v3 contract

The authoritative experiment contract is
`experiments/paper_evaluation_v3.yaml`. The `pzr-paper` CLI runs independent,
resumable stages: `prepare`, `train`, `pilot`, `objective-comparison`,
`headline`, `generalization`, `ablation`, `science-report`, `science-validate`,
`timing`, `report`, and `validate`.
Old learning wrappers and cumulative-primary proposals are historical and do
not define the paper result.

Run or resume the scientific bundle with `tools/run_paper_evaluation.sh evaluate` and
inspect it with `tools/run_paper_evaluation.sh status`. The complete command
runs the release/binding preflight with tests marked `rlola_parity` excluded,
then proceeds directly to scientific stages. A projection above 72 hours requires a later
`evaluate --approve-long-run` invocation; approval cannot be supplied before the
pilot exists.

RLolaEval notebook parity remains available as the standalone
`pzr-rtlola-parity` development diagnostic. It is not invoked by the paper
wrapper, is not required by objective comparison, and does not create a paper
stage or manifest dependency. The ordinary unfiltered release test command
continues to include parity-marked tests.

For a preliminary run, `tools/run_paper_evaluation.sh explore` executes only
release preflight, verified import of seven frozen Clean148 specialists,
and the formal 112-cell nominal pilot. It explicitly excludes every paper-scale matrix,
including the unrelated historical bounded-exploration study. These stages are
source-aware and are reused by a later complete `run`.

## Method identities

- `mpc_terminal_beam` is an offline terminal-loss beam with recorded future
  inputs, horizon four, and width four.
- `mpc_terminal_beam_predictive_linear` uses the same terminal objective with
  causal linear prediction and is the deployable online MPC method.
- `mpc_terminal_full_width` is the exhaustive two-event terminal-loss teacher.
- `mpc_cumulative_beam` is an offline matched comparison only.
- `pairwise_ranking_policy` is one paper table identity backed by seven models:
  specialist `b` is trained only on budget-`b` teacher rows and evaluated only
  at budget `b`. All seven use the same fixed hyperparameters.

All selectors use binding-native transforms and rollout references. Exact
caches provide offline trigger and approximation metrics and never replace the
selection or teaching reference.

## Scope and stopping rule

Clean148 training uses nominal 500-event random-waypoint traces with seeds
0--19, 26--41, and 200--311; validation uses only seeds 20--25. The selected
optimizer seed is 42. The v3 pipeline verifies the frozen seven-model matrix,
its seed lists, subset provenance, and model hashes before copying it into the
paper namespace; it does not retrain during the overnight run. The 112-cell
pilot uses nominal seeds 90--91, all seven
budgets, and eight methods. It records CPU, ten-worker wall, disk, and
per-method projections. A projection above 72 wall hours pauses only the
1,120-cell nominal held-out stage until `--approve-long-run` is supplied; fixed
headline, objective, timing, and ablation work is outside that gate.

Nominal random-trajectory generalization uses seeds 100--119, seven budgets,
and eight methods. The controlled running example uses the four full-length
imported figure-eight variants, seven budgets, and eight headline methods (224
cells). These four fixed traces are controlled case studies rather than
multi-seed fault-population estimates. The H/W ablation uses nominal seeds
60--64, budget 150, and the 4-by-4 grid `{1,2,4,8}` (80 cells). It uses one
experiment worker so the displayed event-loop throughput is contention-free.

The matched terminal-versus-cumulative objective comparison remains 56
scientific cells (`4 fixed traces × 7 budgets × 2 methods`), but only the 28
cumulative cells are newly executed; the 28 terminal cells are verified and
reused from the headline stage. Timing is deferred from `evaluate` and remains
56 warm-ups followed
by 672 measured repetitions (`4 × 7 × 8 × 3`) and 224 summarized
condition/budget/method points.

The adopted split and deferred randomized-fault generator work are documented
in `notes/robot_arm_trace_generation_followup_20260722.md`. No claim is made
about randomized drift or geofence generalization.

## Failure and reporting contract

Cell states are `completed`, `fallback_failed`, `native_failed`, and
`infrastructure_failed`. Any interval fallback invalidates an ordinary run.
The full diagnostic time series is retained, but FPR and completed-run
throughput are unavailable; first fallback event, completed fraction,
pre-fallback mean loss, and pre-fallback throughput are reported separately.
Candidate infeasibility remains an independent count and does not invalidate a
run if an ordinary candidate succeeds.

Aggregation begins at the trace level. Main FPR is a macro mean with a
deterministic 10,000-replicate paired seed bootstrap. Pooled FPR, medians, IQRs,
fallback rates, and valid/failed counts are separate columns. If any run in a
method/condition/budget point fails, the point is unavailable; valid-only
values remain explicitly labelled diagnostics. Figures use log budget axes,
redundant color/marker/line encodings, and do not connect across unavailable
points. Loss uses a log scale only when every displayed completed value is
positive.

Raw artifacts remain ignored under `results/paper-evaluation-v3`. The
timing-free science report writes compact CSV sources, TeX tables, PDF/PNG
figures, and a hash manifest under `results/paper-evaluation-v3/science-report`.
The later final report writes to
`paper/corl2026/Zonotopes_at_CoRL/generated/paper_evaluation_v3`.
The compact sources include the pilot projection, terminal-versus-cumulative
objective comparison, specialist model hashes, fallback diagnostics, separately
labelled nominal and fixed-figure-eight reducer composition, ablation heatmaps,
and contention-free timing.

The configuration, cell, stage, run, and report contracts are schema v3.
Artifacts from v1 and the Clean20 v2 paper run remain historical and cannot be
resumed or reinterpreted as v3 results. V3 imports only the prespecified
Clean148 optimizer-seed-42 model matrix after its freeze manifest, scientific
contract, seed lists, source-dataset hashes, and model hashes pass.
