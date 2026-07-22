# Paper Evaluation

The authoritative experiment contract is
`experiments/paper_evaluation_v1.yaml`. The `pzr-paper` CLI runs independent,
resumable stages: `prepare`, `train`, `pilot`, `objective-comparison`,
`headline`, `generalization`, `ablation`, `timing`, `report`, and `validate`.
Old learning wrappers and cumulative-primary proposals are historical and do
not define the paper result.

Run or resume the complete bundle with `tools/run_paper_evaluation.sh run` and
inspect it with `tools/run_paper_evaluation.sh status`. The complete command
runs release tests and the 576-cell pinned notebook parity before scientific
stages. A projection above 72 hours requires a later
`run --approve-long-run` invocation; approval cannot be supplied before the
pilot exists.

For a preliminary run, `tools/run_paper_evaluation.sh explore` executes only
release preflight, nominal teacher preparation, both policy trainings, and the
formal 54-cell nominal pilot. It explicitly excludes parity and every paper-scale matrix,
including the unrelated historical bounded-exploration study. These stages are
source-aware and are reused by a later complete `run`.

## Method identities

- `mpc_terminal_beam` is an offline terminal-loss beam with recorded future
  inputs, horizon four, and width four.
- `mpc_terminal_beam_predictive_linear` uses the same terminal objective with
  causal linear prediction and is the deployable online MPC method.
- `mpc_terminal_full_width` is the exhaustive two-event terminal-loss teacher.
- `mpc_cumulative_beam` is an offline matched comparison only.
- `pairwise_ranking_policy` is trained across all seven recorded budgets.
- `pairwise_ranking_policy_budget80` is trained from the budget-80 subset of
  the same teacher dataset and is reported only in the extrapolation table.

All selectors use binding-native transforms and rollout references. Exact
caches provide offline trigger and approximation metrics and never replace the
selection or teaching reference.

## Scope and stopping rule

Training and validation use nominal 500-event random-waypoint traces with seeds
0--19 and 20--25. The 54-cell pilot uses nominal seeds 90--91, budgets
40/150/500, and nine policies. It records CPU, four-worker wall, disk, and
per-method projections. A projection above 72 wall hours pauses only the
1,260-cell nominal held-out stage until `--approve-long-run` is supplied; fixed
headline, objective, timing, and ablation work is outside that gate.

Nominal random-trajectory generalization uses seeds 100--119, seven budgets,
and nine policies. The controlled running example uses the four full-length
imported figure-eight variants, seven budgets, and eight headline methods (224
cells). These four fixed traces are controlled case studies rather than
multi-seed fault-population estimates. The H/W ablation uses nominal seeds
60--64, budget 150, and the 4-by-4 grid `{1,2,4,8}` (80 cells). It uses one
experiment worker so the displayed event-loop throughput is contention-free.

The matched terminal-versus-cumulative objective comparison remains 56 cells
(`4 fixed traces × 7 budgets × 2 methods`). Timing remains 56 warm-ups followed
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

Raw artifacts remain ignored under `results/paper-evaluation-v1`. The
report stage writes compact CSV sources, TeX tables, PDF/PNG figures, and a hash
manifest to
`paper/corl2026/Zonotopes_at_CoRL/generated/paper_evaluation_v1`.
The compact sources include the pilot projection, terminal-versus-cumulative
objective comparison, budget-80 extrapolation, fallback diagnostics, separately
labelled nominal and fixed-figure-eight reducer composition, ablation heatmaps,
and contention-free timing.

The filename and output namespace remain `paper_evaluation_v1`, but the
configuration, cell, stage, run, and report contracts are schema v2. Artifacts
from the superseded four-generated-condition contract are stale and must be
archived rather than resumed or reinterpreted.
