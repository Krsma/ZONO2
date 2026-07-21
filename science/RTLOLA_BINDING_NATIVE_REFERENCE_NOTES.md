# RTLola Binding-Native Reference Notes

Updated: 2026-07-20

## Context

PZR now uses the zero-row-aware RTLola stack:

- binding revision `01c92a2bfac58755e3b832bb0094816f3f36e1d1`
- interpreter revision `2724b05ae6c62ed0df14f1401ed8db89472725a6`

The current interpreter retains the `b4cfbf4` logical-zero-row fix when exporting
evaluator states. Native transformations still construct compact nonzero-row
zonotopes for reduction. That split is useful for PZR: exact-reference caches
can compare full logical rows, while reducer budget checks must continue to use
the compact reducer dimension.

The problem is not only that individual reducers such as PCA or Scott can panic.
PZR already catches many native reducer failures during candidate evaluation and
downgrades them to infeasible candidates or fallback use. The deeper issue is
that PZR still treats extracted zonotope matrices as a stable semantic interface.

## Fragile Areas

The following paths depend on stable extracted matrix shape or row identity:

- `state_zonotope(...)` extraction, which can panic on some newer-stack PCA or
  Scott states.
- `engine.metrics(...)`, which derives `state_width`, active generator counts,
  zero generator counts, and dynamic generator counts from extracted matrices.
- exact-reference caching, which stores Python-level center/radius arrays from
  unreduced exact states.
- exact-reference measurement, which later compares those cached rows against
  freshly extracted candidate matrices.
- rowwise semantic tests, which assume that exact and reduced matrices have
  comparable row positions.

Older interpreter behavior could compact or drop exposed all-zero dynamic rows.
Once that happened, an exact cached reference and a reduced candidate state could
expose different numbers of rows, or row `i` could stop referring to the same
logical state component in both matrices.

One observed newer-stack failure had this form:

```text
RTLola exact-reference and candidate dimensions differ
```

This is a correct refusal by PZR: comparing arrays with different exposed
dimensions would be semantically unsafe.

## Why Reference Caching Is A Weak Point

Reference caching is valuable because PZR can compute exact references once per
trace and reuse them across methods and budgets. This supports consistent FPR,
FNR, and approximation-loss reporting without rerunning exact rollouts for every
method.

The weak point is the current cache representation. PZR caches exact state
geometry as naked Python center/radius arrays. Later, it reconstructs an interval
matrix and asks the binding to compare it with a candidate state extracted from
the current monitor.

That creates a cross-run contract:

```text
cached exact rows must align with later candidate rows
```

The `b4cfbf4` interpreter change repairs this specific exported-row contract by
including zero/missing logical rows in state export.

This likely differentiates PZR from RLolaEval. RLolaEval appears not to depend
on reusable Python-side exact state-geometry caches in the same way. It may
evaluate comparisons inside one native path, focus more on verdict/report data,
or avoid treating `state_zonotope(...)` layout as a stable public semantics.

## Candidate Later Fix

The long-term direction should be to keep reference caching, but move the
semantic reference representation into the binding.

Python should stop caching and aligning raw matrix rows. Instead, the binding
should expose layout-aware APIs such as:

- serialized exact-reference snapshots created and owned by the binding;
- native `approx_loss_reference(reference_payload, candidate_state)`;
- native compaction-safe `approx_loss_state(reference_state, candidate_state)`;
- native `state_stats(state)` for dynamic counts, active counts, zero counts,
  total counts, and state width;
- matrix extraction only as a debug or inspection API, not as a benchmark
  correctness dependency.

The key requirement is not merely moving the current matrix subtraction from
Python to Rust. The binding-native comparison must be logical-state aware or
otherwise canonical with respect to interpreter compaction. If Rust still
compares compacted extracted matrices row-by-row, the same problem remains.

## Expected Benefit

A binding-native reference/statistics design would avoid the exact-reference
dimension mismatch class of failures and reduce Python's dependence on unstable
matrix layout. It would also make the benchmark pipeline better aligned with the
project's trusted boundary: Python selectors may inspect and choose actions, but
binding-native RTLola APIs should own monitor-state semantics.

This would not automatically fix every native panic. If a reducer creates an
invalid native state, the binding must still handle or report that robustly. But
it would remove PZR's current row-alignment assumptions from core metrics,
search scoring, and exact-reference reporting.

## Current Decision

Retain the logical-zero-row contract with a narrow PZR compatibility fix:

- exact-reference caches store one full logical center plus dynamic and total
  radius vectors from `current_zonotope(False/True)`;
- cached exact-reference comparison continues checking full logical dimensions
  and centers before invoking binding-native approximation loss;
- `RtlolaMatrixMetrics.dimension` remains the compact reducer dimension, derived
  from rows with nonzero total support, so explicit transform budgets are not
  over-rejected by exported logical zero rows;
- the exported logical dynamic row count is retained separately as
  `logical_dynamic_dimension`.

The binding-native reference/statistics redesign remains the preferred later
cleanup. PZR still should not infer stable row identities for semantic decisions,
and only `rlola_python_binding.ZonotopeConfig` transforms may mutate monitor
state.

Binding `01c92a2` additionally exposes public affine verdict intervals and
volume-ratio methods. PZR accepts affine intervals but intentionally does not
use volume ratios in objectives, caches, reports, or learning targets.
