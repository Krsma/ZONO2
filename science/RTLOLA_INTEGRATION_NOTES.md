# RTLola Integration Notes

The local `rlolapythonbinding` submodule is pinned to commit `a2184eb`, whose
latest changes added reusable evaluator snapshots, type-aware input conversion,
better Python error propagation, and native PCA/Althoff-A approximation
constructors. This removes the main blocker for branch-based planning over
RTLola monitor states and expands the native reducer set.

## Binding API We Can Use

`rlola_python_binding` exposes:

- `RLolaMonitor(spec)` and `RLolaMonitor.from_path(path)`;
- `accept_event(event, time, approximation=None)`;
- `state() -> EvaluatorState`;
- `apply_state(state)`;
- `accept_event_from_state(state, event, time, approximation=None)`;
- `current_zonotope(include_constant_slack=False)`;
- `state_zonotope(state, include_constant_slack=False)`;
- `approx_loss(zonotope_matrix)` and `approx_loss_state(state)`;
- `ZonotopeConfig.none|interval|interval_hull|colinear|colinear_scale|scott|girard|pca|althoff_a`.

The current Python wrapper clones `EvaluatorState` when applying or branching,
so saved snapshots can be reused for repeated planner branches.

## PZR Integration Shape

`src/pzr/rtlola/engine.py` owns two monitors:

- `live`: committed execution, strict event-time progression;
- `planner`: branch evaluation from saved `EvaluatorState` snapshots.

`RtlolaEngine.branch_step` uses `accept_event_from_state` to evaluate candidate
actions without mutating committed state. `RtlolaEngine.live_step` commits the
selected action to the live monitor. Metrics come from `state_zonotope` and
`current_zonotope`, with dynamic and total generator counts separated by the
binding's `include_constant_slack` flag.

This is the correct surface for RTLola-native MPC over built-in zonotope
transforms.

## Important Limitation

The current binding does not expose arbitrary matrix writeback into RTLola.
Earlier planning notes assumed a loop of:

```text
extract RTLola zonotope matrix -> apply native PZR reducer -> write matrix back
```

That is not the current implementation path. PZR can inspect RTLola matrices
and select RTLola `ZonotopeConfig` transforms, but it cannot yet inject a
matrix reduced by PZR's own reducer classes into the RTLola evaluator.

If native PZR reducers must operate inside RTLola, the binding needs a new,
validated matrix-apply API that preserves RTLola evaluator invariants and gives
clear Python exceptions for shape, NaN, infinity, and stream-layout errors.

## Current Validation State

In the default active Python environment before this documentation update:

```bash
pytest tests/test_rtlola_metrics.py tests/test_rtlola_binding_contract.py -q
# 3 passed, 1 skipped
```

The skip means `rlola_python_binding` was not importable. Before any serious
RTLola experiment, use the existing `pzr-rtlola` conda environment or rebuild it
with `tools/setup_rtlola_binding.sh`. The extension currently needs the conda
OpenBLAS preload:

```bash
LD_PRELOAD=/home/vlkr/Faks/phd/ZONO2/external/miniconda3/envs/pzr-rtlola/lib/libopenblas.so \
LD_LIBRARY_PATH=/home/vlkr/Faks/phd/ZONO2/external/miniconda3/envs/pzr-rtlola/lib:${LD_LIBRARY_PATH:-} \
CONDA_NO_PLUGINS=true external/miniconda3/bin/conda run -n pzr-rtlola \
  python -m pytest tests/test_rtlola_metrics.py tests/test_rtlola_binding_contract.py -q

LD_PRELOAD=/home/vlkr/Faks/phd/ZONO2/external/miniconda3/envs/pzr-rtlola/lib/libopenblas.so \
LD_LIBRARY_PATH=/home/vlkr/Faks/phd/ZONO2/external/miniconda3/envs/pzr-rtlola/lib:${LD_LIBRARY_PATH:-} \
CONDA_NO_PLUGINS=true external/miniconda3/bin/conda run -n pzr-rtlola \
  python -m pzr.rtlola.cli --profile smoke --scenario omni_robot --output /tmp/pzr-rtlola-smoke
```

This pass validated that command shape: the binding contract tests produced
`10 passed`, and the RTLola smoke benchmark completed successfully.

## Framing

There are now two clear modes:

- Native PZR mode: Python monitors, certified PZR reducers, protected
  calibration generators when the monitor exposes them.
- RTLola-native mode: RTLola owns monitor semantics, PZR chooses among
  RTLola-native zonotope transforms, and `#[public]` streams provide verdicts.

Do not conflate these modes in paper claims. The shared claim is the
policy-independent soundness boundary; the reducer implementation boundary is
different.

## Low-Cost Arm Scenario

`pzr-rtlola-benchmark --scenario robot_arm` is the first RTLola-native
low-cost 5-DOF arm experiment. It uses the vendored Apache-2.0 MuJoCo model
assets, deterministic figure-eight/square TCP traces, and a repo-owned
Float64 RTLola spec with encoder uncertainty constants `Q`, `J`, `I`, and
`H`. Public streams expose `dist_to_expected` and Euclidean cumulative `tpl`.

The first-pass online selection cost minimizes widths of state-zonotope rows
feeding TCP drift and toolpath metrics. This is deliberately a pragmatic
dashboard objective. Revisit cost-function design before making stronger
claims, especially objectives based directly on public stream widths, trigger
ambiguity, and trigger-confusion risk.

## Audit Note: RTLola Reduction Ordering and Budget Semantics

RTLola currently applies the configured zonotope transform before accepting the
next event. The event then creates fresh slack variables and evaluates streams.
This differs from the original Python monitor loop, where the monitor was
stepped first and reduction was applied to the post-step state.

The RTLola-native benchmark now treats `budget` as the RTLola transform bound:
static reducers and MPC candidates pass exactly `ZonotopeConfig.<method>(budget)`
to the binding. Search reduction is triggered from the pre-event state: if the
current dynamic slot count is within the bound, the benchmark commits `none`;
if it exceeds the bound, the selected RTLola reducer is applied before the
event.

Post-event dynamic slot counts can exceed `budget`, because event processing
happens after the transform and may allocate fresh slack. That condition is now
reported as `post_event_over_bound`, not as a budget violation. The old
reserve-subtraction logic (`budget - inferred_fresh_generators`) was a semantic
mismatch and should not be used for RTLola-native headline runs.

The dense dynamic slot count exposed by `state_zonotope(False)` remains the
main RTLola memory proxy. The benchmark also reports active nonzero dynamic
generators and zero dynamic generator columns so zero slack-ID holes can be
audited separately from the transform-bound semantics.

`infer_fresh_generator_reserve` is retained only as a diagnostic helper. It is
useful for understanding per-cycle growth, but it no longer determines reducer
bounds in static or MPC runs.

The first follow-up run after this semantic cleanup should exclude
`colinear_scale` from both static reports and MPC candidates to avoid hiding
the behavior of the fast methods behind a very slow transform.

After updating the binding to `a2184eb`, `pca` is practical as a static method
and MPC candidate. `althoff_a` remains available as a native static method, but
the first branch-level probe showed it is too slow for the main beam pool on
the robot-arm trace; keep it out of overnight MPC unless a separate audit
reopens that choice.

Earlier execution note: the full exact-reference sweep was too slow for
interactive turnaround, so the previous 120/160/240 budget sweep was run with
`reference-mode off`, plus a short exact audit at budget 160 and length 30.
Those older runs predate the transform-bound cleanup and should not be used as
final RTLola-native budget tables.
