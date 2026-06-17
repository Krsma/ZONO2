# RTLola Integration Notes

Notes from the side discussion on aligning PZR more closely with the original
RTLola zonotope-reduction setup and the experimental Python binding in
`/home/vlkr/Faks/phd/rlolapythonbinding`.

## Current Difference

PZR currently has a calibration-aware monitor model:

- Python monitor dynamics own state evolution.
- `MonitorState` carries `calibration_indices`.
- `ProtectedReducer` preserves selected persistent generator columns exactly.
- PZR-side `TriggerSpec` objects define trigger metrics and costs.
- MPC assumes cloneable Python monitor states.

This differs from the paper-faithful RTLola style, where the monitor owns the
semantics and zonotope/slack variables are reduced more anonymously as part of
bounded-memory monitoring.

Protected generators are a stricter extension, not just a standard zonotope
reduction detail. Under a fixed total generator budget, they make the reduction
problem harder because some columns cannot be merged, rotated, boxed, or
discarded. They can still be a more faithful model for persistent sensor
calibration or bias uncertainty.

## Better Alignment With RTLola

A paper-faithful RTLola path should treat RTLola as the owner of monitor
semantics and PZR as an external reducer/controller:

```text
RTLola accept_event
-> get zonotope matrix
-> convert matrix to PZR Zonotope
-> reduce all generator columns anonymously if over budget
-> apply reduced matrix back to RTLola
-> record RTLola public verdicts and PZR-side metrics
```

In this mode:

- No `calibration_indices`.
- No `ProtectedReducer`.
- Budget means total generator budget.
- RTLola `#[public]` streams are the primary verdict source.
- PZR reducers operate over the full exported zonotope matrix.

The calibration-aware design should remain available, but should be explicitly
framed as an extension rather than the default paper reproduction.

## Suggested Code Shape

Introduce a binding-oriented abstraction separate from the existing
`MonitorAdapter`, because the current binding is mutable and does not expose
cloneable logical monitor states:

```python
class RuntimeZonotopeMonitor:
    def accept_event(self, event, time) -> dict: ...
    def get_zonotope_matrix(self) -> np.ndarray: ...
    def apply_zonotope_matrix(self, matrix: np.ndarray) -> None: ...
```

Then build an online controller around it:

```python
class OnlineReductionController:
    def step(self, event, time):
        verdict = monitor.accept_event(event, time)
        z = matrix_to_zonotope(monitor.get_zonotope_matrix())
        if z.generator_count > budget:
            z = reducer.reduce(z, budget).reduced
            monitor.apply_zonotope_matrix(zonotope_to_matrix(z))
        return verdict
```

This is a better fit for the current binding than forcing RTLola into
`MonitorAdapter`, which assumes pure or cloneable state transitions.

## Binding Gaps

For an online/static pilot, the current binding is close. Useful improvements:

- Validate `apply_state()` inputs and return Python exceptions for shape errors,
  NaN, or infinity instead of panicking in Rust.
- Expose state shape and generator count helpers.
- Expose public input/output stream metadata.
- Optionally expose row/stream names for debugging and metric mapping.

For full MPC over RTLola states, the binding needs one of:

- `clone()`
- `snapshot()` and `restore()`
- `serialize_state()` and `deserialize_state()`

Without that, PZR cannot soundly branch future RTLola monitor states. Replaying
from the beginning is possible but expensive and should be treated as an
experimental fallback.

Column-role metadata is not required for paper-faithful anonymous reduction. It
only becomes important if we want to carry PZR's calibration-aware protected
generator extension into RTLola-backed experiments.

## Recommended Framing

Use two clearly named modes:

- Paper-faithful RTLola mode: anonymous generator reduction, no protected
  columns, RTLola verdicts as truth.
- Calibration-aware extension: persistent protected generators for bias or
  calibration structure, with explicit budget accounting and claims.

This avoids presenting protected generators as if they were part of the
original paper setup, while preserving them as a legitimate extension for
systems with persistent uncertainty.
