# Project Readiness Audit — June 2026

This audit records the current state after inspecting the pulled
`rlolapythonbinding` submodule and the active PZR codebase. Treat this file as
a checklist of claims to verify, not as experimental evidence.

## Baseline Health

- Parent repository worktree was clean before the documentation update.
- `rlolapythonbinding` is a git submodule pinned at
  `72622a3f2e756bafca199cf20cb34b5babee5d6f`.
- Focused RTLola checks in the default active Python environment:
  `pytest tests/test_rtlola_metrics.py tests/test_rtlola_binding_contract.py -q`
  produced `3 passed, 1 skipped`; the skip is because the Rust extension is not
  installed in that environment.
- Binding-enabled validation in `external/miniconda3/envs/pzr-rtlola` requires
  the OpenBLAS preload documented by `tools/setup_rtlola_binding.sh`. With
  `LD_PRELOAD=.../lib/libopenblas.so` and `LD_LIBRARY_PATH=.../lib`, the
  binding import succeeds and
  `python -m pytest tests/test_rtlola_metrics.py tests/test_rtlola_binding_contract.py -q`
  produced `10 passed`.
- RTLola smoke validation with the same preload completed successfully:
  `python -m pzr.rtlola.cli --profile smoke --scenario omni_robot --output /tmp/pzr-rtlola-smoke`.
- Full test suite before edits: `pytest -q` produced
  `291 passed, 1 skipped in 95.57s`.

## RTLola Binding Findings

The binding now provides the API needed for branch-based planning:

- `EvaluatorState` opaque snapshots.
- `RLolaMonitor.state()` to capture a state.
- `RLolaMonitor.apply_state(state)` to restore a cloned snapshot.
- `RLolaMonitor.accept_event_from_state(state, event, time, approximation)` to
  branch from a snapshot and return `(verdict, new_state)`.
- `current_zonotope()` and `state_zonotope()` for NumPy matrix extraction.
- `approx_loss()` for interval-hull error against a saved state.

PZR already uses this shape in `src/pzr/rtlola/engine.py`: a live monitor owns
committed execution and a planner monitor evaluates branches from snapshots.
This is the right architecture for MPC-style reducer selection over RTLola
state.

The binding does not currently expose arbitrary reduced matrix writeback into
RTLola. Therefore, the current RTLola integration should be described as
selection among RTLola-native `ZonotopeConfig` transforms, not as applying PZR's
native reducer objects into RTLola.

## Current Experiment Surface

Active benchmark and experiment paths:

- `pzr-benchmark` / `python -m pzr.cli`
- `pzr-robot-arm-animation`
- `pzr-robotics-replay`
- `pzr-paper-tables`
- `pzr-rtlola-benchmark` / `python -m pzr.rtlola.cli`
- `python -m pzr.experiments.robotics_probe`

Current benchmark defaults are documented in `AGENTS.md`. Deprecated diagnostic
scenarios such as `simple_robot` and `point_mass` remain explicitly runnable
but are not part of `scenario=all`.

## Open Readiness Gaps

- Binding-enabled validation should be repeated before paper-critical RTLola
  runs, using the OpenBLAS preload:

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

- The `tools/run_corl_*` scripts reference old or removed entry points such as
  `pzr-run-corl` and `pzr.experiments.corl_firmware_replay`. They should be
  treated as legacy notes until either removed or ported to the current
  `robotics_probe` / `robotics_replay` infrastructure.
- Paper-facing robotics evidence is not yet final. The current active path is
  the replay/probe workflow, especially high-generator-budget sweeps with
  `mpc_beam3` against the best fixed static reducer.
- Historical documents previously mixed TACAS, CoRL, ICRA, DAgger, and removed
  robotics code paths. The active references have been consolidated into
  `science/EXPERIMENT_READINESS.md`.

## Preserved Invariants

- Soundness remains policy-independent: predictors, MPC, and learned rankers
  choose among certified reducers.
- `ProtectedReducer` and `reduce_with_protection` are the required path when
  persistent calibration generators are present.
- Trigger-derived metrics must use `monitor.trigger_zonotope(state)`.
- `IdentityReducer` remains outside default benchmark, MPC, and learned-policy
  candidate sets unless a task explicitly reopens no-op experiments.

## Recommended Next Preflight

Before a serious experiment:

1. Run `pytest -q`.
2. Build and validate the RTLola binding with the documented BLAS preload if
   the experiment uses `pzr.rtlola`.
3. Run a small `pzr-benchmark` smoke with the intended scenario/method set.
4. For robotics replay, run the `sweep` subcommand before ad hoc eval calls.
5. Check all generated CSVs for nonempty rows, zero budget violations, zero
   unsound certificates, and method differentiation against the best static
   baseline.
