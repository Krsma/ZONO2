# Predictive Zonotope Reduction

Research code for sound, bounded-memory reducer selection in runtime monitors
that track uncertainty as zonotopes. The core invariant is policy-independent:
selectors may be static, predictive, or learned, but only certified reducers
mutate monitor state.

## Current Status

- Main package: Python 3.11+, source under `src/pzr/`.
- Optional RTLola integration: `src/pzr/rtlola/`, backed by the
  `rlolapythonbinding` submodule at commit `72622a3`.
- Current local baseline before this documentation cleanup:
  `pytest -q` -> `291 passed, 1 skipped`; the skipped test requires an
  installed `rlola_python_binding` extension.
- Generated experiment outputs belong under `results/` and should be
  regenerated through CLIs rather than hand-edited.

## Layout

- `zonotope/`: zonotope primitive, certified reducers, scoring, protection.
- `monitoring/`: monitor adapter protocol, states, and trigger evaluation.
- `mpc/`: receding-horizon reducer selection over certified actions.
- `imitation/`: feature extraction and regret/ranking learned selectors.
- `systems/`: math-only benchmark monitors.
- `envs/`: optional MuJoCo-backed monitors.
- `experiments/`: benchmark, replay, robotics probes, figures, tables.
- `rtlola/`: RTLola-native monitor wrapper, action search, benchmark CLI.
- `tests/`: pytest coverage for reducers, monitors, MPC, replay, and RTLola
  contracts.

## Install

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[dev,learning,sim]"
```

Use the second command only when learning and simulator-backed paths are needed.
For the RTLola robot-arm experiment, prefer the dedicated environment helper
instead of the broad `sim` extra:

```bash
tools/setup_robot_arm_env.sh
tools/run_rtlola_robot_arm.sh --length 40 --seeds 1 --method-set static --output /tmp/pzr-rtlola-arm-smoke
```

The robot-arm RTLola/MuJoCO path does not need `safety-gymnasium`. Keeping it
out of this environment avoids the old `pygame`/Gymnasium resolver conflicts
that can downgrade NumPy or MuJoCO.

## Core Commands

```bash
pytest
pytest tests/test_full_eval.py -x -q

pzr-benchmark --profile smoke --scenario omni_robot --output /tmp/pzr-smoke
pzr-benchmark --profile standard --output results/my-run
pzr-benchmark --profile paper --scenario all --method-set paper_core --output results/paper-core

python -m pzr.experiments.robotics_replay sweep \
  --candidate all --trace-source procedural --monitor physical \
  --scenario-family stress --budgets 8,10,12,16,20,24 \
  --length 80 --seeds 2 --output /tmp/pzr-robotics-sweep

pzr-rtlola-benchmark --profile smoke --scenario omni_robot --output /tmp/pzr-rtlola-smoke
pzr-rtlola-benchmark --profile smoke --scenario robot_arm --trace-kind figure8_violated --budget 80 --output /tmp/pzr-rtlola-arm
tools/run_rtlola_robot_arm.sh --output /tmp/pzr-rtlola-arm
```

`scenario=all` runs the current default/headline benchmark scenarios and
excludes deprecated diagnostic scenarios. Explicitly runnable scenarios include
`omni_robot`, `simple_robot`, `point_mass`, and `robot_arm` when optional
MuJoCo imports are available.

## RTLola Binding

The submodule `rlolapythonbinding/` now exposes snapshot-capable monitor state:
`EvaluatorState`, `state()`, `apply_state()`, `accept_event_from_state()`,
`current_zonotope()`, `state_zonotope()`, and `approx_loss()`.

PZR uses this for safe branch search in `src/pzr/rtlola/`. The current binding
does not expose arbitrary matrix writeback into RTLola, so the RTLola path
selects RTLola built-in `ZonotopeConfig` transforms rather than applying native
PZR reducers back into the RTLola engine.

To build the optional extension:

```bash
tools/setup_rtlola_binding.sh
pytest tests/test_rtlola_metrics.py tests/test_rtlola_binding_contract.py -q
```

The robot-arm wrapper handles the OpenBLAS preload needed by the RTLola
extension. For manual RTLola runs, use the preload printed by the relevant
setup script; without it, importing the extension can fail with an unresolved
`cblas_dgemm` symbol.

## Reference Docs

- `AGENTS.md`: operational repository instructions.
- `AUDIT.md`: current readiness audit and known blockers.
- `science/SCIENCE.md`: compact science and soundness notes.
- `science/RTLOLA_INTEGRATION_NOTES.md`: RTLola binding and integration state.
- `science/EXPERIMENT_READINESS.md`: consolidated experiment-readiness notes.
- `paper/related_work_foundation.md`: paper-facing related-work framing.
