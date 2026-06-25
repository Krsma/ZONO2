# Experiment Readiness Notes

This file consolidates the useful parts of the previous CoRL, robotics spike,
and contingency planning notes. It tracks what must be true before running a
serious experiment.

## Current Direction

The project should lead with monitor-side bounded-memory uncertainty tracking:
predictive reducer selection can reduce avoidable monitor imprecision at fixed
memory while preserving soundness because only certified reducers mutate state.

The next serious experiment should be chosen only after a small diagnostic run
shows:

- reductions happen often but triggers are not saturated;
- at least three static reducers differ on trigger width or intervention
  labels;
- `mpc_beam3` or another predictive method improves a paper-facing metric
  against the best fixed static reducer, not only against a weak named static
  reference;
- all methods have zero budget violations and zero unsound certificates;
- visualization artifacts explain the physical state, trigger projection,
  generator count, and reduction events without special pleading.

## Active Experiment Paths

Use current entry points only:

```bash
pzr-benchmark --profile smoke --scenario omni_robot --output /tmp/pzr-smoke

python -m pzr.experiments.robotics_probe \
  --candidate all --trace-source live --length 120 --budget 10 \
  --output /tmp/pzr-robotics-probe

python -m pzr.experiments.robotics_replay sweep \
  --candidate all --trace-source procedural --monitor physical \
  --scenario-family stress --budgets 8,10,12,16,20,24 \
  --length 80 --seeds 2 --horizon 4 --beam-width 4 \
  --output /tmp/pzr-robotics-high-k-sweep

python -m pzr.experiments.robotics_replay render \
  --eval-dir /tmp/pzr-robotics-replay-eval \
  --candidate all --methods scott,mpc_beam3 \
  --output /tmp/pzr-robotics-replay-viz

tools/run_pzr_icra_table_matrix.sh
```

The old `pzr-run-corl` and `pzr.experiments.corl_*` references are not current
active code paths. Keep their idea-level lessons, but do not use those commands
as preflight instructions.

## Scenario Notes

`omni_robot` remains the stable math-only baseline. It is good for quick
regression checks and RTLola integration alignment.

`robot_arm` remains useful for trigger-projection coverage and visualization
sanity checks. It is not currently the main paper visualization.

`drone` and `f1tenth` live in `robotics_probe` / `robotics_replay`, outside
default benchmark scenarios. The preferred current workflow is replay
evaluation with `--monitor physical --scenario-family stress`, followed by a
budget sweep. Exact `mpc_sequence3` is diagnostic; default sweeps should use
`mpc_beam3` to keep runs practical.

`simple_robot` and `point_mass` are deprecated from default/headline runs. They
remain explicitly runnable regression fixtures.

## Robotics Replay Outputs

The `robotics_replay eval` path writes benchmark-style `timeseries.csv`,
`summary.csv`, and `aggregate.csv`, plus policy gain, winner-by-step, trace,
scenario, intervention, metadata, raw payload, and derived-stream artifacts.

The `sweep` path is preferred for budget trade-off questions. It writes:

- `budget_sweep_summary.csv`
- `budget_policy_gain.csv`
- `budget_reducer_counts.csv`
- `budget_runtime.csv`
- `budget_scenario_summary.csv`
- `budget_intervention_summary.csv`
- `budget_sweep_metadata.json`
- figures under `figures/`

The selected visualization budget should maximize robust `mpc_beam3` gain
against the best fixed static reducer, then mean gain, reducer switches, and
runtime.

## RTLola Preflight

Before an RTLola-native experiment:

```bash
tools/setup_rtlola_binding.sh
LD_PRELOAD=/home/vlkr/Faks/phd/ZONO2/external/miniconda3/envs/pzr-rtlola/lib/libopenblas.so \
LD_LIBRARY_PATH=/home/vlkr/Faks/phd/ZONO2/external/miniconda3/envs/pzr-rtlola/lib:${LD_LIBRARY_PATH:-} \
CONDA_NO_PLUGINS=true external/miniconda3/bin/conda run -n pzr-rtlola \
  python -m pytest tests/test_rtlola_metrics.py tests/test_rtlola_binding_contract.py -q
LD_PRELOAD=/home/vlkr/Faks/phd/ZONO2/external/miniconda3/envs/pzr-rtlola/lib/libopenblas.so \
LD_LIBRARY_PATH=/home/vlkr/Faks/phd/ZONO2/external/miniconda3/envs/pzr-rtlola/lib:${LD_LIBRARY_PATH:-} \
CONDA_NO_PLUGINS=true external/miniconda3/bin/conda run -n pzr-rtlola \
  python -m pzr.rtlola.cli --profile smoke --scenario omni_robot --output /tmp/pzr-rtlola-smoke
```

This is required because the normal Python test environment may skip the live
binding contract when `rlola_python_binding` is not installed.

## Learning Scope

Regret/ranking distillation is deployability evidence, not the soundness
argument. A learned selector ranks reducers; certified reducers still provide
the enclosure and budget guarantees.

Treat learned rows as headline evidence only when held-out metrics and regret
diagnostics show low chosen regret, no pathological ranking collapse, no budget
violations, and no unsound certificates.

## Paper Claim Shape

Primary claim:

> Predictive certified reduction improves bounded-memory monitor precision, or
> downstream intervention quality, compared with fixed static reducers.

Safety claim:

> Soundness is policy-independent because every state mutation is performed by
> a certified reducer.

Deployment claim, only when validated:

> A learned ranker can approximate the predictive selector at lower latency
> while remaining outside the trusted soundness boundary.
