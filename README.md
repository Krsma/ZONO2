# Predictive Zonotope Reduction

Research code for experimenting with sound, monitor-aware zonotope reduction
policies for bounded-memory runtime monitoring.

The package is intentionally small at this stage:

- `src/pzr/core` contains the zonotope and certificate primitives.
- `src/pzr/reduction` contains certified reducer interfaces and baselines.
- `src/pzr/monitoring` defines the black-box monitor adapter boundary.
- `src/pzr/control` contains static and receding-horizon reduction policies.
- `src/pzr/benchmarks` contains the robot and thermostat monitor families.
- `science/SCIENCE.md` records the project theory and code mapping.

Run tests with:

```bash
pytest
```

Run the default paper-style robot benchmark with:

```bash
pzr-benchmark robot --length 200 --budget 8 --horizon 4 --seeds 30 --out results/robot
```

Run the non-robot thermostat benchmark with:

```bash
pzr-benchmark thermostat --length 200 --budget 8 --horizon 4 --seeds 30 --out results/thermostat
```

Run the full paper experiment suite with one command:

```bash
pzr-run-experiments --profile paper --out results/experiment-suite
```

Use `--profile smoke` for a fast end-to-end artifact check, or
`--profile standard` for a moderate preflight run. The suite runs the robot,
simple robot, and thermostat benchmarks, trains and evaluates the learned
distilled policy, regenerates robot-focused figures, writes aggregate CSVs,
and creates a manifest, artifact index, and tarball for packaging.

The default suite includes static box, Girard, Combastel, MethA, Scott, PCA,
adaptive, and keep-generator reducers, the original one-action MPC, sequence
MPC, and the rollout MPC variants. Explicit `no_reduction` remains implemented
for future controlled experiments, but it is not part of the current paper
experiment candidate sets. The wide rollout evaluates broad protected
precision reducers as first actions, then uses protected Girard as the fixed
future-overflow policy with box only as a last-resort certified fallback.

Use `--predictor-mode both` to run online and oracle prediction in one
artifact set. The command writes `raw_runs.csv`, `summary.csv`,
`comparisons.csv`, `predictor_comparisons.csv`, `timeseries.csv`,
`bounds_timeseries.csv`, `decision_features.csv`, `selection_summary.csv`,
`predicted_sequence_summary.csv`, `config.json`, and `report.json`.
