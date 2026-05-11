# Predictive Zonotope Reduction

Research code for experimenting with sound, monitor-aware zonotope reduction
policies for bounded-memory runtime monitoring.

The package is intentionally small at this stage:

- `src/pzr/core` contains the zonotope and certificate primitives.
- `src/pzr/reduction` contains certified reducer interfaces and baselines.
- `src/pzr/monitoring` defines the black-box monitor adapter boundary.
- `src/pzr/control` contains static and receding-horizon reduction policies.
- `src/pzr/benchmarks` starts with a Python version of the paper's robot monitor.
- `science/SCIENCE.md` records the project theory and code mapping.

Run tests with:

```bash
pytest
```

Run the default paper-style robot benchmark with:

```bash
pzr-benchmark robot --length 200 --budget 8 --horizon 4 --seeds 30 --out results/robot
```

The default suite includes static box, Girard, Combastel, MethA, Scott, PCA,
adaptive, and keep-generator reducers, plus the original one-action MPC,
sequence MPC, and `mpc_rollout_girard`. The rollout MPC evaluates candidate
first reductions under a short predicted trace, then uses protected Girard as
the fixed future-overflow policy with box as a last-resort certified fallback.

Use `--predictor-mode both` to run online and oracle prediction in one
artifact set. The command writes `raw_runs.csv`, `summary.csv`,
`comparisons.csv`, `predictor_comparisons.csv`, `config.json`, and
`report.json`.
