# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

PhD research code for **predictive zonotope reduction** — sound, bounded-memory reduction policies for runtime monitors that track sensor uncertainty as zonotopes `Z = c + Gξ`, `ξ ∈ [-1,1]^m`. Builds on Finkbeiner et al., "Cutting Corners on Uncertainty" (arXiv:2601.11358). Python 3.11+, no formatter/linter configured.

## Doc drift warning

**`README.md` and `science/SCIENCE.md` reference an older package layout (`pzr.core`, `pzr.reduction`, `pzr.control`, `pzr.benchmarks`, `pzr.robotics`, `pzr.learning`, `pzr.experiments.corl_suite`) and CLI entry points (`pzr-paper-figures`, `pzr-run-experiments`, `pzr-run-corl`) that no longer exist.** They also reference scenarios `robot`, `thermostat`, `iros` that aren't in the current source. Treat those docs as historical context for the research narrative, not as a guide to the code. The current state of the code is what's described below and in `AGENTS.md`. `src/predictive_zonotope_reduction.egg-info/PKG-INFO` is also stale; the next `pip install -e .` will refresh it.

## Architecture (current)

```
src/pzr/
├── zonotope/         # Zonotope dataclass + reducers + ProtectedReducer
│   ├── core.py       # Zonotope: immutable c + G ξ, ξ ∈ [-1,1]^m
│   ├── reduction.py  # BoxReducer, GirardReducer, CombastelReducer,
│   │                 #   PcaReducer, MethAReducer, ScottReducer, IdentityReducer
│   │                 #   ALL_REDUCERS registry
│   ├── protected.py  # ProtectedReducer: preserves calibration columns exactly
│   ├── metrics.py    # widths, interval bounds, inflation ratio
│   └── scoring.py    # heuristics consumed by reducers
├── monitoring/
│   ├── base.py       # MonitorAdapter Protocol + MonitorState
│   └── triggers.py   # TriggerSpec, evaluate_triggers, straddling/violation
├── mpc/              # MPC over reduction *actions* (not physical control)
│   ├── objectives.py # WeightedZonotopeCost (trigger width + straddling + ...)
│   ├── prediction.py # ConstantPredictor (placeholder future inputs)
│   ├── search.py     # Branch-and-bound tree_search
│   └── policies.py   # MPCPolicy, RolloutMPCPolicy, ReductionDecision
├── imitation/        # DAgger pipeline (selector distillation only)
│   ├── features.py   # 24-dim feature vector per decision point
│   ├── traces.py     # ReductionTrace, TraceCollector
│   ├── dataset.py    # Label encoding + train/val
│   ├── policy.py     # LearnedPolicy MLP + train_policy
│   └── dagger.py     # On-policy aggregation loop
├── systems/          # Math-only monitors (no simulator)
│   ├── omni_robot.py     # 5D, 1 calibration gen, 1 fresh noise/step
│   └── simple_robot.py   # 6D, 2 calibration gens, 2 fresh/step
├── envs/             # MuJoCo-backed monitors (require [sim] extra)
│   ├── base.py           # NoisySensorModel
│   ├── point_mass.py + point_mass_monitor.py  # 4D, 2 cal, 2 fresh
│   ├── robot_arm.py + robot_arm_monitor.py    # 6D joint state, triggers
│   │                                            in 2D Cartesian via FK Jacobian
│   └── mujoco_models/    # XML scenes
├── experiments/      # Benchmark orchestration + figures
│   ├── runner.py     # run_single, compute_ground_truth, summarize_results
│   ├── benchmark.py  # default_scenarios/methods, run_benchmark
│   ├── evaluation.py # aggregate_summary (bootstrap 95% CI)
│   ├── figures.py    # plot_* (Fig 4/5 panels, timeseries, bars, sweep)
│   ├── dagger_eval.py# train_and_evaluate_dagger (DAgger into benchmark)
│   ├── tables.py     # markdown/LaTeX/soundness reports
│   └── config.py     # BenchmarkConfig + smoke/standard/paper profiles
├── utils/            # seeding, timing, JSON serialization
└── cli.py            # pzr-benchmark entry point
```

## The soundness boundary (core invariant)

The whole project depends on **policy-independent soundness**: any selector over certified reducers preserves `Z ⊆ ρ(Z)` and `gen(ρ(Z)) ≤ K`. Concretely:

- Every `Reducer.reduce(z, budget)` returns a `ReductionResult` whose `certificate.is_sound` must be true and whose `reduced.generator_count ≤ budget`. `runner.py:214-217` counts violations.
- **`ProtectedReducer(base=R)` is the mechanism for keeping calibration generators across reductions.** It carves out columns named by `state.calibration_indices`, reduces only the residual against `budget - len(protected)`, and recombines. Default benchmark methods (`benchmark.py:146-153`) always wrap reducers in `ProtectedReducer`.
- Per-monitor: `state.calibration_indices` tracks which generator columns are persistent. After every reduction, calibration generators are re-numbered to `(0, ..., k-1)` (see `runner.py:66`).
- The MPC and learned policies only *rank* certified candidates. The actual reduction step always goes through a certified reducer; the learned policy falls back to `ProtectedReducer(BoxReducer())` if no candidate succeeds (`dagger_eval.py:56-65`).
- **`reduce_with_protection(reducer, z, budget, protected_indices)`** in `zonotope/protected.py` is the canonical helper for applying any `Reducer | ProtectedReducer` while honoring calibration protection. All four policy families route through it — static (`runner.py:62-65`), MPC search (`mpc/search.py:53-56`), MPC rollout (`mpc/policies.py:47-50, 150-153, 201-204`), learned (`imitation/policy.py:78-81`). Use the same helper in any new policy; calling `reducer.reduce(z, budget)` directly silently drops calibration for `ProtectedReducer` instances.

## The `trigger_zonotope` pattern (read before touching the MPC or features)

Monitors expose `trigger_zonotope(state) -> Zonotope`. Default returns `state.zonotope`. **`RobotArmMonitor` overrides** to project the 6D joint-space state into a 2D Cartesian end-effector zonotope via the FK Jacobian (`envs/robot_arm_monitor.py:104-105`). Trigger `state_index` values (`EE_X=0`, `EE_Y=1`) index into the *trigger* zonotope, not the raw state.

**Any code that evaluates triggers or computes trigger-axis metrics must call `monitor.trigger_zonotope(state)`, not `state.zonotope` directly.** `runner._trigger_metrics()` does this directly. `WeightedZonotopeCost` (`mpc/objectives.py`) and `extract_features` (`imitation/features.py`) both accept an optional `trigger_zonotope` callback (callable | precomputed `Zonotope` | None); benchmark construction at `benchmark.py:135-139` and every consumer call site (`runner.py:206-209`, `dagger_eval.py:54-57`, `:151-154`) pass `monitor.trigger_zonotope`. The earlier `state.zonotope` bugs at these sites were fixed in May 2026 — preserve the pattern when adding new consumers.

## Ground-truth comparison (Cutting-Corners alignment)

For every (scenario, seed), `compute_ground_truth(monitor, trace)` runs the monitor with NO reduction and records `(lower, upper, width_sum, verdicts)` per step. Each method then runs with `ground_truth=gt`, producing per-step `approx_error_sum` (L1 distance between reduced and exact bounds on trigger axes) and `false_positive` (one-sided: approx says "violation" while exact says "safe"). Aggregated as `mean_approx_error`, `abs_error_range`, `false_positive_rate`. Figures 4/5 in the paper are generated from these.

## Commands

```bash
# Install
pip install -e ".[dev]"                  # core + pytest
pip install -e ".[dev,learning,sim]"     # also torch + mujoco

# Tests (215+ total, all passing)
pytest                                   # full suite
pytest tests/test_full_eval.py -x -q     # one file, fail fast
pytest tests/test_robot_arm.py -k "trigger_zonotope"   # single test
pytest -m "not slow" --no-header         # if marks exist

# Benchmark CLI (the ONLY entry point declared in pyproject.toml)
pzr-benchmark --profile smoke --scenario omni_robot --output /tmp/smoke
pzr-benchmark --profile standard --output results/my_run
pzr-benchmark --profile paper --output results/paper_run

# Profiles: smoke (length=30, seeds=3), standard (200, 10), paper (200, 30)
# Scenarios: all | omni_robot | simple_robot | point_mass | robot_arm
# Method sets: all | static | standard
# DAgger runs by default; --no-dagger skips it
# Budget sweep: --budget-sweep "6,8,10,12" (disables DAgger automatically)
```

The MuJoCo scenarios (`point_mass`, `robot_arm`) load only if `import mujoco` succeeds; `benchmark.py:36-49` wraps the imports in try/except.

### Overnight runs

`--budget-sweep` automatically disables DAgger (`cli.py:82-85`), so a full overnight evaluation is two phases:

```bash
# Pre-flight (minutes): tests + smoke
pytest -q
pzr-benchmark --profile smoke --output /tmp/pzr-preflight

# Phase 1: full paper profile, all 4 scenarios, all 8 methods, with DAgger
pzr-benchmark --profile paper --output results/overnight/main \
  --dagger-iterations 3 --dagger-epochs 200

# Phase 2: budget sweep across the same scenarios/methods (DAgger auto-skipped)
pzr-benchmark --profile paper --budget-sweep "6,8,10,12,14,16" \
  --output results/overnight/sweep
```

Phase 1 produces `results/overnight/main/<scenario>/{timeseries,summary,aggregate}.csv` and `figures/*_fig4_panel.pdf` per scenario plus `learned_dagger` rows in every aggregate. Phase 2 adds `results/overnight/sweep/budget_{N}/` per budget and `budget_sweep/*_vs_budget.pdf` trade-off plots. Swap `--profile paper` for `--profile standard` to run with 10 seeds instead of 30 (~3× faster).

## Test fixtures

`tests/fixtures/cora_reference.json` is the *only* fixture file — it stores MATLAB CORA reference outputs for a 3D/6-generator/budget-4 zonotope and is consumed by `test_reduction.py` to assert parity for `girard`, `combastel`, `pca`, `methA`, `scott`. `BoxReducer` and `IdentityReducer` have no fixture.

## Important non-obvious behavior

- **Trace fidelity:** Per-seed traces are regenerated from `np.random.default_rng(seed)` every call. They are deterministic but NOT cached. Ground truth is also recomputed per (scenario, seed) inside `run_benchmark` (`benchmark.py:233-235`) — once, then shared across methods.
- **Calibration index reset after reduction:** `StaticReductionPolicy.decide` rewrites `calibration_indices` to `tuple(range(len(cal)))` (`runner.py:66`) because `ProtectedReducer` always puts protected columns first. If you write a new policy, do the same.
- **`evaluated_sequence_count` and `pruned_branches`** are tracked in `MPC.search.tree_search` and surface through `ReductionDecision`. They are accumulated only at the top level (not from intermediate recursive calls); see `search.py:170-171`.
- **DAgger label set:** The candidate set during DAgger training and eval is `{name: ProtectedReducer(base=r) for name, r in ALL_REDUCERS.items() if name != "identity"}`. Both sites must match (`dagger_eval.py:136`, `:180`).
- **`pzr.systems/` vs `pzr.envs/`** are not redundant: `systems/` holds synthetic math monitors with hand-tuned noise scales; `envs/` holds MuJoCo-backed monitors that get their noise from a `NoisySensorModel`. They share the `MonitorAdapter` protocol but use different noise abstractions.

## When you find yourself doing something risky

- **Don't add `IdentityReducer` to default candidate sets.** It's a certified no-op kept for theory tests; including it skews aggregates.
- **Don't bypass `ProtectedReducer` for monitors with calibration.** Reducing the raw zonotope will silently drop calibration generators and break long-horizon traces.
- **Don't aggregate per-step metrics into a single mean before per-seed grouping.** `summarize_results` produces per-seed values; `aggregate_summary` bootstraps across seeds. Mixing the order will collapse the variance you need for CIs.
- **Don't trust the trigger-axis index to refer to the state zonotope.** It refers to the trigger zonotope. For all monitors in `systems/` and for `point_mass`, those happen to be the same; for `robot_arm` they're not.

## Where to look for context, in order

1. `science/SCIENCE.md` — overall theory, soundness proofs, paper alignment. Layout claims are stale; the math claims are still correct.
2. `paper/` — current CoRL 2026 draft.
3. `tests/` — most reliable spec of current behavior. Tests pass; specs in stale `.md` files may not match.
4. `results/first_run/` — latest reference run for sanity checks on aggregate numbers.

## Audit findings

See `AUDIT.md` at the repo root for concrete bug/misalignment findings discovered during the deep audit (May 2026), including the verified trigger-zonotope, protected-reducer, and DAgger trace-label issues addressed after the audit.
