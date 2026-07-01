# Repository Guidelines

## Project Structure

This Python 3.11 research package is RTLola-centered:

- `src/pzr/rtlola/`: specifications, trace adapters, binding wrapper, native
  transform catalog, search, benchmark execution, reporting, and CLI.
- `src/pzr/learning/`: generic regret-ranking data/model/training code.
- `rlolapythonbinding/`: pinned binding submodule.
- `tests/`: pure tests plus binding-backed semantic contracts.
- `tools/`: reproducible environment setup and robot-arm smoke execution.

Robot-arm trace CSVs and the vendored MuJoCo model are data/validation assets,
not an alternative Python monitor.

## Setup and Tests

```bash
git submodule update --init --recursive
tools/setup_robot_arm_env.sh

LD_PRELOAD="$PWD/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so" \
PYTHONPATH=src external/miniconda3/envs/pzr-robot-arm/bin/python -m pytest
```

The normal Python environment runs pure tests and skips binding integration
tests. Release validation must use the binding environment with no skips.

Useful smokes:

```bash
pzr-benchmark --profile smoke --scenario omni_robot --method-set core \
  --output /tmp/pzr-omni

tools/run_rtlola_robot_arm.sh --length 20 --seeds 1 --method-set core \
  --output /tmp/pzr-arm

PZR_OUT_DIR=results/rtlola-arm-binding-loss \
  tools/run_rtlola_robot_arm_fpr_overnight.sh

pzr-benchmark --profile smoke --scenario omni_robot --budget 10 \
  --methods girard,mpc_beam --learned-mode regret \
  --regret-iterations 1 --regret-epochs 2 \
  --regret-train-seeds 1 --regret-eval-seeds 1 \
  --output /tmp/pzr-learned
```

## Coding and Testing

Use 4-space indentation, type hints, immutable dataclasses where appropriate,
and grouped standard-library/third-party/local imports. Tests use pytest,
`tmp_path`, and `numpy.testing`.

Changes to scenarios, actions, search, or learning require tests for:

- deterministic state branching and tie behavior;
- exact RTLola transform-bound semantics;
- dense versus active generator accounting;
- outer-bound soundness against an unreduced branch;
- constant calibration generator preservation;
- trigger/public-stream keys from the packaged specification;
- fallback and infeasible-candidate accounting;
- learned candidate alignment and direct-inference behavior;
- non-empty benchmark and learning artifacts.

## Trusted Boundary

Selectors may inspect states and choose actions, but only
`rlola_python_binding.ZonotopeConfig` transforms may mutate monitor state.
Do not add matrix writeback or Python-side reducers.

The default MPC/learning candidates are `girard`, `scott`, `interval_hull`,
and `pca`. Do not add `none`, `interval`, unbounded transforms, clustering, or
Combastel without an explicit experiment change. `none` is the exact baseline
and automatic under-bound action; `interval` is fallback-only.

`budget` is the binding transform bound. Never subtract a fresh-generator
reserve or interpret post-event dense slots as a violation. Preserve the
distinction between dynamic, active, zero, and constant generators.

MPC and teacher costs use binding-native terminal approximation loss. Do not
replace it with width, trigger-straddling, or a Python proxy during unrelated
cleanup.

Benchmark reference mode controls offline metrics and caching only. MPC and
teacher searches construct their own unreduced horizon rollouts.

Robot-arm trigger labels and public metrics come from
`rtlola/specs/robot_arm.lola`. Constant encoder-calibration slack must remain
unchanged by dynamic reduction.

For full-length classification metrics use `--reference-mode verdict`.
Verdict reference caches contain only exact trigger booleans and are reusable
across methods and budgets. FPR uses exact negative steps as its denominator;
FNR uses exact positive steps. Keep `exact` mode for short matrix/loss studies
because it retains the full unreduced state history.

## Repository Safety

Do not discard uncommitted work. Use `git pull --ff-only`, pin submodules
through the superproject, and avoid setup scripts that silently fetch or
checkout another binding revision. Do not hand-edit generated files in
`results/`.

Use concise imperative commits. Report commands run, changed experiment
semantics, binding revision changes, and generated-artifact impact.
