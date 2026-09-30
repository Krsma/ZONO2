# Predictive Zonotope Reduction

Research software for evaluating static, predictive, and learned reducer
selection for RTLola monitors with zonotope uncertainty. RTLola evaluates the
monitor and applies native reductions; Python provides traces, search,
learning, and benchmark reporting.

## Setup

The setup scripts target Linux with Python 3.11, Rust (`cargo` and `rustc`),
and Conda installed at `external/miniconda3`. Access to the pinned RTLola
binding submodule on the CISPA Git server is required.

```bash
git submodule update --init --recursive
tools/setup_robot_arm_env.sh
```

The setup builds the pinned binding in release mode and installs the package
and MuJoCo support into `external/miniconda3/envs/pzr-robot-arm`.

## Quick start

Run a short robot-arm benchmark:

```bash
tools/run_rtlola_robot_arm.sh --length 20 --seeds 1 \
  --method-set core --output /tmp/pzr-arm
```

Outputs include CSV metrics, configuration, and failure records. After
activating the environment, use `pzr-benchmark --help` and
`pzr-learning --help` for benchmark and learning options.

## Tests

Run the full suite with the native binding environment:

```bash
LD_PRELOAD="$PWD/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so" \
PYTHONPATH=src external/miniconda3/envs/pzr-robot-arm/bin/python -m pytest
```

Without the native binding, integration tests are skipped.

## Code layout

- [`src/pzr/rtlola/`](src/pzr/rtlola/): specifications, traces, binding adapter,
  search, and benchmarks.
- [`src/pzr/learning/`](src/pzr/learning/): reducer selection training and evaluation.
- [`tests/`](tests/): unit tests and native binding contracts.
- [`tools/`](tools/): environment setup and experiment utilities.

Robot-arm model attribution and licensing are documented in the
[vendored model README](src/pzr/envs/mujoco_models/low_cost_robot_arm/README.md).
