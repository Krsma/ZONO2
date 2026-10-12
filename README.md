# Predictive Zonotope Reduction

**Adaptive reducer selection for precise runtime monitoring under uncertainty.**

PZR chooses how to reduce a monitor's zonotope representation while retaining
sound outer bounds. It provides static baselines, beam-search model predictive
control, and neural policies trained to rank reduction actions. RTLola performs
monitoring and native transformations; Python provides learning, simulation,
search, and evaluation.

[Paper](https://arxiv.org/abs/2609.03699v1) ·
[Learning guide](docs/learning/README.md) ·
[Tutorials](notebooks/) ·
[Private Docker distribution](https://github.com/Krsma/ZONO2-docker)

## Start experimenting

The quickest source-only workflow trains a neural selector on the bundled
recorded teacher data. It needs Python 3.11 or newer; it does **not** need the
native monitor, a simulator, or access to private submodules.

```bash
git clone https://github.com/Krsma/ZONO2.git
cd ZONO2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[learning]'
jupyter lab notebooks/00_start_here.ipynb
```

Or train directly:

```python
from pzr import learning

# Six recorded trajectories, with disjoint training and validation splits.
task = learning.load_task("robot_arm", bundle="starter", budget=40)
run = learning.train_selector(task, config=learning.TrainingConfig(epochs=2))
run.save("workspace/my-selector")
```

This short run checks the workflow; it does not reproduce the paper's trained
specialists. The starter bundle includes teacher costs and a frozen reference
model. Full Clean148 data is distributed separately.

For a ready-made environment with the native monitor and dependencies, use the
[private Docker repository](https://github.com/Krsma/ZONO2-docker) (access required).
It contains the image-sharing and Jupyter launch instructions.

## What can I change?

| Goal | Start here |
| --- | --- |
| Train and evaluate a reducer selector | [00 — Start here](notebooks/00_start_here.ipynb) |
| Replace the PyTorch model or training loop | [01 — Build your selector](notebooks/01_build_your_selector.ipynb) |
| Inspect geometry and evaluate a live monitor | [02 — Monitor and geometry](notebooks/02_understand_the_monitor.ipynb) |
| Experiment with Gymnasium and representations | [03 — Gym and representations](notebooks/03_use_gym.ipynb) |
| Define input tensors from zonotope properties | [04 — Define zonotope features](notebooks/04_define_zonotope_features.ipynb) |

Selectors score `girard`, `scott`, `pca`, and `combastel`; lower scores are better.
The default input is the pre-event Geometry15 feature vector. Custom feature
functions and embedding models can feed Gym observations and monitor policies.
Recorded datasets contain Geometry15, so supervised training on new geometry
features requires collecting matching teacher-state geometry.

**Soundness boundary:** learned models select native actions. Only the pinned
RTLola binding mutates monitor state. Direct neural zonotope prediction and
certified enclosure checking are [follow-up research](docs/learning/follow_up.md),
not supported live-monitor operations.

## Run the native monitor

Native monitoring and Gym rollouts require the pinned release binding. Its source
submodule currently requires access to the CISPA Git server; installing the Python
package alone does not provide it. Authorized users can build on Linux with
Python 3.11, Rust, and Conda at `external/miniconda3`:

```bash
git submodule update --init --recursive
tools/setup_robot_arm_env.sh
tools/run_rtlola_robot_arm.sh --length 20 --seeds 1 \
  --method-set core --output /tmp/pzr-arm
```

The output contains metrics, configuration, and failure records. Use
`pzr-benchmark --help` and `pzr-learning --help` inside the environment for the
benchmark and data-collection interfaces.

## Reproduce experiments and contribute

See [development and reproducibility](docs/development.md) for tests, experiment
entrypoints, repository organization, and the distinction between a public
source checkout and the separately distributed research artifacts.

## Citation

If you use this work, please cite
[Predictive Zonotope Reduction: Precise Runtime Monitoring under Uncertainty](https://arxiv.org/abs/2609.03699v1).

```bibtex
@misc{krsmanovic2026predictive,
  title = {Predictive Zonotope Reduction: Precise Runtime Monitoring under Uncertainty},
  author = {Vladimir Krsmanovic and Florian Kohn and Bernd Finkbeiner and Milan Simovic},
  year = {2026},
  eprint = {2609.03699},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2609.03699v1}
}
```

## License

PZR's original source code is released under the [MIT License](LICENSE).
Vendored robot-arm assets retain their [Apache-2.0 license and attribution](src/pzr/envs/mujoco_models/low_cost_robot_arm/README.md).
The RTLola binding and other dependencies retain their own licenses and access
requirements; the PZR license does not relicense them.
