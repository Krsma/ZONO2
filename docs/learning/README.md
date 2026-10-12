# Learning interface tutorials and usage guide

The notebooks provide tutorials for training and evaluating reducer selectors,
customizing PyTorch networks, inspecting monitor geometry, and running Gymnasium
experiments. Start with **00_start_here.ipynb** for the supervised workflow, then
**01_build_your_selector.ipynb** for architecture comparisons and custom training.
The main workflow uses recorded teacher costs and ordinary PyTorch modules;
Gymnasium and the native monitor are available for trajectory experiments.

## Launch Jupyter

With the supplied image loaded as `pzr-learning:local`, run from the repository:

```bash
tools/learning/start_lab.sh
```

Open the local URL printed by Jupyter (including its token). The image includes
the dependencies, pinned release binding, six recorded trajectories, teacher
labels, and notebooks. It requires no teacher recollection or private Git
credentials at runtime. Notebooks and results persist in a named Docker volume;
restarting the container does not replace your edits. Reference data is installed
read-only for the unprivileged container user. Stop with Ctrl-C. Keep the volume to retain your work. The launcher needs only
Docker. Registry images and a Compose launcher are maintained in the private
[Docker distribution repository](https://github.com/Krsma/ZONO2-docker).

Linux AMD64 and Apple Silicon use corresponding Linux container variants.
Apple Silicon support is through Docker Desktop, not a native macOS install.
The first release uses CPU training and inference. Select the locally built
ARM64 or full-data image with `PZR_LEARNING_IMAGE=pzr-learning:arm64` or
`PZR_LEARNING_IMAGE=pzr-learning:course` before the same launch command. `PZR_LEARNING_PORT` changes the localhost port,
and `PZR_LEARNING_VOLUME` selects a separate workspace.

For a local Python environment, install `pip install -e '.[learning]'` and run
`jupyter lab notebooks`. Offline training works without RTLola. Monitor and Gym
notebooks additionally require the repository's pinned release binding; use
`tools/setup_robot_arm_env.sh` on supported Linux hosts and its documented
OpenBLAS environment variables. Check the complete environment with:

```bash
python -m pzr.learning.doctor
```

## Train a custom selector

```python
from torch import nn
from pzr import learning

task = learning.load_task(bundle="starter", budget=40)

def my_network(input_dim, num_actions):
    return nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                         nn.Linear(64, num_actions))

run = learning.train_selector(
    task, model_factory=my_network,
    config=learning.TrainingConfig(epochs=20, seed=42),
)
print(learning.evaluate_selector(run.policy, task.validation))
run.save("workspace/my-model")
policy = learning.load_selector("workspace/my-model", model_factory=my_network)
```

The factory runs after seeding. It returns a CPU PyTorch module with normalized
float32 input `[batch, 15]` and finite output `[batch, 4]`. Scores are
**lower-is-better**, in `girard, scott, pca, combastel` order. They are not
calibrated costs or probabilities. Omit the factory for the existing 32–32 ReLU
baseline. Compare networks using the same data, seed, and training configuration.

Training uses the existing tolerance-aware, state-balanced pairwise objective,
AdamW, and best validation-loss checkpoint. Normalization uses training rows
only. The library restores the caller's RNG and deterministic settings afterward.
Saved custom weights require the same factory on load; model code is never
pickled. Save into a new directory to preserve previous experiments.

For a custom training loop, `prepared_samples(dataset, normalizer)` exposes
normalized features, float64 teacher costs, and boolean feasibility masks;
`cost_sensitive_pairwise_loss(scores, costs, feasible)` is the same loss used
by the trainer. Never fit preprocessing on validation data.

## What the recorded data means

The starter has 2,982 pre-event decision states at budget 40: training seeds 0–3
and validation seeds 20–21, with all recorded decision rows and six 500-event
traces. Labels are two-event full-width terminal teacher costs, not measured
long-term returns. Infeasible costs use NaN with an explicit false mask.

The frozen reference model was trained on **Clean148**, a larger training set;
it is not a controlled comparison against a starter-trained network. For fair
architecture comparisons, retrain both models on the same bundle.

The full `clean148` bundle has 531,146 rows over seven budgets, 148 training
trajectories, and six validation trajectories. Each task selects one budget;
there is no automatic cross-budget or joint-budget training. Existing seed roles
are preserved. Default notebooks use validation for iteration and do not open
protected paper test suites. Report these as development results.

To use a separately supplied full bundle, place `clean148/` under
`data/learning/` or set `PZR_LEARNING_DATA` to its parent, then:

```python
task = learning.load_task(bundle="clean148", budget=120)
```

You can also pass a bundle directory directly. Every bundle is hash-checked
before use. Bundle export is a maintainer operation described below; loading
never generates traces or teacher labels.

## Monitor and Gym semantics

`evaluate_monitor(policy, task.validation_traces)` runs complete recorded traces
through the native engine and reports separate final/summed approximation loss,
step-based trigger FPR/FNR, executed-action counts, fallbacks, and timing.
`max_events=20` explicitly requests a prefix diagnostic, not a full-trajectory
result. Timings are development diagnostics, not paper-facing measurements.
Pass `reference_cache="workspace/references"` to share exact references across
methods. A native failure raises an error rather than reporting a successful run.

Only native `ZonotopeConfig` transforms change monitor state. Under-bound states
execute `none`; unavailable ranked candidates are skipped before the native
`interval` fallback. Budgets retain native transform semantics: post-event dense
generator counts are not a bound violation. Constant calibration slack remains
under the native contract.

`MonitorSession.observe(geometry=True)` additionally returns immutable dynamic
and total zonotopes, with `center[d]` and `generators[d,m]`. These are inspection
exports, not a matrix-writeback API. Signed coefficients and zero rows remain
represented. Compact dimension, logical dimension, and generator counts differ;
row/column identities must not be assumed stable. The default neural model still
receives only Geometry15, with no stream values, history, or preview.

```python
from pzr.learning.gym import ReducerEnv

env = ReducerEnv(task.training_traces)
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(0)
```

Gym uses `Discrete(4)`, one recorded event per step, and observations containing
raw Geometry15 by default, a conservative action mask, and a reduction-needed flag. The
mask excludes known bound failures; it cannot guarantee a native transform will
succeed. If all entries are zero, the execution layer still accepts a request
and attempts the native fallback. A failed request tries the remaining actions
in catalog order. `info` records requested/executed actions and overrides.

Default reward is negative terminal native loss at trace completion and zero
otherwise. `objective="cumulative"` returns negative per-event loss instead.
An external `max_steps` cutoff sets truncation and does not fabricate a terminal
reward. Exact references and teacher costs never appear in observations or info.
An RL training algorithm may use rewards as usual. Reset with
`options={"trace_index": 0}` to select a particular supplied trace.

### Customize Gym inputs and embeddings

`ReducerEnv(traces, feature_encoder=encoder)` accepts a public
`learning.FeatureEncoder(name, shape, function, requires_geometry=False,
feature_names=())`. The function receives a pre-event `Observation` and returns
a finite float32-compatible tensor with the declared fixed shape. Set
`requires_geometry=True` to inspect dynamic and total zonotopes. Empty reset
states and variable geometry sizes need explicit handling. The encoder never
receives the monitor, recorded future inputs, teacher costs, or exact references.

The default `learning.GEOMETRY15_ENCODER` returns the unchanged raw paper
features. Its `feature_names` describe each entry; the underlying computation is
`pzr.rtlola.features.ranking_features_from_matrices`. Geometry15's log transform
and fitted normalization belong to the supervised selector, not the Gym adapter.
`Observation.features` always retains its Geometry15 meaning.

Notebook 03 introduces a custom zonotope-property function and a learned
embedding module. [Notebook 04](../../notebooks/04_define_zonotope_features.ipynb)
walks through selecting, ordering, and transforming properties, inspecting the
resulting Gym tensors, and reusing the encoder in native policy evaluation. Keep trainable embeddings inside the policy so gradients flow
through them; a Gym encoder may call a frozen embedding, but the NumPy observation
boundary cannot propagate gradients. Tensor shapes may have multiple axes, for
example for padded graph inputs and masks. Never silently truncate geometry to
fit a capacity. Save encoder code/version, preprocessing and model weights with
your experiment; custom encoders are not serialized by `TrainingRun.save`.

For native evaluation, wrap the same encoder and a single-input scoring function
with `learning.ObservationPolicy(encoder, score_function)` and pass it to
`learning.evaluate_monitor`. Scores must be four finite lower-is-better values in
`girard, scott, pca, combastel` order. Native bounds and fallbacks remain in force.

Recorded starter/Clean148 datasets contain Geometry15 only. `train_selector`
continues to train on those features; an embedding over them can be included in
`model_factory`. Supervised training with new geometry-based inputs requires
capturing geometry at the corresponding teacher decision states. Replaying a
different policy cannot recover those states or justify reusing their labels.

## Build and distribute the environment

Maintainers build from the existing private checkout; researchers consume the
finished image. Keep source/license access rules in mind when distributing it.
Build definitions remain here; the private [Docker distribution repository](https://github.com/Krsma/ZONO2-docker)
contains the validated publishing script and access instructions for GHCR.

```bash
# Uses the pinned checkout and dependencies already in Cargo's local cache.
python tools/learning/prepare_native.py

docker build -f docker/learning/Dockerfile --target runtime \
  -t pzr-learning:local .
```

The preparation script verifies the binding and interpreter pins and vendors
locked dependencies without fetching or changing a revision. Cargo builds
inside the container are offline. If the maintainer cache is incomplete, prepare
it through the normal authorized binding setup first. Vendored source stays in
ignored `build/learning-native`; credentials and Git history are not copied.
Python package versions are recorded in `docker/learning/requirements.lock`;
PyTorch is installed separately from its CPU wheel index.

Build ARM64 with `--platform linux/arm64 -t pzr-learning:arm64`. An AMD64 host needs
a builder with ARM64 support. Emulated native compilation is slower; an ARM64
builder is preferable. Supply the architecture-appropriate image under the
Compose image name, or distribute a multi-platform manifest via your chosen
registry. For offline image transfer use `docker save` / `docker load`.

Rebuild recorded bundles only when intentionally making a new version:

```bash
python tools/learning/export_bundle.py --output /tmp/starter-v1
python tools/learning/export_bundle.py --full --output data/learning/clean148
```

The exporter refuses existing output directories and verifies original datasets,
traces, and frozen model hashes. It does not modify `results/`. The full bundle
can be archived and mounted read-only, or included with the Docker `course`
target. In a runtime image set `PZR_LEARNING_DATA=/data` and mount its parent at
`/data:ro`.

Validate an image:

```bash
docker run --rm pzr-learning:local python -m pzr.learning.doctor
docker run --rm pzr-learning:local python -m pytest /opt/pzr/tests/test_learning_interface.py -q -o cache_dir=/tmp/pytest-cache
docker run --rm pzr-learning:local python /opt/pzr/check_notebooks.py --notebooks /opt/pzr/notebooks
```

Repository release validation still uses the ordinary full binding-backed pytest
suite with no skips. Historical artifacts remain untouched; new source changes
produce new source-aware cache identities rather than reusing old identities.

See [validation results](validation.md) for the tested images and checks, and
[follow-up interfaces](follow_up.md) for representation learning, prediction,
and certification work deferred from this release.

## Naming migration

The public interface is now `pzr.learning`; the former `pzr.student` import has
been removed. Use `pzr.learning.gym` for Gym and `.[learning]` for the optional
dependencies. Tooling and environment variables use `learning` and
`PZR_LEARNING_*`, respectively. New selector checkpoints use `learning.json`
and the `pzr.learning-selector.v1` schema; old custom checkpoints need their
metadata filename and schema updated before loading. The weights are unchanged.
Recorded bundle manifests likewise use `pzr.learning-bundle.v1`; the bundled
starter and local full bundle have been migrated without changing labels.

Existing notebook volumes are not renamed or deleted. To open an earlier
workspace, set `PZR_LEARNING_VOLUME=pzr-student-workspace`; existing notebooks
keep their contents, so update their imports to `from pzr import learning`.
New volumes receive the updated notebook templates. Rebuild images from the
renamed Dockerfile rather than merely retagging an old image.
