# Neural reducer experiments

Start with **00_start_here.ipynb**, then **01_build_your_selector.ipynb**.
The main workflow is supervised training on recorded teacher costs. You do not
need Gym, Rust knowledge, or a custom model base class to change the network.

## Launch Jupyter

With the supplied image loaded as `pzr-student:local`, run from the repository:

```bash
tools/student/start_lab.sh
```

Open the local URL printed by Jupyter (including its token). The image includes
the dependencies, pinned release binding, six recorded trajectories, teacher
labels, and notebooks. It requires no teacher recollection or private Git
credentials at runtime. Notebooks and results persist in a named Docker volume;
restarting the container does not replace your edits. Reference data is installed
read-only for the unprivileged student user. Stop with Ctrl-C. Keep the volume to retain your work. The launcher needs only
Docker; a Compose alternative is provided in `docker/student/compose.yaml`.

Linux AMD64 and Apple Silicon use corresponding Linux container variants.
Apple Silicon support is through Docker Desktop, not a native macOS install.
The first release uses CPU training and inference. Select the locally built
ARM64 or full-data image with `PZR_STUDENT_IMAGE=pzr-student:arm64` or
`PZR_STUDENT_IMAGE=pzr-student:course` before the same launch command. `PZR_STUDENT_PORT` changes the localhost port,
and `PZR_STUDENT_VOLUME` selects a separate workspace.

For a local Python environment, install `pip install -e '.[student]'` and run
`jupyter lab notebooks`. Offline training works without RTLola. Monitor and Gym
notebooks additionally require the repository's pinned release binding; use
`tools/setup_robot_arm_env.sh` on supported Linux hosts and its documented
OpenBLAS environment variables. Check the complete environment with:

```bash
python -m pzr.student.doctor
```

## Your first model

```python
from torch import nn
from pzr import student

task = student.load_task(bundle="starter", budget=40)

def my_network(input_dim, num_actions):
    return nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                         nn.Linear(64, num_actions))

run = student.train_selector(
    task, model_factory=my_network,
    config=student.TrainingConfig(epochs=20, seed=42),
)
print(student.evaluate_selector(run.policy, task.validation))
run.save("workspace/my-model")
policy = student.load_selector("workspace/my-model", model_factory=my_network)
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
`data/student/` or set `PZR_STUDENT_DATA` to its parent, then:

```python
task = student.load_task(bundle="clean148", budget=120)
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
from pzr.student.gym import ReducerEnv

env = ReducerEnv(task.training_traces)
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(0)
```

Gym uses `Discrete(4)`, one recorded event per step, and observations containing
raw Geometry15, a conservative action mask, and a reduction-needed flag. The
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

## Build and distribute the environment

Maintainers build from the existing private checkout; students consume the
finished image. Keep source/license access rules in mind when distributing it.
No registry publication is performed by these scripts.

```bash
# Uses the pinned checkout and dependencies already in Cargo's local cache.
python tools/student/prepare_native.py

docker build -f docker/student/Dockerfile --target runtime \
  -t pzr-student:local .
```

The preparation script verifies the binding and interpreter pins and vendors
locked dependencies without fetching or changing a revision. Cargo builds
inside the container are offline. If the maintainer cache is incomplete, prepare
it through the normal authorized binding setup first. Vendored source stays in
ignored `build/student-native`; credentials and Git history are not copied.
Python package versions are recorded in `docker/student/requirements.lock`;
PyTorch is installed separately from its CPU wheel index.

Build ARM64 with `--platform linux/arm64 -t pzr-student:arm64`. An AMD64 host needs
a builder with ARM64 support. Emulated native compilation is slower; an ARM64
builder is preferable. Supply the architecture-appropriate image under the
Compose image name, or distribute a multi-platform manifest via your chosen
registry. For offline image transfer use `docker save` / `docker load`.

Rebuild recorded bundles only when intentionally making a new version:

```bash
python tools/student/export_bundle.py --output /tmp/starter-v1
python tools/student/export_bundle.py --full --output data/student/clean148
```

The exporter refuses existing output directories and verifies original datasets,
traces, and frozen model hashes. It does not modify `results/`. The full bundle
can be archived and mounted read-only, or included with the Docker `course`
target. In a runtime image set `PZR_STUDENT_DATA=/data` and mount its parent at
`/data:ro`.

Validate an image:

```bash
docker run --rm pzr-student:local python -m pzr.student.doctor
docker run --rm pzr-student:local python -m pytest /opt/pzr/tests/test_student.py -q -o cache_dir=/tmp/pytest-cache
docker run --rm pzr-student:local python /opt/pzr/check_notebooks.py --notebooks /opt/pzr/notebooks
```

Repository release validation still uses the ordinary full binding-backed pytest
suite with no skips. Historical artifacts remain untouched; new source changes
produce new source-aware cache identities rather than reusing old identities.

See [validation results](validation.md) for the tested images and checks, and
[follow-up interfaces](follow_up.md) for representation learning, prediction,
and certification work deferred from this release.
