# CoRL Overnight Runbook

## safe-control-gym Setup

The CoRL headline suite can run the safe-control-gym IROS task through a
sidecar Python environment. This avoids forcing the main PZR Python 3.11+
environment to carry the older simulator dependency stack.

The local checkout used for validation is:

```bash
external/miniconda3/bin/conda --version
# conda 26.3.2

external/miniconda3/envs/pzr-safe-control-fw/bin/python --version
# Python 3.8.x
```

To recreate it from scratch:

```bash
mkdir -p /tmp/pzr-downloads external/conda-home
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -o /tmp/pzr-downloads/Miniconda3-latest-Linux-x86_64.sh

HOME=/home/vlkr/Faks/phd/ZONO2/external/conda-home \
bash /tmp/pzr-downloads/Miniconda3-latest-Linux-x86_64.sh \
  -b -p /home/vlkr/Faks/phd/ZONO2/external/miniconda3

HOME=/home/vlkr/Faks/phd/ZONO2/external/conda-home \
external/miniconda3/bin/conda create -y -n pzr-safe-control-fw \
  -c conda-forge python=3.8 pip gmp compilers swig
```

Install the competition branch and the minimal runtime packages used by the
sidecar:

```bash
git clone --branch beta-iros-competition \
  https://github.com/learnsyslab/safe-control-gym.git \
  external/safe-control-gym

HOME=/home/vlkr/Faks/phd/ZONO2/external/conda-home \
external/miniconda3/envs/pzr-safe-control-fw/bin/python -m pip install \
  --upgrade "pip<25.1" "setuptools<75.4" wheel poetry-core \
  numpy==1.24.4 scipy==1.10.1 PyYAML munch matplotlib==3.7.5 \
  Pillow imageio dict-deep pandas==2.0.3 scikit-optimize termcolor \
  rich casadi pybullet gym==0.23.1

HOME=/home/vlkr/Faks/phd/ZONO2/external/conda-home \
external/miniconda3/envs/pzr-safe-control-fw/bin/python -m pip install \
  -e external/safe-control-gym --no-deps --no-build-isolation
```

Install and build the official Crazyflie firmware wrapper in the same sidecar
environment. The wrapper build must see the conda `swig` first on `PATH`.

```bash
git clone https://github.com/utiasDSL/pycffirmware.git external/pycffirmware
git -C external/pycffirmware submodule update --init --recursive

PATH=/home/vlkr/Faks/phd/ZONO2/external/miniconda3/envs/pzr-safe-control-fw/bin:$PATH \
HOME=/home/vlkr/Faks/phd/ZONO2/external/conda-home \
bash external/pycffirmware/wrapper/build_linux.sh
```

On modern GCC, the upstream wrapper may require these local compatibility
flags in `external/pycffirmware/wrapper/setup.py`:
`-Wno-error=implicit-function-declaration` and
`-Wno-error=int-conversion`.

The main PZR environment still needs the project package plus learning/test
dependencies:

```bash
python -m pip install -e ".[dev,learning]"
```

Use these paths for the current sidecar setup:

```bash
export PZR_SAFE_CONTROL_GYM_ROOT=/home/vlkr/Faks/phd/ZONO2/external/safe-control-gym
export PZR_SAFE_CONTROL_PYTHON=/home/vlkr/Faks/phd/ZONO2/external/miniconda3/envs/pzr-safe-control-fw/bin/python
export PZR_SAFE_CONTROL_CONFIG=competition/level1.yaml
```

## Preflight

Run sidecar preflight before starting an overnight job:

```bash
pzr-run-corl \
  --preflight \
  --safe-control-gym-root "$PZR_SAFE_CONTROL_GYM_ROOT" \
  --safe-control-python "$PZR_SAFE_CONTROL_PYTHON" \
  --safe-control-config "$PZR_SAFE_CONTROL_CONFIG"
```

Expected sidecar checks are `safe_control_python_exists`,
`safe_control_gym_root_exists`, `pycffirmware_available`,
`firmware_wrapper_available`, `firmware_reset`, `firmware_step`,
`sidecar_reset`, and `sidecar_step`. These should be true before running a
headline job. `torch` is reported as an availability diagnostic and is required
only for learned-policy training or checkpoint evaluation. The old no-firmware PID path is diagnostic only and
requires both `--safe-control-controller-mode debug_pid` and
`--allow-debug-pid`; do not use it for headline CoRL evidence.

## Smoke Test

```bash
pzr-run-corl \
  --profile smoke \
  --safe-control-gym-root "$PZR_SAFE_CONTROL_GYM_ROOT" \
  --safe-control-python "$PZR_SAFE_CONTROL_PYTHON" \
  --safe-control-config "$PZR_SAFE_CONTROL_CONFIG" \
  --method-set core \
  --learned-mode none \
  --out /tmp/pzr-corl-real-smoke \
  --force \
  --no-archive
```

Expected core outputs:

- `raw_episodes.csv`
- `intervention_timeseries.csv`
- `monitor_timeseries.csv`
- `decision_features.csv`
- `failure_events.csv`
- `selection_summary.csv`
- `predicted_sequence_summary.csv`
- `headline_table.csv`
- `headline_table.md`
- `headline_quality.md`
- `analysis_notes.json`

`dagger_dataset.csv` and learned-policy checkpoints are written only when
`--learned-mode dagger` is used. Learned rows should be treated as headline
evidence only when `learning/dagger_label_summary.csv` passes the label
diversity gate recorded in `analysis_notes.json`.

## Calibration Sweep

Before treating a safe-control-gym level as a headline setting, run the compact
calibration path:

```bash
tools/run_corl_calibration.sh
```

or directly:

```bash
pzr-run-corl \
  --profile overnight \
  --calibration \
  --safe-control-gym-root "$PZR_SAFE_CONTROL_GYM_ROOT" \
  --safe-control-python "$PZR_SAFE_CONTROL_PYTHON" \
  --out results/corl-calibration \
  --force \
  --no-archive
```

Inspect `calibration_summary.csv`, `calibration_recommendations.json`, and
`failure_events.csv`. Avoid using regimes where nominal control fails, all
bounded methods have saturated fallback duration, missed violations are
nonzero, or reducer/soundness failures are recorded.

## Overnight Command

```bash
mkdir -p results/logs
export PZR_SAFE_CONTROL_GYM_ROOT=/home/vlkr/Faks/phd/ZONO2/external/safe-control-gym
export PZR_SAFE_CONTROL_PYTHON=/home/vlkr/Faks/phd/ZONO2/external/miniconda3/envs/pzr-safe-control-fw/bin/python

nohup pzr-run-corl \
  --profile overnight \
  --safe-control-gym-root "$PZR_SAFE_CONTROL_GYM_ROOT" \
  --safe-control-python "$PZR_SAFE_CONTROL_PYTHON" \
  --safe-control-config competition/level1.yaml \
  --out results/corl-main-$(date +%Y%m%d) \
  --force \
  --method-set core \
  --learned-mode none \
  --budget 8 \
  --horizon 6 \
  --max-steps 1000 \
  --train-seeds 20 \
  --eval-seeds 50 \
  --dagger-iterations 3 \
  --dagger-expert mpc_wide_fixed_girard \
  --bootstrap-samples 5000 \
  --fail-on-unusable \
  > results/logs/corl-main-$(date +%Y%m%d).log 2>&1 &
```

For the learned-policy ablation, add `--learned-mode dagger`. The learned row is
omitted from the headline table unless the label-diversity gate passes, unless
`--include-failed-learned` is explicitly supplied for diagnostics.

## Morning Inspection

Start with:

```bash
tail -f results/corl-main-*/progress.jsonl
cat results/corl-main-*/analysis_notes.json
column -s, -t < results/corl-main-*/headline_table.csv | less -S
cat results/corl-main-*/headline_quality.md
```

Then inspect `intervention_timeseries.csv` for spurious, justified, and missed
interventions, and `monitor_timeseries.csv` for reducer latency, budget
violations, unsound certificates, and sequence-search behavior.
Check `failure_events.csv`; it should be empty for headline evidence.
Check `learning/dagger_label_summary.csv` before treating the learned selector
as paper evidence; the label-diversity gate should pass rather than collapse to
one reducer.
Treat `headline_quality.md` or `analysis_notes.json` reporting
`paper_usable=false` as a failed headline run even if the process completed and
all CSV files were written.
The helper scripts do not overwrite existing output directories unless
`PZR_CORL_FORCE=1` is set.
