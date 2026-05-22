# CoRL Overnight Runbook

## safe-control-gym Setup

The CoRL headline suite can run the safe-control-gym IROS task through a
sidecar Python environment. This avoids forcing the main PZR Python 3.11+
environment to carry the older simulator dependency stack.

The local checkout used for validation is:

```bash
external/miniconda3/bin/conda --version
# conda 26.3.2

external/miniconda3/envs/pzr-safe-control/bin/python --version
# Python 3.10.20
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
external/miniconda3/bin/conda create -y -n pzr-safe-control \
  -c conda-forge python=3.10 pip gmp compilers swig
```

Install the competition branch and the minimal runtime packages used by the
sidecar:

```bash
git clone --branch beta-iros-competition \
  https://github.com/learnsyslab/safe-control-gym.git \
  external/safe-control-gym

HOME=/home/vlkr/Faks/phd/ZONO2/external/conda-home \
external/miniconda3/envs/pzr-safe-control/bin/python -m pip install \
  --upgrade pip setuptools wheel poetry-core numpy==1.26.4 scipy PyYAML munch \
  matplotlib Pillow imageio dict-deep pandas scikit-optimize termcolor rich \
  casadi pybullet gym==0.23.1

HOME=/home/vlkr/Faks/phd/ZONO2/external/conda-home \
external/miniconda3/envs/pzr-safe-control/bin/python -m pip install \
  -e external/safe-control-gym --no-deps --no-build-isolation
```

The main PZR environment still needs the project package plus learning/test
dependencies:

```bash
python -m pip install -e ".[dev,learning]"
```

Use these paths for the current sidecar setup:

```bash
export PZR_SAFE_CONTROL_GYM_ROOT=/home/vlkr/Faks/phd/ZONO2/external/safe-control-gym
export PZR_SAFE_CONTROL_PYTHON=/home/vlkr/Faks/phd/ZONO2/external/miniconda3/envs/pzr-safe-control/bin/python
export PZR_SAFE_CONTROL_CONFIG=competition/level3.yaml
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
`safe_control_gym_root_exists`, `sidecar_reset`, `sidecar_step`, and `torch`.
All should be true before running a headline job.

## Smoke Test

```bash
pzr-run-corl \
  --profile smoke \
  --safe-control-gym-root "$PZR_SAFE_CONTROL_GYM_ROOT" \
  --safe-control-python "$PZR_SAFE_CONTROL_PYTHON" \
  --safe-control-config "$PZR_SAFE_CONTROL_CONFIG" \
  --out /tmp/pzr-corl-real-smoke \
  --force \
  --no-archive
```

Expected core outputs:

- `raw_episodes.csv`
- `intervention_timeseries.csv`
- `monitor_timeseries.csv`
- `decision_features.csv`
- `dagger_dataset.csv`
- `selection_summary.csv`
- `predicted_sequence_summary.csv`
- `headline_table.csv`
- `headline_table.md`
- `analysis_notes.json`

## Overnight Command

```bash
mkdir -p results/logs
export PZR_SAFE_CONTROL_GYM_ROOT=/home/vlkr/Faks/phd/ZONO2/external/safe-control-gym
export PZR_SAFE_CONTROL_PYTHON=/home/vlkr/Faks/phd/ZONO2/external/miniconda3/envs/pzr-safe-control/bin/python

nohup pzr-run-corl \
  --profile overnight \
  --safe-control-gym-root "$PZR_SAFE_CONTROL_GYM_ROOT" \
  --safe-control-python "$PZR_SAFE_CONTROL_PYTHON" \
  --safe-control-config competition/level3.yaml \
  --out results/corl-main-$(date +%Y%m%d) \
  --force \
  --budget 8 \
  --horizon 6 \
  --max-steps 1000 \
  --train-seeds 20 \
  --eval-seeds 50 \
  --dagger-iterations 3 \
  --bootstrap-samples 5000 \
  > results/logs/corl-main-$(date +%Y%m%d).log 2>&1 &
```

## Morning Inspection

Start with:

```bash
cat results/corl-main-*/analysis_notes.json
column -s, -t < results/corl-main-*/headline_table.csv | less -S
```

Then inspect `intervention_timeseries.csv` for spurious, justified, and missed
interventions, and `monitor_timeseries.csv` for reducer latency, budget
violations, unsound certificates, and sequence-search behavior.
