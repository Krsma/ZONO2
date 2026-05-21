# CoRL Overnight Runbook

## safe-control-gym Setup

The CoRL headline suite expects the safe-control-gym IROS task to be installed
outside this repository. The upstream project documents Python 3.10 setup:

```bash
git clone https://github.com/utiasDSL/safe-control-gym.git external/safe-control-gym
cd external/safe-control-gym
git checkout beta-iros-competition

conda create -n pzr-safe-control python=3.10 -y
conda activate pzr-safe-control
python -m pip install --upgrade pip
python -m pip install -e .
```

This repository requires Python 3.11 or newer. If safe-control-gym imports in
the same environment, install this package there too:

```bash
cd /home/vlkr/Faks/phd/ZONO2
python -m pip install -e ".[dev,learning]"
export PZR_SAFE_CONTROL_GYM_ROOT=/home/vlkr/Faks/phd/ZONO2/external/safe-control-gym
```

If dependency versions conflict, use a sidecar Python executable from the
safe-control-gym environment:

```bash
export PZR_SAFE_CONTROL_GYM_ROOT=/home/vlkr/Faks/phd/ZONO2/external/safe-control-gym
export PZR_SAFE_CONTROL_PYTHON=/path/to/pzr-safe-control/bin/python
```

## Preflight

Run preflight before starting an overnight job:

```bash
pzr-run-corl --preflight --safe-control-gym-root "$PZR_SAFE_CONTROL_GYM_ROOT"
```

For the sidecar path:

```bash
pzr-run-corl \
  --preflight \
  --safe-control-gym-root "$PZR_SAFE_CONTROL_GYM_ROOT" \
  --safe-control-python "$PZR_SAFE_CONTROL_PYTHON"
```

The current implementation has a deterministic fake environment for smoke tests
and a fail-fast safe-control-gym boundary. If the concrete beta IROS task factory
is not detected, preflight exits nonzero with setup details rather than writing
headline artifacts from the wrong simulator.

## Smoke Test

```bash
pzr-run-corl --profile smoke --out /tmp/pzr-corl-smoke --force --no-archive
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

PZR_SAFE_CONTROL_GYM_ROOT=/home/vlkr/Faks/phd/ZONO2/external/safe-control-gym \
nohup pzr-run-corl \
  --profile overnight \
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

With sidecar Python:

```bash
PZR_SAFE_CONTROL_GYM_ROOT=/home/vlkr/Faks/phd/ZONO2/external/safe-control-gym \
nohup pzr-run-corl \
  --profile overnight \
  --safe-control-python /path/to/pzr-safe-control/bin/python \
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
