# Raspberry Pi 5 Timing Add-On

This add-on treats paper evaluation v4 as a read-only parent. It does not
modify the v4 configuration, runner, policies, source tree, or results. The
canonical Pi result and combined report use new versioned roots.

## Experimental role and impact on v4

The Pi experiment replaces only the source of the primary deployment-timing
claim. It does not replace or rerun the v4 scientific matrices. We derive
quality, soundness, availability, approximation loss, severe tails, reducer
composition, guard behavior, bounded-memory evidence, and predictor ablation
from the workstation v4 artifact. We derive latency distributions, empirical
on-time rate capacity, model footprint, process memory, and the diagnostic phase
breakdown from the Pi artifact.

This separation is intentional: It prevents a late platform port from changing
the monitor, reducers, policies, traces, or scientific results. The bundle pins
the v4 configuration, runner, scientific source, policy helpers, models, native
stack, and selected traces. Any parent change after the bundle contract is
frozen causes `status`, `bundle`, import, or combined reporting to abort; it
requires an explicit new add-on version or reviewed pin update, never a silent
refresh.

The default `tools/run_paper_evaluation_v4.sh run` command is safe for this
workflow: It stops after `prediction-ablation` and does not execute workstation
timing or its dependent report and validation stages. A local v4 runtime may be
requested explicitly as a clearly labelled portability diagnostic, but it is
not combined with the Pi samples and must not be quoted as the Pi result.

## 1. Finish and verify the scientific parent

Complete the v4 `prepare`, `nominal`, and `prediction-ablation` stages on the
workstation. The bundle command rejects incomplete matrices, nonzero false
negatives in completed nominal cells, stale v4 manifests, or any pinned source,
tool, model, or trace mismatch.

Run the complete workstation scientific sequence with:

```bash
tools/run_paper_evaluation_v4.sh run
```

Check the pins before transferring anything:

```bash
tools/run_paper_evaluation_v4_pi_timing_v1.sh status
```

## 2. Build smoke and full bundles

```bash
tools/run_paper_evaluation_v4_pi_timing_v1.sh bundle --smoke \
  --archive /tmp/pzr-pi-timing-smoke.tar.gz

tools/run_paper_evaluation_v4_pi_timing_v1.sh bundle \
  --archive /tmp/pzr-pi-timing-full.tar.gz
```

Copy both archives to the Pi and extract them into separate directories. The
smoke bundle contains one seed and one bound but all ten method types. A single
environment can be used for both bundles by setting `PZR_PI_TIMING_PYTHON`.

## 3. Set up the Pi

On Raspberry Pi OS Lite 64-bit, inside the extracted full bundle:

The bundled convenience launcher keeps the environment and every run output
outside the immutable bundles.  It is the recommended Pi-side sequence:

```bash
# In the extracted full bundle.  Run this as the regular Pi user (no outer
# sudo): it installs the pinned environment, requests the performance
# governor, then tells you to reboot.
tools/run_paper_evaluation_v4_pi_timing_v1_on_pi.sh setup "$PWD"

# After reboot, first pass the path to the separately extracted smoke bundle.
# The launcher reapplies the performance governor before preflight.
tools/run_paper_evaluation_v4_pi_timing_v1_on_pi.sh smoke "$PWD" \
  /absolute/path/to/extracted-smoke-bundle

# With the Pi idle, runs preflight, semantic contracts, all timing cells,
# phase profiling, and packing.  It prints the archive to copy back.
tools/run_paper_evaluation_v4_pi_timing_v1_on_pi.sh measure "$PWD"
```

Set `PZR_PI_TIMING_ENV_ROOT` or `PZR_PI_TIMING_RUNS_ROOT` before invoking the
launcher only if the default sibling locations are unsuitable.  The manual
commands below remain available for diagnosis or for running an individual
stage.

```bash
tools/setup_paper_evaluation_v4_pi_timing_v1.sh setup "$PWD"
sudo tools/setup_paper_evaluation_v4_pi_timing_v1.sh host-controls "$PWD"
```

The setup requires network access and working CISPA SSH credentials to build
the pinned binding. Reboot immediately before the measured run, reapply
`host-controls`, and ensure no unrelated workload is active.

Set the interpreter for subsequent commands:

```bash
export PZR_PI_TIMING_PYTHON=/absolute/path/to/pzr-pi-timing-env-v1/runtime/bin/python
```

## 4. Smoke-test and run

In the extracted smoke bundle:

```bash
PZR_PI_TIMING_CONFIG="$PWD/experiments/paper_evaluation_v4_pi_timing_v1.yaml" \
  /absolute/path/to/full-bundle/tools/run_paper_evaluation_v4_pi_timing_v1.sh \
  smoke --bundle "$PWD"
```

In the extracted full bundle:

```bash
tools/run_paper_evaluation_v4_pi_timing_v1.sh preflight --bundle "$PWD"
tools/run_paper_evaluation_v4_pi_timing_v1.sh contract-tests --bundle "$PWD"
RUN_OUTPUT=/absolute/path/to/pi-runs/full/paper-evaluation-v4-pi-timing-v1
tools/run_paper_evaluation_v4_pi_timing_v1.sh run --bundle "$PWD" \
  --run-output "$RUN_OUTPUT"
tools/run_paper_evaluation_v4_pi_timing_v1.sh profile --bundle "$PWD" \
  --run-output "$RUN_OUTPUT"
tools/run_paper_evaluation_v4_pi_timing_v1.sh pack --bundle "$PWD" \
  --run-output "$RUN_OUTPUT" \
  --archive /absolute/path/to/pi-runs/paper-evaluation-v4-pi-timing-v1-results.tar.gz
```

The environment and run output deliberately live outside the extracted bundle;
the bundle remains byte-for-byte immutable throughout setup and execution.

Cells are atomic and resumable. Native or infrastructure failures abort.
Expected reducer infeasibility is recorded as an unavailable cell. Packing is
refused until all ten phase-profile cells pass semantic parity.

## 5. Import and combine on the workstation

Copy the result archive back without extracting it manually:

```bash
tools/run_paper_evaluation_v4_pi_timing_v1.sh import-results \
  --archive /path/to/paper-evaluation-v4-pi-timing-v1-results.tar.gz

tools/run_paper_evaluation_v4_pi_timing_v1.sh combine-report \
  --pi-output results/paper-evaluation-v4-pi-timing-v1
```

These commands write only `results/paper-evaluation-v4-pi-timing-v1` and
`results/paper-evaluation-v4-final-report-v1`. They abort instead of
overwriting an existing root.

## Reporting contract

The imported timing result contains 350 method--seed--bound cells: ten methods,
five v4 seeds, and seven bounds. Each completed cell retains all 200 measured
event samples, p50/p90/p95/p99/maximum latency, MAD and IQR, empirical maximum
rates meeting 95%, 99%, and 100% on-time targets, model bytes, and process RSS
snapshots. The ten-cell phase profile reports predictor, selector, and native
commit costs separately and must reproduce the ordinary run's failure semantics
before the result can be packed.

The combined report joins Pi timing to the corresponding v4 nominal cells by
seed, bound, and method. It emits Pi timing summaries, latency--loss and
latency--FPR Pareto indicators, a phase summary, and machine-readable claims
whose hashes identify both parents. The join is analytical rather than
mutating: Neither the v4 directory nor the imported Pi directory is rewritten.

In the manuscript, state the platform next to every timing, memory, or rate
claim and cite the combined manifest. Do not pool workstation and Pi latency
samples. The primary latency is selection plus the live native commit; the
predictor phase remains a separately labelled diagnostic because it is outside
the inherited `decision_time_ms` boundary.
