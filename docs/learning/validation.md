# Learning interface validation

## Naming migration

The renamed `pzr.learning` interface passed the full release-binding suite:
**266 passed, zero skips**. Offline training also passed with native-binding
imports blocked, and the migrated full bundle loaded 76,538 budget-40 rows.
The source SHA-256 is
`1c446984f6b7678036bf236f361acb0f76de2bcbd0ac3235290f7b3d7b2c31ce`.

## Original release validation

The original release was validated before the naming migration, with PZR source SHA-256
`061fe7080ceca3c6397b30f187d39f36610923de36944bc636ca88fe3b400930`.

| Check | Result |
|---|---|
| Full repository suite, release binding environment | 266 passed, zero skips |
| Learning integration tests, AMD64 container | 12 passed, zero skips |
| Learning integration tests, ARM64 container under emulation | 12 passed, zero skips |
| Four onboarding notebooks, AMD64 container | All executed successfully |
| Four onboarding notebooks, ARM64 container under emulation | All executed successfully |
| Offline neural training with binding imports blocked | Passed |
| Default learning trainer versus existing trainer | Identical validation predictions on fixed-seed test |
| Custom network, including single-state BatchNorm inference | Training and persistence checks passed |
| Gymnasium environment checker and terminal/truncated rewards | Passed |
| Plain-Docker launcher and Jupyter login endpoint | Passed |
| Persistent notebook edits across fresh containers | Passed |
| Installed reference data read-only for container user | Passed |
| Course image full bundle load | 148 training and six validation trajectories |

A 12-event recorded validation trace (seed 20, budget 40, Girard requests) was
compared across AMD64 and ARM64: identical executed actions; native losses and
total-state radii agree at `rtol=1e-6, atol=1e-12`, and Geometry15 agrees at
`rtol=1e-5, atol=1e-7`. This is a numerical smoke, not a cross-platform performance
claim. ARM64 was tested through emulation on an AMD64 host, not physical Apple
hardware. A full 500-event validation run also completed without fallback.

Both images report Python 3.11.17, PyTorch 2.12.1+cpu, NumPy 2.4.6,
Gymnasium 1.4.0, and JupyterLab 4.6.4. The unchanged native pins are:

- Binding: `01c92a2bfac58755e3b832bb0094816f3f36e1d1`
- Interpreter: `2724b05ae6c62ed0df14f1401ed8db89472725a6`
- Build profile: release

The original local images use the `pzr-student` repository name and contain
the former API. Rebuilt images use `pzr-learning`; do not treat an old image
retag as validation of the renamed API. No images have been published to a registry. The standalone full bundle is `data/learning/clean148.tar.gz`, about
68 MiB, with SHA-256
`7a251bb51bb57d900855a7b8cb485fa1fb7eb677cd549609cc85d5376a67405f`.
The starter's six recorded trajectories and 2,982 labelled states are packaged
with the library; the full bundle has 531,146 labelled states.

Validation commands using the new names (the ARM64 image must first be rebuilt):

```bash
LD_PRELOAD="$PWD/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONPATH=src \
  external/miniconda3/envs/pzr-robot-arm/bin/python -m pytest -q

docker run --rm pzr-learning:local python -m pytest \
  /opt/pzr/tests/test_learning_interface.py -q -o cache_dir=/tmp/pytest-cache

docker run --rm --platform linux/arm64 pzr-learning:arm64 python -m pytest \
  /opt/pzr/tests/test_learning_interface.py -q -o cache_dir=/tmp/pytest-cache

docker run --rm pzr-learning:local python /opt/pzr/check_notebooks.py \
  --notebooks /opt/pzr/notebooks

docker run --rm --platform linux/arm64 pzr-learning:arm64 \
  python /opt/pzr/check_notebooks.py --notebooks /opt/pzr/notebooks
```

Existing `results/` artifacts and native sources were not modified. The paper
Pi timing source-hash guard remains unchanged and correctly rejects this newer
source tree; its unit test now checks both pinned-source acceptance and
changed-source rejection. Running that frozen study still requires its frozen
source checkout. No paper experiment semantics or objectives were changed.

The naming migration also executed all four notebooks successfully in the local
release-binding environment. The rebuilt AMD64 image passed all 12 interface
tests, and the full-data image loaded all 76,538 budget-40 rows. AMD64 runtime and full-data images are rebuilt as
`pzr-learning:local` and `pzr-learning:course`. The original ARM64 image retains
its old API; rebuild it from `docker/learning/Dockerfile` before distributing it
under the new name.
The additional ARM64 source-overlay rerun was stopped during the native tests
because of slow emulation; it is not counted as a completed migration check.
The original release's ARM64 results above remain historical.
