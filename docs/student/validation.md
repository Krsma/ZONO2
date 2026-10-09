# Student release validation

Validated on the implementation working tree with PZR source SHA-256
`061fe7080ceca3c6397b30f187d39f36610923de36944bc636ca88fe3b400930`.

| Check | Result |
|---|---|
| Full repository suite, release binding environment | 266 passed, zero skips |
| Student integration tests, AMD64 container | 12 passed, zero skips |
| Student integration tests, ARM64 container under emulation | 12 passed, zero skips |
| Four onboarding notebooks, AMD64 container | All executed successfully |
| Four onboarding notebooks, ARM64 container under emulation | All executed successfully |
| Offline neural training with binding imports blocked | Passed |
| Default student trainer versus existing trainer | Identical validation predictions on fixed-seed test |
| Custom network, including single-state BatchNorm inference | Training and persistence checks passed |
| Gymnasium environment checker and terminal/truncated rewards | Passed |
| Plain-Docker launcher and Jupyter login endpoint | Passed |
| Persistent notebook edits across fresh containers | Passed |
| Installed reference data read-only for student user | Passed |
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

Locally built image tags are `pzr-student:local` (AMD64), `pzr-student:arm64`,
and `pzr-student:course` (AMD64 with full data). They have not been published to
a registry. The standalone full bundle is `data/student/clean148.tar.gz`, about
68 MiB, with SHA-256
`ec6b9388d4a7e339942bf2e6726aeb0f76b502422deea54e84cbdd4cfb14e279`.
The starter's six recorded trajectories and 2,982 labelled states are packaged
with the library; the full bundle has 531,146 labelled states.

Representative validation commands:

```bash
LD_PRELOAD="$PWD/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONPATH=src \
  external/miniconda3/envs/pzr-robot-arm/bin/python -m pytest -q

docker run --rm pzr-student:local python -m pytest \
  /opt/pzr/tests/test_student.py -q -o cache_dir=/tmp/pytest-cache

docker run --rm --platform linux/arm64 pzr-student:arm64 python -m pytest \
  /opt/pzr/tests/test_student.py -q -o cache_dir=/tmp/pytest-cache

docker run --rm pzr-student:local python /opt/pzr/check_notebooks.py \
  --notebooks /opt/pzr/notebooks

docker run --rm --platform linux/arm64 pzr-student:arm64 \
  python /opt/pzr/check_notebooks.py --notebooks /opt/pzr/notebooks
```

Existing `results/` artifacts and native sources were not modified. The paper
Pi timing source-hash guard remains unchanged and correctly rejects this newer
source tree; its unit test now checks both pinned-source acceptance and
changed-source rejection. Running that frozen study still requires its frozen
source checkout. No paper experiment semantics or objectives were changed.
