# Development and reproducibility

## Repository map

| Path | Purpose |
| --- | --- |
| `src/pzr/rtlola/` | Specifications, traces, native binding adapter, search, benchmarking |
| `src/pzr/learning/` | Recorded tasks, ranking policies, features, training, Gym interface |
| `notebooks/` | Five runnable tutorials; edit these notebooks directly |
| `tests/` | Pure tests and binding-backed semantic contracts |
| `tools/learning/` | Bundle export, native build preparation, notebook checks, local launcher |
| `tools/` and `experiments/` | Versioned scientific evaluation and setup entrypoints |
| `docker/learning/` | Authoritative image build definition and dependency lock |

Results, full training datasets, paper drafts, local environments, and private
research notes are excluded from Git. The small starter dataset and authoritative
robot-arm traces are included. Images and user-facing container launch files
are distributed through [ZONO2-docker](https://github.com/Krsma/ZONO2-docker).

## Tests

For source-only development:

```bash
python -m pip install -e '.[learning,dev]'
python -m pytest
```

Native integration tests skip when the binding is unavailable. This is useful for
local development, but is not release validation. A release requires the pinned
native environment and **zero skips**:

```bash
LD_PRELOAD="$PWD/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so" \
PYTHONPATH=src external/miniconda3/envs/pzr-robot-arm/bin/python -m pytest
```

Execute the tutorials in a fresh workspace using the same native environment:

```bash
python tools/learning/check_notebooks.py
```

Notebook files are the single source of truth; keep committed outputs empty.
The former notebook generator duplicated all tutorial content and has been removed.

## Scientific entrypoints

| Workflow | Entrypoint |
| --- | --- |
| Short robot-arm benchmark | `tools/run_rtlola_robot_arm.sh` |
| Canonical paper pipeline | `tools/run_paper_evaluation.sh` |
| V4 scientific finalization | `tools/run_paper_evaluation_v4.sh` |
| Separate Raspberry Pi timing | `tools/run_paper_evaluation_v4_pi_timing_v1.sh` |
| DART rescue study | `tools/run_dart_rescue.sh` |

Versioned configurations live in `experiments/`. The default v4 `run` ends after
prediction ablation; workstation timing is an explicit diagnostic. Pi timing
uses a frozen, hash-verified parent and separate output directories.

Full paper reproduction needs the corresponding recorded data, frozen models,
and result prerequisites; these are not all in the public checkout. Start with
the bundled learning tutorials or short benchmark before running a full matrix.
Do not interpret a successful tutorial run as reproduction of paper results.

Older scaling and exploratory scripts remain because current pipelines import
or hash-pin them. Versioned configurations preserve historical experiment
contracts. Removing or moving them would change reproducibility rather than
merely simplify the directory listing. Generated results must not be edited.

## Changes and contributions

Use small functions and explicit data contracts. Changes to learning, search,
actions, or scenarios need native semantic tests, including outer bounds,
constant-generator preservation, candidate alignment, deterministic branching,
and fallback accounting. Python selectors may inspect geometry and choose
actions; native transforms alone may mutate monitor state.

Keep the binding submodule at its pinned revision and build it in release mode.
Document any intended scientific-contract change separately from refactoring.
Open issues and pull requests in the source repository; include the command,
configuration, and package/native revisions when reporting failures.

## Public release scope

The project license covers original PZR source; third-party assets retain their
own notices. The native submodule still has restricted source access, and the
Docker distribution is intentionally private. This checkout can support public
offline learning without claiming that every dependency or paper artifact is
publicly downloadable. Making the GitHub source repository public is a separate
repository-settings action, not performed by these cleanup changes.

The current tracked tree excludes private research notes and paper drafts, but
older Git commits contain them. `.gitignore` does not remove historical content.
Before making the existing GitHub repository public, use a reviewed history
rewrite or publish the current tree into a new repository with fresh history.
Neither action is implied by a normal source push. For sharing the current tree
without its Git history:

```bash
git archive --format=tar.gz --prefix=pzr/ HEAD > /tmp/pzr-public-source.tar.gz
```

This exports committed files only; private submodule contents, ignored results,
credentials, and local uncommitted experiments are not included. Native source
access requirements still apply. An archive of the current tree is not a review
of the entire historical repository.
