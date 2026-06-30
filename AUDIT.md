# Architecture Audit

## Current State

- Experiment execution is exclusively RTLola-backed.
- Monitor semantics live in packaged `.lola` specifications.
- Reducer application is exclusively through native binding transforms.
- Python owns traces, state branching, search, metrics, reporting, and
  scenario-neutral regret ranking.
- Robot-arm MuJoCo files are retained only for trace/FK validation.

## Protected Invariants

- Exact transform-bound semantics; no fresh-reserve subtraction.
- Separate dense, active, zero, and constant generator accounting.
- Binding-native terminal approximation-loss objective.
- No no-op or fallback action in MPC/learning candidate catalogs.
- Constant robot-arm calibration columns preserved across dynamic transforms.
- RTLola public outputs are the trigger source of truth.

## Known External Risk

The pinned binding’s clustering dependency currently needs
`RUSTC_BOOTSTRAP=kmeans` under stable Rust. This is scoped in setup tooling and
must be removed when upstream fixes the dependency.
