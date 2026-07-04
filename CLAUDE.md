# CLAUDE.md

This repository uses `AGENTS.md` as the primary operational guide for coding
agents. Read it first.

## Current Pointers

- `README.md`: concise project layout, install commands, and active CLIs.
- `AGENTS.md`: detailed conventions, soundness boundary, benchmark commands,
  trigger contract, and artifact rules.
- `AUDIT.md`: current project-readiness audit and validation gaps.
- `science/SCIENCE.md`: compact research notes and proof obligations.
- `science/EXPERIMENT_READINESS.md`: active experiment readiness and scenario
  decision notes.
- `science/RTLOLA_INTEGRATION_NOTES.md`: current RTLola binding API and PZR
  integration.

## Non-Negotiable Invariants

- Only RTLola binding transforms may mutate monitor state.
- MPC and teacher costs use binding-native terminal approximation loss.
- Robot-arm trigger booleans normalize sparse `Trigger#N` verdict entries.
- Do not hand-edit generated files in `results/`.
- Do not rely on stale historical commands such as `pzr-run-corl`,
  `pzr-run-experiments`, or `pzr-paper-figures`; they are not declared current
  project entry points.
