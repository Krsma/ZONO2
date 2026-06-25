# RTLola Learning Pipeline Audit

Status as of 2026-06-25.

## Current State

There are two learning pipelines in the repository.

The original Python-monitor pipeline in `src/pzr/experiments/regret_eval.py`
is integrated with the normal benchmark CLI and robotics replay. It trains a
regret/ranking policy from MPC candidate-cost tables, appends learned rows to
`summary.csv`, `timeseries.csv`, and `aggregate.csv`, and writes learning
artifacts under `learning/<scenario>/`.

The RTLola-native pipeline in `src/pzr/rtlola/learning.py` exists, but is not
ready for robot-arm headline comparisons. It currently supports only
`omni_robot`, uses all default RTLola actions as learning candidates, and
evaluates learned decisions through a ranked beam search rather than a cheap
top-1 learned reducer.

## Main Gaps

- `train_and_evaluate_regret` is hard-coded to `omni_robot`.
- Collection and evaluation construct `RtlolaEngine(OMNI_SPEC, ...)` directly
  instead of using `scenario_by_name(config.scenario)`.
- Candidate names are `tuple(action.name for action in default_actions())`,
  which includes actions that are not apples-to-apples MPC candidates:
  `none`, `interval`, `colinear`, `colinear_scale`, and slow static-only
  methods.
- `_evaluate_candidates` calls `beam_search` without opting into the
  binding-native reference-loss objective, so robot-arm learning would not
  train against the current RTLola MPC objective.
- Learned evaluation records `reduction_time_ms=0.0`, so runtime comparisons
  would be misleading.
- The learned policy is currently "ranked beam": it ranks candidates and then
  still calls `beam_search`. This is useful diagnostically, but it is not yet
  the fast learned selector story.

## Desired RTLola Behavior

- Use the same scenario metadata as normal RTLola benchmark runs.
- Use the same MPC candidate set as `mpc_beam` unless an experiment explicitly
  changes it: currently `girard`, `scott`, `interval_hull`, and `pca`.
- Keep `interval` as a fallback, not as a supervised candidate.
- Use the same binding-native reference-loss objective as MPC for regret
  candidate costs; keep width metrics as diagnostics.
- Preserve transform-bound budget semantics: `budget` is the bound passed to
  `ZonotopeConfig.<method>(budget)`.
- Append learned rows to the same benchmark tables as static and MPC methods.
- Write candidate-cost tables, ranking diagnostics, training diagnostics, and
  eval rows under `learning/<scenario>/`.

## Suggested Implementation Checklist

1. Generalize `rtlola.learning` around `RtlolaScenarioSpec`.
2. Replace direct imports of `OMNI_SPEC`, `OMNI_EXPECTED_VERDICT_KEYS`, and
   `generate_omni_events` in the learning path with scenario metadata.
3. Build the supervised candidate list from `mpc_actions(action_by_name(...))`.
4. Pass the binding-native reference-loss objective into all regret-oracle and
   learned-policy beam calls.
5. Record learned decision wall-clock time in `RtlolaStepRecord`.
6. Keep the current ranked-beam policy first, then add a separate top-1
   learned policy only after diagnostics show the teacher labels are useful.
7. Add robot-arm RTLola learning smoke tests that assert learned rows and
   learning artifacts are written.

## Dependency on MPC Objective

Do not use the current RTLola robot-arm teacher as a headline learned-policy
target without first auditing the binding-native reference-loss objective on
full runs. A learner should reproduce the final accepted MPC objective, not the
older hand-rolled width proxy.
