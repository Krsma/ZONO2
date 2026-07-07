# Align headline metric and MPC objective with the rlola-eval notebook

## Context

The loss-discrepancy investigation (science/RTLOLA_MPC_LOSS_INVESTIGATION.md) confirmed both
environments are sound apart from the collaborator's slack-less reference bug, but they differ in
conventions: the notebook's headline is per-step **summed** loss ("total loss") with **cumulative**
beam scoring against a **global stored exact reference**, while ZONO2 headlines mean loss with a
terminal-only beam scored against a local unreduced continuation. To make future cross-checking
trivial, we align conventions:

1. rename `sum_approx_loss` → `total_approx_loss` and promote it to the headline loss;
2. add `mpc_cumulative_beam` (cumulative explicit-horizon loss, no tail) as the new baseline MPC
   method; `mpc_terminal_beam` stays available unchanged;
3. new config `mpc_reference: rollout|cache` (default `rollout`) selecting the MPC/teacher search
   reference: local `none`-continuation (current) vs the exact reference cache rows at the
   corresponding absolute steps. Applies to all MPC variants and the learning teacher.

User decisions: new method + new baseline; default `rollout`; uniform scope incl. teacher; full
rename to `total_approx_loss` (hard-error on legacy artifacts).

## Key verified facts

- Cumulative machinery exists: `explicit_path_loss` is accumulated at search.py:455/489/513;
  `_prefix_cost` (604-611) and `_complete_cost` (701-715) switch on objective.
- Cache alignment invariant: reference row `i` has `approximation.step == i+1`; a search rooted at
  trace index `index` yields depth-`d` states with `step == index+1+d`. Pass
  `reference[index : index+1+horizon(+tail)]`, index by depth. `engine.approx_loss_reference`
  enforces step equality and exact center equality — reuse it **strictly**; a center mismatch
  raises `RtlolaBindingError`, which `_run_single` already records as a `select`-phase failure.
- `_run_single` already has the loaded `reference` tuple in scope (benchmark.py:403) but never
  passes it to searches (443-484).
- Teacher (`learning.py:_evaluate_candidates`, 590-629) uses `beam_search` (terminal); cache mode
  for training traces requires loading/computing exact references per training trace.

## Edits (in order)

### 1. src/pzr/rtlola/search.py
- `MpcObjective`: add `CUMULATIVE = "cumulative_binding_approx_loss"`.
- `MPC_VARIANTS`: add `MpcVariant("mpc_cumulative_beam", CUMULATIVE, GLOBAL,
  uses_configured_horizon=True, uses_tail=False)`.
- `_prefix_cost`: return `explicit_path_loss` for `{INTEGRATED_TAIL, CUMULATIVE}`.
- `_complete_cost`: explicit branch `CUMULATIVE → explicit_path_loss`.
- Thread `reference_rows: Sequence[RtlolaApproximationReference] | None = None` through
  `beam_search`, `search_mpc_variant`, `_search_mpc`; validate length ≥ explicit+tail+1; when set,
  skip `_reference_rollout` (356-359) and make `_score_step` (792-802, 5 call sites) and
  `_evaluate_tail` (679-681) call `engine.approx_loss_reference(reference_rows[depth], step.state)`.
  Leave the under-budget `none` short-circuit (331-352) untouched in both modes.
- `search_mpc_variant`: add `forced_first_action=None` passthrough; make `tail_action` optional
  (`None` allowed for no-tail variants); record `configured_tail_horizon=0` when `uses_tail` False.

### 2. src/pzr/rtlola/benchmark.py
- `BASELINE_MPC_METHODS = ("mpc_cumulative_beam",)` (line 61).
- Replace constant (62) with `CUMULATIVE_BINDING_APPROX_LOSS = "cumulative_binding_approx_loss"`;
  `mpc_objective` field default (120-123) → that constant (stays `init=False`).
- New init field `mpc_reference: str = "rollout"` next to `reference_mode`.
- `run_benchmark` early validation: value in `{rollout, cache}`; `cache` requires
  `reference_mode == "exact"`.
- `_run_single`: in both MPC branches, when cache mode, slice
  `tuple(row.approximation for row in reference[index : index+1+len(future)(+len(tail))])`, guard
  `approximation is not None`, pass as `reference_rows`.
- Rename `sum_approx_loss` → `total_approx_loss` (lines 75, 1068, 1182).
- `_plot_pareto` (1266-1283): headline y = `total_approx_loss` (agg "mean" of the per-run totals),
  keep all-NaN fallback to `mean_state_width`, label "Total approximation loss".

### 3. src/pzr/rtlola/learning.py
- `_evaluate_candidates`: switch to `search_mpc_variant(..., variant=MPC_VARIANTS["mpc_cumulative_beam"],
  root_beam_width=1, forced_first_action=first, reference_rows=..., configured_tail_horizon=0)`;
  new param `reference_rows=None`.
- `_collect_episode`: new `reference=None` param; slice rows per step in cache mode.
- `train_and_evaluate_regret`: same validation as `run_benchmark`; in cache mode call
  `load_or_compute_reference(..., include_approximation=True)` per training trace (cache under
  train-seed names) and thread into `_collect_episode`.

### 4. src/pzr/rtlola/cli.py
- `--mpc-reference {rollout,cache}` default `rollout`; map into params; add `mpc_reference` to the
  learned-metadata payload next to `mpc_objective` (line 223).

### 5. src/pzr/rtlola/sweep_report.py
- Rename at lines 28, 254, 567, 638.
- Legacy guard in `consolidate_sweep`: raise `ValueError` if `"sum_approx_loss"` appears in the
  combined columns (prevents silently-NaN tables from old artifacts).
- `_learned_comparison` (652): compare against `BASELINE_MPC_METHODS[0]` instead of literal
  `"mpc_terminal_beam"`.

### 6. Tests
- tests/test_rtlola_units.py: rename column in fixtures/assertions (82, 103, 124, 200-229);
  keep `test_mpc_objective_is_fixed_and_not_a_cli_option`. New tests:
  - `test_mpc_reference_validation_and_default` (default `rollout`; `cache`+`verdict` raises;
    bogus value raises);
  - `test_cumulative_beam_ranks_by_cumulative_explicit_loss` (FakeEngine, per-step losses A=(5,0)
    B=(1,3): cumulative picks B cost 4.0, terminal picks A; objective strings and
    `configured_tail_horizon==0` asserted);
  - `test_cache_reference_rows_replace_local_rollout` (FakeEngine: `approx_loss` fails the test,
    `approx_loss_reference` records rows; asserts depth indexing incl. tail offsets, no local
    none-rollout branches, short `reference_rows` raises ValueError);
  - `test_sweep_report_rejects_legacy_sum_approx_loss_artifacts`;
  - baseline pinning (`BASELINE_MPC_METHODS`, CORE_METHODS, terminal variant unchanged).
- tests/test_rtlola_binding.py: objective assertions in `test_benchmark_writes_rtlola_native_artifacts`
  (286-288) → cumulative; `test_robot_arm_mpc_uses_binding_terminal_loss` (310) → assert the
  timeseries `mpc_objective` column instead of config; rename at 457, 544, 552, 557. New:
  - `test_robot_arm_cumulative_beam_baseline_runs` (length 8, budget 40, no failures, objective
    column, `realized_tail_steps==0`);
  - `test_robot_arm_cache_reference_search_completes_exactly` (`mpc_reference="cache"` with tmp
    reference cache, methods both beams, `failures == ()` — end-to-end center/step alignment).
- tests/test_rtlola_learning.py: existing teacher test unaffected; new cache-mode regret smoke +
  `ValueError` case.

### 7. tools/
- run_rtlola_robot_arm_fpr_overnight.sh: add `mpc_cumulative_beam` to METHODS (14); add
  `MPC_REFERENCE="${PZR_MPC_REFERENCE:-rollout}"` and `--mpc-reference` to run+learning stages;
  heredoc renames (160, 208).
- run_rtlola_mpc_variant_study.sh: add method to METHODS (16); same flag plumbing.
- run_rtlola_mpc_vs_girard_quick.sh: default methods `girard,mpc_cumulative_beam` (28); heredoc
  literals (54, 57); metric list (77); `native_loss_*` alias (92-97) → `total_approx_loss_*`.
- run_rtlola_omni_fidelity_overnight.sh: baseline method swap (12, 155).

### 8. Docs
- AGENTS.md: add `mpc_cumulative_beam` (baseline) to the MPC method list; Trusted Boundary — baseline
  beam + teacher use undiscounted cumulative binding loss over the explicit horizon; replace "MPC and
  teacher searches construct their own unreduced horizon rollouts" with the `mpc_reference`
  description (`cache` requires exact mode; center mismatch = recorded run failure; under-budget
  `none` gate unchanged); `total_approx_loss` wording (178).
- README.md: method list, `--mpc-reference`, `total_approx_loss`.
- CLAUDE.md: "binding-native terminal approximation loss" → "binding-native approximation loss
  (cumulative over the explicit horizon for the baseline beam and teacher)".
- Do NOT touch science/ or existing results/ (historical names stay; sweep guard prevents mixing).

## Verification

1. Pure: `PYTHONPATH=src python -m pytest tests/test_rtlola_units.py`.
2. Binding gate (no skips):
   `LD_PRELOAD="$PWD/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so" PYTHONPATH=src
   external/miniconda3/envs/pzr-robot-arm/bin/python -m pytest`
3. Smoke: `... -m pzr.rtlola.cli --profile smoke --scenario robot_arm --trace-kind figure8_drift
   --length 40 --seeds 1 --budget 40 --horizon 4 --beam-width 4
   --methods girard,mpc_terminal_beam,mpc_cumulative_beam --reference-mode exact
   --output /tmp/pzr-arm-cumulative` → summary has `total_approx_loss` only, empty
   run_failures.csv, pareto y-axis = total loss, config.yaml shows cumulative objective +
   `mpc_reference: rollout`.
4. Cache-mode smoke: same + `--mpc-reference cache --reference-cache /tmp/pzr-arm-ref.json
   --output /tmp/pzr-arm-cache` → empty run_failures.csv; diff `reducer_used` vs rollout run.
5. Learned smoke (omni, regret 1 iter) with `--mpc-reference cache`; check regret_metadata.json
   records objective + mpc_reference.
6. `grep -rn sum_approx_loss src tests tools AGENTS.md README.md CLAUDE.md` → empty.
