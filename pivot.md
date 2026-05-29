# CoRL Contingency Experiment: Monitor-First Firmware-in-the-Loop Story

## Summary

The current contingency should be monitor-first, not learned-controller-first. The core contribution is certified bounded-memory uncertainty monitoring: predictive zonotope reduction improves monitor precision and reduces avoidable safety interventions while preserving soundness. Learning remains useful only as a selector-distillation/deployability component, not as the main controller claim.

This fits CoRL only if the robotics evidence stays strong. CoRL expects work at the robotics/ML intersection and can reject submissions without a robotics focus. Firmware-in-the-loop safe-control-gym is therefore the right experimental substrate.

Current saved artifacts show why the existing Level1/PPO route is risky: the Level1 monitored run is not headline-usable because nominal completion is too low and bounded reducers are indistinguishable; PPO controller runs do not yet produce credible task progress. Level0 firmware nominal validation, however, passes cleanly.

## Key Direction

- Reframe the CoRL paper around "certified predictive reduction for robot safety monitors," not "learning controllers with our monitor."
- Use safe-control-gym firmware Level0 as the primary contingency setting, because nominal firmware control already completes reliably.
- Treat Level1 as a robustness/stress result only if calibration shows the nominal controller and monitor regime are not saturated.
- Keep the contribution centered on the safety monitor: fixed-memory, certified uncertainty tracking with better intervention quality.
- Keep learning as optional deployability evidence: a learned selector may approximate the predictive reducer at lower latency, but it is not required for the contingency story.

## Experiment Design

Expose or tune monitor-regime knobs needed for a fair calibration sweep:

- sensor bias/noise bounds
- generator budget
- stream-memory decay
- fallback hold duration
- trigger overlap or trigger margin, if current thresholds are too saturated

After calibration, fix the method set:

- `reference_unbounded`
- `box`
- `girard`
- `keep_calibration_aware`
- `mpc_focused_fixed_girard`
- `mpc_wide_fixed_girard`
- DAgger learned selector only if label diversity passes

Do not use PPO controller results as paper-critical evidence unless they become clearly successful.

## Protocol

1. Run a calibration phase on firmware Level0 and optionally Level1.
   - Find a regime where nominal completion is high.
   - Require the reference monitor to have zero or near-zero missed violations.
   - Require fallback to be neither always off nor always on.
   - Require reducer choice to change monitor decisions or intervention metrics.

2. Reject bad calibration configs.
   - Nominal completion below 0.8.
   - Fallback saturated for all bounded methods.
   - Budget violations, unsound certificates, or reduction failures.
   - All bounded methods matching Girard on intervention metrics.

3. Run the main held-out evaluation on the selected config.
   - Use disjoint calibration and evaluation seeds.
   - Primary metrics: task completion, gates passed, fallback duration fraction, fallback activations, spurious interventions, justified interventions, missed violations, reducer latency.
   - Required invariants: zero budget violations, zero unsound certificates, zero reduction failures.

4. Add selector learning only as deployability evidence.
   - Train DAgger against the predictive expert on the calibrated regime.
   - Include learned selector in the headline table only if label diversity is non-collapsed and held-out intervention metrics are close to the expert.
   - Otherwise report it as an ablation or omit it from headline claims.

## Paper Claim Shape

Primary claim:

> Predictive certified reduction reduces avoidable monitor-triggered fallback at fixed memory compared with static reducers.

Safety claim:

> Soundness is policy-independent because only certified reducers mutate monitor state.

Learning claim, only if supported:

> A learned selector can approximate the predictive selector at lower latency while staying outside the trusted soundness boundary.

## Test Plan

- Add smoke tests for any new CLI monitor-calibration knobs and ensure artifacts record the selected config.
- Smoke-run `pzr-run-corl --profile smoke` with firmware paths if available.
- Verify generated artifacts include non-empty `headline_table.csv`, `monitor_timeseries.csv`, `intervention_timeseries.csv`, `selection_summary.csv`, `predicted_sequence_summary.csv`, and `analysis_notes.json`.
- Assert `paper_usable=true` for the selected held-out run.
- Assert no budget violations, unsound certificates, reduction failures, or empty decision-feature rows.

## Assumptions

- The contingency experiment remains entirely in safe-control-gym firmware-in-the-loop simulation.
- The learned physical controller is not required for the contingency story.
- The paper can satisfy CoRL scope through a robot-safety monitor contribution plus optional learned reducer selection, not through learned low-level control.
- Calibration choices must be documented and separated from held-out evaluation to avoid cherry-picking.
