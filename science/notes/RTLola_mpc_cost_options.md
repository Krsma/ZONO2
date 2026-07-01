# RTLola MPC Cost Options

Status as of 2026-06-25.

## Current Implemented Objective

RTLola `mpc_beam` uses the binding-native terminal approximation loss:

```text
reference = unreduced/no-reduction rollout from the same branch point
candidate = reduced candidate rollout
J = RTLola approx_loss_state(reference_H, candidate_H)
```

The binding metric compares interval-bound error between the reference and
candidate total zonotopes, including constant slack. This replaces the earlier
hand-rolled absolute row-width objective for MPC selection.

Offline exact-reference evaluation uses the same binding metric against the
full unreduced ground-truth state. Width columns remain diagnostic outputs.

The objective still does not include generator count, switching penalties, or
method-specific preferences.

## 2026-07-01 Rejected Trigger-Width Experiment

An experimental Python objective minimized terminal normalized widths of the
public streams consumed by robot-arm triggers. It also required a new binding
verdict type exposing affine interval bounds.

Across the full figure-eight and square violated traces at budgets `40`, `80`,
`120`, and `180`, the objective changed MPC from predominantly Scott to
predominantly Girard. It did not change FPR: figure-eight remained `3/2155`
and square remained `2/1818` at every budget. It also removed the small
whole-state-width improvement previously observed with binding loss.

The experiment was therefore rejected. The binding API and active objective
were restored to the binding-native terminal `approx_loss_state`. Generated
artifacts remain under `results/rtlola-arm-trigger-width` for comparison.

## 2026-06-25 Robot-Arm Result

The first full robot-arm run with binding-native terminal reference loss is
recorded in `science/notes/RTLola_binding_loss_eval_20260625.md`.

Main finding: replacing terminal relevant-width scoring with the binding-native
`approx_loss_state` objective did not remove Scott dominance. Across budgets
`120`, `160`, and `240`, `mpc_beam` still chose Scott for roughly 2290-2310 of
2340 steps and improved mean width over static Girard by only about 0.71-0.73%.

This suggests the next objective audit should focus on why finite-horizon
binding loss is weakly discriminative along this trace, rather than treating
the previous hand-rolled width metric as the primary cause of Scott-heavy MPC
behavior.

## Previous Width Baseline

The previous audited objective was terminal-only absolute width:

```text
J = c(s_H)
```

where `c` was the scenario-specific width cost. For robot arm, that meant the
sum of relevant-row interval widths. This was useful as a myopia audit, but it
remained a hand-rolled proxy and was sensitive to row scale.

## Deferred Width-Only Alternatives

### Linear Later-Weighted Ramp

```text
J = (1*c(s_1) + 2*c(s_2) + ... + H*c(s_H)) / (1 + 2 + ... + H)
```

This keeps all predicted states in the score while favoring later states. It
has no free weight parameter beyond the horizon itself.

### Quadratic Later-Weighted Ramp

```text
J = (1^2*c(s_1) + 2^2*c(s_2) + ... + H^2*c(s_H)) / (1^2 + 2^2 + ... + H^2)
```

This is closer to terminal-only than the linear ramp, but still retains
intermediate-state information.

### Max Future Width

```text
J = max(c(s_1), ..., c(s_H))
```

This optimizes the worst uncertainty state encountered during the rollout. It
is most defensible when transiently wide bounds are themselves harmful, for
example because they can cause false positives at intermediate steps.

### Terminal Tail Average

```text
J = mean(c(s_{H-k+1}), ..., c(s_H))
```

This smooths terminal-only scoring when the final horizon state is noisy. A
small fixed tail such as `k=2` would be the likely first audit variant.

### EMA-Style Later Bias

An exponentially later-weighted cost could interpolate between cumulative and
terminal-only scoring. This is lower priority because the decay factor becomes
a tuning parameter unless fixed by convention.

## Explicitly Out of Scope For This Direction

- Generator-count penalties. The RTLola transform bound already controls the
  reducer target, and generator-count objectives previously risked encouraging
  interval-like behavior.
- Switching penalties or reducer-specific regularization. These can be useful
  diagnostics, but they are easy to interpret as method bias. Binding-native
  approximation loss should be audited first.
