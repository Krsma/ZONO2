# May 18 Results Overview

This note summarizes the completed experiment run at `results/tacas-main`.
The run is internally clean and usable:

- `budget_violation_count = 0`
- `unsound_certificate_count = 0`
- `reduction_failure_count = 0`

The main scientific story is not that every MPC variant dominates every
baseline. The strongest supported claim is:

> Focused predictive rollout improves precision on the robot-style monitors at
> fixed memory budget. The wider rollout and the distilled learned policy expose
> a second tradeoff: they often reduce the number of true reductions
> dramatically, with certified soundness preserved, but they are not always the
> best precision methods.

## Plots To Inspect First

1. `results/tacas-main/figures/figures/fig5b_omni_error_by_budget.png`

   Best headline plot for the hard omnidirectional robot. `mpc_rollout_girard`
   beats Girard across budgets.

2. `results/tacas-main/figures/figures/fig5a_omni_error_over_time.png`

   Shows the hard robot error accumulating over time. This is the main
   "predictive reduction helps over long traces" plot.

3. `results/tacas-main/figures/figures/fig3a_simple_error_over_time.png`

   Strong for the simple robot. `mpc_rollout_girard` is clearly best at final
   time.

4. `results/tacas-main/figures/figures/fig3b_simple_error_by_budget.png`

   More nuanced: `mpc_rollout_girard` is best at budget `8`, while Combastel or
   wide/learned can be competitive at larger budgets.

5. `results/tacas-main/figures/figures/fig4a_omni_position_x_trace.png`

   Qualitative bound plot. Use this if it visually shows tighter bands without
   becoming cluttered.

6. `results/tacas-main/figures/figures/fig4b_omni_false_alarm_rates.png`

   Useful as a sanity check, not the central claim. False-alarm rates are small;
   Girard is slightly lower than our focused rollout here.

## Main Numerical Results

For paper numbers, avoid double-counting aggregate rows. Use:

- `results/tacas-main/runs/<scenario>/baseline/raw_runs.csv` for static/MPC
  methods.
- `results/tacas-main/runs/<scenario>/learned/raw_runs.csv`, filtered to
  `learned_distilled`, for the learned policy.

### Hard Robot, Online Mode

| method | trigger MSE | false alarm | reductions | no-ops | mean generators |
|---|---:|---:|---:|---:|---:|
| Girard | 881.5 | 0.0105 | 293 | 0 | 7.93 |
| `mpc_rollout_girard` | 671.1 | 0.0113 | 293 | 7 | 7.93 |
| `mpc_rollout_wide` | 2333 | 0.0171 | 116.9 | 183.1 | 7.05 |
| `learned_distilled` | 2388 | 0.0173 | 105.1 | 194.9 | 6.99 |

Interpretation: focused rollout is the precision winner among our methods.
Wide/learned sacrifice precision but reduce true reductions by roughly 60-65%.

### Simple Robot, Online Mode

| method | trigger MSE | false alarm | reductions | no-ops |
|---|---:|---:|---:|---:|
| Girard | `7.85e-28` | 0 | 297 | 0 |
| `mpc_rollout_girard` | `2.47e-28` | 0 | 297 | 3 |
| `mpc_rollout_wide` | `4.24e-27` | 0 | 297 | 3 |
| `learned_distilled` | `4.24e-27` | 0 | 297 | 3 |

Interpretation: focused rollout is again the best predictive precision story.
Wide/learned are not helpful here.

### Thermostat, Online Mode

| method | trigger MSE | false alarm | reductions | no-ops | time |
|---|---:|---:|---:|---:|---:|
| Combastel | `1.23e-31` | 0 | 293 | 0 | 0.093s |
| `mpc_rollout_girard` | `1.33e-31` | 0 | 293 | 7 | 0.873s |
| `mpc_rollout_wide` | `9.67e-32` | 0 | 63.2 | 236.8 | 2.20s |
| `learned_distilled` | `5.77e-32` | 0 | 59.2 | 240.8 | 0.097s |

Interpretation: thermostat is the best evidence for the learned/wide
timing-control story. Learned distillation performs very well here: low error,
few reductions, and runtime comparable to static reducers.

## Method Recap

`reference`

Unreduced monitor state. This is not bounded memory. It is the precision
reference used to compute interval-hull error, width inflation, and verdict
disagreement.

`box`

Replaces the zonotope with its axis-aligned interval hull. Very cheap and often
uses few generators, but can be coarse because it loses generator correlations.

`girard`

Classic keep-and-box reducer. It keeps generators selected by Girard's
`l1 - linf` score and boxes the rest. This is the strongest static baseline in
the robot results.

`combastel`

Another keep-and-box reducer, but orders generators by L2 norm. It performs
well on thermostat and at some larger simple-robot budgets.

`methA`

Transform-based reduction using a basis from long generators. In these
experiments it is usually much worse than Girard/Combastel.

`scott`

Transform-based reduction using pivoted independent directions. Similar role to
`methA`; also weak in these monitor benchmarks.

`pca`

PCA-basis interval-hull reduction. It performs very poorly on the hard robot and
should mostly be treated as a baseline that illustrates how generic geometric
reductions can fail for monitor semantics.

`adaptive`

A current-step adaptive reducer. It tries certified candidate reducers and picks
the one with lowest immediate monitor-aware cost. It is not predictive. In
these results it often behaves similarly to `box`.

`mpc_rollout_girard`

The focused method to present as "ours." It chooses a first action using
short-horizon rollout, then uses protected Girard for future predicted
overflows and protected box as fallback. It preserves required calibration
generators and only applies certified reducers. This is the strongest precision
method in the robot experiments.

`mpc_rollout_wide`

A broader ablation. It considers many protected static reducers as possible
first actions, plus no-op, then rolls future overflows with protected Girard and
fallback. It often chooses no-op or low-generator reductions, so it reduces less
often, but it is not the precision winner on robot.

`learned_distilled`

A neural policy trained to imitate `mpc_rollout_wide` decisions from
`decision_features.csv`. It still applies certified reducers, so the learned
model is not trusted for soundness. Validation accuracy was about `94.7%`,
top-3 accuracy about `99.9%`. Its best role is "cheap approximation of
expensive rollout decisions," especially on thermostat.

## Strong Supported Claims

- Predictive focused rollout improves interval precision over strong static
  baselines on robot-style monitors.
- The method preserves bounded-memory soundness: all reducers are certified;
  the experiment has zero budget/soundness failures.
- Distillation can approximate expensive rollout choices with much lower online
  cost.
- Wide action sets/no-op decisions show useful reduction-timing control,
  especially on thermostat and the hard robot.

## Claims To Avoid Or Qualify

- Do not claim `mpc_rollout_wide` is always best. It is not.
- Do not claim oracle prediction is essential. The online-vs-oracle gap is
  small or mixed; robust prediction is not yet the main story.
- Do not claim false alarms are dramatically reduced. They are mostly low
  already, and Girard is slightly better in the main false-alarm plot.
