# RTLola Binding-Loss MPC Evaluation, 2026-06-25

## Setup

Evaluation root:

```text
results/rtlola-arm-binding-loss-eval-20260625/
```

Configuration:

- Scenario: `robot_arm`
- Trace: `figure8_violated`
- Length: `2340`
- Seed: `1`
- Budgets: `120`, `160`, `240`
- Methods: `girard`, `scott`, `interval_hull`, `pca`, `mpc_beam`
- Horizon: `4`
- Beam width: `4`
- Reference mode: `off`

Because `reference-mode` was `off`, exact offline `mean_approx_loss` columns
are intentionally unset. The MPC objective still used finite-horizon
binding-native reference loss internally and stored that value in
`timeseries.predicted_cost`.

## Result Table

```text
budget method         mean_width max_width mean_gens max_gens mean_active_gens reductions time_s  post_over fallback infeasible
120    girard         46.2125    93.6995  156.89    157      152.88           2336       14.32   2337      0        0
120    interval_hull  47.3839    96.0191   99.49    155       89.49            584       16.37    585      0        0
120    mpc_beam       45.8757    92.9561  152.96    157      148.95           2336      435.52   2337      0        0
120    pca            67.3272   138.722   156.89    157      152.88           2336       20.87   2337      0        0
120    scott          53.4059   110.683   152.89    153      148.88           2336       28.73   2337      0        0

160    girard         46.2071    93.6902  196.82    197      192.79           2335       12.86   2336      0        0
160    interval_hull  47.0768    95.4000  117.99    192      105.99            467        6.79    468      0        0
160    mpc_beam       45.8789    92.9594  192.87    197      188.85           2335      514.77   2336      0        0
160    pca            60.7133   121.560   196.82    197      192.79           2335       16.63   2336      0        0
160    scott          53.1373   110.245   192.82    193      188.80           2335       16.46   2336      0        0

240    girard         46.2004    93.6781  276.61    277      272.57           2333       26.01   2334      0        0
240    interval_hull  46.7196    94.6895  154.90    266      138.91            334        8.65    334      0        0
240    mpc_beam       45.8733    92.9540  272.69    277      268.65           2333      534.45   2334      0        0
240    pca            54.7096   109.321   276.61    277      272.57           2333       24.79   2334      0        0
240    scott          52.6499   109.430   272.62    273      268.58           2333       31.64   2334      0        0
```

## MPC Choices

```text
budget none girard scott interval_hull pca  mean_pred_cost  max_pred_cost
120       4     38  2296             0   2   1.31e-07       1.29e-04
160       5     27  2307             0   1   3.35e-08       7.52e-05
240       7     40  2292             0   1   7.35e-08       1.70e-04
```

## Interpretation

Switching MPC from the hand-rolled terminal relevant-width objective to the
binding-native terminal `approx_loss_state` objective did not remove Scott
dominance. The resulting MPC trajectories are almost identical to the previous
terminal-width run:

```text
budget  old mpc choices                  new mpc choices
120     Scott 2296, Girard 40, none 4     Scott 2296, Girard 38, PCA 2, none 4
160     Scott 2295, Girard 40, none 5     Scott 2307, Girard 27, PCA 1, none 5
240     Scott 2289, Girard 44, none 7     Scott 2292, Girard 40, PCA 1, none 7
```

MPC still improves slightly over the best static reducer, which is Girard, but
the gain is small relative to runtime:

```text
budget best_static best_width mpc_width abs_gain pct_gain mpc_time_s
120    girard      46.2125    45.8757   0.3368   0.729%   435.5
160    girard      46.2071    45.8789   0.3282   0.710%   514.8
240    girard      46.2004    45.8733   0.3272   0.708%   534.5
```

The immediate conclusion is that the Scott-heavy behavior was not primarily
caused by the old hand-rolled width metric. After aligning the objective with
the RTLola binding, the finite-horizon reference loss remains very small and
weakly discriminative, so a tiny early preference for Scott still leads to a
path-dependent Scott-heavy trajectory.

No fallback or infeasible candidates were recorded. This supports the current
interpretation that the earlier "Girard infeasible" label was a wrapper-budget
artifact, not a reducer availability issue. `post_event_over_bound` is high by
design under transform-bound semantics because event evaluation can allocate
fresh slack after a reducer is applied.
