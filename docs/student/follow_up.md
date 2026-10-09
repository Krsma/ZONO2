# Follow-up research interfaces

The delivered workflow deliberately centers neural selection of native reducers.
These next stages reuse `Task`, `Observation`, `Zonotope`, and the native evaluator.
They are not implemented or claimed sound by the first release.

## Representation learning

Add geometry capture/replay with sample IDs and exact Geometry15 alignment
checks. Existing teacher bundles contain scalar features, not generator matrices.
Replay must reproduce teacher-path states before its geometry is accepted;
otherwise capture a newly versioned dataset. Keep trajectory splits intact
across budgets and fit pretraining/preprocessing only on training trajectories.

Store raw variable-sized geometry, then form set or bipartite graph batches at
load time. Coordinate nodes, generator nodes, and signed coefficients can feed
an encoder producing pooled and per-node embeddings. A scoring head uses the
pooled representation plus budget. PyTorch Geometric can be an optional adapter;
the geometry itself should not depend on it. Preserve zero rows, coordinate
semantics, and generator sign/permutation symmetry. A sign-augmentation example
must not claim exact invariance. Dense zonotopes can produce dense graphs.

## Learned reduction and forecasting

Use distinct tasks and target contracts:

- Reduction maps a current zonotope and budget to a candidate enclosure of the
  current set. Decoder output remains an offline candidate.
- Dynamics prediction maps current geometry, causal context, and proposed actions
  to a future zonotope. Record the horizon and whether future inputs are assumed,
  supplied offline, or predicted causally. Do not imply stable row identities.

Matrix MSE alone does not describe geometric equivalence. Define target geometry
and diagnostics independently of a particular generator ordering. Native state
loss is not automatically available for arbitrary predicted matrices.

## Certification boundary

Keep `predict` separate from `certify` and eventual `repair`. A certificate must
identify the sets/assumptions checked, the enclosure claim, numerical guarantees,
and verification method. An unsuccessful sufficient check means **unknown**;
only an actual counterexample justifies a negative containment claim. Sampled
support directions are diagnostics, not a proof of enclosure.

Reduction certification concerns inclusion of the current set. Forecast
certification concerns reachable sets under explicit input/dynamics uncertainty;
agreement with one recorded future is insufficient. Budget and constant slack
requirements remain part of any eventual reduction contract.

Live prediction requires a separately authorized native extension and a sound
fallback. Python matrix writeback remains prohibited, even if an external checker
reports a certificate. Fast sufficient containment checks, sound numerical
inflation, and certified construction are independent research directions.
