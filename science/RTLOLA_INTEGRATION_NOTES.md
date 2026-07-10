# RTLola Integration Notes

Historical note: this file records an earlier RTLola integration state. The
current required stack is documented in `AGENTS.md` and
`science/RTLOLA_BINDING_NATIVE_REFERENCE_NOTES.md`.

At the time of this note, the superproject pinned `rlolapythonbinding` at
`abe3dab33d0c4aa504db0af63901b66ecafb7f71`, which locks the RTLola
interpreter to `a143dd6a1500d54c1eabe9e83e5b54271734d6b2`.

The binding exposes monitor construction, state snapshots, state restoration,
branch evaluation, dynamic/total zonotope matrices, native approximation loss,
runtime verdict metadata, and these transforms:

- no-op and interval;
- bounded interval hull;
- colinear and bounded colinear scale;
- Girard, Scott, PCA, Althoff A, clustering variants, and Combastel.

The June 2026 update added clustering and Combastel and corrected conversion of
`None` for asynchronous inputs. It did not add experiment search or learning.
Both repository scenarios remain packaged `.lola` resources executed by the
binding. The robot-arm specification and traces are copied verbatim from
RLolaEval commit `f587a0e`.

Combastel is enabled as an additional static comparator and as an MPC and
learning candidate in the primary overnight sweeps. Deterministic clustering
is also an MPC and learning candidate, but not a primary static comparator.
Robot-arm screening at budget 40 found deterministic clustering falling back
on 39 of 80 events;
random and diverse clustering both failed at event 3 because their cluster
bases were singular or ill-conditioned. Althoff A and colinear scale each
exceeded 75 seconds for an 80-event screen. Althoff A, colinear scale, and
static deterministic clustering therefore remain explicit-only methods;
random and diverse clustering are not wired into the benchmark.

A 60-event, horizon-4 robot-arm screen compared the original four MPC
candidates with the five-candidate catalog. Combastel reduced the
exact-reference mean loss by about 12%, 4%, and 20% at bounds 40, 80, and 180,
respectively, while increasing MPC decision time by 11--17%. At bound 120 it
instead changed the action trajectory and increased mean loss from
`5.24e-8` to `2.89e-6` while increasing decision time by 56%. This is an
important evaluation target: MPC optimizes loss relative to an unreduced
rollout from its current approximate state, so adding a locally attractive
action need not improve cumulative loss relative to the globally unreduced
trace.

The optional reducers were screened more deeply before exclusion from the
primary catalog:

- deterministic clustering fell back on 50%, 33%, 18%, and 12% of 300 events
  at bounds 40, 80, 120, and 180; its mean state widths were respectively
  14.6, 10.5, 243, and 546;
- adding deterministic clustering to bound-120 MPC selected it once in 60
  events, left all recorded fidelity metrics unchanged, and increased decision
  time by 44%;
- adding all three clustering variants to a bound-40 MPC screen increased
  decision time from 8.0 to 39.1 seconds without changing the selected actions;
- Althoff A failed on the first rank-deficient reduction and used interval
  fallback; a five-event run took 11.2 seconds versus 9.4 milliseconds for
  Girard and had approximately 17,600 times its mean approximation loss;
- colinear scale took 1.57 seconds over the first three events and its fourth
  transform alone took 111 seconds, without improving early approximation loss
  over Girard or Combastel.

The experimental structured-affine-verdict API was reverted in July 2026.
Symbolic verdict values remain strings; experiment objectives use the
binding-native state loss rather than parsing verdict expressions. Activated
triggers are emitted sparsely as `Trigger#N`; absent trigger keys are false.

The current `kmeans 2.0.2` dependency declares a nightly-only test feature even
for library builds. `tools/setup_rtlola_binding.sh` scopes
`RUSTC_BOOTSTRAP=kmeans` to that crate. Remove this workaround once the
upstream dependency builds on stable Rust.

The extension is installed from a locked CPython 3.11 wheel built with
`maturin build --release`. It currently requires the conda OpenBLAS preload
documented by the setup script. Binding-backed tests are mandatory before
experiment runs.
