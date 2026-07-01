# RTLola Integration Notes

The superproject pins `rlolapythonbinding` at
`3b0443c1eb038b19bc3949ff3ba82d03617616aa`.

The binding exposes monitor construction, state snapshots, state restoration,
branch evaluation, dynamic/total zonotope matrices, native approximation loss,
runtime verdict metadata, structured affine verdict bounds, and these
transforms:

- no-op and interval;
- bounded interval hull;
- colinear and bounded colinear scale;
- Girard, Scott, PCA, Althoff A, clustering, and Combastel.

The June 2026 update added clustering and Combastel and corrected conversion of
`None` for asynchronous inputs. The July 2026 update added immutable
`AffineValue` verdicts with numeric center/lower/upper properties. It did not
add experiment search, learning, or a robot-arm specification. Both repository
scenarios therefore remain packaged `.lola` resources executed by the binding.

The current `kmeans 2.0.2` dependency declares a nightly-only test feature even
for library builds. `tools/setup_rtlola_binding.sh` scopes
`RUSTC_BOOTSTRAP=kmeans` to that crate. Remove this workaround once the
upstream dependency builds on stable Rust.

The extension currently requires the conda OpenBLAS preload documented by the
setup script. Binding-backed tests are mandatory before experiment runs.
