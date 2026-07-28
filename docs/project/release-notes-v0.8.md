# v0.8.0 release notes

- **Release status:** pre-release remediation active under [issue #35](https://github.com/DeloneCommons/pyvoro2/issues/35); final qualification in issue #33 has not begun
- **Release type:** feature-free technical maintenance
- **Previous release:** v0.7.0

v0.8.0 is intended to complete the compatibility removals announced for v0.7, qualify
standard CPython 3.14 and the full binary distribution matrix, and make the
documented public contract match the reorganized source tree. These claims remain
pending until the audit remediation and final qualification are complete. It does not add a
new numerical method, inverse observation family, domain type, or solver.
Prescribed cell measures remain planned for v0.9, and mixed separator-plus-
measure fitting remains planned for v0.10.

## Removed

The following v0.7 transition surfaces are no longer importable or accepted:

- `pyvoro2.powerfit`, its direct submodules, and the lazy top-level
  `pyvoro2.powerfit` attribute;
- broad separator-specific exports from top-level `pyvoro2`;
- the historical separator aliases `PairBisectorConstraints`,
  `resolve_pair_bisector_constraints`, `PowerFitProblem`,
  `PowerWeightFitResult`, and `fit_power_weights`;
- `pyvoro2.planar.PlanarComputeResult` and
  `pyvoro2.planar.result.PlanarComputeResult`; and
- planar `compute(..., return_result=...)`.

Use `pyvoro2.inverse` for the stable fixed-observation separator workflow and
`pyvoro2.inverse.separator` for advanced and experimental separator-specific
work. Both forward namespaces return `TessellationResult` by default.
`output='cells'` remains a supported explicit low-level output mode; it was not
a compatibility-only route and was not removed.

## Changed behavior and fixes

- Separator external IDs now consistently require unique non-negative integer
  values. Python integers and NumPy integer scalars are accepted; floats,
  strings, booleans, and other lossy conversions are rejected. Raw observation
  endpoints follow the same strict integer rule in both index and ID modes.
- Direct `TessellationResult(...)` construction is documented as provisional.
  It validates the documented aligned metadata but does not normalize
  arbitrary backend-shaped records, recompute geometry, or prove geometric
  validity.
- Deep copies and same-version pickle round trips preserve
  `TessellationResult` snapshot and read-only-array state. Cross-version pickle
  restoration is not promised.
- Distribution metadata validation now discovers wheel and source archives in
  Python and passes explicit paths to Twine, avoiding shell-glob differences
  across platforms.

These are API-contract consistency corrections from issue #31. They do not
change the forward tessellation algorithms, separator objective formulas,
solver defaults, gauge policy, result fields, record keys, or report schemas.

## Packaging and platform support

Supported source builds use standard GIL-enabled CPython 3.10, 3.11, 3.12,
3.13, or 3.14. Package metadata declares `Requires-Python: >=3.10`, so installers
do not impose an artificial upper bound, but versions newer than 3.14 are not
part of the v0.8 tested support contract. Source-install CI exercises all five
supported versions on Linux, macOS, and Windows.

The release artifact set is exactly 21 distributions:

| Artifact | Python versions | Platform / architecture | Count |
|---|---|---|---:|
| CPython wheels | 3.10–3.14 | manylinux x86_64 | 5 |
| CPython wheels | 3.10–3.14 | Windows AMD64 | 5 |
| CPython wheels | 3.10–3.14 | macOS arm64 | 5 |
| CPython wheels | 3.10–3.14 | macOS x86_64 | 5 |
| Source distribution | source archive | platform-independent archive | 1 |

Every supported wheel contains both native extensions, `_core` and `_core2d`,
and the workflow installs and exercises each wheel on a compatible runner. The
source distribution is built separately, checked for required content and
metadata, rebuilt into a wheel in build isolation, and smoke-tested from a
fresh no-SciPy environment.

There are no v0.8 wheels for free-threaded CPython, PyPy, GraalPy, musllinux,
Linux architectures other than x86_64, 32-bit Windows, Windows arm64, or macOS
universal2. A source install on an unlisted environment may work, but it is not
part of the v0.8 support claim.

## Import and internal maintenance

- Private pure-Python helpers now live under `pyvoro2._internal`, with shared,
  spatial, and planar ownership made explicit. These routes are internal and
  have no compatibility guarantee.
- The compiled native extensions remain root-owned internal modules named
  `pyvoro2._core` and `pyvoro2._core2d`; they were not moved under
  `_internal`.
- Plain `import pyvoro2` does not import `pyvoro2.inverse`, `_core`, or
  `_core2d`. Native extensions continue to load only when the corresponding
  forward geometry operation first needs them. Canonical inverse-only imports
  also remain usable without eagerly loading a native extension.
- Distribution-content and installed-package checks cover the new internal
  hierarchy, both native extensions, removed obsolete paths, module
  provenance, and representative forward and inverse workflows.

## Tests and documentation

- Tests are organized by responsibility under `tests/forward/common`,
  `tests/forward/spatial`, `tests/forward/planar`,
  `tests/inverse/separator`, `tests/integration`, `tests/tooling`, and
  `tests/fuzz`.
- The public API inventory is finalized against the v0.8 exports, signatures,
  defaults, result fields, raw and inverse record keys, report schemas, and
  lifecycle classifications.
- The README source and generated README, API-selection and migration guides,
  reference navigation, architecture and contributor documentation, notebooks,
  changelog, and these release notes use the same canonical routes.
- Current examples and notebooks use `pyvoro2.inverse` or
  `pyvoro2.inverse.separator`; removed names remain only where migration or
  historical documentation intentionally explains them.

## Lifecycle summary

- **Stable:** 2D and 3D forward domains and operations; the core
  `TessellationResult` fields; diagnostics, validation, normalization,
  annotations, duplicate checks, weight/radius transforms; and the six-name
  high-level `pyvoro2.inverse` surface.
- **Provisional:** direct `TessellationResult` construction and optional
  conveniences, visualization helpers, advanced separator model/problem/
  operator/realization/report objects, and the explicit SciPy sparse quadratic
  backend.
- **Experimental:** realization-aware active-set refinement and its
  separator-specific options, path, termination, diagnostics, and result
  objects.
- **Internal:** `pyvoro2._internal`, `pyvoro2._core`, `pyvoro2._core2d`, and
  underscore-prefixed implementation details.
- **Removed:** the v0.7-only compatibility routes listed above. No
  compatibility-only or deprecated surface remains in the current v0.8 public
  namespace.

See [Choosing an API](../guide/choosing-api.md) for the preferred entry points,
[Migrating from v0.6.3 through v0.8](../guide/migration-v0.7.md) for concrete
replacements, and the [v0.8 public API inventory](../development/api-inventory.md)
for the exhaustive contract.
