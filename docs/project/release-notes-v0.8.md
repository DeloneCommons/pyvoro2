# v0.8.0 release notes

- **Release status:** source finalized with R1–R9 and the accepted post-R9
  `COPYING` distribution correction complete
- **Release type:** feature-free technical maintenance
- **Previous release:** v0.7.0

v0.8.0 completes the compatibility removals announced for v0.7, adds standard
CPython 3.14 support to the documented release matrix, and makes the documented
public contract match the reorganized source tree. The exact final source
commit is frozen after source finalization and independent review; issue #33
qualifies that exact commit and its artifacts before the public tag is created.
This document does not claim that qualification has already succeeded. The
release does not add a new public inverse method, inverse observation family,
domain type, or solver.
Development through v0.9.0 is reserved for functional/API stabilization,
released v0.9.x is the downstream-readiness soak, and 1.0 stabilizes the existing
forward, periodic, and separator-inverse core. Prescribed cell measures begin in
v1.1, and mixed separator-plus-measure fitting begins in v1.2.

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

- Periodic nearest-image inference is now mathematically certified for the
  exact dyadic values represented by supplied binary64 coordinates and lattice
  vectors. Orthogonal/partially periodic cells use an exact per-axis fast path,
  while fully periodic non-orthogonal 3D cells exact-enumerate a proof-derived
  finite box. This fixes the audited skewed-cell case that previously selected
  `(1, 0, -1)` instead of `(2, -1, -1)`. Exact ties respect lattice translation
  and pair reversal; explicit user-provided shifts still select their requested
  image. `image_search` retains its signature and default as a bounded
  performance hint only and can no longer change a successful result. A
  private bounded resource failure raises without returning an approximation.
  Separator inference and periodic duplicate distance evaluation share this
  geometry. Mandatory forward safety always wraps; disabling optional wrapping
  preserves the established unwrapped Cartesian user-threshold check.
- Spatial and planar forward operations now prepare every inserted generator
  through one private boundary. Non-periodic coordinates must lie in `[lo,
  hi)`, periodic coordinates are remapped before dispatch, and temporary ghost
  generators follow the same rule. A fixed inclusive squared-distance floor of
  `1e-10` is always enforced with certified periodic minimum-image geometry;
  `duplicate_check`, its threshold and wrap flag, and pair-report truncation
  cannot weaken it. Safe pairs above the floor retain the existing optional
  off/warn/raise behavior. `DuplicateError` keeps its public positional
  behavior and adds safety/policy/operation/external-ID provenance. Native
  bindings repeat primary containment and local duplicate checks before every
  insertion. Triclinic Python buckets use exact source-binary64 keys and
  inverse-basis bounds; the direct-native backstop uses outward-rounded
  binary64 intervals for every uncertain key and lattice-shift range, without
  requiring extended `long double` precision. Sparse bucket keys retain radius
  locality even when a domain has very many possible bins, and positive
  subnormal diagnostic thresholds remain supported. Malformed raw compute ID
  sets fail before result packaging.
- Spatial and planar Python entry points now use one strict input contract.
  Exact integer fields reject booleans and lossy float/string conversions;
  flags and masks accept only Python or NumPy Booleans; string modes accept
  only Python or NumPy scalar strings and store canonical built-in strings;
  and numerical arrays, scalar options, and tolerances reject non-real or
  non-finite values before numerical work. String arrays, bytes, numeric
  values, and arbitrary equality objects are rejected before choice
  comparison. Domain values are canonical owned tuples, retained inverse
  arrays are owned and read-only, left-handed `PeriodicCell` bases fail at
  construction, and remapping checks signed-int64 shift range before casting.
  Normalization revalidates integer metadata in mutable raw cell records and
  rejects a coordinate/tolerance relationship whose quantized key is not
  finite and signed-int64 representable before constructing topology or
  applying in-place annotations.
  These are invalid-input and ownership corrections: valid forward/inverse
  results, public signatures/defaults/schemas, objective formulas, and native
  resource policy are unchanged.
- Spatial and planar tessellation diagnostics now use one severity-complete
  policy. Missing standard IDs are errors, missing power IDs are informational
  hidden/empty sites, and undeclared-mode absence remains a warning. Malformed,
  non-finite, negative, missing, or impossible empty-cell areas/volumes produce
  explicit measure errors; valid measures use a stable sum and closure gaps or
  overlaps are errors. Required reciprocity failures are errors, while optional
  inspection remains informational/warning-level. Strict validation and
  compute warn/raise consume final `diagnostics.ok`, repeated marked analyses
  clear stale boundary flags, and warning-only planar normalized-topology
  findings no longer fail strict validation. Public names, signatures,
  defaults, result fields, forward geometry, and native safety are unchanged.
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
- Separator squared mismatch and the quadratic Huber branch now share
  `0.5 * residual**2`, while L2 regularization is
  `0.5 * strength * ||weights - reference||**2`. The direct dense and sparse
  normal system remains `L_obs + strength * I`, so ordinary squared-loss/L2
  fitted weights are unchanged; reported mismatch/L2 values and previously
  inconsistent ADMM relative scaling are corrected. Scale-safe evaluation now
  preserves finite weighted objective values and the direct quadratic row
  curvature/RHS when unweighted intermediates exceed binary64 range.
- Reciprocal boundary penalties now use a finite convex tangent continuation
  at and below `epsilon`. Zero-strength scalar penalties are exact no-ops for
  evaluation, coupling, backend selection, and quadratic-operator
  availability.
- Hard-bound status uses one shared scale-aware float64 measurement predicate,
  whose accepted interval is mapped into Bellman–Ford difference bounds
  without applying the measurement tolerance in weight-difference units.
  `PowerFitObjectiveBreakdown` and fit-report JSON add
  `hard_max_tolerance`, while `hard_max_violation` remains the raw maximum
  violation. ADMM success also requires final hard-row satisfaction.
- Solver-produced successful results require finite reported soft-objective
  components and totals. The public result builder rejects falsely successful
  non-finite packaging, and a linear-algebra failure in the optional direct
  ADMM warm start falls back to the existing safe initialization. Large finite
  residual and convergence summaries use scale-safe reductions.
- Final quadratic binary64 weights are certified after public gauge
  canonicalization. Exact-zero proof checks source residuals exactly; every
  nonzero direct or quadratic-ADMM candidate requires a conservative
  source-gradient/singular-value forward objective-gap bound. Unsupported
  output resolution produces structured `numerical_failure`. Bounded exact
  helpers apply that same continuous-objective rule: coordinatewise rounding
  of an exact optimum is not treated as a separate binary64-lattice
  certificate, and helper size limits cannot change the meaning of `optimal`.
- Solver method and linear backend are now separate. The default is
  `solver='direct', linear_backend='dense'`; explicit ADMM executes whenever a
  component solve is required, dense routes never import SciPy, and explicit
  sparse routes require SciPy. No-work fits report `solver='none'` and
  `linear_backend=None`. Active-set fitting forwards both selections, and
  results and reports record them separately. Structured ADMM failures retain
  completed iteration counts, including failure of final quadratic
  certification. The earlier development-only values `auto`, `analytic`, and
  solver value `sparse` and the old unprefixed ADMM keyword names are removed.
- Scalar ADMM proximal coordinates with positive-strength penalties now use a
  certified private bounded solver. It preserves the approved objective,
  handles exact branch breakpoints and raw exponential overflow, and succeeds
  only through equality, point-KKT, or adjacent-binary64-bracket evidence.
  Exhaustion returns the existing structured `numerical_failure` rather than
  the last iterate. Mismatch-only and zero-strength rows retain the vectorized
  path, and no public option, result field, dependency, or ADMM default changes.
- Final active-set refits align only true zero-L2 gauge components and only
  when exact binary64-input checks prove that every within-component weight
  difference is unchanged. Positive L2 solutions are no longer shifted toward
  a previous outer iterate after certification.
- Separator observations, fixed fits, realizations, active diagnostics, and
  reports now share a canonical two-layer identity model. Every valid row has
  a deterministic source-independent `row_id`, and every ordered observation
  set has a fingerprint that remains stable if exact source provenance is
  bound later. Resolver-created observations retain exact caller-order points,
  domain representation, dimension/count, and ID provenance. Valid directly
  constructed observations remain unbound and continue through the public
  row-only problem/result/report chain; a first source-aware use may bind them
  only after independently verifying their connector geometry. Bound/unbound,
  different-source, and length-only associations are rejected. Omitted
  `domain=None` preserves an existing domain binding, while it establishes an
  exact no-domain source for an unbound object.
- Observation-aligned record dictionaries add `row_id`. Fit, realized, and
  active reports retain their existing kind and numerical fields and add the
  common schema-1 `schema`, `producer`, `source`, and `observation_set` blocks.
  An unbound source reports null fingerprint, points, domain, and IDs; this is
  distinct from a bound `{"kind": "none"}` domain. Finite reports round-trip
  exactly through strict JSON. Existing active failure placeholders are
  replaced by the R7 availability design described below.
  Exact-key report consumers must accept these additive keys; no public source
  argument, dependency, or native behavior is added.
- Experimental active-set result assembly is now atomic. Weighted final
  `optimal` and `max_iter` fits recompute realization, full-candidate
  diagnostics, residual summaries, and requested tessellation diagnostics from
  the exact accepted weights. No-weights fits retain the final inner status and
  accepted active subset while realization, candidate diagnostics, residual
  summaries, records, and tessellation diagnostics are `None`; an earlier
  realization is never reused. Outer termination remains separate from final
  inner-fit status and convergence. Computed result properties expose final
  availability, its existing fit-status reason, and final-refit convergence.
  Active reports preserve the schema-1 identity envelope, add an
  `availability` block, use JSON null for unavailable weights-dependent
  sections, and round-trip exactly through strict JSON for success and failure.

The external-ID, direct-result-construction, pickle, and Twine-discovery
corrections came from issue #31. The R1–R9 work is tracked explicitly below:
objective (#36), certified scalar proximal solving (#37), strict/native inputs
(#38–#39), periodic images (#40), generator safety (#41), source/report identity
(#42), atomic active state (#43), diagnostics (#44), and public/distribution
contract synchronization (#45). These changes add no legacy scaling mode.
ADMM results can change where the earlier mismatch/L2 relative scaling was
inconsistent or the former scalar loop returned an uncertified iterate.

## Correctness-remediation traceability

This compact ledger points to the accepted implementation history and its
independent regression basis. The detailed contracts remain in the
[audit](../development/audits/v0.8-pre-release.md),
[remediation plan](../development/plans/archive/v0.8-remediation.md), and ADRs.

| Workstream / issue | Implementation reference | User-visible effect | Migration impact | Independent regression oracle |
|---|---|---|---|---|
| R1 / [#36](https://github.com/DeloneCommons/pyvoro2/issues/36) | [`49a06cf`](https://github.com/DeloneCommons/pyvoro2/commit/49a06cf7a65361d309728588e0073f24f7f673b7) | Correct objective scaling, hard tolerance, finite success packaging, quadratic certification, and solver/backend vocabulary | Corrected earlier development values/status; no compatibility mode | Direct objective recomputation, finite differences, KKT/gap bounds, dense/sparse parity |
| R2 / [#37](https://github.com/DeloneCommons/pyvoro2/issues/37) | [`6aec60d`](https://github.com/DeloneCommons/pyvoro2/commit/6aec60dbf54dc193faa4920e1b79d30e6e2e5912) | Certified scalar proximal results or structured `numerical_failure` | No public tolerance or dependency added | Exact/interval derivative signs, adjacent-float objective comparison, scalar KKT cases |
| R3 / [#38](https://github.com/DeloneCommons/pyvoro2/issues/38), [#39](https://github.com/DeloneCommons/pyvoro2/issues/39) | [`e816570`](https://github.com/DeloneCommons/pyvoro2/commit/e816570), [`76a58ec`](https://github.com/DeloneCommons/pyvoro2/commit/76a58ec), [`d24e0ed`](https://github.com/DeloneCommons/pyvoro2/commit/d24e0ed), [`8baf037`](https://github.com/DeloneCommons/pyvoro2/commit/8baf037), [`555228c`](https://github.com/DeloneCommons/pyvoro2/commit/555228c) | Exact input categories, finite checks, owned values, and safe native construction | Invalid coercions now fail early; valid results unchanged | Strict type matrix, caller-mutation isolation, subprocess exception/survival and resource-bound probes |
| R4 / [#40](https://github.com/DeloneCommons/pyvoro2/issues/40) | [`817e222`](https://github.com/DeloneCommons/pyvoro2/commit/817e222) | Certified nearest/minimum image; explicit shifts stay authoritative; no approximate fallback | `image_search` no longer changes correctness | Exhaustive bounded lattice enumeration plus translation/pair-reversal invariants |
| R5 / [#41](https://github.com/DeloneCommons/pyvoro2/issues/41) | [`85e896e`](https://github.com/DeloneCommons/pyvoro2/commit/85e896e) | Half-open containment, periodic remap, non-disableable duplicate floor, native postconditions | Unsafe/out-of-domain inputs now fail instead of reaching native code | Half-open boundary cases, brute-force periodic pair checks, subprocess no-exit probes |
| R6 / [#42](https://github.com/DeloneCommons/pyvoro2/issues/42) | [`4c2b074`](https://github.com/DeloneCommons/pyvoro2/commit/4c2b074) | Stable row/set identity, optional source binding, schema-1 provenance and strict JSON | Additive row/schema provenance; retained report kinds | Mismatch/equivalent-copy matrices, recomputed fingerprints, exact JSON round trips |
| R7 / [#43](https://github.com/DeloneCommons/pyvoro2/issues/43) | [`721e84f`](https://github.com/DeloneCommons/pyvoro2/commit/721e84f) | Atomic final state and explicit unavailable final layers | Additive availability block/properties; null replaces stale/fabricated data | Forced final-refit status matrix and cross-layer state-origin invariants |
| R8 / [#44](https://github.com/DeloneCommons/pyvoro2/issues/44) | [`d34fd11`](https://github.com/DeloneCommons/pyvoro2/commit/d34fd11) | Severity-complete diagnostics and strict/warn/raise behavior | Corrected erroneous success and warning-only planar failure | Explicit 2D/3D issue-severity/mode matrix over independently constructed raw cells |
| R9 / [#45](https://github.com/DeloneCommons/pyvoro2/issues/45) | [`400137c`](https://github.com/DeloneCommons/pyvoro2/commit/400137c9ba3a71b5cad3b87070a48854474bfb88), [`ea3be52`](https://github.com/DeloneCommons/pyvoro2/commit/ea3be52567b3e0156ad12da6e16f780f735aadf3), [`b410b47`](https://github.com/DeloneCommons/pyvoro2/commit/b410b47ad3b2e6c6ab00b1369947b6f449a9b02d) | Synchronized the public API, documentation, platform, licensing, and distribution contract; made raw wheel-member validation portable | Removed stale development claims without adding a compatibility surface | API/search audits, generated-state checks, distribution fixtures, archive inspection, and cross-platform path cases |
| Post-R9 distribution correction | [`a9a66f4`](https://github.com/DeloneCommons/pyvoro2/commit/a9a66f43df20f002ca6435472f224a3cca739f4c) | Wheel and sdist distributions carry and verify the mandatory canonical `COPYING` payload | No API change; completes the accepted license payload | Byte-for-byte wheel, sdist, and installed-package license checks |

The post-R9 reference identifies the accepted baseline for this source-
finalization pass, not the eventual final release commit.

## Known limitations and named deferrals

- Weight-first `locate`/`ghost_cells`, orientation-neutral `PeriodicCell`, and
  explicit fractional and geometric-parallelepiped helpers are post-v0.8 work;
  they are v0.9 candidates rather than v0.8 promises.
- Representation-robust certified minimum-image basis reduction remains later
  work before broad triclinic measure claims. Certified face/edge image labels
  and complete periodic query/ghost metadata remain pre-1.0 work.
- Prescribed cell measures begin in v1.1, followed by mixed separator-plus-
  measure fitting in v1.2.
- Unsupported wheel platforms and free-threaded interpreters are outside the
  v0.8 matrix. Outside-generator clipping/walls, automatic normalization,
  dominated coincident-site preprocessing, backend-basis canonicalization, and
  persistent containers remain later or demand-driven work.
- Binary64/backend limitations and features requiring substantive local Voro++
  source changes are not claimed solved by v0.8.

## Zenodo

pyvoro2 v0.8.0 creates no new pyvoro2 Zenodo software-version record. Existing
historical pyvoro2 records and project/reproducibility Zenodo records remain
valid.

## Packaging and platform support

Supported source builds use standard GIL-enabled CPython 3.10, 3.11, 3.12,
3.13, or 3.14. Package metadata declares `Requires-Python: >=3.10`, so installers
do not impose an artificial upper bound, but versions newer than 3.14 are not
part of the v0.8 tested support contract. Source-install CI exercises all five
supported versions on Linux, macOS, and Windows.

The release artifact target is exactly 21 distributions. Representative local
wheel and sdist checks do not qualify this matrix; issue #33 must build and test
it from the exact frozen v0.8.0 final source commit accepted after source
finalization and independent review:

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

Every wheel also carries `LICENSE`, `COPYING`, `NOTICE.md`, and the complete
upstream Voro++ license as `LICENSE.voro++` in its standards-compatible metadata
license directory. The sdist carries those four root files plus the
byte-identical `vendor/voro++/LICENSE`. Distribution and installed-package
checks require and byte-compare the complete payload. Package metadata
intentionally does not claim `Operating System :: OS Independent`; the
supported wheel matrix above is the platform claim.

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
- The default `pytest -q` run includes seeded fuzz/property tests with
  `--fuzz-n=10`; `pytest -m fuzz --fuzz-n 100` explicitly selects them at a
  higher iteration count. Independent `pyvoro` cross-checks remain optional and
  dependency-gated.
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
