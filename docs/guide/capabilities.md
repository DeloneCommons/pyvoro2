# Capabilities and limitations

This page separates current supported behavior from restrictions that may be
removed in a later wrapper release, limits inherited from the backend or
binary64 arithmetic, unsupported geometry, and architecture decisions that are
still open. A limitation in one category does not imply a promise in another.

## Current contracts and published baseline

The current development line starts from v0.8 and includes the separately
accepted periodic-foundation changes in
[ADRs 0030–0034](../development/decisions/index.md). A development-version
label alone does not extend any numerical guarantee.

pyvoro2 supports bounded 3D boxes, partially or fully periodic orthorhombic 3D
cells, fully periodic triclinic 3D cells, bounded planar boxes, and rectangular
periodic planar cells. Standard and power/Laguerre `compute(...)` calls share an
input-aligned `TessellationResult`, with explicit differences where a dimension
or domain cannot provide the same optional geometry.

For every non-periodic axis, each generator inserted into a native container
must lie in the half-open storage interval `[lo, hi)`. Periodic coordinates are
remapped before insertion. This containment rule also applies to temporary
ghost generators, while `locate(...)` query points are not generators. It is a
supported storage-domain contract for the current wrapper and backend path; it
is **not** the mathematical claim that a generator outside a clipped domain can
never affect the clipped diagram.

The mandatory duplicate floor is always enforced before native construction
and cannot be disabled. The user-configurable duplicate policy is a separate,
optional diagnostic layer above that floor. Periodic mandatory checks use a
certified minimum image.

Separator observations support source-independent row/set identity and optional
exact source binding. An explicitly supplied periodic image shift is
authoritative; an omitted shift is inferred with a certified nearest-image
solve. Algebraic fit, realized geometry, and the empirical active-set outer
algorithm remain separate result layers.

## Wrapper restrictions and stabilization candidates

Some restrictions reflect the present Python/native interfaces rather than
mathematical impossibilities or permanent backend limits. For example, power
`locate(...)` and `ghost_cells(...)` still use radius-based input. Under
[ADR 0030](../development/decisions/0030-exact-user-lattice-and-native-admission.md),
`PeriodicCell` construction accepts finite, exactly nonsingular binary64 row
lattices of either handedness. The native adapter now supports both signs
through qualified QR and reflection-safe reporting, subject to its numerical
and native admission safeguards. This is distinct from exact user-lattice
validity; see the [domain guide](domains.md).
Periodic boundary-image reconstruction also retains its existing output/search
limitations.

In particular, certified nearest/minimum-image resolution for separator
observations does not make every boundary-image reconstruction path certified.
The remaining periodic boundary-image and search-window limitations are
candidates for separately approved future work. Current search controls must be
read according to their individual contracts: separator `image_search` is only
a performance seed and never a correctness radius, while current forward
face/edge reconstruction controls may still bound supported reconstruction
behavior.

Query operations need distinct caller precautions:

- In both dimensions, `locate` now conservatively checks the complete batch
  for unsafe query-dependent native remapping, worklist/search indices and
  integer owner-image corrections. A query whose arithmetic cannot be admitted
  raises `ValueError`; admitted queries retain the existing numerical path.
  Keep queries near the domain: finiteness and generator admission do not qualify
  arbitrary query magnitudes or guarantee ownership accuracy.
- A 3D `PeriodicCell` ghost batch can still retain temporary periodic images
  between queries. Separate singleton calls avoid that batch-state defect, but
  do not initialize the temporary native ID or establish complete ghost-image
  labels. Both defects remain open.

See the [operation guide](operations.md#current-locate-query-restriction) for
the bounded locate admission and remaining query precautions, and the
[ghost-specific restrictions](operations.md#current-ghost-batch-and-neighbor-label-restrictions)
for the current workarounds. Locate admission does not certify every inherited
native operation, remote-query recovery or extreme-power arithmetic.

The [roadmap](../project/roadmap.md) preserves these candidates without
promising their exact issue grouping or implementation.

## Backend and binary64 limits

Voro++ and the current wrappers evaluate geometry with finite binary64
arithmetic. Very large or small coordinate scales, nearly coincident sites,
nearly degenerate cells, and power diagrams whose squared backend radii or true
weight range overwhelm squared geometric scales can lose resolution. Finite
input and successful weight-to-radius conversion are necessary preconditions,
not a guarantee that the native tessellation is geometrically resolvable.

These numerical limits are distinct from Python API restrictions. pyvoro2
rejects invalid or unsafe input before native construction where it has a
defined precondition, and reports structured failures where a certified
periodic or scalar solve exhausts resources; it does not silently substitute an
approximation. There is no universal scale cutoff that can replace
problem-specific diagnostics.

## Unsupported geometry

The current package does not support arbitrary wall-defined or unbounded
domains, partial triclinic periodicity in 3D, oblique periodicity in 2D,
anisotropic or non-Euclidean diagrams, or generators outside a non-periodic
storage domain. Supporting some of these would require a new public geometry
contract or substantive backend work, not merely a larger search radius.

Prescribed cell-measure inversion, mixed separator-plus-measure fitting, and
moving-site optimization are also not v0.8 capabilities. They are separate
research candidates without approved implementation or release targets.
Prescribed measures and mixed inversion also have no approved public API,
architectural relationship, or pre-1.0 inclusion; see
[ADR 0027](../development/decisions/0027-v0.9-reboot-from-v0.8.md).

## Open architecture decisions

The repository currently builds one distribution containing both forward
native extensions and the inverse Python layer, and it vendors the upstream
Voro++ sources with a bounded accepted robustness fix. Those are current facts,
not permanent governance decisions.

[ADR 0029](../development/decisions/0029-image-qualified-geometry-and-optional-global-complex.md)
selects independent local cells, image-qualified observed contacts and optional
global assembly. Concrete production contracts and release scope remain
undecided under [ADR 0027](../development/decisions/0027-v0.9-reboot-from-v0.8.md).
Permanent package
and backend policies, including distribution boundaries and treatment of
functional Voro++ forks, also require separate maintainer decisions.

For exact callable behavior, see [Choosing an API](choosing-api.md), the
[API reference](../reference/index.md), and the
[v0.8 API inventory](../development/api-inventory.md).
