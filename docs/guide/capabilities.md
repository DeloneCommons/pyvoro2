# Capabilities and limitations

This page separates current supported behavior from restrictions that may be
removed in a later wrapper release, limits inherited from the backend or
binary64 arithmetic, unsupported geometry, and architecture decisions that are
still open. A limitation in one category does not imply a promise in another.

## Supported v0.8 contracts

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

Some current restrictions are properties of the v0.8 Python wrapper rather
than mathematical impossibilities or permanent backend limits. Examples include
radius-only power input for `locate(...)` and `ghost_cells(...)`, orientation
restrictions on triclinic user bases, and public-output/search conditions used
by periodic boundary image reconstruction.

In particular, certified nearest/minimum-image resolution for separator
observations does not make every boundary-image reconstruction path certified.
The remaining periodic boundary-image and search-window limitations are
candidate v0.9 functional-stabilization work. Current search controls must be
read according to their individual contracts: separator `image_search` is only
a performance seed and never a correctness radius, while current forward
face/edge reconstruction controls may still bound supported reconstruction
behavior.

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
future inverse families or unknown types, with prescribed measures planned for
v1.1 and mixed problems for v1.2.

## Architecture policy

The repository builds one distribution containing both forward native extensions
and the inverse Python layer, and it vendors the upstream Voro++ sources with a
bounded accepted robustness fix.

[ADR 0017](../development/decisions/0017-v0.9-functional-stabilization-before-1.0.md)
now fixes one repository and one distribution through 1.0, with post-1.0
reassessment only under a concrete trigger. The remaining pre-1.0 backend-source
question is whether pyvoro2 should require functionally unmodified upstream
Voro++ source or permit a bounded, explicitly maintained downstream patchset
without becoming an independently evolving backend fork.

That policy is intentionally deferred until WP7 establishes the minimum native
change required for correct ghost-boundary semantics. A clean binding-only fix
does not settle it. If a vendored Voro++ source change is actually needed, the
minimum concrete patch is reviewed under D9 before acceptance. The unresolved
choice is about the scope and maintenance burden of downstream divergence, not
about declaring every long-lived patch a functional fork merely because it is
not temporary or upstream-accepted.

For exact callable behavior, see [Choosing an API](choosing-api.md), the
[API reference](../reference/index.md), and the
[v0.8 API inventory](../development/api-inventory.md).
