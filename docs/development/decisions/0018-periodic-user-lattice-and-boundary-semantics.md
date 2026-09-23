# 0018 — User-lattice periodic semantics and certified boundary identity

- **Status:** Accepted
- **Date:** 2026-09-01
- **WP5 clarification:** 2026-09-23; [ADR 0021](0021-wp5-native-occurrence-and-exact-face-certification.md)
- **Related issue:** [#46 — Activate the v0.9.0 functional/API stabilization plan](https://github.com/DeloneCommons/pyvoro2/issues/46)
- **Related plan:** [active v0.9.0 development plan](../plans/v0.9.md)
- **Related decisions:** [ADR 0002](0002-weights-radii-and-gauge.md),
  [ADR 0012](0012-certified-periodic-image-geometry.md),
  [ADR 0013](0013-central-generator-preparation-and-backend-safety.md),
  [ADR 0016](0016-severity-complete-tessellation-diagnostics.md), and
  [ADR 0017](0017-v0.9-functional-stabilization-before-1.0.md)

## Context

The v0.8 periodic implementation contains several notions that are individually
useful but not yet separated strongly enough for a pre-1.0 public contract:
the caller's lattice basis, the lower-triangular frame used by Voro++, and the
basis used by exact proof-oriented geometry. The current implementation also
mixes user-cell wrapping with backend-primary remapping in places, rejects
left-handed bases for representation reasons, reconstructs some periodic
boundary images through finite search windows, and does not have a reliable
semantic identity for every periodic ghost boundary.

The active v0.9 plan fixes the mathematical and public semantics before those
implementations are changed. This ADR is therefore a target contract for v0.9
work; acceptance of the ADR does not claim that the unchanged v0.8 production
code already implements it.

## Decision

### Three periodic coordinate layers are distinct

Let the caller-supplied lattice vectors be the rows of `A`, with origin `o`.
User fractional coordinates are defined by

```text
x = o + f @ A.
```

This **user lattice** owns public Cartesian/fractional conversion, wrapping into
the caller's fundamental parallelepiped, public integer shifts, and handedness.
User vector order is never changed merely to satisfy a backend convention.

The **backend periodic frame** is a separate representation used to call Voro++.
The preferred centralized construction is the sign-normalized factorization

```text
A.T = Q @ R
L = R.T = A @ Q,
```

with positive diagonal entries in `L`. `Q` may be proper or improper: a
left-handed user basis is valid and produces the corresponding orientation of
the orthogonal transform. Existing `cart_to_internal()`, `internal_to_cart()`,
`remap_internal()`, and `remap_cart()` retain backend-frame/backend-primary
meaning; they are not renamed into user-parallelepiped operations.

The **exact proof basis** is private. Proof-oriented code may use

```text
A_reduced = U @ A
```

for an exact integer unimodular matrix `U`. It does not change the user basis or
the basis sent to Voro++. Integer shifts map exactly as

```text
s_user = s_reduced @ U
s_reduced = s_user @ U_inverse.
```

Private exact arithmetic may use Python integers. Public shifts are materialized
only after mapping back to the user basis and must satisfy the signed-int64
public representation contract.

WP4 implements this private proof layer for minimum images and triclinic
duplicate buckets using the accepted WP3 reducer. User coordinate/wrapping and
backend-frame paths still use the user basis. Distance-only duplicate checks
consume exact squared distance before any optional diagnostic float view and
require no fixed-width shift.

### Mathematical validity is orientation-neutral

A `PeriodicCell` basis is mathematically non-degenerate when its finite binary64
components have an exact non-zero determinant. A negative determinant means a
left-handed basis, not invalid input.

Conditioning, backend representability, and proof workload are separate
properties. A mathematically valid cell may therefore construct successfully
and later produce a warning, a backend-representability failure, or a
certified-geometry resource failure at the operation that needs the relevant
representation. Such failures must not be relabelled as lattice degeneracy.

### User wrapping has an exact discrete authority

The user-parallelepiped wrap contract is

```text
f_original = f_wrapped + shift
x_original = x_wrapped + shift @ A.
```

The integer `shift` is decided from the exact dyadic values represented by the
binary64 basis, origin, and Cartesian input (or directly by the exact dyadic
fractional input for fractional wrapping). A rounded inverse followed by
`floor`, epsilon snapping, or nearest-integer repair is not the authority for
that discrete decision.

`f_wrapped` and `x_wrapped` are binary64 views of the exact result. If an exact
interior remainder rounds to a binary64 value equal to an upper endpoint, that
rounding does **not** change the exact wrap shift or authorize epsilon-selected
movement to another image. The returned float view must instead satisfy the
operation's documented reconstruction envelope. Exact seam points still follow
the deterministic half-open convention.

### All public periodic shifts use one user-basis sign convention

The following equations define the public sign convention wherever the
corresponding metadata is exposed:

```text
minimum image:
    displacement = p_j - p_i + image_shift @ A

query wrapping:
    query = query_wrapped + query_shift @ A

located owner image:
    owner_pos = owner_site + owner_shift @ A

persistent generator boundary:
    boundary_image = generator_site + shift @ A

ghost self-image:
    boundary_image = ghost_site + shift @ A
    shift != 0
```

`owner_site`/`generator_site` refer to original caller-supplied persistent
coordinates. `ghost_site` is the active backend-primary/cell-site Cartesian
representative that anchors the returned ghost geometry; it is not the original
unwrapped query. A non-periodic wall has no periodic shift.

Equivalent exact unimodular user bases may therefore return different integer
tuples, but those tuples must reconstruct the same qualified physical image.
Private reduced-basis coefficients are never public output.

### Native Cartesian translations require unique certification

This section describes the generic WP4 native-translation consumer and the
earlier cross-work-package target. For ordinary persistent **3D faces**, the
closed WP5 contract is [ADR 0021](0021-wp5-native-occurrence-and-exact-face-certification.md):
source-complete producer-compatible attribution of observed native support N,
followed independently by native-effective exact ideal E and public-semantic
exact ideal S reconstruction. Both E and S must be positive; disagreement
fails certification. An enclosing numerical box alone
is not the final WP5 candidate predicate or semantic positivity test. This
clarification does not change WP4's existing explicit-box consumer or decide
WP6/WP7 producer contracts.

Approximate Cartesian image positions returned by native code may be converted
to a public integer lattice translation only when exactly one user-lattice
integer shift is compatible with the returned geometry within the declared
native numerical envelope. No nearest-residual or rounded-inverse guess is a
semantic fallback. Zero compatible shifts are inconsistent; multiple compatible
shifts are ambiguous. Both cases fail structurally when certified metadata is
required.

WP4 supplies the reusable private consumer for an explicit exact closed
Cartesian compatibility box. Exact reduced inverse-column extrema enclose all
integer candidates, which are filtered against the original Cartesian box.
Its candidate ceiling is 1,000,000 per call; admitted enumeration completes
before unique/inconsistent/ambiguous outcome selection. Resource and invariant
errors remain separate. No universal numerical tolerance or producer envelope
is claimed, and the kernel does not select the reference anchor. Deriving the
WP5 source-compatible producer attribution is closed in ADR 0021; its
implementation and the separate WP6–WP8 producer integrations remain work.

For WP5 the actual binary64 normal/offset and source arithmetic identify
producer-compatible images at N; exact E uses actual native radii, while
independent exact S uses public mathematical weights. The backend radii and
their source-associated arithmetic do not become the public semantic weights.
See ADR 0021 for the closed 3D
contract; producer contracts for other work packages remain separate.

### Ghost boundary identity is a tagged public record

For every certified positive-measure ghost face/edge whose boundary identity is
returned, the exact v0.9 public spelling is the nested record
`boundary_reference`:

```text
{
    "kind": "generator" | "ghost_self" | "wall",
    "generator_id": int | None,
    "shift": tuple[int, ...] | None,
    "wall_id": int | None,
}
```

The fields are validated together:

| `kind` | `generator_id` | `shift` | `wall_id` |
|---|---|---|---|
| `generator` | required | required for a periodic domain; otherwise `None` | `None` |
| `ghost_self` | `None` | required, user-basis, non-zero | `None` |
| `wall` | `None` | `None` | required where the domain exposes wall identity |

This record is the ghost semantic authority. No new top-level
`BoundaryReference` class/export is introduced by this decision. Existing
ordinary `compute()` adjacency fields need not be replaced for schema symmetry.
For ghost boundaries, `adjacent_cell` may remain only where it already has a
well-defined persistent-generator or wall compatibility meaning; it must never
contain an undefined temporary native ghost ID.

The earlier proposed rule to delete certified-zero native faces from public
output is **superseded for ordinary persistent 3D face certification by
ADR 0021**. For that WP5 path, independent E/S exact ideal affine dimensions
determine positive/zero/absent; their disagreement is a representation conflict,
and zero, absent, unresolved or other certificate failure fails the entire
certified call without deleting any native face. An ordinary non-certified
result has its own numerical validity policy.

Candidate boundary assignments are equivalent only when they define the same
exact cut **and** the same semantic provenance (`kind`, owner where applicable,
and user-basis shift). Different owners, shifts, or kinds require trusted
disambiguating evidence or structured ambiguity/multiplicity handling.

### Structural failures are not heuristic answers

Private basis reduction, proof-complete candidate enumeration, and certified
native-translation recovery may have explicit resource limits. Exceeding such a
limit is a structured resource failure. Backend inability to represent a valid
user lattice is a backend-stage failure. Neither case authorizes an uncertified
answer, a finite-search correctness knob, or relabelling the input lattice as
mathematically invalid.

Separator `image_search` remains the correctness-neutral incumbent seed defined
by ADR 0012. Face/edge reconstruction search, validation, repair, and matching
tolerance parameters lose their public correctness role when certified boundary
reconstruction replaces them, as specified by the active v0.9 plan.

## Consequences

- Public shift tuples have one reconstruction meaning independent of the
  backend frame and private proof basis.
- Left-handed non-degenerate user cells become valid inputs without reordering
  user lattice vectors.
- User-parallelepiped wrapping can be tested at exact seams and difficult
  binary64 rounding cases without an epsilon changing image identity.
- Periodic face/edge/ghost reconstruction must certify completeness and
  uniqueness rather than search an arbitrary coefficient cube.
- Ghost self-images are first-class semantic boundaries instead of being
  overloaded into native integer neighbor IDs.
- Degenerate native boundary artifacts cannot silently enter semantic topology.
- Some mathematically valid poor representations may fail honestly because of
  backend or proof-resource limits; this is preferable to returning an
  uncertified image.

## Alternatives considered

### Make the backend lower-triangular cell the public periodic basis

Rejected. It makes caller vector order, handedness, and public integer shifts
depend on an implementation representation rather than the supplied lattice.

### Canonicalize or reduce the public lattice

Rejected. Private reduction is proof machinery. A public canonical/reduced-cell
API is outside v0.9 and would change the meaning of caller-supplied coefficients.

### Keep epsilon snapping as user-wrap authority

Rejected. It makes the discrete image identity depend on an arbitrary tolerance
and can disagree for equivalent exact inputs near a seam.

### Treat native neighbor integers as ghost boundary identity

Rejected. Current native routes cannot reliably distinguish persistent
neighbors, ghost self-images, and walls, and one 3D route can read an
uninitialized temporary ID before Python normalization.

### Preserve certified-zero boundaries as semantic records

Rejected as an automatic semantic-boundary rule. The former proposal to drop
zero-measure native artifacts from packaged output is superseded for WP5 by
ADR 0021: an exact-zero/absent mismatch fails the certified call atomically.
The planar and ghost packages must specify their own policy.
