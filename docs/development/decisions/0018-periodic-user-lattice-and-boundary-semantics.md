# 0018 — User-lattice periodic semantics and certified boundary identity

- **Status:** Accepted
- **Date:** 2026-09-01
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

Approximate Cartesian image positions returned by native code may be converted
to a public integer lattice translation only when exactly one user-lattice
integer shift is compatible with the returned geometry within the declared
native numerical envelope. No nearest-residual or rounded-inverse guess is a
semantic fallback. Zero compatible shifts are inconsistent; multiple compatible
shifts are ambiguous. Both cases fail structurally when certified metadata is
required.

Boundary-plane compatibility is relative to the binary64 geometry actually
returned by the backend. In power mode, certification uses the exact binary64
radii sent to Voro++ and an envelope covering the backend's radius-squaring and
plane-offset arithmetic. The original mathematical weights remain the public
scientific representation.

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

A boundary receives semantic identity only after its measure is certified
positive within the declared envelope. A native face/edge certified to have zero
measure is dropped from the public packaged boundary list and from normalized
semantic topology; v0.9 adds no separate public raw-zero boundary channel. If
zero versus positive measure cannot be resolved within the envelope, an
operation requesting certified semantic metadata reports structured ambiguity
rather than assigning an owner by residual ordering.

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

Rejected. A zero-measure native artifact has no supported topological boundary
meaning. v0.9 drops it from packaged/normalized boundary lists instead of
inventing a public raw-artifact schema.
