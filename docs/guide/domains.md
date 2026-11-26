# Domains (containers)

> This page focuses on the **3D** top-level domains (`pyvoro2`). For the
> current **2D** scope, see the [Planar (2D)](planar.md) guide and
> `pyvoro2.planar`.

A Voronoi or power/Laguerre tessellation is always defined **inside a domain**:

- for a finite cluster you typically want an explicit boundary (a box),
- for a crystal you typically want periodic boundary conditions,
- for slabs/wires you may want periodicity only in one or two directions.

In Voro++ terminology, these are different *containers*. In pyvoro2 they are exposed as
small Python dataclasses.

This page describes the **3D** domain classes on the top-level `pyvoro2` API.
For the dedicated 2D surface, see [Planar 2D](planar.md).

## Choosing a domain

A practical rule of thumb:

- Use **`Box`** when you want a finite system and you care about boundary faces.
- Use **`OrthorhombicCell`** when your cell is axis-aligned and you want periodicity in
  **some** axes (slabs and wires are the most common cases).
- Use **`PeriodicCell`** when you have a general triclinic unit cell (fully periodic).

## `Box` (finite, non-periodic)

`Box` represents an axis-aligned rectangular domain with *no periodicity*.
Cells are clipped to the box.

```python
from pyvoro2 import Box

box = Box(((-5, 5), (-5, 5), (-5, 5)))
```

This is the most straightforward setting for:

- molecules and clusters,
- toy problems and method development,
- any workflow where “outside the box” should be treated as an explicit boundary.

## `OrthorhombicCell` (axis-aligned, optionally periodic)

`OrthorhombicCell` represents an axis-aligned cell that can be periodic in any subset
of axes. This is a natural geometry for:

- 2D periodic slabs: periodic in $(x,y)$ but not in $z$,
- 1D periodic wires: periodic in $x$ only,
- fully periodic orthorhombic crystals.

```python
from pyvoro2 import OrthorhombicCell

# Fully periodic orthorhombic unit cell
cell = OrthorhombicCell(((0, 10), (0, 10), (0, 10)), periodic=(True, True, True))

# Slab geometry: periodic in x and y only
slab = OrthorhombicCell(((0, 10), (0, 10), (0, 30)), periodic=(True, True, False))
```

Notes:

- Internally this uses the standard Voro++ rectangular container with per-axis periodic flags.
- `return_face_shifts=True` is supported whenever at least one axis is periodic.
- For non-periodic axes, query points outside the bounds are treated as out-of-domain.

Robustness note:

- pyvoro2 vendors a Voro++ snapshot that includes the upstream numeric robustness fix for fully
  periodic **power/Laguerre** tessellations (radical pruning).
- If `return_face_shifts=True` fails with a `ValueError`, it is typically due to a
  genuine geometric degeneracy (for example, nearly co-spherical sites) rather than
  a platform-specific issue.

## `PeriodicCell` (triclinic, fully periodic)

`PeriodicCell` represents a fully periodic triclinic unit cell defined by three vectors
$a, b, c$ (in Cartesian coordinates). This is the most general setting for crystals.

```python
from pyvoro2 import PeriodicCell

cell = PeriodicCell(
    vectors=((10.0, 0.0, 0.0), (2.0, 9.0, 0.0), (1.0, 0.5, 8.0)),
    origin=(0, 0, 0),
)
```

All domain constructors require finite real bounds, vectors, and origins.
Bounds, vectors, origins, and periodic flags are copied into canonical nested
Python tuples, so later mutation of a caller list or array cannot change the
domain. Periodic flags must be actual Python or NumPy Booleans. A
`PeriodicCell` accepts either handedness when the determinant of the validated
binary64 rows is **exactly nonzero**. It preserves the supplied row order,
signs, values and origin. Floating determinant overflow/underflow and poor
conditioning do not invalidate this user lattice.

Native use has a separate, narrower boundary. Both handedness signs are
supported through a qualified orthogonal QR frame, with positive native diagonal
lengths. Numerical row lengths and their product must be finite and positive,
and the floating determinant must be finite, nonzero and agree with the exact
source sign. Relative volume must be at least `1e-12`, and condition number at
most `1e15`. It warns with
`RuntimeWarning` above `1e10` **when preparing the native frame**, rather than
at construction. Native parameters, containment, duplicate, arithmetic and
allocation guards still apply. `compute`, `locate`, `ghost_cells`, detached
snapshots and the existing internal-coordinate/remapping methods all retain
these restrictions. They can raise an operation-level `ValueError` for a valid
user lattice; successful construction does not promise native execution.

`Box.from_points(...)` validates a non-empty finite point matrix before taking
minimums or maximums. Padding must be finite and non-negative. Zero padding is
valid only when the points still give strictly ordered bounds on every axis.

### Constructing from Voro++ parameters

Voro++ also describes triclinic periodic cells using six lower-triangular parameters:

- `bx, bxy, by, bxz, byz, bz`

If you already have these numbers (for example from an existing Voro++ workflow), you can
construct the same cell directly:

```python
cell = PeriodicCell.from_params(bx, bxy, by, bxz, byz, bz)
```

The diagonal lengths `bx`, `by`, `bz` must still be strictly positive;
shear parameters may have either sign. Numerical native admission is deferred
to use, just as for the row constructor.

### Exact user-coordinate helpers (provisional)

With the original ordered rows `A` and origin `o`, use:

```python
f = cell.cart_to_fractional(points)       # exact (x-o) @ inverse(A), rounded
x = cell.fractional_to_cart(fractional)   # exact o + f @ A, rounded
r, s = cell.wrap_fractional(fractional, return_shifts=True)
w, s = cell.wrap_cart(points, return_shifts=True)
```

All inputs must be finite real arrays of shape **(n, 3)**, including `(0, 3)`.
There is no one-point vector shortcut or broadcasting. Boolean, complex,
string and nonfinite coordinates are rejected. Each operation interprets the
validated binary64 values exactly, including values rounded during input
coercion. Returned coordinate arrays are new owned float64 arrays, rounded
componentwise to nearest-even.

Fractional wrapping computes exact `s = floor(f)` and `r = f - s`.
Cartesian wrapping solves exact `f = (x-o) @ inverse(A)`, takes that exact
floor, then rounds exact `w = x - s @ A`. Shifts refer to the original user
basis. The Cartesian operation does not compose separately rounded converters.
No epsilon, native frame, Voro++ extension or native admission is needed.

An exact fractional remainder lies in `[0, 1)`, but its float view can be
`1.0`. For example, wrapping `[-2**-55, 0, 0]` fractionally returns
`[1.0, 0, 0]` with shift `[-1, 0, 0]`. Similarly, for `A=I`,
`o=[2**-55, 0, 0]`, `x=[1, 0, 0]`, the rounded fractional coordinate is one
while the exact Cartesian wrapping shift is zero. Views are not snapped,
clamped or rewrapped. Float-view idempotence and equivalence to composing the
public helpers are not unconditional contracts.

Exact reconstruction gives `f = r + s` and `x = w + s @ A` before view
rounding. Reconstruction from returned views includes that rounding error;
ordinary NumPy arithmetic introduces its own additional error.

`return_shifts` must be a Python/NumPy Boolean. When true, shifts are new owned
int64 arrays, range-checked before casting. When false (the default), shifts
remain arbitrary-size integers internally: for `A=I`, `x=[2**63, 0, 0]`
wraps to zero, but requesting its shift raises `ValueError`. A required
nonfinite float view also raises `ValueError` for that operation, without
invalidating the lattice or partially modifying inputs.

The exact periodic-image kernel supports both signs and signed diagonal rows
for fully periodic 3D source bases, including mandatory duplicate scans.
Planar and partially periodic restrictions are unchanged. Native coordinates
use `y=(x-o) @ Q` and report back with `x=o+y @ Q.T`, where QR of `A.T`
produces a positive-diagonal lower-triangular `L`. Left-handed input uses a
reflection; reported face cycles and cyclic vertex adjacency preserve the
established Cartesian orientation.

Frame qualification checks numerical orthogonality and row-relative backward
error. It does not bound every short-direction relative error or guarantee a
native tessellation. The same integer tuple `s` labels source and native images,
but `L ≈ A @ Q` does not certify exact equality of their Cartesian translations
or native insertion provenance. Existing source snapping, coupled remapping
and endpoint/subnormal limitations remain separate A3 work. See the
[accepted A1 contract](../development/decisions/0030-exact-user-lattice-and-native-admission.md)
and [Proposed A2 decision](../development/decisions/0031-qualified-native-periodic-qr-frame.md).

## Backend-primary remapping helpers

Both periodic domain classes provide remapping utilities that appear in several workflows:

- `remap_cart(points, return_shifts=True|False)`

They wrap coordinates into the primary cell and optionally return the integer lattice shift.

Point arrays and an explicit `eps` are validated before flooring. `eps` must
be finite and non-negative, and `return_shifts` must be a Python or NumPy
Boolean. A remap whose lattice-shift quotient cannot fit signed int64 raises
`ValueError` before integer conversion; the established shift-sign convention
is unchanged.

These helpers are used internally (for example, in visualization wrapping and in periodic graph work),
but they are also useful when you want to align your own data to the primary cell.

### A note on `PeriodicCell.remap_cart` vs the geometric parallelepiped

For `OrthorhombicCell`, “the primary cell” is unambiguous.

For `PeriodicCell`, Voro++ works in a *lower-triangular internal representation*
(`bx, bxy, by, bxz, byz, bz`). The method `PeriodicCell.remap_cart(...)` is
defined by that numerical backend convention, including its existing snapping
rules. It is distinct from the exact-decision `wrap_cart` above.

This means that wrapping via `remap_cart` does not always coincide with wrapping
into the **geometric parallelepiped** spanned by the user vectors.

For visualization, `pyvoro2.viz3d.view_tessellation(..., wrap_cells=True)` wraps
sites into the geometric parallelepiped, because that is typically what users
expect to see.
