# Domains API

`PeriodicCell` preserves its original finite binary64 row lattice and origin.
Exact nonsingularity of either handedness is the construction criterion;
native execution and backend conversion methods apply separate numerical
admission checks. See the [3D domain guide](../guide/domains.md).

The new user-coordinate methods remain **Provisional** under the API lifecycle
policy; their contract was accepted in
[ADR 0030](../development/decisions/0030-exact-user-lattice-and-native-admission.md)
and implemented in reviewed [PR #156](https://github.com/DeloneCommons/pyvoro2/pull/156):

```python
PeriodicCell.cart_to_fractional(points)
PeriodicCell.fractional_to_cart(fractional)
PeriodicCell.wrap_fractional(fractional, *, return_shifts=False)
PeriodicCell.wrap_cart(points, *, return_shifts=False)
```

All coordinate inputs/outputs have shape `(n, 3)` (including empty batches).
Floors and affine/solve expressions use exact validated binary64 operands;
coordinate outputs are owned nearest-even float64 views, with no snapping.
Optional user-basis shifts are owned signed-int64 arrays. Only requested shifts
are range-limited; nonfinite views and requested shift overflow raise
operation-level `ValueError`. Legacy internal-coordinate and remapping methods
retain their backend-oriented meaning. Native helpers use qualified nonpivoted
QR of `A.T`, support both source signs, and extract the six parameters directly
from the sign-normalized triangular factor. Their numerical admission checks
and `ValueError` failures remain separate from exact source validity. See
[the A2 decision](../development/decisions/0031-qualified-native-periodic-qr-frame.md)
for qualification gates and the source/native approximation boundary.

Backend-primary remapping uses exact dyadic quotient/affine decisions and final
rounded coordinates. Rectangular axes use the represented `float(hi-lo)` period;
native triangular axes commit complete rows in z/y/x order. Rounded upper
endpoints advance the numerical chart even at `eps=0`. Exact `2*eps < D` is
required on each active axis, including empty input. Final shifts must fit
int64 even when not returned. Overlapping snap neighborhoods or an incompatible
rectangular representative raise `ValueError`; admissible anisotropic frames
can therefore refuse default remapping. See the [remapping guide](../guide/domains.md#backend-primary-remapping-helpers)
for examples, fixed-point scope and the distinct A1 contract.

`PeriodicCell.remap_cart` removes the exact source lift before projecting the
exact local Cartesian residual through its one accepted frame. It rounds each
complete dot once, applies the shared native chart, and checks only the combined
source/native shift `K=k+n` at the legacy int64 boundary, including unreturned
shifts. It uses the existing Cartesian reporting transform and finite-view
checks. With the same frame and epsilon, admitted exactly represented source
translations preserve native submissions and add their coefficients to K.
This is conditional on the supplied binary64 operands retaining the translation.
The stable remap need not equal a literal composition of `cart_to_internal`,
`remap_internal` and `internal_to_cart`, or a composition through rounded A1
helpers. Those standalone helpers retain their own semantics. Fractionally
local residuals can still be large in Cartesian coordinates for sheared or
anisotropic bases. Projection rounding before reduction can lose local source
phase and change prepared duplicate decisions even without snapping. See the
[guide example](../guide/domains.md#backend-primary-remapping-helpers).

The inserted-generator path shares this policy; native admission, both safety
stages, source identity and inverse realization semantics remain unchanged.
Uninserted locate queries keep direct projection and are outside this stability
contract. See [the guide](../guide/domains.md#backend-primary-remapping-helpers)
and [accepted ADR 0034](../development/decisions/0034-stable-source-lifting.md).

::: pyvoro2.domains
:::
