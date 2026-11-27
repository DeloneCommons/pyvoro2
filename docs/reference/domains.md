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

::: pyvoro2.domains
:::
