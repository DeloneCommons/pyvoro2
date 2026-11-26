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
retain their backend-oriented meaning and admission requirements.

::: pyvoro2.domains
:::
