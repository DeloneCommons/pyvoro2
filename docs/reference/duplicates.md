# Duplicate check

Forward `compute`, `locate`, and `ghost_cells` always enforce a private
backend-safety floor before native insertion. The standalone helpers documented
below remain explicit user-threshold diagnostics; the forward
`duplicate_check`, `duplicate_threshold`, and `duplicate_wrap` arguments cannot
weaken mandatory safety.

With a periodic domain and `wrap=True`, the standalone diagnostic remaps the
points first and evaluates certified minimum-image distances on those prepared
Cartesian operands. It is not a certificate of the original source-point
separation. Snapping, projection and Cartesian reporting can change distances.
In particular, `PeriodicCell` source lifting can create a fractionally local
but Cartesian-large residual whose projection loses detail before reduction;
source-distinct points may become coincident in the prepared view. The
mandatory insertion checks protect the actual prepared/native geometry and
can conservatively refuse such configurations. `wrap=False` instead measures
unwrapped Cartesian distances, not exact original-source periodic distances.
See [backend-primary remapping](../guide/domains.md#backend-primary-remapping-helpers)
for the numerical policy and an example.

::: pyvoro2.duplicates
:::
