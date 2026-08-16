# Diagnostics

Spatial diagnostics use error severity as the final failure policy. Missing or
invalid non-empty cell volumes and valid-measure closure gaps/overlaps are
errors. Missing expected IDs are errors in standard mode, informational hidden
cells in power mode, and warnings when `mode=None`. A requested periodic
reciprocity check is required. See [Topology and neighbor
graphs](../guide/topology.md#diagnostics-catching-subtle-issues-early) for the
cross-dimensional policy.

::: pyvoro2.diagnostics
:::
