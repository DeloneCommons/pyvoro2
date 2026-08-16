# Planar normalization validation API

For normalized topology, warning-only `LOW_VERTEX_INCIDENCE` and
`BAD_POLYGON_COUNT` findings can leave their descriptive subchecks false while
overall `ok` remains true. Strict normalized validation raises only for an
error-severity finding.

::: pyvoro2.planar.validation
:::
