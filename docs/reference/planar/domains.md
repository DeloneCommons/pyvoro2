# Planar domains API

`RectangularCell.remap_cart` shares the spatial rectangular numerical policy:
represented `float(hi-lo)` periods, exact affine reduction, rounded-upper
chart transitions at zero epsilon, nonoverlapping snap neighborhoods and final
int64 validation even for unreturned shifts. Inactive axes are unchanged.
See the [remapping guide](../../guide/domains.md#backend-primary-remapping-helpers)
for containment refusals and compatibility details.

::: pyvoro2.planar.domains
:::
