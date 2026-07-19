# Common tessellation result

`TessellationResult` is the shared structured data contract for spatial and
planar tessellations. The class and its internal construction path are
available now, but the public `compute()` return migration belongs to issue
#10. At this stage, both compute functions retain their existing raw return
behavior, and planar `PlanarComputeResult` remains a separate class.

Direct construction is supported and validates the raw records against all
aligned fields, including the exact weight/shift-to-radius relationship for
weight-first power input. The aligned measure and empty-state arrays are
snapshots: later mutation of the shared raw dictionaries does not update those
arrays, while boundary access revalidates mutable boundary records against the
snapshot before returning them. Empty cells always expose empty boundary
collections; adding edge or face records to a shared empty raw cell makes
boundary access raise. Deep copies and pickle round trips retain the read-only
aligned-array and capability contracts without reinterpreting later permitted
raw-record mutation.

::: pyvoro2.result.TessellationResult
:::
