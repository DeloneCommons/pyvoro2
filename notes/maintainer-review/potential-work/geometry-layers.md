# Geometry and graph layers

**Status:** proposed responsibility split; not an accepted public API design.

Keep distinct objects distinct, and avoid making expensive global reconstruction a prerequisite for ordinary forward tessellation.

| Layer | Represents | Primary uses |
| --- | --- | --- |
| **Local cells (core)** | Independent cell-local vertices, edges/faces, measures, neighbor site IDs | Forward computation, simple pyvoro-like workflows |
| **Face adjacency (derived)** | Realized site-pair contacts, face geometry, periodic image shift for each directed contact | Contact graphs, scientific neighbor analysis, realization diagnostics |
| **Global geometric complex (derived)** | Unique geometric vertices/edges/faces, incidence, periodic identifications and winding | Topology of pores/channels, QTAIM-inspired geometric analogues, geometric networks |

The **separator-observation graph** is separate: its nodes are sites and its edges are measured pairwise constraints for inverse fitting. It is not a tessellation-derived adjacency graph and needs no global cell geometry.

## Intended separation

- Raw/local cells must remain usable if adjacency construction or globalization fails.
- Globalization is a **downstream-required capability**, but need not run on every `compute` call.
- Periodic adjacency must distinguish multiple images of the same site; an unordered site pair alone is insufficient.
- Global vertex identity, coordinate closeness, and face incidence are different assertions. Preserve raw geometry and label any repair/approximation.
- First assess how much the current `TessellationResult` and normalization functions already provide; do not invent a second core API.

Follow-up: [global topology research](global-topology.md) · [current result contract](../../../docs/development/decisions/0005-tessellation-result-contract.md).
