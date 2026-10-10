# Optional geometry-output presets

**Status:** low-priority research/usability candidate; **not approved for implementation**, no assigned milestone or release. Revisit during a later API-soaking period if user feedback supports it. No need to include in v0.9.

## User problem

The current forward APIs allow independent control of `return_vertices`, `return_adjacency`, and `return_faces` (or `return_edges` in planar), which avoids costly Python geometry export. But choosing a common output combination requires several boolean arguments; see [measured use cases and exploratory benchmark](../documentation-notes/geometry-export.md).

Consider adding **one optional convenience preset** while retaining the existing fine-grained controls and unchanged defaults.

## Illustrative proposal, not a chosen design

```python
# Illustrative name and values only; these do not exist today.
pv.compute(points, domain=box, geometry_output='none')
pv.compute(points, domain=box, geometry_output='all')
```

Possible contract to evaluate:

| Preset | Optional geometry returned |
| --- | --- |
| `None` (default) | Existing individual `return_*` flags govern output, exactly as in v0.8 |
| `'all'` | Vertices, cell-local vertex adjacency, and boundaries |
| `'none'` | No optional geometry; only mandatory fields such as ID, measure, site |
| Potential `'boundaries'` | Vertices + faces/edges, omitting cell-local vertex adjacency |

When the preset is not `None`, it would **override the three component flags**, even if they were passed explicitly. This precedence rule needs clear documentation and tests; an alternative would be to reject conflicting explicit arguments. Other presets are speculative.

Do **not** call the new parameter simply `mode`: that name already selects `'standard'` versus `'power'` in the public API. Names like `geometry_output` or `geometry` would need a separate naming review.

## Questions before making a decision

- Are two presets (`all` / `none`) enough, or is `boundaries` worth the extra public surface? Should advanced users keep independent booleans?
- For requested periodic face/edge shifts, topology normalization, annotation, or diagnostics that need geometry, should an incompatible preset raise, force additional fields, or be restricted? Do not silently weaken stated contracts.
- Should a preset apply consistently to `compute`, `ghost_cells`, and 2D/3D; what compatibility rules would follow?
- Does the convenience benefit, verified with actual downstream use, justify another keyword and its long-term stability obligations?
- Can API soaking or documentation examples solve the discoverability problem without any new API?

**Decision trigger:** user feedback, observed repetitive call patterns, and regression tests for existing options; not simply the presence of a performance difference. Keep unchanged implementation and defaults until an accepted decision authorizes otherwise.

Relevant current source: [3D `compute()`](../../../src/pyvoro2/api.py#L235-L268), [3D geometry extraction](../../../cpp/bindings.cpp#L31-L138), [2D geometry extraction](../../../cpp/bindings2d.cpp#L29-L94).
