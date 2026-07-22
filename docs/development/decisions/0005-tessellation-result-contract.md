# 0005 — Common tessellation result contract

- **Status:** Accepted
- **Date:** 2026-07-17
- **Refined by:** [ADR 0006 — v0.8 cleanup release](0006-v0.8-cleanup-release.md)
- **Related plan:** [v0.7 development plan](../plans/archive/v0.7.md)

## Context

The v0.6.3 spatial and planar APIs expose the same mathematical kind of object
through different return conventions:

- spatial `compute(...)` normally returns raw cell dictionaries, optionally
  paired with diagnostics;
- planar `compute(...)` can additionally return `PlanarComputeResult`;
- downstream code must know dimension-specific return details to recover IDs,
  measures, boundaries, periodic shifts, diagnostics, and normalization output.

chemvoro and later inverse methods need one clear result contract. Separate
public result classes would duplicate most fields and force downstream code to
branch on class names or depend on another protocol. A list-like wrapper that
silently impersonates the historical raw return would make type and mutation
semantics harder to understand.

## Decision

v0.7 introduces one public dimension-neutral class:

```text
pyvoro2.TessellationResult
```

Both `pyvoro2.compute(...)` and `pyvoro2.planar.compute(...)` return this class
by default.

The explicit return selector is:

```python
compute(..., output='result')  # preferred; default
compute(..., output='cells')   # explicit raw-output route
```

The implementation may use an internal sentinel while migrating older keyword
combinations, but the documented v0.7 contract uses `output`.

### Common result responsibilities

`TessellationResult` provides one aligned access contract for:

- dimension and domain;
- computation mode;
- input site coordinates and external IDs;
- raw cell records;
- cell measures aligned with input-site order;
- empty-cell state aligned with input-site order;
- mathematical input weights when supplied;
- backend radii and the common representation shift when applicable;
- tessellation diagnostics when requested or computed;
- normalized vertices/topology when available;
- access to realized boundary data and periodic image labels when the required
  geometry was requested.

The precise full list of stable and provisional fields is maintained in the
[v0.7 API inventory](../api-inventory.md). Dimension-specific raw geometry and
normalization types remain explicit; a common class does not imply false
capability parity.

### Raw-output behavior

For v0.7:

- `output='cells'` preserves the established raw list/tuple behavior;
- planar `return_result=` remains as a compatibility-only alias where practical;
- the interaction of `return_diagnostics` with `output='cells'` preserves the
  historical tuple route;
- `PlanarComputeResult` becomes a compatibility alias to
  `TessellationResult`;
- migration documentation explains the new default and raw-output path.

Although `output='cells'` was introduced as the direct migration path from the
historical default, ADR 0006 retains it as a supported explicit low-level mode
rather than scheduling it for automatic removal with the v0.7-only shims.

`TessellationResult` should not implement list-like mutation or pretend to be a
`list`. Callers that need raw records use `result.cells` or `output='cells'`.

### Immutability policy

The preferred result is structurally immutable when this is simple and honest:

- use a frozen, slotted data container if it does not complicate construction or
  extension;
- arrays created specifically for the result should be read-only where this does
  not require surprising copies;
- metadata should not be silently replaced after construction.

Deep immutability is not a v0.7 release requirement. In particular:

- nested raw cell dictionaries are not copied or recursively frozen solely to
  claim immutability;
- expensive defensive copying is not required;
- implementation complexity that does not improve the public contract should be
  deferred.

The documentation must state clearly which contained objects are mutable. API
clarity and correct semantics take precedence over enforcing deep immutability.

### Cost and optional data

Constructing the result must not silently compute expensive geometry that the
caller did not request. Optional data is represented as absent and accompanied
by clear capability checks or `require_*` helpers where useful.

Cheap aligned views such as cell measure and empty-cell masks may be assembled
from returned records because they are part of the common scientific contract.

## Consequences

- 2D and 3D users learn one preferred result model.
- Downstream packages can avoid record-order assumptions.
- The default return change is a deliberate pre-1.0 breaking change with a
  simple explicit migration path.
- Raw cell schema remains available but does not define the entire stable API.
- Future cell-measure fitting can consume the same result without introducing a
  second result abstraction.
- The implementation issue must characterize existing result and diagnostic
  combinations before changing them.

## Alternatives considered

### Aligned dimension-specific public result classes

Rejected because the shared contract is large enough that two classes would
mostly duplicate behavior and push dimension dispatch into downstream code.

### Keep raw returns primary and add adapters only

Rejected because it would preserve the current ambiguity and make the new
contract optional rather than the normal user experience.

### Make `TessellationResult` emulate a list

Rejected because it would blur type, mutation, equality, and compatibility
semantics. Explicit raw access is easier to understand.

### Require deep immutability immediately

Rejected as a release requirement because recursively freezing backend-shaped
records would add copying and wrapper complexity without improving the main API
contract.
