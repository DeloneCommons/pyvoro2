# 0004 — Canonical inverse namespace and separator organization

- **Status:** Accepted
- **Date:** 2026-07-17
- **Related plan:** [v0.7 development plan](../plans/archive/v0.7.md)
- **Refined by:** [ADR 0006 — v0.8 cleanup release](0006-v0.8-cleanup-release.md)

## Context

The v0.6.3 inverse implementation is technically substantial, but its public
organization reflects its history rather than the mathematical structure of the
project:

- implementation lives under `pyvoro2.powerfit`;
- many separator-specific names are re-exported from top-level `pyvoro2`;
- names such as `PairBisectorConstraints` describe an implementation lineage
  more than the observed geometric quantity;
- future prescribed-measure and mixed inverse problems need a common home that
  is not named after the first solver.

The project is still pre-1.0, so a coherent public organization is more valuable
than preserving every historical import indefinitely. At the same time, a very
small transition shim is inexpensive and avoids breaking archived notebooks or
existing downstream code without warning.

## Decision

`pyvoro2.inverse` is the canonical public namespace for inverse weighted-
tessellation functionality.

The first implemented observation family is organized under:

```text
pyvoro2.inverse.separator
```

The separator implementation, not `pyvoro2.powerfit`, owns the numerical code.
Compatibility dependencies point in one direction:

```text
pyvoro2.powerfit -> pyvoro2.inverse.separator
```

The reverse dependency is forbidden.

### Preferred public surface

New-user documentation should lead with a small math-aligned surface such as:

```python
from pyvoro2.inverse import (
    SeparatorObservations,
    SeparatorFitResult,
    fit_weights_from_separators,
)
```

Advanced separator-specific objects remain available from
`pyvoro2.inverse.separator`.

The preferred core terminology is:

| Historical name | Preferred v0.7 name |
|---|---|
| `PairBisectorConstraints` | `SeparatorObservations` |
| `resolve_pair_bisector_constraints` | `resolve_separator_observations` |
| `PowerFitProblem` | `SeparatorFitProblem` |
| `PowerWeightFitResult` | `SeparatorFitResult` |
| `fit_power_weights` | `fit_weights_from_separators` |

The implementation issue may refine secondary names, but it must preserve the
same mathematical distinctions: observations, fitted weights, algebraic
residuals, realized boundaries, and solver diagnostics are not interchangeable.

### Compatibility period

For v0.7:

- `pyvoro2.powerfit` remains as a thin compatibility-only package;
- historical names delegate to the canonical implementation;
- broad separator-specific top-level exports remain compatibility-only;
- importing the historical package may emit a normal `DeprecationWarning`,
  which is hidden by default by Python;
- migration documentation shows the preferred replacements.

The removal release is v0.8. ADR 0006 makes that boundary final and extends the
cleanup to the other compatibility-only inverse and planar transition surfaces.

The compatibility layer should contain imports, aliases, and narrowly necessary
wrappers only. It must not duplicate solver logic.

### Weight/radius transforms

Weight/radius conversion is not separator-specific. Its implementation should
move to a neutral internal module, while the public helpers remain available at
top-level `pyvoro2` and may also be re-exported where useful.

### Experimental outer algorithms

Realization-aware active-set functionality remains separator-specific and should
be exposed from an explicitly documented advanced or experimental part of the
separator namespace. It must not be presented as the exact fixed-observation
solver.

## Consequences

- v0.7 performs a real ownership move, not only a facade over `powerfit`.
- The migration is intentionally bounded to one normal transition release.
- Current paper results remain tied to the archived v0.6.3 environment and do
  not need to use the new namespace.
- New documentation and examples stop teaching broad top-level inverse imports.
- Future inverse families have clear homes such as
  `pyvoro2.inverse.cell_measure` and `pyvoro2.inverse.mixed` without forcing a
  generic observation protocol in v0.7.
- The API inventory must classify every historical path as compatibility-only,
  deprecated, or removed on the stated schedule.

## Alternatives considered

### Delete `pyvoro2.powerfit` immediately in v0.7

Rejected because the architectural benefit is almost identical to a one-way
shim, while a single transition release gives existing users a clear warning
and migration path at very low maintenance cost.

### Keep `pyvoro2.powerfit` indefinitely

Rejected because it would leave the first solver as the conceptual owner of all
future inverse work and weaken the package's mathematical organization.

### Add a top-level `pyvoro2.separator_fit` package

Rejected because it names an operation rather than an observation family and
would not scale cleanly to prescribed measures and mixed problems.

### Keep implementation under `powerfit` and make `inverse` re-export it

Rejected because the compatibility namespace would remain the architectural
owner and future implementation would continue to accumulate under the wrong
name.
