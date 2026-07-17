# 0002 — Weights, radii, gauge, and component offsets

- **Status:** Accepted
- **Date:** 2026-07-17

## Context

A power diagram is defined by scalar weights in

\[
\pi_i(x) = \lVert x-p_i\rVert^2 - w_i.
\]

Voro++ accepts a radius-like representation with \(w_i=r_i^2\). General fitted
weights can be negative, so pyvoro2 currently adds a shift before converting
them to non-negative radii.

The separator inverse problem observes only selected weight differences. If its
informative observation graph is disconnected, the data leave one independent
constant per connected component undetermined. Existing documentation has often
called all such constants “gauge,” although independent component shifts can
change the complete diagram when sites from different components compete.

## Decision

- **Power weights are the public mathematical quantities.**
- **Radii are a backend-compatible representation**, not necessarily physical
  radii and not the unique scientific result of inverse fitting.
- Adding one common constant to every weight is the **global geometric gauge**:
  it leaves every power comparison and the complete diagram unchanged.
- Extra constants associated with disconnected informative observation
  components are **unidentified component offsets**. They are invisible to the
  observed separator equations but may change the complete realized
  tessellation.
- Conversion from weights to radii must record or expose the chosen global
  shift.
- Canonicalization/alignment policies for disconnected components must be
  explicit result metadata and diagnostics, not described as information
  recovered from the observations.
- Downstream packages must not infer unique physical radii from a shifted
  backend representation.

## Consequences

- The forward API should gain a direct weight-first route.
- Results should separate fitted weights, global representation shift/radii,
  observation-graph identifiability, and component alignment policy.
- Connectivity diagnostics are scientifically meaningful even when a numerical
  solution exists.
- Tests should compare weights modulo global gauge, or modulo component offsets
  only when the observation problem—not the full diagram—is being tested.
- Realization checks are required when disconnected component alignment can
  affect the final geometry.
- Existing `radii=` and `r_min=` behavior can remain for compatibility, but new
  docs should not make them the primary mathematical framing.

## Alternatives considered

### Treat fitted radii as the primary unknown

Rejected because the separator laws, graph theory, and future mixed solvers are
naturally formulated in weights, while the radius shift is arbitrary.

### Call every connected-component constant gauge

Rejected because independent component shifts generally change cross-component
power competition and therefore are not symmetries of the complete diagram.

### Force a fixed anchor in each component and hide the ambiguity

Rejected because it makes output depend on arbitrary ordering and conceals a
real identifiability limitation.
