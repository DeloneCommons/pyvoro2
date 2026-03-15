# Roadmap

This page lists intended future improvements. It is not a guarantee of timelines.

## Planned / likely

### Native 2D support

Voro++ ships a dedicated 2D implementation. pyvoro2 plans to expose it as a **separate extension
module** (e.g. `_core2d`) so that 2D and 3D code do not collide at link time.

### Powerfit objective-model expansion

The self-consistent active-set solver is now part of the package, so the next
powerfit design question is not iteration support but **objective-model scope**.

For the 0.5.x series, the built-in family is intentionally compact: quadratic
and Huber mismatch terms, interval and fixed-value hard feasibility,
outside-interval penalties, near-boundary penalties, and L2 regularization.

Additional mismatch or penalty families should be added only after downstream
packages validate a concrete need for them. The goal is to expand from real
workflows rather than to freeze a broad callback surface too early.

## Potential

### Visualization usability

The optional `py3Dmol`-based viewer (`pyvoro2[viz]`) is intended as a lightweight debugging and
exploration tool. Future work is expected to focus on usability (better defaults, more annotations),
not on adding heavy rendering dependencies to the core.

## Release stability

pyvoro2 is currently in **beta**.

A “stable” 1.0 release is expected only after:

- the inverse-fitting workflow matures further
- native 2D support is implemented and tested
