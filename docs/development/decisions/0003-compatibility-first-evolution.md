# 0003 — Compatibility-first API evolution

- **Status:** Accepted
- **Date:** 2026-07-17
- **Refined by:** [ADR 0004 — Canonical inverse namespace](0004-canonical-inverse-namespace.md), [ADR 0005 — Common tessellation result contract](0005-tessellation-result-contract.md), [ADR 0006 — v0.8 cleanup release](0006-v0.8-cleanup-release.md)

## Context

The v0.6.3 package has working forward and separator-fitting APIs, broad
top-level re-exports, notebooks, and an archived manuscript workflow. The v0.7
architecture introduces clearer inverse terminology, direct weight-first
forward computation, and a common structured result.

An unconstrained rewrite would create unnecessary risk for the manuscript
archive and existing downstream code. Preserving every v0.6.3 name and default
indefinitely would make the package incoherent just before its intended API
stabilization.

## Decision

API evolution is **compatibility-first, but not compatibility-forever**.

- Important v0.6.3 workflows receive explicit migration routes, characterization
  tests, and release notes.
- Preferred v0.7 paths may deliberately change defaults or names when the new
  contract is materially clearer.
- Historical paths can remain for one bounded transition release through
  aliases, adapters, or delegation.
- Every compatibility-only surface has a documented replacement and intended
  removal release.
- New-user documentation leads with the preferred API; compatibility docs focus
  on migration rather than teaching the historical design as equal.
- The archived paper remains tied to pyvoro2 v0.6.3 and does not migrate unless
  a scientific correction requires recomputation.
- This decision defines migration policy. Exact namespace and result choices are
  fixed by ADR 0004 and ADR 0005.

For the inverse reorganization, v0.7 is the transition release and v0.8 is the
removal release for `pyvoro2.powerfit`, broad top-level separator exports, and
other compatibility-only routes identified by ADR 0006. The v0.8 cleanup
decision supersedes the earlier possibility of extending the transition based
on usage.

## Consequences

- v0.7 changes need tests for both preferred and stated compatibility routes.
- A compatibility path preserves semantic behavior, not merely importability.
- A simple migration is allowed to require an explicit argument such as
  `output='cells'`; compatibility does not require preserving every old default.
- Release notes distinguish stable, provisional, experimental,
  compatibility-only, and deprecated surfaces.
- Development documentation distinguishes implemented behavior from target
  behavior until each issue lands.
- A generic observation protocol remains deferred until at least two real
  observation families provide evidence for it.

## Alternatives considered

### Break everything once before 1.0

Rejected as a default. Pre-1.0 status permits deliberate change, but migration
tests and bounded shims are inexpensive for the most important scientific
workflows.

### Preserve every current top-level export indefinitely

Rejected because it would make top-level `pyvoro2` an unstructured mirror of
separator implementation details and constrain future inverse families.

### Promise indefinite deprecation periods

Rejected because the project is small and pre-1.0. Bounded, release-specific
migration is clearer for users and cheaper to maintain.

### Update the manuscript and rerun all experiments on every package release

Rejected. Reproducibility requires the exact archived version used for the
reported results, not automatic migration to the newest API.
