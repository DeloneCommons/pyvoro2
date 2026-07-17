# 0003 — Compatibility-first API evolution

- **Status:** Accepted
- **Date:** 2026-07-17

## Context

The v0.6.3 package has a working and documented `pyvoro2.powerfit` surface, broad
top-level re-exports, notebooks, and an archived manuscript workflow. The next
architecture should use clearer inverse terminology and may introduce a
`pyvoro2.inverse` namespace, a weight-first forward API, and common result
concepts.

An abrupt rewrite would create unnecessary risk for the manuscript archive,
existing users, and downstream development. At the same time, freezing every
v0.6.3 name and record shape would prevent the package from becoming coherent.

## Decision

- API evolution is compatibility-first rather than rewrite-first.
- Existing documented `pyvoro2.powerfit` workflows remain available through
  aliases, adapters, or delegation during the v0.7 transition.
- A clearer `pyvoro2.inverse` namespace may become the preferred public home for
  inverse concepts.
- New-user documentation should lead with preferred terminology and imports;
  migration documentation should cover older paths.
- Top-level inverse re-exports may be reduced only through an explicit
  deprecation path.
- The archived paper remains tied to pyvoro2 v0.6.3 and does not need to migrate
  to a later API unless a scientific correction requires recomputation.
- This decision fixes responsibilities and migration policy, not exact future
  class names or result fields.

## Consequences

- Stage 1 changes need compatibility tests.
- Public aliases should preserve semantics, not only importability.
- Release notes must distinguish preferred, compatibility, provisional, and
  experimental surfaces.
- Documentation on the development branch must distinguish current v0.6.3 APIs
  from target v0.7 concepts until implementation catches up.
- The common result schema and observation protocol should receive their own
  decisions only after concrete alternatives are evaluated.

## Alternatives considered

### Break the API once before 1.0

Rejected as a default strategy. Pre-1.0 status permits change, but the existing
scientific and downstream value makes deliberate migration cheaper than a clean
rewrite.

### Preserve every current top-level export indefinitely

Rejected because it would make the top-level namespace an unstructured mirror
of implementation internals and constrain later inverse families.

### Update the manuscript and rerun all experiments on every package release

Rejected. Reproducibility requires the exact archived version used for the
reported results, not automatic migration to the newest API.
