# WP5 implementation architecture and verification map

Status: independently accepted and squash-merged through
[PR #72](https://github.com/DeloneCommons/pyvoro2/pull/72) as
`67395af35cbcdacc9d6202489fcc45f750a0d8a6` on 2026-09-24; #68 is complete.
Checkpoint B remains pending. The source identities below describe the
implementation's historical starting and rebase points.

This engineering note implements [ADR 0021](decisions/0021-wp5-native-occurrence-and-exact-face-certification.md).
Its public action layer follows the maintainer amendment: complete native shift
attribution and exact semantic consistency are separate outcomes. The N/E/S
geometry and producer construction are unchanged. Starting `dev` was
`358f971ea106a98e9a1c80531180e0d5b0e143ea`, tree
`0cbb0c2e0d529b1cb7da63ffe5c5a89f5698de1e` (PR #70).
Refs #68; parent tracker #47. WP6 onward and vendored source are out of scope.
The branch rebases onto the documentation-only policy amendment from PR #71:
`a99c8208cf74be7e32ccfa58983319a1383bb452`, tree
`c624e68e82ee01577cd6c2105ffadbe5f055532d`. No producer implementation changed
between these reviewed bases.

## Components and authorities

| Component | Responsibility and authority |
|---|---|
| `cpp/native_witness.cpp` | Accepted matched ordinary/observing N execution; expose actual container constants, native integer profile and primary storage indices as private replay evidence. No producer arithmetic changes. |
| `_internal/spatial/wp5_common.py` | Private structured findings and deterministic refusal budgets. |
| `_internal/spatial/wp5_binary64.py` | Exact closed B/P preimages, finite float endpoint selection, source integer semantics and bit comparisons. |
| `_internal/spatial/wp5_producer.py` | Preparation/insertion verification and complete rectangular/triclinic source histories; unique image/provenance before ideal classification. |
| `_internal/spatial/wp5_ideal.py` | Reusable exact rational full-cell engine for independent E and S inputs; self/wall outer bound, complete coefficient regions, affine dimension and exact facet area representation. |
| `_internal/spatial/wp5_cycle.py` | Exact projection onto witnessed support and ordered simple convex cycle audit; never public replacement geometry. |
| `_internal/spatial/wp5_certificate.py` | Atomic native attribution and source chart, followed by independent E/S status, projected cycle, multiplicity, positive coverage and reciprocity audit. |
| `api.py`, output/normalization helpers | Validate requests before native work, retain occurrence labels, materialize requested public native views, and apply `tessellation_check` to semantic findings. |
| `inverse/separator/realize.py` | Periodic 3D requires complete attribution and semantic consistency, then consumes S exact measure through an explicit numerical view; failures preserve atomic active state. |

The ideal engine consumes exact sites, row basis, mathematical weights,
periodic axes and real wall bounds. E uses witnessed binary64 storage and
exact squared backend radii. S uses caller data and exact mathematical
weights (exact radius squares for explicit radii). Both use the same algorithm
with independent inputs. Neither filters producer candidates.

Source replay consumes the owned witness packet and prepared native insertion
rows. Its output is exact integer removals and finite compatible provenance
classes, retaining route evidence. Distinct native occurrences survive semantic
findings and normalization. Public coefficients are `sigma + K_i - K_j` in arbitrary Python
integers before the final signed-int64 representation check.

Public native vertices retain ordinary native serialization and frame
transport, translated into the source chart by `K_i @ A`; public sites are
the original `p_i`. This does not identify stored `b_i` with conceptual `a_i`
or assume exact orthogonality. Private proof geometry is never exposed.

## Implementation and independent verification sequence

1. Establish baseline package/tests and verify the characterization archive
   SHA-256 and all internal hashes. Read the pinned source expressions before
   implementing source replay.
2. Add independent RED cases for exact binary64 bins and integer boundaries,
   source routes with hand-derived expected coefficients, and projected cycles.
   Implement these primitives and preserve signed-zero/source association.
3. Add independent exact rational/small-supercell ideal references for
   positive/zero/absent, hidden/lower-dimensional, self, partial-periodic and
   triclinic cells. Implement complete E/S reconstruction and refusal guards.
4. Integrate the witness, chart and complete certificate. Test all public flag
   combinations in both output forms, external IDs, walls/self images,
   normalization and legacy-control invariance. Fault cases exercise every
   fixed diagnostic category without changing scientific authority.
5. Integrate only the periodic 3D realization consumer. Compare scientific
   measure against independent S facets, and verify failed certification never
   installs a partially updated active state.
6. Run the supplied characterization families with semantic assertions,
   ordinary project checks and the normal CI matrix. Search production/tests
   for forbidden winner selection, epsilon positivity, repairs and partial
   enumeration. Record exact head/tree and compact acceptance evidence.

## Review focus

- Large equal radii can change native offsets/topology without changing S;
  include both power expression trees and exact zero/absent fixtures.
- Upper-y-wrap uses a distinct x recomputation; independently pin its
  association and each wrap's exact coefficient bookkeeping.
- Raw rank-three cycles can project validly; nonadjacent touches and
  backtracking must still fail without numerical tolerances.
- Hidden/lower-dimensional owners retain genuine provenance; required reverse
  volumetric coverage is a separate failure.
- Unrequested public-coordinate overflow must not prevent private proof;
  requested nonfinite views and unrepresentable integer shifts fail explicitly.

## Action and resource policy

`has_periodic_shifts` means that every requested native generator occurrence
has a unique source-compatible image and a representable public shift. It does
not imply exact E/S positivity. Real walls omit `adjacent_shift`.

Exact zero/absent contacts, E/S conflicts, invalid/collapsed projected cycles,
multiplicity, coincident provenance, and missing positive/owner/reciprocal
coverage are error-severity tessellation findings (optional reciprocity remains
a nonfatal warning). The exact audit runs when diagnostics are requested, including
without public shifts; a default shifts-only call skips it. `none` and `diagnose` return,
`warn` emits the normal warning, and `raise` rejects non-OK diagnostics. No
finding deletes a native face, changes ownership, or repairs reciprocity.
Malformed native topology/profile, unresolved/inconsistent attribution and
unrepresentable requested outputs remain unconditional structured failures.

Producer and semantic audits have separate deterministic budgets: one million
candidates per complete region, 16 million charged work units per audit, and
16,384 numerator/denominator bits. The accepted exact reduction machinery also
retains its own private limits. A producer refusal cannot publish a searched
prefix. An exact ideal refusal after attribution preserves native shifts and
reports `WP5_RESOURCE_LIMIT` with `audit_scope="semantic"` and
`audit_complete=False`; partial ideal geometry cannot supply scientific results.
These are refusal limits, not completeness windows or public tuning controls.

Exact rational reconstruction adds substantial cost when shifts are requested.
One measured 30-site periodic fixture reconstructed both 30-cell ideals in
16.09 seconds using 8,075,898 charged work units on the implementation host.
This is a characterization, not a complexity or cross-platform timing promise.
Calls without requested shifts keep ordinary native geometry/output behavior;
their explicitly requested diagnostics also run the independent periodic audit.

Periodic inverse realization checks complete semantic consistency independently
of the forward action policy. Its measure is the square root of rational S
facet `area_squared`, converted only when requested using scaled binary64
arithmetic. A positive exact area can underflow in this numerical view without
changing adjacency; overflow is `WP5_NONFINITE_OUTPUT_VIEW`. Returned native
face descriptors retain their existing numerical meaning.
