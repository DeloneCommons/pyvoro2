# WP7 implementation and qualification plan

WP7 is governed by the complete live [issue #77](https://github.com/DeloneCommons/pyvoro2/issues/77),
including its synchronized contract of 2026-09-26. The baseline is `dev`
`1a6ae56ece1923530dcb0798b5710704f0f08697`, tree
`a1d30f45029151e7f60d1b24c7b7481e5461ca2e`. WP5 and WP6 are independently
accepted. This note records delegated engineering choices; it does not replace
the scientific contract or constitute implementation acceptance.

## Components and invariants

| Component | Responsibility and acceptance obligation |
|---|---|
| `cpp/bindings.cpp`, `cpp/native_witness.*` | Replace all four unsafe ghost routes with initialized selected-cell execution. Preserve persistent insertion order, grid, resolved radii, kernel and radius maximum; observe the selected cell only. |
| `cpp/bindings2d.cpp`, `cpp/planar_witness.*` | Fresh augmented container, dense temporary ID `n`, verified actual insertion and outgoing-edge tokens; geometry-only calls use safe initialized storage without claiming certificate support. |
| Existing WP5 producer/packet checks | Add an explicit private selected-source scope while retaining full augmented-population accounting, source replay, parity, and topology checks. Ordinary defaults remain all-source. |
| Existing WP6 packet checks | Add selected-source scope while retaining source/slot/token association, actual storage and profile predicates. Do not reinterpret missing persistent computations as deleted cells. |
| `_internal/ghost.py` | ValueError-compatible fixed `GHOST_*` protocol, exact original-weight construction, S cell/contact eligibility, geometric-facet coverage, public reference materialization and bounded failure details. |
| `_internal/spatial/ghost_certificate.py` | Attribute N first, validate incidence and projected-cycle collapse, use actual stored Cartesian ghost anchor, transport persistent shifts as `sigma-K_j`, certify S, then package requested raw native views. |
| `_internal/planar/ghost_certificate.py` | Attribute native outgoing edges first, classify exact doubled-local endpoint equality, apply stored-ghost chart, certify S and expose planar shift compatibility only when requested. |
| Both public APIs | Preserve validation, batch-wide power gauge, `list[dict]`, query indexing and selectors; remove exactly the four WP7 planar controls. Filter empty cells only after certification. |
| Normalization entry points | Explicitly reject independent ghost records where ordinary cell-owner closure/partition normalization is required. |
| Independent oracle/native harnesses | Rational mathematical expectations, initialized stock selected-cell reference, poisoned-slot before/after identity evidence, source/profile and installed-artifact qualification. |

The temporary native identity is dense `n` in the augmented population
`0..n`, checked against native and token capacity before insertion. Every query
owns a fresh container and includes no other query as a competitor. All original
persistent inputs and the selected ghost must be found in actual native storage
before deletion can mean empty. Public owners are never inferred from integer
signs or the numerical value of the temporary identity.

The 3D private `_observe_ghost_box` and `_observe_ghost_periodic` operations return
one WP5-shaped packet per query. Each packet contains all augmented `sites`, a
single selected `cells` entry, the actual `context`/`build`, `query_index`, and
`ghost_internal_id`. Selected-cell scope is explicit, never implemented by
computing every persistent cell. The ordinary geometry-only ghost entry points
also use initialized insertion, population verification and stored-site output.

The planar `_ghost_box_standard_witness` and `_ghost_box_power_witness`
operations return `(cells, packets)` aligned by original query index. Each packet
contains all augmented `inserted` records, `sources=[selected_source]`, bounds,
periods, mask, profile, `ghost_internal_id` and `query_index`. Even a deleted cell
has a native disposition record. Doubled-local coordinates remain private when
public vertices are omitted.

S uses original persistent Cartesian sites plus the actual materialized stored
ghost site as its selected source. It uses the declared public lattice and exact
original weights, or separately exactified input radii squared. Existing exact
WP5/WP6 ideal primitives may be reused in production; the test oracle does not
import them. The complete semantic family is independent of native attribution.
Coincident contact labels remain separate, while coverage groups identical
positive geometric supports. Collapsed native occurrences retain raw geometry
and `boundary_reference=None`; they never supply positive coverage.
The private `SemanticCertificate` retains exact dimension separately from native
disposition, including accepted empty versus lower-dimensional deletions, until
public packaging. No new public cell field is introduced.

Exact public semantics can refuse otherwise usable native geometry. Independent
rational regressions show that triclinic frame rounding can create tiny positive
S facets missing from N; such boundary-bearing calls raise
`GHOST_SEMANTIC_INCONSISTENT`. Geometry-only native volume/shape invariants are
tested separately. Public coordinate rounding alone never establishes collapse:
observed 2D and 3D cases retain positive references even when public vertices
coincide, because the private doubled-local geometry has positive dimension.

## Refusal budgets and profiles

Use the existing qualified finite budgets as initial private limits: one million
native source IDs, 262,144 final occurrences and 64 MiB observer memory; planar
exact certification has 250,000 candidate constraints, 10,000,000 charged work
units and 16,384-bit rational components; spatial exact/producer certification
has 1,000,000 candidates per complete region, 16,000,000 charged work units and
16,384-bit rational components. These bound memory and arithmetic cost and are
hard refusals, never search-window controls. Charge the complete relevant
operation and preflight whole candidate regions. Public int64 checks apply only
to shifts actually materialized; coordinate finiteness is checked separately.

Native witness batches preflight all `m * (n + 1)` retained insertion rows
before constructing a query. The 64 MiB charged envelope reserves 2,048 bytes
per inserted row and 16,384 per query packet. Planar edges add 2,048 bytes each;
spatial source tokens and faces add 2,048 each, with 512 per vertex and directed
edge entry. The peak additional native observer storage is also charged, and
token/occurrence counters accumulate across queries. Representative recursive
Python object-size tests check these conservative envelopes. This is an
explicit resource policy, not a promise that process RSS equals the charge.

Initially qualify Linux x86_64 GCC 13.3 with the existing strict binary64/SSE2,
nearest/gradual, noncontracting/no-IPO policy. Other certificate-bearing cohorts
must refuse until independently qualified. Safe initialized geometry-only
availability is separate. Bind source closures to actual build commands and
installed artifacts. Changes touching planar binding closure require ordinary
WP6 stock/observed regression qualification, not only a new expected digest.
No functional vendored change is planned; D9 remains open.

## Implementation and verification order

- [x] Establish independent RED analytic tests and rational oracle for the
  stored-ghost chart, self/walls, power families, empty/lower-dimensional cells,
  coincidence, positivity and coverage before production certificates.
- [x] Establish defined initialized native selected-cell reference tests and
  legacy poisoned-slot instrumentation with strictly nonperiodic controls.
- [x] Implement fresh initialized native selected routes, verified insertion,
  retained source association and before/after memory evidence.
- [x] Implement private selected-source N attribution, stored-ghost chart and
  S eligibility/coverage/empty dispositions. Verify public vertices are optional.
- [x] Integrate exact fixed failure protocol, planar removals and normalization
  rejection. Preserve ordinary APIs and all unrelated controls.
- [x] Add the WP7 ADR and update current architecture, plan target wording,
  inventory/lifecycle, guides/reference/migration and changelog.
- [ ] Run focused tests, native parity/profile qualification, relevant WP5/WP6
  regressions, seeded fuzz, full suite, lint, generated-file checks and strict docs.
- [ ] Build direct wheel, sdist and wheel from sdist; test installed artifacts
  with supported optional-dependency conditions and record hashes/commands.
- [ ] Open the issue-linked PR against `dev`, obtain exact-head remote CI,
  and assemble a compact independent-review archive with inventory and hashes.

The final review must deliberately check translated query representatives,
actual insertion seam branches, coincident retained origins, internal versus
public-rounding collapse, and whole-operation resource accounting. Green tests
and engineering review do not replace independent mathematical/native/API
acceptance. Do not merge, close #77, mark WP7 complete, accept Checkpoint B or
enter WP8/WP9/Phase C.
