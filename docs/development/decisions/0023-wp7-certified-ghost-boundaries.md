# 0023 — WP7 source-certified ghost boundaries

- **Status:** Accepted; production implementation independently accepted and merged
- **Date:** 2026-09-26
- **Related issue:** [#77 — WP7 ghost boundary identity](https://github.com/DeloneCommons/pyvoro2/issues/77)
- **Related plan:** [v0.9 WP7](../plans/v0.9.md#wp7-define-ghost-boundary-identity-and-certify-periodic-ghost-shifts)
- **Implementation map:** [WP7 implementation and qualification plan](../wp7-implementation.md)
- **Related decisions:** [ADR 0002](0002-weights-radii-and-gauge.md),
  [ADR 0012](0012-certified-periodic-image-geometry.md),
  [ADR 0013](0013-central-generator-preparation-and-backend-safety.md),
  [ADR 0016](0016-severity-complete-tessellation-diagnostics.md),
  [ADR 0018](0018-periodic-user-lattice-and-boundary-semantics.md),
  [ADR 0021](0021-wp5-native-occurrence-and-exact-face-certification.md), and
  [ADR 0022](0022-wp6-source-certified-planar-edge-provenance.md)

## Context and scope

The four 3D `compute_ghost_cell` routes can leave the temporary native ID
uninitialized before a periodic revisit or image copy reads it. A strictly
nonperiodic rectangular call is a negative control: its skipped primary-self
slot and lack of periodic revisits do not establish such a read. The existing
2D ghost path inserts a synthetic ID but reconstructs requested image shifts
through finite numerical matching. Neither path establishes complete boundary
identity for a persistent image, a ghost self-image or a physical wall.

The complete live [#77 contract](https://github.com/DeloneCommons/pyvoro2/issues/77)
closes the previously open ghost-specific decisions. This ADR records those
durable choices without transferring the distinct ordinary persistent-cell
actions from ADRs 0021 and 0022. It applies to both dimensions, standard and
power mode, and existing nonperiodic, partial-periodic and full-periodic
domains. It does not add a planar oblique domain, a ghost result class, WP8
query metadata, or a general ghost diagnostics selector.

## Defined native execution and provenance

Each query selects its cell in an augmented population of all persistent
generators and exactly that temporary ghost. Initialize an injective internal
ghost ID before any active slot count, image generation, read, copy or
observation can use it. Verify that every input actually reached native storage,
including its ID, block/slot, coordinates and radius, before interpreting
native deletion as an empty cell. The internal ID is not public ownership.
Fresh per-query containers provide state isolation; any reuse must prove that
particles, lazy images, radius maxima, tokens and counters cannot leak across
queries. Preserve the prepared grid, insertion order, actual radii and
batch-wide weight gauge. A public `compute(points + query)` is not this
selected-cell route. No functional vendor-source edit is authorized; one would
trigger the still-open D9 decision before acceptance.

Every returned raw face or edge, including a collapsed occurrence, requires a
same-computation source association. In 2D, initialization-side or particle
tokens propagate with outgoing-edge slots through native topology. The source
region-index proof bounds native periodic coefficients by `{-1,0,+1}` per
periodic axis. In 3D, observed support and topology require the complete
producer-compatible replay of direct/worklist, initialized seed, triclinic
image and actual standard/power offset-expression routes. A support plane alone
does not identify its lattice image. The accepted WP5/WP6 machinery may be
reused only after the selected augmented route and source/build profile have
been qualified.

Classify a verified persistent owner as `generator`, the selected temporary
owner or its nonzero periodic initialization support as `ghost_self`, and a
qualified physical nonperiodic side as `wall`. A primary-self cut, unknown
token, construction-only bound, unexpected owner or missing insertion is not a
wall or a hidden power cell. Coincident producer histories are equivalent only
when their final semantic kind, owner, image and exact cut agree. Native integer
sign, proximity, residual ranking, exact ideal positivity and reciprocity
never select or repair an ambiguous source.

## Stored-ghost chart and mathematical semantics

Let `P_j` be the original persistent site, `A` the public row lattice, `k_j`
its preparation translation, `h_j` its verified native insertion translation,
`K_j=k_j+h_j`, and `sigma` the source-certified native image coefficient in
the corresponding lattice convention. Let `g` be the **actual stored ghost's
materialized Cartesian site**, which anchors returned geometry. Then:

```text
generator:   shift = sigma - K_j
             boundary_image = P_j + shift @ A
ghost_self:  shift = sigma != 0
             boundary_image = g + shift @ A
wall:        no shift
```

The ordinary persistent source-centered formula `sigma+K_i-K_j` does not
apply to this stored-ghost chart. Private coefficient arithmetic uses Python
integers; enforce signed int64 only for a required public shift, and check
finite requested Cartesian views separately. The declared binary64 rectangular
span is the public lattice operand. Left-handed 3D user rows retain their
order and existing orientation transport. A ghost record's `site` is `g`, not
the original unwrapped query.

Standard mode rejects all power inputs. Power mode accepts exactly one
complete `weights`/`ghost_weights` or `radii`/`ghost_radii` family. Original
validated mathematical weights remain distinct from backend radii; weight
mode shares one representation shift across the entire ghost batch. Explicit
radius mode uses the exact square of each supplied binary64 operand for
public-semantic reasoning, not the exactification of a rounded product.

**N** is the actual attributed native occurrence and topology. **S** is the
complete exact public-semantic cell built from original persistent sites,
actual ghost anchor `g`, public lattice/walls and original mathematical
weights. An explanatory native-effective ideal **E** may diagnose storage or
radius-order effects but is not mandatory on each successful ghost call and
cannot choose N's labels. Positive ghost references require N plus S. In
local coordinates `x=X-g`, each persistent image contributes

```text
d = P_j + shift @ A - g
2 d·x <= d·d + W_g - W_j.
```

Ghost self-images use `d=shift@A` and zero weight difference; real walls
constrain nonperiodic axes. Build a bounded exact outer polytope from spanning
self-image cuts and physical walls. Bound every image that could touch or
restrict it by exact/outward rational inverse-column bounds, including equality
contacts; the centered image plus `{-1,0,+1}` per periodic axis is an
alternative complete family for rectangular domains. Enumerate the complete
proved region before testing exact cell contact. A positive boundary requires
a full-dimensional S cell and an active contact of affine dimension one in 2D
or two in 3D. Neither a numerical length/area threshold nor a partial search
establishes positivity.

## Raw occurrences, empties and public action

Preserve native geometry, ordering and multiplicity. Attribute N first, decide
internal collapse exactly, then compare the attributed class with the complete
S contact map. A well-formed noncollapsed occurrence gets a reference only
when its S class is positive; zero or absent S contact is a hard semantic
inconsistency. A proved internally collapsed raw occurrence remains returned
with `boundary_reference=None` and cannot cover a positive facet. In 2D,
collapse is equality of doubled-local endpoints. In 3D, validate cyclic
incidence and project onto the observed support: affine rank below two is
collapse, while an invalid rank-two projected cycle is failure. Public
coordinate rounding does not establish internal collapse.

Every positive geometric S facet needs at least one eligible attributed native
occurrence active on it. Preserve multiple native fragments. Distinct
co-active labels on a coincident facet remain distinct possible provenance;
do not fabricate an unobserved producer for every co-active label. This is
facet coverage, not exact numerical polygon-union equality or one-to-one
fragment reciprocity.

After verified insertion, native deletion is `empty=True` on a boundary-bearing
call only if S has no full-dimensional cell; keep exact empty versus
lower-dimensional status in private evidence. Native/S disposition mismatch
fails. Return zero measure and empty requested geometry for an accepted
empty cell. Apply `include_empty=False` only after all checks. Geometry-only
calls retain native empty disposition without claiming an S certificate. A
zero-query call returns validated empty output. Every hard failure aborts the
entire batch.

`ghost_cells` retains `list[dict]`, `id=-1`, input `query_index`, the spatial
original `query` key, and existing selectors. Planar original `query` and
WP8's unified query fields are not added here. Every requested face/edge
contains the four-field nested `boundary_reference` or the proved-collapse
outer `None`:

| Kind | `generator_id` | `shift` | `wall_id` |
|---|---|---|---|
| `generator` | Persistent external ID | User-basis tuple in any periodic domain, including zero; otherwise `None` | `None` |
| `ghost_self` | `None` | Nonzero user-basis tuple, zero on nonperiodic axes | `None` |
| `wall` | `None` | `None` | Source-qualified existing physical side ID |

`adjacent_cell` remains a compatibility view only for qualified persistent
owners or walls; omit it for ghost self. A collapsed record may retain only a
qualified persistent/wall compatibility label. Planar `return_edge_shifts`
continues to require a periodic domain and requested edges, but no public
vertices. Its `adjacent_shift` is emitted only for eligible generator/self
references when requested; the nested positive reference contains its shift
independently. Walls and collapsed records omit `adjacent_shift`. No ghost
`has_periodic_shifts` flag or `TessellationResult` is introduced.

Remove the four planar **ghost** controls `edge_shift_search`,
`validate_edge_shifts`, `repair_edge_shifts` and `edge_shift_tol` without
aliases. Keep unrelated ordinary 3D controls, normalization/diagnostic
tolerances and separator `image_search` on their own schedules.

Hard ghost certificate failures are `ValueError`-compatible and expose
`code`, `stage`, `query_index` and bounded `details`. The stable codes are
`GHOST_BACKEND_INSERTION`, `GHOST_NATIVE_UNSUPPORTED`,
`GHOST_PROVENANCE_AMBIGUOUS`, `GHOST_PROVENANCE_INCONSISTENT`,
`GHOST_SEMANTIC_INCONSISTENT`, `GHOST_CERTIFICATION_RESOURCE` and
`GHOST_SHIFT_UNREPRESENTABLE`. Source/profile, attribution, exact work,
representation and input failures remain distinct. Structural limits on
tokens, occurrences, complete candidate regions, charged work, rational bits
and observer memory refuse atomically, never certify a checked prefix. See the
implementation note for initial numerical budgets. Generic whole-tessellation
normalization/closure/partition reciprocity rejects batches of independent
ghost records instead of treating `id=-1` as a shared owner. ADR 0016's
ordinary diagnostic severity contract is unchanged.

## Qualification and consequences

Safe initialized identity is distinct from qualified source certification.
The initial admitted certificate-bearing cohort was Linux x86_64 GCC 13.3 under
the reviewed strict binary64/SSE2 noncontracting, nearest-rounding, gradual-
underflow, 32-bit-native-integer and no-fast-math/no-AVX-FMA/no-LTO profile.
That cohort records historical evidence, not a permanent compiler/ISA allowlist.
The 2026-09-28 qualification amendment in
[ADR 0024](0024-external-native-artifact-qualification.md) requires externally
qualified installed artifacts and current-thread FP checks. Spatial selected
ghost admission composes WP5 with its selected spatial component; planar
selected ghost admission composes WP6 with its selected planar component.
Unqualified components explicitly refuse; initialized geometry-only availability
may be broader. Mechanical source identity, actual compile/link evidence, native module and
installed artifact identity must agree. A changed planar binding closure also
requires renewed ordinary WP6 qualification and noninterference evidence, not
merely a digest update. The mathematical and public-action contract above is
unchanged.

Independent rational oracles, a defined initialized selected-cell reference,
legacy poisoned-slot or MemorySanitizer evidence, corrected-path memory tests,
native parity and installed source/wheel/sdist qualification are required by
#77. ASan/UBSan alone cannot demonstrate the old uninitialized read. WP7's
subsequent independent acceptance and merge are recorded in the active plan;
green CI alone did not establish that acceptance. Issue #88's changed
qualification architecture still requires its own exact-head evidence and
independent review. Checkpoint B and #47 acceptance remain separate.

## Alternatives rejected

Replacing an undefined ID after the native computation, overloading integer
sign as boundary kind, choosing nearest residuals or finite image windows,
using exact S positivity to disambiguate N, dropping raw zero artifacts,
calling all persistent cells for one ghost, and making E mandatory for every
success each violate a separate native, mathematical or output obligation.
