# Maintainer reassessment — decision register

**Purpose:** single compact, maintainer-owned index of historical capabilities and new research candidates in the v0.8-based reassessment. **Not** an implementation plan or authorization to change code, CI, public API or release commitments. [ADR 0027](../../docs/development/decisions/0027-v0.9-reboot-from-v0.8.md) remains the governing reboot decision.

**Sources:** [historical capability inventory](../../docs/development/reboot/requirements-inventory.md) (R-IDs), [evidence index](../../docs/development/reboot/evidence-index.md) (independent oracles), and focused [reassessment notes](potential-work/README.md). Historical acceptance of Checkpoints A/B or individual Phase C work is **not** adoption into the reboot; integrated C was unaccepted and D unfinished. Published v0.8 [R-B01–R-B03](../../docs/development/reboot/requirements-inventory.md#baseline) are baseline behavior, not proposals.

**Review status** is the current maintainer decision, independent of historical outcome:

- `pending`: not yet decided for the reboot.
- `selected`: maintainer chose the functional **direction**, but its concrete contract, implementation, migration and issue scope may remain undecided.
- `existing`: already accepted in the present reboot.
- Future statuses when justified: `specified` (contract agreed), `deferred`, `declined`, `delivered`. Do not infer them from legacy commits.

Review sequencing preference: settle absent or mathematics-affecting public APIs first, then bounded safety/numerical work; the goal is not to reimplement Voro++. This is a prioritization principle, not an accepted future release plan.

Keep modest clarifications and follow-up thoughts in the **last comment column**; create a separate `potential-work/` note only for substantial new alternatives, evidence or scientific investigation. Work is authorized only through a separate explicit maintainer decision and the applicable ADR/issue/PR process.

## A. Inputs, lattice and periodic operations

| ID | Capability / question | Historical outcome | Review status | Comment / open choice |
| --- | --- | --- | --- | --- |
| [R-Q01a](../../docs/development/reboot/requirements-inventory.md#r-q01) | Weight-first `locate` (2D/3D) | WP1; A accepted | `selected` | Maintainer chose a `compute`-like `weights`/`radii` input family, preserving direct radii. Exact signature and compatibility not yet specified. |
| [R-Q01b](../../docs/development/reboot/requirements-inventory.md#r-q01) | Weight-first `ghost_cells` (2D/3D) | WP1; A accepted | `selected` | Weight-first capability selected in principle, **not** the full historical API. Shared gauge across persistent and ghost weights is necessary; mixed families and `ghost_radius` compatibility undecided. |
| [R-L01](../../docs/development/reboot/requirements-inventory.md#r-l01) | Caller lattice and backend coordinate frame | WP2; A accepted | `selected` | Maintainer selected accepting both lattice handedness signs and exact public fractional transforms/wrapping (especially integer image shifts). Specify API semantics first. Relaxing conditioning thresholds, native admissibility, and backend numerical safeguards are **not** selected; review separately with bounded scope, not a Voro++ rewrite. |
| [R-L02a](../../docs/development/reboot/requirements-inventory.md#r-l02) | Basis-independent physical tie-breaking for equally close images | WP4; A accepted | `selected` | Maintainer selected the **direction**: compare physically equal-distance images using Cartesian displacement and a documented deterministic tie rule, not reduced-basis coefficients. Specify precise edge cases when implementing. |
| [R-L02b](../../docs/development/reboot/requirements-inventory.md#r-l02) | Private basis reduction to speed up certified image search | WP3–4; A accepted | `pending` | **Proposals only, not an algorithm decision:** first profile existing certified CVP; consider simple verified unimodular size reduction with unchanged exact CVP/fallback; if insufficient, compare exact rank-3 LLL and 3D Minkowski methods. Niggli is primarily crystallographic canonicalization and uses numerical tolerances in common implementations; do not assume it is simpler or appropriate. Preserve exact integer shift transport and independent correctness oracles. No public reduction API or native-cell rebasing selected. |
| [R-L03](../../docs/development/reboot/requirements-inventory.md#r-l03) | Preserve source geometry through remapping | WP4, #60/#65, later PR #67 | `pending` | Source operands before snapping; coupled endpoint wrapping and complete declared-box native translation. Keep defects and independent counterexamples. |
| [R-G01](../../docs/development/reboot/requirements-inventory.md#r-g01) | Ghost execution safety | A remediation + B WP7 | `pending` | Two distinct native hazards: persistent periodic images of earlier ghosts, and uninitialized temporary neighbor ID. Reassess reproducers and remedy independently. |

## B. Native boundaries, identity and topology

| ID | Capability / question | Historical outcome | Review status | Comment / open choice |
| --- | --- | --- | --- | --- |
| [R-T01](../../docs/development/reboot/requirements-inventory.md#r-t01) | 3D native occurrence and exact contacts | WP5; B accepted | `pending` | N native provenance, E contact and S source ideals are different; reproducing actual native arithmetic and strict refusal may alter availability and even normal geometry. |
| [R-T02](../../docs/development/reboot/requirements-inventory.md#r-t02) | 2D edge provenance | WP6; B accepted | `pending` | Native source tokens survive collapsed edges; historical mandatory qualification and refusal are separate decisions. |
| [R-G02](../../docs/development/reboot/requirements-inventory.md#r-g02) | Ghost boundary semantics | WP7; B accepted | `pending` | Distinguish persistent owner, wall and nonzero ghost self-image; native facets may miss source-ideal positive faces. |
| [R-M01](../../docs/development/reboot/requirements-inventory.md#r-m01) | Locate wrapping and output ownership | WP8 + #92/#93; B accepted | `pending` | Source query wrap vs numerical native owner position; certified image metadata and late stripping of private geometry without output aliasing. |
| [R-A01](../../docs/development/reboot/requirements-inventory.md#r-a01) | Obsolete reconstruction knob removal | WP9; B accepted | `pending` | Only consider old API removals after choosing replacement attribution, limits and migration; do not import wholesale. |
| [R-T03](../../docs/development/reboot/requirements-inventory.md#r-t03) | 2D degeneracy and proof-assisted identities | B follow-ups #94–#98 | `pending` | N occurrences, E contacts and S topology need not have the same counts; no equal-coordinate/zero-length identity inference. |
| [R-T04](../../docs/development/reboot/requirements-inventory.md#r-t04) | 3D degeneracy and periodic winding | B follow-ups #96/#99 | `pending` | Missing positive native faces, rounded coordinate collisions and legitimate nonzero winding; no generally exact global complex was achieved. |

## C. Separator inverse and realization

| ID | Capability / question | Historical outcome | Review status | Comment / open choice |
| --- | --- | --- | --- | --- |
| [R-I01](../../docs/development/reboot/requirements-inventory.md#r-i01) | Independent separator spaces | WP10 accepted; C unaccepted | `pending` | Source observation, model mismatch, hard-bound and penalty spaces may differ; examine affine mathematics and minimal API. |
| [R-I02](../../docs/development/reboot/requirements-inventory.md#r-i02) | Bounded row values and applicability | #104/WP10 accepted; C unaccepted | `pending` | Scalar or exact-length row vectors and Boolean hard applicability; preserve row identity across subsets and refits. |
| [R-I03](../../docs/development/reboot/requirements-inventory.md#r-i03) | Row-wise shape: historical NONE gate | PR #115; no implementation | `pending` | Old decision deferred Huber delta, exponential tau and reciprocal margin; no obligation to implement extra row-wise parameters. |
| [R-I04](../../docs/development/reboot/requirements-inventory.md#r-i04) | Realization-aware public facade | WP11 accepted; C unaccepted | `pending` | Thin entry to the empirical outer loop, not a new global solver or universal convergence claim. |
| [R-I05](../../docs/development/reboot/requirements-inventory.md#r-i05) | Consistent final inverse state | PR #117 accepted; C unaccepted | `pending` | F1–F4: feasibility on returned weights, affine residual, diagnostic availability, zero-L2 and no-work conventions. |
| [R-I06](../../docs/development/reboot/requirements-inventory.md#r-i06) | Exact range and nullable diagnostics | PR #119 accepted; C unaccepted | `pending` | Genuine complete-value binary64 range vs intermediate overflow; old report v3 schema is not adopted automatically. |

## D. Numerical engineering and workflow

| ID | Capability / question | Historical outcome | Review status | Comment / open choice |
| --- | --- | --- | --- | --- |
| [R-E01](../../docs/development/reboot/requirements-inventory.md#r-e01) | Native artifact qualification | B era #88; later #109 | `pending` | Source/build/loaded binary identity differs from geometry correctness and reviewer acceptance; old heavy framework is not mandatory in reboot. |
| [R-E02](../../docs/development/reboot/requirements-inventory.md#r-e02) | Old exact-kernel performance | PR #87 merged | `pending` | Specific Fraction clipping optimization for historical exact machinery; relevant only if similar path is selected. |
| [R-E03](../../docs/development/reboot/requirements-inventory.md#r-e03) | Editable iteration and qualification boundaries | #113; reboot PR1 | `existing` | ADR 0027 already approved normal editable workflow and focused testing. Separating Python/native-contract checks from PR/release gates is still worth discussing; no path-filtered CI approved. Short delta from former `development-qualification-boundaries.md` captured here. |

## E. Open historical questions (not implemented)

| ID | Capability / question | Historical outcome | Review status | Comment / open choice |
| --- | --- | --- | --- | --- |
| [R-O01](../../docs/development/reboot/requirements-inventory.md#r-o01) | Global periodic complex | Historical research only | `pending` | ChemVoro long-term need; incidence-first vs direct semantic halfspace construction, periodic identity/winding and missing native facets. Former `global-topology.md` duplicates this except motivation, now recorded here. |
| [R-O02](../../docs/development/reboot/requirements-inventory.md#r-o02) | Voro++ patch policy and allocation ownership | D9/#64 open | `pending` | Upstream-unmodified vendor vs bounded downstream patches and recoverable `bad_alloc` ownership require separate decisions. |
| [R-O03](../../docs/development/reboot/requirements-inventory.md#r-o03) | Extreme power-weight range / upstream Voro++ | Investigation deferred | `pending` | Retained [investigation](potential-work/large-weight-numerics.md) adds native one-site r≈1e8 counterexamples and suspected squared-radius cancellation; root cause/upstream report/patch not approved. |
| [R-O04](../../docs/development/reboot/requirements-inventory.md#r-o04) | Prescribed measures and mixed inverse | Former roadmap only | `pending` | Research candidates without inherited architecture, solver API, priority or release. |
| [R-O05](../../docs/development/reboot/requirements-inventory.md#r-o05) | Other historical candidates | Open/deferred | `pending` | Repeated frames, exterior generators, descriptors, new domain forms and inverse formulations; split when representative use cases appear. |
| [R-O06](../../docs/development/reboot/requirements-inventory.md#r-o06) | PA-001–003 and unfinished gates | Open / unfinished | `pending` | Availability vs certification, loader compatibility, qualification cost; Checkpoint C and Phase D were not completed and do not define reboot gates. |

## F. Substantial reassessment additions

The detailed large-weight note is retained and already indexed under **R-O03**, avoiding a second decision row.

| ID | Capability / question | Historical outcome | Review status | Comment / open choice |
| --- | --- | --- | --- | --- |
| [P-01](../../docs/development/reboot/requirements-inventory.md#r-o01) | Geometry/graph layers and ChemVoro field integrals | Reassessment-only | `pending` | [Detail](potential-work/geometry-layers.md). Local cells vs contacts vs global complex vs observation graph; field integrals/QTAIM comparisons go beyond historical topology. |
| [P-02](../../docs/development/reboot/requirements-inventory.md#r-m01) | Optional geometry-output presets | Reassessment-only | `pending` | [Detail](potential-work/geometry-output-presets.md). `all`/`none` convenience with existing fine-grained flags/defaults; low priority; needs downstream demand. |
| [P-03](../../docs/development/reboot/requirements-inventory.md#r-o05) | Native query performance and contexts | Reassessment-only | `pending` | [Detail](potential-work/native-performance.md). Read-only context, 2D ghost batch overhead and structured-grid locate; different from R-E02 Fraction performance. |

**Reading order:** A → B → C → D, then open problems and independent additions. Update status/comment as decisions are made; link subsequent accepted ADRs or merged PRs without rewriting the historical outcome.
