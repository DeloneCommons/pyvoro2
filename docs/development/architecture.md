# Architecture

This document has three roles. It describes the **factual v0.6.3
implementation**, which is the software baseline used by the separator-inverse
manuscript, records the **v0.7.0 release architecture**, and explains
the cleanup and extension boundaries accepted for later releases.

!!! note "v0.7.0 release documentation"
    The v0.6.3 section remains a historical manuscript baseline. The v0.7
    sections describe the released v0.7.0 contract. The canonical inverse
    namespace and common result contract are fixed by ADR 0004 and ADR 0005;
    lifecycle status is finalized in the
    [v0.7 API inventory](api-inventory.md).

## Architectural principles

1. **Forward tessellation is a first-class core.** pyvoro2 remains useful
   without inverse fitting.
2. **Power weights are mathematical; radii are a backend representation.**
3. **Two and three dimensions are explicit.** Common concepts should align,
   but unsupported parity must not be implied.
4. **Observation data and realized geometry are separate layers.**
5. **Exact inner problems and realization-aware outer algorithms are separate.**
6. **Diagnostics are part of the scientific result.** Non-identifiability,
   infeasibility, empty cells, and wrong periodic images should be inspectable.
7. **API evolution is explicit and time-bounded.** v0.7 provides one documented
   transition release; ADR 0006 removes compatibility-only routes in v0.8.
8. **Real downstream use validates stability.** The chemvoro integration is an
   intended test of the v0.7 contract.

## Architecture at a glance

| Layer | Historical v0.6.3 state | Current v0.7 contract |
|---|---|---|
| Native backends | Separate 3D and legacy planar Voro++ extensions | Preserve explicit dimensional capabilities and backend isolation |
| Forward Python API | Mature domain and operation layers, but asymmetric result containers | Return one dimension-neutral `TessellationResult` by default, with explicit supported raw output |
| Inverse API | Separator fitting under `pyvoro2.powerfit` with broad top-level re-exports | Move ownership to `pyvoro2.inverse.separator`; keep a bounded v0.7 compatibility shim |
| Downstream boundary | Rich records and reports exist, but some callers still need implementation knowledge | Support chemvoro through documented weights, IDs, geometry, and diagnostic contracts |

## Historical implementation baseline: v0.6.3

### Native backend and build layer

The build is defined by `CMakeLists.txt` and scikit-build-core.

- `cpp/bindings.cpp` builds the 3D `_core` pybind11 extension against the
  vendored Voro++ 3D sources.
- `cpp/bindings2d.cpp` builds the planar `_core2d` extension against the
  vendored legacy Voro++ 2D sources.
- `vendor/voro++/` contains the backend snapshot and its upstream licenses.

The Python layer imports the native modules lazily. Importing pyvoro2 and
building its documentation can therefore work without an extension present,
while geometry operations raise an informative error if no compiled backend is
available.

### Spatial forward layer (3D)

The top-level `pyvoro2` namespace is the 3D public surface.

- `domains.py` defines `Box`, partially periodic `OrthorhombicCell`, and fully
  periodic triclinic `PeriodicCell`.
- `api.py` implements `compute`, `locate`, and `ghost_cells`.
- `_inputs.py` centralizes public coercion and validation.
- `_domain_geometry.py` resolves domain geometry for internal consumers.
- `_face_shifts3d.py` reconstructs periodic image labels on realized faces.
- `_cell_output.py` and related helpers shape backend output.

In the v0.6.3 baseline, the ordinary 3D compute path returned a list of Python
cell records. Diagnostics, normalization, and annotations were requested
through separate functions or compute options; that release had no universal
result object. The current v0.7 behavior is described separately below.

### Planar forward layer (2D)

`pyvoro2.planar` is an explicit namespace with its own backend-specific domain
and operation code.

- `planar/domains.py` defines planar `Box` and rectangular periodic
  `RectangularCell`.
- `planar/api.py` implements planar `compute`, `locate`, and `ghost_cells`.
- `planar/_edge_shifts2d.py` reconstructs periodic image labels on edges.
- `planar/result.py` defined `PlanarComputeResult`, a separate optional
  structured wrapper for raw cells, diagnostics, and normalized output.
- planar diagnostics, normalization, validation, and duplicate checks live in
  their corresponding modules.

The v0.6.3 planar API therefore had a structured result path that the 3D API
did not. The current v0.7 tree described below aligns the common result concept
without pretending that every backend capability is identical.

### Shared post-processing and scientific utilities

The top-level Python package adds behavior that is not merely a direct binding:

- duplicate detection and pre-backend safety checks;
- tessellation diagnostics and strict validation;
- vertex and topology normalization;
- face/edge geometric annotations;
- periodic graph-ready neighbor image labels;
- optional 2D and 3D visualization helpers.

These utilities depend on public cell records and domain semantics. They are
part of the forward scientific interface and must remain usable by downstream
packages independently of inverse fitting.

### Separator-based inverse layer

The v0.6.3 inverse implementation lived in `pyvoro2.powerfit`.

| Module | v0.6.3 responsibility |
|---|---|
| `constraints.py` | Resolve pair indices, periodic shifts, connector geometry, target values, and confidence weights. |
| `model.py` | Define mismatch losses, hard feasible sets, soft penalties, and regularization. |
| `problem.py` | Centralize prediction formulas, algebraic diagnostics, bounds, gauge canonicalization, objective evaluation, and public problem export. |
| `solver.py` | Solve the fixed-observation inverse problem and package low-level results. |
| `realize.py` | Compute a power tessellation and match requested pairs/images to realized boundaries. |
| `active.py` | Run the realization-aware hysteretic active-set outer loop. |
| `report.py` | Convert structured numerical results into JSON-friendly records and reports. |
| `transforms.py` | Convert between weights and backend-compatible radii. |
| `types.py` | Shared public dataclasses. |

That package exposed much of this surface both through
`pyvoro2.powerfit` and by top-level re-export from `pyvoro2`. That is convenient
for existing users, but it creates a large accidental top-level stability
surface.

### Core data flows

The inverse flows use the current canonical v0.7 names for clarity. The v0.6.3
implementation exposed the same roles through the historical names recorded in
the [advanced separator API inventory](api-inventory.md#advanced-separator-api).

#### 3D forward computation

```text
points + IDs + domain + mode + radii/options
    -> Python validation and domain resolution
    -> _core Voro++ execution
    -> Python cell records
    -> optional face shifts / diagnostics / normalization / annotations
```

#### Planar forward computation

```text
points + IDs + planar domain + mode + radii/options
    -> planar validation and domain resolution
    -> _core2d execution
    -> planar cell records
    -> optional edge shifts / diagnostics / normalization
    -> raw records or PlanarComputeResult
```

#### Fixed-observation separator fit

```text
raw pair observations
    -> resolved SeparatorObservations
    -> FitModel + SeparatorFitProblem
    -> graph/connectivity and hard-feasibility analysis
    -> quadratic or iterative solver
    -> SeparatorFitResult
    -> optional conversion to radii
```

#### Realization-aware fit

```text
fitted weights/radii
    -> forward power tessellation
    -> requested pair/image matching
    -> RealizedPairDiagnostics
    -> optional active-mask update and refit
    -> SelfConsistentPowerFitResult
```

The fixed-observation fit and the realization-aware outer loop are intentionally
separate computations.

## Current v0.7 implementation status

### Neutral weight/radius transforms

The sole implementations of `weights_to_radii` and `radii_to_weights` now live
in the private shared module `pyvoro2._weight_transforms`. The top-level
`pyvoro2` helpers import from that module directly. Separator problem and
active-set code also import the neutral implementation directly, without going
through a separator-owned module.

`pyvoro2.powerfit.weights_to_radii`, `pyvoro2.powerfit.radii_to_weights`, and
the historical `pyvoro2.powerfit.transforms` module remain compatibility routes
to the same function objects. The compatibility module contains no numerical
formulas. The neutral transforms reject non-finite inputs and any arithmetic
that would produce non-finite weights, radii, or representation shifts. Import
arrows point toward the implementation provider:

```text
top-level pyvoro2 exports -----------------+
separator problem and active-set code -----+--> pyvoro2._weight_transforms
pyvoro2.powerfit compatibility exports ----+
forward power-input resolution ------------+
```

### Canonical separator implementation ownership

The current v0.7 tree physically owns every separator-fitting implementation
module under `pyvoro2.inverse.separator`: observation resolution, objective
models, problem construction, fixed-observation solving, realization matching,
active-set refinement, reports, and result dataclasses. Those modules import
only canonical siblings or neutral/shared `pyvoro2` providers. In particular,
no module under `pyvoro2.inverse` imports `pyvoro2.powerfit`.

`pyvoro2.powerfit` and its historical direct submodules are now explicit
forwarding shims. Their exports are the same class and function objects as the
canonical modules; the shims contain no numerical formulas, solver paths,
report builders, or duplicate dataclasses. Broad historical top-level exports
also bind directly to `pyvoro2.inverse.separator`, so importing the canonical
package does not load the compatibility namespace. The neutral transform
implementation remains in `pyvoro2._weight_transforms`.

The historical `pyvoro2.powerfit` package attribute is resolved lazily on
first access. Plain `import pyvoro2` and canonical-only imports therefore do
not load the compatibility package, while existing `pv.powerfit` calls still
receive the historical module. The attribute remains outside `pyvoro2.__all__`.

Issue #12 adds canonical primary definitions for the five accepted core names
and keeps their historical names as identity aliases. `pyvoro2.inverse`
exposes only the normal fixed-observation workflow and neutral transforms;
advanced separator objects remain in `pyvoro2.inverse.separator`. Importing
the historical `pyvoro2.powerfit` package emits a hidden-by-default
`DeprecationWarning` with the fixed v0.8 removal horizon.

### Direct weight-first forward input

The spatial and planar `compute(...)` functions now accept direct mathematical
`weights=` in power mode. The dimension-neutral private module
`pyvoro2._power_input` validates the input contract once and carries three
values together: supplied mathematical weights, resolved backend radii, and
the common representation shift. It delegates the conversion itself to
`pyvoro2._weight_transforms` and has no separator or native-extension
dependency.

Both forward wrappers pass the resolved backend radii to every native power
call and to periodic face/edge image-shift reconstruction. Issue #8 completed
the forward-input part of WP-02 without adding weights to `locate(...)` or
`ghost_cells(...)` and without changing raw returns. It also left the common
result object to issue #9. Issue #11 subsequently completed the physical
separator-ownership move without changing solver behavior.

### Common forward result data contract

The current v0.7 tree now defines the dimension-neutral
`pyvoro2.TessellationResult` in `pyvoro2.result` and re-exports the identical
class from `pyvoro2.planar`. A single private builder constructs aligned
measures and empty-cell state by final external ID, including cells that the
backend omitted from raw output. The result owns read-only construction-time
snapshots of aligned numerical arrays while retaining the exact mutable
raw-cell list.

Capability flags for boundaries and periodic shifts are validated keyword-only
construction state supplied explicitly by the builder, so replacement, empty
input, and all-hidden output preserve the distinction between unavailable
geometry and requested geometry with no records. Direct construction validates
the raw/aligned invariants. Later raw-cell mutation does not alter the aligned
measure and empty-mask snapshots; boundary access revalidates mutable boundary
records against those snapshots before returning them. Weight-first metadata
is validated against the shared weight-to-radius transform. Deep-copy and
pickle reconstruction preserve the existing snapshot state, including allowed
raw-record divergence, while restoring owned read-only arrays and capability
state. The builder does no native work and does not trigger diagnostics,
normalization, or boundary annotation.

Both public `compute(...)` functions now build through that shared path and
return `TessellationResult` by default. `output='cells'` preserves the
characterized raw list/diagnostics-tuple behavior without rerunning native
computation or post-processing. Diagnostics computed directly or for a
tessellation check are stored in the structured result. Planar normalization
uses the same result and keeps internally requested temporary geometry out of
the final raw-cell capabilities.

`PlanarComputeResult` is an identity alias to `TessellationResult`.
Planar `return_result=` remains compatibility-only, emits a deprecation warning,
and delegates to the explicit `output=` selection rules. The historical
normalization override for an explicitly false legacy selector is retained;
conflicting explicit selectors and explicit raw output with normalization fail
clearly.

## Why stabilization is needed

The v0.6.3 implementation is functional, but several details should be stabilized
before new inverse families are added.

### Result asymmetry

The v0.6.3 3D API normally returned raw records, while the planar API could
return a separate `PlanarComputeResult`. The current v0.7 tree resolves this
asymmetry through the common default result while retaining explicit supported raw
output.

### Radius-first v0.6.3 forward input

The mathematical inverse variable is a power weight, but the v0.6.3 forward API
accepted only `radii=` because Voro++ represents weights as squared radii. The
v0.7.0 resolves that baseline limitation for both
`compute(...)` functions.

### Separator-specific public organization

The v0.6.3 `powerfit` surface grew around one observation family. Prescribed
cell measures should not be implemented as a second unrelated module with its
own geometry parsing, gauge policy, result vocabulary, and failure reporting.
The current v0.7 tree resolves physical ownership under
`pyvoro2.inverse.separator`; the terminology migration remains separate.

### Ambiguous gauge language for disconnected observations

A common shift of every site weight leaves the complete power diagram unchanged.
If the informative separator graph is disconnected, additional independent
component offsets are not determined by those observations, but changing them
can alter competition between components and therefore the realized global
diagram. The API must report the distinction rather than call every component
shift harmless gauge.

### Broad top-level exports

The top-level namespace currently re-exports many inverse implementation types.
The preferred v0.7 organization should be clearer while keeping compatibility
imports available.

## Public architecture for v0.7

### Dependency direction

The intended responsibility graph is:

```text
Domains and site configuration
        |
        v
Forward tessellation core ------> common result concepts
        |                               |
        |                               v
        |                      diagnostics / topology / measures
        |
        v
Inverse weighted-tessellation layer
    - separator observations (implemented)
    - cell measures (later)
    - mixed observations (later)
        |
        v
Observation-specific + common inverse results
        |
        v
Compatibility facades and downstream adapters
```

Dependencies should point downward. The forward core must not depend on inverse
solvers. Observation blocks may use forward computation and common result
concepts, but should not duplicate domain or periodic-image logic. Neutral
weight/radius transforms are a shared provider for forward and inverse callers
and must not depend on `pyvoro2.powerfit` or either native extension.

### Shared geometry input contract

All forward and inverse workflows need a consistent association among:

- site coordinates;
- dimension;
- domain;
- optional external IDs;
- periodic image convention;
- backend-independent weights.

The exact Python object used to group these values is provisional. The stable
requirement is that inverse methods and downstream packages do not repeatedly
re-parse or reorder the same geometry independently.

### Weight-first forward route

The current v0.7 forward power API accepts mathematical weights directly in
addition to the existing `radii=` route. Power mode requires exactly one
representation. Standard mode rejects both representations rather than silently
ignoring unused weighted inputs.

At the backend boundary, a single global shift can be chosen so that

\[
r_i = \sqrt{w_i + c}
\]

is real and non-negative. The implementation uses the established
`weights_to_radii(weights)` default, and the chosen shift is representation
metadata that does not change the diagram. Supplying both weights and radii is
rejected before native computation. The private resolution carrier keeps the
validated weights, backend radii, and shift available for the later common
result implementation without exposing a new public weights object.

### Shared forward result contract

The v0.7 line provides one inspectable conceptual contract across 2D and 3D.
It does not erase backend differences, but users can obtain:

- raw or structured cell records;
- dimension and domain metadata;
- site/ID association;
- computation mode and weight/radius representation metadata;
- cell area in 2D or volume in 3D through a common **cell measure** vocabulary;
- boundary records and boundary measure where requested;
- periodic neighbor image labels;
- empty-cell information;
- tessellation and normalization diagnostics.

[ADR 0005](decisions/0005-tessellation-result-contract.md) selects one public
`pyvoro2.TessellationResult` for both dimensions. The common class, private
construction path, structured default, and explicit `output='cells'` raw-output
route now exist. The result keeps dimension-specific geometry explicit and
does not compute unrequested expensive data merely to fill optional fields.

The outer result is structurally immutable. Owned aligned arrays are read-only,
while nested raw cell records are not deep-frozen or defensively copied solely
to claim immutability. The public documentation states contained mutability
explicitly.

### Preferred inverse organization

[ADR 0004](decisions/0004-canonical-inverse-namespace.md) selects
`pyvoro2.inverse` as the canonical home of math-aligned
inverse concepts and `pyvoro2.inverse.separator` as the implementation owner for
the first observation family. `pyvoro2.powerfit` becomes a thin
compatibility-only shim during v0.7 and is removed in v0.8. Broad
separator-specific top-level exports, historical core aliases, and deprecated
planar selectors follow the same transition schedule under ADR 0006.

The physical ownership, one-way shim direction, canonical core names, and
high-level convenience surface are implemented in the current tree. The
historical names are identity aliases for v0.7, while `pyvoro2.powerfit` and
broad top-level separator exports are deprecated compatibility routes that are
removed in v0.8.

The separator workflow should be described using the following concepts:

- separator observations;
- observation/effective multigraph;
- implied weight differences;
- global gauge and unidentified component offsets;
- algebraic fit diagnostics;
- realized-boundary diagnostics;
- optional realization-aware refinement.

New documentation prefers `SeparatorObservations`, `SeparatorFitResult`, and
`fit_weights_from_separators`. Historical names such as
`PairBisectorConstraints` resolve to those same canonical objects during v0.7.
Compatibility code imports from the canonical namespace; canonical code never
imports from `powerfit`.

### Inspectable algebraic operators

The separator quadratic problem has incidence and weighted-Laplacian structure.
Issue #14 implements the public inspection contract through two provisional,
problem-owned views:

- `SeparatorFitProblem.observation_graph` exposes the oriented observation
  multigraph, row identity and periodic shifts, affine coefficients, implied
  difference targets, effective edge weights, positive-confidence informative
  mask, and existing connectivity/identification diagnostics;
- `SeparatorFitProblem.quadratic_operator` exposes observation and
  L2-regularized right-hand sides, matrix-free products, dense NumPy matrices,
  optional lazy SciPy conversions, and component/nullspace metadata.

The incidence shape is `(n_sites, n_observations)`, with `+1` at the first
endpoint and `-1` at the second endpoint of each observation column. Repeated
rows and periodic parallel observations are not collapsed. Zero-confidence
rows remain columns but have zero effective weight and do not connect the
informative graph.

The quadratic view is conservative: it is available for `SquaredLoss` with
optional L2 regularization and no scalar penalties. Hard bounds remain separate
and the view explicitly distinguishes an unconstrained normal equation from a
constrained optimum. Huber mismatch and scalar-penalty models retain graph
inspection but do not expose a misleading fixed normal system.

SciPy is imported only when sparse conversion or explicit sparse solving is
requested and is not a runtime dependency. Issue #17 reuses the same quadratic
operator for `solver='sparse'`, removes one gauge anchor per unregularized
effective component, and performs a SciPy sparse-direct solve. The default
`solver='auto'` remains dense; v0.7 deliberately has no automatic size
threshold. Sparse execution is limited to the primary fixed-observation
quadratic path and does not extend ADMM, the active-set outer loop, or dynamic
workflows.

### Layered inverse result contract

The v0.7 result vocabulary keeps these concerns distinct:

- **state**: fitted weights, representation shift/radii, and identification
  metadata;
- **objective**: total and observation-specific contributions;
- **observations**: targets, predictions, residuals, and confidence;
- **algebraic diagnostics**: graph, incidence, cycle/projection, and
  identifiability information where meaningful;
- **geometry**: cells, measures, empty-cell flags, and realized boundaries when
  computed;
- **realization diagnostics**: requested shift, other shift, visibility,
  clearance, and active-set path where applicable;
- **solver diagnostics**: termination, iterations, warnings, and convergence
  metrics.

Not every inverse method has every layer. Missing concepts should be absent or
explicitly unsupported rather than filled with misleading placeholders.

### Compatibility boundary

The v0.7 implementation does the following:

- keep documented v0.6.3 inverse imports functioning through a one-way shim;
- keep raw forward returns available through `output='cells'`;
- make `TessellationResult` and `pyvoro2.inverse` the normal new-user paths;
- document aliases, default changes, warnings, and the v0.8 removal of
  historical inverse and planar transition paths;
- avoid encouraging new code to import separator-specific types from top-level
  `pyvoro2`;
- keep the paper's archived v0.6.3 environment independent of later internal
  refactors.

See [API lifecycle](api-lifecycle.md) for the compatibility policy.

## Downstream contract for chemvoro

chemvoro is intended to be a thin chemistry-facing layer. It supplies atomic
information and proposed interatomic separator positions; pyvoro2 supplies the
weighted geometry and inverse mathematics.

The v0.7 contract lets chemvoro rely on:

1. stable association of coordinates, external atom IDs, and output cells;
2. direct forward computation from power weights;
3. stable access to cell measures, boundaries, and periodic image labels;
4. a preferred separator-fitting entry point independent of chemistry;
5. explicit global gauge and disconnected-component-offset metadata;
6. separate algebraic-fit and realized-geometry diagnostics;
7. JSON-/record-friendly outputs for caching and reporting;
8. no dependency on private backend radius shifts, solver internals, or record
   ordering accidents.

A repository-owned chemvoro-shaped integration workflow is part of the v0.7
stabilization review and validates the preferred public boundary without private
imports.

## Extension path after v0.7

### v0.8 cleanup and compatibility removal

ADR 0006 makes v0.8 a feature-free maintenance release. It removes the bounded
v0.7 compatibility layer, reorganizes the flat tests, moves root private Python
helpers under `pyvoro2._internal`, and resolves non-critical audit findings.
Compiled `_core` and `_core2d` names remain private native extension names; no
public `pyvoro2.core` namespace is introduced.

### v0.9 prescribed cell measures

The second inverse family reuses the same geometry and result contracts: fixed
sites and domain, unknown weights, and target areas/volumes. Its first steps are
measure extraction, target validation, residual evaluation, and a validated
sensitivity operator before a nonlinear solver is exposed.

### v0.10 mixed observations

Only after separator and measure workflows both exist should a generic public
observation-block protocol be frozen. The first mixed solver uses fixed sites
and unknown weights only, with explicit scaling between separator and measure
residuals.

### Additional unknowns and observations

Site motion, centroids, sections, and other research extensions should enter as
new explicit unknown or observation families. They must not be hidden options
inside the stable weights-only solver.

## Dependency rules

- Native/backend modules do not depend on high-level inverse code.
- Forward domain and result concepts do not depend on observation families.
- Neutral weight/radius transforms do not depend on separator or native-backend
  modules.
- Inverse observation implementations may depend on forward computation and
  common diagnostics.
- During v0.7 only, `powerfit` compatibility code delegates to
  `inverse.separator`; the canonical implementation never depends on
  `powerfit`. v0.8 removes the compatibility package.
- Visualization remains optional and outside solver requirements.
- Chemistry-specific data and models remain downstream.
- Optional performance backends must not define the only public data format.

## Near-term non-goals

The v0.7 line does not commit to:

- moving-site optimization;
- arbitrary user-defined objective callbacks;
- anisotropic, spherical, or non-Euclidean tessellations;
- GPU acceleration;
- a general computational-geometry framework competing with CGAL;
- guaranteed convergence of the realization-aware active-set loop;
- planar oblique-periodic support solely for symmetry with 3D;
- prescribed-measure work before the v0.8 cleanup is complete;
- mixed-observation work before the prescribed-measure family exists.

## Keeping this document current

When implementation resolves a provisional choice under the active release
plan:

1. record the resolution in the linked issue and plan revision log;
2. add or update a decision record if the choice is durable;
3. change the relevant target description into factual current-architecture
   text;
4. update user guides and API reference;
5. add completed user-visible behavior to the changelog;
6. retain historical context in the decision record and archived plan rather
   than maintaining parallel obsolete descriptions.

The [development workflow](development-workflow.md) defines how architecture,
plans, issues, documentation, changelog entries, and release review move
together.
