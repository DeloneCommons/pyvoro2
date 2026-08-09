# Architecture

This document has three roles. It describes the **factual v0.6.3
implementation**, which is the software baseline used by the separator-inverse
manuscript, records the **v0.7.0 release architecture**, and explains
the **current v0.8 implementation** and accepted extension boundaries.

!!! note "Current and historical documentation"
    The v0.6.3 section remains a historical manuscript baseline. The v0.7
    sections explain the architecture established by the released v0.7.0
    contract. Current sections describe the feature-free v0.8 cleanup fixed by
    ADR 0006. Lifecycle status is finalized in the
    [v0.8 API inventory](api-inventory.md).

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
8. **Real downstream use validates stability.** The chemvoro-shaped integration
   workflow tests the current canonical contract.

## Architecture at a glance

| Layer | Historical v0.6.3 state | Current v0.8 contract |
|---|---|---|
| Native backends | Separate 3D and legacy planar Voro++ extensions | Preserve explicit dimensional capabilities and backend isolation |
| Forward Python API | Mature domain and operation layers, but asymmetric result containers | Return one dimension-neutral `TessellationResult` by default, with explicit supported raw output |
| Inverse API | Separator fitting under `pyvoro2.powerfit` with broad top-level re-exports | Stable high-level workflow at `pyvoro2.inverse`, advanced ownership at `pyvoro2.inverse.separator`, and no compatibility shim |
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
result object. The current v0.8 behavior is described separately below.

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
did not. The current v0.8 tree described below aligns the common result concept
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

The inverse flows use the current canonical names for clarity. The v0.6.3
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

## Current v0.8 implementation status

### Private pure-Python helper ownership

Private pure-Python implementation helpers now have one explicit package,
`pyvoro2._internal`. The package initializers contain no convenience imports:
internal callers import the concrete module that owns the behavior. The current
ownership is:

| Ownership | Modules | Reason |
|---|---|---|
| Dimension-neutral | `_internal.cell_output`, `_internal.inputs`, `_internal.power_input`, `_internal.validation`, `_internal.weight_transforms` | Raw-record post-processing is parameterized by measure and boundary keys; strict scalar/array validation and input coercion are parameterized by dimension; power-input resolution and weight/radius conversion have no dimension-specific geometry. |
| Spatial/3D | `_internal.spatial.domain_geometry`, `_internal.spatial.domain_utils`, `_internal.spatial.face_shifts` | These helpers use the 3D domain classes, three-component lattice operations, or realized face geometry. |
| Planar/2D | `_internal.planar.domain_geometry`, `_internal.planar.edge_shifts` | These helpers use the planar domain classes, two-component lattice operations, or realized edge geometry. |

The obsolete root helper modules and private modules under `pyvoro2.planar`
are absent rather than retained as forwarding shims. `_internal` is not public
API, and implementation-module metadata such as a public function's
`__module__` value does not make an internal path stable. Stable public
weight/radius functions continue to be exported from `pyvoro2` and
`pyvoro2.inverse`.

`pyvoro2.__about__` remains root-owned package metadata because the build
backend reads the version assignment from that file; it is not an
implementation-helper namespace. The compiled extension modules
`pyvoro2._core` and `pyvoro2._core2d` also remain at the package root and keep
their existing lazy loading paths.

### Lazy import boundaries

Plain `import pyvoro2` loads the public pure-Python forward/result surface and
the `pyvoro2.planar` namespace, but it does not import `pyvoro2.inverse`,
`pyvoro2._core`, or `pyvoro2._core2d`. Importing `pyvoro2.inverse` or
`pyvoro2.inverse.separator` also does not load either native extension.

The spatial wrapper imports `_core` only when `compute`, `locate`, or
`ghost_cells` first needs the 3D backend. The planar wrapper does the same for
`_core2d`. Documentation builds and inverse-only work can therefore import the
package without a compiled extension; a forward geometry operation raises an
informative `ImportError` when its required native module is unavailable. Lazy
loading does not change ownership: `_core` and `_core2d` remain root-owned
internal native extensions, while `_internal` owns only pure-Python helpers.

### Native construction safety boundary

The public spatial and planar wrappers apply strict source-type, shape,
finiteness, and range validation before native dispatch. Exact integer policy
lives in `_internal.validation`, shared forward coercion lives in
`_internal.inputs`, and dimension-specific geometry adapters provide validated
native bounds or an owned periodic snapshot.

Every function in the 12-route spatial and 6-route planar native construction
matrix then calls the common C++ preflight before constructing Voro++. The
preflight validates converted arrays and controls, checks every derived C++
`int` and byte-count operation, proves the finite constructor arithmetic used
by the current vendored sources, and enforces the aggregate 1-GiB cap on known
eager construction allocations. The direct `_core` and `_core2d` routes are
therefore defensive even when the public wrapper is bypassed. Exact original
Python type semantics remain the wrapper's responsibility because pybind11
conversion may already have erased that information. ADR 0010 fixes the
ordering, resource policy, source-trace requirement, and R3-A scope.

### Strict public values and pragmatic ownership

Public exact integers use non-Boolean index-protocol semantics, and public
Boolean flags accept only Python or NumPy Boolean scalars. Arrays are checked
for their original numerical category, shape, and finiteness before dtype
conversion. Boolean masks are likewise checked before conversion. The shared
rules live in `_internal.validation` and `_internal.inputs`; public forward,
domain, duplicate, normalization, diagnostic, and separator entry points apply
them before reductions, integer casts, linear algebra, solver loops, or native
dispatch.

Domain structure is canonical Python data: bounds, triclinic vectors, and
origins are owned nested tuples of built-in floats, while periodic flags are
owned tuples of built-in Booleans. Retained numerical inputs are owned
C-contiguous read-only arrays. This protects a frozen/value object from caller
mutation without deep-freezing raw nested tessellation records or
solver-created result graphs. `PeriodicCell` additionally requires a
right-handed basis at construction. Remapping proves every returned lattice
shift representable as signed int64 before conversion. Normalization treats
mutable raw cell records as a fresh public boundary, validates all consumed
integer metadata before topology-key construction, and rejects an
unrepresentable coordinate/tolerance quantization relationship before
constructing topology or applying annotations. This keeps normalization
independent of NumPy warning and floating-point error settings while leaving
the raw records themselves mutable. ADR 0011 fixes this R3-B contract and its
boundary from R4–R9.

### Certified periodic minimum-image geometry

The dimension-neutral private module `_internal.periodic_images` is the one
mathematical source for inferred periodic nearest images and for the distance
of periodic duplicate pairs that the current scanners evaluate when wrapping
is enabled (`wrap=True`, or `duplicate_wrap=True` in forward operations).
Spatial and planar domain adapters provide canonical row-oriented lattice
vectors and periodic-axis masks; they do not implement separate image
algorithms.

Rectangular 2D and orthorhombic 3D domains use exact per-axis floor/ceiling
choices. Fully periodic non-orthogonal 3D cells align the supplied binary64
coordinates and basis to exact dyadic integers, prepare an exact rational
basis inverse, derive a finite coefficient box from an incumbent norm bound,
and exact-compare every integer candidate in that box. Successful private
results include exact distance keys and deterministic work metadata. A bounded
seed can tighten the box, but it has no correctness authority.

Separator inference derives an orientation token from the resolved internal
site indices within the fixed resolved problem. External ID values remain
metadata and do not participate in geometric tie selection. This makes exact
ties respect lattice translation and pair reversal without introducing a
point-array permutation invariant or public tie mode. Explicit observation
shifts remain authoritative and bypass inference. The public `image_search`
parameter keeps its default and exact non-negative-integer contract but is only
a capped incumbent-seeding hint. If exact certification would exceed the
frozen private candidate budget or signed-int64 shift contract, the operation
raises a structured private runtime error without an approximate fallback.

When periodic wrapping is enabled (`wrap=True`, or `duplicate_wrap=True` in
forward operations), periodic duplicate distance evaluation consumes the same
primitive. With wrapping disabled, the established unwrapped Cartesian check
is preserved. R4 does not redesign the candidate scanner or public duplicate
policy. Complete periodic seam scanning, mandatory backend-safety policy
independent of `duplicate_wrap`, and generator containment remain R5. ADR 0012
fixes the exact problem, proof box, tie rule, resource/cache policy, and this
R4/R5 boundary.

### Neutral weight/radius transforms

The sole implementations of `weights_to_radii` and `radii_to_weights` now live
in the private shared module `pyvoro2._internal.weight_transforms`. The top-level
`pyvoro2` helpers import from that module directly. Separator problem and
active-set code also import the neutral implementation directly, without going
through a separator-owned module.

The v0.8 removal leaves the top-level, high-level inverse, and advanced
separator exports bound to the same neutral functions. The transforms reject
non-finite inputs and any arithmetic that would produce non-finite weights,
radii, or representation shifts. Import arrows point toward the implementation
provider:

```text
top-level pyvoro2 exports -----------------+
separator problem and active-set code -----+--> pyvoro2._internal.weight_transforms
pyvoro2.inverse exports -------------------+
forward power-input resolution ------------+
```

### Canonical separator implementation ownership

The current tree physically owns every separator-fitting implementation
module under `pyvoro2.inverse.separator`: observation resolution, objective
models, problem construction, fixed-observation solving, realization matching,
active-set refinement, reports, and result dataclasses. Those modules import
only canonical siblings or neutral/shared `pyvoro2` providers. In particular,
no module under `pyvoro2.inverse` imports the removed compatibility package.
The neutral transform implementation remains in
`pyvoro2._internal.weight_transforms`.

`pyvoro2.inverse` exposes only the normal fixed-observation workflow and
neutral transforms; advanced separator objects remain in
`pyvoro2.inverse.separator`. Issue #28 removed the v0.7-only facade, broad
top-level separator exports, and five historical core aliases without changing
the canonical implementation.

### Direct weight-first forward input

The spatial and planar `compute(...)` functions now accept direct mathematical
`weights=` in power mode. The dimension-neutral private module
`pyvoro2._internal.power_input` validates the input contract once and carries
three values together: supplied mathematical weights, resolved backend radii,
and the common representation shift. It delegates the conversion itself to
`pyvoro2._internal.weight_transforms` and has no separator or native-extension
dependency.

Both forward wrappers pass the resolved backend radii to every native power
call and to periodic face/edge image-shift reconstruction. Issue #8 completed
the forward-input part of WP-02 without adding weights to `locate(...)` or
`ghost_cells(...)` and without changing raw returns. It also left the common
result object to issue #9. Issue #11 subsequently completed the physical
separator-ownership move without changing solver behavior.

### Common forward result data contract

The current v0.8 tree defines the dimension-neutral
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
the documented raw/aligned invariants without normalizing arbitrary
backend-style dictionaries, recomputing geometry, or verifying geometric
validity. Later raw-cell mutation does not alter the aligned measure and
empty-mask snapshots; boundary access revalidates mutable boundary records
against those snapshots before returning them. Weight-first metadata is
validated against the shared weight-to-radius transform. Deep-copy and
same-version pickle reconstruction preserve the existing snapshot state,
including allowed raw-record divergence, while restoring owned read-only arrays
and capability state. The builder does no native work and does not trigger
diagnostics, normalization, or boundary annotation.

Both public `compute(...)` functions now build through that shared path and
return `TessellationResult` by default. `output='cells'` preserves the
characterized raw list/diagnostics-tuple behavior without rerunning native
computation or post-processing. Diagnostics computed directly or for a
tessellation check are stored in the structured result. Planar normalization
uses the same result and keeps internally requested temporary geometry out of
the final raw-cell capabilities.

Issue #28 removed the planar result alias and legacy return selector.
`TessellationResult` and the explicit `output=` rules remain unchanged;
explicit raw output with normalization fails clearly.

### Responsibility-based test ownership

The test tree follows the same responsibility boundaries as the implementation.
Dimension-neutral forward result, weight-input, and API contracts live in
`tests/forward/common`; explicitly 3D and 2D behavior lives in
`tests/forward/spatial` and `tests/forward/planar`, respectively. Canonical
separator tests live in `tests/inverse/separator`. End-to-end and genuine
cross-subsystem contracts, developer tooling, and randomized or independent
cross-wrapper checks live in `tests/integration`, `tests/tooling`, and
`tests/fuzz`.

Root `tests/conftest.py` contains only cross-suite pytest configuration and
fixtures. Ordinary imported support has an explicit subsystem owner, currently
`tests/fuzz/_support.py`. No compatibility directory is needed after issue #28;
the surviving raw-output contract is current forward behavior and is tested
under common forward ownership.

## Why v0.7 stabilization was needed

The v0.6.3 implementation is functional, but several details should be stabilized
before new inverse families are added.

### Result asymmetry

The v0.6.3 3D API normally returned raw records, while the planar API could
return a separate `PlanarComputeResult`. The current v0.8 tree resolves this
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
The current v0.8 tree resolves physical ownership under
`pyvoro2.inverse.separator`; the terminology migration remains separate.

### Ambiguous gauge language for disconnected observations

A common shift of every site weight leaves the complete power diagram unchanged.
If the informative separator graph is disconnected, additional independent
component offsets are not determined by those observations, but changing them
can alter competition between components and therefore the realized global
diagram. The API must report the distinction rather than call every component
shift harmless gauge.

### Broad top-level exports

The v0.6.3 top-level namespace re-exported many inverse implementation types.
v0.7 provided a bounded transition, and v0.8 removes those exports.

## Current public architecture

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

The current forward power API accepts mathematical weights directly in
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

The current API provides one inspectable conceptual contract across 2D and 3D.
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
the first observation family. The v0.7-only `pyvoro2.powerfit` facade, broad
separator-specific top-level exports, historical core aliases, and deprecated
planar selectors were removed in v0.8 under ADR 0006.

The physical ownership, canonical core names, and high-level convenience
surface are implemented in the current tree. Canonical code has no dependency
on the removed facade.

The separator workflow should be described using the following concepts:

- separator observations;
- observation/effective multigraph;
- implied weight differences;
- global gauge and unidentified component offsets;
- algebraic fit diagnostics;
- realized-boundary diagnostics;
- optional realization-aware refinement.

New documentation prefers `SeparatorObservations`, `SeparatorFitResult`, and
`fit_weights_from_separators`. The v0.7 historical aliases no longer resolve in
v0.8.

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
optional L2 regularization and no positive-strength scalar penalties.
Zero-strength penalties are mathematically absent and do not hide the view.
Hard bounds remain separate and the view explicitly distinguishes an
unconstrained normal equation from a constrained optimum. Huber mismatch and
positive-strength scalar-penalty models retain graph inspection but do not
expose a misleading fixed normal system.

[ADR 0007](decisions/0007-separator-objective-contract.md) fixes the common
objective semantics used by direct evaluation, quadratic solves, ADMM,
packaging, and reports: squared/Huber quadratic mismatch and L2 use the
conventional half factors; reciprocal penalties use a finite tangent
continuation; zero-strength penalties are absent; hard bounds use one shared
scale-aware float64 tolerance; and successful solver results require finite
reported soft objectives.

SciPy is imported only when sparse conversion or
`linear_backend='sparse'` is explicitly requested and is not a runtime
dependency. The public solver method and linear backend are independent:
`solver='direct'` performs the certified quadratic solve, while
`solver='admm'` executes ADMM and uses the selected backend for its weight
system and optional warm start. Dense means NumPy and never imports SciPy;
sparse means SciPy. There is no automatic or site-count-based backend switch,
and the active-set outer loop forwards both selections without changing its
own semantics.

[ADR 0009](decisions/0009-certified-scalar-proximal-solver.md) defines the
private scalar update used by ADMM. Ordinary mismatch-only squared and Huber
rows remain vectorized. Rows that need positive-strength scalar-penalty work
use a compiled coordinate specification, exact structural breakpoints,
one-sided derivative enclosures, scaled binary64 accumulation, and a certified
sign bracket backed by one compiled term kernel. A coordinate succeeds only
through proved exact point signs or localization between adjacent numeric
binary64 values; exhaustion and unresolved evaluation become the existing
structured `numerical_failure`. Adjacent endpoints are selected by a direct
termwise objective difference, and high precision is a bounded ambiguity
fallback rather than an iteration path: algebraic signs are exact dyadic
decisions and transcendental signs require an outward interval excluding zero.
Adjacent private certificates retain both endpoint enclosures and resolved
signs. Solver and
linear-backend APIs, ADMM decomposition/stopping options, result schemas, and
active-set semantics are unchanged.

### Layered inverse result contract

The current result vocabulary keeps these concerns distinct:

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

### Current compatibility boundary

The v0.8 implementation does the following:

- remove the documented v0.6.3 inverse imports after the bounded v0.7 shim;
- keep raw forward returns available through `output='cells'`;
- make `TessellationResult` and `pyvoro2.inverse` the normal new-user paths;
- retain aliases, changed defaults, warnings, and removals only in migration
  documentation and historical records;
- avoid encouraging new code to import separator-specific types from top-level
  `pyvoro2`;
- keep the paper's archived v0.6.3 environment independent of later internal
  refactors.

See [API lifecycle](api-lifecycle.md) for the compatibility policy.

## Downstream contract for chemvoro

chemvoro is intended to be a thin chemistry-facing layer. It supplies atomic
information and proposed interatomic separator positions; pyvoro2 supplies the
weighted geometry and inverse mathematics.

The current v0.8 contract lets chemvoro rely on:

1. stable association of coordinates, external atom IDs, and output cells;
2. direct forward computation from power weights;
3. stable access to cell measures, boundaries, and periodic image labels;
4. a preferred separator-fitting entry point independent of chemistry;
5. explicit global gauge and disconnected-component-offset metadata;
6. separate algebraic-fit and realized-geometry diagnostics;
7. JSON-/record-friendly outputs for caching and reporting;
8. no dependency on private backend radius shifts, solver internals, or record
   ordering accidents.

A repository-owned chemvoro-shaped integration workflow validates the preferred
public boundary without private imports.

## Release sequence from v0.8

### v0.8 cleanup and compatibility removal

ADR 0006 makes v0.8 a feature-free maintenance release. It removes the bounded
v0.7 compatibility layer, organizes tests by responsibility, moves root private
Python helpers under `pyvoro2._internal`, and resolves non-critical audit
findings. The helper move is complete in the current tree: shared code is
dimension-neutral, while genuine 3D and 2D behavior has explicit `spatial` and
`planar` ownership. Compiled `_core` and `_core2d` names remain private native
extension names; no public `pyvoro2.core` namespace is introduced.

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
- Private pure-Python helpers live under `pyvoro2._internal`; its package
  initializers do not re-export helper objects.
- Neutral weight/radius transforms do not depend on separator or native-backend
  modules.
- Inverse observation implementations may depend on forward computation and
  common diagnostics.
- The canonical implementation lives under `inverse.separator`; the v0.7
  compatibility package is absent in v0.8.
- Visualization remains optional and outside solver requirements.
- Chemistry-specific data and models remain downstream.
- Optional performance backends must not define the only public data format.

## Near-term non-goals

The current release line does not commit to:

- moving-site optimization;
- arbitrary user-defined objective callbacks;
- anisotropic, spherical, or non-Euclidean tessellations;
- GPU acceleration;
- a general computational-geometry framework competing with CGAL;
- guaranteed convergence of the realization-aware active-set loop;
- planar oblique-periodic support solely for symmetry with 3D;
- prescribed-measure work in the v0.8 release;
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
