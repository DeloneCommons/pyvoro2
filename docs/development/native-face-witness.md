# Private native 3D face witness

This is the observation prerequisite for WP5 G0 under
[#68](https://github.com/DeloneCommons/pyvoro2/issues/68), tracked by
[#47](https://github.com/DeloneCommons/pyvoro2/issues/47). The subsequent
[ADR 0021](decisions/0021-wp5-native-occurrence-and-exact-face-certification.md)
closes G0-N/G0-C/G0-O using this witness; the witness alone did not close those
gates. The [WP5 implementation](wp5-implementation.md) now consumes this witness
for requested periodic 3D face shifts and independent exact-consistency
diagnostics. The packet remains private; it does not expose unrequested public
vertices, adjacency, or shift capabilities.

## Entry points and evidence

The private `_core._observe_box` and `_core._observe_periodic` entry points
accept native-frame points, a permutation of dense IDs `0..n-1`, the existing
native domain/block controls, and optional native radii. They reuse the
ordinary construction preflights. Radii select power mode; they are already
the binary64 backend representation, not mathematical weights.

The returned packet owns its Python data after the containers are destroyed.
Its structure is private and may change at the next G0 review:

| Field | Evidence |
|---|---|
| `context` | Native domain, periodicity, mode, construction controls, actual block widths/reciprocals, mask/image-grid bounds, and triclinic seed tolerances |
| `sites` | Every persistent dense ID, indexed by ID, with actual stored native site/radius, primary block/index/slot; standard mode has `radius=None` |
| `cells` | A computation record for every stored generator, including failed/empty computations |
| `build` | Compiler, signed-native-int profile and arithmetic qualification metadata, with relevant source fingerprints |
| Cell `origins` | Ordered origin occurrences with packet-local token, semantic owner, actual binary64 plane, and legacy label |
| Cell `vertices_doubled` | Final `pts[4*v:4*v+3]`, indexed by the final native vertex identity |
| Cell `vertex_orders`, `adjacency` | Native vertex degrees and ordered edge connections |
| Cell `faces` | Ordered native vertex cycles, the checked token, every edge token, and the separately retained legacy owner |
| Cell `noninterference` | Checks against the matched ordinary native computation |
| Cell `tol`, `tol_cu`, `big_tol` | Actual receiver tolerances, kept separate from the seed's constructor scale |

Tokens identify occurrences within a cell computation. They are not particle
IDs, public face IDs, or lattice coefficients. Identical planes from different
cut calls remain different origin records, including calls that leave no final
face. Final face occurrences are never deduplicated.
For a failed computation, no final geometry is serialized; only the matched
computation status is checked, and geometry comparison fields are `None`.

## Source coupling

`cpp/native_witness.hpp` defines a neighbor-cell subclass. Its five-argument
`nplane` records the received normal, offset, and dense owner, then calls the
unchanged neighbor-cell clipping operation with the private token in the
neighbor-label slot. Voro++ carries that token through its existing copy,
relocation, marginal-vertex, and topology operations. No clipping operation is
reimplemented.

`cpp/native_witness.cpp` instantiates the existing compute source under
binding-owned names. Ordinary production template instantiations remain in
`v_compute.cc`, but every 3D translation unit now has the same qualified
noncontracting binary64 policy. The observation path preserves the native
input and insertion order and checks its result against an ordinary
computation. The authority argument below concerns the ordered arithmetic
before clipping, not an inference from matching final geometry.

Triclinic initialization needs an additional link. The actual native
`unit_voro` is a plain cell; assigning it to a neighbor cell gives zero labels.
The binding re-instantiates the existing `unitcell.hh`/`unitcell.cc` source
with the observing cell type. It does not maintain a second copy of the shell
algorithm. The replay uses the native seed's constructor tolerance scale and
the original plane-overload evaluation order. Before accepting seed tags, it
requires identical final vertex bits and indexed topology against the actual
native seed. The receiver copies the actual seed geometry and imports only
the checked tags. A mismatch is an explicit error.

The origin kinds currently have the following meanings:

| `kind` | Meaning |
|---|---|
| `particle` | An actual five-argument generator cut; owner zero is a valid dense generator |
| `triclinic_seed` | Observed self-image initialization support; semantic owner is the current generator, while legacy label is zero |
| `orthogonal_seed` | One of the six initialized box sides, retaining axis, sense, periodicity and side code |
| `construction_bound` | Temporary bound used while constructing the triclinic seed; it cannot survive as an accepted final support |

For an orthogonal seed, `periodic=False` identifies a real physical wall;
`periodic=True` identifies a self-periodic support. The side codes remain
`-1/-2`, `-3/-4`, `-5/-6` for lower/upper x, y, z. Neither a negative legacy
code nor a legacy zero is a semantic classifier.

A later coincident cut can replace the seed token: the native marginal-face
path can write the current cut ID while retaining the same geometry. The
observer preserves both origin occurrences and the token actually propagated
to the final face. It does not force an originally seeded support to keep its
initial label.

## Face association and noninterference

The face traversal follows the native indexed edge relations and ordering.
It checks every directed edge, including the closing edge, validates each
token, and requires one coherent origin around each complete cycle. Mixed,
unknown, construction-only, malformed, or unvisited edge state is an explicit
private failure. Reading just the first neighbor label is insufficient.

Matched ordinary and observing computations must agree on success/failure.
For successful cells, checks compare native vertex coordinates and volume
bitwise, and vertex degrees, full edge/relation topology, ordered face cycles,
and decoded legacy labels exactly. Native labels necessarily differ in the
instrumented cell because they carry tokens; decoding must recover the
ordinary labels. No tolerance is used to conceal a mismatch.

This comparison establishes noninterference for the particular compiled
computation. It is not a theorem that all supported compilers produce the
same floating geometry. A platform that produces a different observation or
fails the comparison must expose that evidence.

## Shared arithmetic policy

The original prerequisite compiled only the observer with explicit contraction
control. Ordinary producer expressions could contract while their observing
counterparts did not. Identical source and identical final polygons did not
establish identical historical plane operands. The repair deliberately changes
the ordinary 3D build policy; low-order coordinates, volumes, and topology near
native degeneracy can change relative to a contraction-permitted build.
Mathematical weights, supplied radii, periodic semantics, and public signatures
retain their existing meaning.

`cmake/NativeFP.cmake` defines the policy once and applies it to the entire
`_core` target and the standalone native test executable:

| Compiler | Compile policy | Link/IPO policy |
|---|---|---|
| GCC | `-fno-fast-math -ffp-contract=off -fno-lto` | Same flags at the driver link; IPO off |
| Clang / Apple Clang | `-fno-fast-math -ffp-contract=off -fno-lto` | Same flags at the driver link; IPO off |
| MSVC | `/fp:strict /GL-` | `/LTCG:OFF`; IPO off |

`_core` uses `pybind11_add_module(... NO_EXTRAS ...)` to exclude pybind11's
implicit LTO. Target and configuration-specific CMake IPO properties are off;
explicit compiler/link options also counter inherited LTO flags. The GNU/Clang
link policy prevents fast-math startup code from enabling FTZ/DAZ. Optimization
remains enabled (`-O3`, or the normal MSVC Release optimization). The separate
`_core2d` target is outside this repair.

CMake force-includes the binding-owned `cpp/native_fp_contract.hpp` into every
participating translation unit, including unchanged vendored sources. It
rejects fast-math macros, non-IEC binary64 doubles, and `FLT_EVAL_METHOD != 0`.
Thus wider intermediate evaluation is excluded rather than merely rounded at
assignments. No vendored file is modified. Build options injected after the
policy, source pragmas overriding it, or another compiler family require new
qualification; a metadata string is not evidence for such a custom build.

The qualified runtime environment is round-to-nearest, ties-to-even, with
gradual underflow for both inputs and results, and normal nontrapping FP
execution. Private entry points check the rounding mode and exercise subnormal
input and output arithmetic, refusing FTZ/DAZ. Neither native path changes the
floating environment during computation. The argument is conditional on
defined execution and finite observed operands; it does not supply the later
complete overflow/error envelope. FP exception flags are not geometric state
and are not promised identical across label bookkeeping.

The relevant build semantics are described by the
[GCC optimization options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html),
[Clang floating-point model](https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior),
[MSVC floating-point options](https://learn.microsoft.com/en-us/cpp/build/reference/fp-specify-floating-point-behavior),
and [pybind11 CMake helpers](https://pybind11.readthedocs.io/en/stable/cmake/index.html).
Verbose package and standalone build logs qualify the effective commands on
the supported CI matrix. Source hashes in the packet still cover only the
listed files; exact repository source and build logs are required alongside it.

### Translation-unit coverage

| Arithmetic | Ordinary compilation | Observing compilation / shared operation |
|---|---|---|
| `voro_compute::compute_cell` | Explicit standard/power box and periodic instantiations in `v_compute.cc` | Same source included with renamed compute/container types in `native_witness.cpp` |
| `radius_poly` inline arithmetic | Instantiated with compute in `v_compute.cc`, and in binding/container callers | Original methods inherited by the observer; instantiated under the same target policy |
| Unit-cell construction | `unitcell.cc`, containing a plain `voronoicell` | Same source included with observing seed type in `native_witness.cpp` |
| Particle clipping | `cell.cc` specialization `nplane<voronoicell_neighbor>` | The same linked specialization; only the neighbor payload differs |
| Seed clipping | `cell.cc` specialization `nplane<voronoicell>` | Neighbor specialization from that same translation unit, with label-only callbacks |
| Container/image coordinates | `container.cc`, `container_prd.cc`, inline methods from their headers, and `v_compute.cc` | Inherited original container methods and source-coupled compute, all under the same policy |
| Initial bounds, tolerance scale, intersections | `cell.cc`, `unitcell.cc`, container/compute source and inline callers | Covered by the whole-target policy, including `bindings.cpp` and `native_witness.cpp` |

### Producer/observer induction

This is a source/build argument for independent review, not a claim that two
template instantiations emit identical instructions. With contraction,
reassociation and excess precision excluded, a conforming compilation must
preserve the same binary64 results for corresponding ordered scalar operations
even when register allocation or inlining differs. Disabling IPO removes an
additional cross-translation-unit optimization boundary from that argument.

1. The ordinary and observing containers receive identical constructor inputs,
   insertion order and stored site/radius bits. The observing adapters inherit
   the original container implementation; they expose radius operations without
   rewriting them. Initial compute constants, masks, queues and worklists come
   from the same source and numerical inputs.
2. Induct over geometric and algorithmic state: coordinates, indexed topology,
   tolerance values, cached vertex classification slots, `up`, clipping stacks,
   allocation counters, compute masks/queues, image storage and radius state.
   Neighbor payload arrays are separate from that state. All seven particle
   `nplane` call sites in `v_compute.cc` use the same arithmetic expression and
   ordered decisions in the two instantiations. `r_scale` and `r_scale_check`
   remain distinct expression trees.
3. The observer stores each received normal/offset unchanged and passes those
   doubles into the **same linked neighbor-cell clipping specialization** as
   the ordinary producer. The dense owner is recorded before replacing its
   label slot with the occurrence token. In `cell.cc`, `p_id` is passed only to
   neighbor bookkeeping; `cell.hh` neighbor callbacks copy/set/manage label
   storage without reading its values into geometric decisions.
4. Corresponding clipping therefore has the same geometric state transitions
   and success/failure. Subsequent radius tests, block/image visits and cut
   choices agree, extending the induction to the complete operation sequence.
   Private allocation failure, checked invalid provenance, or nonfinite plane
   operands refuses a packet; those cases are not approximately accepted.
   This does not assert a general finite-output guard: for example, volume
   overflow remains outside this bounded repair and the deferred envelope.

This establishes the proposed authority route from observing operands to
ordinary producer operands. The exact final geometry/label checks remain
additional runtime sentinels. They cannot substitute for steps 1–4.

There is no portable interception hook in the existing ordinary producer:
clipping dispatch is static and the real `unit_voro` type is fixed. Directly
recording both ordinary and observing histories would require another
instrumented producer or platform-specific linker instrumentation. This bounded
repair instead retains one witness architecture and supplies the source/build
induction, native arithmetic discriminators, and matched computation checks.

### Triclinic seed induction

The seed proof includes the different plain/neighbor clipping instantiations.
Identical constructor inputs give the same constructor tolerance scale and
initial bounds. Both types use the shared `init_base`. The included `unitcell`
source selects the same shells in the same order, forms the same
`i*bx+j*bxy+k*bxz` image vectors, and uses the same paired signs. The plain
three-argument `plane` overload and its observing counterpart evaluate the
same ordered `x*x+y*y+z*z` expression under the common policy.

Both clipping specializations instantiate the same `cell.cc` algorithm. Plain
`n_*` callbacks are empty; neighbor callbacks maintain only tag arrays and their
storage. They neither alter geometric arithmetic nor branch on labels.
Inductively, indexed geometry, caches/stacks, clipping decisions, intersection
tests, the next shell choice and the stopping decision agree after every cut.
The result is the same indexed final geometry with an independently transported
provenance payload. This relies on defined execution, including valid native
allocation/bookkeeping, rather than on face-count agreement.

The existing exact vertex, topology and tolerance checks against **both** real
container seeds remain. Assignment still copies the actual plain seed geometry,
preserves receiver tolerances, and imports only validated replay tags. It never
replaces ordinary seed geometry with a reconstructed polygon.

## Regression qualification

The original power fixture invokes the actual vendored `radius_poly` methods.
With `r_i=r_j=2**27` and squared distance `1`, separate rounding gives `0`
through `r_scale` and `1` through `r_scale_check`. Its radius square is exact,
so that example distinguishes source association but does not reliably detect
contraction. It remains a separate regression.

The new equal-radius fixture uses `2**27+1`: its exact square is one greater
than its rounded binary64 square. Noncontracting execution still gives `+0`
and `1`; permitted fused evaluation can differ according to the contraction
chosen by the compiler. Tests exercise the actual radius methods and a real
two-generator power computation, rather than an algebraically rewritten model.

A standard-mode fixture uses displacement
`(1-3*2**-27, 2-2**-27, 0)`. The noncontracting squared sum is
`0x1.3fffffb000000p+2`; fusing either nonzero square with the other rounded
square gives `0x1.3fffffb000001p+2`. The test reads the actual compute cut,
checks its normal/offset bits, and retains ordinary/observing parity. Native
standalone tests also cover the actual radius methods and plane overloads.

Focused Python characterization lives in
`tests/forward/spatial/test_native_witness.py`. Native corruption, copy,
marginal, collapse, and allocation tests live in
`tests/native/test_native_witness.cpp`, built and run by
`tests/tooling/test_native_witness_cpp.py`. The latter uses a separate CMake
project with checks active in Release builds. The existing sanitizer CI job
includes the witness tests; the normal CI matrix exercises the package across
Linux, macOS, Windows and supported Python versions.

The seed-owner regression uses the dyadic lattice `(4096,0,0)`,
`(1024,4096,0)`, `(512,1536,4096)` and radii `(0,40960)`. The surviving
generator's seed faces retain legacy zero while their observed semantic owner
is one. Initial qualification found ten surviving seed faces on Linux/GCC
and Windows/MSVC, and eleven on macOS 15/ARM64. On each platform the matched
ordinary and observed computations agreed; the count is not a portable
contract. The regression checks the origins and exact rational noncollinearity
of every surviving seed polygon without an area threshold. CI logs its full
private packet, including compiler metadata and round-trip binary64 values,
before running the full suite so platform differences remain inspectable.
The lower-dimensional power fixture separately retains a genuine cut
owner whose own computation returns no volumetric cell. Neither fixture
implements positive-measure or reciprocity policy.

## Return to G0 review

Review the two binding files, their source includes, `CMakeLists.txt`, and the
focused tests together. Follow the actual observed plane into the unchanged
vendored `v_compute.cc`, `rad_option.hh`, `cell.hh` and `cell.cc`; follow seed
provenance into `unitcell.hh`, `unitcell.cc`, and `container_prd.hh`.

This prerequisite record alone did not prove the producer-compatible
attribution, separate E/S exact ideal classification, finite candidate regions,
true-assignment retention, semantic reciprocity coverage, or source-centered
public integration. Their subsequently closed **specification** separates
actual native support N, exact native-effective ideal E and exact
public-semantic ideal S in
[ADR 0021](decisions/0021-wp5-native-occurrence-and-exact-face-certification.md);
production implementation and acceptance remain pending. A final native
polygon plus its observed support plane is evidence, not a replacement for
that contract.
