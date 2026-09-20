# Private native 3D face witness

This is the observation prerequisite for WP5 G0 under
[#68](https://github.com/DeloneCommons/pyvoro2/issues/68), tracked by
[#47](https://github.com/DeloneCommons/pyvoro2/issues/47). It does not close
G0-N/G0-O or implement certified face-image reconstruction. The public forward
routes, output options, face shifts, normalization, and diagnostics do not
consume this witness.

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
| `context` | Native domain, periodicity, mode, construction controls, and actual triclinic seed tolerances |
| `sites` | Every persistent dense ID, indexed by ID, with its actual stored native site and radius; standard mode has `radius=None` |
| `cells` | A computation record for every stored generator, including failed/empty computations |
| `build` | Compiler and arithmetic qualification metadata |
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
binding-owned names. Ordinary production template instantiations and their
compiler settings remain separate. The observation path preserves the native
input and insertion order and checks its result against an ordinary
computation.

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

## Arithmetic and qualification

The observer requires binary64 doubles, round-to-nearest, and gradual
underflow. Its build excludes uncontrolled reassociation and records its
contraction policy for `native_witness.cpp`. Linked clipping code (`cell.cc`)
retains ordinary producer settings; `fp_contract_scope` makes that boundary
explicit. The recorded source hashes cover the named files only, so the
repository commit and verbose build/CI logs remain necessary qualification
evidence. The ordinary producer is left unchanged; a discrepancy between the
two computations is refused rather than accepted approximately.

The power-offset fixture invokes the vendored `radius_poly` methods directly.
Under strict unfused evaluation, `r_i=r_j=2**27` and input squared distance
`1` yield `0` through `r_scale` and `1` through `r_scale_check`. The observer
must retain the actual value supplied by its compute call, without replacing
either expression by an algebraically equivalent formula.

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
is one. The lower-dimensional power fixture separately retains a genuine cut
owner whose own computation returns no volumetric cell. Neither fixture
implements positive-measure or reciprocity policy.

## Return to G0 review

Review the two binding files, their source includes, `CMakeLists.txt`, and the
focused tests together. Follow the actual observed plane into the unchanged
vendored `v_compute.cc`, `rad_option.hh`, `cell.hh` and `cell.cc`; follow seed
provenance into `unitcell.hh`, `unitcell.cc`, and `container_prd.hh`.

Still unproved here are the complete native arithmetic envelope,
positive/zero/unresolved measure contract, finite candidate bound,
true-assignment retention, semantic reciprocity coverage, and the
source-centered public integration. A final native polygon plus its observed
support plane is evidence for those arguments, not their replacement.
