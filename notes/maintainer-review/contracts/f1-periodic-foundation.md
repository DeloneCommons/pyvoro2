# F1 periodic foundation — candidate contracts and P0 evidence

**Historical P0 research deliverable; candidate contracts were not collectively approved.** This is the bounded
[P0 deliverable (#153)](https://github.com/DeloneCommons/pyvoro2/issues/153),
subordinate to accepted ADRs [0027](../../../docs/development/decisions/0027-v0.9-reboot-from-v0.8.md),
[0028](../../../docs/development/decisions/0028-upstream-voro-vendor-policy.md),
and [0029](../../../docs/development/decisions/0029-image-qualified-geometry-and-optional-global-complex.md).
At the time of this study, v0.8 behavior remained in force. The then-current
[sequencing draft](../../../docs/development/plans/functional-sequencing.md)
and this note approve no API, algorithm, native patch, or F1/G1 implementation.
Historical WP2–4 ([#54](https://github.com/DeloneCommons/pyvoro2/issues/54),
[#56](https://github.com/DeloneCommons/pyvoro2/issues/56),
[#58](https://github.com/DeloneCommons/pyvoro2/issues/58)) and ADRs 0018–0026
are evidence only; see the [requirements](../../../docs/development/reboot/requirements-inventory.md)
and [evidence index E02–E04](../../../docs/development/reboot/evidence-index.md#e02).

**2026-11-27 update:** P0 was delivered in [PR #154](https://github.com/DeloneCommons/pyvoro2/pull/154).
Subsequent accepted [ADR 0030](../../../docs/development/decisions/0030-exact-user-lattice-and-native-admission.md)
and [ADR 0031](../../../docs/development/decisions/0031-qualified-native-periodic-qr-frame.md)
establish A1 and A2, including the narrow R1 correction. The A2 replay recovered
seven of the original discrepancy records; 18 remain. The observations, candidate
semantics, fixtures, validation record and replay below are retained as original
baseline evidence, not descriptions of the current tree. Consult the current
[sequencing draft](../../../docs/development/plans/functional-sequencing.md) for
remaining proposed work; numerical contracts still require focused approval.

## Coordinate authorities: proposed distinctions, P0 baseline inventory

Use row vectors: `x = o + f @ A`, image displacement `s_user @ A`.
Original user rows/order, origin, and original binary64 Cartesian operands
define source geometry. Exact means the rational values of those bits, not
their decimal spelling or a rounded matrix product.

| Authority | Current evidence available | Proposed boundary / missing evidence |
| --- | --- | --- |
| User/source `A`, `o`, original sites | `PeriodicCell` owns tuples; `PreparedGenerators.input_points_cart` and `TessellationResult.sites` own read-only arrays. | Retain originals through inference; preparation must not replace their scientific meaning. |
| Private computational `B = U @ A` | No reducer or persisted `U` in this baseline. Existing exact CVP works in the supplied basis. | If approved separately, exact unimodular integer `U`, inverse, and exact `B`; transport `s_user = s_reduced @ U` before public range checks. This is neither a public lattice abstraction nor the native basis. |
| Numerical native frame | `PeriodicCell._rotation_to_internal` and `_NativePeriodicSnapshot.from_raw` duplicate Gram–Schmidt; snapshot owns rotation, source vectors/origin and six lower-triangular parameters. | Record numerical transform and provenance separately from exact source/computational geometry. A numerical frame need not be an isometry merely because it was named a rotation. |
| Prepared/primary and submitted | `PreparedGenerators` owns `primary_points_cart`, `native_points`, Python `remap_shifts`, IDs and optional `backend_radii`. Forward preparation actually pre-remaps; the current `remap_internal` docstring's contrary statement is stale. | Preserve the actual operands sent to each native operation, including temporary roles. These arrays are preparation evidence, not native insertion disposition. |
| Actually stored and reported | Persistent C++ `compute_cells_impl` reads `loop.pos`; `build_cell_dict` emits that site's coordinates and floating absolute vertices. Ghost routes instead pass submitted query coordinates to `build_cell_dict`/empty records. Python transforms triclinic sites/vertices back to Cartesian. | A returned nonempty **persistent compute** cell provides a stored-site observation before reporting transformation. Successful/empty ghost site fields are query-anchor metadata, not stored-coordinate readback. No general persisted insertion/chart-transition record, stored radius array, or pre-transform vertex snapshot exists. Empty/omitted persistent cells do not supply missing evidence. |
| Weights and radii | `ResolvedPowerInput`/result retain mathematical input weights, prepared backend radii and common representation shift. Direct-radius input has no `input_weights`. | Preserve source weights, or exact squares of supplied binary64 radii for direct-radius semantics; distinguish exact squares of actually stored radii. `sqrt(w + shift)` rounding can change ideal effective geometry. Exactifying a rounded `r*r` does not recover the exact square. Stored radii are not generally returned. |

The ideal geometry of rounded backend operands is a separate mathematical
problem from source geometry; neither is automatically the observed floating
Voro++ boundary set. No mandatory runtime S/E/N certification stack is proposed.

## Candidate user operations and validity

Propose four `PeriodicCell` helpers, with strict finite real `(n,3)` inputs,
including `(0,3)`, owned nonaliasing binary64 outputs of the same shape, and
a Boolean `return_shifts` on wrapping:

| Candidate | Exact authority before rounding |
| --- | --- |
| `cart_to_fractional(points)` | Solve `f* = (x* - o*) @ inverse(A*)`, including exact subtraction. |
| `fractional_to_cart(fractional)` | Evaluate `o* + f* @ A*`. |
| `wrap_fractional(fractional, *, return_shifts=False)` | `s = floor(f*)`, `r* = f* - s`. |
| `wrap_cart(points, *, return_shifts=False)` | Solve on original Cartesian bits, choose exact `s`, evaluate `w* = x* - s @ A*`. Do not compose rounded public helpers. |

The view is componentwise nearest-even binary64; `0 <= r* < 1` holds exactly.
For example, `f=-2^-55` has shift `-1`, remainder `1-2^-55`, and rounded view
`1.0`. With `x=1`, `o=2^-55`, `A=I`, exact shift is zero although a rounded
fractional view is one. Do not clamp, snap, increment shifts, or promise
unconditional float-view idempotence/composition. Exact reconstruction of a
rounded `w_hat` differs from source by its view error; ordinary NumPy
reconstruction adds its own arithmetic error.

Keep shifts as arbitrary-size integers for decisions. Only requested exposed
shifts must fit signed int64 `[-2^63,2^63-1]`, checked before casting; an
unreturned out-of-range shift must not defeat a representable wrapped view.
For `A=I`, `o=0`, `x=(2^63,0,0)`, the exact wrapped view is zero; requesting
the shift must fail, while an unrequested shift need not be materialized.
An unavailable/nonfinite view or exposed-shift overflow is an operation
failure. No new public exact-number type is presumed.

Proposed mathematical validity is finite binary64 inputs with **exactly
nonzero determinant**, either sign, preserving row order. Current construction
instead requires a positive floating determinant, norm/SVD checks,
`relvol >= 1e-12`, `cond <= 1e15`, and warns above `cond=1e10`.
Mathematical validity must remain distinct from native admissibility: pure
coordinate operations need no Voro++; existing positive-diagonal, containment,
duplicate, arithmetic and allocation safety checks must remain effective at
native use. Moving/revising these boundaries needs explicit approval.

User wrapping selects the user parallelepiped. Existing `remap_cart`,
`remap_internal`, `wrap_internal` and internal-coordinate helpers select or
describe the numerical backend frame/primary chart. They are not silently
redefined. Backend snapping policy, including zero epsilon, needs its own
review; the historical numerical endpoint target is not exact user wrapping.

## Shift reference, identity, evidence and lifetime

Propose a named reference on every future image claim. For an owner pair,
the source convention is `x_j_source + s_source @ A - x_i_source`;
primary and stored-native conventions use their respective anchors and basis.
If exact chart relations are independently established as
`x_source_i = x_primary_i + k_i @ A` and
`x_primary_i = x_stored_cart_i + h_i @ A`, then
`s_source = s_native + k_i + h_i - k_j - h_j` after native-basis transport.
Snapping/frame error can invalidate those exact equalities. Python preparation
shift `k` alone is insufficient: `h` needs independently established native
insertion/chart evidence. Matching output numbers alone proves no disposition.

Retain the legacy optional raw `adjacent_shift` reference to the cell-dictionary
site anchors until a separate migration decision. Preserve its existing user
basis/sign, external IDs and mutation behavior; do not relabel it source-exact.

Future evidence must distinguish known owner/image (including proven zero),
physical domain wall, unresolved/ambiguous owner or image, exhausted budget,
unsupported input, and internal invariant failure. Never substitute zero or a
wall for missing evidence. Current 3D face annotation assigns zero to negative
labels and empty geometry; 2D tries resolving negative labels but also assigns
zero when unresolved. Those static observations do not establish true walls.
Single-site finite-box native controls below are genuine walls by construction.

Local occurrence keys should be snapshot-scoped `(snapshot, cell_slot)`, plus
face/edge slot, vertex slot and cycle position where relevant. Owner IDs or
rounded coordinates are not unique occurrence keys. Before reporting transforms,
ID remapping, annotation, normalization or public mutation can destroy consumed
evidence, G1a must capture only the fields it consumes:

- Original and effective generator operands, mathematical weights/direct radii,
  actual prepared radii, internal/external ID correspondence and role.
- Native local vertices, ordered raw face/edge cycles and neighbor labels,
  indexed by occurrence slots; stored-site/radius/disposition status **if
  obtainable**, otherwise an explicit absence and its route.
- Domain kind, finite bounds/wall geometry, origin and periodic-axis mask;
  source/native lattices and transforms, preparation shifts, independently
  justified chart transitions and numerical/model assumptions. Domain adapters
  already provide bounds; preparation retains `domain_kind`/`periodic_axes`.
- Failure/refusal provenance, unresolved labels, budgets and evidence lifetime.

This is the smallest **input-evidence packet**, not a G1a output schema or
matching algorithm. Existing `TessellationResult` arrays are already owned and
read-only, while its exact raw `cells` list and nested dictionaries are mutable;
normalization can shallow-copy cells or mutate in place. Snapshot only consumed
fields, with an explicit lifetime. No blanket deep copy or mandatory G1a work in
ordinary `compute()` is proposed.

Complete image-candidate claims require a proof-complete finite region or
explicit refusal for the admitted claim. An arbitrary finite search cube is
insufficient. Existing certified nearest-image CVP is not enumeration of all
realized contacts; multiple images per pair and nonzero self-images are valid.

## Reproducible baseline characterization

Live `dev` was checked before work and was
`d3b70ba4582a4ebb1a382e6d8a5671d126e1c8a6` (2026-11-22 UTC). All production
source inspected and executed below is that revision. The adjacent
[isolated replay](f1_periodic_characterization.py) emits lossless hexadecimal
float inputs/observations, rational oracle values, route, checks, revision,
environment and module paths. No bug is asserted as desired behavior.

The replay resolves its checkout root from its own file location and obtains
Git HEAD there, independently of the process working directory. Before running
cases and before emitting JSON, it checks every loaded `pyvoro2` Python module
against its expected file in that checkout's `src/pyvoro2` tree. The report's
`source_provenance` records the root, resolved Python module paths and whether
the working tree is dirty. A different working directory is supported with
the correct editable checkout installed; no import-path override is applied.
With `--native`, actual extension paths are recorded and may legitimately be
outside `src/`. These checks establish Python source location, **not native
binary/source identity**; native build provenance remains independently unproven.

```bash
python notes/maintainer-review/contracts/f1_periodic_characterization.py > p0.json
python notes/maintainer-review/contracts/f1_periodic_characterization.py --native > p0-native.json
```

Both commands intentionally **exit 1 on this baseline** for reported
discrepancies; they are outside mandatory pytest discovery, with no xfails or
CI changes. Setup/provenance failures instead **exit 2**, with an explicit
stderr diagnostic and no characterization JSON. A completed report exits zero
only if its discrepancy checks all pass. Default skips native execution;
`--native` requires both compiled modules and does not silently skip.
Promotion owner is the maintainer accepting
F1-A: move corrected source/remap cases into mandatory A3 regressions and frame
controls into A2 tests after approving their contracts. Historical targets in
the script remain explicitly candidates, not adopted epsilon policy.
Historical representative/shift comparisons are descriptive metadata only;
the discrepancy checks use independent range and exact transported tangential
invariants, allowing A's original tiny x seam to snap to zero.

| Case and exact input | Independent oracle / requirement | Actual observation and route |
| --- | --- | --- |
| #60 orthorhombic: `OrthorhombicCell(((0,1),)*3)`, sites `(2^-41,1/4,1/4)`, `(1/2,1/4,1/4)` | `d=1/2-2^-41` lies strictly between zero and half a period, so shift zero is uniquely nearest among all integers; competing squared gap `1-2d=2^-40`. | Public `resolve_separator_observations` returns shift `(-1,0,0)`, delta `(-1/2,0,0)`. Directly run, Python source; snapping precedes exact CVP. No fitting/CVP change. |
| #60 triclinic API: `PeriodicCell(I)`, same exact original sites | Independently assessed against source shift zero, `d=1/2-2^-41`, gap `2^-40`, without using the orthorhombic result as an oracle. | Separately recorded public separator call returns shift `(-1,0,0)`, delta `(-1/2,0,0)`. Direct Python run; the same source-semantic defect is observed, not accepted as correct. |
| #65: rows `(1,0,0),(-1/2,1,0),(0,0,1)`; point `(3/16,-2^-55,3/8)` | Allow existing default y snapping to zero; solve `q @ A = (3/16,0,3/8)-reported` by independent Fraction Gaussian elimination. A legitimate lattice translation requires integral `q`. | Public and detached snapshot return `(0,0,3/8)`, shift zero; `q=(3/16,0,0)` is nonintegral. Direct compiled standard dispatch also receives/stores/reports x=0; structured source stays original. Loss of `3/16` is separate from permitted tiny y snapping. |
| #67 A: rows `(1,0,0),(1/4,1,0),(0,0,1)`; point `(-2^-55,-2^-55,3/8)` | Numerical backend half-open range and preservation of tangential residual; historical candidate `(0,0,3/8)`, shift zero. | Public/snapshot `eps=0` return `(1,0,3/8)`, shift `(-1,0,0)`: excluded endpoint. Default and `1e-12` pass this historical target. Direct Python runs include translations `(0,0,0),(2,-1,3),(-2,1,-3)`. |
| #67 B: rows `(1,0,0),(-1/2,1,0),(0,2^-54,1)`; point `(3/16,-2^-55,-2^-55)` | Same backend requirements; historical candidate `(3/16,0,0)`, shift zero. | `eps=0`: `(11/16,1,0)`, shift `(-1,-1,0)`, fails range **but preserves x=3/16 after exact shift transport**; default/`1e-12`: `(0,0,0)`, loses x residual. Direct public/snapshot Python runs, same translated controls. |
| #67 seams: `t=2^-1074`; each axis receives `[-t,0,t,nextafter(L,0),L,nextafter(L,+inf),-1/8]`, `L=1,2` | Half-open numerical range, positive subnormal retained; ordinary negative quotient has shift `-1`. No new negative-subnormal epsilon policy is inferred. | At `L=1`, 2D rectangular/3D orthorhombic return excluded upper `1` for `-t`; triangular public/snapshot return zero. At `L=2`, all four return negative `-t` (quotient underflows to negative zero). Positive interior and ordinary-negative controls pass. Direct Python runs. |
| #67 coupled underflow: rows `(2,0,0),(-1/2,2,0),(1/4,-1/2,2)`; base `(3/16,3/8,5/8)`, replace each axis in turn by `[-t,t,-1/8]` | Same range/interior/ordinary-negative invariants; historical canonicalization is reported separately. | All three axes leave `-t` negative, with zero shift; positive-subnormal and ordinary-negative controls pass on public/snapshot Python routes. No native computation or new epsilon policy claimed. |
| Near-parallel frame: rows `(1,1,0),(1,1+2^-30,0),(0,0,1)` | Exact determinant `2^-30`; an isometry has orthogonal rows and triangular volume equal to source volume. Independent sign-normalized nonpivoted NumPy QR control. | Both Gram–Schmidt paths: max `abs(T T^T-I)` ≈ `4.76837e-7`, native parameter volume/source volume ≈ `1025.00000048`. QR: orthogonality ≈ `4.44e-16`, relative volume error ≈ `3.98e-8`. Direct arithmetic only, **not a Voro++ tessellation result** or universal QR qualification. |

Bounded frame controls directly run: `I`; rows `(1,1,0),(-1,1,0),(0,0,2)`;
`diag(2^-16,1,2^16)`; and exact cubic shear rows
`(1,0,0),(8,1,0),(0,0,1)`. Public, snapshot and QR pass the stated arithmetic
controls. The anisotropy/shear controls do not establish arbitrary native
admissibility. Historical left-handed variants are inapplicable to current
construction (positive determinant restriction, statically inspected).
Exact-view examples above are mathematical witnesses. The int64 example was
also directly run: current `PeriodicCell(I).remap_cart` raises for `2^63`
with either setting of `return_shifts`; source inspection identifies the early
`floor_to_int64` call. The four proposed helpers do not exist. No execution of
those future APIs is claimed.

Compiled 2D/3D standard single-site unit boxes have unit measure and genuine
negative wall labels. Mutating the caller input or raw returned site leaves the
owned read-only `result.sites` intact; raw nested sites remain mutable. This
bounded output-aliasing seam was directly run. Unresolved-label zero assignment
is a static source finding, not imported native evidence. All historical #67
fixtures/targets were inspected in its PR diff; current observations are fresh.

## Original P0 handoff and decisions pending at that baseline

| Proposed contract / current behavior | Exact affected references; evidence / existing coverage | Compatibility risk / gate |
| --- | --- | --- |
| A1 exact user validity/coordinates / current right-handed floating admission, no user helpers | `domains.py:PeriodicCell.__post_init__`; `periodic_images.py:_prepare_basis`; E02; `test_periodic_cell_degeneracy.py`, `test_periodic_from_params.py`, `test_periodic_images.py`. | Smallest first A1: private exact source-basis solve/validity plus the four helpers and owned views, including signed bases and range checks. Preserve `from_params` and native safety; do not import reduction or remap repair. Maintainer approval of signatures, view/failure semantics and placement of admission checks is required. |
| A2 numerical frame / duplicated nonisometric Gram–Schmidt witness | `domains.py:PeriodicCell._rotation_to_internal`, `to_internal_params`; `spatial/domain_geometry.py:_NativePeriodicSnapshot.from_raw`; frame replay and existing periodic/from-params tests. | Preferred **sign-normalized nonpivoted QR**, now supported by baseline arithmetic evidence, remains pending review. Pro numerical review must qualify scale, right-handed regressions, parity/reflections and local face winding before changing any frame. |
| A3 source authority and backend residual / snapping changes problem | `inverse/separator/constraints.py:_derive_connector_geometry`, `_maybe_remap_points`; `generator_preparation.py:_prepare_coordinates`; spatial/planar `domains.py:remap_cart`, spatial `PeriodicCell.remap_internal`; #60/#65/#67 replay; `test_constraints.py`, `test_generator_preparation.py`, `test_periodic_remap.py`, planar `test_domains.py`. | Separate narrow separator source-consumer correction from coupled backend repair. Stop for epsilon-policy or shift-reference decisions; retain mandatory duplicate safety. Review explicit-shift/reversal cases, origins, 2D/3D differences, standard/weight/direct-radius paths. |
| C evidence/identity/lifetime / mutable raw output and incomplete stored provenance | `result.py:TessellationResult`; `normalize.py:normalize_vertices`, `normalize_edges_faces`; `spatial/face_shifts.py:_add_periodic_face_shifts_inplace`; `planar/edge_shifts.py:_add_periodic_edge_shifts_inplace`; `duplicate_scanning.py`; `cpp/bindings.cpp:compute_cells_impl`, both bindings' `build_cell_dict`; result/normalization/face/edge tests plus bounded native replay. | Decide snapshot fields, unknown/wall statuses and legacy shift migration separately. Pro mathematical/provenance review must establish complete search claims, integer transport and native chart assumptions before G1a admission. Existing tests do not fill the missing insertion record. |

Paths without a prefix in this matrix are under `src/pyvoro2/`, or its
`_internal/` subtree for geometry/preparation/image/shift/scanning helpers;
test basenames refer to existing forward/inverse suites. No new implementation
issues or full A1–A3 prompts are part of P0.

Further pending gates:

- **B1:** exact deterministic LLL at delta `3/4` is a preferred candidate after
  bounded comparison; [R-L02b](../review-register.md) remains **pending**. No
  reducer, fallback, native rebasing or algorithm is approved here.
- **C4:** analysis and a reviewed disposition are **mandatory before F1→G1a**.
  Name G1a/G1b consumers and justify the native-error model before deciding
  whether a reusable exact finite-box translation-recovery primitive is needed.
  Alternatives are F1-C code, a justified G1a-specific alternative, or concrete
  evidence permitting deferred code. A unique solution inside a supplied box
  does not establish the producer's error bound. P0 neither implements nor
  completes C4.
- **G1a/G1b/source diagnostics:** observed contacts, optional global incidence
  and optional independent source-semantic diagnostics remain separate.
  F1-A numerical/source review, B/C mathematical/integration review and F1→G1a
  evidence/provenance review need independent Pro-level scrutiny. Accepting
  P0 is not accepting any later scope or a source-completeness guarantee.

## Validation record and limits

Executed on Linux x86_64, CPython 3.12.14, NumPy 2.5.5, GCC-built editable
`pyvoro2 0.9.0.dev0`, both native extensions. Setup used
`CC=gcc CXX=g++ python -m pip install -e '.[all]'` in an isolated venv after
the default build selected an unavailable `clang++`. No package dependency
was added to the project. The replay's pure arithmetic/source paths and its
three native controls are distinguished above; no near-parallel tessellation,
power-mode requalification, upstream comparison, sanitizer or platform matrix
was run as part of the replay. Local checks do not establish remote CI or
maintainer acceptance.

Validation commands (venv executables; production tree unchanged):

- `python -m pytest -q`: **2780 passed, 2 skipped**, 6 environment warnings,
  55.02 s on the remediation rerun. Skips are the optional independent `pyvoro`
  cross-checks; warnings
  concern unwritable IPython state using temporary storage.
- `python notes/maintainer-review/contracts/f1_periodic_characterization.py --native`:
  complete JSON, **48 route records, 27 with discrepancies**, expected exit 1.
  The default pure-Python run completes with 45 records, 26 with discrepancies,
  expected exit 1, explicitly skipping native execution.
- `python -m flake8 src tests tools benchmarks examples notes/maintainer-review/contracts/f1_periodic_characterization.py`:
  pass; `python tools/export_notebooks.py --check`: pass.
- `zensical build --clean --strict`: pass; relative file links in both modified
  Markdown files checked separately and resolve; `git diff --check`: pass.
  Initial obsolete `mkdocs build --strict`/`tools/gen_readme.py --check`
  attempts were inapplicable: current dev uses Zensical and no README generator.

Focused M1/M2 remediation was based on reviewed PR HEAD
`6c7204b8468a18eae226edf1f51796970828849a`; only this note and the replay changed.
`python -m pytest -q /workspace/work/test_p0_remediation.py` ran a local
five-case harness: **5 passed**. It independently checked checkout revision
and Python module paths, execution inside a foreign Git checkout, rejection
of an unrelated copied installation and of one mixed-origin internal module
(both exit 2, stderr only), both #60 routes against the exact oracle, and
unchanged pre-existing numerical records. The harness is scratch validation,
not new repository infrastructure. Direct commands also run:

- From `/workspace/pyvoro2`, `python notes/maintainer-review/contracts/f1_periodic_characterization.py --native > /workspace/work/p0-remediation-native.json`:
  expected exit 1, counts above, 53 Python source paths checked, both external
  editable extension paths recorded.
- From `/workspace/work`, `python /workspace/pyvoro2/notes/maintainer-review/contracts/f1_periodic_characterization.py > /workspace/work/p0-remediation-other-cwd.json`:
  expected exit 1, correct checkout revision and source paths, native skipped.

Both pre-commit reports accurately mark the worktree dirty; this is source
location evidence, not an assertion that every file equals committed HEAD.

No complete native requalification, installed-wheel/distribution check,
notebook execution or remote CI result is claimed. Independent PR review and
maintainer approval of the proposed contracts and A1 boundary remain required.
