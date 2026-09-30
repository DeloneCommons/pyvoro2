# WP8 implementation and qualification

The normative specification is [issue #79](https://github.com/DeloneCommons/pyvoro2/issues/79).
Base: `6a57db7da693a40a777c9a6e3cfb4df588248d13`. This is an implementation
map and proof record, not independent acceptance.

WP8 was subsequently independently accepted and merged, as recorded in the
[active plan](plans/v0.9.md). Issue #88's
[artifact qualification](native-qualification.md) remains a separate gate.
WP8 keeps its own spatial/planar source-enclosure components and the bounded
final-transform association/FMA allowance below; it does not inherit WP5/WP6
occurrence qualification merely because they share a native module.

## Architecture and constraints

Keep the existing native locate selection and native Cartesian output. A small
binding adapter verifies the actual persistent population, checks query integer
arithmetic before execution, and returns actual storage and producer operands.
WP4 recovers requested owner-image coefficients from a complete source-derived
Cartesian enclosure anchored at the original persistent row. Public query
wrapping is a separate exact affine solve. Ghost query views are added only
after all existing WP7 checks and retained-row selection.

No vendor modification, dependency, public selector, ownership solver, WP9
removal, or acceptance action is authorized. Native safety, private proof
resources and public int64 materialization have separate failure meanings.

## Implementation sequence

- [x] Independent Fraction oracle and mandatory red public fixtures: exact
  query wrapping, native owner-position seam, insertion omission, large filtered
  ghost shifts, sentinels, selectors and source identity.
- [x] Exact query views and ghost integration in `_internal/query_metadata.py`,
  both APIs and arbitrary-precision locate preparation. Verify without deriving
  expected values from production wrapping.
- [x] Native insertion/query guard and source packet in `cpp/locate_source.hpp`,
  existing six locate bindings, and `_internal/locate.py`. Keep stock selection
  and qualify source-dependent enclosure and WP4 uniqueness failures.
- [x] Adversarial/seeded/domain and selector coverage, native stock parity,
  source/build qualification and relevant predecessor noninterference.
- [x] ADR 0018, plan, inventory, lifecycle, architecture, public guides/reference,
  docstrings and changelog reconciliation.
- [ ] Full suite, lint, generated checks, strict docs, direct wheel and sdist
  wheel installed tests with optional dependencies present/absent; review and PR.

## Deliberate review focus

Check exact rectangular span operands, copied triclinic owner images,
query-remap double counting, original-row association before external IDs,
signed-int64 cancellation and materialization after ghost filtering. A
nonperiodic insertion omission cannot become not-found. Zero-query calls do
not construct a native container. A proof-invariant exception retains its
reason and cause rather than being relabeled as a geometric input error.

## Source-to-read map and enclosure derivation

The six binding routes still invoke the stock `find_voronoi_cell` exactly once
per query, in input order, with unchanged prepared points/radii, block counts,
capacity, query coordinates and output-frame conversion. No vendor file is
modified. Observation is outside the selection computation:

| Source | Obligation |
|---|---|
| `container.cc` / `container_2d.cc`: `put_remap`, `remap`, `find_voronoi_cell` | Replay actual Step/Mod insertion, then verify each original internal ID, block, coordinate bits and radius bits in storage before querying. Final owner image adds query removal and at most one search-period displacement per periodic axis. |
| `container_prd.cc`: `put_locate_block`, `remap`, both `find_voronoi_cell` functions | Coupled z/y/x removal, actual primary-storage association, and original-query final assembly. The selected slot may be an image; its copied coefficient is not reduced to the query-remap coefficient. |
| `container_prd.cc`: `create_side_image`, `create_vertical_image`, `put_image` | Each image reads a primary slot and copies its initialized ID/radius. Displacement is an integer combination of the same native rows. Include x seam adjustments, upper-y recomputation, and adjacent-image writes. |
| `container_prd.hh`: `region_index`, `initialize_search`; `v_compute.cc` / `v_compute_2d.cc`: `find_voronoi_cell` | Search x displacement spans one period either side; y/z image blocks stay in the allocated extended grid. Guard the worklist float-to-int conversion before the stock call. |
| `cpp/locate_source.hpp` | Replays safety-relevant arithmetic, verifies complete actual primary population, and emits source operands without changing storage or selecting an owner. |
| `_internal/locate.py` | Checks found/internal ID/shape association, builds the complete source box, invokes WP4, then maps labels and materializes requested arrays. |

Dense internal IDs are injective. Every occupied primary slot is matched once
to an original input row before any images exist. Images copy those IDs without
reinterpretation and grow ID/coordinate arrays together. The stock return reads
ID and position from one selected slot. Validating its returned ID therefore
associates the selected position with that verified original source, even when
the slot is a lazy copy. The enclosure covers **every** source-compatible copy;
it does not choose a candidate image by inspecting the reported coordinate.

### Complete image and integer bounds

`Step(v)` is the source's truncation for nonnegative `v`, and truncation minus
one for negative `v` (including negative integers). The adapter guards finite
values with `abs(v) < INT_MAX/4` before reproducing that conversion. Div/Mod,
block offsets and adjacent-block additions then have explicit range margin.
Existing construction preconditions bound products and allocation dimensions.
The new grid checks leave margins for source sums and copy calculations.
A nonperiodic insertion block outside its legal range is omission, including
`[-1,1]`, one block, and `nextafter(1,0)` where subtraction rounds to 2.

For triclinic image creation write `B(v)=ceil(2*v)+8`. For nonnegative finite
source expressions below `INT_MAX/64`, this bounds both rounding and subsequent
Step/Div endpoint contributions. All expressions have fewer than 16 elementary
operations; their relative rounding factor is below 2, and the additive eight
covers truncation/floor offsets and subnormal absolute error. Sums use absolute
operands, so cancellation cannot shrink the bound. The adapter computes:

```text
Z = B((oz+ez)/nz + 1)
Y = B((oy+ey+B(Z*abs(byz)*ysp))/ny + 1)
X = B((nx+B((Z*abs(bxz)+(Y+1)*abs(bxy))*xsp))/nx + 2)
copy_bounds = (X+2, Y+1, Z)
```

These follow respectively `ima=Div(dk-ez,nz)`, vertical `qjdiv`, and both
original/recomputed `qidiv`; side copies are included by the same larger
bounds. The two extra x periods cover left/right and wrapped source-block
branches. Upper-y recomputation changes `qjdiv` by one and changes x by
`dqidiv`; both old and recomputed x coefficients are bounded by X. This also
bounds their difference by 2X. Source grid values and intermediate conversion
bounds are checked before any lazy copy can execute.

For rectangular storage there are no copied slots. For either route, let `a`
be the replayed native original-query removal and `C_i=copy_bounds_i+abs(a_i)+1`
on periodic axes (zero otherwise). Every complete selected native coefficient
`sigma` obeys `abs(sigma_i)<=C_i`. The extra one covers final search-region
x displacement (all periodic axes for rectangular search). Native worklist
indices use `int(local*reciprocal*8)`; negative values at or below -1 and
out-of-range/nonfinite conversions are refused. Source high-side reflection
clamps large valid nonnegative subindices to zero. ID-only calls use these
safety checks but do not require the owner-image arithmetic profile.

### Exact defect and floating-point allowance

Use exact Fractions of all binary64 operands. Let b be actual stored primary
coordinates, L native rows, R the existing internal-to-Cartesian row rotation,
o its origin (identity/zero in rectangular routes), and `K=k+h` the complete
preparation/insertion removal. Define:

```text
D = o + b @ R - (P_original - K @ A)
M_j = abs(b_j) + sum_i C_i * abs(L_ij)
u = 2^-53; eta = 2^-1074; gamma32 = 32*u/(1-32*u)
E_j = gamma32 * 8*M_j + 64*eta
```

Storage/preparation error D is retained **exactly**, not approximated by a
small relative error of the original coordinate. The scalar source expression
for any copied coordinate plus final owner assembly has fewer than 32 rounded
operations. Its sum of absolute leaves is at most `8*M_j`: the upper-y branch
may contain both old x displacement and an x difference bounded by 2X, followed
by at most two seam periods and final query removal. Side and rectangular
paths use fewer terms. `gamma32*8*M` bounds all product/addition roundings;
64 eta covers propagated gradual-underflow errors. Finite final output is
checked, and source integer/product guards refuse unsafe image construction.
The policy is target-local noncontracting binary64, FLT_EVAL_METHOD=0, nearest
rounding and gradual underflow. It has no WP7 compiler-version restriction.

Cartesian radius component j is

```text
rho_j = sum_i E_i*abs(R_ij)
      + sum_i C_i*abs((L@R-A)_ij)
      + output_frame_error_j
```

The first term transports native assembly error. The second accounts exactly
for the represented native rows differing from the user's rows. For triclinic
output, `output_frame_error_j = gamma32*(abs(o_j) +
sum_i (M_i+E_i)*abs(R_ij)) + 64*eta`; it encloses the three-product NumPy dot
under ordinary binary64 associations/FMA and the origin addition. Rectangular
output has no frame arithmetic. This bound depends on source operands, grid
and query remap, never on a guessed/recovered image or residual.

Consequently the true `s=sigma-K` satisfies, componentwise,

```text
s @ A in [owner_pos - P_original - D - rho,
          owner_pos - P_original - D + rho].
```

WP4 enumerates the complete coefficient region and filters against this exact
closed box. Only a unique compatible shift is returned, followed by an
independent source coefficient-bound check and final int64 materialization.
A larger conservative box can refuse ambiguity; it cannot justify choosing a
nearest residual. This establishes image identity for the native selection,
not an exact Voronoi/power ownership theorem. The mandatory seam fixture keeps
native owner coordinate 0 while its exact image is `-2^-53`.

## Finite refusal budgets

The additional integrity observer charges `128*n + 16*m + 4096` bytes before
allocation, capped at 64 MiB. This includes coordinate/removal arrays, expected
blocks, hash nodes/buckets and the seen vector, beyond existing native output
and construction budgets. Owner-image certification accepts at most 4096
queries per call, at most 1,000,000 candidates for one complete WP4 region,
and at most 16,000,000 candidates over the batch. Remaining complete-region
budget is passed into WP4 before enumeration. Its existing integer/rational
bit and reduction limits remain in force. No prefix or partial batch is
returned. ID-only calls do not acquire the owner-proof budgets. Fixed-dimension
query wrapping uses exact affine formulas over finite binary64 source operands;
private shifts remain Python integers until required public materialization.

## Independent evidence and qualification scope

`test_wp8_metadata.py` established the initial failing contract fixtures before
implementation. `test_wp8_matrix.py` uses its own Fraction Gauss–Jordan oracle,
fixed analytic separated images and exact cubic unimodular transforms; no
production wrapping/enclosure output defines expected images. Failure tests
label packet/enclosure corruption as constructed cases and exercise actual WP4
zero/multiple/complete-region resource outcomes plus invariant cause retention.

`tools/native/check_locate_source.py` compares pre-WP8 accepted native modules
against the changed modules on 56 fixed/seeded route/grid/mask/power cases
(1400 queries), preserving selection and position bits. WP6 ordinary
stock/observed probes cover 48 cases and WP7 planar selected-ghost probes cover
30 cases, including insertion refusal, allocation growth, source identity and
batch resources. Qualification probes are separate builds of the same source
closure; production wheels omit them. The native witness and initialized-stock
WP7 harnesses, WP5/6/7 regressions, sanitizer builds, installed wheel checks and
checksummed source/build/module records accompany the review archive. The
unchanged vendor/source geometry and narrowed changes justify reusing accepted
WP5/6/7 scientific proofs with this noninterference evidence; this is not a new
acceptance of those contracts or a hash-only qualification.


The implementation review found and corrected profile-check ordering: requested
owner certificates check the binary64 environment before the triclinic duplicate
preflight, so multi-site calls retain `LOCATE_NATIVE_UNSUPPORTED` rather than a
plain preflight ValueError. Standard, weight and radius regression cases first
failed, then passed under restored-process rounding-mode tests. The change does
not impose this certificate check on ID-only calls. The ADR query equation now
explicitly names the exact pre-rounded representative. This engineering review
is not the later independent preliminary/Pro acceptance required by #79.
