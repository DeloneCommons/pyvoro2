# F1-A3-R local validation

This is bounded evidence for the shared native-primary remapper, not a release
qualification or a general numerical certification framework. The original
[P0 characterization](../f1_periodic_characterization.py) and its mathematical
oracles are unchanged. The sequencing plan remains Draft revision 0.3.

## Revisions and native provenance

- Verified remote `dev` and implementation base:
  `9f8fbc137aab7dc307c261d9c541f973896505ac`.
- Arithmetic, wrappers, tests and documentation:
  `471230fd23423715aa06a901cbf3daa206fc31b5`.
- Final P0 source revision, after a documentation-only endpoint clarification:
  `96ca70ce68a1775324cc9b821b4288e2a78be8c1`.
- Subsequent evidence commit adds this directory only; the PR records its exact
  final HEAD. No native, runtime, test or policy change is hidden in that commit.
- Python 3.12.14, NumPy 2.3.5, Linux x86-64/glibc 2.41. Both `_core` and
  `_core2d` were freshly built from this checkout at the base with GCC
  (Debian 14.2.0-19) 14.2.0, C++17, the existing Release/`-O3` configuration
  and standard scikit-build-core editable installation. This is not a donor
  binary or historical wheel control. ELF `.comment` independently identifies
  GCC 14.2.0. Neither native sources nor build configuration changes in the PR.

The first `python -m pip install -e ".[dev]"` failed because inherited `CXX`
named absent `clang++`. Selecting installed GCC built successfully, but pip's
default user destination was read-only. The successful setup was:

```bash
python -m venv --system-site-packages /workspace/work/f1-a3-r/venv
CC=gcc CXX=g++ /workspace/work/f1-a3-r/venv/bin/python -m pip install -e ".[dev]"
```

All subsequent Python commands below use that environment. NumPy/SciPy are
available through its system-site-packages setting. Documentation dependencies
were installed with `python -m pip install -r docs/requirements.txt`.

| Native/build input | SHA-256 or Git object |
|---|---|
| `_core.cpython-312-x86_64-linux-gnu.so` | SHA-256 `22e898a36b44e95547738b15721497817cb679042e9606400feed9f57744d976` |
| `_core2d.cpython-312-x86_64-linux-gnu.so` | SHA-256 `65c84701f10388f7c1e1023a0bae8ef60fa61a4ac774b933980e011d24ea947c` |
| `cpp` | Git tree `ecc745e3f3070c2bc9e5862783aca1b82d307184` |
| `vendor` | Git tree `61a56ea2c6f677570feb76baf9c4a4e87367c3c1` |
| `CMakeLists.txt` | Git blob `b999a159a333f543230a1098f40be3f361d63605` |
| Unmodified P0 script | SHA-256 `f0b989c718895a98d275ff70b2edc5a78f3d4da882779975dc7fa025cee4c51e` |

The [baseline](p0-baseline.json) and [candidate](p0-candidate.json) retain the
original full P0 outputs, including lossless inputs, oracles, per-predicate
results, Python source paths and native module paths. The baseline dirty flag
records newly added regression tests only: production Python and native sources
still matched the base. The final candidate replay used a clean checkout.
Both runs used the same binary hashes above. The script's own paths-only native
provenance disclaimer is intentionally retained; the separate fresh build and
hash evidence here supplies the additional provenance.

## Implementation and compatibility

One private `native_remap` module aligns canonical binary64 dyadics over a
power-of-two denominator, computes exact integer divmod and affine translations,
rounds only the final coordinate, and compares snap distances exactly. Both
rectangular dimensions share it. The triangular path commits z, y, x once,
propagating each full row before the next coordinate. Arbitrary-size shifts
are returned privately, then final coefficients are checked before public
int64 materialization, including calls with `return_shifts=False`.

Removed the repeated triangular passes/final reduction, duplicated rectangular
arithmetic, detached-snapshot-as-public-`self` call and dead intermediate
`floor_to_int64`/`checked_int64_add` helpers. The caller inventory had no remaining
production users; the obsolete checked-add-only unit test was replaced by
final-boundary coverage. Public composite Cartesian remapping prepares one frame.

Numerical changes are deliberate: rounded upper chart endpoints advance even at
zero epsilon; rectangular periods remain `RN(hi-lo)`; exact `2*eps < D` is
required on active axes even for empty input; incompatible declared rectangular
intervals and overlapping snap neighborhoods refuse. This can reject a highly
anisotropic frame that still passes A2 admission. Explicit smaller epsilon is
available on existing helpers. Owned outputs, strict validation, nonmutation,
legacy signatures, final shift convention and mandatory native safety remain.

The domain docstrings, spatial/planar guides and references, architecture and
changelog describe these consequences. [Proposed ADR 0032](../../../../docs/development/decisions/0032-native-primary-remapping.md)
records the narrow intermediate-int64 supersession; it is not marked Accepted.
The accepted A1/A2 decision records and their algorithms are unchanged.

## Independent RED/GREEN and local checks

Before production edits, the new shared-remap, frame-lifetime and admission
tests gave **57 failures and 236 passes**. Failures reproduced lost quotient
remainders, the `.1` shear-product remainder, endpoint/subnormal range errors,
incorrect final int64 admission/refusal, absent interval/snap-band refusals and
three frame preparations. The initial candidate passed all 293 tests in that
same run. Additional coverage brings those three modules to **306 passes**.

Fraction-based expectations are independent of the shared implementation.
Explicit witnesses include D=3 at `2**54` and `-2**55`, `.975` after exact
large shear subtraction, final `INT64_MAX` after coupling, lower overflow and
rounded-upper overflow with either shift-return setting, decimal bounds,
chart-gap refusals and unrounded `lo+r`. Tests cover both dimensions, masks,
empty batches, subnormals, translated positive/negative shears, fixed points,
wide private coefficients, conditioning warning counts and refusal before
dispatch. A1's literal floor differs intentionally from the numerical chart.

Commands and observed outcomes:

```bash
# Existing baseline R1/reflection/preparation suites, before implementation:
python -m pytest -q tests/forward/spatial/test_periodic_remap.py \
  tests/forward/spatial/test_periodic_coupled_snap.py \
  tests/forward/spatial/test_periodic_reflection.py \
  tests/forward/planar/test_domains.py \
  tests/forward/common/test_generator_preparation.py
# 406 passed; same suites pass on candidate.

# Initial RED, then GREEN on candidate:
python -m pytest -q tests/forward/common/test_native_remap.py \
  tests/forward/spatial/test_periodic_backend_frame.py \
  tests/forward/spatial/test_periodic_native_admission.py
# Baseline: 57 failed, 236 passed. Initial candidate: 293 passed.
# With additional affine/band/subnormal/private-boundary controls: 306 passed.

# Combined focused run:
python -m pytest -q -rs tests/forward/common/test_native_remap.py \
  tests/forward/spatial/test_periodic_remap.py \
  tests/forward/spatial/test_periodic_coupled_snap.py \
  tests/forward/spatial/test_periodic_backend_frame.py \
  tests/forward/spatial/test_periodic_native_admission.py \
  tests/forward/spatial/test_periodic_reflection.py \
  tests/forward/planar/test_domains.py \
  tests/forward/common/test_generator_preparation.py
# 712 passed.

python -m pytest -q
# 3569 passed, 3 skipped in 51.16 s.
python -m flake8 src tests tools benchmarks examples
python tools/export_notebooks.py --check
zensical build --clean --strict
# All three commands passed; final docs build: No issues found.
git diff --check
# Passed.

python notes/maintainer-review/contracts/f1_periodic_characterization.py --native
# Baseline and candidate both exit 1 for their explicitly reported discrepancies.
```

The default-suite skips are two optional `pyvoro` cross-checks (dependency
absent) and the notebook-tool module (`nbformat` absent), confirmed with `-rs`.
Each pytest invocation reports one environment-provided `anyio` plugin warning
about pytest's deprecated `CallSpec2` alias. No arithmetic overflow or invalid
cast warnings occurred. Actual compiled extensions execute the standard/power
R1 ID-associated volumes, unique locate owners, singleton ghost-volume controls,
reflection and native-safety integration tests. Total volume alone is not their
source-geometry oracle.

An independent reviewer found no Critical/Important issue, reran the 306 tests,
and matched 2,500 triangular rows plus 2,500 rectangular cases against a separate
Fraction evaluator (258 expected rectangular containment refusals). It also
rechecked P0. Its sole minor finding, ambiguous rectangular endpoint wording,
was clarified in the documentation-only commit above. This is local review,
not maintainer acceptance or observed remote CI.

## P0 record reconciliation

Both runs contain **48 records**: baseline **18** discrepancy records, candidate
**2**. Every record retains identical `case`, `route`, original `inputs`,
`oracle` and predicate names. All previously passing individual predicates
remain passing. Exactly the 16 targeted records change observations/checks;
the other 32 entire records are identical. The seven observations corrected by
A2 R1, all frame records and compiled wall/lifetime controls remain unchanged.

The four coupled zero-epsilon, six ordinary seam and six coupled-underflow
records now satisfy their original half-open and tangential predicates.
Historical representative/shift metadata was never substituted for those
oracles. The two #60 source shift/delta failures remain byte-for-byte unchanged
and belong to A3-S. No original predicate conflicts with the approved policy.

| Record (0-based) | Case / parameters | Route | Baseline failing predicates | Candidate failing predicates |
|---|---|---|---|---|
| 0 | #60 | Python separator consumer (OrthorhombicCell) | source_shift, source_delta | source_shift, source_delta |
| 1 | #60 | Python separator consumer (PeriodicCell) | source_shift, source_delta | source_shift, source_delta |
| 2 | #65 | public (Python) | none | none |
| 3 | #65 | detached snapshot (Python) | none | none |
| 4 | #67 A-y-to-x; eps=0.0 | public (Python) | half_open | none |
| 5 | #67 A-y-to-x; eps=default | public (Python) | none | none |
| 6 | #67 A-y-to-x; eps=1e-12 | public (Python) | none | none |
| 7 | #67 A-y-to-x; eps=0.0 | snapshot (Python) | half_open | none |
| 8 | #67 A-y-to-x; eps=default | snapshot (Python) | none | none |
| 9 | #67 A-y-to-x; eps=1e-12 | snapshot (Python) | none | none |
| 10 | #67 B-z-to-y-to-x; eps=0.0 | public (Python) | half_open | none |
| 11 | #67 B-z-to-y-to-x; eps=default | public (Python) | none | none |
| 12 | #67 B-z-to-y-to-x; eps=1e-12 | public (Python) | none | none |
| 13 | #67 B-z-to-y-to-x; eps=0.0 | snapshot (Python) | half_open | none |
| 14 | #67 B-z-to-y-to-x; eps=default | snapshot (Python) | none | none |
| 15 | #67 B-z-to-y-to-x; eps=1e-12 | snapshot (Python) | none | none |
| 16 | #67 seams; D=1.0; eps=0.0 | planar (Python) | half_open | none |
| 17 | #67 seams; D=1.0; eps=0.0 | orthorhombic (Python) | half_open | none |
| 18 | #67 seams; D=1.0; eps=0.0 | triclinic (Python) | none | none |
| 19 | #67 seams; D=1.0; eps=0.0 | snapshot (Python) | none | none |
| 20 | #67 seams; D=2.0; eps=0.0 | planar (Python) | half_open | none |
| 21 | #67 seams; D=2.0; eps=0.0 | orthorhombic (Python) | half_open | none |
| 22 | #67 seams; D=2.0; eps=0.0 | triclinic (Python) | half_open | none |
| 23 | #67 seams; D=2.0; eps=0.0 | snapshot (Python) | half_open | none |
| 24 | #67 coupled underflow; axis=0; eps=0.0 | public (Python) | half_open | none |
| 25 | #67 coupled underflow; axis=0; eps=0.0 | snapshot (Python) | half_open | none |
| 26 | #67 coupled underflow; axis=1; eps=0.0 | public (Python) | half_open | none |
| 27 | #67 coupled underflow; axis=1; eps=0.0 | snapshot (Python) | half_open | none |
| 28 | #67 coupled underflow; axis=2; eps=0.0 | public (Python) | half_open | none |
| 29 | #67 coupled underflow; axis=2; eps=0.0 | snapshot (Python) | half_open | none |
| 30 | frame near_parallel | public (Python/NumPy) | none | none |
| 31 | frame near_parallel | snapshot (Python/NumPy) | none | none |
| 32 | frame near_parallel | QR control (Python/NumPy) | none | none |
| 33 | frame identity | public (Python/NumPy) | none | none |
| 34 | frame identity | snapshot (Python/NumPy) | none | none |
| 35 | frame identity | QR control (Python/NumPy) | none | none |
| 36 | frame right_handed | public (Python/NumPy) | none | none |
| 37 | frame right_handed | snapshot (Python/NumPy) | none | none |
| 38 | frame right_handed | QR control (Python/NumPy) | none | none |
| 39 | frame anisotropic | public (Python/NumPy) | none | none |
| 40 | frame anisotropic | snapshot (Python/NumPy) | none | none |
| 41 | frame anisotropic | QR control (Python/NumPy) | none | none |
| 42 | frame cubic_shear | public (Python/NumPy) | none | none |
| 43 | frame cubic_shear | snapshot (Python/NumPy) | none | none |
| 44 | frame cubic_shear | QR control (Python/NumPy) | none | none |
| 45 | #65 native | compiled 3D standard | none | none |
| 46 | walls and lifetime | compiled 3D standard | none | none |
| 47 | walls and lifetime | compiled 2D standard | none | none |

## Representative preparation cost

[measure.py](measure.py) is the bounded local experiment, with raw
[baseline samples](performance-baseline.json) and
[candidate samples](performance-candidate.json). Run it from an editable checkout:

```bash
python notes/maintainer-review/contracts/f1-a3-r/measure.py
```

The 1,000/4,000-point clouds use seeded uniform fractional coordinates plus
integer translations from -2 through 2. Both cells have origin `(1,-2,.5)`:
rectangular periods `(10,10,10)`; triclinic native parameters
`(10,2.5,10,-1.25,.75,10)`. Seeds are `20261127+n`; rectangular then triclinic
consume the same generator in that order. Public Cartesian remapping includes
frame preparation. Full preparation includes one snapshot, containment and the
mandatory duplicate scan. Compute uses actual native standard geometry with
vertices/adjacency/faces disabled and optional duplicate reporting off; mandatory
safety remains enabled.

After one warmup, report median of five repeats of five remaps, or three
single-call repeats for preparation/compute. Candidate timing source is
`471230fd23423715aa06a901cbf3daa206fc31b5`; later edits only clarify docstrings
and record evidence. Native binaries are identical across both measurements.

| Workload | Points | Remap baseline → candidate (ms) | Full preparation baseline → candidate (ms) | Compute baseline → candidate (ms) |
|---|---|---|---|---|
| rectangular | 1000 | 2.27 → 8.69 | 24.41 → 31.64 | 51.69 → 60.06 |
| triclinic | 1000 | 5.63 → 8.34 | 29.60 → 36.60 | 63.54 → 73.90 |
| rectangular | 4000 | 6.31 → 34.92 | 103.97 → 130.54 | 212.31 → 237.78 |
| triclinic | 4000 | 20.46 → 31.99 | 121.16 → 144.01 | 257.25 → 291.63 |

The extra exact arithmetic costs about 23–27 ms in full 4,000-point preparation
on this host. Isolated rectangular remapping is appreciably slower; complete
compute has a smaller relative change. These are local observations, not
portable timing promises or CI thresholds. No alternate fast/fallback path or
new benchmark framework was introduced.

## Remaining boundaries

A3-S source connectors/realization labels and A3-P exact pre-projection lifting
remain unimplemented. Remote source lifts can still lose physical phase during
the retained Cartesian QR projection. Fixed-point guarantees cover accepted
rectangular/native-internal inputs, not Cartesian QR round trips or A1 views.
No universal exact reconstruction or single absolute error tolerance is claimed.

Uninserted locate recovery, independent distinct-query ghost batching and ghost
ID/weight APIs (F2), B1/C1, C4 and G1 remain separate. The existing singleton
ghost controls do not certify the distinct-query batching defect. No native
vendor patch, frame-threshold change, optional-safety bypass, source rebasing,
persistent cache/context, release scope or extended CI dispatch is included.
Implementation handoff stops after one PR against `dev`; no merge or unobserved
CI success is claimed.
