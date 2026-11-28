# F1-A3-S implementation and local evidence

This implements original-source separator connectors and consistent image labels
for the existing observed numerical contact family. It is not independent source
tessellation reconstruction or general native insertion certification.

## Revisions and prerequisites

- Verified integrated `dev` base:
  `c7b66f59ca6d887a80df6406c5876945e8d4b8a2`.
- PR [#163](https://github.com/DeloneCommons/pyvoro2/pull/163), reviewed at
  `49d294163be6f40b5e77079ce246ecf3687f9953`, was merged as
  `c23ba170d4b51ba80355c5bfc3ecd681b9ef73b9`. This merge is an ancestor of the
  base. Integrated ADR 0032 is **Accepted**, following PR #164.
- Branch: `fix/f1-a3-s-source-connectors`.
- Main implementation and independent-review target:
  `d125b12cbb0d3a043525908454381fb7d1018b49`.
- Final code, tests and documentation, including review fixes:
  `2e93a6695795dc12cb323cbeb5319400941acf84`.
- The subsequent evidence-only commit adds this directory. The PR description
  records its exact final HEAD. No runtime or policy change is hidden there.

## Arithmetic, architecture and compatibility

The resolver passes original canonical binary64 endpoints to the existing exact
nearest-image kernel. Explicit periodic rows use a narrow fixed-shift evaluator
in `periodic_images.py`: align endpoint and basis dyadics, evaluate
`p_j - p_i + s @ A` with Python integers, and round each complete component once.
This reuses existing affine arithmetic without search, inversion, native
admission or a representable translated endpoint. Explicit nonnearest images
remain valid. Inferred shifts retain existing search, tie, budget and int64
contracts. Nonperiodic zero-shift subtraction is unchanged.

ADR 0014's distance convention remains: squared distance derives from the
canonical rounded displacement, rather than the exact image-search comparison
key. `_identity.py` already uses this derivation; there is no second source
verification implementation. Source/row association, inferred provenance,
unbound construction, subsetting, copying, replacement and same-version pickle
remain supported. Old remapped-reference geometry can now fail exact binding:
re-resolve original observations against original points and interpret explicit
coefficients in the source basis.

Public spatial and planar compute wrappers share one private worker with
`_compute_for_matching`. Each matching operation has one preparation, one native
frame where applicable, and one native invocation. Periodic matching captures
only submitted coordinates `v`, nominal preparation coefficients `K`, actual
represented native lattice `L`, native stored anchors `a`, anchor-presence flags
and the periodic mask. All arrays are detached, owned and read-only. Capture
follows internal-ID validation and precedes synthetic empties, external-ID
replacement and Cartesian reporting. Public compute and nonperiodic matching do
no capture work. Public signatures, schemas, native checks and warning locations
are preserved.

The effective numerical family is the algebraic `E_i = v_i + K_i @ L`. For an
observed pair the bridge requires exact allowed-integer membership
`(v_j-v_i)-(a_j-a_i) = h @ L`, using the actual rectangular periods or represented
lower-triangular native parameters. Exact reverse substitution includes inactive
axes. It converts raw label `c` to `c - K_j + K_i - h` with Python integers,
caches correspondence per pair, and converts boundary-measure keys before all
unions, matching, classification and unaccounted-pair reporting. Raw returned
cell dictionaries retain their existing labels and can differ from diagnostics.

Missing required periodic labels or genuine anchors, nonintegral correspondence
and incompatible inactive-axis coefficients raise a pair-specific `ValueError`.
They never become fabricated same/other-image classifications. Non-observed
pairs and genuinely empty endpoints do not require invented evidence. Existing
active iteration and final-state boundaries propagate these refusals naturally;
`active.py`, solver policy and ADR 0015's other unavailable-fit handling are
unchanged. Wall/unknown-boundary attribution remains the existing limitation.

## Independent regressions and compiled controls

The seven assignment-listed baseline suites passed **470 tests** before
production edits. Initial source and compiled translated-consumer regressions
then produced **58 failures, 7 passes** on that baseline. They exposed source
snapping, explicit affine cancellation, the source/native admission coupling,
and untranslated raw labels in matching and active-state decisions. New helper
and capture tests initially also failed because their private entry points did
not yet exist; those API-absence failures are not numerical counterexamples.

The dedicated tests use independent Fraction or analytic expectations:

- The `2^-41` unit-period witness, positive gap `2^-40`, original nonzero origins,
  modest translations, both pair directions, mixed explicit/inferred rows,
  partial masks and equivalent signed/permuted unique-minimum bases.
- Explicit results `0.49999999999999994`, `1` after `-2^53 + (2^53+1)`, and
  `-1/4` after cancellation at `2^53`; oblique affine rows, nonnearest images,
  unrepresentable intermediate endpoints, and finite-result refusal.
- Source inference on `diag(2^-60,1,1)` succeeds while unchanged native admission
  refuses. The existing triclinic replacement fixture was independently
  enumerated and bounded: source shift is `(-1,-1,-1)`, with displacement hex
  `(-0x1.428f5c28f5c26p-2, -0x1.851eb851eb853p-2, -0x1.3333333333334p-1)`.
- Genuine compiled planar, orthorhombic and triclinic matching for x coordinates
  `(1/8,3+3/8)`, explicit/inferred and reversed requests, multiple images,
  analytic unit boundary measures and unaccounted pairs. Raw labels `0/-1`
  remain unchanged while forward diagnostics report `-3/-4`; absent `-5` is
  correctly an other-image request. Untranslated controls also pass.
- Independent zero/integral/common-translation/nonintegral anchor controls,
  inactive-axis mismatch, missing anchors, synthetic empties, wide integer
  transport and reversal. Compiled capture verifies ID/order independence and
  one preparation/frame/native invocation.
- The `105/1024`, `nextafter(period,0)` endpoint and `float(1/3)` shear use actual
  same-operation native readback with an independent Fraction membership oracle.
  Tests branch on the observed correspondence, not a compiler-pinned transition.
  Deterministic synthetic controls separately establish refusal.
- Injected observed-reference failures cannot publish realization diagnostics
  or update active decisions. Both outer iteration and accepted final-state
  construction refuse; no prior result is reused.
- Exact source-binding negative controls, replacement/copy/pickle/subset paths,
  public optional outputs, and genuine empty/absent boundary behavior.

An independent whole-branch review found no Critical or Important issues at
`d125b12`. Its 106 dedicated cases and 1,000 additional seeded Fraction affine
and triangular membership/reversal examples passed. Two Minor findings were
fixed: public duplicate warning attribution and unnecessary nonperiodic capture.
Their six added regressions failed before the fixes and passed afterward.

## P0 record comparison

The original [P0 script](../f1_periodic_characterization.py) is unmodified.
The local base replay has the same 48 complete case records as the existing
[tracked A3-R candidate](../f1-a3-r/p0-candidate.json). The baseline dirty flag
reflects added regression tests only; production files still matched the base.
Final replay used clean source `2e93a66` and the same freshly built native modules.

[p0-comparison.json](p0-comparison.json) records all 48 cases individually and
retains the changed observed values and oracles. Inputs, oracles, check keys and
every previously passing predicate are preserved. **Only records 0 and 1 change**;
the other 46 records are entirely identical. All candidate predicates pass.

| Records | Route | Baseline | Candidate |
|---|---|---|---|
| 0, 1 | #60 orthorhombic / `PeriodicCell` separator | shift `(-1,0,0)`, delta x `-1/2`; source predicates fail | shift `(0,0,0)`, delta x `1/2-2^-41`; both pass |
| 2–47 | All other original P0 routes | All predicates pass | Entire records unchanged |

Reproduce from this checkout (JSON output outside the checkout avoids a dirty
provenance flag):

```bash
python notes/maintainer-review/contracts/f1_periodic_characterization.py \
  --native > /tmp/a3-s-candidate.json
python - <<'PY'
import json
from pathlib import Path
baseline = json.loads(Path(
    'notes/maintainer-review/contracts/f1-a3-r/p0-candidate.json').read_text())
candidate = json.loads(Path('/tmp/a3-s-candidate.json').read_text())
old, new = baseline['cases'], candidate['cases']
assert len(old) == len(new) == 48
changed = []
for index, (before, after) in enumerate(zip(old, new)):
    assert before.keys() == after.keys()
    for field in before.keys() - {'observed', 'checks'}:
        assert before[field] == after[field], (index, field)
    assert before['checks'].keys() == after['checks'].keys()
    assert all(not passed or after['checks'][key]
               for key, passed in before['checks'].items())
    assert all(after['checks'].values()), index
    if before != after:
        changed.append(index)
assert changed == [0, 1], changed
print('48 records; only #60 records 0/1 changed; all predicates pass')
PY
```

## Validation and native provenance

Standard editable setup in a new venv:

```bash
python -m venv /workspace/work/a3-s-venv
CC=gcc CXX=g++ /workspace/work/a3-s-venv/bin/python -m pip install -e '.[dev]'
python -m pip install -r docs/requirements.txt
```

The latter command and all checks use that venv. The first editable attempt
failed because inherited `CXX=clang++` was unavailable; selecting installed GCC
succeeded without a build-configuration change. Python 3.12.14, NumPy 2.5.5,
SciPy 1.18.1, pytest 9.2.0, Linux x86-64/glibc 2.41; GCC (Debian 14.2.0-19)
14.2.0, C++17 and the existing Release configuration. Both extensions were
freshly built from this checkout at the base; they are not donor binaries.
Their hashes happen to equal the previous A3-R evidence. Native/build inputs
remain unchanged through final code.

| Input/artifact | SHA-256 or Git object |
|---|---|
| `_core.cpython-312-x86_64-linux-gnu.so` | SHA-256 `22e898a36b44e95547738b15721497817cb679042e9606400feed9f57744d976` |
| `_core2d.cpython-312-x86_64-linux-gnu.so` | SHA-256 `65c84701f10388f7c1e1023a0bae8ef60fa61a4ac774b933980e011d24ea947c` |
| `cpp` | Git tree `ecc745e3f3070c2bc9e5862783aca1b82d307184` |
| `vendor` | Git tree `61a56ea2c6f677570feb76baf9c4a4e87367c3c1` |
| `CMakeLists.txt` | Git blob `b999a159a333f543230a1098f40be3f361d63605` |
| `pyproject.toml` | Git blob `ce1745ca8dd4d284eb3f9bced44e2998194cb00c` |
| Original P0 script | SHA-256 `f0b989c718895a98d275ff70b2edc5a78f3d4da882779975dc7fa025cee4c51e` |

Native modules were loaded from
`/workspace/work/a3-s-venv/lib/python3.12/site-packages/pyvoro2/`; Python sources
were loaded from `/workspace/work/pyvoro2-a3-s/src/pyvoro2/`.

Final focused command and observed result:

```bash
python -m pytest -q \
  tests/forward/common/test_periodic_images.py \
  tests/forward/common/test_native_remap.py \
  tests/forward/common/test_generator_preparation.py \
  tests/forward/common/test_realization_reference.py \
  tests/forward/common/test_realization_capture.py \
  tests/inverse/separator/test_constraints.py \
  tests/inverse/separator/test_identity.py \
  tests/inverse/separator/test_realization.py \
  tests/inverse/separator/test_active_state.py \
  tests/inverse/separator/test_source_connectors.py \
  tests/inverse/separator/test_realization_source_reference.py \
  tests/forward/spatial/test_periodic_backend_frame.py \
  tests/forward/spatial/test_periodic_native_admission.py \
  tests/forward/spatial/test_periodic_reflection.py \
  tests/forward/spatial/test_periodic_coupled_snap.py
# 940 passed in 10.25s

python -m pytest -q -rs
# 3681 passed, 3 skipped in 47.65s
python -m flake8 src tests tools benchmarks examples
python tools/export_notebooks.py --check
zensical build --clean --strict
git diff --check
# All successful; strict docs: "No issues found".
```

The three default-suite skips are `nbformat` missing for notebook-tool tests,
and two optional `pyvoro` cross-checks. Public compute parameter/default and
return-annotation ASTs were independently compared with the base and are
identical. The native R1/frame/reflection/admission controls remain green.
No remote CI, sanitizer, distribution or cross-platform result is claimed.

## Representative cost check

[timing.py](timing.py) is a small manual reproducer, with raw samples in
[timing.json](timing.json). At final code it times 256 sites, 31 observations,
one warmup and seven alternating-order samples for public compute, the private
capture entry with the same native output flags, and complete matching. Domains
are the unit rectangular cell and a moderate dyadic triangular cell. No timing
threshold or CI framework is introduced.

| Domain | Public compute | Private capture | Full matching |
|---|---:|---:|---:|
| rectangular | 245.11 ms | 246.42 ms | 287.42 ms |
| triclinic | 265.11 ms | 259.86 ms | 308.62 ms |

These noisy local medians are a representative check, not a performance bound.
Capture is optional and correspondence is cached per observed pair. Tests
enforce no capture work in ordinary compute or nonperiodic matching; the latter
also avoids retaining 49/73 bytes per site of unnecessary 2D/3D evidence arrays.

## Documentation and remaining boundaries

Separator API docstrings, guide, reference, theory, architecture and changelog
now describe original operands, rounded-distance semantics, the effective
numerical family, label conversion, refusal and migration. Durable semantics
are recorded in [ADR 0033](../../../../docs/development/decisions/0033-source-connectors-and-realization-reference.md),
which remains **Proposed**, pending maintainer review. Its narrow ADR 0013
amendment is worker sharing; ADRs 0014/0015, A1/A2/A3-R admission and safety,
native/vendor code, dependencies and build/workflow settings are preserved.

A3-P pre-QR phase loss/preparation stabilization, F2 uninserted locate and ghost
handling, B1 reduction, C1 tie-policy changes, C2/C4 general insertion/translation
certification, and G1 contact attribution/topology remain separate. Exact
coefficient transport does not certify source/native Cartesian or weight
equivalence, source facets, complete native history or contact completeness.
P0 success is bounded evidence for its original cases. This PR does not merge or
declare integrated R/S acceptance.
