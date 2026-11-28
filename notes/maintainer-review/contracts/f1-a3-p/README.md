# F1-A3-P implementation evidence

Bounded evidence for [#166](https://github.com/DeloneCommons/pyvoro2/issues/166),
not universal periodic geometry qualification or F1 acceptance.

## Revisions and scope

- Verified live `dev` and implementation base:
  `07a619b0a723ff860ea577a07489d0c21fefb01b`.
- Implementation and preliminary independent-review target:
  `42cc4776f6e5a4231079264ada0e6d5249b6548c`.
- Branch: `codex/f1-a3-p-stable-source-lifting`; PR targets `dev`.
- Subsequent changes correct two test-frame portability assumptions and add
  this evidence. Production arithmetic is unchanged. The PR records final HEAD.

GitHub's email-privacy protection rejected the original unpushed commits. They
were rewritten with the verified account's GitHub no-reply address, without
source changes. Published implementation commit
`05ca29652258891ebc72cb99319e7b5b6d491d50` has exactly the same tree as measured/
reviewed `42cc4776f6e5a4231079264ada0e6d5249b6548c`. Raw artifacts retain the actual
original measurement revisions; this metadata correction does not reinterpret
the measurements. Final source/tests also match the successful reviewed suite.

The coordinator removes exact source lifts, projects exact local residuals once
through the operation Q, and calls the unchanged R reducer. A1's bounded
immutable basis/cofactor data is shared. Both intermediate coefficient vectors
stay wide; only combined K is checked at the existing int64 destination.
Public/snapshot Cartesian remap, persistent compute/locate/ghost generators and
inserted temporary ghosts share it. Original source arrays, fields, IDs, radii,
one-frame lifetime, both safety stages and S's same-operation reference remain.
Standalone converters, all A1 helpers, rectangular routes and uninserted locate
queries keep their existing semantics. No native/vendor/dependency changes.

## Baseline failures and independent scientific controls

Before implementation, the original R1/frame/reflection/central-preparation
selection passed **520 tests**, and unmodified P0 passed all **48 records**.
The initial new projection/R1 selection produced **40 failures, 103 passes**
after correcting a test's requested result form. The complete new selection,
replayed on an isolated exact-base checkout with its own fresh native build,
produced **52 failures, 119 passes**. That run uses the corrected exactly
representable signed-translation fixture; a positive `2**50 + 5/8` is not an
exact binary64 translation, so the signed-basis batch test uses `2**48` instead.
The specified negative-direction R1 `2**50` witness is retained unchanged.
The candidate passes that complete selection; the later alternate-QR fixture
brings it to **172 passes with warnings treated as errors**.

[observe.py](observe.py) records real compiled calls through the existing
capture worker. [native-comparison.json](native-comparison.json) preserves
lossless hexadecimal submitted coordinates, actual represented L, genuine
readback anchors, final K, per-ID volumes, bounded owners and singleton ghosts
for untranslated, modest and `2**50` lifts. Coordinate records are deduplicated
only after asserting standard/weight/radius calls used identical arrays/K.

For the exact R1 middle-site translation, the fresh integrated baseline gives:

| Mode, IDs (11,13,17) | Baseline volumes | Independent exact target, attained by candidate |
|---|---|---|
| Standard | (0.2632881829, 0.3585600198, 0.3781517973) | (3/8, 1/4, 3/8) |
| Weights (3/32,0,0) | (0.5155160687, 0.2257764616, 0.2587074697) | (21/32, 1/16, 9/32) |
| Radii (1/4,0,0) | (0.4247014986, 0.2724405409, 0.3028579605) | (9/16, 1/8, 5/16) |

Targets follow independent rational one-dimensional power bisectors times unit
area, not total-volume closure. Candidate errors are below `2e-12`. Current
baseline numbers differ from the earlier donor-binary research but reproduce
the same pre-projection phase-loss mechanism. At the bounded original source
positions, owners change from `(11,11,17)` to `(11,13,17)`; owner positions keep
the existing reporting reference. With untranslated persistent sites and only
the singleton ghost translated, baseline volume is `0.3415193079271581` instead
of `1/8`; candidate volume is approximately `0.12499999999999985`. Local and
translated temporary ghosts retain original query metadata. No distinct-query
batch is used as this volume oracle.

For the candidate middle site, native submission is identically
`(0x1.6a09e667f3bccp-1, 0x1.ffffffffffffbp-2, 0x0.0p+0)` in both local and remote
calls. Its local K is `(-1,0,0)`; remote K is `(1125899906842623,0,0)`.
Native readback equals submission for this fixture. The baseline remote native
submission is instead `(0x1.a827999fcef34p-2, 0x1.2bec333018869p-3,
0x1.a827999fcef30p-4)`. The final coefficients alone could not reveal this loss.

Additional committed tests establish:

- Independent Fraction elimination, residual, complete-dot rounding and R
  chart decisions using the actual operation Q; signed/permuted/sheared bases,
  nonzero origins, positive/negative lifts, seams, epsilon variants, empty
  shapes, strict categories, nonmutation and controlled finite-view refusal.
  Exact translation v/K covariance holds alone, reordered and in mixed batches.
- The unit-basis origin `2**-55` witness: source k=`INT64_MIN-1` combines with
  n=1 and succeeds; `INT64_MAX+1` refuses at final materialization, also without
  returning shifts. The shared preparation boundary has the same contract.
- Large origin `2**54`, unit rows: exact distinct native residues survive but
  primary reports collapse; actual prepared-view safety refuses before dispatch.
  With `diag(5,1,1)` and x=`1 +/- 2**-22`, reports differ by 4 while native
  separation is `2**-21`; the actual compiled backstop refuses. No safety bypass.
- Oblique R1 source observations and actual matching capture preserve v/K/L and
  anchors; forward/reverse effective shifts change by `t_i-t_j`, raw labels
  stay unchanged, requested/other-image classification and unit measures agree.
  Translated active-set states retain the requested contacts and slab volumes.
  Existing independent membership/refusal, source-identity and active-state
  tests are preserved by the full suite.
- Safe negative query seam `-2**-55` retains the 3/4-side owner. Positive query
  `2**-41` retains its different owner from query zero with the displaced site.
  Actual native query arrays are unchanged. ID-only and nonperiodic outside/
  missing-owner reporting, plus existing periodic NaN sentinel tests, remain.

## P0 reconciliation and numerical compatibility

The original script is unmodified, SHA-256
`f0b989c718895a98d275ff70b2edc5a78f3d4da882779975dc7fa025cee4c51e`.
Both recorded replays used clean production checkouts. All **48 complete case
objects are identical**, including inputs, oracles, observed values and checks.
[p0-comparison.json](p0-comparison.json) classifies every record individually.
This preserves the seven A2 corrections and S's two #60 source records.

One existing default-suite assertion required an explained change. Public
`duplicate_check(wrap=True)` already measures the prepared Cartesian view.
For its decimal sheared fixture, exact k_x=-2 produces `RN(p_x+2)-2`, changing
x from `0.01652764` to `0.016527640000000066`. An independent rational nearest
image gives squared distance `0.23935148791850694` and distance
`0.4892356159546307`, one ULP below the old literal. The test now evaluates this
prescribed view independently and retains a tight source-distance check.
Neither the duplicate algorithm nor its safety threshold changed. This is
permitted local chart rounding, not a P0 expectation rewrite or exact
source/native equivalence claim.

The guide, reference, preparation architecture and changelog document changed
literal-composition semantics, exactly represented translation conditions,
nominal K and unchanged safety/query boundaries. [ADR 0034](../../../../docs/development/decisions/0034-stable-source-lifting.md)
remains **Proposed** for acceptance in this implementation PR after review.

## Native provenance and validation

Both `_core` and `_core2d` were freshly built from integrated base sources with
GCC (Debian 14.2.0-19) 14.2.0, C++17 and existing Release build settings, using
the standard editable backend. The candidate reuses those unchanged binaries;
the separate baseline replay rebuilt them and obtained the same hashes.
Python 3.12.14, NumPy 2.3.5, Linux x86-64/glibc 2.41. This is not a historical
donor/wheel control. Raw evidence includes actual module paths and Git tree IDs.

| Artifact | SHA-256 |
|---|---|
| `_core` | `22e898a36b44e95547738b15721497817cb679042e9606400feed9f57744d976` |
| `_core2d` | `65c84701f10388f7c1e1023a0bae8ef60fa61a4ac774b933980e011d24ea947c` |

Workspace setup used an isolated venv and explicitly selected installed GCC:

```bash
python -m venv --system-site-packages /workspace/work/f1-a3-p/venv
CC=gcc CXX=g++ /workspace/work/f1-a3-p/venv/bin/python -m pip install -e '.[dev]'
# Subsequent commands use that venv:
python -m pip install -r docs/requirements.txt nbformat nbclient ipykernel matplotlib py3Dmol
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids unrelated preinstalled plugins (an
initial run emitted an external anyio/pytest deprecation). Final suite uses
`IPYTHONDIR=/workspace/work/f1-a3-p/ipython` to avoid an unwritable home-directory
warning; no repository or runtime policy was changed for either adjustment.

Observed validation commands:

```bash
python -m pytest -q -W error \
  tests/forward/spatial/test_periodic_source_lifting.py \
  tests/forward/spatial/test_periodic_coupled_snap.py \
  tests/forward/spatial/test_source_lifting_safety.py \
  tests/inverse/separator/test_source_lifting_integration.py
# 172 passed.
python -m pytest -q -rs
# 3782 passed, 2 skipped (optional pyvoro cross-checks).
python -m flake8 src tests tools benchmarks examples
python tools/export_notebooks.py --check
python tools/check_notebooks.py
# All 8 notebooks validated in fresh kernels, without editing sources.
zensical build --clean --strict
# No issues found.
git diff --check
python notes/maintainer-review/contracts/f1_periodic_characterization.py --native
# 48/48 original records pass, unchanged.
```

The preliminary independent code review found no Critical or Important
production defects. It independently ran 171 focused cases and 1,035 Fraction
checks across 115 signed bases. It also exercised actual temporary-ghost safety
and reviewed the original P0 and S capture. Two test portability findings were
corrected: use the actual public operation frame for reporting, and avoid a
LAPACK-specific residual-rounding literal. An independently rounded qualified
QR pair reproduced the fragile assertion before correction; both frames now
exercise the exact-dot distinction. This preliminary review does not substitute
for the requested later independent F1-A integration acceptance audit.

## Integrated costs

[measure.py](measure.py) uses deterministic dyadic 1,000/4,000-point ordinary
and modest translated cohorts, rectangular and genuinely oblique domains. It
proves every source translation exactly with Fraction, warms each operation
once, then records three single-call samples. Preparation includes frame,
validation and mandatory duplicate work; complete standard compute additionally
executes the actual native container/cells with optional geometry outputs off.
No safety option is bypassed. [performance.json](performance.json) retains raw
samples, seeds, revisions, native hashes and input tree IDs.

Median milliseconds (base → candidate):

| Points | Domain/cohort | Integrated preparation | Complete compute |
|---|---|---:|---:|
| 1000 | Rectangular ordinary | 30.05 → 29.95 | 57.19 → 59.96 |
| 1000 | Rectangular translated | 29.70 → 29.84 | 56.49 → 60.36 |
| 1000 | Oblique ordinary | 35.68 → 40.99 | 73.00 → 81.48 |
| 1000 | Oblique translated | 36.49 → 40.93 | 73.19 → 81.77 |
| 4000 | Rectangular ordinary | 118.29 → 118.99 | 226.31 → 247.35 |
| 4000 | Rectangular translated | 118.77 → 119.79 | 229.46 → 243.47 |
| 4000 | Oblique ordinary | 141.87 → 176.41 | 280.26 → 322.87 |
| 4000 | Oblique translated | 145.33 → 165.94 | 279.45 → 314.57 |

The final candidate replay used committed implementation without concurrent
notebook/test runs. An earlier replay overlapped notebook validation and was
replaced for that reason. These sequential local samples have visible machine
variation even in the unchanged rectangular compute control; they do not isolate
an exact causal percentage. Oblique preparation adds about 4–6 ms at 1,000 and
21–35 ms at 4,000 in these measurements. A bounded cProfile follow-up attributed
the added work to exact local residual/dot arithmetic: 4,001 residual iterator
calls took 55 ms cumulative under instrumentation, versus 225 ms in the existing
duplicate scan and 410 ms overall. Profiled times include instrumentation and
are not benchmark samples. No superlinear new search or material unexplained
cost appeared; no speculative optimization or timing-sensitive CI was added.

## Remaining boundaries

Local projection, snapping and reporting can still round; caller-lost phase
cannot be recovered. Numerical native and exact source geometry are not
universally equivalent. Admission, reporting collapse, native distance floors,
image-search budgets and S correspondence can still refuse. Uninserted remote
locate stepping/index safety and query recovery, independent ghost batching,
F2 metadata, B/C/G1 work and release qualification remain outside this change.
No remote CI, sanitizer, distribution or cross-platform result is claimed, and
the implementation PR is not merged.
