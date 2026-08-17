# pyvoro2

[![CI](https://github.com/DeloneCommons/pyvoro2/actions/workflows/ci.yml/badge.svg)](https://github.com/DeloneCommons/pyvoro2/actions/workflows/ci.yml) [![Docs](https://github.com/DeloneCommons/pyvoro2/actions/workflows/docs.yml/badge.svg)](https://github.com/DeloneCommons/pyvoro2/actions/workflows/docs.yml) [![PyPI](https://img.shields.io/pypi/v/pyvoro2.svg)](https://pypi.org/project/pyvoro2/) [![Python Versions](https://img.shields.io/pypi/pyversions/pyvoro2.svg)](https://pypi.org/project/pyvoro2/) [![License](https://img.shields.io/pypi/l/pyvoro2.svg)](https://github.com/DeloneCommons/pyvoro2/blob/main/LICENSE)

**Documentation:** https://delonecommons.github.io/pyvoro2/


---

**pyvoro2** is a scientific Python package for computing **2D and 3D Voronoi
and power/Laguerre tessellations**, with particular support for periodic
topology and inverse fitting of power weights from partial geometric data.

> **pyvoro2 0.8.0:** this source describes the finalized v0.8 release contract.
> One exact source commit is independently reviewed and qualified with its
> artifacts under issue #33 before the public tag is created. The archived
> v0.6.3 release remains the software baseline cited by the current
> separator-inverse manuscript.

The v0.8 tree provides:

- standard Voronoi tessellations;
- power/Laguerre tessellations directly from mathematical weights, with the
  existing radius representation retained;
- a dedicated `pyvoro2.planar` namespace for 2D rectangular domains;
- bounded, partially periodic, and triclinic periodic 3D domains;
- explicit periodic neighbor-image shifts;
- diagnostics, validation, topology normalization, and visualization helpers;
- separator-based inverse fitting in 2D and 3D, including graph/connectivity
  diagnostics, hard-constraint witnesses, realized-boundary matching, and an
  optional realization-aware active-set loop.

The package is evolving toward a stable architecture for **forward and inverse
weighted tessellations**. v0.8 is a feature-free maintenance and
compatibility-removal release. v0.9 is reserved for functional/API
stabilization and downstream readiness, 1.0 stabilizes the existing core,
prescribed cell measures begin in v1.1, and mixed separator-plus-measure fitting
begins in v1.2.

pyvoro2 is designed to be explicit and predictable:

- it vendors and wraps upstream Voro++ sources, including a small numeric
  robustness fix already accepted upstream for power/Laguerre pruning;
- 3D and planar APIs remain separate where their backends and domain support
  differ;
- power **weights** are the mathematical quantities, while Voro++ **radii** are
  a backend representation;
- algebraic inverse fit and realized geometry are reported as separate layers;
- numerical and topological failure modes are exposed through diagnostics
  rather than hidden fallback.

**License note:** pyvoro2-authored code is released under **LGPLv3+** starting
with version 0.6.0. Earlier versions were released under MIT. Vendored
third-party code remains under its own licenses.

## Quickstart

### 1) Standard Voronoi in a 3D box

For 3D visualization, install the optional dependency with
`pip install "pyvoro2[viz]"`.

```python
import numpy as np
import pyvoro2 as pv
from pyvoro2.viz3d import view_tessellation

points = np.random.default_rng(0).uniform(-1.5, 1.5, size=(10, 3))
box = pv.Box(((-2, 2), (-2, 2), (-2, 2)))
result = pv.compute(points, domain=box, mode='standard')

view_tessellation(
    result.cells,
    domain=box,
    show_vertices=False,
)
```

<img src="https://raw.githubusercontent.com/DeloneCommons/pyvoro2/main/docs/assets/quickstart_box.png" width="50%" alt="Voronoi tessellation in a box" />

### 2) Planar periodic workflow

```python
import numpy as np
import pyvoro2.planar as pv2

points2d = np.array([
    [0.2, 0.2],
    [0.8, 0.25],
    [0.4, 0.8],
], dtype=float)

cell2d = pv2.RectangularCell(
    ((0.0, 1.0), (0.0, 1.0)),
    periodic=(True, True),
)
result2d = pv2.compute(
    points2d,
    domain=cell2d,
    return_diagnostics=True,
    normalize='topology',
)

diagnostics2d = result2d.require_tessellation_diagnostics()
topology2d = result2d.require_normalized_topology()
```

### 3) Power/Laguerre tessellation

The forward `compute(...)` APIs accept mathematical power weights directly:

```python
weights = np.linspace(-0.2, 0.2, len(points))

result = pv.compute(
    points,
    domain=box,
    mode='power',
    weights=weights,
    include_empty=True,
)
```

The power function is `||x - p_i||^2 - w_i`: weights have squared-length
units, may be negative, and are converted to non-negative backend radii using
one common global shift. Adding the same constant to every weight leaves the
complete diagram unchanged. Existing `radii=` calls remain available, but the
resulting length-unit radii are a non-unique backend representation rather than
necessarily physical radii. Supply exactly one of `weights=` or `radii=` in
power mode. Finite representability is necessary for conversion but does not
guarantee a numerically resolvable native tessellation. Voro++ evaluates radical
geometry with binary64 squared-radius arithmetic, so very large absolute
`radii**2` values or genuine weight ranges relative to squared coordinate/domain
scales can lose geometric resolution. There is no universal safe cutoff: the
onset depends on scale, geometry, platform, and compiler, and periodic power
tessellations are a particularly sensitive regime. See
[Power diagrams](https://delonecommons.github.io/pyvoro2/theory/power-diagrams/) for the precise distinction.

### 4) Periodic crystal cell with neighbor image shifts

```python
cell = pv.PeriodicCell(
    vectors=(
        (10.0, 0.0, 0.0),
        (2.0,  9.0, 0.0),
        (1.0,  0.5, 8.0),
    )
)

result = pv.compute(points, domain=cell, return_face_shifts=True)

# result.require_boundaries() returns faces aligned with input-site order.
# Each face can include:
#   adjacent_cell  (neighbor site id)
#   adjacent_shift (which periodic image produced the face)
```

### 5) Fit weights from separator observations

```python
import pyvoro2.inverse as inverse
import pyvoro2.inverse.separator as separator

points_pair = np.array([
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
])
pair_box = pv.Box(((-5, 5), (-5, 5), (-5, 5)))

observations = inverse.resolve_separator_observations(
    points_pair,
    [(0, 1, 0.25)],
    measurement='fraction',
    domain=pair_box,
)

fit = inverse.fit_weights_from_separators(
    points_pair,
    observations,
    model=separator.FitModel(mismatch=separator.SquaredLoss()),
)
```

The small `pyvoro2.inverse` surface is the normal fixed-observation route.
Advanced models, realized-boundary checks, reports, and the experimental
active-set workflow are available explicitly from
`pyvoro2.inverse.separator`.

## Numerical safety notes

Voro++ uses fixed absolute tolerances internally. pyvoro2 always rejects
generator pairs whose squared distance is at most `1e-10` before native
insertion; very small or very large coordinate systems can still lose
geometric accuracy.

pyvoro2 does not silently rescale coordinates. Rescale explicitly when using
unusual units.

A user policy can add a diagnostic range above the mandatory `1e-5` safety
distance:

```python
result = pv.compute(
    points,
    domain=cell,
    duplicate_check='raise',
    duplicate_threshold=1e-3,
)
```

`duplicate_check='off'`, `warn`, and `raise` affect only safe pairs below the
configured user threshold. Mandatory violations always raise. For periodic
domains mandatory safety always uses certified minimum-image geometry;
`duplicate_wrap=False` changes only the optional diagnostic metric.

Inserted generators must lie in each non-periodic half-open interval
`[lo, hi)`; periodic axes are remapped. This includes temporary ghost
generators. Locate query points themselves are not inserted.

For stricter post-hoc checks, see:

- `pyvoro2.validate_tessellation(..., level='strict')`;
- `pyvoro2.validate_normalized_topology(..., level='strict')`;
- `pyvoro2.planar.validate_tessellation(..., level='strict')`;
- `pyvoro2.planar.validate_normalized_topology(..., level='strict')`.

The vendored Voro++ snapshot includes the upstream robustness fix for radical
pruning in power mode. This avoids rare cross-platform cases where fully
periodic power tessellations could produce a non-reciprocal face/neighbor graph
under aggressive floating-point code generation.

## Why use pyvoro2?

Voro++ is fast and mature, but its low-level C++ interface does not provide all
of the Python-side contracts needed in scientific workflows. pyvoro2 adds:

- triclinic periodic cells and coordinate mapping in 3D;
- partially periodic orthorhombic cells for slabs and wires;
- explicit planar support through `pyvoro2.planar`;
- periodic image-labelled faces and edges for graph construction;
- diagnostics and normalization for reproducible topology;
- owner lookup with `locate(...)`;
- non-inserting probe cells with `ghost_cells(...)`;
- separator-based inverse fitting with inspectable graph, feasibility,
  realization, and active-set diagnostics.

## Documentation overview

| Section | What it contains |
|---|---|
| [Choosing an API](https://delonecommons.github.io/pyvoro2/guide/choosing-api/) | Preferred forward and inverse entry points, lifecycle status, result layers, and the static scalability contract. |
| [Capabilities and limitations](https://delonecommons.github.io/pyvoro2/guide/capabilities/) | Current supported contracts, wrapper restrictions, backend limits, unsupported geometry, and open architecture decisions. |
| [Concepts](https://delonecommons.github.io/pyvoro2/guide/concepts/) | A concise user introduction to Voronoi and power/Laguerre tessellations. |
| [Glossary](https://delonecommons.github.io/pyvoro2/guide/glossary/) | Power weights, backend radii, gauge, separator observations, realization, and active-set terminology. |
| [Domains (3D)](https://delonecommons.github.io/pyvoro2/guide/domains/) | `Box`, `OrthorhombicCell`, and `PeriodicCell`. |
| [Planar (2D)](https://delonecommons.github.io/pyvoro2/guide/planar/) | The planar namespace, rectangular periodicity, diagnostics, normalization, and plotting. |
| [Operations](https://delonecommons.github.io/pyvoro2/guide/operations/) | Forward tessellation, owner lookup, and ghost-cell workflows. |
| [Topology and graphs](https://delonecommons.github.io/pyvoro2/guide/topology/) | Periodic image-labelled adjacency and normalized topology. |
| [Separator fitting](https://delonecommons.github.io/pyvoro2/guide/powerfit/) | Current inverse API, result diagnostics, realization matching, and active-set refinement. |
| [v0.6.3–v0.8 migration](https://delonecommons.github.io/pyvoro2/guide/migration-v0.7/) | Canonical replacements and the completed v0.8 compatibility removal. |
| [Theory](https://delonecommons.github.io/pyvoro2/theory/) | API-independent definitions of power diagrams, weights, gauge, and separator inversion. |
| [Development](https://delonecommons.github.io/pyvoro2/development/) | Architecture, workflow, documentation conventions, release plans, API lifecycle, and decision records. |
| [Visualization](https://delonecommons.github.io/pyvoro2/guide/visualization/) | Optional `py3Dmol` and `matplotlib` helpers. |
| [Examples](https://delonecommons.github.io/pyvoro2/guide/notebooks/) | Executable notebook workflows. |
| [API reference](https://delonecommons.github.io/pyvoro2/reference/) | Exact signatures and docstring reference for spatial, planar, and separator-fitting APIs. |
| [v0.8.0 release notes](https://delonecommons.github.io/pyvoro2/project/release-notes-v0.8/) | Removals, fixes, maintenance, documentation, and the release distribution contract. |
| [Roadmap](https://delonecommons.github.io/pyvoro2/project/roadmap/) | v0.8 cleanup, v0.9 functional stabilization, the stable 1.0 core, v1.1 prescribed measures, v1.2 mixed fitting, and future research. |

## Installation

Most users should install a prebuilt wheel:

```bash
pip install pyvoro2
```

The v0.8 qualification workflow targets 20 standard GIL-enabled CPython
wheels: Python 3.10, 3.11, 3.12, 3.13, and 3.14 for each of manylinux x86_64,
Windows AMD64, macOS arm64, and macOS x86_64. It also builds one source
distribution. Other operating systems, architectures, interpreter
implementations, free-threaded CPython builds, musllinux, 32-bit targets,
Windows arm64, and macOS universal2 do not have v0.8 prebuilt wheels.

Optional extras:

- `pyvoro2[sparse]` for optional SciPy sparse-direct static quadratic
  separator fitting;
- `pyvoro2[viz]` for 3D `py3Dmol` and 2D plotting;
- `pyvoro2[viz2d]` for 2D matplotlib plotting only;
- `pyvoro2[all]` for the full local notebook, docs, lint, test, and release
  validation stack.

Supported source builds use standard GIL-enabled CPython 3.10–3.14 and require a
C++17 compiler, CMake 3.20 or newer, and Python development headers.
Free-threaded CPython builds are not currently supported. Ninja is recommended
because the build backend uses CMake efficiently with it. Typical toolchains
are GCC or Clang on Linux, Xcode Command Line Tools on macOS, and Visual Studio
Build Tools with the "Desktop development with C++" workload on Windows. These
are source-build requirements, not pyvoro2 runtime dependencies.

For an editable runtime-only build:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For local repository development and all validation tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

See [Contributing](https://github.com/DeloneCommons/pyvoro2/blob/main/CONTRIBUTING.md)
for platform notes and clean-environment verification.

## Testing

The default suite includes deterministic tests and seeded fuzz/property tests
with `--fuzz-n=10`:

```bash
pip install -e ".[test]"
pytest -q
```

Increased seeded fuzz coverage and the optional independent-wrapper group:

```bash
# Explicitly select fuzz tests and raise their iteration count
pytest -m fuzz --fuzz-n 100

# Independent wrapper cross-checks; requires pyvoro
pip install pyvoro
pytest -m pyvoro --fuzz-n 100
```

For a complete local publishability pass:

```bash
python tools/release_check.py
```

## Project status and support

pyvoro2 is currently **beta**. This source describes v0.8.0, including the
completed compatibility removal, Python 3.14 and wheel-matrix preparation, and
the accepted bounded API-contract corrections. The v0.8 source is finalized
before one exact commit and its artifacts are qualified under issue #33; any
tracked correction changes that candidate. The archived v0.6.3 release remains
the software baseline cited by the separator-inverse manuscript. The v0.8.0
release uses the project Git tag, GitHub Release, and package-distribution
process and intentionally creates no new pyvoro2 Zenodo software-version
record. Existing historical pyvoro2 and project/reproducibility Zenodo records
remain valid. New inverse families begin after the stable 1.0 core. See the
[v0.8.0 release notes](https://delonecommons.github.io/pyvoro2/project/release-notes-v0.8/) and the
[completed v0.8 plans](https://delonecommons.github.io/pyvoro2/development/plans/archive/).

Reproducible bugs and focused feature proposals are welcome through GitHub
issues. Development is currently led by one maintainer, so support is
best-effort. Contribution and decision policies are described in
[`CONTRIBUTING.md`](https://github.com/DeloneCommons/pyvoro2/blob/main/CONTRIBUTING.md).

## AI-assisted development

The project has used the latest Chat and Codex models available at the time of
development for planning, implementation support, testing, and documentation.
The maintainer reviews and validates all integrated changes and remains
responsible for the software and scientific claims.

See [AI-assisted development](https://delonecommons.github.io/pyvoro2/project/ai/) for details.

## License

- pyvoro2-authored code is **LGPLv3+** starting with version 0.6.0;
- versions before 0.6.0 were released under MIT;
- vendored Voro++ code remains under its upstream license.

---

*This README is auto-generated from the MkDocs sources in `docs/`.*
To update it, edit the docs pages and re-run: `python tools/gen_readme.py`.

