# pyvoro2

[![CI](https://github.com/DeloneCommons/pyvoro2/actions/workflows/ci.yml/badge.svg)](https://github.com/DeloneCommons/pyvoro2/actions/workflows/ci.yml)
[![Docs](https://github.com/DeloneCommons/pyvoro2/actions/workflows/docs.yml/badge.svg)](https://github.com/DeloneCommons/pyvoro2/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/pyvoro2.svg)](https://pypi.org/project/pyvoro2/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyvoro2.svg)](https://pypi.org/project/pyvoro2/)
[![License](https://img.shields.io/pypi/l/pyvoro2.svg)](https://github.com/DeloneCommons/pyvoro2/blob/dev/LICENSE)

**pyvoro2** computes 2D and 3D Voronoi and power/Laguerre tessellations using
vendored Voro++ backends. It also solves a fixed-site inverse problem: fitting
power weights from selected pairwise separator observations. Algebraic fit and
realized cell boundaries are separate, inspectable results.

[Documentation](https://delonecommons.github.io/pyvoro2/) ·
[API reference](https://delonecommons.github.io/pyvoro2/reference/) ·
[Issues](https://github.com/DeloneCommons/pyvoro2/issues)

## Install

```bash
python -m pip install pyvoro2
```

NumPy is the only required runtime dependency. Standard, GIL-enabled CPython
3.10–3.14 is supported. Published v0.8 wheels cover manylinux x86_64, Windows
AMD64, and macOS arm64/x86_64. Source builds require a C++17 compiler, CMake
3.20+, and Python development headers; free-threaded CPython is unsupported.
See [Contributing](https://github.com/DeloneCommons/pyvoro2/blob/dev/CONTRIBUTING.md)
for source builds and editable development.

Optional extras include `pyvoro2[sparse]` for the explicit SciPy sparse quadratic
separator solver, `pyvoro2[viz2d]` for Matplotlib, and `pyvoro2[viz]` for
Matplotlib and py3Dmol. The examples below need only pyvoro2 and NumPy.

## Forward: compute cells

Two sites split the unit cube into equal volumes:

```python
import numpy as np
import pyvoro2 as pv

points = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
box = pv.Box(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
result = pv.compute(points, domain=box)

np.testing.assert_allclose(result.cell_measures, [0.5, 0.5])
print(result.cell_measures)  # [0.5 0.5]

# A power diagram moves the separator by changing the weights.
power = pv.compute(points, domain=box, mode="power", weights=[-0.05, 0.05])
np.testing.assert_allclose(power.cell_measures, [0.4, 0.6])
```

`compute` returns a `TessellationResult` with input-aligned `ids`,
`cell_measures`, `empty_mask`, raw `cells`, and capability-checked geometry.
Measures are volumes in 3D and areas in 2D. Use the explicit planar namespace
for a rectangular 2D domain:

```python
import numpy as np
import pyvoro2.planar as pv2

points = np.array([[0.25, 0.5], [0.75, 0.5]])
box = pv2.Box(((0.0, 1.0), (0.0, 1.0)))
result = pv2.compute(points, domain=box)
np.testing.assert_allclose(result.cell_measures, [0.5, 0.5])
```

The power function is `||x - p_i||² - w_i`. Mathematical weights have
squared-length units, may be negative, and share one global additive gauge:
adding the same constant to all weights leaves the diagram unchanged.
`radii=` is an alternative, non-unique backend representation in length units;
supply exactly one of `weights=` or `radii=` in power mode.

## Inverse: fit weights from separators

Use the stable `pyvoro2.inverse` facade for a fixed set of observations. Here,
`(0, 1, 0.25)` requests a separator one quarter of the way from site 0 to site 1
along their connector, whose length is 2:

```python
import numpy as np
import pyvoro2.inverse as inverse

points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
observations = inverse.resolve_separator_observations(
    points, [(0, 1, 0.25)], measurement="fraction"
)
fit = inverse.fit_weights_from_separators(points, observations)

assert fit.status == "optimal"
np.testing.assert_allclose(fit.predicted, [0.25])
np.testing.assert_allclose(fit.weights[0] - fit.weights[1], -2.0)
print(fit.weights)  # [0. 2.] in squared-length units
```

The sites stay fixed. Use fitted weights with `compute(..., mode="power",
weights=fit.weights)` in a supported domain to inspect the resulting cells.
A small observation residual does **not** prove that a requested pair or
periodic image is a realized face or edge. Disconnected observation components
also leave relative weight offsets unidentified; those offsets can change the
complete diagram.

The [separator-fitting guide](https://delonecommons.github.io/pyvoro2/guide/powerfit/)
explains confidence, connectivity, feasibility, and realization diagnostics.
Advanced models and reports are in `pyvoro2.inverse.separator`; its experimental
active-set refinement has no universal convergence guarantee.

## Supported capabilities and limits

- **3D `pyvoro2`:** bounded `Box`, `OrthorhombicCell` with per-axis periodicity,
  and fully periodic triclinic `PeriodicCell`. Partial triclinic periodicity is
  unsupported.
- **2D `pyvoro2.planar`:** bounded `Box` and axis-aligned `RectangularCell`
  with per-axis periodicity. Oblique periodic cells are unsupported.
- Both dimensions support standard and power tessellations, owner lookup with
  `locate`, non-inserting probe cells with `ghost_cells`, periodic neighbor-image
  labels, diagnostics, and topology normalization, with explicit capability
  differences. Power diagrams may contain empty cells.
- Inverse fitting uses selected pairwise separator observations in 2D or 3D.
  Prescribed-area/volume fitting, mixed separator-plus-measure inversion, and
  moving-site optimization are not implemented capabilities.

**Numerical care:** Voro++ uses fixed absolute tolerances. pyvoro2 rejects
inserted site pairs at squared separation at most `1e-10`; the optional
`duplicate_check` policy cannot disable this safety floor. Generators on
non-periodic axes must lie in half-open intervals `[lo, hi)`; periodic axes are
remapped. Very small/large coordinate scales and squared backend radii or true
weight ranges that overwhelm squared geometric scales can lose resolution.
Finite inputs and successful weight conversion do not guarantee reliable
geometry; rescale explicitly and inspect diagnostics. See
[capabilities and limitations](https://delonecommons.github.io/pyvoro2/guide/capabilities/)
and [power-diagram theory](https://delonecommons.github.io/pyvoro2/theory/power-diagrams/).

## Learn more

- [Choosing an API](https://delonecommons.github.io/pyvoro2/guide/choosing-api/)
  and [glossary](https://delonecommons.github.io/pyvoro2/guide/glossary/)
- [3D domains](https://delonecommons.github.io/pyvoro2/guide/domains/),
  [2D workflows](https://delonecommons.github.io/pyvoro2/guide/planar/), and
  [operations](https://delonecommons.github.io/pyvoro2/guide/operations/)
- [Topology and graphs](https://delonecommons.github.io/pyvoro2/guide/topology/)
- [Notebook examples](https://delonecommons.github.io/pyvoro2/guide/notebooks/)
  and [optional visualization](https://delonecommons.github.io/pyvoro2/guide/visualization/)

## Status, support, and contributing

pyvoro2 is **beta**. PyPI installs the published release; the `dev` branch is
the v0.8-based `0.9.0.dev0` reboot, preserving v0.8 numerical behavior and public
APIs. The former v0.9 attempt is historical evidence, and future architecture
and release scope remain undecided under
[ADR 0027](https://github.com/DeloneCommons/pyvoro2/blob/dev/docs/development/decisions/0027-v0.9-reboot-from-v0.8.md).
The archived v0.6.3 release remains the separator-inverse manuscript baseline.

Report reproducible bugs or focused proposals through
[GitHub issues](https://github.com/DeloneCommons/pyvoro2/issues). Support is
best-effort from a single maintainer. See
[Contributing](https://github.com/DeloneCommons/pyvoro2/blob/dev/CONTRIBUTING.md)
for development and validation, and the
[AI-assisted development policy](https://delonecommons.github.io/pyvoro2/project/ai/)
for the maintainer's review and scientific responsibility.

## License

pyvoro2-authored code is **LGPLv3+** from version 0.6.0; earlier releases were
MIT-licensed. See [LICENSE](https://github.com/DeloneCommons/pyvoro2/blob/dev/LICENSE)
and [COPYING](https://github.com/DeloneCommons/pyvoro2/blob/dev/COPYING).
Vendored Voro++ retains its upstream license; see
[LICENSE.voro++](https://github.com/DeloneCommons/pyvoro2/blob/dev/LICENSE.voro++)
and [NOTICE.md](https://github.com/DeloneCommons/pyvoro2/blob/dev/NOTICE.md).
