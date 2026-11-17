# Triclinic Voronoi hero

[![Four translucent Voronoi cells with one dark site marker in each and a dashed skew lattice outline](../../docs/assets/landing/triclinic-voronoi.png)](../../docs/assets/landing/triclinic-voronoi.png)

**Asset** `docs/assets/landing/triclinic-voronoi.png` (1600 × 900, opaque warm white).
The maintainer selected B2-layered in [PR #136](https://github.com/DeloneCommons/pyvoro2/pull/136).
The final pass raised the face opacity of cells 70 and 20 from 9% to 14%
and increased the assembly's frame occupancy from 84% to 88% along the limiting
dimension. The original B2 orthographic viewing direction and edge styles
carry through. At 360 px the four sites remain distinct and the dashed lattice
recedes behind the solid cell edges. Some interior edges overlap in projection.

**Caption proposal.** Four complete Voronoi cells in a fully periodic triclinic
lattice. Their volumes sum to the volume of one lattice cell (3 cubic units).
The dashed lines outline its fundamental parallelepiped.

**Alt-text proposal.** Four translucent Voronoi polyhedra surround four dark
site markers. Blue and green foreground cells overlap two paler cells. Solid
cell edges extend across the dashed outline of a skew lattice cell.

## Geometry and provenance

The [forward guide](../../docs/guide/forward/index.md) supplies the four
fractional sites, external IDs 40, 10, 70, 20 and row lattice vectors.
The generator multiplies fractional coordinates by the row matrix and calls
public `pyvoro2.compute` in standard mode. Each polyhedron uses the site and
vertices returned together for its external ID, including the remapped site
20. Polygons retain their Cartesian positions on both sides of the lattice
outline. Coincident interfaces between displayed cells are drawn once,
using their image-qualified neighbors (56 oriented faces, 51 drawn polygons).
Colors, opacity and marker sizes are illustrative.

[Reference data](reference.json) retain the checked input, returned sites,
volumes and reference contact polygon from the reviewed example on pyvoro2
revision `9788f4e487e04b660c290a94daa83de9b5c1e16f` (0.9.0.dev0).
The generator checks nonempty cells and the volume sum, reconstructs each
volume from its polygons, tests every face's bisector and translated reciprocal
vertex set, and checks the reference contact `(40, 10, (-1, 0, 0))` against
neighbor images in the finite stencil and for an interior connector crossing.
B2 gives each interface the ordinary surface treatment.

[Render record](provenance.json) contains the measured residuals, camera,
opacity, renderer, dependency versions and SHA-256 hashes of the PNG and
generator. The [generator](../render_landing.py) specifies the lighting, line
weights, depth peeling and supersampling. This repository-authored illustration
and generator use LGPL-3.0-or-later under [LICENSE](../../LICENSE).
Voro++ attribution is in [NOTICE.md](../../NOTICE.md) and
[LICENSE.voro++](../../LICENSE.voro++).

## Reproduction

From the repository root at the asset commit, use a C++17 toolchain and a
Python 3.12 environment (recorded run: 3.12.14, Linux, EGL/Mesa llvmpipe).
Install the pinned packages into a dedicated offline-rendering environment.

```bash
python3.12 -m venv work/hero-venv
. work/hero-venv/bin/activate
python -m pip install -r tools/landing/requirements.txt
CC=gcc CXX=g++ python -m pip install -e .
python tools/render_landing.py --check-only
MPLCONFIGDIR="$PWD/work/mpl" XDG_CACHE_HOME="$PWD/work/cache" \
  VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow \
  python tools/render_landing.py --output work/hero.png > work/hero-record.json
sha256sum work/hero.png docs/assets/landing/triclinic-voronoi.png
```

Without `--output`, the generator writes the production PNG. The check-only
path uses pyvoro2 and NumPy. Rendering requires a VTK backend with working
depth peeling. The normal Zensical build consumes the committed static PNG.
On the recorded stack two independent runs produced identical PNGs and
records. Other graphics drivers can produce different raster bytes.
