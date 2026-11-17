# SPDX-License-Identifier: LGPL-3.0-or-later
"""Offline DOC-W4-A exploration; public pyvoro2 geometry, optional PyVista.

See landing-exploration/README.md. Nothing here runs during the docs build.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pyvoro2 as pv


ROOT = Path(__file__).resolve().parents[1]
VECTORS = np.array([[2., 0., 0.], [.5, 1.5, 0.], [.25, .375, 1.]])
FRACTIONAL = np.array([[.15, .20, .30], [.65, .25, .35],
                       [.35, .75, .70], [.80, .70, .85]])
IDS = [40, 10, 70, 20]
CONTACT = (40, 10, (-1, 0, 0))
BACKGROUND = '#fafaf7'
INK = '#354950'
ACCENT = '#c77843'


def polygon(cell, face):
    return np.asarray(cell['vertices'])[face['vertices']]


def find_face(cells, i, j, shift):
    matches = [f for f in cells[i]['faces'] if f['adjacent_cell'] == j
               and tuple(f['adjacent_shift']) == tuple(shift)]
    assert len(matches) == 1, (i, j, shift, len(matches))
    return matches[0]


def compute_and_check():
    """Check returned records and independent elementary geometry before drawing."""
    result = pv.compute(
        FRACTIONAL @ VECTORS, domain=pv.PeriodicCell(vectors=VECTORS),
        ids=IDS, return_vertices=True, return_faces=True, return_face_shifts=True,
    )
    cells = {c['id']: c for c in result.cells}
    assert len(result.cells) == 4 and set(cells) == set(IDS)
    np.testing.assert_array_equal(result.ids, IDS)
    assert not result.empty_mask.any()
    np.testing.assert_allclose(result.cell_measures.sum(), 3., atol=1e-12,
                               rtol=1e-10)
    np.testing.assert_allclose(np.linalg.det(VECTORS), 3., atol=1e-12)
    plane_error = reciprocal_error = volume_error = 0.
    count = 0
    for i, cell in cells.items():
        vertices = np.asarray(cell['vertices'])
        site = np.asarray(cell['site'])
        assert vertices.shape[0] >= 4 and np.isfinite(vertices).all()
        assert cell['faces'] and cell['volume'] > 0
        # Sum tetrahedra from an interior site and each face's triangle fan.
        # This uses actual polygons, independently of the reported volumes.
        volume = 0.
        for face in cell['faces']:
            p = polygon(cell, face)
            assert len(p) >= 3
            for k in range(1, len(p) - 1):
                volume += abs(np.dot(p[0] - site, np.cross(
                    p[k] - site, p[k + 1] - site))) / 6
            j, shift = face['adjacent_cell'], np.asarray(face['adjacent_shift'])
            assert shift.shape == (3,) and np.equal(shift, np.rint(shift)).all()
            delta = shift @ VECTORS
            neighbor = np.asarray(cells[j]['site']) + delta
            normal = neighbor - site
            error = np.max(np.abs((p - (site + neighbor) / 2) @ normal))
            error /= np.linalg.norm(normal)
            plane_error = max(plane_error, float(error))
            assert error < 1e-10, (i, j, shift, error)
            reciprocal = find_face(cells, j, i, -shift)
            q = polygon(cells[j], reciprocal) + delta
            assert p.shape == q.shape
            distances = np.linalg.norm(p[:, None] - q[None, :], axis=2)
            error = max(distances.min(axis=0).max(), distances.min(axis=1).max())
            reciprocal_error = max(reciprocal_error, float(error))
            assert error < 1e-10, (i, j, shift, error)
            count += 1
        volume_error = max(volume_error, abs(volume - cell['volume']))
        np.testing.assert_allclose(volume, cell['volume'], atol=1e-12, rtol=1e-10)
    i, j, shift = CONTACT
    assert any(shift)
    face = find_face(cells, i, j, shift)
    p = polygon(cells[i], face)
    si = np.asarray(cells[i]['site'])
    sj = np.asarray(cells[j]['site']) + np.asarray(shift) @ VECTORS
    midpoint = (si + sj) / 2
    # For this contact, the site connector actually pierces the face interior.
    # That is not guaranteed for an arbitrary Voronoi neighbor pair.
    side = np.cross(np.roll(p, -1, axis=0) - p, midpoint - p) @ (sj - si)
    assert np.all(side < -1e-10) or np.all(side > 1e-10)
    # Independent nearest-site check on the highlighted face, against a stated
    # finite stencil (not a general certificate for arbitrary lattices).
    images = np.concatenate([
        np.asarray([cells[k]['site'] for k in IDS]) + np.array(s) @ VECTORS
        for s in itertools.product(range(-2, 3), repeat=3)
    ])
    probes = np.vstack([p, p.mean(axis=0), midpoint])
    distance2 = ((probes - cells[i]['site']) ** 2).sum(axis=1)
    competitor2 = ((probes[:, None] - images[None, :]) ** 2).sum(axis=2)
    violation = float(np.max(distance2 - competitor2.min(axis=1)))
    assert violation < 1e-10
    report = dict(
        vectors=VECTORS.tolist(), fractional_sites=FRACTIONAL.tolist(), ids=IDS,
        returned_sites={str(k): c['site'] for k, c in cells.items()},
        volumes_by_input_id=result.cell_measures.tolist(),
        volume_sum=float(result.cell_measures.sum()), lattice_volume=3.,
        oriented_faces_checked=count, max_bisector_distance=plane_error,
        max_reciprocal_vertex_distance=reciprocal_error,
        max_independent_volume_error=volume_error,
        contact=CONTACT, reciprocal=(j, i, tuple(-s for s in shift)),
        contact_vertices=p.tolist(), contact_nearest_site_violation=violation,
        connector_midpoint_inside_face=True,
        nearest_site_stencil='[-2,2]^3; face vertices, centroid, connector midpoint',
    )
    return cells, report


def mesh(pvplot, cell, delta=None, only_face=None):
    points = np.asarray(cell['vertices'])
    if delta is not None:
        points = points + delta
    faces = cell['faces'] if only_face is None else [only_face]
    indices = []
    site = np.asarray(cell['site'])
    if delta is not None:
        site = site + delta
    for face in faces:
        loop = list(face['vertices'])
        xyz = points[loop]
        normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
        if np.dot(normal, xyz.mean(axis=0) - site) < 0:
            # Rendering winding only: exactly the same polygon and coordinates.
            loop.reverse()
        indices.extend([len(loop), *loop])
    return pvplot.PolyData(points, indices)


def edges(pvplot, cell, delta=None):
    points = np.asarray(cell['vertices'])
    if delta is not None:
        points = points + delta
    pairs = set()
    for face in cell['faces']:
        v = face['vertices']
        pairs.update(tuple(sorted((a, b))) for a, b in zip(v, v[1:] + v[:1]))
    lines = [n for a, b in sorted(pairs) for n in (2, a, b)]
    return pvplot.PolyData(points, lines=lines)


def lattice(plot, pvplot, color='#6e7f83', width=2.8, dashed=True):
    for corner in itertools.product((0, 1), repeat=3):
        for axis in range(3):
            if corner[axis]:
                continue
            start = np.array(corner) @ VECTORS
            end = start + VECTORS[axis]
            if dashed:
                for t in np.arange(0, 1, .07):
                    segment = pvplot.Line(
                        start + t * (end - start),
                        start + min(t + .04, 1) * (end - start))
                    plot.add_mesh(segment,
                                  color=color, line_width=width, lighting=False)
            else:
                plot.add_mesh(pvplot.Line(start, end), color=color,
                              line_width=width, lighting=False)


def render(cells, name, output, camera=None):
    import pyvista as p
    from PIL import Image

    plot = p.Plotter(off_screen=True, window_size=(1600, 900), lighting='none')
    plot.set_background(BACKGROUND)
    plot.enable_anti_aliasing('ssaa')
    plot.add_light(p.Light(position=(-3, -4, 8), focal_point=(1, .5, .5),
                           intensity=.8, light_type='scene light'))
    plot.add_light(p.Light(position=(-4, -2, 1), focal_point=(1, .5, .5),
                           intensity=.45, light_type='scene light'))
    i, j, shift = CONTACT
    delta = np.array(shift) @ VECTORS
    face = find_face(cells, *CONTACT)

    def surface(cell, color, translation=None):
        plot.add_mesh(mesh(p, cell, translation), color=color, smooth_shading=False,
                      ambient=.5, diffuse=.5, specular=0, show_edges=True,
                      edge_color=INK, line_width=1.8)

    def wire(cell, color=INK, translation=None, width=2.4):
        plot.add_mesh(edges(p, cell, translation), color=color, line_width=width,
                      lighting=False)

    def marker(point, color=INK, radius=.035):
        plot.add_mesh(p.Sphere(radius=radius, center=point,
                               theta_resolution=32, phi_resolution=32),
                      color=color, ambient=.4, specular=0)

    if name == 'a-cell':
        surface(cells[i], '#a8bcc0')
        lattice(plot, p)
        # Replace the face's appearance, never its geometry; polygon offset
        # avoids z fighting without displacing the returned vertices.
        actor = plot.add_mesh(mesh(p, cells[i], only_face=face), color=ACCENT,
                              ambient=.4, diffuse=.6, specular=0,
                              show_edges=True, edge_color=INK, line_width=2.4)
        actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        actor.mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)
        position, focal, scale = (-5, 5, 2.5), (1., .85, .48), 1.2
    elif name == 'b-lattice':
        for k, color in [(70, '#d4d9d5'), (20, '#bdc7c5'),
                         (10, '#96aeb4'), (40, '#6b8c96')]:
            surface(cells[k], color)
        lattice(plot, p, color='#45565c', width=3.)
        position, focal, scale = (5, -7, 5), (1.35, .9, .5), 1.75
    elif name == 'c-contact':
        wire(cells[i], '#617d87', width=3.2)
        wire(cells[j], '#617d87', delta, width=3.2)
        plot.add_mesh(mesh(p, cells[i], only_face=face), color=ACCENT,
                      ambient=.5, diffuse=.5, specular=0,
                      show_edges=True, edge_color=INK, line_width=3.5)
        si, sj = np.asarray(cells[i]['site']), np.asarray(cells[j]['site']) + delta
        marker(si, INK, .042)
        marker(sj, INK, .042)
        plot.add_mesh(p.Line(si, sj), color=INK, line_width=3, lighting=False)
        lattice(plot, p, color='#88989b', width=2.4)
        position, focal, scale = (-3.5, -6, 3.5), (.5, .5, .45), 1.65
    else:
        raise ValueError(name)
    if camera is not None:
        position, focal, scale = camera
    plot.camera_position = [position, focal, (0, 0, 1)]
    plot.enable_parallel_projection()
    plot.camera.parallel_scale = scale
    plot.reset_camera_clipping_range()
    plot.screenshot(str(output / f'{name}.png'))
    renderer = plot.ren_win.GetClassName()
    prefixes = ('OpenGL vendor', 'OpenGL renderer', 'OpenGL version')
    opengl = [line for line in plot.ren_win.ReportCapabilities().splitlines()
              if line.startswith(prefixes)]
    plot.close()
    with Image.open(output / f'{name}.png') as im:
        im.save(output / f'{name}.png', optimize=True)
        for width in (800, 360):
            im.resize((width, round(width * 9 / 16)), Image.Resampling.LANCZOS).save(
                output / f'{name}-{width}.png', optimize=True)
    return dict(position=position, focal_point=focal, view_up=(0, 0, 1),
                parallel_scale=scale, projection='orthographic', renderer=renderer,
                opengl=opengl)


def review_sheet(output):
    """Native 360 px thumbnails, grayscale, and a 60% contrast stress test."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

    sheet = Image.new('RGB', (1080, 765), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for col, name in enumerate(('a-cell', 'b-lattice', 'c-contact')):
        im = Image.open(output / f'{name}-360.png').convert('RGB')
        for row, (label, thumb) in enumerate([
            ('360 px / color', im),
            ('360 px / grayscale', ImageOps.grayscale(im).convert('RGB')),
            ('360 px / 60% contrast', ImageEnhance.Contrast(im).enhance(.6)),
        ]):
            x, y = col * 360, row * 255
            draw.text((x + 14, y + 12), f'{name[0].upper()}   {label}',
                      fill=INK, font=font)
            sheet.paste(thumb, (x, y + 42))
    sheet.save(output / 'small-scale-review.png', optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'tools/landing-exploration/candidates')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    cells, report = compute_and_check()
    print(json.dumps(report, indent=2))
    if args.check_only:
        return
    import pyvista
    import PIL
    args.output.mkdir(parents=True, exist_ok=True)
    report['cameras'] = {name: render(cells, name, args.output)
                         for name in ('a-cell', 'b-lattice', 'c-contact')}
    review_sheet(args.output)
    report['versions'] = dict(pyvoro2=pv.__version__, numpy=np.__version__,
                              pyvista=pyvista.__version__, pillow=PIL.__version__,
                              vtk=pyvista.vtk_version_info,
                              python=platform.python_version())
    report['source_revision'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    report['generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['images_sha256'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.glob('*.png'))
    }
    (args.output / 'checks.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
