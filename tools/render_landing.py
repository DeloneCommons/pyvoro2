# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate the selected four-cell triclinic documentation hero offline."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np
import pyvoro2 as pv


ROOT = Path(__file__).resolve().parents[1]
VECTORS = np.array([[2., 0., 0.], [.5, 1.5, 0.], [.25, .375, 1.]])
FRACTIONAL = np.array([[.15, .20, .30], [.65, .25, .35],
                       [.35, .75, .70], [.80, .70, .85]])
IDS = [40, 10, 70, 20]
CONTACT = (40, 10, (-1, 0, 0))
BACKGROUND = '#fafaf7'
COLORS = {40: '#77a4bc', 10: '#8db5a6', 70: '#b7c8ce', 20: '#bbc0b2'}
OPACITY = {40: .34, 10: .24, 70: .14, 20: .14}


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
    # Independent nearest-site check on the reference contact face, against a stated
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


def render(cells, output, *, dark=False):
    import pyvista as p
    from PIL import Image

    plot = p.Plotter(off_screen=True, window_size=(3200, 1800), lighting='none')
    # VTK captures premultiplied RGB, even with transparent_background=True.
    # Clear to transparent black so no background color is baked into RGB.
    plot.set_background('black')
    plot.renderer.SetBackgroundAlpha(0.)
    plot.ren_win.SetAlphaBitPlanes(1)
    colors = ({40: '#80abc0', 10: '#92baa7', 70: '#adbfc6', 20: '#bcc2ad'}
              if dark else COLORS)
    opacity = {40: .40, 10: .30, 70: .20, 20: .18} if dark else OPACITY
    edge_color = '#c1d0d1' if dark else '#41616d'
    edge_width = 5.6 if dark else 4.4
    site_color = '#e0e3df' if dark else '#274550'
    frame_color = '#879c9e' if dark else '#829093'
    frame_width = 4.4 if dark else 3.8
    # Supersample explicitly rather than stacking an AA render pass on peeling.
    plot.ren_win.SetMultiSamples(0)
    plot.enable_depth_peeling(number_of_peels=100, occlusion_ratio=0.)
    plot.add_light(p.Light(position=(-3, -4, 8), focal_point=(.5, .5, .5),
                           intensity=.7, light_type='scene light'))
    plot.add_light(p.Light(position=(4, 1, 4), focal_point=(.5, .5, .5),
                           intensity=.3, light_type='scene light'))
    # A pair of unshifted cells shares a rendered polygon only at zero shift.
    # Other shifts refer to distinct images, including self-image contacts.
    seen = set()
    contributions = 0
    for i in IDS:
        cell = cells[i]
        vertices = np.asarray(cell['vertices'])
        site = np.asarray(cell['site'])
        edges = set()
        for face in cell['faces']:
            contributions += 1
            loop = face['vertices']
            edges.update(tuple(sorted(edge)) for edge in zip(loop, loop[1:] + loop[:1]))
            j = face['adjacent_cell']
            shift = tuple(face['adjacent_shift'])
            # Anchor the key to the actual displayed images in Cartesian space.
            key = tuple(sorted(((i, (0, 0, 0)), (j, shift))))
            if key in seen:
                continue
            seen.add(key)
            points = vertices[loop]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            if normal @ (points.mean(axis=0) - site) < 0:
                points = points[::-1]
            poly = p.PolyData(points, [len(points), *range(len(points))])
            plot.add_mesh(poly, color=colors[i], opacity=opacity[i], ambient=.7,
                          diffuse=.3, specular=0, smooth_shading=False)
        lines = [v for a, b in sorted(edges) for v in (2, a, b)]
        plot.add_mesh(p.PolyData(vertices, lines=lines),
                      color=edge_color,
                      line_width=edge_width, lighting=False)
        plot.add_mesh(p.Sphere(radius=.032, center=site),
                      color=site_color,
                      ambient=.7, diffuse=.3, specular=0)
    assert (contributions, len(seen)) == (56, 51)
    for corner in itertools.product((0, 1), repeat=3):
        start = np.array(corner) @ VECTORS
        for bit, vector in zip(corner, VECTORS):
            if bit == 0:
                for t in np.arange(0, 1, .07):
                    segment = p.Line(start + t * vector,
                                     start + min(t + .04, 1) * vector)
                    plot.add_mesh(segment,
                                  color=frame_color, line_width=frame_width,
                                  lighting=False)
    # B2 orthographic camera; fit 88% of the limiting dimension (formerly 84%).
    position = (-3.892966418634691, -7.3811054742874065, 3.481510546940376)
    focal = (0.8458246034380476, 0.9117788143398852, 0.5197661581449148)
    up = (0.146943671674129, 0.2571514254297257, 0.9551338658818385)
    scale = 1.1206679898605656
    plot.camera_position = [position, focal, up]
    plot.enable_parallel_projection()
    plot.camera.parallel_scale = scale
    plot.reset_camera_clipping_range()
    pixels = plot.screenshot(return_img=True, transparent_background=True)
    peeling = bool(plot.renderer.GetLastRenderingUsedDepthPeeling())
    assert peeling, 'Depth peeling unavailable; do not publish unordered alpha'
    prefixes = ('OpenGL vendor', 'OpenGL renderer', 'OpenGL version')
    opengl = [line for line in plot.ren_win.ReportCapabilities().splitlines()
              if line.startswith(prefixes)]
    plot.close()
    assert pixels.shape == (1800, 3200, 4)
    assert np.all(pixels[:, :, :3] <= pixels[:, :, 3:4]), 'Expected premultiplied RGBA'
    # Average coverage and premultiplied color together, then encode straight
    # alpha for PNG. Positive box weights avoid Lanczos coverage overshoot and
    # ringing; floating arithmetic avoids Pillow's intermediate 8-bit clipping.
    samples = pixels.astype(float).reshape(900, 2, 1600, 2, 4).mean(axis=(1, 3))
    alpha = samples[:, :, 3:4] / 255
    rgb = np.divide(samples[:, :, :3], alpha,
                    out=np.zeros_like(samples[:, :, :3]), where=alpha > 0)
    rgba = np.rint(np.concatenate([rgb, samples[:, :, 3:4]], axis=2)).astype(np.uint8)
    assert (rgba[:, :, 3] == 0).any() and (rgba[:, :, 3] == 255).any()
    assert ((rgba[:, :, 3] > 0) & (rgba[:, :, 3] < 255)).any()
    full = Image.fromarray(rgba)
    full.save(output, optimize=True)
    return dict(displayed_polygons=len(seen), opacity=opacity, colors=colors,
                theme='carbon' if dark else 'light', transparent_background=True,
                background=None, framebuffer_clear_rgba=(0, 0, 0, 0),
                alpha_encoding='native premultiplied RGBA to straight PNG',
                cell_edges=dict(color=edge_color, width=edge_width),
                lattice_frame=dict(color=frame_color, width=frame_width,
                                   dash_step=.07, dash_length=.04),
                sites=dict(color=site_color, radius=.032),
                material=dict(ambient=.7, diffuse=.3, specular=0),
                lights=[dict(position=(-3, -4, 8), intensity=.7),
                        dict(position=(4, 1, 4), intensity=.3)],
                light_focal_point=(.5, .5, .5),
                render_size=(3200, 1800), output_size=(1600, 900),
                resampling='2x2 area average in floating premultiplied RGBA',
                multisamples=0,
                camera=dict(position=position, focal_point=focal, view_up=up,
                            parallel_scale=scale, projection='orthographic'),
                depth_peeling_used=peeling, opengl=opengl)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--theme', choices=('light', 'dark'), default='light')
    args = parser.parse_args()
    light = ROOT / 'docs/assets/landing/triclinic-voronoi.png'
    dark = ROOT / 'docs/assets/landing/triclinic-voronoi-dark.png'
    args.output = args.output or (dark if args.theme == 'dark' else light)
    protected = light if args.theme == 'dark' else dark
    if args.output.resolve() == protected:
        parser.error('Output would overwrite the other theme asset')
    if not __debug__:
        parser.error('Run with assertions enabled to check the scientific geometry.')
    cells, report = compute_and_check()
    geometry = json.dumps(cells, sort_keys=True, separators=(',', ':')).encode()
    report['geometry_sha256'] = hashlib.sha256(geometry).hexdigest()
    baseline = json.loads((ROOT / 'tools/landing/reference.json').read_text())
    for field in ('vectors', 'fractional_sites', 'ids'):
        assert report[field] == baseline[field], field
    for i in IDS:
        np.testing.assert_allclose(cells[i]['site'], baseline['returned_sites'][str(i)],
                                   atol=1e-12, rtol=0)
    for field in ('volumes_by_input_id', 'contact_vertices'):
        np.testing.assert_allclose(report[field], baseline[field], atol=1e-12, rtol=0)
    if not args.check_only:
        import PIL
        import pyvista
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report['render'] = render(cells, args.output, dark=args.theme == 'dark')
        report['versions'] = dict(pyvoro2=pv.__version__, numpy=np.__version__,
                                  pyvista=pyvista.__version__, pillow=PIL.__version__,
                                  vtk=pyvista.vtk_version_info,
                                  python=platform.python_version())
        report['png_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
        source = Path(__file__).read_bytes()
        report['generator_sha256'] = hashlib.sha256(source).hexdigest()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
