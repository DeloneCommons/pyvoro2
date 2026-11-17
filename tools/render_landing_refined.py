# SPDX-License-Identifier: LGPL-3.0-or-later
"""Second DOC-W4-A exploration: unchanged data/checks, translucent B/C studies."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

import render_landing as original


ROOT = Path(__file__).resolve().parents[1]
COLORS = {40: '#77a4bc', 10: '#8db5a6', 70: '#b7c8ce', 20: '#bbc0b2'}
SCENES = {
    'b1-open': dict(direction=(-4, 7, 3.5), opacity={i: .16 for i in original.IDS}),
    'b2-layered': dict(direction=(-4, -7, 2.5),
                       opacity={40: .34, 10: .24, 70: .09, 20: .09}),
    'c1-open': dict(direction=(-5, -7, 3), opacity={40: .18, 10: .18}),
    'c2-layered': dict(direction=(-4, 7, 2.8), opacity={40: .34, 10: .3}),
}


def members(name):
    if name.startswith('b'):
        return [(i, (0, 0, 0)) for i in original.IDS]
    return [(40, (0, 0, 0)), (10, original.CONTACT[2])]


def unique_faces(cells, displayed):
    """Retain every geometric face, drawing shared interfaces once.

    Match using returned image-qualified relationships, not rounded coordinates.
    Original checks verify reciprocal vertex sets before this function runs.
    """
    seen = set()
    records = []
    for i, image_shift in displayed:
        for face in cells[i]['faces']:
            neighbor = (face['adjacent_cell'], tuple(
                int(a + b) for a, b in zip(image_shift, face['adjacent_shift'])))
            key = tuple(sorted(((i, image_shift), neighbor)))
            if key in seen:
                continue
            seen.add(key)
            records.append((i, image_shift, face, neighbor in displayed))
    contributions = sum(len(cells[i]['faces']) for i, _ in displayed)
    assert contributions - len(records) == sum(r[3] for r in records)
    return records, contributions


def camera_for(points, direction, aspect=16 / 9):
    """Fit all complete cells and lattice with an 8% margin; no geometric crop."""
    towards_camera = np.asarray(direction, dtype=float)
    towards_camera /= np.linalg.norm(towards_camera)
    right = np.cross(-towards_camera, (0, 0, 1))
    right /= np.linalg.norm(right)
    up = np.cross(right, -towards_camera)
    center = points.mean(axis=0)
    projected = np.column_stack((points @ right, points @ up))
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    midpoint = (lo + hi) / 2
    focal = center + (midpoint[0] - center @ right) * right
    focal += (midpoint[1] - center @ up) * up
    scale = max((hi[1] - lo[1]) / 2, (hi[0] - lo[0]) / (2 * aspect)) / .84
    return focal + 10 * towards_camera, focal, up, float(scale)


def render(cells, name, output):
    import pyvista as p
    from PIL import Image

    config = SCENES[name]
    displayed = members(name)
    records, contributions = unique_faces(cells, displayed)
    plot = p.Plotter(off_screen=True, window_size=(3200, 1800), lighting='none')
    plot.set_background(original.BACKGROUND)
    # Supersample explicitly rather than stacking an AA render pass on peeling.
    plot.ren_win.SetMultiSamples(0)
    plot.enable_depth_peeling(number_of_peels=100, occlusion_ratio=0.)
    plot.add_light(p.Light(position=(-3, -4, 8), focal_point=(.5, .5, .5),
                           intensity=.7, light_type='scene light'))
    plot.add_light(p.Light(position=(4, 1, 4), focal_point=(.5, .5, .5),
                           intensity=.3, light_type='scene light'))
    highlighted = 0
    for i, shift, face, shared in records:
        delta = np.asarray(shift) @ original.VECTORS
        contact = (i, face['adjacent_cell'], tuple(face['adjacent_shift']))
        accent = name.startswith('c') and contact == original.CONTACT
        highlighted += accent
        color = original.ACCENT if accent else COLORS[i]
        opacity = .86 if accent else config['opacity'][i]
        plot.add_mesh(original.mesh(p, cells[i], delta, only_face=face),
                      color=color, opacity=opacity, smooth_shading=False,
                      ambient=.7, diffuse=.3, specular=0)
        if accent:
            polygon = original.polygon(cells[i], face) + delta
            plot.add_mesh(p.lines_from_points(polygon, close=True),
                          color='#92542f', line_width=6., lighting=False)
    assert highlighted == int(name.startswith('c'))
    all_points = [np.array(list(itertools.product((0, 1), repeat=3)))
                  @ original.VECTORS]
    for i, shift in displayed:
        delta = np.asarray(shift) @ original.VECTORS
        vertices = np.asarray(cells[i]['vertices']) + delta
        all_points.append(vertices)
        plot.add_mesh(original.edges(p, cells[i], delta), color='#41616d',
                      line_width=4.4, lighting=False)
        site = np.asarray(cells[i]['site']) + delta
        plot.add_mesh(p.Sphere(radius=.032, center=site), color='#274550',
                      ambient=.7, diffuse=.3, specular=0)
    if name.startswith('c'):
        si = np.asarray(cells[40]['site'])
        sj = np.asarray(cells[10]['site']) + np.array(original.CONTACT[2]) @ (
            original.VECTORS)
        plot.add_mesh(p.Line(si, sj), color='#274550', line_width=5., lighting=False)
    original.lattice(plot, p, color='#829093', width=3.8)
    position, focal, up, scale = camera_for(
        np.concatenate(all_points), config['direction'])
    plot.camera_position = [position, focal, up]
    plot.enable_parallel_projection()
    plot.camera.parallel_scale = scale
    plot.reset_camera_clipping_range()
    pixels = plot.screenshot(return_img=True)
    peeling = bool(plot.renderer.GetLastRenderingUsedDepthPeeling())
    assert peeling, 'Depth peeling unavailable; do not publish unordered alpha'
    prefixes = ('OpenGL vendor', 'OpenGL renderer', 'OpenGL version')
    opengl = [line for line in plot.ren_win.ReportCapabilities().splitlines()
              if line.startswith(prefixes)]
    plot.close()
    full = Image.fromarray(pixels).resize((1600, 900), Image.Resampling.LANCZOS)
    full.save(output / f'{name}.png', optimize=True)
    for width in (800, 360):
        full.resize((width, round(width * 9 / 16)), Image.Resampling.LANCZOS).save(
            output / f'{name}-{width}.png', optimize=True)
    return dict(
        displayed=[dict(id=i, image_shift=s) for i, s in displayed],
        original_face_contributions=contributions, displayed_polygons=len(records),
        shared_interfaces_drawn_once=sum(r[3] for r in records),
        highlighted_faces=highlighted, colors=COLORS, opacity=config['opacity'],
        contact_opacity=.86 if highlighted else None,
        camera=dict(position=position.tolist(), focal_point=focal.tolist(),
                    view_up=up.tolist(), parallel_scale=scale,
                    projection='orthographic'),
        depth_peeling_used=peeling, opengl=opengl,
    )


def sheets(output):
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

    # C left, B right, with variants paired by visual treatment.
    pairs = [('c1-open', 'b1-open'), ('c2-layered', 'b2-layered')]
    for width in (1600, 800, 360):
        h = round(width * 9 / 16)
        pad = round(width * .06)
        sheet = Image.new('RGB', (2 * width, 2 * (h + pad)), original.BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=round(width * .035))
        for row, pair in enumerate(pairs):
            for col, name in enumerate(pair):
                suffix = '' if width == 1600 else f'-{width}'
                im = Image.open(output / f'{name}{suffix}.png').convert('RGB')
                x, y = col * width, row * (h + pad)
                draw.text((x + pad // 2, y + pad // 4), name.upper(),
                          font=font, fill=original.INK)
                sheet.paste(im, (x, y + pad))
        sheet.save(output / f'comparison-{width}.png', optimize=True)
    sheet = Image.new('RGB', (1440, 765), original.BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=17)
    for col, name in enumerate(('c1-open', 'b1-open', 'c2-layered', 'b2-layered')):
        im = Image.open(output / f'{name}-360.png').convert('RGB')
        for row, (label, thumb) in enumerate([
            ('color', im), ('grayscale', ImageOps.grayscale(im).convert('RGB')),
            ('60% contrast', ImageEnhance.Contrast(im).enhance(.6)),
        ]):
            x, y = col * 360, row * 255
            draw.text((x + 12, y + 12), f'{name.upper()} / {label}',
                      font=font, fill=original.INK)
            sheet.paste(thumb, (x, y + 42))
    sheet.save(output / 'small-scale-review.png', optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'tools/landing-exploration/round-2')
    parser.add_argument('--only', choices=SCENES)
    args = parser.parse_args()
    cells, report = original.compute_and_check()
    original_report = json.loads((ROOT / 'tools/landing-exploration/candidates/'
                                 'checks.json').read_text())
    # Freeze inputs exactly; compare output coordinates at a stated tolerance
    # so compiler/BLAS rounding alone need not prevent reproduction elsewhere.
    for field in ('vectors', 'fractional_sites', 'ids'):
        assert report[field] == original_report[field], field
    for site_id in report['returned_sites']:
        np.testing.assert_allclose(report['returned_sites'][site_id],
                                   original_report['returned_sites'][site_id],
                                   rtol=0, atol=1e-12)
    for field in ('volumes_by_input_id', 'contact_vertices'):
        np.testing.assert_allclose(report[field], original_report[field],
                                   rtol=0, atol=1e-12)
    report['round_1_input_match'] = True
    report['round_1_output_comparison_atol'] = 1e-12
    args.output.mkdir(parents=True, exist_ok=True)
    names = [args.only] if args.only else SCENES
    report['scenes'] = {name: render(cells, name, args.output) for name in names}
    if not args.only:
        sheets(args.output)
    import PIL
    import pyvista
    report['versions'] = dict(pyvoro2=original.pv.__version__,
                              numpy=np.__version__, pillow=PIL.__version__,
                              pyvista=pyvista.__version__, vtk=pyvista.vtk_version_info,
                              python=platform.python_version())
    report['source_revision'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    report['generator_sha256'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(original.__file__))
    }
    report['images_sha256'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.glob('*.png'))
    }
    (args.output / 'checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
