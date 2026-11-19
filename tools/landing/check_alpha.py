"""Offline native-alpha/compositing regression; requires the optional renderer stack.

Run with VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow. No production files change.
"""

import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import pyvista as pv


ROOT = Path(__file__).resolve().parents[2]


def area(pixels):
    """Independent reference integral: four equal-area samples per output pixel."""
    return sum(pixels[y::2, x::2].astype(float)
               for y in (0, 1) for x in (0, 1)) / 4


def check_layers():
    """Two analytical translucent layers, independent of B2 geometry/order."""
    centers = []
    for reverse in (False, True):
        plot = pv.Plotter(off_screen=True, window_size=(64, 64), lighting='none')
        plot.set_background('black')
        plot.renderer.SetBackgroundAlpha(0.)
        plot.ren_win.SetAlphaBitPlanes(1)
        plot.ren_win.SetMultiSamples(0)
        plot.enable_depth_peeling(number_of_peels=100, occlusion_ratio=0.)
        layers = [(0., 'blue', .25), (.2, 'red', .5)]
        for z, color, opacity in layers[::(-1 if reverse else 1)]:
            plot.add_mesh(pv.Plane(center=(0, 0, z)), color=color,
                          opacity=opacity, lighting=False)
        plot.camera_position = [(0, 0, 3), (0, 0, 0), (0, 1, 0)]
        plot.enable_parallel_projection()
        plot.camera.parallel_scale = .8
        image = plot.screenshot(return_img=True, transparent_background=True)
        assert plot.renderer.GetLastRenderingUsedDepthPeeling()
        # Front red contributes .5; back blue .25 * (1-.5); alpha is .625.
        np.testing.assert_allclose(image[32, 32], np.array([.5, 0, .125, .625]) * 255,
                                   atol=1, rtol=0)
        np.testing.assert_array_equal(image[0, 0], [0, 0, 0, 0])
        centers.append(image[32, 32].tolist())
        plot.close()
    assert centers[0] == centers[1], 'Actor insertion order changed depth order'
    return centers[0]


def main():
    if not __debug__:
        raise SystemExit('Assertions must be enabled')
    spec = importlib.util.spec_from_file_location(
        'landing', ROOT / 'tools/render_landing.py')
    landing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(landing)
    cells, _ = landing.compute_and_check()
    original = pv.Plotter.screenshot
    reports = {'analytical_layers_rgba': check_layers()}
    with tempfile.TemporaryDirectory() as temporary:
        for dark in (False, True):
            captures = {}

            def capture(plot, *args, **kwargs):
                native = original(plot, *args, **kwargs)
                assert native.shape == (1800, 3200, 4), 'Expected native RGBA'
                assert plot.renderer.GetLastRenderingUsedDepthPeeling()
                captures['native'] = native.copy()
                for name, color in [('white', 'white'),
                                    ('theme', '#242728' if dark else '#fafaf7')]:
                    plot.set_background(color)
                    plot.render()
                    captures[name] = original(
                        plot, return_img=True, transparent_background=False)
                    assert plot.renderer.GetLastRenderingUsedDepthPeeling()
                return native

            pv.Plotter.screenshot = capture
            try:
                output = Path(temporary) / 'candidate.png'
                landing.render(cells, output, dark=dark)
            finally:
                pv.Plotter.screenshot = original
            native = captures['native'].astype(float)
            assert np.all(native[:, :, :3] <= native[:, :, 3:4]), 'Not premultiplied'
            a = native[:, :, 3:4] / 255
            assert (a == 0).any() and (a == 1).any() and ((a > 0) & (a < 1)).any()
            assert not native[a[:, :, 0] == 0].any(), 'Unused pixels are not zero'
            response_error = np.abs(
                captures['white'] - native[:, :, :3] - 255 * (1 - a)).max()
            assert response_error <= 1, response_error
            final = np.asarray(Image.open(output)).astype(float)
            assert final.shape == (900, 1600, 4)
            np.testing.assert_array_equal(
                final[:, :, 3], np.rint(area(native)[:, :, 3]))
            alpha = final[:, :, 3:4] / 255
            errors = {}
            backgrounds = [('white', [255, 255, 255]),
                           ('theme', [36, 39, 40] if dark else [250, 250, 247])]
            for name, background in backgrounds:
                composite = final[:, :, :3] * alpha + np.array(background) * (1 - alpha)
                error = float(np.abs(composite - area(captures[name])).max())
                assert error <= 2, (name, error)
                errors[name] = error
            reports['carbon' if dark else 'light'] = dict(
                native_response_max_error=float(response_error),
                composite_max_error=errors)
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
