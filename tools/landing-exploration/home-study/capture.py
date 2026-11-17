# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build a disposable classic site, then capture the two local homepage variants."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from threading import Thread

from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OLD_ALT = ('Two translucent cells share a copper face between site 40 and the periodic '
           'image of site 10 shifted by minus the first lattice vector.')
NEW_ALT = ('Four translucent Voronoi cells surround their generating sites. Two '
           'stronger fills expose the layered assembly across the dashed lattice.')


def build(directory):
    shutil.copytree(REPO / 'docs', directory / 'docs')
    shutil.copy(REPO / 'mkdocs.yml', directory)
    shutil.copy(REPO / 'CHANGELOG.md', directory)
    docs = directory / 'docs'
    shutil.copy(HERE / 'home.md', docs / 'index.md')
    with (docs / 'stylesheets/extra.css').open('a') as css:
        css.write('\n' + (HERE / 'home.css').read_text())
    assets = docs / 'assets/study'
    assets.mkdir()
    for name in ('c1-open', 'b2-layered'):
        shutil.copy(HERE.parent / 'round-2' / f'{name}.png', assets)
    subprocess.run(['zensical', 'build', '--clean', '--strict'], cwd=directory,
                   check=True)
    site = directory / 'site'
    original = (site / 'index.html').read_text()
    assert original.count('assets/study/c1-open.png') == original.count(OLD_ALT) == 1
    (site / 'b2.html').write_text(original.replace('assets/study/c1-open.png',
                                                   'assets/study/b2-layered.png')
                                  .replace(OLD_ALT, NEW_ALT))
    return site


def capture(site, output, chromium):
    handler = partial(SimpleHTTPRequestHandler, directory=str(site))
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    with httpd as server, sync_playwright() as pw:
        Thread(target=server.serve_forever, daemon=True).start()
        browser = pw.chromium.launch(executable_path=chromium)
        for kind, width, height in [('desktop', 1440, 900), ('mobile', 390, 844)]:
            page = browser.new_page(viewport=dict(width=width, height=height),
                                    device_scale_factor=1, is_mobile=kind == 'mobile',
                                    color_scheme='light', reduced_motion='reduce')
            # Capture offline. Both variants use the classic theme's font fallback.
            page.route('**/*', lambda r: r.continue_() if r.request.url.startswith(
                f'http://127.0.0.1:{server.server_port}/') else r.abort())
            shots = []
            for name, url in [('c1-open', 'index.html'), ('b2-layered', 'b2.html')]:
                page.goto(f'http://127.0.0.1:{server.server_port}/{url}',
                          wait_until='networkidle')
                img = page.locator('.geometry img')
                img.evaluate('(el) => el.decode()')
                page.evaluate('document.fonts.ready')
                box = img.bounding_box()
                assert box and abs(box['width'] / box['height'] - 16 / 9) < .001
                assert page.evaluate(
                    'document.documentElement.scrollWidth <= innerWidth')
                print(kind, name, box)
                path = output / f'{kind}-{name}.png'
                page.screenshot(path=path, animations='disabled')
                shots.append(Image.open(path).convert('RGB'))
            diff = ImageChops.difference(*shots).getbbox()
            assert diff and diff[0] >= int(box['x']) and diff[1] >= int(box['y'])
            assert diff[2] <= box['x'] + box['width'] + 1
            assert diff[3] <= box['y'] + box['height'] + 1
            page.close()
        browser.close()
        server.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep-preview', type=Path,
                        help='Copy the local site into a new directory for inspection')
    parser.add_argument('--chromium', default=shutil.which('chromium'))
    parser.add_argument('--output', type=Path, default=HERE / 'screenshots')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmp:
        preview = build(Path(tmp))
        capture(preview, args.output, args.chromium)
        if args.keep_preview:
            shutil.copytree(preview, args.keep_preview)
