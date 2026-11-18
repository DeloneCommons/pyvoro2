"""Offline regression: regenerate all scientific assets twice, without overwriting.

Requires tools/landing/requirements.txt and the recorded EGL rendering stack.
Use --records DIR to retain generated PNGs and JSON records for inspection.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def verify(directory):
    records = {}
    expected = json.loads(
        (ROOT / 'tools/landing/production-provenance.json').read_text(),
    )['records']
    for name, generator in (
        ('triclinic-voronoi', 'tools/render_landing.py'),
        ('separator-cycle', 'tools/landing/render_separator_cycle.py'),
    ):
        geometries = []
        for theme in ('light', 'dark'):
            stem = name + ('-dark' if theme == 'dark' else '')
            asset = ROOT / 'docs/assets/landing' / (stem + '.png')
            runs = []
            for repeat in (1, 2):
                output = directory / f'{stem}-{repeat}.png'
                run = subprocess.run(
                    [sys.executable, str(ROOT / generator), '--theme', theme,
                     '--output', str(output)], check=True, capture_output=True,
                    text=True, env={**os.environ,
                                    'VTK_DEFAULT_OPENGL_WINDOW': 'vtkEGLRenderWindow'},
                )
                record = json.loads(run.stdout)
                assert output.read_bytes() == asset.read_bytes(), stem
                digest = hashlib.sha256(asset.read_bytes()).hexdigest()
                assert record['png_sha256'] == digest
                for key in ('geometry_sha256', 'generator_sha256'):
                    assert record[key] == expected[stem][key], f'{stem}: {key}'
                runs.append(record)
            assert runs[0] == runs[1], f'{stem}: records differ between runs'
            records[stem] = runs[0]
            geometries.append(runs[0]['geometry_sha256'])
            (directory / (stem + '.json')).write_text(
                json.dumps(runs[0], indent=2)+'\n',
            )
        assert geometries[0] == geometries[1], f'{name}: theme changed geometry'
    # The approved B2 camera and complete-cell polygon count are unchanged.
    historical = json.loads((ROOT / 'tools/landing/provenance.json').read_text())
    for name in ('triclinic-voronoi', 'triclinic-voronoi-dark'):
        assert records[name]['render']['camera'] == historical['render']['camera']
        assert records[name]['render']['displayed_polygons'] == 51
    print(json.dumps({'runs': 8, 'all_assets_byte_identical': True,
                      'light_dark_geometry_identical': True,
                      'approved_camera_unchanged': True,
                      'records': records}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path)
    args = parser.parse_args()
    if not __debug__:
        parser.error('Assertions must be enabled')
    if args.records:
        args.records.mkdir(parents=True, exist_ok=True)
        verify(args.records)
    else:
        with tempfile.TemporaryDirectory() as directory:
            verify(Path(directory))


if __name__ == '__main__':
    main()
