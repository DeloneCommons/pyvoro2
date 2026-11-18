"""Build both C2-1 candidates with native Zensical under /pyvoro2/.

This review-only helper varies one stylesheet in the same configuration.
Neither candidate introduces browser theme state or changes page content.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Use a fresh output directory to avoid stale previews.')
    config = yaml.safe_load((ROOT / 'mkdocs.yml').read_text())
    # Zensical requires site_dir inside the project root. Keep both temporary
    # paths there so previews never replace the ordinary site/ output.
    with tempfile.TemporaryDirectory(prefix='.theme-site-', dir=ROOT) as site, \
         tempfile.NamedTemporaryFile(
        prefix='.theme-preview-', suffix='.yml', dir=ROOT, mode='w',
    ) as scratch:
        for direction in ('graphite', 'midnight'):
            config['site_dir'] = str(Path(site).relative_to(ROOT))
            config['extra_css'] = [
                f'stylesheets/palettes/{direction}.css'
                if css.startswith('stylesheets/palettes/') else css
                for css in config['extra_css']
            ]
            filename = Path(scratch.name)
            filename.write_text(yaml.safe_dump(config, sort_keys=False))
            subprocess.run(
                [sys.executable, '-m', 'zensical', 'build', '--clean',
                 '--strict', '-f', str(filename)], cwd=ROOT, check=True,
            )
            shutil.copytree(site, output / direction / 'pyvoro2')


if __name__ == '__main__':
    main()
