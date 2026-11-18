"""Build C1 review alternatives at the same /pyvoro2/ path on separate servers.

No HTML generation: only temporary configuration for the native Zensical build.
Run from the repository root in the docs environment.
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
    if any((output / label / 'pyvoro2').exists() for label in ('a', 'b')):
        parser.error('Use a fresh output directory to avoid stale preview files.')
    for label, direction in [('a', 'editorial'), ('b', 'showcase')]:
        config = yaml.safe_load((ROOT / 'mkdocs.yml').read_text())
        config['site_dir'] = 'site'
        config.setdefault('extra', {})['landing_direction'] = direction
        # Zensical requires docs_dir inside the configuration's project root.
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yml', prefix='.c1-preview-', dir=ROOT,
        ) as temporary:
            temporary.write(yaml.safe_dump(config, sort_keys=False))
            temporary.flush()
            subprocess.run(
                [sys.executable, '-m', 'zensical', 'build', '--clean', '--strict',
                 '-f', temporary.name], cwd=ROOT, check=True,
            )
        shutil.copytree(ROOT / 'site', output / label / 'pyvoro2')


if __name__ == '__main__':
    main()
