"""Build the selected Home for local review under the /pyvoro2/ path.

No HTML generation: invoke the native Zensical build and copy its output.
Run from the repository root in the docs environment.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / 'pyvoro2').exists():
        parser.error('Use a fresh output directory to avoid stale preview files.')
    subprocess.run(
        [sys.executable, '-m', 'zensical', 'build', '--clean', '--strict'],
        cwd=ROOT, check=True,
    )
    shutil.copytree(ROOT / 'site', output / 'pyvoro2')


if __name__ == '__main__':
    main()
