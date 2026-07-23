#!/usr/bin/env python3
"""Build exactly one wheel from exactly one generated sdist."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


class SdistWheelBuildError(RuntimeError):
    """Raised when the sdist-to-wheel artifact path is ambiguous."""


def select_only_sdist(dist_dir: Path) -> Path:
    """Return the only sdist in a directory or raise a focused error."""

    sdists = sorted(dist_dir.glob('*.tar.gz'))
    if len(sdists) != 1:
        raise SdistWheelBuildError(
            f'expected exactly one sdist in {dist_dir}, found {len(sdists)}'
        )
    return sdists[0]


def build_wheel_from_sdist(dist_dir: Path, wheel_dir: Path) -> Path:
    """Build one wheel from the selected sdist using pip build isolation."""

    sdist = select_only_sdist(dist_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    existing_wheels = sorted(wheel_dir.glob('*.whl'))
    if existing_wheels:
        raise SdistWheelBuildError(
            f'wheel output directory contains existing wheel artifacts: {wheel_dir}'
        )

    print(f'building wheel from sdist: {sdist.resolve()}')
    subprocess.run(
        [
            sys.executable,
            '-m',
            'pip',
            'wheel',
            '--no-cache-dir',
            '--no-deps',
            '--wheel-dir',
            str(wheel_dir),
            str(sdist),
        ],
        check=True,
    )

    wheels = sorted(wheel_dir.glob('*.whl'))
    if len(wheels) != 1:
        raise SdistWheelBuildError(
            f'expected exactly one rebuilt wheel in {wheel_dir}, '
            f'found {len(wheels)}'
        )
    print(f'rebuilt wheel: {wheels[0].resolve()}')
    return wheels[0]


def main() -> int:
    """Select one sdist and rebuild its wheel in an isolated PEP 517 build."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dist_dir',
        type=Path,
        nargs='?',
        default=Path('dist'),
        help='directory containing exactly one .tar.gz sdist',
    )
    parser.add_argument(
        '--wheel-dir',
        type=Path,
        default=None,
        help=(
            'output directory containing no existing wheel artifacts '
            '(default: dist_dir)'
        ),
    )
    args = parser.parse_args()

    wheel_dir = args.dist_dir if args.wheel_dir is None else args.wheel_dir
    build_wheel_from_sdist(args.dist_dir, wheel_dir)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SdistWheelBuildError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
