#!/usr/bin/env python3
"""Run the full release-preparation checks for the repository."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv


REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = REPO_ROOT / 'dist'
BUILD_DIR = REPO_ROOT / 'build'


def _run(*args: str, env: dict[str, str] | None = None) -> None:
    """Run one subprocess command in the repository root."""

    print('+', ' '.join(args))
    subprocess.run(args, cwd=REPO_ROOT, check=True, env=env)


def _fresh_build_dirs() -> None:
    """Remove build artifacts from previous runs."""

    shutil.rmtree(DIST_DIR, ignore_errors=True)
    shutil.rmtree(BUILD_DIR, ignore_errors=True)


def _distribution_artifacts() -> list[Path]:
    """Return built distributions without relying on shell glob expansion."""

    return sorted((*DIST_DIR.glob('*.tar.gz'), *DIST_DIR.glob('*.whl')))


def _smoke_test_wheel() -> None:
    """Install the rebuilt wheel into a temporary base environment and test it."""

    wheels = sorted(DIST_DIR.glob('*.whl'))
    if len(wheels) != 1:
        raise RuntimeError(
            f'expected exactly one rebuilt wheel in dist/, found {len(wheels)}'
        )
    wheel = wheels[0]

    with tempfile.TemporaryDirectory(prefix='pyvoro2-release-check-') as tmp:
        env_dir = Path(tmp) / 'venv'
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(env_dir)
        bindir = 'Scripts' if sys.platform.startswith('win') else 'bin'
        python = env_dir / bindir / 'python'
        _run(str(python), '-m', 'pip', 'install', str(wheel))
        _run(str(python), '-m', 'pip', 'check')
        _run(
            str(python),
            str(REPO_ROOT / 'tools' / 'check_installed_package.py'),
            '--repo-root',
            str(REPO_ROOT),
            '--forbid-scipy',
        )


def main() -> int:
    """Run lint, tests, docs, build, metadata, and wheel smoke checks."""

    parser = argparse.ArgumentParser(
        description='Run the full release-preparation checks for the repository.',
    )
    parser.add_argument(
        '--skip-docs',
        action='store_true',
        help='skip the strict MkDocs build step',
    )
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='skip building distributions and validating dist artifacts',
    )
    parser.add_argument(
        '--skip-smoke-test',
        action='store_true',
        help='skip the temporary-virtualenv wheel smoke test',
    )
    args = parser.parse_args()

    _run('flake8', 'src', 'tests', 'tools', 'benchmarks', 'examples')
    # Notebook checking executes clean in-memory copies and must not refresh
    # committed outputs during release validation.
    _run(sys.executable, 'tools/check_notebooks.py')
    _run(sys.executable, 'tools/export_notebooks.py', '--check')
    _run(sys.executable, 'tools/gen_readme.py', '--check')
    _run(sys.executable, '-m', 'pytest', '-q')
    if not args.skip_docs:
        _run('mkdocs', 'build', '--strict')

    if args.skip_build:
        return 0

    _fresh_build_dirs()
    _run(sys.executable, '-m', 'build', '--sdist')
    _run(sys.executable, 'tools/build_wheel_from_sdist.py', 'dist')
    artifacts = _distribution_artifacts()
    _run(
        sys.executable,
        '-m',
        'twine',
        'check',
        *(str(path) for path in artifacts),
    )
    _run(sys.executable, 'tools/check_dist.py', 'dist')
    if not args.skip_smoke_test:
        _smoke_test_wheel()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
