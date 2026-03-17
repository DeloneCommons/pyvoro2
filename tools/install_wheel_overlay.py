#!/usr/bin/env python3
"""Install a dev overlay that keeps wheel extension modules and uses repo code.

This script is intended for the workflow where:

1. a prebuilt pyvoro2 wheel is installed into the current Python environment,
2. the checked-out repository contains newer pure-Python code under ``src/``,
3. we want imports to resolve to the repository sources while still loading the
   compiled extension module(s) from the wheel.

The script performs three steps:

- copies or symlinks compiled binaries such as ``_core`` and ``_core2d`` into
  ``src/pyvoro2/``;
- writes a ``.pth`` file into the active environment to insert ``repo/src`` at
  the front of ``sys.path``;
- verifies in a fresh Python process that ``import pyvoro2`` resolves to the
  repository sources and that the copied extension module(s) are importable.

Typical usage::

    python -m pip install /path/to/pyvoro2-...whl
    python tools/install_wheel_overlay.py

If the wheel is not yet installed, the script can also extract extension
binaries directly from a wheel file via ``--wheel``. In that mode it still
writes the ``.pth`` overlay for the current environment.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
import textwrap
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = 'pyvoro2'
PACKAGE_SRC = PROJECT_ROOT / 'src' / PACKAGE_NAME
EXTENSION_PREFIXES = ('_core', '_core2d')


class OverlayError(RuntimeError):
    """Raised when the dev overlay cannot be installed."""


def _candidate_site_packages() -> list[Path]:
    keys = ('purelib', 'platlib')
    out: list[Path] = []
    for key in keys:
        path_str = sysconfig.get_paths().get(key)
        if not path_str:
            continue
        path = Path(path_str).resolve()
        if path not in out:
            out.append(path)
    return out


def _installed_extension_paths() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for site_dir in _candidate_site_packages():
        pkg_dir = site_dir / PACKAGE_NAME
        if not pkg_dir.exists():
            continue
        for prefix in EXTENSION_PREFIXES:
            cores = sorted(pkg_dir.glob(f'{prefix}*.so')) + sorted(
                pkg_dir.glob(f'{prefix}*.pyd')
            )
            if cores and prefix not in found:
                found[prefix] = cores[0]
    return found


def _copy_or_symlink(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == 'symlink':
        dst.symlink_to(src)
    elif mode == 'copy':
        shutil.copy2(src, dst)
    else:  # pragma: no cover - argparse guards this
        raise OverlayError(f'unsupported mode: {mode!r}')


def _extract_extensions_from_wheel(
    wheel_path: Path,
    target_dir: Path,
) -> dict[str, Path]:
    if not wheel_path.exists():
        raise OverlayError(f'wheel file not found: {wheel_path}')

    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(wheel_path) as zf:
        for prefix in EXTENSION_PREFIXES:
            names = [
                name
                for name in zf.namelist()
                if name.startswith(f'{PACKAGE_NAME}/{prefix}')
                and (name.endswith('.so') or name.endswith('.pyd'))
            ]
            if not names:
                continue
            member = names[0]
            target = target_dir / Path(member).name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            extracted[prefix] = target
    if '_core' not in extracted:
        raise OverlayError(f'no {PACKAGE_NAME}._core binary found in {wheel_path}')
    return extracted


def _write_pth(repo_src: Path, *, pth_name: str) -> Path:
    site_dirs = _candidate_site_packages()
    if not site_dirs:
        raise OverlayError('could not determine site-packages directories')
    pth_path = site_dirs[0] / pth_name
    repo_src_str = str(repo_src.resolve())
    payload = (
        'import sys; p = ' + repr(repo_src_str) + '; '
        'sys.path.insert(0, p) if p not in sys.path else None\n'
    )
    pth_path.write_text(payload, encoding='utf-8')
    return pth_path


def _verify_overlay(repo_src: Path) -> tuple[str, str, str]:
    code = textwrap.dedent(
        '''
        import pyvoro2
        import pyvoro2.api as api
        import pyvoro2.planar.api as api2
        print(pyvoro2.__file__)
        print(api._core.__file__)
        print('MISSING' if api2._core2d is None else api2._core2d.__file__)
        '''
    )
    proc = subprocess.run(
        [sys.executable, '-c', code],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) != 3:
        raise OverlayError(f'unexpected verification output: {proc.stdout!r}')
    py_file, core_file, core2d_file = lines
    repo_prefix = str(repo_src.resolve())
    if not py_file.startswith(repo_prefix):
        raise OverlayError(
            'overlay verification failed: Python package was not imported '
            f'from repo/src ({py_file})'
        )
    if not core_file.startswith(str((repo_src / PACKAGE_NAME).resolve())):
        raise OverlayError(
            'overlay verification failed: _core was not imported from the '
            f'repository package directory ({core_file})'
        )
    if core2d_file != 'MISSING' and not core2d_file.startswith(
        str((repo_src / PACKAGE_NAME).resolve())
    ):
        raise OverlayError(
            'overlay verification failed: _core2d was not imported from the '
            f'repository package directory ({core2d_file})'
        )
    return py_file, core_file, core2d_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--repo',
        type=Path,
        default=PROJECT_ROOT,
        help='pyvoro2 repository root (default: %(default)s)',
    )
    parser.add_argument(
        '--wheel',
        type=Path,
        default=None,
        help='optional wheel file to extract extension modules from',
    )
    parser.add_argument(
        '--mode',
        choices=('copy', 'symlink'),
        default='copy',
        help='how to place extension binaries into src/pyvoro2',
    )
    parser.add_argument(
        '--pth-name',
        default='pyvoro2_dev_overlay.pth',
        help='name of the .pth file written into site-packages',
    )
    args = parser.parse_args()

    repo_root = args.repo.resolve()
    repo_src = repo_root / 'src'
    package_dir = repo_src / PACKAGE_NAME
    if not package_dir.exists():
        raise OverlayError(f'package directory not found: {package_dir}')

    if args.wheel is not None:
        placed = _extract_extensions_from_wheel(args.wheel.resolve(), package_dir)
        core_source_note = f'extracted from wheel {args.wheel.resolve()}'
    else:
        installed = _installed_extension_paths()
        if '_core' not in installed:
            searched = ', '.join(str(p) for p in _candidate_site_packages())
            raise OverlayError(
                'could not find an installed pyvoro2 wheel core in the current '
                f'environment (searched: {searched}). Install a wheel first or '
                'pass --wheel /path/to/pyvoro2-...whl.'
            )
        placed = {}
        for prefix, src in installed.items():
            dst = package_dir / src.name
            _copy_or_symlink(src, dst, mode=args.mode)
            placed[prefix] = dst
        core_source_note = 'copied/symlinked from installed wheel extensions'

    pth_path = _write_pth(repo_src, pth_name=args.pth_name)
    py_file, core_file, core2d_file = _verify_overlay(repo_src)

    print('pyvoro2 dev overlay installed successfully')
    print(f'  repo src:    {repo_src}')
    print(f'  package:     {py_file}')
    print(f'  core:        {core_file}')
    if core2d_file == 'MISSING':
        print('  core2d:      MISSING (no planar extension in the installed wheel yet)')
    else:
        print(f'  core2d:      {core2d_file}')
    print(f'  .pth file:   {pth_path}')
    print(f'  core source: {core_source_note}')
    print('To remove the overlay later, delete the .pth file shown above.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except OverlayError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
