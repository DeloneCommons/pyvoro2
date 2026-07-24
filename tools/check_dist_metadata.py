#!/usr/bin/env python3
"""Run Twine metadata checks on explicitly discovered distributions."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


class DistributionMetadataCheckError(RuntimeError):
    """Raised when distribution artifacts cannot be selected."""


def distribution_artifacts(directory: Path) -> tuple[Path, ...]:
    """Return wheel and sdist files in deterministic path order."""

    artifacts = tuple(
        sorted(
            path
            for pattern in ('*.tar.gz', '*.whl')
            for path in directory.glob(pattern)
            if path.is_file()
        )
    )
    if not artifacts:
        raise DistributionMetadataCheckError(
            f'no .whl or .tar.gz distributions found in {directory}'
        )
    return artifacts


def run_twine_check(
    directory: Path,
    *,
    python_executable: str = sys.executable,
) -> tuple[Path, ...]:
    """Run Twine with one explicit argument per discovered artifact."""

    artifacts = distribution_artifacts(directory)
    command = [
        python_executable,
        '-m',
        'twine',
        'check',
        *(str(path) for path in artifacts),
    ]
    print('+', ' '.join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return artifacts


def main(argv: Sequence[str] | None = None) -> int:
    """Discover distributions and check their package metadata."""

    parser = argparse.ArgumentParser(
        description=(
            'Run Twine metadata checks for wheel and sdist files without '
            'shell glob expansion.'
        ),
    )
    parser.add_argument(
        'dist_dir',
        type=Path,
        help='directory containing .whl and .tar.gz distributions',
    )
    args = parser.parse_args(argv)

    try:
        run_twine_check(args.dist_dir)
    except DistributionMetadataCheckError as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
