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

    try:
        resolved_directory = directory.resolve(strict=True)
    except OSError as exc:
        raise DistributionMetadataCheckError(
            f'no .whl or .tar.gz distributions found in {directory}'
        ) from exc
    artifacts = tuple(
        sorted(
            (
                path.resolve(strict=True)
                for pattern in ('*.tar.gz', '*.whl')
                for path in resolved_directory.glob(pattern)
                if path.is_file()
            ),
            key=lambda artifact: str(artifact),
        )
    )
    if not artifacts:
        raise DistributionMetadataCheckError(
            f'no .whl or .tar.gz distributions found in {directory}'
        )
    return artifacts


def select_distribution_artifacts(paths: Sequence[Path]) -> tuple[Path, ...]:
    """Resolve directories and explicit artifact paths deterministically."""

    selected: set[Path] = set()
    for path in paths or (Path('dist'),):
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise DistributionMetadataCheckError(
                f'not a distribution directory, wheel, or sdist: {path}'
            ) from exc
        if resolved.is_dir():
            selected.update(distribution_artifacts(resolved))
        elif resolved.is_file() and (
            resolved.suffix == '.whl' or resolved.name.endswith('.tar.gz')
        ):
            selected.add(resolved)
        else:
            raise DistributionMetadataCheckError(
                f'not a distribution directory, wheel, or sdist: {path}'
            )
    artifacts = tuple(sorted(selected, key=lambda artifact: str(artifact)))
    if not artifacts:
        raise DistributionMetadataCheckError(
            'no .whl or .tar.gz distributions found in selected paths'
        )
    return artifacts


def run_twine_check_artifacts(
    artifacts: Sequence[Path],
    *,
    python_executable: str = sys.executable,
) -> tuple[Path, ...]:
    """Run Twine with one explicit argument per selected artifact."""

    selected = tuple(
        sorted(
            {Path(path).resolve(strict=True) for path in artifacts},
            key=lambda artifact: str(artifact),
        )
    )
    command = [
        python_executable,
        '-m',
        'twine',
        'check',
        *(str(path) for path in selected),
    ]
    print('+', ' '.join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return selected


def run_twine_check(
    directory: Path,
    *,
    python_executable: str = sys.executable,
) -> tuple[Path, ...]:
    """Run Twine with one explicit argument per discovered artifact."""

    artifacts = distribution_artifacts(directory)
    return run_twine_check_artifacts(
        artifacts,
        python_executable=python_executable,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Discover distributions and check their package metadata."""

    parser = argparse.ArgumentParser(
        description=(
            'Run Twine metadata checks for wheel and sdist files without '
            'shell glob expansion.'
        ),
    )
    parser.add_argument(
        'dist_dir_or_artifact',
        type=Path,
        nargs='*',
        help='directory or explicit .whl/.tar.gz distributions',
    )
    args = parser.parse_args(argv)

    try:
        artifacts = select_distribution_artifacts(args.dist_dir_or_artifact)
        run_twine_check_artifacts(artifacts)
    except DistributionMetadataCheckError as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
