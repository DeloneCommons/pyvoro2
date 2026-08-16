#!/usr/bin/env python3
"""Verify that built distributions contain the project's key files."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
from pathlib import Path
from pathlib import PurePosixPath
from pathlib import PureWindowsPath
import tarfile
import zipfile

try:
    from check_wheel_matrix import native_module_members
except ModuleNotFoundError:  # Imported as ``tools.check_dist`` in tests.
    _matrix_path = Path(__file__).with_name('check_wheel_matrix.py')
    _matrix_spec = importlib.util.spec_from_file_location(
        '_pyvoro2_check_wheel_matrix_shared',
        _matrix_path,
    )
    if _matrix_spec is None or _matrix_spec.loader is None:
        raise ImportError(f'could not load wheel-matrix helpers from {_matrix_path}')
    _matrix_module = importlib.util.module_from_spec(_matrix_spec)
    _matrix_spec.loader.exec_module(_matrix_module)
    native_module_members = _matrix_module.native_module_members


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_LICENSE_PATH = REPO_ROOT / 'LICENSE'
NOTICE_PATH = REPO_ROOT / 'NOTICE.md'
VORO_LICENSE_PATH = REPO_ROOT / 'vendor' / 'voro++' / 'LICENSE'
PACKAGED_VORO_LICENSE_PATH = REPO_ROOT / 'LICENSE.voro++'
OS_INDEPENDENT_CLASSIFIER = b'Classifier: Operating System :: OS Independent'
WHEEL_LICENSE_RELATIVE_PATHS = {
    'project': 'licenses/LICENSE',
    'notice': 'licenses/NOTICE.md',
    'voro': 'licenses/LICENSE.voro++',
}
LICENSE_SOURCE_LABELS = {
    'project': 'repository root LICENSE',
    'notice': 'repository root NOTICE.md',
    'voro': 'vendor/voro++/LICENSE',
}


REQUIRED_WHEEL_FILES = {
    'pyvoro2/__init__.py',
    'pyvoro2/__about__.py',
    'pyvoro2/_internal/__init__.py',
    'pyvoro2/_internal/cell_output.py',
    'pyvoro2/_internal/inputs.py',
    'pyvoro2/_internal/power_input.py',
    'pyvoro2/_internal/weight_transforms.py',
    'pyvoro2/_internal/spatial/__init__.py',
    'pyvoro2/_internal/spatial/domain_geometry.py',
    'pyvoro2/_internal/spatial/domain_utils.py',
    'pyvoro2/_internal/spatial/face_shifts.py',
    'pyvoro2/_internal/planar/__init__.py',
    'pyvoro2/_internal/planar/domain_geometry.py',
    'pyvoro2/_internal/planar/edge_shifts.py',
    'pyvoro2/inverse/__init__.py',
    'pyvoro2/inverse/separator/__init__.py',
    'pyvoro2/inverse/separator/solver.py',
    'pyvoro2/planar/__init__.py',
    'pyvoro2/viz2d.py',
    'pyvoro2/viz3d.py',
}

REQUIRED_SDIST_FILES = {
    'README.md',
    'CHANGELOG.md',
    'AGENTS.md',
    'CONTRIBUTING.md',
    'LICENSE',
    'NOTICE.md',
    'LICENSE.voro++',
    'vendor/voro++/LICENSE',
    'pyproject.toml',
    'PKG-INFO',
    'src/pyvoro2/_internal/__init__.py',
    'src/pyvoro2/_internal/cell_output.py',
    'src/pyvoro2/_internal/inputs.py',
    'src/pyvoro2/_internal/power_input.py',
    'src/pyvoro2/_internal/weight_transforms.py',
    'src/pyvoro2/_internal/spatial/__init__.py',
    'src/pyvoro2/_internal/spatial/domain_geometry.py',
    'src/pyvoro2/_internal/spatial/domain_utils.py',
    'src/pyvoro2/_internal/spatial/face_shifts.py',
    'src/pyvoro2/_internal/planar/__init__.py',
    'src/pyvoro2/_internal/planar/domain_geometry.py',
    'src/pyvoro2/_internal/planar/edge_shifts.py',
    'src/pyvoro2/inverse/__init__.py',
    'src/pyvoro2/inverse/separator/__init__.py',
    'src/pyvoro2/inverse/separator/solver.py',
    'benchmarks/README.md',
    'benchmarks/benchmark_sparse_separator.py',
    'examples/README.md',
    'examples/__init__.py',
    'examples/chemvoro_workflow.py',
    'examples/paper_regressions.py',
    'examples/static_separator_cases.py',
    'notebooks/01_basic_compute.ipynb',
    'notebooks/02_periodic_graph.ipynb',
    'notebooks/03_locate_and_ghost.ipynb',
    'notebooks/04_powerfit.ipynb',
    'notebooks/05_visualization.ipynb',
    'notebooks/06_powerfit_reports.ipynb',
    'notebooks/07_powerfit_infeasibility.ipynb',
    'notebooks/08_powerfit_active_path.ipynb',
    'docs/notebooks/01_basic_compute.md',
    'docs/notebooks/02_periodic_graph.md',
    'docs/notebooks/03_locate_and_ghost.md',
    'docs/notebooks/04_powerfit.md',
    'docs/notebooks/05_visualization.md',
    'docs/notebooks/06_powerfit_reports.md',
    'docs/notebooks/07_powerfit_infeasibility.md',
    'docs/notebooks/08_powerfit_active_path.md',
    'docs/theory/power-diagrams.md',
    'docs/guide/choosing-api.md',
    'docs/guide/glossary.md',
    'docs/guide/migration-v0.7.md',
    'docs/theory/separator-inverse.md',
    'docs/reference/index.md',
    'docs/development/architecture.md',
    'docs/development/api-lifecycle.md',
    'docs/development/api-inventory.md',
    'docs/development/documentation-conventions.md',
    'docs/development/development-workflow.md',
    'docs/development/release-checklist-v0.7.md',
    'docs/development/decisions/0004-canonical-inverse-namespace.md',
    'docs/development/decisions/0005-tessellation-result-contract.md',
    'docs/development/decisions/0006-v0.8-cleanup-release.md',
    'docs/development/plans/index.md',
    'docs/development/plans/v0.8.md',
    'docs/development/plans/template.md',
    'docs/development/plans/archive/index.md',
    'docs/project/roadmap.md',
    'tools/build_wheel_from_sdist.py',
    'tools/check_installed_package.py',
    'tools/check_dist.py',
    'tools/check_dist_metadata.py',
    'tools/check_wheel_matrix.py',
    'tools/_notebook_tools.py',
    'tools/check_notebooks.py',
    'tools/execute_notebooks.py',
    'tools/export_notebooks.py',
    'tools/gen_readme.py',
    'tools/release_check.py',
    'tools/README.md',
}

FORBIDDEN_WHEEL_MARKERS = (
    'pyvoro2/powerfit/',
    'pyvoro2/planar/result.py',
    'pyvoro2/_cell_output.py',
    'pyvoro2/_domain_geometry.py',
    'pyvoro2/_face_shifts3d.py',
    'pyvoro2/_inputs.py',
    'pyvoro2/_power_input.py',
    'pyvoro2/_util.py',
    'pyvoro2/_weight_transforms.py',
    'pyvoro2/planar/_domain_geometry.py',
    'pyvoro2/planar/_edge_shifts2d.py',
)

FORBIDDEN_SDIST_MARKERS = (
    'src/pyvoro2/powerfit/',
    'src/pyvoro2/planar/result.py',
    'src/pyvoro2/_cell_output.py',
    'src/pyvoro2/_domain_geometry.py',
    'src/pyvoro2/_face_shifts3d.py',
    'src/pyvoro2/_inputs.py',
    'src/pyvoro2/_power_input.py',
    'src/pyvoro2/_util.py',
    'src/pyvoro2/_weight_transforms.py',
    'src/pyvoro2/planar/_domain_geometry.py',
    'src/pyvoro2/planar/_edge_shifts2d.py',
)


class DistCheckError(RuntimeError):
    """Raised when a built distribution is missing required members."""


def _expected_license_payload() -> dict[str, bytes]:
    """Return repository license bytes after checking the root Voro++ copy."""

    try:
        expected = {
            'project': PROJECT_LICENSE_PATH.read_bytes(),
            'notice': NOTICE_PATH.read_bytes(),
            'voro': VORO_LICENSE_PATH.read_bytes(),
        }
        packaged_voro = PACKAGED_VORO_LICENSE_PATH.read_bytes()
    except OSError as exc:
        raise DistCheckError(
            f'cannot read repository licensing files: {exc}'
        ) from exc
    if packaged_voro != expected['voro']:
        raise DistCheckError(
            'LICENSE.voro++ does not match vendor/voro++/LICENSE byte-for-byte'
        )
    return expected


def _assert_members_present(
    actual: set[str],
    required: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(required - actual)
    if missing:
        joined = ', '.join(missing)
        raise DistCheckError(f'{label} is missing required members: {joined}')


def _assert_safe_archive_names(names: list[str], *, label: str) -> None:
    """Reject archive member spellings that can alias another path."""

    for name in names:
        raw_parts = name.split('/')
        path_parts = raw_parts[:-1] if name.endswith('/') else raw_parts
        first = path_parts[0] if path_parts else ''
        if (
            not name
            or '\x00' in name
            or '\\' in name
            or PurePosixPath(name).is_absolute()
            or bool(PureWindowsPath(name).drive)
            or first.endswith(':')
            or any(part in {'', '.', '..'} for part in path_parts)
        ):
            raise DistCheckError(
                f'{label} contains unsafe archive member name {name!r}'
            )


def _assert_unique_file_names(names: list[str], *, label: str) -> None:
    duplicates = sorted(
        name for name, count in Counter(names).items() if count > 1
    )
    if duplicates:
        raise DistCheckError(
            f'{label} contains duplicate file members: {", ".join(duplicates)}'
        )


def _wheel_dist_info_root(
    file_names: list[str],
    *,
    label: str,
) -> str:
    roots = sorted(
        {
            PurePosixPath(name).parts[0]
            for name in file_names
            if len(PurePosixPath(name).parts) >= 2
            and PurePosixPath(name).parts[0].endswith('.dist-info')
        }
    )
    if len(roots) != 1:
        raise DistCheckError(
            f'{label} expected exactly one .dist-info root, found {len(roots)}'
        )
    return roots[0]


def _assert_notice_points_to_packaged_license(data: bytes, *, label: str) -> None:
    if b'LICENSE.voro++' not in data:
        raise DistCheckError(
            f'{label} does not point installed users to LICENSE.voro++'
        )


def _assert_matches_repository(
    data: bytes,
    expected: bytes,
    *,
    label: str,
    source_label: str,
) -> None:
    if data != expected:
        raise DistCheckError(
            f'{label} does not match {source_label} byte-for-byte'
        )


def _assert_platform_metadata(data: bytes, *, label: str) -> None:
    if OS_INDEPENDENT_CLASSIFIER in data:
        raise DistCheckError(
            f'{label} contains the unsupported OS Independent classifier'
        )


def _assert_members_absent(
    actual: set[str],
    forbidden_markers: tuple[str, ...],
    *,
    label: str,
) -> None:
    unexpected = sorted(
        name
        for name in actual
        if any(marker in name for marker in forbidden_markers)
    )
    if unexpected:
        joined = ', '.join(unexpected)
        raise DistCheckError(f'{label} contains removed members: {joined}')


def check_wheel(path: Path) -> None:
    """Validate the contents of one built wheel."""

    with zipfile.ZipFile(path) as zf:
        entries = zf.infolist()
        # ``filename`` is platform-normalized; safety depends on raw spelling.
        raw_names = [entry.orig_filename for entry in entries]
        _assert_safe_archive_names(
            raw_names,
            label=path.name,
        )
        for entry in entries:
            if entry.orig_filename != entry.filename:
                raise DistCheckError(
                    f'{path.name} contains unsafe archive member name '
                    f'{entry.orig_filename!r}'
                )

        file_entries = [entry for entry in entries if not entry.is_dir()]
        file_names = [entry.orig_filename for entry in file_entries]
        _assert_unique_file_names(file_names, label=path.name)
        files = set(file_names)
        entries_by_name = {
            entry.orig_filename: entry for entry in file_entries
        }
        _assert_members_present(files, REQUIRED_WHEEL_FILES, label=path.name)
        _assert_members_absent(files, FORBIDDEN_WHEEL_MARKERS, label=path.name)

        for module_name in ('_core', '_core2d'):
            members = native_module_members(file_names, module_name)
            if len(members) != 1:
                raise DistCheckError(
                    f'{path.name} expected exactly one pyvoro2/{module_name} '
                    f'native module, found {len(members)}'
                )

        dist_info_root = _wheel_dist_info_root(file_names, label=path.name)
        metadata_member = f'{dist_info_root}/METADATA'
        license_members = {
            kind: f'{dist_info_root}/{relative}'
            for kind, relative in WHEEL_LICENSE_RELATIVE_PATHS.items()
        }
        _assert_members_present(
            files,
            {metadata_member, *license_members.values()},
            label=path.name,
        )
        expected_licenses = _expected_license_payload()
        packaged_licenses = {
            kind: zf.read(entries_by_name[member])
            for kind, member in license_members.items()
        }
        _assert_notice_points_to_packaged_license(
            packaged_licenses['notice'],
            label=f'{path.name} NOTICE.md',
        )
        for kind, data in packaged_licenses.items():
            _assert_matches_repository(
                data,
                expected_licenses[kind],
                label=f'{path.name} packaged {license_members[kind]}',
                source_label=LICENSE_SOURCE_LABELS[kind],
            )
        _assert_platform_metadata(
            zf.read(entries_by_name[metadata_member]),
            label=f'{path.name} METADATA',
        )


def check_sdist(path: Path) -> None:
    """Validate the contents of one built source distribution."""

    with tarfile.open(path, 'r:gz') as tf:
        members = tf.getmembers()
        _assert_safe_archive_names(
            [member.name for member in members],
            label=path.name,
        )
        file_members = [member for member in members if member.isfile()]
        file_names = [member.name for member in file_members]
        _assert_unique_file_names(file_names, label=path.name)

        split_names = [name.split('/', 1) for name in file_names]
        if any(len(parts) != 2 for parts in split_names):
            raise DistCheckError(
                f'{path.name} contains files outside a generated top-level root'
            )
        roots = sorted({parts[0] for parts in split_names})
        if len(roots) != 1:
            raise DistCheckError(
                f'{path.name} expected exactly one top-level root, '
                f'found {len(roots)}: {", ".join(roots)}'
            )
        relative_names = [parts[1] for parts in split_names]
        _assert_unique_file_names(
            relative_names,
            label=f'{path.name} relative paths',
        )
        relative = set(relative_names)
        _assert_members_present(
            relative,
            REQUIRED_SDIST_FILES,
            label=path.name,
        )
        _assert_members_absent(
            relative,
            FORBIDDEN_SDIST_MARKERS,
            label=path.name,
        )

        by_relative = {
            member.name.split('/', 1)[1]: member
            for member in file_members
        }

        def read_member(relative_name: str) -> bytes:
            extracted = tf.extractfile(by_relative[relative_name])
            if extracted is None:
                raise DistCheckError(
                    f'{path.name} could not read {relative_name}'
                )
            return extracted.read()

        expected_licenses = _expected_license_payload()
        packaged_notice = read_member('NOTICE.md')
        _assert_notice_points_to_packaged_license(
            packaged_notice,
            label=f'{path.name} NOTICE.md',
        )
        packaged_licenses = (
            ('LICENSE', read_member('LICENSE'), 'project'),
            ('NOTICE.md', packaged_notice, 'notice'),
            ('LICENSE.voro++', read_member('LICENSE.voro++'), 'voro'),
            (
                'vendor/voro++/LICENSE',
                read_member('vendor/voro++/LICENSE'),
                'voro',
            ),
        )
        for relative_name, data, kind in packaged_licenses:
            _assert_matches_repository(
                data,
                expected_licenses[kind],
                label=f'{path.name} {relative_name}',
                source_label=LICENSE_SOURCE_LABELS[kind],
            )
        _assert_platform_metadata(
            read_member('PKG-INFO'),
            label=f'{path.name} PKG-INFO',
        )


def distribution_artifacts(paths: list[Path]) -> tuple[Path, ...]:
    """Select wheel and sdist artifacts from directories or explicit paths."""

    selected: set[Path] = set()
    for path in paths or [Path('dist')]:
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise DistCheckError(
                f'not a distribution directory, wheel, or sdist: {path}'
            ) from exc
        if resolved.is_dir():
            selected.update(
                candidate.resolve(strict=True)
                for pattern in ('*.whl', '*.tar.gz')
                for candidate in resolved.glob(pattern)
                if candidate.is_file()
            )
        elif resolved.is_file() and (
            resolved.suffix == '.whl' or resolved.name.endswith('.tar.gz')
        ):
            selected.add(resolved)
        else:
            raise DistCheckError(
                f'not a distribution directory, wheel, or sdist: {path}'
            )
    return tuple(sorted(selected, key=lambda artifact: str(artifact)))


def main() -> None:
    """Validate wheel and sdist artifacts found in a distribution directory."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dist_dir_or_artifact',
        type=Path,
        nargs='*',
        help='distribution directory or explicit .whl/.tar.gz artifacts',
    )
    args = parser.parse_args()

    artifacts = distribution_artifacts(args.dist_dir_or_artifact)
    wheels = [path for path in artifacts if path.suffix == '.whl']
    sdists = [path for path in artifacts if path.name.endswith('.tar.gz')]
    if not wheels:
        raise DistCheckError('no wheel files found in selected artifacts')
    if not sdists:
        raise DistCheckError('no source distributions found in selected artifacts')

    for wheel in wheels:
        check_wheel(wheel)
    for sdist in sdists:
        check_sdist(sdist)


if __name__ == '__main__':
    main()
