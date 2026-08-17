from __future__ import annotations

from email.message import Message
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tarfile
from types import ModuleType, SimpleNamespace
import warnings
import zipfile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_LICENSE_BYTES = (REPO_ROOT / 'LICENSE').read_bytes()
COPYING_BYTES = (REPO_ROOT / 'COPYING').read_bytes()
NOTICE_BYTES = (REPO_ROOT / 'NOTICE.md').read_bytes()
VORO_LICENSE_BYTES = (
    REPO_ROOT / 'vendor' / 'voro++' / 'LICENSE'
).read_bytes()
TEST_DIST_INFO_ROOT = 'pyvoro2-0.8.0.dev0.dist-info'
TEST_WHEEL_LICENSE = f'{TEST_DIST_INFO_ROOT}/licenses/LICENSE'
TEST_WHEEL_COPYING = f'{TEST_DIST_INFO_ROOT}/licenses/COPYING'
TEST_WHEEL_NOTICE = f'{TEST_DIST_INFO_ROOT}/licenses/NOTICE.md'
TEST_WHEEL_VORO_LICENSE = f'{TEST_DIST_INFO_ROOT}/licenses/LICENSE.voro++'
TEST_WHEEL_METADATA = f'{TEST_DIST_INFO_ROOT}/METADATA'
TEST_WHEEL_FILES = (
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
    'pyvoro2/_core.test.so',
    'pyvoro2/_core2d.test.so',
    TEST_WHEEL_METADATA,
    TEST_WHEEL_LICENSE,
    TEST_WHEEL_COPYING,
    TEST_WHEEL_NOTICE,
    TEST_WHEEL_VORO_LICENSE,
)
TEST_SDIST_FILES = tuple(
    """
README.md
CHANGELOG.md
AGENTS.md
CONTRIBUTING.md
LICENSE
COPYING
NOTICE.md
LICENSE.voro++
vendor/voro++/LICENSE
pyproject.toml
PKG-INFO
src/pyvoro2/_internal/__init__.py
src/pyvoro2/_internal/cell_output.py
src/pyvoro2/_internal/inputs.py
src/pyvoro2/_internal/power_input.py
src/pyvoro2/_internal/weight_transforms.py
src/pyvoro2/_internal/spatial/__init__.py
src/pyvoro2/_internal/spatial/domain_geometry.py
src/pyvoro2/_internal/spatial/domain_utils.py
src/pyvoro2/_internal/spatial/face_shifts.py
src/pyvoro2/_internal/planar/__init__.py
src/pyvoro2/_internal/planar/domain_geometry.py
src/pyvoro2/_internal/planar/edge_shifts.py
src/pyvoro2/inverse/__init__.py
src/pyvoro2/inverse/separator/__init__.py
src/pyvoro2/inverse/separator/solver.py
benchmarks/README.md
benchmarks/benchmark_sparse_separator.py
examples/README.md
examples/__init__.py
examples/chemvoro_workflow.py
examples/paper_regressions.py
examples/static_separator_cases.py
notebooks/01_basic_compute.ipynb
notebooks/02_periodic_graph.ipynb
notebooks/03_locate_and_ghost.ipynb
notebooks/04_powerfit.ipynb
notebooks/05_visualization.ipynb
notebooks/06_powerfit_reports.ipynb
notebooks/07_powerfit_infeasibility.ipynb
notebooks/08_powerfit_active_path.ipynb
docs/notebooks/01_basic_compute.md
docs/notebooks/02_periodic_graph.md
docs/notebooks/03_locate_and_ghost.md
docs/notebooks/04_powerfit.md
docs/notebooks/05_visualization.md
docs/notebooks/06_powerfit_reports.md
docs/notebooks/07_powerfit_infeasibility.md
docs/notebooks/08_powerfit_active_path.md
docs/theory/power-diagrams.md
docs/guide/choosing-api.md
docs/guide/glossary.md
docs/guide/migration-v0.7.md
docs/theory/separator-inverse.md
docs/reference/index.md
docs/development/architecture.md
docs/development/api-lifecycle.md
docs/development/api-inventory.md
docs/development/documentation-conventions.md
docs/development/development-workflow.md
docs/development/release-checklist-v0.7.md
docs/development/decisions/0004-canonical-inverse-namespace.md
docs/development/decisions/0005-tessellation-result-contract.md
docs/development/decisions/0006-v0.8-cleanup-release.md
docs/development/plans/index.md
docs/development/plans/archive/v0.8.md
docs/development/plans/template.md
docs/development/plans/archive/index.md
docs/project/roadmap.md
tools/build_wheel_from_sdist.py
tools/check_installed_package.py
tools/check_dist.py
tools/check_dist_metadata.py
tools/check_wheel_matrix.py
tools/_notebook_tools.py
tools/check_notebooks.py
tools/execute_notebooks.py
tools/export_notebooks.py
tools/gen_readme.py
tools/release_check.py
tools/README.md
""".split()
)


def _load_tool_module(script_name: str) -> ModuleType:
    path = REPO_ROOT / 'tools' / f'{script_name}.py'
    spec = importlib.util.spec_from_file_location(
        f'_pyvoro2_test_{script_name}',
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f'could not load tool module from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_wheel_tool = _load_tool_module('build_wheel_from_sdist')
SdistWheelBuildError = build_wheel_tool.SdistWheelBuildError
build_wheel_from_sdist = build_wheel_tool.build_wheel_from_sdist
select_only_sdist = build_wheel_tool.select_only_sdist

dist_metadata_tool = _load_tool_module('check_dist_metadata')
DistributionMetadataCheckError = (
    dist_metadata_tool.DistributionMetadataCheckError
)
distribution_artifacts = dist_metadata_tool.distribution_artifacts
select_distribution_artifacts = (
    dist_metadata_tool.select_distribution_artifacts
)
run_twine_check_artifacts = dist_metadata_tool.run_twine_check_artifacts
run_twine_check = dist_metadata_tool.run_twine_check

installed_package_tool = _load_tool_module('check_installed_package')
InstalledPackageCheckError = installed_package_tool.InstalledPackageCheckError
assert_outside_repository = installed_package_tool.assert_outside_repository

dist_tool = _load_tool_module('check_dist')
DistCheckError = dist_tool.DistCheckError
check_sdist = dist_tool.check_sdist
check_wheel = dist_tool.check_wheel
select_content_artifacts = dist_tool.distribution_artifacts

overlay_tool = _load_tool_module('install_wheel_overlay')

wheel_matrix_tool = _load_tool_module('check_wheel_matrix')
WheelMatrixError = wheel_matrix_tool.WheelMatrixError
classify_platform_tag = wheel_matrix_tool.classify_platform_tag
parse_wheel_filename = wheel_matrix_tool.parse_wheel_filename
validate_wheel_matrix = wheel_matrix_tool.validate_wheel_matrix


WHEEL_MATRIX_VERSION = '0.8.0.dev0'
WHEEL_MATRIX_PLATFORM_TAGS = (
    'manylinux_2_17_x86_64.manylinux2014_x86_64',
    'win_amd64',
    'macosx_11_0_arm64',
    'macosx_10_15_x86_64',
)
WHEEL_MATRIX_RUNTIME_REQUIREMENTS = (
    'numpy<2,>=1.23; python_version < "3.11"',
    'numpy<3,>=1.23; python_version >= "3.11"',
)
WHEEL_MATRIX_OPTIONAL_REQUIREMENTS = (
    'scipy>=1.8; extra == "test"',
)


def _run_help(script_name: str) -> str:
    result = subprocess.run(
        [sys.executable, f'tools/{script_name}', '--help'],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_release_check_help() -> None:
    assert 'release-preparation checks' in _run_help('release_check.py')


def test_check_notebooks_help() -> None:
    assert 'optional notebook filenames' in _run_help('check_notebooks.py')


def test_execute_notebooks_help() -> None:
    assert 'optional notebook filenames' in _run_help('execute_notebooks.py')


def test_check_dist_help() -> None:
    assert 'dist_dir' in _run_help('check_dist.py')


def test_check_dist_metadata_help() -> None:
    assert 'without shell glob' in _run_help('check_dist_metadata.py')


def test_check_installed_package_help() -> None:
    assert 'representative public workflows' in _run_help(
        'check_installed_package.py'
    )


def test_check_wheel_matrix_help() -> None:
    assert 'merged wheels and sdist' in _run_help('check_wheel_matrix.py')


def _write_content_wheel(
    path: Path,
    *,
    extra_members: tuple[str, ...] = (),
    omitted_members: tuple[str, ...] = (),
    member_data: dict[str, bytes] | None = None,
) -> None:
    overrides = {} if member_data is None else member_data
    defaults = {
        TEST_WHEEL_LICENSE: PROJECT_LICENSE_BYTES,
        TEST_WHEEL_COPYING: COPYING_BYTES,
        TEST_WHEEL_NOTICE: NOTICE_BYTES,
        TEST_WHEEL_VORO_LICENSE: VORO_LICENSE_BYTES,
    }
    entries = [
        (member, overrides.get(member, defaults.get(member, b'')))
        for member in TEST_WHEEL_FILES
        if member not in omitted_members
    ]
    entries.extend((member, b'') for member in extra_members)
    _write_wheel_entries(path, entries)


def _write_wheel_entries(
    path: Path,
    entries: list[tuple[str, bytes]],
) -> None:
    with zipfile.ZipFile(path, 'w') as zf:
        for member, data in entries:
            _writestr_exact(zf, member, data)


def _writestr_exact(
    zf: zipfile.ZipFile,
    member: str,
    data: str | bytes,
) -> None:
    """Write an exact member spelling, including deliberately unsafe names."""

    entry = zipfile.ZipInfo(member)
    entry.filename = member
    zf.writestr(entry, data)


def _write_content_sdist(
    path: Path,
    *,
    extra_members: tuple[str, ...] = (),
    omitted_members: tuple[str, ...] = (),
    member_data: dict[str, bytes] | None = None,
) -> None:
    prefix = 'pyvoro2-0.8.0.dev0'
    overrides = {} if member_data is None else member_data
    defaults = {
        'LICENSE': PROJECT_LICENSE_BYTES,
        'COPYING': COPYING_BYTES,
        'NOTICE.md': NOTICE_BYTES,
        'LICENSE.voro++': VORO_LICENSE_BYTES,
        'vendor/voro++/LICENSE': VORO_LICENSE_BYTES,
    }
    entries = [
        (
            f'{prefix}/{suffix}',
            overrides.get(suffix, defaults.get(suffix, b'')),
        )
        for suffix in TEST_SDIST_FILES
        if suffix not in omitted_members
    ]
    entries.extend((f'{prefix}/{member}', b'') for member in extra_members)
    _write_sdist_entries(path, entries)


def _write_sdist_entries(
    path: Path,
    entries: list[tuple[str, bytes]],
) -> None:
    with tarfile.open(path, 'w:gz') as tf:
        for name, data in entries:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tf.addfile(member, io.BytesIO(data))


def _read_wheel_entries(path: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(path) as zf:
        return [
            (entry.orig_filename, zf.read(entry))
            for entry in zf.infolist()
            if not entry.is_dir()
        ]


def _read_sdist_entries(path: Path) -> list[tuple[str, bytes]]:
    with tarfile.open(path, 'r:gz') as tf:
        return [
            (member.name, tf.extractfile(member).read())
            for member in tf.getmembers()
            if member.isfile()
        ]


def test_distribution_content_checks_require_internal_hierarchy(
    tmp_path: Path,
) -> None:
    required_internal = {
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
    }
    assert required_internal <= dist_tool.REQUIRED_WHEEL_FILES
    assert {
        f'src/{member}'
        for member in required_internal
    } <= dist_tool.REQUIRED_SDIST_FILES

    wheel = tmp_path / 'pyvoro2-test.whl'
    sdist = tmp_path / 'pyvoro2-test.tar.gz'
    _write_content_wheel(wheel)
    _write_content_sdist(sdist)

    check_wheel(wheel)
    check_sdist(sdist)


def test_wheel_content_check_does_not_treat_core2d_as_core(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / 'pyvoro2-test.whl'
    _write_content_wheel(
        wheel,
        omitted_members=('pyvoro2/_core.test.so',),
    )

    with pytest.raises(DistCheckError, match=r'pyvoro2/_core native.*found 0'):
        check_wheel(wheel)


def test_wheel_content_check_requires_core2d(tmp_path: Path) -> None:
    wheel = tmp_path / 'pyvoro2-test.whl'
    _write_content_wheel(
        wheel,
        omitted_members=('pyvoro2/_core2d.test.so',),
    )

    with pytest.raises(DistCheckError, match=r'pyvoro2/_core2d native.*found 0'):
        check_wheel(wheel)


def test_wheel_content_check_rejects_misleading_core_name(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / 'pyvoro2-test.whl'
    _write_content_wheel(
        wheel,
        omitted_members=('pyvoro2/_core.test.so',),
        extra_members=('pyvoro2/_coreevil.test.so',),
    )

    with pytest.raises(DistCheckError, match=r'pyvoro2/_core native.*found 0'):
        check_wheel(wheel)


@pytest.mark.parametrize('module_name', ['_core', '_core2d'])
def test_wheel_content_check_rejects_ambiguous_native_modules(
    tmp_path: Path,
    module_name: str,
) -> None:
    wheel = tmp_path / 'pyvoro2-test.whl'
    _write_content_wheel(
        wheel,
        extra_members=(f'pyvoro2/{module_name}.other.so',),
    )

    with pytest.raises(
        DistCheckError,
        match=rf'pyvoro2/{module_name} native.*found 2',
    ):
        check_wheel(wheel)


def test_wheel_content_check_rejects_prefixed_required_python_files(
    tmp_path: Path,
) -> None:
    valid = tmp_path / 'valid.whl'
    malformed = tmp_path / 'prefixed.whl'
    _write_content_wheel(valid)
    entries = [
        (f'evil/{name}' if name.endswith('.py') else name, data)
        for name, data in _read_wheel_entries(valid)
    ]
    _write_wheel_entries(malformed, entries)

    with pytest.raises(DistCheckError, match='missing required members'):
        check_wheel(malformed)


@pytest.mark.parametrize(
    'target',
    [TEST_WHEEL_COPYING, TEST_WHEEL_VORO_LICENSE],
)
def test_wheel_content_check_rejects_duplicate_license_before_valid_copy(
    tmp_path: Path,
    target: str,
) -> None:
    valid = tmp_path / 'valid.whl'
    malformed = tmp_path / 'duplicate-license.whl'
    _write_content_wheel(valid)
    entries: list[tuple[str, bytes]] = []
    for name, data in _read_wheel_entries(valid):
        if name == target:
            entries.append((name, b'corrupt'))
        entries.append((name, data))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        _write_wheel_entries(malformed, entries)

    with pytest.raises(DistCheckError, match='duplicate file members'):
        check_wheel(malformed)


@pytest.mark.parametrize('target_suffix', ['COPYING', 'LICENSE.voro++'])
def test_sdist_content_check_rejects_duplicate_license_before_valid_copy(
    tmp_path: Path,
    target_suffix: str,
) -> None:
    valid = tmp_path / 'valid.tar.gz'
    malformed = tmp_path / 'duplicate-license.tar.gz'
    _write_content_sdist(valid)
    target = f'pyvoro2-0.8.0.dev0/{target_suffix}'
    entries: list[tuple[str, bytes]] = []
    for name, data in _read_sdist_entries(valid):
        if name == target:
            entries.append((name, b'corrupt'))
        entries.append((name, data))
    _write_sdist_entries(malformed, entries)

    with pytest.raises(DistCheckError, match='duplicate file members'):
        check_sdist(malformed)


def test_sdist_content_check_rejects_required_files_split_between_roots(
    tmp_path: Path,
) -> None:
    valid = tmp_path / 'valid.tar.gz'
    malformed = tmp_path / 'split-roots.tar.gz'
    _write_content_sdist(valid)
    entries = []
    for index, (name, data) in enumerate(_read_sdist_entries(valid)):
        relative = name.split('/', 1)[1]
        root = 'root-A' if index % 2 else 'root-B'
        entries.append((f'{root}/{relative}', data))
    _write_sdist_entries(malformed, entries)

    with pytest.raises(DistCheckError, match='exactly one top-level root'):
        check_sdist(malformed)


def test_wheel_content_check_rejects_drive_relative_member(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / 'pyvoro2-test.whl'
    _write_content_wheel(
        wheel,
        extra_members=('C:evil/payload.txt',),
    )

    with pytest.raises(DistCheckError, match='unsafe archive member name'):
        check_wheel(wheel)


def test_sdist_content_check_rejects_drive_relative_root(
    tmp_path: Path,
) -> None:
    valid = tmp_path / 'valid.tar.gz'
    malformed = tmp_path / 'drive-relative.tar.gz'
    _write_content_sdist(valid)
    entries = [
        (f'C:root/{name.split("/", 1)[1]}', data)
        for name, data in _read_sdist_entries(valid)
    ]
    _write_sdist_entries(malformed, entries)

    with pytest.raises(DistCheckError, match='unsafe archive member name'):
        check_sdist(malformed)


@pytest.mark.parametrize(
    ('artifact_kind', 'original', 'alias'),
    [
        ('wheel', 'pyvoro2/__init__.py', '/pyvoro2/__init__.py'),
        ('wheel', 'pyvoro2/__init__.py', 'pyvoro2\\__init__.py'),
        (
            'wheel',
            'pyvoro2/__init__.py',
            'pyvoro2/__init__.py\x00ignored',
        ),
        ('wheel', 'pyvoro2/__init__.py', 'pyvoro2/./__init__.py'),
        ('wheel', 'pyvoro2/__init__.py', 'evil/../pyvoro2/__init__.py'),
        (
            'wheel',
            TEST_WHEEL_COPYING,
            f'{TEST_DIST_INFO_ROOT}/licenses/./COPYING',
        ),
        (
            'sdist',
            'pyvoro2-0.8.0.dev0/README.md',
            '/pyvoro2-0.8.0.dev0/README.md',
        ),
        (
            'sdist',
            'pyvoro2-0.8.0.dev0/README.md',
            'pyvoro2-0.8.0.dev0\\README.md',
        ),
        (
            'sdist',
            'pyvoro2-0.8.0.dev0/README.md',
            'pyvoro2-0.8.0.dev0/./README.md',
        ),
        (
            'sdist',
            'pyvoro2-0.8.0.dev0/README.md',
            'evil/../pyvoro2-0.8.0.dev0/README.md',
        ),
        (
            'sdist',
            'pyvoro2-0.8.0.dev0/COPYING',
            'pyvoro2-0.8.0.dev0/./COPYING',
        ),
    ],
)
def test_distribution_content_checks_reject_unsafe_required_path_aliases(
    tmp_path: Path,
    artifact_kind: str,
    original: str,
    alias: str,
) -> None:
    if artifact_kind == 'wheel':
        valid = tmp_path / 'valid.whl'
        malformed = tmp_path / 'unsafe.whl'
        _write_content_wheel(valid)
        entries = _read_wheel_entries(valid)
        writer = _write_wheel_entries
        checker = check_wheel
    else:
        valid = tmp_path / 'valid.tar.gz'
        malformed = tmp_path / 'unsafe.tar.gz'
        _write_content_sdist(valid)
        entries = _read_sdist_entries(valid)
        writer = _write_sdist_entries
        checker = check_sdist
    writer(
        malformed,
        [(alias if name == original else name, data) for name, data in entries],
    )

    with pytest.raises(DistCheckError, match='unsafe archive member name'):
        checker(malformed)


def test_wheel_content_check_rejects_raw_normalized_name_collision(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / 'normalized-collision.whl'
    _write_content_wheel(
        wheel,
        extra_members=('pyvoro2\\__init__.py',),
    )

    with pytest.raises(DistCheckError, match='unsafe archive member name'):
        check_wheel(wheel)


def test_wheel_content_check_requires_one_dist_info_root(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / 'pyvoro2-test.whl'
    _write_content_wheel(
        wheel,
        extra_members=('other-0.dist-info/METADATA',),
    )

    with pytest.raises(DistCheckError, match='exactly one .dist-info root'):
        check_wheel(wheel)


def test_distribution_content_checks_require_metadata_tool() -> None:
    assert 'tools/check_dist_metadata.py' in dist_tool.REQUIRED_SDIST_FILES


def test_distribution_content_checks_accept_complete_license_payload(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / 'pyvoro2-test.whl'
    sdist = tmp_path / 'pyvoro2-test.tar.gz'
    _write_content_wheel(wheel)
    _write_content_sdist(sdist)

    check_wheel(wheel)
    check_sdist(sdist)


@pytest.mark.parametrize(
    'suffix',
    [
        TEST_WHEEL_LICENSE,
        TEST_WHEEL_COPYING,
        TEST_WHEEL_NOTICE,
        TEST_WHEEL_VORO_LICENSE,
    ],
)
def test_wheel_content_check_requires_complete_license_payload(
    tmp_path: Path,
    suffix: str,
) -> None:
    wheel = tmp_path / 'pyvoro2-test.whl'
    _write_content_wheel(wheel, omitted_members=(suffix,))

    with pytest.raises(DistCheckError, match='missing required members'):
        check_wheel(wheel)


@pytest.mark.parametrize(
    'member',
    [
        'LICENSE',
        'COPYING',
        'NOTICE.md',
        'LICENSE.voro++',
        'vendor/voro++/LICENSE',
    ],
)
def test_sdist_content_check_requires_complete_license_payload(
    tmp_path: Path,
    member: str,
) -> None:
    sdist = tmp_path / 'pyvoro2-test.tar.gz'
    _write_content_sdist(sdist, omitted_members=(member,))

    with pytest.raises(DistCheckError, match='missing required members'):
        check_sdist(sdist)


@pytest.mark.parametrize(
    ('artifact_kind', 'member'),
    [
        ('wheel', TEST_WHEEL_LICENSE),
        ('wheel', TEST_WHEEL_COPYING),
        ('wheel', TEST_WHEEL_NOTICE),
        ('wheel', TEST_WHEEL_VORO_LICENSE),
        ('sdist', 'LICENSE'),
        ('sdist', 'COPYING'),
        ('sdist', 'NOTICE.md'),
        ('sdist', 'LICENSE.voro++'),
        ('sdist', 'vendor/voro++/LICENSE'),
    ],
)
def test_distribution_content_checks_reject_mutated_license_payload(
    tmp_path: Path,
    artifact_kind: str,
    member: str,
) -> None:
    artifact = tmp_path / (
        'pyvoro2-test.whl'
        if artifact_kind == 'wheel'
        else 'pyvoro2-test.tar.gz'
    )
    mutation = (
        NOTICE_BYTES + b'\nmutated\n'
        if member.endswith('NOTICE.md')
        else b'mutated'
    )
    if artifact_kind == 'wheel':
        _write_content_wheel(artifact, member_data={member: mutation})
        checker = check_wheel
    else:
        _write_content_sdist(artifact, member_data={member: mutation})
        checker = check_sdist

    with pytest.raises(DistCheckError, match='does not match'):
        checker(artifact)


@pytest.mark.parametrize('artifact_kind', ['wheel', 'sdist'])
def test_distribution_content_checks_reject_stale_notice_target(
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    artifact = tmp_path / (
        'pyvoro2-test.whl'
        if artifact_kind == 'wheel'
        else 'pyvoro2-test.tar.gz'
    )
    if artifact_kind == 'wheel':
        _write_content_wheel(
            artifact,
            member_data={TEST_WHEEL_NOTICE: b'vendored path only'},
        )
        checker = check_wheel
    else:
        _write_content_sdist(
            artifact,
            member_data={'NOTICE.md': b'vendored path only'},
        )
        checker = check_sdist

    with pytest.raises(DistCheckError, match='LICENSE.voro'):
        checker(artifact)


def test_distribution_content_checks_reject_os_independent_metadata(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / 'pyvoro2-test.whl'
    sdist = tmp_path / 'pyvoro2-test.tar.gz'
    _write_content_wheel(
        wheel,
        member_data={
            TEST_WHEEL_METADATA: (
                b'Classifier: Operating System :: OS Independent\n'
            )
        },
    )
    _write_content_sdist(
        sdist,
        member_data={
            'PKG-INFO': b'Classifier: Operating System :: OS Independent\n'
        },
    )

    with pytest.raises(DistCheckError, match='OS Independent'):
        check_wheel(wheel)
    with pytest.raises(DistCheckError, match='OS Independent'):
        check_sdist(sdist)


def test_r9_source_metadata_and_api_inventory_are_current() -> None:
    pyproject = (REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    conftest = (REPO_ROOT / 'tests' / 'conftest.py').read_text(encoding='utf-8')
    assert 'Operating System :: OS Independent' not in pyproject
    assert (
        'fuzz: seeded randomized fuzz/property tests included in the default suite'
        in pyproject
    )
    assert 'default=10' in conftest
    assert 'pytest_collection_modifyitems' not in conftest

    inventory = (
        REPO_ROOT / 'docs' / 'development' / 'api-inventory.md'
    ).read_text(encoding='utf-8')
    current = inventory.split('## Current v0.8 contract', 1)[1]
    assert 'There is no current `pyvoro2.powerfit`' in current
    for retained_name in (
        'PowerFitBounds',
        'PowerFitPredictions',
        'PowerFitObjectiveBreakdown',
        'SelfConsistentPowerFitResult',
    ):
        assert retained_name in current
    for report_kind in (
        'power_weight_fit',
        'realized_pair_diagnostics',
        'self_consistent_power_fit',
    ):
        assert report_kind in current


def test_distribution_content_checks_reject_obsolete_private_paths(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / 'pyvoro2-test.whl'
    sdist = tmp_path / 'pyvoro2-test.tar.gz'
    _write_content_wheel(
        wheel,
        extra_members=('pyvoro2/_weight_transforms.py',),
    )
    _write_content_sdist(
        sdist,
        extra_members=('src/pyvoro2/planar/_domain_geometry.py',),
    )

    with pytest.raises(DistCheckError, match='_weight_transforms'):
        check_wheel(wheel)
    with pytest.raises(DistCheckError, match='_domain_geometry'):
        check_sdist(sdist)


def test_overlay_verification_imports_extensions_explicitly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_src = tmp_path / 'src'
    package_dir = repo_src / 'pyvoro2'
    py_file = package_dir / '__init__.py'
    core_file = package_dir / '_core.test.so'
    core2d_file = package_dir / '_core2d.test.so'
    observed: dict[str, str] = {}

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert capture_output is True
        assert text is True
        observed['code'] = command[-1]
        stdout = f'{py_file}\n{core_file}\n{core2d_file}\n'
        return subprocess.CompletedProcess(command, 0, stdout=stdout)

    monkeypatch.setattr(overlay_tool.subprocess, 'run', fake_run)

    assert overlay_tool._verify_overlay(repo_src) == (
        str(py_file),
        str(core_file),
        str(core2d_file),
    )
    assert "import_module('pyvoro2._core')" in observed['code']
    assert "import_module('pyvoro2._core2d')" in observed['code']
    assert 'api._core.__file__' not in observed['code']
    assert 'api2._core2d' not in observed['code']


def _mock_installed_license_distribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    omitted_members: tuple[str, ...] = (),
    member_data: dict[str, bytes] | None = None,
) -> None:
    site_packages = tmp_path / 'site-packages'
    overrides = {} if member_data is None else member_data
    defaults = {
        'LICENSE': PROJECT_LICENSE_BYTES,
        'COPYING': COPYING_BYTES,
        'NOTICE.md': NOTICE_BYTES,
        'LICENSE.voro++': VORO_LICENSE_BYTES,
    }
    files: list[str] = []
    for filename in installed_package_tool.REQUIRED_LICENSE_FILES:
        if filename in omitted_members:
            continue
        relative = f'{TEST_DIST_INFO_ROOT}/licenses/{filename}'
        path = site_packages / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(overrides.get(filename, defaults[filename]))
        files.append(relative)

    distribution = SimpleNamespace(
        files=tuple(files),
        metadata=Message(),
        locate_file=lambda member: site_packages / str(member),
    )
    monkeypatch.setattr(
        installed_package_tool.importlib_metadata,
        'distribution',
        lambda name: distribution,
    )


def test_installed_license_contract_matches_artifact_contract() -> None:
    assert (
        installed_package_tool.REQUIRED_LICENSE_FILES
        == dist_tool.REQUIRED_LICENSE_FILES
    )


def test_installed_distribution_accepts_complete_license_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_installed_license_distribution(monkeypatch, tmp_path)

    installed_package_tool._check_distribution_metadata(REPO_ROOT)


@pytest.mark.parametrize('failure', ['missing', 'mutated'])
def test_installed_distribution_rejects_invalid_copying(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    omitted = ('COPYING',) if failure == 'missing' else ()
    overrides = {'COPYING': b'mutated'} if failure == 'mutated' else None
    _mock_installed_license_distribution(
        monkeypatch,
        tmp_path,
        omitted_members=omitted,
        member_data=overrides,
    )

    with pytest.raises(InstalledPackageCheckError, match='COPYING'):
        installed_package_tool._check_distribution_metadata(REPO_ROOT)


def test_installed_provenance_rejects_repository_import(tmp_path: Path) -> None:
    repository = tmp_path / 'checkout'
    package_file = repository / 'src' / 'pyvoro2' / '__init__.py'
    package_file.parent.mkdir(parents=True)
    package_file.touch()
    module = ModuleType('pyvoro2')
    module.__file__ = str(package_file)

    with pytest.raises(
        InstalledPackageCheckError,
        match='imported from the repository checkout',
    ):
        assert_outside_repository(module, 'pyvoro2', repository)


def test_installed_provenance_accepts_external_import(tmp_path: Path) -> None:
    repository = tmp_path / 'checkout'
    package_file = tmp_path / 'environment' / 'pyvoro2' / '__init__.py'
    package_file.parent.mkdir(parents=True)
    package_file.touch()
    module = ModuleType('pyvoro2')
    module.__file__ = str(package_file)

    assert assert_outside_repository(
        module,
        'pyvoro2',
        repository,
    ) == package_file.resolve()


def test_select_only_sdist_requires_one_artifact(tmp_path: Path) -> None:
    with pytest.raises(SdistWheelBuildError, match='found 0'):
        select_only_sdist(tmp_path)

    first = tmp_path / 'pyvoro2-1.tar.gz'
    second = tmp_path / 'pyvoro2-2.tar.gz'
    first.touch()
    assert select_only_sdist(tmp_path) == first

    second.touch()
    with pytest.raises(SdistWheelBuildError, match='found 2'):
        select_only_sdist(tmp_path)


def test_distribution_metadata_artifacts_are_filtered_and_sorted(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / 'pyvoro2-2.whl'
    sdist = tmp_path / 'pyvoro2-1.tar.gz'
    ignored = tmp_path / 'checksums.txt'
    wheel.touch()
    sdist.touch()
    ignored.touch()

    assert distribution_artifacts(tmp_path) == (
        sdist.resolve(),
        wheel.resolve(),
    )


def test_distribution_artifact_selection_canonicalizes_equivalent_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel = tmp_path / 'pyvoro2-2.whl'
    sdist = tmp_path / 'pyvoro2-1.tar.gz'
    wheel.touch()
    sdist.touch()
    monkeypatch.chdir(tmp_path.parent)
    relative_dir = Path(tmp_path.name)
    relative_wheel = relative_dir / wheel.name
    inputs = [relative_dir, relative_wheel, wheel.resolve()]
    expected = (sdist.resolve(), wheel.resolve())

    assert select_content_artifacts(inputs) == expected
    assert select_distribution_artifacts(inputs) == expected


def test_twine_artifact_selection_is_unique_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel = tmp_path / 'pyvoro2-2.whl'
    sdist = tmp_path / 'pyvoro2-1.tar.gz'
    wheel.touch()
    sdist.touch()
    monkeypatch.chdir(tmp_path.parent)
    relative_dir = Path(tmp_path.name)
    selected = select_distribution_artifacts(
        [relative_dir, relative_dir / wheel.name, wheel.resolve()]
    )
    observed: dict[str, object] = {}

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        observed['command'] = command
        observed['cwd'] = cwd
        observed['check'] = check
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(dist_metadata_tool.subprocess, 'run', fake_run)

    assert run_twine_check_artifacts(
        (*selected, wheel.resolve()),
        python_executable='python-for-test',
    ) == (sdist.resolve(), wheel.resolve())
    assert observed == {
        'command': [
            'python-for-test',
            '-m',
            'twine',
            'check',
            str(sdist.resolve()),
            str(wheel.resolve()),
        ],
        'cwd': REPO_ROOT,
        'check': True,
    }


def test_distribution_metadata_requires_artifacts(tmp_path: Path) -> None:
    with pytest.raises(
        DistributionMetadataCheckError,
        match=r'no \.whl or \.tar\.gz distributions found',
    ):
        distribution_artifacts(tmp_path)


def test_twine_check_uses_explicit_shell_independent_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dist_dir = tmp_path / 'release dist'
    dist_dir.mkdir()
    wheel = dist_dir / 'pyvoro2-2.whl'
    sdist = dist_dir / 'pyvoro2-1.tar.gz'
    wheel.touch()
    sdist.touch()
    observed: dict[str, object] = {}

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        observed['command'] = command
        observed['cwd'] = cwd
        observed['check'] = check
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(dist_metadata_tool.subprocess, 'run', fake_run)

    assert run_twine_check(
        dist_dir,
        python_executable='python-for-test',
    ) == (sdist, wheel)
    assert observed == {
        'command': [
            'python-for-test',
            '-m',
            'twine',
            'check',
            str(sdist),
            str(wheel),
        ],
        'cwd': REPO_ROOT,
        'check': True,
    }


def test_sdist_wheel_build_rejects_existing_wheel(tmp_path: Path) -> None:
    (tmp_path / 'pyvoro2-1.tar.gz').touch()
    (tmp_path / 'direct-checkout-build.whl').touch()

    with pytest.raises(
        SdistWheelBuildError,
        match='output directory contains existing wheel artifacts',
    ):
        build_wheel_from_sdist(tmp_path, tmp_path)


def _fake_runtime_metadata(
    *,
    requires_python: str | None,
    runtime_requirements: tuple[str, ...],
    optional_requirements: tuple[str, ...],
) -> str:
    fields: list[str] = []
    if requires_python is not None:
        fields.append(f'Requires-Python: {requires_python}\n')
    fields.extend(
        f'Requires-Dist: {requirement}\n'
        for requirement in runtime_requirements + optional_requirements
    )
    return ''.join(fields)


def _write_fake_wheel(
    directory: Path,
    python_tag: str,
    platform_tag: str,
    *,
    abi_tag: str | None = None,
    metadata_name: str = 'pyvoro2',
    metadata_version: str = WHEEL_MATRIX_VERSION,
    requires_python: str | None = '>=3.10',
    runtime_requirements: tuple[str, ...] = WHEEL_MATRIX_RUNTIME_REQUIREMENTS,
    optional_requirements: tuple[str, ...] = WHEEL_MATRIX_OPTIONAL_REQUIREMENTS,
    include_core: bool = True,
    include_core2d: bool = True,
    extra_native_members: tuple[str, ...] = (),
) -> Path:
    abi = python_tag if abi_tag is None else abi_tag
    filename = (
        f'pyvoro2-{WHEEL_MATRIX_VERSION}-{python_tag}-{abi}-'
        f'{platform_tag}.whl'
    )
    path = directory / filename
    dist_info = f'pyvoro2-{WHEEL_MATRIX_VERSION}.dist-info'
    wheel_tags = ''.join(
        f'Tag: {python_tag}-{abi}-{tag}\n'
        for tag in platform_tag.split('.')
    )
    runtime_metadata = _fake_runtime_metadata(
        requires_python=requires_python,
        runtime_requirements=runtime_requirements,
        optional_requirements=optional_requirements,
    )

    with zipfile.ZipFile(path, 'w') as zf:
        _writestr_exact(
            zf,
            f'{dist_info}/METADATA',
            (
                'Metadata-Version: 2.2\n'
                f'Name: {metadata_name}\n'
                f'Version: {metadata_version}\n'
                f'{runtime_metadata}'
            ),
        )
        _writestr_exact(
            zf,
            f'{dist_info}/WHEEL',
            (
                'Wheel-Version: 1.0\n'
                'Generator: pyvoro2-test\n'
                'Root-Is-Purelib: false\n'
                f'{wheel_tags}'
            ),
        )
        if include_core:
            _writestr_exact(
                zf,
                'pyvoro2/_core.test.so',
                b'native-core',
            )
        if include_core2d:
            _writestr_exact(
                zf,
                'pyvoro2/_core2d.test.so',
                b'native-core2d',
            )
        for member in extra_native_members:
            _writestr_exact(zf, member, b'extra-native')
    return path


def _write_fake_sdist(
    directory: Path,
    *,
    metadata_name: str = 'pyvoro2',
    metadata_version: str = WHEEL_MATRIX_VERSION,
    requires_python: str | None = '>=3.10',
    runtime_requirements: tuple[str, ...] = WHEEL_MATRIX_RUNTIME_REQUIREMENTS,
    optional_requirements: tuple[str, ...] = WHEEL_MATRIX_OPTIONAL_REQUIREMENTS,
) -> Path:
    path = directory / f'pyvoro2-{WHEEL_MATRIX_VERSION}.tar.gz'
    runtime_metadata = _fake_runtime_metadata(
        requires_python=requires_python,
        runtime_requirements=runtime_requirements,
        optional_requirements=optional_requirements,
    )
    pkg_info = (
        'Metadata-Version: 2.2\n'
        f'Name: {metadata_name}\n'
        f'Version: {metadata_version}\n'
        f'{runtime_metadata}'
    ).encode()
    member = tarfile.TarInfo(
        f'pyvoro2-{WHEEL_MATRIX_VERSION}/PKG-INFO'
    )
    member.size = len(pkg_info)
    with tarfile.open(path, 'w:gz') as tf:
        tf.addfile(member, io.BytesIO(pkg_info))
    return path


def _write_complete_wheel_matrix(directory: Path) -> None:
    for python_tag in wheel_matrix_tool.EXPECTED_PYTHON_TAGS:
        for platform_tag in WHEEL_MATRIX_PLATFORM_TAGS:
            _write_fake_wheel(directory, python_tag, platform_tag)
    _write_fake_sdist(directory)


def test_wheel_matrix_accepts_complete_release_set(tmp_path: Path) -> None:
    _write_complete_wheel_matrix(tmp_path)

    summary = validate_wheel_matrix(tmp_path)

    assert summary.project_name == 'pyvoro2'
    assert summary.version == WHEEL_MATRIX_VERSION
    assert summary.wheel_count == 20
    assert summary.sdist_count == 1


def test_wheel_matrix_rejects_duplicate_contract(tmp_path: Path) -> None:
    _write_complete_wheel_matrix(tmp_path)
    missing = (
        tmp_path
        / (
            f'pyvoro2-{WHEEL_MATRIX_VERSION}-cp314-cp314-'
            f'{WHEEL_MATRIX_PLATFORM_TAGS[0]}.whl'
        )
    )
    missing.unlink()
    _write_fake_wheel(tmp_path, 'cp310', 'manylinux2014_x86_64')

    with pytest.raises(WheelMatrixError, match='duplicate wheel contract'):
        validate_wheel_matrix(tmp_path)


def test_wheel_matrix_rejects_free_threaded_python(tmp_path: Path) -> None:
    _write_complete_wheel_matrix(tmp_path)
    replaced = (
        tmp_path
        / (
            f'pyvoro2-{WHEEL_MATRIX_VERSION}-cp314-cp314-'
            f'{WHEEL_MATRIX_PLATFORM_TAGS[0]}.whl'
        )
    )
    replaced.unlink()
    _write_fake_wheel(tmp_path, 'cp314t', WHEEL_MATRIX_PLATFORM_TAGS[0])

    with pytest.raises(WheelMatrixError, match='unsupported interpreter tag'):
        validate_wheel_matrix(tmp_path)


@pytest.mark.parametrize(
    'python_tag',
    ['cp313t', 'pp310', 'graalpy310', 'cp39', 'cp315'],
)
def test_wheel_filename_rejects_unsupported_interpreters(
    python_tag: str,
) -> None:
    path = Path(
        f'pyvoro2-1.0-{python_tag}-{python_tag}-win_amd64.whl'
    )

    with pytest.raises(WheelMatrixError, match='unsupported interpreter tag'):
        parse_wheel_filename(path)


@pytest.mark.parametrize(
    'platform_tag',
    [
        'musllinux_1_2_x86_64',
        'manylinux_2_17_i686',
        'manylinux_2_17_aarch64',
        'win32',
        'win_arm64',
        'macosx_11_0_universal2',
        'linux_x86_64',
    ],
)
def test_wheel_platform_rejects_unsupported_variants(
    platform_tag: str,
) -> None:
    with pytest.raises(WheelMatrixError, match='unsupported platform tag'):
        classify_platform_tag(platform_tag)


def test_wheel_matrix_rejects_inconsistent_metadata(tmp_path: Path) -> None:
    _write_complete_wheel_matrix(tmp_path)
    _write_fake_wheel(
        tmp_path,
        'cp310',
        'win_amd64',
        metadata_version='0.8.0.dev1',
    )

    with pytest.raises(WheelMatrixError, match='has version'):
        validate_wheel_matrix(tmp_path)


@pytest.mark.parametrize(
    (
        'requires_python',
        'runtime_requirements',
        'error_match',
    ),
    [
        (
            None,
            WHEEL_MATRIX_RUNTIME_REQUIREMENTS,
            'Requires-Python',
        ),
        (
            '>=3.11',
            WHEEL_MATRIX_RUNTIME_REQUIREMENTS,
            'Requires-Python',
        ),
        (
            '>=3.10',
            WHEEL_MATRIX_RUNTIME_REQUIREMENTS[:1],
            'runtime Requires-Dist',
        ),
        (
            '>=3.10',
            WHEEL_MATRIX_RUNTIME_REQUIREMENTS + ('scipy>=1.8',),
            'runtime Requires-Dist',
        ),
    ],
)
def test_wheel_matrix_rejects_invalid_runtime_metadata(
    tmp_path: Path,
    requires_python: str | None,
    runtime_requirements: tuple[str, ...],
    error_match: str,
) -> None:
    _write_complete_wheel_matrix(tmp_path)
    _write_fake_wheel(
        tmp_path,
        'cp310',
        'win_amd64',
        requires_python=requires_python,
        runtime_requirements=runtime_requirements,
    )

    with pytest.raises(WheelMatrixError, match=error_match):
        validate_wheel_matrix(tmp_path)


def test_wheel_matrix_accepts_equivalent_specifier_order(
    tmp_path: Path,
) -> None:
    _write_complete_wheel_matrix(tmp_path)
    _write_fake_wheel(
        tmp_path,
        'cp310',
        'win_amd64',
        runtime_requirements=(
            'numpy>=1.23,<2; python_version < "3.11"',
            'numpy>=1.23,<3; python_version >= "3.11"',
        ),
    )

    validate_wheel_matrix(tmp_path)


def test_wheel_matrix_requires_both_native_modules(tmp_path: Path) -> None:
    _write_complete_wheel_matrix(tmp_path)
    _write_fake_wheel(
        tmp_path,
        'cp310',
        'win_amd64',
        include_core2d=False,
    )

    with pytest.raises(WheelMatrixError, match=r'pyvoro2/_core2d'):
        validate_wheel_matrix(tmp_path)


@pytest.mark.parametrize('module_name', ['_core', '_core2d'])
@pytest.mark.parametrize(
    'member_template',
    [
        'pyvoro2/{module}.test.so/',
        'pyvoro2/./{module}.test.so',
        'pyvoro2//{module}.test.so',
    ],
)
def test_wheel_matrix_rejects_non_file_or_aliased_native_members(
    tmp_path: Path,
    module_name: str,
    member_template: str,
) -> None:
    _write_complete_wheel_matrix(tmp_path)
    _write_fake_wheel(
        tmp_path,
        'cp310',
        'win_amd64',
        include_core=module_name != '_core',
        include_core2d=module_name != '_core2d',
        extra_native_members=(
            member_template.format(module=module_name),
        ),
    )

    with pytest.raises(
        WheelMatrixError,
        match=rf'pyvoro2/{module_name} native module, found 0',
    ):
        validate_wheel_matrix(tmp_path)


@pytest.mark.parametrize(
    ('module_name', 'malformed_member'),
    [
        ('_core', 'pyvoro2\\_core.test.so'),
        ('_core2d', 'pyvoro2\\_core2d.test.pyd'),
    ],
)
def test_wheel_matrix_rejects_backslash_native_aliases(
    tmp_path: Path,
    module_name: str,
    malformed_member: str,
) -> None:
    _write_complete_wheel_matrix(tmp_path)
    _write_fake_wheel(
        tmp_path,
        'cp310',
        'win_amd64',
        include_core=module_name != '_core',
        include_core2d=module_name != '_core2d',
        extra_native_members=(malformed_member,),
    )

    with pytest.raises(
        WheelMatrixError,
        match=rf'pyvoro2/{module_name} native module, found 0',
    ):
        validate_wheel_matrix(tmp_path)


@pytest.mark.parametrize(
    ('module_name', 'extra_member'),
    [
        ('_core', 'pyvoro2/_core.extra.so'),
        ('_core2d', 'pyvoro2/_core2d.extra.so'),
    ],
)
def test_wheel_matrix_rejects_ambiguous_native_modules(
    tmp_path: Path,
    module_name: str,
    extra_member: str,
) -> None:
    _write_complete_wheel_matrix(tmp_path)
    _write_fake_wheel(
        tmp_path,
        'cp310',
        'win_amd64',
        extra_native_members=(extra_member,),
    )

    with pytest.raises(
        WheelMatrixError,
        match=rf'exactly one pyvoro2/{module_name} native module, found 2',
    ):
        validate_wheel_matrix(tmp_path)


def test_wheel_matrix_rejects_inconsistent_sdist_metadata(
    tmp_path: Path,
) -> None:
    _write_complete_wheel_matrix(tmp_path)
    _write_fake_sdist(tmp_path, metadata_version='0.8.0.dev1')

    with pytest.raises(WheelMatrixError, match='has version'):
        validate_wheel_matrix(tmp_path)


def test_wheel_matrix_rejects_inconsistent_sdist_runtime_metadata(
    tmp_path: Path,
) -> None:
    _write_complete_wheel_matrix(tmp_path)
    _write_fake_sdist(
        tmp_path,
        runtime_requirements=WHEEL_MATRIX_RUNTIME_REQUIREMENTS[:1],
    )

    with pytest.raises(WheelMatrixError, match='runtime Requires-Dist'):
        validate_wheel_matrix(tmp_path)
