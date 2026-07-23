from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tarfile
from types import ModuleType
import zipfile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


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

installed_package_tool = _load_tool_module('check_installed_package')
InstalledPackageCheckError = installed_package_tool.InstalledPackageCheckError
assert_outside_repository = installed_package_tool.assert_outside_repository

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


def test_check_installed_package_help() -> None:
    assert 'representative public workflows' in _run_help(
        'check_installed_package.py'
    )


def test_check_wheel_matrix_help() -> None:
    assert 'merged wheels and sdist' in _run_help('check_wheel_matrix.py')


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
        zf.writestr(
            f'{dist_info}/METADATA',
            (
                'Metadata-Version: 2.2\n'
                f'Name: {metadata_name}\n'
                f'Version: {metadata_version}\n'
                f'{runtime_metadata}'
            ),
        )
        zf.writestr(
            f'{dist_info}/WHEEL',
            (
                'Wheel-Version: 1.0\n'
                'Generator: pyvoro2-test\n'
                'Root-Is-Purelib: false\n'
                f'{wheel_tags}'
            ),
        )
        if include_core:
            zf.writestr('pyvoro2/_core.test.so', b'native-core')
        if include_core2d:
            zf.writestr('pyvoro2/_core2d.test.so', b'native-core2d')
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
