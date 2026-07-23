from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType

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
