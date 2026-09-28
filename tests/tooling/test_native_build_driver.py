"""Controlled distribution driver: explicit approval and exact wheel payloads."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def driver():
    path = ROOT / 'tools/native/qualification/build.py'
    spec = importlib.util.spec_from_file_location('isolated_build_driver', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wheel_extraction_refuses_path_escape(driver, tmp_path):
    wheel = tmp_path / 'bad.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('../outside', b'bad')
    with pytest.raises(driver.DriverError, match='path'):
        driver.extract_wheel(wheel, tmp_path / 'stage')
    assert not (tmp_path / 'outside').exists()


def test_issued_wheel_regenerates_record_without_changing_native_bytes(
        driver, tmp_path):
    wheel = tmp_path / 'pyvoro2-0-cp312-cp312-linux_x86_64.whl'
    original = {
        'pyvoro2/_core.so': b'native payload',
        'pyvoro2/_internal/_qualification_installation.py': b'RECORD_SHA256 = None\n',
        'pyvoro2-0.dist-info/METADATA': b'Name: pyvoro2\nVersion: 0\n',
        'pyvoro2-0.dist-info/RECORD': b'',
    }
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, data in original.items():
            archive.writestr(name, data)
    stage = tmp_path / 'stage'
    driver.extract_wheel(wheel, stage)
    internal = stage / 'pyvoro2/_internal'
    (internal / '_qualification_installation.py').write_bytes(
        b'RECORD_SHA256 = "issued"\n')
    (internal / 'native_qualification_record.json').write_bytes(b'{"issued":true}\n')
    cache = internal / '__pycache__'
    cache.mkdir()
    (cache / 'unused.pyc').write_bytes(b'not a wheel member')
    output = tmp_path / 'final' / wheel.name
    driver.repack_wheel(wheel, stage, output)
    with zipfile.ZipFile(output) as archive:
        assert archive.read('pyvoro2/_core.so') == original['pyvoro2/_core.so']
        assert not any('__pycache__' in name for name in archive.namelist())
        rows = list(csv.reader(io.StringIO(archive.read(
            'pyvoro2-0.dist-info/RECORD').decode('utf8'))))
        assert {row[0] for row in rows} == set(archive.namelist())
        for name, digest, size in rows:
            if name.endswith('.dist-info/RECORD'):
                assert digest == size == ''
            else:
                data = archive.read(name)
                expected = base64.urlsafe_b64encode(
                    hashlib.sha256(data).digest()).rstrip(b'=').decode('ascii')
                assert digest == 'sha256=' + expected
                assert size == str(len(data))


def test_process_receipt_contains_actual_exit_and_output_identity(driver, tmp_path):
    receipt = driver.run_process([sys.executable, '-c', 'print("observed")'],
                                 cwd=tmp_path, directory=tmp_path / 'process')
    assert receipt['exit_code'] == 0
    assert Path(receipt['stdout']['path']).read_text().strip() == 'observed'
    assert receipt['stdout']['sha256'] == hashlib.sha256(b'observed\n').hexdigest()
    expected = driver.file_identity(sys.executable)['sha256']
    assert receipt['executable']['sha256'] == expected


def test_process_preserves_interpreter_invocation_and_environment(driver, tmp_path):
    receipt = driver.run_process(
        [sys.executable, '-c', 'import sys; print(sys.prefix)'],
        cwd=tmp_path, directory=tmp_path / 'venv-process')
    assert receipt['argv'][0] == str(Path(sys.executable).absolute())
    assert Path(receipt['stdout']['path']).read_text().strip() == sys.prefix


def test_failed_process_reports_bounded_redacted_tails_and_retains_receipt(
        driver, tmp_path):
    environment = driver.build_environment()
    environment['FIXTURE_API_TOKEN'] = 'fixture-sensitive-token'
    script = (
        'import os, sys\n'
        'for index in range(120):\n'
        '    print("discarded stdout prefix " + str(index) + "x" * 256)\n'
        '    print("discarded stderr prefix " + str(index) + "y" * 256, '
        'file=sys.stderr)\n'
        'print("final stdout marker")\n'
        'print(os.environ["FIXTURE_API_TOKEN"], file=sys.stderr)\n'
        'print("https://fixture-user:fixture-password@example.invalid/", '
        'file=sys.stderr)\n'
        'print("ValueError: concrete failure detail", file=sys.stderr)\n'
        'raise SystemExit(7)\n')
    directory = tmp_path / 'failed-process'
    with pytest.raises(driver.DriverError) as failure:
        driver.run_process([sys.executable, '-c', script], cwd=tmp_path,
                           directory=directory, environment=environment)
    message = str(failure.value)
    assert 'process failed (7)' in message
    assert 'final stdout marker' in message
    assert 'ValueError: concrete failure detail' in message
    assert 'discarded stdout prefix 0x' not in message
    assert len(message.encode('utf8')) < 2 * 12288 + 2048
    assert len(message.splitlines()) <= 165
    assert 'fixture-sensitive-token' not in message
    assert 'fixture-user:fixture-password' not in message
    assert '[redacted]' in message
    receipt = json.loads((directory / 'process.json').read_text())
    assert receipt['exit_code'] == 7
    assert 'fixture-sensitive-token' in (directory / 'stderr').read_text()


def test_output_inside_source_is_rejected_before_build(driver, tmp_path):
    with pytest.raises(driver.DriverError, match='outside'):
        driver.build(source_root=tmp_path, output=tmp_path / 'inside', repair='none')


def test_sanitizer_runtime_is_excluded_from_build_environment(driver, monkeypatch):
    for key in ('LD_PRELOAD', 'ASAN_OPTIONS', 'UBSAN_OPTIONS',
                'PYVORO2_NATIVE_TEST_SANITIZERS'):
        monkeypatch.setenv(key, 'runtime-only-value')
    monkeypatch.setenv('CXXFLAGS', '-fno-fast-math')
    environment = driver.build_environment()
    assert environment['CXXFLAGS'] == '-fno-fast-math'
    assert not set(driver.SANITIZER_RUNTIME_KEYS) & environment.keys()


def test_inherited_python_and_pytest_controls_are_not_build_inputs(driver, monkeypatch):
    for key in ('PYTHONOPTIMIZE', 'PYTHONPATH', 'PYTHONHOME', 'PYTEST_ADDOPTS',
                'PYTEST_PLUGINS'):
        monkeypatch.setenv(key, 'uncontrolled')
    environment = driver.build_environment()
    assert 'PYTHONOPTIMIZE' not in environment
    assert 'PYTHONPATH' not in environment
    assert 'PYTHONHOME' not in environment
    assert 'PYTEST_ADDOPTS' not in environment
    assert 'PYTEST_PLUGINS' not in environment
    assert environment['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'
    assert environment['PYTHONNOUSERSITE'] == '1'
    assert environment['PYTHONDONTWRITEBYTECODE'] == '1'


def test_optimized_driver_refuses_before_source_approval(driver, tmp_path):
    process = subprocess.run(
        [sys.executable, '-O', str(ROOT / 'tools/native/qualification/build.py'),
         '--source-root', str(ROOT), '--output', str(tmp_path / 'build')],
        env=driver.build_environment(), text=True, capture_output=True, check=False)
    assert process.returncode != 0
    assert 'optimized Python' in process.stderr
    assert not (tmp_path / 'build').exists()


def test_controlled_wheel_build_selects_ninja_and_preserves_compiler_environment(
        driver, tmp_path, monkeypatch):
    # An independently measured fixture reaches the first external process.
    # No project source approval or native artifact is changed by this test.
    source = tmp_path / 'source'
    internal = source / 'src/pyvoro2/_internal'
    for name in ('vendor/voro++', 'cpp', 'cmake', 'src/pyvoro2/_internal'):
        (source / name).mkdir(parents=True)
    for name in ('CMakeLists.txt', 'pyproject.toml'):
        (source / name).write_text('# fixture\n', encoding='utf8')
    shutil.copyfile(ROOT / 'src/pyvoro2/_internal/native_qualification.py',
                    internal / 'native_qualification.py')
    measurement = driver.measure_source(source)
    approval = {'approval_schema': driver._contract(source).APPROVAL_SCHEMA,
                'approved': True,
                **{key: measurement[key] for key in (
                    'policy_revision', 'source_sha256', 'schema_sha256',
                    'consumer_sha256', 'components')}}
    (internal / 'native_approval.json').write_text(
        json.dumps(approval), encoding='utf8')
    inherited = {'CMAKE_GENERATOR': 'Visual Studio 18 2026',
                 'CMAKE_GENERATOR_PLATFORM': 'x64',
                 'CMAKE_GENERATOR_TOOLSET': 'v145',
                 'CMAKE_GENERATOR_INSTANCE': 'fixture-vs-install',
                 'CXX': 'fixture-cl.exe', 'INCLUDE': 'fixture-msvc-headers',
                 'LIB': 'fixture-msvc-libraries', 'LIBPATH': 'fixture-msvc-metadata'}
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)
    observed = {}

    class StopBeforeNativeBuild(Exception):
        pass

    def capture(command, **kwargs):
        observed.update(command=command, **kwargs)
        raise StopBeforeNativeBuild

    monkeypatch.setattr(driver, 'run_process', capture)
    with pytest.raises(StopBeforeNativeBuild):
        driver.build(source_root=source, output=tmp_path / 'output')
    assert '-Ccmake.args=-GNinja' in observed['command']
    assert '-Cinstall.strip=false' in observed['command']
    environment = observed['environment']
    assert environment['CMAKE_GENERATOR'] == 'Ninja'
    assert not {'CMAKE_GENERATOR_PLATFORM', 'CMAKE_GENERATOR_TOOLSET',
                'CMAKE_GENERATOR_INSTANCE'} & environment.keys()
    for name in ('CXX', 'INCLUDE', 'LIB', 'LIBPATH'):
        assert environment[name] == inherited[name]
