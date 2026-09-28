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


@pytest.mark.parametrize('qualified', [False, True])
def test_safety_wheel_preserves_unqualified_bytes_and_refuses_issued_input(
        driver, tmp_path, qualified):
    wheel = tmp_path / 'pyvoro2-0-cp312-cp312-linux_x86_64.whl'
    anchor_path = ROOT / 'src/pyvoro2/_internal/_qualification_installation.py'
    anchor = anchor_path.read_bytes()
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('pyvoro2/_core.so', b'instrumented native fixture')
        archive.writestr('pyvoro2/_internal/_qualification_installation.py', anchor)
        if qualified:
            archive.writestr('pyvoro2/_internal/native_qualification_record.json', '{}')
    stage = driver.extract_wheel(wheel, tmp_path / 'stage')
    output = tmp_path / 'sanitizer-wheel' / wheel.name
    if qualified:
        with pytest.raises(driver.FinalizationError, match='unqualified'):
            driver.copy_safety_wheel(ROOT, wheel, stage, output)
        assert not output.exists()
    else:
        driver.copy_safety_wheel(ROOT, wheel, stage, output)
        assert output.read_bytes() == wheel.read_bytes()
        with zipfile.ZipFile(output) as archive:
            assert archive.read(
                'pyvoro2/_internal/_qualification_installation.py') == anchor
            assert not any('native_qualification_record.json' in name
                           for name in archive.namelist())


@pytest.mark.parametrize('change_wheel', [False, True])
def test_safety_distribution_binds_the_wheel_that_was_installed(
        driver, tmp_path, monkeypatch, change_wheel):
    # Native execution is exercised by the controlled suite integration. Here
    # emulate its external process boundaries while retaining real file hashes,
    # installation copies, anchor checks and distribution manifest generation.
    stage = tmp_path / 'stage'
    internal = stage / 'pyvoro2/_internal'
    internal.mkdir(parents=True)
    shutil.copyfile(ROOT / 'src/pyvoro2/_internal/_qualification_installation.py',
                    internal / '_qualification_installation.py')
    for name in ('_core', '_core2d', '_fpguard'):
        (stage / 'pyvoro2' / (name + '.so')).write_bytes(name.encode())
    wheel = tmp_path / 'input.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for path in stage.rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(stage))

    def imports(root, _output, _environment=None):
        return {name: {**driver.file_identity(path), 'identity': {}, 'loaded': []}
                for path in (root / 'pyvoro2').glob('*.so')
                for name in ['pyvoro2.' + path.stem]}, []

    def execute(command, **kwargs):
        if 'pip' in command:
            shutil.copytree(stage, command[command.index('--target') + 1])
        return {'exit_code': 0}

    def safety(**kwargs):
        kwargs['output'].mkdir()
        (kwargs['output'] / 'sanitizer-safety-evidence.json').write_text('{}')
        if change_wheel:
            (tmp_path / 'sanitizer-wheel/input.whl').write_bytes(b'untested payload')

    monkeypatch.setattr(driver, 'run_process', execute)
    monkeypatch.setattr(driver, 'inspect_installation', imports)
    monkeypatch.setattr(driver, 'exercise_sanitizer_safety', safety)
    arguments = dict(
        source_root=ROOT, output=tmp_path, measurement=driver.measure_source(ROOT),
        contract=driver._contract(ROOT), required={'wp5-spatial'}, direct=wheel,
        repaired=wheel, stage=stage, imports=imports(stage, None)[0],
        records=tmp_path / 'records', postprocess_path=tmp_path / 'postprocess.json',
        candidate_module=None, candidate_records=None, runtime_environment={},
        runtime_libraries=[], processes=[])
    if change_wheel:
        with pytest.raises(driver.DriverError, match='wheel.*changed'):
            driver._finish_sanitizer_build(**arguments)
        assert not (tmp_path / 'distribution-evidence.json').exists()
    else:
        manifest = driver._finish_sanitizer_build(**arguments)
        assert manifest['schema'] == 'pyvoro2-native-sanitizer-distribution-evidence-v1'
        assert 'qualification_record_sha256' not in manifest
        issued = Path(manifest['sanitizer_wheel']['path'])
        assert issued.read_bytes() == wheel.read_bytes()


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


def test_failure_tail_drops_url_credentials_cut_by_the_read_boundary(driver, tmp_path):
    url = 'https://example-user:fixture-password-fragment@example.invalid/\n'
    cut = url.index('fixture-password-fragment') + 4
    ending = 'last diagnostic line\n'
    raw = url + 'x' * (12288 + cut - len(url) - len(ending)) + ending
    path = tmp_path / 'stderr'
    path.write_text(raw, encoding='utf8')
    tail = driver._failed_output_tail(path, {})
    assert 'password-fragment' not in tail
    assert tail.endswith(ending.rstrip('\n'))
    assert len(tail.encode('utf8')) <= 12288
    assert path.read_text(encoding='utf8') == raw


@pytest.mark.parametrize('cut_inside_context', [False, True])
def test_failure_tail_keeps_multiline_credentials_out_of_the_original_window(
        driver, tmp_path, cut_inside_context):
    first = 'prefix-credential\nfixture-secret-suffix'
    second = 'other-fixture-credential'
    cut = 4 + (len(first) if cut_inside_context else 0)
    ending = '\nlast diagnostic line\n'
    padding_size = 12288 + cut - len(first) - 1 - len(ending)
    padding = second * (padding_size // len(second))
    raw = first + '\n' + padding + 'x' * (padding_size % len(second)) + ending
    path = tmp_path / 'stderr'
    path.write_text(raw, encoding='utf8')
    tail = driver._failed_output_tail(path, {'ONE_TOKEN': first, 'TWO_TOKEN': second})
    assert 'fixture-secret-suffix' not in tail
    assert 'other-fixture-credential' not in tail
    assert tail.endswith('last diagnostic line')
    assert len(tail.encode('utf8')) <= 12288
    assert path.read_text(encoding='utf8') == raw


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
