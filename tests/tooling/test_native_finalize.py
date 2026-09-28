"""The external issuer requires completed build, repair, and route evidence."""
from __future__ import annotations

import importlib.util
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def finalizer():
    path = ROOT / 'tools/native/qualification/finalize.py'
    spec = importlib.util.spec_from_file_location('isolated_finalizer', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_no_postprocess_receipt_means_no_issuance(finalizer, tmp_path):
    with pytest.raises(finalizer.FinalizationError, match='postprocess'):
        finalizer.verify_postprocess({}, {}, tmp_path)


def test_identity_postprocess_cannot_hide_changed_native_bytes(finalizer, tmp_path):
    original = tmp_path / 'build/_core.so'
    original.parent.mkdir()
    original.write_bytes(b'qualified linked native bytes')
    installed = tmp_path / 'install/pyvoro2/_core.so'
    installed.parent.mkdir(parents=True)
    installed.write_bytes(b'different repaired native bytes')
    before = finalizer.file_identity(original)
    after = finalizer.file_identity(installed)
    receipt = {
        'schema': 'pyvoro2-native-postprocess-v1',
        'modules': {'pyvoro2._core': {
            'input': before,
            'output': {**after, 'path': 'pyvoro2/_core.so'},
            'dependencies': [],
        }},
        'dependencies': {},
        'operations': [{'kind': 'identity',
                        'inputs': [before['sha256']],
                        'outputs': [after['sha256']]}],
    }
    build = {'components': {'_core': {'output': before}}}
    with pytest.raises(finalizer.FinalizationError, match='identity'):
        finalizer.verify_postprocess(receipt, build, tmp_path / 'install')


def test_boolean_route_success_is_not_complete_evidence(finalizer):
    with pytest.raises(finalizer.FinalizationError, match='route'):
        finalizer.validate_routes({'qualified': True}, {}, {}, {'wp5-spatial'})


def test_route_child_ignores_inherited_python_and_pytest_controls(
        finalizer, tmp_path, monkeypatch):
    runner = tmp_path / 'tools/native/qualification/route_suite.py'
    runner.parent.mkdir(parents=True)
    runner.write_text(
        'import json, os, pathlib, sys\n'
        'args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n'
        'report = {"optimize": sys.flags.optimize, "debug": __debug__,\n'
        '          "controls": {k: v for k, v in os.environ.items()\n'
        '                       if k.startswith(("PYTHON", "PYTEST_"))}}\n'
        'pathlib.Path(args["--output"]).write_text(json.dumps(report))\n',
        encoding='utf8')
    for name, value in {'PYTHONOPTIMIZE': '1', 'PYTEST_ADDOPTS': '-k only_one',
                        'PYTEST_PLUGINS': 'uncontrolled'}.items():
        monkeypatch.setenv(name, value)
    output = tmp_path / 'evidence'
    output.mkdir()
    report, _ = finalizer.run_routes(
        source_root=tmp_path, installation_root=tmp_path / 'installed',
        corpus=tmp_path / 'corpus', output=output,
        build_evidence_path=tmp_path / 'build.json', components={'wp5-spatial'})
    assert report['optimize'] == 0 and report['debug'] is True
    assert report['controls'] == {
        'PYTHONPATH': str(tmp_path / 'installed'),
        'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}


def test_optimized_issuer_refuses_before_source_approval(finalizer, tmp_path):
    command = [sys.executable, '-O',
               str(ROOT / 'tools/native/qualification/finalize.py')]
    for name in ('source-root', 'installation-root', 'records-dir',
                 'postprocess-receipt', 'corpus', 'output'):
        command.extend(('--' + name, str(tmp_path / name)))
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    assert process.returncode != 0
    assert 'optimized Python' in process.stderr
    assert not (tmp_path / 'output').exists()


def test_console_launcher_identity_does_not_attest_repair_implementation(finalizer):
    with pytest.raises(finalizer.FinalizationError, match='implementation/helper'):
        finalizer._check_repair_observation({'kind': 'auditwheel'})


def test_unknown_external_header_is_not_approved_by_dependency_capture(
        finalizer, tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    unit = source / 'local.cpp'
    unit.write_text('#include "external.h"\n', encoding='utf8')
    external = tmp_path / 'external.h'
    external.write_text('#define FACTOR 2\n', encoding='utf8')
    local_identity = finalizer.file_identity(unit)
    dependencies = [local_identity, finalizer.file_identity(external)]
    build = {'translation_units': [{'source': str(unit),
                                    'dependencies': dependencies}],
             'dependencies': dependencies}
    measurement = {'files': {'local.cpp': local_identity['sha256']}}
    with pytest.raises(finalizer.FinalizationError, match='external'):
        finalizer._check_source_inputs(build, measurement, source)


def test_primary_translation_unit_requires_source_approval(finalizer, tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    external = tmp_path / 'external.cpp'
    external.write_text('double injected() { return 2; }\n', encoding='utf8')
    identity = finalizer.file_identity(external)
    build = {'translation_units': [{'source': str(external),
                                    'dependencies': [identity]}],
             'dependencies': [identity]}
    with pytest.raises(finalizer.FinalizationError, match='translation unit'):
        finalizer._check_source_inputs(build, {'files': {}}, source)


@pytest.mark.skipif(not shutil.which('g++-13') or sys.platform != 'linux',
                    reason='GNU13 actual external forced-header control')
def test_actual_strict_build_cannot_import_unapproved_arithmetic_header(
        finalizer, tmp_path):
    source = tmp_path / 'approved'
    source.mkdir()
    unit = source / 'local.cpp'
    unit.write_text('#ifndef FACTOR\n#define FACTOR 1\n#endif\n'
                    'extern "C" double local(double value) '
                    '{ return FACTOR*value; }\n', encoding='utf8')
    external = tmp_path / 'external.h'
    external.write_text('#define FACTOR 2\n', encoding='utf8')
    records = tmp_path / 'records'
    strict = ['-O3', '-fno-fast-math', '-ffp-contract=off', '-fno-lto', '-fPIC']
    commands = [
        ['g++-13', *strict, '-include', str(external), '-c', str(unit),
         '-o', str(tmp_path / 'local.o')],
        ['g++-13', *strict, '-shared', str(tmp_path / 'local.o'),
         '-o', str(tmp_path / '_core.so')],
    ]
    for command in commands:
        process = subprocess.run(
            [sys.executable, str(ROOT / 'tools/native/qualification/record_command.py'),
             '--output-dir', str(records), '--', *command], cwd=tmp_path,
            env=finalizer.controlled_environment(), text=True, capture_output=True,
            check=False)
        assert process.returncode == 0, process.stdout + process.stderr
    native = ctypes.CDLL(str(tmp_path / '_core.so'))
    native.local.argtypes = [ctypes.c_double]
    native.local.restype = ctypes.c_double
    assert native.local(3) == 6
    measurement = {'files': {'local.cpp': finalizer.file_identity(unit)['sha256']}}
    with pytest.raises((finalizer.FinalizationError, finalizer.BuildEvidenceError),
                       match='external|unapproved|provenance'):
        build = finalizer.verify_build(records, source)
        finalizer._check_source_inputs(build, measurement, source)


@pytest.mark.skipif(not shutil.which('g++-13') or sys.platform != 'linux',
                    reason='GNU13 actual external header provenance control')
def test_actual_toolchain_python_and_pybind_headers_have_independent_provenance(
        finalizer, tmp_path):
    pybind11 = pytest.importorskip('pybind11')
    source = tmp_path / 'approved'
    source.mkdir()
    unit = source / 'local.cpp'
    unit.write_text('#include <pybind11/pybind11.h>\n#include <cmath>\n'
                    'extern "C" double local(double x) { return std::fabs(x); }\n',
                    encoding='utf8')
    records = tmp_path / 'records'
    strict = ['-O3', '-fno-fast-math', '-ffp-contract=off', '-fno-lto', '-fPIC']
    commands = [
        ['g++-13', *strict, '-I' + sysconfig.get_path('include'),
         '-I' + pybind11.get_include(), '-c', str(unit),
         '-o', str(tmp_path / 'local.o')],
        ['g++-13', *strict, '-shared', str(tmp_path / 'local.o'),
         '-o', str(tmp_path / '_core.so')],
    ]
    for command in commands:
        process = subprocess.run(
            [sys.executable, str(ROOT / 'tools/native/qualification/record_command.py'),
             '--output-dir', str(records), '--', *command], cwd=tmp_path,
            env=finalizer.controlled_environment(), text=True, capture_output=True,
            check=False)
        assert process.returncode == 0, process.stdout + process.stderr
    build = finalizer.verify_build(records, source)
    measurement = {'files': {'local.cpp': finalizer.file_identity(unit)['sha256']}}
    finalizer._check_source_inputs(build, measurement, source)
    assert {item['provenance_kind'] for item in build['external_inputs']} == {
        'compiler_builtin_header', 'python_header', 'pybind11_distribution'}


def test_runtime_anchor_is_not_written_for_draft_approval(finalizer, tmp_path):
    source = tmp_path / 'source'
    for name in ('vendor/voro++', 'cpp', 'cmake', 'src/pyvoro2/_internal'):
        (source / name).mkdir(parents=True)
    for name in ('CMakeLists.txt', 'pyproject.toml'):
        (source / name).write_text('# fixture\n', encoding='utf8')
    internal = source / 'src/pyvoro2/_internal'
    shutil.copyfile(ROOT / 'src/pyvoro2/_internal/native_qualification.py',
                    internal / 'native_qualification.py')
    (internal / 'native_approval.json').write_text('{"approved":false}',
                                                   encoding='utf8')
    with pytest.raises((finalizer.FinalizationError, ValueError),
                       match='approval'):
        finalizer.finalize(
            source_root=source, installation_root=tmp_path,
            records_dir=tmp_path / 'commands',
            postprocess_path=tmp_path / 'postprocess.json',
            corpus=tmp_path / 'archive', output=tmp_path / 'evidence')
    assert not (tmp_path / 'pyvoro2/_internal/'
                '_qualification_installation.py').exists()


@pytest.fixture
def controlled_fixture(finalizer, tmp_path, monkeypatch):
    """Hermetic issuer plumbing; this fixture is not production native evidence."""
    source = tmp_path / 'source'
    internal = source / 'src/pyvoro2/_internal'
    for directory in (internal, source / 'vendor/voro++', source / 'cpp',
                      source / 'cmake', source / 'tools/native/qualification'):
        directory.mkdir(parents=True)
    for name in ('CMakeLists.txt', 'pyproject.toml', 'cpp/fixture.cpp',
                 'src/pyvoro2/__init__.py', 'cpp/bindings.cpp', 'cpp/bindings2d.cpp',
                 'cpp/native_witness.cpp', 'cpp/planar_witness.cpp', 'cpp/fpguard.cpp'):
        (source / name).write_text('# fixture\n', encoding='utf8')
    for name in ('native_qualification.py', '_qualification_installation.py'):
        shutil.copyfile(ROOT / 'src/pyvoro2/_internal' / name, internal / name)
    # The fixed child is a reviewed fixture source. A caller cannot substitute
    # a command or import its receipt through the production finalizer API.
    runner = source / 'tools/native/qualification/route_suite.py'
    runner.write_text(
        'import json, pathlib, sys\n'
        'args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n'
        'build = json.loads(pathlib.Path(args["--build-evidence"]).read_text())\n'
        'pathlib.Path(args["--output"]).write_text(json.dumps(build["fixture"]))\n',
        encoding='utf8')
    q = finalizer._contract(source)
    target = q.current_target()
    supported = next((item for item in q.ADAPTERS.values()
                      if item[:2] == (target['platform'], target['machine'])), None)
    if supported is None:
        pytest.skip('no reviewed platform adapter in this fixture')
    required = supported[3]
    measured = finalizer.measure_source(source)
    approval = {'approval_schema': q.APPROVAL_SCHEMA, 'approved': True,
                **{field: measured[field] for field in (
                    'policy_revision', 'source_sha256', 'schema_sha256',
                    'consumer_sha256', 'components')}}
    (internal / 'native_approval.json').write_bytes(q.canonical_json(approval))
    install = tmp_path / 'installed'
    shutil.copytree(source / 'src/pyvoro2', install / 'pyvoro2')
    build = {'components': {}, 'dependencies': [],
             'toolchain': {'family': supported[2], 'version': 'fixture-1',
                           'compiler_sha256': 'c' * 64, 'linker_sha256': 'd' * 64}}
    build['adapter'] = next(name for name, item in q.ADAPTERS.items()
                            if item == supported)
    receipt = {'schema': 'pyvoro2-native-postprocess-v1', 'modules': {},
               'dependencies': {}, 'operations': []}
    modules = {}
    for short in ('_core', '_core2d', '_fpguard'):
        linked = tmp_path / ('linked-' + short + '.so')
        linked.write_bytes(('native fixture ' + short).encode('ascii'))
        identity = finalizer.file_identity(linked)
        relative = 'pyvoro2/' + short + '.fixture.so'
        shutil.copyfile(linked, install / relative)
        build['components'][short] = {'output': identity}
        name = 'pyvoro2.' + short
        receipt['modules'][name] = {
            'input': identity, 'output': {**identity, 'path': relative},
            'dependencies': []}
        receipt['operations'].append({'kind': 'identity',
                                      'inputs': [identity['sha256']],
                                      'outputs': [identity['sha256']]})
        modules[name] = identity['sha256']
    guard = receipt['modules']['pyvoro2._fpguard']['output']
    receipt['dependencies'][guard['path']] = {
        key: guard[key] for key in ('sha256', 'size')}
    for short in ('_core', '_core2d'):
        receipt['modules']['pyvoro2.' + short]['dependencies'].append(guard['path'])
    receipt_path = tmp_path / 'postprocess.json'
    receipt_path.write_text(json.dumps(receipt), encoding='utf8')
    components = {name: {'passed': True,
                         'tests': [{'nodeid': name, 'outcome': 'passed'}]}
                  for name in required}
    if 'wp6-planar' in required:
        components['wp6-planar']['corpora'] = {
            'current': {'cases': 48, 'occurrences': 1718},
            'archived': {'cases': 92, 'occurrences': 2298}}
    if 'wp7-planar' in required:
        components['wp7-planar']['corpora'] = {
            'selected': {'cases': 30, 'occurrences': 287}}
    native_identity = {
        'record_schema': q.RECORD_SCHEMA, 'policy_revision': q.POLICY_REVISION,
        **{field: measured[field] for field in (
            'source_sha256', 'schema_sha256', 'consumer_sha256')}}
    report = {
        'schema': 'pyvoro2-native-route-evidence-v1',
        'source_sha256': measured['source_sha256'], 'modules': modules,
        'native_identities': {name: native_identity for name in modules},
        'components': components,
        'discriminators': {
            'standard_strict_bits': '4013fffffb000000',
            'standard_unsafe_bits': '4013fffffb000001',
            'power_offsets': [
                {'radius_hex': radius, 'r_scale_bits': '0000000000000000',
                 'r_scale_check_bits': '3ff0000000000000'}
                for radius in ('0x1.0000000000000p+27', '0x1.0000002000000p+27',
                               '0x1.0000004000000p+27')],
            'guard_disassembly': {
                'tool_sha256': 'a' * 64, 'disassembly_sha256': 'b' * 64,
                'inspect_symbols': ['raw_control'], 'dispatch_symbols': ['entry'],
                'require_environment_symbols': ['require_environment'],
                'fp_before_guard': False},
        },
    }
    controls = report['discriminators']
    controls['power_unsafe_offsets'] = [
        {**row, 'r_scale_bits': '3ff0000000000000'}
        for row in controls['power_offsets']]
    manifests, units = [], []
    for name in ('bindings.cpp', 'bindings2d.cpp', 'native_witness.cpp',
                 'planar_witness.cpp', 'fpguard.cpp'):
        obj, disassembly = tmp_path / (name + '.o'), tmp_path / (name + '.txt')
        obj.write_bytes(('object ' + name).encode())
        disassembly.write_bytes(('disassembly ' + name).encode())
        item = finalizer.file_identity(obj)
        dependency = finalizer.file_identity(source / 'cpp' / name)
        build['dependencies'].append(dependency)
        units.append({'source': dependency['path'], 'output': item,
                      'dependencies': [dependency]})
        manifests.append({'object': item,
                          'disassembly': finalizer.file_identity(disassembly),
                          'argv': [sys.executable, str(obj)], 'exit_code': 0})
    tool = finalizer.file_identity(sys.executable)
    controls['guard_disassembly'].update(
        tool_sha256=tool['sha256'], objects=manifests,
        disassembly_sha256=finalizer.build_digest({'tool': tool, 'objects': manifests}))
    build['translation_units'] = units
    build['fixture'] = report
    report['candidate'] = {'native_identity': native_identity,
                           'sha256': modules['pyvoro2._core2d']}
    monkeypatch.setattr(finalizer, 'verify_build', lambda *_: build)
    arguments = dict(source_root=source, installation_root=install,
                     records_dir=tmp_path / 'commands', postprocess_path=receipt_path,
                     corpus=tmp_path / 'corpus', output=tmp_path / 'evidence',
                     candidate_records=tmp_path / 'candidate-commands',
                     candidate_module=tmp_path / 'linked-_core2d.so')
    return finalizer, arguments, report, q


def test_fixed_child_completes_before_atomic_record_and_anchor(controlled_fixture):
    finalizer, arguments, report, q = controlled_fixture
    record = finalizer.finalize(**arguments)
    internal = arguments['installation_root'] / 'pyvoro2/_internal'
    anchor = {}
    exec((internal / '_qualification_installation.py').read_bytes(), anchor)
    data = (internal / 'native_qualification_record.json').read_bytes()
    assert anchor['RECORD_SHA256'] == q.hashlib.sha256(data).hexdigest()
    assert anchor['INSTALLATION_ID'] == record['installation_id']
    assert set(record['components']) == set(report['components'])
    assert (arguments['output'] / 'qualification-evidence.json').is_file()
    assert (arguments['output'] / 'route.stdout').is_file()


@pytest.mark.parametrize('change', [
    'missing_component', 'failed_test', 'wrong_native_hash', 'wrong_native_source',
    'guard_before_control', 'unsafe_not_discriminated', 'wrong_candidate_hash',
    'missing_unsafe_power', 'missing_guard_fpguard', 'missing_require_guard',
])
def test_incomplete_child_never_issues_production_anchor(controlled_fixture, change):
    finalizer, arguments, report, _ = controlled_fixture
    if change == 'missing_component':
        report['components'].pop(next(iter(report['components'])))
    elif change == 'failed_test':
        next(iter(report['components'].values()))['tests'][0]['outcome'] = 'failed'
    elif change == 'wrong_native_hash':
        report['modules']['pyvoro2._core'] = '0' * 64
    elif change == 'wrong_native_source':
        report['native_identities']['pyvoro2._core']['source_sha256'] = '0' * 64
    elif change == 'guard_before_control':
        report['discriminators']['guard_disassembly']['fp_before_guard'] = True
    elif change == 'wrong_candidate_hash':
        report['candidate']['sha256'] = '0' * 64
    elif change == 'missing_unsafe_power':
        del report['discriminators']['power_unsafe_offsets']
    elif change == 'missing_guard_fpguard':
        report['discriminators']['guard_disassembly']['objects'].pop()
    elif change == 'missing_require_guard':
        guard = report['discriminators']['guard_disassembly']
        guard['require_environment_symbols'] = []
    else:
        report['discriminators']['standard_unsafe_bits'] = '4013fffffb000000'
    internal = arguments['installation_root'] / 'pyvoro2/_internal'
    before = (internal / '_qualification_installation.py').read_bytes()
    with pytest.raises(finalizer.FinalizationError):
        finalizer.finalize(**arguments)
    assert (internal / '_qualification_installation.py').read_bytes() == before
    assert not (internal / 'native_qualification_record.json').exists()


def test_candidate_path_without_observed_build_cannot_qualify(controlled_fixture):
    finalizer, arguments, _, _ = controlled_fixture
    arguments['candidate_records'] = None
    with pytest.raises(finalizer.FinalizationError, match='candidate'):
        finalizer.finalize(**arguments)


def test_candidate_copy_with_different_path_is_not_the_observed_link_output(
        controlled_fixture, tmp_path):
    finalizer, arguments, _, _ = controlled_fixture
    copy = tmp_path / 'unobserved-candidate.so'
    shutil.copyfile(arguments['candidate_module'], copy)
    arguments['candidate_module'] = copy
    with pytest.raises(finalizer.FinalizationError, match='observed candidate'):
        finalizer.finalize(**arguments)
