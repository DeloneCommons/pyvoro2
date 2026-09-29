"""Apple default controls must retain native, non-LTO link closure."""
from pathlib import Path
import json
import struct
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/native'))

from qualification.effective_build import (  # noqa: E402
    BuildEvidenceError, effective_options, file_identity,
)
from qualification.link_provenance import (  # noqa: E402
    verify_link_closure, verify_linker_options,
)


def _apple_runtime_provider(tmp_path, monkeypatch):
    from qualification import link_provenance
    sdk = tmp_path / 'installed-sdk'
    system = sdk / 'usr/lib/system'
    system.mkdir(parents=True)
    resource = tmp_path / 'installed-clang-resource'
    resource.mkdir()
    driver = tmp_path / 'clang'
    driver.write_bytes(b'independently selected compiler')
    # The actual SDK runtime closure contains these Libsystem sublibraries;
    # their names do not share the libsystem_ prefix.
    names = ('libcache.tbd', 'libcommonCrypto.tbd', 'libcompiler_rt.tbd',
             'libcopyfile.tbd', 'libcorecrypto.tbd', 'libdispatch.tbd',
             'libdyld.tbd', 'libkeymgr.tbd', 'libmacho.tbd',
             'libquarantine.tbd', 'libremovefile.tbd', 'libunwind.tbd',
             'libxpc.tbd')
    paths = [system / name for name in names]
    paths += [system / 'libsystem_c.tbd', system / 'libunreviewed.tbd']
    for path in paths:
        path.write_text('--- !tapi-tbd\ninstall-name: ' + path.name + '\n')
    shadow = tmp_path / 'candidate/libcache.tbd'
    shadow.parent.mkdir()
    shadow.write_bytes(paths[0].read_bytes())
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        expected = ([str(driver), '-print-resource-dir'],
                    ['/usr/bin/xcrun', '--sdk', 'macosx', '--show-sdk-path'])
        assert argv in expected
        path = resource if argv == expected[0] else sdk
        return SimpleNamespace(returncode=0, stdout=str(path) + '\n', stderr='')

    monkeypatch.setattr(link_provenance.subprocess, 'run', run)
    env = {'SDKROOT': str(shadow.parent), 'LIBRARY_PATH': str(shadow.parent),
           'LC_ALL': 'C'}
    identities, evidence = link_provenance.platform_runtime_inputs(
        driver, 'clang', cwd=tmp_path, env=env, directory=tmp_path)
    return paths, shadow, identities, evidence, calls


def test_apple_libsystem_sublibraries_have_independent_sdk_authority(
        tmp_path, monkeypatch):
    paths, _, identities, evidence, calls = _apple_runtime_provider(
        tmp_path, monkeypatch)
    assert all(file_identity(path) in identities for path in paths[:-1])
    assert file_identity(paths[-1]) not in identities
    assert all('SDKROOT' not in kwargs['env'] and
               'LIBRARY_PATH' not in kwargs['env'] for _, kwargs in calls)
    assert json.loads(evidence.read_text())['installed_files'] == identities
    obj = _object(tmp_path)
    identity = file_identity(obj)
    row = {'family': 'clang', 'runtime_link_inputs': identities,
           'link_inputs': [identity, *(file_identity(p) for p in paths[:-1])],
           'object_inputs': [identity]}
    assert verify_link_closure(row, {str(obj): {'output': identity}}) == [identity]


@pytest.mark.parametrize('mutation', ['shadow', 'unknown', 'changed-bytes'])
def test_apple_runtime_provider_does_not_approve_candidate_or_changed_inputs(
        tmp_path, monkeypatch, mutation):
    paths, shadow, identities, _, _ = _apple_runtime_provider(tmp_path, monkeypatch)
    target = shadow if mutation == 'shadow' else paths[-1]
    if mutation == 'changed-bytes':
        target = paths[0]
        target.write_text('changed after independent provider measurement\n')
    obj = _object(tmp_path)
    identity = file_identity(obj)
    row = {'family': 'clang', 'runtime_link_inputs': identities,
           'link_inputs': [identity, file_identity(target)],
           'object_inputs': [identity]}
    with pytest.raises(BuildEvidenceError, match='opaque/unapproved|toolchain link'):
        verify_link_closure(row, {str(obj): {'output': identity}})


def _object(tmp_path, *, segment=b'__TEXT', section=b'__text', cpu=0x01000007):
    # Mach-O 64 header, one LC_SEGMENT_64 and one section_64 with one code byte.
    header = struct.pack('<8I', 0xfeedfacf, cpu, 3, 1, 1, 152, 0, 0)
    command = struct.pack('<II16s4Q4I', 0x19, 152, b'', 0, 1, 184, 1,
                          7, 7, 1, 0)
    entry = struct.pack('<16s16sQQ8I', section, segment, 0, 1, 184,
                        0, 0, 0, 0x80000400, 0, 0, 0)
    path = tmp_path / 'native.o'
    path.write_bytes(header + command + entry + b'\xc3')
    return path


def _verify(path):
    identity = file_identity(path)
    row = {'family': 'clang', 'link_inputs': [identity],
           'object_inputs': [identity]}
    return verify_link_closure(row, {str(path): {'output': identity}})


def test_apple_disabled_module_search_does_not_grant_module_or_fp_controls():
    vector = ['clang', '-cc1', '-ffp-contract=off',
              '-fno-modulemap-allow-subdirectory-search', '-fno-cxx-modules']
    assert effective_options(vector, 'clang', backend=True)['strict_fp']
    for flag in ('-fmodulemap-allow-subdirectory-search', '-fmodules',
                 '-fmodule-map-file=external.modulemap', '-ffp-contract=fast'):
        with pytest.raises(BuildEvidenceError):
            effective_options([*vector, flag], 'clang', backend=True)


def test_apple_linker_defaults_have_a_bounded_allowance():
    verify_linker_options(['ld', '-O3', '-mllvm',
                           '-enable-linkonceodr-outlining', 'native.o'], 'clang')
    for options in (['-Ofast'], ['-mllvm'], ['-mllvm', '-enable-unsafe-fp-math'],
                    ['-mllvm', '-enable-linkonceodr-outlining=1']):
        with pytest.raises(BuildEvidenceError, match='linker control'):
            verify_linker_options(['ld', *options, 'native.o'], 'clang')


@pytest.mark.parametrize('backend', [False, True])
@pytest.mark.parametrize('flag', [
    '-cl-no-signed-zeros', '-cl-fast-relaxed-math',
    '-cl-unsafe-math-optimizations', '-cl-finite-math-only',
    '-cl-mad-enable', '-cl-single-precision-constant',
])
def test_cpp_adapter_refuses_opencl_arithmetic_controls(flag, backend):
    argv = ['clang', '-fno-fast-math', '-ffp-contract=off', flag]
    with pytest.raises(BuildEvidenceError, match='unreviewed.*control'):
        effective_options(argv, 'clang', backend=backend)


@pytest.mark.parametrize('cpu', [0x01000007, 0x0100000c])
def test_apple_native_project_object_is_admissible(tmp_path, cpu):
    path = _object(tmp_path, cpu=cpu)
    assert _verify(path) == [file_identity(path)]


@pytest.mark.parametrize('segment,section', [
    (b'__LLVM', b'__bitcode'), (b'__LLVM', b'__bundle'),
    (b'__TEXT', b'__bitcode'),
])
def test_apple_linker_defaults_cannot_admit_embedded_bitcode(
        tmp_path, segment, section):
    path = _object(tmp_path, segment=segment, section=section)
    with pytest.raises(BuildEvidenceError, match='LTO|bitcode'):
        _verify(path)


@pytest.mark.parametrize('mutation', [
    'truncated', 'command-size', 'section-count', 'section-range', 'unknown-cpu',
])
def test_apple_native_closure_refuses_incomplete_object_inspection(tmp_path, mutation):
    path = _object(tmp_path)
    data = bytearray(path.read_bytes())
    if mutation == 'truncated':
        del data[80:]
    elif mutation == 'command-size':
        struct.pack_into('<I', data, 36, 8)
    elif mutation == 'section-count':
        struct.pack_into('<I', data, 96, 2)
    elif mutation == 'section-range':
        struct.pack_into('<I', data, 152, 10000)
    else:
        struct.pack_into('<I', data, 4, 7)
    path.write_bytes(data)
    with pytest.raises(BuildEvidenceError, match='Mach-O'):
        _verify(path)


def _pipeline(tmp_path):
    # Clang removes the source suffix from foo.cpp.o's saved intermediate name.
    common = {'cwd': str(tmp_path), 'exit_code': 0,
              'observation': 'direct-child-execution'}
    return [
        {**common, 'expanded_argv': [
            'clang', '-cc1', '-E', '-o', 'foo.ii', '-x', 'c++', 'foo.cpp']},
        {**common, 'expanded_argv': [
            'clang', '-cc1', '-emit-llvm-bc', '-o', 'foo.bc',
            '-x', 'c++-cpp-output', 'foo.ii']},
        {**common, 'expanded_argv': [
            'clang', '-cc1', '-S', '-o', 'foo.s', '-x', 'ir', 'foo.bc']},
    ]


def test_apple_preprocessing_is_bound_to_executed_jobs_not_filename_guess(tmp_path):
    from qualification.adapters import apple_preprocessed_input
    assert apple_preprocessed_input(_pipeline(tmp_path), tmp_path / 'foo.cpp') == (
        tmp_path / 'foo.ii')


@pytest.mark.parametrize('mutation', [
    'missing', 'duplicate', 'unconsumed', 'wrong-source', 'failed', 'plan-only',
])
def test_apple_preprocessing_refuses_unbound_expansion(tmp_path, mutation):
    from qualification.adapters import apple_preprocessed_input
    jobs = _pipeline(tmp_path)
    if mutation == 'missing':
        jobs.pop(0)
    elif mutation == 'duplicate':
        jobs.append(jobs[0].copy())
    elif mutation == 'unconsumed':
        jobs[1]['expanded_argv'][-1] = 'different.ii'
    elif mutation == 'wrong-source':
        jobs[0]['expanded_argv'][-1] = 'different.cpp'
    elif mutation == 'failed':
        jobs[0]['exit_code'] = 1
    else:
        jobs[0]['observation'] = 'executed-driver-plan-only'
    with pytest.raises(BuildEvidenceError, match='preprocess'):
        apple_preprocessed_input(jobs, tmp_path / 'foo.cpp')


def _macro_query(tmp_path, monkeypatch, expansion, *, exit_code=0):
    from qualification import record_command
    driver = tmp_path / 'clang'
    driver.write_bytes(b'observed compiler identity')
    source = tmp_path / 'source.cpp'
    source.write_text('int value;\n')
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if '-dM' in command:
            # Clang's dynamic builtin is deliberately absent from -dM output.
            return SimpleNamespace(returncode=0, stderr=b'', stdout=(
                b'#define __DBL_MANT_DIG__ 53\n'
                b'#define __DBL_MAX_EXP__ 1024\n'
                b'#define __LDBL_MANT_DIG__ 64\n'
                b'#define __SIZEOF_INT__ 4\n'
                b'#define __FINITE_MATH_ONLY__ 0\n'))
        return SimpleNamespace(returncode=exit_code, stdout=expansion,
                               stderr=b'diagnostic' if exit_code else b'')

    monkeypatch.setattr(record_command.subprocess, 'run', run)
    options = ['-O3', '-fno-fast-math', '-ffp-contract=off', '-fno-lto',
               '-arch', 'x86_64', '-isysroot', '/selected/sdk']
    argv = [str(driver), *options, '-c', str(source), '-o', 'source.o',
            '-save-temps=obj', '-MD', '-MF', 'source.d']
    environment = {'SDKROOT': '/selected/sdk', 'LANG': 'C'}
    return (record_command, driver, source, argv, options, environment, calls)


def test_clang_evaluation_builtin_is_expanded_under_the_effective_options(
        tmp_path, monkeypatch):
    setup = _macro_query(tmp_path, monkeypatch, b'PYVORO2_EVAL_METHOD 0\n')
    recorder, driver, source, argv, options, environment, calls = setup
    macros, evidence = recorder.query_macros(
        driver, argv, source, 'clang', tmp_path, environment, tmp_path)
    assert macros['__FLT_EVAL_METHOD__'] == '0'
    assert len(calls) == 2
    command, kwargs = calls[1]
    assert command == [str(driver), *options, '-E', '-P', '-x', 'c++',
                       str(tmp_path / 'evaluation-query.cpp')]
    assert kwargs['cwd'] == tmp_path and kwargs['env'] == environment
    assert (tmp_path / 'evaluation-query.cpp').read_text() == (
        'PYVORO2_EVAL_METHOD __FLT_EVAL_METHOD__\n')
    query = json.loads((tmp_path / 'evaluation-query.json').read_text())
    assert query == {'argv': command, 'cwd': str(tmp_path),
                     'environment': environment, 'executable': file_identity(driver),
                     'exit_code': 0}
    assert {path.name for path in evidence} >= {
        'evaluation-query.cpp', 'evaluation-query.txt', 'evaluation-query.stderr',
        'evaluation-query.json', 'type-macros.txt', 'macro-query.json'}


@pytest.mark.parametrize('expansion,exit_code', [
    (b'', 0), (b'PYVORO2_EVAL_METHOD __FLT_EVAL_METHOD__\n', 0),
    (b'PYVORO2_EVAL_METHOD 0 1\n', 0),
    (b'PYVORO2_EVAL_METHOD 0\nPYVORO2_EVAL_METHOD 0\n', 0),
    (b'PYVORO2_EVAL_METHOD -1\n', 0),
    (b'PYVORO2_EVAL_METHOD 1\n', 0), (b'PYVORO2_EVAL_METHOD 2\n', 0),
    (b'PYVORO2_EVAL_METHOD 0\n', 1),
])
def test_clang_evaluation_query_refuses_missing_ambiguous_unsafe_or_failed_evidence(
        tmp_path, monkeypatch, expansion, exit_code):
    setup = _macro_query(tmp_path, monkeypatch, expansion, exit_code=exit_code)
    recorder, driver, source, argv, _, environment, _ = setup
    with pytest.raises(BuildEvidenceError, match='evaluation'):
        recorder.query_macros(
            driver, argv, source, 'clang', tmp_path, environment, tmp_path)
