"""Apple default controls must retain native, non-LTO link closure."""
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/native'))

from qualification.effective_build import (  # noqa: E402
    BuildEvidenceError, effective_options, file_identity,
)
from qualification.link_provenance import (  # noqa: E402
    verify_link_closure, verify_linker_options,
)


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
