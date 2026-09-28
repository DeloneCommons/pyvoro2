"""Independent controls for observed native build commands, not CMake intent."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import struct
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'native'))

from qualification.adapters import parse_clang_plan  # noqa: E402
from qualification.effective_build import (  # noqa: E402
    BuildEvidenceError, digest, effective_options, verify_build, windows_words,
    unsafe_source_directives,
)
from qualification.loader_trace import (  # noqa: E402
    apple_system_baseline, capture_glibc, elf_interpreter, macho_uuids,
    parse_dyld_images, parse_glibc_maps,
)
from qualification.link_provenance import verify_linker_options  # noqa: E402


STRICT = ['-O2', '-fno-fast-math', '-ffp-contract=off', '-fno-lto']
HAS_GNU = shutil.which('g++-13')


def record(tmp_path, args):
    result = subprocess.run([
        sys.executable, str(ROOT / 'tools/native/qualification/record_command.py'),
        '--output-dir', str(tmp_path / 'records'), '--', *args,
    ], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return tmp_path / 'records'


def test_effective_order_is_not_a_historical_flag_ban():
    effective_options(['-ffast-math', *STRICT], 'gnu')
    with pytest.raises(BuildEvidenceError, match='unsafe'):
        effective_options([*STRICT, '-ffinite-math-only'], 'gnu')
    with pytest.raises(BuildEvidenceError, match='contraction'):
        effective_options([*STRICT, '-ffp-contract=fast'], 'gnu')
    with pytest.raises(BuildEvidenceError, match='LTO'):
        effective_options([*STRICT, '-flto'], 'gnu')


def test_msvc_effective_environment_suffix_wins():
    effective_options(['/fp:fast', '/fp:strict', '/GL-'], 'msvc')
    with pytest.raises(BuildEvidenceError, match='strict'):
        effective_options(['/fp:strict', '/GL-', '/fp:fast'], 'msvc')


def test_trap_and_rounding_premises_are_adapter_specific_and_ordered():
    effective_options(['-fno-trapping-math', *STRICT], 'gnu')
    effective_options([*STRICT, '-fno-trapping-math', '-ftrapping-math'], 'gnu')
    effective_options([*STRICT, '-fno-rounding-math'], 'gnu')
    effective_options([*STRICT, '-fno-trapping-math'], 'clang')
    with pytest.raises(BuildEvidenceError, match='trapping-math'):
        effective_options([*STRICT, '-fno-trapping-math'], 'gnu')


@pytest.mark.parametrize('options', [
    ['-mfpmath=387', '-D__FLT_EVAL_METHOD__=0'],
    ['-mlong-double-64', '-D__LDBL_MANT_DIG__=64'],
    ['-U', '__FAST_MATH__'],
    ['-funknown-arithmetic-mode'],
])
def test_proof_premises_cannot_be_redefined_or_opaquely_changed(options):
    with pytest.raises(BuildEvidenceError, match='premise|evaluation|unreviewed'):
        effective_options([*STRICT, *options], 'gnu')


def test_windows_response_quoting_preserves_paths_and_empty_arguments():
    assert windows_words(r'"C:\Program Files\cl.exe" /I"C:\Headers Here" ""') == [
        r'C:\Program Files\cl.exe', r'/IC:\Headers Here', '']


def test_loader_trace_requires_resolved_images_for_every_generated_mapping():
    trace = ('7: file=libmpfr.so.6 [0]; generating link map\n'
             '7: calling init: /lib64/ld-linux-x86-64.so.2\n'
             '7: calling init: /lib/libmpfr.so.6\n'
             '7: transferring control: /gcc/cc1plus\n')
    result = parse_glibc_maps(trace)
    assert '/lib/libmpfr.so.6' in result['images']
    with pytest.raises(BuildEvidenceError, match='unresolved'):
        parse_glibc_maps(trace.replace('7: calling init: /lib/libmpfr.so.6\n', ''))
    with pytest.raises(BuildEvidenceError, match='missing'):
        parse_glibc_maps('')


def elf_with_interpreter(path, *, count=1):
    data = path + b'\0'
    header = b'\x7fELF\x02\x01\x01' + b'\0' * 9
    header += struct.pack('<HHIQQQIHHHHHH', 3, 62, 1, 0, 64, 0, 0,
                          64, 56, count, 0, 0, 0)
    program = struct.pack('<IIQQQQQQ', 3, 4, 64 + 56 * count,
                          0, 0, len(data), len(data), 1)
    return header + program * count + data


def test_elf_interpreter_is_explicit_bounded_and_unambiguous():
    path = b'/lib64/ld-linux-x86-64.so.2'
    assert elf_interpreter(elf_with_interpreter(path)) == path.decode()
    for data in (elf_with_interpreter(path, count=0),
                 elf_with_interpreter(path, count=2),
                 elf_with_interpreter(path)[:-1],
                 elf_with_interpreter(b'relative/loader'),
                 elf_with_interpreter(b'/loader\0trailing')):
        with pytest.raises(BuildEvidenceError, match='interpreter|ELF'):
            elf_interpreter(data)


@pytest.mark.skipif(not HAS_GNU, reason='ELF GNU13 tool identity')
def test_glibc_without_loader_initializer_binds_actual_program_interpreter(tmp_path):
    program = Path(HAS_GNU).resolve()
    trace = tmp_path / 'old-glibc.7'
    trace.write_text('7: calling init: ' + str(program) + '\n'
                     '7: transferring control: ' + str(program) + '\n')
    evidence = capture_glibc(tmp_path / 'old-glibc')
    interpreter = Path(elf_interpreter(program.read_bytes())).resolve()
    assert str(interpreter) in {row['path'] for row in evidence['images']}
    assert evidence['program_interpreters'][0]['program']['path'] == str(program)


def test_apple_loader_requires_actual_pid_uuid_and_matching_image_structure():
    identifier = uuid.UUID('01234567-89ab-cdef-0123-456789abcdef')
    line = f'dyld[42]: <{identifier}> /xcode/lib/libLLVM.dylib\n'
    assert parse_dyld_images(line, 42) == [
        {'path': '/xcode/lib/libLLVM.dylib', 'uuid': str(identifier)}]
    with pytest.raises(BuildEvidenceError, match='process'):
        parse_dyld_images(line, 43)
    with pytest.raises(BuildEvidenceError, match='missing'):
        parse_dyld_images('compiler diagnostics only', 42)
    header = struct.pack('<IIIIIIII', 0xfeedfacf, 0x1000007, 3, 6, 1, 24, 0, 0)
    image = header + struct.pack('<II', 0x1b, 24) + identifier.bytes
    assert macho_uuids(image) == {str(identifier)}
    with pytest.raises(BuildEvidenceError, match='invalid'):
        macho_uuids(image[:-1])


def test_apple_delayed_images_remain_bound_to_the_same_observed_process():
    identifier = '01234567-89ab-cdef-0123-456789abcdef'
    image = f'dyld[42]: <{identifier}> /System/Library/XPCSupport\n'
    transitions = ('dyld[42]: move loaded to delayed: XPCSupport\n'
                   'dyld[42]: move delayed to loaded: XPCSupport\n')
    assert parse_dyld_images(image + transitions, 42) == [
        {'path': '/System/Library/XPCSupport', 'uuid': identifier}]
    for bad in ('dyld[43]: move loaded to delayed: XPCSupport\n',
                'dyld[42]: move loaded to delayed: Unknown\n',
                'dyld[42]: unreviewed status: XPCSupport\n'):
        with pytest.raises(BuildEvidenceError):
            parse_dyld_images(image + bad, 42)


def test_apple_loaded_executable_uuid_requires_the_native_slice():
    identifier = uuid.UUID('01234567-89ab-cdef-0123-456789abcdef')
    header = struct.pack('<IIIIIIII', 0xfeedfacf, 0x1000007, 3, 2, 1, 24, 0, 0)
    image = header + struct.pack('<II', 0x1b, 24) + identifier.bytes
    assert macho_uuids(image, architecture='x86_64') == {str(identifier)}
    assert macho_uuids(image, architecture='arm64') == set()


def test_apple_cache_provenance_does_not_initialize_delayed_frameworks(monkeypatch):
    import ctypes
    import platform
    from types import SimpleNamespace

    for name in list(os.environ):
        if name.startswith('DYLD_'):
            monkeypatch.delenv(name)
    calls = []
    library = SimpleNamespace(
        _dyld_get_shared_cache_uuid=lambda output: True,
        _dyld_shared_cache_contains_path=lambda path: True,
        sysctlbyname=lambda *args: 0,
    )

    def existing_process(path, **kwargs):
        calls.append(path)
        assert path is None, 'cache observation must not initialize a framework'
        return library

    monkeypatch.setattr(ctypes, 'CDLL', existing_process)
    monkeypatch.setattr(platform, 'machine', lambda: 'arm64')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='native-os-build\n', stderr=''))
    images = [{'path': '/System/Library/XPCSupport', 'uuid': str(uuid.uuid4())}]
    assert apple_system_baseline(images)['images'] == images
    assert calls == [None]
    monkeypatch.setenv('DYLD_SHARED_CACHE_DIR', '/alternate/cache')
    with pytest.raises(BuildEvidenceError, match='observer loader environment'):
        apple_system_baseline(images)


@pytest.mark.parametrize('frame_pointer', ['all', 'non-leaf'])
def test_actual_apple17_backend_metadata_does_not_weaken_fp(frame_pointer):
    options = [
        '-main-file-name', 'bindings.cpp', '-mframe-pointer=' + frame_pointer,
        '-mllvm', '-extra-vectorizer-passes', '-fno-strict-return',
        '-fobjc-msgsend-selector-stubs', '-faligned-alloc-unavailable',
        '-fcompatibility-qualified-id-block-type-checking',
        '-fvisibility-inlines-hidden-static-local-var',
        '-fbuiltin-headers-in-system-modules', '-fdefine-target-os-macros',
        '-fno-assume-unique-vtables', '-fdebug-compilation-dir=/build',
        '-fcoverage-compilation-dir=/build', '-fstack-check',
        '-mdarwin-stkchk-strong-link', '-fno-cxx-modules', '-fcommon',
        '-fno-odr-hash-protocols',
    ]
    effective_options([*STRICT, *options], 'clang', backend=True)
    for extra in (['-mllvm'], ['-mllvm', '-enable-unsafe-fp-math'],
                  ['-mframe-pointer=opaque']):
        with pytest.raises(BuildEvidenceError, match='unreviewed'):
            effective_options([*STRICT, *options, *extra], 'clang', backend=True)


@pytest.mark.parametrize('option', [
    '/Yuheaders.h', '/Fpstate.pch', '/headerUnit:x', '-Yuheaders.h', '-Fpstate.pch',
])
def test_msvc_serialized_frontend_inputs_are_not_silently_approved(option):
    with pytest.raises(BuildEvidenceError, match='serialized'):
        effective_options(['/fp:strict', '/GL-', option], 'msvc')


@pytest.mark.parametrize('version', ['10.9', '10.13', '10.15', '11.0', '15.2.1'])
def test_apple_deployment_target_is_recorded_platform_configuration(version):
    effective_options([*STRICT, '-mmacosx-version-min=' + version], 'clang')
    with pytest.raises(BuildEvidenceError, match='unreviewed'):
        effective_options([*STRICT, '-mmacosx-version-min=' + version + '-opaque'],
                          'clang')


def test_clang_plan_is_an_explicit_supported_execution_plan():
    plan = parse_clang_plan(
        'Apple clang version 17\n'
        ' "/xcode/bin/clang" "-cc1" "-triple" "arm64-apple-macosx15" '
        '"-emit-obj" "-o" "a.o" "a.cpp"\n',
        Path('/xcode/bin/clang'),
    )
    assert plan == [['/xcode/bin/clang', '-cc1', '-triple',
                     'arm64-apple-macosx15', '-emit-obj', '-o', 'a.o', 'a.cpp']]
    assembly = parse_clang_plan(
        ' "/xcode/bin/clang" "-cc1as" "-triple" "arm64-apple-macosx15" '
        '"-filetype" "obj" "-o" "a.o" "a.s"\n', Path('/xcode/bin/clang'))
    assert assembly[0][1] == '-cc1as'
    with pytest.raises(BuildEvidenceError, match='unsupported'):
        parse_clang_plan(' "/tmp/unknown-wrapper" "a.cpp"\n',
                         Path('/xcode/bin/clang'))
    with pytest.raises(BuildEvidenceError, match='no executable'):
        parse_clang_plan('Apple clang version 17\n', Path('/xcode/bin/clang'))


def test_missing_and_modified_records_do_not_qualify(tmp_path):
    with pytest.raises(BuildEvidenceError, match='no command'):
        verify_build(tmp_path, tmp_path)
    (tmp_path / 'forged.json').write_text(json.dumps({
        'schema': 'pyvoro2.effective-build.v1', 'complete': True,
        'argv': ['g++', *STRICT], 'record_sha256': '0' * 64,
    }))
    with pytest.raises(BuildEvidenceError, match='record digest'):
        verify_build(tmp_path, tmp_path)


@pytest.mark.skipif(not HAS_GNU, reason='GNU13 actual-child observation control')
def test_actual_gnu_compile_and_link_are_bound(tmp_path):
    source = tmp_path / 'primitive.cpp'
    source.write_text('extern "C" double primitive(double a, double b, double c) '
                      '{ return a*b+c; }\n')
    records = record(tmp_path, ['g++-13', *STRICT, '-fPIC', '-c',
                                str(source), '-o', 'primitive.o',
                                '-MF', 'ninja-consumed.d'])
    # Ninja removes its depfile after the compiler launcher finishes.
    (tmp_path / 'ninja-consumed.d').unlink()
    record(tmp_path, ['g++-13', *STRICT, '-shared', 'primitive.o',
                      '-o', '_core.so'])
    evidence = verify_build(records, tmp_path)
    assert len(evidence['translation_units']) == 1
    assert evidence['components']['_core']['translation_units'] == [
        str(source.resolve())]
    assert evidence['link_outputs'][0]['sha256']
    assert evidence['toolchain']['family'] == 'GNU'
    assert evidence['adapter'] == 'gnu-linux-x86_64-v1'
    assert any('libmpfr.so' in Path(t['path']).name for t in evidence['tools'])
    assert any('libgmp.so' in Path(t['path']).name for t in evidence['tools'])
    link_record = next(p for p in records.glob('*.json')
                       if json.loads(p.read_text())['kind'] == 'link')
    original = link_record.read_text()
    row = json.loads(original)
    row['invocations'] = []
    del row['record_sha256']
    row['record_sha256'] = digest(row)
    link_record.write_text(json.dumps(row))
    with pytest.raises(BuildEvidenceError, match='invocation'):
        verify_build(records, tmp_path)
    link_record.write_text(original)
    source.write_text(source.read_text() + '// changed source\n')
    with pytest.raises(BuildEvidenceError, match='changed input'):
        verify_build(records, tmp_path)


@pytest.mark.skipif(not HAS_GNU, reason='GNU13 actual debug/inactive-branch control')
def test_inactive_platform_directive_and_debug_metadata_are_not_effective(tmp_path):
    source = tmp_path / 'inactive.cpp'
    source.write_text('#if defined(_MSC_VER) && defined(PY_LOCAL_AGGRESSIVE)\n'
                      '#pragma optimize("gt", on)\n#endif\n'
                      '#if 0\n#pragma GCC optimize("fast-math")\n#endif\n'
                      'extern "C" double local(double x) { return x * 0; }\n')
    records = record(tmp_path, ['g++-13', *STRICT, '-g', '-fPIC', '-c',
                                str(source), '-o', 'inactive.o'])
    record(tmp_path, ['g++-13', *STRICT, '-shared', 'inactive.o', '-o', '_core.so'])
    assert verify_build(records, tmp_path)['properties']['contraction'] == 'off'


@pytest.mark.skipif(not HAS_GNU, reason='GNU13 actual-child observation control')
def test_real_late_launcher_override_is_observed(tmp_path):
    source = tmp_path / 'arithmetic.cpp'
    source.write_text('double arithmetic(double a, double b, double c) '
                      '{ return a*b+c; }\n')
    launcher = tmp_path / 'late.sh'
    launcher.write_text('#!/bin/sh\nexec "$@" -ffp-contract=fast\n')
    launcher.chmod(0o755)
    records = record(tmp_path, [str(launcher), 'g++-13', *STRICT, '-c',
                                str(source), '-o', 'arithmetic.o'])
    rows = [json.loads(p.read_text()) for p in records.glob('*.json')]
    actual = [i for row in rows for i in row['invocations']
              if Path(i['executable']['path']).name == 'cc1plus']
    assert any('-ffp-contract=fast' in i['argv'] for i in actual)
    with pytest.raises(BuildEvidenceError, match='contraction'):
        verify_build(records, tmp_path)


@pytest.mark.skipif(not HAS_GNU, reason='GNU13 actual-child observation control')
@pytest.mark.parametrize('prefix', [
    '#pragma GCC optimize ("fast-math")\n',
    '#pragma GCC optimize ("no-trapping-math")\n',
    '__attribute__((optimize(\n "fp-contract=fast")))\n',
    '__attribute__((optimize("fp-" "contract=fast")))\n',
    '__attribute__((__optimize__("fp-contract=fast")))\n',
    '__attribute__((target("fpmath=387")))\n',
    '__attribute__((__target__("fpmath=387")))\n',
    '[[gnu::optimize("fp-contract=fast")]]\n',
])
def test_source_local_unsafe_directive_is_not_hidden_by_strict_flags(tmp_path, prefix):
    source = tmp_path / 'local.cpp'
    source.write_text(prefix + 'double local(double x) { return x * 0; }\n')
    records = record(tmp_path, ['g++-13', *STRICT, '-c', str(source),
                                '-o', 'local.o'])
    with pytest.raises(BuildEvidenceError, match='source-local'):
        verify_build(records, tmp_path)


def test_source_directive_polarity_and_comments_do_not_forge_unsafe_settings():
    assert unsafe_source_directives('#pragma GCC optimize ("no-fast-math")\n') == []
    assert unsafe_source_directives('/* #pragma GCC optimize ("fast-math") */') == []
    assert unsafe_source_directives(
        '__attribute__((optimize(\n "fp-contract=fast"))) double f();')
    assert unsafe_source_directives(
        '__attribute__((optimize("fp-" "contract=fast"))) double f();')
    assert unsafe_source_directives('__pragma(float_control(precise, off))')
    assert unsafe_source_directives('_Pragma("GCC optimize(\\"fast-math\\")")')
    assert unsafe_source_directives('__pragma(float_control(precise, on))') == []
    assert unsafe_source_directives(
        '#undef __FLT_EVAL_METHOD__\n', macro_overrides_only=True)
    assert unsafe_source_directives(
        '#pragma optimize("gt", on)\n', macro_overrides_only=True) == []


@pytest.mark.skipif(os.name != 'nt', reason='actual MSVC preprocessor semantics')
@pytest.mark.parametrize('directive', [
    '#pragma float_control(precise, off)',
    '__pragma(float_control(precise, off))',
    '_Pragma("float_control(precise, off)")',
])
def test_actual_msvc_preprocessed_source_preserves_local_controls(tmp_path, directive):
    source, output = tmp_path / 'pragma.cpp', tmp_path / 'pragma.i'
    for active in (False, True):
        source.write_text(f'#if {int(active)}\n{directive}\n#endif\n'
                          'double local(double x) { return x * 0; }\n')
        run = subprocess.run(['cl', '/nologo', '/P', '/Fi' + str(output),
                              str(source)], text=True, capture_output=True)
        assert run.returncode == 0, run.stdout + run.stderr
        assert bool(unsafe_source_directives(output.read_text())) == active


@pytest.mark.skipif(not HAS_GNU, reason='GNU13 actual-child observation control')
@pytest.mark.parametrize('hidden_kind', ['response', 'renamed_object', 'archive'])
def test_every_actual_linked_object_requires_observed_compilation(
        tmp_path, hidden_kind):
    source = tmp_path / 'known.cpp'
    source.write_text('extern "C" int known() { return 1; }\n')
    records = record(tmp_path, ['g++-13', *STRICT, '-fPIC', '-c', str(source),
                                '-o', 'known.o'])
    extra = tmp_path / 'extra.cpp'
    extra.write_text('extern "C" int extra() { return 2; }\n')
    subprocess.run(['g++-13', *STRICT, '-fPIC', '-c', str(extra),
                    '-o', str(tmp_path / 'extra.o')], check=True)
    if hidden_kind == 'archive':
        subprocess.run(['ar', 'rcs', str(tmp_path / 'extra.a'),
                        str(tmp_path / 'extra.o')], check=True)
        injected = ['-Wl,--whole-archive', 'extra.a', '-Wl,--no-whole-archive']
    else:
        hidden = tmp_path / (
            'extra.data' if hidden_kind == 'renamed_object' else 'extra.o')
        if hidden.name != 'extra.o':
            (tmp_path / 'extra.o').rename(hidden)
        (tmp_path / 'extra.rsp').write_text(str(hidden) + '\n')
        injected = ['-Wl,@' + str(tmp_path / 'extra.rsp')]
    record(tmp_path, ['g++-13', *STRICT, '-shared', 'known.o', *injected,
                      '-o', '_core.so'])
    with pytest.raises(BuildEvidenceError, match='linked|archive|link input'):
        verify_build(records, tmp_path)


@pytest.mark.skipif(not HAS_GNU, reason='GNU13 actual-child observation control')
def test_linker_cannot_rewrite_observed_source_call_targets(tmp_path):
    source = tmp_path / 'local.cpp'
    source.write_text(
        'extern "C" __attribute__((noinline)) double first(){return 1;}\n'
        'extern "C" __attribute__((noinline)) double second(){return 2;}\n'
        'extern "C" double local(){return first();}\n')
    records = record(tmp_path, ['g++-13', *STRICT, '-fPIC', '-c', str(source),
                                '-o', 'local.o'])
    record(tmp_path, ['g++-13', *STRICT, '-shared', 'local.o',
                      '-Wl,--defsym,first=second', '-o', '_core.so'])
    with pytest.raises(BuildEvidenceError, match='linker control'):
        verify_build(records, tmp_path)


@pytest.mark.parametrize('family,argv', [
    ('gnu', ['ld', '--wrap=compute']),
    ('clang', ['ld', '-alias', 'first', 'second']),
    ('msvc', ['link.exe', '/alternatename:first=second']),
])
def test_platform_linker_semantic_overrides_are_unreviewed(family, argv):
    with pytest.raises(BuildEvidenceError, match='linker control'):
        verify_linker_options(argv, family)


@pytest.mark.skipif(not HAS_GNU, reason='GNU13 actual-child observation control')
def test_response_file_bytes_and_order_are_observed(tmp_path):
    source = tmp_path / 'response.cpp'
    source.write_text('double response(double x) { return x + 1; }\n')
    response = tmp_path / 'options.rsp'
    response.write_text('-ffast-math ' + ' '.join(STRICT) +
                        ' -c response.cpp -o response.o\n')
    records = record(tmp_path, ['g++-13', '@options.rsp'])
    rows = [json.loads(p.read_text()) for p in records.glob('*.json')]
    assert rows[0]['response_files'][0]['text'] == response.read_text()
    assert rows[0]['complete'] is True


@pytest.mark.skipif(not HAS_GNU, reason='GNU13 actual-child observation control')
@pytest.mark.parametrize('flag,reason', [
    ('-fassociative-math', 'unsafe'),
    ('-ffinite-math-only', 'unsafe'),
    ('-fno-signed-zeros', 'unsafe'),
    ('-flto', 'LTO'),
    ('-mfpmath=387', 'excess-evaluation'),
    ('-fno-trapping-math', 'trapping-math'),
    ('-mlong-double-64', 'evaluation'),
])
def test_actual_effective_property_negatives(tmp_path, flag, reason):
    source = tmp_path / 'subset.cpp'
    source.write_text('double subset(double x, double y) { return x+y; }\n')
    records = record(tmp_path, ['g++-13', *STRICT, flag, '-c', str(source),
                                '-o', 'subset.o'])
    with pytest.raises(BuildEvidenceError, match=reason):
        verify_build(records, tmp_path)


@pytest.mark.skipif(not HAS_GNU, reason='GNU13 actual-child observation control')
def test_builtin_macro_override_cannot_mask_actual_excess_precision(tmp_path):
    source = tmp_path / 'forged.cpp'
    source.write_text('double forged(double a,double b,double c){return a*b+c;}\n')
    records = record(tmp_path, ['g++-13', *STRICT, '-mfpmath=387',
                                '-D__FLT_EVAL_METHOD__=0', '-c', str(source),
                                '-o', 'forged.o'])
    with pytest.raises(BuildEvidenceError, match='evaluation|premise'):
        verify_build(records, tmp_path)
