"""The guard inspector must fail on unsafe, bypassed, or absent evidence."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/native'))
from qualification import discriminators  # noqa: E402
from qualification.discriminators import (  # noqa: E402
    _control_arithmetic_argv, _control_compile_options, _control_compiler,
    inspect_guard_symbols, parse_disassembly, parse_power, tool_environment,
    verify_control_records,
)
from qualification.effective_build import (  # noqa: E402
    BuildEvidenceError, digest, effective_options, file_identity,
)


RAW = '''0000 <pyvoro2::native_runtime::inspect()>:
 0: fnstcw (%rax)
 2: stmxcsr (%rax)
 5: ret
0020 <pyvoro2::native_runtime::require_environment()>:
 20: sub $0x40,%rsp
 24: call 29 <pending>
     25: R_X86_64_PLT32 pyvoro2::native_runtime::inspect()-0x4
 29: add $0x40,%rsp
 2d: ret
0010 <pyvoro2::native_runtime::Dispatch<int ()>::operator()() const>:
 10: push %rbx
 11: call 16 <pending>
     12: R_X86_64_PLT32 pyvoro2::native_runtime::require_environment()-0x4
 16: addsd %xmm0,%xmm1
 1a: ret
'''


def test_guard_inspector_accepts_only_raw_then_guarded_arithmetic():
    result = inspect_guard_symbols(parse_disassembly(RAW))
    assert result['fp_before_guard'] is False
    assert len(result['inspect_symbols']) == len(result['dispatch_symbols']) == 1


def test_guard_inspector_accepts_integer_width_extension():
    text = RAW.replace('0: fnstcw (%rax)', '0: movzwl %ax,%eax')
    assert inspect_guard_symbols(parse_disassembly(text))['fp_before_guard'] is False


@pytest.mark.parametrize('text,reason', [
    (RAW.replace('10: push %rbx', '10: cvtsi2sd %rax,%xmm0'), 'before guard'),
    (RAW.replace('10: push %rbx', '10: jmp 16 <bypass>'), 'before guard'),
    (RAW.replace('0: fnstcw (%rax)', '0: fldl (%rax)'), 'raw inspect'),
    (RAW.replace('0: fnstcw (%rax)', '0: fadd %st(1),%st'), 'raw inspect'),
    (RAW.replace('0: fnstcw (%rax)', '0: cmpps $0,%xmm0,%xmm1'), 'raw inspect'),
    (RAW.replace('0: fnstcw (%rax)', '0: data16 fadd %st(1),%st'), 'raw inspect'),
    (RAW.replace('10: push %rbx', '10: mysteryop %xmm1'), 'unreviewed'),
    (RAW.replace('10: push %rbx', '10: (bad)'), 'unparsed'),
    (RAW.replace('29: add $0x40,%rsp', '29: addsd %xmm0,%xmm1'),
     'raw refusal'),
    (RAW.replace('PLT32 pyvoro2::native_runtime::require_environment()',
                 'PLT32 arbitrary_function()'), 'unreviewed call'),
    ('', 'missing'),
])
def test_guard_inspector_refuses_incomplete_or_unsafe_input(text, reason):
    with pytest.raises(BuildEvidenceError, match=reason):
        inspect_guard_symbols(parse_disassembly(text))


def test_raw_guard_module_requires_inspection_but_no_dispatch_symbol():
    result = inspect_guard_symbols(
        parse_disassembly(RAW.split('0010 <')[0]), require_dispatch=False)
    assert result['fp_before_guard'] is False
    assert result['require_environment_symbols']
    assert result['dispatch_symbols'] == []


@pytest.mark.skipif(not (shutil.which('objdump') or shutil.which('llvm-objdump')),
                    reason='actual guard inspector requires objdump')
def test_production_discriminators_refuse_arithmetic_without_actual_guard_objects(
        tmp_path, monkeypatch):
    # The expensive independent arithmetic subprocesses are outside this gate.
    # Keep the real guard consumer: returning the arithmetic report must fail.
    monkeypatch.setattr(discriminators, 'run_arithmetic_controls',
                        lambda *args: {'standard_strict_bits': '4013fffffb000000',
                                       'standard_unsafe_bits': '4013fffffb000001'})
    with pytest.raises(BuildEvidenceError, match='missing production guard'):
        discriminators.run_discriminators(ROOT, {'translation_units': []}, tmp_path)


def test_sanitizer_environment_is_removed_only_for_tool_children():
    original = {'PATH': '/tools', 'ASAN_OPTIONS': 'halt_on_error=1',
                'UBSAN_OPTIONS': 'halt_on_error=1', 'LD_PRELOAD': '/lib/asan.so',
                'PYVORO2_NATIVE_TEST_SANITIZERS': '1'}
    assert tool_environment(original) == {'PATH': '/tools'}
    assert original['LD_PRELOAD'] == '/lib/asan.so'


def test_power_rows_use_the_frozen_platform_independent_hex_spelling():
    assert parse_power('power 0x1.0000002p+27 0000000000000000 '
                       '3ff0000000000000 1\n') == [
        ('0x1.0000002000000p+27', '0000000000000000', '3ff0000000000000')]


@pytest.mark.parametrize('arguments', [
    ['-fno-sanitize=all', '-fsanitize=address,undefined,float-cast-overflow'],
    ['-fsanitize=address,undefined,float-cast-overflow', '-fno-sanitize=all',
     '-fsanitize=address,undefined,float-cast-overflow'],
    ['-fsanitize=address,undefined,float-cast-overflow'],
    ['-fsanitize=thread', '-fno-sanitize=all'],
    ['-fsanitize=address,undefined,float-cast-overflow', '-fno-sanitize=all',
     '-fno-sanitize=all'],
    ['-fsanitize=address,undefined,float-cast-overflow', '-fno-sanitize=undefined'],
])
def test_companion_override_requires_known_bundle_then_final_disable(arguments):
    with pytest.raises(BuildEvidenceError, match='companion instrumentation'):
        _control_arithmetic_argv(arguments, 'gnu', sanitizer_companion=True)


def test_companion_sanitizer_disable_remains_forbidden_for_production():
    strict = ['-fno-fast-math', '-ffp-contract=off', '-fno-lto',
              '-fsanitize=address,undefined,float-cast-overflow']
    with pytest.raises(BuildEvidenceError, match='unreviewed optimization control'):
        effective_options([*strict, '-fno-sanitize=all'], 'gnu')
    with pytest.raises(BuildEvidenceError, match='strict control cannot disable'):
        verify_control_records(Path('.'), {}, unsafe=False, sanitizer_companion=True)
    with pytest.raises(BuildEvidenceError, match='unreviewed unsafe sanitizer'):
        _control_compile_options(strict, 'clang', unsafe=True)
    with pytest.raises(BuildEvidenceError, match='unreviewed unsafe sanitizer'):
        _control_compile_options(['-fsanitize=thread'], 'gnu', unsafe=True)


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX executable symlink mode')
def test_apple_control_link_keeps_verified_production_cxx_driver_mode(tmp_path):
    compiler = tmp_path / 'clang'
    compiler.write_text('#!/bin/sh\ncase "${0##*/}" in '
                        'clang++) exit 0;; *) exit 9;; esac\n')
    compiler.chmod(0o755)
    (tmp_path / 'clang++').symlink_to(compiler)
    identity = file_identity(compiler)
    build = {'tools': [identity],
             'toolchain': {'compiler_sha256': identity['sha256']}}
    unit = {'command': {'argv': ['/usr/bin/clang++', '-c', 'source.cpp']}}
    command = _control_compiler(build, unit, 'clang')
    # An object-only link still gets the C++ mode selected by argv[0].
    assert subprocess.run([command, 'already-built.o']).returncode == 0
    (tmp_path / 'clang++').unlink()
    (tmp_path / 'clang++').write_text('a different executable')
    with pytest.raises(BuildEvidenceError, match='compiler identity'):
        _control_compiler(build, unit, 'clang')


@pytest.mark.skipif(not shutil.which('g++-13'), reason='GNU actual sanitizer control')
def test_unsafe_power_companion_keeps_frozen_bits_without_changing_strict(tmp_path):
    # Use the unchanged vendor operations, not a copied arithmetic expression.
    source = tmp_path / 'power.cpp'
    source.write_text(r'''
#include "rad_option.hh"
#include <cstdio>
#include <cstring>
unsigned long long bits(double value) {
    unsigned long long result;
    static_assert(sizeof result == sizeof value, "binary64 bits");
    std::memcpy(&result, &value, sizeof value);
    return result;
}
struct Radius : voro::radius_poly {
    void run(double radius) {
        double storage[8] = {0, 0, 0, radius, 0, 0, 0, radius};
        double* blocks[] = {storage};
        ppr = blocks; max_radius = radius; r_init(0, 0);
        const double ordinary = r_scale(1., 0, 1);
        double checked = 1.;
        const bool cuts = r_scale_check(checked, 16., 0, 1);
        std::printf("power %a %016llx %016llx %d\n", radius,
                    bits(ordinary), bits(checked), static_cast<int>(cuts));
    }
};
int main() {
    Radius radius;
    for (int i = 0; i != 3; ++i) {
        volatile double value = 134217728. + i;
        radius.run(value);
    }
}
''')
    production = ['-O3', '-std=gnu++17', '-fno-fast-math', '-ffp-contract=off',
                  '-fno-lto', '-fsanitize=address,undefined,float-cast-overflow',
                  '-fno-omit-frame-pointer', '-g',
                  '-I' + str(ROOT / 'vendor/voro++/src')]
    original = list(production)
    recorder = ROOT / 'tools/native/qualification/record_command.py'
    for name, unsafe, expected in (
            ('strict', False, '0000000000000000'),
            ('unsafe_power', True, '3ff0000000000000')):
        options = _control_compile_options(production, 'gnu', unsafe=unsafe)
        if unsafe:
            options += ['-funsafe-math-optimizations', '-ffp-contract=off']
        object_path, executable = tmp_path / (name + '.o'), tmp_path / name
        records = tmp_path / (name + '-records')
        prefix = [sys.executable, str(recorder), '--output-dir', str(records), '--']
        compile_run = subprocess.run(
            [*prefix, 'g++-13', *options, '-c', str(source), '-o', str(object_path)],
            capture_output=True, text=True)
        assert compile_run.returncode == 0, compile_run.stderr
        symbols = subprocess.run(['nm', '-u', str(object_path)],
                                 capture_output=True, text=True, check=True).stdout
        if not unsafe:
            assert '__asan_report_' in symbols
        link_run = subprocess.run(
            [*prefix, 'g++-13', '-fno-fast-math', '-ffp-contract=off', '-fno-lto',
             '-fsanitize=address,undefined,float-cast-overflow',
             str(object_path), '-o', str(executable)], capture_output=True, text=True)
        assert link_run.returncode == 0, link_run.stderr
        environment = tool_environment(os.environ)
        environment.update(ASAN_OPTIONS='detect_leaks=0:halt_on_error=1',
                           UBSAN_OPTIONS='halt_on_error=1')
        run = subprocess.run([str(executable)], env=environment,
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        assert parse_power(run.stdout) == [
            ('0x1.0000000000000p+27', expected, '3ff0000000000000'),
            ('0x1.0000002000000p+27', expected, '3ff0000000000000'),
            ('0x1.0000004000000p+27', expected, '3ff0000000000000')]
        rows = [json.loads(path.read_text()) for path in records.glob('*.json')]
        build = {'tools': [tool for row in rows for tool in row['tools']]}
        assert verify_control_records(records, build, unsafe=unsafe,
                                      sanitizer_companion=unsafe)
        if unsafe:
            with pytest.raises(BuildEvidenceError,
                               match='unreviewed optimization control'):
                verify_control_records(records, build, unsafe=True)
    assert production == original


@pytest.mark.skipif(not shutil.which('g++-13'), reason='GNU actual command control')
def test_discriminator_refuses_successful_build_with_incomplete_records(tmp_path):
    source = tmp_path / 'control.cpp'
    source.write_text('int main() { return 0; }\n')
    records = tmp_path / 'records'
    recorder = ROOT / 'tools/native/qualification/record_command.py'
    strict = ['-fno-fast-math', '-ffp-contract=off', '-fno-lto']
    for args in ([*strict, '-c', str(source), '-o', str(tmp_path / 'c.o')],
                 [*strict, str(tmp_path / 'c.o'), '-o', str(tmp_path / 'control')]):
        run = subprocess.run([sys.executable, str(recorder), '--output-dir',
                              str(records), '--', 'g++-13', *args],
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
    rows = [json.loads(path.read_text()) for path in records.glob('*.json')]
    tools = [tool for row in rows for tool in row['tools']]
    compiler = next(i['executable'] for r in rows for i in r['invocations']
                    if i['role'] == 'compiler_driver')
    linker = next(i['executable'] for r in rows for i in r['invocations']
                  if i['role'] == 'linker')
    build = {'tools': tools, 'toolchain': {
        'compiler_sha256': compiler['sha256'], 'linker_sha256': linker['sha256']}}
    assert verify_control_records(records, build, unsafe=False)
    record = next(records.glob('*.json'))
    row = json.loads(record.read_text())
    row['complete'] = False
    del row['record_sha256']
    row['record_sha256'] = digest(row)
    record.write_text(json.dumps(row))
    with pytest.raises(BuildEvidenceError, match='incomplete'):
        verify_control_records(records, build, unsafe=False)
