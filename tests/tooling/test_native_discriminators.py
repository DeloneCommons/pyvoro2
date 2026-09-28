"""The guard inspector must fail on unsafe, bypassed, or absent evidence."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/native'))
from qualification.discriminators import (  # noqa: E402
    inspect_guard_symbols, parse_disassembly, parse_power, tool_environment,
    verify_control_records,
)
from qualification.effective_build import BuildEvidenceError, digest  # noqa: E402


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
