"""Bounded object grammar must preserve every protected callable identity."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/native'))
from qualification.effective_build import BuildEvidenceError  # noqa: E402
from qualification.guard_symbols import (  # noqa: E402
    PROTECTED_ENTRY_COUNTS, call_target, parse_disassembly, protected_entries,
)


def closure(number=1):
    return (
        'pyvoro2::native_runtime::Dispatch<int (int, int)>::wrap<'
        'pybind11_init__core(pybind11::module_&)::{lambda(int, int)#'
        + str(number) + '}>(source_lambda, pyvoro2::native_runtime::Family)'
        '::{lambda(pybind11::args, pybind11::kwargs)#1}')


def callback(wrapped):
    return ('int pybind11::detail::function_ref<int (pybind11::args, '
            'pybind11::kwargs)>::callback_fn<' + wrapped +
            '>(long, pybind11::args, pybind11::kwargs)')


def operator(wrapped):
    return wrapped + '::operator()(pybind11::args, pybind11::kwargs) const'


def body(name, instruction='call 0 <guard>'):
    return '0000 <' + name + '>:\n 0: ' + instruction + '\n 5: ret\n'


def test_elf_same_signature_lambdas_are_two_independent_entries():
    first, second = closure(1), closure(2)
    symbols = parse_disassembly(body(callback(first)) + body(callback(second)))
    rows = protected_entries(symbols)
    assert len(rows) == 2
    assert rows[0]['identity'] != rows[1]['identity']
    assert all(len(row['callbacks']) == 1 for row in rows)


def test_real_operator_and_callback_are_one_entry_not_initializer_or_destructor():
    wrapped = closure()
    text = (body(callback(wrapped)) + body(operator(wrapped)) +
            body(wrapped + '::~()') +
            body('void pybind11::cpp_function::initialize<' + wrapped +
                 '>()::{lambda()#1}::operator()() const'))
    rows = protected_entries(parse_disassembly(text))
    assert len(rows) == 1
    assert rows[0]['callbacks'] == [callback(wrapped)]
    assert rows[0]['operators'] == [operator(wrapped)]


def test_macho_inlined_dispatch_and_mangled_guard_relocation():
    wrapped = (
        'auto pyvoro2::native_runtime::Dispatch<int (long)>::wrap<'
        'pybind11_init__core(pybind11::module_&)::$_0>(source_lambda)'
        "::'lambda'(pybind11::args, pybind11::kwargs)")
    name = callback(wrapped)
    guard = 'pyvoro2::native_runtime::require_environment()'
    text = ('object: file format mach-o arm64\n' + body(name, 'bl 0x10') +
            '0010 <' + guard + '>:\n 10: ret\n')
    text = text.replace(' 0: bl 0x10\n', ' 0: bl 0x10\n'
                        ' 0: ARM64_RELOC_BRANCH26 '
                        '__ZN7pyvoro214native_runtime19require_environmentEv\n')
    raw = text.replace(name, '_callback').replace(
        '<' + guard + '>',
        '<_ZN7pyvoro214native_runtime19require_environmentEv>')
    symbols = parse_disassembly(text, raw)
    assert protected_entries(symbols)[0]['callbacks'] == [name]
    assert call_target(symbols[name][0]) == guard


def test_apple_power_pointer_operator_preserves_reviewed_substitution_spelling():
    signature = ('pybind11::tuple (pybind11::array_t<double, 17>, '
                 'std::__1::array<std::__1::array<double, 2ul>, 2ul>)')
    pointer = signature.replace('tuple (', 'tuple (*)(')
    wrapped = ('auto pyvoro2::native_runtime::Dispatch<' + signature +
               '>::wrap<' + pointer + '>(function_pointer)'
               "::'lambda'(pybind11::args, pybind11::kwargs)")
    name = (wrapped + "::operator()(pybind11::kwargs, "
            "'lambda'(pybind11::args, pybind11::kwargs)) const")
    rows = protected_entries(parse_disassembly(body(name) + body(callback(wrapped))))
    assert rows[0]['operators'] == [name]
    wrong = name.replace('::operator()(pybind11::kwargs,',
                         '::operator()(pybind11::object,')
    rows = protected_entries(parse_disassembly(body(wrong) + body(callback(wrapped))))
    assert rows[0]['operators'] == []


@pytest.mark.parametrize('kind,addend,expected', [
    ('R_X86_64_PLT32', '-0x4', 'guard'),
    ('R_X86_64_PC32', '-0x4', 'guard'),
    ('R_X86_64_PLT32', '-0x8', 'guard-0x8'),
    ('R_X86_64_PLT32', '+0x4', 'guard+0x4'),
    ('IMAGE_REL_AMD64_REL32', '-0x4', 'guard-0x4'),
    ('ARM64_RELOC_BRANCH26', '-0x4', 'guard-0x4'),
])
def test_only_reviewed_pc_relative_addends_are_removed(kind, addend, expected):
    symbols = parse_disassembly(
        '0000 <entry>:\n 0: call 5 <pending>\n'
        ' 1: ' + kind + ' guard' + addend + '\n')
    assert call_target(symbols['entry'][0]) == expected


def test_direct_symbol_operand_does_not_hide_wrong_offset_or_indirect_call():
    assert call_target({'operands': '0x20 <guard+0x4>'}) == 'guard+0x4'
    assert call_target({'operands': '*%rax'}) == ''


@pytest.mark.parametrize('raw', [
    '0010 <raw>:\n 10: ret\n',
    '0000 <raw>:\n 0: ret\n0010 <extra>:\n 10: ret\n',
])
def test_raw_demangled_header_pairing_must_match(raw):
    with pytest.raises(BuildEvidenceError, match='header'):
        parse_disassembly('0000 <entry>:\n 0: ret\n', raw)


def test_llvm_split_prefix_preserves_prefix_start_and_instruction_boundary():
    symbols = parse_disassembly('0000 <destructor>:\n'
                                ' 0: lock\n 1: xaddq %rax,(%rbx)\n 5: retq\n')
    assert symbols['destructor'][0] == {
        'address': 0, 'mnemonic': 'xaddq', 'operands': '%rax,(%rbx)',
        'relocation': '', 'relocation_kind': '', 'prefixes': ['lock'],
    }
    assert symbols['destructor'][1]['address'] == 5


def test_split_prefix_cannot_hide_floating_instruction_from_the_checker():
    symbols = parse_disassembly('0000 <entry>:\n'
                                ' 0: data16\n 1: addsd %xmm0,%xmm1\n')
    assert symbols['entry'][0]['address'] == 0
    assert symbols['entry'][0]['mnemonic'] == 'addsd'
    assert symbols['entry'][0]['prefixes'] == ['data16']


def test_known_prefix_does_not_erase_unknown_following_text():
    instruction = parse_disassembly(
        '0000 <entry>:\n 0: lock (bad)\n')['entry'][0]
    assert instruction['parse_error'] == 'unparsed prefixed instruction'


@pytest.mark.parametrize('bad', ['0: (bad)', '0: <unknown>', '0: .byte 0xff',
                                 '0: lock\n 2: xaddq %rax,(%rbx)', '0: lock'])
def test_unparsed_instruction_is_a_local_sentinel_never_globally_skipped(bad):
    symbols = parse_disassembly('0000 <destructor>:\n ' + bad + '\n' +
                                body(operator(closure())))
    assert any(i.get('parse_error') for i in symbols['destructor'])
    assert len(protected_entries(symbols)) == 1
    relevant = parse_disassembly('0000 <' + operator(closure()) + '>:\n ' +
                                 bad + '\n')
    assert any(i.get('parse_error') for i in relevant[operator(closure())])


def test_unknown_instruction_is_retained_for_the_guard_checker():
    symbols = parse_disassembly(body(operator(closure()), 'mysteryop %rax'))
    assert symbols[operator(closure())][0]['mnemonic'] == 'mysteryop'


def test_destructor_witness_cannot_replace_a_missing_callable_body():
    with pytest.raises(BuildEvidenceError, match='missing.*body'):
        protected_entries(parse_disassembly(body(closure() + '::~()')))


def test_production_inventory_requires_all_full_identities_and_callbacks():
    text = ''.join(body(callback(closure(i))) for i in range(1, 24))
    symbols = parse_disassembly(text)
    assert len(protected_entries(symbols, 'bindings.cpp')) == 23
    assert PROTECTED_ENTRY_COUNTS['bindings.cpp'] == 23
    del symbols[callback(closure(2))]
    with pytest.raises(BuildEvidenceError, match='inventory'):
        protected_entries(symbols, 'bindings.cpp')
    symbols[operator(closure(2))] = parse_disassembly(
        body(operator(closure(2))))[operator(closure(2))]
    with pytest.raises(BuildEvidenceError, match='callback'):
        protected_entries(symbols, 'bindings.cpp')


def test_unknown_source_refuses_and_fpguard_has_explicit_zero_dispatch_entries():
    assert protected_entries({}, 'fpguard.cpp') == []
    with pytest.raises(BuildEvidenceError, match='source'):
        protected_entries({}, 'unknown.cpp')


MSVC_F = '<lambda_eeeacb894cd04b95133bcf9f8b87b773>'
MSVC_G = '<lambda_6c103daf5339aad532ff51b49fcb70ef>'
MSVC_REG = ('void __cdecl pyvoro2::native_runtime::guarded_def<class ' + MSVC_F +
            ', struct pybind11::arg>(class pybind11::module_&, char const *, '
            'class ' + MSVC_F + ', struct pybind11::arg &&)')
MSVC_DEF = ('public: class pybind11::module_& __cdecl '
            'pybind11::module_::def<class ' + MSVC_G + '>(char const *, class ' +
            MSVC_G + ' &&)')
MSVC_OP = ('public: int __cdecl ' + MSVC_G +
           '::operator()(class pybind11::args, class pybind11::kwargs) const')
MSVC_CB = ('private: static int __cdecl pybind11::detail::function_ref<'
           'int __cdecl(class pybind11::args, class pybind11::kwargs)>'
           '::callback_fn<class ' + MSVC_G +
           '>(__int64, class pybind11::args, class pybind11::kwargs)')


def coff():
    text = (body(MSVC_REG, 'callq 0x5 <pending>') + body(MSVC_DEF) +
            body(MSVC_OP) + body(MSVC_CB))
    text = text.replace(' 0: callq 0x5 <pending>\n',
                        ' 0: callq 0x5 <pending>\n'
                        ' 1: IMAGE_REL_AMD64_REL32 _raw_def\n')
    raw = text
    for i, name in enumerate((MSVC_REG, MSVC_DEF, MSVC_OP, MSVC_CB)):
        raw = raw.replace('<' + name + '>',
                          '<_raw_def>' if i == 1 else '<raw_' + str(i) + '>')
    return parse_disassembly(text, raw)


def test_coff_opaque_closure_is_bound_through_actual_guarded_def_relocation():
    rows = protected_entries(coff())
    assert len(rows) == 1
    assert MSVC_F in rows[0]['identity']
    assert rows[0]['registration'] == MSVC_REG
    assert rows[0]['callbacks'] == [MSVC_CB]
    assert rows[0]['operators'] == [MSVC_OP]


@pytest.mark.parametrize('missing', [MSVC_OP, MSVC_CB, MSVC_REG])
def test_coff_missing_closure_body_cannot_be_replaced_by_arbitrary_lambdas(missing):
    symbols = coff()
    del symbols[missing]
    if missing == MSVC_REG:
        assert protected_entries(symbols) == []
    else:
        with pytest.raises(BuildEvidenceError, match='missing'):
            protected_entries(symbols)


def test_coff_unknown_or_offset_def_edge_refuses():
    symbols = coff()
    symbols[MSVC_REG][0]['relocation'] += '+0x4'
    with pytest.raises(BuildEvidenceError, match='registration'):
        protected_entries(symbols)
