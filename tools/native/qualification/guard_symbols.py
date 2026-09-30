"""Reviewed GNU/LLVM object grammar for the five native guard objects.

This module identifies callable boundaries; its caller must inspect their
instructions. Registration helpers and destructors are inventory witnesses,
never substitutes for a protected callable body.
"""
from __future__ import annotations

import re

from .effective_build import BuildEvidenceError


PROTECTED_ENTRY_COUNTS = {
    'bindings.cpp': 23, 'native_witness.cpp': 5, 'bindings2d.cpp': 4,
    'planar_witness.cpp': 6, 'fpguard.cpp': 0,
}

_HEADER = re.compile(r'^\s*([0-9a-fA-F]+) <(.+)>:$')
_ADDRESSED = re.compile(r'^\s*([0-9a-fA-F]+):\s*(.*)$')
_INSTRUCTION = re.compile(r'^([A-Za-z][\w.]*)\s*(.*)$')
_ADDEND = re.compile(r'([+-]0x[0-9a-fA-F]+)$')
_PREFIXES = {
    'data16', 'addr32', 'cs', 'ds', 'es', 'ss', 'fs', 'gs',
    'rep', 'repz', 'repe', 'repnz', 'repne', 'lock', 'bnd',
}
_RELOCATIONS = ('R_', 'IMAGE_REL_', 'ARM64_RELOC_', 'X86_64_RELOC_')
_DISPATCH = 'pyvoro2::native_runtime::Dispatch<'
_FUNCTION_REF = 'pybind11::detail::function_ref<'
_GUARDED_DEF = 'pyvoro2::native_runtime::guarded_def<'
_LAMBDA = r'<lambda_[a-f0-9]+>'


def _header_sequence(text):
    return [(int(match[1], 16), match[2]) for line in text.splitlines()
            if (match := _HEADER.fullmatch(line))]


def _symbol_map(text, raw_text):
    if raw_text is None:
        return {}
    headers, raw_headers = _header_sequence(text), _header_sequence(raw_text)
    if len(headers) != len(raw_headers) or any(
            left[0] != right[0] for left, right in zip(headers, raw_headers)):
        raise BuildEvidenceError('raw/demangled disassembly header mismatch')
    result = {}
    macho = bool(re.search(r'file format mach-o', raw_text, re.I))
    for (_, canonical), (_, raw) in zip(headers, raw_headers):
        aliases = [raw]
        # LLVM Mach-O headers may omit the object-format underscore while
        # relocations retain it. Do not trim arbitrary leading underscores.
        if macho and raw.startswith('_Z'):
            aliases.append('_' + raw)
        for alias in aliases:
            if alias in result and result[alias] != canonical:
                raise BuildEvidenceError('ambiguous raw disassembly header')
            result[alias] = canonical
    return result


def _canonical_target(target, names):
    match = _ADDEND.search(target)
    base = target[:match.start()] if match else target
    return names.get(base, base) + (match[0] if match else '')


def _instruction(address, mnemonic, operands, prefixes=()):
    return {'address': address, 'mnemonic': mnemonic, 'operands': operands,
            'relocation': '', 'relocation_kind': '', 'prefixes': list(prefixes)}


def _unparsed(address, text, reason):
    item = _instruction(address, '<unparsed>', text)
    item['parse_error'] = reason
    return item


def parse_disassembly(text, raw_text=None):
    """Preserve instructions, including locally unparsed/unsupported ranges.

    Same-tool raw and demangled header sequences bind relocation identities.
    An unparsed destructor does not abort parsing; a proof-relevant unparsed
    instruction remains a sentinel that the guard checker must reject.
    """
    names = _symbol_map(text, raw_text)
    symbols = {}
    current, previous, pending = None, None, None

    def flush_prefix():
        nonlocal pending
        if pending is not None:
            current.append(_unparsed(pending['address'], pending['text'],
                                     'incomplete/nonconsecutive prefix'))
            pending = None

    for line in text.splitlines():
        header = _HEADER.fullmatch(line)
        if header:
            flush_prefix()
            name = header[2]
            current = symbols.setdefault(name, [])
            if current:
                current[0]['parse_error'] = 'duplicate canonical symbol body'
            previous = None
            continue
        if line.startswith('Disassembly of section '):
            flush_prefix()
            current, previous = None, None
            continue
        if current is None or not line.strip():
            continue
        addressed = _ADDRESSED.fullmatch(line)
        if not addressed:
            # GNU's elided zero bytes and any unrecognized content cannot
            # become an invisible gap in a selected proof range.
            flush_prefix()
            address = previous['address'] + 1 if previous else 0
            previous = _unparsed(address, line.strip(), 'unparsed disassembly')
            current.append(previous)
            continue
        address, content = int(addressed[1], 16), addressed[2]
        match = _INSTRUCTION.fullmatch(content)
        if not match:
            flush_prefix()
            previous = _unparsed(address, content, 'unparsed instruction')
            current.append(previous)
            continue
        mnemonic, operands = match.groups()
        if mnemonic.startswith(_RELOCATIONS):
            if previous is None or pending is not None:
                flush_prefix()
                previous = _unparsed(address, content, 'orphan relocation')
                current.append(previous)
            elif previous['relocation']:
                previous['parse_error'] = 'multiple instruction relocations'
            else:
                previous['relocation'] = _canonical_target(operands, names)
                previous['relocation_kind'] = mnemonic
            continue
        prefixes = []
        while mnemonic in _PREFIXES:
            prefixes.append(mnemonic)
            words = operands.split(None, 1)
            if not words:
                mnemonic = ''
                break
            mnemonic, operands = words[0], words[1] if len(words) == 2 else ''
        start = address
        if pending is not None:
            if address != pending['address'] + len(pending['prefixes']):
                flush_prefix()
            else:
                start = pending['address']
                prefixes = pending['prefixes'] + prefixes
                pending = None
        if not mnemonic:
            pending = {'address': start, 'prefixes': prefixes, 'text': content}
            continue
        if not _INSTRUCTION.fullmatch(mnemonic):
            previous = _unparsed(start, content, 'unparsed prefixed instruction')
        else:
            previous = _instruction(start, mnemonic.lower(), operands, prefixes)
        if any(item['address'] == start for item in current[-2:]):
            previous['parse_error'] = 'duplicate instruction address'
            if current:
                current[0]['parse_error'] = 'duplicate instruction address'
        current.append(previous)
    flush_prefix()
    return symbols


def call_target(instruction):
    """Return an exact target; unsupported offsets never match a boundary."""
    target = instruction.get('relocation', '')
    if target:
        if (instruction.get('relocation_kind') in
                ('R_X86_64_PLT32', 'R_X86_64_PC32') and target.endswith('-0x4')):
            target = target[:-4]
        return target
    operands = instruction.get('operands', '')
    if operands.lstrip().startswith('*'):
        return ''
    match = re.search(r'<(.+)>\s*$', operands)
    return match[1] if match else ''


def _closing(text, start, opening='<', closing='>'):
    if start >= len(text) or text[start] != opening:
        return None
    depth = 1
    for index in range(start + 1, len(text)):
        depth += (text[index] == opening) - (text[index] == closing)
        if depth == 0:
            return index
    return None


def _first_template_argument(text, start, end):
    """Read F without splitting commas inside a function-pointer type."""
    angles, parentheses = 0, 0
    for index in range(start + 1, end):
        character = text[index]
        angles += (character == '<') - (character == '>')
        parentheses += (character == '(') - (character == ')')
        if character == ',' and angles == parentheses == 0:
            return text[start + 1:index].strip()
    return text[start + 1:end].strip()


def _dispatch_identity(name):
    start = name.find(_DISPATCH)
    if start < 0:
        return None
    signature_end = _closing(name, start + len(_DISPATCH) - 1)
    if signature_end is None or not name.startswith('::wrap<', signature_end + 1):
        return None
    function_end = _closing(name, signature_end + len('::wrap<'))
    if function_end is None:
        return None
    return name[start:function_end + 1], function_end + 1


def _dispatch_closure(name):
    parsed = _dispatch_identity(name)
    if parsed is None:
        return None
    identity, end = parsed
    if name[:name.index(_DISPATCH)] not in ('', 'auto '):
        return None
    arguments_end = _closing(name, end, '(', ')')
    if arguments_end is None:
        return None
    suffix = re.match(
        r"::(?:\{lambda\(pybind11::args, pybind11::kwargs\)#1\}|"
        r"'lambda[0-9]*'\(pybind11::args, pybind11::kwargs\))",
        name[arguments_end + 1:])
    if not suffix:
        return None
    return identity, arguments_end + 1 + suffix.end()


def _callback_type(name):
    start = name.find(_FUNCTION_REF)
    if (start <= 0 or not name[start - 1].isspace() or
            name[:start].count('<') != name[:start].count('>')):
        return None
    end = _closing(name, start + len(_FUNCTION_REF) - 1)
    if end is None or not name.startswith('::callback_fn<', end + 1):
        return None
    argument_start = end + len('::callback_fn<')
    argument_end = _closing(name, argument_start)
    if argument_end is None:
        return None
    if not re.fullmatch(
            r'\((?:long|__int64), (?:class )?pybind11::args, '
            r'(?:class )?pybind11::kwargs\)', name[argument_end + 1:]):
        return None
    return name[argument_start + 1:argument_end]


def _dispatch_operator(name):
    parsed = _dispatch_closure(name)
    if parsed:
        identity, end = parsed
        if re.fullmatch(
                r'::operator\(\)\(pybind11::args, pybind11::kwargs\) const'
                r'(?: \[clone \.(?:isra|constprop)\.[0-9]+\])?', name[end:]):
            return identity
        # The retained Apple power-function-pointer closure has this exact
        # substitution spelling in both LLVM and GNU demanglers. Its closure
        # declaration still identifies args/kwargs; do not generalize this to
        # arbitrary operator parameter lists or unrelated wrappers.
        apple_power = (
            '::wrap<pybind11::tuple (*)(pybind11::array_t<double, 17>,'
            in identity and
            'std::__1::array<std::__1::array<double, 2ul>, 2ul>' in identity)
        if apple_power and name[end:] == (
                "::operator()(pybind11::kwargs, "
                "'lambda'(pybind11::args, pybind11::kwargs)) const"):
            return identity
    # Small instruction-check fixtures can represent a direct Dispatch
    # operator. Production also requires real callback bodies and full count.
    if name.startswith(_DISPATCH):
        end = _closing(name, len(_DISPATCH) - 1)
        if end is not None and name[end + 1:] == '::operator()() const':
            return name[:end + 1]
    return None


def _module_def_closure(target):
    marker = 'pybind11::module_::def<'
    start = target.find(marker)
    if (start <= 0 or not target[start - 1].isspace() or
            target[:start].count('<') != target[:start].count('>')):
        return None
    end = _closing(target, start + len(marker) - 1)
    if end is None or not target.endswith(')'):
        return None
    match = re.fullmatch(r'class (' + _LAMBDA + ')',
                         target[start + len(marker):end])
    return match[1] if match else None


def protected_entries(symbols, source_name=None):
    """Inventory full source F identities and actual emitted callable bodies.

    GNU numbers lambdas per signature; Clang uses scoped ordinals. MSVC hides
    the scope in opaque lambda names, so actual guarded_def -> module_.def
    relocations bind each source F to its protected closure instead.
    """
    if source_name is not None and source_name not in PROTECTED_ENTRY_COUNTS:
        raise BuildEvidenceError('unreviewed protected-entry source')
    entries, callbacks, witnesses = {}, {}, set()
    msvc_helpers, validated_msvc_types = {}, set()

    def entry(identity):
        return entries.setdefault(identity, {'identity': identity,
                                             'callbacks': [], 'operators': []})

    for name in symbols:
        if name.endswith(' [clone .cold]'):
            continue
        witness = _dispatch_identity(name)
        if witness:
            witnesses.add(witness[0])
            # Current Windows LLVM demangles auto-return wrap constructors;
            # LLVM18 left them mangled. Match this exact helper's complete F
            # to an independently validated guarded_def entry below. It is
            # never itself a callable boundary or an excuse to drop a witness.
            if name.startswith('public: static auto __cdecl ' + _DISPATCH):
                start = witness[0].index('::wrap<') + len('::wrap<')
                msvc_helpers[witness[0]] = witness[0][start:-1]
        wrapped = _callback_type(name)
        if wrapped is not None:
            callbacks.setdefault(wrapped, []).append(name)
            parsed = _dispatch_closure(wrapped)
            if parsed is not None and parsed[1] == len(wrapped):
                entry(parsed[0])['callbacks'].append(name)
        identity = _dispatch_operator(name)
        if identity:
            entry(identity)['operators'].append(name)

    assigned_closures = set()
    for name, instructions in symbols.items():
        prefix = 'void __cdecl ' + _GUARDED_DEF
        if not name.startswith(prefix):
            continue
        end = _closing(name, len(prefix) - 1)
        if end is None:
            raise BuildEvidenceError('malformed MSVC protected registration')
        identity = name[len('void __cdecl '):end + 1]
        targets = [call_target(item) for item in instructions
                   if item['mnemonic'] in ('call', 'callq')]
        closures = [closure for target in targets
                    if (closure := _module_def_closure(target)) is not None]
        if len(closures) != 1 or closures[0] in assigned_closures:
            raise BuildEvidenceError('ambiguous/missing MSVC registration edge')
        closure = closures[0]
        assigned_closures.add(closure)
        operators = [symbol for symbol in symbols if re.fullmatch(
            r'public: .+ __cdecl ' + re.escape(closure) +
            r'::operator\(\)\(class pybind11::args, '
            r'class pybind11::kwargs\) const', symbol)]
        actual_callbacks = callbacks.get('class ' + closure, [])
        if len(operators) != 1 or not actual_callbacks:
            raise BuildEvidenceError('missing MSVC protected callable body')
        row = entry(identity)
        row.update(operators=operators, callbacks=actual_callbacks,
                   registration=name)
        validated_msvc_types.add(_first_template_argument(
            name, len(prefix) - 1, end))

    resolved_helpers = {identity for identity, source_type in msvc_helpers.items()
                        if source_type in validated_msvc_types}
    if witnesses - entries.keys() - resolved_helpers:
        raise BuildEvidenceError('missing protected callable body')
    rows = sorted(entries.values(), key=lambda row: row['identity'])
    if (source_name is not None and
            len(rows) != PROTECTED_ENTRY_COUNTS[source_name]):
        raise BuildEvidenceError('incomplete protected-entry inventory: ' +
                                 source_name)
    for row in rows:
        if source_name is not None and not row['callbacks']:
            raise BuildEvidenceError('missing protected callback body')
        for kind in ('callbacks', 'operators'):
            row[kind] = sorted(row[kind])
            if any(not symbols[name] for name in row[kind]):
                raise BuildEvidenceError('missing/empty protected callable body')
    return rows
