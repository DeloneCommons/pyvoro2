"""Validate externally collected, actual native-build evidence.

This module does not approve source, issue artifact records, or infer compiler
correctness from a probe. Its caller must own the controlled build and retain
the separate approval, route, discriminator, and final-payload evidence.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import struct

SCHEMA = 'pyvoro2.effective-build.v1'
EVIDENCE_SCHEMA = 'pyvoro2.effective-build-evidence.v1'


class BuildEvidenceError(RuntimeError):
    """The observed build does not establish the required properties."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True).encode('ascii')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_identity(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return {'path': str(path), 'size': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def expand_response(argv, cwd, windows=False, snapshots=None, active=()):
    """Expand the supported compiler response grammar, preserving order.

    The Windows adapter passes the raw command line as well. Command-line
    reconstruction is not substituted for the observed process invocation.
    """
    snapshots = {} if snapshots is None else snapshots
    expanded = []
    for arg in argv:
        if not arg.startswith('@'):
            expanded.append(arg)
            continue
        path = (Path(cwd) / arg[1:]).resolve(strict=True)
        if path in active:
            raise BuildEvidenceError('recursive response file')
        data = path.read_bytes()
        encoding = ('utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff'))
                    else 'utf-8-sig')
        try:
            content = data.decode(encoding)
        except UnicodeDecodeError as error:
            raise BuildEvidenceError('unsupported response encoding') from error
        identity = file_identity(path)
        identity.update(text=content, encoding=encoding,
                        bytes_base64=base64.b64encode(data).decode('ascii'))
        previous = snapshots.setdefault(str(path), identity)
        if previous['sha256'] != identity['sha256']:
            raise BuildEvidenceError('response file changed during observation')
        words = windows_words(content) if windows else shlex.split(content)
        expanded.extend(expand_response(words, cwd, windows, snapshots,
                                        (*active, path)))
    return expanded


def windows_words(command):
    """MSVC/CRT backslash-before-double-quote grammar, without shell parsing."""
    result, token = [], []
    quoted = False
    started = False
    i = 0
    while i < len(command):
        char = command[i]
        if char in ' \t\r\n' and not quoted:
            if started:
                result.append(''.join(token))
                token, started = [], False
            i += 1
            continue
        started = True
        if char == '\\':
            j = i
            while j < len(command) and command[j] == '\\':
                j += 1
            count = j - i
            if j < len(command) and command[j] == '"':
                token.extend('\\' * (count // 2))
                if count % 2:
                    token.append('"')
                else:
                    quoted = not quoted
                i = j + 1
            else:
                token.extend('\\' * count)
                i = j
            continue
        if char == '"':
            if quoted and i + 1 < len(command) and command[i + 1] == '"':
                token.append('"')
                i += 2
                continue
            quoted = not quoted
        else:
            token.append(char)
        i += 1
    if quoted:
        raise BuildEvidenceError('unterminated Windows argument quote')
    if started:
        result.append(''.join(token))
    return result


_PROOF_MACRO = re.compile(
    r'(?:__(?:(?:FLT|DBL|LDBL)_(?:EVAL_METHOD|MANT_DIG|MAX_EXP|MIN_EXP|'
    r'DECIMAL_DIG|HAS_DENORM|DENORM_MIN|MIN|MAX|EPSILON)|'
    r'SIZEOF_(?:INT|DOUBLE|LONG_DOUBLE)|FAST_MATH|FINITE_MATH_ONLY|'
    r'SSE\w*|AVX\w*|FMA\w*|x86_64|aarch64)__|_M_(?:FP\w*|X64|ARM64))$')
_NON_ARITHMETIC_F = {
    '-fPIC', '-fpic', '-fPIE', '-fpie', '-fno-pie', '-fno-PIE',
    '-fasynchronous-unwind-tables', '-funwind-tables', '-fno-unwind-tables',
    '-fexceptions', '-fcxx-exceptions', '-fno-exceptions',
    '-fno-omit-frame-pointer', '-fomit-frame-pointer', '-fno-inline',
    '-fvisibility=hidden', '-fvisibility=default', '-fvisibility-inlines-hidden',
    '-fstack-clash-protection', '-fstack-protector-strong', '-fstack-protector',
    '-fno-stack-protector', '-fcf-protection', '-fcf-protection=full',
    '-fcf-protection=branch', '-fcf-protection=none',
    '-fverbose-asm', '-fno-verbose-asm', '-fpreprocessed', '-fpch-preprocess',
    '-fworking-directory', '-fno-working-directory',
    '-fno-rounding-math', '-frounding-math', '-fno-gnu-unique',
    '-fno-integrated-cc1', '-fno-integrated-as', '-fintegrated-as',
    '-fcolor-diagnostics', '-fno-color-diagnostics', '-fno-caret-diagnostics',
}
_CLANG_NON_ARITHMETIC_F = {
    '-fdeprecated-macro', '-fblocks', '-fencode-extended-block-signature',
    '-fregister-global-dtors-with-atexit', '-fno-use-cxa-atexit',
    '-fno-implicit-modules', '-fno-implicit-module-maps',
    '-fno-modules', '-faddrsig', '-fno-addrsig',
    '-fdebug-compilation-dir', '-fcoverage-compilation-dir', '-ferror-limit',
    '-fmodule-file-home-is-cwd', '-fskip-odr-check-in-gmf',
    # Defaults observed in the actual Xcode 16.4 / AppleClang 17 child plans.
    # These control C++/ObjC ABI, diagnostics, stack checks and module disabling;
    # none grants unsafe FP algebra, contraction, excess precision or LTO.
    '-fno-strict-return', '-fobjc-msgsend-selector-stubs',
    '-faligned-alloc-unavailable',
    '-fcompatibility-qualified-id-block-type-checking',
    '-fvisibility-inlines-hidden-static-local-var',
    '-fbuiltin-headers-in-system-modules', '-fdefine-target-os-macros',
    '-fno-assume-unique-vtables', '-fstack-check', '-fno-cxx-modules',
    '-fcommon', '-fno-odr-hash-protocols',
}


def _check_control_option(arg, family):
    """Fail closed for unreviewed machine/optimization mechanisms."""
    if arg.startswith('-m'):
        allowed = {'-mavx', '-mfma', '-mno-avx', '-mno-fma', '-msse2',
                   '-msse4.2', '-m64', '-mfpmath=sse', '-mlong-double-80',
                   '-march=x86-64', '-mtune=generic', '-mtune=core2'}
        if family == 'clang':
            allowed.update({'-mrelocation-model', '-mframe-pointer',
                            '-mconstructor-aliases', '-munwind-tables',
                            '-main-file-name', '-mdarwin-stkchk-strong-link',
                            '-mframe-pointer=all', '-mframe-pointer=non-leaf',
                            '-mframe-pointer=none'})
        deployment = (family == 'clang' and re.fullmatch(
            r'-mmacosx-version-min=\d+\.\d+(?:\.\d+)?', arg))
        if arg not in allowed and not deployment:
            raise BuildEvidenceError(
                'unreviewed machine/excess-evaluation control: ' + arg)
    if arg.startswith('-f'):
        allowed = arg in _NON_ARITHMETIC_F
        allowed |= arg.startswith(('-fdiagnostics-', '-ffile-prefix-map=',
                                   '-fdebug-prefix-map=', '-fmacro-prefix-map='))
        allowed |= arg in (
            '-fsanitize=address,undefined,float-cast-overflow',
            '-fsanitize=undefined,address,float-cast-overflow',
            '-fno-sanitize-recover=all',
        )
        if family == 'clang':
            allowed |= arg in _CLANG_NON_ARITHMETIC_F
            allowed |= arg.startswith(('-fgnuc-version=', '-fobjc-runtime=',
                                       '-fmax-type-align=', '-funwind-tables=',
                                       '-fdebug-compilation-dir=',
                                       '-fcoverage-compilation-dir='))
            allowed |= arg in ('-ffp-exception-behavior=ignore',
                               '-ffp-exception-behavior=strict')
        if not allowed:
            raise BuildEvidenceError('unreviewed optimization control: ' + arg)


def _check_proof_macro_options(argv):
    for index, arg in enumerate(argv):
        if arg in ('-D', '-U', '/D', '/U', '/d', '/u'):
            name = argv[index + 1] if index + 1 < len(argv) else ''
        elif arg.startswith(('-D', '-U', '/D', '/U', '/d', '/u')):
            name = arg[2:]
        else:
            continue
        if _PROOF_MACRO.match(name.partition('=')[0]):
            raise BuildEvidenceError('compiler-owned proof premise macro override')


def effective_options(argv, family, *, link=False, backend=False):
    """Evaluate the closed reviewed option subset in its actual order.

    Unknown optimization/plugin/specs mechanisms require another adapter.
    A later strict reset overrides early unsafe input. Backend evidence,
    dependencies, actual preprocessed source, and exact discriminators remain
    independently required; this parser is not the qualification argument.

    The GNU adapter retains its default trapping-math premise. AppleClang's
    default exception behavior differs: its adapter requires actual optimized
    raw-guard inspection rather than asserting a universal trapping flag.
    Both admit nearest-only evaluation; -fno-rounding-math is the compiler
    default, not permission to enter the numerical body under hostile state.
    """
    _check_proof_macro_options(argv)
    if family == 'msvc':
        fp, lto = None, False
        for option in argv:
            option = option.lower()
            if option.startswith('-'):
                option = '/' + option[1:]
            if option.startswith(('/fp:', '-fp:')):
                fp = option.split(':', 1)[1]
            if option in ('/gl', '-gl'):
                lto = True
            if option in ('/gl-', '-gl-'):
                lto = False
            if option.startswith(('/ltcg', '-ltcg')):
                lto = option not in ('/ltcg:off', '-ltcg:off')
            if option.startswith(('/b1', '/b2', '/d1', '/d2')):
                raise BuildEvidenceError('opaque MSVC backend override')
            if (option.startswith(('/yu', '/yc', '/reference', '/headerunit',
                                   '/interface', '/ifc', '/experimental:module',
                                   '/translateinclude')) or
                    option.startswith('/fp') and not option.startswith('/fp:')):
                raise BuildEvidenceError('unreviewed serialized MSVC frontend input')
        if lto:
            raise BuildEvidenceError('effective LTO/IPO is forbidden')
        if not link and fp != 'strict':
            raise BuildEvidenceError('effective MSVC /fp:strict is missing')
        if link and not any(a.lower() in ('/ltcg:off', '-ltcg:off')
                            for a in argv):
            raise BuildEvidenceError('explicit /LTCG:OFF is missing')
        return {'strict_fp': True, 'contraction': 'off', 'lto': False}

    unsafe = {name: False for name in (
        'reassociation', 'finite_only', 'discard_signed_zero', 'reciprocal',
        'unsafe_math', 'approximate_functions')}
    contraction, strict_seen, lto, trapping = None, False, False, True
    if backend and family == 'clang':
        strict_seen = True  # Explicit cc1 algebra flags below are authoritative.
    skip = False
    for index, arg in enumerate(argv):
        if skip:
            skip = False
            continue
        if arg == '-mllvm':
            # Apple's ordinary optimized pipeline requests cleanup after loop
            # vectorization. This exact pass toggle retains the strict FP
            # semantics; arbitrary LLVM backend options remain opaque.
            if (family != 'clang' or not backend or index + 1 >= len(argv) or
                    argv[index + 1] != '-extra-vectorizer-passes'):
                raise BuildEvidenceError('unreviewed LLVM backend control')
            skip = True
            continue
        if arg in ('-ffast-math', '-Ofast', '-funsafe-math-optimizations'):
            unsafe = dict.fromkeys(unsafe, True)
            trapping = False
        elif arg == '-fno-fast-math':
            unsafe = dict.fromkeys(unsafe, False)
            strict_seen = True
            trapping = True
        elif arg == '-fno-trapping-math':
            trapping = False
        elif arg == '-ftrapping-math':
            trapping = True
        elif arg in ('-fassociative-math', '-mreassociate'):
            unsafe['reassociation'] = True
        elif arg == '-fno-associative-math':
            unsafe['reassociation'] = False
        elif arg in ('-ffinite-math-only', '-menable-no-infs',
                     '-menable-no-nans'):
            unsafe['finite_only'] = True
        elif arg == '-fno-finite-math-only':
            unsafe['finite_only'] = False
        elif arg == '-fno-signed-zeros':
            unsafe['discard_signed_zero'] = True
        elif arg == '-fsigned-zeros':
            unsafe['discard_signed_zero'] = False
        elif arg in ('-freciprocal-math', '-mrecip', '-mrecip=all'):
            unsafe['reciprocal'] = True
        elif arg == '-fno-reciprocal-math':
            unsafe['reciprocal'] = False
        elif arg == '-fapprox-func':
            unsafe['approximate_functions'] = True
        elif arg == '-menable-unsafe-fp-math':
            unsafe['unsafe_math'] = True
        elif (arg in ('-mpc32', '-mpc64', '-mdaz-ftz') or
              arg.startswith(('-fdenormal-fp-math=', '-fdenormal-fp-math-f32='))
              and not arg.endswith('=ieee')):
            raise BuildEvidenceError('unsafe evaluation/subnormal mode')
        elif arg.startswith('-ffp-contract='):
            contraction = arg.partition('=')[2]
        elif arg == '-fno-lto':
            lto = False
        elif arg == '-flto' or arg.startswith('-flto='):
            lto = True
        elif arg.startswith(('-fplugin', '-specs=', '--specs=', '--config',
                             '-fpass-plugin', '-fprofile-use', '-fauto-profile')):
            raise BuildEvidenceError('opaque compiler extension/configuration')
        else:
            _check_control_option(arg, family)
    if lto:
        raise BuildEvidenceError('effective LTO/IPO is forbidden')
    if contraction != 'off':
        raise BuildEvidenceError('effective proof-sensitive contraction is not off')
    if any(unsafe.values()):
        raise BuildEvidenceError('effective unsafe FP properties: ' +
                                 ', '.join(k for k, v in unsafe.items() if v))
    if family == 'gnu' and not trapping:
        raise BuildEvidenceError('effective GNU -fno-trapping-math is forbidden')
    if not strict_seen:
        raise BuildEvidenceError('no effective strict FP reset was observed')
    return {'strict_fp': True, 'contraction': 'off', 'lto': False,
            'unsafe_properties': unsafe}


_SOURCE_TOKEN = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S)
_PRAGMA_FP = re.compile(
    r'#\s*pragma[^\n]*(?:GCC\s+(?:optimize|target|push_options|pop_options|'
    r'reset_options|pch_preprocess)|clang\s+(?:fp|attribute)|float_control|fenv_access|'
    r'optimize|fp_contract|FP_CONTRACT|FENV_ACCESS)[^\n]*', re.I)
_ATTRIBUTE = re.compile(r'__attribute(?:__)?\s*\(\s*\(|\[\[')
_LOCAL_CONTROL = re.compile(
    r'\b(?:optimize|target|target_clones|__optimize__|__target__|'
    r'__target_clones__)\s*\(')
# This is an allow grammar, not a list of dangerous substrings. Unrecognized
# local controls, adjacent/escaped option strings and target attributes are
# opaque even when their particular spelling happens to look harmless.
_STRICT_PRAGMAS = [re.compile(pattern, re.I) for pattern in (
    r'#\s*pragma\s+GCC\s+optimize\s*\(\s*"no-fast-math"\s*\)\s*',
    r'#\s*pragma\s+STDC\s+FP_CONTRACT\s+OFF\s*',
    r'#\s*pragma\s+STDC\s+FENV_ACCESS\s+ON\s*',
    r'#\s*pragma\s+fp_contract\s*\(\s*off\s*\)\s*',
    r'#\s*pragma\s+clang\s+fp\s+contract\s*\(\s*off\s*\)\s*',
    r'#\s*pragma\s+float_control\s*\(\s*(?:precise|except)\s*,\s*on'
    r'(?:\s*,\s*push)?\s*\)\s*',
    r'#\s*pragma\s+float_control\s*\(\s*(?:push|pop)\s*\)\s*',
)]


def _pragma_calls(text):
    """Preserve equivalent MSVC/C pragma forms in preprocessor output."""
    for match in re.finditer(r'\b(__pragma|_Pragma)\s*\(', text):
        depth, quote, escape, end = 1, False, False, match.end()
        while end < len(text) and depth:
            char = text[end]
            if quote:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    quote = False
            elif char == '"':
                quote = True
            elif char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            end += 1
        if depth:
            raise BuildEvidenceError('incomplete preprocessed pragma call')
        body = text[match.end():end - 1].strip()
        if match[1] == '_Pragma':
            # Standard destringizing removes only escaped quotes/backslashes.
            # Other literal grammars require review, never substring guessing.
            if not re.fullmatch(r'"(?:\\[\\"]|[^"\\])*"', body, re.S):
                raise BuildEvidenceError('opaque preprocessed pragma literal')
            body = re.sub(r'\\([\\"])', r'\1', body[1:-1])
        yield '#pragma ' + body.replace('\n', ' ')


def unsafe_source_directives(text, *, macro_overrides_only=False):
    # Join preprocessing continuations and remove comments without erasing
    # quoted optimization options. Actual saved preprocessed input is checked
    # as well, so macro-generated pragmas/attributes remain visible.
    text = text.replace('\\\n', '')
    text = _SOURCE_TOKEN.sub(
        lambda m: ' ' if m.group().startswith(('//', '/*')) else m.group(), text)
    directives = [m.group() for m in re.finditer(
        r'^\s*#\s*(?:define|undef)\s+([^\s(]+)[^\n]*', text, re.M)
        if _PROOF_MACRO.match(m.group(1))]
    if macro_overrides_only:
        return directives
    directives.extend(m.group() for m in _PRAGMA_FP.finditer(text)
                      if not any(p.fullmatch(m.group()) for p in _STRICT_PRAGMAS))
    directives.extend(pragma for pragma in _pragma_calls(text)
                      if _PRAGMA_FP.fullmatch(pragma) and
                      not any(p.fullmatch(pragma) for p in _STRICT_PRAGMAS))
    for match in _ATTRIBUTE.finditer(text):
        opening, closing = ('[', ']') if match.group() == '[[' else ('(', ')')
        depth, quote, escape = 2, None, False
        end = match.end()
        while end < len(text) and depth:
            char = text[end]
            if quote:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == quote:
                    quote = None
            elif char in ('"', "'"):
                quote = char
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
            end += 1
        attribute = text[match.start():end]
        if _LOCAL_CONTROL.search(attribute):
            if depth:
                raise BuildEvidenceError('incomplete source-local optimize attribute')
            directives.append(attribute)
    return directives


def _verify_identity(identity, label):
    try:
        current = file_identity(identity['path'])
    except (OSError, KeyError) as error:
        raise BuildEvidenceError(f'missing {label}') from error
    if current != identity:
        raise BuildEvidenceError(f'changed {label}: {identity["path"]}')


def _check_gnu_object(identity):
    """Reject bitcode/LTO and non-object substitutions in observed ELF output."""
    data = Path(identity['path']).read_bytes()
    if data[:4] != b'\x7fELF' or data[4:6] != b'\x02\x01':
        raise BuildEvidenceError('unsupported/non-ELF native object')
    if struct.unpack_from('<HH', data, 16) != (1, 62):
        raise BuildEvidenceError('object is not ELF x86-64 relocatable code')
    section_offset = struct.unpack_from('<Q', data, 40)[0]
    entry_size, count, names_index = struct.unpack_from('<HHH', data, 58)
    if entry_size != 64 or not count or names_index >= count:
        raise BuildEvidenceError('unsupported/incomplete ELF section evidence')
    try:
        name_header = section_offset + names_index * entry_size
        offset, size = struct.unpack_from('<QQ', data, name_header + 24)
        strings = data[offset:offset + size]
        for index in range(count):
            name_offset = struct.unpack_from(
                '<I', data, section_offset + index * entry_size)[0]
            name = strings[name_offset:].split(b'\0', 1)[0]
            if name.startswith((b'.gnu.lto_', b'.llvm.lto', b'.llvmbc')):
                raise BuildEvidenceError('LTO/bitcode in actual object output')
    except struct.error as error:
        raise BuildEvidenceError('truncated ELF object evidence') from error


def verify_build(records_dir: Path, source_root: Path) -> dict:
    """Return build evidence only; never an artifact qualification record."""
    paths = sorted(Path(records_dir).glob('*.json'))
    if not paths:
        raise BuildEvidenceError('no command records')
    units, links, records, tools, inputs = {}, [], [], {}, {}
    observers, compilers, linkers, providers = set(), {}, {}, {}
    versions, targets, families = set(), set(), set()
    system_loaders = []
    source_root = Path(source_root).resolve(strict=True)
    for path in paths:
        try:
            row = json.loads(path.read_text())
            expected = row.pop('record_sha256')
            if expected != digest(row):
                raise ValueError('digest')
        except (ValueError, KeyError) as error:
            raise BuildEvidenceError(f'invalid record digest: {path}') from error
        if row.get('schema') != SCHEMA:
            raise BuildEvidenceError('unknown command schema')
        # Check the actual terminal properties before wrapper completeness, so
        # negative controls expose a late unsafe override rather than hide it.
        actual = row.get('effective_argv')
        if actual:
            effective_options(actual, row['family'],
                              link=row.get('kind') == 'link')
        for invocation in row.get('invocations', []):
            if invocation.get('role') == 'compiler_backend':
                effective_options(invocation['expanded_argv'], row['family'],
                                  backend=True)
            elif invocation.get('role') == 'linker':
                from .link_provenance import verify_linker_options
                verify_linker_options(invocation['expanded_argv'], row['family'])
        if not row.get('complete') or row.get('exit_code') != 0:
            raise BuildEvidenceError('incomplete effective-command evidence: ' +
                                     '; '.join(row.get('problems', [])))
        if row.get('adapter') not in (
                'linux-strace-v1', 'gnu-child-wrapper-v1',
                'apple-plan-replay-v1', 'windows-debug-v1'):
            raise BuildEvidenceError('unknown observation adapter')
        if not actual or not row.get('invocations'):
            raise BuildEvidenceError('incomplete actual invocation evidence')
        if row.get('opaque_wrappers'):
            raise BuildEvidenceError('opaque/unreviewed compiler wrapper')
        for item in row.get('evidence_files', []):
            _verify_identity(item, 'evidence')
        for item in row.get('tools', []):
            _verify_identity(item, 'tool')
            tools[item['path']] = item
        for item in row.get('toolchain_inputs', []):
            _verify_identity(item, 'toolchain input')
            tools[item['path']] = item
        for invocation in row['invocations']:
            identity = invocation['executable']
            if invocation['role'] == 'compiler_driver':
                compilers[identity['sha256']] = identity
            elif invocation['role'] == 'linker':
                linkers[identity['sha256']] = identity
        if row.get('driver_version'):
            versions.add(row['driver_version'])
        if row.get('driver_target'):
            targets.add(row['driver_target'])
        observers.add(row['adapter'])
        if row['family'] == 'clang':
            baselines = row.get('system_loader_evidence')
            if (not baselines or any(p.get('schema') != 'pyvoro2.apple-system-loader.v1'
                                     for p in baselines)):
                raise BuildEvidenceError('missing actual Apple loader/cache provenance')
            system_loaders.extend(baselines)
        families.add(row['family'])
        for provider in row.get('input_providers', []):
            previous = providers.setdefault(provider['provider_id'], provider)
            if previous != provider:
                raise BuildEvidenceError('conflicting header provider provenance')
        for item in row.get('dependencies', []):
            _verify_identity(item, 'input')
            inputs[item['path']] = item
            # Raw dependencies retain byte identity and cannot redefine the
            # compiler-owned proof macros. Other local controls are judged in
            # the actual expanded input below: inactive platform branches are
            # not effective optimizer settings (e.g. CPython 3.10 pyport.h).
            if unsafe_source_directives(Path(item['path']).read_text(
                    errors='replace'), macro_overrides_only=True):
                raise BuildEvidenceError('source-local unsafe FP directive: ' +
                                         item['path'])
        for item in row.get('preprocessed', []):
            _verify_identity(item, 'preprocessed source')
            if unsafe_source_directives(Path(item['path']).read_text(
                    errors='replace')):
                raise BuildEvidenceError('source-local unsafe FP directive')
        if row['kind'] == 'compile':
            if not row.get('dependencies') or not row.get('preprocessed'):
                raise BuildEvidenceError('missing actual source/dependency evidence')
            macros = row.get('macros', {})
            if macros.get('__FLT_EVAL_METHOD__') != '0':
                raise BuildEvidenceError('excess-evaluation mismatch')
            if macros.get('__DBL_MANT_DIG__') != '53':
                raise BuildEvidenceError('binary64 representation mismatch')
            if row['family'] == 'gnu' and macros.get('__LDBL_MANT_DIG__') != '64':
                raise BuildEvidenceError('unreviewed GNU long-double evaluation ABI')
            if row['family'] != 'msvc' and macros.get('__SIZEOF_INT__') != '4':
                raise BuildEvidenceError('native integer-width mismatch')
            if not any(i.get('role') == 'compiler_backend'
                       for i in row['invocations']):
                raise BuildEvidenceError('missing actual compiler backend')
            output = row['output']
            _verify_identity(output, 'object')
            if row['family'] == 'gnu':
                _check_gnu_object(output)
            if output['path'] in units:
                raise BuildEvidenceError('multiple builds of one object in evidence')
            units[output['path']] = {
                'source': row['source'], 'output': output,
                'dependencies': row['dependencies'], 'record_sha256': expected,
                'command': {'argv': actual, 'cwd': row['cwd'],
                            'environment': row['environment']},
            }
        elif row['kind'] == 'link':
            if not any(i.get('role') == 'linker' for i in row['invocations']):
                raise BuildEvidenceError('missing actual linker invocation')
            _verify_identity(row['output'], 'link output')
            links.append(row)
        else:
            raise BuildEvidenceError('unsupported command role')
        records.append(expected)
    if not links:
        raise BuildEvidenceError('no final link record')
    components = {}
    linked_objects = set()
    for row in links:
        from .link_provenance import verify_link_closure
        objects = verify_link_closure(row, units)
        for item in objects:
            if item['path'] not in units or units[item['path']]['output'] != item:
                raise BuildEvidenceError('unobserved/mismatched linked object')
            linked_objects.add(item['path'])
        output = row['output']
        name = Path(output['path']).name.split('.')[0]
        if name in components:
            raise BuildEvidenceError('ambiguous component link output')
        components[name] = {
            'output': output,
            'translation_units': sorted(units[p['path']]['source'] for p in objects),
        }
    if linked_objects != set(units):
        raise BuildEvidenceError('unlinked translation unit in evidence')
    if (len(compilers) != 1 or len(linkers) != 1 or len(families) != 1 or
            len(versions) != 1 or len(targets) != 1):
        raise BuildEvidenceError('missing/ambiguous compiler/linker provenance')
    family = next(iter(families))
    target = next(iter(targets))
    from .input_provenance import classify_external_inputs
    external = classify_external_inputs(list(providers.values()),
                                        list(inputs.values()), source_root)
    if family == 'gnu' and target.startswith('x86_64') and 'linux' in target:
        runtime_adapter = 'gnu-linux-x86_64-v1'
    elif family == 'clang' and 'apple' in target:
        runtime_adapter = (
            'appleclang-darwin-arm64-v1'
            if target.startswith(('arm64', 'aarch64')) else
            'appleclang-darwin-x86_64-v1')
    elif family == 'msvc' and target == 'x86_64-windows-msvc':
        runtime_adapter = 'msvc-win32-x86_64-v1'
    else:
        raise BuildEvidenceError('unsupported target observation adapter')
    evidence = {
        'schema': EVIDENCE_SCHEMA,
        'record_sha256s': sorted(records),
        'source_root': str(source_root),
        'adapter': runtime_adapter, 'observation_adapters': sorted(observers),
        'toolchain': {
            'family': {'gnu': 'GNU', 'clang': 'AppleClang', 'msvc': 'MSVC'}[family],
            'version': next(iter(versions)), 'target': target,
            'compiler_sha256': next(iter(compilers)),
            'linker_sha256': next(iter(linkers)),
        },
        'properties': {
            'strict_fp': True, 'contraction': 'off', 'lto': False,
            'flt_eval_method': 0, 'double_digits': 53,
            'rounding_assumption': 'nearest_only',
            'trap_ordering_policy': {
                'gnu': 'trapping_math_and_inspected_guard',
                'clang': 'apple_adapter_and_inspected_guard',
                'msvc': 'fp_strict_and_inspected_guard',
            }[family],
        },
        'translation_units': sorted(units.values(), key=lambda r: r['source']),
        'components': components,
        'link_outputs': [r['output'] for r in links],
        'tools': sorted(tools.values(), key=lambda r: r['path']),
        'dependencies': sorted(inputs.values(), key=lambda r: r['path']),
        'input_providers': sorted(providers.values(), key=lambda p: p['provider_id']),
        'external_inputs': sorted(external, key=lambda p: p['identity']['path']),
        'system_loader_evidence': system_loaders,
    }
    evidence['evidence_sha256'] = digest(evidence)
    return evidence
