"""Independent source-sensitive arithmetic and optimized-entry evidence.

These checks corroborate the controlled toolchain/source argument. They are
not a claim that a passing toy expression establishes compiler correctness.
Expected bit strings below are frozen independent expectations from #88's
predecessor characterization; they are not obtained from the candidate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .effective_build import (BuildEvidenceError, SCHEMA, canonical, digest,
                              effective_options, file_identity,
                              unsafe_source_directives)
from .record_command import query_options
from .guard_symbols import (PROTECTED_ENTRY_COUNTS, call_target, parse_disassembly,
                            protected_entries)

STRICT_BITS = '4013fffffb000000'
UNSAFE_BITS = '4013fffffb000001'
STANDARD_FIXTURES = {
    'standard-xy-v1': ['0x1.ffffff4p-1', '0x1.ffffffep0', '0x0p0'],
    'standard-xz-v1': ['0x1.ffffff4p-1', '0x0p0', '0x1.ffffffep0'],
}
_RUNTIME_ENVIRONMENT = ('LD_PRELOAD', 'ASAN_OPTIONS', 'UBSAN_OPTIONS',
                        'PYVORO2_NATIVE_TEST_SANITIZERS')
_SANITIZER_BUNDLES = {
    '-fsanitize=address,undefined,float-cast-overflow',
    '-fsanitize=undefined,address,float-cast-overflow',
}
_SANITIZER_DISABLE = '-fno-sanitize=all'


def tool_environment(environment):
    """Keep runtime sanitizer preloads out of compiler/inspection children."""
    return {key: value for key, value in environment.items()
            if key not in _RUNTIME_ENVIRONMENT}


_HARNESS = r'''
#if defined(PYVORO2_UNSAFE_POWER) && defined(_MSC_VER)
#pragma float_control(precise, off)
#pragma fp_contract(off)
#endif
#include "voro++.hh"
#include "v_compute.cc"
#include <cstdint>
#include <cstdio>
#include <cstring>

static unsigned long long bits(double value) {
    std::uint64_t result;
    std::memcpy(&result, &value, sizeof result);
    return static_cast<unsigned long long>(result);
}

// The unchanged vendored compute template calls this overload with its
// actual plane operands. No squared-displacement expression is copied here.
struct Observed : voro::voronoicell_neighbor {
    double offset = -1;
    int count = 0;
    bool nplane(double x, double y, double z, double rsq, int id) {
        offset = rsq;
        ++count;
        return voro::voronoicell_neighbor::nplane(x, y, z, rsq, id);
    }
};

struct Radius : voro::radius_poly {
    void run(double radius) {
        double storage[8] = {0, 0, 0, radius, 0, 0, 0, radius};
        double* blocks[] = {storage};
        ppr = blocks;
        max_radius = radius;
        r_init(0, 0);
        const double ordinary = r_scale(1., 0, 1);
        double checked = 1.;
        const bool cuts = r_scale_check(checked, 16., 0, 1);
        std::printf("power %a %016llx %016llx %d\n", radius,
                    bits(ordinary), bits(checked), static_cast<int>(cuts));
    }
};

static int standard(const char* name, double x, double y, double z) {
    voro::container container(-4., 4., -4., 4., -4., 4.,
                              1, 1, 1, false, false, false, 8);
    container.put(0, 0., 0., 0.);
    container.put(1, x, y, z);
    voro::c_loop_all loop(container);
    if (!loop.start()) return 2;
    do {
        Observed cell;
        if (!container.compute_cell(cell, loop) || cell.count != 1) return 3;
        std::printf("%s %d %016llx\n", name, loop.pid(), bits(cell.offset));
    } while (loop.inc());
    return 0;
}

int main() {
    volatile double dx = 0x1.ffffff4p-1, dy = 0x1.ffffffep0;
    if (int result = standard("standard", dx, dy, 0.)) return result;
    // Independent companion exposes fusion of the final nonzero square.
    // Keep the original XY fixture and its archived expectations unchanged.
    if (int result = standard("standard-xz", dx, 0., dy)) return result;
    Radius radius;
    for (int index = 0; index != 3; ++index) {
        volatile double value = 134217728. + index;
        radius.run(value);
    }
}
'''


def _trapping_fp(mnemonic):
    # Register moves, SIMD integer instructions, FNSTCW/FNSTSW/STMXCSR, and
    # ARM MRS are raw transport/control inspection, not floating evaluation.
    if mnemonic in ('fnstcw', 'fnstsw', 'fnstenv', 'fnstsave'):
        return False
    if mnemonic.startswith('f'):
        return mnemonic not in ('fmov',)
    return bool(re.match(
        r'v?(?:add|sub|mul|div|sqrt|min|max|comi|ucomi|cmp|round|rcp|rsqrt|'
        r'hadd|hsub|addsub|dp)'
        r'(?:ss|sd|ps|pd)$|v?cvt|v?fm(?:add|sub)|v?fnm(?:add|sub)', mnemonic))


_SAFE_X86 = re.compile(
    r'(?:mov(?:abs|zx|sx|sxd|dqu|dqa|ups|upd|aps|apd|ss|sd|d|q)?|'
    r'lea|add|adc|sub|sbb|and|or|xor|test|cmp|inc|dec|neg|not|'
    r'shl|shr|sar|sal|rol|ror|push|pop|imul|mul|idiv|div|'
    r'bt|bts|btr|btc|bsf|bsr|xchg|nop|ret|leave|stos|lods|scas|movs)'
    r'[bwlq]?$|mov[zs][bwl][wlq]$|(?:j|set|cmov)[a-z]+$|'
    r'v?(?:mov(?:dqu|dqa|ups|upd|aps|apd|ss|sd|d|q)|'
    r'pxor|pand|por|xorps|xorpd|andps|andpd|orps|orpd|'
    r'punpck[a-z]+|unpck[a-z]+|pshuf[a-z]+|psll[a-z]+|psrl[a-z]+)$|'
    r'callq?$|fnstcw$|fnstsw$|stmxcsr$|endbr64$|vzeroupper$|pinsrb$')
_SAFE_ARM = {
    'adc', 'adcs', 'add', 'adds', 'adr', 'adrp', 'and', 'ands', 'asr',
    'bic', 'bics', 'ccmp', 'cinc', 'cinv', 'clz', 'cmn', 'cmp', 'cneg',
    'csel', 'cset', 'csetm', 'csinc', 'csinv', 'csneg', 'eor', 'extr',
    'lsl', 'lsr', 'mov', 'movk', 'movn', 'movz', 'neg', 'negs', 'orn',
    'orr', 'rbit', 'rev', 'ror', 'sbfm', 'sbfiz', 'sbfx', 'sub', 'subs',
    'tst', 'ubfm', 'ubfiz', 'ubfx', 'ldp', 'ldr', 'ldrb', 'ldrh',
    'ldrsw', 'ldur', 'ldurb', 'ldurh', 'stp', 'str', 'strb', 'strh',
    'stur', 'sturb', 'sturh', 'mrs', 'fmov', 'movi', 'dup', 'ins', 'umov',
    'b', 'bl', 'blr', 'br', 'cbz', 'cbnz', 'tbz', 'tbnz', 'ret', 'nop',
    'pacibsp', 'paciasp', 'autibsp', 'autiasp', 'bti',
}


def _check_instruction(instruction, context):
    if instruction.get('parse_error'):
        raise BuildEvidenceError('unparsed instruction in ' + context + ': ' +
                                 instruction['parse_error'])
    mnemonic = instruction['mnemonic']
    if _trapping_fp(mnemonic):
        raise BuildEvidenceError('floating operation in ' + context)
    if not (_SAFE_X86.fullmatch(mnemonic) or mnemonic in _SAFE_ARM or
            mnemonic.startswith('b.')):
        raise BuildEvidenceError('unreviewed instruction in ' + context +
                                 ': ' + mnemonic)


def inspect_guard_symbols(symbols, *, require_dispatch=True, source_name=None):
    """Prove the selected emitted entries reach a raw guard before FP work.

    Every reachable path before the guard is inspected. Unrecognized calls,
    indirect branches, branches outside the selected symbol, and missing
    instruction/symbol data refuse rather than become a successful report.
    """
    inspections = [name for name in symbols if re.search(
        r'(?:^| )pyvoro2::native_runtime::inspect\((?:void)?\)$', name)]
    refusals = [name for name in symbols if re.search(
        r'(?:^| )pyvoro2::native_runtime::require_environment\((?:void)?\)$', name)]
    entries = protected_entries(symbols, source_name=source_name)
    owners = {name: entry for entry in entries
              for name in entry['callbacks'] + entry['operators']}
    dispatches = sorted(owners)
    if not inspections or not refusals or require_dispatch and not dispatches:
        raise BuildEvidenceError('missing inspect/require/Dispatch disassembly symbols')
    for name in inspections:
        body = symbols[name]
        if not body:
            raise BuildEvidenceError('empty raw-control inspection symbol')
        for instruction in body:
            mnemonic = instruction['mnemonic']
            _check_instruction(instruction, 'raw inspect: ' + name)
            if mnemonic in ('call', 'callq', 'bl', 'blr'):
                raise BuildEvidenceError('unreviewed helper call in raw inspect')
        if not any(i['mnemonic'] in ('fnstcw', 'stmxcsr', 'mrs') for i in body):
            raise BuildEvidenceError('raw-control instruction missing')
    for name in refusals:
        body = symbols[name]
        for index, instruction in enumerate(body):
            # MSVC emits a final INT3 byte after the nonreturning throw call.
            # Exempt only this observed COFF padding from the whole-body FP
            # scan. The reachable pre-inspect CFG below still rejects INT3.
            previous = body[index - 1] if index else None
            if (index == len(body) - 1 and instruction['mnemonic'] == 'int3'
                    and not instruction.get('parse_error')
                    and not instruction['prefixes'] and previous
                    and not previous.get('parse_error')
                    and previous['mnemonic'] in ('call', 'callq')
                    and previous['relocation_kind'] == 'IMAGE_REL_AMD64_REL32'
                    and call_target(previous) == '_CxxThrowException'
                    and instruction['address'] == previous['address'] + 5):
                continue
            _check_instruction(instruction, 'raw refusal: ' + name)
    for name in dispatches + refusals:
        body = symbols[name]
        if not body:
            raise BuildEvidenceError('empty Dispatch symbol')
        by_address = {i['address']: index for index, i in enumerate(body)}
        pending, seen, guarded = [0], set(), False
        boundaries = set(inspections if name in refusals else refusals)
        if name in owners and name in owners[name]['callbacks']:
            # A callback may delegate to its own out-of-line Dispatch. Every
            # such body is independently inspected below; arbitrary wrappers
            # or another registration cannot establish this entry's boundary.
            boundaries.update(owners[name]['operators'])
        while pending:
            index = pending.pop()
            if index in seen:
                continue
            seen.add(index)
            instruction = body[index]
            mnemonic = instruction['mnemonic']
            operands = instruction['relocation'] or instruction['operands']
            _check_instruction(instruction, 'before guard: ' + name)
            target_symbol = call_target(instruction)
            if mnemonic in ('call', 'callq', 'bl', 'blr'):
                if target_symbol in boundaries:
                    guarded = True
                    continue
                if target_symbol not in ('__chkstk', '___chkstk_ms', '__chkstk_ms'):
                    raise BuildEvidenceError(
                        'unreviewed call before guard: ' + operands)
            if mnemonic.startswith('ret'):
                raise BuildEvidenceError('Dispatch return bypasses raw guard')
            is_branch = (mnemonic.startswith('j') or mnemonic == 'b' or
                         mnemonic.startswith('b.') or mnemonic in
                         ('cbz', 'cbnz', 'tbz', 'tbnz', 'br'))
            unconditional = mnemonic in ('jmp', 'jmpq', 'b', 'br')
            if is_branch:
                if unconditional and target_symbol in boundaries:
                    guarded = True
                    continue
                if instruction['relocation']:
                    raise BuildEvidenceError('external branch before guard')
                target = re.search(r'(?:^|,\s*)(?:0x)?([0-9a-fA-F]+)\s*(?:<|$)',
                                   instruction['operands'])
                if not target or int(target.group(1), 16) not in by_address:
                    raise BuildEvidenceError('unresolved branch before guard')
                pending.append(by_address[int(target.group(1), 16)])
            if not unconditional:
                if index + 1 >= len(body):
                    raise BuildEvidenceError('incomplete pre-guard instruction range')
                pending.append(index + 1)
        if not guarded:
            raise BuildEvidenceError('Dispatch never reaches raw guard')
    return {'inspect_symbols': sorted(inspections),
            'require_environment_symbols': sorted(refusals),
            'dispatch_symbols': sorted(dispatches),
            'protected_entries': entries, 'protected_entry_count': len(entries),
            'fp_before_guard': False}


def guard_disassembly(build_evidence, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    program = shutil.which('llvm-objdump') or shutil.which('objdump')
    if not program and os.name == 'nt':
        candidate = Path(r'C:\Program Files\LLVM\bin\llvm-objdump.exe')
        if candidate.exists():
            program = str(candidate)
    if not program:
        raise BuildEvidenceError('GNU/LLVM objdump is required for guard inspection')
    program = str(Path(program).resolve())
    selected = [row for row in build_evidence['translation_units']
                if Path(row['source']).name in PROTECTED_ENTRY_COUNTS]
    sources = {Path(row['source']).name for row in selected}
    if (len(selected) != len(PROTECTED_ENTRY_COUNTS) or
            sources != set(PROTECTED_ENTRY_COUNTS)):
        raise BuildEvidenceError('missing production guard/dispatch objects')
    reports, manifests = [], []
    for row in selected:
        identity = file_identity(row['output']['path'])
        if identity != row['output']:
            raise BuildEvidenceError('changed production object before disassembly')
        source_name = Path(row['source']).name
        item, outputs = {'source': source_name, 'object': identity}, {}
        for label, flags in (('', '-drC'), ('raw_', '-dr')):
            command = [program, flags, '--no-show-raw-insn', identity['path']]
            run = subprocess.run(command, text=True, capture_output=True,
                                 env=tool_environment(os.environ))
            if run.returncode:
                raise BuildEvidenceError('object disassembler failed: ' + run.stderr)
            path = output_dir / (source_name + '.' + label + 'disassembly.txt')
            path.write_text(run.stdout)
            item.update({label + 'disassembly': file_identity(path),
                         label + 'argv': command, label + 'exit_code': run.returncode})
            outputs[label] = run.stdout
        if file_identity(identity['path']) != identity:
            raise BuildEvidenceError('production object changed during disassembly')
        inspection = inspect_guard_symbols(
            parse_disassembly(outputs[''], outputs['raw_']),
            require_dispatch=source_name != 'fpguard.cpp', source_name=source_name)
        reports.append(inspection)
        manifests.append({**item, 'inspection': inspection})
    manifest = {'tool': file_identity(program), 'objects': manifests}
    (output_dir / 'guard-disassembly.json').write_bytes(canonical(manifest) + b'\n')
    return {
        'tool_sha256': manifest['tool']['sha256'],
        'disassembly_sha256': hashlib.sha256(canonical(manifest)).hexdigest(),
        **guard_summary(reports),
        'fp_before_guard': False, 'objects': manifests,
    }


def guard_summary(reports):
    return {**{key: sorted({name for report in reports for name in report[key]})
               for key in ('inspect_symbols', 'require_environment_symbols',
                           'dispatch_symbols')},
            'protected_entry_count': sum(r['protected_entry_count'] for r in reports)}


def _primitive_options(unit, family):
    args = unit['command']['argv']
    options = query_options(args, Path(unit['source']), family)
    # Remove only the observer's generated ld shim, identified by its paired
    # recorded GCC -wrapper callback directory. Ordinary -B inputs are kept.
    if '-wrapper' in args:
        wrapper = args[args.index('-wrapper') + 1].split(',')
        if '--directory' not in wrapper:
            raise BuildEvidenceError('unrecognized recorded GCC wrapper')
        directory = Path(wrapper[wrapper.index('--directory') + 1]).parent
        generated_prefix = '-B' + str(directory / 'linker') + os.sep
        options = [arg for arg in options if arg != generated_prefix]
    return options


def _control_compile_options(options, family, *, unsafe):
    """Keep strict instrumentation; explicitly optimize unsafe companions.

    Sanitizer checks alter the optimizer's legal unsafe reassociation choices.
    The fixed unsafe arithmetic expectations describe an optimized companion,
    not the instrumented production artifact. Only its compile receives this
    override; inherited production objects and runtime linkage are unchanged.
    """
    result = list(options)
    sanitizers = [arg for arg in options if arg.startswith('-fsanitize=')]
    if unsafe and sanitizers:
        if family != 'gnu' or any(arg not in _SANITIZER_BUNDLES
                                  for arg in sanitizers):
            raise BuildEvidenceError('unreviewed unsafe sanitizer companion')
        result.append(_SANITIZER_DISABLE)
    return result


def _control_arithmetic_argv(argv, family, *, sanitizer_companion):
    """Validate a companion-only override before checking its FP properties.

    The original vectors remain in the actual command records. The global
    production option checker continues to reject this override. A GNU driver
    and its observed cc1plus jobs must each record the inherited reviewed
    sanitizer bundle followed by exactly one disable and no later re-enable.
    """
    if not sanitizer_companion:
        return argv
    if family != 'gnu':
        raise BuildEvidenceError('unreviewed unsafe sanitizer companion')
    enabled, disabled = False, False
    for arg in argv:
        if arg.startswith('-fsanitize='):
            if arg not in _SANITIZER_BUNDLES or disabled:
                raise BuildEvidenceError('invalid unsafe companion instrumentation')
            enabled = True
        elif arg.startswith('-fno-sanitize='):
            if arg != _SANITIZER_DISABLE or not enabled or disabled:
                raise BuildEvidenceError('invalid unsafe companion instrumentation')
            enabled, disabled = False, True
    if enabled or not disabled:
        raise BuildEvidenceError('missing unsafe companion instrumentation override')
    return [arg for arg in argv if arg != _SANITIZER_DISABLE]


def _run_recorded(command, *, cwd, env, records, logs, name):
    recorder = Path(__file__).with_name('record_command.py')
    run = subprocess.run([sys.executable, str(recorder), '--output-dir',
                          str(records), '--', *command], cwd=cwd,
                         env=tool_environment(env),
                         text=True, capture_output=True)
    (logs / (name + '.stdout')).write_text(run.stdout)
    (logs / (name + '.stderr')).write_text(run.stderr)
    if run.returncode:
        raise BuildEvidenceError(
            'discriminator build failed: ' + name + ': ' + run.stderr)


def verify_control_records(records, build_evidence, *, unsafe,
                           inherited_objects=(), sanitizer_companion=False):
    """Bind control bytes to completed real jobs from the production toolchain.

    Unsafe controls must be rejected by the same effective-option checker;
    a successful executable exit alone is insufficient command evidence.
    """
    if sanitizer_companion and not unsafe:
        raise BuildEvidenceError('strict control cannot disable instrumentation')
    rows, files = [], {}
    known_tools = {r['sha256'] for r in build_evidence['tools']}
    for path in sorted(Path(records).glob('*.json')):
        row = json.loads(path.read_text())
        recorded_hash = row.pop('record_sha256', None)
        if (row.get('schema') != SCHEMA or recorded_hash != digest(row) or
                not row.get('complete') or row.get('exit_code') != 0 or
                row.get('opaque_wrappers')):
            raise BuildEvidenceError('incomplete discriminator command record')
        files[str(path)] = file_identity(path)
        for key in ('evidence_files', 'preprocessed', 'dependencies', 'tools',
                    'toolchain_inputs', 'object_inputs', 'link_inputs'):
            for identity in row.get(key, []):
                if file_identity(identity['path']) != identity:
                    raise BuildEvidenceError('changed discriminator ' + key)
                if key in ('evidence_files', 'preprocessed'):
                    files[identity['path']] = identity
                if (key == 'preprocessed' and not unsafe and
                        unsafe_source_directives(Path(identity['path']).read_text(
                            errors='replace'))):
                    raise BuildEvidenceError('source-local unsafe strict control')
        if file_identity(row['output']['path']) != row['output']:
            raise BuildEvidenceError('changed discriminator output')
        files[row['output']['path']] = row['output']
        roles = {i['role'] for i in row['invocations']}
        needed = 'compiler_backend' if row['kind'] == 'compile' else 'linker'
        if needed not in roles:
            raise BuildEvidenceError('missing actual discriminator ' + needed)
        for invocation in row['invocations']:
            identity = invocation['executable']
            if identity['sha256'] not in known_tools:
                raise BuildEvidenceError('discriminator toolchain changed')
            if invocation['role'] == 'linker':
                from .link_provenance import verify_linker_options
                verify_linker_options(invocation['expanded_argv'], row['family'])
        vectors = [(row['effective_argv'], False)]
        vectors.extend((i['expanded_argv'], True) for i in row['invocations']
                       if i['role'] == 'compiler_backend')
        for argv, backend in vectors:
            companion_compile = sanitizer_companion and row['kind'] == 'compile'
            arithmetic_argv = _control_arithmetic_argv(
                argv, row['family'], sanitizer_companion=companion_compile)
            rejected = False
            try:
                effective_options(arithmetic_argv, row['family'], backend=backend,
                                  link=row['kind'] == 'link')
            except BuildEvidenceError as error:
                if not unsafe or row['kind'] != 'compile' or not any(
                        reason in str(error) for reason in
                        ('contraction', 'unsafe FP properties', '/fp:strict')):
                    raise
                rejected = True
            if unsafe and row['kind'] == 'compile' and not rejected:
                # MSVC source-local power control has otherwise precise argv.
                source_unsafe = any(unsafe_source_directives(
                    Path(p['path']).read_text(errors='replace'))
                    for p in row['preprocessed'])
                if not source_unsafe:
                    raise BuildEvidenceError('unsafe control was not discriminated')
        rows.append(row)
    if sorted(row['kind'] for row in rows) != ['compile', 'link']:
        raise BuildEvidenceError('incomplete/ambiguous discriminator compile/link')
    compile_row = next(r for r in rows if r['kind'] == 'compile')
    link_row = next(r for r in rows if r['kind'] == 'link')
    expected = {r['path']: r for r in (*inherited_objects, compile_row['output'])}
    from .link_provenance import verify_link_closure
    verified_inputs = verify_link_closure(
        link_row, {path: {'output': item} for path, item in expected.items()})
    actual = {r['path']: r for r in verified_inputs}
    if actual != expected:
        raise BuildEvidenceError('discriminator link inputs changed')
    return sorted(files.values(), key=lambda r: r['path'])


def _link_options(options):
    result, paired = [], False
    for option in options:
        if paired:
            result.append(option)
            paired = False
        elif option in ('-arch', '-target', '-isysroot', '--sysroot'):
            result.append(option)
            paired = True
        elif option.startswith(('-fsanitize=', '-fno-sanitize=', '-stdlib=',
                                '-mmacosx-version-min=', '--sysroot=')):
            result.append(option)
    return result


def parse_power(stdout):
    rows = re.findall(r'^power (\S+) ([0-9a-f]{16}) ([0-9a-f]{16}) 1$',
                      stdout, re.M)
    return [(float.fromhex(radius).hex(), ordinary, checked)
            for radius, ordinary, checked in rows]


def parse_standard(stdout):
    """Require both owners of each fixed, independently derived fixture."""
    rows = {}
    labels = {'standard': 'standard-xy-v1', 'standard-xz': 'standard-xz-v1'}
    for line in stdout.splitlines():
        if not line.startswith('standard'):
            continue
        match = re.fullmatch(r'(standard(?:-xz)?) ([01]) ([0-9a-f]{16})', line)
        if match is None:
            raise BuildEvidenceError('unknown or malformed standard fixture')
        label, owner, value = match.groups()
        key = (labels[label], int(owner))
        if key in rows:
            raise BuildEvidenceError('duplicate standard fixture owner')
        rows[key] = value
    if set(rows) != {(name, owner) for name in STANDARD_FIXTURES for owner in (0, 1)}:
        raise BuildEvidenceError('missing standard fixture observation')
    return {name: [rows[name, owner] for owner in (0, 1)] for name in STANDARD_FIXTURES}


def standard_fixture_report(strict, unsafe, family):
    fixtures, discriminating = [], []
    for name, coordinates in STANDARD_FIXTURES.items():
        if strict[name] != [STRICT_BITS] * 2:
            raise BuildEvidenceError('strict actual-vendor standard fixture mismatch')
        if unsafe[name] not in ([STRICT_BITS] * 2, [UNSAFE_BITS] * 2):
            raise BuildEvidenceError('unreviewed unsafe standard fixture output')
        if unsafe[name] == [UNSAFE_BITS] * 2:
            discriminating.append(name)
        fixtures.append({'name': name, 'coordinates_hex': coordinates,
                         'expected_strict_bits': STRICT_BITS,
                         'expected_fused_bits': UNSAFE_BITS,
                         'strict_bits': strict[name], 'unsafe_bits': unsafe[name]})
    if not discriminating or family == 'GNU' and 'standard-xy-v1' not in discriminating:
        raise BuildEvidenceError('actual-vendor standard control did not discriminate')
    return fixtures, discriminating


def _control_compiler(build_evidence, unit, family):
    """Reuse the verified bytes without losing Apple's lexical driver mode."""
    compiler_hash = build_evidence['toolchain']['compiler_sha256']
    identity = next((r for r in build_evidence['tools']
                     if r['sha256'] == compiler_hash), None)
    if not identity:
        raise BuildEvidenceError('verified compiler executable identity missing')
    compiler = identity['path']
    if family == 'clang':
        name = Path(unit['command']['argv'][0]).name
        if name not in ('clang', 'clang++'):
            raise BuildEvidenceError('unreviewed production Apple compiler spelling')
        # Production may name /usr/bin's developer selector; use its actual
        # verified physical compiler directory with that same invocation mode.
        # Resolving clang++ here would turn an object-only link into C mode.
        compiler = str(Path(compiler).with_name(name))
        selected = file_identity(compiler)
        if any(selected[key] != identity[key] for key in ('sha256', 'size')):
            raise BuildEvidenceError('Apple control compiler identity differs')
    return compiler


def run_arithmetic_controls(source_root, build_evidence, output_dir):
    """Run source-sensitive arithmetic companions without a raw-guard claim.

    A sanitizer safety runner may consume this evidence separately. Production
    qualification must use run_discriminators, which also inspects the actual
    production guard objects without relaxing its instruction/call checker.
    """
    source_root, output_dir = Path(source_root).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    family_name = build_evidence['toolchain']['family']
    family = {'GNU': 'gnu', 'AppleClang': 'clang', 'MSVC': 'msvc'}[family_name]
    units = build_evidence['translation_units']
    unit = next((r for r in units if Path(r['source']).resolve() ==
                 source_root / 'vendor/voro++/src/v_compute.cc'), None)
    if unit is None:
        raise BuildEvidenceError('actual standard compute compilation is missing')
    compiler = _control_compiler(build_evidence, unit, family)
    objects = [r['output'] for r in units
               if Path(r['source']).parent == source_root / 'vendor/voro++/src'
               and Path(r['source']).name != 'v_compute.cc']
    if not objects:
        raise BuildEvidenceError('verified vendor link inputs missing')
    for identity in objects:
        if file_identity(identity['path']) != identity:
            raise BuildEvidenceError('changed vendor discriminator input')
    options = _primitive_options(unit, family)
    env = unit['command']['environment']
    target = build_evidence['toolchain']['target']
    x86 = target.startswith('x86_64')
    if x86 and sys.platform.startswith('linux'):
        flags = Path('/proc/cpuinfo').read_text().split()
        if 'fma' not in flags or 'avx' not in flags:
            raise BuildEvidenceError('required AVX/FMA control needs capable hardware')
    source = output_dir / 'vendor-primitives.cpp'
    source.write_text(_HARNESS)
    observations, control_files, executions, control_builds = {}, [], [], []
    for name, unsafe in (('strict', False), ('unsafe', True), ('unsafe_power', True)):
        directory = output_dir / name
        directory.mkdir(exist_ok=True)
        suffix = '.obj' if family == 'msvc' else '.o'
        object_path = directory / ('primitive' + suffix)
        executable = directory / ('primitive.exe' if family == 'msvc' else 'primitive')
        flags = _control_compile_options(options, family, unsafe=unsafe)
        sanitizer_companion = _SANITIZER_DISABLE in flags
        if family == 'msvc':
            flags.extend(['/I' + str(source_root / 'vendor/voro++/src')])
            if x86:
                flags.append('/arch:AVX2')
            if unsafe:
                flags.extend(['/fp:precise', '/fp:contract'] if name == 'unsafe'
                             else ['/fp:precise', '/DPYVORO2_UNSAFE_POWER=1'])
            compile_command = [compiler, *flags, '/c', str(source),
                               '/Fo' + str(object_path)]
        else:
            flags.extend(['-I' + str(source_root / 'vendor/voro++/src')])
            if x86:
                flags.extend(['-mavx', '-mfma'])
            if unsafe:
                flags.extend(['-ffp-contract=fast'] if name == 'unsafe' else
                             ['-funsafe-math-optimizations', '-ffp-contract=off'])
            compile_command = [compiler, *flags, '-c', str(source),
                               '-o', str(object_path)]
        records = directory / 'records'
        _run_recorded(compile_command, cwd=directory, env=env, records=records,
                      logs=directory, name='compile')
        inputs = [str(object_path), *[r['path'] for r in objects]]
        if family == 'msvc':
            linker_hash = build_evidence['toolchain']['linker_sha256']
            linker = next(r['path'] for r in build_evidence['tools']
                          if r['sha256'] == linker_hash)
            link_command = [linker, '/LTCG:OFF', '/SUBSYSTEM:CONSOLE',
                            '/OUT:' + str(executable), *inputs]
        else:
            link_command = [compiler, '-fno-fast-math', '-ffp-contract=off',
                            '-fno-lto', *_link_options(options),
                            *inputs, '-o', str(executable)]
        _run_recorded(link_command, cwd=directory, env=env, records=records,
                      logs=directory, name='link')
        runtime_env = dict(env)
        runtime_env.update({key: os.environ[key] for key in _RUNTIME_ENVIRONMENT
                            if key in os.environ})
        run = subprocess.run([str(executable)], cwd=directory, env=runtime_env,
                             text=True, capture_output=True)
        result_path = directory / 'result.txt'
        result_path.write_text(run.stdout + run.stderr)
        executions.append({'argv': [str(executable)], 'cwd': str(directory),
                           'environment': runtime_env, 'exit_code': run.returncode,
                           'executable': file_identity(executable),
                           'result': file_identity(result_path)})
        if run.returncode:
            raise BuildEvidenceError('source-sensitive discriminator execution failed')
        standard = parse_standard(run.stdout)
        powers = parse_power(run.stdout)
        if not unsafe and (len(powers) != 3 or any(
                ordinary != '0000000000000000' or checked != '3ff0000000000000'
                for _, ordinary, checked in powers)):
            raise BuildEvidenceError('strict actual-vendor power order mismatch')
        if name == 'unsafe_power' and (len(powers) != 3 or any(
                ordinary != '3ff0000000000000' or checked != '3ff0000000000000'
                for _, ordinary, checked in powers)):
            raise BuildEvidenceError('unsafe actual-vendor power order mismatch')
        observations[name] = {'standard': standard, 'power': powers}
        control_files.extend([file_identity(executable), file_identity(result_path)])
        control_files.extend(verify_control_records(
            records, build_evidence, unsafe=unsafe, inherited_objects=objects,
            sanitizer_companion=sanitizer_companion))
        control_builds.append({
            'name': name,
            'compile_instrumentation': ('unsafe-optimized-companion'
                                        if sanitizer_companion else 'production'),
            'inherited_sanitizer_options': [arg for arg in options
                                            if arg.startswith('-fsanitize=')],
            'compile_only_overrides': ([_SANITIZER_DISABLE]
                                       if sanitizer_companion else []),
            'compile_argv': compile_command, 'link_argv': link_command,
            'records': [file_identity(path) for path in sorted(records.glob('*.json'))],
        })
    fixtures, discriminating = standard_fixture_report(
        observations['strict']['standard'], observations['unsafe']['standard'],
        family_name)
    return {
        'standard_strict_bits': observations['strict']['standard']['standard-xy-v1'][0],
        'standard_unsafe_bits': observations['unsafe']['standard']['standard-xy-v1'][0],
        'standard_fixtures': fixtures,
        'standard_discriminating_fixtures': discriminating,
        'power_offsets': [{'radius_hex': radius, 'r_scale_bits': ordinary,
                           'r_scale_check_bits': checked}
                          for radius, ordinary, checked in
                          observations['strict']['power']],
        'power_unsafe_offsets': [{'radius_hex': radius, 'r_scale_bits': ordinary,
                                  'r_scale_check_bits': checked}
                                 for radius, ordinary, checked in
                                 observations['unsafe_power']['power']],
        'source': file_identity(source),
        'inherited_vendor_inputs': objects, 'control_files': control_files,
        'executions': executions, 'control_builds': control_builds,
        'strict_avx_fma_control': x86,
        'qualification_scope': 'source-sensitive controlled-toolchain arithmetic',
    }


def run_discriminators(source_root, build_evidence, output_dir):
    """Production qualification always requires actual raw-guard inspection."""
    report = run_arithmetic_controls(source_root, build_evidence, output_dir)
    report['guard_disassembly'] = guard_disassembly(
        build_evidence, Path(output_dir).resolve() / 'guards')
    report['qualification_scope'] = (
        'source-sensitive controlled-toolchain corroboration')
    return report
