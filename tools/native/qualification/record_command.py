#!/usr/bin/env python3
"""Record one actual compiler/link command for the controlled external issuer.

Usage: python record_command.py --output-dir DIR -- compiler [arguments...]
Ordinary builds do not run an issuer. These records alone grant no runtime
qualification. A command can succeed while its evidence is incomplete; the
external verifier then refuses issuance explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from qualification.adapters import (observe_apple, observe_gnu_wrapper,
                                        observe_windows, tool_role)
    from qualification.effective_build import (
        BuildEvidenceError, SCHEMA, canonical, digest, expand_response,
        file_identity, windows_words,
    )
    from qualification.link_provenance import native_kind, platform_runtime_inputs
    from qualification.input_provenance import collect_input_providers
else:
    from .adapters import (observe_apple, observe_gnu_wrapper, observe_windows,
                           tool_role)
    from .effective_build import (
        BuildEvidenceError, SCHEMA, canonical, digest, expand_response,
        file_identity, windows_words,
    )
    from .link_provenance import native_kind, platform_runtime_inputs
    from .input_provenance import collect_input_providers

# Pass and record only the compiler/toolchain environment. CI credentials and
# unrelated application configuration must never enter distributable evidence.
_ENVIRONMENT = {
    'PATH', 'HOME', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ', 'TMPDIR', 'TMP', 'TEMP',
    'SystemRoot', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'ComSpec', 'PATHEXT',
    'NUMBER_OF_PROCESSORS', 'PROCESSOR_ARCHITECTURE', 'USERPROFILE', 'APPDATA',
    'LOCALAPPDATA', 'INCLUDE', 'LIB', 'LIBPATH', 'CL', '_CL_', 'LINK', '_LINK_',
    'VCINSTALLDIR', 'VCToolsInstallDir', 'UniversalCRTSdkDir', 'UCRTVersion',
    'WindowsSdkDir', 'WindowsSDKVersion', 'SDKROOT', 'DEVELOPER_DIR',
    'MACOSX_DEPLOYMENT_TARGET', 'CPATH', 'CPLUS_INCLUDE_PATH', 'C_INCLUDE_PATH',
    'OBJC_INCLUDE_PATH', 'LIBRARY_PATH', 'COMPILER_PATH', 'GCC_EXEC_PREFIX',
    'LD_LIBRARY_PATH', 'GCC_COMPARE_DEBUG', 'GCC_COLORS', 'SOURCE_DATE_EPOCH',
    'ZERO_AR_DATE', 'CLANG_CONFIG_FILE_SYSTEM_DIR', 'CLANG_CONFIG_FILE_USER_DIR',
    'CLANG_NO_DEFAULT_CONFIG', 'CCC_OVERRIDE_OPTIONS',
}
_PAIR_OPTIONS = {
    '-o', '-MF', '-MT', '-MQ', '-include', '-imacros', '-I', '-L', '-isystem',
    '-iquote', '-idirafter', '-isysroot', '--sysroot', '-arch', '-target',
    '-x', '-B', '-Xclang', '-Xlinker', '-Xpreprocessor',
}


def path_at(value, cwd):
    return (Path(cwd) / value).resolve()


def output_path(argv, cwd, family):
    if family == 'msvc':
        for index, arg in enumerate(argv):
            lower = arg.lower()
            if lower.startswith('/out:'):
                return path_at(arg[5:], cwd)
            if lower.startswith('/fo'):
                value = arg[3:] or argv[index + 1]
                return path_at(value, cwd)
    for index, arg in enumerate(argv):
        if arg == '-o' and index + 1 < len(argv):
            return path_at(argv[index + 1], cwd)
    raise BuildEvidenceError('explicit output path is required')


def sources_in(argv, cwd):
    result, skip = [], False
    for arg in argv[1:]:
        if skip:
            skip = False
        elif arg in _PAIR_OPTIONS:
            skip = True
        elif not arg.startswith('-') and Path(arg).suffix.lower() in (
                '.c', '.cc', '.cpp', '.cxx'):
            result.append(path_at(arg, cwd))
    return result


def dependencies_from_make(path, cwd):
    text = path.read_text().replace('\\\n', '')
    if ':' not in text:
        raise BuildEvidenceError('invalid actual dependency file')
    # Only the first rule contains prerequisites; -MP may add empty rules.
    first = text.splitlines()[0].split(':', 1)[1]
    words = shlex.split(first, posix=True)
    return sorted({path_at(p.replace('$$', '$'), cwd) for p in words})


def query_options(argv, source, family):
    """Keep all semantic options, remove only action/output/observer controls."""
    result, skip = [], False
    for arg in argv[1:]:
        if skip:
            skip = False
            continue
        if arg in ('-o', '-MF', '-MT', '-MQ', '-wrapper'):
            skip = True
        elif arg in ('-c', '-MD', '-MMD', '-MP'):
            continue
        elif arg.startswith(('-save-temps', '-fverbose-asm')):
            continue
        elif family == 'msvc' and arg.lower() in ('/c', '/bv'):
            continue
        elif family == 'msvc' and arg.lower() in ('/sourcedependencies', '/fa'):
            skip = True
        elif family == 'msvc' and arg.lower().startswith(('/fo', '/fa', '/fd')):
            continue
        elif Path(arg).suffix.lower() in ('.c', '.cc', '.cpp', '.cxx'):
            if Path(arg).resolve() == source:
                continue
            # Source spelling may be relative to the recorded compiler cwd.
            if Path(arg).name == source.name:
                continue
            result.append(arg)
        else:
            result.append(arg)
    return result


def query_macros(driver, argv, source, family, cwd, env, directory):
    options = query_options(argv, source, family)
    evidence = []
    if family == 'msvc':
        probe = directory / 'type-macros.cpp'
        probe.write_text('#include <cfloat>\n'
                         'PYVORO2_EVAL FLT_EVAL_METHOD\n'
                         'PYVORO2_DIGITS DBL_MANT_DIG\n')
        output = directory / 'type-macros.i'
        command = [str(driver), *options, '/P', '/Fi' + str(output), str(probe)]
    else:
        command = [str(driver), *options, '-dM', '-E', str(source)]
        output = directory / 'type-macros.txt'
    run = subprocess.run(command, cwd=cwd, env=env, capture_output=True)
    (directory / 'macro-query.json').write_bytes(canonical({
        'argv': command, 'cwd': str(cwd), 'environment': env,
        'executable': file_identity(driver), 'exit_code': run.returncode,
    }) + b'\n')
    error = directory / 'macro-query.stderr'
    error.write_bytes(run.stderr)
    if family != 'msvc':
        output.write_bytes(run.stdout)
    if run.returncode or not output.exists():
        raise BuildEvidenceError('effective preprocessor/evaluation query failed')
    text = output.read_text(errors='replace')
    if family == 'msvc':
        eval_match = re.search(r'PYVORO2_EVAL\s+(-?\d+)', text)
        digit_match = re.search(r'PYVORO2_DIGITS\s+(\d+)', text)
        macros = {'__FLT_EVAL_METHOD__': eval_match.group(1) if eval_match else '',
                  '__DBL_MANT_DIG__': digit_match.group(1) if digit_match else ''}
        evidence.append(probe)
    else:
        macros = dict(re.findall(r'^#define (\w+) (.*)$', text, re.M))
    evidence.extend([output, error, directory / 'macro-query.json'])
    return {key: macros.get(key, '') for key in (
        '__FLT_EVAL_METHOD__', '__DBL_MANT_DIG__', '__DBL_MAX_EXP__',
        '__LDBL_MANT_DIG__', '__SIZEOF_INT__', '__FAST_MATH__',
        '__FINITE_MATH_ONLY__')}, evidence


def record_command(command, output_dir):
    cwd = Path.cwd().resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identifier = uuid.uuid4().hex
    directory = output_dir / 'details' / identifier
    directory.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k in _ENVIRONMENT}
    family = ('msvc' if os.name == 'nt' else
              'clang' if sys.platform == 'darwin' else 'gnu')
    response_files = {}
    expanded = expand_response(command, cwd, family == 'msvc', response_files)
    if family == 'msvc':
        for name in ('CL', '_CL_', 'LINK', '_LINK_'):
            expand_response(windows_words(env.get(name, '')), cwd, True,
                            response_files)
    kind = ('compile' if any(a in ('-c', '/c', '/C') for a in expanded[1:])
            else 'link')
    output = output_path(expanded, cwd, family)
    source_paths = sources_in(expanded, cwd)
    if kind == 'compile' and len(source_paths) != 1:
        raise BuildEvidenceError('one explicit translation unit is required')
    source = source_paths[0] if source_paths else None
    launched = list(command)
    dep_path = directory / ('dependencies.json' if family == 'msvc' else 'deps.d')
    if kind == 'compile':
        if family == 'msvc':
            launched.extend(['/sourceDependencies', str(dep_path), '/Bv',
                             '/FAs', '/Fa' + str(directory / 'native.asm')])
        else:
            launched.extend(['-save-temps=obj', '-fverbose-asm', '-MD'])
            if '-MF' in expanded:
                dep_path = path_at(expanded[expanded.index('-MF') + 1], cwd)
            else:
                launched.extend(['-MF', str(dep_path)])
    elif family == 'msvc':
        launched.extend(['/LINKREPROFULLPATHRSP:' +
                         str(directory / 'actual-link-inputs.rsp')])
    else:
        launched.append('-Wl,-t')
    observer = {'gnu': observe_gnu_wrapper, 'clang': observe_apple,
                'msvc': observe_windows}[family]
    observation = observer(launched, cwd=cwd, env=env, directory=directory)
    problems = observation['problems']
    evidence_paths = observation['evidence_paths']
    invocations = observation['invocations']
    tools = {i['executable']['path']: i['executable'] for i in invocations}
    tools[observation['observer']['path']] = observation['observer']
    interpreter = file_identity(sys.executable)
    tools[interpreter['path']] = interpreter
    for name in ('record_command.py', 'effective_build.py', 'link_provenance.py',
                 'input_provenance.py'):
        identity = file_identity(Path(__file__).with_name(name))
        tools[identity['path']] = identity
    for identity in observation.get('extra_tools', []):
        tools[identity['path']] = identity
    for identity in observation.get('loaded_images', []):
        tools[identity['path']] = identity
    effective = expand_response(launched, cwd, family == 'msvc', response_files)
    direct = Path(shutil.which(command[0], path=env.get('PATH')) or command[0])
    direct = direct.resolve(strict=True)
    opaque = tool_role(direct) not in ('compiler_driver', 'linker')
    driver = direct
    for invocation in invocations:
        local_snapshots = {}
        # Controlled child callbacks snapshot transient response files before
        # execution, including responses generated by the compiler driver.
        for item in invocation.get('response_files', []):
            response_files.setdefault(item['path'], item)
        try:
            if (invocation.get('observation') == 'direct-child-execution' and
                    'expanded_argv' in invocation):
                words = invocation['expanded_argv']
            else:
                words = expand_response(invocation['argv'], cwd,
                                        family == 'msvc', local_snapshots)
            for path, item in local_snapshots.items():
                if path not in response_files:
                    problems.append(
                        'response contents not captured before invocation: ' + path)
                elif item != response_files[path]:
                    problems.append('response contents changed during invocation: ' +
                                    path)
            invocation['expanded_argv'] = words
        except (OSError, BuildEvidenceError) as error:
            problems.append('incomplete actual response evidence: ' + str(error))
            invocation['expanded_argv'] = []
        if invocation['role'] == 'compiler_driver' and family in ('gnu', 'clang'):
            driver = Path(invocation['executable']['path'])
            if family == 'gnu':
                effective = invocation['expanded_argv']
    if family == 'msvc':
        prefix = env.get('LINK' if kind == 'link' else 'CL', '')
        suffix = env.get('_LINK_' if kind == 'link' else '_CL_', '')
        effective = [
            effective[0],
            *expand_response(windows_words(prefix), cwd, True, response_files),
            *effective[1:],
            *expand_response(windows_words(suffix), cwd, True, response_files),
        ]
        for invocation in invocations:
            invocation['expanded_argv'] = effective
    dependencies, preprocessed, macros, link_inputs, objects = [], [], {}, [], []
    toolchain_inputs = {}
    runtime_link_inputs = []
    input_providers = []
    output_identity = None
    if observation['exit_code'] == 0:
        try:
            output_identity = file_identity(output)
            if kind == 'compile':
                if family == 'msvc':
                    data = json.loads(dep_path.read_text())['Data']
                    if any(data.get(name) for name in ('ImportedModules',
                                                       'ImportedHeaderUnits')):
                        raise BuildEvidenceError('unreviewed MSVC module dependency')
                    paths = [source, *[Path(p) for p in data['Includes']]]
                    # Same fixed driver/options and immutable inputs. This
                    # corroborates the actual dependency and image events.
                    preprocessed_path = directory / 'source.i'
                    options = query_options(effective, source, family)
                    query = subprocess.run([str(driver), *options, '/P',
                                            '/Fi' + str(preprocessed_path),
                                            str(source)], cwd=cwd, env=env,
                                           capture_output=True)
                    if query.returncode:
                        raise BuildEvidenceError('MSVC source expansion failed')
                else:
                    paths = dependencies_from_make(dep_path, cwd)
                    preprocessed_path = output.with_suffix(
                        '.i' if source.suffix == '.c' else '.ii')
                dependencies = [file_identity(p) for p in sorted(set(paths))]
                input_providers, provider_paths = collect_input_providers(
                    driver, family, cwd=cwd, env=env, directory=directory)
                evidence_paths.extend(provider_paths)
                preprocessed = [file_identity(preprocessed_path)]
                macros, query_paths = query_macros(driver, effective, source,
                                                   family, cwd, env, directory)
                # Ninja may ingest and delete the depfile immediately after
                # this launcher returns. Preserve the exact bytes now.
                archived_dep = directory / ('captured-dependencies' + dep_path.suffix)
                archived_dep.write_bytes(dep_path.read_bytes())
                evidence_paths.extend([archived_dep, *query_paths])
            else:
                # Final object association comes from the actual link argv,
                # never from compile_commands.json or target-name guesses.
                linker_jobs = [i for i in invocations if i['role'] == 'linker']
                link_paths = set()
                for invocation in linker_jobs:
                    for arg in invocation['expanded_argv'][1:]:
                        path = path_at(arg, cwd)
                        if (not arg.startswith('-') and path.is_file() and
                                path != output):
                            link_paths.add(path)
                if family in ('clang', 'gnu'):
                    # ld -t lists resolved actual library inputs. Its output is
                    # per-invocation, not interleaved parallel shell output.
                    logs = list(directory.glob('job-*.stdout'))
                    logs.extend(directory.glob('stdout.txt'))
                    for log in logs:
                        for line in log.read_text(errors='replace').splitlines():
                            path = path_at(line.strip(), cwd)
                            if path.is_file() and path != output:
                                link_paths.add(path)
                if family == 'msvc':
                    repro = directory / 'actual-link-inputs.rsp'
                    if not repro.is_file():
                        raise BuildEvidenceError(
                            'MSVC actual link input report missing '
                            '(VS 17.11+ required)')
                    data = repro.read_bytes()
                    encoding = ('utf-16' if data.startswith(b'\xff\xfe')
                                else 'utf-8-sig')
                    names = windows_words(data.decode(encoding))
                    if not names or any(not Path(name).is_absolute()
                                        for name in names):
                        raise BuildEvidenceError(
                            'incomplete MSVC actual link input paths')
                    evidence_paths.append(repro)
                    link_paths.update(Path(name).resolve(strict=True)
                                      for name in names)
                # MSVC and Apple direct jobs do not use GNU's collect2 wrapper.
                for arg in effective[1:]:
                    if Path(arg).suffix.lower() in ('.o', '.obj'):
                        link_paths.add(path_at(arg, cwd))
                link_inputs = [file_identity(p) for p in sorted(link_paths)]
                runtime_link_inputs, runtime_evidence = platform_runtime_inputs(
                    driver, family, cwd=cwd, env=env, directory=directory)
                evidence_paths.append(runtime_evidence)
                runtime_paths = {item['path'] for item in runtime_link_inputs}
                objects = [item for item in link_inputs
                           if native_kind(item['path']) == 'object'
                           and item['path'] not in runtime_paths]
                if not objects:
                    raise BuildEvidenceError('no observed final object association')
        except (OSError, KeyError, ValueError, BuildEvidenceError) as error:
            problems.append(str(error))
    else:
        problems.append('actual command failed')
    for invocation in invocations:
        for name in invocation.get('opened_files', []):
            path = Path(name)
            # Record relevant compiler specs, dynamic tool dependencies, and
            # configuration files actually opened, as well as source headers.
            if path.is_file() and (path.suffix in ('.so', '.cfg', '.specs') or
                                   '.so.' in path.name or path.name == 'specs'):
                toolchain_inputs[str(path)] = file_identity(path)
    for identity in response_files.values():
        if (Path(identity['path']).exists() and
                file_identity(identity['path'])['sha256'] != identity['sha256']):
            problems.append('response file changed after invocation')
    row = {
        'schema': SCHEMA, 'record_id': identifier,
        'adapter': observation['adapter'], 'family': family, 'kind': kind,
        'argv': command, 'launched_argv': launched, 'effective_argv': effective,
        'environment': env, 'cwd': str(cwd), 'invocations': invocations,
        'tools': sorted(tools.values(), key=lambda r: r['path']),
        'response_files': sorted(response_files.values(), key=lambda r: r['path']),
        'source': str(source) if source else None, 'output': output_identity,
        'dependencies': dependencies, 'preprocessed': preprocessed,
        'macros': macros, 'link_inputs': link_inputs, 'object_inputs': objects,
        'runtime_link_inputs': runtime_link_inputs,
        'input_providers': input_providers,
        'system_loader_evidence': observation.get('system_loader_evidence', []),
        'toolchain_inputs': sorted(toolchain_inputs.values(), key=lambda r: r['path']),
        'driver_version': observation.get('driver_version'),
        'driver_target': observation.get('driver_target'),
        'evidence_files': [file_identity(p) for p in sorted(set(evidence_paths))],
        'opaque_wrappers': opaque, 'exit_code': observation['exit_code'],
        'complete': not problems and observation['exit_code'] == 0,
        'problems': problems,
    }
    row['record_sha256'] = digest(row)
    temporary = output_dir / (identifier + '.tmp')
    temporary.write_bytes(canonical(row) + b'\n')
    temporary.replace(output_dir / (identifier + '.json'))
    for name in ('stdout.txt', 'stderr.txt'):
        path = directory / name
        if path.exists():
            target = (sys.stdout.buffer if name.startswith('stdout') else
                      sys.stderr.buffer)
            target.write(path.read_bytes())
            target.flush()
    if problems:
        print('qualification evidence incomplete: ' + '; '.join(problems),
              file=sys.stderr)
    return observation['exit_code']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('an actual compiler/link command is required after --')
    try:
        return record_command(command, args.output_dir)
    except (BuildEvidenceError, OSError) as error:
        print('qualification recorder refused: ' + str(error), file=sys.stderr)
        return 125


if __name__ == '__main__':
    raise SystemExit(main())
