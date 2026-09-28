"""Controlled platform observation adapters.

The adapters collect evidence; their availability never implies a qualified
platform. Each release still requires the platform's positive route evidence.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from .effective_build import BuildEvidenceError, expand_response, file_identity
from .loader_trace import (capture_dyld, capture_glibc, dyld_environment,
                           glibc_environment)


def tool_role(path):
    name = Path(path).name.lower()
    if name in ('cc1', 'cc1plus'):
        return 'compiler_backend'
    if (name in ('ld', 'ld.bfd', 'ld.gold', 'ld.lld', 'ld-classic',
                 'ld64.lld', 'link.exe') or
            re.search(r'-ld(?:\.bfd|\.gold|\.lld)?$', name)):
        return 'linker'
    if name in ('as', 'llvm-as') or name.endswith('-as'):
        return 'assembler'
    if name == 'collect2':
        return 'link_driver'
    if (re.search(r'(?:^|-)(?:g\+\+|gcc)(?:-\d+(?:\.\d+)*)?$', name) or
            name in ('clang', 'clang++', 'cl.exe')):
        return 'compiler_driver'
    return 'unknown'


def _call_arguments(text):
    """Split a strace call at top-level commas, preserving quoted strings."""
    result, begin, depth, quote, escape = [], 0, 0, False, False
    for index, char in enumerate(text):
        if quote:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == '"':
                quote = False
        elif char == '"':
            quote = True
        elif char in '[{(':
            depth += 1
        elif char in ']})':
            depth -= 1
        elif char == ',' and depth == 0:
            result.append(text[begin:index].strip())
            begin = index + 1
    result.append(text[begin:].strip())
    return result


def observe_linux(command, *, cwd, env, directory):
    tracer = shutil.which('strace', path=env.get('PATH'))
    if not tracer:
        raise BuildEvidenceError('strace is required for GNU actual-process evidence')
    prefix = directory / 'process'
    stdout, stderr = directory / 'stdout.txt', directory / 'stderr.txt'
    with stdout.open('wb') as out, stderr.open('wb') as err:
        run = subprocess.run([
            tracer, '-ff', '-qq', '-v', '-s', '0', '-yy',
            '-e', 'trace=execve,execveat,chdir,fchdir,openat,open',
            '-o', str(prefix), '--', *command,
        ], cwd=cwd, env=env, stdout=out, stderr=err)
    invocations, problems = [], []
    traces = sorted(directory.glob('process.*'))
    for trace in traces:
        current = None
        for line in trace.read_text(errors='strict').splitlines():
            if line.startswith(('chdir(', 'fchdir(')) and line.endswith('= 0'):
                problems.append('unsupported child working-directory mutation')
            if line.startswith('execveat('):
                problems.append('execveat requires an additional reviewed adapter')
            if line.startswith('execve(') and re.search(r'\)\s+= 0$', line):
                try:
                    args = _call_arguments(line[7:line.rfind(')')])
                    executable, argv, environment = map(ast.literal_eval, args)
                    executable = (Path(cwd) / executable).resolve(strict=True)
                    if not (isinstance(argv, list) and
                            isinstance(environment, list)):
                        raise ValueError('incomplete argv/environment')
                    current = {
                        'pid': int(trace.suffix[1:]),
                        'executable': file_identity(executable),
                        'argv': argv, 'cwd': str(cwd),
                        'environment': dict(e.split('=', 1) for e in environment),
                        'role': tool_role(executable), 'opened_files': [],
                    }
                    invocations.append(current)
                except (ValueError, SyntaxError, OSError) as error:
                    problems.append('unparsed actual execve: ' + str(error))
            if current is not None and line.startswith(('open(', 'openat(')):
                match = re.search(r'= \d+<([^>]+)>$', line)
                if match and 'O_RDONLY' in line:
                    path = Path(match.group(1))
                    if path.is_file():
                        current['opened_files'].append(str(path.resolve()))
        if '<unfinished ...>' in trace.read_text() and not invocations:
            problems.append('incomplete process trace')
    for invocation in invocations:
        invocation['opened_files'] = sorted(set(invocation['opened_files']))
    if not invocations:
        problems.append('no successful actual process invocation')
    return {
        'adapter': 'linux-strace-v1', 'exit_code': run.returncode,
        'invocations': invocations, 'problems': problems,
        'evidence_paths': [stdout, stderr, *traces],
        'observer': file_identity(tracer),
    }


def observe_gnu_wrapper(command, *, cwd, env, directory):
    """Execute GCC's actual child jobs through controlled, recorded callbacks.

    GCC's documented -wrapper covers cc1plus/as/collect2. collect2's linker
    child is separately redirected through a private -B directory containing
    only the fixed ld callback. The actual linker executable is resolved before
    creating that directory. There is no compiler-log or dry-run replay here.
    """
    from .child_record import collect_child_receipts

    driver_word = next((a for a in command if tool_role(a) == 'compiler_driver'),
                       None)
    if driver_word is None:
        raise BuildEvidenceError('GNU adapter requires an identifiable driver')
    driver = Path(shutil.which(driver_word, path=env.get('PATH')) or driver_word)
    driver = driver.resolve(strict=True)
    callback = Path(__file__).with_name('child_record.py').resolve()
    children = directory / 'children'
    children.mkdir()
    linker_query = subprocess.run([str(driver), '-print-prog-name=ld'],
                                  cwd=cwd, env=env, text=True, capture_output=True)
    if linker_query.returncode:
        raise BuildEvidenceError('cannot resolve the fixed GNU linker')
    linker_word = linker_query.stdout.strip()
    linker = Path(shutil.which(linker_word, path=env.get('PATH')) or linker_word)
    linker = linker.resolve(strict=True)
    shim_dir = directory / 'linker'
    shim_dir.mkdir()
    shim = shim_dir / 'ld'
    shim.write_text('#!' + sys.executable + '\n'
                    'import os, sys\n'
                    'os.execv(' + repr(sys.executable) + ', ' +
                    repr([sys.executable, str(callback), '--directory',
                          str(children), '--', str(linker)]) +
                    ' + sys.argv[1:])\n')
    shim.chmod(0o755)
    if any(',' in p for p in (sys.executable, str(callback), str(children))):
        raise BuildEvidenceError('comma in GCC wrapper path is unsupported')
    wrapper = ','.join([sys.executable, str(callback), '--directory',
                        str(children), '--'])
    launched = [*command, '-wrapper', wrapper, '-B' + str(shim_dir) + os.sep]
    stdout, stderr = directory / 'stdout.txt', directory / 'stderr.txt'
    direct = Path(shutil.which(command[0], path=env.get('PATH')) or command[0])
    direct = direct.resolve(strict=True)
    # Canonical argv[0] also makes the loader's program identity unambiguous.
    launched[0] = str(direct)
    launched[command.index(driver_word)] = str(driver)
    loader_prefix = directory / 'driver-loader'
    execution_env = glibc_environment(env, loader_prefix)
    root = {'executable': file_identity(direct), 'argv': launched,
            'cwd': str(cwd), 'environment': execution_env, 'role': tool_role(direct),
            'opened_files': [], 'observation': 'direct-driver-execution'}
    with stdout.open('wb') as out, stderr.open('wb') as err:
        run = subprocess.run(launched, cwd=cwd, env=execution_env,
                             stdout=out, stderr=err)
    root['loader'] = capture_glibc(loader_prefix)
    root['opened_files'] = [item['path'] for item in root['loader']['images']]
    child_invocations, records = collect_child_receipts(children)
    invocations = [root, *child_invocations]
    problems = []
    loader_files, loader_images = [], {}
    known_programs = {file_identity(sys.executable)['sha256'],
                      *[row['executable']['sha256'] for row in invocations]}
    for row in invocations:
        for identity in row['loader']['images']:
            loader_images[identity['path']] = identity
        loader_files.extend(Path(item['path'])
                            for item in row['loader']['evidence_files'])
        if any(item['sha256'] not in known_programs
               for item in row['loader']['programs']):
            problems.append('unobserved GNU descendant program in loader evidence')
    for row in invocations[1:]:
        if not row['executable_unchanged'] or not row['responses_unchanged']:
            problems.append('child executable/response changed during execution')
        if row['role'] == 'unknown':
            problems.append('unknown GNU child executable: ' +
                            row['executable']['path'])
        if Path(row['executable']['path']).read_bytes()[:4] != b'\x7fELF':
            problems.append('opaque GNU child executable wrapper')
        if row['exit_code']:
            problems.append('actual GNU child failed')
    # Driver and specs identities are recorded alongside the actual callbacks.
    provenance = directory / 'driver-provenance.json'
    queries = {}
    for option in ('--version', '-dumpmachine', '-dumpspecs', '-print-search-dirs'):
        result = subprocess.run([str(driver), option], cwd=cwd, env=env,
                                text=True, capture_output=True)
        queries[option] = {'argv': [str(driver), option],
                           'exit_code': result.returncode,
                           'stdout': result.stdout, 'stderr': result.stderr}
        if result.returncode:
            problems.append('GNU driver provenance query failed')
    provenance.write_text(json.dumps(queries, sort_keys=True, indent=2) + '\n')
    return {
        'adapter': 'gnu-child-wrapper-v1', 'exit_code': run.returncode,
        'invocations': invocations, 'problems': problems,
        'evidence_paths': [stdout, stderr, provenance, shim, *records, *loader_files],
        'observer': file_identity(callback),
        'extra_tools': [file_identity(driver), file_identity(linker),
                        file_identity(sys.executable), file_identity(__file__),
                        file_identity(Path(__file__).with_name('loader_trace.py'))],
        'loaded_images': list(loader_images.values()),
        'driver_version': queries['--version']['stdout'].splitlines()[0],
        'driver_target': queries['-dumpmachine']['stdout'].strip(),
    }


def parse_clang_plan(text, compiler):
    """Accept only explicit cc1, assembler, and native-linker child jobs.

    This is deliberately a closed plan grammar. Offload, universal/lipo,
    modules, shell fragments, and unrecognized diagnostics never silently
    become trusted command evidence.
    """
    jobs = []
    compiler = Path(compiler)
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith('"'):
            if '(in-process)' in stripped:
                raise BuildEvidenceError('unsupported in-process clang plan')
            continue
        try:
            argv = shlex.split(stripped, posix=True)
        except ValueError as error:
            raise BuildEvidenceError('unsupported clang plan quoting') from error
        if not argv or not Path(argv[0]).is_absolute():
            raise BuildEvidenceError('unsupported clang executable path')
        executable = Path(argv[0])
        same_clang = (executable == compiler or
                      executable.parent == compiler.parent and
                      executable.name in ('clang', 'clang++'))
        if same_clang:
            if len(argv) < 2 or argv[1] not in ('-cc1', '-cc1as'):
                raise BuildEvidenceError('unsupported clang child job')
            if any(a.startswith(('-fmodules', '-load', '-plugin')) for a in argv):
                raise BuildEvidenceError('unsupported clang module/plugin job')
        elif executable.name not in ('as', 'ld', 'ld-classic', 'ld64.lld'):
            raise BuildEvidenceError('unsupported clang plan executable')
        jobs.append(argv)
    if not jobs:
        raise BuildEvidenceError('clang plan has no executable jobs')
    return jobs


def prepare_apple_driver(command, *, cwd, env):
    """Resolve the developer selector without discarding its mode or SDK.

    clang++ commonly symlinks to clang, but its lexical argv[0] selects the
    C++ driver. File identities are resolved separately. A selected macOS SDK
    is explicit in the actual execution environment and independent queries.
    """
    from .link_provenance import clean_toolchain_environment

    configuration_variables = ('CCC_OVERRIDE_OPTIONS',
                               'CLANG_CONFIG_FILE_SYSTEM_DIR',
                               'CLANG_CONFIG_FILE_USER_DIR')
    if any(env.get(name) for name in configuration_variables):
        raise BuildEvidenceError('unreviewed Apple driver configuration environment')
    compiler = Path(shutil.which(command[0], path=env.get('PATH')) or command[0])
    compiler = compiler.absolute()
    if compiler.name not in ('clang', 'clang++'):
        raise BuildEvidenceError('Apple adapter requires direct resolved clang')
    selector = None
    environment = dict(env)
    if compiler.parent == Path('/usr/bin'):
        # /usr/bin/clang is Apple's developer-tool selector. Execute the
        # selected real driver directly so SIP cannot strip our child tracing.
        clean = clean_toolchain_environment(env)
        selector_argv = ['/usr/bin/xcrun', '--sdk', 'macosx', '--find', compiler.name]
        selected = subprocess.run(selector_argv, cwd=cwd, env=clean,
                                  text=True, capture_output=True)
        if selected.returncode:
            raise BuildEvidenceError('cannot resolve selected Apple compiler')
        selected_path = Path(selected.stdout.strip())
        if (not selected_path.is_absolute() or selected_path.name != compiler.name
                or not selected_path.is_file()):
            raise BuildEvidenceError('invalid selected Apple compiler invocation')
        compiler = selected_path
        selector = {'argv': selector_argv, 'stdout': selected.stdout,
                    'stderr': selected.stderr, 'exit_code': selected.returncode,
                    'executable': file_identity('/usr/bin/xcrun')}
        sdk_argv = ['/usr/bin/xcrun', '--sdk', 'macosx', '--show-sdk-path']
        selected_sdk = subprocess.run(sdk_argv, cwd=cwd, env=clean,
                                      text=True, capture_output=True)
        sdk = Path(selected_sdk.stdout.strip())
        if selected_sdk.returncode or not sdk.is_absolute() or not sdk.is_dir():
            raise BuildEvidenceError('cannot resolve selected Apple macOS SDK')
        sdk = sdk.resolve(strict=True)
        requested = [env['SDKROOT']] if env.get('SDKROOT') else []
        expanded = expand_response(command, cwd)
        for index, arg in enumerate(expanded):
            if arg in ('-isysroot', '--sysroot'):
                if index + 1 == len(expanded):
                    raise BuildEvidenceError('incomplete Apple sysroot option')
                requested.append(expanded[index + 1])
            elif arg.startswith(('-isysroot', '--sysroot=')):
                value = (arg.removeprefix('-isysroot').lstrip('=')
                         if arg.startswith('-isysroot') else arg.split('=', 1)[1])
                requested.append(value)
        if any((Path(cwd) / root).resolve() != sdk for root in requested):
            raise BuildEvidenceError('unreviewed selected Apple sysroot')
        environment['SDKROOT'] = str(sdk)
        settings = sdk / 'SDKSettings.json'
        if not settings.is_file():
            raise BuildEvidenceError('selected Apple SDK lacks configuration identity')
        selector['sdk_query'] = {
            'argv': sdk_argv, 'environment': clean, 'stdout': selected_sdk.stdout,
            'stderr': selected_sdk.stderr, 'exit_code': selected_sdk.returncode,
        }
        selector['sdk_configuration'] = [file_identity(settings)]
    # Bind the real bytes while returning the original driver-mode spelling.
    file_identity(compiler)
    return {'compiler': compiler, 'environment': environment, 'selector': selector}


def apple_preprocessed_input(invocations, source):
    """Bind the actual saved expansion to its producing and consuming jobs.

    Clang's saved-intermediate basename differs from GCC's for e.g. foo.cpp.o.
    Filenames are evidence from executed cc1 jobs, not a suffix convention.
    """
    jobs = [row for row in invocations
            if row.get('expanded_argv', [])[1:2] == ['-cc1']]
    producers = [row for row in jobs if '-E' in row['expanded_argv']]
    if len(producers) != 1:
        raise BuildEvidenceError('missing/ambiguous Apple preprocessing job')
    producer = producers[0]
    argv = producer['expanded_argv']
    cwd = Path(producer['cwd'])
    if (producer.get('observation') != 'direct-child-execution' or
            producer.get('exit_code') != 0 or
            (cwd / argv[-1]).resolve() != Path(source).resolve() or
            argv.count('-o') != 1 or argv.index('-o') + 1 >= len(argv)):
        raise BuildEvidenceError('incomplete Apple preprocessing source/output')
    output = (cwd / argv[argv.index('-o') + 1]).resolve()
    consumers = []
    for row in jobs:
        args = row['expanded_argv']
        if '-x' not in args or args.index('-x') + 1 >= len(args):
            continue
        language = args[args.index('-x') + 1]
        if language not in ('c++-cpp-output', 'cpp-output'):
            continue
        if (row.get('observation') == 'direct-child-execution' and
                row.get('exit_code') == 0 and
                (Path(row['cwd']) / args[-1]).resolve() == output):
            consumers.append(row)
    if len(consumers) != 1:
        raise BuildEvidenceError('Apple preprocessed output lacks one actual consumer')
    return output


def observe_apple(command, *, cwd, env, directory):
    prepared = prepare_apple_driver(command, cwd=cwd, env=env)
    compiler, selector = prepared['compiler'], prepared['selector']
    execution_env = dyld_environment(prepared['environment'])
    # The driver is used only to expand the jobs. Every emitted job is then
    # executed by this trusted adapter using the recorded argv, with no shell.
    plan_argv = [str(compiler), *command[1:], '-###', '-fno-integrated-cc1']
    process = subprocess.Popen(plan_argv, cwd=cwd, env=execution_env,
                               text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    plan_stdout, plan_stderr = process.communicate()
    plan = subprocess.CompletedProcess(plan_argv, process.returncode,
                                       plan_stdout, plan_stderr)
    plan_path = directory / 'driver-plan.txt'
    plan_path.write_text(plan.stdout + plan.stderr)
    if plan.returncode:
        raise BuildEvidenceError('Apple driver cannot emit the supported job plan')
    if 'Configuration file:' in plan.stderr:
        raise BuildEvidenceError('unreviewed Apple driver configuration file')
    jobs = parse_clang_plan(plan.stderr, compiler)
    driver_loader = capture_dyld(plan.stderr, compiler, process.pid)
    invocations = [{
        'executable': file_identity(compiler), 'argv': plan_argv,
        'cwd': str(cwd), 'environment': execution_env, 'role': 'compiler_driver',
        'opened_files': [], 'exit_code': plan.returncode,
        'observation': 'executed-driver-plan-only', 'selector': selector,
        'loader': driver_loader,
    }]
    paths, exit_code, problems = [plan_path], 0, []
    for index, argv in enumerate(jobs):
        executable = Path(argv[0]).resolve(strict=True)
        before = file_identity(executable)
        snapshots = {}
        expanded = expand_response(argv, cwd, snapshots=snapshots)
        out_path = directory / f'job-{index}.stdout'
        err_path = directory / f'job-{index}.stderr'
        with out_path.open('wb') as out, err_path.open('wb') as err:
            process = subprocess.Popen(argv, cwd=cwd, env=execution_env,
                                       stdout=out, stderr=err)
            returncode = process.wait()
        role = ('assembler' if argv[1:2] == ['-cc1as'] else
                'compiler_backend' if argv[1:2] == ['-cc1'] and
                '-E' not in argv else tool_role(executable))
        invocations.append({
            'executable': before, 'argv': argv, 'expanded_argv': expanded,
            'cwd': str(cwd), 'environment': execution_env, 'role': role,
            'opened_files': [], 'exit_code': returncode,
            'observation': 'direct-child-execution',
            'loader': capture_dyld(err_path.read_text(), executable, process.pid),
            'response_files': list(snapshots.values()),
        })
        if file_identity(executable) != before:
            problems.append('Apple child executable changed during execution')
        if any(file_identity(item['path'])['sha256'] != item['sha256']
               for item in snapshots.values()):
            problems.append('Apple child response changed during execution')
        paths.extend([out_path, err_path])
        if returncode:
            exit_code = returncode
            break
    loaded = {item['path']: item for row in invocations
              for item in row['loader']['images']}
    extra = [file_identity(__file__),
             file_identity(Path(__file__).with_name('loader_trace.py'))]
    if selector:
        extra.append(selector['executable'])
        extra.extend(selector['sdk_configuration'])
    return {
        'adapter': 'apple-plan-replay-v1', 'exit_code': exit_code,
        'invocations': invocations, 'problems': problems, 'evidence_paths': paths,
        'observer': file_identity(compiler),
        'extra_tools': extra,
        'loaded_images': list(loaded.values()),
        'effective_environment': prepared['environment'],
        'system_loader_evidence': [row['loader']['system_loader']
                                   for row in invocations],
        'driver_version': next((line for line in plan.stderr.splitlines()
                                if 'clang version' in line), ''),
        'driver_target': next((line.partition(':')[2].strip()
                               for line in plan.stderr.splitlines()
                               if line.startswith('Target:')), ''),
    }


def observe_windows(command, *, cwd, env, directory):
    if os.name != 'nt':
        raise BuildEvidenceError('Windows debug adapter requires Windows')
    from .windows_trace import observe
    return observe(command, cwd=cwd, env=env, directory=directory)
