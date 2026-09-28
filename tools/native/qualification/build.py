"""Controlled build, repair, finalization, and fresh installed-wheel acceptance.

Run outside the source tree. Source approval is an input; this driver cannot
create or refresh it. Every build tree, observed command, process receipt, and
pre/post-repair wheel is retained. A failed stage never produces successful
distribution evidence.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import zipfile


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification.effective_build import (
    BuildEvidenceError, file_identity, verify_build,
)
from qualification.finalize import (
    FinalizationError, SANITIZER_RUNTIME_KEYS, controlled_environment, finalize,
)
from qualification.source_policy import _contract, check_approval, measure_source


class DriverError(RuntimeError):
    """An observed build/distribution stage did not establish qualification."""


_SANITIZER_TESTS = (
    'tests/forward/common/test_native_boundary_validation.py',
    'tests/forward/common/test_native_preconditions.py',
    'tests/forward/common/test_generator_preparation.py',
    'tests/forward/spatial/test_duplicate_check.py',
    'tests/forward/spatial/test_ghost_cells.py',
    'tests/forward/spatial/test_native_witness.py',
    'tests/forward/spatial/test_wp7_native_selected.py',
    'tests/forward/planar/test_api_dispatch.py',
    'tests/forward/planar/test_wp6_native.py',
    'tests/forward/planar/test_wp6_profile_refusal.py',
    'tests/forward/planar/test_wp7_native.py',
    'tests/forward/test_wp7_oracle.py',
)


def build_environment():
    """Runtime injection is forbidden in compiler, repair, and installer processes."""
    return controlled_environment()


def _require(condition, message):
    if not condition:
        raise DriverError(message)


def _failed_output_tail(path, environment):
    """Bound console diagnostics without copying credentials from child output."""
    sensitive = re.compile(
        r'TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTHORIZATION|(?:^|_)KEY(?:$|_)',
        re.I)
    secrets = sorted({value for name, value in environment.items()
                      if value and sensitive.search(name)}, key=len, reverse=True)
    # Extra overlap allows complete credential redaction before taking the tail.
    overlap = max((len(value.encode('utf8')) for value in secrets), default=0)
    limit = 12 * 1024
    with Path(path).open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - limit - overlap))
        text = stream.read().decode('utf8', errors='replace')
    if secrets:
        text = re.sub('|'.join(re.escape(value) for value in secrets),
                      '[redacted]', text)
    text = re.sub(r'(https?://)[^/\s@]+@', r'\1[redacted]@', text)
    text = '\n'.join(text.splitlines()[-80:])
    return text.encode('utf8')[-limit:].decode('utf8', errors='ignore')


def run_process(command, *, cwd, directory, environment=None):
    """Capture the actual process outcome and retained output, without a shell."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    command = [str(value) for value in command]
    executable = shutil.which(command[0])
    _require(executable is not None,
             'required executable is unavailable: ' + command[0])
    # Invocation spelling can select a virtual environment or compiler mode.
    # Hash the resolved executable, but never replace the observed argv[0].
    command[0] = str(Path(executable).absolute())
    if environment is None:
        environment = build_environment()
    executable_identity = file_identity(command[0])
    with (directory / 'stdout').open('wb') as stdout:
        with (directory / 'stderr').open('wb') as stderr:
            result = subprocess.run(command, cwd=cwd, env=environment, stdout=stdout,
                                    stderr=stderr, check=False)
    receipt = {'argv': command, 'cwd': str(Path(cwd).resolve()),
               'executable': executable_identity, 'exit_code': result.returncode,
               'stdout': file_identity(directory / 'stdout'),
               'stderr': file_identity(directory / 'stderr'),
               'runtime_environment': {key: environment[key]
                                       for key in SANITIZER_RUNTIME_KEYS
                                       if key in environment}}
    text = json.dumps(receipt, sort_keys=True, indent=2) + '\n'
    (directory / 'process.json').write_text(text, encoding='utf8')
    if result.returncode != 0:
        details = [f'process failed ({result.returncode}); retained logs: {directory}']
        for name in ('stdout', 'stderr'):
            tail = _failed_output_tail(directory / name, environment)
            if tail:
                details.append(f'{name} (last 80 lines, at most 12 KiB):\n{tail}')
        raise DriverError('\n'.join(details))
    _require(file_identity(command[0]) == executable_identity,
             'process executable changed during execution')
    return receipt


def extract_wheel(wheel, stage):
    """Extract this package's ordinary wheel layout, rejecting unobserved relocation."""
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(wheel) as archive:
        seen = set()
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            _require(not path.is_absolute() and '..' not in path.parts
                     and '\\' not in member.filename
                     and path.as_posix().rstrip('/') == member.filename.rstrip('/'),
                     'wheel contains a noncanonical path')
            _require(member.filename not in seen, 'wheel contains a duplicate path')
            seen.add(member.filename)
            _require(not stat.S_ISLNK(member.external_attr >> 16),
                     'wheel symlink needs a reviewed installation adapter')
            _require(not any(part.endswith('.data') for part in path.parts),
                     'wheel data relocation needs a reviewed installation adapter')
            destination = stage.joinpath(*path.parts)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
    return stage


def repack_wheel(original, stage, output):
    """Add record/anchor and regenerate RECORD, preserving native bytes."""
    stage, output = Path(stage), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(original) as source:
        infos = {info.filename: info for info in source.infolist() if not info.is_dir()}
        records = [name for name in infos if name.endswith('.dist-info/RECORD')]
        _require(len(records) == 1, 'wheel must have exactly one distribution RECORD')
        _require(not any(name.endswith(('/RECORD.jws', '/RECORD.p7s'))
                         for name in infos),
                 'preexisting wheel signature cannot survive finalization')
        record_name = records[0]
        names = set(infos) | {'pyvoro2/_internal/native_qualification_record.json'}
        data = {name: (stage / name).read_bytes()
                for name in names if name != record_name}
        for name in infos:
            path = Path(name)
            if '.so' in path.suffixes or path.suffix in ('.pyd', '.dll', '.dylib'):
                _require(data[name] == source.read(name),
                         'native bytes changed while adding qualification to wheel')
        rows = []
        for name in sorted(data):
            digest = base64.urlsafe_b64encode(hashlib.sha256(data[name]).digest())
            rows.append((name, 'sha256=' + digest.rstrip(b'=').decode('ascii'),
                         str(len(data[name]))))
        rows.append((record_name, '', ''))
        buffer = io.StringIO(newline='')
        csv.writer(buffer, lineterminator='\n').writerows(rows)
        data[record_name] = buffer.getvalue().encode('utf8')
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(data):
                info = infos.get(name)
                if info is None:
                    info = zipfile.ZipInfo(name, infos[record_name].date_time)
                    info.external_attr = 0o100644 << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data[name])
    return output


_IMPORT_PROBE = r'''
import ctypes
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]).resolve()
module = importlib.import_module(sys.argv[2])
module._require_runtime_environment()
path = Path(module.__file__).resolve()
if not path.is_relative_to(root):
    raise RuntimeError('native import escaped fresh installation')
loaded = set()
if sys.platform.startswith('linux'):
    for row in Path('/proc/self/maps').read_text().splitlines():
        fields = row.split(None, 5)
        if len(fields) == 6 and fields[5].startswith('/'):
            name = re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), fields[5])
            if not name.endswith(' (deleted)'):
                loaded.add(str(Path(name).resolve()))
elif sys.platform == 'darwin':
    library = ctypes.CDLL(None)
    library._dyld_image_count.restype = ctypes.c_uint32
    library._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    library._dyld_get_image_name.restype = ctypes.c_char_p
    for index in range(library._dyld_image_count()):
        name = library._dyld_get_image_name(index)
        if name:
            loaded.add(str(Path(name.decode()).resolve()))
elif sys.platform == 'win32':
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.K32EnumProcessModules.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_uint32, ctypes.c_void_p]
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p,
                                        ctypes.c_uint32]
    handles = (ctypes.c_void_p * 4096)()
    needed = ctypes.c_uint32()
    if not kernel.K32EnumProcessModules(kernel.GetCurrentProcess(), handles,
                                       ctypes.sizeof(handles), ctypes.byref(needed)):
        raise ctypes.WinError(ctypes.get_last_error())
    if needed.value > ctypes.sizeof(handles):
        raise RuntimeError('native dependency enumeration exceeded capacity')
    for handle in handles[:needed.value // ctypes.sizeof(ctypes.c_void_p)]:
        buffer = ctypes.create_unicode_buffer(32768)
        if not kernel.GetModuleFileNameW(handle, buffer, len(buffer)):
            raise ctypes.WinError(ctypes.get_last_error())
        loaded.add(str(Path(buffer.value).resolve()))
else:
    raise RuntimeError('no native dependency enumeration adapter')
print(json.dumps({'module': module.__name__, 'path': str(path),
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'identity': module._qualification_identity(),
                  'loaded': sorted(loaded)}, sort_keys=True))
'''


def inspect_installation(stage, output, runtime_environment=None):
    environment = build_environment()
    environment.update(runtime_environment or {})
    environment['PYTHONPATH'] = str(stage)
    result, receipts = {}, []
    for short in ('_fpguard', '_core', '_core2d'):
        name = 'pyvoro2.' + short
        receipt = run_process([sys.executable, '-c', _IMPORT_PROBE, stage, name],
                              cwd=output, directory=output / ('import-' + short),
                              environment=environment)
        report = json.loads(Path(receipt['stdout']['path']).read_text(encoding='utf8'))
        _require(report['module'] == name, 'fresh native import identity differs')
        result[name] = report
        receipts.append(receipt)
    return result, receipts


def _wheel_in(directory):
    wheels = list(directory.glob('*.whl'))
    _require(len(wheels) == 1, 'build/repair must produce exactly one wheel')
    return wheels[0]


def _native_files(stage):
    return {path.relative_to(stage).as_posix(): file_identity(path)
            for path in stage.rglob('*') if path.is_file()
            and ('.so' in path.suffixes or path.suffix in ('.pyd', '.dll', '.dylib'))}


def _postprocess(build_evidence, stage, imports, repair, repair_receipt):
    files = _native_files(stage)
    modules, dependencies = {}, {}
    module_paths = {item['path'] for item in imports.values()}
    native_dependency_paths = {item['path'] for item in files.values()} - module_paths
    guard = Path(imports['pyvoro2._fpguard']['path']).relative_to(stage).as_posix()
    for name, item in imports.items():
        path = Path(item['path']).relative_to(stage).as_posix()
        output = {**files[path], 'path': path}
        linked = build_evidence['components'][name.split('.')[-1]]['output']
        deps = {Path(dep).relative_to(stage).as_posix()
                for dep in set(item['loaded']) & native_dependency_paths}
        if name != 'pyvoro2._fpguard':
            deps.add(guard)
        for dep in deps:
            dependencies[dep] = {key: files[dep][key] for key in ('size', 'sha256')}
        modules[name] = {'input': linked, 'output': output,
                         'dependencies': sorted(deps)}
    inputs = sorted({item['input']['sha256'] for item in modules.values()})
    outputs = sorted({item['sha256'] for item in files.values()})
    operation = {'kind': 'identity' if repair == 'none' else repair,
                 'inputs': inputs, 'outputs': outputs}
    if repair_receipt is not None:
        operation.update(repair_receipt)
    return {'schema': 'pyvoro2-native-postprocess-v1', 'modules': modules,
            'dependencies': dependencies, 'operations': [operation]}


def _git_identity(root):
    commands = {'head': ['rev-parse', 'HEAD'], 'tree': ['rev-parse', 'HEAD^{tree}'],
                'status': ['status', '--porcelain']}
    result = {}
    for key, command in commands.items():
        process = subprocess.run(['git', '-C', str(root), *command],
                                 text=True, capture_output=True, check=False,
                                 env=build_environment())
        result[key] = process.stdout.strip() if process.returncode == 0 else None
    return result


def sanitizer_runtime(build_evidence, output):
    """Resolve runtime libraries with the exact verified GNU compiler."""
    _require(build_evidence['adapter'] == 'gnu-linux-x86_64-v1',
             'sanitizer runtime has no reviewed adapter on this target')
    expected = build_evidence['toolchain']['compiler_sha256']
    compilers = {item['path'] for item in build_evidence['tools']
                 if item['sha256'] == expected}
    _require(len(compilers) == 1, 'ambiguous verified sanitizer compiler')
    compiler = next(iter(compilers))
    libraries, receipts = [], []
    for name in ('libasan.so', 'libubsan.so', 'libstdc++.so'):
        receipt = run_process([compiler, '-print-file-name=' + name], cwd=output,
                              directory=output / ('runtime-' + name),
                              environment=build_environment())
        path = Path(Path(receipt['stdout']['path']).read_text().strip())
        _require(path.is_absolute() and path.is_file(),
                 'verified compiler did not resolve sanitizer runtime: ' + name)
        libraries.append(file_identity(path))
        receipts.append(receipt)
    environment = {
        'LD_PRELOAD': ':'.join(item['path'] for item in libraries),
        'ASAN_OPTIONS': 'detect_leaks=0:halt_on_error=1',
        'UBSAN_OPTIONS': 'print_stacktrace=1:halt_on_error=1',
        'PYVORO2_NATIVE_TEST_SANITIZERS': '1',
    }
    return environment, libraries, receipts


def build(*, source_root, output, repair='none', suite='full', sanitizers=False):
    _require(__debug__ and sys.flags.optimize == 0,
             'optimized Python cannot run distribution qualification')
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    _require(not output.is_relative_to(source_root),
             'build output must be outside source')
    _require(not output.exists(), 'controlled build output must be fresh')
    expected_platform = {'auditwheel': 'linux', 'delocate': 'darwin',
                         'delvewheel': 'win32'}
    _require(repair == 'none' or expected_platform.get(repair) == sys.platform,
             'repair tool does not match this platform')
    _require(suite in ('full', 'adapter'), 'unknown installed test selection')
    _require(not sanitizers or (sys.platform == 'linux' and repair == 'none'),
             'isolated sanitizer safety builds require Linux and --repair none')
    measurement = measure_source(source_root)
    approval = json.loads((source_root / 'src/pyvoro2/_internal/'
                           'native_approval.json').read_text(encoding='utf8'))
    check_approval(measurement, approval)
    q = _contract(source_root)
    output.mkdir(parents=True)
    processes = []
    environment = build_environment()
    # Visual Studio generators ignore the recorder launcher properties. The
    # controlled driver uses Ninja for individual observed compile/link actions.
    environment['CMAKE_GENERATOR'] = 'Ninja'
    for key in ('CMAKE_GENERATOR_PLATFORM', 'CMAKE_GENERATOR_TOOLSET',
                'CMAKE_GENERATOR_INSTANCE'):
        environment.pop(key, None)
    environment.setdefault('CMAKE_BUILD_PARALLEL_LEVEL', '2')
    if sanitizers:
        flags = '-fsanitize=address,undefined,float-cast-overflow'
        environment['CXXFLAGS'] = (environment.get('CXXFLAGS', '') + ' ' + flags
                                   + ' -fno-omit-frame-pointer -g').strip()
        environment['LDFLAGS'] = (environment.get('LDFLAGS', '') + ' ' + flags).strip()
    records = output / 'production-commands'
    environment['PYVORO2_EVIDENCE_DIR'] = str(records)
    direct_dir = output / 'direct-wheel'
    processes.append(run_process(
        [sys.executable, '-m', 'build', '--wheel', '--no-isolation',
         '--outdir', direct_dir, '-Cbuild-dir=' + str(output / 'production-build'),
         '-Ccmake.args=-GNinja', '-Cinstall.strip=false', source_root], cwd=output,
        directory=output / 'process-build', environment=environment))
    direct = _wheel_in(direct_dir)
    build_evidence = verify_build(records, source_root)
    runtime_environment, runtime_libraries = {}, []
    if sanitizers:
        runtime_environment, runtime_libraries, receipts = sanitizer_runtime(
            build_evidence, output)
        processes.extend(receipts)
    required = q.ADAPTERS[build_evidence['adapter']][3]
    direct_stage = extract_wheel(direct, output / 'direct-unpacked')
    direct_native = _native_files(direct_stage)
    for short, item in build_evidence['components'].items():
        members = [info for name, info in direct_native.items()
                   if Path(name).name.split('.')[0] == short]
        _require(len(members) == 1 and members[0]['sha256'] == item['output']['sha256'],
                 'direct wheel bytes differ from observed unstripped link output')
    repaired_dir = output / 'repaired-wheel'
    repaired_dir.mkdir()
    repair_receipt = None
    if repair == 'none':
        repaired = repaired_dir / direct.name
        shutil.copyfile(direct, repaired)
    else:
        repair_evidence = output / 'repair-implementation'
        command = [sys.executable,
                   source_root / 'tools/native/qualification/repair_runner.py',
                   '--kind', repair, '--wheel', direct, '--output', repaired_dir,
                   '--evidence', repair_evidence]
        repair_receipt = run_process(command, cwd=output,
                                     directory=output / 'process-repair')
        repaired = _wheel_in(repaired_dir)
        repair_receipt['wheel_input'] = file_identity(direct)
        repair_receipt['wheel_output'] = file_identity(repaired)
        observed = repair_evidence / 'repair-observation.json'
        repair_receipt['observation_file'] = file_identity(observed)
        repair_receipt['observation'] = json.loads(observed.read_text(encoding='utf8'))
        processes.append(repair_receipt)
    stage = extract_wheel(repaired, output / 'prospective-installation')
    import_dir = output / 'prospective-imports'
    import_dir.mkdir()
    imports, receipts = inspect_installation(stage, import_dir, runtime_environment)
    processes.extend(receipts)
    postprocess = _postprocess(build_evidence, stage, imports, repair, repair_receipt)
    postprocess_path = output / 'postprocess.json'
    postprocess_path.write_bytes(q.canonical_json(postprocess))
    candidate_module = candidate_records = None
    if {'wp6-planar', 'wp7-planar'} & required:
        import pybind11
        candidate_dir = output / 'candidate-build'
        candidate_records = output / 'candidate-commands'
        candidate_environment = environment.copy()
        candidate_environment['PYVORO2_EVIDENCE_DIR'] = str(candidate_records)
        processes.append(run_process(
            ['cmake', '-S', source_root, '-B', candidate_dir, '-G', 'Ninja',
             '-DCMAKE_BUILD_TYPE=Release', '-DPYVORO2_PLANAR_QUALIFICATION=ON',
             '-DPython_EXECUTABLE=' + sys.executable,
             '-Dpybind11_DIR=' + pybind11.get_cmake_dir()], cwd=output,
            directory=output / 'process-candidate-configure',
            environment=candidate_environment))
        processes.append(run_process(
            ['cmake', '--build', candidate_dir, '--target', '_core2d'],
            cwd=output, directory=output / 'process-candidate-build',
            environment=candidate_environment))
        candidate = verify_build(candidate_records, source_root)
        candidate_module = Path(candidate['components']['_core2d']['output']['path'])
    record = finalize(
        source_root=source_root, installation_root=stage, records_dir=records,
        postprocess_path=postprocess_path,
        corpus=source_root / 'tools/native/qualification/fixtures/wp6-predecessor.zip',
        output=output / 'finalization', candidate_module=candidate_module,
        candidate_records=candidate_records, runtime_environment=runtime_environment)
    wheel_dir = 'sanitizer-wheel' if sanitizers else 'final-wheel'
    final_wheel = repack_wheel(repaired, stage, output / wheel_dir / repaired.name)
    installed = output / 'installed'
    processes.append(run_process(
        [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
         '--no-compile',
         '--target', installed, final_wheel], cwd=output,
        directory=output / 'process-final-install'))
    final_import_dir = output / 'final-imports'
    final_import_dir.mkdir()
    final_imports, receipts = inspect_installation(
        installed, final_import_dir, runtime_environment)
    processes.extend(receipts)
    for name, item in final_imports.items():
        _require(item['sha256'] == record['modules'][name]['sha256'],
                 'fresh installed payload differs from qualified final wheel')
    installed_environment = build_environment()
    installed_environment.update(runtime_environment)
    installed_environment['PYTHONPATH'] = str(installed)
    smoke = [sys.executable, str(source_root / 'tools/check_installed_package.py'),
             '--repo-root', str(source_root), '--require-scipy']
    if 'wp6-planar' not in required:
        smoke.append('--planar-refusal')
    if 'wp7-spatial' not in required or 'wp7-planar' not in required:
        smoke.append('--ghost-refusal')
    processes.append(run_process(smoke, cwd=output,
                                 directory=output / 'process-installed-smoke',
                                 environment=installed_environment))
    tests = [str(source_root / 'tests')]
    if sanitizers:
        tests = [str(source_root / name) for name in _SANITIZER_TESTS]
    elif suite == 'adapter':
        from qualification.route_suite import selections
        tests = [str(source_root / name) for name in sorted({
            path for paths in selections(source_root, required).values()
            for path in paths})]
    processes.append(run_process(
        [sys.executable, '-m', 'pytest', '-q', '-o', 'addopts=',
         '--rootdir', source_root, *tests],
        cwd=output, directory=output / 'process-installed-tests',
        environment=installed_environment))
    _require(measure_source(source_root) == measurement,
             'source closure changed during distribution acceptance')
    _require(all(file_identity(item['path']) == item for item in runtime_libraries),
             'sanitizer runtime libraries changed during safety evidence')
    manifest = {'schema': 'pyvoro2-native-distribution-evidence-v1',
                'mode': 'sanitizer-safety' if sanitizers else 'optimized-release',
                'source': measurement, 'git': _git_identity(source_root),
                'direct_wheel': file_identity(direct),
                'repaired_wheel': file_identity(repaired),
                'final_wheel': file_identity(final_wheel),
                'qualification_record_sha256': q.canonical_sha256(record),
                'imported_modules': final_imports,
                'suite': 'sanitizer-safety' if sanitizers else suite,
                'runtime_libraries': runtime_libraries,
                'processes': processes}
    (output / 'distribution-evidence.json').write_bytes(q.canonical_json(manifest))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
                        default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repair',
                        choices=('none', 'auditwheel', 'delocate', 'delvewheel'),
                        default='none')
    parser.add_argument('--suite', choices=('full', 'adapter'), default='full')
    parser.add_argument('--sanitizers', action='store_true',
                        help='separate Linux ASan/UBSan safety artifact and evidence')
    args = parser.parse_args(argv)
    try:
        manifest = build(source_root=args.source_root, output=args.output,
                         repair=args.repair, suite=args.suite,
                         sanitizers=args.sanitizers)
    except (DriverError, BuildEvidenceError, FinalizationError,
            ValueError, OSError) as exc:
        parser.exit(1, f'distribution qualification refused: {exc}\n')
    print(json.dumps(manifest['final_wheel'], sort_keys=True))


if __name__ == '__main__':
    main()
