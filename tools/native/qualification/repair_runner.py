"""Observe reviewed Python wheel-repair implementations and their helper launches.

The repair distribution and loaded module files are immutable inputs, separately
from their small console-script launcher. Python helper process creation must
pass through the observed Popen surface. Opaque shells and unsupported spawning
APIs refuse. Native system helper binaries remain reviewed tool inputs; this is
not a security monitor for malicious native code or undisclosed kernel activity.
"""
from __future__ import annotations

import argparse
import base64
from contextvars import ContextVar
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


class RepairObservationError(RuntimeError):
    """The repair path cannot supply a complete reviewed observation."""


def canonical(value):
    text = json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n'
    return text.encode('utf8')


def identity(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return {'path': str(path), 'size': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def _require(condition, message):
    if not condition:
        raise RepairObservationError(message)


class ProcessObserver:
    """Observe Popen without changing the repair implementation's stream choices."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.records = []
        self.processes = []
        self.enabled = False
        self.spawning = ContextVar('pyvoro2_repair_spawn', default=False)

    def audit(self, event, arguments):
        if not self.enabled:
            return
        if event in ('subprocess.Popen', 'os.posix_spawn'):
            _require(self.spawning.get(), 'unobserved repair helper process API')
        elif event in ('os.system', 'os.exec', 'os.spawn', 'os.fork', 'os.forkpty'):
            raise RepairObservationError('opaque repair process API: ' + event)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.original = subprocess.Popen
        owner = self

        class ObservedPopen(owner.original):
            def __init__(self, args, *positional, **kwargs):
                _require(not positional, 'positional Popen controls are unreviewed')
                _require(not kwargs.get('shell'), 'opaque repair shell is unqualified')
                _require(isinstance(args, (tuple, list)) and args
                         and all(isinstance(arg, (str, bytes, os.PathLike))
                                 for arg in args), 'opaque repair command line')
                argv = [os.fsdecode(arg) for arg in args]
                cwd = Path(kwargs.get('cwd') or Path.cwd()).resolve()
                supplied_environment = kwargs.get('env')
                environment = dict(os.environ if supplied_environment is None
                                   else supplied_environment)
                executable = os.fsdecode(kwargs.get('executable') or argv[0])
                if os.path.dirname(executable):
                    executable = str((cwd / executable).absolute())
                else:
                    executable = shutil.which(
                        executable, path=environment.get('PATH', os.defpath))
                _require(executable is not None, 'repair helper executable not found')
                _require(Path(executable).name.lower() not in (
                    'sh', 'bash', 'dash', 'zsh', 'cmd', 'cmd.exe',
                    'powershell', 'powershell.exe', 'pwsh', 'pwsh.exe'),
                    'opaque repair shell helper is unqualified')
                index = len(owner.records)
                relevant = ('PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
                            'DYLD_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES', 'SYSTEMROOT',
                            'WINDIR', 'PATHEXT', 'TEMP', 'TMP', 'SOURCE_DATE_EPOCH',
                            'CODESIGN_ALLOCATE', 'AUDITWHEEL_PLAT')
                self.observation = {
                    'argv': argv, 'cwd': str(cwd), 'executable': identity(executable),
                    'environment': {key: environment[key] for key in relevant
                                    if key in environment},
                    'environment_sha256': hashlib.sha256(
                        canonical(environment)).hexdigest(),
                    'responses': [],
                    'stdout_mode': self._stream_mode(kwargs.get('stdout')),
                    'stderr_mode': self._stream_mode(kwargs.get('stderr')),
                    'stdout': None, 'stderr': None, 'exit_code': None,
                }
                for number, arg in enumerate(argv):
                    if arg.startswith('@'):
                        path = (cwd / arg[1:]).resolve(strict=True)
                        data = path.read_bytes()
                        retained = owner.directory / f'{index}-{number}-response.bin'
                        retained.write_bytes(data)
                        self.observation['responses'].append({
                            'original_path': str(path), 'snapshot': identity(retained),
                            'bytes_base64': base64.b64encode(data).decode('ascii')})
                self.observation_index = index
                owner.records.append(self.observation)
                token = owner.spawning.set(True)
                try:
                    super().__init__(args, **kwargs)
                finally:
                    owner.spawning.reset(token)
                owner.processes.append(self)

            @staticmethod
            def _stream_mode(stream):
                if stream == subprocess.PIPE:
                    return 'pipe'
                if stream == subprocess.STDOUT:
                    return 'stdout'
                if stream == subprocess.DEVNULL:
                    return 'discarded'
                if stream is None:
                    return 'inherited'
                # A tool-owned file stream is not replaced or rewound. Opaque
                # redirected streams need a separate adapter to retain bytes.
                raise RepairObservationError('opaque redirected repair stream')

            def communicate(self, *args, **kwargs):
                result = super().communicate(*args, **kwargs)
                for key, data in zip(('stdout', 'stderr'), result):
                    if data is None:
                        continue
                    if isinstance(data, str):
                        # Popen may retain the TextIOWrapper-only selector
                        # "locale". The stream binds the resolved codec used
                        # to decode this output, including on Python 3.10.
                        encoding = getattr(self, key).encoding
                        data = data.encode(encoding)
                        self.observation[key + '_encoding'] = encoding
                    path = owner.directory / f'{self.observation_index}-{key}'
                    path.write_bytes(data)
                    self.observation[key] = identity(path)
                return result

        subprocess.Popen = ObservedPopen
        sys.addaudithook(self.audit)
        self.enabled = True
        return self

    def __exit__(self, kind, value, traceback):
        self.enabled = False
        subprocess.Popen = self.original
        if kind is not None:
            return False
        for process in self.processes:
            row = process.observation
            row['exit_code'] = process.poll()
            _require(row['exit_code'] is not None, 'unfinished repair helper')
            _require(identity(row['executable']['path']) == row['executable'],
                     'repair helper executable changed during operation')
            for stream in ('stdout', 'stderr'):
                _require(row[stream + '_mode'] != 'pipe' or row[stream] is not None,
                         'repair helper PIPE output was not retained by communicate')
        return False


def _distribution_inputs(distribution):
    files = []
    for item in distribution.files or ():
        if Path(item).suffix in ('.pyc', '.pyo'):
            continue
        path = Path(distribution.locate_file(item))
        if path.is_file():
            files.append(identity(path))
    _require(files, 'repair distribution has no inspectable implementation files')
    return sorted(files, key=lambda row: row['path'])


def run_repair(*, kind, wheel, output, evidence):
    names = {'auditwheel': ('auditwheel', 'auditwheel'),
             'delocate': ('delocate', 'delocate-wheel'),
             'delvewheel': ('delvewheel', 'delvewheel')}
    _require(kind in names, 'unknown repair implementation')
    package, script = names[kind]
    distribution = metadata.distribution(package)
    inputs = _distribution_inputs(distribution)
    entries = [entry for entry in distribution.entry_points
               if entry.group == 'console_scripts' and entry.name == script]
    _require(len(entries) == 1, 'ambiguous repair entry point')
    evidence = Path(evidence).resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    output = Path(output).resolve()
    argv = ([script, 'repair', '--wheel-dir', str(output), str(wheel)]
            if kind != 'delocate' else
            [script, '--wheel-dir', str(output), str(wheel)])
    previous = sys.argv
    observer = ProcessObserver(evidence / 'helpers')
    try:
        sys.argv = argv
        with observer:
            entry = entries[0].load()
            try:
                status = entry()
            except SystemExit as exc:
                status = exc.code
            _require(status in (None, 0), 'repair entry point failed')
    finally:
        sys.argv = previous
    _require(_distribution_inputs(distribution) == inputs,
             'repair implementation changed while running')
    loaded = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).is_file():
            loaded.add(str(Path(filename).resolve()))
    report = {
        'schema': 'pyvoro2-native-repair-observation-v1', 'kind': kind,
        'distribution': {'name': package, 'version': distribution.version,
                         'entry_point': entries[0].value, 'files': inputs},
        'loaded_modules': [identity(path) for path in sorted(loaded)],
        'processes': observer.records,
    }
    (evidence / 'repair-observation.json').write_bytes(canonical(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('auditwheel', 'delocate', 'delvewheel'),
                        required=True)
    for name in ('wheel', 'output', 'evidence'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_repair(kind=args.kind, wheel=args.wheel.resolve(), output=args.output,
                   evidence=args.evidence)
    except (RepairObservationError, OSError, ValueError) as exc:
        parser.exit(1, f'repair observation refused: {exc}\n')


if __name__ == '__main__':
    main()
