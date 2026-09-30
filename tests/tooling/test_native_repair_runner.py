"""Repair implementation and actual helper-process observation."""
from __future__ import annotations

import importlib.util
import hashlib
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def repair_runner():
    path = ROOT / 'tools/native/qualification/repair_runner.py'
    spec = importlib.util.spec_from_file_location('isolated_repair_runner', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('encoding', [None, 'locale', 'utf-8', 'utf-16-le'])
def test_observer_retains_communicate_output_without_changing_return_value(
        repair_runner, tmp_path, encoding):
    observer = repair_runner.ProcessObserver(tmp_path)
    output = ('helper output\n'.encode(encoding)
              if encoding not in (None, 'locale') else b'helper output\n')
    with observer:
        result = subprocess.run(
            [sys.executable, '-c',
             'import sys; sys.stdout.buffer.write(' + repr(output) + ')'],
            capture_output=True, text=True, encoding=encoding, check=True)
    assert result.stdout == 'helper output\n'
    row, = observer.records
    assert row['exit_code'] == 0
    assert row['argv'][0] == sys.executable
    assert Path(row['stdout']['path']).read_bytes() == output
    assert row['stdout_encoding'] == observer.processes[0].stdout.encoding
    assert row['executable']['sha256']


def test_opaque_shell_is_refused_before_execution(repair_runner, tmp_path):
    with repair_runner.ProcessObserver(tmp_path):
        with pytest.raises(repair_runner.RepairObservationError, match='shell'):
            subprocess.run('exit 0', shell=True, check=True)


def test_explicit_empty_helper_environment_is_recorded_exactly(repair_runner, tmp_path):
    command = [sys.executable, '-c', 'pass']
    empty_block_unsupported = False
    if sys.platform == 'win32' and sys.version_info[:2] == (3, 10):
        # CPython gh-105436: the older Windows launcher does not terminate an
        # empty environment block correctly. Establish the real unwrapped
        # outcome first; the observer must preserve both the error and env={}.
        try:
            subprocess.run(command, env={}, check=True)
        except OSError as error:
            if error.winerror != 87:
                raise
            empty_block_unsupported = True
    observer = repair_runner.ProcessObserver(tmp_path)
    outcome = pytest.raises(OSError) if empty_block_unsupported else nullcontext()
    with outcome as caught:
        with observer:
            subprocess.run(command, env={}, check=True)
    row, = observer.records
    assert row['environment'] == {}
    assert row['environment_sha256'] == hashlib.sha256(
        repair_runner.canonical({})).hexdigest()
    if empty_block_unsupported:
        assert caught.value.winerror == 87
        assert row['exit_code'] is None
        assert observer.processes == []
    else:
        assert row['exit_code'] == 0


def test_explicit_helper_environment_does_not_inherit_parent(
        repair_runner, tmp_path, monkeypatch):
    monkeypatch.setenv('ISSUE88_PARENT_ONLY', 'must not reach helper')
    environment = {'ISSUE88_CHILD_ONLY': 'observed'}
    if sys.platform == 'win32':
        environment['SYSTEMROOT'] = os.environ['SYSTEMROOT']
    observer = repair_runner.ProcessObserver(tmp_path)
    with observer:
        subprocess.run([sys.executable, '-c',
                        'import os; assert "ISSUE88_PARENT_ONLY" not in os.environ; '
                        'assert os.environ["ISSUE88_CHILD_ONLY"] == "observed"'],
                       env=environment, check=True)
    row, = observer.records
    assert row['exit_code'] == 0
    assert row['environment'] == {key: value for key, value in environment.items()
                                  if key == 'SYSTEMROOT'}
    assert row['environment_sha256'] == hashlib.sha256(
        repair_runner.canonical(environment)).hexdigest()


def test_unconsumed_pipe_refuses_complete_receipt(repair_runner, tmp_path):
    observer = repair_runner.ProcessObserver(tmp_path)
    with pytest.raises(repair_runner.RepairObservationError, match='PIPE'):
        with observer:
            process = subprocess.Popen([sys.executable, '-c', 'print("lost")'],
                                       stdout=subprocess.PIPE)
            process.wait()
    process.stdout.close()
