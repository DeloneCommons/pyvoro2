"""Repair implementation and actual helper-process observation."""
from __future__ import annotations

import importlib.util
import hashlib
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


def test_observer_retains_communicate_output_without_changing_return_value(
        repair_runner, tmp_path):
    observer = repair_runner.ProcessObserver(tmp_path)
    with observer:
        result = subprocess.run(
            [sys.executable, '-c', 'print("helper output")'],
            capture_output=True, text=True, check=True)
    assert result.stdout == 'helper output\n'
    row, = observer.records
    assert row['exit_code'] == 0
    assert row['argv'][0] == sys.executable
    assert Path(row['stdout']['path']).read_text() == result.stdout
    assert row['executable']['sha256']


def test_opaque_shell_is_refused_before_execution(repair_runner, tmp_path):
    with repair_runner.ProcessObserver(tmp_path):
        with pytest.raises(repair_runner.RepairObservationError, match='shell'):
            subprocess.run('exit 0', shell=True, check=True)


def test_explicit_empty_helper_environment_is_recorded_exactly(repair_runner, tmp_path):
    observer = repair_runner.ProcessObserver(tmp_path)
    with observer:
        subprocess.run([sys.executable, '-c', 'pass'], env={}, check=True)
    row, = observer.records
    assert row['environment'] == {}
    assert row['environment_sha256'] == hashlib.sha256(
        repair_runner.canonical({})).hexdigest()


def test_unconsumed_pipe_refuses_complete_receipt(repair_runner, tmp_path):
    observer = repair_runner.ProcessObserver(tmp_path)
    with pytest.raises(repair_runner.RepairObservationError, match='PIPE'):
        with observer:
            process = subprocess.Popen([sys.executable, '-c', 'print("lost")'],
                                       stdout=subprocess.PIPE)
            process.wait()
    process.stdout.close()
