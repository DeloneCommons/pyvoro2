"""Immutable GNU child lifecycle evidence; fixture receipts do not qualify code."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/native'))

from qualification import child_record  # noqa: E402
from qualification.effective_build import BuildEvidenceError  # noqa: E402


@pytest.fixture
def receipt_pair(tmp_path):
    identifier = 'a' * 32
    executable = child_record.file_identity(sys.executable)
    invocation = {
        'pid': 123, 'executable': executable,
        'argv': [sys.executable, '-c', 'pass'],
        'expanded_argv': [sys.executable, '-c', 'pass'],
        'cwd': str(tmp_path), 'environment': {'PATH': 'fixture'},
        'role': 'compiler_backend', 'opened_files': [], 'response_files': [],
        'observation': 'direct-child-execution',
    }
    start = {'schema': 'pyvoro2.gnu-child-start.v1', 'receipt_id': identifier,
             'invocation': invocation}
    start_data = child_record.canonical(start) + b'\n'
    complete = {
        'schema': 'pyvoro2.gnu-child-completion.v1', 'receipt_id': identifier,
        'start_sha256': hashlib.sha256(start_data).hexdigest(),
        'invocation': {**copy.deepcopy(invocation), 'exit_code': 0,
                       'executable_unchanged': True, 'responses_unchanged': True,
                       'loader': {'images': [executable], 'programs': [executable],
                                  'evidence_files': []},
                       'opened_files': [executable['path']]},
    }
    start_path = tmp_path / (identifier + '.start.json')
    complete_path = tmp_path / (identifier + '.complete.json')
    start_path.write_bytes(start_data)
    complete_path.write_bytes(child_record.canonical(complete) + b'\n')
    return tmp_path, start_path, complete_path, start, complete


def test_start_remains_immutable_after_completion(receipt_pair):
    directory, start_path, complete_path, start, complete = receipt_pair
    before = start_path.read_bytes()
    invocations, evidence = child_record.collect_child_receipts(directory)
    assert invocations == [complete['invocation']]
    assert set(evidence) == {start_path, complete_path}
    assert start_path.read_bytes() == before
    assert json.loads(before) == start


def test_start_writer_is_exclusive_and_binds_the_persisted_bytes(receipt_pair):
    _, start_path, _, start, _ = receipt_pair
    expected = start_path.read_bytes()
    start_path.unlink()
    digest = child_record._write_start(start_path, start)
    assert start_path.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    with pytest.raises(FileExistsError):
        child_record._write_start(start_path, {'replacement': True})
    assert start_path.read_bytes() == expected


def test_completion_is_synced_before_atomic_publication(receipt_pair, monkeypatch):
    directory, start_path, complete_path, _, complete = receipt_pair
    start_data = start_path.read_bytes()
    expected = complete_path.read_bytes()
    complete_path.unlink()
    fsync, replace = os.fsync, os.replace
    synced = []

    def observe_fsync(descriptor):
        assert not complete_path.exists()
        staged, = (directory / '.staging').iterdir()
        assert staged.read_bytes() == expected
        if os.name == 'posix':
            assert stat.S_IMODE(staged.stat().st_mode) == 0o644
        fsync(descriptor)
        synced.append(True)

    def observe_replace(source, target):
        assert synced
        assert Path(source).read_bytes() == expected
        assert not complete_path.exists()
        assert Path(target) == complete_path
        with pytest.raises(BuildEvidenceError, match='unmatched'):
            child_record.collect_child_receipts(directory)
        replace(source, target)

    monkeypatch.setattr(child_record.os, 'fsync', observe_fsync)
    monkeypatch.setattr(child_record.os, 'replace', observe_replace)
    child_record._publish_completion(complete_path, complete)
    assert complete_path.read_bytes() == expected
    assert start_path.read_bytes() == start_data
    assert child_record.collect_child_receipts(directory)[0] == [
        complete['invocation']]


@pytest.mark.parametrize('operation', ['chmod', 'fsync', 'replace'])
def test_completion_publication_failure_leaves_unmatched_start(
        receipt_pair, monkeypatch, operation):
    directory, start_path, complete_path, _, complete = receipt_pair
    before = start_path.read_bytes()
    complete_path.unlink()

    def fail(*args):
        raise OSError('injected publication failure')

    monkeypatch.setattr(child_record.os, operation, fail)
    with pytest.raises(OSError, match='injected publication failure'):
        child_record._publish_completion(complete_path, complete)
    assert not complete_path.exists()
    assert start_path.read_bytes() == before
    with pytest.raises(BuildEvidenceError, match='unmatched'):
        child_record.collect_child_receipts(directory)


def test_completion_writer_refuses_to_replace_published_receipt(receipt_pair):
    _, _, complete_path, _, _ = receipt_pair
    before = complete_path.read_bytes()
    with pytest.raises(BuildEvidenceError, match='already exists'):
        child_record._publish_completion(complete_path, {'replacement': True})
    assert complete_path.read_bytes() == before


@pytest.mark.parametrize('field,value', [
    ('pid', 456), ('argv', ['different']), ('expanded_argv', ['different']),
    ('cwd', '/different'), ('environment', {'PATH': 'different'}),
    ('executable', {'path': '/different', 'size': 1, 'sha256': 'f' * 64}),
    ('role', 'linker'), ('response_files', [{'different': True}]),
    ('observation', 'different'),
])
def test_completion_cannot_change_an_original_invocation_field(
        receipt_pair, field, value):
    directory, _, complete_path, _, complete = receipt_pair
    complete['invocation'][field] = value
    complete_path.write_bytes(child_record.canonical(complete) + b'\n')
    with pytest.raises(BuildEvidenceError, match='start|original|invocation'):
        child_record.collect_child_receipts(directory)


@pytest.mark.parametrize('damage', [
    'missing_start', 'missing_completion', 'truncated_start', 'truncated_completion',
    'wrong_start_hash', 'wrong_id', 'missing_exit', 'invented_opened_file',
    'extra_start_field', 'legacy_pending', 'legacy_json',
])
def test_incomplete_or_mismatched_child_lifecycle_refuses(receipt_pair, damage):
    directory, start_path, complete_path, start, complete = receipt_pair
    if damage == 'missing_start':
        start_path.unlink()
    elif damage == 'missing_completion':
        complete_path.unlink()
    elif damage == 'truncated_start':
        start_path.write_bytes(b'{')
    elif damage == 'truncated_completion':
        complete_path.write_bytes(b'{')
    elif damage == 'legacy_pending':
        (directory / 'old.pending').write_bytes(b'{}')
    elif damage == 'legacy_json':
        (directory / 'old.json').write_bytes(b'{}')
    elif damage == 'extra_start_field':
        start['invocation']['new_original_field'] = 'must be preserved'
        data = child_record.canonical(start) + b'\n'
        start_path.write_bytes(data)
        complete['start_sha256'] = hashlib.sha256(data).hexdigest()
        complete_path.write_bytes(child_record.canonical(complete) + b'\n')
    else:
        if damage == 'wrong_start_hash':
            complete['start_sha256'] = '0' * 64
        elif damage == 'wrong_id':
            complete['receipt_id'] = 'b' * 32
        elif damage == 'missing_exit':
            del complete['invocation']['exit_code']
        elif damage == 'invented_opened_file':
            complete['invocation']['opened_files'].append('/not-an-observed-image')
        complete_path.write_bytes(child_record.canonical(complete) + b'\n')
    with pytest.raises(BuildEvidenceError):
        child_record.collect_child_receipts(directory)


def test_staging_files_cannot_issue_completion_or_hide_unmatched_start(receipt_pair):
    directory, _, complete_path, _, _ = receipt_pair
    staging = directory / '.staging'
    staging.mkdir()
    (staging / 'resurrected-temporary').write_bytes(b'truncated staging data')
    assert len(child_record.collect_child_receipts(directory)[0]) == 1
    complete_path.unlink()
    with pytest.raises(BuildEvidenceError, match='completion|unmatched'):
        child_record.collect_child_receipts(directory)


@pytest.mark.skipif(sys.platform != 'linux', reason='GNU/glibc child observation')
@pytest.mark.parametrize('umask,start_mode', [(0o022, 0o644), (0o002, 0o664)])
def test_actual_child_keeps_initial_receipt_and_publishes_bound_completion(
        tmp_path, umask, start_mode):
    process = subprocess.run(
        [sys.executable, str(ROOT / 'tools/native/qualification/child_record.py'),
         '--directory', str(tmp_path), '--', sys.executable, '-c', 'pass'],
        text=True, capture_output=True, check=False, umask=umask)
    assert process.returncode == 0, process.stderr
    invocations, evidence = child_record.collect_child_receipts(tmp_path)
    assert len(invocations) == 1
    assert invocations[0]['exit_code'] == 0
    assert len(evidence) == 2
    start_path, = tmp_path.glob('*.start.json')
    complete_path, = tmp_path.glob('*.complete.json')
    assert stat.S_IMODE(start_path.stat().st_mode) == start_mode
    # The host artifact uploader is not the container's root writer. Completion
    # evidence must be readable across that boundary without other-user writes.
    assert stat.S_IMODE(complete_path.stat().st_mode) == 0o644
    assert not list(tmp_path.glob('*.pending'))
