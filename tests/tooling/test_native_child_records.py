"""Immutable GNU child lifecycle evidence; fixture receipts do not qualify code."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
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
def test_actual_child_keeps_initial_receipt_and_publishes_bound_completion(tmp_path):
    process = subprocess.run(
        [sys.executable, str(ROOT / 'tools/native/qualification/child_record.py'),
         '--directory', str(tmp_path), '--', sys.executable, '-c', 'pass'],
        text=True, capture_output=True, check=False)
    assert process.returncode == 0, process.stderr
    invocations, evidence = child_record.collect_child_receipts(tmp_path)
    assert len(invocations) == 1
    assert invocations[0]['exit_code'] == 0
    assert len(evidence) == 2
    assert len(list(tmp_path.glob('*.start.json'))) == 1
    assert len(list(tmp_path.glob('*.complete.json'))) == 1
    assert not list(tmp_path.glob('*.pending'))
