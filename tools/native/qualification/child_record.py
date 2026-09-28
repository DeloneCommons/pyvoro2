#!/usr/bin/env python3
"""GCC callback with immutable start and separately published completion receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qualification.adapters import tool_role  # noqa: E402
from qualification.effective_build import (  # noqa: E402
    BuildEvidenceError, canonical, expand_response, file_identity,
)
from qualification.loader_trace import (  # noqa: E402
    capture_glibc, glibc_environment,
)


START_SCHEMA = 'pyvoro2.gnu-child-start.v1'
COMPLETION_SCHEMA = 'pyvoro2.gnu-child-completion.v1'
_RECEIPT_NAME = re.compile(r'([0-9a-f]{32})\.(start|complete)\.json\Z')
_INITIAL_FIELDS = frozenset((
    'pid', 'executable', 'argv', 'expanded_argv', 'cwd', 'environment', 'role',
    'opened_files', 'response_files', 'observation',
))
_RESULT_FIELDS = frozenset((
    'exit_code', 'executable_unchanged', 'responses_unchanged', 'loader',
))


def _require(condition, message):
    if not condition:
        raise BuildEvidenceError(message)


def _receipt_data(receipt):
    return canonical(receipt) + b'\n'


def _read_receipt(path, schema, identifier):
    try:
        data = path.read_bytes()
        receipt = json.loads(data)
    except (OSError, ValueError) as error:
        raise BuildEvidenceError(
            'truncated or unavailable GNU child receipt') from error
    _require(isinstance(receipt, dict) and receipt.get('schema') == schema
             and receipt.get('receipt_id') == identifier
             and data == _receipt_data(receipt),
             'noncanonical or incompatible GNU child receipt')
    return receipt, data


def collect_child_receipts(directory):
    """Require exactly one immutable start and bound completion for each child.

    Atomic-write staging is not an observation and never completes a start.
    Its presence or deletion cannot change whether published evidence qualifies.
    Legacy pending/single-file receipts require a fresh build, not reinterpretation.
    """
    starts, completions = {}, {}
    for path in Path(directory).iterdir():
        if not path.is_file():
            continue
        match = _RECEIPT_NAME.fullmatch(path.name)
        if match:
            target = starts if match[2] == 'start' else completions
            target[match[1]] = path
        elif path.suffix in ('.pending', '.json'):
            raise BuildEvidenceError('legacy or unrecognized GNU child receipt')
    _require(starts.keys() == completions.keys(),
             'unmatched GNU child start/completion receipts')
    _require(starts, 'no immutable GNU child start receipts')
    invocations, evidence = [], []
    for identifier in sorted(starts):
        start, start_data = _read_receipt(starts[identifier], START_SCHEMA, identifier)
        complete, _ = _read_receipt(completions[identifier], COMPLETION_SCHEMA,
                                    identifier)
        _require(set(start) == {'schema', 'receipt_id', 'invocation'}
                 and set(complete) == {'schema', 'receipt_id', 'start_sha256',
                                       'invocation'}
                 and complete['start_sha256'] == hashlib.sha256(start_data).hexdigest(),
                 'GNU child completion is not bound to its immutable start')
        initial, final = start['invocation'], complete['invocation']
        _require(isinstance(initial, dict) and set(initial) == _INITIAL_FIELDS
                 and isinstance(final, dict)
                 and set(final) == _INITIAL_FIELDS | _RESULT_FIELDS,
                 'incomplete GNU child start/completion invocation')
        _require(type(initial['pid']) is int and initial['pid'] > 0
                 and isinstance(initial['cwd'], str)
                 and isinstance(initial['environment'], dict)
                 and isinstance(initial['executable'], dict)
                 and isinstance(initial['response_files'], list)
                 and initial['observation'] == 'direct-child-execution'
                 and all(isinstance(initial[key], list) and initial[key]
                         and all(isinstance(arg, str) for arg in initial[key])
                         for key in ('argv', 'expanded_argv')),
                 'incomplete original GNU child invocation')
        _require(all(final[key] == value for key, value in initial.items()
                     if key != 'opened_files'),
                 'GNU child completion changed an original invocation field')
        _require(type(final['exit_code']) is int
                 and type(final['executable_unchanged']) is bool
                 and type(final['responses_unchanged']) is bool
                 and isinstance(final['loader'], dict)
                 and isinstance(final['loader'].get('images'), list),
                 'incomplete GNU child completion outcome')
        images = final['loader']['images']
        _require(all(isinstance(item, dict) and isinstance(item.get('path'), str)
                     for item in images), 'incomplete GNU child loader image evidence')
        _require(all(isinstance(row['opened_files'], list)
                     and all(isinstance(path, str) for path in row['opened_files'])
                     and len(row['opened_files']) == len(set(row['opened_files']))
                     for row in (initial, final))
                 and set(final['opened_files']) == (
                     set(initial['opened_files']) | {item['path'] for item in images}),
                 'GNU child opened_files changed without loader evidence')
        invocations.append(final)
        evidence.extend((starts[identifier], completions[identifier]))
    return invocations, evidence


def _write_start(path, receipt):
    data = _receipt_data(receipt)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def _publish_completion(path, receipt):
    staging = path.parent / '.staging'
    staging.mkdir(exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=staging)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(_receipt_data(receipt))
        stream.flush()
        os.fsync(stream.fileno())
    _require(not path.exists(), 'GNU child completion already exists')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('missing child command')
    directory = args.directory.resolve()
    cwd = Path.cwd().resolve()
    executable = Path(shutil.which(command[0]) or command[0]).resolve(strict=True)
    command[0] = str(executable)
    responses = {}
    expanded = expand_response(command, cwd, snapshots=responses)
    identity = file_identity(executable)
    identifier = uuid.uuid4().hex
    prefix = directory / (identifier + '-loader')
    environment = glibc_environment(dict(os.environ), prefix)
    record = {
        'pid': os.getpid(), 'executable': identity, 'argv': command,
        'expanded_argv': expanded, 'cwd': str(cwd),
        'environment': environment, 'role': tool_role(executable),
        'opened_files': [], 'response_files': list(responses.values()),
        'observation': 'direct-child-execution',
    }
    start = {'schema': START_SCHEMA, 'receipt_id': identifier, 'invocation': record}
    start_sha256 = _write_start(directory / (identifier + '.start.json'), start)
    run = subprocess.run(command, env=environment)
    record['exit_code'] = run.returncode
    record['executable_unchanged'] = file_identity(executable) == identity
    record['responses_unchanged'] = all(
        file_identity(row['path'])['sha256'] == row['sha256']
        for row in responses.values())
    record['loader'] = capture_glibc(prefix)
    record['opened_files'] = [item['path'] for item in record['loader']['images']]
    complete = {'schema': COMPLETION_SCHEMA, 'receipt_id': identifier,
                'start_sha256': start_sha256, 'invocation': record}
    _publish_completion(directory / (identifier + '.complete.json'), complete)
    return run.returncode


if __name__ == '__main__':
    raise SystemExit(main())
