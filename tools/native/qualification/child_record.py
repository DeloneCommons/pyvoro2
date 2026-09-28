#!/usr/bin/env python3
"""GCC -wrapper/fixed-linker callback: record then execute the actual child."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qualification.adapters import tool_role  # noqa: E402
from qualification.effective_build import (  # noqa: E402
    canonical, expand_response, file_identity,
)
from qualification.loader_trace import (  # noqa: E402
    capture_glibc, glibc_environment,
)


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
    pending = directory / (identifier + '.pending')
    pending.write_bytes(canonical(record) + b'\n')
    run = subprocess.run(command, env=environment)
    record['exit_code'] = run.returncode
    record['executable_unchanged'] = file_identity(executable) == identity
    record['responses_unchanged'] = all(
        file_identity(row['path'])['sha256'] == row['sha256']
        for row in responses.values())
    record['loader'] = capture_glibc(prefix)
    record['opened_files'] = [item['path'] for item in record['loader']['images']]
    pending.write_bytes(canonical(record) + b'\n')
    pending.replace(directory / (identifier + '.json'))
    return run.returncode


if __name__ == '__main__':
    raise SystemExit(main())
