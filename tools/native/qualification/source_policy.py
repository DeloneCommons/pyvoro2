"""Measure the conservative native/consumer closure without issuing approval.

The approval JSON and generated installation anchor are intentionally excluded
from the source digest. The detached record separately binds canonical approval
identity and the generated anchor's installation ID. Qualification records are
also outputs, never inputs to their own source identities.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
_SKIP_DIRS = frozenset(('__pycache__', '.git', '.pytest_cache'))
_OUTPUT_SUFFIXES = frozenset(('.pyc', '.pyo', '.o', '.obj', '.a', '.so',
                              '.pyd', '.dll', '.dylib'))
_EXCLUDED = frozenset((
    'src/pyvoro2/_internal/native_approval.json',
    'src/pyvoro2/_internal/_qualification_installation.py',
    'src/pyvoro2/_internal/native_qualification_record.json',
))
_TREES = ('vendor/voro++', 'cpp', 'cmake', 'src/pyvoro2', 'tools', 'tests',
          '.github/workflows')
_FILES = ('CMakeLists.txt', 'pyproject.toml')


def _contract(root):
    path = root / 'src/pyvoro2/_internal/native_qualification.py'
    name = '_pyvoro2_source_policy_contract'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous
    return module


def _included(path, relative):
    return (relative not in _EXCLUDED
            and not any(part in _SKIP_DIRS for part in path.parts)
            and path.suffix not in _OUTPUT_SUFFIXES)


def measure_source(root: Path = ROOT) -> dict:
    """Read all relevant files, including unknown additions; never update approval."""
    root = Path(root).resolve()
    contract = _contract(root)
    paths = set()
    for name in _FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(f'missing source policy input: {name}')
        paths.add(path)
    for name in _TREES:
        tree = root / name
        if name in _TREES[:4] and not tree.is_dir():
            raise ValueError(f'missing source closure: {name}')
        paths.update(path for path in tree.rglob('*') if path.is_file())
    files = {}
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        if not _included(path, relative):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'symlink is not a reviewed source input: {relative}')
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    consumers = {name.removeprefix('src/'): digest
                 for name, digest in files.items()
                 if name.startswith('src/pyvoro2/') and name.endswith('.py')}
    components = {}
    for name, schema in contract.COMPONENT_SCHEMAS.items():
        opposite = ('vendor/voro++/src/' if name.endswith('-planar')
                    else 'vendor/voro++/2d/')
        closure = {path: digest for path, digest in files.items()
                   if not path.startswith(opposite)}
        components[name] = {
            'source_sha256': contract.canonical_sha256(
                {'component': name, 'files': closure}),
            'schema_sha256': contract.canonical_sha256(schema),
        }
    return {'measurement_schema': 'pyvoro2-native-source-measurement-v1',
            'policy_revision': contract.POLICY_REVISION,
            'source_sha256': contract.canonical_sha256(files),
            'schema_sha256': contract.canonical_sha256(contract.COMPONENT_SCHEMAS),
            'consumer_sha256': contract.canonical_sha256(consumers),
            'files': files, 'consumers': consumers, 'components': components}


def check_approval(measurement: dict, approval: dict) -> None:
    """Compare separately reviewed literals; no measurement grants approval."""
    if (approval.get('approval_schema') != 'pyvoro2-native-source-approval-v1'
            or approval.get('approved') is not True):
        raise ValueError('source closure has no independent approval')
    for field in ('policy_revision', 'source_sha256', 'schema_sha256',
                  'consumer_sha256', 'components'):
        if approval.get(field) != measurement.get(field):
            raise ValueError(f'reviewed source approval differs: {field}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--measure', action='store_true',
                        help='emit measurement JSON (the default action)')
    parser.add_argument('--check-approval', action='store_true')
    args = parser.parse_args(argv)
    measurement = measure_source(args.root)
    if args.check_approval:
        approval = json.loads((args.root / 'src/pyvoro2/_internal/'
                               'native_approval.json').read_text(encoding='utf8'))
        check_approval(measurement, approval)
    print(json.dumps(measurement, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
