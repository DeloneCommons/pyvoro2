"""Measure, check or explicitly refresh the mechanical native source identity.

Measurement is read-only. The implementer owns manifest refresh; controlled
builds and finalization only check it. Records and installation anchors remain
outputs, never inputs to their own source identities.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
_SKIP_DIRS = frozenset(('__pycache__', '.git', '.pytest_cache'))
_OUTPUT_SUFFIXES = frozenset(('.pyc', '.pyo', '.o', '.obj', '.a', '.so',
                              '.pyd', '.dll', '.dylib'))
_EXCLUDED = frozenset((
    'src/pyvoro2/_internal/native_source_manifest.json',
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
        # Loading the private contract must not create a source-tree bytecode
        # cache: measure/check are read-only and refresh writes only its output.
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
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


def _raise_walk_error(error):
    raise error


def _source_path_kind(path):
    # Python 3.14's Path predicates suppress stat errors. Preserve ordinary
    # missing-path behavior, but never omit an unreadable source input.
    try:
        return stat.S_IFMT(path.stat().st_mode)
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP):
            return 0
        raise


def measure_source(root: Path = ROOT) -> dict:
    """Read all relevant files, including unknown additions; never write."""
    root = Path(root).resolve()
    contract = _contract(root)
    paths = set()
    for name in _FILES:
        path = root / name
        if _source_path_kind(path) != stat.S_IFREG:
            raise ValueError(f'missing source policy input: {name}')
        paths.add(path)
    for name in _TREES:
        tree = root / name
        if _source_path_kind(tree) != stat.S_IFDIR:
            if name in _TREES[:4]:
                raise ValueError(f'missing source closure: {name}')
            continue
        # rglob silently omits unreadable directories, which would publish an
        # incomplete identity. Keep the same file filters but fail traversal.
        for directory, subdirectories, filenames in os.walk(
                tree, onerror=_raise_walk_error):
            subdirectories[:] = [name for name in subdirectories
                                 if name not in _SKIP_DIRS]
            paths.update(path for name in filenames
                         if _source_path_kind(path := Path(directory) / name)
                         == stat.S_IFREG)
    files = {}
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        if not _included(path, relative):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'symlink is not a source policy input: {relative}')
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


def source_manifest(measurement: dict, contract) -> dict:
    """Project full measurement onto the six-key committed identity."""
    manifest = {'manifest_schema': contract.SOURCE_MANIFEST_SCHEMA,
                **{field: measurement[field] for field in (
                    'policy_revision', 'source_sha256', 'schema_sha256',
                    'consumer_sha256', 'components')}}
    contract.validate_source_manifest(manifest)
    return manifest


def _destination(internal: Path, contract) -> Path:
    retired = internal / 'native_approval.json'
    if retired.exists() or retired.is_symlink():
        raise ValueError('forbidden retired file: native_approval.json; remove it')
    path = internal / contract.SOURCE_MANIFEST_FILENAME
    # Never repair missing source directories or follow destination parents.
    for parent in (internal.parent.parent, internal.parent, internal):
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError(f'unsafe or missing source manifest parent: {parent}')
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError('source manifest destination must be a regular file')
    return path


def read_manifest(internal: Path, contract, *, expected=None) -> tuple[dict, bytes]:
    """Read only the fixed canonical manifest; reject the retired path."""
    path = _destination(internal, contract)
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError('missing source manifest; run --update-manifest') from exc
    try:
        manifest = contract.parse_source_manifest(data)
    except (ValueError, UnicodeError, TypeError) as exc:
        differences = []
        if expected is not None:
            try:
                old = json.loads(data, object_pairs_hook=contract._unique_object,
                                 parse_float=contract._reject_number,
                                 parse_constant=contract._reject_number)
            except (ValueError, UnicodeError, TypeError):
                pass
            else:
                differences = field_differences(old, expected)
        detail = '\n' + '\n'.join(differences) if differences else ''
        raise ValueError(f'invalid source manifest: {exc}{detail}') from exc
    return manifest, data


def field_differences(old, new, prefix='') -> list[str]:
    """Sorted dotted-field changes, independent of inventories/debugging data."""
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(old.keys() | new.keys()):
            name = f'{prefix}.{key}' if prefix else key
            if key not in old or key not in new:
                result.append(f'{name}: recorded={old.get(key)!r} '
                              f'measured={new.get(key)!r}')
            else:
                result.extend(field_differences(old[key], new[key], name))
        return result
    return ([] if old == new else
            [f'{prefix}: recorded={old!r} measured={new!r}'])


def check_manifest(measurement: dict, root: Path = ROOT) -> tuple[dict, bytes]:
    root = Path(root).resolve()
    contract = _contract(root)
    expected = source_manifest(measurement, contract)
    manifest, data = read_manifest(root / 'src/pyvoro2/_internal', contract,
                                   expected=expected)
    differences = field_differences(manifest, expected)
    if differences:
        raise ValueError('stale source manifest:\n' + '\n'.join(differences)
                         + '\nrun --update-manifest')
    return manifest, data


def update_manifest(root: Path = ROOT) -> tuple[str, bytes, list[str], dict]:
    root = Path(root).resolve()
    measurement = measure_source(root)
    contract = _contract(root)
    manifest = source_manifest(measurement, contract)
    data = contract.canonical_json(manifest)
    internal = root / 'src/pyvoro2/_internal'
    path = _destination(internal, contract)
    old_data = path.read_bytes() if path.exists() else None
    mode = stat.S_IMODE(path.stat().st_mode) if old_data is not None else 0o644
    details = []
    state = 'created'
    if old_data is not None:
        try:
            old = contract.parse_source_manifest(old_data, require_canonical=False)
        except (ValueError, UnicodeError, TypeError) as exc:
            state = 'replaced-invalid'
            details.append('previous_raw_sha256='
                           + hashlib.sha256(old_data).hexdigest())
            details.append(f'previous validation failure: {exc}')
            try:
                old = json.loads(old_data, object_pairs_hook=contract._unique_object,
                                 parse_float=contract._reject_number,
                                 parse_constant=contract._reject_number)
            except (ValueError, UnicodeError, TypeError):
                old = None
        else:
            state = ('unchanged' if old_data == data else
                     'normalized' if old == manifest else 'updated')
        if old is not None:
            details.extend(field_differences(old, manifest))
    if state == 'unchanged':
        return state, data, details, measurement
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.',
                                            suffix='.tmp', dir=internal)
        temporary = Path(name)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        _destination(internal, contract)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return state, data, details, measurement


def _summary(state, data, measurement):
    return (f'{state}: source_manifest_sha256={hashlib.sha256(data).hexdigest()} '
            f'files={len(measurement["files"])} '
            f'consumers={len(measurement["consumers"])}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--measure', action='store_true',
                         help='emit measurement JSON (the default action)')
    actions.add_argument('--check-manifest', action='store_true')
    actions.add_argument('--update-manifest', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.update_manifest:
            state, data, details, measurement = update_manifest(args.root)
            print(_summary(state, data, measurement))
            if details:
                print('\n'.join(details))
        elif args.check_manifest:
            measurement = measure_source(args.root)
            _, data = check_manifest(measurement, args.root)
            print(_summary('current', data, measurement))
        else:
            measurement = measure_source(args.root)
            print(json.dumps(measurement, sort_keys=True, indent=2))
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(1, f'source manifest operation failed: {exc}\n'
                    'refresh explicitly: python tools/native/qualification/'
                    'source_policy.py --root ROOT --update-manifest\n')


if __name__ == '__main__':
    main()
