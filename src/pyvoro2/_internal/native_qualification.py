"""Private, externally issued native artifact qualification.

The generated installation module is the trust anchor. Candidate metadata and
an adjacent JSON file cannot issue qualification. The installation is immutable
from native import through its last use: registration binds the imported payload,
Linux also checks its mapped device/inode, and subsequent changes fail closed.
Replacing this verifier and its trusted installation is outside the numerical
build contract; this is not a native-code security sandbox.

The owning route must check the current executing-thread FP state before this
filesystem verifier. Even stat calls can construct floating timestamps inside
Python. This verifier does not inspect or cache the mutable FP control state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import struct
import sys
import sysconfig
import threading
from types import ModuleType


RECORD_SCHEMA = 'pyvoro2-native-qualification-v1'
APPROVAL_SCHEMA = 'pyvoro2-native-source-approval-v1'
POLICY_REVISION = 'issue88-p1'
COMPONENT_SCHEMAS = {
    'wp5-spatial': {'producer': 'wp5-native-occurrences-v1',
                    'consumer': 'wp5-N-E-S-v1'},
    'wp6-planar': {'producer': 'pyvoro2.planar.occurrences.v1',
                   'consumer': 'wp6-direct-occurrences-v1'},
    'wp7-spatial': {'producer': 'wp7-selected-ghost-3d-v1',
                    'consumer': 'wp7-spatial-selected-v1'},
    'wp7-planar': {'producer': 'pyvoro2.planar.occurrences.v1',
                   'consumer': 'wp7-planar-selected-v1'},
    'wp8-spatial': {'producer': 'locate-source-v1',
                    'consumer': 'wp8-spatial-enclosure-v1'},
    'wp8-planar': {'producer': 'locate-source-v1',
                   'consumer': 'wp8-planar-enclosure-v1'},
}
COMPONENT_REQUIREMENTS = {
    name: ((('wp5-spatial', name) if name == 'wp7-spatial' else
            ('wp6-planar', name)) if name.startswith('wp7-') else (name,))
    for name in COMPONENT_SCHEMAS
}
_ORDINARY_COMPONENTS = frozenset(('wp5-spatial', 'wp8-spatial', 'wp8-planar'))
_ALL_COMPONENTS = frozenset(COMPONENT_SCHEMAS)
# These select reviewed property adapters, never an exact compiler version.
ADAPTERS = {
    'gnu-linux-x86_64-v1': ('linux', 'x86_64', 'GNU', _ALL_COMPONENTS),
    'clang-linux-x86_64-v1': ('linux', 'x86_64', 'Clang', _ALL_COMPONENTS),
    'appleclang-darwin-x86_64-v1':
        ('darwin', 'x86_64', 'AppleClang', _ORDINARY_COMPONENTS),
    'appleclang-darwin-arm64-v1':
        ('darwin', 'arm64', 'AppleClang', _ORDINARY_COMPONENTS),
    'msvc-win32-x86_64-v1': ('win32', 'x86_64', 'MSVC', _ORDINARY_COMPONENTS),
}
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
_EXCLUDED_CONSUMER = 'pyvoro2/_internal/_qualification_installation.py'


class NativeQualificationError(RuntimeError):
    """Private reason codes are translated by the owning WP5/WP6/GHOST/LOCATE route."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f'native qualification:{reason}: {detail}')


def _refuse(reason, detail):
    raise NativeQualificationError(reason, detail)


def canonical_json(value) -> bytes:
    """One serialization for source, record, evidence, and approval identities."""
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def canonical_sha256(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def current_target() -> dict:
    machine = platform.machine().lower()
    machine = {'amd64': 'x86_64', 'aarch64': 'arm64'}.get(machine, machine)
    return {'platform': sys.platform, 'machine': machine,
            'implementation': sys.implementation.name,
            'cache_tag': sys.implementation.cache_tag,
            'soabi': sysconfig.get_config_var('SOABI') or '',
            'pointer_bits': struct.calcsize('P') * 8,
            'byteorder': sys.byteorder}


def _is_digest(value):
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _reject_number(value):
    raise ValueError('qualification JSON must not contain floating numbers')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate qualification JSON member')
        result[key] = value
    return result


def _decode(data, reason):
    try:
        result = json.loads(data, object_pairs_hook=_unique_object,
                            parse_float=_reject_number,
                            parse_constant=_reject_number)
        if not isinstance(result, dict):
            raise ValueError('qualification JSON must be an object')
        return result
    except (ValueError, UnicodeError, TypeError) as exc:
        _refuse(reason, str(exc))


def _stat_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


@dataclass(frozen=True)
class _FileIdentity:
    path: Path
    stat: tuple
    sha256: str

    def unchanged(self, reason='payload_mismatch'):
        try:
            current = _stat_identity(self.path.stat())
        except OSError:
            _refuse(reason, f'installed file disappeared: {self.path.name}')
        if current != self.stat:
            _refuse(reason, f'immutable installed file changed: {self.path.name}')


def _read_file(path, reason):
    try:
        with path.open('rb') as stream:
            before = _stat_identity(os.fstat(stream.fileno()))
            data = stream.read()
            after = _stat_identity(os.fstat(stream.fileno()))
        if before != after or after != _stat_identity(path.stat()):
            _refuse(reason, f'installed file changed while reading: {path.name}')
    except OSError:
        _refuse(reason, f'cannot read installed file: {path.name}')
    return data, _FileIdentity(path, after, hashlib.sha256(data).hexdigest())


def _require_mapping(path, identity):
    """Bind Linux loader mappings, including a file replaced before registration.

    Other reviewed platforms use registration immediately following native import
    and require an immutable installation for that module's entire lifetime.
    """
    if not sys.platform.startswith('linux'):
        return
    try:
        lines = Path('/proc/self/maps').read_text(encoding='utf8').splitlines()
    except OSError:
        _refuse('payload_mismatch', 'loaded native mappings are unavailable')
    wanted = str(path)
    found = False
    for line in lines:
        fields = line.split(None, 5)
        if len(fields) != 6:
            continue
        mapped = fields[5].removesuffix(' (deleted)')
        # Linux proc escapes newlines and backslashes in filenames.
        mapped = re.sub(r'\\([0-7]{3})',
                        lambda match: chr(int(match.group(1), 8)), mapped)
        if mapped != wanted:
            continue
        major, minor = (int(part, 16) for part in fields[3].split(':'))
        if (os.makedev(major, minor), int(fields[4])) != identity.stat[:2]:
            _refuse('payload_mismatch', 'loaded native mapping differs from file')
        found = True
    if not found:
        _refuse('payload_mismatch', f'native file is not mapped: {path.name}')


class _Verifier:
    """One trusted installation and its registered native-module lifetimes."""

    def __init__(self, internal_dir: Path, anchor):
        self.internal = internal_dir.resolve()
        self.root = self.internal.parent.parent
        self.record_digest = anchor.RECORD_SHA256
        self.installation_id = anchor.INSTALLATION_ID
        self.record_filename = anchor.RECORD_FILENAME
        self.registrations = {}
        self.record = None
        self.files = []
        self.consumer_paths = None
        self.checked_modules = set()

    def _path(self, relative, reason='untrusted_qualification'):
        if not isinstance(relative, str) or '\\' in relative:
            _refuse(reason, 'invalid installation-relative path')
        parts = PurePosixPath(relative)
        if parts.is_absolute() or '..' in parts.parts or parts.as_posix() != relative:
            _refuse(reason, 'noncanonical installation-relative path')
        result = (self.root / relative).resolve()
        if not result.is_relative_to(self.root):
            _refuse(reason, 'record path escapes the trusted installation')
        return result

    def _native_path(self, module):
        if (not isinstance(module, ModuleType)
                or module.__name__ not in ('pyvoro2._core', 'pyvoro2._core2d',
                                           'pyvoro2._fpguard')
                or sys.modules.get(module.__name__) is not module):
            _refuse('payload_mismatch', 'unknown imported native module')
        spec = getattr(module, '__spec__', None)
        path = getattr(module, '__file__', None)
        loader = getattr(spec, 'loader', None)
        # scikit-build-core wraps the ordinary extension loader for editable
        # installs. Unwrap its data attribute without invoking loader callbacks.
        if (type(loader).__name__ == '_ScikitBuildLoaderWrapper'
                and type(loader).__module__.startswith('_editable_skbc_')):
            loader = vars(loader).get('_skbuild_loader')
        if (not isinstance(path, str) or spec is None
                or not isinstance(loader, importlib.machinery.ExtensionFileLoader)
                or not isinstance(spec.origin, str)):
            _refuse('payload_mismatch', 'module is not an imported native extension')
        path = Path(path).resolve()
        if Path(spec.origin).resolve() != path:
            _refuse('payload_mismatch', 'native import origin changed')
        return path

    def register_native(self, module):
        """Call immediately after import, including for unqualified builds."""
        path = self._native_path(module)
        old = self.registrations.get(module)
        if old is not None:
            if old.path != path:
                _refuse('payload_mismatch', 'registered native import path changed')
            old.unchanged()
            _require_mapping(path, old)
            return
        _, identity = _read_file(path, 'payload_mismatch')
        _require_mapping(path, identity)
        self.registrations[module] = identity

    def _check_consumers(self, record):
        consumers = record.get('consumers')
        if (not isinstance(consumers, dict) or not consumers
                or canonical_sha256(consumers) != record['consumer_sha256']):
            _refuse('source_schema_mismatch', 'consumer manifest identity differs')
        self.consumer_paths = self._consumer_inventory()
        if set(consumers) != self.consumer_paths:
            _refuse('source_schema_mismatch', 'installed consumer inventory differs')
        for name, digest in consumers.items():
            _, identity = _read_file(self._path(name), 'source_schema_mismatch')
            if not _is_digest(digest) or identity.sha256 != digest:
                _refuse('source_schema_mismatch', f'consumer source differs: {name}')
            self.files.append((identity, 'source_schema_mismatch'))

    def _consumer_inventory(self):
        return {path.relative_to(self.root).as_posix()
                for path in (self.root / 'pyvoro2').rglob('*.py')
                if path.relative_to(self.root).as_posix() != _EXCLUDED_CONSUMER}

    def _load_record(self):
        if not _is_digest(self.record_digest) or not _is_digest(self.installation_id):
            _refuse('untrusted_qualification',
                    'installation has no issued record anchor')
        if self.record_filename != 'native_qualification_record.json':
            _refuse('untrusted_qualification', 'unknown qualification record filename')
        data, identity = _read_file(self.internal / self.record_filename,
                                    'missing_qualification')
        if identity.sha256 != self.record_digest:
            _refuse('untrusted_qualification',
                    'record does not match installation anchor')
        record = _decode(data, 'untrusted_qualification')
        if (data != canonical_json(record)
                or record.get('record_schema') != RECORD_SCHEMA
                or record.get('policy_revision') != POLICY_REVISION
                or record.get('installation_id') != self.installation_id):
            _refuse('untrusted_qualification',
                    'record schema, policy, or anchor differs')
        self.files = [(identity, 'untrusted_qualification')]
        data, identity = _read_file(self.internal / 'native_approval.json',
                                    'source_schema_mismatch')
        approval = _decode(data, 'source_schema_mismatch')
        if (approval.get('approval_schema') != APPROVAL_SCHEMA
                or approval.get('policy_revision') != POLICY_REVISION
                or approval.get('approved') is not True
                or canonical_sha256(approval) != record.get('approval_sha256')):
            _refuse('source_schema_mismatch', 'source approval is absent or differs')
        self.files.append((identity, 'source_schema_mismatch'))
        for field in ('source_sha256', 'schema_sha256', 'consumer_sha256'):
            if (not _is_digest(record.get(field))
                    or record[field] != approval.get(field)):
                _refuse('source_schema_mismatch', f'approved {field} differs')
        if record['schema_sha256'] != canonical_sha256(COMPONENT_SCHEMAS):
            _refuse('source_schema_mismatch', 'consumer schema revision differs')
        build = record.get('effective_build')
        if (not isinstance(build, dict) or build.get('qualified') is not True
                or not _is_digest(build.get('manifest_sha256'))
                or not _is_digest(record.get('evidence_sha256'))):
            _refuse('effective_build_failure',
                    'completed effective-build evidence is absent')
        if record.get('target') != current_target():
            _refuse('target_abi_mismatch', 'target or Python ABI differs')
        toolchain = record.get('toolchain')
        if (not isinstance(toolchain, dict)
                or not isinstance(toolchain.get('version'), str)
                or not toolchain['version']
                or not _is_digest(toolchain.get('compiler_sha256'))
                or not _is_digest(toolchain.get('linker_sha256'))):
            _refuse('effective_build_failure', 'toolchain evidence is incomplete')
        adapter_name = build.get('adapter')
        adapter = ADAPTERS.get(adapter_name) if isinstance(adapter_name, str) else None
        target = record['target']
        if adapter is None or adapter[:3] != (
                target['platform'], target['machine'], toolchain.get('family')):
            _refuse('unsupported_target',
                    'target has no reviewed effective-build adapter')
        if (not isinstance(record.get('modules'), dict)
                or not isinstance(record.get('dependencies'), dict)
                or not isinstance(record.get('components'), dict)
                or not isinstance(approval.get('components'), dict)):
            _refuse('untrusted_qualification', 'incomplete artifact/component manifest')
        self.approval = approval
        self.allowed_components = adapter[3]
        self._check_consumers(record)
        self.record = record

    def _check_module(self, module, identity):
        record = self.record
        claimed = record['modules'].get(module.__name__)
        if not isinstance(claimed, dict):
            _refuse('payload_mismatch',
                    'record does not bind this imported native module')
        if (self._path(claimed.get('path')) != identity.path
                or claimed.get('sha256') != identity.sha256):
            _refuse('payload_mismatch',
                    'imported native module differs from qualified bytes')
        dependencies = claimed.get('dependencies')
        if (not isinstance(dependencies, list)
                or any(not isinstance(name, str) for name in dependencies)
                or len(set(dependencies)) != len(dependencies)):
            _refuse('payload_mismatch', 'native dependency manifest is incomplete')
        if module.__name__ != 'pyvoro2._fpguard':
            guard = sys.modules.get('pyvoro2._fpguard')
            guard_identity = self.registrations.get(guard)
            guard_claim = record['modules'].get('pyvoro2._fpguard')
            if guard_identity is None:
                _refuse('unregistered_payload', 'FP guard was not registered at import')
            if (not isinstance(guard_claim, dict)
                    or guard_claim.get('path') not in dependencies):
                _refuse('payload_mismatch', 'native module does not bind its FP guard')
            guard_identity.unchanged()
            _require_mapping(guard_identity.path, guard_identity)
            if guard not in self.checked_modules:
                self._check_module(guard, guard_identity)
        for name in dependencies:
            claim = record['dependencies'].get(name)
            if not isinstance(claim, dict) or not _is_digest(claim.get('sha256')):
                _refuse('payload_mismatch', 'native dependency identity is absent')
            _, dependency = _read_file(self._path(name), 'payload_mismatch')
            if dependency.sha256 != claim['sha256']:
                _refuse('payload_mismatch', f'native dependency differs: {name}')
            _require_mapping(dependency.path, dependency)
            self.files.append((dependency, 'payload_mismatch'))
        describe = getattr(module, '_qualification_identity', None)
        if not callable(describe):
            _refuse('source_schema_mismatch', 'native consistency identity is absent')
        native = describe()
        fields = ('record_schema', 'policy_revision', 'source_sha256',
                  'schema_sha256', 'consumer_sha256')
        if (not isinstance(native, dict)
                or any(native.get(key) != record[key] for key in fields)):
            _refuse('source_schema_mismatch', 'native source/schema identity differs')
        self.checked_modules.add(module)

    def require_native(self, module, component: str):
        if component not in COMPONENT_REQUIREMENTS:
            _refuse('missing_component', 'unknown native qualification component')
        expected = ('pyvoro2._core2d' if component.endswith('-planar')
                    else 'pyvoro2._core')
        if getattr(module, '__name__', None) != expected:
            _refuse('payload_mismatch', 'native module and route dimensions differ')
        identity = self.registrations.get(module)
        if identity is None:
            _refuse('unregistered_payload',
                    'native module was not registered at import')
        if self._native_path(module) != identity.path:
            _refuse('payload_mismatch', 'native import path changed after registration')
        identity.unchanged()
        _require_mapping(identity.path, identity)
        if self.record is None:
            self._load_record()
        for file_identity, reason in self.files:
            file_identity.unchanged(reason)
        if self._consumer_inventory() != self.consumer_paths:
            _refuse('source_schema_mismatch', 'installed consumer inventory changed')
        if module not in self.checked_modules:
            self._check_module(module, identity)
        for name in COMPONENT_REQUIREMENTS[component]:
            if name not in self.allowed_components:
                _refuse('missing_component',
                        f'reviewed adapter does not qualify {name}')
            claim = self.record['components'].get(name)
            approved = self.approval['components'].get(name)
            if (not isinstance(claim, dict) or claim.get('qualified') is not True
                    or not _is_digest(claim.get('evidence_sha256'))):
                _refuse('missing_component', f'completed evidence is absent for {name}')
            if (not isinstance(approved, dict)
                    or not _is_digest(claim.get('source_sha256'))
                    or claim['source_sha256'] != approved.get('source_sha256')
                    or claim.get('schema_sha256') != approved.get('schema_sha256')
                    or claim.get('schema_sha256') != canonical_sha256(
                        COMPONENT_SCHEMAS[name])):
                _refuse('source_schema_mismatch', f'approved component differs: {name}')


_VERIFIER = None
_VERIFIER_LOCK = threading.RLock()


def _verifier():
    global _VERIFIER
    with _VERIFIER_LOCK:
        if _VERIFIER is None:
            from . import _qualification_installation
            _VERIFIER = _Verifier(Path(__file__).parent, _qualification_installation)
    return _VERIFIER


def register_native(module) -> None:
    _verifier().register_native(module)


def require_native(module, component: str) -> None:
    """Require artifact compatibility; the caller also checks current FP state."""
    _verifier().require_native(module, component)
