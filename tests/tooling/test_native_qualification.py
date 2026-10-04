"""External qualification authority and immutable installed payload binding."""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
INTERNAL = ROOT / 'src/pyvoro2/_internal'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def qualification():
    return _load('isolated_native_qualification',
                 INTERNAL / 'native_qualification.py')


def _digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def installation(tmp_path, monkeypatch, qualification):
    q = qualification
    internal = tmp_path / 'pyvoro2/_internal'
    internal.mkdir(parents=True)
    consumers = {'pyvoro2/__init__.py': _digest(b'# reviewed consumer\n')}
    (tmp_path / 'pyvoro2/__init__.py').write_bytes(b'# reviewed consumer\n')
    manifest = {
        'manifest_schema': q.SOURCE_MANIFEST_SCHEMA,
        'policy_revision': q.POLICY_REVISION,
        'source_sha256': _digest(b'reviewed source'),
        'consumer_sha256': q.canonical_sha256(consumers),
        'schema_sha256': q.canonical_sha256(q.COMPONENT_SCHEMAS),
        'components': {
            name: {'source_sha256': _digest(name.encode()),
                   'schema_sha256': q.canonical_sha256(schema)}
            for name, schema in q.COMPONENT_SCHEMAS.items()
        },
    }
    module_path = tmp_path / 'pyvoro2/_core.fixture.so'
    module_path.write_bytes(b'qualified native payload')
    dependency = tmp_path / 'pyvoro2.libs/libfixture.so'
    dependency.parent.mkdir()
    dependency.write_bytes(b'qualified bundled dependency')
    module = ModuleType('pyvoro2._core')
    module.__file__ = str(module_path)
    module.__spec__ = importlib.util.spec_from_file_location(
        module.__name__, module_path,
        loader=importlib.machinery.ExtensionFileLoader(
            module.__name__, str(module_path)),
    )
    identity = {
        'record_schema': q.RECORD_SCHEMA,
        'policy_revision': q.POLICY_REVISION,
        'source_sha256': manifest['source_sha256'],
        'schema_sha256': manifest['schema_sha256'],
        'consumer_sha256': manifest['consumer_sha256'],
    }
    module._qualification_identity = lambda: identity.copy()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    guard_path = tmp_path / 'pyvoro2/_fpguard.fixture.so'
    guard_path.write_bytes(b'qualified raw FP guard')
    guard = ModuleType('pyvoro2._fpguard')
    guard.__file__ = str(guard_path)
    guard.__spec__ = importlib.util.spec_from_file_location(
        guard.__name__, guard_path,
        loader=importlib.machinery.ExtensionFileLoader(guard.__name__, str(guard_path)))
    guard._qualification_identity = lambda: identity.copy()
    monkeypatch.setitem(sys.modules, guard.__name__, guard)
    monkeypatch.setattr(q, '_require_mapping', lambda path, identity: None)
    target = q.current_target()
    monkeypatch.setattr(q, 'current_target', lambda: {
        **target, 'platform': 'linux', 'machine': 'x86_64'})
    record = {
        'record_schema': q.RECORD_SCHEMA,
        'policy_revision': q.POLICY_REVISION,
        'installation_id': _digest(b'controlled installation'),
        'source_manifest_sha256': q.canonical_sha256(manifest),
        'source_sha256': manifest['source_sha256'],
        'schema_sha256': manifest['schema_sha256'],
        'consumer_sha256': manifest['consumer_sha256'],
        'target': q.current_target(),
        'toolchain': {'family': 'GNU', 'version': '14.2.1',
                      'compiler_sha256': _digest(b'compiler'),
                      'linker_sha256': _digest(b'linker')},
        'effective_build': {'qualified': True,
                            'adapter': 'gnu-linux-x86_64-v1',
                            'manifest_sha256': _digest(b'effective commands')},
        'evidence_sha256': _digest(b'completed independent evidence'),
        'modules': {
            module.__name__: {'path': 'pyvoro2/_core.fixture.so',
                              'sha256': _digest(module_path.read_bytes()),
                              'dependencies': ['pyvoro2.libs/libfixture.so',
                                               'pyvoro2/_fpguard.fixture.so']},
            guard.__name__: {'path': 'pyvoro2/_fpguard.fixture.so',
                             'sha256': _digest(guard_path.read_bytes()),
                             'dependencies': []},
        },
        'dependencies': {'pyvoro2.libs/libfixture.so': {
            'sha256': _digest(dependency.read_bytes())},
            'pyvoro2/_fpguard.fixture.so': {
                'sha256': _digest(guard_path.read_bytes())}},
        'components': {
            name: {**component, 'qualified': True,
                   'evidence_sha256': _digest((name + ' evidence').encode())}
            for name, component in manifest['components'].items()
        },
        'consumers': consumers,
    }

    def write(*, trust=True, register=True):
        (internal / 'native_source_manifest.json').write_bytes(
            q.canonical_json(manifest))
        record['source_manifest_sha256'] = q.canonical_sha256(manifest)
        data = q.canonical_json(record)
        (internal / 'native_qualification_record.json').write_bytes(data)
        anchor = SimpleNamespace(
            RECORD_FILENAME='native_qualification_record.json',
            RECORD_SHA256=_digest(data) if trust else None,
            INSTALLATION_ID=record['installation_id'] if trust else None,
        )
        verifier = q._Verifier(internal, anchor)
        if register:
            verifier.register_native(guard)
            verifier.register_native(module)
        return verifier

    return SimpleNamespace(q=q, root=tmp_path, internal=internal,
                           manifest=manifest, record=record, module=module,
                           module_path=module_path, dependency=dependency,
                           identity=identity, write=write)


def _refuses(installation, verifier, reason, component='wp5-spatial'):
    with pytest.raises(installation.q.NativeQualificationError) as caught:
        verifier.require_native(installation.module, component)
    assert caught.value.reason == reason


def test_missing_record_and_favorable_metadata_cannot_qualify(installation):
    s = installation
    verifier = s.write()
    (s.internal / 'native_qualification_record.json').unlink()
    s.module.qualified = True
    _refuses(s, verifier, 'missing_qualification')


def test_handwritten_record_without_installation_anchor_refuses(installation):
    s = installation
    _refuses(s, s.write(trust=False), 'untrusted_qualification')


def test_detached_record_tampering_refuses(installation):
    s = installation
    verifier = s.write()
    path = s.internal / 'native_qualification_record.json'
    data = json.loads(path.read_bytes())
    data['evidence_sha256'] = _digest(b'forged evidence')
    path.write_bytes(s.q.canonical_json(data))
    _refuses(s, verifier, 'untrusted_qualification')


def test_record_copied_to_different_module_refuses(installation):
    s = installation
    s.module_path.write_bytes(b'different native payload')
    _refuses(s, s.write(), 'payload_mismatch')


@pytest.mark.parametrize('field', ['source_sha256', 'schema_sha256',
                                   'consumer_sha256'])
def test_source_or_schema_mismatch_refuses(installation, field):
    s = installation
    s.record[field] = _digest(b'unknown source or consumer contract')
    _refuses(s, s.write(), 'source_schema_mismatch')


def test_unapproved_measured_source_does_not_approve_itself(installation):
    s = installation
    s.manifest['approved'] = False
    _refuses(s, s.write(), 'source_schema_mismatch')


@pytest.mark.parametrize('field', ['evidence_sha256', 'effective_build'])
def test_missing_qualification_evidence_refuses(installation, field):
    s = installation
    del s.record[field]
    _refuses(s, s.write(), 'effective_build_failure')


def test_missing_selected_component_does_not_break_independent_wp8(installation):
    s = installation
    del s.record['components']['wp5-spatial']
    verifier = s.write()
    _refuses(s, verifier, 'missing_component', 'wp7-spatial')
    verifier.require_native(s.module, 'wp8-spatial')


@pytest.mark.parametrize('route', ['wp5-spatial', 'wp7-spatial', 'wp8-spatial'])
def test_distinct_conforming_rebuild_has_its_own_valid_record(installation, route):
    s = installation
    s.write().require_native(s.module, route)
    s.module_path.write_bytes(b'a second conforming compiled native payload')
    s.record['modules'][s.module.__name__]['sha256'] = _digest(
        s.module_path.read_bytes())
    s.record['effective_build']['manifest_sha256'] = _digest(b'rebuild commands')
    s.record['evidence_sha256'] = _digest(b'rebuild route evidence')
    s.record['installation_id'] = _digest(b'second controlled installation')
    s.write().require_native(s.module, route)


@pytest.mark.parametrize('target', ['module_path', 'dependency'])
def test_payload_mutation_after_verification_refuses(installation, target):
    s = installation
    verifier = s.write()
    verifier.require_native(s.module, 'wp5-spatial')
    getattr(s, target).write_bytes(b'changed after native import')
    _refuses(s, verifier, 'payload_mismatch')


def test_loaded_payload_cannot_be_rebound_after_path_replacement(installation):
    s = installation
    verifier = s.write()
    replacement = s.module_path.with_suffix('.replacement')
    replacement.write_bytes(s.module_path.read_bytes())
    replacement.replace(s.module_path)
    with pytest.raises(s.q.NativeQualificationError, match='payload_mismatch'):
        verifier.register_native(s.module)
    _refuses(s, verifier, 'payload_mismatch')


def test_consumer_change_invalidates_record(installation):
    s = installation
    verifier = s.write()
    (s.root / 'pyvoro2/__init__.py').write_bytes(b'# changed consumer\n')
    _refuses(s, verifier, 'source_schema_mismatch')


def test_unknown_installed_consumer_invalidates_record(installation):
    s = installation
    verifier = s.write()
    (s.root / 'pyvoro2/new_consumer.py').write_bytes(b'# unknown consumer\n')
    _refuses(s, verifier, 'source_schema_mismatch')


def test_unregistered_module_refuses(installation):
    s = installation
    _refuses(s, s.write(register=False), 'unregistered_payload')


def test_abi_mismatch_and_unreviewed_adapter_refuse(installation):
    s = installation
    s.record['target']['cache_tag'] = 'other-abi'
    _refuses(s, s.write(), 'target_abi_mismatch')
    s.record['target'] = s.q.current_target()
    s.record['effective_build']['adapter'] = 'unknown-adapter'
    _refuses(s, s.write(), 'unsupported_target')


def test_native_self_identity_is_only_consistency_data(installation):
    s = installation
    s.identity['source_sha256'] = _digest(b'other measured source')
    _refuses(s, s.write(), 'source_schema_mismatch')


def test_unqualified_editable_native_module_can_register(installation):
    s = installation
    underlying = s.module.__spec__.loader
    wrapper = type('_ScikitBuildLoaderWrapper', (),
                   {'__module__': '_editable_skbc_pyvoro2'})()
    wrapper._skbuild_loader = underlying
    s.module.__spec__.loader = wrapper
    verifier = s.write(trust=False)
    verifier.register_native(s.module)
    _refuses(s, verifier, 'untrusted_qualification')


@pytest.mark.parametrize('component, expected', [
    ('wp5-spatial', None), ('wp8-spatial', None),
    ('wp7-spatial', 'missing_component'),
])
def test_apple_adapter_keeps_independent_wp5_wp8_support(
        installation, monkeypatch, component, expected):
    s = installation
    target = {**s.q.current_target(), 'platform': 'darwin', 'machine': 'arm64'}
    monkeypatch.setattr(s.q, 'current_target', lambda: target)
    s.record['target'] = target
    s.record['toolchain']['family'] = 'AppleClang'
    s.record['effective_build']['adapter'] = 'appleclang-darwin-arm64-v1'
    verifier = s.write()
    if expected:
        _refuses(s, verifier, expected, component)
    else:
        verifier.require_native(s.module, component)


def test_planar_selected_requires_planar_ordinary_component(
        installation, monkeypatch):
    s = installation
    old_name = s.module.__name__
    s.module.__name__ = 'pyvoro2._core2d'
    s.record['modules'][s.module.__name__] = s.record['modules'].pop(old_name)
    monkeypatch.setitem(sys.modules, s.module.__name__, s.module)
    del s.record['components']['wp6-planar']
    verifier = s.write()
    _refuses(s, verifier, 'missing_component', 'wp7-planar')
    verifier.require_native(s.module, 'wp8-planar')


@pytest.mark.parametrize('which', ['native_source_manifest.json',
                                   'native_qualification_record.json'])
def test_authority_change_after_cached_verification_refuses(installation, which):
    s = installation
    verifier = s.write()
    verifier.require_native(s.module, 'wp5-spatial')
    (s.internal / which).write_bytes(b'{}\n')
    reason = ('source_schema_mismatch' if which == 'native_source_manifest.json'
              else 'untrusted_qualification')
    _refuses(s, verifier, reason)


def test_registered_path_cannot_be_redirected(installation):
    s = installation
    verifier = s.write()
    s.module.__file__ = str(s.root / 'other.so')
    _refuses(s, verifier, 'payload_mismatch')


def test_dependency_mismatch_before_first_admission_refuses(installation):
    s = installation
    verifier = s.write()
    s.dependency.write_bytes(b'wrong bundled library')
    _refuses(s, verifier, 'payload_mismatch')


def test_record_path_cannot_escape_installation(installation):
    s = installation
    s.record['modules'][s.module.__name__]['path'] = '../outside.so'
    _refuses(s, s.write(), 'untrusted_qualification')


def _file_stat(info, **changes):
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    return SimpleNamespace(**{**{key: getattr(info, key) for key in fields}, **changes})


@pytest.mark.parametrize('phase', ['read', 'cached'])
def test_immutable_file_uses_consistent_descriptor_timestamps(
        qualification, tmp_path, monkeypatch, phase):
    q = qualification
    path = tmp_path / 'immutable.pyd'
    path.write_bytes(b'immutable native payload')
    with path.open('rb') as stream:
        descriptor = q.os.fstat(stream.fileno())
    original_stat = Path.stat

    def creation_time_stat(self, *args, **kwargs):
        if self == path:
            # CPython 3.12+ Windows path stat preserves creation-time ctime,
            # whereas fstat reports the file's metadata-change time.
            return _file_stat(descriptor, st_ctime_ns=descriptor.st_ctime_ns - 100)
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', creation_time_stat)
    if phase == 'read':
        data, identity = q._read_file(path, 'payload_mismatch')
        assert data == b'immutable native payload'
    else:
        identity = q._FileIdentity(path, q._stat_identity(descriptor),
                                   _digest(b'immutable native payload'))
    assert identity.stat == q._stat_identity(descriptor)
    identity.unchanged()


@pytest.mark.parametrize('phase', ['during_read', 'after_registration'])
def test_immutable_file_change_time_alone_still_refuses(
        qualification, tmp_path, monkeypatch, phase):
    q = qualification
    path = tmp_path / 'changed.pyd'
    path.write_bytes(b'unchanged size and modification time')
    original_fstat, original_stat = q.os.fstat, Path.stat
    with path.open('rb') as stream:
        descriptor = original_fstat(stream.fileno())
    identity = q._FileIdentity(path, q._stat_identity(descriptor),
                               _digest(path.read_bytes()))
    calls = 0

    def changed_fstat(fd):
        nonlocal calls
        info = original_fstat(fd)
        if (info.st_dev, info.st_ino) == (descriptor.st_dev, descriptor.st_ino):
            calls += 1
            if phase == 'after_registration' or calls > 1:
                return _file_stat(info, st_ctime_ns=descriptor.st_ctime_ns + 100)
        return info

    def unchanged_path_stat(self, *args, **kwargs):
        # The pathname API's legacy creation clock does not see ChangeTime.
        return descriptor if self == path else original_stat(self, *args, **kwargs)

    monkeypatch.setattr(q.os, 'fstat', changed_fstat)
    monkeypatch.setattr(Path, 'stat', unchanged_path_stat)
    with pytest.raises(q.NativeQualificationError, match='payload_mismatch'):
        if phase == 'during_read':
            q._read_file(path, 'payload_mismatch')
        else:
            identity.unchanged()


def test_read_refuses_same_payload_path_replacement_while_original_is_open(
        qualification, tmp_path, monkeypatch):
    q = qualification
    path, replacement = tmp_path / 'original.pyd', tmp_path / 'replacement.pyd'
    path.write_bytes(b'identical native payload bytes')
    shutil.copy2(path, replacement)
    original_open, original_stat = Path.open, Path.stat
    replaced, reader = False, None

    class ReplaceAfterRead:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            reader.close()

        def fileno(self):
            return reader.fileno()

        def read(self):
            nonlocal replaced
            data = reader.read()
            # Simulate the pathname selecting another file without depending
            # on whether this host permits replacing an open loaded DLL.
            replaced = True
            return data

    def selected_open(self, *args, **kwargs):
        nonlocal reader
        if self != path:
            return original_open(self, *args, **kwargs)
        if replaced:
            assert not reader.closed
            return original_open(replacement, *args, **kwargs)
        reader = original_open(path, *args, **kwargs)
        return ReplaceAfterRead()

    def selected_stat(self, *args, **kwargs):
        if self == path:
            return original_stat(replacement if replaced else path, *args, **kwargs)
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', selected_open)
    monkeypatch.setattr(Path, 'stat', selected_stat)
    with pytest.raises(q.NativeQualificationError, match='payload_mismatch'):
        q._read_file(path, 'payload_mismatch')


def test_actual_linux_loader_mapping_is_checked(qualification):
    if not sys.platform.startswith('linux'):
        pytest.skip('Linux mapping adapter')
    import numpy.linalg._umath_linalg as extension
    q = qualification
    path = Path(extension.__file__).resolve()
    _, identity = q._read_file(path, 'payload_mismatch')
    q._require_mapping(path, identity)
    wrong_stat = (identity.stat[0], identity.stat[1] + 1, *identity.stat[2:])
    wrong = q._FileIdentity(path, wrong_stat, identity.sha256)
    with pytest.raises(q.NativeQualificationError, match='payload_mismatch'):
        q._require_mapping(path, wrong)


def test_default_installation_anchor_is_unqualified():
    anchor = _load('unqualified_installation',
                   INTERNAL / '_qualification_installation.py')
    assert anchor.RECORD_SHA256 is None
    assert anchor.INSTALLATION_ID is None


def test_source_measurement_covers_vendor_build_consumers_and_excludes_anchors(
        tmp_path):
    policy = _load('isolated_source_policy',
                   ROOT / 'tools/native/qualification/source_policy.py')
    for name in ('vendor/voro++/src/unused.hh',
                 'vendor/voro++/2d/src/cell_2d.cc', 'cpp/bindings.cpp',
                 'cmake/NativeFP.cmake', 'CMakeLists.txt', 'pyproject.toml',
                 'src/pyvoro2/__init__.py',
                 'src/pyvoro2/_internal/native_qualification.py',
                 'src/pyvoro2/_internal/native_source_manifest.json',
                 'src/pyvoro2/_internal/_qualification_installation.py',
                 'tools/native/qualification/source_policy.py'):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('native_qualification.py'):
            shutil.copyfile(INTERNAL / 'native_qualification.py', destination)
        else:
            destination.write_text('# fixture\n', encoding='utf8')
    measured = policy.measure_source(tmp_path)
    assert 'vendor/voro++/src/unused.hh' in measured['files']
    assert 'cpp/bindings.cpp' in measured['files']
    assert 'cmake/NativeFP.cmake' in measured['files']
    assert 'src/pyvoro2/__init__.py' in measured['files']
    assert all('native_source_manifest.json' not in p for p in measured['files'])
    assert all('_qualification_installation.py' not in p
               for p in measured['files'])
    anchor = tmp_path / 'src/pyvoro2/_internal/native_source_manifest.json'
    anchor.write_text('arbitrary new manifest text', encoding='utf8')
    assert policy.measure_source(tmp_path)['source_sha256'] == measured['source_sha256']
    added = tmp_path / 'vendor/voro++/src/new_dependency.hh'
    added.write_text('unknown relevant dependency', encoding='utf8')
    changed = policy.measure_source(tmp_path)
    assert changed['source_sha256'] != measured['source_sha256']
    assert changed['components']['wp5-spatial']['source_sha256'] != (
        measured['components']['wp5-spatial']['source_sha256'])


def test_measured_source_never_updates_manifest():
    policy = _load('isolated_source_policy_check',
                   ROOT / 'tools/native/qualification/source_policy.py')
    path = INTERNAL / 'native_source_manifest.json'
    before = path.read_bytes()
    measured = policy.measure_source(ROOT)
    assert path.read_bytes() == before
    draft = {'manifest_schema': 'pyvoro2-native-source-manifest-v1',
             **{field: measured[field] for field in (
                 'policy_revision', 'source_sha256', 'schema_sha256',
                 'consumer_sha256', 'components')}, 'approved': True}
    with pytest.raises(ValueError, match='unknown or missing keys'):
        qualification = policy._contract(ROOT)
        qualification.validate_source_manifest(draft)
