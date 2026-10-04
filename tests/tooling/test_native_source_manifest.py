"""Independent source identities and the explicit developer refresh command."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'tools/native/qualification/source_policy.py'
MANIFEST = 'src/pyvoro2/_internal/native_source_manifest.json'
SCHEMAS = {
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


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def cli(source, *args):
    return subprocess.run([sys.executable, str(SCRIPT), '--root', str(source), *args],
                          capture_output=True, text=True, check=False)


@pytest.fixture
def source(tmp_path):
    for name in ('vendor/voro++/src', 'vendor/voro++/2d', 'cpp', 'cmake',
                 'src/pyvoro2/_internal', 'tools', 'tests', '.github/workflows'):
        (tmp_path / name).mkdir(parents=True)
    for name in ('CMakeLists.txt', 'pyproject.toml', 'cpp/fixture.cpp',
                 'vendor/voro++/src/fixture.cc', 'vendor/voro++/2d/fixture.cc',
                 'src/pyvoro2/__init__.py', 'tests/fixture.py'):
        (tmp_path / name).write_bytes(b'# independent fixture\n')
    shutil.copyfile(ROOT / 'src/pyvoro2/_internal/native_qualification.py',
                    tmp_path / 'src/pyvoro2/_internal/native_qualification.py')
    return tmp_path


def expected(source):
    # Independently enumerate this fixture; do not round-trip production hashes.
    files = {p.relative_to(source).as_posix(): digest(p.read_bytes())
             for p in source.rglob('*') if p.is_file()
             and p.relative_to(source).as_posix() != MANIFEST}
    consumers = {p.removeprefix('src/'): h for p, h in files.items()
                 if p.startswith('src/pyvoro2/') and p.endswith('.py')}
    return {'manifest_schema': 'pyvoro2-native-source-manifest-v1',
            'policy_revision': 'issue88-p2',
            'source_sha256': digest(canonical(files)),
            'consumer_sha256': digest(canonical(consumers)),
            'schema_sha256': digest(canonical(SCHEMAS)),
            'components': {
                name: {'source_sha256': digest(canonical({
                    'component': name,
                    'files': {p: h for p, h in files.items() if not p.startswith(
                        'vendor/voro++/src/' if name.endswith('-planar')
                        else 'vendor/voro++/2d/')}})),
                       'schema_sha256': digest(canonical(schema))}
                for name, schema in SCHEMAS.items()}}


def test_update_creates_exact_mechanical_manifest_and_is_noop(source):
    before = expected(source)
    result = cli(source, '--update-manifest')
    assert result.returncode == 0, result.stderr
    assert 'created' in result.stdout
    path = source / MANIFEST
    assert path.read_bytes() == canonical(before)
    identity = path.stat().st_mtime_ns
    result = cli(source, '--update-manifest')
    assert result.returncode == 0, result.stderr
    assert 'unchanged' in result.stdout
    assert path.stat().st_mtime_ns == identity
    checked = cli(source, '--check-manifest')
    assert checked.returncode == 0, checked.stderr
    assert digest(path.read_bytes()) in checked.stdout


def test_measure_is_readonly_and_does_not_need_a_manifest(source):
    for data in (None, b'{invalid'):
        path = source / MANIFEST
        if data is not None:
            path.write_bytes(data)
        default, explicit = cli(source), cli(source, '--measure')
        assert default.returncode == explicit.returncode == 0
        assert default.stdout == explicit.stdout
        measurement = json.loads(default.stdout)
        assert measurement['measurement_schema'] == (
            'pyvoro2-native-source-measurement-v1')
        assert measurement['source_sha256'] == expected(source)['source_sha256']
        assert path.read_bytes() == data if data else not path.exists()


def test_missing_check_is_actionable_and_does_not_write(source):
    result = cli(source, '--check-manifest')
    assert result.returncode == 1
    assert 'missing' in result.stderr and '--update-manifest' in result.stderr
    assert not (source / MANIFEST).exists()


@pytest.mark.parametrize('args', [('--check-approval',),
                                  ('--measure', '--update-manifest'),
                                  ('--check-manifest', '--update-manifest'),
                                  ('--output', 'arbitrary.json')])
def test_cli_rejects_retired_or_ambiguous_actions(source, args):
    result = cli(source, *args)
    assert result.returncode == 2
    assert not (source / MANIFEST).exists()


def test_stale_check_reports_sorted_fields_and_refreshes(source):
    (source / MANIFEST).write_bytes(canonical(expected(source)))
    (source / 'cpp/fixture.cpp').write_bytes(b'changed measured input\n')
    result = cli(source, '--check-manifest')
    assert result.returncode == 1 and 'stale' in result.stderr
    fields = [line.split(':', 1)[0].strip() for line in result.stderr.splitlines()
              if 'recorded=' in line]
    assert fields == sorted(fields)
    assert 'source_sha256' in fields
    before = (source / MANIFEST).read_bytes()
    assert before != canonical(expected(source))
    result = cli(source, '--update-manifest')
    assert result.returncode == 0 and 'updated' in result.stdout
    assert (source / MANIFEST).read_bytes() == canonical(expected(source))


@pytest.mark.parametrize('data, state', [
    (b'{malformed', 'replaced-invalid'),
    (b'{"manifest_schema":"x","manifest_schema":"y"}', 'replaced-invalid'),
    (b'{"unknown":true}', 'replaced-invalid'),
    (b'null', 'replaced-invalid'),
])
def test_invalid_check_refuses_and_explicit_update_replaces(source, data, state):
    path = source / MANIFEST
    path.write_bytes(data)
    checked = cli(source, '--check-manifest')
    assert checked.returncode == 1 and 'invalid' in checked.stderr
    assert path.read_bytes() == data
    updated = cli(source, '--update-manifest')
    assert updated.returncode == 0 and state in updated.stdout
    assert digest(data) in updated.stdout and 'validation' in updated.stdout
    assert path.read_bytes() == canonical(expected(source))


def test_explicit_update_normalizes_only_the_manifest(source):
    value = expected(source)
    path = source / MANIFEST
    path.write_text(json.dumps(value, indent=2))
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file() and p != path}
    result = cli(source, '--check-manifest')
    assert result.returncode == 1 and 'noncanonical' in result.stderr
    result = cli(source, '--update-manifest')
    assert result.returncode == 0 and 'normalized' in result.stdout
    assert path.read_bytes() == canonical(value)
    after = {p: p.read_bytes() for p in source.rglob('*')
             if p.is_file() and p != path}
    assert after == before


def test_retired_file_is_measured_and_forbidden_without_parsing(source):
    retired = source / 'src/pyvoro2/_internal/native_approval.json'
    retired.write_bytes(b'not old schema JSON')
    measured = cli(source, '--measure')
    assert measured.returncode == 0
    files = json.loads(measured.stdout)['files']
    assert retired.relative_to(source).as_posix() in files
    for action in ('--check-manifest', '--update-manifest'):
        result = cli(source, action)
        assert result.returncode == 1 and 'forbidden' in result.stderr
    assert not (source / MANIFEST).exists()


@pytest.mark.parametrize('unsafe', ['missing-root', 'input-symlink',
                                    'destination-symlink', 'destination-directory',
                                    'escaping-parent'])
def test_failed_update_keeps_existing_bytes(source, tmp_path, unsafe):
    path = source / MANIFEST
    path.write_bytes(b'previous bytes')
    sentinel = source.parent / (source.name + '-sentinel')
    sentinel.write_bytes(b'outside bytes')
    if unsafe == 'missing-root':
        shutil.rmtree(source / 'cpp')
    elif unsafe == 'input-symlink':
        (source / 'cpp/fixture.cpp').unlink()
        (source / 'cpp/fixture.cpp').symlink_to(sentinel)
    elif unsafe == 'destination-symlink':
        path.unlink()
        path.symlink_to(sentinel)
    elif unsafe == 'destination-directory':
        path.unlink()
        path.mkdir()
    else:
        external = source.parent / (source.name + '-external')
        shutil.move(path.parent, external)
        path.parent.symlink_to(external, target_is_directory=True)
    result = cli(source, '--update-manifest')
    assert result.returncode == 1, result.stdout + result.stderr
    assert sentinel.read_bytes() == b'outside bytes'
    if path.is_file() and unsafe != 'destination-symlink':
        assert path.read_bytes() == b'previous bytes'
    assert not list(path.parent.glob('.*.tmp'))


@pytest.mark.parametrize('field,value', [
    ('source_sha256', 'A' * 64), ('source_sha256', 'a' * 63),
    ('source_sha256', 5), ('consumer_sha256', True),
    ('schema_sha256', '0' * 64), ('manifest_schema', 'obsolete'),
    ('policy_revision', 'issue88-p1'), ('components', []),
    ('components', {}), ('reviewer', 'not an identity'),
])
def test_strict_contract_rejects_invalid_values_and_reports_differences(
        source, field, value):
    manifest = expected(source)
    manifest[field] = value
    (source / MANIFEST).write_bytes(canonical(manifest))
    result = cli(source, '--check-manifest')
    assert result.returncode == 1 and 'invalid' in result.stderr
    differences = [line for line in result.stderr.splitlines() if 'recorded=' in line]
    assert differences == sorted(differences)
    assert any(line.startswith((field + ':', field + '.')) for line in differences)


@pytest.mark.parametrize('data', [
    b'{"source_sha256":1.5}', b'{"source_sha256":NaN}',
    b'{"source_sha256":Infinity}', b'{"source_sha256":-Infinity}',
    b'{"components":{"wp5-spatial":{},"wp5-spatial":{}}}',
])
def test_strict_parser_refuses_floats_constants_and_nested_duplicates(source, data):
    path = source / MANIFEST
    path.write_bytes(data)
    result = cli(source, '--check-manifest')
    assert result.returncode == 1 and 'invalid' in result.stderr
    assert path.read_bytes() == data


@pytest.mark.parametrize('change', ['edit', 'add', 'delete'])
def test_covered_changes_invalidate_each_owning_component(source, change):
    before = expected(source)
    (source / MANIFEST).write_bytes(canonical(before))
    path = source / 'vendor/voro++/2d/fixture.cc'
    if change == 'edit':
        path.write_bytes(b'changed planar source')
    elif change == 'add':
        (path.parent / 'unknown.hh').write_bytes(b'new relevant file')
    else:
        path.unlink()
    result = cli(source, '--check-manifest')
    assert result.returncode == 1 and 'stale' in result.stderr
    after = json.loads(cli(source, '--measure').stdout)
    for name in SCHEMAS:
        assert (after['components'][name] != before['components'][name]) == (
            name.endswith('-planar'))


def test_generated_outputs_do_not_enter_identities(source):
    before = json.loads(cli(source, '--measure').stdout)
    for name in ('src/pyvoro2/_internal/_qualification_installation.py',
                 'src/pyvoro2/_internal/native_qualification_record.json',
                 'cpp/fixture.o', 'src/pyvoro2/_core.fixture.so'):
        (source / name).write_bytes(b'generated output')
    after = json.loads(cli(source, '--measure').stdout)
    assert before == after


def test_refresh_preserves_mode_and_cleans_temporary_on_replace_failure(
        source, monkeypatch):
    spec = importlib.util.spec_from_file_location('manifest_policy_atomic', SCRIPT)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    path = source / MANIFEST
    path.write_bytes(b'previous bytes')
    path.chmod(0o600)

    def refuse_replace(*args):
        raise OSError('fixture atomic publication failure')

    with monkeypatch.context() as context:
        context.setattr(policy.os, 'replace', refuse_replace)
        with pytest.raises(OSError, match='atomic publication'):
            policy.update_manifest(source)
    assert path.read_bytes() == b'previous bytes'
    assert not list(path.parent.glob('.*.tmp'))
    policy.update_manifest(source)
    assert path.stat().st_mode & 0o777 == 0o600


def test_missing_required_directory_is_not_repaired(source):
    shutil.rmtree(source / 'src/pyvoro2/_internal')
    result = cli(source, '--update-manifest')
    assert result.returncode == 1
    assert not (source / 'src/pyvoro2/_internal').exists()


def test_refresh_refuses_unreadable_directory_without_rewriting(source, monkeypatch):
    restricted = source / 'cpp/restricted'
    restricted.mkdir()
    (restricted / 'hidden.cpp').write_bytes(b'measured input\n')
    assert cli(source, '--update-manifest').returncode == 0
    path = source / MANIFEST
    before = path.read_bytes(), path.stat().st_mtime_ns
    spec = importlib.util.spec_from_file_location('manifest_unreadable', SCRIPT)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    scandir = policy.os.scandir

    def unreadable(directory):
        if Path(directory) == restricted:
            raise PermissionError('fixture unreadable source directory')
        return scandir(directory)

    monkeypatch.setattr(policy.os, 'scandir', unreadable)
    with pytest.raises(PermissionError, match='unreadable source directory'):
        policy.update_manifest(source)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert not list(path.parent.glob('.*.tmp'))
