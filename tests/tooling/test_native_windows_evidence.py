"""Windows link evidence must bind actual inputs and manifest resource edits."""
from __future__ import annotations

from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'native'))

from qualification.effective_build import (  # noqa: E402
    BuildEvidenceError, file_identity, verify_build,
)
from qualification.link_provenance import (  # noqa: E402
    verify_link_closure, verify_linker_options,
)
from qualification import windows_link  # noqa: E402


MANIFEST = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3"><security>
    <requestedPrivileges>
      <requestedExecutionLevel level="asInvoker" uiAccess="false" />
    </requestedPrivileges>
  </security></trustInfo>
</assembly>'''


def _archive(path, members):
    """Small COFF archives, with duplicate import member names on purpose."""
    symbols = b''.join(symbol.encode() + b'\0' for _, symbol, _ in members)
    index_size = 4 + 4 * len(members) + len(symbols)
    cursor = 8 + 60 + index_size + index_size % 2
    positions = []
    for _, _, body in members:
        positions.append(cursor)
        cursor += 60 + len(body) + len(body) % 2

    def entry(name, body):
        fields = (name.ljust(16) + '0'.ljust(12) + '0'.ljust(6) +
                  '0'.ljust(6) + '0'.ljust(8) + str(len(body)).ljust(10) + '`\n')
        return fields.encode() + body + (b'\n' if len(body) % 2 else b'')

    index = struct.pack('>I', len(members))
    index += b''.join(struct.pack('>I', offset) for offset in positions) + symbols
    path.write_bytes(b'!<arch>\n' + entry('/', index) +
                     b''.join(entry(name + '/', body) for name, _, body in members))
    return positions


def _library_report(tmp_path):
    obj = tmp_path / 'unit.obj'
    obj.write_bytes(b'\x64\x86' + b'\0' * 62)
    archive = tmp_path / 'MSVCRT.lib'
    positions = _archive(archive, [('runtime.dll', '__imp_first', b'first import'),
                                   ('runtime.dll', '__imp_second', b'second import')])
    rsp = tmp_path / 'actual-link-inputs.rsp'
    rsp.write_text('"' + str(obj) + '"\n')  # DEFAULTLIB archive is absent here.
    verbose = tmp_path / 'stdout.txt'
    verbose.write_text('Searching libraries\n'
                       f'    Searching {archive}:\n'
                       '      Found __imp_second\n'
                       '        Referenced in unit.obj\n'
                       '        Loaded MSVCRT.lib(runtime.dll)\n'
                       'Finished searching libraries\n')
    return obj, archive, positions, rsp, verbose


def _pe_image(path, *, dll=True, manifest=None, resource_id=2, code=b'\xc3\x90'):
    """Literal PE32+ fixture with .text and an optional ID/language resource tree."""
    data = bytearray(1536 if manifest is not None else 1024)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 60, 128)
    data[128:132] = b'PE\0\0'
    struct.pack_into('<HH', data, 132, 0x8664, 2 if manifest is not None else 1)
    struct.pack_into('<HH', data, 148, 240, 0x2022 if dll else 0x22)
    struct.pack_into('<H', data, 152, 0x20b)
    struct.pack_into('<I', data, 260, 16)
    data[392:400] = b'.text\0\0\0'
    struct.pack_into('<IIII', data, 400, len(code), 4096, 512, 512)
    struct.pack_into('<I', data, 428, 0x60000020)
    data[512:512 + len(code)] = code
    if manifest is not None:
        struct.pack_into('<II', data, 280, 8192, 88 + len(manifest))
        data[432:440] = b'.rsrc\0\0\0'
        struct.pack_into('<IIII', data, 440, 88 + len(manifest), 8192, 512, 1024)
        struct.pack_into('<I', data, 468, 0x40000040)
        for offset in (0, 24, 48):
            struct.pack_into('<HH', data, 1024 + offset + 12, 0, 1)
        struct.pack_into('<II', data, 1040, 24, 0x80000018)
        struct.pack_into('<II', data, 1064, resource_id, 0x80000030)
        struct.pack_into('<II', data, 1088, 1033, 72)
        struct.pack_into('<IIII', data, 1096, 8192 + 88, len(manifest), 0, 0)
        data[1112:1112 + len(manifest)] = manifest
    path.write_bytes(data)


def _manifest_row(tmp_path, monkeypatch, *, dll=True):
    directory = tmp_path / 'manifest'
    directory.mkdir()
    donor = directory / 'donor.pe'
    output = directory / ('image.pyd' if dll else 'image.exe')
    _pe_image(donor, dll=dll)
    _pe_image(output, dll=dll, manifest=MANIFEST, resource_id=2 if dll else 1)
    sidecar, embedded = directory / 'link.manifest', directory / 'embedded.manifest'
    sidecar.write_bytes(MANIFEST)
    embedded.write_bytes(MANIFEST)
    tool, driver, image = (directory / name
                           for name in ('mt.exe', 'link.exe', 'ntdll.dll'))
    for path in (tool, driver, image):
        path.write_bytes(b'MZ' + path.name.encode())
    monkeypatch.setattr(windows_link, 'manifest_tool',
                        lambda _: (tool, {'sdk': 'fixture'}))
    environment = {'VSLANG': '1033'}
    argv = [str(tool), '-nologo', '-manifest', str(sidecar),
            '-outputresource:' + str(output) + ';#' + ('2' if dll else '1')]
    invocation = {'pid': 42, 'executable': file_identity(tool), 'argv': argv,
                  'expanded_argv': argv, 'cwd': str(tmp_path),
                  'environment': environment,
                  'role': 'manifest_tool', 'exit_code': 0,
                  'observation': 'direct-debug-execution',
                  'loaded_images': [file_identity(image)]}
    logs = directory / 'manifest-tool'
    logs.mkdir()
    events = [{'event': 3, 'pid': 42, 'tid': 43, 'image': file_identity(tool)},
              {'event': 6, 'pid': 42, 'tid': 43, 'image': file_identity(image)},
              {'event': 5, 'pid': 42, 'tid': 43, 'exit_code': 0}]
    events_path = logs / 'windows-debug-events.json'
    events_path.write_text(json.dumps(events))
    for name in ('stdout.txt', 'stderr.txt'):
        (logs / name).write_text('')
    donor_id = file_identity(donor)
    manifest = {'schema': 'pyvoro2.windows-manifest.v1', 'donor': donor_id,
                'linked_output': {**donor_id, 'path': str(output)},
                'sidecar': file_identity(sidecar), 'output': file_identity(output),
                'resource': {'resource_id': 2 if dll else 1,
                             'language': 1033, 'codepage': 0},
                'embedded': file_identity(embedded),
                'xml_sha256': hashlib.sha256(
                    windows_link.manifest_xml(MANIFEST)).hexdigest(),
                'tool': file_identity(tool), 'installation': {'sdk': 'fixture'},
                'invocations': [invocation], 'loaded_images': [file_identity(image)],
                'evidence_files': [file_identity(path)
                                   for path in sorted(logs.iterdir())]}
    link_argv = [str(driver), '/MANIFEST', '/MANIFESTFILE:' + str(sidecar),
                 '/OUT:' + str(output), *(['/DLL'] if dll else [])]
    row = {'family': 'msvc', 'windows_manifest': manifest,
           'output': file_identity(output),
           'cwd': str(tmp_path), 'environment': environment,
           'invocations': [{'role': 'linker', 'executable': file_identity(driver),
                            'expanded_argv': link_argv}, invocation],
           'tools': [file_identity(path) for path in (tool, driver, image)],
           'evidence_files': [*manifest['evidence_files'], donor_id,
                              file_identity(sidecar), file_identity(embedded)]}
    return row, events_path


@pytest.mark.parametrize('option', ['/MANIFEST:EMBED', '/MANIFEST:NO'])
def test_windows_linker_requires_observed_manifest_resource_edit(option):
    with pytest.raises(BuildEvidenceError, match='manifest'):
        verify_linker_options(['link.exe', '/DLL', option], 'msvc')


@pytest.mark.parametrize('variable', ['LINK', '_LINK_'])
@pytest.mark.parametrize('option', [
    '/MANIFEST:EMBED', '-MANIFEST:EMBED', '/MANIFEST:NO',
    '/MANIFESTFILE:unobserved.manifest', '/LINKREPROFULLPATHRSP:fake.rsp',
    '/VERBOSE:UNUSEDLIBS',
])
def test_windows_environment_cannot_replace_manifest_or_input_observer(
        tmp_path, variable, option):
    response = tmp_path / 'override.rsp'
    response.write_text(option)
    with pytest.raises(BuildEvidenceError, match='observer override'):
        windows_link.prepare_link(['link.exe', '/DLL'], cwd=tmp_path,
                                  env={variable: '@' + str(response)},
                                  directory=tmp_path)


def test_windows_explicit_inputs_do_not_prove_default_library_selection(tmp_path):
    obj = tmp_path / 'unit.obj'
    obj.write_bytes(b'\x64\x86' + b'\0' * 62)
    identity = file_identity(obj)
    row = {'family': 'msvc', 'link_inputs': [identity],
           'object_inputs': [identity], 'runtime_link_inputs': []}
    with pytest.raises(BuildEvidenceError, match='Windows.*input'):
        verify_link_closure(row, {identity['path']: {'output': identity}})


def test_windows_actual_defaultlib_selection_resolves_duplicate_import_members(
        tmp_path):
    obj, archive, offsets, rsp, verbose = _library_report(tmp_path)
    report = windows_link.collect_link_inputs(rsp, verbose)
    assert report['explicit_inputs'] == [file_identity(obj)]
    assert report['searched_libraries'] == [file_identity(archive)]
    assert report['loaded_members'] == [{
        'archive': str(archive), 'symbol': '__imp_second',
        'reported_member': 'runtime.dll', 'referenced_by': ['unit.obj'],
        'member': {'name': 'runtime.dll', 'offset': offsets[1], 'size': 13,
                   'sha256': hashlib.sha256(b'second import').hexdigest()},
    }]


@pytest.mark.parametrize('before,after', [
    ('Finished searching libraries', ''),
    ('Loaded MSVCRT.lib(runtime.dll)', 'Loaded shadow.lib(runtime.dll)'),
    ('Loaded MSVCRT.lib(runtime.dll)', 'Loaded MSVCRT.lib(missing.obj)'),
    ('Found __imp_second', 'Found __imp_unrecorded'),
    ('Referenced in unit.obj', 'Unreviewed library selection'),
])
def test_windows_library_report_cannot_guess_missing_paths_or_members(
        tmp_path, before, after):
    _, _, _, rsp, verbose = _library_report(tmp_path)
    verbose.write_text(verbose.read_text().replace(before, after))
    with pytest.raises(BuildEvidenceError, match='Windows'):
        windows_link.collect_link_inputs(rsp, verbose)


@pytest.mark.parametrize('dll', [True, False])
def test_windows_manifest_binds_actual_metadata_and_dll_or_exe_resource_id(
        tmp_path, monkeypatch, dll):
    row, _ = _manifest_row(tmp_path, monkeypatch, dll=dll)
    windows_link.verify_manifest(row)
    resources = windows_link.pe_manifest_resources(row['output']['path'])
    assert [(r['resource_id'], r['language'], r['bytes']) for r in resources] == [
        (2 if dll else 1, 1033, MANIFEST)]


@pytest.mark.parametrize('mutation', [
    'resource_id', 'metadata', 'code', 'argv', 'donor',
])
def test_windows_manifest_refuses_changed_resource_or_unbound_pe_edit(
        tmp_path, monkeypatch, mutation):
    row, _ = _manifest_row(tmp_path, monkeypatch)
    receipt = row['windows_manifest']
    if mutation in ('resource_id', 'metadata', 'code'):
        output = Path(row['output']['path'])
        manifest = MANIFEST.replace(b'asInvoker', b'requireAdministrator') if (
            mutation == 'metadata') else MANIFEST
        _pe_image(output, manifest=manifest,
                  resource_id=1 if mutation == 'resource_id' else 2,
                  code=b'\xcc\x90' if mutation == 'code' else b'\xc3\x90')
        row['output'] = receipt['output'] = file_identity(output)
    elif mutation == 'argv':
        receipt['invocations'][0]['argv'].append('-hashupdate:unobserved')
    else:
        receipt['linked_output']['sha256'] = '0' * 64
    with pytest.raises(BuildEvidenceError, match='Windows manifest'):
        windows_link.verify_manifest(row)


def test_windows_manifest_cannot_accept_unobserved_child_image(tmp_path, monkeypatch):
    row, events_path = _manifest_row(tmp_path, monkeypatch)
    events = json.loads(events_path.read_text())
    child = tmp_path / 'rc.exe'
    child.write_bytes(b'MZ unobserved child')
    events.insert(1, {'event': 3, 'pid': 44, 'tid': 45, 'image': file_identity(child)})
    events_path.write_text(json.dumps(events))
    for container in (row['evidence_files'], row['windows_manifest']['evidence_files']):
        for index, item in enumerate(container):
            if item['path'] == str(events_path):
                container[index] = file_identity(events_path)
    with pytest.raises(BuildEvidenceError, match='Windows manifest.*(child|process)'):
        windows_link.verify_manifest(row)


def test_windows_manifest_cannot_detach_mapped_image_or_process_evidence(
        tmp_path, monkeypatch):
    row, _ = _manifest_row(tmp_path, monkeypatch)
    row['windows_manifest']['loaded_images'] = []
    with pytest.raises(BuildEvidenceError, match='Windows manifest.*image'):
        windows_link.verify_manifest(row)


def test_windows_link_closure_rejects_shadowed_default_library(tmp_path, monkeypatch):
    row, _ = _manifest_row(tmp_path, monkeypatch)
    obj, archive, _, rsp, verbose = _library_report(tmp_path)
    report = windows_link.collect_link_inputs(rsp, verbose)
    row.update(windows_link_inputs=report,
               link_inputs=[file_identity(obj), file_identity(archive)],
               object_inputs=[file_identity(obj)], runtime_link_inputs=[])
    row['invocations'][0]['expanded_argv'].extend([
        '/VERBOSE:LIB', '/LINKREPROFULLPATHRSP:' + str(rsp)])
    row['evidence_files'].extend([file_identity(rsp), file_identity(verbose)])
    with pytest.raises(BuildEvidenceError, match='unapproved native archive'):
        verify_link_closure(row, {str(obj): {'output': file_identity(obj)}})
    row['runtime_link_inputs'] = [file_identity(archive)]
    assert verify_link_closure(row, {str(obj): {'output': file_identity(obj)}}) == [
        file_identity(obj)]
    incomplete = copy.deepcopy(row)
    incomplete['link_inputs'] = [file_identity(obj)]
    with pytest.raises(BuildEvidenceError, match='Windows actual link input inventory'):
        verify_link_closure(incomplete, {str(obj): {'output': file_identity(obj)}})
    detached = copy.deepcopy(row)
    detached['invocations'][0]['expanded_argv'].remove('/VERBOSE:LIB')
    with pytest.raises(BuildEvidenceError, match='Windows.*selection.*invocation'):
        verify_link_closure(detached, {str(obj): {'output': file_identity(obj)}})


@pytest.mark.skipif(os.name != 'nt', reason='actual MSVC/Win32 debugger evidence')
@pytest.mark.parametrize('dll', [True, False])
def test_actual_windows_link_observes_default_libraries_and_manifest_resource(
        tmp_path, dll):
    compiler, linker = shutil.which('cl.exe'), shutil.which('link.exe')
    if not compiler or not linker:
        pytest.skip('MSVC developer environment required')
    source = tmp_path / 'observed.cpp'
    source.write_text('extern "C" __declspec(dllexport) double observed(double x) '
                      '{ return x + 1.0; }\n' +
                      ('' if dll else 'int main() { return observed(1.0) != 2.0; }\n'))
    obj = tmp_path / 'observed.obj'
    output = tmp_path / ('_observed.pyd' if dll else '_observed.exe')
    records = tmp_path / 'records'
    recorder = ROOT / 'tools/native/qualification/record_command.py'
    commands = [
        [compiler, '/nologo', '/O2', '/MD', '/fp:strict', '/GL-', '/c', str(source),
         '/Fo' + str(obj)],
        [linker, '/DLL' if dll else '/SUBSYSTEM:CONSOLE', '/LTCG:OFF', '/MANIFEST',
         "/MANIFESTUAC:level='asInvoker' uiAccess='false'", '/OUT:' + str(output),
         str(obj), 'kernel32.lib'],
    ]
    for command in commands:
        run = subprocess.run([sys.executable, str(recorder), '--output-dir',
                              str(records), '--', *command], cwd=tmp_path,
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
    build = verify_build(records, tmp_path)
    assert build['components']['_observed']['output'] == file_identity(output)
    row = next(json.loads(path.read_text()) for path in records.glob('*.json')
               if json.loads(path.read_text())['kind'] == 'link')
    assert any(Path(item['path']).name.lower() == 'msvcrt.lib'
               for item in row['windows_link_inputs']['searched_libraries'])
    assert row['windows_manifest']['resource']['resource_id'] == (2 if dll else 1)
    assert [item['role'] for item in row['invocations']] == ['linker', 'manifest_tool']
