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
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'native'))

from qualification.effective_build import (  # noqa: E402
    BuildEvidenceError, file_identity, verify_build,
)
from qualification.link_provenance import (  # noqa: E402
    verify_link_closure, verify_linker_options,
)
from qualification.discriminators import _primitive_options  # noqa: E402
from qualification.record_command import query_options, sources_in  # noqa: E402
from qualification import windows_link, windows_trace  # noqa: E402


MANIFEST = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3"><security>
    <requestedPrivileges>
      <requestedExecutionLevel level="asInvoker" uiAccess="false" />
    </requestedPrivileges>
  </security></trustInfo>
</assembly>'''


@pytest.mark.parametrize('family', ['gnu', 'clang', 'msvc'])
def test_query_options_interprets_shared_switches_by_compiler_family(
        tmp_path, family):
    source = tmp_path / 'unit.cpp'
    arguments = ['compiler', '-MD', '-MT', '-DKEEP=1', '-MP', '-c', str(source)]
    # GNU -MT consumes a dependency target; MSVC -MT selects the static CRT.
    expected = ['-MD', '-MT', '-DKEEP=1', '-MP'] if family == 'msvc' else []
    assert query_options(arguments, source, family) == expected


@pytest.mark.parametrize('runtime', [
    '/MD', '-MD', '/MDd', '-MDd', '/MT', '-MT', '/MTd', '-MTd',
])
@pytest.mark.parametrize('prefix', ['/', '-'])
def test_msvc_captured_semantic_options_reach_discriminator_harness(
        tmp_path, runtime, prefix):
    source = tmp_path / 'v_compute.cc'
    # Model the effective production command retained by the Windows adapter,
    # including its injected observation options and CMake's dash-prefixed CRT.
    semantic = ['/nologo', '/TP', '-DPy_NO_LINK_LIB', '-Ivendor/voro++/src',
                '/EHsc', '/O2', '-std:c++17', runtime, '/fp:strict', '/GL-',
                '/FIcpp/native_fp_contract.hpp', '/favor:AMD64',
                '-favor:INTEL64', '/fastfail', '/FS']
    arguments = ['cl.exe', *semantic, prefix + 'c', str(source),
                 prefix + 'Fo' + str(tmp_path / 'v_compute.obj'),
                 prefix + 'Fd' + str(tmp_path / 'compile.pdb'),
                 prefix + 'sourceDependencies', str(tmp_path / 'dependencies.json'),
                 prefix + 'Bv', prefix + 'FAs',
                 prefix + 'Fa' + str(tmp_path / 'native.asm')]
    unit = {'source': str(source), 'command': {'argv': arguments}}
    assert query_options(arguments, source, 'msvc') == semantic
    assert _primitive_options(unit, 'msvc') == semantic


@pytest.mark.parametrize('prefix', ['/', '-'])
def test_msvc_query_removes_separate_outputs_without_consuming_semantic_options(
        tmp_path, prefix):
    source = tmp_path / 'unit.cpp'
    arguments = ['cl.exe', prefix + 'FA', prefix + 'MD', prefix + 'c', str(source),
                 prefix + 'Fo', str(tmp_path / 'unit.obj'),
                 prefix + 'Fa', str(tmp_path / 'unit.asm'),
                 prefix + 'Fd', str(tmp_path / 'unit.pdb'), '/fp:strict', '/GL-']
    assert query_options(arguments, source, 'msvc') == [
        prefix + 'MD', '/fp:strict', '/GL-']


@pytest.mark.parametrize('family', ['gnu', 'clang', 'msvc'])
def test_source_capture_interprets_runtime_and_dependency_target_by_family(
        tmp_path, family):
    source = tmp_path / 'unit.cpp'
    arguments = ['compiler', '-MT']
    if family != 'msvc':
        arguments.append('dependency-target.cpp')
    arguments.append(str(source))
    assert sources_in(arguments, tmp_path, family) == [source]


def _compiler_passes():
    """Actual /nologo /Bv grammar retained by Windows artifact 11006036952."""
    directory = (r'C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC'
                 r'\Tools\MSVC\14.51.36231\bin\Hostx64\x64')
    modules = [('cl.exe', '19.51.36257.0'), ('c1.dll', '19.51.36257.0'),
               ('c1xx.dll', '19.51.36257.0'), ('c2.dll', '19.51.36257.0'),
               ('c1xx.dll', '19.51.36257.0'), ('link.exe', '14.51.36257.0'),
               ('mspdb140.dll', '14.51.36257.0'),
               (r'1033\clui.dll', '19.51.36257.0')]
    text = 'Compiler Passes:\r\n' + ''.join(
        f' {directory}\\{name}:        Version {version}\r\n'
        for name, version in modules) + '\r\n'
    return text, {'path': directory + r'\cl.exe', 'size': 1, 'sha256': 'a' * 64}


def test_windows_nologo_version_binds_actual_cl_pass_without_guessing_a_banner():
    text, compiler = _compiler_passes()
    assert 'Compiler Version' not in text
    assert windows_trace.compiler_version(text, compiler) == '19.51.36257.0'


@pytest.mark.parametrize('mutation', [
    'no_passes', 'only_banner', 'missing_driver', 'relative_driver',
    'shadow_driver', 'duplicate_driver', 'conflicting_driver',
    'malformed_version', 'unparsed_pass', 'unterminated_passes', 'multiple_blocks',
])
def test_windows_compiler_version_refuses_unbound_or_ambiguous_pass_rows(mutation):
    text, compiler = _compiler_passes()
    driver = next(line for line in text.splitlines() if '\\cl.exe:' in line)
    if mutation == 'no_passes':
        text = text.replace('Compiler Passes:\r\n', '')
    elif mutation == 'only_banner':
        text = ('Microsoft (R) C/C++ Optimizing Compiler Version '
                '19.51.36257 for x64\r\n')
    elif mutation == 'missing_driver':
        text = text.replace(driver + '\r\n', '')
    elif mutation == 'relative_driver':
        text = text.replace(compiler['path'], 'cl.exe')
    elif mutation == 'shadow_driver':
        text = text.replace(compiler['path'], r'C:\shadow\cl.exe')
    elif mutation == 'duplicate_driver':
        text = text.replace(driver, driver + '\r\n' + driver)
    elif mutation == 'conflicting_driver':
        text = text.replace(driver, driver + '\r\n' +
                            driver.replace('19.51.36257.0', '19.52.99999.0'))
    elif mutation == 'malformed_version':
        text = text.replace('19.51.36257.0', 'unknown')
    elif mutation == 'unparsed_pass':
        text = text.replace(driver, ' Unsupported compiler pass information')
    elif mutation == 'unterminated_passes':
        text = text.rstrip() + '\r\n'
    else:
        text += text
    with pytest.raises(BuildEvidenceError, match='MSVC compiler version'):
        windows_trace.compiler_version(text, compiler)


@pytest.mark.parametrize('tool,diagnostics_present', [
    ('cl.exe', True), ('cl.exe', False), ('link.exe', False), ('mt.exe', False),
])
def test_windows_observer_requires_version_only_for_actual_compiler_process(
        tmp_path, monkeypatch, tool, diagnostics_present):
    """Exercise observe's role handling with literal process/image events."""
    images = [tmp_path / tool]
    if tool == 'cl.exe':
        images.extend([tmp_path / 'c1xx.dll', tmp_path / 'c2.dll'])
    for path in images:
        path.write_bytes(b'MZ fixture ' + path.name.encode())
    # Simulated Win32 image paths must remain drive-qualified even when this
    # portable fixture runs on a POSIX host. Python 3.13+ ntpath.isabs correctly
    # refuses a POSIX /tmp path as a Windows absolute path.
    image_root = 'C:/observed-msvc'
    mapped_images = {image_root + '/' + path.name: path for path in images}

    def mapped_identity(path):
        if str(path) in mapped_images:
            return {**file_identity(mapped_images[str(path)]), 'path': str(path)}
        return file_identity(path)

    monkeypatch.setattr(windows_trace, 'file_identity', mapped_identity)
    directory = tmp_path / 'observation'
    directory.mkdir()
    events = iter([(3, 1), *[(6, index) for index in range(2, len(images) + 1)],
                   (5, 0)])

    def create_process(*args):
        process = args[-1]._obj
        process.dwProcessId, process.dwThreadId = 42, 43
        process.hProcess, process.hThread = 100, 101
        if diagnostics_present:
            text, compiler = _compiler_passes()
            text = text.replace(compiler['path'].rsplit('\\', 1)[0], image_root)
            # This fixture already contains the compiler's literal CRLF bytes.
            (directory / 'stderr.txt').write_bytes(text.encode('utf8'))
        return True

    def wait_event(pointer, timeout):
        code, handle = next(events)
        event = pointer._obj
        event.code, event.pid, event.tid = code, 42, 43
        if code == 3:
            event.data.create.hFile = handle
        elif code == 6:
            event.data.load.hFile = handle
        else:
            event.data.exit_code = 0
        return True

    def image_path(handle, buffer, length, flags):
        buffer.value = image_root + '/' + images[handle - 1].name
        return len(buffer.value)

    kernel = SimpleNamespace(
        CreateProcessW=create_process, WaitForDebugEvent=wait_event,
        GetFinalPathNameByHandleW=image_path, ContinueDebugEvent=lambda *args: True,
        CloseHandle=lambda *args: True, GetStdHandle=lambda *args: 0)
    monkeypatch.setitem(sys.modules, 'msvcrt',
                        SimpleNamespace(get_osfhandle=lambda n: n))
    monkeypatch.setattr(windows_trace.C, 'WinDLL', lambda *args, **kwargs: kernel,
                        raising=False)
    monkeypatch.setattr(windows_trace.os, 'set_handle_inheritable',
                        lambda *args: None, raising=False)
    result = windows_trace.observe([str(images[0])], cwd=tmp_path, env={},
                                   directory=directory)
    assert result['exit_code'] == 0
    if tool == 'cl.exe' and diagnostics_present:
        assert result['driver_version'] == '19.51.36257.0'
        assert result['problems'] == []
    elif tool == 'cl.exe':
        assert result['driver_version'] is None
        assert result['problems'] == ['missing/ambiguous MSVC compiler version pass']
    else:
        assert result['driver_version'] is None
        assert result['problems'] == []


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


def _library_only_report(tmp_path):
    """Observed VS 18 grammar: six complete passes, 16 archives, no member rows.

    4f04e90 Windows310 artifact 11004577833 has this same pass shape in all
    three LINK stdout files. Local fixture paths replace the runner paths.
    """
    obj, archive, _, rsp, verbose = _library_report(tmp_path)
    names = ('python310', 'kernel32', 'user32', 'gdi32', 'winspool', 'shell32',
             'ole32', 'oleaut32', 'uuid', 'comdlg32', 'advapi32', 'msvcprt',
             'MSVCRT', 'OLDNAMES', 'vcruntime', 'ucrt')
    archives = [tmp_path / (name + '.lib') for name in names]
    content = archive.read_bytes()
    for path in archives:
        path.write_bytes(content)
    rsp.write_text('\n'.join('"' + str(path) + '"'
                             for path in [obj, *archives[:11]]) + '\n')
    passes = [archives * 2 + archives[:1], *([archives] * 4), archives[:-1]]
    blocks = ['\nSearching libraries\n' +
              ''.join('    Searching ' + str(path) + ':\n' for path in paths) +
              '\nFinished searching libraries\n' for paths in passes]
    blocks[0] += '   Creating library image.lib and object image.exp\n'
    verbose.write_text(''.join(blocks))
    return obj, archives, rsp, verbose


def _pe_image(path, *, dll=True, manifest=None, resource_id=2, code=b'\xc3\x90'):
    """Literal PE32+ fixture with .text and an optional ID/language resource tree."""
    data = bytearray(1536 if manifest is not None else 1024)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 60, 128)
    data[128:132] = b'PE\0\0'
    struct.pack_into('<HH', data, 132, 0x8664, 2 if manifest is not None else 1)
    struct.pack_into('<HH', data, 148, 240, 0x2022 if dll else 0x22)
    struct.pack_into('<H', data, 152, 0x20b)
    struct.pack_into('<III', data, 156, 512, 512 if manifest is not None else 0, 0)
    struct.pack_into('<I', data, 168, 4096)
    struct.pack_into('<Q', data, 176, 0x180000000)
    struct.pack_into('<II', data, 184, 4096, 512)
    struct.pack_into('<II', data, 208, 12288 if manifest is not None else 8192, 512)
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


def _pe_with_relocations(path, *, manifest=None):
    """mt's observed insertion moves .reloc storage, not its relocation targets."""
    _pe_image(path, manifest=manifest)
    data = bytearray(path.read_bytes()) + bytearray(512)
    final = manifest is not None
    struct.pack_into('<H', data, 134, 3 if final else 2)
    struct.pack_into('<I', data, 160, 1024 if final else 512)
    struct.pack_into('<I', data, 208, 16384 if final else 12288)
    header = 472 if final else 432
    address, raw = (12288, 1536) if final else (8192, 1024)
    data[header:header + 8] = b'.reloc\0\0'
    struct.pack_into('<IIII', data, header + 8, 12, address, 512, raw)
    struct.pack_into('<I', data, header + 36, 0x42000040)
    struct.pack_into('<II', data, 304, address, 12)  # Base relocation directory.
    struct.pack_into('<IIHH', data, raw, 4096, 12, 0xa008, 0)
    path.write_bytes(data)


def _refresh_pe_claims(row):
    """Let negative controls reach structural checks after rebinding file bytes."""
    receipt = row['windows_manifest']
    receipt['donor'] = file_identity(receipt['donor']['path'])
    receipt['linked_output'] = {**receipt['donor'], 'path': row['output']['path']}
    receipt['output'] = row['output'] = file_identity(row['output']['path'])
    for index, item in enumerate(row['evidence_files']):
        if item['path'] == receipt['donor']['path']:
            row['evidence_files'][index] = receipt['donor']


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
    assert report['coverage'] == {
        'mode': 'selected-members-and-searched-archives', 'search_passes': 1,
    }
    assert report['loaded_members'] == [{
        'archive': str(archive), 'symbol': '__imp_second',
        'reported_member': 'runtime.dll', 'referenced_by': ['unit.obj'],
        'member': {'name': 'runtime.dll', 'offset': offsets[1], 'size': 13,
                   'sha256': hashlib.sha256(b'second import').hexdigest()},
    }]


def test_windows_library_only_report_binds_complete_conservative_archive_closure(
        tmp_path, monkeypatch):
    row, _ = _manifest_row(tmp_path, monkeypatch)
    obj, archives, rsp, verbose = _library_only_report(tmp_path)
    report = windows_link.collect_link_inputs(rsp, verbose)
    assert report['coverage'] == {
        'mode': 'conservative-searched-archives', 'search_passes': 6,
    }
    assert report['loaded_members'] == []
    assert report['searched_libraries'] == sorted(
        [file_identity(path) for path in archives], key=lambda item: item['path'])
    row.update(windows_link_inputs=report,
               link_inputs=[file_identity(path) for path in [obj, *archives]],
               object_inputs=[file_identity(obj)],
               runtime_link_inputs=[file_identity(path) for path in archives])
    row['invocations'][0]['expanded_argv'].extend([
        '/VERBOSE:LIB', '/LINKREPROFULLPATHRSP:' + str(rsp)])
    row['evidence_files'].extend([file_identity(rsp), file_identity(verbose)])
    assert verify_link_closure(row, {str(obj): {'output': file_identity(obj)}}) == [
        file_identity(obj)]


@pytest.mark.parametrize('mutation', [
    'omit_claim', 'omit_input', 'log_tamper', 'partial_search',
    'provider_substitution', 'shadow_archive', 'archive_tamper',
])
def test_windows_library_only_coverage_cannot_hide_missing_or_untrusted_inputs(
        tmp_path, monkeypatch, mutation):
    row, _ = _manifest_row(tmp_path, monkeypatch)
    obj, archives, rsp, verbose = _library_only_report(tmp_path)
    report = windows_link.collect_link_inputs(rsp, verbose)
    row.update(windows_link_inputs=report,
               link_inputs=[file_identity(path) for path in [obj, *archives]],
               object_inputs=[file_identity(obj)],
               runtime_link_inputs=[file_identity(path) for path in archives])
    row['invocations'][0]['expanded_argv'].extend([
        '/VERBOSE:LIB', '/LINKREPROFULLPATHRSP:' + str(rsp)])
    row['evidence_files'].extend([file_identity(rsp), file_identity(verbose)])
    if mutation == 'omit_claim':
        report['searched_libraries'].pop()
    elif mutation == 'omit_input':
        row['link_inputs'].pop()
    elif mutation == 'log_tamper':
        verbose.write_text(verbose.read_text() + 'Unrecorded library search\n')
    elif mutation == 'partial_search':
        verbose.write_text(verbose.read_text().rsplit(
            'Finished searching libraries', 1)[0])
        report['verbose_log'] = file_identity(verbose)
        row['evidence_files'][-1] = file_identity(verbose)
    elif mutation == 'provider_substitution':
        row['runtime_link_inputs'].pop()
    elif mutation == 'shadow_archive':
        shadow = tmp_path / 'shadow' / archives[-1].name
        shadow.parent.mkdir()
        shadow.write_bytes(archives[-1].read_bytes())
        verbose.write_text(verbose.read_text().replace(str(archives[-1]), str(shadow)))
        row['windows_link_inputs'] = windows_link.collect_link_inputs(rsp, verbose)
        row['link_inputs'][-1] = file_identity(shadow)
        row['evidence_files'][-1] = file_identity(verbose)
    else:
        archives[-1].write_bytes(archives[-1].read_bytes() + b'changed')
    with pytest.raises(BuildEvidenceError):
        verify_link_closure(row, {str(obj): {'output': file_identity(obj)}})


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


def test_windows_manifest_preserves_relocation_semantics_when_table_storage_moves(
        tmp_path, monkeypatch):
    row, _ = _manifest_row(tmp_path, monkeypatch)
    _pe_with_relocations(Path(row['windows_manifest']['donor']['path']))
    _pe_with_relocations(Path(row['output']['path']), manifest=MANIFEST)
    _refresh_pe_claims(row)
    windows_link.verify_manifest(row)


@pytest.mark.parametrize('mutation', [
    'code_rva', 'data_rva', 'relocation_bytes', 'relocation_targets',
    'relocation_directory', 'duplicate_sections', 'image_base', 'entrypoint',
    'unaligned_raw', 'overlapping_raw', 'virtual_header_overlap',
])
def test_windows_manifest_relocation_move_cannot_change_image_semantics(
        tmp_path, monkeypatch, mutation):
    row, _ = _manifest_row(tmp_path, monkeypatch)
    donor, final = (Path(row['windows_manifest']['donor']['path']),
                    Path(row['output']['path']))
    _pe_with_relocations(donor)
    _pe_with_relocations(final, manifest=MANIFEST)
    data = bytearray(final.read_bytes())
    if mutation == 'code_rva':
        struct.pack_into('<I', data, 404, 16384)
    elif mutation == 'data_rva':
        before = bytearray(donor.read_bytes())
        for image in (before, data):
            image[392:400] = b'.data\0\0\0'
            struct.pack_into('<I', image, 428, 0xc0000040)
        donor.write_bytes(before)
        struct.pack_into('<I', data, 404, 16384)
    elif mutation == 'relocation_bytes':
        struct.pack_into('<H', data, 1544, 0xa010)
    elif mutation == 'relocation_targets':
        before = bytearray(donor.read_bytes())
        struct.pack_into('<I', before, 1024, 8192)
        donor.write_bytes(before)
        struct.pack_into('<I', data, 1536, 8192)
    elif mutation == 'relocation_directory':
        struct.pack_into('<I', data, 304, 12292)
    elif mutation == 'duplicate_sections':
        data[432:440] = b'.reloc\0\0'
    elif mutation == 'image_base':
        struct.pack_into('<Q', data, 176, 0x140000000)
    elif mutation == 'entrypoint':
        struct.pack_into('<I', data, 168, 4100)
    elif mutation == 'unaligned_raw':
        data += b'\0' + data[512:1024]
        struct.pack_into('<I', data, 412, 2049)
    elif mutation == 'virtual_header_overlap':
        struct.pack_into('<I', data, 484, 0)
        struct.pack_into('<I', data, 304, 0)
        struct.pack_into('<I', data, 208, 12288)
    else:
        # Identical bytes cannot justify aliased raw section mappings.
        before = bytearray(donor.read_bytes())
        before[512:1024] = before[1024:1536]
        data[512:1024] = data[1536:2048]
        donor.write_bytes(before)
        struct.pack_into('<I', data, 412, 1536)
    final.write_bytes(data)
    _refresh_pe_claims(row)
    with pytest.raises(BuildEvidenceError, match='Windows manifest'):
        windows_link.verify_manifest(row)


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
        [compiler, '/nologo', '/O2', '-MD', '/fp:strict', '/GL-', '/c', str(source),
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
    rows = [json.loads(path.read_text()) for path in records.glob('*.json')]
    compile_row = next(row for row in rows if row['kind'] == 'compile')
    options = _primitive_options({'source': compile_row['source'],
                                  'command': {'argv': compile_row['effective_argv']}},
                                 'msvc')
    assert '-MD' in options
    assert '/fp:strict' in options
    row = next(row for row in rows if row['kind'] == 'link')
    assert any(Path(item['path']).name.lower() == 'msvcrt.lib'
               for item in row['windows_link_inputs']['searched_libraries'])
    assert row['windows_manifest']['resource']['resource_id'] == (2 if dll else 1)
    assert [item['role'] for item in row['invocations']] == ['linker', 'manifest_tool']
