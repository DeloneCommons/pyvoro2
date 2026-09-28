"""Actual process/image observation for the closed native Windows adapter.

Only direct x64 cl.exe (/c, one TU, no /MP) and link.exe are supported.
Windows' debugger events identify the images actually mapped in the compiler
and linker. MSVC's in-process backend has no independent OS argv: the direct
cl argv, CL/_CL_ expansion, loaded c1xx/c2 images, and completed object output
form its invocation evidence. /Bv is corroborating version information only.

This build-only observer uses documented Win32 debugging APIs. It does not
inspect a private PEB layout or require a package runtime dependency.
"""
from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import shutil
import subprocess

from .effective_build import BuildEvidenceError, file_identity


def observe(command, *, cwd, env, directory):
    import msvcrt

    if C.sizeof(C.c_void_p) != 8:
        raise BuildEvidenceError('Windows observation requires an x64 process')
    executable = Path(shutil.which(command[0], path=env.get('PATH')) or command[0])
    executable = executable.resolve(strict=True)
    name = executable.name.lower()
    if name not in ('cl.exe', 'link.exe'):
        raise BuildEvidenceError('opaque Windows wrapper; direct cl/link required')
    if any(a.lower().startswith('/mp') for a in command[1:]):
        raise BuildEvidenceError('Windows adapter requires Ninja, not internal /MP')
    kernel = C.WinDLL('kernel32', use_last_error=True)
    pointer = C.c_void_p

    class StartInfo(C.Structure):
        _fields_ = [('cb', W.DWORD), ('lpReserved', W.LPWSTR),
                    ('lpDesktop', W.LPWSTR), ('lpTitle', W.LPWSTR),
                    ('dwX', W.DWORD), ('dwY', W.DWORD),
                    ('dwXSize', W.DWORD), ('dwYSize', W.DWORD),
                    ('dwXCountChars', W.DWORD), ('dwYCountChars', W.DWORD),
                    ('dwFillAttribute', W.DWORD), ('dwFlags', W.DWORD),
                    ('wShowWindow', W.WORD), ('cbReserved2', W.WORD),
                    ('lpReserved2', pointer), ('hStdInput', W.HANDLE),
                    ('hStdOutput', W.HANDLE), ('hStdError', W.HANDLE)]

    class ProcessInfo(C.Structure):
        _fields_ = [('hProcess', W.HANDLE), ('hThread', W.HANDLE),
                    ('dwProcessId', W.DWORD), ('dwThreadId', W.DWORD)]

    class CreateInfo(C.Structure):
        _fields_ = [('hFile', W.HANDLE), ('hProcess', W.HANDLE),
                    ('hThread', W.HANDLE), ('lpBaseOfImage', pointer),
                    ('dwDebugInfoFileOffset', W.DWORD),
                    ('nDebugInfoSize', W.DWORD), ('lpThreadLocalBase', pointer),
                    ('lpStartAddress', pointer), ('lpImageName', pointer),
                    ('fUnicode', W.WORD)]

    class LoadInfo(C.Structure):
        _fields_ = [('hFile', W.HANDLE), ('lpBaseOfDll', pointer),
                    ('dwDebugInfoFileOffset', W.DWORD),
                    ('nDebugInfoSize', W.DWORD), ('lpImageName', pointer),
                    ('fUnicode', W.WORD)]

    class EventUnion(C.Union):
        _fields_ = [('create', CreateInfo), ('load', LoadInfo),
                    ('exit_code', W.DWORD), ('exception_code', W.DWORD),
                    ('raw', C.c_byte * 160)]

    class DebugEvent(C.Structure):
        _fields_ = [('code', W.DWORD), ('pid', W.DWORD), ('tid', W.DWORD),
                    ('data', EventUnion)]

    kernel.CreateProcessW.argtypes = [
        W.LPCWSTR, W.LPWSTR, pointer, pointer, W.BOOL, W.DWORD, pointer,
        W.LPCWSTR, C.POINTER(StartInfo), C.POINTER(ProcessInfo)]
    kernel.CreateProcessW.restype = W.BOOL
    kernel.WaitForDebugEvent.argtypes = [C.POINTER(DebugEvent), W.DWORD]
    kernel.WaitForDebugEvent.restype = W.BOOL
    kernel.ContinueDebugEvent.argtypes = [W.DWORD, W.DWORD, W.DWORD]
    kernel.ContinueDebugEvent.restype = W.BOOL
    kernel.CloseHandle.argtypes = [W.HANDLE]
    kernel.GetFinalPathNameByHandleW.argtypes = [
        W.HANDLE, W.LPWSTR, W.DWORD, W.DWORD]
    kernel.GetFinalPathNameByHandleW.restype = W.DWORD
    kernel.GetStdHandle.argtypes = [W.DWORD]
    kernel.GetStdHandle.restype = W.HANDLE

    def image_identity(handle):
        if not handle:
            raise BuildEvidenceError('Windows image event lacks its file handle')
        buffer = C.create_unicode_buffer(32768)
        size = kernel.GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0)
        if not size or size >= len(buffer):
            raise BuildEvidenceError('cannot resolve actual mapped Windows image')
        return file_identity(buffer.value)

    stdout, stderr = directory / 'stdout.txt', directory / 'stderr.txt'
    events, invocations, problems, loaded = [], [], [], []
    process_info = ProcessInfo()
    raw_command = subprocess.list2cmdline([str(executable), *command[1:]])
    environment = C.create_unicode_buffer(
        '\0'.join(f'{k}={v}' for k, v in sorted(env.items())) + '\0\0')
    with stdout.open('wb') as out, stderr.open('wb') as err:
        out_handle = msvcrt.get_osfhandle(out.fileno())
        err_handle = msvcrt.get_osfhandle(err.fileno())
        os.set_handle_inheritable(out_handle, True)
        os.set_handle_inheritable(err_handle, True)
        start = StartInfo()
        start.cb = C.sizeof(start)
        start.dwFlags = 0x100  # STARTF_USESTDHANDLES
        start.hStdInput = kernel.GetStdHandle(W.DWORD(-10).value)
        start.hStdOutput, start.hStdError = out_handle, err_handle
        command_buffer = C.create_unicode_buffer(raw_command)
        if not kernel.CreateProcessW(
                str(executable), command_buffer, None, None, True,
                0x1 | 0x400, environment, str(cwd), C.byref(start),
                C.byref(process_info)):  # DEBUG_PROCESS | UNICODE_ENVIRONMENT
            raise C.WinError(C.get_last_error())
        live, root_exit = set(), None
        try:
            while root_exit is None or live:
                event = DebugEvent()
                if not kernel.WaitForDebugEvent(C.byref(event), 0xFFFFFFFF):
                    raise C.WinError(C.get_last_error())
                continuation = 0x00010002  # DBG_CONTINUE
                entry = {'event': event.code, 'pid': event.pid, 'tid': event.tid}
                if event.code == 3:  # CREATE_PROCESS_DEBUG_EVENT
                    live.add(event.pid)
                    try:
                        identity = image_identity(event.data.create.hFile)
                    finally:
                        if event.data.create.hFile:
                            kernel.CloseHandle(event.data.create.hFile)
                    entry['image'] = identity
                    if event.pid == process_info.dwProcessId:
                        invocations.append({
                            'pid': event.pid, 'executable': identity,
                            'argv': [str(executable), *command[1:]],
                            'raw_command_line': raw_command, 'cwd': str(cwd),
                            'environment': env,
                            'role': 'linker' if name == 'link.exe' else
                                    'compiler_driver',
                            'opened_files': [],
                        })
                    else:
                        # The closed adapter does not guess a child's argv.
                        # A PDB service is not a numerical compiler/link input.
                        if not Path(identity['path']).name.lower().startswith(
                                'mspdbsrv'):
                            problems.append('unobserved Windows child inputs: ' +
                                            identity['path'])
                elif event.code == 6:  # LOAD_DLL_DEBUG_EVENT
                    try:
                        identity = image_identity(event.data.load.hFile)
                    except BuildEvidenceError as error:
                        problems.append(str(error))
                    else:
                        entry['image'] = identity
                        if event.pid == process_info.dwProcessId:
                            loaded.append(identity)
                    finally:
                        if event.data.load.hFile:
                            kernel.CloseHandle(event.data.load.hFile)
                elif event.code == 5:  # EXIT_PROCESS_DEBUG_EVENT
                    entry['exit_code'] = event.data.exit_code
                    live.discard(event.pid)
                    if event.pid == process_info.dwProcessId:
                        root_exit = int(event.data.exit_code)
                elif event.code == 1:  # Initial loader breakpoint is expected.
                    entry['exception_code'] = event.data.exception_code
                    if event.data.exception_code not in (0x80000003, 0x80000004):
                        continuation = 0x80010001  # DBG_EXCEPTION_NOT_HANDLED
                events.append(entry)
                if not kernel.ContinueDebugEvent(event.pid, event.tid, continuation):
                    raise C.WinError(C.get_last_error())
        finally:
            kernel.CloseHandle(process_info.hThread)
            kernel.CloseHandle(process_info.hProcess)
    backend = [i for i in loaded if Path(i['path']).name.lower() == 'c2.dll']
    frontend = [i for i in loaded if Path(i['path']).name.lower() == 'c1xx.dll']
    if name == 'cl.exe':
        if not backend or not frontend:
            problems.append('missing actual MSVC front/back-end image loads')
        for identity in backend:
            invocations.append({
                **invocations[0], 'executable': identity,
                'role': 'compiler_backend',
                'observation': 'in-process codegen DLL load plus object output',
            })
    elif backend:
        problems.append('linker loaded code-generation backend (possible LTCG)')
    event_path = directory / 'windows-debug-events.json'
    event_path.write_text(json.dumps(events, sort_keys=True, indent=2) + '\n')
    version_text = (stdout.read_text(errors='replace') +
                    stderr.read_text(errors='replace'))
    version = next((line.strip() for line in version_text.splitlines()
                    if 'Compiler Version' in line), '')
    if name == 'link.exe':
        # Link jobs bind the same compiler through their matching compile
        # records; linker version is still present in the captured diagnostics.
        version = None
    return {
        'adapter': 'windows-debug-v1', 'exit_code': root_exit,
        'invocations': invocations, 'problems': problems,
        'evidence_paths': [stdout, stderr, event_path],
        'loaded_images': loaded, 'observer': file_identity(__file__),
        'driver_version': version, 'driver_target': 'x86_64-windows-msvc',
    }
