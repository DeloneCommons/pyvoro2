"""Close actual link inputs against recorded objects and installed runtimes."""
from __future__ import annotations

from pathlib import Path
import struct
import subprocess
import sys

from .effective_build import BuildEvidenceError, canonical, file_identity

# These are implementation inputs of the supported GNU ABI, not a wildcard for
# any library found on a candidate's -L path. Each is independently resolved by
# the fixed compiler without candidate flags or search-path environment.
GNU_RUNTIME_FILES = (
    'crt1.o', 'Scrt1.o', 'crti.o', 'crtn.o', 'crtbegin.o', 'crtbeginS.o',
    'crtend.o', 'crtendS.o', 'libgcc.a', 'libgcc_eh.a', 'libgcc_s.so',
    'libgcc_s.so.1', 'libstdc++.so', 'libstdc++.so.6', 'libstdc++.a',
    'libsupc++.a', 'libstdc++_nonshared.a', 'libc.so', 'libc.so.6',
    'libc_nonshared.a', 'libm.so', 'libm.so.6', 'libmvec.so.1',
    'libmvec_nonshared.a', 'libpthread.so', 'libpthread.so.0',
    'libdl.so', 'libdl.so.2', 'librt.so', 'librt.so.1',
    'ld-linux-x86-64.so.2', 'libasan.so', 'libasan_preinit.o',
    'libubsan.so', 'libatomic.so', 'libatomic.so.1',
)

# Libsystem's requiredlibs/optionallibs include these sublibraries in addition
# to libsystem_*. They resolve only inside the independently selected SDK;
# candidate search paths and arbitrary SDK library names grant no authority.
# https://github.com/apple-oss-distributions/Libsystem/tree/
# c23b51183e46e613375dc7639fb4bf11fbf59ba8
APPLE_SYSTEM_RUNTIME_FILES = (
    'libcache.tbd', 'libcommonCrypto.tbd', 'libcompiler_rt.tbd',
    'libcopyfile.tbd', 'libcorecrypto.tbd', 'libdispatch.tbd',
    'libdyld.tbd', 'libkeymgr.tbd', 'libmacho.tbd', 'libquarantine.tbd',
    'libremovefile.tbd', 'libunwind.tbd', 'libxpc.tbd',
)


def clean_toolchain_environment(env):
    excluded = {
        'CPATH', 'CPLUS_INCLUDE_PATH', 'C_INCLUDE_PATH', 'OBJC_INCLUDE_PATH',
        'INCLUDE', 'LIB', 'LIBPATH', 'LIBRARY_PATH', 'COMPILER_PATH',
        'GCC_EXEC_PREFIX', 'CL', '_CL_', 'LINK', '_LINK_', 'SDKROOT',
        'CCC_OVERRIDE_OPTIONS', 'CLANG_CONFIG_FILE_SYSTEM_DIR',
        'CLANG_CONFIG_FILE_USER_DIR',
    }
    return {k: v for k, v in env.items() if k not in excluded}


def native_kind(path):
    """Classify actual bytes, so renaming an object cannot hide its role."""
    data = Path(path).read_bytes()[:64]
    if data.startswith((b'!<arch>\n', b'!<thin>\n')):
        return 'archive'
    if data.startswith((b'BC\xc0\xde', b'\xde\xc0\x17\x0b')):
        return 'bitcode'
    if data.startswith(b'\x7fELF') and len(data) >= 20:
        byteorder = '<' if data[5] == 1 else '>'
        kind = struct.unpack_from(byteorder + 'H', data, 16)[0]
        return 'object' if kind == 1 else 'image'
    if data[:4] in (b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf'):
        byteorder = '<' if data[0] == 0xcf else '>'
        kind = struct.unpack_from(byteorder + 'I', data, 12)[0]
        return 'object' if kind == 1 else 'image'
    if data[:2] in (b'\x64\x86', b'\x64\xaa') or data[:4] == b'\0\0\xff\xff':
        return 'object'  # COFF / bigobj; /GL is independently forbidden.
    if data[:2] == b'MZ':
        return 'image'
    return 'other'


def verify_apple_native_object(path):
    """Exclude LTO payloads in every observed project/control link object.

    Apple always forwards an LTO outliner default to ld. That does not grant
    cross-unit optimization: the linked source closure must contain native
    Mach-O code, with no embedded LLVM segment/bitcode section. Installed
    runtime archives remain independently resolved toolchain inputs.
    """
    data = Path(path).read_bytes()

    def invalid():
        raise BuildEvidenceError('incomplete/unsupported Mach-O native object')

    if len(data) < 32 or data[:4] != b'\xcf\xfa\xed\xfe':
        invalid()
    _, cpu, _, kind, count, size, _, _ = struct.unpack_from('<8I', data)
    end = 32 + size
    if (cpu not in (0x01000007, 0x0100000c) or kind != 1 or not count or
            count > size // 8 or end > len(data)):
        invalid()
    cursor, sections = 32, 0
    for _ in range(count):
        if cursor + 8 > end:
            invalid()
        command, length = struct.unpack_from('<II', data, cursor)
        if length < 8 or length % 8 or cursor + length > end:
            invalid()
        if command == 0x19:  # LC_SEGMENT_64, followed by section_64 entries.
            if length < 72:
                invalid()
            segment = data[cursor + 8:cursor + 24].split(b'\0', 1)[0]
            section_count = struct.unpack_from('<I', data, cursor + 64)[0]
            if length != 72 + section_count * 80:
                invalid()
            if segment == b'__LLVM':
                raise BuildEvidenceError('LTO/bitcode in actual Mach-O object')
            for index in range(section_count):
                offset = cursor + 72 + 80 * index
                name = data[offset:offset + 16].split(b'\0', 1)[0]
                segment = data[offset + 16:offset + 32].split(b'\0', 1)[0]
                if segment == b'__LLVM' or name.startswith((b'__bitcode', b'__llvm')):
                    raise BuildEvidenceError('LTO/bitcode in actual Mach-O object')
                section_size = struct.unpack_from('<Q', data, offset + 40)[0]
                file_offset = struct.unpack_from('<I', data, offset + 48)[0]
                flags = struct.unpack_from('<I', data, offset + 64)[0]
                # Zero-fill sections have no file payload; others must be complete.
                if flags & 0xff not in (1, 12, 18) and section_size:
                    if file_offset < end or file_offset + section_size > len(data):
                        invalid()
                sections += 1
        cursor += length
    if cursor != end or not sections:
        invalid()


def gnu_runtime_inputs(driver, *, cwd, env, directory):
    identities, queries = {}, []
    environment = clean_toolchain_environment(env)
    for name in GNU_RUNTIME_FILES:
        argv = [str(driver), '-print-file-name=' + name]
        run = subprocess.run(argv, cwd=cwd, env=environment,
                             text=True, capture_output=True)
        queries.append({'argv': argv, 'exit_code': run.returncode,
                        'stdout': run.stdout, 'stderr': run.stderr})
        if run.returncode:
            raise BuildEvidenceError('fixed toolchain link input query failed')
        word = run.stdout.strip()
        if word == name:  # This optional runtime is not installed.
            continue
        path = Path(word).resolve(strict=True)
        identity = file_identity(path)
        identities[identity['path']] = identity
    evidence = directory / 'toolchain-link-inputs.json'
    evidence.write_bytes(canonical({'environment': environment,
                                    'queries': queries}) + b'\n')
    return list(identities.values()), evidence


def platform_runtime_inputs(driver, family, *, cwd, env, directory):
    if family == 'gnu':
        return gnu_runtime_inputs(driver, cwd=cwd, env=env, directory=directory)
    clean = clean_toolchain_environment(env)
    queries, candidates = [], []
    if family == 'clang':
        for argv in ([str(driver), '-print-resource-dir'],
                     ['/usr/bin/xcrun', '--sdk', 'macosx', '--show-sdk-path']):
            run = subprocess.run(argv, cwd=cwd, env=clean,
                                 text=True, capture_output=True)
            queries.append({'argv': argv, 'exit_code': run.returncode,
                            'stdout': run.stdout, 'stderr': run.stderr})
            if run.returncode or not Path(run.stdout.strip()).is_dir():
                raise BuildEvidenceError('independent Apple runtime query failed')
        resource, sdk = (Path(row['stdout'].strip()).resolve() for row in queries)
        for name in ('libSystem.tbd', 'libc++.tbd', 'libc++abi.tbd',
                     'libobjc.tbd', 'libSystem.B.tbd', 'libc++.1.tbd'):
            candidates.append(sdk / 'usr' / 'lib' / name)
        candidates.extend((sdk / 'usr' / 'lib' / 'system').glob('libsystem_*.tbd'))
        candidates.extend(sdk / 'usr' / 'lib' / 'system' / name
                          for name in APPLE_SYSTEM_RUNTIME_FILES)
        for name in ('libclang_rt.osx.a', 'libclang_rt.asan_osx_dynamic.dylib',
                     'libclang_rt.ubsan_osx_dynamic.dylib'):
            candidates.append(resource / 'lib' / 'darwin' / name)
        candidates.append(Path(driver).parent.parent / 'lib' / 'libLTO.dylib')
    elif family == 'msvc':
        from .input_provenance import windows_installation_roots
        installed = windows_installation_roots(driver)
        queries.append({'installation': installed})
        names = {
            'libcmt.lib', 'libcpmt.lib', 'libvcruntime.lib', 'libucrt.lib',
            'msvcrt.lib', 'msvcprt.lib', 'vcruntime.lib', 'ucrt.lib',
            'oldnames.lib', 'legacy_stdio_definitions.lib', 'kernel32.lib',
            'user32.lib', 'gdi32.lib', 'winspool.lib', 'shell32.lib',
            'ole32.lib', 'oleaut32.lib', 'uuid.lib', 'comdlg32.lib',
            'advapi32.lib', 'ws2_32.lib', 'bcrypt.lib', 'ntdll.lib',
        }
        for root in installed['libraries']:
            candidates.extend(Path(root) / name for name in names)
        python_library = f'python{sys.version_info.major}{sys.version_info.minor}.lib'
        candidates.append(Path(sys.base_prefix) / 'libs' / python_library)
    else:
        raise BuildEvidenceError('unsupported native runtime provenance adapter')
    identities = {str(path.resolve()): file_identity(path)
                  for path in candidates if path.is_file()}
    evidence = directory / 'toolchain-link-inputs.json'
    evidence.write_bytes(canonical({
        'environment': clean, 'queries': queries,
        'installed_files': list(identities.values()),
    }) + b'\n')
    return list(identities.values()), evidence


def verify_linker_options(argv, family):
    """The observed linker may not rewrite the source-coupled call graph."""
    index = 1
    while index < len(argv):
        arg = argv[index]
        index += 1
        if family == 'msvc':
            option = arg.lower()
            if not option.startswith(('/', '-')):
                continue
            if option.startswith('/manifest:'):
                raise BuildEvidenceError(
                    'unobserved Windows manifest resource control: ' + arg)
            if option in ('/dll', '/nologo', '/incremental:no', '/manifest',
                          '/machine:x64', '/ltcg:off', '/verbose:lib',
                          '/opt:ref', '/opt:icf', '/debug', '/release',
                          '/subsystem:console', '/subsystem:windows'):
                continue
            if option == "/manifestuac:level='asinvoker' uiaccess='false'":
                continue
            if option.startswith(('/out:', '/implib:', '/pdb:', '/libpath:',
                                  '/linkreprofullpathrsp:', '/defaultlib:',
                                  '/version:', '/manifestfile:')):
                continue
            raise BuildEvidenceError('unreviewed actual linker control: ' + arg)
        if not arg.startswith('-'):
            continue
        common = {'-o', '-L', '-l'}
        if arg in common:
            index += 1
            continue
        if arg.startswith(('-L', '-l')) and arg not in ('-lto_library',):
            continue
        if family == 'gnu':
            if arg in ('--build-id', '--eh-frame-hdr', '--as-needed',
                       '--no-as-needed', '-shared', '-pie', '-t',
                       '--no-add-needed', '--no-copy-dt-needed-entries',
                       '--hash-style=gnu', '--hash-style=both',
                       '--push-state', '--pop-state', '--whole-archive',
                       '--no-whole-archive', '-Bstatic', '-Bdynamic',
                       '--enable-new-dtags', '--disable-new-dtags'):
                continue
            if arg.startswith(('--dependency-file=', '--build-id=')):
                continue
            if arg in ('-m', '-z') and index < len(argv):
                allowed = {'-m': ('elf_x86_64',),
                           '-z': ('relro', 'now', 'noexecstack', 'defs')}
                if argv[index] in allowed[arg]:
                    index += 1
                    continue
            if arg == '-dynamic-linker':
                index += 1
                continue
        else:
            # ld's -O3 preserves its ordinary native-code deduplication policy.
            # The exact LLVM option is the Clang driver's default for LTO only;
            # verify_link_closure independently excludes project/control bitcode.
            # Other optimization/LLVM controls remain unreviewed.
            if arg == '-O3':
                continue
            if (arg == '-mllvm' and
                    argv[index:index + 1] == ['-enable-linkonceodr-outlining']):
                index += 1
                continue
            if arg in ('-demangle', '-no_deduplicate', '-dynamic', '-dylib',
                       '-bundle', '-t', '-headerpad_max_install_names',
                       '-search_paths_first', '-adhoc_codesign',
                       '-no_adhoc_codesign'):
                continue
            if arg in ('-arch', '-syslibroot', '-lto_library', '-undefined'):
                if arg == '-undefined' and argv[index:index + 1] != ['dynamic_lookup']:
                    raise BuildEvidenceError('unreviewed undefined-symbol policy')
                index += 1
                continue
            if arg == '-platform_version':
                index += 3
                continue
        raise BuildEvidenceError('unreviewed actual linker control: ' + arg)


def verify_link_closure(row, units):
    """Use every observed linker input, not just driver-level .o arguments."""
    from .effective_build import _verify_identity

    if row.get('family') == 'msvc':
        from .windows_link import verify_windows_link
        verify_windows_link(row)
    trusted = {item['path']: item for item in row.get('runtime_link_inputs', [])}
    for item in trusted.values():
        _verify_identity(item, 'toolchain link input')
    actual = row.get('link_inputs', [])
    if not actual:
        raise BuildEvidenceError('missing actual link input inventory')
    objects = {}
    for item in actual:
        _verify_identity(item, 'link input')
        if trusted.get(item['path']) == item:
            continue
        kind = native_kind(item['path'])
        if kind == 'object':
            if item['path'] not in units or units[item['path']]['output'] != item:
                raise BuildEvidenceError('unobserved/mismatched linked object')
            if row.get('family') == 'clang':
                verify_apple_native_object(item['path'])
            objects[item['path']] = item
        elif kind in ('archive', 'bitcode', 'image'):
            raise BuildEvidenceError(
                'unapproved native ' + kind + ' link input: ' + item['path'])
        else:
            # A candidate linker script can name further objects/libraries.
            # Only independently resolved installed runtime scripts are trusted.
            raise BuildEvidenceError('opaque/unapproved link input: ' + item['path'])
    if not objects:
        raise BuildEvidenceError('missing final link object inputs')
    claimed = {item['path']: item for item in row.get('object_inputs', [])}
    if claimed != objects:
        raise BuildEvidenceError('incomplete actual linked object association')
    return list(objects.values())
