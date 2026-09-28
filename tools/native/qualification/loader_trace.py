"""Actual platform-loader observations for build-tool image dependencies."""
from __future__ import annotations

from pathlib import Path
import re
import struct

from .effective_build import BuildEvidenceError, file_identity


def glibc_environment(environment, prefix):
    return {**environment, 'LD_DEBUG': 'files', 'LD_DEBUG_OUTPUT': str(prefix)}


def parse_glibc_maps(text):
    """Resolve every generated loader map from the same process's init log.

    These are diagnostics from the real executing glibc loader, not ldd or
    a prospective dependency listing. Unsupported/omitted observation refuses.
    """
    images, generated, programs = set(), set(), set()
    for line in text.splitlines():
        body = re.sub(r'^\s*\d+:\s*', '', line).strip()
        if 'generating link map' in body:
            match = re.fullmatch(r'file=(.+) \[\d+\];\s+generating link map', body)
            if not match:
                raise BuildEvidenceError('unparsed glibc generated map')
            generated.add(match.group(1))
        elif 'calling init:' in body:
            value = body.partition('calling init:')[2].strip()
            if not value.startswith('/'):
                raise BuildEvidenceError('unresolved glibc initialized image')
            images.add(value)
        elif 'transferring control:' in body:
            value = body.partition('transferring control:')[2].strip()
            if not value.startswith('/'):
                raise BuildEvidenceError('unresolved glibc program identity')
            programs.add(value)
    if (not images or not programs or
            not any('ld-linux' in Path(path).name for path in images)):
        raise BuildEvidenceError('missing actual glibc loader/image observation')
    for path in generated:
        if path.startswith('/'):
            resolved = path in images
        else:
            resolved = any(Path(image).name == path for image in images)
        if not resolved:
            raise BuildEvidenceError('unresolved actual glibc mapping: ' + path)
    return {'images': sorted(images), 'programs': sorted(programs)}


def capture_glibc(prefix):
    prefix = Path(prefix)
    paths = sorted(p for p in prefix.parent.glob(prefix.name + '.*')
                   if p.suffix[1:].isdigit())
    if not paths:
        raise BuildEvidenceError('missing actual glibc loader diagnostics')
    images, programs = {}, {}
    for path in paths:
        result = parse_glibc_maps(path.read_text(errors='strict'))
        for name in result['images']:
            identity = file_identity(name)
            images[identity['path']] = identity
        for name in result['programs']:
            identity = file_identity(name)
            programs[identity['path']] = identity
    return {'images': list(images.values()), 'programs': list(programs.values()),
            'evidence_files': [file_identity(path) for path in paths]}


def dyld_environment(environment):
    # The recorder does not forward candidate DYLD interposition/cache overrides.
    return {**{k: v for k, v in environment.items() if not k.startswith('DYLD_')},
            'DYLD_PRINT_LIBRARIES': '1'}


def parse_dyld_images(text, pid):
    images = []
    pattern = re.compile(r'dyld\[(\d+)\]: <([0-9A-Fa-f-]{36})> (/.*)')
    for line in text.splitlines():
        if not line.startswith('dyld['):
            continue
        match = pattern.fullmatch(line)
        if not match or int(match[1]) != pid:
            raise BuildEvidenceError('unparsed/unobserved Apple loader process')
        import uuid
        try:
            identifier = str(uuid.UUID(match[2]))
        except ValueError as error:
            raise BuildEvidenceError('invalid actual Apple image UUID') from error
        images.append({'path': match[3], 'uuid': identifier})
    if not images:
        raise BuildEvidenceError('missing actual Apple loader image observation')
    return images


def macho_uuids(data):
    """UUIDs of each supported Mach-O slice, checked against real dyld output."""
    import uuid
    if data[:4] in (b'\xca\xfe\xba\xbe', b'\xca\xfe\xba\xbf'):
        if len(data) < 8:
            raise BuildEvidenceError('invalid universal tool image')
        stride = 20 if data[3] == 0xbe else 32
        count = struct.unpack_from('>I', data, 4)[0]
        if count > 16 or 8 + count * stride > len(data):
            raise BuildEvidenceError('invalid universal tool image')
        result = set()
        for index in range(count):
            pos = 8 + index * stride
            offset, size = struct.unpack_from('>II' if stride == 20 else '>QQ',
                                              data, pos + 8)
            if offset + size > len(data):
                raise BuildEvidenceError('truncated universal tool image')
            result.update(macho_uuids(data[offset:offset + size]))
        return result
    if data[:4] != b'\xcf\xfa\xed\xfe' or len(data) < 32:
        raise BuildEvidenceError('unsupported Apple compiler/library image')
    count, size = struct.unpack_from('<II', data, 16)
    if count > 4096 or size + 32 > len(data):
        raise BuildEvidenceError('invalid Apple image command table')
    offset, result = 32, set()
    for _ in range(count):
        if offset + 8 > size + 32:
            raise BuildEvidenceError('truncated Apple image command table')
        command, length = struct.unpack_from('<II', data, offset)
        if length < 8 or offset + length > size + 32:
            raise BuildEvidenceError('truncated Apple image load command')
        if command == 0x1b:  # LC_UUID
            if length != 24:
                raise BuildEvidenceError('invalid Apple UUID command')
            result.add(str(uuid.UUID(bytes=data[offset + 8:offset + 24])))
        offset += length
    if not result:
        raise BuildEvidenceError('Apple compiler/library image has no UUID')
    return result


def apple_system_baseline(images):
    """Verify cache-only system images against the actual native OS cache.

    OS/loader correctness is a stated trust baseline. Cache images bind UUIDs
    and OS build, not invented individual file hashes. Compiler/Xcode libraries
    always require ordinary file-byte identities instead.
    """
    import ctypes as c
    import errno
    import platform
    import subprocess
    import uuid

    library = c.CDLL(None, use_errno=True)
    uuid_type = c.c_ubyte * 16
    cache_uuid = uuid_type()
    get_cache = library._dyld_get_shared_cache_uuid
    get_cache.argtypes, get_cache.restype = [c.POINTER(c.c_ubyte)], c.c_bool
    if not get_cache(cache_uuid):
        raise BuildEvidenceError('Apple system cache identity is unavailable')
    contains = library._dyld_shared_cache_contains_path
    contains.argtypes, contains.restype = [c.c_char_p], c.c_bool
    count = library._dyld_image_count
    count.argtypes, count.restype = [], c.c_uint32
    get_name = library._dyld_get_image_name
    get_name.argtypes, get_name.restype = [c.c_uint32], c.c_char_p
    get_header = library._dyld_get_image_header
    get_header.argtypes, get_header.restype = [c.c_uint32], c.c_void_p
    get_uuid = library._dyld_get_image_uuid
    get_uuid.argtypes, get_uuid.restype = [c.c_void_p, c.POINTER(c.c_ubyte)], c.c_bool
    arch = platform.machine()
    translated, size = c.c_int(0), c.c_size_t(c.sizeof(c.c_int))
    sysctl = library.sysctlbyname
    sysctl.argtypes = [
        c.c_char_p, c.c_void_p, c.POINTER(c.c_size_t), c.c_void_p, c.c_size_t,
    ]
    sysctl.restype = c.c_int
    result = sysctl(
        b'sysctl.proc_translated', c.byref(translated), c.byref(size), None, 0)
    # Apple's documented ENOENT case is a native Intel process, not an
    # observation failure. Other errors and Rosetta execution are unqualified.
    if ((result and c.get_errno() != errno.ENOENT) or translated.value
            or arch not in ('arm64', 'x86_64')):
        raise BuildEvidenceError(
            'Apple observer must use the native cache architecture')
    kept_alive = []
    for image in images:
        path = image['path']
        if (not path.startswith(('/usr/lib/', '/System/Library/'))
                or not contains(path.encode())):
            raise BuildEvidenceError(
                'unresolved non-system Apple loaded library: ' + path)
        # Resolve the same cache image in this native observer and compare its
        # actual in-memory UUID with the compiler process's loader diagnostic.
        kept_alive.append(c.CDLL(path))
        found = False
        for index in range(count()):
            name = get_name(index)
            if name and name.decode() == path:
                value = uuid_type()
                if get_uuid(get_header(index), value):
                    found = str(uuid.UUID(bytes=bytes(value))) == image['uuid']
                break
        if not found:
            raise BuildEvidenceError('Apple cache image UUID differs from actual child')
    queries = []
    for option in ('-productVersion', '-buildVersion'):
        run = subprocess.run(['/usr/bin/sw_vers', option], text=True,
                             capture_output=True)
        if run.returncode or not run.stdout.strip():
            raise BuildEvidenceError('Apple OS build provenance unavailable')
        queries.append({'argv': ['/usr/bin/sw_vers', option],
                        'stdout': run.stdout, 'stderr': run.stderr,
                        'exit_code': run.returncode})
    return {'schema': 'pyvoro2.apple-system-loader.v1',
            'cache_uuid': str(uuid.UUID(bytes=bytes(cache_uuid))),
            'architecture': arch, 'os_queries': queries, 'images': images,
            'identity_policy': 'trusted-native-os-cache-uuid-not-file-bytes'}


def capture_dyld(text, executable, pid):
    images = parse_dyld_images(text, pid)
    ordinary, cached = {}, []
    for image in images:
        path = Path(image['path'])
        if path.is_file():
            identity = file_identity(path)
            if image['uuid'] not in macho_uuids(path.read_bytes()):
                raise BuildEvidenceError('Apple loaded UUID differs from file bytes')
            ordinary[identity['path']] = identity
        else:
            cached.append(image)
    if str(Path(executable).resolve(strict=True)) not in ordinary:
        raise BuildEvidenceError('Apple loader did not identify the actual executable')
    if not cached:
        raise BuildEvidenceError('missing Apple system cache image observation')
    try:
        baseline = apple_system_baseline(cached)
    except (AttributeError, OSError) as error:
        raise BuildEvidenceError(
            'actual Apple cache observation unavailable') from error
    return {'images': list(ordinary.values()), 'system_loader': baseline}
