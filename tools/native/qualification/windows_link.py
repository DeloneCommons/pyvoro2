"""Closed MSVC library selection and SDK manifest resource evidence.

LINKREPROFULLPATHRSP lists explicit inputs, not DEFAULTLIB resolution. The
actual /VERBOSE:LIB stream supplies a conservative whole-searched-archive
closure. When LINK also emits member selections, those are reconciled too;
library-only diagnostics do not claim that individual members were observed.
The linker writes its actual manifest to a retained sidecar; a separately
observed SDK mt.exe embeds that XML without opaque rc/cvtres children.
"""
from __future__ import annotations

import hashlib
import json
import ntpath
from pathlib import Path
import re
import shutil
import struct
import xml.etree.ElementTree as ET

from .effective_build import BuildEvidenceError, file_identity, windows_words


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def archive_members(path):
    """Read COFF archive member identities and the first linker symbol index.

    Microsoft PE/COFF Archive (Library) File Format specifies the big-endian
    first linker member and NUL-terminated long-name table. Duplicate import
    member names are disambiguated by the actual resolved public symbol.
    """
    data = Path(path).read_bytes()
    if not data.startswith(b'!<arch>\n'):
        raise BuildEvidenceError('Windows searched input is not a COFF archive')
    offset, members, names, symbols = 8, {}, b'', None
    while offset < len(data):
        header = data[offset:offset + 60]
        if len(header) != 60 or header[58:] != b'`\n':
            raise BuildEvidenceError('invalid COFF archive member header')
        try:
            name = header[:16].decode('ascii').rstrip()
            size = int(header[48:58])
        except (ValueError, UnicodeError) as error:
            raise BuildEvidenceError('invalid COFF archive member fields') from error
        body = data[offset + 60:offset + 60 + size]
        if size < 0 or len(body) != size:
            raise BuildEvidenceError('truncated COFF archive member')
        if name == '/' and symbols is None:
            try:
                count = struct.unpack_from('>I', body)[0]
                if count > (len(body) - 4) // 4:
                    raise ValueError('symbol count')
                positions = struct.unpack_from('>' + 'I' * count, body, 4)
                words = body[4 + count * 4:].split(b'\0')
                if len(words) < count + 1:
                    raise ValueError('symbol names')
                symbols = {}
                for word, position in zip(words[:count], positions):
                    symbols.setdefault(word.decode('ascii'), []).append(position)
            except (ValueError, UnicodeError, struct.error) as error:
                raise BuildEvidenceError('invalid COFF archive symbol index') from error
        elif name == '//':
            names = body
        elif name != '/':
            members[offset] = {'name': name, 'offset': offset, 'size': size,
                               'sha256': _sha(body)}
        offset += 60 + size + size % 2
    if offset != len(data) or symbols is None:
        raise BuildEvidenceError('incomplete COFF archive symbol evidence')
    for member in members.values():
        name = member['name']
        if re.fullmatch(r'/\d+', name):
            begin = int(name[1:])
            end = names.find(b'\0', begin)
            if begin >= len(names) or end < begin:
                raise BuildEvidenceError('invalid COFF archive long member name')
            try:
                name = names[begin:end].decode('utf-8')
            except UnicodeError as error:
                raise BuildEvidenceError(
                    'unsupported COFF archive member encoding') from error
        else:
            name = name.removesuffix('/')
        member['name'] = name
    if any(position not in members for rows in symbols.values() for position in rows):
        raise BuildEvidenceError('COFF symbol points outside an archive member')
    return members, symbols


def collect_link_inputs(response, verbose):
    """Reconcile explicit paths, complete search blocks, and loaded members."""
    response, verbose = Path(response), Path(verbose)
    data = response.read_bytes()
    encoding = 'utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    names = windows_words(data.decode(encoding))
    if not names or any(not Path(name).is_absolute() for name in names):
        raise BuildEvidenceError('incomplete Windows explicit link input paths')
    identities = [file_identity(name) for name in names]
    explicit = {identity['path']: identity for identity in identities}
    libraries, selected, archives = {}, [], {}
    active, completed, current, found, references = False, 0, None, None, []
    for line in verbose.read_text().splitlines():
        word = line.strip()
        if not word:
            continue
        if word == 'Searching libraries':
            if active or found is not None:
                raise BuildEvidenceError('nested Windows library search report')
            active, current = True, None
        elif word == 'Finished searching libraries':
            if not active or found is not None:
                raise BuildEvidenceError('incomplete Windows library member selection')
            active, current, completed = False, None, completed + 1
        elif word.startswith('Searching ') and word.endswith(':') and active:
            if found is not None:
                raise BuildEvidenceError('missing Windows loaded archive member')
            path = Path(word[len('Searching '):-1])
            if not path.is_absolute():
                raise BuildEvidenceError('relative Windows library search result')
            current = file_identity(path)
            libraries[current['path']] = current
            if current['path'] not in archives:
                archives[current['path']] = archive_members(path)
        elif word.startswith('Found ') and active and current and found is None:
            found, references = word[len('Found '):], []
        elif word.startswith('Referenced in ') and found is not None:
            references.append(word[len('Referenced in '):])
        elif word.startswith('Loaded ') and active and current and found is not None:
            match = re.fullmatch(r'Loaded (.+)\(([^()]*)\)', word)
            if not match:
                raise BuildEvidenceError('unparsed Windows loaded archive member')
            library, member_name = match.groups()
            # LINK prints a basename for Loaded; only its immediately preceding
            # absolute Searching path can resolve that spelling.
            if ntpath.isabs(library):
                matches = file_identity(library) == current
            else:
                matches = (library.lower() == ntpath.basename(current['path']).lower())
            if not matches:
                raise BuildEvidenceError(
                    'Windows loaded archive path differs from search')
            members, symbols = archives[current['path']]
            candidates = [found]
            decorated = re.search(r'\(([^()]*)\)$', found)
            if decorated:
                candidates.append(decorated[1])
            offsets = {offset for symbol in candidates
                       for offset in symbols.get(symbol, [])}
            matches = [members[offset] for offset in offsets
                       if ntpath.basename(members[offset]['name']).lower() ==
                       ntpath.basename(member_name).lower()]
            if len(matches) != 1:
                raise BuildEvidenceError(
                    'unresolved/ambiguous Windows selected archive member')
            selected.append({'archive': current['path'], 'symbol': found,
                             'reported_member': member_name,
                             'referenced_by': references, 'member': matches[0]})
            found, references = None, []
        elif word.startswith(('Processed /DEFAULTLIB:', 'Processed /DISALLOWLIB:')):
            # These directives are part of the same trusted archive bytes;
            # the following actual search and selection still need evidence.
            if not active:
                raise BuildEvidenceError('Windows library directive outside search')
        elif (not active and re.fullmatch(r'Creating library .+ and object .+', word)):
            continue
        else:
            raise BuildEvidenceError(
                'unparsed Windows library selection output: ' + word)
    if active or found is not None or not completed or not libraries:
        raise BuildEvidenceError('incomplete Windows actual library selection report')
    if any(item['path'] not in libraries for item in explicit.values()
           if Path(item['path']).suffix.lower() == '.lib'):
        raise BuildEvidenceError(
            'Windows explicit archive missing from actual library search')
    # /VERBOSE:LIB on VS 18 reports complete searched-library passes without
    # Found/Loaded rows. Every byte of every searched archive remains in the
    # conservative closure; verify_link_closure separately requires its exact
    # identity to be independently approved by the installed runtime provider.
    # A partial member stanza still refuses above and cannot select this mode.
    mode = ('selected-members-and-searched-archives' if selected else
            'conservative-searched-archives')
    return {'schema': 'pyvoro2.windows-link-inputs.v2',
            'coverage': {'mode': mode, 'search_passes': completed},
            'explicit_response': file_identity(response),
            'verbose_log': file_identity(verbose),
            'explicit_inputs': sorted(explicit.values(), key=lambda row: row['path']),
            'searched_libraries': sorted(libraries.values(),
                                         key=lambda row: row['path']),
            'loaded_members': selected}


def _pe(path):
    data = Path(path).read_bytes()
    try:
        header = struct.unpack_from('<I', data, 60)[0]
        if data[:2] != b'MZ' or data[header:header + 4] != b'PE\0\0':
            raise ValueError('signature')
        machine, count = struct.unpack_from('<HH', data, header + 4)
        optional_size, characteristics = struct.unpack_from('<HH', data, header + 20)
        optional = header + 24
        if (machine != 0x8664 or optional_size < 136 or
                struct.unpack_from('<H', data, optional)[0] != 0x20b):
            raise ValueError('x64 optional header')
        if struct.unpack_from('<I', data, optional + 108)[0] < 3:
            raise ValueError('resource directory')
        resource_rva, resource_size = struct.unpack_from('<II', data, optional + 128)
        sections = []
        for index in range(count):
            start = optional + optional_size + 40 * index
            name = data[start:start + 8].rstrip(b'\0')
            size, address, raw_size, raw_offset = struct.unpack_from(
                '<IIII', data, start + 8)
            flags = struct.unpack_from('<I', data, start + 36)[0]
            body = data[raw_offset:raw_offset + raw_size]
            if len(body) != raw_size:
                raise ValueError('truncated section')
            sections.append({'name': name, 'address': address, 'size': size,
                             'raw_offset': raw_offset, 'raw_size': raw_size,
                             'flags': flags, 'bytes': body})
    except (ValueError, struct.error) as error:
        raise BuildEvidenceError(
            'invalid Windows PE evidence: ' + str(error)) from error
    return data, characteristics, sections, resource_rva, resource_size


def pe_manifest_resources(path):
    """Read actual numeric RT_MANIFEST resources without executing the PE."""
    data, _, sections, rva, size = _pe(path)
    if not rva and not size:
        return []

    def location(address, length):
        matches = [section['raw_offset'] + address - section['address']
                   for section in sections if section['address'] <= address and
                   address + length <= section['address'] + section['raw_size']]
        if len(matches) != 1:
            raise BuildEvidenceError('unmapped/ambiguous Windows PE resource bytes')
        return matches[0]

    base = location(rva, size)

    def unpack(fmt, offset):
        if offset < 0 or offset + struct.calcsize(fmt) > size:
            raise BuildEvidenceError('truncated Windows PE resource directory')
        return struct.unpack_from(fmt, data, base + offset)

    def entries(offset):
        named, numeric = unpack('<HH', offset + 12)
        return [unpack('<II', offset + 16 + index * 8)
                for index in range(named + numeric)]

    resources = []
    for kind, type_offset in entries(0):
        if kind != 24:  # RT_MANIFEST
            continue
        if not type_offset & 0x80000000:
            raise BuildEvidenceError('invalid Windows manifest resource type')
        for identifier, name_offset in entries(type_offset & 0x7fffffff):
            if identifier & 0x80000000 or not name_offset & 0x80000000:
                raise BuildEvidenceError('unsupported named Windows manifest resource')
            for language, value_offset in entries(name_offset & 0x7fffffff):
                if language & 0x80000000 or value_offset & 0x80000000:
                    raise BuildEvidenceError(
                        'invalid Windows manifest resource language')
                address, length, codepage, reserved = unpack('<IIII', value_offset)
                if reserved or not length:
                    raise BuildEvidenceError('empty/invalid Windows manifest resource')
                start = location(address, length)
                resources.append({'resource_id': identifier, 'language': language,
                                  'codepage': codepage,
                                  'bytes': data[start:start + length]})
    return resources


def verify_pe_manifest_transform(donor, output):
    """Permit mt to insert .rsrc and rebind unchanged base-relocation storage.

    PE base relocations address ImageBase + block PageRVA + entry offset;
    the .reloc section's storage RVA is not part of those target addresses.
    Only that storage section may move. All numerical/import/data section
    RVAs, payloads, and the relocation table bytes and target mappings remain
    identical. The base-relocation data directory must follow the table.
    """
    before, after = _pe(donor), _pe(output)

    def headers(parsed):
        data, _, sections, _, _ = parsed
        pe = struct.unpack_from('<I', data, 60)[0]
        coff = bytearray(data[pe:pe + 24])
        optional_size = struct.unpack_from('<H', coff, 20)[0]
        optional = bytearray(data[pe + 24:pe + 24 + optional_size])
        if optional_size < 160 or len(optional) != optional_size:
            raise BuildEvidenceError('Windows manifest lacks complete PE headers')
        directories = struct.unpack_from('<I', optional, 108)[0]
        if directories < 6 or 112 + directories * 8 > len(optional):
            raise BuildEvidenceError('Windows manifest has invalid PE data directories')
        section_alignment, file_alignment = struct.unpack_from('<II', optional, 32)
        if (file_alignment < 512 or file_alignment & (file_alignment - 1) or
                section_alignment < file_alignment or
                section_alignment & (section_alignment - 1)):
            raise BuildEvidenceError('Windows manifest has unsupported PE alignment')
        names = {section['name']: section for section in sections}
        if len(names) != len(sections):
            raise BuildEvidenceError('Windows manifest has ambiguous PE sections')
        raw_end = struct.unpack_from('<I', optional, 60)[0]
        if (raw_end % file_alignment or
                raw_end < pe + 24 + optional_size + 40 * len(sections)):
            raise BuildEvidenceError('Windows manifest has invalid PE header storage')
        for section in sorted(sections, key=lambda row: row['raw_offset']):
            if not section['raw_size']:  # Uninitialized sections use zero fill.
                continue
            if (section['raw_offset'] % file_alignment or
                    section['raw_size'] % file_alignment or
                    section['raw_offset'] < raw_end or
                    section['raw_offset'] + section['raw_size'] > len(data)):
                raise BuildEvidenceError(
                    'Windows manifest has unaligned/overlapping raw PE sections')
            raw_end = section['raw_offset'] + section['raw_size']
        header_size = struct.unpack_from('<I', optional, 60)[0]
        end = ((header_size + section_alignment - 1) // section_alignment *
               section_alignment)
        for section in sorted(sections, key=lambda row: row['address']):
            if section['address'] % section_alignment or section['address'] < end:
                raise BuildEvidenceError('Windows manifest has overlapping PE sections')
            end = section['address'] + max(section['size'], section['raw_size'])
        directory = struct.unpack_from('<II', optional, 152)
        original = bytes(optional)
        # mt recomputes storage size/checksum fields and the two directories.
        for offset, size in ((8, 4), (56, 12), (128, 8), (152, 4)):
            optional[offset:offset + size] = b'\0' * size
        coff[6:8] = b'\0\0'  # The new .rsrc adds exactly one section below.
        return names, original, bytes(coff), bytes(optional), directory, pe

    old, _, old_coff, old_control, old_reloc, _ = headers(before)
    new, new_header, new_coff, new_control, new_reloc, pe = headers(after)
    if (old_coff != new_coff or old_control != new_control or b'.rsrc' in old or
            set(new) != {*old, b'.rsrc'}):
        raise BuildEvidenceError(
            'Windows manifest changed PE controls or section inventory')
    for name, section in old.items():
        keys = ('name', 'size', 'flags', 'bytes')
        if name != b'.reloc':
            keys += ('address',)
        if any(section[key] != new[name][key] for key in keys):
            raise BuildEvidenceError(
                'Windows manifest changed non-resource PE payload/RVA')
    resource = new[b'.rsrc']
    if (resource['flags'] != 0x40000040 or after[3] != resource['address'] or
            not 0 < after[4] <= min(resource['size'], resource['raw_size'])):
        raise BuildEvidenceError('Windows manifest resource directory binding differs')

    def aligned(value, alignment):
        return (value + alignment - 1) // alignment * alignment

    section_alignment, file_alignment = struct.unpack_from('<II', new_header, 32)
    image_size = aligned(max(section['address'] +
                             max(section['size'], section['raw_size'])
                             for section in new.values()), section_alignment)
    header_size = aligned(pe + 24 + len(new_header) + 40 * len(new), file_alignment)
    initialized_size = sum(section['raw_size'] for section in new.values()
                           if section['flags'] & 0x40)
    if (struct.unpack_from('<II', new_header, 56) != (image_size, header_size) or
            struct.unpack_from('<I', new_header, 8)[0] != initialized_size or
            any(section['raw_size'] and section['raw_offset'] < header_size
                for section in new.values())):
        raise BuildEvidenceError('Windows manifest PE storage size binding differs')
    if b'.reloc' not in old:
        if old_reloc != (0, 0) or new_reloc != (0, 0):
            raise BuildEvidenceError(
                'Windows manifest has an unbound relocation directory')
        return
    relocation = old[b'.reloc']
    if (old_reloc[0] != relocation['address'] or
            new_reloc != (new[b'.reloc']['address'], old_reloc[1]) or
            not 0 < old_reloc[1] <= min(relocation['size'], relocation['raw_size']) or
            relocation['flags'] & 0xa0000020 or
            relocation['flags'] & 0x40000040 != 0x40000040):
        raise BuildEvidenceError(
            'Windows manifest base relocation directory binding differs')

    def targets(sections):
        table = sections[b'.reloc']['bytes'][:old_reloc[1]]
        extents = {name: (section['address'], section['address'] +
                          max(section['size'], section['raw_size']))
                   for name, section in sections.items()
                   if name not in (b'.reloc', b'.rsrc')}
        position, result = 0, []
        while position < len(table):
            if position + 8 > len(table):
                raise BuildEvidenceError(
                    'Windows manifest has truncated base relocations')
            page, size = struct.unpack_from('<II', table, position)
            if (page % 4096 or size < 8 or size % 4 or position + size > len(table)):
                raise BuildEvidenceError(
                    'Windows manifest has invalid base relocations')
            for offset in range(position + 8, position + size, 2):
                entry = struct.unpack_from('<H', table, offset)[0]
                kind, target = entry >> 12, page + (entry & 0xfff)
                if kind == 0:  # IMAGE_REL_BASED_ABSOLUTE is padding.
                    continue
                owners = [name for name, (begin, end) in extents.items()
                          if begin <= target and target + 8 <= end]
                if kind != 10 or len(owners) != 1:  # IMAGE_REL_BASED_DIR64.
                    raise BuildEvidenceError(
                        'Windows manifest changed relocation targets')
                result.append((target, owners[0]))
            position += size
        return result

    if targets(old) != targets(new):
        raise BuildEvidenceError('Windows manifest changed relocation target mapping')


def manifest_xml(data):
    """Compare XML metadata, allowing mt's insignificant serialization changes."""
    if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
        raise BuildEvidenceError('unsupported Windows manifest declaration')
    try:
        root = ET.fromstring(data)
        if root.tag != '{urn:schemas-microsoft-com:asm.v1}assembly' or not len(root):
            raise ValueError('missing assembly metadata')
        return ET.canonicalize(ET.tostring(root, encoding='unicode'),
                               strip_text=True, rewrite_prefixes=True).encode('utf-8')
    except (ET.ParseError, ValueError) as error:
        raise BuildEvidenceError('invalid/empty Windows manifest XML') from error


def manifest_tool(driver):
    from .input_provenance import windows_installation_roots
    installed = windows_installation_roots(driver)
    sdk = Path(installed['sdk'])
    candidates = sorted(path for path in (sdk / 'bin').glob('*/x64/mt.exe')
                        if (sdk / 'Include' / path.parents[1].name /
                            'ucrt' / 'corecrt.h').is_file())
    if not candidates:
        raise BuildEvidenceError('registered Windows SDK lacks an x64 manifest tool')
    return candidates[-1].resolve(strict=True), installed


def prepare_link(command, *, cwd, env, directory):
    """Add observer controls; never turn an opaque embed request into approval."""
    from .effective_build import expand_response
    words = [*windows_words(env.get('LINK', '')),
             *expand_response(command[1:], cwd, True),
             *windows_words(env.get('_LINK_', ''))]
    words = expand_response(words, cwd, True)
    if any(word.lower().lstrip('-/').startswith(
            ('manifest:', 'manifestfile:', 'linkrepro', 'verbose')) for word in words):
        raise BuildEvidenceError('unreviewed Windows manifest/link observer override')
    return [*command, '/NOLOGO', '/INCREMENTAL:NO', '/MANIFEST',
            '/MANIFESTFILE:' + str(directory / 'link.manifest'), '/VERBOSE:LIB',
            '/LINKREPROFULLPATHRSP:' + str(directory / 'actual-link-inputs.rsp')]


def embed_manifest(output, *, driver, cwd, env, directory, observer):
    """Observe mt's one-input XML edit and retain the pre-edit linked PE."""
    sidecar = directory / 'link.manifest'
    sidecar_identity = file_identity(sidecar)
    xml = manifest_xml(sidecar.read_bytes())
    resource_id = 2 if _pe(output)[1] & 0x2000 else 1  # IMAGE_FILE_DLL
    if pe_manifest_resources(output):
        raise BuildEvidenceError('Windows donor already contains a manifest resource')
    linked = file_identity(output)
    donor = directory / 'link-donor.pe'
    shutil.copyfile(output, donor)
    donor_identity = file_identity(donor)
    tool, installed = manifest_tool(driver)
    tool_identity = file_identity(tool)
    argv = [str(tool), '-nologo', '-manifest', str(sidecar),
            '-outputresource:' + str(output) + ';#' + str(resource_id)]
    mt_directory = directory / 'manifest-tool'
    mt_directory.mkdir()
    observed = observer(argv, cwd=cwd, env=env, directory=mt_directory)
    if observed['exit_code'] or observed['problems']:
        raise BuildEvidenceError('incomplete observed Windows manifest edit: ' +
                                 '; '.join(observed['problems']))
    resources = pe_manifest_resources(output)
    if (len(resources) != 1 or resources[0]['resource_id'] != resource_id or
            manifest_xml(resources[0]['bytes']) != xml):
        raise BuildEvidenceError(
            'Windows embedded manifest differs from actual linker XML')
    resource = resources[0]
    retained = directory / 'embedded.manifest'
    retained.write_bytes(resource['bytes'])
    if (file_identity(sidecar) != sidecar_identity or
            file_identity(tool) != tool_identity):
        raise BuildEvidenceError('Windows manifest tool/input changed during embedding')
    receipt = {'schema': 'pyvoro2.windows-manifest.v1',
               'linked_output': linked, 'donor': donor_identity,
               'sidecar': sidecar_identity, 'output': file_identity(output),
               'resource': {key: value for key, value in resource.items()
                            if key != 'bytes'},
               'embedded': file_identity(retained), 'xml_sha256': _sha(xml),
               'tool': tool_identity, 'installation': installed,
               'invocations': observed['invocations'],
               'loaded_images': observed.get('loaded_images', []),
               'evidence_files': [file_identity(p) for p in observed['evidence_paths']]}
    observed['evidence_paths'].extend([donor, sidecar, retained])
    return receipt, observed


def verify_windows_link(row):
    from .effective_build import _verify_identity
    report = row.get('windows_link_inputs')
    if not report or report.get('schema') != 'pyvoro2.windows-link-inputs.v2':
        raise BuildEvidenceError('missing Windows actual link input selection evidence')
    for key in ('explicit_response', 'verbose_log'):
        _verify_identity(report[key], 'Windows link input report')
    linkers = [item for item in row.get('invocations', []) if item['role'] == 'linker']
    if len(linkers) != 1:
        raise BuildEvidenceError('ambiguous Windows library selection invocation')
    argv = linkers[0]['expanded_argv']
    responses = [word.split(':', 1)[1] for word in argv
                 if word.lower().startswith('/linkreprofullpathrsp:')]
    files = {item['path']: item for item in row.get('evidence_files', [])}
    verbose = report['verbose_log']
    if (len(responses) != 1 or
            file_identity(Path(row['cwd']) / responses[0]) !=
            report['explicit_response'] or
            [word.lower() for word in argv].count('/verbose:lib') != 1 or
            row['environment'].get('VSLANG') != '1033' or
            Path(verbose['path']) != Path(report['explicit_response']['path']).parent /
            'stdout.txt' or any(files.get(report[key]['path']) != report[key]
                                for key in ('explicit_response', 'verbose_log'))):
        raise BuildEvidenceError(
            'Windows library selection detached from actual invocation')
    actual = collect_link_inputs(report['explicit_response']['path'],
                                 report['verbose_log']['path'])
    if actual != report:
        raise BuildEvidenceError('changed Windows actual link input selection')
    expected = {item['path']: item for key in ('explicit_inputs', 'searched_libraries')
                for item in report[key]}
    if expected != {item['path']: item for item in row.get('link_inputs', [])}:
        raise BuildEvidenceError('incomplete Windows actual link input inventory')
    verify_manifest(row)


def verify_manifest(row):
    from .effective_build import _verify_identity
    receipt = row.get('windows_manifest')
    if not receipt or receipt.get('schema') != 'pyvoro2.windows-manifest.v1':
        raise BuildEvidenceError('missing observed Windows manifest resource edit')
    for key in ('donor', 'sidecar', 'output', 'embedded', 'tool'):
        _verify_identity(receipt[key], 'Windows manifest ' + key)
    for item in (*receipt['loaded_images'], *receipt['evidence_files']):
        _verify_identity(item, 'Windows manifest observation')
    linked, donor = receipt['linked_output'], receipt['donor']
    if (receipt['output'] != row['output'] or linked['path'] != row['output']['path'] or
            any(linked[key] != donor[key] for key in ('sha256', 'size'))):
        raise BuildEvidenceError('Windows manifest donor/output lineage differs')
    linkers = [item for item in row['invocations'] if item['role'] == 'linker']
    if len(linkers) != 1:
        raise BuildEvidenceError('ambiguous Windows manifest linker')
    argv = linkers[0]['expanded_argv']
    sidecars = [word[len('/manifestfile:'):] for word in argv
                if word.lower().startswith('/manifestfile:')]
    if (len(sidecars) != 1 or
            file_identity(Path(row['cwd']) / sidecars[0]) != receipt['sidecar'] or
            '/manifest' not in [word.lower() for word in argv]):
        raise BuildEvidenceError(
            'Windows manifest sidecar is not an actual linker output')
    outputs = [word[5:] for word in argv if word.lower().startswith('/out:')]
    if (len(outputs) != 1 or
            str((Path(row['cwd']) / outputs[0]).resolve()) != linked['path']):
        raise BuildEvidenceError(
            'Windows manifest donor is not the actual linker output')
    resource_id = 2 if _pe(donor['path'])[1] & 0x2000 else 1
    if ('/dll' in [word.lower() for word in argv]) != (resource_id == 2):
        raise BuildEvidenceError(
            'Windows manifest resource does not match linked image kind')
    if pe_manifest_resources(donor['path']):
        raise BuildEvidenceError('Windows manifest donor contains an earlier resource')
    selected, installation = manifest_tool(linkers[0]['executable']['path'])
    if (file_identity(selected) != receipt['tool'] or
            installation != receipt['installation']):
        raise BuildEvidenceError('Windows manifest tool differs from registered SDK')
    invocations = receipt['invocations']
    expected_argv = [str(selected), '-nologo', '-manifest', receipt['sidecar']['path'],
                     '-outputresource:' + row['output']['path'] +
                     ';#' + str(resource_id)]
    if (len(invocations) != 1 or invocations[0].get('role') != 'manifest_tool' or
            invocations[0].get('exit_code') != 0 or
            invocations[0]['argv'] != expected_argv or
            invocations[0]['cwd'] != row['cwd'] or
            invocations[0]['environment'] != row['environment'] or
            file_identity(invocations[0]['executable']['path']) !=
            file_identity(receipt['tool']['path'])):
        raise BuildEvidenceError('incomplete Windows manifest process invocation')
    invocation = invocations[0]
    if ([item for item in row['invocations'] if item['role'] == 'manifest_tool'] !=
            invocations or invocation.get('observation') != 'direct-debug-execution'):
        raise BuildEvidenceError(
            'Windows manifest process detached from command record')
    evidence = {Path(item['path']).name: item for item in receipt['evidence_files']}
    if (set(evidence) != {'stdout.txt', 'stderr.txt', 'windows-debug-events.json'} or
            len(receipt['evidence_files']) != 3):
        raise BuildEvidenceError('incomplete Windows manifest process event evidence')
    events = json.loads(Path(evidence['windows-debug-events.json']['path']).read_text())
    creates = [event for event in events if event['event'] == 3]
    exits = [event for event in events if event['event'] == 5]
    if (len(creates) != 1 or creates[0]['pid'] != invocation['pid'] or
            creates[0]['image'] != receipt['tool'] or len(exits) != 1 or
            exits[0]['pid'] != invocation['pid'] or exits[0]['exit_code'] != 0 or
            any(event['pid'] != invocation['pid'] for event in events)):
        raise BuildEvidenceError(
            'Windows manifest has an unobserved child/process event')
    mapped = [event['image'] for event in events if event['event'] == 6]
    if (not mapped or mapped != receipt['loaded_images'] or
            mapped != invocation.get('loaded_images')):
        raise BuildEvidenceError('Windows manifest mapped image inventory differs')
    tool_inventory = {item['path']: item for item in row.get('tools', [])}
    if any(tool_inventory.get(item['path']) != item
           for item in [receipt['tool'], *mapped]):
        raise BuildEvidenceError(
            'Windows manifest tool/image missing from command tools')
    file_inventory = {item['path']: item for item in row.get('evidence_files', [])}
    if any(file_inventory.get(item['path']) != item for item in
           [receipt['donor'], receipt['sidecar'], receipt['embedded'],
            *evidence.values()]):
        raise BuildEvidenceError(
            'Windows manifest process files detached from command record')
    resources = pe_manifest_resources(row['output']['path'])
    xml = manifest_xml(Path(receipt['sidecar']['path']).read_bytes())
    if (len(resources) != 1 or resources[0]['resource_id'] != resource_id or
            {key: value for key, value in resources[0].items() if key != 'bytes'} !=
            receipt['resource'] or _sha(xml) != receipt['xml_sha256'] or
            resources[0]['bytes'] != Path(receipt['embedded']['path']).read_bytes() or
            manifest_xml(resources[0]['bytes']) != xml):
        raise BuildEvidenceError('Windows manifest resource equality failed')
    verify_pe_manifest_transform(donor['path'], row['output']['path'])
