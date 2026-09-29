"""Issue a detached record only after the controlled installed-payload suite.

This program consumes reviewed source approval, verified observed build records,
and receipts from the controlled repair/install workflow. It executes the fixed
repository route runner itself; it does not import a caller's success report.
The runner may exercise candidates in its isolated process before issuance.
Only after it completes does this program write the production installation's
record and trust anchor. A subsequent unpatched installed smoke remains a
distribution gate. Arbitrary replacement of this trusted workflow is outside
the numerical/build qualification boundary.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import subprocess
import sys
import tempfile
from types import SimpleNamespace


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification.effective_build import (
    BuildEvidenceError, digest as build_digest, file_identity, verify_build,
)
from qualification.source_policy import _contract, check_approval, measure_source


class FinalizationError(RuntimeError):
    """Incomplete or mismatched authority cannot issue an installation anchor."""


SANITIZER_RUNTIME_KEYS = ('LD_PRELOAD', 'ASAN_OPTIONS', 'UBSAN_OPTIONS',
                          'PYVORO2_NATIVE_TEST_SANITIZERS')


def controlled_environment():
    """Inherited interpreter controls cannot weaken the fixed acceptance suite."""
    environment = {name: value for name, value in os.environ.items()
                   if name not in SANITIZER_RUNTIME_KEYS
                   and not name.startswith(('PYTHON', 'PYTEST_'))}
    environment.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    return environment


def _require(condition, detail):
    if not condition:
        raise FinalizationError(detail)


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def _relative(root, name):
    _require(isinstance(name, str) and '\\' not in name,
             'invalid installed artifact path')
    part = PurePosixPath(name)
    _require(not part.is_absolute() and '..' not in part.parts
             and part.as_posix() == name, 'noncanonical installed artifact path')
    path = (root / name).resolve()
    _require(path.is_relative_to(root), 'installed artifact escapes installation')
    return path


def _check_file(identity, *, root=None):
    _require(isinstance(identity, dict), 'missing file identity in evidence')
    try:
        path = (_relative(root, identity['path']) if root is not None
                else Path(identity['path']).resolve(strict=True))
        actual = file_identity(path)
    except (KeyError, OSError, TypeError) as exc:
        raise FinalizationError('unavailable evidence file identity') from exc
    _require(actual['sha256'] == identity.get('sha256')
             and actual['size'] == identity.get('size'),
             f'evidence file changed: {path.name}')
    return actual


def _check_repair_observation(operation):
    observation = operation.get('observation')
    _require(isinstance(observation, dict)
             and observation.get('schema') == 'pyvoro2-native-repair-observation-v1'
             and observation.get('kind') == operation['kind'],
             'repair implementation/helper observation is missing')
    retained = _check_file(operation.get('observation_file'))
    data = json.loads(Path(retained['path']).read_text(encoding='utf8'))
    _require(data == observation,
             'retained repair implementation observation differs')
    distribution = observation.get('distribution')
    _require(isinstance(distribution, dict)
             and distribution.get('name') == operation['kind']
             and isinstance(distribution.get('version'), str)
             and isinstance(distribution.get('entry_point'), str)
             and isinstance(distribution.get('files'), list) and distribution['files'],
             'repair distribution implementation identity is incomplete')
    loaded = observation.get('loaded_modules')
    _require(isinstance(loaded, list) and loaded,
             'loaded repair implementation identities are absent')
    for item in distribution['files'] + loaded:
        _check_file(item)
    children = observation.get('processes')
    _require(isinstance(children, list), 'repair helper process evidence is absent')
    for child in children:
        _require(isinstance(child, dict)
                 and isinstance(child.get('argv'), list) and child['argv']
                 and all(isinstance(arg, str) for arg in child['argv'])
                 and isinstance(child.get('cwd'), str)
                 and isinstance(child.get('environment'), dict)
                 and _digest(child.get('environment_sha256'))
                 and type(child.get('exit_code')) is int,
                 'repair helper process evidence is incomplete')
        _check_file(child.get('executable'))
        for stream in ('stdout', 'stderr'):
            mode = child.get(stream + '_mode')
            _require(mode in ('pipe', 'inherited', 'stdout', 'discarded'),
                     'repair helper has an opaque stream')
            if mode == 'pipe':
                _check_file(child.get(stream))
        responses = child.get('responses')
        _require(isinstance(responses, list), 'repair response-file evidence is absent')
        for response in responses:
            snapshot = _check_file(response.get('snapshot'))
            data = base64.b64decode(response['bytes_base64'], validate=True)
            _require(hashlib.sha256(data).hexdigest()
                     == snapshot['sha256'], 'repair response-file bytes differ')


def verify_postprocess(receipt, build, installation_root):
    """Bind every observed link output through explicit repair to installed bytes."""
    installation_root = Path(installation_root).resolve()
    _require(receipt.get('schema') == 'pyvoro2-native-postprocess-v1',
             'missing or unknown postprocess receipt')
    modules = receipt.get('modules')
    dependencies = receipt.get('dependencies')
    operations = receipt.get('operations')
    _require(isinstance(modules, dict) and modules
             and isinstance(dependencies, dict)
             and isinstance(operations, list) and operations,
             'incomplete postprocess receipt')
    linked = build.get('components', {})
    reachable, outputs, module_claims, dependency_claims = set(), set(), {}, {}
    referenced_dependencies = set()
    for name, item in modules.items():
        _require(name in ('pyvoro2._core', 'pyvoro2._core2d', 'pyvoro2._fpguard')
                 and isinstance(item, dict), 'unknown postprocess module')
        expected = linked.get(name.split('.')[-1], {}).get('output')
        _require(expected is not None and item.get('input') == expected,
                 'postprocess input is not the observed link output')
        _check_file(item['input'])
        after = _check_file(item.get('output'), root=installation_root)
        names = item.get('dependencies')
        _require(isinstance(names, list)
                 and all(isinstance(path, str) for path in names)
                 and len(set(names)) == len(names), 'missing native dependency list')
        referenced_dependencies.update(names)
        reachable.add(expected['sha256'])
        outputs.add(after['sha256'])
        module_claims[name] = {'path': item['output']['path'],
                               'sha256': after['sha256'], 'dependencies': names}
    _require(set(linked) == {name.split('.')[-1] for name in modules},
             'postprocess receipt omits an observed native module')
    _require(referenced_dependencies == set(dependencies),
             'postprocess native dependency inventory differs')
    for name, item in dependencies.items():
        _require(isinstance(item, dict), 'missing native dependency identity')
        after = _check_file({**item, 'path': name}, root=installation_root)
        dependency_claims[name] = {'sha256': after['sha256']}
        outputs.add(after['sha256'])
    touched = set()
    for operation in operations:
        _require(isinstance(operation, dict), 'invalid postprocess operation')
        inputs, results = operation.get('inputs'), operation.get('outputs')
        _require(isinstance(inputs, list) and inputs
                 and isinstance(results, list) and results
                 and all(_digest(item) for item in inputs + results),
                 'missing postprocess operation input/output identities')
        _require(set(inputs) <= reachable,
                 'postprocess operation has an unobserved input')
        kind = operation.get('kind')
        if kind == 'identity':
            _require(set(inputs) == set(results),
                     'identity postprocess operation changes native bytes')
        else:
            _require(kind in ('auditwheel', 'delocate', 'delvewheel', 'strip', 'copy'),
                     'unreviewed native postprocess operation')
            argv = operation.get('argv')
            _require(isinstance(argv, list) and argv
                     and all(isinstance(arg, str) for arg in argv)
                     and isinstance(operation.get('cwd'), str)
                     and operation.get('exit_code') == 0,
                     'incomplete native postprocess process receipt')
            for key in ('executable', 'stdout', 'stderr'):
                _check_file(operation.get(key))
            _check_repair_observation(operation)
        reachable.update(results)
        touched.update(results)
    _require(outputs <= touched,
             'installed native payload lacks complete postprocess lineage')
    native_inventory = set()
    for base in (installation_root / 'pyvoro2',
                 installation_root / 'pyvoro2.libs'):
        for path in base.rglob('*'):
            if path.is_file() and ('.so' in path.suffixes
                                   or path.suffix in ('.dll', '.dylib', '.pyd')):
                native_inventory.add(path.relative_to(installation_root).as_posix())
    expected_inventory = {item['path'] for item in module_claims.values()}
    expected_inventory.update(dependency_claims)
    _require(native_inventory == expected_inventory,
             'installed native files differ from postprocess inventory')
    return module_claims, dependency_claims


def validate_routes(report, measurement, modules, required, *, family=None):
    """Validate a report from our own fixed child, never a submitted success bit."""
    _require(report.get('schema') == 'pyvoro2-native-route-evidence-v1',
             'missing or unknown route evidence schema')
    _validate_route_claims(report, measurement, modules, required)
    _validate_arithmetic(report, family=family)
    guard = report['discriminators'].get('guard_disassembly')
    _require(isinstance(guard, dict)
             and _digest(guard.get('tool_sha256'))
             and _digest(guard.get('disassembly_sha256'))
             and isinstance(guard.get('inspect_symbols'), list)
             and guard['inspect_symbols']
             and isinstance(guard.get('dispatch_symbols'), list)
             and guard['dispatch_symbols']
             and isinstance(guard.get('require_environment_symbols'), list)
             and guard['require_environment_symbols']
             and isinstance(guard.get('objects'), list) and len(guard['objects']) == 5
             and guard.get('fp_before_guard') is False,
             'optimized raw guard/dispatch evidence is incomplete')


def _validate_route_claims(report, measurement, modules, required):
    _require(report.get('source_sha256') == measurement.get('source_sha256'),
             'route evidence belongs to another source closure')
    expected_modules = {name: item['sha256'] for name, item in modules.items()}
    _require(report.get('modules') == expected_modules,
             'route evidence belongs to another native payload')
    claims = report.get('components')
    _require(isinstance(claims, dict) and set(claims) == set(required),
             'incomplete route component evidence')
    expected_corpora = {
        'wp6-planar': {'current': {'cases': 48, 'occurrences': 1718},
                       'archived': {'cases': 92, 'occurrences': 2298}},
        'wp7-planar': {'selected': {'cases': 30, 'occurrences': 287}},
    }
    for name, claim in claims.items():
        _require(isinstance(claim, dict) and claim.get('passed') is True,
                 f'route evidence failed: {name}')
        tests = claim.get('tests')
        _require(isinstance(tests, list) and tests
                 and all(isinstance(row, dict)
                         and isinstance(row.get('nodeid'), str) and row['nodeid']
                         and row.get('outcome') in ('passed', 'skipped')
                         for row in tests)
                 and any(row['outcome'] == 'passed' for row in tests),
                 f'route tests are incomplete or failed: {name}')
        for corpus, counts in expected_corpora.get(name, {}).items():
            _require(claim.get('corpora', {}).get(corpus) == counts,
                     f'required route corpus coverage differs: {name}/{corpus}')


def _validate_arithmetic(report, *, family=None):
    discriminators = report.get('discriminators')
    _require(isinstance(discriminators, dict), 'missing independent discriminators')
    expected = {
        'standard-xy-v1': ['0x1.ffffff4p-1', '0x1.ffffffep0', '0x0p0'],
        'standard-xz-v1': ['0x1.ffffff4p-1', '0x0p0', '0x1.ffffffep0'],
    }
    fixtures = discriminators.get('standard_fixtures')
    _require(isinstance(fixtures, list) and len(fixtures) == len(expected)
             and all(isinstance(row, dict) for row in fixtures)
             and [row.get('name') for row in fixtures] == list(expected),
             'standard fixture inventory differs')
    discriminating = []
    for row in fixtures:
        _require(row.get('coordinates_hex') == expected[row['name']]
                 and row.get('expected_strict_bits') == '4013fffffb000000'
                 and row.get('expected_fused_bits') == '4013fffffb000001'
                 and row.get('strict_bits') == ['4013fffffb000000'] * 2
                 and row.get('unsafe_bits') in (['4013fffffb000000'] * 2,
                                                ['4013fffffb000001'] * 2),
                 'standard fixture expectation or actual observation differs')
        if row['unsafe_bits'] == ['4013fffffb000001'] * 2:
            discriminating.append(row['name'])
    claimed = discriminators.get('standard_discriminating_fixtures')
    historical_unsafe = discriminators.get('standard_unsafe_bits')
    _require(discriminating
             and claimed == discriminating
             and (family != 'GNU' or 'standard-xy-v1' in discriminating)
             and discriminators.get('standard_strict_bits') == '4013fffffb000000'
             and historical_unsafe == fixtures[0]['unsafe_bits'][0],
             'standard operation/contraction control did not discriminate')
    radii = ('0x1.0000000000000p+27', '0x1.0000002000000p+27',
             '0x1.0000004000000p+27')
    expected_power = [{'radius_hex': radius, 'r_scale_bits': '0000000000000000',
                       'r_scale_check_bits': '3ff0000000000000'} for radius in radii]
    _require(discriminators.get('power_offsets') == expected_power,
             'independent power operation-order discriminator differs')
    unsafe_power = [{**row, 'r_scale_bits': '3ff0000000000000'}
                    for row in expected_power]
    _require(discriminators.get('power_unsafe_offsets') == unsafe_power,
             'unsafe power operation-order control did not discriminate')


def validate_safety_routes(report, measurement, modules, required, *, family=None):
    from qualification.route_suite import SANITIZER_TESTS
    _require(report.get('schema') == 'pyvoro2-native-sanitizer-routes-v1'
             and report.get('scope') == 'sanitizer-safety-only',
             'missing or unknown sanitizer safety evidence schema')
    _validate_route_claims(report, measurement, modules, required)
    _validate_arithmetic(report, family=family)
    _require('guard_disassembly' not in report['discriminators'],
             'sanitizer safety evidence cannot claim optimized guard qualification')
    tests = report.get('safety_tests')
    _require(isinstance(tests, list) and tests
             and all(isinstance(row, dict) and isinstance(row.get('nodeid'), str)
                     and row.get('outcome') in ('passed', 'skipped') for row in tests),
             'sanitizer safety tests are incomplete or failed')
    _require({row['nodeid'].split('::')[0] for row in tests} == set(SANITIZER_TESTS)
             and len({row['nodeid'] for row in tests}) == len(tests)
             and all(any(row['nodeid'].split('::')[0] == name
                         and row['outcome'] == 'passed' for row in tests)
                     for name in SANITIZER_TESTS),
             'required sanitizer safety test coverage differs')


def _check_guard_objects(report, build):
    from qualification.discriminators import guard_summary, inspect_guard_symbols
    from qualification.guard_symbols import PROTECTED_ENTRY_COUNTS, parse_disassembly
    guard = report['discriminators']['guard_disassembly']
    units = {row['output']['path']: row
             for row in build['translation_units']
             if Path(row['source']).name in PROTECTED_ENTRY_COUNTS}
    expected = {path: row['output'] for path, row in units.items()}
    _require(len(units) == len(PROTECTED_ENTRY_COUNTS) and
             {Path(row['source']).name for row in units.values()} ==
             set(PROTECTED_ENTRY_COUNTS), 'guard object inventory is incomplete')
    actual, inspections = {}, []
    tool = None
    for item in guard['objects']:
        obj = _check_file(item.get('object'))
        _check_file(item.get('disassembly'))
        _check_file(item.get('raw_disassembly'))
        argv = item.get('argv')
        _require(isinstance(argv, list) and argv
                 and argv == [argv[0], '-drC', '--no-show-raw-insn', obj['path']]
                 and item.get('raw_argv') ==
                 [argv[0], '-dr', '--no-show-raw-insn', obj['path']]
                 and item.get('exit_code') == item.get('raw_exit_code') == 0,
                 'guard disassembler process evidence is incomplete')
        current_tool = file_identity(argv[0])
        _require(current_tool['sha256'] == guard['tool_sha256']
                 and (tool is None or current_tool == tool),
                 'guard disassembler identity differs')
        tool = current_tool
        _require(obj['path'] in units and obj['path'] not in actual,
                 'unknown or duplicate guard production object')
        name = Path(units[obj['path']]['source']).name
        _require(item.get('source') == name, 'guard source association differs')
        try:
            inspection = inspect_guard_symbols(
                parse_disassembly(Path(item['disassembly']['path']).read_text(),
                                  Path(item['raw_disassembly']['path']).read_text()),
                require_dispatch=name != 'fpguard.cpp', source_name=name)
        except BuildEvidenceError as exc:
            raise FinalizationError(
                'guard coverage replay failed: ' + str(exc)) from exc
        _require(item.get('inspection') == inspection,
                 'guard protected-entry inventory differs from disassembly')
        inspections.append(inspection)
        actual[obj['path']] = obj
    _require(len(expected) == 5 and actual == expected,
             'guard evidence belongs to different production objects')
    _require(build_digest({'tool': tool, 'objects': guard['objects']})
             == guard['disassembly_sha256'], 'guard disassembly manifest differs')
    _require(all(guard.get(key) == value
                 for key, value in guard_summary(inspections).items()),
             'guard aggregate boundary coverage differs')


def _failed_route_output_tail(path, environment):
    """Match the driver's bounded diagnostics without exporting credentials."""
    sensitive = re.compile(
        r'TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTHORIZATION|(?:^|_)KEY(?:$|_)',
        re.I)
    secrets = {value.encode('utf8') for name, value in environment.items()
               if value and sensitive.search(name)}
    overlap = max((len(value) for value in secrets), default=1)
    limit = 12 * 1024
    with Path(path).open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        display_offset = max(0, stream.tell() - limit)
        offset = max(0, display_offset - overlap)
        stream.seek(offset)
        data = stream.read()
    start = display_offset - offset
    if display_offset and data[start - 1:start] != b'\n':
        # An incomplete line can contain URL userinfo whose scheme was cut off.
        newline = data.find(b'\n', start)
        start = len(data) if newline < 0 else newline + 1
    # Choose the original window before any length-changing replacement. Extra
    # context locates complete multiline secrets but can never become output.
    ranges = [match.span(1) for match in re.finditer(
        rb'https?://([^/\s@]+)@', data)]
    for value in secrets:
        position = data.find(value)
        while position >= 0:
            ranges.append((position, position + len(value)))
            position = data.find(value, position + 1)
    parts, cursor = [], start
    for begin, end in sorted(ranges):
        if end > cursor:
            parts.extend((data[cursor:max(cursor, begin)], b'[redacted]'))
            cursor = end
    parts.append(data[cursor:])
    text = b''.join(parts).decode('utf8', errors='replace')
    text = '\n'.join(text.splitlines()[-80:])
    return text.encode('utf8')[-limit:].decode('utf8', errors='ignore')


def run_routes(*, source_root, installation_root, corpus, output,
               build_evidence_path, components, candidate_module=None,
               candidate_build_path=None, runtime_environment=None,
               safety_only=False):
    """Execute only the repository-owned runner, with no caller command option."""
    _require(__debug__ and sys.flags.optimize == 0,
             'optimized Python cannot issue route qualification')
    runner = source_root / 'tools/native/qualification/route_suite.py'
    _require(runner.is_file(), 'controlled route runner is unavailable')
    report_path = output / 'route-evidence.json'
    _require(not report_path.exists(), 'route report output must be fresh')
    command = [sys.executable, str(runner), '--source-root', str(source_root),
               '--installation-root', str(installation_root),
               '--corpus', str(corpus), '--output', str(report_path),
               '--build-evidence', str(build_evidence_path),
               '--components', ','.join(sorted(components))]
    if safety_only:
        command.extend(('--mode', 'sanitizer-safety'))
    if candidate_module is not None:
        command.extend(('--candidate-module', str(candidate_module)))
        command.extend(('--candidate-build-evidence', str(candidate_build_path)))
    _require(set(runtime_environment or {}) <= set(SANITIZER_RUNTIME_KEYS),
             'unreviewed qualification runtime environment override')
    environment = controlled_environment()
    environment.update(runtime_environment or {})
    environment['PYTHONPATH'] = str(installation_root)
    with (output / 'route.stdout').open('wb') as stdout:
        with (output / 'route.stderr').open('wb') as stderr:
            process = subprocess.run(command, cwd=output, env=environment,
                                     stdout=stdout, stderr=stderr, check=False)
    if process.returncode != 0:
        details = [f'controlled route runner failed ({process.returncode}); '
                   f'retained logs: {output}']
        for name in ('route.stdout', 'route.stderr'):
            tail = _failed_route_output_tail(output / name, environment)
            if tail:
                details.append(f'{name} (last 80 lines, at most 12 KiB):\n{tail}')
        raise FinalizationError('\n'.join(details))
    _require(report_path.is_file(), 'controlled runner did not emit route evidence')
    try:
        report = json.loads(report_path.read_text(encoding='utf8'))
    except (ValueError, OSError) as exc:
        raise FinalizationError('invalid controlled route report') from exc
    _require(isinstance(report, dict), 'invalid controlled route report')
    receipt = {'argv': command, 'cwd': str(output), 'exit_code': process.returncode,
               'runner': file_identity(runner),
               'interpreter': file_identity(sys.executable),
               'stdout': file_identity(output / 'route.stdout'),
               'stderr': file_identity(output / 'route.stderr'),
               'report': file_identity(report_path),
               'runtime_environment': {key: environment[key]
                                       for key in SANITIZER_RUNTIME_KEYS
                                       if key in environment}}
    return report, receipt


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _check_source_inputs(build, measurement, source_root):
    """Every source input is approved or independently provided by the toolchain.

    ``verify_build`` establishes external provider provenance using fixed queries
    and installed metadata. A captured candidate include path cannot establish a
    provider, and a toolchain header provider never authorizes a primary TU.
    """
    source_root = Path(source_root).resolve()
    dependencies = {}
    for item in build['dependencies']:
        actual = _check_file(item)
        _require(actual == item and actual['path'] not in dependencies,
                 'ambiguous effective source dependency identity')
        dependencies[actual['path']] = actual
    units = build.get('translation_units')
    _require(isinstance(units, list) and units,
             'effective primary translation unit inventory is missing')
    for unit in units:
        path = Path(unit['source']).resolve()
        _require(path.is_relative_to(source_root),
                 'primary translation unit is outside approved source: ' + str(path))
        name = path.relative_to(source_root).as_posix()
        identity = dependencies.get(str(path))
        _require(identity is not None
                 and measurement['files'].get(name) == identity['sha256']
                 and identity in unit.get('dependencies', []),
                 'primary translation unit lacks approved dependency identity: ' + name)
    providers = {}
    allowed_kinds = ('compiler_builtin_header', 'python_header',
                     'pybind11_distribution')
    for provider in build.get('input_providers', []):
        _require(isinstance(provider, dict)
                 and _digest(provider.get('provider_id'))
                 and provider.get('kind') in allowed_kinds
                 and provider['provider_id'] not in providers,
                 'external input provider provenance is incomplete or ambiguous')
        providers[provider['provider_id']] = provider
    external = {}
    for claim in build.get('external_inputs', []):
        _require(isinstance(claim, dict), 'external input provenance is incomplete')
        provider = providers.get(claim.get('provider_id'))
        _require(provider is not None
                 and claim.get('provenance_kind') == provider['kind'],
                 'external input has no verified provider provenance')
        identity = _check_file(claim.get('identity'))
        _require(dependencies.get(identity['path']) == identity
                 and identity['path'] not in external,
                 'external input is not an exact observed dependency')
        external[identity['path']] = identity
    for name, item in dependencies.items():
        path = Path(name)
        if path.is_relative_to(source_root):
            key = path.relative_to(source_root).as_posix()
            _require(measurement['files'].get(key) == item['sha256'],
                     'effective dependency omitted from source closure: ' + key)
        else:
            _require(external.get(name) == item,
                     'unapproved external source dependency: ' + name)


def _require_default_anchor(source_root, installation_root):
    """Safety artifacts cannot contain an issued anchor or adjacent record."""
    internal = installation_root / 'pyvoro2/_internal'
    path = internal / '_qualification_installation.py'
    data = path.read_bytes()
    _require(data == (source_root / 'src/pyvoro2/_internal/'
                      '_qualification_installation.py').read_bytes(),
             'sanitizer safety installation anchor differs from the default')
    values = {}
    try:
        for node in ast.parse(data).body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                _require(isinstance(node.value.value, str), 'invalid default anchor')
            else:
                _require(isinstance(node, ast.Assign) and len(node.targets) == 1
                         and isinstance(node.targets[0], ast.Name),
                         'sanitizer safety anchor contains executable code')
                name = node.targets[0].id
                _require(name not in values, 'duplicate default anchor field')
                values[name] = ast.literal_eval(node.value)
    except (SyntaxError, ValueError) as exc:
        raise FinalizationError('invalid sanitizer safety anchor') from exc
    _require(values == {'RECORD_FILENAME': 'native_qualification_record.json',
                        'RECORD_SHA256': None, 'INSTALLATION_ID': None}
             and not (internal / 'native_qualification_record.json').exists(),
             'sanitizer safety artifact must remain unqualified')
    return data


def _require_sanitizers(build, records_dir):
    _require(build.get('adapter') == 'gnu-linux-x86_64-v1',
             'sanitizer safety has no reviewed adapter on this target')
    units = build.get('translation_units')
    _require(isinstance(units, list) and units,
             'sanitizer safety requires observed instrumented translation units')
    records = {}
    for path in Path(records_dir).glob('*.json'):
        row = json.loads(path.read_text(encoding='utf8'))
        expected = row.pop('record_sha256', None)
        _require(expected == build_digest(row),
                 'sanitizer command record changed after build verification')
        records[expected] = row
    for unit in units:
        row = records.get(unit.get('record_sha256'), {})
        _require(row.get('kind') == 'compile' and row.get('source') == unit['source']
                 and row.get('output') == unit['output']
                 and row.get('effective_argv') == unit.get('command', {}).get('argv'),
                 'sanitizer instrumentation record differs from observed build')
        backends = [item.get('expanded_argv', []) for item in row.get('invocations', [])
                    if item.get('role') == 'compiler_backend']
        _require(backends, 'sanitizer instrumentation lacks an observed backend')
        for argv in [row['effective_argv'], *backends]:
            enabled = {value for arg in argv if arg.startswith('-fsanitize=')
                       for value in arg.partition('=')[2].split(',')}
            _require({'address', 'undefined', 'float-cast-overflow'} <= enabled
                     and not any(arg.startswith(('-fno-sanitize=',
                                                 '-fsanitize-ignorelist',
                                                 '-fsanitize-blacklist'))
                                 for arg in argv),
                     'sanitizer instrumentation is missing or disabled: '
                     + unit['source'])


def _candidate_evidence(*, source_root, installation_root, records_dir,
                        postprocess_path, corpus, output, candidate_module=None,
                        candidate_records=None, runtime_environment=None,
                        safety_only=False):
    """Read-only source/build/payload gate shared by the two fixed evidence paths."""
    _require(__debug__ and sys.flags.optimize == 0,
             'optimized Python cannot issue native qualification')
    source_root = Path(source_root).resolve()
    installation_root = Path(installation_root).resolve()
    output = Path(output).resolve()
    contract = _contract(source_root)
    measurement = measure_source(source_root)
    approval_path = source_root / 'src/pyvoro2/_internal/native_approval.json'
    approval = json.loads(approval_path.read_text(encoding='utf8'))
    check_approval(measurement, approval)
    _require(not output.exists(), 'finalization evidence directory must be fresh')
    output.mkdir(parents=True)
    build = verify_build(Path(records_dir), source_root)
    _check_source_inputs(build, measurement, source_root)
    anchor = None
    if safety_only:
        anchor = _require_default_anchor(source_root, installation_root)
        _require_sanitizers(build, records_dir)
    toolchain = build.get('toolchain')
    _require(isinstance(toolchain, dict)
             and isinstance(toolchain.get('version'), str) and toolchain['version']
             and _digest(toolchain.get('compiler_sha256'))
             and _digest(toolchain.get('linker_sha256')),
             'missing verified toolchain provenance')
    target = contract.current_target()
    adapter = next((name for name, item in contract.ADAPTERS.items()
                    if item[:3] == (target['platform'], target['machine'],
                                    toolchain.get('family'))), None)
    _require(adapter is not None, 'no reviewed target/toolchain adapter')
    _require(build.get('adapter') == adapter,
             'effective build adapter differs from installation target')
    required = contract.ADAPTERS[adapter][3]
    build_data = contract.canonical_json(build)
    build_path = output / 'effective-build.json'
    _atomic_write(build_path, build_data)
    candidate_build = None
    candidate_build_path = None
    candidate_identity = None
    if {'wp6-planar', 'wp7-planar'} & required:
        _require(candidate_module is not None and candidate_records is not None,
                 'planar occurrence evidence requires verified candidate build records')
    if candidate_module is not None or candidate_records is not None:
        _require(candidate_module is not None and candidate_records is not None,
                 'candidate module and build records must be supplied together')
        candidate_module = Path(candidate_module).resolve(strict=True)
        candidate_build = verify_build(Path(candidate_records), source_root)
        _check_source_inputs(candidate_build, measurement, source_root)
        if safety_only:
            _require_sanitizers(candidate_build, candidate_records)
        _require(candidate_build.get('toolchain') == toolchain
                 and candidate_build.get('adapter') == adapter
                 and candidate_build.get('properties') == build.get('properties'),
                 'candidate and production effective build properties differ')
        candidate_identity = file_identity(candidate_module)
        _require(candidate_build['components'].get('_core2d', {}).get('output')
                 == candidate_identity,
                 'candidate module is not the exact observed candidate link output')
        candidate_build_path = output / 'candidate-effective-build.json'
        _atomic_write(candidate_build_path, contract.canonical_json(candidate_build))
    try:
        postprocess = json.loads(Path(postprocess_path).read_text(encoding='utf8'))
    except (ValueError, OSError) as exc:
        raise FinalizationError('missing or invalid postprocess receipt') from exc
    modules, dependencies = verify_postprocess(postprocess, build, installation_root)
    required_modules = {'pyvoro2._core2d' if name.endswith('-planar')
                        else 'pyvoro2._core' for name in required}
    required_modules.add('pyvoro2._fpguard')
    _require(set(modules) == required_modules,
             'installed native modules do not cover all adapter components')
    guard_path = modules['pyvoro2._fpguard']['path']
    _require(all(guard_path in item['dependencies'] for name, item in modules.items()
                 if name != 'pyvoro2._fpguard'),
             'native modules do not bind their FP guard dependency')
    internal = installation_root / 'pyvoro2/_internal'
    installed_approval = json.loads((internal / 'native_approval.json').read_text(
        encoding='utf8'))
    _require(installed_approval == approval, 'installed source approval differs')
    consumers = {path.relative_to(installation_root).as_posix():
                 file_identity(path)['sha256']
                 for path in (installation_root / 'pyvoro2').rglob('*.py')
                 if path != internal / '_qualification_installation.py'}
    _require(consumers == measurement['consumers'],
             'installed consumer files differ from reviewed source')
    report, execution = run_routes(
        source_root=source_root, installation_root=installation_root,
        corpus=Path(corpus).resolve(), output=output,
        build_evidence_path=build_path, components=required,
        candidate_module=candidate_module, candidate_build_path=candidate_build_path,
        runtime_environment=runtime_environment, safety_only=safety_only)
    expected_identity = {'record_schema': contract.RECORD_SCHEMA,
                         'policy_revision': contract.POLICY_REVISION,
                         **{name: measurement[name] for name in (
                             'source_sha256', 'schema_sha256', 'consumer_sha256')}}
    _require(report.get('native_identities') == {
        name: expected_identity for name in modules},
        'installed native source/schema consistency identity differs')
    if candidate_build is not None:
        _require(report.get('candidate') == {
            'sha256': candidate_identity['sha256'],
            'native_identity': expected_identity},
            'route runner used a different candidate payload or source identity')
        _require(verify_build(Path(candidate_records), source_root) == candidate_build
                 and file_identity(candidate_module) == candidate_identity,
                 'candidate effective build or payload changed during route evidence')
    _require(measure_source(source_root) == measurement,
             'source closure changed during finalization')
    _require(verify_build(Path(records_dir), source_root) == build,
             'effective build evidence changed during finalization')
    _require(verify_postprocess(postprocess, build, installation_root)
             == (modules, dependencies),
             'installed payload changed during route evidence')
    current_consumers = {path.relative_to(installation_root).as_posix()
                         for path in (installation_root / 'pyvoro2').rglob('*.py')
                         if path != internal / '_qualification_installation.py'}
    _require(current_consumers == set(consumers),
             'installed consumer inventory changed during route evidence')
    for name, digest in consumers.items():
        _require(file_identity(_relative(installation_root, name))['sha256'] == digest,
                 'installed consumer changed during route evidence')
    _require(json.loads(approval_path.read_text(encoding='utf8')) == approval
             and json.loads((internal / 'native_approval.json').read_text(
                 encoding='utf8')) == approval,
             'source approval changed during evidence')
    if safety_only:
        _require(_require_default_anchor(source_root, installation_root) == anchor,
                 'sanitizer safety anchor changed during evidence')
    evidence = {'schema': 'pyvoro2-native-finalization-evidence-v1',
                'build': build, 'postprocess': postprocess, 'routes': report,
                'route_execution': execution, 'source': measurement}
    if candidate_build is not None:
        evidence['candidate_build'] = candidate_build
        evidence['candidate_payload'] = candidate_identity
    return SimpleNamespace(
        evidence=evidence, contract=contract, measurement=measurement,
        approval=approval, target=target, toolchain=toolchain, adapter=adapter,
        required=required, build_data=build_data, modules=modules,
        dependencies=dependencies, consumers=consumers, internal=internal,
        output=output)


def exercise_sanitizer_safety(**arguments):
    """Execute instrumented safety evidence without any qualification issuance."""
    checked = _candidate_evidence(**arguments, safety_only=True)
    report = checked.evidence['routes']
    validate_safety_routes(report, checked.measurement, checked.modules,
                           checked.required, family=checked.toolchain['family'])
    evidence = {**checked.evidence,
                'schema': 'pyvoro2-native-sanitizer-safety-evidence-v1',
                'scope': 'sanitizer-safety-only'}
    _atomic_write(checked.output / 'sanitizer-safety-evidence.json',
                  checked.contract.canonical_json(evidence))
    return evidence


def finalize(*, source_root, installation_root, records_dir, postprocess_path,
             corpus, output, candidate_module=None, candidate_records=None,
             runtime_environment=None):
    """Run the final controlled checks and write the installation anchor last."""
    checked = _candidate_evidence(
        source_root=source_root, installation_root=installation_root,
        records_dir=records_dir, postprocess_path=postprocess_path, corpus=corpus,
        output=output, candidate_module=candidate_module,
        candidate_records=candidate_records, runtime_environment=runtime_environment)
    contract, report = checked.contract, checked.evidence['routes']
    validate_routes(report, checked.measurement, checked.modules, checked.required,
                    family=checked.toolchain['family'])
    _check_guard_objects(report, checked.evidence['build'])
    evidence_data = contract.canonical_json(checked.evidence)
    _atomic_write(checked.output / 'qualification-evidence.json', evidence_data)
    record = {'record_schema': contract.RECORD_SCHEMA,
              'policy_revision': contract.POLICY_REVISION,
              'installation_id': secrets.token_hex(32),
              'approval_sha256': contract.canonical_sha256(checked.approval),
              **{name: checked.measurement[name] for name in (
                  'source_sha256', 'schema_sha256', 'consumer_sha256')},
              'target': checked.target, 'toolchain': checked.toolchain,
              'effective_build': {
                  'qualified': True, 'adapter': checked.adapter,
                  'manifest_sha256': hashlib.sha256(checked.build_data).hexdigest()},
              'evidence_sha256': hashlib.sha256(evidence_data).hexdigest(),
              'modules': checked.modules, 'dependencies': checked.dependencies,
              'consumers': checked.consumers,
              'components': {name: {**checked.measurement['components'][name],
                                    'qualified': True,
                                    'evidence_sha256': contract.canonical_sha256(
                                        report['components'][name])}
                             for name in sorted(checked.required)}}
    data = contract.canonical_json(record)
    digest = hashlib.sha256(data).hexdigest()
    anchor = ('"""Generated by the controlled external qualification finalizer."""\n'
              "RECORD_FILENAME = 'native_qualification_record.json'\n"
              f"RECORD_SHA256 = '{digest}'\n"
              f"INSTALLATION_ID = '{record['installation_id']}'\n").encode('ascii')
    _atomic_write(checked.output / 'native_qualification_record.json', data)
    _atomic_write(checked.internal / 'native_qualification_record.json', data)
    _atomic_write(checked.internal / '_qualification_installation.py', anchor)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'installation-root', 'records-dir',
                 'postprocess-receipt', 'corpus', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate-module', type=Path)
    parser.add_argument('--candidate-records', type=Path)
    args = parser.parse_args(argv)
    try:
        record = finalize(source_root=args.source_root,
                          installation_root=args.installation_root,
                          records_dir=args.records_dir,
                          postprocess_path=args.postprocess_receipt,
                          corpus=args.corpus, output=args.output,
                          candidate_module=args.candidate_module,
                          candidate_records=args.candidate_records)
    except (FinalizationError, BuildEvidenceError, ValueError, OSError) as exc:
        parser.exit(1, f'qualification refused: {exc}\n')
    print(json.dumps({'installation_id': record['installation_id'],
                      'components': sorted(record['components'])}, sort_keys=True))


if __name__ == '__main__':
    main()
