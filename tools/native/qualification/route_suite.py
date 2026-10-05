"""Controlled pre-issuance route exercise of the exact installed payload.

This external test process temporarily replaces admission with a gate over
the measured source and allowlisted candidate bytes. It cannot issue a record.
No bypass is compiled into production or installed in the package. The issuer
first checks the current source manifest and actual effective build evidence, invokes
this fixed suite itself, then independently binds its results. Distribution
tests subsequently run the issued installation without these replacements.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import zipfile


ARCHIVE_SHA256 = '85e2f0e8ac3678d83dc3b6d289abd580c9f36c80ee7934f46a194caf4e1ffe38'

# Union of the prior driver and CI native safety selections, including the
# warning callback hostile-state regression. These execute in the fixed child.
SANITIZER_TESTS = (
    'tests/forward/common/test_native_boundary_validation.py',
    'tests/forward/common/test_native_preconditions.py',
    'tests/forward/common/test_native_runtime_entry.py',
    'tests/forward/common/test_native_import_registration.py',
    'tests/forward/common/test_native_coercion_state.py',
    'tests/forward/common/test_native_route_admission.py',
    'tests/forward/common/test_native_warning_state.py',
    'tests/forward/common/test_native_certificate_refusal.py',
    'tests/forward/common/test_generator_preparation.py',
    'tests/forward/spatial/test_duplicate_check.py',
    'tests/forward/spatial/test_ghost_cells.py',
    'tests/forward/spatial/test_native_witness.py',
    'tests/forward/spatial/test_wp7_native_selected.py',
    'tests/tooling/test_native_witness_cpp.py',
    'tests/forward/planar/test_api_dispatch.py',
    'tests/forward/planar/test_wp6_native.py',
    'tests/forward/planar/test_wp6_profile_refusal.py',
    'tests/forward/planar/test_wp7_native.py',
    'tests/forward/test_wp7_oracle.py',
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack_corpus(archive, output):
    """Verify the independently archived bytes before reading any expectations."""
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError('original WP6 corpus identity differs')
    output.mkdir()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = output / member.filename
            if not path.resolve().is_relative_to(output.resolve()):
                raise ValueError('invalid corpus archive path')
            source.extract(member, output)
    for line in (output / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if sha256(output / name) != digest:
            raise ValueError('archived input bytes differ: ' + name)
    return output


def load_tool(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke_tool(module, arguments, output):
    previous = sys.argv
    try:
        sys.argv = [module.__file__, *map(str, arguments), '--output', str(output)]
        result = module.main()
        if result not in (None, 0):
            raise RuntimeError('qualification corpus runner failed')
    finally:
        sys.argv = previous
    return json.loads(output.read_text())


def corpus_count(report):
    return {'cases': len(report['cases']),
            'occurrences': sum(row.get('occurrences', 0) for row in report['cases'])}


def planar_corpora(root, production, candidate, archive, output):
    if candidate is None:
        raise ValueError('planar qualification requires observed stock candidate')
    manifest = candidate.parent.parent / 'planar-source-closure.txt'
    tool = load_tool(root / 'tools/native/check_planar_witness.py', 'wp6_fixed_suite')
    arguments = ['--module', production, '--qualification-module', candidate,
                 '--source-manifest', manifest]
    current = invoke_tool(tool, arguments, output / 'wp6-current.json')
    old = invoke_tool(tool, [*arguments, '--archive', archive],
                      output / 'wp6-archived.json')
    ghost = load_tool(root / 'tools/native/check_ghost_planar.py', 'wp7_fixed_suite')
    selected = invoke_tool(
        ghost, ['--module', candidate, '--production-module', production,
                '--source-manifest', manifest], output / 'wp7-selected.json')
    return {'wp6-planar': {'current': corpus_count(current),
                           'archived': corpus_count(old)},
            'wp7-planar': {'selected': corpus_count(selected)}}


def selections(root, components):
    spatial = root / 'tests/forward/spatial'
    planar = root / 'tests/forward/planar'
    common = root / 'tests/forward/common'
    wp8 = sorted(common.glob('test_wp8_*.py'))
    result = {
        'wp5-spatial': sorted(spatial.glob('test_wp5_*.py')),
        'wp6-planar': sorted(planar.glob('test_wp6_*.py')),
        'wp7-spatial': (sorted(spatial.glob('test_wp7_*.py')) +
                        sorted(common.glob('test_wp7_*.py'))),
        'wp7-planar': (sorted(planar.glob('test_wp7_*.py')) +
                       sorted(common.glob('test_wp7_*.py'))),
        'wp8-spatial': wp8 + [spatial / 'test_locate.py'],
        'wp8-planar': wp8,
    }
    return {name: [path.relative_to(root).as_posix() for path in result[name]]
            for name in components}


class TestResults:
    def __init__(self):
        self.results = {}
        self.collected = []
        self.deselected = []

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.failed or report.skipped:
            self.results[report.nodeid] = report.outcome

    def require_complete(self):
        if self.deselected:
            raise RuntimeError('fixed qualification tests were deselected')
        if (not self.collected or len(set(self.collected)) != len(self.collected)
                or set(self.results) != set(self.collected)):
            raise RuntimeError('fixed qualification test execution was incomplete')


def standard_offset(core):
    import numpy as np
    displacement = [float.fromhex('0x1.ffffff4000000p-1'),
                    float.fromhex('0x1.ffffffe000000p+0'), 0.]
    packet = core._observe_box(
        np.array([[0., 0., 0.], displacement]), np.array([0, 1], dtype=np.int32),
        ((-4., 4.),) * 3, (1, 1, 1), (False,) * 3, 8)
    offsets = []
    for cell in packet['cells']:
        rows = [row for row in cell['origins'] if row['kind'] == 'particle']
        assert len(rows) == 1
        row = rows[0]
        sign = 1 if cell['id'] == 0 else -1
        assert [struct.pack('>d', value).hex() for value in row['normal']] == [
            struct.pack('>d', sign * value if value else 0.).hex()
            for value in displacement]
        offsets.append(struct.pack('>d', row['offset']).hex())
    assert offsets == ['4013fffffb000000'] * 2
    return offsets[0]


def main():
    if sys.flags.optimize or not __debug__:
        raise RuntimeError('qualification requires enabled Python assertions')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'installation-root', 'corpus', 'output',
                 'build-evidence'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate-module', type=Path)
    parser.add_argument('--candidate-build-evidence', type=Path)
    parser.add_argument('--components', required=True)
    parser.add_argument('--mode', choices=('qualification', 'sanitizer-safety'),
                        default='qualification')
    args = parser.parse_args()
    root, installation = args.source_root.resolve(), args.installation_root.resolve()
    sys.path.insert(0, str(installation))
    sys.path.insert(0, str(root / 'tools/native'))
    from qualification.source_policy import measure_source
    from qualification.discriminators import run_arithmetic_controls, run_discriminators
    from pyvoro2 import _core, _core2d, _fpguard
    from pyvoro2._internal import native_qualification as authority
    from pyvoro2._internal import native_admission
    measurement = measure_source(root)
    modules = {'pyvoro2._core': _core, 'pyvoro2._core2d': _core2d,
               'pyvoro2._fpguard': _fpguard}
    identities = {name: module._qualification_identity()
                  for name, module in modules.items()}
    allowed = {}
    for module in modules.values():
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(installation):
            raise ValueError('candidate runner imported another installation')
        allowed[path] = sha256(path)
    candidate = args.candidate_module.resolve() if args.candidate_module else None
    if candidate:
        build = json.loads(args.candidate_build_evidence.read_text())
        expected = build['components']['_core2d']['output']
        assert Path(expected['path']).resolve() == candidate
        assert expected['sha256'] == sha256(candidate)
        allowed[candidate] = expected['sha256']
    components = frozenset(args.components.split(','))

    def candidate_registration(module):
        path = Path(module.__file__).resolve()
        assert allowed.get(path) == sha256(path), 'unobserved candidate native bytes'
        identity = module._qualification_identity()
        assert all(identity[key] == measurement[key] for key in (
            'source_sha256', 'schema_sha256', 'consumer_sha256'))
        module._require_runtime_environment()

    def candidate_admission(module, component):
        assert component in components, 'component is outside the reviewed adapter'
        candidate_registration(module)

    authority.register_native = candidate_registration
    authority.require_native = candidate_admission
    # native_admission resolves authority functions dynamically. No production
    # package code contains or detects this external test-process substitution.
    native_admission.require_environment()
    test_hooks = ('_planar_stock_', '_planar_candidate_',
                  '_planar_ghost_stock_', '_planar_ghost_candidate_')
    for module in modules.values():
        assert not any(name.startswith(test_hooks) for name in dir(module)), (
            'qualification hook in production')
    outdir = args.output.parent
    corpora = {}
    if 'wp6-planar' in components:
        archive = unpack_corpus(args.corpus, outdir / 'archived-inputs')
        corpora = planar_corpora(root, Path(_core2d.__file__), candidate,
                                 archive, outdir)
    selection = selections(root, components)
    test_results = TestResults()
    import pytest
    paths = sorted({name for files in selection.values() for name in files})
    if args.mode == 'sanitizer-safety':
        paths = sorted(set(paths) | set(SANITIZER_TESTS))
    code = pytest.main(['-q', '-x', '-o', 'addopts=', '--rootdir', str(root),
                        *[str(root / name) for name in paths]], plugins=[test_results])
    if code != 0:
        raise RuntimeError('fixed route regression suite failed')
    test_results.require_complete()
    claims = {}
    for name, files in selection.items():
        rows = [{'nodeid': node, 'outcome': outcome}
                for node, outcome in sorted(test_results.results.items())
                if node.split('::')[0] in files]
        assert rows and any(row['outcome'] == 'passed' for row in rows)
        assert all(row['outcome'] in ('passed', 'skipped') for row in rows)
        claims[name] = {'passed': True, 'tests': rows,
                        'corpora': corpora.get(name, {})}
    build = json.loads(args.build_evidence.read_text())
    controls = (run_arithmetic_controls if args.mode == 'sanitizer-safety'
                else run_discriminators)
    discriminators = controls(root, build, outdir / 'discriminators')
    discriminators['standard_strict_bits'] = standard_offset(_core)
    report = {'schema': 'pyvoro2-native-route-evidence-v1',
              'source_sha256': measurement['source_sha256'],
              'modules': {name: sha256(module.__file__)
                          for name, module in modules.items()},
              'native_identities': identities, 'components': claims,
              'discriminators': discriminators}
    if args.mode == 'sanitizer-safety':
        report.update(schema='pyvoro2-native-sanitizer-routes-v1',
                      scope='sanitizer-safety-only',
                      safety_tests=[{'nodeid': node, 'outcome': outcome}
                                    for node, outcome in sorted(
                                        test_results.results.items())
                                    if node.split('::')[0] in SANITIZER_TESTS])
    if candidate:
        spec = importlib.util.spec_from_file_location('recorded._core2d', candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        candidate_registration(module)
        report['candidate'] = {'sha256': sha256(candidate),
                               'native_identity': module._qualification_identity()}
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')


if __name__ == '__main__':
    main()
