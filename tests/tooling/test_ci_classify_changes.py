"""CI routing must err toward qualification when source intent is uncertain."""

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'ci_classify_changes', ROOT / 'tools' / 'ci_classify_changes.py'
)


@pytest.fixture(scope='module')
def classifier():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


DOCS = {'lint-sync', 'docs-and-notebooks'}
DIST_ONLY = DOCS | {'build-dist'}
FULL = DOCS | {
    'native-sanitizers', 'native-avx-fma',
    'test-linux-qualified', 'test-other-platforms',
    'build-dist', 'wheels',
}
RUNTIME = DOCS | {
    'test-linux-runtime', 'test-linux-compat', 'test-other-platforms',
    'build-dist',
}
PACKAGING = DOCS | {'build-dist', 'wheels'}


@pytest.mark.parametrize(('paths', 'expected'), [
    (['docs/development/plans/v0.9.md'], DOCS),
    (['docs/stylesheets/extra.css'], DOCS),
    (['docs/javascripts/mathjax.js'], DOCS),
    (['docs/assets/quickstart_box.png'], DOCS),
    (['docs/requirements.txt'], DOCS),
    (['src/pyvoro2/inverse/separator/solver.py'], RUNTIME),
    (['cpp/bindings.cpp'], FULL),
    (['cmake/NativeFP.cmake'], FULL),
    (['vendor/voro++/src/cell.cc'], FULL),
    (['pyproject.toml'], FULL),
    (['.github/workflows/ci.yml'], FULL),
    (['tools/ci_classify_changes.py'], FULL),
    (['tools/check_dist.py'], PACKAGING),
    (['docs/development/plans/v0.9.md', 'cpp/bindings.cpp'], FULL),
    (['unclassified/data.bin'], FULL),
    (['src/pyvoro2/_internal/planar/wp6_certificate.py'], FULL),
    (['src/pyvoro2/_internal/native_source_manifest.json'], FULL),
    (['tests/forward/planar/test_wp6_native.py'], FULL),
    (['README.md'], DIST_ONLY),
    (['docs/index.md'], DIST_ONLY),
    (['notebooks/01_basic_compute.ipynb'], DIST_ONLY),
    (['tests/tooling/test_release_tools.py'], PACKAGING),
    ([], FULL),
])
def test_representative_profiles(classifier, paths, expected):
    assert classifier.classify_paths(paths) == expected


def test_push_full_regardless_of_paths(classifier):
    assert classifier.classify_paths(['docs/development/plans/v0.9.md'],
                                     integration=True) == FULL


@pytest.mark.parametrize('path', ['../docs/a.md', '/docs/a.md',
                                  'docs/../cpp/bindings.cpp', '',
                                  '.github/unknown.yml'])
def test_untrusted_or_ambiguous_path_is_full(classifier, path):
    assert classifier.classify_paths([path]) == FULL


def test_git_diff_covers_entire_pr_not_latest_commit(classifier, tmp_path):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path,
                                       text=True).strip()

    git('init', '-q')
    git('config', 'user.email', 'ci@example.invalid')
    git('config', 'user.name', 'CI')
    (tmp_path / 'cpp').mkdir()
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'cpp' / 'bindings.cpp').write_text('base\n')
    git('add', '.')
    git('commit', '-qm', 'base')
    base = git('rev-parse', 'HEAD')
    (tmp_path / 'cpp' / 'bindings.cpp').write_text('changed\n')
    git('commit', '-qam', 'native')
    (tmp_path / 'docs' / 'note.md').write_text('docs\n')
    git('add', '.')
    git('commit', '-qm', 'docs')
    head = git('rev-parse', 'HEAD')
    assert classifier.classify_paths(
        classifier.changed_paths(base, head, cwd=tmp_path)
    ) == FULL


def test_aggregate_accepts_only_intentional_skips(classifier):
    requirements = DOCS
    results = {job: 'skipped' for job in FULL | RUNTIME}
    results['lint-sync'] = 'success'
    results['docs-and-notebooks'] = 'success'
    assert classifier.gate_failures(requirements, results) == []
    results['docs-and-notebooks'] = 'skipped'
    assert classifier.gate_failures(requirements, results) == [
        'docs-and-notebooks: skipped'
    ]
    results['docs-and-notebooks'] = 'cancelled'
    assert classifier.gate_failures(requirements, results) == [
        'docs-and-notebooks: cancelled'
    ]
    results['docs-and-notebooks'] = 'failure'
    assert classifier.gate_failures(requirements, results) == [
        'docs-and-notebooks: failure'
    ]


def test_aggregate_fails_missing_classifier_or_required_wheels(classifier):
    results = {job: 'success' for job in FULL}
    results['wheels'] = 'skipped'
    assert classifier.gate_failures(FULL, results) == ['wheels: skipped']
    assert classifier.gate_failures(None, results) == [
        'classification: missing or failed'
    ]


@pytest.mark.parametrize('result', ['missing', 'skipped', 'failure', 'cancelled'])
@pytest.mark.parametrize('job', ['native-sanitizers', 'native-avx-fma'])
def test_native_changes_require_safety_and_strict_controls(classifier, result, job):
    required = classifier.classify_paths(['cpp/native_runtime.hpp'])
    assert job in required
    results = {job: 'success' for job in required}
    if result == 'missing':
        results.pop(job)
    else:
        results[job] = result
    assert classifier.gate_failures(required, results) == [
        f'{job}: {result}'
    ]


def _gate_needs(requirements=DOCS):
    needs = {job: {'result': 'success'} for job in FULL | RUNTIME}
    needs['classify'] = {
        'result': 'success',
        'outputs': {'requirements': json.dumps(sorted(requirements))},
    }
    return needs


def _run_gate(needs, *, raw=False):
    environment = os.environ.copy()
    environment.pop('NEEDS_JSON', None)
    if needs is not None:
        environment['NEEDS_JSON'] = needs if raw else json.dumps(needs)
    return subprocess.run(
        [sys.executable, str(ROOT / 'tools/ci_classify_changes.py'),
         '--check-gate'], env=environment, capture_output=True, text=True,
    )


def _assert_gate_refused(result):
    assert result.returncode == 1
    assert result.stderr.startswith('CI gate failed:')
    assert 'all required jobs succeeded' not in result.stdout
    assert 'Traceback' not in result.stderr


@pytest.mark.parametrize('requirements', [
    DOCS, DIST_ONLY, PACKAGING, RUNTIME, FULL, PACKAGING | RUNTIME, FULL | RUNTIME,
])
def test_gate_cli_accepts_valid_profiles_with_nonrequired_cancellation(requirements):
    needs = _gate_needs(requirements)
    for index, job in enumerate(sorted((FULL | RUNTIME) - requirements)):
        needs[job]['result'] = 'skipped' if index % 2 else 'cancelled'
    result = _run_gate(needs)
    assert result.returncode == 0, result.stderr
    assert 'all required jobs succeeded' in result.stdout


@pytest.mark.parametrize('record', [
    None, [], 'success', {},
    *({'result': result, 'outputs': {'requirements': json.dumps(sorted(DOCS))}}
      for result in ('failure', 'cancelled', 'skipped', '', None, True)),
])
def test_gate_cli_refuses_bad_classifier_despite_favorable_jobs(record):
    needs = _gate_needs()
    if record is None:
        del needs['classify']
    else:
        needs['classify'] = record
    _assert_gate_refused(_run_gate(needs))


@pytest.mark.parametrize('raw', [
    None, '', '{', 'null', '[]', 'true', '"success"',
    '{"classify":{"result":"failure"},"classify":'
    '{"result":"success","outputs":{"requirements":'
    '"[\\"lint-sync\\",\\"docs-and-notebooks\\"]"}},'
    '"lint-sync":{"result":"success"},'
    '"docs-and-notebooks":{"result":"success"}}',
])
def test_gate_cli_refuses_missing_or_malformed_needs(raw):
    _assert_gate_refused(_run_gate(raw, raw=True))


@pytest.mark.parametrize('outputs', [
    None, [], 'success', {}, {'requirements': None}, {'requirements': []},
])
def test_gate_cli_refuses_missing_or_malformed_outputs(outputs):
    needs = _gate_needs()
    if outputs is None:
        del needs['classify']['outputs']
    else:
        needs['classify']['outputs'] = outputs
    _assert_gate_refused(_run_gate(needs))


@pytest.mark.parametrize('requirements', [
    '', '{', '[]', '{}', '{"lint-sync":"success"}', 'null', 'true',
    '"lint-sync"', '1',
    json.dumps(['lint-sync', 'docs-and-notebooks', 'lint-sync']),
    json.dumps(['lint-sync', 'docs-and-notebooks', 'unknown-job']),
    json.dumps(['lint-sync', 'docs-and-notebooks', 'classify']),
    json.dumps(['lint-sync', 'docs-and-notebooks', 'ci-gate']),
    json.dumps(['lint-sync', 'docs-and-notebooks', 1]),
    json.dumps(['lint-sync', 'docs-and-notebooks', None]),
    json.dumps(['lint-sync', 'docs-and-notebooks', []]),
    json.dumps(['lint-sync']),
    json.dumps(['docs-and-notebooks']),
    json.dumps(['lint-sync', 'docs-and-notebooks', 'native-sanitizers']),
])
def test_gate_cli_refuses_invalid_or_incomplete_requirements(requirements):
    needs = _gate_needs()
    needs['classify']['outputs']['requirements'] = requirements
    _assert_gate_refused(_run_gate(needs))


@pytest.mark.parametrize('record', [
    None, [], 'success', {}, {'result': True}, {'result': {'success': True}},
    *({'result': result} for result in ('failure', 'cancelled', 'skipped', '')),
])
def test_gate_cli_requires_literal_success_from_each_required_job(record):
    needs = _gate_needs()
    if record is None:
        del needs['docs-and-notebooks']
    else:
        needs['docs-and-notebooks'] = record
    _assert_gate_refused(_run_gate(needs))


@pytest.fixture(scope='module')
def workflows():
    return {name: yaml.safe_load(
        (ROOT / '.github/workflows' / f'{name}.yml').read_text()
    ) for name in ('ci', 'wheels')}


def _expression(value, github):
    """Evaluate our small context-only policy expressions, not scheduling."""
    if not isinstance(value, str):
        return value
    expression = value.removeprefix('${{').removesuffix('}}').strip()

    def context(match):
        item = github
        for key in match[0].split('.')[1:]:
            item = item.get(key) if isinstance(item, dict) else None
        return repr(item)

    expression = re.sub(r'github\.[\w.]+', context, expression)
    expression = expression.replace('&&', ' and ').replace('||', ' or ')
    return eval(expression, {'__builtins__': {},
                             'format': lambda template, *args:
                             template.format(*args)})


def _concurrency_group(policy, github):
    return re.sub(r'\$\{\{.*?\}\}',
                  lambda match: str(_expression(match[0], github)),
                  policy['group'])


def test_workflow_supersedes_only_same_pr_and_keeps_nonpr_evidence_unique(workflows):
    policy = workflows['ci'].get('concurrency')
    assert isinstance(policy, dict)
    first = {'workflow': 'CI', 'event_name': 'pull_request', 'run_id': 100,
             'run_attempt': 1, 'event': {'pull_request': {'number': 113}}}
    second = {**first, 'run_id': 101}
    group = _concurrency_group(policy, first)
    assert group == _concurrency_group(policy, second)
    assert _expression(policy['cancel-in-progress'], first) is True
    assert group != _concurrency_group(policy, {
        **first, 'event': {'pull_request': {'number': 114}},
    })
    assert group != _concurrency_group(policy, {**first, 'workflow': 'Other CI'})
    for event in ('push', 'workflow_dispatch', 'workflow_call', 'release'):
        for ref in ('refs/heads/dev', 'refs/heads/main',
                    'refs/heads/issue88-native-qualification', 'refs/tags/v0.9.0'):
            nonpr = {**first, 'event_name': event, 'ref': ref, 'event': {}}
            assert _expression(policy['cancel-in-progress'], nonpr) is False
            assert _concurrency_group(policy, nonpr) != _concurrency_group(
                policy, {**nonpr, 'run_id': 101})
            assert _concurrency_group(policy, nonpr) != _concurrency_group(
                policy, {**nonpr, 'run_attempt': 2})
            assert _concurrency_group(policy, nonpr) != group
    # Reusable github.workflow is the caller name: no callee group may collide.
    assert 'concurrency' not in workflows['wheels']
    assert all('concurrency' not in job
               for job in workflows['wheels']['jobs'].values())
    assert 'concurrency' not in workflows['ci']['jobs']['wheels']


MATRICES = {
    'ci': ('test-linux-qualified', 'test-linux-compat', 'test-other-platforms'),
    'wheels': ('wheels-manylinux-x86_64', 'wheels-windows-amd64',
               'wheels-macos-arm64', 'wheels-macos-x86_64'),
}


@pytest.mark.parametrize(('workflow', 'job'), [
    (workflow, job) for workflow, jobs in MATRICES.items() for job in jobs
])
def test_every_matrix_is_failfast_only_in_pr_caller_context(workflows, workflow, job):
    policy = workflows[workflow]['jobs'][job]['strategy']['fail-fast']
    for event in ('pull_request', 'push', 'workflow_dispatch',
                  'workflow_call', 'release'):
        assert _expression(policy, {'event_name': event}) is (event == 'pull_request')
    assert workflows['ci']['jobs']['wheels']['uses'] == './.github/workflows/wheels.yml'
    assert set(workflows['wheels']['on']) == {
        'workflow_call', 'push', 'workflow_dispatch',
    }


def test_gate_directly_needs_classifier_and_all_selectable_jobs(workflows, classifier):
    jobs = workflows['ci']['jobs']
    gate = jobs['ci-gate']
    selectable = FULL | RUNTIME
    assert classifier.FULL | classifier.RUNTIME == selectable
    assert set(jobs) == selectable | {'classify', 'ci-gate'}
    assert set(gate['needs']) == selectable | {'classify'}
    assert gate['name'] == 'CI gate'
    assert gate['if'].removeprefix('${{').removesuffix('}}').strip() == 'always()'
    checker = [step for step in gate['steps']
               if '--check-gate' in step.get('run', '')]
    assert len(checker) == 1
    assert checker[0]['run'].split() == [
        'python', 'tools/ci_classify_changes.py', '--check-gate',
    ]
    assert checker[0]['env']['NEEDS_JSON'].replace(' ', '') == '${{toJSON(needs)}}'


def test_workflows_have_no_allowed_failures_or_actions_write_escape(workflows):
    def records(value):
        if isinstance(value, dict):
            yield value
            for item in value.values():
                yield from records(item)
        elif isinstance(value, list):
            for item in value:
                yield from records(item)

    for record in records(workflows):
        assert record.get('continue-on-error', False) is False
        assert record.get('permissions') != 'write-all'
        if isinstance(record.get('permissions'), dict):
            assert record['permissions'].get('actions') != 'write'


def test_ci_preserves_every_python_and_platform_matrix_member(workflows):
    jobs = workflows['ci']['jobs']
    versions = ['3.10', '3.11', '3.12', '3.13', '3.14']
    assert jobs['test-linux-qualified']['strategy']['matrix'] == {
        'python-version': versions,
    }
    assert jobs['test-linux-compat']['strategy']['matrix'] == {
        'python-version': ['3.10', '3.11', '3.12', '3.14'],
    }
    assert jobs['test-other-platforms']['strategy']['matrix'] == {
        'os': ['macos-15', 'windows-latest'], 'python-version': versions,
    }


def test_wheels_preserves_twenty_artifacts_and_validator_dependency_closure(workflows):
    jobs = workflows['wheels']['jobs']
    versions = ['3.10', '3.11', '3.12', '3.13', '3.14']
    assert jobs['wheels-manylinux-x86_64']['strategy']['matrix'] == {
        'include': [{'python-version': version, 'python-abi': abi}
                    for version, abi in zip(versions, (
                        'cp310-cp310', 'cp311-cp311', 'cp312-cp312',
                        'cp313-cp313', 'cp314-cp314'))],
    }
    for job, runner in (
        ('wheels-windows-amd64', 'windows-latest'),
        ('wheels-macos-arm64', 'macos-15'),
        ('wheels-macos-x86_64', 'macos-15-intel'),
    ):
        assert jobs[job]['strategy']['matrix'] == {'python-version': versions}
        assert jobs[job]['runs-on'] == runner
    assert set(jobs['validate']['needs']) == set(MATRICES['wheels']) | {'sdist'}
    assert set(jobs) == set(MATRICES['wheels']) | {'sdist', 'validate'}
    commands = '\n'.join(step.get('run', '') for step in jobs['validate']['steps'])
    for command in (
        'python tools/check_wheel_matrix.py --require-qualification release-dist',
        'python tools/check_dist.py --require-qualification release-dist',
        'python tools/check_dist_metadata.py release-dist',
        'tools/check_installed_package.py', '--forbid-scipy',
    ):
        assert command in commands
    for job in (*MATRICES['wheels'], 'sdist'):
        uploads = [step for step in jobs[job]['steps']
                   if step.get('uses', '').startswith('actions/upload-artifact@')]
        assert any(step['with'].get('if-no-files-found') == 'error'
                   for step in uploads)


def test_ci_keeps_controlled_qualification_and_sdist_roundtrip(workflows):
    jobs = workflows['ci']['jobs']
    for job, suite in (
        ('test-linux-qualified', 'full'), ('test-linux-runtime', 'full'),
        ('test-linux-compat', 'adapter'), ('test-other-platforms', 'adapter'),
        ('native-avx-fma', 'full'), ('native-sanitizers', 'adapter'),
    ):
        commands = ' '.join(' '.join(step.get('run', '').split())
                            for step in jobs[job]['steps'])
        assert 'tools/native/qualification/build.py' in commands
        assert f'--repair none --suite {suite}' in commands
        if job == 'native-sanitizers':
            assert '--sanitizers' in commands
    commands = ' '.join(' '.join(step.get('run', '').split())
                        for step in jobs['build-dist']['steps'])
    for command in (
        'python -m build --sdist', 'tar -xf', '--source-root "$sdist_source"',
        '--repair none --suite full', 'tools/check_dist.py --require-qualification',
        'tools/check_dist_metadata.py', '--forbid-scipy',
    ):
        assert command in commands
