"""CI routing must err toward qualification when source intent is uncertain."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


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
    'native-sanitizers', 'test-linux-qualified', 'test-other-platforms',
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
