#!/usr/bin/env python3
"""Conservative PR changed-path routing; integration pushes always run full CI.

All paths are repository-relative. Unrecognized or malformed paths are treated
as proof-sensitive. Keep this mapping in sync with the workflow job IDs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys


DOCS = frozenset({'lint-sync', 'docs-and-notebooks'})
DIST_ONLY = DOCS | {'build-dist'}
PACKAGING = DIST_ONLY | {'wheels'}
RUNTIME = DOCS | {
    'test-linux-runtime', 'test-linux-compat', 'test-other-platforms',
    'build-dist',
}
FULL = DOCS | {
    'native-sanitizers', 'native-avx-fma',
    'test-linux-qualified', 'test-other-platforms',
    'build-dist', 'wheels',
}

PACKAGING_TOOLS = frozenset({
    'build_wheel_from_sdist.py', 'build_wheels_wsl.sh', 'check_dist.py',
    'check_dist_metadata.py', 'check_wheel_matrix.py',
    'install_wheel_overlay.py', 'release_check.py',
})
DOC_TOOLS = frozenset({
    '_notebook_tools.py', 'check_notebooks.py', 'execute_notebooks.py',
    'export_notebooks.py', 'gen_readme.py',
})


def _classify_path(path: str) -> frozenset[str]:
    parts = PurePosixPath(path).parts
    if (not path or '\\' in path or path.startswith('/') or
            any(part in {'.', '..'} for part in path.split('/')) or
            not parts or any(not part for part in path.split('/'))):
        return FULL

    if path == 'pyproject.toml' or path == 'CMakeLists.txt':
        return FULL
    if path.startswith(('.github/', 'cpp/', 'vendor/', 'cmake/')):
        return FULL
    if path.startswith('src/pyvoro2/'):
        if (path.startswith(('src/pyvoro2/_internal/',
                             'src/pyvoro2/planar/')) or
                path in {'src/pyvoro2/api.py',
                         'src/pyvoro2/__about__.py'}):
            return FULL
        return RUNTIME if path.endswith('.py') else FULL
    if path.startswith('src/'):
        return FULL
    if path.startswith('tests/'):
        if path == 'tests/tooling/test_release_tools.py':
            return PACKAGING
        if path.startswith('tests/tooling/test_notebook') or path in {
                'tests/tooling/test_readme_sync.py',
                'tests/tooling/test_text_generation_tools.py',
                'tests/tooling/test_notebooks_meta.py'}:
            return DIST_ONLY
        return FULL  # Proof/oracle/fuzz/test routing and unknown tests.
    if path.startswith('tools/'):
        tool = path.removeprefix('tools/')
        if tool in PACKAGING_TOOLS:
            return PACKAGING
        if tool in DOC_TOOLS or tool == 'README.md':
            return DIST_ONLY
        if tool == 'check_scalar_prox_audit.py':
            return RUNTIME
        return FULL  # Includes native tools, installed checker, and router.
    if path == 'mkdocs.yml':
        return DOCS
    if path.startswith('docs/'):
        if path == 'docs/index.md' or path.startswith('docs/notebooks/'):
            return DIST_ONLY  # Generated README or packaged notebook export.
        return DOCS if path.endswith((
            '.md', '.yml', '.yaml', '.txt', '.css', '.js', '.png', '.svg',
            '.jpg', '.jpeg', '.gif', '.webp',
        )) else FULL
    if path.startswith('notebooks/'):
        return DIST_ONLY if path.endswith('.ipynb') else FULL
    if path.startswith(('examples/', 'benchmarks/')):
        return RUNTIME if path.endswith('.py') else DIST_ONLY
    if '/' not in path and (path.endswith('.md') or path in {
            'LICENSE', 'COPYING', 'LICENSE.voro++'}):
        return DIST_ONLY
    return FULL


def classify_paths(paths, *, integration: bool = False) -> set[str]:
    """Return required CI job IDs for a whole PR diff or an integration push."""
    if integration:
        return set(FULL)
    paths = list(paths)
    if not paths:
        return set(FULL)  # Diff acquisition must never silently skip CI.
    required = set(DOCS)
    for path in paths:
        required.update(_classify_path(path))
    return required


def changed_paths(base: str, head: str, *, cwd: Path | None = None) -> list[str]:
    """Read the full merge-base-to-head PR diff, counting rename ends separately."""
    raw = subprocess.check_output(
        ['git', 'diff', '--name-only', '-z', '--no-renames',
         f'{base}...{head}'], cwd=cwd,
    )
    return [os.fsdecode(path) for path in raw.split(b'\0') if path]


def gate_failures(requirements, results: dict[str, str]) -> list[str]:
    """Fail closed for missing classification or skipped/failed required jobs."""
    if requirements is None:
        return ['classification: missing or failed']
    return [f'{job}: {results.get(job, "missing")}'
            for job in sorted(requirements)
            if results.get(job) != 'success']


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--full', action='store_true',
                       help='integration or qualification push: ignore changed paths')
    group.add_argument('--pr-base', help='pull request base commit SHA')
    group.add_argument('--check-gate', action='store_true',
                       help='verify results passed in NEEDS_JSON')
    parser.add_argument('--pr-head', help='pull request head commit SHA')
    args = parser.parse_args(argv)

    if args.check_gate:
        needs = json.loads(os.environ['NEEDS_JSON'])
        classifier = needs.get('classify', {})
        raw = classifier.get('outputs', {}).get('requirements')
        requirements = json.loads(raw) if raw else None
        results = {name: info['result'] for name, info in needs.items()}
        failures = gate_failures(requirements, results)
        if failures:
            print('CI gate failed: ' + ', '.join(failures), file=sys.stderr)
            return 1
        print('CI gate: all required jobs succeeded')
        return 0

    if args.pr_base and not args.pr_head:
        parser.error('--pr-head is required with --pr-base')
    if args.pr_head and not args.pr_base:
        parser.error('--pr-head requires --pr-base')
    paths = changed_paths(args.pr_base, args.pr_head) if args.pr_base else []
    required = sorted(classify_paths(paths, integration=args.full))
    print(f'Changed paths: {paths if not args.full else "integration push"}')
    print(f'Required jobs: {required}')
    output = os.environ.get('GITHUB_OUTPUT')
    if output:
        with open(output, 'a', encoding='utf-8') as stream:
            stream.write(f'requirements={json.dumps(required)}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
