"""Shared implementation for repository notebook maintenance commands."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import re
from typing import Sequence

import nbformat
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / 'src'
NOTEBOOKS = REPO_ROOT / 'notebooks'
SKIP_TAG = 'skip-execution'
_ANSI_ESCAPE = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


class NotebookMaintenanceError(RuntimeError):
    """Raised when notebook validation or execution fails."""


def iter_notebooks(selected: Sequence[str] | None = None) -> tuple[Path, ...]:
    """Return repository notebook paths in maintenance order."""

    if not selected:
        paths = tuple(sorted(NOTEBOOKS.glob('*.ipynb')))
        if not paths:
            raise NotebookMaintenanceError(
                f'no notebooks found under {NOTEBOOKS}'
            )
        return paths

    names = tuple(selected)
    paths: list[Path] = []
    for name in names:
        if Path(name).name != name or not name.endswith('.ipynb'):
            raise NotebookMaintenanceError(
                f'{name!r}: expected a notebook filename under notebooks/'
            )
        paths.append(NOTEBOOKS / name)
    return tuple(paths)


def load_notebook(path: Path) -> nbformat.NotebookNode:
    """Load and schema-validate one version-4 notebook."""

    if not path.exists():
        raise NotebookMaintenanceError(f'missing notebook: {path}')
    try:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
    except Exception as exc:  # nbformat exposes several parse/schema errors
        raise NotebookMaintenanceError(
            f'{path.name}: invalid notebook structure or metadata: {exc}'
        ) from exc
    return notebook


def _is_executable(cell: nbformat.NotebookNode) -> bool:
    return cell.cell_type == 'code' and bool(cell.source.strip())


def _is_skipped(cell: nbformat.NotebookNode) -> bool:
    return SKIP_TAG in cell.metadata.get('tags', [])


def validate_committed_metadata(
    notebook: nbformat.NotebookNode,
    *,
    path: Path,
) -> None:
    """Validate committed execution counts and reject stored errors."""

    code_cells = 0
    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != 'code':
            continue
        code_cells += 1
        if _is_executable(cell) and not _is_skipped(cell):
            if cell.execution_count is None:
                raise NotebookMaintenanceError(
                    f'{path.name}: nonempty code cell {index} has no '
                    'execution count; execute it or add the skip-execution tag'
                )
        for output in cell.outputs:
            if output.output_type == 'error':
                ename = output.get('ename', 'error')
                evalue = output.get('evalue', '')
                raise NotebookMaintenanceError(
                    f'{path.name}: code cell {index} contains a stored error '
                    f'output: {ename}: {evalue}'
                )
    if not code_cells:
        raise NotebookMaintenanceError(f'{path.name}: contains no code cells')


def _kernel_environment(*, use_src: bool) -> dict[str, str]:
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    env['PYTHONHASHSEED'] = '0'
    if use_src:
        current = env.get('PYTHONPATH')
        entries = [str(SRC)]
        if current:
            entries.append(current)
        env['PYTHONPATH'] = os.pathsep.join(entries)
    return env


def _prepare_for_execution(
    notebook: nbformat.NotebookNode,
) -> nbformat.NotebookNode:
    """Clear executable cells while preserving explicitly skipped output."""

    original_metadata = deepcopy(notebook.metadata)
    original_metadata.pop('widgets', None)
    for cell in notebook.cells:
        cell.metadata.pop('execution', None)
        if _is_executable(cell) and not _is_skipped(cell):
            cell.execution_count = None
            cell.outputs = []
    return original_metadata


def _execution_failure_context(
    notebook: nbformat.NotebookNode,
    *,
    path: Path,
    exception: Exception,
) -> str:
    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != 'code':
            continue
        for output in cell.outputs:
            if output.output_type != 'error':
                continue
            traceback = '\n'.join(output.get('traceback', []))
            if not traceback:
                traceback = (
                    f"{output.get('ename', 'error')}: "
                    f"{output.get('evalue', '')}"
                )
            traceback = _ANSI_ESCAPE.sub('', traceback)
            return (
                f'{path.name}: execution failed in notebook cell {index}\n'
                f'Source:\n{cell.source.rstrip()}\n'
                f'Traceback:\n{traceback.rstrip()}'
            )
    detail = _ANSI_ESCAPE.sub('', str(exception)).rstrip()
    return f'{path.name}: notebook execution failed: {detail}'


def execute_clean(
    notebook: nbformat.NotebookNode,
    *,
    path: Path,
    use_src: bool = False,
) -> nbformat.NotebookNode:
    """Clear and execute a notebook in a fresh kernel rooted at the repository."""

    original_metadata = _prepare_for_execution(notebook)
    kernel_name = notebook.metadata.get('kernelspec', {}).get('name', 'python3')
    client = NotebookClient(
        notebook,
        kernel_name=kernel_name,
        timeout=120,
        startup_timeout=60,
        allow_errors=False,
        force_raise_errors=True,
        record_timing=False,
        store_widget_state=False,
        skip_cells_with_tag=SKIP_TAG,
    )
    try:
        executed = client.execute(
            cwd=str(REPO_ROOT),
            env=_kernel_environment(use_src=use_src),
        )
    except Exception as exc:
        message = _execution_failure_context(
            notebook,
            path=path,
            exception=exc,
        )
        raise NotebookMaintenanceError(message) from exc

    # nbclient records kernel language metadata during execution. Preserve the
    # reviewed source metadata and omit widget/timing state from committed files.
    executed.metadata = original_metadata
    for cell in executed.cells:
        cell.metadata.pop('execution', None)
    return executed


def check_notebook(path: Path, *, use_src: bool = False) -> None:
    """Validate committed metadata, then execute a clean in-memory copy."""

    notebook = load_notebook(path)
    validate_committed_metadata(notebook, path=path)
    execute_clean(notebook, path=path, use_src=use_src)


def refresh_notebook(path: Path, *, use_src: bool = False) -> bool:
    """Execute one notebook and write it only after successful completion."""

    original = path.read_text(encoding='utf-8') if path.exists() else ''
    notebook = load_notebook(path)
    executed = execute_clean(notebook, path=path, use_src=use_src)
    validate_committed_metadata(executed, path=path)
    rendered = nbformat.writes(executed, version=4)
    if original.replace('\r\n', '\n') == rendered:
        return False
    path.write_text(rendered, encoding='utf-8', newline='\n')
    return True
