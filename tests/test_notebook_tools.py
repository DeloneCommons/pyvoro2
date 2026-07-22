from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


nbformat = pytest.importorskip('nbformat')
pytest.importorskip('nbclient')
pytest.importorskip('ipykernel')

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / 'tools'
MODULE_PATH = TOOLS / '_notebook_tools.py'
CHECKER_PATH = TOOLS / 'check_notebooks.py'

spec = importlib.util.spec_from_file_location('_notebook_tools', MODULE_PATH)
assert spec is not None and spec.loader is not None
notebook_tools = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = notebook_tools
spec.loader.exec_module(notebook_tools)

checker_spec = importlib.util.spec_from_file_location(
    'check_notebooks_test', CHECKER_PATH
)
assert checker_spec is not None and checker_spec.loader is not None
check_notebooks = importlib.util.module_from_spec(checker_spec)
checker_spec.loader.exec_module(check_notebooks)


def _notebook(*cells: object) -> object:
    return nbformat.v4.new_notebook(
        cells=list(cells),
        metadata={
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {'name': 'python'},
        },
    )


def _write(path: Path, notebook: object) -> None:
    nbformat.write(notebook, path)


def test_executed_code_cell_passes_metadata_validation(tmp_path: Path) -> None:
    path = tmp_path / 'executed.ipynb'
    cell = nbformat.v4.new_code_cell('value = 1')
    cell.execution_count = 1
    notebook_tools.validate_committed_metadata(_notebook(cell), path=path)


def test_unexecuted_code_cell_fails_metadata_validation(tmp_path: Path) -> None:
    path = tmp_path / 'unexecuted.ipynb'
    notebook = _notebook(nbformat.v4.new_code_cell('value = 1'))
    with pytest.raises(
        notebook_tools.NotebookMaintenanceError,
        match='code cell 1 has no execution count',
    ):
        notebook_tools.validate_committed_metadata(notebook, path=path)


def test_default_checker_discovers_new_unexecuted_notebook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    probe = tmp_path / '09_unexecuted_probe.ipynb'
    _write(
        probe,
        _notebook(
            nbformat.v4.new_code_cell("raise RuntimeError('must not run')")
        ),
    )
    monkeypatch.setattr(notebook_tools, 'NOTEBOOKS', tmp_path)

    assert check_notebooks.main([]) == 1

    captured = capsys.readouterr()
    assert 'Validating 09_unexecuted_probe.ipynb...' in captured.out
    assert '09_unexecuted_probe.ipynb' in captured.err
    assert 'has no execution count' in captured.err


def test_malformed_cell_metadata_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / 'malformed.ipynb'
    path.write_text(
        json.dumps(
            {
                'cells': [
                    {
                        'cell_type': 'code',
                        'execution_count': 1,
                        'id': 'bad-cell',
                        'metadata': [],
                        'outputs': [],
                        'source': 'value = 1',
                    }
                ],
                'metadata': {},
                'nbformat': 4,
                'nbformat_minor': 5,
            }
        ),
        encoding='utf-8',
    )
    with pytest.raises(
        notebook_tools.NotebookMaintenanceError,
        match='invalid notebook structure or metadata',
    ):
        notebook_tools.load_notebook(path)


def test_real_kernel_stores_stream_and_final_expression_outputs(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'jupyter-semantics.ipynb'
    notebook = _notebook(
        nbformat.v4.new_code_cell("print('stream-value')"),
        nbformat.v4.new_code_cell('6 * 7'),
    )

    executed = notebook_tools.execute_clean(notebook, path=path)

    stream = executed.cells[0].outputs[0]
    result = executed.cells[1].outputs[0]
    assert stream.output_type == 'stream'
    assert stream.text == 'stream-value\n'
    assert result.output_type == 'execute_result'
    assert result.data['text/plain'] == '42'
    assert [cell.execution_count for cell in executed.cells] == [1, 2]


def test_execution_error_includes_notebook_cell_source_and_traceback(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'failure.ipynb'
    source = "raise ValueError('kernel boom')"
    notebook = _notebook(nbformat.v4.new_code_cell(source))

    with pytest.raises(notebook_tools.NotebookMaintenanceError) as caught:
        notebook_tools.execute_clean(notebook, path=path)

    message = str(caught.value)
    assert 'failure.ipynb' in message
    assert 'notebook cell 1' in message
    assert source in message
    assert 'ValueError' in message
    assert 'kernel boom' in message


def test_validation_executes_in_memory_without_modifying_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'unchanged.ipynb'
    cell = nbformat.v4.new_code_cell("print('fresh')")
    cell.execution_count = 9
    cell.outputs = [
        nbformat.v4.new_output('stream', name='stdout', text='stale\n')
    ]
    _write(path, _notebook(cell))
    before = path.read_bytes()

    notebook_tools.check_notebook(path)

    assert path.read_bytes() == before


def test_refresh_clears_stale_ordinary_output_before_execution(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'refresh.ipynb'
    cell = nbformat.v4.new_code_cell("print('fresh')")
    cell.execution_count = 9
    cell.outputs = [
        nbformat.v4.new_output('stream', name='stdout', text='stale\n')
    ]
    _write(path, _notebook(cell))

    assert notebook_tools.refresh_notebook(path)

    refreshed = nbformat.read(path, as_version=4)
    assert refreshed.cells[0].execution_count == 1
    assert refreshed.cells[0].outputs[0].text == 'fresh\n'
    assert 'stale' not in refreshed.cells[0].outputs[0].text


def test_skipped_rich_output_is_permitted_preserved_and_not_run(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'rich.ipynb'
    ordinary = nbformat.v4.new_code_cell('value = 1')
    ordinary.execution_count = 1
    skipped = nbformat.v4.new_code_cell("raise RuntimeError('must not run')")
    skipped.execution_count = 7
    skipped.metadata['tags'] = [notebook_tools.SKIP_TAG]
    skipped.outputs = [
        nbformat.v4.new_output(
            'display_data',
            data={
                'text/html': '<div>reviewed rich output</div>',
                'application/x-reviewed': '{"stable": true}',
            },
        )
    ]
    notebook = _notebook(ordinary, skipped)
    _write(path, notebook)
    expected_output = dict(skipped.outputs[0])

    notebook_tools.validate_committed_metadata(notebook, path=path)
    assert notebook_tools.refresh_notebook(path)

    refreshed = nbformat.read(path, as_version=4)
    assert refreshed.cells[1].execution_count == 7
    assert dict(refreshed.cells[1].outputs[0]) == expected_output


def test_use_src_overlay_is_visible_inside_spawned_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / 'src'
    source_root.mkdir()
    (source_root / 'kernel_overlay_probe.py').write_text(
        'OVERLAY_VALUE = 73\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(notebook_tools, 'SRC', source_root)
    path = tmp_path / 'overlay.ipynb'
    notebook = _notebook(
        nbformat.v4.new_code_cell(
            'from kernel_overlay_probe import OVERLAY_VALUE\nOVERLAY_VALUE'
        )
    )

    executed = notebook_tools.execute_clean(notebook, path=path, use_src=True)

    assert executed.cells[0].outputs[0].data['text/plain'] == '73'


def test_selected_notebook_cli_is_usable() -> None:
    result = subprocess.run(
        [
            sys.executable,
            'tools/check_notebooks.py',
            '07_powerfit_infeasibility.ipynb',
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert 'Validated 1 notebook(s).' in result.stdout
