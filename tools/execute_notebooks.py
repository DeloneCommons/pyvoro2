#!/usr/bin/env python3
"""Refresh stored outputs by executing repository notebooks."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Clear ordinary stored outputs, execute each notebook in a fresh '
            'Jupyter kernel, and save it after successful execution.'
        ),
    )
    parser.add_argument(
        'notebooks',
        nargs='*',
        help=(
            'optional notebook filenames under notebooks/ to refresh; '
            'defaults to every *.ipynb file'
        ),
    )
    parser.add_argument(
        '--use-src',
        action='store_true',
        help='prepend repo/src to PYTHONPATH inside each spawned kernel',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Refresh every requested notebook and report whether it changed."""

    args = _parser().parse_args(argv)
    from _notebook_tools import (
        NotebookMaintenanceError,
        iter_notebooks,
        refresh_notebook,
    )

    try:
        notebooks = iter_notebooks(args.notebooks or None)
        for notebook in notebooks:
            print(f'Executing {notebook.name}...')
            changed = refresh_notebook(notebook, use_src=args.use_src)
            print('  saved' if changed else '  unchanged')
    except NotebookMaintenanceError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f'Executed {len(notebooks)} notebook(s).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
