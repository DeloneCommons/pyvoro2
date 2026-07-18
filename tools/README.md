# Tooling helpers

This directory contains repository-maintenance helpers used in local
publishability checks and CI.

Main entry points:

- `python tools/export_notebooks.py` — regenerate `docs/notebooks/*.md` from
  the source notebooks in the repo-root `notebooks/` directory.
- `python tools/check_notebooks.py` — validate notebook JSON and execute
  notebook code cells against the installed `pyvoro2` package in the current
  environment. Pass `--use-src` only in a wheel-overlay developer setup where
  the compiled extensions are already available beside `src/pyvoro2/`.
- `python tools/gen_readme.py` — regenerate `README.md` from the MkDocs source.
- `python tools/release_check.py` — run the combined local release-preparation
  checks.
- `python tools/check_dist.py dist` — verify that built sdists and wheels
  contain the expected key files.

The current notebook checker executes cell source as plain Python and discards
stdout; it does not update stored notebook outputs. The active v0.7
[#20](https://github.com/DeloneCommons/pyvoro2/issues/20) work replaces this
with clean Jupyter-kernel execution and adds a separate refresh tool after the
notebooks migrate to the preferred v0.7 API. Markdown export remains
non-executing.

For a full local pre-release pass after installing the project with all optional
extras, run:

```bash
pip install -e ".[all]"
python tools/release_check.py
```
