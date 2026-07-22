# Preferred-API integration and regression examples

These repository-owned scripts exercise the v0.7 public API the way a
downstream scientific package can use it. They are chemistry-neutral and make
no interpretation of backend radii as fitted scientific quantities.

- `chemvoro_workflow.py` is the canonical small periodic workflow. It keeps
  downstream metadata in an external-ID-keyed sidecar, resolves separator
  observations by those IDs, fits mathematical weights, computes the forward
  diagram from `weights=`, handles empty cells in input-aligned result arrays,
  inspects periodic boundaries and realization diagnostics, and exports
  ID-labelled records and reports.
- `paper_regressions.py` runs the compact deterministic scientific ladder:
  exact connected recovery, disconnected and zero-confidence behavior, a hard
  contradiction witness, requested/wrong/unrealized boundary support,
  algebraic fit without realization, active-set path diagnostics,
  weight/radius representation equivalence, and dense/sparse agreement.
- `static_separator_cases.py` is the single deterministic source for the
  molecular-shaped locality inputs shared by the CI-scale sparse regression and
  the optional large benchmark.

From the repository root, run the NumPy-only workflow and ladder with:

```bash
python examples/chemvoro_workflow.py
python examples/paper_regressions.py --skip-sparse
```

Install the optional backend to include dense/sparse downstream validation:

```bash
python -m pip install -e ".[sparse]"
python examples/paper_regressions.py
```

The large optional static scaling case remains in
`benchmarks/benchmark_sparse_separator.py`; see `benchmarks/README.md`. These
assets do not claim trajectory processing, repeated-frame throughput, warm
starts, all-pairs scaling, parallel tessellation, or GPU/distributed execution.

The import ownership and lifecycle used by these scripts are summarized in `docs/guide/choosing-api.md`; v0.6.3 callers should use `docs/guide/migration-v0.7.md`.
