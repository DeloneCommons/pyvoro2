# Static quadratic separator benchmarks

`benchmark_sparse_separator.py` characterizes the optional sparse-direct path
on deterministic, molecular-shaped k-nearest-neighbor graphs. It measures
matrix assembly, direct solve time, complete public-fit time, matrix storage,
and gauge-invariant dense/sparse agreement.

The generated coordinates, external IDs, locality rows, and compatible target
weights come from `examples/static_separator_cases.py`. The same generator
drives the 32-site CI-scale downstream regression in
`examples/paper_regressions.py`, so correctness and optional scaling runs use
one deterministic input definition without making tests depend on timing.

Install the optional backend and run the complete suite from the repository
root:

```bash
python -m pip install -e ".[sparse]"
python benchmarks/benchmark_sparse_separator.py --repeat 3
```

Use repeated `--case` options for a subset. The small, medium, and disconnected
cases execute both backends. The large case deliberately avoids allocating the
dense matrix; its output still reports the full dense operator's
`8 * n_sites**2` byte storage estimate and the sparse normal-equation residual.

For the larger optional static-only workflow case without the other timings:

```bash
python benchmarks/benchmark_sparse_separator.py --case large_knn --repeat 1
```

## Recorded issue-#17 evidence

The complete harness was run with `--repeat 3` on 2026-07-21 under WSL2 on an
AMD Ryzen 7 7730U (8 cores/16 threads), Python 3.11.7, NumPy 1.26.4, and SciPy
1.11.4. Each timing is the best of three runs. Times are evidence for this
machine, not cross-platform performance guarantees.

| Case | Sites | Rows | Components | Dense assembly | Sparse assembly | Dense solve | Sparse solve | Dense fit | Sparse fit | Dense matrix | Sparse matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small kNN | 64 | 272 | 1 | 0.07 ms | 0.87 ms | 0.14 ms | 0.17 ms | 1.92 ms | 2.47 ms | 0.031 MiB | 0.007 MiB |
| medium kNN | 384 | 1,587 | 1 | 1.62 ms | 0.96 ms | 1.29 ms | 0.57 ms | 13.47 ms | 6.77 ms | 1.125 MiB | 0.042 MiB |
| large kNN | 4,096 | 16,863 | 1 | not allocated | 3.71 ms | not run | 5.15 ms | not run | 66.00 ms | 128 MiB estimated | 0.448 MiB |
| disconnected kNN | 1,024 | 4,244 | 4 | 1.72 ms | 1.34 ms | 8.50 ms | 1.15 ms | 30.94 ms | 16.34 ms | 8 MiB | 0.113 MiB |

The small complete fit favored dense execution. Sparse became faster in the
medium case and reduced matrix storage by about 27 times there. At 4,096 sites,
the sparse matrix used about 0.45 MiB while the dense matrix alone would use
128 MiB; dense allocation and solving were intentionally omitted. Dense and
sparse edge differences agreed to `7.4e-16` or better in compared cases,
objective totals agreed to `3.1e-18` or better, and sparse infinity-norm normal
residuals were below `5.1e-15` across all four cases.

## Scope

These are fixed-site, fixed-observation, static fits. The harness does not
provide or measure trajectory processing, reuse across molecular-dynamics
frames, parallel tessellation, GPU/distributed execution, or scalable all-pairs
observation construction.
