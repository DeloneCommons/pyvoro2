# Planar diagnostics API

Planar diagnostics share the spatial severity contract. Invalid cell areas and
closure failures are errors; expected-ID handling distinguishes standard,
power, and undeclared (`None`) modes; and explicitly requested periodic edge
reciprocity is required. Marked analyses reset analyzer-owned edge flags before
recording current failures. Strict tessellation validation consumes final
`TessellationDiagnostics.ok`; optional reciprocity findings remain nonfatal.

Standalone helpers inspect the supplied raw numerical records. They retain
generator occurrences, inspect reciprocal provenance classes without requiring
one-to-one fragment counts, and do not use a short-edge threshold as semantic
positivity. They cannot reconstruct full exact E/S cells or native internal
collapse without the private compute witness and original operands.

| Counter | Meaning |
|---|---|
| `n_edges_total` | All raw generator-edge occurrences, including records without shifts or with reciprocity checking disabled. |
| `n_edges_orphan` | Raw occurrences whose provenance class has no reciprocal class. |
| `n_edges_mismatched` | Unordered reciprocal class pairs whose numerical segment unions differ. |

In ordinary compute, orphan counts and `orphan`/`reciprocal_missing`
annotations describe raw N occurrences, including harmless collapsed artifacts.
A nonzero orphan count can coexist with `ok=True`; counts alone do not decide
error severity or semantic positivity. `reciprocal_mismatch` annotations mark
the positive-class groups with numerical union mismatches.

Compute's independent WP6 audit adds findings for exact contact-status
conflicts, nonpositive noncollapsed native records, missing or collapsed-only
positive coverage, degenerate cells and required reciprocal coverage. A
consistently nonpositive internally collapsed extra artifact is nonfatal when
positive coverage is complete. Audit resource exhaustion is an incomplete
error finding; it does not erase already attributed shifts under an action that
permits return. Hard attribution failures use the same structured planar
exception mechanism but cannot be disabled by `tessellation_check`.

After a complete E/S audit, ordinary compute also inspects reciprocal numerical
segment unions for classes positive in both ideals. The retained
`tessellation_line_offset_tol` and `tessellation_line_angle_tol` govern only
this numerical check. Native-local endpoints are exactly translated to a
common chart before conversion, without public vertices or public int64
shift materialization. No one-to-one fragment pairing is required.

The complete directional segment-pair count is preflighted against a separate
262,144-comparison limit. `WP6_NUMERICAL_AUDIT_RESOURCE` and
`WP6_NUMERICAL_AUDIT_REPRESENTATION` are error findings for incomplete numerical
inspection, independent of the exact-audit outcome. Attributed raw shifts
remain available when the requested diagnostic action permits return.
On either numerical refusal, `reciprocity_checked` and `ok_reciprocity` are
false even when the private exact audit completed.

::: pyvoro2.planar.diagnostics
:::
