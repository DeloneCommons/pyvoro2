# Large-weight / large-radius native numerical failure

**Status:** `selected for research`, **pre-C++ upstream prerequisite** (R-O03), coordinated with R-O02b. A reproducible, independently checked fix and focused upstream issue should precede planned edits to pyvoro2's C++-related source layers. This is **not** an approved vendor patch; Chris's upstream merge is not a prerequisite.

## Reproducer and likely mechanism

A one-site periodic tessellation must have the full fundamental-domain measure regardless of a common power-weight offset. Earlier native 2D/3D checks with radius around `10^8` produced incorrect geometry and, with some block layouts, a missing cell.

The weighted-cut expression in [`rad_option.hh`](../../../vendor/voro++/src/rad_option.hh) uses arithmetic of the form `d² + r_i² - r_j²`. With very large, nearly equal squared radii, the small geometric term can be lost through rounding. This is a strong candidate mechanism, **not proof that every observed failure is limited to that line**.

## Investigation plan

1. Preserve minimal deterministic examples and independent geometric expected results (2D/3D, periodic/nonperiodic as relevant; ordinary versus radical).
2. Trace all squared-radius expressions in separator computation, skip/pruning checks, and weight-to-radius conversion. Check behavior under common weight shifts and block-size changes.
3. Test proposed numerically stable algebra against adversarial cases, degeneracies, and the historical pruning regression; do not treat volume sums alone as sufficient evidence.
4. For **each independently confirmed failure**, make its own standalone reproduction repository, pinning the original Voro++ revision and showing the bug before the fix and the repaired behavior after it; keep original-vs-patched test artifacts and independent oracles in that same repository. Do not bundle unrelated defects merely because they concern large weights.
5. Evaluate narrow upstream-compatible source changes and regressions (including platform/codegen variants). **After** the correction is reproduced and tested, draft/file a focused upstream Voro++ issue for Chris Rycroft, linking its reproducible repository and candidate fix. Upstream contact is an investigative step, not a promise that a particular patch is correct.
6. Choose a separately reviewed vendor-update and native build/test path if an actual pyvoro2 native change is subsequently approved; the Python-only review kit cannot validate a modified native binary.

Evidence: [R-O03: genuine power-weight range](../../../docs/development/reboot/requirements-inventory.md#r-o03) · [E06](../../../docs/development/reboot/evidence-index.md#e06) · [R-O02: vendor patching policy](../../../docs/development/reboot/requirements-inventory.md#r-o02).

**No upstream issue, PR, or repository change is authorized by this note.**
