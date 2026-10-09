# Large-weight / large-radius native numerical failure

**Status:** confirmed failure class; exact root cause and patch scope under investigation. Candidate for a focused upstream Voro++ report before further native-dependent v0.9 development, **not** an approved vendor patch.

## Reproducer and likely mechanism

A one-site periodic tessellation must have the full fundamental-domain measure regardless of a common power-weight offset. Earlier native 2D/3D checks with radius around `10^8` produced incorrect geometry and, with some block layouts, a missing cell.

The weighted-cut expression in [`rad_option.hh`](../../../vendor/voro++/src/rad_option.hh) uses arithmetic of the form `d² + r_i² - r_j²`. With very large, nearly equal squared radii, the small geometric term can be lost through rounding. This is a strong candidate mechanism, **not proof that every observed failure is limited to that line**.

## Investigation plan

1. Preserve minimal deterministic examples and independent geometric expected results (2D/3D, periodic/nonperiodic as relevant; ordinary versus radical).
2. Trace all squared-radius expressions in separator computation, skip/pruning checks, and weight-to-radius conversion. Check behavior under common weight shifts and block-size changes.
3. Test proposed numerically stable algebra against adversarial cases, degeneracies, and the historical pruning regression; do not treat volume sums alone as sufficient evidence.
4. Decide whether a narrow upstream-compatible patch suffices or wider changes are needed. Prepare a concise upstream report and, if supported by evidence, a candidate patch for Chris Rycroft to evaluate and adapt.
5. Choose a separately reviewed vendor-update and native build/test path; the Python-only review kit cannot validate a modified native binary.

Evidence: [R-O03: genuine power-weight range](../../../docs/development/reboot/requirements-inventory.md#r-o03) · [E06](../../../docs/development/reboot/evidence-index.md#e06) · [R-O02: vendor patching policy](../../../docs/development/reboot/requirements-inventory.md#r-o02).

**No upstream issue, PR, or repository change is authorized by this note.**
