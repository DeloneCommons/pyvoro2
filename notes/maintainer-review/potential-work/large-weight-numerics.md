# Large-weight / large-radius native numerical failure

**Status:** `selected for research` (R-O03). Under [ADR 0028](../../../docs/development/decisions/0028-upstream-voro-vendor-policy.md), compare each independently reproduced defect on the current vendored version and a pinned upstream `dev` commit before deciding on a minimal local backport, a safe mitigation or documented deferral. This investigation no longer blocks unrelated pyvoro2 C++ work. No vendor patch or upstream issue is approved by this note.

## Reproducer and likely mechanism

A one-site periodic tessellation must have the full fundamental-domain measure regardless of a common power-weight offset. Earlier native 2D/3D checks with radius around `10^8` produced incorrect geometry and, with some block layouts, a missing cell.

The weighted-cut expression in [`rad_option.hh`](../../../vendor/voro++/src/rad_option.hh) uses arithmetic of the form `d² + r_i² - r_j²`. With very large, nearly equal squared radii, the small geometric term can be lost through rounding. This is a strong candidate mechanism, **not proof that every observed failure is limited to that line**.

## Investigation plan

1. Preserve minimal deterministic examples and independent geometric expected results (2D/3D, periodic/nonperiodic as relevant; ordinary versus radical).
2. Trace all squared-radius expressions in separator computation, skip/pruning checks, and weight-to-radius conversion. Check behavior under common weight shifts and block-size changes.
3. Test proposed numerically stable algebra against adversarial cases, degeneracies, and the historical pruning regression; do not treat volume sums alone as sufficient evidence.
4. For **each independently confirmed failure**, compare the vendored version with a pinned upstream `dev` commit. Prefer a small standalone reproducer with the original-vs-fixed case and independent oracles if a local backport is being considered. Do not bundle unrelated defects merely because they concern large weights.
5. Evaluate narrow upstream-compatible changes and platform/codegen regressions where an approved local correction is warranted. If upstream `dev` still fails, distinguish ordinary documented numerical limits from silent wrong-geometry cases requiring a reviewed correction or sound refusal. Share focused evidence with Chris when appropriate; submitting an upstream `master` issue is not an unconditional gate.
6. Choose a separately reviewed vendor-update and native build/test path if an actual pyvoro2 native change is subsequently approved; the Python-only review kit cannot validate a modified native binary.

Evidence: [R-O03: genuine power-weight range](../../../docs/development/reboot/requirements-inventory.md#r-o03) · [E06](../../../docs/development/reboot/evidence-index.md#e06) · [R-O02: vendor patching policy](../../../docs/development/reboot/requirements-inventory.md#r-o02).

**No upstream issue, PR, or vendor code change is authorized by this note.**
