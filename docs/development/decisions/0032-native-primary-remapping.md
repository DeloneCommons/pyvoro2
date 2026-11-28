# 0032 — Exact arithmetic for numerical native-primary remapping

- **Status:** Accepted — maintainer-approved for A3-R integration;
  combined independent R/S acceptance review completed
- **Date:** 2026-11-27
- **Scope:** shared rectangular and triangular remapping and immediate wrappers
- **Narrow supersession:** ADR 0011's requirement that every remapping
  quotient and accumulated shift fit int64 before arithmetic; its remaining
  strict input, ownership and safety contract is preserved
- **Implementation:** [PR #163](https://github.com/DeloneCommons/pyvoro2/pull/163),
  squash `c23ba170d4b51ba80355c5bfc3ecd681b9ef73b9`
- **Reviewed HEAD:** `49d294163be6f40b5e77079ce246ecf3687f9953`;
  [six required CI jobs successful](https://github.com/DeloneCommons/pyvoro2/actions/runs/46071756274)
- **Related:** [ADR 0011](0011-strict-input-and-ownership-contract.md),
  [ADR 0013](0013-central-generator-preparation-and-backend-safety.md),
  [A1](0030-exact-user-lattice-and-native-admission.md),
  [A2](0031-qualified-native-periodic-qr-frame.md),
  [F1 sequencing, Draft 0.3](../plans/functional-sequencing.md)

## Context

Rounded division and large floating shear products can lose meaningful
remainders or choose incorrect lattice coefficients. Iterative coupled snapping
also duplicates final reduction. The A2 R1 correction preserves important
tangential residuals but does not resolve quotient, endpoint or underflow errors.

The F1-A3-R assignment selects a numerical primary chart, distinct from A1's
literal exact-floor user wrapping. This record captures that narrow accepted policy;
it does not activate a release or amend the historical A1/A2 decisions.

## Decision

Use canonical binary64 operands as exact dyadics for division, affine
subtraction, comparisons and integer bookkeeping. A compact shared private
integer implementation owns both rectangular dimensions and triangular native
remapping. Only emitted coordinates round to binary64, nearest-even.

For triangular rows `(bx,0,0)`, `(bxy,by,0)`, `(bxz,byz,bz)`, process z, y, x
once. At each step compute exact `q,r = divmod(residual,D)` and `v = RN(r)`:

| Condition, in order | Emitted coordinate | Committed coefficient |
|---|---|---|
| `v == D` | 0 | `q+1` |
| `eps > 0` and `v < eps` | 0 | `q` |
| `eps > 0` and `D-v <= eps` | 0 | `q+1` |
| Otherwise | `v` | `q` |

Propagate the committed coefficient through the full lattice row before
evaluating lower coordinates. Later steps never alter a finalized coordinate.
Returned coordinates satisfy `[0,bx) × [0,by) × [0,bz)`.

For each active rectangular axis, the sole period is `D = RN(hi-lo)`, matching
the existing rectangular lattice. Compute exact `q,r = divmod(x-lo,D)` and
round the complete affine value `v = RN(lo+r)`, without rounding r first.
With exact `U=lo+D`, first select `(lo,q+1)` if `v >= U` or positive epsilon
satisfies `U-v <= eps`; otherwise select `(lo,q)` if positive epsilon satisfies
`v-lo < eps`; otherwise select `(v,q)`. Require `lo <= emitted < hi`, or raise
`ValueError`. Do not introduce another exact-width period or reject a contained
result because it differs from that alternative. Inactive axes are copied
unchanged with zero shifts.

Keep the existing scale-relative epsilon default: the maximum active
rectangular period or maximum native diagonal, with no absolute floor.
Explicit epsilon must be finite and nonnegative. Require exact `2*eps < D`
on every active axis, including empty batches. Touching or overlapping snap
neighborhoods raise `ValueError`; never silently shrink epsilon or rescale.

Intermediate coefficients and affine arithmetic remain arbitrary-precision
integers. Check **final** coefficients before int64 materialization. Legacy
remapping retains that final range requirement even with `return_shifts=False`.
Private reduction returns wide coefficients to permit later A3-P composition.
Outputs retain owned float64/int64 arrays, shapes and strict validation; failure
does not mutate input or leak overflow/invalid-cast warnings.

Public composite `PeriodicCell.remap_cart` prepares one qualified frame, without
a persistent cache. Preserve the A2 QR algorithm/admission thresholds and the
existing Cartesian projection arithmetic. Internal reporting retains missing
owner NaN sentinels. Containment, duplicate and allocation safeguards are unchanged.

## Numerical and compatibility consequences

- A mathematical interior remainder that rounds to an excluded upper endpoint
  advances the numerical chart even at zero epsilon. For period 2 and negative
  minimum subnormal input, emit zero with shift zero. A1 instead retains exact
  floor -1 and its rounded endpoint view 2; its contract is unchanged.
- For `[-0.1,0.2]` and x=0.4, emit `0.09999999999999998`, shift 1. The
  represented period is authoritative. With `[2^-54,1]`, x=1 refuses at zero
  epsilon but emits the lower endpoint and shift 1 at epsilon `1e-12`.
  `[-2^60,-1]`, x=-1, zero epsilon refuses its chart gap.
- Some anisotropic frames remain A2-admissible but refuse default remapping,
  for example `diag(1,1,1e-13)`. Public remap helpers permit a smaller explicit
  epsilon; forward operations retain their default and have no new parameter.
- Repeating an accepted rectangular or native-internal remap with unchanged
  epsilon preserves coordinates and adds zero shifts. This guarantee excludes
  Cartesian QR round trips and A1 rounded views.
- `input ≈ primary + s @ L` includes final-coordinate rounding and intentional
  snapping. No universal exact reconstruction or absolute error bound is claimed.
  Remapping cannot recover phase already lost by projection of remote source
  lifts; A3-P owns that separate mechanism.

## Alternatives and review boundary

Retaining rounded quotients or floating shear products preserves known scientific
errors. Multiple retries and separate final reduction obscure which coefficient
was committed. An independent exact endpoint-width model unnecessarily rejects
ordinary decimal bounds. General exact-arithmetic machinery, special numerical
fast/fallback paths, source prelifting and automatic scaling are outside this unit.

Review requires independent Fraction/analytic regressions, preservation of R1
scientific controls, record-level unmodified P0 replay, measured preparation
cost and existing validation. The two P0 #60 source discrepancies remain A3-S.
A3-S/P, uninserted locate recovery, F2 ghost state/query work, B1/C1 and G1 remain
separate. The maintainer accepted A3-R after preliminary technical review and
successful required PR CI, with 18 to 2 P0 discrepancy records and preserved
A2 R1 predicates. The [implementation evidence](https://github.com/DeloneCommons/pyvoro2/blob/49d294163be6f40b5e77079ce246ecf3687f9953/notes/maintainer-review/contracts/f1-a3-r/README.md)
retains its original pre-acceptance status and provenance. The combined
independent R/S audit of [PR #165](https://github.com/DeloneCommons/pyvoro2/pull/165)
at `140d4f8258b017e8b5691209a8eaf99c14f9427e` confirmed the R policy without
requiring arithmetic changes; see [ADR 0033](0033-source-connectors-and-realization-reference.md)
for the acceptance record. This decision is not a claim of completed integrated
F1-A qualification, native certification or release readiness.
