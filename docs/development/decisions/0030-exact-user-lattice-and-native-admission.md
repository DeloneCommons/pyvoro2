# 0030 — Exact user lattice with separate native admission

- **Status:** Proposed — acceptance requires explicit maintainer approval of the A1 PR
- **Date:** 2026-11-26
- **Issue:** [F1-A1 #155](https://github.com/DeloneCommons/pyvoro2/issues/155)
- **Proposed narrow supersession:** ADR 0011's `PeriodicCell` construction
  handedness and numerical-admission placement only
- **Related:** [ADR 0011](0011-strict-input-and-ownership-contract.md),
  [ADR 0010](0010-native-construction-preconditions.md),
  [ADR 0013](0013-central-generator-preparation-and-backend-safety.md),
  [ADR 0027](0027-v0.9-reboot-from-v0.8.md),
  [ADR 0029](0029-image-qualified-geometry-and-optional-global-complex.md),
  [F1 sequencing](../plans/functional-sequencing.md)

## Context

ADR 0011 requires a positive floating determinant, right-handed rows and
constructor-level numerical conditioning thresholds. That historical decision
remains intact. Issue #155 authorizes a focused implementation for review:
mathematical user-lattice validity must be distinguished from the narrower
admission policy of the current Voro++ adapter. P0's known frame and remapping
defects remain targets of A2/A3; A1 does not qualify or replace those algorithms.

## Proposed decision

1. Preserve canonical owned finite binary64 rows and origin with row convention
   `x = o + f @ A`. Mathematical validity is exactly `det(A) != 0`, either sign.
   Never reorder, flip, rotate or reduce the public rows. Strict input categories
   remain ADR 0011's; `from_params` still requires positive diagonal lengths.
2. Add only the four provisional methods documented in the
   [domain reference](../../reference/domains.md). Solve and affine expressions
   use exact rational values of canonical binary64 operands. Wrapping floors
   exactly, including exact `x-o`; Cartesian wrapping rounds `x - floor(f) @ A`
   without composing public rounded converters. Owned float64 views round to
   nearest-even and can show a fractional endpoint of one without changing the
   shift. Do not promise float-view idempotence or commutativity of converters.
3. Keep internal shifts as arbitrary-size integers. Check signed-int64 range
   before materializing requested shifts only. A nonfinite required view or
   requested shift overflow raises operation-level `ValueError`; the lattice
   remains valid. Inputs/outputs have strict `(n, 3)` shape, including `(0, 3)`.
   Pure helpers need no native module, numerical frame or native admission.
4. Relocate the old native safeguards to both public frame preparation and
   detached snapshot preparation. Require right-handed rows, finite positive
   numerical lengths/determinant, finite positive singular values, reject
   `relvol < 1e-12` or `cond > 1e15`, and warn for `cond > 1e10`.
   Failed numerical preparation is an adapter error. Existing parameter,
   containment, duplicate, arithmetic, allocation and C++ direct-call backstops
   remain. Forward operations prepare one detached snapshot and warn once there;
   legacy helpers warn on frame preparation (a composed remap may prepare more
   than once). Warning-as-error now stops native use rather than construction.
5. Cache only bounded immutable exact basis/solve data; prepare the rational
   inverse lazily for user inverse-coordinate operations. No new mandatory
   dependency, public exact type, lattice class or second search solver.

## Consequences and retained boundaries

Signed, near-singular and extreme-scale lattices can construct and use pure
helpers where the requested views are representable. Existing backend helpers
retain their numerical meanings and reject unsupported native use before
unsafe dispatch. The private periodic-image kernel has an independent positive-diagonal/right-handed admission boundary; this proposal does not change
its decisions, tie rules or resource budgets. Even an admitted native frame
can encounter that consumer's existing controlled refusal.

The current Gram–Schmidt frame, remap algorithms and #60/#65/#67 P0 observations
are unchanged. This decision does not authorize A2 QR/parity work, A3 remap or
source-consumer repairs, B1 reduction, matching changes, broader 2D domains,
native/vendor changes or any release/merge acceptance. The historical v0.8
inventory and ADR 0011 are not rewritten to imply prior approval.

## Alternatives considered

- Keep constructor-level native admission: prevents legitimate exact user-only
  operations and mistakes numerical feasibility for mathematical validity.
- Relax the native checks or import historical WP2 wholesale: violates A1's
  safety and review boundary; frame and downstream support need separate work.
- Use floating inverses/floors or clamp rounded endpoints: changes discrete
  shifts near seams and under cancellation; conflicts with the exact contract.

## Evidence required for acceptance

Independent Fraction/int determinant, solve, floor and reconstruction tests;
nearest-even seam/range/cancellation/ownership cases; all native route refusals;
compiled ordinary/power volume and owner controls; unchanged P0 replay; broad
2D/3D/API regression, strict documentation build and remote CI results reported
separately. Independent mathematical and native-safety review and explicit
maintainer approval remain required.
