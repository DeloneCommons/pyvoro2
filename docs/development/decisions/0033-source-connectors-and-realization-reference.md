# 0033 — Original-source connectors and numerical realization image reference

- **Status:** Accepted — maintainer-approved after combined independent R/S
  acceptance audit and documentation corrections
- **Date:** 2026-11-28
- **Scope:** F1-A3-S, following integrated A3-R and accepted ADR 0032
- **Implementation:** [PR #165](https://github.com/DeloneCommons/pyvoro2/pull/165)
- **Independently reviewed HEAD:** `140d4f8258b017e8b5691209a8eaf99c14f9427e`;
  [six required CI jobs successful on the reviewed implementation](https://github.com/DeloneCommons/pyvoro2/actions/runs/46084590317)
- **Related:** [sequencing Draft 0.3](../plans/functional-sequencing.md),
  [ADR 0012](0012-certified-periodic-image-geometry.md),
  [ADR 0013](0013-central-generator-preparation-and-backend-safety.md),
  [ADR 0014](0014-separator-observation-and-source-identity.md),
  [ADR 0015](0015-atomic-separator-active-state.md),
  [A2](0031-qualified-native-periodic-qr-frame.md),
  [A3-R](0032-native-primary-remapping.md)

## Context

Remapping before separator resolution changes the source problem. In a unit
period, source x coordinates `(2^-41, 1/2)` have unique nearest shift zero,
but snapping the first coordinate to zero introduces a tie. Independently,
rounding a translated endpoint before subtraction loses explicit affine
connectors, including `1/4 - 2^-54 + 1 - 3/4`.

A source-only correction would break realization matching: source coordinates
`(1/8, 3+3/8)` require shift -3 while their prepared zero-shift contact has raw
label zero. Preparation coefficients alone are insufficient when insertion
changes stored native anchors. Exact relative correspondence is sufficient for
the existing numerical family; complete insertion-history reconstruction is
unnecessary for this task.

## Decision

1. Use original canonical binary64 endpoints and original user row basis for
   `delta = RN(p_j - p_i + s @ A)`. Inferred rows consume the existing complete
   certified nearest-image choice and displacement. Explicit rows evaluate the
   complete fixed-shift affine expression exactly before rounding; they need no
   search, inverse, native admission or representable translated endpoint.
   Preserve explicit nonnearest images, strict validation, masks, tie policy,
   budgets, typed certification failures and final int64 shift boundaries.
   Nonperiodic/domain-free zero-shift rows retain ordinary source subtraction.
2. Preserve ADR 0014: distance squared derives from canonical rounded delta,
   distance from that square, and targets from distance. Use the same geometry
   derivation for resolution and exact source verification. Inference status,
   row identity, owned arrays and the unbound direct-constructor workflow remain.
3. Extract private workers shared by public compute and a private matching
   entry. There is one preparation, one operation frame where applicable, and
   one native compute. The private matching entry captures only owned immutable
   submitted coordinates `v`, preparation coefficients `K`, the actual
   represented native lattice `L`, genuine stored anchors `a` and their presence,
   and the periodic mask. Associate anchors by validated internal ID before
   synthetic empty insertion, ID replacement or Cartesian reporting. Public
   signatures/results stay unchanged. Ordinary compute and nonperiodic matching
   do no capture work; public warning attribution remains at the caller.
   This narrowly replaces ADR 0013's statement that inverse consumers call the
   public wrapper: both now share the same worker and all its safety checks.
4. Define the effective numerical family algebraically as `E_i = v_i + K_i @ L`.
   For each observed pair require exact allowed integer membership
   `(v_j-v_i)-(a_j-a_i) = h @ L`. Zero relative difference admits zero;
   otherwise use rectangular divisibility or triangular reverse substitution,
   including exact zero on inactive axes. Use `L` from the actual native
   parameters/bounds, never the source basis or another frame.
5. Convert raw label `c` to `s_effective = c - K_j + K_i - h` before unions,
   membership, other-image classification, diagnostic reporting, measure
   selection and unaccounted-pair reporting. Cache correspondence per pair
   during collection; reversal negates it. Use Python integers throughout;
   diagnostic tuples gain no new int64 restriction. Public raw dictionaries
   retain their existing anchor-relative `adjacent_shift` values.
6. Missing required periodic labels/anchors, nonintegral correspondence or an
   inactive-axis mismatch raise `ValueError` with pair/reason details, never a
   fabricated zero or same/other-image classification. An observed pair here
   means any non-wall pair in the complete boundary maps after existing
   face/edge attribution, not just a requested observation. The matcher validates
   and transports these maps before selecting requested rows, including when
   `unaccounted_pair_check='none'`; an incompatible unrequested pair therefore
   refuses the call. A pair absent from the maps needs no anchor evidence, and
   synthetic empty records supply no genuine anchors. Nonperiodic contacts
   retain zero-shift behavior. Existing active iteration and accepted-state
   boundaries naturally propagate this refusal without a rollback framework
   or changes to ADR 0015's unrelated unavailable-fit handling.

## Consequences and compatibility

Source-bound geometry cached under the old remapped convention may fail exact
verification. Re-resolve from original points and raw observations; interpret
explicit shifts in the documented source reference. Do not silently reinterpret
stale coefficients or implement a cross-version migration framework. Inferred
status survives verification, replacement, subsetting, copies and same-version
pickle. Source/row association remains exact.

Diagnostic image labels can differ from returned raw cell labels. The coefficient
identity is exact for the effective numerical family, but does not establish
source/native Cartesian or weight equivalence, observed-boundary completeness,
existence of every source facet, or insertion history. Existing face/edge
attribution and wall/unknown-label limitations remain for G1a. Ordinary local
compute is independent of optional correspondence checks.

A1/A2/A3-R admission, mandatory safety, epsilon, native/vendor sources, search and
tie policy remain unchanged. A3-P pre-QR phase loss, F2 uninserted locate/ghost
handling, B1/C1, C2/C4 and G1 are separate. P0 success alone establishes none of
those broader claims.

## Alternatives

- Resolver-only correction: rejected because translated observations would
  incorrectly control active-set decisions through mismatched references.
- Subtract floating endpoints in another order: rejected; it cannot generally
  evaluate a complete affine expression with one rounding.
- Use preparation coefficients without stored anchors, or round relative
  coefficients to integers: rejected; native chart transitions can be integral
  or nonintegral in the represented lattice.
- Capture callbacks, global state, repeated native runs or a public provenance
  field: unnecessary for a bounded one-operation consumer.
- Reconstruct source facets or certify all native insertion history: exceeds
  the selected contact-family convention and A3-S scope.

## Validation boundary

Independent Fraction/analytic source and membership tests, compiled translated
2D/3D controls, refusal propagation through both active-state boundaries,
immutable operation capture, unchanged public signatures, record-level replay of
the original P0 and the default suite are required. See the
[implementation evidence](https://github.com/DeloneCommons/pyvoro2/blob/140d4f8258b017e8b5691209a8eaf99c14f9427e/notes/maintainer-review/contracts/f1-a3-s/README.md).

The combined independent R/S audit (2026-11-28) found no Blocking or Important
production defect. A fresh current native build passed 940 focused repository
tests, 93 additional source/identity cases and independent arithmetic and
geometric controls, including sheared/signed bases. All 64 configurations in a
bounded ordinary-use matching survey succeeded. Independent replays of the
unchanged P0 script reproduced 27 → 18 → 2 → 0 discrepancy records, preserving
original oracles and previously passing predicates. The audit required only
documentation corrections (durable evidence link and complete-map refusal
scope); the migration example was optional. This decision accepts the bounded
R/S implementation, not A3-P, integrated F1-A qualification, source-complete
contacts or release readiness.
