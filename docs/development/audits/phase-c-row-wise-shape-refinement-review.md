# Phase C row-wise shape refinement — final parameter-level decision

- **Status:** Accepted maintainer parameter-level decision
- **Date:** 2026-10-06
- **Closure baseline (live dev):** `d247f5db79006adbc3e14bb000f0693754a754db`
- **Baseline integration CI:** [run 37388664124](https://github.com/DeloneCommons/pyvoro2/actions/runs/37388664124) — successful
- **Parent tracker:** [#47 — Complete v0.9.0 functional and API stabilization](https://github.com/DeloneCommons/pyvoro2/issues/47)
- **Decision authority:** [ADR 0019](../decisions/0019-separator-measurement-spaces-and-supported-realization.md)
- **Related plan:** [active v0.9.0 plan](../plans/v0.9.md)

## Purpose and baseline

The mandatory late-Phase-C gate gives each of the five staged separator
shape/robustness parameters a final pre-Checkpoint-C disposition. This record
closes that gate against accepted and merged WP10 (#107 / PR #108), the
source-identity gate (#109 / PR #110), WP11 (#111 / PR #112), and
infrastructure/process hardening (#113 / PR #114). It records the accepted
maintainer analysis; Checkpoint C itself remains unaccepted.

The [Phase C entry review](phase-c-entry-row-policy-review.md) accepted bounded
A+B row policy and staged this separate parameter-level decision. That history
is preserved; the dispositions below are its final gate outcome.

## Final dispositions

| Parameter | Disposition | Rationale |
|---|---|---|
| `HuberLoss.delta` | **DEFER** | Genuine robustness behavior that confidence does not emulate; insufficient downstream evidence for row-wise exposure in v0.9. |
| `ExponentialBoundaryPenalty.margin` | **RETAIN TERM-GLOBAL** | With fixed `tau`, substantially duplicates the existing strength/amplitude mechanism. |
| `ExponentialBoundaryPenalty.tau` | **DEFER** | Genuine decay/shape-scale variation; insufficient downstream evidence for row-wise exposure in v0.9. |
| `ReciprocalBoundaryPenalty.margin` | **DEFER** | Genuine pair-dependent activation/boundary-layer width; insufficient downstream evidence for row-wise exposure in v0.9. |
| `ReciprocalBoundaryPenalty.epsilon` | **RETAIN TERM-GLOBAL** | Continuation/regularization should remain a common numerical/model policy within a term. |

**Recommended Phase-C C-refinement implementation: NONE.**

`DEFER` preserves a semantically meaningful possible row-wise freedom whose
downstream need is not established sufficiently to freeze it before Checkpoint
C. `RETAIN TERM-GLOBAL` is the stronger current conclusion that row-wise
exposure is not justified by the existing mechanisms or common term policy.
All five public parameters remain scalar/term-global; none accepts row arrays.

## Why no implementation is required

Implemented WP10/WP11 already provide the main chemical heterogeneity required
by the established initial ChemVoro workflow:

- row-wise separator targets and confidence;
- row-wise hard lower/upper values and applicability;
- row-wise soft/boundary lower/upper values and strength;
- multiple penalty instances with row masks through zero strengths;
- independent term-global `fraction`/`position` measurement spaces;
- realization-aware supported fitting.

`position` uses absolute spatial units; `fraction` is separator coordinate
normalized by that pair's distance, with errors in that same normalized space.
Different terms may use different spaces; this gate does not add per-row spaces.

This A+B functionality is sufficient for the currently established v0.9
downstream requirement. Additional C freedoms have no established useful
pair-type/environment dependence that warrants pre-Checkpoint-C implementation.
The narrower public surface is retained without declaring these freedoms
permanently unnecessary.

## Primary deferred candidates

- `HuberLoss.delta`
- `ReciprocalBoundaryPenalty.margin`
- `ExponentialBoundaryPenalty.tau`

All three remain **DEFER**. During the post-Checkpoint-C whole-code maintainer
comprehension / architecture-reconciliation stage, reconsider them only if
reconstructed understanding or concrete ChemVoro/downstream evidence shows a
genuine need for row-wise variation. This watchlist is not planned
implementation, authorization to create an implementation issue, or a
commitment to later promotion.

## Next gate

No optional C refinement is required before Checkpoint C. Independently review
and merge the docs-only closure PR, then update tracker #47 against merged
authority and proceed to Checkpoint C review. Whole-code recovery remains after
accepted Checkpoint C and before Phase D/WP12; this record does not start it.
