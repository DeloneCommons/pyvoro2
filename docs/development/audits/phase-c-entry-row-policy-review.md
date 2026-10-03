# Phase C entry review — bounded row-bound separator policy

- **Status:** Accepted Phase-C contract review
- **Date:** 2026-10-03
- **Gate:** [#104 — Phase C entry gate](https://github.com/DeloneCommons/pyvoro2/issues/104)
- **Parent tracker:** [#47 — Complete v0.9.0 functional and API stabilization](https://github.com/DeloneCommons/pyvoro2/issues/47)
- **Reviewed dev:** `05ae1e33999a8976cceb383168faaa43b4f43e68`
- **Accepted Checkpoint-B runtime authority:** `351f53e2e5fab8978d7c3f3d28dd7beb1b847357` / tree `7fc6a0e2ef8986981b6f2da382f94dbc28abf679`

## Outcome

Accept a **bounded Phase-C amendment** before WP10.

WP10 should implement the already planned independent separator measurement
spaces together with a proper row-bound **A+B** model-policy architecture.
Row-wise shape/robustness parameters (**C**) are not accepted wholesale in the
first implementation pass. Their final v0.9 disposition is a mandatory focused
review after WP10/WP11 and before Checkpoint C.

This is public-surface staging, not compiler staging: WP10 must remove the
assumption that one scalar-common objective specification applies to every row.

## Source conclusions that drive the amendment

The current separator compiler, proximal machinery, connectivity logic, and
active engine assume one common scalar model in several places. A+B therefore
requires more than accepting NumPy arrays in model dataclasses:

- reciprocal/exponential breakpoints and affine conversions become row-specific;
- hard feasibility/coupling must account for row-local applicability;
- component solves and active-set subsets must project model policy with rows;
- removal/re-entry must recover candidate policy;
- scalar-prox caches must include all policy capable of changing the objective;
- reports must retain model/row association separately from ADR 0014
  observation/source identity.

The existing mathematical family remains suitable: heterogeneous coefficients
still form the same affine, separable convex fixed-separator problem.

## A — accepted row-wise scientific values

WP10 accepts scalar broadcasting or one-dimensional exact-length row values for:

```text
Interval:                    lower, upper
FixedValue:                  value
SoftIntervalPenalty:         lower, upper, strength
ExponentialBoundaryPenalty:  lower, upper, strength
ReciprocalBoundaryPenalty:   lower, upper, strength
```

Arrays are positional policy until binding, defensively owned/read-only, and
strictly validated. Hard closed intervals may use equal endpoints to express
equality. This does not relax soft/boundary penalty width requirements.

`L2Regularization` remains site/weight policy and is not generalized by
observation row.

## B — accepted hard applicability / absence

`Interval` and `FixedValue` gain keyword-only scalar-or-row
`applicable=True`.

For an inapplicable hard row there is:

- no feasibility restriction;
- no hard-conflict edge;
- no hard-violation classification;
- no hard-induced structural coupling.

Mismatch confidence is not hard applicability. Very broad finite bounds are not
absence. Dropping an observation is not equivalent because it also drops
mismatch/candidate identity.

For penalties, strength zero remains exact mathematical absence and must be
recognized before dangerous branch evaluation and unnecessary solver/coupling
selection.

## Row binding and identity

A `FitModel` containing row vectors is an **unbound positional policy** until
problem construction. Construction binds one resolved owned policy to one
ordered `SeparatorObservations` set.

Every component subset, active subset, final refit, and later row re-entry must
project observations and bound policy through the same ordered selection.
Duplicate physical pairs and distinct periodic images remain distinct rows.

Model policy does not enter ADR 0014 row IDs, observation-set fingerprints, or
source binding. No second public UUID/fingerprint system is required for model
policy. Exact association and invariant checks are sufficient.

Hard policy remains **active-conditional**. It constrains a row when that row is
selected into the current fixed problem; it does not mean that a face must
exist in the realized tessellation.

## C — staged parameter-level decision

The first WP10 pass keeps these fields term-global:

```text
HuberLoss.delta
ExponentialBoundaryPenalty.margin
ExponentialBoundaryPenalty.tau
ReciprocalBoundaryPenalty.margin
ReciprocalBoundaryPenalty.epsilon
```

Before Checkpoint C, review each field separately using the implemented WP10/WP11
stack and downstream-shaped examples.

Current review disposition:

- **Huber `delta`** — genuine additional robustness behavior; confidence does
  not emulate it. First candidate for later row-wise promotion.
- **Exponential `margin`** — with fixed `tau`, largely amplitude-equivalent
  to strength in the implemented formula; weak reason to generalize.
- **Exponential `tau`** — genuine shape/decay-scale parameter; plausible
  evidence-driven later candidate.
- **Reciprocal `margin`** — genuine activation/cutoff parameter; plausible
  later candidate with low marginal branch-architecture cost after A.
- **Reciprocal `epsilon`** — explicit continuation/regularization policy;
  retain term-global by default unless a concrete heterogeneous need appears.

A few discrete penalty-shape regimes can already be represented by multiple
existing penalty instances with row-masked strengths after B. Generic per-row
term-class dispatch is not required.

## Solver and numerical contract

A+B preserves:

- convexity;
- affine hard feasibility;
- scalar proximal separability;
- direct/quadratic structure where already mathematically applicable;
- ADMM as the general constrained/nonquadratic path;
- existing gauge/component semantics;
- exact hard tolerance and conflict-witness meanings.

The implementation should retain the current scalar numerical kernels and
certificates while selecting a complete immutable scalar specification per row.
Identical complete specifications may share compilation/batching. Cache identity
must include the complete relevant row policy.

## Active-set integration

WP10 owns correct row-policy propagation through the existing advanced active
engine; WP11 later promotes the stabilized workflow to a supported facade.

The accepted final fit must use exactly the candidate policy projected through
the accepted active mask. Candidate policy survives removal/re-entry. Outer
termination remains distinct from final inner status and unavailable final
layers retain ADR 0015 semantics.

## Periodic integration

The completed Phase-B periodic implementation is a suitable basis for Phase C.
Final cells/image-qualified boundaries remain authoritative for additional and
self-image topology. Structural/resource/certificate failures continue to
propagate as structural failures rather than being converted into ordinary
"face absent" outcomes.

The Phase C amendment does not reopen the periodic N/E/S, source/backend, or
boundary-identity architecture.

## Report schema strategy

WP10 moves strict separator reports to schema version 2.

Schema v2 must represent resolved model policy aligned with ordered row IDs and
support both:

```text
uniform/scalar value
row-varying values
```

from its first implementation. This permits a later accepted C parameter to
become row-wise without another schema generation merely for that extension.

Active reports retain full candidate policy; nested fit reports retain the
projected selected policy. Observation/source identity records remain
model-independent.

## Phase-C sequence

```text
#104 entry gate
→ merge bounded contract amendment
→ WP10: spaces + A+B row-bound policy
→ independent WP10 review
→ WP11: supported realization-aware facade
→ independent WP11 review
→ mandatory C parameter-level refinement gate
→ optional narrow accepted C implementation/review
→ Checkpoint C
```

The later whole-code maintainer comprehension reread remains after WP13; it is
not pulled forward for the C decision.

## Explicitly deferred scope

This amendment does not authorize:

- per-row measurement spaces;
- arbitrary callbacks or user-defined losses;
- generic mixed observation blocks;
- generic per-row term-class dispatch;
- topology constraints or prescribed measures;
- chemistry-specific core types;
- site-coordinate fitting;
- a new inverse family.

ChemVoro may generate row arrays or penalty-instance masks from chemical
classes downstream; pyvoro2 does not need to model those classes.
