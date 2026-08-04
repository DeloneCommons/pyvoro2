# 0009 — Certified scalar proximal solver

- **Status:** Accepted
- **Date:** 2026-08-01
- **Related issue:** [#37 — Replace the scalar proximal loop with a certified bounded solver](https://github.com/DeloneCommons/pyvoro2/issues/37)
- **Related decisions:** [ADR 0007](0007-separator-objective-contract.md), [ADR 0008](0008-separator-solver-and-linear-backend.md)
- **Related plan:** [v0.8 remediation execution plan](../plans/v0.8-remediation.md)

## Context

ADR 0007 fixes the scientific separator objective, and ADR 0008 fixes the
public method/backend selection contract. The original scalar ADMM proximal
update used 60 projected Newton steps without a bracket or final certificate.
The first R2 replacement removed that exhaustion-as-success path, but review
found three release blockers: private and public exponential paths evaluated
different complete expressions, a contribution-scale KKT tolerance could
certify a point far from the unique minimizer under large cancellation, and
fixed 180-digit transcendental arithmetic ran in every scalar iteration.

A second independent review of commit `7a012bf` found that its replacement
ordinary enclosure was also not a proof. It multiplied already-rounded
`ln(2)` limbs during exponential range reduction, assigned heuristic radii to
cancelled double-double sums, and formed Huber endpoint differences by
subtracting complete branch values. Those defects produced both a false
adjacent derivative bracket and a provably wrong endpoint choice. A
compensated center without a derived outward radius is therefore never sign
authority in this solver.

The correction must certify the existing strongly convex scalar problem. It
must not redefine the scientific objective on the binary64 lattice, tune ADMM,
add a public tolerance or schema field, or make another dependency mandatory.

## Decision

### Objective, arithmetic authority, and private ownership

For each measurement coordinate, solve

\[
F(y)=c\,\ell(y-y^{\mathrm{obs}})
+\sum_p p(y)+\frac{\rho}{2}(y-v)^2,
\qquad \ell\le y\le u,
\]

where \(\rho\) is positive and finite. Every mismatch half-factor,
confidence placement, scalar-penalty strength, reciprocal continuation, and
zero-strength convention is inherited unchanged from ADR 0007. The proximal
quadratic makes the continuous objective strongly convex and coercive, so each
valid equality, finite, one-sided, or unbounded domain has one unique
continuous minimizer.

Every binary64 input denotes its exact real dyadic value. Complete ADR 0007
source expressions are evaluated before derived intermediates are rounded. In
particular, exponential boundaries use `lower + margin - y` and
`y - (upper - margin)` as complete expressions. Constructor overlap checks,
breakpoints, vector adapters, final objective recomputation, breakdowns,
reports, and JSON follow the same rule.

The implementation is private:

- `_objective.py` owns one compiled term kernel for source values, exact branch
  classification, one-sided derivative and curvature enclosures, structural
  breakpoints, and direct term differences;
- `_numerics.py` owns the numeric binary64 lattice, twofold balls, physical and
  base-two-scaled ball operations, directed outward bounds, and stable direct-
  difference primitives;
- `_scalar_prox.py` owns compiled coordinate data, bracketing, safeguarded
  iteration, endpoint selection, bounded fallback decisions, and certificates;
  and
- `solver.py` owns vector routing, ADMM integration, observation-row metadata,
  and mapping to the existing public numerical-failure result.

Static mismatch and positive-strength penalty data are compiled once per ADMM
component solve and reused across rows and iterations. Zero-strength penalties
are omitted before their parameters or branches are evaluated. Vectorized
ordinary adapters consume the compiled kernel or shared primitives and are not
an independent objective implementation. No private type is exported.

### Domains and structural breakpoints

NaN endpoints, a reversed interval, nonpositive or non-finite `rho`, and
non-finite required scalar data are structured failures. An equality domain is
handled before iteration; its sole point succeeds only when the authoritative
objective there is finite. The received interval is authoritative and does not
reapply the hard-feasibility tolerance.

The solver records exact dyadic locations for Huber thresholds, soft
boundaries, reciprocal epsilon and margin points, and finite hard bounds.
Coincident locations are deduplicated. A nonrepresentable exact location
contributes both bracketing binary64 neighbors. Positive reciprocal margin
activations retain the one-sided derivative intervals

\[
[-s/m^2,0]\quad\text{and}\quad[0,s/m^2].
\]

Breakpoints are search and branch locations, never first-acceptable success
opportunities.

### Derivative enclosures and exact point success

At a candidate, the evaluator returns rigorous outer enclosures for both true
one-sided derivatives,

\[
g_-(y)\in[L_-,U_-],\qquad g_+(y)\in[L_+,U_+].
\]

Evaluation uncertainty is not optimizer tolerance. In particular, a threshold
proportional to the magnitudes of cancelling contributions is never a success
condition. Strong convexity may turn a rigorous upper bound
\(R_{\rm upper}\) on constrained stationarity into
\(|y-y^\star|\le R_{\rm upper}/\rho\). The distance from zero to a widened
outer derivative interval is generally a lower bound and cannot be called such
a certificate.

Point success requires a proved exact KKT sign:

- an interior smooth point or kink requires \(U_-\le0\) and \(L_+\ge0\);
- a finite lower endpoint requires \(L_+\ge0\);
- a finite upper endpoint requires \(U_-\le0\); and
- an equality domain succeeds only when its sole objective is finite.

If ordinary binary64 enclosures do not prove the condition, the bounded
high-precision fallback may resolve it. An unresolved sign is not approximate
point success.

### Principal sign bracket and ordered binary64 lattice

The principal non-point certificate is a feasible derivative-sign bracket

\[
U_+(y_{\rm lo})<0,\qquad L_-(y_{\rm hi})>0.
\]

It is maintained through finite endpoints, exact breakpoints and their
neighbors, bounded expansion on unbounded sides, safeguarded Newton proposals
inside one smooth piece, and ordered-binary64 midpoint fallback. The limits
remain 128 expansions, 128 scalar iterations, and an expansion exponent step
of eight. At a limit the solver recomputes fresh endpoint enclosures and makes
one final exact-point or adjacent-bracket check. A limit itself is never
success.

Ordered-float operations describe numeric values, not bit patterns. `-0.0` and
`+0.0` are one lattice point; adjacency and midpoint contraction are correct
through signed zero, subnormals, ordinary finite values, and finite extrema.
Success occurs when an exact point condition is proved or the certified bracket
endpoints are adjacent numeric binary64 values.

### Direct terminal objective difference

For an adjacent bracket, select the returned endpoint from the direct source
difference \(F(y_{\rm hi})-F(y_{\rm lo})\), accumulated term by term. Squared
and proximal quadratics and soft squares use factored differences; Huber and
reciprocal/tangent terms use branch-aware differences; exponential terms use a
scaled balanced difference. Complete objective totals are never subtracted and
a large common additive constant never defines the comparison tolerance.

Huber differences partition the exact source interval at its thresholds and
integrate the appropriate linear or quadratic expression on each piece; they
do not subtract two complete Huber values. Exponential differences form their
argument step from the complete source endpoint gap and use a certified
balanced `expm1` factor.

A proved positive difference selects the lower endpoint, a proved negative
difference selects the upper endpoint, and proved exact equality uses one
centrally documented binary64 round-to-nearest-ties-to-even endpoint rule.
Candidate or breakpoint order cannot choose the result. An unresolved
difference after the bounded fallback is numerical failure.

### Binary64 common path and bounded fallback

Ordinary sign authority is a ball `(high, low, radius)` whose center is the
exact real sum `high + low` and whose nonnegative outward radius includes
input uncertainty, every rounded correction, and every subsequent operation.
Physical lower and upper bounds are rounded outward. Strict signs are accepted
only when the corresponding physical bound excludes zero; non-finite centers,
radii, or unsafe scaling are unresolved. Base-two-scaled balls retain the same
invariant while comparing terms with widely separated exponents, so opposing
exponentials never materialize `inf - inf`.

Certified exponential evaluation reduces by an integer multiple of a proved
ball for `ln(2)`, evaluates a degree-20 Taylor polynomial with coefficient
balls, adds a derived Lagrange remainder, and applies directed power-of-two
scaling. The integer reduction is only a proposal; its reduced ball must prove
that the polynomial domain is valid. Balanced `expm1` differences use the same
ball operations. Derivative/curvature ratios are formed in their common scale.
A mathematically finite source value is evaluated stably. A value outside the
finite binary64 range is positive infinity and is never clipped.

The ordinary path performs no `Decimal.ln()` or `Decimal.exp()` and does
not construct `Fraction` or high-precision signed-log representations for every
term in every scalar iteration. Exact dyadic `Fraction` work is permitted at
one-time compilation for breakpoints and neighboring floats and in rare exact
branch-sign fallbacks when a binary64 predicate is cancellation-sensitive.

Fallback expressions start from exact binary64 dyadics. Purely algebraic
derivative and terminal-difference decisions use their exact `Fraction` sign.
Expressions containing exponentials use outward `Decimal` intervals: rational
arguments are rounded in both directions, monotonic exponential endpoint
bounds surround the correctly rounded result by neighboring Decimal values,
and signed terms are accumulated with directed rounding. The fallback tries 80
and then 160 digits and accepts a transcendental sign only when the interval
excludes zero. Exact equality still requires symbolic collection; agreement
between two nearest-rounded values is not a certificate. There are at most
four fallback decisions per coordinate. Exhausting either precision or
decision budget returns the existing structured numerical failure.

### Fast paths, failure evidence, and performance

Squared and Huber rows without positive-strength scalar penalties retain the
vectorized mismatch-only path. A row with only soft or reciprocal terms also
stays on that path when its projected mismatch-only candidate is in every
term's zero-valued region with zero in its subgradient. Zero-strength terms do
not enter the general solver.

Heterogeneous ordinary penalty rows use a vectorized implementation of the
same ball algebra and exact source formulas. Untrusted vector Newton values are
proposals only. Each accepted row must independently prove its derivative
signs and direct terminal difference; an exceptional or unresolved row is
routed through the scalar certified path rather than accepted by the batch
calculation. Repeated identical coordinates retain the existing scalar-result
cache. The array proof accelerator supports soft and exponential terms;
reciprocal rows are excluded before array arithmetic and use the scalar path.
Its floating-point exception policy is scoped locally, so caller
`numpy.errstate` settings cannot change routing or results. Exceptional numeric
lanes are consumed only through explicit finite/resolved masks. A whole-array
`FloatingPointError` declines all array certificates, while type, shape, and
other programming errors remain visible rather than being converted into
numeric fallback.

Private success evidence records the selected value, certificate kind, point
derivative enclosures and resolved signs or both adjacent-bracket endpoint
enclosures and resolved signs, scalar iterations, expansions, a rigorous
residual/localization bound when available, fallback count, execution path,
and objective. Adjacent certificates additionally retain the ordinary direct
endpoint-difference enclosure, its resolved sign, any outward fallback
interval or exact algebraic sign, the selection reason, and selected endpoint.
Batch-accepted rows use this same evidence-bearing result shape and record
their probe count; they are not bare float successes. Thus fallback-resolved
bracket signs and endpoint choices remain reconstructable even when an
ordinary enclosure contains zero. Failure evidence records a reason, counts,
last candidate, last finite bracket, last derivative or residual enclosure,
localization bound when available, and fallback count. `solver.py` adds
original and component-local row indices and maps failure to
`status='numerical_failure'`,
`converged=False`, no weights or objective breakdown, and unchanged
solver/backend fields. A failed proximal attempt does not increment completed
ADMM iterations. Component aggregation and the active-set wrapper preserve
their existing forwarding semantics.

The ordinary benchmark reports identical and heterogeneous penalty batches at
1, 10, 100, and 1000 rows after warm-up, using the median of repeated calls and
recording environment metadata. It instruments the actual array and scalar
entry points and records the actual unique-key count, batch-eligible and
batch-certified rows, scalar dispatch calls, high-precision fallback
decisions, structured failures, arithmetic exceptions, and observed branches.
The identical
1000-row case is gated at 1.0 second with
`T(1000) <= 12*T(100) + 0.10`. The heterogeneous 100- and 1000-row cases are
gated at 0.35 and 3.0 seconds, respectively, with
`T(1000) <= 12*T(100) + 0.30`. Every identical case must show one scalar
dispatch for its one repeated key. The 10/100/1000 heterogeneous cases must
show that every unique row was batch eligible and certified, with no scalar
dispatch, high-precision fallback, arithmetic exception, or structured
failure.

## Consequences

- Scalar solving, public evaluation, result breakdowns, reports, and JSON have
  one complete-expression objective meaning.
- Every successful coordinate proves exact point signs or localizes the unique
  continuous minimizer in an adjacent numeric-binary64 sign bracket.
- Large derivative cancellation cannot create tolerance-based false point
  success, heuristic double-double radii cannot create a false sign bracket,
  and large additive constants cannot mask endpoint selection.
- Ordinary ADMM scalar work remains on a bounded-cost binary64 path; rare
  unresolved decisions either use the bounded fallback or fail honestly.
- Public solver options, result fields, report/JSON schemas, ADMM
  decomposition, and active-set semantics do not change.

## Alternatives considered

### Retain contribution-scale KKT tolerance

Rejected because evaluation error proportional to large cancelling terms can
contain zero far from the minimizer. Such an outer interval may reject or
trigger fallback, but cannot certify approximate point optimality.

### Compare complete endpoint objectives

Rejected because subtracting totals loses a small local difference beneath a
large common additive constant. Direct source differences provide the relevant
sign without redefining the continuous objective.

### Use fixed high precision for every iteration

Rejected because it is incompatible with repeated ADMM proximal work. High
precision is bounded ambiguity handling, not the production arithmetic model.

### Use a generic SciPy optimizer or expose a tolerance

Rejected because SciPy is optional, generic status does not establish this
certificate, and numerical success criteria are package invariants rather than
public tuning parameters.

### Minimize over the binary64 lattice or accept a resource limit

Rejected because ADR 0007 defines a continuous scientific objective. Adjacent
floats localize its unique minimizer; neither the lattice nor a private limit
replaces that objective or its proof obligations.
