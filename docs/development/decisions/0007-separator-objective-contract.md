# 0007 — Separator inverse objective contract

- **Status:** Accepted
- **Date:** 2026-07-28
- **Related issue:** [#36 — Correct and freeze the separator inverse objective contract](https://github.com/DeloneCommons/pyvoro2/issues/36)
- **Related plan:** [v0.8 remediation execution plan](../plans/v0.8-remediation.md)

## Context

The separator inverse implementation reached v0.8 with incompatible objective
conventions. Direct evaluation reported squared mismatch and L2 terms without
one-half factors, while the quadratic normal system already represented the
conventional half-factor objective. The Huber quadratic branch used the
half-factor convention, but the squared-loss ADMM proximal update did not.
Reciprocal values clipped at `epsilon` while their derivatives retained the
unclipped reciprocal formulas. Zero-strength penalties still changed solver
selection and graph coupling, hard-bound checks used one fixed absolute
tolerance, and non-finite objective values could be associated with successful
termination metadata.

These discrepancies affect scientific interpretation, backend equivalence,
reports, and downstream reproducibility. Issue #36 approves a single
prerelease correctness contract before the scalar proximal algorithm is
redesigned in the next remediation workstream.

## Decision

For row \(r\), define the measurement-space residual

\[
e_r(w)=\beta_r+\alpha_r(w_{i_r}-w_{j_r})-y_r^{\mathrm{obs}}.
\]

Row confidence \(c_r\) multiplies only the mismatch loss. It does not multiply
scalar penalties or hard restrictions.

### Mismatch and L2 normalization

Squared mismatch is

\[
\ell_{\mathrm{sq}}(e)=\frac12e^2,
\qquad
\ell_{\mathrm{sq}}'(e)=e,
\qquad
\ell_{\mathrm{sq}}''(e)=1.
\]

Huber mismatch with threshold \(\delta>0\) is

\[
\ell_\delta(e)=
\begin{cases}
\frac12e^2, & |e|\le\delta,\\
\delta\left(|e|-\frac12\delta\right), & |e|>\delta.
\end{cases}
\]

Consequently, Huber loss is exactly the squared loss whenever all evaluated
residuals stay in its quadratic branch.

These formulas are evaluated in a scale-safe order. A zero-confidence row
returns exact zero value and derivatives before forming its residual. Positive
confidence is incorporated before a residual square or a potentially
overflowing subtraction is formed. The quadratic scale is assembled as
\(\sqrt{c_r}\sqrt{1/2}\), so applying the half factor cannot erase the minimum
positive binary64 confidence before its square root is taken. Huber branches
are evaluated explicitly so the inactive quadratic branch is not executed. A
mathematically non-representable binary64 result remains non-finite; finite
weighted results are not discarded because an unweighted intermediate exceeds
binary64 range. Objective mismatch values and residual reporting evaluate
\(\beta+\alpha w_i-\alpha w_j-y^{\mathrm{obs}}\) as one affine linear
combination rather than subtracting the target from an already rounded
prediction. The mismatch path applies its confidence and loss scaling to that
complete affine expression directly, so its finite value does not require the
unscaled residual itself to fit binary64. The stored public prediction or
residual may therefore be non-finite even when the directly evaluated mismatch
is finite. The Huber linear branch likewise evaluates its complete scaled
expression without subtracting separately overflowing terms. Ordinary affine
rows remain vectorized. For mixed-sign rows, a forward-error bound based on the
sum of absolute terms is compared with
\(64\epsilon_{64}\) times the result magnitude, matching the relative scale
resolved by the hard-row policy. Rows that cannot meet that relative-accuracy
target use vectorized error-free product/sum compensation for the complete
expression. Compensation is used only when the operands, products, correction
terms, and result have enough exponent margin for every raw Dekker product to
remain normal; otherwise the row uses the exact binary64-input fallback before
executing those products. The ordinary pass constructs each affine product and
forward-error bound once. For a representable compensated residual, squared
mismatch is formed as the scale-safe complete product
\(0.5c_re_r^2\) from that same reported residual. Exact exceptional rows and
non-representable residuals instead form the weighted square from the exact
affine operands.

For strength \(\lambda\ge0\) and reference \(w^{\mathrm{ref}}\), L2
regularization is

\[
\frac{\lambda}{2}\lVert w-w^{\mathrm{ref}}\rVert_2^2.
\]

Its gradient is \(\lambda(w-w^{\mathrm{ref}})\), its Hessian contribution is
\(\lambda I\), and the quadratic normal system remains

\[
\rho_r=c_r\alpha_r^2,
\qquad
q_r=c_r\alpha_r(y_r^{\mathrm{obs}}-\beta_r),
\]

\[
A=B\operatorname{diag}(\rho)B^\mathsf{T}+\lambda I,
\qquad
b=Bq+\lambda w^{\mathrm{ref}}.
\]

A reference supplied with \(\lambda=0\) contributes nothing to the objective
and does not identify component offsets. The documented output
gauge/alignment policy may still use that reference to select a representative.
Zero strength is checked before subtracting the reference. Positive L2 is
evaluated by scaling weight/reference differences before a stable
sum-of-squares reduction.

The public row values \(\rho_r\) and \(q_r\) are constructed directly with
scale-safe products for graph and operator diagnostics. Implementations must
not reconstruct \(q_r\) as \(\rho_r z_r^{\mathrm{obs}}\): the implied
difference \(z_r^{\mathrm{obs}}\) remains useful diagnostics but may be outside
binary64 range even when the normal system is finite. Solver internals do not
depend on the rounded public \(\rho_r\) and \(q_r\) as their only
representation. The scaled augmented least-squares problem is the canonical
numerical model:

\[
\min_w \frac12\left\|
\begin{bmatrix}
\operatorname{diag}(\sqrt{c})\,\operatorname{diag}(\alpha)B^\mathsf{T}\\
\sqrt{\lambda}I
\end{bmatrix}w-
\begin{bmatrix}
\operatorname{diag}(\sqrt{c})(y^{\mathrm{obs}}-\beta)\\
\sqrt{\lambda}w^{\mathrm{ref}}
\end{bmatrix}
\right\|_2^2.
\]

The augmented formulation avoids explicitly squaring the condition number and
preserves minimum-positive confidence and L2 terms through their square roots.
Each connected component receives exact power-of-two variable and objective
scales before its design or right-hand side is assembled. The error-free high
and low parts of \(y^{\mathrm{obs}}-\beta\) are carried as separate
right-hand-side contributions, so target-offset cancellation is not lost
before the solve. Any mathematically nonzero scaled coefficient or
right-hand-side term that disappears from binary64 is treated as an unsupported
component rather than silently removed from the objective.

Without positive L2, the internal solve fixes the site with the largest scaled
weighted degree and solves the remaining site contrasts. This gauge choice
reduces avoidable forward error on multiscale graphs; one final uniform shift
restores the documented public component gauge. With positive L2, variables are
centered around the reference mean but all site degrees of freedom are retained;
the augmented \(\sqrt{\lambda}I\) rows therefore represent both contrast and
mean directions directly. The implementation does not alter selected
coordinates to force an exact floating-point reference sum. It accepts the
nearest output only when objective and stationarity certificates show that the
rounding is immaterial; otherwise it returns the existing structured
``numerical_failure`` status.

The augmented design is the authoritative numerical representation. For an
ordinary component whose conservative design-condition bound is below one half
of the inverse square root of binary64 epsilon, the direct quadratic solver may
first form a guarded dense or sparse normal system as a performance candidate.
That candidate uses the same power-of-two scales and high/low right-hand-side
parts. Any lost curvature, factorization error, failed source-objective
certificate, or failed stationarity certificate discards the candidate and
falls back to the unsquared augmented solve. This fast path is therefore not a
second objective contract and is never used to rescue an unsupported
multiscale system.

Outside that guarded ordinary path, dense components use QR with SVD
information and sparse components use a sparse KKT factorization of the same
unsquared augmented system. Candidate generation never crosses the selected
linear backend and has no site-count backend threshold. ADMM uses the selected
backend for both its optional direct warm start and every weight-system solve.
An explicitly requested ADMM solve of a purely squared-plus-L2 objective
actually executes ADMM; its final iterate must pass the same quadratic
certificate as a direct candidate. Supported numerical warm-start failure
remains nonfatal, while missing SciPy after explicit sparse selection is not
swallowed.

Successful quadratic output is certified after scale reversal and public gauge
canonicalization under the universal rule below. Floating objective underflow
is not proof of zero. Exact-zero acceptance checks every positive-confidence
affine residual exactly and, with positive L2, every coordinate against its
reference. Every other candidate must pass a conservative forward
objective-gap upper bound. Bounded exact-input helpers may improve a candidate
or establish a tighter gradient bound, but their absence never weakens the
success certificate. Unsupported output resolution therefore produces the
structured ``numerical_failure`` status rather than a false ``optimal`` result.

When the bounded exact helper is available, its rational objective-gap
calculation implements that same rule; it does not introduce a second
``best binary64`` optimization problem. Coordinatewise rounding of an exact
continuous optimum is only another candidate-generation step and is not a
proof of optimality on the coupled binary64 lattice. The exact helper therefore
accepts a nonzero candidate only when its exact continuous forward gap satisfies
the same relative bound against the returned candidate's source objective. In
particular, if the continuous optimum is exactly zero but no returned binary64
vector makes every source term exactly zero, the fit is a structured
``numerical_failure`` even when coordinatewise rounding is the nearest obvious
representation.

The active-set wrapper may preserve a previous iterate's offsets only across
true zero-L2 gauge components. Such a post-fit alignment is accepted only when
exact binary64-input arithmetic proves that every component weight difference
is unchanged. Positive L2 removes the component-constant gauge, so no
post-certification active-set alignment is applied in that case.

For Huber mismatch without scalar penalties or hard bounds, ADMM compares
candidates through the authoritative Huber-plus-L2 objective and retains a
materially better candidate; an exactly zero-objective warm start is always
retained because nonnegativity certifies it as globally optimal.

Normal right-hand sides and operator products use vectorized incidence
accumulation at well-conditioned sites. Mixed-sign sites whose forward-error
bound reaches \(64\epsilon_{64}\) times the result magnitude, and sites with
range-risking partial sums, use an exactly rounded finite-input sum.
Squared-loss proximal weighted averages use the same \(64\epsilon_{64}\)
complete-expression criterion whenever either the
mathematical numerator or barycentric evaluation is ill-conditioned,
normalized weights would underflow, or positive weights are sufficiently
imbalanced to make their normalized arithmetic exceptional. Product range
checks account for all operand exponents and route potentially subnormal
products through exponent-scaled arithmetic. A sum of products also falls
back to its complete exact expression if finite nonzero factors produce a term
that rounds to zero before aggregation.
The Huber linear proximal shift evaluates \(c_r\delta/\rho\) as one stable
ratio-product rather than forming \(1/\rho\) first. Its proximal region is
selected from the unnormalized signs of
\(\rho(v-y^{\mathrm{obs}}+\delta)+c_r\delta\) and
\(\rho(v-y^{\mathrm{obs}}-\delta)-c_r\delta\), evaluated as complete
sum-of-products expressions rather than comparisons with rounded values of
\(y^{\mathrm{obs}}\mathbin{\pm}\delta\). When a nonzero exceptional test lies
below binary64 range, its exact binary64-input sign is retained instead of
converting its magnitude to zero. Direct Huber values and derivatives use the
same exact comparison of the complete residual with
\(\mathbin{\pm}\delta\), so rounding a residual onto a branch boundary cannot
change the selected derivative.

### Existing scalar-penalty strengths

`SoftIntervalPenalty` retains the value

\[
s\left(\max(a-y,0)^2+\max(y-b,0)^2\right).
\]

`ExponentialBoundaryPenalty` retains the value

\[
s\left[
\exp\left(\frac{a+m-y}{\tau}\right)
+\exp\left(\frac{y-(b-m)}{\tau}\right)
\right].
\]

Their strengths are not rescaled to follow the mismatch and L2 half-factor
convention. Positive soft-interval values and first derivatives apply strength
while forming the active-boundary displacement, so a finite scaled quadratic
term is not lost to an overflowing unscaled difference. Stable
positive-strength exponential evaluation is defined by ADR 0009. Objective
breakdowns derive soft-interval displacements directly from
\(\beta+\alpha w_i-\alpha w_j\) and the active boundary, rather than from a
rounded stored prediction.

All binary64 inputs denote their exact real dyadic values. A compound source
expression is evaluated as that complete expression before a derived
intermediate is rounded. In particular, the exponential numerators are
\(a+m-y\) and \(y-(b-m)\); implementations must not first store rounded
binary64 values for \(a+m\) or \(b-m\). The same complete-expression rule
governs validation, branch predicates, breakpoints, scalar solving, direct and
affine objective evaluation, result breakdowns, reports, and JSON. A
mathematically finite source value is evaluated in a range-safe order. A value
outside the finite binary64 range is positive infinity, not a clipped finite
surrogate.

One compiled private term kernel owns scalar-penalty values, one-sided
derivative and curvature enclosures, structural branches and breakpoints, and
direct candidate differences. Vectorized ordinary adapters may batch those
shared primitives, but are not a second objective definition. ADR 0009 fixes
the numerical enclosure, certification, fallback, and performance contract for
that kernel.

For one inward boundary distance \(d\), reciprocal repulsion is

\[
q(d)=
\begin{cases}
0, & d\ge m,\\
s\left(\dfrac1d-\dfrac1m\right), & \epsilon<d<m,\\
s\left[
\left(\dfrac1\epsilon-\dfrac1m\right)
-\dfrac{d-\epsilon}{\epsilon^2}
\right], & d\le\epsilon.
\end{cases}
\]

The full interval penalty is \(q(y-a)+q(b-y)\). The last branch is the finite
convex tangent continuation of the reciprocal branch. At \(d=\epsilon\), the
linear branch applies; at \(d=m\), the inactive branch applies. Value and first
derivative are continuous at \(\epsilon\). Value is continuous at \(m\), but
the activation point has a derivative jump and no unique derivative.

The reciprocal parameters require \(b>a\), \(m>0\),
\(0<\epsilon<m\), \(2m\le b-a\), and \(s\ge0\).
Value, first derivative, and second derivative are evaluated separately while
sharing this branch classification. Reciprocal powers are implemented as
scale-safe quotient products rather than by first forming
\(\epsilon^2\), \(d^2\), or \(d^3\). Reciprocal differences and the complete
tangent continuation are evaluated as combined ratios, so individually
overflowing quotients are not subtracted as infinities. Objective breakdowns
likewise derive reciprocal inward distances directly from the complete affine
expression and each boundary. If a rounded scalar or complete-affine distance
is not safely separated from \(\epsilon\) or \(m\), the branch is selected
from the exact binary64 operands before its requested value or derivative is
evaluated.
Exact exceptional arithmetic uses distinct range conversions: approximate
hard-bound endpoints may saturate to finite binary64 endpoints, while a
genuinely non-representable objective value converts to infinity and is never
clipped to the largest finite float.

### Zero strength

A scalar penalty with strength zero is mathematically absent. Its value and
derivatives are exactly zero without evaluating exponentials, reciprocals, or
other branch expressions. It does not force ADMM, hide the quadratic operator,
couple observation components, or change the fitted solution. Reports may
retain an exactly zero named component.

### Hard-bound tolerance and reporting

Hard-bound classification uses one float64 absolute-plus-relative policy:

\[
\operatorname{tol}(v_1,\ldots,v_k)
=10^{-12}
+64\,\epsilon_{64}\max_j|v_j|,
\]

where \(\epsilon_{64}\) is NumPy's float64 machine epsilon. For lower bound
\(a_r\), prediction \(y_r\), and upper bound \(b_r\),

\[
v_r=\max(a_r-y_r,\ y_r-b_r,\ 0),
\qquad
t_r=\operatorname{tol}(a_r,y_r,b_r).
\]

The row is satisfied exactly when \(v_r\le t_r\). Non-finite lower,
prediction, or upper values never satisfy a row.

The finite measurement values accepted by this predicate form one closed
interval. The feasibility precheck first derives that tolerance-expanded
measurement interval, then maps its endpoints through
\(y_r=\beta_r+\alpha_r z_r\) into difference bounds. Bellman–Ford operates on
those mapped bounds without applying the measurement-space absolute tolerance
a second time to path distances in weight-difference units. Thus precheck and
final classification describe the same accepted measurement set. ADMM projects
its measurement coordinate onto this same tolerance-expanded interval, while
raw violation and tolerance reporting continue to use the original user
bounds.
When an approximate expanded endpoint lies outside binary64 range, its private
hard-bound representation saturates to the corresponding finite endpoint
without performing an overflowing float addition. This saturation is only an
accepted-set construction detail; it is not used for objective values.
The relative tolerance term is also formed through scale-safe multiplication,
so a subnormal term that cannot affect the final absolute-plus-relative sum
does not emit an underflow warning.

`hard_max_violation` remains the maximum raw \(v_r\);
`hard_max_tolerance` is the maximum tolerance actually used, or zero when no
hard-bound row exists. Solver convergence tolerances are separate and remain
configurable as before.

### Finite results and warm starts

No solver-produced result may claim `status='optimal'` or
`converged=True` with a non-finite reported soft-objective component or total.
Solver paths convert such outcomes to the existing structured
`numerical_failure` result. The public result builder rejects requests to
package a falsely optimal or converged non-finite soft objective. Direct
objective evaluation may still return positive infinity for hard
infeasibility or a genuine extended-real objective.

For a hard-constrained native solve, primal/dual convergence is necessary but
not sufficient for success. Final predicted measurements must also satisfy the
authoritative hard-row predicate. Otherwise iteration continues, and
exhaustion of `max_iter` returns the existing non-success status.

Residual and ADMM convergence summaries use scale-safe Euclidean norm,
sum-of-squares, RMS, and mean-absolute reductions so a finite statistic is not
reported as infinity merely because an intermediate square overflowed.
Mean-absolute reduction only forms normalized ratios whose exponents can
affect the rounded result; irrelevant ratios below the normal range are
omitted without executing an underflowing division.

The direct ADMM warm start is optional acceleration. Its supported
linear-algebra failure falls back to the existing reference or zero
initialization; unrelated exceptions are not swallowed.

### Universal quadratic success certification

Candidate generation and success certification are separate responsibilities.
Dense and sparse normal-equation candidates, dense and sparse augmented
least-squares candidates, bounded exact helpers, and a final quadratic ADMM
iterate all represent the same authoritative source objective. One common
certifier decides whether the final public binary64 weight vector may be
reported as ``optimal``.

Certification runs after internal scaling is reversed and the documented public
gauge is restored. It recomputes the source objective from the original
confidence values, affine coefficients, targets, L2 strength, and L2 reference.
A floating evaluation equal to zero is not itself a proof: exact-zero acceptance
requires every active affine residual, and every positive-L2 displacement, to be
exactly zero for the returned binary64 values.

Every accepted nonzero candidate requires a conservative upper bound on

\[
f(w)-f(w^\star),
\]

where \(w^\star\) is the continuous optimum. For a gauge-fixed augmented
least-squares problem this bound is derived from the authoritative source
gradient and a defensible lower bound on the smallest singular value of the
augmented design. A small stationarity quantity or a lower bound on objective
excess may reject a candidate or support diagnostics, but is not sufficient for
acceptance. The forward-gap requirement applies uniformly to guarded normal,
dense augmented, sparse augmented, and explicit ADMM output for a purely
quadratic model. If no defensible singular-value lower bound or exact-zero proof
is available, the result is the existing structured ``numerical_failure``.

Condition estimates guide routing, refinement, and diagnostics. They do not
independently reject an exact-zero candidate or a candidate whose forward
objective-gap upper certificate passes. Bounded exact helpers may improve or
rescue a candidate, but their private size limits may affect only runtime and
recovery rate, never the meaning of ``optimal``. In particular, the absence of a
helper above a threshold must not weaken the universal certificate.

An exact helper may compute the continuous optimum and an exact rational gap
for a proposed binary64 vector. That exact gap is itself a stronger forward-gap
calculation, so a separate condition-number cutoff cannot veto it. Conversely,
the exact optimum's coordinatewise rounding does not establish discrete
binary64 optimality for a coupled quadratic Hessian and cannot replace the
continuous-objective certificate.

The implementation records the lower bound explicitly. Positive L2 contributes
the scaled \(\sqrt{\lambda}I\) block, which directly bounds every singular
direction. Without L2, a maximum-bottleneck spanning tree of the scaled
incidence rows is rooted at the gauge anchor. The path structure of its inverse
reduced incidence gives
\(\sigma_{\min}\geq 1/\lVert T^{-1}\rVert_F\); that Frobenius bound is
accumulated conservatively after normalization by the bottleneck coefficient.
Guarded normal factors, dense and sparse augmented factors, and final quadratic
ADMM iterates all use this same identity/tree bound; dense SVD information
remains a rank and condition diagnostic rather than being promoted to an
unproved lower bound. The source-gradient norm uses a conservative binary64
evaluation-error allowance, a tighter per-row forward-error bound when the
platform provides genuinely wider NumPy arithmetic, and a bounded exact-dyadic
accumulator as a final certification aid. It then gives
\[
f(w)-f(w^\star)\leq
\frac{\lVert\nabla f(w)\rVert_2^2}
     {2\sigma_{\min}^2}.
\]
If a positive lower bound cannot be represented or justified, certification
fails.

This completion amendment supersedes the earlier path-specific acceptance
wording in this ADR wherever that wording gave guarded normal candidates a
stronger forward certificate than augmented or sparse candidates, treated
numerical zero as a complete proof, or allowed private helper thresholds to
change success semantics. Solver and linear-backend selection are recorded
separately in ADR 0008.

### Compatibility

No legacy objective-scaling mode, compatibility flag, deprecated formula, or
alternate convention is provided. These corrections precede v0.8 release.
Ordinary direct dense and direct sparse squared-loss/L2 fitted weights remain
unchanged because their normal system already used
\(L_{\mathrm{obs}}+\lambda I\).
Reported mismatch and L2 values adopt the coherent half-factor convention.
ADMM solutions may change where the earlier relative scaling was inconsistent.

## Consequences

- Direct evaluation, result breakdowns, reports, scalar proximal derivatives,
  and all solver backends describe the same objective.
- Large-delta Huber and squared loss agree in their shared quadratic region.
- Reciprocal penalties are finite and convex below `epsilon`, with honest
  branch derivatives.
- Zero-strength penalties are safe no-ops across evaluation, graph structure,
  operator availability, and backend selection.
- Hard-bound status is scale-aware, and reports expose the tolerance used.
- Hard feasibility and final hard classification use one measurement-space
  accepted set rather than tolerances in incompatible units.
- Successful solver metadata implies a finite reported soft objective.

## Alternatives considered

### Preserve the old squared-loss and L2 report scaling

Rejected because it would keep Huber and squared loss relatively inconsistent
or require a second convention in ADMM and downstream reports.

### Add a legacy scaling switch

Rejected because v0.8 has not been released, the previous behavior was
internally contradictory, and a compatibility mode would make every future
objective consumer carry both formulas.

### Keep reciprocal clipping and set its derivatives to zero

Rejected because a flat plateau removes boundary repulsion exactly where it is
most needed. The tangent continuation is finite, convex, and preserves the
value and slope at `epsilon`.

### Expose a public hard-tolerance option

Rejected for this correction. One shared roundoff classification policy is
needed; configurable solver convergence tolerances already serve a different
purpose.
