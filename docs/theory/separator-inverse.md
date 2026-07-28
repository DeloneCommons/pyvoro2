# Inverse fitting from separator observations

Consider a power diagram with fixed sites and unknown weights. Suppose that the
available data do not describe complete cells. Instead, they specify where the
separator should lie for selected pairs of sites. The inverse problem is to find
one set of site weights that reconciles these local observations.

The central reduction is simple: the position of one pairwise separator
determines one difference of weights. A collection of observations therefore
becomes a graph of desired weight differences. Exact compatibility is a cycle
condition on that graph; noisy fitting leads to a gauge-aware graph-Laplacian
problem. Geometric realization remains a separate question because a third site
may block a pair even when its requested separator equation is fitted exactly.

The current Python API stores these data in `SeparatorObservations`. This page
uses the same term **separator observations** while stating the main
mathematical ideas independently of API details.

## Observation model for one pair

Fix site \(i\) and one selected image \(q_j\) of site \(j\). In a nonperiodic
domain, \(q_j=p_j\). Define

\[
\Delta_{ij}=q_j-p_i, \qquad d_{ij}=\lVert\Delta_{ij}\rVert>0,
\]

and parameterize the connector line by

\[
x(t)=p_i+t\Delta_{ij}.
\]

The pairwise separator meets this line where the two power distances agree.
Solving the resulting scalar equation gives the normalized connector coordinate

\[
t_{ij}=\frac12+\frac{w_i-w_j}{2d_{ij}^2}.
\]

The intersection always exists on the connector line, although it need not lie
between the two sites. If the observation is the signed distance from site
\(i\) along that line, write

\[
s_{ij}=d_{ij}t_{ij}
=\frac{d_{ij}}2+\frac{w_i-w_j}{2d_{ij}}.
\]

Both conventions, and any affine reparameterization of them, can be written as

\[
y_r=\beta_r+\alpha_r(w_{i_r}-w_{j_r}),
\qquad \alpha_r\ne 0.
\]

Thus observation \(r\) implies the desired weight difference

\[
z_r^{\mathrm{obs}}
=\frac{y_r^{\mathrm{obs}}-\beta_r}{\alpha_r}.
\]

The data are local, but the same site weight appears in every observation that
touches that site. Global consistency is therefore a graph problem.

## Compatibility on an observation graph

The simplest nontrivial example has three sites. If observations prescribe the
differences \(z_{12}\), \(z_{23}\), and \(z_{31}\), then any global weight vector
must satisfy

\[
z_{12}+z_{23}+z_{31}=0.
\]

A nonzero sum means that the three local requirements cannot all come from one
set of weights. The same principle holds on every cycle of a larger observation
graph.

Formally, create one graph vertex per site and one oriented edge per observation.
Parallel edges are retained, so repeated measurements and different periodic
images of the same site pair remain distinguishable. Let \(B\) be the
vertex-edge incidence matrix, with the column for observation \(r\) equal to
\(e_{i_r}-e_{j_r}\). Exact compatibility is the linear system

\[
z^{\mathrm{obs}}=B^\mathsf{T}w.
\]

The observations are compatible if and only if their signed sum vanishes around
every cycle of the observation multigraph. Equivalently,
\(z^{\mathrm{obs}}\) lies in the image of \(B^\mathsf{T}\), or is orthogonal to
the graph cycle space.

Keeping parallel edges is mathematically important. In a periodic problem, two
observations may refer to different images of the same site pair. They still
constrain the same weight difference, and together they form a two-edge cycle
whose consistency would be hidden by a simple unlabelled pair set.

## Identifiability, global gauge, and component offsets

Only observations with positive confidence contribute to the objective and
identify differences. These observations define the **effective observation
graph**.

A zero-confidence observation is still a distinct row of the resolved data and
a distinct column of the full incidence matrix. Its effective weight is zero,
however, so it contributes neither to the Laplacian nor to its right-hand side
and cannot connect effective components.

Within one connected effective component, all relative weights are determined
by exact compatible data. One additive constant per component remains
unobserved. When the effective graph is connected, that constant is exactly the
global additive gauge of the complete power diagram.

When the effective graph has several components, the interpretation is more
subtle. One common shift of all sites is still harmless geometric gauge, but the
additional relative offsets between components are not determined by the
separator data and may change the full tessellation. A solver may choose a
canonical centering or align components to a reference, but that choice is a
policy or prior, not information recovered from the disconnected observations.

## Noisy observations and the Laplacian estimator

Let \(\lambda_r\ge 0\) be the confidence assigned to observation \(r\). The
quadratic estimator minimizes measurement-space mismatch:

\[
\min_w \frac12\sum_r
\lambda_r\left(y_r^{\mathrm{obs}}-
\beta_r-\alpha_r(w_{i_r}-w_{j_r})\right)^2.
\]

Define

\[
\rho_r=\lambda_r\alpha_r^2,
\qquad
q_r=\lambda_r\alpha_r(y_r^{\mathrm{obs}}-\beta_r),
\qquad R=\operatorname{diag}(\rho_r).
\]

After converting observations to implied differences, the same problem is

\[
\min_w \frac12\left\|R^{1/2}
(B^\mathsf{T}w-z^{\mathrm{obs}})\right\|_2^2.
\]

Its normal equations are

\[
Lw=Bq,
\qquad L=BRB^\mathsf{T}.
\]

The equality \(q=Rz^{\mathrm{obs}}\) is algebraically valid, but numerical
implementations construct \(q_r\) directly. An implied difference may overflow
even when \(\lambda_r\alpha_r(y_r^{\mathrm{obs}}-\beta_r)\) is finite, so
`z_obs` is diagnostic data rather than a required normal-equation
intermediate.

The matrix \(L\) is the weighted graph Laplacian of the effective observation
graph. Its nullspace consists of vectors that are constant on each connected
component, so one explicit alignment condition is required per component.
After those conditions are chosen, the quadratic fit is unique.

The fitted edge differences are the weighted projection of the observed edge
data onto the space of compatible differences. This interpretation separates
two useful diagnostics:

- **measurement-space residuals**, expressed in the units of the observed
  separator positions;
- **difference-space residuals**, expressed as errors in the implied values of
  \(w_i-w_j\).

They are related through \(\alpha_r\), but they are not numerically
interchangeable and should not share an ambiguous summary name.

With L2 strength \(\lambda\ge 0\) and reference
\(w^{\mathrm{ref}}\), add

\[
\frac{\lambda}{2}\lVert w-w^{\mathrm{ref}}\rVert_2^2
\]

under the one-half objective convention used above. The regularized normal
system is

\[
Aw=b,
\qquad
A=L+\lambda I,
\qquad
b=Bq+\lambda w^{\mathrm{ref}}.
\]

The one-half factors are part of the reported objective contract; they are not
only a normal-equation convenience. Positive L2 regularization removes the
component-constant null directions from \(A\), but it does not connect the
observation graph or turn prior-selected component offsets into information
identified by the separator data.

## Robust losses and regularization

The graph structure does not depend on squared loss. Convex edge-separable
losses, such as the Huber loss, can reduce the influence of outlying separator
observations while retaining the same site-difference model.

Regularization may stabilize weakly informed problems or choose among
unidentified component alignments. It also adds information to the estimation
problem. A solution selected by a reference or penalty should therefore be
reported as such, rather than described as uniquely identified by the separator
observations alone.

For measurement residual

\[
e_r(w)=\beta_r+\alpha_r(w_{i_r}-w_{j_r})-y_r^{\mathrm{obs}},
\]

and confidence \(c_r\), the implemented mismatch contribution is
\(\sum_r c_r\ell(e_r)\). Confidence multiplies only this mismatch term. It
does not multiply scalar penalties or hard restrictions. The two mismatch
losses are

\[
\ell_{\mathrm{sq}}(e)=\frac12e^2
\]

and

\[
\ell_\delta(e)=
\begin{cases}
\frac12e^2, & |e|\le\delta,\\
\delta\left(|e|-\frac12\delta\right), & |e|>\delta.
\end{cases}
\]

Thus a Huber model with every residual in the quadratic branch is exactly the
squared-loss model. L2 regularization is
\(\frac{\lambda}{2}\lVert w-w^{\mathrm{ref}}\rVert_2^2\), with gradient
\(\lambda(w-w^{\mathrm{ref}})\) and Hessian contribution \(\lambda I\).
A reference with \(\lambda=0\) contributes nothing to the objective, although
the documented output alignment policy may use it to choose otherwise
unidentified component offsets.

Evaluation preserves the same formulas at extreme binary64 scales. A
zero-confidence row returns exact zeros before its residual is formed.
Positive confidence is applied in a scale-safe order before squaring. In
particular, the quadratic scale uses
\(\sqrt{c_r}\sqrt{1/2}\), rather than first halving \(c_r\), so the minimum
positive binary64 confidence remains effective. Huber evaluates only the
active branch. Zero L2 strength likewise returns zero before subtracting the
reference; positive L2 scales differences before a stable sum-of-squares
reduction. These rules preserve finite weighted values without clipping
genuinely non-representable results.

The residual is evaluated directly as the complete affine combination
\(\beta+\alpha w_i-\alpha w_j-y^{\mathrm{obs}}\). It is not reconstructed by
subtracting the target from a stored prediction. Consequently, a prediction
that lies outside binary64 range may be non-finite while a cancellation with
the target leaves a finite, reportable residual and mismatch value. Mismatch
evaluation also applies confidence and the loss scale directly to the complete
affine combination, so a finite weighted objective does not require the
unscaled residual itself to fit binary64. The Huber linear expression is
evaluated as one scaled combination rather than as the difference of two
potentially overflowing terms. Ordinary rows use vectorized arithmetic; rows
whose forward-error bound exceeds
\(64\epsilon_{64}\) times their result magnitude use vectorized error-free
product/sum compensation. Every raw compensated product must have normal-range
exponent margin; otherwise the row takes the exact binary64-input fallback
before that product is executed. The ordinary pass reuses its affine products
and one forward-error calculation. A representable compensated residual and
its squared mismatch share the same rounded residual, while exact exceptional
rows and non-representable residuals form the objective from the exact affine
operands. This conditioning criterion resolves the same relative scale as the
hard-row policy and preserves finite destructive cancellation before errors
become material to that policy.

The existing soft-interval strength \(s\) means

\[
s\left(\max(a-y,0)^2+\max(y-b,0)^2\right),
\]

and the existing exponential-boundary strength means

\[
s\left[
\exp\left(\frac{a+m-y}{\tau}\right)
+\exp\left(\frac{y-(b-m)}{\tau}\right)
\right].
\]

These terms do not acquire a one-half factor. For reciprocal repulsion, define
the contribution from one inward boundary distance \(d\) as

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

The full interval contribution is \(q(y-a)+q(b-y)\). The tangent continuation
is finite and convex below \(\epsilon\). Value and first derivative are
continuous at \(\epsilon\), where the linear branch applies. At \(m\), the
inactive branch applies: value is continuous, but the derivative jumps, so the
activation point has no unique derivative.

Value, first derivative, and second derivative share the same branch
classification but are evaluated separately. Scale-safe reciprocal products
avoid first forming \(d^2\), \(d^3\), or \(\epsilon^2\), including for valid
very small parameters.

For a positive soft-interval strength, the active-boundary displacement is
scaled before its square is formed, and its first derivative is evaluated as a
scaled difference. Reciprocal quotient differences and continuation terms are
likewise combined before binary64 range conversion. These evaluation choices
preserve the formulas above; they do not clip true extended-real results.
Objective evaluation derives soft and reciprocal boundary distances directly
from the complete affine prediction formula and the boundary. It therefore
does not require the separately stored public prediction to be representable.
When a rounded inward distance is not safely separated from \(\epsilon\) or
\(m\), reciprocal branch selection uses the exact binary64 operands for scalar
and complete-affine expressions. Thus rounding a distance onto a branch
boundary cannot silently deactivate a value or its derivatives.
The exact exceptional conversion for an objective returns infinity on genuine
range overflow. It is intentionally distinct from the finite saturation used
only to approximate tolerance-expanded hard-bound endpoints.

A zero-strength scalar penalty is absent. Its value and derivatives are
exactly zero without evaluating its exponential, reciprocal, or other branch
expressions; it does not affect graph coupling, backend selection, or the
quadratic operator.

A single fixed normal system represents squared measurement mismatch plus
optional L2 regularization. Hard intervals and equalities restrict its feasible
set but do not change the quadratic objective matrix; their constrained optimum
need not satisfy the unconstrained normal equation. Huber mismatch and
additional scalar penalties are not, in general, represented by the same fixed
Laplacian system.

Numerically, the scaled weighted least-squares form is the canonical solver
model. For one component, its observation rows are

\[
\sqrt{c_r}\,\alpha_r(e_{i_r}-e_{j_r})^\mathsf{T}w
=\sqrt{c_r}(y_r^{\mathrm{obs}}-\beta_r),
\]

and positive L2 adds the rows
\(\sqrt{\lambda}I w=\sqrt{\lambda}w^{\mathrm{ref}}\). The direct
augmented solve avoids explicitly squaring its condition number.

The implementation applies exact power-of-two scales to the component variables
and objective rows before assembling the design. It also keeps the high and low
parts of each target offset \(y_r^{\mathrm{obs}}-\beta_r\) as separate
right-hand-side contributions. Consequently, a minimum-positive confidence or
L2 strength is not erased merely because another row is much larger. A
mathematically nonzero term that cannot be represented in the scaled binary64
system produces a structured numerical failure rather than a modified
objective.

For an unregularized component, the internal gauge fixes the site with the
largest scaled weighted degree and solves the remaining contrasts. The result is
then shifted uniformly to the documented public gauge, so this conditioning
choice does not change any predicted separator. Positive L2 removes the gauge
mathematically; the implementation centers variables around the reference mean
while retaining
all site coordinates and the full \(\sqrt{\lambda}I\) block. It does not
change one or two selected coordinates after solving in order to force an exact
floating-point component sum.

The augmented design remains the authoritative representation. For ordinary
components with a conservative design-condition bound comfortably below the
binary64 normal-equation limit, the direct solver may try a guarded dense or
sparse normal-system candidate for performance. It uses the same component
scales and split right-hand side, and it is accepted only after the same
source-objective and stationarity checks. Assembly, solve, or certification
failure falls back to the unsquared augmented system.

Outside that guarded path, dense components use a QR solve with an SVD rank and
condition estimate. High-condition designs also compare iterative refinement
with a rank-revealing truncated candidate and retain truncation only when an
accurately evaluated augmented residual is smaller. Sparse components solve the
unsquared augmented problem through a sparse KKT factorization and retain the
incidence structure without a dense site matrix. Neither candidate path changes
linear backend at a site-count threshold. ADMM uses the selected dense or
sparse prepared weight system for every iteration. Explicit ADMM on squared
mismatch plus optional L2 still performs ADMM, and its final iterate must pass
the common quadratic certificate.

A quadratic result is reported as successful only after certification. The
final binary64 weights are checked after scale reversal and public gauge
canonicalization. A floating objective equal to zero is insufficient:
exact-zero proof requires every positive-confidence affine residual to vanish
exactly and, under positive L2, every coordinate to equal its reference.
Otherwise every guarded normal, dense augmented, sparse augmented, and final
quadratic ADMM candidate needs a conservative forward objective-gap bound.

Positive L2 supplies the singular-value lower bound from its scaled
\(\sqrt{\lambda}I\) block. Without L2, root a maximum-bottleneck spanning tree
at the gauge anchor. The path structure of its inverse reduced incidence gives
the conservative bound
\(\sigma_{\min}\geq 1/\lVert T^{-1}\rVert_F\). The same proved identity/tree
lower bound is used for normal, dense augmented, sparse augmented, and final
quadratic ADMM candidates; dense SVD information remains a rank and condition
diagnostic. Combining this lower bound with an upper bound on the source
gradient norm gives
\[
f(w)-f(w^\star)\leq
\frac{\lVert\nabla f(w)\rVert_2^2}
     {2\sigma_{\min}^2}.
\]
The gradient upper bound starts with ordinary binary64 forward-error
accounting, tightens it with per-row extended-precision accounting when the
platform provides genuinely wider arithmetic, and may finally use a bounded
exact-dyadic accumulation. None of these paths changes the source objective.
No factor path can report success without such a bound or an exact-zero proof.
Bounded exact binary64-input helpers may improve a candidate or tighten its
gradient bound, but are not an unbounded production matrix solver and do not
change the meaning of success above a helper threshold. If a material weak term
or the continuous optimum cannot be represented accurately by the public
binary64 weight vector, the solver returns ``numerical_failure`` instead of a
false ``optimal`` result.

For a bounded small component, exact rational arithmetic may compute both the
continuous optimum and the exact source-objective gap of a binary64 candidate.
This is a tighter implementation of the same continuous forward-gap
certificate, not a change of optimization domain. Rounding each coordinate of
the exact optimum independently does not in general minimize a coupled
quadratic over the binary64 lattice. Accordingly, an exactly zero continuous
optimum is successful only when the returned binary64 weights make every
source-objective term exactly zero; otherwise the representability limitation
is reported as ``numerical_failure``.

The optional direct warm start for a nonquadratic ADMM problem is only an
acceleration. Supported numerical failure falls back to the reference or zero
seed. For Huber mismatch without scalar penalties or hard bounds, the solver
also retains the best candidate according to the authoritative Huber-plus-L2
objective and never discards an exactly zero-objective candidate.

Well-conditioned normal right-hand sides, operator products, and
mismatch-only weighted averages use vectorized arithmetic under the same
\(64\epsilon_{64}\) relative forward-error target as complete affine
evaluation. Nearly balanced
opposite-sign averages, same-sign barycentric forms that lose a small weighted
contribution, sufficiently imbalanced positive weights, normalized weights that
would underflow, and mixed-sign incidence sums that fail the relative
forward-error criterion use exact finite-input exceptional fallbacks. Product
range checks use operand exponents to route potentially subnormal
multiplication through exponent-scaled arithmetic without globally suppressing
underflow. Complete sum-of-products evaluation also prevents a finite nonzero
term that rounds to zero in isolation from disappearing before aggregation.
The Huber linear proximal displacement similarly evaluates its confidence,
threshold, and ADMM denominator as one ratio-product rather than first forming
the reciprocal denominator. Huber proximal branch selection evaluates the
unnormalized lower and upper optimality tests as complete sum-of-products
expressions, avoiding comparisons with rounded
target-plus-or-minus-threshold values. Exceptional tests retain their exact
sign even when their nonzero magnitude lies below binary64 range. Direct Huber
values and derivatives classify the complete residual against both threshold
boundaries by the same exact-sign rule. Mean-absolute diagnostics use
exponent-aware normalization and omit only ratios too small to affect the
rounded result, without executing an underflowing division.

## Hard interval and equality restrictions

An admissible interval in measurement space,

\[
a_r\le y_r\le b_r,
\]

can be converted, with the orientation of \(\alpha_r\) taken into account, into
a bound on one weight difference:

\[
\ell_r\le w_{i_r}-w_{j_r}\le u_r.
\]

Each upper or lower bound is a directed difference inequality. The full system
is feasible if and only if its directed constraint graph contains no negative
cycle. When a negative cycle exists, it provides a compact contradiction
witness that identifies a mutually incompatible subset of restrictions.

This graph precheck distinguishes genuine infeasibility from ordinary numerical
non-convergence.

Floating-point hard-bound classification uses a scale-aware roundoff policy,
separate from solver convergence tolerance. For lower bound \(a_r\), prediction
\(y_r\), upper bound \(b_r\), and binary64 machine epsilon
\(\epsilon_{64}\), define

\[
v_r=\max(a_r-y_r,\ y_r-b_r,\ 0),
\]

\[
t_r=10^{-12}
+64\epsilon_{64}\max(|a_r|,|y_r|,|b_r|).
\]

The row is satisfied exactly when \(v_r\le t_r\). A row containing a
non-finite bound or prediction is never satisfied.

For finite bounds, this predicate defines one tolerance-expanded closed
interval in measurement space. The feasibility precheck derives that accepted
measurement interval first and then maps its endpoints through
\(y_r=\beta_r+\alpha_r(w_{i_r}-w_{j_r})\) into difference bounds.
Bellman–Ford uses those bounds directly; it does not apply the
measurement-space \(10^{-12}\) tolerance again to path distances measured in
weight-difference units. Precheck feasibility and final classification
therefore refer to the same accepted set. A constrained ADMM solve projects
onto that same tolerance-expanded measurement interval, not the narrower
unexpanded user interval. The original bounds still define raw violation and
tolerance reporting.
The relative tolerance product is evaluated scale-safely, so subnormal finite
bounds do not raise merely because their relative contribution rounds away
next to the absolute term.

Reports retain the maximum raw violation and separately expose the maximum
tolerance actually used. A hard-constrained solver reports success only after
both its primal/dual convergence tests and this row predicate pass.

## Algebraic fitting and geometric realization

The affine observation law concerns the separator of two sites considered as a
pair. It does not guarantee that the same pair is a boundary in the complete
power diagram. A competitor can block the pair, one endpoint cell can be empty,
or a different periodic image can be realized.

The computational workflow therefore has four distinct layers:

1. **fixed-observation fit:** reconcile the requested weight differences;
2. **forward realization:** compute the power diagram from the fitted weights;
3. **boundary matching:** compare requested pairs and image shifts with the
   realized boundaries;
4. **optional outer refinement:** revise the active observation set and refit.

A fit may be algebraically excellent and geometrically unsupported. Conversely,
a realized pair may be absent from the observation set. Both outcomes are
scientifically meaningful and should remain visible in the result.

## Periodic image bookkeeping

A periodic observation refers to one selected image of the second site. The
realization layer should therefore distinguish four outcomes:

- the requested image is realized;
- the same site pair is realized only through another image;
- the pair is not realized;
- one or both endpoint cells are empty.

Silently replacing the requested image by the nearest or currently realized
image changes the observation. Such a change may be offered as an explicit
repair strategy, but not as a hidden fallback.

## Connector visibility and clearance

Full boundary realization is a global tessellation question. The manuscript
also studies a narrower local diagnostic: whether the connector line passes
through the relative interior of the pair face.

Let \(x_{ij}(w)\) be the connector-separator point. For each competitor
\(k\ne i,j\), define its slack at that point by

\[
g_{ijk}(w)=\pi_k\bigl(x_{ij}(w)\bigr)
            -\pi_i\bigl(x_{ij}(w)\bigr),
\]

and define the visibility score

\[
\sigma_{ij}(w)=\min_{k\ne i,j} g_{ijk}(w),
\]

with the value \(+\infty\) when no competitor exists. Each competitor slack is
affine in the fitted weights. Under the nondegeneracy
assumption that no competitor bisector coincides with the pair bisector,
\(\sigma_{ij}(w)>0\) exactly when the connector point lies in the relative
interior of the unblocked pair face. A zero score marks a higher-order tie, and
a negative score means that at least one competitor blocks the connector point.

For a connector-visible pair, **clearance** is the in-face distance from the
connector point to the nearest blocking competitor hyperplane. Domain clipping
can impose an additional limiting margin. Visibility is weaker than full face
realization, and clearance is a geometric ranking or stability diagnostic—not a
universal application-specific decision rule.

## Realization-aware active-set refinement

The implemented outer algorithm alternates between fitting and realization:

1. fit the currently active observations;
2. compute the resulting power diagram;
3. match requested boundaries and periodic images;
4. update the active set using hysteresis;
5. stop on self-consistency, a detected cycle, numerical failure, or an
   iteration limit.

The graph compatibility results and the convex fixed-observation estimator form
the exact inner theory. The hysteretic outer loop is a practical,
realization-aware algorithm with inspectable path diagnostics; it is not claimed
to be globally convergent for every candidate set.

After the final active set is refitted, component-offset alignment is applied
only to zero-L2 gauge components and only when exact binary64-input checks show
that all component weight differences are unchanged. Positive L2 fixes the
component means as part of the objective, so aligning such a fit to a previous
iterate would change the optimization problem and is not performed.

## Interpreting a result

A scientifically useful report should answer separate questions:

- Are the observations exactly or approximately compatible?
- Which relative weights are identified by the effective graph?
- Which component offsets were chosen by policy or prior information?
- Are hard restrictions feasible, and is there a contradiction witness if not?
- Which requested pairs and periodic images are realized?
- Are any endpoint cells empty?
- Did outer refinement converge, cycle, fail numerically, or reach a limit?

These questions should not be collapsed into one generic “fit succeeded” flag.
In particular, a solver cannot report an optimal or converged result when any
reported soft-objective component or total is non-finite. A failed optional
direct ADMM warm start falls back to the reference or zero initialization and
does not by itself constitute solver failure.

## Scope and further reading

The theory above assumes fixed sites and scalar power weights. It does not infer
site positions, guarantee global convergence of the active-set loop, or claim
that pairwise separator data reconstruct a complete application-specific
partition.

The current calls and result objects are documented in the
[separator-fitting guide](../guide/powerfit.md) and API reference. Full proofs,
perturbation results, synthetic experiments, periodic-image tests, and the
molecular benchmark are given in the manuscript *Fixed-site inverse fitting of
power-diagram weights from pairwise separator data* and its archived
reproducibility materials.
