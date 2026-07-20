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
\qquad R=\operatorname{diag}(\rho_r).
\]

After converting observations to implied differences, the same problem is

\[
\min_w \frac12\left\|R^{1/2}
(B^\mathsf{T}w-z^{\mathrm{obs}})\right\|_2^2.
\]

Its normal equations are

\[
Lw=BRz^{\mathrm{obs}},
\qquad L=BRB^\mathsf{T}.
\]

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
b=BRz^{\mathrm{obs}}+\lambda w^{\mathrm{ref}}.
\]

Writing both objective terms without one-half factors gives the same system,
because the common gradient factor cancels. Positive L2 regularization removes
the component-constant null directions from \(A\), but it does not connect the
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

A single fixed normal system represents squared measurement mismatch plus
optional L2 regularization. Hard intervals and equalities restrict its feasible
set but do not change the quadratic objective matrix; their constrained optimum
need not satisfy the unconstrained normal equation. Huber mismatch and
additional scalar penalties are not, in general, represented by the same fixed
Laplacian system.

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
