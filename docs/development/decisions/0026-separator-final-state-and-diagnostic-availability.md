# 0026 — Separator final-state certification and diagnostic availability

- **Status:** Accepted
- **Date:** 2026-10-06
- **Related issue:** [#116 — Checkpoint-C separator remediation](https://github.com/DeloneCommons/pyvoro2/issues/116)
- **Related plan:** [v0.9 Phase C](../plans/v0.9.md#integration-checkpoint-c-inverse-candidate-api)
- **Amends:** [ADR 0007](0007-separator-objective-contract.md),
  [ADR 0015](0015-atomic-separator-active-state.md), and
  [ADR 0019](0019-separator-measurement-spaces-and-supported-realization.md)

## Context

The Checkpoint-C review found that a final zero-L2 reference shift could destroy
hard feasibility after ADMM convergence, that active residuals lost complete
affine cancellation, and that singleton references depended on dispatch.
Correct source-space diagnostics could also exceed binary64 even when the
model, final weights, objective and geometry remained available. Strict JSON
then rejected an otherwise coherent result.

Issue #116 selects bounded private final-state consolidation (R1), existing
scientific owners and schema v3. This decision records that selected contract;
implementation and independent remediation review are a prerequisite to
Checkpoint C, not acceptance of that checkpoint.

## Decision

### Certify the exact final representative

Every native `optimal` or converged result requires a finite soft objective and
every applicable authoritative hard row to satisfy the existing per-row
predicate on the exact returned weights. Global maxima of violation and
tolerance are descriptive and cannot replace that predicate. Representative
selection and any existing recovery precede final certification; the successful
quadratic vector is unchanged after its certificate.

A failed final success claim becomes `numerical_failure`, with no weights,
radii, shift, predictions, residuals, summaries or objective breakdown. Actual
solver/backend and completed iterations remain; source-only edge diagnostics,
policy, provenance and connectivity remain inspectable. Precheck
`hard_feasible` keeps its existing meaning; no conflict or infeasibility is
invented. Native nonfinite soft objectives are refused even for nonconverged
candidates. The public builder rejects false success with `ValueError`, also
when `canonicalize_gauge=False`; an explicitly nonconverged external candidate
and a genuine finite `max_iter` candidate remain inspectable.
Bound export and active reconstruction repeat the same guard from exact final
weights and selected policy, so relabeling an unsuccessful candidate or changing
its objective metadata cannot license false success.

One private standalone representative policy covers direct, ADMM and no-work
paths. Positive L2 fixes its objective reference without gauge shifts. At zero
L2, multiple model-coupling components use supplied reference means or implicit
zero means; singleton values equal their reference entries exactly. One
connected component with more than one site retains its solver anchor. Empty
native fits use the supplied reference or zeros, including one site; public
one-site canonicalization remains a no-op. Coupling is established by positive
confidence, applicable hard rows or positive penalty rows, not solely the
informative data graph.

Optional active alignment uses the preceding `weights_eval`, after relaxation,
only when it preserves contrasts exactly; otherwise it retains the certified
vector. It is disabled with positive L2. Certification covers the selected
projected policy; excluded candidate hard rows are not enforced. Outer
termination and final inner availability keep ADR 0015's separate meanings.

### Keep source and model mathematics separate

The accepted binary64 row operands define
`y = beta + alpha*w_i - alpha*w_j`. Source and mismatch operands remain distinct.
Problem/graph coefficients and quadratic operators belong to mismatch space;
normal RHS is the complete/scaled `c*alpha*(target-beta)`, never `rho*z_obs`.
The squared-plus-L2 operator boundary and existing empty-row exceptions remain.

`AlgebraicEdgeDiagnostics` retains source coefficients and descriptive
`c*alpha_source**2`, not mixed-model curvature or converted likelihood weights.
Its residual is `z_obs_source-z_fit`. Source weighted norms describe
`sqrt(sum(c*source_residual**2))`; weighted RMSE divides by the number of rows,
not `sum(c)`. Algebraic RMSE/MAE retain their difference-space residual meaning.
Physical equivalence between source encodings need not imply equal row IDs.

One private owner evaluates complete source affine residuals for fixed results,
full active candidates, final records and each iteration's own relaxed
`weights_eval`. A small residual may coexist with `predicted == target`.
Final validation independently evaluates the outer and nested residuals from
final weights; equality between two copied wrong arrays is insufficient.
Full source statistics retain inactive and confidence-zero rows. Full-candidate
evaluation may disable hard compilation without changing selected-fit policy.

### Bounded numerical availability

Finite diagnostic values retain binary64 rounding and underflow semantics.
Only a typed producer can establish either reason:

- `out_of_binary64_range`: the final mathematical diagnostic is finite but
  outside finite binary64 magnitude; raw numerical data may retain signed
  infinity;
- `unavailable_dependency`: a required accepted operand is unavailable, so the
  diagnostic cannot be evaluated validly; raw NaN is permitted only with this
  producer-established dependency.

Reasons are never inferred merely from a caller's NaN, infinity or metadata.
Final exports recompute from bound observations/policy and exact final weights
and compare supplied values. Unavailable alpha cannot create a false
`z_obs=0` by division by infinity. Independent values remain available, and
confidence-zero weighted work/rho/RHS are exactly zero before dangerous work.
Solver admission for nonrepresentable model operands is not extended.

Weighted complete affine evaluation applies `sqrt(c)` before materializing a
possibly overflowing residual. RMS and other reductions establish availability
at aggregate level; a row outside range does not prove its RMS or weighted norm
is outside range. Supported scale-safe evaluations recover finite aggregates;
unresolved dependencies propagate without dropping rows.

Active final layers remain atomic with respect to finite weights/radii, selected
fit, genuine realization and source association. Proven unavailable diagnostic
cells do not remove their container. Whole-container `None` still denotes an
absent final state, and no geometry or NaN placeholder row is fabricated.

Standalone candidate diagnostics retain a private owned row/value-bound
evaluation snapshot. A complete active reconstruction also checks that snapshot
against actual final weights. History retains immutable iteration/source,
summary-value and reason records, O(history length), without weight snapshots.
It cannot independently recertify discarded historical weights. Supported copy,
replacement and pickle preserve these private bindings; ordinary finite manual
reconstruction remains supported. Nonfinite manual values lack authority.
There are no new public dataclass fields or mandatory constructor arguments.

### Exact schema v3 grammar

All fixed, realized and active reports, including nested reports, use
`schema={"name":"pyvoro2.inverse.separator.report","version":3}`.
Every report root and every fixed/candidate/enriched active record has required
`unavailable_diagnostics`, empty `{}` in ordinary output. Observation-only and
realized-only rows do not gain it; history rows have no separate map.

The map contains local JSON Pointers to eligible numeric leaves converted to
null, with exactly the two reason strings above. Pointers start with `/`, use
ordinary `~0`/`~1` escaping and zero-based decimal array positions without
leading zeros. Report maps exhaust their whole subtree, including nested
reports and marginal-record copies; nested reports and records retain their
local maps. Rebasing copies maps and enrichment merges source/mismatch maps.
Structural nulls have no entry; finite values have no entry.

| Container | Closed nullable diagnostic whitelist |
|---|---|
| Fixed record; report `/fit_records/*` | `predicted`, `predicted_fraction`, `predicted_position`, `residual`, `mismatch_predicted`, `mismatch_residual`, `alpha`, `beta`, `z_obs`, `z_fit`, `algebraic_residual`, `edge_weight` |
| Fixed `/edge_diagnostics` | Elements of `alpha`, `beta`, `z_obs`, `z_fit`, `residual`, `edge_weight`; scalars `weighted_l2`, `weighted_rmse`, `rmse`, `mae` |
| Fixed `/summary` | `rms_residual`, `max_residual` |
| Standalone candidate record | `predicted`, `predicted_fraction`, `predicted_position`, `residual` |
| Enriched active record; `/diagnostics/*`, `/marginal_records/*` | Candidate fields plus `mismatch_predicted`, `mismatch_residual` |
| Active `/summary`, `/history/*` | `rms_residual_all`, `max_residual_all` |
| Active `/fit` | Complete nested fixed-report whitelist, rebased with `/fit` |
| Realized report | No newly nullable scalar; required empty root map |

Targets and their representations, confidence, model policy, inputs, geometry,
IDs/provenance, successful weights/radii/shift, objective components and hard
status/violation/tolerance remain strict. History `weight_step_norm` participates
in stopping and is outside this policy. All other existing v2 key/type/absence
rules remain. Source/row identity versions and report kind names do not change.

Typed leaf conversion occurs before the general JSON normalizer. That helper
still rejects arbitrary raw NaN/infinity even when supplied with a reason map;
writers retain `allow_nan=False`. No v2 compatibility writer is introduced.

## Consequences and alternatives

Consumers must explicitly accept schema v3 and distinguish structural absence
from present numerical unavailability through the local/exhaustive maps.
Provisional/Experimental classifications and the completed C-parameter gate
remain unchanged. This boundary adds no solver family, dependency, public
AcceptedWeightState, universal nonquadratic certificate or release qualification.

Source-to-model substitution would change units and scientific meaning.
Blanket nonfinite JSON relaxation would license invalid inputs/objectives.
Blind reduction of stored infinities would lose representable aggregates.
Retaining earlier geometry or historical weight vectors would mix final states
or exceed the bounded provenance requirement. These alternatives are rejected.
