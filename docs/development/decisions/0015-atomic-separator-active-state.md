# 0015 — Atomic separator active-set final state

- **Status:** Accepted
- **Date:** 2026-08-15
- **Related issue:** [#43 — Make the separator active-set final state atomic and self-consistent](https://github.com/DeloneCommons/pyvoro2/issues/43)
- **Related plan:** [v0.8 remediation execution plan](../plans/archive/v0.8-remediation.md)
- **Depends on:** [ADR 0014](0014-separator-observation-and-source-identity.md)

## Context

The experimental separator active-set solver evaluates a sequence of weighted
states, stops for an outer-loop reason, and then refits the accepted active
subset. Before R7, a post-loop refit without weights could be combined with the
realization from the preceding outer evaluation and synthetic NaN candidate
diagnostics. A numerical final refit also replaced an already established
outer stop reason. The returned mask, fit, geometry, records, and termination
could therefore describe different state generations even though ADR 0014
correctly associated them with the same source observations.

Observation identity alone cannot prove weight-state identity. The active
result needs one atomic assembly boundary that consumes ADR 0014 provenance
and distinguishes historical path evidence from final weights-dependent data.

## Decision

### One private accepted state

Every return from `solve_self_consistent_power_weights(...)` is constructed
through one private accepted-state representation. Its private origin records:

- the full candidate observation/source identity from ADR 0014;
- the ordered active row IDs and exact active mask;
- the accepted outer iteration; and
- whether the state was created by an outer failure or a post-loop final refit.

The accepted state validates that the final fit originates from exactly
`constraints.subset(active_mask)`. Candidate realization and diagnostics, when
present, originate from the full candidate observations. Their active mask,
row IDs, realization arrays, final weights, and optional tessellation
diagnostics must agree before the public result is built. Active reports
reconstitute and validate the same private state from the public result before
serialization.

This consumes the ADR 0014 association and row-ID helpers. It adds no UUID,
public state fingerprint, or alternative source identity.

### Outer stop and final inner status remain separate

`SelfConsistentPowerFitResult.termination` remains the outer-loop stop reason.
A failure confined to the post-loop final refit does not replace an established
`self_consistent`, `cycle_detected`, or `max_outer_iter` stop.

`result.fit.status` and `result.fit.converged` describe the final accepted
inner fit/refit. `result.converged` remains true exactly when the outer
termination is `self_consistent`. The computed property
`final_refit_converged` exposes `bool(result.fit.converged)` without changing
either meaning.

### Available and unavailable final layers

A final `optimal` or `max_iter` fit with complete finite weights and backend
radii is available. Existing certified zero-L2 gauge alignment may be applied,
after which the fit is rebuilt and all candidate predictions, residuals,
realization, diagnostics, summaries, marginal state, and requested
tessellation diagnostics are recomputed from that exact final vector. A
weighted `max_iter` fit remains non-converged at the inner layer.

When the final fit has no usable weights, the fit and active subset remain, but
these weights-dependent result fields are `None`:

```text
realized
diagnostics
rms_residual_all
max_residual_all
tessellation_diagnostics
```

The corresponding `final_realization`, `candidate_diagnostics`, and
`to_records()` views also return `None`. Path history, path summary,
connectivity, warnings, cycle metadata, and path-derived marginal indices
remain available. No earlier realization is reused and no NaN diagnostic row
is fabricated.

`final_state_available` is true exactly for a coherent available state.
`final_state_unavailable_reason` is `None` when available and otherwise is the
existing final `fit.status`. No second failure vocabulary or stored public
availability field is introduced.

A fit claiming a weighted status without a complete finite weight/radius
vector is normalized to the existing structured `numerical_failure` boundary
where safe. Other inconsistent combinations fail clearly rather than producing
a misleading active result.

### Active report availability

The ADR 0014 report envelope, schema name/version, source provenance, row IDs,
and `kind="self_consistent_power_fit"` remain unchanged. Active reports add:

```json
"availability": {
  "weights": true,
  "realization": true,
  "records": true,
  "reason": null
}
```

All three flags are false for an unavailable state and `reason` is the final
fit status. The nested fit report remains present. Unavailable realization,
candidate diagnostics, marginal records, tessellation diagnostics, and final
realization/residual summary values are JSON null. The active summary retains
the outer stop while the nested fit retains the final inner status and
convergence.

Every active report, including a no-weights failure, round-trips exactly
through strict JSON with no NaN or infinity.

### Algorithm boundary

This decision changes final-state assembly, not the active-set algorithm. It
does not change hysteresis, add/drop rules, cycle detection, relaxation,
weight-step termination, connectivity policy, gauge-alignment certification,
the fixed-row objective, or solver mathematics. Positive-L2 components remain
ineligible for final gauge alignment. The outer loop remains experimental and
has no new global-convergence claim.

## Consequences

- Callers can distinguish outer self-consistency from final inner convergence.
- A no-weights result is inspectable and serializable without implying that
  final geometry exists.
- Existing successful field names and record schemas remain; the experimental
  active fields and views become optional only where their inputs do not exist.
- Active report consumers must accept the additive `availability` block and
  JSON null in unavailable weights-dependent sections.
- ADR 0014 provenance remains the only observation/source identity system.

## Alternatives considered

### Reuse the last successful realization

Rejected. Source identity can match while the accepted weight vector or active
mask differs, so the result would still mix state generations.

### Fabricate empty geometry and NaN records

Rejected. This presents unavailable geometry as computed data and violates the
strict finite-JSON report contract.

### Replace the outer stop with the final fit status

Rejected. Outer-loop termination and fixed-fit termination answer different
questions and both are needed for diagnosis.

### Add a public state ID or new failure reason vocabulary

Rejected. Private ADR 0014 identity plus invariant checks are sufficient, and
the existing final fit status already states why weights are unavailable.
