# 0015 — Atomic separator active-set final state

- **Status:** Accepted
- **Date:** 2026-08-15
- **Related issue:** [#43 — Make the separator active-set final state atomic and self-consistent](https://github.com/DeloneCommons/pyvoro2/issues/43)
- **Related plan:** [v0.8 remediation execution plan](../plans/archive/v0.8-remediation.md)
- **Depends on:** [ADR 0014](0014-separator-observation-and-source-identity.md)
- **Numerical-availability amendment:** [ADR 0026](0026-separator-final-state-and-diagnostic-availability.md), issue #116

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
`constraints.subset(active_mask)`. When the v0.9 row-bound model policy is
present, the final fit's resolved policy must likewise be exactly the projection
of the retained full candidate policy through that same ordered active mask;
row removal and later re-entry recover the original candidate policy.
Candidate realization and diagnostics, when present, originate from the full
candidate observations. Their active mask, row IDs, realization arrays, final
weights, optional tessellation diagnostics, and applicable resolved-policy
association must agree before the public result is built. Active reports
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

Issue #116 amends the former finite-only source-diagnostic gate. An otherwise
coherent state with finite weights/radii, selected fit, genuine realization and
source association retains its diagnostic container when individual derived
source values are producer-proven outside binary64 or have an unavailable
dependency. These raw cells may be signed infinity or proven dependency NaN;
they are not placeholders for missing geometry. Arbitrary nonfinite values,
stale producer snapshots and matching-but-wrong outer/nested residuals remain
invalid. Final validation recomputes complete affine rows from exact final
weights and checks the nested fit residuals independently.

Each historical source summary describes its own aligned/relaxed
`weights_eval`. Immutable private iteration/source/value/reason records survive
supported copying and reconstruction, without retaining historical weight
vectors or reconstructing history from the final fit. Ordinary finite manual
history remains supported; unexplained nonfinite history is rejected.

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

The ADR 0014 source provenance, row IDs, schema name and
`kind="self_consistent_power_fit"` remain unchanged. ADR 0026 advances all
report envelopes to schema v3 and adds the bounded diagnostic availability map.
Active reports retain:

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
through strict JSON with no NaN or infinity. Present but numerically unavailable
eligible diagnostic cells become null with local/exhaustive
`unavailable_diagnostics` reasons. Structural nulls receive no reason. Nested
fit/realized reports and marginal copies retain their own maps; history uses
the containing report map. The closed whitelist and pointer grammar are fixed
by ADR 0026, not extended to geometry, objectives or stopping quantities.

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
