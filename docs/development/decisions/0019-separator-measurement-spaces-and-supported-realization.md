# 0019 — Independent separator measurement spaces and supported realization-aware fitting

- **Status:** Accepted
- **Date:** 2026-09-01
- **Related issue:** [#46 — Activate the v0.9.0 functional/API stabilization plan](https://github.com/DeloneCommons/pyvoro2/issues/46)
- **Related plan:** [active v0.9.0 development plan](../plans/v0.9.md)
- **Related decisions:** [ADR 0007](0007-separator-objective-contract.md),
  [ADR 0014](0014-separator-observation-and-source-identity.md),
  [ADR 0015](0015-atomic-separator-active-state.md), and
  [ADR 0017](0017-v0.9-functional-stabilization-before-1.0.md)

## Context

The v0.8 separator model uses `SeparatorObservations.measurement` both to state
how observation targets were supplied and, effectively, to select the space for
mismatch, hard feasibility, and scalar boundary penalties. Those are separate
scientific choices. A common example is to fit absolute separator position while
requiring a relative/fractional separator interval.

The existing realization-aware active-set engine already has the atomic
final-state and outer-termination contract fixed by ADR 0015, but ordinary use
still goes through the advanced separator namespace. v0.9 must separate these
two concerns without creating the later generic mixed-observation architecture
or redesigning the active-state contract.

This ADR fixes the v0.9 target contract before implementation. It does not claim
that the unchanged v0.8 source already exposes these surfaces.

## Decision

### Observation space remains source identity

`SeparatorObservations.measurement` remains exactly the space in which the
observation target was supplied: `"fraction"` or `"position"`. It remains part
of observation/source identity under ADR 0014. Changing model evaluation space
does not change row IDs, the observation-set fingerprint, source binding, or the
meaning of observation-facing target/prediction/residual views.

### Mismatch, hard feasibility, and each scalar penalty select space independently

The v0.9 model terms gain an optional keyword-only `space` with values
`"fraction"`, `"position"`, or `None`. `None` means “inherit
`SeparatorObservations.measurement` for this problem.” The target constructor
shapes are:

```text
SquaredLoss(*, space=None)
HuberLoss(delta=1.0, *, space=None)
Interval(lower, upper, *, space=None)
FixedValue(value, *, space=None)
SoftIntervalPenalty(lower, upper, strength, *, space=None)
ExponentialBoundaryPenalty(
    lower=0.0, upper=1.0, margin=0.02,
    strength=1.0, tau=0.01, *, space=None,
)
ReciprocalBoundaryPenalty(
    lower=0.0, upper=1.0, margin=0.05,
    strength=1.0, epsilon=1e-6, *, space=None,
)
```

Making `space` keyword-only preserves the meaning of every existing positional
constructor. `L2Regularization` remains a weight regularizer and does not gain a
separator space. `FitModel` remains the composition point; it does not gain a
second global measurement selector.

A mismatch term, the optional hard-feasibility term, and every scalar penalty
resolve inheritance independently. Different penalties may therefore use
different spaces without introducing arbitrary per-row policy.

### Compilation is an affine coordinate conversion, not a new model family

Each separator coordinate is compiled rowwise through the existing affine
relation between separator coordinate and the fitted weight difference. Hard
intervals become weight-difference feasibility bounds. Penalty limits, margins,
robust-loss scales, and strengths remain expressed in the units of the term's
declared space; any row-dependent coefficients created by conversion are
internal mathematical consequences.

v0.9 does **not** add per-row measurement spaces, per-row robust scales or
strengths, site anchors, generic observation blocks, prescribed cell measures,
or separator-plus-measure composition. Public point-centered/symmetric-bound
convenience is also not a release gate.

### Results keep observation views and add explicit effective model-space views

Existing observation-facing fields retain their source-space meaning. In
particular, `SeparatorFitResult.measurement`, `target`, `predicted`,
`residuals`, `rms_residual`, and `max_residual` continue to describe the
observation measurement space.

The v0.9 target adds read-only effective-space views rather than silently
reinterpreting those names:

```text
SeparatorFitProblem.mismatch_space
SeparatorFitProblem.hard_constraint_space
SeparatorFitProblem.penalty_spaces

SeparatorFitResult.mismatch_space
SeparatorFitResult.hard_constraint_space
SeparatorFitResult.penalty_spaces
SeparatorFitResult.mismatch_target
SeparatorFitResult.mismatch_predicted
SeparatorFitResult.mismatch_residuals
```

`hard_constraint_space` is `None` when no hard-feasibility term exists;
`penalty_spaces` follows `FitModel.penalties` order. `PowerFitBounds` gains a
`space` field identifying the effective hard-bound space while retaining its
measurement-space and weight-difference bound arrays. The existing
`PowerFitPredictions.measurement` remains the observation-space prediction.

These additions are model/result views; they do not alter observation identity
or introduce a new public result hierarchy.

### Report schema changes explicitly

Because fit-record exact keys change, separator reports move from schema version
`1` to version `2` when WP10 lands. The schema name remains
`pyvoro2.inverse.separator.report`.

Fit records add exactly:

```text
mismatch_space
mismatch_target
mismatch_predicted
mismatch_residual
```

while the existing `measurement`, `target`, `predicted`, and `residual` keys
remain observation-facing. Active per-constraint records add the same four
keys. Fit report summaries retain `measurement` as observation space and add
`mismatch_space`. Fit reports also add

```text
model_spaces = {
    "mismatch": "fraction" | "position",
    "hard_constraint": "fraction" | "position" | None,
    "penalties": ["fraction" | "position", ...],
}
```

using resolved effective spaces rather than unresolved `None` inheritance.
Nested fit information in the realization-aware report follows the same
contract. Observation-only row records and the `observation_set` identity block
do not gain model-space identity.

### Realization-aware fitting gets one supported high-level facade

The fixed-observation function `fit_weights_from_separators(...)` remains the
mathematical inner solve. v0.9 adds the distinct supported high-level function

```text
pyvoro2.inverse.fit_self_consistent_weights_from_separators(...)
```

returning the existing `SelfConsistentPowerFitResult`. The target public facade
accepts the normal resolver/model/inner-solver options already used by the
advanced engine plus `max_outer_iter=25` and requested final output layers. It
does not expose initial active masks, add/drop hysteresis, relaxation,
cycle-window, weight-step, history, or path/research controls.

The exact target signature is maintained in the v0.9 API inventory. Internally
the facade may construct the advanced `ActiveSetOptions` needed to reuse the
current engine; that type does not become part of the preferred namespace.

`SelfConsistentPowerFitResult` keeps ADR 0015's atomic accepted-state contract.
Outer termination (`self_consistent`, cycle, iteration limit, infeasibility,
numerical failure, and other defined structural outcomes) remains distinct from
final inner-fit status. Optional final layers remain `None` when they are not
available; the facade must not manufacture placeholders.

The supported facade and result are **Provisional** during the released v0.9.x
downstream-soak phase: they are normal supported public API, not Experimental,
but remain eligible for the final pre-1.0 lifecycle audit. The low-level
`solve_self_consistent_power_weights`, `ActiveSetOptions`, iteration/path views,
hysteresis controls, and research history remain **Experimental** in the
advanced `pyvoro2.inverse.separator` surface.

## Consequences

- A model can optimize one separator coordinate while constraining or penalizing
  another without changing observation provenance.
- Existing same-space models keep their meaning through `space=None`
  inheritance.
- Positional constructors do not acquire a silent new positional argument.
- Result/report consumers can distinguish observation residuals from the values
  actually used by mismatch evaluation.
- Report readers receive an explicit schema-version boundary instead of a
  silent exact-key change.
- Ordinary realization-aware fitting no longer requires an Experimental import,
  while advanced algorithm controls remain free to evolve before 1.0.
- ADR 0015 remains the authority for atomic final-state availability and is not
  redesigned by this promotion.

## Alternatives considered

### Change `SeparatorObservations.measurement` to mean objective space

Rejected. It would conflate source identity with model policy and invalidate the
provenance contract established by ADR 0014.

### Add one global model measurement override

Rejected. Mismatch, hard feasibility, and scalar penalties are independent
scientific choices; one override would preserve the current coupling in a new
form.

### Add per-row spaces or generic mixed observation blocks now

Rejected. That is broader than the v0.9 separator-stabilization scope and would
pre-empt the later mixed inverse architecture.

### Put realization awareness behind a mode flag on `fit_weights_from_separators`

Rejected. Fixed-observation convex fitting and empirical realization-aware
outer refinement have different termination semantics and should remain
visibly different algorithms.

### Promote every active-set control with the supported facade

Rejected. Normal callers need a supported workflow, not a commitment to the
research/path-control surface.
