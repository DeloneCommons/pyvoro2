# 0019 — Independent separator measurement spaces and supported realization-aware fitting

- **Status:** Accepted
- **Date:** 2026-09-01
- **Amended:** 2026-10-03 — Phase C entry gate [#104](https://github.com/DeloneCommons/pyvoro2/issues/104) accepts a bounded row-bound A+B policy; row-wise shape/robustness parameters are staged for a mandatory late-Phase-C decision before Checkpoint C.
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

### Measurement spaces remain term-global; selected term values may be row-wise

The v0.9 model terms gain an optional keyword-only `space` with values
`"fraction"`, `"position"`, or `None`. `None` means “inherit
`SeparatorObservations.measurement` for this problem.” Measurement-space
selection remains **term-global**: v0.9 does not add per-row spaces.

The target constructor shapes are:

```text
SquaredLoss(*, space=None)
HuberLoss(delta=1.0, *, space=None)
Interval(lower, upper, *, applicable=True, space=None)
FixedValue(value, *, applicable=True, space=None)
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

Making `space` and hard `applicable` keyword-only preserves the meaning of
existing positional constructors. `L2Regularization` remains a weight
regularizer and does not gain a separator space or observation-row strength.
`FitModel` remains the composition point; it does not gain a global
measurement selector.

WP10 accepts either a scalar or one owned one-dimensional length-`m` row
vector for this bounded whitelist:

- `Interval.lower`, `Interval.upper`, and `Interval.applicable`;
- `FixedValue.value` and `FixedValue.applicable`;
- `SoftIntervalPenalty.lower`, `.upper`, and `.strength`;
- `ExponentialBoundaryPenalty.lower`, `.upper`, and `.strength`;
- `ReciprocalBoundaryPenalty.lower`, `.upper`, and `.strength`.

Scalars broadcast. Row vectors are positional policy until problem
construction, require exact length `m`, are defensively owned/read-only, and
reject matrices, column vectors, non-finite scientific values, or invalid
Boolean applicability values. A vector-valued model reused with another
equal-length observation set is a new positional assignment, not proof of
semantic identity.

Hard applicability is independent of mismatch confidence and active-set
membership. For an inapplicable hard row there is no feasibility restriction,
hard-conflict edge, hard-violation classification, or hard-induced structural
coupling. Hard closed intervals may use `lower == upper` to represent
equality; this does not relax the existing positive-width requirements of soft
or boundary penalties. `FixedValue` remains the convenience spelling for an
equality restriction.

Penalty strength zero remains the mathematical absence mechanism defined by
ADR 0007 and is removed before numerically dangerous branch evaluation. A
positive-strength penalty remains structural model coupling even if its value
happens to vanish at the current iterate.

The following shape/robustness fields remain term-global in the first WP10
implementation: `HuberLoss.delta`, exponential `margin`/`tau`, and
reciprocal `margin`/`epsilon`. Their row-wise disposition is a mandatory
parameter-by-parameter late-Phase-C decision after WP10/WP11 and before
Checkpoint C. This staging does not permit a temporary scalar-common compiler:
the bound/compiler architecture must already support row-specific scalar
objective specifications created by the accepted A+B fields.

### Compilation binds row policy before numerical solving

Each separator coordinate is compiled rowwise through the existing affine
relation between separator coordinate and fitted weight difference. Hard
intervals become weight-difference feasibility bounds. Penalty parameters
remain expressed in their term's declared measurement units.

A `FitModel` containing row vectors is an **unbound positional policy** until
problem construction. Construction binds it to one ordered
`SeparatorObservations` set and produces one resolved owned row policy. Every
component solve, active subset, final refit, and later row re-entry must project
observations and that bound policy through the same ordered selection. The
candidate policy is retained so re-entering rows recover their original
configuration.

Binding is model policy, not observation/source identity. Row IDs,
observation-set fingerprints, and ADR 0014 source binding are unchanged. No
public model-policy UUID or second identity system is introduced; exact
association plus invariant checks are sufficient.

The numerical implementation keeps the existing convex scalar objective
family. A row-specific immutable scalar objective specification may reuse the
existing scalar evaluator/proximal/certificate machinery. Identical complete
specifications may share compilation or batching, but caches must include all
policy that can affect the scalar objective. Rows may not exchange proximal
solutions merely because targets/confidences/hard endpoints happen to match.

Internal coefficients that become row-dependent solely because one term space
is converted through a row-specific affine map remain implementation
consequences rather than new public row parameters. A+B therefore adds
heterogeneous coefficients inside the existing separable convex model, not a
new inverse family.

### Results keep observation views and expose resolved model policy

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
`penalty_spaces` follows `FitModel.penalties` order. `PowerFitBounds` gains
a `space` field identifying the effective hard-bound space while retaining its
measurement-space and weight-difference bound arrays. The existing
`PowerFitPredictions.measurement` remains the observation-space prediction.

Problems/results must also provide an inspectable read-only resolved-policy
view sufficient to determine the selected hard applicability/values and
penalty values associated with their ordered observation rows. The concrete
container name may remain provisional within WP10, but it must describe the
**bound/projected** policy rather than only the unresolved input template.

These additions are model/result views; they do not alter observation identity
or introduce a second public identity hierarchy.

### Report schema v2 represents resolved row policy

Separator reports move from schema version `1` to version `2` when WP10
lands. The schema name remains `pyvoro2.inverse.separator.report`. The v2
grammar is intentionally able to represent both uniform and row-varying
resolved policy so a later accepted C parameter does not require another schema
generation.

Fit records add:

```text
mismatch_space
mismatch_target
mismatch_predicted
mismatch_residual
```

while existing `measurement`, `target`, `predicted`, and `residual`
remain observation-facing. Active per-constraint records add the same mismatch
fields. Fit summaries retain `measurement` as observation space and add
`mismatch_space`.

Fit reports retain an exact effective-space block:

```text
model_spaces = {
    "mismatch": "fraction" | "position",
    "hard_constraint": "fraction" | "position" | None,
    "penalties": ["fraction" | "position", ...],
}
```

and add a resolved `model_policy` block aligned with
`observation_set.row_ids`. Each scientific parameter is represented either as

```json
{"kind": "uniform", "value": 0.02}
```

or

```json
{"kind": "rows", "values": [0.02, 0.05, 0.03]}
```

with Boolean hard applicability using the same uniform/rows distinction.
`model_policy` records configured term order, effective term spaces, hard
kind/applicability/values, and penalty kind/parameters. It records zero-strength
penalties and inapplicable hard rows as configured policy rather than erasing
them. Effective absence must not be serialized as infinite bounds.

For the active/self-consistent report, the outer report retains the full
candidate policy aligned with candidate row IDs. Its nested final fit retains
the projected selected policy aligned with the nested fit's row IDs. Their
association must therefore be reconstructible without treating physical pair
identity or equal row count as a key. History does not repeat the full policy at
every iteration.

Observation-only records and the `observation_set` identity block remain
model-independent. Reports with unavailable final weights still retain known
policy and source/row association while weights-dependent sections remain null
under ADR 0015.

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
- Heterogeneous hard, unrestricted, equality-like, and soft-prior rows can be
  represented in one joint fixed-separator problem without abusing confidence
  or artificial wide bounds.
- Scalar model inputs remain valid through broadcasting.
- Row-bound policy follows component and active subsets atomically with
  observation rows; removal/re-entry preserves candidate policy.
- Existing same-space models keep their meaning through `space=None`
  inheritance.
- Positional constructors do not acquire a silent new positional argument.
- Result/report consumers can distinguish observation residuals from the values
  actually used by mismatch evaluation and can inspect resolved row policy.
- Report readers receive one schema-v2 boundary designed to accommodate a later
  accepted C parameter without changing record meaning.
- The fixed problem remains in the existing affine/separable convex solver
  family; A+B does not authorize generic mixed observation blocks or callbacks.
- Ordinary realization-aware fitting no longer requires an Experimental import,
  while advanced algorithm controls remain free to evolve before 1.0.
- ADR 0015 remains the authority for atomic final-state availability and gains
  only the requirement that final selected policy be the exact projection of
  candidate policy.

## Alternatives considered

### Change `SeparatorObservations.measurement` to mean objective space

Rejected. It would conflate source identity with model policy and invalidate the
provenance contract established by ADR 0014.

### Add one global model measurement override

Rejected. Mismatch, hard feasibility, and scalar penalties are independent
scientific choices; one override would preserve the current coupling in a new
form.

### Add per-row spaces or generic mixed observation blocks now

Rejected. The accepted Phase-C amendment is deliberately narrower: selected
existing **values/applicability** may vary by observation row, but measurement
space remains term-global and the model remains one existing separator
objective family. Per-row spaces or generic mixed blocks would pre-empt the
later mixed inverse architecture.

### Make every shape/robustness parameter row-wise in the first WP10 pass

Rejected as an initial scope rule. A+B already requires the correct row-bound
compiler architecture, while the scientific need for each of
`HuberLoss.delta`, exponential `margin`/`tau`, and reciprocal
`margin`/`epsilon` is not equally established. Their exact disposition is a
mandatory late-Phase-C gate after WP10/WP11 and before Checkpoint C.

### Put realization awareness behind a mode flag on `fit_weights_from_separators`

Rejected. Fixed-observation convex fitting and empirical realization-aware
outer refinement have different termination semantics and should remain
visibly different algorithms.

### Promote every active-set control with the supported facade

Rejected. Normal callers need a supported workflow, not a commitment to the
research/path-control surface.
