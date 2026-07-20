# Advanced separator fitting

`pyvoro2.inverse.separator` owns the separator implementation and exposes the
canonical core names alongside advanced model, problem, realization, report,
and diagnostic objects.

The fixed-observation fit is distinct from realization-aware active-set
refinement. The active-set API is experimental and separator-specific; it is a
practical outer algorithm without a universal convergence guarantee.

Historical names remain identity aliases during v0.7 and are planned for
removal in v0.8. New code should use the canonical five-name map documented in
the [compatibility reference](../powerfit/index.md).

## Layered result access

The provisional view types below organize existing result data without adding
fields to the established result dataclasses or copying their arrays.

| Owning result | Access | View or reused object |
|---|---|---|
| `SeparatorFitResult` | `.state` | `SeparatorFitStateView` |
| `SeparatorFitResult` | `.identification` | `SeparatorIdentificationView` |
| `SeparatorFitResult` | `.observation_view(observations)` | `SeparatorObservationView` |
| `SeparatorFitResult` | `.objective` | existing `PowerFitObjectiveBreakdown` or `None` |
| `SeparatorFitResult` | `.algebraic` | `SeparatorAlgebraicView` containing existing edge and connectivity diagnostics |
| `SeparatorFitResult` | `.solver_termination` | `SeparatorSolverTerminationView` |
| `RealizedPairDiagnostics` | `.requested_image_matching` | `RequestedImageMatchView` |
| `RealizedPairDiagnostics` | `.geometry` | `RealizedGeometryView` |
| `SelfConsistentPowerFitResult` | `.inner_fit`, `.final_realization`, `.candidate_diagnostics` | existing final objects |
| `SelfConsistentPowerFitResult` | `.outer_termination`, `.path` | `ActiveSetTerminationView`, `ActiveSetPathView` |

`observation_view(...)` accepts the originating resolved observations or a
fully equivalent independently resolved set. It checks pair indices,
confidence, targets, requested shifts, and resolved geometry before combining
observation-owned arrays with fit-owned predictions. The private binding is
retained by shallow copies, deep copies, pickle round trips, and
`dataclasses.replace(...)`. Directly constructed results without source
observations fail closed when this accessor is called.
`inspect.signature(SeparatorFitResult)` consequently includes the private
optional keyword-only parameter `_originating_observations_init=None`; it is an
init-only reconstruction channel, not a public result field or user input.

`SeparatorIdentificationView.unconstrained_sites` contains sites isolated in
the informative observation graph, which contains only positive-confidence
separator rows. Zero-confidence rows remain excluded even when hard
restrictions or penalties affect them: those terms may constrain or bound
component offsets, but they are not observational identification. Positive L2
regularization is the only currently supported additional objective reported as
guaranteed to select otherwise free component offsets. The compatibility
diagnostic `ConnectivityDiagnostics.unconstrained_points` retains its
established candidate-graph meaning.

The historical `SeparatorFitProblem.offset_identifying_constraint_mask` name
is preserved for compatibility. That mask includes rows touched by hard
restrictions or penalties because the numerical solver must keep coupled
variables in one subproblem; it does not define the informative observation
graph or claim unique offset selection. Exact hard equalities may fix offsets,
but the current identification view does not provide a general
constraint-identifiability classification.

`global_representation_shift` is a backend representation choice made by adding
one common constant, so it selects a representative within the global geometric
gauge. It is distinct from independent offsets between disconnected observation
components and is not inferred from observations.
`component_alignment_policy` is the canonical access to the policy string also
stored under the historical compatibility name `ConnectivityDiagnostics.gauge_policy`.
Public incidence, Laplacian, and normal-operator representations are not part
of these views.

::: pyvoro2.inverse.separator
:::
