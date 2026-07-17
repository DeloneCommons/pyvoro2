# API lifecycle and compatibility

This policy defines what pyvoro2 means by a stable, provisional, experimental,
compatibility-only, or internal interface. It applies not only to Python imports
and signatures, but also to defaults, result schemas, documented record keys,
units, orientation conventions, and scientifically meaningful semantics.

The policy is intentionally stronger than “the name still imports.” Numerical
and geometric meaning are part of the API.

## Lifecycle categories

| Category | Intended use | Compatibility expectation |
|---|---|---|
| Stable | Recommended public API | Normal caller behavior is preserved or changed through deprecation |
| Provisional | Public API undergoing downstream validation | Deliberate pre-1.0 refinement is possible with migration guidance |
| Experimental | Opt-in research functionality | Scope and limitations are explicit; schemas and algorithms may evolve |
| Compatibility-only | Older public paths retained during migration | Kept working and tested, but not presented as the preferred entry point |
| Internal | Implementation details | No compatibility guarantee |

### Stable

A stable surface is documented, tested, and covered by compatibility policy.
Changes should preserve normal caller behavior or follow the deprecation process.

Stable does not mean that every implementation detail is fixed. Backend,
algorithm, and performance changes are allowed when documented numerical
semantics and result contracts are preserved.

### Provisional

A provisional surface is public and documented because downstream testing is
needed, but refinement before 1.0 remains possible. Breaking changes require an
issue, changelog entry, migration notes, and a deliberate release decision.

### Experimental

An experimental surface supports active research. It may have limited domain
coverage, weaker convergence guarantees, or a changing result schema. It must be
explicitly labelled in documentation and should normally require an opt-in
namespace, flag, or class.

Experimental functionality must still fail honestly and return structured
diagnostics where practical.

### Compatibility-only

A compatibility surface exists to keep older callers working while new code is
directed to a preferred API. It is tested, but documentation may contain only a
migration summary rather than full new-user guidance.

### Internal

Internal modules and names are not covered by compatibility promises. A leading
underscore is one signal, but the decisive criterion is whether a name is
included in public documentation or exports.

Deprecation is a transition state rather than a separate maturity category. A
stable or compatibility-only surface may be deprecated when a replacement and
migration path are available.

## The v0.7 stabilization release

v0.7.0 is intended to establish a credible stable/provisional boundary for:

- forward domain and site input;
- forward standard and power computation;
- direct weight-first power computation;
- common access to cells, measures, boundaries, periodic shifts, and
  diagnostics;
- the preferred separator-observation fitting workflow;
- global gauge and disconnected-component-offset reporting;
- compatibility with the existing `pyvoro2.powerfit` workflow.

The exact public inventory must be published with the release. The
[v0.7 development plan](plans/v0.7.md) requires a dedicated inventory
page: this policy defines the categories, while the inventory assigns them to
concrete names, return routes, record schemas, and defaults. Future
prescribed-measure and mixed solvers may remain experimental without weakening
the stable forward and separator core.

v0.7.0 is not the same promise as 1.0. It is a stabilization release designed to
support real downstream integration and to expose remaining rough edges before
the stronger 1.0 commitment.

## What counts as a breaking change

Examples include:

- removing or moving a documented import without a compatibility path;
- adding a required argument;
- changing a default in a way that changes common results;
- changing accepted units, coordinate conventions, orientation, or periodic
  shift meaning;
- changing return type, field name, record key, or field interpretation;
- changing whether empty cells are omitted or represented;
- reordering outputs when order is documented or carries ID association;
- changing exception/status behavior on a scientifically meaningful failure;
- relabelling unidentified component offsets as uniquely determined values;
- silently switching between requested and nearest periodic images;
- changing numerical normalization in a reported metric without a new name or
  explicit migration note.

Performance improvements, internal refactors, and additional optional fields are
not breaking when documented behavior remains compatible.

## Deprecation process

For a normal stable API change:

1. document the preferred replacement;
2. keep an alias, adapter, or compatibility facade where practical;
3. add a warning when it is useful and not excessively noisy;
4. test the old and new paths;
5. add migration notes and a changelog entry;
6. retain the compatibility path for at least one minor release when feasible.

Correctness, security, or native-crash fixes may require faster action. Such
exceptions must be explicit in release notes.

Before 1.0, pyvoro2 may make a breaking change without a long deprecation period,
but it should still be issue-driven, documented, and accompanied by a usable
migration path whenever possible.

## Renaming and namespace evolution

A `pyvoro2.inverse` namespace is the leading candidate for the preferred home
of inverse concepts, but the v0.7 namespace choice remains an open decision.
Whatever is selected, `pyvoro2.powerfit` should remain a compatibility and
convenience surface during the transition.

Preferred practice:

- new guides use the preferred namespace and terminology;
- old imports remain functional through aliases or delegation;
- top-level re-exports are reduced only through a documented deprecation path;
- historical class names can remain as compatibility names even when prose uses
  clearer mathematical language.

## Result schemas

Result objects and exported record dictionaries are scientific interfaces.
Their stability includes:

- field names and types;
- units and orientation;
- missing/empty representation;
- relationship to site IDs and periodic shifts;
- definitions of residuals, objective values, and summary metrics;
- whether a value is measured, fitted, derived, or selected by policy.

New optional diagnostic fields may be added compatibly. Removing a field,
changing its meaning, or changing a normalized metric requires migration
handling.

Where possible, results should expose plain-record export separately from rich
NumPy objects. This gives downstream packages a stable serialization boundary
without forcing internal storage details to remain fixed.

## Numerical behavior and tolerances

Exact floating-point identity is usually not an API promise. Tests and docs
should instead specify stable invariants and tolerances appropriate to the
operation.

However, the following are semantic and require explicit review:

- gauge/component alignment policy;
- objective scaling and confidence interpretation;
- periodic image selection;
- empty-cell policy;
- connectivity and feasibility classification;
- residual and measure definitions;
- active-set termination/status vocabulary.

## Experimental inverse methods

A prescribed-measure or mixed solver can be released experimentally when:

- supported domains are explicit;
- target validation is implemented;
- generated-data recovery tests exist;
- gauge and empty-cell behavior are documented;
- non-convergence returns structured diagnostics;
- no stable API is forced to depend on the experimental solver.

Promotion from experimental to provisional or stable requires downstream use,
benchmark coverage, and a public API audit.

## Downstream validation

chemvoro is an intended early downstream consumer. Before declaring the v0.7 API
stable, the project should verify that a chemistry-facing package can:

- preserve atom IDs through forward and inverse workflows;
- compute directly from weights;
- consume cells, measures, boundaries, and shifts without private imports;
- interpret gauge and disconnected-component ambiguity correctly;
- serialize fit and realization diagnostics.

A downstream workaround based on private fields is evidence that the core API is
not yet stable.

## Reproducibility of archived research

The inverse-separator manuscript and archived experiments remain tied to
pyvoro2 v0.6.3. Later API changes do not require those experiments to be rerun
unless a change reveals a numerical or scientific issue affecting the reported
results.

Documentation may show a newer equivalent API, but the archived version and
paper-side environment remain the authoritative executable record for the
published numbers.

## Release audit

Before a stabilization or 1.0 release, follow the
[development workflow](development-workflow.md) and:

- inventory public exports and documented record schemas;
- label each surface by lifecycle category;
- verify migration aliases and warnings;
- run downstream integration examples;
- update guides, reference, changelog, and release notes;
- ensure experimental features are visibly marked;
- verify generated documentation and distributions.
