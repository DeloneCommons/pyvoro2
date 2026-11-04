# Documentation refresh — workstream draft

- **Status:** Draft
- **Plan revision:** 0.1
- **Target:** Documentation of current v0.8 behavior, followed by incremental v0.9 updates; not an activated v0.9 release plan
- **Base release:** v0.8.0 (reboot source, ADR 0027)
- **Integration branch:** `dev`
- **Maintainer:** pyvoro2 maintainer
- **Approved:** Not yet approved
- **Last revised:** 2026-11-04
- **Milestone:** None

## Intended outcome

Make pyvoro2 understandable to new users, scientific readers and maintainers: clear first contact, runnable recipes, API-independent mathematics, precise callable reference, and source/data-flow architecture explanations. Explicitly separate implemented behavior from candidates.

## Baseline and supporting material

- The current documentation has User Guide, Theory, API Reference, Examples, and Development sections under MkDocs Material.
- Follow the [documentation conventions](../documentation-conventions.md), [API lifecycle](../api-lifecycle.md), and [ADR 0027](../decisions/0027-v0.9-reboot-from-v0.8.md).
- The complete maintainer notebook is retained in [notes/maintainer-review](https://github.com/DeloneCommons/pyvoro2/tree/dev/notes/maintainer-review), with one [candidate register](https://github.com/DeloneCommons/pyvoro2/blob/dev/notes/maintainer-review/review-register.md), a [non-blocking functional coverage checklist](https://github.com/DeloneCommons/pyvoro2/blob/dev/notes/maintainer-review/functional-coverage.md), and [documentation working notes](https://github.com/DeloneCommons/pyvoro2/tree/dev/notes/maintainer-review/documentation-notes). These are research and planning evidence, not implemented guarantees or automatic release commitments.
- The revised powerfit manuscript can inform theory and figures after verifying terminology, correctness, attribution, permissions, and public availability. See [issue #19](https://github.com/DeloneCommons/pyvoro2/issues/19); do not close it merely because documentation work starts.

## Proposed sequence

| Part | Goal | Timing |
| --- | --- | --- |
| Foundation | Evaluate Zensical migration, existing plugins/MathJax/mkdocstrings/notebook compatibility, landing page, navigation and a polished static hero image. | First documentation work, with compatibility review before engine switch. |
| v0.8 exposition | Improve guides, theory, architecture and API-contract cross-links for implemented forward, periodic, result, diagnostics and inverse workflows. | Before or alongside first high-priority v0.9 implementation, not blocked by full candidate triage. |
| v0.9 changes | Update relevant pages and examples together with each separately approved feature. | During actual implementation. |
| Final pass | Synchronize terminology, versions, screenshots/figures, notebook outputs, API reference and strict builds. | After eventual v0.9 implementation and scope decisions. |

## Presentation and illustrations

Use four levels: introduction, scientific/user workflows, implementation architecture, and development/maintenance. For each important capability cross-link three views: a runnable recipe, its mathematical/API contract, and its code/data-flow route. Begin with simple two-site geometry and distinguish exact mathematics, numerical limitations, implementation and unapproved research.

For the first Zensical landing page, prefer a static illustration built from genuine pyvoro2 geometry by an optional external PyVista script. Do not make improved py3Dmol or K3D a documentation prerequisite. Reassess visuals at the final pass; Matplotlib remains the primary 2D engine, while K3D and Plotly are only possible future additions. Do not mistake visually assembled local cells for a certified global periodic complex.

## Non-goals and boundaries

- No new public API, numeric policy, C++/Voro++ change, visualization backend, optional dependency becoming mandatory, or adopted release scope follows from this draft.
- The [functional review checklist](https://github.com/DeloneCommons/pyvoro2/blob/dev/notes/maintainer-review/functional-coverage.md) is **not a development gate**. Independently approved high-priority work on global point/edge identity and inverse per-row model parameters can proceed without completing it.
- Do not copy whole numerical experiments from the manuscript or treat its molecular application as the only use case of the generic package.
- Preserve generated README and notebook synchronization rules until a separately approved modification changes them.

## Open decision gates

| Gate | Question | Evidence needed | Status |
| --- | --- | --- | --- |
| DOC-1 | Zensical configuration and extension compatibility | Compare representative builds: math, API reference, notebooks, navigation and strict checks. | Open |
| DOC-2 | README and landing page ownership | Preserve useful GitHub README without unnecessary duplicated sources. | Open |
| DOC-3 | Hero image / manuscript assets | Reproducible geometry, scientific fidelity, rights and attribution. | Open |
| DOC-4 | Final v0.9 documentation acceptance | Determine after functionality and release scope are selected. | Open |

## Draft work packages

| ID | Suggested work | Depends on | Status |
| --- | --- | --- | --- |
| DOC-W1 | Zensical, navigation, first landing page and static image | DOC-1 to DOC-3 | Proposed |
| DOC-W2 | v0.8 user, theory, architecture and API-contract documentation | Current source/tests and existing notes | Proposed |
| DOC-W3 | Feature-driven documentation updates | Corresponding separately approved v0.9 features | Conditional |
| DOC-W4 | Final consistency and illustration audit | Delivered functionality, DOC-4 | Conditional |

## Validation

Require runnable examples for documented versions, correct present/proposed language, preserved notebook and generated-README synchronization, clear links and mathematical notation, and successful strict documentation builds. The currently applicable build is `mkdocs build --strict`; choose an equivalent check if the engine migration is approved. This is a documentation quality target, **not a new v0.9 release gate**.

## Risks

Prevent documentation work from expanding into renderer/backend refactoring, avoid claims for unimplemented candidate features, and keep one decision register rather than two contradictory backlogs. A separate release plan and feature-level approvals govern actual implementation.

## Changes and outcome

| Date | Revision | Change |
| --- | --- | --- |
| 2026-11-04 | 0.1 | Initial draft after maintainer reassessment and documentation-order discussion. |

**Outcome:** Not applicable while Draft. Record implementation outcomes and actual validation when work is accepted and completed.
