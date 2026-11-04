# Documentation refresh — workstream draft

- **Status:** Draft
- **Plan revision:** 0.2
- **Target:** Documentation of current v0.8 behavior, followed by incremental v0.9 updates; not an activated v0.9 release plan
- **Base release:** v0.8.0 (reboot source, ADR 0027)
- **Integration branch:** `dev`
- **Maintainer:** pyvoro2 maintainer
- **Planning agreement:** Initial order and working principles agreed on 2026-11-04; workstream activation and later content scope remain open
- **Last revised:** 2026-11-04
- **Milestone:** None

## Intended outcome

Make pyvoro2 understandable to new users, scientific readers and maintainers: clear first contact, runnable recipes, API-independent mathematics, precise callable reference, and source/data-flow architecture explanations. Explicitly separate implemented behavior from candidates.

## Baseline and supporting material

- The existing site has User Guide, Theory, API Reference, Examples and Development sections built with MkDocs Material.
- Follow [documentation conventions](../documentation-conventions.md), [API lifecycle](../api-lifecycle.md), and [ADR 0027](../decisions/0027-v0.9-reboot-from-v0.8.md).
- The complete [maintainer notebook](https://github.com/DeloneCommons/pyvoro2/tree/dev/notes/maintainer-review) holds a [candidate register](https://github.com/DeloneCommons/pyvoro2/blob/dev/notes/maintainer-review/review-register.md), a [non-blocking functional coverage checklist](https://github.com/DeloneCommons/pyvoro2/blob/dev/notes/maintainer-review/functional-coverage.md), and [documentation working notes](https://github.com/DeloneCommons/pyvoro2/tree/dev/notes/maintainer-review/documentation-notes). They are not automatic release commitments.
- The revised powerfit manuscript may inform theory and figures after checking correctness, terminology, attribution, reuse permissions and public availability. [Issue #19](https://github.com/DeloneCommons/pyvoro2/issues/19) remains separately scoped.

## Sequencing and evolving scope

**DOC-W1 → DOC-W2 → DOC-W3 → DOC-W4** are the agreed initial order. **DOC-W99** reserves the final whole-site scientific and editorial audit, deliberately separate from the initial sequence.

Between DOC-W4 and DOC-W99, add only concrete, reviewable tasks as the documentation develops: an examples gallery, getting-started paths, guides, theory, architecture, or feature-driven v0.9 updates. **Do not define DOC-W5 or DOC-W6 yet.** There is no fixed number of intermediate tasks or requirement to complete the entire functional coverage review before starting important implementation work.

An evolving **umbrella issue** is the operational tracker: initially show DOC-W1–W4 and DOC-W99, then add PR links and further tasks as they are actually scoped. This document describes the durable writing strategy and boundaries, not a duplicate completion checklist. Change this plan when major scope or decisions change; update the issue for routine progress.

Work in focused PRs. A major rewrite may temporarily keep a comparison draft outside published `docs/`, but replace the canonical page only after comparing content, links and limitations. Remove superseded `*_old` pages before merging; do not publish parallel contradictory versions. Favor stable canonical URLs.

## Initial work packages and reserved final audit

| ID | Work package | Boundaries and dependency | Status |
| --- | --- | --- | --- |
| **DOC-W1** | **Separate README from the site.** Maintain a concise, independent GitHub/PyPI `README.md`; remove `tools/gen_readme.py`, its sync test, and related CI, release, packaging and contributor references. Keep `docs/index.md` intact for now. | No site engine change and no numerical/API changes. | Selected first |
| **DOC-W2** | **Migrate site builds to Zensical.** Validate navigation, search, MathJax, mkdocstrings, CSS/JS, notebooks, GitHub Pages and strict CI; update dependencies and build commands. Keep page content materially unchanged. | After DOC-W1. Preserve notebook validation and the current non-executing notebook-to-Markdown exporter. | Selected second |
| **DOC-W3** | **Design site navigation.** Agree at least two levels of structure, Forward/Inverse entry paths, example gallery vs tutorials, destination URLs and Theory/API/Development relationships. | After DOC-W2; agree destinations before rebuilding the landing page. | Selected third |
| **DOC-W4** | **Rebuild the landing page.** Present Forward tessellation and Inverse fitting as two prominent entry points, with an accurate introductory message, a reproducible static illustration, accessible lightweight hover/focus behavior and working destination links. | After DOC-W3. No new renderer backend required. | Selected fourth |
| **DOC-W99** | **Final whole-site audit.** Align actual delivered v0.9 behavior, API/math claims, examples, notebooks, illustrations, navigation, links, accessibility, attribution and strict build. | Reserved for the end of the eventual documentation workstream; not the immediate successor of DOC-W4. | Reserved; not scheduled |

**No DOC-W5 or DOC-W6 is specified by this plan.** New intermediate work is introduced only when its content, validation and dependencies are understood.

## Presentation and visual direction

Use four reader levels: introduction; scientific/user workflows; implementation architecture; and development/maintenance. For important capabilities link three complementary views: a runnable recipe, its mathematical/API contract, and its implementation/data-flow route. Begin with two-site geometry and separate exact mathematics, numerical limitations, current code and research proposals.

The initial landing image can be rendered from actual pyvoro2 geometry by a standalone optional PyVista script. The two entry points should explain (1) Voro++-based native 2D/3D Voronoi/power computations, supported periodic domains and derived geometry; (2) how observations of pairwise separator positions constrain weight differences, with algebraic compatibility distinct from realized face contacts. Do not promise unsupported periodicity or certified global topology.

Explore a gallery of thumbnail-first scientific examples with short reproducible code: supported 2D/3D, domain/periodicity and standard/power cases; plus a pedagogical inverse collection. This is a **future candidate**, not a deliverable of DOC-W4. The gallery complements executable notebooks. Hover styles must have keyboard-focus and reduced-motion equivalents and must not hide essential navigation.

Matplotlib remains the current 2D visualization helper. K3D and Plotly are potential later additions; PyVista is not a new required runtime dependency. Do not block the initial landing page on improvements to package visualizers.

## Optional upstream research: Zensical notebooks and galleries

Direct publication of `.ipynb` and an examples-gallery module may be useful to other scientific packages, but are **not part of DOC-W2**. Revisit [Zensical notebook backlog #9](https://github.com/zensical/backlog/issues/9) and [gallery discussion #96](https://github.com/zensical/zensical/issues/96), and consult upstream maintainers about a supported module API before promising an external plugin. The current non-executing Markdown exporter remains the default until a separately validated replacement exists.

## Decision gates

| Gate | Question | Status |
| --- | --- | --- |
| DOC-1 | Independent README wording, links and removal of the generator/checks without breaking package metadata or release validation | Initial direction agreed; validate in DOC-W1 |
| DOC-2 | Zensical feature parity and new strict build/deployment procedure | Open for DOC-W2 |
| DOC-3 | Landing destinations, gallery/navigation blueprint, image provenance and accessible interactions | Open for DOC-W3/W4 |
| DOC-4 | Final acceptance of documentation relative to the actual v0.9 scope | Deferred to DOC-W99 |

## Non-goals

- No implementation of new pyvoro2 numerical algorithms, inverse models, renderer backends, public API changes, or release guarantees is authorized here.
- No full-site rewrite, gallery generator or native `.ipynb` renderer is required to complete DOC-W1–W4.
- The [functional review checklist](https://github.com/DeloneCommons/pyvoro2/blob/dev/notes/maintainer-review/functional-coverage.md) is not a blocking dependency; approved high-priority geometry/inverse implementation can proceed independently.
- Do not assume the powerfit manuscript is already public or reproduce its experiments wholesale.

## Validation and maintenance

Check that current examples run for their documented version, math and API claims are correct, internal links and assets resolve, notebook validation/export remain synchronized, and the strict site build succeeds. DOC-W1 removes the *README generation* check, not the README itself or normal package validation. DOC-W2 must verify visual output and API pages, not only a successful exit status. At the moment `mkdocs build --strict` remains authoritative; replace it with a verified Zensical equivalent during DOC-W2.

This workstream is not a newly imposed v0.9 release gate. Implementation outcomes and PR progress belong in the umbrella issue; accepted durable decisions or major plan scope changes belong here.

## Risks

Avoid coupling content rewrites to a new site engine, treating tentative v0.9 features as delivered, or maintaining both the review notes and public site as independent normative authorities. Review and retire temporary notes as their useful content is integrated.

## Plan revisions

| Date | Revision | Change |
| --- | --- | --- |
| 2026-11-04 | 0.1 | Initial draft after maintainer reassessment and documentation-order discussion. |
| 2026-11-04 | 0.2 | Define DOC-W1–W4, reserve DOC-W99, leave the middle open to umbrella updates, and record notebook-module research as optional. |

## Outcome

Not applicable while Draft. Record delivered content and checks when the workstream is completed.
