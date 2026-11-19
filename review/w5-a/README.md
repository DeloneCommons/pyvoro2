# DOC-W5 PR A — Project and shared overview patterns

Proposal for [#143](https://github.com/DeloneCommons/pyvoro2/issues/143), under
[#127](https://github.com/DeloneCommons/pyvoro2/issues/127). Base `dev` was checked
against the remote before editing and resolved to
`87f01569015a2cc79908a002c03da12325d1d6cb`.
PR B and merge are explicitly outside this change.

## Approved composition

A text-led scientific project profile: serif title and introductory positioning,
a six-item Markdown capability list with scientific qualifications, a
published/development status pair, and three quiet information groups. All section headings are at one level.
The native documentation shell, navigation tree, URLs, Search and palette stay
in place. Project does not repeat the Home figures: neither explains licensing,
release status or project responsibility better than text.

The shared stylesheet is opt-in under `.pv-overview`. Light uses the approved
Home ink/accent/rule family; Carbon and cold Auto inherit existing theme tokens.
Only `docs/project/index.md`, `docs/stylesheets/overview.css` and one stylesheet
registration in `mkdocs.yml` affect production.

## Editorial inventory and destinations

This map covers every original section, including unique qualifications.
Linked destinations already exist; no technical content was migrated into new pages.

| Original content | Treatment and authoritative destination |
| --- | --- |
| Opening metadata routes | Release/status pair and information groups retain Roadmap, release notes, Changelog, License and AI policy. Native tabs retain Guide and Examples routes. |
| Summary: 2D/3D standard/power, periodic geometry, diagnostics, inverse | Condensed into the introduction and six concrete capability descriptions. [Forward guide](../../docs/guide/forward/index.md), [inverse guide](../../docs/guide/inverse/index.md) and [API Reference](../../docs/reference/index.md) retain callable details. |
| Voro++ identity, vendored 3D/legacy 2D, separate pybind11 extensions | Retain Voro++ attribution; omit native-extension implementation from Project. Details remain in [Architecture](../../docs/development/architecture.md#native-backend-and-build-layer). |
| Upstream radical-pruning fix | Omit the implementation detail from Project; its original fix history remains discoverable through the status card’s Changelog link and in [Changelog](../../CHANGELOG.md#063---2026-03-19). No new numerical assertion or rendering. |
| Explicit namespaces and domain differences | Retain dimensional capability warning; [capabilities](../../docs/guide/capabilities.md), [3D domains](../../docs/guide/domains.md), [planar guide](../../docs/guide/planar.md) and [API choice](../../docs/guide/choosing-api.md) contain exact namespaces/domain support. |
| Validation, duplicate safety, normalization, properties, locate/ghost, visualization | Explain geometry, shared topology, neighbor images and checks in plain language without promising automatic normalization; [operations](../../docs/guide/operations.md), [forward output](../../docs/guide/forward/index.md#geometric-output-and-normalization), [periodic contacts](../../docs/guide/periodic-contacts.md), [visualization](../../docs/guide/visualization.md) and [planar guide](../../docs/guide/planar.md) retain workflows. |
| Inverse confidence, robust losses, restrictions, compatibility, identifiability, feasibility, reports and active-set paths | Retain fixed-site inverse role and fit-versus-realization distinction, with direct Guide and Theory links. [Separator fitting and diagnostics](../../docs/guide/inverse/separator-fitting.md) and [advanced reference](../../docs/reference/inverse/separator.md) retain details and provisional/experimental lifecycle. |
| Weights/radii, gauge, component offsets, realized boundaries | Keep explicit Theory route. [Power diagrams](../../docs/theory/power-diagrams.md) and [separator inverse](../../docs/theory/separator-inverse.md) are authoritative. |
| Forward independently useful; new inverse families unapproved | Retain independent forward role. Future architecture/scope remains undecided; [Roadmap](../../docs/project/roadmap.md) and linked ADR 0027 explicitly govern unapproved measures/mixed research. |
| Domain-agnostic boundary, chemvoro atomic models and other applications | Intentionally omit the chemvoro example and downstream-boundary sentence from Project; no substitute application is named. The technical boundary remains in [Architecture's downstream contract](../../docs/development/architecture.md#downstream-contract-for-chemvoro); omit illustrative application list. |
| Stateless calls and possible persistent index | Omit the implementation explanation from Project; it remains in [operations](../../docs/guide/operations.md). Persistent containers remain demand-driven candidates in [Roadmap](../../docs/project/roadmap.md#demand-driven-engineering-candidates). |
| Test layers and local commands | Replace command block/list with contributor setup and Development routes. [CONTRIBUTING](../../CONTRIBUTING.md) covers unit/fuzz/optional wrapper, notebook, strict docs and distribution checks; setup uses the current development workflow. |
| Independent README/site; synchronized notebook exports | Contributor route leads to [documentation conventions](../../docs/development/documentation-conventions.md#readme-and-site-landing-page) and [notebook maintenance](../../docs/development/notebook-workflow.md). These are contributor responsibilities, not project identity. |
| Design/roadmap link list | Roadmap and API lifecycle retain direct links. Development and Decision records route to Architecture, workflow, archived v0.7 and other history without duplicating the Development index. |
| Licensing and provenance | Add a concise explicit distinction: LGPLv3+ project code from v0.6.0, historical MIT, separate upstream Voro++ license. License page retains texts/notices. AI policy summarizes maintainer responsibility without treating this draft as accepted work. |

## Reusable vocabulary

| Pattern | Usage and limits |
| --- | --- |
| `.pv-overview` and `.pv-overview-lead` | Native article typography, scoped colors, rules and visible link focus. Not a site-wide layout. |
| `.pv-overview-grid` + `.pv-overview-card` | Two-column image-first catalogue or fully finished text-first cards. Project uses status cards; omitted images leave no empty slots. Natural image aspect ratio; no crop. |
| `.pv-overview-row`, optional `--reverse` | Text and figure alongside each other when the article is wide enough; alternating visual placement. Text then figure in source and mobile reading order. |
| `.pv-overview-links` | A small, single-level supporting link group. Project uses this for license, contribution and policy. |

The [specimen source](specimen.md) exercises a two-thumbnail catalogue, a
no-image card, links and both row directions. It is **review only**, outside
`docs/` and production navigation. Its repeated figures are deliberate layout
exercises, not an eight-notebook gallery or proposed Theory content.
Both figures are the exact approved Home examples, with native light/dark
variants, dimensions, alt text and captions. [Scientific provenance](../../tools/landing/README.md)
remains unchanged; no dataset or image was generated.

These patterns can support the existing Examples catalogue, Forward/Inverse
Guide routes and two Theory topics after PR A approval. Their content, image
selection and composition remain PR B decisions. Do not make every section a
card or pretend the Home figures depict other notebooks.

## Browser captures

Actual full-page native Zensical builds. Project captures reflect the editorial
correction; the approved specimen captures are unchanged. Click for original PNGs.

| Width | Light | Carbon |
| --- | --- | --- |
| 1366 | [Full Project](screenshots/light-1366.png) | [Full Project](screenshots/carbon-1366.png) |
| 390 | [Full Project](screenshots/light-390.png) | [Full Project](screenshots/carbon-390.png) |
| 320 | [Full Project](screenshots/light-320.png) | [Full Project](screenshots/carbon-320.png) |
| Specimen, 1366 | [Full specimen](screenshots/light-specimen.png) | [Full specimen](screenshots/carbon-specimen.png) |

## Reproduction

Use the repository docs/development environment. Browser tools are disposable
review dependencies, not project/build dependencies: Playwright 1.65.1,
`@axe-core/playwright` 4.14.0 and `pngjs` 7.0.0. Use the same docs stack
for base and candidate. Serve built directories under a local HTTP server.

```bash
zensical build --clean --strict
python review/w5-a/build_specimen.py /absolute/fresh/work/specimen
python review/w5-a/check_site.py site /absolute/base/site
NODE_PATH=/absolute/scratch/node_modules CHROMIUM_PATH=/usr/bin/chromium \
  node review/w5-a/verify.cjs CANDIDATE_URL BASELINE_URL SPECIMEN_URL
NODE_PATH=/absolute/scratch/node_modules CHROMIUM_PATH=/usr/bin/chromium \
  node review/w5-a/check_contrast.cjs CANDIDATE_URL
flake8 src tests tools benchmarks examples
python tools/export_notebooks.py --check
pytest -q tests/tooling/test_text_generation_tools.py tests/integration/test_documentation_examples.py
git diff --check
```

URLs end in `/`; specimen source builds at `SPECIMEN_URL/specimen/`.
The browser script records all axe violations and exits 2 for accessibility
findings, separately from functional/assertion failures (exit 1). No rules or
nodes are suppressed. Native limitations in [#142](https://github.com/DeloneCommons/pyvoro2/issues/142)
and their Development record remain intact.

## Focused editorial correction

Verified PR #144 HEAD `6e2e2b30810f5cbe6c98ec499d8f2966ac04b84c` before editing.
The only production change since that reviewed version is `docs/project/index.md`:
remove the chemvoro sentence, replace implementation-heavy prose with six ordinary
capability bullets, retain the dimensional and inverse-realization qualifications,
and simplify both status cards. Detailed geometry is explicitly available when
requested. Layout, CSS, lower information groups and specimen are unchanged.

- Strict native build: **pass**, no issues. Flake8, notebook export synchronization
  and `git diff --check`: **pass**. Focused documentation/tooling tests: **6 passed**.
- Static link/anchor/asset check: **93 pages, no failures**; **91 non-Project
  article bodies unchanged** against the reviewed version. Scientific PNGs unchanged.
- [Focused Chromium results](editorial-browser-results.json): all six Project
  captures inspected at 1366/390/320 px in Light/Carbon; no horizontal overflow,
  including 150% and 200% root text size. All 13 article links keyboard reachable.
  Native palette switching, Auto, persistence, instant navigation across sections,
  history, Search and the 320 px / 200% text mobile drawer checks passed.
- Project axe scans: no findings at the six viewports. The existing light Search
  breadcrumb contrast issue remains visible (12 nodes, #142); the harness exits
  **2** for that finding, with **zero functional/layout failures**. The original
  API-table finding below remains outside this correction and was not rerun.

The commands above were rerun except specimen generation, full-site pixel
comparisons and the unchanged-style contrast sweep. For this focused browser run:

```bash
PROJECT_ONLY=1 NODE_PATH=/absolute/scratch/node_modules \
  node review/w5-a/verify.cjs CANDIDATE_URL
```

The six screenshots and `editorial-browser-results.json` are refreshed; original
design evidence below is retained separately. Final pushed-head CI is reported
in PR #144 rather than predicted here.

## Initial design validation and limits (reviewed HEAD `6e2e2b3`)

- Native `zensical build --clean --strict`: pass, no issues, for production,
  baseline and the isolated specimen. Zensical 0.1.2 / zensical-ui 0.0.42;
  Chromium 155.0.8059.79 on Linux.
- Repository flake8 and notebook export synchronization: pass. Focused
  documentation/tooling tests: **6 passed**. `git diff --check`: pass.
- [Static results](static-results.json): **93 pages**, no broken local
  links/anchors/assets; **91 non-Project article bodies unchanged**, including
  Home. All four scientific PNGs remain byte-identical. Configuration comparison
  confirms only the stylesheet registration changed; navigation is identical.
- [Browser results](browser-results.json) record responsive layout, text reflow,
  keyboard focus, native palette, instant navigation, history, Search, mobile
  drawer and the separate specimen. Every Project link is keyboard reachable;
  added focus outlines are 3 px. No new production JavaScript is involved.
- Visual regression: **42 full-page comparisons** across Home, Guide, Examples,
  Theory, API Reference, Development and License (both themes, all three widths).
  Exact pixel differences are recorded. At most 16 pixels with channel delta
  at most 8/255 are permitted for native table-corner raster noise; dimensions
  must match. This tolerance does not apply to links, content or axe findings.
- [Computed contrast](contrast-results.json): body text **11.92:1 Light /
  11.62:1 Carbon**; muted text **6.29:1 / 7.82:1**; lowest new text treatment
  (accent on status-card surface) **6.07:1 / 6.36:1**.
- Native light Search still reports 12 breadcrumb contrast nodes; the native
  API table still reports `scrollable-region-focusable` at 320 px in both themes.
  These are #142 findings, not new overview styles. No WCAG-conformance claim,
  shadow-root patch, accessibility exclusion or upstream-limit suppression.

The browser environment uses the system font fallback because Google Fonts
requests are unavailable; baseline and candidate use the same stack. The
review harness accepts the environment proxy certificate for external MathJax;
no production network/security setting changes. Browser checks use Chromium,
not Firefox/Safari, a physical phone or a human screen-reader session. Local
validation does not replace the PR's required CI gate or maintainer approval.

Local editable setup initially failed because the inherited `CXX=clang++`
pointed to a missing executable. Selecting the installed GCC toolchain with
`CC=gcc CXX=g++` completed the standard editable install; no repository build
configuration or dependency was changed.

## Critical self-review

The approved composition remains text-led, with the status cards providing its
main boxed emphasis. The revised six-item list makes capabilities scannable;
optional geometry, dimensional differences and the fit-versus-realization
qualification avoid overstating behavior. Implementation terminology now belongs
to the linked technical destinations rather than the introduction.

The list adds vertical length, especially at 320 px, but each capability remains
one ordinary bullet rather than another card. Natural figure aspect ratios and
alternating rows retain the limitations noted in the approved specimen; neither
was redesigned here. PR B remains outside this correction.
