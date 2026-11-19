# DOC-W5 PR B — Examples, User Guide and Theory

Review evidence for #143, under #127. Base `dev` was verified remotely at
`7b5dec0bc24507d29abc578562a38c54e975095d`, the accepted PR A squash merge.
Branch: `docs/doc-w5-b-overviews`. Draft PR targets `dev`; stop before merge
and leave #143 open.

## Three compositions

- **Examples:** eight text-first notebook cards in the original two groups.
  Linked titles open rendered notebooks; numbered source links open the
  original `.ipynb`. Notebook 01 and the inverse quickstart provide distinct
  starting points. Advanced/provisional/experimental qualifications are visible.
- **User Guide:** two alternating editorial paths, with text first in the DOM
  and on narrow screens, followed by four compact task groups. The paths connect
  domain definition, computation, geometry, contacts and validation; and
  separator observations, fitting, residuals, power cells and actual contacts.
- **Theory:** exactly two mathematical topics. A native MathJax power-distance
  definition accompanies Power diagrams; the approved separator-cycle figure
  accompanies inverse fitting. Neither introduction depends on Python names.

The approved Project, Home, Reference and Development source, all child pages,
notebooks/exports, scientific images/generators, navigation, dependencies,
numerical code and CI workflows are unchanged.

## Complete notebook inventory

Titles below identify the rendered page and the unchanged GitHub source route.
Original notebook numbers remain visible within the cards.

| No. | Rendered notebook | Source | Verified content and qualification |
| --- | --- | --- | --- |
| 01 | [Basic 3D computations](../../docs/notebooks/01_basic_compute.md) | [01_basic_compute.ipynb](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/01_basic_compute.ipynb) | Finite/fully periodic triclinic cells, standard/power computation, empty cells, geometry and normalization. Starting point. |
| 02 | [Periodic neighbor graphs](../../docs/notebooks/02_periodic_graph.md) | [02_periodic_graph.ipynb](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/02_periodic_graph.ipynb) | Directed image-labelled contacts and adjacency lists. The retained [Guide recipe](../../docs/guide/periodic-contacts.md#building-a-periodic-graph-in-practice) supplies an undirected key preserving self-images, which the notebook's `i < j` construction omits. |
| 03 | [Locate and probe cells](../../docs/notebooks/03_locate_and_ghost.md) | [03_locate_and_ghost.ipynb](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/03_locate_and_ghost.ipynb) | Owner lookup, hypothetical insertion and out-of-domain query behavior in a finite box. |
| 05 | [Inspect 3D geometry](../../docs/notebooks/05_visualization.md) | [05_visualization.ipynb](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/05_visualization.ipynb) | Optional py3Dmol wireframes, styles, triclinic wrapping and global vertex labels; provisional helpers and compatible notebook frontend. |
| 04 | [Fit separators, then check geometry](../../docs/notebooks/04_powerfit.md) | [04_powerfit.ipynb](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/04_powerfit.ipynb) | External IDs, explicit periodic images, fitted weights and requested-image realization; provisional models and experimental active sets. Stored output includes an observation fitted exactly but realized through another image. |
| 06 | [Fit reports and exports](../../docs/notebooks/06_powerfit_reports.md) | [06_powerfit_reports.ipynb](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/06_powerfit_reports.ipynb) | ID-labelled records, residuals, actual contacts, final/path reports and JSON; provisional helpers, including an experimental active-set run. |
| 07 | [Diagnose incompatible constraints](../../docs/notebooks/07_powerfit_infeasibility.md) | [07_powerfit_infeasibility.ipynb](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/07_powerfit_infeasibility.ipynb) | Contradictory hard restrictions, infeasible status, conflict observations and exported witness; provisional constraint models. |
| 08 | [Follow an active-set path](../../docs/notebooks/08_powerfit_active_path.md) | [08_powerfit_active_path.ipynb](https://github.com/DeloneCommons/pyvoro2/blob/main/notebooks/08_powerfit_active_path.ipynb) | Initially empty active set reconnecting during iteration; transient versus final diagnostics. Explicitly Experimental, without a universal convergence guarantee. |

Descriptions were checked against the source notebook cells, committed Markdown
exports and lifecycle inventory. The catalogue remains 3D-oriented and retains
both planar Guide/plotting links. It explains committed outputs, ordered local
execution and non-executing export, linking contributor procedures separately.
The old `#how-the-published-examples-are-maintained` anchor is retained alongside
the new reading/running heading.

## Old Guide task links → new locations

Every old outgoing URL is retained; see [machine-readable inventory](link-inventory.json).

| Old task | Existing destination(s) retained | New location |
| --- | --- | --- |
| Finite 3D cells | `domains.md#box-finite-non-periodic`; `operations.md#1-compute-tessellate-all-sites` | Domains and computation: Finite 3D boxes / Compute standard or power cells |
| Fully periodic triclinic 3D | `forward/index.md` | Main tessellation path |
| Full/partial orthorhombic periodicity | `domains.md` | Domains and computation: Full or partial 3D periodicity |
| Finite/rectangular-periodic 2D | `planar.md` | Domains and computation |
| Standard/power cells with weights | `operations.md#1-compute-tessellate-all-sites` | Domains and computation |
| Periodic contacts and image shifts | `periodic-contacts.md` | Geometry and contacts |
| Global vertices, edges and 3D faces | `forward/index.md#geometric-output-and-normalization` | Geometry and contacts: Cell results and normalization |
| Plot 2D / inspect 3D | `visualization.md`; `planar.md#planar-plotting` | Validation and visualization |
| Fit fixed separator observations | `inverse/index.md` | Main inverse path and Inverse fitting task group |
| Unsupported domains / numerical limits | `capabilities.md` | Validation and visualization |
| Namespaces, result layers, lifecycle | `choosing-api.md` | Closing API-choice paragraph |
| Longer workflows, mathematics, signatures | `../examples/index.md`; `../theory/index.md`; `../reference/index.md` | Closing supporting routes |

Added direct routes: planar normalization, tessellation diagnostics, detailed
separator fitting and the fit-to-geometry discussion. Dimensional caveats remain
explicit: fully periodic triclinic 3D; orthorhombic domains for partial 3D
periodicity; rectangular planar periodicity; differing 2D/3D output features.

## Theory and scientific illustration checks

1. [Power diagrams](../../docs/theory/power-diagrams.md): sites, scalar weights,
   minimum power distance, equal-weight Voronoi limit, pairwise separator and
   realized boundary. The displayed definition is
   `pi_i(x) = ||x - p_i||^2 - w_i`, set with the configured native MathJax as one
   compact two-line expression. It uses no unweighted picture to imply weighted
   geometry. The complete equation remains one mathematical/copyable object.
2. [Inverse fitting](../../docs/theory/separator-inverse.md): observed positions,
   weight differences, signed cycle compatibility, approximate fitting and
   separate geometric realization. The approved three-site figure's observations
   are inconsistent; ticks are observations and edges are computed boundaries.
   All three pairs meet only in this particular result.

The Guide's B2 image is the exact four-site standard Voronoi computation from
its forward quickstart: complete polyhedra, total volume 3, dashed fundamental
domain rather than clipping boundary. It is not a new weighted example.
Both uses of the separator cycle retain the same accurate interpretation.
`#only-light` and `#only-dark` select the original Light/Carbon asset pairs;
width/height, aspect ratio, descriptive alt text and captions are preserved.
[Existing provenance](../../tools/landing/production-provenance.json) documents
identical geometry between themes. Static checks confirm all four PNG hashes
are unchanged. No illustrations were generated or altered.

## Browser evidence

Full-page captures of the real Zensical build, not a standalone template:

| Page / width | Light | Carbon |
| --- | --- | --- |
| Examples 1366 | [Full page](screenshots/examples-light-1366.png) | [Full page](screenshots/examples-carbon-1366.png) |
| Examples 390 | [Full page](screenshots/examples-light-390.png) | [Full page](screenshots/examples-carbon-390.png) |
| Examples 320 | [Full page](screenshots/examples-light-320.png) | [Full page](screenshots/examples-carbon-320.png) |
| User Guide 1366 | [Full page](screenshots/guide-light-1366.png) | [Full page](screenshots/guide-carbon-1366.png) |
| User Guide 390 | [Full page](screenshots/guide-light-390.png) | [Full page](screenshots/guide-carbon-390.png) |
| User Guide 320 | [Full page](screenshots/guide-light-320.png) | [Full page](screenshots/guide-carbon-320.png) |
| Theory 1366 | [Full page](screenshots/theory-light-1366.png) | [Full page](screenshots/theory-carbon-1366.png) |
| Theory 390 | [Full page](screenshots/theory-light-390.png) | [Full page](screenshots/theory-carbon-390.png) |
| Theory 320 | [Full page](screenshots/theory-light-320.png) | [Full page](screenshots/theory-carbon-320.png) |

## Narrow CSS additions

All changes are opt-in additions to `overview.css`:

- `pv-overview-grid--catalogue`: flex card content aligns source links at the
  bottom of each equal-height desktop card row despite differing description
  lengths. No fixed heights; mobile cards grow naturally.
- `pv-overview-links--tasks`: uses the existing 16rem minimum-column pattern
  for two balanced pairs of Guide groups instead of an orphan fourth group
  below three columns; removes the groups' native heading top margin.
- `pv-overview-equation`: a 0.75rem formula (15px at normal desktop text;
  doubled with 200% root text) and removal of native negative math gutters
  inside its figure. Real browser review exposed clipping with a single-line
  display expression at 320px/200%. Two lines keep the whole definition
  together, readable and inside its row.

No existing selector or global token changed. Project status cards do not
opt into the catalogue modifier; other article/math styles are unchanged.

## Validation

- **PASS:** `zensical build --clean --strict` on the final production tree;
  Zensical 0.1.2 / UI 0.0.42. Baseline was built with the same installed stack.
- **PASS:** local link, anchor and asset validation across 93 generated HTML
  pages; all 89 other article bodies and all four scientific PNG hashes unchanged.
  All old outgoing overview destinations are retained. The eight GitHub notebook
  source paths exist on published `main` and their files match the verified base.
- **PASS:** repository flake8 targets plus the review checker; notebook export
  synchronization; `git diff --check`; focused text-generation/notebook-metadata
  tests (**4 passed**).
- **PASS:** 18 full-page browser views (three pages, Light/Carbon, 1366/390/320px),
  visually inspected. Card alignment, alternating rows, image aspect ratios,
  active variants, alt/caption behavior and native MathJax checked.
- **PASS:** 36 enlarged-text checks at 150% and 200%, including 320px; no document,
  card, row or mathematical-content horizontal overflow. Root-font enlargement
  tests text reflow; it is not a claim about every browser's zoom implementation.
- **PASS:** real Tab traversal of every article link (23 Examples, 20 Guide,
  4 Theory per theme), visible 3px focus outlines and logical DOM reading order.
  Native palette keyboard controls, Auto/Light/Dark persistence and live OS-theme
  changes, Search/results, seven-section instant navigation, history and MathJax
  on entry work. Search and the nested navigation drawer also work at 320px/200%.
- **PASS:** 24 full-page pixel comparisons against PR A are identical: Home,
  Project, Reference, Development, forward Guide and Power diagrams, in both
  themes at 1366px and 320px.
- **KNOWN FINDINGS, UNSUPPRESSED:** 22 axe scans found no violations on the
  18 overview views. Native Light Search has 15 breadcrumb contrast nodes;
  the native API table at 320px has one `scrollable-region-focusable` finding
  per theme. These inherited issues remain tracked in
  [#142](https://github.com/DeloneCommons/pyvoro2/issues/142). The browser script
  deliberately exits **2** for these findings, after functional/layout checks
  pass; it does not suppress them or report a clean accessibility audit.
- GitHub CI on the final PR HEAD is recorded in the PR description after the
  remote run completes. Local documentation checks do not establish CI success.

Raw results: [static checks](static-results.json),
[browser checks and complete axe findings](browser-results.json).
Review tooling and screenshots are evidence only, with no new mandatory CI jobs
or repository dependencies. Browser coverage is Chromium 155 on Linux;
Firefox, Safari, physical devices and human screen-reader use were not tested.
No full WCAG-conformance claim is made.

## Critical editorial and visual review

The catalogue has no illustrative filler: linked titles, restrained numbering,
workflow descriptions and separate source links are sufficient to browse it.
The periodic graph card is denser because its self-image caveat matters.
Guide images accompany connected learning sequences; its task index remains
secondary to the two entry points. Theory uses a mathematical expression where
a reused geometric picture would imply the wrong computation.

The pages share PR A typography, colors and components without sharing a single
composition. Mobile necessarily becomes a longer reading sequence, especially
Examples; there are no fixed card heights or empty image slots. The Theory
formula deliberately occupies two lines even on desktop to retain one native
mathematical object at narrow/enlarged sizes. The figure captions carry essential
qualifications rather than expecting readers to infer scientific meaning.

A fresh reviewer found an old-anchor regression, an overly broad manuscript
pointer and fragmented MathJax accessibility in an intermediate draft. All three
were addressed before final validation. No new design or API decisions arose.

## Deferred

The final v0.9 feature scope, notebook selection/replacement/retirement,
extensive child-Guide rewrites, Reference and Development restructuring, new
inverse methods and DOC-W99 whole-site audit remain outside this change.
No release-readiness or full WCAG-conformance claim is made.

## Reproduction

Use the project's declared documentation/development stack in an isolated venv.
This environment required `CC=gcc CXX=g++` because its inherited `clang++`
setting named an unavailable compiler. No repository toolchain setting changed.

```sh
zensical build --clean --strict
python review/w5-b/check_site.py site /path/to/baseline/site
flake8 src tests tools benchmarks examples review/w5-b/check_site.py
python tools/export_notebooks.py --check
pytest -q tests/tooling/test_text_generation_tools.py tests/tooling/test_notebooks_meta.py
git diff --check
# Serve candidate and an identically built archive of 7b5dec0 on separate ports.
# Install playwright, @axe-core/playwright, pngjs only in a review scratch directory.
NODE_PATH=/path/to/review/node_modules node review/w5-b/verify.cjs \
  http://127.0.0.1:8000/ http://127.0.0.1:8001/
```
