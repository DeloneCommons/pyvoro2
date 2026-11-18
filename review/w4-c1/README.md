# DOC-W4-C1 — selected Scientific editorial Home

The maintainer selected **Direction A** as the production foundation in
[#139](https://github.com/DeloneCommons/pyvoro2/pull/139), for
[#138](https://github.com/DeloneCommons/pyvoro2/issues/138). This final refinement
is ready for visual review, not accepted or authorized to merge. C2 themes and
C3 interactive geometry remain deferred.

Starting PR HEAD was verified locally and remotely as
`0115d0ed342e92ebf11266deffe76318cca11da0`, after the earlier `4160c5b` review.
The original `dev` baseline remains `f5a65347ae3d4ae90fcf891dce4495030ed81f7c`.

## Selected page

[![Desktop, 1366 × 768](screenshots/a-desktop.png)](screenshots/a-desktop-full.png)

[Full desktop](screenshots/a-desktop-full.png) ·
[390 px](screenshots/a-mobile.png) · [Full 390 px](screenshots/a-mobile-full.png) ·
[320 px](screenshots/a-narrow.png) · [Full 320 px](screenshots/a-narrow-full.png).
Mobile viewports are 844 px tall. Every capture is Chromium rendering the native
Zensical build under `/pyvoro2/`, with approved Markdown as the content source.

## Focused comparisons

| Detail | Before / control | Refined |
| --- | --- | --- |
| Hero and caption | [Previous A](selected/before-hero.png) | [Selected A](selected/after-hero.png) |
| Closing navigation and metadata | [Previous spacing](selected/before-footer.png) | [Tighter spacing](selected/after-footer.png) |
| Inverse figure, desktop | [Opaque](selected/opaque-inverse.png) | [Transparent](selected/after-inverse.png) |
| Inverse figure, 390 px | [Opaque](selected/opaque-inverse-mobile.png) | [Transparent](selected/after-inverse-mobile.png) |

**Headline:** retained an 8 px separation before the italic teal “in Python,”
now consistently on its own line within the same `h1`. The wording and reading
order are unchanged. The desktop heading grows by only 8 px. At 390 px the
extra line is a visible height trade-off, but gives the scientific subject a
clearer hierarchy; at 320 px it already occupied its own line. The serif size,
color and weight stay unchanged rather than introducing another large heading.

**B2 and caption:** retained a modest grid adjustment from approximately 677 to
705 px image width at desktop (4%). All four cells and the lattice frame remain
complete, with the unchanged aspect ratio and source PNG. Removed redundant
paragraph margins around the image and reduced the caption margin to 8 px.
The caption remains centered, 12 px, muted, linked and unruled. Further shrinking
or lightening the caption was not adopted: its existing size and contrast remain
useful for reading the scientific metadata. Both workflow entrances and their
introductions remain visible at 1366 × 768; the opening does not grow overall.

**Footer:** reduced combined Explore/footer padding by 18 px. Links and metadata
now read as one quiet closing region, with their hierarchy preserved. No borders,
labels, icons or content were added.

**Inverse figure:** retained the transparent PNG. It removes the white rectangular
plate without weakening panel boundaries or the pale computed-cell fills. Site
markers, orange observation ticks, dashed connectors, teal boundaries and titles
remain legible at desktop, 390 and 320 px. No visible halos or clipping appeared
in the rendered comparisons. SVG was not pursued: PNG transparency solves the
specific problem with one generator argument and no format/build changes.

The complete page retains A's two adjacent desktop workflows and compact mobile
reading order. It remains a scientific editorial layout; no B composition was
introduced. Final visual acceptance belongs to the maintainer.

## Scientific fidelity

The only generator change is `facecolor='white'` → `transparent=True` in
`tools/landing/render_separator_cycle.py`. Points, observations, fit, polygons,
edges, titles, colors, dimensions and all scientific assertions are unchanged.

The original script regenerated the previous opaque PNG byte for byte in this
environment. Two transparent renders were also byte-identical. All fully opaque
pixels retain exactly their original RGB values. Compositing the transparent PNG
onto white differs by at most 2/255 per channel on about 1.97% of pixels, consistent
with antialias compositing; there is no changed geometry. See
[figure verification](figure-results.json).

The generator's optimal-fit, inconsistent-cycle, total-area, polygon-area and
three-pair-contact assertions all pass. Relevant fit, realization, planar-edge
and documentation-example tests also pass. The B2 PNG is byte-identical to the
approved baseline. `docs/index.md` is byte-identical to `0115d0e`; no scientific
copy, alt text, links or code examples changed.

## Production cleanup and history

Only A remains in production CSS. The template no longer reads a direction
setting or emits a variant class. Native inheritance, content rendering, search
and instant navigation are preserved. `build_preview.py` now invokes the ordinary
strict native build once and copies the result beneath `/pyvoro2/`; temporary
A/B configuration and the old plural helper are removed. Browser checks now take
one Home URL and one baseline URL, retaining navigation/accessibility regressions.

B and the previous experiments remain safely accessible in published history:

- [Both refined directions and evidence at 0115d0e](https://github.com/DeloneCommons/pyvoro2/tree/0115d0ed342e92ebf11266deffe76318cca11da0/review/w4-c1).
- [B full desktop](https://github.com/DeloneCommons/pyvoro2/blob/0115d0ed342e92ebf11266deffe76318cca11da0/review/w4-c1/screenshots/b-desktop-full.png),
  [390 px](https://github.com/DeloneCommons/pyvoro2/blob/0115d0ed342e92ebf11266deffe76318cca11da0/review/w4-c1/screenshots/b-mobile-full.png),
  [320 px](https://github.com/DeloneCommons/pyvoro2/blob/0115d0ed342e92ebf11266deffe76318cca11da0/review/w4-c1/screenshots/b-narrow-full.png).
- [Previous A full desktop](https://github.com/DeloneCommons/pyvoro2/blob/0115d0ed342e92ebf11266deffe76318cca11da0/review/w4-c1/screenshots/a-desktop-full.png)
  and [390 px](https://github.com/DeloneCommons/pyvoro2/blob/0115d0ed342e92ebf11266deffe76318cca11da0/review/w4-c1/screenshots/a-mobile-full.png).
- [First exploration at 4160c5b](https://github.com/DeloneCommons/pyvoro2/tree/4160c5bd11e15398b3c2c2ba4d7e977569ee4cb5/review/w4-c1).

Historical B screenshots and superseded typography/width experiments are removed
from the current tree, not from Git history. No production A/B cleanup remains.
Review screenshots and checks remain outside the published documentation.

## Template architecture and limits

`theme.custom_dir: overrides` and Home's `template: landing.html` use native
MiniJinja inheritance. The small template extends `main.html`, adds Home navigation
in `hero`, renders `page.content`, and suppresses the theme footer because the
approved Markdown contains its own footer. Standard theme files are not copied.

Instant navigation replaces the container, not the full native header. Home keeps
native search mounted; a `.pv-home` marker inside the replaced container scopes
CSS through `:has()`. No site JavaScript, parallel search or forced reload is
needed. The standard documentation header returns automatically off Home.
Recheck this integration when upgrading Zensical, since it uses native DOM classes.

Scoped color tokens remain suitable for C2, but transparency alone does not make
the inverse figure dark-theme-ready. No palette switching, dark B2, backgrounds,
side patterns or animation are included. Tested with Zensical 0.1.2, zensical-ui
0.0.42, Python 3.12 and Chromium 155; Firefox/Safari and a full screen-reader audit
were not run. Home uses system Arial/Georgia. Google Fonts were unavailable in
this environment; documentation comparisons use the same font availability.

## Reproduce and validate

With the editable development and docs environment installed:

```sh
zensical build --clean --strict
python tools/landing/build_preview.py --output /tmp/pyvoro2-c1-selected
python -m http.server 8023 --directory /tmp/pyvoro2-c1-selected
```

The helper requires a fresh destination. Open `http://localhost:8023/pyvoro2/`.
Serve a separate original-baseline build on port 8022 beneath the same subpath.
With disposable Playwright and `@axe-core/playwright` dependencies available:

```sh
CHROMIUM_PATH=/usr/bin/chromium node review/w4-c1/verify.cjs \
  http://localhost:8023/pyvoro2/ http://localhost:8022/pyvoro2/
```

`NODE_PATH` may be needed for the external browser dependencies. They are not
site runtime or repository dependencies. Regenerating the inverse PNG uses the
existing optional Matplotlib/visualization environment:

```sh
python tools/landing/render_separator_cycle.py
```

Local checks executed:

- `zensical build --clean --strict` through the preview helper: no issues.
- `python tools/export_notebooks.py --check` and
  `flake8 src tests tools benchmarks examples`: passed.
- `pytest -q tests/integration/test_documentation_examples.py tests/inverse/separator/test_fit.py tests/inverse/separator/test_realization.py tests/forward/planar/test_edge_properties.py`:
  30 passed.
- Generator assertions, repeat-render `cmp`, and opaque/transparent pixel checks:
  passed, as described above.
- Internal links/assets/fragments across 93 HTML pages: no missing targets.
- Direct Home, Guide/API round trips, Back/Forward, native search result navigation,
  mobile drawer return, `#installation`, keyboard skip/focus, 200% text and image
  aspect ratios: passed. Instant-navigation sentinel retained.
- Axe WCAG A/AA: zero violations at 1366, 390 and 320 px; no overflow or page errors.
- Guide overview, forward guide, Theory and API Reference: pixel-identical to the
  original `dev` baseline at 1366 and 390 px. Non-Home HTML is identical after
  removing the stylesheet link added in the first exploration.
- `git diff --check`: passed.

[Browser results](browser-results.json) · [Static results](static-results.json).
The first browser run could not reach an old baseline server; the final run uses
a fresh server. GitHub CI is separate from local checks; its final status and run
link are recorded in the PR description after pushing. No public API, numerical
algorithm, dependency declaration, standalone README, documentation navigation or
Pages workflow changed. The PR remains draft and unmerged; issue #138 stays open.
