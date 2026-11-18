# DOC-W4-C1 — two landing-page directions

Review checkpoint for [#138](https://github.com/DeloneCommons/pyvoro2/issues/138)
and [#127](https://github.com/DeloneCommons/pyvoro2/issues/127). Neither direction
is accepted. This draft must remain unmerged until the maintainer chooses a
direction and requests finalization. C2 themes and C3 animation are deferred.

Base: live `dev` verified as `f5a65347ae3d4ae90fcf891dce4495030ed81f7c`.
The checkout initially had older refs; the explicit `refs/heads/dev` fetch and
fast-forward were completed before editing.

## Compare the actual builds

All captures use Chromium, the native Zensical build, the same Markdown,
unchanged PNGs, and the same `/pyvoro2/` route on separate local servers.
The ordinary build temporarily shows A for review; this is not a design decision.

| A — Scientific editorial | B — Geometric showcase |
| --- | --- |
| [![A desktop](screenshots/a-desktop.png)](screenshots/a-desktop-full.png) | [![B desktop](screenshots/b-desktop.png)](screenshots/b-desktop-full.png) |
| [Full desktop](screenshots/a-desktop-full.png) | [Full desktop](screenshots/b-desktop-full.png) |
| [![A mobile](screenshots/a-mobile.png)](screenshots/a-mobile-full.png) | [![B mobile](screenshots/b-mobile.png)](screenshots/b-mobile-full.png) |
| [Full 390 px](screenshots/a-mobile-full.png) · [320 px](screenshots/a-narrow.png) · [Full 320 px](screenshots/a-narrow-full.png) | [Full 390 px](screenshots/b-mobile-full.png) · [320 px](screenshots/b-narrow.png) · [Full 320 px](screenshots/b-narrow-full.png) |

Desktop: **1366 × 768**. Mobile/narrow: **390/320 × 844**. Full-page versions
include both workflows, the inverse figure, Quick start, navigation and footer.

## Design assessment

| Criterion | A — Scientific editorial | B — Geometric showcase |
| --- | --- | --- |
| Opening composition | Serif identity beside the scientific plate; balanced text and image. | Broad sans-serif identity above a large, centered scientific plate. |
| Geometry | Integrated with reading; subtle figure rule and caption. | The geometry dominates; the larger plate makes periodic boundaries and complete cells easier to inspect. |
| Workflow rhythm | Compute and Fit are equal adjacent entrances, visible at the bottom of the first desktop screen. | Sequential horizontal rows pair each entrance with its detail or figure, giving the inverse figure more space. |
| Navigation | Compact wordmark and a quiet documentation route; native Search remains visible. | Serif wordmark, a wider header, the same destinations and native Search. |
| Mobile | More compact headline and a shorter opening; familiar reading rhythm. | Larger sans-serif headline and slightly wider hero; stronger identity, but more scrolling before workflows. |
| Cost | The hero has less visual impact than B; some readers may still see an editorial documentation page. | The first desktop screen does not expose the workflow entrances. The long hero and sequential rows delay Quick start. |

**Recommendation: develop A further.** It gives the geometry a clear identity
while making the two scientific workflows immediately discoverable. Its type
and rules fit the project's calm, precise tone. B is a credible alternative if
inspectable geometry should be the dominant first impression; its larger stage
would accommodate a future interactive hero particularly well. A could also
accommodate one inside its existing figure area. No viewer infrastructure has
been added. The choice remains with the maintainer.

Both concepts remove the blue Home-header mismatch while keeping the normal
blue header as the familiar documentation interface. The same project name,
search system, typography for body text, and destinations connect the two.

## Template and CSS architecture

The implementation follows the official [customization](https://zensical.org/docs/customization/)
and [front matter](https://zensical.org/docs/authoring/frontmatter/) interfaces.

- `theme.custom_dir: overrides` and `template: landing.html` select the inherited
  template for Home alone. `main.html` and `base.html` are not copied or replaced.
- The installed `zensical-ui` templates were read before choosing blocks.
  `main.html` extends `base.html`; `hero`, `content`, and `footer` are inside
  `data-md-component="container"`. The native header is outside it.
- The 25-line override adds navigation in `hero`, emits `page.content` in
  `content`, and omits the theme footer because the approved Markdown already
  contains the project/license footer. There is no duplicated scientific prose.
- The native instant-navigation implementation swaps the container and header
  topic, but does not replace the full header. Therefore Home keeps the native
  header/search mounted. CSS hides its ordinary navigation controls and presents
  its real search button alongside the Home navigation. A Home marker in the
  swapped container drives the styling through `:has()`; no persistent body
  class, DOM-patching script, or reload workaround is needed.
- The existing Home-only CSS was moved out of `extra.css` into `landing.css`.
  The documentation's three sticky-header rules are unchanged. Both alternatives
  share tokens and content; `extra.landing_direction: showcase` enables B only
  in a temporary review configuration. No duplicate public Home URLs are built.
- `build_previews.py` writes temporary configuration in the project root (a
  Zensical requirement), invokes the normal build twice, and copies each result
  to a fresh review directory. It does not compile Markdown or alter source.

The native header's DOM classes are an integration dependency. Re-run these
browser checks when upgrading Zensical/UI. This exploration was tested with
Zensical **0.1.2**, zensical-ui **0.0.42**, Python **3.12**, and Chromium
**155.0.8059.79**. CSS `:has()` requires a current browser. Firefox and Safari
were not tested. The cloud could not fetch Google Fonts; Home deliberately uses
system Arial/Georgia, and documentation comparisons used identical font
availability on baseline and candidates.

Home surface, text, accent, rule and code colors use scoped tokens. C2 can map
these tokens to the native theme without changing content or layout. The current
light PNG presentation remains intentional; neither a theme toggle nor a dark
hero has been implemented.

## Reproduce

Use the supported editable environment and documentation requirements:

```sh
python -m pip install -e '.[dev]' -r docs/requirements.txt
zensical build --clean --strict
python tools/landing/build_previews.py --output /tmp/pyvoro2-c1-review
```

The output must be fresh: existing variant destinations are rejected to prevent
stale pages/assets surviving a rebuild. Serve each variant in a separate terminal:

```sh
python -m http.server 8005 --directory /tmp/pyvoro2-c1-review/a
python -m http.server 8006 --directory /tmp/pyvoro2-c1-review/b
```

Open `http://localhost:8005/pyvoro2/` and
`http://localhost:8006/pyvoro2/`. For browser verification, build the unchanged
base separately and serve it on port 8003 under `/pyvoro2/`. Install Playwright
and `@axe-core/playwright` in a disposable review environment, outside the
repository's runtime and docs dependencies. Then:

```sh
CHROMIUM_PATH=/usr/bin/chromium node review/w4-c1/verify.cjs \
  http://localhost:8005/pyvoro2/ \
  http://localhost:8006/pyvoro2/ \
  http://localhost:8003/pyvoro2/
```

`NODE_PATH` may be needed to locate the disposable browser dependencies.
The script refreshes screenshots and `browser-results.json` in this directory.

## Validation evidence

See [browser results](browser-results.json) and
[static results](static-results.json). These are local checks, not GitHub CI.
The PR description records the observed CI state at handoff. All listed final
local checks passed. The browser run reports zero WCAG A/AA axe violations at
1366, 390 and 320 px in both variants, no page exceptions, and no page overflow.
Home/Guide/API transitions and Back/Forward retain a same-document sentinel;
native search returns and follows the periodic-contacts result. Mobile search
and drawer return, `#installation`, skip/focus navigation, and 200% root text
size also pass. Guide overview, forward guide, Theory and API Reference are
pixel-identical to baseline at both 1366 and 390 px (eight comparisons per
candidate). This is representative Chromium evidence, not a complete
screen-reader or cross-browser accessibility audit.

Local commands executed:

- `zensical build --clean --strict` — normal build and both preview builds.
- `python tools/export_notebooks.py --check`.
- `flake8 src tests tools benchmarks examples`.
- `pytest -q tests/integration/test_documentation_examples.py` — four tests.
- `git diff --check`.
- `node review/w4-c1/verify.cjs ...` — browser checks and captures.
- A Python standard-library HTML parser scanned all 93 HTML pages in each build
  for internal link, asset and fragment resolution. No missing targets.
- Source comparison confirms `docs/index.md` differs only by `template:` and
  both scientific PNGs retain their baseline SHA-256 hashes. Non-Home HTML is
  identical to the baseline after removing the added stylesheet link.

The initial editable install failed because inherited `CXX=clang++` pointed to
an unavailable compiler. Repeating with `CC=gcc CXX=g++` succeeded. Early checks
caught Home syntax-color contrast and a B mobile margin-specificity issue; both
were corrected before final captures. The browser test was corrected to follow the native drawer's section hierarchy
and wait beyond the theme's existing 125 ms delayed close on navigation;
clicking sooner races that native behavior. No site workaround was added.

No package code, API, README, notebook sources, navigation inventory,
Markdown extensions, MathJax integration, dependencies, or deployment workflows
were changed. GitHub Pages remains main-only. This work does not authorize
merging, design acceptance, C2, or C3.
