# Shared transparent B2 pair — issue #146

Baseline: `fe2f3b085a321ba88098a607b892693e82eadd1e` (`dev`, verified before work).
Scope: [#146](https://github.com/DeloneCommons/pyvoro2/issues/146), under
[#127](https://github.com/DeloneCommons/pyvoro2/issues/127).

Both canonical 1600×900 B2 PNGs are transparent. Home and Guide image sources,
copy, CSS and layout are unchanged. No extra production images were introduced;
both inverse PNGs retain their approved bytes. Maintainer acceptance is pending.

## Browser comparison

Actual Chromium captures, before on the left and after on the right. Each sheet
contains Home Light, Home Carbon, Guide Light and Guide Carbon. Figure pixels
are unscaled; labels and whitespace only were added to arrange the captures.

- [Desktop, 1366 px](screenshots/desktop.png)
- [Mobile, 390 px](screenshots/mobile.png)
- [Narrow, 320 px](screenshots/narrow.png)

The candidates were first served through request interception over the unchanged
baseline site; production assets were replaced only after all twelve contexts
were reviewed. The final sweep compared separately served baseline and production
builds. All four site markers, complete polyhedra, translucent overlaps, solid
edges and dashed distant lattice remain visible at each size. The Light Guide's
warm-white plate is removed. No bright/dark fringe or depth-order regression was
observed. Colors, material opacity and camera have not been adjusted to compensate
for transparency.

[Browser results](browser-results.json): 24 before/after views, 12 native theme
switches, decoded images, native theme selection, unchanged captions/alt text,
identical geometry on the page, correct aspect ratio and no horizontal overflow.
[Static results](static-results.json): 93 HTML pages, working local links/assets,
92 unchanged article bodies, and unchanged non-B2 assets.

## Rendering evidence

The original offline renderer reproduced both approved opaque PNGs byte-for-byte
before any renderer changes. Independent scientific assertions and every
scientific provenance field match the baseline, including raw geometry SHA-256
`7cc373bbdcfe3a0b28ca17be08f7a0e5c3134a81d4472bfd20326c9b1b6d19af`, four cells,
volume sum 3, 56 oriented faces, 51 drawn polygons, periodic shifts/contact and
orthographic camera. Eight independent production renders matched the four
committed assets and each other's records exactly.

A direct `transparent_background=True` export retained background-contaminated
RGB. Interpreting it as straight alpha caused raw composite errors up to 39.18
Light / 33.11 Carbon RGB levels. Native transparent-black RGBA instead has correct
premultiplied color and alpha: the black/white response agrees with alpha to
floating-point precision. Standard unpremultiplication reproduces the original
opaque full-resolution renders within half a level. No chroma key or inferred
alpha is involved.

Pillow's 8-bit RGBA/Lanczos resampling produced edge errors up to 68 Light / 31
Carbon levels. Floating Lanczos still overshot coverage because of its negative
weights. A positive 2×2 area average of native premultiplied RGBA, followed by
straight-alpha PNG encoding, preserves compositing without these edge artifacts.
Native alpha is only averaged and quantized, never arbitrarily modified.

The optional [alpha checker](../../tools/landing/check_alpha.py) verifies two
analytical overlapping layers in both insertion orders, unused pixels, opaque and
fractional alpha, and the real B2 scene over white and original theme backgrounds.
Maximum final composite error against independently area-filtered opaque captures:

| Theme | White | Original theme background |
|---|---:|---:|
| Light | 0.7834 | 0.8667 |
| Carbon | 0.7177 | 0.8295 |

Values are in 8-bit channel levels (0–255), over every pixel including edges and
overlaps. These measure equivalent filtering, not byte equality with the former
Lanczos-filtered opaque files. Visual comparisons above cover that filter change.
The checker failed on the baseline renderer with `Expected native RGBA`, then
passed with the new implementation. Optimized Python is rejected.

## Reproduce

Use the optional pinned stack in `tools/landing/requirements.txt` and the EGL
backend documented in [rendering instructions](../../tools/landing/README.md).

```bash
python tools/render_landing.py --check-only
python tools/render_landing.py --theme dark --check-only
VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow \
  python tools/landing/verify_assets.py --records work/reproduced
zensical build --clean --strict
python review/b2-transparency/check_site.py site /path/to/baseline-site
node review/b2-transparency/verify.cjs \
  http://127.0.0.1:8001/ http://127.0.0.1:8000/ work/browser
flake8 src tests tools benchmarks examples
pytest -q
git diff --check
```

Playwright and Chromium are review-only tools. Rendering dependencies remain
outside normal docs, runtime and required CI. Final PR-head CI is tracked by the
pull request; local validation is not a claim of maintainer acceptance.
