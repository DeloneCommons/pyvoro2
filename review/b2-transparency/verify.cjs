/* Review-only Chromium sweep. No production dependency or required CI change.
   node review/b2-transparency/verify.cjs BASE_URL CANDIDATE_URL OUTPUT_DIRECTORY */
const {chromium} = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const [baseline, candidate, out] = process.argv.slice(2);
assert(baseline && candidate && out, 'Supply baseline URL, candidate URL and output directory');
fs.mkdirSync(out, {recursive: true});
(async () => {
  const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium', args: ['--no-sandbox']});
  const report = {browser: browser.version(), views: [], themeSwitches: 0};
  try {
    for (const theme of ['light', 'dark']) {
      for (const width of [1366, 390, 320]) {
        for (const route of ['', 'guide/']) {
          const context = await browser.newContext({viewport: {width, height: 900}, colorScheme: theme, reducedMotion: 'reduce'});
          const page = await context.newPage();
          const errors = [];
          page.on('pageerror', e => errors.push(e.message));
          const pair = [];
          for (const [variant, base] of [['before', baseline], ['after', candidate]]) {
            await page.goto(new URL(route, base).href);
            await page.evaluate(() => document.fonts.ready);
            const img = page.locator('img[src*="triclinic-voronoi"]:visible');
            await img.waitFor();
            await img.evaluate(e => e.decode());
            await img.scrollIntoViewIfNeeded();
            const data = await img.evaluate(e => ({
              src: e.getAttribute('src'), loaded: e.complete && e.naturalWidth === 1600 && e.naturalHeight === 900,
              alt: e.alt, rect: e.getBoundingClientRect().toJSON(),
              caption: e.closest('figure').querySelector('figcaption').innerText,
              scheme: document.body.dataset.mdColorScheme,
              documentWidth: document.documentElement.scrollWidth,
            }));
            assert(data.loaded && data.alt && data.caption);
            assert.equal(data.scheme, theme === 'light' ? 'default' : 'slate');
            assert(data.src.endsWith(theme === 'light' ? '#only-light' : '#only-dark'));
            assert(Math.abs(data.rect.width / data.rect.height - 1600 / 900) < .01);
            assert(data.rect.x >= 0 && data.rect.right <= width && data.documentWidth <= width);
            const name = `${route ? 'guide' : 'home'}-${theme}-${width}-${variant}`;
            await page.screenshot({path: path.join(out, `${name}.png`)});
            await img.locator('..').screenshot({path: path.join(out, `${name}-figure.png`)});
            pair.push(data);
            report.views.push({theme, width, route, variant, ...data});
          }
          assert.equal(new URL(pair[0].src, baseline).pathname, new URL(pair[1].src, candidate).pathname);
          for (const key of ['alt', 'caption', 'rect']) assert.deepEqual(pair[0][key], pair[1][key], key);
          const target = theme === 'light' ? 'slate' : 'default';
          for (let i = 0; i < 3 && await page.locator('body').getAttribute('data-md-color-scheme') !== target; i++) {
            await page.locator('[data-md-component="palette"] label:visible').click();
            await page.waitForTimeout(200);
          }
          await page.waitForFunction(s => document.body.dataset.mdColorScheme === s, target);
          const switched = page.locator('img[src*="triclinic-voronoi"]:visible');
          assert.equal(await switched.count(), 1);
          await switched.evaluate(e => e.decode());
          assert((await switched.getAttribute('src')).endsWith(theme === 'light' ? '#only-dark' : '#only-light'));
          report.themeSwitches++;
          assert.deepEqual(errors, []);
          await context.close();
        }
      }
    }
    fs.writeFileSync(path.join(out, 'browser-results.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(`${report.views.length} before/after views and ${report.themeSwitches} native theme switches passed`);
  } finally {
    await browser.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
