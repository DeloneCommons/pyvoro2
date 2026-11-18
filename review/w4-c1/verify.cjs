// Review-only browser checks. No browser dependencies are part of the site.
// Usage: node review/w4-c1/verify.cjs HOME_URL BASELINE_URL
const { chromium } = require('playwright');
const { AxeBuilder } = require('@axe-core/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const urls = process.argv.slice(2);
if (urls.length !== 2) throw Error('Supply exactly Home and baseline URLs ending in /pyvoro2/');
const [a, baseline] = urls;
const output = path.join(__dirname, 'screenshots');
fs.mkdirSync(output, {recursive: true});
const report = { browser: '', variants: {}, documentation: [] };
const launch = {headless: true};
if (process.env.CHROMIUM_PATH) launch.executablePath = process.env.CHROMIUM_PATH;
launch.args = ['--no-sandbox'];
const noOverflow = p => p.evaluate(() => document.documentElement.scrollWidth <= innerWidth);
async function home(p, url) {
  await p.goto(url);
  await p.locator('.pv-home').waitFor();
  await p.evaluate(() => document.fonts.ready);
}
async function loaded(p, pathname) {
  await p.waitForURL(url => url.pathname === pathname);
  await p.waitForFunction(() => !document.querySelector('.pv-home'));
  // Native bundle delays closing drawer/search by 125ms after navigation.
  // Do not race that existing theme behavior when opening the next control.
  await p.waitForTimeout(150);
}
(async () => {
  const browser = await chromium.launch(launch);
  report.browser = browser.version();
  try {
    for (const [variant, url] of [['a', a]]) {
      const context = await browser.newContext({viewport: {width: 1366, height: 768}, reducedMotion: 'reduce'});
      const p = await context.newPage();
      const result = report.variants[variant] = {viewports: [], navigation: [], axe: [], pageErrors: []};
      p.on('pageerror', e => result.pageErrors.push(e.message));
      for (const [name, width, height] of [['desktop',1366,768], ['mobile',390,844], ['narrow',320,844]]) {
        await p.setViewportSize({width, height});
        await home(p, url);
        assert(await noOverflow(p), `${variant} overflow at ${width}`);
        const images = await p.locator('.landing img').evaluateAll(es => es.map(e => ({
          loaded: e.complete && e.naturalWidth > 0,
          ratio: e.getBoundingClientRect().width / e.getBoundingClientRect().height,
          naturalRatio: e.naturalWidth / e.naturalHeight
        })));
        assert(images.every(i => i.loaded && Math.abs(i.ratio - i.naturalRatio) < .01));
        await p.screenshot({path: path.join(output, `${variant}-${name}.png`)});
        await p.screenshot({path: path.join(output, `${variant}-${name}-full.png`), fullPage: true});
        result.viewports.push({width, height, overflow: false, images});
        console.log(`${variant}: captured ${width}px`);
        const axe = await new AxeBuilder({page:p}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
        result.axe.push({width, violations: axe.violations.map(v => ({id:v.id, targets:v.nodes.map(n=>n.target)}))});
        assert.equal(axe.violations.length, 0, JSON.stringify(result.axe));
      }
      await p.setViewportSize({width:1366,height:768});
      await home(p,url);
      await p.evaluate(() => window.c1NavigationSentinel = 'same-document');
      await p.getByRole('link', {name:'Documentation', exact:true}).click();
      await loaded(p, '/pyvoro2/guide/');
      assert.equal(await p.evaluate(() => window.c1NavigationSentinel), 'same-document');
      await p.locator('.md-tabs').getByRole('link', {name:'Home',exact:true}).click();
      await p.locator('.pv-home').waitFor();
      await p.locator('.landing-explore').getByRole('link', {name:'API Reference',exact:true}).click();
      await loaded(p,'/pyvoro2/reference/');
      await p.goBack();
      await p.locator('.pv-home').waitFor();
      await p.goForward();
      await loaded(p,'/pyvoro2/reference/');
      await p.locator('.md-tabs').getByRole('link', {name:'Home',exact:true}).click();
      await p.locator('.pv-home').waitFor();
      assert.equal(await p.evaluate(() => window.c1NavigationSentinel), 'same-document');
      result.navigation.push('Home → Guide → Home → API → Back → Forward → Home: instant, sentinel retained');
      await p.getByRole('button', {name:'Search',exact:true}).click();
      await p.getByPlaceholder('Search',{exact:true}).fill('periodic');
      const searchResult = p.locator('[role="option"][href*="/guide/periodic-contacts/"]').first();
      await searchResult.waitFor();
      await p.screenshot({path:path.join(output,`${variant}-search.png`)});
      await searchResult.click();
      await loaded(p,'/pyvoro2/guide/periodic-contacts/');
      assert.equal(await p.evaluate(() => window.c1NavigationSentinel), 'same-document');
      result.navigation.push('Native search: periodic → Periodic contacts and image shifts, instant');
      await home(p,`${url}#installation`);
      const top = await p.locator('#installation').evaluate(e=>e.getBoundingClientRect().top);
      assert(top >= 0 && top < 768, `installation offscreen: ${top}`);
      result.navigation.push('Direct #installation lands in viewport');
      await home(p,url);
      await p.keyboard.press('Tab');
      assert(await p.locator('.md-skip').evaluate(e=>e===document.activeElement));
      await p.keyboard.press('Enter');
      assert.equal(await p.evaluate(()=>document.activeElement.id),'__skip');
      await p.keyboard.press('Tab');
      const focus = await p.evaluate(()=>({tag:document.activeElement.tagName,outline:getComputedStyle(document.activeElement).outlineWidth}));
      assert.equal(focus.tag,'A'); assert.equal(focus.outline,'3px');
      await p.screenshot({path:path.join(output,`${variant}-focus.png`)});
      result.navigation.push('Keyboard skip → main → link; 3px visible focus');
      await home(p,url);
      await p.evaluate(()=>document.documentElement.style.fontSize='40px');
      assert(await noOverflow(p), '200% root text overflow');
      assert(await p.evaluate(() => document.querySelector('.pv-home__links').getBoundingClientRect().right - parseFloat(getComputedStyle(document.querySelector('.pv-home__links')).paddingRight) < document.querySelector('.md-search__button').getBoundingClientRect().left), '200% header overlap');
      await p.screenshot({path:path.join(output,`${variant}-text-200.png`), fullPage:true});
      result.navigation.push('200% root text size at desktop: no page overflow');
      await p.setViewportSize({width:390,height:844});
      await home(p,url);
      await p.getByRole('button',{name:'Search',exact:true}).click();
      await p.getByPlaceholder('Search',{exact:true}).fill('periodic');
      await searchResult.waitFor();
      await p.keyboard.press('Escape');
      await p.getByRole('link',{name:'Documentation',exact:true}).click();
      await loaded(p,'/pyvoro2/guide/');
      await p.locator('.md-header label[for="__drawer"]').click();
      // The native drawer opens in the active section; go up to its root.
      await p.locator('label.md-nav__title[for="__nav_2"]').click();
      await p.locator('.md-sidebar--primary').getByRole('link',{name:'Home',exact:true}).click();
      await p.locator('.pv-home').waitFor();
      assert(await noOverflow(p));
      result.navigation.push('390px: native search; Home → Guide → drawer Home');
      assert.equal(result.pageErrors.length,0,JSON.stringify(result.pageErrors));
      await context.close();
    }
    // Same browser, viewport and font availability; compare actual pixels.
    for (const route of ['guide/','guide/forward/','theory/','reference/']) {
      for (const width of [1366,390]) {
        const c = await browser.newContext({viewport:{width,height:768},reducedMotion:'reduce'});
        const p = await c.newPage();
        const shots = [];
        for (const url of [baseline,a]) {
          await p.goto(new URL(route,url).href);
          await p.evaluate(()=>document.fonts.ready);
          shots.push(await p.screenshot());
        }
        const identical = shots.slice(1).every(s=>s.equals(shots[0]));
        report.documentation.push({route,width,identical});
        assert(identical, `Documentation rendering changed: ${route} ${width}`);
        await c.close();
      }
    }
  } finally {
    fs.writeFileSync(path.join(__dirname,'browser-results.json'),JSON.stringify(report,null,2)+'\n');
    await browser.close();
  }
  console.log(JSON.stringify(report,null,2));
})().catch(e=>{console.error(e);process.exitCode=1});
