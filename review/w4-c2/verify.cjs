// Public MathJax CDN requires ignoreHTTPSErrors under this environment proxy.
// Review-only browser evidence; no dependencies or JavaScript are shipped by the site.
// NODE_PATH=<disposable node_modules> CHROMIUM_PATH=/usr/bin/chromium node review/w4-c2/verify.cjs PRODUCTION_URL BASELINE_URL
const {chromium} = require('playwright');
const {AxeBuilder} = require('@axe-core/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const [a,baseline] = process.argv.slice(2);
if (!a || !baseline) throw Error('Supply production and C1 baseline URLs ending in /pyvoro2/');
const out = path.join(__dirname,'screenshots');
fs.mkdirSync(out,{recursive:true});
const report = {browser:'',variants:{},lightBaseline:[],failures:[],accessibilityFailures:[]};
const overflow = p => p.evaluate(()=>document.documentElement.scrollWidth > innerWidth);
const scheme = p => p.locator('body').getAttribute('data-md-color-scheme');
async function expectScheme(p,s) {
 await p.waitForFunction(s=>document.body.dataset.mdColorScheme===s,s);
 if(await p.locator('.landing').count()) {
  const figures=p.locator('.landing figure');
  assert.equal(await figures.count(),2);
  for(const figure of await figures.all()) {
   assert.equal(await figure.getByRole('img').count(),1);
   const img=figure.locator('img:visible');
   assert((await img.getAttribute('src')).endsWith(s==='slate'?'#only-dark':'#only-light'));
   assert((await img.getAttribute('alt')).length>30);
   assert((await figure.locator('figcaption').innerText()).length>30);
  }
 }
}
async function ready(p) {
  await p.locator('.md-content').waitFor();
  await p.evaluate(()=>window.MathJax?.startup?.promise);
  await p.evaluate(()=>document.fonts.ready);
  await p.waitForTimeout(200); // native search/drawer cleanup after instant routing (125ms)
}
async function go(p,url) { await p.goto(url); await ready(p); }
async function shot(p,name,fullPage=false) { await p.screenshot({path:path.join(out,name+'.png'),fullPage}); }
async function axe(p,r,where) {
  const a=await new AxeBuilder({page:p}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
  const violations=a.violations.map(v=>({id:v.id,impact:v.impact,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))}));
  r.axe.push({where,violations});
  if(violations.length) report.accessibilityFailures.push(`${r.name} ${where}: axe ${violations.map(v=>v.id)}`);
}
async function palette(p,title) {
 const before=await p.locator('.landing figure').evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();return [r.x,r.y,r.width,r.height]}));
 await p.locator(`[data-md-component="palette"] label[title="${title}"]`).click();
 const after=await p.locator('.landing figure').evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();return [r.x,r.y,r.width,r.height]}));
 assert.deepEqual(after,before,'Theme change must not move/resize scientific figures');
}
async function route(p,url) { await p.waitForURL(u=>u.pathname===new URL(url).pathname); await ready(p); }
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined,args:['--no-sandbox']});
 report.browser=browser.version();
 try {
  for(const [name,url,mode] of [['light',a,'light'],['carbon',a,'dark']]) {
   const r=report.variants[name]={name,axe:[],viewports:[],functional:[],pageErrors:[],paint:[]};
   const c=await browser.newContext({ignoreHTTPSErrors:true,viewport:{width:1366,height:768},colorScheme:mode,reducedMotion:'reduce'});
   const p=await c.newPage();p.on('pageerror',e=>r.pageErrors.push(e.message));
   await p.addInitScript(()=>{
    window.themeFrames=[];let n=0;
    const sample=()=>{if(document.querySelector('.landing')&&document.querySelector('link[rel=stylesheet]')?.sheet){window.themeFrames.push({scheme:document.body.dataset.mdColorScheme,bg:getComputedStyle(document.body).backgroundColor});n++} if(n<30)requestAnimationFrame(sample)};requestAnimationFrame(sample);
   });
   for(const [size,width,height] of [['desktop',1366,768],['mobile',390,844],['narrow',320,844]]) {
    await p.setViewportSize({width,height});await go(p,url);await expectScheme(p,mode==='dark'?'slate':'default');
    assert.equal(await overflow(p),false,`${name} ${width}: overflow`);
    const images=await p.locator('.landing img:visible').evaluateAll(es=>es.map(e=>({src:e.getAttribute('src'),loaded:e.complete&&e.naturalWidth>0,ratio:e.width/e.height,naturalRatio:e.naturalWidth/e.naturalHeight})));
    assert.equal(images.length,2);assert(images.every(i=>i.src.endsWith(mode==='dark'?'#only-dark':'#only-light')));assert.equal(await p.locator('.landing').getByRole('img').count(),2);assert(images.every(i=>i.loaded&&Math.abs(i.ratio-i.naturalRatio)<.02));
    const bounds=await p.locator('[data-md-component="palette"]').boundingBox();assert(bounds.width>=36&&bounds.x>=0&&bounds.x+bounds.width<=width);
    await shot(p,`${name}-${size}`);await shot(p,`${name}-${size}-full`,true);
    await axe(p,r,`Home ${width}`);r.viewports.push({width,height,overflow:false,images});
    const frames=await p.evaluate(()=>window.themeFrames);
    assert(frames.length>0);assert(frames.every(f=>f.bg===(mode==='dark'?'rgb(36, 39, 40)':'rgb(250, 250, 247)')));
    r.paint.push({width,sampledFrames:frames.length,backgrounds:[...new Set(frames.map(f=>f.bg))]});
   }
   await p.setViewportSize({width:1366,height:768});
   for(const [label,routeName] of [['guide','guide/forward/'],['api','reference/api/'],['theory','theory/power-diagrams/'],['examples','examples/'],['admonition','development/architecture/']]) {
    await go(p,new URL(routeName,url).href);assert.equal(await overflow(p),false);
    if(label==='theory')assert(await p.locator('mjx-container').count()>0,'Theory equations must render');
    await shot(p,`${name}-${label}`);await axe(p,r,label);
    if(label==='api') {const table=p.locator('.md-typeset table').first();await table.scrollIntoViewIfNeeded();await shot(p,`${name}-api-table`);}
   }
   for(const [label,routeName] of [['guide','guide/forward/'],['api','reference/api/'],['theory','theory/power-diagrams/']]) for(const width of [390,320]) {
    await p.setViewportSize({width,height:844});await go(p,new URL(routeName,url).href);assert.equal(await overflow(p),false);
    await shot(p,`${name}-${label}-${width}`);await axe(p,r,`${label} ${width}`);
   }
   await p.setViewportSize({width:1366,height:768});
   await go(p,url);await p.getByRole('button',{name:'Search',exact:true}).click();await p.getByPlaceholder('Search',{exact:true}).fill('periodic');
   const hit=p.locator('[role="option"][href*="/guide/periodic-contacts/"]').first();await hit.waitFor();await shot(p,`${name}-search`);await axe(p,r,'search');
   await hit.click();await route(p,new URL('guide/periodic-contacts/',url).href);r.functional.push('Native search result navigation');
   await p.setViewportSize({width:390,height:844});await go(p,new URL('guide/',url).href);await p.locator('.md-header label[for="__drawer"]').click();await shot(p,`${name}-drawer`);await axe(p,r,'mobile drawer');
   await p.locator('label.md-nav__title[for="__nav_2"]').click();await p.locator('.md-sidebar--primary').getByRole('link',{name:'Home',exact:true}).click();await p.locator('.pv-home').waitFor();r.functional.push('Mobile drawer Guide → Home');
   // Cycle manual choices and verify persisted overrides across OS changes.
   await palette(p,'Switch to light mode');await expectScheme(p,'default');await p.emulateMedia({colorScheme:'dark'});await p.reload();await ready(p);await expectScheme(p,'default');
   await palette(p,'Switch to dark mode');await expectScheme(p,'slate');await p.emulateMedia({colorScheme:'light'});await p.reload();await ready(p);await expectScheme(p,'slate');
   r.functional.push('Manual light/dark survives reload and opposite OS preference');
   await p.setViewportSize({width:1366,height:768});
   await p.evaluate(()=>window.c2Sentinel='instant');
   await p.getByRole('link',{name:'Documentation',exact:true}).click();await route(p,new URL('guide/',url).href);await expectScheme(p,'slate');
   await p.locator('.md-tabs').getByRole('link',{name:'API Reference',exact:true}).click();await route(p,new URL('reference/',url).href);await expectScheme(p,'slate');
   await p.locator('.md-tabs').getByRole('link',{name:'Home',exact:true}).click();await p.locator('.pv-home').waitFor();await expectScheme(p,'slate');
   await p.goBack();await route(p,new URL('reference/',url).href);await expectScheme(p,'slate');await p.goForward();await p.locator('.pv-home').waitFor();await expectScheme(p,'slate');
   assert.equal(await p.evaluate(()=>window.c2Sentinel),'instant');r.functional.push('Home → Guide → API → Home → Back → Forward: persisted slate, instant sentinel retained');
   await palette(p,'Switch to system preference');await expectScheme(p,'default');await p.emulateMedia({colorScheme:'dark'});await expectScheme(p,'slate');await p.reload();await ready(p);await expectScheme(p,'slate');await p.emulateMedia({colorScheme:'light'});await expectScheme(p,'default');r.functional.push('Auto follows live system changes and survives reload');
   // Keyboard reaches native radio. Arrow keys operate the existing radio group.
   await p.emulateMedia({colorScheme:mode});await go(p,url);await p.keyboard.press('Tab');assert(await p.evaluate(()=>document.activeElement.classList.contains('md-skip')));await p.keyboard.press('Tab');
   assert.equal(await p.evaluate(()=>document.activeElement.name),'__palette');
   const focus=await p.evaluate(()=>getComputedStyle(document.activeElement.nextElementSibling).outlineWidth);assert.equal(focus,'3px');await shot(p,`${name}-keyboard`);
   await p.keyboard.press('ArrowRight');await expectScheme(p,'default');await p.keyboard.press('ArrowRight');await expectScheme(p,'slate');r.functional.push('Keyboard Tab → native palette, visible 3px focus, arrow-key selection');
   await go(p,url);await p.evaluate(()=>document.documentElement.style.fontSize='40px');assert.equal(await overflow(p),false);r.functional.push('200% root text at desktop: no horizontal page overflow');
   assert.equal(r.pageErrors.length,0,JSON.stringify(r.pageErrors));await c.close();console.log(name,'done');
  }
  // Approved light sections remain pixel-identical; only the footer gap changes.
  for(const width of [1366,390,320]) {
   const c=await browser.newContext({ignoreHTTPSErrors:true,viewport:{width,height:width===1366?768:844},colorScheme:'light',reducedMotion:'reduce'});const p=await c.newPage();let images=[];
   const selectors=['.landing-hero','.landing-paths','.landing-start','.landing-explore p'];
   const gaps=[];
   for(const [name,url] of [['baseline',baseline],['light',a]]) {
    await go(p,url);
    // Full-width capture includes transformed arrows beyond section bounds.
    // Cropping at a rotated border can change edge rasterization at 320 px.
    await p.locator('.landing-explore').scrollIntoViewIfNeeded();
    await p.locator('.landing img:visible').evaluateAll(es=>Promise.all(es.map(e=>e.decode())));
    await p.evaluate(()=>window.scrollTo(0,0));
    const clip=await p.evaluate(()=>{
     const first=document.querySelector('.landing-hero').getBoundingClientRect();
     const last=document.querySelector('.landing-explore p').getBoundingClientRect();
     return {x:0,y:first.top+scrollY,width:innerWidth,height:last.bottom-first.top};
    });
    images.push(await p.screenshot({fullPage:true,clip}));
    gaps.push(await p.evaluate(()=>{const nav=document.querySelector('.landing-explore p').getBoundingClientRect();const meta=document.querySelector('.landing-footer p').getBoundingClientRect();return meta.top-nav.bottom}));
   }
   const identical=images[0].equals(images[1]);
   report.lightBaseline.push({width,sections:selectors,contentPixelsIdentical:identical,exploreToMetadataGap:{before:gaps[0],after:gaps[1]}});
   assert(identical,`Light content differs at ${width}`);assert.equal(gaps[0]-gaps[1],8);await c.close();
  }
 }catch(e){report.failures.push(e.stack);throw e}finally{fs.writeFileSync(path.join(__dirname,'browser-results.json'),JSON.stringify(report,null,2)+'\n');await browser.close()}
 assert.equal(report.failures.length,0,report.failures.join('\n'));console.log('PASS: native behavior and approved light sections');
 assert.equal(report.accessibilityFailures.length,0,report.accessibilityFailures.join('\n'));console.log('PASS: automated accessibility');
})().catch(e=>{console.error(e);process.exitCode=1});
