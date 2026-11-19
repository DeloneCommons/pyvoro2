// Review-only: NODE_PATH=<scratch node_modules> node review/w5-a/verify.cjs
// CANDIDATE_URL BASELINE_URL SPECIMEN_URL (all ending in /).
// PROJECT_ONLY=1 refreshes only Project screenshots and functional checks.
// Existing specimen captures and design-baseline evidence are preserved.
const {chromium} = require('playwright');
const {PNG} = require('pngjs');
const {AxeBuilder} = require('@axe-core/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const [candidate, baseline, specimen] = process.argv.slice(2);
const projectOnly=process.env.PROJECT_ONLY==='1';
assert(candidate && (projectOnly || (baseline && specimen)), 'Supply site URLs; PROJECT_ONLY=1 needs only candidate');
const compareOnly=process.env.COMPARE_ONLY==='1';
const resultFile=projectOnly?'editorial-browser-results.json':'browser-results.json';
const report = compareOnly ? JSON.parse(fs.readFileSync(path.join(__dirname,resultFile),'utf8')) : {browser:'', project:[], specimen:[], reflow:[], keyboard:[], functional:[], unchangedPixels:[], axe:[], failures:[]};
if(compareOnly){report.failures=[];report.unchangedPixels=[];}
const out = path.join(__dirname, 'screenshots');
fs.mkdirSync(out, {recursive:true});
async function ready(p) {
  await p.locator('article').waitFor();
  await p.evaluate(()=>document.fonts.ready);
  await p.evaluate(()=>window.MathJax?.startup?.promise);
  await p.waitForTimeout(200);
}
async function go(p, base, route='project/') { await p.goto(new URL(route,base).href); await ready(p); }
async function reflow(p) {
  const dimensions = await p.evaluate(()=>({viewport:innerWidth, document:document.documentElement.scrollWidth}));
  assert(dimensions.document <= dimensions.viewport, JSON.stringify(dimensions));
  return dimensions;
}
async function axe(p, where) {
  const results = await new AxeBuilder({page:p}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
  report.axe.push({where, violations:results.violations.map(v=>({id:v.id,impact:v.impact,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))}))});
}
async function palette(p,title,scheme) {
  await p.locator(`[data-md-component="palette"] label[title="${title}"]`).click();
  await p.waitForFunction(s=>document.body.dataset.mdColorScheme===s,scheme);
}
(async()=>{
 const browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH||'/usr/bin/chromium',args:['--no-sandbox']});
 report.browser=browser.version();
 try {
  for (const [name,mode,scheme] of (compareOnly ? [] : [['light','light','default'],['carbon','dark','slate']])) {
   const context=await browser.newContext({viewport:{width:1366,height:768},colorScheme:mode,reducedMotion:'reduce',ignoreHTTPSErrors:true});
   const p=await context.newPage();
   const pageErrors=[];p.on('pageerror',e=>pageErrors.push(e.message));
   for (const width of [1366,390,320]) {
    await p.setViewportSize({width,height:width===1366?768:844});await go(p,candidate);
    assert.equal(await p.locator('body').getAttribute('data-md-color-scheme'),scheme);
    const dimensions=await reflow(p);
    if(projectOnly){assert.equal(await p.locator('.pv-overview > ul > li').count(),6);assert.equal(await p.locator('.pv-overview-card').count(),2);assert.equal(await p.locator('.pv-overview-links > div').count(),3);assert(!/chemvoro/i.test(await p.locator('article').innerText()));}
    assert.equal(await p.locator('article h1').count(),1);
    assert.equal(await p.locator('article h2').count(),3);
    assert.equal(await p.locator('article h3').count(),0,'Project must keep one level of section headings');
    await p.screenshot({path:path.join(out,`${name}-${width}.png`),fullPage:true});
    report.project.push({theme:name,...dimensions});
    await axe(p,`Project ${name} ${width}`);
    for(const factor of [1.5,2]) {
     const normal=await p.evaluate(()=>parseFloat(getComputedStyle(document.documentElement).fontSize));
     await p.evaluate(size=>document.documentElement.style.fontSize=`${size}px`,normal*factor);
     report.reflow.push({page:'Project',theme:name,factor,...await reflow(p)});
     await p.evaluate(()=>document.documentElement.style.removeProperty('font-size'));
    }
   }
   // Real Tab traversal through every article link, with visible focus and no clipping.
   await p.setViewportSize({width:1366,height:768});await go(p,candidate);
   const expected=await p.locator('.pv-overview a').count();const reached=new Set();
   for(let n=0;n<160;n++) {
    await p.keyboard.press('Tab');
    const state=await p.evaluate(()=>{const e=document.activeElement;if(!e?.matches('.pv-overview a'))return null;const r=e.getBoundingClientRect(),s=getComputedStyle(e);return {index:[...document.querySelectorAll('.pv-overview a')].indexOf(e),text:e.textContent,width:r.width,x:r.x,bottom:r.bottom,top:r.top,outline:s.outlineWidth,style:s.outlineStyle}});
    if(state){assert.equal(state.outline,'3px');assert.equal(state.style,'solid');assert(state.x>=0&&state.x+state.width<=1366);assert(state.bottom>48&&state.top<768);reached.add(state.index);}
    if(reached.size===expected)break;
   }
   assert.equal(reached.size,expected);report.keyboard.push({theme:name,articleLinksReached:reached.size,focus:'3px solid'});
   // Keyboard operates the native radio group as well as article links.
   await go(p,candidate);
   for(let n=0;n<12;n++){await p.keyboard.press('Tab');if(await p.evaluate(()=>document.activeElement.name==='__palette'))break;}
   assert.equal(await p.evaluate(()=>document.activeElement.name),'__palette');
   await p.keyboard.press('ArrowRight');await p.waitForFunction(()=>document.body.dataset.mdColorScheme==='default');
   await p.keyboard.press('ArrowRight');await p.waitForFunction(()=>document.body.dataset.mdColorScheme==='slate');
   await p.keyboard.press('ArrowRight');
   report.keyboard.push({theme:name,nativePalette:'Tab and arrow keys'});
   // Native palette, persistence and Auto behavior.
   await go(p,candidate);
   await palette(p,'Switch to light mode','default');await p.emulateMedia({colorScheme:'dark'});await p.reload();await ready(p);assert.equal(await p.locator('body').getAttribute('data-md-color-scheme'),'default');
   await palette(p,'Switch to dark mode','slate');await p.emulateMedia({colorScheme:'light'});await p.reload();await ready(p);assert.equal(await p.locator('body').getAttribute('data-md-color-scheme'),'slate');
   await palette(p,'Switch to system preference','default');await p.emulateMedia({colorScheme:'dark'});await p.waitForFunction(()=>document.body.dataset.mdColorScheme==='slate');
   await p.emulateMedia({colorScheme:mode});
   report.functional.push(`${name}: native Auto/Light/Dark, persistence, live OS change`);
   // Instant routing across Home and normal docs retains the same JS document.
   await p.evaluate(()=>window.w5Sentinel='retained');
   for(const [label,route] of [['User Guide','guide/'],['Examples','examples/'],['Theory','theory/'],['API Reference','reference/'],['Development','development/'],['Project','project/'],['Home','']]) {
    await p.locator('.md-tabs').getByRole('link',{name:label,exact:true}).click();await p.waitForURL(new URL(route,candidate).href);await ready(p);assert.equal(await p.evaluate(()=>window.w5Sentinel),'retained');
   }
   await p.locator('.landing-footer').getByRole('link',{name:'Project',exact:true}).click();await p.waitForURL(new URL('project/',candidate).href);await ready(p);
   await p.goBack();await p.locator('.pv-home').waitFor();await ready(p);await p.goForward();await p.locator('.pv-overview').waitFor();await ready(p);assert.equal(await p.evaluate(()=>window.w5Sentinel),'retained');
   report.functional.push(`${name}: instant routes through all sections + Home + history`);
   await p.getByRole('button',{name:/^Search/}).click();await p.getByPlaceholder('Search',{exact:true}).fill('periodic');
   const hit=p.locator('[role="option"][href*="/guide/periodic-contacts/"]').first();await hit.waitFor();await axe(p,`Search ${name}`);await hit.click();await p.waitForURL(u=>u.pathname===new URL('guide/periodic-contacts/',candidate).pathname);await ready(p);
   report.functional.push(`${name}: native Search result navigation`);console.log(name,'Project, keyboard, palette, instant navigation and Search passed');
   // At 320px with doubled text, Search and the Project drawer remain usable.
   await p.setViewportSize({width:320,height:844});await go(p,candidate);
   await p.evaluate(()=>document.documentElement.style.fontSize=`${parseFloat(getComputedStyle(document.documentElement).fontSize)*2}px`);
   await reflow(p);await p.locator('.md-header label[for="__search"]').click();await p.getByPlaceholder('Search',{exact:true}).fill('Voronoi');await p.locator('[role="option"]').first().waitFor();await p.keyboard.press('Escape');
   await p.locator('.md-header label[for="__drawer"]').click();
   await p.locator('.md-sidebar--primary').getByRole('link',{name:'Roadmap',exact:true}).click();await p.waitForURL(new URL('project/roadmap/',candidate).href);await ready(p);await reflow(p);
   report.functional.push(`${name}: 320px / 200% text Search and drawer navigation`);
   await p.evaluate(()=>document.documentElement.style.removeProperty('font-size'));
   // Specimen: actual cards, native light/dark image selection, alternating rows.
   for(const width of (projectOnly?[]:[1366,390,320])) {
    await p.setViewportSize({width,height:844});await go(p,specimen,'specimen/');await reflow(p);
    const images=await p.locator('.pv-overview img:visible').evaluateAll(es=>es.map(e=>({src:e.getAttribute('src'),loaded:e.complete&&e.naturalWidth>0,ratio:e.width/e.height,native:e.naturalWidth/e.naturalHeight,alt:e.alt})));
    assert.equal(images.length,4);assert.equal(await p.locator('.pv-overview').getByRole('img').count(),4);
    assert(images.every(i=>i.loaded&&i.alt.length>30&&Math.abs(i.ratio-i.native)<.02&&i.src.endsWith(mode==='dark'?'#only-dark':'#only-light')));
    const positions=await p.locator('.pv-overview-row').evaluateAll(es=>es.map(e=>{const t=e.querySelector('div').getBoundingClientRect(),f=e.querySelector('figure').getBoundingClientRect();return {tx:t.x,ty:t.y,fx:f.x,fy:f.y}}));
    if(width===1366){assert(positions[0].tx<positions[0].fx);assert(positions[1].tx>positions[1].fx);await p.screenshot({path:path.join(out,`${name}-specimen.png`),fullPage:true});}
    else {assert(positions.every(r=>r.ty<r.fy&&r.tx===r.fx));}
    report.specimen.push({theme:name,width,images:images.length,positions});await axe(p,`Specimen ${name} ${width}`);
    await p.evaluate(()=>document.documentElement.style.fontSize=`${parseFloat(getComputedStyle(document.documentElement).fontSize)*2}px`);report.reflow.push({page:'Specimen',theme:name,factor:2,...await reflow(p)});await p.evaluate(()=>document.documentElement.style.removeProperty('font-size'));
   }
   assert.deepEqual(pageErrors,[]);await context.close();console.log(name,'functional checks complete');
  }
  // Same docs stack, browser and font fallback: full-page baseline comparisons.
  for(const mode of ['light','dark']) for(const width of (projectOnly?[]:[1366,390,320])) {
   const c=await browser.newContext({viewport:{width,height:844},colorScheme:mode,reducedMotion:'reduce',ignoreHTTPSErrors:true});const p=await c.newPage();
   for(const route of ['', 'guide/','examples/','theory/','reference/','development/','project/license/']) {
    const shots=[];
    for(const base of [baseline,candidate]) {await go(p,base,route);await reflow(p);await p.waitForTimeout(400);shots.push(await p.screenshot({fullPage:true,animations:'disabled'}));}
    const images=shots.map(s=>PNG.sync.read(s));
    assert.equal(images[0].width,images[1].width);assert.equal(images[0].height,images[1].height);
    let pixelsChanged=0,maxChannelDelta=0;
    for(let i=0;i<images[0].data.length;i+=4){let delta=0;for(let c=0;c<4;c++)delta=Math.max(delta,Math.abs(images[0].data[i+c]-images[1].data[i+c]));if(delta)pixelsChanged++;maxChannelDelta=Math.max(maxChannelDelta,delta);}
    // Repeated captures of unchanged native table corners vary by a few RGB
    // levels on single pixels. Record exact differences; permit only tiny raster noise.
    const withinRasterTolerance=pixelsChanged<=16&&maxChannelDelta<=8;
    report.unchangedPixels.push({theme:mode,width,route,identical:pixelsChanged===0,pixelsChanged,maxChannelDelta,withinRasterTolerance});
    assert(withinRasterTolerance,`${mode} ${width} ${route}: ${pixelsChanged} pixels differ, max channel delta ${maxChannelDelta}`);
   }
   await c.close();
  }
 } catch(e) {report.failures.push(e.stack);throw e;}
 finally {fs.writeFileSync(path.join(__dirname,resultFile),JSON.stringify(report,null,2)+'\n');await browser.close();}
 const violations=report.axe.filter(x=>x.violations.length);
 console.log('Functional/layout checks passed. Accessibility scans with findings:',violations.map(x=>x.where));
 // Leave all native findings visible: a nonzero exit is not a functional failure.
 if(violations.length)process.exitCode=2;
})().catch(e=>{console.error(e);process.exitCode=1});
