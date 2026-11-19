/* Review-only browser sweep. No production dependencies or CI job.
   NODE_PATH=/path/to/review/node_modules node review/w5-b/verify.cjs [candidate] [base]
   Needs playwright, @axe-core/playwright and pngjs. Exit 2 retains axe findings. */
const {chromium}=require('playwright');
const {default:AxeBuilder}=require('@axe-core/playwright');
const {PNG}=require('pngjs');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const candidate=process.argv[2]||'http://127.0.0.1:8000/';
const baseline=process.argv[3]||'http://127.0.0.1:8001/';
const report={browser:'',views:[],reflow:[],keyboard:[],functional:[],axe:[],unchangedPixels:[],failures:[]};
const out=path.join(__dirname,'screenshots');fs.mkdirSync(out,{recursive:true});
async function ready(p){await p.locator('article').waitFor();await p.evaluate(()=>document.fonts.ready);await p.evaluate(()=>window.MathJax?.startup?.promise);await p.waitForTimeout(200);}
async function go(p,route,base=candidate){await p.goto(new URL(route,base).href);await ready(p);}
async function flow(p){return p.evaluate(()=>{
 const width=document.documentElement.clientWidth;
 const escaped=[...document.querySelectorAll('.pv-overview :is(p,h1,h2,h3,figure,img,a), .pv-overview .arithmatex > mjx-container > mjx-math')].filter(e=>e.getClientRects().length&&!e.closest('mjx-assistive-mml')).filter(e=>{const r=e.getBoundingClientRect();return r.x < -1||r.right>width+1;}).map(e=>({tag:e.tagName,text:e.textContent.slice(0,80),rect:e.getBoundingClientRect().toJSON()}));
 const localOverflow=[...document.querySelectorAll('.pv-overview-card, .pv-overview-row, .pv-overview-equation, .pv-overview-equation .arithmatex')].filter(e=>e.scrollWidth>e.clientWidth+1).map(e=>({class:e.className,width:e.clientWidth,scroll:e.scrollWidth}));
 return {viewport:innerWidth,width,scroll:document.documentElement.scrollWidth,escaped,localOverflow};
});}
async function checkFlow(p){const f=await flow(p);assert(f.scroll<=f.viewport);assert.deepEqual(f.escaped,[]);assert.deepEqual(f.localOverflow,[]);return f;}
async function axe(p,where){const r=await new AxeBuilder({page:p}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();report.axe.push({where,violations:r.violations.map(v=>({id:v.id,impact:v.impact,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))}))});}
async function figures(p,mode){const imgs=await p.locator('.pv-overview img:visible').evaluateAll(es=>es.map(e=>({src:e.getAttribute('src'),loaded:e.complete&&e.naturalWidth>0,alt:e.alt,ratio:e.width/e.height,native:e.naturalWidth/e.naturalHeight})));
 assert(imgs.every(i=>i.loaded&&i.alt.length>30&&Math.abs(i.ratio-i.native)<.02&&i.src.endsWith(mode==='dark'?'#only-dark':'#only-light')));assert.equal(await p.locator('.pv-overview').getByRole('img').count(),imgs.length);return imgs;
}
async function palette(p,title,scheme){await p.locator(`[data-md-component="palette"] label[title="${title}"]`).click();await p.waitForFunction(s=>document.body.dataset.mdColorScheme===s,scheme);}
(async()=>{const b=await chromium.launch({executablePath:process.env.CHROMIUM_PATH||'/usr/bin/chromium',args:['--no-sandbox']});report.browser=b.version();
 try{
 for(const [theme,mode,scheme] of [['light','light','default'],['carbon','dark','slate']]){
  const c=await b.newContext({viewport:{width:1366,height:900},colorScheme:mode,reducedMotion:'reduce',ignoreHTTPSErrors:true});const p=await c.newPage();p.setDefaultTimeout(20000);const errors=[];p.on('pageerror',e=>errors.push(e.message));
  for(const route of ['examples/','guide/','theory/']){
   for(const width of [1366,390,320]){
    await p.setViewportSize({width,height:width===1366?900:844});await go(p,route);
    assert.equal(await p.locator('body').getAttribute('data-md-color-scheme'),scheme);
    assert.equal(await p.locator('article h1').count(),1);
    const imgs=await figures(p,mode);assert.equal(imgs.length,route==='guide/'?2:route==='theory/'?1:0);
    const positions=await p.locator('.pv-overview-row').evaluateAll(es=>es.map(e=>{const t=e.querySelector(':scope > div').getBoundingClientRect(),f=e.querySelector('figure').getBoundingClientRect();return {tx:t.x,ty:t.y,fx:f.x,fy:f.y,textFirst:e.firstElementChild.tagName==='DIV'};}));
    assert(positions.every(r=>r.textFirst));
    if(positions.length){if(width===1366){assert(positions[0].tx<positions[0].fx);assert(positions[1].tx>positions[1].fx);}else assert(positions.every(r=>r.ty<r.fy&&r.tx===r.fx));}
    if(route==='examples/'){
     assert.equal(await p.locator('.pv-overview-card').count(),8);assert.equal(await p.locator('.pv-overview-card h3 a').count(),8);assert.equal(await p.locator('.pv-overview-card a[href$=".ipynb"]').count(),8);
     const rects=await p.locator('.pv-overview-card').evaluateAll(es=>es.map(e=>({y:e.offsetTop,h:e.offsetHeight,source:e.lastElementChild.getBoundingClientRect().bottom})));
     if(width===1366)for(let i=0;i<8;i+=2){assert.equal(rects[i].h,rects[i+1].h);assert(Math.abs(rects[i].source-rects[i+1].source)<1);}
    }
    if(route==='theory/'){assert.equal(await p.locator('article h2').count(),2);assert.equal(await p.locator('.pv-overview-equation > .arithmatex > mjx-container').count(),1);}
    await p.screenshot({path:path.join(out,`${route.slice(0,-1)}-${theme}-${width}.png`),fullPage:true});
    report.views.push({route,theme,...await checkFlow(p),images:imgs,positions});await axe(p,`${route} ${theme} ${width}`);
    for(const factor of [1.5,2]){await p.evaluate(f=>document.documentElement.style.fontSize=parseFloat(getComputedStyle(document.documentElement).fontSize)*f+'px',factor);report.reflow.push({route,theme,factor,...await checkFlow(p)});await p.evaluate(()=>document.documentElement.style.removeProperty('font-size'));}
   }
   await p.setViewportSize({width:1366,height:900});await go(p,route);const expected=await p.locator('.pv-overview a').count(),seen=new Set();
   for(let n=0;n<220;n++){await p.keyboard.press('Tab');const s=await p.evaluate(()=>{const e=document.activeElement;if(!e.matches('.pv-overview a'))return null;const r=e.getBoundingClientRect(),s=getComputedStyle(e);return {index:[...document.querySelectorAll('.pv-overview a')].indexOf(e),outline:s.outlineWidth,style:s.outlineStyle,x:r.x,right:r.right,top:r.top,bottom:r.bottom};});if(s){assert.equal(s.outline,'3px');assert.equal(s.style,'solid');assert(s.x>=0&&s.right<=1366&&s.bottom>96&&s.top<900);seen.add(s.index);}if(seen.size===expected)break;}
   assert.equal(seen.size,expected);report.keyboard.push({route,theme,links:seen.size,focus:'3px solid'});console.log(theme,route,'views and keyboard passed');
  }
  await go(p,'guide/');await palette(p,'Switch to light mode','default');await figures(p,'light');await p.emulateMedia({colorScheme:'dark'});await p.reload();await ready(p);assert.equal(await p.locator('body').getAttribute('data-md-color-scheme'),'default');
  await palette(p,'Switch to dark mode','slate');await figures(p,'dark');await p.emulateMedia({colorScheme:'light'});await p.reload();await ready(p);assert.equal(await p.locator('body').getAttribute('data-md-color-scheme'),'slate');
  await palette(p,'Switch to system preference','default');await p.emulateMedia({colorScheme:'dark'});await p.waitForFunction(()=>document.body.dataset.mdColorScheme==='slate');await figures(p,'dark');await p.emulateMedia({colorScheme:mode});
  report.functional.push(`${theme}: native Auto/Light/Dark, persistence, live OS changes and image variants`);
  await go(p,'guide/');for(let n=0;n<12;n++){await p.keyboard.press('Tab');if(await p.evaluate(()=>document.activeElement.name==='__palette'))break;}assert.equal(await p.evaluate(()=>document.activeElement.name),'__palette');await p.keyboard.press('ArrowRight');await p.waitForFunction(()=>document.body.dataset.mdColorScheme==='default');await p.keyboard.press('ArrowRight');await p.waitForFunction(()=>document.body.dataset.mdColorScheme==='slate');await p.keyboard.press('ArrowRight');
  report.functional.push(`${theme}: native palette via Tab and arrow keys`);
  await p.evaluate(()=>window.w5Sentinel='retained');
  for(const [label,route] of [['Examples','examples/'],['Theory','theory/'],['API Reference','reference/'],['Development','development/'],['Project','project/'],['Home','']]){await p.locator('.md-tabs').getByRole('link',{name:label,exact:true}).click();await p.waitForURL(new URL(route,candidate).href);await ready(p);assert.equal(await p.evaluate(()=>window.w5Sentinel),'retained');if(route==='theory/')assert.equal(await p.locator('.pv-overview-equation > .arithmatex > mjx-container').count(),1);}
  await p.locator('.landing-paths a[href*="guide/forward"]').first().click();await p.waitForURL(new URL('guide/forward/',candidate).href);await ready(p);await p.goBack();await p.locator('.pv-home').waitFor();await p.goForward();await p.locator('article').waitFor();await ready(p);assert.equal(await p.evaluate(()=>window.w5Sentinel),'retained');
  report.functional.push(`${theme}: instant navigation across seven sections, MathJax on entry, child route and history`);
  await p.getByRole('button',{name:/^Search/}).click();await p.getByPlaceholder('Search',{exact:true}).fill('periodic');const hit=p.locator('[role="option"][href*="/guide/periodic-contacts/"]').first();await hit.waitFor();await axe(p,`Search ${theme}`);await hit.click();await p.waitForURL(u=>u.pathname===new URL('guide/periodic-contacts/',candidate).pathname);await ready(p);report.functional.push(`${theme}: native Search and result navigation`);
  await p.setViewportSize({width:320,height:844});await go(p,'guide/');await p.evaluate(()=>document.documentElement.style.fontSize=parseFloat(getComputedStyle(document.documentElement).fontSize)*2+'px');await checkFlow(p);await p.locator('.md-header label[for="__search"]').click();await p.getByPlaceholder('Search',{exact:true}).fill('Voronoi');await p.locator('[role="option"]').first().waitFor();await p.keyboard.press('Escape');await p.locator('.md-header label[for="__drawer"]').click();await p.locator('.md-sidebar--primary label.md-nav__link').filter({hasText:'Tessellations & periodic geometry'}).click();await p.locator('.md-sidebar--primary').getByRole('link',{name:'Planar 2D',exact:true}).click();await p.waitForURL(new URL('guide/planar/',candidate).href);await ready(p);await p.evaluate(()=>document.documentElement.style.removeProperty('font-size'));report.functional.push(`${theme}: 320px / 200% text Search and drawer`);
  await go(p,'reference/api/');await axe(p,`Native API table ${theme} 320`);assert.deepEqual(errors,[]);await c.close();console.log(theme,'overview, reflow, keyboard and functional checks passed');
 }
 for(const mode of ['light','dark'])for(const width of [1366,320]){
  const c=await b.newContext({viewport:{width,height:900},colorScheme:mode,reducedMotion:'reduce',ignoreHTTPSErrors:true});const p=await c.newPage();p.setDefaultTimeout(20000);
  for(const route of ['','project/','reference/','development/','guide/forward/','theory/power-diagrams/']){
   const shots=[];for(const base of [baseline,candidate]){await go(p,route,base);shots.push(PNG.sync.read(await p.screenshot({fullPage:true,animations:'disabled'})));}
   assert.equal(shots[0].width,shots[1].width);assert.equal(shots[0].height,shots[1].height);let changed=0,maxDelta=0;for(let i=0;i<shots[0].data.length;i+=4){let d=0;for(let k=0;k<4;k++)d=Math.max(d,Math.abs(shots[0].data[i+k]-shots[1].data[i+k]));if(d)changed++;maxDelta=Math.max(maxDelta,d);}const withinRasterTolerance=changed<=16&&maxDelta<=8;report.unchangedPixels.push({route,mode,width,changed,maxDelta,withinRasterTolerance});assert(withinRasterTolerance,`${route}: ${changed} pixels, max delta ${maxDelta}`);
  }await c.close();
 }
 }catch(e){report.failures.push(e.stack);throw e;}finally{fs.writeFileSync(path.join(__dirname,'browser-results.json'),JSON.stringify(report,null,2)+'\n');await b.close();}
 const findings=report.axe.filter(x=>x.violations.length);console.log('Functional/layout checks passed; unsuppressed axe findings:',findings.map(x=>x.where));if(findings.length)process.exitCode=2;
})().catch(e=>{console.error(e);process.exitCode=1;});
