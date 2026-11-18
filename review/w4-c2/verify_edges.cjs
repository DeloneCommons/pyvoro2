// Test contexts ignore the environment proxy certificate error for the public MathJax CDN.
// Targeted regression checks for cold native bootstrap and enlarged-text header.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const urls=process.argv.slice(2);
if(urls.length!==1)throw Error('Supply PRODUCTION_URL');
const report={cold:[],reflow:[]};
const selectors=['.pv-home__brand','[data-md-component="palette"] label:not([hidden])','.md-search__button','.pv-home__links a:nth-child(1)','.pv-home__links a:nth-child(2)','.pv-home__links a:nth-child(3)'];
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH||undefined,args:['--no-sandbox']});
 try{
 for(const url of urls){
  for(const os of ['light','dark'])for(const route of ['','guide/forward/']){
   const c=await browser.newContext({ignoreHTTPSErrors:true,colorScheme:os,viewport:{width:1366,height:768}});const p=await c.newPage();
   await p.route('**/assets/javascripts/bundle.*.min.js',async r=>{await new Promise(r=>setTimeout(r,1500));await r.continue()});
   const loading=p.goto(new URL(route,url).href);loading.catch(()=>{});
   await p.waitForFunction(()=>performance.getEntriesByName('first-contentful-paint').length>0);
   const before=await p.evaluate(()=>({media:document.body.getAttribute('data-md-color-media'),background:getComputedStyle(document.body).backgroundColor,images:[...document.querySelectorAll('.landing img')].filter(e=>getComputedStyle(e).display!=='none').map(e=>e.getAttribute('src')),fcp:performance.getEntriesByName('first-contentful-paint')[0].startTime}));
   assert.equal(before.media,null,'Test must observe cold page before bundle initialization');
   const expected=os==='dark'?'rgb(36, 39, 40)':(route?'rgb(255, 255, 255)':'rgb(250, 250, 247)');
   assert.equal(before.background,expected);if(!route)assert.equal(before.images.length,2);if(!route)assert(before.images.every(s=>s.endsWith(os==='dark'?'#only-dark':'#only-light')));
   report.cold.push({theme:'carbon',os,route,delayMs:1500,...before});
   await loading;await p.waitForFunction(()=>document.body.hasAttribute('data-md-color-media'));
   // Saved manual light must win over a dark OS even during delayed reload.
   if(os==='dark'&&!route){await p.locator('label[title="Switch to light mode"]').click();const reload=p.reload();reload.catch(()=>{});await p.waitForFunction(()=>performance.getEntriesByName('first-contentful-paint').length>0);assert.equal(await p.evaluate(()=>getComputedStyle(document.body).backgroundColor),'rgb(250, 250, 247)');await reload;report.cold.push({manualLightSurvivesDelayedReload:true})}
   if(os==='light'&&!route){
    await p.locator('label[title="Switch to light mode"]').click();await p.locator('label[title="Switch to dark mode"]').click();
    const reload=p.reload();reload.catch(()=>{});await p.waitForFunction(()=>performance.getEntriesByName('first-contentful-paint').length>0);
    const state=await p.evaluate(()=>({bg:getComputedStyle(document.body).backgroundColor,images:[...document.querySelectorAll('.landing img')].filter(e=>getComputedStyle(e).display!=='none').map(e=>e.getAttribute('src'))}));
    assert.equal(state.bg,'rgb(36, 39, 40)');assert.equal(state.images.length,2);assert(state.images.every(s=>s.endsWith('#only-dark')));await reload;report.cold.push({manualDarkSurvivesDelayedReload:true});
   }
   await c.close();
  }
  for(const mode of ['light','dark']) {
  const c=await browser.newContext({ignoreHTTPSErrors:true,colorScheme:mode});const p=await c.newPage();
  for(const width of [1366,390,320])for(const fontSize of [20,25,30,40]){
   await p.setViewportSize({width,height:844});await p.goto(url);await p.evaluate(n=>document.documentElement.style.fontSize=n+'px',fontSize);
   const boxes=await Promise.all(selectors.map(s=>p.locator(s).boundingBox()));
   for(let a=0;a<boxes.length;a++)for(let b=a+1;b<boxes.length;b++){
    const x=boxes[a],y=boxes[b];assert(x&&y);assert(!(x.x<y.x+y.width&&x.x+x.width>y.x&&x.y<y.y+y.height&&x.y+x.height>y.y),`overlap ${width}/${fontSize}: ${selectors[a]} ${selectors[b]}`);
   }
   assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   for(const s of selectors.slice(1,3))assert(await p.locator(s).evaluate(e=>{const r=e.getBoundingClientRect();const hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);return e===hit||e.contains(hit)}));
   await p.locator('label[title="Switch to light mode"]').click();await p.locator('label[title="Switch to dark mode"]').click();await p.locator('label[title="Switch to system preference"]').click();
   report.reflow.push({theme:mode,width,fontSize,disjoint:true,hitTest:true});
   if(fontSize===40)await p.screenshot({path:path.join(__dirname,'screenshots',`${mode==='dark'?'carbon':'light'}-text-200-${width}.png`),fullPage:true});
  }
  for(const route of ['guide/forward/','reference/api/','theory/power-diagrams/'])for(const width of [1366,390,320]) {
   await p.setViewportSize({width,height:844});await p.goto(new URL(route,url).href);await p.evaluate(()=>window.MathJax?.startup?.promise);await p.evaluate(()=>document.documentElement.style.fontSize='40px');
   assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`200% ${route} ${width}`);
   await p.locator('label[title="Switch to light mode"]').click();await p.locator('label[title="Switch to dark mode"]').click();await p.locator('label[title="Switch to system preference"]').click();
   if(width<=390) {
   const search=p.locator('.md-header label[for="__search"]');
   assert(await search.evaluate(e=>{const r=e.getBoundingClientRect();const hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);return e===hit||e.contains(hit)}));
   await search.click();await p.getByPlaceholder('Search',{exact:true}).waitFor();await p.keyboard.press('Escape');
   await p.locator('.md-header label[for="__drawer"]').click();
   await p.waitForFunction(()=>{const r=document.querySelector('.md-sidebar--primary').getBoundingClientRect();return r.x>=0&&r.right<=innerWidth});
   const drawer=await p.locator('.md-sidebar--primary').boundingBox();assert(drawer.x>=0&&drawer.x+drawer.width<=width);
   if(width===320&&route==='guide/forward/')await p.screenshot({path:path.join(__dirname,'screenshots',`${mode==='dark'?'carbon':'light'}-drawer-text-200.png`)});
   // Follow the native section-back control and Home link, which closes the drawer.
   if(route.startsWith('guide'))await p.locator('label.md-nav__title[for="__nav_2_2"]').click();
   if(route.startsWith('reference'))await p.locator('label.md-nav__title[for="__nav_5_3"]').click();
   await p.locator(`label.md-nav__title[for="${route.startsWith('guide')?'__nav_2':route.startsWith('reference')?'__nav_5':'__nav_4'}"]`).click();
   await p.locator('.md-sidebar--primary').getByRole('link',{name:'Home',exact:true}).click();await p.locator('.pv-home').waitFor();
   await p.goto(new URL(route,url).href);await p.evaluate(()=>window.MathJax?.startup?.promise);await p.evaluate(()=>document.documentElement.style.fontSize='40px');
   }
   report.reflow.push({theme:mode,route,width,fontSize:40,noOverflow:true,paletteAccessible:true,mobileSearchAndDrawer:width<=390?'passed':'desktop layout'});
   if(width===390)await p.screenshot({path:path.join(__dirname,'screenshots',`${mode==='dark'?'carbon':'light'}-${route.split('/')[0]}-text-200.png`)});
  }
  await c.close();
  }
 }
 }finally{fs.writeFileSync(path.join(__dirname,'edge-results.json'),JSON.stringify(report,null,2)+'\n');await browser.close()}
 console.log('PASS cold paint and header reflow');
})().catch(e=>{console.error(e);process.exitCode=1});
