// Targeted regression checks for cold native bootstrap and enlarged-text header.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const urls=process.argv.slice(2);
if(urls.length!==2)throw Error('Supply A_URL B_URL');
const report={cold:[],reflow:[]};
const selectors=['.pv-home__brand','[data-md-component="palette"] label:not([hidden])','.md-search__button','.pv-home__links a:nth-child(1)','.pv-home__links a:nth-child(2)','.pv-home__links a:nth-child(3)'];
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH||undefined,args:['--no-sandbox']});
 try{
 for(const [i,url] of urls.entries()){
  for(const os of ['light','dark'])for(const route of ['','guide/forward/']){
   const c=await browser.newContext({colorScheme:os,viewport:{width:1366,height:768}});const p=await c.newPage();
   await p.route('**/assets/javascripts/bundle.*.min.js',async r=>{await new Promise(r=>setTimeout(r,1500));await r.continue()});
   const loading=p.goto(new URL(route,url).href);loading.catch(()=>{});
   await p.waitForFunction(()=>performance.getEntriesByName('first-contentful-paint').length>0);
   const before=await p.evaluate(()=>({media:document.body.getAttribute('data-md-color-media'),background:getComputedStyle(document.body).backgroundColor,images:[...document.querySelectorAll('.landing img')].filter(e=>getComputedStyle(e).display!=='none').map(e=>e.getAttribute('src')),fcp:performance.getEntriesByName('first-contentful-paint')[0].startTime}));
   assert.equal(before.media,null,'Test must observe cold page before bundle initialization');
   const expected=os==='dark'?(i===0?'rgb(36, 39, 40)':'rgb(16, 45, 53)'):(route?'rgb(255, 255, 255)':'rgb(250, 250, 247)');
   assert.equal(before.background,expected);if(!route)assert(before.images.every(s=>s.endsWith(os==='dark'?'#only-dark':'#only-light')));
   report.cold.push({candidate:i===0?'graphite':'midnight',os,route,delayMs:1500,...before});
   await loading;await p.waitForFunction(()=>document.body.hasAttribute('data-md-color-media'));
   // Saved manual light must win over a dark OS even during delayed reload.
   if(os==='dark'&&!route){await p.locator('label[title="Switch to light mode"]').click();const reload=p.reload();reload.catch(()=>{});await p.waitForFunction(()=>performance.getEntriesByName('first-contentful-paint').length>0);assert.equal(await p.evaluate(()=>getComputedStyle(document.body).backgroundColor),'rgb(250, 250, 247)');await reload;report.cold.push({candidate:i,manualLightSurvivesDelayedReload:true})}
   await c.close();
  }
  const c=await browser.newContext({colorScheme:'dark'});const p=await c.newPage();
  for(const width of [1366,390,320])for(const fontSize of [20,25,30,40]){
   await p.setViewportSize({width,height:844});await p.goto(url);await p.evaluate(n=>document.documentElement.style.fontSize=n+'px',fontSize);
   const boxes=await Promise.all(selectors.map(s=>p.locator(s).boundingBox()));
   for(let a=0;a<boxes.length;a++)for(let b=a+1;b<boxes.length;b++){
    const x=boxes[a],y=boxes[b];assert(x&&y);assert(!(x.x<y.x+y.width&&x.x+x.width>y.x&&x.y<y.y+y.height&&x.y+x.height>y.y),`overlap ${width}/${fontSize}: ${selectors[a]} ${selectors[b]}`);
   }
   assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   for(const s of selectors.slice(1,3))assert(await p.locator(s).evaluate(e=>{const r=e.getBoundingClientRect();const hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);return e===hit||e.contains(hit)}));
   await p.locator('label[title="Switch to light mode"]').click();await p.locator('label[title="Switch to dark mode"]').click();await p.locator('label[title="Switch to system preference"]').click();
   report.reflow.push({candidate:i,width,fontSize,disjoint:true,hitTest:true});
   if(fontSize===40)await p.screenshot({path:path.join(__dirname,'screenshots',`${i?'midnight':'graphite'}-text-200-${width}.png`),fullPage:true});
  }
  await c.close();
 }
 }finally{fs.writeFileSync(path.join(__dirname,'edge-results.json'),JSON.stringify(report,null,2)+'\n');await browser.close()}
 console.log('PASS cold paint and header reflow');
})().catch(e=>{console.error(e);process.exitCode=1});
