// Review-only computed text contrast and unchanged native API-table finding.
const {chromium}=require('playwright');
const {AxeBuilder}=require('@axe-core/playwright');
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const base=process.argv[2];assert(base,'Supply site URL ending in /');
function luminance(color){const rgb=color.match(/[\d.]+/g).slice(0,3).map(Number).map(c=>{c/=255;return c<=.04045?c/12.92:((c+.055)/1.055)**2.4});return rgb[0]*.2126+rgb[1]*.7152+rgb[2]*.0722}
function contrast(a,b){const x=luminance(a),y=luminance(b);return (Math.max(x,y)+.05)/(Math.min(x,y)+.05)}
(async()=>{const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH||'/usr/bin/chromium',args:['--no-sandbox']});const report={contrast:[],nativeApi:[]};
try{for(const mode of ['light','dark']){const c=await browser.newContext({viewport:{width:320,height:844},colorScheme:mode,ignoreHTTPSErrors:true});const p=await c.newPage();await p.goto(new URL('project/',base).href);await p.waitForFunction(()=>document.body.hasAttribute('data-md-color-media'));
const samples=await p.evaluate(()=>{const body=getComputedStyle(document.body).backgroundColor;return ['.pv-overview > p','.pv-overview a','.pv-overview-links p:nth-child(2)','.pv-overview-card','.pv-overview-label','.pv-overview-card a'].map(selector=>{const e=document.querySelector(selector);return {selector,foreground:getComputedStyle(e).color,background:e.closest('.pv-overview-card')?getComputedStyle(e.closest('.pv-overview-card')).backgroundColor:body}})});
for(const s of samples){s.ratio=Number(contrast(s.foreground,s.background).toFixed(2));assert(s.ratio>=4.5,JSON.stringify(s));report.contrast.push({theme:mode,...s})}
await p.goto(new URL('reference/api/',base).href);await p.evaluate(()=>document.fonts.ready);const a=await new AxeBuilder({page:p}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();report.nativeApi.push({theme:mode,violations:a.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))}))});await c.close()}}
finally{fs.writeFileSync(path.join(__dirname,'contrast-results.json'),JSON.stringify(report,null,2)+'\n');await browser.close()}
console.log('Text contrast passes; native API findings:',report.nativeApi.map(x=>[x.theme,x.violations.map(v=>v.id)]));if(report.nativeApi.some(x=>x.violations.length))process.exitCode=2;
})().catch(e=>{console.error(e);process.exitCode=1});
