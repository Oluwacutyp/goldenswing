"""Interactive terminal page: live candles from the engine's stack.

Served by ``terminal_server`` (same origin, ``/api`` fetches only —
no CDN, no external assets). The page itself holds no market data:
every TF switch fetches that timeframe's bars, and a 30s poll
refreshes the current timeframe + the freshness banner.

The workspace overlays come from ``charts.structure``, measured on
the frame's own bars: named session levels (the last completed
sessions' highs/lows), the session range band, swings with HH/LH/
HL/LL labels, breaks and failed breaks, liquidity sweeps, FVGs,
volatility regime, and the expected path — a drawn line joining
measured swing points to the named session target (the prior
session's low), marked as a projection, never a forecast.  Missing
overlays are listed on the page, never invented.

Studies provenance, stated on the page: candles + session shading
come from the live stack; nearby levels, zone band, Fib+OTE and
premium/discount come from the LATEST SIGNAL's studies (served by
the engine, labeled with that signal's id and time). No signal yet
means those toggles show nothing — never invented levels.

Light theme is the default (the owner's reference chart is light,
blue up candles / black down candles).  Historical navigation:
pick a date and the page shows the bars up to then, with the
structure measured on those bars only — what was knowable then.
The multi-timeframe strip shows each TF's own measured state
(trend from its own swings, regime from its own ATR).

Read-only by construction: the page issues GETs only. There is no
order, command, or control endpoint for it to call.
"""
from __future__ import annotations

TERMINAL_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GoldenSwing terminal</title>
<style>
body{background:var(--bg);color:var(--txt);font-family:system-ui,sans-serif;margin:0;padding:12px}
body[data-theme="dark"]{--bg:#11141a;--txt:#e0e0e0;--panel:#0b0e13;--line:#3a4356;--muted:#aaa;--gold:#f7b924;--btn:#1e2530;--live:#26a69a;--stale:#ef5350}
body[data-theme="light"]{--bg:#f6f7f9;--txt:#1a1d23;--panel:#ffffff;--line:#c9ced6;--muted:#5a6472;--gold:#8a6d00;--btn:#e8ecf1;--live:#00897b;--stale:#d32f2f}
h1{font-size:18px;margin:0 0 4px}
#fresh{font-size:14px;margin-bottom:8px}
#fresh .live{color:var(--live)}#fresh .stale{color:var(--stale)}
#prov{color:var(--muted);font-size:13px;margin-bottom:10px}
#missing{color:var(--muted);font-size:13px;margin-bottom:10px}
button.tf,button#theme-btn,button.nav{background:var(--btn);color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:8px 14px;margin:0 6px 8px 0;font-size:15px}
button.tf.on{background:var(--gold);color:#111;border-color:var(--gold)}
#toggles{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
#toggles label{display:flex;align-items:center;gap:6px;min-height:44px;padding:0 12px;border:1px solid var(--line);border-radius:8px;font-size:14px}
#toggles input{width:18px;height:18px}
#mtf{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
#mtf .chip{border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-size:13px;cursor:pointer;background:var(--panel)}
#mtf .chip.on{border-color:var(--gold)}
#mtf .chip .t{font-weight:700}
#navrow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}
#navrow input{padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--txt);font-size:14px}
#wrap{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:8px}
canvas{display:block;width:100%;height:520px;touch-action:none}
#readout{font-family:monospace;font-size:13px;color:var(--gold);min-height:18px;margin:6px 0}
#foot{color:var(--muted);font-size:12px;margin-top:10px}
</style></head><body data-theme="light">
<h1 id="title">GoldenSwing terminal · __INSTRUMENT__</h1>
<div id="fresh">Connecting…</div>
<div id="prov"></div>
<div id="missing"></div>
<div id="mtf"></div>
<div id="tfs"><button class="tf" data-tf="M5">M5</button><button class="tf" data-tf="M15">M15</button><button class="tf" data-tf="M30">M30</button><button class="tf" data-tf="H1">H1</button><button class="tf" data-tf="H4">H4</button><button class="tf" data-tf="D1">D1</button></div>
<div><button id="theme-btn">Dark theme</button></div>
<div id="toggles"><label><input type="checkbox" id="ov-sess" checked> Sessions</label><label><input type="checkbox" id="ov-nearby" checked> Nearby levels</label><label><input type="checkbox" id="ov-fib" checked> Fib + OTE</label><label><input type="checkbox" id="ov-prem" checked> Premium/Discount</label><label><input type="checkbox" id="ov-zone" checked> Signal zone</label><label><input type="checkbox" id="ov-slevels" checked> Session levels</label><label><input type="checkbox" id="ov-srange" checked> Session range</label><label><input type="checkbox" id="ov-swings" checked> Swings</label><label><input type="checkbox" id="ov-breaks" checked> Breaks</label><label><input type="checkbox" id="ov-fvg" checked> FVGs</label><label><input type="checkbox" id="ov-liq" checked> Liquidity</label><label><input type="checkbox" id="ov-vol" checked> Vol expansion</label><label><input type="checkbox" id="ov-path" checked> Expected path</label></div>
<div id="navrow"><input type="datetime-local" id="jump" step="3600"><button class="nav" id="jump-go">Go to date</button><button class="nav" id="jump-prev">◀ day</button><button class="nav" id="jump-next">day ▶</button><button class="nav" id="jump-latest">Latest</button><span id="hist"></span></div>
<div id="wrap"><canvas id="cv" width="900" height="520"></canvas></div>
<div id="readout">Hover a candle for its values. Scroll to zoom, drag to pan, double-click to reset.</div>
<div id="foot">Read-only view — it never trades. Orders stay on the engine / Telegram path. Polls every 30s; a stale feed is labeled stale, never shown as live. The expected path joins measured swings to the named session target — an annotation, not a forecast.</div>
<script>
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const PALETTES={dark:{UP:'#26a69a',DN:'#ef5350',GOLD:'#f7b924',TXT:'#e0e0e0',GRID:'#2a3242',PANEL:'#0b0e13'},
  light:{UP:'#1a5fd7',DN:'#111111',GOLD:'#8a6d00',TXT:'#1a1d23',GRID:'#d5dae2',PANEL:'#ffffff'}};
let PAL=PALETTES.light;
let UP=PAL.UP,DN=PAL.DN,GOLD=PAL.GOLD,TXT=PAL.TXT,GRID=PAL.GRID,PANEL=PAL.PANEL;
function setTheme(name){PAL=PALETTES[name];UP=PAL.UP;DN=PAL.DN;GOLD=PAL.GOLD;TXT=PAL.TXT;GRID=PAL.GRID;PANEL=PAL.PANEL;
  document.body.dataset.theme=name;
  document.getElementById('theme-btn').textContent=name==='dark'?'Light theme':'Dark theme';draw();}
let tf='H1',bars=[],sessions=[],studies=null,status=null,structure=null,mtf=null;
let start=0,end=0,hover=-1,dragX=null,before=null,xDiv=1;
function ov(id){return document.getElementById(id).checked}
function fmt(x){return x.toLocaleString(undefined,{minimumFractionDigits:1,maximumFractionDigits:1})}
function resetWin(){const n=bars.length;end=n;start=Math.max(0,n-120)}
function yRange(bs,lo,hi){let mn=Infinity,mx=-Infinity;
  for(let i=lo;i<hi;i++){mn=Math.min(mn,bs[i][3]);mx=Math.max(mx,bs[i][2])}
  const span=Math.max(mx-mn,1e-9);
  const extra=[];
  if(studies){if(studies.zone)extra.push(studies.zone.top,studies.zone.bottom);
    studies.nearby.forEach(z=>extra.push(z.level));
    if(studies.fib)studies.fib.levels.forEach(l=>extra.push(l[1]));
    if(studies.fib&&studies.fib.range)extra.push(studies.fib.range[0],studies.fib.range[1]);}
  if(structure){structure.session_levels.forEach(l=>extra.push(l.high,l.low));
    if(structure.session_range&&structure.session_range.prior)extra.push(structure.session_range.prior.top,structure.session_range.prior.bottom);
    if(structure.expected_path)extra.push(structure.expected_path.target_price);}
  extra.forEach(v=>{if(v>=mn-1.6*span&&v<=mx+1.6*span){mn=Math.min(mn,v);mx=Math.max(mx,v)}});
  const pad=0.07*(mx-mn);return [mn-pad,mx+pad];}
function draw(){if(!bars.length){ctx.clearRect(0,0,cv.clientWidth,cv.height);return;}
  const W=cv.clientWidth,H=cv.height;cv.width=W;
  const hasVol=bars[0].length>5;
  const priceH=hasVol?Math.round(H*0.84):H;
  const lo=start,hi=end,n=hi-lo;if(n<=0)return;
  const [mn,mx]=yRange(bars,lo,hi);
  const showPath=ov('ov-path')&&structure&&structure.expected_path;
  const n2=showPath?n+4:n;xDiv=n2;
  const X=i=>((i-lo+0.5)/n2)*W, Y=v=>priceH-((v-mn)/(mx-mn))*priceH;
  ctx.clearRect(0,0,W,H);
  const seenY=[];
  function lab(txt,y,color,bold){if(seenY.some(v=>Math.abs(v-y)<13))return;seenY.push(y);ctx.font=(bold?'bold ':'')+'11px monospace';const w=ctx.measureText(txt).width;ctx.fillStyle=PANEL;ctx.fillRect(6,y-11,w+4,14);ctx.fillStyle=color;ctx.fillText(txt,8,y);}
  ctx.strokeStyle=GRID;ctx.lineWidth=0.5;
  for(let g=0;g<=4;g++){const y=priceH*g/4;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();
    ctx.fillStyle='#777';ctx.font='11px monospace';ctx.fillText(fmt(mx-(mx-mn)*g/4),4,y-2);}
  if(ov('ov-sess')){ctx.fillStyle='rgba(247,185,36,0.07)';
    sessions.forEach(s=>{const a=Math.max(s[0],lo),b=Math.min(s[1],hi);
      if(b>a)ctx.fillRect(X(a)-W/(2*n2),0,(b-a)*W/n2,priceH);});}
  if(studies){
    if(ov('ov-prem')&&studies.fib&&studies.fib.range){const rlo=studies.fib.range[0],rhi=studies.fib.range[1],eq=(rlo+rhi)/2;
      ctx.fillStyle='rgba(239,83,80,0.05)';ctx.fillRect(0,Y(rhi),W,Y(eq)-Y(rhi));
      ctx.fillStyle='rgba(38,166,154,0.05)';ctx.fillRect(0,Y(eq),W,Y(rlo)-Y(eq));
      ctx.strokeStyle=TXT;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(0,Y(eq));ctx.lineTo(W,Y(eq));ctx.stroke();ctx.setLineDash([]);}
    if(ov('ov-zone')&&studies.zone){ctx.fillStyle='rgba(247,185,36,0.25)';
      ctx.fillRect(0,Y(studies.zone.top),W,Y(studies.zone.bottom)-Y(studies.zone.top));}
    if(ov('ov-nearby')){studies.nearby.forEach(z=>{ctx.strokeStyle=z.side==='support'?UP:z.side==='resistance'?DN:TXT;
      ctx.lineWidth=z.tier==='MAJOR'?1.2:0.7;ctx.setLineDash(z.tier==='MAJOR'?[]:[6,3]);
      ctx.beginPath();ctx.moveTo(0,Y(z.level));ctx.lineTo(W,Y(z.level));ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle=ctx.strokeStyle;ctx.font='11px monospace';
      ctx.fillText((z.tier?z.tier+' ':'')+fmt(z.level),W-110,Y(z.level)-2);});}
    if(ov('ov-fib')&&studies.fib){studies.fib.levels.forEach(l=>{ctx.strokeStyle='#8d6e00';ctx.lineWidth=0.7;ctx.setLineDash([4,3]);
      ctx.beginPath();ctx.moveTo(0,Y(l[1]));ctx.lineTo(W,Y(l[1]));ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle='#8d6e00';ctx.font='11px monospace';ctx.fillText(l[0].toFixed(3),W-70,Y(l[1])-2);});
      if(studies.fib.ote){ctx.fillStyle='rgba(141,110,0,0.12)';ctx.fillRect(0,Y(studies.fib.ote[1]),W,Y(studies.fib.ote[0])-Y(studies.fib.ote[1]));}}}
  const bw=Math.max(2,(W/n2)*0.7);
  for(let i=lo;i<hi;i++){const b=bars[i],up=b[4]>=b[1],col=up?UP:DN;
    ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(X(i),Y(b[2]));ctx.lineTo(X(i),Y(b[3]));ctx.stroke();
    const t=Y(Math.max(b[1],b[4])),bt=Y(Math.min(b[1],b[4]));
    ctx.fillRect(X(i)-bw/2,t,bw,Math.max(bt-t,1));}
  if(structure){
    const vol=structure.volatility;
    if(ov('ov-vol')&&vol&&vol.regime==='expansion'&&vol.run_start_bar!=null){
      const a=Math.max(vol.run_start_bar,lo),b=hi;
      if(b>a){ctx.fillStyle='rgba(239,83,80,0.06)';ctx.fillRect(X(a)-W/(2*n2),0,(b-a)*W/n2,priceH);}}
    if(ov('ov-srange')&&structure.session_range&&structure.session_range.prior){
      const r=structure.session_range.prior;
      ctx.fillStyle='rgba(37,99,235,0.18)';ctx.fillRect(0,Y(r.top),W,Y(r.bottom)-Y(r.top));
      lab(r.label+' '+fmt(r.bottom)+' – '+fmt(r.top),Y(r.top)+14,'#1a5fd7',true);}
    if(ov('ov-fvg')){structure.fvgs.forEach(g=>{
      const a=Math.max(g.created_bar,lo),b=Math.min(g.mitigated_bar!=null?g.mitigated_bar:hi-1,hi-1);
      if(b<a)return;
      ctx.fillStyle=g.direction>0?'rgba(26,95,215,0.16)':'rgba(17,17,17,0.16)';
      ctx.fillRect(X(a)-W/(2*n2),Y(g.top),(b-a+1)*W/n2,Y(g.bottom)-Y(g.top));});}
    if(ov('ov-slevels')){structure.session_levels.forEach(l=>{
      ctx.strokeStyle='#1a5fd7';ctx.lineWidth=1;ctx.setLineDash([6,3]);
      ctx.beginPath();ctx.moveTo(0,Y(l.high));ctx.lineTo(W,Y(l.high));ctx.stroke();
      ctx.strokeStyle='#555555';
      ctx.beginPath();ctx.moveTo(0,Y(l.low));ctx.lineTo(W,Y(l.low));ctx.stroke();ctx.setLineDash([]);
      lab(l.label_high+' '+fmt(l.high),Y(l.high)-3,TXT);
      lab(l.label_low+' '+fmt(l.low),Y(l.low)+13,TXT);});}
    if(ov('ov-swings')){structure.swings.forEach(s=>{
      const x=s.bar;if(x<lo||x>=hi)return;
      const above=s.kind==='high';
      ctx.fillStyle=TXT;ctx.beginPath();ctx.arc(X(x),Y(s.price),3,0,7);ctx.fill();
      ctx.font='11px monospace';ctx.fillText(s.label,X(x)-8,Y(s.price)+(above?-8:16));});}
    if(ov('ov-breaks')){structure.breaks.forEach(b=>{
      const x=b.bar;if(x<lo||x>=hi)return;
      const up=b.direction>0;
      ctx.fillStyle=b.kind==='failed_break'?'#d32f2f':'#2e7d32';
      ctx.beginPath();
      if(up){ctx.moveTo(X(x),Y(b.level)-7);ctx.lineTo(X(x)-6,Y(b.level)+4);ctx.lineTo(X(x)+6,Y(b.level)+4);}
      else{ctx.moveTo(X(x),Y(b.level)+7);ctx.lineTo(X(x)-6,Y(b.level)-4);ctx.lineTo(X(x)+6,Y(b.level)-4);}
      ctx.closePath();ctx.fill();
      if(b.kind==='failed_break'){ctx.font='bold 10px monospace';ctx.fillText('✕',X(x)+8,Y(b.level)+(up?-4:12));}});}
    if(ov('ov-liq')){const seenSw=[];structure.liquidity.forEach(s=>{
      const x=s.bar;if(x<lo||x>=hi)return;
      ctx.fillStyle='#e08a00';ctx.font='11px monospace';
      const txt='◉ sweep '+fmt(s.level);
      const px=Math.min(X(x)+6,W-ctx.measureText(txt).width-4);
      const py=Y(s.level)-6;
      if(seenSw.some(q=>Math.abs(q[0]-px)<70&&Math.abs(q[1]-py)<14))return;
      seenSw.push([px,py]);
      ctx.fillText(txt,px,py);});}
    if(ov('ov-path')&&structure.expected_path){
      const p=structure.expected_path;
      const pts=p.points.map(q=>({x:q.bar!=null?X(q.bar):X(hi-1)+3*(W/n2),y:Y(q.price),proj:q.projected}));
      ctx.strokeStyle='#111111';ctx.lineWidth=1.6;ctx.beginPath();
      pts.forEach((q,i)=>{if(i===0)ctx.moveTo(q.x,q.y);else ctx.lineTo(q.x,q.y);});
      ctx.stroke();
      const last=pts[pts.length-1];
      ctx.beginPath();ctx.moveTo(last.x,last.y);ctx.lineTo(last.x-9,last.y-4);ctx.lineTo(last.x-9,last.y+4);ctx.closePath();
      ctx.fillStyle='#111111';ctx.fill();
      ctx.fillStyle='#2e7d32';ctx.font='bold 13px monospace';
      ctx.fillText('$',Math.min(last.x+4,W-20),last.y-6);
      ctx.fillStyle=TXT;ctx.font='11px monospace';
      ctx.fillText(p.target_label+' '+fmt(p.target_price),Math.max(8,Math.min(last.x-140,W-170)),last.y+16);}}
  if(hasVol){let vmax=0;for(let i=lo;i<hi;i++)vmax=Math.max(vmax,bars[i][5]||0);
    const vTop=priceH+6,vH=H-vTop-4;
    for(let i=lo;i<hi;i++){const b=bars[i];if(!b[5]||!vmax)continue;
      ctx.fillStyle=b[4]>=b[1]?UP:DN;ctx.globalAlpha=0.45;
      const h=Math.max(1,(b[5]/vmax)*vH);
      ctx.fillRect(X(i)-bw/2,vTop+vH-h,bw,h);ctx.globalAlpha=1;}}
  if(hover>=lo&&hover<hi){const b=bars[hover];
    ctx.strokeStyle='#fff';ctx.lineWidth=0.8;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(X(hover),0);ctx.lineTo(X(hover),priceH);ctx.stroke();
    ctx.beginPath();ctx.moveTo(0,Y(b[4]));ctx.lineTo(W,Y(b[4]));ctx.stroke();ctx.setLineDash([]);
    document.getElementById('readout').textContent=
      b[0].slice(0,16).replace('T',' ')+'  O '+fmt(b[1])+'  H '+fmt(b[2])+'  L '+fmt(b[3])+'  C '+fmt(b[4]);}
  else{document.getElementById('readout').textContent='Hover a candle for its values. Scroll to zoom, drag to pan, double-click to reset.';}}
function barAt(e){const r=cv.getBoundingClientRect();const x=(e.clientX-r.left)/r.width;
  return Math.floor(start+x*xDiv);}
cv.addEventListener('mousemove',e=>{hover=barAt(e);draw();});
cv.addEventListener('mouseleave',()=>{hover=-1;draw();});
cv.addEventListener('wheel',e=>{e.preventDefault();const c=barAt(e);const n=end-start;
  const f=e.deltaY<0?0.8:1.25;let nn=Math.max(10,Math.round(n*f));
  let ns=Math.round(c-(c-start)*nn/n);ns=Math.max(0,Math.min(bars.length-nn,ns));
  start=ns;end=ns+nn;draw();},{passive:false});
cv.addEventListener('mousedown',e=>{dragX=e.clientX;});
window.addEventListener('mousemove',e=>{if(dragX===null)return;
  const r=cv.getBoundingClientRect();const dx=e.clientX-dragX;dragX=e.clientX;
  const db=Math.round(dx/r.width*(end-start));let ns=start-db;
  ns=Math.max(0,Math.min(bars.length-(end-start),ns));start=ns;end=ns+(end-start);draw();});
window.addEventListener('mouseup',()=>{dragX=null;});
cv.addEventListener('dblclick',()=>{resetWin();draw();});
function showFresh(){const el=document.getElementById('fresh');
  if(!status){el.textContent='Connecting…';return;}
  const s=status.tfs[tf];
  let hist='';
  if(before){hist=' · viewing up to '+before.replace('T',' ')+' (historical)';}
  if(!status.market_open){el.innerHTML='MARKET CLOSED · last '+tf+' bar '+(s&&s.last_bar?s.last_bar:'unknown')+hist;el.className='';}
  else if(!s){el.textContent='No '+tf+' bars from the engine.'+hist;}
  else if(s.stale){el.innerHTML='<span class="stale">STALE · '+tf+' last bar '+s.last_bar+' ('+Math.round(s.age_s)+'s ago)</span>'+hist;}
  else{el.innerHTML='<span class="live">LIVE · '+tf+' last bar '+s.last_bar+' ('+Math.round(s.age_s)+'s ago)</span>'+hist;}
  const pv=document.getElementById('prov');
  if(studies&&studies.signal){pv.textContent='Studies from latest signal '+studies.signal+' at '+studies.signal_time+'.';}
  else{pv.textContent='No signal studies yet — candles + sessions only.';}
  const ms=document.getElementById('missing');
  if(structure&&structure.missing&&structure.missing.length){
    ms.textContent='Not measured: '+structure.missing.map(m=>m.what+' ('+m.reason+')').join('; ')+'.';}
  else{ms.textContent='';}
  const hist_el=document.getElementById('hist');
  if(before){hist_el.textContent='Historical view — bars up to '+before.replace('T',' ')+'. Structure measured on those bars only.';}
  else{hist_el.textContent='';}}
function showMtf(){const el=document.getElementById('mtf');if(!mtf)return;
  el.innerHTML='';
  Object.keys(mtf.tfs).forEach(t=>{
    const s=mtf.tfs[t];const chip=document.createElement('span');
    chip.className='chip'+(t===tf?' on':'');
    if(!s.available){chip.innerHTML='<span class="t">'+t+'</span> · unavailable';}
    else{chip.innerHTML='<span class="t">'+t+'</span> · '+s.trend+' · '+s.regime;}
    chip.addEventListener('click',()=>loadTF(t,false));
    el.appendChild(chip);});}
async function loadTF(t,keepWin){tf=t;
  document.querySelectorAll('button.tf').forEach(b=>b.classList.toggle('on',b.dataset.tf===t));
  try{let url='/api/frames?tf='+t;
    if(before)url+='&before='+encodeURIComponent(before);
    const r=await fetch(url);
    if(!r.ok){document.getElementById('fresh').textContent='Engine returned '+r.status+' for '+t+'.';return;}
    const d=await r.json();bars=d.bars;sessions=d.sessions;studies=d.studies;structure=d.structure;
    if(!keepWin)resetWin();draw();showFresh();}
  catch(e){document.getElementById('fresh').textContent='Cannot reach the engine: '+e;}}
async function poll(){try{const r=await fetch('/api/status');
    if(r.ok){status=await r.json();showFresh();}}
  catch(e){}
  try{const r=await fetch('/api/mtf');
    if(r.ok){mtf=await r.json();showMtf();}}
  catch(e){}
  loadTF(tf,true);}
document.querySelectorAll('button.tf').forEach(b=>b.addEventListener('click',()=>loadTF(b.dataset.tf,false)));
['ov-sess','ov-nearby','ov-fib','ov-prem','ov-zone','ov-slevels','ov-srange','ov-swings','ov-breaks','ov-fvg','ov-liq','ov-vol','ov-path'].forEach(id=>
  document.getElementById(id).addEventListener('change',draw));
document.getElementById('theme-btn').addEventListener('click',()=>
  setTheme(document.body.dataset.theme==='dark'?'light':'dark'));
function shiftDay(days){if(!before){const last=bars.length?bars[bars.length-1][0]:null;
    if(!last)return;before=last.slice(0,16);}
  const d=new Date(before.replace(' ','T'));d.setDate(d.getDate()+days);
  before=d.toISOString().slice(0,16);
  document.getElementById('jump').value=before;loadTF(tf,false);}
document.getElementById('jump-go').addEventListener('click',()=>{
  const v=document.getElementById('jump').value;
  if(v){before=v;loadTF(tf,false);}});
document.getElementById('jump-prev').addEventListener('click',()=>shiftDay(-1));
document.getElementById('jump-next').addEventListener('click',()=>shiftDay(1));
document.getElementById('jump-latest').addEventListener('click',()=>{
  before=null;document.getElementById('jump').value='';loadTF(tf,false);});
loadTF('H1',false);poll();setInterval(poll,30000);
</script></body></html>
"""


def build_terminal_page(*, instrument: str = "XAUUSD") -> str:
    """The terminal page with the instrument baked into the title."""
    return TERMINAL_PAGE.replace("__INSTRUMENT__", instrument)
