"""Interactive terminal page: live candles from the engine's stack.

Served by ``terminal_server`` (same origin, ``/api`` fetches only —
no CDN, no external assets). The page itself holds no market data:
every TF switch fetches that timeframe's bars, and a 30s poll
refreshes the current timeframe + the freshness banner.

Studies provenance, stated on the page: candles + session shading
come from the live stack; nearby levels, zone band, Fib+OTE and
premium/discount come from the LATEST SIGNAL's studies (served by
the engine, labeled with that signal's id and time). No signal yet
means those toggles show nothing — never invented levels.

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
button.tf,button#theme-btn{background:var(--btn);color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:8px 14px;margin:0 6px 8px 0;font-size:15px}
button.tf.on{background:var(--gold);color:#111;border-color:var(--gold)}
#toggles{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
#toggles label{display:flex;align-items:center;gap:6px;min-height:44px;padding:0 12px;border:1px solid var(--line);border-radius:8px;font-size:14px}
#toggles input{width:18px;height:18px}
#wrap{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:8px}
canvas{display:block;width:100%;height:520px;touch-action:none}
#readout{font-family:monospace;font-size:13px;color:var(--gold);min-height:18px;margin:6px 0}
#foot{color:var(--muted);font-size:12px;margin-top:10px}
</style></head><body data-theme="dark">
<h1 id="title">GoldenSwing terminal · __INSTRUMENT__</h1>
<div id="fresh">Connecting…</div>
<div id="prov"></div>
<div id="tfs"><button class="tf" data-tf="M5">M5</button><button class="tf" data-tf="M15">M15</button><button class="tf" data-tf="M30">M30</button><button class="tf" data-tf="H1">H1</button><button class="tf" data-tf="H4">H4</button><button class="tf" data-tf="D1">D1</button></div>
<div><button id="theme-btn">Light theme</button></div>
<div id="toggles"><label><input type="checkbox" id="ov-sess" checked> Sessions</label><label><input type="checkbox" id="ov-nearby" checked> Nearby levels</label><label><input type="checkbox" id="ov-fib" checked> Fib + OTE</label><label><input type="checkbox" id="ov-prem" checked> Premium/Discount</label><label><input type="checkbox" id="ov-zone" checked> Signal zone</label></div>
<div id="wrap"><canvas id="cv" width="900" height="520"></canvas></div>
<div id="readout">Hover a candle for its values. Scroll to zoom, drag to pan, double-click to reset.</div>
<div id="foot">Read-only view — it never trades. Orders stay on the engine / Telegram path. Polls every 30s; a stale feed is labeled stale, never shown as live.</div>
<script>
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const PALETTES={dark:{UP:'#26a69a',DN:'#ef5350',GOLD:'#f7b924',TXT:'#e0e0e0',GRID:'#2a3242'},
  light:{UP:'#00897b',DN:'#d32f2f',GOLD:'#8a6d00',TXT:'#1a1d23',GRID:'#d5dae2'}};
let PAL=PALETTES.dark;
let UP=PAL.UP,DN=PAL.DN,GOLD=PAL.GOLD,TXT=PAL.TXT,GRID=PAL.GRID;
function setTheme(name){PAL=PALETTES[name];UP=PAL.UP;DN=PAL.DN;GOLD=PAL.GOLD;TXT=PAL.TXT;GRID=PAL.GRID;
  document.body.dataset.theme=name;
  document.getElementById('theme-btn').textContent=name==='dark'?'Light theme':'Dark theme';draw();}
let tf='H1',bars=[],sessions=[],studies=null,status=null;
let start=0,end=0,hover=-1,dragX=null;
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
  extra.forEach(v=>{if(v>=mn-1.6*span&&v<=mx+1.6*span){mn=Math.min(mn,v);mx=Math.max(mx,v)}});
  const pad=0.07*(mx-mn);return [mn-pad,mx+pad];}
function draw(){if(!bars.length){ctx.clearRect(0,0,cv.clientWidth,cv.height);return;}
  const W=cv.clientWidth,H=cv.height;cv.width=W;
  const lo=start,hi=end,n=hi-lo;if(n<=0)return;
  const [mn,mx]=yRange(bars,lo,hi);
  const X=i=>((i-lo+0.5)/n)*W, Y=v=>H-((v-mn)/(mx-mn))*H;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle=GRID;ctx.lineWidth=0.5;
  for(let g=0;g<=4;g++){const y=H*g/4;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();
    ctx.fillStyle='#777';ctx.font='11px monospace';ctx.fillText(fmt(mx-(mx-mn)*g/4),4,y-2);}
  if(ov('ov-sess')){ctx.fillStyle='rgba(247,185,36,0.07)';
    sessions.forEach(s=>{const a=Math.max(s[0],lo),b=Math.min(s[1],hi);
      if(b>a)ctx.fillRect(X(a)-W/(2*n),0,(b-a)*W/n,H);});}
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
  const bw=Math.max(2,(W/n)*0.7);
  for(let i=lo;i<hi;i++){const b=bars[i],up=b[4]>=b[1],col=up?UP:DN;
    ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(X(i),Y(b[2]));ctx.lineTo(X(i),Y(b[3]));ctx.stroke();
    const t=Y(Math.max(b[1],b[4])),bt=Y(Math.min(b[1],b[4]));
    ctx.fillRect(X(i)-bw/2,t,bw,Math.max(bt-t,1));}
  if(hover>=lo&&hover<hi){const b=bars[hover];
    ctx.strokeStyle='#fff';ctx.lineWidth=0.8;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(X(hover),0);ctx.lineTo(X(hover),H);ctx.stroke();
    ctx.beginPath();ctx.moveTo(0,Y(b[4]));ctx.lineTo(W,Y(b[4]));ctx.stroke();ctx.setLineDash([]);
    document.getElementById('readout').textContent=
      b[0].slice(0,16).replace('T',' ')+'  O '+fmt(b[1])+'  H '+fmt(b[2])+'  L '+fmt(b[3])+'  C '+fmt(b[4]);}
  else{document.getElementById('readout').textContent='Hover a candle for its values. Scroll to zoom, drag to pan, double-click to reset.';}}
function barAt(e){const r=cv.getBoundingClientRect();const x=(e.clientX-r.left)/r.width;
  return Math.floor(start+x*(end-start));}
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
  if(!status.market_open){el.innerHTML='MARKET CLOSED · last '+tf+' bar '+(s&&s.last_bar?s.last_bar:'unknown');el.className='';}
  else if(!s){el.textContent='No '+tf+' bars from the engine.';}
  else if(s.stale){el.innerHTML='<span class="stale">STALE · '+tf+' last bar '+s.last_bar+' ('+Math.round(s.age_s)+'s ago)</span>';}
  else{el.innerHTML='<span class="live">LIVE · '+tf+' last bar '+s.last_bar+' ('+Math.round(s.age_s)+'s ago)</span>';}
  const pv=document.getElementById('prov');
  if(studies&&studies.signal){pv.textContent='Studies from latest signal '+studies.signal+' at '+studies.signal_time+'.';}
  else{pv.textContent='No signal studies yet — candles + sessions only.';}}
async function loadTF(t,keepWin){tf=t;
  document.querySelectorAll('button.tf').forEach(b=>b.classList.toggle('on',b.dataset.tf===t));
  try{const r=await fetch('/api/frames?tf='+t);
    if(!r.ok){document.getElementById('fresh').textContent='Engine returned '+r.status+' for '+t+'.';return;}
    const d=await r.json();bars=d.bars;sessions=d.sessions;studies=d.studies;
    if(!keepWin)resetWin();draw();showFresh();}
  catch(e){document.getElementById('fresh').textContent='Cannot reach the engine: '+e;}}
async function poll(){try{const r=await fetch('/api/status');
    if(r.ok){status=await r.json();showFresh();}}
  catch(e){}
  loadTF(tf,true);}
document.querySelectorAll('button.tf').forEach(b=>b.addEventListener('click',()=>loadTF(b.dataset.tf,false)));
['ov-sess','ov-nearby','ov-fib','ov-prem','ov-zone'].forEach(id=>
  document.getElementById(id).addEventListener('change',draw));
document.getElementById('theme-btn').addEventListener('click',()=>
  setTheme(document.body.dataset.theme==='dark'?'light':'dark'));
loadTF('H1',false);poll();setInterval(poll,30000);
</script></body></html>
"""


def build_terminal_page(*, instrument: str = "XAUUSD") -> str:
    """The terminal page with the instrument baked into the title."""
    return TERMINAL_PAGE.replace("__INSTRUMENT__", instrument)
