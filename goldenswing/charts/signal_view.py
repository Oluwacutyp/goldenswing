"""Per-signal HTML terminal view: one self-contained page per signal.

Baked at signal time: per-timeframe bars (capped), per-timeframe
analysis lines, trade levels, the signal zone, nearby cache levels,
auto-Fib + OTE + dealing range, and killzone session spans. The page
needs no server and no network: open it in a browser, switch
timeframes, toggle overlays, zoom/pan, crosshair. No CDN, no external
assets — the page must open on a phone with poor connectivity.

Analysis is FROZEN at signal time: the record of what the engine saw
when it flagged the signal, not a live view. A live-updating
current-market view is a separate step, not this page.

Nothing here is invented: a timeframe with no verdict read says so
plainly; a missing study is left off, never defaulted.
"""
from __future__ import annotations

import html as _html
import json
import math

from ..reader import session_timeframes as sessions  # Step 3: DST-aware authority (was fixed-UTC sessions)
from .signal_model import (  # Step 7: one shared signal-view model
    effective_r,
    panel_role,
    score_text,
)

TF_ORDER = ("D1", "H4", "H1", "M30", "M15", "M5")
_BARS_CAP = 500


def _json_for_html(obj) -> str:
    """JSON safe to embed in a <script> block: <, >, & escaped so a
    payload string can never break out of the block."""
    return (json.dumps(obj)
            .replace("&", "\\u0026")
            .replace("<", "\\u003c")
            .replace(">", "\\u003e"))


def _fmt(x: float) -> str:
    return f"{x:,.1f}"


def _bars_for_json(frame) -> list:
    rows = []
    for ts, row in frame.iloc[-_BARS_CAP:].iterrows():
        try:
            iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            rows.append([iso, float(row["open"]), float(row["high"]),
                         float(row["low"]), float(row["close"])])
        except Exception:
            continue
    return rows


def _session_spans(frame) -> list:
    """Killzone runs over the capped window, as [start_i, end_i]."""
    view = frame.iloc[-_BARS_CAP:]
    spans = []
    run_start = None
    n = len(view)
    for i, ts in enumerate(view.index):
        try:
            _name, kz = sessions.session_of(ts)
        except Exception:
            kz = None
        if kz and run_start is None:
            run_start = i
        if (not kz or i == n - 1) and run_start is not None:
            run_end = i if not kz else i + 1
            spans.append([run_start, run_end])
            run_start = None
    return spans


def _analysis_for_tf(tf: str, *, verdict, entry_clock, zone,
                     fib_info, fib_range, entry, stop, targets,
                     direction, broker_stop=None,
                     eff_r=None) -> list[str]:
    """Analysis lines for one timeframe. Missing reads are stated,
    never filled in."""
    v = verdict or {}
    lines: list[str] = []
    if tf == "D1":
        bias = v.get("bias_d1")
        if bias is None:
            lines.append("No D1 read recorded.")
        else:
            lines.append(f"D1 bias {int(bias):+d} — context only, "
                         "never a veto on the trade.")
    elif tf == "H4":
        bias = v.get("bias_h4")
        if bias is None:
            lines.append("No H4 read recorded.")
        else:
            lines.append(f"H4 bias {int(bias):+d} — structure timeframe.")
        if zone and zone.get("top") and zone.get("bottom"):
            lines.append(f"Zone {zone.get('kind', '')} "
                         f"{_fmt(float(zone['bottom']))}–"
                         f"{_fmt(float(zone['top']))}.")
    elif tf == "H1":
        trigger = v.get("trigger")
        if trigger:
            lines.append(f"Trigger: {trigger}.")
        if fib_range is not None:
            lo, hi = fib_range
            eq = 0.5 * (lo + hi)
            pos = ("premium" if entry > eq else "discount"
                   if entry < eq else "equilibrium")
            lines.append(f"Dealing range {_fmt(lo)}–{_fmt(hi)} "
                         f"(EQ {_fmt(eq)}) — entry sits in {pos}.")
        if fib_info:
            nf = fib_info.get("nearest_frac")
            if nf is not None:
                lines.append(f"Nearest Fib retracement: {nf:.3f}.")
            if fib_info.get("at_fib"):
                lines.append("Price is at a Fib level (within tolerance).")
            if fib_info.get("in_ote_long"):
                lines.append("Price is in the OTE band (long entry zone).")
            if fib_info.get("in_ote_short"):
                lines.append("Price is in the OTE band (short entry zone).")
        if not lines:
            lines.append("No H1 read recorded.")
    elif tf == entry_clock:
        arrow = "▲ LONG" if direction == 1 else "▼ SHORT" if direction == -1 else "FLAT"
        lines.append(f"Entry timeframe {tf}: {arrow}.")
        stop_line = (f"Entry {_fmt(entry)} · "
                     f"Structural stop {_fmt(stop)}")
        if broker_stop is not None:
            stop_line += f" · Broker SL {_fmt(broker_stop)}"
            if eff_r is not None:
                stop_line += f" (effective R {eff_r:.2f})"
        else:
            stop_line += " · Broker SL pending at placement"
        lines.append(stop_line)
        if targets:
            lines.append(f"Targets {', '.join(_fmt(t) for t in targets)}.")
    else:
        lines.append("Bars + studies only — the verdict names no read "
                     "for this timeframe.")
    return lines


def build_signal_view_html(*, frames, entry: float, stop: float,
                           targets, direction: int, playbook: str,
                           score: float | None = None,
                           decision_score: float | None = None,
                           broker_stop: float | None = None,
                           verdict, zone,
                           nearby_zones, fib_levels, fib_range,
                           ote_band, fib_info, alert_id: str,
                           instrument: str = "XAUUSD",
                           fill_price: float | None = None,
                           signal_time: str | None = None) -> str:
    """Render the self-contained signal view page (HTML string).

    Step 7: the page shows the DECISION score (``decision_score``;
    ``score`` is only the legacy fallback for old callers) and BOTH
    stops — structural invalidation and broker SL when one is known.
    No score at all renders as "Score unavailable"; no broker stop
    yet renders as "pending at placement". Nothing is defaulted.

    Raises ValueError when no timeframe has bars — the caller turns
    that into 'no view', never into an empty page.
    """
    tfs = [tf for tf in TF_ORDER if frames.get(tf) is not None
           and len(frames[tf]) > 0]
    if not tfs:
        raise ValueError("signal view needs at least one frame")
    v = verdict or {}
    entry_clock = v.get("entry_clock") or "M15"
    shown_score = (decision_score if decision_score is not None
                   else score)
    eff_r = effective_r(entry=entry, broker_stop=broker_stop,
                        targets=targets)
    zone_d = None
    if zone and zone.get("top") and zone.get("bottom"):
        zone_d = {"top": float(zone["top"]), "bottom": float(zone["bottom"]),
                  "kind": str(zone.get("kind", ""))}
    header_lines = []
    if v.get("amd_phase") not in (None, "?"):
        header_lines.append(f"AMD phase: {v.get('amd_phase')}")
    qm = v.get("qm") or {}
    if isinstance(qm, dict) and qm.get("status"):
        header_lines.append(f"QM: {qm.get('status')}")
    vol = v.get("volume") or {}
    if isinstance(vol, dict) and vol.get("volume_ratio") is not None:
        header_lines.append(f"Volume ratio: {vol.get('volume_ratio')}")
    payload = {
        "meta": {
            "instrument": instrument, "playbook": playbook,
            "direction": int(direction),
            "score": float(shown_score) if shown_score is not None else None,
            "score_text": score_text(decision_score, fallback=score),
            "entry": float(entry), "stop": float(stop),
            "structural_stop": float(stop),
            "broker_stop": (float(broker_stop)
                            if broker_stop is not None else None),
            "effective_r": eff_r,
            "targets": [float(t) for t in (targets or [])],
            "fill": float(fill_price) if fill_price is not None else None,
            "alert_id": alert_id, "signal_time": signal_time or "",
            "verdict_lines": header_lines,
            "entry_clock": entry_clock,
        },
        "zone": zone_d,
        "nearby": [{"level": float(z["level"]),
                    "tier": str(z.get("tier", "")),
                    "side": str(z.get("side", ""))}
                   for z in (nearby_zones or [])],
        "fib": {
            "levels": [[float(f), float(p)] for f, p in (fib_levels or [])],
            "ote": ([float(ote_band[0]), float(ote_band[1])]
                    if ote_band is not None else None),
            "range": ([float(fib_range[0]), float(fib_range[1])]
                      if fib_range is not None else None),
        },
        "tfs": {
            tf: {
                "bars": _bars_for_json(frames[tf]),
                "sessions": _session_spans(frames[tf]),
                "role": panel_role(tf, playbook=playbook,
                                   entry_clock=entry_clock),
                "analysis": _analysis_for_tf(
                    tf, verdict=v, entry_clock=entry_clock, zone=zone,
                    fib_info=fib_info, fib_range=fib_range, entry=entry,
                    stop=stop, targets=targets, direction=direction,
                    broker_stop=broker_stop, eff_r=eff_r),
            }
            for tf in tfs
        },
    }
    buttons = "".join(
        f'<button class="tf" data-tf="{tf}">'
        f"{tf} · {panel_role(tf, playbook=playbook, entry_clock=entry_clock)}"
        f"</button>" for tf in tfs)
    data_json = _json_for_html(payload)
    return (_PAGE.replace("{buttons}", buttons)
                 .replace("{data}", data_json))


_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signal view</title>
<style>
body{background:var(--bg);color:var(--txt);font-family:system-ui,sans-serif;margin:0;padding:12px}
body[data-theme="dark"]{--bg:#11141a;--txt:#e0e0e0;--panel:#0b0e13;--line:#3a4356;--muted:#aaa;--gold:#f7b924;--btn:#1e2530}
body[data-theme="light"]{--bg:#f6f7f9;--txt:#1a1d23;--panel:#ffffff;--line:#c9ced6;--muted:#5a6472;--gold:#8a6d00;--btn:#e8ecf1}
h1{font-size:18px;margin:0 0 4px}h2{font-size:14px;margin:14px 0 4px;color:var(--gold)}
#meta{color:var(--muted);font-size:13px;margin-bottom:10px}
button.tf{background:var(--btn);color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:8px 14px;margin:0 6px 8px 0;font-size:15px}
button.tf.on{background:var(--gold);color:#111;border-color:var(--gold)}
#theme-btn{background:var(--btn);color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:8px 14px;font-size:15px;margin:0 0 8px 0}
#toggles{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
#toggles label{display:flex;align-items:center;gap:6px;min-height:44px;padding:0 12px;border:1px solid var(--line);border-radius:8px;font-size:14px}
#toggles input{width:18px;height:18px}
#wrap{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:8px}
canvas{display:block;width:100%;height:520px;touch-action:none}
#readout{font-family:monospace;font-size:13px;color:var(--gold);min-height:18px;margin:6px 0}
#analysis{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 14px;line-height:1.6}
#foot{color:var(--muted);font-size:12px;margin-top:10px}
</style></head><body data-theme="dark">
<h1 id="title">Signal view</h1><div id="meta"></div>
<div id="tfs">{buttons}</div>
<div><button id="theme-btn">Light theme</button></div>
<div id="toggles"><label><input type="checkbox" id="ov-zone" checked> Zone</label><label><input type="checkbox" id="ov-nearby" checked> Nearby levels</label><label><input type="checkbox" id="ov-fib" checked> Fib + OTE</label><label><input type="checkbox" id="ov-prem" checked> Premium/Discount</label><label><input type="checkbox" id="ov-sess" checked> Sessions</label><label><input type="checkbox" id="ov-levels" checked> Entry/SL/TP</label></div>
<div id="wrap"><canvas id="cv" width="900" height="520"></canvas></div>
<div id="readout">Hover a candle for its values. Scroll to zoom, drag to pan, double-click to reset.</div>
<h2>Analysis on this timeframe</h2><div id="analysis"></div>
<div id="foot">Frozen at signal time — the record of what the engine saw when it flagged this signal, not a live view.</div>
<script id="signal-data" type="application/json">{data}</script>
<script>
const D=JSON.parse(document.getElementById('signal-data').textContent);
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const PALETTES={dark:{UP:'#26a69a',DN:'#ef5350',GOLD:'#f7b924',TXT:'#e0e0e0',GRID:'#2a3242'},
  light:{UP:'#00897b',DN:'#d32f2f',GOLD:'#8a6d00',TXT:'#1a1d23',GRID:'#d5dae2'}};
let PAL=PALETTES.dark;
let UP=PAL.UP,DN=PAL.DN,GOLD=PAL.GOLD,TXT=PAL.TXT,GRID=PAL.GRID;
function setTheme(name){PAL=PALETTES[name];UP=PAL.UP;DN=PAL.DN;GOLD=PAL.GOLD;TXT=PAL.TXT;GRID=PAL.GRID;
  document.body.dataset.theme=name;
  document.getElementById('theme-btn').textContent=name==='dark'?'Light theme':'Dark theme';draw();}
let tf=(D.meta.entry_clock&&D.tfs[D.meta.entry_clock])?D.meta.entry_clock:(D.tfs.H1?'H1':Object.keys(D.tfs)[0]);
let start=0,end=0,hover=-1,dragX=null;
function bars(){return D.tfs[tf].bars}
function resetWin(){const n=bars().length;end=n;start=Math.max(0,n-120)}
function ov(id){return document.getElementById(id).checked}
function fmt(x){return x.toLocaleString(undefined,{minimumFractionDigits:1,maximumFractionDigits:1})}
function showTF(t){tf=t;resetWin();document.querySelectorAll('button.tf').forEach(b=>b.classList.toggle('on',b.dataset.tf===t));
  const an=document.getElementById('analysis');an.innerHTML='';
  D.tfs[t].analysis.forEach(l=>{const d=document.createElement('div');d.textContent=l;an.appendChild(d)});
  draw();}
function yRange(bs,lo,hi){let mn=Infinity,mx=-Infinity;
  for(let i=lo;i<hi;i++){mn=Math.min(mn,bs[i][3]);mx=Math.max(mx,bs[i][2])}
  const span=Math.max(mx-mn,1e-9);
  [D.meta.entry,D.meta.stop].concat(D.meta.broker_stop===null?[]:[D.meta.broker_stop]).concat(D.meta.targets).concat(D.zone?[D.zone.top,D.zone.bottom]:[]).forEach(v=>{
    if(v>=mn-1.6*span&&v<=mx+1.6*span){mn=Math.min(mn,v);mx=Math.max(mx,v)}});
  const pad=0.07*(mx-mn);return [mn-pad,mx+pad];}
function draw(){const bs=bars();if(!bs.length)return;
  const W=cv.clientWidth,H=cv.height;cv.width=W;
  const lo=start,hi=end,n=hi-lo;if(n<=0)return;
  const [mn,mx]=yRange(bs,lo,hi);
  const X=i=>((i-lo+0.5)/n)*W, Y=v=>H-((v-mn)/(mx-mn))*H;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle=GRID;ctx.lineWidth=0.5;
  for(let g=0;g<=4;g++){const y=H*g/4;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();
    ctx.fillStyle='#777';ctx.font='11px monospace';ctx.fillText(fmt(mx-(mx-mn)*g/4),4,y-2);}
  if(ov('ov-sess')){ctx.fillStyle='rgba(247,185,36,0.07)';
    D.tfs[tf].sessions.forEach(s=>{const a=Math.max(s[0],lo),b=Math.min(s[1],hi);
      if(b>a)ctx.fillRect(X(a)-W/(2*n),0,(b-a)*W/n,H);});}
  if(ov('ov-prem')&&D.fib.range){const rlo=D.fib.range[0],rhi=D.fib.range[1],eq=(rlo+rhi)/2;
    ctx.fillStyle='rgba(239,83,80,0.05)';ctx.fillRect(0,Y(rhi),W,Y(eq)-Y(rhi));
    ctx.fillStyle='rgba(38,166,154,0.05)';ctx.fillRect(0,Y(eq),W,Y(rlo)-Y(eq));
    ctx.strokeStyle=TXT;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(0,Y(eq));ctx.lineTo(W,Y(eq));ctx.stroke();ctx.setLineDash([]);}
  if(ov('ov-zone')&&D.zone){ctx.fillStyle='rgba(247,185,36,0.25)';
    ctx.fillRect(0,Y(D.zone.top),W,Y(D.zone.bottom)-Y(D.zone.top));}
  if(ov('ov-nearby')){D.nearby.forEach(z=>{ctx.strokeStyle=z.side==='support'?UP:z.side==='resistance'?DN:TXT;
    ctx.lineWidth=z.tier==='MAJOR'?1.2:0.7;ctx.setLineDash(z.tier==='MAJOR'?[]:[6,3]);
    ctx.beginPath();ctx.moveTo(0,Y(z.level));ctx.lineTo(W,Y(z.level));ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=ctx.strokeStyle;ctx.font='11px monospace';
    ctx.fillText((z.tier?z.tier+' ':'')+fmt(z.level),W-110,Y(z.level)-2);});}
  if(ov('ov-fib')){D.fib.levels.forEach(l=>{ctx.strokeStyle='#8d6e00';ctx.lineWidth=0.7;ctx.setLineDash([4,3]);
    ctx.beginPath();ctx.moveTo(0,Y(l[1]));ctx.lineTo(W,Y(l[1]));ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='#8d6e00';ctx.font='11px monospace';ctx.fillText(l[0].toFixed(3),W-70,Y(l[1])-2);});
    if(D.fib.ote){ctx.fillStyle='rgba(141,110,0,0.12)';ctx.fillRect(0,Y(D.fib.ote[1]),W,Y(D.fib.ote[0])-Y(D.fib.ote[1]));}}
  const bw=Math.max(2,(W/n)*0.7);
  for(let i=lo;i<hi;i++){const b=bs[i],up=b[4]>=b[1],col=up?UP:DN;
    ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(X(i),Y(b[2]));ctx.lineTo(X(i),Y(b[3]));ctx.stroke();
    const t=Y(Math.max(b[1],b[4])),bt=Y(Math.min(b[1],b[4]));
    ctx.fillRect(X(i)-bw/2,t,bw,Math.max(bt-t,1));}
  if(ov('ov-levels')){
    const rows=[];
    const push=(v,col,w,dash,lab)=>{if(v===null||v===undefined)return;rows.push({v,col,w,dash,lab});};
    push(D.meta.entry,GOLD,1.6,[],'ENTRY '+fmt(D.meta.entry));
    push(D.meta.stop,'#d32f2f',1.6,[],'STOP '+fmt(D.meta.stop)+' (structural)');
    push(D.meta.broker_stop,'#ff7043',1.6,[],'BROKER SL '+fmt(D.meta.broker_stop));
    D.meta.targets.forEach((t,j)=>push(t,'#2e7d32',1.1,[6,3],'T'+(j+1)+' '+fmt(t)));
    push(D.meta.fill,'#7e57c2',1.4,[8,3],'FILL '+fmt(D.meta.fill));
    rows.forEach(r=>{r.y=Y(r.v);});
    rows.sort((a,b)=>a.y-b.y);
    for(let i=1;i<rows.length;i++){if(rows[i].y<rows[i-1].y+16)rows[i].y=rows[i-1].y+16;}
    rows.forEach(r=>{
      const yTrue=Y(r.v);
      ctx.strokeStyle=r.col;ctx.lineWidth=r.w;ctx.setLineDash(r.dash);
      ctx.beginPath();ctx.moveTo(0,yTrue);ctx.lineTo(W,yTrue);ctx.stroke();ctx.setLineDash([]);
      if(Math.abs(r.y-yTrue)>1){ctx.strokeStyle=r.col;ctx.lineWidth=0.7;
        ctx.beginPath();ctx.moveTo(W-150,yTrue);ctx.lineTo(W-150,r.y);ctx.stroke();}
      ctx.fillStyle=r.col;ctx.font='bold 12px monospace';
      ctx.fillText(r.lab,W-150,r.y-3);});}
  if(hover>=lo&&hover<hi){const b=bs[hover];
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
  let ns=Math.round(c-(c-start)*nn/n);ns=Math.max(0,Math.min(bsLen()-nn,ns));
  start=ns;end=ns+nn;draw();},{passive:false});
function bsLen(){return bars().length}
cv.addEventListener('mousedown',e=>{dragX=e.clientX;});
window.addEventListener('mousemove',e=>{if(dragX===null)return;
  const r=cv.getBoundingClientRect();const dx=e.clientX-dragX;dragX=e.clientX;
  const db=Math.round(dx/r.width*(end-start));let ns=start-db;
  ns=Math.max(0,Math.min(bsLen()-(end-start),ns));start=ns;end=ns+(end-start);draw();});
window.addEventListener('mouseup',()=>{dragX=null;});
cv.addEventListener('dblclick',()=>{resetWin();draw();});
document.querySelectorAll('button.tf').forEach(b=>b.addEventListener('click',()=>showTF(b.dataset.tf)));
['ov-zone','ov-nearby','ov-fib','ov-prem','ov-sess','ov-levels'].forEach(id=>
  document.getElementById(id).addEventListener('change',draw));
document.getElementById('theme-btn').addEventListener('click',()=>
  setTheme(document.body.dataset.theme==='dark'?'light':'dark'));
(function(){const m=D.meta,arrow=m.direction===1?'▲ LONG':m.direction===-1?'▼ SHORT':'FLAT';
  document.getElementById('title').textContent=m.instrument+' · Signal view · '+m.playbook+' · '+arrow;
  document.getElementById('meta').textContent=m.score_text+' · Entry '+fmt(m.entry)+' · Structural stop '+fmt(m.structural_stop)+
    (m.broker_stop===null?' · Broker SL pending at placement':' · Broker SL '+fmt(m.broker_stop)+
      (m.effective_r===null?'':' (effective R '+m.effective_r.toFixed(2)+')'))+
    (m.targets.length?' · Targets '+m.targets.map(fmt).join(', '):'')+(m.alert_id?' · '+m.alert_id:'')+
    (m.verdict_lines.length?' · '+m.verdict_lines.join(' · '):'');
  resetWin();showTF(tf);})();
</script></body></html>
"""
