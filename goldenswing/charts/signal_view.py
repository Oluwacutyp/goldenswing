"""Per-signal HTML terminal view: one self-contained page per signal.

Baked at signal time: per-timeframe bars (capped), per-timeframe
analysis lines, trade levels, the signal zone, nearby cache levels,
auto-Fib + OTE + dealing range, and killzone session spans. The page
needs no server and no network: open it in a browser, switch
timeframes, toggle overlays, zoom/pan, crosshair. No CDN, no external
assets — the page must open on a phone with poor connectivity.

Rendering is TradingView Lightweight Charts™ via the ONE shared
page builder (``charts.tv_page``) — the same builder that produces
the alert PNGs, so the study and the PNG can never drift apart.

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
    chart_targets,
    effective_r,
    panel_role,
    score_text,
)

TF_ORDER = ("D1", "H4", "H1", "M30", "M15", "M5", "M1")
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
            vol = None
            if "volume" in frame.columns:
                v = float(row["volume"])
                vol = v if v == v and v > 0 else None
            rows.append([iso, float(row["open"]), float(row["high"]),
                         float(row["low"]), float(row["close"]), vol])
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


def _hhmm(ts_epoch) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(
        float(ts_epoch), tz=timezone.utc).strftime("%H:%M")


def _lifecycle_lines(lifecycle) -> list[str]:
    """Lifecycle facts for the entry-timeframe analysis: the same
    LifecycleStudy the PNG role A renders — fill, every confirmed
    stop move, partial legs, the exit. Nothing reconstructed."""
    if lifecycle is None:
        return []
    lines: list[str] = []
    if lifecycle.fill_price is not None:
        lines.append(f"Filled {_fmt(lifecycle.fill_price)}"
                     + (f" at {_hhmm(lifecycle.fill_ts)} UTC"
                        if lifecycle.fill_ts else "")
                     + (f" · ticket {lifecycle.ticket}"
                        if lifecycle.ticket is not None else ""))
    for ev in lifecycle.stop_events:
        lines.append(f"{_hhmm(ev.ts)} SL → {_fmt(ev.level)}"
                     + (f" ({ev.reason})" if ev.reason else ""))
    for p in lifecycle.partials:
        lines.append(f"Partial {p.volume:g} @ {_fmt(p.price)}"
                     + (f" at {_hhmm(p.ts)} UTC" if p.ts else ""))
    if lifecycle.exit_price is not None:
        lines.append(f"Exit {_fmt(lifecycle.exit_price)}"
                     + (f" — {lifecycle.exit_reason}"
                        if lifecycle.exit_reason else ""))
    return lines


def _analysis_for_tf(tf: str, *, verdict, entry_clock, zone,
                     fib_info, fib_range, entry, stop, targets,
                     direction, broker_stop=None,
                     eff_r=None, lifecycle=None) -> list[str]:
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
        lines.extend(_lifecycle_lines(lifecycle))
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
                           signal_time: str | None = None,
                           qm_pattern: dict | None = None,
                           lifecycle=None) -> str:
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
    # Ladder-aware (2026-11-05): the page draws the same chart
    # targets as the PNGs (signal_model.chart_targets) — a
    # 20+-leg hot_scalp ladder must not tile the canvas either.
    _shown_t, _total_t = chart_targets(targets or [], entry=entry,
                                       direction=direction)
    drawn_targets = [t for _rank, t in _shown_t]
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
    if _total_t > len(drawn_targets):
        header_lines.append(
            f"Targets: {_total_t} planned legs — chart draws the "
            f"first {len(drawn_targets) - 1} + ladder top")
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
            "targets": [float(t) for t in drawn_targets],
            "target_ranks": [int(r) for r, _t in _shown_t],
            "targets_total": _total_t,
            "fill": float(fill_price) if fill_price is not None else None,
            "alert_id": alert_id, "signal_time": signal_time or "",
            "verdict_lines": header_lines,
            "entry_clock": entry_clock,
        },
        "lifecycle": (
            {"state": lifecycle.state, "ticket": lifecycle.ticket,
             "fill": lifecycle.fill_price, "fill_ts": lifecycle.fill_ts,
             "initial_broker_stop": lifecycle.initial_broker_stop,
             "current_broker_stop": lifecycle.current_broker_stop,
             "stop_events": [
                 {"ts": e.ts, "level": e.level, "reason": e.reason}
                 for e in lifecycle.stop_events],
             "partials": [
                 {"ts": p.ts, "price": p.price, "volume": p.volume}
                 for p in lifecycle.partials],
             "exit": lifecycle.exit_price,
             "exit_reason": lifecycle.exit_reason}
            if lifecycle is not None else None),
        "zone": zone_d,
        "qm": ({"head": float(qm_pattern["head"]),
                "neckline": float(qm_pattern["neckline"]),
                "level": (float(qm_pattern["level"])
                          if qm_pattern.get("level") is not None
                          else None)}
               if qm_pattern else None),
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
                    broker_stop=broker_stop, eff_r=eff_r,
                    lifecycle=lifecycle),
            }
            for tf in tfs
        },
    }
    buttons = "".join(
        f'<button class="tf" data-tf="{tf}">'
        f"{tf} · {panel_role(tf, playbook=playbook, entry_clock=entry_clock)}"
        f"</button>" for tf in tfs)
    data_json = _json_for_html(payload)
    from . import tv_page
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signal view</title>
<style>{_STUDY_CSS}</style></head><body data-theme="dark">
<h1 id="title">Signal view</h1><div id="meta"></div>
<div id="tfs">{buttons}</div>
<div><button id="theme-btn">Light theme</button></div>
<div id="toggles"><label><input type="checkbox" id="ov-zone" checked> Zone</label><label><input type="checkbox" id="ov-nearby" checked> Nearby levels</label><label><input type="checkbox" id="ov-fib" checked> Fib + OTE</label><label><input type="checkbox" id="ov-prem" checked> Premium/Discount</label><label><input type="checkbox" id="ov-sess" checked> Sessions</label><label><input type="checkbox" id="ov-levels" checked> Entry/SL/TP</label><label><input type="checkbox" id="ov-qm" checked> QM pattern</label></div>
<div id="wrap"><div id="chart"></div></div>
<div id="readout">Hover a candle for its values. Scroll to zoom, drag to pan, double-click to reset.</div>
<h2>Analysis on this timeframe</h2><div id="analysis"></div>
<div id="foot">Frozen at signal time — the record of what the engine saw when it flagged this signal, not a live view. Rendered by TradingView Lightweight Charts™ (vendored — works offline).</div>
<script>{tv_page.lwc_js()}</script>
<script id="signal-data" type="application/json">{data_json}</script>
<script>{tv_page.CHART_BUILDER_JS}</script>
<script>{_STUDY_JS}</script>
</body></html>"""


_STUDY_CSS = """
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
#chart{width:100%;height:520px}
#readout{font-family:monospace;font-size:13px;color:var(--gold);min-height:18px;margin:6px 0}
#analysis{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 14px;line-height:1.6}
#foot{color:var(--muted);font-size:12px;margin-top:10px}
"""

#: The study driver: builds the shared-builder config from the
#: frozen payload. Overlay semantics mirror the retired canvas
#: page exactly (same toggles, same studies, same admission rule
#: for the price range) — only the renderer changed.
_STUDY_JS = r"""
(function(){
"use strict";
const D = JSON.parse(document.getElementById('signal-data').textContent);
const GOLD = () => document.body.dataset.theme === 'dark'
  ? '#f7b924' : '#8a6d00';
const SOLID = 0, DOTTED = 1, DASHED = 2, LARGE_DASHED = 3;
let tf = (D.meta.entry_clock && D.tfs[D.meta.entry_clock])
  ? D.meta.entry_clock
  : (D.tfs.H1 ? 'H1' : Object.keys(D.tfs)[0]);
let built = null;
function ov(id){ return document.getElementById(id).checked; }
function fmt(x){ return x.toLocaleString('en-US',
  {minimumFractionDigits: 1, maximumFractionDigits: 1}); }
function epoch(iso){ return Math.floor(Date.parse(iso) / 1000); }

function tfData(t){
  const rows = D.tfs[t].bars;
  const bars = [], volume = [];
  const dark = document.body.dataset.theme === 'dark';
  rows.forEach(function(r){
    const time = epoch(r[0]);
    bars.push({time: time, open: r[1], high: r[2], low: r[3],
               close: r[4]});
    if (r[5] !== null && r[5] !== undefined){
      volume.push({time: time, value: r[5],
        color: (r[4] >= r[1] ? 'rgba(38,166,154,' : 'rgba(239,83,80,')
               + (dark ? '0.75)' : '0.55)')});
    }
  });
  return {bars: bars, volume: volume.length ? volume : null,
          rows: rows};
}

function yRange(bars){
  const n = bars.length;
  const win = bars.slice(Math.max(0, n - 120));
  let mn = Infinity, mx = -Infinity;
  win.forEach(function(b){
    mn = Math.min(mn, b.low); mx = Math.max(mx, b.high); });
  const span = Math.max(mx - mn, 1e-9);
  const cand = [D.meta.entry, D.meta.stop]
    .concat(D.meta.broker_stop === null ? [] : [D.meta.broker_stop])
    .concat(D.meta.targets)
    .concat(D.zone ? [D.zone.top, D.zone.bottom] : [])
    .concat(D.qm ? [D.qm.head, D.qm.neckline]
      .concat(D.qm.level === null ? [] : [D.qm.level]) : []);
  cand.forEach(function(v){
    if (v >= mn - 1.6 * span && v <= mx + 1.6 * span){
      mn = Math.min(mn, v); mx = Math.max(mx, v); } });
  const pad = 0.07 * Math.max(mx - mn, 1e-9);
  return [mn - pad, mx + pad];
}

function tfCfg(t){
  const data = tfData(t);
  const lines = [], bands = [], timeBands = [];
  const L = function(price, color, title, width, style){
    lines.push({price: price, color: color, title: title,
                width: width || 2, style: style || 0}); };
  if (ov('ov-sess')){
    const rows = data.rows;
    (D.tfs[t].sessions || []).forEach(function(s){
      const a = Math.max(0, Math.min(s[0], rows.length - 1));
      const b = Math.max(0, Math.min(s[1], rows.length - 1));
      if (rows[a] && rows[b] && b >= a){
        timeBands.push({from: epoch(rows[a][0]),
                        to: epoch(rows[b][0]),
                        fill: 'rgba(247,185,36,0.07)'}); } });
  }
  if (ov('ov-prem') && D.fib.range){
    const rlo = D.fib.range[0], rhi = D.fib.range[1],
          eq = (rlo + rhi) / 2;
    bands.push({top: rhi, bottom: eq, from: null, to: null,
                fill: 'rgba(239,83,80,0.05)', stroke: null});
    bands.push({top: eq, bottom: rlo, from: null, to: null,
                fill: 'rgba(38,166,154,0.05)', stroke: null});
    L(eq, '#787b86', 'EQ ' + fmt(eq), 1, DASHED);
  }
  if (ov('ov-zone') && D.zone){
    bands.push({top: D.zone.top, bottom: D.zone.bottom,
                from: null, to: null,
                fill: 'rgba(247,185,36,0.25)', stroke: null});
    L(D.zone.top, GOLD(),
      (D.zone.kind ? D.zone.kind + ' ' : '') + 'ZONE ' +
      fmt(D.zone.bottom) + '–' + fmt(D.zone.top), 1, DASHED);
  }
  if (ov('ov-nearby')){
    D.nearby.forEach(function(z){
      const col = z.side === 'support' ? '#26a69a'
        : z.side === 'resistance' ? '#ef5350' : '#787b86';
      L(z.level, col,
        (z.tier ? z.tier + ' ' : '') + fmt(z.level),
        z.tier === 'MAJOR' ? 2 : 1,
        z.tier === 'MAJOR' ? SOLID : DASHED); });
  }
  if (ov('ov-fib')){
    D.fib.levels.forEach(function(l){
      L(l[1], '#8d6e00', l[0].toFixed(3), 1, DASHED); });
    if (D.fib.ote){
      bands.push({top: D.fib.ote[1], bottom: D.fib.ote[0],
                  from: null, to: null,
                  fill: 'rgba(141,110,0,0.12)', stroke: null}); }
  }
  if (ov('ov-qm') && D.qm){
    L(D.qm.head, '#8e24aa', 'QM HEAD ' + fmt(D.qm.head), 1, DOTTED);
    L(D.qm.neckline, '#8e24aa', 'QM NECKLINE ' + fmt(D.qm.neckline),
      1, DASHED);
    if (D.qm.level !== null){
      L(D.qm.level, '#8e24aa', 'QM LEVEL ' + fmt(D.qm.level), 1,
        LARGE_DASHED); }
  }
  if (ov('ov-levels')){
    L(D.meta.entry, GOLD(), 'ENTRY ' + fmt(D.meta.entry), 2, SOLID);
    L(D.meta.stop, '#d32f2f',
      'STOP ' + fmt(D.meta.stop) + ' (structural)', 2, SOLID);
    if (D.meta.broker_stop !== null){
      L(D.meta.broker_stop, '#ff7043',
        'BROKER SL ' + fmt(D.meta.broker_stop), 2, SOLID); }
    D.meta.targets.forEach(function(tgt, j){
      const rank = (D.meta.target_ranks || [])[j] || (j + 1);
      L(tgt, '#2e7d32', 'T' + rank + ' ' + fmt(tgt) + ' (planned)',
        1, DASHED); });
    if (D.meta.fill !== null){
      L(D.meta.fill, '#7e57c2', 'FILL ' + fmt(D.meta.fill), 2,
        DASHED); }
  }
  return {bars: data.bars, volume: data.volume, lines: lines,
          lineSeries: [], bands: bands, timeBands: timeBands,
          markers: [], edgeTags: [], priceRange: yRange(data.bars)};
}

function defaultRange(){
  const n = D.tfs[tf].bars.length;
  return {from: Math.max(0, n - 120), to: n + 2};
}

function rebuild(keepRange){
  const el = document.getElementById('chart');
  let prev = null;
  if (built){
    try { prev = built.chart.timeScale().getVisibleLogicalRange(); }
    catch (e) { prev = null; }
    built.chart.remove();
    built = null;
  }
  built = window.GsChart.build(el, tfCfg(tf), {
    interactive: true, autoSize: true, fontSize: 13,
    dark: document.body.dataset.theme === 'dark'});
  const ts = built.chart.timeScale();
  ts.setVisibleLogicalRange(
    keepRange && prev ? prev : defaultRange());
  built.chart.subscribeCrosshairMove(function(param){
    const ro = document.getElementById('readout');
    if (!param || !param.time || !built){
      ro.textContent = 'Hover a candle for its values. Scroll to ' +
        'zoom, drag to pan, double-click to reset.';
      return; }
    const bar = param.seriesData.get(built.candle);
    if (!bar){ return; }
    const iso = new Date(param.time * 1000).toISOString();
    ro.textContent = iso.slice(0, 16).replace('T', ' ') +
      '  O ' + fmt(bar.open) + '  H ' + fmt(bar.high) +
      '  L ' + fmt(bar.low) + '  C ' + fmt(bar.close);
  });
}

function showTF(t){
  tf = t;
  document.querySelectorAll('button.tf').forEach(function(b){
    b.classList.toggle('on', b.dataset.tf === t); });
  const an = document.getElementById('analysis');
  an.innerHTML = '';
  D.tfs[t].analysis.forEach(function(l){
    const d = document.createElement('div');
    d.textContent = l; an.appendChild(d); });
  rebuild(false);
}

function setTheme(name){
  document.body.dataset.theme = name;
  document.getElementById('theme-btn').textContent =
    name === 'dark' ? 'Light theme' : 'Dark theme';
  rebuild(true);
}

document.querySelectorAll('button.tf').forEach(function(b){
  b.addEventListener('click', function(){ showTF(b.dataset.tf); });
});
['ov-zone', 'ov-nearby', 'ov-fib', 'ov-prem', 'ov-sess', 'ov-levels',
 'ov-qm'].forEach(function(id){
  document.getElementById(id).addEventListener('change',
    function(){ rebuild(true); });
});
document.getElementById('theme-btn').addEventListener('click',
  function(){
    setTheme(document.body.dataset.theme === 'dark'
      ? 'light' : 'dark'); });
document.getElementById('chart').addEventListener('dblclick',
  function(){
    if (built){
      built.chart.timeScale().setVisibleLogicalRange(
        defaultRange()); } });

(function(){
  const m = D.meta;
  const arrow = m.direction === 1 ? '▲ LONG'
    : m.direction === -1 ? '▼ SHORT' : 'FLAT';
  document.getElementById('title').textContent =
    m.instrument + ' · Signal view · ' + m.playbook + ' · ' + arrow;
  document.getElementById('meta').textContent =
    m.score_text + ' · Entry ' + fmt(m.entry) +
    ' · Structural stop ' + fmt(m.structural_stop) +
    (m.broker_stop === null
      ? ' · Broker SL pending at placement'
      : ' · Broker SL ' + fmt(m.broker_stop) +
        (m.effective_r === null
          ? '' : ' (effective R ' + m.effective_r.toFixed(2) + ')')) +
    (m.targets.length
      ? ' · Targets ' + m.targets.map(fmt).join(', ') : '') +
    (m.alert_id ? ' · ' + m.alert_id : '') +
    (m.verdict_lines.length
      ? ' · ' + m.verdict_lines.join(' · ') : '');
  showTF(tf);
})();
})();
"""
