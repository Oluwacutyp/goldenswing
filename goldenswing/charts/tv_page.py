"""TradingView Lightweight Charts pages — one generator, two outputs.

The alert PNG and the HTML signal study are rendered from the SAME
page machinery: this module turns the canonical study model
(:mod:`goldenswing.charts.signal_model` — roles A/B/C, the recorded
lifecycle, the capped ladder) into a self-contained HTML page whose
chart is drawn by TradingView Lightweight Charts™ (vendored,
Apache-2.0 — see ``assets/``; no CDN, the page opens offline on a
phone). The alert PNG is a headless screenshot of the role page
(:mod:`goldenswing.charts.screenshot`); the study page is the same
builder driven interactively. The two mediums cannot drift apart
because there is only one builder.

Truth contracts carried over from the matplotlib roles, unchanged:
planned levels are titled "(planned)" and dashed, never drawn as
fills; the structural stop and the broker stop are separate
objects; the broker stop's history is the step path through the
recorded lifecycle events; the ladder is capped (first 3 + top,
true ranks); a far level outside the crop is an edge tag with its
value, never an axis stretch; a missing frame renders a labeled
unavailable page — bars are never invented.

Library: TradingView Lightweight Charts™ v5.0.8 (pinned,
``assets/lightweight-charts.standalone.production.js``,
sha256 bcdca2a528db7c9b386918c99c544bde3eda3f0204ec2d23a64411d4
c4686c9). API verified against the v5.0.8 typings: addSeries with
series definitions, createSeriesMarkers, createPriceLine with
axisLabelColor/TextColor (the axis chips), series primitives for
the zone bands and edge tags, LineType.WithSteps (=1) for the
stop path.
"""
from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .signal_model import (
    ROLE_CONTEXT,
    ROLE_ENTRY,
    ROLE_ORDER,
    ROLE_TRADE,
    ChartClocks,
    admit_level_bounds,
    chart_targets,
)
from .studies import asof_align, study_levels, study_series

LWC_VERSION = "5.0.8"
ASSETS_DIR = Path(__file__).parent / "assets"
_LIB_PATH = ASSETS_DIR / "lightweight-charts.standalone.production.js"

#: Alert page geometry (portrait, phone-native): Telegram shows
#: portrait images large; the old 1600x700 landscape panels shrank
#: to illegibility on the owner's phone.
PAGE_W = 1242
PAGE_H = 1500

_HEADER_H = 96
_FOOTER_H = 30

_LIB_CACHE: str | None = None


def lwc_js() -> str:
    """The vendored Lightweight Charts bundle (read once)."""
    global _LIB_CACHE
    if _LIB_CACHE is None:
        _LIB_CACHE = _LIB_PATH.read_text(encoding="utf-8")
    return _LIB_CACHE


def _json_for_html(obj) -> str:
    """JSON safe to embed in a <script> block (same rule as the
    signal view's: <, >, & escaped so data can never break out)."""
    return (json.dumps(obj)
            .replace("&", "\\u0026")
            .replace("<", "\\u003c")
            .replace(">", "\\u003e"))


def _fmt(x: float) -> str:
    return f"{x:,.1f}"


def _hhmm(ts_epoch: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(
        float(ts_epoch), tz=timezone.utc).strftime("%H:%M")


def _epoch(ts) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp())


def _ts_to_pos(index, ts_epoch) -> int | None:
    """Panel-frame position of the bar containing a timestamp.

    None when the timestamp predates the frame — a marker is
    omitted, never placed by guessing (the matplotlib rule)."""
    if ts_epoch is None:
        return None
    try:
        ns = index.asi8
        pos = int(np.searchsorted(
            ns, int(float(ts_epoch) * 1e9), side="right")) - 1
    except Exception:
        return None
    return pos if 0 <= pos < len(index) else None


# ------------------------------------------------------------------ JS core
#: The one chart builder. Both the static alert pages and the
#: interactive study page call window.GsChart.build(el, cfg, opts).
#: cfg is plain JSON (see the builders below); opts.interactive
#: turns scrolling/zoom/crosshair on for the study.
CHART_BUILDER_JS = r"""
(function(){
"use strict";
const FONT = "-apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif";
function fmtPrice(v){ return v.toLocaleString('en-US',
  {minimumFractionDigits:1, maximumFractionDigits:1}); }

class BandsRenderer {
  constructor(src){ this._src = src; }
  draw(target){
    const src = this._src;
    if (!src._chart || !src._series) return;
    target.useMediaCoordinateSpace(function(scope){
      const ctx = scope.context, W = scope.mediaSize.width,
            H = scope.mediaSize.height;
      const ts = src._chart.timeScale();
      const xOf = function(t){
        if (t === null || t === undefined) return null;
        return ts.timeToCoordinate(t);
      };
      const yOf = function(p){ return src._series.priceToCoordinate(p); };
      ctx.save();
      for (const b of src._bands){
        const y1 = yOf(b.top), y2 = yOf(b.bottom);
        if (y1 === null || y2 === null) continue;
        let x1 = b.from == null ? 0 : xOf(b.from);
        let x2 = b.to == null ? W : xOf(b.to);
        if (x1 === null) x1 = 0;
        if (x2 === null) x2 = W;
        const yTop = Math.max(Math.min(y1, y2), -60),
              yBot = Math.min(Math.max(y1, y2), H + 60);
        ctx.fillStyle = b.fill;
        ctx.fillRect(Math.min(x1, x2), yTop,
                     Math.abs(x2 - x1), Math.max(yBot - yTop, 0));
        if (b.stroke){
          ctx.strokeStyle = b.stroke; ctx.lineWidth = 1;
          ctx.strokeRect(Math.min(x1, x2) + 0.5, yTop + 0.5,
                         Math.abs(x2 - x1) - 1,
                         Math.max(yBot - yTop - 1, 0));
        }
      }
      for (const tb of src._timeBands){
        let x1 = xOf(tb.from), x2 = xOf(tb.to);
        if (x1 === null) x1 = 0;
        if (x2 === null) x2 = W;
        ctx.fillStyle = tb.fill;
        ctx.fillRect(Math.min(x1, x2), 0, Math.abs(x2 - x1), H);
      }
      ctx.restore();
    });
  }
}
class BandsView {
  constructor(src){ this._src = src; }
  zOrder(){ return 'bottom'; }
  renderer(){ return new BandsRenderer(this._src); }
}
class BandsPrimitive {
  constructor(bands, timeBands){
    this._bands = bands || []; this._timeBands = timeBands || [];
    this._view = new BandsView(this);
  }
  attached(param){ this._chart = param.chart; this._series = param.series; }
  detached(){ this._chart = null; this._series = null; }
  paneViews(){ return [this._view]; }
}

class TagsRenderer {
  constructor(src){ this._src = src; }
  draw(target){
    const tags = this._src._tags;
    if (!tags.length) return;
    target.useMediaCoordinateSpace(function(scope){
      const ctx = scope.context, W = scope.mediaSize.width,
            H = scope.mediaSize.height;
      ctx.save();
      ctx.font = "700 21px " + FONT;
      for (const t of tags){
        const w = ctx.measureText(t.text).width + 22;
        const y = t.edge === 'top' ? 8 : H - 36;
        const x = Math.max(4, W - w - 8);
        ctx.fillStyle = t.color;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x, y, w, 28, 6); else ctx.rect(x, y, w, 28);
        ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.textBaseline = 'middle';
        ctx.fillText(t.text, x + 11, y + 15);
      }
      ctx.restore();
    });
  }
}
class TagsView {
  constructor(src){ this._src = src; }
  zOrder(){ return 'top'; }
  renderer(){ return new TagsRenderer(this._src); }
}
class TagsPrimitive {
  constructor(tags){ this._tags = tags || []; this._view = new TagsView(this); }
  attached(param){ this._chart = param.chart; }
  paneViews(){ return [this._view]; }
}

function build(el, cfg, opts){
  opts = opts || {};
  const LWC = window.LightweightCharts;
  const dark = !!opts.dark;
  const chart = LWC.createChart(el, {
    width: cfg.width || el.clientWidth,
    height: cfg.height || el.clientHeight,
    autoSize: !!opts.autoSize,
    layout: {
      background: {type: 'solid', color: dark ? '#131722' : '#ffffff'},
      textColor: dark ? '#d1d4dc' : '#131722',
      fontSize: opts.fontSize || 23,
      fontFamily: FONT,
      attributionLogo: false,
    },
    localization: { priceFormatter: fmtPrice },
    grid: {
      vertLines: {visible: false},
      horzLines: {color: dark ? '#1e2530' : '#eef0f4'},
    },
    crosshair: opts.interactive
      ? {mode: 0}
      : {vertLine: {visible: false, labelVisible: false},
         horzLine: {visible: false, labelVisible: false}},
    rightPriceScale: {borderVisible: false},
    timeScale: {borderVisible: false, timeVisible: true,
                secondsVisible: false, rightOffset: 2},
    handleScroll: !!opts.interactive,
    handleScale: !!opts.interactive,
  });
  const out = {chart: chart, lines: [], aux: [], prims: [],
               groups: {}};
  const candle = chart.addSeries(LWC.CandlestickSeries, {
    upColor: '#26a69a', downColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    borderVisible: false,
    priceFormat: {type: 'price', precision: 1, minMove: 0.1},
  });
  out.candle = candle;
  candle.setData(cfg.bars || []);
  if (cfg.volume && cfg.volume.length){
    const vol = chart.addSeries(LWC.HistogramSeries, {
      priceScaleId: '', priceLineVisible: false,
      lastValueVisible: false,
    });
    vol.setData(cfg.volume);
    chart.priceScale('').applyOptions(
      {scaleMargins: {top: 0.82, bottom: 0}});
    out.volume = vol;
  }
  chart.priceScale('right').applyOptions(
    {scaleMargins: {top: 0.05, bottom: 0.21}});
  const addLine = function(L){
    const pl = candle.createPriceLine({
      price: L.price, color: L.color, lineWidth: L.width || 2,
      lineStyle: L.style || 0, lineVisible: true,
      axisLabelVisible: true, title: L.title || '',
      axisLabelColor: L.color, axisLabelTextColor: '#ffffff',
    });
    out.lines.push(pl);
    return pl;
  };
  (cfg.lines || []).forEach(addLine);
  out.addLine = addLine;
  out.removeLines = function(lines){
    lines.forEach(function(pl){ candle.removePriceLine(pl); });
  };
  (cfg.lineSeries || []).forEach(function(ls){
    const s = chart.addSeries(LWC.LineSeries, {
      color: ls.color, lineWidth: ls.width || 2,
      lineType: ls.stepped ? 1 : 0,
      priceLineVisible: false, lastValueVisible: !!ls.lastChip,
      crosshairMarkerVisible: false,
      priceFormat: {type: 'price', precision: 1, minMove: 0.1},
    });
    s.setData(ls.points || []);
    out.aux.push(s);
  });
  (cfg.subSeries || []).forEach(function(ss, i){
    if (i === 0){
      const pane = chart.addPane();
      pane.setHeight(110);
    }
    const def = ss.kind === 'histogram' ? LWC.HistogramSeries
                                        : LWC.LineSeries;
    const s = chart.addSeries(def, {
      color: ss.color, lineWidth: ss.width || 2,
      priceLineVisible: false, lastValueVisible: true,
      crosshairMarkerVisible: false,
    }, 1);
    s.setData(ss.points || []);
    out.sub = out.sub || [];
    out.sub.push(s);
  });
  if ((cfg.bands && cfg.bands.length) ||
      (cfg.timeBands && cfg.timeBands.length)){
    const prim = new BandsPrimitive(cfg.bands, cfg.timeBands);
    candle.attachPrimitive(prim);
    out.prims.push(prim);
    out.bandsPrim = prim;
  }
  out.addBands = function(bands, timeBands){
    const prim = new BandsPrimitive(bands, timeBands);
    candle.attachPrimitive(prim);
    out.prims.push(prim);
    return prim;
  };
  out.removePrim = function(prim){ candle.detachPrimitive(prim); };
  if (cfg.edgeTags && cfg.edgeTags.length){
    const prim = new TagsPrimitive(cfg.edgeTags);
    candle.attachPrimitive(prim);
    out.prims.push(prim);
  }
  out.markersApi = null;
  out.setMarkers = function(markers){
    if (out.markersApi){ out.markersApi.detach(); out.markersApi = null; }
    if (markers && markers.length){
      const sorted = markers.slice().sort(function(a, b){
        return a.time - b.time; });
      out.markersApi = LWC.createSeriesMarkers(candle, sorted);
    }
  };
  out.setMarkers(cfg.markers);
  chart.timeScale().fitContent();
  if (cfg.priceRange){
    // AFTER fitContent: fitting the time scale must never be
    // allowed to re-open price autoscaling over the crop.
    const ps = candle.priceScale();
    ps.setAutoScale(false);
    ps.setVisibleRange({from: cfg.priceRange[0], to: cfg.priceRange[1]});
  }
  return out;
}
window.GsChart = {build: build, fmtPrice: fmtPrice, FONT: FONT};
})();
"""

# LineStyle values (v5 typings, passed numerically — const enums
# are erased from the production bundle).
_SOLID, _DOTTED, _DASHED, _LARGE_DASHED = 0, 1, 2, 3

_UP_HEX, _DOWN_HEX = "#26a69a", "#ef5350"
_UP_RGBA = "rgba(38,166,154,{})"
_DOWN_RGBA = "rgba(239,83,80,{})"


def _bars_json(view) -> list:
    return [{"time": _epoch(ts),
             "open": float(r["open"]), "high": float(r["high"]),
             "low": float(r["low"]), "close": float(r["close"])}
            for ts, r in view.iterrows()]


def _volume_json(view):
    """TV-style volume values, or None — volume is never invented."""
    if "volume" not in view.columns:
        return None
    vol = view["volume"].to_numpy(dtype=float)
    if not np.isfinite(vol).any() or float(np.nanmax(vol)) <= 0:
        return None
    out = []
    for (ts, r), v in zip(view.iterrows(), vol):
        up = float(r["close"]) >= float(r["open"])
        out.append({"time": _epoch(ts), "value": float(v),
                    "color": (_UP_RGBA if up else _DOWN_RGBA).format(
                        0.55)})
    return out


def _view_times(view) -> list:
    return [_epoch(ts) for ts in view.index]


def _price_range(view, candidates, *, mult: float = 1.6):
    """The crop: candle range + admitted near levels, padded.

    Admission is the shipped contract (signal_model.
    admit_level_bounds against the ORIGINAL candle bounds — the
    anti-ratchet rule). Levels outside the result are the
    caller's edge tags, never an axis stretch."""
    lows = view["low"].to_numpy(dtype=float)
    highs = view["high"].to_numpy(dtype=float)
    lo, hi = float(np.nanmin(lows)), float(np.nanmax(highs))
    lo, hi = admit_level_bounds(lo, hi, candidates, mult=mult)
    pad = 0.05 * max(hi - lo, 1e-9)
    return lo - pad, hi + pad


def _split_lines(lines: list, price_range) -> tuple:
    """(in-range lines, edge tags) for one role's level list.

    A line outside the crop becomes an edge tag carrying its full
    title + direction arrow; its axis chip would sit off-pane and
    say nothing. Titles of lines clustered within ~1.4% of the
    crop span are merged onto one line of the cluster (every line
    and chip is still drawn; the values are all named once, in
    one readable string instead of a smeared pile — the cluster
    case is real: TP1/TP2 0.1 apart on ticket 540274935)."""
    lo, hi = price_range
    inside, tags = [], []
    for ln in lines:
        if ln["price"] < lo:
            tags.append({"price": ln["price"], "edge": "bottom",
                         "color": ln["color"],
                         "text": f"↓ {ln['title']}"})
        elif ln["price"] > hi:
            tags.append({"price": ln["price"], "edge": "top",
                         "color": ln["color"],
                         "text": f"↑ {ln['title']}"})
        else:
            inside.append(ln)
    eps = max((hi - lo) * 0.014, 1e-9)
    ordered = sorted(inside, key=lambda l: l["price"])
    groups: list = []
    for ln in ordered:
        if groups and ln["price"] - groups[-1][-1]["price"] <= eps:
            groups[-1].append(ln)
        else:
            groups.append([ln])
    for group in groups:
        if len(group) > 1:
            titles = [l["title"] for l in group if l.get("title")]
            if titles:
                keep = group[len(group) // 2]
                for ln in group:
                    ln["title"] = ""
                keep["title"] = " · ".join(titles)
    return inside, tags


def _line(price, color, title, *, width=2, style=_SOLID):
    return {"price": float(price), "color": color,
            "width": width, "style": style, "title": title}


# ------------------------------------------------------------------ role A
def _trade_cfg(frame, lc, *, tf_label: str) -> dict:
    anchor = _ts_to_pos(frame.index, lc.fill_ts)
    if anchor is None:
        anchor = len(frame) - 1
    start = max(0, anchor - 46)
    view = frame.iloc[start:]
    offset = start
    times = _view_times(view)
    shown, total = chart_targets(lc.targets, entry=lc.entry_ref,
                                 direction=lc.direction)

    lines: list = []
    bands: list = []
    if lc.entry_zone:
        zlo, zhi = lc.entry_zone
        bands.append({"top": float(zhi), "bottom": float(zlo),
                      "from": None, "to": None,
                      "fill": "rgba(247,185,36,0.20)", "stroke": None})
        lines.append(_line(zhi, "#b8860b",
                           f"ENTRY ZONE {_fmt(zlo)}–{_fmt(zhi)}",
                           width=1, style=_DASHED))
        lines.append(_line(zlo, "#b8860b", "", width=1,
                           style=_DASHED))
    lines.append(_line(lc.entry_ref, "#b8860b",
                       f"ENTRY {_fmt(lc.entry_ref)} (planned)",
                       width=1, style=_DOTTED))
    if lc.fill_price is not None:
        lines.append(_line(lc.fill_price, "#7e57c2",
                           f"FILL {_fmt(lc.fill_price)}", width=2,
                           style=_DASHED))
    if lc.initial_broker_stop is not None and not lc.stop_events:
        lines.append(_line(lc.initial_broker_stop, "#e65100",
                           f"BROKER SL {_fmt(lc.initial_broker_stop)}"
                           " (initial)", width=2, style=_DASHED))
    lines.append(_line(lc.structural_stop, "#d32f2f",
                       f"STOP {_fmt(lc.structural_stop)} (structural)",
                       width=2))
    for rank, t in shown:
        lines.append(_line(t, "#2e7d32",
                           f"T{rank} {_fmt(t)} (planned)", width=1,
                           style=_DASHED))
    if lc.exit_price is not None:
        lines.append(_line(lc.exit_price, "#37474f",
                           f"EXIT {_fmt(lc.exit_price)}", width=2))

    step_series: list = []
    markers: list = []
    if lc.fill_price is not None:
        fx = anchor - offset
        markers.append({
            "time": times[fx],
            "position": "belowBar" if lc.direction >= 0 else "aboveBar",
            "shape": "arrowUp" if lc.direction >= 0 else "arrowDown",
            "color": "#7e57c2", "text": "FILL", "size": 1.6})
    if lc.initial_broker_stop is not None and lc.stop_events:
        pts = [{"time": times[anchor - offset],
                "value": float(lc.initial_broker_stop)}]
        for ev in lc.stop_events:
            ex = _ts_to_pos(frame.index, ev.ts)
            if ex is None:
                continue
            pts.append({"time": times[ex - offset],
                        "value": float(ev.level)})
            markers.append({
                "time": times[ex - offset],
                "position": "atPriceMiddle", "price": float(ev.level),
                "shape": "circle", "color": "#ff7043", "text": "",
                "size": 0.9})
        if pts[-1]["time"] < times[-1]:
            pts.append({"time": times[-1], "value": pts[-1]["value"]})
        # Two events inside one bar collapse to that bar's final
        # level (series times must be unique); every event still
        # carries its marker above and its ledger line below.
        deduped = []
        for pt in pts:
            if deduped and deduped[-1]["time"] == pt["time"]:
                deduped[-1] = pt
            else:
                deduped.append(pt)
        pts = deduped
        # WithSteps renders each point as reach-then-jump at its
        # own bar: exactly the recorded stop history.
        step_series.append({"points": pts, "color": "#ff7043",
                            "width": 3, "stepped": True,
                            "lastChip": True})
    for p in lc.partials:
        px = _ts_to_pos(frame.index, p.ts) if p.ts else None
        if px is None:
            continue
        markers.append({
            "time": times[px - offset],
            "position": "atPriceMiddle", "price": float(p.price),
            "shape": "circle", "color": "#7e57c2",
            "text": f"P {p.volume:g}", "size": 1.1})
    if lc.exit_price is not None:
        ex = _ts_to_pos(frame.index, lc.exit_ts)
        if ex is not None:
            markers.append({
                "time": times[ex - offset],
                "position": "atPriceMiddle",
                "price": float(lc.exit_price), "shape": "square",
                "color": "#37474f", "text": "EXIT", "size": 1.4})

    cand = [lc.entry_ref, lc.structural_stop]
    cand += [t for _r, t in shown]
    for v in (lc.fill_price, lc.initial_broker_stop,
              lc.current_broker_stop, lc.exit_price):
        if v is not None:
            cand.append(v)
    cand += [p.price for p in lc.partials]
    if lc.entry_zone:
        cand += list(lc.entry_zone)
    rng = _price_range(view, cand)
    inside, tags = _split_lines(lines, rng)

    arrow = "▲ LONG" if lc.direction >= 0 else "▼ SHORT"
    ledger = []
    if lc.state == "planned":
        ledger.append("plan — no fill yet · broker SL pending "
                      "at placement")
    else:
        state_line = f"ticket {lc.ticket} · {lc.state.upper()}"
        if lc.volume is not None:
            state_line += f" · {lc.volume:g} lots"
        ledger.append(state_line)
    events = []
    if lc.fill_price is not None and lc.fill_ts:
        events.append(f"{_hhmm(lc.fill_ts)} fill @ "
                      f"{_fmt(lc.fill_price)}")
    for ev in lc.stop_events:
        events.append(f"{_hhmm(ev.ts)} SL → {_fmt(ev.level)}"
                      + (f" ({ev.reason})" if ev.reason else ""))
    for p in lc.partials:
        events.append(f"{_hhmm(p.ts) if p.ts else '--:--'} partial "
                      f"{p.volume:g} @ {_fmt(p.price)}")
    if lc.exit_price is not None:
        events.append(f"{_hhmm(lc.exit_ts) if lc.exit_ts else '--:--'}"
                      f" exit {_fmt(lc.exit_price)}"
                      + (f" — {lc.exit_reason}" if lc.exit_reason
                         else ""))
    if events:
        ledger.append(" · ".join(events))
    if total > len(shown):
        ledger.append(f"targets: {total} planned legs — first "
                      f"{len(shown) - 1} + ladder top charted")
    return {
        "bars": _bars_json(view), "volume": _volume_json(view),
        "lines": inside, "lineSeries": step_series,
        "bands": bands, "timeBands": [], "markers": markers,
        "edgeTags": tags, "priceRange": [rng[0], rng[1]],
        "header": {"role": f"A · Trade — fill overview · {tf_label}"
                           f" · {arrow}",
                   "arrow": arrow},
        "ledger": ledger,
    }


# ------------------------------------------------------------------ studies
_STUDY_COLORS = {
    ("bollinger", "mid"): "#5a6472",
    ("bollinger", "upper"): "#2962ff",
    ("bollinger", "lower"): "#2962ff",
    ("atr_band", "upper"): "#00838f",
    ("atr_band", "lower"): "#00838f",
    ("vwap", "vwap"): "#8e24aa",
}


def _studies_cfg(studies, panel_frame, view, times, frames):
    """Declared studies → (aux, lines, sub, ledger, extra_prices).

    Each study is computed on ITS OWN clock's frame (its resolved
    ``clock_tf``) and as-of aligned onto the panel — never
    recomputed on the panel's clock.  A study whose own frame is
    missing is NAMED in the ledger as unavailable, never silently
    dropped and never drawn from the wrong frame.  ema_pair / qm
    keep their dedicated paths in the role builders and are
    skipped here."""
    aux: list = []
    lines: list = []
    sub: list = []
    ledger: list = []
    extra: list = []
    for spec in studies or ():
        if spec.kind in ("ema_pair", "qm"):
            continue
        src_tf = spec.clock_tf or ""
        src = None
        if frames is not None and src_tf:
            src = frames.get(src_tf)
        if src is None and not src_tf:
            src = panel_frame
        if src is None or len(src) == 0:
            ledger.append(f"{spec.label} · {src_tf} — frame "
                          f"unavailable, study not drawn")
            continue
        if spec.kind == "volume_profile":
            levels = study_levels(src, "volume_profile")
            if not levels:
                ledger.append(f"{spec.label} · {src_tf} — no "
                              f"value area yet")
                continue
            for name, price in levels.items():
                lines.append(_line(
                    price, "#7e57c2",
                    f"{name} {_fmt(price)} · VP {src_tf}",
                    width=2 if name == "POC" else 1,
                    style=_SOLID if name == "POC" else _DASHED))
                extra.append(price)
            ledger.append(f"VP POC {_fmt(levels['POC'])} · "
                          f"{src_tf}" if "POC" in levels
                          else f"Volume profile · {src_tf}")
            continue
        comps = study_series(src, spec.kind, spec.params)
        if not comps:
            ledger.append(f"{spec.label} · {src_tf} — not "
                          f"computable on this frame")
            continue
        if spec.pane == "sub":
            values = asof_align(src.index, comps[spec.kind],
                                view.index)
            pts = []
            prev = None
            for t, v in zip(times, values):
                if v != v:  # NaN — the study has not started
                    continue
                pts.append({"time": t, "value": float(v),
                            "color": "#26a69a" if prev is None
                            or v >= prev else "#ef5350"})
                prev = v
            sub.append({"points": pts, "color": "#26a69a",
                        "width": 2, "kind": "histogram"})
            if prev is not None:
                ledger.append(f"{spec.label} {_fmt(prev)} · "
                              f"{src_tf}")
            continue
        last_seen = None
        for comp_name, comp_values in comps.items():
            aligned = asof_align(src.index, comp_values,
                                 view.index)
            pts = [{"time": t, "value": float(v)}
                   for t, v in zip(times, aligned) if v == v]
            if not pts:
                continue
            aux.append({"points": pts,
                        "color": _STUDY_COLORS.get(
                            (spec.kind, comp_name), "#5a6472"),
                        "width": 1 if comp_name != "vwap" else 2,
                        "stepped": False, "lastChip": True})
            last_seen = pts[-1]["value"]
        if last_seen is not None:
            ledger.append(f"{spec.label} {_fmt(last_seen)} · "
                          f"{src_tf}")
    return aux, lines, sub, ledger, extra


def _fresh_line(tf: str, freshness) -> str:
    """The SPEC §8.4 freshness stamp for one role's header."""
    if not freshness:
        return ""
    row = freshness.get(tf) or {}
    bits = []
    if row.get("last_bar"):
        bits.append(f"last bar {row['last_bar']}")
    if row.get("quality"):
        bits.append(f"data {row['quality']}")
    return " · ".join(bits)


# ------------------------------------------------------------------ role B
def _context_cfg(h1, ctx, lc, *, window: int = 120,
                 tf_label: str = "H1", frames=None) -> dict:
    view = h1.iloc[-window:]
    offset = len(h1) - len(view)
    times = _view_times(view)
    lines: list = []
    bands: list = []
    markers: list = []
    aux: list = []
    sub: list = []
    study_ledger: list = []
    study_extra: list = []
    shown, _total = chart_targets(lc.targets, entry=lc.entry_ref,
                                  direction=lc.direction)

    if ctx is not None:
        for z in ctx.zones:
            col = _UP_HEX if z.direction >= 0 else _DOWN_HEX
            alpha = 0.15 if z.kind == "FVG" else 0.11
            fill = (_UP_RGBA if z.direction >= 0 else _DOWN_RGBA)
            frm = (times[z.created_bar - offset]
                   if 0 <= z.created_bar - offset < len(times)
                   else None)
            bands.append({"top": float(z.top),
                          "bottom": float(z.bottom), "from": frm,
                          "to": None, "fill": fill.format(alpha),
                          "stroke": None})
            lines.append(_line(0.5 * (z.top + z.bottom), col,
                               f"{z.kind} · {tf_label} · "
                               f"{z.age_bars}b",
                               width=1, style=_DOTTED))
        for price, side, strength in ctx.pools:
            col = "#e08a00" if side == "above" else "#00838f"
            lines.append(_line(
                price, col,
                f"{'BSL' if side == 'above' else 'SSL'} "
                f"{_fmt(price)} x{strength}", width=1,
                style=_DASHED))
        for price, tf, tier in ctx.htf_levels:
            lines.append(_line(price, "#7e57c2",
                               f"{tf} {tier} {_fmt(price)}",
                               width=2 if tier == "MAJOR" else 1,
                               style=_SOLID if tier == "MAJOR"
                               else _DASHED))
        for bar, price, lab in ctx.swings:
            x = bar - offset
            if not (0 <= x < len(view)):
                continue
            above = lab in ("H", "HH", "LH")
            markers.append({
                "time": times[x],
                "position": "aboveBar" if above else "belowBar",
                "shape": "circle", "color": "#131722",
                "text": lab, "size": 0.8})
        ev = ctx.structure_event
        if ev is not None:
            x = ev["bar"] - offset
            if 0 <= x < len(view):
                # No marker text: the event's price-line tag on
                # the axis names it (BOS 4,124.7) — a second copy
                # floated over the candles collided with it.
                markers.append({
                    "time": times[x],
                    "position": ("aboveBar" if ev["direction"] >= 0
                                 else "belowBar"),
                    "shape": ("arrowUp" if ev["direction"] >= 0
                              else "arrowDown"),
                    "color": "#37474f", "text": "",
                    "size": 1.2})
            lines.append(_line(ev["level"], "#37474f",
                               f"{ev['kind']} {_fmt(ev['level'])}",
                               width=1, style=_DASHED))
        if ctx.trigger:
            markers.append({
                "time": times[-1], "position": "aboveBar",
                "shape": "arrowDown", "color": "#7e57c2",
                "text": f"TRIGGER {ctx.trigger}", "size": 1.3})

    lines.append(_line(lc.entry_ref, "#b8860b",
                       f"ENTRY {_fmt(lc.entry_ref)} (ref)", width=1))
    lines.append(_line(lc.structural_stop, "#d32f2f",
                       f"STOP {_fmt(lc.structural_stop)} (ref)",
                       width=1))
    for rank, t in shown:
        lines.append(_line(t, "#2e7d32", f"T{rank} {_fmt(t)} (ref)",
                           width=1, style=_DASHED))
    if ctx is not None and getattr(ctx, "studies", ()):
        s_aux, s_lines, s_sub, s_ledger, s_extra = _studies_cfg(
            ctx.studies, h1, view, times, frames)
        aux += s_aux
        lines += s_lines
        sub += s_sub
        study_ledger += s_ledger
        study_extra += s_extra

    cand = [lc.entry_ref, lc.structural_stop] + [t for _r, t in shown] \
        + study_extra
    if ctx is not None:
        cand += [p for p, _s, _n in ctx.pools]
        cand += [p for p, _tf, _tier in ctx.htf_levels]
        for z in ctx.zones:
            cand += [z.top, z.bottom]
    rng = _price_range(view, cand)
    inside, tags = _split_lines(lines, rng)

    title = f"B · {tf_label} — context"
    if ctx is not None:
        title += (f" · D1 {ctx.bias_d1:+d} / H4 {ctx.bias_h4:+d}"
                  + (f" / {ctx.trigger}" if ctx.trigger else ""))
    else:
        title += " · structure read unavailable"
    ledger = []
    if ctx is not None:
        bits = []
        if ctx.zones:
            bits.append(f"{len(ctx.zones)} zone"
                        f"{'s' if len(ctx.zones) != 1 else ''}")
        if ctx.pools:
            bits.append(f"{len(ctx.pools)} pool"
                        f"{'s' if len(ctx.pools) != 1 else ''}")
        if ctx.structure_event:
            bits.append(f"{ctx.structure_event['kind']} "
                        f"{ctx.structure_event['age_bars']}b ago")
        if bits:
            ledger.append(" · ".join(bits)
                          + " — as recorded at the signal bar")
    return {
        "bars": _bars_json(view), "volume": _volume_json(view),
        "lines": inside, "lineSeries": aux, "bands": bands,
        "timeBands": [], "markers": markers, "edgeTags": tags,
        "priceRange": [rng[0], rng[1]],
        "header": {"role": title, "arrow": ""},
        "ledger": ledger + study_ledger,
        "subSeries": sub,
    }


# ------------------------------------------------------------------ role C
def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """EMA identical to the evidence pipeline's (pandas ewm)."""
    return pd.Series(values).ewm(span=span, adjust=False) \
        .mean().to_numpy(dtype=float)


def _entry_cfg(frame, es, lc, *, tf_label: str,
               window: int = 44, frames=None) -> dict:
    anchor = _ts_to_pos(frame.index, lc.fill_ts)
    if anchor is None:
        anchor = len(frame) - 1
    start = max(0, anchor - (window - 8))
    view = frame.iloc[start:]
    offset = start
    times = _view_times(view)
    lines: list = []
    bands: list = []
    markers: list = []
    aux: list = []
    shown, _total = chart_targets(lc.targets, entry=lc.entry_ref,
                                  direction=lc.direction)

    if es.entry_zone:
        zlo, zhi = es.entry_zone
        bands.append({"top": float(zhi), "bottom": float(zlo),
                      "from": None, "to": None,
                      "fill": "rgba(247,185,36,0.22)", "stroke": None})
        kind = f"{es.zone_kind} " if es.zone_kind else ""
        lines.append(_line(zhi, "#b8860b",
                           f"{kind}ZONE {_fmt(zlo)}–{_fmt(zhi)}",
                           width=1, style=_DASHED))
        lines.append(_line(zlo, "#b8860b", "", width=1,
                           style=_DASHED))
    for ts_val, tag, col in (
            (es.formation_ts, "FORMED", "#b8860b"),
            (es.first_touch_ts, "FIRST TOUCH", "#00838f")):
        if ts_val is None:
            continue
        pos = _ts_to_pos(frame.index, ts_val)
        if pos is None:
            continue
        x = pos - offset
        if not (0 <= x < len(view)):
            continue
        markers.append({
            "time": times[x], "position": "aboveBar",
            "shape": "circle", "color": col,
            "text": f"{tag} {_hhmm(ts_val)}", "size": 1.0})
    trig_x = anchor - offset
    markers.append({
        "time": times[trig_x], "position": "inBar",
        "shape": "arrowUp" if lc.direction >= 0 else "arrowDown",
        "color": "#7e57c2",
        "text": (f"TRIGGER {es.trigger_label}"
                 if es.trigger_label else "TRIGGER"),
        "size": 1.4})

    ledger: list = []
    if "ema_pair" in es.overlays:
        closes = frame["close"].to_numpy(dtype=float)
        e9 = _ema(closes, 9)
        e21 = _ema(closes, 21)
        sl = slice(offset, offset + len(view))
        aux.append({"points": [{"time": t, "value": float(v)}
                               for t, v in zip(times, e9[sl])],
                    "color": "#2962ff", "width": 2, "stepped": False,
                    "lastChip": False})
        aux.append({"points": [{"time": t, "value": float(v)}
                               for t, v in zip(times, e21[sl])],
                    "color": "#ff6d00", "width": 2, "stepped": False,
                    "lastChip": False})
        ledger.append(f"EMA9 {_fmt(e9[anchor])} · "
                      f"EMA21 {_fmt(e21[anchor])}")
    if "qm" in es.overlays and es.qm_pattern:
        pat = es.qm_pattern
        for key, lab, style in (("head", "QM HEAD", _DOTTED),
                                ("neckline", "QM NECKLINE", _DASHED),
                                ("level", "QM LEVEL", _LARGE_DASHED)):
            if pat.get(key) is not None:
                lines.append(_line(pat[key], "#8e24aa",
                                   f"{lab} {_fmt(pat[key])}",
                                   width=1, style=style))
        touch = _ts_to_pos(frame.index, pat.get("touch_ts"))
        if touch is not None and 0 <= touch - offset < len(view):
            markers.append({
                "time": times[touch - offset],
                "position": "aboveBar", "shape": "circle",
                "color": "#8e24aa", "text": "QM TOUCH", "size": 1.0})

    lines.append(_line(lc.entry_ref, "#8a6d00",
                       f"ENTRY {_fmt(lc.entry_ref)}", width=2))
    lines.append(_line(lc.structural_stop, "#d32f2f",
                       f"STOP {_fmt(lc.structural_stop)} (structural)",
                       width=2))
    for rank, t in shown:
        lines.append(_line(t, "#2e7d32",
                           f"T{rank} {_fmt(t)} (planned)", width=1,
                           style=_DASHED))
    if lc.fill_price is not None:
        # Line + axis chip carry the fill; a second FILL marker
        # on the trigger bar crowded the TRIGGER tag (the shipped
        # matplotlib role C likewise marks only the trigger).
        lines.append(_line(lc.fill_price, "#7e57c2",
                           f"FILL {_fmt(lc.fill_price)}", width=2,
                           style=_DASHED))

    sub: list = []
    if getattr(es, "studies", ()):
        s_aux, s_lines, s_sub, s_ledger, s_extra = _studies_cfg(
            es.studies, frame, view, times, frames)
        aux += s_aux
        lines += s_lines
        sub += s_sub
        ledger.extend(s_ledger)
    else:
        s_extra = []

    cand = [lc.entry_ref, lc.structural_stop] + [t for _r, t in shown] \
        + s_extra
    if lc.fill_price is not None:
        cand.append(lc.fill_price)
    if es.entry_zone:
        cand += list(es.entry_zone)
    rng = _price_range(view, cand)
    inside, tags = _split_lines(lines, rng)
    ledger.extend(es.reason_lines)
    return {
        "bars": _bars_json(view), "volume": _volume_json(view),
        "lines": inside, "lineSeries": aux, "bands": bands,
        "timeBands": [], "markers": markers, "edgeTags": tags,
        "priceRange": [rng[0], rng[1]],
        "header": {"role": f"C · {tf_label} — entry precision",
                   "arrow": ""},
        "ledger": ledger,
        "subSeries": sub,
    }


# ------------------------------------------------------------------ pages
_PAGE_CSS = """
html,body{margin:0;padding:0;background:#ffffff;color:#131722;
  font-family:-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif}
body{width:1242px;height:1500px;overflow:hidden;display:flex;
  flex-direction:column}
.hdr{padding:14px 20px 8px 20px}
.hdr .row1{font-size:31px;line-height:1.25}
.hdr .sym{font-weight:700}
.hdr .dim{color:#5a6472}
.hdr .role{float:right;font-size:24px;color:#5a6472;font-weight:600}
.hdr .ohlc{font-size:25px;color:#5a6472;margin-top:2px}
.hdr .ohlc b{font-weight:600}
#chart{flex:none;margin:0 4px}
.ledger{padding:10px 20px 4px 20px;font-family:ui-monospace,
  Menlo,Consolas,monospace;font-size:21px;line-height:1.4;
  color:#2a3242}
.foot{padding:2px 20px 8px 20px;font-size:17px;color:#9aa3b2}
.msg{flex:1;display:flex;align-items:center;justify-content:center;
  font-size:30px;color:#5a6472;text-align:center;padding:0 60px}
"""


def _ledger_height(lines) -> int:
    if not lines:
        return 0
    wrapped = 0
    for line in lines:
        wrapped += max(1, -(-len(line) // 96))
    return 14 + 30 * wrapped


def role_page(cfg, *, instrument: str, playbook: str,
              score_line: str, tf_label: str) -> str:
    """One static alert page (screenshot-ready, portrait)."""
    ledger = cfg.get("ledger") or []
    chart_h = (PAGE_H - _HEADER_H - _ledger_height(ledger)
               - _FOOTER_H)
    cfg = dict(cfg)
    cfg["width"] = PAGE_W - 8
    cfg["height"] = chart_h
    last = cfg["bars"][-1] if cfg.get("bars") else None
    ohlc = ""
    if last:
        chg = last["close"] - last["open"]
        col = "#26a69a" if chg >= 0 else "#ef5350"
        ohlc = (f"O <b>{_fmt(last['open'])}</b> · "
                f"H <b>{_fmt(last['high'])}</b> · "
                f"L <b>{_fmt(last['low'])}</b> · "
                f"C <b style='color:{col}'>{_fmt(last['close'])}</b>")
    header = cfg.get("header") or {}
    fresh = header.get("fresh")
    if fresh:
        ohlc = (ohlc + " · " if ohlc else "") + _html_escape(fresh)
    title_bits = " · ".join(
        x for x in (playbook, tf_label, score_line) if x)
    ledger_html = "".join(
        f"<div>{_html_escape(line)}</div>" for line in ledger)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{_html_escape(instrument)} {_html_escape(header.get('role', ''))}</title>
<style>{_PAGE_CSS}</style></head>
<body>
<div class="hdr"><div class="row1"><span class="sym">{_html_escape(instrument)}</span><span class="dim"> · {_html_escape(title_bits)}</span><span class="role">{_html_escape(header.get('role', ''))}</span></div>
<div class="ohlc">{ohlc}</div></div>
<div id="chart" style="width:{PAGE_W - 8}px;height:{chart_h}px"></div>
<div class="ledger">{ledger_html}</div>
<div class="foot">TradingView Lightweight Charts™ · the journal record, rendered</div>
<script>{lwc_js()}</script>
<script id="chart-cfg" type="application/json">{_json_for_html(cfg)}</script>
<script>{CHART_BUILDER_JS}</script>
<script>
(function(){{
  const cfg = JSON.parse(document.getElementById('chart-cfg').textContent);
  window.GsChart.build(document.getElementById('chart'), cfg, {{interactive: false}});
}})();
</script>
</body></html>"""


def _html_escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def unavailable_page(*, instrument: str, playbook: str,
                     score_line: str, role_title: str,
                     tf_label: str) -> str:
    """A role whose frame is missing: labeled, never invented."""
    title_bits = " · ".join(
        x for x in (playbook, tf_label, score_line) if x)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{_html_escape(instrument)} {_html_escape(role_title)}</title>
<style>{_PAGE_CSS}</style></head>
<body>
<div class="hdr"><div class="row1"><span class="sym">{_html_escape(instrument)}</span><span class="dim"> · {_html_escape(title_bits)}</span><span class="role">{_html_escape(role_title)}</span></div></div>
<div class="msg">{_html_escape(tf_label)} frame unavailable — panel not charted (no bars invented)</div>
<div class="foot">TradingView Lightweight Charts™ · the journal record, rendered</div>
</body></html>"""


def build_role_pages(*, frames: dict, lifecycle, context=None,
                     entry_study=None, instrument: str = "XAUUSD",
                     playbook: str = "", score_line: str = "",
                     entry_clock: str = "M15",
                     roles=ROLE_ORDER, clocks: ChartClocks = None,
                     freshness: dict = None) -> list:
    """The role trio as self-contained pages, in ROLE_ORDER.

    Same inputs and the same role split as the matplotlib
    ``render_mt5_roles`` — the screenshot rung and the fallback
    rung differ only in medium, never in what the trade was.

    Frames come from the resolved ChartClocks (SPEC §8.1): role A
    the management clock, role B the context clock, role C the
    entry clock.  A caller passing no clocks gets the legacy
    derivation (A/C on the entry clock, B on H1) — the runner
    always passes resolved clocks."""
    from .signal_model import EntryStudy
    eff = clocks or ChartClocks(
        management_tf=entry_clock, context_tf="H1",
        entry_tf=entry_clock, source="legacy_entry_clock")
    out: list = []
    for role in roles:
        if role == ROLE_TRADE:
            tf = eff.management_tf
            frame = frames.get(tf)
            if frame is None or len(frame) == 0:
                out.append(unavailable_page(
                    instrument=instrument, playbook=playbook,
                    score_line=score_line,
                    role_title=f"A · Trade — fill overview · "
                               f"{tf}", tf_label=tf))
            else:
                cfg = _trade_cfg(frame, lifecycle, tf_label=tf)
                cfg["header"]["fresh"] = _fresh_line(tf, freshness)
                out.append(role_page(
                    cfg, instrument=instrument, playbook=playbook,
                    score_line=score_line, tf_label=tf))
        elif role == ROLE_CONTEXT:
            tf = eff.context_tf
            frame = frames.get(tf)
            if frame is None or len(frame) == 0:
                out.append(unavailable_page(
                    instrument=instrument, playbook=playbook,
                    score_line=score_line,
                    role_title=f"B · {tf} — context", tf_label=tf))
            else:
                cfg = _context_cfg(frame, context, lifecycle,
                                   tf_label=tf, frames=frames)
                cfg["header"]["fresh"] = _fresh_line(tf, freshness)
                out.append(role_page(
                    cfg, instrument=instrument, playbook=playbook,
                    score_line=score_line, tf_label=tf))
        elif role == ROLE_ENTRY:
            tf = eff.entry_tf
            frame = frames.get(tf)
            if frame is None or len(frame) == 0:
                out.append(unavailable_page(
                    instrument=instrument, playbook=playbook,
                    score_line=score_line,
                    role_title=f"C · {tf} — entry precision",
                    tf_label=tf))
            else:
                cfg = _entry_cfg(frame,
                                 entry_study or EntryStudy(),
                                 lifecycle, tf_label=tf,
                                 frames=frames)
                cfg["header"]["fresh"] = _fresh_line(tf, freshness)
                out.append(role_page(
                    cfg, instrument=instrument, playbook=playbook,
                    score_line=score_line, tf_label=tf))
    return out
