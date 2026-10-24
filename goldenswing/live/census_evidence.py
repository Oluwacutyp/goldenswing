"""Census-evidence consumer — the live MT5 path's read of the
capabilities the Phase 2 census found implemented but UNWIRED
(or, for the named candlestick taxonomy, absent).

One snapshot per evaluation bar, computed from CAUSAL inputs
only (the frames exactly as the runner holds them at emit
time; every detector below is trailing-only) and recorded on
the funnel evaluation record as ``evidence["census"]``:

- ``chart_patterns`` — features/patterns.py double top/bottom
  + head & shoulders, run on the record frame's reader states
  (the batch detector, consumed live at last);
- ``premium_discount`` — live/patterns.py ``premium_discount``
  (dealing-range EQ / premium-discount / OTE), consumed live;
- ``breaker_blocks`` / ``mitigation_blocks`` — live/patterns.py
  block helpers, consumed live;
- ``relative_strength`` — features/relstrength.py vs the DXY
  tracker's H1 frame (the peer the engine actually fetches;
  silver has no live feed and reports unavailable, never a
  fabricated value);
- ``candles.engulfing`` — live/revival.py ``has_engulfing``
  (the experiments/candles.py definition), which until now
  reached only the legacy paper loop;
- ``candles.taxonomy`` — live/candle_taxonomy.py named
  patterns (hammer / doji family / engulfing-by-name / …).

This is RECORDED EVIDENCE, surfaced truthfully in assessments.
It carries NO decision weight: promotion of any detector to
the decision path is the owner's shadow-first/promotion gate,
not this consumer's business.  Every section degrades to an
explicit ``unavailable`` with its reason — never zeros, never
invented values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import patterns as _feat_patterns
from ..features import relstrength as _relstrength
from ..features.config import FeatureConfig
from ..reader.reader import MarketReader
from . import patterns as _live_patterns
from .candle_taxonomy import detect_named_patterns
from .revival import has_engulfing as _has_engulfing

#: How many of the freshest blocks of each kind ride the record
#: (the full lists stay reproducible from the frame; the record
#: carries the actionable head, not an unbounded dump).
_MAX_BLOCKS = 3
#: Named-pattern lookback window, in bars, on each record.
_TAXONOMY_WINDOW = 10
#: The consumer's evidence window: detectors read at most this
#: many trailing bars of a frame.  Every wired detector's
#: memory is shorter — classical patterns die 150 bars after
#: forming (pat_max_age), the RS percentile spans 252 + a
#: 24-bar ROC, premium_discount looks back 120, the block
#: helpers 60 — so the window changes no last-bar read; it
#: bounds the per-bar cost of the read on long frames.
_EVIDENCE_WINDOW = 600


def _num(v):
    """JSON-safe float or None (NaN/inf collapse to None)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _pattern_dict(inst, frame: pd.DataFrame) -> dict:
    formed = int(inst.formed_at)
    return {
        "kind": inst.kind, "direction": int(inst.direction),
        "status": inst.status,
        "formed_bar_ts": (str(frame.index[formed])
                          if 0 <= formed < len(frame) else None),
        "neckline": _num(inst.neckline),
        "extreme": _num(inst.extreme),
        "target": _num(inst.target)}


def _block_dict(b: dict, frame: pd.DataFrame) -> dict:
    idx = b.get("bar_index")
    return {
        "kind": b.get("kind"), "direction": b.get("direction"),
        "top": _num(b.get("top")), "bottom": _num(b.get("bottom")),
        "level": _num(b.get("level")),
        "bar_ts": (str(frame.index[int(idx)])
                   if idx is not None and 0 <= int(idx) < len(frame)
                   else None)}


def chart_patterns_snapshot(frame: pd.DataFrame,
                            cfg: FeatureConfig) -> dict:
    """features/patterns.py at the frame's last bar: the active
    classical patterns + the pattern event (if any) on that bar."""
    states = MarketReader().read(frame)
    series = _feat_patterns.compute(frame, states, cfg)
    last = series[-1] if series else None
    if last is None:
        return {"status": "unavailable",
                "reason": "no bars to evaluate"}
    return {
        "status": "recorded",
        "active": [_pattern_dict(p, frame) for p in last.active],
        "event": (_pattern_dict(last.event, frame)
                  if last.event is not None else None)}


def premium_discount_snapshot(frame: pd.DataFrame) -> dict:
    pd_read = _live_patterns.premium_discount(frame)
    if pd_read.get("eq") is None:
        return {"status": "unavailable",
                "reason": pd_read.get("detail")
                or "premium_discount produced no read"}
    ote = pd_read.get("ote") or {}
    return {
        "status": "recorded",
        "zone": pd_read.get("zone"),
        "eq": _num(pd_read.get("eq")),
        "range_high": _num(pd_read.get("range_high")),
        "range_low": _num(pd_read.get("range_low")),
        "ote_low": _num(ote.get("low")),
        "ote_high": _num(ote.get("high")),
        "ote_50": _num(pd_read.get("ote_50")),
        "allowed_direction": pd_read.get("allowed_direction"),
        "detail": pd_read.get("detail")}


def blocks_snapshot(frame: pd.DataFrame) -> tuple[dict, dict]:
    """Breaker + mitigation blocks (live/patterns.py helpers):
    counts + the freshest few of each kind.

    The legacy helpers index bars POSITIONALLY (``int(sub.index
    [i])`` is their bar_index) — they were written against the
    batch layer's integer-indexed frames and raise on a live
    datetime-indexed frame.  The consumer therefore feeds them
    a positional view of the same bars and maps bar_index back
    to the live frame's timestamps: the detectors run exactly
    as written, on exactly the live bars."""
    positional = frame.reset_index(drop=True)

    def _snap(blocks: list[dict]) -> dict:
        return {"status": "recorded", "count": len(blocks),
                "freshest": [_block_dict(b, frame)
                             for b in blocks[:_MAX_BLOCKS]]}
    return (_snap(_live_patterns.breaker_blocks(positional)),
            _snap(_live_patterns.mitigation_blocks(positional)))


def relative_strength_snapshot(gold_h1: pd.DataFrame | None,
                               dxy_frame: pd.DataFrame | None,
                               cfg: FeatureConfig) -> dict:
    """features/relstrength.py at the last H1 bar, gold vs the
    DXY tracker's frame (the engine's one live peer)."""
    if gold_h1 is None or len(gold_h1) < 2:
        return {"status": "unavailable",
                "reason": "no H1 frame in scope for the RS read"}
    states = MarketReader().read(gold_h1)
    peers = {"dollar": dxy_frame} if dxy_frame is not None \
        else None
    series = _relstrength.compute(gold_h1, states, cfg, peers)
    last = series[-1] if series else None
    if last is None:
        return {"status": "unavailable",
                "reason": "relative strength produced no state"}

    def _peer(p) -> dict:
        return {"available": bool(p.available),
                "trend": int(p.trend) if p.available else None,
                "percentile": _num(p.percentile)
                if p.available else None,
                "ratio": _num(p.ratio) if p.available else None}

    out = {"frame": "H1",
           "dollar": _peer(last.dollar),
           "silver": _peer(last.silver),
           "score": _num(last.score)}
    if last.available:
        out["status"] = "recorded"
    else:
        out["status"] = "unavailable"
        out["reason"] = ("no peer series usable at the last H1 "
                         "bar (DXY frame missing or fewer than "
                         "60 causally-aligned bars; silver has "
                         "no live feed)")
    return out


def candles_snapshot(frame: pd.DataFrame) -> dict:
    """Engulfing (revival/experiments definition, last two
    closed bars) + the named-pattern taxonomy (last
    ``_TAXONOMY_WINDOW`` bars) on the record frame."""
    o = frame["open"].to_numpy(dtype=float)
    h = frame["high"].to_numpy(dtype=float)
    l = frame["low"].to_numpy(dtype=float)
    c = frame["close"].to_numpy(dtype=float)
    t = len(frame) - 1
    direction = 0
    if _has_engulfing(o, h, l, c, t, 1):
        direction = 1
    elif _has_engulfing(o, h, l, c, t, -1):
        direction = -1
    return {
        "status": "recorded",
        "engulfing": {
            "direction": direction,
            "checked": "last two bars (t, t-1)",
            "detector": "live/revival.has_engulfing "
                        "(experiments/candles.engulfing)"},
        "taxonomy": detect_named_patterns(
            frame, window=_TAXONOMY_WINDOW)}


def census_snapshot(*, frame: pd.DataFrame | None, tf: str,
                    h1_frame: pd.DataFrame | None = None,
                    dxy_frame: pd.DataFrame | None = None,
                    cfg: FeatureConfig | None = None) -> dict:
    """The full census-evidence snapshot for ONE evaluation
    record.  Frame-derived sections read the record's own
    frame; relative strength reads the H1/DXY pair the engine
    tracks.  Sections fail independently and say so."""
    cfg = cfg or FeatureConfig()
    if frame is None or len(frame) < 2:
        return {"status": "unavailable",
                "reason": "no evaluation frame in scope at "
                          "emit time"}
    if len(frame) > _EVIDENCE_WINDOW:
        frame = frame.iloc[-_EVIDENCE_WINDOW:]
    if h1_frame is not None and len(h1_frame) > _EVIDENCE_WINDOW:
        h1_frame = h1_frame.iloc[-_EVIDENCE_WINDOW:]
    breakers, mitigation = blocks_snapshot(frame)
    return {
        "status": "recorded",
        "frame": tf,
        "bar_ts": str(frame.index[-1]),
        "basis": "detector reads on the frame as held at emit "
                 "time; recorded evidence only — no decision "
                 "weight",
        "chart_patterns": chart_patterns_snapshot(frame, cfg),
        "premium_discount": premium_discount_snapshot(frame),
        "breaker_blocks": breakers,
        "mitigation_blocks": mitigation,
        "relative_strength": relative_strength_snapshot(
            h1_frame, dxy_frame, cfg),
        "candles": candles_snapshot(frame)}


def census_unavailable(reason: str) -> dict:
    """The explicit not-recorded shape (warmup/replay
    evaluations, or no live frame in scope): the consumer is
    wired, this record carries no snapshot — a marked absence,
    never a fabricated read."""
    return {"status": "not_recorded", "reason": reason}
