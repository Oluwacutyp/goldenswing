"""Three trendline claims from Element Of Trading Vol V (Doctor Pips).

Source: user-mined PDFs (docs/mining/sources/element_of_trading_ii-vi.md).
Claims #29-#31 in docs/mining/CLAIMS.md.

#29 — anchor convention: the book claims shadow-to-shadow trendline
    construction gives lower-risk entries (less floating/drawdown) than
    body-to-body.  Our production lifecycle (features/trendlines.py) anchors
    on shadow extremes already (reader swings are high/low fractals).  This
    builds BOTH conventions on the same swing sequence and compares
    (a) touch quality — adverse excursion after touches, bounce rate,
    failed-touch rate; (b) entry context — anchor-signal shadow R split by
    touch proximity under each convention.

#30 — wedges as continuation: rising/falling wedges (converging support +
    resistance, both respected) signal continuation in the prior-trend
    direction.  H&S and double top/bottom were killed (#12); wedges were
    never measured.  Bounded, low-prior test: causal wedge detection,
    post-breakout drift in the prior-trend direction vs the unconditional
    prior-trend base rate, plus a traded book with Phase-2 exits.

#31 — trendline breakout-retest entry: after a trendline break, enter on the
    retest touch of the flipped (RBS/SBR) trendline.  The JeaFx zone-retest
    was killed structurally (#9) — that kill is NOT inherited; a
    trendline-retest is a different mechanic.  Break -> price moves away ->
    retest touch within 12 bars -> enter, structural stop beyond the retest
    extreme, Phase-2 exits.  If it dies too, trendline-retests are buried
    for good.

Causality: every construction uses confirmed swings only; a line becomes
known on its second anchor's confirmation bar; touches/breaks/retests are
evaluated bar-by-bar from the creation bar onward.  Forward bars are used
only to *measure* outcomes (drift, adverse excursion), never to form a
signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.ablation import P2_FLAGS
from ..backtest.data import load_gold_h1, load_peers
from ..backtest.harness import (book_metrics, build_book, simulate_signal_v2,
                                slice_stats)
from ..backtest.validation import COMBO_A, run_config
from ..features.config import FeatureConfig
from ..features.engine import read_and_featurize
from ..features.state import TrendlinesState
from ..features.trendlines import _Builder
from ..golden.detector import (DetectorConfig, GoldenSignal,
                               GoldenSwingDetector)

ANCHOR_FAMS = COMBO_A | {"aoc"}

# ------------------------------------------------------------------ builders
#: Forward window (bars) for adverse-excursion / failed-touch measurement.
TOUCH_FWD_BARS = 12
#: Touch tolerance, mirroring the production builder's tl_tol_atr.
TL_TOL_ATR = 0.20


class _BodyBuilder(_Builder):
    """Production lifecycle, but anchors on body extremes.

    Swing-high bar -> max(open, close); swing-low bar -> min(open, close).
    Validation, dedupe, lifecycle and tolerances are inherited unchanged,
    so any measured difference is the anchor convention alone.
    """

    def __init__(self, df: pd.DataFrame, cfg: FeatureConfig):
        super().__init__(df, cfg)
        self.open = df["open"].to_numpy(dtype=float)

    def _ingest(self, s) -> list[tuple]:
        new = []
        for sp in list(s.recent_swing_highs) + list(s.recent_swing_lows):
            key = (sp.kind, sp.index)
            if key not in self.seen and sp.confirmed_at <= s.index:
                self.seen.add(key)
                i = int(sp.index)
                price = (max(self.open[i], self.close[i])
                         if sp.kind == "high"
                         else min(self.open[i], self.close[i]))
                entry = (sp.kind, float(price), i, int(sp.confirmed_at))
                self.stream.append(entry)
                new.append(entry)
        return new


def build_lines(df: pd.DataFrame, states, cfg: FeatureConfig,
                mode: str) -> list[TrendlinesState]:
    """Streaming trendline states under one anchor convention."""
    assert mode in ("shadow", "body")
    builder = _Builder(df, cfg) if mode == "shadow" else _BodyBuilder(df, cfg)
    return [builder.update(t, s) for t, s in enumerate(states)]


# ------------------------------------------------------------- line indexing
def _line_key(tl) -> tuple:
    return (tl.direction, tl.anchor1_index, tl.anchor2_index)


def _price_at(tl, t: int) -> float:
    return tl.anchor1_price + tl.slope * (t - tl.anchor1_index)


def _match_snapshot(line_states, t: int, ev):
    """Snapshot whose line fired this event (exact float identity)."""
    for tl in line_states[t].active:
        if tl.direction == ev.direction \
                and abs(_price_at(tl, t) - ev.line_price) < 1e-6:
            return tl
    return None


def index_lines(line_states) -> dict:
    """key -> {tl, touches: [(bar, kind, line_price)], breaks: [bar]}.

    ``tl`` is any snapshot (anchors/slope are immutable); touches and
    breaks are matched to their line by exact price identity.
    """
    lines: dict = {}
    for t, ls in enumerate(line_states):
        for tl in ls.active:
            key = _line_key(tl)
            rec = lines.setdefault(key, {"tl": tl, "touches": [],
                                         "breaks": []})
            rec["last_seen"] = t
        for ev in ls.events:
            if ev.kind in ("touch", "bounce"):
                tl = _match_snapshot(line_states, t, ev)
                if tl is not None:
                    lines[_line_key(tl)]["touches"].append(
                        (t, ev.kind, ev.line_price))
            elif ev.kind == "break":
                tl = _match_snapshot(line_states, t, ev)
                if tl is not None:
                    lines[_line_key(tl)]["breaks"].append(t)
    return lines


def touch_bars(line_states) -> set[int]:
    """Bars with any touch/bounce event (for the entry-context split)."""
    out = set()
    for t, ls in enumerate(line_states):
        if any(ev.kind in ("touch", "bounce") for ev in ls.events):
            out.add(t)
    return out

# ============================================================ claim #29 ====
def claim29_touch_quality(df: pd.DataFrame, states, line_states) -> dict:
    """Per-convention touch quality: the book's 'less floating' claim.

    For every touch/bounce event: adverse excursion over the next 12 bars
    (ATR units) and whether price closed through the line within 12 bars
    (a failed touch).  Forward bars are outcome measurement only.
    """
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    lines = index_lines(line_states)
    adv: list[float] = []
    failed = 0
    touches = 0
    bounces = 0
    for _key, rec in lines.items():
        tl = rec["tl"]
        d = tl.direction
        for (t, kind, _lp) in rec["touches"]:
            touches += 1
            if kind == "bounce":
                bounces += 1
            atr = states[t].atr
            if atr is None or atr <= 0:
                continue
            tol = TL_TOL_ATR * atr
            worst = 0.0
            broke = False
            for k in range(1, TOUCH_FWD_BARS + 1):
                tt = t + k
                if tt >= n:
                    break
                lpk = _price_at(tl, tt)
                if d == 1:
                    worst = max(worst, (lpk - low[tt]) / atr)
                    if c[tt] < lpk - tol:
                        broke = True
                        break
                else:
                    worst = max(worst, (h[tt] - lpk) / atr)
                    if c[tt] > lpk + tol:
                        broke = True
                        break
            adv.append(worst)
            failed += 1 if broke else 0
    return {
        "n_lines": len(lines),
        "n_touches": touches,
        "n_bounces": bounces,
        "bounce_rate": bounces / touches if touches else float("nan"),
        "mean_adverse_atr": float(np.mean(adv)) if adv else float("nan"),
        "median_adverse_atr": float(np.median(adv)) if adv else float("nan"),
        "failed_touch_rate": failed / touches if touches else float("nan"),
    }


def claim29_entry_context(line_states, outcomes, window: int = 2) -> dict:
    """Anchor-signal shadow R split by trendline-touch proximity.

    A signal is 'at a touch' when a touch/bounce event fired within
    ``window`` bars before its signal bar.  Mirrors the divergence
    experiment's entry-context role.
    """
    tb = touch_bars(line_states)
    buckets = {"touch": [], "no_touch": []}
    for oc in outcomes:
        t = oc.signal.index
        near = any(b in tb for b in range(t - window, t + 1))
        buckets["touch" if near else "no_touch"].append(oc.r_net)
    return {k: _rs(v) for k, v in buckets.items()}


def _rs(rs: list[float]) -> dict:
    n = len(rs)
    if n == 0:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0, "pf": 0.0}
    a = np.array(rs)
    wins = a[a > 0].sum()
    losses = -a[a <= 0].sum()
    return {"n": n, "win_rate": float(np.mean(a > 0) * 100.0),
            "avg_r": float(a.mean()),
            "pf": float(wins / losses) if losses > 0 else float("inf")}


# ============================================================ claim #30 ====
#: Wedge geometry knobs (pre-committed, not fitted).
WEDGE_MIN_TOUCHES = 3      # combined touches across both boundary lines
WEDGE_LOOKBACK = 30        # contraction measured over this many bars
WEDGE_CONTRACT = 0.75      # dist(t) <= 0.75 * dist(t - lookback)
WEDGE_BREAK_WINDOW = 20    # bars to wait for the breakout after detection
WEDGE_DRIFT_BARS = 24      # post-breakout drift horizon
WEDGE_PRIOR_BARS = 50      # prior-trend lookback


def detect_wedges(line_states, states, df) -> list[dict]:
    """Causal wedge detection: converging support + resistance, both
    respected (>= 1 touch each), price between them, distance contracting.
    One record per boundary pair (first detection bar)."""
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    wedges: list[dict] = []
    seen: set[tuple] = set()
    start = max(WEDGE_LOOKBACK, WEDGE_PRIOR_BARS)
    for t in range(start, n):
        atr = states[t].atr
        if atr is None or atr <= 0:
            continue
        sup = [tl for tl in line_states[t].active
               if tl.direction == 1 and tl.touches >= 1
               and tl.created_at <= t - WEDGE_LOOKBACK]
        res = [tl for tl in line_states[t].active
               if tl.direction == -1 and tl.touches >= 1
               and tl.created_at <= t - WEDGE_LOOKBACK]
        for s in sup:
            for r in res:
                key = (_line_key(s), _line_key(r))
                if key in seen:
                    continue
                ps, pr = _price_at(s, t), _price_at(r, t)
                if not ps < pr:
                    continue
                if not (ps - atr <= c[t] <= pr + atr):
                    continue
                d_now = pr - ps
                d_then = (_price_at(r, t - WEDGE_LOOKBACK)
                          - _price_at(s, t - WEDGE_LOOKBACK))
                if d_then <= 0 or d_now > WEDGE_CONTRACT * d_then:
                    continue
                if s.touches + r.touches < WEDGE_MIN_TOUCHES:
                    continue
                seen.add(key)
                wtype = ("rising" if s.slope > 0 and r.slope > 0
                         else "falling" if s.slope < 0 and r.slope < 0
                         else "mixed")
                d_prior = 1 if c[t] > c[t - WEDGE_PRIOR_BARS] else -1
                wedges.append({"support": s, "resist": r, "detect_bar": t,
                               "type": wtype, "d_prior": d_prior})
    return wedges


def scan_wedge_breakouts(wedges, df, states) -> list[dict]:
    """First breakout per wedge (either direction), fully causal."""
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    out: list[dict] = []
    for w in wedges:
        s, r, t0, d_prior = (w["support"], w["resist"],
                             w["detect_bar"], w["d_prior"])
        for k in range(t0 + 1, min(t0 + WEDGE_BREAK_WINDOW, n - 1)):
            atr = states[k].atr
            if atr is None or atr <= 0:
                continue
            tol = TL_TOL_ATR * atr
            ps, pr = _price_at(s, k), _price_at(r, k)
            brk_up, brk_dn = c[k] > pr + tol, c[k] < ps - tol
            if not (brk_up or brk_dn):
                continue
            bdir = 1 if brk_up else -1
            kind = "in_trend" if bdir == d_prior else "counter"
            # drift measured in the claimed direction (prior trend) for
            # in-trend breaks; in the breakout direction for counter ones
            # (diagnostic: is it a continuation or a breakout system?).
            d = d_prior if kind == "in_trend" else bdir
            kk = min(k + WEDGE_DRIFT_BARS, n - 1)
            drift = d * (c[kk] - o[k + 1]) / atr
            out.append({"wedge": w, "break_bar": k, "kind": kind,
                        "bdir": bdir, "drift_r": float(drift),
                        "atr": float(atr)})
            break
    return out


def wedge_base_rate(df, states) -> dict:
    """Unconditional prior-trend drift: does buying the prior-trend
    direction work WITHOUT the wedge?  The honest control."""
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    drifts: list[float] = []
    for t in range(WEDGE_PRIOR_BARS, n - WEDGE_DRIFT_BARS - 1):
        atr = states[t].atr
        if atr is None or atr <= 0:
            continue
        d = 1 if c[t] > c[t - WEDGE_PRIOR_BARS] else -1
        drifts.append(d * (c[t + WEDGE_DRIFT_BARS] - o[t + 1]) / atr)
    a = np.array(drifts)
    return {"n": len(a), "mean_drift": float(a.mean()),
            "win_rate": float(np.mean(a > 0) * 100.0)}


def claim30_book(df, states, det, breakouts) -> dict:
    """Traded book: in-trend wedge breakouts, Phase-2 exits."""
    n = len(df)
    yrs = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    signals: list[GoldenSignal] = []
    for b in breakouts:
        if b["kind"] != "in_trend":
            continue
        w, k = b["wedge"], b["break_bar"]
        d = w["d_prior"]
        s = states[k]
        atr = s.atr
        # invalidation: the opposite boundary at the breakout bar
        bound = w["support"] if d == 1 else w["resist"]
        inv = _price_at(bound, k)
        if not _passes_gates(det, s, d, inv):
            continue
        pools_ahead = s.pools_above if d == 1 else s.pools_below
        signals.append(GoldenSignal(
            index=k, timestamp=s.timestamp, direction=d, score=100.0,
            kind="wedge_breakout", invalidation=float(inv),
            targets=[float(p.price) for p in pools_ahead[:3]],
            atr=float(atr), entry_ref=float(s.close)))
    outcomes = [oc for sig in signals
                if (oc := simulate_signal_v2(df, sig)) is not None]
    book = build_book(outcomes, 0.0)
    half = n // 2
    h1 = [oc for oc in outcomes if oc.signal.index < half]
    h2 = [oc for oc in outcomes if oc.signal.index >= half]
    return {"signals": signals, "outcomes": outcomes,
            "shadow": slice_stats(outcomes),
            "book": book_metrics(book, yrs),
            "book_h1": book_metrics(build_book(h1, 0.0), yrs / 2),
            "book_h2": book_metrics(build_book(h2, 0.0), yrs / 2)}

# ============================================================ claim #31 ====
#: Bars after a break in which the retest may occur.
RETEST_WINDOW = 12
#: Bars price must spend away from the broken line before a return counts
#: as a retest (mirrors the builder's tl_rearm_bars).
RETEST_REARM_BARS = 2


def _passes_gates(det: GoldenSwingDetector, s, d: int, inv: float) -> bool:
    """The detector's non-score gates: daily veto (hard — only sweep
    reversals get the major-level exception), room to the first opposing
    pool, stop-distance band.  Mirrors engulf_trigger's gates so the
    trigger/entry comparisons stay about the entry, not the plumbing."""
    cfg = det.cfg
    atr = s.atr
    if atr is None or atr <= 0:
        return False
    if cfg.daily_veto and s.daily_bias != 0 and d == -s.daily_bias:
        return False
    pools_ahead = s.pools_above if d == 1 else s.pools_below
    room_atr = abs(pools_ahead[0].price - s.close) / atr if pools_ahead \
        else 2.5
    if room_atr < cfg.min_room_atr:
        return False
    stop_atr = abs(s.close - inv) / atr
    if not (cfg.min_stop_atr <= stop_atr <= cfg.max_stop_atr):
        return False
    return True


def _make_signal(s, d: int, inv: float, kind: str) -> GoldenSignal:
    pools_ahead = s.pools_above if d == 1 else s.pools_below
    return GoldenSignal(
        index=s.index, timestamp=s.timestamp, direction=d, score=100.0,
        kind=kind, invalidation=float(inv),
        targets=[float(p.price) for p in pools_ahead[:3]],
        atr=float(s.atr), entry_ref=float(s.close))


def scan_retests(df: pd.DataFrame, states, line_states,
                 det: GoldenSwingDetector) -> list[GoldenSignal]:
    """Trendline breakout-retest entries, fully causal.

    A break event on a respected line (touches >= 1, i.e. ACTIVE) arms the
    setup in the breakout direction.  Price must first spend
    RETEST_REARM_BARS away from the broken line; the first return touch
    (wick reaches the line, close stays on the breakout side) within
    RETEST_WINDOW bars is the entry.  One trade per break.
    """
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    # broken lines awaiting a retest
    pending: list[dict] = []
    signals: list[GoldenSignal] = []
    for t in range(n):
        s = states[t]
        atr = s.atr
        # age out stale setups
        pending = [b for b in pending
                   if t - b["break_bar"] <= RETEST_WINDOW
                   and not b["traded"]]
        for ev in line_states[t].events:
            if ev.kind != "break" or ev.touches < 1:
                continue
            tl = _match_snapshot(line_states, t, ev)
            if tl is None:
                continue
            pending.append({
                "slope": tl.slope, "a1i": tl.anchor1_index,
                "a1p": tl.anchor1_price,
                "d_trade": -tl.direction,   # breakout direction
                "break_bar": t, "traded": False, "away": 0, "armed": False,
            })
        if atr is None or atr <= 0:
            continue
        tol = TL_TOL_ATR * atr
        for b in pending:
            if b["traded"]:
                continue
            lp = b["a1p"] + b["slope"] * (t - b["a1i"])
            d = b["d_trade"]
            if d == -1:  # broken support -> resistance; short the retest
                near = h[t] >= lp - tol
                away = h[t] < lp - tol
                retest = near and c[t] < lp - tol
                extreme = h[t]
            else:        # broken resistance -> support; long the retest
                near = low[t] <= lp + tol
                away = low[t] > lp + tol
                retest = near and c[t] > lp + tol
                extreme = low[t]
            if not b["armed"]:
                if away:
                    b["away"] += 1
                    if b["away"] >= RETEST_REARM_BARS:
                        b["armed"] = True
                elif near:
                    b["away"] = 0  # hovering: re-arm clock resets
                continue
            if retest:
                if t + 1 < n and _passes_gates(det, s, d, extreme):
                    signals.append(_make_signal(s, d, extreme,
                                                "trendline_retest"))
                b["traded"] = True  # one trade per break, gated or not
    return signals


def claim31_book(df: pd.DataFrame, signals: list[GoldenSignal]) -> dict:
    n = len(df)
    yrs = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
    outcomes = [oc for sig in signals
                if (oc := simulate_signal_v2(df, sig)) is not None]
    book = build_book(outcomes, 0.0)  # scores are 100: keep everything
    half = n // 2
    h1 = [oc for oc in outcomes if oc.signal.index < half]
    h2 = [oc for oc in outcomes if oc.signal.index >= half]
    reasons: dict[str, int] = {}
    for oc in outcomes:
        reasons[oc.exit_reason] = reasons.get(oc.exit_reason, 0) + 1
    return {"outcomes": outcomes, "shadow": slice_stats(outcomes),
            "book": book_metrics(book, yrs),
            "book_h1": book_metrics(build_book(h1, 0.0), yrs / 2),
            "book_h2": book_metrics(build_book(h2, 0.0), yrs / 2),
            "exit_mix": reasons}


# ================================================================ runner ====
def _years(df: pd.DataFrame) -> float:
    return (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25


def anchor_outcomes(df: pd.DataFrame, states) -> list:
    """All anchor (COMBO_A + aoc) signals simulated — the shadow universe
    for claim #29's entry-context split."""
    detector = GoldenSwingDetector(DetectorConfig(
        emit_floor=35.0, golden_threshold=60.0,
        feature_families=ANCHOR_FAMS, **P2_FLAGS))
    outcomes = []
    for sig in detector.scan(states):
        oc = simulate_signal_v2(df, sig)
        if oc is not None:
            outcomes.append(oc)
    return outcomes


def run_experiment(df: pd.DataFrame | None = None,
                   refresh: bool = False) -> dict:
    """Full pipeline: read -> featurize -> anchor check -> 3 claims."""
    if df is None:
        df = load_gold_h1(refresh=refresh)
    peers = load_peers()          # the canary: relstr needs its peers
    states = read_and_featurize(df, peers)
    cfg = FeatureConfig()
    det = GoldenSwingDetector(DetectorConfig(**P2_FLAGS))
    res: dict = {"bars": len(df),
                 "start": str(df.index[0]), "end": str(df.index[-1])}

    # ---- anchor reproduction (the canary) ----
    anchor = run_config(df, states, ANCHOR_FAMS, "anchor")
    res["anchor"] = anchor

    # ---- claim #29: anchor conventions ----
    lines_shadow = build_lines(df, states, cfg, "shadow")
    lines_body = build_lines(df, states, cfg, "body")
    outcomes = anchor_outcomes(df, states)
    res["c29"] = {
        "quality_shadow": claim29_touch_quality(df, states, lines_shadow),
        "quality_body": claim29_touch_quality(df, states, lines_body),
        "context_shadow": claim29_entry_context(lines_shadow, outcomes),
        "context_body": claim29_entry_context(lines_body, outcomes),
        "n_outcomes": len(outcomes),
    }

    # ---- claim #30: wedges ----
    wedges = detect_wedges(lines_shadow, states, df)
    breakouts = scan_wedge_breakouts(wedges, df, states)
    res["c30"] = {
        "n_wedges": len(wedges),
        "wedge_types": {w: sum(1 for x in wedges if x["type"] == w)
                        for w in ("rising", "falling", "mixed")},
        "n_breakouts": len(breakouts),
        "base_rate": wedge_base_rate(df, states),
        "book": claim30_book(df, states, det, breakouts),
        "breakouts": breakouts,
    }

    # ---- claim #31: breakout-retest ----
    retest_signals = scan_retests(df, states, lines_shadow, det)
    res["c31"] = {"n_signals": len(retest_signals),
                  **claim31_book(df, retest_signals)}
    return res
