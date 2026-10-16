"""Footprints: identify WHO is in the market, and use it.

Footprints are CONFLUENCE, never a veto.  Nothing here blocks a
trade alone — footprints route a setup (trend continuation vs
mean-reversion sweep) and size it.  The retail/CTA/central-bank
labels name the likely sponsor of a move; the score decides the
route.

- Retail Liquidity Pool / Trap: clean double/triple tops/bottoms,
  trendline liquidity, retail piles into the breakout /
  trendline bounce, volume spikes on small overlapping candles
  (drift).  Breakout attempt with body/wick < 1.0, displacement
  < 1.0 ATR in > 3 M15 bars, no FVG >= 0.25 ATR, volume < 1.2x
  (or volume spike on overlapping candles = retail churn, not
  sponsorship), retracement > 50% of the breakout leg within
  2 bars, wick back inside.  A retail trap is a SIGNAL for a
  sweep, never for chasing the breakout.  Owner note: once
  retail steps in it is often a quick bearish move — a retail
  buy-stop trap (pop above the high on low volume, long wick
  back inside) means institutional short-sellers used retail
  liquidity: sweep SHORT arms.  Retail sell-stop trap -> sweep
  LONG.
- Accumulation: tight consolidation, compression, ATR < 0.5x
  average ATR, volume < 0.8x average, range width < 1.5 ATR,
  during off-peak (Asian 00:00-06:00 UTC), duration >= 6 H1 bars.
- Manipulation / Judas Swing: sudden sweep outside the Asian
  range right before London (02:00-05:00 NY) / NY (07:00-10:00
  NY) open, wick beyond the Asian high/low >= 0.25 ATR, close
  back inside the Asian range, volume spike.  The engine sits
  out the manipulation, lets retail get trapped, and catches
  the distribution leg (STATE 3 displacement after Judas).
- Distribution: true institutional impulse — displacement
  >= 1.5 ATR in <= 3 M15 bars, FVG >= 0.25 ATR, CHoCH/BOS,
  volume >= 1.5x.
- CTA / Algo Momentum: fast body-dominated candles, body/wick
  >= 1.5, displacement >= 2 ATR in <= 2 M15 bars, volume >= 2x,
  clean break of multi-bar structure without deep retracement
  (retracement < 38% Fib of the breakout leg), no FVG fill.
  BREAKOUT_TREND path: scale in on retests (retest holds as
  S/R, or after 4 H1 bars without retest, BOS + 2 closes at
  0.5x size).  Do NOT fight momentum, do NOT mean-revert
  against CTA.
- Central Bank / Macro: large displacement on high-impact news
  (NFP, CPI, FOMC, central bank rate), volume >= 3x average,
  displacement >= 3 ATR in <= 1 H1 bar, news window.  New
  entries pause 15 min before/after high-impact news (existing
  rule); after news, if displacement + CHoCH + FVG, trade with
  wider SL (news volatility), size -20%.

FOOTPRINT SCORE — weak retail breakout vs strong institutional
momentum breakout, one composite number, no conflicting signals:
- Displacement: >= 2 ATR +20, 1.5-2 +15, 1.0-1.5 +5, < 1.0 0
- Speed: <= 2 M15 bars +15, 3 bars +10, > 3 bars 0
- Body/wick: >= 1.5 +15, 1.0-1.5 +5, < 1.0 0 (wick = retail)
- Volume: >= 2x +15, 1.5-2x +10, 1.2-1.5x +5, < 1.2x 0
- FVG: >= 0.25 ATR +15, < 0.25 ATR +5, none 0
- BOS/CHoCH: MSB +10, BOS or CHoCH +5, none 0
- Retracement: < 38% Fib +10, 38-50% +5, > 50% 0 (deep = retail)
- Liquidity: breakout takes a pool +5, no pool 0
- >= 70 = INSTITUTIONAL MOMENTUM BREAKOUT -> BREAKOUT_TREND,
  trend continuation, scale in on retests, do NOT mean-revert.
- 40-69 = AMBIGUOUS -> the decisive next-bar test
  (``live/nextbar.py``) casts the vote, never a guess.
- < 40 = WEAK RETAIL BREAKOUT -> MEAN-REVERSION SWEEP path
  (Turtle Soup / sweep setup), do NOT chase the breakout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .multitf import find_swings
from .patterns import session_label, trend_bias
from .zones import SWING_N

# Footprint score thresholds.
INST_MIN = 70.0
AMBIG_MIN = 40.0
NEWS_DISP_ATR = 3.0
NEWS_VOLUME_MIN = 3.0
CTA_DISP_ATR = 2.0
CTA_VOLUME_MIN = 2.0
CTA_BODY_WICK_MIN = 1.5
CTA_MAX_RETRACE = 0.38


@dataclass
class FootprintScore:
    """Composite footprint score for one breakout attempt."""

    total: float = 0.0
    components: dict = field(default_factory=dict)
    classification: str = "retail"  # institutional | ambiguous | retail
    route: str = "sweep"  # breakout_trend | wait_next_bar | sweep
    log: str = ""
    # The attempt window + FVG, so callers (narrative STATE 2)
    # can hand the attempt to the decisive next-bar test.
    attempt_start: int | None = None
    attempt_end: int | None = None
    fvg: dict | None = None


def _rows(frame):
    return frame


def _body_wick(row) -> float:
    o, h, l, c = (float(row["open"]), float(row["high"]),
                  float(row["low"]), float(row["close"]))
    body = abs(c - o)
    wick = (h - max(o, c)) + (min(o, c) - l)
    return body / wick if wick > 0 else float("inf")


def _mean_volume(frame, upto: int, *, window: int = 20) -> float:
    if "volume" not in frame.columns or upto <= 0:
        return 0.0
    try:
        vols = frame["volume"].to_numpy(dtype=float)
        return float(np.mean(vols[max(0, upto - window):upto]))
    except Exception:
        return 0.0


def _gap(frame, start: int, end: int, direction: int,
         atr: float) -> dict | None:
    """First FVG gap in [start, end] for ``direction``."""
    if frame is None or len(frame) < 3 or atr <= 0:
        return None
    h = frame["high"].to_numpy(dtype=float)
    l = frame["low"].to_numpy(dtype=float)
    for i in range(max(2, start), min(end, len(frame) - 1) + 1):
        if direction > 0 and float(l[i]) > float(h[i - 2]):
            return {"top": float(l[i]), "bottom": float(h[i - 2]),
                    "size_atr": (float(l[i]) - float(h[i - 2])) / atr,
                    "bar_pos": i}
        if direction < 0 and float(h[i]) < float(l[i - 2]):
            return {"top": float(l[i - 2]), "bottom": float(h[i]),
                    "size_atr": (float(l[i - 2]) - float(h[i])) / atr,
                    "bar_pos": i}
    return None


def _attempt_window(frame, edge: float, breakout_dir: int,
                    *, lookback: int = 8) -> tuple[int, int] | None:
    """(start, end) positions of the breakout attempt: from the
    first bar closing beyond ``edge`` (scanning back) to the
    last bar.  None when no bar closed beyond."""
    n = len(frame)
    c = frame["close"].to_numpy(dtype=float)
    start = None
    for i in range(n - 1, max(-1, n - 1 - lookback), -1):
        beyond = (float(c[i]) - edge) * breakout_dir
        if beyond > 0:
            start = i
    if start is None:
        return None
    return start, n - 1


def score_breakout(frame, edge: float, breakout_dir: int,
                   atr: float) -> FootprintScore:
    """Composite footprint score for a breakout attempt.

    ``frame`` = closed bars ending at the attempt; ``edge`` =
    the broken level; ``breakout_dir`` = +1 up / -1 down.
    """
    out = FootprintScore()
    if frame is None or len(frame) < 2 or atr <= 0:
        out.log = "no bars for footprint score"
        return out
    win = _attempt_window(frame, edge, breakout_dir)
    if win is None:
        out.log = "no breakout attempt (no close beyond edge)"
        return out
    start, end = win
    c = frame["close"].to_numpy(dtype=float)
    n_bars = end - start + 1
    out.attempt_start, out.attempt_end = start, end
    # Displacement: farthest close in direction vs attempt start.
    leg = c[start:end + 1]
    extreme = float(np.max(leg)) if breakout_dir > 0 \
        else float(np.min(leg))
    first = float(c[start])
    disp = abs(extreme - first) / atr
    comp: dict[str, float] = {}
    comp["displacement"] = 20.0 if disp >= 2.0 else (
        15.0 if disp >= 1.5 else (5.0 if disp >= 1.0 else 0.0))
    comp["speed"] = 15.0 if n_bars <= 2 else (
        10.0 if n_bars == 3 else 0.0)
    bws = [_body_wick(frame.iloc[i]) for i in range(start, end + 1)]
    finite = [b for b in bws if b != float("inf")]
    bw = float(np.mean(finite)) if finite else float("inf")
    comp["body_wick"] = 15.0 if bw >= 1.5 else (
        5.0 if bw >= 1.0 else 0.0)
    mean_v = _mean_volume(frame, start)
    vol_ratio = None
    if mean_v > 0 and "volume" in frame.columns:
        vols = frame["volume"].to_numpy(dtype=float)
        vol_ratio = float(np.max(vols[start:end + 1])) / mean_v
        comp["volume"] = 15.0 if vol_ratio >= 2.0 else (
            10.0 if vol_ratio >= 1.5 else (
                5.0 if vol_ratio >= 1.2 else 0.0))
    else:
        comp["volume"] = 0.0
    gap = _gap(frame, start, end, breakout_dir, atr)
    out.fvg = gap
    if gap is None:
        comp["fvg"] = 0.0
    elif gap["size_atr"] >= 0.25:
        comp["fvg"] = 15.0
    else:
        comp["fvg"] = 5.0
    # BOS / CHoCH / MSB vs the last opposing swing before the
    # attempt, and the older swing for MSB.
    bos = choch = msb = False
    if start > 2 * SWING_N + 5:
        sub = frame.iloc[:start]
        try:
            highs, lows = find_swings(
                sub["high"].to_numpy(dtype=float),
                sub["low"].to_numpy(dtype=float), SWING_N)
            swings = highs if breakout_dir > 0 else lows
            if swings:
                lvl = float(swings[-1][1])
                bos = (extreme > lvl) if breakout_dir > 0 \
                    else (extreme < lvl)
                if bos and len(swings) >= 2:
                    lvl2 = float(swings[-2][1])
                    msb = (extreme > lvl2) if breakout_dir > 0 \
                        else (extreme < lvl2)
                if bos:
                    try:
                        trend = int(
                            trend_bias(sub)["bias"])
                        choch = trend == -breakout_dir
                    except Exception:
                        choch = False
        except Exception:
            pass
    comp["structure"] = 10.0 if msb else (5.0 if bos else 0.0)
    # Retracement of the leg, as a fraction of the leg move.
    leg_move = abs(extreme - first)
    retr = abs(extreme - float(c[end])) / leg_move \
        if leg_move > 0 else 1.0
    comp["retracement"] = 10.0 if retr < 0.38 else (
        5.0 if retr <= 0.50 else 0.0)
    # Liquidity: a prior swing within 0.5 ATR of the edge.
    pool = False
    if start > 2 * SWING_N + 5:
        sub = frame.iloc[:start]
        try:
            highs, lows = find_swings(
                sub["high"].to_numpy(dtype=float),
                sub["low"].to_numpy(dtype=float), SWING_N)
            pool = any(abs(float(p) - edge) <= 0.5 * atr
                       for _, p in list(highs) + list(lows))
        except Exception:
            pass
    comp["liquidity"] = 5.0 if pool else 0.0
    total = min(100.0, float(sum(comp.values())))
    out.total = total
    out.components = comp
    if total >= INST_MIN:
        out.classification, out.route = "institutional", \
            "breakout_trend"
    elif total >= AMBIG_MIN:
        out.classification, out.route = "ambiguous", "wait_next_bar"
    else:
        out.classification, out.route = "retail", "sweep"
    out.log = (f"footprint score {total:.0f} "
               f"({out.classification}): disp {disp:.1f} ATR, "
               f"{n_bars} bars, body/wick {bw:.1f}, volume "
               f"{vol_ratio if vol_ratio is not None else 0:.1f}x, "
               f"FVG {comp['fvg']:.0f}, structure {comp['structure']:.0f}, "
               f"retracement {retr * 100:.0f}%")
    return out


def detect_retail_trap(frame, *, direction: int, atr: float) -> dict:
    """Retail stop trap for the trade ``direction`` (sweep setup).

    Short setups (direction -1): the last bar pops above the
    prior swing high on a long wick and closes back below it,
    small body, no displacement — retail buy stops triggered,
    institutional distribution likely.  Long setups mirror it.
    """
    out = {"present": False, "kind": "", "setup_direction": direction,
           "log": ""}
    if frame is None or len(frame) < 2 * SWING_N + 5 or atr <= 0:
        return out
    last = frame.iloc[-1]
    sub = frame.iloc[:-1]
    try:
        highs, lows = find_swings(
            sub["high"].to_numpy(dtype=float),
            sub["low"].to_numpy(dtype=float), SWING_N)
    except Exception:
        return out
    if direction < 0:
        if not highs:
            return out
        level = float(highs[-1][1])
        wick_beyond = (float(last["high"]) - level) / atr
        closed_back = float(last["close"]) < level
        disp = (float(last["close"]) - float(sub["close"].iloc[-1])) \
            / atr
        kind = "retail_buy_stop_trap"
    else:
        if not lows:
            return out
        level = float(lows[-1][1])
        wick_beyond = (level - float(last["low"])) / atr
        closed_back = float(last["close"]) > level
        disp = (float(sub["close"].iloc[-1]) - float(last["close"])) \
            / atr
        kind = "retail_sell_stop_trap"
    if wick_beyond < 0.25 or not closed_back:
        return out
    bw = _body_wick(last)
    if bw >= 1.0:
        return out
    if abs(disp) >= 1.0:
        return out
    mean_v = _mean_volume(frame, len(frame) - 1)
    vol_ratio = None
    if mean_v > 0 and "volume" in frame.columns:
        vol_ratio = float(last["volume"]) / mean_v
    churn = vol_ratio is not None and vol_ratio >= 1.2 and bw < 0.5
    if vol_ratio is not None and vol_ratio >= 1.2 and not churn:
        return out
    out.update({"present": True, "kind": kind, "level": level,
                "wick_beyond_atr": wick_beyond,
                "body_wick": bw, "volume_ratio": vol_ratio,
                "log": (f"retail trap: {kind} at {level:.1f}, "
                        f"wick {wick_beyond:.2f} ATR, close back "
                        f"inside, body/wick {bw:.2f}"
                        + (", volume churn" if churn else "")
                        + (" — institutional distribution "
                           "likely, sweep "
                           + ("SHORT" if direction < 0 else "LONG")
                           + " arms"))})
    return out


def detect_accumulation(frame, *, atr: float) -> dict:
    """Tight consolidation / compression (STATE 0/1 context)."""
    out = {"present": False, "log": ""}
    if frame is None or len(frame) < 12 or atr <= 0:
        return out
    win = frame.iloc[-12:]
    width = float(win["high"].max() - win["low"].min()) / atr
    ranges = (win["high"] - win["low"]).to_numpy(dtype=float)
    mean_range = float(np.mean(ranges))
    vol_ratio = None
    if "volume" in frame.columns:
        mean_v = _mean_volume(frame, len(frame) - 12, window=20)
        if mean_v > 0:
            vol_ratio = float(np.mean(
                win["volume"].to_numpy(dtype=float))) / mean_v
    asian = False
    try:
        ts = pd.Timestamp(frame.index[-1])
        utc = ts.tz_convert("UTC") if ts.tzinfo is not None \
            else ts.tz_localize("UTC")
        asian = 0 <= utc.hour < 6
    except Exception:
        pass
    if width < 1.5 and mean_range < 0.5 * atr and \
            (vol_ratio is None or vol_ratio < 0.8):
        out.update({"present": True, "width_atr": width,
                    "volume_ratio": vol_ratio, "asian": asian,
                    "log": (f"accumulation: compression, low "
                            f"ATR/volume, range "
                            f"{float(win['low'].min()):.1f}-"
                            f"{float(win['high'].max()):.1f}"
                            + (" (Asian)" if asian else ""))})
    return out


def detect_judas(frame, *, asian_high: float, asian_low: float,
                 atr: float) -> dict:
    """Judas swing: sweep outside the Asian range right before
    the London / NY open, close back inside, volume spike."""
    out = {"present": False, "log": ""}
    if frame is None or len(frame) < 2 or atr <= 0:
        return out
    last = frame.iloc[-1]
    try:
        sess = session_label(frame.index[-1])
    except Exception:
        sess = {"killzone": None}
    if sess.get("killzone") not in ("london_killzone", "ny_killzone"):
        return out
    mean_v = _mean_volume(frame, len(frame) - 1)
    vol_ratio = float(last["volume"]) / mean_v \
        if mean_v > 0 and "volume" in frame.columns else None
    if vol_ratio is not None and vol_ratio < 1.2:
        return out
    if float(last["high"]) >= asian_high + 0.25 * atr and \
            float(last["close"]) < asian_high:
        side = "high"
    elif float(last["low"]) <= asian_low - 0.25 * atr and \
            float(last["close"]) > asian_low:
        side = "low"
    else:
        return out
    out.update({"present": True, "side": side,
                "volume_ratio": vol_ratio,
                "killzone": sess.get("killzone"),
                "log": (f"Judas swing: sweep Asian {side}, "
                        "manipulation, retail trapped — sit out, "
                        "catch the distribution leg")})
    return out


def detect_distribution(frame, *, direction: int, atr: float) -> dict:
    """True institutional impulse (STATE 3 shape, no sweep
    required — the footprint view of displacement)."""
    out = {"present": False, "log": ""}
    if frame is None or len(frame) < 5 or atr <= 0:
        return out
    c = frame["close"].to_numpy(dtype=float)
    start = float(c[-4])
    leg = c[-3:]
    extreme = float(np.max(leg)) if direction > 0 \
        else float(np.min(leg))
    move = (extreme - start) * direction / atr
    if move < 1.5:
        return out
    gap = _gap(frame, len(frame) - 3, len(frame) - 1, direction, atr)
    if gap is None or gap["size_atr"] < 0.25:
        return out
    mean_v = _mean_volume(frame, len(frame) - 3)
    vol_ratio = None
    if mean_v > 0 and "volume" in frame.columns:
        vols = frame["volume"].to_numpy(dtype=float)
        vol_ratio = float(np.max(vols[-3:])) / mean_v
    if vol_ratio is not None and vol_ratio < 1.5:
        return out
    out.update({"present": True, "move_atr": move, "fvg": gap,
                "volume_ratio": vol_ratio,
                "log": (f"distribution: displacement "
                        f"{move:.1f} ATR + FVG + volume "
                        f"{vol_ratio if vol_ratio is not None else 0:.1f}x"
                        " — institutional impulse")})
    return out


def detect_cta(frame, *, direction: int, atr: float) -> dict:
    """CTA / algo momentum: body-dominated, fast, no deep
    retracement, no FVG fill.  Never mean-revert against it."""
    out = {"present": False, "log": ""}
    if frame is None or len(frame) < 4 or atr <= 0:
        return out
    c = frame["close"].to_numpy(dtype=float)
    start = float(c[-3])
    leg = c[-2:]
    extreme = float(np.max(leg)) if direction > 0 \
        else float(np.min(leg))
    move = (extreme - start) * direction / atr
    if move < CTA_DISP_ATR:
        return out
    bws = [_body_wick(frame.iloc[i]) for i in range(len(frame) - 2,
                                                    len(frame))]
    finite = [b for b in bws if b != float("inf")]
    bw = float(np.mean(finite)) if finite else float("inf")
    if bw < CTA_BODY_WICK_MIN:
        return out
    mean_v = _mean_volume(frame, len(frame) - 2)
    vol_ratio = None
    if mean_v > 0 and "volume" in frame.columns:
        vols = frame["volume"].to_numpy(dtype=float)
        vol_ratio = float(np.max(vols[-2:])) / mean_v
    if vol_ratio is not None and vol_ratio < CTA_VOLUME_MIN:
        return out
    retr = abs(extreme - float(c[-1])) / abs(extreme - start) \
        if extreme != start else 1.0
    if retr >= CTA_MAX_RETRACE:
        return out
    gap = _gap(frame, len(frame) - 2, len(frame) - 1, direction, atr)
    if gap is not None:
        # A filled FVG kills the CTA read (price retraced in).
        ftop, fbot = float(gap["top"]), float(gap["bottom"])
        if bool(np.any((c >= fbot) & (c <= ftop))):
            return out
    out.update({"present": True, "move_atr": move,
                "body_wick": bw, "volume_ratio": vol_ratio,
                "retracement": retr,
                "log": (f"CTA momentum: body-dominated, volume "
                        f"spike, no retracement — BREAKOUT_TREND "
                        "path, do NOT mean-revert")})
    return out


def detect_macro_news(frame, *, atr: float, news: bool = False,
                      news_ts=None) -> dict:
    """Central bank / macro footprint: >= 3 ATR in <= 1 bar on
    >= 3x volume inside a high-impact news window."""
    out = {"present": False, "log": ""}
    if not news or frame is None or len(frame) < 2 or atr <= 0:
        return out
    sub = frame
    if news_ts is not None:
        try:
            sub = frame[frame.index >= pd.Timestamp(news_ts)]
        except Exception:
            pass
    if len(sub) == 0:
        return out
    ranges = (sub["high"] - sub["low"]).to_numpy(dtype=float)
    i = int(np.argmax(ranges))
    if ranges[i] < NEWS_DISP_ATR * atr:
        return out
    mean_v = _mean_volume(frame, len(frame) - len(sub) + i)
    vol_ratio = None
    if mean_v > 0 and "volume" in frame.columns:
        vol_ratio = float(sub["volume"].iloc[i]) / mean_v
    if vol_ratio is not None and vol_ratio < NEWS_VOLUME_MIN:
        return out
    out.update({"present": True, "bar_ts": str(sub.index[i]),
                "range_atr": float(ranges[i]) / atr,
                "volume_ratio": vol_ratio,
                "log": (f"macro news displacement, central bank / "
                        "macro fund footprint, news — wider SL, "
                        "size -20%")})
    return out


def classify_footprint(frame, *, direction: int, atr: float,
                       asian_range=None, news: bool = False,
                       news_ts=None) -> dict:
    """Dominant footprint for the current bars (priority:
    macro news -> CTA -> distribution -> Judas -> accumulation
    -> retail trap).  Confluence, never a veto."""
    macro = detect_macro_news(frame, atr=atr, news=news,
                              news_ts=news_ts)
    if macro["present"]:
        return {"footprint": "macro_news",
                "route": "breakout_trend_wide_sl",
                "log": macro["log"], "detail": macro}
    cta = detect_cta(frame, direction=direction, atr=atr)
    if cta["present"]:
        return {"footprint": "cta_momentum",
                "route": "breakout_trend",
                "log": cta["log"], "detail": cta}
    dist = detect_distribution(frame, direction=direction, atr=atr)
    if dist["present"]:
        return {"footprint": "distribution",
                "route": "trend_continuation",
                "log": dist["log"], "detail": dist}
    if asian_range is not None:
        judas = detect_judas(frame, asian_high=float(asian_range[0]),
                             asian_low=float(asian_range[1]), atr=atr)
        if judas["present"]:
            return {"footprint": "judas_manipulation",
                    "route": "sit_out_await_distribution",
                    "log": judas["log"], "detail": judas}
    acc = detect_accumulation(frame, atr=atr)
    if acc["present"]:
        return {"footprint": "accumulation",
                "route": "await_manipulation",
                "log": acc["log"], "detail": acc}
    trap = detect_retail_trap(frame, direction=direction, atr=atr)
    if trap["present"]:
        return {"footprint": "retail_trap",
                "route": "sweep",
                "log": trap["log"], "detail": trap}
    return {"footprint": "none", "route": "none",
            "log": "no dominant footprint", "detail": {}}
