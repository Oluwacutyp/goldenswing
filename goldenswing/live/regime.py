"""Bedrock regime engine for XAUUSD — honest, early, no circular logic.

Bedrock answers ONE question: "What is the market's current personality?"

It does NOT decide entries, does NOT use FVG / Order Block / Fibonacci /
any entry tool, does NOT score trades, and does NOT grant trade
permission.  Router, context, entry, and process score live on top of it.

Inputs are OHLC bars only, plus the bar timestamp for session:
- market structure (swings: HH/HL vs LH/LL vs equal/overlapping)
- candle character over the last 10-20 bars (bodies vs dojis vs wicks)
- volatility (ATR behaviour: expanding / stable / contracting / spike)
- two windows: short (last 40 bars) = current personality,
  long (last 80 bars) = is this a real shift or just noise.
  Candle character uses the last 20 bars (freshest personality).

Honesty rules (user-explicit, do not weaken):
1. Structure leads.  If swing structure is unclear, regime is Unclear.
2. Short window and long window must agree on direction.  Disagreement
   defaults to Transition / Unclear — never pick the window you like.
3. Volatility must confirm structure.  Trend + compressed ATR is
   suspicious.  Range + very low ATR is Compression, not a tradeable Range.
4. Candle character is the truth serum.  Strong bodies one way support
   Trend.  Dojis / small bodies / long opposing wicks argue against it.
5. Permission to say "I don't know".  Transition / Unclear is a
   successful decision, not a failure.  Forcing a label when mixed is
   how bedrock starts lying.
6. No circular logic.  Regime is readable with FVGs, OBs, Fibs, and all
   entry concepts removed.  Those tools live upstairs as context /
   confirmation, never inside bedrock.
7. Early, not late.  First break in structure behaviour, candle
   character change before displacement, ATR starting to expand /
   contract, HTF/LTF divergence — whispers, not shouts.  Waiting for
   multiple confirmed swings / completed patterns / aligned indicators
   is late and useless.

Regime is a COMBINATION, not one of 5 buckets (user-explicit: no
capping at 5, no laziness):
    Regime = Direction x Structure x Volatility x Phase x Session
- Direction: up / down / flat
- Structure: clean_trend / range / compression / breakout / reversal /
  choppy / unclear
- Volatility: low / normal / high / spike / unclear
- Phase (AMD): accumulation / manipulation / distribution /
  transition / unclear
- Session: asia / london / ny / overlap / offhours / unknown

Every real combination gets a routing, or an explicit SIT OUT with a
reason.  Unlisted combinations default to SIT OUT / Unclear — never a
forced label.

Tools are dual-role (user-explicit): the SAME tool is a SENSOR in
bedrock and CONFIRMATION in the trigger stage.  Engulfing candles
everywhere = trending/expansion sensor.  Engulfing at a pre-marked POI
after sweep + displacement = entry confirmation.  Locking a tool to
one job is as dumb as locking it to entry only.

Veto stays OFF (2026-10-13 backtest: hard veto mostly hurt, removed
winners, did not reduce blow-ups).  Bedrock does not implement veto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .multitf import atr14, evaluate_bias, find_swings

SHORT_WINDOW = 40  # structure needs ~2 confirmed swings each side (n=3)
LONG_WINDOW = 80
CANDLE_WINDOW = 20  # candle character: last 20 bars (current personality)
SWING_N = 3

# ATR ratio thresholds: short-window ATR vs long-window ATR.
ATR_EXPANDING = 1.25
ATR_CONTRACTING = 0.75
ATR_SPIKE = 1.80

# Candle-character thresholds (fractions over the short window).
STRONG_BODY_MIN = 0.55  # |close-open| / range >= this counts as strong body
DOJI_MAX = 0.15  # body ratio <= this counts as doji / indecision


@dataclass(frozen=True)
class Regime:
    """One bedrock classification.  Frozen: bedrock never mutates state."""

    direction: str  # "up" | "down" | "flat"
    structure: str  # clean_trend | range | compression | breakout |
    # reversal | choppy | unclear
    volatility: str  # low | normal | high | spike | unclear
    phase: str  # accumulation | manipulation | distribution |
    # transition | unclear
    session: str  # asia | london | ny | overlap | offhours | unknown
    confidence: float  # 0.0 - 1.0, low means "do not trust this label"
    reason: str
    route: str  # core name or "SIT_OUT"
    tradeable: bool = False
    short_bias: int = 0
    long_bias: int = 0

    def label(self) -> str:
        return (
            f"{self.direction}/{self.structure}/{self.volatility}/"
            f"{self.phase}/{self.session}"
        )


# ---------------------------------------------------------------------------
# Helpers — all pure, OHLC-only, no entry tools.
# ---------------------------------------------------------------------------

def session_of(ts) -> str:
    """Session from a UTC timestamp.  Unknown when no timestamp."""
    try:
        hour = pd.Timestamp(ts).tz_convert("UTC").hour if pd.Timestamp(ts).tzinfo is not None else pd.Timestamp(ts).hour
    except Exception:
        return "unknown"
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 16:
        return "overlap"
    if 16 <= hour < 21:
        return "ny"
    return "offhours"


def _candle_stats(bars: pd.DataFrame) -> dict:
    """Body / doji / wick fractions over the given bars."""
    if len(bars) == 0:
        return {"strong": 0.0, "doji": 0.0, "wicky": 0.0, "avg_body": 0.0}
    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    l = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    rng = np.maximum(h - l, 1e-9)
    body = np.abs(c - o) / rng
    upper_wick = (h - np.maximum(o, c)) / rng
    lower_wick = (np.minimum(o, c) - l) / rng
    strong = float(np.mean(body >= STRONG_BODY_MIN))
    doji = float(np.mean(body <= DOJI_MAX))
    wicky = float(np.mean((upper_wick >= 0.45) | (lower_wick >= 0.45)))
    return {
        "strong": strong,
        "doji": doji,
        "wicky": wicky,
        "avg_body": float(np.mean(body)),
    }


def _sweep_in_window(bars: pd.DataFrame, swing_n: int = SWING_N) -> bool:
    """True if a wick swept beyond a prior confirmed swing extreme and
    closed back inside, in the given window.  Wick-only detection:
    no FVG, no OB, no entry logic."""
    if len(bars) < 2 * swing_n + 3:
        return False
    highs, lows = find_swings(bars["high"], bars["low"], swing_n)
    if not highs or not lows:
        return False
    # Use the earliest confirmed extremes in the window as the levels
    # that could be swept by later bars.
    first_high = highs[0][1]
    first_low = lows[0][1]
    first_high_i = highs[0][0]
    first_low_i = lows[0][0]
    hi = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    cl = bars["close"].to_numpy(dtype=float)
    for i in range(max(first_high_i, first_low_i) + 1, len(bars)):
        swept_high = hi[i] > first_high and cl[i] < first_high
        swept_low = lo[i] < first_low and cl[i] > first_low
        if swept_high or swept_low:
            return True
    return False


def _displacement_in_window(bars: pd.DataFrame) -> bool:
    """True if at least one strong-body candle moved decisively in the
    window (body >= 0.55 of range and range >= 0.8x ATR).  Displacement
    is a candle fact, not an entry signal."""
    if len(bars) < 2:
        return False
    atr = atr14(bars)
    if atr <= 0:
        return False
    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    l = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    rng = h - l
    body = np.abs(c - o)
    return bool(np.any((body / np.maximum(rng, 1e-9) >= STRONG_BODY_MIN) & (rng >= 0.8 * atr)))


def _range_width(bars: pd.DataFrame) -> float:
    if len(bars) == 0:
        return 0.0
    return float(bars["high"].max() - bars["low"].min())


def _structure_of(bars: pd.DataFrame) -> tuple[str, int]:
    """Structure label + bias for one window.  Returns (label, bias)."""
    bias = evaluate_bias(bars, swing_n=SWING_N)
    highs, lows = find_swings(bars["high"], bars["low"], SWING_N)
    if len(highs) < 2 or len(lows) < 2:
        return "unclear", 0
    # Equal / overlapping extremes: highs and lows both flat.
    h_flat = abs(highs[-1][1] - highs[-2][1]) <= 1e-9
    l_flat = abs(lows[-1][1] - lows[-2][1]) <= 1e-9
    if bias != 0:
        return "clean_trend", bias
    # No directional bias: distinguish range / compression / choppy by
    # width vs ATR and candle character (caller refines with volatility).
    width = _range_width(bars)
    atr = atr14(bars)
    if atr > 0 and width <= 6.0 * atr:
        return "compression", 0
    return "range", 0


def _volatility_of(short_bars: pd.DataFrame, long_bars: pd.DataFrame) -> str:
    """Volatility from ATR behaviour, not from price direction."""
    atr_s = atr14(short_bars)
    atr_l = atr14(long_bars)
    if atr_s <= 0 or atr_l <= 0:
        return "unclear"
    ratio = atr_s / atr_l
    # Spike: last bar range far above long ATR (news / shock).
    last_range = float(short_bars["high"].iloc[-1] - short_bars["low"].iloc[-1])
    if last_range >= 2.5 * atr_l or ratio >= ATR_SPIKE:
        return "spike"
    if ratio >= ATR_EXPANDING:
        return "high"
    if ratio <= ATR_CONTRACTING:
        return "low"
    return "normal"


def _phase_of(short_bars: pd.DataFrame, swept: bool, displaced: bool,
               structure: str, volatility: str) -> str:
    """AMD phase from sweep / displacement / structure facts."""
    if swept and displaced:
        return "distribution"
    if swept and not displaced:
        return "manipulation"
    if structure in ("range", "compression") and volatility in ("low", "normal"):
        return "accumulation"
    if structure == "clean_trend" and displaced:
        return "distribution"
    if structure == "unclear":
        return "transition"
    return "transition"


# ---------------------------------------------------------------------------
# Routing — regime combination -> core or SIT OUT.
# Every real combination is covered by these rules; anything not matched
# defaults to SIT OUT / Unclear.  No forced labels.
# ---------------------------------------------------------------------------

def route_for_regime(direction: str, structure: str, volatility: str,
                     phase: str, session: str) -> tuple[str, bool, str]:
    """Return (route, tradeable, reason).  Pure function, no I/O."""
    # 1. Unclear structure always sits out.
    if structure == "unclear":
        return "SIT_OUT", False, "structure unclear"
    # 2. Spike / news volatility: sit out or tiny size.  Bedrock never
    #    grants normal size in a spike.
    if volatility == "spike":
        return "SIT_OUT", False, "spike/news volatility — sit out"
    # 3. Choppy: no core fits.
    if structure == "choppy":
        return "SIT_OUT", False, "choppy — no core fits"
    # 4. Compression: wait for expansion, do not trade the coil.
    if structure == "compression":
        return "SIT_OUT", False, "compression — wait for expansion"
    # 5. Clean trend: trend + pullback core, in trend direction.
    if structure == "clean_trend":
        if direction == "flat":
            return "SIT_OUT", False, "trend structure without direction"
        if phase == "manipulation":
            return "SIT_OUT", False, "trend + manipulation — wait for displacement"
        if volatility == "low":
            return "SIT_OUT", False, "trend + compressed ATR — suspicious trap"
        return "TREND_PULLBACK", True, "clean trend + healthy vol"
    # 6. Range: sweep / rejection at edges.
    if structure == "range":
        if phase == "accumulation":
            return "SIT_OUT", False, "range + accumulation — wait for sweep"
        if phase == "manipulation":
            return "SWEEP_DISPLACEMENT_RETRACEMENT", True, "range + sweep — wait displacement + retracement"
        if phase == "distribution":
            return "SWEEP_DISPLACEMENT_RETRACEMENT", True, "range + distribution after sweep"
        return "RANGE_EDGES", True, "range edges"
    # 7. Breakout: breakout + retest core, only after expansion.
    if structure == "breakout":
        if volatility == "low":
            return "SIT_OUT", False, "breakout + low vol — false break likely"
        return "BREAKOUT_RETEST", True, "breakout + expansion — wait retest"
    # 8. Reversal: sweep + displacement + retracement, counter prior move.
    if structure == "reversal":
        if phase == "manipulation":
            return "SIT_OUT", False, "reversal + manipulation — wait displacement"
        return "SWEEP_DISPLACEMENT_RETRACEMENT", True, "reversal after displacement"
    return "SIT_OUT", False, "unmatched combination — sit out"


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------

def classify_regime(bars: pd.DataFrame) -> Regime:
    """Classify the regime of the last bars.  Pure, OHLC-only.

    ``bars`` must have open/high/low/close columns and a DatetimeIndex
    (for session).  Returns Transition/Unclear with low confidence
    whenever the evidence is mixed, short/long windows disagree, or
    there is not enough data.  Never forces a label.
    """
    if bars is None or len(bars) < SHORT_WINDOW + 5:
        return Regime(
            direction="flat", structure="unclear", volatility="unclear",
            phase="unclear", session="unknown", confidence=0.0,
            reason="not enough bars",
            route="SIT_OUT", tradeable=False,
        )
    short_bars = bars.iloc[-SHORT_WINDOW:]
    long_bars = bars.iloc[-LONG_WINDOW:] if len(bars) >= LONG_WINDOW else bars
    session = session_of(bars.index[-1])

    short_struct, short_bias = _structure_of(short_bars)
    long_struct, long_bias = _structure_of(long_bars)

    # Two-window disagreement on DIRECTION -> usually Unclear, EXCEPT
    # the reversal case: long trend + short sweep + short displacement
    # in the OPPOSITE direction.  That is a genuine reversal forming,
    # not noise.  It gets a reversal label with modest confidence, and
    # routes to sweep->displacement->retracement (counter prior move).
    # Without sweep + displacement, disagreement stays Unclear.
    if short_bias != 0 and long_bias != 0 and short_bias != long_bias:
        _swept_early = _sweep_in_window(short_bars)
        _disp_early = _displacement_in_window(short_bars)
        if _swept_early and _disp_early:
            _rev_dir = "up" if short_bias == 1 else "down"
            _vol_early = _volatility_of(short_bars, long_bars)
            _route, _trad, _rr = route_for_regime(
                _rev_dir, "reversal", _vol_early, "distribution", session)
            return Regime(
                direction=_rev_dir, structure="reversal",
                volatility=_vol_early, phase="distribution",
                session=session, confidence=0.45,
                reason="reversal: long trend + short sweep + displacement opposite",
                route=_route, tradeable=_trad,
                short_bias=short_bias, long_bias=long_bias,
            )
        return Regime(
            direction="flat", structure="unclear", volatility="unclear",
            phase="transition", session=session, confidence=0.15,
            reason="short/long direction disagree",
            route="SIT_OUT", tradeable=False,
            short_bias=short_bias, long_bias=long_bias,
        )

    # Structure: short window leads (current personality).  Long window
    # confirms or downgrades.  If short is unclear, regime is unclear
    # even if long looks clean — current personality is what matters.
    if short_struct == "unclear":
        return Regime(
            direction="flat", structure="unclear", volatility="unclear",
            phase="transition", session=session, confidence=0.2,
            reason="short-window structure unclear",
            route="SIT_OUT", tradeable=False,
            short_bias=short_bias, long_bias=long_bias,
        )

    structure = short_struct
    direction = "up" if short_bias == 1 else ("down" if short_bias == -1 else "flat")

    # Volatility from ATR behaviour.
    volatility = _volatility_of(short_bars, long_bars)
    # Compression refinement: range + very low ATR = compression.
    if structure == "range" and volatility == "low":
        structure = "compression"
    # Breakout refinement: range width expanding fast + displacement.
    if structure == "range" and volatility in ("high", "spike"):
        width_s = _range_width(short_bars)
        width_l = _range_width(long_bars)
        if width_l > 0 and width_s / width_l >= 0.55:
            # Short window already covers >55% of long range: expanding.
            structure = "breakout"
    # Reversal refinement: bias flipped vs long window.
    if short_bias != 0 and long_bias != 0 and short_bias != long_bias:
        # Handled above as Unclear for direction; unreachable.
        pass
    if short_bias == 0 and long_bias != 0:
        # Was trending, now flat: possible reversal forming, but not
        # confirmed — keep as range/compression, not reversal.
        pass

    # Candle character truth serum: freshest 20 bars.
    stats = _candle_stats(bars.iloc[-CANDLE_WINDOW:])
    swept = _sweep_in_window(short_bars)
    displaced = _displacement_in_window(short_bars)
    # Choppy: flat structure + many wicks + many dojis = liquidity
    # hunting, no core fits.  Distinct from clean range.
    if structure == "range" and stats["wicky"] >= 0.55 and stats["doji"] >= 0.30:
        structure = "choppy"
    phase = _phase_of(short_bars, swept, displaced, structure, volatility)

    # Confidence: start 0.5, adjust on agreement / disagreement.
    conf = 0.5
    if short_struct == long_struct or long_struct == "unclear":
        conf += 0.1
    else:
        conf -= 0.15
    if structure == "clean_trend":
        if stats["strong"] >= 0.4 and stats["doji"] <= 0.25:
            conf += 0.15
        elif stats["doji"] >= 0.4 or stats["wicky"] >= 0.5:
            conf -= 0.2
    if structure in ("range", "compression"):
        if stats["doji"] >= 0.3 or stats["wicky"] >= 0.4:
            conf += 0.1
    if volatility in ("low", "normal", "high"):
        conf += 0.05
    if phase == "transition":
        conf -= 0.1
    conf = float(max(0.0, min(1.0, conf)))

    # Low confidence -> Unclear, even if a label was computed.  This is
    # the "permission to say I don't know" rule.
    if conf < 0.35:
        return Regime(
            direction="flat", structure="unclear", volatility=volatility,
            phase="transition", session=session, confidence=conf,
            reason="low confidence — mixed evidence",
            route="SIT_OUT", tradeable=False,
            short_bias=short_bias, long_bias=long_bias,
        )

    route, tradeable, route_reason = route_for_regime(
        direction, structure, volatility, phase, session)
    # If routing says sit out, keep the computed structure but mark
    # tradeable False.  The label is still honest; the decision is sit out.
    reason = f"{structure} / {volatility} / {phase} — {route_reason}"
    return Regime(
        direction=direction, structure=structure, volatility=volatility,
        phase=phase, session=session, confidence=conf,
        reason=reason, route=route, tradeable=tradeable,
        short_bias=short_bias, long_bias=long_bias,
    )


def describe(regime: Regime) -> str:
    """One-line human summary for logs / paper journal."""
    flag = "TRADEABLE" if regime.tradeable else "SIT_OUT"
    return (
        f"[{flag}] {regime.label()} conf={regime.confidence:.2f} "
        f"route={regime.route} — {regime.reason}"
    )
