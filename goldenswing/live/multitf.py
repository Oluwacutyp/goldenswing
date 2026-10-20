"""Multi-timeframe structural engine — fractal, hierarchy-gated.

The SAME structural engine runs fractally on every preset: bias on the
slow clock, points of interest (POIs) on the middle clock, trigger on
the fast clock.  There is no voting and no score averaging.  A signal
fires only when every gate passes in order:

1. HTF bias != 0 (last closed swing structure: higher highs + higher
   lows = bull, lower highs + lower lows = bear, anything else = flat).
2. The trigger direction equals the bias (counter-trend triggers are
   vetoed, never averaged in).
3. The trigger bar closes AT the POI: inside the zone band, never
   above it for longs, never below it for shorts.  No chasing.
4. A structural event (BOS / MSCH / CHoCH) fires on the trigger TF,
   with a displacement gate on the trigger bar.  No indicator
   crossovers are ever triggers; ATR is context only.
5. The stop is the adverse structural invalidation plus a buffer.  It
   is never tightened to fit a risk budget.  If the stop is invalid,
   there is no signal.

The engine is PURE: no journal writes, no broker calls, no I/O.  The
coordinator wires journaling and execution around :func:`evaluate`.

$12 HONESTY (user-explicit, do not paper over)
--------------------------------------------
Paper and backtest runs use ``PAPER_START_BALANCE_12 = 12.0`` as the
starting balance, passed explicitly — the defaults of
``PaperExecutor`` and ``EngineContext`` are NOT changed (existing
tests depend on them).  At 1% risk the risk budget is $0.12 per trade.

A $12 account CANNOT execute most swing trades at the OANDA 1-unit
(1 oz) minimum: a $20 stop means 1-unit risk of $20.00, 167x the
$0.12 budget.  So results are reported in TWO ledgers, never mixed:

- fractional-unit R accounting — strategy quality (R-multiples, win
  rate, drawdown).  Fractional units are a measurement device, NOT a
  claim that OANDA would fill them.
- the ``oanda_1unit_feasible`` flag per trade — feasible iff the risk
  of ONE ounce (stop distance in dollars) fits inside the risk
  budget.  Feasible trades are reported as a separate subset.
  Margin is reported separately (``margin_50x``) and is NOT folded
  into the flag: OANDA leverage varies by jurisdiction, so a single
  margin rule would be a false universal.  At 50:1, 1 oz of gold at
  ~$4000 needs ~$80 margin — a further, separate blocker on $12.

Never present fractional-unit results as OANDA-executable.

PRESETS
-------
- SWING: D1 bias / H4 zones / H1 trigger, swing_n = 3. Hard veto: W1.
- INTRADAY: H4 bias / H1 zones / M15 trigger, swing_n = 3. Hard veto: D1.
- SCALP: H1 bias / M15 zones / M5 trigger, swing_n = 3.
  Hard veto: H4, second veto D1 (if data available).
- SCALP_M1: M5 bias / M5 zones / M1 trigger, swing_n = 5. Bias and
  zones share M5 at this scale; there is no slower clock worth
  gating on for an M1 trigger. Hard veto: H1.

HARD VETO (user-explicit, 2026-10-13)
--------------------------------------
Main bias TF = holding-period TF. The next higher TF has HARD VETO
power: if it STRONGLY disagrees with the main bias, there is no trade
— no averaging, no vote.  "Strong" is defined precisely in
:func:`check_hard_veto` (opposite 2-swing structure AND both swing
legs >= 0.25 x ATR14 on the veto TF).  Flat/ranging veto TF = no
veto.  Marginal opposite structure (tiny swing legs) = no veto.
A missing veto TF is fail-OPEN (no veto): a missing W1 must never
kill swing trades, it only means the veto could not be evaluated.
The veto definition is OUR construction, not a proven rule — it needs
out-of-sample + forward-paper validation before anyone claims it
helps.  See :func:`check_hard_veto`.

``swing_n`` values are STARTING HYPOTHESES, not proven thresholds:
N=5 on M1 (11-bar fractal) and N=3 on M5 (7-bar fractal) were chosen
so the fractal width scales with the noise of the clock.  They have
not been validated as optimal and must not be quoted as such.

CAUSALITY
---------
Every function reads CLOSED bars only.  ``find_swings`` returns only
confirmed swings (a swing at index ``i`` needs ``n`` bars after it).
``evaluate`` computes trigger-TF swings from ``bars[:-1]`` so the
trigger bar can never confirm its own reference swing.

CONTRACT — THREE SEPARATE CONCERNS, NEVER CONFUSED
--------------------------------------------------
1. ``evaluate_bias()`` = STRUCTURE direction, the ONLY bias that
   gates trades (hierarchy leg 1, ``d1_bias``): +1 iff the last two
   confirmed swings show higher high AND higher low; -1 iff lower
   high AND lower low; else 0 (flat/mixed — no trade).  A swing at
   index ``i`` needs ``swing_n`` bars AFTER it (default 3), so a D1
   bias flip is only recognised >= 3 daily closes after the extreme
   printed.  Price action alone cannot flip it.  ``d1_flat`` is a
   legitimate HOLD during choppy / transitioning regimes, NOT a feed
   error, NOT a bug — see ``live/hierarchy.py::D1BiasState`` /
   ``bias_hold_state`` for the pending-swing / bars-remaining /
   flip-requirement tracking, and ``log_hold_state`` (INFO, HOLD)
   vs ``log_feed_error`` (ERROR, FEED_ERROR).
2. ``live/macrotrend.py::compute_macrotrend()`` = SIZING ONLY:
   EMA200 bias per TF (D1/W1/MN1), Golden Swing (all equal nonzero)
   1.5x, D1_W1 1.25x, D1_Only 1.0x, D1/W1 conflict 0.5x.  It
   never gates, never blocks, never touches hierarchy legs.
3. DXY (``live/dxy.py``) = CONFLUENCE NUDGE ONLY: correlation <
   -0.8 -> +0.5, > -0.2 -> -0.5, else 0; unavailable -> 0.  It
   never gates.  A missing DXY feed degrades to 0, it is not a
   trade blocker and not a bias.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# $12 starting balance for paper + backtest runs (user-explicit).
# Passed explicitly into new code/tests; the defaults of PaperExecutor
# and EngineContext are deliberately left untouched.
PAPER_START_BALANCE_12 = 12.0


# ------------------------------------------------------------------ config
@dataclass(frozen=True)
class TimeframeConfig:
    """One preset's clocks + gates.  Each preset OWNS its entry clock:
    the trigger TF is the only clock that can fire."""

    bias_tf: str
    zone_tf: str
    trigger_tf: str
    swing_n: int
    # displacement gate on the trigger bar
    displacement_body_mult: float = 1.5  # body > 1.5x avg body (20 bars)
    atr_mult: float = 0.5                # body > 0.5x ATR14
    body_range_min: float = 0.6          # body / range >= 0.6
    breakout_atr: float = 0.1           # close beyond swing by >= 0.1x ATR
    # hard-veto clocks (next higher TF(s) than bias_tf).  None = no veto
    # on that level (e.g. custom configs, or the bias TF is the top).
    veto_tf: str | None = None
    veto_tf2: str | None = None


SWING = TimeframeConfig("D1", "H4", "H1", swing_n=3, veto_tf="W1")
INTRADAY = TimeframeConfig("H4", "H1", "M15", swing_n=3, veto_tf="D1")
SCALP = TimeframeConfig("H1", "M15", "M5", swing_n=3, veto_tf="H4",
                        veto_tf2="D1")
SCALP_M1 = TimeframeConfig("M5", "M5", "M1", swing_n=5, veto_tf="H1")

PRESETS: dict[str, TimeframeConfig] = {
    "swing": SWING,
    "intraday": INTRADAY,
    "scalp": SCALP,
    "scalp_m1": SCALP_M1,
}


def get_preset(name: str) -> TimeframeConfig:
    """Preset by name (``swing`` | ``intraday`` | ``scalp`` | ``scalp_m1``)."""
    try:
        return PRESETS[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown preset {name!r}; expected one of {sorted(PRESETS)}"
        ) from None


# ------------------------------------------------------------- indicators
def atr14(bars: pd.DataFrame, period: int = 14) -> float:
    """Wilder's ATR (EWMA smoothing, alpha = 1/period) at the last bar.

    Returns 0.0 when the frame is too short to measure.  ATR is
    CONTEXT only — it sizes gates and buffers, never triggers.
    """
    if len(bars) < 2:
        return 0.0
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close),
                               np.abs(low - prev_close)))
    if len(tr) < period:
        return float(np.mean(tr))
    return float(pd.Series(tr).ewm(alpha=1.0 / period, adjust=False,
                                   min_periods=period).mean().iloc[-1])


def _as_float(value, bars: pd.DataFrame | None = None) -> float:
    """Coerce an ATR argument: scalar, trailing Series/array, or None
    (None -> compute from ``bars``)."""
    if value is None:
        if bars is None:
            return 0.0
        return atr14(bars)
    if isinstance(value, (pd.Series, np.ndarray, list, tuple)):
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return 0.0
        return float(arr[-1])
    return float(value)


# ------------------------------------------------------------------ swings
def find_swings(high, low, n: int) -> tuple[list[tuple[int, float]],
                                            list[tuple[int, float]]]:
    """Confirmed fractal swings, as ``(position, price)`` pairs.

    Swing high at ``i``: ``high[i]`` strictly greater than every other
    bar in ``[i-n, i+n]``.  Swing low: strictly less.  Only CONFIRMED
    swings are returned — ``i`` ranges over ``[n, len-n)``, so a swing
    needs ``n`` bars after it.  The last ``n`` bars can never contain
    a swing: no lookahead, by construction.  Vectorised (rolling
    max/min on both sides); ties are never swings.
    """
    if n < 1:
        raise ValueError(f"swing_n must be >= 1 (got {n})")
    h = pd.Series(np.asarray(high, dtype=float))
    lo = pd.Series(np.asarray(low, dtype=float))
    m = len(h)
    if m < 2 * n + 1:
        return [], []

    def _levels(s: pd.Series, want_high: bool) -> list[tuple[int, float]]:
        agg = "max" if want_high else "min"
        left = s.shift(1).rolling(n, min_periods=n).agg(agg)
        rev = s.iloc[::-1]
        right = rev.shift(1).rolling(n, min_periods=n).agg(agg).iloc[::-1]
        right.index = s.index
        if want_high:
            mask = (s > left) & (s > right)
        else:
            mask = (s < left) & (s < right)
        return [(int(i), float(s.iloc[i]))
                for i in np.where(mask.to_numpy())[0]]

    return _levels(h, True), _levels(lo, False)


# ------------------------------------------------------- structure events
def _level(value) -> float | None:
    """A swing level: scalar -> itself; list of (idx, price) pairs or
    prices -> the LAST entry (most recent)."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        last = value[-1]
        if isinstance(last, (list, tuple)):
            return float(last[-1])
        return float(last)
    return float(value)


def detect_bos_msch_choch(bars: pd.DataFrame, swing_high, swing_low, atr,
                          *, trend: int = 0,
                          minor_high=None, minor_low=None,
                          config: TimeframeConfig | None = None
                          ) -> str | None:
    """Structure event on the LAST (closed) bar of ``bars``.

    ``swing_high`` / ``swing_low`` are the reference (most recent)
    trigger-TF swing levels; ``minor_high`` / ``minor_low`` are the
    second-most-recent swings.  The event taxonomy:

    - BOS (break of structure): the close breaks BOTH the reference
      and the minor level on one side (past the more extreme of the
      two).  Bull when the prior trend is up/flat, CHoCH when the
      prior trend is down — and mirrored for bears.  With no minor
      level supplied, breaking the reference level alone is a BOS/CHoCH.
    - MSCH (minor structure change): the close breaks the reference
      (most recent) swing but NOT the older minor level — the
      pullback configuration, where the recent swing sits inside the
      older one.  Breaking an OLDER level while still inside the
      most recent swing is mid-range chop, not an event: None.
    - No event when the displacement gate fails: the trigger bar's
      body must exceed ``displacement_body_mult`` x the 20-bar
      average body AND ``atr_mult`` x ATR14, with body/range >=
      ``body_range_min``, and the close must clear the swing by >=
      ``breakout_atr`` x ATR14.

    Returns one of ``BOS_bull`` / ``BOS_bear`` / ``MSCH_bull`` /
    ``MSCH_bear`` / ``CHoCH_bull`` / ``CHoCH_bear`` / ``None``.
    """
    if len(bars) < 2:
        return None
    cfg = config or TimeframeConfig("D1", "H4", "H1", swing_n=3)
    last = bars.iloc[-1]
    body = abs(float(last["close"]) - float(last["open"]))
    rng = float(last["high"]) - float(last["low"])
    if rng <= 0:
        return None
    prior = bars.iloc[:-1]
    bodies = (prior["close"] - prior["open"]).abs()
    avg_body = float(bodies.tail(20).mean()) if len(bodies) else 0.0
    atr_v = _as_float(atr, bars)
    if avg_body <= 0 or atr_v <= 0:
        return None
    # displacement gate — all four conditions, no partial credit
    if body <= cfg.displacement_body_mult * avg_body:
        return None
    if body <= cfg.atr_mult * atr_v:
        return None
    if body / rng < cfg.body_range_min:
        return None

    ref_high = _level(swing_high)
    ref_low = _level(swing_low)
    min_high = _level(minor_high)
    min_low = _level(minor_low)
    if ref_high is None or ref_low is None:
        return None
    buf = cfg.breakout_atr * atr_v
    close = float(last["close"])

    # BOS / CHoCH: past the MORE EXTREME of reference + minor.
    top = ref_high if min_high is None else max(ref_high, min_high)
    bottom = ref_low if min_low is None else min(ref_low, min_low)
    if close > top + buf:
        return "BOS_bull" if trend >= 0 else "CHoCH_bull"
    if close < bottom - buf:
        return "BOS_bear" if trend <= 0 else "CHoCH_bear"
    # MSCH: past the reference (most recent) swing only, while the
    # older minor level still holds — the pullback configuration.
    if min_high is not None and ref_high < close <= min_high + buf:
        # close cleared ref_high (strictly) but not the older level;
        # the +buf tolerance keeps a close that merely grazes the
        # older level from counting as a break of it.
        if close > ref_high + buf:
            return "MSCH_bull"
    if min_low is not None and min_low - buf <= close < ref_low:
        if close < ref_low - buf:
            return "MSCH_bear"
    return None


# ------------------------------------------------------------------- bias
def evaluate_bias(bias_bars: pd.DataFrame, *, swing_n: int = 3) -> int:
    """HTF bias from the last closed swing structure: +1 bull (higher
    highs AND higher lows on the last two confirmed swings each side),
    -1 bear (lower highs AND lower lows), else 0 (flat/mixed — no
    trade).  Fewer than two swings on either side -> 0."""
    highs, lows = find_swings(bias_bars["high"], bias_bars["low"], swing_n)
    if len(highs) < 2 or len(lows) < 2:
        return 0
    hh = highs[-1][1] > highs[-2][1]
    hl = lows[-1][1] > lows[-2][1]
    lh = highs[-1][1] < highs[-2][1]
    ll = lows[-1][1] < lows[-2][1]
    if hh and hl:
        return 1
    if lh and ll:
        return -1
    return 0


# ------------------------------------------------------------- hard veto
# Fraction of veto-TF ATR14 that BOTH swing legs must exceed for an
# opposite veto-TF bias to count as STRONG disagreement.  0.25 is a
# construction, not a measured optimum: it filters marginal wiggles
# (swing levels differing by dust) while not demanding a full ATR per
# leg (fractal swing legs on a trending TF routinely move < 1 ATR).
VETO_STRONG_LEG_ATR = 0.25


def check_hard_veto(veto_bars: pd.DataFrame, main_bias: int, *,
                    swing_n: int = 3) -> dict:
    """Hard-veto verdict for ONE veto TF against the main bias.

    STRONG disagreement (the ONLY kind that vetoes) requires ALL of:

    1. Veto-TF bias (``evaluate_bias``: last two confirmed swings on
       each side) is the OPPOSITE sign of ``main_bias``.  Flat (0) is
       NOT disagreement — a ranging veto TF never vetoes.  Agreement
       obviously never vetoes.
    2. BOTH veto-TF swing legs clear ``VETO_STRONG_LEG_ATR`` x
       ATR14(veto TF): ``|last_high - prev_high|`` AND
       ``|last_low - prev_low|``.  A one-candle flicker or a
       dust-sized opposite wiggle fails this: the veto TF must show
       real trend structure opposite to the main bias, i.e.
       displacement + structure, not just one candle.

    Fewer than two confirmed swings on either side, ATR <= 0, or
    ``main_bias == 0`` -> not strong, no veto (``vetoed: False``).

    Returns ``{vetoed, veto_bias, strong, leg_high, leg_low, atr,
    threshold}``.  ``vetoed`` is True iff the veto TF strongly
    disagrees.  Pure function, closed bars only.
    """
    out = {"vetoed": False, "veto_bias": 0, "strong": False,
           "leg_high": 0.0, "leg_low": 0.0, "atr": 0.0,
           "threshold": 0.0}
    if main_bias == 0:
        return out
    if veto_bars is None or len(veto_bars) < 2 * swing_n + 5:
        return out
    veto_bias = evaluate_bias(veto_bars, swing_n=swing_n)
    out["veto_bias"] = int(veto_bias)
    if veto_bias == 0 or veto_bias == main_bias:
        return out
    # opposite bias: measure the swing legs.
    highs, lows = find_swings(veto_bars["high"], veto_bars["low"], swing_n)
    if len(highs) < 2 or len(lows) < 2:
        return out
    atr_v = atr14(veto_bars)
    out["atr"] = float(atr_v)
    if atr_v <= 0:
        return out
    leg_high = abs(highs[-1][1] - highs[-2][1])
    leg_low = abs(lows[-1][1] - lows[-2][1])
    out["leg_high"] = float(leg_high)
    out["leg_low"] = float(leg_low)
    threshold = VETO_STRONG_LEG_ATR * atr_v
    out["threshold"] = float(threshold)
    strong = leg_high >= threshold and leg_low >= threshold
    out["strong"] = bool(strong)
    out["vetoed"] = bool(strong)
    return out


def check_vetoes(multitf_bars: dict[str, pd.DataFrame],
                 preset: TimeframeConfig, main_bias: int) -> dict:
    """Check every configured veto TF (``veto_tf``, then ``veto_tf2``).

    A veto TF absent from ``multitf_bars`` (or too short) is
    fail-OPEN: recorded as ``available: False`` and skipped, never a
    veto.  Either veto TF strongly disagreeing vetoes the trade.

    Returns ``{vetoed, veto_tf, veto_bias, vetoes: {tf: verdict}}``.
    """
    verdicts: dict[str, dict] = {}
    vetoed = False
    veto_tf_hit: str | None = None
    veto_bias_hit = 0
    for tf in (preset.veto_tf, preset.veto_tf2):
        if not tf:
            continue
        bars = multitf_bars.get(tf)
        if bars is None or len(bars) < 2 * preset.swing_n + 5:
            verdicts[tf] = {"vetoed": False, "veto_bias": 0,
                            "strong": False, "available": False}
            continue
        v = check_hard_veto(bars, main_bias, swing_n=preset.swing_n)
        v["available"] = True
        v["veto_tf"] = tf
        verdicts[tf] = v
        if v["vetoed"] and not vetoed:
            vetoed = True
            veto_tf_hit = tf
            veto_bias_hit = int(v["veto_bias"])
    return {"vetoed": vetoed, "veto_tf": veto_tf_hit,
            "veto_bias": veto_bias_hit, "vetoes": verdicts}


# ------------------------------------------------------------------ zones
def _zone(kind: str, level: float, index: int, atr: float) -> dict:
    half = 0.5 * atr
    return {"kind": kind, "level": float(level),
            "top": float(level + half), "bottom": float(level - half),
            "index": int(index)}


def zones_by_side(zone_bars: pd.DataFrame, *,
                  swing_n: int = 3) -> dict:
    """POI zones from the most recent confirmed zone-TF swings.

    Support = most recent swing low, resistance = most recent swing
    high.  Each zone is a band of half-width 0.5 x ATR14(zone TF)
    around its swing level.  ``nearest`` is whichever zone sits
    closest to the last close.  Returns None entries when a side has
    no confirmed swing (or ATR is unmeasurable).
    """
    atr = atr14(zone_bars)
    highs, lows = find_swings(zone_bars["high"], zone_bars["low"], swing_n)
    support = _zone("support", lows[-1][1], lows[-1][0], atr) \
        if lows and atr > 0 else None
    resistance = _zone("resistance", highs[-1][1], highs[-1][0], atr) \
        if highs and atr > 0 else None
    nearest = None
    if support is not None and resistance is not None:
        last_close = float(zone_bars["close"].iloc[-1])
        nearest = support if abs(support["level"] - last_close) <= \
            abs(resistance["level"] - last_close) else resistance
    elif support is not None:
        nearest = support
    elif resistance is not None:
        nearest = resistance
    return {"support": support, "resistance": resistance, "nearest": nearest}


def poi_for_direction(zone_bars: pd.DataFrame, direction: int, *,
                      swing_n: int = 3) -> dict | None:
    """Freshest TRADEABLE-side POI band for a direction, or None.

    ``zones_by_side`` returns the most recent swing per side with no
    check of where it sits relative to price: a stale or
    broken-through swing stays "the resistance" forever (live
    2026-10-20: the most recent confirmed H4 swing high sat at
    4073.2 with price at 4140, so the short arm paired the H1 ref
    against a band UNDER the market and containment could never
    pass).  This selector walks the confirmed swings on the
    direction's side from most recent backwards and picks the
    first band on the tradeable side of the last close: for shorts
    (``direction < 0``) the freshest resistance whose level is
    at/above the last close (overhead supply); for longs the
    freshest support whose level is at/below it.  When the most
    recent swing is already on the tradeable side this is exactly
    the ``zones_by_side`` pick — the side check only skips swings
    price has already left behind.  Bands are built exactly as
    :func:`zones_by_side` builds them (+/- 0.5 x ATR14 around the
    swing level).

    Returns None when no correct-side POI exists (price beyond
    every confirmed swing on that side) — callers must NOT fall
    back to a wrong-side band; a wrong-side anchor is how the live
    gate died.
    """
    if direction == 0:
        return None
    atr = atr14(zone_bars)
    if atr <= 0 or not len(zone_bars):
        return None
    highs, lows = find_swings(zone_bars["high"], zone_bars["low"],
                              swing_n)
    last_close = float(zone_bars["close"].iloc[-1])
    if direction < 0:
        for idx, level in reversed(highs):
            if float(level) >= last_close:
                return _zone("resistance", float(level), int(idx),
                             atr)
        return None
    for idx, level in reversed(lows):
        if float(level) <= last_close:
            return _zone("support", float(level), int(idx), atr)
    return None


def evaluate_zones(zone_bars: pd.DataFrame, *,
                   swing_n: int = 3) -> dict | None:
    """Nearest POI zone (swing high/low band) on the zone TF, or None."""
    return zones_by_side(zone_bars, swing_n=swing_n)["nearest"]


# --------------------------------------------------------------- evaluate
def _no_signal(reason: str, **extra) -> dict:
    out = {"signal": False, "direction": 0, "bias": 0, "zone": None,
           "trigger": None, "entry": None, "stop": None, "target": None,
           "risk": None, "reason": reason, "atr_trigger": None,
           "zone_atr": None}
    out.update(extra)
    return out


def _evaluate(multitf_bars: dict[str, pd.DataFrame],
              preset: TimeframeConfig, *,
              enable_veto: bool = True) -> dict:
    """Hierarchy-gated signal at the last closed trigger bar.

    ``multitf_bars`` maps TF name -> DataFrame of CLOSED bars
    (``open``/``high``/``low``/``close``).  Veto TFs (``preset.veto_tf``
    / ``preset.veto_tf2``) are OPTIONAL in the dict: absent = veto
    unavailable = fail-open, never a block.  Pass
    ``enable_veto=False`` for before/after comparisons (the veto is
    skipped entirely).  Returns a dict with ``signal`` (bool),
    ``direction`` (+1/-1/0), ``bias``, ``zone`` (the POI band),
    ``trigger`` (event name), ``entry`` (trigger-bar close), ``stop``
    (adverse structural invalidation + buffer), ``target`` (nearest
    opposing zone-TF pool, 2R floor), ``risk``, ``reason`` ("" on
    success, the failed gate otherwise), and ``veto`` (the
    :func:`check_vetoes` verdict).
    """
    for tf in (preset.bias_tf, preset.zone_tf, preset.trigger_tf):
        if tf not in multitf_bars:
            return _no_signal(f"missing_tf:{tf}")
    bias_bars = multitf_bars[preset.bias_tf]
    zone_bars = multitf_bars[preset.zone_tf]
    trig = multitf_bars[preset.trigger_tf]
    if len(trig) < max(30, 2 * preset.swing_n + 5):
        return _no_signal("insufficient_trigger_bars")
    if len(bias_bars) < 2 * preset.swing_n + 5:
        return _no_signal("insufficient_bias_bars")
    if len(zone_bars) < 2 * preset.swing_n + 5:
        return _no_signal("insufficient_zone_bars")

    bias = evaluate_bias(bias_bars, swing_n=preset.swing_n)
    if bias == 0:
        return _no_signal("bias_flat", bias=0)

    # HARD VETO: next-higher TF strongly disagreeing with the main
    # bias kills the trade outright.  Flat / agreeing / weak-opposite
    # veto TFs pass through.  Missing veto TF = fail-open.
    veto = {"vetoed": False, "veto_tf": None, "veto_bias": 0,
            "vetoes": {}}
    if enable_veto and (preset.veto_tf or preset.veto_tf2):
        veto = check_vetoes(multitf_bars, preset, bias)
        if veto["vetoed"]:
            return _no_signal("hard_veto", bias=bias, veto=veto,
                              veto_tf=veto["veto_tf"],
                              veto_bias=veto["veto_bias"])

    zone_highs, zone_lows = find_swings(
        zone_bars["high"], zone_bars["low"], preset.swing_n)
    zone_atr = atr14(zone_bars)
    support = _zone("support", zone_lows[-1][1], zone_lows[-1][0],
                    zone_atr) if zone_lows and zone_atr > 0 else None
    resistance = _zone("resistance", zone_highs[-1][1],
                       zone_highs[-1][0], zone_atr) \
        if zone_highs and zone_atr > 0 else None
    poi = support if bias == 1 else resistance
    if poi is None:
        return _no_signal("no_poi", bias=bias, zone_atr=zone_atr)

    # entry AT the POI: inside the band, never beyond it.
    entry = float(trig["close"].iloc[-1])
    if bias == 1:
        if entry > poi["top"]:
            return _no_signal("entry_above_poi", bias=bias, zone=poi,
                              zone_atr=zone_atr)
        if entry < poi["bottom"]:
            return _no_signal("entry_below_poi", bias=bias, zone=poi,
                              zone_atr=zone_atr)
    else:
        if entry < poi["bottom"]:
            return _no_signal("entry_below_poi", bias=bias, zone=poi,
                              zone_atr=zone_atr)
        if entry > poi["top"]:
            return _no_signal("entry_above_poi", bias=bias, zone=poi,
                              zone_atr=zone_atr)

    # trigger event on the trigger TF; swings from bars[:-1] so the
    # trigger bar can never confirm its own reference swing.
    atr_t = atr14(trig)
    trig_highs, trig_lows = find_swings(
        trig["high"].to_numpy()[:-1], trig["low"].to_numpy()[:-1],
        preset.swing_n)
    if not trig_highs or not trig_lows:
        return _no_signal("no_trigger_swings", bias=bias, zone=poi,
                          entry=entry, zone_atr=zone_atr,
                          atr_trigger=atr_t)
    ref_high = trig_highs[-1][1]
    minor_high = trig_highs[-2][1] if len(trig_highs) >= 2 else None
    ref_low = trig_lows[-1][1]
    minor_low = trig_lows[-2][1] if len(trig_lows) >= 2 else None
    event = detect_bos_msch_choch(
        trig, ref_high, ref_low, atr_t, trend=bias,
        minor_high=minor_high, minor_low=minor_low, config=preset)
    if event is None:
        return _no_signal("no_trigger", bias=bias, zone=poi, entry=entry,
                          zone_atr=zone_atr, atr_trigger=atr_t)
    direction = 1 if event.endswith("bull") else -1
    if direction != bias:
        return _no_signal("trigger_against_bias", bias=bias, zone=poi,
                          trigger=event, entry=entry,
                          zone_atr=zone_atr, atr_trigger=atr_t)

    # stop = adverse structural invalidation + buffer.  Never tightened.
    buf = preset.breakout_atr * atr_t
    if direction == 1:
        stop = ref_low - buf
        if not stop < entry:
            return _no_signal("stop_invalid", bias=bias, zone=poi,
                              trigger=event, entry=entry,
                              zone_atr=zone_atr, atr_trigger=atr_t)
    else:
        stop = ref_high + buf
        if not stop > entry:
            return _no_signal("stop_invalid", bias=bias, zone=poi,
                              trigger=event, entry=entry,
                              zone_atr=zone_atr, atr_trigger=atr_t)
    risk = abs(entry - stop)

    # target: nearest opposing zone-TF pool beyond entry, 2R floor
    # (same rule as the backtest harness).
    if direction == 1:
        cands = [p for _, p in zone_highs if p > entry]
    else:
        cands = [p for _, p in zone_lows if p < entry]
    target = entry + direction * 2.0 * risk
    if cands:
        nearest = min(cands, key=lambda p: abs(p - entry))
        if abs(nearest - entry) >= 2.0 * risk:
            target = nearest

    return {"signal": True, "direction": direction, "bias": bias,
            "zone": poi, "trigger": event, "entry": entry,
            "stop": float(stop), "target": float(target), "risk": risk,
            "reason": "", "atr_trigger": atr_t, "zone_atr": zone_atr,
            "veto": veto}


def evaluate(multitf_bars: dict[str, pd.DataFrame],
             preset: TimeframeConfig, *,
             enable_veto: bool = True, asof=None) -> dict:
    """Hierarchy-gated signal at the last closed trigger bar.

    Same gates as :func:`_evaluate` — the session mapping never
    changes a gate, a bias, or a direction.  When ``asof`` (a
    timestamp for the trigger bar) is given, the DST-aware session ->
    timeframe mapping (``reader/session_timeframes.py``) is attached
    as ``result["session"]``: which TF set is live, which entry clock
    the session selects, which killzone (if any) is live.  Annotation
    only — a signal that fires without ``asof`` is identical to one
    with it, apart from the added ``"session"`` key.
    """
    result = _evaluate(multitf_bars, preset, enable_veto=enable_veto)
    if asof is not None:
        from ..reader.session_timeframes import compose
        result["session"] = compose(asof)
    return result
