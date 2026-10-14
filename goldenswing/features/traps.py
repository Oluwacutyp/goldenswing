"""Trap-intelligence features — swing-failure patterns, fake breakouts,
beyond-FVG traps, inside-gap bait, and inverted-FVG tests.

Salvaged from the GOD Conviction Engine's trap layer
(``goldenswing.godce.stateful.zones_traps_features`` §§1–5), which was
killed as a full strategy but whose trap components were never measured
standalone.  This module re-implements those five state machines as a
clean, self-contained, godce-independent feature family: no OB engine,
no quantum gravity, no H&S, no scoring — none of the trap machines use
them.  It computes everything it needs internally (ATR via RMA,
confirmed swing pivots, volume SMA, the progressive Asian range, and
prior-day/week levels on the closed-bar convention).

Direction semantics (ported from the pine — read before interpreting):

- ``sfp_bull_active``: bulls just got trapped — price wicked through a
  swing high and reclaimed back below on confirming volume.  The memory
  window stays active for ``trap_memory_bars`` bars.  This is BEARISH
  context: it fades longs.  ``sfp_bear_active`` is the mirror — the
  bears got trapped, BULLISH context.
- ``beyond_bull``: a bull-FVG state machine reached state 2 — price
  swept THROUGH the gap (state 1) and reclaimed back out (state 2),
  trapping the crowd that faded the sweep.  Bullish.
  ``beyond_bull_full`` additionally reclaimed through the far side.
  ``beyond_bear*`` is the mirror on bear gaps.
- ``ifvg_demand``: a bear gap failed (close through its top), flipped
  to demand, and price is now testing it.  Bullish.  ``ifvg_supply``
  is the mirror (failed bull gap flipped to supply).
- ``bait_long``: close is parked INSIDE an active, un-reclaimed bull
  FVG while ``beyond_bull`` is NOT active — longs taken here are the
  bait.  Blocks longs.  ``bait_short`` is the mirror.
- ``lure_long``: a suspicious upside breakout — a close through the
  swing high on BELOW-average volume.  Blocks longs.
  ``lure_short`` is the mirror.

Convenience composites (the pine's own usage):

- ``trap_boost_long``  := sfp_bear_active OR beyond_bull OR ifvg_demand
- ``trap_boost_short`` := sfp_bull_active OR beyond_bear OR ifvg_supply

``sfp_*_eng`` marks a trap whose raid landed on engineered liquidity:
within 0.35 ATR of an EQH/EQL pool, the Asian-range extreme, the
prior-day high/low, or the prior-week high/low.  (The task spec lists
the first three; the pine checks all four, so the port keeps the
prior-week clause for exactness — see the regression test notes.)

Causality: every output at bar ``i`` is a pure function of bars
``<= i``.  ATR is Wilder's RMA, swings are ``swing_len``-bar fractals
positioned at their confirmation bar, the Asian range is the
progressive 00–07 UTC range held after the session, and prior-day/week
levels use the last CLOSED period (shift-then-ffill, no lookahead).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: EQH/EQL ATR tolerance (pine ``eqTol`` — not on the public contract)
_EQ_TOL = 0.30
#: FVG store cap (pine ``maxFVGs`` — not on the public contract)
_MAX_FVGS = 5
#: engineered-liquidity proximity (pine: ``a * 0.35``)
_ENG_ATR = 0.35
#: Asian session window, UTC hours [start, end)
_ASIAN_START, _ASIAN_END = 0, 7

#: output columns, in contract order
COLUMNS = (
    "trap_boost_long", "trap_boost_short",
    "bait_long", "bait_short",
    "lure_long", "lure_short",
    "sfp_bull_active", "sfp_bear_active",
    "sfp_bull_eng", "sfp_bear_eng",
    "fake_brk_up", "fake_brk_dn",
    "beyond_bull", "beyond_bear",
    "beyond_bull_full", "beyond_bear_full",
    "ifvg_demand", "ifvg_supply",
)


@dataclass
class TrapParams:
    """Trap-intelligence parameters.  Defaults mirror the GOD-CE pine
    inputs (``swingLen``/``atrLen``/``volLen``/``trapVolMult``/
    ``trapMemoryBars``/``brkLookback``/``fvgMinPct``/``zoneMaxAge``/
    ``maxInvFvgs``/``useTrapEngine``/``useBeyondFvg``/``useInvFvg``),
    with the pine's gold ``symFvgScl=0.75`` exposed as
    ``fvg_atr_scale``."""
    swing_len: int = 10
    atr_len: int = 14
    vol_len: int = 20
    trap_vol_mult: float = 1.15   # SFP volume confirmation × volSMA
    trap_memory_bars: int = 12
    brk_lookback: int = 6
    fvg_min_pct: float = 20.0     # FVG min size as % ATR
    fvg_atr_scale: float = 0.75    # gold symbol scale from the pine
    zone_max_age: int = 120
    max_inv_fvgs: int = 4
    use_trap_engine: bool = True
    use_beyond_fvg: bool = True
    use_inv_fvg: bool = True


# ── internal primitives (same recipes as the godce ports) ───────────

def _rma(values: np.ndarray, n: int) -> np.ndarray:
    """Wilder's RMA — ``ewm(alpha=1/n, adjust=False)``."""
    return pd.Series(values).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    pc = np.empty_like(c)
    pc[0] = np.nan
    pc[1:] = c[:-1]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return _rma(tr, n)


def _confirmed_pivots(x: np.ndarray, swing_len: int, high: bool) -> np.ndarray:
    """``swing_len``-bar fractal pivots, positioned at the confirmation
    bar (pine ``ta.pivothigh/ta.pivotlow`` semantics)."""
    w = 2 * swing_len + 1
    s = pd.Series(x)
    roll = s.rolling(w, center=True, min_periods=w)
    is_p = (s == roll.max()).to_numpy() if high else (s == roll.min()).to_numpy()
    return pd.Series(np.where(is_p, x, np.nan)).shift(swing_len).to_numpy()


def _asian_range(h: pd.Series, l: pd.Series, in_asian) -> tuple[np.ndarray, np.ndarray]:
    """Progressive 00–07 UTC range high/low, held after the session."""
    day = pd.Series(h.index.date, index=h.index)
    ah = h.where(in_asian).groupby(day).cummax().ffill()
    al = l.where(in_asian).groupby(day).cummin().ffill()
    return ah.to_numpy(), al.to_numpy()


def _closed_levels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prior-day and prior-week high/low on the last-CLOSED-bar
    convention (shift-then-ffill — no lookahead)."""
    idx = df.index
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    d1 = df.resample("1D").agg(agg)
    w1 = df.resample("1W").agg(agg)

    def closed(s: pd.Series) -> np.ndarray:
        return s.shift(1).reindex(idx, method="ffill").to_numpy()

    return (closed(d1["high"]), closed(d1["low"]),
            closed(w1["high"]), closed(w1["low"]))


def _add_fvg(store: list, top: float, bot: float, bar: int) -> None:
    if len(store) >= _MAX_FVGS:
        store.pop(0)
    store.append({"top": top, "bot": bot, "bar": bar, "ok": True})


# ── main entry point ─────────────────────────────────────────────────

def compute_trap_features(df: pd.DataFrame,
                          params: TrapParams | None = None) -> pd.DataFrame:
    """Per-bar trap-intelligence features.

    ``df``: OHLCV with columns open/high/low/close/volume and a
    tz-aware UTC DatetimeIndex.  Returns a boolean DataFrame on the
    same index with exactly the contract columns (see ``COLUMNS``).
    Pure function — no I/O, no godce imports.
    """
    P = params if params is not None else TrapParams()
    need = ("open", "high", "low", "close", "volume")
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"compute_trap_features: missing columns {missing}")

    n = len(df)
    O = df["open"].to_numpy(dtype=float)
    H = df["high"].to_numpy(dtype=float)
    L = df["low"].to_numpy(dtype=float)
    C = df["close"].to_numpy(dtype=float)
    V = df["volume"].to_numpy(dtype=float)

    # ── vectorized precomputes ──
    atr = _atr(H, L, C, P.atr_len)
    vol_sma = pd.Series(V).rolling(P.vol_len).mean().to_numpy()
    PH = _confirmed_pivots(H, P.swing_len, high=True)
    PL = _confirmed_pivots(L, P.swing_len, high=False)
    ts = df.index
    if getattr(ts, "tz", None) is not None:
        ts = ts.tz_convert("UTC")
    in_asian = pd.Series((ts.hour >= _ASIAN_START) & (ts.hour < _ASIAN_END),
                         index=df.index)
    AH, AL = _asian_range(df["high"], df["low"], in_asian)
    pdH, pdL, pwH, pwL = _closed_levels(df)

    out = {name: np.zeros(n, dtype=bool) for name in COLUMNS}

    # ── zone stores + trap state ──
    bFvg: list = []
    rFvg: list = []
    ifvgSup: list = []
    ifvgDem: list = []

    bullTrapBar = np.nan
    bullTrapWasEng = False
    bearTrapBar = np.nan
    bearTrapWasEng = False
    brkUpBar, brkUpLvl = np.nan, np.nan
    brkDnBar, brkDnLvl = np.nan, np.nan
    bfgState, bfgBot, bfgTop, bfgBar = 0, np.nan, np.nan, np.nan
    bfgReclaimFull = False
    rfgState, rfgTop, rfgBot, rfgBar = 0, np.nan, np.nan, np.nan
    rfgReclaimFull = False
    lastPH = lastPH2 = np.nan
    lastPL = lastPL2 = np.nan

    fvg_min_mult = (P.fvg_min_pct / 100.0) * P.fvg_atr_scale

    for i in range(n):
        a = atr[i]
        a_ok = not np.isnan(a) and a > 0
        vs = vol_sma[i]
        v_ok = not np.isnan(vs) and vs > 0

        # last confirmed pivots (registered at the confirmation bar)
        if not np.isnan(PH[i]):
            lastPH2, lastPH = lastPH, PH[i]
        if not np.isnan(PL[i]):
            lastPL2, lastPL = lastPL, PL[i]
        is_eqh = (not np.isnan(lastPH) and not np.isnan(lastPH2)
                  and a_ok and abs(lastPH - lastPH2) <= a * _EQ_TOL)
        is_eql = (not np.isnan(lastPL) and not np.isnan(lastPL2)
                  and a_ok and abs(lastPL - lastPL2) <= a * _EQ_TOL)
        eqh = (lastPH + lastPH2) / 2 if is_eqh else np.nan
        eql = (lastPL + lastPL2) / 2 if is_eql else np.nan

        # ── FVG detection ──
        fvg_min = a * fvg_min_mult if a_ok else np.nan
        if a_ok and i >= 2:
            if L[i] > H[i - 2] and (L[i] - H[i - 2]) >= fvg_min:
                _add_fvg(bFvg, L[i], H[i - 2], i - 1)
            if H[i] < L[i - 2] and (L[i - 2] - H[i]) >= fvg_min:
                _add_fvg(rFvg, L[i - 2], H[i], i - 1)

        # ── FVG hit + inverted-FVG flips ──
        for z in bFvg:
            if not z["ok"]:
                continue
            zt, zb, zbar = z["top"], z["bot"], z["bar"]
            failed = C[i] < zb
            if failed or (i - zbar) > P.zone_max_age:
                z["ok"] = False
                if P.use_inv_fvg and failed:
                    if len(ifvgSup) >= P.max_inv_fvgs:
                        ifvgSup.pop(0)
                    ifvgSup.append({"top": zt, "bot": zb, "bar": i, "ok": True})
        for z in rFvg:
            if not z["ok"]:
                continue
            zt, zb, zbar = z["top"], z["bot"], z["bar"]
            failed = C[i] > zt
            if failed or (i - zbar) > P.zone_max_age:
                z["ok"] = False
                if P.use_inv_fvg and failed:
                    if len(ifvgDem) >= P.max_inv_fvgs:
                        ifvgDem.pop(0)
                    ifvgDem.append({"top": zt, "bot": zb, "bar": i, "ok": True})

        # ── 1) swing failure patterns ──
        bull_trap = (P.use_trap_engine and not np.isnan(lastPH)
                     and H[i] > lastPH and C[i] < lastPH and C[i] < O[i]
                     and v_ok and V[i] > vs * P.trap_vol_mult)
        bear_trap = (P.use_trap_engine and not np.isnan(lastPL)
                     and L[i] < lastPL and C[i] > lastPL and C[i] > O[i]
                     and v_ok and V[i] > vs * P.trap_vol_mult)
        prox = a * _ENG_ATR if a_ok else np.nan
        bull_eng = bull_trap and a_ok and (
            (is_eqh and abs(lastPH - eqh) <= prox)
            or (not np.isnan(AH[i]) and abs(H[i] - AH[i]) <= prox)
            or (not np.isnan(pdH[i]) and abs(H[i] - pdH[i]) <= prox)
            or (not np.isnan(pwH[i]) and abs(H[i] - pwH[i]) <= prox))
        bear_eng = bear_trap and a_ok and (
            (is_eql and abs(lastPL - eql) <= prox)
            or (not np.isnan(AL[i]) and abs(L[i] - AL[i]) <= prox)
            or (not np.isnan(pdL[i]) and abs(L[i] - pdL[i]) <= prox)
            or (not np.isnan(pwL[i]) and abs(L[i] - pwL[i]) <= prox))
        if bull_trap:
            bullTrapBar, bullTrapWasEng = i, bool(bull_eng)
        if bear_trap:
            bearTrapBar, bearTrapWasEng = i, bool(bear_eng)
        bull_active = (not np.isnan(bullTrapBar)
                       and i - bullTrapBar <= P.trap_memory_bars)
        bear_active = (not np.isnan(bearTrapBar)
                       and i - bearTrapBar <= P.trap_memory_bars)

        # ── 2) fake-breakout machine ──
        brk_up = (P.use_trap_engine and not np.isnan(lastPH)
                  and C[i] > lastPH and i >= 1 and C[i - 1] <= lastPH)
        brk_dn = (P.use_trap_engine and not np.isnan(lastPL)
                  and C[i] < lastPL and i >= 1 and C[i - 1] >= lastPL)
        brk_up_sus = brk_up and v_ok and V[i] < vs
        brk_dn_sus = brk_dn and v_ok and V[i] < vs
        if brk_up:
            brkUpBar, brkUpLvl = i, lastPH
        if brk_dn:
            brkDnBar, brkDnLvl = i, lastPL
        f_up = f_dn = False
        if (not np.isnan(brkUpBar) and i - brkUpBar <= P.brk_lookback
                and C[i] < (brkUpLvl if not np.isnan(brkUpLvl) else C[i])):
            f_up = True
            brkUpBar = np.nan
            bullTrapBar, bullTrapWasEng = i, bool(bullTrapWasEng or bull_eng)
            bull_active = True
        if (not np.isnan(brkDnBar) and i - brkDnBar <= P.brk_lookback
                and C[i] > (brkDnLvl if not np.isnan(brkDnLvl) else C[i])):
            f_dn = True
            brkDnBar = np.nan
            bearTrapBar, bearTrapWasEng = i, bool(bearTrapWasEng or bear_eng)
            bear_active = True
        if not np.isnan(brkUpBar) and i - brkUpBar > P.brk_lookback:
            brkUpBar = np.nan
        if not np.isnan(brkDnBar) and i - brkDnBar > P.brk_lookback:
            brkDnBar = np.nan

        out["sfp_bull_active"][i] = bull_active
        out["sfp_bear_active"][i] = bear_active
        out["sfp_bull_eng"][i] = bool(bullTrapWasEng) and bull_active
        out["sfp_bear_eng"][i] = bool(bearTrapWasEng) and bear_active
        out["fake_brk_up"][i] = f_up
        out["fake_brk_dn"][i] = f_dn
        out["lure_long"][i] = brk_up_sus
        out["lure_short"][i] = brk_dn_sus

        # ── 3) beyond-FVG machines (global 0 → 1 → 2 per gap side) ──
        if P.use_beyond_fvg:
            for z in bFvg:
                zt, zb, zbar = z["top"], z["bot"], z["bar"]
                if i - zbar > P.zone_max_age:
                    continue
                if bfgState == 0 and L[i] < zb and C[i] > zb:
                    # one-bar raid-and-reclaim straight to state 2
                    bfgState, bfgBot, bfgTop, bfgBar = 2, zb, zt, i
                    bfgReclaimFull = C[i] > zt
                elif bfgState == 0 and C[i] < zb:
                    bfgState, bfgBot, bfgTop, bfgBar = 1, zb, zt, i
                    bfgReclaimFull = False
                elif bfgState == 1 and C[i] > bfgBot:
                    bfgState, bfgBar = 2, i
                    bfgReclaimFull = C[i] > bfgTop
            for z in rFvg:
                zt, zb, zbar = z["top"], z["bot"], z["bar"]
                if i - zbar > P.zone_max_age:
                    continue
                if rfgState == 0 and H[i] > zt and C[i] < zt:
                    rfgState, rfgTop, rfgBot, rfgBar = 2, zt, zb, i
                    rfgReclaimFull = C[i] < zb
                elif rfgState == 0 and C[i] > zt:
                    rfgState, rfgTop, rfgBot, rfgBar = 1, zt, zb, i
                    rfgReclaimFull = False
                elif rfgState == 1 and C[i] < rfgTop:
                    rfgState, rfgBar = 2, i
                    rfgReclaimFull = C[i] < rfgBot
        if bfgState > 0 and not np.isnan(bfgBar) and i - bfgBar > P.trap_memory_bars:
            bfgState = 0
        if rfgState > 0 and not np.isnan(rfgBar) and i - rfgBar > P.trap_memory_bars:
            rfgState = 0
        out["beyond_bull"][i] = bfgState == 2
        out["beyond_bear"][i] = rfgState == 2
        out["beyond_bull_full"][i] = bool(bfgReclaimFull) and bfgState == 2
        out["beyond_bear_full"][i] = bool(rfgReclaimFull) and rfgState == 2

        # ── 4) inside-gap bait ──
        in_bull = in_bear = False
        if P.use_beyond_fvg:
            for z in bFvg:
                if z["ok"] and z["bot"] <= C[i] <= z["top"]:
                    in_bull = True
                    break
            for z in rFvg:
                if z["ok"] and z["bot"] <= C[i] <= z["top"]:
                    in_bear = True
                    break
        out["bait_long"][i] = P.use_beyond_fvg and in_bull and bfgState != 2
        out["bait_short"][i] = P.use_beyond_fvg and in_bear and rfgState != 2

        # ── 5) inverted-FVG tests ──
        sup_test = False
        if P.use_inv_fvg:
            for z in ifvgSup:
                if not z["ok"]:
                    continue
                zt, zb, zbar = z["top"], z["bot"], z["bar"]
                if H[i] >= zb and C[i] <= zt and C[i] <= O[i]:
                    sup_test = True
                if C[i] > zt or i - zbar > P.zone_max_age:
                    z["ok"] = False
        dem_test = False
        if P.use_inv_fvg:
            for z in ifvgDem:
                if not z["ok"]:
                    continue
                zt, zb, zbar = z["top"], z["bot"], z["bar"]
                if L[i] <= zt and C[i] >= zb and C[i] >= O[i]:
                    dem_test = True
                if C[i] < zb or i - zbar > P.zone_max_age:
                    z["ok"] = False
        out["ifvg_supply"][i] = sup_test
        out["ifvg_demand"][i] = dem_test

        # ── composites (the pine's own usage) ──
        out["trap_boost_long"][i] = bear_active or bfgState == 2 or dem_test
        out["trap_boost_short"][i] = bull_active or rfgState == 2 or sup_test

    data = {name: pd.Series(out[name], index=df.index, dtype=bool)
            for name in COLUMNS}
    return pd.DataFrame(data, index=df.index)[list(COLUMNS)]
