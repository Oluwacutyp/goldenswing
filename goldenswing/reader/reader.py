"""MarketReader — the awareness layer.

Reads an OHLCV frame in ONE forward pass and produces a :class:`MarketState`
per bar: structure (fractal swings, BOS/CHoCH, trend), a liquidity map
(pools of resting stops + sweep events), ICT levels (fair value gaps,
order blocks), premium/discount position inside the dealing range, the
volatility/momentum regime, and session context.

Causality contract: the state at bar ``t`` is built only from bars ``<= t``.
Swing points are timestamped at their *confirmation* bar (a fractal at bar
``i`` is knowable at ``i + n``), FVGs at the close of their third candle,
order blocks when the structure break prints.  Mitigation/sweep outcomes
are recorded with their bar so a state never "knows" a future outcome.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators, sessions
from .state import (
    FairValueGap,
    FlipLevel,
    LiquidityPool,
    MarketState,
    OrderBlock,
    StructureEvent,
    SweepEvent,
    SwingPoint,
)


@dataclass
class ReaderConfig:
    fractal_n: int = 2            # bars each side for a fractal swing
    atr_period: int = 14
    atr_pct_window: int = 252     # trailing window for ATR percentile
    range_lookback: int = 150     # bars for the dealing range
    displacement_min: float = 0.8  # body/ATR required for an OB displacement leg
    ob_search: int = 10           # bars back to find displacement / OB candle
    pool_merge_tol_atr: float = 0.1
    max_level_age: int = 300      # bars before an unmitigated FVG/OB expires
    max_pool_age: int = 800
    disp_body_atr: float = 1.0    # body/ATR that counts as displacement
    daily_fractal_n: int = 2      # fractal width on daily bars
    daily_ema_period: int = 50
    daily_min_days: int = 10      # completed days before an EMA-tilt bias
    flip_max_age: int = 30        # bars a broken S/R level stays a flip candidate


class MarketReader:
    def __init__(self, config: ReaderConfig | None = None):
        self.cfg = config or ReaderConfig()

    # -- fractal helper ----------------------------------------------------
    def _is_swing(self, arr: np.ndarray, i: int, n: int, kind: str) -> bool:
        v = arr[i]
        left = arr[i - n : i]
        right = arr[i + 1 : i + n + 1]
        if kind == "high":
            return bool(np.all(v > left) and np.all(v >= right))
        return bool(np.all(v < left) and np.all(v <= right))

    def read(self, df: pd.DataFrame) -> list[MarketState]:
        cfg = self.cfg
        df = df.sort_index()
        n_bars = len(df)
        if n_bars == 0:
            return []

        open_ = df["open"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)

        atr_s = indicators.atr(df, cfg.atr_period)
        atr_arr = atr_s.to_numpy(dtype=float)
        atr_pct = indicators.rolling_percentile(atr_arr, cfg.atr_pct_window)
        br = indicators.body_ratio(df, atr_s)  # signed body / ATR
        br = np.nan_to_num(br, nan=0.0)

        # EMA stack (9/21/50) for directional alignment gating
        ema9 = df["close"].ewm(span=9, adjust=False).mean().to_numpy()
        ema21 = df["close"].ewm(span=21, adjust=False).mean().to_numpy()
        ema50 = df["close"].ewm(span=50, adjust=False).mean().to_numpy()

        index = df.index
        if isinstance(index, pd.DatetimeIndex):
            if index.tz is not None:
                index_utc = index.tz_convert("UTC")
            else:
                index_utc = index.tz_localize("UTC")
        else:  # object index of datetimes (e.g. CSV round-trip)
            index_utc = pd.to_datetime(index, utc=True)
        sess_names, sess_kz = sessions.tag_sessions(index_utc)

        # ---- daily (higher-timeframe) context, completed days only -----
        bar_day = index_utc.normalize()
        snap_by_day = self._daily_snapshots(df, bar_day, cfg)

        # ---- running structures -------------------------------------------
        swing_highs: list[SwingPoint] = []
        swing_lows: list[SwingPoint] = []
        pools: list[LiquidityPool] = []
        fvgs: list[FairValueGap] = []
        obs: list[OrderBlock] = []
        flips: list[FlipLevel] = []
        trend = "range"
        trend_since: int | None = None
        last_sh: SwingPoint | None = None
        last_sl: SwingPoint | None = None
        sh_broken = sl_broken = True
        last_event: StructureEvent | None = None
        last_sweep: SweepEvent | None = None
        last_bull_disp: int | None = None
        last_bear_disp: int | None = None
        last_bull_disp_fvg: int | None = None
        last_bear_disp_fvg: int | None = None

        sess_key: tuple | None = None
        sess_hi = sess_lo = float("nan")

        states: list[MarketState] = []

        for t in range(n_bars):
            a = atr_arr[t] if np.isfinite(atr_arr[t]) and atr_arr[t] > 0 else 1e-9

            # -- 1. mitigation of levels created before t -------------------
            for g in fvgs:
                if g.active and g.created_at < t:
                    if t - g.created_at > cfg.max_level_age:
                        g.mitigated_at = t
                    elif g.direction == 1 and low[t] <= g.bottom:
                        g.mitigated_at = t
                    elif g.direction == -1 and high[t] >= g.top:
                        g.mitigated_at = t
            for ob in obs:
                if ob.active and ob.created_at < t:
                    if t - ob.created_at > cfg.max_level_age:
                        ob.mitigated_at = t
                    elif ob.direction == 1 and close[t] < ob.bottom:
                        ob.mitigated_at = t
                    elif ob.direction == -1 and close[t] > ob.top:
                        ob.mitigated_at = t

            # -- 2. liquidity pool interaction (sweep / clear) --------------
            for p in pools:
                if not p.active or p.created_at >= t:
                    continue
                if t - p.created_at > cfg.max_pool_age:
                    p.cleared_at = t
                    continue
                if p.side == "bsl":
                    if high[t] > p.price and close[t] < p.price:
                        p.swept_at = t
                        last_sweep = SweepEvent(
                            index=t, level=p.price, extreme=high[t], direction=-1
                        )
                    elif close[t] > p.price:
                        p.cleared_at = t
                else:  # ssl
                    if low[t] < p.price and close[t] > p.price:
                        p.swept_at = t
                        last_sweep = SweepEvent(
                            index=t, level=p.price, extreme=low[t], direction=+1
                        )
                    elif close[t] < p.price:
                        p.cleared_at = t

            # -- 3. fractal swing confirmation ------------------------------
            i = t - cfg.fractal_n
            if i >= cfg.fractal_n:
                if self._is_swing(high, i, cfg.fractal_n, "high"):
                    sp = SwingPoint(price=float(high[i]), index=i,
                                    confirmed_at=t, kind="high")
                    swing_highs.append(sp)
                    last_sh, sh_broken = sp, False
                    self._add_pool(pools, sp.price, "bsl", t, a, cfg)
                if self._is_swing(low, i, cfg.fractal_n, "low"):
                    sp = SwingPoint(price=float(low[i]), index=i,
                                    confirmed_at=t, kind="low")
                    swing_lows.append(sp)
                    last_sl, sl_broken = sp, False
                    self._add_pool(pools, sp.price, "ssl", t, a, cfg)

            # -- 4. structure break on close --------------------------------
            event_now: StructureEvent | None = None
            if last_sh is not None and not sh_broken and close[t] > last_sh.price:
                kind = "CHOCH" if trend == "down" else "BOS"
                event_now = StructureEvent(index=t, kind=kind, direction=+1,
                                           level=last_sh.price)
                if trend != "up":
                    trend_since = t
                trend = "up"
                sh_broken = True
            elif last_sl is not None and not sl_broken and close[t] < last_sl.price:
                kind = "CHOCH" if trend == "up" else "BOS"
                event_now = StructureEvent(index=t, kind=kind, direction=-1,
                                           level=last_sl.price)
                if trend != "down":
                    trend_since = t
                trend = "down"
                sl_broken = True
            if event_now is not None:
                last_event = event_now
                ob = self._make_order_block(event_now, t, open_, high, low,
                                              close, br, cfg)
                if ob is not None:
                    obs.append(ob)
                # -- S/R flip tracking: a broken level becomes a flip
                # candidate. Broken resistance (bullish break) -> support
                # (long setup on retest); broken support (bearish break)
                # -> resistance (short setup on retest).
                if event_now.direction == 1:
                    flips.append(FlipLevel(
                        price=event_now.level, broken_at=t, direction=+1,
                        swing_index=last_sh.index if last_sh else t))
                else:
                    flips.append(FlipLevel(
                        price=event_now.level, broken_at=t, direction=-1,
                        swing_index=last_sl.index if last_sl else t))
            # deactivate flip levels that price has reclaimed (closed back
            # beyond the level = the flip failed) or that are too old
            for fl in flips:
                if not fl.active:
                    continue
                if t - fl.broken_at > cfg.flip_max_age:
                    fl.active = False
                elif fl.direction == -1 and close[t] > fl.price + 0.2 * a:
                    fl.active = False  # reclaimed: not resistance anymore
                elif fl.direction == 1 and close[t] < fl.price - 0.2 * a:
                    fl.active = False  # lost: not support anymore

            # -- 5. fair value gap creation (candles t-2, t-1, t) -----------
            if t >= 2:
                if low[t] > high[t - 2]:
                    fvgs.append(FairValueGap(top=float(low[t]),
                                             bottom=float(high[t - 2]),
                                             direction=+1, created_at=t))
                    if br[t] >= cfg.disp_body_atr:
                        last_bull_disp_fvg = t
                elif high[t] < low[t - 2]:
                    fvgs.append(FairValueGap(top=float(low[t - 2]),
                                             bottom=float(high[t]),
                                             direction=-1, created_at=t))
                    if br[t] <= -cfg.disp_body_atr:
                        last_bear_disp_fvg = t

            # -- 5b. displacement candle tracking ---------------------------
            if br[t] >= cfg.disp_body_atr:
                last_bull_disp = t
            if br[t] <= -cfg.disp_body_atr:
                last_bear_disp = t

            # -- 6. session range -------------------------------------------
            key = (index_utc[t].date(), sess_names[t])
            if key != sess_key:
                sess_key = key
                sess_hi, sess_lo = float(high[t]), float(low[t])
            else:
                sess_hi = max(sess_hi, float(high[t]))
                sess_lo = min(sess_lo, float(low[t]))

            # -- 7. snapshot -------------------------------------------------
            d_bias, d_trend, d_sh, d_sl = snap_by_day.get(
                bar_day[t], (0, "range", None, None))
            states.append(
                self._snapshot(
                    t=t, ts=index[t], close=float(close[t]), atr=float(a),
                    trend=trend, swing_highs=swing_highs, swing_lows=swing_lows,
                    pools=pools, fvgs=fvgs, obs=obs, last_event=last_event,
                    last_sweep=last_sweep, atr_pct=float(atr_pct[t]),
                    momentum=float((close[t] - close[t - 10]) / a) if t >= 10 else 0.0,
                    body=float(br[t]),
                    bull_disp=float(np.max(np.maximum(br[max(0, t - 2): t + 1], 0.0))),
                    bear_disp=float(np.max(np.maximum(-br[max(0, t - 2): t + 1], 0.0))),
                    session=sess_names[t], killzone=sess_kz[t],
                    sess_hi=sess_hi, sess_lo=sess_lo, high=high, low=low,
                    trend_since=trend_since,
                    bull_disp_age=(t - last_bull_disp) if last_bull_disp is not None else None,
                    bear_disp_age=(t - last_bear_disp) if last_bear_disp is not None else None,
                    bull_disp_fvg_age=(t - last_bull_disp_fvg) if last_bull_disp_fvg is not None else None,
                    bear_disp_fvg_age=(t - last_bear_disp_fvg) if last_bear_disp_fvg is not None else None,
                    daily_trend=d_trend, daily_bias=d_bias,
                    daily_swing_high=d_sh, daily_swing_low=d_sl,
                    ema_stack=(1 if ema9[t] > ema21[t] > ema50[t] else
                               -1 if ema9[t] < ema21[t] < ema50[t] else 0),
                    flips=flips,
                )
            )

        return states

    # -- daily context -------------------------------------------------------
    def _daily_snapshots(self, df: pd.DataFrame, bar_day: pd.DatetimeIndex,
                         cfg: ReaderConfig) -> dict:
        """Per-calendar-day (UTC) bias snapshots built from COMPLETED days.

        Returns {day: (daily_bias, daily_trend, last_d1_swing_high,
        last_d1_swing_low)} where the entry for day ``j`` uses only days
        strictly before ``j`` — fractal swings on daily bars with the same
        confirmation lag as the H1 reader, trend from daily close breaks
        of confirmed swings, and a close-vs-EMA50 tilt while the daily
        structure is still "range".

        Note: GC=F trades nearly 24h, so a "day" here is the UTC calendar
        date of the hourly bars — an approximation of the futures daily
        candle, documented as such.
        """
        if len(df) == 0:
            return {}
        daily = df.groupby(bar_day).agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"))
        days = list(daily.index)
        d_high = daily["high"].to_numpy(dtype=float)
        d_low = daily["low"].to_numpy(dtype=float)
        d_close = daily["close"].to_numpy(dtype=float)
        d_ema = indicators.ema(d_close, cfg.daily_ema_period)
        n = cfg.daily_fractal_n

        snaps: dict = {}
        trend = "range"
        last_sh: float | None = None
        last_sl: float | None = None
        sh_broken = sl_broken = True
        for j, day in enumerate(days):
            # snapshot for day j: state after completing days 0..j-1
            if trend == "up":
                bias = 1
            elif trend == "down":
                bias = -1
            elif j >= cfg.daily_min_days and np.isfinite(d_ema[j - 1]):
                diff = d_close[j - 1] - d_ema[j - 1]
                bias = 1 if diff > 0 else -1 if diff < 0 else 0
            else:
                bias = 0
            snaps[day] = (bias, trend, last_sh, last_sl)

            # complete day j: fractal swing centred at j-n confirms now
            i = j - n
            if i >= n:
                if self._is_swing(d_high, i, n, "high"):
                    last_sh, sh_broken = float(d_high[i]), False
                if self._is_swing(d_low, i, n, "low"):
                    last_sl, sl_broken = float(d_low[i]), False
            # daily structure break on day j's close
            if last_sh is not None and not sh_broken and d_close[j] > last_sh:
                trend = "up"
                sh_broken = True
            elif last_sl is not None and not sl_broken and d_close[j] < last_sl:
                trend = "down"
                sl_broken = True
        return snaps

    # -- helpers -------------------------------------------------------------
    def _add_pool(self, pools: list[LiquidityPool], price: float, side: str,
                  t: int, atr_t: float, cfg: ReaderConfig) -> None:
        tol = cfg.pool_merge_tol_atr * atr_t
        for p in pools:
            if p.active and p.side == side and abs(p.price - price) <= tol:
                p.strength += 1
                p.price = max(p.price, price) if side == "bsl" else min(p.price, price)
                return
        pools.append(LiquidityPool(price=float(price), side=side, strength=1,
                                   created_at=t))

    def _make_order_block(self, event: StructureEvent, t: int,
                          open_: np.ndarray, high: np.ndarray,
                          low: np.ndarray, close: np.ndarray,
                          br: np.ndarray, cfg: ReaderConfig) -> OrderBlock | None:
        d = event.direction
        # strongest displacement bar in the leg, same direction as the break
        m, best = None, cfg.displacement_min
        for k in range(max(0, t - cfg.ob_search), t + 1):
            if d * br[k] >= best:
                m, best = k, d * br[k]
        if m is None:
            return None
        # last opposite candle before the displacement bar
        for j in range(m, max(-1, m - cfg.ob_search - 1), -1):
            if d == 1 and close[j] < open_[j]:
                return OrderBlock(top=float(high[j]), bottom=float(low[j]),
                                  direction=+1, created_at=t, candle_index=j)
            if d == -1 and close[j] > open_[j]:
                return OrderBlock(top=float(high[j]), bottom=float(low[j]),
                                  direction=-1, created_at=t, candle_index=j)
        return None

    def _snapshot(self, *, t, ts, close, atr, trend, swing_highs, swing_lows,
                  pools, fvgs, obs, last_event, last_sweep, atr_pct, momentum,
                  body, bull_disp, bear_disp, session, killzone, sess_hi,
                  sess_lo, high, low, trend_since=None, bull_disp_age=None,
                  bear_disp_age=None, bull_disp_fvg_age=None,
                  bear_disp_fvg_age=None, daily_trend="range", daily_bias=0,
                  daily_swing_high=None, daily_swing_low=None,
                  ema_stack=0, flips=None) -> MarketState:
        cfg = self.cfg
        active_pools = [p for p in pools if p.active]
        above = sorted((p for p in active_pools if p.price > close),
                       key=lambda p: p.price)[:6]
        below = sorted((p for p in active_pools if p.price < close),
                       key=lambda p: p.price, reverse=True)[:6]

        # dealing range from confirmed swings inside the lookback
        hs = [s.price for s in swing_highs
              if s.index >= t - cfg.range_lookback and s.confirmed_at <= t]
        ls = [s.price for s in swing_lows
              if s.index >= t - cfg.range_lookback and s.confirmed_at <= t]
        if hs and ls:
            r_hi, r_lo = max(hs), min(ls)
        else:
            w = slice(max(0, t - 50), t + 1)
            r_hi, r_lo = float(np.max(high[w])), float(np.min(low[w]))
        if r_hi > r_lo:
            eq = float(np.clip((close - r_lo) / (r_hi - r_lo), 0.0, 1.0))
        else:
            eq = 0.5
        zone = "premium" if eq > 0.62 else "discount" if eq < 0.38 else "equilibrium"

        if atr_pct < 30:
            vol = "compressed"
        elif atr_pct > 72:
            vol = "expanded"
        else:
            vol = "normal"

        return MarketState(
            index=t,
            timestamp=ts,
            close=close,
            atr=atr,
            trend=trend,
            bias=1 if trend == "up" else -1 if trend == "down" else 0,
            recent_swing_highs=swing_highs[-8:],
            recent_swing_lows=swing_lows[-8:],
            last_structure_event=last_event,
            structure_event_age=(t - last_event.index) if last_event else None,
            pools_above=above,
            pools_below=below,
            last_sweep=last_sweep,
            sweep_age=(t - last_sweep.index) if last_sweep else None,
            active_fvgs=sorted((g for g in fvgs if g.active),
                               key=lambda g: g.created_at, reverse=True)[:8],
            active_obs=sorted((o for o in obs if o.active),
                              key=lambda o: o.created_at, reverse=True)[:8],
            range_high=r_hi,
            range_low=r_lo,
            eq_position=eq,
            zone=zone,
            atr_percentile=atr_pct,
            vol_regime=vol,
            momentum=momentum,
            body_ratio=body,
            bull_disp_3=bull_disp,
            bear_disp_3=bear_disp,
            bull_disp_age=bull_disp_age,
            bear_disp_age=bear_disp_age,
            bull_disp_fvg_age=bull_disp_fvg_age,
            bear_disp_fvg_age=bear_disp_fvg_age,
            trend_since=trend_since,
            daily_trend=daily_trend,
            daily_bias=daily_bias,
            daily_swing_high=daily_swing_high,
            daily_swing_low=daily_swing_low,
            ema_stack=ema_stack,
            active_flips=[fl for fl in (flips or []) if fl.active],
            session=session,
            killzone=killzone,
            session_high=sess_hi,
            session_low=sess_lo,
        )
