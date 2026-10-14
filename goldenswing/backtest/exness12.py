"""Exness $12 -> $100 TIMING backtest. NOT OANDA. NOT R-grinding.

User correction (2026-10-13): Exness Standard MT5, max leverage 1:2000,
min spread 0.2, commission 0.00 USD per side. Min lot 0.01.

A $12 account for fast growth is NOT about stop-loss / 1% R grinding.
It is about TIMING + leverage: catching sweep -> displacement -> impulse,
riding the move, exiting on reversal / opposing liquidity. Stop is
structural invalidation / margin-call reality, NOT the edge.

Specs (from user + screenshot, account number never recorded):
- Starting balance $12.00
- Fixed size 0.01 lot = 1 oz (1 lot XAUUSD = 100 oz). Lot scaling tested
  separately: 0.02 lot = 2 oz, 0.05 lot = 5 oz.
- Leverage configurable 1:2 (lowest) to unlimited (user-explicit).
  Baseline tested: 1:2000. Margin per trade = notional / leverage.
  At $4000 gold: 1:2000 -> ~$2.00 margin. 1:2 -> ~$2000 margin, so a
  $12 account CANNOT open even 0.01 lot at 1:2 (shown as impossible,
  not viable). Unlimited -> margin ~$0. Free margin = equity - margin.
  CRITICAL: leverage changes MARGIN, not PnL per lot. PnL per 0.01 lot
  stays $1 per $1 gold move at any leverage. Unlimited only increases
  PnL if lot size scales up, which also scales loss per move and
  blow-up speed: $12 adverse at 0.01 = blow, $6 at 0.02 = blow,
  $2.40 at 0.05 = blow (before spread).
- Spread 0.2 interpreted as $0.20 per oz per trade (round-trip cost on
  entry, scales with lot: 0.01 -> $0.20, 0.02 -> $0.40, 0.05 -> $1.00).
  Commission 0.00. Slippage assumption 0.00 (stated, optimistic).
- LIMITATION (not modeled, do not invent a rule): Exness can restrict /
  reduce leverage around news / weekends / high-margin-requirement
  periods. This backtest uses constant leverage per run. Real live
  leverage may drop during those windows, raising margin and
  stop-out risk beyond what is modeled here.
- Stop-out: Exness Standard margin call 60%, stop-out 0% (verified
  against Exness Help Center, Margin call and stop out levels by
  account type, plus multiple secondary sources). Margin level =
  equity / used_margin * 100. Stop-out when equity <= 0.
  Margin call (60%) is a warning only, never closes.
  Negative balance protection restores to $0 (assumed, not relied on
  for PnL beyond flooring at $0).
- No $0.12 / 1% risk, no R-multiples, no daily -2R limit, no max-stop
  gate (wide stops are NOT rejected; margin is the real constraint).
  A trade whose structural stop is wider than margin room will
  stop-out before the structural stop if price goes that far.

Signal logic (reuses real_trader.py helpers, same causal rules):
- HTF veto, never democracy. Bias != 0 required.
- Sweep (Asia/PDH/PDL/range) -> displacement (strong body + FVG) ->
  retracement -> LIMIT fill AT POI mid. Gap out = skip. No retracement
  = no trade. No chasing.
- AMD: only Manipulation -> Distribution.
- Killzones: Asia 00-06, London KZ 07-10, NY KZ 12-15 UTC.
- Target = nearest opposing liquidity pool with move >= MIN_TARGET_MOVE
  ($1.00, see below). If no pool qualifies, skip.
- Exit = stop-out blow-up (equity <= 0) > structural stop > target >
  opposing structure break > emergency time stop only.

MIN_TARGET_MOVE = $1.00: a target closer than $1.00 cannot cover the
$0.20 spread + noise in a timing model. This is a stated assumption,
not market truth. It replaces the old >=1.5R rule (R does not exist
here).

Data: real OANDA spot XAU candles used as proxy for Exness spot.
Basis is immaterial for timing logic (stated). Coarser frames resampled
from finer via build_tf_frames; finer never reconstructed from coarser.

PnL in $: PnL_net = direction * (exit - entry) * 1 oz - spread - commission
- slippage. Balance evolves trade by trade (compounding). One position
at a time. If balance < margin at entry, cannot open -> simulation ends
as "cannot_open" (account too small to open 0.01 lot).

Blow-up: if equity hits <= 0 intra-trade, position closed at stop-out
price, balance = $0.00, simulation stops (reset_on_blowup=False) or
resets to $12 and continues counting blow-ups (reset_on_blowup=True,
for blow-up-rate estimation).
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from ..live.multitf import PRESETS, TimeframeConfig, atr14, check_hard_veto, evaluate_bias, find_swings, get_preset
from ..live.revival import (VARIANTS as REVIVAL_VARIANTS, body_stop_level, find_breaker, godce_allow, has_bodybreak, has_engulfing, marco_entry, ote_zone, retest_edge_limit)
from .mtf12 import TF_MINUTES, build_tf_frames, infer_tf, load_ohlc_csv, session_of
from .real_trader import EMERGENCY_BARS, PATIENCE_BARS, daily_pools, detect_displacement, detect_sweep, killzone_of

# Exness Standard MT5 specs (user screenshot + verified stop-out)
START_BALANCE_12 = 12.00
LOT_OZ = 1.0  # 0.01 lot XAUUSD = 1 oz
LEVERAGE = 2000.0
SPREAD_PER_OZ = 0.20  # 0.2 interpreted as $0.20/oz
COMMISSION_PER_TRADE = 0.00
MARGIN_CALL_LEVEL = 60.0  # % margin level, warning only
STOP_OUT_LEVEL = 0.0  # % margin level, equity <= 0 closes
MIN_TARGET_MOVE = 1.00  # $ minimum impulse to opposing pool
WINDOW = 300


def margin_required(entry: float, lot_oz: float = LOT_OZ, leverage: float = LEVERAGE) -> float:
    """Margin = notional / leverage. Notional = entry * oz.

    Unlimited leverage (inf) -> margin ~$0. 1:2 -> ~$2000 at $4000 gold.
    """
    if leverage is None or (isinstance(leverage, float) and math.isinf(leverage)):
        return 0.0
    if leverage <= 0:
        raise ValueError("leverage must be positive or inf for unlimited")
    return float(entry * lot_oz / leverage)


def stop_out_price(entry: float, direction: int, balance_before: float, spread: float = SPREAD_PER_OZ, lot_oz: float = LOT_OZ) -> float:
    """Price at which equity hits $0 (stop-out 0%).

    Equity(t) = balance_before + direction*(price-entry)*oz - spread_total,
    where spread_total = spread_per_oz * oz.
    Set = 0 -> price = entry -/+ (balance_before - spread_total) / oz.
    For 1 oz: long stop-out = entry - (balance - spread).
    For 2 oz: long stop-out = entry - (balance - 2*spread)/2.
    """
    spread_total = float(spread * lot_oz)
    room = float(balance_before - spread_total)
    if direction == 1:
        return float(entry - room / lot_oz)
    return float(entry + room / lot_oz)


def parse_leverage(value: str | float) -> float:
    """Parse CLI leverage: 'unlimited' / 'inf' -> inf, else float."""
    if isinstance(value, float):
        return value
    s = str(value).strip().lower()
    if s in ("unlimited", "inf", "infinity", "1:unlimited", "1:inf"):
        return float("inf")
    # allow "1:2000" or "2000"
    if ":" in s:
        s = s.split(":")[-1]
    return float(s)


def leverage_label(leverage: float) -> str:
    if isinstance(leverage, float) and math.isinf(leverage):
        return "1:Unlimited"
    return f"1:{leverage:.0f}"


def causal_asia_arrays(trigger: pd.DataFrame) -> tuple[list[float | None], list[float | None]]:
    """Rolling Asia (00-06 UTC) high/low using only bars <= i. No lookahead.

    pools_by_date Asia is full-day (future bars leak). This is the causal replacement.
    """
    highs = trigger["high"].to_numpy(dtype=float)
    lows = trigger["low"].to_numpy(dtype=float)
    times = trigger.index
    high_out: list[float | None] = [None] * len(trigger)
    low_out: list[float | None] = [None] * len(trigger)
    cur_date = None
    ah: float | None = None
    al: float | None = None
    for idx, ts in enumerate(times):
        d = ts.date()
        if d != cur_date:
            cur_date = d
            ah = None
            al = None
        if 0 <= ts.hour < 6:
            h = float(highs[idx])
            l = float(lows[idx])
            ah = h if ah is None else max(ah, h)
            al = l if al is None else min(al, l)
        high_out[idx] = ah
        low_out[idx] = al
    return high_out, low_out


def _summarize_dollar(trades: list[dict]) -> dict:
    if not trades:
        return {"total_trades": 0, "wins": 0, "win_rate": 0.0, "total_pnl": 0.0,
                "avg_pnl": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "profit_factor": 0.0, "max_drawdown_dollar": 0.0, "moves_caught": 0}
    pnls = [t["pnl_net"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    if losses and sum(losses) != 0:
        pf = sum(wins) / abs(sum(losses))
    elif wins:
        pf = float("inf")
    else:
        pf = 0.0
    # max drawdown on balance curve is computed by caller; here use cumulative PnL curve
    peak, max_dd, cum = 0.0, 0.0, 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return {"total_trades": len(trades), "wins": len(wins),
            "win_rate": 100.0 * len(wins) / len(trades),
            "total_pnl": float(sum(pnls)), "avg_pnl": float(sum(pnls) / len(pnls)),
            "avg_win": float(sum(wins) / len(wins)) if wins else 0.0,
            "avg_loss": float(sum(losses) / len(losses)) if losses else 0.0,
            "profit_factor": float(pf), "max_drawdown_dollar": float(max_dd),
            "moves_caught": int(sum(1 for t in trades if t["exit_reason"] == "target"))}


def run_exness_backtest(df_by_tf: dict[str, pd.DataFrame], preset: TimeframeConfig, *,
                        start_balance: float = START_BALANCE_12,
                        leverage: float = LEVERAGE,
                        spread_per_oz: float = SPREAD_PER_OZ,
                        commission_per_trade: float = COMMISSION_PER_TRADE,
                        slippage_per_oz: float = 0.0,
                        lot_oz: float = LOT_OZ,
                        reset_on_blowup: bool = False,
                        enable_veto: bool = True,
                        variant: str = "base",
                        godce_filter: bool = False) -> dict:
    """Run Exness timing logic over the trigger frame. One position at a time.

    ``enable_veto=False`` skips the hard veto (before/after runs).
    The result carries ``veto_blocked`` and ``veto_enabled``.
    Veto TFs absent from ``df_by_tf`` are fail-open (no veto).

    ``variant`` selects a revived strategy variant (see
    ``goldenswing/live/revival.py``): base | ote | engulf | retest |
    breaker_t3 | marco | body_stop | bodybreak. All variants keep
    hard veto + Exness timing + LIMIT discipline except ``marco``,
    which deliberately skips the confirmation wait (that IS the
    experiment, flagged high-risk). ``godce_filter=True`` adds the
    GODCE regime filter (filter only, never an entry).
    """
    if variant not in REVIVAL_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {REVIVAL_VARIANTS}")
    missing = [tf for tf in (preset.bias_tf, preset.zone_tf, preset.trigger_tf) if tf not in df_by_tf]
    if missing:
        raise ValueError(f"missing frames for TFs {missing}; call build_tf_frames first")
    trigger = df_by_tf[preset.trigger_tf].sort_index()
    zone = df_by_tf[preset.zone_tf].sort_index()
    bias_df = df_by_tf[preset.bias_tf].sort_index()
    veto_dfs: dict[str, pd.DataFrame] = {}
    for vtf in (getattr(preset, "veto_tf", None),
                getattr(preset, "veto_tf2", None)):
        if vtf and vtf in df_by_tf:
            veto_dfs[vtf] = df_by_tf[vtf].sort_index()
    n = len(trigger)
    if n < 2:
        raise ValueError("trigger frame too short")
    pools_by_date = daily_pools(trigger)
    # CAUSAL Asia pools: daily_pools() computes Asia high/low from the FULL
    # day, including bars after t (lookahead). For Exness we must use only
    # Asia bars with index <= i. PDH/PDL from pools_by_date are previous-day
    # only, so they are causal.
    opens = trigger["open"].to_numpy(dtype=float)
    highs = trigger["high"].to_numpy(dtype=float)
    lows = trigger["low"].to_numpy(dtype=float)
    closes = trigger["close"].to_numpy(dtype=float)
    times = trigger.index
    asia_high_causal, asia_low_causal = causal_asia_arrays(trigger)

    balance = float(start_balance)
    equity_curve = [float(start_balance)]
    peak_balance = float(start_balance)
    max_dd_dollar = 0.0
    trades: list[dict] = []
    blocked_trades: list[dict] = []
    rejects: Counter = Counter()
    veto_blocked_by_direction: Counter = Counter()
    blocked_blowups = 0
    evaluated = 0
    blowups = 0
    cannot_open = False
    emergency = EMERGENCY_BARS.get(preset.trigger_tf, 120)
    i = 30
    while i <= n - 2:
        t = times[i]
        day = t.date()
        zpos = int(zone.index.searchsorted(t, side="right"))
        bpos = int(bias_df.index.searchsorted(t, side="right"))
        trig_win = trigger.iloc[max(0, i - WINDOW + 1):i + 1]
        zone_win = zone.iloc[max(0, zpos - WINDOW):zpos]
        bias_win = bias_df.iloc[max(0, bpos - WINDOW):bpos]
        evaluated += 1
        # Shadow flag: veto-blocked setup evaluated as a COUNTERFACTUAL
        # (what the blocked trade would have done, same entry/exit
        # logic) into blocked_trades.  It never touches balance,
        # equity_curve, or blowups, and the main loop advances i+1.
        shadow_blocked = False
        bias = evaluate_bias(bias_win, swing_n=preset.swing_n)
        if bias == 0:
            rejects["bias_flat"] += 1
            i += 1
            continue
        # HARD VETO: next-higher TF strongly disagreeing with the main
        # bias kills the setup outright.  A vetoed setup FALLS THROUGH
        # to the same pipeline below in shadow mode, so blocked_trades
        # measures what the veto removed instead of guessing it.
        if enable_veto:
            vetoed = False
            for vtf, vdf in veto_dfs.items():
                vpos = int(vdf.index.searchsorted(t, side="right"))
                vwin = vdf.iloc[max(0, vpos - WINDOW):vpos]
                if check_hard_veto(vwin, bias,
                                   swing_n=preset.swing_n)["vetoed"]:
                    vetoed = True
                    break
            if vetoed:
                rejects["hard_veto"] += 1
                veto_blocked_by_direction[
                    "long" if bias == 1 else "short"] += 1
                shadow_blocked = True
        direction = bias
        atr_t = atr14(trig_win)
        if atr_t <= 0:
            rejects["no_atr"] += 1
            i += 1
            continue
        th, tl = find_swings(trig_win["high"].to_numpy(), trig_win["low"].to_numpy(), preset.swing_n)
        pool_vals: list[float] = []
        if direction == 1:
            pool_vals += [p for _, p in tl[-3:]]
        else:
            pool_vals += [p for _, p in th[-3:]]
        dp = pools_by_date.get(day, {})
        if dp.get("pdl") is not None and direction == 1:
            pool_vals.append(dp["pdl"])
        if dp.get("pdh") is not None and direction == -1:
            pool_vals.append(dp["pdh"])
        # Causal Asia only (bars <= i). pools_by_date Asia is full-day lookahead, never use it.
        if asia_low_causal[i] is not None and direction == 1:
            pool_vals.append(asia_low_causal[i])
        if asia_high_causal[i] is not None and direction == -1:
            pool_vals.append(asia_high_causal[i])
        # GODCE regime filter (filter only, never an entry). Blocks
        # counter-regime MOM trades; every other mode allows both sides.
        if godce_filter:
            allowed, _mode_name = godce_allow(direction, bias_win)
            if not allowed:
                rejects["godce_regime_block"] += 1
                i += 1
                continue
        sweep = detect_sweep(trig_win, pool_vals, direction, atr_t)
        if sweep is None:
            rejects["no_sweep"] += 1
            i += 1
            continue
        # Absolute trigger-frame index of the sweep bar (trig_win ends
        # at i, so window start = i - len + 1).
        _win_start = i - len(trig_win) + 1
        sweep_abs = _win_start + int(sweep["bar_pos"])
        # Variant dispatch. marco skips displacement entirely (enter at
        # next-bar open after the sweep). bodybreak replaces
        # displacement with a close-through of the sweep candle's body.
        # Every other variant keeps sweep -> displacement.
        disp = None
        entry_override = None
        stop_override = None
        stop_mode = "sweep"
        prefilled = False
        filled_idx = None
        if variant == "marco":
            if i + 1 > n - 1:
                rejects["no_next_bar"] += 1
                i += 1
                continue
            entry_override, stop_override = marco_entry(
                float(opens[i + 1]), float(sweep["extreme"]), direction,
                float(atr_t))
            stop_mode = "marco"
            prefilled = True
            filled_idx = i + 1
            # POI band for bookkeeping: sweep extreme <-> entry.
            poi_top = max(entry_override, float(sweep["extreme"]))
            poi_bottom = min(entry_override, float(sweep["extreme"]))
            amd_phase = "manipulation_to_distribution"
            poi_mid = 0.5 * (poi_top + poi_bottom)
            kz = killzone_of(t)
            gap_skip = False
            last_pat = filled_idx
        elif variant == "bodybreak":
            if not has_bodybreak(opens, closes, sweep_abs,
                                 int(sweep["bar_pos"]),
                                 len(trig_win) - 1, direction):
                rejects["no_bodybreak"] += 1
                i += 1
                continue
            # POI: displacement FVG when one exists, else the sweep
            # candle's body band (LIMIT still at a real zone, never
            # mid-air).
            disp = detect_displacement(trig_win, atr_t, direction)
            if disp is not None:
                poi_top, poi_bottom = disp["fvg_top"], disp["fvg_bottom"]
            else:
                poi_top = max(float(opens[sweep_abs]),
                              float(closes[sweep_abs]))
                poi_bottom = min(float(opens[sweep_abs]),
                                 float(closes[sweep_abs]))
            if poi_top <= poi_bottom:
                rejects["bad_poi"] += 1
                i += 1
                continue
            amd_phase = "manipulation_to_distribution"
            poi_mid = 0.5 * (poi_top + poi_bottom)
            kz = killzone_of(t)
            filled_idx = None
            gap_skip = False
            last_pat = min(i + PATIENCE_BARS, n - 1)
            for j in range(i + 1, last_pat + 1):
                if direction == 1 and opens[j] < poi_bottom:
                    gap_skip = True
                    break
                if direction == -1 and opens[j] > poi_top:
                    gap_skip = True
                    break
                if lows[j] <= poi_top and highs[j] >= poi_bottom:
                    filled_idx = j
                    break
        else:
            disp = detect_displacement(trig_win, atr_t, direction)
            if disp is None:
                rejects["no_displacement"] += 1
                i += 1
                continue
            if sweep["bar_pos"] >= len(trig_win) - 1:
                rejects["sweep_displacement_order"] += 1
                i += 1
                continue
            if variant == "engulf":
                _tw = trig_win
                if not has_engulfing(
                        _tw["open"].to_numpy(dtype=float),
                        _tw["high"].to_numpy(dtype=float),
                        _tw["low"].to_numpy(dtype=float),
                        _tw["close"].to_numpy(dtype=float),
                        len(_tw) - 1, direction):
                    rejects["no_engulf"] += 1
                    i += 1
                    continue
            amd_phase = "manipulation_to_distribution"
            poi_top, poi_bottom = disp["fvg_top"], disp["fvg_bottom"]
            if variant == "ote":
                # OTE zone: 62-79% retracement of the sweep->disp leg.
                # Leg extreme in the trade direction over bars
                # [sweep_pos, end of window].
                _seg_h = trig_win["high"].to_numpy(dtype=float)[int(sweep["bar_pos"]):]
                _seg_l = trig_win["low"].to_numpy(dtype=float)[int(sweep["bar_pos"]):]
                disp_extreme = (float(_seg_h.max()) if direction == 1
                                else float(_seg_l.min()))
                poi_bottom, poi_top, _ote_limit = ote_zone(
                    float(sweep["extreme"]), disp_extreme, direction)
                entry_override = float(_ote_limit)
            if poi_top <= poi_bottom:
                rejects["bad_poi"] += 1
                i += 1
                continue
            poi_mid = 0.5 * (poi_top + poi_bottom)
            kz = killzone_of(t)
            if variant == "breaker_t3":
                # Placement T3: touch then close-through against the
                # thesis within the patience window. Trade the breaker.
                _last_pat = min(i + PATIENCE_BARS, n - 1)
                brk = find_breaker(opens, highs, lows, closes, i,
                                   _last_pat, float(poi_top),
                                   float(poi_bottom), direction)
                if brk is None:
                    rejects["no_breaker"] += 1
                    i += 1
                    continue
                # Veto re-check against the BREAKER direction (breaker
                # is counter-bias by construction; it must not fight
                # the veto TF). Missing veto TF = fail-open.
                if enable_veto:
                    _brk_vetoed = False
                    for _vtf, _vdf in veto_dfs.items():
                        _vpos = int(_vdf.index.searchsorted(
                            times[brk["violation_idx"]], side="right"))
                        _vwin = _vdf.iloc[max(0, _vpos - WINDOW):_vpos]
                        if check_hard_veto(
                                _vwin, int(brk["direction"]),
                                swing_n=preset.swing_n)["vetoed"]:
                            _brk_vetoed = True
                            break
                    if _brk_vetoed:
                        rejects["breaker_veto"] += 1
                        i += 1
                        continue
                direction = int(brk["direction"])
                entry_override = float(brk["entry"])
                filled_idx = int(brk["violation_idx"])
                prefilled = True
                # Breaker stop: far edge of the consumed zone + buffer.
                # Long breaker: stop below the zone bottom.
                # Short breaker: stop above the zone top.
                stop_override = (float(poi_bottom) - 0.1 * atr_t
                                 if direction == 1 else
                                 float(poi_top) + 0.1 * atr_t)
                stop_mode = "breaker"
                gap_skip = False
                last_pat = filled_idx
            else:
                # patient LIMIT
                filled_idx = None
                gap_skip = False
                last_pat = min(i + PATIENCE_BARS, n - 1)
                for j in range(i + 1, last_pat + 1):
                    if direction == 1 and opens[j] < poi_bottom:
                        gap_skip = True
                        break
                    if direction == -1 and opens[j] > poi_top:
                        gap_skip = True
                        break
                    if lows[j] <= poi_top and highs[j] >= poi_bottom:
                        filled_idx = j
                        break
        if gap_skip:
            rejects["gap_out_of_poi"] += 1
            i = i + 1 if shadow_blocked else (j + 1 if j + 1 > i else i + 1)
            continue
        if filled_idx is None:
            rejects["no_retracement"] += 1
            i = i + 1 if shadow_blocked else last_pat + 1
            continue
        # Retest variant: LIMIT at the near edge (not mid). The fill
        # scan already proved the edge was touched (band touch from
        # the retracement side reaches the near edge).
        if variant == "retest" and not prefilled:
            entry_override, _retest_far = retest_edge_limit(
                float(poi_top), float(poi_bottom), direction)
            entry_override = float(entry_override)
        # OTE honesty: the fill bar must actually reach the 70% limit
        # price, not merely touch the 62-79% zone edge. A touch that
        # never reaches the limit is a missed fill, not a fill.
        if variant == "ote" and not prefilled and entry_override is not None:
            _reached = (lows[filled_idx] <= entry_override if direction == 1
                        else highs[filled_idx] >= entry_override)
            if not _reached:
                rejects["ote_limit_not_reached"] += 1
                i = i + 1 if shadow_blocked else filled_idx + 1
                continue
        entry = float(entry_override) if entry_override is not None else float(poi_mid)
        # margin check at entry: need balance >= margin to open 0.01 lot.
        # Shadow trades never open: insufficient margin is just a
        # shadow reject, never cannot_open / break.
        marg = margin_required(entry, lot_oz=lot_oz, leverage=leverage)
        if balance < marg:
            rejects["insufficient_margin"] += 1
            if shadow_blocked:
                i += 1
                continue
            cannot_open = True
            break
        if balance <= spread_per_oz * lot_oz:
            rejects["insufficient_for_spread"] += 1
            if shadow_blocked:
                i += 1
                continue
            cannot_open = True
            break
        # structural stop (invalidation, NOT a rejection gate)
        buf = 0.1 * atr_t
        if stop_override is not None:
            # marco (level + 0.25 ATR) / breaker (far edge + 0.1 ATR)
            # already carry their buffer; do not add buf again.
            stop = float(stop_override)
            if direction == 1 and not stop < entry:
                rejects["stop_invalid"] += 1
                i = i + 1 if shadow_blocked else filled_idx + 1
                continue
            if direction == -1 and not stop > entry:
                rejects["stop_invalid"] += 1
                i = i + 1 if shadow_blocked else filled_idx + 1
                continue
        elif variant == "body_stop":
            # Body-anchored stop (original body_stop.py): sweep-candle
            # body extreme +/- 0.15 ATR. No extra buffer on top.
            stop = float(body_stop_level(opens, closes, sweep_abs,
                                         direction, float(atr_t)))
            if direction == 1 and not stop < entry:
                rejects["stop_invalid"] += 1
                i = i + 1 if shadow_blocked else filled_idx + 1
                continue
            if direction == -1 and not stop > entry:
                rejects["stop_invalid"] += 1
                i = i + 1 if shadow_blocked else filled_idx + 1
                continue
        elif variant == "retest":
            # Retest stop: far edge of the zone + buffer.
            _, _far = retest_edge_limit(float(poi_top), float(poi_bottom),
                                        direction)
            stop = float(_far - buf) if direction == 1 else float(_far + buf)
            if direction == 1 and not stop < entry:
                rejects["stop_invalid"] += 1
                i = i + 1 if shadow_blocked else filled_idx + 1
                continue
            if direction == -1 and not stop > entry:
                rejects["stop_invalid"] += 1
                i = i + 1 if shadow_blocked else filled_idx + 1
                continue
        elif direction == 1:
            stop = float(sweep["extreme"] - buf)
            if not stop < entry:
                rejects["stop_invalid"] += 1
                i = i + 1 if shadow_blocked else filled_idx + 1
                continue
        else:
            stop = float(sweep["extreme"] + buf)
            if not stop > entry:
                rejects["stop_invalid"] += 1
                i = i + 1 if shadow_blocked else filled_idx + 1
                continue
        stop_dist = abs(entry - stop)
        # target: nearest opposing pool with move >= MIN_TARGET_MOVE
        cands: list[float] = []
        if direction == 1:
            if dp.get("pdh") is not None:
                cands.append(dp["pdh"])
            if asia_high_causal[i] is not None:
                cands.append(asia_high_causal[i])
            zh, _ = find_swings(zone_win["high"].to_numpy(), zone_win["low"].to_numpy(), preset.swing_n)
            cands += [p for _, p in zh if p > entry]
        else:
            if dp.get("pdl") is not None:
                cands.append(dp["pdl"])
            if asia_low_causal[i] is not None:
                cands.append(asia_low_causal[i])
            _, zl = find_swings(zone_win["high"].to_numpy(), zone_win["low"].to_numpy(), preset.swing_n)
            cands += [p for _, p in zl if p < entry]
        target = None
        target_pool = None
        for pool in sorted(cands, key=lambda p: abs(p - entry)):
            move = direction * (pool - entry)
            if move >= MIN_TARGET_MOVE:
                target, target_pool = float(pool), float(pool)
                break
        if target is None:
            rejects["target_too_small"] += 1
            i = i + 1 if shadow_blocked else filled_idx + 1
            continue
        target_move = direction * (target - entry)
        balance_before = float(balance)
        spread_total = float(spread_per_oz * lot_oz)
        so_price = stop_out_price(entry, direction, balance_before, spread=spread_per_oz, lot_oz=lot_oz)
        margin_level_entry = (balance_before / marg * 100.0) if marg > 0 else float("inf")
        # exit scan: MUST start at filled_idx+1, NOT filled_idx.
        # Fill is intra-bar at POI mid. Using the fill bar's own high/low
        # for exit assumes you can enter AND exit in the same bar, which
        # is lookahead (72/93 trades exited on hold_bars==0 before fix,
        # inflating win rate to ~84%). Exit only on bars AFTER entry.
        last = min(filled_idx + emergency, n - 1)
        exit_price, exit_idx, reason = float(closes[last]), last, "timeout"
        is_blowup = False
        min_equity = float(balance_before)
        max_adverse = 0.0
        # If fill is on the last bar, no exit bar exists: exit at close (timeout).
        _exit_start = filled_idx + 1 if filled_idx + 1 <= last else filled_idx
        for k in range(_exit_start, last + 1):
            # track min equity / max adverse for this trade
            if direction == 1:
                adverse = entry - float(lows[k])
                eq_low = balance_before + (float(lows[k]) - entry) * lot_oz - spread_total
            else:
                adverse = float(highs[k]) - entry
                eq_low = balance_before + (entry - float(highs[k])) * lot_oz - spread_total
            max_adverse = max(max_adverse, float(adverse))
            min_equity = min(min_equity, float(eq_low))
            if direction == 1:
                # stop-out first (conservative: worst case wins)
                if lows[k] <= so_price:
                    exit_price = so_price if opens[k] > so_price else float(opens[k])
                    exit_idx, reason, is_blowup = k, "stop_out_blowup", True
                    break
                if lows[k] <= stop:
                    exit_price = stop if opens[k] > stop else float(opens[k])
                    exit_idx, reason = k, "stop"
                    break
                if highs[k] >= target:
                    exit_price = target if opens[k] < target else float(opens[k])
                    exit_idx, reason = k, "target"
                    break
                if k > filled_idx + 2:
                    w = trigger.iloc[max(0, k - 50):k + 1]
                    _, wl = find_swings(w["high"].to_numpy(), w["low"].to_numpy(), 3)
                    if len(wl) >= 2 and closes[k] < wl[-1][1] - buf:
                        exit_price = float(closes[k])
                        exit_idx, reason = k, "structure_break"
                        break
            else:
                if highs[k] >= so_price:
                    exit_price = so_price if opens[k] < so_price else float(opens[k])
                    exit_idx, reason, is_blowup = k, "stop_out_blowup", True
                    break
                if highs[k] >= stop:
                    exit_price = stop if opens[k] < stop else float(opens[k])
                    exit_idx, reason = k, "stop"
                    break
                if lows[k] <= target:
                    exit_price = target if opens[k] > target else float(opens[k])
                    exit_idx, reason = k, "target"
                    break
                if k > filled_idx + 2:
                    w = trigger.iloc[max(0, k - 50):k + 1]
                    wh, _ = find_swings(w["high"].to_numpy(), w["low"].to_numpy(), 3)
                    if len(wh) >= 2 and closes[k] > wh[-1][1] + buf:
                        exit_price = float(closes[k])
                        exit_idx, reason = k, "structure_break"
                        break
        move = direction * (exit_price - entry) * lot_oz
        # Spread scales with lot (per-oz). Commission is flat per trade (0.00 here).
        pnl_net = float(move - spread_total - commission_per_trade - slippage_per_oz * lot_oz)
        if is_blowup:
            # Exness negative balance protection floors at $0
            balance_after = 0.0
            pnl_net = float(-balance_before)  # lose everything, not more
            if shadow_blocked:
                blocked_blowups += 1
            else:
                blowups += 1
        else:
            balance_after = float(balance_before + pnl_net)
            # floor at 0 (negative balance protection)
            if balance_after < 0:
                balance_after = 0.0
                pnl_net = float(-balance_before)
                is_blowup = True
                if shadow_blocked:
                    blocked_blowups += 1
                else:
                    blowups += 1
                reason = "stop_out_blowup"
        # Unlimited (margin 0) has no margin-call level; only equity stop-out applies.
        margin_call_touched = bool(marg > 0 and min_equity <= (MARGIN_CALL_LEVEL / 100.0) * marg)
        trade = {
            "signal_index": int(i), "entry_index": int(filled_idx),
            "exit_index": int(exit_idx), "exit_reason": reason,
            "direction": int(direction), "entry": float(entry),
            "stop": float(stop), "target": float(target),
            "target_pool": target_pool, "target_move": float(target_move),
            "exit_price": float(exit_price),
            "move_dollar": float(move), "pnl_net": float(pnl_net),
            "balance_before": float(balance_before), "balance_after": float(balance_after),
            "margin_required": float(marg), "margin_level_entry": float(margin_level_entry),
            "stop_out_price": float(so_price), "is_blowup": bool(is_blowup),
            "margin_call_touched": bool(margin_call_touched),
            "min_equity": float(min_equity), "max_adverse": float(max_adverse),
            "stop_dist": float(stop_dist),
            "lot_oz": float(lot_oz), "leverage": float(leverage),
            "session": session_of(times[i]), "killzone": kz,
            "regime": "trend" if bias != 0 else "range", "htf_state": int(bias),
            "amd_phase": amd_phase,
            "hold_bars": int(exit_idx - filled_idx),
            "hold_minutes": float((times[exit_idx] - times[filled_idx]).total_seconds() / 60.0),
            "zone_level": float(poi_mid), "trigger": f"sweep_disp_{'bull' if direction == 1 else 'bear'}",
            "poi_top": float(poi_top), "poi_bottom": float(poi_bottom),
            "sweep_pool": float(sweep["pool"]), "sweep_extreme": float(sweep["extreme"]),
            "stop_atr": float(stop_dist / atr_t) if atr_t else 0.0,
            "entry_type": f"limit_poi:{variant}" if variant != "marco" else "market_next_open:marco",
            "veto_blocked": bool(shadow_blocked),
        }
        if shadow_blocked:
            # Counterfactual only: never touches balance / equity curve.
            blocked_trades.append(trade)
            i = i + 1
            continue
        trades.append(trade)
        if is_blowup:
            if reset_on_blowup:
                balance = float(start_balance)
                equity_curve.append(float(balance))
                i = exit_idx + 1
                continue
            balance = 0.0
            equity_curve.append(0.0)
            break
        balance = float(balance_after)
        equity_curve.append(float(balance))
        peak_balance = max(peak_balance, float(balance))
        # max DD on balance curve
        peak_so_far = max(equity_curve)
        max_dd_dollar = max(max_dd_dollar, peak_so_far - float(balance))
        if balance < marg:
            # cannot open next trade, but let loop hit insufficient_margin naturally;
            # continue to allow rejects to record
            pass
        i = exit_idx + 1 if exit_idx + 1 > i else i + 1

    summary = _summarize_dollar(trades)
    # balance-curve DD (more honest than PnL-curve DD for compounding)
    peak_c, dd_c = float(start_balance), 0.0
    for b in equity_curve:
        peak_c = max(peak_c, b)
        dd_c = max(dd_c, peak_c - b)
    by_killzone = {}
    for key in ("Asia", "London KZ", "NY KZ", "Off-KZ"):
        by_killzone[key] = _summarize_dollar([t for t in trades if t["killzone"] == key])
    by_session = {}
    for key in ("Asia", "London", "Overlap", "NY"):
        by_session[key] = _summarize_dollar([t for t in trades if t["session"] == key])
    by_direction = {
        "long": _summarize_dollar([t for t in trades if t["direction"] == 1]),
        "short": _summarize_dollar([t for t in trades if t["direction"] == -1]),
    }
    by_regime = {key: _summarize_dollar([t for t in trades if t["regime"] == key]) for key in ("trend", "range")}
    blocked_by_direction = {
        "long": _summarize_dollar([t for t in blocked_trades if t["direction"] == 1]),
        "short": _summarize_dollar([t for t in blocked_trades if t["direction"] == -1]),
    }
    return {
        "preset": {"bias_tf": preset.bias_tf, "zone_tf": preset.zone_tf, "trigger_tf": preset.trigger_tf, "swing_n": preset.swing_n},
        "engine": "exness12_timing_v1",
        "variant": variant,
        "godce_filter": bool(godce_filter),
        "start_balance": float(start_balance),
        "final_balance": float(balance),
        "growth_multiple": float(balance / start_balance) if start_balance else 0.0,
        "peak_balance": float(peak_balance),
        "max_drawdown_dollar": float(dd_c),
        "leverage": float(leverage), "lot_oz": float(lot_oz),
        "spread_per_oz": float(spread_per_oz), "commission_per_trade": float(commission_per_trade),
        "slippage_per_oz": float(slippage_per_oz),
        "margin_call_level": MARGIN_CALL_LEVEL, "stop_out_level": STOP_OUT_LEVEL,
        "n_trigger_bars": n, "evaluated_bars": evaluated,
        "rejects": dict(rejects),
        "veto_blocked": int(rejects.get("hard_veto", 0)),
        "veto_enabled": bool(enable_veto),
        "veto_blocked_by_direction": dict(veto_blocked_by_direction),
        "blocked_trades": blocked_trades,
        "blocked_summary": _summarize_dollar(blocked_trades),
        "blocked_by_direction": blocked_by_direction,
        "blocked_blowups": int(blocked_blowups),
        "trades": trades, "summary": summary,
        "equity_curve": equity_curve,
        "blowups": int(blowups), "blow_up": bool(blowups > 0),
        "cannot_open": bool(cannot_open),
        "by_killzone": by_killzone, "by_session": by_session,
        "by_direction": by_direction, "by_regime": by_regime,
    }


def _row_dollar(label: str, s: dict) -> str:
    pf = "inf" if math.isinf(s["profit_factor"]) else f"{s['profit_factor']:.2f}"
    return (f"| {label} | {s['total_trades']} | {s['win_rate']:.1f} | "
            f"${s['total_pnl']:+.2f} | ${s['avg_pnl']:+.2f} | "
            f"${s['max_drawdown_dollar']:.2f} | {pf} | {s['moves_caught']} |")


def format_report(result: dict) -> str:
    p = result["preset"]
    lev = result["leverage"]
    lev_label = leverage_label(lev) if not (isinstance(lev, float) and math.isinf(lev)) else "1:Unlimited"
    if isinstance(lev, float) and math.isinf(lev):
        margin_txt = "~$0.00 (unlimited)"
    else:
        margin_txt = f"~${4000.0 / lev * result['lot_oz']:.2f} at $4000 gold"
    lot_label = f"{result['lot_oz']:.2f} oz ({result['lot_oz'] / 100.0:.2f} lot)"
    lines = [
        "# GoldenSwing - Exness $12 Timing Backtest (NOT OANDA)",
        "",
        f"**Engine:** {result.get('engine')} | **Variant:** {result.get('variant', 'base')} | **GODCE filter:** {'ON' if result.get('godce_filter') else 'OFF'} | **Preset:** {p['bias_tf']} bias / {p['zone_tf']} zones / {p['trigger_tf']} trigger (swing_n={p['swing_n']}). Trigger bars: {result['n_trigger_bars']:,}, evaluated: {result['evaluated_bars']:,}.",
        "",
        f"**Balance:** ${result['start_balance']:.2f} -> ${result['final_balance']:.2f} (x{result['growth_multiple']:.2f}) | **Peak:** ${result['peak_balance']:.2f} | **MaxDD:** ${result['max_drawdown_dollar']:.2f} | **Blow-ups:** {result['blowups']} | **Cannot open:** {result['cannot_open']} | **Hard veto:** {'ON' if result.get('veto_enabled', True) else 'OFF'} — blocked {result.get('veto_blocked', 0)} setups",
        "",
        f"**Size:** {lot_label} fixed | **Leverage:** {lev_label} | **Margin/trade:** {margin_txt} | **Spread:** ${result['spread_per_oz']:.2f}/oz (total ${result['spread_per_oz'] * result['lot_oz']:.2f}/trade) | **Commission:** ${result['commission_per_trade']:.2f} | **Slippage:** ${result['slippage_per_oz']:.2f} (assumed 0) | **Margin call 60% / Stop-out 0% (Exness Standard)**",
        "",
        "| Slice | Trades | Win% | Total $ | Avg $ | MaxDD $ | PF | Moves caught |",
        "|---|---|---|---|---|---|---|---|",
        _row_dollar("ALL", result["summary"]),
        "",
        "## By direction",
        "",
        "| Slice | Trades | Win% | Total $ | Avg $ | MaxDD $ | PF | Moves caught |",
        "|---|---|---|---|---|---|---|---|",
        _row_dollar("Long", result["by_direction"]["long"]),
        _row_dollar("Short", result["by_direction"]["short"]),
        "",
        "## Blocked trades (what the hard veto removed — counterfactual,",
        "## same entry/exit logic, evaluated independently per setup;",
        "## blocked trades may overlap each other, unlike real trades;",
        "## blocked PnL uses the balance at block time, never applied)",
        "",
        "| Slice | Trades | Win% | Total $ | Avg $ | MaxDD $ | PF | Moves caught |",
        "|---|---|---|---|---|---|---|---|",
        _row_dollar("Blocked (would-have-taken)", result.get("blocked_summary", _summarize_dollar([]))),
        _row_dollar("Blocked long", result.get("blocked_by_direction", {}).get("long", _summarize_dollar([]))),
        _row_dollar("Blocked short", result.get("blocked_by_direction", {}).get("short", _summarize_dollar([]))),
        "",
        f"Veto-blocked setups by direction: {result.get('veto_blocked_by_direction', {})} (setups blocked, incl. ones that later failed sweep/displacement). Blocked-trade blow-ups (would-have-blown): {result.get('blocked_blowups', 0)}.",
        "",
        "## By killzone (UTC)",
        "",
        "| Slice | Trades | Win% | Total $ | Avg $ | MaxDD $ | PF | Moves caught |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in ("Asia", "London KZ", "NY KZ", "Off-KZ"):
        lines.append(_row_dollar(key, result["by_killzone"][key]))
    lines += [
        "",
        "## By session (UTC, rough)",
        "",
        "| Slice | Trades | Win% | Total $ | Avg $ | MaxDD $ | PF | Moves caught |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in ("Asia", "London", "Overlap", "NY"):
        lines.append(_row_dollar(key, result["by_session"][key]))
    lines += [
        "",
        "## Rejection reasons",
        "",
    ]
    for reason, cnt in sorted(result["rejects"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {reason}: {cnt}")
    lines += [
        "",
        "## Honesty notes",
        "",
        "- PnL in $ = move * lot_oz - spread_total - commission. No R-multiples. Leverage changes margin, not PnL per lot.",
        "- Stop-out 0% = equity <= $0 closes. Margin call 60% is warning only. Unlimited -> margin ~$0, no margin call, only equity stop-out.",
        "- Leverage range 1:2 to unlimited. At 1:2, 0.01 lot margin ~$2000, so $12 CANNOT open (shown as impossible, not viable).",
        "- LIMITATION: Exness can restrict/reduce leverage around news/weekends/HMR. Not modeled; constant leverage per run. Real live margin may be higher in those windows.",
        "- Spread 0.2 interpreted as $0.20/oz. Commission $0. Slippage $0 assumed (optimistic).",
        "- POI-mid LIMIT fill assumes intra-bar path OHLC cannot prove; real fills may be at edge or missed.",
        "- Data is OANDA spot XAU as proxy for Exness spot; basis immaterial for timing logic.",
        "- AMD is simplified (sweep -> displacement).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GoldenSwing Exness $12 timing backtest")
    ap.add_argument("--csv", required=True, help="base OHLC CSV (ts,open,high,low,close,volume)")
    ap.add_argument("--preset", default="swing", choices=sorted(PRESETS))
    ap.add_argument("--balance", type=float, default=START_BALANCE_12)
    ap.add_argument("--leverage", default="2000", help="e.g. 2000, 1:2000, 2, unlimited")
    ap.add_argument("--lot-oz", type=float, default=LOT_OZ, help="1.0 = 0.01 lot, 2.0 = 0.02 lot, 5.0 = 0.05 lot")
    ap.add_argument("--spread", type=float, default=SPREAD_PER_OZ)
    ap.add_argument("--slippage", type=float, default=0.0)
    ap.add_argument("--reset-on-blowup", action="store_true", help="reset to $12 after blow-up to estimate blow-up rate")
    ap.add_argument("--no-veto", action="store_true", help="disable the hard veto (before/after comparison)")
    ap.add_argument("--variant", default="base", choices=list(REVIVAL_VARIANTS), help="revived strategy variant (default base)")
    ap.add_argument("--godce-filter", action="store_true", help="enable GODCE regime filter (filter only, never an entry)")
    for tf in ("d1", "h4", "m15", "m5", "m1"):
        ap.add_argument(f"--{tf}", default=None)
    args = ap.parse_args(argv)
    leverage = parse_leverage(args.leverage)
    preset = get_preset(args.preset)
    base = load_ohlc_csv(args.csv)
    frames = {infer_tf(base): base}
    for tf, path in (("D1", args.d1), ("H4", args.h4), ("M15", args.m15), ("M5", args.m5), ("M1", args.m1)):
        if path:
            frames[tf] = load_ohlc_csv(path)
    try:
        frames = build_tf_frames(frames, preset)
        result = run_exness_backtest(frames, preset, start_balance=args.balance, leverage=leverage, lot_oz=args.lot_oz, spread_per_oz=args.spread, slippage_per_oz=args.slippage, reset_on_blowup=args.reset_on_blowup, enable_veto=not args.no_veto, variant=args.variant, godce_filter=args.godce_filter)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print(format_report(result))
    print(f"Frames used: " + ", ".join(f"{tf}={len(frames[tf]):,}" for tf in (preset.bias_tf, preset.zone_tf, preset.trigger_tf)))
    # if single-run blew up, also estimate blow-up rate with resets
    if result["blow_up"] and not args.reset_on_blowup:
        reset_res = run_exness_backtest(frames, preset, start_balance=args.balance, leverage=leverage, lot_oz=args.lot_oz, spread_per_oz=args.spread, slippage_per_oz=args.slippage, reset_on_blowup=True, enable_veto=not args.no_veto)
        total = len(reset_res["trades"]) + reset_res["blowups"]
        rate = 100.0 * reset_res["blowups"] / total if total else 0.0
        print(f"\nBlow-up-rate (reset mode): {reset_res['blowups']} blow-ups in {total} attempts = {rate:.1f}% blow-up rate, final balance ${reset_res['final_balance']:.2f} after resets.")


if __name__ == "__main__":
    main()
