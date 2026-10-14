"""Real-trader backtest: BIAS -> LOCATION -> TRIGGER -> ATR RISK -> SIZING.

Replaces the textbook mtf12 harness logic (next-bar-open chase, fixed
2R floor, fixed 72-bar time stop, 4-condition displacement gate,
amd_phase="unknown", rough sessions).

Real-trader rules, all causal (closed bars only):

- HTF veto, never democracy. Bias != 0 required.
- Sweep of liquidity (Asia high/low, PDH/PDL, range high/low) ->
  displacement -> retracement to FVG/OB -> LIMIT fill AT POI.
  Patient limit, never market-chase. Gap out of POI = SKIP.
  No retracement = NO TRADE.
- Displacement = strong body closing beyond swing + FVG created.
  Confirmation, not a 4-condition textbook gate that kills all trades.
- AMD: Accumulation = range, Manipulation = sweep, Distribution =
  expansion after sweep. Only Manipulation -> Distribution trades.
- Killzones: London 07-10 UTC, NY 12-15 UTC, Asia 00-06 UTC as range.
  No expansion trades in deep Asia unless clear manipulation +
  displacement (which AMD already requires).
- Stop = structural invalidation (sweep extreme + buffer) BUT with
  max-stop gate: stop > 2.5x ATR14 -> REJECT, do not take absurd stops.
- Target = opposing liquidity pool (PDH/PDL, Asia high/low, zone-TF
  swing) with >= 1.5R. If nearest pool gives < 1.5R, SKIP.
- Exit = stop (wins ties), target, opposing structural break,
  emergency time stop only (not primary).
- Risk: $0.12/trade (1% of $12), position = risk / stop_distance.
  Daily loss limit: stop after -2R/day. No compounding.
  Margin: margin_50x reported separately, infeasible if > balance.

$12 honesty: same two ledgers as mtf12 (fractional R + 1-unit
feasible). Margin is NOT folded into feasibility.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from ..live.multitf import (
    PAPER_START_BALANCE_12, PRESETS, TimeframeConfig, atr14,
    check_hard_veto, evaluate_bias, find_swings, get_preset,
)
from .mtf12 import (
    TF_MINUTES, build_tf_frames, infer_tf, load_ohlc_csv, session_of,
)

# patience window for limit revisit, in trigger bars
PATIENCE_BARS = 20
# emergency time stop, by trigger TF (emergency only, not primary)
EMERGENCY_BARS = {"M1": 300, "M5": 200, "M15": 120, "H1": 100, "H4": 60, "D1": 30}
MAX_STOP_ATR = 2.5
MIN_TARGET_R = 1.5
DAILY_LOSS_LIMIT_R = -2.0
WINDOW = 300


# ------------------------------------------------------------------ sessions
def killzone_of(ts: pd.Timestamp) -> str:
    """Real killzone label from UTC hour."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    h = ts.hour
    if 0 <= h < 6:
        return "Asia"
    if 7 <= h < 10:
        return "London KZ"
    if 12 <= h < 15:
        return "NY KZ"
    return "Off-KZ"


# ------------------------------------------------------------------ pools
def daily_pools(trigger: pd.DataFrame) -> dict:
    """Per-date PDH/PDL + Asia (00-06 UTC) high/low, from trigger bars.

    PDH/PDL for date D = previous calendar day's high/low.
    Asia high/low for date D = bars of D with hour in [0,6).
    All computed from closed bars only; lookup by date is causal
    because the backtest only queries dates <= trigger bar date,
    and PDH/PDL use the *previous* day.
    """
    out: dict = {}
    dates = sorted(set(trigger.index.date))
    prev_high = prev_low = None
    for d in dates:
        day = trigger[trigger.index.date == d]
        asia = day[(day.index.hour >= 0) & (day.index.hour < 6)]
        out[d] = {
            "pdh": prev_high,
            "pdl": prev_low,
            "asia_high": float(asia["high"].max()) if len(asia) else None,
            "asia_low": float(asia["low"].min()) if len(asia) else None,
        }
        if len(day):
            prev_high = float(day["high"].max())
            prev_low = float(day["low"].min())
    return out


def find_fvgs(bars: pd.DataFrame) -> list[dict]:
    """FVGs as {direction, top, bottom, index}.

    Bull FVG at i: low[i] > high[i-2], gap = [high[i-2], low[i]].
    Bear FVG at i: high[i] < low[i-2], gap = [high[i], low[i-2]].
    """
    out = []
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    for i in range(2, len(bars)):
        if lows[i] > highs[i - 2]:
            out.append({"direction": 1, "top": float(lows[i]),
                        "bottom": float(highs[i - 2]), "index": i})
        elif highs[i] < lows[i - 2]:
            out.append({"direction": -1, "top": float(lows[i - 2]),
                        "bottom": float(highs[i]), "index": i})
    return out


def detect_sweep(trig_win: pd.DataFrame, pools: list[float],
                 direction: int, atr: float) -> dict | None:
    """Sweep of liquidity in the last 20 trigger bars.

    For a LONG (direction +1): sweep of LOWS = low dips below a pool
    then close recovers above it (bear trap).
    For a SHORT (direction -1): sweep of HIGHS = high spikes above a
    pool then close falls back below it.
    Returns {pool, extreme, bar_pos} of the most recent sweep, or None.
    """
    if atr <= 0:
        return None
    buf = 0.05 * atr
    n = len(trig_win)
    start = max(0, n - 20)
    highs = trig_win["high"].to_numpy(dtype=float)
    lows = trig_win["low"].to_numpy(dtype=float)
    closes = trig_win["close"].to_numpy(dtype=float)
    best = None
    for k in range(start, n):
        for pool in pools:
            if pool is None:
                continue
            if direction == 1:
                if lows[k] < pool - buf and closes[k] > pool:
                    best = {"pool": float(pool), "extreme": float(lows[k]),
                            "bar_pos": int(k)}
            else:
                if highs[k] > pool + buf and closes[k] < pool:
                    best = {"pool": float(pool), "extreme": float(highs[k]),
                            "bar_pos": int(k)}
    return best


def detect_displacement(trig_win: pd.DataFrame, atr: float,
                        direction: int) -> dict | None:
    """Displacement: strong body closing beyond recent swing + FVG.

    Real-trader confirmation, not the 4-condition textbook gate.
    Requires: close beyond the most recent swing on the displacement
    side, body >= 0.5x ATR OR body/range >= 0.5, and an FVG in the
    same direction in the last 3 bars.
    """
    if atr <= 0 or len(trig_win) < 5:
        return None
    last = trig_win.iloc[-1]
    body = abs(float(last["close"]) - float(last["open"]))
    rng = float(last["high"]) - float(last["low"])
    if rng <= 0:
        return None
    if not (body >= 0.5 * atr or body / rng >= 0.5):
        return None
    # recent swing on the displacement side (from bars[:-1])
    highs, lows = find_swings(
        trig_win["high"].to_numpy()[:-1], trig_win["low"].to_numpy()[:-1], 3)
    if direction == 1:
        if not highs:
            return None
        ref_high = highs[-1][1]
        if not (float(last["close"]) > ref_high + 0.05 * atr):
            return None
    else:
        if not lows:
            return None
        ref_low = lows[-1][1]
        if not (float(last["close"]) < ref_low - 0.05 * atr):
            return None
    fvgs = [f for f in find_fvgs(trig_win.tail(4)) if f["direction"] == direction]
    if not fvgs:
        return None
    fvg = fvgs[-1]
    return {"fvg_top": fvg["top"], "fvg_bottom": fvg["bottom"],
            "body": float(body), "range": float(rng)}


def _summarize(trades: list[dict], risk_money: float) -> dict:
    if not trades:
        return {"total_trades": 0, "wins": 0, "win_rate": 0.0,
                "total_R": 0.0, "avg_R": 0.0, "max_drawdown_R": 0.0,
                "profit_factor": 0.0, "dollar_pnl": 0.0}
    rs = [t["r_net"] for t in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    if losses and sum(losses) != 0:
        pf = sum(wins) / abs(sum(losses))
    elif wins:
        pf = float("inf")
    else:
        pf = 0.0
    peak, max_dd, cum = 0.0, 0.0, 0.0
    for r in rs:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return {"total_trades": len(trades), "wins": len(wins),
            "win_rate": 100.0 * len(wins) / len(trades),
            "total_R": float(sum(rs)), "avg_R": float(sum(rs) / len(rs)),
            "max_drawdown_R": float(max_dd), "profit_factor": float(pf),
            "dollar_pnl": float(sum(rs) * risk_money)}


def run_real_trader_backtest(df_by_tf: dict[str, pd.DataFrame],
                             preset: TimeframeConfig, *,
                             start_balance: float = PAPER_START_BALANCE_12,
                             cost_per_oz: float = 0.45,
                             risk_pct: float = 0.01,
                             slippage_per_oz: float = 0.0,
                             enable_veto: bool = True) -> dict:
    """Run real-trader logic over the trigger frame.

    ``enable_veto=False`` skips the hard veto (before/after runs).
    The result carries ``veto_blocked`` (setups killed by the hard
    veto) and ``veto_enabled``.  Veto TFs absent from ``df_by_tf``
    are fail-open (no veto), so callers that pass only
    bias/zone/trigger frames still run.
    """
    missing = [tf for tf in (preset.bias_tf, preset.zone_tf,
                             preset.trigger_tf) if tf not in df_by_tf]
    if missing:
        raise ValueError(f"missing frames for TFs {missing}; call "
                         f"build_tf_frames first")
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
    risk_money = float(start_balance) * float(risk_pct)
    pools_by_date = daily_pools(trigger)
    opens = trigger["open"].to_numpy(dtype=float)
    highs = trigger["high"].to_numpy(dtype=float)
    lows = trigger["low"].to_numpy(dtype=float)
    closes = trigger["close"].to_numpy(dtype=float)
    times = trigger.index

    trades: list[dict] = []
    blocked_trades: list[dict] = []
    rejects: Counter = Counter()
    veto_blocked_by_direction: Counter = Counter()
    daily_r: dict = {}
    evaluated = 0
    emergency = EMERGENCY_BARS.get(preset.trigger_tf, 120)
    i = 30
    while i <= n - 2:
        t = times[i]
        day = t.date()
        if daily_r.get(day, 0.0) <= DAILY_LOSS_LIMIT_R:
            # daily loss limit hit: skip to next day
            j = i
            while j < n and times[j].date() == day:
                j += 1
            i = max(j, i + 1)
            continue
        zpos = int(zone.index.searchsorted(t, side="right"))
        bpos = int(bias_df.index.searchsorted(t, side="right"))
        trig_win = trigger.iloc[max(0, i - WINDOW + 1):i + 1]
        zone_win = zone.iloc[max(0, zpos - WINDOW):zpos]
        bias_win = bias_df.iloc[max(0, bpos - WINDOW):bpos]
        evaluated += 1
        # Shadow flag: this iteration was veto-blocked, so the setup
        # below is evaluated as a COUNTERFACTUAL (what the blocked
        # trade would have done, same entry/exit logic) and lands in
        # blocked_trades, never in trades / daily_r.  The main loop
        # advances only i+1 on shadow iterations — a blocked trade
        # never occupies the account.
        shadow_blocked = False

        bias = evaluate_bias(bias_win, swing_n=preset.swing_n)
        if bias == 0:
            rejects["bias_flat"] += 1
            i += 1
            continue
        # HARD VETO: next-higher TF strongly disagreeing with the main
        # bias kills the setup outright.  Flat / agreeing /
        # weak-opposite veto TFs pass.  Missing veto frame = fail-open.
        # A vetoed setup FALLS THROUGH to the same pipeline below in
        # shadow mode, so blocked_trades measures what the veto
        # removed instead of guessing it.
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
        direction = bias  # HTF veto: trigger must equal bias
        atr_t = atr14(trig_win)
        if atr_t <= 0:
            rejects["no_atr"] += 1
            i += 1
            continue
        # liquidity pools for sweep: recent swings + Asia + PDH/PDL
        th, tl = find_swings(trig_win["high"].to_numpy(),
                             trig_win["low"].to_numpy(), preset.swing_n)
        pool_vals: list[float] = []
        if direction == 1:
            pool_vals += [p for _, p in tl[-3:]]
        else:
            pool_vals += [p for _, p in th[-3:]]
        dp = pools_by_date.get(day, {})
        # previous-day pools + today's Asia (already closed portion is
        # causal: Asia bars up to t are closed; today's Asia high/low
        # so far is known at t)
        if dp.get("pdl") is not None and direction == 1:
            pool_vals.append(dp["pdl"])
        if dp.get("pdh") is not None and direction == -1:
            pool_vals.append(dp["pdh"])
        if dp.get("asia_low") is not None and direction == 1:
            pool_vals.append(dp["asia_low"])
        if dp.get("asia_high") is not None and direction == -1:
            pool_vals.append(dp["asia_high"])

        sweep = detect_sweep(trig_win, pool_vals, direction, atr_t)
        if sweep is None:
            rejects["no_sweep"] += 1
            i += 1
            continue
        disp = detect_displacement(trig_win, atr_t, direction)
        if disp is None:
            rejects["no_displacement"] += 1
            i += 1
            continue
        # sweep must precede displacement (sweep bar before last bar)
        if sweep["bar_pos"] >= len(trig_win) - 1:
            # sweep on the trigger bar itself is not manipulation-then-
            # distribution; it is the same bar. Require separation.
            rejects["sweep_displacement_order"] += 1
            i += 1
            continue
        amd_phase = "manipulation_to_distribution"

        # POI = FVG gap from displacement
        poi_top, poi_bottom = disp["fvg_top"], disp["fvg_bottom"]
        if poi_top <= poi_bottom:
            rejects["bad_poi"] += 1
            i += 1
            continue
        poi_mid = 0.5 * (poi_top + poi_bottom)
        kz = killzone_of(t)
        # deep Asia filter: already requires sweep+displacement via AMD,
        # so Asia is allowed only when both are present (they are).
        # No extra block; killzone is reported.

        # ---- patient LIMIT: wait for revisit to POI ----
        filled_idx = None
        gap_skip = False
        last_pat = min(i + PATIENCE_BARS, n - 1)
        for j in range(i + 1, last_pat + 1):
            # gap out of POI on the far side: price jumped through,
            # do not assume a fill, do not chase.
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
        entry = float(poi_mid)
        # stop = sweep extreme + buffer (structural invalidation)
        buf = 0.1 * atr_t
        if direction == 1:
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
        if stop_dist > MAX_STOP_ATR * atr_t:
            rejects["stop_too_wide"] += 1
            i = i + 1 if shadow_blocked else filled_idx + 1
            continue
        # ---- target: opposing liquidity pool >= 1.5R ----
        # candidates: PDH/PDL, Asia high/low, zone-TF swings
        cands: list[float] = []
        if direction == 1:
            if dp.get("pdh") is not None:
                cands.append(dp["pdh"])
            if dp.get("asia_high") is not None:
                cands.append(dp["asia_high"])
            zh, _ = find_swings(zone_win["high"].to_numpy(),
                                zone_win["low"].to_numpy(), preset.swing_n)
            cands += [p for _, p in zh if p > entry]
        else:
            if dp.get("pdl") is not None:
                cands.append(dp["pdl"])
            if dp.get("asia_low") is not None:
                cands.append(dp["asia_low"])
            _, zl = find_swings(zone_win["high"].to_numpy(),
                                zone_win["low"].to_numpy(), preset.swing_n)
            cands += [p for _, p in zl if p < entry]
        target = None
        target_pool = None
        for pool in sorted(cands, key=lambda p: abs(p - entry)):
            r_pool = direction * (pool - entry) / stop_dist
            if r_pool >= MIN_TARGET_R:
                target, target_pool = float(pool), float(pool)
                break
        if target is None:
            rejects["target_lt_1p5R"] += 1
            i = i + 1 if shadow_blocked else filled_idx + 1
            continue
        target_r = direction * (target - entry) / stop_dist

        # ---- exit scan: stop wins ties, then target, then opposing
        # structural break, then emergency time stop ----
        last = min(filled_idx + emergency, n - 1)
        exit_price, exit_idx, reason = float(closes[last]), last, "timeout"
        for k in range(filled_idx, last + 1):
            if direction == 1:
                if lows[k] <= stop:
                    exit_price = stop if opens[k] > stop else opens[k]
                    exit_idx, reason = k, "stop"
                    break
                if highs[k] >= target:
                    exit_price = target if opens[k] < target else opens[k]
                    exit_idx, reason = k, "target"
                    break
                # opposing structural break: close below the most
                # recent higher low (CHoCH against the position)
                if k > filled_idx + 2:
                    w = trigger.iloc[max(0, k - 50):k + 1]
                    _, wl = find_swings(w["high"].to_numpy(),
                                        w["low"].to_numpy(), 3)
                    if len(wl) >= 2 and closes[k] < wl[-1][1] - buf:
                        exit_price = float(closes[k])
                        exit_idx, reason = k, "structure_break"
                        break
            else:
                if highs[k] >= stop:
                    exit_price = stop if opens[k] < stop else opens[k]
                    exit_idx, reason = k, "stop"
                    break
                if lows[k] <= target:
                    exit_price = target if opens[k] > target else opens[k]
                    exit_idx, reason = k, "target"
                    break
                if k > filled_idx + 2:
                    w = trigger.iloc[max(0, k - 50):k + 1]
                    wh, _ = find_swings(w["high"].to_numpy(),
                                        w["low"].to_numpy(), 3)
                    if len(wh) >= 2 and closes[k] > wh[-1][1] + buf:
                        exit_price = float(closes[k])
                        exit_idx, reason = k, "structure_break"
                        break
        r_gross = direction * (exit_price - entry) / stop_dist
        r_net = r_gross - (cost_per_oz + slippage_per_oz) / stop_dist
        feasible = stop_dist <= risk_money
        trade = {
            "signal_index": int(i), "entry_index": int(filled_idx),
            "exit_index": int(exit_idx), "exit_reason": reason,
            "direction": int(direction), "entry": float(entry),
            "stop": float(stop), "target": float(target),
            "target_pool": target_pool, "target_R": float(target_r),
            "exit_price": float(exit_price),
            "r_gross": float(r_gross), "r_net": float(r_net),
            "fractional_units": float(risk_money / stop_dist),
            "oanda_1unit_feasible": bool(feasible),
            "margin_50x": float(entry / 50.0),
            "session": session_of(times[i]),
            "killzone": kz,
            "regime": "trend" if bias != 0 else "range",
            "htf_state": int(bias),
            "amd_phase": amd_phase,
            "hold_bars": int(exit_idx - filled_idx),
            "zone_level": float(poi_mid),
            "trigger": f"sweep_disp_{'bull' if direction == 1 else 'bear'}",
            "poi_top": float(poi_top), "poi_bottom": float(poi_bottom),
            "sweep_pool": float(sweep["pool"]),
            "sweep_extreme": float(sweep["extreme"]),
            "stop_atr": float(stop_dist / atr_t) if atr_t else 0.0,
            "entry_type": "limit_poi",
            "veto_blocked": bool(shadow_blocked),
        }
        if shadow_blocked:
            # Counterfactual only: never touches daily_r / trades.
            blocked_trades.append(trade)
            i = i + 1
            continue
        fill_day = times[filled_idx].date()
        daily_r[fill_day] = daily_r.get(fill_day, 0.0) + r_net
        trades.append(trade)
        i = exit_idx

    feasible = [t for t in trades if t["oanda_1unit_feasible"]]
    by_session = {}
    for key in ("Asia", "London", "Overlap", "NY"):
        by_session[key] = _summarize(
            [t for t in trades if t["session"] == key], risk_money)
    by_killzone = {}
    for key in ("Asia", "London KZ", "NY KZ", "Off-KZ"):
        by_killzone[key] = _summarize(
            [t for t in trades if t["killzone"] == key], risk_money)
    by_regime = {key: _summarize(
        [t for t in trades if t["regime"] == key], risk_money)
        for key in ("trend", "range")}
    by_direction = {
        "long": _summarize([t for t in trades if t["direction"] == 1],
                           risk_money),
        "short": _summarize([t for t in trades if t["direction"] == -1],
                            risk_money),
    }
    by_amd = {key: _summarize(
        [t for t in trades if t["amd_phase"] == key], risk_money)
        for key in ("manipulation_to_distribution",)}
    blocked_by_direction = {
        "long": _summarize(
            [t for t in blocked_trades if t["direction"] == 1], risk_money),
        "short": _summarize(
            [t for t in blocked_trades if t["direction"] == -1], risk_money),
    }
    return {
        "preset": {"bias_tf": preset.bias_tf, "zone_tf": preset.zone_tf,
                   "trigger_tf": preset.trigger_tf,
                   "swing_n": preset.swing_n},
        "engine": "real_trader_v1",
        "start_balance": float(start_balance),
        "risk_pct": float(risk_pct),
        "risk_money": risk_money,
        "cost_per_oz": float(cost_per_oz),
        "slippage_per_oz": float(slippage_per_oz),
        "n_trigger_bars": n,
        "evaluated_bars": evaluated,
        "rejects": dict(rejects),
        "veto_blocked": int(rejects.get("hard_veto", 0)),
        "veto_enabled": bool(enable_veto),
        "veto_blocked_by_direction": dict(veto_blocked_by_direction),
        "blocked_trades": blocked_trades,
        "blocked_summary": _summarize(blocked_trades, risk_money),
        "blocked_by_direction": blocked_by_direction,
        "trades": trades,
        "summary": _summarize(trades, risk_money),
        "feasible_1unit": _summarize(feasible, risk_money),
        "by_session": by_session,
        "by_killzone": by_killzone,
        "by_regime": by_regime,
        "by_direction": by_direction,
        "by_amd": by_amd,
    }


def _row(label: str, s: dict) -> str:
    pf = "inf" if math.isinf(s["profit_factor"]) else f"{s['profit_factor']:.2f}"
    return (f"| {label} | {s['total_trades']} | {s['win_rate']:.1f} | "
            f"{s['total_R']:+.2f} | {s['avg_R']:+.3f} | "
            f"{s['max_drawdown_R']:.2f} | {pf} | "
            f"${s['dollar_pnl']:+.2f} |")


def format_report(result: dict) -> str:
    p = result["preset"]
    lines = [
        "# GoldenSwing — Real-Trader $12 Backtest",
        "",
        f"**Engine:** {result.get('engine')} | **Preset:** {p['bias_tf']} bias / "
        f"{p['zone_tf']} zones / {p['trigger_tf']} trigger "
        f"(swing_n={p['swing_n']}). Trigger bars: {result['n_trigger_bars']:,}, "
        f"evaluated: {result['evaluated_bars']:,}.",
        "",
        f"**Balance:** ${result['start_balance']:.2f} | "
        f"**Risk:** {result['risk_pct']:.1%} = "
        f"${result['risk_money']:.2f}/trade (fixed, no compounding) | "
        f"**Cost:** ${result['cost_per_oz']:.2f}/oz + "
        f"${result['slippage_per_oz']:.2f} slippage | "
        f"**Daily loss limit:** -2R/day | "
        f"**Max stop:** {MAX_STOP_ATR}x ATR | "
        f"**Min target:** {MIN_TARGET_R}R | "
        f"**Entries:** LIMIT at POI only | "
        f"**Hard veto:** {'ON' if result.get('veto_enabled', True) else 'OFF'}"
        f" — blocked {result.get('veto_blocked', 0)} setups.",
        "",
        "| Slice | Trades | Win% | Total R | Avg R | MaxDD R | PF | $ P&L |",
        "|---|---|---|---|---|---|---|---|",
        _row("ALL (fractional-unit R)", result["summary"]),
        _row("OANDA 1-unit feasible", result["feasible_1unit"]),
        "",
        "## By direction (shorts are NOT hidden)",
        "",
        "| Slice | Trades | Win% | Total R | Avg R | MaxDD R | PF | $ P&L |",
        "|---|---|---|---|---|---|---|---|",
        _row("Long", result["by_direction"]["long"]),
        _row("Short", result["by_direction"]["short"]),
        "",
        "## Blocked trades (what the hard veto removed — counterfactual,",
        "## same entry/exit logic, evaluated independently per setup;",
        "## blocked trades may overlap each other, unlike real trades)",
        "",
        "| Slice | Trades | Win% | Total R | Avg R | MaxDD R | PF | $ P&L |",
        "|---|---|---|---|---|---|---|---|",
        _row("Blocked (would-have-taken)",
             result.get("blocked_summary", _summarize([], 0.12))),
        _row("Blocked long",
             result.get("blocked_by_direction", {}).get(
                 "long", _summarize([], 0.12))),
        _row("Blocked short",
             result.get("blocked_by_direction", {}).get(
                 "short", _summarize([], 0.12))),
        "",
        f"Veto-blocked setups by direction: "
        f"{result.get('veto_blocked_by_direction', {})} "
        f"(setups blocked, incl. ones that later failed sweep/displacement).",
        "",
        "## By killzone (UTC)",
        "",
        "| Slice | Trades | Win% | Total R | Avg R | MaxDD R | PF | $ P&L |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in ("Asia", "London KZ", "NY KZ", "Off-KZ"):
        lines.append(_row(key, result["by_killzone"][key]))
    lines += [
        "",
        "## By session (UTC, rough)",
        "",
        "| Slice | Trades | Win% | Total R | Avg R | MaxDD R | PF | $ P&L |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in ("Asia", "London", "Overlap", "NY"):
        lines.append(_row(key, result["by_session"][key]))
    lines += [
        "",
        "## By regime (HTF bias)",
        "",
        "| Slice | Trades | Win% | Total R | Avg R | MaxDD R | PF | $ P&L |",
        "|---|---|---|---|---|---|---|---|",
        _row("Trend", result["by_regime"]["trend"]),
        _row("Range", result["by_regime"]["range"]),
        "",
        "## By AMD phase",
        "",
        "| Slice | Trades | Win% | Total R | Avg R | MaxDD R | PF | $ P&L |",
        "|---|---|---|---|---|---|---|---|",
        _row("Manipulation->Distribution",
             result["by_amd"]["manipulation_to_distribution"]),
        "",
        "## Rejection reasons (why setups did NOT become trades)",
        "",
    ]
    for reason, cnt in sorted(result["rejects"].items(),
                              key=lambda kv: -kv[1]):
        lines.append(f"- {reason}: {cnt}")
    lines += [
        "",
        "## Honesty notes",
        "",
        "- Fractional-unit R measures STRATEGY quality, not OANDA fills.",
        "- 1-unit feasible = stop distance <= $0.12 risk budget. Margin "
        "is separate: at 50:1, 1 oz at ~$4000 needs ~$80 margin.",
        "- Entries are LIMIT at POI mid. Gap out of POI = skip, no chase.",
        "- AMD is implemented (sweep -> displacement), not 'unknown'.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="GoldenSwing real-trader $12 backtest")
    ap.add_argument("--csv", required=True,
                    help="base OHLC CSV (ts,open,high,low,close,volume)")
    ap.add_argument("--preset", default="swing",
                    choices=sorted(PRESETS))
    ap.add_argument("--balance", type=float,
                    default=PAPER_START_BALANCE_12)
    ap.add_argument("--risk-pct", type=float, default=0.01)
    ap.add_argument("--cost-per-oz", type=float, default=0.45)
    ap.add_argument("--slippage-per-oz", type=float, default=0.0)
    ap.add_argument("--no-veto", action="store_true",
                    help="disable the hard veto (before/after comparison)")
    for tf in ("d1", "h4", "m15", "m5", "m1"):
        ap.add_argument(f"--{tf}", default=None)
    args = ap.parse_args(argv)
    preset = get_preset(args.preset)
    base = load_ohlc_csv(args.csv)
    frames = {infer_tf(base): base}
    for tf, path in (("D1", args.d1), ("H4", args.h4), ("M15", args.m15),
                     ("M5", args.m5), ("M1", args.m1)):
        if path:
            frames[tf] = load_ohlc_csv(path)
    try:
        frames = build_tf_frames(frames, preset)
        result = run_real_trader_backtest(
            frames, preset, start_balance=args.balance,
            cost_per_oz=args.cost_per_oz, risk_pct=args.risk_pct,
            slippage_per_oz=args.slippage_per_oz,
            enable_veto=not args.no_veto)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print(format_report(result))
    print(f"Frames used: "
          + ", ".join(f"{tf}={len(frames[tf]):,}"
                      for tf in (preset.bias_tf, preset.zone_tf,
                                 preset.trigger_tf)))


if __name__ == "__main__":
    main()
