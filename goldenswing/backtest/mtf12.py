"""Multi-timeframe $12 backtest for the structural engine.

Runs :func:`goldenswing.live.multitf.evaluate` per trigger bar on
rolling windows (causal, closed bars only), one position at a time.
Entry is at the NEXT bar's open (harness parity: the signal is known
at the trigger bar's close), but only if that open is still inside the
POI band — a gap out of the POI skips the trade instead of chasing it.
Risk is recomputed from the actual fill.  Exit: stop wins same-bar
ties, then target, then a 72 trigger-TF-bar time stop.  Cost is
charged in R exactly as the harness does it:
``R_net = R_gross - cost_per_oz / risk_per_oz``.

$12 HONESTY (user-explicit, do not paper over)
--------------------------------------------
``start_balance`` defaults to $12.00 (``PAPER_START_BALANCE_12``).
At the default 1% risk the risk budget is $0.12 per trade.

Every trade is reported in TWO ledgers, never mixed:

- fractional-unit R accounting — ``fractional_units`` is the size
  that would risk exactly the budget (``risk_money / stop_distance``).
  It is a MEASUREMENT device for strategy quality (R-multiples, win
  rate, drawdown).  It is NOT a claim that OANDA would fill it.
- ``oanda_1unit_feasible`` — True iff the risk of ONE ounce (the stop
  distance in dollars, since 1 unit = 1 oz) fits inside the risk
  budget.  Feasible trades are aggregated as a separate subset.
  Margin is reported separately as ``margin_50x`` (1 oz at 50:1) and
  is NOT folded into the flag: OANDA leverage varies by jurisdiction,
  so a single margin rule would be a false universal.  Note that at
  50:1, 1 oz of gold at ~$4000 needs ~$80 margin — a further,
  separate blocker on a $12 account.

Dollar P&L is ``total_R * risk_money`` — it will be tiny.  That is
expected and reported as-is, never inflated.

``amd_phase`` is always ``"unknown"``: AMD-phase classification is not
implemented, and a placeholder label is more honest than a faked one.

TIMEFRAME DATA
--------------
The backtest needs one frame per preset TF.  Coarser frames may be
DERIVED from a finer frame by aggregating closed bars
(:func:`resample_bars`) — that is honest aggregation, not invented
data.  A preset that needs a FINER frame than any supplied data is
refused with a clear error: finer bars cannot be reconstructed from
coarser ones.  E.g. with H1-only data, SWING (D1/H4/H1) runs via
resampling, while INTRADAY (needs M15) is refused.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from ..live.multitf import (PAPER_START_BALANCE_12, PRESETS,
                            TimeframeConfig, evaluate, get_preset)

# minutes per bar, for coarser/finer reasoning
TF_MINUTES: dict[str, float] = {
    "M1": 1.0, "M5": 5.0, "M15": 15.0, "H1": 60.0, "H4": 240.0,
    "D1": 1440.0, "W1": 10080.0,
}
_RESAMPLE_RULE: dict[str, str] = {
    "M1": "1min", "M5": "5min", "M15": "15min", "H1": "1h", "H4": "4h",
    "D1": "1D", "W1": "1W",
}

TIME_STOP_BARS = 72          # trigger-TF bars, harness parity
WINDOW = 300                 # rolling window per TF passed to evaluate


# ---------------------------------------------------------------- helpers
def session_of(ts: pd.Timestamp) -> str:
    """Rough UTC session label from the bar timestamp (naive stamps
    are treated as UTC).  Boundaries are approximate, not exchange
    definitions: Asia 00-08, London 08-13, Overlap 13-17, NY 17-22,
    Asia again 22-24."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    h = ts.hour
    if h < 8:
        return "Asia"
    if h < 13:
        return "London"
    if h < 17:
        return "Overlap"
    if h < 22:
        return "NY"
    return "Asia"


def regime_of(bias: int) -> str:
    """Trend/range from the HTF bias: biased = trending, flat = range."""
    return "trend" if bias != 0 else "range"


def oanda_1unit_feasible(stop_distance: float, risk_money: float) -> bool:
    """Can ONE OANDA unit (1 oz) take this trade inside the risk budget?

    1-unit risk in dollars IS the stop distance (1 oz x $/oz).  Feasible
    iff ``stop_distance <= risk_money``.  Risk-only by design (see the
    module docstring): margin is reported separately as ``margin_50x``.
    """
    if stop_distance <= 0 or risk_money <= 0:
        return False
    return stop_distance <= risk_money


def resample_bars(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate a finer frame into coarser ``tf`` bars.

    Input bars are stamped at their OPEN; the output bars are stamped
    at their CLOSE (period end), so ``index <= trigger_time`` selects
    exactly the bars that had closed by the trigger bar.  The final
    partial bucket (if any) carries a close stamp in the future and is
    therefore never selected by a causal slice — no lookahead.
    """
    if tf not in _RESAMPLE_RULE:
        raise ValueError(f"cannot resample to unknown TF {tf!r}")
    if len(df) < 2:
        return df.iloc[0:0].copy()
    diffs = df.index.to_series().diff().dropna()
    step = diffs.median()
    shifted = df.copy()
    shifted.index = df.index + step
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in shifted.columns:
        agg["volume"] = "sum"
    out = shifted.resample(_RESAMPLE_RULE[tf], closed="right",
                           label="right").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def build_tf_frames(frames: dict[str, pd.DataFrame],
                    preset: TimeframeConfig) -> dict[str, pd.DataFrame]:
    """Ensure a frame exists for every TF the preset needs.

    Coarser frames are resampled from the finest available frame;
    a preset needing a FINER frame than supplied raises ValueError.
    Veto TFs (``preset.veto_tf`` / ``preset.veto_tf2``) are
    best-effort: if no finer frame exists to resample from, the veto
    TF is left missing and the engine treats it as fail-open (no
    veto), never an error.
    """
    have = dict(frames)
    for tf in (preset.bias_tf, preset.zone_tf, preset.trigger_tf):
        if tf in have:
            continue
        finer = [t for t in have if TF_MINUTES[t] < TF_MINUTES[tf]]
        if not finer:
            raise ValueError(
                f"preset needs {tf} bars (or finer) but only "
                f"{sorted(have)} available — finer bars cannot be "
                f"reconstructed from coarser ones")
        src = min(finer, key=lambda t: TF_MINUTES[t])
        have[tf] = resample_bars(have[src], tf)
    for tf in (getattr(preset, "veto_tf", None),
               getattr(preset, "veto_tf2", None)):
        if not tf or tf in have:
            continue
        finer = [t for t in have
                 if t in TF_MINUTES and TF_MINUTES[t] < TF_MINUTES[tf]]
        if not finer:
            continue  # veto unavailable -> fail-open, not an error
        src = min(finer, key=lambda t: TF_MINUTES[t])
        have[tf] = resample_bars(have[src], tf)
    return have


# ------------------------------------------------------------------ core
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
    # max drawdown on the cumulative-R curve (peak-to-trough, in R)
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


def run_mtf12_backtest(df_by_tf: dict[str, pd.DataFrame],
                       preset: TimeframeConfig, *,
                       start_balance: float = PAPER_START_BALANCE_12,
                       cost_per_oz: float = 0.45,
                       risk_pct: float = 0.01,
                       slippage_per_oz: float = 0.0,
                       enable_veto: bool = True) -> dict:
    """Run the multi-TF engine over the trigger frame.

    ``df_by_tf`` must contain a frame for every TF the preset needs
    (use :func:`build_tf_frames` first).  Frames need a DatetimeIndex
    (UTC) — timestamp alignment is what keeps the HTF slices causal.
    Risk is a FIXED fraction of the starting balance per trade (no
    compounding): with a $12 balance the compounding difference is
    negligible and fixed risk keeps R accounting clean.
    ``enable_veto=False`` skips the hard veto (before/after runs).
    The result carries ``veto_blocked`` (signals killed by the hard
    veto) and ``veto_enabled``.
    """
    missing = [tf for tf in (preset.bias_tf, preset.zone_tf,
                             preset.trigger_tf) if tf not in df_by_tf]
    if missing:
        raise ValueError(f"missing frames for TFs {missing}; call "
                         f"build_tf_frames first")
    trigger = df_by_tf[preset.trigger_tf].sort_index()
    zone = df_by_tf[preset.zone_tf].sort_index()
    bias_df = df_by_tf[preset.bias_tf].sort_index()
    for name, frame in (("trigger", trigger), ("zone", zone),
                        ("bias", bias_df)):
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"{name} frame needs a DatetimeIndex "
                             f"(got {type(frame.index).__name__})")
    # veto frames, sorted once (not per bar)
    veto_dfs: dict[str, pd.DataFrame] = {}
    for vtf in (getattr(preset, "veto_tf", None),
                getattr(preset, "veto_tf2", None)):
        if vtf and vtf in df_by_tf:
            veto_dfs[vtf] = df_by_tf[vtf].sort_index()

    n = len(trigger)
    if n < 2:
        raise ValueError("trigger frame too short")

    risk_money = float(start_balance) * float(risk_pct)
    opens = trigger["open"].to_numpy(dtype=float)
    highs = trigger["high"].to_numpy(dtype=float)
    lows = trigger["low"].to_numpy(dtype=float)
    closes = trigger["close"].to_numpy(dtype=float)
    times = trigger.index

    trades: list[dict] = []
    skipped_gap = 0
    skipped_bad_stop = 0
    veto_blocked = 0
    evaluated = 0
    i = 30  # need a trigger window for ATR / avg-body / swings
    while i <= n - 2:  # need bar i+1 for the next-open entry
        t = times[i]
        zpos = int(zone.index.searchsorted(t, side="right"))
        bpos = int(bias_df.index.searchsorted(t, side="right"))
        trig_win = trigger.iloc[max(0, i - WINDOW + 1):i + 1]
        zone_win = zone.iloc[max(0, zpos - WINDOW):zpos]
        bias_win = bias_df.iloc[max(0, bpos - WINDOW):bpos]
        eval_frames = {preset.bias_tf: bias_win, preset.zone_tf: zone_win,
                       preset.trigger_tf: trig_win}
        for vtf, vdf in veto_dfs.items():
            vpos = int(vdf.index.searchsorted(t, side="right"))
            eval_frames[vtf] = vdf.iloc[max(0, vpos - WINDOW):vpos]
        res = evaluate(eval_frames, preset, enable_veto=enable_veto)
        evaluated += 1
        if not res["signal"]:
            if res.get("reason") == "hard_veto":
                veto_blocked += 1
            i += 1
            continue
        # enter at the NEXT bar's open, only if still inside the POI band
        fill = float(opens[i + 1])
        band = res["zone"]
        if not (band["bottom"] <= fill <= band["top"]):
            skipped_gap += 1
            i += 1
            continue
        stop = float(res["stop"])
        d = int(res["direction"])
        risk = (fill - stop) * d
        if risk <= 0:
            skipped_bad_stop += 1
            i += 1
            continue
        # target: evaluate()'s pool target, with the 2R floor
        # re-measured from the actual fill (harness rule).
        pool = float(res["target"])
        if (pool - fill) * d >= 2.0 * risk:
            target = pool
        else:
            target = fill + d * 2.0 * risk

        # exit scan from the entry bar; stop wins same-bar ties.
        last = min(i + TIME_STOP_BARS, n - 1)
        exit_price, exit_idx, reason = float(closes[last]), last, "timeout"
        for k in range(i + 1, last + 1):
            if d == 1:
                if lows[k] <= stop:
                    exit_price = stop if opens[k] > stop else opens[k]
                    exit_idx, reason = k, "stop"
                    break
                if highs[k] >= target:
                    exit_price = target if opens[k] < target else opens[k]
                    exit_idx, reason = k, "target"
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
        r_gross = d * (exit_price - fill) / risk
        r_net = r_gross - (cost_per_oz + slippage_per_oz) / risk
        trades.append({
            "signal_index": int(i), "entry_index": int(i + 1),
            "exit_index": int(exit_idx), "exit_reason": reason,
            "direction": d, "entry": fill, "stop": stop,
            "target": float(target), "exit_price": float(exit_price),
            "r_gross": float(r_gross), "r_net": float(r_net),
            "fractional_units": float(risk_money / risk),
            "oanda_1unit_feasible": bool(
                oanda_1unit_feasible(risk, risk_money)),
            "margin_50x": float(fill / 50.0),
            "session": session_of(times[i]),
            "regime": regime_of(int(res["bias"])),
            "htf_state": int(res["bias"]),
            "amd_phase": "unknown",
            "hold_bars": int(exit_idx - i),
            "zone_level": float(band["level"]),
            "trigger": res["trigger"],
        })
        # resume at the exit bar: the next signal enters at exit+1,
        # so positions can never overlap.
        i = exit_idx

    feasible = [t for t in trades if t["oanda_1unit_feasible"]]
    by_session = {}
    for key in ("Asia", "London", "Overlap", "NY"):
        by_session[key] = _summarize(
            [t for t in trades if t["session"] == key], risk_money)
    by_regime = {key: _summarize(
        [t for t in trades if t["regime"] == key], risk_money)
        for key in ("trend", "range")}
    by_direction = {
        "long": _summarize([t for t in trades if t["direction"] == 1],
                           risk_money),
        "short": _summarize([t for t in trades if t["direction"] == -1],
                            risk_money),
    }
    return {
        "preset": {"bias_tf": preset.bias_tf, "zone_tf": preset.zone_tf,
                   "trigger_tf": preset.trigger_tf,
                   "swing_n": preset.swing_n},
        "start_balance": float(start_balance),
        "risk_pct": float(risk_pct),
        "risk_money": risk_money,
        "cost_per_oz": float(cost_per_oz),
        "slippage_per_oz": float(slippage_per_oz),
        "n_trigger_bars": n,
        "evaluated_bars": evaluated,
        "skipped_gap_out_of_poi": skipped_gap,
        "skipped_bad_stop": skipped_bad_stop,
        "veto_blocked": veto_blocked,
        "veto_enabled": bool(enable_veto),
        "trades": trades,
        "summary": _summarize(trades, risk_money),
        "feasible_1unit": _summarize(feasible, risk_money),
        "by_session": by_session,
        "by_regime": by_regime,
        "by_direction": by_direction,
    }


# ------------------------------------------------------------------ report
def _row(label: str, s: dict) -> str:
    pf = "inf" if math.isinf(s["profit_factor"]) else f"{s['profit_factor']:.2f}"
    return (f"| {label} | {s['total_trades']} | {s['win_rate']:.1f} | "
            f"{s['total_R']:+.2f} | {s['avg_R']:+.3f} | "
            f"{s['max_drawdown_R']:.2f} | {pf} | "
            f"${s['dollar_pnl']:+.2f} |")


def format_report(result: dict) -> str:
    p = result["preset"]
    lines = [
        "# GoldenSwing — Multi-TF $12 Backtest",
        "",
        f"**Preset:** {p['bias_tf']} bias / {p['zone_tf']} zones / "
        f"{p['trigger_tf']} trigger (swing_n={p['swing_n']}). "
        f"Trigger bars: {result['n_trigger_bars']:,}, evaluated: "
        f"{result['evaluated_bars']:,}, skipped (gap out of POI): "
        f"{result['skipped_gap_out_of_poi']}, skipped (bad stop): "
        f"{result['skipped_bad_stop']}, hard-veto blocked: "
        f"{result.get('veto_blocked', 0)} "
        f"(veto {'ON' if result.get('veto_enabled', True) else 'OFF'}).",
        "",
        f"**Balance:** ${result['start_balance']:.2f} | "
        f"**Risk:** {result['risk_pct']:.1%} = "
        f"${result['risk_money']:.2f}/trade (fixed, no compounding) | "
        f"**Cost:** ${result['cost_per_oz']:.2f}/oz + "
        f"${result['slippage_per_oz']:.2f} slippage.",
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
        "## By session (UTC)",
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
        "## Honesty notes",
        "",
        "- Fractional-unit R measures STRATEGY quality.  It is not a "
        "claim that OANDA would fill those sizes.",
        "- The OANDA 1-unit feasible subset counts only trades whose "
        "1-oz risk fits the $0.12 budget.  Margin is NOT in that flag: "
        "at 50:1, 1 oz of gold at ~$4000 needs ~$80 margin — a further, "
        "separate blocker on a $12 account.",
        "- `amd_phase` is `unknown` throughout: AMD classification is "
        "not implemented, not faked.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------- CLI
def load_ohlc_csv(path: str | Path) -> pd.DataFrame:
    """Load an OANDA-style CSV (``ts,open,high,low,close,volume``) with
    a UTC DatetimeIndex."""
    df = pd.read_csv(path, parse_dates=["ts"])
    df = df.rename(columns={"ts": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    return df


def infer_tf(df: pd.DataFrame) -> str:
    """TF name from the median bar interval."""
    diffs = df.index.to_series().diff().dropna()
    minutes = diffs.median().total_seconds() / 60.0
    best = min(TF_MINUTES, key=lambda t: abs(TF_MINUTES[t] - minutes))
    if abs(TF_MINUTES[best] - minutes) > 1.0:
        raise ValueError(
            f"cannot infer TF: median bar is {minutes:.1f} minutes")
    return best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="GoldenSwing multi-TF $12 backtest")
    ap.add_argument("--csv", required=True,
                    help="base OHLC CSV (ts,open,high,low,close,volume)")
    ap.add_argument("--preset", default="swing",
                    choices=sorted(PRESETS),
                    help="preset (default: swing)")
    ap.add_argument("--balance", type=float,
                    default=PAPER_START_BALANCE_12)
    ap.add_argument("--risk-pct", type=float, default=0.01)
    ap.add_argument("--cost-per-oz", type=float, default=0.45)
    ap.add_argument("--slippage-per-oz", type=float, default=0.0)
    ap.add_argument("--no-veto", action="store_true",
                    help="disable the hard veto (before/after comparison)")
    for tf in ("d1", "h4", "m15", "m5", "m1"):
        ap.add_argument(f"--{tf}", default=None,
                        help=f"optional extra {tf.upper()} CSV")
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
        result = run_mtf12_backtest(
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
