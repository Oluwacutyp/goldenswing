"""Real multi-timeframe backtest — ALL timeframes, ONE hierarchy.

Runs the SAME :func:`goldenswing.live.hierarchy.evaluate_hierarchy`
the live loop uses (shared code — no lookalike second implementation)
over historical bars, one position at a time.  The hierarchy is the
D1 -> H4 -> H1 -> session-entry-clock chain; a bar that fails any leg
never trades.  Reports per-leg stats (how often each gate blocks)
and per-timeframe contribution (by entry clock / session / QM leg).

METHOD (causal, closed bars only)
---------------------------------
- Frames: H1 (base), H4 + D1 resampled from H1 (honest aggregation
  of closed bars), M15 + M5 from their CSVs, M30 resampled from M15.
  All frames are sliced to their intersection window.
- Per H1 bar ``i``: every TF is sliced to bars that had CLOSED by the
  H1 bar's close (see ``hierarchy._causal_window``); the verdict at
  ``i`` uses exactly the bars the live loop would have held.
- AMD phase comes from the real ``features/phases.py`` machine run
  over H1 (tag only, never a veto).  If the phase machine fails,
  ``amd_phase`` is ``"unknown"`` — never faked.
- QM touches come from the real ``QMM5Playbook`` batch scan over M5
  (annotation only — QM never gates, never fires standalone).
- Entry: next H1 bar's open, only if still inside the H4 POI band
  (a gap out of the POI skips the trade instead of chasing it).
  Risk is recomputed from the actual fill; target is the hierarchy's
  pool target with the 2R floor re-measured from the fill.
- Exit: stop wins same-bar ties, then target, then a 72-H1-bar time
  stop.  Cost in R exactly as the harness does it.
- Hard veto is OFF (``evaluate_hierarchy`` never enables it).

$12 HONESTY: same two ledgers as ``mtf12.py`` — fractional-unit R
accounting (strategy quality, NOT a claim OANDA would fill it) and
the ``oanda_1unit_feasible`` flag per trade.  Dollar P&L is
``total_R * risk_money`` — tiny by design, reported as-is.

CLI:
    python -m goldenswing.backtest.mtf_hierarchy [--h1 CSV] [--m15 CSV]
        [--m5 CSV] [--balance 12] [--risk-pct 0.01] [--max-bars N]
        [--swing-n 3]
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from ..live.hierarchy import (STACK_TFS, TF_MINUTES, TimeframeStack,
                              evaluate_hierarchy)
from .mtf12 import _summarize, load_ohlc_csv

#: Resample rules incl. M30 (``mtf12.resample_bars`` has no M30 rule —
#: this module needs it for the session entry clock, so the rule lives
#: here rather than changing mtf12's shared helper).
_RESAMPLE_RULE: dict[str, str] = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h", "D1": "1D", "W1": "1W",
}


def _resample_bars(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate a finer frame into coarser ``tf`` bars (same
    convention as ``mtf12.resample_bars``: input stamped at open,
    output stamped at period CLOSE, so ``index <= trigger_time``
    selects exactly the closed bars)."""
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

TIME_STOP_BARS = 72          # H1 bars, harness parity
WINDOW = 300                 # rolling window per TF passed to evaluate
COST_PER_OZ = 0.45
RISK_PCT = 0.01

#: Stamp conventions for the production frames: resampled frames
#: (H4/D1 from H1, M30 from M15) are CLOSE-stamped; CSV-native frames
#: (H1/M15/M5) are OPEN-stamped.  See hierarchy.evaluate_hierarchy.
PRODUCTION_STAMPS: dict[str, str] = {
    "H4": "close", "D1": "close", "M30": "close",
}


# ------------------------------------------------------------------ frames
def build_hierarchy_frames(h1: pd.DataFrame, m15: pd.DataFrame,
                           m5: pd.DataFrame | None = None
                           ) -> dict[str, pd.DataFrame]:
    """Full stack from the OANDA CSVs: H4/D1 resampled from H1, M30
    resampled from M15, M5 from its file when supplied.  Everything
    is sliced to the intersection window (M5 starts 2026-04-01, so a
    full-stack run covers 2026-04 -> 2026-10).

    Stamp conventions: CSV-native frames (H1/M15/M5) are OPEN-stamped;
    resampled frames (H4/D1/M30) are CLOSE-stamped — see
    ``PRODUCTION_STAMPS`` and ``hierarchy.evaluate_hierarchy``.
    """
    frames: dict[str, pd.DataFrame] = {"H1": h1.sort_index()}
    frames["H4"] = _resample_bars(h1, "H4")
    frames["D1"] = _resample_bars(h1, "D1")
    frames["M15"] = m15.sort_index()
    frames["M30"] = _resample_bars(m15, "M30")
    if m5 is not None:
        frames["M5"] = m5.sort_index()
    # Intersection window across all supplied frames.
    start = max(f.index[0] for f in frames.values())
    end = min(f.index[-1] for f in frames.values())
    return {tf: f.loc[(f.index >= start) & (f.index <= end)]
            for tf, f in frames.items()}


def _closed_slice(frame: pd.DataFrame, tf: str,
                  h1_close: pd.Timestamp, stamp: str = "open"
                  ) -> pd.DataFrame:
    """Bars of ``frame`` closed by ``h1_close``.

    ``stamp="open"`` (CSV-native): open + TF <= close.
    ``stamp="close"`` (resampled): stamp <= close.
    ``searchsorted`` keeps this O(log n) per bar — a boolean mask
    over the full frame per H1 bar would be quadratic.
    """
    if stamp == "close":
        cutoff = h1_close
    else:
        cutoff = h1_close - pd.Timedelta(minutes=TF_MINUTES[tf])
    pos = int(frame.index.searchsorted(cutoff, side="right"))
    return frame.iloc[:pos]


def _amd_phases(h1: pd.DataFrame) -> list[str]:
    """AMD phase per H1 bar from the real phases machine (tag only).

    Returns ``["unknown", ...]`` on any failure — a missing tag is
    more honest than a faked phase.
    """
    try:
        from ..features.config import FeatureConfig
        from ..features.phases import compute as phases_compute
        from ..reader.reader import MarketReader
        states = MarketReader().read(h1)
        out = phases_compute(h1, states, FeatureConfig())
        return [str(p.phase) for p in out]
    except Exception:
        return ["unknown"] * len(h1)


def _qm_touches(m5: pd.DataFrame | None) -> list[dict]:
    """QM touches on M5 from the real playbook batch scan (annotation).

    Returns ``[{touch_ts, direction, qm_level, head, neckline}]``
    sorted by time.  Empty on any failure — QM is annotation, so a
    failed scan degrades to ``qm_absent``, never to a crash.
    """
    if m5 is None or len(m5) < 30:
        return []
    try:
        from ..playbooks.base import EngineContext
        from ..playbooks.qm_m5 import QMM5Playbook
        ctx = EngineContext(df=m5, states=[], years=0.0,
                            instrument="XAUUSD")
        touches = []
        for cand in QMM5Playbook().candidates(ctx):
            lc = dict(getattr(cand.outcome.signal, "live_context",
                              None) or {})
            touches.append({
                "touch_ts": m5.index[cand.signal_index],
                "direction": int(cand.direction),
                "qm_level": lc.get("qm_level"),
                "head": lc.get("head"),
                "neckline": lc.get("neckline"),
            })
        return sorted(touches, key=lambda t: t["touch_ts"])
    except Exception:
        return []


def _latest_qm_touch(touches: list[dict], asof: pd.Timestamp
                     ) -> dict | None:
    """Latest QM touch at/before ``asof`` (causal: future touches are
    invisible), or None."""
    best = None
    for touch in touches:
        if touch["touch_ts"] <= asof:
            best = touch
        else:
            break
    return best


# -------------------------------------------------------------------- core
def run_hierarchy_backtest(frames: dict[str, pd.DataFrame], *,
                           swing_n: int = 3,
                           start_balance: float = 12.0,
                           cost_per_oz: float = COST_PER_OZ,
                           risk_pct: float = RISK_PCT,
                           time_stop_bars: int = TIME_STOP_BARS,
                           window: int = WINDOW,
                           max_bars: int | None = None,
                           stamps: dict | None = None) -> dict:
    """Run the shared hierarchy over the H1 frame.

    ``frames`` needs H1 + every other stack TF (use
    :func:`build_hierarchy_frames`).  ``stamps`` maps TF ->
    "open" | "close" (default: all "open" — the convention of
    synthetic / OANDA-native frames; production CSV runs pass
    ``PRODUCTION_STAMPS``).  Returns the trade book plus per-leg gate
    stats, per-timeframe contribution, and the per-bar verdict log
    (``verdicts``) the live/backtest parity test reads.
    """
    missing = [tf for tf in STACK_TFS if tf not in frames]
    if missing:
        raise ValueError(f"missing frames for TFs {missing}; call "
                         f"build_hierarchy_frames first")
    h1 = frames["H1"].sort_index()
    if max_bars is not None:
        h1 = h1.iloc[:max_bars]
    n = len(h1)
    if n < 3:
        raise ValueError("H1 frame too short")
    # Sort once: the per-bar loop must never re-sort full frames.
    sframes = {tf: frames[tf].sort_index() for tf in STACK_TFS}

    risk_money = float(start_balance) * float(risk_pct)
    opens = h1["open"].to_numpy(dtype=float)
    highs = h1["high"].to_numpy(dtype=float)
    lows = h1["low"].to_numpy(dtype=float)
    closes = h1["close"].to_numpy(dtype=float)
    times = h1.index

    amd = _amd_phases(h1)
    qm_touches = _qm_touches(frames.get("M5"))
    stamps = stamps or {}

    trades: list[dict] = []
    verdicts: list[dict] = []
    gate_counts: Counter = Counter()
    skipped_gap = 0
    skipped_bad_stop = 0
    evaluated = 0
    i = 60  # warmup: ATR / swings / phase machine need history
    while i <= n - 2:  # need bar i+1 for the next-open entry
        asof = times[i]
        h1_close = asof + pd.Timedelta(minutes=60)
        stack_frames = {
            tf: _closed_slice(sframes[tf], tf, h1_close,
                              stamps.get(tf, "open")).tail(window)
            for tf in STACK_TFS
        }
        stack = TimeframeStack.from_frames(stack_frames)
        verdict = evaluate_hierarchy(
            stack, asof=asof, amd_phase=amd[i],
            qm_touch=_latest_qm_touch(qm_touches, asof),
            swing_n=swing_n, window=window, stamps=stamps)
        evaluated += 1
        verdicts.append({
            "i": int(i), "ts": asof.isoformat(),
            "allowed": bool(verdict["allowed"]),
            "blocked_leg": verdict["blocked_leg"],
            "direction": int(verdict["direction"]),
            "entry_clock": verdict.get("entry_clock"),
            "session": (verdict.get("session") or {}).get("session"),
            "killzone": (verdict.get("session") or {}).get("killzone"),
            "amd_phase": verdict.get("amd_phase"),
            "qm_status": verdict.get("qm", {}).get("status"),
            "trigger": verdict.get("trigger"),
        })
        if not verdict["allowed"]:
            gate_counts[verdict["blocked_leg"]] += 1
            i += 1
            continue
        gate_counts["passed"] += 1
        # Enter at the NEXT bar's open, only if still inside the POI.
        fill = float(opens[i + 1])
        band = verdict["zone"]
        if not (band["bottom"] <= fill <= band["top"]):
            skipped_gap += 1
            i += 1
            continue
        stop = float(verdict["stop"])
        d = int(verdict["direction"])
        risk = (fill - stop) * d
        if risk <= 0:
            skipped_bad_stop += 1
            i += 1
            continue
        pool = float(verdict["target"])
        target = pool if (pool - fill) * d >= 2.0 * risk \
            else fill + d * 2.0 * risk

        last = min(i + time_stop_bars, n - 1)
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
        r_net = r_gross - cost_per_oz / risk
        session = verdict.get("session") or {}
        trades.append({
            "signal_index": int(i), "entry_index": int(i + 1),
            "exit_index": int(exit_idx), "exit_reason": reason,
            "direction": d, "entry": fill, "stop": stop,
            "target": float(target), "exit_price": float(exit_price),
            "r_gross": float(r_gross), "r_net": float(r_net),
            "fractional_units": float(risk_money / risk),
            "oanda_1unit_feasible": bool(
                risk <= risk_money) if risk > 0 else False,
            "margin_50x": float(fill / 50.0),
            "session": session.get("session"),
            "killzone": session.get("killzone"),
            "entry_clock": verdict.get("entry_clock"),
            "amd_phase": verdict.get("amd_phase"),
            "qm_status": verdict.get("qm", {}).get("status"),
            "hold_bars": int(exit_idx - i),
            "zone_level": float(band["level"]),
            "trigger": verdict.get("trigger"),
        })
        # Resume at the exit bar: one position at a time, never overlap.
        i = exit_idx

    feasible = [t for t in trades if t["oanda_1unit_feasible"]]
    by_entry_clock = {clock: _summarize(
        [t for t in trades if t["entry_clock"] == clock], risk_money)
        for clock in ("M15", "M30")}
    by_direction = {
        "long": _summarize([t for t in trades if t["direction"] == 1],
                           risk_money),
        "short": _summarize([t for t in trades if t["direction"] == -1],
                            risk_money),
    }
    qm_confirmed = [t for t in trades if t["qm_status"] == "qm_confirmed"]
    return {
        "hierarchy": "D1 bias -> H4 zones -> H1 trigger -> entry clock",
        "swing_n": swing_n, "start_balance": float(start_balance),
        "risk_pct": float(risk_pct), "risk_money": risk_money,
        "cost_per_oz": float(cost_per_oz),
        "n_h1_bars": n, "evaluated_bars": evaluated,
        "skipped_gap_out_of_poi": skipped_gap,
        "skipped_bad_stop": skipped_bad_stop,
        "veto_enabled": False,
        "trades": trades,
        "verdicts": verdicts,
        "gate_counts": dict(gate_counts),
        "gate_block_rate": {
            leg: round(100.0 * c / evaluated, 1) if evaluated else 0.0
            for leg, c in gate_counts.items()},
        "summary": _summarize(trades, risk_money),
        "feasible_1unit": _summarize(feasible, risk_money),
        "by_entry_clock": by_entry_clock,
        "by_direction": by_direction,
        "qm_confirmed": _summarize(qm_confirmed, risk_money),
        "qm_confirmed_trades": len(qm_confirmed),
    }


# ------------------------------------------------------------------ report
def _row(label: str, s: dict) -> str:
    pf = "inf" if math.isinf(s["profit_factor"]) else f"{s['profit_factor']:.2f}"
    return (f"| {label} | {s['total_trades']} | {s['win_rate']:.1f} | "
            f"{s['total_R']:+.2f} | {s['avg_R']:+.3f} | "
            f"{s['max_drawdown_R']:.2f} | {pf} | "
            f"${s['dollar_pnl']:+.2f} |")


def format_report(result: dict) -> str:
    lines = [
        "# GoldenSwing — All-Timeframe Hierarchy Backtest",
        "",
        f"**Hierarchy:** {result['hierarchy']} (swing_n="
        f"{result['swing_n']}). Hard veto: OFF (veto_enabled="
        f"{result['veto_enabled']}).",
        "",
        f"**Bars:** {result['n_h1_bars']:,} H1, evaluated: "
        f"{result['evaluated_bars']:,}, skipped (gap out of POI): "
        f"{result['skipped_gap_out_of_poi']}, skipped (bad stop): "
        f"{result['skipped_bad_stop']}.",
        "",
        f"**Balance:** ${result['start_balance']:.2f} | "
        f"**Risk:** {result['risk_pct']:.1%} = "
        f"${result['risk_money']:.2f}/trade (fixed, no compounding) | "
        f"**Cost:** ${result['cost_per_oz']:.2f}/oz.",
        "",
        "| Slice | Trades | Win% | Total R | Avg R | MaxDD R | PF | $ P&L |",
        "|---|---|---|---|---|---|---|---|",
        _row("ALL (fractional-unit R)", result["summary"]),
        _row("OANDA 1-unit feasible", result["feasible_1unit"]),
        _row("QM-confirmed subset", result["qm_confirmed"]),
        "",
        "## Per-leg gate stats — how often each gate blocks",
        "",
        "| Leg | Blocked bars | % of evaluated |",
        "|---|---|---|",
    ]
    for leg, count in sorted(result["gate_counts"].items(),
                             key=lambda kv: -kv[1]):
        pct = result["gate_block_rate"].get(leg, 0.0)
        lines.append(f"| {leg} | {count} | {pct:.1f}% |")
    lines += [
        "",
        "``passed`` counts bars where every leg passed (a trade then "
        "needs a next-open fill inside the POI + a valid stop).",
        "",
        "## Per-timeframe contribution (entry clock)",
        "",
        "| Slice | Trades | Win% | Total R | Avg R | MaxDD R | PF | $ P&L |",
        "|---|---|---|---|---|---|---|---|",
        _row("M15 entry (killzone)", result["by_entry_clock"]["M15"]),
        _row("M30 entry (off-killzone)", result["by_entry_clock"]["M30"]),
        "",
        "## By direction",
        "",
        "| Slice | Trades | Win% | Total R | Avg R | MaxDD R | PF | $ P&L |",
        "|---|---|---|---|---|---|---|---|",
        _row("Long", result["by_direction"]["long"]),
        _row("Short", result["by_direction"]["short"]),
        "",
        "## Honesty notes",
        "",
        "- Fractional-unit R measures STRATEGY quality.  It is not a "
        "claim that OANDA would fill those sizes.",
        "- The hierarchy's win rate is UNPROVEN until this backtest + "
        "forward paper evidence exists: the numbers above are the "
        "first measurement, on one bullish window, not a validated edge.",
        "- ``amd_phase`` is a tag (never a veto); QM is annotation "
        "(never a gate).  Neither changes any trade in this book.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="GoldenSwing all-timeframe hierarchy backtest")
    ap.add_argument("--h1", default="data/OANDA_XAU_USD_H1_20241001_20261012.csv",
                    help="H1 CSV (ts,open,high,low,close,volume)")
    ap.add_argument("--m15", default="data/OANDA_XAU_USD_M15_20241001_20261012.csv",
                    help="M15 CSV")
    ap.add_argument("--m5", default="data/OANDA_XAU_USD_M5_20260401_20261012.csv",
                    help="M5 CSV (QM leg; omit with --no-m5)")
    ap.add_argument("--no-m5", action="store_true",
                    help="run without the M5 frame (QM leg = absent)")
    ap.add_argument("--balance", type=float, default=12.0)
    ap.add_argument("--risk-pct", type=float, default=RISK_PCT)
    ap.add_argument("--cost-per-oz", type=float, default=COST_PER_OZ)
    ap.add_argument("--swing-n", type=int, default=3)
    ap.add_argument("--max-bars", type=int, default=None,
                    help="limit H1 bars (quick runs)")
    args = ap.parse_args(argv)

    h1 = load_ohlc_csv(args.h1)
    m15 = load_ohlc_csv(args.m15)
    m5 = None if args.no_m5 else load_ohlc_csv(args.m5)
    frames = build_hierarchy_frames(h1, m15, m5)
    if args.no_m5:
        # QM leg absent: the backtest still needs an M5 frame key for
        # the stack check — an empty frame means "no M5 data".
        frames["M5"] = m15.iloc[0:0].copy()
    result = run_hierarchy_backtest(
        frames, swing_n=args.swing_n, start_balance=args.balance,
        cost_per_oz=args.cost_per_oz, risk_pct=args.risk_pct,
        max_bars=args.max_bars, stamps=PRODUCTION_STAMPS)
    print(format_report(result))
    print(f"Frames used: "
          + ", ".join(f"{tf}={len(frames[tf]):,}" for tf in STACK_TFS))


if __name__ == "__main__":
    main()
