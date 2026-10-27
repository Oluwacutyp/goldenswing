"""Measure the shared evidence builder on historical frames (step 5).

Replays the CURRENT `_build_shared_signal_evidence` over sampled
OANDA M5 bars (proxy feed, NOT the MT5 broker feed) with M15
context, for BOTH directions, and labels each evaluation from
forward M5 bars: first touch of +1 ATR vs -1 ATR within the next
48 M5 bars (4h).  Reports per-category support win rates plus
overall direction win rates, no-touch rates, and sample sizes.

Causality rules (no lookahead):
- M5/M15 windows contain only bars CLOSED at or before the
  evaluation time (bar open + timeframe <= now).
- The forward window starts strictly after the sampled bar.
- A forward bar touching both levels first is "ambiguous" and
  excluded from win rates, never silently counted as a win.

Assumptions, stated plainly in the output:
- OANDA is a proxy feed, not broker data.
- Spread is a fixed assumption (default 0.30; mid +/- half).
- No M1 history exists in the local files, so the M1 EMA leg and
  the spread-vs-M1-ATR leg report unavailable on every sample.
- Weights stay research-only and non-gating whatever this
  reports: one in-sample pass is measurement, not calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import pandas as pd

from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.live.hierarchy import TimeframeStack


class _NoOrderExec:
    """Executor that must never be called by a measurement run."""

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return []

    def pending_orders(self, symbol=None):
        return []

    def cancel_pending(self, ticket):
        raise AssertionError("measurement must not place orders")

    def place_order(self, request):
        raise AssertionError("measurement must not place orders")


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    df = df.set_index("ts").sort_index()
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    return df


def _atr14(frame: pd.DataFrame) -> float:
    if len(frame) < 15:
        return 0.0
    w = frame.tail(15)
    prev_close = w["close"].shift(1)
    tr = pd.concat([
        w["high"] - w["low"],
        (w["high"] - prev_close).abs(),
        (w["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.iloc[1:].mean())


def _closed_asof(frame: pd.DataFrame, minutes: int,
                 now: pd.Timestamp) -> pd.DataFrame:
    """Bars whose close time is at or before ``now``."""
    return frame[frame.index + pd.Timedelta(minutes=minutes) <= now]


def measure(m5: pd.DataFrame, m15: pd.DataFrame, *, window: int,
            step: int, horizon_bars: int, spread: float) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="gs-measure-evidence-"))
    cfg = MT5RunnerConfig(
        journal_path=tmp / "mt5_journal.jsonl",
        risk_state_path=tmp / "mt5_risk.json",
        kill_switch_path=tmp / "mt5_KILL",
        news_pause_enabled=False)
    runner = MT5Runner(cfg, feed=None, executor=_NoOrderExec())
    runner._hot_scalp_aux_frames = {}

    tail = m5.tail(window)
    samples = list(range(100, len(tail), step))
    rows: list[dict] = []
    per_category: dict[str, dict] = {}
    for i in samples:
        bar = tail.iloc[i]
        now = tail.index[i] + pd.Timedelta(minutes=5)
        m5w = _closed_asof(m5, 5, now).tail(600)
        m15w = _closed_asof(m15, 15, now).tail(600)
        if len(m5w) < 60 or len(m15w) < 35:
            continue
        atr = _atr14(m5w)
        if not math.isfinite(atr) or atr <= 0:
            continue
        entry = float(bar["close"])
        runner._stack = TimeframeStack.from_frames(
            {"M5": m5w, "M15": m15w})
        runner._last_tick = {
            "time": now, "mid": entry,
            "bid": entry - spread / 2.0, "ask": entry + spread / 2.0}
        # NOTE: i indexes the TAIL window, not the full frame — the
        # forward window must be offset by the tail's start, or
        # outcomes are read from the wrong months entirely.
        full_pos = len(m5) - len(tail) + i
        fwd = m5.iloc[full_pos + 1: full_pos + 1 + horizon_bars]
        up, down = entry + atr, entry - atr
        for direction in (1, -1):
            snap = runner._build_shared_signal_evidence(
                direction=direction, playbook="measurement",
                entry=entry, bar_ts=now, context={})
            if snap.get("status") != "measured":
                rows.append({"ts": now.isoformat(), "direction": direction,
                             "status": snap.get("status"),
                             "reason": snap.get("reason_code")})
                continue
            outcome, r = "open", None
            for _, fb in fwd.iterrows():
                hi, lo = float(fb["high"]), float(fb["low"])
                hit_up, hit_down = hi >= up, lo <= down
                if hit_up and hit_down:
                    outcome = "ambiguous"
                    break
                if direction > 0:
                    if hit_up:
                        outcome, r = "win", 1.0
                        break
                    if hit_down:
                        outcome, r = "loss", -1.0
                        break
                else:
                    if hit_down:
                        outcome, r = "win", 1.0
                        break
                    if hit_up:
                        outcome, r = "loss", -1.0
                        break
            counted_keys = {c.get("key") for c in
                            snap.get("evidence_score", {}).get("counted", [])}
            stances = {}
            for item in snap.get("evidence", []):
                if item.get("key") in counted_keys and item.get("available"):
                    stances.setdefault(item["category"], item["stance"])
            for category, stance in stances.items():
                bucket = per_category.setdefault(
                    category, {"support": {"n": 0, "wins": 0, "resolved": 0},
                               "contradict": {"n": 0, "wins": 0,
                                              "resolved": 0}})
                side = "support" if stance == "supports" else \
                    "contradict" if stance == "contradicts" else None
                if side is None:
                    continue
                bucket[side]["n"] += 1
                if outcome in ("win", "loss"):
                    bucket[side]["resolved"] += 1
                    if outcome == "win":
                        bucket[side]["wins"] += 1
            rows.append({"ts": now.isoformat(), "direction": direction,
                         "status": "measured", "score": snap["score"],
                         "outcome": outcome, "r": r,
                         "evidence_categories": stances})

    def _rate(bucket: dict) -> dict:
        resolved = bucket["resolved"]
        return {"n": bucket["n"], "resolved": resolved,
                "win_rate": (bucket["wins"] / resolved
                             if resolved else None)}

    per_direction: dict[int, dict] = {}
    for direction in (1, -1):
        sub = [r for r in rows if r.get("direction") == direction
               and r.get("status") == "measured"]
        resolved = [r for r in sub if r["outcome"] in ("win", "loss")]
        wins = sum(1 for r in resolved if r["outcome"] == "win")
        per_direction[direction] = {
            "measured": len(sub),
            "wins": wins, "losses": len(resolved) - wins,
            "open": sum(1 for r in sub if r["outcome"] == "open"),
            "ambiguous": sum(1 for r in sub
                             if r["outcome"] == "ambiguous"),
            "win_rate_resolved": (wins / len(resolved)
                                  if resolved else None),
        }
    return {
        "samples": len(rows),
        "per_direction": {str(k): v for k, v in per_direction.items()},
        "per_category": {
            name: {"support": _rate(b["support"]),
                   "contradict": _rate(b["contradict"])}
            for name, b in sorted(per_category.items())},
        "rows": rows,
        "assumptions": {
            "feed": "OANDA proxy, not MT5 broker data",
            "spread": spread,
            "horizon_bars": horizon_bars,
            "m1": "unavailable (no local M1 history)",
            "weights": "research-only, non-gating",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure the shared evidence builder on history.")
    parser.add_argument("--m5", type=Path, required=True)
    parser.add_argument("--m15", type=Path, required=True)
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--step", type=int, default=12)
    parser.add_argument("--horizon-bars", type=int, default=48)
    parser.add_argument("--spread", type=float, default=0.30)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = measure(_load(args.m5), _load(args.m15),
                     window=args.window, step=args.step,
                     horizon_bars=args.horizon_bars, spread=args.spread)
    summary = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                       default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
