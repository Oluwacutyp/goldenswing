"""Cost-inclusive outcome scoring for Hot Scalp replay output.

Reads candidate_shadow events (from scripts/replay_hot_scalp.py
output JSONL) plus a ticks CSV (time,bid,ask) and scores each
candidate on forward bid/ask touches:

- long: win when bid first reaches the target, loss when bid
  first reaches the structural stop;
- short: win when ask first reaches the target, loss when ask
  first reaches the structural stop.

R is measured from the SPREAD-ADJUSTED entry (long: mid +
spread/2, i.e. the ask actually paid; short: mid - spread/2), so
the score is net of the cost the candidate itself reported — not
a mid-price fantasy.  A candidate whose spread-adjusted entry
already sits beyond its stop is scored "invalid_entry" and kept
out of expectancy, not silently counted as a loss.

Expectancy is reported over RESOLVED candidates only (wins +
losses); open candidates (neither level touched within the
horizon) are counted separately.  This is measurement, never a
promotion: nothing here submits orders.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def score_candidates(candidates, ticks: pd.DataFrame,
                     horizon_minutes: float = 120.0) -> dict:
    """Score candidate dicts against forward ticks.

    ``candidates``: dicts with trigger_ts, direction,
    entry_reference, structural_stop, target_reference, spread
    (optional, default 0), strategy (optional), expiry_ts
    (optional).  ``ticks``: DataFrame with a DatetimeIndex and
    bid/ask columns, sorted ascending.
    """
    ticks = ticks.sort_index()
    per_strategy: dict[str, dict] = {}
    scored: list[dict] = []

    def _bump(bucket: dict, outcome: str, r: float | None) -> None:
        bucket["candidates"] += 1
        if outcome == "win":
            bucket["wins"] += 1
            bucket["resolved"] += 1
            bucket["r_sum"] += float(r)
        elif outcome == "loss":
            bucket["losses"] += 1
            bucket["resolved"] += 1
            bucket["r_sum"] += float(r)
        elif outcome == "open":
            bucket["open"] += 1
        else:
            bucket["invalid_entry"] += 1

    for cand in candidates:
        strategy = str(cand.get("strategy") or "unknown")
        bucket = per_strategy.setdefault(
            strategy, {"candidates": 0, "wins": 0, "losses": 0,
                       "open": 0, "invalid_entry": 0, "resolved": 0,
                       "r_sum": 0.0})
        direction = 1 if int(cand["direction"]) > 0 else -1
        entry = float(cand["entry_reference"])
        stop = float(cand["structural_stop"])
        target = float(cand["target_reference"])
        spread = float(cand.get("spread") or 0.0)
        eff_entry = entry + direction * spread / 2.0
        trigger = pd.Timestamp(cand["trigger_ts"])
        horizon_end = trigger + pd.Timedelta(minutes=horizon_minutes)
        expiry = cand.get("expiry_ts")
        if expiry:
            try:
                horizon_end = min(horizon_end, pd.Timestamp(expiry))
            except (TypeError, ValueError):
                pass
        # Cost-inclusive validity: the adjusted entry must still
        # sit on the right side of its own stop.
        if (direction > 0 and eff_entry <= stop) or \
                (direction < 0 and eff_entry >= stop):
            _bump(bucket, "invalid_entry", None)
            scored.append({**cand, "outcome": "invalid_entry",
                           "r": None})
            continue
        risk = abs(eff_entry - stop)
        reward = abs(target - eff_entry)
        fwd = ticks[(ticks.index > trigger) & (ticks.index <= horizon_end)]
        outcome, r = "open", None
        for _, row in fwd.iterrows():
            bid, ask = float(row["bid"]), float(row["ask"])
            if direction > 0:
                if bid >= target:
                    outcome, r = "win", reward / risk
                    break
                if bid <= stop:
                    outcome, r = "loss", -1.0
                    break
            else:
                if ask <= target:
                    outcome, r = "win", reward / risk
                    break
                if ask >= stop:
                    outcome, r = "loss", -1.0
                    break
        _bump(bucket, outcome, r)
        scored.append({**cand, "outcome": outcome, "r": r})

    def _finish(bucket: dict) -> dict:
        resolved = bucket["resolved"]
        return {
            "candidates": bucket["candidates"],
            "wins": bucket["wins"], "losses": bucket["losses"],
            "open": bucket["open"],
            "invalid_entry": bucket["invalid_entry"],
            "resolved": resolved,
            "win_rate": (bucket["wins"] / resolved
                         if resolved else None),
            "expectancy_r": (bucket["r_sum"] / resolved
                             if resolved else None),
        }

    totals = {"candidates": 0, "wins": 0, "losses": 0, "open": 0,
              "invalid_entry": 0, "resolved": 0, "r_sum": 0.0}
    for bucket in per_strategy.values():
        for key in totals:
            totals[key] += bucket[key]
    return {
        "totals": _finish(totals),
        "per_strategy": {name: _finish(bucket)
                         for name, bucket in sorted(per_strategy.items())},
        "scored": scored,
        "method": "bid/ask first-touch vs structural stop/target; "
                  "R from spread-adjusted entry; expectancy over "
                  "resolved candidates only",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score Hot Scalp replay candidates cost-inclusively.")
    parser.add_argument("replay_jsonl", type=Path)
    parser.add_argument("ticks_csv", type=Path)
    parser.add_argument("--horizon-minutes", type=float, default=120.0)
    args = parser.parse_args(argv)
    candidates = []
    for line in args.replay_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        event = row.get("event", row)
        if isinstance(event, dict) and event.get("event") == "candidate_shadow":
            candidates.append(event)
    ticks = pd.read_csv(args.ticks_csv, parse_dates=["time"])
    ticks = ticks.set_index("time").sort_index()
    result = score_candidates(candidates, ticks,
                              horizon_minutes=args.horizon_minutes)
    result.pop("scored")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
