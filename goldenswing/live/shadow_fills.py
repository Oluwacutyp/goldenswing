"""Shadow fill tracker: empirical limit-order fill measurement.

Problem: backtests assume fills at the touch price. Live queues don't work
that way. This module runs SHADOW orders alongside the paper loop — for every
signal it tracks two hypothetical limits (edge vs offset) and logs what
actually happened: fill rate, time-to-fill, realized price.

No capital, no orders placed. Pure measurement. After N weeks the log
answers: does the offset buy real fill certainty, and at what price cost?

Usage (any loop frequency — H4, D1, not just hourly):
    tracker = ShadowFillTracker(state_path)
    tracker.on_signal(ts, direction, zone_edge, atr, offsets=(0.0, 0.05))
    ...
    tracker.on_bar(ts, high, low)   # each new closed bar
    # fills / expiries append to JSONL automatically
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class ShadowOrder:
    signal_ts: str
    direction: int          # +1 buy limit, -1 sell limit
    zone_edge: float
    offset_atr: float
    limit_price: float
    atr: float
    status: str = "OPEN"    # OPEN | FILLED | EXPIRED
    fill_ts: str | None = None
    fill_price: float | None = None
    bars_to_fill: int = 0

    def to_dict(self) -> dict:
        return {
            "signal_ts": self.signal_ts, "direction": self.direction,
            "zone_edge": self.zone_edge, "offset_atr": self.offset_atr,
            "limit_price": self.limit_price, "atr": self.atr,
            "status": self.status, "fill_ts": self.fill_ts,
            "fill_price": self.fill_price, "bars_to_fill": self.bars_to_fill,
        }


class ShadowFillTracker:
    """Tracks hypothetical limit fills bar-by-bar. JSONL log + JSON state."""

    def __init__(self, state_dir: str, max_open_bars: int = 12):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.log_path = os.path.join(state_dir, "shadow_fills.jsonl")
        self.state_path = os.path.join(state_dir, "shadow_state.json")
        self.max_open_bars = max_open_bars
        self.open_orders: list[ShadowOrder] = []
        self._load()

    def _load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                d = json.load(f)
            self.open_orders = [ShadowOrder(**o) for o in d.get("open", [])]

    def _save(self):
        with open(self.state_path, "w") as f:
            json.dump({"open": [o.to_dict() for o in self.open_orders]}, f)

    def _log(self, order: ShadowOrder):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(order.to_dict()) + "\n")

    def on_signal(self, signal_ts: str, direction: int, zone_edge: float,
                  atr: float, offsets: tuple = (0.0, 0.05)) -> None:
        """Register shadow limits at edge and at each offset. No real orders."""
        for off in offsets:
            limit = zone_edge - off * atr * direction
            self.open_orders.append(ShadowOrder(
                signal_ts=signal_ts, direction=direction,
                zone_edge=zone_edge, offset_atr=off,
                limit_price=limit, atr=atr,
            ))
        self._save()

    def on_bar(self, bar_ts: str, high: float, low: float) -> list[ShadowOrder]:
        """Feed one closed bar. Returns newly filled/expired orders."""
        done = []
        for o in self.open_orders:
            o.bars_to_fill += 1
            filled = (low <= o.limit_price) if o.direction == 1 \
                else (high >= o.limit_price)
            if filled:
                o.status = "FILLED"
                o.fill_ts = bar_ts
                o.fill_price = o.limit_price  # limit = price guarantee
                done.append(o)
                self._log(o)
            elif o.bars_to_fill >= self.max_open_bars:
                o.status = "EXPIRED"
                done.append(o)
                self._log(o)
        self.open_orders = [o for o in self.open_orders if o.status == "OPEN"]
        if done:
            self._save()
        return done

    def report(self) -> dict:
        """Fill-rate comparison: edge vs offset. Reads the JSONL log."""
        from collections import defaultdict
        stats = defaultdict(lambda: {"n": 0, "filled": 0, "avg_bars": []})
        if not os.path.exists(self.log_path):
            return {}
        with open(self.log_path) as f:
            for line in f:
                d = json.loads(line)
                key = f"offset_{d['offset_atr']:.2f}"
                stats[key]["n"] += 1
                if d["status"] == "FILLED":
                    stats[key]["filled"] += 1
                    stats[key]["avg_bars"].append(d["bars_to_fill"])
        out = {}
        for k, s in sorted(stats.items()):
            out[k] = {
                "signals": s["n"],
                "fill_rate": s["filled"] / s["n"] if s["n"] else 0,
                "avg_bars_to_fill": (
                    sum(s["avg_bars"]) / len(s["avg_bars"])
                    if s["avg_bars"] else None),
            }
        return out
