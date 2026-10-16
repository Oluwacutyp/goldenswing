"""Deterministic pipeline: ingest -> merge -> bar aggregation -> transition.

One BarClosed = one atomic transition + one committed journal frame.
Effects are NEVER executed inside the transition: the transition
returns an ordered effect list, the caller journals first (persist
then act), then routes effects.  Replay runs the same transition
with effects discarded.

Merge: total order by (timestamp, source_id, source_seq).  Dedup by
(source_id, source_seq).  Watermark = max_seen_timestamp - skew;
events below the watermark are late-dropped and reported, never
stalling progress on a dead feed.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MergedEvent:
    timestamp: float
    source_id: str
    source_seq: int
    kind: str
    data: dict = field(default_factory=dict)


def merge_events(events: list[dict], *, skew_s: float = 1.0) -> dict:
    """Merge raw events into total order. Returns dict with keys:
    ordered (list[MergedEvent]), dropped_late (list), watermark (float).
    """
    seen: set[tuple[str, int]] = set()
    uniq: list[MergedEvent] = []
    for e in events:
        key = (str(e.get("source_id", "")), int(e.get("source_seq", 0)))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(MergedEvent(
            timestamp=float(e.get("timestamp", 0.0)),
            source_id=str(e.get("source_id", "")),
            source_seq=int(e.get("source_seq", 0)),
            kind=str(e.get("kind", e.get("event_type", ""))),
            data=dict(e.get("data") or e.get("payload") or {})))
    uniq.sort(key=lambda m: (m.timestamp, m.source_id, m.source_seq))
    max_seen = max((m.timestamp for m in uniq), default=0.0)
    watermark = max_seen - float(skew_s)
    ordered: list[MergedEvent] = []
    dropped_late: list[MergedEvent] = []
    for m in uniq:
        if m.timestamp < watermark and ordered:
            dropped_late.append(m)
        else:
            ordered.append(m)
    return {"ordered": ordered, "dropped_late": dropped_late,
            "watermark": watermark}


def aggregate_bars(ticks: list[dict], *, timeframe_s: float = 3600.0) -> list[dict]:
    """Aggregate ticks into OHLC bars keyed by floor(timestamp / tf)."""
    buckets: dict[int, dict] = {}
    for t in ticks:
        ts = float(t.get("timestamp", 0.0))
        price = float(t.get("price", t.get("mid", 0.0)))
        bucket = int(ts // timeframe_s)
        bar = buckets.get(bucket)
        if bar is None:
            buckets[bucket] = {"open_time": bucket * timeframe_s,
                               "open": price, "high": price,
                               "low": price, "close": price,
                               "count": 1}
        else:
            bar["high"] = max(bar["high"], price)
            bar["low"] = min(bar["low"], price)
            bar["close"] = price
            bar["count"] += 1
    return [buckets[k] for k in sorted(buckets)]


def transition(state: dict, bar_closed: dict) -> tuple[dict, list[dict]]:
    """Pure single transition for one closed bar.

    Returns (new_state, effects).  Effects are descriptions only;
    the caller journals first, then executes.  Replay calls this
    with the same inputs and discards effects.
    """
    new_state = dict(state)
    bars = list(new_state.get("bars", []))
    bars.append(dict(bar_closed))
    new_state["bars"] = bars
    new_state["watermark"] = float(bar_closed.get("open_time", 0.0))
    new_state["bar_count"] = int(new_state.get("bar_count", 0)) + 1
    effects = [{"type": "bar_closed",
                "open_time": bar_closed.get("open_time"),
                "bar_count": new_state["bar_count"]}]
    return new_state, effects
