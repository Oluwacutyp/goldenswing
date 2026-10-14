"""Fair Value Gap (FVG) detection — streaming core for backtest AND live.

A bullish FVG forms when three consecutive bars show:
    Low[T] > High[T-2]
The middle bar (T-1) is the displacement candle. The gap between
High[T-2] and Low[T] is unfilled imbalance — institutional orders
left resting there.

A bearish FVG mirrors: High[T] < Low[T-2].

Design: the core is STREAMING. It accepts one closed bar at a time via
FVGTracker.on_bar(). The backtest feeds history in a loop; the live paper
loop feeds one bar per hourly tick. Same code path, zero environment
branching. The bulk scan_fvgs() is a thin wrapper over the tracker.

Zones are tracked through their lifecycle:
    ACTIVE → MITIGATED (price traded into zone) → FILLED (traded through)

Only closed bars are ever evaluated. No repainting.
State is JSON-serializable so the live loop can persist it between ticks.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class FVGZone:
    """A single Fair Value Gap zone."""
    top: float
    bottom: float
    mid: float
    direction: int          # +1 bullish (support), -1 bearish (resistance)
    formed_idx: int         # bar index where the gap was confirmed (T)
    anchor_idx: int         # bar index of the displacement candle (T-1)
    status: str = "ACTIVE"  # ACTIVE | MITIGATED | FILLED
    mitigated_idx: int | None = None
    # Epistemic flag (not a lifecycle state): True when price action during a
    # feed gap may have interacted with this zone but was never observed.
    # Suspect zones are excluded from signal generation until a fresh
    # OBSERVED bar interacts with them (re-validation) — we never reconstruct
    # unobserved history, so unobserved interaction stays unknown.
    suspect: bool = False

    def to_dict(self) -> dict:
        return {
            "top": self.top, "bottom": self.bottom, "mid": self.mid,
            "direction": self.direction, "formed_idx": self.formed_idx,
            "anchor_idx": self.anchor_idx, "status": self.status,
            "mitigated_idx": self.mitigated_idx, "suspect": self.suspect,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FVGZone":
        # Old dicts predate "suspect" -> default False (backward compatible).
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


class FVGTracker:
    """Streaming FVG core. Feed one closed bar at a time.

    Usage (identical for backtest and live):
        tracker = FVGTracker()
        for i, bar in enumerate(bars):       # backtest
            new_zones = tracker.on_bar(bar.high, bar.low, i)
        # --- or live ---
        new_zones = tracker.on_bar(h, l, idx)  # one bar per tick
        state = tracker.to_dict()              # persist between ticks
        tracker = FVGTracker.from_dict(state)  # restore next tick
    """

    def __init__(self, max_age_bars: int = 100, keep_bars: int = 5):
        self.max_age_bars = max_age_bars
        # Rolling window of recent (high, low) — only 3 needed for detection,
        # a few extra for safety. Stores (idx, high, low).
        self._bars: deque[tuple[int, float, float]] = deque(maxlen=keep_bars)
        self.zones: list[FVGZone] = []
        self.last_idx: int = -1
        # Feed gaps the tracker was told about (gap_before=True): wall-clock
        # bars missed with no backfill. Detection was skipped across them.
        self.gaps: list[dict] = []

    def on_bar(self, high: float, low: float, idx: int,
               gap_before: bool = False) -> list[FVGZone]:
        """Feed one newly CLOSED bar. Returns zones confirmed at this bar.

        Also advances lifecycle of all tracked zones using this bar.
        Raises ValueError on out-of-order or duplicate bar indices —
        silent reordering is how live bugs hide.

        gap_before=True: the feeder knows wall-clock bars were missed and
        backfill was impossible. Detection is SKIPPED for this bar (the T-2
        pattern cannot be trusted across a time gap), but lifecycle updates
        still run — the price action in this bar really happened. The gap is
        recorded in state for audit. Callers must treat zones formed around
        a gap as suspect: prefer backfill (ordered re-feed, no flag needed).
        """
        if idx <= self.last_idx:
            raise ValueError(
                f"Bars must arrive in order: got idx={idx}, last={self.last_idx}"
            )
        if gap_before:
            self.gaps.append({"before_idx": idx, "after_last_idx": self.last_idx})
        self.last_idx = idx
        self._bars.append((idx, float(high), float(low)))

        new_zones: list[FVGZone] = []

        # Detection needs bars idx-2, idx-1, idx
        if len(self._bars) >= 3 and not gap_before:
            b0 = self._bars[-3]
            b2 = self._bars[-1]
            # Indices must be consecutive for a valid 3-bar pattern
            if b2[0] == b0[0] + 2:
                zone = self._detect(b0[1], b0[2], b2[1], b2[2], idx)
                if zone:
                    self.zones.append(zone)
                    new_zones.append(zone)

        # Lifecycle update for all existing zones (skip the ones just formed)
        self._update_lifecycle(float(high), float(low), idx)

        # Prune: drop filled zones older than max_age to bound memory
        self.zones = [
            z for z in self.zones
            if not (z.status == "FILLED" and idx - z.formed_idx > self.max_age_bars)
        ]
        return new_zones

    @staticmethod
    def _detect(h0: float, l0: float, h2: float, l2: float,
                idx: int) -> FVGZone | None:
        """Pure 3-bar gap check. No state, no arrays — the mathematical core."""
        if l2 > h0:  # bullish
            return FVGZone(top=l2, bottom=h0, mid=(l2 + h0) / 2,
                           direction=1, formed_idx=idx, anchor_idx=idx - 1)
        if h2 < l0:  # bearish
            return FVGZone(top=l0, bottom=h2, mid=(l0 + h2) / 2,
                           direction=-1, formed_idx=idx, anchor_idx=idx - 1)
        return None

    def _update_lifecycle(self, high: float, low: float, idx: int) -> None:
        for z in self.zones:
            if z.status == "FILLED" or z.formed_idx >= idx:
                continue
            # Any OBSERVED bar interacting with a suspect zone re-validates it:
            # we now know its current state honestly, so the epistemic flag
            # clears and normal lifecycle resumes. Bars that don't touch it
            # leave it suspect — no new information.
            touched = (low <= z.top and high >= z.bottom)
            if touched and z.suspect:
                z.suspect = False
            if z.direction == 1:  # bullish zone = support
                if touched:
                    if z.status == "ACTIVE":
                        z.status = "MITIGATED"
                        z.mitigated_idx = idx
                    if low < z.bottom:
                        z.status = "FILLED"
            else:  # bearish zone = resistance
                if touched:
                    if z.status == "ACTIVE":
                        z.status = "MITIGATED"
                        z.mitigated_idx = idx
                    if high > z.top:
                        z.status = "FILLED"

    def active(self, direction: int | None = None) -> list[FVGZone]:
        """Tradeable zones: ACTIVE or MITIGATED, fresh enough, optional direction.
        Suspect zones (unobserved gap interaction) are excluded until a fresh
        observed bar re-validates them."""
        return [
            z for z in self.zones
            if z.status != "FILLED"
            and not z.suspect
            and self.last_idx - z.formed_idx <= self.max_age_bars
            and (direction is None or z.direction == direction)
        ]

    def nearest(self, price: float,
                direction: int | None = None) -> FVGZone | None:
        """Closest active zone to price (for confluence scoring)."""
        zones = self.active(direction)
        return min(zones, key=lambda z: abs(z.mid - price)) if zones else None

    # -- persistence for the live loop (state survives between hourly ticks) --

    def to_dict(self) -> dict:
        return {
            "max_age_bars": self.max_age_bars,
            "last_idx": self.last_idx,
            "bars": [list(b) for b in self._bars],
            "zones": [z.to_dict() for z in self.zones],
            "gaps": self.gaps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FVGTracker":
        t = cls(max_age_bars=d.get("max_age_bars", 100))
        t.last_idx = d.get("last_idx", -1)
        t._bars = deque(
            [(int(b[0]), float(b[1]), float(b[2])) for b in d.get("bars", [])],
            maxlen=5,
        )
        t.zones = [FVGZone.from_dict(z) for z in d.get("zones", [])]
        t.gaps = d.get("gaps", [])
        return t


# ---------------------------------------------------------------------------
# Bulk convenience wrapper (backtest). Thin layer over the streaming core.
# ---------------------------------------------------------------------------

def scan_fvgs(high, low, max_age_bars: int = 100) -> list[FVGZone]:
    """Scan all closed bars. Equivalent to feeding the tracker in a loop."""
    tracker = FVGTracker(max_age_bars=max_age_bars)
    for i in range(len(high)):
        tracker.on_bar(high[i], low[i], i)
    return tracker.zones
