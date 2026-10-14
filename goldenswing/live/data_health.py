"""Data-health gate: fail closed on stale or incomplete feed data.

The system is closed-bar-driven — intra-bar tick delays are irrelevant; only
missing CLOSED bars matter. This module answers one question per cycle: "is
the bar data fresh enough to trade on?" If not, the live loop must SKIP the
entire cycle: no tier evaluation, no monitor logging, no new entries. Open
positions are untouched (broker-held stops).

This is NOT a monitor Status (NOMINAL/WATCH/DEGRADED/HALT) — those describe
strategy health. Data health is a pipeline precondition. Conflating them
would corrupt recovery logic (e.g., a feed outage must not start the 7-day
HALT recovery clock).

Monitor interaction: skipped cycles write no lines to checks.jsonl, so
_sustained_degraded()'s window counter pauses naturally — neither advancing
(false HALT) nor resetting (amnesty for a degrading strategy).
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os

H4_SECONDS = 4 * 3600
H4_TOLERANCE = timedelta(minutes=30)   # normal broker feed delay after close
D1_TOLERANCE = timedelta(hours=6)      # daily bar must arrive within 6h of 00:00Z


@dataclass
class DataHealth:
    healthy: bool
    reason: str
    h4_gap_bars: int = 0      # expected H4 closes missing
    d1_stale: bool = False

    def __bool__(self) -> bool:
        return self.healthy


def _floor_4h(ts: datetime) -> datetime:
    epoch = int(ts.timestamp())
    floored = epoch - (epoch % H4_SECONDS)
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def check_data_health(latest_h4_close: datetime | None,
                      latest_d1_close: datetime | None,
                      now: datetime | None = None) -> DataHealth:
    """Validate bar freshness. All timestamps UTC.

    latest_h4_close: close time of the newest H4 bar we hold (None = none).
    latest_d1_close: close time of the newest D1 bar we hold (None = none).
    """
    now = now or datetime.now(timezone.utc)

    # --- H4: every completed 4h window must have its bar ---
    expected_h4 = _floor_4h(now)  # last completed window closed here
    h4_gap = 0
    if now - expected_h4 >= H4_TOLERANCE:
        # The expected bar is overdue: do we have it?
        if latest_h4_close is None or latest_h4_close < expected_h4:
            have = latest_h4_close or expected_h4 - timedelta(hours=4)
            h4_gap = max(1, round((expected_h4 - have).total_seconds() / H4_SECONDS))

    # --- D1: yesterday's bar must exist within tolerance of 00:00Z ---
    d1_stale = False
    today_open = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if now - today_open >= D1_TOLERANCE:
        # It's late enough that today's 00:00Z close must be in hand.
        if latest_d1_close is None or latest_d1_close < today_open:
            d1_stale = True

    if h4_gap:
        return DataHealth(False, f"H4 feed gap: {h4_gap} expected bar(s) missing",
                          h4_gap_bars=h4_gap, d1_stale=d1_stale)
    if d1_stale:
        return DataHealth(False, "D1 feed stale: latest daily close overdue",
                          d1_stale=True)
    return DataHealth(True, "feed fresh")


# ---------------------------------------------------------------------------
# Staleness escalation policy.
#
# The maximum allowable stale duration is POSITION-AWARE, not a single number:
#
#   FLAT (no open positions): indefinite skipping is safe. Nothing is at
#     risk, so there is nothing to protect. Ops alert at 48h anyway — a gap
#     that long usually means feed misconfiguration (wrong symbols, changed
#     bar times, dead credentials), not a transient outage.
#
#   HOLDING POSITIONS: 24h maximum. Rationale: 24h = 6 H4 bars = 1 full D1
#     bar. Beyond 24h blind, an entire daily regime reassessment (the T1 gate
#     that authorized the position) has been missed. The broker-held stops
#     still bound the loss, but the trade thesis can no longer be monitored.
#
# The intervention is ALERTING + RECONCILIATION, never blind trading:
#   - Alert the operator (Telegram): gap duration, open heat, last bar times.
#   - Do NOT auto-flatten: liquidating without prices is panic, not
#     fail-closed. The stops already bound the risk.
#   - On recovery: backfill missing bars in order, re-run T1/T2 on fresh
#     data, reconcile journal against broker positions.
# Each threshold fires ONCE per gap (no alert spam every cycle).
# ---------------------------------------------------------------------------

MAX_STALE_WITH_POSITIONS = timedelta(hours=24)
MAX_STALE_FLAT = timedelta(hours=48)


@dataclass
class GapEvent:
    kind: str  # "started" | "escalated" | "ended"
    started_at: str
    ended_at: str | None
    duration: timedelta
    h4_gap_bars: int
    heat_at_start_pct: float
    reason: str


class GapTracker:
    """Tracks feed gaps across cycles; fires position-aware escalations."""

    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.gap_path = os.path.join(state_dir, "data_gaps.jsonl")
        self.state_path = os.path.join(state_dir, "gap_state.json")
        self._state = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                pass
        return {"open_gap": None}

    def _save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._state, f)
        os.replace(tmp, self.state_path)

    def _log(self, event: GapEvent):
        with open(self.gap_path, "a") as f:
            f.write(json.dumps({
                "kind": event.kind, "started_at": event.started_at,
                "ended_at": event.ended_at,
                "duration_s": event.duration.total_seconds(),
                "h4_gap_bars": event.h4_gap_bars,
                "heat_at_start_pct": event.heat_at_start_pct,
                "reason": event.reason,
            }) + "\n")

    def update(self, health: DataHealth, now: datetime,
               current_heat_pct: float,
               started_at: datetime | None = None) -> GapEvent | None:
        """Call once per cycle with the freshness result.

        Returns a GapEvent when something noteworthy happens (gap started,
        escalation threshold crossed, gap ended), else None. Thresholds fire
        once per gap.

        started_at: backdate a newly-opened gap (startup path only). When the
        sidecars were lost during downtime, the feed's own staleness
        (health.h4_gap_bars) is the honest start estimate — better than
        pretending the gap began at restart, which would delay escalation.
        If the backdated start already exceeds the threshold, escalation
        fires on the first update, which is the correct fail-safe.
        """
        now_iso = now.isoformat()
        open_gap = self._state["open_gap"]

        if health.healthy:
            if open_gap is None:
                return None
            # Gap ended: close it out.
            started = datetime.fromisoformat(open_gap["started_at"])
            event = GapEvent(
                kind="ended", started_at=open_gap["started_at"],
                ended_at=now_iso, duration=now - started,
                h4_gap_bars=open_gap["h4_gap_bars"],
                heat_at_start_pct=open_gap["heat_at_start_pct"],
                reason=(f"feed recovered after {now - started}; "
                        f"backfill + reconcile before resuming entries"),
            )
            self._state["open_gap"] = None
            self._save()
            self._log(event)
            return event

        # Unhealthy: open or extend the gap.
        if open_gap is None:
            start_iso = started_at.isoformat() if started_at else now_iso
            open_gap = {
                "started_at": start_iso,
                "h4_gap_bars": health.h4_gap_bars,
                "heat_at_start_pct": current_heat_pct,
                "alerted": False,
            }
            self._state["open_gap"] = open_gap
            self._save()
            event = GapEvent(
                kind="started", started_at=start_iso, ended_at=None,
                duration=timedelta(0), h4_gap_bars=health.h4_gap_bars,
                heat_at_start_pct=current_heat_pct, reason=health.reason,
            )
            self._log(event)
            return event

        # Gap ongoing: check escalation (once per gap).
        started = datetime.fromisoformat(open_gap["started_at"])
        duration = now - started
        holding = open_gap["heat_at_start_pct"] > 0
        threshold = MAX_STALE_WITH_POSITIONS if holding else MAX_STALE_FLAT
        if not open_gap["alerted"] and duration >= threshold:
            open_gap["alerted"] = True
            self._save()
            why = ("24h blind with open positions: full D1 regime cycle missed"
                   if holding else
                   "48h stale while flat: likely feed misconfiguration")
            event = GapEvent(
                kind="escalated", started_at=open_gap["started_at"],
                ended_at=None, duration=duration,
                h4_gap_bars=health.h4_gap_bars,
                heat_at_start_pct=open_gap["heat_at_start_pct"],
                reason=f"OPERATOR ALERT — {why}; "
                       f"do not auto-flatten (no prices = no trading)",
            )
            self._log(event)
            return event

        # Update the running gap bar count (informational).
        open_gap["h4_gap_bars"] = max(open_gap["h4_gap_bars"], health.h4_gap_bars)
        self._save()
        return None

# ---------------------------------------------------------------------------
# Live-loop contract (for the paper/broker loop — the loop owns this sequence):
#
#   1. Fetch latest closed H4 + D1 bars WITH their close timestamps.
#   2. health = check_data_health(h4_ts, d1_ts, now)
#      event = gap_tracker.update(health, now, current_heat_pct)
#      if not health: SKIP cycle entirely (no tiers, no monitor.check(),
#                     no entries). If event and event.kind == "escalated":
#                     page operator. If event.kind == "ended": backfill missing
#                     bars from broker history, feed them through on_bar() IN
#                     ORDER, reconcile journal vs broker positions, then resume.
#   3. Normal path: run T1->T2->T3->T4, monitor, journal.
#
# Open positions are never touched by this gate. Their stops live at the
# broker; a feed gap does not cancel risk controls already in place.
# ---------------------------------------------------------------------------
