"""Deterministic engine startup: validate journal vs sidecars, recover, gate.

The boundary between the journal and the sidecars:
- JOURNAL: source of truth for trades/decisions AND for whether recovery
  committed (the reconnection_complete marker). Append-only; survives crashes.
- gap_state.json: authority for CURRENT gap status (open or closed). Atomic
  replace on every write; save-then-log ordering means it is always at
  least as fresh as the log tail.
- data_gaps.jsonl: advisory audit history. May be missing tail events from
  crash windows (GapTracker saves state BEFORE logging). Never used alone
  to decide recovery.
- Fresh health observation at startup: ground truth about NOW. Overrides
  stale sidecars (e.g., a gap that ended during downtime but was never
  observed as ended).

The open-gap/no-marker rule (the question this module answers):
  An open gap with NO reconnection_complete marker is NORMAL — the marker
  is the recovery's commit point, written only after a gap ENDS and the
  batch commits. Expecting a marker for an open gap is a category error.
  The interrupted-recovery signal is an ENDED gap (per gap_state.json, or an
  "ended" event in data_gaps.jsonl) with no marker. Init must NEVER run
  reconciliation for an open gap — the broker state would be a moving target
  and the blind window has no end yet.

Startup phases:
  0. Journal integrity: existing startup_check, fail-closed. HALT -> return.
  1. Sidecar alignment: gap_state.json wins for current status.
  2. Fresh ground truth: observe health NOW.
       healthy + open gap -> the gap ended (possibly during downtime):
           close it via GapTracker, then recover.
       healthy + no open gap -> nothing to do.
       stale + open gap -> gap continues; tick loop will health-skip.
       stale + no open gap -> gap started during downtime (sidecars lost):
           open it backdated from the feed's own staleness (h4_gap_bars).
  3. Historical catch-up: recover_if_needed for any other ended gap
     without a marker (covers crashes between "ended" and the sequence).
  4. Divergence gate: unresolved DIVERGENCE (journal-closed/broker-open)
     blocks engine start — sizing entries against a wrong heat model is
     overleverage. Other needs_review items alert but don't block. A later
     divergence_cleared (see resolve_divergence) releases the block.

On any recovery exception: recorded in the report, ready=False. The next
scheduled run retries from the journal — nothing is half-committed thanks
to the marker-as-commit-point design.

Locking and time budgets:
  The whole sequence runs inside one journal_transaction (the init lock),
  making boot a true transactional boundary across the journal and sidecars.
  Every blocking call inside it is bounded: per-call watchdog timeouts plus
  a global INIT_BUDGET_S deadline (monotonic clock). A per-call TimeoutError
  in Phase 2 falls back to the existing stale path; InitBudgetExhausted
  anywhere aborts init outright — never degraded, never partial. Lock
  acquisition itself is bounded too: concurrent boots serialize, and a
  wedged holder can't block boot past the deadline.
"""
from __future__ import annotations

import functools
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from goldenswing.live.data_health import GapTracker, DataHealth, H4_SECONDS
from goldenswing.live.journal_lock import journal_transaction, call_with_timeout
from goldenswing.live.reconcile import (
    DIVERGENCE,
    _open_trade_ids,
    _unresolved_divergences,
    recover_if_needed,
)

log = logging.getLogger(__name__)

# Init time budgets (seconds). The whole initialize_engine body runs inside
# one journal_transaction, so every blocking call inside it must be bounded —
# otherwise a hung network call holds the exclusive lock forever and every
# tick skips until the operator intervenes.
INIT_BUDGET_S = 300.0
HEALTH_TIMEOUT_S = 30.0
BROKER_TIMEOUT_S = 90.0
BACKFILL_TIMEOUT_S = 120.0
PRICE_TIMEOUT_S = 30.0


class InitBudgetExhausted(Exception):
    """Global init deadline breached: abort init immediately.

    Deliberately NOT a TimeoutError subclass. Per-call TimeoutError means
    "this probe failed" (Phase 2 handles it via the stale path); this means
    "we are out of time entirely" (abort, never degrade). The two signals
    stay orthogonal so handler ordering stays meaningful.
    """


@dataclass
class InitReport:
    startup_status: str = "UNKNOWN"
    startup_reason: str = ""
    gaps_recovered: list = field(default_factory=list)  # [(gap_start, status|None)]
    gaps_open: list = field(default_factory=list)       # [started_at_iso]
    needs_review: dict = field(default_factory=dict)    # {trade_id: event}
    blocking_divergences: list = field(default_factory=list)
    ready: bool = False
    ready_reason: str = ""
    error: str | None = None


def align_gap_sidecars(gap_tracker: GapTracker) -> dict | None:
    """Authoritative current gap (dict) or None.

    gap_state.json wins for current status: GapTracker persists state BEFORE
    appending the log event, so in every crash window the state file is at
    least as fresh as the log tail. The log is advisory history — it may be
    missing the final event of a gap, which is why recovery decisions also
    consult the journal marker (see initialize_engine).
    """
    open_gap = gap_tracker._state.get("open_gap")
    if not isinstance(open_gap, dict) or not open_gap.get("started_at"):
        return None
    return open_gap


def _backdated_gap_start(health, now: datetime) -> datetime:
    """Estimate when the current stale period began, from the feed itself.

    h4_gap_bars = expected H4 closes missing — a direct measurement of
    staleness. Conservative (may overstate by part of a bar), which is the
    safe direction for the 24h/48h escalation thresholds.
    """
    bars = max(1, int(getattr(health, "h4_gap_bars", 0) or 0))
    return now - timedelta(seconds=H4_SECONDS * bars)


@contextmanager
def _bounded_init_lock(journal_path: str, deadline: float):
    """Acquire the journal lock, polling against the init deadline.

    Serializes concurrent boots: the loser waits out the winner's init, then
    re-runs against committed markers (usually a no-op). A genuinely wedged
    holder can't block boot forever — the deadline still applies.
    """
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise InitBudgetExhausted("init lock acquisition timed out")
        with journal_transaction(journal_path, blocking=False) as acquired:
            if acquired:
                yield
                return
        time.sleep(min(1.0, remaining))


def initialize_engine(*, journal_path: str,
                      recovery,
                      gap_tracker: GapTracker,
                      tracker,
                      tracker_state_path: str | None,
                      check_data_health_fn,
                      fetch_broker_state,
                      fetch_missing_bars,
                      last_observed_price_fn,
                      tier1_fn=None, tier2_fn=None,
                      on_alert=None,
                      now: datetime | None = None) -> InitReport:
    """Run the deterministic startup sequence. See module docstring.

    recovery: RecoveryController (owns startup_check and the monitor).
    """
    now = now or datetime.now(timezone.utc)
    monitor = recovery.monitor
    report = InitReport()
    # The whole sequence runs inside one journal_transaction: init is a
    # read-modify-write transaction across the journal and the sidecars, and
    # the lock is what makes it atomic w.r.t. concurrent boots and ticks.
    # Every blocking call inside it is therefore bounded by the deadline —
    # a hung network call must never hold the exclusive lock forever.
    deadline = time.monotonic() + INIT_BUDGET_S

    def _budget_for(phase_cap_s: float, phase: str) -> float:
        """Effective timeout for one call: its own cap clamped to the time
        left in the global budget. Raises InitBudgetExhausted instead of
        returning a non-positive timeout — never start a call with no time."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise InitBudgetExhausted(f"budget exhausted before {phase}")
        return min(phase_cap_s, remaining)

    def _check_deadline(phase: str) -> None:
        """Deadline check for non-network stretches (sidecar alignment,
        journal scans) that never pass through the watchdog."""
        if time.monotonic() >= deadline:
            raise InitBudgetExhausted(f"budget exhausted during {phase}")

    def _timed(fn, phase_cap_s: float, label: str):
        """Wrap an injected blocking call with per-call + global budgets.

        The clamp is evaluated at call time, so later phases in a long
        sequence automatically get smaller timeouts as the budget drains.
        """
        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            return call_with_timeout(
                lambda: fn(*args, **kwargs),
                _budget_for(phase_cap_s, label),
                label=f"init-{label}")
        return _wrapper

    def _alert(msg: str) -> None:
        log.warning(msg)
        if on_alert is not None:
            try:
                on_alert(msg)
            except Exception:
                log.exception("on_alert failed during initialize_engine")

    def _abort_budget(exc: InitBudgetExhausted) -> InitReport:
        # Pathological, not routine: the network is effectively down. Page
        # the operator; the next scheduled run retries from the journal.
        _alert(f"startup: init budget exhausted: {exc}")
        report.error = str(exc)
        report.ready = False
        report.ready_reason = f"init budget exhausted: {exc}; next run retries"
        return report

    try:
        with _bounded_init_lock(journal_path, deadline):
            # Phase 0 — journal integrity, fail-closed. Lock acquisition
            # already repaired any torn tail from a crashed writer.
            try:
                status, reason = recovery.startup_check(journal_path)
            except InitBudgetExhausted:
                raise
            except Exception as e:  # fail-closed on unexpected failure
                log.exception("startup_check raised")
                report.startup_status = "HALT"
                report.startup_reason = f"startup_check raised: {e}"
                report.ready = False
                report.ready_reason = ("startup_check raised; "
                                       "manual review required")
                report.error = str(e)
                return report
            _check_deadline("startup-check")
            report.startup_status = status.name
            report.startup_reason = reason
            if status.name == "HALT":
                report.ready = False
                report.ready_reason = f"startup_check HALT: {reason}"
                return report

            # Phase 1 — sidecar alignment.
            open_gap = align_gap_sidecars(gap_tracker)
            _check_deadline("sidecar-alignment")

            # Phase 2 — fresh ground truth.
            # Handler order is load-bearing: InitBudgetExhausted must precede
            # the broad except Exception. Python matches handlers top-down,
            # and the broad clause would swallow the budget signal —
            # misclassifying "out of time, stop everything" as "probe
            # failed, take the stale path".
            try:
                health = _timed(check_data_health_fn, HEALTH_TIMEOUT_S,
                                "health-check")()
            except InitBudgetExhausted:
                raise  # global abort — never the stale path
            except Exception as e:
                log.exception("health check failed at startup; assuming stale")
                health = None
                health_error = str(e)
            else:
                health_error = None

            if health is not None and health.healthy:
                if open_gap is not None:
                    # The gap ended — possibly during downtime, unobserved.
                    # Close it officially, then let Phase 3 recover it.
                    event = gap_tracker.update(health, now, current_heat_pct=0.0)
                    log.info("startup: closed unobserved-ended gap started=%s",
                             open_gap.get("started_at"))
                    open_gap = None
            else:
                # Stale (or health check failed): ensure a gap is open.
                if open_gap is None:
                    if health is None:
                        # Health unknowable: assume stale, fail-closed. No
                        # backdate — the start is genuinely unknown.
                        health = DataHealth(
                            healthy=False,
                            reason=f"health check failed at startup: {health_error}",
                            h4_gap_bars=0)
                        started_at = now
                    else:
                        started_at = _backdated_gap_start(health, now)
                    # Boolean-proxy heat: at startup only presence/absence of
                    # open positions is knowable from the journal. The threshold
                    # only tests > 0, so this is honest for its purpose.
                    holding = 1.0 if _open_trade_ids(journal_path) else 0.0
                    gap_tracker.update(health, now, holding,
                                       started_at=started_at)
                    log.info("startup: opened gap backdated to %s (reason=%s)",
                             started_at.isoformat(), health.reason)
                    open_gap = align_gap_sidecars(gap_tracker)
                # else: gap continues; the tick loop will health-skip.
            _check_deadline("phase-2")

            # Phase 3 — historical catch-up (idempotent; marker-guarded).
            # Any per-call timeout or budget exhaustion aborts init: pressing
            # on with half-observed broker state is worse than retrying.
            recovered = recover_if_needed(
                journal_path, tracker, monitor, gap_tracker.state_dir,
                tracker_state_path,
                _timed(fetch_broker_state, BROKER_TIMEOUT_S, "broker-state"),
                _timed(fetch_missing_bars, BACKFILL_TIMEOUT_S, "backfill"),
                _timed(last_observed_price_fn, PRICE_TIMEOUT_S, "last-price"),
                tier1_fn=tier1_fn, tier2_fn=tier2_fn,
                now=now)
            for gs, _status, rep in recovered:
                report.gaps_recovered.append(
                    (gs, _status.name if _status is not None else None,
                     list(rep.trades_closed)))
            _check_deadline("phase-3")

            if open_gap is not None:
                report.gaps_open.append(open_gap.get("started_at"))

            # Phase 4 — divergence gate.
            report.needs_review = _unresolved_divergences(journal_path)
            report.blocking_divergences = [
                tid for tid, ev in report.needs_review.items() if ev == DIVERGENCE
            ]
            if report.needs_review:
                _alert(f"startup: {len(report.needs_review)} unresolved "
                       f"reconciliation item(s): {report.needs_review}")
            if report.blocking_divergences:
                report.ready = False
                report.ready_reason = (
                    f"unresolved position divergence(s): "
                    f"{report.blocking_divergences} — human must resolve via "
                    f"resolve_divergence() before entries resume")
                return report

            report.ready = True
            report.ready_reason = "startup validation complete"
            return report
    except InitBudgetExhausted as e:
        return _abort_budget(e)
    except Exception as e:
        log.exception("initialize_engine failed")
        report.error = str(e)
        report.ready = False
        report.ready_reason = f"initialization raised: {e}; next run retries"
        return report
