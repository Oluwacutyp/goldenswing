"""Journal auditability helpers (owner, 2026-10-26).

The 2026-10-26 assessment read exposed three ways a MARKET
ASSESSMENT could mislead without lying: a record with no
evaluation timestamp ("evaluated not recorded"), a record with
no freshness measurement, and an observed candidate whose only
refusal explanation was the outcome code
``path.trigger_not_fired`` — an outcome, not the trigger
decision itself.  This module is the pure, testable core of
the fix.  It decides nothing and changes no order path; it
only computes the audit facts every evaluation record must
carry so the renderer can print them truthfully:

- FRESHNESS: seconds from the last CLOSED bar's close to the
  evaluation clock, per timeframe period — the same basis the
  QM routes already use (last bar + one period, tick clock
  preferred, wall clock fallback).  A market-closed feed is
  not "stale" by this measure alone; the number is reported
  and the calendar verdict lives in freshness_line / the
  market-hours gate.  ``None`` means unmeasurable (no bar, no
  clock, unknown timeframe) — never 0.0 by default, because
  0.0 reads as "just now".
- SOURCE: only ``live_close`` and ``trigger_path`` evaluations
  are live decisions.  ``warmup_scan``, ``replay`` and
  ``backfill`` are analytical records: they describe what a
  scan saw on historical bars and can never be mistaken for a
  live decision, whatever verdict they carry.
- TRIGGER DECISION: for a scheduled-scan candidate, the
  complete, checkable account of why no order followed: the
  scan route holds no order authority by design, the scan
  does NOT run the h1_trigger streaming evaluation (so it
  cannot claim that evaluation fired or failed — only that
  this scan is not it), the source eligibility, and the
  detector's own facts (the signal's live_context, with bar
  indices mapped to timestamps when a frame is given).
  For roster playbooks the detector facts ARE the trigger
  predicates in substance: e.g. inversion fires exactly when
  a gap formed, closed through, and was retested without a
  reclaim — those indices, on the record, are the predicate
  outcomes.  What the scan cannot supply is the streaming
  trigger-path verdict for the same instance; the block says
  so instead of inventing it.
"""
from __future__ import annotations

#: Bar period in seconds per timeframe (MT5 clock: a bar's
#: close is its open timestamp + one period).
TF_PERIOD_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
    "MN1": 2592000,
}

#: Sources whose evaluations are live decisions.
LIVE_DECISION_SOURCES = ("live_close", "trigger_path")

#: Sources that are analytical records only, with the note the
#: renderer prints for each.
NON_LIVE_SOURCE_NOTES = {
    "warmup_scan": "startup scan over historical bars — "
                   "analytical only, never routed to orders",
    "replay": "replay evaluation — analytical only, never "
              "routed to orders",
    "backfill": "backfill evaluation — analytical only, "
                "never routed to orders",
}


def freshness_seconds(now, last_bar_ts, timeframe) -> float | None:
    """Seconds from the last closed bar's close to ``now``.

    ``last_bar_ts`` is the bar's OPEN timestamp; its close is
    open + one period of ``timeframe``.  Returns ``None`` when
    the measurement cannot be made (missing clock, missing
    bar, unknown timeframe) — an unmeasured feed must not
    render as fresh.
    """
    period = TF_PERIOD_SECONDS.get(str(timeframe or "").upper())
    if period is None or now is None or last_bar_ts is None:
        return None
    try:
        import pandas as pd

        now_ts = pd.Timestamp(now)
        bar_ts = pd.Timestamp(last_bar_ts)
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("UTC")
        close_ts = bar_ts + pd.Timedelta(seconds=period)
        return max(0.0, (now_ts - close_ts).total_seconds())
    except Exception:
        return None


def is_live_decision_source(source) -> bool:
    """True only for evaluations that are live decisions."""
    return str(source or "") in LIVE_DECISION_SOURCES


def source_note(source) -> str | None:
    """The analytical-only note for a non-live source, else None."""
    return NON_LIVE_SOURCE_NOTES.get(str(source or ""))


def detector_facts(signal, frame=None) -> dict:
    """The detector's own facts for one signal, as plain scalars.

    From the signal's ``live_context``: numbers, strings and
    booleans pass through; bar-index keys (``*_idx``) gain a
    timestamp twin (``*_ts``) mapped through ``frame`` when the
    index is in range.  Nested structures are dropped — a
    journal fact must be readable, not a dump.  Returns {} when
    the signal carries no live_context.
    """
    ctx = getattr(signal, "live_context", None) or {}
    if not isinstance(ctx, dict):
        return {}
    facts: dict = {}
    for key in sorted(ctx):
        value = ctx[key]
        if isinstance(value, bool) or isinstance(
                value, (int, float, str)):
            facts[str(key)] = value
        elif value is None:
            facts[str(key)] = None
        if isinstance(value, int) and not isinstance(value, bool) \
                and str(key).endswith("_idx") and frame is not None:
            try:
                if 0 <= value < len(frame):
                    facts[str(key) + "_ts"] = str(frame.index[value])
            except Exception:
                pass
        if len(facts) >= 32:
            break
    return facts


def trigger_decision_block(*, route: str, source: str,
                           reason_code: str | None,
                           shadow: bool,
                           detector_facts: dict | None) -> dict:
    """The complete trigger account for a scheduled-scan candidate.

    The scan route never orders: that is the first fact, not a
    footnote.  The second fact is the scan's own limit — it did
    not run the h1_trigger streaming evaluation, so it records
    no verdict for it.  A reader who wants the streaming
    verdict finds it on the h1_trigger record for the same
    playbook + instance, never inferred from this scan.
    """
    if shadow:
        authority = ("none — §G shadow route: journaled for "
                     "grading, holds NO order authority")
    else:
        authority = ("none — scheduled scan: the scan observes "
                     "and journals; only the h1_trigger streaming "
                     "evaluation can route to orders")
    if source == "live_close":
        eligibility = ("live-close scan — still no order "
                       "authority on this route (see above)")
    else:
        eligibility = (source_note(source) or
                       f"source {source!r} — analytical record")
    return {
        "route": route,
        "order_authority": authority,
        "source": source,
        "source_is_live_decision": is_live_decision_source(source),
        "source_eligibility": eligibility,
        "trigger_path_evaluated_by_this_scan": False,
        "trigger_path_note": (
            "this scan does not run the h1_trigger streaming "
            "evaluation and records no verdict for it — it "
            "neither fired nor failed here.  Execution requires "
            "the streaming instance to emit the same instance "
            "on a live close; that verdict, if any, lives on "
            "the h1_trigger record for the same playbook and "
            "instance key"),
        "reason_code": reason_code,
        "detector_facts": dict(detector_facts or {}),
    }
