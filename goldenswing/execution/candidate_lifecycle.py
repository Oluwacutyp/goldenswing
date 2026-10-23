"""Candidate provenance, validity and execution eligibility.

Phase 2 (P0-1).  Before this module, a candidate record carried
the EVALUATED bar's timestamp and nothing about the pattern
itself: no formation time, no touch time, no creation time.
The native QM routes scan the whole frame (batch
``candidates()``), so on a warm start every historical touch
newer than an in-memory cursor was routed through the live
order path wearing the LATEST bar's timestamp and freshness —
EC2 run cb9b3fb522b7 emitted 32 such candidates in eight
seconds, patterns touched over the prior nine days, entries up
to $180 from the market.  The risk and friction gates refused
every one, but gates must not be the only thing standing
between a stale setup and an order.

The contract this module owns:

- PROVENANCE: every candidate record carries five distinct
  timestamps — pattern formation, pattern touch/trigger,
  candidate creation, evaluated bar, evaluation — plus its
  source and a stable instance key.  The evaluated bar's
  freshness never makes an old setup fresh: eligibility is
  computed from the TOUCH's age, not the frame's freshness.

- VALIDITY: each playbook declares how many bars after the
  touch a candidate stays live-order eligible (its validity
  window).  For the QM routes the window is 1 bar: the touch
  bar itself, plus the single re-evaluation close the RULING 6
  friction rule needs (a friction refusal does not consume the
  setup — it is re-evaluated on the NEXT close, then the
  window closes).

- SOURCE ELIGIBILITY: candidates produced by warmup scans,
  restart catch-up, replay or backfill are ANALYTICAL ONLY —
  categorically execution-ineligible, whatever their age.
  Only a candidate whose touch is discovered on a live close
  (``source == "live_close"``) inside its validity window may
  reach the order path.

- FAIL CLOSED: an age or provenance that cannot be established
  is ineligible.  "Unknown" is never a free pass.

- DEFENCE IN DEPTH: eligibility is stamped on the signal
  object and re-checked at the top of the shared order path —
  a candidate refused here is refused again there, so no
  caller bug can route around this module.
"""
from __future__ import annotations

#: Evaluation sources.  Only LIVE_CLOSE can produce an
#: execution-eligible candidate.
SOURCE_LIVE_CLOSE = "live_close"
SOURCE_WARMUP_SCAN = "warmup_scan"
SOURCE_TRIGGER_PATH = "trigger_path"
SOURCE_REPLAY = "replay"

#: Reason code for a candidate whose SOURCE forbids execution
#: (warmup / catch-up / replay / backfill provenance).
CODE_WARMUP_INELIGIBLE = "path.warmup_ineligible"
#: Reason code for a candidate whose age/provenance cannot be
#: established or exceeds its validity window (a data fact).
CODE_STALE_CANDIDATE = "gate.data"

#: Default validity window (bars after the touch) for the QM
#: native routes: the touch bar + the one re-evaluation close
#: the RULING 6 friction rule requires.
QM_VALIDITY_WINDOW_BARS = 1


def instance_key(playbook: str, *, pattern_ts, direction: int,
                 level) -> str:
    """A stable identity for ONE setup instance.

    Keyed on the pattern's TOUCH timestamp (an absolute time,
    invariant across frame-window slides — a bar INDEX is not:
    the frame is rebuilt every run and indices shift), the
    direction and the defended level.  The same setup instance
    produces the same key on every rescan, restart and replay,
    which is what makes idempotent routing provable.
    """
    return (f"{playbook}:{pattern_ts}:{int(direction)}:"
            f"{float(level):.2f}")


def assess_eligibility(*, source: str, touch_age_bars,
                       validity_window_bars: int
                       ) -> tuple[bool, str | None, str]:
    """May this candidate reach the order path?

    Returns (eligible, reason_code, reason).  Fail closed on
    every doubt: unknown source, unknown age, expired window.
    """
    if source != SOURCE_LIVE_CLOSE:
        return (False, CODE_WARMUP_INELIGIBLE,
                f"candidate source is {source!r}, not a live "
                "close: warmup / catch-up / replay candidates "
                "are analytical only and never reach the order "
                "path")
    if touch_age_bars is None:
        return (False, CODE_STALE_CANDIDATE,
                "candidate age cannot be established (no touch "
                "bar on record) — fail closed, no order")
    if int(touch_age_bars) < 0:
        return (False, CODE_STALE_CANDIDATE,
                f"candidate touch is {-int(touch_age_bars)} bars "
                "in the FUTURE of the evaluated bar — provenance "
                "is corrupt, fail closed, no order")
    if int(touch_age_bars) > int(validity_window_bars):
        return (False, CODE_STALE_CANDIDATE,
                f"candidate expired: touch is {int(touch_age_bars)}"
                f" bars older than the evaluated bar (validity "
                f"window {int(validity_window_bars)} bar(s)) — "
                "the evaluated bar's freshness does not revive "
                "an old setup")
    return (True, None,
            f"live-close candidate inside its validity window "
            f"(touch age {int(touch_age_bars)} bar(s))")


def provenance_block(*, key: str, formation_ts, touch_ts,
                     created_ts, evaluated_bar_ts, evaluation_ts,
                     source: str, touch_age_bars,
                     validity_window_bars: int,
                     eligible: bool,
                     eligibility_code: str | None) -> dict:
    """The provenance block carried on every candidate record.

    Five timestamps, kept distinct on purpose — conflating them
    is the defect this module exists to close:

    - ``pattern_formation_ts``: when the pattern's first pivot
      formed (the head, for QM);
    - ``pattern_touch_ts``: when price touched/triggered the
      level (the candidate's birth event);
    - ``candidate_created_ts``: when this candidate object was
      created by the scan that emitted it;
    - ``evaluated_bar_ts``: the bar the evaluation ran on;
    - ``evaluation_ts``: when the evaluation happened.
    """
    return {
        "pattern_formation_ts": (str(formation_ts)
                                 if formation_ts is not None
                                 else None),
        "pattern_touch_ts": (str(touch_ts)
                             if touch_ts is not None else None),
        "candidate_created_ts": (str(created_ts)
                                 if created_ts is not None
                                 else None),
        "evaluated_bar_ts": (str(evaluated_bar_ts)
                             if evaluated_bar_ts is not None
                             else None),
        "evaluation_ts": (str(evaluation_ts)
                          if evaluation_ts is not None
                          else None),
        "source": source,
        "instance_key": key,
        "touch_age_bars": (int(touch_age_bars)
                           if touch_age_bars is not None else None),
        "validity_window_bars": int(validity_window_bars),
        "execution_eligible": bool(eligible),
        "eligibility_code": eligibility_code,
    }
