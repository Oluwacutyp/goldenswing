"""Session-aware timeframe mapping — DST-aware, tags-only, never a veto.

WHY THIS MODULE EXISTS
----------------------
The repo has THREE fixed-UTC session labellers (``reader/sessions.py``,
``live/regime.py``, ``backtest/mtf12.py``), and ``reader/sessions.py``
says it outright: "DST ignored — documented approximation".  A fixed
UTC hour is wrong twice a year: the ICT killzones the repo uses
(London 07:00-10:00 UTC, NY 12:00-15:00 UTC in
``STRATEGIES_AND_LOGICS.md`` §2.5 / ``backtest/real_trader.py``) are
the WINTER values of wall-clock windows.  After the US springs forward
the same New York wall clock is one UTC hour earlier, so a fixed-UTC
killzone fires an hour late all summer — and fires an hour wrong in
both directions during the March / October-November gap weeks when
the US and the UK are on different DST states.

This module is the DST-aware replacement for NEW code.  The three
legacy labellers are deliberately NOT changed: tests pin them and
backtest comparability depends on them.  Nothing here re-labels
history; this is the mapping the live engine consults going forward.

SESSION DEFINITIONS (mined, not invented)
-----------------------------------------
Boundaries are wall-clock times in the exchange's own IANA zone, so
``zoneinfo`` applies the DST rules — no fixed UTC offset anywhere:

- Asia session opens 09:00 Asia/Tokyo (Tokyo has no DST, so this is
  always 00:00 UTC — the repo's Asia 00:00-07:00 UTC bucket).
- London session opens 07:00 Europe/London, closes 16:00 Europe/London.
- New York session opens 07:00 America/New_York, closes 16:00
  America/New_York.
- The day partitions as: asia [Tokyo open, London open) -> london
  [London open, NY open) -> overlap [NY open, London close) ->
  newyork [London close, NY close) -> off [NY close, Tokyo open).
  In northern winter this reproduces the repo's buckets EXACTLY:
  Asia 00-07, London 07-12, Overlap 12-16, NY 16-21, Off 21-24 UTC.
  In summer every boundary shifts with its own clock (e.g. NY open
  07:00 EDT = 11:00 UTC, London close 16:00 BST = 15:00 UTC, so the
  summer overlap is 11:00-15:00 UTC).  Gap weeks fall out of the same
  rule instead of being special-cased.

KILLZONES (ICT, anchored to New York wall clock — the ICT origin)
------------------------------------------------------------------
- London killzone: 02:00-05:00 America/New_York.  Winter (EST, UTC-5)
  that is 07:00-10:00 UTC — the repo's London KZ.  Summer (EDT, UTC-4)
  it is 06:00-09:00 UTC.
- NY killzone: 07:00-10:00 America/New_York.  Winter 12:00-15:00 UTC
  — the repo's NY KZ.  Summer 11:00-14:00 UTC.
- The killzone is an independent flag, not a session: the NY killzone
  sits at the start of the overlap bucket; the London killzone sits
  inside the London bucket (02:00 NY == 07:00 London except in gap
  weeks, when it is off by one hour — documented, not fudged).

TIMEFRAME MAPPING (composes with the repo's preset clocks)
-----------------------------------------------------------
The mapping selects the ENTRY CLOCK only.  Structure clocks are fixed
and session-independent — the session can never flip bias, because
bias lives on H4 and the mapping never carries a direction:

- bias_tf = H4 always (the Intraday bias clock in ``live/multitf.py``;
  the Swing D1 / Scalp H1 bias clocks belong to their presets, not
  to the session — a session must never re-assign which TF gives
  bias).
- zone_tf = H1 always (the Intraday zone clock; POIs live on H1).
- trigger_tf = M15 inside a killzone (London KZ or NY KZ — peak
  liquidity, the Intraday trigger clock), M30 everywhere else
  (Asia / off-session range-building, London / overlap / NY outside
  the killzones — a mid clock that does not chase M5 noise).
- active_tfs = (bias_tf, zone_tf, trigger_tf).

COMPOSITION WITH AMD + QM
-------------------------
- AMD phase is carried through as a tag (``amd:<phase>``), never
  re-derived and never changed by the session.  A manipulation phase
  inside a killzone earns the descriptive tag
  ``manipulation_in_killzone`` — a tag, never a gate.
- QM M5 keeps its OWN entry clock (M5, ``playbooks/qm_m5.py``):
  ``compose(..., playbook="qm_m5")`` reports
  ``playbook_entry_clock="M5"`` with the note that the session
  mapping annotates the QM signal but never overrides its clock.
  QM is still session-agnostic until measured — this module does not
  claim a session edge for it.

CONTRACT (standing rules)
-------------------------
- Tags / context only.  The mapping NEVER flips bias, NEVER vetoes,
  NEVER gates a trade.  It has no direction field at all.
- Pure functions, no state, no cursors: calling twice with the same
  timestamp returns the same answer.  There is no ``_processed`` to
  rewind, so cursor-safety is by construction.
- Naive timestamps are treated as UTC (the repo's reader convention:
  ``reader/reader.py`` localizes naive index to UTC).
- This is a clock, not a market calendar: weekends/holidays still map
  (the market is closed, but the clock is not).  Do not use it to
  decide whether the market is open.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

TOKYO = ZoneInfo("Asia/Tokyo")
LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc

#: Asia session opens at this Tokyo wall-clock time (no DST in Tokyo).
ASIA_OPEN = time(9, 0)
#: London session opens / closes at these London wall-clock times.
LONDON_OPEN = time(7, 0)
LONDON_CLOSE = time(16, 0)
#: New York session opens / closes at these NY wall-clock times.
NY_OPEN = time(7, 0)
NY_CLOSE = time(16, 0)
#: ICT killzones in New York wall-clock time (see module docstring).
LONDON_KZ = (time(2, 0), time(5, 0))
NY_KZ = (time(7, 0), time(10, 0))

#: Structure clocks: fixed, session-independent (never carry direction).
BIAS_TF = "H4"
ZONE_TF = "H1"
#: Entry clock inside a killzone (peak liquidity, Intraday trigger).
KILLZONE_TRIGGER_TF = "M15"
#: Entry clock outside a killzone (range-building / fading liquidity).
OFF_KILLZONE_TRIGGER_TF = "M30"

#: AMD phases this module knows how to tag (from features/phases.py).
#: Unknown strings pass through verbatim as ``amd:<value>`` — the
#: mapping never validates or re-derives phase.
AMD_PHASES = (
    "accumulation", "distribution", "consolidation", "manipulation",
    "expansion_up", "expansion_down", "trend_up", "trend_down",
    "unknown",
)

SESSIONS = ("asia", "london", "overlap", "newyork", "off")
KILLZONES = ("london_kz", "ny_kz")


def _as_utc(ts) -> datetime:
    """Coerce a timestamp to an aware UTC datetime (naive == UTC)."""
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _latest_daily_boundary(now_utc: datetime, zone: ZoneInfo,
                           wall: time) -> datetime:
    """Most recent occurrence (<= now) of ``wall`` o'clock in ``zone``.

    Takes the wall time on the local date of ``now``; if that instant
    is still in the future, steps back one local day.  zoneinfo
    resolves the UTC offset for that date, so DST transitions are
    handled by the tz database, not by arithmetic here.
    """
    local_date = now_utc.astimezone(zone).date()
    cand = datetime.combine(local_date, wall, tzinfo=zone)
    if cand > now_utc:
        cand = datetime.combine(local_date - timedelta(days=1), wall,
                                tzinfo=zone)
    return cand.astimezone(UTC)


def session_of(ts) -> tuple[str, str | None]:
    """DST-aware (session, killzone) for a timestamp.

    Returns e.g. ``("london", "london_kz")``, ``("overlap", "ny_kz")``,
    ``("asia", None)``.  Killzone is ``"london_kz"`` | ``"ny_kz"`` |
    ``None``.  Pure function — no state.
    """
    now = _as_utc(ts)
    # --- session bucket: latest boundary wins.  Boundaries occur in
    # a fixed order through the day (Asia open ~00Z, London open
    # ~06-07Z, NY open ~11-12Z, London close ~15-16Z, NY close
    # ~20-21Z), so "latest boundary <= now" is a clean partition.
    boundaries = [
        ("asia", _latest_daily_boundary(now, TOKYO, ASIA_OPEN)),
        ("london", _latest_daily_boundary(now, LONDON, LONDON_OPEN)),
        ("overlap", _latest_daily_boundary(now, NEW_YORK, NY_OPEN)),
        ("newyork", _latest_daily_boundary(now, LONDON, LONDON_CLOSE)),
        ("off", _latest_daily_boundary(now, NEW_YORK, NY_CLOSE)),
    ]
    session = max(boundaries, key=lambda kv: kv[1])[0]
    # --- killzone: New York wall-clock windows (independent flag).
    ny_wall = now.astimezone(NEW_YORK).time()
    killzone: str | None = None
    if LONDON_KZ[0] <= ny_wall < LONDON_KZ[1]:
        killzone = "london_kz"
    elif NY_KZ[0] <= ny_wall < NY_KZ[1]:
        killzone = "ny_kz"
    return session, killzone


def tag_sessions(index) -> tuple[list[str], list[str | None]]:
    """Vectorised (session, killzone) labelling for a DatetimeIndex."""
    names: list[str] = []
    kzs: list[str | None] = []
    for ts in index:
        n, k = session_of(ts)
        names.append(n)
        kzs.append(k)
    return names, kzs


@dataclass(frozen=True)
class SessionMapping:
    """The session -> timeframe mapping at one instant.

    Deliberately has NO direction / bias-sign field: the session
    selects the entry clock, it never says long or short.  Frozen +
    pure-constructed, so it is safe to attach to signals / journal
    rows without aliasing.
    """

    session: str                      # asia|london|overlap|newyork|off
    killzone: str | None              # london_kz|ny_kz|None
    bias_tf: str = BIAS_TF            # H4 — structure, session-proof
    zone_tf: str = ZONE_TF            # H1 — POIs live here
    trigger_tf: str = OFF_KILLZONE_TRIGGER_TF  # M15 in KZ, M30 otherwise
    amd_phase: str | None = None      # carried through, never derived
    playbook: str | None = None       # e.g. "qm_m5" — clock override note
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def entry_clock(self) -> str:
        """The clock that may fire: the session trigger, unless the
        playbook owns its own clock (QM M5 keeps M5, QM D1 keeps D1)."""
        if self.playbook == "qm_m5":
            return "M5"
        if self.playbook == "qm_d1":
            return "D1"
        return self.trigger_tf

    @property
    def active_tfs(self) -> tuple[str, ...]:
        """The TF set live at this instant: (bias, zone, trigger)."""
        return (self.bias_tf, self.zone_tf, self.entry_clock)

    @property
    def in_killzone(self) -> bool:
        return self.killzone is not None

    def as_dict(self) -> dict:
        """JSON-safe form for journal rows / signal live_context."""
        return {
            "session": self.session,
            "killzone": self.killzone,
            "in_killzone": self.in_killzone,
            "bias_tf": self.bias_tf,
            "zone_tf": self.zone_tf,
            "trigger_tf": self.trigger_tf,
            "entry_clock": self.entry_clock,
            "active_tfs": list(self.active_tfs),
            "amd_phase": self.amd_phase,
            "playbook": self.playbook,
            "tags": list(self.tags),
        }


def mapping_for(ts, *, amd_phase: str | None = None,
                playbook: str | None = None) -> SessionMapping:
    """Session -> timeframe mapping at ``ts`` (see module docstring).

    ``amd_phase`` is carried through as a tag only.  ``playbook``
    (e.g. ``"qm_m5"``) records that the playbook keeps its own entry
    clock; the session mapping annotates, never overrides it.
    """
    session, killzone = session_of(ts)
    trigger = KILLZONE_TRIGGER_TF if killzone else \
        OFF_KILLZONE_TRIGGER_TF
    tags = [f"session:{session}"]
    if killzone:
        tags.append(f"killzone:{killzone}")
    tags.append(f"entry_clock:{trigger}")
    if amd_phase is not None:
        tags.append(f"amd:{amd_phase}")
        if amd_phase == "manipulation" and killzone:
            # Descriptive tag only — manipulation inside a killzone is
            # context, never a gate, never a veto.
            tags.append("manipulation_in_killzone")
    if playbook == "qm_m5":
        tags.append("qm_m5 keeps M5 clock; session is annotation only")
    elif playbook == "qm_d1":
        tags.append("qm_d1 keeps D1 clock; session is annotation only")
    elif playbook is not None:
        tags.append(f"playbook:{playbook}")
    return SessionMapping(
        session=session, killzone=killzone, trigger_tf=trigger,
        amd_phase=amd_phase, playbook=playbook, tags=tuple(tags),
    )


def compose(ts, *, amd_phase: str | None = None,
            playbook: str | None = None) -> dict:
    """One-call composition: session mapping + AMD phase + playbook.

    Returns the JSON-safe dict (``mapping_for(...).as_dict()``) for
    attaching to a signal's ``live_context`` or a journal row.  The
    session mapping never changes the AMD phase, never supplies a
    direction, and never gates anything — composition is annotation.
    """
    return mapping_for(ts, amd_phase=amd_phase,
                       playbook=playbook).as_dict()


def attach_session_context(signal, ts, *, amd_phase: str | None = None,
                           playbook: str | None = None):
    """Attach the session mapping to a signal's ``live_context``.

    Additive only: existing ``live_context`` keys are preserved, the
    mapping lands under ``live_context["session"]``.  Returns the
    signal (for chaining).  Never touches direction, score, entry,
    stop, or targets — annotation, not a gate.
    """
    ctx = dict(getattr(signal, "live_context", None) or {})
    ctx["session"] = compose(ts, amd_phase=amd_phase, playbook=playbook)
    signal.live_context = ctx
    return signal
