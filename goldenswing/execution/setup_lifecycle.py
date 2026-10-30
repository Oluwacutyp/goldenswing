"""Persistent setup lifecycle (owner rulings 2026-10-30 §1).

Before this module, a candidate found by a scheduled scan was
journaled and DISCARDED: the record said "observed, never routed"
and nothing anywhere remembered the setup.  The trigger path —
the only route with order authority — evaluates patterns fresh
on the closing bar, so a scan setup that did not coincide with a
brand-new pattern at fire time could never be evaluated for
execution.  On 2026-10-30 the inversion playbook found valid
shorts at 01:00 and 06:00 UTC and fvg_limit at 16:00; all three
died as observations while the market cascaded.

The contract this module owns:

- PERSISTENCE: every non-shadow roster candidate a live scan
  detects is registered here with its formation, trigger,
  invalidation and expiry facts, and survives restarts (atomic
  JSON beside the journal).  A registered setup is never
  silently dropped: it leaves the active set only through a
  journaled terminal transition (filled / rejected / expired /
  invalidated).

- LIFECYCLE: candidate_detected -> setup_armed -> trigger_fired
  -> execution_evaluated -> order_submitted -> filled, with
  rejected / expired / invalidated as terminal exits from the
  states where they apply.  Every transition is an event the
  runner journals with the setup's stable ID and parent ID.

- IDENTITY: the setup ID is candidate_lifecycle.instance_key —
  keyed on the pattern's touch timestamp, direction and level,
  invariant across rescans, restarts and frame-window slides.
  Re-detection of an active instance is a no-op, never a
  duplicate setup.

- AUTHORITY: this module grants none.  Registration is not an
  order; routing happens only when the declared trigger fires
  (the h1_trigger streaming path, same direction), and the
  routed candidate passes through the same decision flow and
  shared order path as any fresh candidate — every gate intact.
  Shadow-evaluation routes are never registered here.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Lifecycle stages (the journaled event names).
STAGE_DETECTED = "candidate_detected"
STAGE_ARMED = "setup_armed"
STAGE_TRIGGER_FIRED = "trigger_fired"
STAGE_EVALUATED = "execution_evaluated"
STAGE_SUBMITTED = "order_submitted"
STAGE_FILLED = "filled"
STAGE_REJECTED = "rejected"
STAGE_EXPIRED = "expired"
STAGE_INVALIDATED = "invalidated"

#: Stages after which a setup leaves the active set.
TERMINAL_STAGES = frozenset(
    {STAGE_FILLED, STAGE_REJECTED, STAGE_EXPIRED, STAGE_INVALIDATED})

#: Stages in which a setup can still be routed / resolved.
ACTIVE_STAGES = frozenset(
    {STAGE_DETECTED, STAGE_ARMED, STAGE_SUBMITTED})

#: The trigger that routes an armed roster setup in v1: the
#: h1_trigger streaming path fires in the setup's direction.
TRIGGER_H1_FIRE = "h1_trigger_fire"


@dataclass
class TrackedSetup:
    """One setup instance and its lifecycle facts.

    The signal fields mirror GoldenSignal so the runner can
    rebuild the exact candidate the detector emitted when the
    trigger fires — the persisted candidate is evaluated, not a
    fresh re-detection."""

    setup_id: str
    parent_setup_id: str
    playbook: str
    route: str
    direction: int
    kind: str = ""
    score: float = 0.0
    factors: list = field(default_factory=list)
    entry_zone: list = field(default_factory=list)  # [lo, hi]
    entry_ref: float = 0.0
    invalidation: float = 0.0
    targets: list = field(default_factory=list)
    atr: float = 0.0
    sweep_index: int | None = None
    live_context: dict = field(default_factory=dict)
    formation_ts: str | None = None
    touch_ts: str | None = None
    created_ts: str | None = None
    trigger_kind: str = TRIGGER_H1_FIRE
    trigger_tf: str = "H1"
    validity_bars: int = 6
    expiry_ts: str | None = None
    stage: str = STAGE_DETECTED
    alert_id: str | None = None
    ticket: int | None = None
    history: list = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.stage in ACTIVE_STAGES

    @property
    def armed(self) -> bool:
        return self.stage in (STAGE_DETECTED, STAGE_ARMED)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrackedSetup":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def sanitize_context(ctx) -> tuple[dict, int]:
    """JSON-safe subset of a signal's live_context.

    Returns (clean_dict, dropped_key_count).  Values that are
    not plain data (frames, objects) are dropped and COUNTED —
    the count rides on the setup record so a lossy rebuild is a
    journaled fact, never a silent one."""

    def clean(v):
        if v is None or isinstance(v, (bool, int, float, str)):
            return True, v
        if isinstance(v, (list, tuple)):
            out = []
            for item in v:
                ok, cv = clean(item)
                if not ok:
                    return False, None
                out.append(cv)
            return True, out
        if isinstance(v, dict):
            out = {}
            for k, item in v.items():
                ok, cv = clean(item)
                if not ok:
                    return False, None
                out[str(k)] = cv
            return True, out
        return False, None

    if not isinstance(ctx, dict):
        return {}, 0
    out: dict = {}
    dropped = 0
    for k, v in ctx.items():
        ok, cv = clean(v)
        if ok:
            out[str(k)] = cv
        else:
            dropped += 1
    return out, dropped


class SetupLifecycleStore:
    """The active + recently-terminal setups, persisted.

    Persistence is an atomic JSON write (tmp + rename) beside
    the journal, so a crash mid-write never leaves a torn file.
    A store that cannot be read raises: the runner logs the
    failure loudly and starts empty (the journal still holds
    every candidate_detected event, so the audit trail survives
    even when the live set does not)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._setups: dict[str, TrackedSetup] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            for d in raw.get("setups", []):
                s = TrackedSetup.from_dict(d)
                self._setups[s.setup_id] = s

    # -- persistence -------------------------------------------------
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"setups": [s.to_dict() for s in
                              self._setups.values()]}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self.path)

    # -- registration --------------------------------------------------
    def register(self, setup: TrackedSetup, *, ts: str) -> bool:
        """Register a newly detected setup.  Returns False (and
        changes nothing) when an ACTIVE setup with the same ID
        already exists — re-detection is the same setup, not a
        new one.  A terminal setup with the same ID is replaced
        only when the new detection is a genuinely new instance
        (a later touch at the same level is the same instance
        key by construction, so a terminal record stays as the
        audit of the first life; the caller journals the
        re-detection on the scan record itself)."""
        if setup.setup_id in self._setups:
            return False
        setup.stage = STAGE_DETECTED
        setup.history.append(
            {"stage": STAGE_DETECTED, "ts": ts, "detail": ""})
        self._setups[setup.setup_id] = setup
        self.save()
        return True

    # -- transitions ---------------------------------------------------
    def transition(self, setup_id: str, stage: str, *, ts: str,
                   detail: str = "", alert_id: str | None = None,
                   ticket: int | None = None) -> dict | None:
        """Move one setup to a new stage; returns the event dict
        for the runner to journal (None when the setup is unknown
        or already terminal — a terminal setup never moves
        again)."""
        s = self._setups.get(setup_id)
        if s is None or s.stage in TERMINAL_STAGES:
            return None
        s.stage = stage
        if alert_id is not None:
            s.alert_id = alert_id
        if ticket is not None:
            s.ticket = int(ticket)
        event = {"stage": stage, "ts": ts, "detail": detail}
        s.history.append(event)
        self.save()
        return {"setup_id": s.setup_id,
                "parent_setup_id": s.parent_setup_id,
                "playbook": s.playbook, **event}

    # -- lookups -------------------------------------------------------
    def get(self, setup_id: str) -> TrackedSetup | None:
        return self._setups.get(setup_id)

    def active(self, *, direction: int | None = None,
               armed_only: bool = False) -> list[TrackedSetup]:
        out = [s for s in self._setups.values() if s.active]
        if armed_only:
            out = [s for s in out if s.armed]
        if direction is not None:
            out = [s for s in out if int(s.direction) == int(direction)]
        out.sort(key=lambda s: (s.created_ts or "", s.setup_id))
        return out

    def by_ticket(self, ticket: int) -> TrackedSetup | None:
        for s in self._setups.values():
            if s.ticket is not None and int(s.ticket) == int(ticket):
                return s
        return None

    def by_alert_id(self, alert_id: str) -> TrackedSetup | None:
        for s in self._setups.values():
            if s.alert_id is not None and s.alert_id == alert_id:
                return s
        return None
