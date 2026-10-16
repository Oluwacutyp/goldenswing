"""Institutional narrative state machine: HTF Zone -> Liquidity Sweep
-> Displacement -> CHoCH/BOS + FVG -> Entry -> Management -> Exit.

Real traders don't trade isolated indicators or single-bar patterns;
they trade a holistic market narrative.  This machine is that
narrative, as a precise state machine.  AMD maps onto it directly:

    Accumulation  = IDLE / HTF_ZONE_APPROACH
    Manipulation  = LIQUIDITY_SWEEP (stop hunt at the HTF zone)
    Distribution  = INDUCEMENT/DISPLACEMENT -> LTF_CONFIRMATION
                    -> ENTRY -> MANAGEMENT

States (per setup, persisted, survives restarts):

- 0 IDLE: no HTF zone near.  Scan the 3-year zone DB with the
  STATE 1 freshness verifier (``live/freshness.py``): only zones
  that are truly ACTIVE arm; consumed zones log a skip.  No trades.
- 1 HTF_ZONE_APPROACH: price within 0.5 ATR of a fresh ACTIVE
  MAJOR HTF zone (MN1/W1/D1 unmitigated OB / liquidity pool,
  score >= 60; MINOR arms with an extra-confirm flag).  Arm
  the watch; POI band = zone +/- 0.5 ATR.  On entering the POI
  band -> STATE 2.  No trade yet.
- 2 LIQUIDITY_SWEEP: STATE 2 math (``live/sweep.py``) classifies
  every closed M5/M15 bar: valid/weak sweep (wick in
  [0.25, 1.5] ATR, close back inside, wick/body >= 2.0,
  liquidity cleared) vs momentum breach (2 closes beyond +
  BOS + body-dominated) vs false breakout vs ambiguous (body
  too large — wait for the next bar) vs wick-without-liquidity
  (not a sweep).  Closes beyond but unconfirmed (breach_pending)
  -> footprint score (``live/footprints.py``): >= 70 breaks out
  now, < 40 is a weak retail breakout (sweep logic, no chase),
  40-69 AMBIGUOUS waits for the decisive next-bar test
  (``live/nextbar.py``).  No sweep in 3 H1 bars -> back to
  STATE 1.
- 3 INDUCEMENT_DISPLACEMENT: STATE 3 math
  (``live/displacement.py``): displacement >= 1.5 ATR_H4 in
  <= 3 M15 bars + unmitigated FVG >= 0.25 ATR + BOS +
  body-dominated, non-chop leg; inducement taken (or flagged
  weaker, -5); retracement > 50% invalidates.  No displacement
  -> invalidate, back to STATE 1.
- 4 LTF_CONFIRMATION: STATE 4 math (``live/confirmation.py``):
  CHoCH (+10) / BOS / MSB (+15) + FVG must overlap in PRICE
  (break within FVG or 0.25 ATR of its edge) and TIME (CHoCH
  within 3 M15 bars of FVG formation) + FVG unmitigated-or-
  inverted at CHoCH time + retracement into the FVG (3 M15
  normal, 12 M15 complex pullback).  CHoCH far from FVG waits;
  stale / mitigated FVG -> back to STATE 3; missed retracement
  -> back to STATE 1, no chase.
- 5 ENTRY: STATE 5 math (``live/entry.py``): LIMIT at FVG CE
  (preferred), MARKET on confirm close (no retracement in 1
  M15 bar, price within 0.25 ATR of the FVG edge), STOP for
  the breakout path only.  SL = min/max(sweep extreme, FVG
  edge, CHoCH extreme) -/+ 0.25 ATR, in [0.5, 3.0] ATR or
  invalid.  TP1 +1R / TP2 +2R-or-opposite-zone / TP3
  +3R-or-opposite-zone, R:R >= 2 or skip.  Sizing: MAJOR 1.0x,
  MINOR 0.5x, SMT fail -20%, complex -10% (margin shield and
  one-position-per-zone are runner-owned).  Nested M5/M1
  sub-CHoCH + sub-FVG inside the M15 FVG gives a tighter
  surgical entry.  Full narrative logged.
- 6 MANAGEMENT: tick management (breakeven +1R, partials
  50%/25%/25% at +1R/+2R/+3R or HTF zones, trailing behind
  structure/FVG, time stop).  Invalidation: close back through
  the CHoCH extreme + FVG mitigated -> exit.
- 7 EXIT: TP / SL / invalidation / time stop.  Log R + reason,
  journal, back to IDLE.

BREAKOUT PATH (from APPROACH / SWEEP): close beyond the edge +
2 closes + BOS + retest holds -> switch to BREAKOUT_TREND,
restart at APPROACH with the broken edge as the new breaker
zone — never range mean-reversion.  False breakout (close
beyond, then back inside within 2 bars) -> back to SWEEP
logic, mean-reversion arms.

Persistence: the state persists to ``narrative_state.json``
(journal-adjacent, atomic tmp + fsync + os.replace, SHA-256
checksum + version).  A corrupt file rebuilds as fresh IDLE
with a WARNING — counts are never fabricated.  A missing file
is a first boot.  The runner advances the machine on every
TF-close refresh, before any order decision on the new bars.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .atr_regime import (atr_regime, baseline_atr,
                          scaled_thresholds, scaled_time_stop)
from .confirmation import evaluate_confirmation
from .displacement import evaluate_displacement
from .entry import evaluate_entry
from .exit import build_journal_entry, normalize_exit_reason, realized_r
from .footprints import score_breakout
from .freshness import evaluate_zone_freshness, scan_state1
from .management import (check_invalidation, evaluate_management,
                         plan_management)
from .multitf import atr14, find_swings
from .nextbar import NextBarDecision, decide_m5_pair, decide_next_bar, is_high_vol
from .opposing import (adjust_tps, apply_tier_to_management,
                          opposing_zone_for, risk_tier)
from .patterns import session_label
from .styles import derive_style
from .sweep import classify_sweep
from .zones import ZoneStore as RealZoneStore
from .zones import major_zones as real_major_zones

IDLE = 0
HTF_ZONE_APPROACH = 1
LIQUIDITY_SWEEP = 2
INDUCEMENT_DISPLACEMENT = 3
LTF_CONFIRMATION = 4
ENTRY = 5
MANAGEMENT = 6
EXIT = 7

STATE_NAMES = {
    IDLE: "IDLE",
    HTF_ZONE_APPROACH: "HTF_ZONE_APPROACH",
    LIQUIDITY_SWEEP: "LIQUIDITY_SWEEP",
    INDUCEMENT_DISPLACEMENT: "INDUCEMENT_DISPLACEMENT",
    LTF_CONFIRMATION: "LTF_CONFIRMATION",
    ENTRY: "ENTRY",
    MANAGEMENT: "MANAGEMENT",
    EXIT: "EXIT",
}

STATE_VERSION = 1
APPROACH_TOL_ATR = 0.5
IDLE_SCAN_ATR = 2.0
SWEEP_MAX_AGE_M15 = 12  # 3 H1 bars of M15 bars without a sweep
DISPLACEMENT_MIN_ATR = 1.5


@dataclass
class NarrativeState:
    """One persisted snapshot of the narrative machine."""

    state: int = IDLE
    direction: int = 0  # +1 long | -1 short | 0 none
    regime: str = "unknown"  # trend | range | breakout_trend | hold
    regime_breakout: bool = False
    zone: dict = field(default_factory=dict)
    sweep: dict = field(default_factory=dict)
    displacement: dict = field(default_factory=dict)
    confirm: dict = field(default_factory=dict)
    entry: dict = field(default_factory=dict)
    bars_in_state: int = 0
    invalidation: float | None = None
    exit_reason: str = ""
    exit_r: float | None = None
    narrative: list = field(default_factory=list)
    last_bar_ts: str | None = None
    version: int = STATE_VERSION
    checksum: str = ""
    # STATE 1-5 bookkeeping.
    band_entry_ts: str | None = None
    sweep_confirmed: bool = False
    stronger_confirm: bool = False
    complex_pullback: bool = False
    freshness: dict = field(default_factory=dict)
    pending_ambiguous: dict = field(default_factory=dict)
    # Hierarchical decisive test: a second-bar institutional
    # confirm is a LATE CONFIRM (0.5x size, differentiated
    # management).  Never a vote — one decisive condition.
    late_confirm: bool = False
    high_vol: bool = False
    management: dict = field(default_factory=dict)
    # Opposing zone + risk tier + ATR regime (for /status,
    # management, and the TP adjustment).
    opposing: dict = field(default_factory=dict)
    entry_tier: str = ""
    opposing_atr: float = 0.0
    tp_adjustment: dict = field(default_factory=dict)
    atr_ratio: float = 0.0
    atr_regime: str = "NORMAL"
    atr_value: float = 0.0
    # /status desk: regime + state lineage (filled by the
    # runner / regime tracker, persisted with the state).
    regime_since: str | None = None
    regime_lock_remaining_h1: int = 0
    breakout_confirm_count: int = 0
    regime_state_file: str | None = None
    regime_state_last_saved: str | None = None
    regime_state_version: int = 0
    state_since: str | None = None
    state_narrative: str = ""
    # STATE 7: journal entry of the last exited setup (kept
    # across the reset to IDLE so /status + the runner can
    # report the exit after the machine has re-armed).
    last_exit_journal: dict = field(default_factory=dict)

    def log(self, msg: str) -> None:
        self.narrative.append(msg)
        if len(self.narrative) > 50:
            self.narrative = self.narrative[-50:]

    def payload(self) -> dict:
        return {
            "version": STATE_VERSION, "state": int(self.state),
            "direction": int(self.direction), "regime": self.regime,
            "regime_breakout": bool(self.regime_breakout),
            "zone": self.zone, "sweep": self.sweep,
            "displacement": self.displacement,
            "confirm": self.confirm, "entry": self.entry,
            "bars_in_state": int(self.bars_in_state),
            "invalidation": self.invalidation,
            "exit_reason": self.exit_reason, "exit_r": self.exit_r,
            "narrative": self.narrative,
            "last_bar_ts": self.last_bar_ts,
            "band_entry_ts": self.band_entry_ts,
            "sweep_confirmed": bool(self.sweep_confirmed),
            "stronger_confirm": bool(self.stronger_confirm),
            "complex_pullback": bool(self.complex_pullback),
            "freshness": self.freshness,
            "pending_ambiguous": self.pending_ambiguous,
            "late_confirm": bool(self.late_confirm),
            "high_vol": bool(self.high_vol),
            "management": self.management,
            "opposing": self.opposing,
            "entry_tier": self.entry_tier,
            "opposing_atr": self.opposing_atr,
            "tp_adjustment": self.tp_adjustment,
            "atr_ratio": self.atr_ratio,
            "atr_regime": self.atr_regime,
            "atr_value": self.atr_value,
            "regime_since": self.regime_since,
            "regime_lock_remaining_h1":
                self.regime_lock_remaining_h1,
            "breakout_confirm_count": self.breakout_confirm_count,
            "regime_state_file": self.regime_state_file,
            "regime_state_last_saved":
                self.regime_state_last_saved,
            "regime_state_version": self.regime_state_version,
            "state_since": self.state_since,
            "state_narrative": self.state_narrative,
            "last_exit_journal": self.last_exit_journal,
        }

    @classmethod
    def from_payload(cls, d: dict) -> "NarrativeState":
        st = cls(
            state=int(d.get("state", IDLE)),
            direction=int(d.get("direction", 0)),
            regime=str(d.get("regime", "unknown")),
            regime_breakout=bool(d.get("regime_breakout", False)),
            zone=dict(d.get("zone") or {}),
            sweep=dict(d.get("sweep") or {}),
            displacement=dict(d.get("displacement") or {}),
            confirm=dict(d.get("confirm") or {}),
            entry=dict(d.get("entry") or {}),
            bars_in_state=int(d.get("bars_in_state", 0)),
            invalidation=d.get("invalidation"),
            exit_reason=str(d.get("exit_reason", "")),
            exit_r=d.get("exit_r"),
            narrative=list(d.get("narrative") or []),
            last_bar_ts=d.get("last_bar_ts"),
            version=int(d.get("version", STATE_VERSION)))
        st.band_entry_ts = d.get("band_entry_ts")
        st.sweep_confirmed = bool(d.get("sweep_confirmed", False))
        st.stronger_confirm = bool(d.get("stronger_confirm", False))
        st.complex_pullback = bool(d.get("complex_pullback", False))
        st.freshness = dict(d.get("freshness") or {})
        st.pending_ambiguous = dict(d.get("pending_ambiguous") or {})
        st.late_confirm = bool(d.get("late_confirm", False))
        st.high_vol = bool(d.get("high_vol", False))
        st.management = dict(d.get("management") or {})
        st.opposing = dict(d.get("opposing") or {})
        st.entry_tier = str(d.get("entry_tier", ""))
        st.opposing_atr = float(d.get("opposing_atr", 0.0) or 0.0)
        st.tp_adjustment = dict(d.get("tp_adjustment") or {})
        st.atr_ratio = float(d.get("atr_ratio", 0.0) or 0.0)
        st.atr_regime = str(d.get("atr_regime", "NORMAL"))
        st.atr_value = float(d.get("atr_value", 0.0) or 0.0)
        st.regime_since = d.get("regime_since")
        st.regime_lock_remaining_h1 = int(
            d.get("regime_lock_remaining_h1", 0) or 0)
        st.breakout_confirm_count = int(
            d.get("breakout_confirm_count", 0) or 0)
        st.regime_state_file = d.get("regime_state_file")
        st.regime_state_last_saved = d.get("regime_state_last_saved")
        st.regime_state_version = int(
            d.get("regime_state_version", 0) or 0)
        st.state_since = d.get("state_since")
        st.state_narrative = str(d.get("state_narrative", ""))
        st.last_exit_journal = dict(d.get("last_exit_journal") or {})
        return st

    def describe(self) -> str:
        name = STATE_NAMES.get(self.state, "?")
        if self.state == IDLE:
            return f"narrative: IDLE — {self.exit_reason or 'scanning'}"
        zone = self.zone.get("level")
        z = f" zone {zone:.1f}" if zone else ""
        d = "LONG" if self.direction > 0 else (
            "SHORT" if self.direction < 0 else "")
        extra = ""
        if self.state == ENTRY and self.entry:
            extra = (f" entry {self.entry.get('price')} "
                     f"stop {self.entry.get('stop')} "
                     f"TP {self.entry.get('targets')}")
        return (f"narrative: {name} {d}{z} "
                f"({self.bars_in_state} bars){extra}").strip()


def _checksum(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def save_narrative_state(path: str | Path,
                         state: NarrativeState) -> None:
    """Atomic persist: tmp + fsync + os.replace + dir fsync."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = state.payload()
    payload["checksum"] = _checksum(payload)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    try:
        dfd = os.open(str(p.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass


def load_narrative_state(path: str | Path
                         ) -> tuple[NarrativeState | None, str, str]:
    """Load persisted state.

    Returns (state, status, msg): status ``ok`` / ``missing`` /
    ``corrupt``.  Corrupt -> caller rebuilds as fresh IDLE with a
    WARNING; counts are never fabricated from a torn file.
    """
    p = Path(path)
    if not p.exists():
        return None, "missing", "no narrative state file yet"
    try:
        with open(p, encoding="utf-8") as fh:
            payload = json.load(fh)
        want = payload.pop("checksum", "")
        if payload.get("version") != STATE_VERSION:
            return None, "corrupt", "version mismatch"
        if _checksum(payload) != want:
            return None, "corrupt", "checksum mismatch"
        return NarrativeState.from_payload(payload), "ok", "restored"
    except Exception as exc:
        return None, "corrupt", f"unreadable: {exc}"


# ------------------------------------------------------------------ helpers

def _frames_for(frames: dict) -> dict:
    return {k: v for k, v in (frames or {}).items()
            if v is not None and len(v)}


def _entry_frame(frames: dict):
    for tf in ("M15", "M5", "H1"):
        if frames.get(tf) is not None and len(frames[tf]):
            return tf, frames[tf]
    return None, None


def _htf_atr(frames: dict, entry) -> float:
    """STATE 1-5 math uses HTF ATR (H4), never M15 ATR."""
    try:
        h4 = frames.get("H4")
        if h4 is not None and len(h4):
            return float(atr14(h4))
    except Exception:
        pass
    try:
        return float(atr14(entry))
    except Exception:
        return 0.0


def _major_zones(frames: dict) -> list:
    """MAJOR active zones from the REAL 3-year store
    (``live/zones.py`` ZoneStore, in-memory for the snapshot)."""
    htf = {tf: frames[tf] for tf in ("MN1", "W1", "D1", "H4")
           if frames.get(tf) is not None and len(frames[tf])}
    if not htf:
        return []
    store = RealZoneStore(":memory:")
    try:
        store.seed(htf)
        return real_major_zones(store.active_zones())
    finally:
        store.close()


def _nearest_zone(zones: list, price: float):
    if not zones:
        return None
    return min(zones, key=lambda z: abs(z.level - price))


def _breakout(frame: pd.DataFrame, edge: float,
              direction: int) -> dict | None:
    """Breakout = 2 consecutive closes beyond the edge + BOS.

    Returns breakout evidence or None.  A single close beyond
    is not a breakout (that is the false-breakout trap).
    """
    if frame is None or len(frame) < 4:
        return None
    c = frame["close"].to_numpy(dtype=float)
    if direction > 0:
        if c[-1] > edge and c[-2] > edge:
            return {"present": True, "detail": "2 closes above edge",
                    "closes": [float(c[-2]), float(c[-1])]}
    else:
        if c[-1] < edge and c[-2] < edge:
            return {"present": True, "detail": "2 closes below edge",
                    "closes": [float(c[-2]), float(c[-1])]}
    return None


def _zone_alive(zone: dict, zones: list) -> bool:
    zid = zone.get("zone_id")
    return any(z.zone_id == zid and z.status == "active"
               for z in zones) if zid else True


def _fresh_start(regime: str) -> NarrativeState:
    """Fresh IDLE state, keeping only the regime."""
    return NarrativeState(regime=regime)


def _bars_since_m15(entry_frame, ts: str | None) -> int:
    """Closed M15 bars with ts strictly after ``ts``."""
    if ts is None or entry_frame is None or len(entry_frame) == 0:
        return 0
    try:
        cut = pd.Timestamp(ts)
        return int((entry_frame.index > cut).sum())
    except Exception:
        return 0


def _high_vol(frames: dict, atr: float) -> bool:
    """High-volatility gate for the M5 fast path: the Dynamic
    ATR Regime (``live/atr_regime.py``) says HIGH/EXTREME
    (current ATR > 1.5x its 100-bar median baseline), or the
    NY open killzone."""
    try:
        reg = atr_regime(atr, baseline_atr(frames.get("H4")))
        if reg["regime"] in ("HIGH", "EXTREME"):
            return True
        killzone = None
        try:
            killzone = session_label(frames["M15"].index[-1]
                                     if frames.get("M15") is not None
                                     else None).get("killzone")
        except Exception:
            killzone = None
        return is_high_vol(atr, 0.0, session_killzone=killzone)
    except Exception:
        return False


def _bar_dict(frame, pos: int) -> dict:
    """A closed bar as the decisive-test dict (with a volume
    ratio vs the prior 20 bars when volume is available)."""
    row = frame.iloc[pos]
    vol_ratio = None
    if "volume" in frame.columns and pos > 0:
        try:
            vols = frame["volume"].to_numpy(dtype=float)
            mean_v = float(np.mean(
                vols[max(0, pos - 20):pos]))
            if mean_v > 0:
                vol_ratio = float(row["volume"]) / mean_v
        except Exception:
            vol_ratio = None
    return {"open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume_ratio": vol_ratio}


# ------------------------------------------------------------------ advance

def advance(state: NarrativeState, frames: dict,
            *, regime: str = "unknown") -> NarrativeState:
    """Advance the machine one refresh over CLOSED bars.

    Loops ``_advance_once`` while the state keeps changing (up to
    8 transitions): a refresh may legitimately progress several
    stages when the evidence already accumulated in the bars
    (snapshot use).  The runner calls it once per TF-close
    refresh with the full stack.  Pure: no I/O, no fabricated
    counts — every transition names its evidence.
    """
    for _ in range(8):
        before = state.state
        state = _advance_once(state, frames, regime=regime)
        if state.state == before:
            break
    return state


def _advance_once(state: NarrativeState, frames: dict,
                  *, regime: str = "unknown") -> NarrativeState:
    frames = _frames_for(frames)
    if not frames:
        return state
    entry_tf, entry = _entry_frame(frames)
    if entry is None:
        return state
    price = float(entry["close"].iloc[-1])
    atr = _htf_atr(frames, entry)
    if atr <= 0:
        return state
    zones = _major_zones(frames)
    # Dynamic ATR Regime (live/atr_regime.py): all ATR-based
    # thresholds scale with CURRENT ATR.  HIGH/EXTREME raises
    # the displacement bar (1.75x), the volume bar (2.0x), and
    # switches confirmation to the M5 fast path.
    atr_info = atr_regime(atr, baseline_atr(frames.get("H4")))
    atr_thresh = scaled_thresholds(atr_info["regime"])
    if atr_info["regime"] != state.atr_regime:
        state.log(atr_info["log"])
    state.atr_ratio = float(atr_info["ratio"])
    state.atr_regime = str(atr_info["regime"])
    state.atr_value = float(atr)
    state.bars_in_state += 1
    try:
        ts = str(entry.index[-1])
    except Exception:
        ts = None
    state.last_bar_ts = ts
    if regime != "unknown":
        state.regime = regime

    # EXIT is terminal: the next advance re-scans from IDLE.
    # Safety net for states that entered EXIT via mark_exited
    # / invalidation / time-stop without finalize_exit: keep
    # the exit reason/R and any journal entry already made.
    if state.state == EXIT:
        _journal = dict(state.last_exit_journal or {})
        state = NarrativeState(regime=state.regime,
                               exit_reason=state.exit_reason,
                               exit_r=state.exit_r)
        state.last_exit_journal = _journal
        state.log("narrative reset to IDLE after EXIT")

    # ---- STATE 0 IDLE: STATE 1 freshness scan. ----
    if state.state == IDLE:
        scan = scan_state1(zones, frames, price, atr,
                           radius_atr=APPROACH_TOL_ATR)
        if scan.armed is not None:
            v = scan.armed
            zone = next((z for z in zones
                         if z.tf == v.zone_tf and abs(
                             z.level - v.zone_level) < 1e-9), None)
            if zone is None:
                return state
            state.state = HTF_ZONE_APPROACH
            state.direction = 1 if zone.side == "support" else -1
            state.zone = {**zone.to_dict(),
                          "zone_id": zone.zone_id}
            state.freshness = asdict(v)
            state.stronger_confirm = v.needs_stronger_confirm
            state.bars_in_state = 0
            state.log(v.log)
        else:
            for line in scan.skipped:
                state.log(line)
            near = _nearest_zone(zones, price)
            if near is not None:
                state.log(f"IDLE: nearest MAJOR zone "
                          f"{near.level:.1f} "
                          f"({abs(near.level - price) / atr:.1f} ATR "
                          f"away)")
        return state

    # ---- STATE 1 HTF_ZONE_APPROACH ----
    if state.state == HTF_ZONE_APPROACH:
        zone = state.zone
        # Re-verify freshness every refresh: a zone that
        # consumed since arming never arms a sweep.
        zobj = next((z for z in zones
                     if z.zone_id == zone.get("zone_id")), None)
        if zobj is not None:
            v = evaluate_zone_freshness(
                zobj, frames.get(zobj.tf), atr,
                m15=frames.get("M15"), others=zones)
            if not v.arms:
                state.log(f"HTF zone no longer fresh: {v.log} "
                          "— back to IDLE")
                return _fresh_start(state.regime)
            state.freshness = asdict(v)
            state.stronger_confirm = v.needs_stronger_confirm
        elif zones and not _zone_alive(zone, zones):
            state.log("HTF zone mitigated/broken before sweep — "
                      "back to IDLE")
            return _fresh_start(state.regime)
        level = float(zone.get("level", price))
        # Breakout check first: a close beyond + 2 closes is a
        # breakout, not a sweep (the two paths diverge here).
        edge = float(zone.get("top", level)) \
            if state.direction < 0 else float(zone.get("bottom", level))
        bo = _breakout(entry, edge, -state.direction)
        if bo is not None:
            # First-bar confirm: clean, not late.
            state.late_confirm = False
            state.high_vol = _high_vol(frames, atr)
            state.regime_breakout = True
            state.regime = "breakout_trend"
            state.direction = -state.direction
            state.zone = {**zone, "kind": "breaker",
                          "level": edge, "top": edge + 0.5 * atr,
                          "bottom": edge - 0.5 * atr}
            state.bars_in_state = 0
            state.log(f"BREAKOUT: 2 closes beyond edge {edge:.1f} "
                      "-> BREAKOUT_TREND, broken edge is the new "
                      "breaker zone")
            return state
        if abs(price - level) <= APPROACH_TOL_ATR * atr:
            state.state = LIQUIDITY_SWEEP
            state.band_entry_ts = ts
            state.bars_in_state = 0
            state.log(f"STATE 1 -> STATE 2: entered POI band "
                      f"{zone.get('bottom', level):.1f}-"
                      f"{zone.get('top', level):.1f}")
        return state

    # ---- STATE 2 LIQUIDITY_SWEEP ----
    if state.state == LIQUIDITY_SWEEP:
        if not state.sweep_confirmed:
            # Pending ambiguous breakout: the decisive test
            # (live/nextbar.py, hierarchical, never a vote).
            # Fast path (high volatility): the next 2 M5 bars
            # decide, max 10 min.  Normal path: the next M15
            # bar decides, max 2 bars total.
            pend = state.pending_ambiguous
            if isinstance(pend, dict) and \
                    pend.get("kind") == "pending_breakout":
                attempt_end = int(pend.get("attempt_end_pos", -1))
                test_kwargs = dict(
                    edge=float(pend["edge"]),
                    breakout_dir=int(pend["breakout_dir"]),
                    atr=atr,
                    zone_low=float(state.zone.get(
                        "bottom", pend["edge"])),
                    zone_high=float(state.zone.get(
                        "top", pend["edge"])),
                    swing_level=pend.get("swing_level"),
                    fvg=pend.get("fvg"),
                    waits_used=int(pend.get("waits_used", 0)))
                verdict = None
                nb_close = None
                if pend.get("fast_path"):
                    m5 = frames.get("M5")
                    m5_end = int(pend.get("m5_attempt_end", -1))
                    if m5 is not None and len(m5) - 1 >= m5_end + 2:
                        pair = decide_m5_pair(
                            pend, _bar_dict(m5, len(m5) - 2),
                            _bar_dict(m5, len(m5) - 1),
                            **test_kwargs)
                        if pair.decision in ("institutional",
                                             "retail"):
                            verdict = pair
                            nb_close = float(m5["close"].iloc[-1])
                        else:
                            # Fast path: M5 disagreement = no
                            # decisive outcome, max 2 M5 bars
                            # spent -> invalidate.
                            verdict = NextBarDecision(
                                decision="invalidate",
                                reason="m5_disagreement_fast",
                                log=("M5 bars disagree, max 2 M5 "
                                     "bars, invalidated, next "
                                     "setup — back to STATE 1"))
                    else:
                        return state
                elif len(entry) - 1 > attempt_end:
                    nb_close = float(
                        entry["close"].iloc[attempt_end + 1])
                    verdict = decide_next_bar(
                        pend, _bar_dict(entry, attempt_end + 1),
                        **test_kwargs)
                else:
                    return state
                if verdict is None:
                    return state
                state.log("STATE 2 next-bar decisive test: "
                          + verdict.log)
                if verdict.decision == "institutional":
                    # Second-bar confirm = LATE CONFIRM: 0.5x
                    # size, differentiated management.
                    state.late_confirm = \
                        int(pend.get("waits_used", 0)) >= 1
                    state.high_vol = bool(pend.get("fast_path",
                                                   False))
                    if state.late_confirm:
                        state.log("late confirm, 0.5x size")
                    state.regime_breakout = True
                    state.regime = "breakout_trend"
                    state.direction = -state.direction
                    state.zone = {
                        **state.zone, "kind": "breaker",
                        "level": float(pend["edge"]),
                        "top": float(pend["edge"]) + 0.5 * atr,
                        "bottom": float(pend["edge"])
                        - 0.5 * atr}
                    state.sweep = {}
                    state.sweep_confirmed = False
                    state.pending_ambiguous = {}
                    state.bars_in_state = 0
                    state.state = HTF_ZONE_APPROACH
                    return state
                if verdict.decision == "retail":
                    # Close back inside = sweep.  The sweep
                    # extreme is the attempt's extreme; the
                    # displacement leg starts from it.  Return
                    # now: the advance() loop re-enters and
                    # runs the displacement check with the
                    # sweep already confirmed (no
                    # re-classification).
                    state.sweep = {
                        "bar_pos": attempt_end,
                        "wick": float(pend["sweep_wick"]),
                        "edge": float(pend["edge"]),
                        "close": nb_close}
                    state.sweep_confirmed = True
                    state.pending_ambiguous = {}
                    state.bars_in_state = 0
                    state.log("STATE 2: next-bar decisive "
                              "test retail -> sweep confirmed")
                    return state
                elif verdict.decision == "wait":
                    state.pending_ambiguous = {
                        **pend,
                        "attempt_end_pos": attempt_end + 1,
                        "waits_used":
                            int(pend.get("waits_used", 0)) + 1}
                    return state
                else:  # invalidate
                    state.log("STATE 2: next-bar decisive "
                              "test invalidated — back to "
                              "STATE 1")
                    state.pending_ambiguous = {}
                    state.state = HTF_ZONE_APPROACH
                    state.bars_in_state = 0
                    return state
            # Stale: no sweep within 3 H1 bars of band entry.
            if _bars_since_m15(entry, state.band_entry_ts) \
                    > SWEEP_MAX_AGE_M15:
                state.log("no sweep in 3 H1 bars (zone still "
                          "active) — back to STATE 1")
                state.state = HTF_ZONE_APPROACH
                state.band_entry_ts = None
                state.bars_in_state = 0
                return state
            sv = classify_sweep(entry, state.zone,
                                direction=state.direction, atr=atr)
            if sv.kind in ("valid_sweep", "weak_sweep"):
                state.sweep = asdict(sv)
                state.sweep_confirmed = True
                state.pending_ambiguous = {}
                state.bars_in_state = 0
                state.log("STATE 2 sweep: " + sv.log)
            elif sv.kind == "breach":
                # First-bar confirm: clean, not late.
                state.late_confirm = False
                state.high_vol = _high_vol(frames, atr)
                state.regime_breakout = True
                state.regime = "breakout_trend"
                state.direction = -state.direction
                state.zone = {**state.zone, "kind": "breaker",
                              "level": sv.edge,
                              "top": sv.edge + 0.5 * atr,
                              "bottom": sv.edge - 0.5 * atr}
                state.sweep = {}
                state.sweep_confirmed = False
                state.bars_in_state = 0
                state.log("STATE 2 -> BREAKOUT: " + sv.log)
                state.state = HTF_ZONE_APPROACH
                return state
            elif sv.kind == "breach_pending":
                # Closes beyond, but BOS/body/displacement not
                # all confirmed: let the footprint score route
                # it — institutional (>=70) breaks out now,
                # retail (<40) is a weak breakout (stay in sweep
                # logic), ambiguous (40-69) waits for the
                # decisive next-bar test (live/nextbar.py).
                fs = score_breakout(entry, sv.edge,
                                    -state.direction, atr)
                state.log("STATE 2: " + sv.log + f" | {fs.log}")
                if fs.total >= 70:
                    # First-bar confirm: clean, not late.
                    state.late_confirm = False
                    state.high_vol = _high_vol(frames, atr)
                    state.regime_breakout = True
                    state.regime = "breakout_trend"
                    state.direction = -state.direction
                    state.zone = {**state.zone, "kind": "breaker",
                                  "level": sv.edge,
                                  "top": sv.edge + 0.5 * atr,
                                  "bottom": sv.edge - 0.5 * atr}
                    state.sweep = {}
                    state.sweep_confirmed = False
                    state.bars_in_state = 0
                    state.log("STATE 2 -> BREAKOUT (footprint "
                              f"score {fs.total:.0f}): " + sv.log)
                    state.state = HTF_ZONE_APPROACH
                    return state
                if fs.total < 40:
                    state.log("STATE 2: weak retail breakout "
                              "(footprint score "
                              f"{fs.total:.0f}) — sweep logic, "
                              "no chase")
                    return state
                # Ambiguous: store the attempt for the
                # next-bar decisive test on the next advance.
                # High volatility -> M5 fast path (next 2 M5
                # bars decide, max 10 min), never 2 M15.
                fast = _high_vol(frames, atr)
                m5_end = -1
                if fast and frames.get("M5") is not None:
                    m5_end = len(frames["M5"]) - 1
                swing_lvl = None
                if fs.attempt_start is not None and \
                        fs.attempt_start > 2 * 3 + 5:
                    sub = entry.iloc[:fs.attempt_start]
                    try:
                        h_, l_ = find_swings(
                            sub["high"].to_numpy(dtype=float),
                            sub["low"].to_numpy(dtype=float), 3)
                        sw = h_ if -state.direction > 0 else l_
                        if sw:
                            swing_lvl = float(sw[-1][1])
                    except Exception:
                        swing_lvl = None
                attempt = entry.iloc[
                    fs.attempt_start:fs.attempt_end + 1] \
                    if fs.attempt_start is not None else entry
                if -state.direction > 0:
                    sweep_wick = float(attempt["high"].max())
                else:
                    sweep_wick = float(attempt["low"].min())
                state.pending_ambiguous = {
                    "kind": "pending_breakout",
                    "score": fs.total, "edge": sv.edge,
                    "breakout_dir": -state.direction,
                    "attempt_end_pos": fs.attempt_end,
                    "attempt_start_pos": fs.attempt_start,
                    "sweep_wick": sweep_wick,
                    "swing_level": swing_lvl, "fvg": fs.fvg,
                    "waits_used": 0, "fast_path": fast,
                    "m5_attempt_end": m5_end}
                state.log(f"STATE 2: ambiguous breakout, score "
                          f"{fs.total:.0f}, awaiting next-bar "
                          "decisive test"
                          + (" — high-volatility fast path, M5 "
                             "decisive test, max 10 min"
                             if fast else ""))
                return state
            elif sv.kind == "false_breakout":
                state.log("STATE 2: " + sv.log)
                return state
            elif sv.kind == "wick_without_liquidity":
                state.log("STATE 2: " + sv.log
                          + " — back to STATE 1")
                state.state = HTF_ZONE_APPROACH
                state.bars_in_state = 0
                return state
            elif sv.kind in ("ambiguous", "ambiguous_pending"):
                state.pending_ambiguous = asdict(sv)
                state.log("STATE 2: " + sv.log)
                return state
            else:
                return state
        # Sweep confirmed -> displacement check (Dynamic ATR
        # Regime thresholds: HIGH raises the bar).
        disp = evaluate_displacement(
            entry, sweep=state.sweep, direction=state.direction,
            atr=atr, m5_frame=frames.get("M5"),
            disp_min_atr=atr_thresh["disp_min_atr"],
            volume_spike_min=atr_thresh["volume_spike_min"])
        if disp.valid:
            state.state = INDUCEMENT_DISPLACEMENT
            state.displacement = asdict(disp)
            state.bars_in_state = 0
            state.log("STATE 2 -> STATE 3: " + disp.log)
            return state
        if disp.complete or disp.hard:
            state.log("STATE 2: " + disp.log + " — back to STATE 1")
            state.state = HTF_ZONE_APPROACH
            state.sweep = {}
            state.sweep_confirmed = False
            state.displacement = {}
            state.bars_in_state = 0
            return state
        # Awaiting more bars; the sweep itself ages out after
        # 12 M15 bars without displacement.
        sweep_pos = int(state.sweep.get("bar_pos", -1))
        if sweep_pos >= 0 and (len(entry) - 1 - sweep_pos) \
                > SWEEP_MAX_AGE_M15:
            state.log("sweep without displacement (aged out) — "
                      "back to STATE 1")
            state.state = HTF_ZONE_APPROACH
            state.sweep = {}
            state.sweep_confirmed = False
            state.displacement = {}
            state.bars_in_state = 0
        return state

    # ---- STATE 3 INDUCEMENT_DISPLACEMENT ----
    if state.state == INDUCEMENT_DISPLACEMENT:
        disp = evaluate_displacement(
            entry, sweep=state.sweep, direction=state.direction,
            atr=atr, m5_frame=frames.get("M5"),
            disp_min_atr=atr_thresh["disp_min_atr"],
            volume_spike_min=atr_thresh["volume_spike_min"])
        if disp.valid:
            state.displacement = asdict(disp)
            if disp.inducement_taken is False:
                state.stronger_confirm = True
                state.log("inducement not taken, weaker — needs "
                          "stronger FVG confirm")
            if disp.strength == "weak":
                state.stronger_confirm = True
            state.state = LTF_CONFIRMATION
            state.bars_in_state = 0
            state.log("STATE 3 -> STATE 4: " + disp.log)
            return state
        if disp.complete or disp.hard:
            state.log("STATE 3: " + disp.log + " — back to STATE 1")
            state.state = HTF_ZONE_APPROACH
            state.sweep = {}
            state.sweep_confirmed = False
            state.displacement = {}
            state.bars_in_state = 0
            return state
        return state

    # ---- STATE 4 LTF_CONFIRMATION ----
    if state.state == LTF_CONFIRMATION:
        conf = evaluate_confirmation(
            entry, displacement=state.displacement,
            direction=state.direction, atr=atr)
        state.confirm = asdict(conf)
        if conf.decision in ("confirm", "wait_retracement",
                             "complex_pullback"):
            if conf.decision == "complex_pullback":
                state.complex_pullback = True
            # Opposing zone + risk tier (never gates — the tier
            # decides size/management, the opposing zone moves
            # TPs).  The entry zone tier is computed from the
            # HTF zone itself (score / touches / displacement /
            # TF / volume).
            entry_tier = risk_tier(state.zone or {})
            plan = evaluate_entry(
                entry, confirmation=conf,
                displacement=state.displacement,
                sweep=state.sweep, direction=state.direction,
                atr=atr, zone=state.zone,
                opposite_levels=None,
                m5_frame=frames.get("M5"),
                m1_frame=frames.get("M1"),
                dxy_frame=frames.get("DXY"),
                silver_frame=frames.get("SILVER"),
                late_confirm=bool(state.late_confirm),
                risk_tier=entry_tier)
            if plan.valid:
                # Opposing-zone TP adjustment on the entry plan
                # (management recomputes it for its own ladder
                # in mark_entered).  The opposing zone is the
                # nearest ACTIVE MAJOR zone beyond the entry
                # in the trade direction.
                opp = opposing_zone_for(
                    zones, entry_price=plan.entry_price,
                    direction=state.direction)
                tp_adj = {}
                if opp:
                    tp_adj = adjust_tps(
                        entry_price=plan.entry_price,
                        stop=plan.stop,
                        direction=state.direction,
                        opposing=opp, atr=atr,
                        tp1=plan.tp1, tp2=plan.tp2, tp3=plan.tp3,
                        tp1_pct=0.50, tp2_pct=0.25, tp3_pct=0.25)
                    if tp_adj.get("tp2") is not None:
                        plan.tp2 = float(tp_adj["tp2"])
                    if tp_adj.get("tp3") is not None:
                        plan.tp3 = float(tp_adj["tp3"])
                    else:
                        plan.tp3 = None
                    plan.rr_tp2 = abs(plan.tp2 - plan.entry_price) \
                        / abs(plan.entry_price - plan.stop) \
                        if plan.entry_price != plan.stop else 0.0
                state.opposing = dict(opp or {})
                state.entry_tier = entry_tier
                state.opposing_atr = float(
                    (float(opp["level"]) - plan.entry_price)
                    * state.direction / atr
                    if opp and atr > 0 else 0.0)
                state.tp_adjustment = dict(tp_adj or {})
                state.state = ENTRY
                state.entry = {
                    "price": plan.entry_price, "stop": plan.stop,
                    "targets": [plan.tp1, plan.tp2, plan.tp3],
                    "entry_type": plan.entry_type,
                    "plan": asdict(plan),
                    "invalidation": conf.choch_extreme}
                state.invalidation = conf.choch_extreme
                state.bars_in_state = 0
                state.log(plan.log)
                if tp_adj:
                    state.log(tp_adj.get("log", ""))
                state.log(f"Zone tier: {entry_tier}")
                # Unified fractal: the style is DERIVED from the
                # HTF zone TF + trigger TF + target distance —
                # never hardcoded per playbook.  One engine, all
                # horizons.
                style = derive_style(
                    str(state.zone.get("tf", "")), entry_tf,
                    abs(plan.tp2 - plan.entry_price) / atr
                    if atr > 0 else None)
                state.log(f"style={style} (zone "
                          f"{state.zone.get('tf')} -> trigger "
                          f"{entry_tf})")
            else:
                state.log(f"STATE 5 entry invalid: "
                          f"{plan.invalid_reason} — back to STATE 1")
                return _fresh_start(state.regime)
            return state
        if conf.decision in ("wait_fvg", "wait_structure"):
            state.log("STATE 4: " + conf.log)
            return state
        if conf.decision in ("stale", "mitigated"):
            state.log("STATE 4: " + conf.log
                      + " — back to STATE 3 (await new FVG)")
            state.state = INDUCEMENT_DISPLACEMENT
            state.confirm = {}
            state.bars_in_state = 0
            return state
        # missed / invalid -> back to STATE 1, never chase.
        state.log("STATE 4: " + conf.log + " — back to STATE 1")
        return _fresh_start(state.regime)

    # ---- STATE 5 ENTRY ----
    if state.state == ENTRY:
        # Chain complete: the runner's gates (score, confluence,
        # manipulation, cost, margin) decide the order.  The
        # machine holds ENTRY until marked entered or invalidated.
        # Invalidation: close back through the CHoCH extreme, or
        # close THROUGH the confirm FVG's far side (level failed).
        fvg = (state.confirm or {}).get("fvg") or {}
        through_fvg = False
        if fvg:
            if state.direction > 0 and price < float(fvg["bottom"]):
                through_fvg = True
            if state.direction < 0 and price > float(fvg["top"]):
                through_fvg = True
        if (state.invalidation is not None and (
                (state.direction > 0 and price < state.invalidation)
                or (state.direction < 0
                    and price > state.invalidation))) \
                or through_fvg:
            state.log("entry invalidated before fill — back to "
                      "STATE 1")
            state.state = HTF_ZONE_APPROACH
            state.entry = {}
            state.confirm = {}
            state.bars_in_state = 0
        return state

    if state.state == MANAGEMENT:
        ok, reason = narrative_invalidation(state, frames)
        if ok:
            state.state = EXIT
            state.exit_reason = reason
            state.log(f"management invalidation: {reason}")
            return state
        # STATE 6 bar-close bookkeeping (breakeven / partials
        # / trailing / time-stop / upgrade) from the plan
        # attached by mark_entered.  Invalidation above is
        # authoritative; evaluate_management is told the
        # position is not invalidated (fvg_mitigated=False
        # with through recomputed inside is consistent:
        # invalidation already ruled out).  Elapsed H1 is
        # bars_in_state: advance() is called once per H1
        # close by the runner, so one refresh is one H1 bar.
        # Volume is not known on the bar-close path, so the
        # upgrade gate (needs volume) stays for the tick path.
        if state.management and state.entry:
            fvg = state.confirm.get("fvg") or {}
            struct = None
            if fvg:
                struct = float(fvg["bottom"]) \
                    if state.direction > 0 else float(fvg["top"])
            action = evaluate_management(
                plan=state.management,
                direction=state.direction,
                entry_price=float(state.entry.get("price", price)),
                stop=float(state.entry.get("stop", price)),
                current_price=price,
                elapsed_h1=state.bars_in_state,
                atr=atr, atr_ratio=state.atr_ratio,
                choch_extreme=state.invalidation,
                fvg_mitigated=False,
                structure_level=struct)
            state.management = action.plan
            if action.breakeven and not state.management.get(
                    "breakeven_moved"):
                state.management["breakeven_moved"] = True
                state.log("STATE 6: breakeven due")
            taken = list(state.management.get("partials_taken", []))
            newly = [t for t in action.partials if t not in taken]
            if newly:
                state.management["partials_taken"] = taken + newly
                state.log("STATE 6: partials hit: "
                          + ", ".join(newly))
            if action.new_sl is not None:
                state.management["trailing_stop"] = action.new_sl
                state.log(f"STATE 6: trailing SL -> "
                          f"{action.new_sl:.2f}")
            if action.exit == "time_stop":
                state.state = EXIT
                state.exit_reason = "time_stop"
                state.log("STATE 6: time stop — exit")
            for line in action.log:
                state.log(line)
        return state

    return state


def mark_entered(state: NarrativeState, *, entry_price: float,
                 stop: float, targets: list) -> NarrativeState:
    """Runner calls this on fill: ENTRY -> MANAGEMENT.

    Attaches the differentiated management plan (clean vs
    late confirm, entry-zone risk tier, opposing-zone TP
    adjustment, Dynamic ATR time-stop scaling) so the runner
    manages from one plan, not ad-hoc rules.
    """
    if state.state == ENTRY:
        state.state = MANAGEMENT
        state.entry = {**state.entry, "price": float(entry_price),
                       "stop": float(stop), "targets": list(targets)}
        # Management ladder: clean 1.0x TP 1R/2R/3R vs late
        # 0.5x TP 0.75R/1.5R/2R, then the entry-zone tier
        # (most conservative ladder wins), then the opposing
        # zone moves TPs (never aborts).
        mgmt = asdict(plan_management(
            direction=state.direction, entry_price=entry_price,
            stop=stop, late_confirm=bool(state.late_confirm),
            high_vol=bool(state.high_vol)))
        if state.entry_tier:
            mgmt = apply_tier_to_management(
                mgmt, state.entry_tier, entry_price=entry_price,
                stop=stop, direction=state.direction)
        if state.opposing and state.atr_value > 0:
            tp_adj = adjust_tps(
                entry_price=entry_price, stop=stop,
                direction=state.direction, opposing=state.opposing,
                atr=state.atr_value,
                tp1=mgmt["tp1"], tp2=mgmt["tp2"],
                tp3=mgmt.get("tp3"),
                tp1_pct=mgmt["tp1_pct"], tp2_pct=mgmt["tp2_pct"],
                tp3_pct=mgmt.get("tp3_pct", 0.0))
            mgmt["tp1"] = tp_adj["tp1"]
            mgmt["tp2"] = tp_adj["tp2"]
            mgmt["tp3"] = tp_adj.get("tp3")
            mgmt["tp1_pct"] = tp_adj["tp1_pct"]
            mgmt["tp2_pct"] = tp_adj["tp2_pct"]
            mgmt["tp3_pct"] = tp_adj["tp3_pct"]
            # Keep the R multiples consistent with the
            # adjusted prices.
            _r = abs(entry_price - stop)
            if _r > 0:
                for _k in ("tp1", "tp2", "tp3"):
                    if mgmt.get(_k) is not None:
                        mgmt[f"{_k}_r"] = (
                            mgmt[_k] - entry_price) \
                            * state.direction / _r
            mgmt["exit_all_at_tp2"] = tp_adj["exit_all_at_tp2"]
            mgmt["allow_runner"] = tp_adj["allow_runner"]
            mgmt["trail_cap_price"] = tp_adj["trail_cap_price"]
            mgmt["trail_lock_r"] = tp_adj["trail_lock_r"]
            mgmt["tp_adjustment_log"] = tp_adj["log"]
            state.log(tp_adj["log"])
        # Dynamic ATR Regime: time-stop scales with the ATR
        # ratio at fill (never above baseline, never below
        # 3 H1).  SL is locked at fill — ATR never widens it.
        if state.atr_ratio > 1.0:
            mgmt["time_stop_h1"] = scaled_time_stop(
                int(mgmt.get("time_stop_h1", 12)),
                state.atr_ratio)
        mgmt["atr_ratio"] = state.atr_ratio
        mgmt["atr_regime"] = state.atr_regime
        state.management = mgmt
        state.bars_in_state = 0
        state.log(f"filled at {entry_price:.1f} — MANAGEMENT")
        state.log(mgmt.get("log", ""))
    return state


def mark_exited(state: NarrativeState, *, reason: str,
                r: float | None = None) -> NarrativeState:
    """Runner calls this on TP / SL / time stop: -> EXIT."""
    state.state = EXIT
    state.exit_reason = reason
    state.exit_r = r
    state.log(f"exit: {reason} R={r}")
    return state


def finalize_exit(state: NarrativeState, *, reason: str,
                  partials: list | None = None,
                  final_r: float = 0.0,
                  pnl_dollars: float | None = None,
                  exit_price: float | None = None
                  ) -> tuple[NarrativeState, dict]:
    """STATE 7 Exit/Journal — the canonical runner path on exit.

    Journals the full narrative chain (``live/exit.py``), logs
    ``STATE 7 EXIT: <reason>, R=<r>, -> STATE 0``, and resets the
    machine to STATE 0 IDLE.  The reset clears the setup (zone,
    sweep, displacement, confirm, entry, management, opposing,
    TP adjustment, invalidation, late/high-vol flags) but KEEPS
    the regime, the exit reason/R, and the journal entry (on
    ``last_exit_journal``).  The zone cache lives in the
    ZoneStore, not in this state — nothing to clear there.

    ``partials`` is a list of ``{"fraction": f, "r": r}`` for each
    partial already banked; ``final_r`` is the R of the final
    exit on the remainder (measured against the SL locked at
    fill, never a recalculated one).  Returns (idle_state,
    journal_entry).
    """
    canon = normalize_exit_reason(reason)
    total = realized_r(partials, final_r)
    journal = build_journal_entry(
        state, reason=canon, realized=total,
        pnl_dollars=pnl_dollars, exit_price=exit_price,
        partials=partials)
    idle = NarrativeState(
        regime=state.regime, exit_reason=canon, exit_r=total)
    idle.last_exit_journal = journal
    idle.narrative = list(state.narrative)
    idle.log(f"STATE 7 EXIT: {canon}, R={total:.2f}, -> STATE 0")
    return idle, journal


def narrative_invalidation(state: NarrativeState,
                           frames: dict) -> tuple[bool, str]:
    """Management invalidation: close back through the CHoCH
    extreme AND the confirm FVG mitigated.  Both required —
    one alone is noise, not invalidation."""
    entry_tf, entry = _entry_frame(_frames_for(frames))
    if entry is None or state.state != MANAGEMENT:
        return False, ""
    price = float(entry["close"].iloc[-1])
    inv = state.invalidation
    if inv is None:
        return False, ""
    through = (state.direction > 0 and price < inv) or \
        (state.direction < 0 and price > inv)
    if not through:
        return False, ""
    fvg = state.confirm.get("fvg") or {}
    mitigated = False
    if fvg:
        top, bottom = float(fvg["top"]), float(fvg["bottom"])
        c = entry["close"].to_numpy(dtype=float)
        mitigated = bool(np.any((c >= bottom) & (c <= top)))
    else:
        # No FVG recorded (e.g. breakout-stop entry): the
        # FVG condition is vacuous — the CHoCH break alone
        # invalidates.  Behaviour preserved from before the
        # STATE 6 refactor.
        return True, ("close back through CHoCH extreme "
                       "(no confirm FVG recorded)")
    ok, reason = check_invalidation(
        through_choch=through, fvg_mitigated=mitigated)
    if not ok:
        return False, ""
    return True, reason


def chain_check(state: NarrativeState | dict,
                direction: int) -> tuple[bool, str]:
    """Gate 0e: does this signal carry a COMPLETE narrative chain?

    Only playbooks with ``requires_narrative_chain`` are checked.
    The chain is complete at ENTRY / MANAGEMENT, at the signal's
    direction.  Everything earlier is 'chain incomplete' — the
    pattern fired alone, which never trades.
    """
    st = state.state if isinstance(state, NarrativeState) \
        else int((state or {}).get("state", IDLE))
    d = state.direction if isinstance(state, NarrativeState) \
        else int((state or {}).get("direction", 0))
    if st in (ENTRY, MANAGEMENT) and d == int(direction):
        return True, "narrative chain complete"
    return False, (f"narrative chain incomplete: "
                   f"{STATE_NAMES.get(st, '?')} "
                   f"(need ENTRY at direction {direction:+d})")


def amd_for_state(state: NarrativeState) -> str:
    """AMD narrative mapping for the current state."""
    if state.state in (IDLE, HTF_ZONE_APPROACH):
        return "accumulation"
    if state.state == LIQUIDITY_SWEEP:
        return "manipulation"
    if state.state in (INDUCEMENT_DISPLACEMENT, LTF_CONFIRMATION,
                       ENTRY, MANAGEMENT, EXIT):
        return "distribution"
    return "unknown"
