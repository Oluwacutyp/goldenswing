"""All-timeframe hierarchy — the REAL-TRADER engine (D1 -> H4 -> H1 -> M15/M5).

One shared implementation behind BOTH the live loop and the
multi-timeframe backtest.  There is no second, lookalike hierarchy
anywhere: ``goldenswing/backtest/mtf_hierarchy.py`` calls
:func:`evaluate_hierarchy` from this module, and the live loop
(``execution/runner.py`` via ``confluence.gate_live_signal``) calls
the same function.  Live/backtest parity on the hierarchy path is
by construction, and covered by a parity test.

HIERARCHY, NEVER DEMOCRACY
--------------------------
A signal fires only when every leg passes, in order.  Higher
timeframes grant permission downward; a lower timeframe can NEVER
override or outvote a higher one.  H4 does NOT get a second vote:
it supplies the POI zone; D1 supplies the direction.

1. ``d1_bias`` — D1 bias != 0 (last closed swing structure).
2. ``h4_agree`` — H4 context, recorded never a vote: ``aligned``
   (H4 trends with D1), ``pullback`` (H4 trends against D1 — the
   pullback INTO the D1-side POI, the lecture setup), ``flat`` (no
   H4 opinion — not a veto).  This leg always passes once D1 has
   a bias; the zone / trigger / entry legs below decide.  (Before
   2026-10-16 this leg required H4 == D1 and was the zero-signal
   root cause: it armed 21% of H1 bars on real gold and, because
   playbooks never ran, ``/status`` showed ``blocked legs: none``.)
3. ``h4_zone`` — the H1 trigger bar closes AT the H4 POI (inside the
   zone band, never beyond it).  No chasing.
4. ``h1_trigger`` — a structural event (BOS / MSCH / CHoCH) fires on
   the H1 trigger bar, in the bias direction, with a valid stop.
5. ``entry`` — the session entry clock (M15 inside a killzone, M30
   elsewhere) confirms with a same-direction structure event on one
   of the entry-TF bars composing the trigger H1 bar (the 4 M15 /
   2 M30 bars ending at the H1 close).  The confirmation happened
   during the trigger bar's formation; the trade enters at the next
   H1 open.  An entry without its H1 trigger does not fire (legs 1-4
   are evaluated first and block first).

STAMP CONVENTIONS.  OANDA bars are stamped at their OPEN; resampled
bars (``backtest/mtf_hierarchy.py``: H4/D1 from H1, M30 from M15)
are stamped at their CLOSE.  ``evaluate_hierarchy`` takes a
``stamps`` map (TF -> "open" | "close", default "open") and slices
each frame accordingly: an open-stamped bar counts iff
``open + tf_minutes <= h1_close``; a close-stamped bar counts iff
``close_stamp <= h1_close``.  Getting this wrong silently drops the
latest closed H4/D1 bar — the most relevant one — so the convention
is explicit, never inferred.

Legs 3-4 reuse :func:`goldenswing.live.multitf.evaluate` (SWING
preset: D1 bias / H4 zones / H1 trigger) with the hard veto OFF;
leg 2 records H4 context only (see above).  Leg 5 is the
session-clock entry leg (``reader/session_timeframes.py``).

HARD VETO IS OFF — ALWAYS.  This module never enables it: every call
into ``multitf`` passes ``enable_veto=False``, and the verdict
carries ``veto_enabled: False`` as proof.  The hard veto was measured
to remove winning trades in-sample and is a failed hypothesis except
possibly on Scalp (see SYSTEM_SCOPE_REVIEW.md §7.6); nothing in this
module re-enables it under any default.

AMD PHASE RIDES AS A TAG, NEVER A VETO.  The caller supplies the
phase (live: the H1 feature state's ``PhaseState``; backtest: the
same ``features/phases.py`` machine run over H1).  It is attached to
the verdict + session mapping and never blocks anything.

QM M5 IS THE M5 ENTRY LEG INSIDE THE HIERARCHY, NOT STANDALONE.  A QM
touch supplied via ``qm_touch`` is evaluated as part of the verdict
(``qm`` leg: present / direction-ok / recent).  It NEVER gates: a QM
touch alone never fires (the QM playbook is not in the live loop's
H1 path), and QM disagreement never blocks a sleeve signal.  QM stays
shadow — this leg is annotation + confirmation context.

CAUSALITY.  ``evaluate_hierarchy`` slices every frame to bars that
had CLOSED by the H1 trigger bar's close (``asof`` + 60min): a bar
with open ``t`` counts iff ``t + tf_minutes <= h1_close``.  The live
stack holds bars fetched together (including bars after the bar
being evaluated); the slice — not the stack's full extent — is what
the hierarchy reads, so a verdict at bar ``t`` can never peek at
``t+1``.  Only closed bars are ever evaluated; forming bars are
never used.

CURSORS.  :class:`TimeframeStack` is the per-timeframe cursor owner:
each TF's cursor only moves forward.  Out-of-order / duplicate bars
are rejected and counted, never merged, never rewinding the cursor
(the 2026-10-13 FVG rewind crash must not recur).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..entry.validator import SHAPE_PRIORITY, confirm_at_poi
from ..reader.session_timeframes import compose as _compose_session
from .multitf import (SWING, TimeframeConfig, atr14, detect_bos_msch_choch,
                      evaluate as _mtf_evaluate, evaluate_bias, find_swings,
                      zones_by_side)

#: The full timeframe stack, slowest -> fastest.
STACK_TFS: tuple[str, ...] = ("D1", "H4", "H1", "M15", "M30", "M5")

#: Minutes per bar, for causal closed-bar slicing.
#: MN1 is nominal 30 days (43200 min): a calendar month is 28-31
#: days, so duration arithmetic is NOT exact for MN1 — the
#: calendar-exact close logic lives in ``feed._closed_only``,
#: ``feed.closed_bar_close``, and ``_causal_window`` below, which
#: special-case MN1 by month.  This entry exists so the generic
#: ``TF_MINUTES[tf]`` lookups (next-close estimates) do not KeyError.
TF_MINUTES: dict[str, float] = {
    "M1": 1.0, "M5": 5.0, "M15": 15.0, "M30": 30.0, "H1": 60.0,
    "H4": 240.0, "D1": 1440.0, "W1": 10080.0, "MN1": 43200.0,
}

#: Hard veto policy: OFF, unconditionally, in this module.
HARD_VETO_ENABLED = False

#: Rolling window (bars per TF) passed to the structural engine.
WINDOW = 300

#: QM touch recency: a touch older than this vs ``asof`` is stale.
QM_RECENT_MINUTES = 30.0

#: Leg order — the hierarchy.  ``blocked_leg`` is the FIRST failed leg.
LEG_ORDER: tuple[str, ...] = (
    "data", "d1_bias", "h4_agree", "h4_zone", "h1_trigger", "entry",
)


# ------------------------------------------------------------------- stack
@dataclass
class TimeframeStack:
    """Bars-by-timeframe stack with per-TF forward-only cursors.

    ``frames`` maps TF -> OHLC DataFrame (UTC DatetimeIndex, sorted).
    ``cursors`` maps TF -> last accepted closed-bar timestamp.  Only
    :meth:`append` mutates the stack, and it never rewinds: a bar
    with ``ts <= cursor[tf]`` is rejected (duplicate / out-of-order),
    counted in ``rejected``, and never merged.
    """

    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    cursors: dict[str, pd.Timestamp | None] = field(default_factory=dict)
    rejected: dict[str, int] = field(default_factory=dict)
    accepted: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame]
                    ) -> "TimeframeStack":
        stack = cls()
        for tf, df in frames.items():
            if df is not None and len(df):
                stack.frames[tf] = df.sort_index()
                stack.cursors[tf] = df.index[-1]
        return stack

    def append(self, tf: str, df: pd.DataFrame
               ) -> tuple[int, int]:
        """Merge closed bars for one TF.  Returns (accepted, rejected).

        Only bars strictly newer than the TF cursor are accepted.
        Bars at/before the cursor (duplicates, out-of-order replays,
        rebase re-deliveries) are rejected, never merged.  The cursor
        moves to the newest accepted bar, or stays put when nothing
        is accepted.  NEVER rewinds.
        """
        if df is None or len(df) == 0:
            return 0, 0
        df = df.sort_index()
        cursor = self.cursors.get(tf)
        if cursor is not None:
            new = df[df.index > cursor]
            n_rejected = int(len(df) - len(new))
        else:
            new = df
            n_rejected = 0
        # Drop in-batch duplicates (keep last), then merge.
        new = new[~new.index.duplicated(keep="last")]
        if len(new):
            cur = self.frames.get(tf)
            merged = pd.concat([cur, new]) if cur is not None \
                and len(cur) else new
            merged = merged.sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            self.frames[tf] = merged
            self.cursors[tf] = merged.index[-1]
        self.rejected[tf] = self.rejected.get(tf, 0) + n_rejected
        self.accepted[tf] = self.accepted.get(tf, 0) + int(len(new))
        return int(len(new)), n_rejected

    def last_ts(self, tf: str) -> pd.Timestamp | None:
        return self.cursors.get(tf)

    def window(self, tf: str, n: int) -> pd.DataFrame:
        df = self.frames.get(tf)
        if df is None or len(df) == 0:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"])
        return df.tail(n)


# ------------------------------------------------------------- causal slice
def _causal_window(frame: pd.DataFrame, tf: str,
                   h1_close: pd.Timestamp, window: int,
                   *, stamp: str = "open") -> pd.DataFrame:
    """Bars of ``frame`` that had CLOSED by ``h1_close`` (tail ``window``).

    - ``stamp="open"`` (OANDA bars): a bar stamped at its open ``t``
      is closed iff ``t + TF_MINUTES[tf] <= h1_close``.
    - ``stamp="close"`` (resampled bars): the stamp IS the close, so
      the bar counts iff ``stamp <= h1_close``.
    The still-forming bar is excluded by construction — never by
    dropping the last row blindly.
    """
    if frame is None or len(frame) == 0:
        return frame.iloc[0:0] if frame is not None else pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"])
    if stamp == "close":
        cutoff = h1_close
    elif tf == "MN1":
        # Calendar-exact: a monthly bar stamped at its open is closed
        # iff its open month is strictly before h1_close's month.
        # Duration arithmetic (open + 43200min) would release a
        # 31-day month's bar a day early — never do that.  Year*12 +
        # month integer comparison: tz-safe, no Period warning.
        h1_month = h1_close.year * 12 + h1_close.month
        return frame[
            (frame.index.year * 12 + frame.index.month) < h1_month
        ].tail(window)
    else:
        minutes = TF_MINUTES.get(tf)
        if minutes is None:
            return frame.tail(window)
        cutoff = h1_close - pd.Timedelta(minutes=minutes)
    return frame[frame.index <= cutoff].tail(window)


# ------------------------------------------------------------------ verdict
def _leg(passed: bool, reason: str = "") -> dict:
    return {"passed": bool(passed), "reason": reason}


def _structural_reason_to_leg(reason: str) -> str:
    """Map a ``multitf.evaluate`` failure reason onto a hierarchy leg."""
    if reason in ("no_poi", "entry_above_poi", "entry_below_poi"):
        return "h4_zone"
    if reason in ("no_trigger_swings", "no_trigger",
                  "trigger_against_bias", "stop_invalid"):
        return "h1_trigger"
    if reason == "bias_flat":
        return "d1_bias"
    if reason in ("missing_tf", "insufficient_trigger_bars",
                  "insufficient_bias_bars", "insufficient_zone_bars"):
        return "data"
    return "h1_trigger"  # unknown structural reason: fail at the trigger


def evaluate_hierarchy(stack: TimeframeStack | dict,
                       *, asof=None, amd_phase: str | None = None,
                       amd_via: str | None = None,
                       amd_confidence: float | None = None,
                       qm_touch: dict | None = None,
                       swing_n: int = 3,
                       window: int = WINDOW,
                       stamps: dict | None = None) -> dict:
    """Hierarchy verdict at one H1 trigger bar (closed bars only).

    ``stack`` is a :class:`TimeframeStack` (live) or a plain
    ``{tf: DataFrame}`` mapping (backtest / tests).  ``asof`` is the
    H1 trigger bar's OPEN timestamp; when omitted it defaults to the
    last H1 bar.  Every frame is causally sliced to bars closed by
    ``asof + 60min`` before any leg reads it.  ``stamps`` maps TF ->
    "open" | "close" (default "open" for every TF; the backtest
    passes "close" for its resampled H4/D1/M30 frames).

    Returns a JSON-safe verdict dict: ``allowed`` (bool),
    ``blocked_leg`` (first failed leg in :data:`LEG_ORDER`, or ""),
    ``legs`` (per-leg ``{passed, reason}``), ``bias_d1`` / ``bias_h4``,
    ``direction``, ``zone`` / ``trigger`` / ``entry`` / ``stop`` /
    ``target`` / ``risk`` (structural signal, when legs 1-4 pass),
    ``session`` (DST-aware session mapping incl. ``entry_clock``),
    ``amd_phase`` / ``amd_via`` / ``amd_confidence`` (tag only),
    ``qm`` (the M5 entry-leg annotation), ``veto_enabled`` (always
    False), ``asof`` (ISO string).
    """
    if isinstance(stack, dict):
        stack = TimeframeStack.from_frames(stack)
    legs: dict[str, dict] = {leg: _leg(False, "not evaluated")
                             for leg in LEG_ORDER}
    verdict: dict = {
        "allowed": False, "blocked_leg": "data", "legs": legs,
        "bias_d1": 0, "bias_h4": 0, "h4_context": "", "direction": 0,
        "zone": None, "trigger": None, "entry": None, "stop": None,
        "target": None, "risk": None, "session": None,
        "amd_phase": amd_phase or "unknown",
        "amd_via": amd_via, "amd_confidence": amd_confidence,
        "qm": {"present": False, "direction_ok": False,
               "recent": False, "status": "qm_absent"},
        "veto_enabled": HARD_VETO_ENABLED, "asof": None,
        "entry_clock": None,
    }

    h1_frame = stack.frames.get("H1")
    if h1_frame is None or len(h1_frame) == 0:
        legs["data"] = _leg(False, "missing H1 frame")
        return verdict
    if asof is None:
        asof = h1_frame.index[-1]
    asof = pd.Timestamp(asof)
    verdict["asof"] = asof.isoformat()
    h1_close = asof + pd.Timedelta(minutes=60)

    # --- causal slices (closed bars only) for every TF in the stack.
    stamps = stamps or {}
    sliced: dict[str, pd.DataFrame] = {}
    for tf, frame in stack.frames.items():
        sliced[tf] = _causal_window(frame, tf, h1_close, window,
                                    stamp=stamps.get(tf, "open"))
    for tf in ("D1", "H4", "H1"):
        if tf not in sliced or len(sliced[tf]) < 2 * swing_n + 5:
            legs["data"] = _leg(
                False, f"insufficient closed {tf} bars "
                        f"({len(sliced.get(tf, []))})")
            return verdict
    legs["data"] = _leg(True, "")

    # --- leg 1: D1 bias.
    bias_d1 = evaluate_bias(sliced["D1"], swing_n=swing_n)
    verdict["bias_d1"] = int(bias_d1)
    if bias_d1 == 0:
        legs["d1_bias"] = _leg(False, "d1_bias_flat")
        return _finish(verdict, legs)
    legs["d1_bias"] = _leg(True, "")

    # --- leg 2: H4 context (recorded, never a vote).
    # H4 supplies the POI zone; D1 supplies the direction.  An H4
    # bias against D1 is the pullback INTO the D1-side POI — the
    # lecture setup — and a flat H4 is a missing second opinion,
    # not a veto.  Requiring H4 == D1 here was the 2026-10-16
    # zero-signal root cause (it armed 21% of H1 bars on real
    # gold; without it 46% arm, the rest gated by zone /
    # containment / trigger / score downstream).  The leg stays
    # in LEG_ORDER as the recorded context so verdict shape and
    # /status leg reporting do not change.
    bias_h4 = evaluate_bias(sliced["H4"], swing_n=swing_n)
    verdict["bias_h4"] = int(bias_h4)
    verdict["h4_context"] = _h4_context(bias_d1, bias_h4)
    if bias_h4 == 0:
        legs["h4_agree"] = _leg(
            True, "h4_flat: no H4 bias — H4 supplies the POI only")
    elif bias_h4 != bias_d1:
        legs["h4_agree"] = _leg(
            True, f"h4_pullback: H4 bias {bias_h4:+d} vs D1 "
                   f"{bias_d1:+d} — pullback into the D1-side POI; "
                   f"direction stays D1")
    else:
        legs["h4_agree"] = _leg(True, "h4_aligned")

    # --- legs 3-4: H4 zone + H1 trigger via the shared structural
    # engine (SWING preset, hard veto OFF — always).
    struct = _mtf_evaluate(
        {"D1": sliced["D1"], "H4": sliced["H4"], "H1": sliced["H1"]},
        TimeframeConfig("D1", "H4", "H1", swing_n=swing_n),
        enable_veto=HARD_VETO_ENABLED)
    if not struct["signal"]:
        leg = _structural_reason_to_leg(struct.get("reason", ""))
        if leg == "h4_zone":
            legs["h4_zone"] = _leg(False, struct.get("reason", ""))
            legs["h1_trigger"] = _leg(False, "not evaluated (zone blocked)")
        elif leg == "d1_bias":
            legs["d1_bias"] = _leg(False, struct.get("reason", ""))
            legs["h4_zone"] = _leg(False, "not evaluated (bias blocked)")
            legs["h1_trigger"] = _leg(False, "not evaluated (bias blocked)")
        elif leg == "data":
            legs["data"] = _leg(False, struct.get("reason", ""))
        else:
            legs["h4_zone"] = _leg(True, "")
            legs["h1_trigger"] = _leg(False, struct.get("reason", ""))
        return _finish(verdict, legs, qm_touch=qm_touch, bias=bias_d1)
    legs["h4_zone"] = _leg(True, "")
    legs["h1_trigger"] = _leg(True, "")
    verdict["direction"] = int(struct["direction"])
    verdict["zone"] = struct["zone"]
    verdict["trigger"] = struct["trigger"]
    verdict["entry"] = float(struct["entry"])
    verdict["stop"] = float(struct["stop"])
    verdict["target"] = float(struct["target"])
    verdict["risk"] = float(struct["risk"])

    # --- session mapping (annotation + entry-clock selection).
    session = _compose_session(asof, amd_phase=amd_phase)
    verdict["session"] = session
    entry_clock = session["entry_clock"]
    verdict["entry_clock"] = entry_clock

    # --- leg 5: session entry-clock confirmation (window scan: any
    # entry-TF bar composing the trigger H1 bar may confirm).
    entry_frame = sliced.get(entry_clock)
    if entry_frame is None or len(entry_frame) < 2 * swing_n + 5:
        legs["entry"] = _leg(
            False, f"entry_tf_missing: no closed {entry_clock} bars")
        return _finish(verdict, legs, qm_touch=qm_touch, bias=bias_d1)
    entry_window = max(1, int(60 // TF_MINUTES[entry_clock]))
    entry_event, _conf_idx = _entry_event_window(
        entry_frame, bias_d1, swing_n, entry_window)
    if entry_event is None:
        legs["entry"] = _leg(
            False, f"entry_no_confirm: no {entry_clock} structure event "
                    f"in the bias direction on the entry bars composing "
                    f"the trigger H1 bar")
        return _finish(verdict, legs, qm_touch=qm_touch, bias=bias_d1)
    if entry_event != bias_d1:
        legs["entry"] = _leg(
            False, f"entry_against_bias: {entry_clock} event "
                    f"{entry_event:+d} vs bias {bias_d1:+d}")
        return _finish(verdict, legs, qm_touch=qm_touch, bias=bias_d1)
    legs["entry"] = _leg(True, f"{entry_clock} event {entry_event:+d}")

    # --- QM M5 leg: annotation only, never a gate.
    verdict["qm"] = _qm_leg(qm_touch, bias_d1, asof)

    verdict["allowed"] = True
    verdict["blocked_leg"] = ""
    return verdict


def _finish(verdict: dict, legs: dict, *,
            qm_touch: dict | None = None, bias: int = 0) -> dict:
    """Fill ``blocked_leg`` (first failed leg) + session/QM annotation."""
    for leg in LEG_ORDER:
        if not legs[leg]["passed"]:
            verdict["blocked_leg"] = leg
            break
    else:
        verdict["blocked_leg"] = ""
    # Session + QM annotation ride along even on blocked verdicts:
    # per-leg stats and journaling need the clock + QM context for
    # every evaluated bar, not just the ones that passed.
    if verdict["session"] is None and verdict["asof"] is not None:
        verdict["session"] = _compose_session(
            pd.Timestamp(verdict["asof"]),
            amd_phase=verdict.get("amd_phase"))
        verdict["entry_clock"] = verdict["session"]["entry_clock"]
    if qm_touch is not None and bias != 0:
        verdict["qm"] = _qm_leg(qm_touch, bias, verdict["asof"])
    return verdict


def _entry_event_at(frame: pd.DataFrame, j: int, bias: int,
                    swing_n: int) -> int | None:
    """Structure event direction at closed bar ``j`` of ``frame``
    (swings from bars strictly before ``j``), or None."""
    sub = frame.iloc[:j + 1]
    if len(sub) < 2 * swing_n + 5:
        return None
    cfg = TimeframeConfig("H4", "H1", "M15", swing_n=swing_n)
    highs, lows = find_swings(sub["high"].to_numpy()[:-1],
                              sub["low"].to_numpy()[:-1], swing_n)
    if not highs or not lows:
        return None
    ref_high = highs[-1][1]
    minor_high = highs[-2][1] if len(highs) >= 2 else None
    ref_low = lows[-1][1]
    minor_low = lows[-2][1] if len(lows) >= 2 else None
    event = detect_bos_msch_choch(
        sub, ref_high, ref_low, atr14(sub), trend=bias,
        minor_high=minor_high, minor_low=minor_low, config=cfg)
    if event is None:
        return None
    return 1 if event.endswith("bull") else -1


def _entry_event(entry_frame: pd.DataFrame, bias: int,
                 swing_n: int) -> int | None:
    """Same-direction structure event on the entry TF's last closed bar.

    Returns +1 / -1 for a confirming event, None when there is no
    event (displacement gate failed, no swings, or the bar is chop).
    Direction is checked by the caller (``evaluate_hierarchy``).
    Kept for single-bar callers; the hierarchy uses the window scan.
    """
    if len(entry_frame) < 2 * swing_n + 5:
        return None
    return _entry_event_at(entry_frame, len(entry_frame) - 1, bias,
                           swing_n)


def _entry_event_window(entry_frame: pd.DataFrame, bias: int,
                        swing_n: int, window: int
                        ) -> tuple[int | None, int | None]:
    """Most recent structure event in the last ``window`` entry bars.

    Scans newest-first and returns ``(direction, index)`` for the
    first event found (either direction — the caller compares it to
    the bias), or ``(None, None)`` when no bar in the window fires.
    The most recent event wins: an against-bias event on the last bar
    is not rescued by an older with-bias event.
    """
    n = len(entry_frame)
    for j in range(n - 1, max(-1, n - 1 - window), -1):
        direction = _entry_event_at(entry_frame, j, bias, swing_n)
        if direction is not None:
            return direction, j
    return None, None


def _qm_leg(qm_touch: dict | None, bias: int, asof) -> dict:
    """QM M5 leg: annotation, never a gate."""
    out = {"present": False, "direction_ok": False, "recent": False,
           "status": "qm_absent", "touch_ts": None, "direction": 0}
    if not qm_touch:
        return out
    out["present"] = True
    direction = int(qm_touch.get("direction", 0))
    out["direction"] = direction
    out["direction_ok"] = direction == bias and direction != 0
    touch_ts = qm_touch.get("touch_ts") or qm_touch.get("touch_time")
    if touch_ts is not None:
        touch_ts = pd.Timestamp(touch_ts)
        out["touch_ts"] = touch_ts.isoformat()
        age_min = (pd.Timestamp(asof) - touch_ts).total_seconds() / 60.0
        out["recent"] = 0.0 <= age_min <= QM_RECENT_MINUTES
    if not out["direction_ok"]:
        out["status"] = "qm_against_bias"
    elif not out["recent"]:
        out["status"] = "qm_stale"
    else:
        out["status"] = "qm_confirmed"
    return out


def verdict_permits(verdict: dict, direction: int) -> tuple[bool, str]:
    """Does a hierarchy verdict permit a signal of ``direction``?

    Returns ``(True, "")`` or ``(False, "hierarchy:<leg>:<reason>")``.
    A verdict that allows the opposite direction does NOT permit
    this signal — direction mismatch blocks, it is never averaged.
    """
    if not verdict.get("allowed"):
        leg = verdict.get("blocked_leg", "data")
        reason = verdict.get("legs", {}).get(leg, {}).get("reason", "")
        return False, f"hierarchy:{leg}:{reason}"
    if int(verdict.get("direction", 0)) != int(direction):
        return False, (
            f"hierarchy:direction_mismatch: hierarchy allows "
            f"{verdict.get('direction'):+d}, signal is {direction:+d}")
    return True, ""


def zones_for_direction(zone_bars: pd.DataFrame, direction: int, *,
                        swing_n: int = 3) -> dict | None:
    """H4 POI zone for a direction (support for longs, resistance for
    shorts), or None.  Small helper for callers that need the zone
    without a full verdict."""
    zones = zones_by_side(zone_bars, swing_n=swing_n)
    if direction > 0:
        return zones.get("support")
    if direction < 0:
        return zones.get("resistance")
    return None


# ------------------------------------------------------- live evaluation clock
# The closed-bar hierarchy above (evaluate_hierarchy) is the shared
# implementation behind the backtest and the legacy paper loop.  The
# MT5 engine must NOT wait for an H1 bar to close before acting.
# These helpers implement the LIVE clock — touch arms on tick-cross,
# the confirmation family fires on entry-TF closes, TF closes only
# refresh levels — reusing the same primitives (evaluate_bias,
# zones_by_side, find_swings, atr14) and the same gate formulas as
# _evaluate.  Levels come from CLOSED bars only; the forming bar
# never confirms its own swing and never feeds a level.  The live
# verdict the runner builds from these helpers has the same shape
# as evaluate_hierarchy's output, so confluence.gate_live_signal
# needs no special case.
#
# TRIGGER, REDESIGNED (2026-10-15).  The old trigger demanded price
# PAST the H1 swing level AND inside the H4 POI zone at the same
# tick — geometrically near-impossible (full-engine backtest:
# 0/573 armed windows).  The new trigger is the owner's doctrine:
# entries AT the marked POI, never above it.  A tick whose price
# sits AT the H4 POI band arms the pending; the POI confirmation
# family (entry/validator.confirm_at_poi, all 19 shapes) decides
# whether it fires, on the entry-clock bars composing the forming
# H1 bar.  BOS/ChoCH survive as confirmation shapes inside the
# family — they are NOT a separate gate, and "past the swing level"
# is no longer a condition anywhere in the live trigger.


def _h4_context(bias_d1: int, bias_h4: int) -> str:
    """H4's role is context, never a second vote.

    The hierarchy is D1 decides direction, H4 supplies the POI zone.
    ``aligned`` = H4 trends with D1.  ``pullback`` = H4 trends
    against D1 — the canonical pullback INTO the D1 POI, the setup
    the lecture specifies (D1 bias + pullback into H4 zone + H1
    trigger).  ``flat`` = H4 has no bias; a missing H4 opinion is
    not a veto.  None of the three blocks; the zone, containment,
    trigger, score, confluence, manipulation, cost and margin gates
    below still decide whether anything fires.
    """
    if bias_h4 == 0:
        return "flat"
    if bias_h4 == bias_d1:
        return "aligned"
    return "pullback"


def _live_setup_impl(stack: TimeframeStack | dict, *,
                     swing_n: int = 3) -> tuple[dict | None, str]:
    """``live_setup`` + the reason it is / is not armed.

    Reason strings (``armed``, ``d1_flat``, ``no_h4_poi``,
    ``no_h1_swings``, ``no_atr``, ``containment``,
    ``insufficient_bars``) exist so ``/status`` can tell the owner
    WHY no setup is armed instead of showing ``blocked legs: none``
    for a runner that is merely waiting — the 2026-10-16
    ``last signal: none journaled yet`` screenshot.
    """
    if isinstance(stack, dict):
        stack = TimeframeStack.from_frames(stack)
    d1 = stack.frames.get("D1")
    h4 = stack.frames.get("H4")
    h1 = stack.frames.get("H1")
    need = 2 * swing_n + 5
    if d1 is None or len(d1) < need or h4 is None or len(h4) < need \
            or h1 is None or len(h1) < need:
        return None, "insufficient_bars"
    bias_d1 = evaluate_bias(d1, swing_n=swing_n)
    if bias_d1 == 0:
        return None, "d1_flat: D1 bias is flat — no direction, no setup"
    bias_h4 = evaluate_bias(h4, swing_n=swing_n)
    # NO H4-agreement requirement here — that was the 2026-10-16
    # zero-signal root cause.  Requiring H4 bias == D1 bias is
    # democracy (two votes), not hierarchy: during a pullback H4
    # NECESSARILY trends against D1 while price retraces into the
    # D1-side H4 POI.  Measured on real gold bars (last 200 H1
    # bars, causal): the agreement vote armed 21% of bars; without
    # it 46% arm, and the newly armed bars are H4-flat (no H4
    # opinion — not a veto) plus genuine pullbacks.  Direction
    # stays D1, always; H4's bias is recorded as context.
    zones = zones_by_side(h4, swing_n=swing_n)
    zone = zones.get("support") if bias_d1 > 0 \
        else zones.get("resistance")
    if zone is None:
        return None, "no_h4_poi: no H4 POI on the D1 side"
    trig_highs, trig_lows = find_swings(
        h1["high"].to_numpy(), h1["low"].to_numpy(), swing_n)
    if not trig_highs or not trig_lows:
        return None, "no_h1_swings: no confirmed H1 swings"
    atr_h1 = atr14(h1)
    if atr_h1 <= 0:
        return None, "no_atr: H1 ATR unmeasurable"
    buf = float(SWING.breakout_atr * atr_h1)
    # CONTAINMENT FIRST: the H1 reference level on the zone side
    # (swing low for longs, swing high for shorts) must NEST
    # inside the H4 zone (± buf, the codebase's existing level
    # tolerance — the same buf the stop sits beyond the zone
    # edge).  If the levels live at different coordinates, no
    # tick can ever satisfy both — the setup is invalid, skip it
    # instead of waiting forever on an impossible tick.  This
    # replaces the old simultaneous-cross requirement: nesting is
    # checked once here, at arming, not per tick.
    ref_level = float(trig_lows[-1][1]) if bias_d1 > 0 \
        else float(trig_highs[-1][1])
    if not (zone["bottom"] - buf <= ref_level
            <= zone["top"] + buf):
        return None, (
            "containment: H1 ref level "
            f"{ref_level:.1f} outside H4 POI "
            f"{zone['bottom']:.1f}–{zone['top']:.1f} "
            "(structure not at the zone yet)")
    h4_highs, h4_lows = find_swings(
        h4["high"].to_numpy(), h4["low"].to_numpy(), swing_n)
    context = _h4_context(bias_d1, bias_h4)
    return {
        "bias_d1": int(bias_d1), "bias_h4": int(bias_h4),
        "h4_context": context,
        "direction": int(bias_d1), "zone": zone,
        "ref_high": float(trig_highs[-1][1]),
        "minor_high": float(trig_highs[-2][1])
        if len(trig_highs) >= 2 else None,
        "ref_low": float(trig_lows[-1][1]),
        "minor_low": float(trig_lows[-2][1])
        if len(trig_lows) >= 2 else None,
        "ref_level": ref_level,
        "atr_h1": float(atr_h1),
        "buf": buf,
        "h4_highs": [float(p) for _, p in h4_highs],
        "h4_lows": [float(p) for _, p in h4_lows],
        "swing_n": int(swing_n),
    }, "armed"


def live_setup(stack: TimeframeStack | dict, *,
               swing_n: int = 3) -> dict | None:
    """Trigger levels from CLOSED bars only (refreshed on TF closes).

    Returns None when D1 bias is flat or no H4 POI / H1 swing
    levels exist — the setup is not armed.  H4 bias agreement is
    deliberately NOT required: H4 supplies the POI zone, D1
    supplies the direction (see ``_live_setup_impl``).  Otherwise
    returns the levels the tick clock checks every tick:
    ``bias_d1`` / ``bias_h4`` / ``h4_context`` / ``direction``,
    ``zone`` (the H4 POI band for the bias direction), H1
    reference/minor swing levels (retained for diagnostics; the
    trigger no longer requires a break past them), ``atr_h1``,
    ``buf`` (the stop buffer beyond the zone edge), and the H4
    swing price lists used for the target.  The stack frames must
    hold CLOSED bars only (the live invariant); the forming bar is
    never here.
    """
    setup, _reason = _live_setup_impl(stack, swing_n=swing_n)
    return setup


def live_setup_reason(stack: TimeframeStack | dict, *,
                      swing_n: int = 3) -> str:
    """Why :func:`live_setup` would / would not arm (``/status``)."""
    _setup, reason = _live_setup_impl(stack, swing_n=swing_n)
    return reason


def check_zone_touch(setup: dict, *, mid: float) -> bool:
    """Arm condition: the live price sits AT the H4 POI band.

    This is the trigger's first half: a touch, not a breakout.  A
    touch arms the pending; the confirmation family decides
    whether it fires (see :func:`check_live_trigger`).
    """
    zone = setup["zone"]
    return zone["bottom"] <= mid <= zone["top"]


def check_live_trigger(setup: dict, *, entry_frame: pd.DataFrame,
                       h1_open, close: float) -> dict | None:
    """Confirm an armed zone touch with the POI confirmation family.

    Runs :func:`confirm_at_poi` on the entry-clock bars composing
    the forming H1 bar (index >= ``h1_open``) against the setup's
    zone band, with the full entry frame as lookback context.  The
    confirmation window is therefore the entry bars composing the
    forming H1 bar — up to 4 for M15, 2 for M30, 12 for M5 — the
    same window the old entry leg used; the pending expires at the
    H1 close, so a touch-to-confirm span is bounded by one H1 bar.

    AT-POI: the confirming bar must overlap the POI band.  The
    family itself does NOT require this for bos / choch /
    sweep_reclaim / hh_hl (their detectors only need a bar inside
    the frame), so the trigger enforces it here: a reaction far
    from the zone is not a POI reaction, and a BOS above the zone
    must never fire the trigger (that was the old failure mode in
    reverse).  When several shapes confirm, the trigger is the
    first (earliest bar, then SHAPE_PRIORITY) whose bar overlaps.

    Returns None when nothing confirms at the POI, else
    ``{trigger, confirm_shapes, direction, entry, stop, target,
    risk, level, zone}``.  ``stop`` sits just beyond the defended
    zone edge plus the breakout buffer (long: zone bottom - buf;
    short: zone top + buf); ``level`` is the same stop level — a
    price through it invalidates the touch.  ``target`` is 2R,
    capped at the nearest H4 swing level beyond the entry when that
    level is at least 2R away (same rule as the old trigger).
    """
    zone = setup["zone"]
    direction = int(setup["direction"])
    if entry_frame is None or len(entry_frame) == 0:
        return None
    composing = entry_frame[entry_frame.index >= pd.Timestamp(h1_open)]
    if len(composing) == 0:
        return None
    ann = confirm_at_poi(composing, direction, zone["bottom"],
                         zone["top"], context=entry_frame)
    if not ann.confirmed:
        return None
    prio = {s: i for i, s in enumerate(SHAPE_PRIORITY)}
    trigger = None
    for shape in sorted(ann.shapes,
                        key=lambda s: (ann.bar_indices[s],
                                       prio.get(s, 99))):
        label = ann.bar_indices[shape]
        if label not in entry_frame.index:
            continue
        bar = entry_frame.loc[[label]].iloc[0]
        if bar["low"] <= zone["top"] and bar["high"] >= zone["bottom"]:
            trigger = shape
            break
    if trigger is None:
        return None
    buf = float(setup["buf"])
    if direction > 0:
        stop = zone["bottom"] - buf
        if not stop < close:
            return None
        cands = [p for p in setup["h4_highs"] if p > close]
    else:
        stop = zone["top"] + buf
        if not stop > close:
            return None
        cands = [p for p in setup["h4_lows"] if p < close]
    risk = abs(close - stop)
    if risk <= 0:
        return None
    target = close + direction * 2.0 * risk
    if cands:
        nearest = min(cands, key=lambda p: abs(p - close))
        if abs(nearest - close) >= 2.0 * risk:
            target = nearest
    return {"trigger": trigger, "confirm_shapes": list(ann.shapes),
            "direction": direction, "entry": float(close),
            "stop": float(stop), "target": float(target),
            "risk": float(risk), "level": float(stop), "zone": zone}


def qm_annotation(qm_touch: dict | None, bias: int, asof) -> dict:
    """QM M5 leg annotation (public wrapper over :func:`_qm_leg`)."""
    return _qm_leg(qm_touch, bias, asof)
