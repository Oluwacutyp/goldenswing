"""MarketSituation — the shared market read (Pass 0, shadow).

Spec: gs-marketsituation/SPEC.md §3 + §9 Pass 0; AMENDMENTS-1 §A4.3.

Before this module, every consumer of the market derived its own
read: the hierarchy read D1/H4/H1 through its gate chain, Judgment
parsed a regime label string computed on H1 only, each playbook
read its own clock's bars.  There was no single object that said
"this is the market, per timeframe, right now" — so two parts of
the engine could disagree about the same bars and neither would
know.

This module is that object.  It is a MEASUREMENT layer:

- Per-timeframe states are built ONLY from the desk's canonical
  detector organs — ``multitf.evaluate_bias``,
  ``multitf.detect_bos_msch_choch``, ``multitf.find_swings``, the
  pending-swing math in ``hierarchy``, the bedrock classifier in
  ``regime``, and the ATR regime method in ``atr_regime`` — so the
  situation can never disagree with the hierarchy's own math on
  identical bars (parity is test-pinned).
- NO verdict enters any state field.  The multitf ``evaluate``
  orchestration and the hierarchy's ``evaluate_hierarchy`` are
  sleeve verdicts; they are never called here and nothing they
  return is carried (pinned by test, behaviorally and at source
  level).
- Absent stays absent (house honesty law): a timeframe whose frame
  is missing or too short carries ``None`` fields and a readiness
  word that says why — ``"not_ready"`` (no data), ``"collecting"``
  (insufficient bars for the full read), ``"ready"``.  Nothing is
  synthesized to fill a gap.

Cadence (SPEC §3.5, Pass 0 form): the runner calls
:meth:`SituationKeeper.on_tf_close` after the append phase of its
TF-close pipeline.  Any TF close rebuilds that TF's state; an H1
close journals the full ``situation_snapshot`` (and refreshes the
latest-snapshot file); any other TF close journals a compact
``situation_delta`` ONLY when a measurement field of that TF's
state actually changed.  Every record carries its ``situation_id``
plus the previous one, forming the version chain: given any past
id, :meth:`SituationKeeper.reconstruct` rebuilds exactly what the
system knew at that point from the snapshot + delta records
(AMENDMENTS-1 §A4.3).

SHADOW: nothing in the decision path reads the situation in
Pass 0.  The only outward effect is journal records and the
``situation_id`` stamp on the funnel records of the cycle that
consumed the situation (execution/mt5_runner._emit_eval).

Pass 1 completes the §3.1 schema and adds the A4 update layer:

- The snapshot gains the §3.3 situation-level blocks (regime /
  location / activity / cross_asset / data_quality).  The pure
  builder derives the regime skeleton and the data-quality
  skeleton from the per-TF states themselves; the runner
  supplies the blocks that need live engine organs (day object,
  zone store, DXY tracker, news calendar) through the keeper,
  and a block the runner cannot supply reads absent — named in
  ``data_quality.missing`` — never synthesized.
- ``relationships`` (live/relationships.py) rides every
  assembled situation: the cross-timeframe pair kinds plus, on
  H1 closes, the hierarchy's own D1→H1 reading as one row among
  rows.  Any TF close recomputes the pairs touching it; a pair
  kind change journals a delta even when no TF state changed
  (price alone can break a defended swing).
- ``situation_event`` records (AMENDMENTS-1 §A4.1) journal the
  transitions detected between refreshes: structure-event
  episodes (appearance = ``structure_event``, kind/direction
  change = ``structural_break``), volatility-state changes and
  liquidity-pool entries/takes measured against the location
  block.  An event anchors a delta on the situation_id chain so
  the moment is reconstructable (A4.3).
- The ``live`` block (A4.2) carries intrabar reaction — price
  inside active zones and open positions — fed by the runner's
  management cycle via :meth:`SituationKeeper.note_live`.
  Every field is stamped ``provisional``: it is reaction, not
  truth, and it NEVER writes into the closed-bar state fields
  (the separation is test-pinned).

SHADOW discipline is unchanged: with the keeper disabled the
record shapes are byte-identical to before Pass 0, and no
decision record changes shape in any mode.

Momentum + AMD phase (the §9 ship remainder — "momentum /
AMD-phase TFState fields — build_tf_state carries no such
keys"): every TFState now also carries the SPEC §3.1
``momentum`` and ``phase`` fields, measured here from the
frame's own bars with the desk's existing organs:

- ``momentum.velocity_atr`` is the ATR-normalised close
  velocity over ``VELOCITY_BARS`` bars — the construction
  live/trend_momentum names as the situation schema's own
  (close displacement per bar, in ATR, signed).  ``band``
  bands that number: |v| < 0.05 ATR/bar is ``flat`` (over the
  velocity window that is under the AMD exit buffer — noise,
  not motion); |v| >= 0.25 ATR/bar is ``strong_*`` (sustained
  over the window that pace is a displacement-scale move,
  >= DISP_MIN_ATR).  ``displacement`` is the latest bar's
  body-dominated close in the displacement doctrine's own
  geometry (body >= DISP_BODY_WICK_MIN x wick, either
  direction, else None).  ``candle_character`` is the latest
  bar's named pattern from live/candle_taxonomy, or None.
  ``divergence`` is the features/divergence RSI family read
  at the last bar — ``{"active", "direction", "family"}``,
  with a measured-inactive dict when the family ran and found
  nothing live, and None when the frame is too short for the
  family's pivots (that absence is the mean-reversion family's
  SPEC §6.4 evidence input, previously always absent because
  no builder produced the key).
- ``phase`` is a STRUCTURAL AMD read (phase by structure,
  never an EMA): the trailing AMD window's range geometry
  against the TF's own ATR, in the bar-level AMD semantics of
  the desk's own live/patterns.amd_phase and the sweep
  definition of features/phases ("a liquidity sweep through
  a base extreme that closes back inside is manipulation").
  The features/phases state machine itself is NOT run here:
  it consumes the full per-bar FeatureSet (sweeps, FVGs,
  displacement series) that a pure frame builder does not
  carry, so this read is defined from the measurements this
  builder already makes, and says so.  Labels:
  ``distribution`` — the last close has expanded beyond the
  base segment's range extreme by the AMD exit buffer with
  a displacement-scale move over the recent segment (the
  expansion / delivery leg, including an ongoing trend);
  ``manipulation`` — a recent bar swept beyond a range-bound
  base's extreme and closed back inside, and price is still
  inside (the false break / retrace shape); ``accumulation``
  — price is still basing inside a range-bound trailing
  window (the AMD machine's own base width / drift ceilings).
  None of the three signatures measured -> ``undetermined``
  WITH the measured facts as the reason; too few bars ->
  the label stays None (absent) with the reason.  It is
  never defaulted to accumulation.

Both are state/reporting fields only: no consumer's decision
path is changed by their presence (relationships read bias /
structure_phase, Judgment reads the situation's other blocks,
and the one playbook that reads ``momentum.divergence``
records it as evidence context, exactly as it was built to).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.config import FeatureConfig as _FeatureConfig
from ..features.divergence import detect_divergences as _detect_divergences
from ..features.divergence import rsi_wilder as _rsi_wilder
from . import multitf
from . import relationships as _relationships
from .atr_regime import atr_regime as _atr_regime
from .atr_regime import atr_series as _atr_series
from .candle_taxonomy import patterns_at as _candle_patterns_at
from .displacement import DISP_BODY_WICK_MIN, DISP_MIN_ATR
from .hierarchy import _pending_swings
from .regime import ATR_CONTRACTING, SHORT_WINDOW, classify_regime
from .trend_momentum import VELOCITY_BARS as _VELOCITY_BARS

SITUATION_VERSION = 1

#: The timeframes the situation reads (SPEC §3.1).  M1 is reported
#: in ``clocks`` only — Hot Scalp keeps its own M1 stream.
SITUATION_TFS = ("D1", "H4", "H1", "M30", "M15", "M5")

#: Readiness words (the brief's mapping: NO DATA -> not_ready,
#: INSUFFICIENT -> collecting, else ready).
READINESS_NOT_READY = "not_ready"
READINESS_COLLECTING = "collecting"
READINESS_READY = "ready"

#: Bars a frame needs before the bedrock classifier's read counts
#: as the full state (regime.py's own floor: SHORT_WINDOW + 5).
MIN_BARS_READY = SHORT_WINDOW + 5

#: The feature family's own default constants, sourced (never
#: re-typed) from features/config.py: the divergence family's
#: RSI period / pivot geometry / active age, and the AMD
#: machine's range window, base width / drift ceilings, exit
#: buffer and manipulation-episode memory.
_FEATURE_DEFAULTS = _FeatureConfig()

#: Momentum band edges, in ATR per bar of velocity_atr.  Below
#: FLAT, the velocity window's total travel is under the AMD
#: exit buffer (0.3 ATR) — noise, not motion.  At STRONG, the
#: pace sustained over the velocity window is a
#: displacement-scale move (>= DISP_MIN_ATR = 1.5 ATR).
MOMENTUM_FLAT_ATR = 0.05
MOMENTUM_STRONG_ATR = 0.25

#: AMD phase measurement (see the module docstring): the
#: trailing window and base ceilings are the AMD machine's
#: own (features/config.py amd_*); the sweep must clear the
#: base extreme by this much to count (live/patterns.amd_phase's
#: sweep clearance), and the manipulation episode memory is
#: the AMD machine's amd_manip_bars.
AMD_WINDOW = int(_FEATURE_DEFAULTS.amd_range_bars)
AMD_BASE_WIDTH_ATR = float(_FEATURE_DEFAULTS.amd_base_width_atr)
AMD_BASE_DRIFT_ATR = float(_FEATURE_DEFAULTS.amd_base_drift_atr)
AMD_EXIT_BUFFER_ATR = float(_FEATURE_DEFAULTS.amd_exit_buffer_atr)
AMD_MANIP_BARS = int(_FEATURE_DEFAULTS.amd_manip_bars)
AMD_SWEEP_BEYOND_ATR = 0.25
#: Bars a frame needs before an AMD label is measured at all
#: (live/patterns.amd_phase's own floor for its bar-level read).
AMD_MIN_BARS = 30

SNAPSHOT_KIND = "situation_snapshot"
DELTA_KIND = "situation_delta"
EVENT_KIND = "situation_event"

#: The §3.3 situation-level block keys, in schema order.
BLOCK_KEYS = ("regime", "location", "activity", "cross_asset",
              "data_quality")

#: Event types journaled as ``situation_event`` records
#: (AMENDMENTS-1 §A4.1).
EVENT_STRUCTURE = "structure_event"
EVENT_STRUCTURAL_BREAK = "structural_break"
EVENT_VOLATILITY = "volatility_change"
EVENT_LIQUIDITY = "liquidity_event"

_JOURNAL_FILE = "situations.jsonl"
_LATEST_FILE = "situation_latest.json"

#: Fields excluded from the delta change comparison: freshness is
#: a function of the rebuild instant, not of the market — if it
#: counted, every close would journal a delta and the journal
#: would flood (SPEC §10 risk 1, controlled here).
_CHANGE_EXCLUDED = frozenset({"freshness_s"})


# ------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------
def _iso(ts) -> str | None:
    """Timestamp -> the house journal string form (``str`` of a
    tz-aware UTC Timestamp).  Naive inputs are read as UTC.
    None in -> None out; unparseable -> None (never raises)."""
    if ts is None:
        return None
    try:
        stamp = pd.Timestamp(ts)
    except Exception:  # noqa: BLE001 — an absent clock stays absent
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return str(stamp)


def _finite(value) -> float | None:
    """float(value) when it is a real, finite number; else None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _swing_point(frame: pd.DataFrame, point) -> dict | None:
    """A ``(position, price)`` swing -> ``{"price", "ts"}`` with the
    bar's own timestamp; out-of-range positions degrade to a
    price-only point rather than raising."""
    if point is None:
        return None
    pos, price = point
    ts = None
    try:
        if 0 <= int(pos) < len(frame):
            ts = _iso(frame.index[int(pos)])
    except Exception:  # noqa: BLE001
        ts = None
    return {"price": _finite(price), "ts": ts}


# ------------------------------------------------------------------
# momentum + AMD phase (SPEC §3.1 — the §9 remainder fields)
# ------------------------------------------------------------------
def _divergence_read(frame: pd.DataFrame, n: int) -> dict | None:
    """The features/divergence RSI family at the frame's last
    bar: ``{"active", "direction", "family"}`` for the newest
    divergence whose confirmation is still within the family's
    own active age, a measured-inactive dict when the family
    ran and nothing is live, and None when the frame is too
    short for two confirmed pivots (absent, never guessed)."""
    cfg = _FEATURE_DEFAULTS
    floor = (cfg.div_rsi_period + cfg.div_pivot_left
             + cfg.div_pivot_right + cfg.div_min_sep)
    if n < floor:
        return None
    try:
        rsi = _rsi_wilder(
            frame["close"].to_numpy(dtype=float), cfg.div_rsi_period)
        found = _detect_divergences(
            frame["high"].to_numpy(dtype=float),
            frame["low"].to_numpy(dtype=float), rsi,
            cfg.div_rsi_period, left=cfg.div_pivot_left,
            right=cfg.div_pivot_right, min_sep=cfg.div_min_sep,
            max_sep=cfg.div_max_sep, min_drsi=cfg.div_min_drsi)
    except Exception:  # noqa: BLE001 — a malformed frame degrades
        return None  # to absent, never a raise
    live = [d for d in found
            if 0 <= (n - 1) - d.confirmed_at <= cfg.div_max_age]
    if not live:
        return {"active": False, "direction": 0, "family": None}
    newest = live[-1]  # detect_divergences sorts by confirmation
    return {"active": True, "direction": int(newest.direction),
            "family": str(newest.kind)}


def _momentum_read(frame: pd.DataFrame, atr: float | None) -> dict:
    """The per-TF momentum read (SPEC §3.1 ``momentum``) from
    the frame's own bars.  Every subfield is a measurement or
    None; ``reason`` names why the headline (velocity) is
    absent when it is."""
    read: dict = {"velocity_atr": None, "band": None,
                  "displacement": None, "candle_character": None,
                  "divergence": None, "reason": None}
    n = len(frame)
    if atr is None:
        read["reason"] = ("no ATR measured (frame too short) — "
                          "momentum unmeasured")
        return read
    try:
        opens = frame["open"].to_numpy(dtype=float)
        highs = frame["high"].to_numpy(dtype=float)
        lows = frame["low"].to_numpy(dtype=float)
        closes = frame["close"].to_numpy(dtype=float)
    except Exception:  # noqa: BLE001
        read["reason"] = "frame columns unreadable — momentum unmeasured"
        return read
    if n > _VELOCITY_BARS:
        velocity = _finite(
            (closes[-1] - closes[-1 - _VELOCITY_BARS])
            / (_VELOCITY_BARS * atr))
        read["velocity_atr"] = velocity
        if velocity is not None:
            mag = abs(velocity)
            if mag < MOMENTUM_FLAT_ATR:
                read["band"] = "flat"
            elif velocity > 0:
                read["band"] = ("strong_bull"
                                if mag >= MOMENTUM_STRONG_ATR
                                else "bull")
            else:
                read["band"] = ("strong_bear"
                                if mag >= MOMENTUM_STRONG_ATR
                                else "bear")
    else:
        read["reason"] = (
            f"insufficient bars ({n}) for a velocity read over "
            f"{_VELOCITY_BARS} bars")
    # Displacement: the latest bar's body-dominated close
    # (displacement doctrine geometry, either direction).
    body = closes[-1] - opens[-1]
    wick = ((highs[-1] - max(opens[-1], closes[-1]))
            + (min(opens[-1], closes[-1]) - lows[-1]))
    if body > 0 and body >= DISP_BODY_WICK_MIN * max(wick, 0.0):
        read["displacement"] = "bull"
    elif body < 0 and -body >= DISP_BODY_WICK_MIN * max(wick, 0.0):
        read["displacement"] = "bear"
    # Candle character: the latest bar's named pattern, if any.
    try:
        named = _candle_patterns_at(opens, highs, lows, closes,
                                    float(atr), n - 1)
    except Exception:  # noqa: BLE001
        named = []
    if named:
        read["candle_character"] = str(named[0][0])
    read["divergence"] = _divergence_read(frame, n)
    return read


def _amd_phase_read(frame: pd.DataFrame,
                    atr: float | None) -> dict:
    """The structural AMD phase (SPEC §3.1 ``phase``) — see the
    module docstring for the definitions and their provenance.
    ``label`` is None (absent) only when the frame cannot
    support the read at all; a measured frame with no AMD
    signature is ``undetermined`` with the facts as its reason,
    never a defaulted label."""
    read: dict = {"label": None, "direction": None,
                  "range_high": None, "range_low": None,
                  "reason": None}
    n = len(frame)
    if n < AMD_MIN_BARS:
        read["reason"] = (
            f"insufficient bars ({n}) for an AMD read "
            f"(needs {AMD_MIN_BARS})")
        return read
    if atr is None:
        read["reason"] = "no ATR measured — AMD phase unmeasured"
        return read
    try:
        window = frame.tail(AMD_WINDOW)
        highs = window["high"].to_numpy(dtype=float)
        lows = window["low"].to_numpy(dtype=float)
        closes = window["close"].to_numpy(dtype=float)
    except Exception:  # noqa: BLE001
        read["reason"] = "frame columns unreadable — AMD phase unmeasured"
        return read
    w = len(closes)
    recent = min(AMD_MANIP_BARS, w // 3)
    base_high = float(np.max(highs[:-recent]))
    base_low = float(np.min(lows[:-recent]))
    read["range_high"] = _finite(base_high)
    read["range_low"] = _finite(base_low)
    base_width_atr = (base_high - base_low) / atr
    last_close = float(closes[-1])
    recent_net_atr = float(closes[-1] - closes[-recent]) / atr
    inside_base = base_low <= last_close <= base_high
    # Distribution: expansion beyond the base range with a
    # displacement-scale move over the recent segment.
    if last_close > base_high + AMD_EXIT_BUFFER_ATR * atr \
            and recent_net_atr >= DISP_MIN_ATR:
        read.update({
            "label": "distribution", "direction": 1,
            "reason": (
                f"expansion up: last close beyond the base range "
                f"high {base_high:.1f} by "
                f"{(last_close - base_high) / atr:.1f} ATR with a "
                f"{recent_net_atr:.1f} ATR move over the last "
                f"{recent} bars")})
        return read
    if last_close < base_low - AMD_EXIT_BUFFER_ATR * atr \
            and recent_net_atr <= -DISP_MIN_ATR:
        read.update({
            "label": "distribution", "direction": -1,
            "reason": (
                f"expansion down: last close beyond the base "
                f"range low {base_low:.1f} by "
                f"{(base_low - last_close) / atr:.1f} ATR with a "
                f"{recent_net_atr:.1f} ATR move over the last "
                f"{recent} bars")})
        return read
    # Manipulation: a sweep through a range-bound base's extreme
    # that closes back inside, price still inside (the false
    # break — features/phases' manipulation definition).
    if base_width_atr <= AMD_BASE_WIDTH_ATR and inside_base:
        for i in range(w - recent, w):
            if highs[i] >= base_high + AMD_SWEEP_BEYOND_ATR * atr \
                    and closes[i] <= base_high:
                read.update({
                    "label": "manipulation", "direction": -1,
                    "reason": (
                        f"sweep above the base range high "
                        f"{base_high:.1f} closed back inside "
                        f"{w - 1 - i} bar(s) ago — manipulation "
                        f"(upthrust shape)")})
                return read
            if lows[i] <= base_low - AMD_SWEEP_BEYOND_ATR * atr \
                    and closes[i] >= base_low:
                read.update({
                    "label": "manipulation", "direction": 1,
                    "reason": (
                        f"sweep below the base range low "
                        f"{base_low:.1f} closed back inside "
                        f"{w - 1 - i} bar(s) ago — manipulation "
                        f"(spring shape)")})
                return read
    # Accumulation: still basing inside a range-bound window
    # (the AMD machine's base width / drift ceilings).
    full_width_atr = float(np.max(highs) - np.min(lows)) / atr
    full_drift_atr = abs(float(closes[-1] - closes[0])) / atr
    if inside_base and full_width_atr <= AMD_BASE_WIDTH_ATR \
            and full_drift_atr <= AMD_BASE_DRIFT_ATR:
        read.update({
            "label": "accumulation", "direction": 0,
            "reason": (
                f"range-bound base: trailing {w}-bar range "
                f"{full_width_atr:.1f} ATR wide, net drift "
                f"{full_drift_atr:.1f} ATR, price inside — "
                f"accumulation")})
        return read
    read.update({
        "label": "undetermined", "direction": None,
        "reason": (
            f"no AMD signature measured: base range "
            f"{base_width_atr:.1f} ATR wide, trailing drift "
            f"{full_drift_atr:.1f} ATR, last close "
            f"{'inside' if inside_base else 'outside'} the base "
            f"range — neither a range-bound base, a "
            f"sweep-and-retrace, nor a displacement expansion")})
    return read


# ------------------------------------------------------------------
# per-timeframe state (SPEC §3.2 — Pass 0 fields)
# ------------------------------------------------------------------
def build_tf_state(tf: str, frame: pd.DataFrame | None, *,
                   asof=None, swing_n: int = 3) -> dict:
    """One timeframe's measured state from its CLOSED bars.

    Pure function of the frame.  Every field is a measurement from
    the canonical organs named in the module docstring; a field
    the frame cannot support is ``None``, never a guess.
    """
    state: dict = {
        "tf": tf,
        "readiness": READINESS_NOT_READY,
        "bars": 0,
        "last_close": None,
        "bias": None,
        "direction_word": None,
        "structure_phase": None,
        "regime_label": None,
        "regime_confidence": None,
        "last_structure_event": None,
        "swings": {"confirmed_high": None, "confirmed_low": None,
                   "pending_high": None, "pending_low": None},
        "volatility": {"atr": None, "atr_ratio": None,
                       "atr_state": None},
        "momentum": {"velocity_atr": None, "band": None,
                     "displacement": None, "candle_character": None,
                     "divergence": None,
                     "reason": "no bars — momentum unmeasured"},
        "phase": {"label": None, "direction": None,
                  "range_high": None, "range_low": None,
                  "reason": "no bars — AMD phase unmeasured"},
        "freshness_s": None,
    }
    if frame is None or len(frame) == 0:
        return state
    n = len(frame)
    state["bars"] = int(n)
    last_ts = frame.index[-1] if hasattr(frame, "index") else None
    state["last_close"] = _iso(last_ts)
    state["readiness"] = (READINESS_READY if n >= MIN_BARS_READY
                          else READINESS_COLLECTING)
    if asof is not None and last_ts is not None:
        try:
            age = (pd.Timestamp(asof) - pd.Timestamp(last_ts))
            state["freshness_s"] = _finite(age.total_seconds())
        except Exception:  # noqa: BLE001
            state["freshness_s"] = None

    # --- swings (confirmed + pending) ---------------------------
    highs: list = []
    lows: list = []
    try:
        highs, lows = multitf.find_swings(
            frame["high"], frame["low"], swing_n)
    except Exception:  # noqa: BLE001 — a malformed frame degrades
        highs, lows = [], []  # to absent swings, never a raise
    if highs:
        state["swings"]["confirmed_high"] = _swing_point(
            frame, highs[-1])
    if lows:
        state["swings"]["confirmed_low"] = _swing_point(
            frame, lows[-1])
    try:
        pend_high, pend_low = _pending_swings(
            frame["high"].to_numpy(dtype=float),
            frame["low"].to_numpy(dtype=float), swing_n)
    except Exception:  # noqa: BLE001
        pend_high, pend_low = None, None
    state["swings"]["pending_high"] = _swing_point(frame, pend_high)
    state["swings"]["pending_low"] = _swing_point(frame, pend_low)

    # --- bias (canonical: the only bias the desk gates on) ------
    bias = None
    if n >= 2 * swing_n + 1:
        try:
            bias = int(multitf.evaluate_bias(frame, swing_n=swing_n))
        except Exception:  # noqa: BLE001
            bias = None
    state["bias"] = bias

    # --- structure event on the last closed bar -----------------
    atr = None
    if n >= 2:
        measured = _finite(multitf.atr14(frame))
        atr = measured if measured and measured > 0 else None
    if bias is not None and highs and lows and atr is not None:
        try:
            raw = multitf.detect_bos_msch_choch(
                frame, highs, lows, atr, trend=bias,
                minor_high=highs[:-1] or None,
                minor_low=lows[:-1] or None)
        except Exception:  # noqa: BLE001
            raw = None
        if raw:
            kind, _, side = raw.partition("_")
            direction = 1 if side == "bull" else -1
            if direction > 0:
                level = highs[-1][1] if kind == "msch" else max(
                    h[1] for h in highs[-2:]) if len(highs) >= 2 \
                    else highs[-1][1]
            else:
                level = lows[-1][1] if kind == "msch" else min(
                    l[1] for l in lows[-2:]) if len(lows) >= 2 \
                    else lows[-1][1]
            state["last_structure_event"] = {
                "kind": kind,
                "direction": direction,
                "ts": state["last_close"],
                "level": _finite(level),
                "against_bias": bool(bias != 0
                                     and direction == -bias),
            }

    # --- bedrock structure / direction (per TF) ------------------
    if n >= MIN_BARS_READY:
        try:
            regime = classify_regime(frame)
        except Exception:  # noqa: BLE001
            regime = None
        if regime is not None and regime.reason != "not enough bars":
            state["structure_phase"] = regime.structure
            state["direction_word"] = regime.direction
            state["regime_label"] = regime.label()
            state["regime_confidence"] = _finite(regime.confidence)

    # --- volatility (atr_regime method, per TF) ------------------
    if atr is not None:
        state["volatility"]["atr"] = atr
        try:
            series = _atr_series(frame).dropna()
            baseline = (float(series.tail(100).median())
                        if len(series) else 0.0)
        except Exception:  # noqa: BLE001
            baseline = 0.0
        if baseline > 0:
            verdict = _atr_regime(atr, baseline)
            ratio = _finite(verdict.get("ratio"))
            state["volatility"]["atr_ratio"] = ratio
            if ratio is not None:
                state["volatility"]["atr_state"] = (
                    "contracting" if ratio < ATR_CONTRACTING
                    else str(verdict.get("regime") or "").lower()
                    or None)

    # --- momentum + AMD phase (SPEC §3.1, §9 remainder) -------
    # Both reads stand on the ATR measured above and the frame's
    # own bars; each degrades to its absent form (label/velocity
    # None + the reason in words) rather than raising.
    state["momentum"] = _momentum_read(frame, atr)
    state["phase"] = _amd_phase_read(frame, atr)
    return state


def stack_clocks(stack, *, m1_last_close=None) -> dict:
    """Last CLOSED bar per TF from the stack cursors (SPEC §3.1),
    plus M1's aux-frame last close when the runner supplies it."""
    clocks: dict = {}
    cursors = getattr(stack, "cursors", {}) or {}
    for tf in SITUATION_TFS:
        clocks[tf] = _iso(cursors.get(tf))
    clocks["M1"] = _iso(m1_last_close)
    return clocks


def _missing_inputs(per_tf: dict, blocks: dict) -> list[str]:
    """The named absent inputs of a situation (SPEC §3.3
    data_quality.missing): every timeframe that could not be
    fully read, and every situation-level block that no supplier
    provided.  Absent stays absent — and named."""
    missing: list[str] = []
    for tf in SITUATION_TFS:
        state = per_tf.get(tf) or {}
        readiness = state.get("readiness")
        if readiness != READINESS_READY:
            missing.append(f"{tf}:{readiness or 'unknown'}")
    for key in ("location", "activity", "cross_asset"):
        if blocks.get(key) is None:
            missing.append(key)
    return missing


def assemble_blocks(per_tf: dict, supplied: dict | None) -> dict:
    """The §3.3 blocks of a situation.  ``supplied`` is the
    runner's organ-derived blocks (any subset of BLOCK_KEYS);
    the regime and data-quality skeletons are derived here from
    the per-TF states and MERGED under whatever the supplier
    provided (its day object, its health verdict), so a supplied
    block never has to repeat what the states already measured.
    location / activity / cross_asset have no state-derived
    skeleton: unsupplied, they are None and named missing."""
    supplied = supplied or {}
    regime = {
        "bedrock_label_h1": (per_tf.get("H1") or {}).get(
            "regime_label"),
        "per_tf_labels": {tf: (per_tf.get(tf) or {}).get(
            "regime_label") for tf in SITUATION_TFS},
        "day": None,
    }
    if isinstance(supplied.get("regime"), dict):
        regime.update(copy.deepcopy(supplied["regime"]))
    data_quality = {
        "health": None,
        "per_tf_freshness_s": {tf: (per_tf.get(tf) or {}).get(
            "freshness_s") for tf in SITUATION_TFS},
        "missing": [],
    }
    if isinstance(supplied.get("data_quality"), dict):
        extra = copy.deepcopy(supplied["data_quality"])
        extra_missing = extra.pop("missing", None)
        data_quality.update(extra)
        if extra_missing:
            data_quality["missing"] = list(extra_missing)
    blocks = {
        "regime": regime,
        "location": copy.deepcopy(supplied.get("location")),
        "activity": copy.deepcopy(supplied.get("activity")),
        "cross_asset": copy.deepcopy(supplied.get("cross_asset")),
        "data_quality": data_quality,
    }
    seen = list(data_quality.get("missing") or [])
    for name in _missing_inputs(per_tf, blocks):
        if name not in seen:
            seen.append(name)
    data_quality["missing"] = seen
    return blocks


def build_situation(stack, *, symbol: str = "XAUUSD", asof=None,
                    situation_id: str = "",
                    prev_situation_id: str | None = None,
                    forecast_outlook: dict | None = None,
                    m1_last_close=None,
                    swing_n: int = 3,
                    blocks: dict | None = None,
                    relationships: list | None = None,
                    price: float | None = None,
                    live: dict | None = None) -> dict:
    """The full MarketSituation from the whole stack (pure).

    ``forecast_outlook`` is carried VERBATIM (a deep copy — the
    situation is a carrier, never a re-interpretation, SPEC §3.4);
    ``None`` when no outlook exists.  ``blocks`` is the runner's
    §3.3 organ blocks (see :func:`assemble_blocks`);
    ``relationships`` when omitted is computed from the per-TF
    states at ``price``; ``live`` is the keeper's provisional
    reaction block, carried — never merged into the closed-bar
    fields.  Nothing here raises on a short or missing frame:
    that TF's state says so instead.
    """
    frames = getattr(stack, "frames", {}) or {}
    per_tf = {tf: build_tf_state(tf, frames.get(tf), asof=asof,
                                 swing_n=swing_n)
              for tf in SITUATION_TFS}
    assembled = assemble_blocks(per_tf, blocks)
    if relationships is None:
        relationships = _relationships.compute_relationships(
            per_tf, price=price)
    situation = {
        "version": SITUATION_VERSION,
        "situation_id": situation_id,
        "prev_situation_id": prev_situation_id,
        "symbol": symbol,
        "created_at": _iso(asof),
        "clocks": stack_clocks(stack, m1_last_close=m1_last_close),
        "per_tf": per_tf,
        "relationships": copy.deepcopy(relationships),
        "forecasts": {"outlook": copy.deepcopy(forecast_outlook)},
        "live": copy.deepcopy(live),
    }
    for key in BLOCK_KEYS:
        situation[key] = assembled[key]
    return situation


def _state_changed(old: dict | None, new: dict) -> bool:
    """Did any MEASUREMENT field change?  (Freshness excluded —
    see _CHANGE_EXCLUDED.)  A first build counts as a change only
    when it measured something (an all-absent state journals
    nothing)."""
    if old is None:
        return any(v is not None for k, v in new.items()
                   if k not in _CHANGE_EXCLUDED
                   and k not in ("tf", "bars", "last_close",
                                 "readiness"))
    a = {k: v for k, v in old.items() if k not in _CHANGE_EXCLUDED}
    b = {k: v for k, v in new.items() if k not in _CHANGE_EXCLUDED}
    return a != b


def _compact_id(created_iso: str | None, seq: int) -> str:
    """``<yyyymmddThhmmssZ>-<seq>`` — greppable, sortable, unique
    per keeper journal."""
    stamp = "unknown"
    if created_iso:
        try:
            stamp = pd.Timestamp(created_iso).strftime(
                "%Y%m%dT%H%M%SZ")
        except Exception:  # noqa: BLE001
            stamp = "unknown"
    return f"{stamp}-{seq:04d}"


# ------------------------------------------------------------------
# Pass 1 event layer (AMENDMENTS-1 §A4.1) + price derivation
# ------------------------------------------------------------------
def _derive_price(stack) -> float | None:
    """The current price from the stack itself: the last close
    of the finest TF frame present (all frames share one price
    axis).  None when no frame has bars."""
    frames = getattr(stack, "frames", {}) or {}
    for tf in ("M5", "M15", "M30", "H1", "H4", "D1"):
        frame = frames.get(tf)
        if frame is not None and len(frame):
            try:
                return _finite(frame["close"].iloc[-1])
            except Exception:  # noqa: BLE001
                continue
    return None


def _event_fingerprint(state: dict | None) -> dict:
    """What the event layer remembers of one TF between
    refreshes: the structure-event EPISODE (kind + direction —
    consecutive continuation bars of the same episode are one
    event, not a flood) and the volatility state."""
    event = (state or {}).get("last_structure_event") or {}
    volatility = (state or {}).get("volatility") or {}
    return {
        "event_kind": (str(event.get("kind")).upper()
                       if event.get("kind") else None),
        "event_direction": event.get("direction"),
        "atr_state": volatility.get("atr_state"),
    }


def _detect_state_events(tf: str, old_state: dict | None,
                         new_state: dict) -> list[dict]:
    """Transitions between two measured states of one TF, as
    event payloads (``event_type`` / ``tf`` / ``detail``):
    a new structure-event episode, a kind/direction change of
    the episode (a structural break of the previous read), and
    a volatility-state change.  Appearance of an episode whose
    kind matches the previous episode's kind but follows a
    silent gap still counts as a new episode."""
    events: list[dict] = []
    old_fp = _event_fingerprint(old_state)
    new_fp = _event_fingerprint(new_state)
    new_event = new_state.get("last_structure_event")
    if new_event and (new_fp["event_kind"], new_fp["event_direction"]) != (
            old_fp["event_kind"], old_fp["event_direction"]):
        if old_fp["event_kind"] is None:
            etype = EVENT_STRUCTURE
        else:
            etype = EVENT_STRUCTURAL_BREAK
        events.append({
            "event_type": etype, "tf": tf,
            "detail": {
                "event": copy.deepcopy(new_event),
                "previous": {
                    "kind": old_fp["event_kind"],
                    "direction": old_fp["event_direction"]},
            }})
    old_atr = old_fp["atr_state"]
    new_atr = new_fp["atr_state"]
    if old_atr and new_atr and old_atr != new_atr:
        vol = new_state.get("volatility") or {}
        events.append({
            "event_type": EVENT_VOLATILITY, "tf": tf,
            "detail": {"from": old_atr, "to": new_atr,
                       "atr": vol.get("atr"),
                       "atr_ratio": vol.get("atr_ratio")}})
    return events


def _nearest_pools(location: dict | None) -> dict:
    """The location block's nearest pool per side (zone dicts,
    or None)."""
    location = location or {}
    above = location.get("zones_above") or []
    below = location.get("zones_below") or []
    return {"above": above[0] if above else None,
            "below": below[0] if below else None}


def _liquidity_events(prev_pools: dict, location: dict | None,
                      price: float | None, inside_before,
                      taken_before: set):
    """Pool entries and takes at one refresh.  Entries are read
    against the NEW location block (price inside a nearest
    pool's band [bottom, top]); takes are read against the
    PREVIOUS refresh's nearest pools — a pool price has swept
    through (beyond its distal edge) leaves the nearest list
    precisely because it was taken, so the old map is the only
    place the take is visible.  Nearest-pool identity changes
    themselves ride the location block silently: they are map
    changes, not price action.

    Returns ``(events, inside_now, taken_now, pools_now)``;
    ``taken`` is cumulative — a taken pool stays taken."""
    pools_now = _nearest_pools(location)
    taken_now = set(taken_before or ())
    inside_now = None
    events: list[dict] = []
    if price is None:
        return events, inside_before, taken_now, pools_now
    for side in ("above", "below"):
        zone = pools_now.get(side)
        if not zone:
            continue
        bottom, top = zone.get("bottom"), zone.get("top")
        if bottom is not None and top is not None \
                and bottom <= price <= top:
            inside_now = zone.get("zone_id")
    if inside_now and inside_now != inside_before:
        events.append({
            "event_type": EVENT_LIQUIDITY, "tf": None,
            "detail": {"change": "entered_pool",
                       "zone_id": inside_now, "price": price}})
    for side, beyond in (("above", 1), ("below", -1)):
        zone = (prev_pools or {}).get(side)
        if not zone:
            continue
        zone_id = zone.get("zone_id")
        edge = zone.get("distal")
        if zone_id is None or edge is None \
                or zone_id in taken_now:
            continue
        if (beyond > 0 and price > edge) or \
                (beyond < 0 and price < edge):
            taken_now.add(zone_id)
            events.append({
                "event_type": EVENT_LIQUIDITY, "tf": None,
                "detail": {"change": "pool_taken",
                           "zone_id": zone_id, "price": price}})
    return events, inside_now, taken_now, pools_now


# ------------------------------------------------------------------
# keeper: cadence, journaling, recovery (SPEC §3.5)
# ------------------------------------------------------------------
class SituationKeeper:
    """Owns the situation's cadence and durable records.

    Durable state: an append-only JSONL (``situations.jsonl`` in
    the journal directory) holding snapshot + delta records, and
    the latest-snapshot file (``situation_latest.json``, atomic
    tmp + fsync + replace with a SHA-256 checksum — the
    regime_state.py pattern).  An optional ``journal_append``
    callable (the runner's TradeJournal.append) mirrors every
    record into the live trade journal, which is where the demo
    reads them.

    On construction the JSONL tail is replayed: the keeper wakes
    with the last situation, its per-TF states and the next
    sequence number, so the id chain survives restarts.
    """

    def __init__(self, journal_dir, *, symbol: str = "XAUUSD",
                 journal_append=None, ring_size: int = 128,
                 swing_n: int = 3):
        self.dir = Path(journal_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.dir / _JOURNAL_FILE
        self.latest_path = self.dir / _LATEST_FILE
        self.symbol = symbol
        self._journal_append = journal_append
        self._swing_n = swing_n
        self._ring: deque = deque(maxlen=max(64, int(ring_size)))
        self._states: dict[str, dict | None] = {
            tf: None for tf in SITUATION_TFS}
        self._clocks: dict = {tf: None for tf in SITUATION_TFS}
        self._clocks["M1"] = None
        self._forecast_outlook: dict | None = None
        self._seq = 0
        self._latest: dict | None = None
        # Pass 1 retained views: the runner-supplied §3.3 blocks,
        # the current relationship list, the provisional live
        # block, and the transition fingerprints the event layer
        # diffs between refreshes (per-TF structure-event episode
        # + volatility state; the location pools being sat in /
        # already taken).
        self._blocks: dict | None = None
        self._relationships: list = []
        self._live: dict | None = None
        self._event_fp: dict[str, dict] = {}
        self._pool_inside = None
        self._pool_taken: set = set()
        self._pool_prev: dict = {"above": None, "below": None}
        self._recover()

    # ------------------------------------------------ recovery
    def _load_records(self) -> list[dict]:
        if not self.jsonl_path.exists():
            return []
        records = []
        try:
            with self.jsonl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:  # noqa: BLE001 — a torn tail
                        continue  # line is skipped, never fatal
                    if isinstance(rec, dict):
                        records.append(rec)
        except OSError:
            return records
        return records

    @staticmethod
    def _apply(records: list[dict], *,
               upto_id: str | None = None) -> dict | None:
        """Replay snapshot + delta records into the situation as
        known at ``upto_id`` (the latest record when None)."""
        current: dict | None = None
        for rec in records:
            kind = rec.get("kind")
            if kind == SNAPSHOT_KIND:
                current = copy.deepcopy(rec.get("situation"))
            elif kind == DELTA_KIND:
                if current is None:
                    current = {
                        "version": SITUATION_VERSION,
                        "situation_id": None,
                        "prev_situation_id": None,
                        "symbol": None,
                        "created_at": None,
                        "clocks": {tf: None
                                   for tf in SITUATION_TFS},
                        "per_tf": {tf: None
                                   for tf in SITUATION_TFS},
                        "regime": None, "location": None,
                        "activity": None, "cross_asset": None,
                        "data_quality": None,
                        "relationships": [],
                        "forecasts": {"outlook": None},
                        "live": None,
                    }
                tf = rec.get("tf")
                if tf in SITUATION_TFS:
                    current["per_tf"][tf] = copy.deepcopy(
                        rec.get("state"))
                if isinstance(rec.get("clocks"), dict):
                    current["clocks"] = copy.deepcopy(rec["clocks"])
                if isinstance(rec.get("relationships"), list):
                    current["relationships"] = copy.deepcopy(
                        rec["relationships"])
                current["situation_id"] = rec.get("situation_id")
                current["prev_situation_id"] = rec.get(
                    "prev_situation_id")
                current["created_at"] = rec.get("ts")
                current["symbol"] = rec.get("symbol") \
                    or current.get("symbol")
            else:
                continue
            if upto_id is not None and current is not None \
                    and current.get("situation_id") == upto_id:
                return current
        # An upto_id that no record carried was never known here:
        # absent stays absent, not "the latest, mislabeled".
        return current if upto_id is None else None

    def _recover(self) -> None:
        records = self._load_records()
        if not records:
            return
        latest = self._apply(records)
        if latest is None:
            return
        self._latest = latest
        self._states = {tf: copy.deepcopy(
            (latest.get("per_tf") or {}).get(tf))
            for tf in SITUATION_TFS}
        self._clocks = dict(latest.get("clocks") or self._clocks)
        self._forecast_outlook = copy.deepcopy(
            (latest.get("forecasts") or {}).get("outlook"))
        blocks = {key: copy.deepcopy(latest.get(key))
                  for key in BLOCK_KEYS
                  if latest.get(key) is not None}
        self._blocks = blocks or None
        self._relationships = copy.deepcopy(
            latest.get("relationships") or [])
        self._live = copy.deepcopy(latest.get("live"))
        self._event_fp = {tf: _event_fingerprint(
            self._states.get(tf)) for tf in SITUATION_TFS}
        self._pool_prev = _nearest_pools(latest.get("location"))
        # Sequence continues from the last record's own seq stamp.
        last = records[-1]
        seq = last.get("seq")
        if seq is None and last.get("kind") == SNAPSHOT_KIND:
            seq = (last.get("situation") or {}).get("seq")
        try:
            self._seq = int(seq) if seq is not None else len(records)
        except (TypeError, ValueError):
            self._seq = len(records)
        self._ring.append(copy.deepcopy(latest))

    # ------------------------------------------------ accessors
    def latest(self) -> dict | None:
        """A COPY of the latest assembled situation (or None)."""
        return copy.deepcopy(self._latest)

    def latest_id(self) -> str | None:
        if self._latest is None:
            return None
        return self._latest.get("situation_id")

    def history(self) -> list[dict]:
        """The in-memory ring of assembled situations, oldest
        first (copies)."""
        return [copy.deepcopy(s) for s in self._ring]

    def reconstruct(self, situation_id: str) -> dict | None:
        """What the system knew at ``situation_id``: the latest
        snapshot at or before it, with later deltas applied in
        chain order (AMENDMENTS-1 §A4.3).  None when the id is
        not in this keeper's journal."""
        if not situation_id:
            return None
        return self._apply(self._load_records(),
                           upto_id=situation_id)

    # ------------------------------------------------ cadence
    def on_tf_close(self, tf: str, stack, *, asof=None,
                    forecast_outlook=..., m1_last_close=None,
                    blocks=..., price: float | None = None,
                    hierarchy_reading=...) -> dict | None:
        """One TF closed a bar.  Returns the journaled record, or
        None when nothing was journaled (unchanged state).

        H1 closes always journal the full snapshot; any other
        situation TF journals a delta only on a real measurement
        change — a changed TF state, a changed relationship
        kind, or a detected event (Pass 1).  ``forecast_outlook``
        uses ``...`` as "keep the carried outlook"; pass a dict
        or None to replace it.  ``blocks`` likewise: ``...``
        keeps the runner-supplied §3.3 blocks, a dict replaces
        them.  ``hierarchy_reading`` (H1 closes): ``...`` keeps
        the retained hierarchy row, a dict replaces it, None
        drops it for this snapshot.
        """
        if tf not in SITUATION_TFS:
            return None
        if forecast_outlook is not ...:
            self._forecast_outlook = copy.deepcopy(forecast_outlook)
        if blocks is not ...:
            self._blocks = copy.deepcopy(blocks)
        clocks = stack_clocks(stack, m1_last_close=m1_last_close)
        self._clocks = clocks
        created = _iso(asof) or _iso(pd.Timestamp.now(tz="UTC"))
        if price is None:
            price = _derive_price(stack)
        if tf == "H1":
            situation = build_situation(
                stack, symbol=self.symbol, asof=asof,
                forecast_outlook=self._forecast_outlook,
                m1_last_close=m1_last_close,
                swing_n=self._swing_n, blocks=self._blocks,
                price=price, live=self._live)
            events: list[dict] = []
            for t in SITUATION_TFS:
                events.extend(_detect_state_events(
                    t, self._states.get(t),
                    situation["per_tf"][t]))
            self._states = {t: situation["per_tf"][t]
                            for t in SITUATION_TFS}
            pairs = _relationships.compute_relationships(
                situation["per_tf"], price=price)
            reading = self._retained_hierarchy() \
                if hierarchy_reading is ... else hierarchy_reading
            situation["relationships"] = \
                _relationships.with_hierarchy(pairs, reading)
            self._relationships = situation["relationships"]
            events.extend(self._refresh_pools(price))
            record = self._publish_snapshot(situation, created)
            self._sync_fingerprints()
            for event in events:
                self._publish_event(event, created)
            return record
        frames = getattr(stack, "frames", {}) or {}
        new_state = build_tf_state(tf, frames.get(tf), asof=asof,
                                   swing_n=self._swing_n)
        events = _detect_state_events(
            tf, self._states.get(tf), new_state)
        changed = _state_changed(self._states.get(tf), new_state)
        self._states[tf] = new_state
        pairs = _relationships.compute_relationships(
            self._states, price=price)
        new_relationships = _relationships.with_hierarchy(
            pairs, self._retained_hierarchy())
        kinds_changed = _relationships.kinds_map(
            new_relationships) != _relationships.kinds_map(
                self._relationships)
        self._relationships = new_relationships
        events.extend(self._refresh_pools(price))
        if not changed and not kinds_changed and not events:
            self._sync_fingerprints()
            return None
        record = self._publish_delta(
            tf, new_state, created,
            relationships=(new_relationships
                           if kinds_changed else None),
            trigger=("event" if events and not changed
                     and not kinds_changed else None))
        self._sync_fingerprints()
        for event in events:
            self._publish_event(event, created)
        return record

    # ------------------------------------------------ Pass 1 views
    def _retained_hierarchy(self) -> dict | None:
        """The hierarchy row of the current relationship list,
        retained across non-H1 refreshes (it is an H1-cadence
        reading)."""
        for rel in self._relationships or []:
            if rel.get("source") == "hierarchy":
                return rel
        return None

    def _refresh_pools(self, price: float | None) -> list[dict]:
        """Liquidity events against the retained location block;
        updates the pool fingerprints either way."""
        location = (self._blocks or {}).get("location")
        events, inside, taken, pools = _liquidity_events(
            self._pool_prev, location, price, self._pool_inside,
            self._pool_taken)
        self._pool_inside = inside
        self._pool_taken = taken
        self._pool_prev = pools
        return events

    def _sync_fingerprints(self) -> None:
        self._event_fp = {tf: _event_fingerprint(
            self._states.get(tf)) for tf in SITUATION_TFS}

    def note_live(self, price: float, *, asof=None,
                  zones_inside=None, positions=None) -> None:
        """The provisional live-reaction block (A4.2): intrabar
        price behaviour inside active zones and open positions,
        fed by the runner's management cycle.  Stored on the
        keeper and stamped into the latest situation + every
        subsequent record — stamped ``provisional`` with its own
        asof, and NEVER written into a closed-bar state field:
        the per-TF states are rebuilt only from closed bars, so
        reaction can never contaminate truth."""
        block = {
            "asof": _iso(asof)
            or _iso(pd.Timestamp.now(tz="UTC")),
            "price": _finite(price),
            "provisional": True,
            "zones": copy.deepcopy(zones_inside or []),
            "positions": copy.deepcopy(positions or []),
        }
        self._live = block
        if self._latest is not None:
            self._latest["live"] = copy.deepcopy(block)

    # ------------------------------------------------ publishing
    def _publish_snapshot(self, situation: dict,
                          created: str) -> dict:
        self._seq += 1
        situation_id = _compact_id(created, self._seq)
        prev = self._latest.get("situation_id") if self._latest \
            else None
        situation["situation_id"] = situation_id
        situation["prev_situation_id"] = prev
        situation["created_at"] = created
        situation["seq"] = self._seq
        record = {"type": "situation", "kind": SNAPSHOT_KIND,
                  "ts": created, "seq": self._seq,
                  "situation_id": situation_id,
                  "prev_situation_id": prev,
                  "symbol": self.symbol, "situation": situation}
        self._commit(record, situation)
        self._write_latest_file(situation)
        return record

    def _publish_delta(self, tf: str, state: dict,
                       created: str, *,
                       relationships: list | None = None,
                       trigger: str | None = None) -> dict:
        self._seq += 1
        situation_id = _compact_id(created, self._seq)
        prev = self._latest.get("situation_id") if self._latest \
            else None
        per_tf = {t: copy.deepcopy(self._states.get(t))
                  for t in SITUATION_TFS}
        assembled = assemble_blocks(per_tf, self._blocks)
        situation = {
            "version": SITUATION_VERSION,
            "situation_id": situation_id,
            "prev_situation_id": prev,
            "symbol": self.symbol,
            "created_at": created,
            "seq": self._seq,
            "clocks": dict(self._clocks),
            "per_tf": per_tf,
            "relationships": copy.deepcopy(self._relationships),
            "forecasts": {"outlook": copy.deepcopy(
                self._forecast_outlook)},
            "live": copy.deepcopy(self._live),
        }
        for key in BLOCK_KEYS:
            situation[key] = assembled[key]
        record = {"type": "situation", "kind": DELTA_KIND,
                  "ts": created, "seq": self._seq,
                  "situation_id": situation_id,
                  "prev_situation_id": prev, "symbol": self.symbol,
                  "tf": tf, "state": copy.deepcopy(state),
                  "clocks": dict(self._clocks)}
        if relationships is not None:
            record["relationships"] = copy.deepcopy(relationships)
        if trigger:
            record["trigger"] = trigger
        self._commit(record, situation)
        return record

    def _publish_event(self, event: dict, created: str) -> dict:
        """A ``situation_event`` record (A4.1), anchored to the
        situation the refresh just published (its id + seq), so
        the transition is reconstructable on the chain.  Events
        consume no sequence number of their own — the chain is
        the snapshot/delta chain; events hang off it."""
        latest = self._latest or {}
        record = {"type": "situation", "kind": EVENT_KIND,
                  "ts": created, "seq": self._seq,
                  "situation_id": latest.get("situation_id"),
                  "prev_situation_id": latest.get(
                      "prev_situation_id"),
                  "symbol": self.symbol,
                  "event_type": event.get("event_type"),
                  "tf": event.get("tf"),
                  "detail": copy.deepcopy(event.get("detail")
                                          or {})}
        self._append_record(record)
        return record

    def _append_record(self, record: dict) -> None:
        """Append to the keeper JSONL and mirror into the live
        journal when wired (best-effort; the JSONL is the
        authority)."""
        line = json.dumps(record, default=str)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if self._journal_append is not None:
            try:
                self._journal_append(record)
            except Exception:  # noqa: BLE001 — mirroring is
                pass  # best-effort; the JSONL is the authority

    def _commit(self, record: dict, situation: dict) -> None:
        """Append the record and hold the assembled situation
        as the latest."""
        self._append_record(record)
        self._latest = situation
        self._ring.append(copy.deepcopy(situation))

    def _write_latest_file(self, situation: dict) -> None:
        """Atomic latest-snapshot file (regime_state.py pattern:
        tmp in the same directory -> fsync -> os.replace ->
        directory fsync), payload checksummed over the canonical
        JSON of the situation itself."""
        canonical = json.dumps(situation, sort_keys=True,
                               default=str)
        payload = {
            "version": SITUATION_VERSION,
            "saved_at": _iso(pd.Timestamp.now(tz="UTC")),
            "checksum": hashlib.sha256(
                canonical.encode("utf-8")).hexdigest(),
            "situation": situation,
        }
        tmp = self.latest_path.with_suffix(
            self.latest_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.latest_path)
        try:
            dir_fd = os.open(self.dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass  # directory fsync is unavailable on some
            # platforms (Windows) — the replace already landed
