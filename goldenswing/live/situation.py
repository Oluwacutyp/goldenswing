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
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import deque
from pathlib import Path

import pandas as pd

from . import multitf
from .atr_regime import atr_regime as _atr_regime
from .atr_regime import atr_series as _atr_series
from .hierarchy import _pending_swings
from .regime import ATR_CONTRACTING, SHORT_WINDOW, classify_regime

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

SNAPSHOT_KIND = "situation_snapshot"
DELTA_KIND = "situation_delta"

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


def build_situation(stack, *, symbol: str = "XAUUSD", asof=None,
                    situation_id: str = "",
                    prev_situation_id: str | None = None,
                    forecast_outlook: dict | None = None,
                    m1_last_close=None,
                    swing_n: int = 3) -> dict:
    """The full MarketSituation from the whole stack (pure).

    ``forecast_outlook`` is carried VERBATIM (a deep copy — the
    situation is a carrier, never a re-interpretation, SPEC §3.4);
    ``None`` when no outlook exists.  Nothing here raises on a
    short or missing frame: that TF's state says so instead.
    """
    frames = getattr(stack, "frames", {}) or {}
    per_tf = {tf: build_tf_state(tf, frames.get(tf), asof=asof,
                                 swing_n=swing_n)
              for tf in SITUATION_TFS}
    return {
        "version": SITUATION_VERSION,
        "situation_id": situation_id,
        "prev_situation_id": prev_situation_id,
        "symbol": symbol,
        "created_at": _iso(asof),
        "clocks": stack_clocks(stack, m1_last_close=m1_last_close),
        "per_tf": per_tf,
        "forecasts": {"outlook": copy.deepcopy(forecast_outlook)},
    }


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
                        "forecasts": {"outlook": None},
                    }
                tf = rec.get("tf")
                if tf in SITUATION_TFS:
                    current["per_tf"][tf] = copy.deepcopy(
                        rec.get("state"))
                if isinstance(rec.get("clocks"), dict):
                    current["clocks"] = copy.deepcopy(rec["clocks"])
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
                    forecast_outlook=..., m1_last_close=None
                    ) -> dict | None:
        """One TF closed a bar.  Returns the journaled record, or
        None when nothing was journaled (unchanged state).

        H1 closes always journal the full snapshot; any other
        situation TF journals a delta only on a real measurement
        change.  ``forecast_outlook`` uses ``...`` as "keep the
        carried outlook"; pass a dict or None to replace it.
        """
        if tf not in SITUATION_TFS:
            return None
        if forecast_outlook is not ...:
            self._forecast_outlook = copy.deepcopy(forecast_outlook)
        clocks = stack_clocks(stack, m1_last_close=m1_last_close)
        self._clocks = clocks
        created = _iso(asof) or _iso(pd.Timestamp.now(tz="UTC"))
        if tf == "H1":
            situation = build_situation(
                stack, symbol=self.symbol, asof=asof,
                forecast_outlook=self._forecast_outlook,
                m1_last_close=m1_last_close,
                swing_n=self._swing_n)
            self._states = {t: situation["per_tf"][t]
                            for t in SITUATION_TFS}
            return self._publish_snapshot(situation, created)
        frames = getattr(stack, "frames", {}) or {}
        new_state = build_tf_state(tf, frames.get(tf), asof=asof,
                                   swing_n=self._swing_n)
        changed = _state_changed(self._states.get(tf), new_state)
        self._states[tf] = new_state
        if not changed:
            return None
        return self._publish_delta(tf, new_state, created)

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
                       created: str) -> dict:
        self._seq += 1
        situation_id = _compact_id(created, self._seq)
        prev = self._latest.get("situation_id") if self._latest \
            else None
        situation = {
            "version": SITUATION_VERSION,
            "situation_id": situation_id,
            "prev_situation_id": prev,
            "symbol": self.symbol,
            "created_at": created,
            "seq": self._seq,
            "clocks": dict(self._clocks),
            "per_tf": {t: copy.deepcopy(self._states.get(t))
                       for t in SITUATION_TFS},
            "forecasts": {"outlook": copy.deepcopy(
                self._forecast_outlook)},
        }
        record = {"type": "situation", "kind": DELTA_KIND,
                  "ts": created, "seq": self._seq,
                  "situation_id": situation_id,
                  "prev_situation_id": prev, "symbol": self.symbol,
                  "tf": tf, "state": copy.deepcopy(state),
                  "clocks": dict(self._clocks)}
        self._commit(record, situation)
        return record

    def _commit(self, record: dict, situation: dict) -> None:
        """Append to the keeper JSONL, mirror into the live
        journal when wired, and hold the assembled situation."""
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
