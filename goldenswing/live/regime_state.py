"""Persisted regime / boundary state — the hysteresis clock survives restarts.

The anti-whipsaw machine (``live/range.py`` :class:`BoundaryTracker`)
counts H1 bars: breakout confirmation closes (0/1/2), regime-lock
bars remaining, per-edge cooldown bars, sweep wick / reclaim progress.
If those counters lived only in memory, a restart or failover would
reset the hysteresis clock — a half-confirmed breakout would get a
fresh 2-close allowance, a lock would evaporate, a cooldown would
vanish.  The anti-whipsaw would be useless exactly when the engine
is most fragile (restart after a crash / terminal restart).

So the counters persist to disk, journal-adjacent
(``<journal dir>/regime_state.json`` — same directory as the journal,
never ``/tmp``), on every H1 close, BEFORE any order decision that
uses the new counts (the persist happens in the TF-close refresh,
which runs before the confirm / order phases of that close).

Atomicity: write to ``regime_state.json.tmp`` in the SAME directory,
flush + ``fsync`` the file, ``os.replace`` over the live file, then
``fsync`` the directory.  A kill mid-write leaves either the old
file or the new file intact — never a torn file.  Integrity: the
payload carries a SHA-256 checksum over the canonical JSON plus a
``version``.  A load that fails checksum / version / parse is
CORRUPT: the caller rebuilds conservatively (regime re-derived from
bars, lock = 0, confirm count = 0) and logs a WARNING — counts are
never fabricated from a torn file.

``RegimeState`` is the persisted shape.  The tracker converts to /
from it (``BoundaryTracker.to_regime_state`` /
``apply_regime_state``); the runner owns the file (load once at warm
start, save on every TF-close refresh).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

STATE_VERSION = 1


@dataclass
class RegimeState:
    """One persisted snapshot of the regime / boundary machine."""

    # Regime as the runner reports it: TREND / RANGE /
    # BREAKOUT_TREND / HOLD.  Stored lower-case in code
    # ("trend" / "range" / "breakout_trend" / "hold").
    regime: str = "hold"
    # When the current regime started: wall timestamp + the H1 bar
    # close timestamp that switched it.
    regime_since_ts: str | None = None
    regime_since_bar: str | None = None
    # Hysteresis counters (the whole point of this file).
    lock_remaining_h1: int = 0
    breakout_confirm_count: int = 0  # 0/1/2 closes beyond the edge
    breakout_direction: int = 0
    breakout_edge: str | None = None  # "high" / "low"
    breakout_first_close_time: str | None = None
    # Pending sweep wicks: one entry per edge with a live wick:
    # {edge, wick_price, wick_ts, wick_age_m15, reclaim_bars_confirmed}.
    # "reclaim bars confirmed" = M5/M15 bars since the wick that
    # closed back inside the range.
    sweep_pending: list = field(default_factory=list)
    # Cooldown remaining per edge (H1 bars): {"high": n, "low": n}.
    cooldown_remaining_h1: dict = field(default_factory=dict)
    # Last known range (so a restart can rebuild the tracker before
    # the next live_setup refresh recomputes the range from bars).
    last_range_high: float | None = None
    last_range_low: float | None = None
    last_range_mid: float | None = None
    last_range_atr: float | None = None
    # Idempotency: last H1 / M15 bar close processed into the counts
    # above.  On restart, bars at or before these timestamps are
    # skipped — the same bar is never counted twice.
    last_processed_bar_close: str | None = None
    last_processed_m15_close: str | None = None
    # Bookkeeping (not checksummed inputs — saved_at is set at save).
    saved_at: str | None = None
    checksum: str = ""

    def payload(self) -> dict:
        """Checksum input: every field except ``checksum`` itself."""
        return {
            "version": STATE_VERSION,
            "regime": self.regime,
            "regime_since_ts": self.regime_since_ts,
            "regime_since_bar": self.regime_since_bar,
            "lock_remaining_h1": int(self.lock_remaining_h1),
            "breakout_confirm_count": int(self.breakout_confirm_count),
            "breakout_direction": int(self.breakout_direction),
            "breakout_edge": self.breakout_edge,
            "breakout_first_close_time":
                self.breakout_first_close_time,
            "sweep_pending": self.sweep_pending,
            "cooldown_remaining_h1": self.cooldown_remaining_h1,
            "last_range_high": self.last_range_high,
            "last_range_low": self.last_range_low,
            "last_range_mid": self.last_range_mid,
            "last_range_atr": self.last_range_atr,
            "last_processed_bar_close": self.last_processed_bar_close,
            "last_processed_m15_close":
                self.last_processed_m15_close,
            "saved_at": self.saved_at,
        }

    @classmethod
    def from_payload(cls, d: dict) -> "RegimeState":
        return cls(
            regime=str(d.get("regime", "hold")),
            regime_since_ts=d.get("regime_since_ts"),
            regime_since_bar=d.get("regime_since_bar"),
            lock_remaining_h1=int(d.get("lock_remaining_h1", 0)),
            breakout_confirm_count=int(
                d.get("breakout_confirm_count", 0)),
            breakout_direction=int(d.get("breakout_direction", 0)),
            breakout_edge=d.get("breakout_edge"),
            breakout_first_close_time=d.get(
                "breakout_first_close_time"),
            sweep_pending=list(d.get("sweep_pending") or []),
            cooldown_remaining_h1=dict(
                d.get("cooldown_remaining_h1") or {}),
            last_range_high=d.get("last_range_high"),
            last_range_low=d.get("last_range_low"),
            last_range_mid=d.get("last_range_mid"),
            last_range_atr=d.get("last_range_atr"),
            last_processed_bar_close=d.get("last_processed_bar_close"),
            last_processed_m15_close=d.get(
                "last_processed_m15_close"),
            saved_at=d.get("saved_at"),
            checksum=str(d.get("checksum", "")),
        )


def checksum_of(payload: dict) -> str:
    """SHA-256 over canonical JSON (sorted keys, compact separators)."""
    canonical = json.dumps(payload, sort_keys=True,
                           separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_regime_state(path: str | Path, state: RegimeState,
                      *, saved_at: str | None = None) -> Path:
    """Atomically persist ``state`` to ``path``.

    Write temp file in the SAME directory -> flush -> fsync file ->
    ``os.replace`` -> fsync directory.  A kill at any point leaves
    either the old file or the new file intact, never a torn file.
    Returns the final path.  Raises on I/O failure — the caller
    logs a WARNING and keeps running (a persist failure must never
    crash the engine), but the failure is never silent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if saved_at is not None:
        state.saved_at = saved_at
    payload = state.payload()
    payload["checksum"] = checksum_of(payload)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    # fsync the directory so the rename itself is durable.
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass  # directory fsync is best-effort (Windows lacks it)
    return path


def load_regime_state(path: str | Path
                      ) -> tuple[RegimeState | None, str, str]:
    """Load + verify ``path``.

    Returns ``(state, status, message)`` where ``status`` is one of:
    ``"ok"`` (checksum + version verified), ``"missing"`` (no file),
    ``"corrupt"`` (unparseable / bad checksum / bad version — the
    caller rebuilds conservatively, never fabricates counts from
    this file).  Never raises on a bad file — a corrupt state file
    is a WARNING + rebuild, not a crash.
    """
    path = Path(path)
    if not path.exists():
        return None, "missing", f"no regime state file at {path}"
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:  # unreadable / unparseable JSON
        return None, "corrupt", f"regime state unreadable: {exc}"
    if not isinstance(data, dict):
        return None, "corrupt", "regime state is not a JSON object"
    if data.get("version") != STATE_VERSION:
        return None, "corrupt", (
            f"regime state version {data.get('version')!r} != "
            f"expected {STATE_VERSION} — rebuild")
    stored = str(data.get("checksum", ""))
    payload = {k: v for k, v in data.items() if k != "checksum"}
    if checksum_of(payload) != stored:
        return None, "corrupt", (
            "regime state checksum mismatch — torn or tampered "
            "file, rebuilding")
    try:
        return RegimeState.from_payload(data), "ok", \
            "regime state verified"
    except Exception as exc:
        return None, "corrupt", \
            f"regime state fields invalid: {exc}"
