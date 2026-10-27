"""SQLite-backed setup identity and atomic trigger claims.

The database is a shared state file for processes on the same EC2 host. It
uses SQLite transactions and a UNIQUE setup_id; in-memory caches are never the
source of truth. A claimed trigger is not automatically released on timeout:
callers must reconcile broker state first, or a crash after order submission
could cause a duplicate order.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SetupStoreError(RuntimeError):
    """Setup state cannot be safely read or changed."""


class SetupStore:
    def __init__(self, path: str | Path, *, timeout_s: float = 10.0):
        self.path = Path(path)
        self.timeout_s = max(0.1, float(timeout_s))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connect() as con:
                con.execute("""CREATE TABLE IF NOT EXISTS hot_scalp_setups (
                    setup_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    direction INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    fencing_token INTEGER NOT NULL DEFAULT 0,
                    claimed_by TEXT,
                    formed_at TEXT NOT NULL,
                    armed_at TEXT NOT NULL,
                    touched_at TEXT,
                    triggered_at TEXT,
                    invalidated_at TEXT,
                    expires_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    terminal_reason TEXT
                )""")
                con.execute("""CREATE TABLE IF NOT EXISTS hot_scalp_events (
                    event_id TEXT PRIMARY KEY,
                    setup_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )""")
                con.execute("CREATE INDEX IF NOT EXISTS idx_hot_scalp_active ON hot_scalp_setups(status, expires_at)")
        except sqlite3.Error as exc:
            raise SetupStoreError(f"setup store initialization failed: {exc}") from exc

    def _connect(self):
        con = sqlite3.connect(str(self.path), timeout=self.timeout_s,
                              isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout = 10000")
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA synchronous = FULL")
        return con

    @staticmethod
    def _now_text(now: Any = None) -> str:
        if now is None:
            return datetime.now(timezone.utc).isoformat()
        # Normalize pandas/Python timestamps and common ISO strings to the
        # same UTC ISO representation before lexical expiry comparisons.
        try:
            if hasattr(now, "to_pydatetime"):
                now = now.to_pydatetime()
            if isinstance(now, datetime):
                dt = now
            else:
                raw = str(now).strip()
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError, OverflowError):
            # Non-time diagnostic strings are not silently parsed as valid
            # timestamps; callers should only pass real timestamps here.
            return str(now)

    def arm(self, setup: dict) -> dict:
        required = ("setup_id", "symbol", "strategy", "direction",
                    "formation_ts", "armed_ts", "expires_at")
        if any(k not in setup for k in required):
            raise ValueError(f"setup missing required fields: {required}")
        payload = json.dumps(setup, sort_keys=True, separators=(",", ":"),
                             default=str)
        now = self._now_text(setup.get("updated_at") or setup.get("armed_ts"))
        try:
            with self._connect() as con:
                con.execute("BEGIN IMMEDIATE")
                cur = con.execute("""INSERT OR IGNORE INTO hot_scalp_setups
                    (setup_id, symbol, strategy, direction, status, version,
                     formed_at, armed_at, expires_at, updated_at, payload_json)
                    VALUES (?, ?, ?, ?, 'armed', 1, ?, ?, ?, ?, ?)""",
                    (str(setup["setup_id"]), str(setup["symbol"]),
                     str(setup["strategy"]), int(setup["direction"]),
                     str(setup["formation_ts"]), str(setup["armed_ts"]),
                     str(setup["expires_at"]), now, payload))
                row = con.execute("SELECT status, version FROM hot_scalp_setups WHERE setup_id=?",
                                   (str(setup["setup_id"]),)).fetchone()
                con.execute("COMMIT")
            return {"inserted": cur.rowcount == 1,
                    "status": str(row["status"]),
                    "version": int(row["version"])}
        except sqlite3.Error as exc:
            raise SetupStoreError(f"could not arm setup: {exc}") from exc

    def active(self) -> list[dict]:
        try:
            with self._connect() as con:
                rows = con.execute("""SELECT setup_id, symbol, strategy,
                    direction, status, version, fencing_token, claimed_by,
                    formed_at, armed_at, touched_at, triggered_at,
                    invalidated_at, expires_at, updated_at, payload_json
                    FROM hot_scalp_setups WHERE status IN ('armed','touched')
                    ORDER BY formed_at, setup_id""").fetchall()
            out = []
            for row in rows:
                item = dict(row)
                item["payload"] = json.loads(item.pop("payload_json"))
                out.append(item)
            return out
        except (sqlite3.Error, ValueError) as exc:
            raise SetupStoreError(f"could not read active setups: {exc}") from exc

    def mark_touched(self, setup_id: str, now: Any) -> bool:
        try:
            with self._connect() as con:
                cur = con.execute("""UPDATE hot_scalp_setups
                    SET status='touched', touched_at=COALESCE(touched_at, ?),
                        updated_at=?, version=version+1
                    WHERE setup_id=? AND status='armed'""",
                    (self._now_text(now), self._now_text(now), str(setup_id)))
            return cur.rowcount == 1
        except sqlite3.Error as exc:
            raise SetupStoreError(f"could not mark setup touched: {exc}") from exc

    def terminate(self, setup_id: str, status: str, reason: str, now: Any) -> bool:
        allowed = {"expired", "invalidated", "blocked", "shadow_triggered", "order_reconciled"}
        if status not in allowed:
            raise ValueError(f"invalid terminal status: {status}")
        try:
            with self._connect() as con:
                cur = con.execute("""UPDATE hot_scalp_setups
                    SET status=?, terminal_reason=?, updated_at=?, version=version+1,
                        invalidated_at=CASE WHEN ?='invalidated' THEN ? ELSE invalidated_at END
                    WHERE setup_id=? AND status IN ('armed','touched','claimed')""",
                    (status, str(reason), self._now_text(now), status,
                     self._now_text(now), str(setup_id)))
            return cur.rowcount == 1
        except sqlite3.Error as exc:
            raise SetupStoreError(f"could not terminate setup: {exc}") from exc

    def claim_trigger(self, setup_id: str, claimant: str, now: Any) -> dict | None:
        """Atomically claim a touched setup once; return its fencing token."""
        try:
            with self._connect() as con:
                con.execute("BEGIN IMMEDIATE")
                row = con.execute("SELECT status, version, fencing_token, expires_at FROM hot_scalp_setups WHERE setup_id=?",
                                  (str(setup_id),)).fetchone()
                if row is None or row["status"] != "touched":
                    con.execute("ROLLBACK")
                    return None
                now_text = self._now_text(now)
                if str(row["expires_at"]) <= now_text:
                    con.execute("UPDATE hot_scalp_setups SET status='expired', terminal_reason='expired_before_trigger', updated_at=?, version=version+1 WHERE setup_id=? AND status='touched'",
                                (now_text, str(setup_id)))
                    con.execute("COMMIT")
                    return None
                token = int(row["fencing_token"]) + 1
                cur = con.execute("""UPDATE hot_scalp_setups
                    SET status='claimed', claimed_by=?, fencing_token=?,
                        triggered_at=?, updated_at=?, version=version+1
                    WHERE setup_id=? AND status='touched' AND version=?""",
                    (str(claimant), token, now_text, now_text,
                     str(setup_id), int(row["version"])))
                if cur.rowcount != 1:
                    con.execute("ROLLBACK")
                    return None
                con.execute("COMMIT")
            return {"setup_id": str(setup_id), "claimant": str(claimant),
                    "fencing_token": token}
        except sqlite3.Error as exc:
            raise SetupStoreError(f"could not claim trigger: {exc}") from exc

    def write_event_once(self, *, event_id: str, setup_id: str,
                         event_type: str, now: Any, payload: dict) -> bool:
        try:
            with self._connect() as con:
                cur = con.execute("""INSERT OR IGNORE INTO hot_scalp_events
                    (event_id, setup_id, event_type, created_at, payload_json)
                    VALUES (?, ?, ?, ?, ?)""",
                    (str(event_id), str(setup_id), str(event_type),
                     self._now_text(now), json.dumps(payload, sort_keys=True,
                                                      separators=(",", ":"),
                                                      default=str)))
            return cur.rowcount == 1
        except sqlite3.Error as exc:
            raise SetupStoreError(f"could not persist event: {exc}") from exc

    def finish_shadow_trigger(self, *, setup_id: str, claimant: str,
                              fencing_token: int, now: Any) -> bool:
        """Close a shadow-only trigger. This is not used to authorize orders."""
        try:
            with self._connect() as con:
                cur = con.execute("""UPDATE hot_scalp_setups
                    SET status='shadow_triggered', updated_at=?, version=version+1
                    WHERE setup_id=? AND status='claimed' AND claimed_by=?
                      AND fencing_token=?""",
                    (self._now_text(now), str(setup_id), str(claimant),
                     int(fencing_token)))
            return cur.rowcount == 1
        except sqlite3.Error as exc:
            raise SetupStoreError(f"could not finish shadow trigger: {exc}") from exc

    def resolve_claim_after_broker_reconciliation(self, *, setup_id: str,
                                                  fencing_token: int,
                                                  broker_order_exists: bool,
                                                  now: Any) -> bool:
        """Resolve a crashed execution claim only after an authoritative broker lookup.

        ``broker_order_exists=False`` means an order lookup was completed and
        positively found no matching order, not that the lookup failed.
        Ambiguous/unknown results are deliberately rejected.
        """
        if not isinstance(broker_order_exists, bool):
            raise ValueError("broker_order_exists must be an authoritative bool")
        target = "order_reconciled" if broker_order_exists else "touched"
        try:
            with self._connect() as con:
                cur = con.execute("""UPDATE hot_scalp_setups
                    SET status=?, claimed_by=NULL, updated_at=?, version=version+1,
                        terminal_reason=CASE WHEN ?='order_reconciled'
                            THEN 'broker_confirmed_order_exists' ELSE NULL END
                    WHERE setup_id=? AND status='claimed' AND fencing_token=?""",
                    (target, self._now_text(now), target, str(setup_id),
                     int(fencing_token)))
            return cur.rowcount == 1
        except sqlite3.Error as exc:
            raise SetupStoreError(f"could not resolve claim: {exc}") from exc
