"""Dashboard divergence-resolution endpoint (HTTP-free core + thin wrapper).

The dashboard resolves a broker/journal divergence (POST
/api/divergences/resolve) through this module. The CORE
(:func:`resolve_divergence_provenance`) is HTTP-free so tests — and the
CLI — can call it directly with identical semantics.

Ordering guarantees (this order is the whole point)
---------------------------------------------------
1. Validate inputs.
2. IDEMPOTENCY FIRST: a prior clearance with the same idempotency_key
   decides the outcome before any other state is consulted.
3. Claim the key (flock on a per-key lock file) so exactly one worker
   executes for it; losers wait, then re-check idempotency inside the
   claim and return the winner's result as ``duplicate``.
4. CAS: the active divergence's journal seq must equal ``divergence_seq``.
5. Broker recheck. On :class:`BrokerUnavailable` the claim is removed, NO
   clearance is written, NO terminal record is written, and the key
   stays retryable.
6. Append the clearance via :func:`append_journal_record` (the ONLY
   journal write path — no duplicated journal logic here). On append
   failure the claim file STAYS, so a retry reclaims it instead of
   double-appending.

Claim protocol
--------------
``claims_dir/<key>.json`` holds ``{owner_token, pid, fingerprint,
created_at}``. A separate ``claims_dir/<key>.lock`` file carries the
flock, held for the whole operation. The holder takes the flock BEFORE
creating the claim file, so a claim file present while the flock is free
can only mean its holder died (the OS releases flock on death) — that is
an orphan, reclaimed by deleting it and retrying once. Orphanhood is
NEVER decided by age alone: a live holder keeps the flock, however old
the claim file is.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone

from goldenswing.live.journal_lock import append_journal_record
from goldenswing.live.monitor import JOURNAL_RECORD_VERSION

log = logging.getLogger(__name__)

DIVERGENCE_CLEARED = "divergence_cleared"
REQUIRED_FIELDS = ("trade_id", "divergence_seq", "reason", "idempotency_key")

# ---------------------------------------------------------------------------
# Exceptions — each carries .http_status for the HTTP wrapper.
# ---------------------------------------------------------------------------

class DivergenceApiError(Exception):
    http_status: int = 500


class Unauthenticated(DivergenceApiError):
    http_status = 401


class ValidationError(DivergenceApiError, ValueError):
    http_status = 400


class StaleDivergence(DivergenceApiError):
    http_status = 409


class NoActiveDivergence(DivergenceApiError):
    http_status = 404


class IdempotencyKeyReused(DivergenceApiError):
    http_status = 409


class PendingOperation(DivergenceApiError):
    http_status = 409


class BrokerUnavailable(DivergenceApiError):
    http_status = 503


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fingerprint(trade_id: str, divergence_seq: int, reason: str,
                 actor: str, channel: str) -> str:
    payload = "|".join([trade_id, str(divergence_seq), reason, actor, channel])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_divergence_seq(value) -> int:
    """Accept an int, or a digit string; anything else is a validation error."""
    if isinstance(value, bool):
        raise ValidationError("field 'divergence_seq': must be an integer seq, "
                              f"got {value!r}")
    if isinstance(value, int):
        if value < 1:
            raise ValidationError(
                f"field 'divergence_seq': must be >= 1, got {value}")
        return value
    if isinstance(value, str) and value.strip().isdigit():
        n = int(value.strip())
        if n < 1:
            raise ValidationError(
                f"field 'divergence_seq': must be >= 1, got {value!r}")
        return n
    raise ValidationError(
        "field 'divergence_seq': must be an integer seq, got "
        f"{value!r}")


def _validate_str_field(name: str, value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"field '{name}': must be a non-empty string, got {value!r}")
    return value


def _read_journal_records(journal_path: str) -> list[dict]:
    """Raw journal records in file order. No upcasting: upcast rewrites v1
    record types, which would misclassify reconciliation/review records."""
    if not os.path.exists(journal_path):
        return []
    out = []
    with open(journal_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict):
                out.append(d)
    return out


def _find_prior_resolution(journal_path: str, idempotency_key: str) -> dict | None:
    """Prior clearance with this idempotency_key, if one exists."""
    for rec in _read_journal_records(journal_path):
        if (rec.get("type") == "review"
                and rec.get("event") == DIVERGENCE_CLEARED
                and rec.get("idempotency_key") == idempotency_key):
            return rec
    return None


def _find_active_divergence_seq(journal_path: str, trade_id: str) -> int | None:
    """Seq of the latest still-active divergence for ``trade_id``.

    A reconciliation record with needs_review opens a divergence; a later
    divergence_cleared review closes it. Returns None when none is active.
    """
    active: int | None = None
    for rec in _read_journal_records(journal_path):
        if rec.get("trade_id") != trade_id:
            continue
        if rec.get("type") == "reconciliation" and rec.get("needs_review"):
            seq = rec.get("seq")
            if isinstance(seq, int):
                active = seq
        elif (rec.get("type") == "review"
                and rec.get("event") == DIVERGENCE_CLEARED):
            active = None
    return active


def _json_safe(value):
    """Best-effort JSON-safety for broker_state: never let an exotic
    broker payload fail the journal append."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return {"_unserializable": repr(value)}


# ---------------------------------------------------------------------------
# Claim protocol
# ---------------------------------------------------------------------------

_SAFE_KEY = re.compile(r"^[A-Za-z0-9._-]+$")


def _claim_paths(claims_dir: str, idempotency_key: str) -> tuple[str, str]:
    """(claim_file, lock_file). Keys with path-unsafe characters are mapped
    to their sha256 so a key can never escape claims_dir."""
    name = idempotency_key if _SAFE_KEY.match(idempotency_key) else \
        hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    return (os.path.join(claims_dir, name + ".json"),
            os.path.join(claims_dir, name + ".lock"))


class _Claim:
    """A held claim: flock on the lock file + our claim file on disk."""

    def __init__(self, claim_path: str, lock_path: str, lock_fd: int,
                 owner_token: str):
        self.claim_path = claim_path
        self.lock_path = lock_path
        self.lock_fd = lock_fd
        self.owner_token = owner_token
        self._released = False

    def _is_mine(self) -> bool:
        """Verify the claim file still belongs to us before deleting it."""
        try:
            with open(self.claim_path) as f:
                return json.load(f).get("owner_token") == self.owner_token
        except (OSError, ValueError):
            return False

    def remove(self) -> None:
        """Delete ONLY our own claim file (owner_token-checked)."""
        if self._is_mine():
            try:
                os.remove(self.claim_path)
            except OSError:
                pass

    def release(self, *, remove_claim: bool) -> None:
        """Unlock. ``remove_claim=True`` deletes our claim file first;
        False leaves it on disk for a retry to reclaim (unknown outcome)."""
        if self._released:
            return
        self._released = True
        try:
            if remove_claim:
                self.remove()
        finally:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self.lock_fd)


def _acquire_claim(claims_dir: str, idempotency_key: str, fingerprint: str,
                   *, timeout_s: float) -> _Claim:
    """Acquire the per-key claim, waiting up to ``timeout_s`` for a live
    holder to finish. A claim file present with a FREE flock is an orphan
    (holder died — flock is OS-released on death): delete and proceed.
    Raises PendingOperation if a live holder still holds the flock when
    the deadline expires."""
    os.makedirs(claims_dir, exist_ok=True)
    claim_path, lock_path = _claim_paths(claims_dir, idempotency_key)
    deadline = time.monotonic() + max(0.0, timeout_s)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break  # we hold the flock
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise PendingOperation(
                        f"resolution for idempotency_key "
                        f"{idempotency_key!r} is already in progress "
                        f"(claim held by a live holder)")
                time.sleep(0.02)
        # Flock held: any existing claim file is an orphan — its holder is
        # gone, because a live holder holds this flock. Reclaim it.
        if os.path.exists(claim_path):
            try:
                os.remove(claim_path)
            except OSError:
                pass
        owner_token = uuid.uuid4().hex
        fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            payload = {
                "owner_token": owner_token,
                "pid": os.getpid(),
                "fingerprint": fingerprint,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            os.write(fd, (json.dumps(payload) + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        return _Claim(claim_path, lock_path, lock_fd, owner_token)
    except BaseException:
        # Never leak the fd on the failure path. Once a _Claim is
        # returned, its release() owns the fd — this only fires before
        # that point.
        if not acquired:
            try:
                os.close(lock_fd)
            except OSError:
                pass
        else:
            # Flock held but claim creation failed: release flock + fd so
            # the (absent) claim cannot wedge later attempts.
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        raise


# ---------------------------------------------------------------------------
# Core logic (HTTP-free)
# ---------------------------------------------------------------------------

def resolve_divergence_provenance(
    journal_path: str, *,
    trade_id: str,
    divergence_seq,
    reason: str,
    idempotency_key: str,
    actor: str,
    channel: str = "dashboard",
    broker_state=None,
    fetch_broker_state=None,
    now: datetime | None = None,
    claims_dir: str | None = None,
    pending_timeout_s: float = 5.0,
) -> dict:
    """Resolve one divergence with full provenance. See module docstring
    for the ordering guarantees.

    ``broker_state`` is used as-is when given; otherwise
    ``fetch_broker_state()`` is called for the broker recheck. If neither
    is supplied the recheck cannot run -> BrokerUnavailable.
    """
    # (a) Validate.
    trade_id = _validate_str_field("trade_id", trade_id)
    reason = _validate_str_field("reason", reason)
    idempotency_key = _validate_str_field("idempotency_key", idempotency_key)
    actor = _validate_str_field("actor", actor)
    channel = _validate_str_field("channel", channel)
    seq = _normalize_divergence_seq(divergence_seq)
    fingerprint = _fingerprint(trade_id, seq, reason, actor, channel)

    def _idempotency_outcome():
        """(b) Idempotency-first check. Returns a result dict, or None
        when no prior resolution exists, or raises IdempotencyKeyReused
        when the key was used with a different payload."""
        prior = _find_prior_resolution(journal_path, idempotency_key)
        if prior is None:
            return None
        prior_fp = prior.get("fingerprint")
        if prior_fp is not None and prior_fp != fingerprint:
            raise IdempotencyKeyReused(
                f"idempotency_key {idempotency_key!r} was already used "
                f"with a different payload (fingerprint mismatch)")
        if prior_fp is None:
            # Legacy clearance without a fingerprint (e.g. the old
            # resolve_divergence()): same key cannot be proven identical,
            # so refuse rather than risk a double-clear.
            raise IdempotencyKeyReused(
                f"idempotency_key {idempotency_key!r} was already used "
                f"(prior clearance carries no fingerprint)")
        return {
            "status": "duplicate",
            "journal_seq": prior.get("seq"),
            "trade_id": trade_id,
            "divergence_seq": seq,
            "idempotency_key": idempotency_key,
            "actor": actor,
            "channel": channel,
        }

    fast = _idempotency_outcome()
    if fast is not None:
        return fast

    # (d) Claim the key. Losers wait, then re-check idempotency inside the
    # claim and return the winner's result as "duplicate" — never a
    # second append.
    if claims_dir is None:
        claims_dir = os.path.join(
            os.path.dirname(os.path.abspath(journal_path)),
            "divergence_claims")
    claim = _acquire_claim(claims_dir, idempotency_key, fingerprint,
                           timeout_s=pending_timeout_s)
    try:
        again = _idempotency_outcome()
        if again is not None:
            claim.release(remove_claim=True)
            return again

        # (c) CAS: the active divergence's seq must equal divergence_seq.
        active_seq = _find_active_divergence_seq(journal_path, trade_id)
        if active_seq is None:
            claim.release(remove_claim=True)
            raise NoActiveDivergence(
                f"no active divergence for trade_id {trade_id!r}")
        if active_seq != seq:
            claim.release(remove_claim=True)
            raise StaleDivergence(
                f"stale divergence_seq for trade_id {trade_id!r}: "
                f"active divergence is seq {active_seq}, got {seq}")

        # (e) Broker recheck.
        if fetch_broker_state is not None:
            broker = fetch_broker_state()
        elif broker_state is not None:
            broker = broker_state
        else:
            claim.release(remove_claim=True)
            raise BrokerUnavailable(
                "no broker state source provided for recheck")

        # (f) Append the clearance. The claim file STAYS if this raises
        # (unknown outcome — a retry reclaims it and idempotency decides).
        now = now or datetime.now(timezone.utc)
        record = {
            "v": JOURNAL_RECORD_VERSION,
            "type": "review",
            "event": DIVERGENCE_CLEARED,
            "trade_id": trade_id,
            "divergence_seq": seq,
            "reason": reason,
            "idempotency_key": idempotency_key,
            "fingerprint": fingerprint,
            "cleared_by": actor,
            "actor": actor,
            "channel": channel,
            "cleared_at": now.isoformat(),
            "broker_state": _json_safe(broker),
        }
        journal_seq = append_journal_record(journal_path, record)
        claim.release(remove_claim=True)
        return {
            "status": "resolved",
            "journal_seq": journal_seq,
            "trade_id": trade_id,
            "divergence_seq": seq,
            "idempotency_key": idempotency_key,
            "actor": actor,
            "channel": channel,
        }
    except BrokerUnavailable:
        # Broker down: remove ONLY our claim, NO clearance, NO terminal
        # record — the key stays retryable.
        claim.release(remove_claim=True)
        raise
    except DivergenceApiError:
        # Decided errors (stale / no-active / key-reused): claim already
        # released above where raised; release() is idempotent, so a
        # second call here is a safe no-op.
        claim.release(remove_claim=True)
        raise
    except BaseException:
        # Unknown outcome (append failure, crash, ...): claim file STAYS
        # for a retry to reclaim. Only the flock is released.
        claim.release(remove_claim=False)
        raise


# ---------------------------------------------------------------------------
# HTTP wrapper
# ---------------------------------------------------------------------------

def _resolve_actor(actor_from_token, token: str) -> str | None:
    """Verify a dashboard token -> actor name, or None when invalid.

    ``actor_from_token`` may be:
      - a callable verify_token(token) -> actor | None,
      - a dict {token: actor},
      - a single expected-token string (actor becomes "dashboard"),
      - None (no verifier configured -> every token is invalid).
    """
    if callable(actor_from_token):
        try:
            actor = actor_from_token(token)
        except Exception:
            return None
        return actor if isinstance(actor, str) and actor.strip() else None
    if isinstance(actor_from_token, dict):
        actor = actor_from_token.get(token)
        return actor if isinstance(actor, str) and actor.strip() else None
    if isinstance(actor_from_token, str):
        return "dashboard" if token == actor_from_token else None
    return None


def handle_resolve_request(
    headers: dict, body_json: dict, *,
    journal_path: str,
    fetch_broker_state=None,
    actor_from_token=None,
    claims_dir: str | None = None,
) -> tuple[int, dict]:
    """HTTP wrapper for POST /api/divergences/resolve.

    Returns (http_status, response_dict). Never leaks internals: error
    responses carry only the status and a safe message.
    """
    try:
        # Auth: "Bearer <token>" must verify to an actor. No journal work
        # happens before this passes.
        auth = (headers or {}).get("Authorization") if isinstance(headers, dict) else None
        actor = None
        if isinstance(auth, str) and auth.startswith("Bearer "):
            token = auth[len("Bearer "):].strip()
            if token:
                actor = _resolve_actor(actor_from_token, token)
        if actor is None:
            return 401, {"status": "error", "error": "unauthorized"}

        # Body: must be a dict with EXACTLY the 4 required fields.
        if not isinstance(body_json, dict):
            return 400, {"status": "error",
                         "error": "field 'body': must be a JSON object"}
        missing = [f for f in REQUIRED_FIELDS if f not in body_json]
        if missing:
            return 400, {"status": "error",
                         "error": f"field '{missing[0]}': missing"}
        extra = [k for k in body_json if k not in REQUIRED_FIELDS]
        if extra:
            return 400, {"status": "error",
                         "error": f"field '{extra[0]}': unexpected field"}

        result = resolve_divergence_provenance(
            journal_path,
            trade_id=body_json["trade_id"],
            divergence_seq=body_json["divergence_seq"],
            reason=body_json["reason"],
            idempotency_key=body_json["idempotency_key"],
            actor=actor,
            channel="dashboard",
            fetch_broker_state=fetch_broker_state,
            claims_dir=claims_dir,
        )
        return 200, result
    except DivergenceApiError as exc:
        status = exc.http_status
        if isinstance(exc, PendingOperation):
            return status, {"status": "pending", "error": str(exc)}
        return status, {"status": "error", "error": str(exc)}
    except Exception:
        log.exception("divergence resolve failed")
        return 500, {"status": "error", "error": "internal error"}
