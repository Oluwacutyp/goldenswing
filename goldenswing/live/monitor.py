"""Divergence monitor: backtest baselines vs live paper performance.

The backtest makes claims (85% zone respect, PF ~1.25, fill assumptions).
Live trading tests those claims. This module watches for divergence and
escalates: NOMINAL -> WATCH -> DEGRADED -> HALT.

Small-sample honesty: H4 produces ~10 trades/month. Halting on 3 bad trades
is noise-trading. Bands use Wilson score intervals (rates) and widen with
small n. Escalation requires SUSTAINED divergence across windows, not one
bad week.

Type I/II balance — two layers, because signals arrive at different speeds:
  FAST layer (zone touches, ~15/month): zone respect is observable on every
      touch, not just closed trades. A streak of 5 consecutive violations has
      P ≈ 0.008% under the 85% baseline — that's a structural break, no need
      to wait for n=20. Fast layer drives WATCH only (never cuts risk alone).
  SLOW layer (closed trades, ~10/month): win rate and avg R need n>=10.
      These drive DEGRADED/HALT because they measure money, not setups.
Asymmetry: Type II (missing real decay) costs capital; Type I (false alarm)
costs opportunity. The fast layer leans toward catching decay early (cheap
to be wrong — WATCH changes nothing); the slow layer demands proof before
touching size (expensive to be wrong — DEGRADED halves risk).

Baselines are the validated H4 FVG numbers. Update them only with new
validated backtests, never by hand-tuning to match live.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from goldenswing.live.journal_lock import (
    journal_transaction, _journal_length, append_journal_record)
from enum import Enum


class Status(Enum):
    NOMINAL = "nominal"      # live matches backtest within bands
    WATCH = "watch"          # one metric outside 95% band — notify
    DEGRADED = "degraded"    # two metrics out, or one outside 99% — halve risk
    HALT = "halt"            # sustained degradation — stop new trades


@dataclass(frozen=True)
class Baseline:
    """Validated backtest claims. Change only via new validated research."""
    name: str = "H4-FVG-limit-v1"
    zone_respect: float = 0.85   # 241/300 touched zones held
    win_rate: float = 0.40       # buf 0.75-1.0 ATR entry test
    avg_r: float = 0.10          # per-trade expectancy
    profit_factor: float = 1.25
    n_zones: int = 300
    n_trades: int = 226


@dataclass
class MetricResult:
    """A version-aware metric: value + explicit data-boundary signal.

    When a metric spans schema versions, some records can't contribute
    (legacy versions lack the field). This object forces the boundary into
    the open instead of letting a silently-biased number through.

    Downstream contract (monitor):
      OK           -> use normally
      THIN         -> use, but bands are already wide at small n
      INSUFFICIENT -> SKIP the check (unknown, not nominal, not degraded)
      N/A          -> SKIP (no records carry the field at all)
    """
    value: float | None
    n: int              # records that contributed
    total: int          # records in scope
    min_n: int = 10

    @property
    def coverage(self) -> float:
        return self.n / self.total if self.total else 0.0

    @property
    def status(self) -> str:
        if self.n == 0:
            return "N/A"
        if self.n < self.min_n:
            return "INSUFFICIENT"
        if self.coverage < 0.5:
            return "THIN"
        return "OK"

    def __bool__(self) -> bool:
        # Truthy = the monitor would build a check from this (OK or THIN).
        # INSUFFICIENT/N/A are skipped downstream.
        return self.status in ("OK", "THIN")


@dataclass
class LiveStats:
    zones_touched: int = 0
    zones_respected: int = 0
    consecutive_violations: int = 0   # fast-layer streak signal
    max_consecutive_violations: int = 0
    trades: int = 0
    wins: int = 0
    total_r: float = 0.0
    gross_win_r: float = 0.0
    gross_loss_r: float = 0.0   # negative sum
    shadow_fills: int = 0
    shadow_signals: int = 0
    recent_rs: list = field(default_factory=list)  # per-trade R, last 60
    # Per-record field values for version-aware segmentation (parallel to
    # recent_rs, same order, same bound). Each entry is the upcast record dict.
    recent_records: list = field(default_factory=list)
    # Field availability across the FULL journal (rebuilt on replay).
    field_counts: dict = field(default_factory=dict)
    # Execution quality (v2 fill events): realized vs theoretical R.
    # execution_cost_r = sum(r_theoretical - r_multiple): positive = slippage
    # ate into results. None-aware: legacy v1 trades contribute 0 (unknown,
    # not zero — but the difference is unmeasurable, so they are excluded
    # from the average via trades_with_exec_data).
    total_execution_cost_r: float = 0.0
    trades_with_exec_data: int = 0
    # Out-of-band broker events reconciled (blind-window closes etc.).
    reconciliations: int = 0

    def record_zone_outcome(self, respected: bool):
        self.zones_touched += 1
        if respected:
            self.zones_respected += 1
            self.consecutive_violations = 0
        else:
            self.consecutive_violations += 1
            self.max_consecutive_violations = max(
                self.max_consecutive_violations, self.consecutive_violations)

    def record_trade(self, r_multiple: float, record: dict | None = None):
        """Record a closed trade. Call for every trade, probation or not."""
        self.trades += 1
        self.total_r += r_multiple
        if r_multiple > 0:
            self.wins += 1
            self.gross_win_r += r_multiple
        else:
            self.gross_loss_r += r_multiple
        self.recent_rs.append(r_multiple)
        if len(self.recent_rs) > 60:
            self.recent_rs.pop(0)
        if record is not None:
            self.recent_records.append(record)
            if len(self.recent_records) > 200:
                self.recent_records.pop(0)
            for k in record:
                if k not in ("seq", "ts_closed", "ts", "v", "type", "_note"):
                    self.field_counts[k] = self.field_counts.get(k, 0) + 1
            # Execution cost: only when the trade carries fill provenance
            # AND was closed under observation. Legacy v1 (fill_seqs == [])
            # and broker-reconciled closes are excluded — unknown cost,
            # not zero cost.
            if record.get("fill_seqs") and record.get("exit_provenance", "observed") == "observed":
                r_theo = record.get("r_theoretical")
                if r_theo is not None:
                    self.total_execution_cost_r += r_theo - r_multiple
                    self.trades_with_exec_data += 1

    @property
    def avg_execution_cost_r(self) -> float | None:
        """Mean slippage cost per measured trade, in R. None if no v2 data."""
        if not self.trades_with_exec_data:
            return None
        return self.total_execution_cost_r / self.trades_with_exec_data

    @classmethod
    def from_journal(cls, journal_path: str,
                     snapshot_path: str | None = None) -> "LiveStats":
        """Rebuild from the trade journal (single source of truth).

        With a valid snapshot: load counters from the snapshot, replay only
        the journal tail after snapshot.journal_seq — O(tail), not O(history).
        Without/corrupt snapshot, or snapshot newer than journal: full replay.
        Full replay is always correct; the snapshot is purely a speed cache.

        At H4 frequency (~115 trades/yr), even full replay of a decade is
        ~1k lines / milliseconds. Snapshots matter if frequency ever rises.
        The journal itself is never compacted or deleted — it's the audit trail.
        """
        stats = cls()
        total = _journal_length(journal_path)

        start_seq = 0
        if snapshot_path and os.path.exists(snapshot_path):
            try:
                with open(snapshot_path) as f:
                    snap = json.load(f)
                if _snapshot_usable(snap, total):
                    stats.trades = int(snap["trades"])
                    stats.wins = int(snap["wins"])
                    stats.total_r = float(snap["total_r"])
                    stats.gross_win_r = float(snap["gross_win_r"])
                    stats.gross_loss_r = float(snap["gross_loss_r"])
                    stats.recent_rs = [float(r) for r in snap["recent_rs"]]
                    stats.recent_records = snap.get("recent_records", [])
                    stats.field_counts = snap.get("field_counts", {})
                    stats.total_execution_cost_r = float(snap.get("total_execution_cost_r", 0.0))
                    stats.trades_with_exec_data = int(snap.get("trades_with_exec_data", 0))
                    start_seq = int(snap["journal_seq"])
                # else: stale/future snapshot -> fall through to full replay
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass  # corrupt snapshot -> full replay below

        if total == 0:
            return stats
        with open(journal_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue  # corrupt line -> skip, never abort rebuild
                try:
                    d = upcast_trade_record(d)
                except ValueError:
                    raise  # future-version record -> fail closed, no silent skip
                if int(d.get("seq", 0)) <= start_seq:
                    continue  # already covered by snapshot
                # Three record types, one journal:
                #   "trade"         -> P&L projection (r_multiple on actual fills)
                #   "fill"          -> execution-quality evidence (slippage audit)
                #   "reconciliation"-> out-of-band broker facts (blind-window
                #                      closes/amendments). Never moves P&L
                #                      directly — the linked trade summary is
                #                      the single P&L authority.
                #   "reconnection"  -> commit marker for a finished recovery
                #                      sequence. Audit only.
                rtype = d.get("type", "trade")
                if rtype == "fill":
                    stats.record_fill(d)
                    continue
                if rtype in ("reconciliation", "reconnection", "review"):
                    stats.reconciliations += 1
                    continue
                r = d.get("r_multiple")
                if r is None:
                    continue
                stats.record_trade(float(r), record=d)
        return stats

    def record_fill(self, fill: dict):
        """Tally a fill event for execution-quality stats. Fills never touch
        P&L counters — the v2 trade summary is the sole P&L authority."""
        self.field_counts["__fills"] = self.field_counts.get("__fills", 0) + 1

    @property
    def respect_rate(self): return self.zones_respected / self.zones_touched if self.zones_touched else 0
    @property
    def win_rate(self): return self.wins / self.trades if self.trades else 0
    @property
    def avg_r(self): return self.total_r / self.trades if self.trades else 0
    @property
    def pf(self):
        return self.gross_win_r / -self.gross_loss_r if self.gross_loss_r < 0 else float('inf')
    @property
    def fill_rate(self): return self.shadow_fills / self.shadow_signals if self.shadow_signals else 0

    def segmented_win_rate(self, field: str, predicate=None,
                           min_n: int = 10) -> MetricResult:
        """Version-aware win rate: only records carrying `field` contribute.

        Legacy records lacking the field are EXCLUDED — never mixed in with
        a guessed value, never silently dropped from the denominator either.
        The returned MetricResult carries n, coverage, and status so the
        monitor can distinguish "edge is fine" from "not enough data yet."

        Example (once v2 adds confluence_score):
            live.segmented_win_rate("confluence_score", lambda s: s >= 3)
        """
        matching = [r for r in self.recent_records
                    if field in r and (predicate is None or predicate(r[field]))]
        n = len(matching)
        if n == 0:
            return MetricResult(None, 0, len(self.recent_records), min_n)
        wins = sum(1 for r in matching if r.get("r_multiple", 0) > 0)
        return MetricResult(wins / n, n, len(self.recent_records), min_n)


# ---------------------------------------------------------------------------
# Schema versioning.
#
# Journal records are IMMUTABLE HISTORY: old records keep their shape forever.
# When the record structure changes, bump JOURNAL_RECORD_VERSION and extend
# upcast_trade_record() — old records are migrated on read, never rewritten.
# A record NEWER than the code is a downgrade scenario -> fail closed.
#
# Snapshots are DISPOSABLE CACHE: on schema mismatch, discard and full-replay.
# Upcasting a cache buys nothing (replay is cheap); discarding is always safe.
# ---------------------------------------------------------------------------
JOURNAL_RECORD_VERSION = 2
SNAPSHOT_SCHEMA_VERSION = 1


def upcast_trade_record(d: dict) -> dict:
    """Migrate an old journal record to the current shape. Never mutates input.

    DECISION RULE — what belongs here vs in LiveStats projection logic:
      UPCASTER (this function): pure structural translation. Every transform
        must be deterministic from the old record's own fields, and must
        satisfy: project_current(upcast(r_old)) == project_old(r_old) for all
        pre-existing metrics. Field renames, unit conversions, splitting or
        merging fields, computing a derived field from recorded facts — fine.
      PROJECTION (LiveStats): every change to HOW metrics are calculated.
        New formula, new filter, new weighting — applies uniformly to
        current-shape records.
      FORBIDDEN IN UPCASTER: estimating/fabricating data old records never
        had. If new projection logic needs a field v1 records lack, the
        PROJECTION branches on version (or marks the metric prospective-only).
        Filling a modeled guess in the upcaster rewrites history — replay
        would no longer reflect what actually happened.

    The projection-equivalence test (tests/test_journal_upcasting.py) enforces
    this mechanically: same trades in old vs new shape must yield identical
    LiveStats on the overlapping metric set.
    """
    v = d.get("v", 1)  # v1 records predate the version field
    if not isinstance(v, int) or v < 1:
        raise ValueError(f"invalid trade record version: {v!r} — refusing to guess")
    if v > JOURNAL_RECORD_VERSION:
        raise ValueError(
            f"trade record v{v} newer than code v{JOURNAL_RECORD_VERSION} — "
            f"refusing to interpret future data (downgrade suspected)")
    if v == 1:
        # v1 -> v2: journal gained fill-level events. v1 has trade-level records
        # only, so the upcast marks them type="trade" and maps entry/exit onto
        # both realized and theoretical fields. Legacy slippage is UNKNOWN —
        # not fabricated: r_theoretical = r_multiple means "no measured
        # execution cost," which is the honest default, not a guess.
        d = {**d, "v": 2, "type": "trade",
             "trade_id": f"legacy-{d.get('seq', '?')}",
             "qty": None,
             "vwap_entry": d.get("entry"), "vwap_exit": d.get("exit"),
             "signal_entry": d.get("entry"),
             "r_theoretical": d.get("r_multiple"),
             "fill_seqs": [], "state_at_signal": "unknown"}
    return d
    v = d.get("v", 1)  # v1 records predate the version field
    if not isinstance(v, int) or v < 1:
        raise ValueError(f"invalid trade record version: {v!r} — refusing to guess")
    if v > JOURNAL_RECORD_VERSION:
        raise ValueError(
            f"trade record v{v} newer than code v{JOURNAL_RECORD_VERSION} — "
            f"refusing to interpret future data (downgrade suspected)")
    # v1 is current: no migration needed. Future versions add branches here,
    # e.g.: if v == 1: d = {**d, "v": 2, "strategy": d.get("strategy", "fvg_limit")}
    return d


def _snapshot_usable(snap: dict, journal_len: int) -> bool:
    """A snapshot is usable only if its schema matches AND it doesn't claim
    more history than the journal contains."""
    if snap.get("schema_v", 1) != SNAPSHOT_SCHEMA_VERSION:
        return False  # stale projection format -> discard, full replay
    sseq = int(snap.get("journal_seq", 0))
    return 0 < sseq <= journal_len


def append_fill_to_journal(journal_path: str, trade_id: str, side: str,
                          qty: float, price: float, signal_price: float,
                          state_at_signal: str, state_at_fill: str,
                          ts: str | None = None) -> int:
    """Journal one fill event. Write-ahead: call the moment the broker
    confirms the fill, BEFORE updating any in-memory position state.

    Each fill carries BOTH states: state_at_signal (which priced the position
    — the entry-gate contract) and state_at_fill (contemporaneous monitor
    state). A fill landing during WATCH on a NOMINAL-sized position is
    expected and correct, not a reconciliation conflict — the two fields make
    that auditable instead of ambiguous.

    Returns the journal sequence number.
    """
    from datetime import datetime, timezone
    # Signed slippage: positive = adverse (paid more / received less).
    # Caller passes direction via side; slippage sign resolved at trade close.
    record = {
        "v": JOURNAL_RECORD_VERSION,
        "type": "fill",
        "trade_id": trade_id,
        "ts": ts or datetime.now(timezone.utc).isoformat(),
        "side": side,            # "buy" | "sell"
        "qty": qty,              # partial fills are separate records
        "price": price,          # actual fill
        "signal_price": signal_price,  # intended limit
        "state_at_signal": state_at_signal,
        "state_at_fill": state_at_fill,
    }
    return append_journal_record(journal_path, record)


def append_trade_to_journal(journal_path: str, r_multiple: float,
                            direction: int = 0, entry: float = 0.0,
                            exit_: float = 0.0, ts_closed: str | None = None,
                            trade_id: str | None = None,
                            qty: float | None = None,
                            vwap_entry: float | None = None,
                            vwap_exit: float | None = None,
                            signal_entry: float | None = None,
                            r_theoretical: float | None = None,
                            fill_seqs: list | None = None,
                            state_at_signal: str = "unknown",
                            exit_provenance: str = "observed") -> int:
    """Append one closed trade (v2 trade summary). Write-ahead: journal before
    updating in-memory state.

    r_multiple is computed on ACTUAL fills (VWAP). r_theoretical is the same
    trade at signal prices; the gap is measured execution cost (slippage),
    not modeled. Partial fills: qty is the total actually filled; unfilled
    remainder simply never becomes a fill event — never fabricated.

    exit_provenance: "observed" (normal close, we saw it) or
    "broker_reconciliation" (closed out-of-band during a blind window —
    exit price is broker fact, path unobserved).
    """
    from datetime import datetime, timezone
    record = {
        "v": JOURNAL_RECORD_VERSION,
        "type": "trade",
        "trade_id": trade_id,
        "ts_closed": ts_closed or datetime.now(timezone.utc).isoformat(),
        "direction": direction,
        "entry": entry,   # legacy alias = vwap_entry (kept for readers)
        "exit": exit_,    # legacy alias = vwap_exit
        "r_multiple": r_multiple,
        "qty": qty,
        "vwap_entry": vwap_entry if vwap_entry is not None else entry,
        "vwap_exit": vwap_exit if vwap_exit is not None else exit_,
        "signal_entry": signal_entry if signal_entry is not None else entry,
        "r_theoretical": r_theoretical if r_theoretical is not None else r_multiple,
        "fill_seqs": fill_seqs or [],
        "state_at_signal": state_at_signal,
        "exit_provenance": exit_provenance,
    }
    # The default trade_id derives from seq, so it must be resolved inside
    # the same critical section as the seq assignment — never before the lock.
    os.makedirs(os.path.dirname(os.path.abspath(journal_path)), exist_ok=True)
    with journal_transaction(journal_path):
        seq = _journal_length(journal_path) + 1
        record["seq"] = seq
        if not record["trade_id"]:
            record["trade_id"] = f"t-{seq}"
        with open(journal_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return record["seq"]


def write_snapshot(snapshot_path: str, live: "LiveStats", journal_seq: int) -> None:
    """Persist a materialized projection as a replay cache.

    The snapshot is an OPTIMIZATION, not a source of truth. Atomic write
    (temp + rename) so a crash never leaves a half-written snapshot. If the
    snapshot is missing, corrupt, or newer than the journal, the rebuild
    falls back to full replay — slower, always correct.
    """
    tmp = snapshot_path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(snapshot_path)), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump({
            "schema_v": SNAPSHOT_SCHEMA_VERSION,
            "journal_seq": journal_seq,
            "trades": live.trades,
            "wins": live.wins,
            "total_r": live.total_r,
            "gross_win_r": live.gross_win_r,
            "gross_loss_r": live.gross_loss_r,
            "recent_rs": live.recent_rs[-60:],
            "recent_records": live.recent_records[-200:],
            "field_counts": live.field_counts,
            "total_execution_cost_r": live.total_execution_cost_r,
            "trades_with_exec_data": live.trades_with_exec_data,
        }, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, snapshot_path)  # atomic on POSIX



def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial rate. Safe at small n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class Check:
    metric: str
    live: float
    baseline: float
    low_95: float
    high_95: float
    low_99: float
    high_99: float
    n: int
    # None = core metric (full escalation rights). Otherwise the
    # MetricResult.status ("OK"/"THIN"): THIN may trigger WATCH but never
    # DEGRADED/HALT on its own — a new segment hasn't earned automatic
    # risk cuts yet; a dramatic deviation gets human eyes, not auto-punishment.
    segment_status: str | None = None
    # False during graduation quarantine: an OK segment earns DEGRADED/HALT
    # rights only after SEGMENT_GRADUATION_CONFIRMATIONS consecutive
    # OK-and-divergent evaluations with new data. Prevents the THIN->OK
    # transition moment itself from triggering escalation on boundary noise.
    escalation_eligible: bool = True

    @property
    def outside_95(self):
        """Divergence: the backtest claim is incompatible with live observations."""
        return self.baseline < self.low_95 or self.baseline > self.high_95
    @property
    def outside_99(self):
        return self.baseline < self.low_99 or self.baseline > self.high_99
    @property
    def worse(self): return self.live < self.baseline  # only downside matters


class DivergenceMonitor:
    """Compares live paper stats against backtest baselines."""

    MIN_TRADES_FOR_CHECK = 10   # below this: insufficient data, always NOMINAL
    MIN_ZONES_FOR_CHECK = 20
    # Graduation hysteresis: a segment newly reaching OK must show divergence
    # across this many consecutive OK evaluations (with new trades) before it
    # can trigger DEGRADED/HALT. Mirrors RECOVERY_CLEAN_CHECKS = 2.
    SEGMENT_GRADUATION_CONFIRMATIONS = 2

    def __init__(self, baseline: Baseline | None = None, state_dir: str = "data/monitor",
                 segment_baselines: dict[str, float] | None = None):
        self.baseline = baseline or Baseline()
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.history_path = os.path.join(state_dir, "checks.jsonl")
        # Segment-matched baselines: {"win_rate_score>=3": 0.52}. A segmented
        # metric is compared ONLY against its own segment's backtest baseline —
        # never against the global baseline (different population). No entry
        # here = observe-only, cannot escalate. Populate from segmented
        # backtests when a new schema field goes live.
        self.segment_baselines = segment_baselines or {}

    def check(self, live: LiveStats, segment_checks: list[Check] | None = None) -> tuple[Status, list[Check]]:
        checks = self._build_checks(live)
        # Graduation quarantine: resolve escalation rights for segmented
        # checks BEFORE partitioning. The streak file persists across restarts.
        streaks = self._load_segment_streaks()
        for c in (segment_checks or []):
            streaks = self._update_segment_streak(streaks, c)
            c.escalation_eligible = self._segment_eligible(c, streaks)
            checks.append(c)
        self._save_segment_streaks(streaks)
        self._log(checks, live)

        # FAST layer: 5 consecutive zone violations ≈ 0.008% under the 85%
        # baseline. Structural-break signal — jumps straight to WATCH, never
        # cuts size on its own. Responds in days, not months.
        if live.consecutive_violations >= 5:
            return Status.WATCH, checks

        # SLOW layer: trade outcomes need n>=10 before any judgment.
        if live.trades < self.MIN_TRADES_FOR_CHECK:
            return Status.NOMINAL, checks  # not enough data to judge

        # RESOLUTION PRINCIPLE — strict severity ordering, not voting.
        # Operational states form a total order by risk posture:
        #   HALT > DEGRADED > WATCH > NOMINAL.
        # Each metric votes for a MINIMUM severity; the final state is the max.
        # There are no conflicting signals to arbitrate because the states are
        # nested, not competing: DEGRADED already implies everything WATCH
        # would do. When in doubt the more protective state wins (one smoke
        # detector alarming means evacuate; you don't poll the others).
        # Upside deviations never cancel downside ones: a metric beating its
        # baseline is simply not "worse" and drops out of the bad lists, but
        # it does not rescue a different metric that is genuinely degraded.
        # Note win_rate and avg_R are correlated (same trades) — that is fine:
        # two metrics outside 95% is stronger evidence than one, and a single
        # metric outside 99% triggers on its own regardless of the others.
        escalating = [c for c in checks if c.escalation_eligible]
        watch_only = [c for c in checks
                      if not c.escalation_eligible
                      and c.segment_status in ("THIN", "OK")]

        bad_99 = [c for c in escalating if c.outside_99 and c.worse]
        bad_95 = [c for c in escalating if c.outside_95 and c.worse]
        thin_bad = [c for c in watch_only if c.outside_95 and c.worse]

        if self._sustained_degraded():
            return Status.HALT, checks
        if bad_99 or len(bad_95) >= 2:
            return Status.DEGRADED, checks
        if bad_95 or thin_bad:
            return Status.WATCH, checks
        return Status.NOMINAL, checks

    def _build_checks(self, live: LiveStats) -> list[Check]:
        b = self.baseline
        out = []
        # zone respect (binomial vs baseline rate)
        lo95, hi95 = wilson_interval(live.zones_respected, live.zones_touched, 1.96)
        lo99, hi99 = wilson_interval(live.zones_respected, live.zones_touched, 2.576)
        # Compare live interval against baseline point: divergent if baseline
        # point falls outside the live interval (small-n safe direction).
        out.append(Check("zone_respect", live.respect_rate, b.zone_respect,
                         lo95, hi95, lo99, hi99, live.zones_touched))
        lo95, hi95 = wilson_interval(live.wins, live.trades, 1.96)
        lo99, hi99 = wilson_interval(live.wins, live.trades, 2.576)
        out.append(Check("win_rate", live.win_rate, b.win_rate,
                         lo95, hi95, lo99, hi99, live.trades))
        # avg R: band around LIVE avg_r; divergent if baseline claim is outside.
        # Tolerance shrinks with sqrt(n): +/- 0.30R at n=10 -> +/- 0.10R at n=100.
        tol95 = 0.30 * math.sqrt(10 / max(live.trades, 1))
        tol99 = 0.45 * math.sqrt(10 / max(live.trades, 1))
        out.append(Check("avg_r", live.avg_r, b.avg_r,
                         live.avg_r - tol95, live.avg_r + tol95,
                         live.avg_r - tol99, live.avg_r + tol99, live.trades))
        return out

    def _segment_streak_path(self) -> str:
        return os.path.join(self.state_dir, "segment_streaks.json")

    def _load_segment_streaks(self) -> dict:
        p = self._segment_streak_path()
        if not os.path.exists(p):
            return {}
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {}

    def _save_segment_streaks(self, streaks: dict) -> None:
        tmp = self._segment_streak_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(streaks, f)
        os.replace(tmp, self._segment_streak_path())

    def _update_segment_streak(self, streaks: dict, c: Check) -> dict:
        """Graduation hysteresis bookkeeping. Streak = consecutive OK-and-
        divergent evaluations WITH NEW TRADES since the last count. A clean
        evaluation, or dropping below OK, resets. Stale evaluations (same n,
        no new trades) neither advance nor reset — they carry no new evidence.
        """
        name = c.metric
        divergent = c.outside_95 and c.worse
        if c.segment_status != "OK" or not divergent:
            streaks.pop(name, None)
            return streaks
        entry = streaks.get(name, {"streak": 0, "last_n": 0})
        if c.n != entry["last_n"]:
            entry = {"streak": entry["streak"] + 1, "last_n": c.n}
            streaks[name] = entry
        return streaks

    def _segment_eligible(self, c: Check, streaks: dict) -> bool:
        """Can this segmented check trigger DEGRADED/HALT?"""
        if c.segment_status is None:
            return True   # core metric
        if c.segment_status == "THIN":
            return False  # watch-only, always
        if c.segment_status == "OK":
            # Graduated, but the transition moment itself must not escalate:
            # require a persistent divergent signal, not a single boundary
            # crossing.
            return streaks.get(c.metric, {}).get("streak", 0) >= self.SEGMENT_GRADUATION_CONFIRMATIONS
        return False  # INSUFFICIENT/N/A never arrive as checks

    def check_segmented(self, result: MetricResult, baseline: float,
                        name: str) -> Check | None:
        """Build a Check from a version-aware MetricResult.

        INSUFFICIENT/N/A -> returns None: the metric is SKIPPED, not treated
        as nominal (which would hide decay) nor as degraded (which would
        punish new fields for being new). The absence of a check IS the
        signal — visible in reports as 'no data yet.'
        The Check carries the MetricResult status so the slow layer can
        grant THIN watch-only escalation rights (never DEGRADED/HALT).
        """
        if result.status in ("INSUFFICIENT", "N/A"):
            return None
        lo95, hi95 = wilson_interval(
            round(result.value * result.n), result.n, 1.96)
        lo99, hi99 = wilson_interval(
            round(result.value * result.n), result.n, 2.576)
        return Check(name, result.value, baseline,
                     lo95, hi95, lo99, hi99, result.n,
                     segment_status=result.status)

    def check_segmented_metric(self, live: LiveStats, field: str,
                               predicate, name: str,
                               min_n: int = 10) -> Check | None:
        """Full pipeline: segment -> MetricResult -> segment-matched baseline.

        Returns None when (a) the segment lacks data, or (b) no baseline
        was registered for this segment — observe-only until a segmented
        backtest supplies one. Never falls back to the global baseline.
        """
        result = live.segmented_win_rate(field, predicate, min_n)
        if result.status in ("INSUFFICIENT", "N/A"):
            return None
        baseline = self.segment_baselines.get(name)
        if baseline is None:
            return None
        return self.check_segmented(result, baseline, name)

    def _log(self, checks: list[Check], live: LiveStats):
        with open(self.history_path, "a") as f:
            f.write(json.dumps({
                "trades": live.trades,
                "checks": [{k: getattr(c, k) for k in
                            ("metric", "live", "baseline", "low_95", "high_95",
                             "n", "segment_status", "escalation_eligible")}
                           for c in checks],
            }) + "\n")

    def _sustained_degraded(self) -> bool:
        """HALT only if the last 3 checks were all DEGRADED-or-worse."""
        if not os.path.exists(self.history_path):
            return False
        with open(self.history_path) as f:
            lines = f.readlines()[-3:]
        if len(lines) < 3:
            return False
        # Recompute status for each historical snapshot (lightweight: any
        # check outside 99% downside, or 2+ outside 95% downside).
        # Only escalation-eligible checks count: THIN and quarantined
        # segments must not contribute to a HALT. Legacy lines predate the
        # flag and default to eligible (core metrics always were).
        bad_windows = 0
        for line in lines:
            d = json.loads(line)
            eligible = [c for c in d["checks"]
                        if c.get("escalation_eligible", True)]
            n_bad95 = sum(1 for c in eligible
                          if c["live"] < c["baseline"]
                          and (c["baseline"] < c["low_95"]
                               or c["baseline"] > c["high_95"]))
            if n_bad95 >= 2:
                bad_windows += 1
        return bad_windows >= 3

    def report(self, live: LiveStats) -> str:
        status, checks = self.check(live)
        lines = [f"Monitor [{self.baseline.name}]: {status.value.upper()}",
                 f"  live trades: {live.trades} | zones: {live.zones_touched}"]
        for c in checks:
            flag = " !!" if c.outside_99 and c.worse else (" !" if c.outside_95 and c.worse else (" *" if c.outside_95 else ""))
            seg = ""
            if c.segment_status == "THIN":
                seg = " [THIN: watch-only]"
            elif c.segment_status == "OK" and not c.escalation_eligible:
                seg = " [graduating: watch-only]"
            lines.append(
                f"  {c.metric:<13} live={c.live:.3f} baseline={c.baseline:.3f} "
                f"95%=[{c.low_95:.3f},{c.high_95:.3f}] n={c.n}{flag}{seg}")
        lines.append("  !! = outside 99% (worse) | ! = outside 95% (worse) | * = outside band (better)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recovery controller: deterministic climb-back with hysteresis.
#
# The deadlock: HALT stops new trades, so trade-based metrics freeze and can
# never "recover." Solution: recovery flows through OBSERVATION first, then
# probationary capital.
#
#   HALT --(fast layer clean 20 zones + 7 days)--> DEGRADED (probation)
#   DEGRADED --(all metrics clean, 2 consecutive checks)--> NOMINAL
#   WATCH --(no bad metrics, 1 check)--> NOMINAL
#
# Hysteresis: recovery demands MORE evidence than the trip did.
#   - Trip to DEGRADED: 2 metrics outside 95% (or 1 outside 99%).
#   - Recovery: ALL metrics inside 95%, twice in a row.
# Second HALT within 30 days requires manual clearance (escalation).
# ---------------------------------------------------------------------------

from datetime import datetime, timezone


@dataclass
class RecoveryState:
    status: Status = Status.NOMINAL
    consecutive_clean: int = 0
    halt_since: str | None = None
    halt_timestamps: list = field(default_factory=list)
    probation_start_trades: int = 0
    is_probation: bool = False
    # LiveStats snapshot at probation start — recovery is judged on
    # probation-only trades, not lifetime stats diluted by pre-halt history.
    probation_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "consecutive_clean": self.consecutive_clean,
            "halt_since": self.halt_since,
            "halt_timestamps": self.halt_timestamps,
            "probation_start_trades": self.probation_start_trades,
            "is_probation": self.is_probation,
            "probation_snapshot": self.probation_snapshot,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RecoveryState":
        return cls(
            status=Status(d.get("status", "nominal")),
            consecutive_clean=d.get("consecutive_clean", 0),
            halt_since=d.get("halt_since"),
            halt_timestamps=d.get("halt_timestamps", []),
            probation_start_trades=d.get("probation_start_trades", 0),
            is_probation=d.get("is_probation", False),
            probation_snapshot=d.get("probation_snapshot", {}),
        )


class RecoveryController:
    """Stateful wrapper around DivergenceMonitor. Owns the status."""

    PROBATION_MIN_DAYS = 7
    PROBATION_MIN_ZONES = 20   # fast-layer clean observations before probation
    RECOVERY_CLEAN_CHECKS = 2  # consecutive clean slow checks to exit DEGRADED
    # Probation hurdle (strictly harder than initial DEGRADED recovery):
    # post-halt, the prior is "edge was broken" — need positive evidence,
    # not just "can't rule out baseline." Judged on probation-only trades.
    PROBATION_MIN_TRADES = 20
    PROBATION_MIN_WIN_RATE = 0.35  # vs 0.40 baseline: striking distance, not exact
    PROBATION_MIN_AVG_R = 0.0      # non-negative expectancy: minimum viability

    def __init__(self, monitor: DivergenceMonitor):
        self.monitor = monitor
        self.state_path = os.path.join(monitor.state_dir, "recovery.json")
        self.state = self._load()

    def _load(self) -> RecoveryState:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return RecoveryState.from_dict(json.load(f))
        return RecoveryState()

    def _save(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state.to_dict(), f)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def evaluate(self, live: LiveStats) -> tuple[Status, str]:
        """Advance the state machine. Returns (status, reason)."""
        signal, checks = self.monitor.check(live)
        st = self.state
        reason = ""

        if st.status == Status.HALT:
            return self._recover_from_halt(live)

        if st.status == Status.DEGRADED:
            if st.is_probation:
                return self._probation_check(live)
            bad = [c for c in checks if c.outside_95 and c.worse]
            if not bad and live.trades >= self.monitor.MIN_TRADES_FOR_CHECK:
                st.consecutive_clean += 1
                if st.consecutive_clean >= self.RECOVERY_CLEAN_CHECKS:
                    st.status = Status.NOMINAL
                    st.consecutive_clean = 0
                    reason = (f"recovered: all metrics inside 95% for "
                              f"{self.RECOVERY_CLEAN_CHECKS} consecutive checks")
            else:
                st.consecutive_clean = 0
                if signal == Status.HALT:
                    return self._enter_halt("sustained degradation")
            self._save()
            return st.status, reason or f"degraded: holding at half risk"

        # NOMINAL or WATCH: follow the signal, with WATCH easy to clear
        if signal == Status.NOMINAL:
            st.status = Status.NOMINAL
            st.consecutive_clean = 0
            reason = "metrics nominal"
        elif signal == Status.WATCH:
            st.status = Status.WATCH
            reason = "watch: " + ", ".join(
                c.metric for c in checks if c.outside_95 and c.worse) or "streak"
        elif signal == Status.DEGRADED:
            st.status = Status.DEGRADED
            st.consecutive_clean = 0
            reason = "degraded: risk halved"
        elif signal == Status.HALT:
            return self._enter_halt("sustained degradation")
        self._save()
        return st.status, reason

    def _enter_halt(self, reason: str) -> tuple[Status, str]:
        st = self.state
        st.status = Status.HALT
        st.halt_since = self._now()
        st.halt_timestamps.append(st.halt_since)
        # keep 30-day window
        cutoff = datetime.now(timezone.utc).timestamp() - 30 * 86400
        st.halt_timestamps = [
            t for t in st.halt_timestamps
            if datetime.fromisoformat(t).timestamp() > cutoff
        ]
        st.consecutive_clean = 0
        self._save()
        return Status.HALT, f"HALT: {reason} — new entries stopped"

    def _recover_from_halt(self, live: LiveStats) -> tuple[Status, str]:
        """HALT recovery: observation first, then probationary capital."""
        st = self.state
        # Escalation: second halt in 30 days needs a human.
        if len(st.halt_timestamps) >= 2:
            return Status.HALT, "HALT: second halt in 30d — manual clearance required"

        # 7-day minimum prevents intraday whipsaw.
        halt_dt = datetime.fromisoformat(st.halt_since)
        days_halted = (datetime.now(timezone.utc) - halt_dt).total_seconds() / 86400
        if days_halted < self.PROBATION_MIN_DAYS:
            return Status.HALT, f"HALT: {days_halted:.1f}d elapsed, need {self.PROBATION_MIN_DAYS}d"

        # Fast layer must show structural behavior normalized: zone respect
        # back inside 95% over recent observations. The tracker keeps running
        # during HALT — we get fresh zone data without risking capital.
        _, checks = self.monitor.check(live)
        respect = next((c for c in checks if c.metric == "zone_respect"), None)
        zones_ok = (respect is not None
                    and not (respect.outside_95 and respect.worse)
                    and live.zones_touched >= self.PROBATION_MIN_ZONES)
        if not zones_ok:
            return Status.HALT, "HALT: fast layer not yet normalized"

        # Probation: DEGRADED (half risk). Real proof comes from trade data.
        st.status = Status.DEGRADED
        st.is_probation = True
        st.probation_start_trades = live.trades
        st.probation_snapshot = {"trades": live.trades, "wins": live.wins,
                                "total_r": live.total_r}
        st.consecutive_clean = 0
        self._save()
        return Status.DEGRADED, ("probation: fast layer normalized, "
                                 "half risk until slow layer confirms")

    def _probation_check(self, live: LiveStats) -> tuple[Status, str]:
        """Probation recovery via sliding segmented window.

        Regime drift problem: 20 trades over ~2 months may span two different
        market regimes. Treating them as one homogeneous block can mask a
        failing adaptation (good half + bad half averages to "fine") or reward
        a temporary bounce (bad half + hot half averages to "fine").

        Solution: split the most recent 20 probation trades into two 10-trade
        blocks. Each block must independently clear a relaxed bar; the full 20
        must clear the strict bar; and this must hold for 2 consecutive
        evaluations. A regime shift mid-window fails the recent block, so
        promotion waits until 20 good trades accumulate IN THE NEW REGIME.
        The window slides — no arbitrary reset points.
        """
        st = self.state
        # Probation-only slice (deltas not needed — recent_rs is ordered).
        rs = live.recent_rs[-(live.trades - st.probation_start_trades):]
        rs = rs[-20:]  # sliding: most recent 20
        n = len(rs)
        if n < self.PROBATION_MIN_TRADES:
            return st.status, (f"probation: {n}/{self.PROBATION_MIN_TRADES} "
                               f"trades — gathering evidence")

        def block_stats(b):
            w = sum(1 for r in b if r > 0) / len(b)
            a = sum(b) / len(b)
            return w, a

        half = n // 2
        w1, a1 = block_stats(rs[:half])
        w2, a2 = block_stats(rs[half:])
        w_all, a_all = block_stats(rs)

        # Hard failure in any block: edge not recovered -> human review.
        if w1 < 0.15 or w2 < 0.15 or a1 < -0.3 or a2 < -0.3:
            return self._enter_halt("probation failed: block-level collapse")

        blocks_ok = (w1 >= 0.30 and a1 >= -0.05
                     and w2 >= 0.30 and a2 >= -0.05)
        full_ok = (w_all >= self.PROBATION_MIN_WIN_RATE
                   and a_all >= self.PROBATION_MIN_AVG_R)
        if blocks_ok and full_ok:
            st.consecutive_clean += 1
            if st.consecutive_clean >= self.RECOVERY_CLEAN_CHECKS:
                st.status = Status.NOMINAL
                st.is_probation = False
                st.consecutive_clean = 0
                self._save()
                return st.status, (f"probation complete: {n} trades, "
                                   f"win {w_all:.1%}, avg {a_all:+.3f}R, "
                                   f"blocks consistent — full risk restored")
            self._save()
            return st.status, (f"probation: evidence building "
                               f"({st.consecutive_clean}/{self.RECOVERY_CLEAN_CHECKS})")
        st.consecutive_clean = 0
        self._save()
        weak = []
        if not (w1 >= 0.30 and a1 >= -0.05):
            weak.append(f"block1 win {w1:.0%}/avg {a1:+.2f}R")
        if not (w2 >= 0.30 and a2 >= -0.05):
            weak.append(f"block2 win {w2:.0%}/avg {a2:+.2f}R")
        detail = "; ".join(weak) if weak else f"combined win {w_all:.0%}/avg {a_all:+.2f}R"
        return st.status, (f"probation: inconsistent blocks ({detail}) — "
                           f"holding at half risk, window slides")

    def startup_check(self, journal_path: str) -> tuple[Status, str]:
        """Call once on engine start, before any evaluation.

        Rebuilds LiveStats from the journal and validates state consistency.
        Fail-closed: unknown or inconsistent state -> HALT for human review.
        A disk glitch must never silently resume full risk after a HALT.
        Returns (status, reason) and the rebuilt LiveStats is stored on self.
        """
        live = LiveStats.from_journal(journal_path)
        self.live = live
        st = self.state

        if not os.path.exists(self.state_path):
            # No state file at all: could be first run OR lost state.
            # Distinguish by journal: empty journal = first run (NOMINAL);
            # non-empty journal with no state = lost state (HALT).
            if live.trades == 0:
                # Persist the first-run state: otherwise a crash after the
                # very first trade (but before any later save) would be
                # misread as "lost state" and HALT spuriously.
                self._save()
                return Status.NOMINAL, "first run: no history"
            st.status = Status.HALT
            st.halt_since = self._now()
            st.halt_timestamps.append(st.halt_since)
            self._save()
            return Status.HALT, ("HALT: state file missing with "
                                 f"{live.trades} journaled trades — manual review required")

        # State exists: validate coordinates against the journal.
        if st.probation_start_trades > live.trades:
            st.status = Status.HALT
            self._save()
            return Status.HALT, ("HALT: state/journal mismatch "
                                 f"(probation_start={st.probation_start_trades} > "
                                 f"journal={live.trades}) — manual review required")

        # Block separation integrity: recent_rs must be a suffix of the journal
        # in order. from_journal guarantees this by construction; assert it.
        assert len(live.recent_rs) <= 60
        return st.status, (f"recovered: {live.trades} journaled trades, "
                           f"status={st.status.value}")

    def manual_clear(self, note: str = "") -> tuple[Status, str]:
        """Human override: clear HALT/DEGRADED back to NOMINAL."""
        self.state = RecoveryState()  # full reset
        self._save()
        return Status.NOMINAL, f"manual clearance: {note}"

    def risk_multiplier(self) -> float:
        """Tier 4 reads this: 1.0 nominal, 0.5 degraded/probation, 0.0 halt.

        ENTRY-GATE ONLY: multiplies the size of NEW positions. Open positions
        are never resized on state transitions (see Tier 4 position lifecycle
        contract). A probation-era half-size position stays half-size after
        graduation; HALT blocks entries, it does not liquidate.
        """
        return {Status.NOMINAL: 1.0, Status.WATCH: 1.0,
                Status.DEGRADED: 0.5, Status.HALT: 0.0}[self.state.status]
