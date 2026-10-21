"""Multi-day structural zones: historical levels, immediate price decides.

ROOT CAUSE, confirmed in code (2026-10-16, owner-confirmed): the
engine was STARVED OF DATA — 20 D1 bars, short history, no
multi-year zones, no persistent zone memory, an H1-centric view.
The bias gate and the session were SYMPTOMS, not the cause: with
20 D1 bars the D1 structure bias flips on noise and the zone book
is empty, so the engine looks dead.  The fix is data, not gates:
fetch 3+ years (feed ``_WARM_BARS``: D1 2000, H4 4200, W1 500,
MN1 300), catalog every swing into a persistent store, and serve
the execution path from an in-memory cache.  There is NO 20-bar
cap anywhere in this module.

Real traders read historical candles from days/weeks/years prior:
past levels hold as unmitigated order blocks, liquidity pools form
at equal highs/lows, major turning points echo.  This module is
the engine's historical memory — the WHERE.  It never triggers a
trade on its own: a zone alone fires nothing.  The immediate
price decides WHEN (POI touch + POI confirm family on the M5/M15
entry clock, score >= 60, confluence, manipulation filter, cost
gate, margin — the existing gates, unchanged).

Architecture (three layers, one writer):

1. :class:`ZoneStore` — the DURABLE store (SQLite ``zones.db``,
   journal-adjacent, never ``/tmp``).  Historical swing points
   cataloged once, updated incrementally on each bar close (new
   swing -> insert, touch -> increment touch count, mitigation /
   break -> mark, never delete).  Major zones NEVER expire on time
   alone — only on mitigation or break.  Minor zones fade after
   20 D1 bars / 100 H4 bars untested.
2. :class:`ZoneCache` — the EXECUTION path (in-memory, versioned,
   copy-on-write).  The tick loop NEVER queries SQLite: it
   bisects the cache's sorted list (``O(log n + k)``, sub-ms).
   Zones are price-stationary, so a price move changes which
   zones are nearby, not the zones themselves — ``nearby()``
   re-bisects per call, always fresh.  The snapshot rebuilds
   only on bar closes (new / mitigated / broken zones).
3. Transactions — the single-writer ordered atomic commit (see
   :meth:`ZoneStore.apply`).  The tick loop is SINGLE-THREADED:
   one writer, no concurrent cache writers.  Each transaction:
   snapshot closed bars at tick start (forming bar excluded) ->
   compute ALL outcomes from that SAME snapshot in deterministic
   order (TF slowest first MN1 -> W1 -> D1 -> H4 -> H1 -> M15 ->
   M5, then price ascending) -> persist all outcomes in ONE
   SQLite transaction -> swap the cache reference (version++).
   Readers see the old version until the swap, then the new
   version — never a torn read (the snapshot is immutable).
   If outcome computation crashes, the transaction is discarded,
   the old cache is kept, nothing is persisted — never partial.
   Idempotent: each transaction carries a ``tick_id`` (max closed
   bar timestamp); replaying the same ``tick_id`` re-applies
   nothing (``last_applied_tick_id`` check).

One outcome per zone per transaction: MITIGATED OR BROKEN OR
TOUCHED OR NONE — never both.  The CLOSE governs:

- MITIGATED = close INSIDE the zone + penetration >= 0.5x ATR
  from the near edge (the OB got filled / liquidity taken).
- BROKEN = close BEYOND the far edge, 2 consecutive closes
  beyond, + M15 BOS in the breakout direction (resistance breaks
  UP: BOS = close above the last confirmed M15 swing high;
  support breaks DOWN: close below the last confirmed M15 swing
  low).  A wick beyond the far edge + BOS present + close INSIDE
  is MITIGATED — the close governs, the wick is noted in the
  reason, never double-counted.  (v1 had this BOS direction
  inverted — resistance mapped to a bearish BOS — which would
  have made BROKEN almost never fire on real breaks.  Fixed
  here; v1 was never run.)
- Bars are applied sequentially; the FIRST terminal transition
  (MITIGATED or BROKEN) wins and stops further evaluation for
  that zone in this transaction.
- Cluster rule: if a cluster's slowest-TF zone BROKE, faster-TF
  sub-zones inside its band are also BROKEN (not mitigated) —
  logged "cluster broken, sub-zones broken with parent".

Significance score (0-100, not binary):

- TF weight: MN1 50, W1 45, D1 40, H4 25, H1 10, M15/M5 5.
- Touches: each respected touch (wick into the zone + close
  away) +10, max 30.  A touch that broke the zone (single close
  beyond, returned) = -20 (weakened, once).
- Displacement: move away from the zone after formation >= 2x
  ATR14 = +15, 1-2x = +5, <1x = 0.
- Confluence (split buckets): overlap another TF zone within
  0.5 ATR = +15, liquidity pool (3+ equal points) = +10, round
  number = +5.
- Recency: provisional >= 60 (MAJOR track) -> age <5 D1 +10,
  5-20 +5, older +0 — a MAJOR is never penalised for age alone.
  Else -> age <5 +10, 5-20 +5, 20-100 +0, >100 -10 (minor fades).
- Volume: above-average volume at the swing bar +10.
- MAJOR >= 60, MINOR 40-59, noise < 40.

Cluster / conflict: zones within 0.5 ATR of each other (pairwise
tolerance 0.5 x max(atr1, atr2)).  Same direction -> CLUSTER,
combined score = max + 10 (cap 100), one stronger POI.  Opposite
directions -> CONFLICT: the engine waits for immediate confirm
(tick touch + BOS/FVG/QM on M5/M15), never picks historical
alone.  Example: 2-year-old W1 OB at 4100 (score 90) vs fresh H4
FVG at 4110 (score 70), same direction -> CLUSTER score 100.

Last pattern: each zone keeps ``last_tested`` (date) and
``last_pattern`` (OB/FVG/QM/sweep).  A touch bar whose extreme
exceeded the zone's far edge records "sweep", else "OB".
(Playbook-to-store wiring for QM/FVG patterns is an explicit
follow-up — bar-shape mapping only, never faked.)

Deliberate non-gate: the score is PRIORITISATION, not an arming
gate.  A fresh swing has no touches / displacement yet by
construction, so gating arming on MAJOR-only would blind the
engine to fresh levels — the exact failure the range regime was
built to fix.  Arming stays on POI touch + confirm family; the
zones annotate the setup (``major_zone`` when the armed POI
coincides with a MAJOR zone) and fill /status.
"""
from __future__ import annotations

import bisect
import copy
import logging
import math
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from .multitf import atr14, find_swings

_log = logging.getLogger(__name__)

# TF slowest-first (deterministic transaction order).
TF_ORDER = ["MN1", "W1", "D1", "H4", "H1", "M15", "M5"]
TF_WEIGHT = {"MN1": 50.0, "W1": 45.0, "D1": 40.0, "H4": 25.0,
             "H1": 10.0, "M15": 5.0, "M5": 5.0}
SWING_N = 3
ZONE_HALF_ATR = 0.5         # zone = swing level +/- 0.5 ATR
OVERLAP_ATR = 0.5           # cross-TF overlap / pool / round tolerance
ROUND_STEP = 50.0           # gold round numbers: multiples of 50
MITIGATION_PUSH_ATR = 0.5   # penetration into the zone to mitigate
MAJOR_MIN = 60.0
MINOR_MIN = 40.0
# A2 lifecycle thresholds — the freshness consumption model is the
# single grading source (spec §11 A1/A2); these mirror
# live/freshness.py TIRED_TOUCH / EXHAUSTED_TOUCH exactly.
TIRED_TOUCH = 3
EXHAUSTED_TOUCH = 4
# A1 touch grade (v2): touches CONSUME a zone, they never add to it.
# Untested = a small freshness premium that spends down with each
# respected touch (kept small on purpose: the premium must order
# fresh > tested without re-tiering zones wholesale).  The
# tired/exhausted PENALTIES live in freshness.py (conditional on
# the reaction after the touch) — applied once, there, not here.
TOUCH_GRADE = {0: 5.0, 1: 2.0, 2: 0.0}
# Expiry: H4 untested after 100 H4 bars (~400h); H1/M15/M5
# untested after 20 D1 bars.  D1/W1/MN1 NEVER expire on age.
H4_EXPIRY_HOURS = 100 * 4
INTRADAY_EXPIRY_DAYS = 20.0
EXPIRY_TFS = ("H4", "H1", "M15", "M5")
# Cache radii (in ATR14 of the H1 frame).
NEARBY_RADIUS_ATR = 2.0
MAJOR_PRECOMPUTE_ATR = 5.0
VACUUM_THRESHOLD = 10000    # VACUUM + ANALYZE on open above this many zones
SCHEMA_VERSION = 1


def _ts_iso(ts) -> str:
    try:
        return pd.Timestamp(ts).isoformat()
    except Exception:
        return str(ts)


def _age_d1(swing_ts, now_ts) -> float:
    """Zone age in days from swing timestamp to ``now_ts``."""
    try:
        delta = pd.Timestamp(now_ts) - pd.Timestamp(swing_ts)
        return float(delta.total_seconds() / 86400.0)
    except Exception:
        return 0.0


def _fmt_age(age_d1: float) -> str:
    if age_d1 >= 365:
        return f"{age_d1 / 365:.1f}y"
    if age_d1 >= 1:
        return f"{age_d1:.0f}d"
    return f"{age_d1 * 24:.0f}h"


# ---------------------------------------------------------------- zones

@dataclass
class StructuralZone:
    """One historical level with its significance + lifecycle."""

    tf: str              # "MN1" | "W1" | "D1" | "H4" | "H1" | "M15" | "M5"
    side: str            # "resistance" (swing high) | "support" (swing low)
    level: float
    top: float
    bottom: float
    score: float = 0.0
    tier: str = "noise"  # "MAJOR" | "MINOR" | "noise"
    status: str = "active"  # "active" | "mitigated" | "broken" | "expired" | "noise"
    touches: int = 0
    displacement_atr: float = 0.0
    confluence: bool = False   # overlaps another TF's zone
    liquidity: bool = False    # 3+ equal highs/lows at this level
    round_number: bool = False  # within 0.5 ATR of a round number
    weak_break: bool = False   # a single close beyond, returned (-20)
    age_d1: float = 0.0       # age in D1 bars at last score refresh
    swing_ts: object = None
    volume_ratio: float | None = None  # swing-bar volume / frame mean
    atr: float = 0.0          # ATR14 at formation (stable)
    last_tested: str | None = None  # ISO ts of last touch/mitigation/break
    last_pattern: str | None = None  # "OB" | "sweep" (bar-shape mapping)
    seen_inside: bool = False  # a close inside the zone (no mitigate yet)
    consec_beyond: int = 0    # consecutive closes beyond the far edge
    # --- v2 grading geometry (spec §11 A3/A4) -------------------------
    # Formation departure: max excursion AWAY from the zone achieved
    # before the first touch/inside close, in formation ATR.  Unlike
    # displacement_atr (monotonic max excursion, later travel
    # included), this separates the departure that CREATED the zone
    # from everything price did afterwards (mining F2).
    departure_atr: float = 0.0
    # Base length in bars at formation (A4): consecutive small-range
    # bars ending at the swing whose ranges overlap the band.
    base_bars: int = 0
    _key_cache: tuple | None = field(  # lazy key() memo, compare=False
        default=None, compare=False, repr=False)  # (eq ignores it)

    # ------------------------------------------------ A11 edges
    @property
    def proximal(self) -> float:
        """A11 proximal edge: the band edge nearest incoming price —
        the entry reference (support: top edge; resistance: bottom
        edge).  This engine's zones are swing bands (level ± 0.5 ATR
        at formation), so the base geometry IS the band; P3's zone
        rebuild may refine these to base-candle body edges."""
        return self.top if self.side == "support" else self.bottom

    @property
    def distal(self) -> float:
        """A11 distal edge: the far band edge — the stop reference."""
        return self.bottom if self.side == "support" else self.top

    # ------------------------------------------------ A2 lifecycle
    @property
    def lifecycle(self) -> str:
        """Lifecycle class (spec §11 A2): ``fresh`` | ``tested`` |
        ``used_up``, from the freshness consumption thresholds.
        Behaviour mapping consumed by the decision flow:
        fresh -> resting-limit eligible (A6); tested -> score /
        funding entry only; used_up -> abandoned, not traded even
        on confirmation.  A non-active zone (mitigated / broken /
        expired) is used_up by definition — a broken zone survives
        in the store as its A7 breaker record, never as a trade."""
        if self.status != "active":
            return "used_up"
        if self.touches <= 0:
            return "fresh"
        if self.touches < TIRED_TOUCH:
            return "tested"
        return "used_up"

    @property
    def resting_limit_eligible(self) -> bool:
        """A6: only a FRESH MAJOR zone may hold a resting limit."""
        return self.lifecycle == "fresh" and self.tier == "MAJOR"

    @property
    def key(self) -> tuple:
        """Stable identity: (tf, side, swing_ts).  Unique in the DB.

        Lazily cached: tf/side/swing_ts never mutate after
        construction, and key is read once per zone per bar in the
        per-bar loop — recomputing pd.Timestamp().isoformat() on
        every access was the seed's remaining hotspot.
        """
        cached = self._key_cache
        if cached is None:
            cached = (self.tf, self.side, _ts_iso(self.swing_ts))
            self._key_cache = cached
        return cached

    @property
    def zone_id(self) -> str:
        return f"{self.tf}:{self.side}:{_ts_iso(self.swing_ts)}"

    def to_dict(self) -> dict:
        return {
            "tf": self.tf, "side": self.side, "level": float(self.level),
            "top": float(self.top), "bottom": float(self.bottom),
            "score": float(self.score), "tier": self.tier,
            "status": self.status, "touches": int(self.touches),
            "displacement_atr": float(self.displacement_atr),
            "confluence": bool(self.confluence),
            "liquidity": bool(self.liquidity),
            "round_number": bool(self.round_number),
            "weak_break": bool(self.weak_break),
            "age_d1": float(self.age_d1),
            "swing_ts": _ts_iso(self.swing_ts),
            "volume_ratio": self.volume_ratio,
            "atr": float(self.atr),
            "last_tested": self.last_tested,
            "last_pattern": self.last_pattern,
            # v2 grading surface (A2/A3/A4/A11): the decision flow
            # and the journal read lifecycle + geometry from here.
            "lifecycle": self.lifecycle,
            "departure_atr": float(self.departure_atr),
            "base_bars": int(self.base_bars),
            "proximal": float(self.proximal),
            "distal": float(self.distal),
        }

    def describe(self, *, now=None, cluster: bool = False) -> str:
        """``W1 4100, 2.1y old, score 90, 3 touches, unmitigated``.

        ``now`` recomputes the live age; without it the stored
        ``age_d1`` (age at the last score refresh) is used.
        """
        age = _age_d1(self.swing_ts, now) if now is not None \
            else float(self.age_d1)
        word = "unmitigated" if self.status == "active" else self.status
        out = (f"{self.tf} {self.level:.1f}, {_fmt_age(age)} old, "
               f"score {self.score:.0f}, {self.touches} touches, {word}")
        if cluster:
            out += ", CLUSTER"
        return out


@dataclass
class ZoneOutcome:
    """Exactly one outcome for one zone in one transaction."""

    zone_key: tuple          # (tf, side, swing_ts iso)
    outcome: str             # MITIGATED | BROKEN | TOUCHED | INSERTED | EXPIRED | FADED | ACTIVATED
    reason: str = ""
    detail: str = ""         # human detail (zone describe + note)


def active_zones(zones: list[StructuralZone]) -> list[StructuralZone]:
    """Tradeable/watchable zones: status active (MAJOR + MINOR)."""
    return [z for z in zones if z.status == "active"]


def major_zones(zones: list[StructuralZone]) -> list[StructuralZone]:
    """Active MAJOR zones, score desc."""
    return sorted((z for z in zones
                   if z.status == "active" and z.tier == "MAJOR"),
                  key=lambda z: -z.score)


# ---------------------------------------------------------------- scoring

def score_zone(zone: StructuralZone, others, *, age_d1: float) -> tuple[float, str]:
    """New scoring (replaces the v1 buckets).  Pure function.

    ``others`` = all zones in the working set (for confluence /
    pool).  Returns ``(score, tier)``.  Recency is resolved
    without circularity: provisional (everything except recency)
    >= 60 -> MAJOR track (recency +10/+5/0, never negative);
    else recency = +10/+5/0/-10 by age.
    """
    atr = zone.atr if zone.atr > 0 else 1.0
    provisional = TF_WEIGHT.get(zone.tf, 0.0)
    # A1 (v2): the v1 term paid +10 per touch (cap +30) — the
    # inverted freshness sign.  Touches consume; the grade now pays
    # a freshness premium that spends down (TOUCH_GRADE).  Tired /
    # exhausted penalties are freshness.py's, applied once there.
    provisional += TOUCH_GRADE.get(int(zone.touches), 0.0)
    if zone.weak_break:
        provisional -= 20.0
    # A3 (v2): when the formation departure is known it replaces
    # the monotonic excursion as the displacement input — a zone is
    # validated by the move that LEFT its base (impulse >= 2 ATR
    # full marks; < 1 ATR never validated it).  Legacy rows
    # (departure unknown = 0) fall back to displacement_atr.
    disp = zone.departure_atr if zone.departure_atr > 0 \
        else zone.displacement_atr
    if disp >= 2.0:
        provisional += 15.0
    elif disp >= 1.0:
        provisional += 5.0
    elif zone.departure_atr > 0:
        provisional -= 10.0
    # A4 (v2): base length grades the departure's quality — a short
    # base precedes a strong departure; a long base weakens it.
    # Graded, never a veto (Wyckoff range-edge caveat).
    if zone.base_bars > 6:
        provisional -= 5.0
    elif 0 < zone.base_bars <= 3:
        provisional += 5.0
    confluence = False
    pool_n = 0
    for other in others:
        if other is zone or other.status not in ("active", "noise"):
            continue
        tol = OVERLAP_ATR * max(atr, other.atr if other.atr > 0 else atr)
        if abs(other.level - zone.level) <= tol:
            pool_n += 1
            if other.tf != zone.tf:
                confluence = True
    pool = pool_n + 1 >= 3  # +1 = this zone itself
    nearest_round = round(zone.level / ROUND_STEP) * ROUND_STEP
    round_number = abs(zone.level - nearest_round) <= OVERLAP_ATR * atr
    if confluence:
        provisional += 15.0
    if pool:
        provisional += 10.0
    if round_number:
        provisional += 5.0
    if zone.volume_ratio is not None and zone.volume_ratio > 1.0:
        provisional += 10.0
    if provisional >= MAJOR_MIN:
        recency = 10.0 if age_d1 < 5 else (5.0 if age_d1 < 20 else 0.0)
    else:
        if age_d1 < 5:
            recency = 10.0
        elif age_d1 < 20:
            recency = 5.0
        elif age_d1 <= 100:
            recency = 0.0
        else:
            recency = -10.0
    score = max(0.0, min(100.0, provisional + recency))
    tier = "MAJOR" if score >= MAJOR_MIN else (
        "MINOR" if score >= MINOR_MIN else "noise")
    # Stash the confluence / pool / round flags on the zone for
    # to_dict / describe consumers (scoring is the only writer).
    zone.confluence = confluence
    zone.liquidity = pool
    zone.round_number = round_number
    return score, tier


# ---------------------------------------------------------------- clusters

def find_clusters(zones, *, include_statuses=("active",)) -> list[dict]:
    """Group zones within 0.5 ATR of each other (pairwise tolerance
    0.5 x max(atr1, atr2), transitive sweep by price).

    Same direction -> CLUSTER, combined = max + 10 (cap 100).
    Opposite directions -> CONFLICT (engine waits for immediate
    confirm, never picks historical alone).  Singletons are not
    clusters.  ``include_statuses`` defaults to active-only; the
    cluster rule passes ("active", "broken") so a parent that
    broke THIS transaction still clusters with its sub-zones.
    """
    ordered = sorted((z for z in zones
                      if z.status in include_statuses),
                     key=lambda z: z.level)
    groups: list[list] = []
    cur: list = []
    for z in ordered:
        if not cur:
            cur = [z]
            continue
        prev = cur[-1]
        tol = OVERLAP_ATR * max(
            z.atr if z.atr > 0 else 1.0,
            prev.atr if prev.atr > 0 else 1.0)
        if abs(z.level - prev.level) <= tol:
            cur.append(z)
        else:
            groups.append(cur)
            cur = [z]
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        if len(g) < 2:
            continue
        sides = {z.side for z in g}
        same = len(sides) == 1
        out.append({
            "zones": list(g),
            "same_direction": same,
            "conflict": not same,
            "combined_score": (min(100.0, max(z.score for z in g) + 10.0)
                               if same else None),
            "levels": [z.level for z in g],
        })
    return out


def describe_zone(zone, *, now=None, cluster: bool = False) -> str:
    """Module-level describe (same format as ``zone.describe``)."""
    return zone.describe(now=now, cluster=cluster)


# ---------------------------------------------------------------- store

class ZoneStore:
    """Persistent zone store (SQLite, journal-adjacent ``zones.db``).

    Durable, NOT the execution path: the tick loop never queries
    this store (see :class:`ZoneCache`).  The store is written
    only from :meth:`apply`, which runs on bar closes.

    Transaction model (single writer, copy-on-write):

    - :meth:`compute` is PURE: it reads the in-memory mirror,
      mutates only copies, and returns ``(outcomes, changed,
      new_cursors)``.  A crash inside compute leaves the mirror,
      the DB, and the cache exactly as before.
    - :meth:`persist` writes ``changed`` + cursors + tick id in
      ONE SQLite transaction (atomic).  On success the mirror
      adopts the copies; on failure (rollback) the mirror is
      untouched and the next refresh re-derives the same
      transaction cleanly from the old cursors.
    - :meth:`apply` = compute + persist, the single commit point.
    """

    def __init__(self, path):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._mirror: dict[tuple, StructuralZone] = {}
        self._cursors: dict[str, str] = {}      # tf -> last bar ts (ISO)
        self._swing_marks: dict[str, str] = {}  # tf -> last swing ts checked
        self._last_applied_tick_id: str | None = None
        self._bos_cache: dict | None = None  # per-transaction _bos memo;
        # None outside a transaction (see apply() / _bos()).
        self.vacuumed = False
        self.rtree_available = False
        self._load()
        # VACUUM / ANALYZE on open when the book is large — keeps
        # the R-tree + covering indexes fast on rebuild.
        try:
            n = self._conn.execute(
                "SELECT COUNT(*) FROM zones").fetchone()[0]
            if n > VACUUM_THRESHOLD:
                self._conn.execute("VACUUM")
                self._conn.execute("ANALYZE")
                self.vacuumed = True
                _log.info("zone store: VACUUM + ANALYZE on %d zones",
                          n)
        except Exception as exc:
            _log.warning("zone store VACUUM/ANALYZE skipped: %s", exc)

    # -- schema ------------------------------------------------------

    def _init_schema(self) -> None:
        c = self._conn
        c.execute(
            "CREATE TABLE IF NOT EXISTS zones ("
            "id INTEGER PRIMARY KEY, tf TEXT NOT NULL, "
            "side TEXT NOT NULL, level REAL NOT NULL, "
            "top REAL NOT NULL, bottom REAL NOT NULL, "
            "score REAL NOT NULL DEFAULT 0, tier TEXT NOT NULL, "
            "status TEXT NOT NULL, touches INTEGER NOT NULL DEFAULT 0, "
            "displacement_atr REAL NOT NULL DEFAULT 0, "
            "confluence INTEGER NOT NULL DEFAULT 0, "
            "liquidity INTEGER NOT NULL DEFAULT 0, "
            "round_number INTEGER NOT NULL DEFAULT 0, "
            "weak_break INTEGER NOT NULL DEFAULT 0, "
            "age_d1 REAL NOT NULL DEFAULT 0, "
            "swing_ts TEXT NOT NULL, volume_ratio REAL, atr REAL, "
            "last_tested TEXT, last_pattern TEXT, "
            "seen_inside INTEGER NOT NULL DEFAULT 0, "
            "consec_beyond INTEGER NOT NULL DEFAULT 0, "
            "UNIQUE(tf, side, swing_ts))")
        # v2 migration: formation-geometry columns (A3/A4).  Guarded
        # ALTERs so an existing zones.db (EC2) gains them in place.
        cols = {r["name"] for r in
                c.execute("PRAGMA table_info(zones)").fetchall()}
        if "departure_atr" not in cols:
            c.execute("ALTER TABLE zones ADD COLUMN "
                      "departure_atr REAL NOT NULL DEFAULT 0")
        if "base_bars" not in cols:
            c.execute("ALTER TABLE zones ADD COLUMN "
                      "base_bars INTEGER NOT NULL DEFAULT 0")
        c.execute(
            "CREATE TABLE IF NOT EXISTS meta ("
            "key TEXT PRIMARY KEY, value TEXT)")
        # B-tree: (status, tf, price) for active-by-TF reads.
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_zones_status_tf_level "
            "ON zones(status, tf, level)")
        # B-tree: price for range reads.
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_zones_level ON zones(level)")
        # Covering index for score-sorted active reads.
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_zones_status_score "
            "ON zones(status, score DESC, level)")
        # R-tree for price-range queries on rebuild (RTREE may be
        # unavailable in minimal SQLite builds — fall back to the
        # B-tree price index, never crash).
        try:
            c.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS zones_rtree "
                "USING rtree(id, min_price, max_price)")
            self.rtree_available = True
        except Exception as exc:
            _log.warning("zone store: R-tree unavailable (%s) — "
                         "using B-tree price index", exc)
            self.rtree_available = False
        c.commit()

    def _load(self) -> None:
        rows = self._conn.execute("SELECT * FROM zones").fetchall()
        for row in rows:
            zone = self._row_to_zone(row)
            self._mirror[zone.key] = zone
        for row in self._conn.execute("SELECT key, value FROM meta"):
            # Cursors are TIMESTAMPS (ISO), not frame indices: a
            # re-fetched frame shifts indices, timestamps never
            # move — a re-warm can never double-count a bar.
            if row["key"].startswith("cursor_"):
                self._cursors[row["key"][7:]] = row["value"]
            elif row["key"].startswith("swing_"):
                self._swing_marks[row["key"][6:]] = row["value"]
            elif row["key"] == "last_applied_tick_id":
                self._last_applied_tick_id = row["value"]

    @staticmethod
    def _row_to_zone(row) -> StructuralZone:
        keys = row.keys()
        return StructuralZone(
            tf=row["tf"], side=row["side"], level=float(row["level"]),
            top=float(row["top"]), bottom=float(row["bottom"]),
            score=float(row["score"]), tier=row["tier"],
            status=row["status"], touches=int(row["touches"]),
            displacement_atr=float(row["displacement_atr"]),
            confluence=bool(row["confluence"]),
            liquidity=bool(row["liquidity"]),
            round_number=bool(row["round_number"]),
            weak_break=bool(row["weak_break"]),
            age_d1=float(row["age_d1"]),
            swing_ts=row["swing_ts"],
            volume_ratio=(float(row["volume_ratio"])
                          if row["volume_ratio"] is not None else None),
            atr=float(row["atr"] or 0.0),
            last_tested=row["last_tested"],
            last_pattern=row["last_pattern"],
            seen_inside=bool(row["seen_inside"]),
            consec_beyond=int(row["consec_beyond"]),
            departure_atr=(float(row["departure_atr"])
                           if "departure_atr" in keys else 0.0),
            base_bars=(int(row["base_bars"])
                       if "base_bars" in keys else 0))

    @staticmethod
    def _zone_to_row(zone: StructuralZone) -> tuple:
        return (zone.tf, zone.side, float(zone.level), float(zone.top),
                float(zone.bottom), float(zone.score), zone.tier,
                zone.status, int(zone.touches),
                float(zone.displacement_atr), int(zone.confluence),
                int(zone.liquidity), int(zone.round_number),
                int(zone.weak_break), float(zone.age_d1),
                _ts_iso(zone.swing_ts), zone.volume_ratio,
                float(zone.atr), zone.last_tested, zone.last_pattern,
                int(zone.seen_inside), int(zone.consec_beyond),
                float(zone.departure_atr), int(zone.base_bars))

    # -- reads -------------------------------------------------------

    def active_zones(self) -> list[StructuralZone]:
        """Active zones from the mirror (no SQL on the tick path).

        Expiry is applied at compute time (bar closes), not at
        read time: a zone ages out only when new bars arrive, and
        compute() runs on every bar close.
        """
        return [z for z in self._mirror.values() if z.status == "active"]

    def major_zones(self) -> list[StructuralZone]:
        return major_zones(self.active_zones())

    def load_active_via_rtree(self, price: float, atr: float, *,
                              radius_atr: float = NEARBY_RADIUS_ATR
                              ) -> list[StructuralZone]:
        """Rebuild path: active zones near ``price`` via the R-tree
        (or the B-tree price index when RTREE is unavailable).
        Used when the mirror must be rebuilt from disk without a
        full scan — e.g. cache recovery.  Not on the tick path.
        """
        lo, hi = price - radius_atr * atr, price + radius_atr * atr
        if self.rtree_available:
            rows = self._conn.execute(
                "SELECT z.* FROM zones z JOIN zones_rtree r "
                "ON z.id = r.id WHERE z.status = 'active' "
                "AND r.max_price >= ? AND r.min_price <= ?",
                (lo, hi)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM zones WHERE status = 'active' "
                "AND bottom <= ? AND top >= ?", (hi, lo)).fetchall()
        return [self._row_to_zone(r) for r in rows]

    def bulk_insert(self, zones) -> int:
        """Insert zones in ONE transaction (tests / seeding).

        Existing keys are skipped (INSERT OR IGNORE).  Mirror
        adopts the inserted rows.
        """
        zones = list(zones)
        if not zones:
            return 0
        inserted = 0
        with self._conn:
            for zone in zones:
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO zones (tf, side, level, "
                    "top, bottom, score, tier, status, touches, "
                    "displacement_atr, confluence, liquidity, "
                    "round_number, weak_break, age_d1, swing_ts, "
                    "volume_ratio, atr, last_tested, last_pattern, "
                    "seen_inside, consec_beyond, departure_atr, "
                    "base_bars) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self._zone_to_row(zone))
                if cur.rowcount:
                    inserted += 1
                    self._mirror[zone.key] = zone
                    if self.rtree_available:
                        zid = self._conn.execute(
                            "SELECT id FROM zones WHERE tf = ? AND "
                            "side = ? AND swing_ts = ?",
                            (zone.tf, zone.side,
                             _ts_iso(zone.swing_ts))).fetchone()[0]
                        self._conn.execute(
                            "INSERT OR REPLACE INTO zones_rtree "
                            "(id, min_price, max_price) VALUES (?, ?, ?)",
                            (zid, float(zone.bottom), float(zone.top)))
        return inserted

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # -- transaction: compute (pure) + persist (one txn) -------------

    def apply(self, frames: dict, tick_id) -> list[ZoneOutcome]:
        """Compute + persist one atomic zone transaction.

        ``frames`` = {TF: closed-bars DataFrame} snapshotted at
        tick start (forming bar excluded).  ``tick_id`` = max
        closed-bar timestamp (idempotency key).  Returns the
        outcomes; on ANY failure raises (the caller keeps the old
        cache — never partial).
        """
        # Per-transaction _bos memo: _bos depends only on the (fixed)
        # M15 snapshot + (direction, bar ts), but _apply_bar calls it
        # once per zone per bar — without the memo a seed re-runs
        # find_swings over the whole M15 prefix thousands of times
        # (the full-engine backtest never finished its first
        # refresh).  The cache lives only for this transaction and
        # is cleared in finally, so a later transaction with a
        # different M15 snapshot can never read a stale entry.
        self._bos_cache = {}
        try:
            outcomes, changed, new_marks = self.compute(frames, tick_id)
        finally:
            self._bos_cache = None
        if not changed and not new_marks:
            return outcomes
        self.persist(changed, new_marks, tick_id)
        return outcomes

    def seed(self, frames: dict) -> list[ZoneOutcome]:
        """Replay all frames through :meth:`apply` (tick_id 'seed').

        No-op for bars already processed — cursors persist per TF,
        so a later boot replays only new bars.  Seeding covers
        MN1/W1/D1/H4/H1; M15/M5 are cataloged incrementally from
        live bars onward (replaying 3y of M15/M5 is not worth the
        cost — their zones are intraday, they fade in 20 days).
        """
        return self.apply(frames, "seed")

    def compute(self, frames: dict, tick_id
                ) -> tuple[list[ZoneOutcome], list[StructuralZone],
                           dict[str, dict]]:
        """Pure outcome computation.  Reads the mirror, mutates
        only COPIES.  Never writes.  Returns ``(outcomes, changed,
        new_marks)`` where ``changed`` holds the copies and
        ``new_marks[tf] = {"cursor": last-bar-ts, "swing":
        last-checked-swing-ts}`` (timestamps, never indices — a
        re-fetched frame shifts indices, timestamps never move, so
        a re-warm can never double-count a bar or a swing).
        """
        if tick_id is not None and \
                tick_id == self._last_applied_tick_id:
            return [], [], {}
        outcomes: list[ZoneOutcome] = []
        working: dict[tuple, StructuralZone] = {}
        m15 = frames.get("M15")

        def _working(zone: StructuralZone) -> StructuralZone:
            """Copy-on-touch: the first mutation of a zone in this
            transaction works on a copy; the mirror stays pristine
            until persist()."""
            key = zone.key
            if key not in working:
                working[key] = copy.copy(zone)
            return working[key]

        def _swings(frame):
            """{index -> (side, price)} for swings in the frame
            (computed once per TF per transaction)."""
            try:
                highs, lows = find_swings(
                    frame["high"].to_numpy(dtype=float),
                    frame["low"].to_numpy(dtype=float), SWING_N)
            except Exception:
                return {}
            out = {}
            for idx, price in highs:
                out[int(idx)] = ("resistance", float(price))
            for idx, price in lows:
                out[int(idx)] = ("support", float(price))
            return out

        new_marks: dict[str, dict] = {}
        for tf in TF_ORDER:
            frame = frames.get(tf)
            if frame is None or len(frame) < 2 * SWING_N + 5:
                continue
            if not frame.index.is_monotonic_increasing:
                frame = frame.sort_index()
            last_ts = frame.index[-1]
            cursor_ts = self._cursors.get(tf)
            swing_mark = self._swing_marks.get(tf) or None
            # New bars: timestamps strictly after the cursor.
            if cursor_ts is None:
                new_pos = list(range(len(frame)))
            else:
                try:
                    cut = pd.Timestamp(cursor_ts)
                    new_pos = [p for p, ts in enumerate(frame.index)
                               if pd.Timestamp(ts) > cut]
                except Exception:
                    new_pos = list(range(len(frame)))
            swings = _swings(frame)
            # Active zones of this TF, maintained incrementally:
            # inserts append, terminal zones drop out (identity).
            actives: list[StructuralZone] = []
            for zone in self._mirror.values():
                if zone.tf == tf and zone.status == "active":
                    actives.append(working.get(zone.key, zone))
            for key, zone in working.items():
                if zone.tf == tf and zone.status == "active" \
                        and key not in self._mirror:
                    actives.append(zone)
            max_swing_ts = swing_mark
            for p in new_pos:
                if p < 2 * SWING_N:
                    continue  # swing window [p-6, p] incomplete
                bar = self._bar_at(frame, p)
                just_inserted: set[tuple] = set()
                # 1) Swing confirmed at q = p-3 (3 bars closed
                # after it).  Insert + apply bars q+1..p.
                q = p - 3
                swing_ts = _ts_iso(frame.index[q])
                if (swing_mark is None or swing_ts > swing_mark) \
                        and swings.get(q) is not None:
                    side, price = swings[q]
                    key = (tf, side, swing_ts)
                    if key not in self._mirror and key not in working:
                        zone = self._insert_zone(
                            tf, side, price, frame, q, p, m15,
                            outcomes, working)
                        if zone is not None:
                            just_inserted.add(key)
                            if zone.status == "active":
                                actives.append(zone)
                if max_swing_ts is None or swing_ts > max_swing_ts:
                    max_swing_ts = swing_ts
                # 2) Apply bar p to every other active zone of tf.
                for zone in list(actives):
                    if zone.key in just_inserted:
                        continue
                    w = _working(zone)
                    term, note = self._apply_bar(w, bar, m15)
                    if term is not None:
                        outcomes.append(ZoneOutcome(
                            zone_key=w.key, outcome=term,
                            reason=note,
                            detail=w.describe(now=last_ts)))
                        if term in ("MITIGATED", "BROKEN"):
                            actives[:] = [z for z in actives
                                          if z is not zone]
            # 3) Expiry (untested minors only; majors never expire
            # on age — only on mitigation / break).
            for zone in list(actives):
                if zone.touches != 0:
                    continue
                age_days = _age_d1(zone.swing_ts, last_ts)
                if tf == "H4" and age_days * 24.0 > H4_EXPIRY_HOURS:
                    w = _working(zone)
                    w.status = "expired"
                    outcomes.append(ZoneOutcome(
                        zone_key=w.key, outcome="EXPIRED",
                        reason="H4 zone untested 100 bars",
                        detail=w.describe(now=last_ts)))
                    actives[:] = [z for z in actives
                                  if z is not zone]
                elif tf in ("H1", "M15", "M5") and \
                        age_days > INTRADAY_EXPIRY_DAYS:
                    w = _working(zone)
                    w.status = "expired"
                    outcomes.append(ZoneOutcome(
                        zone_key=w.key, outcome="EXPIRED",
                        reason="intraday zone untested 20 D1 bars",
                        detail=w.describe(now=last_ts)))
                    actives[:] = [z for z in actives
                                  if z is not zone]
            # 4) Score refresh for the TF's actives (live age).
            # Copy-first: score_zone mutates confluence flags.
            view = self._view(working)
            for zone in list(actives):
                w = _working(zone)
                age = _age_d1(w.swing_ts, last_ts)
                score, tier = score_zone(w, view, age_d1=age)
                if score != w.score or tier != w.tier:
                    w.score, w.tier = score, tier
                    w.age_d1 = age
                    if w.status == "active" and tier == "noise":
                        w.status = "noise"
                        outcomes.append(ZoneOutcome(
                            zone_key=w.key, outcome="FADED",
                            reason="score fell to noise",
                            detail=w.describe(now=last_ts)))
                        actives[:] = [z for z in actives
                                      if z is not zone]
            new_marks[tf] = {"cursor": _ts_iso(last_ts),
                             "swing": max_swing_ts or swing_mark or ""}
        # 5) Cluster rule: slowest-TF BROKEN propagates to
        # sub-zones inside its band (BROKEN wins over MITIGATED).
        # Same-transaction scope only: a parent that broke in an
        # earlier transaction is already dead; a sub-zone that
        # holds on its own afterwards is a genuine hold.
        self._apply_cluster_rule(outcomes, working, frames)
        changed = [w for w in working.values()
                   if w != self._mirror.get(w.key)]
        return outcomes, changed, new_marks

    def _view(self, working: dict) -> list[StructuralZone]:
        """Current-transaction working set (copies + untouched)."""
        view = dict(self._mirror)
        view.update(working)
        return list(view.values())

    @staticmethod
    def _bar_at(frame, p: int) -> dict:
        row = frame.iloc[p]
        return {"ts": frame.index[p],
                "high": float(row["high"]), "low": float(row["low"]),
                "close": float(row["close"])}

    def _insert_zone(self, tf, side, price, frame, q, p, m15,
                     outcomes, working) -> StructuralZone | None:
        """Create a zone from a confirmed swing at ``q``; apply the
        bars after the swing (``q+1..p``) immediately.  A zone that
        goes terminal inside its own formation window is stored
        with that terminal status (never briefly active)."""
        try:
            atr = float(atr14(frame.iloc[:p + 1]))
        except Exception:
            return None
        if atr <= 0:
            return None
        try:
            vol_col = "real_volume" if "real_volume" in frame \
                and float(frame.iloc[:p + 1]["real_volume"].sum()) > 0 \
                else "volume"
            vols = frame.iloc[:p + 1][vol_col].to_numpy(dtype=float)
            mean_vol = float(vols.mean()) if len(vols) else 0.0
            swing_vol = float(frame.iloc[q][vol_col]) \
                if vol_col in frame else 0.0
            volume_ratio = (swing_vol / mean_vol) if mean_vol > 0 \
                else None
        except Exception:
            volume_ratio = None
        # A4 base length: consecutive bars ending at the swing
        # whose ranges overlap the band and stay within 1 ATR —
        # the consolidation price sat in before departing.
        base_bars = 0
        try:
            band_lo = float(price) - ZONE_HALF_ATR * atr
            band_hi = float(price) + ZONE_HALF_ATR * atr
            j = q
            while j >= 0 and base_bars < 12:
                row = frame.iloc[j]
                hi, lo = float(row["high"]), float(row["low"])
                if hi - lo > 1.0 * atr or hi < band_lo or lo > band_hi:
                    break
                base_bars += 1
                j -= 1
        except Exception:
            base_bars = 0
        zone = StructuralZone(
            tf=tf, side=side, level=float(price),
            top=float(price) + ZONE_HALF_ATR * atr,
            bottom=float(price) - ZONE_HALF_ATR * atr,
            status="active", swing_ts=frame.index[q],
            atr=atr, volume_ratio=volume_ratio, base_bars=base_bars,
            age_d1=_age_d1(frame.index[q], frame.index[p]))
        for j in range(q + 1, p + 1):
            term, note = self._apply_bar(
                zone, self._bar_at(frame, j), m15)
            if term is not None:
                outcomes.append(ZoneOutcome(
                    zone_key=zone.key, outcome=term, reason=note,
                    detail=zone.describe(now=frame.index[p])))
                break
        if zone.status == "active":
            score, tier = score_zone(zone, self._view(working),
                                     age_d1=zone.age_d1)
            zone.score, zone.tier = score, tier
            if tier == "noise":
                zone.status = "noise"
        working[zone.key] = zone
        return zone

    def _apply_bar(self, zone: StructuralZone, bar: dict, m15
                   ) -> tuple[str | None, str]:
        """One bar against one zone.  Returns ``(terminal, note)``
        where terminal is MITIGATED / BROKEN / TOUCHED / None.

        The CLOSE governs: inside -> mitigation candidate, beyond
        -> break candidate, away -> touch candidate.  A wick
        beyond the far edge never decides by itself.
        """
        if zone.status != "active":
            return None, ""
        c, h, l = bar["close"], bar["high"], bar["low"]
        atr = zone.atr if zone.atr > 0 else 1.0
        # Displacement: max excursion AWAY from the zone.
        if zone.side == "resistance":
            exc = (zone.level - l) / atr
        else:
            exc = (h - zone.level) / atr
        if exc > zone.displacement_atr:
            zone.displacement_atr = exc
        # A3: formation departure tracks the same excursion ONLY
        # until the zone is first tested (a touch or an inside
        # close) — after that, later travel must not rewrite the
        # grade the departure earned at formation.
        if zone.touches == 0 and not zone.seen_inside \
                and exc > zone.departure_atr:
            zone.departure_atr = exc
        far = zone.top if zone.side == "resistance" else zone.bottom
        near = zone.bottom if zone.side == "resistance" else zone.top
        inside = zone.bottom <= c <= zone.top
        beyond = (c > far) if zone.side == "resistance" else (c < far)
        away = (c < near) if zone.side == "resistance" else (c > near)
        wick_beyond = (h > far) if zone.side == "resistance" \
            else (l < far)
        breakout_dir = 1 if zone.side == "resistance" else -1
        ts_iso = _ts_iso(bar["ts"])
        if inside:
            if zone.consec_beyond > 0:
                zone.weak_break = True  # beyond close, returned
            zone.consec_beyond = 0
            zone.seen_inside = True
            zone.last_tested = ts_iso
            penetration = (c - near) if zone.side == "resistance" \
                else (near - c)
            if penetration >= MITIGATION_PUSH_ATR * atr:
                zone.status = "mitigated"
                zone.last_pattern = "sweep" if wick_beyond else "OB"
                note = ""
                if wick_beyond and self._bos(
                        m15, breakout_dir, bar["ts"]):
                    note = ("wick beyond far edge + BOS present, "
                            "close inside -> MITIGATED (close governs)")
                return "MITIGATED", note
            return None, ""
        if beyond:
            zone.consec_beyond += 1
            zone.last_tested = ts_iso
            zone.last_pattern = "sweep"
            if zone.consec_beyond >= 2 and self._bos(
                    m15, breakout_dir, bar["ts"]):
                zone.status = "broken"
                note = ""
                if zone.seen_inside:
                    note = ("prior inside close, then 2 closes "
                            "beyond + BOS")
                return "BROKEN", note
            return None, ""
        # Away: close never entered the zone.
        if zone.consec_beyond > 0:
            zone.weak_break = True  # beyond close, returned
        zone.consec_beyond = 0
        reached = (h >= near) if zone.side == "resistance" \
            else (l <= near)
        if reached:
            zone.touches += 1
            zone.last_tested = ts_iso
            zone.last_pattern = "sweep" if wick_beyond else "OB"
            return "TOUCHED", ""
        return None, ""

    def _bos(self, m15, direction: int, up_to_ts) -> bool:
        """M15 break of structure in ``direction``, causal: only
        M15 bars with ts <= ``up_to_ts``.  No/short M15 -> True
        (closes alone decide; documented, not fabricated).

        Pure in (M15 snapshot, direction, up_to_ts): within one
        transaction the M15 snapshot is fixed, so results are
        memoized per transaction by apply() (see _bos_cache).
        Outside a transaction the cache is None and this computes
        directly.
        """
        cache = self._bos_cache
        key = None
        if cache is not None:
            try:
                key = (int(direction), _ts_iso(up_to_ts))
            except Exception:
                key = None
            if key is not None and key in cache:
                return cache[key]
        result = self._bos_fast(m15, direction, up_to_ts)
        if cache is not None and key is not None:
            cache[key] = result
        return result

    def _bos_ctx(self, m15):
        """Precomputed M15 swings for this transaction, or None.

        ``_bos`` is called once per zone per bar; recomputing
        ``find_swings`` over the whole M15 prefix on every call is
        O(bars x zones x M15) and never finished the first refresh
        in the full-engine backtest.  The swings are computed ONCE
        per transaction over the full M15 frame and reused: a swing
        at position ``i`` depends only on bars ``[i-n, i+n]``, so
        the swings valid for a prefix of length ``L`` are exactly
        the full-frame swings with ``i < L - n`` (no lookahead —
        same window, same values).  Returns ``(index, highs,
        lows)`` or None (no cache / non-monotonic index -> the
        caller falls back to the direct computation).
        """
        cache = self._bos_cache
        if cache is None or m15 is None:
            return None
        ctx = cache.get("_ctx")
        if ctx is not None:
            if ctx[0] is m15:
                return ctx
            # Different M15 snapshot inside one transaction (should
            # not happen): recompute rather than reuse.
        try:
            if not m15.index.is_monotonic_increasing:
                return None
            highs, lows = find_swings(
                m15["high"].to_numpy(dtype=float),
                m15["low"].to_numpy(dtype=float), SWING_N)
        except Exception:
            return None
        ctx = (m15, m15.index, highs, lows,
               [p[0] for p in highs], [p[0] for p in lows])
        cache["_ctx"] = ctx
        return ctx

    def _bos_fast(self, m15, direction: int, up_to_ts) -> bool:
        """BOS via the precomputed per-transaction M15 swings."""
        ctx = self._bos_ctx(m15)
        if ctx is None:
            return self._bos_uncached(m15, direction, up_to_ts)
        _, index, highs, lows, high_pos, low_pos = ctx
        try:
            ts = pd.Timestamp(up_to_ts)
        except Exception:
            return True
        length = int(index.searchsorted(ts, side="right"))
        if length < 2 * SWING_N + 5:
            return True
        try:
            close = float(m15["close"].iloc[length - 1])
        except Exception:
            return True
        # Swings valid for this prefix: position i < length - n
        # (their [i-n, i+n] window lies fully inside the prefix).
        limit = length - SWING_N
        if direction > 0:
            k = bisect.bisect_right(high_pos, limit - 1)
            return k > 0 and close > highs[k - 1][1]
        k = bisect.bisect_right(low_pos, limit - 1)
        return k > 0 and close < lows[k - 1][1]

    def _bos_uncached(self, m15, direction: int, up_to_ts) -> bool:
        if m15 is None or len(m15) < 2 * SWING_N + 5:
            return True
        try:
            sub = m15.loc[:pd.Timestamp(up_to_ts)]
        except Exception:
            return True
        if len(sub) < 2 * SWING_N + 5:
            return True
        try:
            highs, lows = find_swings(
                sub["high"].to_numpy(dtype=float),
                sub["low"].to_numpy(dtype=float), SWING_N)
            close = float(sub.iloc[-1]["close"])
        except Exception:
            return True
        if direction > 0:
            return bool(highs) and close > highs[-1][1]
        return bool(lows) and close < lows[-1][1]

    def _apply_cluster_rule(self, outcomes, working, frames) -> None:
        """Cluster consistency: a BROKEN slowest-TF zone breaks
        its faster-TF sub-zones inside its band (BROKEN wins over
        MITIGATED / TOUCHED / NONE).  Each zone keeps exactly one
        outcome: an overridden outcome REPLACES the original in
        the outcomes list (the original is noted in the reason).
        """
        view = self._view(working)
        # Candidates: active zones + zones that went terminal THIS
        # transaction (BROKEN parents + MITIGATED sub-zones are the
        # override's whole point — a mitigated sub-zone inside a
        # broken parent's band becomes BROKEN, not mitigated).
        # A parent broken in an EARLIER transaction is already
        # dead and does not propagate; EXPIRED / FADED zones keep
        # their own terminal state.
        terminal = {"BROKEN", "MITIGATED"}
        # Precompute once: the original code re-scanned ALL outcomes
        # per zone (zones x outcomes) while recomputing z.key
        # (pd.Timestamp().isoformat()) on every comparison — 176s
        # of a 189s seed.  Set lookups are exactly equivalent here:
        # candidates are computed before the loop mutates outcomes.
        terminal_keys = {o.zone_key for o in outcomes
                         if o.outcome in terminal}
        candidates = [
            z for z in view
            if z.status == "active" or z.key in terminal_keys]
        for cluster in find_clusters(
                candidates,
                include_statuses=("active", "broken", "mitigated")):
            zones = cluster["zones"]
            # Recomputed per cluster: prior iterations mutate
            # outcomes (sub-zone -> BROKEN), and the original code
            # saw the mutated list here.
            broken_keys = {o.zone_key for o in outcomes
                           if o.outcome == "BROKEN"}
            broken = [z for z in zones if z.key in broken_keys]
            if not broken:
                continue
            parent = min(broken, key=lambda z: TF_ORDER.index(z.tf))
            for sub in zones:
                if sub.key == parent.key:
                    continue
                if TF_ORDER.index(sub.tf) <= TF_ORDER.index(parent.tf):
                    continue
                if not (parent.bottom <= sub.level <= parent.top):
                    continue
                w = working.get(sub.key)
                if w is None:
                    w = copy.copy(sub)
                    working[sub.key] = w
                if w.status == "broken":
                    continue
                prev = next((o.outcome for o in outcomes
                             if o.zone_key == sub.key), "NONE")
                w.status = "broken"
                w.last_pattern = "sweep"
                # Replace the sub-zone's outcome (one per zone).
                outcomes[:] = [o for o in outcomes
                               if o.zone_key != sub.key]
                outcomes.append(ZoneOutcome(
                    zone_key=sub.key, outcome="BROKEN",
                    reason=("cluster broken, sub-zones broken with "
                            f"parent (was {prev})"),
                    detail=w.describe(
                        now=_ts_iso(frames.get(sub.tf).index[-1])
                        if frames.get(sub.tf) is not None
                        and len(frames[sub.tf]) else None)))

    def persist(self, changed, new_marks, tick_id) -> None:
        """Write ``changed`` + watermarks + tick id in ONE SQLite
        transaction.  On success the mirror adopts the copies +
        watermarks; on failure the transaction rolls back and the
        mirror is untouched (the next refresh re-derives it).
        """
        with self._conn:
            for zone in changed:
                self._conn.execute(
                    "INSERT INTO zones (tf, side, level, top, "
                    "bottom, score, tier, status, touches, "
                    "displacement_atr, confluence, liquidity, "
                    "round_number, weak_break, age_d1, swing_ts, "
                    "volume_ratio, atr, last_tested, last_pattern, "
                    "seen_inside, consec_beyond, departure_atr, "
                    "base_bars) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(tf, side, swing_ts) DO UPDATE SET "
                    "level=excluded.level, top=excluded.top, "
                    "bottom=excluded.bottom, score=excluded.score, "
                    "tier=excluded.tier, status=excluded.status, "
                    "touches=excluded.touches, "
                    "displacement_atr=excluded.displacement_atr, "
                    "confluence=excluded.confluence, "
                    "liquidity=excluded.liquidity, "
                    "round_number=excluded.round_number, "
                    "weak_break=excluded.weak_break, "
                    "age_d1=excluded.age_d1, "
                    "volume_ratio=excluded.volume_ratio, "
                    "atr=excluded.atr, last_tested=excluded.last_tested, "
                    "last_pattern=excluded.last_pattern, "
                    "seen_inside=excluded.seen_inside, "
                    "consec_beyond=excluded.consec_beyond, "
                    "departure_atr=excluded.departure_atr, "
                    "base_bars=excluded.base_bars",
                    self._zone_to_row(zone))
                if self.rtree_available:
                    zid = self._conn.execute(
                        "SELECT id FROM zones WHERE tf = ? AND "
                        "side = ? AND swing_ts = ?",
                        (zone.tf, zone.side,
                         _ts_iso(zone.swing_ts))).fetchone()[0]
                    self._conn.execute(
                        "INSERT OR REPLACE INTO zones_rtree "
                        "(id, min_price, max_price) VALUES (?, ?, ?)",
                        (zid, float(zone.bottom), float(zone.top)))
            for tf, marks in new_marks.items():
                self._conn.execute(
                    "INSERT INTO meta (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (f"cursor_{tf}", str(marks["cursor"])))
                self._conn.execute(
                    "INSERT INTO meta (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (f"swing_{tf}", str(marks["swing"])))
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ("last_applied_tick_id", str(tick_id)))
        for zone in changed:
            self._mirror[zone.key] = zone
        for tf, marks in new_marks.items():
            self._cursors[tf] = str(marks["cursor"])
            self._swing_marks[tf] = str(marks["swing"])
        self._last_applied_tick_id = str(tick_id)


# ---------------------------------------------------------------- cache

@dataclass(frozen=True)
class _Snapshot:
    """One immutable cache version.  Readers hold a reference —
    never torn (the snapshot never mutates after publication)."""

    version: int
    zones: tuple            # all active zones, sorted by level
    prices: tuple           # their levels (bisect index)
    majors: tuple           # active MAJOR zones, sorted by level
    major_prices: tuple     # their levels (bisect index)
    outcomes: tuple         # last applied outcomes (ZoneOutcome)
    tick_id: str
    reason: str
    refreshed_at: str


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1,
              max(0, math.ceil(p / 100.0 * len(sorted_vals)) - 1))
    return sorted_vals[idx]


class ZoneCache:
    """In-memory execution-path zone cache (versioned, copy-on-write).

    The tick loop queries THIS, never SQLite.  Single writer (the
    tick loop): :meth:`swap` builds a new immutable snapshot and
    swaps the reference (version++).  Readers copy the reference
    and read only from that snapshot — old or new, never mixed.
    Query timings are recorded per call (deque, maxlen 2000) for
    the /status p50/p95/p99 line.
    """

    def __init__(self):
        self._current = _Snapshot(
            version=0, zones=(), prices=(), majors=(),
            major_prices=(), outcomes=(), tick_id="",
            reason="startup", refreshed_at="")
        self._lock = threading.Lock()
        self._durations: deque = deque(maxlen=2000)
        self._queries = 0

    def swap(self, zones, outcomes, tick_id, *,
             reason: str = "bar close") -> int:
        """Publish a new snapshot (copy-on-write).  Returns the new
        version.  ``zones`` = full active list (from the store's
        mirror).  Infallible reference swap — the caller persists
        to SQLite BEFORE swapping, so a persist failure never
        leaves a swapped-but-unpersisted cache."""
        actives = sorted((z for z in zones if z.status == "active"),
                         key=lambda z: z.level)
        majors = tuple(z for z in actives if z.tier == "MAJOR")
        with self._lock:
            version = self._current.version + 1
            self._current = _Snapshot(
                version=version,
                zones=tuple(actives),
                prices=tuple(z.level for z in actives),
                majors=majors,
                major_prices=tuple(z.level for z in majors),
                outcomes=tuple(outcomes),
                tick_id=str(tick_id),
                reason=reason,
                refreshed_at=pd.Timestamp.now(tz="UTC").isoformat())
        return version

    def nearby(self, price: float, atr: float, *,
               radius_atr: float = NEARBY_RADIUS_ATR):
        """Zones within price +/- radius*ATR.  O(log n + k), no DB,
        no SQL.  Returns ``(version, [zones])`` — the version tags
        the snapshot the result came from (torn-read tests)."""
        snap = self._current
        t0 = time.perf_counter()
        try:
            if atr <= 0:
                return snap.version, []
            lo, hi = price - radius_atr * atr, price + radius_atr * atr
            i = bisect.bisect_left(snap.prices, lo)
            j = bisect.bisect_right(snap.prices, hi)
            return snap.version, list(snap.zones[i:j])
        finally:
            self._durations.append(
                (time.perf_counter() - t0) * 1000.0)
            self._queries += 1

    def major_nearby(self, price: float, atr: float, *,
                     radius_atr: float = MAJOR_PRECOMPUTE_ATR):
        """Active MAJOR zones within price +/- radius*ATR — the
        precomputed MAJOR list (built once per swap, bisected per
        call).  O(log n + k), no DB."""
        snap = self._current
        t0 = time.perf_counter()
        try:
            if atr <= 0:
                return snap.version, []
            lo, hi = price - radius_atr * atr, price + radius_atr * atr
            i = bisect.bisect_left(snap.major_prices, lo)
            j = bisect.bisect_right(snap.major_prices, hi)
            return snap.version, list(snap.majors[i:j])
        finally:
            self._durations.append(
                (time.perf_counter() - t0) * 1000.0)
            self._queries += 1

    def last_outcomes_summary(self) -> str:
        """``D1 BROKEN 4227.5, H1 MITIGATED 4210.0`` or ``""``."""
        parts = []
        for o in self._current.outcomes:
            if o.outcome in ("MITIGATED", "BROKEN", "EXPIRED",
                             "FADED", "ACTIVATED"):
                # detail = "D1 4227.5, ..." — level is the 2nd token.
                lvl = o.detail.split(",")[0].split(" ")[-1] \
                    if o.detail else "?"
                parts.append(f"{o.zone_key[0]} {o.outcome} {lvl}")
        return ", ".join(parts)

    def stats(self) -> dict:
        """Cache stats for /status: version, counts, query p50/p95/
        p99 (ms), last refresh reason/ts, last outcomes."""
        snap = self._current
        vals = sorted(self._durations)
        return {
            "version": snap.version,
            "n_zones": len(snap.zones),
            "n_major": len(snap.majors),
            "queries": self._queries,
            "p50_ms": _pct(vals, 50),
            "p95_ms": _pct(vals, 95),
            "p99_ms": _pct(vals, 99),
            "last_tick_id": snap.tick_id,
            "last_refresh_reason": snap.reason,
            "last_refresh_ts": snap.refreshed_at,
            "last_outcomes": self.last_outcomes_summary(),
        }
