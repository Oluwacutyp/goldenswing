"""QMM5Playbook — Quasimodo reversal on M5 (SHADOW, unvalidated).

Mined definition (practitioner QM, ICT/SMC reversal variant — NOT
head-and-shoulders: a QM has no second shoulder):

Bearish QM (reversal at the top of an uptrend), pivots in order:
  1. HEAD — swing high H0 (the higher high).
  2. NECKLINE — pullback swing low L1 (the higher low).
  3. QM LEVEL — rally swing high H2 that FAILS below the head
     (the lower high; H2 < H0 by a genuine margin, not an equal high).
     Late buyers are trapped at H2.
  4. BREAK — a CLOSE below the neckline L1 (break of structure).
  5. RETRACEMENT — price returns to the QM level H2.  Entry is the
     resting LIMIT at H2: no retracement, no trade (doctrine:
     "entry AT POI only, never above it").

Bullish QM is the mirror: lower low (head) -> lower high (neckline)
-> higher low that fails (QM level) -> close above the neckline ->
retracement down to the QM level, long.

Utilization (QM as an entry pattern inside the house doctrine
sweep -> displacement -> retracement -> LIMIT AT POI, NOT a separate
system):
  - Entry: resting limit at the QM level on the FIRST touch after the
    break (one trade per pattern).
  - Stop: head extreme +/- 0.25 ATR (the repo's standard structural
    buffer; the raw head is the invalidation).
  - Invalidation: a CLOSE beyond the head before the touch (pattern
    dead), or a close beyond the head while working the trade.
  - Target: the broken neckline level (the practitioner target: "the
    structural level that price broke").  The setup only qualifies if
    the neckline sits >= 1.0R from the QM level — a compressed pattern
    that cannot pay is skipped, not re-targeted.
  - Expiry: break must come within 120 bars of the head (an M5 QM is
    an intraday structure; slower formations mix sessions), the
    retracement within 48 bars of the break, the trade within 48 bars
    of the touch (the fvg_limit MAX_HOLD precedent).
  - Confluence TAGS (never gates, never vetoes): ``sweep_head`` when
    the head wicked above the prior swing high (the head leg itself
    was the sweep); ``displacement_break`` when the break bar's body
    is >= 1.0x ATR (the repo's Phase-2 displacement definition).

Pivot machinery: confirmed fractal pivots, swing_n = 3 — the repo's M5
trigger swing_n (SCALP: H1 bias / M15 zones / M5 trigger, swing_n=3).
A pivot at bar i is only usable once bar i+3 has closed (no
lookahead).  Consecutive same-side pivots keep only the more extreme
one (the MQL5 zig-zag rule); the QM level freezes at the break bar.

STATUS: SHADOW-ONLY (live=False).  QM has NO measured validation in
this repo — no backtest, no win rate, no context map.  It appears in
batch candidates, the journal, and the shadow book, never in live
P&L.  Promotion to live needs forward paper observation on a real M5
feed (the live paper loop is H1-only today, so QM has no live feed
yet).  The base score is a documented placeholder like the other
sleeves, NOT a measured score.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..backtest.harness import Outcome
from ..contextmap import HuntingGround
from ..golden.detector import GoldenSignal
from .base import Candidate, EngineContext, Playbook

#: M5 trigger swing_n (SCALP config: M5 trigger, swing_n=3).
SWING_N = 3
#: 14-bar mean true range, same formula as fvg_limit.
ATR_N = 14
#: Each QM leg must span >= 0.5 ATR — a smaller "leg" is noise, not
#: structure.
MIN_LEG_ATR = 0.5
#: The failed rally must fall short of the head by >= 0.25 ATR — a
#: genuine failure, not an equal high.
MIN_FAIL_ATR = 0.25
#: Structural stop buffer beyond the head (repo standard).
STOP_BUFFER_ATR = 0.25
#: The neckline target must pay >= 1.0R or the setup is skipped.
MIN_TARGET_R = 1.0
#: Formation window: head -> break (M5: 120 bars ~= 10h, one session).
MAX_PATTERN_BARS = 120
#: Retracement window: break -> first touch of the QM level.
MAX_RETRACE_BARS = 48
#: Maximum hold after the touch (fvg_limit MAX_HOLD precedent).
MAX_HOLD_BARS = 48
#: Break-bar body >= 1.0x ATR counts as displacement (Phase-2 rule).
DISPLACEMENT_BODY_ATR = 1.0
#: Stop-distance band re-verified by check_live_signal (trendline
#: precedent: 0.25–3.5 ATR).
MIN_STOP_ATR, MAX_STOP_ATR = 0.25, 3.5

#: Base score is a documented placeholder, NOT a measured score — QM
#: has no validation yet.  OOS promotion must calibrate it; until then
#: it only ranks sleeve-vs-sleeve.
SLEEVE_BASE_SCORE = 60.0

#: Engine universe (XAUUSD M5 bars).  This is the feed universe, NOT
#: a validation claim — QM is unvalidated on every instrument.
VALIDATED_INSTRUMENTS = ("XAUUSD", "GC=F")


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         i: int) -> float:
    """14-bar mean true range at closed bar i (needs i >= 14)."""
    if i < ATR_N:
        return 0.0
    trs = [
        max(high[k] - low[k], abs(high[k] - close[k - 1]),
            abs(low[k] - close[k - 1]))
        for k in range(i - ATR_N + 1, i + 1)
    ]
    return float(sum(trs) / ATR_N)


@dataclass
class _Pivot:
    index: int
    price: float
    side: int  # +1 swing high | -1 swing low

    def to_dict(self) -> dict:
        return {"index": self.index, "price": self.price, "side": self.side}

    @classmethod
    def from_dict(cls, d: dict) -> "_Pivot":
        return cls(index=int(d["index"]), price=float(d["price"]),
                   side=int(d["side"]))


@dataclass
class _Triple:
    """A formed H-L-H / L-H-L pivot triple awaiting its neckline break.

    Bearish (direction -1): p0 = head high, p1 = neckline low,
    p2 = QM-level high.  Bullish (direction +1): mirrored.
    """
    direction: int
    p0: _Pivot
    p1: _Pivot
    p2: _Pivot
    sweep_head: bool = False

    def to_dict(self) -> dict:
        return {"direction": self.direction,
                "p0": self.p0.to_dict(), "p1": self.p1.to_dict(),
                "p2": self.p2.to_dict(), "sweep_head": self.sweep_head}

    @classmethod
    def from_dict(cls, d: dict) -> "_Triple":
        return cls(direction=int(d["direction"]),
                   p0=_Pivot.from_dict(d["p0"]),
                   p1=_Pivot.from_dict(d["p1"]),
                   p2=_Pivot.from_dict(d["p2"]),
                   sweep_head=bool(d.get("sweep_head", False)))

    @property
    def head(self) -> float:
        return self.p0.price

    @property
    def neckline(self) -> float:
        return self.p1.price

    @property
    def qm_level(self) -> float:
        return self.p2.price


@dataclass
class _Broken:
    """A triple whose neckline broke, awaiting retracement to the QM
    level.  The QM level is FROZEN at the break bar."""
    triple: _Triple
    break_index: int
    displacement: bool = False

    def to_dict(self) -> dict:
        return {"triple": self.triple.to_dict(),
                "break_index": self.break_index,
                "displacement": self.displacement}

    @classmethod
    def from_dict(cls, d: dict) -> "_Broken":
        return cls(triple=_Triple.from_dict(d["triple"]),
                   break_index=int(d["break_index"]),
                   displacement=bool(d.get("displacement", False)))


@dataclass
class _Touch:
    """A completed QM setup: price touched the QM level after the
    neckline break.  The batch path simulates it; the live path emits
    it as a signal."""
    direction: int
    touch_index: int
    qm_level: float
    head: float
    neckline: float
    break_index: int
    p0_index: int
    p1_index: int
    p2_index: int
    atr: float
    tags: list[str] = field(default_factory=list)

    @property
    def stop(self) -> float:
        buf = STOP_BUFFER_ATR * self.atr
        return self.head - self.direction * buf

    @property
    def risk(self) -> float:
        return (self.qm_level - self.stop) * self.direction

    @property
    def target_r(self) -> float:
        if self.risk <= 0:
            return 0.0
        return (self.neckline - self.qm_level) * self.direction / self.risk


def _simulate(touch: _Touch, o: np.ndarray, h: np.ndarray,
              l: np.ndarray, c: np.ndarray) -> tuple | None:
    """The QM trade machine.  Returns (entry, stop, target, r,
    exit_index, exit_reason) or None when the trade is untradeable
    (target pays < MIN_TARGET_R — checked again here so the batch path
    can never drift from the live gate).

    Entry is the QM level (resting limit, fills on touch).  Walk
    forward from the touch bar: buffered stop first (harness
    convention — stop wins same-bar ties), then close-beyond-head
    invalidation at the close, then the neckline target.  Timeout
    exits at the close, booked (not dropped).
    """
    n = len(c)
    if touch.risk <= 0 or touch.target_r < MIN_TARGET_R:
        return None
    entry, stop, target = touch.qm_level, touch.stop, touch.neckline
    d = touch.direction
    r, exit_idx, reason = None, None, None
    last = min(touch.touch_index + MAX_HOLD_BARS, n - 1)
    for k in range(touch.touch_index + 1, last + 1):
        if d == 1:
            if l[k] <= stop:
                r, exit_idx, reason = (stop - entry) / touch.risk, k, "stop"
                break
            if c[k] < touch.head:
                r, exit_idx, reason = (c[k] - entry) / touch.risk, k, \
                    "invalidation"
                break
            if h[k] >= target:
                r, exit_idx, reason = touch.target_r, k, "target"
                break
        else:
            if h[k] >= stop:
                r, exit_idx, reason = (entry - stop) / touch.risk, k, "stop"
                break
            if c[k] > touch.head:
                r, exit_idx, reason = (entry - c[k]) / touch.risk, k, \
                    "invalidation"
                break
            if l[k] <= target:
                r, exit_idx, reason = touch.target_r, k, "target"
                break
    if r is None:
        r, exit_idx, reason = (d * (c[last] - entry) / touch.risk), last, \
            "timeout"
    return entry, stop, target, r, exit_idx, reason


class _QMEngine:
    """The single QM state machine behind BOTH paths.

    Batch runs it over all bars with collect=True; live advances it
    bar-by-bar with collect only on the latest bar.  One machine, no
    drift, parity by construction.
    """

    def __init__(self) -> None:
        self.pivots: list[_Pivot] = []
        self.pending: list[_Triple] = []    # formed, awaiting break
        self.broken: list[_Broken] = []     # broken, awaiting touch

    # ---------------------------------------------------------- pivots
    def _add_pivot(self, piv: _Pivot, atr: float) -> None:
        """Insert a confirmed pivot, keeping the alternating zig-zag.

        Same-side repeats keep only the more extreme one.  A
        replacement of a pending triple's QM pivot re-validates the
        triple against the new pivot (a rally that reaches the head
        un-forms the QM); broken patterns keep their frozen QM level.
        """
        if self.pivots and self.pivots[-1].side == piv.side:
            last = self.pivots[-1]
            more_extreme = (piv.price > last.price if piv.side == 1
                            else piv.price < last.price)
            if not more_extreme:
                return  # weaker repeat: ignore
            old_index = last.index
            self.pivots[-1] = piv
            for tr in list(self.pending):
                if tr.p2.index == old_index:
                    tr.p2 = piv
                    if not _triple_valid(tr, atr):
                        self.pending.remove(tr)
            # A replacement can also complete a triple that was never
            # formed (the old pivot failed validation); the pending
            # guard inside _maybe_form_triple prevents duplicates.
            self._maybe_form_triple(atr)
            return
        self.pivots.append(piv)
        self._maybe_form_triple(atr)

    def _maybe_form_triple(self, atr: float) -> None:
        if len(self.pivots) < 3 or atr <= 0:
            return
        a, b, c = self.pivots[-3], self.pivots[-2], self.pivots[-1]
        if not (a.index < b.index < c.index):
            return
        if not (a.side == -b.side == c.side):
            return  # not alternating: not a triple
        if any(tr.p2.index == c.index for tr in self.pending):
            return  # already pending for this QM pivot: no duplicate
        direction = -a.side  # H-L-H -> short (-1); L-H-L -> long (+1)
        sweep = False
        if len(self.pivots) >= 5:
            prior = self.pivots[-5]
            if prior.side == a.side:
                if direction == -1 and prior.price < a.price:
                    sweep = True  # head wicked above the prior swing high
                elif direction == 1 and prior.price > a.price:
                    sweep = True  # head wicked below the prior swing low
        tr = _Triple(direction=direction, p0=a, p1=b, p2=c,
                     sweep_head=sweep)
        if _triple_valid(tr, atr):
            self.pending.append(tr)

    # ---------------------------------------------------------- one bar
    def process(self, o, h, l, c, t: int, collect: bool) -> list[_Touch]:
        """Advance the machine through closed bar ``t``.

        Pivot at t-SWING_N confirms on this bar; breaks and touches
        are evaluated on this bar's close.  Touch events on past bars
        still advance state but are only returned when ``collect``.
        """
        out: list[_Touch] = []
        atr = _atr(h, l, c, t)
        # 1. confirm the pivot at t - SWING_N (strictly extreme over
        #    [i-SWING_N, i+SWING_N] — ties are never pivots).  Pivot
        #    confirmation is price-only: it must NOT wait for the
        #    14-bar ATR warmup (a pivot confirming on bar 13 would
        #    otherwise be lost forever — only triple validation
        #    needs ATR, and it gates on atr <= 0 itself).
        i = t - SWING_N
        if i >= SWING_N:
            lo_i, hi_i = i - SWING_N, i + SWING_N
            if h[i] > max(h[k] for k in range(lo_i, hi_i + 1) if k != i):
                self._add_pivot(_Pivot(index=i, price=float(h[i]), side=1),
                                atr)
            if l[i] < min(l[k] for k in range(lo_i, hi_i + 1) if k != i):
                self._add_pivot(_Pivot(index=i, price=float(l[i]), side=-1),
                                atr)
        # 2. neckline breaks (stale formations expire first).
        kept: list[_Triple] = []
        for tr in self.pending:
            if t - tr.p0.index > MAX_PATTERN_BARS:
                continue  # stale: never broke in time
            d = tr.direction
            broke = (c[t] < tr.neckline) if d == -1 \
                else (c[t] > tr.neckline)
            if broke:
                body_atr = (abs(c[t] - o[t]) / atr) if atr > 0 else 0.0
                self.broken.append(_Broken(
                    triple=tr, break_index=t,
                    displacement=body_atr >= DISPLACEMENT_BODY_ATR))
            else:
                kept.append(tr)
        self.pending = kept
        # 3. retracement touches (expiry, then invalidation, then touch;
        #    a pattern broken ON this bar cannot touch on the same bar —
        #    the retracement must come after the break).
        kept_b: list[_Broken] = []
        for br in self.broken:
            tr = br.triple
            d = tr.direction
            if br.break_index >= t:
                kept_b.append(br)
                continue
            if t - br.break_index > MAX_RETRACE_BARS:
                continue  # expired: never retraced to the QM level
            invalidated = (c[t] > tr.head) if d == -1 \
                else (c[t] < tr.head)
            if invalidated:
                continue  # close beyond the head: pattern dead
            touched = (h[t] >= tr.qm_level) if d == -1 \
                else (l[t] <= tr.qm_level)
            if not touched:
                kept_b.append(br)
                continue
            touch = _Touch(
                direction=d, touch_index=t, qm_level=tr.qm_level,
                head=tr.head, neckline=tr.neckline,
                break_index=br.break_index, p0_index=tr.p0.index,
                p1_index=tr.p1.index, p2_index=tr.p2.index, atr=atr,
                tags=(["sweep_head"] if tr.sweep_head else [])
                + (["displacement_break"] if br.displacement else []),
            )
            # The live gate, applied here so batch and live can never
            # diverge: a compressed pattern that cannot pay >= 1R is
            # dropped, not emitted.
            if touch.risk > 0 and touch.target_r >= MIN_TARGET_R \
                    and collect:
                out.append(touch)
            # One trade per pattern, gated or not.
        self.broken = kept_b
        return out


def _triple_valid(tr: _Triple, atr: float) -> bool:
    """Structural inequalities for one pivot triple, in ATR units."""
    if atr <= 0:
        return False
    if tr.direction == -1:  # bearish: H0 head, L1 neckline, H2 QM level
        leg1 = tr.p0.price - tr.p1.price
        leg2 = tr.p2.price - tr.p1.price
        fail = tr.p0.price - tr.p2.price
    else:                   # bullish mirror
        leg1 = tr.p1.price - tr.p0.price
        leg2 = tr.p1.price - tr.p2.price
        fail = tr.p2.price - tr.p0.price
    return (leg1 >= MIN_LEG_ATR * atr
            and leg2 >= MIN_LEG_ATR * atr
            and fail >= MIN_FAIL_ATR * atr)


def _arrays(df):
    return (
        df["open"].to_numpy(dtype=float),
        df["high"].to_numpy(dtype=float),
        df["low"].to_numpy(dtype=float),
        df["close"].to_numpy(dtype=float),
    )


def _to_signal(touch: _Touch, timestamp) -> GoldenSignal:
    sig = GoldenSignal(
        index=touch.touch_index, timestamp=timestamp,
        direction=touch.direction, score=SLEEVE_BASE_SCORE,
        kind="qm_m5",
        factors=["qm_m5"] + list(touch.tags),
        entry_zone=(touch.qm_level, touch.qm_level),
        entry_ref=touch.qm_level,
        invalidation=touch.stop,
        targets=[touch.neckline],
        atr=touch.atr,
    )
    sig.live_context = {
        "touch_index": touch.touch_index,
        "qm_level": touch.qm_level, "head": touch.head,
        "neckline": touch.neckline, "break_index": touch.break_index,
        "p0_index": touch.p0_index, "p1_index": touch.p1_index,
        "p2_index": touch.p2_index, "direction": touch.direction,
        "target_r": touch.target_r, "tags": list(touch.tags),
    }
    return sig


class QMM5Playbook(Playbook):
    name = "qm_m5"
    description = (
        "Quasimodo reversal on M5: head -> neckline pullback -> failed "
        "rally (QM level) -> close through the neckline -> limit at the "
        "QM level, stop beyond the head, target the broken neckline"
    )
    live = False  # SHADOW: unvalidated — shadow book + journal only,
    # never live P&L, until forward paper observation promotes it.
    live_threshold = 0.0  # every qualifying touch is graded
    context_description = (
        "UNVALIDATED (no backtest, no win rate, no context map yet). "
        "M5 Quasimodo reversal: confirmed fractal pivots (swing_n=3), "
        "both legs >= 0.5 ATR, failed rally short of the head by "
        ">= 0.25 ATR, close through the neckline within 120 bars of "
        "the head, first retracement touch of the QM level within 48 "
        "bars, neckline target >= 1.0R. Session-agnostic UNTIL "
        "measured — timezone/session mapping is a separate task and "
        "must not be assumed here."
    )
    #: Session-agnostic until measured (timezone mapping is Task 2).
    session_scope = "any"
    #: No measured context map yet — fires everywhere, no map bonuses.
    hunting_ground = HuntingGround(excluded=set(), bonuses={})

    def __init__(self) -> None:
        self._engine = _QMEngine()
        self._processed = -1  # last bar run through the machine

    # ------------------------------------------------------- batch path
    def candidates(self, ctx: EngineContext) -> list[Candidate]:
        o, h, l, c = _arrays(ctx.df)
        engine = _QMEngine()
        out: list[Candidate] = []
        for t in range(len(o)):
            for touch in engine.process(o, h, l, c, t, collect=True):
                sim = _simulate(touch, o, h, l, c)
                if sim is None:
                    continue
                entry, stop, target, r, exit_idx, reason = sim
                sig = _to_signal(touch, ctx.df.index[touch.touch_index])
                outcome = Outcome(
                    signal=sig, entry_index=touch.touch_index,
                    exit_index=exit_idx, exit_reason=reason, r_net=r,
                    hold_bars=exit_idx - touch.touch_index,
                )
                side = "long" if touch.direction == 1 else "short"
                thesis = (
                    f"qm_m5 {side} (touch bar {touch.touch_index}): "
                    f"head {touch.head:.1f} (bar {touch.p0_index}) -> "
                    f"neckline {touch.neckline:.1f} (bar "
                    f"{touch.p1_index}) -> failed rally to QM level "
                    f"{touch.qm_level:.1f} (bar {touch.p2_index}) -> "
                    f"close through neckline (bar {touch.break_index}) "
                    f"-> limit at QM level {entry:.1f}, stop {stop:.1f}, "
                    f"target neckline {target:.1f} "
                    f"({touch.target_r:.2f}R)"
                    + (f", tags: {','.join(touch.tags)}"
                       if touch.tags else "")
                )
                out.append(Candidate(
                    playbook=self.name, outcome=outcome,
                    base_score=SLEEVE_BASE_SCORE, thesis=thesis,
                    context=dict(sig.live_context)))
        return out

    def check_live_signal(self, signal, ctx: EngineContext,
                          context: dict | None = None) -> tuple[bool, str]:
        # Independent re-verification of the QM facts.  The scanner
        # guarantees all of these; a failure here is a scanner
        # regression — a system bug, never a trade.
        if ctx.instrument not in VALIDATED_INSTRUMENTS:
            return False, (
                f"validated universe is XAUUSD, got {ctx.instrument!r}")
        c = context or getattr(signal, "live_context", None) or {}
        need = ("touch_index", "qm_level", "head", "neckline",
                "break_index", "direction")
        missing = [k for k in need if k not in c]
        if missing:
            return False, f"missing setup facts: {missing}"
        if c["break_index"] >= c["touch_index"]:
            return False, "touch must come strictly after the break"
        atr = signal.atr
        if atr and atr > 0:
            stop_atr = abs(signal.entry_ref - signal.invalidation) / atr
            if not (MIN_STOP_ATR <= stop_atr <= MAX_STOP_ATR):
                return False, (
                    f"stop {stop_atr:.2f} ATR outside the validated "
                    f"{MIN_STOP_ATR}–{MAX_STOP_ATR} band")
            risk = (signal.entry_ref - signal.invalidation) \
                * signal.direction
            if risk > 0 and signal.targets:
                target_r = (signal.targets[0] - signal.entry_ref) \
                    * signal.direction / risk
                if target_r < MIN_TARGET_R:
                    return False, (
                        f"neckline target {target_r:.2f}R below the "
                        f"{MIN_TARGET_R}R qualification floor")
        return True, ""

    # ------------------------------------------------------- live path
    def live_signal(self, states: list, t: int, df=None) -> list:
        """Incremental QM check at closed bar ``t``.

        Returns the signal(s) whose QM level was touched on bar ``t``
        — the same signal the batch path books the trade from.
        """
        if df is None or t < 0:
            return []
        o, h, l, c = _arrays(df)
        if t >= len(o):
            return []
        # Never rewind the cursor (see fvg_limit: rebase fresh loop
        # re-calls the tail; rewinding re-emits duplicates and can
        # corrupt the streaming machine).
        if t <= self._processed:
            return []
        out: list = []
        for t2 in range(self._processed + 1, t + 1):
            for touch in self._engine.process(o, h, l, c, t2,
                                              collect=(t2 == t)):
                out.append(_to_signal(touch, df.index[t2]))
        self._processed = t
        return out

    def export_state(self) -> dict:
        return {"processed": self._processed,
                "pivots": [p.to_dict() for p in self._engine.pivots],
                "pending": [tr.to_dict()
                            for tr in self._engine.pending],
                "broken": [br.to_dict()
                           for br in self._engine.broken]}

    def import_state(self, state: dict) -> None:
        # {} means RESET (fresh load / rebase fast-forward): build a
        # FRESH engine, never keep stale streaming state.
        state = state or {}
        self._processed = int(state.get("processed", -1))
        self._engine = _QMEngine()
        self._engine.pivots = [_Pivot.from_dict(p)
                                for p in state.get("pivots", [])]
        self._engine.pending = [_Triple.from_dict(tr)
                                for tr in state.get("pending", [])]
        self._engine.broken = [_Broken.from_dict(br)
                                for br in state.get("broken", [])]
