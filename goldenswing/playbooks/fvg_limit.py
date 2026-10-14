"""FVGLimitPlaybook — limit-at-touch FVG entry (paper-live sleeve).

Validated: scripts/test_fvg_entry_ab.py on OANDA XAUUSD H1 2024-10->2026-10,
all zones both directions (exactly as tested):

  limit-at-touch, fill requires >=0.10 ATR edge penetration: +32.0R, PF 1.07
  limit-at-touch, fill requires >=0.05 ATR edge penetration: +106.1R, PF 1.21
  confirmation entry (wait for rejection):                    -71.4R, PF 0.82

Rules — EXACTLY as tested (the iron rule: no mixing):
  - FVGTracker feeds closed bars; each zone is traded at most ONCE
    (first touch). No repainting, strictly causal.
  - Entry: resting limit at the NEAR edge (bullish: zone top; bearish:
    zone bottom).
  - Fill realism: a touch bar fills ONLY if it penetrates the edge by at
    least MIN_FILL_PENETRATION_ATR x ATR. A graze that reverses is a miss,
    not a fill. Fill price is the limit edge (price guarantee, not fill
    guarantee).
  - Stop: far extreme +/- 0.25 ATR (catastrophic backstop).
  - Invalidation: close through the zone -> exit immediately at close.
  - Target: 1.5R fixed. Trades unresolved after 48 bars are dropped,
    as in the validation script (no timeout exits were booked).
  - Risk: 1R per trade; results in R multiples.

OPEN (not part of this validation): the long-only + bull-EMA-stack gate
from the hybrid engine, and the long/short asymmetry measured elsewhere.
Adding any of that is a new experiment, not a refactor.

Live path note: the paper executor fills at next bar's open, not at the
limit edge — the loop convention (same as fvg_midline). The batch path
grades edge fills. The parity test proves the SIGNAL sets are identical
between paths; the fill-price gap between edge and next-open is measured
by the live book, not assumed away.
"""
from __future__ import annotations

import numpy as np

from ..contextmap import HuntingGround
from ..golden.detector import GoldenSignal
from ..backtest.harness import Outcome
from ..v2_confluence.fvg import FVGTracker, FVGZone
from .base import Candidate, EngineContext, Playbook

#: Validated instruments (XAUUSD H1).
VALIDATED_INSTRUMENTS = ("XAUUSD", "GC=F")

#: 14-bar mean true range, same formula as scripts/test_fvg_entry_ab.py.
ATR_N = 14
#: Catastrophic backstop buffer beyond the far zone extreme.
STOP_BUFFER_ATR = 0.25
#: Fill realism: conservative end of the validated 0.05-0.10 band.
MIN_FILL_PENETRATION_ATR = 0.10
#: Fixed target, in R.
TARGET_R = 1.5
#: Maximum hold after entry (trades still open here were dropped in
#: the validation script — mirrored exactly).
MAX_HOLD_BARS = 48

#: Base score is a documented placeholder, NOT a measured score — the
#: sleeve was validated by R totals, not by scoring.  OOS promotion must
#: calibrate it; until then it only ranks sleeve-vs-sleeve.
SLEEVE_BASE_SCORE = 60.0


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


def _first_touch_events(high: np.ndarray, low: np.ndarray) -> list:
    """Feed the tracker bar-by-bar, return (zone, touch_idx) for the
    FIRST touch of each zone — verbatim logic from the validated A/B
    script so the sleeve can never drift from it."""
    n = len(high)
    tracker = FVGTracker(max_age_bars=100)
    events = []
    seen = set()  # one zone per formed bar -> formed_idx is unique
    for i in range(n):
        tracker.on_bar(float(high[i]), float(low[i]), i)
        for z in tracker.active():
            if i <= z.formed_idx:
                continue
            if z.formed_idx in seen:
                continue
            if z.direction == 1:
                touched = low[i] <= z.top and high[i] >= z.bottom
            else:
                touched = high[i] >= z.bottom and low[i] <= z.top
            if touched:
                seen.add(z.formed_idx)
                events.append((z, i))
    return events


def _simulate(zone: FVGZone, touch_idx: int,
              o: np.ndarray, h: np.ndarray, l: np.ndarray,
              c: np.ndarray) -> tuple | None:
    """The validated trade machine.  Returns
    (entry, stop, target, r_multiple, exit_index, exit_reason) or None
    when the touch bar fails the penetration requirement (missed fill)
    or the trade is unresolved after MAX_HOLD_BARS (dropped, as in the
    validation script)."""
    n = len(c)
    a = _atr(h, l, c, touch_idx)
    if a <= 0:
        return None
    buf = STOP_BUFFER_ATR * a
    min_pen = MIN_FILL_PENETRATION_ATR * a
    d = zone.direction
    edge = zone.top if d == 1 else zone.bottom      # near edge
    extreme = zone.bottom if d == 1 else zone.top   # far edge

    # --- fill realism: penetration, not a graze ---
    if d == 1:
        if not l[touch_idx] <= edge - min_pen:
            return None
    else:
        if not h[touch_idx] >= edge + min_pen:
            return None
    entry = edge

    stop = extreme - d * buf
    risk = (entry - stop) * d
    if risk <= 0:
        return None
    target = entry + TARGET_R * risk * d

    # --- walk forward: invalidation first, then stop/target stop-first ---
    r, exit_idx, reason = None, None, None
    for k in range(touch_idx + 1, min(touch_idx + MAX_HOLD_BARS + 1, n)):
        if d == 1 and c[k] < zone.bottom:
            r, exit_idx, reason = (c[k] - entry) / risk, k, "invalidation"
            break
        if d == -1 and c[k] > zone.top:
            r, exit_idx, reason = (entry - c[k]) / risk, k, "invalidation"
            break
        if d == 1:
            if l[k] <= stop:
                r, exit_idx, reason = (stop - entry) / risk, k, "stop"
                break
            if h[k] >= target:
                r, exit_idx, reason = TARGET_R, k, "target"
                break
        else:
            if h[k] >= stop:
                r, exit_idx, reason = (entry - stop) / risk, k, "stop"
                break
            if l[k] <= target:
                r, exit_idx, reason = TARGET_R, k, "target"
                break
    if r is None:
        return None  # unresolved after 48 bars: dropped, as validated
    return entry, stop, target, r, exit_idx, reason


def _arrays(df):
    return (
        df["open"].to_numpy(dtype=float),
        df["high"].to_numpy(dtype=float),
        df["low"].to_numpy(dtype=float),
        df["close"].to_numpy(dtype=float),
    )


class FVGLimitPlaybook(Playbook):
    name = "fvg_limit"
    description = (
        "Limit at the FVG near edge on first touch (validated A/B winner), "
        "0.10 ATR penetration fill model, 0.25 ATR stop, 1.5R target"
    )
    live = True   # paper-live capable; wiring into the loop's exec_set is
                  # the owner's decision
    live_threshold = 0.0  # every fill is graded
    context_description = (
        "XAUUSD H1, any session. The experiment measured no regime, HTF, or "
        "session restriction — the playbook declares no hunting ground of "
        "its own: it trades every first FVG touch that meets the fill "
        "model, exactly as validated."
    )
    session_scope = "any"
    hunting_ground = HuntingGround()

    # ------------------------------------------------------- batch path
    def candidates(self, ctx: EngineContext) -> list[Candidate]:
        o, h, l, c = _arrays(ctx.df)
        out: list[Candidate] = []
        for zone, touch_idx in _first_touch_events(h, l):
            sim = _simulate(zone, touch_idx, o, h, l, c)
            if sim is None:
                continue
            entry, stop, target, r, exit_idx, reason = sim
            d = zone.direction
            sig = GoldenSignal(
                index=touch_idx,
                timestamp=ctx.df.index[touch_idx],
                direction=d,
                score=SLEEVE_BASE_SCORE,
                kind="fvg_limit",
                factors=["fvg_first_touch"],
                entry_zone=(entry, entry),
                entry_ref=entry,
                invalidation=stop,
                targets=[target],
                atr=_atr(h, l, c, touch_idx),
            )
            outcome = Outcome(
                signal=sig, entry_index=touch_idx, exit_index=exit_idx,
                exit_reason=reason, r_net=r,
                hold_bars=exit_idx - touch_idx,
            )
            side = "long" if d == 1 else "short"
            thesis = (
                f"fvg_limit {side} (touch bar {touch_idx}): limit at near "
                f"edge {entry:.1f} of zone {zone.bottom:.1f}-{zone.top:.1f} "
                f"(formed bar {zone.formed_idx}), fill required "
                f">={MIN_FILL_PENETRATION_ATR:.2f} ATR penetration, stop "
                f"{stop:.1f}, target {target:.1f} ({TARGET_R}R)"
            )
            out.append(Candidate(
                playbook=self.name, outcome=outcome,
                base_score=SLEEVE_BASE_SCORE, thesis=thesis,
                context={
                    "touch_index": touch_idx,
                    "edge": entry, "stop": stop,
                    "zone_top": zone.top, "zone_bottom": zone.bottom,
                    "zone_formed_idx": zone.formed_idx,
                    "direction": d,
                }))
        return out

    def check_live_signal(self, signal, ctx: EngineContext,
                          context: dict | None = None) -> tuple[bool, str]:
        if ctx.instrument not in VALIDATED_INSTRUMENTS:
            return False, (
                f"validated universe is XAUUSD H1, got {ctx.instrument!r}")
        c = context or getattr(signal, "live_context", None) or {}
        need = ("touch_index", "edge", "stop", "direction")
        missing = [k for k in need if k not in c]
        if missing:
            return False, f"missing setup facts: {missing}"
        return True, ""

    # ------------------------------------------------------- live path
    def __init__(self) -> None:
        # Resting-limit working state: the tracker holds zone lifecycle,
        # the consumed set guarantees one trade per zone (first touch
        # only, as validated).
        self._tracker = FVGTracker(max_age_bars=100)
        self._consumed: set[int] = set()  # zone.formed_idx, traded
        self._processed = -1              # last bar run through the machine

    def _process_bar(self, o, h, l, c, t, collect) -> list:
        """One bar of the resting-limit machine.

        Bars <= t are always fed to the tracker (state advances); a
        fill is emitted only for bar t itself (``collect``), mirroring
        the fvg_midline pattern so the loop trades the latest close.
        """
        out = []
        self._tracker.on_bar(float(h[t]), float(l[t]), t)
        for z in self._tracker.active():
            if t <= z.formed_idx or z.formed_idx in self._consumed:
                continue
            if z.direction == 1:
                touched = l[t] <= z.top and h[t] >= z.bottom
            else:
                touched = h[t] >= z.bottom and l[t] <= z.top
            if not touched:
                continue
            self._consumed.add(z.formed_idx)
            if not collect:
                continue
            sim = _simulate(z, t, o, h, l, c)
            if sim is None:
                continue  # graze: queue never cleared, zone consumed
            entry, stop, target, r, exit_idx, reason = sim
            d = z.direction
            sig = GoldenSignal(
                index=t, timestamp=None, direction=d,
                score=SLEEVE_BASE_SCORE, kind="fvg_limit",
                factors=["fvg_first_touch"],
                entry_zone=(entry, entry), entry_ref=entry,
                invalidation=stop, targets=[target],
                atr=_atr(h, l, c, t),
            )
            sig.live_context = {
                "touch_index": t, "edge": entry, "stop": stop,
                "zone_top": z.top, "zone_bottom": z.bottom,
                "zone_formed_idx": z.formed_idx, "direction": d,
                "r_net_batch": r, "exit_reason_batch": reason,
                "exit_index_batch": exit_idx,
            }
            out.append(sig)
        return out

    def live_signal(self, states: list, t: int, df=None) -> list:
        """Incremental resting-limit check at closed bar ``t``.

        Returns the signal(s) whose limit filled on bar ``t`` — the same
        signal the batch path books the trade from.
        """
        if df is None or t < 0:
            return []
        o, h, l, c = _arrays(df)
        if t >= len(o):
            return []
        # Never rewind the cursor: after a rebase replay the fresh
        # loop re-calls live_signal for the tail bars, and an
        # unconditional `self._processed = t` used to move the cursor
        # BACKWARDS (e.g. 1999 -> 1997). The next forward call then
        # re-fed bar 1998 to the strict-order FVGTracker (last=1999)
        # and crashed every tick. Already-processed bars return [].
        if t <= self._processed:
            return []
        out: list = []
        for t2 in range(self._processed + 1, t + 1):
            out.extend(self._process_bar(o, h, l, c, t2,
                                         collect=(t2 == t)))
        self._processed = t
        return out

    def export_state(self) -> dict:
        return {"processed": self._processed,
                "consumed": sorted(self._consumed),
                "tracker": self._tracker.to_dict()}

    def import_state(self, state: dict) -> None:
        state = state or {}
        self._processed = int(state.get("processed", -1))
        self._consumed = set(state.get("consumed", []))
        tracker_state = state.get("tracker")
        # {} means RESET (fresh load, or rebase fast-forward in runner.py).
        # Must create a FRESH tracker, never keep the stale one: rebase
        # replays from idx=0, and a stale tracker with last_idx=1999
        # crashes on out-of-order bars. This was the previous bug.
        if tracker_state is not None:
            self._tracker = FVGTracker.from_dict(tracker_state)
        else:
            self._tracker = FVGTracker(max_age_bars=100)
