"""FVGMidlinePlaybook — the LuxAlgo sleeve (Phase 9; PAPER-LIVE).

User-mined entry, validated as a candidate sleeve (experiments/
CHoCH_FVG_MID_OB.md): fresh CHoCH -> resting limit at the midline of a
fresh, untraded FVG -> fixed-2R exit (+25.88%, PF 1.41, 97 trades,
positive in both halves).

Iron rule: the entry AND the fixed-2R exit are kept exactly as tested.
The opposing-OB exit variant diluted and is NOT ported.  Paper-live in
the paper loop (live_signal parity-proven); MT5-live only after OOS
promotes it (see CONFLUENCE.md).
"""
from __future__ import annotations

import numpy as np

from ..contextmap import HuntingGround
from ..experiments.choch_fvg import (
    CHOCH_FRESH_BARS,
    FIXED_TP_R,
    FVG_PAIR_BACK_BARS,
    MAX_RISK_ATR,
    MIN_RISK_ATR,
    ORDER_EXPIRY_BARS,
    STOP_BUFFER_ATR,
    SleeveFill,
    SleeveSetup,
    _fill_price,
    _fvg_alive,
    _to_signal,
    find_fill,
    scan_setups,
    simulate_trade,
)
from ..features.common import ohlcv_arrays
from .base import Candidate, EngineContext, Playbook

#: Base score is a documented placeholder, NOT a measured score — the
#: sleeve was validated without scoring.  OOS promotion must calibrate
#: it (see CONFLUENCE.md); until then it only ranks sleeve-vs-sleeve.
SLEEVE_BASE_SCORE = 60.0

#: Validated context envelope, from the experiment report: the CHoCH
#: was required fresh (≤24 bars) AT THE SETUP bar and the resting
#: limit lives ≤24 bars — so at the fill bar the CHoCH is at most
#: 48 bars old.  A fill beyond that envelope was never validated.
MAX_CHOCH_AGE_AT_FILL = 48
#: The stop-distance band the experiment enforced (scan_setups skips
#: outside it).  Re-verified here as the tripwire — imported from the
#: experiment so the band can never drift from the validated one.

VALIDATED_INSTRUMENTS = ("GC=F", "XAUUSD")


class FVGMidlinePlaybook(Playbook):
    name = "fvg_midline"
    description = (
        "Fresh CHoCH -> resting limit at an untraded FVG midline, "
        "fixed-2R exit (LuxAlgo sleeve, variant B)"
    )
    live = True   # paper-live: real live_signal, parity-proven
    live_threshold = 0.0  # every fill is graded
    context_description = (
        "Fresh CHoCH (≤24 bars at setup) + untraded FVG midline on the "
        "pullback side + stop 0.25–4.0 ATR (experiments/CHoCH_FVG_MID_OB.md). "
        "The report's honest finding: the CONTEXT (fresh CHoCH + untraded "
        "gap) was the edge, not the midline — the midline alone loses "
        "(-0.379R). Validated on XAUUSD H1; no session/regime restriction "
        "measured. Long/short asymmetry (+0.78R vs +0.16R shadow) matches "
        "the bullish window — a market artifact, not a context rule."
    )
    #: Session-agnostic: no session cell was decisively bad in the map.
    session_scope = "any"
    #: Hunting ground from the measured context map (97 trades).
    #: trend=down at the fill bar: -0.119R (n=52, negative in both halves)
    #: — excluded, and the sequential book IMPROVES without it (+25.88% →
    #: +34.61%, PF 1.41 → 2.47), so this is not the Phase-5 routing trap.
    #: Proven: trend=up +0.676R (+4), with-bias +0.482R (+2), trend_down
    #: phase +0.426R (+2).
    hunting_ground = HuntingGround(
        excluded={("trend", "down")},
        bonuses={("trend", "up"): 4.0,
                 ("htf", "with-bias"): 2.0,
                 ("phase", "trend_down"): 2.0},
    )

    def candidates(self, ctx: EngineContext) -> list[Candidate]:
        out: list[Candidate] = []
        setups = scan_setups(ctx.df, ctx.states)
        for setup in setups:
            fill = find_fill(ctx.df, setup)
            if fill is None:
                continue  # expired or FVG died: cancelled, not a trade
            d = setup.direction
            target = fill.entry + d * FIXED_TP_R * setup.risk
            oc = simulate_trade(ctx.df, fill, target)
            side = "long" if d == 1 else "short"
            thesis = (
                f"fvg_midline {side} (setup bar {setup.index}, fill bar "
                f"{fill.fill_index}): fresh CHoCH @ {setup.choch_level:.1f} "
                f"(bar {setup.choch_index}), resting limit at untraded FVG "
                f"midline {setup.mid:.1f} (gap {setup.fvg_bottom:.1f}-"
                f"{setup.fvg_top:.1f}, created bar {setup.fvg_created_at}), "
                f"stop {setup.stop:.1f} ({setup.risk / setup.atr:.2f} ATR), "
                f"fixed 2R target {target:.1f}"
            )
            out.append(Candidate(
                playbook=self.name, outcome=oc,
                base_score=SLEEVE_BASE_SCORE, thesis=thesis,
                context={"setup_index": setup.index,
                         "fill_index": fill.fill_index,
                         "mid": setup.mid, "stop": setup.stop,
                         "risk_atr": setup.risk / setup.atr,
                         "choch_index": setup.choch_index,
                         "same_bar_stop": fill.same_bar_stop}))
        return out

    def check_live_signal(self, signal, ctx: EngineContext,
                          context: dict | None = None) -> tuple[bool, str]:
        # Independent re-verification of the validated context envelope.
        # The scanner guarantees all of these; if one fails here, the
        # scanner regressed — a system bug, never a trade.  The live
        # loop passes no Candidate context, so live_signal attaches the
        # setup facts to the signal itself (signal.live_context).
        if ctx.instrument not in VALIDATED_INSTRUMENTS:
            return False, (
                f"validated universe is XAUUSD H1, got {ctx.instrument!r}")
        c = context or getattr(signal, "live_context", None) or {}
        if "choch_index" in c:
            choch_age = signal.index - c["choch_index"]
            if choch_age > MAX_CHOCH_AGE_AT_FILL:
                return False, (
                    f"CHoCH {choch_age} bars old at fill — outside the "
                    f"validated freshness envelope (≤{MAX_CHOCH_AGE_AT_FILL})")
        if "risk_atr" in c:
            risk_atr = c["risk_atr"]
            if not (MIN_RISK_ATR <= risk_atr <= MAX_RISK_ATR):
                return False, (
                    f"stop {risk_atr:.2f} ATR outside the validated "
                    f"{MIN_RISK_ATR}–{MAX_RISK_ATR} band")
        return True, ""

    # ------------------------------------------------------- live signal
    def __init__(self) -> None:
        # Resting-limit working state: setups scanned on their setup bar,
        # worked bar-by-bar until fill, expiry, or FVG death.
        self._pending: list[SleeveSetup] = []
        self._processed = -1     # last bar run through the machine

    def _scan_bar(self, df, states, t: int,
                  arrays: dict) -> SleeveSetup | None:
        """The scan_setups loop body for a single bar ``t`` — verbatim
        logic, so the live scanner can never drift from the validated
        one.  Every field of the setup is knowable at the close of bar
        ``t``."""
        low, high, close = arrays["low"], arrays["high"], arrays["close"]
        st = states[t]
        ev = st.last_structure_event
        if ev is None or ev.kind != "CHOCH":
            return None
        if st.structure_event_age is None \
                or st.structure_event_age > CHOCH_FRESH_BARS:
            return None
        d = ev.direction
        atr = st.atr
        if not (atr and atr > 0 and np.isfinite(atr)):
            return None
        best = None
        for g in st.active_fvgs:
            if g.direction != d:
                continue
            if g.created_at > t \
                    or g.created_at < ev.index - FVG_PAIR_BACK_BARS:
                continue
            mid = g.midpoint
            # The limit must be a pullback: price on the favourable side.
            if d == 1 and not close[t] > mid:
                continue
            if d == -1 and not close[t] < mid:
                continue
            # Midline untraded since the FVG was created.
            seg = low[g.created_at: t + 1] if d == 1 \
                else high[g.created_at: t + 1]
            if d == 1 and bool((seg <= mid).any()):
                continue
            if d == -1 and bool((seg >= mid).any()):
                continue
            if best is None or g.created_at > best.created_at:
                best = g
        if best is None:
            return None
        mid = best.midpoint
        stop = ev.level - d * STOP_BUFFER_ATR * atr
        risk = (mid - stop) * d
        if risk < MIN_RISK_ATR * atr or risk > MAX_RISK_ATR * atr:
            return None
        return SleeveSetup(
            index=t, direction=d, mid=mid, stop=stop, risk=risk, atr=atr,
            fvg_top=best.top, fvg_bottom=best.bottom,
            fvg_created_at=best.created_at,
            choch_index=ev.index, choch_level=ev.level)

    def _process_bar(self, df, states, t: int, arrays: dict,
                     collect: bool) -> list:
        """One bar of the resting-limit machine.

        Pending setups are checked for FVG death then for a fill on bar
        ``t`` (the find_fill loop body, verbatim); afterwards the bar is
        scanned for a fresh setup.  Fills on past bars are applied to
        working state but not emitted (``collect=False``) — the loop
        only trades the latest close.
        """
        out = []
        low, high, open_ = arrays["low"], arrays["high"], arrays["open"]
        kept: list[SleeveSetup] = []
        for setup in self._pending:
            if t > setup.index + ORDER_EXPIRY_BARS:
                continue                      # expired: cancelled, not a trade
            if not _fvg_alive(low, high, setup.fvg_top, setup.fvg_bottom,
                              setup.direction, setup.fvg_created_at, t):
                continue                      # FVG died: cancelled
            d, mid = setup.direction, setup.mid
            entry = _fill_price(d, mid, open_[t])
            same_bar_stop = False
            filled = False
            if d == 1:
                if low[t] <= setup.stop:
                    filled, same_bar_stop = True, True
                elif low[t] <= mid:
                    filled = True
            else:
                if high[t] >= setup.stop:
                    filled, same_bar_stop = True, True
                elif high[t] >= mid:
                    filled = True
            if filled:
                if collect:
                    target = entry + d * FIXED_TP_R * setup.risk
                    fill = SleeveFill(setup=setup, fill_index=t,
                                      entry=entry,
                                      same_bar_stop=same_bar_stop)
                    sig = _to_signal(fill, target)
                    sig.timestamp = df.index[t]
                    sig.live_context = {
                        "setup_index": setup.index,
                        "fill_index": t,
                        "mid": setup.mid, "stop": setup.stop,
                        "risk_atr": setup.risk / setup.atr,
                        "choch_index": setup.choch_index,
                        "same_bar_stop": same_bar_stop,
                    }
                    out.append(sig)
                continue                      # filled: leave pending
            kept.append(setup)
        self._pending = kept
        setup = self._scan_bar(df, states, t, arrays)
        if setup is not None:
            self._pending.append(setup)
        return out

    def live_signal(self, states: list, t: int, df=None) -> list:
        """Incremental resting-limit check at closed bar ``t``.

        Returns the batch ``_to_signal`` signal(s) for setups whose
        limit filled on bar ``t`` — the same signal the backtest books
        the trade from.  The paper executor fills at the next bar's
        open ± half spread; the parity test proves the fill bars match.
        """
        if df is None or t < 0 or t >= len(states):
            return []
        # Never rewind the cursor (see fvg_limit: rebase fresh loop
        # re-calls the tail; rewinding re-emits duplicates).
        if t <= self._processed:
            return []
        arrays = ohlcv_arrays(df)
        out: list = []
        for t2 in range(self._processed + 1, t + 1):
            out.extend(self._process_bar(df, states, t2, arrays,
                                         collect=(t2 == t)))
        self._processed = t
        return out

    def export_state(self) -> dict:
        return {"processed": self._processed,
                "pending": [s.__dict__ for s in self._pending]}

    def import_state(self, state: dict) -> None:
        self._processed = int(state.get("processed", -1))
        self._pending = [SleeveSetup(**s)
                         for s in state.get("pending", [])]
