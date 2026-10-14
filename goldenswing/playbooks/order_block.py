"""OrderBlockPlaybook — last opposing candle before displacement (ICT).

Definition (precise, causal, no repainting):

Bullish OB (bearish is the exact mirror):
  1. DISPLACEMENT — a bar that (a) has body >= max(1.0 x ATR14,
     1.5 x 20-bar average body) and (b) CLOSES beyond a confirmed
     swing level (fractal, n=3) in the displacement direction.  The
     close-through is what makes it displacement, not a wick.
  2. ORDER BLOCK — the LAST bar in the 5 bars before the displacement
     bar whose direction opposes it (for a bullish displacement: the
     last down bar).  The OB zone is that candle's [low, high] —
     the last opposing candle before the move is where the
     institutional orders sat.  No opposing candle in the window =
     no OB (the displacement has no block to return to).
  3. RETRACEMENT — the FIRST bar after the displacement bar whose
     range trades into the OB zone (low <= top and high >= bottom).
     The retracement is the entry: price came back to the block.
  4. ENTRY — on the retracement bar's close, at the near edge (the
     OB top for a long — the first level price meets coming down).
     Stop = OB bottom - 0.25 ATR.  Invalidation = a close below the
     OB bottom (the block failed).  Target = 1.5R fixed (shared
     exit discipline, playbooks/common.py).

One trade per OB.  OBs expire 100 bars after formation (same max age
as FVG zones).  The displacement bar itself never counts as the
retracement — the touch must come strictly after it.

Why this is not fvg_midline: fvg_midline enters at the midline of a
fresh FVG after a CHoCH.  The OB entry is the last opposing candle
before a displacement close-through — a different structure (a
single candle's range, not a 3-bar gap), and it needs no CHoCH.

Exits: shared discipline (common.py) — stop = invalidation +/- 0.25
ATR, target 1.5R, invalidation exit at close, 48-bar max hold.

VALIDATION (scripts/validate_new_playbooks.py, OANDA XAUUSD H1
2024-10 -> 2026-10, 11987 bars): 495 trades, 36% win, total R -134.3,
PF 0.67, negative in BOTH halves (-56.3 / -78.0).  FAILS the iron
rule.  STAYS SHADOW — OB retracements on H1 get run through more
often than they hold; the block is not the edge by itself.  Do not
execute without a new experiment (e.g. OB + FVG confluence, or OB
only at HTF POI).
"""
from __future__ import annotations

import numpy as np

from ..contextmap import HuntingGround
from ..golden.detector import GoldenSignal
from ..backtest.harness import Outcome
from .base import Candidate, EngineContext, Playbook
from .common import (STOP_BUFFER_ATR, TARGET_R, atr, avg_body,
                     confirmed_swings, simulate_exit)

#: Validated instruments (XAUUSD H1, like the other sleeves).
VALIDATED_INSTRUMENTS = ("XAUUSD", "GC=F")
#: Base score is a documented placeholder, NOT a measured score — the
#: sleeve is validated by R totals, not by scoring (fvg_limit rule).
SLEEVE_BASE_SCORE = 60.0
#: Fractal swing confirmation radius (bars each side).
SWING_N = 3
#: Displacement bar: body >= this x ATR14.
DISP_BODY_ATR = 1.0
#: Displacement bar: body >= this x 20-bar average body.
DISP_BODY_AVG = 1.5
#: Look back this many bars for the last opposing candle.
OB_LOOKBACK = 5
#: OB lifetime after formation (bars).
OB_MAX_AGE = 100


class _Engine:
    """Streaming OB detector: displacement -> OB -> first retracement."""

    def __init__(self) -> None:
        # Pending OBs: zone, direction, formed bar, consumed on touch.
        self.pending: list[dict] = []

    def process(self, o, h, l, c, t: int, collect: bool) -> list[dict]:
        out: list[dict] = []
        a = atr(h, l, c, t)
        if a <= 0 or t < 2 * SWING_N + 5:
            return out
        # 1) Retracement touch on bar t against pending OBs.
        for ob in list(self.pending):
            if t - ob["formed_idx"] > OB_MAX_AGE:
                self.pending.remove(ob)
                continue
            if t <= ob["formed_idx"]:
                continue  # touch must come strictly after displacement
            if float(l[t]) <= ob["top"] and float(h[t]) >= ob["bottom"]:
                self.pending.remove(ob)
                if collect:
                    out.append(self._setup(ob, a, t))
        # 2) New displacement on bar t -> register its OB.
        ob = self._detect_displacement(o, h, l, c, t, a)
        if ob is not None:
            self.pending.append(ob)
        return out

    def _detect_displacement(self, o, h, l, c, t: int, a: float
                             ) -> dict | None:
        body = abs(float(c[t] - o[t]))
        if body < DISP_BODY_ATR * a:
            return None
        if body < DISP_BODY_AVG * avg_body(c, o, t):
            return None
        highs, lows = confirmed_swings(h, l, t - SWING_N, n=SWING_N)
        direction = 0
        if float(c[t]) > float(o[t]):  # bullish displacement
            # Close beyond a confirmed swing high.
            if highs and float(c[t]) > highs[-1][1] + 0.10 * a:
                direction = 1
        else:  # bearish displacement
            if lows and float(c[t]) < lows[-1][1] - 0.10 * a:
                direction = -1
        if direction == 0:
            return None
        # Last opposing candle in the lookback window.
        for j in range(t - 1, max(-1, t - 1 - OB_LOOKBACK), -1):
            opposing = (float(c[j]) < float(o[j])) if direction == 1 \
                else (float(c[j]) > float(o[j]))
            if opposing:
                return {
                    "direction": direction,
                    "top": float(h[j]), "bottom": float(l[j]),
                    "ob_idx": j, "formed_idx": t}
        return None  # no opposing candle: displacement has no block

    @staticmethod
    def _setup(ob: dict, a: float, t: int) -> dict:
        d = ob["direction"]
        if d == 1:
            entry = ob["top"]
            stop = ob["bottom"] - STOP_BUFFER_ATR * a
            invalidation = ob["bottom"]
        else:
            entry = ob["bottom"]
            stop = ob["top"] + STOP_BUFFER_ATR * a
            invalidation = ob["top"]
        risk = (entry - stop) * d
        return {
            "direction": d, "entry": float(entry), "stop": float(stop),
            "invalidation": float(invalidation),
            "target": float(entry + TARGET_R * risk * d),
            "ob_top": ob["top"], "ob_bottom": ob["bottom"],
            "ob_idx": ob["ob_idx"], "formed_idx": ob["formed_idx"],
            "retest_idx": t, "atr": a,
        }

    def to_dict(self) -> dict:
        return {"pending": [dict(p) for p in self.pending]}

    @classmethod
    def from_dict(cls, state: dict | None) -> "_Engine":
        eng = cls()
        eng.pending = [dict(p) for p in (state or {}).get("pending", [])]
        return eng


def _to_signal(setup: dict, timestamp) -> GoldenSignal:
    d = setup["direction"]
    sig = GoldenSignal(
        index=setup["retest_idx"], timestamp=timestamp, direction=d,
        score=SLEEVE_BASE_SCORE, kind="order_block",
        factors=["order_block"],
        entry_zone=(setup["entry"], setup["entry"]),
        entry_ref=setup["entry"], invalidation=setup["invalidation"],
        targets=[setup["target"]], atr=setup["atr"])
    sig.live_context = dict(setup)
    return sig


def _arrays(df):
    return (
        df["open"].to_numpy(dtype=float),
        df["high"].to_numpy(dtype=float),
        df["low"].to_numpy(dtype=float),
        df["close"].to_numpy(dtype=float),
    )


class OrderBlockPlaybook(Playbook):
    name = "order_block"
    description = (
        "Displacement close-through a swing level; enter on the first "
        "retracement into the last opposing candle (the order block)"
    )
    live = False  # FAILED validation (PF 0.67, -134.3R) — shadow only
    live_threshold = 0.0  # every OB retracement is graded
    context_description = (
        "XAUUSD H1, any session. Entry at the OB near edge; stop beyond "
        "the OB far edge +/- 0.25 ATR; target 1.5R; invalidation is a "
        "close through the OB far edge; 48-bar max hold."
    )
    session_scope = "any"
    hunting_ground = HuntingGround()

    def __init__(self) -> None:
        self._engine = _Engine()
        self._processed = -1

    # ------------------------------------------------------- batch path
    def candidates(self, ctx: EngineContext) -> list[Candidate]:
        o, h, l, c = _arrays(ctx.df)
        eng = _Engine()
        out: list[Candidate] = []
        for t in range(len(o)):
            for setup in eng.process(o, h, l, c, t, collect=True):
                sig = _to_signal(setup, ctx.df.index[t])
                sim = simulate_exit(
                    setup["direction"], setup["entry"], setup["stop"],
                    setup["invalidation"], setup["target"],
                    o, h, l, c, t)
                if sim is None:
                    continue
                r, exit_idx, reason = sim
                outcome = Outcome(
                    signal=sig, entry_index=t, exit_index=exit_idx,
                    exit_reason=reason, r_net=r,
                    hold_bars=exit_idx - t)
                side = "long" if setup["direction"] == 1 else "short"
                thesis = (
                    f"order_block {side} (retest bar {t}): OB "
                    f"{setup['ob_bottom']:.1f}-{setup['ob_top']:.1f} "
                    f"(candle {setup['ob_idx']}, displacement bar "
                    f"{setup['formed_idx']}), entry "
                    f"{setup['entry']:.1f}, stop {setup['stop']:.1f}, "
                    f"target {setup['target']:.1f} ({TARGET_R}R)")
                out.append(Candidate(
                    playbook=self.name, outcome=outcome,
                    base_score=SLEEVE_BASE_SCORE, thesis=thesis,
                    context=dict(setup)))
        return out

    def check_live_signal(self, signal, ctx: EngineContext,
                          context: dict | None = None) -> tuple[bool, str]:
        if ctx.instrument not in VALIDATED_INSTRUMENTS:
            return False, (
                f"validated universe is XAUUSD H1, got {ctx.instrument!r}")
        c = context or getattr(signal, "live_context", None) or {}
        need = ("ob_idx", "formed_idx", "retest_idx", "direction")
        missing = [k for k in need if k not in c]
        if missing:
            return False, f"missing setup facts: {missing}"
        if c["retest_idx"] <= c["formed_idx"]:
            return False, "retest must come strictly after displacement"
        if c["formed_idx"] - c["ob_idx"] > OB_LOOKBACK:
            return False, "OB candle outside the lookback window"
        return True, ""

    # ------------------------------------------------------- live path
    def live_signal(self, states: list, t: int, df=None) -> list:
        if df is None or t < 0:
            return []
        o, h, l, c = _arrays(df)
        if t >= len(o):
            return []
        if t <= self._processed:
            return []
        out: list = []
        for t2 in range(self._processed + 1, t + 1):
            for setup in self._engine.process(
                    o, h, l, c, t2, collect=(t2 == t)):
                out.append(_to_signal(setup, df.index[t2]))
        self._processed = t
        return out

    def export_state(self) -> dict:
        return {"processed": self._processed,
                "engine": self._engine.to_dict()}

    def import_state(self, state: dict) -> None:
        state = state or {}
        self._processed = int(state.get("processed", -1))
        self._engine = _Engine.from_dict(state.get("engine"))
