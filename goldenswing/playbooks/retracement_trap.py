"""RetracementTrapPlaybook — false-breakout trap (ICT/SMC).

Definition (precise, causal, no repainting):

Bullish trap = FALSE BREAKDOWN (bearish is the exact mirror):
  1. LEVEL — a CONFIRMED swing low (fractal, n=3: needs 3 bars after
     it to confirm — the level is fixed before it can ever be used).
  2. FALSE BREAK — a bar CLOSES below the level by >= 0.10 ATR.
     A wick through the level that closes back inside is NOT a false
     break — that is a sweep, and the sweep playbook owns it.  Trap
     and sweep are mutually exclusive by construction: trap requires
     a CLOSE beyond, sweep requires NO close beyond.
  3. TRAP FAILURE — within 3 bars after the false break, a bar closes
     back ABOVE the level.  The breakout traders who sold the break
     are trapped; the reclaim is the entry.
  4. ENTRY — on the reclaim bar's close.  Stop = the lowest low of
     the false-break bars - 0.25 ATR.  Invalidation = a close below
     that extreme (the breakdown was real after all).  Target = 1.5R
     fixed (shared exit discipline, playbooks/common.py).

One trap per level: once a level breaks (true or false), it is
consumed — no second trap on the same level.  Levels must be recent
(swing within the last 30 bars) so ancient levels never fire.

Exits: shared discipline (common.py) — stop = invalidation +/- 0.25
ATR, target 1.5R, invalidation exit at close, 48-bar max hold.

VALIDATION (scripts/validate_new_playbooks.py, OANDA XAUUSD H1
2024-10 -> 2026-10, 11987 bars): 434 trades, 38% win, total R -48.7,
PF 0.84, negative in BOTH halves (-14.8 / -33.9).  That is the
historical MAP, kept as measured: standalone on H1 under the shared
fixed-exit discipline, false-breakout traps resolved into the break
more often than they reversed.  STATUS: LIVE on the demo (owner
directive 2026-10-22).  The map reading does not gate execution;
forward live-demo performance is the verdict — the live question is
the trap as a trigger inside structure (HTF confluence, session
extremes), which the standalone frame never measured.
"""
from __future__ import annotations

import numpy as np

from ..contextmap import HuntingGround
from ..golden.detector import GoldenSignal
from ..backtest.harness import Outcome
from .base import Candidate, EngineContext, Playbook
from .common import (STOP_BUFFER_ATR, TARGET_R, atr, confirmed_swings,
                     simulate_exit)

#: Validated instruments (XAUUSD H1, like the other sleeves).
VALIDATED_INSTRUMENTS = ("XAUUSD", "GC=F")
#: Base score is a documented placeholder, NOT a measured score — the
#: sleeve is validated by R totals, not by scoring (fvg_limit rule).
SLEEVE_BASE_SCORE = 60.0
#: Fractal swing confirmation radius (bars each side).
SWING_N = 3
#: A false break must close beyond the level by this x ATR.
FALSE_BREAK_ATR = 0.10
#: Reclaim window after the false-break bar (bars).
RECLAIM_BARS = 3
#: A swing level older than this (bars) never fires.
LEVEL_MAX_AGE = 30


class _Engine:
    """Streaming trap detector: confirmed swings, pending false breaks."""

    def __init__(self) -> None:
        # Pending false breaks: level, extreme, direction, deadline.
        self.pending: list[dict] = []
        # Consumed swing levels (index of the swing bar) — one trap
        # per level, ever.
        self.consumed: set[int] = set()

    def process(self, o, h, l, c, t: int, collect: bool) -> list[dict]:
        out: list[dict] = []
        a = atr(h, l, c, t)
        if a <= 0 or t < 2 * SWING_N + 5:
            return out
        # 1) Advance pending extremes with bar t FIRST: the reclaim
        # bar itself can extend the false-break excursion, and the
        # stop must sit beyond the FULL excursion, not just the
        # break bar's.
        for pb in self.pending:
            if pb["direction"] == 1:
                pb["extreme"] = min(pb["extreme"], float(l[t]))
            else:
                pb["extreme"] = max(pb["extreme"], float(h[t]))
        # 2) Reclaim on bar t against pending false breaks.
        for pb in list(self.pending):
            if t > pb["deadline"]:
                self.pending.remove(pb)
                continue
            d = pb["direction"]
            if d == 1 and c[t] > pb["level"]:
                self.pending.remove(pb)
                if collect:
                    out.append(self._setup(pb, float(c[t]), a, t))
            elif d == -1 and c[t] < pb["level"]:
                self.pending.remove(pb)
                if collect:
                    out.append(self._setup(pb, float(c[t]), a, t))
        # 3) New false break on bar t (close beyond a confirmed level).
        fb = self._detect_false_break(h, l, c, t, a)
        if fb is not None:
            self.pending.append(fb)
        return out

    def _detect_false_break(self, h, l, c, t: int, a: float
                            ) -> dict | None:
        # Confirmed swings through bar t - SWING_N (a swing at i
        # confirms when bar i + n closes).
        highs, lows = confirmed_swings(h, l, t - SWING_N, n=SWING_N)
        buf = FALSE_BREAK_ATR * a
        # Bullish trap: close below a recent confirmed swing low.
        for idx, level in reversed(lows):
            if idx in self.consumed or t - idx > LEVEL_MAX_AGE:
                continue
            if float(c[t]) < level - buf:
                self.consumed.add(idx)
                return {"direction": 1, "level": float(level),
                        "extreme": float(l[t]), "swing_idx": idx,
                        "break_idx": t, "deadline": t + RECLAIM_BARS}
            break  # only the most recent swing low can break now
        # Bearish trap: exact mirror.
        for idx, level in reversed(highs):
            if idx in self.consumed or t - idx > LEVEL_MAX_AGE:
                continue
            if float(c[t]) > level + buf:
                self.consumed.add(idx)
                return {"direction": -1, "level": float(level),
                        "extreme": float(h[t]), "swing_idx": idx,
                        "break_idx": t, "deadline": t + RECLAIM_BARS}
            break
        return None

    @staticmethod
    def _setup(pb: dict, entry: float, a: float, t: int) -> dict:
        d = pb["direction"]
        stop = pb["extreme"] - d * STOP_BUFFER_ATR * a
        risk = (entry - stop) * d
        return {
            "direction": d, "entry": entry, "stop": float(stop),
            "invalidation": float(pb["extreme"]),
            "target": float(entry + TARGET_R * risk * d),
            "level": pb["level"], "swing_idx": pb["swing_idx"],
            "break_idx": pb["break_idx"], "reclaim_idx": t, "atr": a,
        }

    def to_dict(self) -> dict:
        return {"pending": [dict(p) for p in self.pending],
                "consumed": sorted(self.consumed)}

    @classmethod
    def from_dict(cls, state: dict | None) -> "_Engine":
        eng = cls()
        eng.pending = [dict(p) for p in (state or {}).get("pending", [])]
        eng.consumed = set((state or {}).get("consumed", []))
        return eng


def _to_signal(setup: dict, timestamp) -> GoldenSignal:
    d = setup["direction"]
    sig = GoldenSignal(
        index=setup["reclaim_idx"], timestamp=timestamp, direction=d,
        score=SLEEVE_BASE_SCORE, kind="retracement_trap",
        factors=["retracement_trap"],
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


class RetracementTrapPlaybook(Playbook):
    name = "retracement_trap"
    description = (
        "False breakdown/breakout (CLOSE beyond a confirmed swing level) "
        "that fails within 3 bars — enter on the reclaim close"
    )
    live = True  # measured PF 0.84 standalone (the map, in the
    # docstring); promoted to live demo by owner directive 2026-10-22
    # for forward testing — the demo decides.
    live_threshold = 0.0  # every trap failure is graded
    context_description = (
        "XAUUSD H1, any session. Entry is the reclaim close; stop is the "
        "false-break extreme +/- 0.25 ATR; target 1.5R; invalidation is "
        "a close through the false-break extreme; 48-bar max hold."
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
                    f"retracement_trap {side} (reclaim bar {t}): false "
                    f"break of swing level {setup['level']:.1f} at bar "
                    f"{setup['break_idx']}, reclaim close "
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
        need = ("swing_idx", "break_idx", "reclaim_idx", "direction",
                "level")
        missing = [k for k in need if k not in c]
        if missing:
            return False, f"missing setup facts: {missing}"
        if c["reclaim_idx"] - c["break_idx"] > RECLAIM_BARS:
            return False, "reclaim outside the 3-bar window"
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
