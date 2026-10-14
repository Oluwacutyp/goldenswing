"""KillzoneOpenPlaybook — session-open drive: range break + retest (ICT).

Definition (precise, causal, no repainting):

The day partitions into DST-aware sessions (reader/session_timeframes:
asia -> london -> overlap -> newyork -> off).  Two opens drive the
day: London open and New York open (the first OVERLAP bar).  Each
open trades the same structure:

  1. PRE-RANGE — the high/low of the session block that just ended:
     Asia block for the London open, London block for the NY open
     (first overlap bar).  The pre-range is fixed at the session
     boundary, before any open-drive bar can touch it.
  2. BREAKOUT — in the opening session, a bar CLOSES beyond the
     pre-range edge with displacement (body >= 0.5 x ATR14).  The
     first edge broken wins; the other edge is then void (a later
     close beyond the opposite edge kills the drive).
  3. RETEST — within 6 bars after the breakout, in the same session,
     a bar pulls back to the broken edge: its low touches the edge
     (low <= edge, within a 0.5 ATR wick tolerance below it) and its
     close holds the edge (close >= edge).  The retest is the entry:
     price came back to the broken range edge and held.
  4. ENTRY — on the retest bar's close, at the broken edge.  Stop =
     the pullback extreme (lowest low since the breakout bar) -
     0.25 ATR.  Invalidation = a close below the pullback extreme
     (the retest failed).  Target = 1.5R fixed (shared exit
     discipline, playbooks/common.py).

The setup dies at the session boundary or 6 bars after the breakout
without a retest — an open drive that never retests is not chased.

Why this is not trendline_retest: trendline retest enters on a
return touch of a broken TRENDLINE (a diagonal, any session).  The
killzone open enters on a return touch of a SESSION-RANGE edge at a
session open (a horizontal, two opens only).  Different levels,
different clock.

Exits: shared discipline (common.py) — stop = invalidation +/- 0.25
ATR, target 1.5R, invalidation exit at close, 48-bar max hold.

VALIDATION (scripts/validate_new_playbooks.py, OANDA XAUUSD H1
2024-10 -> 2026-10, 11987 bars): 191 trades, 69% win, total R +108.2,
PF 2.23, avg +0.57R, maxDD -13.1R, positive in BOTH halves (+56.8 /
+51.4).  PASSES the iron rule cleanly — the strongest new setup.
Executes.
"""
from __future__ import annotations

import numpy as np

from ..contextmap import HuntingGround
from ..golden.detector import GoldenSignal
from ..backtest.harness import Outcome
from ..reader.session_timeframes import session_of
from .base import Candidate, EngineContext, Playbook
from .common import (STOP_BUFFER_ATR, TARGET_R, atr, simulate_exit)

#: Validated instruments (XAUUSD H1, like the other sleeves).
VALIDATED_INSTRUMENTS = ("XAUUSD", "GC=F")
#: Base score is a documented placeholder, NOT a measured score — the
#: sleeve is validated by R totals, not by scoring (fvg_limit rule).
SLEEVE_BASE_SCORE = 60.0
#: Breakout bar: body >= this x ATR14 (displacement, not a wick).
BREAKOUT_BODY_ATR = 0.5
#: Retest window after the breakout bar (bars).
RETEST_BARS = 6
#: Retest wick tolerance below the broken edge (x ATR).
RETEST_TOLERANCE_ATR = 0.5
#: Sessions that arm an open-drive setup (London open, NY open).
OPEN_SESSIONS = ("london", "overlap")


class _Engine:
    """Streaming killzone-open detector: session blocks + armed setups."""

    def __init__(self) -> None:
        self.block_session: str | None = None
        self.block_high: float = float("-inf")
        self.block_low: float = float("inf")
        self.armed: dict | None = None

    def process(self, o, h, l, c, t: int, collect: bool,
                ts=None) -> list[dict]:
        out: list[dict] = []
        if ts is None:
            return out
        sess = session_of(ts[t])[0]
        # --- session-block bookkeeping (runs on EVERY bar, ATR or not).
        if sess != self.block_session:
            # Boundary: the just-ended block is the pre-range for an
            # open in the new session (London <- Asia, NY <- London).
            if sess in OPEN_SESSIONS and self.block_session is not None:
                self.armed = {
                    "session": sess,
                    "pre_low": self.block_low, "pre_high": self.block_high,
                    "breakout": None, "pullback": None,
                    "armed_idx": t}
            else:
                self.armed = None
            self.block_session = sess
            self.block_high = float(h[t])
            self.block_low = float(l[t])
        else:
            self.block_high = max(self.block_high, float(h[t]))
            self.block_low = min(self.block_low, float(l[t]))
        a = atr(h, l, c, t)
        if a <= 0 or self.armed is None:
            return out
        if sess != self.armed["session"]:
            return out  # setup belongs to its opening session only
        arm = self.armed
        body = abs(float(c[t] - o[t]))
        if arm["breakout"] is None:
            # First edge broken wins (needs displacement).
            if body >= BREAKOUT_BODY_ATR * a:
                if float(c[t]) > arm["pre_high"]:
                    arm["breakout"] = {"direction": 1,
                                       "edge": arm["pre_high"],
                                       "idx": t}
                    arm["pullback"] = None
                elif float(c[t]) < arm["pre_low"]:
                    arm["breakout"] = {"direction": -1,
                                       "edge": arm["pre_low"],
                                       "idx": t}
                    arm["pullback"] = None
            return out
        br = arm["breakout"]
        d = br["direction"]
        # Drive failed: close beyond the OPPOSITE pre-range edge.
        if (d == 1 and float(c[t]) < arm["pre_low"]) or \
                (d == -1 and float(c[t]) > arm["pre_high"]):
            self.armed = None
            return out
        # Retest window: 6 bars after the breakout bar, same session.
        if t - br["idx"] > RETEST_BARS:
            self.armed = None
            return out
        # Track the pullback extreme (bars AFTER the breakout bar).
        if d == 1:
            arm["pullback"] = float(l[t]) if arm["pullback"] is None \
                else min(arm["pullback"], float(l[t]))
        else:
            arm["pullback"] = float(h[t]) if arm["pullback"] is None \
                else max(arm["pullback"], float(h[t]))
        if t <= br["idx"]:
            return out  # retest must come strictly after the breakout
        tol = RETEST_TOLERANCE_ATR * a
        if d == 1 and float(l[t]) <= br["edge"] \
                and float(l[t]) >= br["edge"] - tol \
                and float(c[t]) >= br["edge"]:
            setup = self._setup(arm, a, t)
            self.armed = None
            if collect:
                out.append(setup)
        elif d == -1 and float(h[t]) >= br["edge"] \
                and float(h[t]) <= br["edge"] + tol \
                and float(c[t]) <= br["edge"]:
            setup = self._setup(arm, a, t)
            self.armed = None
            if collect:
                out.append(setup)
        return out

    @staticmethod
    def _setup(arm: dict, a: float, t: int) -> dict:
        br = arm["breakout"]
        d = br["direction"]
        if d == 1:
            entry = br["edge"]
            invalidation = arm["pullback"]
            stop = invalidation - STOP_BUFFER_ATR * a
        else:
            entry = br["edge"]
            invalidation = arm["pullback"]
            stop = invalidation + STOP_BUFFER_ATR * a
        risk = (entry - stop) * d
        return {
            "direction": d, "entry": float(entry), "stop": float(stop),
            "invalidation": float(invalidation),
            "target": float(entry + TARGET_R * risk * d),
            "session": arm["session"],
            "pre_low": arm["pre_low"], "pre_high": arm["pre_high"],
            "breakout_idx": br["idx"], "retest_idx": t, "atr": a,
        }

    def to_dict(self) -> dict:
        return {"block_session": self.block_session,
                "block_high": self.block_high,
                "block_low": self.block_low,
                "armed": self.armed}

    @classmethod
    def from_dict(cls, state: dict | None) -> "_Engine":
        eng = cls()
        state = state or {}
        eng.block_session = state.get("block_session")
        eng.block_high = float(state.get("block_high", float("-inf")))
        eng.block_low = float(state.get("block_low", float("inf")))
        eng.armed = state.get("armed")
        return eng


def _to_signal(setup: dict, timestamp) -> GoldenSignal:
    d = setup["direction"]
    sig = GoldenSignal(
        index=setup["retest_idx"], timestamp=timestamp, direction=d,
        score=SLEEVE_BASE_SCORE, kind="killzone_open",
        factors=["killzone_open"],
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


class KillzoneOpenPlaybook(Playbook):
    name = "killzone_open"
    description = (
        "London / NY open drive: pre-session range break with "
        "displacement, then retest of the broken edge — enter on the "
        "retest close"
    )
    live = True  # validated: PF 2.23, +108.2R, both halves positive
    live_threshold = 0.0  # every open-drive retest is graded
    context_description = (
        "XAUUSD H1, London-open and NY-open (first overlap bar) sessions "
        "only. Entry at the broken pre-range edge; stop beyond the "
        "pullback extreme +/- 0.25 ATR; target 1.5R; invalidation is a "
        "close through the pullback extreme; 48-bar max hold."
    )
    # The gate's single-session field cannot express two sessions:
    # "any" passes the gate, and check_live_signal enforces
    # london/overlap from the signal's own session fact.
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
            for setup in eng.process(o, h, l, c, t, collect=True,
                                     ts=ctx.df.index):
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
                    f"killzone_open {side} ({setup['session']} open, "
                    f"retest bar {t}): pre-range "
                    f"{setup['pre_low']:.1f}-{setup['pre_high']:.1f}, "
                    f"breakout bar {setup['breakout_idx']}, retest entry "
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
        need = ("session", "breakout_idx", "retest_idx", "direction",
                "pre_low", "pre_high")
        missing = [k for k in need if k not in c]
        if missing:
            return False, f"missing setup facts: {missing}"
        if c["session"] not in OPEN_SESSIONS:
            return False, (
                f"killzone_open fires on london/overlap opens only, "
                f"got {c['session']!r}")
        if c["retest_idx"] <= c["breakout_idx"]:
            return False, "retest must come strictly after breakout"
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
                    o, h, l, c, t2, collect=(t2 == t), ts=df.index):
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
