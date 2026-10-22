"""VReversalPlaybook — sharp V bottom/top (ICT/SMC capitulation reclaim).

Definition (precise, causal, no repainting):

Bullish V (bearish is the exact mirror):
  1. SPIKE — a bar that (a) makes a FRESH 20-bar low, (b) has range
     >= 0.8 x ATR14, (c) carries a lower wick >= 60% of its range
     (capitulation: sellers exhausted into the low, price rejected),
     and (d) arrives after a real down leg (close < close 5 bars ago
     - 0.5 x ATR).  The spike bar is the V bottom.
  2. RECLAIM — within 3 bars after the spike (the spike bar itself
     counts: a hammer IS the V bottom), a bar closes above the spike
     bar's midpoint with a bullish body (close > open).  The V is
     confirmed by the reclaim, not by the spike alone.
  3. ENTRY — on the reclaim bar's close.  Stop = spike low - 0.25 ATR.
     Invalidation = a close below the spike low (the V failed).
     Target = 1.5R fixed (shared exit discipline, playbooks/common.py).

Why this is not the sweep: a sweep is a wick through a level that
closes back inside; a V is a capitulation spike + reclaim of the spike
bar's midpoint, with no level required.  Different structures, no
overlap by construction.

Exits: shared discipline (common.py) — stop = invalidation +/- 0.25
ATR, target 1.5R, invalidation exit at close, 48-bar max hold.

VALIDATION (scripts/validate_new_playbooks.py, OANDA XAUUSD H1
2024-10 -> 2026-10, 11987 bars): 163 trades, 35% win, total R -39.4,
PF 0.68, negative in BOTH halves (-18.6 / -20.8).  That is the
historical MAP, kept as measured: standalone on H1 under the shared
fixed-exit discipline, capitulation spikes reclaimed, then failed.
STATUS: LIVE on the demo (owner directive 2026-10-22).  The map
reading does not gate execution; forward live-demo performance is
the verdict — the live question is the V as a trigger at HTF levels
with displacement behind it, which the standalone frame never
measured.
"""
from __future__ import annotations

import numpy as np

from ..contextmap import HuntingGround
from ..golden.detector import GoldenSignal
from ..backtest.harness import Outcome
from .base import Candidate, EngineContext, Playbook
from .common import (MAX_HOLD_BARS, STOP_BUFFER_ATR, TARGET_R, atr,
                     simulate_exit)

#: Validated instruments (XAUUSD H1, like the other sleeves).
VALIDATED_INSTRUMENTS = ("XAUUSD", "GC=F")
#: Base score is a documented placeholder, NOT a measured score — the
#: sleeve is validated by R totals, not by scoring (fvg_limit rule).
SLEEVE_BASE_SCORE = 60.0
#: Fresh-low lookback for the spike (bars).
FRESH_LOW_BARS = 20
#: Reclaim window after the spike bar (bars, inclusive of the spike bar).
RECLAIM_BARS = 3
#: Spike bar: range >= this x ATR14.
SPIKE_RANGE_ATR = 0.8
#: Spike bar: rejection wick >= this fraction of the range.
SPIKE_WICK_FRAC = 0.6
#: Preceding down leg: close < close-5 - this x ATR.
LEG_DECLINE_ATR = 0.5


class _Engine:
    """Streaming V detector: pending spikes await their reclaim."""

    def __init__(self) -> None:
        # Pending spikes: one per direction (the freshest wins).
        self.pending: list[dict] = []

    def process(self, o, h, l, c, t: int, collect: bool) -> list[dict]:
        out: list[dict] = []
        a = atr(h, l, c, t)
        if a <= 0 or t < FRESH_LOW_BARS:
            return out
        spike = self._detect_spike(o, h, l, c, t, a)
        if spike is not None:
            # Freshest spike per direction wins: the newest low is
            # the relevant V bottom, the older one is superseded.
            self.pending = [p for p in self.pending
                            if p["direction"] != spike["direction"]]
            self.pending.append(spike)
        for sp in list(self.pending):
            if t > sp["expiry"]:
                self.pending.remove(sp)
                continue
            d = sp["direction"]
            if d == 1 and c[t] > sp["mid"] and c[t] > o[t]:
                self.pending.remove(sp)
                if collect:
                    out.append(self._setup(sp, float(c[t]), a, t))
            elif d == -1 and c[t] < sp["mid"] and c[t] < o[t]:
                self.pending.remove(sp)
                if collect:
                    out.append(self._setup(sp, float(c[t]), a, t))
        return out

    def _detect_spike(self, o, h, l, c, t: int, a: float) -> dict | None:
        rng = float(h[t] - l[t])
        if rng < SPIKE_RANGE_ATR * a:
            return None
        lo = max(0, t - FRESH_LOW_BARS)
        # Bullish V: fresh low + rejection wick + preceding decline.
        if float(l[t]) <= float(np.min(np.asarray(l[lo:t], dtype=float))) \
                and float(min(o[t], c[t]) - l[t]) >= SPIKE_WICK_FRAC * rng \
                and float(c[t]) < float(c[t - 5]) - LEG_DECLINE_ATR * a:
            return {"direction": 1, "spike_idx": t,
                    "extreme": float(l[t]),
                    "mid": float((h[t] + l[t]) / 2.0),
                    "expiry": t + RECLAIM_BARS}
        # Bearish V: exact mirror.
        if float(h[t]) >= float(np.max(np.asarray(h[lo:t], dtype=float))) \
                and float(h[t] - max(o[t], c[t])) >= SPIKE_WICK_FRAC * rng \
                and float(c[t]) > float(c[t - 5]) + LEG_DECLINE_ATR * a:
            return {"direction": -1, "spike_idx": t,
                    "extreme": float(h[t]),
                    "mid": float((h[t] + l[t]) / 2.0),
                    "expiry": t + RECLAIM_BARS}
        return None

    @staticmethod
    def _setup(sp: dict, entry: float, a: float, t: int) -> dict:
        d = sp["direction"]
        stop = sp["extreme"] - d * STOP_BUFFER_ATR * a
        risk = (entry - stop) * d
        return {
            "direction": d, "entry": entry, "stop": float(stop),
            "invalidation": float(sp["extreme"]),
            "target": float(entry + TARGET_R * risk * d),
            "spike_idx": sp["spike_idx"], "reclaim_idx": t, "atr": a,
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
        index=setup["reclaim_idx"], timestamp=timestamp, direction=d,
        score=SLEEVE_BASE_SCORE, kind="v_reversal",
        factors=["v_reversal"],
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


class VReversalPlaybook(Playbook):
    name = "v_reversal"
    description = (
        "Capitulation spike (fresh low/high + rejection wick) + reclaim "
        "of the spike midpoint within 3 bars; enter on the reclaim close"
    )
    live = True  # measured PF 0.68 standalone (the map, in the
    # docstring); promoted to live demo by owner directive 2026-10-22
    # for forward testing — the demo decides.
    live_threshold = 0.0  # every reclaim is graded
    context_description = (
        "XAUUSD H1, any session. Entry is the reclaim close; stop is the "
        "spike extreme +/- 0.25 ATR; target 1.5R; invalidation is a close "
        "through the spike extreme; 48-bar max hold."
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
                    f"v_reversal {side} (reclaim bar {t}): spike bar "
                    f"{setup['spike_idx']} extreme "
                    f"{setup['invalidation']:.1f}, reclaim close "
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
        need = ("spike_idx", "reclaim_idx", "direction")
        missing = [k for k in need if k not in c]
        if missing:
            return False, f"missing setup facts: {missing}"
        if c["reclaim_idx"] - c["spike_idx"] > RECLAIM_BARS:
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
