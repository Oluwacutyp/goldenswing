"""Narrative playbook: the full institutional chain as one playbook.

    HTF Zone -> Liquidity Sweep -> Displacement -> CHoCH/BOS + FVG
    -> Entry (FVG / OB / breaker / mitigation block)

This is the chain executor.  It never fires on a single pattern:
``requires_narrative_chain = True`` is enforced by the confluence
gate (gate 0e) against the verdict's ``narrative`` state — a
signal whose chain is below ENTRY is blocked as
``narrative chain incomplete``.  The runner's own narrative
machine (``live/narrative.py``, advanced on every TF-close
refresh with the FULL stack) is the same machine; this playbook
is its trigger-path / backtest surface.

``live_signal`` reads multi-TF frames from ``df.attrs["frames"]``
when the runner attaches them; with a bare single frame it
treats that frame as the H1 leg and the chain can still arm on
its swings (documented degradation — the live runner always
attaches the full stack).
"""

from __future__ import annotations

import pandas as pd

from ..backtest.harness import Outcome, simulate_signal_v2
from ..contextmap import HuntingGround
from ..golden.detector import GoldenSignal
from ..live.narrative import (ENTRY, NarrativeState, advance,
                              chain_check)
from .base import Candidate, EngineContext, Playbook

MIN_CHAIN_SCORE = 60.0


def _frames_from(df, states=None) -> dict:
    frames: dict = {}
    if df is not None:
        try:
            attached = dict(getattr(df, "attrs", {}) or {}).get(
                "frames") or {}
        except Exception:
            attached = {}
        if attached:
            frames.update(attached)
        if "H1" not in frames and len(df):
            frames["H1"] = df
    return frames


def _signal_from_state(state: NarrativeState, entry_frame) -> GoldenSignal | None:
    if state.state != ENTRY or not state.entry:
        return None
    entry_price = float(state.entry["price"])
    stop = float(state.entry["stop"])
    targets = [float(t) for t in state.entry.get("targets", [])]
    sig = GoldenSignal(
        index=int(len(entry_frame) - 1),
        timestamp=entry_frame.index[-1],
        direction=int(state.direction),
        score=72.0,
        kind="narrative",
        factors=["htf_zone", "liquidity_sweep", "displacement",
                 "choch_bos_fvg", "narrative_chain_complete"],
        entry_zone=(entry_price, entry_price),
        entry_ref=entry_price,
        invalidation=stop,
        targets=targets,
        atr=float(state.entry.get("atr", 0.0) or 0.0))
    sig.live_context = {
        "narrative_state": int(state.state),
        "narrative": state.describe(),
        "zone": dict(state.zone),
        "sweep": dict(state.sweep),
        "displacement": dict(state.displacement),
        "confirm": {k: (str(v) if not isinstance(v, dict) else v)
                    for k, v in dict(state.confirm).items()},
    }
    return sig


class NarrativePlaybook(Playbook):
    name = "narrative"
    description = ("Full institutional chain: HTF zone -> liquidity "
                   "sweep -> displacement -> CHoCH/BOS + FVG -> "
                   "entry at FVG/OB/breaker/mitigation block.")
    live = True
    regime_scope = "both"
    requires_narrative_chain = True
    live_threshold = MIN_CHAIN_SCORE
    context_description = (
        "Fires only with a COMPLETE narrative chain (ENTRY state): "
        "HTF MAJOR zone approached, liquidity swept, displacement "
        "with FVG, LTF CHoCH/BOS + FVG confirmed. Never on a "
        "single pattern alone.")
    session_scope = "any"
    hunting_ground = HuntingGround()

    def _chain_state(self, frames: dict) -> NarrativeState:
        return advance(NarrativeState(), frames)

    def live_signal(self, states: list, t: int, df=None) -> list:
        if df is None or len(df) == 0:
            return []
        frames = _frames_from(df)
        state = self._chain_state(frames)
        if state.state != ENTRY:
            return []
        entry_frame = frames.get("M15") or frames.get("M5") \
            or frames.get("H1") or df
        sig = _signal_from_state(state, entry_frame)
        if sig is None:
            return []
        return [sig]

    def candidates(self, ctx: EngineContext) -> list[Candidate]:
        df = ctx.df
        if df is None or len(df) < 40:
            return []
        frames = _frames_from(df)
        state = self._chain_state(frames)
        if state.state != ENTRY:
            return []
        entry_frame = frames.get("M15") or frames.get("M5") \
            or frames.get("H1") or df
        sig = _signal_from_state(state, entry_frame)
        if sig is None:
            return []
        oc = simulate_signal_v2(df, sig)
        if oc is None:
            return []
        side = "long" if sig.direction > 0 else "short"
        return [Candidate(
            playbook=self.name, outcome=oc,
            base_score=float(sig.score),
            thesis=(f"narrative {side} @ {sig.entry_ref:.1f}: "
                    f"{state.describe()}"),
            context={"narrative_state": int(state.state),
                     "narrative": state.describe(),
                     "zone": dict(state.zone)})]

    def _tactical_pattern(self, bars, direction: int,
                          trigger_ctx: dict) -> dict | None:
        """Trigger-path evaluation: bars may be a frames dict or
        a single DataFrame.  Returns the chain's entry geometry
        when the chain is complete at ``direction``, else None."""
        if isinstance(bars, dict):
            frames = dict(bars)
        elif isinstance(bars, pd.DataFrame):
            frames = _frames_from(bars)
        else:
            return None
        state = self._chain_state(frames)
        ok, _ = chain_check(state, direction)
        if not ok or not state.entry:
            return None
        entry_price = float(state.entry["price"])
        stop = float(state.entry["stop"])
        targets = [float(t) for t in state.entry.get("targets", [])]
        return {
            "score": 72.0,
            "invalidation": stop,
            "costs": {"stop_distance": abs(entry_price - stop)},
            "entry": entry_price, "stop": stop,
            "target": targets[0] if targets else None,
            "pattern": self.name,
            "location": state.zone.get("level"),
            "reason": state.describe(),
        }
