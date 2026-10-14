"""SweepReversalPlaybook — the anchor core (Phase 9).

The Phase-2 detector, unchanged: sweep -> displacement + CHoCH/BOS
confirmation, daily-bias veto, scored with Combo A + AOC families,
Phase-2 swing exits.  This playbook IS the live book; its machinery is
byte-identical to the validated anchor (+28.92%, 199 trades, PF 1.25).
"""
from __future__ import annotations

from ..backtest.harness import simulate_signal_v2
from ..backtest.validation import COMBO_A
from ..contextmap import HuntingGround
from ..golden.detector import DetectorConfig, GoldenSwingDetector
from .base import Candidate, EngineContext, Playbook

#: The validated anchor configuration.  Do not touch without a new
#: experiment — the confluence book's canary depends on it.
ANCHOR_FAMILIES = COMBO_A | {"aoc"}
ANCHOR_FLAGS = dict(require_sweep_confirmation=True, daily_veto=True)


#: Instruments the anchor was validated on.  The detector's gates are
#: the detector's tested machinery — the playbook contract covers what
#: is EXTERNAL to it: the validated universe and the book-membership
#: score rule.
VALIDATED_INSTRUMENTS = ("GC=F", "XAUUSD")


class SweepReversalPlaybook(Playbook):
    name = "sweep_reversal"
    description = (
        "Phase-2 detector: confirmed sweep reversals and trend-aligned "
        "continuations, scored with {relstr, volume, phases, aoc}"
    )
    live = True
    live_threshold = 60.0
    context_description = (
        "Validated on XAUUSD H1 (GC=F, Oct 2024–Oct 2026): sweep → "
        "displacement + CHoCH/BOS confirmation, daily-bias veto, score ≥ 60 "
        "(Combo A + AOC). No session/regime restriction was measured — the "
        "setup's own gates (confirmation, veto, score) ARE the required "
        "context. (BACKTEST2.md)"
    )
    #: Session-agnostic core: the context map (scripts/
    #: measure_context_maps.py) shows no decisively bad session — the
    #: anchor fires in every session.
    session_scope = "any"
    #: Hunting ground from the measured context map (199 trades):
    #: no exclusions; proven cells earn confluence bonuses.
    #: newyork +0.397R, expanded vol +0.329R, Friday +0.323R.
    hunting_ground = HuntingGround(
        excluded=set(),
        bonuses={("session", "newyork"): 2.0,
                 ("vol_regime", "expanded"): 2.0,
                 ("dow", "Fri"): 2.0},
    )

    def __init__(self) -> None:
        self.detector = GoldenSwingDetector(DetectorConfig(
            emit_floor=35.0, golden_threshold=self.live_threshold,
            feature_families=ANCHOR_FAMILIES, **ANCHOR_FLAGS))

    def check_live_signal(self, signal, ctx: EngineContext,
                          context: dict | None = None) -> tuple[bool, str]:
        # The detector owns its gates (confirmation, daily veto, score);
        # the contract covers what is external to it: the validated
        # universe.  A book-eligible candidate outside it means the
        # playbook is running somewhere it was never validated.
        if ctx.instrument not in VALIDATED_INSTRUMENTS:
            return False, (
                f"validated universe is XAUUSD H1, got {ctx.instrument!r}")
        return True, ""

    def candidates(self, ctx: EngineContext) -> list[Candidate]:
        out: list[Candidate] = []
        for sig in self.detector.scan(ctx.states):
            oc = simulate_signal_v2(ctx.df, sig)
            if oc is None:
                continue
            thesis = (
                f"{sig.kind} {'long' if sig.direction == 1 else 'short'} "
                f"@ score {sig.score:.0f} (bar {sig.index}): "
                + "; ".join(sig.factors)
            )
            out.append(Candidate(
                playbook=self.name, outcome=oc, base_score=sig.score,
                thesis=thesis,
                context={"kind": sig.kind, "factors": list(sig.factors),
                         "atr": sig.atr}))
        return out

    def live_signal(self, states: list, t: int, df=None) -> list:
        """Incremental detector evaluation at bar ``t`` — no
        pre-simulation.  The live executor manages the position with
        the playbook's Phase-2 exit policy."""
        if t < 0 or t >= len(states):
            return []
        sig = self.detector.evaluate(states[t])
        return [sig] if sig is not None else []
