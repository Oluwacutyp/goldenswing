"""Volatility-expansion family (MarketSituation Pass 7a,
SPEC §6.2) — the two hypotheses of the coil:

- ``vol_expansion_escape``: compression → expansion escape,
  entered at the ACCEPTANCE close (the bar after the
  displacement trigger whose whole range holds beyond the
  coil).  Stop just inside the coil edge; target the
  measured move (edge ± coil width).
- ``vol_failed_expansion``: the escape that gets rejected —
  a close back inside the coil after the trigger.  Faded
  toward the coil mid, then the opposite edge; stop beyond
  the escape's excursion extreme.

The measurement core is ``live/vol_expansion.py`` (pure,
causal).  This module is the playbook plumbing: frames in,
``GoldenSignal`` out, on the family's scheduled shadow
route (playbook_registry.ROUTE_VOLATILITY_EXPANSION).  The
runner threads two facts in per scan, in the
``set_h4_frame`` idiom:

- ``set_formation_tf`` — the clock this scan evaluates
  (the family is vol-state-defined on H4 / H1 / M15; the
  scan's frame IS the formation frame);
- ``set_situation`` — the current MarketSituation snapshot.
  Compression corroboration reads the situation's TFStates:
  with a snapshot threaded, a detection becomes a
  hypothesis only when ANOTHER timeframe's state also
  reads contracting (the multi-timeframe state the family
  is defined on — never a single-indicator read).  Without
  a snapshot (the research path), the corroborating read
  is the formation frame's own bedrock structure phase
  ("compression").  Which source decided rides on every
  signal's context — stated, never hidden.

``live = False``: the family's candidates are scored and
logged, never in the live P&L book — the scheduled route
is shadow (journal + scenario book only) until the
family's own promotion evaluation earns a separate owner
decision (SPEC §6 common contract).
"""
from __future__ import annotations

from ..backtest.harness import simulate_signal_v2
from ..contextmap import HuntingGround
from ..golden.detector import GoldenSignal
from ..live import vol_expansion as _vol
from ..live.regime import classify_regime
from .base import Candidate, EngineContext, Playbook

FAMILY = "volatility_expansion"


def _ts(frame, k):
    try:
        return str(frame.index[k])
    except Exception:
        return None


class _VolExpansionBase(Playbook):
    """Shared coil-sequence plumbing for the two
    hypotheses.  Subclasses name the resolution phase they
    trade (``_phase``: "accepted" | "rejected")."""

    regime_scope = "both"
    session_scope = "any"
    hunting_ground = HuntingGround(excluded=set(), bonuses={})
    live = False
    live_threshold = 60.0

    _phase: str = ""

    def __init__(self) -> None:
        self._processed = -1  # last bar run through live_signal
        # Threaded by the runner per scan (see the module
        # docstring).  Defaults are the bare/research path:
        # formation on the family's home clock, no snapshot.
        self._tf = "H1"
        self._situation = None

    def set_formation_tf(self, tf) -> None:
        """The formation clock of the scan about to run:
        the frame handed to live_signal is this TF's."""
        if tf:
            self._tf = str(tf)

    def set_situation(self, snapshot) -> None:
        """Thread the current MarketSituation snapshot (a
        dict in live/situation.py's schema, or None)."""
        self._situation = snapshot

    # -- corroboration ---------------------------------------

    def _corroboration(self, df, j: int) -> dict | None:
        """The multi-timeframe corroboration read (see the
        module docstring): the fact that turns a measured
        coil + sequence into a hypothesis.  None = the
        market's wider state does not corroborate — no
        signal, and the scan's no_setup record is the
        abstention."""
        snap = self._situation
        if isinstance(snap, dict):
            per_tf = snap.get("per_tf") or {}
            # SPEC §6.2 names the formation TF's phase among
            # the family's situation inputs: it rides on the
            # corroboration record as context (the bedrock
            # structure vocabulary — on the no-snapshot path
            # below, the phase read IS the corroboration).
            own = per_tf.get(self._tf) or {}
            phase = own.get("structure_phase")
            for tf in sorted(per_tf):
                if tf == self._tf:
                    continue
                state = per_tf.get(tf) or {}
                vol = state.get("volatility") or {}
                if vol.get("atr_state") == "contracting":
                    return {"source": "situation",
                            "by_tf": tf,
                            "formation_phase": phase,
                            "situation_id":
                                snap.get("situation_id")}
            return None
        try:
            regime = classify_regime(df.iloc[:j + 1])
        except Exception:
            return None
        if regime is not None \
                and regime.structure == "compression":
            return {"source": "frame_structure",
                    "by_tf": None, "situation_id": None}
        return None

    # -- arming (the runner's detection journal) --------------

    def pending_state(self, df) -> dict | None:
        """The family's live ARM as of the frame's last
        closed bar: a coil awaiting its trigger (phase
        "coil") or a triggered escape awaiting acceptance /
        rejection (phase "triggered") — with the
        corroboration read attached (an uncorroborated coil
        is a detection, not yet a hypothesis; the arm record
        says which).  None when no sequence is live.  The
        runner journals arm transitions from this read; the
        read itself is pure."""
        if df is None or len(df) < 2:
            return None
        j = len(df) - 1
        prep = _vol.PreparedFrame(df)
        seq = _vol.sequence_at(prep, j)
        if seq is None or seq["phase"] not in (
                "coil", "triggered"):
            return None
        corr = self._corroboration(df, j)
        coil = seq["coil"]
        out = {
            "phase": seq["phase"],
            "formation_tf": self._tf,
            "coil": {
                "start_ts": _ts(df, coil["start"]),
                "end_ts": _ts(df, coil["end"]),
                "bars": coil["bars"],
                "high": coil["high"], "low": coil["low"],
                "mid": coil["mid"], "width": coil["width"],
                "atr": coil["atr"],
                "atr_ratio": coil["atr_ratio"],
                "bw_percentile": coil["bw_percentile"]},
            "corroborated": corr is not None,
            "corroboration": corr,
            "direction": (int(seq["direction"])
                          if seq["phase"] == "triggered"
                          else None),
            "trigger": None,
        }
        if seq["phase"] == "triggered":
            out["trigger"] = {
                "bar_ts": _ts(df, seq["trigger_bar"]),
                "close": float(prep.c[seq["trigger_bar"]])}
        return out

    # -- detection -------------------------------------------

    def _detect_at(self, prep: _vol.PreparedFrame, j: int,
                   df) -> GoldenSignal | None:
        raise NotImplementedError

    def _scan(self, df, start: int, end: int):
        """Yield (j, signal) for bars start..end.  One
        causal code path for live_signal and candidates:
        the prepared series are functions of the frame;
        each bar's sequence read sees bars <= that bar."""
        prep = _vol.PreparedFrame(df)
        for j in range(start, end + 1):
            sig = self._detect_at(prep, j, df)
            if sig is not None:
                yield j, sig

    def live_signal(self, states: list, t: int, df=None) -> list:
        if df is None or t < 0 or t >= len(df):
            return []
        out = [sig for _j, sig in
               self._scan(df, self._processed + 1, t)]
        self._processed = t
        return out

    def candidates(self, ctx: EngineContext) -> list[Candidate]:
        df = ctx.df
        if df is None or len(df) < 60:
            return []
        out: list[Candidate] = []
        for j, sig in self._scan(df, 0, len(df) - 1):
            oc = simulate_signal_v2(df, sig)
            if oc is None:
                continue
            side = "long" if sig.direction > 0 else "short"
            out.append(Candidate(
                playbook=self.name, outcome=oc,
                base_score=float(sig.score),
                thesis=(f"{self.name} {side} @ "
                        f"{sig.entry_ref:.1f} (bar {j}): "
                        f"{'; '.join(sig.factors)}"),
                context=dict(sig.live_context or {})))
        return out

    def export_state(self) -> dict:
        return {"processed": self._processed}

    def import_state(self, state: dict) -> None:
        self._processed = int((state or {}).get("processed", -1))

    # -- shared signal construction ---------------------------

    def _context(self, df, prep, seq, corr, *, hypothesis,
                 terms) -> dict:
        coil = seq["coil"]
        entry_tf = _vol.ENTRY_TF_DOWN.get(self._tf, "M5")
        return {
            "family": FAMILY,
            "hypothesis": hypothesis,
            "formation_tf": self._tf,
            "entry_tf": entry_tf,
            # The two additive wiring facts the runner's
            # shared builders read (management_block_for's
            # context clock; the book's per-scenario
            # clocks).  No other family's signals carry
            # these keys — their blocks are untouched.
            "management_context_timeframe": self._tf,
            "scenario_clocks": {
                "formation_tf": self._tf,
                "refinement_tf": None,
                "trigger_tf": self._tf,
                "entry_tf": entry_tf,
                "management_tf": self._tf},
            "coil": {
                "start_ts": _ts(df, coil["start"]),
                "end_ts": _ts(df, coil["end"]),
                "bars": coil["bars"],
                "high": coil["high"], "low": coil["low"],
                "mid": coil["mid"], "width": coil["width"],
                "atr": coil["atr"],
                "atr_ratio": coil["atr_ratio"],
                "bw_percentile": coil["bw_percentile"]},
            "trigger": {
                "bar_ts": _ts(df, seq["trigger_bar"]),
                "close": float(prep.c[seq["trigger_bar"]])},
            "corroboration": dict(corr),
            "situation_id": corr.get("situation_id"),
            "invalidation_terms": list(terms),
        }

    def _signal(self, *, index, timestamp, direction, score,
                factors, entry_ref, invalidation, targets, atr,
                context) -> GoldenSignal:
        sig = GoldenSignal(
            index=index, timestamp=timestamp,
            direction=direction, score=float(score),
            kind=self.name, factors=list(factors),
            entry_zone=(float(entry_ref), float(entry_ref)),
            entry_ref=float(entry_ref),
            invalidation=float(invalidation),
            targets=[float(t) for t in targets],
            atr=float(atr))
        sig.live_context = dict(context)
        return sig


class VolExpansionEscapePlaybook(_VolExpansionBase):
    name = "vol_expansion_escape"
    _phase = "accepted"
    description = (
        "Volatility expansion escape: a multi-timeframe-"
        "corroborated volatility coil (Bollinger squeeze run "
        "+ ATR contraction) escapes on a displacement close; "
        "entry at the ACCEPTANCE close — the following bar "
        "whose whole range holds beyond the coil.  Stop just "
        "inside the coil edge; target the measured move."
    )
    context_description = (
        "Volatility state, any session: squeeze run >= 5 bars "
        "with ATR contracting on the formation TF (H4/H1/M15), "
        "corroborated by another TF's contracting state in "
        "the MarketSituation; displacement trigger close "
        "beyond the coil edge + 0.25 ATR; acceptance = a "
        "following close (within 3 bars) beyond the edge with "
        "the bar's whole range beyond it.  A close back "
        "inside voids the escape."
    )

    def _detect_at(self, prep, j, df):
        seq = _vol.sequence_at(prep, j)
        if seq is None or seq["phase"] != "accepted" \
                or seq["event_bar"] != j:
            return None
        corr = self._corroboration(df, j)
        if corr is None:
            return None
        geo = _vol.escape_geometry(prep, seq)
        d = int(seq["direction"])
        a = int(seq["event_bar"])
        score = 64.0 + (4.0 if prep.body_dominated(a, d)
                        else 0.0)
        terms = [
            f"{self._tf} close back inside the compression "
            f"range (beyond {geo['edge']:.2f}) voids the "
            "escape — the follow-through is the trade; the "
            "return inside is the thesis exit (Pass 6 plan)",
            "no acceptance within "
            f"{_vol.ACCEPT_WINDOW_BARS} formation bars of "
            "the trigger = drift, not expansion — no entry "
            "was taken",
        ]
        context = self._context(
            df, prep, seq, corr, hypothesis="expansion_escape",
            terms=terms)
        context["acceptance"] = {
            "bar_ts": _ts(df, a), "close": geo["entry"],
            "edge": geo["edge"]}
        return self._signal(
            index=j, timestamp=df.index[j], direction=d,
            score=score,
            factors=["vol_expansion_escape", "squeeze_coil",
                     "atr_contraction", "displacement_trigger",
                     "acceptance_close"],
            entry_ref=geo["entry"], invalidation=geo["stop"],
            targets=geo["targets"], atr=geo["atr"],
            context=context)


class VolFailedExpansionPlaybook(_VolExpansionBase):
    name = "vol_failed_expansion"
    _phase = "rejected"
    description = (
        "Failed volatility expansion: an escape trigger out "
        "of a corroborated coil is REJECTED — a close back "
        "inside the coil within the acceptance window.  Fade "
        "toward the coil mid, then the opposite edge; stop "
        "beyond the escape's excursion extreme."
    )
    context_description = (
        "The mirror of vol_expansion_escape: same coil and "
        "trigger; the resolution is a close back inside the "
        "coil before any acceptance (within 3 formation bars "
        "of the trigger).  A close back beyond the escape "
        "extreme voids the rejection."
    )

    def _detect_at(self, prep, j, df):
        seq = _vol.sequence_at(prep, j)
        if seq is None or seq["phase"] != "rejected" \
                or seq["event_bar"] != j:
            return None
        corr = self._corroboration(df, j)
        if corr is None:
            return None
        geo = _vol.failed_geometry(prep, seq)
        d = int(seq["direction"])      # the escape's direction
        r = int(seq["event_bar"])
        score = 62.0 + (4.0 if prep.body_dominated(r, -d)
                        else 0.0)
        terms = [
            f"{self._tf} close back beyond the escape "
            f"extreme {geo['extreme']:.2f} voids the "
            "rejection — the escape was real after all",
            "price re-accepting beyond the coil edge ends "
            "the fade (thesis exit, Pass 6 plan)",
        ]
        context = self._context(
            df, prep, seq, corr,
            hypothesis="failed_expansion", terms=terms)
        context["rejection"] = {
            "bar_ts": _ts(df, r), "close": geo["entry"],
            "escape_direction": d, "extreme": geo["extreme"]}
        return self._signal(
            index=j, timestamp=df.index[j], direction=-d,
            score=score,
            factors=["vol_failed_expansion", "squeeze_coil",
                     "atr_contraction", "displacement_trigger",
                     "rejection_close_inside"],
            entry_ref=geo["entry"], invalidation=geo["stop"],
            targets=geo["targets"], atr=geo["atr"],
            context=context)


VOL_PLAYBOOKS = (
    VolExpansionEscapePlaybook,
    VolFailedExpansionPlaybook,
)
