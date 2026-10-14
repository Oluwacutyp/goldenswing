"""InversionPlaybook — FVG inversion retest (ICT/SMC).

Definition (precise, causal, no repainting):

A Fair Value Gap is a support/resistance zone while it holds.  When
price CLOSES through the whole gap, the gap INVERTS: support becomes
resistance (bullish FVG broken down) or resistance becomes support
(bearish FVG broken up).  The entry is the retest of the inverted gap.

Bullish FVG inverted (bearish is the exact mirror):
  1. FVG — a bullish gap [bottom, top] from the FVGTracker (3-bar
     pattern, confirmed on close — the tracker's own detection).
  2. INVERSION — a bar CLOSES below the gap's bottom (through the
     whole gap, not a wick).  The gap is now resistance.  A close
     back above the top before any retest VOIDS the inversion (the
     break failed) — no trade.
  3. RETEST — a LATER bar (strictly after the inversion bar) trades
     into the inverted zone from below (high >= bottom) while its
     close stays below the top (no reclaim).  The retest is the
     entry: price came back to the broken support and was rejected.
  4. ENTRY — on the retest bar's close, at the near edge (the zone
     bottom for the short).  Stop = zone top + 0.25 ATR.
     Invalidation = a close above the zone top (reclaim — the
     inversion failed).  Target = 1.5R fixed (shared exit
     discipline, playbooks/common.py).

One trade per inverted zone.  Zones expire 100 bars after formation
(the tracker's max age).  The registry is the playbook's own: it
snapshots zones from the tracker on confirmation, so inversion /
retest detection never depends on the tracker's lifecycle semantics
(whether a broken zone stays ACTIVE, MITIGATED, or FILLED).

Why this is not fvg_limit: fvg_limit trades the FIRST TOUCH of a
holding gap (limit at the near edge, fill on penetration).  Inversion
trades the RETEST of a BROKEN gap — the opposite condition (the gap
failed first).  A gap is either holding or broken; the two setups
never fire on the same zone state.

Exits: shared discipline (common.py) — stop = invalidation +/- 0.25
ATR, target 1.5R, invalidation exit at close, 48-bar max hold.

VALIDATION (scripts/validate_new_playbooks.py, OANDA XAUUSD H1
2024-10 -> 2026-10, 11987 bars): 1282 trades, 49% win, total R +97.1,
PF 1.11, maxDD -45.3R.  PASSES the iron rule mechanically (PF >=
1.10, R > 0, >= 30 trades) — but MARGINAL and DECAYING: first half
+95.5R, second half +1.6R.  The edge is thin, high-frequency (~1
trade per 9 bars), and fading.  Executes with the standard gate;
treat as the weakest live sleeve and re-validate on the next window
before trusting it further.
"""
from __future__ import annotations

import numpy as np

from ..contextmap import HuntingGround
from ..golden.detector import GoldenSignal
from ..backtest.harness import Outcome
from ..v2_confluence.fvg import FVGTracker
from .base import Candidate, EngineContext, Playbook
from .common import (STOP_BUFFER_ATR, TARGET_R, atr, simulate_exit)

#: Validated instruments (XAUUSD H1, like the other sleeves).
VALIDATED_INSTRUMENTS = ("XAUUSD", "GC=F")
#: Base score is a documented placeholder, NOT a measured score — the
#: sleeve is validated by R totals, not by scoring (fvg_limit rule).
SLEEVE_BASE_SCORE = 60.0
#: Zone lifetime after formation (bars) — the tracker's max age.
ZONE_MAX_AGE = 100


class _Engine:
    """Streaming inversion detector: own zone registry + inversion state."""

    def __init__(self) -> None:
        self._tracker = FVGTracker(max_age_bars=ZONE_MAX_AGE)
        # formed_idx -> zone record.  Own registry: never depends on
        # the tracker's lifecycle (a broken gap may leave ACTIVE).
        self.registry: dict[int, dict] = {}
        self.consumed: set[int] = set()  # formed_idx, traded or voided

    def process(self, o, h, l, c, t: int, collect: bool) -> list[dict]:
        out: list[dict] = []
        # The tracker is fed EVERY bar, unconditionally: skipping a
        # bar would leave the tracker with a gap in its index stream
        # (it raises on out-of-order/duplicate indices and marks
        # gap-crossed zones suspect).  Only the inversion/retest
        # evaluation needs ATR.
        for z in self._tracker.on_bar(float(h[t]), float(l[t]), t):
            self.registry[z.formed_idx] = {
                "top": float(z.top), "bottom": float(z.bottom),
                "direction": int(z.direction), "formed_idx": z.formed_idx,
                "state": "holding", "inversion_idx": None}
        for key in list(self.registry):
            if t - self.registry[key]["formed_idx"] > ZONE_MAX_AGE:
                del self.registry[key]
        a = atr(h, l, c, t)
        if a <= 0:
            return out
        # Inversion / void / retest, in causal order.
        for key, rec in list(self.registry.items()):
            if key in self.consumed:
                continue
            if rec["state"] == "holding":
                if rec["direction"] == 1 and c[t] < rec["bottom"]:
                    rec["state"] = "inverted"
                    rec["inversion_idx"] = t
                elif rec["direction"] == -1 and c[t] > rec["top"]:
                    rec["state"] = "inverted"
                    rec["inversion_idx"] = t
            elif rec["state"] == "inverted":
                if t <= rec["inversion_idx"]:
                    continue  # retest must come strictly after inversion
                if rec["direction"] == 1:
                    # Broken support is now resistance: retest from
                    # below (touch) without reclaim (close above top).
                    if c[t] > rec["top"]:
                        self.consumed.add(key)  # reclaim: inversion void
                    elif h[t] >= rec["bottom"]:
                        self.consumed.add(key)
                        if collect:
                            out.append(self._setup(rec, a, t, -1))
                else:
                    if c[t] < rec["bottom"]:
                        self.consumed.add(key)
                    elif l[t] <= rec["top"]:
                        self.consumed.add(key)
                        if collect:
                            out.append(self._setup(rec, a, t, 1))
        return out

    @staticmethod
    def _setup(rec: dict, a: float, t: int, direction: int) -> dict:
        # Entry at the near edge (the side price retests from);
        # stop beyond the far edge; invalidation = reclaim close.
        if direction == -1:  # inverted bullish FVG -> short
            entry = rec["bottom"]
            stop = rec["top"] + STOP_BUFFER_ATR * a
            invalidation = rec["top"]
        else:  # inverted bearish FVG -> long
            entry = rec["top"]
            stop = rec["bottom"] - STOP_BUFFER_ATR * a
            invalidation = rec["bottom"]
        risk = (entry - stop) * direction
        return {
            "direction": direction, "entry": float(entry),
            "stop": float(stop), "invalidation": float(invalidation),
            "target": float(entry + TARGET_R * risk * direction),
            "zone_top": rec["top"], "zone_bottom": rec["bottom"],
            "zone_formed_idx": rec["formed_idx"],
            "inversion_idx": rec["inversion_idx"],
            "retest_idx": t, "atr": a,
        }

    def to_dict(self) -> dict:
        return {"registry": {str(k): dict(v)
                             for k, v in self.registry.items()},
                "consumed": sorted(self.consumed),
                "tracker": self._tracker.to_dict()}

    @classmethod
    def from_dict(cls, state: dict | None) -> "_Engine":
        eng = cls()
        state = state or {}
        eng.registry = {int(k): dict(v)
                        for k, v in state.get("registry", {}).items()}
        eng.consumed = set(state.get("consumed", []))
        tracker_state = state.get("tracker")
        if tracker_state is not None:
            eng._tracker = FVGTracker.from_dict(tracker_state)
        return eng


def _to_signal(setup: dict, timestamp) -> GoldenSignal:
    d = setup["direction"]
    sig = GoldenSignal(
        index=setup["retest_idx"], timestamp=timestamp, direction=d,
        score=SLEEVE_BASE_SCORE, kind="inversion",
        factors=["inversion"],
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


class InversionPlaybook(Playbook):
    name = "inversion"
    description = (
        "Broken FVG flips support <-> resistance; enter on the first "
        "retest of the inverted gap"
    )
    live = True  # validated: PF 1.11, +97.1R (marginal, decaying — see docstring)
    live_threshold = 0.0  # every inversion retest is graded
    context_description = (
        "XAUUSD H1, any session. Entry at the inverted gap's near edge; "
        "stop beyond the far edge +/- 0.25 ATR; target 1.5R; "
        "invalidation is a reclaim close through the far edge; "
        "48-bar max hold."
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
                    f"inversion {side} (retest bar {t}): gap "
                    f"{setup['zone_bottom']:.1f}-{setup['zone_top']:.1f} "
                    f"(formed {setup['zone_formed_idx']}) inverted at "
                    f"bar {setup['inversion_idx']}, retest entry "
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
        need = ("zone_formed_idx", "inversion_idx", "retest_idx",
                "direction")
        missing = [k for k in need if k not in c]
        if missing:
            return False, f"missing setup facts: {missing}"
        if c["retest_idx"] <= c["inversion_idx"]:
            return False, "retest must come strictly after inversion"
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
