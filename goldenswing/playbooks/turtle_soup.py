"""TurtleSoupPlaybook — failed breakout of a 20-bar extreme (ICT/SMC).

Definition (precise, causal, no repainting):

Bearish soup (bullish is the exact mirror):
  1. EXTREME — the highest high of the last 20 closed bars.
  2. SWEEP — a bar wicks beyond the extreme by >= 0.10 ATR but
     CLOSES back inside (close < extreme).  A bar that closes beyond
     is a real break, not a sweep.  A bar that sweeps BOTH extremes
     (a full-range expansion bar) is not a sweep setup at all.
  3. TRIGGER — within 5 bars after the sweep, a bar closes inside
     AND below the sweep bar's close (the fade begins).
  4. CONFIRMATION — within 10 bars after the sweep, price trades
     through the sweep bar's OPPOSITE extreme (low < sweep-bar low):
     the breakout traders are trapped.  A close beyond the sweep
     extreme (wick) at any point voids the setup (the break was real).
  5. FVG — at confirmation, the most recently formed FVG of the
     rejection direction (bearish FVG for a short), formed strictly
     after the sweep bar and not FILLED, is FROZEN (top / bottom /
     entry / stop / invalidation / target / ATR).  The rejection leg
     is the displacement move that trades through the opposite
     extreme — NOT the sweep leg.  No FVG -> no setup.
  6. ENTRY — armed for 12 bars after confirmation: the first bar
     whose range touches the frozen FVG AND whose close is back on
     the fade side of the near edge (short: close <= FVG bottom)
     enters at the near edge on that bar's close.  A touch whose
     close does NOT hold (close inside/through the FVG) is not a
     rejection: the setup stays armed and waits for a later holding
     retest or expiry.  Price never retraces -> no trade (expiry).
  7. STOP = sweep extreme (wick) +/- 0.25 ATR buffer.
     INVALIDATION = a close through the FVG far edge (against you).
     Target = 1.5R fixed (shared exit discipline, playbooks/common.py).

One trade per sweep.  Levels are frozen at confirmation, never
recomputed.  State is one dataclass per sweep, keyed by sweep bar
index, serializable for warm resume.

Why this is not retracement_trap: the trap needs a CLOSE beyond a
confirmed swing level; the soup needs a WICK beyond a 20-bar extreme
with the close back inside (trap and sweep are mutually exclusive by
construction).  Why not sweep_reversal: that sleeve fades the sweep
itself; the soup waits for the trigger + trade-through of the sweep
bar's opposite extreme + an FVG retracement — a strictly later,
strictly stronger structure.

Exits: shared discipline (common.py) — stop = invalidation +/- 0.25
ATR, target 1.5R, invalidation exit at close, 48-bar max hold.

VALIDATION (scripts/validate_new_playbooks.py, OANDA XAUUSD H1
2024-10 -> 2026-10, 11987 bars): 78 trades, 27% win, total R +2.1,
PF 1.07, avg +0.03R, maxDD -6.6R, halves -0.0 / +2.1.  That is the
historical MAP, kept as measured: under the old bar it read
MARGINAL (PF 1.07 vs the 1.10 reference line) — the rejection-leg
FVG retracement barely broke even on H1; the 3-bar imbalance did
not hold as reliably as a session pool (cf. Judas PF 1.93 on the
same window).  STATUS: LIVE on the demo (owner directive
2026-10-22).  The map reading does not gate execution; forward
live-demo performance is the verdict.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

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
#: 20-bar extreme lookback (closed bars, excluding the sweep bar).
LOOKBACK_BARS = 20
#: A sweep must wick beyond the extreme by this x ATR (dust wicks
#: are not manipulation — same margin as live/manipulation.py).
SWEEP_MARGIN_ATR = 0.10
#: Trigger window after the sweep bar (bars).
TRIGGER_BARS = 5
#: Confirmation window after the sweep bar (bars).
CONFIRM_BARS = 10
#: Armed window after the confirmation bar (bars): price must
#: retrace into the frozen FVG within this, or the setup expires.
ARMED_BARS = 12


@dataclass
class SweepRecord:
    """One armed turtle-soup sweep.  Levels frozen at confirmation,
    never recomputed.  JSON-safe (floats / ints / strs / None)."""

    sweep_idx: int
    direction: int          # -1 fades a high sweep (short), +1 fades a low
    phase: str              # "swept" | "triggered" | "armed"
    sweep_level: float      # the 20-bar extreme that was swept
    sweep_high: float       # sweep bar OHLC (frozen)
    sweep_low: float
    sweep_close: float
    sweep_open: float
    sweep_extreme: float    # sweep bar's wick extreme (high for short)
    opp_extreme: float      # sweep bar's opposite extreme (low for short)
    trigger_level: float    # = sweep_close: trigger closes beyond it
    trigger_idx: int | None = None
    confirm_idx: int | None = None
    fvg_top: float | None = None
    fvg_bottom: float | None = None
    fvg_formed_idx: int | None = None
    entry: float | None = None        # frozen FVG near edge
    stop: float | None = None         # frozen: sweep extreme +/- buffer
    invalidation: float | None = None  # frozen: FVG far edge
    target: float | None = None       # frozen: 1.5R
    atr: float | None = None          # frozen: confirmation-bar ATR
    expiry_bar: int = 0               # current phase deadline

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SweepRecord":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


class _Engine:
    """Streaming turtle-soup detector: sweep -> trigger -> confirm ->
    FVG-armed -> holding retracement.  One dataclass per sweep."""

    def __init__(self) -> None:
        self._tracker = FVGTracker(max_age_bars=100)
        self.pending: dict[int, SweepRecord] = {}

    def process(self, o, h, l, c, t: int, collect: bool) -> list[dict]:
        out: list[dict] = []
        # The tracker is fed EVERY bar, unconditionally (inversion
        # rule: skipping a bar would leave a gap in its index stream).
        for _z in self._tracker.on_bar(float(h[t]), float(l[t]), t):
            pass
        a = atr(h, l, c, t)
        if a <= 0:
            return out
        # --- advance pending sweeps (one phase step per bar each).
        for key in list(self.pending):
            rec = self.pending[key]
            d = rec.direction
            if rec.phase == "swept":
                if t - rec.sweep_idx > TRIGGER_BARS:
                    del self.pending[key]
                    continue
                if (d == -1 and c[t] > rec.sweep_extreme) or \
                        (d == 1 and c[t] < rec.sweep_extreme):
                    del self.pending[key]  # real break: void
                    continue
                if (d == -1 and c[t] < rec.trigger_level) or \
                        (d == 1 and c[t] > rec.trigger_level):
                    rec.phase = "triggered"
                    rec.trigger_idx = t
                    rec.expiry_bar = rec.sweep_idx + CONFIRM_BARS
            elif rec.phase == "triggered":
                if t - rec.sweep_idx > CONFIRM_BARS:
                    del self.pending[key]
                    continue
                if (d == -1 and c[t] > rec.sweep_extreme) or \
                        (d == 1 and c[t] < rec.sweep_extreme):
                    del self.pending[key]  # real break: void
                    continue
                hit = (d == -1 and l[t] < rec.opp_extreme) or \
                    (d == 1 and h[t] > rec.opp_extreme)
                if hit:
                    armed = self._arm(rec, a, t)
                    if armed is None:
                        del self.pending[key]  # no FVG: no setup
            elif rec.phase == "armed":
                if t - rec.confirm_idx > ARMED_BARS:
                    del self.pending[key]
                    continue
                if (d == -1 and c[t] > rec.sweep_extreme) or \
                        (d == 1 and c[t] < rec.sweep_extreme):
                    del self.pending[key]  # real break: void
                    continue
                if t <= rec.confirm_idx:
                    continue  # touch strictly after confirmation
                touched = float(l[t]) <= rec.fvg_top \
                    and float(h[t]) >= rec.fvg_bottom
                if not touched:
                    continue
                # Retest must HOLD: the touch bar's close back on the
                # fade side of the near edge (killzone_open's retest
                # rule).  A touch that closes inside/through the FVG
                # is not a rejection — stay armed, wait or expire.
                near = rec.fvg_bottom if d == -1 else rec.fvg_top
                if (d == -1 and c[t] > near) or \
                        (d == 1 and c[t] < near):
                    continue
                del self.pending[key]
                if collect:
                    out.append(self._setup(rec, t))
        # --- new sweep on bar t (evaluated from the NEXT bar on).
        if t >= LOOKBACK_BARS:
            hi20 = float(max(h[t - LOOKBACK_BARS:t]))
            lo20 = float(min(l[t - LOOKBACK_BARS:t]))
            hi_sweep = h[t] > hi20 + SWEEP_MARGIN_ATR * a \
                and c[t] < hi20
            lo_sweep = l[t] < lo20 - SWEEP_MARGIN_ATR * a \
                and c[t] > lo20
            if hi_sweep and lo_sweep:
                pass  # both-sweep bar: expansion bar, not a sweep
            elif hi_sweep:
                self.pending[t] = SweepRecord(
                    sweep_idx=t, direction=-1, phase="swept",
                    sweep_level=hi20, sweep_high=float(h[t]),
                    sweep_low=float(l[t]), sweep_close=float(c[t]),
                    sweep_open=float(o[t]), sweep_extreme=float(h[t]),
                    opp_extreme=float(l[t]), trigger_level=float(c[t]),
                    expiry_bar=t + TRIGGER_BARS)
            elif lo_sweep:
                self.pending[t] = SweepRecord(
                    sweep_idx=t, direction=1, phase="swept",
                    sweep_level=lo20, sweep_high=float(h[t]),
                    sweep_low=float(l[t]), sweep_close=float(c[t]),
                    sweep_open=float(o[t]), sweep_extreme=float(l[t]),
                    opp_extreme=float(h[t]), trigger_level=float(c[t]),
                    expiry_bar=t + TRIGGER_BARS)
        return out

    def _arm(self, rec: SweepRecord, a: float, t: int) -> bool | None:
        """Freeze the rejection-leg FVG + levels at confirmation.

        Returns True when armed, None when no FVG (setup dies)."""
        d = rec.direction
        cands = [z for z in self._tracker.zones
                 if z.direction == d
                 and z.formed_idx > rec.sweep_idx
                 and z.formed_idx <= t
                 and z.status != "FILLED"]
        if not cands:
            return None
        # Most recently formed FVG of the rejection leg (the one the
        # displacement left closest to current price).
        z = max(cands, key=lambda zz: zz.formed_idx)
        rec.fvg_top = float(z.top)
        rec.fvg_bottom = float(z.bottom)
        rec.fvg_formed_idx = int(z.formed_idx)
        rec.entry = float(z.bottom) if d == -1 else float(z.top)
        rec.stop = float(rec.sweep_extreme + (-d) * STOP_BUFFER_ATR * a)
        rec.invalidation = float(z.top) if d == -1 else float(z.bottom)
        risk = (rec.entry - rec.stop) * d
        if risk <= 0:
            return None
        rec.target = float(rec.entry + TARGET_R * risk * d)
        rec.atr = float(a)
        rec.phase = "armed"
        rec.confirm_idx = t
        rec.expiry_bar = t + ARMED_BARS
        return True

    @staticmethod
    def _setup(rec: SweepRecord, t: int) -> dict:
        return {
            "direction": rec.direction, "entry": float(rec.entry),
            "stop": float(rec.stop),
            "invalidation": float(rec.invalidation),
            "target": float(rec.target),
            "sweep_idx": rec.sweep_idx,
            "sweep_level": rec.sweep_level,
            "sweep_extreme": rec.sweep_extreme,
            "trigger_idx": rec.trigger_idx,
            "confirm_idx": rec.confirm_idx,
            "retest_idx": t, "atr": float(rec.atr),
            "fvg_top": rec.fvg_top, "fvg_bottom": rec.fvg_bottom,
            "fvg_formed_idx": rec.fvg_formed_idx,
        }

    def to_dict(self) -> dict:
        return {"pending": {str(k): v.to_dict()
                            for k, v in self.pending.items()},
                "tracker": self._tracker.to_dict()}

    @classmethod
    def from_dict(cls, state: dict | None) -> "_Engine":
        eng = cls()
        state = state or {}
        eng.pending = {int(k): SweepRecord.from_dict(v)
                       for k, v in state.get("pending", {}).items()}
        tracker_state = state.get("tracker")
        if tracker_state is not None:
            eng._tracker = FVGTracker.from_dict(tracker_state)
        return eng


def _to_signal(setup: dict, timestamp) -> GoldenSignal:
    d = setup["direction"]
    sig = GoldenSignal(
        index=setup["retest_idx"], timestamp=timestamp, direction=d,
        score=SLEEVE_BASE_SCORE, kind="turtle_soup",
        factors=["turtle_soup"],
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


class TurtleSoupPlaybook(Playbook):
    name = "turtle_soup"
    description = (
        "Failed breakout of a 20-bar extreme: sweep + trigger + "
        "trade-through of the sweep bar's opposite extreme, then a "
        "holding retracement into the rejection-leg FVG — enter at "
        "the FVG near edge"
    )
    live = True  # measured PF 1.07 standalone (the map, in the
    # docstring); promoted to live demo by owner directive 2026-10-22
    # for forward testing — the demo decides.
    live_threshold = 0.0  # every holding retracement is graded
    context_description = (
        "XAUUSD H1, any session. Entry at the rejection-leg FVG near "
        "edge on a holding retracement; stop beyond the sweep extreme "
        "+/- 0.25 ATR; target 1.5R; invalidation is a close through "
        "the FVG far edge; 48-bar max hold."
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
                    f"turtle_soup {side} (retest bar {t}): swept "
                    f"20-bar extreme {setup['sweep_level']:.1f} at bar "
                    f"{setup['sweep_idx']} (wick "
                    f"{setup['sweep_extreme']:.1f}), confirmed bar "
                    f"{setup['confirm_idx']}, FVG "
                    f"{setup['fvg_bottom']:.1f}-{setup['fvg_top']:.1f} "
                    f"(formed {setup['fvg_formed_idx']}), entry "
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
        need = ("sweep_idx", "trigger_idx", "confirm_idx", "retest_idx",
                "direction", "fvg_formed_idx", "entry", "stop",
                "invalidation")
        missing = [k for k in need if k not in c]
        if missing:
            return False, f"missing setup facts: {missing}"
        if not (c["sweep_idx"] < c["trigger_idx"] < c["confirm_idx"]
                < c["retest_idx"]):
            return False, "sweep < trigger < confirm < retest ordering"
        if c["trigger_idx"] - c["sweep_idx"] > TRIGGER_BARS:
            return False, "trigger outside the 5-bar window"
        if c["confirm_idx"] - c["sweep_idx"] > CONFIRM_BARS:
            return False, "confirmation outside the 10-bar window"
        if c["retest_idx"] - c["confirm_idx"] > ARMED_BARS:
            return False, "retest outside the 12-bar armed window"
        if not (c["sweep_idx"] < c["fvg_formed_idx"] <= c["confirm_idx"]):
            return False, "FVG must form after the sweep, by confirmation"
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
