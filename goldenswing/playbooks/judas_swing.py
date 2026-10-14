"""JudasSwingPlaybook — London-open Judas swing: Asia-edge sweep that
fails, then the real move (ICT).

Definition (precise, causal, no repainting):

Bearish Judas (bullish is the exact mirror):
  1. ASIA RANGE — the high/low of the Asia session block, FROZEN at
     the first London bar (session_of from
     reader/session_timeframes.py — the DST-aware session clock, no
     invented clock).  The Asia block is the level the Judas hunts.
  2. SWEEP (MANIPULATION) — during the London session, a bar wicks
     beyond the Asia high by >= 0.10 ATR but CLOSES back inside
     (close < Asia high): the London open fakes a breakout and fails.
     A bar that sweeps BOTH Asia edges is an expansion bar, not a
     Judas.  A close beyond the sweep extreme (wick) voids.
  3. TRIGGER — within 5 bars after the sweep, a bar closes inside
     AND below the sweep bar's close (the fade begins).
  4. CONFIRMATION (DISPLACEMENT) — within 10 bars after the sweep,
     price trades through the sweep bar's OPPOSITE extreme (low <
     sweep-bar low): the real move begins.  A close beyond the
     sweep extreme at any point voids the setup (the break was real).
  5. ENTRY — armed for 12 bars after confirmation: the first bar
     whose range touches the swept Asia level (high >= Asia high)
     AND whose close HOLDS the fade side (close <= Asia high — the
     killzone_open retest rule: a retracement that closes through
     the level is not a rejection) enters at the Asia level on that
     bar's close.  A touch that closes through stays armed (waits
     for a later holding retest or expiry).  Levels are frozen at
     confirmation (entry / stop / invalidation / target / ATR),
     never recomputed.
  6. STOP = sweep extreme (wick) +/- 0.25 ATR buffer.
     INVALIDATION = a close back beyond the sweep extreme.
     Target = 1.5R fixed (shared exit discipline, playbooks/common.py).

One trade per sweep.  State is one dataclass per sweep, keyed by
sweep bar index, serializable for warm resume.

Why this is not killzone_open: killzone_open trades a REAL break of
the pre-range (close beyond with displacement) + retest.  Judas
trades the FAILED break: the sweep closes back inside, displaces the
other way, and retests the swept level from the other side.  Same
Asia level, opposite premise — the two setups never fire on the same
break.

Why this is not turtle_soup: same fade mechanics, different level
source.  Turtle Soup sweeps a 20-bar extreme in any session and
enters at the rejection-leg FVG; Judas sweeps the Asia edge in the
London session and enters at the swept Asia level itself.  The Asia
edge is the session liquidity pool; the FVG is a 3-bar imbalance —
different levels, different clocks.

Exits: shared discipline (common.py) — stop = invalidation +/- 0.25
ATR, target 1.5R, invalidation exit at close, 48-bar max hold.

VALIDATION (scripts/validate_new_playbooks.py, OANDA XAUUSD H1
2024-10 -> 2026-10, 11987 bars): 98 trades, 65% win, total R +46.2,
PF 1.93, avg +0.47R, maxDD -6.5R, positive in BOTH halves (+20.6 /
+25.6).  PASSES the iron rule cleanly — the holding-retest filter
is load-bearing: without it (any touch enters) the same structure
is breakeven (PF 0.99).  Executes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

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
#: A sweep must wick beyond the Asia edge by this x ATR (dust wicks
#: are not manipulation — same margin as live/manipulation.py).
SWEEP_MARGIN_ATR = 0.10
#: Trigger window after the sweep bar (bars).
TRIGGER_BARS = 5
#: Confirmation window after the sweep bar (bars).
CONFIRM_BARS = 10
#: Armed window after the confirmation bar (bars): price must
#: retest the swept Asia level within this, or the setup expires.
ARMED_BARS = 12


@dataclass
class SweepRecord:
    """One armed Judas sweep.  Levels frozen at confirmation, never
    recomputed.  JSON-safe (floats / ints / strs / None)."""

    sweep_idx: int
    direction: int          # -1 fades an Asia-high sweep (short)
    phase: str              # "swept" | "triggered" | "armed"
    sweep_level: float      # the Asia edge that was swept
    sweep_high: float       # sweep bar OHLC (frozen)
    sweep_low: float
    sweep_close: float
    sweep_open: float
    sweep_extreme: float    # sweep bar's wick extreme (high for short)
    opp_extreme: float      # sweep bar's opposite extreme (low for short)
    trigger_level: float    # = sweep_close: trigger closes beyond it
    sweep_session: str      # session of the sweep bar ("london")
    trigger_idx: int | None = None
    confirm_idx: int | None = None
    entry: float | None = None        # frozen: swept Asia level
    stop: float | None = None         # frozen: sweep extreme +/- buffer
    invalidation: float | None = None  # frozen: sweep extreme
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
    """Streaming Judas detector: Asia range -> London sweep -> trigger
    -> displacement -> holding retest of the swept Asia level."""

    def __init__(self) -> None:
        self.pending: dict[int, SweepRecord] = {}
        self.asia_high: float | None = None
        self.asia_low: float | None = None
        self.asia_session: str | None = None
        self.frozen_asia: tuple[float, float] | None = None

    def process(self, o, h, l, c, t: int, collect: bool,
                ts=None) -> list[dict]:
        out: list[dict] = []
        if ts is None:
            return out
        sess = session_of(ts[t])[0]
        # --- Asia-range bookkeeping (runs on EVERY bar, ATR or not —
        # killzone_open rule: the session block must not depend on
        # whether ATR is warm yet).
        if sess == "asia":
            if self.asia_session != "asia":
                self.asia_high = float(h[t])
                self.asia_low = float(l[t])
                self.asia_session = "asia"
            else:
                self.asia_high = max(self.asia_high, float(h[t]))
                self.asia_low = min(self.asia_low, float(l[t]))
        elif sess == "london" and self.asia_session == "asia":
            # First London bar: freeze the Asia range.
            self.frozen_asia = (self.asia_high, self.asia_low)
            self.asia_session = "london"
        elif sess not in ("asia", "london") \
                and self.asia_session == "london":
            self.asia_session = sess  # London over; frozen range kept
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
                    self._arm(rec, a, t)
            elif rec.phase == "armed":
                if t - rec.confirm_idx > ARMED_BARS:
                    del self.pending[key]
                    continue
                if (d == -1 and c[t] > rec.sweep_extreme) or \
                        (d == 1 and c[t] < rec.sweep_extreme):
                    del self.pending[key]  # real break: void
                    continue
                if t <= rec.confirm_idx:
                    continue  # retest strictly after confirmation
                touched = (d == -1 and h[t] >= rec.sweep_level) or \
                    (d == 1 and l[t] <= rec.sweep_level)
                if not touched:
                    continue
                # Retest must HOLD: the touch bar's close on the fade
                # side of the swept level (killzone_open's retest
                # rule).  A touch that closes through the level is
                # not a rejection — stay armed, wait or expire.
                if (d == -1 and c[t] > rec.sweep_level) or \
                        (d == 1 and c[t] < rec.sweep_level):
                    continue
                del self.pending[key]
                if collect:
                    out.append(self._setup(rec, t))
        # --- new sweep on bar t: London session only, of the frozen
        # Asia edge (evaluated from the NEXT bar on).
        if sess == "london" and self.frozen_asia is not None:
            ah, al = self.frozen_asia
            hi_sweep = h[t] > ah + SWEEP_MARGIN_ATR * a and c[t] < ah
            lo_sweep = l[t] < al - SWEEP_MARGIN_ATR * a and c[t] > al
            if hi_sweep and lo_sweep:
                pass  # both-sweep bar: expansion bar, not a Judas
            elif hi_sweep:
                self.pending[t] = SweepRecord(
                    sweep_idx=t, direction=-1, phase="swept",
                    sweep_level=float(ah), sweep_high=float(h[t]),
                    sweep_low=float(l[t]), sweep_close=float(c[t]),
                    sweep_open=float(o[t]), sweep_extreme=float(h[t]),
                    opp_extreme=float(l[t]), trigger_level=float(c[t]),
                    sweep_session=sess, expiry_bar=t + TRIGGER_BARS)
            elif lo_sweep:
                self.pending[t] = SweepRecord(
                    sweep_idx=t, direction=1, phase="swept",
                    sweep_level=float(al), sweep_high=float(h[t]),
                    sweep_low=float(l[t]), sweep_close=float(c[t]),
                    sweep_open=float(o[t]), sweep_extreme=float(l[t]),
                    opp_extreme=float(h[t]), trigger_level=float(c[t]),
                    sweep_session=sess, expiry_bar=t + TRIGGER_BARS)
        return out

    @staticmethod
    def _arm(rec: SweepRecord, a: float, t: int) -> None:
        """Freeze entry / stop / invalidation / target at confirmation."""
        d = rec.direction
        rec.entry = float(rec.sweep_level)
        rec.stop = float(rec.sweep_extreme + (-d) * STOP_BUFFER_ATR * a)
        rec.invalidation = float(rec.sweep_extreme)
        risk = (rec.entry - rec.stop) * d
        rec.target = float(rec.entry + TARGET_R * risk * d)
        rec.atr = float(a)
        rec.phase = "armed"
        rec.confirm_idx = t
        rec.expiry_bar = t + ARMED_BARS

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
            "sweep_session": rec.sweep_session,
            "trigger_idx": rec.trigger_idx,
            "confirm_idx": rec.confirm_idx,
            "retest_idx": t, "atr": float(rec.atr),
        }

    def to_dict(self) -> dict:
        return {"pending": {str(k): v.to_dict()
                            for k, v in self.pending.items()},
                "asia_high": self.asia_high,
                "asia_low": self.asia_low,
                "asia_session": self.asia_session,
                "frozen_asia": self.frozen_asia}

    @classmethod
    def from_dict(cls, state: dict | None) -> "_Engine":
        eng = cls()
        state = state or {}
        eng.pending = {int(k): SweepRecord.from_dict(v)
                       for k, v in state.get("pending", {}).items()}
        eng.asia_high = state.get("asia_high")
        eng.asia_low = state.get("asia_low")
        eng.asia_session = state.get("asia_session")
        frozen = state.get("frozen_asia")
        eng.frozen_asia = tuple(frozen) if frozen is not None else None
        return eng


def _to_signal(setup: dict, timestamp) -> GoldenSignal:
    d = setup["direction"]
    sig = GoldenSignal(
        index=setup["retest_idx"], timestamp=timestamp, direction=d,
        score=SLEEVE_BASE_SCORE, kind="judas_swing",
        factors=["judas_swing"],
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


class JudasSwingPlaybook(Playbook):
    name = "judas_swing"
    description = (
        "London-open Judas swing: Asia-edge sweep that closes back "
        "inside + trigger + displacement through the sweep bar's "
        "opposite extreme, then a holding retest of the swept Asia "
        "level — enter at the Asia level"
    )
    live = True  # validated: PF 1.93, +46.2R, both halves positive
    live_threshold = 0.0  # every holding retest is graded
    context_description = (
        "XAUUSD H1, sweep in the London session only (retest may run "
        "into later sessions). Entry at the swept Asia level on a "
        "holding retest; stop beyond the sweep extreme +/- 0.25 ATR; "
        "target 1.5R; invalidation is a close back beyond the sweep "
        "extreme; 48-bar max hold."
    )
    # The gate's single-session field cannot express "sweep in
    # London, retest anywhere": "any" passes the gate, and
    # check_live_signal enforces the London sweep from the signal's
    # own sweep_session fact (killzone_open pattern).
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
                    f"judas_swing {side} (retest bar {t}): swept Asia "
                    f"edge {setup['sweep_level']:.1f} at bar "
                    f"{setup['sweep_idx']} (wick "
                    f"{setup['sweep_extreme']:.1f}, "
                    f"{setup['sweep_session']} session), confirmed bar "
                    f"{setup['confirm_idx']}, holding retest entry "
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
                "direction", "sweep_session", "sweep_level",
                "sweep_extreme", "entry", "stop", "invalidation")
        missing = [k for k in need if k not in c]
        if missing:
            return False, f"missing setup facts: {missing}"
        if c["sweep_session"] != "london":
            return False, (
                f"judas_swing sweeps in the London session only, got "
                f"{c['sweep_session']!r}")
        if not (c["sweep_idx"] < c["trigger_idx"] < c["confirm_idx"]
                < c["retest_idx"]):
            return False, "sweep < trigger < confirm < retest ordering"
        if c["trigger_idx"] - c["sweep_idx"] > TRIGGER_BARS:
            return False, "trigger outside the 5-bar window"
        if c["confirm_idx"] - c["sweep_idx"] > CONFIRM_BARS:
            return False, "confirmation outside the 10-bar window"
        if c["retest_idx"] - c["confirm_idx"] > ARMED_BARS:
            return False, "retest outside the 12-bar armed window"
        if c["entry"] != c["sweep_level"]:
            return False, "entry must be the swept Asia level"
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
