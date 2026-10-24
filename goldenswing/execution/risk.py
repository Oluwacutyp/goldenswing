"""Risk layer — a pure guard layer.  It never places, changes or
closes a trade; it answers two questions and remembers the answers:

- ``size_position(...)`` -> how many lots may this trade take?
  (ATS CAtsSizer pattern: lots are FLOORED to the volume step, never
  rounded up — rounding up to the broker minimum is how "1% risk"
  systems quietly risk more.  If the allowed risk cannot buy the
  minimum lot, the answer is 0 lots plus a human-readable reason.)
- ``RiskGuards.check_new_trade(...)`` -> may a new trade open at all?
  (ATS day/drawdown/exposure guards, in Python: a max-drawdown halt,
  a consecutive-loss halt, a cap on open positions and on total
  money-at-stops exposure.)

Owner ruling 2026-10-23: maximum drawdown is THE account-level
hard halt.  The former daily and weekly loss caps are REMOVED —
they were halt constraints whose configured percentages produced
an arbitrary fixed-dollar "daily room" that moved with the
balance and vetoed analytically-qualified trades; they no longer
exist in the config, the checks, or the latch.  Day/week realised
P/L is still BOOKED (state + status facts), it just never halts
or refuses anything.

Guard state is PERSISTED as JSON on disk (the ATS library uses MT5
GlobalVariables for the same reason): a process or terminal restart
can never reset a halt that was already latched.  Day/week periods
roll over by calendar date (an injectable clock keeps the rollover
testable), and a tripped halt LATCHES — it stays tripped, with its
reason, until a human clears it via :meth:`RiskGuards.reset_halt`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path


# ------------------------------------------------------------ symbol spec
@dataclass(frozen=True)
class SymbolSpec:
    """The broker facts position sizing needs.  ``tick_value`` is the
    account-currency value of one tick (``tick_size`` of price) for a
    1.0-lot position — exactly MT5's ``trade_tick_value``."""

    name: str
    tick_value: float
    tick_size: float
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    digits: int = 2
    point: float = 0.01

    def risk_per_lot(self, stop_distance: float) -> float:
        """Money lost by 1.0 lot if price travels ``stop_distance``."""
        if self.tick_size <= 0 or self.tick_value <= 0:
            return 0.0
        return abs(stop_distance) / self.tick_size * self.tick_value

    def money_for_move(self, distance: float, volume: float) -> float:
        """Money value of a price move of ``distance`` at ``volume``."""
        if self.tick_size <= 0:
            return 0.0
        return abs(distance) / self.tick_size * self.tick_value * volume


def gold_spec() -> SymbolSpec:
    """XAUUSD as most MT5 brokers quote it: 1 lot = 100 oz, so a
    $0.01 tick is worth $1.00 per lot."""
    return SymbolSpec(name="XAUUSD", tick_value=1.0, tick_size=0.01,
                      volume_min=0.01, volume_max=50.0, volume_step=0.01,
                      digits=2, point=0.01)


# ----------------------------------------------------------------- sizing
@dataclass(frozen=True)
class SizingResult:
    lots: float
    risk_money: float          # actual money at risk at ``lots`` (<= budget)
    reason: str = ""           # non-empty exactly when lots == 0

    @property
    def ok(self) -> bool:
        return self.lots > 0


def size_position(*, balance: float, risk_pct: float, entry: float,
                  stop: float, spec: SymbolSpec) -> SizingResult:
    """Lots for a trade risking at most ``risk_pct`` of ``balance``.

    Floors to the volume step, NEVER rounds up; caps at volume_max.
    Returns 0 lots + a reason string whenever the trade cannot be
    sized honestly (bad inputs, or the risk budget cannot buy the
    broker minimum).
    """
    if balance <= 0:
        return SizingResult(0.0, 0.0, f"balance must be positive (got {balance})")
    if risk_pct <= 0 or risk_pct > 0.25:
        return SizingResult(0.0, 0.0, f"risk_pct out of sane range (got {risk_pct})")
    distance = abs(entry - stop)
    if distance <= 0:
        return SizingResult(0.0, 0.0, "stop distance is zero — no stop, no size")
    per_lot = spec.risk_per_lot(distance)
    if per_lot <= 0:
        return SizingResult(
            0.0, 0.0, f"symbol {spec.name} has unusable tick value/size "
                      f"({spec.tick_value}/{spec.tick_size})")
    budget = balance * risk_pct
    raw = budget / per_lot
    step = spec.volume_step if spec.volume_step > 0 else spec.volume_min
    # The 1e-9 epsilon only absorbs binary representation error
    # (0.29/0.01 == 28.999999999999996); it can never push a size UP
    # across a real step boundary.
    lots = math.floor(raw / step + 1e-9) * step
    lots = round(min(lots, spec.volume_max), 10)
    if lots < spec.volume_min:
        return SizingResult(
            0.0, 0.0,
            f"risk budget {budget:.2f} buys {raw:.4f} lots but the broker "
            f"minimum is {spec.volume_min} (stop {distance:.2f} = "
            f"{per_lot:.2f} per lot on {spec.name})")
    return SizingResult(lots, lots * per_lot, "")


# ----------------------------------------------------------------- guards
@dataclass
class RiskConfig:
    # Calibrated 2026-10-11 from the anchor book's loss clusters
    # (scripts/risk_calibration.py): guards protect the TAIL, not the
    # normal.  A halt that fires in normal operation is a bug.
    #
    # Owner ruling 2026-10-23: max_drawdown_pct is the SOLE
    # account-level hard halt.  The daily/weekly loss caps that
    # used to live here (daily_loss_pct 0.03 / weekly_loss_pct
    # 0.05, with optional money overrides) are REMOVED — not
    # zeroed, gone: no config field, no room check, no latch.
    max_drawdown_pct: float = 0.15     # from peak balance -> latched halt
                                       # (1.5x in-sample max 10.38%)
    max_consecutive_losses: int = 6    # -> latched halt (3-runs hit 14x
                                       # in 2y; 6+ hit once — tail only)
    max_positions: int = 3
    max_total_risk_pct: float = 0.03   # money-at-stops cap, of balance
    require_stops: bool = True         # an unstopped position blocks new risk


@dataclass
class GuardDecision:
    allowed: bool
    reason: str = ""


def _day_key(d: date) -> str:
    return d.isoformat()


def _week_key(d: date) -> str:
    iso = d.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


class RiskGuards:
    """Persisted, latching risk guards.  One instance owns one state
    file; constructing a second instance on the same file sees the
    first instance's state (that is the restart-proofing)."""

    def __init__(self, state_path: str | Path,
                 config: RiskConfig | None = None,
                 kill_switch_path: str | Path | None = None,
                 today_fn=None):
        self.state_path = Path(state_path)
        self.cfg = config or RiskConfig()
        self.kill_switch_path = (Path(kill_switch_path)
                                 if kill_switch_path else None)
        self._today_fn = today_fn or (lambda: datetime.now(timezone.utc).date())
        self.load_error: str | None = None
        self._state = self._load()
        self._roll_periods()

    # ---------------------------------------------------------- state io
    def _fresh_state(self) -> dict:
        today = self._today_fn()
        return {
            "day": _day_key(today), "day_start_balance": None,
            "day_realized": 0.0,
            "week": _week_key(today), "week_start_balance": None,
            "week_realized": 0.0,
            "peak_balance": None,
            "consecutive_losses": 0,
            "halted": False, "halt_reason": "",
            "trades_today": 0,
        }

    def _load(self) -> dict:
        if not self.state_path.exists():
            return self._fresh_state()
        try:
            raw = json.loads(self.state_path.read_text())
            state = self._fresh_state()
            state.update({k: raw[k] for k in state if k in raw})
            return state
        except Exception as exc:  # corrupt state must not silently pass
            self.load_error = f"risk state unreadable ({exc}); started fresh"
            return self._fresh_state()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state, indent=1))
        tmp.replace(self.state_path)

    def _roll_periods(self) -> None:
        today = self._today_fn()
        changed = False
        if self._state["day"] != _day_key(today):
            self._state.update(day=_day_key(today), day_start_balance=None,
                               day_realized=0.0, trades_today=0)
            changed = True
        if self._state["week"] != _week_key(today):
            self._state.update(week=_week_key(today),
                               week_start_balance=None, week_realized=0.0)
            changed = True
        if changed:
            self._save()

    # ------------------------------------------------------------- views
    @property
    def state(self) -> dict:
        return dict(self._state)

    @property
    def halted(self) -> bool:
        return bool(self._state["halted"])

    def kill_switch_active(self) -> bool:
        return self.kill_switch_path is not None \
            and self.kill_switch_path.exists()

    # ------------------------------------------------------------ checks
    def note_balance(self, balance: float) -> None:
        """Feed the current balance: anchors day/week starts and the
        drawdown peak.  Called on every check and outcome."""
        self._roll_periods()
        changed = False
        if self._state["day_start_balance"] is None and balance > 0:
            self._state["day_start_balance"] = balance
            changed = True
        if self._state["week_start_balance"] is None and balance > 0:
            self._state["week_start_balance"] = balance
            changed = True
        peak = self._state["peak_balance"]
        if balance > 0 and (peak is None or balance > peak):
            self._state["peak_balance"] = balance
            changed = True
        if changed:
            self._save()

    def check_new_trade(self, *, balance: float, open_risk_money: float,
                        new_risk_money: float, open_count: int,
                        unstopped_positions: int = 0) -> GuardDecision:
        """May one more trade, risking ``new_risk_money``, open now?"""
        if self.kill_switch_active():
            return GuardDecision(
                False, f"kill switch active ({self.kill_switch_path})")
        if balance <= 0:
            return GuardDecision(False, f"balance must be positive (got {balance})")
        self.note_balance(balance)
        st = self._state
        if st["halted"]:
            return GuardDecision(False, f"risk halt latched: {st['halt_reason']}")
        if unstopped_positions > 0 and self.cfg.require_stops:
            return GuardDecision(
                False, f"{unstopped_positions} open position(s) have NO stop — "
                       "exposure is unknowable, new risk refused")
        if open_count >= self.cfg.max_positions:
            return GuardDecision(
                False, f"max positions reached ({open_count}/{self.cfg.max_positions})")
        # max_total_risk_pct == 0 disables the exposure cap, matching
        # the other 0-disables in this file (drawdown /
        # consecutive-loss). The Exness $12 loop sets it to 0 because
        # margin + equity stop-out are its real constraint, not R.
        if self.cfg.max_total_risk_pct and \
                open_risk_money + new_risk_money > \
                self.cfg.max_total_risk_pct * balance:
            cap = self.cfg.max_total_risk_pct * balance
            return GuardDecision(
                False, f"exposure cap: open risk {open_risk_money:.2f} + new "
                       f"{new_risk_money:.2f} > {cap:.2f} "
                       f"({self.cfg.max_total_risk_pct:.0%} of balance)")
        # Owner ruling 2026-10-23: NO daily/weekly loss-room check
        # exists here anymore — max drawdown (latched in
        # record_outcome) is the sole account-level hard halt.
        return GuardDecision(True, "")

    # ---------------------------------------------------------- outcomes
    def record_outcome(self, pnl_money: float, balance: float) -> None:
        """Book one closed trade's realised P/L.  Trips (and latches)
        the halt when max drawdown or the consecutive-loss limit is
        breached."""
        self.note_balance(balance)
        st = self._state
        st["day_realized"] += pnl_money
        st["week_realized"] += pnl_money
        st["trades_today"] += 1
        if pnl_money < 0:
            st["consecutive_losses"] += 1
        elif pnl_money > 0:
            st["consecutive_losses"] = 0
        reasons = []
        # Owner ruling 2026-10-23: daily/weekly loss limits no
        # longer latch a halt — day/week realised P/L above is
        # bookkeeping only.  Max drawdown is the sole account-level
        # hard halt; the consecutive-loss halt (a streak fact, not
        # a loss-room allowance) is unchanged.
        peak = st["peak_balance"]
        if peak and self.cfg.max_drawdown_pct:
            dd = (peak - balance) / peak
            if dd >= self.cfg.max_drawdown_pct:
                reasons.append(f"max drawdown hit ({dd:.1%} from peak "
                               f"{peak:.2f} >= {self.cfg.max_drawdown_pct:.0%})")
        if self.cfg.max_consecutive_losses and \
                st["consecutive_losses"] >= self.cfg.max_consecutive_losses:
            reasons.append(f"{st['consecutive_losses']} consecutive losses "
                           f"(limit {self.cfg.max_consecutive_losses})")
        if reasons and not st["halted"]:
            st["halted"] = True
            st["halt_reason"] = "; ".join(reasons)
        self._save()

    def reset_halt(self) -> None:
        """Human action: clear a latched halt (state stays on disk)."""
        self._state["halted"] = False
        self._state["halt_reason"] = ""
        self._save()
