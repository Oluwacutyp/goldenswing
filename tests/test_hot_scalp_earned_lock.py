"""hot_scalp EARNED profit-lock — ticket 540791314 (2026-11-05).

The second same-morning instance, four hours after 540274935
was fixed (9904040: progress-arm exclusion + entry-zone cap):

- FILL hot_scalp +1 0.01 @ 4124.49, SL 4110.43 (risk 14.06).
- 14:32 UTC alert: "PROFIT LOCK SL -> 4131.52" — entry +
  0.50 x risk EXACTLY (4124.49 + 7.03).  This lock was armed
  by the surviving best-R >= 1.0 arm, which the first fix
  deliberately left in place.
- 14:36 UTC close at 4131.37 -> +0.49R.

The pattern across both tickets: the entry+0.5R lock level
becomes the de-facto exit — it converts runner-shaped trades
into half-R scratches whenever a normal pullback follows the
first push.  The level formula is not the disease; WHEN the
lock may fire is.

Fix pinned here (mirrors the codebase's own P3 earned-
breakeven doctrine — the touch alone does not move the stop;
an earning event or the backstop does):

- hot_scalp's profit-lock is EARNED (TickConfig
  .profit_lock_earned_playbooks).  Arming events, either:
  (a) ACCEPTANCE — an M5 close beyond the position's first
      target in its favour, stamped by the runner's bar-close
      pass via mark_pl_earned (acceptance-by-closes, the
      reaction doctrine — a wick touch earns nothing); or
  (b) the BACKSTOP excursion best_r >= profit_lock_backstop_r
      (2.0R), recorded on the state the tick it first holds.
- The lock LEVEL formula, the zone cap and the progress
  exclusion are untouched; every other playbook keeps the
  touch arms byte-identically (control pins below).

Fixture arithmetic mirrors the ticket: ATR 4.0 -> lock
distance max(0.5 x 14.06, 0.75 x 4.0) = 7.03 -> the lock
level is 4131.52, the alert's exact number.  TP1 is set at
4140.00 (+1.10R): the +1.2R touch (4141.36) WICKS through it
— under the old arm that touch locked the trade; under the
earned rule a wick earns nothing until an M5 bar CLOSES
beyond it.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.backtest.harness import ExitConfig
from goldenswing.execution.executor import OrderRequest, OrderResult
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.execution.tick_manager import (
    TickConfig, TickManager)
from goldenswing.live.hierarchy import TimeframeStack
from goldenswing.live.management_profiles import profile_for

TICKET = 540791314
ENTRY = 4124.49
STOP0 = 4110.43             # broker stop; risk 14.06
RAW_INV = 4111.43           # STOP0 + 0.25 * ATR(4.0)
ATR = 4.0
TP1 = 4140.00               # targets[0], +1.10R
LOCK = 4131.52              # entry + 0.5 * 14.06 — the alert's level
TOUCH_1R = 4138.55          # best_r 1.00 — the old arm's trigger
TOUCH_12R = 4141.36         # best_r 1.20 — wicks through TP1
ACCEPT_CLOSE = 4140.80      # an M5 CLOSE beyond TP1
BACKSTOP_TICK = 4152.70     # best_r 2.007


class FakeSender:
    def __init__(self):
        self.texts = []

    def send_text(self, text, silent=False, reply_markup=None):
        self.texts.append(text)
        return SimpleNamespace(delivered=True, detail="ok")

    def send_photo(self, png, caption):
        self.texts.append(caption)
        return SimpleNamespace(delivered=True, detail="ok")


class FakeExecutor:
    def __init__(self):
        self.modify_calls = []

    def symbol_spec(self, symbol):
        return gold_spec()

    def modify_sl(self, ticket, sl, tp=None):
        self.modify_calls.append((ticket, float(sl)))
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        return OrderResult(True, ticket=ticket)

    def positions(self, symbol=None):
        return []


def _runner(tmp_path, executor, sender=None):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg, executor=executor)
    if sender is not None:
        runner.senders = [sender]
    return runner


def _open(tm, *, playbook="hot_scalp", ticket=TICKET, targets=(TP1,)):
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=RAW_INV, entry_ref=ENTRY,
                       targets=list(targets), atr=ATR,
                       alert_id="MT5-hot_scalp-2008", bar_index=10,
                       magic=1)
    return tm.open(ticket=ticket, request=req, fill_price=ENTRY,
                   entry_bar=10, broker_sl=STOP0, playbook=playbook)


def _locks(actions):
    return [a for a in actions if a.kind == "modify_sl"
            and a.reason == "profit_lock"]


def _notes(tmp_path):
    path = tmp_path / "mt5_journal.jsonl"
    return path.read_text() if path.exists() else ""


# ------------------------------------------- the ticket's shape
def _tm(**tick_kw):
    """TickManager in the LIVE posture: the runner opts into
    earned breakeven (P3), so a +1R touch moves nothing there
    either — the fixture isolates the profit-lock arming."""
    return TickManager(tick=TickConfig(breakeven_mode="earned",
                                       **tick_kw))


def test_touch_to_1r_and_beyond_arms_nothing():
    """The live trade's path: a touch to +1.0R, then a wick to
    +1.2R THROUGH TP1 — no acceptance close, no 2R backstop —
    produces no lock, no stop move, no earning on record.
    (A pool partial may bank at the TP1 cross; that is the
    desk's banking rule, not this fix's subject — the pin is
    that no modify_sl of any kind fires.)"""
    tm = _tm()
    pos = _open(tm)
    assert pos.risk_price == pytest.approx(14.06)
    for price in (TOUCH_1R, TOUCH_12R, 4131.60):
        acts = tm.on_tick(TICKET, price)
        assert _locks(acts) == []
        assert [a for a in acts if a.kind == "modify_sl"] == []
    # The pullback that closed the live trade at the lock met
    # only the placement stop, far below (the 4131.60 tick).
    assert tm.effective_stop(TICKET) == pytest.approx(STOP0)
    state = tm.pl_state(TICKET)
    assert state["earned"] is False
    assert state["done"] is False
    assert tm.last_sl_reason(TICKET) == ""


def test_acceptance_stamp_arms_lock_at_formula_level():
    """Once acceptance is stamped (an M5 close beyond TP1 —
    the runner's feed, exercised end-to-end below), the next
    tick fires the lock at the unchanged formula level."""
    tm = _tm()
    _open(tm)
    assert _locks(tm.on_tick(TICKET, TOUCH_12R)) == []
    via = ("acceptance: M5 close 4140.80 beyond TP1 4140.00 "
           "(bar 2026-11-05 14:30:00+00:00)")
    assert tm.mark_pl_earned(TICKET, via) is True
    locks = _locks(tm.on_tick(TICKET, ACCEPT_CLOSE))
    assert len(locks) == 1
    assert locks[0].level == pytest.approx(LOCK)
    tm.apply(TICKET, locks[0])
    state = tm.pl_state(TICKET)
    assert state["earned"] is True
    assert state["by"] == via
    assert state["done"] is True
    assert tm.last_sl_reason(TICKET) == "profit_lock"


def test_backstop_excursion_arms_lock_and_names_itself():
    """No acceptance: a 2R excursion earns the lock on the
    backstop.  Trail/Chandelier are configured out of this
    fixture so the profit-lock candidate is the modify that
    fires (with default configs they can outrank the +0.5R
    level at +2R — the earning record below is the pin that
    holds either way)."""
    tm = TickManager(exits=ExitConfig(trail_trigger_r=99.0),
                     tick=TickConfig(chandelier_enabled=False,
                                     breakeven_mode="earned"))
    _open(tm)
    assert _locks(tm.on_tick(TICKET, TOUCH_12R)) == []
    locks = _locks(tm.on_tick(TICKET, BACKSTOP_TICK))
    assert len(locks) == 1
    assert locks[0].level == pytest.approx(LOCK)
    by = tm.pl_state(TICKET)["by"]
    assert by.startswith("backstop (best_r 2.0")
    tm.apply(TICKET, locks[0])
    assert tm.pl_state(TICKET)["done"] is True


def test_backstop_earning_recorded_under_default_config():
    """Default configs at +2R: whatever candidate wins the
    tick, the backstop earning is on record by name."""
    tm = TickManager()
    _open(tm)
    acts = tm.on_tick(TICKET, BACKSTOP_TICK)
    assert tm.pl_state(TICKET)["earned"] is True
    assert tm.pl_state(TICKET)["by"].startswith("backstop (best_r")
    assert any(a.kind == "modify_sl" for a in acts)


def test_mark_pl_earned_unknown_and_done_tickets():
    tm = TickManager()
    assert tm.mark_pl_earned(999, "acceptance: x") is False
    assert tm.pl_state(999) == {}
    _open(tm)
    assert tm.mark_pl_earned(TICKET, "acceptance: first") is True
    # Idempotent: a second stamp never overwrites the first.
    assert tm.mark_pl_earned(TICKET, "acceptance: second") is True
    assert tm.pl_state(TICKET)["by"] == "acceptance: first"


# ------------------------------------------------------ controls
def test_control_playbook_touch_arm_unchanged():
    """A playbook outside the earned set keeps the touch arms
    byte-identically: the plain +1.0R touch locks at the same
    formula level, and no earning is recorded for it."""
    tm = TickManager()
    _open(tm, playbook="fvg_limit")
    locks = _locks(tm.on_tick(TICKET, TOUCH_1R))
    assert len(locks) == 1
    assert locks[0].level == pytest.approx(LOCK)
    assert tm.pl_state(TICKET)["earned"] is False


def test_earned_scope_is_exactly_hot_scalp():
    cfg = TickConfig()
    assert cfg.profit_lock_earned_playbooks == ("hot_scalp",)
    assert cfg.profit_lock_backstop_r == pytest.approx(2.0)
    assert "hot_scalp" in cfg.profit_lock_progress_excluded_playbooks
    assert cfg.fixed_bracket_playbooks == ("qm_m5",)


# --------------------------------------- runner: acceptance feed
def _frame(closes, start, freq):
    idx = pd.date_range(start, periods=len(closes), freq=freq,
                        tz="UTC")
    return pd.DataFrame(
        {"open": list(closes), "high": [c + 0.4 for c in closes],
         "low": [c - 0.4 for c in closes], "close": list(closes),
         "volume": [100.0] * len(closes),
         "real_volume": [0.0] * len(closes),
         "spread": [30] * len(closes)}, index=idx)


def _stack(m5_last_close):
    base = [4120.0 + 0.05 * i for i in range(20)]
    m5_closes = base[:-1] + [m5_last_close]
    frames = {
        "D1": _frame(base[:10], "2026-10-27 00:00", "D"),
        "H4": _frame(base[:10], "2026-11-04 00:00", "4h"),
        "H1": _frame(base, "2026-11-04 12:00", "h"),
        "M15": _frame(base, "2026-11-05 09:00", "15min"),
        "M5": _frame(m5_closes, "2026-11-05 12:55", "5min"),
    }
    return TimeframeStack.from_frames(frames)


def _ctx(runner, *, playbook="hot_scalp"):
    m5_now = len(runner._stack.frames["M5"]) - 1
    return {
        "ticket": TICKET, "playbook": playbook,
        "profile": profile_for(style="scalp", clock_tf="M5"),
        "progress_trimmed": False, "progress_trim_bar": None,
        "drift_trimmed": False, "preprint_key": None,
        "zone_banked": False, "adr_banked": False,
        "defended_stop": None, "entry_total": None,
        "opposing_prox": None, "style_key": "scalp",
        "invalidation": STOP0, "floor": 50.0, "entry_bar": 10,
        "clock_tf": "M5", "management_clock": "M5",
        "entry_mgmt_bar": m5_now, "last_mgmt_bar": m5_now - 1,
    }


def _runner_with_position(tmp_path, m5_last_close, *,
                          playbook="hot_scalp"):
    ex = FakeExecutor()
    sender = FakeSender()
    runner = _runner(tmp_path, ex, sender)
    runner._stack = _stack(m5_last_close)
    _open(runner.tick_manager, playbook=playbook)
    runner._position_ctx = {TICKET: _ctx(runner, playbook=playbook)}
    return runner, ex, sender


def test_runner_acceptance_close_earns_then_lock_journals(tmp_path):
    """End to end: an M5 bar closes beyond TP1 -> the bar-close
    pass stamps the earning (journalled once, with the close
    price and bar time) -> the next tick fires the lock, and
    the lock's own journal record names the earning event."""
    runner, ex, sender = _runner_with_position(tmp_path, ACCEPT_CLOSE)
    runner._manage_positions_bar_close({}, clocks=("M5",))
    state = runner.tick_manager.pl_state(TICKET)
    assert state["earned"] is True
    assert state["by"].startswith(
        "acceptance: M5 close 4140.80 beyond TP1 4140.00 (bar ")
    notes = _notes(tmp_path)
    assert notes.count("profit lock EARNED for ticket "
                       f"{TICKET}") == 1
    assert '"pl_earned: acceptance:' in notes
    # A later close beyond TP1 never re-journals the earning.
    m5 = runner._stack.frames["M5"]
    extra = m5.iloc[[-1]].copy()
    extra.index = [m5.index[-1] + pd.Timedelta(minutes=5)]
    extra["close"] = [4141.20]
    runner._stack.frames["M5"] = pd.concat([m5, extra])
    runner._manage_positions_bar_close({}, clocks=("M5",))
    assert _notes(tmp_path).count(
        "profit lock EARNED for ticket "
        f"{TICKET}") == 1
    # The lock fires on the next tick, journalled with its
    # earning, and the owner alert goes out at the formula level.
    runner._manage_tick(ACCEPT_CLOSE, {"tick_actions": 0},
                        bid=ACCEPT_CLOSE, ask=ACCEPT_CLOSE + 0.2)
    assert len(ex.modify_calls) == 1
    assert ex.modify_calls[0][1] == pytest.approx(LOCK)
    notes = _notes(tmp_path)
    # (The journal JSON-escapes the note's em dash; assert the
    # ASCII spine of the line and the earning text separately.)
    assert "SL -> 4131.52 (profit_lock)" in notes
    assert "earned: acceptance: M5 close 4140.80 beyond " \
           "TP1 4140.00" in notes
    assert any("MT5 PROFIT LOCK hot_scalp ticket "
               f"{TICKET}: SL -> 4131.52" in t for t in sender.texts)


def test_runner_wick_without_close_earns_nothing(tmp_path):
    """The M5 bar's HIGH crossed TP1 intrabar but the bar
    CLOSED below it: no acceptance, no earning, no journal."""
    runner, ex, sender = _runner_with_position(tmp_path, 4139.90)
    runner._manage_positions_bar_close({}, clocks=("M5",))
    assert runner.tick_manager.pl_state(TICKET)["earned"] is False
    assert "profit lock EARNED" not in _notes(tmp_path)


def test_runner_acceptance_feed_scoped_to_hot_scalp(tmp_path):
    """A control playbook in the same seat: an M5 close beyond
    its TP1 stamps nothing — the feed is earned-set only."""
    runner, ex, sender = _runner_with_position(
        tmp_path, ACCEPT_CLOSE, playbook="fvg_limit")
    runner._manage_positions_bar_close({}, clocks=("M5",))
    assert runner.tick_manager.pl_state(TICKET)["earned"] is False
    assert "profit lock EARNED" not in _notes(tmp_path)


# ------------------------------------- the collapse (owner's chart)
def test_collapse_after_unproven_push_rides_to_structural_stop(
        tmp_path):
    """540791314's FULL shape, per the owner's M15 chart: the
    push, then — minutes after the live trade's lock-exit — a
    ~30-point collapse straight through the structural stop
    (4110.43) to ~4095-4100.

    What the fixed engine does on this path, exactly:
    - NO acceptance close occurs in this construction: the
      push WICKED through TP1 (ticks to +1.2R) but every M5
      bar CLOSED below it (last close 4139.90 < TP1 4140.00),
      so the bar-close pass stamps nothing.
    - The lock never arms; no modify_sl of any kind fires on
      the way up or on the way down (the runner's posture is
      earned breakeven too, and trail/chandelier sit above
      their 1.5R activations at a 1.2R best excursion).
    - The position rides to its STRUCTURAL stop: the exit on
      this path is 4110.43 for the full -1R (before any
      slip-through), booked by the broker stop — the tick
      manager takes no management action at all.

    That is the earned rule's honest price on an UNPROVEN
    move, and the fixture pins it deliberately: the old touch
    arm banked +0.49R here by accident of timing — the same
    arm that noise-killed 540274935 for +0.46R against a
    +1.43R objective.  The control leg below shows the old
    arm's protection on the identical path, so the trade-off
    is on record, not hidden."""
    runner, ex, sender = _runner_with_position(tmp_path, 4139.90)
    tm = runner.tick_manager
    runner._manage_positions_bar_close({}, clocks=("M5",))
    assert tm.pl_state(TICKET)["earned"] is False
    hits = []
    for price in (TOUCH_1R, TOUCH_12R,          # the push
                  4120.0, 4112.0, 4110.20,      # the collapse
                  4100.0, 4096.0):              # through the stop
        acts = tm.on_tick(TICKET, price)
        assert [a for a in acts if a.kind == "modify_sl"] == []
        hits += [(price, a) for a in acts if a.kind == "stop_hit"]
    # The engine books the exit itself: the first stop_hit
    # is at the STRUCTURAL stop, on the tick that crossed it
    # (the unit manager keeps reporting it below the stop;
    # the runner closes the position on the first) — the -1R
    # exit, exactly.
    assert hits
    assert hits[0][0] == 4110.20
    assert all(a.level == pytest.approx(STOP0) for _, a in hits)
    assert tm.effective_stop(TICKET) == pytest.approx(STOP0)
    assert tm.pl_state(TICKET)["done"] is False
    assert tm.last_sl_reason(TICKET) == ""
    # Control: the old touch arm (any non-earned playbook) on
    # the identical path locks at the touch and the collapse
    # then exits at the lock — the protection being traded
    # away on unproven moves, shown not hidden.
    tm2 = TickManager(tick=TickConfig(breakeven_mode="earned"))
    _open(tm2, playbook="fvg_limit")
    locks = _locks(tm2.on_tick(TICKET, TOUCH_1R))
    assert len(locks) == 1
    assert locks[0].level == pytest.approx(LOCK)
    tm2.apply(TICKET, locks[0])
    hits = [a for a in tm2.on_tick(TICKET, 4100.0)
            if a.kind == "stop_hit"]
    assert len(hits) == 1
    assert hits[0].level == pytest.approx(LOCK)
    assert tm2.effective_stop(TICKET) == pytest.approx(LOCK)


def test_collapse_after_acceptance_exits_at_the_lock(tmp_path):
    """The PROVEN move keeps its protection: acceptance is
    stamped at the M5 close, the lock fires on the next tick,
    and the same collapse then exits at the lock (4131.52,
    +0.5R class) — never at the structural stop.  No extra
    guard machinery is needed for this path: the first tick
    after the stamp fires the lock at any side-valid price."""
    runner, ex, sender = _runner_with_position(tmp_path, ACCEPT_CLOSE)
    tm = runner.tick_manager
    runner._manage_positions_bar_close({}, clocks=("M5",))
    assert tm.pl_state(TICKET)["earned"] is True
    runner._manage_tick(ACCEPT_CLOSE, {"tick_actions": 0},
                        bid=ACCEPT_CLOSE, ask=ACCEPT_CLOSE + 0.2)
    assert len(ex.modify_calls) == 1
    assert ex.modify_calls[0][1] == pytest.approx(LOCK)
    hits = []
    for price in (4133.0, 4131.60, 4112.0, 4096.0):
        acts = tm.on_tick(TICKET, price)
        assert [a for a in acts if a.kind == "modify_sl"] == []
        hits += [(price, a) for a in acts if a.kind == "stop_hit"]
    # The engine books the exit at the LOCK on the tick that
    # crossed it — the proven move's +0.5R protection, intact.
    assert hits
    assert hits[0][0] == 4112.0
    assert all(a.level == pytest.approx(LOCK) for _, a in hits)
    assert tm.effective_stop(TICKET) == pytest.approx(LOCK)
    assert tm.pl_state(TICKET)["done"] is True


def test_gap_through_lock_level_before_first_tick_stands_down(
        tmp_path):
    """The one uncovered edge, pinned honestly: acceptance is
    stamped, but the FIRST tick after the stamp is already a
    collapse tick BELOW the lock level — the proposed lock is
    side-invalid (a long's stop cannot sit above the bid), the
    runner stands it down, and the position rides to the
    structural stop.  No re-arm rule can place a stop at the
    broker retroactively, and in every non-gap path the lock
    fires on the first post-stamp tick (previous test) — so
    no displacement-rearm machinery is added; the lock stays
    pending and lands if price recovers, as the recovery tick
    at the end shows."""
    runner, ex, sender = _runner_with_position(tmp_path, ACCEPT_CLOSE)
    tm = runner.tick_manager
    runner._manage_positions_bar_close({}, clocks=("M5",))
    assert tm.pl_state(TICKET)["earned"] is True
    runner._manage_tick(4120.0, {"tick_actions": 0},
                        bid=4120.0, ask=4120.2)
    assert ex.modify_calls == []
    assert "STOOD DOWN (profit_lock)" in _notes(tmp_path)
    assert tm.effective_stop(TICKET) == pytest.approx(STOP0)
    assert tm.pl_state(TICKET)["done"] is False
    # Recovery: the pending lock lands at the formula level.
    runner._manage_tick(4135.0, {"tick_actions": 0},
                        bid=4135.0, ask=4135.2)
    assert len(ex.modify_calls) == 1
    assert ex.modify_calls[0][1] == pytest.approx(LOCK)
    assert tm.pl_state(TICKET)["done"] is True
