"""Step 4 (2026-10-27): route-aware pending management.

Proven against the 2026-10-26 harvest, not asserted from the brief:

- 2026-10-26 11:55:02 UTC a qm_m5 SHORT pending (ticket 514827226,
  entry 4019.16, structural stop 4021.04) was placed, then cancelled
  12:00:27 "bias_flipped: setup unarmed" + decay 1/2, never filled.
  Root causes: (a) _check_pending_bias applied the H1 trigger
  premise (_live_setup, armed ONLY by the H1 path) to a native
  pending that never had that premise; (b) the H1 roster cannot
  re-score a native playbook, so decay read "gone" for every native
  pending.  Under step 4 the native pending must survive BOTH
  checks while H1-premise pendings keep their guards.
- A native-frame re-emission decay was investigated and REJECTED:
  the QM live_signal is touch-only and cursor-based (signals
  touched on bar t, never re-emits a processed bar), so absence
  of a re-emission proves nothing about a resting limit.  Native
  pendings are guarded by entry-clock expiry + invalidation
  instead, and H1 decay skips ALL native premises (step 2 exempted
  only the no-floor qm_m5; qm_d1, which carries a floor, died the
  same way).
- Entry-clock expiry preserves the H1 window's wall duration
  (pending_expiry_bars H1 bars) counted in entry-clock bars:
  qm_m5 -> 48 M5 bars (4h, same duration, finer granularity);
  qm_d1 -> 1 D1 bar (a daily route cannot be evaluated more
  finely than daily closes).
"""

from types import SimpleNamespace

import pandas as pd

from goldenswing.execution.executor import OrderResult
from goldenswing.execution.mt5_runner import (
    MT5Runner, MT5RunnerConfig)
from goldenswing.live.hierarchy import TimeframeStack


# ---------------------------------------------------------------- helpers
class _FakeSpec:
    tick_size = 0.01
    tick_value = 1.0

    def money_for_move(self, distance, volume):
        return abs(distance) / self.tick_size * self.tick_value * volume


class _Exec:
    def __init__(self):
        self.cancelled: list[int] = []

    def cancel_pending(self, ticket):
        self.cancelled.append(ticket)
        return OrderResult(True, ticket=ticket)

    def positions(self, symbol=None):
        return []

    def symbol_spec(self, symbol):
        return _FakeSpec()


def _h1_frame(n=60):
    return pd.DataFrame(
        {"open": [2000.0] * n, "high": [2001.0] * n,
         "low": [1999.0] * n, "close": [2000.5] * n,
         "volume": [100] * n, "real_volume": [0] * n,
         "spread": [30] * n},
        index=pd.date_range("2026-01-01", periods=n, freq="h",
                            tz="UTC"))


def _m5_frame(n=100, breach_high=None):
    highs = [4018.0] * n
    if breach_high is not None and n:
        highs[-1] = breach_high
    return pd.DataFrame(
        {"open": [4017.0] * n, "high": highs,
         "low": [4016.0] * n, "close": [4017.5] * n,
         "volume": [100] * n, "real_volume": [0] * n,
         "spread": [30] * n},
        index=pd.date_range("2026-01-01", periods=n, freq="5min",
                            tz="UTC"))


def _d1_frame(n=40):
    return pd.DataFrame(
        {"open": [2000.0] * n, "high": [2001.0] * n,
         "low": [1999.0] * n, "close": [2000.5] * n,
         "volume": [100] * n, "real_volume": [0] * n,
         "spread": [30] * n},
        index=pd.date_range("2026-01-01", periods=n, freq="D",
                            tz="UTC"))


def _runner(tmp_path, m5=None, d1=None):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg, feed=None, executor=_Exec())
    frames = {"H1": _h1_frame()}
    if m5 is not None:
        frames["M5"] = m5
    if d1 is not None:
        frames["D1"] = d1
    runner._stack = TimeframeStack.from_frames(frames)
    runner._playbooks = []
    return runner


def _order(ticket, direction, level, invalidation):
    return SimpleNamespace(
        ticket=ticket, kind="limit", direction=direction,
        price=level, volume=0.01, sl=invalidation, tp=None,
        alert_id=f"MT5-test-{ticket}")


def _track(runner, playbook, ticket=514827226, direction=-1,
           level=4019.16, invalidation=4021.04):
    return runner._track_pending_order(
        _order(ticket, direction, level, invalidation),
        playbook=playbook, zone=None, invalidation=invalidation,
        placed_bar=runner._current_h1_bar(),
        sig=SimpleNamespace(score=30.75), verdict=None,
        track_request=None, decision_score=30.75)


def _note_text(runner):
    return "\n".join(
        r.get("text", "") for r in runner.journal.records()
        if r.get("type") == "note")


# ------------------------------------------------------- premise derivation
def test_native_premise_and_entry_clock_derived_at_tracking(tmp_path):
    runner = _runner(tmp_path, m5=_m5_frame(100), d1=_d1_frame(40))
    native = _track(runner, "qm_m5")
    assert native.premise == "native"
    assert native.entry_clock == "M5"
    assert native.entry_placed_bar == 100
    # 4 H1 bars of wall time, counted in M5 bars.
    assert native.entry_expiry_bar == 100 + 48
    # H1 pendings are unchanged: entry_* == placed / expiry.
    h1_state = _track(runner, "sweep_reversal", ticket=1)
    assert h1_state.premise == "h1"
    assert h1_state.entry_clock == "H1"
    assert h1_state.entry_placed_bar == h1_state.placed_bar
    assert h1_state.entry_expiry_bar == h1_state.expiry_bar
    # Daily route: 1 D1 bar (cannot be evaluated more finely).
    d1_state = _track(runner, "qm_d1", ticket=2)
    assert d1_state.premise == "native"
    assert d1_state.entry_clock == "D1"
    assert d1_state.entry_expiry_bar == d1_state.entry_placed_bar + 1
    # Restart-adopted (unknown playbook) keeps the old H1 behaviour.
    adopted = _track(runner, "", ticket=3)
    assert adopted.premise == "h1"
    assert adopted.entry_clock == "H1"


# ------------------------------------------------------- the 4019.16 sequence
def test_4019_16_native_survives_setup_unarmed_and_decay(tmp_path):
    runner = _runner(tmp_path, m5=_m5_frame(100))
    state = _track(runner, "qm_m5")
    runner._live_setup = None
    stats: dict = {}
    # The 2026-10-26 sequence, replayed: D1/H4 close with no live
    # setup must NOT cancel the native pending.
    runner._check_pending_bias(stats)
    assert state.ticket in runner._pending_orders
    assert runner.executor.cancelled == []
    assert "native premise" in _note_text(runner)
    # And H1-clock decay must not touch it either (qm_d1 case too).
    runner._check_pending_score_decay(stats)
    assert state.ticket in runner._pending_orders
    assert state.decay_bars == 0


def test_qm_d1_pending_skips_h1_decay(tmp_path):
    # Step 2 exempted only the no-floor route; qm_d1 carries a
    # floor and still died by "gone" re-scores on H1 closes.
    runner = _runner(tmp_path, m5=_m5_frame(100), d1=_d1_frame(40))
    state = _track(runner, "qm_d1", ticket=7)
    runner._check_pending_score_decay({})
    assert state.ticket in runner._pending_orders
    assert state.decay_bars == 0


def test_h1_pending_cancelled_setup_unarmed(tmp_path):
    runner = _runner(tmp_path)
    state = _track(runner, "sweep_reversal", ticket=11, direction=-1)
    runner._live_setup = None
    runner._check_pending_bias({})
    assert state.ticket not in runner._pending_orders
    assert state.ticket in runner.executor.cancelled
    # Renamed reason: an un-armed premise is not a bias flip.
    assert "setup_unarmed" in _note_text(runner)


def test_h1_pending_cancelled_on_direction_flip_native_survives(tmp_path):
    runner = _runner(tmp_path, m5=_m5_frame(100))
    h1_state = _track(runner, "sweep_reversal", ticket=21, direction=-1)
    native = _track(runner, "qm_m5", ticket=22, direction=-1)
    runner._live_setup = {"direction": +1}
    runner._check_pending_bias({})
    assert h1_state.ticket not in runner._pending_orders
    assert native.ticket in runner._pending_orders
    assert "bias_flipped" in _note_text(runner)


# ------------------------------------------------------- native-clock expiry
def test_native_expiry_fires_on_m5_close_not_h1_units(tmp_path):
    runner = _runner(tmp_path, m5=_m5_frame(100))
    state = _track(runner, "qm_m5")
    assert state.entry_expiry_bar == 148
    # Advance the M5 frame past entry expiry (H1 frame untouched).
    runner._stack = TimeframeStack.from_frames(
        {"H1": _h1_frame(), "M5": _m5_frame(150)})
    runner._check_native_pendings("M5", {})
    assert state.ticket not in runner._pending_orders
    assert state.ticket in runner.executor.cancelled
    assert "48 M5 bars" in _note_text(runner)


def test_h1_close_loop_skips_native_pending(tmp_path):
    # Even with the H1 expiry pair already passed, the H1-close
    # loop must not kill a native pending (its clock governs it).
    runner = _runner(tmp_path, m5=_m5_frame(100))
    state = _track(runner, "qm_m5")
    state.expiry_bar = 10  # H1 pair long passed (last H1 index 59)
    runner._pending = None
    runner._on_h1_close(
        {"mid": 4017.5, "time": "2026-01-03T00:00:00Z"}, {})
    assert state.ticket in runner._pending_orders
    assert state.ticket not in runner.executor.cancelled


def test_native_invalidation_fires_on_m5_close(tmp_path):
    runner = _runner(tmp_path, m5=_m5_frame(100))
    state = _track(runner, "qm_m5", direction=-1,
                   level=4019.16, invalidation=4021.04)
    # A later M5 bar trades through the structural invalidation.
    runner._stack = TimeframeStack.from_frames(
        {"H1": _h1_frame(), "M5": _m5_frame(101, breach_high=4022.0)})
    runner._check_native_pendings("M5", {})
    assert state.ticket not in runner._pending_orders
    assert "invalidated" in _note_text(runner)


def test_audit_uses_native_clock_for_native_pending(tmp_path):
    runner = _runner(tmp_path, m5=_m5_frame(100))
    state = _track(runner, "qm_m5")
    # H1 pair passed, native pair not passed: audit must NOT call
    # it expired (it falls through to the POI check instead).
    state.expiry_bar = 10
    result = runner._audit_pending_reason(
        state, _h1_frame(), 4017.5, 1.0)
    assert result is not None and result[0] != "expired"
    # Native pair passed: audit calls it expired on the M5 clock.
    runner._stack = TimeframeStack.from_frames(
        {"H1": _h1_frame(), "M5": _m5_frame(150)})
    result = runner._audit_pending_reason(
        state, _h1_frame(), 4017.5, 1.0)
    assert result is not None and result[0] == "expired"
    assert "M5" in result[1]
