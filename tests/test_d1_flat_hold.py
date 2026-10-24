"""d1_flat HOLD state: flat is a HOLD, never a feed error.

Regression / contract tests for the 2026-10-16 d1_flat diagnosis:
D1 structure bias stays flat during choppy / transitioning regimes
because a structural swing needs ``swing_n`` (3) bars AFTER it before
``find_swings`` recognises it, so a D1 flip is only recognised >= 3
daily closes after the extreme printed.  The engine must handle /
log that as HOLD — pending swings, bars-remaining countdown, flip
requirements, earliest flip — and must never misdiagnose it as a
feed error.  Feed errors ride the separate FEED_ERROR channel.
"""
import logging

import pandas as pd
import pytest

from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.live.hierarchy import (
    D1BiasState, bias_hold_state, evaluate_hierarchy, live_setup,
    live_setup_hold_state, live_setup_reason, log_feed_error,
    log_hold_state)
from goldenswing.live import multitf as multitf_mod
from goldenswing.live import macrotrend as macrotrend_mod
from goldenswing.live.multitf import evaluate_bias
from goldenswing.live.hierarchy import TimeframeStack


# ------------------------------------------------------------------ builders
def _d1_flat_frame() -> pd.DataFrame:
    """40 D1 bars: confirmed pair LH + HL (flat), pending low forming.

    Confirmed highs 4400 @4 -> 4300 @12 -> 4250 @36 (LH).
    Confirmed lows 4100 @8 -> 4150 @16 (HL).  Mixed -> bias 0.
    Pending low 4100 @38 (1/3 bars confirmed, 2 remaining) — below
    the last confirmed low, so the SHORT low leg is forming; no
    pending high, so the SHORT high leg is missing.
    """
    idx = pd.date_range("2026-09-01", periods=40, freq="D", tz="UTC")
    rows = []
    # Gentle rising filler: strictly increasing, no fractal swings.
    filler = [4180 + 2.5 * t for t in range(17)]
    spec = {
        4: (4400.0, 4250.0, 4325.0),   # peak 1 (confirmed high)
        8: (4200.0, 4100.0, 4150.0),   # trough 1 (confirmed low)
        12: (4300.0, 4240.0, 4270.0),  # peak 2 — lower high
        16: (4200.0, 4150.0, 4175.0),  # trough 2 — higher low -> flat
        36: (4250.0, 4220.0, 4235.0),  # peak 3 — lower high, confirmed
        38: (4120.0, 4100.0, 4110.0),  # pending trough — lower low
    }
    for i in range(40):
        if i in spec:
            hi, lo, cl = spec[i]
        elif 17 <= i <= 33:
            cl = filler[i - 17]
            hi, lo = cl + 3.0, cl - 3.0
        elif i in (5, 6, 7):
            cl = 4300.0 - 30.0 * (i - 4)
            hi, lo = cl + 10.0, cl - 10.0
        elif i in (9, 10, 11):
            cl = 4150.0 + 20.0 * (i - 8)
            hi, lo = cl + 10.0, cl - 10.0
        elif i in (13, 14, 15):
            cl = 4270.0 - 30.0 * (i - 12)
            hi, lo = cl + 10.0, cl - 10.0
        elif i in (0, 1, 2, 3):
            cl = 4000.0 + 10.0 * i
            hi, lo = cl + 10.0, cl - 10.0
        elif i == 34:
            cl, hi, lo = 4210.0, 4213.0, 4207.0
        elif i == 35:
            cl, hi, lo = 4200.0, 4203.0, 4197.0
        elif i == 37:
            cl, hi, lo = 4200.0, 4203.0, 4197.0
        elif i == 39:
            cl, hi, lo = 4150.0, 4153.0, 4147.0
        else:  # pragma: no cover - every index is covered above
            raise AssertionError(f"uncovered bar {i}")
        rows.append((cl - 5.0, hi, lo, cl))
    return pd.DataFrame(
        rows, columns=["open", "high", "low", "close"], index=idx
    ).assign(volume=100, real_volume=0, spread=30)


def _flat_stack() -> dict:
    """Full stack: flat D1 + ordinary H4/H1 frames (bias leg only)."""
    d1 = _d1_flat_frame()
    h4_idx = pd.date_range("2026-10-09", periods=40, freq="4h", tz="UTC")
    h4_rows = [(4000.0 + i, 4010.0 + i, 3990.0 + i, 4005.0 + i)
               for i in range(40)]
    h4 = pd.DataFrame(h4_rows, columns=["open", "high", "low", "close"],
                      index=h4_idx).assign(volume=100, real_volume=0,
                                           spread=30)
    h1_idx = pd.date_range("2026-10-10", periods=40, freq="h", tz="UTC")
    h1_rows = [(4000.0 + i, 4010.0 + i, 3990.0 + i, 4005.0 + i)
               for i in range(40)]
    h1 = pd.DataFrame(h1_rows, columns=["open", "high", "low", "close"],
                      index=h1_idx).assign(volume=100, real_volume=0,
                                           spread=30)
    return {"D1": d1, "H4": h4, "H1": h1}


class _FakeFeed:
    def fetch_new_all(self, cursors):
        empty = pd.DataFrame(columns=["open", "high", "low", "close",
                                      "volume", "real_volume", "spread"])
        return {tf: empty for tf in
                ("D1", "H4", "H1", "M15", "M30", "M5")}

    def account_state(self):
        return {"balance": 10000.0, "equity": 10000.0,
                "free_margin": 10000.0}

    def order_margin(self, *, volume, price, direction):
        return 2.0


class _FakeExecutor:
    def connect(self, **kw):
        return True, "fake connected"


def _runner(tmp_path, frames):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        execute_playbooks=(),
        # sizing: dynamic risk rule (fixed_lots revoked 2026-10-23)
        news_pause_enabled=False,
        golden_swing_confirm_enabled=False)
    runner = MT5Runner(cfg, feed=_FakeFeed(), executor=_FakeExecutor())
    runner._stack = TimeframeStack.from_frames(frames)
    return runner


# ------------------------------------------------------------------- HOLD state
def test_hold_state_flat_mixed_pending_low():
    d1 = _d1_flat_frame()
    assert evaluate_bias(d1, swing_n=3) == 0
    state = bias_hold_state(d1, swing_n=3)
    assert isinstance(state, D1BiasState)
    assert state.bias == 0
    assert state.reason == "flat_mixed"
    assert state.confirmed_highs[-2:] == [4300.0, 4250.0]
    assert state.confirmed_lows[-2:] == [4100.0, 4150.0]
    # Pending low 4100 @ bar 38: 1 bar closed after it, 2 remaining.
    assert state.pending_low is not None
    assert state.pending_low["price"] == 4100.0
    assert state.pending_low["bars_remaining"] == 2
    assert state.pending_low["bars_confirmed"] == 1
    # No pending high in the last 3 bars.
    assert state.pending_high is None
    # SHORT low leg forming, SHORT high leg missing.
    assert "lower high" in state.flip_needed
    assert state.earliest_flip is None  # no direction has both legs forming


def test_hold_state_countdown_then_flip():
    d1 = _d1_flat_frame()
    # Append 1 bar above the pending low: remaining 2 -> 1.
    idx = pd.date_range("2026-10-11", periods=1, freq="D", tz="UTC")
    extra = pd.DataFrame([(4120.0, 4130.0, 4120.0, 4125.0)],
                         columns=["open", "high", "low", "close"],
                         index=idx).assign(volume=100, real_volume=0,
                                           spread=30)
    d1b = pd.concat([d1, extra])
    state = bias_hold_state(d1b, swing_n=3)
    assert state.pending_low is not None
    assert state.pending_low["bars_remaining"] == 1
    assert state.pending_low["bars_confirmed"] == 2
    assert evaluate_bias(d1b, swing_n=3) == 0  # still unconfirmed
    # Append 1 more bar above the pending low: it confirms (LL) and
    # with the confirmed LH pair the bias flips to -1.
    idx2 = pd.date_range("2026-10-12", periods=1, freq="D", tz="UTC")
    extra2 = pd.DataFrame([(4130.0, 4140.0, 4130.0, 4135.0)],
                          columns=["open", "high", "low", "close"],
                          index=idx2).assign(volume=100, real_volume=0,
                                            spread=30)
    d1c = pd.concat([d1b, extra2])
    assert evaluate_bias(d1c, swing_n=3) == -1
    state_c = bias_hold_state(d1c, swing_n=3)
    assert state_c.bias == -1
    assert state_c.pending_low is None  # confirmed, no longer pending


def test_live_setup_hold_state_and_reason():
    frames = _flat_stack()
    assert live_setup(frames) is None
    assert live_setup_reason(frames).startswith("d1_flat")
    state = live_setup_hold_state(frames)
    assert state is not None
    assert state.bias == 0
    assert state.pending_low is not None


def test_evaluate_hierarchy_flat_carries_hold_state():
    frames = _flat_stack()
    verdict = evaluate_hierarchy(frames)
    assert verdict["allowed"] is False
    assert verdict["blocked_leg"] == "d1_bias"
    hold = verdict["hold_state"]
    assert hold is not None
    assert hold["bias"] == 0
    assert hold["pending_low"]["price"] == 4100.0
    assert hold["pending_low"]["bars_remaining"] == 2
    assert "lower high" in hold["flip_needed"]


def test_hold_state_insufficient_swings():
    idx = pd.date_range("2026-10-01", periods=8, freq="D", tz="UTC")
    rows = [(4000.0 + i, 4010.0 + i, 3990.0 + i, 4005.0 + i)
            for i in range(8)]
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                         index=idx).assign(volume=100, real_volume=0,
                                           spread=30)
    state = bias_hold_state(frame, swing_n=3)
    assert state.reason == "insufficient_confirmed_swings"
    assert "2 confirmed swings" in state.flip_needed


# ------------------------------------------------------- logging separation
def test_hold_logs_info_not_feed_error(caplog):
    logger = logging.getLogger("test_hold_channel")
    with caplog.at_level(logging.INFO, logger="test_hold_channel"):
        log_hold_state(logger, bias_hold_state(_d1_flat_frame()))
    text = caplog.text
    assert "HOLD" in text
    assert "FEED_ERROR" not in text
    assert "d1_flat" in text


def test_feed_error_logs_error_not_hold(caplog):
    logger = logging.getLogger("test_feed_channel")
    with caplog.at_level(logging.INFO, logger="test_feed_channel"):
        log_feed_error(logger, symbol="XAUUSDm",
                       reason="terminal not connected",
                       last_bar_age="41.5h")
    text = caplog.text
    assert "FEED_ERROR" in text
    assert "HOLD" not in text
    assert "XAUUSDm" in text


def test_runner_hold_vs_feed_error_channels(tmp_path, caplog):
    runner = _runner(tmp_path, _flat_stack())
    with caplog.at_level(
            logging.INFO, logger="goldenswing.execution.mt5_runner"):
        runner._refresh_live_setup()
        # Healthy feed: nothing on the FEED_ERROR channel.
        runner.report_feed_status({"ok": True, "reason": None})
    assert "HOLD" in caplog.text
    assert "FEED_ERROR" not in caplog.text
    assert runner._live_hold_state is not None
    caplog.clear()
    with caplog.at_level(
            logging.INFO, logger="goldenswing.execution.mt5_runner"):
        # Stale feed + (separately) a nonzero bias would be
        # FEED_ERROR; here the feed report alone must not mention HOLD.
        runner.report_feed_status(
            {"ok": False, "reason": "D1 newest closed bar 41.5h old"})
    assert "FEED_ERROR" in caplog.text
    assert "HOLD" not in caplog.text


# ------------------------------------------------------------ /status render
def test_status_hold_rendering(tmp_path):
    runner = _runner(tmp_path, _flat_stack())
    runner._refresh_live_setup()
    line = runner.setup_line()
    assert line.startswith("HOLD d1_flat")
    assert "pending swing low 4100.0" in line
    assert "1/3 bars confirmed" in line
    assert "need" in line
    assert "FEED_ERROR" not in line
    # Macro alignment section is untouched (sizing only) — the
    # runner still computes it from the stack.
    assert "D1" in runner.alignment_line()


# ------------------------------------------------------------------- contract
def test_contract_documented():
    assert "CONTRACT" in (multitf_mod.__doc__ or "")
    assert "d1_flat" in (multitf_mod.__doc__ or "")
    assert "sizing only" in (macrotrend_mod.__doc__ or "").lower()


def test_describe_example_shape():
    state = bias_hold_state(_d1_flat_frame(), swing_n=3)
    line = state.describe()
    assert line.startswith("HOLD d1_flat")
    assert "pending swing low 4100.0 (1/3 bars confirmed)" in line
