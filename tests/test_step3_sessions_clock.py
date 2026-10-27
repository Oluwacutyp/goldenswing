"""Step 3 (2026-10-27): one live session authority + clock provenance.

- Live charts shade killzones from the DST-aware authority, not the
  fixed-UTC legacy mapper (the runner already decided on the
  authority — charts disagreeing with decisions around DST
  transitions was the live divergence).  Backtest/experiment
  mappers stay untouched by design: tests pin them and backtest
  comparability depends on them (see session_timeframes docstring).
- Every funnel evaluation record carries the session of the
  candidate bar AND of the evaluation instant.
- The order guard requires broker-open AGREEMENT: a broker-open
  flag inside the engine's weekly close is a stale flag and
  refuses; broker-closed (a real holiday) stays closed; legacy
  feeds without market_open() keep their old behaviour.
- Tick replay refuses future-dated ticks (the quarantined
  2026-10-27/28 harvest tick files were stamped after harvest).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from goldenswing.charts import engine as chart_engine
from goldenswing.charts import signal_view as chart_view
from goldenswing.execution.executor import OrderResult
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.live.hierarchy import TimeframeStack
from goldenswing.reader import session_timeframes
from goldenswing.reader import sessions as legacy_sessions
from scripts.replay_hot_scalp import replay


# ---------------------------------------------------------------- helpers
class _FakeSpec:
    """XAUUSD gold_spec shape: $0.01 tick worth $1.00 per lot."""

    tick_size = 0.01
    tick_value = 1.0

    def money_for_move(self, distance, volume):
        return abs(distance) / self.tick_size * self.tick_value * volume


class _Exec:
    def cancel_pending(self, ticket):
        return OrderResult(True, ticket=ticket)

    def positions(self, symbol=None):
        return []

    def symbol_spec(self, symbol):
        return _FakeSpec()


def _h1_frame():
    return pd.DataFrame(
        {"open": [2000.0] * 60, "high": [2001.0] * 60,
         "low": [1999.0] * 60, "close": [2000.5] * 60,
         "volume": [100] * 60, "real_volume": [0] * 60,
         "spread": [30] * 60},
        index=pd.date_range("2026-01-01", periods=60, freq="h",
                            tz="UTC"))


def _runner(tmp_path):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg, feed=None, executor=_Exec())
    runner._stack = TimeframeStack.from_frames({"H1": _h1_frame()})
    runner._playbooks = []
    return runner


def _frame(start, periods, freq):
    return pd.DataFrame(
        {"open": [4000.0] * periods, "high": [4001.0] * periods,
         "low": [3999.0] * periods, "close": [4000.5] * periods,
         "volume": [100] * periods},
        index=pd.date_range(start, periods=periods, freq=freq, tz="UTC"))


def _ticks(times):
    return pd.DataFrame(
        {"time": pd.to_datetime(times, utc=True),
         "bid": [4000.0] * len(times), "ask": [4000.1] * len(times)})


# ------------------------------------------------------- charts authority
def test_charts_use_dst_aware_session_authority():
    assert chart_engine.sessions is session_timeframes
    assert chart_view.sessions is session_timeframes


def test_authority_differs_from_legacy_across_dst():
    # Summer: NY wall clock is one UTC hour earlier, so the fixed-UTC
    # legacy killzone fires an hour late.  2026-07-01 11:30 UTC is
    # 07:30 New York (inside the NY killzone) but 11:30 UTC is
    # outside the legacy 12:00-15:00 UTC NY window.
    _, auth_kz = session_timeframes.session_of(
        pd.Timestamp("2026-07-01 11:30", tz="UTC"))
    _, legacy_kz = legacy_sessions.session_of(
        pd.Timestamp("2026-07-01 11:30", tz="UTC"))
    assert auth_kz == "ny_kz"
    assert legacy_kz is False


# ------------------------------------------------------- eval dual session
def test_eval_record_carries_both_sessions(tmp_path):
    runner = _runner(tmp_path)
    # Tick clock pins the evaluation instant (02:00 UTC, winter).
    runner._last_tick = {"time": "2026-01-05 02:00:00+00:00"}
    rec = runner._emit_eval(
        playbook="qm_m5", evaluation_clock="M5",
        last_bar_ts=pd.Timestamp("2026-01-05 14:00", tz="UTC"),
        source="live_close")
    assert rec is not None
    # Candidate bar 14:00 UTC winter: overlap bucket, 09:00 New York
    # wall clock inside the NY killzone.
    assert rec["session"]["candidate_bar"] == {
        "session": "overlap", "killzone": "ny_kz"}
    # Evaluation instant 02:00 UTC winter: asia bucket, no killzone.
    assert rec["session"]["evaluation"] == {
        "session": "asia", "killzone": None}
    assert rec["ts"].startswith("2026-01-05 02:00")


def test_eval_session_none_when_no_bar(tmp_path):
    runner = _runner(tmp_path)
    rec = runner._emit_eval(
        playbook="qm_m5", evaluation_clock="M5",
        last_bar_ts=None, source="live_close")
    assert rec["session"]["candidate_bar"] is None
    assert rec["session"]["evaluation"] is not None


# ------------------------------------------------- market-open agreement
def test_market_open_requires_calendar_agreement(tmp_path):
    runner = _runner(tmp_path)
    runner.feed = SimpleNamespace(market_open=lambda: (True, ""))
    with patch("goldenswing.live.market_hours.xauusd_market_open",
               return_value=False):
        ok, reason = runner._market_open()
    assert ok is False
    assert "calendar disagreement" in reason
    with patch("goldenswing.live.market_hours.xauusd_market_open",
               return_value=True):
        assert runner._market_open() == (True, "")


def test_market_open_broker_closed_stays_closed(tmp_path):
    runner = _runner(tmp_path)
    runner.feed = SimpleNamespace(
        market_open=lambda: (False, "symbol disabled"))
    with patch("goldenswing.live.market_hours.xauusd_market_open",
               return_value=True):
        ok, _reason = runner._market_open()
    assert ok is False


def test_market_open_legacy_feed_unchanged(tmp_path):
    runner = _runner(tmp_path)
    runner.feed = SimpleNamespace()  # no market_open method
    assert runner._market_open() == (True, "")


# ------------------------------------------------------- replay provenance
def test_replay_refuses_future_dated_ticks(tmp_path):
    now = pd.Timestamp.now(tz="UTC")
    frames = {"M1": _frame("2026-10-27", 60, "1min"),
              "M5": _frame("2026-10-27", 60, "5min"),
              "M15": _frame("2026-10-27", 60, "15min")}
    ticks = _ticks([now + pd.Timedelta(days=1)])
    with pytest.raises(ValueError, match="future-dated"):
        replay(symbol="XAUUSD", ticks=ticks, frames=frames,
               store_path=tmp_path / "s.db",
               output_path=tmp_path / "o.jsonl")


def test_replay_historical_ticks_still_run(tmp_path):
    frames = {"M1": _frame("2026-10-27", 60, "1min"),
              "M5": _frame("2026-10-27", 60, "5min"),
              "M15": _frame("2026-10-27", 60, "15min")}
    ticks = _ticks(pd.date_range("2026-10-27 00:01", periods=5,
                                 freq="1min", tz="UTC"))
    res = replay(symbol="XAUUSD", ticks=ticks, frames=frames,
                 store_path=tmp_path / "s.db",
                 output_path=tmp_path / "o.jsonl")
    assert res["ticks_processed"] == 5
    assert res["tick_first"].startswith("2026-10-27 00:01")


def test_replay_tolerates_small_clock_skew(tmp_path):
    now = pd.Timestamp.now(tz="UTC")
    start = now - pd.Timedelta(minutes=30)
    frames = {"M1": _frame(start, 60, "1min"),
              "M5": _frame(start, 60, "5min"),
              "M15": _frame(start, 60, "15min")}
    ticks = _ticks([now + pd.Timedelta(minutes=1)])
    res = replay(symbol="XAUUSD", ticks=ticks, frames=frames,
                 store_path=tmp_path / "s.db",
                 output_path=tmp_path / "o.jsonl", max_ticks=1)
    assert res["ticks_processed"] == 1
