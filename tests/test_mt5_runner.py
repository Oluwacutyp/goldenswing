"""MT5 runner: demo guard, tick management, live orders, bar
boundaries, hierarchy verdicts on MT5 bars — against fake terminal +
fake executor (no real MT5, no real money)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.execution.executor import OrderResult, Position
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
from goldenswing.live.hierarchy import TF_MINUTES, TimeframeStack
from mt5_fakes import FakeMT5


class FakeExecutor:
    def __init__(self):
        self.orders: list[dict] = []
        self.modifies: list[tuple] = []
        self.closes: list[tuple] = []
        self._ticket = 9000
        self._positions: list[Position] = []

    def connect(self, **kw):
        return True, "fake connected"

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return [p for p in self._positions
                if symbol is None or p.symbol == symbol]

    def place_order(self, request):
        self._ticket += 1
        self.orders.append({"ticket": self._ticket,
                            "symbol": request.symbol,
                            "direction": request.direction,
                            "volume": request.volume,
                            "sl": request.stop_loss,
                            "tp": request.take_profit})
        self._positions.append(Position(
            ticket=self._ticket, symbol=request.symbol,
            direction=request.direction, volume=request.volume,
            entry_price=request.entry_ref, sl=request.stop_loss,
            tp=request.take_profit, magic=request.magic,
            alert_id=request.alert_id))
        return OrderResult(True, ticket=self._ticket,
                           price=request.entry_ref,
                           volume=request.volume)

    def modify_sl(self, ticket, sl, tp=None):
        self.modifies.append((ticket, sl))
        for p in self._positions:
            if p.ticket == ticket:
                p.sl = sl
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        self.closes.append((ticket, volume))
        self._positions = [p for p in self._positions
                           if p.ticket != ticket]
        return OrderResult(True, ticket=ticket)


def _runner(tmp_path, *, feed=None, executor=None, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        **cfg_kw)
    return MT5Runner(cfg, feed=feed, executor=executor)


def test_demo_guard_refuses_real_account(tmp_path):
    feed = MT5Feed(MT5FeedConfig(),
                   mt5_module=FakeMT5(trade_mode=2))
    runner = _runner(tmp_path, feed=feed, executor=FakeExecutor())
    with pytest.raises(RuntimeError, match="refusing to trade a REAL"):
        runner.run(max_ticks=1)


def test_full_run_pumps_ticks(tmp_path):
    script = [(2000.0, 2000.30, 0), (2000.10, 2000.40, 1)]
    feed = MT5Feed(MT5FeedConfig(tick_interval=0.001),
                   mt5_module=FakeMT5(tick_script=script))
    runner = _runner(tmp_path, feed=feed, executor=FakeExecutor())
    stats = runner.run(max_ticks=2)
    assert stats["ticks"] == 2


def test_manage_tick_breakeven_via_runner(tmp_path):
    runner = _runner(tmp_path, executor=FakeExecutor())
    from goldenswing.execution.executor import OrderRequest
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=1.0,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2030.0], atr=10.0, bar_index=5)
    runner.tick_manager.open(ticket=101, request=req, fill_price=2000.0,
                             entry_bar=6)
    stats: dict = {"tick_actions": 0}
    runner._manage_tick(2012.5, stats)
    assert runner.executor.modifies == [(101, pytest.approx(2000.0))]
    assert stats["tick_actions"] == 1


def test_place_live_order_uses_buffered_stop_and_tracks(tmp_path):
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    feed.connect()
    runner = _runner(tmp_path, feed=feed, executor=FakeExecutor())
    runner._stack = TimeframeStack.from_frames(feed.warm_start_all())
    sig = SimpleNamespace(direction=1, score=80.0, entry_ref=2000.0,
                          invalidation=1990.0, targets=[2030.0],
                          atr=10.0, index=100)
    stats: dict = {"executed": 0}
    runner._place_live_order(SimpleNamespace(name="fvg_midline"), sig,
                             stats)
    assert len(runner.executor.orders) == 1
    order = runner.executor.orders[0]
    assert order["sl"] == pytest.approx(1987.5)  # buffered stop0
    assert stats["executed"] == 1
    assert runner.tick_manager.position(order["ticket"]) is not None


def test_refresh_bars_only_on_boundary_cross(tmp_path):
    now = pd.Timestamp.now(tz="UTC")
    h1_open = now.floor("h") - pd.Timedelta(hours=1)  # last closed bar
    h1 = pd.DataFrame(
        {"open": [2000.0], "high": [2001.0], "low": [1999.0],
         "close": [2000.5], "volume": [100], "real_volume": [0],
         "spread": [30]},
        index=pd.DatetimeIndex([h1_open], tz="UTC"))

    class FakeFeed:
        def __init__(self):
            self.fetches = 0

        def fetch_new_all(self, cursors):
            self.fetches += 1
            empty = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume",
                         "real_volume", "spread"])
            return {"H1": empty}

    feed = FakeFeed()
    runner = _runner(tmp_path, feed=feed, executor=FakeExecutor())
    runner._stack = TimeframeStack.from_frames({"H1": h1})
    runner._last_processed = {
        "H1": h1_open + pd.Timedelta(minutes=TF_MINUTES["H1"])}
    runner._qm_touch = None
    stats: dict = {}
    # Inside the current H1 bar: no fetch.
    runner._refresh_bars(
        {"time": (h1_open + pd.Timedelta(minutes=90)).to_pydatetime()},
        stats)
    assert feed.fetches == 0
    # Past the H1 boundary: fetch once.
    runner._refresh_bars(
        {"time": (h1_open + pd.Timedelta(minutes=121)).to_pydatetime()},
        stats)
    assert feed.fetches == 1


def test_verdict_on_mt5_bars_carries_volume(tmp_path):
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    feed.connect()
    runner = _runner(tmp_path, feed=feed, executor=FakeExecutor())
    # Slice to 400 bars/TF: plenty for the hierarchy window (300),
    # far cheaper to featurize than the full 2000-bar warm start.
    frames = {tf: f.tail(400)
              for tf, f in feed.warm_start_all().items()}
    runner._stack = TimeframeStack.from_frames(frames)
    runner._qm_touch = None
    h1 = runner._stack.frames["H1"]
    from goldenswing.features.engine import read_and_featurize
    states = read_and_featurize(h1, None)
    verdict = runner._verdict_for_bar(h1, len(h1) - 1, states)
    assert verdict is not None
    assert "volume" in verdict
    assert "legs" in verdict and "blocked_leg" in verdict
    assert verdict["veto_enabled"] is False
