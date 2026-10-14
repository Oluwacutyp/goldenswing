"""Layering / scale-in + manipulation filter in the MT5 signal path.

Tests run against a netting fake executor: MT5 nets same-direction
deals into ONE position (same ticket, volume-weighted entry), so the
fake does the same — a plain append-per-order fake would model a
hedging broker, not MT5.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.execution.executor import OrderRequest, OrderResult, Position
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.execution.tick_manager import TickManager
from goldenswing.live.manipulation import chase_blocked
from test_mt5_manipulation import _df, OFF, ASIA


class NettingFakeExecutor:
    """MT5 netting semantics: same symbol+direction deal merges."""

    def __init__(self):
        self.orders: list[dict] = []
        self.modifies: list[tuple] = []
        self.closes: list[tuple] = []
        self._positions: list[Position] = []
        self._ticket = 9000

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return [p for p in self._positions
                if symbol is None or p.symbol == symbol]

    def place_order(self, request):
        self._ticket += 1
        self.orders.append({"ticket": self._ticket,
                            "direction": request.direction,
                            "volume": request.volume,
                            "sl": request.stop_loss})
        for p in self._positions:
            if p.symbol == request.symbol \
                    and p.direction == request.direction:
                total = p.volume + request.volume
                p.entry_price = (p.entry_price * p.volume
                                 + request.entry_ref * request.volume) / total
                p.volume = total
                return OrderResult(True, ticket=p.ticket,
                                   price=request.entry_ref,
                                   volume=request.volume)
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
        for p in list(self._positions):
            if p.ticket == ticket:
                if volume is None or volume >= p.volume:
                    self._positions.remove(p)
                else:
                    p.volume -= volume
        return OrderResult(True, ticket=ticket)


def _runner(tmp_path, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        fixed_lots=0.01,
        **cfg_kw)
    feed = SimpleNamespace(account_state=lambda: {"equity": 10000.0})
    return MT5Runner(cfg, feed=feed, executor=NettingFakeExecutor())


def _base_position(runner, ticket=101):
    """A long base: entry 2000, invalidation 1990, ATR 10 -> stop0
    1987.5, risk 12.5, 1.0 lot."""
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=1.0,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2030.0], atr=10.0, bar_index=5)
    runner.tick_manager.open(ticket=ticket, request=req, fill_price=2000.0,
                             entry_bar=6, broker_sl=1987.5)
    return req


def _layer_sig(**kw):
    d = dict(direction=1, score=80.0, entry_ref=2010.0,
             invalidation=2000.0, targets=[2040.0], atr=10.0, index=100)
    d.update(kw)
    return SimpleNamespace(**d)


# ----------------------------------------------------------- TickManager
def test_tick_manager_base_layer_and_add_layer():
    tm = TickManager()
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=1.0,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2030.0], atr=10.0, bar_index=5)
    tm.open(ticket=101, request=req, fill_price=2000.0, entry_bar=6,
            broker_sl=1987.5)
    assert len(tm.layers(101)) == 1
    assert tm.layers(101)[0].entry == 2000.0
    layer = tm.add_layer(101, entry=2010.0, volume=0.5, sl=1997.5,
                         playbook="fvg_midline")
    assert layer is not None
    assert len(tm.layers(101)) == 2
    assert tm.position(101).volume == pytest.approx(1.5)


def test_tick_manager_layer_stop_closes_layer_only():
    tm = TickManager()
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=1.0,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2030.0], atr=10.0, bar_index=5)
    tm.open(ticket=101, request=req, fill_price=2000.0, entry_bar=6,
            broker_sl=1987.5)
    tm.add_layer(101, entry=2010.0, volume=0.5, sl=1997.5)
    # Tick to the layer SL (above the base stop 1987.5): layer stops.
    actions = tm.on_tick(101, 1997.0)
    assert [a.kind for a in actions] == ["layer_stop"]
    assert actions[0].volume == pytest.approx(0.5)
    tm.apply(101, actions[0])
    assert len(tm.layers(101)) == 1
    assert tm.position(101).volume == pytest.approx(1.0)
    assert tm.position(101).closed is False  # base rides


def test_tick_manager_trim_closes_top_layer():
    tm = TickManager()
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=1.0,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2030.0], atr=10.0, bar_index=5)
    tm.open(ticket=101, request=req, fill_price=2000.0, entry_bar=6,
            broker_sl=1987.5)
    tm.add_layer(101, entry=2010.0, volume=0.5, sl=1997.5)
    tm.add_layer(101, entry=2015.0, volume=0.25, sl=2002.5)
    action = tm.close_top_layer(101)
    assert action is not None and action.kind == "trim"
    assert action.volume == pytest.approx(0.25)
    tm.apply(101, action)
    assert len(tm.layers(101)) == 2
    assert tm.position(101).volume == pytest.approx(1.5)
    # Trim with only the base layer: nothing to trim.
    tm2 = TickManager()
    tm2.open(ticket=7, request=req, fill_price=2000.0, entry_bar=6,
             broker_sl=1987.5)
    assert tm2.close_top_layer(7) is None


# ------------------------------------------------------------- _try_layer
def test_layer_added_only_in_profit_same_direction(tmp_path):
    runner = _runner(tmp_path)
    _base_position(runner)
    runner._last_tick = {"mid": 2010.0}  # +0.8R: in profit
    runner._stack = SimpleNamespace(frames={"H1": _df(OFF + ASIA)})
    stats: dict = {}
    runner._try_layer(SimpleNamespace(name="fvg_midline"),
                      _layer_sig(), stats)
    assert stats.get("layers") == 1
    layers = runner.tick_manager.layers(101)
    assert len(layers) == 2
    assert layers[1].entry == pytest.approx(2010.0)
    assert layers[1].volume == pytest.approx(0.01)
    # Broker SL moved to breakeven (base entry), tightening only.
    assert runner.executor.modifies[-1] == (101, pytest.approx(2000.0))


def test_layer_skipped_when_flat_or_against(tmp_path):
    runner = _runner(tmp_path)
    _base_position(runner)
    runner._last_tick = {"mid": 2002.0}  # +0.16R: not in profit
    stats: dict = {}
    runner._try_layer(SimpleNamespace(name="fvg_midline"),
                      _layer_sig(), stats)
    assert stats.get("layers", 0) == 0
    assert len(runner.tick_manager.layers(101)) == 1
    # Against the position: no layer, no flip.
    runner._last_tick = {"mid": 2010.0}
    runner._try_layer(SimpleNamespace(name="fvg_midline"),
                      _layer_sig(direction=-1, entry_ref=1990.0,
                                 invalidation=2000.0), stats)
    assert stats.get("layers", 0) == 0
    assert len(runner.tick_manager.layers(101)) == 1


def test_layer_risk_cap_blocks(tmp_path):
    # Total risk cap 2x base: base risk $1250, one layer of 0.01 lot
    # adds ~$12.50 — a tiny cap (1.01x) must refuse a second layer
    # once the first is in (budget < 0.01 lot).
    runner = _runner(tmp_path, layer_max_total_risk_mult=1.01)
    _base_position(runner)
    runner._last_tick = {"mid": 2010.0}
    runner._stack = SimpleNamespace(frames={"H1": _df(OFF + ASIA)})
    stats: dict = {}
    runner._try_layer(SimpleNamespace(name="fvg_midline"),
                      _layer_sig(), stats)
    assert stats.get("layers") == 1
    # Second layer: budget = 1.01*1250 - (1250 + 12.50) = -12.50 < 0.
    runner._try_layer(SimpleNamespace(name="trendline_retest"),
                      _layer_sig(index=101, entry_ref=2012.0,
                                 invalidation=2002.0), stats)
    assert stats.get("layers") == 1  # still one


def test_layer_max_layers(tmp_path):
    runner = _runner(tmp_path, layer_max_layers=2)
    _base_position(runner)
    runner._last_tick = {"mid": 2010.0}
    runner._stack = SimpleNamespace(frames={"H1": _df(OFF + ASIA)})
    stats: dict = {}
    runner._try_layer(SimpleNamespace(name="fvg_midline"),
                      _layer_sig(), stats)
    assert stats.get("layers") == 1
    runner._try_layer(SimpleNamespace(name="trendline_retest"),
                      _layer_sig(index=101), stats)
    assert stats.get("layers") == 1  # max 2 layers incl. base


def test_layer_stop_via_runner_closes_layer_only(tmp_path):
    runner = _runner(tmp_path)
    _base_position(runner)
    runner._last_tick = {"mid": 2010.0}
    runner._stack = SimpleNamespace(frames={"H1": _df(OFF + ASIA)})
    stats: dict = {}
    # Layer with a TIGHT stop (invalidation 2005 -> stop0 2002.5), above
    # the base breakeven 2000: the layer SL is reachable before the
    # broker SL.  (A layer SL below breakeven is unreachable by design
    # — the broker SL at breakeven fires first and protects the stack.)
    runner._try_layer(SimpleNamespace(name="fvg_midline"),
                      _layer_sig(invalidation=2005.0), stats)
    assert len(runner.tick_manager.layers(101)) == 2
    # Tick to the layer SL (2002.5): layer closes, base rides.
    stats2: dict = {"tick_actions": 0}
    runner._manage_tick(2002.0, stats2)
    assert len(runner.tick_manager.layers(101)) == 1
    assert runner.tick_manager.position(101).closed is False
    assert runner.tick_manager.position(101).volume == pytest.approx(1.0)
    assert (101, pytest.approx(0.01)) in runner.executor.closes


def test_trim_top_layer_via_runner(tmp_path):
    runner = _runner(tmp_path)
    _base_position(runner)
    runner._last_tick = {"mid": 2010.0}
    runner._stack = SimpleNamespace(frames={"H1": _df(OFF + ASIA)})
    stats: dict = {}
    runner._try_layer(SimpleNamespace(name="fvg_midline"),
                      _layer_sig(), stats)
    assert len(runner.tick_manager.layers(101)) == 2
    msg = runner.trim_top_layer()
    assert "trimmed top layer" in msg
    assert len(runner.tick_manager.layers(101)) == 1
    assert runner.tick_manager.position(101).volume == pytest.approx(1.0)
    assert runner.trim_top_layer() == "only the base layer — nothing to trim"


def test_pending_layer_fills_onto_position(tmp_path):
    runner = _runner(tmp_path)
    _base_position(runner)
    # A pending layer: entry behind price (pullback) -> LIMIT.
    runner._last_tick = {"mid": 2015.0}
    runner._stack = SimpleNamespace(frames={"H1": _df(OFF + ASIA)})
    stats: dict = {}
    runner._try_layer(SimpleNamespace(name="fvg_midline"),
                      _layer_sig(entry_ref=2010.0, invalidation=2000.0),
                      stats)
    assert stats.get("pending") == 1
    assert len(runner._pending_orders) == 1
    state = next(iter(runner._pending_orders.values()))
    assert state.is_layer is True
    # Fill it: the layer lands on the held position, not a new one.
    pos = SimpleNamespace(ticket=101, entry_price=2010.0)
    runner._record_pending_fill(state, pos, stats)
    assert stats.get("layer_fills") == 1
    assert len(runner.tick_manager.layers(101)) == 2
    assert runner.tick_manager.tickets() == [101]


# ------------------------------------------------- manipulation in signal path
def _signal_path_runner(tmp_path, h1):
    runner = _runner(tmp_path)
    runner._stack = SimpleNamespace(frames={"H1": h1})
    runner._last_tick = {"mid": 102.0}
    runner._build_chart = lambda *a, **k: (None, None)
    runner._send_alert = lambda *a, **k: None
    return runner


def _stub_engine():
    class Stub:
        def gate_live_signal(self, pb, sig, ctx, hierarchy_verdict=None):
            return True, ""

        def score_live_signal(self, pb, sig, ctx, hierarchy_verdict=None):
            return 80.0, ["stub"]

    return Stub()


def _verdict():
    return {"bias_d1": 1, "bias_h4": 1, "trigger": "BOS", "entry_clock": "M15",
            "amd_phase": "distribution", "qm": {"status": "none"},
            "volume": {"volume_ratio": 1.0}, "zone": None}


def test_signal_path_blocks_manipulation_chase(tmp_path):
    h1 = _df(OFF + ASIA + [
        (101.0, 103.5, 101.0, 101.5),   # Asia high swept, close back
        (101.5, 102.0, 101.0, 101.8),
    ])
    runner = _signal_path_runner(tmp_path, h1)
    sig = SimpleNamespace(direction=1, score=80.0, entry_ref=103.0,
                          invalidation=100.0, targets=[105.0], atr=1.0,
                          index=len(h1) - 1)
    pb = SimpleNamespace(name="fvg_midline", live=True,
                         live_signal=lambda states, t, df: [sig])
    runner._get_playbooks = lambda: [pb]
    runner._get_engine = lambda: _stub_engine()
    stats: dict = {"signals": 0}
    runner._run_signal_path(h1, len(h1) - 1, [], None, _verdict(), stats)
    assert stats.get("manipulation_blocked") == 1
    assert runner.executor.orders == []  # no chase order
    assert runner.tick_manager.tickets() == []


def test_signal_path_allows_retest_after_sweep(tmp_path):
    h1 = _df(OFF + ASIA + [
        (101.0, 103.5, 101.0, 101.5),
        (101.5, 102.0, 101.0, 101.8),
    ])
    runner = _signal_path_runner(tmp_path, h1)
    sig = SimpleNamespace(direction=1, score=80.0, entry_ref=102.0,
                          invalidation=100.0, targets=[105.0], atr=1.0,
                          index=len(h1) - 1)
    pb = SimpleNamespace(name="fvg_midline", live=True,
                         live_signal=lambda states, t, df: [sig])
    runner._get_playbooks = lambda: [pb]
    runner._get_engine = lambda: _stub_engine()
    stats: dict = {"signals": 0}
    runner._run_signal_path(h1, len(h1) - 1, [], None, _verdict(), stats)
    assert stats.get("manipulation_blocked", 0) == 0
    assert len(runner.executor.orders) == 1  # retest entry allowed
