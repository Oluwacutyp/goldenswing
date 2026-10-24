"""MacroTrend wiring: feed W1/MN1, alignment refresh, sizing,
Golden Swing confirm (stage / confirm / reject / timeout), and the
unchanged hierarchy leg set.  Against fake MT5 + fake executor —
no real MT5, no real money."""
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.execution.executor import OrderRequest, OrderResult, Position
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
from goldenswing.live.hierarchy import (LEG_ORDER, STACK_TFS,
                                        TimeframeStack)
from goldenswing.live.macrotrend import MacroTrendResult
from mt5_fakes import FakeMT5


class FakeExecutor:
    def __init__(self):
        self.orders: list[dict] = []
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
                            "volume": request.volume})
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
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        self._positions = [p for p in self._positions
                           if p.ticket != ticket]
        return OrderResult(True, ticket=ticket)


class FakeSender:
    """Captures sends; accepts the reply_markup the confirm path
    passes (legacy senders without it get the TypeError fallback)."""

    def __init__(self):
        self.texts: list[tuple] = []
        self.photos: list = []

    def send_text(self, text, reply_markup=None):
        self.texts.append((text, reply_markup))
        return SimpleNamespace(delivered=True, detail="sent")

    def send_photo(self, png, caption):
        self.photos.append((png, caption))
        return SimpleNamespace(delivered=True, detail="sent")


def _runner(tmp_path, *, feed=None, executor=None, senders=None,
            **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        # News pause OFF: the order path's news check reads the
        # REAL wall-clock calendar — these tests must not fail on
        # a real news day (e.g. PPI).  The pause itself is covered
        # by test_news_calendar.py.
        news_pause_enabled=False,
        **cfg_kw)
    return MT5Runner(cfg, feed=feed, executor=executor, senders=senders)


def _sig(**kw):
    base = dict(direction=1, score=80.0, entry_ref=2000.0,
                invalidation=1990.0, targets=[2030.0], atr=10.0,
                index=100, factors=[])
    base.update(kw)
    return SimpleNamespace(**base)


def _pb(name="fvg_midline"):
    return SimpleNamespace(name=name)


def _golden():
    return MacroTrendResult(
        d1_ema=1, w1_ema=1, mn1_ema=1,
        d1_structure=1, w1_structure=1, mn1_structure=1,
        alignment_level=1.5, alignment_label="Golden_Swing",
        golden_swing=True, conflict=False, structure_confirms=True,
        size_multiplier=1.5)


def _conflict():
    return MacroTrendResult(
        d1_ema=1, w1_ema=-1, mn1_ema=1,
        alignment_level=1.0, alignment_label="D1_Only",
        golden_swing=False, conflict=True, structure_confirms=False,
        size_multiplier=0.5)


def _ready_runner(tmp_path, **cfg_kw):
    """Runner with feed + stack + live setup, ready to place."""
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    feed.connect()
    runner = _runner(tmp_path, feed=feed, executor=FakeExecutor(),
                     senders=[FakeSender()], **cfg_kw)
    runner._stack = TimeframeStack.from_frames(feed.warm_start_all())
    runner._live_setup = {"direction": 1}
    return runner


# ------------------------------------------------------------------ feed
def test_warm_start_and_fetch_include_w1_mn1():
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    feed.connect()
    warm = feed.warm_start_all()
    assert {"W1", "MN1"} <= set(warm)
    assert len(warm["W1"]) > 200  # enough for EMA200
    assert len(warm["MN1"]) > 200
    fresh = feed.fetch_new_all({})
    assert {"W1", "MN1"} <= set(fresh)


def test_closed_only_w1_mn1_calendar():
    # A bar inside the current month/week is forming; the bar
    # before it is closed.  MN1 must NOT use the 43200min duration
    # (a 31-day month's bar would close a day early).
    idx = pd.date_range("2026-09-01", periods=3, freq="MS", tz="UTC")
    frame = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
         "volume": 1.0}, index=idx)
    now = pd.Timestamp("2026-10-15 12:00", tz="UTC")
    closed = MT5Feed._closed_only(frame, "MN1", now)
    # Oct 1 bar is the CURRENT month's forming bar -> excluded;
    # only Sep 1 is closed.
    assert list(closed.index) == list(idx[:1])
    idx_w = pd.date_range("2026-10-05", periods=3, freq="W-MON",
                          tz="UTC")
    frame_w = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
         "volume": 1.0}, index=idx_w)
    closed_w = MT5Feed._closed_only(frame_w, "W1", now)
    # Oct 12 bar is the CURRENT week's forming bar -> excluded;
    # only Oct 5 is closed.
    assert list(closed_w.index) == list(idx_w[:1])


def test_closed_bar_close_w1_mn1():
    ts = pd.Timestamp("2026-10-15 12:34", tz="UTC")  # a Thursday
    assert MT5Feed.closed_bar_close("W1", ts) == \
        pd.Timestamp("2026-10-12", tz="UTC")  # Monday
    assert MT5Feed.closed_bar_close("MN1", ts) == \
        pd.Timestamp("2026-10-01", tz="UTC")


# ------------------------------------------------------------- alignment
def test_alignment_refresh_from_stack(tmp_path):
    runner = _ready_runner(tmp_path)
    runner._refresh_alignment()
    # FakeMT5 bars rise on every TF -> all three EMA biases +1.
    assert runner._alignment.golden_swing is True
    assert "Golden_Swing" in runner.alignment_line()


def test_alignment_refresh_only_journals_on_change(tmp_path):
    runner = _ready_runner(tmp_path)
    runner._refresh_alignment()
    journal = tmp_path / "mt5_journal.jsonl"
    before = journal.read_text().count("macro alignment")
    runner._refresh_alignment()  # same alignment: no new note
    after = journal.read_text().count("macro alignment")
    assert after == before


# --------------------------------------------------------------- sizing
def test_alignment_sizing_multiplier(tmp_path):
    runner = _ready_runner(tmp_path)
    spec = gold_spec()
    runner._alignment = _golden()
    assert runner._apply_alignment_multiplier(0.10, spec, "x") == \
        pytest.approx(0.15)
    runner._alignment = _conflict()
    assert runner._apply_alignment_multiplier(0.10, spec, "x") == \
        pytest.approx(0.05)
    runner._alignment = MacroTrendResult()  # neutral 1.0x
    assert runner._apply_alignment_multiplier(0.10, spec, "x") == \
        pytest.approx(0.10)


def test_alignment_sizing_floors_to_step(tmp_path):
    runner = _ready_runner(tmp_path)
    spec = gold_spec()
    runner._alignment = _golden()
    # 0.07 x 1.5 = 0.105 -> floor to step 0.01 -> 0.10, never up.
    assert runner._apply_alignment_multiplier(0.07, spec, "x") == \
        pytest.approx(0.10)


def test_alignment_sizing_disabled_is_noop(tmp_path):
    runner = _ready_runner(tmp_path,
                           golden_swing_sizing_enabled=False)
    runner._alignment = _golden()
    assert runner._apply_alignment_multiplier(0.10, gold_spec(),
                                              "x") == pytest.approx(0.10)


def test_dynamic_lots_are_alignment_multiplied(tmp_path):
    # PREMISE CORRECTION (owner ruling 2026-10-23): the original
    # test pinned "fixed lots are never multiplied" — the
    # fixed_lots override is REVOKED.  On the sole dynamic path
    # the alignment multiplier DOES apply to the risk-sized
    # base lots (floored to the broker step, never up).
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    plain = _ready_runner(plain_dir)          # neutral alignment
    stats: dict = {"executed": 0}
    plain._place_live_order(_pb(), _sig(), stats)
    assert len(plain.executor.orders) == 1
    base = plain.executor.orders[0]["volume"]

    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    golden = _ready_runner(golden_dir)
    golden._alignment = _golden()           # Golden Swing 1.5x
    stats2: dict = {"executed": 0}
    golden._place_live_order(_pb(), _sig(), stats2)
    assert len(golden.executor.orders) == 1
    assert golden.executor.orders[0]["volume"] == \
        pytest.approx(base * 1.5)


# -------------------------------------------------------------- confirm
def _stage(runner, sig=None, **kw):
    sig = sig or _sig()
    runner._stage_order_confirm(_pb(), sig, {}, verdict={}, entry_bar=5)
    return sig


def test_stage_places_nothing(tmp_path):
    runner = _ready_runner(tmp_path)
    _stage(runner)
    assert runner.executor.orders == []
    assert len(runner._order_confirms) == 1
    token, staged = next(iter(runner._confirm_tokens.items()))
    assert staged["action"] == "place_order"
    # The Confirm message carries the inline keyboard.
    texts = runner.senders[0].texts
    assert texts and texts[-1][1] is not None
    assert "GOLDEN SWING" in texts[-1][0]


def test_confirm_places_order(tmp_path):
    runner = _ready_runner(tmp_path)
    _stage(runner)
    token = next(iter(runner._confirm_tokens))
    ok, text = runner.confirm_pending(token, {})
    assert ok is True
    assert "order placed" in text
    assert len(runner.executor.orders) == 1
    assert runner._order_confirms == {}
    assert runner._confirm_tokens == {}


def test_reject_cancels_nothing_placed(tmp_path):
    runner = _ready_runner(tmp_path)
    _stage(runner)
    token = next(iter(runner._confirm_tokens))
    text = runner.cancel_confirm(token)
    assert "rejected" in text
    assert runner.executor.orders == []
    assert runner._order_confirms == {}
    assert runner._confirm_tokens == {}


def test_confirm_is_single_use(tmp_path):
    runner = _ready_runner(tmp_path)
    _stage(runner)
    token = next(iter(runner._confirm_tokens))
    assert runner.confirm_pending(token, {})[0] is True
    ok, text = runner.confirm_pending(token, {})
    assert ok is False
    assert "expired" in text
    assert len(runner.executor.orders) == 1


def test_confirm_refused_when_paused(tmp_path):
    runner = _ready_runner(tmp_path)
    _stage(runner)
    token = next(iter(runner._confirm_tokens))
    runner.cfg.paused = True
    ok, text = runner.confirm_pending(token, {})
    assert ok is False
    assert "paused" in text
    assert runner.executor.orders == []


def test_confirm_refused_when_setup_flipped(tmp_path):
    runner = _ready_runner(tmp_path)
    _stage(runner)
    token = next(iter(runner._confirm_tokens))
    runner._live_setup = {"direction": -1}
    ok, text = runner.confirm_pending(token, {})
    assert ok is False
    assert "setup gone" in text
    assert runner.executor.orders == []


def test_confirm_with_held_position_routes_to_layer(tmp_path):
    runner = _ready_runner(tmp_path)
    # Hold one position in the same direction, at entry (not in
    # profit) -> _try_layer skips; no second base order is placed.
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2030.0], atr=10.0, bar_index=5)
    runner.tick_manager.open(ticket=101, request=req,
                             fill_price=2000.0, entry_bar=6)
    _stage(runner)
    token = next(iter(runner._confirm_tokens))
    ok, text = runner.confirm_pending(token, {})
    assert ok is True
    assert "layer" in text
    assert runner.executor.orders == []


def test_timeout_fires_per_config(tmp_path):
    runner = _ready_runner(
        tmp_path, golden_swing_confirm_timeout_sec=0.0,
        golden_swing_confirm_timeout_action="fire")
    _stage(runner)
    runner._sweep_order_confirms({})
    assert len(runner.executor.orders) == 1
    assert runner._order_confirms == {}


def test_timeout_cancels_per_config(tmp_path):
    runner = _ready_runner(
        tmp_path, golden_swing_confirm_timeout_sec=0.0,
        golden_swing_confirm_timeout_action="cancel")
    _stage(runner)
    runner._sweep_order_confirms({})
    assert runner.executor.orders == []
    assert runner._order_confirms == {}
    assert runner._confirm_tokens == {}


def test_sweep_ignores_unexpired(tmp_path):
    runner = _ready_runner(
        tmp_path, golden_swing_confirm_timeout_sec=3600.0)
    _stage(runner)
    runner._sweep_order_confirms({})
    assert len(runner._order_confirms) == 1
    assert runner.executor.orders == []


def test_sweep_ignores_control_tokens(tmp_path):
    runner = _ready_runner(
        tmp_path, golden_swing_confirm_timeout_sec=0.0)
    token = runner.request_confirm("pause")
    runner._sweep_order_confirms({})
    assert token in runner._confirm_tokens  # never expired


# -------------------------------------------------- signal-path gating
def _run_one(runner, *, golden: bool):
    runner._alignment = _golden() if golden else MacroTrendResult()
    sig = _sig()
    pb = _pb()
    pb.live_signal = lambda states, t, h1: [sig]
    engine = SimpleNamespace(
        gate_live_signal=lambda *a, **k: (True, ""),
        score_live_signal=lambda *a, **k: (80.0, []))
    verdict = {"bias_d1": 1, "bias_h4": 1, "trigger": "t",
               "entry_clock": "M15", "amd_phase": None,
               "qm": {"status": None}, "volume": {}}
    stats: dict = {"gated": 0, "signals": 0}
    h1 = runner._stack.frames["H1"]
    runner._run_one_playbook(pb, sig_ctx=(SimpleNamespace(states=None),
                                          engine),
                             h1=h1, t=len(h1) - 1, verdict=verdict,
                             stats=stats)


def test_signal_path_stages_when_golden(tmp_path):
    runner = _ready_runner(tmp_path, manipulation_filter=False)
    _run_one(runner, golden=True)
    assert runner.executor.orders == []  # nothing placed yet
    assert len(runner._order_confirms) == 1


def test_signal_path_places_when_not_golden(tmp_path):
    runner = _ready_runner(tmp_path, manipulation_filter=False)
    _run_one(runner, golden=False)
    assert len(runner.executor.orders) == 1
    assert runner._order_confirms == {}


def test_signal_path_stages_off_when_confirm_disabled(tmp_path):
    runner = _ready_runner(tmp_path, manipulation_filter=False,
                           golden_swing_confirm_enabled=False)
    _run_one(runner, golden=True)
    assert len(runner.executor.orders) == 1
    assert runner._order_confirms == {}


# ------------------------------------------------- hierarchy leg set
def test_hierarchy_leg_set_unchanged():
    assert LEG_ORDER == ("data", "d1_bias", "h4_agree", "h4_zone",
                         "h1_trigger", "entry")
    assert STACK_TFS == ("D1", "H4", "H1", "M15", "M30", "M5")
    assert "W1" not in LEG_ORDER and "MN1" not in LEG_ORDER
