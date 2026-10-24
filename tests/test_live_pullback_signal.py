"""Live-path pullback signal: D1 bias + H4 pullback into the POI.

Regression test for the 2026-10-16 zero-signal root cause: the live
setup required H4 bias == D1 bias (``h4_agree``), a second vote —
democracy, not hierarchy.  During a pullback H4 NECESSARILY trends
against D1 while price retraces into the D1-side H4 POI, so the
canonical lecture setup (D1 bias -> H4 zone -> H1 trigger -> M15/M5
entry) could never arm: measured on real gold bars, the agreement
vote armed 21% of H1 bars and every armed-then-blocked case showed
``blocked legs: none`` because playbooks never ran.  The fix: D1
decides direction, H4 supplies the POI zone; H4 bias is context
(``aligned`` / ``pullback`` / ``flat``), never a vote.

These tests drive the REAL live clock (ticks -> touch -> POI confirm
-> signal path -> order) on synthetic bars: D1 bearish + H4 bullish
pullback + M15 BOS confirm -> SHORT at the H4 resistance POI.
"""
import math
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.alerts.commands import CommandContext
from goldenswing.execution.executor import OrderResult, Position
from goldenswing.execution.mt5_commands import _mt5_status
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.live.hierarchy import (TimeframeStack, check_live_trigger,
                                        evaluate_hierarchy, live_setup,
                                        live_setup_reason)
from goldenswing.live.multitf import evaluate_bias, find_swings


# ------------------------------------------------------------------ builders
def _frame(closes, idx):
    rows = []
    prev = closes[0]
    for c in closes:
        rows.append((prev, c + 0.25, c - 0.25, c))
        prev = c
    return pd.DataFrame(
        rows, columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex(idx, tz="UTC")).assign(
            volume=100, real_volume=0, spread=30)


def _zigzag(direction, n, start, slope, amp, idx):
    """Trending zigzag: strict fractal swings (period-7 sine).

    ``amp`` must dominate ``slope`` or the trend swallows the
    oscillation and no fractal swings form (bias 0).
    """
    prices = [start + direction * slope * t
              + amp * math.sin(2 * math.pi * t / 7.0)
              for t in range(n)]
    return _frame(prices, idx)


def _pullback_stack(*, h4_slope=0.4, forming_open="2026-10-14 08:00"):
    """D1 -1, H4 +1 (pullback), H1 swing high nested in H4 resistance.

    H1 is shifted vertically so its most recent confirmed swing high
    lands exactly on the H4 resistance level (containment passes by
    construction).  The forming H1 bar opens at ``forming_open``
    (08:00 UTC = London killzone -> M15 entry clock).  M15 carries
    lookback bars + the composing bars for the forming bar, with a
    bearish BOS breaker at 08:30.
    """
    d1_idx = pd.date_range(end="2026-10-13", periods=40, freq="D", tz="UTC")
    d1 = _zigzag(-1, 40, 4300.0, 5.0, 30.0, d1_idx)
    assert evaluate_bias(d1, swing_n=3) == -1
    h4_idx = pd.date_range(end="2026-10-14 04:00", periods=40, freq="4h",
                           tz="UTC")
    h4 = _zigzag(+1, 40, 4100.0, h4_slope, 20.0, h4_idx)
    if h4_slope > 0:
        assert evaluate_bias(h4, swing_n=3) == +1
    else:
        assert evaluate_bias(h4, swing_n=3) == 0
    highs, _lows = find_swings(h4["high"].to_numpy(),
                                h4["low"].to_numpy(), 3)
    poi = float(highs[-1][1])
    h1_idx = pd.date_range(end="2026-10-14 07:00", periods=40, freq="h",
                           tz="UTC")
    h1 = _zigzag(+1, 40, 4150.0, 3.0, 15.0, h1_idx)
    h1_highs, _ = find_swings(h1["high"].to_numpy(),
                              h1["low"].to_numpy(), 3)
    shift = poi - float(h1_highs[-1][1])
    for col in ("open", "high", "low", "close"):
        h1[col] = h1[col] + shift
    # M15: lookback chop inside/near the zone + composing bars.
    m15_idx = pd.date_range("2026-10-14 05:00", periods=12, freq="15min",
                            tz="UTC")
    m15 = _frame([poi - 2.0 + 1.0 * math.sin(i) for i in range(12)],
                 m15_idx)
    comp_idx = pd.date_range(forming_open, periods=4, freq="15min",
                             tz="UTC")
    comp = _frame([poi - 2.0, poi + 3.0, poi - 18.0, poi - 8.0],
                  comp_idx)
    # Breaker bar (08:30): displacement bar closing below the zone.
    comp.loc[comp_idx[2], ["open", "high", "low", "close"]] = [
        poi + 3.0, poi + 5.0, poi - 22.0, poi - 18.0]
    m15 = pd.concat([m15, comp])
    m30_idx = pd.date_range("2026-10-14 00:00", periods=40, freq="30min",
                            tz="UTC")
    m30 = _frame([poi - 2.0] * 40, m30_idx)
    m5_idx = pd.date_range("2026-10-14 06:00", periods=60, freq="5min",
                           tz="UTC")
    m5 = _frame([poi - 2.0] * 60, m5_idx)
    return {"D1": d1, "H4": h4, "H1": h1, "M15": m15, "M30": m30,
            "M5": m5}


def _tick(when, mid):
    return {"time": pd.Timestamp(when, tz="UTC").to_pydatetime(),
            "mid": mid, "bid": mid, "ask": mid + 0.30,
            "tick_volume": 5, "real_volume": 0, "spread": 0.30}  # ask-bid, price units (real-feed convention)


class FakeExecutor:
    def __init__(self):
        self.orders: list[dict] = []
        self._ticket = 9000
        self._positions: list[Position] = []

    def connect(self, **kw):
        return True, "fake connected"

    def symbol_spec(self, symbol):
        from goldenswing.execution.risk import gold_spec
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
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        self._positions = [p for p in self._positions
                           if p.ticket != ticket]
        return OrderResult(True, ticket=ticket)


class FakeFeed:
    """MT5Feed stand-in: no new bars, $10k demo account, $2 margin."""

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


class StubPlaybook:
    """A playbook that always emits one high-score SHORT at the POI.

    Stands in for the M5 FVG short the lecture specifies: the
    playbook-pattern detection itself is covered by the playbook
    suites; this stub isolates the runner path (verdict -> gate ->
    score -> order) so the test proves the live clock REACHES the
    order step for a pullback setup the old code never armed.
    """

    name = "stub"
    session_scope = "any"
    hunting_ground = SimpleNamespace(excluded=[], bonuses={})

    def __init__(self, entry, stop, target):
        self._entry = entry
        self._stop = stop
        self._target = target

    def check_live_signal(self, signal, ctx):
        return True, ""

    def live_signal(self, states, t, df=None):
        return [SimpleNamespace(
            index=t, score=100.0, direction=-1,
            entry_ref=self._entry, invalidation=self._stop,
            targets=[self._target], factors=[], atr=1.0,
            kind="fvg_limit")]


def _runner(tmp_path, frames, *, playbook=None):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        execute_playbooks=("stub",),
        # sizing: dynamic risk rule (fixed_lots revoked 2026-10-23)
        news_pause_enabled=False,
        golden_swing_confirm_enabled=False)
    runner = MT5Runner(cfg, feed=FakeFeed(), executor=FakeExecutor())
    runner._stack = TimeframeStack.from_frames(frames)
    from goldenswing.live.hierarchy import TF_MINUTES
    for tf, frame in runner._stack.frames.items():
        runner._last_processed[tf] = (
            frame.index[-1] + pd.Timedelta(minutes=TF_MINUTES[tf]))
    runner._refresh_live_setup()
    if playbook is not None:
        runner._playbooks = [playbook]
    return runner


# ------------------------------------------------------------ hierarchy level
def _eval(runner, tick, stats):
    """Live-loop contract: _last_tick is the tick under evaluation.

    CLOCK ISOLATION (gs-clock pass): _on_tick sets _last_tick
    before evaluating, but direct _evaluate_live_tick callers
    left it None — the order path then fell back to the wall
    clock and the Friday-close gate refused orders on Friday
    evenings (2026-10-23).  Every direct evaluation goes
    through here so the engine clock is always the tick clock.
    """
    runner._last_tick = tick
    runner._evaluate_live_tick(tick, stats)


def test_pullback_setup_arms_short():
    frames = _pullback_stack()
    setup = live_setup(frames)
    assert setup is not None
    assert setup["direction"] == -1
    assert setup["bias_d1"] == -1
    assert setup["bias_h4"] == +1
    assert setup["h4_context"] == "pullback"
    assert setup["zone"]["kind"] == "resistance"
    assert live_setup_reason(frames) == "armed"


def test_pullback_trigger_fires_short_at_poi():
    frames = _pullback_stack()
    setup = live_setup(frames)
    assert setup is not None
    trig = check_live_trigger(
        setup, entry_frame=frames["M15"],
        h1_open=pd.Timestamp("2026-10-14 08:00", tz="UTC"),
        close=float(frames["M15"]["close"].iloc[-1]))
    assert trig is not None
    assert trig["direction"] == -1
    # SHAPE_PRIORITY picks engulfing over bos when both confirm;
    # either is a valid POI confirmation for the short.
    assert trig["trigger"] in ("bos", "engulfing")
    assert "bos" in trig["confirm_shapes"]
    # Entry is the confirm-tick close (just below the zone on a
    # bearish breakdown); stop sits just beyond the defended edge.
    zone = setup["zone"]
    assert trig["entry"] <= zone["top"] + setup["buf"]
    assert trig["stop"] == pytest.approx(zone["top"] + setup["buf"],
                                         abs=1e-9)
    assert trig["target"] < trig["entry"]  # short target below entry


def test_h4_flat_arms_when_poi_exists():
    # H4 with no bias (flat zigzag, slope 0) still has a POI zone:
    # a missing H4 opinion is not a veto.
    frames = _pullback_stack(h4_slope=0.0)
    assert evaluate_bias(frames["H4"], swing_n=3) == 0
    setup = live_setup(frames)
    assert setup is not None
    assert setup["direction"] == -1
    assert setup["h4_context"] == "flat"


def test_d1_flat_never_arms():
    frames = _pullback_stack()
    d1 = frames["D1"]
    frames["D1"] = _zigzag(0, 40, 4200.0, 0.0, 3.0, d1.index)
    assert evaluate_bias(frames["D1"], swing_n=3) == 0
    assert live_setup(frames) is None
    assert live_setup_reason(frames).startswith("d1_flat")


def test_containment_still_blocks():
    # H1 structure far from the H4 POI: no arm.  The fix removes
    # the H4 vote, NOT the containment gate.
    frames = _pullback_stack()
    h1 = frames["H1"]
    for col in ("open", "high", "low", "close"):
        h1[col] = h1[col] - 50.0  # structure $50 below the zone
    frames["H1"] = h1
    assert live_setup(frames) is None
    assert live_setup_reason(frames).startswith("containment")


def test_evaluate_hierarchy_pullback_leg_passes():
    v = evaluate_hierarchy(_pullback_stack(),
                           asof=pd.Timestamp("2026-10-14 08:00",
                                             tz="UTC"), swing_n=3)
    assert v["legs"]["h4_agree"]["passed"] is True
    assert v["h4_context"] == "pullback"
    assert v["blocked_leg"] != "h4_agree"


# -------------------------------------------------------------- runner level
def test_live_clock_pullback_reaches_order(tmp_path):
    frames = _pullback_stack()
    setup = live_setup(frames)
    assert setup is not None
    zone = setup["zone"]
    mid = (zone["bottom"] + zone["top"]) / 2
    stub = StubPlaybook(entry=mid, stop=zone["top"] + setup["buf"],
                        target=mid - 2 * (zone["top"] + setup["buf"]
                                          - mid))
    runner = _runner(tmp_path, frames, playbook=stub)
    assert runner._live_setup is not None
    assert runner._live_setup["h4_context"] == "pullback"
    stats: dict = {"ticks": 0, "bars": 0, "signals": 0, "executed": 0,
                   "shadow": 0, "gated": 0, "hierarchy_blocked": 0,
                   "tick_actions": 0, "reconciled_closed": 0,
                   "triggers": 0, "invalidated": 0}
    # One tick AT the POI: arms + confirms on the preloaded M15
    # breaker bars -> trigger fires -> signal path -> order.
    _eval(runner, _tick("2026-10-14 08:05", mid), stats)
    assert stats.get("triggers", 0) == 1
    assert runner._pending is None  # consumed by the confirmation
    assert len(runner.executor.orders) == 1
    order = runner.executor.orders[0]
    assert order["direction"] == -1
    # Dynamic sizing (fixed_lots revoked 2026-10-23): a real
    # risk-sized volume at or above the broker minimum.
    assert order["volume"] >= 0.01
    assert stats.get("signals", 0) == 1
    journal = (tmp_path / "mt5_journal.jsonl").read_text()
    assert "accepted" in journal
    # The journal must record the MEASURED H4 bias (+1 pullback),
    # not a hardcoded "H4 agrees" equal to the direction (-1).
    import json as _json
    accepted = [_json.loads(line) for line in journal.splitlines()
                if '"accepted"' in line]
    assert accepted, "no accepted signal journaled"
    assert accepted[0]["hierarchy"]["bias_h4"] == +1
    assert accepted[0]["direction"] == -1


def test_no_signal_when_truly_no_setup(tmp_path):
    # D1 flat: ticks at any price must not arm, confirm, or order.
    frames = _pullback_stack()
    d1 = frames["D1"]
    frames["D1"] = _zigzag(0, 40, 4200.0, 0.0, 3.0, d1.index)
    runner = _runner(tmp_path, frames,
                      playbook=StubPlaybook(entry=4200.0, stop=4201.0,
                                             target=4198.0))
    assert runner._live_setup is None
    assert runner._live_setup_reason.startswith("d1_flat")
    stats: dict = {"ticks": 0, "bars": 0, "signals": 0, "executed": 0,
                   "shadow": 0, "gated": 0, "hierarchy_blocked": 0,
                   "tick_actions": 0, "reconciled_closed": 0,
                   "triggers": 0, "invalidated": 0}
    for price in (4199.0, 4200.0, 4201.0):
        _eval(runner,
            _tick("2026-10-14 08:05", price), stats)
    assert runner._pending is None
    assert stats.get("triggers", 0) == 0
    assert runner.executor.orders == []


def test_status_shows_setup_state(tmp_path):
    frames = _pullback_stack()
    runner = _runner(tmp_path, frames)
    assert "ARMED SHORT" in runner.setup_line()
    assert "pullback" in runner.setup_line()
    assert "M15" in runner.stack_line()
    assert "H1" in runner.stack_line()
    ctx = CommandContext(feed_status={}, last_price=4114.0,
                         last_bar_time="2026-10-14 07:00",
                         journal=runner.journal, mt5_control=runner)
    text = _mt5_status(ctx, []).text
    assert "Live Setup" in text
    assert "ARMED SHORT" in text
    assert "pullback" in text
    # The old screenshot showed only the H1 last bar; the stack
    # line proves all six TFs are live.
    assert "bars (last closed, UTC)" in text
