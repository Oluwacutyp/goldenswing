"""Role-based alert charts (2026-11-05) + the ladder-tiling fix.

The live failure these pins exist against: ticket 540791314's
fill charts (EC2, 2026-11-05 07:01) — a hot_scalp signal carries
every planned hierarchy leg (20+ targets in two clusters); the
old renderer drew each as its own dashed line and its y-axis
loop re-tested candidates against the GROWING bounds, so the
ladder ratcheted the axis hundreds of points above the market
and the panels arrived as green tiled bands with the candles
crushed into the bottom.

Pins: the ladder is capped with true ranks everywhere it is
drawn (PNG trio, role charts, HTML view); axis admission is
tested against the ORIGINAL candle bounds; planned vs actual
are distinct objects in the role-A study and its labels; the
TickManager's executed-event log feeds the lifecycle; role
overlay families obey the <=4 law.
"""
import io
import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from goldenswing.charts.engine import (  # noqa: E402
    _mt5_panel_ylim, _mt5_trade_levels, render_mt5_roles,
    zones_to_study_dicts)
from goldenswing.charts.signal_model import (  # noqa: E402
    MAX_ROLE_OVERLAYS, admit_level_bounds, chart_targets,
    context_study_from_state, lifecycle_from_events,
    trade_label_specs)
from goldenswing.charts.signal_view import (  # noqa: E402
    build_signal_view_html)
from goldenswing.execution.executor import OrderRequest  # noqa: E402
from goldenswing.execution.tick_manager import (  # noqa: E402
    TickAction, TickManager)
from goldenswing.reader.state import (  # noqa: E402
    FairValueGap, LiquidityPool, MarketState, OrderBlock,
    StructureEvent, SwingPoint)

ENTRY = 4124.49
STOP = 4110.43
LADDER = [4126.0 + 1.5 * i for i in range(12)] + \
         [4250.0 + 6.0 * i for i in range(12)]


def _frame(n, freq, t0, base=4100.0, drift=0.3):
    idx = pd.date_range(t0, periods=n, freq=freq, tz="UTC")
    close = base + np.cumsum(np.full(n, drift)) + \
        np.sin(np.arange(n) * 0.7) * 1.2
    open_ = np.roll(close, 1)
    open_[0] = close[0] - 0.2
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + 0.5,
         "low": np.minimum(open_, close) - 0.5, "close": close,
         "volume": np.full(n, 150.0), "real_volume": 0, "spread": 30},
        index=idx)


# ------------------------------------------------------- ladder capping
def test_chart_targets_caps_ladder_with_true_ranks():
    shown, total = chart_targets(LADDER, entry=ENTRY, direction=1)
    assert total == 24
    assert [r for r, _t in shown] == [1, 2, 3, 24]
    assert shown[-1][1] == LADDER[-1]


def test_chart_targets_small_lists_pass_through():
    shown, total = chart_targets([4136.181, 4138.671], entry=4134.4,
                                 direction=1)
    assert total == 2
    assert [r for r, _t in shown] == [1, 2]


def test_chart_targets_short_direction_orders_by_proximity():
    legs = [4100.0 - 1.5 * i for i in range(10)]
    shown, total = chart_targets(legs, entry=4110.0, direction=-1)
    assert total == 10
    assert shown[0] == (1, 4100.0)
    assert shown[-1][0] == 10


def test_trade_levels_draw_at_most_cap_plus_top_for_a_ladder():
    fig, ax = plt.subplots()
    theme = {"up": "#26a69a", "down": "#ef5350", "text": "#131722",
             "bg": "#ffffff", "grid": "#e6e8ee", "candle_alpha": 1.0}
    specs = _mt5_trade_levels(
        ax, 100.0, entry=ENTRY, stop=STOP, broker_stop=STOP,
        targets=LADDER, direction=1, risk=ENTRY - STOP, fill_price=ENTRY,
        theme=theme)
    plt.close(fig)
    target_specs = [s for s in specs
                    if s["text"].strip().startswith("T")]
    assert len(target_specs) == 4  # 3 operative + ladder top
    assert "T24" in target_specs[-1]["text"]
    assert "ladder top" in target_specs[-1]["text"]


# ------------------------------------------------------- axis admission
def test_admit_level_bounds_never_ratchets_on_a_staircase():
    lo, hi = admit_level_bounds(4095.0, 4125.0, LADDER, mult=1.6)
    # Original span 30 -> admission ceiling 4125 + 48 = 4173, no
    # matter how the staircase climbs.
    assert hi <= 4173.0
    assert lo == 4095.0


def test_panel_ylim_ignores_the_ladder_top():
    view = _frame(60, "15min", "2026-11-05 04:00", base=4095.0)
    lo, hi = _mt5_panel_ylim(view, entry=ENTRY, stop=STOP,
                             broker_stop=STOP, targets=LADDER,
                             fill_price=ENTRY, zone=None)
    candle_top = float(view["high"].max())
    assert hi < candle_top + 1.7 * (candle_top
                                    - float(view["low"].min()))


# ------------------------------------------------------- lifecycle study
def _lifecycle(**over):
    kw = dict(
        direction=1, entry_ref=4134.507, structural_stop=4131.469,
        targets=(4136.181, 4136.296, 4138.671),
        entry_zone=(4132.01, 4135.883), state="closed",
        ticket=540274935, volume=0.01, fill_price=4134.437,
        initial_broker_stop=4131.469,
        events=[{"kind": "open", "ts": 1762340821.0,
                 "price": 4134.437},
                {"kind": "sl", "ts": 1762341248.0, "level": 4135.92,
                 "reason": "profit_lock"}],
        partials=[{"ts": 1762341000.0, "price": 4136.181,
                   "volume": 0.005}],
        exit_price=4135.797, exit_ts=1762341259.0,
        exit_reason="SL hit (profit_lock)")
    kw.update(over)
    return lifecycle_from_events(**kw)


def _fmt(x):
    return f"{x:,.1f}"


def test_lifecycle_events_become_stop_path_and_partials():
    lc = _lifecycle()
    assert lc.fill_ts == 1762340821.0  # from the open event
    assert [e.level for e in lc.stop_events] == [4135.92]
    assert lc.current_broker_stop == 4135.92
    assert len(lc.partials) == 1


def test_trade_label_specs_keep_planned_and_actual_distinct():
    texts = [s["text"] for s in trade_label_specs(_lifecycle(),
                                                  fmt=_fmt)]
    joined = " | ".join(texts)
    assert "FILL 4,134.4" in joined
    assert "STOP 4,131.5 (structural)" in joined
    assert "BROKER SL 4,131.5 (initial)" in joined
    assert "BROKER SL 4,135.9 (now)" in joined
    assert "EXIT 4,135.8" in joined
    assert "PARTIAL" in joined
    planned = [t for t in texts if "(planned)" in t]
    assert len(planned) == 3
    assert all(t.strip().startswith("T") for t in planned)


def test_trade_label_specs_planned_state_has_no_fill_or_exit():
    lc = _lifecycle(state="planned", fill_price=None, events=[],
                    partials=[], exit_price=None, exit_reason="",
                    initial_broker_stop=None)
    texts = [s["text"] for s in trade_label_specs(lc, fmt=_fmt)]
    joined = " | ".join(texts)
    assert "ENTRY 4,134.5 (planned)" in joined
    assert "FILL" not in joined
    assert "EXIT" not in joined
    assert "BROKER SL" not in joined


# ------------------------------------------------------- context study
def _dense_state():
    fvgs = [FairValueGap(top=4130.0 + 3 * i, bottom=4128.5 + 3 * i,
                         direction=1, created_at=100 + i)
            for i in range(8)]
    obs = [OrderBlock(top=4090.0 - 2 * i, bottom=4088.0 - 2 * i,
                      direction=1, created_at=90 + i, candle_index=89 + i)
           for i in range(8)]
    return MarketState(
        index=119, timestamp=pd.Timestamp("2026-11-05 11:00",
                                          tz="UTC"),
        close=4124.0, atr=3.5, trend="up", bias=1,
        recent_swing_highs=[SwingPoint(4130.0, 110, 112, "high")],
        recent_swing_lows=[SwingPoint(4100.0, 105, 107, "low")],
        last_structure_event=StructureEvent(112, "BOS", 1, 4128.0),
        structure_event_age=7,
        pools_above=[LiquidityPool(4140.0, "bsl", 2, 108),
                     LiquidityPool(4160.0, "bsl", 1, 100)],
        pools_below=[LiquidityPool(4105.0, "ssl", 3, 102)],
        active_fvgs=fvgs, active_obs=obs,
        range_high=4140.0, range_low=4100.0)


def test_context_study_caps_zones_and_picks_nearest_pools():
    ctx = context_study_from_state(_dense_state(), entry=4124.0,
                                   trigger="BOS")
    assert len(ctx.zones) <= 4
    # nearest pool each side only
    assert [(p, s) for p, s, _n in ctx.pools] == [(4140.0, "above"),
                                                  (4105.0, "below")]
    assert ctx.structure_event["kind"] == "BOS"
    assert ctx.structure_event["age_bars"] == 7
    assert len(ctx.overlay_families()) <= MAX_ROLE_OVERLAYS


def test_role_overlay_families_obey_the_law():
    lc = _lifecycle()
    assert len(lc.overlay_families()) <= MAX_ROLE_OVERLAYS
    from goldenswing.charts.signal_model import EntryStudy
    es = EntryStudy(entry_zone=(4132.01, 4135.883),
                    overlays=("ema_pair", "qm"))
    assert len(es.overlay_families()) <= MAX_ROLE_OVERLAYS


# ------------------------------------------------------- tick events
def test_tick_manager_records_the_executed_lifecycle():
    tm = TickManager()
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=4110.43, entry_ref=ENTRY,
                       targets=[4136.181], atr=3.6,
                       alert_id="HS-1", bar_index=10, magic=1)
    tm.open(ticket=4242, request=req, fill_price=ENTRY, entry_bar=10,
            broker_sl=4110.43, playbook="hot_scalp")
    tm.apply(4242, TickAction(kind="modify_sl", level=4131.52,
                              reason="profit_lock"))
    tm.bank_leg(4242, volume=0.005, price=4136.181, entry=ENTRY,
                stop=4110.43, pnl_money=0.5)
    events = tm.lifecycle_events(4242)
    assert [e["kind"] for e in events] == ["open", "sl", "partial"]
    assert events[1]["level"] == 4131.52
    assert events[1]["reason"] == "profit_lock"
    assert events[2]["price"] == 4136.181
    assert tm.lifecycle_events(999999) == []


# ------------------------------------------------------- study dicts
def test_zones_to_study_dicts_carry_source_timeframe():
    z = SimpleNamespace(level=4146.5, tier="MAJOR", side="resistance",
                        tf="H4",
                        swing_ts=pd.Timestamp("2026-11-02", tz="UTC"))
    out = zones_to_study_dicts([z], entry=4124.0, signal_zone=None)
    assert out[0]["tf"] == "H4"
    assert out[0]["swing_ts"].startswith("2026-11-02")


# ------------------------------------------------------- role renders
def _roles_pngs(**over):
    frames = {"M5": _frame(30, "5min", "2026-11-05 10:40",
                           base=4128.0, drift=0.35),
              "H1": _frame(50, "h", "2026-11-03 20:00",
                           base=4090.0, drift=0.9)}
    kw = dict(frames=frames, lifecycle=_lifecycle(), context=None,
              entry_study=None, instrument="XAUUSDm",
              playbook="hot_scalp", score_line="Score 79",
              entry_clock="M5")
    kw.update(over)
    return render_mt5_roles(**kw)


def test_render_mt5_roles_is_three_real_pngs():
    pngs = _roles_pngs()
    assert len(pngs) == 3
    for png in pngs:
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        arr = plt.imread(io.BytesIO(png))
        # PIN MOVED (TV renderer pass): the canvas size now
        # depends on the renderer rung — portrait 1242x1500 for
        # the Lightweight Charts rung, the shipped 1600x700
        # landscape for the matplotlib fallback. Both geometries
        # are pinned deterministically in test_mt5_chart_tv.py
        # (fallback forced / TV rung skip-gated); here the trio
        # contract is rung-agnostic.
        assert (arr.shape[1], arr.shape[0]) in ((1242, 1500),
                                                (1600, 700))
        assert float(arr.std()) > 0.01


def test_render_mt5_roles_missing_frame_is_labeled_not_invented():
    pngs = _roles_pngs(frames={"H1": _frame(50, "h", "2026-11-03",
                                            base=4090.0)})
    assert len(pngs) == 3
    assert all(p[:8] == b"\x89PNG\r\n\x1a\n" for p in pngs)


def test_role_trade_render_changes_with_the_lifecycle():
    planned = _roles_pngs(lifecycle=_lifecycle(
        state="planned", fill_price=None, events=[], partials=[],
        exit_price=None, exit_reason="", initial_broker_stop=None))
    closed = _roles_pngs()
    assert planned[0] != closed[0]


# ------------------------------------------------------- HTML view
def _payload(html):
    start = html.index('<script id="signal-data" '
                       'type="application/json">')
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def _html(**kw):
    params = {
        "frames": {"H1": _frame(60, "h", "2026-11-03"),
                   "M5": _frame(60, "5min", "2026-11-05 08:00")},
        "entry": ENTRY, "stop": STOP, "targets": list(LADDER),
        "direction": 1, "playbook": "hot_scalp",
        "verdict": {"entry_clock": "M5", "trigger": "FVG retest"},
        "zone": {"top": 4135.883, "bottom": 4132.01, "kind": "FVG"},
        "nearby_zones": [], "fib_levels": [], "fib_range": None,
        "ote_band": None, "fib_info": None,
        "alert_id": "MT5-hot_scalp-2008",
    }
    params.update(kw)
    return build_signal_view_html(**params)


def test_html_view_caps_the_ladder_and_names_the_total():
    payload = _payload(_html())
    assert payload["meta"]["targets_total"] == 24
    assert len(payload["meta"]["targets"]) == 4


def test_html_view_carries_the_lifecycle():
    payload = _payload(_html(lifecycle=_lifecycle()))
    lc = payload["lifecycle"]
    assert lc["ticket"] == 540274935
    assert lc["stop_events"][0]["level"] == 4135.92
    assert lc["exit"] == 4135.797
    analysis = " ".join(payload["tfs"]["M5"]["analysis"])
    assert "SL → 4,135.9 (profit_lock)" in analysis
    assert "Exit 4,135.8" in analysis
