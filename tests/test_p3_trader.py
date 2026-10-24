"""P3 — THE FULL TRADER: scenario/day/posture/news/SMT/clock/
management/open-space coverage (spec v3 §12).

Offline throughout: pure modules get synthetic frames; runner
tests reuse the confluence suite's harness shape (fake MT5 feed
+ fake executor, tmp journal).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.execution.tick_manager import TickConfig, TickManager
from goldenswing.execution.executor import OrderRequest
from goldenswing.live import day_context as dc
from goldenswing.live import posture as posture_mod
from goldenswing.live import smt as smt_mod
from goldenswing.live.management_profiles import (
    drift_triggered, progress_due, profile_for, tp_ladder)
from goldenswing.live.news import NewsCalendar, NewsEvent
from goldenswing.live.posture import (LadderState, compose_posture,
                                      loss_ladder, session_posture)


# ------------------------------------------------------------ helpers

def _frame(rows, start="2026-01-05 00:00", freq="h"):
    """rows: list of (open, high, low, close)."""
    idx = pd.date_range(start, periods=len(rows), freq=freq,
                        tz="UTC")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [100.0] * len(rows),
         "real_volume": [0.0] * len(rows),
         "spread": [30] * len(rows)}, index=idx)


def _bedrock(structure="clean_trend", direction="up",
             volatility="normal"):
    return SimpleNamespace(
        structure=structure, direction=direction,
        volatility=volatility, phase="distribution",
        session="london", confidence=0.8, route="core",
        label=lambda: f"{direction}/{structure}/{volatility}")


def _session(name="london", dead=False, scalar=1.0):
    return posture_mod.SessionPosture(
        name=name, dead=dead, size_scalar=scalar, note="test")


def _day(day_type=dc.DAY_TREND, adr_pct=40.0):
    return dc.DayContext(day_key="2026-01-06", day_type=day_type,
                         adr=10.0, adr_consumed_pct=adr_pct)


# ------------------------------------------------- A: bedrock posture

def test_posture_matrix_every_scenario_defined():
    """Each P2 matrix row maps to a DEFINED posture — none is a
    silent dead end (block_entries only where waiting IS the
    trade, always with the reason attached)."""
    ladder = LadderState()
    # Trend day / clean trend bedrock.
    p = compose_posture(bedrock=_bedrock(), day=_day(),
                        session=_session(), ladder=ladder)
    assert p.name == "trend_day" and not p.block_entries
    assert p.target_philosophy == "trend"
    # Rotation/range day.
    p = compose_posture(bedrock=_bedrock(structure="range",
                                         direction="flat"),
                        day=_day(dc.DAY_RANGE),
                        session=_session(), ladder=ladder)
    assert p.name == "rotation_day"
    assert "continuation" in p.shelved
    # Compression coil: entries shelved per family, resting
    # interest stays live.
    p = compose_posture(bedrock=_bedrock(structure="compression",
                                         direction="flat"),
                        day=_day(dc.DAY_NONTREND),
                        session=_session(), ladder=ladder)
    assert p.name == "coil" and p.resting_interest_live
    assert "continuation" in p.shelved and "scalp" in p.shelved
    # Choppy: reduced posture, not a veto.
    p = compose_posture(bedrock=_bedrock(structure="choppy",
                                         direction="flat"),
                        day=_day(dc.DAY_DEVELOPING),
                        session=_session(), ladder=ladder)
    assert p.name == "choppy_range" and p.size_scalar == 0.5
    assert not p.block_entries
    # Spike volatility: stand aside WITH the reason.
    p = compose_posture(bedrock=_bedrock(volatility="spike"),
                        day=_day(), session=_session(),
                        ladder=ladder)
    assert p.name == "spike_stand_aside" and p.block_entries
    assert "spike" in p.reason
    # News window: event posture (existing pause, composed).
    p = compose_posture(bedrock=_bedrock(), day=_day(),
                        session=_session(), ladder=ladder,
                        news_block=True)
    assert p.name == "event_spike" and p.block_entries
    # Dead session window.
    p = compose_posture(bedrock=_bedrock(), day=_day(),
                        session=_session("rollover", dead=True),
                        ladder=ladder)
    assert p.name == "dead_window" and p.block_entries
    assert p.resting_interest_live
    # Loss cooldown.
    p = compose_posture(
        bedrock=_bedrock(), day=_day(), session=_session(),
        ladder=loss_ladder(consecutive_losses=3,
                           style_counts={}, style_ceiling=12))
    assert p.name == "loss_cooldown" and p.block_entries
    # NY rejection: reversal posture, continuation shelved.
    p = compose_posture(bedrock=_bedrock(), day=_day(),
                        session=_session(), ladder=ladder,
                        checkpoint="rejected_up")
    assert p.name == "ny_rejection"
    assert p.target_philosophy == "value"
    # Nothing special: open.
    p = compose_posture(bedrock=_bedrock(structure="unclear",
                                         direction="flat"),
                        day=_day(dc.DAY_DEVELOPING),
                        session=_session(), ladder=ladder)
    assert p.name == "open" and not p.block_entries


def test_trend_day_shelves_only_counter_trend_range():
    p = compose_posture(bedrock=_bedrock(direction="up"),
                        day=_day(), session=_session(),
                        ladder=LadderState())
    # With-trend edge (long fade at support on an up day) hunts.
    assert p.shelved_reason("range", direction=1,
                            trend_direction="up") is None
    # Counter-trend edge (short at resistance) is shelved.
    r = p.shelved_reason("range", direction=-1,
                         trend_direction="up")
    assert r and "counter-trend" in r
    # Zone/continuation families are never direction-shelved here.
    assert p.shelved_reason("zone", direction=-1,
                            trend_direction="up") is None


def test_loss_ladder_graduated():
    l0 = loss_ladder(consecutive_losses=0, style_counts={},
                     style_ceiling=12)
    assert l0.traffic_cap is None and not l0.cooldown
    l2 = loss_ladder(consecutive_losses=2, style_counts={},
                     style_ceiling=12)
    assert l2.traffic_cap == "YELLOW" and l2.layer_cap_delta == -1
    assert not l2.cooldown
    l3 = loss_ladder(consecutive_losses=3, style_counts={},
                     style_ceiling=12)
    assert l3.cooldown
    churn = loss_ladder(consecutive_losses=0,
                        style_counts={"MICRO_SCALP": 12},
                        style_ceiling=12)
    assert "MICRO_SCALP" in churn.style_blocked


# ------------------------------------------------------- B: the day

def _two_week_frame(today_rows):
    """15 prior trading days of range-10 bars + today's rows.
    Days start 23:00 UTC (18:00 ET).  Prior days: 24 bars each,
    high 2010 / low 2000."""
    rows = []
    prior = [(2005.0, 2010.0, 2000.0, 2005.0)] * 24
    for _ in range(15):
        rows.extend(prior)
    rows.extend(today_rows)
    start = pd.Timestamp("2025-12-21 23:00", tz="UTC")
    idx = pd.date_range(start, periods=len(rows), freq="h")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [100.0] * len(rows),
         "real_volume": [0.0] * len(rows),
         "spread": [30] * len(rows)}, index=idx)


def test_day_context_math_trend_day():
    # Today: IB bar 2002-2008, then a one-sided drive to 2023.
    today = [(2005.0, 2008.0, 2002.0, 2006.0)]
    for i in range(1, 12):
        c = 2006.0 + 1.6 * i
        today.append((c - 1.0, c + 0.8, c - 1.2, c))
    ctx = dc.compute_day_context({"H1": _two_week_frame(today)})
    assert ctx.adr == pytest.approx(10.0, abs=0.6)
    assert ctx.ib_complete and ctx.ib_high == pytest.approx(2008.0)
    assert ctx.ib_low == pytest.approx(2002.0)
    # Day range ~21.8 on ADR 10 -> deep into extension territory.
    assert ctx.adr_consumed_pct > 150.0
    assert ctx.adr_state in ("extended", "exhausted")
    assert ctx.day_type == dc.DAY_TREND
    assert ctx.prior["high"] == pytest.approx(2010.0)
    assert ctx.prior["low"] == pytest.approx(2000.0)
    assert ctx.prior["poc"] > 0


def test_day_type_retypes_as_day_develops():
    # Early: balance inside a narrow band -> not a trend day.
    early = [(2005.0, 2008.0, 2002.0, 2005.0)] * 6
    ctx1 = dc.compute_day_context({"H1": _two_week_frame(early)})
    assert ctx1.day_type != dc.DAY_TREND
    # Later: the same balance open, then a one-sided late
    # extension -> the day RE-TYPES as it develops (a balance
    # day that breaks late is Dalton's double-distribution day:
    # the second distribution IS the trend leg).
    late = list(early)
    for i in range(1, 10):
        c = 2005.0 + 2.2 * i
        late.append((c - 1.0, c + 0.8, c - 1.2, c))
    ctx2 = dc.compute_day_context({"H1": _two_week_frame(late)})
    assert ctx2.day_type == dc.DAY_DOUBLE_DIST
    assert ctx2.day_type != ctx1.day_type


def test_classify_open_types():
    kw = dict(prior_vah=2010.0, prior_val=2000.0, ib_width=6.0)
    # Open-drive: opened above value, drove up, no dip.
    assert dc.classify_open(
        open_price=2012.0, first_closes=[2014.0, 2016.5],
        period_high=2017.0, period_low=2011.5, **kw) == dc.OPEN_DRIVE
    # Open-rejection-reverse: opened above value, rejected back in.
    assert dc.classify_open(
        open_price=2012.0, first_closes=[2011.0, 2008.0],
        period_high=2013.0, period_low=2007.0,
        **kw) == dc.OPEN_REJECTION_REVERSE
    # Open-auction inside value, no probe through the open.
    assert dc.classify_open(
        open_price=2005.0, first_closes=[2005.5, 2004.5],
        period_high=2006.5, period_low=2003.5,
        **kw) == dc.OPEN_AUCTION


def test_adr_de_rate_prices_never_vetoes():
    """S8: a 110%-consumed trend day DE-RATES the session share;
    it does not block — a strong read still clears the floor."""
    from goldenswing.live.asymmetry import (ScoreContext,
                                              score_asymmetry)
    # Clean rising structure with a fresh BOS + a graded zone at
    # price: every component strong.
    rows = []
    px = 2000.0
    for i in range(60):
        o = px
        c = px + (2.0 if i % 5 else -1.0)
        rows.append((o, max(o, c) + 0.7, min(o, c) - 0.7, c))
        px = c
    f = _frame(rows)
    zone = {"level": px, "top": px + 2.0, "bottom": px - 2.0,
            "proximal": px - 2.0, "tier": "MAJOR",
            "lifecycle": "fresh", "touches": 0, "status": "active",
            "score": 60.0}
    base = dict(direction=1, frame=f, price=px, zone=zone,
                verdict={"bias_d1": 1, "bias_h4": 1}, atr=6.0)
    fresh = score_asymmetry(ScoreContext(
        adr_consumed_pct=40.0, adr_applies=True, **base))
    spent = score_asymmetry(ScoreContext(
        adr_consumed_pct=110.0, adr_applies=True, **base))
    # De-rated: the session share shrinks by the curve's factor.
    assert spent.component("session") < fresh.component("session")
    # Priced, never vetoed: ONLY the session share moved — the
    # total drops by that delta alone (no component zeroed, no
    # hidden block), so a read that clears its floor at 40% still
    # clears it at 110% whenever its margin exceeds ~2 points.
    delta = fresh.component("session") - spent.component("session")
    assert spent.total == pytest.approx(fresh.total - delta,
                                         abs=0.05)
    assert spent.total > 0.9 * fresh.total
    # Family scoping: a non-continuation read is not de-rated at
    # all (the gauge prices continuation fuel, not reversals).
    other = score_asymmetry(ScoreContext(
        adr_consumed_pct=110.0, adr_applies=False, **base))
    assert other.component("session") == \
        pytest.approx(fresh.component("session"))


# ------------------------------------------------- C: session reads

def test_session_posture_rollover_needs_spread_proof():
    # Tuesday 17:30 ET (= 22:30 UTC in January).
    tue = datetime(2026, 1, 6, 22, 30, tzinfo=timezone.utc)
    dead = session_posture(tue, spread_ratio=2.5)
    assert dead.dead and dead.name == "rollover"
    # Same clock, normal spreads: NOT a dead window — the clock
    # alone never gates (S3).
    live = session_posture(tue, spread_ratio=1.1)
    assert not live.dead
    # No spread read at all: still not dead on the clock alone.
    assert not session_posture(tue, spread_ratio=None).dead


def test_session_posture_scalars():
    # Wednesday 05:00 UTC = 00:00 ET (asia).
    asia = session_posture(datetime(2026, 1, 7, 5, 0,
                                    tzinfo=timezone.utc),
                           spread_ratio=1.0)
    assert asia.name == "asia" and asia.size_scalar == 0.85


def test_ny_checkpoint_accept_and_reject():
    day = dc.DayContext(
        day_key="2026-01-06",
        prior={"high": 2010.0, "low": 2000.0},
        session_levels={"london_high": 2014.0,
                        "london_low": 2003.0})
    # London swept the prior high; NY closes back inside -> reject.
    assert posture_mod.ny_checkpoint(
        day, ny_close=2008.0, ny_low=2006.0,
        ny_high=2013.0) == "rejected_up"
    # NY holds beyond the swept level -> accepted.
    assert posture_mod.ny_checkpoint(
        day, ny_close=2015.0, ny_low=2011.0,
        ny_high=2016.0) == "accepted_up"
    # London never left the prior range -> nothing to judge.
    day2 = dc.DayContext(
        day_key="2026-01-06",
        prior={"high": 2010.0, "low": 2000.0},
        session_levels={"london_high": 2009.0,
                        "london_low": 2001.0})
    assert posture_mod.ny_checkpoint(
        day2, ny_close=2005.0, ny_low=2003.0,
        ny_high=2007.0) is None


# --------------------------------------------------------- D: news

def _calendar(events):
    cal = NewsCalendar()
    cal._cached = list(events)
    import time as _time
    cal._cached_at = _time.time()
    return cal


def test_news_next_event_and_reaction_state():
    now = datetime(2026, 1, 6, 13, 0, tzinfo=timezone.utc)
    cpi = NewsEvent("CPI y/y", now + timedelta(minutes=25))
    cal = _calendar([cpi])
    ev, mins = cal.next_event(now)
    # Minutes are to the WINDOW OPEN (the print minus the pause
    # lead), not to the print itself.
    assert ev.title == "CPI y/y" and 0.0 < mins < 25.0
    # Inside the closed window (13:20-13:50 for a 13:30 print):
    # no reaction state yet (that's the pause, not the reaction).
    assert cal.reaction_state(now + timedelta(minutes=35)) is None
    # At the EXTREME window's close: the reaction state opens
    # (titles match the calendar's EXTREME_TITLES keys).
    print_t = now + timedelta(minutes=30)
    cal2 = _calendar([NewsEvent("NFP", print_t)])
    after = print_t + timedelta(minutes=45)
    st = cal2.reaction_state(after)
    assert st is not None and st.title == "NFP"
    # An hour later: state over.
    assert cal2.reaction_state(after + timedelta(
        minutes=60)) is None


def test_preprint_action_matrix():
    from goldenswing.live.news import preprint_action
    # Swing with a stop beyond 2x the event range: exempt.
    assert preprint_action(style_family="swing", stop_distance=30.0,
                           event_range=12.0) == "exempt"
    # Swing inside the blast radius: secure.
    assert preprint_action(style_family="swing", stop_distance=15.0,
                           event_range=12.0) == "secure"
    # Intraday is never exempt.
    assert preprint_action(style_family="intraday",
                           stop_distance=99.0,
                           event_range=1.0) == "secure"


# ---------------------------------------------------------- E: SMT

def _smt_frames():
    # Built from explicit closes so the swing geometry is exact
    # (ties are never swings).  Gold: confirmed swing high 2060
    # at bar 29, pullback, grind UNDER it, bar 41 sweeps to 2063
    # and closes back below.  DXY: confirmed swing low 95.0 at
    # bar 29; it never breaks that low afterwards.
    gold_closes = [2000.0 + 2.0 * i for i in range(30)]     # -> 2058
    gold_closes += [2056.0, 2054.0, 2052.0, 2050.0, 2049.0,
                    2048.0]                                  # pullback
    gold_closes += [2050.0, 2052.0, 2053.0, 2055.0, 2056.0]  # grind
    gold_rows = []
    prev = gold_closes[0] - 2.0
    for c in gold_closes:
        gold_rows.append((prev, c + 2.0 if c == 2058.0 else
                          c + 1.0, c - 1.0, c))
        prev = c
    gold_rows.append((2056.0, 2063.0, 2050.0, 2055.0))       # sweep
    gold = _frame(gold_rows)
    dxy_closes = [100.0 - (4.8 / 29.0) * i for i in range(30)]
    # Bounce with a dip (a confirmed swing high forms mid-bounce)
    # and lows that never threaten 95.0.
    dxy_closes += [95.4, 95.55, 95.7, 95.45, 95.5, 95.55, 95.6,
                   95.62, 95.64, 95.66, 95.68]
    dxy_rows = []
    prev = dxy_closes[0] + 0.2
    for c in dxy_closes:
        low = 95.0 if abs(c - 95.2) < 1e-9 else c - 0.15
        dxy_rows.append((prev, c + 0.05, low, c))
        prev = c
    dxy_rows.append((95.68, 96.35, 95.4, 96.3))              # refuses
    dxy = _frame(dxy_rows)
    return gold, dxy


def test_smt_bearish_detected_and_mirror():
    gold, dxy = _smt_frames()
    read = smt_mod.detect_smt(gold, dxy, corr=-0.8)
    assert read.available and read.event is not None
    assert read.event.kind == "smt_bearish"
    # Mirror: invert both frames' direction of travel.
    gold2 = gold.copy()
    dxy2 = dxy.copy()
    for df in (gold2, dxy2):
        o, h, l, c = (df["open"].copy(), df["high"].copy(),
                      df["low"].copy(), df["close"].copy())
        mid = 2000.0 if df is gold2 else 100.0
        df["open"] = 2 * mid - o
        df["high"] = 2 * mid - l
        df["low"] = 2 * mid - h
        df["close"] = 2 * mid - c
    read2 = smt_mod.detect_smt(gold2, dxy2, corr=-0.8)
    assert read2.available and read2.event is not None
    assert read2.event.kind == "smt_bullish"


def test_smt_unavailable_when_correlation_broken():
    gold, dxy = _smt_frames()
    read = smt_mod.detect_smt(gold, dxy, corr=-0.1)
    assert not read.available and read.event is None
    assert "correlation" in read.reason
    read = smt_mod.detect_smt(gold, dxy, corr=None)
    assert not read.available


# ------------------------------------------- F: management as trader

def _tm(**cfgkw):
    # R-trail pushed out of the way (trigger 99R): these tests
    # isolate the breakeven rule — where the trail CAN fire it is
    # tighter than BE and correctly wins the tick.
    from goldenswing.backtest.harness import ExitConfig
    cfg = TickConfig(partial_tiers=(), profit_lock_enabled=False,
                     chandelier_enabled=False, **cfgkw)
    return TickManager(tick=cfg,
                       exits=ExitConfig(trail_trigger_r=99.0))


def _req(direction=1):
    return OrderRequest(
        symbol="XAUUSD", direction=direction, volume=1.0,
        stop_loss=1990.0 if direction == 1 else 2010.0,
        entry_ref=2000.0, targets=[2030.0], atr=10.0, bar_index=5,
        alert_id="T1")


def test_be_earned_not_touched():
    """F13/S9: +1.2R on price alone does NOT move the stop in
    earned mode; a funding/structure earning event DOES."""
    mgr = _tm(breakeven_mode="earned", breakeven_backstop_r=2.0)
    mgr.open(ticket=101, request=_req(), fill_price=2000.0,
             entry_bar=6)
    # +1.2R (risk 12.5 -> 2015): nothing.
    assert mgr.on_tick(101, 2015.0) == []
    assert mgr.mark_be_earned(101, "funding:sweep_retest")
    acts = mgr.on_tick(101, 2015.0)
    assert [a.reason for a in acts] == ["breakeven"]
    assert acts[0].level == pytest.approx(2000.0)
    assert mgr.be_state(101)["by"] == "funding:sweep_retest"


def test_be_backstop_without_earning():
    mgr = _tm(breakeven_mode="earned", breakeven_backstop_r=2.0)
    mgr.open(ticket=101, request=_req(), fill_price=2000.0,
             entry_bar=6)
    # +2.1R with NO earning event: the backstop still protects.
    acts = mgr.on_tick(101, 2026.25)
    assert [a.reason for a in acts] == ["breakeven"]
    mgr.apply(101, acts[0])
    assert mgr.be_state(101)["by"] == "backstop"


def test_be_plan_mode_earned():
    mgr = _tm()  # config default stays touch (legacy)
    mgr.open(ticket=101, request=_req(), fill_price=2000.0,
             entry_bar=6)
    mgr.attach_narrative_plan(101, {
        "style": "clean", "be_r": 1.0, "be_mode": "earned",
        "be_backstop_r": 2.0, "trail_mode": "structure",
        "trail_atr_mult": 1.5, "partials": [],
        "time_stop_h1": None})
    assert mgr.on_tick(101, 2015.0) == []       # +1.2R, unearned
    assert mgr.mark_be_earned(101, "funding:displacement")
    acts = mgr.on_tick(101, 2015.0)
    assert any(a.reason == "breakeven" for a in acts)


def test_progress_and_drift_pure():
    trend = profile_for(style="intraday", philosophy="trend")
    # 8 H1 bars (32 trigger bars at M15) under +0.5R: due.
    assert progress_due(trend, elapsed_h1_bars=9, best_r=0.3)
    assert not progress_due(trend, elapsed_h1_bars=9, best_r=0.8)
    assert not progress_due(trend, elapsed_h1_bars=3, best_r=0.3)
    # Drift: entry floor 65, margin 10 -> decay below 55 triggers,
    # unless price already cut the invalidation (hard lines own it).
    assert drift_triggered(trend, entry_floor=65.0,
                           current_total=50.0,
                           through_invalidation=False)
    assert not drift_triggered(trend, entry_floor=65.0,
                               current_total=50.0,
                               through_invalidation=True)
    assert not drift_triggered(trend, entry_floor=65.0,
                               current_total=58.0,
                               through_invalidation=False)


def test_tp_ladder_from_zones():
    zones = [{"proximal": 2020.0, "level": 2022.0},
             {"proximal": 2050.0, "level": 2052.0}]
    lad = tp_ladder(entry=2000.0, stop=1990.0, direction=1,
                    zones_ahead=zones)
    # Intermediate magnet banks 0.25R early; final opposing zone
    # is the runner's target, also 1R early.
    assert lad["tp1"] == pytest.approx(2020.0 - 2.5)
    assert lad["tp2"] == pytest.approx(2050.0 - 10.0)
    assert lad["tp1_pct"] == pytest.approx(0.5)
    assert lad["tp2_pct"] == pytest.approx(0.5)
    # No zones ahead: plain R rungs.
    lad = tp_ladder(entry=2000.0, stop=1990.0, direction=1,
                    zones_ahead=[])
    assert lad["tp1"] == pytest.approx(2010.0)
    assert lad["tp3"] == pytest.approx(2030.0)


# -------------------------------------------------- I: clock float

def test_clock_float_rules():
    from goldenswing.live.posture import clock_for
    # Vol spike promotes the scalp funding clock M5 -> H1 path
    # (funding_tf steps up the ladder) and the trigger with it.
    c = clock_for(style="scalp", vol_regime="HIGH", clarity={})
    assert c.funding_tf in ("M15", "H1") and not c.stand_aside
    # Dead-calm tape: the scalp stands aside (reasoned), other
    # styles demote the funding clock one step FINER (a slow tape
    # can be read closely — S12's documented direction).
    c = clock_for(style="scalp", vol_regime="LOW", clarity={})
    assert c.stand_aside and "volatility" in c.reason
    c = clock_for(style="intraday", vol_regime="LOW", clarity={})
    assert not c.stand_aside and c.funding_tf == "M5"
    # Defaults hold at NORMAL.
    c = clock_for(style="intraday", vol_regime="NORMAL",
                  clarity={})
    assert (c.score_tf, c.funding_tf, c.trigger_tf) == \
        ("H1", "M15", "M15")
    c = clock_for(style="qm_d1", vol_regime="NORMAL", clarity={})
    assert c.score_tf == "D1"


# ============================================================ runner

from goldenswing.execution.executor import OrderResult, Position
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.live.funding import FundingReport
from goldenswing.live.smt import SmtEvent, SmtRead


class _FakeExecutor:
    def __init__(self):
        self.orders = []
        self.sls = []
        self.closes = []
        self._ticket = 9000
        self._positions = []

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return list(self._positions)

    def place_order(self, request):
        self._ticket += 1
        self.orders.append({"ticket": self._ticket,
                            "volume": request.volume,
                            "kind": request.order_kind,
                            "alert_id": request.alert_id,
                            "targets": list(request.targets or [])})
        self._positions.append(Position(
            ticket=self._ticket, symbol=request.symbol,
            direction=request.direction, volume=request.volume,
            entry_price=request.entry_ref or request.order_price,
            sl=request.stop_loss, tp=request.take_profit,
            magic=request.magic, alert_id=request.alert_id))
        return OrderResult(True, ticket=self._ticket,
                           price=request.entry_ref
                           or request.order_price,
                           volume=request.volume)

    def modify_sl(self, ticket, sl, tp=None):
        self.sls.append((ticket, sl))
        return OrderResult(True, ticket=ticket)

    def cancel_pending(self, ticket):
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        self.closes.append((ticket, volume))
        return OrderResult(True, ticket=ticket)


def _feed():
    return SimpleNamespace(
        account_state=lambda: {"equity": 10000.0,
                               "balance": 10000.0,
                               "free_margin": 10000.0},
        order_margin=lambda *, volume, price, direction:
            float(volume) * 100.0 * float(price) / 500.0)


def _runner(tmp_path, **cfg_kw):
    news_calendar = cfg_kw.pop("news_calendar", None)
    cfg_kw = {"news_pause_enabled": False, **cfg_kw}
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        **cfg_kw)
    runner = MT5Runner(cfg, feed=_feed(), executor=_FakeExecutor(),
                       news_calendar=news_calendar)
    runner._send_alert = lambda *a, **k: None
    runner._build_chart = lambda *a, **k: (None, None)
    # Pinned Wednesday tick default: with no tick the order path
    # falls back to the wall clock, so order-path tests fail on
    # Fridays after 20:00 UTC (Friday-close gate).  Tests that set
    # their own _last_tick keep it; this is only the default.
    runner._last_tick = {"mid": 2000.0, "bid": 1999.8, "ask": 2000.2,
                         "time": "2026-10-14 12:00:00"}
    return runner


def _flat_h1(n=30, price=2000.0):
    return _frame([(price, price + 1.0, price - 1.0, price + 0.5)]
                  * n)


class _StubPB:
    session_scope = "any"
    hunting_ground = SimpleNamespace(excluded=[], bonuses={})

    def __init__(self, name, sig):
        self.name = name
        self._sig = sig

    def live_signal(self, states, t, df=None):
        return [self._sig]

    def check_live_signal(self, signal, ctx):
        return True, ""


class _StubEngine:
    def gate_live_signal(self, pb, sig, ctx, hierarchy_verdict=None,
                         spread_px=None):
        return True, ""

    def score_live_signal(self, pb, sig, ctx, hierarchy_verdict=None):
        return 85.0, ["stub-bonus"]


class _FakeScore:
    flipped = False

    def __init__(self, total=70.0, structure=25.0, efficiency=15.0):
        self.total = total
        self._c = {"structure": structure,
                   "efficiency": efficiency, "location": 15.0,
                   "pullback": 10.0, "session": 5.0, "macro": 0.0}

    def component(self, name):
        return self._c.get(name, 0.0)

    def breakdown(self):
        return {"total": self.total}


def _sig(direction=1, entry=2000.0, stop=1990.0, index=29):
    return SimpleNamespace(
        direction=direction, score=80.0, entry_ref=entry,
        invalidation=stop, targets=[2030.0], atr=10.0,
        index=index, factors=["stub"])


def _drive_one(runner, pb, h1, verdict=None, zone_d=None,
               funding=None):
    from goldenswing.playbooks.base import EngineContext
    ctx = EngineContext(df=h1, states=[], years=0.0,
                        instrument="XAUUSD")
    stats = {"signals": 0, "gated": 0}
    runner._run_one_playbook(
        pb, sig_ctx=(ctx, _StubEngine()), h1=h1, t=len(h1) - 1,
        verdict=verdict or {"bias_d1": 1, "bias_h4": 1,
                            "regime": "trend", "direction": 1,
                            "trigger": "tick-cross",
                            "entry_clock": "M15",
                            "amd_phase": "accumulation",
                            "qm": {"status": "none"}, "volume": {}},
        stats=stats, zone_d=zone_d, funding=funding)
    return stats


def _signal_records(runner):
    return [r for r in runner.journal.records()
            if r.get("type") == "signal"]


# ------------------------------------------------- G: open space

def test_open_space_path_b_accepted_with_floor_applied(tmp_path):
    runner = _runner(tmp_path, execute_playbooks=("fvg_midline",))
    h1 = _flat_h1()
    runner._stack = SimpleNamespace(frames={"H1": h1})
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    _drive_one(runner, _StubPB("fvg_midline", _sig()), h1)
    recs = _signal_records(runner)
    assert recs and recs[-1]["decision"] == "accepted"
    # The record carries the P3 scenario context (J).
    assert recs[-1].get("context") is not None
    assert len(runner.executor.orders) == 1


def test_open_space_path_b_combined_rule_binds(tmp_path):
    """Total 70 >= the 65 floor is NOT enough: Structure +
    Efficiency 20 < 35 refuses (Path B's combined rule) — the
    stretched-pivot continuation the desk will not chase."""
    runner = _runner(tmp_path, execute_playbooks=("fvg_midline",))
    h1 = _flat_h1()
    runner._stack = SimpleNamespace(frames={"H1": h1})
    runner._score_for = lambda **kw: _FakeScore(70.0, 10.0, 10.0)
    _drive_one(runner, _StubPB("fvg_midline", _sig()), h1)
    recs = _signal_records(runner)
    assert recs and recs[-1]["decision"] == "below_floor"
    assert "INTRADAY_CONTINUATION" in recs[-1]["reason"]
    assert runner.executor.orders == []


def test_counter_journey_refused_with_reason(tmp_path):
    runner = _runner(tmp_path, execute_playbooks=("fvg_midline",))
    h1 = _flat_h1()
    runner._stack = SimpleNamespace(frames={"H1": h1})
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    # Long at 2000: a support 7 away BEHIND (the nearer draw),
    # the magnet 60 ahead.  No nested faster-TF zone at price.
    runner._zone_dicts = lambda: [
        {"level": 1993.0, "proximal": 1993.0, "tf": "H4",
         "tier": "MAJOR", "status": "active", "side": "support"},
        {"level": 2060.0, "proximal": 2060.0, "tf": "H4",
         "tier": "MAJOR", "status": "active",
         "side": "resistance"}]
    _drive_one(runner, _StubPB("fvg_midline", _sig()), h1)
    recs = _signal_records(runner)
    assert recs and recs[-1]["decision"] == "gated"
    assert recs[-1]["reason"].startswith("counter_journey")
    assert runner.executor.orders == []


def test_counter_journey_allowed_at_nested_zone_with_room(tmp_path):
    runner = _runner(tmp_path, execute_playbooks=("fvg_midline",))
    h1 = _flat_h1()
    runner._stack = SimpleNamespace(frames={"H1": h1})
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    # Same counter geometry, but an M15 zone holds price AND the
    # room to the H4 magnet is 6R (>= the 3R rule): allowed.
    runner._zone_dicts = lambda: [
        {"level": 1993.0, "proximal": 1993.0, "tf": "H4",
         "tier": "MAJOR", "status": "active", "side": "support"},
        {"level": 2000.0, "proximal": 2001.0, "tf": "M15",
         "tier": "MINOR", "bottom": 1998.0, "top": 2003.0,
         "status": "active", "side": "support"},
        {"level": 2060.0, "proximal": 2060.0, "tf": "H4",
         "tier": "MAJOR", "status": "active",
         "side": "resistance"}]
    _drive_one(runner, _StubPB("fvg_midline", _sig()), h1)
    recs = _signal_records(runner)
    assert recs and recs[-1]["decision"] == "accepted"
    assert len(runner.executor.orders) == 1


def test_smt_funding_leg_reaches_the_order(tmp_path):
    runner = _runner(tmp_path, execute_playbooks=("v_reversal",))
    h1 = _flat_h1()
    runner._stack = SimpleNamespace(frames={"H1": h1})
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    runner._smt_read = SmtRead(
        available=True, corr=-0.8,
        event=SmtEvent(direction=1, kind="smt_bullish",
                       level=1995.0, detail="test", bars_ago=1))
    captured = {}
    orig = runner._place_live_order

    def _capture(pb, sig, stats, **kw):
        captured.update(kw)
        return orig(pb, sig, stats, **kw)

    runner._place_live_order = _capture
    zone_d = {"level": 2000.0, "top": 2005.0, "bottom": 1995.0,
              "proximal": 1995.0, "tier": "MAJOR",
              "lifecycle": "fresh", "touches": 0, "status": "active",
              "score": 60.0, "tf": "H4"}
    _drive_one(runner, _StubPB("v_reversal", _sig()), h1,
               zone_d=zone_d, funding=FundingReport())
    funding = captured.get("funding")
    assert funding is not None
    # The funding event carries the SMT kind (direction-bearing).
    assert "smt_bullish" in [e.kind for e in funding.events]


# ------------------------------------------------------- G: qm_d1

class _FakeQMD1:
    name = "qm_d1"
    session_scope = "any"
    hunting_ground = SimpleNamespace(excluded=[], bonuses={})

    def candidates(self, ctx):
        # Touch on the frame's LAST bar — the live-close shape
        # (Phase 2 candidate lifecycle: a touch buried inside
        # the frame is expired provenance and never routes).
        last = len(ctx.df) - 1
        sig = SimpleNamespace(
            direction=-1, index=last, score=70.0, entry_ref=4100.0,
            invalidation=4130.0, targets=[4040.0], atr=25.0,
            factors=["qm_d1"], kind="qm",
            live_context={"qm_level": 4100.0, "head": 4125.0,
                          "neckline": 4040.0, "tags": []})
        yield SimpleNamespace(
            signal_index=last, direction=-1,
            outcome=SimpleNamespace(signal=sig, r_net=1.2,
                                    exit_reason="target"))


def test_qm_d1_real_order_path_on_d1_clock(tmp_path, monkeypatch):
    import goldenswing.execution.mt5_runner as mr
    monkeypatch.setattr(mr, "QMD1Playbook", _FakeQMD1)
    runner = _runner(tmp_path)
    d1 = _frame([(4100.0, 4110.0, 4090.0, 4105.0)] * 40,
                  freq="1d")
    h1 = _flat_h1(30, 4100.0)
    runner._stack = SimpleNamespace(frames={"H1": h1, "D1": d1})
    runner._last_tick = {"mid": 4100.0, "bid": 4099.8,
                         "ask": 4100.2, "spread": 30,
                         # Pinned Wednesday: a wall-clock tick makes
                         # this fail on Fridays after 20:00 UTC.
                         "time": "2026-10-14 12:00:00"}
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    runner._get_engine = lambda: _StubEngine()
    out = runner._run_qm_d1(d1, {})
    assert len(out) == 1
    assert len(runner.executor.orders) == 1
    recs = [r for r in _signal_records(runner)
            if r.get("playbook") == "qm_d1"]
    assert recs and recs[-1]["decision"] == "accepted"
    assert recs[-1]["trigger_source"] == "qm_d1"
    # Dedupe: the same touch never re-evaluates.
    assert runner._run_qm_d1(d1, {}) == []
    assert len(runner.executor.orders) == 1


def test_qm_d1_kill_switch(tmp_path, monkeypatch):
    import goldenswing.execution.mt5_runner as mr
    monkeypatch.setattr(mr, "QMD1Playbook", _FakeQMD1)
    runner = _runner(tmp_path, disabled_playbooks=("qm_d1",))
    d1 = _frame([(4100.0, 4110.0, 4090.0, 4105.0)] * 40,
                freq="1d")
    assert runner._run_qm_d1(d1, {}) == []
    assert runner.executor.orders == []


# ------------------------------------------------- I: watch slots

def test_pullback_watch_multi_slot(tmp_path):
    runner = _runner(tmp_path)
    h1 = _flat_h1()
    runner._stack = SimpleNamespace(frames={"H1": h1})
    pb_a = _StubPB("fvg_midline", _sig())
    pb_b = _StubPB("order_block", _sig())
    zone_a = {"level": 2000.0, "bottom": 1995.0, "top": 2005.0,
              "tf": "H4", "atr": 10.0}
    zone_b = {"level": 2100.0, "bottom": 2095.0, "top": 2105.0,
              "tf": "H4", "atr": 10.0}
    score = _FakeScore(70.0)
    runner._enter_pullback_watch(pb_a, _sig(), {}, zone_a, score,
                                 None)
    runner._enter_pullback_watch(pb_b, _sig(), {}, zone_b, score,
                                 None)
    assert len(runner._pullback_watches) == 2
    # Re-print of A refreshes its slot instead of stacking.
    runner._enter_pullback_watch(pb_a, _sig(), {}, zone_a, score,
                                 None)
    assert len(runner._pullback_watches) == 2
    # Arrival inside B clears ONLY B.
    runner._maintain_pullback_watch(2100.0)
    assert len(runner._pullback_watches) == 1
    assert runner._pullback_watches[0]["playbook"] == "fvg_midline"


# ---------------------------------------------------- C/D: runner

def test_day_line_content(tmp_path):
    runner = _runner(tmp_path)
    runner._day_ctx = dc.DayContext(
        day_key="2026-01-06", day_type=dc.DAY_TREND, adr=12.0,
        adr_days=14, adr_consumed_pct=62.0, adr_state="working",
        ib_high=2010.0, ib_low=2002.0, ib_complete=True)
    runner._posture = posture_mod.Posture(
        name="trend_day", reason="test")
    line = runner.day_line()
    assert line.startswith("day: ")
    assert "trend" in line and "62%" in line
    assert "posture trend_day" in line


def test_log_signal_context_round_trip(tmp_path):
    from goldenswing.execution.learning import TradeJournal
    from goldenswing.execution.signal_log import log_signal
    j = TradeJournal(tmp_path / "j.jsonl")
    log_signal(j, signal_id="x", playbook="pb", direction=1,
               score=1.0, decision="management",
               reason="be_earned: funding:sweep",
               context={"regime": "up/clean_trend/normal",
                          "posture": "trend_day",
                          "day_type": "trend", "adr_pct": 62.0})
    rec = j.records()[-1]
    assert rec["context"]["posture"] == "trend_day"
    assert rec["context"]["adr_pct"] == 62.0
    assert rec["decision"] == "management"


def _open_position(runner, *, style_key="intraday", stop=1990.0):
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.10,
                       stop_loss=stop, entry_ref=2000.0,
                       targets=[2030.0], atr=10.0, bar_index=5,
                       alert_id="MT5-test-1")
    runner.tick_manager.open(ticket=4242, request=req,
                             fill_price=2000.0, entry_bar=5)
    runner._signal_ctx["MT5-test-1"] = {
        "playbook": "fvg_midline", "style_key": style_key,
        "style_bucket": "INTRADAY", "family": "continuation",
        "floor": 65.0, "path": "INTRADAY_CONTINUATION",
        "entry_total": 72.0, "zone_d": None, "clock_tf": "M15"}
    ctx = runner._register_position_ctx(
        4242, alert_id="MT5-test-1", entry=2000.0, stop=stop,
        direction=1)
    pos = runner.tick_manager.position(4242)
    return pos, ctx


def test_preprint_secures_intraday_position(tmp_path):
    now = datetime(2026, 1, 6, 13, 0, tzinfo=timezone.utc)
    cal = _calendar([NewsEvent("NFP", now + timedelta(minutes=20))])
    runner = _runner(tmp_path, news_pause_enabled=True,
                     news_calendar=cal)
    runner._last_tick = {"mid": 2000.0, "bid": 1999.8,
                         "ask": 2000.2, "spread": 30, "time": now}
    pos, ctx = _open_position(runner)
    acted = runner._preprint_for_position(
        4242, pos, ctx, 2000.0, 5.0, {"reconciled_closed": 0})
    assert acted
    # Stop secured to breakeven (the entry), per the default
    # doctrine action.
    assert (4242, 2000.0) in runner.executor.sls
    notes = [r.get("text", "") for r in runner.journal.records()]
    assert any("pre-print" in n for n in notes)


def test_preprint_swing_exempt_beyond_event_range(tmp_path):
    now = datetime(2026, 1, 6, 13, 0, tzinfo=timezone.utc)
    cal = _calendar([NewsEvent("NFP", now + timedelta(minutes=20))])
    runner = _runner(tmp_path, news_pause_enabled=True,
                     news_calendar=cal)
    runner._last_tick = {"mid": 2000.0, "bid": 1999.8,
                         "ask": 2000.2, "spread": 30, "time": now}
    # Swing stop 75 away vs a 15-price event range (3 x ATR 5):
    # beyond 2x the print's range — securing would be theatre.
    pos, ctx = _open_position(runner, style_key="swing",
                              stop=1925.0)
    acted = runner._preprint_for_position(
        4242, pos, ctx, 2000.0, 5.0, {"reconciled_closed": 0})
    assert acted  # journaled either way
    assert runner.executor.sls == []
    notes = [r.get("text", "") for r in runner.journal.records()]
    assert any("EXEMPT" in n for n in notes)


# ------------------------------------------------- I: clock runner

def test_clock_float_runner_high_vol_promotes_funding(tmp_path):
    runner = _runner(tmp_path)
    # H4: 50 quiet bars (range 2), then 10 hot bars (range 24):
    # the live ATR regime reads HIGH against its own baseline.
    rows = [(2000.0, 2001.0, 1999.0, 2000.0)] * 50
    rows += [(2000.0, 2012.0, 1988.0, 2006.0)] * 10
    h4 = _frame(rows, freq="4h")
    frames = {"H4": h4, "H1": _flat_h1(), "M15": _flat_h1(),
              "M5": _flat_h1()}
    clock = runner._clock_for_signal("intraday", frames)
    assert clock.funding_tf == "H1"  # M15 promoted one step
    assert "HIGH" in clock.reason or "EXTREME" in clock.reason
