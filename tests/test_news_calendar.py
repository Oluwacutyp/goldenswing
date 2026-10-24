"""Feature 3: news pause + news-adaptive execution.

The calendar never touches the network in tests (injected fetcher
+ fixed clock).  Times are UTC.
"""
from datetime import datetime, timezone

import pytest

from goldenswing.live.news import (
    NewsCalendar, NewsEvent, NEWS_SIZE_MULT)


def _utc(y, m, d, h=0, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


def _cal(events, now):
    return NewsCalendar(fetcher=lambda url, timeout: events,
                        clock=lambda: now)


def _ff_event(title, when):
    return {"title": title, "date": when.isoformat(), "impact": "High"}


# ------------------------------------------------------------- calendar
def test_fetch_filters_high_impact_only_and_sorts():
    now = _utc(2026, 10, 15, 12, 0)
    cal = _cal([
        {"title": "Low Thing", "date": _utc(2026, 10, 15, 14, 0).isoformat(),
         "impact": "Low"},
        {"title": "CPI", "date": _utc(2026, 10, 15, 13, 30).isoformat(),
         "impact": "high"},
        _ff_event("GDP", _utc(2026, 10, 15, 12, 30)),
        {"title": "Bad Row", "date": "not-a-date", "impact": "High"},
    ], now)
    events = cal.events()
    assert [e.title for e in events] == ["GDP", "CPI"]


def test_fetch_failure_falls_back_to_nfp():
    # 2026-11-06 is the first Friday of November 2026.
    now = _utc(2026, 11, 2, 12, 0)  # Monday
    def boom(url, timeout):
        raise ConnectionError("down")
    cal = NewsCalendar(fetcher=boom, clock=lambda: now)
    events = cal.events()
    assert len(events) == 1
    assert events[0].title == "NFP"
    assert events[0].time == _utc(2026, 11, 6, 13, 30)


def test_fallback_never_cached_and_fetch_never_raises():
    now = _utc(2026, 11, 2, 12, 0)
    calls = []
    def boom(url, timeout):
        calls.append(1)
        raise ConnectionError("down")
    cal = NewsCalendar(fetcher=boom, clock=lambda: now)
    cal.events()
    cal.events()  # fallback recomputed, not cached
    assert len(calls) == 2


def test_cache_round_trip_and_ttl(tmp_path):
    now = _utc(2026, 10, 15, 12, 0)
    path = tmp_path / "news_cache.json"
    calls = []
    def fetch(url, timeout):
        calls.append(1)
        return [_ff_event("CPI", _utc(2026, 10, 15, 13, 30))]
    cal = NewsCalendar(cache_path=path, fetcher=fetch,
                       clock=lambda: now)
    assert [e.title for e in cal.events()] == ["CPI"]
    assert path.exists()
    # Second calendar, fetcher that explodes: the disk cache serves.
    cal2 = NewsCalendar(cache_path=path,
                        fetcher=lambda u, t: (_ for _ in ()).throw(
                            ConnectionError("down")),
                        clock=lambda: now)
    assert [e.title for e in cal2.events()] == ["CPI"]
    assert len(calls) == 1


# ------------------------------------------------------- execution state
def test_pause_window_around_high_impact_event():
    now = _utc(2026, 10, 15, 13, 30)
    cal = _cal([_ff_event("CPI", _utc(2026, 10, 15, 13, 30))], now)
    # 15 min before: paused (extreme title -> EXTREME_NEWS, 0.4x).
    st = cal.execution_state(_utc(2026, 10, 15, 13, 15))
    assert st.paused is True and st.regime == "EXTREME_NEWS"
    assert st.size_mult == NEWS_SIZE_MULT["EXTREME_NEWS"]
    assert st.event_title == "CPI"
    # 45 min after: still paused (window end is inclusive).
    assert cal.execution_state(_utc(2026, 10, 15, 14, 15)).paused is True
    # 46 min after: cooldown, not paused.
    st = cal.execution_state(_utc(2026, 10, 15, 14, 16))
    assert st.paused is False and st.regime == "NORMAL"
    # Non-extreme high-impact title -> HIGH_VOL_NEWS, 0.5x.
    cal2 = _cal([_ff_event("Retail Sales", _utc(2026, 10, 15, 13, 30))],
                now)
    st = cal2.execution_state(_utc(2026, 10, 15, 13, 30))
    assert st.regime == "HIGH_VOL_NEWS"
    assert st.size_mult == NEWS_SIZE_MULT["HIGH_VOL_NEWS"]


def test_cooldown_blends_size_back_to_one():
    now = _utc(2026, 10, 15, 13, 30)
    cal = _cal([_ff_event("CPI", _utc(2026, 10, 15, 13, 30))], now)
    # Window ends 14:15.  45 min into the 90-min cooldown
    # (15:00): alpha = 0.5 -> mult = 0.5 * 1.0 + 0.5 * 0.4 = 0.7.
    st = cal.execution_state(_utc(2026, 10, 15, 15, 0))
    assert st.paused is False
    assert st.size_mult == pytest.approx(0.7)
    # 90 min after the window end (15:45): fully back to 1.0.
    st = cal.execution_state(_utc(2026, 10, 15, 15, 45))
    assert st.size_mult == pytest.approx(1.0)
    # Far from any event: NORMAL, 1.0.
    st = cal.execution_state(_utc(2026, 10, 16, 9, 0))
    assert (st.paused, st.size_mult, st.regime) == (False, 1.0, "NORMAL")


def test_execution_state_never_raises_on_garbage():
    cal = NewsCalendar(fetcher=lambda u, t: [{"bad": "row"}],
                       clock=lambda: _utc(2026, 10, 15, 12, 0))
    st = cal.execution_state()
    assert st.paused is False and st.size_mult == 1.0


# ------------------------------------------------------- runner wiring
def test_news_pause_blocks_base_order_not_layer(tmp_path):
    from types import SimpleNamespace
    from goldenswing.execution.executor import OrderRequest
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    from goldenswing.execution.risk import gold_spec

    class FakeFeed:
        def account_state(self):
            return {"equity": 10000.0, "balance": 10000.0,
                    "free_margin": 10000.0}

    class FakeExecutor:
        def symbol_spec(self, symbol):
            return gold_spec()

        def place_order(self, req):
            return SimpleNamespace(ok=True, ticket=9, price=2005.0)

    now = _utc(2026, 10, 15, 13, 30)
    cal = _cal([_ff_event("CPI", _utc(2026, 10, 15, 13, 30))], now)
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "j.jsonl",
        risk_state_path=tmp_path / "r.json",
        kill_switch_path=tmp_path / "KILL")
    runner = MT5Runner(cfg, feed=FakeFeed(), executor=FakeExecutor(),
                       news_calendar=cal)
    # Neutral alignment + a market that will not refuse: stub the
    # guards the order path consults.
    runner._market_open = lambda: (True, "open")
    runner._check_risk_guards = lambda *a: True
    runner._check_margin = lambda *a: True
    runner._alignment = SimpleNamespace(size_multiplier=1.0,
                                        alignment_label="D1_Only")
    # Pinned Wednesday tick: a timeless tick makes the order path
    # fall back to the wall clock, so the second runner below fails
    # on Fridays after 20:00 UTC (Friday-close gate).  The news
    # state itself comes from the calendar's pinned clock, not this
    # tick, so pinning changes no news outcome.
    runner._last_tick = {"mid": 2005.0, "bid": 2004.8, "ask": 2005.2,
                         "time": "2026-10-14 12:00:00"}
    sig = SimpleNamespace(direction=1, entry_ref=2005.0,
                          invalidation=1990.0, atr=10.0,
                          targets=[2040.0], score=80.0, index=42,
                          factors=[])
    pb = SimpleNamespace(name="sweep_reversal")
    stats: dict = {}
    runner._place_live_order(pb, sig, stats)
    assert stats.get("gate_news") == 1  # refused, no order placed
    assert runner.tick_manager.position(9) is None

    # Same clock, news pause disabled -> the order flows.
    cfg2 = MT5RunnerConfig(
        journal_path=tmp_path / "j2.jsonl",
        risk_state_path=tmp_path / "r2.json",
        kill_switch_path=tmp_path / "KILL",
        news_pause_enabled=False)
    runner2 = MT5Runner(cfg2, feed=FakeFeed(), executor=FakeExecutor(),
                        news_calendar=cal)
    runner2._market_open = lambda: (True, "open")
    runner2._check_risk_guards = lambda *a: True
    runner2._check_margin = lambda *a: True
    runner2._alignment = SimpleNamespace(size_multiplier=1.0,
                                        alignment_label="D1_Only")
    runner2._last_tick = {"mid": 2005.0, "bid": 2004.8, "ask": 2005.2,
                          "time": "2026-10-14 12:00:00"}
    runner2._stack = SimpleNamespace(frames={"H1": [1, 2, 3]})
    stats2: dict = {}
    runner2._place_live_order(pb, sig, stats2)
    assert stats2.get("gate_news", 0) == 0
    assert stats2.get("executed") == 1

    # News paused, but a LAYER still flows (management, never
    # paused): entry behind price -> pending LIMIT layer.
    stats3: dict = {}
    layer_sig = SimpleNamespace(direction=1, entry_ref=2010.0,
                                invalidation=1990.0, atr=10.0,
                                targets=[2040.0], score=80.0, index=43,
                                factors=[])
    runner._place_live_order(pb, layer_sig, stats3, is_layer=True,
                             layer_lots=0.01)
    assert stats3.get("gate_news", 0) == 0
    assert stats3.get("pending") == 1
