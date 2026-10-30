"""Step 15 (owner rulings 2026-10-30 §2): execution intent.

The Oct 30 failure shape: order type was chosen by pure price-vs-
level geometry, with no record of what the SETUP'S THESIS intended.
An accepted qm_m5 short was parked as a LIMIT at 02:20 and expired
unfilled in a cascade — while continuation-shaped candidates had no
way to declare that their premise was the displacement itself.

Acceptance shape (per the ruling):
- every candidate's record carries its intended execution type
  (family) and why it matches its setup thesis;
- pullback / mitigation / return-to-zone setups use limit orders;
- confirmed breakout / displacement / immediate-reaction setups
  may use market or stop-entry orders when their trigger fires —
  they are never forced into a passive limit by geometry alone;
- a missed or expired limit NEVER auto-converts to a market order;
  a different method needs a fresh qualifying trigger;
- the same thesis expressed by different playbooks cannot stack
  orders merely because they prefer different execution methods.
"""
from __future__ import annotations

from types import SimpleNamespace

from goldenswing.execution.mt5 import select_order_type
from goldenswing.execution.mt5_runner import execution_intent_for

from test_p3_trader import (_FakeScore, _StubPB, _drive_one, _flat_h1,
                            _runner, _sig)


# ---------------------------------------------------------------- resolver

def test_resolver_declared_theses():
    assert execution_intent_for("v_reversal")["family"] == "market"
    assert execution_intent_for(
        "range_compression_breakout")["family"] == "market"
    assert execution_intent_for("hot_scalp")["family"] == "market"
    assert execution_intent_for("breakout")["family"] == "market"
    for name in ("fvg_midline", "fvg_limit", "inversion",
                 "turtle_soup", "order_block", "narrative",
                 "qm_m5", "qm_d1", "sweep_reversal",
                 "trend_continuation"):
        assert execution_intent_for(name)["family"] == "limit", name
    # Unknown playbooks default to the passive thesis.
    assert execution_intent_for("no_such_playbook")["family"] \
        == "limit"
    # Every intent carries its why.
    for name in ("v_reversal", "fvg_midline", "hot_scalp"):
        assert execution_intent_for(name)["basis"]


def test_resolver_kind_breakout_is_market():
    sig = SimpleNamespace(kind="breakout")
    assert execution_intent_for("anything", sig)["family"] \
        == "market"
    sig = SimpleNamespace(kind="squeeze_breakout")
    assert execution_intent_for("anything", sig)["family"] \
        == "market"


# ------------------------------------------------------- select_order_type

def test_select_market_family_past_poi_is_market_not_limit():
    # Short, price already 1.5 below the entry (inside tolerance):
    # geometry alone rests a no-chase LIMIT; a displacement thesis
    # enters at market.
    kw = dict(direction=-1, price=3998.5, entry_ref=4000.0,
              atr=10.0)
    kind, _lvl, _why = select_order_type(**kw)
    assert kind == "limit"
    kind, lvl, why = select_order_type(intent_family="market", **kw)
    assert kind == "market" and lvl == 3998.5
    assert "execution intent MARKET" in why
    kind, _lvl, _why = select_order_type(intent_family="limit", **kw)
    assert kind == "limit"


def test_select_market_family_pullback_geometry_is_market():
    # Long, entry well below price (the break already ran):
    # geometry rests a pullback limit; market family goes market.
    kw = dict(direction=1, price=4010.0, entry_ref=4000.0, atr=10.0)
    assert select_order_type(**kw)[0] == "limit"
    kind, _lvl, why = select_order_type(
        intent_family="market", **kw)
    assert kind == "market"
    assert "no resting limit" in why


def test_select_market_family_stop_ahead_preserved():
    # Entry still ahead of price: stop-entry on the unbroken level
    # for BOTH families (the ruling names stop-entry explicitly).
    kw = dict(direction=1, price=3990.0, entry_ref=4000.0, atr=10.0)
    assert select_order_type(intent_family="market", **kw)[0] \
        == "stop"
    assert select_order_type(intent_family="limit", **kw)[0] \
        == "stop"


def test_select_market_family_zone_failed_still_invalid():
    zone = {"level": 4000.0, "top": 4001.0, "bottom": 3999.0}
    kind, _lvl, _why = select_order_type(
        direction=-1, price=4010.0, entry_ref=4000.0, zone=zone,
        atr=10.0, intent_family="market")
    assert kind == "invalid"


def test_select_market_family_skips_urgency_downgrade():
    profile = {"poc": 4000.0, "high_vol_nodes": []}
    kw = dict(direction=1, price=3999.5, entry_ref=4000.0, atr=10.0,
              urgency=50.0, profile=profile)
    # Thick level + moderate urgency downgrades the legacy call...
    assert select_order_type(**kw)[0] == "limit"
    # ...but never a market-thesis entry.
    assert select_order_type(intent_family="market", **kw)[0] \
        == "market"


# ---------------------------------------------------------------- runner

def _short_sig(entry=4000.0, stop=4010.0, index=29):
    return SimpleNamespace(
        direction=-1, score=80.0, entry_ref=entry,
        invalidation=stop, targets=[3980.0], atr=10.0,
        index=index, factors=["stub"], kind="stub")


def _short_verdict():
    return {"bias_d1": -1, "bias_h4": -1, "regime": "trend",
            "direction": -1, "trigger": "tick-cross",
            "entry_clock": "M15", "amd_phase": "distribution",
            "qm": {"status": "none"}, "volume": {}}


def _past_poi_runner(tmp_path, names):
    runner = _runner(tmp_path, execute_playbooks=names)
    h1 = _flat_h1(40, 2000.0)
    runner._stack = SimpleNamespace(frames={"H1": h1})
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    # Price already 1.5pt past the short entry (inside tolerance).
    runner._last_tick = {"mid": 3998.5, "bid": 3998.4,
                         "ask": 3998.6,
                         "time": "2026-10-14 12:00:00"}
    return runner, h1


def test_market_thesis_candidate_fills_at_market(tmp_path):
    runner, h1 = _past_poi_runner(tmp_path, ("v_reversal",))
    _drive_one(runner, _StubPB("v_reversal", _short_sig()), h1,
               verdict=_short_verdict())
    assert runner.executor.orders, "no order placed"
    assert runner.executor.orders[-1]["kind"] == "market"
    assert runner.tick_manager.tickets(), "market fill must open"
    assert not runner._pending_orders


def test_limit_thesis_same_geometry_rests_a_limit(tmp_path):
    runner, h1 = _past_poi_runner(tmp_path, ("fvg_midline",))
    _drive_one(runner, _StubPB("fvg_midline", _short_sig()), h1,
               verdict=_short_verdict())
    assert runner.executor.orders, "no order placed"
    assert runner.executor.orders[-1]["kind"] == "limit"
    assert runner._pending_orders, "the limit must be working"
    assert not runner.tick_manager.tickets()


def test_intent_is_journaled_per_candidate(tmp_path):
    runner, h1 = _past_poi_runner(tmp_path, ("fvg_midline",))
    _drive_one(runner, _StubPB("fvg_midline", _short_sig()), h1,
               verdict=_short_verdict())
    evals = [r for r in runner.journal.records()
             if r.get("type") == "evaluation"
             and r.get("playbook") == "fvg_midline"]
    with_intent = [r for r in evals
                   if (r.get("candidate") or {}).get(
                       "execution_intent")]
    assert with_intent, "no candidate record carries the intent"
    intent = with_intent[-1]["candidate"]["execution_intent"]
    assert intent["family"] == "limit"
    assert "limit thesis" in intent["basis"]
    notes = [r.get("text", "") for r in runner.journal.records()
             if r.get("type") == "note"]
    assert any("execution intent fvg_midline: family=limit" in n
               for n in notes)


def test_duplicate_working_order_refused_across_methods(tmp_path):
    runner, h1 = _past_poi_runner(
        tmp_path, ("fvg_midline", "v_reversal"))
    _drive_one(runner, _StubPB("fvg_midline", _short_sig()), h1,
               verdict=_short_verdict())
    assert runner._pending_orders
    placed = len(runner.executor.orders)
    # A second playbook (market thesis, same direction) reaches the
    # shared order path while the limit works: refused, journaled.
    stats: dict = {}
    sig = _short_sig(index=30)
    runner._place_live_order(
        SimpleNamespace(name="v_reversal"), sig, stats,
        verdict={"entry": 3998.5})
    assert len(runner.executor.orders) == placed
    assert stats.get("duplicate_working_order") == 1
    notes = [r.get("text", "") for r in runner.journal.records()
             if r.get("type") == "note"]
    assert any("duplicate working order" in n for n in notes)


def test_different_level_order_is_not_a_duplicate(tmp_path):
    # The scoped guard: a working pending at one level does NOT
    # block another playbook's order at a DIFFERENT level — the
    # desk legitimately works both on one frame (the telegram-
    # control suite pins this on realistic data: inversion parks
    # a pending while judas_swing / turtle_soup fill at market).
    runner, h1 = _past_poi_runner(
        tmp_path, ("fvg_midline", "v_reversal"))
    _drive_one(runner, _StubPB("fvg_midline", _short_sig()), h1,
               verdict=_short_verdict())
    assert runner._pending_orders  # limit working at 4000
    placed = len(runner.executor.orders)
    runner._last_tick = {"mid": 3950.0, "bid": 3949.9,
                         "ask": 3950.1,
                         "time": "2026-10-14 12:00:00"}
    stats: dict = {}
    sig = SimpleNamespace(
        direction=-1, score=80.0, entry_ref=3950.0,
        invalidation=3960.0, targets=[3920.0], atr=10.0,
        index=30, factors=["stub"], kind="stub")
    runner._place_live_order(
        SimpleNamespace(name="v_reversal"), sig, stats,
        verdict={"entry": 3950.0})
    assert len(runner.executor.orders) == placed + 1
    assert "duplicate_working_order" not in stats


def test_expired_limit_never_converts_to_market(tmp_path):
    runner, h1 = _past_poi_runner(tmp_path, ("fvg_midline",))
    _drive_one(runner, _StubPB("fvg_midline", _short_sig()), h1,
               verdict=_short_verdict())
    assert runner._pending_orders
    ticket = next(iter(runner._pending_orders))
    placed = len(runner.executor.orders)
    # The pending expires unfilled (the Oct 30 02:20 shape).
    runner._cancel_pending(ticket, "expired: no fill after 4 bars",
                           {})
    assert not runner._pending_orders
    # No fresh trigger arrives: subsequent evaluations place
    # NOTHING — no conversion, no chase, of any method.
    quiet = SimpleNamespace(
        name="fvg_midline",
        live_signal=lambda states, t, df=None: [],
        check_live_signal=lambda s, c: (True, ""))
    _drive_one(runner, quiet, h1, verdict=_short_verdict())
    assert len(runner.executor.orders) == placed
    assert all(o["kind"] == "limit" for o in runner.executor.orders)
    assert not runner.tick_manager.tickets()
