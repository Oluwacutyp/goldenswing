"""Confluence rework (arrival pass) — spec v2 contract tests.

Covers: the Asymmetry Score (geometric structure, flip rule),
A13 Decimal sizing + RiskPool ledger, A1/A2 zone grading +
lifecycle, A12 absorption funding, the §3.2 narrative fill-plan
re-point, A6 resting limits, A14 unshadow — and the acceptance
scenario from the owner session: a point-3-shaped setup (price
lingering in a fresh H4 POI with absorption prints, D1 bias
AGAINST the trade) must produce a layer-1 authorization with no
structural break anywhere in the frames.
"""
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.live.asymmetry import ScoreContext, score_asymmetry
from goldenswing.live.funding import detect_funding_events
from goldenswing.live.opposing import risk_tier
from goldenswing.live.sizing import (RiskPool, VolumeConstraints,
                                     quantize_down, size_for_risk, D)
from goldenswing.live.zones import StructuralZone, score_zone


# ------------------------------------------------------------ frames

def _frame(closes, *, end="2026-10-21 13:00", freq="1h", spread=0.6,
           volume=1000.0):
    """OHLC frame from closes.  Opens sit at the prev-close/close
    midpoint so turning bars are STRICT local extremes (fractal
    swings reject ties by construction)."""
    rows = []
    prev = closes[0]
    for c in closes:
        o = (prev + c) / 2.0
        rows.append((o, max(o, c) + spread / 2.0,
                     min(o, c) - spread / 2.0, c, volume))
        prev = c
    idx = pd.date_range(end=end, periods=len(closes), freq=freq,
                        tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close",
                                       "volume"], index=idx)


def _wave(n_cycles, *, drop, rise, low_step, start=100.0):
    """Sawtooth closes: 3 bars down, 5 bars up per cycle — swing
    lows at the junctions, rising by low_step each cycle."""
    closes = []
    price = start
    low = start
    for k in range(n_cycles):
        target_low = start - drop + k * low_step
        for i in range(3):
            price = price + (target_low - price) * (i + 1) / 3.0 \
                if i < 2 else target_low
            closes.append(price)
        low = target_low
        for i in range(5):
            price = low + rise * (i + 1) / 5.0
            closes.append(price)
    return closes


# ------------------------------------------------------- asymmetry

def test_structure_tight_pivots_outscore_stretched():
    tight = _frame(_wave(6, drop=1.5, rise=1.8, low_step=0.3),
                   spread=0.2)
    stretched = _frame(_wave(6, drop=1.5, rise=4.5, low_step=3.0),
                       spread=0.2)
    s_tight = score_asymmetry(ScoreContext(
        direction=1, frame=tight, atr=0.5, price=float(
            tight["close"].iloc[-1])))
    s_far = score_asymmetry(ScoreContext(
        direction=1, frame=stretched, atr=0.5, price=float(
            stretched["close"].iloc[-1])))
    assert s_tight.component("structure") \
        > s_far.component("structure") + 5.0


def test_structure_flip_rule_zeroes_and_flips_side():
    closes = _wave(6, drop=1.5, rise=1.8, low_step=0.3)
    frame = _frame(closes, spread=0.2)
    price = float(frame["close"].iloc[-1])
    base = score_asymmetry(ScoreContext(
        direction=1, frame=frame, atr=0.5, price=price))
    assert not base.flipped
    # Plunge through the last defended pivot and stay there.
    last_low = float(frame["low"].min())
    plunge = closes + [last_low - 2.0, last_low - 2.6,
                       last_low - 2.4, last_low - 2.5]
    frame2 = _frame(plunge, spread=0.2)
    flipped = score_asymmetry(ScoreContext(
        direction=1, frame=frame2, atr=0.5,
        price=float(frame2["close"].iloc[-1])))
    assert flipped.flipped is True
    assert flipped.direction == -1


def test_structure_geometric_mean_drags_on_dead_defense():
    # Two defended pivots, BOTH tested and closed through (failed),
    # a third intact + untested: defense (over tested pivots) = 0
    # -> the geometric mean takes structure to 0 even though
    # spacing/frequency are alive.  An arithmetic mean would not.
    closes = []
    # Decline to pivot 1 (bar 10, close 90, wick low 88.5).
    closes += [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0,
               92.0, 91.0, 90.0]
    # Bounce; bar 13 TESTS pivot 1 (wick within 0.25 ATR); bar 15
    # CLOSES THROUGH it (87.5 < 88.5 - breach) -> defense fails.
    closes += [92.0, 94.0, 90.8, 93.0, 87.5]
    # Recovery, decline to pivot 2 (bar 21, close 95, low 93.5).
    closes += [92.0, 96.0, 100.0, 98.0, 96.5, 95.0]
    # Bar 23 tests pivot 2; bar 25 closes through it -> fails.
    closes += [97.0, 95.8, 96.5, 92.5]
    # Recovery, decline to pivot 3 (bar 31, close 100, low 98.5):
    # intact and never retested.
    closes += [97.0, 101.0, 105.0, 103.0, 101.5, 100.0]
    closes += [103.0, 106.0, 109.0, 108.0, 107.0, 106.0, 105.0]
    frame = _frame(closes, spread=3.0)
    atr = float((frame["high"] - frame["low"]).mean())
    s = score_asymmetry(ScoreContext(
        direction=1, frame=frame, atr=atr, price=105.0))
    subs = s.components["structure"].subs
    assert subs["defense"] == 0.0
    assert s.component("structure") == 0.0


# ------------------------------------------------------------- sizing

def test_sizing_quantization_step_001():
    r = size_for_risk(Decimal("37"), 10.0, gold_spec())
    assert r.ok and r.lots == Decimal("0.03")  # raw 0.037 -> 0.03
    assert r.drag == Decimal("7")              # $7 of risk unspent


def test_sizing_quantization_coarse_steps():
    cons = VolumeConstraints(min_volume=D("0.05"),
                             max_volume=D("50"), step=D("0.05"))
    r = size_for_risk(Decimal("130"), 10.0, gold_spec(),
                      constraints=cons)
    assert r.ok and r.lots == Decimal("0.10")  # raw 0.13 -> 0.10
    cons25 = VolumeConstraints(min_volume=D("0.25"),
                               max_volume=D("50"), step=D("0.25"))
    r = size_for_risk(Decimal("900"), 10.0, gold_spec(),
                      constraints=cons25)
    assert r.ok and r.lots == Decimal("0.75")  # raw 0.90 -> 0.75


def test_sizing_subminimum_floors_to_minimum():
    # Owner ruling 2026-10-24 (supersedes the old decline): a
    # setup is account-agnostic — when the stop risk exceeds the
    # balance the engine trades the broker MINIMUM, never
    # abandons.  risk 5 over a $10 stop -> raw 0.005 lots ->
    # floored UP to the 0.01 minimum, flagged min_override,
    # estimated stop exposure $10, drag -$5 (the honest
    # arithmetic: the floor carries more than authorized).
    # Step quantization for raw >= minimum still floors DOWN.
    r = size_for_risk(Decimal("5"), 10.0, gold_spec())
    assert r.ok and r.verdict == "ok"
    assert r.lots == Decimal("0.01")
    assert r.min_override is True
    assert r.raw_lots == Decimal("0.005")
    assert r.estimated_stop_exposure == Decimal("10")
    assert r.drag == Decimal("-5")


def test_sizing_decimal_discipline_beats_float_math():
    # raw = 29 / (100 ticks x $1) = exactly 0.29.  In float,
    # 0.29 / 0.01 = 28.999999999999996 -> floor 28 -> 0.28 lots.
    # The Decimal path must land exactly 0.29.
    r = size_for_risk(Decimal("29"), 1.0, gold_spec())
    assert r.ok and r.lots == Decimal("0.29")
    assert D(0.1) == Decimal("0.1")
    assert quantize_down(D("0.29"), D("0.01")) == Decimal("0.29")


def test_sizing_volume_limit_respected():
    cons = VolumeConstraints(min_volume=D("0.01"),
                             max_volume=D("50"), step=D("0.01"),
                             limit=D("0.05"))
    r = size_for_risk(Decimal("10000"), 10.0, gold_spec(),
                      constraints=cons, open_volume=D("0.04"))
    assert r.ok and r.lots == Decimal("0.01")
    r2 = size_for_risk(Decimal("10000"), 10.0, gold_spec(),
                       constraints=cons, open_volume=D("0.05"))
    assert not r2.ok and r2.verdict == "decline_volume_limit"


# ------------------------------------------------------------- pool

def _pool(budget="150"):
    return RiskPool(Decimal(budget))


def test_pool_cap_enforced_across_layers():
    pool = _pool()
    spec = gold_spec()
    assert pool.authorize("base", "0.10", 2000.0, 1990.0, 1, spec)
    assert pool.consumption() == Decimal("100")
    # A second $100 layer would take consumption to $200 > $150.
    assert not pool.authorize("l2", "0.10", 2005.0, 1995.0, 1, spec)
    assert pool.authorize("l2", "0.04", 2005.0, 1995.0, 1, spec)
    assert pool.consumption() == Decimal("140")


def test_pool_breakeven_refund_net_of_friction_reserve():
    pool = _pool()
    spec = gold_spec()
    pool.authorize("base", "0.10", 2000.0, 1990.0, 1, spec)
    refunded = pool.move_stop("base", 2000.0)   # stop -> breakeven
    # $100 risk released minus the 5% friction reserve: the pool
    # still books $5 for this layer, $95 comes back.
    assert pool.consumption() == Decimal("5")
    assert refunded == Decimal("95")
    assert pool.headroom() == Decimal("145")


def test_pool_consumption_floored_at_zero():
    pool = _pool()
    spec = gold_spec()
    pool.authorize("base", "0.10", 2000.0, 1990.0, 1, spec)
    pool.move_stop("base", 2010.0)   # stop trailed INTO profit
    # Locked profit mints no budget: consumption is the small
    # friction reserve only, never negative.
    assert pool.consumption() == Decimal("5")
    pool.remove("base")
    assert pool.consumption() == Decimal("0")


def test_pool_residual_carry_across_layers():
    pool = RiskPool(Decimal("100"))
    spec = gold_spec()
    pool.authorize("a", "0.03", 2000.0, 1990.0, 1, spec)   # $30
    pool.move_stop("a", 2000.0)          # BE: consumption -> $1.50
    # The refund is spendable by the next layer (residual carry):
    assert pool.authorize("b", "0.09", 2010.0, 2000.0, 1, spec)
    assert pool.consumption() == Decimal("91.50")
    assert not pool.authorize("c", "0.02", 2010.0, 2000.0, 1, spec)
    assert pool.authorize("c", "0.008", 2010.0, 2000.0, 1, spec)


# ------------------------------------------------- zone grading (A1/A2)

def _zone(**kw):
    base = dict(tf="W1", side="resistance", level=4300.0,
                top=4310.0, bottom=4290.0, atr=20.0,
                displacement_atr=2.5)
    base.update(kw)
    return StructuralZone(**base)


def test_score_zone_freshness_sign():
    fresh = score_zone(_zone(touches=0), [], age_d1=2.0)[0]
    once = score_zone(_zone(touches=1), [], age_d1=2.0)[0]
    twice = score_zone(_zone(touches=2), [], age_d1=2.0)[0]
    thrice = score_zone(_zone(touches=3), [], age_d1=2.0)[0]
    # Consumption model: fresh strongest, touches only spend.
    assert fresh > once > twice
    assert thrice <= twice  # never a reward for the 3rd touch


def test_score_zone_departure_validation():
    weak = score_zone(_zone(departure_atr=0.5,
                            displacement_atr=0.5), [], age_d1=2.0)[0]
    strong = score_zone(_zone(departure_atr=2.5), [], age_d1=2.0)[0]
    assert strong > weak + 10.0


def test_zone_lifecycle_classes_and_mapping():
    assert _zone(touches=0).lifecycle == "fresh"
    assert _zone(touches=1).lifecycle == "tested"
    assert _zone(touches=2).lifecycle == "tested"
    assert _zone(touches=3).lifecycle == "used_up"
    assert _zone(touches=4).lifecycle == "used_up"
    assert _zone(status="broken", touches=0).lifecycle == "used_up"
    # A6 mapping: only fresh MAJOR rests a limit.
    z = _zone(touches=0)
    z.tier = "MAJOR"
    assert z.resting_limit_eligible is True
    z2 = _zone(touches=1)
    z2.tier = "MAJOR"
    assert z2.resting_limit_eligible is False
    # A11 edges: resistance proximal = bottom (entry), distal = top.
    assert z.proximal == 4290.0 and z.distal == 4310.0


def test_risk_tier_green_does_not_require_touches():
    fresh_w1 = {"score": 90.0, "touches": 0,
                "displacement_atr": 2.5, "tf": "W1"}
    assert risk_tier(fresh_w1) == "GREEN"


# ------------------------------------------------------------- funding

def test_absorption_funding_event():
    rows = []
    price = 4130.0
    for i in range(32):  # decline toward the zone
        price -= 0.5
        rows.append((price + 0.5, price + 0.8, price - 0.8,
                     price, 1000.0))
    # Two failed pushes into the zone, closing back inside with
    # rejection wicks (long lower wicks at a support zone).
    rows.append((4116.0, 4117.0, 4101.0, 4114.0, 1200.0))
    rows.append((4114.0, 4115.0, 4102.0, 4113.0, 1100.0))
    idx = pd.date_range(end="2026-10-21 13:00", periods=len(rows),
                        freq="15min", tz="UTC")
    m15 = pd.DataFrame(rows, columns=["open", "high", "low", "close",
                                      "volume"], index=idx)
    zone = {"top": 4122.0, "bottom": 4098.0, "level": 4110.0}
    rep = detect_funding_events(m15, direction=1, atr=12.0,
                                zone=zone)
    kinds = [e.kind for e in rep.events]
    assert "absorption" in kinds
    ev = next(e for e in rep.events if e.kind == "absorption")
    assert ev.colour == 4.0


# ------------------------------------------------------------- runner

class _FakeExecutor:
    def __init__(self):
        self.orders = []
        self._ticket = 9000
        self._positions = []

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return list(self._positions)

    def place_order(self, request):
        from goldenswing.execution.executor import (OrderResult,
                                                    Position)
        self._ticket += 1
        self.orders.append({"ticket": self._ticket,
                            "volume": request.volume,
                            "kind": request.order_kind,
                            "price": request.order_price,
                            "sl": request.stop_loss,
                            "alert_id": request.alert_id})
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
        from goldenswing.execution.executor import OrderResult
        return OrderResult(True, ticket=ticket)

    def cancel_pending(self, ticket):
        from goldenswing.execution.executor import OrderResult
        self._positions = [p for p in self._positions
                           if p.ticket != ticket]
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        from goldenswing.execution.executor import OrderResult
        return OrderResult(True, ticket=ticket)


def _feed():
    return SimpleNamespace(
        account_state=lambda: {"equity": 10000.0, "balance": 10000.0,
                               "free_margin": 10000.0},
        order_margin=lambda *, volume, price, direction:
            float(volume) * 100.0 * float(price) / 500.0)


def _runner(tmp_path, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        news_pause_enabled=False, **cfg_kw)
    runner = MT5Runner(cfg, feed=_feed(), executor=_FakeExecutor())
    runner._send_alert = lambda *a, **k: None
    runner._build_chart = lambda *a, **k: (None, None)
    return runner


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


def _point3_frames():
    """The acceptance geometry: decline into a fresh H4 POI, then
    price LINGERING inside it with absorption prints.  No close
    anywhere breaks a swing high — there is no structural break
    for a chain to confirm."""
    # D1: dealing range 4090-4320, price at the discount extreme.
    d1_closes = [4320.0 - 7.0 * i for i in range(30)]
    d1_closes[-1] = 4115.0
    d1 = _frame(d1_closes, freq="1d", spread=6.0)
    d1.iloc[3, d1.columns.get_loc("low")] = 4090.0
    # H4: down-leg into the zone.
    h4_closes = [4250.0 - 3.4 * i for i in range(40)]
    h4 = _frame(h4_closes, freq="4h", spread=4.0)
    # H1: sharp impulsive approach, then lingering inside the band
    # 4098-4122 (small bodies, closes holding).
    h1_closes = [4250.0 - 9.0 * i for i in range(15)]
    h1_closes += [4120.0, 4112.0, 4116.0, 4110.0, 4114.0, 4112.0,
                  4115.0, 4113.0, 4114.0, 4115.0]
    h1 = _frame(h1_closes, spread=1.2)
    h1["volume"] = 1000.0
    h1.iloc[-1, h1.columns.get_loc("volume")] = 1400.0
    # M15: absorption — repeated pushes toward the zone bottom,
    # closes back inside, rejection wicks.
    rows = []
    p = 4126.0
    for i in range(30):
        p -= 0.4
        rows.append((p + 0.4, p + 0.7, p - 0.7, p, 1000.0))
    rows.append((4116.0, 4117.0, 4101.0, 4114.0, 1200.0))
    rows.append((4114.0, 4115.5, 4102.0, 4113.0, 1150.0))
    rows.append((4113.0, 4116.0, 4108.0, 4115.0, 1100.0))
    idx = pd.date_range(end="2026-10-21 13:00", periods=len(rows),
                        freq="15min", tz="UTC")
    m15 = pd.DataFrame(rows, columns=["open", "high", "low", "close",
                                      "volume"], index=idx)
    return {"H1": h1, "M15": m15, "H4": h4, "D1": d1}


def test_point3_fresh_h4_poi_absorption_produces_layer1_authorization_without_structural_break(
        tmp_path):
    runner = _runner(tmp_path)
    frames = _point3_frames()
    zone = StructuralZone(
        tf="H4", side="support", level=4110.0, top=4122.0,
        bottom=4098.0, score=85.0, tier="MAJOR", status="active",
        touches=0, atr=12.0, displacement_atr=2.5,
        departure_atr=2.5, swing_ts="2026-10-18T00:00:00+00:00")
    runner._structural_zones = [zone]
    runner._stack = SimpleNamespace(frames=frames)
    runner._last_tick = {"mid": 4114.0, "spread": 0.3}
    h1 = frames["H1"]
    sig = SimpleNamespace(
        direction=1, score=72.0, entry_ref=4114.0,
        invalidation=4090.0, targets=[4180.0], atr=12.0,
        index=len(h1) - 1, factors=["absorption"])
    pb = _StubPB("fvg_midline", sig)
    runner._get_playbooks = lambda: [pb]
    verdict = {
        "allowed": True, "bias_d1": -1, "bias_h4": 1,
        "regime": "trend", "direction": 1,
        "zone": {"top": 4122.0, "bottom": 4098.0, "level": 4110.0},
        "major_zone": zone.to_dict(),
        "trigger": "absorption", "entry": 4114.0, "stop": 4090.0,
        "target": 4180.0, "risk": 24.0, "entry_clock": "M15",
        "session": {}, "amd_phase": "accumulation",
        "qm": {"status": "none"}, "volume": {"volume_ratio": 1.2},
        "dxy_corr": -0.35, "urgency": None,
    }
    stats: dict = {"signals": 0, "gated": 0}
    runner._run_signal_path(h1, len(h1) - 1, [], 1.0, verdict, stats)
    # Layer 1 authorized + filled — with NO structural break:
    # the funding detectors confirm no structure shift fired.
    funding = runner._funding_report(verdict, zone.to_dict())
    assert "structure_shift" not in [e.kind for e in funding.events]
    assert stats.get("executed") == 1
    assert len(runner.executor.orders) == 1
    tickets = runner.tick_manager.tickets()
    assert len(tickets) == 1
    # The position carries a RiskPool ledger (A13).
    pool = runner._pools.get(tickets[0])
    assert pool is not None and pool.consumption() > 0
    # Journal: the accepted record carries the full breakdown +
    # trigger source (spec §6).
    import json
    records = [json.loads(line) for line in
               (tmp_path / "mt5_journal.jsonl").read_text().splitlines()
               if line.strip()]
    sig_recs = [r for r in records if r.get("type") == "signal"
                and r.get("decision") == "accepted"]
    assert sig_recs, "an accepted signal record must exist"
    rec = sig_recs[-1]
    assert rec["playbook"] == "fvg_midline"
    assert rec["trigger_source"] == "arrival_score"
    bd = rec["score_breakdown"]
    for comp in ("structure", "efficiency", "pullback", "location",
                 "session", "macro"):
        assert comp in bd
    assert bd["total"] >= 50.0
    assert bd["location"]["points"] >= 12.0


def test_resting_limit_placed_for_fresh_major_zone(tmp_path):
    runner = _runner(tmp_path, resting_limit_enabled=True)
    frames = _point3_frames()
    runner._stack = SimpleNamespace(frames=frames)
    zone = StructuralZone(
        tf="H4", side="support", level=4110.0, top=4122.0,
        bottom=4098.0, score=85.0, tier="MAJOR", status="active",
        touches=0, atr=12.0, displacement_atr=2.5,
        departure_atr=2.5, swing_ts="2026-10-18T00:00:00+00:00")
    runner._structural_zones = [zone]
    runner._live_setup = {
        "direction": 1, "regime": "trend", "atr_h1": 12.0,
        "buf": 3.0, "major_zone": zone.to_dict(),
        "zone": {"top": 4122.0, "bottom": 4098.0, "level": 4110.0},
        "bias_d1": 0, "bias_h4": 1,
    }
    # Price 0.3 ATR above the proximal edge — ahead of arrival.
    tick = {"mid": 4122.0 + 3.6, "spread": 0.3,
            "time": pd.Timestamp("2026-10-21 13:00", tz="UTC")}
    stats: dict = {}
    runner._maybe_place_resting_limit(tick, stats)
    assert len(runner._pending_orders) == 1
    state = next(iter(runner._pending_orders.values()))
    assert state.kind == "limit"
    assert state.level == pytest.approx(4122.0)  # proximal edge
    assert state.playbook == "arrival_zone"
    import json
    records = [json.loads(line) for line in
               (tmp_path / "mt5_journal.jsonl").read_text().splitlines()
               if line.strip()]
    recs = [r for r in records if r.get("type") == "signal"
            and r.get("trigger_source") == "resting_limit"]
    assert recs, "resting limit must journal its source distinctly"


def test_narrative_fill_plan_built_when_chain_not_in_entry(tmp_path):
    runner = _runner(tmp_path)
    frames = _point3_frames()
    runner._stack = SimpleNamespace(frames=frames)
    sig = SimpleNamespace(kind="narrative", direction=1,
                          invalidation=4090.0, targets=[4180.0])
    state, kwargs = runner._narrative_fill_plan(sig, 4114.0)
    # The chain never reached ENTRY on these frames (no zone
    # approach sequence) — the re-point still builds the plan.
    assert kwargs.get("narrative_plan")
    assert "tp1" in kwargs["narrative_plan"]


def test_a14_no_playbook_is_log_only_by_default(tmp_path):
    runner = _runner(tmp_path)
    names = {pb.name for pb in runner._get_playbooks()}
    assert names  # the H1 path evaluates a real roster
    # Execute set == evaluated set (config default).
    assert runner.cfg.execute_playbooks is None
    for name in names:
        assert runner._executes(name), \
            f"{name} would be log-only by default (A14 violation)"
    # The kill switch narrows EVALUATION, not execution tier:
    # a disabled playbook is skipped before evaluation.
    runner2 = _runner(tmp_path, disabled_playbooks=("qm_m5",))
    assert "qm_m5" in (runner2.cfg.disabled_playbooks or ())
    assert runner2._executes("qm_m5")  # still execute-eligible;
    # it is the disabled list, checked in _run_signal_path, that
    # keeps it out — one switch, owner-controlled.


# ------------------------------------------- style routing (floors)
# The per-style floor routes on the trade's actual style: a
# CONTINUATION-arm signal (hierarchy F2: closed-H1 break + retest,
# verdict arm == "continuation") is floored by the
# INTRADAY_CONTINUATION row (floor 65, structure >= 18,
# efficiency >= 12, pullback >= 6, NO location minimum) — not by
# the ZONE_PATH arrival row, whose Location >= 12 is a
# zone-arrival trade-TYPE condition.  These tests pin the routing
# with the exact diagnosed score shape: total 69.54 (structure
# 22.7, efficiency 13.24, pullback 15.0, location 3.6, session 10,
# macro 5) — declined on ZONE_PATH (Location 3.6 < 12), passing
# every term of INTRADAY_CONTINUATION.  The scorer is stubbed so
# the ROUTING is the unit under test; the tick-path end-to-end
# case with the real scorer is test_arming_gate_fix.py::test_e.

class _FloorEngine:
    """Engine double for the floor-routing drive: hard lines pass,
    confluence is a stub number (the floor decides before it)."""

    def hard_line_check(self, pb, sig, ctx, spread_px=None):
        return True, ""

    def score_live_signal(self, pb, sig, ctx, hierarchy_verdict=None):
        return 80.0, ["stub"]


def _fake_asym(total, *, structure, efficiency, pullback, location,
               session=10.0, macro=5.0):
    from goldenswing.live.asymmetry import (AsymmetryScore,
                                            ComponentResult)
    comps = {}
    for name, pts, mx in (("structure", structure, 30.0),
                          ("efficiency", efficiency, 25.0),
                          ("pullback", pullback, 15.0),
                          ("location", location, 15.0),
                          ("session", session, 10.0),
                          ("macro", macro, 5.0)):
        comps[name] = ComponentResult(name, float(pts), float(mx))
    return AsymmetryScore(total=float(total), direction=1,
                          flipped=False, components=comps)


_DIAGNOSED = dict(total=69.54, structure=22.7, efficiency=13.24,
                  pullback=15.0, location=3.6)


def _drive_floor(tmp_path, *, verdict_extra, fake):
    """One stub signal through _run_one_playbook with a stubbed
    asymmetry score and a REAL zone context (the broken level's
    band / POI), so the per-style floor actually applies."""
    from goldenswing.playbooks.base import EngineContext
    runner = _runner(tmp_path)
    frames = _point3_frames()
    zone = StructuralZone(
        tf="H4", side="support", level=4110.0, top=4122.0,
        bottom=4098.0, score=85.0, tier="MAJOR", status="active",
        touches=0, atr=12.0, displacement_atr=2.5,
        departure_atr=2.5, swing_ts="2026-10-18T00:00:00+00:00")
    runner._structural_zones = [zone]
    runner._stack = SimpleNamespace(frames=frames)
    runner._last_tick = {"mid": 4114.0, "spread": 0.3}
    runner._score_for = lambda **kw: fake
    h1 = frames["H1"]
    sig = SimpleNamespace(
        direction=1, score=72.0, entry_ref=4114.0,
        invalidation=4090.0, targets=[4180.0], atr=12.0,
        index=len(h1) - 1, factors=["retest"])
    pb = _StubPB("fvg_midline", sig)
    verdict = {
        "allowed": True, "bias_d1": -1, "bias_h4": 1,
        "regime": "trend", "direction": 1,
        "zone": {"top": 4122.0, "bottom": 4098.0, "level": 4110.0},
        "major_zone": zone.to_dict(),
        "trigger": "retest", "entry": 4114.0, "stop": 4090.0,
        "target": 4180.0, "risk": 24.0, "entry_clock": "M15",
        "session": {}, "amd_phase": "accumulation",
        "qm": {"status": "none"}, "volume": {"volume_ratio": 1.2},
        "dxy_corr": -0.35, "urgency": None,
    }
    verdict.update(verdict_extra)
    ctx = EngineContext(df=h1, states=[], years=1.0,
                        instrument="XAUUSD")
    stats: dict = {"signals": 0, "gated": 0}
    runner._run_one_playbook(
        pb, sig_ctx=(ctx, _FloorEngine()), h1=h1, t=len(h1) - 1,
        verdict=verdict, stats=stats, zone_d=zone.to_dict(),
        funding=None)
    recs = [r for r in runner.journal.records()
            if r.get("type") == "signal"]
    return runner, stats, recs


def test_continuation_signal_routes_to_intraday_continuation_floor(
        tmp_path):
    # (i) Diagnosed shape + arm == "continuation": Location 3.6
    # would fail ZONE_PATH's arrival condition, but the
    # continuation row has no location minimum and every one of
    # its terms passes (69.54 >= 65, 22.7 >= 18, 13.24 >= 12,
    # 15.0 >= 6) — the signal is authorized and ordered.
    runner, stats, recs = _drive_floor(
        tmp_path, verdict_extra={"arm": "continuation"},
        fake=_fake_asym(**_DIAGNOSED))
    assert [r["decision"] for r in recs] == ["accepted"]
    assert recs[0]["score_breakdown"]["location"]["points"] \
        == pytest.approx(3.6)
    assert recs[0]["score_breakdown"]["total"] == pytest.approx(69.54)
    assert stats.get("signals") == 1
    assert len(runner.executor.orders) == 1


def test_continuation_floor_has_teeth_structure_below_18(tmp_path):
    # (ii) Same arm, total ABOVE the continuation floor (70 >= 65)
    # but structure 10 < 18: the row's structure minimum declines
    # it, and the journaled reason names the row actually used.
    runner, stats, recs = _drive_floor(
        tmp_path, verdict_extra={"arm": "continuation"},
        fake=_fake_asym(total=70.0, structure=10.0, efficiency=20.0,
                        pullback=10.0, location=15.0))
    assert [r["decision"] for r in recs] == ["below_floor"]
    assert "INTRADAY_CONTINUATION" in recs[0]["reason"]
    assert stats.get("below_floor") == 1
    assert runner.executor.orders == []


def test_zone_arrival_location_below_12_still_declines_zone_path(
        tmp_path):
    # (iii) The SAME diagnosed score with NO continuation arm is a
    # zone-arrival signal: ZONE_PATH's Location >= 12 condition is
    # unmoved — it declines, and the reason names ZONE_PATH.
    runner, stats, recs = _drive_floor(
        tmp_path, verdict_extra={}, fake=_fake_asym(**_DIAGNOSED))
    assert [r["decision"] for r in recs] == ["below_floor"]
    assert "ZONE_PATH" in recs[0]["reason"]
    assert "INTRADAY_CONTINUATION" not in recs[0]["reason"]
    assert stats.get("below_floor") == 1
    assert runner.executor.orders == []
