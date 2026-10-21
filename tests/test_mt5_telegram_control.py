"""MT5 Telegram control: inline buttons, /config /set, confirm flow,
owner-only enforcement, blocked-leg stats, playbook wiring.

Covers: _get_playbooks instantiates all 17 playbooks and (A14)
every one of them is execute-eligible by default — the shadow tier
is abolished, the per-playbook kill switch is the only opt-out;
judas_swing fires live into a real order on realistic geometry;
callback_query handling (answer + act + message edited); Pause /
Resume / Trim / Cancel pending / Refresh buttons; config get/set
(valid applies + persists + survives a simulated restart; invalid
rejected, never applied); confirm required for every mutation;
non-owner chat/callback refused; no secret leakage in any output.
"""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.alerts.commands import CommandContext
from goldenswing.execution.executor import OrderResult
from goldenswing.execution.mt5_commands import (
    mt5_control_keyboard, mt5_router, parse_mt5_callback)
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import gold_spec
from goldenswing.live.hierarchy import TimeframeStack

OWNER = "12345"
OWNER_INT = 12345
FAKE_TOKEN = "123456:FAKE-TOKEN-VALUE-abcdef"


class FakeExecutor:
    def __init__(self):
        self.orders: list = []
        self.closes: list[tuple] = []
        self.cancels: list[int] = []
        self._positions: list = []
        self._ticket = 9000
        self._pendings: list = []

    def connect(self, **kw):
        return True, "fake connected"

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return [p for p in self._positions
                if symbol is None or p.symbol == symbol]

    def account_state(self):
        return {"equity": 10000.0}

    def place_order(self, request):
        self._ticket += 1
        self.orders.append(request)
        return OrderResult(True, ticket=self._ticket,
                           price=request.entry_ref,
                           volume=request.volume)

    def modify_sl(self, ticket, sl, tp=None):
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        self.closes.append((ticket, volume))
        self._positions = [p for p in self._positions
                           if p.ticket != ticket]
        return OrderResult(True, ticket=ticket)

    def pending_orders(self, symbol=None):
        return list(self._pendings)

    def cancel_pending(self, ticket):
        self.cancels.append(ticket)
        return OrderResult(True, ticket=ticket)


class FakeSender:
    def __init__(self):
        self.texts: list[str] = []
        self.markups: list = []
        self.documents: list = []

    def send_text(self, text, silent=False, reply_markup=None):
        self.texts.append(text)
        self.markups.append(reply_markup)
        return SimpleNamespace(delivered=True, detail="ok")

    def send_document(self, path, caption=None):
        self.documents.append((str(path), caption))
        return SimpleNamespace(delivered=True, detail="ok")


def _runner(tmp_path, *, executor=None, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        poll_interval_seconds=0,
        fixed_lots=0.01,
        # News pause OFF: the order path's news check reads the
        # REAL wall-clock calendar — these tests must not fail on
        # a real news day (e.g. PPI).  The pause itself is covered
        # by test_news_calendar.py.
        news_pause_enabled=False,
        **cfg_kw)
    runner = MT5Runner(cfg, executor=executor or FakeExecutor())
    runner.feed = SimpleNamespace(
        account_state=lambda: {"equity": 10000.0, "balance": 10000.0,
                               "free_margin": 10000.0},
        order_margin=lambda *, volume, price, direction:
            float(volume) * 100.0 * float(price) / 500.0,
        poll_status=lambda: {"ok": True})
    return runner


def _h1():
    idx = pd.date_range("2026-01-01", periods=40, freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": [2000.0] * 40, "high": [2001.0] * 40,
         "low": [1999.0] * 40, "close": [2000.0] * 40,
         "volume": [100.0] * 40},
        index=idx)


def _verdict(**over):
    v = {"allowed": True, "blocked_leg": "", "legs": {},
         "bias_d1": 1, "bias_h4": 1, "direction": 1,
         "zone": {"bottom": 1990.0, "top": 2010.0},
         "trigger": "bos", "entry": 2000.0, "stop": 1990.0,
         "target": 2030.0, "risk": 10.0, "session": {},
         "entry_clock": "M15", "amd_phase": "markup",
         "qm": {"status": "none"}, "volume": {"volume_ratio": 1.0}}
    v.update(over)
    return v


def _judas_sig():
    return SimpleNamespace(
        direction=1, score=80.0, entry_ref=2000.0, invalidation=1990.0,
        targets=[2030.0], atr=10.0, index=5, factors=[], kind="judas",
        live_context={"sweep_idx": 0, "trigger_idx": 1, "confirm_idx": 2,
                      "retest_idx": 3, "direction": 1,
                      "sweep_session": "london", "sweep_level": 2000.0,
                      "sweep_extreme": 1990.0, "entry": 2000.0,
                      "stop": 1990.0, "invalidation": 1990.0})


def _turtle_sig():
    return SimpleNamespace(
        direction=1, score=80.0, entry_ref=2000.0, invalidation=1990.0,
        targets=[2030.0], atr=10.0, index=5, factors=[], kind="turtle",
        live_context={"sweep_idx": 0, "trigger_idx": 1, "confirm_idx": 2,
                      "retest_idx": 3, "direction": 1,
                      "fvg_formed_idx": 2, "entry": 2000.0,
                      "stop": 1990.0, "invalidation": 1990.0})


def _judas_frame():
    """Realistic judas-swing geometry (40 H1 bars): a decline from
    2058, a defended staircase of higher lows (swing lows 1958 /
    1972 / 1986, each held on closes), then the London raid — bar
    35 wicks to 1979 THROUGH the 1986 pivot and closes back at
    1993 — and a three-bar displacement reclaim into bar 39.

    The old harness fed 40 perfectly flat bars: no pivots, no leg,
    no arrival — the Asymmetry Score has nothing to measure there
    (total ~42, all of it location/session/macro context) and the
    per-style floor correctly declines.  This frame gives the
    scorer the playbook's actual setup: structure 26 / location
    14 / session 10 / macro 5 -> total ~65 >= the 50 floor."""
    rows = [
        (2052, 2058, 2044, 2046, 900), (2046, 2050, 2036, 2038, 920),
        (2038, 2042, 2028, 2030, 940), (2030, 2034, 2018, 2020, 960),
        (2020, 2024, 2008, 2010, 980), (2010, 2014, 1998, 2000, 1000),
        (2000, 2004, 1988, 1990, 1020), (1990, 1993, 1978, 1980, 1040),
        (1980, 1983, 1966, 1968, 1060), (1968, 1974, 1958, 1966, 1150),
        (1966, 1978, 1963, 1975, 1080), (1975, 1985, 1970, 1982, 1030),
        (1982, 1990, 1976, 1988, 1010), (1988, 1993, 1982, 1985, 960),
        (1985, 1995, 1984, 1993, 1020), (1993, 2001, 1989, 1999, 1050),
        (1999, 2004, 1993, 1996, 990), (1996, 2002, 1988, 1990, 960),
        (1990, 1994, 1979, 1981, 1010), (1981, 1990, 1972, 1986, 1060),
        (1986, 1996, 1983, 1994, 1030), (1994, 2002, 1990, 2000, 1050),
        (2000, 2006, 1994, 1998, 1000), (1998, 2003, 1991, 1993, 970),
        (1993, 2000, 1989, 1997, 1010), (1997, 2005, 1992, 2002, 1040),
        (2002, 2007, 1995, 1999, 990), (1999, 2003, 1986, 1990, 1020),
        (1990, 1998, 1989, 1996, 1030), (1996, 2004, 1993, 2001, 1050),
        (2001, 2008, 1996, 2004, 1020), (2004, 2009, 1998, 2001, 1000),
        (2001, 2006, 1995, 2003, 990), (2003, 2008, 1999, 2005, 1010),
        (2005, 2007, 1996, 1998, 1080), (1998, 2000, 1979, 1993, 1400),
        (1993, 2003, 1989, 2001, 1350), (2001, 2009, 1997, 2007, 1300),
        (2007, 2012, 2002, 2009, 1250), (2009, 2013, 2003, 2006, 1500),
    ]
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="h",
                        tz="UTC")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [float(r[4]) for r in rows]},
        index=idx)


def _run_path(runner, verdict=None, stats=None, frame=None):
    h1 = frame if frame is not None else _h1()
    runner._stack = TimeframeStack.from_frames({"H1": h1})
    stats = stats if stats is not None else {
        "signals": 0, "gated": 0, "executed": 0}
    runner._run_signal_path(h1, len(h1) - 1, [], 2.0,
                            verdict or _verdict(), stats)
    return stats


def _update(text, chat_id=OWNER, update_id=1):
    return {"update_id": update_id,
            "message": {"message_id": 1,
                        "chat": {"id": chat_id, "type": "private"},
                        "text": text}}


def _callback(data, chat_id=OWNER_INT, from_id=OWNER_INT,
              update_id=1, message_id=5, query_id="q1"):
    return {"update_id": update_id,
            "callback_query": {
                "id": query_id,
                "from": {"id": from_id},
                "message": {"message_id": message_id,
                            "chat": {"id": chat_id, "type": "private"}},
                "data": data}}


class FakeApi:
    """Captures Bot API calls; serves one queued update per poll."""

    def __init__(self, updates):
        self.updates = list(updates)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, method, payload):
        self.calls.append((method, payload))
        if method == "getUpdates":
            if self.updates:
                return {"result": [self.updates.pop(0)]}
            return {"result": []}
        return {"ok": True, "result": {}}


# ------------------------------------------------------- playbook wiring

def test_get_playbooks_instantiates_all_seventeen(tmp_path):
    runner = _runner(tmp_path)
    names = {pb.name for pb in runner._get_playbooks()}
    assert names == {"sweep_reversal", "fvg_midline", "trendline_retest",
                     "fvg_limit", "inversion", "killzone_open",
                     "judas_swing", "turtle_soup", "v_reversal",
                     "retracement_trap", "order_block", "qm_m5",
                     "range_sweep_reclaim", "range_qm",
                     "range_mean_reversion",
                     "range_compression_breakout",
                     # CHUNK 3: NarrativePlaybook is wired into the
                     # runner (fill -> management plan, close ->
                     # STATE 7), so it is the 17th playbook.
                     "narrative"}
    # A14: the default config carries NO execute list — None means
    # every evaluated playbook executes (the shadow tier is gone);
    # the formerly-shadow names are ordinary members of the set.
    assert runner.cfg.execute_playbooks is None
    assert all(runner._executes(n) for n in names)
    assert {"turtle_soup", "v_reversal", "retracement_trap",
            "order_block", "qm_m5"} <= names


def test_judas_swing_fires_live_into_real_order(tmp_path):
    runner = _runner(tmp_path)
    judas = [pb for pb in runner._get_playbooks()
             if pb.name == "judas_swing"][0]
    judas.live_signal = lambda *a, **k: [_judas_sig()]
    stats = _run_path(runner, frame=_judas_frame())
    # The judas signal cleared the score floor on its real geometry
    # and placed exactly one market order.  (Assertions are scoped
    # to judas's own alert_id: on realistic data other playbooks
    # legitimately act too — inversion parks a pending limit on
    # this frame — so raw order/executed totals are not judas's.)
    judas_orders = [o for o in runner.executor.orders
                    if o.alert_id.startswith("MT5-judas_swing-")]
    assert len(judas_orders) == 1
    assert stats["executed"] >= 1
    assert runner._signal_counts.get("judas_swing") == 1
    recs = [r for r in runner.journal.records()
            if r.get("type") == "signal"
            and r.get("playbook") == "judas_swing"]
    assert len(recs) == 1 and recs[0]["decision"] == "accepted"
    assert recs[0]["score_breakdown"]["total"] >= 50.0


def test_unshadowed_playbook_executes_kill_switch_disables(tmp_path):
    """A14 (replaces the shadow-tier pin): turtle_soup — a
    formerly-shadow playbook — routes through the same decision
    flow as everyone else and CAN order when its signal clears the
    score floor; no "[SHADOW]" note and no log-only "shadow"
    decision is emitted for it on the MT5 path.  The only opt-out
    is the per-playbook kill switch (disabled_playbooks)."""
    runner = _runner(tmp_path)
    turtle = [pb for pb in runner._get_playbooks()
              if pb.name == "turtle_soup"][0]
    turtle.live_signal = lambda *a, **k: [_turtle_sig()]
    stats = _run_path(runner, frame=_judas_frame())
    assert runner._signal_counts.get("turtle_soup") == 1
    turtle_orders = [o for o in runner.executor.orders
                     if o.alert_id.startswith("MT5-turtle_soup-")]
    assert len(turtle_orders) == 1
    assert stats["executed"] >= 1
    recs = [r for r in runner.journal.records()
            if r.get("type") == "signal"
            and r.get("playbook") == "turtle_soup"]
    assert len(recs) == 1 and recs[0]["decision"] == "accepted"
    notes = [r.get("text", "") for r in runner.journal.records()
             if r.get("type") == "note"]
    assert not any("[SHADOW] turtle_soup" in n for n in notes)
    # Kill switch: the same playbook, disabled, is not evaluated.
    runner2 = _runner(tmp_path)
    runner2.cfg.disabled_playbooks = ("turtle_soup",)
    turtle2 = [pb for pb in runner2._get_playbooks()
               if pb.name == "turtle_soup"][0]
    turtle2.live_signal = lambda *a, **k: [_turtle_sig()]
    _run_path(runner2, frame=_judas_frame())
    assert runner2._signal_counts.get("turtle_soup") is None
    assert not any(o.alert_id.startswith("MT5-turtle_soup-")
                   for o in runner2.executor.orders)


def test_disabled_playbook_not_evaluated(tmp_path):
    runner = _runner(tmp_path)
    runner.cfg.disabled_playbooks = ("judas_swing",)
    judas = [pb for pb in runner._get_playbooks()
             if pb.name == "judas_swing"][0]
    judas.live_signal = lambda *a, **k: [_judas_sig()]
    _run_path(runner)
    assert runner.executor.orders == []
    assert runner._signal_counts.get("judas_swing") is None


# ------------------------------------------------------- blocked-leg stats

def test_counter_bias_signal_scored_by_floor_counted_in_status(tmp_path):
    """Choke (a) removed: gate 0 no longer BLOCKS on the MT5 path.
    A counter-bias verdict (the legacy shape still carries
    allowed=False / blocked_leg='d1_bias'; D1 bias is AGAINST this
    long) does not gate the signal — the verdict is demoted to
    Macro CONTEXT.  The signal is scored like any other, the Macro
    component shows the disagreement (aligned 0.5: D1 against, H4
    with), and the per-style FLOOR makes the decision — here the
    flat harness data totals 42 < 50, so the decline is booked
    under the floor leg, never under a hierarchy leg."""
    runner = _runner(tmp_path)
    judas = [pb for pb in runner._get_playbooks()
             if pb.name == "judas_swing"][0]
    judas.live_signal = lambda *a, **k: [_judas_sig()]
    blocked = _verdict(allowed=False, blocked_leg="d1_bias",
                       bias_d1=-1,
                       legs={"d1_bias": {"passed": False,
                                         "reason": "D1 bias down"}})
    _run_path(runner, verdict=blocked)
    assert runner.executor.orders == []
    legs = runner.blocked_stats()["legs"]
    assert legs.get("d1_bias") is None     # gate 0 counts nothing
    assert legs.get("min_score") == 1      # the floor decided
    recs = [r for r in runner.journal.records()
            if r.get("type") == "signal"
            and r.get("playbook") == "judas_swing"]
    assert len(recs) == 1
    assert recs[0]["decision"] == "below_floor"
    assert recs[0]["reason"].startswith("asymmetry_floor")
    breakdown = recs[0]["score_breakdown"]
    assert breakdown["total"] < 50.0
    # The counter-bias is visible as scored context, not a veto:
    # macro aligned 0.5 (D1 -1 against the long, H4 +1 with it).
    assert breakdown["macro"]["subs"]["aligned"] == 0.5
    ctx = CommandContext(journal=runner.journal, mt5_control=runner)
    res, _ = mt5_router(OWNER).dispatch(_update("/status"), ctx)
    assert "why no trades" in res.text
    assert "min_score" in res.text         # the floor leg, counted
    assert "judas_swing" in res.text       # last-blocked names it
    assert "d1_bias" not in res.text       # no hierarchy leg exists
    assert res.reply_markup == mt5_control_keyboard()


def test_pause_blocks_orders_and_counts_paused_leg(tmp_path):
    runner = _runner(tmp_path)
    runner.cfg.paused = True
    judas = [pb for pb in runner._get_playbooks()
             if pb.name == "judas_swing"][0]
    judas.live_signal = lambda *a, **k: [_judas_sig()]
    _run_path(runner)
    assert runner.executor.orders == []
    assert runner.blocked_stats()["legs"].get("paused") == 1


# ------------------------------------------------------- config get / set

def test_config_shows_safe_keys_only(tmp_path):
    runner = _runner(tmp_path)
    ctx = CommandContext(journal=runner.journal, mt5_control=runner)
    res, _ = mt5_router(OWNER).dispatch(_update("/config"), ctx)
    assert "fixed_lots" in res.text and "max_layers" in res.text
    assert "manipulation_filter" in res.text
    assert "pending_expiry_bars" in res.text
    assert "judas_swing: on" in res.text
    for banned in ("token", "secret", "password", "allow_live",
                   "GS_TELEGRAM", "api_key", "login"):
        assert banned not in res.text.lower()


def _set_and_confirm(runner, text):
    """Dispatch /set, extract the Confirm token, confirm it."""
    ctx = CommandContext(journal=runner.journal, mt5_control=runner)
    res, _ = mt5_router(OWNER).dispatch(_update(text), ctx)
    assert res.reply_markup is not None
    data = res.reply_markup["inline_keyboard"][0][0]["callback_data"]
    action, token = parse_mt5_callback(data)
    assert action == "confirm"
    ok, msg = runner.confirm_pending(token)
    assert ok, msg
    return msg


def test_set_valid_applies_and_persists(tmp_path):
    runner = _runner(tmp_path)
    _set_and_confirm(runner, "/set fixed_lots 0.05")
    assert runner.cfg.fixed_lots == 0.05
    stored = json.loads((tmp_path / "mt5_config.json").read_text())
    assert stored["fixed_lots"] == 0.05
    # Simulated restart: a fresh runner loads the persisted value.
    runner2 = _runner(tmp_path)
    assert runner2.cfg.fixed_lots == 0.05


def test_set_playbook_and_filter_persist(tmp_path):
    runner = _runner(tmp_path)
    _set_and_confirm(runner, "/set playbook.judas_swing off")
    assert "judas_swing" in runner.cfg.disabled_playbooks
    _set_and_confirm(runner, "/set manipulation_filter off")
    assert runner.cfg.manipulation_filter is False
    _set_and_confirm(runner, "/set pending_expiry_bars 12")
    assert runner.cfg.pending_expiry_bars == 12
    _set_and_confirm(runner, "/set max_layers 1")
    assert runner.cfg.layer_max_layers == 1
    runner2 = _runner(tmp_path)
    assert "judas_swing" in runner2.cfg.disabled_playbooks
    assert runner2.cfg.manipulation_filter is False
    assert runner2.cfg.pending_expiry_bars == 12
    assert runner2.cfg.layer_max_layers == 1


def test_set_rejects_invalid_and_dangerous(tmp_path):
    runner = _runner(tmp_path)
    ctx = CommandContext(journal=runner.journal, mt5_control=runner)
    router = mt5_router(OWNER)
    cases = [
        "/set fixed_lots 5",          # out of range
        "/set fixed_lots abc",         # not a number
        "/set max_layers 9",           # out of range
        "/set pending_expiry_bars 0",  # out of range
        "/set manipulation_filter maybe",
        "/set playbook.bogus on",      # no such playbook
        "/set allow_live true",        # live-trading permission: never
        "/set token FAKESECRET123",    # credential key: never
        "/set pause on",               # pause is not a /set key
        "/set",                        # usage
    ]
    for text in cases:
        res, _ = router.dispatch(_update(text), ctx)
        assert "rejected" in res.text or "usage" in res.text, text
        assert res.reply_markup is None  # no confirm staged
        assert "FAKESECRET123" not in res.text  # value never echoed
    assert runner.cfg.fixed_lots == 0.01
    assert runner.cfg.disabled_playbooks == ()
    assert runner.cfg.paused is False


# ------------------------------------------------------- confirm flow

def _poll_with(runner, api):
    import os
    # The router reads GS_TELEGRAM_CHAT_ID per call; restore afterwards
    # so the env does not leak into other tests in the same process.
    old = os.environ.get("GS_TELEGRAM_CHAT_ID")
    os.environ["GS_TELEGRAM_CHAT_ID"] = OWNER
    try:
        runner.senders = [FakeSender()]
        runner._tg_api = api
        runner._tg_synced = True
        runner._tg_offset = 0
        runner._last_poll = 0.0
        runner._poll_telegram_commands()
    finally:
        if old is None:
            os.environ.pop("GS_TELEGRAM_CHAT_ID", None)
        else:
            os.environ["GS_TELEGRAM_CHAT_ID"] = old


def test_pause_requires_confirm_tap(tmp_path):
    runner = _runner(tmp_path)
    api = FakeApi([_callback("mt5:pause")])
    _poll_with(runner, api)
    # Pause button only staged the confirmation — nothing acted.
    assert runner.cfg.paused is False
    assert len(runner._confirm_tokens) == 1
    token = next(iter(runner._confirm_tokens))
    sends = [p for m, p in api.calls if m == "sendMessage"]
    assert sends and "Confirm" in sends[0]["text"]
    assert any(m == "answerCallbackQuery" for m, _ in api.calls)
    # Cancel aborts: still not paused.
    api2 = FakeApi([_callback(f"mt5:cancel:{token}")])
    _poll_with(runner, api2)
    assert runner.cfg.paused is False
    assert runner._confirm_tokens == {}
    edits = [p for m, p in api2.calls if m == "editMessageText"]
    assert edits and "cancelled" in edits[0]["text"].lower()


def test_confirm_tap_executes_pause_then_resume(tmp_path):
    runner = _runner(tmp_path)
    api = FakeApi([_callback("mt5:pause")])
    _poll_with(runner, api)
    token = next(iter(runner._confirm_tokens))
    api2 = FakeApi([_callback(f"mt5:confirm:{token}")])
    _poll_with(runner, api2)
    assert runner.cfg.paused is True
    edits = [p for m, p in api2.calls if m == "editMessageText"]
    assert edits and "PAUSED" in edits[0]["text"]
    # Resume: same two-step dance.
    api3 = FakeApi([_callback("mt5:resume")])
    _poll_with(runner, api3)
    assert runner.cfg.paused is True  # still paused: no Confirm yet
    token2 = next(iter(runner._confirm_tokens))
    api4 = FakeApi([_callback(f"mt5:confirm:{token2}")])
    _poll_with(runner, api4)
    assert runner.cfg.paused is False


def test_trim_button_trims_only_on_confirm(tmp_path):
    from goldenswing.execution.executor import OrderRequest
    runner = _runner(tmp_path)
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=0.01,
                       stop_loss=1990.0, entry_ref=2000.0,
                       targets=[2030.0], atr=10.0, bar_index=5)
    runner.tick_manager.open(ticket=101, request=req, fill_price=2000.0,
                             entry_bar=6)
    runner.tick_manager.add_layer(101, entry=2010.0, volume=0.01,
                                  sl=2000.0, alert_id="x",
                                  playbook="judas_swing")
    api = FakeApi([_callback("mt5:trim")])
    _poll_with(runner, api)
    assert runner.executor.closes == []  # no Confirm yet
    token = next(iter(runner._confirm_tokens))
    api2 = FakeApi([_callback(f"mt5:confirm:{token}")])
    _poll_with(runner, api2)
    assert runner.executor.closes == [(101, 0.01)]
    edits = [p for m, p in api2.calls if m == "editMessageText"]
    assert edits and "trimmed" in edits[0]["text"].lower()


def test_cancel_pending_button_cancels_only_on_confirm(tmp_path):
    from goldenswing.execution.executor import PendingOrder
    runner = _runner(tmp_path)
    order = PendingOrder(ticket=777, symbol="XAUUSD", kind="limit",
                         direction=1, price=2000.0, volume=0.01,
                         sl=1990.0, tp=2030.0, magic=260101,
                         alert_id="MT5-judas_swing-5")
    runner._track_pending_order(
        order, playbook="judas_swing", zone=None, invalidation=1990.0,
        placed_bar=0, sig=None, verdict=None, track_request=None)
    api = FakeApi([_callback("mt5:cancel_pending")])
    _poll_with(runner, api)
    assert runner.executor.cancels == []  # no Confirm yet
    token = next(iter(runner._confirm_tokens))
    api2 = FakeApi([_callback(f"mt5:confirm:{token}")])
    _poll_with(runner, api2)
    assert runner.executor.cancels == [777]
    assert runner._pending_orders == {}


def test_refresh_button_edits_message_with_fresh_status(tmp_path):
    runner = _runner(tmp_path)
    runner.senders = [FakeSender()]
    api = FakeApi([_callback("mt5:refresh")])
    _poll_with(runner, api)
    edits = [p for m, p in api.calls if m == "editMessageText"]
    assert edits and "MT5 ENGINE STATUS" in edits[0]["text"]
    assert "reply_markup" in edits[0]
    assert any(m == "answerCallbackQuery" for m, _ in api.calls)


def test_poller_routes_callback_query_updates(tmp_path, monkeypatch):
    monkeypatch.setenv("GS_TELEGRAM_CHAT_ID", OWNER)
    runner = _runner(tmp_path)
    runner.senders = [FakeSender()]
    api = FakeApi([_callback("mt5:refresh")])
    runner._tg_api = api
    runner._tg_synced = True
    runner._tg_offset = 0
    runner._last_poll = 0.0
    runner._poll_telegram_commands()
    assert any(m == "editMessageText" for m, _ in api.calls)
    assert any(m == "answerCallbackQuery" for m, _ in api.calls)


# ------------------------------------------------------- owner-only

def test_non_owner_callback_refused_and_acts_on_nothing(tmp_path):
    runner = _runner(tmp_path)
    api = FakeApi([_callback("mt5:pause", from_id=99999,
                             chat_id=99999)])
    _poll_with(runner, api)
    assert runner.cfg.paused is False
    assert runner._confirm_tokens == {}
    answers = [p for m, p in api.calls if m == "answerCallbackQuery"]
    assert answers and "not authorized" in answers[0]["text"]
    assert not any(m == "sendMessage" for m, _ in api.calls)


def test_non_owner_command_silently_ignored(tmp_path):
    runner = _runner(tmp_path)
    ctx = CommandContext(journal=runner.journal, mt5_control=runner)
    res, chat = mt5_router(OWNER).dispatch(
        _update("/set fixed_lots 0.05", chat_id="99999"), ctx)
    assert res is None and chat is None
    assert runner.cfg.fixed_lots == 0.01


def test_non_owner_status_gets_nothing(tmp_path):
    ctx = CommandContext(journal=None)
    res, chat = mt5_router(OWNER).dispatch(
        _update("/status", chat_id="99999"), ctx)
    assert res is None


# ------------------------------------------------------- no secret leakage

def test_no_command_output_leaks_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("GS_TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("GS_TELEGRAM_CHAT_ID", OWNER)
    runner = _runner(tmp_path)
    ctx = CommandContext(journal=runner.journal, mt5_control=runner,
                         live_positions=[],
                         pending_orders=[])
    router = mt5_router(OWNER)
    outputs = []
    for text in ("/status", "/config", "/position", "/pending",
                 "/trades", "/help"):
        res, _ = router.dispatch(_update(text), ctx)
        outputs.append(res.text or "")
    blob = "\n".join(outputs)
    assert FAKE_TOKEN not in blob
    assert "123456:" not in blob
    for banned in ("GS_TELEGRAM_BOT_TOKEN", "GS_MT5_PASSWORD",
                   "password", "secret", "api_key"):
        assert banned not in blob.lower()


def test_callback_api_payloads_never_carry_token(tmp_path, monkeypatch):
    monkeypatch.setenv("GS_TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("GS_TELEGRAM_CHAT_ID", OWNER)
    runner = _runner(tmp_path)
    api = FakeApi([_callback("mt5:refresh"), _callback("mt5:config")])
    _poll_with(runner, api)
    blob = json.dumps(api.calls)
    assert FAKE_TOKEN not in blob
    assert "GS_TELEGRAM_BOT_TOKEN" not in blob


# ------------------------------------------------------- misc commands

def test_position_and_pending_commands_carry_buttons(tmp_path):
    runner = _runner(tmp_path)
    ctx = CommandContext(journal=runner.journal, mt5_control=runner,
                         pending_orders=[
                             {"ticket": 777, "kind": "limit",
                              "direction": 1, "level": 2000.0,
                              "volume": 0.01, "sl": 1990.0,
                              "playbook": "judas_swing",
                              "expiry_bar": 12}])
    router = mt5_router(OWNER)
    res, _ = router.dispatch(_update("/position"), ctx)
    assert "OPEN POSITIONS" in res.text
    assert res.reply_markup == mt5_control_keyboard()
    res, _ = router.dispatch(_update("/pending"), ctx)
    assert "#777" in res.text and "LIMIT" in res.text
    assert res.reply_markup == mt5_control_keyboard()


def test_pause_resume_cancel_commands_stage_confirm(tmp_path):
    runner = _runner(tmp_path)
    ctx = CommandContext(journal=runner.journal, mt5_control=runner)
    router = mt5_router(OWNER)
    for cmd in ("/pause", "/resume", "/trim", "/cancel_pending"):
        res, _ = router.dispatch(_update(cmd), ctx)
        assert res.reply_markup is not None, cmd
        assert "Confirm" in res.text
        assert runner._confirm_tokens  # staged, not acted
    assert runner.cfg.paused is False


def test_confirm_token_is_single_use(tmp_path):
    runner = _runner(tmp_path)
    token = runner.request_confirm("pause")
    ok, _ = runner.confirm_pending(token)
    assert ok and runner.cfg.paused is True
    ok2, msg = runner.confirm_pending(token)
    assert not ok2 and "expired" in msg


def test_set_without_live_runner_is_unavailable(tmp_path):
    ctx = CommandContext(journal=None)  # no mt5_control attached
    res, _ = mt5_router(OWNER).dispatch(_update("/set fixed_lots 0.05"),
                                        ctx)
    assert "unavailable" in res.text
    res, _ = mt5_router(OWNER).dispatch(_update("/pause"), ctx)
    assert "unavailable" in res.text


# ------------------------------------------------- persistent reply menu

def test_start_and_menu_show_persistent_keyboard(tmp_path):
    from goldenswing.execution.mt5_commands import MENU_KEYBOARD
    runner = _runner(tmp_path)
    ctx = CommandContext(journal=runner.journal, mt5_control=runner)
    router = mt5_router(OWNER)
    for cmd in ("/start", "/menu"):
        res, _ = router.dispatch(_update(cmd), ctx)
        assert res.reply_markup == MENU_KEYBOARD
        kb = res.reply_markup
        assert kb["is_persistent"] is True
        assert kb["resize_keyboard"] is True
        assert kb["one_time_keyboard"] is False
        rows = [[b["text"] for b in row] for row in kb["keyboard"]]
        # Septorch-style: bottom-menu buttons carry emoji labels.
        assert rows == [["📊 Status", "📈 Positions", "⏳ Pendings"],
                        ["🎯 Playbooks", "⚙️ Config", "📊 Trades"],
                        ["📦 Harvest", "⏸️ Pause", "▶️ Resume"]]


def test_menu_buttons_route_to_handlers(tmp_path):
    runner = _runner(tmp_path)
    ctx = CommandContext(journal=runner.journal, mt5_control=runner,
                         pending_orders=[])
    router = mt5_router(OWNER)
    cases = {
        "Status": "MT5 ENGINE STATUS",
        "status": "MT5 ENGINE STATUS",  # case-insensitive
        "Positions": "MT5 OPEN POSITIONS",
        "Pendings": "MT5 PENDING ORDERS",
        "Playbooks": "judas_swing",
        "Config": "MT5 LIVE CONFIG",
        "Trades": "no closed MT5 trades yet",
    }
    for label, needle in cases.items():
        res, _ = router.dispatch(_update(label), ctx)
        assert res is not None and needle in (res.text or ""), label
    # Pause / Resume buttons stage the same confirm flow as /pause.
    res, _ = router.dispatch(_update("Pause"), ctx)
    assert res.reply_markup is not None and "Confirm" in res.text
    assert runner._confirm_tokens
    res, _ = router.dispatch(_update("Resume"), ctx)
    assert res.reply_markup is not None and "Confirm" in res.text
    # Harvest button runs the same handler as /harvest.
    res, _ = router.dispatch(_update("Harvest"), ctx)
    assert res.document is not None


def test_menu_buttons_owner_only(tmp_path):
    runner = _runner(tmp_path)
    ctx = CommandContext(journal=runner.journal, mt5_control=runner)
    for label in ("Status", "Pause", "Harvest", "Config"):
        res, chat = mt5_router(OWNER).dispatch(
            _update(label, chat_id="99999"), ctx)
        assert res is None and chat is None, label
    assert runner._confirm_tokens == {}


def test_menu_button_via_poller_is_answered(tmp_path):
    runner = _runner(tmp_path)
    api = FakeApi([{"update_id": 1,
                    "message": {"message_id": 1,
                                "chat": {"id": OWNER_INT,
                                         "type": "private"},
                                "text": "Status"}}])
    _poll_with(runner, api)
    sender = runner.senders[0]
    assert sender.texts and "MT5 ENGINE STATUS" in sender.texts[0]


def test_playbooks_command_shows_state_and_counts(tmp_path):
    runner = _runner(tmp_path)
    runner._signal_counts["judas_swing"] = 3
    runner.cfg.disabled_playbooks = ("turtle_soup",)
    ctx = CommandContext(journal=runner.journal, mt5_control=runner)
    res, _ = mt5_router(OWNER).dispatch(_update("/playbooks"), ctx)
    assert "judas_swing: executing, on, signals 3" in res.text
    assert "turtle_soup: shadow, off, signals 0" in res.text
    assert "qm_m5: shadow, on" in res.text
    assert res.reply_markup == mt5_control_keyboard()


def test_reply_menu_and_inline_buttons_coexist(tmp_path):
    """Two surfaces, two messages: /menu sends the reply keyboard,
    /status sends inline buttons — neither clobbers the other."""
    from goldenswing.execution.mt5_commands import MENU_KEYBOARD
    runner = _runner(tmp_path)
    sender = FakeSender()
    runner.senders = [sender]
    ctx = CommandContext(journal=runner.journal, mt5_control=runner)
    router = mt5_router(OWNER)
    menu_res, _ = router.dispatch(_update("/menu"), ctx)
    runner._reply_command(menu_res)
    status_res, _ = router.dispatch(_update("/status"), ctx)
    runner._reply_command(status_res)
    assert sender.markups[0] == MENU_KEYBOARD
    assert "inline_keyboard" in (sender.markups[1] or {})
