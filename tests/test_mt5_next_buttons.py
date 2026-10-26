"""Next-action inline buttons (owner note 2026-10-26).

Every ``➡️ Next:`` text line on a live MT5 message is now a real
inline keyboard: one button per next command (``mt5:cmd:<command>``),
a ``« Back to Status`` row unless ``/status`` is already among them,
and the hub model — tapping re-renders in place for text results.
Text formatting is untouched by construction: the conversion only
deletes the trailing Next line and attaches a keyboard; bodies,
section headers, code spans, emoji, and progress bars survive
byte-identical, and the send/edit path is the same one the nav
buttons already used.

Scope boundary, pinned: the legacy OANDA/paper command set in
``alerts.commands`` keeps its text CTAs (dead paper path — the
paper loops were stopped by owner order 2026-10-23). Only the
shared functions the live MT5 bot delegates to (backtest ack,
snapshot, reconcile) converted, plus every MT5 handler.
"""

import json
from datetime import datetime, timezone

from goldenswing.alerts.commands import (
    BacktestJob,
    CommandContext,
    _finish_backtest_job,
    _reconcile,
    _snapshot,
    build_backtest_export_result,
    fmt_next_keyboard,
    next_button_rows,
    start_backtest_from_args,
)
from goldenswing.execution import mt5_commands as _mc
from goldenswing.execution.mt5_commands import (
    MENU_KEYBOARD,
    _mt5_assess,
    _mt5_config,
    _mt5_d1,
    _mt5_db,
    _mt5_harvest,
    _mt5_health,
    _mt5_help,
    _mt5_journal,
    _mt5_marketread,
    _mt5_pending,
    _mt5_playbooks,
    _mt5_position,
    _mt5_recap,
    _mt5_report,
    _mt5_start,
    _mt5_status,  # noqa: F401  (imported for registry completeness)
    _mt5_ticks,
    _mt5_trades,
    mt5_back_keyboard,
    mt5_control_keyboard,
    mt5_control_with_next,
    mt5_nav_keyboard,
    mt5_nav_with_next,
)
from test_mt5_nav_buttons import _Journal, _record
from test_mt5_telegram_control import OWNER, _callback, _runner, FakeApi
from test_telegram_formatting import _journal, _mt5_ctx, _mt5_text


def _datas(kb):
    return [b["callback_data"] for row in kb["inline_keyboard"]
            for b in row]


def _texts(kb):
    return [b["text"] for row in kb["inline_keyboard"] for b in row]


class _Ctrl:
    def get_live_config(self):
        return {}

    def blocked_stats(self):
        return {}

    def market_read_line(self):
        return "trend: flat (test stub)"


def _trade_journal():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    open_rec = {"type": "open", "alert_id": "a1", "playbook": "qm_m5",
                "direction": 1, "score": 80, "entry": 4000.0,
                "stop": 3990.0, "volume": 0.01, "risk_money": 10.0,
                "opened_ts": now}
    close_rec = {"type": "close", "alert_id": "a1", "exit": 4010.0,
                 "reason": "tp", "pnl_money": 10.0, "r_net": 1.0,
                 "hold_bars": 5, "closed_ts": now}
    return _Journal([open_rec, close_rec])


# ------------------------------------------------------------ helpers

def test_next_keyboard_rows_shape_and_back_rule():
    kb = fmt_next_keyboard("/status", "/pending", "/trades")
    assert _datas(kb) == ["mt5:cmd:/status", "mt5:cmd:/pending",
                          "mt5:cmd:/trades"]
    # /status present -> no Back row (the status screen IS the hub).
    assert "« Back to Status" not in _texts(kb)
    kb = fmt_next_keyboard("/playbooks", "/health", "/trades")
    rows = kb["inline_keyboard"]
    assert len(rows) == 3  # 2 + 1 buttons, then the Back row
    assert rows[-1] == [{"text": "« Back to Status",
                         "callback_data": "mt5:cmd:/status"}]
    assert _datas(kb) == ["mt5:cmd:/playbooks", "mt5:cmd:/health",
                          "mt5:cmd:/trades", "mt5:cmd:/status"]


def test_next_keyboard_placeholder_mapping_dedupe_and_cap():
    kb = fmt_next_keyboard("/assess <playbook>", "/assess all", "/status",
                           "/status")
    assert _datas(kb) == ["mt5:cmd:/assess", "mt5:cmd:/assess all",
                          "mt5:cmd:/status"]
    for data in _datas(fmt_next_keyboard("/status", "/position",
                                          "/cancel_pending", "/playbooks",
                                          "/health", "/trades", "/report",
                                          "/recap", "/snapshot", "/reconcile",
                                          "/journal", "/assess all")):
        assert len(data) <= 64  # Telegram callback_data cap


def test_control_and_nav_merges_keep_base_rows():
    merged = mt5_control_with_next("/playbooks", "/health", "/trades")
    assert merged["inline_keyboard"][:3] == \
        mt5_control_keyboard()["inline_keyboard"]
    assert "mt5:cmd:/trades" in _datas(merged)
    assert "« Back to Status" in _texts(merged)
    nav = mt5_nav_with_next("/status", "/reconcile", "/journal")
    assert nav["inline_keyboard"][:2] == \
        mt5_nav_keyboard()["inline_keyboard"]
    assert "mt5:cmd:/reconcile" in _datas(nav)
    # Nav rows already carry Status -> no extra Back row.
    assert "« Back to Status" not in _texts(nav)
    assert mt5_back_keyboard()["inline_keyboard"] == [
        [{"text": "« Back to Status", "callback_data": "mt5:cmd:/status"}]]


def test_document_results_carry_next_keyboards(tmp_path):
    """Audit check 3: document results used to go out with zero
    buttons on both paths.  Every file result now carries the same
    kind of Next keyboard as the text screens — and never a button
    to its own command."""
    j = _journal(tmp_path)
    ctx = _mt5_ctx(j, tmp_path)
    res = _mt5_db(ctx, [])
    assert res.document is not None
    assert _datas(res.reply_markup) == ["mt5:cmd:/status",
                                        "mt5:cmd:/journal",
                                        "mt5:cmd:/report"]
    res = _mt5_report(ctx, [])
    assert res.document is not None
    assert _datas(res.reply_markup) == ["mt5:cmd:/status",
                                        "mt5:cmd:/trades",
                                        "mt5:cmd:/recap"]
    # Long weekly recap goes out as a file, not text — same buttons
    # as the short-recap text branch.
    for i in range(80):
        j.log_open(alert_id=f"MT5-x-{i}", instrument="XAUUSD",
                   mode="MT5", playbook="x", phase="live", direction=1,
                   score=70, source="mt5", ticket=100 + i, entry=4000.0,
                   stop=3990.0, volume=0.01, risk_money=10.0)
        j.log_close(alert_id=f"MT5-x-{i}", ticket=100 + i,
                    exit_price=4010.0, reason="tp", pnl_money=10.0,
                    r_net=1.0, hold_bars=3)
    res = _mt5_recap(ctx, ["weekly"])
    assert res.document is not None
    assert _datas(res.reply_markup) == ["mt5:cmd:/status",
                                        "mt5:cmd:/trades",
                                        "mt5:cmd:/report"]
    res = _mt5_harvest(ctx, ["journal"])
    assert res.document is not None
    assert _datas(res.reply_markup) == ["mt5:cmd:/status",
                                        "mt5:cmd:/trades",
                                        "mt5:cmd:/report"]
    res = build_backtest_export_result(ctx)
    assert res.document is not None
    assert _datas(res.reply_markup) == ["mt5:cmd:/status",
                                        "mt5:cmd:/backtest",
                                        "mt5:cmd:/report"]


def test_feed_export_documents_carry_next_keyboards(tmp_path, monkeypatch):
    """Ticks + D1 exports are feed-gated; stub the feed edge only,
    the handler's document + keyboard code runs for real."""
    j = _journal(tmp_path)
    ctx = _mt5_ctx(j, tmp_path)

    def _fake_export(feed, start, end, out_path):
        out_path.write_text("time,bid,ask\n")
        return out_path, 123

    monkeypatch.setattr(_mc, "export_tick_history", _fake_export)
    ctx.feed = object()
    res = _mt5_ticks(ctx, ["1"])
    assert res.document is not None
    assert _datas(res.reply_markup) == ["mt5:cmd:/status",
                                        "mt5:cmd:/marketread"]

    import pandas as pd
    from datetime import timezone
    idx = pd.date_range("2026-10-01", periods=5, freq="D",
                        tz=timezone.utc)
    frame = pd.DataFrame(
        {"open": 4000.0, "high": 4010.0, "low": 3990.0,
         "close": 4005.0, "volume": 100, "real_volume": 100,
         "spread": 20}, index=idx)

    class _FakeFeed:
        def bars(self, tf, n):
            return frame

    ctx.feed = _FakeFeed()
    res = _mt5_d1(ctx, ["5"])
    assert res.document is not None
    assert _datas(res.reply_markup) == ["mt5:cmd:/status",
                                        "mt5:cmd:/marketread"]


def test_async_backtest_delivery_carries_next_keyboard(tmp_path):
    """The async backtest report is delivered via ctx.deliver — it
    used to arrive with no buttons either."""
    import goldenswing.alerts.commands as _ac

    j = _journal(tmp_path)
    ctx = _mt5_ctx(j, tmp_path)
    got = []
    ctx.deliver = got.append
    out = tmp_path / "reports"
    out.mkdir(parents=True)
    old = _ac._BACKTEST_RUNNER_OVERRIDE
    _ac._BACKTEST_RUNNER_OVERRIDE = lambda job: {
        "window": {},
        "trades": {"closed": 2, "win_rate": 0.5, "total_r": 1.0,
                   "total_dollar": 20.0, "profit_factor": 1.5,
                   "max_drawdown_r": 0.5},
        "per_playbook": {}}
    try:
        job = BacktestJob(days=7, playbook=None, drop_back=False,
                          out_dir=out)
        _finish_backtest_job(job, ctx, False)
    finally:
        _ac._BACKTEST_RUNNER_OVERRIDE = old
    assert job.status == "done"
    res = got[0]
    assert res.document is not None
    assert _datas(res.reply_markup) == ["mt5:cmd:/backtest_export",
                                        "mt5:cmd:/status",
                                        "mt5:cmd:/health"]


def test_status_screen_never_shows_back_to_itself():
    """Owner complaint 2026-10-26: the status page carried a
    « Back to Status » row pointing at itself.  A screen never
    shows a button to itself: the hub screen's merge passes
    back=False, and the /status result itself carries no
    self-Back button.  The default merge (back=True) is
    unchanged — other screens still get their Back row."""
    merged = mt5_control_with_next("/playbooks", "/health",
                                   "/trades", back=False)
    assert "« Back to Status" not in _texts(merged)
    assert "mt5:cmd:/playbooks" in _datas(merged)
    res = _mt5_status(CommandContext(feed_status={}), [])
    kb = res.reply_markup or {}
    assert "« Back to Status" not in _texts(kb)
    assert "mt5:cmd:/playbooks" in _datas(kb)
    assert "mt5:cmd:/health" in _datas(kb)
    assert "mt5:cmd:/trades" in _datas(kb)


# ------------------------------------------------------------ MT5 handlers

def test_start_drops_next_line_keeps_menu():
    res = _mt5_start(CommandContext(), [])
    assert "➡️ Next:" not in res.text
    assert res.reply_markup == MENU_KEYBOARD


def test_help_playbooks_config_carry_next_buttons():
    ctx = CommandContext(mt5_control=_Ctrl())
    res = _mt5_help(ctx, [])
    assert "➡️ Next:" not in res.text
    assert _datas(res.reply_markup) == [
        "mt5:cmd:/status", "mt5:cmd:/playbooks", "mt5:cmd:/health"]
    res = _mt5_playbooks(ctx, [])
    assert "➡️ Next:" not in res.text
    datas = _datas(res.reply_markup)
    assert "mt5:pause" in datas and "mt5:cmd:/set" in datas
    res = _mt5_config(ctx, [])
    assert "➡️ Next:" not in res.text
    datas = _datas(res.reply_markup)
    assert "mt5:pause" in datas and "mt5:cmd:/set" in datas


def test_marketread_merged_keyboard_and_no_runner_branch():
    res = _mt5_marketread(CommandContext(mt5_control=_Ctrl()), [])
    assert "➡️ Next:" not in res.text
    datas = _datas(res.reply_markup)
    assert "mt5:pause" in datas and "mt5:cmd:/assess" in datas
    res = _mt5_marketread(CommandContext(), [])
    assert "needs the live MT5 engine" in res.text
    assert "➡️ Next:" not in res.text


def test_assess_branches_carry_merged_nav_keyboards(tmp_path):
    ctx = CommandContext(journal=_Journal([_record("short read")]))
    index = _mt5_assess(ctx, [])
    assert "➡️ Next:" not in index.text
    assert index.reply_markup == mt5_nav_with_next(
        "/assess <playbook>", "/assess all", "/status")
    single = _mt5_assess(ctx, ["sweep_reversal"])
    assert "➡️ Next:" not in single.text
    assert single.reply_markup == mt5_nav_with_next("/assess all", "/status")
    all_text = _mt5_assess(ctx, ["all"])
    assert "➡️ Next:" not in all_text.text
    assert all_text.reply_markup == mt5_nav_with_next("/assess", "/status")


def test_position_pending_trades_journal_health():
    ctx = CommandContext(live_positions=[], pending_orders=[])
    res = _mt5_position(ctx, [])
    assert "➡️ Next:" not in res.text
    assert "MT5 OPEN POSITIONS" in res.text
    datas = _datas(res.reply_markup)
    assert "mt5:pause" in datas and "mt5:cmd:/trades" in datas
    res = _mt5_pending(ctx, [])
    assert "➡️ Next:" not in res.text
    assert "MT5 PENDING ORDERS" in res.text
    assert "mt5:cmd:/cancel_pending" in _datas(res.reply_markup)
    res = _mt5_trades(CommandContext(journal=_trade_journal()), [])
    assert "➡️ Next:" not in res.text
    assert "qm_m5" in res.text and "+1.00R" in res.text
    assert _datas(res.reply_markup) == [
        "mt5:cmd:/status", "mt5:cmd:/report", "mt5:cmd:/recap"]
    res = _mt5_recap(CommandContext(journal=_trade_journal(),
                                    reports_dir=None), [])
    assert "➡️ Next:" not in res.text
    assert "mt5:cmd:/trades" in _datas(res.reply_markup)
    res = _mt5_journal(CommandContext(journal=_Journal([{"seq": 1}]),
                                      journal_path="/tmp/j.jsonl"), [])
    assert "➡️ Next:" not in res.text
    assert _datas(res.reply_markup) == [
        "mt5:cmd:/snapshot", "mt5:cmd:/health", "mt5:cmd:/status"]
    res = _mt5_health(CommandContext(feed_status={}, journal=None,
                                     live_positions=[], pending_orders=[]),
                      [])
    assert "➡️ Next:" not in res.text
    assert res.reply_markup == mt5_nav_with_next(
        "/status", "/reconcile", "/journal")


# ------------------------------------------------------------ shared (delegated)

def test_backtest_ack_snapshot_reconcile_carry_buttons(tmp_path, monkeypatch):
    import goldenswing.alerts.commands as ac
    monkeypatch.setattr(ac, "start_backtest_job",
                        lambda *a, **k: (object(), None))
    monkeypatch.setattr(ac, "_d1_coverage", lambda ctx: ([], False))
    res = start_backtest_from_args(CommandContext(), ["5"],
                                   drop_back=False)
    assert "➡️ Next:" not in res.text
    assert _datas(res.reply_markup) == [
        "mt5:cmd:/backtest_export", "mt5:cmd:/status", "mt5:cmd:/health"]
    engine = tmp_path / "engine"
    engine.mkdir()
    (engine / "manifest.json").write_text(json.dumps({"snapshots": [
        {"watermark": 5, "snapshot_root": "abcdef1234567890",
         "chain_n": 3}]}))
    res = _snapshot(CommandContext(journal_path=str(tmp_path / "j.jsonl")),
                    [])
    assert "➡️ Next:" not in res.text
    assert _datas(res.reply_markup) == [
        "mt5:cmd:/journal", "mt5:cmd:/health", "mt5:cmd:/status"]

    class Rec:
        phase = "live"
        reconciliation_complete = True
        holding = []

        def unresolved(self):
            return []

    res = _reconcile(CommandContext(reconciler=Rec()), [])
    assert "➡️ Next:" not in res.text
    assert _datas(res.reply_markup) == [
        "mt5:cmd:/health", "mt5:cmd:/journal", "mt5:cmd:/status"]


def test_legacy_alerts_handlers_keep_text_cta():
    # Scope boundary: the dead OANDA/paper set keeps its text lines.
    from goldenswing.alerts.commands import _help, _journal as _ajournal
    res = _help(CommandContext(), [])
    assert "➡️ Next:" in res.text
    res = _ajournal(CommandContext(journal=_Journal([{"seq": 1}]),
                                   journal_path="/tmp/j.jsonl"), [])
    assert "➡️ Next:" in res.text


# ------------------------------------------------------------ formatting survives

def test_status_formatting_survives_conversion(tmp_path):
    text = _mt5_text(_journal(tmp_path), tmp_path, "/status")
    assert "📊 MT5 ENGINE STATUS" in text
    assert "📡 Feed" in text and "📈 Open Positions" in text
    assert "`" in text  # code spans still present
    assert "✅" in text and "🟢 engine: RUNNING" in text
    assert "➡️ Next:" not in text


# ------------------------------------------------------------ callback routing

def _edits(api):
    return [p for m, p in api.calls if m == "editMessageText"]


def test_cmd_routing_rerenders_in_place(tmp_path, monkeypatch):
    monkeypatch.setenv("GS_TELEGRAM_CHAT_ID", OWNER)
    runner = _runner(tmp_path)
    api = FakeApi([])
    runner._tg_api = api
    runner._handle_callback_query(_callback("mt5:cmd:/pending"), {})
    edits = _edits(api)
    assert edits and "MT5 PENDING ORDERS" in edits[0]["text"]
    assert "mt5:cmd:/position" in json.dumps(
        json.loads(edits[0]["reply_markup"]))
    api.calls.clear()
    # Seed one evaluation so /assess all takes the text branch
    # (empty journal returns the keyboard-less early return).
    (tmp_path / "mt5_journal.jsonl").write_text(json.dumps(
        _record("short read")) + "\n")
    runner._handle_callback_query(_callback("mt5:cmd:/assess all"), {})
    edits = _edits(api)
    assert edits
    assert "mt5:cmd:/assess" in json.dumps(
        json.loads(edits[0]["reply_markup"]))
    api.calls.clear()
    runner._handle_callback_query(_callback("mt5:cmd:/set"), {})
    edits = _edits(api)
    assert edits and "usage: /set" in edits[0]["text"]
    # Keyboard-less result falls back to a bare Back row.
    assert json.loads(edits[0]["reply_markup"]) == mt5_back_keyboard()
    api.calls.clear()
    runner._handle_callback_query(_callback("mt5:cmd:/bogus"), {})
    assert not _edits(api)


def test_refresh_and_config_keep_merged_keyboards(tmp_path, monkeypatch):
    monkeypatch.setenv("GS_TELEGRAM_CHAT_ID", OWNER)
    runner = _runner(tmp_path)
    api = FakeApi([])
    runner._tg_api = api
    runner._handle_callback_query(_callback("mt5:refresh"), {})
    edits = _edits(api)
    assert edits
    markup = json.dumps(json.loads(edits[0]["reply_markup"]))
    assert "mt5:cmd:/trades" in markup
    assert "mt5:cmd:/playbooks" in markup
    assert "mt5:cmd:/health" in markup
    api.calls.clear()
    runner._handle_callback_query(_callback("mt5:config"), {})
    edits = _edits(api)
    assert edits
    assert "mt5:cmd:/set" in json.dumps(
        json.loads(edits[0]["reply_markup"]))
