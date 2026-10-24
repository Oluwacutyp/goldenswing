"""Telegram formatting conformance — Septorch-style replies.

Every command reply is plain text (no Telegram HTML — HTML would
leak into WhatsApp-style plain-text surfaces), uses emoji section
headers in a fixed order, monospace values, ➖ empty states, and an
explicit next-action CTA.  This test pins that contract: same
structured payload through each renderer, no cross-platform markup
bleed, emojis + sections present and ordered.
"""
import re
from pathlib import Path

import pytest

from goldenswing.alerts.commands import (
    COMMANDS, CommandContext, CommandRouter,
    fmt_progress, fmt_r,
)
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.mt5_commands import (
    MENU_ALIASES, mt5_router,
)

OWNER = "12345"

# Telegram HTML tags that must never appear in a plain-text reply.
# Angle-bracket placeholders like `<key>` in usage text are not HTML
# and are intentionally NOT matched here.
HTML_TAG_RE = re.compile(
    r"<(b|i|u|code|a|pre|strong|em|span|div|p|br|href)\b[^>]*>",
    re.IGNORECASE,
)


def _update(text, chat_id=OWNER, update_id=1):
    return {"update_id": update_id,
            "message": {"message_id": 1,
                        "chat": {"id": chat_id, "type": "private"},
                        "text": text}}


def _journal(tmp_path):
    j = TradeJournal(tmp_path / "journal.jsonl")
    j.log_open(alert_id="MT5-fvg_midline-10", instrument="XAUUSD",
               mode="MT5", playbook="fvg_midline", phase="live",
               direction=1, score=70, source="mt5", ticket=1,
               entry=4160.0, stop=4150.0, volume=0.01, risk_money=10.0)
    j.log_close(alert_id="MT5-fvg_midline-10", ticket=1,
                exit_price=4170.0, reason="trail", pnl_money=10.0,
                r_net=1.0, hold_bars=3)
    return j


class _FakeControl:
    """Minimal mt5_control: config + blocked stats + one-liners."""

    def __init__(self, *, paused=False):
        self._paused = paused

    def get_live_config(self):
        return {"sizing_mode": "risk", "risk_pct": 0.01,
                "max_layers": 3,
                "manipulation_filter": True, "pending_expiry_bars": 4,
                "paused": self._paused,
                "playbooks": {"judas_swing": True, "turtle_soup": False}}

    def blocked_stats(self):
        return {"legs": {"d1_bias": 2}, "manipulation_blocked": 0,
                "last": "d1_bias",
                "signal_counts": {"judas_swing": 3}}

    def alignment_line(self):
        return "macro alignment: D1 -1 / W1 +0 / MN1 +0 -> D1_Only 1x"

    def dxy_line(self):
        return "DXY: unavailable (no DXY data yet)"

    def health_line(self):
        return "health: tick 1s ago | latency n/a | slippage n/a"


def _mt5_ctx(journal, tmp_path, *, control=None, feed_ok=True):
    return CommandContext(
        journal=journal, journal_path=journal.path,
        reports_dir=tmp_path / "reports",
        feed_status={"ok": feed_ok,
                     "reason": "" if feed_ok else "feed stale"},
        last_bar_time="2026-10-16T09:00:00+00:00", last_price=4176.4,
        mt5_control=control if control is not None else _FakeControl(),
        live_positions=[], pending_orders=[])


def _mt5_text(journal, tmp_path, cmd, **kw):
    ctx = _mt5_ctx(journal, tmp_path, **kw)
    res, _ = mt5_router(OWNER).dispatch(_update(cmd), ctx)
    assert res is not None and res.text is not None, cmd
    return res.text


def _oanda_text(journal, tmp_path, cmd):
    ctx = CommandContext(
        journal=journal, journal_path=journal.path,
        feed_status={"ok": True}, last_bar_time="2026-10-16T09:00:00+00:00",
        last_price=4176.4)
    res, _ = CommandRouter(owner_chat_id=OWNER).dispatch(
        _update(cmd), ctx)
    assert res is not None and res.text is not None, cmd
    return res.text


# ------------------------------------------------------------ helpers

def test_fmt_r_sign_markers():
    assert fmt_r(1.0) == "🟢 +1.00R"
    assert fmt_r(-0.5) == "🔴 -0.50R"
    assert fmt_r(0.0) == "➖ +0.00R"


def test_fmt_progress_bar():
    assert fmt_progress(0.5) == "▓▓▓▓▓░░░░░ 50%"
    assert fmt_progress(0.0) == "░░░░░░░░░░ 0%"
    assert fmt_progress(1.0) == "▓▓▓▓▓▓▓▓▓▓ 100%"


# ------------------------------------------------------------ /status MT5

def test_mt5_status_sections_in_order(tmp_path):
    text = _mt5_text(_journal(tmp_path), tmp_path, "/status")
    sections = ["📊 MT5 ENGINE STATUS", "📡 Feed", "⚙️ Engine",
                "🌍 Macro Alignment", "💵 DXY", "❤️ Health",
                "🔧 Config", "📈 Open Positions", "⏳ Pending Orders",
                "📅 Today (UTC)", "🚫 why no trades",
                "🎯 Per-Playbook Signals", "➡️ Next:"]
    positions = [text.index(s) for s in sections]
    assert positions == sorted(positions), text
    assert "✅" in text and "`4,176.4`" in text
    assert "🟢 engine: RUNNING" in text
    # Runtime Truth item 4: /status reports the dynamic sizing
    # rule — there is no fixed_lots key anymore.  Owner ruling
    # 2026-10-24: the display renders the formula (coefficient k,
    # confidence C, scalars S, broker-min floor), never the old
    # "risk 0.01/trade" shorthand that presented k as a fixed
    # per-trade risk rule.
    assert "`sizing=`" in text and "dynamic k=0.01" in text
    assert "C=1.0 pending" in text
    assert "fixed_lots" not in text
    assert "/playbooks" in text  # CTA


def test_mt5_status_stale_feed_and_paused(tmp_path):
    journal = _journal(tmp_path)
    text = _mt5_text(journal, tmp_path, "/status",
                     control=_FakeControl(paused=True), feed_ok=False)
    assert "STALE ⚠️" in text
    assert "⏸️ engine: PAUSED" in text


def test_mt5_status_blocked_legs_and_counts(tmp_path):
    text = _mt5_text(_journal(tmp_path), tmp_path, "/status")
    assert "d1_bias" in text
    assert "judas_swing" in text
    assert "why no trades" in text  # lowercase contract, kept


# ------------------------------------------------------------ other MT5

def test_mt5_position_pending_none_markers(tmp_path):
    assert "➖ (none)" in _mt5_text(_journal(tmp_path), tmp_path,
                                   "/position")
    assert "➖ (none)" in _mt5_text(_journal(tmp_path), tmp_path,
                                   "/pending")


def test_mt5_playbooks_rows_keep_contract(tmp_path):
    text = _mt5_text(_journal(tmp_path), tmp_path, "/playbooks")
    assert "judas_swing: executing, on, signals 3" in text
    assert "turtle_soup: executing, off, signals 0" in text


def test_mt5_config_health_journal_reconcile_have_headers(tmp_path):
    journal = _journal(tmp_path)
    assert "⚙️ MT5 LIVE CONFIG" in _mt5_text(journal, tmp_path, "/config")
    health = _mt5_text(journal, tmp_path, "/health")
    assert "❤️ MT5 HEALTH" in health and "📡 Feed" in health
    assert "📓 MT5 Journal" in _mt5_text(journal, tmp_path, "/journal")
    assert "🔄 Reconciliation" in _mt5_text(journal, tmp_path,
                                           "/reconcile")
    assert "📋 MT5 engine commands" in _mt5_text(journal, tmp_path,
                                                "/help")


def test_mt5_trades_sign_markers(tmp_path):
    text = _mt5_text(_journal(tmp_path), tmp_path, "/trades")
    assert "📊 last 1 MT5 trades:" in text
    assert "🟢 +1.00R" in text


def test_menu_emoji_buttons_route(tmp_path):
    journal = _journal(tmp_path)
    ctx = _mt5_ctx(journal, tmp_path)
    router = mt5_router(OWNER)
    res, _ = router.dispatch(_update("📊 Status"), ctx)
    assert res is not None and "MT5 ENGINE STATUS" in res.text
    res, _ = router.dispatch(_update("Status"), ctx)
    assert res is not None and "MT5 ENGINE STATUS" in res.text
    assert "📊 status" in MENU_ALIASES  # emoji alias registered


# ------------------------------------------------------------ OANDA

def test_oanda_status_has_emoji_sections_and_cta(tmp_path):
    text = _oanda_text(_journal(tmp_path), tmp_path, "/status")
    assert "📊 PAPER LOOP STATUS" in text
    assert "📡 Feed" in text and "📈 Open Paper Positions" in text
    assert "📅 Today (UTC)" in text and "➡️ Next:" in text
    assert "✅" in text


def test_oanda_trades_health_help(tmp_path):
    journal = _journal(tmp_path)
    assert "📊 last 1 paper trades:" in _oanda_text(journal, tmp_path,
                                                   "/trades")
    assert "❤️ HEALTH" in _oanda_text(journal, tmp_path, "/health")
    assert "📋 GoldenSwing commands:" in _oanda_text(journal, tmp_path,
                                                    "/help")


# ------------------------------------------------------------ no HTML leak

def test_no_html_leak_in_command_outputs(tmp_path):
    journal = _journal(tmp_path)
    mt5_cmds = ["/status", "/position", "/pending", "/playbooks",
                "/config", "/health", "/trades", "/journal",
                "/reconcile", "/help"]
    for cmd in mt5_cmds:
        text = _mt5_text(journal, tmp_path, cmd)
        assert not HTML_TAG_RE.search(text), (cmd, text)
    oanda_cmds = ["/status", "/trades", "/health", "/help",
                  "/journal", "/reconcile"]
    for cmd in oanda_cmds:
        text = _oanda_text(journal, tmp_path, cmd)
        assert not HTML_TAG_RE.search(text), (cmd, text)
