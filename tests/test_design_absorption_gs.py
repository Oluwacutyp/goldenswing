"""Design absorption #5 + #6 (GoldenSwing) — pins.

#5: the bento daily-recap image (charts/recap.py) fed by
reporting.daily_recap_image_stats and wired into the MT5 runner's
rollover recap behind the same trigger as the text recap.
#6: the bot-screen state grammar (mt5_commands._screen_state) in
the /status header, count states on /position + /pending, and
designed empty states on /position /pending /trades — replacing
the bare "(none)" / bare sentence the screens used to answer.
"""
from __future__ import annotations

import io
from datetime import date
from types import SimpleNamespace

import pandas as pd
from PIL import Image

from goldenswing.alerts.commands import CommandContext
from goldenswing.charts.recap import render_daily_recap
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.mt5_commands import MT5_COMMANDS, _screen_state
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.reporting import daily_recap_image_stats
from goldenswing.execution.risk import gold_spec


# ---------------------------------------------------------------- journal

def _journal_with_trades(tmp_path):
    journal = TradeJournal(tmp_path / "j.jsonl")
    for aid, ts, entry, exit_, r in (
            ("a1", "2026-10-13T10:00:00+00:00", 2005.0, 2031.0, 1.5),
            ("a2", "2026-10-13T15:00:00+00:00", 2005.0, 1987.5, -1.0),
            ("a3", "2026-10-12T10:00:00+00:00", 2005.0, 2022.5, 1.0)):
        journal.append({"type": "open", "alert_id": aid,
                        "instrument": "XAUUSD", "playbook": "fvg_limit",
                        "direction": 1, "entry": entry, "stop": 1990.0,
                        "volume": 0.01, "risk_money": 17.5,
                        "ts": ts})
        journal.append({"type": "close", "alert_id": aid, "ticket": 1,
                        "exit": exit_, "reason": "target" if r > 0
                        else "stop", "pnl_money": r * 17.5, "r_net": r,
                        "hold_bars": 3, "ts": ts})
    return journal


_VITALS_OK = {"ok": True, "balance": 67.24, "equity": 67.24,
              "currency": "USD", "drawdown_pct": 0.212,
              "limit_pct": 0.15, "halted": True,
              "halt_reason": "max drawdown hit"}


# ------------------------------------------------------- #5 stats builder

def test_image_stats_match_text_recap_numbers(tmp_path):
    journal = _journal_with_trades(tmp_path)
    stats = daily_recap_image_stats(
        journal, date(2026, 10, 13), vitals=_VITALS_OK, open_count=1)
    assert stats["day"] == "2026-10-13"
    assert stats["trades"] == 2
    assert stats["wins"] == 1 and stats["losses"] == 1
    assert stats["pnl"] == 8.75            # +26.25 - 17.50
    assert stats["r_net"] == 0.5
    assert stats["win_rate"] == 50.0
    assert stats["open_positions"] == 1
    assert stats["equity"] == 67.24 and stats["balance"] == 67.24
    assert stats["drawdown_pct"] == 0.212
    assert stats["drawdown_limit"] == 0.15
    assert stats["halted"] is True


def test_image_stats_empty_day_winrate_undefined(tmp_path):
    journal = TradeJournal(tmp_path / "empty.jsonl")
    stats = daily_recap_image_stats(journal, date(2026, 10, 13))
    assert stats["trades"] == 0
    assert stats["win_rate"] is None       # undefined, never 0%
    assert stats["pnl"] == 0.0             # a REAL zero: nothing closed


def test_image_stats_unavailable_vitals_stay_none(tmp_path):
    journal = _journal_with_trades(tmp_path)
    for vitals in (None, {"ok": False, "error": "account read failed"}):
        stats = daily_recap_image_stats(
            journal, date(2026, 10, 13), vitals=vitals)
        assert stats["equity"] is None
        assert stats["balance"] is None
        assert stats["drawdown_pct"] is None
        assert stats["halted"] is False
        assert stats["pnl"] == 8.75        # journal numbers still real


# ------------------------------------------------------------ #5 renderer

def _full_stats(**over):
    stats = {"day": "2026-10-13", "pnl": 8.75, "r_net": 0.5,
             "trades": 2, "wins": 1, "losses": 1, "win_rate": 50.0,
             "open_positions": 1, "equity": 67.24, "balance": 67.24,
             "currency": "USD", "drawdown_pct": 0.212,
             "drawdown_limit": 0.15, "halted": True,
             "halt_reason": "max drawdown hit"}
    stats.update(over)
    return stats


def test_render_is_png_deterministic(tmp_path):
    png = render_daily_recap(_full_stats())
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(png))
    assert img.size == (1200, 700)
    assert render_daily_recap(_full_stats()) == png


def _accent_pixels(png: bytes, rgb: tuple) -> int:
    """Pixels of (near) the accent colour inside the P&L tile."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    tile = img.crop((24, 98, 696, 490))    # the P&L tile's box
    count = 0
    for px in tile.getdata():
        if all(abs(px[i] - rgb[i]) <= 10 for i in range(3)):
            count += 1
    return count


def test_render_pnl_accent_follows_sign():
    up = render_daily_recap(_full_stats(pnl=125.0))
    down = render_daily_recap(_full_stats(pnl=-125.0))
    assert _accent_pixels(up, (0x26, 0xA6, 0x9A)) > 200
    assert _accent_pixels(up, (0xEF, 0x53, 0x50)) == 0
    assert _accent_pixels(down, (0xEF, 0x53, 0x50)) > 200
    assert _accent_pixels(down, (0x26, 0xA6, 0x9A)) == 0


def test_render_unavailable_tiles_still_png():
    bare = {"day": "2026-10-13", "pnl": 0.0, "r_net": 0.0,
            "trades": 0, "wins": 0, "losses": 0, "win_rate": None,
            "open_positions": None, "equity": None, "balance": None,
            "currency": "", "drawdown_pct": None,
            "drawdown_limit": 0.15, "halted": False, "halt_reason": ""}
    png = render_daily_recap(bare)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(io.BytesIO(png)).size == (1200, 700)


# ------------------------------------------------- #5 rollover wiring

class _PhotoSender:
    def __init__(self):
        self.texts: list[str] = []
        self.photos: list[tuple] = []

    def send_text(self, text, silent=False, reply_markup=None):
        self.texts.append(text)
        return SimpleNamespace(delivered=True, detail="ok")

    def send_photo(self, png, caption):
        self.photos.append((png, caption))
        return SimpleNamespace(delivered=True, detail="ok")


class _FakeExecutor:
    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return []


def test_rollover_daily_recap_carries_bento_image(tmp_path):
    sender = _PhotoSender()
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg, executor=_FakeExecutor())
    runner.senders = [sender]
    runner.journal = _journal_with_trades(tmp_path)
    runner._last_tick_date = date(2026, 10, 13)
    runner._maybe_send_rollover_recap(
        {"time": pd.Timestamp("2026-10-14 00:00", tz="UTC"),
         "mid": 2000.0})
    # The text recap rides as the photo's caption (the _send_alert
    # chart seam) — exactly one daily send, now with the image.
    assert sender.texts == []
    assert len(sender.photos) == 1
    png, caption = sender.photos[0]
    assert "daily — 2026-10-13" in caption
    assert "Losses: 1" in caption
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


# ------------------------------------------------------- #6 state grammar

class _Ctrl:
    def __init__(self, *, vitals=None, paused=False, truth=None):
        self._vitals = vitals
        self._paused = paused
        self._truth = truth or {}

    def account_vitals(self):
        return self._vitals

    def get_live_config(self):
        return {"paused": self._paused}

    def feed_truth(self):
        return self._truth


def _ctx(ctrl=None, feed_status=None):
    return CommandContext(journal=None, mt5_control=ctrl,
                          feed_status=feed_status)


def test_screen_state_halted_beats_paused():
    ctrl = _Ctrl(vitals={"ok": True, "halted": True}, paused=True)
    assert _screen_state(_ctx(ctrl, {"ok": True})) == ("⛔", "HALTED")


def test_screen_state_paused():
    ctrl = _Ctrl(vitals={"ok": True, "halted": False}, paused=True)
    assert _screen_state(_ctx(ctrl, {"ok": True})) == ("⏸️", "PAUSED")


def test_screen_state_market_closed():
    ctrl = _Ctrl(truth={"tick_state": "closed_expected"})
    assert _screen_state(_ctx(ctrl, {"ok": True})) == \
        ("🌙", "MARKET CLOSED")


def test_screen_state_stale():
    ctrl = _Ctrl(truth={"tick_state": "stale"})
    assert _screen_state(_ctx(ctrl, {"ok": True})) == ("🟡", "STALE")
    # No feed status at all reads stale too (fmt_feed_line agrees).
    assert _screen_state(_ctx(None, None)) == ("🟡", "STALE")


def test_screen_state_live():
    assert _screen_state(_ctx(None, {"ok": True})) == ("🟢", "LIVE")
    # A failed vitals read cannot demote a live feed to halted.
    ctrl = _Ctrl(vitals={"ok": False, "error": "read failed"})
    assert _screen_state(_ctx(ctrl, {"ok": True})) == ("🟢", "LIVE")


def test_status_header_carries_state(tmp_path):
    journal = _journal_with_trades(tmp_path)
    ctx = CommandContext(journal=journal, feed_status={"ok": True})
    res = MT5_COMMANDS["/status"][1](ctx, [])
    first = (res.text or "").splitlines()[0]
    assert "MT5 ENGINE STATUS" in first
    assert "🟢 LIVE" in first


# ----------------------------------------------------- #6 empty states

def test_position_empty_is_designed(tmp_path):
    journal = TradeJournal(tmp_path / "empty.jsonl")
    ctx = CommandContext(journal=journal, live_positions=[])
    res = MT5_COMMANDS["/position"][1](ctx, [])
    assert "MT5 OPEN POSITIONS:" in res.text
    assert "➖ flat" in res.text
    assert "flat — no open positions" in res.text
    assert "➖ (none)" not in res.text


def test_position_header_counts_live():
    live = [{"ticket": 5, "playbook": "qm_m5", "direction": 1,
             "entry_price": 4000.0, "sl": 3990.0, "layers": []}]
    ctx = CommandContext(journal=None, live_positions=live,
                         last_price=4001.0)
    res = MT5_COMMANDS["/position"][1](ctx, [])
    assert "🟢 1 open" in res.text
    assert "#5" in res.text


def test_pending_empty_is_designed(tmp_path):
    journal = TradeJournal(tmp_path / "empty.jsonl")
    ctx = CommandContext(journal=journal, pending_orders=[])
    res = MT5_COMMANDS["/pending"][1](ctx, [])
    assert "MT5 PENDING ORDERS:" in res.text
    assert "none resting" in res.text
    assert "nothing resting — no pending orders" in res.text
    assert "➖ (none)" not in res.text


def test_trades_empty_is_designed(tmp_path):
    journal = TradeJournal(tmp_path / "empty.jsonl")
    ctx = CommandContext(journal=journal)
    res = MT5_COMMANDS["/trades"][1](ctx, [])
    assert "MT5 TRADES:" in res.text
    assert "no closed MT5 trades yet" in res.text
    assert "R multiple and exit reason" in res.text
    assert res.reply_markup is not None
