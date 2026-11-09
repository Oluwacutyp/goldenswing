"""Design absorption #12 (GoldenSwing) — desk / LinkedIn template pins.

The desk card (charts/desk.py) is the bento recap's brand-facing
sibling: the same day, the same numbers (reporting's stats
builder + account_vitals), rendered at LinkedIn's 1200x627 with
the wordmark and the "Powered by Peacethefirst" stamp — and the
/desk command that serves it on demand as ONE photo whose caption
is the one-line day summary, through the reply seam the recap
alerts use (photo failure falls back honestly, never silently).
"""
from __future__ import annotations

import io
from datetime import date
from types import SimpleNamespace

from PIL import Image

from goldenswing.alerts.commands import CommandContext, CommandResult
from goldenswing.alerts.model import SendResult
from goldenswing.alerts.telegram import TelegramSender
from goldenswing.charts import desk as desk_mod
from goldenswing.charts import recap as recap_mod
from goldenswing.charts import tiles
from goldenswing.charts.desk import render_desk_card
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.mt5_commands import MT5_COMMANDS, _desk_caption
from goldenswing.execution.mt5_runner import MT5Runner
from goldenswing.execution.reporting import daily_recap_image_stats


# ---------------------------------------------------------------- fixtures

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


def _full_stats(**over):
    stats = {"day": "2026-10-13", "pnl": 8.75, "r_net": 0.5,
             "trades": 2, "wins": 1, "losses": 1, "win_rate": 50.0,
             "open_positions": 1, "equity": 67.24, "balance": 67.24,
             "currency": "USD", "drawdown_pct": 0.212,
             "drawdown_limit": 0.15, "halted": True,
             "halt_reason": "max drawdown hit"}
    stats.update(over)
    return stats


def _bare_stats():
    return {"day": "2026-10-13", "pnl": 0.0, "r_net": 0.0,
            "trades": 0, "wins": 0, "losses": 0, "win_rate": None,
            "open_positions": None, "equity": None, "balance": None,
            "currency": "", "drawdown_pct": None,
            "drawdown_limit": 0.15, "halted": False, "halt_reason": ""}


# ------------------------------------------------------------- tile engine

def test_one_tile_engine_shared_by_recap_and_desk():
    # No duplicate parallel tile engine: the recap's helpers ARE
    # the shared module's, and the desk draws from the same theme.
    assert recap_mod._tile is tiles.tile
    assert recap_mod._label is tiles.tile_label
    assert recap_mod._unavailable is tiles.unavailable
    assert desk_mod._tile is tiles.tile
    assert desk_mod._BG == tiles.BG == recap_mod._BG
    assert desk_mod._UP == tiles.UP and desk_mod._DOWN == tiles.DOWN


# --------------------------------------------------------------- renderer

def test_render_is_linkedin_shaped_png_deterministic():
    png = render_desk_card(_full_stats())
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(io.BytesIO(png)).size == (1200, 627)
    assert render_desk_card(_full_stats()) == png


def test_render_values_come_from_the_recap_stats_builder(tmp_path):
    # The desk draws the fixture journal through the SAME builder
    # the recap uses — the numbers the tiles carry are these.
    journal = _journal_with_trades(tmp_path)
    stats = daily_recap_image_stats(
        journal, date(2026, 10, 13), vitals=_VITALS_OK, open_count=1)
    png = render_desk_card(stats)
    assert png == render_desk_card(_full_stats())


def _accent_pixels(png: bytes, rgb: tuple) -> int:
    """Pixels of (near) the accent colour inside the P&L tile."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    tile = img.crop((24, 116, 696, 458))   # the desk P&L tile's box
    count = 0
    for px in tile.getdata():
        if all(abs(px[i] - rgb[i]) <= 10 for i in range(3)):
            count += 1
    return count


def test_render_pnl_accent_follows_sign():
    up = render_desk_card(_full_stats(pnl=125.0))
    down = render_desk_card(_full_stats(pnl=-125.0))
    assert _accent_pixels(up, (0x26, 0xA6, 0x9A)) > 200
    assert _accent_pixels(up, (0xEF, 0x53, 0x50)) == 0
    assert _accent_pixels(down, (0xEF, 0x53, 0x50)) > 200
    assert _accent_pixels(down, (0x26, 0xA6, 0x9A)) == 0


def test_render_unavailable_tiles_still_png():
    png = render_desk_card(_bare_stats())
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(io.BytesIO(png)).size == (1200, 627)
    # Honesty is in the data first: nothing was zero-filled.
    stats = _bare_stats()
    assert stats["win_rate"] is None and stats["equity"] is None
    assert stats["drawdown_pct"] is None and stats["pnl"] == 0.0


def _ink(png: bytes, box: tuple) -> int:
    """Non-white pixels in a box — something is drawn there."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    region = img.crop(box)
    return sum(1 for px in region.getdata()
               if px != (255, 255, 255))


def test_brand_furniture_wordmark_and_stamp():
    assert desk_mod.WORDMARK == "GOLDENSWING"
    assert desk_mod.STAMP == "Powered by Peacethefirst"
    png = render_desk_card(_full_stats())
    # Header band carries the wordmark; footer band the stamp.
    assert _ink(png, (0, 0, 500, 60)) > 200
    assert _ink(png, (0, 585, 500, 627)) > 50


# ------------------------------------------------------------ /desk wiring

class _Ctrl:
    def __init__(self, *, vitals=None, guard_day=None):
        self._vitals = vitals
        self.guards = SimpleNamespace(state={"day": guard_day})

    def account_vitals(self):
        return self._vitals


def _today_journal(tmp_path):
    """A journal whose one close lands TODAY (the /desk day)."""
    today = date.today().isoformat()
    journal = TradeJournal(tmp_path / "today.jsonl")
    journal.append({"type": "open", "alert_id": "t1",
                    "instrument": "XAUUSD", "playbook": "fvg_limit",
                    "direction": 1, "entry": 2005.0, "stop": 1990.0,
                    "volume": 0.01, "risk_money": 17.5,
                    "ts": today + "T10:00:00+00:00"})
    journal.append({"type": "close", "alert_id": "t1", "ticket": 1,
                    "exit": 2031.0, "reason": "target",
                    "pnl_money": 26.25, "r_net": 1.5, "hold_bars": 3,
                    "ts": today + "T10:05:00+00:00"})
    return journal, today


def test_desk_registered_with_help_text():
    desc, _fn = MT5_COMMANDS["/desk"]
    assert "desk image" in desc and "drawdown" in desc


def test_desk_handler_sends_one_photo_with_summary_caption(tmp_path):
    journal, today = _today_journal(tmp_path)
    ctrl = _Ctrl(vitals=dict(_VITALS_OK, drawdown_pct=0.0,
                             halted=False, halt_reason=""),
                 guard_day=today)
    ctx = CommandContext(journal=journal, mt5_control=ctrl,
                         live_positions=[])
    res = MT5_COMMANDS["/desk"][1](ctx, [])
    assert res.text is None and res.document is None
    assert res.photo is not None and res.photo[:4] == b"\x89PNG"
    assert Image.open(io.BytesIO(res.photo)).size == (1200, 627)
    assert res.caption is not None and "\n" not in res.caption
    assert today in res.caption and "+26.25 USD" in res.caption
    assert "100% win rate" in res.caption
    assert "1 closed" in res.caption and "equity 67.24 USD" in res.caption
    assert res.reply_markup is not None      # path back (R2)


def test_desk_handler_unavailable_vitals_stay_honest(tmp_path):
    journal, today = _today_journal(tmp_path)
    ctx = CommandContext(journal=journal, mt5_control=None,
                         live_positions=[])
    res = MT5_COMMANDS["/desk"][1](ctx, [])
    assert res.photo is not None             # the card still renders
    assert "equity" not in res.caption       # unread -> omitted
    assert "0.00 USD" not in res.caption     # never zero-filled


def test_desk_handler_stale_guard_day_drops_drawdown(tmp_path):
    # Before today's first tick the guard stands on yesterday:
    # its drawdown must not dress up as today's (the recap rule).
    journal, today = _today_journal(tmp_path)
    vitals = dict(_VITALS_OK)
    stale = _Ctrl(vitals=vitals, guard_day="1999-01-01")
    ctx = CommandContext(journal=journal, mt5_control=stale,
                         live_positions=[])
    res = MT5_COMMANDS["/desk"][1](ctx, [])
    expect = render_desk_card(daily_recap_image_stats(
        journal, date.today(),
        vitals=dict(vitals, drawdown_pct=None), open_count=0))
    assert res.photo == expect
    fresh = _Ctrl(vitals=vitals, guard_day=today)
    res2 = MT5_COMMANDS["/desk"][1](
        CommandContext(journal=journal, mt5_control=fresh,
                       live_positions=[]), [])
    assert res2.photo != res.photo           # today's bar is drawn


def test_desk_handler_usage_and_no_journal(tmp_path):
    ctx = CommandContext(journal=None)
    assert MT5_COMMANDS["/desk"][1](ctx, []).text == \
        "no journal attached."
    journal, _today = _today_journal(tmp_path)
    ctx = CommandContext(journal=journal)
    assert MT5_COMMANDS["/desk"][1](ctx, ["x"]).text == "usage: /desk"


def test_desk_caption_omits_what_the_day_does_not_have():
    cap = _desk_caption(_full_stats())
    assert cap.startswith("🖥️ Desk 2026-10-13: +8.75 USD")
    assert "50% win rate" in cap and "2 closed" in cap
    assert "equity 67.24 USD" in cap and "⛔ HALTED" in cap
    bare = _desk_caption(_bare_stats())
    assert "win rate" not in bare and "equity" not in bare
    assert "no closed trades yet" in bare and "HALTED" not in bare


# ------------------------------------------------------------- reply seam

def _bare_runner(tmp_path, sender):
    runner = MT5Runner.__new__(MT5Runner)
    runner.senders = [sender]
    runner.journal = TradeJournal(tmp_path / "mt5_journal.jsonl")
    runner.executor = None
    return runner


class _PhotoSender:
    """Markup-aware photo sender (the TelegramSender shape)."""

    def __init__(self, *, result=None):
        self.photos: list[tuple] = []
        self.texts: list[str] = []
        self._result = result

    def send_photo(self, png, caption=None, reply_markup=None):
        self.photos.append((png, caption, reply_markup))
        return self._result or SendResult("telegram", True, "live", "ok")

    def send_text(self, text, silent=False, reply_markup=None):
        self.texts.append(text)
        return SendResult("telegram", True, "live", "ok")

    def send(self, text):
        self.texts.append(text)
        return SendResult("telegram", True, "live", "ok")


class _LegacyPhotoSender:
    """send_photo without the reply_markup kwarg (older builds)."""

    def __init__(self):
        self.photos: list[tuple] = []
        self.texts: list[str] = []

    def send_photo(self, png, caption):
        self.photos.append((png, caption))
        return SendResult("telegram", True, "live", "ok")

    def send_text(self, text, silent=False, reply_markup=None):
        self.texts.append(text)
        return SendResult("telegram", True, "live", "ok")


class _TextOnlySender:
    def __init__(self):
        self.sent: list[str] = []
        self.texts: list[str] = []

    def send(self, text):
        self.sent.append(text)
        return SendResult("telegram", True, "live", "ok")

    def send_text(self, text, silent=False, reply_markup=None):
        self.texts.append(text)
        return SendResult("telegram", True, "live", "ok")


def test_reply_command_sends_photo_once_with_caption_and_markup(tmp_path):
    sender = _PhotoSender()
    runner = _bare_runner(tmp_path, sender)
    markup = {"inline_keyboard": []}
    runner._reply_command(CommandResult(
        photo=b"\x89PNG-fake", caption="🖥️ Desk summary",
        reply_markup=markup))
    assert sender.photos == [(b"\x89PNG-fake", "🖥️ Desk summary", markup)]
    assert sender.texts == []


def test_reply_command_photo_legacy_sender_still_delivers(tmp_path):
    sender = _LegacyPhotoSender()
    runner = _bare_runner(tmp_path, sender)
    runner._reply_command(CommandResult(
        photo=b"\x89PNG-fake", caption="cap",
        reply_markup={"inline_keyboard": []}))
    assert sender.photos == [(b"\x89PNG-fake", "cap")]
    assert sender.texts == []


def test_reply_command_photo_failure_is_never_silent(tmp_path):
    sender = _PhotoSender(result=SendResult(
        "telegram", False, "live", "photo too large"))
    runner = _bare_runner(tmp_path, sender)
    runner._reply_command(CommandResult(
        photo=b"\x89PNG-fake", caption="cap"))
    assert any("photo not delivered" in t and "photo too large" in t
               for t in sender.texts)
    notes = [r.get("text", "") for r in runner.journal.records()]
    assert any("photo not delivered" in n for n in notes)


def test_reply_command_photo_without_photo_sender_sends_caption(tmp_path):
    sender = _TextOnlySender()
    runner = _bare_runner(tmp_path, sender)
    runner._reply_command(CommandResult(
        photo=b"\x89PNG-fake", caption="🖥️ Desk summary"))
    assert sender.sent == ["🖥️ Desk summary"]


# ------------------------------------------------------- Telegram sender

class _CapturingSender(TelegramSender):
    def __init__(self):
        super().__init__(bot_token="tok", chat_id="42")
        self.posts: list[dict] = []

    def _post_with_retry(self, url, data=None, files=None):
        self.posts.append({"url": url, "data": data, "files": files})
        return True, "ok"


def test_telegram_send_photo_carries_reply_markup():
    import json as _json
    sender = _CapturingSender()
    markup = {"inline_keyboard": [[{"text": "📊 Status",
                                     "callback_data": "mt5:status"}]]}
    res = sender.send_photo(b"\x89PNG-fake", "cap", reply_markup=markup)
    assert res.delivered
    data = sender.posts[0]["data"]
    assert _json.loads(data["reply_markup"]) == markup
    # And without markup the payload stays markup-free.
    sender2 = _CapturingSender()
    sender2.send_photo(b"\x89PNG-fake", "cap")
    assert "reply_markup" not in sender2.posts[0]["data"]
