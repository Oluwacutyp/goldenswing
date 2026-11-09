"""Telegram markup standard — conformance (owner directive 2026-11-09).

The long bot messages were being CUT OFF: the sender sliced any
text over Telegram's 4096-char limit to ``text[:4095] + "…"`` and
the owner's /status screenshot ended mid-data ("72h up: unproven —
top-band 1/2...").  The standard pinned here:

* handlers keep emitting PLAIN text (test_telegram_formatting owns
  that contract); markup happens at the Telegram boundary —
  :mod:`goldenswing.alerts.markup` renders HTML (bold section
  headers, ``<code>`` spans, expandable quotes for the deep-detail
  blocks) and splits losslessly at section boundaries;
* nothing is ever truncated: concatenated parts contain every
  source line exactly once, each part <= 4096 incl. markup, tags
  balanced per part;
* keyboards ride the LAST part of a split send;
* /health carries the account vitals (equity / balance / drawdown
  vs the 15% sole halt) from the runner's real account read, and
  renders an unavailable source honestly — never as 0.00.
"""
import html as _html
import re
from types import SimpleNamespace

from goldenswing.alerts.markup import (
    MESSAGE_LIMIT, caption_parts, render_messages)
from goldenswing.alerts.telegram import TelegramSender
from goldenswing.execution.risk import drawdown_snapshot

_TAG_RE = re.compile(r"<[^>]+>")


def _plain(rendered_parts):
    """Strip tags + unescape: the fact content of rendered parts."""
    joined = "\n".join(rendered_parts)
    return _html.unescape(_TAG_RE.sub("", joined))


def _fact_lines(text):
    """Source lines as facts: backtick code-span markers are
    presentation, so they come off both sides of a comparison."""
    return [line.replace("`", "") for line in text.split("\n")]


def _assert_balanced(part):
    assert part.count("<b>") == part.count("</b>"), part
    assert part.count("<code>") == part.count("</code>"), part
    assert part.count("<blockquote") == part.count("</blockquote>"), part


# ------------------------------------------------------------ rendering

def test_headers_bold_code_spans_and_escaping():
    text = ("📊 MT5 ENGINE STATUS\n\n📡 Feed\n"
            "  mid: `2,650.12` (live)\n"
            "usage: /assess <symbol> & go")
    (part,) = render_messages(text)
    assert "<b>📊 MT5 ENGINE STATUS</b>" in part
    assert "<b>📡 Feed</b>" in part
    assert "<code>2,650.12</code>" in part
    # Dynamic angle brackets are escaped, never tags.
    assert "&lt;symbol&gt;" in part and "&amp;" in part
    assert "<symbol>" not in part
    _assert_balanced(part)


def test_content_lines_are_not_bolded():
    text = ("⚙️ Engine\n"
            "🟢 engine: RUNNING\n"
            "🔮 forecast: READY — builder=ensemble\n"
            "🟢 +1.00R  fvg_midline")
    (part,) = render_messages(text)
    assert part.count("<b>") == 1          # only the section header
    assert "<b>⚙️ Engine</b>" in part


_FORECAST_BLOCK = """🔮 forecast: READY — builder=ensemble
 outlook (F1 — numbers, not ranks):
    12h: NEUTRAL → 4,098.6  range [4,054.2 … 4,147.2]  exp move ±1.13% ($46.4)
    24h: NEUTRAL → 4,100.7  range [4,041.0 … 4,160.4]  exp move ±1.46% ($59.7)
 probability book (F2 — stated vs realized):
    12h: trained n=47  ECE 0.1948  Brier 0.2410 (46/47 in band)
    24h: trained n=36  ECE 0.2530  Brier 0.2490 (36/36 in band)
    graduation gate: trained n≥60 and ECE≤0.10
  retrain accrual: ensemble 234/500 new H1 bars since the last train
    trained (F2): 79/500 graded forecasts (age 5.4d / 21d)
 battery (F3 acceptance — per horizon):
    12h: pass 40/44  coverage in [0.76,0.84]
 forecast trust (E3 — weighed only when trusted):
    12h up: unproven — top-band 1/2 hits (n=4)
    72h up: unproven — top-band 1/2 hits (n=2)"""


def test_forecast_detail_blocks_are_expandable_outlook_is_not():
    (part,) = render_messages("🔮 Forecast\n" + _FORECAST_BLOCK)
    quotes = re.findall(r"<blockquote expandable>(.*?)</blockquote>",
                        part, re.DOTALL)
    blob = "\n".join(quotes)
    # Probability book detail, retrain accrual, battery + trust
    # detail lines collapse; the F1 outlook lines stay visible.
    assert "trained n=47" in blob
    assert "retrain accrual: ensemble 234/500" in blob
    assert "12h: pass 40/44" in blob
    assert "72h up: unproven — top-band 1/2 hits (n=2)" in blob
    assert "NEUTRAL → 4,098.6" not in blob
    assert "outlook (F1 — numbers, not ranks):" not in blob
    # Section openers stay visible too.
    assert "probability book (F2" in part.split("<blockquote")[0] \
        or "probability book (F2" in re.sub(
            r"<blockquote expandable>.*?</blockquote>", "", part,
            flags=re.DOTALL)
    visible = re.sub(r"<blockquote expandable>.*?</blockquote>", "",
                     part, flags=re.DOTALL)
    assert "probability book (F2" in visible
    assert "forecast trust (E3" in visible
    _assert_balanced(part)
    assert _plain([part]).split("\n") == \
        _fact_lines("🔮 Forecast\n" + _FORECAST_BLOCK)


def _maximal_status() -> str:
    """Worst case: every section full — all horizons, all bands,
    sixty open positions, a long journal digest."""
    lines = ["📊 MT5 ENGINE STATUS", "", "📡 Feed",
             "  symbol: XAUUSD", "  mid: `2,650.12` (live)", "",
             "⚙️ Engine", "🟢 engine: RUNNING", "",
             "📈 Open Positions"]
    for i in range(60):
        lines.append(f"  #{1000 + i} XAUUSD buy 0.10 @ 2,650.1{i % 10}"
                     f"  pnl +12.3{i % 7}  playbook judas_swing")
    lines += ["", "⏳ Pending Orders"]
    for i in range(20):
        lines.append(f"  #{2000 + i} buy-limit 0.05 @ 2,601.0{i % 10}")
    lines += ["", "🔮 Forecast", _FORECAST_BLOCK, "", "📓 Journal"]
    for i in range(80):
        lines.append(f"  2026-11-0{i % 9 + 1} close ticket {3000 + i} "
                     f"pnl +10.0{i % 5} reason trail")
    return "\n".join(lines)


def test_maximal_status_splits_losslessly():
    text = _maximal_status()
    assert len(text) > MESSAGE_LIMIT       # the fixture IS worst-case
    parts = render_messages(text)
    assert len(parts) > 1
    for part in parts:
        assert len(part) <= MESSAGE_LIMIT, len(part)
        _assert_balanced(part)
    # Every source line exactly once, in order, across the parts.
    assert _plain(parts).split("\n") == _fact_lines(text)
    # No ellipsis was introduced by the splitter (the source's own
    # range "…" characters are data and survive as-is).
    assert _plain(parts).count("…") == text.count("…")


def test_split_prefers_section_boundaries():
    text = _maximal_status()
    parts = render_messages(text)
    # A part never OPENS mid-section when the section fits a message
    # of its own: every part after the first starts either with a
    # bold section header or inside the oversized positions block.
    for part in parts[1:]:
        first = part.split("\n", 1)[0]
        assert first.startswith("<b>") or first.startswith("  #") \
            or first.startswith("  2026-") \
            or first.startswith("<blockquote"), first


def test_monster_single_line_is_wrapped_not_cut():
    text = "📓 Journal\n" + "  " + ("word " * 1500).strip()
    parts = render_messages(text)
    assert all(len(p) <= MESSAGE_LIMIT for p in parts)
    assert _plain(parts).replace("\n", " ").split() == text.split()


def test_caption_parts_overflow_returns_full_text():
    cap, overflow = caption_parts("short `caption`")
    assert overflow is None and "<code>caption</code>" in cap
    long_text = "📊 report\n" + "\n".join(
        f"  line {i} with some data" for i in range(80))
    cap, overflow = caption_parts(long_text)
    assert cap == "" and overflow == long_text


# ------------------------------------------------------------ sender

class _FakeResp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body if body is not None else {
            "ok": True, "result": {"message_id": 7}}
        self.headers = {"content-type": "application/json"}
        self.text = str(self._body)

    def json(self):
        return self._body


class _CapturingSession:
    def __init__(self):
        self.payloads = []

    def post(self, url, **kw):
        self.payloads.append(kw.get("json") or kw.get("data") or {})
        return _FakeResp(200)


def _live_sender(session):
    s = TelegramSender(bot_token="123456789:abcdefghijklmnopqrstuvwxyzABCD",
                       chat_id="1", session=session,
                       sleeper=lambda _s: None)
    assert s.mode == "live"
    return s


def test_sender_long_text_sends_all_parts_keyboard_on_last():
    sess = _CapturingSession()
    s = _live_sender(sess)
    kb = {"inline_keyboard": [[{"text": "🔄 Refresh",
                                 "callback_data": "mt5:refresh"}]]}
    res = s.send_text(_maximal_status(), silent=True, reply_markup=kb)
    assert res.delivered
    assert len(sess.payloads) > 1
    for payload in sess.payloads:
        assert payload["parse_mode"] == "HTML"
        assert len(payload["text"]) <= MESSAGE_LIMIT
        _assert_balanced(payload["text"])
    assert "reply_markup" not in sess.payloads[0]
    assert sess.payloads[-1]["reply_markup"]     # keyboard on LAST part
    sent_plain = _plain([p["text"] for p in sess.payloads])
    assert sent_plain.split("\n") == _fact_lines(_maximal_status())
    # The line the owner's screenshot lost is delivered now.
    assert "72h up: unproven — top-band 1/2 hits (n=2)" in sent_plain


def test_sender_short_text_single_part_no_markup_loss():
    sess = _CapturingSession()
    s = _live_sender(sess)
    res = s.send_text("📡 Feed\n  mid: `2,650.12`")
    assert res.delivered and len(sess.payloads) == 1
    assert sess.payloads[0]["text"] == \
        "<b>📡 Feed</b>\n  mid: <code>2,650.12</code>"


def test_sender_part_failure_reports_not_delivered():
    class _FailSecond(_CapturingSession):
        def post(self, url, **kw):
            self.payloads.append(kw.get("json") or {})
            if len(self.payloads) == 2:
                return _FakeResp(401, {"ok": False,
                                       "description": "unauthorized"})
            return _FakeResp(200)

    sess = _FailSecond()
    s = _live_sender(sess)
    res = s.send_text(_maximal_status())
    assert not res.delivered and "part 2" in res.detail


# ------------------------------------------------- drawdown arithmetic

def test_drawdown_snapshot_math():
    # Daily basis (owner correction 2026-11-09): the snapshot is
    # measured from the DAY's starting balance, never the peak.
    snap = drawdown_snapshot(day_start_balance=10_000.0, balance=9_250.0,
                             limit=0.15)
    assert snap["drawdown_pct"] == 0.075
    assert snap["distance_pct"] == 0.075
    assert snap["breached"] is False


def test_drawdown_snapshot_boundary_breaches_at_limit():
    at = drawdown_snapshot(day_start_balance=10_000.0, balance=8_500.0,
                           limit=0.15)
    assert at["drawdown_pct"] == 0.15 and at["breached"] is True
    over = drawdown_snapshot(day_start_balance=10_000.0, balance=8_499.0,
                             limit=0.15)
    assert over["breached"] is True
    under = drawdown_snapshot(day_start_balance=10_000.0, balance=8_501.0,
                              limit=0.15)
    assert under["breached"] is False


def test_drawdown_snapshot_no_day_start_is_none_not_zero():
    snap = drawdown_snapshot(day_start_balance=None, balance=9_000.0,
                             limit=0.15)
    assert snap["drawdown_pct"] is None
    assert snap["distance_pct"] is None
    assert snap["breached"] is False


# ------------------------------------------------- runner account vitals

class _FakeFeed:
    def __init__(self, state=None, exc=None):
        self._state, self._exc = state, exc

    def account_state(self):
        if self._exc is not None:
            raise self._exc
        return self._state


def _vitals(feed_state=None, *, peak=10_000.0, day_start=10_000.0,
            halted=False, halt_reason="", exc=None):
    from goldenswing.execution.mt5_runner import MT5Runner
    fake = SimpleNamespace(
        feed=_FakeFeed(feed_state, exc),
        guards=SimpleNamespace(
            state={"peak_balance": peak, "day_start_balance": day_start,
                   "halted": halted, "halt_reason": halt_reason},
            cfg=SimpleNamespace(max_drawdown_pct=0.15)))
    return MT5Runner.account_vitals(fake)


def test_account_vitals_populated():
    v = _vitals({"balance": 9_250.0, "equity": 9_310.5,
                 "margin": 120.0, "free_margin": 9_190.5,
                 "currency": "USD"})
    assert v["ok"] is True
    assert v["balance"] == 9_250.0 and v["equity"] == 9_310.5
    assert v["floating_pnl"] == 60.5
    # Today's drawdown comes from the day start (10,000 -> 9,250);
    # the all-time peak rides along as information only.
    assert v["drawdown_pct"] == 0.075 and v["distance_pct"] == 0.075
    assert v["breached"] is False
    assert v["day_start_balance"] == 10_000.0
    assert v["peak_balance"] == 10_000.0


def test_account_vitals_peak_does_not_drive_drawdown():
    # Balance is 21% under the all-time peak but only 2% under
    # today's start: the daily figures must show 2%, not a breach.
    v = _vitals({"balance": 9_800.0, "equity": 9_800.0,
                 "margin": None, "free_margin": 9_800.0,
                 "currency": "USD"},
                peak=12_400.0, day_start=10_000.0)
    assert v["ok"] is True
    assert v["drawdown_pct"] == 0.02
    assert v["breached"] is False
    assert v["peak_balance"] == 12_400.0


def test_account_vitals_feed_failure_is_honest():
    v = _vitals(exc=RuntimeError("terminal down"))
    assert v["ok"] is False and "terminal down" in v["error"]
    v2 = _vitals({"balance": None, "equity": None})
    assert v2["ok"] is False


# ------------------------------------------------------- /health surface

class _VitalsControl:
    def __init__(self, vitals):
        self._vitals = vitals

    def account_vitals(self):
        return self._vitals


def _health_text(control):
    from goldenswing.alerts.commands import CommandContext
    from goldenswing.execution.mt5_commands import _mt5_health
    ctx = CommandContext(
        feed_status={"ok": True}, last_price=4176.4,
        last_bar_time="2026-11-09T09:00:00+00:00",
        mt5_control=control, live_positions=[{"ticket": 1}],
        pending_orders=[])
    return _mt5_health(ctx, []).text


def test_health_shows_vitals_above_the_fold():
    text = _health_text(_VitalsControl({
        "ok": True, "error": "", "balance": 9_250.0,
        "equity": 9_310.5, "floating_pnl": 60.5, "margin": 120.0,
        "free_margin": 9_190.5, "currency": "USD",
        "peak_balance": 12_400.0, "day_start_balance": 10_000.0,
        "drawdown_pct": 0.075,
        "distance_pct": 0.075, "limit_pct": 0.15, "breached": False,
        "halted": False, "halt_reason": ""}))
    assert "💰 Account Vitals" in text
    assert "`9,250.00`" in text and "`9,310.50`" in text
    assert "`+60.50`" in text and "open positions: 1" in text
    assert "`7.5%`" in text and "room to the 15% halt" in text
    # Day start drives the drawdown; the peak still shows as info.
    assert "day start: `10,000.00`" in text
    assert "peak balance: `12,400.00`" in text
    # Above the fold: vitals precede the journal/positions sections.
    assert text.index("💰 Account Vitals") < \
        text.index("📈 Open Positions")


def test_health_vitals_unavailable_renders_reason_not_zero():
    text = _health_text(_VitalsControl(
        {"ok": False, "error": "account read failed: terminal down"}))
    assert "⚠️ account vitals unavailable" in text
    assert "terminal down" in text
    assert "0.00" not in text.split("💰 Account Vitals")[1] \
        .split("📈 Open Positions")[0]


def test_health_vitals_halt_boundary_shows_halted():
    text = _health_text(_VitalsControl({
        "ok": True, "error": "", "balance": 8_400.0,
        "equity": 8_400.0, "floating_pnl": 0.0, "margin": None,
        "free_margin": None, "currency": "USD",
        "peak_balance": 10_000.0, "day_start_balance": 10_000.0,
        "drawdown_pct": 0.16,
        "distance_pct": -0.01, "limit_pct": 0.15, "breached": True,
        "halted": True,
        "halt_reason": "daily drawdown hit (16.0% from day start "
                       "10000.00 >= 15%)"}))
    assert "⛔ HALTED" in text and "daily drawdown hit" in text


def test_health_without_account_read_shows_unavailable():
    class _NoVitals:
        pass

    text = _health_text(_NoVitals())
    assert "⚠️ account vitals unavailable" in text


# -------------------------------------------- callback edit path (runner)

def test_edit_command_result_single_part_edits_with_html():
    from goldenswing.alerts.commands import CommandResult
    from goldenswing.execution.mt5_runner import MT5Runner
    calls = []
    fake = SimpleNamespace(
        _tg_api=lambda method, payload: calls.append((method, payload)),
        _reply_command=lambda result: calls.append(("reply", result)))
    MT5Runner._edit_command_result(
        fake, "1", 42, CommandResult(text="📡 Feed\n  mid: `1.0`"),
        fallback_markup={"inline_keyboard": []})
    assert len(calls) == 1 and calls[0][0] == "editMessageText"
    payload = calls[0][1]
    assert payload["parse_mode"] == "HTML"
    assert payload["text"] == "<b>📡 Feed</b>\n  mid: <code>1.0</code>"


def test_edit_command_result_long_result_falls_back_to_split_send():
    from goldenswing.alerts.commands import CommandResult
    from goldenswing.execution.mt5_runner import MT5Runner
    calls = []
    fake = SimpleNamespace(
        _tg_api=lambda method, payload: calls.append((method, payload)),
        _reply_command=lambda result: calls.append(("reply", result)))
    kb = {"inline_keyboard": [[{"text": "x", "callback_data": "y"}]]}
    MT5Runner._edit_command_result(
        fake, "1", 42, CommandResult(text=_maximal_status()),
        fallback_markup=kb)
    assert len(calls) == 1 and calls[0][0] == "reply"
    # The fallback carries the effective keyboard into the send path.
    assert calls[0][1].reply_markup == kb
