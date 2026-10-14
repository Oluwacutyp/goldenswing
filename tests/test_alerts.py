"""Alert service: composition, dedup/cooldown, outbox fallback for
both senders, and the callback interface (no fake execution)."""
import pandas as pd
import pytest

from goldenswing.alerts import (AlertAction, AlertService, CallbackRegistry,
                                EmailSender, TelegramSender, build_alert,
                                make_callback_data, parse_callback_data)
from goldenswing.alerts.telegram import inline_keyboard
from goldenswing.golden.detector import GoldenSignal


def _signal(index=500, direction=1, score=78.0):
    return GoldenSignal(
        index=index, timestamp=pd.Timestamp("2025-03-01 13:00", tz="UTC"),
        direction=direction, score=score, kind="sweep_reversal",
        factors=["liquidity sweep of sell-side pool @ 2,900.0, 2 bars ago",
                 "CHoCH confirms the sweep reversal"],
        entry_zone=(2909.0, 2913.0), entry_ref=2912.0,
        invalidation=2901.0, targets=[2925.0, 2941.0], atr=10.0)


def _alert(**kw):
    return build_alert(_signal(**kw), instrument="XAUUSD", mode="swing",
                       chart_png=b"\x89PNG-fake",
                       forecast_note="Forecaster: P(move) 0.61")


# ------------------------------------------------------------- composition
def test_alert_text_carries_the_thesis():
    a = _alert()
    assert a.alert_id == "XAU500L"
    for needle in ("LONG", "XAUUSD", "score 78", "GOLDEN",
                   "Entry: 2,912.0", "Stop (invalidation): 2,901.0",
                   "T1 2,925.0", "Backing:", "CHoCH confirms",
                   "Forecaster: P(move) 0.61"):
        assert needle in a.text, needle
    assert a.direction_label == "LONG"
    assert a.risk_distance() == pytest.approx(11.0)


# -------------------------------------------------------------- callbacks
def test_callback_round_trip():
    data = make_callback_data(AlertAction.EXECUTE, "XAU500L")
    assert data == "gs:execute:XAU500L"
    action, alert_id = parse_callback_data(data)
    assert action is AlertAction.EXECUTE and alert_id == "XAU500L"
    with pytest.raises(ValueError):
        parse_callback_data("bogus")
    with pytest.raises(ValueError):
        parse_callback_data("gs:dance:XAU500L")


def test_keyboard_has_three_wired_buttons():
    kb = inline_keyboard(_alert())
    row = kb["inline_keyboard"][0]
    assert [b["text"] for b in row] == ["✅ Execute", "📄 Paper", "⏭ Skip"]
    actions = [parse_callback_data(b["callback_data"])[0] for b in row]
    assert actions == [AlertAction.EXECUTE, AlertAction.PAPER,
                       AlertAction.SKIP]


class _StubHandler:
    def __init__(self):
        self.calls = []

    def handle(self, action, alert):
        self.calls.append((action, alert.alert_id))
        return f"did {action.value} on {alert.alert_id}"


def test_registry_dispatch_and_no_fake_execution():
    reg = CallbackRegistry()
    data = make_callback_data(AlertAction.PAPER, "XAU500L")
    # no handler registered -> NOT handled, and says why (never fakes)
    res = reg.dispatch(data, {"XAU500L": _alert()})
    assert not res.handled and "not wired" in res.message
    # unknown alert
    res = reg.dispatch(data, {})
    assert not res.handled and "unknown alert" in res.message
    # with a handler -> handled, handler's reply propagated
    handler = _StubHandler()
    reg.register(handler)
    res = reg.dispatch(data, {"XAU500L": _alert()})
    assert res.handled and res.message == "did paper on XAU500L"
    assert handler.calls == [(AlertAction.PAPER, "XAU500L")]
    with pytest.raises(TypeError):
        reg.register(object())


# ---------------------------------------------------------------- senders
@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    for var in ("GS_TELEGRAM_BOT_TOKEN", "GS_TELEGRAM_CHAT_ID",
                "GS_SMTP_HOST", "GS_SMTP_USER", "GS_SMTP_PASSWORD",
                "GS_ALERT_EMAIL_TO", "GS_ALERT_OUTBOX"):
        monkeypatch.delenv(var, raising=False)


def test_telegram_outbox_fallback(tmp_path):
    s = TelegramSender(outbox_dir=tmp_path)
    assert s.mode == "outbox"
    res = s.send(_alert())
    assert not res.delivered and res.mode == "outbox"
    names = {p.name for p in tmp_path.iterdir()}
    assert "XAU500L.telegram.png" in names
    assert "XAU500L.telegram.txt" in names
    assert "XAU500L.json" in names
    txt = (tmp_path / "XAU500L.telegram.txt").read_text()
    assert "NOT sent" in txt and "LONG" in txt


def test_email_outbox_fallback_and_message(tmp_path):
    s = EmailSender(outbox_dir=tmp_path)
    assert s.mode == "outbox"
    res = s.send(_alert())
    assert not res.delivered and res.mode == "outbox"
    assert (tmp_path / "XAU500L.email.txt").exists()
    live = EmailSender(host="smtp.example.com", recipient="a@b.c",
                       user="u@x.y")
    msg = live.build_message(_alert())
    assert "LONG" in msg["Subject"]
    atts = list(msg.iter_attachments())
    assert len(atts) == 1
    assert atts[0].get_content_type() == "image/png"


def test_senders_go_live_only_with_credentials():
    assert TelegramSender(bot_token="t", chat_id="1").mode == "live"
    assert TelegramSender(bot_token="t").mode == "outbox"
    assert EmailSender(host="h").mode == "outbox"
    assert EmailSender(host="h", recipient="r@x").mode == "live"


# ----------------------------------------------------------------- service
class _RecordingSender:
    def __init__(self):
        self.sent = []

    def send(self, alert):
        from goldenswing.alerts.model import SendResult
        self.sent.append(alert.alert_id)
        return SendResult("stub", False, "outbox", "recorded")


def _stub_render(df, state, signal, **kw):
    return b"\x89PNG-stub"


def test_service_dedup_and_cooldown():
    sender = _RecordingSender()
    svc = AlertService([sender], cooldown_bars=24, chart_renderer=_stub_render)
    a1 = svc.process(None, None, _signal(index=500))
    assert a1 is not None and sender.sent == ["XAU500L"]
    # same signal again -> seen id
    assert svc.process(None, None, _signal(index=500)) is None
    # same direction inside cooldown -> suppressed
    assert svc.process(None, None, _signal(index=510)) is None
    # opposite direction is a different key -> alerts
    a2 = svc.process(None, None, _signal(index=512, direction=-1))
    assert a2 is not None and a2.alert_id == "XAU512S"
    # same direction after cooldown -> alerts again
    a3 = svc.process(None, None, _signal(index=530))
    assert a3 is not None
    assert sender.sent == ["XAU500L", "XAU512S", "XAU530L"]
    assert a1.text and a1.chart_png == b"\x89PNG-stub"


# ------------------------------------------------- telegram resilience
class _FakeResp:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body if body is not None else {"ok": True, "result": {"message_id": 7}}
        self.headers = {"content-type": "application/json"}
        self.text = str(self._body)

    def json(self):
        return self._body


class _FakeSession:
    """Scripted responses; records posts and sleeps are injected."""

    def __init__(self, script):
        self.script = list(script)   # each: _FakeResp or Exception
        self.posts = 0

    def post(self, url, **kw):
        self.posts += 1
        item = self.script.pop(0) if self.script else _FakeResp(200)
        if isinstance(item, Exception):
            raise item
        return item


def _live_sender(session, **kw):
    sleeps = []
    kw.setdefault("sleeper", sleeps.append)
    s = TelegramSender(bot_token="tok", chat_id="1", session=session, **kw)
    assert s.mode == "live"
    return s, sleeps


def test_telegram_retries_transient_then_succeeds():
    import requests
    sess = _FakeSession([_FakeResp(500, {"ok": False}),
                         _FakeResp(503, {"ok": False}),
                         _FakeResp(200)])
    s, sleeps = _live_sender(sess, max_retries=3, backoff_base=1.0)
    res = s.send_text("hello")
    assert res.delivered and res.mode == "live"
    assert sess.posts == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff


def test_telegram_fatal_does_not_retry():
    sess = _FakeSession([_FakeResp(401, {"ok": False, "description": "unauthorized"})])
    s, sleeps = _live_sender(sess, max_retries=3)
    res = s.send_text("hello")
    assert not res.delivered
    assert sess.posts == 1 and sleeps == []
    assert "fatal" in res.detail


def test_telegram_429_honours_retry_after():
    sess = _FakeSession([
        _FakeResp(429, {"ok": False, "parameters": {"retry_after": 5}}),
        _FakeResp(200)])
    s, sleeps = _live_sender(sess, max_retries=2, backoff_base=1.0)
    res = s.send_text("hello")
    assert res.delivered
    assert sleeps == [5.0]  # API's wait wins over the backoff schedule


def test_telegram_network_error_retries():
    import requests
    sess = _FakeSession([requests.exceptions.ConnectionError("down"),
                         _FakeResp(200)])
    s, sleeps = _live_sender(sess, max_retries=2)
    res = s.send_text("hello")
    assert res.delivered and sess.posts == 2


def test_telegram_photo_failure_falls_back_to_text():
    sess = _FakeSession([_FakeResp(500, {"ok": False})] * 4 + [_FakeResp(200)])
    s, sleeps = _live_sender(sess, max_retries=3, backoff_base=0.0)
    res = s.send(_alert())
    assert res.delivered and res.mode == "live"
    assert "text fallback sent" in res.detail


def test_telegram_gives_up_after_max_retries():
    sess = _FakeSession([_FakeResp(500, {"ok": False})] * 10)
    s, sleeps = _live_sender(sess, max_retries=2, backoff_base=0.0)
    res = s.send_text("hello")
    assert not res.delivered
    assert sess.posts == 3  # 1 initial + 2 retries
    assert "gave up" in res.detail


def test_error_deduper_window():
    from goldenswing.alerts.telegram import ErrorDeduper
    d = ErrorDeduper(window_seconds=600.0)
    assert d.should_send("boom", now=1000.0)
    assert not d.should_send("boom", now=1200.0)      # same text, in window
    assert d.should_send("different failure", now=1200.0)
    assert d.should_send("boom", now=1700.0)          # window expired
