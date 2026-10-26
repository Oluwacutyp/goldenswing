"""Inline navigation button tests (owner note 2026-10-25).

Read-only messages (assess, health, forecast_train) carry the nav
keyboard — Status / Assess all / Health / Train forecast — and the
callback router handles those taps. Status/position/pending keep
the pinned control keyboard (tests pin it exactly); the nav
keyboard is a separate markup, never a mutation of it.
"""

import json

from goldenswing.alerts.commands import CommandContext
from goldenswing.execution.mt5_commands import (
    _mt5_assess,
    _mt5_forecast_train,
    _mt5_health,
    mt5_nav_keyboard,
    mt5_nav_with_next,
)
from test_mt5_telegram_control import (
    OWNER,
    _callback,
    _runner,
    FakeApi,
)


class _Journal:
    def __init__(self, records):
        self._records = records

    def records(self):
        return list(self._records)


def _record(text, name="sweep_reversal"):
    return {"type": "evaluation", "kind": "playbook_eval",
            "playbook": name, "assessment_text": text}


def test_nav_keyboard_shape():
    kb = mt5_nav_keyboard()
    buttons = [b for row in kb["inline_keyboard"] for b in row]
    assert [b["callback_data"] for b in buttons] == [
        "mt5:status", "mt5:assess_all", "mt5:health", "mt5:forecast_train"]
    assert all("text" in b for b in buttons)


def test_assess_returns_carry_nav_keyboard(tmp_path):
    ctx = CommandContext(journal=_Journal([_record("short read")]))
    index = _mt5_assess(ctx, [])
    assert index.reply_markup == mt5_nav_with_next(
        "/assess <playbook>", "/assess all", "/status")
    single = _mt5_assess(ctx, ["sweep_reversal"])
    assert single.reply_markup == mt5_nav_with_next(
        "/assess all", "/status")
    all_text = _mt5_assess(ctx, ["all"])
    assert all_text.reply_markup == mt5_nav_with_next("/assess", "/status")
    long_ctx = CommandContext(
        journal=_Journal([_record("🧭 long\n" + "x" * 3600)]),
        reports_dir=tmp_path)
    doc = _mt5_assess(long_ctx, ["all"])
    assert doc.document is not None
    assert doc.reply_markup == mt5_nav_keyboard()


def test_health_and_forecast_train_carry_nav_keyboard():
    ctx = CommandContext(feed_status={}, journal=None,
                         live_positions=[], pending_orders=[])
    health = _mt5_health(ctx, [])
    assert health.reply_markup == mt5_nav_with_next(
        "/status", "/reconcile", "/journal")

    class Ctrl:
        def forecast_train_now(self):
            return "forecast_train: bundle current"

    trained = _mt5_forecast_train(
        CommandContext(mt5_control=Ctrl()), [])
    assert "bundle current" in trained.text
    assert trained.reply_markup == mt5_nav_keyboard()
    unavailable = _mt5_forecast_train(CommandContext(), [])
    assert unavailable.reply_markup == mt5_nav_keyboard()


def test_reply_command_passes_markup_for_documents(tmp_path):
    from goldenswing.alerts.commands import CommandResult

    runner = _runner(tmp_path)

    class NewSender:
        def __init__(self):
            self.docs = []

        def send_document(self, path, caption=None, reply_markup=None):
            self.docs.append((str(path), caption, reply_markup))

    sender = NewSender()
    runner.senders = [sender]
    path = tmp_path / "mt5-assess-x.md"
    path.write_text("🧭 test", encoding="utf-8")
    runner._reply_command(CommandResult(
        document=path, caption="cap", reply_markup=mt5_nav_keyboard()))
    assert sender.docs and sender.docs[0][2] == mt5_nav_keyboard()

    class LegacySender:
        def __init__(self):
            self.docs = []

        def send_document(self, path, caption=None):
            self.docs.append(str(path))

    legacy = LegacySender()
    runner.senders = [legacy]
    runner._reply_command(CommandResult(
        document=path, caption="cap", reply_markup=mt5_nav_keyboard()))
    assert legacy.docs == [str(path)]  # fallback, no crash


def test_callback_routing_for_nav_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("GS_TELEGRAM_CHAT_ID", OWNER)
    runner = _runner(tmp_path)
    api = FakeApi([])
    runner._tg_api = api
    runner._handle_callback_query(_callback("mt5:health"), {})
    edits = [p for m, p in api.calls if m == "editMessageText"]
    assert edits, api.calls
    markup = json.loads(edits[0]["reply_markup"])
    assert "mt5:assess_all" in json.dumps(markup)
    api.calls.clear()
    runner._handle_callback_query(_callback("mt5:forecast_train"), {})
    edits = [p for m, p in api.calls if m == "editMessageText"]
    assert edits and "forecast_train" in edits[0]["text"]
    api.calls.clear()
    runner._handle_callback_query(_callback("mt5:assess_all"), {})
    edits = [p for m, p in api.calls if m == "editMessageText"]
    assert edits  # empty journal -> text edit, not a document
    assert "mt5:status" in json.dumps(
        json.loads(edits[0]["reply_markup"]))
