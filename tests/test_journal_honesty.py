"""Journal honesty: tolerant reads, truthful cards, /db that answers.

Pins the fixes for the 2026-11-04 live failure shape: the MT5
journal held 16,170 records, /journal answered, and /db earned
SILENCE — the journal had outgrown Telegram's 50MB bot document
cap, the sender refused the file, and the reply path discarded
the refusal.  Also pins TradeJournal.records() surviving torn /
non-locale-encoded lines instead of failing the whole read, and
/journal rendering failures as failures (never "records: 0").
"""
import gzip
import json
import uuid
from pathlib import Path

import pytest

from goldenswing.alerts import commands as C
from goldenswing.alerts.commands import (CommandContext, CommandResult,
                                         CommandRouter)
from goldenswing.alerts.model import SendResult
from goldenswing.alerts.telegram import TelegramSender
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.mt5_commands import mt5_router

OWNER = "12345"


def _update(text: str, chat_id: str = OWNER) -> dict:
    return {"update_id": 1,
            "message": {"message_id": 1,
                        "chat": {"id": chat_id, "type": "private"},
                        "text": text}}


def _write_raw(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------ records()
def test_records_survive_torn_middle_line(tmp_path):
    path = tmp_path / "j.jsonl"
    good1 = json.dumps({"type": "note", "text": "one"})
    good2 = json.dumps({"type": "note", "text": "two"})
    _write_raw(path, [good1, '{"type": "note", "text": "tor', good2])
    j = TradeJournal(path)
    records, bad = j.records_with_health()
    assert [r["text"] for r in records] == ["one", "two"]
    assert bad == 1
    assert j.last_read_bad_lines == 1
    # records() keeps its plain signature for the 12 call sites.
    assert [r["text"] for r in j.records()] == ["one", "two"]


def test_records_survive_torn_final_line(tmp_path):
    path = tmp_path / "j.jsonl"
    good = json.dumps({"type": "note", "text": "complete"})
    # Crash mid-append: partial last line, no trailing newline.
    path.write_bytes((good + "\n" + '{"type": "open", "alert_id": "x')
                     .encode("utf-8"))
    j = TradeJournal(path)
    records, bad = j.records_with_health()
    assert [r["text"] for r in records] == ["complete"]
    assert bad == 1


def test_records_read_utf8_beyond_cp1252(tmp_path):
    path = tmp_path / "j.jsonl"
    note = {"type": "note", "text": "sweep → entry, paid ₦1500 🎯"}
    path.write_text(json.dumps(note, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    j = TradeJournal(path)
    records, bad = j.records_with_health()
    assert bad == 0
    assert records[0]["text"] == "sweep → entry, paid ₦1500 🎯"


def test_records_append_roundtrip_still_clean(tmp_path):
    j = TradeJournal(tmp_path / "j.jsonl")
    j.log_note("plain")
    j.log_open(alert_id="a1", instrument="XAUUSD", mode="swing",
               playbook="fvg_midline", phase="m", direction=1, score=70,
               source="paper", ticket=1, entry=2600.0, stop=2590.0,
               volume=1.0, risk_money=100.0)
    records, bad = j.records_with_health()
    assert bad == 0 and len(records) == 2
    assert j.last_read_bad_lines == 0


def test_records_missing_file_is_empty_not_error(tmp_path):
    j = TradeJournal(tmp_path / "nope.jsonl")
    assert j.records() == []
    assert j.records_with_health() == ([], 0)


# ------------------------------------------------------------ /journal cards
class _RaisingJournal:
    """A journal whose every read blows up (unreadable file)."""

    def records(self):
        raise RuntimeError("disk gone")


def _ctx(journal, tmp_path, *, reports=True):
    return CommandContext(
        journal=journal,
        journal_path=getattr(journal, "path", tmp_path / "j.jsonl"),
        reports_dir=tmp_path / "reports" if reports else None)


def test_journal_card_reports_read_failure_honestly(tmp_path):
    router = CommandRouter(owner_chat_id=OWNER)
    ctx = _ctx(_RaisingJournal(), tmp_path)
    result, _ = router.dispatch(_update("/journal"), ctx)
    assert "journal read failed" in result.text
    assert "RuntimeError" in result.text
    assert "records: 0" not in result.text


def test_mt5_journal_card_reports_read_failure_honestly(tmp_path):
    router = mt5_router(owner_chat_id=OWNER)
    ctx = _ctx(_RaisingJournal(), tmp_path)
    result, _ = router.dispatch(_update("/journal"), ctx)
    assert "journal read failed" in result.text
    assert "records: 0" not in result.text


def test_journal_card_counts_unreadable_lines(tmp_path):
    path = tmp_path / "j.jsonl"
    _write_raw(path, [json.dumps({"type": "note", "text": "ok"}),
                      '{"torn'])
    j = TradeJournal(path)
    router = CommandRouter(owner_chat_id=OWNER)
    result, _ = router.dispatch(_update("/journal"), _ctx(j, tmp_path))
    assert "records: 1" in result.text
    assert "1 unreadable line skipped" in result.text


def test_mt5_journal_card_uses_real_fields_not_seq_question_mark(tmp_path):
    j = TradeJournal(tmp_path / "mt5_journal.jsonl")
    j.log_open(alert_id="MT5-fvg-1", instrument="XAUUSD", mode="MT5",
               playbook="fvg_midline", phase="live", direction=1,
               score=70, source="mt5", ticket=1, entry=4160.0,
               stop=4150.0, volume=0.01, risk_money=10.0)
    j.log_note("boot complete")
    router = mt5_router(owner_chat_id=OWNER)
    ctx = _ctx(j, tmp_path)
    result, _ = router.dispatch(_update("/journal"), ctx)
    assert "records: 2" in result.text
    assert "seq=" not in result.text       # MT5 records carry no seq
    assert "type=`note`" in result.text
    # An open as the last record shows its alert_id + playbook.
    j.log_open(alert_id="MT5-sweep-2", instrument="XAUUSD", mode="MT5",
               playbook="sweep_reversal", phase="live", direction=-1,
               score=66, source="mt5", ticket=2, entry=4150.0,
               stop=4160.0, volume=0.01, risk_money=10.0)
    result, _ = router.dispatch(_update("/journal"), ctx)
    assert "alert_id=`MT5-sweep-2`" in result.text
    assert "playbook=`sweep_reversal`" in result.text


# ------------------------------------------------------------ /db sizing
def _padded_journal(tmp_path, n=40, pad="x" * 120):
    j = TradeJournal(tmp_path / "mt5_journal.jsonl")
    for i in range(n):
        j.log_note(f"note {i} {pad}")
    return j


def test_db_oversize_sends_gzip_with_explanation(tmp_path, monkeypatch):
    j = _padded_journal(tmp_path, n=60)
    monkeypatch.setattr(C, "DB_DOCUMENT_LIMIT_BYTES", 3_000)
    router = CommandRouter(owner_chat_id=OWNER)
    result, _ = router.dispatch(_update("/db"), _ctx(j, tmp_path))
    # The raw file was NOT the document; the gzip was, with an
    # explanation that names the cap.
    assert result.document != j.path
    assert str(result.document).endswith(".gz")
    assert "50MB" in result.text and "gzip" in result.text
    with gzip.open(result.document, "rt", encoding="utf-8") as fh:
        restored = [json.loads(line) for line in fh if line.strip()]
    assert len(restored) == 60          # lossless: every record


def test_mt5_db_oversize_sends_gzip_too(tmp_path, monkeypatch):
    j = _padded_journal(tmp_path, n=60)
    monkeypatch.setattr(C, "DB_DOCUMENT_LIMIT_BYTES", 3_000)
    router = mt5_router(owner_chat_id=OWNER)
    result, _ = router.dispatch(_update("/db"), _ctx(j, tmp_path))
    assert str(result.document).endswith(".gz")
    assert result.reply_markup is not None   # navigation rides along


def test_db_huge_and_incompressible_sends_labelled_slice(
        tmp_path, monkeypatch):
    # Random hex barely compresses: raw AND gzip stay over the cap.
    j = TradeJournal(tmp_path / "mt5_journal.jsonl")
    for _ in range(30):
        j.log_note("".join(uuid.uuid4().hex for _ in range(8)))
    monkeypatch.setattr(C, "DB_DOCUMENT_LIMIT_BYTES", 2_000)
    monkeypatch.setattr(C, "DB_ARTIFACT_TARGET_BYTES", 1_200)
    router = CommandRouter(owner_chat_id=OWNER)
    result, _ = router.dispatch(_update("/db"), _ctx(j, tmp_path))
    assert result.document != j.path
    assert str(result.document).endswith("_recent.jsonl")
    assert "newest" in result.text and "→" in result.text
    lines = Path(result.document).read_text(
        encoding="utf-8").strip().splitlines()
    assert 0 < len(lines) < 30           # a slice, not the whole file
    newest = j.records()[-1]
    assert json.loads(lines[-1])["ts"] == newest["ts"]


def test_db_fitting_journal_still_sends_raw_file(tmp_path):
    j = _padded_journal(tmp_path, n=2)
    router = CommandRouter(owner_chat_id=OWNER)
    result, _ = router.dispatch(_update("/db"), _ctx(j, tmp_path))
    assert Path(result.document) == j.path
    assert result.text is None


def test_db_gzip_failure_is_a_text_answer(tmp_path, monkeypatch):
    j = _padded_journal(tmp_path)
    monkeypatch.setattr(C, "DB_DOCUMENT_LIMIT_BYTES", 1_000)

    def _boom(src, out):
        raise OSError("no space left")

    monkeypatch.setattr(C, "_gzip_journal", _boom)
    router = CommandRouter(owner_chat_id=OWNER)
    result, _ = router.dispatch(_update("/db"), _ctx(j, tmp_path))
    assert result.document is None
    assert "50MB" in result.text and "no space left" in result.text


# ------------------------------------------------------------ reply path
class _RecordingSender(TelegramSender):
    """A TelegramSender (LiveRunner._telegram_sender isinstance-
    filters for one) that records instead of sending."""

    def __init__(self, *, doc_result=None, doc_exc=None):
        super().__init__(bot_token="tok", chat_id=OWNER)
        self.texts: list[str] = []
        self.documents: list[str] = []
        self._doc_result = doc_result
        self._doc_exc = doc_exc

    def send_text(self, text, silent=False, reply_markup=None):
        self.texts.append(text)
        return SendResult("telegram", True, "live", "ok")

    def send_document(self, path, caption=None, reply_markup=None):
        self.documents.append(str(path))
        if self._doc_exc is not None:
            raise self._doc_exc
        return self._doc_result or SendResult(
            "telegram", True, "live", "ok")


def _bare_mt5_runner(tmp_path, sender):
    from goldenswing.execution.mt5_runner import MT5Runner
    runner = MT5Runner.__new__(MT5Runner)
    runner.senders = [sender]
    runner.journal = TradeJournal(tmp_path / "mt5_journal.jsonl")
    runner.executor = None
    return runner


def test_reply_command_surfaces_refused_document(tmp_path):
    # The live /db silence: sendDocument refuses (file over the
    # cap) and returns delivered=False — the owner must hear why.
    sender = _RecordingSender(doc_result=SendResult(
        "telegram", False, "live",
        "mt5_journal.jsonl is 61.0MB — over Telegram's 50MB bot "
        "document limit"))
    runner = _bare_mt5_runner(tmp_path, sender)
    runner._reply_command(CommandResult(
        document=tmp_path / "mt5_journal.jsonl", caption="📓 MT5 journal"))
    assert any("document not delivered" in t and "50MB" in t
               for t in sender.texts)
    notes = [r.get("text", "") for r in runner.journal.records()]
    assert any("document not delivered" in n for n in notes)


def test_reply_command_surfaces_send_exception(tmp_path):
    sender = _RecordingSender(doc_exc=RuntimeError("network down"))
    runner = _bare_mt5_runner(tmp_path, sender)
    runner._reply_command(CommandResult(
        document=tmp_path / "x.jsonl", caption="cap"))
    assert any("command reply failed" in t and "network down" in t
               for t in sender.texts)


def test_reply_command_sends_text_then_document(tmp_path):
    sender = _RecordingSender()
    runner = _bare_mt5_runner(tmp_path, sender)
    runner._reply_command(CommandResult(
        text="explanation first", document=tmp_path / "x.jsonl",
        caption="cap"))
    assert sender.texts[0] == "explanation first"
    assert sender.documents == [str(tmp_path / "x.jsonl")]


def test_dispatch_command_surfaces_refused_document(
        tmp_path, monkeypatch):
    from goldenswing.execution.runner import LiveRunner, RunnerConfig
    monkeypatch.setenv("GS_TELEGRAM_CHAT_ID", OWNER)
    monkeypatch.delenv("GS_TELEGRAM_BOT_TOKEN", raising=False)
    sender = _RecordingSender(doc_result=SendResult(
        "telegram", False, "live", "file is 61.0MB — too big"))
    cfg = RunnerConfig(
        journal_path=tmp_path / "journal.jsonl",
        risk_state_path=tmp_path / "risk.json",
        kill_switch_path=tmp_path / "KILL",
        state_dir=tmp_path / "live_state",
        forecast_bundle_path=None)
    runner = LiveRunner(cfg, senders=[sender])
    runner.journal.log_note("hello")
    runner._dispatch_command(_update("/db"))
    assert sender.documents == [str(tmp_path / "journal.jsonl")]
    assert any("too big" in t for t in sender.texts)


# ------------------------------------------------------------ trust check
def test_position_state_trustworthy_flags_damaged_journal(tmp_path):
    runner = _bare_mt5_runner(tmp_path, _RecordingSender())
    j = runner.journal
    j.log_note("clean")
    ok, why = runner._position_state_trustworthy()
    assert ok is False and "no executor" in why   # journal itself fine
    with j.path.open("a", encoding="utf-8") as fh:
        fh.write('{"type": "note", "text": "tor\n')
    ok, why = runner._position_state_trustworthy()
    assert ok is False and "journal corrupt" in why
    assert "1 unreadable" in why
