"""Telegram command surface + paper --once tick.

Covers: command parsing/gating (owner-only, silent for strangers),
file generation (/report, /db), status/trades readouts, the
PaperExecutor export/import round-trip the cron --once mode depends
on, sendDocument, and the runner's shared-poller wiring.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from goldenswing.alerts import commands as C
from goldenswing.alerts.commands import (COMMANDS, CommandContext,
                                         CommandResult, CommandRouter,
                                         command)
from goldenswing.alerts.telegram import TelegramSender
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.paper import PaperExecutor
from goldenswing.execution.reporting import (build_report_file,
                                             iter_paper_trades,
                                             load_measured_oos,
                                             paper_book_stats)


OWNER = "12345"
STRANGER = "99999"


def _update(text: str, chat_id: str = OWNER, update_id: int = 1) -> dict:
    return {"update_id": update_id,
            "message": {"message_id": 1,
                        "chat": {"id": chat_id, "type": "private"},
                        "text": text}}


@pytest.fixture
def journal(tmp_path):
    j = TradeJournal(tmp_path / "journal.jsonl")
    # two closed trades: fvg +1.5R win, trendline -1R loss
    j.log_open(alert_id="a1", instrument="XAUUSD", mode="swing",
               playbook="fvg_midline", phase="m", direction=1, score=70,
               source="paper", ticket=1, entry=2600.0, stop=2590.0,
               volume=1.0, risk_money=100.0)
    j.log_close(alert_id="a1", ticket=1, exit_price=2615.0, reason="trail",
                pnl_money=150.0, r_net=1.5, hold_bars=5)
    j.log_open(alert_id="a2", instrument="XAUUSD", mode="swing",
               playbook="trendline_retest", phase="m", direction=-1,
               score=65, source="paper", ticket=2, entry=2620.0,
               stop=2630.0, volume=1.0, risk_money=100.0)
    j.log_close(alert_id="a2", ticket=2, exit_price=2630.0, reason="stop",
                pnl_money=-100.0, r_net=-1.0, hold_bars=3)
    return j


@pytest.fixture
def ctx(journal, tmp_path):
    return CommandContext(
        journal=journal, journal_path=journal.path,
        measured_oos=load_measured_oos(), reports_dir=tmp_path / "reports")


@pytest.fixture
def router():
    return CommandRouter(owner_chat_id=OWNER)


# ------------------------------------------------------------ gating
def test_non_command_ignored(router, ctx):
    assert router.dispatch(_update("hello"), ctx) == (None, None)


def test_stranger_ignored_silently(router, ctx):
    # Even a valid command from another chat: nothing, no reply.
    assert router.dispatch(_update("/help", chat_id=STRANGER), ctx) == \
        (None, None)
    assert router.dispatch(_update("/db", chat_id=777), ctx) == (None, None)


def test_unconfigured_owner_ignores_everyone(ctx):
    r = CommandRouter(owner_chat_id=None)
    assert r.dispatch(_update("/help"), ctx) == (None, None)


def test_unknown_command_helpful_to_owner(router, ctx):
    result, chat = router.dispatch(_update("/nope"), ctx)
    assert chat == OWNER
    assert "unknown command" in result.text


def test_help_lists_commands(router, ctx):
    result, _ = router.dispatch(_update("/help"), ctx)
    for name in ("/report", "/db", "/status", "/trades", "/help", "/harvest"):
        assert name in result.text


# ------------------------------------------------------------ /harvest
def test_harvest_bundles_journal_state_results(router, ctx, tmp_path):
    import zipfile
    # Extra state files the harvest should pick up from the data dir.
    data_dir = Path(ctx.journal_path).parent
    (data_dir / "risk_state.json").write_text('{"risk": "ok"}')
    (data_dir / "confluence_journal.jsonl").write_text('{"type": "note"}\n')
    (data_dir / "paper_state").mkdir(exist_ok=True)
    (data_dir / "paper_state" / "paper.json").write_text(
        '{"balance": 9999.0, "pending_orders": [], "tracked": {}, '
        '"next_ticket": 3}')
    (data_dir / "paper_state" / "meta.json").write_text('{"last_bar_time": "x"}')
    # Bulky market data must NEVER land in the bundle.
    (data_dir / "OANDA_XAU_USD_H1_2y.csv").write_text("a,b\n" * 1000)
    result, _ = router.dispatch(_update("/harvest"), ctx)
    zpath = Path(result.document)
    assert zpath.exists() and zpath.suffix == ".zip"
    assert "harvest" in (result.caption or "")
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
        assert "RESULTS.md" in names and "manifest.json" in names
        assert "trade_journal.jsonl" not in names  # fixture journal is journal.jsonl
        assert "journal.jsonl" in names
        assert "risk_state.json" in names
        assert "paper_state/paper.json" in names
        assert "paper_state/meta.json" in names
        assert "confluence_journal.jsonl" in names
        assert not any(n.endswith(".csv") for n in names)
        results = zf.read("RESULTS.md").decode()
        assert "+0.50R" in results  # 1.5 - 1.0 from the fixture book
        assert "fvg_midline" in results
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["scope"] == "all"
        assert manifest["files"]
        assert all(f["sha256"] for f in manifest["files"])
        # Every listed file's hash matches the bytes in the ZIP.
        import hashlib
        for f in manifest["files"]:
            blob = zf.read(f["arcname"])
            assert hashlib.sha256(blob).hexdigest() == f["sha256"]
            assert len(blob) == f["bytes"]


def test_harvest_scopes(router, ctx, tmp_path):
    import zipfile
    data_dir = Path(ctx.journal_path).parent
    (data_dir / "risk_state.json").write_text('{"risk": "ok"}')
    (data_dir / "paper_state").mkdir(exist_ok=True)
    (data_dir / "paper_state" / "paper.json").write_text('{"balance": 1}')
    result, _ = router.dispatch(_update("/harvest journal"), ctx)
    with zipfile.ZipFile(result.document) as zf:
        names = set(zf.namelist())
        assert "journal.jsonl" in names
        assert "risk_state.json" not in names
    result, _ = router.dispatch(_update("/harvest state"), ctx)
    with zipfile.ZipFile(result.document) as zf:
        names = set(zf.namelist())
        assert "risk_state.json" in names or "paper_state/paper.json" in names
        assert "journal.jsonl" not in names
    result, _ = router.dispatch(_update("/harvest bogus"), ctx)
    assert "usage" in result.text


def test_harvest_no_journal(router, tmp_path):
    c = CommandContext(journal=None,
                       journal_path=tmp_path / "nope.jsonl",
                       reports_dir=tmp_path / "reports")
    result, _ = router.dispatch(_update("/harvest"), c)
    assert "no journal" in result.text


def test_extension_contract(router, ctx):
    @command("/pingxyz", "test-only command")
    def _ping(c, args):
        return CommandResult(text=f"pong {' '.join(args)}")

    try:
        result, _ = router.dispatch(_update("/pingxyz hi"), ctx)
        assert result.text == "pong hi"
        result, _ = router.dispatch(_update("/help"), ctx)
        assert "/pingxyz" in result.text
    finally:
        del COMMANDS["/pingxyz"]


def test_command_exception_does_not_raise(router, ctx):
    @command("/boomxyz", "test-only")
    def _boom(c, args):
        raise RuntimeError("kablam")

    try:
        result, _ = router.dispatch(_update("/boomxyz"), ctx)
        assert "failed" in result.text and "kablam" in result.text
    finally:
        del COMMANDS["/boomxyz"]


# ------------------------------------------------------------ /trades
def test_trades_lists_r_multiples(router, ctx):
    result, _ = router.dispatch(_update("/trades 2"), ctx)
    assert "+1.50R" in result.text and "-1.00R" in result.text
    assert "fvg_midline" in result.text


def test_trades_defaults_and_clamps(router, ctx):
    result, _ = router.dispatch(_update("/trades"), ctx)
    assert "paper trades" in result.text
    result, _ = router.dispatch(_update("/trades 99"), ctx)
    assert result.text is not None  # clamped to 20, not an error
    result, _ = router.dispatch(_update("/trades abc"), ctx)
    assert "usage" in result.text


# ------------------------------------------------------------ files
def test_db_sends_journal_path(router, ctx):
    result, _ = router.dispatch(_update("/db"), ctx)
    assert Path(result.document) == ctx.journal_path
    assert "journal" in (result.caption or "")


def test_db_missing_journal(router, tmp_path):
    c = CommandContext(journal=None,
                       journal_path=tmp_path / "nope.jsonl")
    result, _ = router.dispatch(_update("/db"), c)
    assert "not found" in result.text


def test_report_builds_fresh_file(router, ctx):
    result, _ = router.dispatch(_update("/report"), ctx)
    path = Path(result.document)
    assert path.exists()
    body = path.read_text()
    assert "fvg_midline" in body and "trendline_retest" in body
    assert "+0.50" in body          # book total R: 1.5 - 1.0
    # measured-OOS section comes from the versioned records, not code
    assert "measured_oos.json" in body
    assert "sweep_reversal" in body  # FAILED walk-forward row present


# ------------------------------------------------------------ /status
def _paper_with_open():
    from goldenswing.execution.executor import OrderRequest
    paper = PaperExecutor(balance=10_000.0)
    req = OrderRequest(symbol="XAUUSD", direction=1, volume=1.0,
                       stop_loss=2590.0, entry_ref=2600.0, atr=5.0,
                       risk_money=100.0, alert_id="a9", bar_index=10)
    paper.place_order(req)
    bar = {"index": 11, "open": 2601.0, "high": 2605.0,
           "low": 2599.0, "close": 2603.0}
    paper.advance(bar)              # fill at the open
    return paper


def test_status_reports_health(router, journal, tmp_path):
    paper = _paper_with_open()
    c = CommandContext(
        journal=journal, journal_path=journal.path, paper=paper,
        feed_status={"ok": False, "reason": "no successful poll yet"},
        last_price=2603.0, last_bar_time="2026-10-11T12:00:00+00:00",
        measured_oos=[], reports_dir=tmp_path)
    result, _ = router.dispatch(_update("/status"), c)
    t = result.text
    assert "STALE" in t and "no successful poll yet" in t
    assert "#1" in t and "LONG" in t            # the open position
    assert "today (UTC)" in t
    assert "last signal" in t


def test_status_ok_feed(router, journal, tmp_path):
    c = CommandContext(
        journal=journal, journal_path=journal.path, paper=None,
        feed_status={"ok": True, "reason": None},
        measured_oos=[], reports_dir=tmp_path)
    result, _ = router.dispatch(_update("/status"), c)
    assert "feed: OK" in result.text


# ------------------------------------------------------------ reporting
def test_iter_paper_trades_joins_open_close(journal):
    trades = iter_paper_trades(journal)
    assert len(trades) == 2
    assert trades[0]["playbook"] == "fvg_midline"
    assert trades[0]["r_net"] == 1.5
    assert trades[0]["entry"] == 2600.0 and trades[0]["exit"] == 2615.0


def test_paper_book_stats(journal):
    stats = paper_book_stats(journal)
    assert stats["per_playbook"]["fvg_midline"]["total_r"] == 1.5
    assert stats["per_playbook"]["trendline_retest"]["total_r"] == -1.0
    assert stats["book"]["trades"] == 2
    assert stats["book"]["total_r"] == pytest.approx(0.5)


def test_load_measured_oos_has_provenance():
    rows = load_measured_oos()
    assert len(rows) >= 4
    names = {r["playbook"] for r in rows}
    assert {"fvg_midline", "trendline_retest", "sweep_reversal"} <= names
    for r in rows:
        assert r.get("source")  # every number carries its provenance


# ------------------------------------------------------------ paper state
def test_paper_export_import_roundtrip():
    paper = _paper_with_open()
    assert len(paper.positions()) == 1
    bal_before = paper.account_state().balance
    snap = paper.export_state()
    json.dumps(snap)               # must be JSON-serializable

    restored = PaperExecutor(balance=999.0)   # wrong on purpose
    restored.import_state(snap)
    assert restored.account_state().balance == pytest.approx(bal_before)
    pos = restored.positions()
    assert len(pos) == 1
    assert pos[0].entry_price == pytest.approx(2601.0 + 0.225, abs=1.0)
    assert pos[0].direction == 1

    # Management continues identically after the restore: run both
    # executors forward over the same bars and compare outcomes.
    bars = [{"index": 12 + i, "open": 2603.0, "high": 2604.0,
             "low": 2580.0 - i, "close": 2585.0} for i in range(5)]
    for b in bars:
        paper.advance(b)
        restored.advance(b)
    assert len(paper.positions()) == len(restored.positions()) == 0
    assert paper.account_state().balance == \
        pytest.approx(restored.account_state().balance)


def test_paper_shift_bar_indices():
    paper = _paper_with_open()
    pos = paper.positions()[0]
    entry_bar = paper._tracked[pos.ticket].entry_bar
    paper.shift_bar_indices(-50)
    assert paper._tracked[pos.ticket].entry_bar == entry_bar - 50


def test_state_store_paper_roundtrip(tmp_path):
    from goldenswing.execution.state_store import StateStore
    store = StateStore(tmp_path / "s")
    paper = _paper_with_open()
    store.save_paper(paper.export_state())
    loaded = store.load_paper()
    assert loaded["balance"] == pytest.approx(
        paper.account_state().balance)


# ------------------------------------------------------------ sendDocument
class _FakeResp:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
        self._body = body or {"ok": True, "result": {"message_id": 7}}

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self, resp=None):
        self.resp = resp or _FakeResp()
        self.calls = []

    def post(self, url, data=None, files=None, json=None, timeout=None):
        self.calls.append({"url": url, "data": data, "files": files})
        return self.resp


def _live_sender(tmp_path, session):
    return TelegramSender(bot_token="tok", chat_id=OWNER,
                          outbox_dir=tmp_path / "outbox", session=session)


def test_send_document_live(tmp_path):
    f = tmp_path / "report.md"
    f.write_text("# hi")
    sess = _FakeSession()
    res = _live_sender(tmp_path, sess).send_document(f, caption="cap")
    assert res.delivered
    call = sess.calls[0]
    assert call["url"].endswith("/sendDocument")
    assert call["files"]["document"][0] == "report.md"


def test_send_document_outbox_mode(tmp_path):
    f = tmp_path / "report.md"
    f.write_text("# hi")
    sender = TelegramSender(bot_token=None, chat_id=None,
                            outbox_dir=tmp_path / "outbox",
                            session=_FakeSession())
    res = sender.send_document(f)
    assert not res.delivered and res.mode == "outbox"


def test_send_document_missing_file(tmp_path):
    res = _live_sender(tmp_path, _FakeSession()).send_document(
        tmp_path / "nope.md")
    assert not res.delivered


def test_send_document_oversize_refused(tmp_path, monkeypatch):
    import goldenswing.alerts.telegram as tg
    monkeypatch.setattr(tg, "_DOCUMENT_LIMIT_MB", 0)
    f = tmp_path / "big.md"
    f.write_text("x")
    res = _live_sender(tmp_path, _FakeSession()).send_document(f)
    assert not res.delivered
    assert "over Telegram" in res.detail


# ------------------------------------------------------------ runner wiring
class _RecordingSender(TelegramSender):
    def __init__(self):
        super().__init__(bot_token="tok", chat_id=OWNER)
        self.texts = []
        self.documents = []

    def send_text(self, text, **kw):
        from goldenswing.alerts.model import SendResult
        self.texts.append(text)
        return SendResult("telegram", True, "live", "recorded")

    def send_document(self, path, **kw):
        from goldenswing.alerts.model import SendResult
        self.documents.append(str(path))
        return SendResult("telegram", True, "live", "recorded")


def _runner_with_sender(tmp_path, monkeypatch):
    from goldenswing.execution.runner import LiveRunner, RunnerConfig
    monkeypatch.setenv("GS_TELEGRAM_CHAT_ID", OWNER)
    monkeypatch.delenv("GS_TELEGRAM_BOT_TOKEN", raising=False)
    sender = _RecordingSender()
    cfg = RunnerConfig(
        journal_path=tmp_path / "journal.jsonl",
        risk_state_path=tmp_path / "risk.json",
        kill_switch_path=tmp_path / "KILL",
        state_dir=tmp_path / "live_state",
        forecast_bundle_path=None)
    runner = LiveRunner(cfg, senders=[sender])
    return runner, sender


def test_dispatch_command_answers_owner(tmp_path, monkeypatch):
    runner, sender = _runner_with_sender(tmp_path, monkeypatch)
    runner._dispatch_command(_update("/help"))
    assert len(sender.texts) == 1
    assert "/report" in sender.texts[0]


def test_dispatch_command_ignores_stranger(tmp_path, monkeypatch):
    runner, sender = _runner_with_sender(tmp_path, monkeypatch)
    runner._dispatch_command(_update("/db", chat_id=STRANGER))
    assert sender.texts == [] and sender.documents == []


def test_dispatch_command_sends_document(tmp_path, monkeypatch, journal):
    runner, sender = _runner_with_sender(tmp_path, monkeypatch)
    # point the runner at the fixture journal
    runner.journal = journal
    runner.cfg.journal_path = journal.path
    runner._dispatch_command(_update("/db"))
    assert sender.documents == [str(journal.path)]


# ------------------------------------------------------------ paper --once
class _FakeFeed:
    """Feed over an extendable frame: warm_start returns the initial
    segment; fetch_new returns whatever is newer than `since`."""

    def __init__(self, df: pd.DataFrame, warm_bars: int):
        self._df = df
        self._warm_bars = warm_bars

    def extend(self, df: pd.DataFrame):
        self._df = df

    def warm_start(self):
        return self._df.iloc[:self._warm_bars].copy()

    def fetch_new(self, since):
        return self._df[self._df.index > since].copy()

    def poll_status(self):
        return {"ok": True, "reason": None}

    @property
    def poll_interval(self):
        return 0


def _once_runner(tmp_path, monkeypatch):
    from goldenswing.execution.runner import LiveRunner, RunnerConfig
    monkeypatch.delenv("GS_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("GS_TELEGRAM_CHAT_ID", raising=False)
    cfg = RunnerConfig(
        journal_path=tmp_path / "journal.jsonl",
        risk_state_path=tmp_path / "risk.json",
        kill_switch_path=tmp_path / "KILL",
        state_dir=tmp_path / "live_state",
        forecast_bundle_path=None)
    return LiveRunner(cfg, senders=[])


def test_paper_once_cold_then_warm(tmp_path, monkeypatch):
    from goldenswing.backtest.data import load_gold_h1
    full = load_gold_h1().iloc[-600:].copy()
    feed = _FakeFeed(full, warm_bars=590)
    runner = _once_runner(tmp_path, monkeypatch)
    store_dir = tmp_path / "paper_state"

    # Tick 1: cold start — fast-forwards history, processes nothing new.
    s1 = runner.run_paper_once(feed, state_store_path=store_dir,
                               peers={}, poll_commands=False)
    assert s1["bars"] == 0
    assert (store_dir / "bars.pkl").exists()
    assert (store_dir / "meta.json").exists()

    # Tick 2: 10 new bars appear — they get processed, state advances.
    feed.extend(full)  # already the full frame; warm had 590 of 600
    runner2 = _once_runner(tmp_path, monkeypatch)
    s2 = runner2.run_paper_once(feed, state_store_path=store_dir,
                                 peers={}, poll_commands=False)
    assert s2["bars"] == 10
    meta = json.loads((store_dir / "meta.json").read_text())
    assert pd.Timestamp(meta["last_bar_time"]) == full.index[-1]

    # Tick 3: nothing new — a no-op tick that still exits cleanly.
    runner3 = _once_runner(tmp_path, monkeypatch)
    s3 = runner3.run_paper_once(feed, state_store_path=store_dir,
                                 peers={}, poll_commands=False)
    assert s3["bars"] == 0


def test_live_signal_never_rewinds_cursor():
    """Regression: 2026-10-13 21:38 UTC tick crashed every hour.

    After a rebase replay the fresh loop re-called live_signal for
    tail bars already processed (t=1997 after processed=1999), and
    the unconditional ``self._processed = t`` rewound the cursor.
    The next forward call re-fed bar 1998 to the strict-order
    FVGTracker (last=1999) -> ValueError. All three cursor playbooks
    must treat t <= processed as a no-op that never moves _processed.
    """
    import numpy as np

    from goldenswing.features.engine import read_and_featurize
    from goldenswing.playbooks.fvg_limit import FVGLimitPlaybook
    from goldenswing.playbooks.fvg_midline import FVGMidlinePlaybook
    from goldenswing.playbooks.trendline_retest import (
        TrendlineRetestPlaybook,
    )

    rng = np.random.default_rng(0)
    n = 40
    close = np.cumsum(rng.standard_normal(n)) + 4000.0
    df = pd.DataFrame(
        {"open": close, "high": close + 2, "low": close - 2,
         "close": close, "volume": 100},
        index=pd.date_range("2026-01-01", periods=n, freq="h"))
    states = read_and_featurize(df, None)
    for cls in (FVGLimitPlaybook, FVGMidlinePlaybook,
                TrendlineRetestPlaybook):
        pb = cls()
        for t in range(n):
            pb.live_signal(states, t, df)
        assert pb._processed == n - 1
        # Rewind: must be a no-op, never move the cursor, never crash.
        assert pb.live_signal(states, n - 3, df) == []
        assert pb._processed == n - 1
        assert pb.live_signal(states, n - 2, df) == []
        assert pb._processed == n - 1


def test_paper_once_rebase_does_not_crash(tmp_path, monkeypatch):
    """Regression: rebase + fresh tail crashed on FVG idx=1998 vs 1999.

    Tick 1 cold-starts 2000 bars (fills the buffer). Tick 2 adds
    5 bars -> overflow -> rebase. The rebase replay must stop at
    the already-processed prefix and leave the 5 fresh bars for
    the _paper_loop_bar pass; replaying them then re-calling them
    rewound the FVG cursor and crashed every tick.
    """
    from goldenswing.backtest.data import load_gold_h1

    full = load_gold_h1().iloc[-2005:].copy()
    feed = _FakeFeed(full, warm_bars=2000)
    store_dir = tmp_path / "paper_state"

    s1 = _once_runner(tmp_path, monkeypatch).run_paper_once(
        feed, state_store_path=store_dir, peers={}, poll_commands=False)
    assert s1["bars"] == 0

    # 5 new bars -> buffer 2005 > 2000 -> rebase path. Must not crash.
    s2 = _once_runner(tmp_path, monkeypatch).run_paper_once(
        feed, state_store_path=store_dir, peers={}, poll_commands=False)
    assert s2["bars"] == 5
    meta = json.loads((store_dir / "meta.json").read_text())
    assert pd.Timestamp(meta["last_bar_time"]) == full.index[-1]

    s3 = _once_runner(tmp_path, monkeypatch).run_paper_once(
        feed, state_store_path=store_dir, peers={}, poll_commands=False)
    assert s3["bars"] == 0
