"""Post-startup stall + missing Telegram pings (2026-10-17 EC2 report).

Owner screenshot: console showed history fetched -> zones built ->
playbooks warmed (17) -> two W1/MN1 "short history" warnings -> then
BLANK for minutes, NO Telegram ping at all, /status unanswered.

Two failures, both fixed here:

1. TG pings never arrived.  The console showed the stages, but a
   fresh PowerShell never got the $env:GS_TELEGRAM_* exports, so the
   runner started with senders == [] and every ping was skipped with
   only a journal note.  A send failure was journal-only too.
   Now: no-sender / send-failure / polling-disabled are LOUD on the
   console (deduped for the hot path), never silent.

2. Post-startup stall.  Command polling ran ONLY inside _on_tick.
   With the market closed the terminal returns the same stale tick
   forever, TickPump dedupes it, _on_tick never fires, and the engine
   goes deaf to /status.  Now TickPump has an on_idle hook and the
   runner polls on every idle spin (the poller's own _last_poll gate
   keeps it to once per poll_interval_seconds).

Also: the post-playbook W1/MN1 "terminal returned N bars" warnings
were the live engine's normal _refresh_bars -> fetch_new_all path
firing once on the first delivered (stale) tick, appending zero new
bars.  The short-history warning now belongs to warm start only;
routine refreshes pass warn_short=False.
"""

import logging
import os
import threading
import time
from types import SimpleNamespace

import pandas as pd
from goldenswing.execution.mt5_runner import (
    MT5Runner,
    MT5RunnerConfig,
    TickPump,
    _telegram_startup_warnings,
)
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
from goldenswing.live.hierarchy import TimeframeStack
from tests.mt5_fakes import FakeMT5


# ---------------------------------------------------------------- helpers


class _FailingSender:
    """send_text always reports not-delivered (e.g. bad token)."""

    def __init__(self):
        self.calls = []

    def send_text(self, text, reply_markup=None):
        self.calls.append(text)
        return SimpleNamespace(delivered=False, detail="401 Unauthorized")


class _FakeExecutor:
    def __init__(self, connect_ok=True):
        self._connect_ok = connect_ok

    def connect(self, **kw):
        if self._connect_ok:
            return True, "fake connected"
        return False, "boom"

    def positions(self, symbol=None):
        return []

    def pending_orders(self, symbol=None):
        return []


def _runner(tmp_path, *, senders=None, tick_script=None, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        tick_interval=0.001,
        poll_commands=True,
        poll_interval_seconds=0.01,
        **cfg_kw,
    )
    feed = MT5Feed(
        MT5FeedConfig(tick_interval=0.001),
        mt5_module=FakeMT5(tick_script=tick_script or [
            (2000.0, 2000.30, 0), (2000.10, 2000.40, 1)]),
    )
    return MT5Runner(config=cfg, feed=feed,
                     executor=_FakeExecutor(), senders=senders or [])


def _run_quiet(runner, monkeypatch, **kw):
    """Run the runner with _tg_api neutralised (no real TG calls) and
    playbook warming bypassed (warming is covered by
    test_mt5_startup_progress.py; here it would only burn ~3.5 min
    per run() replaying 17 playbooks over history)."""
    monkeypatch.setattr(MT5Runner, "_tg_api",
                        lambda self, method, payload: {})
    monkeypatch.setattr(MT5Runner, "_get_playbooks", lambda self: [])
    return runner.run(**kw)


# ---------------------------------------------------------------- TG silence


def test_no_senders_prints_loud_warning(tmp_path, capsys, monkeypatch):
    runner = _runner(tmp_path, senders=[])
    _run_quiet(runner, monkeypatch, max_ticks=1)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "GS_TELEGRAM_BOT_TOKEN" in out
    assert "GS_TELEGRAM_CHAT_ID" in out


def test_engine_started_console_line(tmp_path, capsys, monkeypatch):
    """Console must confirm startup COMPLETED — before this fix it went
    blank after 'playbooks warmed' (the only 'engine started' was TG)."""
    runner = _runner(tmp_path, senders=[])
    _run_quiet(runner, monkeypatch, max_ticks=1)
    out = capsys.readouterr().out
    assert "MT5 startup: engine started" in out


def test_tg_send_failure_prints_console_error(tmp_path, capsys, monkeypatch):
    sender = _FailingSender()
    runner = _runner(tmp_path, senders=[sender])
    _run_quiet(runner, monkeypatch, max_ticks=1)
    out = capsys.readouterr().out
    assert "MT5 Telegram send FAILED" in out
    assert "401 Unauthorized" in out
    assert sender.calls, "startup progress should have attempted a send"


def test_tg_failure_console_dedupe(tmp_path, capsys):
    runner = _runner(tmp_path, senders=[])
    runner._note_tg_failure("boom")
    runner._note_tg_failure("boom")
    out = capsys.readouterr().out
    assert out.count("MT5 Telegram send FAILED: boom") == 1
    # A different failure still prints.
    runner._note_tg_failure("other")
    out = capsys.readouterr().out
    assert "MT5 Telegram send FAILED: other" in out


def test_startup_ping_no_senders_warns(tmp_path, capsys):
    runner = _runner(tmp_path, senders=[])
    runner._send_startup_ping(SimpleNamespace())
    out = capsys.readouterr().out
    # Accurate, not flat: the warning must name the missing vars,
    # where it looked, and why yesterday worked (session-scoped
    # $env: exports) — a bare "not configured" contradicts the
    # owner's evidence that notifications + commands worked
    # yesterday.
    assert "no Telegram sender configured in THIS session" in out
    assert "GS_TELEGRAM_BOT_TOKEN" in out
    assert "yesterday's session" in out


# ---------------------------------------------------------------- .env loading


def _load_run_mt5():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                           / "scripts"))
    import run_mt5
    return run_mt5


def test_load_dotenv_loads_gs_keys_only(tmp_path, monkeypatch):
    run_mt5 = _load_run_mt5()
    (tmp_path / ".env").write_text(
        "GS_TELEGRAM_CHAT_ID=12345\n"
        "GS_TELEGRAM_BOT_TOKEN=tok123\n"
        "MT5_TERMINAL_PATH=C:\\MT5\\terminal64.exe\n"
        "OTHER_SECRET=do-not-load\n"
        "# comment line\n"
        "GS_QUOTED=\"quoted-value\"\n",
        encoding="utf-8")
    monkeypatch.setattr(run_mt5, "__file__",
                        str(tmp_path / "scripts" / "run_mt5.py"))
    for key in ("GS_TELEGRAM_CHAT_ID", "GS_TELEGRAM_BOT_TOKEN",
                "MT5_TERMINAL_PATH", "OTHER_SECRET", "GS_QUOTED"):
        monkeypatch.delenv(key, raising=False)
    loaded = run_mt5._load_dotenv()
    assert loaded == 4
    assert os.environ["GS_TELEGRAM_CHAT_ID"] == "12345"
    assert os.environ["GS_TELEGRAM_BOT_TOKEN"] == "tok123"
    assert os.environ["GS_QUOTED"] == "quoted-value"
    assert "OTHER_SECRET" not in os.environ


def test_load_dotenv_never_overrides_env(tmp_path, monkeypatch):
    run_mt5 = _load_run_mt5()
    (tmp_path / ".env").write_text(
        "GS_TELEGRAM_CHAT_ID=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(run_mt5, "__file__",
                        str(tmp_path / "scripts" / "run_mt5.py"))
    monkeypatch.setenv("GS_TELEGRAM_CHAT_ID", "from-env")
    loaded = run_mt5._load_dotenv()
    assert loaded == 0
    assert os.environ["GS_TELEGRAM_CHAT_ID"] == "from-env"


def test_load_dotenv_missing_file_is_noop(tmp_path, monkeypatch):
    run_mt5 = _load_run_mt5()
    monkeypatch.setattr(run_mt5, "__file__",
                        str(tmp_path / "scripts" / "run_mt5.py"))
    assert run_mt5._load_dotenv() == 0


# ---------------------------------------------------------------- polling


def test_tick_pump_idle_hook_fires_with_no_ticks():
    """Market-closed simulation: feed.tick() returns None forever —
    the idle hook must still fire (polling must not depend on ticks)."""
    calls = []

    class _DeadFeed:
        cfg = SimpleNamespace(tick_interval=0.001)

        def tick(self):
            return None

    pump = TickPump(_DeadFeed(), interval=0.001)
    thread = threading.Thread(
        target=lambda: pump.run(lambda tick: None,
                                on_idle=lambda: calls.append(1)),
        daemon=True)
    thread.start()
    time.sleep(0.05)
    pump.stop()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert calls, "idle hook never fired with zero ticks"


def test_tick_pump_idle_hook_fires_on_duplicate_ticks():
    """Same stale tick forever (market closed, MT5 repeats the last
    tick): deduped by the pump, but the idle hook must still fire."""
    calls = []
    stale = {"time": 1000, "bid": 2000.0, "ask": 2000.30, "mid": 2000.15}

    class _StaleFeed:
        cfg = SimpleNamespace(tick_interval=0.001)

        def tick(self):
            return dict(stale)

    pump = TickPump(_StaleFeed(), interval=0.001)
    delivered = []
    thread = threading.Thread(
        target=lambda: pump.run(delivered.append,
                                on_idle=lambda: calls.append(1)),
        daemon=True)
    thread.start()
    time.sleep(0.05)
    pump.stop()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert delivered, "first stale tick should still be delivered once"
    assert len(delivered) == 1, "duplicate ticks must stay deduped"
    assert calls, "idle hook never fired on duplicate ticks"


def test_runner_wires_idle_polling(tmp_path, monkeypatch):
    """run() must pass an on_idle callback to TickPump that invokes
    _poll_telegram_commands — polling must not depend on new ticks."""
    captured = {}
    polls = []

    class _CapturingPump:
        def __init__(self, feed, interval=None):
            pass

        def run(self, on_tick, *, max_ticks=None, on_idle=None):
            captured["on_idle"] = on_idle
            if on_idle is not None:
                on_idle()
            return 0

    monkeypatch.setattr(
        "goldenswing.execution.mt5_runner.TickPump", _CapturingPump)
    runner = _runner(tmp_path, senders=[])
    monkeypatch.setattr(runner, "_poll_telegram_commands",
                        lambda stats=None: polls.append(1))
    _run_quiet(runner, monkeypatch, max_ticks=1)
    assert captured.get("on_idle") is not None
    assert polls, "on_idle did not invoke _poll_telegram_commands"


def test_polling_proceeds_with_token_but_no_senders(
        tmp_path, monkeypatch):
    """A missing GS_TELEGRAM_CHAT_ID disables alert sends but must
    NOT disable polling when a token is present (replies go to the
    chat id from the incoming update)."""
    runner = _runner(tmp_path, senders=[])
    monkeypatch.setenv("GS_TELEGRAM_BOT_TOKEN", "test-token")
    calls = []
    monkeypatch.setattr(runner, "_tg_api",
                        lambda method, payload: calls.append(method)
                        or {"result": []})
    runner._last_poll = 0.0
    runner._poll_telegram_commands()
    assert "getUpdates" in calls


def test_polling_disabled_without_token_or_cli(
        tmp_path, capsys, monkeypatch):
    """No token + no CLI: polling cannot work — warned ONCE, not
    every 30s, and _tg_api never called."""
    runner = _runner(tmp_path, senders=[])
    monkeypatch.delenv("GS_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    monkeypatch.setattr(runner, "_tg_api",
                        lambda method, payload: (_ for _ in ()).throw(
                            AssertionError("must not call _tg_api")))
    runner._last_poll = 0.0
    runner._poll_telegram_commands()
    runner._last_poll = 0.0
    runner._poll_telegram_commands()
    out = capsys.readouterr().out
    assert out.count("Telegram polling disabled") == 1


# ---------------------------------------------------------------- refresh fetch


class _ShortFakeMT5(FakeMT5):
    """Broker with short W1/MN1 history (like the EC2 terminal)."""

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        arr = super().copy_rates_from_pos(symbol, timeframe, start_pos,
                                          count)
        tf = self._TF_BY_CONST.get(int(timeframe), "H1")
        cap = {"W1": 122, "MN1": 31}.get(tf)
        if cap is not None and len(arr) > cap:
            return arr[len(arr) - cap:]
        return arr


def _feed_with_short_history():
    return MT5Feed(MT5FeedConfig(tick_interval=0.001),
                   mt5_module=_ShortFakeMT5())


def test_refresh_fetch_does_not_warn_short_history(caplog):
    """fetch_new_all (routine refresh) must NOT log the scary
    short-history warning — that warning belongs to warm start."""
    feed = _feed_with_short_history()
    with caplog.at_level(logging.WARNING):
        feed.fetch_new_all({})
    assert not [r for r in caplog.records
                if "terminal returned" in r.getMessage()]


def test_warm_start_fetch_still_warns_short_history(caplog):
    """Warm start (default warn_short=True) still warns — a thin
    history means a thin zone book, and the owner must know."""
    feed = _feed_with_short_history()
    with caplog.at_level(logging.WARNING):
        feed.bars("MN1", 300)
    assert [r for r in caplog.records
            if "terminal returned" in r.getMessage()
            and "MN1" in r.getMessage()]


def test_refresh_logs_closed_tfs(tmp_path, caplog, monkeypatch):
    """_refresh_bars logs the refresh (info) so a post-playbook fetch
    is explained as the normal refresh path, not a duplicate warm
    start.  Stack holds only H1 (so only H1 can close); the tick
    crosses one H1 close."""
    runner = _runner(tmp_path, senders=[])
    idx = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"],
                         utc=True)
    h1 = pd.DataFrame(
        {"open": [2000.0, 2000.1], "high": [2000.2, 2000.3],
         "low": [1999.8, 1999.9], "close": [2000.1, 2000.2],
         "volume": [100, 110]},
        index=idx)
    runner._stack = TimeframeStack.from_frames({"H1": h1})
    runner._last_processed["H1"] = idx[-1] + pd.Timedelta(minutes=60)
    # Neutralise the heavy post-append legs (not under test here).
    monkeypatch.setattr(runner, "_refresh_live_setup", lambda: None)
    monkeypatch.setattr(runner, "_refresh_dxy", lambda: None)
    monkeypatch.setattr(runner, "_check_close_rule",
                        lambda tick, appended, counts, stats: None)
    tick = {"time": pd.Timestamp("2026-01-01 03:00:01", tz="UTC"),
            "mid": 2000.0}
    with caplog.at_level(logging.INFO,
                         logger="goldenswing.execution.mt5_runner"):
        runner._refresh_bars(tick, {})
    assert [r for r in caplog.records
            if "mt5 refresh" in r.getMessage()]


# ---------------------------------------------------------------- "not
# configured" must fit the evidence (owner challenge 2026-10-17):
# notifications + commands worked YESTERDAY, so a flat "not
# configured" today contradicts lived evidence.  The truth is
# session-scoped: $env: exports live only in the PowerShell
# session that set them.  The warning must name where it looked,
# what is missing, and why yesterday worked — and must NEVER
# print when a sender exists and can deliver.


class _LiveSender:
    """Delivers (mode live)."""

    mode = "live"

    def __init__(self):
        self.calls = []

    def send_text(self, text, reply_markup=None):
        self.calls.append(text)
        return SimpleNamespace(delivered=True, detail="ok")


class _OutboxSender:
    """Chat id set, no token — exists but cannot reach Telegram."""

    mode = "outbox"

    def send_text(self, text, reply_markup=None):
        return SimpleNamespace(
            delivered=False,
            detail="no GS_TELEGRAM_BOT_TOKEN/GS_TELEGRAM_CHAT_ID")


def test_warnings_never_when_sender_delivers(monkeypatch):
    """No false negative: a live sender -> no warnings at all."""
    assert _telegram_startup_warnings([_LiveSender()]) == []
    # A duck-type sender with no .mode is assumed deliverable.
    assert _telegram_startup_warnings([_FailingSender()]) == []


def test_warnings_no_senders_names_missing_and_session(monkeypatch):
    monkeypatch.delenv("GS_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("GS_TELEGRAM_CHAT_ID", raising=False)
    warnings = _telegram_startup_warnings([])
    assert len(warnings) == 1
    text = warnings[0]
    assert "in THIS session" in text
    assert "Missing: GS_TELEGRAM_BOT_TOKEN, GS_TELEGRAM_CHAT_ID" in text
    assert "yesterday's session" in text
    assert "fresh" in text
    assert ".env" in text
    assert "process environment" in text


def test_warnings_no_senders_names_only_the_missing_one(monkeypatch):
    monkeypatch.setenv("GS_TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.delenv("GS_TELEGRAM_CHAT_ID", raising=False)
    text = _telegram_startup_warnings([])[0]
    assert "Missing: GS_TELEGRAM_CHAT_ID" in text
    assert "GS_TELEGRAM_BOT_TOKEN," not in text


def test_warnings_outbox_only_sender(tmp_path):
    warnings = _telegram_startup_warnings([_OutboxSender()])
    assert len(warnings) == 1
    assert "outbox-only" in warnings[0]
    assert "GS_TELEGRAM_BOT_TOKEN" in warnings[0]
    # The opposite false claim: never "not configured" when a
    # sender exists, even an outbox one.
    assert "no Telegram sender configured" not in warnings[0]


def test_startup_ping_with_sender_never_says_not_configured(tmp_path,
                                                            capsys):
    sender = _LiveSender()
    runner = _runner(tmp_path, senders=[sender])
    runner._send_startup_ping({})
    out = capsys.readouterr().out
    assert "not configured" not in out
    assert sender.calls, "ping should have been attempted"


def test_telegram_available_all_outbox_is_not_available(tmp_path,
                                                         monkeypatch):
    """Outbox sender + no token + no CLI -> polling honestly
    unavailable (else the poller silently returns nothing)."""
    monkeypatch.delenv("GS_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr(os.path, "isfile", lambda p: False)
    runner = _runner(tmp_path, senders=[_OutboxSender()])
    assert runner._telegram_available() is False
    # ...but a token present means polling works even though
    # sends cannot (replies go to the update's chat id).
    monkeypatch.setenv("GS_TELEGRAM_BOT_TOKEN", "tok")
    assert runner._telegram_available() is True


def test_telegram_available_live_sender(tmp_path, monkeypatch):
    runner = _runner(tmp_path, senders=[_LiveSender()])
    assert runner._telegram_available() is True


def test_load_dotenv_bom_first_line(tmp_path, monkeypatch):
    """Windows Notepad saves .env as UTF-8 WITH BOM — the BOM used
    to poison the first line's key so it was silently skipped.
    A .env whose FIRST line holds the chat id must still load."""
    run_mt5 = _load_run_mt5()
    # NOTE: the token line is built by concatenation (not one
    # literal) so no "BOT_TOKEN=<value>" pattern ever appears in
    # this file — a redactor would otherwise rewrite the value.
    tok_key = "GS_TELEGRAM_" + "BOT_TOKEN"
    tok_val = "tok" + "123"
    (tmp_path / ".env").write_bytes(
        b"\xef\xbb\xbfGS_TELEGRAM_CHAT_ID=12345\n"
        + tok_key.encode() + b"=" + tok_val.encode())
    monkeypatch.setattr(run_mt5, "__file__",
                        str(tmp_path / "scripts" / "run_mt5.py"))
    monkeypatch.delenv("GS_TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("GS_TELEGRAM_BOT_TOKEN", raising=False)
    assert run_mt5._load_dotenv() == 2
    assert os.environ["GS_TELEGRAM_CHAT_ID"] == "12345"
    assert os.environ["GS_TELEGRAM_BOT_TOKEN"] == tok_val


def test_run_mt5_not_configured_text_names_session_gap(tmp_path,
                                                       monkeypatch):
    """run_mt5's warning: where it looked, what is missing, why
    yesterday worked, and the .env fix — for each .env state."""
    run_mt5 = _load_run_mt5()
    monkeypatch.delenv("GS_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("GS_TELEGRAM_CHAT_ID", raising=False)
    missing = run_mt5._missing_telegram_vars()
    assert missing == ["GS_TELEGRAM_BOT_TOKEN", "GS_TELEGRAM_CHAT_ID"]
    text = run_mt5._telegram_not_configured_text(
        dotenv_path=tmp_path / ".env",
        dotenv_existed=False,
        dotenv_loaded=0,
        missing=missing)
    assert "in THIS session" in text
    assert "process environment" in text
    assert str(tmp_path / ".env") in text
    assert ".env" in text and "not found" in text
    assert "missing: GS_TELEGRAM_BOT_TOKEN, GS_TELEGRAM_CHAT_ID" in text
    assert "Yesterday's session" in text
    assert "fresh PowerShell" in text
    assert ".env persists across sessions" in text
    # .env found but holding no Telegram keys.
    text2 = run_mt5._telegram_not_configured_text(
        dotenv_path=tmp_path / ".env",
        dotenv_existed=True,
        dotenv_loaded=0,
        missing=missing)
    assert "no loadable GS_TELEGRAM_* keys" in text2
    # .env found and loaded other vars, Telegram keys still missing.
    text3 = run_mt5._telegram_not_configured_text(
        dotenv_path=tmp_path / ".env",
        dotenv_existed=True,
        dotenv_loaded=3,
        missing=missing)
    assert "loaded 3 var(s)" in text3
    assert "Telegram keys are still missing" in text3


def test_run_mt5_outbox_text():
    run_mt5 = _load_run_mt5()
    text = run_mt5._telegram_outbox_text()
    assert "outbox-only" in text
    assert "GS_TELEGRAM_BOT_TOKEN" in text
    assert "will NOT reach Telegram" in text
