"""MT5 engine Telegram commands, polling, charts, /harvest.

Covers: polling ON by default (and the OANDA poller off-switch),
MT5-registry isolation from the OANDA registry, /status + /harvest
answering from MT5 journal/state (never OANDA's), tick CSV rows
matching the ticks fed in, chart PNGs on signals with the real
levels, and /harvest ZIP contents + size caps.
"""
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.alerts.commands import (COMMANDS, CommandContext,
                                         CommandRouter)
from goldenswing.charts.engine import render_mt5_signal
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.mt5_commands import (MT5_COMMANDS,
                                                build_mt5_harvest_bundle,
                                                mt5_router)
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
from goldenswing.live.hierarchy import TimeframeStack
from mt5_fakes import FakeMT5

OWNER = "12345"


class FakeExecutor:
    """Minimal executor fake (local — no cross-test-module import)."""

    def __init__(self):
        self._positions = []
        self.orders = []

    def connect(self, **kw):
        return True, "fake connected"

    def positions(self, symbol=None):
        return [p for p in self._positions
                if symbol is None or p.symbol == symbol]

    def symbol_spec(self, symbol):
        from goldenswing.execution.risk import gold_spec
        return gold_spec()

    def place_order(self, request):
        from goldenswing.execution.executor import OrderResult
        self.orders.append(request)
        return OrderResult(True, ticket=9001, price=request.entry_ref,
                           volume=request.volume)

    def modify_sl(self, ticket, sl, tp=None):
        from goldenswing.execution.executor import OrderResult
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        from goldenswing.execution.executor import OrderResult
        return OrderResult(True, ticket=ticket)


def _update(text, chat_id=OWNER, update_id=1):
    return {"update_id": update_id,
            "message": {"message_id": 1,
                        "chat": {"id": chat_id, "type": "private"},
                        "text": text}}


def _mt5_journal(tmp_path):
    j = TradeJournal(tmp_path / "mt5_journal.jsonl")
    j.log_open(alert_id="MT5-fvg_midline-10", instrument="XAUUSD",
               mode="MT5", playbook="fvg_midline", phase="live",
               direction=1, score=70, source="mt5", ticket=1,
               entry=4160.0, stop=4150.0, volume=0.01, risk_money=10.0)
    j.log_close(alert_id="MT5-fvg_midline-10", ticket=1,
                exit_price=4170.0, reason="trail", pnl_money=10.0,
                r_net=1.0, hold_bars=3)
    return j


def _mt5_ctx(journal, tmp_path):
    return CommandContext(
        journal=journal, journal_path=journal.path,
        reports_dir=tmp_path / "reports")


def _runner(tmp_path, *, feed=None, executor=None, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        poll_interval_seconds=0,
        **cfg_kw)
    return MT5Runner(cfg, feed=feed, executor=executor)


class FakeSender:
    def __init__(self):
        self.texts: list[str] = []
        self.photos: list[tuple[bytes, str]] = []
        self.documents: list[tuple[str, str | None]] = []

    def send_text(self, text, silent=False):
        self.texts.append(text)
        return SimpleNamespace(delivered=True, detail="ok")

    def send_document(self, path, caption=None):
        self.documents.append((str(path), caption))
        return SimpleNamespace(delivered=True, detail="ok")

    def send_photo(self, png, caption=None, filename="chart.png"):
        self.photos.append((png, caption or ""))
        return SimpleNamespace(delivered=True, detail="ok")


# ------------------------------------------------------------ polling
def test_polling_on_by_default():
    assert MT5RunnerConfig().poll_commands is True
    assert MT5RunnerConfig.from_env().poll_commands is True


def test_polling_env_off(monkeypatch):
    monkeypatch.setenv("GS_MT5_POLL_COMMANDS", "0")
    assert MT5RunnerConfig.from_env().poll_commands is False


def test_polling_disabled_does_nothing(tmp_path):
    runner = _runner(tmp_path, poll_commands=False)
    runner._tg_api = lambda m, p: (_ for _ in ()).throw(
        AssertionError("must not call the API when disabled"))
    runner._poll_telegram_commands()  # no raise, no API call
    assert runner._tg_synced is False


def test_poller_first_poll_drains_without_answering(tmp_path, monkeypatch):
    monkeypatch.setenv("GS_TELEGRAM_CHAT_ID", OWNER)
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    sender = FakeSender()
    runner = _runner(tmp_path, feed=feed, executor=FakeExecutor())
    runner.senders = [sender]
    calls = []

    def fake_api(method, payload):
        calls.append((method, payload))
        return {"result": [_update("/status", update_id=7)]}

    runner._tg_api = fake_api
    runner._poll_telegram_commands()
    assert runner._tg_synced is True
    assert runner._tg_offset == 8
    assert sender.texts == []  # backlog drained, NOT answered


def test_poller_second_poll_answers_mt5_status(tmp_path, monkeypatch):
    monkeypatch.setenv("GS_TELEGRAM_CHAT_ID", OWNER)
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    sender = FakeSender()
    runner = _runner(tmp_path, feed=feed, executor=FakeExecutor())
    runner.senders = [sender]
    runner._tg_synced = True  # skip the drain phase
    runner._tg_offset = 10

    def fake_api(method, payload):
        if payload.get("offset") == 10:
            return {"result": [_update("/status", update_id=10)]}
        return {"result": []}

    runner._tg_api = fake_api
    runner._poll_telegram_commands()
    assert runner._tg_offset == 11
    assert len(sender.texts) == 1
    assert "MT5 ENGINE STATUS" in sender.texts[0]


def test_oanda_poller_off_switch(monkeypatch):
    """GS_TELEGRAM_POLL=0 silences the OANDA loop's poller (cutover)."""
    import os
    from goldenswing.execution.runner import LiveRunner
    monkeypatch.setenv("GS_TELEGRAM_POLL", "0")
    r = LiveRunner.__new__(LiveRunner)
    assert r._poll_telegram() is None  # returns before any transport


def test_mt5_registry_does_not_clobber_oanda(tmp_path):
    import goldenswing.execution.mt5_commands  # noqa: F401 (registers MT5)
    journal = _mt5_journal(tmp_path)
    ctx = _mt5_ctx(journal, tmp_path)
    oanda = CommandRouter(owner_chat_id=OWNER)
    res, _ = oanda.dispatch(_update("/status"), ctx)
    assert "PAPER LOOP STATUS" in res.text
    mt5 = mt5_router(OWNER)
    res, _ = mt5.dispatch(_update("/status"), ctx)
    assert "MT5 ENGINE STATUS" in res.text
    # The registries are different objects with different handlers.
    assert COMMANDS["/status"] is not MT5_COMMANDS["/status"]


# ------------------------------------------------------------ commands
def test_mt5_status_and_trades_from_mt5_journal(tmp_path):
    journal = _mt5_journal(tmp_path)
    ctx = _mt5_ctx(journal, tmp_path)
    router = mt5_router(OWNER)
    res, _ = router.dispatch(_update("/status"), ctx)
    assert "MT5 ENGINE STATUS" in res.text
    assert "+1.00R" in res.text  # today's closed trade from MT5 journal
    res, _ = router.dispatch(_update("/trades"), ctx)
    assert "MT5 trades" in res.text and "fvg_midline" in res.text


def test_mt5_status_prefers_live_terminal_positions(tmp_path):
    journal = _mt5_journal(tmp_path)
    ctx = _mt5_ctx(journal, tmp_path)
    ctx.live_positions = [{"ticket": 4242, "direction": -1,
                           "entry_price": 4170.0, "sl": 4180.0,
                           "volume": 0.01, "playbook": "fvg_midline"}]
    ctx.last_price = 4160.0
    router = mt5_router(OWNER)
    res, _ = router.dispatch(_update("/status"), ctx)
    assert "#4242" in res.text
    assert "SHORT" in res.text
    assert "+1.00R" in res.text  # (4170-4160)/10 for a short


def test_mt5_ticks_command_exports_terminal_history(tmp_path):
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    journal = _mt5_journal(tmp_path)
    ctx = _mt5_ctx(journal, tmp_path)
    ctx.feed = feed
    router = mt5_router(OWNER)
    res, _ = router.dispatch(_update("/ticks"), ctx)
    assert res.document is not None
    rows = Path(res.document).read_text().strip().splitlines()
    assert rows[0] == "time,bid,ask,tick_volume,real_volume,spread"
    assert len(rows) == 4  # header + FakeMT5's 3 history rows


def test_mt5_ticks_without_feed(tmp_path):
    journal = _mt5_journal(tmp_path)
    ctx = _mt5_ctx(journal, tmp_path)
    router = mt5_router(OWNER)
    res, _ = router.dispatch(_update("/ticks"), ctx)
    assert "needs the live MT5 feed" in res.text


# ------------------------------------------------------------ harvest
def test_mt5_harvest_bundles_journal_state_ticks_results(tmp_path):
    journal = _mt5_journal(tmp_path)
    (tmp_path / "mt5_risk_state.json").write_text('{"risk": "ok"}')
    (tmp_path / "mt5_ticks_20261014.csv").write_text(
        "time,bid,ask,tick_volume,real_volume,spread\n"
        "2026-10-14T10:00:00+00:00,4160.10,4160.40,3,0,0.30\n")
    ctx = _mt5_ctx(journal, tmp_path)
    router = mt5_router(OWNER)
    res, _ = router.dispatch(_update("/harvest"), ctx)
    zpath = Path(res.document)
    assert zpath.exists() and zpath.suffix == ".zip"
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
        assert "RESULTS.md" in names and "manifest.json" in names
        assert "mt5_journal.jsonl" in names
        assert "mt5_risk_state.json" in names
        assert "mt5_ticks_20261014.csv" in names
        results = zf.read("RESULTS.md").decode()
        assert "MT5" in results and "fvg_midline" in results
        assert "+1.00R" in results
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["engine"] == "mt5" and manifest["scope"] == "all"
        assert all(f["sha256"] for f in manifest["files"])


def test_mt5_harvest_scopes(tmp_path):
    journal = _mt5_journal(tmp_path)
    (tmp_path / "mt5_risk_state.json").write_text('{"risk": "ok"}')
    (tmp_path / "mt5_ticks_20261014.csv").write_text(
        "time,bid,ask,tick_volume,real_volume,spread\n")
    ctx = _mt5_ctx(journal, tmp_path)
    router = mt5_router(OWNER)
    res, _ = router.dispatch(_update("/harvest ticks"), ctx)
    with zipfile.ZipFile(res.document) as zf:
        names = set(zf.namelist())
        assert "mt5_ticks_20261014.csv" in names
        assert "mt5_journal.jsonl" not in names
    res, _ = router.dispatch(_update("/harvest state"), ctx)
    with zipfile.ZipFile(res.document) as zf:
        names = set(zf.namelist())
        assert "mt5_risk_state.json" in names
        assert "mt5_journal.jsonl" not in names
    res, _ = router.dispatch(_update("/harvest bogus"), ctx)
    assert "usage" in res.text


def test_mt5_harvest_caps_skip_oversize_file(tmp_path, monkeypatch):
    import goldenswing.alerts.commands as C
    monkeypatch.setattr(C, "HARVEST_MAX_FILE_BYTES", 100)
    journal = _mt5_journal(tmp_path)
    (tmp_path / "mt5_ticks_20261014.csv").write_text("x" * 200)
    ctx = _mt5_ctx(journal, tmp_path)
    zip_path, manifest = build_mt5_harvest_bundle(ctx, "all")
    assert any("20MB cap" in s["reason"] or "100" in s["reason"]
               for s in manifest["skipped"])
    with zipfile.ZipFile(zip_path) as zf:
        assert "mt5_ticks_20261014.csv" not in zf.namelist()


def _raw_journal(tmp_path, n_notes=1500):
    """A journal file written as raw JSONL: one boot manifest, then
    many note records (the SL-move notes are the payload that must
    never be lost), then a close."""
    import json as _json
    path = tmp_path / "mt5_journal.jsonl"
    lines = [_json.dumps({
        "ts": "2026-10-20T00:00:00+00:00", "type": "manifest",
        "commit": "abc1234", "dirty_tree": False,
        "config_fingerprint": "fp123"})]
    for i in range(n_notes):
        lines.append(_json.dumps({
            "ts": f"2026-10-20T01:{i % 60:02d}:00+00:00", "type": "note",
            "text": f"mt5 tick: ticket 777 SL -> 400{i % 10}.00 "
                    f"(profit_lock) seq {i}"}))
    lines.append(_json.dumps({
        "ts": "2026-10-20T02:00:00+00:00", "type": "close",
        "ticket": 777, "r_net": 0.8}))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_mt5_harvest_oversize_journal_rides_as_slice(tmp_path, monkeypatch):
    # Step 12a: a journal over the per-file cap must still ride the
    # harvest — as a newest-records slice — and the bundle must say
    # which build wrote it.
    import goldenswing.alerts.commands as C
    monkeypatch.setattr(C, "HARVEST_MAX_FILE_BYTES", 200_000)
    path = _raw_journal(tmp_path, n_notes=3000)
    assert path.stat().st_size > 200_000
    ctx = CommandContext(journal=None, journal_path=path,
                         reports_dir=tmp_path / "reports")
    zip_path, manifest = build_mt5_harvest_bundle(ctx, "all")
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "mt5_journal.jsonl" not in names
        assert "mt5_journal_slice.jsonl" in names
        raw = zf.read("mt5_journal_slice.jsonl").decode()
        results = zf.read("RESULTS.md").decode()
    slice_lines = [ln for ln in raw.splitlines() if ln.strip()]
    assert slice_lines
    import json as _json
    for ln in slice_lines:
        _json.loads(ln)  # every slice line is a complete record
    assert _json.loads(slice_lines[-1])["type"] == "close"
    assert manifest["journal_slice"]["source_bytes"] == \
        path.stat().st_size
    assert manifest["journal_slice"]["records"] == len(slice_lines)
    assert manifest["engine_build"]["commit"] == "abc1234"
    assert "engine_build: abc1234" in results
    assert "journal_slice: mt5_journal_slice.jsonl" in results
    assert any(s["arcname"] == "mt5_journal.jsonl"
               for s in manifest["skipped"])


def test_journal_slice_helpers(tmp_path):
    from goldenswing.alerts.commands import (last_journal_manifest,
                                             write_journal_slice)
    path = _raw_journal(tmp_path, n_notes=50)
    # Generous budget: the whole file rides, no prepend needed.
    out = tmp_path / "slice_all.jsonl"
    prov = write_journal_slice(path, out, 10 * 1024 * 1024)
    assert prov is not None and prov["manifest_prepended"] is False
    assert prov["records"] == 52
    assert last_journal_manifest(path)["commit"] == "abc1234"
    # Tight budget: only a tail fits; the manifest is prepended so
    # the slice still names its build.
    out2 = tmp_path / "slice_tail.jsonl"
    prov2 = write_journal_slice(path, out2, 400)
    assert prov2 is not None and prov2["manifest_prepended"] is True
    import json as _json
    first = _json.loads(out2.read_text().splitlines()[0])
    assert first["type"] == "manifest"
    last = _json.loads(out2.read_text().splitlines()[-1])
    assert last["type"] == "close"


# ------------------------------------------------------------ tick CSV
def test_tick_csv_rows_match_ticks_fed(tmp_path):
    script = [(2000.0, 2000.30, 0), (2000.10, 2000.40, 1)]
    feed = MT5Feed(MT5FeedConfig(tick_interval=0.001),
                   mt5_module=FakeMT5(tick_script=script))
    runner = _runner(tmp_path, feed=feed, executor=FakeExecutor())
    stats = runner.run(max_ticks=2)
    assert stats["ticks"] == 2
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    csv_path = tmp_path / f"mt5_ticks_{day}.csv"
    assert csv_path.exists()
    rows = csv_path.read_text().strip().splitlines()
    assert rows[0] == "time,bid,ask,tick_volume,real_volume,spread"
    assert len(rows) == 3  # header + exactly the 2 ticks pumped
    # connect()'s UTC-offset calibration consumes script entry 0, so
    # the pump delivers entry 1, then the last price repeated with
    # advancing time (pump dedup keys on time+bid+ask).
    assert rows[1].split(",")[1:3] == ["2000.10000", "2000.40000"]
    assert rows[2].split(",")[1:3] == ["2000.10000", "2000.40000"]
    assert rows[1].split(",")[0] != rows[2].split(",")[0]


# ------------------------------------------------------------ charts
def _h1_frame(n=150, base=4160.0):
    idx = pd.date_range("2026-10-14", periods=n, freq="h", tz="UTC")
    close = [base + (i % 9) * 0.5 for i in range(n)]
    return pd.DataFrame(
        {"open": close, "high": [c + 0.4 for c in close],
         "low": [c - 0.4 for c in close], "close": close,
         "volume": [100] * n, "real_volume": [0] * n,
         "spread": [30] * n},
        index=idx)


def test_chart_png_carries_real_levels():
    h1 = _h1_frame()
    zone = {"kind": "support", "level": 4150.0, "top": 4152.0,
            "bottom": 4148.0, "index": 100}
    verdict = {"bias_d1": 1, "bias_h4": 1, "trigger": "BOS",
               "entry_clock": "M15", "amd_phase": "distribution",
               "qm": {"status": "qm_absent"}, "volume": {"volume_ratio": 1.2}}
    png_a = render_mt5_signal(
        h1, zone=zone, entry=4155.0, stop=4145.0, targets=[4175.0],
        direction=1, verdict=verdict, playbook="fvg_midline", score=72)
    assert png_a[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png_a) > 10_000  # a real chart, not an empty canvas
    # Levels flow into the render: a different entry changes the bytes.
    png_b = render_mt5_signal(
        h1, zone=zone, entry=4165.0, stop=4145.0, targets=[4175.0],
        direction=1, verdict=verdict, playbook="fvg_midline", score=72)
    assert png_a != png_b
    # No zone: still renders (band skipped, never faked).
    png_c = render_mt5_signal(
        h1, zone=None, entry=4155.0, stop=4145.0, targets=[4175.0],
        direction=-1, verdict=None, playbook="fvg_midline", score=72)
    assert png_c[:8] == b"\x89PNG\r\n\x1a\n"
    # Fill marker renders.
    png_d = render_mt5_signal(
        h1, zone=zone, entry=4155.0, stop=4145.0, targets=[4175.0],
        direction=1, verdict=verdict, playbook="fvg_midline", score=72,
        fill_price=4155.5)
    assert png_d[:8] == b"\x89PNG\r\n\x1a\n" and png_d != png_a


def test_runner_build_chart_saves_png(tmp_path):
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    feed.connect()
    runner = _runner(tmp_path, feed=feed, executor=FakeExecutor())
    runner._stack = TimeframeStack.from_frames(feed.warm_start_all())
    sig = SimpleNamespace(direction=1, score=80.0, entry_ref=2000.0,
                          invalidation=1990.0, targets=[2030.0], atr=10.0,
                          index=100)
    verdict = {"bias_d1": 1, "bias_h4": 1, "trigger": "BOS",
               "entry_clock": "M15", "amd_phase": "distribution",
               "qm": {"status": "qm_absent"},
               "volume": {"volume_ratio": 1.0},
               "zone": {"kind": "support", "level": 1995.0,
                        "top": 1997.0, "bottom": 1993.0, "index": 50}}
    png, path = runner._build_chart(sig, verdict, alert_id="MT5-test-100",
                                    playbook="fvg_midline")
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"
    assert path is not None and path.exists()
    assert path.name == "MT5-test-100.png"


def test_runner_alert_sends_photo_when_chart_present(tmp_path):
    runner = _runner(tmp_path, executor=FakeExecutor())
    sender = FakeSender()
    runner.senders = [sender]
    runner._send_alert("hello", chart_png=b"FAKEPNG")
    assert len(sender.photos) == 1
    assert sender.photos[0][1] == "hello"
    runner._send_alert("text only")
    assert sender.texts == ["text only"]


# ------------------------------------------------------------ /d1 export
def _d1_ctx(tmp_path, feed):
    return CommandContext(reports_dir=tmp_path / "reports", feed=feed)


def _d1_fn():
    return MT5_COMMANDS["/d1"][1]


def test_d1_registered_in_mt5_commands():
    assert "/d1" in MT5_COMMANDS
    assert MT5_COMMANDS["/d1"][0]  # description shown in help


def test_d1_export_writes_csv_with_d1_bars(tmp_path):
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    ctx = _d1_ctx(tmp_path, feed)
    res = _d1_fn()(ctx, ["10"])
    assert res.document is not None
    path = Path(res.document)
    assert path.exists()
    assert "10bars" in path.name
    df = pd.read_csv(path)
    assert list(df.columns) == ["time", "open", "high", "low", "close",
                                "volume", "real_volume", "spread"]
    assert 1 <= len(df) <= 10
    # Daily spacing: consecutive bars exactly one day apart.
    ts = pd.to_datetime(df["time"], utc=True)
    assert (ts.diff().dropna() == pd.Timedelta(days=1)).all()
    assert "closed bars" in (res.caption or "")
    assert "no resampling" in (res.caption or "")


def test_d1_export_default_is_multiyear(tmp_path):
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    ctx = _d1_ctx(tmp_path, feed)
    res = _d1_fn()(ctx, [])
    assert "2000bars" in Path(res.document).name
    assert len(pd.read_csv(res.document)) > 1000  # default covers years


def test_d1_export_caps_and_rejects_bad_input(tmp_path):
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    ctx = _d1_ctx(tmp_path, feed)
    fn = _d1_fn()
    res = fn(ctx, ["99999"])
    assert "5000bars" in Path(res.document).name
    assert len(pd.read_csv(res.document)) <= 5000
    for bad in (["abc"], ["0"], ["-5"], ["1.5"]):
        res = fn(ctx, bad)
        assert "usage" in (res.text or "")
        assert res.document is None


def test_d1_export_no_feed_is_clean_error(tmp_path):
    ctx = _d1_ctx(tmp_path, feed=None)
    res = _d1_fn()(ctx, ["10"])
    assert res.document is None
    assert "live MT5 feed" in (res.text or "")


def test_d1_export_feed_failure_is_clean_error(tmp_path):
    from goldenswing.feed.mt5 import MT5Unavailable

    class BoomFeed:
        def bars(self, tf, count):
            raise MT5Unavailable("terminal not connected")

    ctx = _d1_ctx(tmp_path, feed=BoomFeed())
    res = _d1_fn()(ctx, ["10"])
    assert res.document is None
    assert "D1 export failed" in (res.text or "")
    assert "MT5Unavailable" in (res.text or "")
