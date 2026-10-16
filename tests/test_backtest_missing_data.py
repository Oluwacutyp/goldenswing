"""Tests for the missing-backtest-data failure path.

Regression: on EC2 ``/backtest`` failed with a pandas traceback ending
in ``FileNotFoundError: .../OANDA_XAU_USD_H1_20241001_20261012.csv``
because ``data/*.csv`` is gitignored and the CSVs never shipped.
The script must now fail gracefully with a clear message, and the
Telegram path must surface that message — never a traceback.
"""
import importlib.util
from pathlib import Path

import pytest

from goldenswing.alerts.commands import (
    BacktestJob, CommandContext, _finish_backtest_job,
    _run_backtest_subprocess, reset_backtest_jobs_for_tests,
    set_backtest_runner_for_tests)
from goldenswing.alerts import commands as alert_cmds

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backtest_full_engine.py"
MISSING_LINE = ("Backtest data missing: "
                "OANDA_XAU_USD_H1_20241001_20261012.csv. On EC2: "
                "git pull origin main (CSVs ship with the repo), "
                "or run /harvest. Do not run backtest without feed.")


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "backtest_full_engine_missing_data", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_missing_backtest_data_lists_absent_files(tmp_path):
    mod = _load_script()
    missing = mod.missing_backtest_data(tmp_path)
    assert missing == list(mod.BACKTEST_CSVS)
    for name in mod.BACKTEST_CSVS:
        (tmp_path / name).write_text("x")
    assert mod.missing_backtest_data(tmp_path) == []


def test_main_missing_data_returns_3_with_clear_message(
        tmp_path, capsys, monkeypatch):
    mod = _load_script()
    empty = tmp_path / "empty-data"
    empty.mkdir()
    monkeypatch.setattr(mod, "DATA_DIR", empty)
    assert mod.main(["--out", str(tmp_path / "out")]) == \
        mod.BACKTEST_MISSING_DATA_EXIT
    out, err = capsys.readouterr()
    assert "Backtest data missing:" in err
    assert "git pull origin main" in err
    assert "Traceback" not in err
    assert "FileNotFoundError" not in err


def test_main_missing_data_notify_without_env_still_clean(
        tmp_path, capsys, monkeypatch):
    mod = _load_script()
    empty = tmp_path / "empty-data"
    empty.mkdir()
    monkeypatch.setattr(mod, "DATA_DIR", empty)
    monkeypatch.delenv("GS_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("GS_TELEGRAM_CHAT_ID", raising=False)
    assert mod.main(["--out", str(tmp_path / "out"),
                     "--notify-telegram"]) == \
        mod.BACKTEST_MISSING_DATA_EXIT
    out, err = capsys.readouterr()
    assert "Backtest data missing:" in err
    assert "skipped" in out  # notify skipped honestly, run still clean


def test_run_subprocess_surfaces_clear_message_not_traceback(
        tmp_path, monkeypatch):
    class _Proc:
        returncode = 1
        stdout = ""
        stderr = (
            "File \".../pandas/io/parsers/readers.py\", line 1907\n"
            "Traceback (most recent call last):\n"
            "FileNotFoundError: [Errno 2] No such file or directory\n"
            + MISSING_LINE + "\n")

    monkeypatch.setattr(alert_cmds.subprocess, "run",
                        lambda *a, **k: _Proc())
    job = BacktestJob(days=90, playbook=None, drop_back=False,
                      out_dir=tmp_path)
    with pytest.raises(RuntimeError) as exc_info:
        _run_backtest_subprocess(job)
    msg = str(exc_info.value)
    assert "Backtest data missing:" in msg
    assert "git pull origin main" in msg
    assert "Traceback" not in msg
    assert "readers.py" not in msg


def test_finish_job_delivers_clear_missing_data_text(tmp_path):
    def _raise_missing(job):
        raise RuntimeError(MISSING_LINE)

    set_backtest_runner_for_tests(_raise_missing)
    try:
        delivered = []
        ctx = CommandContext(
            journal=None, journal_path=tmp_path / "j.jsonl",
            reports_dir=tmp_path / "reports",
            feed_status={"ok": True}, deliver=delivered.append)
        job = BacktestJob(days=90, playbook=None, drop_back=False,
                          out_dir=tmp_path)
        _finish_backtest_job(job, ctx, d1_stale=False)
        assert delivered, "failed run must still deliver a report"
        text = delivered[0].text or ""
        assert "Backtest data missing" in text
        assert "pull the latest" in text
        assert "the run itself errored" not in text
        assert "Traceback" not in text
    finally:
        set_backtest_runner_for_tests(None)
        reset_backtest_jobs_for_tests()
