"""Owner addendum: no shortcuts, full run, Telegram export,
test-what-is-live. Tests for analyze() TP/regime + script guards."""
import json
import pathlib
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "scripts"))
import backtest_full_engine as bfe  # noqa: E402
import backtest_mt5_full as mt5full  # noqa: E402


def _journal(tmp_path):
    records = [
        {"type": "open", "ticket": 1, "playbook": "range_sweep_reclaim",
         "direction": 1, "entry": 4100.0, "stop": 4090.0, "volume": 1.0},
        {"type": "note", "text":
         "mt5 tick: ticket 1 partial close 0.5 lots at 4110.0"},
        {"type": "close", "ticket": 1, "r_net": 0.5, "pnl_money": 50.0,
         "reason": "stop"},
        {"type": "open", "ticket": 2, "playbook": "trend_pullback",
         "direction": -1, "entry": 4200.0, "stop": 4210.0, "volume": 1.0},
        {"type": "close", "ticket": 2, "r_net": -1.0, "pnl_money": -100.0,
         "reason": "stop"},
    ]
    path = tmp_path / "journal.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def test_analyze_tp_hit_rates_and_regime(tmp_path):
    runner = SimpleNamespace(blocked_stats=lambda: {})
    broker = SimpleNamespace(account=lambda: {"balance": 10000})
    frames = {"W1": pd.DataFrame(), "MN1": pd.DataFrame()}
    report = bfe.analyze(_journal(tmp_path), runner, {}, frames,
                         broker, [{"mid": 4150.0}])
    tp = report["tp_hit_rates"]
    assert tp["tp1"] == {"hits": 1, "denominator": 2, "rate": 0.5}
    assert tp["tp2"]["hits"] == 0
    assert tp["tp3"]["hits"] == 0
    reg = report["regime_breakdown"]
    assert reg["RANGE"]["trades"] == 1
    assert reg["RANGE"]["total_r"] == pytest.approx(0.75)
    assert reg["RANGE"]["win_rate"] == 1.0
    assert reg["TREND"]["trades"] == 1
    assert reg["TREND"]["total_r"] == pytest.approx(-1.0)
    assert reg["TREND"]["win_rate"] == 0.0


def test_script_has_no_shortcut_flags():
    src = pathlib.Path(mt5full.__file__).read_text()    # No argparse flag defines them (docstring may name them as
    # removed), and nothing assigns a playbook subset.
    assert 'add_argument("--days"' not in src
    assert 'add_argument("--only-playbook"' not in src
    assert 'add_argument("--max-bars"' not in src
    assert "execute_playbooks =" not in src
    # Full ticks: the runner consumes EVERY tick.
    assert "max_ticks=len(ticks)" in src


def test_script_rejects_removed_flags():
    with pytest.raises(SystemExit) as excinfo:
        mt5full.main(["--days", "30"])
    assert excinfo.value.code == 2
    with pytest.raises(SystemExit) as excinfo:
        mt5full.main(["--only-playbook", "range_qm"])
    assert excinfo.value.code == 2


def _report():
    return {
        "trades": {"closed": 2, "open_at_end": 0, "win_rate": 0.5,
                   "profit_factor": 1.5, "avg_r": 0.25,
                   "max_drawdown_r": 1.0, "total_r": 0.5,
                   "total_dollar": 50.0},
        "window": {"start": "2024-01-01", "end_open": "2026-01-01",
                   "ticks": 1000, "elapsed_s": 12.3},
        "data_provenance": {"ticks": "real_mt5_ticks",
                            "H1": {"rows": 100}},
        "tp_hit_rates": {"tp1": {"hits": 1, "denominator": 2,
                                 "rate": 0.5}},
        "regime_breakdown": {"RANGE": {"trades": 1, "total_r": 0.75,
                                       "win_rate": 1.0}},
    }


def test_telegram_summary_covers_required_fields():
    text = mt5full.build_telegram_summary(_report())
    for needle in ("Window:", "TFs", "Ticks:", "Win rate:", "PF:",
                   "Expectancy:", "Max DD:", "TP hit rates:",
                   "Regime breakdown:", "Resampling: NONE",
                   "Full engine", "all playbooks"):
        assert needle in text


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_telegram_send_success(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _Proc(0, '{"ok": true}', "")

    monkeypatch.setattr("subprocess.run", fake_run)
    res = mt5full.send_telegram_summary("hello", chat_id="5478650254",
                                        tg_call="/fake/tg-call")
    assert res == {"sent": True, "error": None}
    assert "--method" in seen["cmd"] and "sendMessage" in seen["cmd"]


def test_telegram_send_failure_reported_not_skipped(monkeypatch):
    def fake_run(cmd, **kw):
        return _Proc(1, "{}", "boom")

    monkeypatch.setattr("subprocess.run", fake_run)
    res = mt5full.send_telegram_summary("hello", chat_id="5478650254",
                                        tg_call="/fake/tg-call")
    assert res["sent"] is False
    assert "boom" in res["error"]


def test_script_symbol_defaults_to_xauusdm():
    # Owner's Exness terminal lists gold as XAUUSDm. The script's
    # --symbol default (the broker symbol data is fetched under)
    # must be XAUUSDm, not XAUUSD.
    import pathlib as _pl

    src = _pl.Path(mt5full.__file__).read_text()
    assert 'default=DEFAULT_SYMBOL' in src
    assert mt5full.DEFAULT_SYMBOL == "XAUUSDm"
    # And the fetch path actually receives the CLI symbol.
    assert "ensure_mt5_native(args.symbol" in src
