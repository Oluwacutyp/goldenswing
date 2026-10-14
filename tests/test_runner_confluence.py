"""Tests for the Phase-10 runner rewire (confluence path)."""
import pytest

from goldenswing.execution.runner import LiveRunner, RunnerConfig


def _cfg(tmp_path, **kw):
    return RunnerConfig(
        journal_path=tmp_path / "journal.jsonl",
        risk_state_path=tmp_path / "risk.json",
        kill_switch_path=tmp_path / "KILL",
        forecast_bundle_path=None,
        auto_execute=False,
        **kw)


def test_confluence_is_default(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.use_confluence is True


def test_confluence_path_runs_end_to_end(tmp_path):
    """Partial run (fast): the confluence path produces signals, alerts,
    and the expected result structure."""
    cfg = _cfg(tmp_path)
    runner = LiveRunner(config=cfg, senders=[])
    out = runner.run_bars(max_bars=3000)
    assert out["bars"] == 3000
    assert out["signals"] > 0
    assert out["alerts"] >= 0
    assert out["paper_balance"] > 0


def test_legacy_flag_still_works(tmp_path):
    """use_confluence=False runs the old detector path."""
    from goldenswing.golden.detector import GoldenSignal
    import pandas as pd

    idx = pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC")
    df = pd.DataFrame({
        "open": 2000.0, "high": 2005.0, "low": 1995.0,
        "close": 2002.0, "volume": 100.0}, index=idx)
    sig = GoldenSignal(
        index=100, timestamp=idx[100], direction=1, score=80.0,
        kind="continuation", factors=["test"],
        entry_zone=(2049.0, 2051.0), entry_ref=2050.0,
        invalidation=2040.0, targets=[2075.0], atr=10.0)

    class _Stub:
        def scan(self, states):
            return [sig]

    cfg = _cfg(tmp_path, use_confluence=False)
    runner = LiveRunner(config=cfg, detector=_Stub(), senders=[])
    out = runner.run_bars(df)
    assert out["path"] == "legacy-detector"
    assert out["signals"] == 1


def test_playbooks_lazy_build(tmp_path):
    cfg = _cfg(tmp_path)
    runner = LiveRunner(config=cfg, senders=[])
    pbs = runner._get_playbooks()
    assert len(pbs) == 5
    names = {pb.name for pb in pbs}
    assert names == {"sweep_reversal", "fvg_midline", "trendline_retest",
                     "fvg_limit", "qm_m5"}
    # cached
    assert runner._get_playbooks() is pbs


def test_engine_lazy_build(tmp_path):
    cfg = _cfg(tmp_path)
    runner = LiveRunner(config=cfg, senders=[])
    eng = runner._get_engine()
    assert eng.risk_mode == "shadow"
    assert runner._get_engine() is eng
