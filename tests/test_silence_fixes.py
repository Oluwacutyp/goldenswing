"""Silence-bug fixes: causal fvg_limit, OANDA granularity, below-floor log.

Bug 1 — fvg_limit live path used the batch forward-walk sim, which needs
future bars and always failed at the live edge; worse, the zone was
consumed before the sim check, so it died on first touch without ever
emitting. The live path now uses the causal fill check and consumes
only on a confirmed fill.

Bug 2 — the OANDA feed passed "D1" verbatim as granularity; OANDA's
CandlestickGranularity enum has no D1 (daily is "D"), so every D1
request was rejected and the D1 feed never succeeded once. Mapping is
applied at the OANDA request boundary only.

Bug 3 — the paper loop's score floor dropped below-floor emissions
silently (no log, no counter). They are now journal-logged, one
aggregated record per bar.
"""
import pandas as pd
import pytest

from goldenswing.backtest import mtf_data
from goldenswing.execution.runner import LiveRunner, RunnerConfig
from goldenswing.feed import oanda as oanda_mod
from goldenswing.feed.oanda import (
    OandaFeed, fetch_oanda, to_oanda_granularity)
from goldenswing.golden.detector import GoldenSignal
from goldenswing.playbooks.base import EngineContext
from goldenswing.playbooks.fvg_limit import FVGLimitPlaybook

# OANDA CandlestickGranularity enum, verbatim from
# https://developer.oanda.com/rest-live-v20/instrument-df/
OANDA_ENUM = {
    "S5", "S10", "S15", "S30", "M1", "M2", "M4", "M5", "M10", "M15",
    "M30", "H1", "H2", "H3", "H4", "H6", "H8", "H12", "D", "W", "M",
}


# ------------------------------------------------------------------ Bug 1
def _fvg_bars(touch_low: float, n_prefix: int = 14) -> pd.DataFrame:
    """Synthetic bars: flat prefix, then a bullish FVG [100, 101] formed
    at bar n_prefix+2, then a touch bar.

    touch_low controls the touch bar: <= ~100.89 penetrates the near
    edge (fill), above it grazes (touch without fill).
    """
    rows = []
    for _ in range(n_prefix):
        rows.append((100.0, 100.5, 99.5, 100.0))
    rows.append((100.0, 100.0, 99.0, 99.5))    # bar0 of the pattern
    rows.append((100.2, 100.8, 100.2, 100.5))  # anchor (no gap)
    rows.append((101.5, 102.0, 101.0, 101.8))  # bar2: low 101 > high0 100
    rows.append((101.0, 101.5, touch_low, 101.2))  # touch bar
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="h",
                        tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=idx)


def test_fvg_limit_truncated_df_still_emits():
    """Truncated at the touch bar (the live edge): the old forward-walk
    sim returned None here, so the zone died without emitting."""
    df = _fvg_bars(touch_low=100.5)
    t = len(df) - 1
    pb = FVGLimitPlaybook()
    sigs = pb.live_signal([], t, df)
    assert len(sigs) == 1
    assert sigs[0].kind == "fvg_limit"
    assert sigs[0].direction == 1
    assert sigs[0].entry_ref == pytest.approx(101.0)  # near edge


def test_fvg_limit_graze_does_not_consume_zone():
    """A grazed first touch leaves the zone unconsumed; a later
    penetrating touch emits."""
    df_graze = _fvg_bars(touch_low=100.95)
    t = len(df_graze) - 1
    pb = FVGLimitPlaybook()
    assert pb.live_signal([], t, df_graze) == []
    assert pb._consumed == set()
    # Append a penetrating bar: same zone, now it fills.
    extra = pd.DataFrame(
        [(101.0, 101.5, 100.5, 101.2)],
        columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex([df_graze.index[-1] + pd.Timedelta(hours=1)],
                               tz="UTC"))
    df2 = pd.concat([df_graze, extra])
    sigs = pb.live_signal([], len(df2) - 1, df2)
    assert len(sigs) == 1
    assert sigs[0].entry_ref == pytest.approx(101.0)


def test_fvg_limit_batch_path_unchanged():
    """Batch grading still walks forward and books the outcome — the
    refactor must not change batch results."""
    df = _fvg_bars(touch_low=100.5)
    # Follow-through bar hits the 1.5R target (target ~= 102.93).
    extra = pd.DataFrame(
        [(101.2, 103.5, 100.9, 102.0)],
        columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex([df.index[-1] + pd.Timedelta(hours=1)],
                               tz="UTC"))
    df2 = pd.concat([df, extra])
    ctx = EngineContext(df=df2, states=[], years=0.5,
                         instrument="XAUUSD")
    cands = FVGLimitPlaybook().candidates(ctx)
    assert len(cands) == 1
    assert cands[0].outcome.r_net == pytest.approx(1.5)
    assert cands[0].outcome.exit_reason == "target"


# ------------------------------------------------------------------ Bug 2
def test_oanda_granularity_mapping():
    assert to_oanda_granularity("D1") == "D"
    assert to_oanda_granularity("W1") == "W"
    assert to_oanda_granularity("MN1") == "M"
    # Valid OANDA tokens pass through untouched.
    for tf in ("H1", "H4", "M1", "M5", "M15", "M30", "H2", "H12"):
        assert to_oanda_granularity(tf) == tf
    # Every mapped value is in OANDA's documented enum.
    for tf in ("D1", "H4", "H1", "M15", "M30", "M5", "W1", "MN1"):
        assert to_oanda_granularity(tf) in OANDA_ENUM


def test_fetch_oanda_uses_mapped_granularity(monkeypatch):
    seen: dict = {}

    def fake_get(path, params, timeout=15.0):
        seen.update(params)
        return {"candles": []}

    monkeypatch.setattr(oanda_mod, "_get", fake_get)
    fetch_oanda(granularity="D1", count=5)
    assert seen["granularity"] == "D"


def test_fetch_oanda_range_uses_mapped_granularity(monkeypatch):
    seen: dict = {}

    def fake_get(path, params, timeout=15.0):
        seen.update(params)
        return {"candles": []}

    monkeypatch.setattr(mtf_data, "_get", fake_get)
    out = mtf_data.fetch_oanda_range(
        "XAU_USD", "D1", start="2026-01-01T00:00:00Z",
        end="2026-01-02T00:00:00Z")
    assert seen["granularity"] == "D"
    assert len(out) == 0


def test_poll_status_surfaces_error_before_first_success():
    feed = OandaFeed()
    assert feed.poll_status()["reason"] == "no successful poll yet"
    feed._note_failure("OANDA HTTP 400: invalid granularity D1")
    status = feed.poll_status()
    assert status["ok"] is False
    assert "OANDA HTTP 400" in status["reason"]


# ------------------------------------------------------------------ Bug 3
def _cfg(tmp_path):
    return RunnerConfig(
        journal_path=tmp_path / "journal.jsonl",
        risk_state_path=tmp_path / "risk.json",
        kill_switch_path=tmp_path / "KILL",
        forecast_bundle_path=None,
        auto_execute=False)


class _StubPlaybook:
    def __init__(self, name, score):
        self.name = name
        self._score = score

    def live_signal(self, states, t, df=None):
        return [GoldenSignal(
            index=t, timestamp=None, direction=1, score=self._score,
            kind="stub", factors=[], entry_zone=(100.0, 100.0),
            entry_ref=100.0, invalidation=99.0, targets=[101.0], atr=1.0)]


class _StubEngine:
    def gate_live_signal(self, pb, sig, ctx, hierarchy_verdict=None):
        return False, "test gate"


def _run_bar(tmp_path, stubs, floor=60.0):
    from collections import defaultdict

    from goldenswing.backtest.data import load_gold_h1

    runner = LiveRunner(config=_cfg(tmp_path), senders=[])
    df = load_gold_h1().iloc[-200:].copy()
    t = len(df) - 1
    stats = defaultdict(int)
    runner._paper_loop_bar(df, t, stubs, _StubEngine(), None, set(),
                           floor, 0, stats, {}, 0.5)
    notes = [r for r in runner.journal.records()
             if r.get("type") == "note"
             and "below score floor" in r.get("text", "")]
    return runner, stats, notes


def test_below_floor_emission_is_journaled(tmp_path):
    _, stats, notes = _run_bar(
        tmp_path, [_StubPlaybook("stub_pb", 40.0)])
    assert len(notes) == 1
    assert "stub_pb" in notes[0]["text"]
    assert "score=40" in notes[0]["text"]
    assert "floor=60" in notes[0]["text"]
    assert stats["signals"] == 0  # still dropped, just no longer silent


def test_below_floor_aggregates_one_record_per_bar(tmp_path):
    _, stats, notes = _run_bar(
        tmp_path, [_StubPlaybook("pb_a", 40.0),
                   _StubPlaybook("pb_b", 55.0)])
    assert len(notes) == 1
    assert "pb_a" in notes[0]["text"]
    assert "pb_b" in notes[0]["text"]
    assert stats["signals"] == 0


def test_above_floor_is_not_logged_as_below_floor(tmp_path):
    runner, stats, notes = _run_bar(
        tmp_path, [_StubPlaybook("stub_pb", 70.0)])
    assert notes == []
    assert stats["gated"] == 1  # reached the gate, blocked there instead


# ------------------------------------------------------------------ Item 7
# Live-loop score floor: run_live had the same silent-drop pattern the
# paper loop just got fixed (bugs phase). Same helper, same one-record-
# per-bar treatment, "live loop:" prefix.
def test_live_loop_below_floor_logged_via_helper(tmp_path):
    runner = LiveRunner(config=_cfg(tmp_path), senders=[])
    runner._log_below_floor([("stub_pb", 40.0)], 60.0, 199,
                             "2026-01-01 00:00:00+00:00",
                             prefix="live loop")
    notes = [r for r in runner.journal.records()
             if r.get("type") == "note"
             and "below score floor" in r.get("text", "")]
    assert len(notes) == 1
    assert notes[0]["text"].startswith("live loop: below score floor")
    assert "stub_pb" in notes[0]["text"]
    assert "score=40" in notes[0]["text"]
    assert "floor=60" in notes[0]["text"]


def test_log_below_floor_empty_is_silent(tmp_path):
    runner = LiveRunner(config=_cfg(tmp_path), senders=[])
    runner._log_below_floor([], 60.0, 199, "2026-01-01", prefix="live loop")
    assert runner.journal.records() == []
