"""Regression tests for QMD1Playbook (Quasimodo on D1).

Same textbook QM series as test_qm_m5.py, read as DAILY bars: head
100 (bar 10) -> neckline 90 (bar 17) -> failed rally to QM level 96
(bar 23) -> close below the neckline (bar 27) -> retracement touch
of 96 (bar 31) -> neckline target 90 (bar 34).  The D1 expiries
(60 / 20 / 20) all clear this series (head->break 17, break->touch
4, touch->target 3), so the clean series fires identically — the D1
params change the WINDOWS, not the structure.

Expiry tests insert flat bars (no strict extremes, so no new pivots)
to push one leg past its D1 window: a formation slower than a
quarter (60 bars) goes stale; a retracement slower than a month
(20 bars) expires.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from goldenswing.confluence import ConfluenceEngine
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.mt5_config import (
    EXECUTE_PLAYBOOK_NAMES, SHADOW_PLAYBOOK_NAMES)
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.risk import RiskGuards
from goldenswing.golden.detector import playbook_of
from goldenswing.playbooks.base import EngineContext
from goldenswing.playbooks.qm_d1 import D1_PARAMS, QMD1Playbook
from goldenswing.playbooks.qm_m5 import M5_PARAMS, QMM5Playbook
from goldenswing.reader.session_timeframes import mapping_for

# Same textbook series as test_qm_m5.py (bearish QM).
BEARISH = [
    (90.0, 90.5, 89.5, 90.0),
    (90.0, 91.5, 89.5, 91.0),
    (91.0, 92.5, 90.5, 92.0),
    (92.0, 93.5, 91.5, 93.0),
    (93.0, 94.5, 92.5, 94.0),
    (94.0, 95.5, 93.5, 95.0),
    (95.0, 96.5, 94.5, 96.0),
    (96.0, 97.5, 95.5, 97.0),
    (97.0, 98.5, 96.5, 98.0),
    (98.0, 99.5, 97.5, 99.0),
    (99.0, 100.0, 99.0, 99.5),   # 10: HEAD high 100
    (99.5, 99.5, 99.0, 99.0),
    (99.0, 99.0, 98.0, 98.0),
    (98.0, 98.0, 97.0, 97.0),
    (97.0, 97.0, 96.0, 96.0),
    (96.0, 96.0, 95.0, 95.0),
    (95.0, 95.0, 94.0, 94.0),
    (94.0, 94.0, 90.0, 92.0),   # 17: NECKLINE low 90 (wick)
    (92.0, 92.0, 91.0, 91.0),
    (91.0, 92.0, 91.0, 92.0),
    (92.0, 93.0, 92.0, 93.0),
    (93.0, 94.0, 93.0, 94.0),
    (94.0, 95.0, 94.0, 95.0),
    (95.0, 96.0, 95.0, 95.5),   # 23: QM LEVEL high 96
    (95.5, 95.5, 95.0, 95.0),
    (95.0, 95.0, 94.0, 94.0),
    (94.0, 94.0, 92.0, 92.0),
    (92.0, 92.0, 89.5, 89.5),   # 27: BREAK close 89.5 < 90
    (89.5, 90.0, 89.5, 90.0),
    (90.0, 92.0, 90.0, 92.0),
    (92.0, 94.0, 92.0, 94.0),
    (94.0, 96.3, 94.0, 95.0),   # 31: TOUCH high 96.3 >= 96
    (95.0, 95.0, 94.0, 94.0),
    (94.0, 94.0, 92.0, 92.0),
    (92.0, 92.0, 89.5, 90.0),   # 34: TARGET low 89.5 <= 90
    (90.0, 90.0, 89.0, 89.0),
    (89.0, 89.0, 88.5, 88.5),
    (88.5, 88.5, 88.0, 88.0),
    (88.0, 88.0, 87.5, 87.5),
    (87.5, 87.5, 87.0, 87.0),
    (87.0, 87.0, 86.5, 86.5),
    (86.5, 86.5, 86.0, 86.0),
    (86.0, 86.0, 85.5, 85.5),
    (85.5, 85.5, 85.0, 85.0),
    (85.0, 85.0, 84.5, 84.5),
    (84.5, 84.5, 84.0, 84.0),
    (84.0, 84.0, 83.5, 83.5),
    (83.5, 83.5, 83.0, 83.0),
    (83.0, 83.0, 82.5, 82.5),
    (82.5, 82.5, 82.0, 82.0),
    (82.0, 82.0, 81.5, 81.5),
]


def _df(rows, freq="1D"):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [1000.0] * len(rows)},
        index=idx,
    )


def _mirror(rows):
    return [(200.0 - o, 200.0 - l, 200.0 - h, 200.0 - c)
            for (o, h, l, c) in rows]


def _ctx(df):
    return EngineContext(df=df, states=[], years=1.0, instrument="XAUUSD")


def _live_keys(pb, df):
    out = []
    for t in range(len(df)):
        for sig in pb.live_signal([], t, df):
            out.append((sig.index, sig.direction, round(sig.entry_ref, 6),
                        round(sig.invalidation, 6),
                        round(sig.targets[0], 6)))
    return out


def _batch_keys(cands):
    return sorted((c.signal_index, c.direction,
                   round(c.outcome.signal.entry_ref, 6),
                   round(c.outcome.signal.invalidation, 6),
                   round(c.outcome.signal.targets[0], 6))
                  for c in cands)


# ------------------------------------------------------------- params
def test_d1_params_are_calendar_equivalents_not_blind_scales():
    # ATR-relative thresholds are scale-free: identical to M5.
    assert D1_PARAMS.min_leg_atr == M5_PARAMS.min_leg_atr == 0.5
    assert D1_PARAMS.min_fail_atr == M5_PARAMS.min_fail_atr == 0.25
    assert D1_PARAMS.stop_buffer_atr == M5_PARAMS.stop_buffer_atr == 0.25
    assert D1_PARAMS.min_target_r == M5_PARAMS.min_target_r == 1.0
    assert D1_PARAMS.swing_n == M5_PARAMS.swing_n == 3
    assert D1_PARAMS.atr_n == M5_PARAMS.atr_n == 14
    # Bar-count expiries are the D1 calendar equivalents (quarter /
    # month / month), NOT the M5 counts.
    assert D1_PARAMS.max_pattern_bars == 60
    assert D1_PARAMS.max_retrace_bars == 20
    assert D1_PARAMS.max_hold_bars == 20
    assert (D1_PARAMS.max_pattern_bars,
            D1_PARAMS.max_retrace_bars,
            D1_PARAMS.max_hold_bars) != (
        M5_PARAMS.max_pattern_bars, M5_PARAMS.max_retrace_bars,
        M5_PARAMS.max_hold_bars)


# ------------------------------------------------------------- clean fires
def test_clean_bearish_qm_fires_on_daily():
    df = _df(BEARISH)
    cands = QMD1Playbook().candidates(_ctx(df))
    assert len(cands) == 1
    c = cands[0]
    assert c.playbook == "qm_d1"
    assert c.direction == -1
    assert c.signal_index == 31
    sig = c.outcome.signal
    assert sig.entry_ref == pytest.approx(96.0)
    assert sig.targets == [pytest.approx(90.0)]
    assert sig.invalidation == pytest.approx(100.37, abs=0.15)
    assert c.outcome.exit_reason == "target"
    assert c.outcome.r_net == pytest.approx(1.37, rel=0.05)
    assert c.context["head"] == pytest.approx(100.0)
    assert c.context["break_index"] == 27


def test_clean_bullish_qm_fires_mirrored():
    df = _df(_mirror(BEARISH))
    cands = QMD1Playbook().candidates(_ctx(df))
    assert len(cands) == 1
    assert cands[0].direction == 1
    assert cands[0].signal_index == 31
    assert cands[0].outcome.signal.entry_ref == pytest.approx(104.0)
    assert cands[0].outcome.exit_reason == "target"


# ------------------------------------------------------------- D1 expiries
def test_no_fire_when_formation_slower_than_a_quarter():
    """45 flat bars after the head push head->break to 62 bars:
    past the 60-bar (one quarter) formation window — stale."""
    flat = [(99.0, 99.0, 99.0, 99.0)] * 45
    rows = BEARISH[:11] + flat + BEARISH[11:]
    df = _df(rows)
    assert QMD1Playbook().candidates(_ctx(df)) == []
    # ...but the SAME slow formation still fires on M5 windows
    # (120 bars): the D1 expiry is what kills it, not the structure.
    df_m5 = _df(rows, freq="5min")
    assert len(QMM5Playbook().candidates(
        EngineContext(df=df_m5, states=[], years=1.0,
                      instrument="XAUUSD"))) == 1


def test_no_fire_when_retracement_slower_than_a_month():
    """25 flat bars after the break push break->touch to 29 bars:
    past the 20-bar (one month) retracement window — expired."""
    flat = [(90.0, 90.0, 89.5, 90.0)] * 25
    rows = BEARISH[:28] + flat + BEARISH[28:]
    df = _df(rows)
    assert QMD1Playbook().candidates(_ctx(df)) == []
    # ...but the M5 48-bar retracement window still accepts it.
    df_m5 = _df(rows, freq="5min")
    assert len(QMM5Playbook().candidates(
        EngineContext(df=df_m5, states=[], years=1.0,
                      instrument="XAUUSD"))) == 1


# ------------------------------------------------------- cursor + parity
def test_batch_live_parity_synthetic():
    df = _df(BEARISH)
    batch = _batch_keys(QMD1Playbook().candidates(_ctx(df)))
    live = _live_keys(QMD1Playbook(), df)
    assert batch == sorted(live)
    assert len(batch) == 1


def test_batch_live_parity_real_daily():
    raw = pd.read_csv("data/OANDA_XAU_USD_H1_20241001_20261012.csv",
                      parse_dates=["ts"], index_col="ts")
    df = raw[["open", "high", "low", "close", "volume"]].resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum"}).dropna()
    batch = _batch_keys(QMD1Playbook().candidates(_ctx(df)))
    live = _live_keys(QMD1Playbook(), df)
    assert batch == sorted(live)


def test_export_import_round_trip():
    df = _df(BEARISH)
    pb1 = QMD1Playbook()
    for t in range(29):
        pb1.live_signal([], t, df)
    state = pb1.export_state()
    pb2 = QMD1Playbook()
    pb2.import_state(state)
    tail1, tail2 = [], []
    for t in range(29, len(df)):
        tail1 += [s.index for s in pb1.live_signal([], t, df)]
        tail2 += [s.index for s in pb2.live_signal([], t, df)]
    assert tail1 == tail2 == [31]


# ------------------------------------------------------- gates + wiring
def test_check_live_signal_gates():
    df = _df(BEARISH)
    pb = QMD1Playbook()
    sigs = []
    for t in range(len(df)):
        sigs += pb.live_signal([], t, df)
    assert len(sigs) == 1
    ok, _ = pb.check_live_signal(sigs[0], _ctx(df))
    assert ok
    sigs[0].targets = [sigs[0].entry_ref]
    ok, reason = pb.check_live_signal(sigs[0], _ctx(df))
    assert not ok and "0.00R" in reason


def test_router_maps_qm_d1_kind():
    assert playbook_of(None, "qm_d1") == "qm_d1"


def test_session_clock_is_d1():
    m = mapping_for(pd.Timestamp("2026-03-06 08:00", tz="UTC"),
                    playbook="qm_d1")
    assert m.entry_clock == "D1"
    assert "qm_d1 keeps D1 clock; session is annotation only" in m.tags


def test_confluence_live_book_not_shadow():
    """Owner directive 2026-10-22: qm_d1 is live — its candidates
    book in the LIVE book, not the shadow book.  Detection coverage
    is unchanged (the same fixture still scores exactly 1 candidate)."""
    df = _df(BEARISH)
    tmp = Path(tempfile.mkdtemp())
    eng = ConfluenceEngine([QMD1Playbook()],
                           RiskGuards(tmp / "risk.json"),
                           TradeJournal(tmp / "j.jsonl"),
                           risk_mode="shadow")
    rep = eng.run(_ctx(df))
    assert len(rep.scored["qm_d1"]) == 1
    assert "qm_d1" not in rep.shadow  # live: never in the shadow book
    assert len(rep.live_outcomes) == 1  # the candidate books live


def test_qm_d1_is_executing_in_config_not_shadow():
    """Owner directive 2026-10-22: qm_d1 executes — the roster of
    record lists it as executing, never shadow."""
    assert "qm_d1" not in SHADOW_PLAYBOOK_NAMES
    assert "qm_d1" in EXECUTE_PLAYBOOK_NAMES
    assert QMD1Playbook.live is True


def test_runner_scans_d1_frame_not_h1_path(tmp_path):
    """The runner evaluates qm_d1 on the D1 frame (_run_qm_d1), NOT
    through the H1-clocked _get_playbooks path (which would run D1
    logic on H1 bars).  P3/A14: the scan is a REAL order path on
    the demo (the demo IS the shadow) — this bare runner has no
    live executor wired, so the touch is evaluated + journaled
    and the order attempt fails safe inside the per-candidate
    guard; the full fill path is covered in test_p3_trader.py."""
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL")
    runner = MT5Runner(cfg)
    names = {pb.name for pb in runner._get_playbooks()}
    assert "qm_d1" not in names
    # Phase 2: present the frame as of the touch's own close
    # (touch at bar 31 = the last bar).  The full-history frame
    # is late-discovery provenance — expired, never routed
    # (candidate_lifecycle; covered in test_phase2_fixes.py).
    df = _df(BEARISH).iloc[:32]
    stats: dict = {}
    first = runner._run_qm_d1(df, stats)
    assert len(first) == 1
    assert first[0]["touch_index"] == 31
    assert stats["qm_d1_signals"] == 1
    journal = (tmp_path / "mt5_journal.jsonl").read_text()
    assert "mt5 qm_d1 -1 touch" in journal
    assert "evaluating for a live order on the D1 clock" in journal
    # Second scan: same touch is NOT re-evaluated.
    second = runner._run_qm_d1(df, stats)
    assert second == []
    assert stats["qm_d1_signals"] == 1


def test_runner_scan_respects_disabled(tmp_path):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        disabled_playbooks=("qm_d1",))
    runner = MT5Runner(cfg)
    assert runner._run_qm_d1(_df(BEARISH), {}) == []
