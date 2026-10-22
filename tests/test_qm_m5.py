"""Regression tests for QMM5Playbook (Quasimodo on M5).

Synthetic bars are hand-built OHLC with chained opens.  The bearish
series is a clean textbook QM: head 100 (bar 10) -> neckline 90
(bar 17) -> failed rally to QM level 96 (bar 23) -> close below the
neckline (bar 27) -> retracement touch of 96 (bar 31) -> neckline
target 90 (bar 34).  The bullish series is the exact mirror
(price' = 200 - price, high/low swapped).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from goldenswing.confluence import ConfluenceEngine
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.risk import RiskGuards
from goldenswing.golden.detector import playbook_of
from goldenswing.playbooks.base import EngineContext
from goldenswing.playbooks.qm_m5 import QMM5Playbook

# (open, high, low, close) — opens chain from the previous close.
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


def _df(rows, freq="5min"):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [1000.0] * len(rows)},
        index=idx,
    )


def _mirror(rows):
    """Exact bullish mirror: price' = 200 - price, high/low swapped."""
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


# ------------------------------------------------------------- clean fires
def test_clean_bearish_qm_fires():
    df = _df(BEARISH)
    cands = QMM5Playbook().candidates(_ctx(df))
    assert len(cands) == 1
    c = cands[0]
    assert c.playbook == "qm_m5"
    assert c.direction == -1
    assert c.signal_index == 31  # the retracement-touch bar
    sig = c.outcome.signal
    assert sig.entry_ref == pytest.approx(96.0)        # QM level
    assert sig.targets == [pytest.approx(90.0)]        # neckline target
    assert sig.invalidation == pytest.approx(100.37, abs=0.15)  # head + buffer
    assert c.outcome.exit_reason == "target"
    assert c.outcome.r_net == pytest.approx(1.37, rel=0.05)
    assert c.context["head"] == pytest.approx(100.0)
    assert c.context["neckline"] == pytest.approx(90.0)
    assert c.context["qm_level"] == pytest.approx(96.0)
    assert c.context["break_index"] == 27
    assert "displacement_break" in c.context["tags"]  # break bar body >= 1 ATR
    assert "sweep_head" not in c.context["tags"]


def test_clean_bullish_qm_fires_mirrored():
    df = _df(_mirror(BEARISH))
    cands = QMM5Playbook().candidates(_ctx(df))
    assert len(cands) == 1
    c = cands[0]
    assert c.direction == 1
    assert c.signal_index == 31
    assert c.outcome.signal.entry_ref == pytest.approx(104.0)
    assert c.outcome.signal.targets == [pytest.approx(110.0)]
    assert c.outcome.exit_reason == "target"
    assert c.outcome.r_net == pytest.approx(1.37, rel=0.05)


# ------------------------------------------------------------- near misses
def test_no_fire_when_rally_exceeds_head():
    """No lower high: the rally takes out the head, so there is no
    failed rally / no QM level — the pattern must not fire."""
    rows = list(BEARISH)
    rows[23] = (95.0, 101.0, 95.0, 100.5)   # rally ABOVE the head
    rows[24] = (100.5, 100.5, 100.0, 100.0)
    df = _df(rows)
    assert QMM5Playbook().candidates(_ctx(df)) == []


def test_no_fire_without_neckline_break():
    """Price never closes through the neckline, so no break of
    structure and no retracement trade — must not fire."""
    rows = list(BEARISH)
    rows[27] = (92.0, 93.0, 92.0, 93.0)
    rows[28] = (93.0, 95.0, 93.0, 95.0)
    rows[29] = (95.0, 98.0, 95.0, 98.0)
    rows[30] = (98.0, 100.5, 98.0, 100.5)  # rallies through the head
    rows[31] = (100.5, 100.5, 99.0, 99.5)
    rows[32] = (99.5, 99.5, 94.0, 94.0)
    rows[33] = (94.0, 94.0, 92.0, 92.0)
    rows[34] = (92.0, 92.0, 89.5, 90.0)
    df = _df(rows)
    assert QMM5Playbook().candidates(_ctx(df)) == []


def test_no_fire_when_invalidation_before_touch():
    """Break happens, but price closes beyond the head before ever
    touching the QM level — pattern dead, must not fire."""
    rows = list(BEARISH)
    rows[28] = (89.5, 101.0, 89.5, 100.5)  # close beyond the head
    df = _df(rows)
    assert QMM5Playbook().candidates(_ctx(df)) == []


def test_short_series_is_safe():
    df = _df(BEARISH[:5])
    assert QMM5Playbook().candidates(_ctx(df)) == []


# ------------------------------------------------------- cursor + parity
def test_live_signal_never_rewinds_cursor():
    df = _df(BEARISH)
    pb = QMM5Playbook()
    assert pb.live_signal([], 20, df) == []
    assert pb._processed == 20
    assert pb.live_signal([], 15, df) == []      # rewind: no-op
    assert pb._processed == 20                   # cursor did not move back
    sigs = pb.live_signal([], 31, df)
    assert len(sigs) == 1 and sigs[0].index == 31
    assert pb.live_signal([], 31, df) == []      # same bar twice: no dup


def test_batch_live_parity_synthetic():
    df = _df(BEARISH)
    batch = _batch_keys(QMM5Playbook().candidates(_ctx(df)))
    live = _live_keys(QMM5Playbook(), df)
    assert batch == sorted(live)
    assert len(batch) == 1


def test_batch_live_parity_real_m5():
    path = Path("data/OANDA_XAU_USD_M5_20260401_20261012.csv")
    raw = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
    df = raw[["open", "high", "low", "close", "volume"]]
    batch = _batch_keys(QMM5Playbook().candidates(_ctx(df)))
    live = _live_keys(QMM5Playbook(), df)
    assert batch == sorted(live)


def test_export_import_round_trip():
    df = _df(BEARISH)
    pb1 = QMM5Playbook()
    for t in range(29):
        pb1.live_signal([], t, df)
    state = pb1.export_state()
    pb2 = QMM5Playbook()
    pb2.import_state(state)
    tail1, tail2 = [], []
    for t in range(29, len(df)):
        tail1 += [s.index for s in pb1.live_signal([], t, df)]
        tail2 += [s.index for s in pb2.live_signal([], t, df)]
    assert tail1 == tail2 == [31]


def test_import_empty_state_resets():
    df = _df(BEARISH)
    pb = QMM5Playbook()
    for t in range(32):
        pb.live_signal([], t, df)
    assert pb._processed == 31
    pb.import_state({})  # reset: fresh engine, cursor back to -1
    assert pb._processed == -1
    assert [k[0] for k in _live_keys(pb, df)] == [31]


# ------------------------------------------------------- gates + wiring
def test_check_live_signal_gates():
    df = _df(BEARISH)
    pb = QMM5Playbook()
    sigs = []
    for t in range(len(df)):
        sigs += pb.live_signal([], t, df)
    assert len(sigs) == 1
    ok, _ = pb.check_live_signal(sigs[0], _ctx(df))
    assert ok
    # Tampered: neckline target moved to the entry -> 0R, must fail.
    sigs[0].targets = [sigs[0].entry_ref]
    ok, reason = pb.check_live_signal(sigs[0], _ctx(df))
    assert not ok and "0.00R" in reason


def test_router_maps_qm_kind():
    assert playbook_of(None, "qm_m5") == "qm_m5"


def test_confluence_live_book_not_shadow():
    """Owner directive 2026-10-22: qm_m5 is live — its candidates
    book in the LIVE book, not the shadow book.  Detection coverage
    is unchanged (the same fixture still scores exactly 1 candidate)."""
    df = _df(BEARISH)
    tmp = Path(tempfile.mkdtemp())
    eng = ConfluenceEngine([QMM5Playbook()],
                           RiskGuards(tmp / "risk.json"),
                           TradeJournal(tmp / "j.jsonl"),
                           risk_mode="shadow")
    rep = eng.run(_ctx(df))
    assert len(rep.scored["qm_m5"]) == 1
    assert "qm_m5" not in rep.shadow  # live: never in the shadow book
    assert len(rep.live_outcomes) == 1  # the candidate books live
