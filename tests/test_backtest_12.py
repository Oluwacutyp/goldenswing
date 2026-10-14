"""Tests for the multi-TF $12 backtest (synthetic bars only).

The $12 honesty rule: 1% risk = $0.12.  A $20 stop means 1-unit risk
of $20.00 — infeasible.  A $0.10 stop means 1-unit risk of $0.10 —
feasible.  Fractional-unit R measures strategy quality; it is never
presented as OANDA-executable.
"""
from pathlib import Path

import pandas as pd
import pytest

from goldenswing.backtest.mtf12 import (
    build_tf_frames, format_report, oanda_1unit_feasible, resample_bars,
    run_mtf12_backtest, session_of,
)
from goldenswing.live.multitf import PAPER_START_BALANCE_12, TimeframeConfig

PRESET = TimeframeConfig("D1", "H4", "H1", swing_n=1)


def _df(rows, start="2026-01-01"):
    idx = pd.date_range(start, periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=idx)


# ------------------------------------------------------- synthetic series
# One trigger frame: long pattern -> target exit -> flat -> short
# pattern -> target exit -> flat.  Bias/zone frames carry the HTF
# structure on the same hourly clock (separate frames, same index).
def trigger_frame():
    flat_a = [(98.5, 100, 98, 99.5)] * 25
    long_pat = [
        (99, 100, 98, 99), (97, 99, 96, 97), (97.5, 98, 97, 97.5),
        (100, 101, 98, 100), (98, 99, 97, 98), (99.5, 100, 98, 99.5),
        (101, 102, 99, 101), (99, 100, 98, 99), (100, 101, 99, 100),
        (98, 99, 97, 98), (99, 100, 98, 99), (97, 99, 96, 97),
        (97.5, 98, 97, 97.5), (99, 100, 98, 99), (99, 104, 98, 103),
    ]
    entry_long = [(103, 118, 102, 117)]
    flat_b = [(117.5, 119, 117, 118.5)] * 10
    short_pat = [
        (101, 102, 100, 101), (103, 104, 101, 103),
        (102.5, 103, 102, 102.5), (100, 102, 99, 100),
        (102, 103, 101, 102), (100.5, 102, 100, 100.5),
        (99, 101, 98, 99), (101, 102, 100, 101),
        (100, 101, 99, 100), (102, 103, 101, 102),
        (101, 102, 100, 101), (103, 104, 101, 103),
        (102.5, 103, 102, 102.5), (101, 102, 100, 101),
        (101, 102, 96, 97),
    ]
    entry_short = [(97, 98, 79, 80)]
    flat_c = [(82.5, 84, 82, 83.5)] * 5
    rows = flat_a + long_pat + entry_long + flat_b + short_pat \
        + entry_short + flat_c
    assert len(rows) == 72
    return _df(rows)


def bias_frame():
    flat = [(100, 101, 99, 100)]
    bull = [(95, 100, 90, 95), (90, 95, 85, 90), (100, 105, 95, 100),
            (95, 100, 90, 95), (90, 95, 85, 90), (105, 110, 100, 105),
            (100, 105, 95, 100), (95, 100, 90, 95), (110, 115, 105, 110),
            (105, 110, 100, 105), (100, 105, 95, 100)]
    bear = [(115, 120, 110, 115), (110, 115, 105, 110),
            (105, 110, 100, 105), (110, 115, 105, 110),
            (100, 105, 95, 100), (105, 110, 100, 105),
            (95, 100, 90, 95), (100, 105, 95, 100),
            (90, 95, 85, 90), (95, 100, 90, 95), (85, 90, 80, 85)]
    rows = flat * 25 + bull + flat * 15 + bear + flat * 10
    assert len(rows) == 72
    return _df(rows)


def zone_frame():
    flat = [(95, 105, 95, 105)]
    support = [(115, 120, 110, 115), (110, 115, 105, 110),
               (105, 110, 100, 105), (110, 115, 105, 110),
               (115, 120, 110, 115), (113, 118, 108, 113),
               (109, 114, 104, 109), (107, 112, 102, 107),
               (105, 110, 100, 105), (108, 113, 103, 108),
               (106, 111, 101, 106)]
    resistance = [(85, 90, 80, 85), (90, 95, 85, 90), (95, 100, 90, 95),
                  (90, 95, 85, 90), (85, 90, 80, 85), (87, 92, 82, 87),
                  (91, 96, 86, 91), (93, 98, 88, 93), (95, 100, 90, 95),
                  (92, 97, 87, 92), (94, 99, 89, 94)]
    rows = flat * 28 + support + flat * 15 + resistance + flat * 7
    assert len(rows) == 72
    return _df(rows)


def frames():
    return {"D1": bias_frame(), "H4": zone_frame(), "H1": trigger_frame()}


# -------------------------------------------------------------------- $12
def test_twelve_dollar_balance_and_risk_budget():
    assert PAPER_START_BALANCE_12 == 12.0
    res = run_mtf12_backtest(frames(), PRESET)
    assert res["start_balance"] == 12.0
    assert res["risk_money"] == pytest.approx(0.12)
    assert res["risk_pct"] == pytest.approx(0.01)


def test_r_accounting_on_known_outcome():
    res = run_mtf12_backtest(frames(), PRESET)
    trades = res["trades"]
    assert len(trades) == 2
    long, short = trades
    # directions reported separately — shorts are not hidden
    assert [t["direction"] for t in trades] == [1, -1]
    assert long["exit_reason"] == "target"
    assert short["exit_reason"] == "target"
    # R recomputed from each trade's own fill/stop/exit (cost in R)
    for t in trades:
        risk = abs(t["entry"] - t["stop"])
        expect = t["direction"] * (t["exit_price"] - t["entry"]) / risk \
            - 0.45 / risk
        assert t["r_net"] == pytest.approx(expect, abs=1e-9)
        assert t["fractional_units"] == pytest.approx(0.12 / risk)
        assert t["amd_phase"] == "unknown"
    # long: target = fill + 2R by construction -> exactly +2R gross
    # short: nearest pool (87) sits inside 2R, so the 2R floor applies
    # -> exactly +2R gross as well
    assert long["r_gross"] == pytest.approx(2.0, abs=1e-9)
    assert short["r_gross"] == pytest.approx(2.0, abs=1e-9)
    assert long["r_net"] == pytest.approx(2.0 - 0.45 / 7.26, abs=0.05)
    assert short["r_net"] == pytest.approx(2.0 - 0.45 / 7.35, abs=0.05)
    # aggregates tie out; dollar P&L is tiny, as it must be on $12
    assert res["summary"]["total_R"] == pytest.approx(
        sum(t["r_net"] for t in trades))
    assert res["summary"]["dollar_pnl"] == pytest.approx(
        res["summary"]["total_R"] * 0.12)
    assert res["by_direction"]["long"]["total_trades"] == 1
    assert res["by_direction"]["short"]["total_trades"] == 1
    assert res["by_direction"]["long"]["total_R"] == pytest.approx(
        long["r_net"])
    assert res["by_direction"]["short"]["total_R"] == pytest.approx(
        short["r_net"])
    # sessions from the UTC clock: +39h = 15:00 Overlap, +65h = 17:00 NY
    assert long["session"] == "Overlap"
    assert short["session"] == "NY"


def test_oanda_1unit_feasibility():
    # $20 stop on a $12 account: 1-oz risk $20 >> $0.12 budget
    assert oanda_1unit_feasible(20.0, 0.12) is False
    # small-stop scalp: 1-oz risk $0.10 fits the $0.12 budget
    assert oanda_1unit_feasible(0.10, 0.12) is True
    assert oanda_1unit_feasible(0.12, 0.12) is True  # boundary: fits
    assert oanda_1unit_feasible(0.0, 0.12) is False
    # end-to-end: both synthetic trades are far too wide for 1 unit
    res = run_mtf12_backtest(frames(), PRESET)
    assert all(t["oanda_1unit_feasible"] is False for t in res["trades"])
    assert res["feasible_1unit"]["total_trades"] == 0


def test_session_and_regime_labels():
    assert session_of(pd.Timestamp("2026-01-02 03:00", tz="UTC")) == "Asia"
    assert session_of(pd.Timestamp("2026-01-02 10:00", tz="UTC")) == "London"
    assert session_of(pd.Timestamp("2026-01-02 15:00", tz="UTC")) == "Overlap"
    assert session_of(pd.Timestamp("2026-01-02 19:00", tz="UTC")) == "NY"
    res = run_mtf12_backtest(frames(), PRESET)
    assert all(t["regime"] == "trend" for t in res["trades"])
    assert res["by_regime"]["trend"]["total_trades"] == 2
    assert res["by_regime"]["range"]["total_trades"] == 0


def test_resample_aggregates_closed_bars_causally():
    h1 = _df([(100, 101, 99, 100)] * 8 + [(102, 103, 101, 102)] * 4,
             start="2026-01-05 00:00")
    h4 = resample_bars(h1, "H4")
    assert len(h4) == 3  # 12 H1 bars -> 3 H4 bars
    # stamped at period CLOSE: first H4 bar closes 04:00, not 00:00
    assert h4.index[0] == pd.Timestamp("2026-01-05 04:00", tz="UTC")
    assert h4["high"].iloc[0] == 101.0
    assert h4["close"].iloc[-1] == 102.0


def test_build_frames_refuses_finer_tf():
    h1 = trigger_frame()
    with pytest.raises(ValueError, match="M15"):
        build_tf_frames({"H1": h1},
                        TimeframeConfig("H4", "H1", "M15", swing_n=3))
    # coarser frames derive fine from H1
    out = build_tf_frames({"H1": h1}, TimeframeConfig("D1", "H4", "H1",
                                                       swing_n=1))
    assert set(out) == {"H1", "D1", "H4"}
    assert len(out["H4"]) == 18  # 72 H1 bars -> 18 H4 bars
    assert len(out["D1"]) == 3


def test_report_mentions_feasible_subset_and_honesty():
    text = format_report(run_mtf12_backtest(frames(), PRESET))
    assert "OANDA 1-unit feasible" in text
    assert "Fractional-unit R measures STRATEGY quality" in text
    assert "amd_phase" in text or "AMD" in text


def test_no_ten_thousand_anywhere_in_new_code():
    repo = Path(__file__).resolve().parents[1]
    for rel in ("goldenswing/live/multitf.py",
                "goldenswing/backtest/mtf12.py"):
        src = (repo / rel).read_text()
        assert "10000" not in src, rel
        assert "10_000" not in src, rel
        assert "10,000" not in src, rel
