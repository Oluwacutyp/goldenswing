"""Tests for goldenswing/features/traps.py — trap-intelligence features.

Unit tests use small synthetic OHLCV frames with hand-computed
expectations.  The calm template drifts highs down / lows up by 0.001
per bar so no accidental fractal pivots form (pivot ties would); only
the engineered swing bars become confirmed pivots.

The regression test compares every contract column against the GOD-CE
port (``goldenswing.godce.stateful.zones_traps_features``) on two years
of H1 gold — godce is imported ONLY here, never in the module.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from goldenswing.features.traps import COLUMNS, TrapParams, compute_trap_features


def _frame(n, overrides=None, volumes=None, start="2024-01-01"):
    """n hourly UTC bars; calm(i) = (100, 102-0.001i, 100.5+0.001i, 101).

    The drift guarantees no accidental swing pivots or FVGs on calm
    bars.  ``overrides``: {bar: (o,h,l,c)}.  ``volumes``: {bar: vol}.
    """
    rows = [(100.0, 102.0 - 0.001 * i, 100.5 + 0.001 * i, 101.0)
            for i in range(n)]
    for i, r in (overrides or {}).items():
        rows[i] = r
    vols = [1000.0] * n
    for i, v in (volumes or {}).items():
        vols[i] = v
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": vols},
        index=idx,
    )


def _bull_sfp_frame():
    """Swing highs 110 @12 / 110.3 @24 (EQH pair, both with full pivot
    windows); raid @40 (16:00 UTC, outside the Asian window so the
    engineered flag must come from the EQH clause) wicks through 110.3
    and reclaims below on 1.3x volume."""
    return _frame(70, overrides={
        12: (100, 110, 100.5, 101),
        24: (100, 110.3, 100.5, 101),
        40: (110, 111.5, 104, 109.5),
    }, volumes={40: 1300.0})


# ── contract ─────────────────────────────────────────────────────────

def test_contract_columns_order_dtype_index():
    df = _frame(40)
    out = compute_trap_features(df)
    assert list(out.columns) == [
        "trap_boost_long", "trap_boost_short",
        "bait_long", "bait_short",
        "lure_long", "lure_short",
        "sfp_bull_active", "sfp_bear_active",
        "sfp_bull_eng", "sfp_bear_eng",
        "fake_brk_up", "fake_brk_dn",
        "beyond_bull", "beyond_bear",
        "beyond_bull_full", "beyond_bear_full",
        "ifvg_demand", "ifvg_supply",
    ]
    assert list(out.columns) == list(COLUMNS)
    assert len(out) == len(df)
    assert out.index.equals(df.index)
    assert all(str(dt) == "bool" for dt in out.dtypes)


def test_params_defaults_match_contract():
    p = TrapParams()
    assert (p.swing_len, p.atr_len, p.vol_len) == (10, 14, 20)
    assert p.trap_vol_mult == 1.15
    assert p.trap_memory_bars == 12
    assert p.brk_lookback == 6
    assert p.fvg_min_pct == 20.0
    assert p.fvg_atr_scale == 0.75
    assert p.zone_max_age == 120
    assert p.max_inv_fvgs == 4
    assert p.use_trap_engine and p.use_beyond_fvg and p.use_inv_fvg


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        compute_trap_features(pd.DataFrame({"close": [1.0, 2.0]}))


def test_short_frame_no_crash():
    out = compute_trap_features(_frame(5))
    assert out.shape == (5, 18)
    assert not out.any().any()


def test_calm_frame_all_quiet():
    out = compute_trap_features(_frame(40))
    assert not out.any().any()


# ── 1) swing failure patterns ────────────────────────────────────────

def test_sfp_bull_fires_on_raid_bar_with_eng_and_memory():
    out = compute_trap_features(_bull_sfp_frame())
    # confirmed pivots: PH[22]=110, PH[34]=110.3 (EQH pair) → raid @40
    assert not out["sfp_bull_active"].iloc[39]
    assert out["sfp_bull_active"].iloc[40]
    # engineered: raid within 0.35 ATR of the EQH pool (the Asian and
    # prior-day clauses provably miss on this frame)
    assert out["sfp_bull_eng"].iloc[40]
    # memory: active for trap_memory_bars=12 bars after the raid
    assert out["sfp_bull_active"].iloc[52]
    assert not out["sfp_bull_active"].iloc[53]
    assert not out["sfp_bull_eng"].iloc[53]
    # direction semantics: bulls trapped → bearish context
    assert out["trap_boost_short"].iloc[40]
    assert not out["trap_boost_long"].iloc[40]
    assert not out["sfp_bear_active"].any()


def test_sfp_bear_mirror():
    df = _frame(70, overrides={
        12: (100, 102, 90, 101),
        40: (90, 96, 89, 90.5),
    }, volumes={40: 1300.0})
    out = compute_trap_features(df)
    assert out["sfp_bear_active"].iloc[40]
    assert not out["sfp_bear_active"].iloc[39]
    # bears trapped → bullish context
    assert out["trap_boost_long"].iloc[40]
    assert not out["trap_boost_short"].iloc[40]
    # the raid bar's reclaim close (90.5) sits above the swing low (90),
    # so no breakdown was ever armed — the memory is purely the SFP latch
    assert not out["fake_brk_dn"].any()
    assert out["sfp_bear_active"].iloc[52]   # 40 + trap_memory_bars
    assert not out["sfp_bear_active"].iloc[53]


def test_use_trap_engine_off_kills_sfp_fake_lure():
    out = compute_trap_features(
        _bull_sfp_frame(), TrapParams(use_trap_engine=False))
    for c in ("sfp_bull_active", "sfp_bear_active", "sfp_bull_eng",
              "sfp_bear_eng", "fake_brk_up", "fake_brk_dn",
              "lure_long", "lure_short"):
        assert not out[c].any(), c
    assert not out["trap_boost_short"].any()


# ── 2) fake breakouts ────────────────────────────────────────────────

def test_fake_breakout_up_fires_on_failed_reclaim():
    df = _frame(60, overrides={
        12: (100, 110, 100.5, 101),
        30: (109.5, 111, 109, 110.5),   # close-break of 110, low volume
        31: (110.2, 111, 110, 110.3),   # holds above
        32: (110, 110.5, 109, 109.8),   # fails back inside within 6 bars
    }, volumes={30: 500.0})
    out = compute_trap_features(df)
    # suspicious breakout: close through swing high on below-avg volume
    assert out["lure_long"].iloc[30]
    assert not out["lure_long"].iloc[29]
    assert not out["fake_brk_up"].iloc[30]
    assert not out["fake_brk_up"].iloc[31]
    assert out["fake_brk_up"].iloc[32]
    # a failed upside breakout traps the bulls → bearish memory
    assert not out["sfp_bull_active"].iloc[31]
    assert out["sfp_bull_active"].iloc[32]
    assert out["trap_boost_short"].iloc[32]


def test_breakout_that_holds_never_fakes():
    df = _frame(60, overrides={
        12: (100, 110, 100.5, 101),
        30: (109.5, 111, 109, 110.5),
        31: (110.2, 111, 110, 110.3),
        32: (110.3, 111, 110, 110.5),
        33: (110.5, 111, 110, 110.6),
        34: (110.5, 111, 110, 110.6),
        35: (110.5, 111, 110, 110.6),
        36: (110.5, 111, 110, 110.6),
        37: (110.5, 111, 110, 110.6),
    }, volumes={30: 500.0})
    out = compute_trap_features(df)
    assert out["lure_long"].iloc[30]          # the break was still sus
    assert not out["fake_brk_up"].any()       # but it held → no fake


def test_fake_breakout_dn_mirror():
    df = _frame(60, overrides={
        12: (100, 102, 92, 101),
        30: (90.5, 91, 89, 89.5),         # close-break of 92, low volume
        31: (89.8, 90.5, 89, 89.7),       # holds below
        32: (91, 93, 90, 92.5),           # fails back inside within 6 bars
    }, volumes={30: 500.0})
    out = compute_trap_features(df)
    assert out["lure_short"].iloc[30]
    assert not out["lure_short"].iloc[29]
    assert not out["fake_brk_dn"].iloc[31]
    assert out["fake_brk_dn"].iloc[32]
    assert not out["sfp_bear_active"].iloc[31]
    assert out["sfp_bear_active"].iloc[32]
    assert out["trap_boost_long"].iloc[32]


# ── 3) beyond-FVG ────────────────────────────────────────────────────

def _beyond_bull_frame():
    # bull FVG {top:102, bot:100} formed @20; swing high 108 @12 keeps
    # the reclaim bar from tripping the breakout machine
    return _frame(60, overrides={
        12: (100, 108, 100.5, 101),
        18: (99, 100, 97, 99),
        20: (101, 104, 102, 103),
        22: (100.5, 101.5, 100.2, 101),   # parked inside the gap
        25: (102, 105, 99, 103),           # one-bar raid-and-reclaim
        26: (100.5, 101.5, 100.2, 101),
    })


def test_beyond_bull_direct_0_to_2_with_full_reclaim():
    out = compute_trap_features(_beyond_bull_frame())
    assert not out["beyond_bull"].iloc[24]
    assert out["beyond_bull"].iloc[25]
    assert out["beyond_bull_full"].iloc[25]    # closed through the far side
    assert not out["beyond_bear"].iloc[25]
    assert not out["beyond_bear"].any()
    # state persists for trap_memory_bars then expires
    assert out["beyond_bull"].iloc[37]
    assert not out["beyond_bull"].iloc[38]
    assert out["trap_boost_long"].iloc[25]
    assert not out["trap_boost_short"].iloc[25]


def test_beyond_bull_0_1_2_sequence_partial_reclaim():
    df = _frame(60, overrides={
        12: (100, 108, 100.5, 101),
        18: (99, 100, 97, 99),
        20: (101, 104, 102, 103),
        25: (99, 101, 98, 99),      # sweep: close through the gap bottom
        26: (100, 103, 99, 101),    # reclaim back above the bottom only
    })
    out = compute_trap_features(df)
    assert not out["beyond_bull"].iloc[25]     # state 1: swept, not reclaimed
    assert out["beyond_bull"].iloc[26]        # state 2: reclaimed
    assert not out["beyond_bull_full"].iloc[26]  # never cleared the top
    assert not out["beyond_bull_full"].any()
    assert out["trap_boost_long"].iloc[26]


def test_beyond_bear_direct_0_to_2_with_full_reclaim():
    # the depressed bars 21-24 keep price from gapping back over the
    # gap bar's high (which would print an accidental bull FVG)
    df = _frame(60, overrides={
        12: (100, 102, 92, 101),
        18: (101, 103, 100.5, 101),
        20: (100, 99.9, 99.5, 99.7),   # bear gap {top:100.5, bot:99.9}
        21: (99.8, 100.6, 99.4, 99.6),
        22: (99.8, 100.6, 99.4, 99.6),
        23: (99.8, 100.6, 99.4, 99.6),
        24: (99.8, 100.6, 99.4, 99.6),
        25: (98, 101, 97, 97),         # raid through the top, reclaim below
    })
    out = compute_trap_features(df)
    assert out["beyond_bear"].iloc[25]
    assert out["beyond_bear_full"].iloc[25]
    assert not out["beyond_bull"].any()
    assert out["trap_boost_short"].iloc[25]
    assert not out["trap_boost_long"].iloc[25]


# ── 4) inside-gap bait ───────────────────────────────────────────────

def test_bait_long_while_parked_inside_unreclaimed_gap():
    out = compute_trap_features(_beyond_bull_frame())
    assert out["bait_long"].iloc[22]       # inside the gap, beyond not active
    assert out["bait_short"].iloc[22] == False
    assert not out["bait_short"].any()
    # after the reclaim the same parking spot is no longer bait
    assert out["bait_long"].iloc[26] == False


# ── 5) inverted FVG ──────────────────────────────────────────────────

def test_ifvg_supply_flip_test_kill():
    df = _frame(60, overrides={
        12: (100, 108, 100.5, 101),
        18: (99, 100, 97, 99),
        20: (101, 104, 102, 103),     # bull gap {102,100}
        25: (98, 100, 97, 99),       # fails → flips to supply
        26: (100, 101, 99.5, 100),
        27: (100, 101, 99.5, 100),
        28: (100, 101, 99.5, 100),
        29: (100, 101, 99.5, 100),
        30: (101.5, 101, 100.2, 101),  # rallies into the flipped zone
        31: (102, 104, 101, 103),      # closes through the top → killed
    })
    out = compute_trap_features(df)
    assert not out["ifvg_supply"].iloc[25]  # flip bar: close > open, no test
    assert out["ifvg_supply"].iloc[26]
    assert out["ifvg_supply"].iloc[30]
    assert not out["ifvg_supply"].iloc[31]
    assert not out["ifvg_supply"].iloc[32]
    assert not out["ifvg_demand"].any()
    assert out["trap_boost_short"].iloc[30]


def test_ifvg_demand_flip_test_kill():
    # depressed bars 21-24: no accidental bull FVG on the snap-back,
    # no second bear FVG off bar 19's low
    df = _frame(60, overrides={
        12: (100, 102, 92, 101),
        18: (101, 103, 100.5, 101),
        20: (100, 99.9, 99.5, 99.7),   # bear gap {top:100.5, bot:99.9}
        21: (99.8, 100.6, 99.4, 99.6),
        22: (99.8, 100.6, 99.4, 99.6),
        23: (99.8, 100.6, 99.4, 99.6),
        24: (99.8, 100.6, 99.4, 99.6),
        25: (102, 103, 100, 101),      # fails → flips to demand
        26: (100, 100.3, 99.8, 100.1),
        27: (100, 100.3, 99.8, 100.1),
        28: (100, 100.3, 99.8, 100.1),
        29: (100, 100.3, 99.8, 100.1),
        30: (99.9, 100.2, 99.7, 100.0),  # dips into the flipped zone
        31: (99.8, 100.0, 99.5, 99.6),   # closes through the bottom → killed
    })
    out = compute_trap_features(df)
    assert not out["ifvg_demand"].iloc[25]   # flip bar: close < open, no test
    assert out["ifvg_demand"].iloc[26]
    assert out["ifvg_demand"].iloc[30]
    assert not out["ifvg_demand"].iloc[31]
    assert not out["ifvg_supply"].any()
    assert out["trap_boost_long"].iloc[30]


# ── regression vs the GOD-CE port ────────────────────────────────────

def test_regression_vs_godce():
    """Every contract column must match the godce trap layer on 2y of H1
    gold (≥99% per-column agreement).

    Mapping (contract → godce ``zones_traps_features`` output):

    - sfp_bull_active/bear_active/eng → bullTrapActive / bearTrapActive /
      bullTrapWasEng / bearTrapWasEng
    - fake_brk_up/dn → fakeBrkUp / fakeBrkDn
    - lure_long/short → brkUpSus / brkDnSus
    - beyond_bull/bear(_full) → bfgState==2 / rfgState==2 (/ bfgFull / rfgFull)
    - ifvg_demand/supply → ifvgDemTest / ifvgSupTest
    - bait_long/short → baitLong / baitShort
    - trap_boost_long → bearTrapActive | (bfgState==2) | ifvgDemTest
    - trap_boost_short → bullTrapActive | (rfgState==2) | ifvgSupTest

    Deliberate deviations: none — the engineered-liquidity check keeps
    the pine's fourth clause (prior-week high/low) alongside the three
    named in the workstream spec, because the source port has it.
    """
    from goldenswing.godce.backtest import build_features, load_ohlc
    from goldenswing.godce.params import GodceParams

    root = Path(__file__).resolve().parents[1]
    df = load_ohlc(str(root / "data" / "GCF_H1_2y.csv"), "America/New_York")
    dxy = load_ohlc(str(root / "data" / "DOLLAR_H1_2y.csv"), "UTC")
    _, feat, _ = build_features(df, dxy, GodceParams())

    mine = compute_trap_features(df)
    assert mine.index.equals(feat.index)

    bfg2 = feat["bfgState"].to_numpy() == 2
    rfg2 = feat["rfgState"].to_numpy() == 2
    mapping = {
        "sfp_bull_active": feat["bullTrapActive"].to_numpy(),
        "sfp_bear_active": feat["bearTrapActive"].to_numpy(),
        "sfp_bull_eng": feat["bullTrapWasEng"].to_numpy(),
        "sfp_bear_eng": feat["bearTrapWasEng"].to_numpy(),
        "fake_brk_up": feat["fakeBrkUp"].to_numpy(),
        "fake_brk_dn": feat["fakeBrkDn"].to_numpy(),
        "lure_long": feat["brkUpSus"].to_numpy(),
        "lure_short": feat["brkDnSus"].to_numpy(),
        "beyond_bull": bfg2,
        "beyond_bear": rfg2,
        "beyond_bull_full": feat["bfgFull"].to_numpy(),
        "beyond_bear_full": feat["rfgFull"].to_numpy(),
        "ifvg_demand": feat["ifvgDemTest"].to_numpy(),
        "ifvg_supply": feat["ifvgSupTest"].to_numpy(),
        "bait_long": feat["baitLong"].to_numpy(),
        "bait_short": feat["baitShort"].to_numpy(),
        "trap_boost_long": (feat["bearTrapActive"].to_numpy() | bfg2
                            | feat["ifvgDemTest"].to_numpy()),
        "trap_boost_short": (feat["bullTrapActive"].to_numpy() | rfg2
                             | feat["ifvgSupTest"].to_numpy()),
    }
    assert set(mapping) == set(COLUMNS)

    failures = []
    for col, want in mapping.items():
        got = mine[col].to_numpy()
        agree = float((got == want).mean())
        if agree < 0.99:
            bad = np.nonzero(got != want)[0]
            failures.append(
                f"{col}: {agree:.4f} agreement "
                f"({len(bad)} mismatches, first bars {bad[:5].tolist()})")
    assert not failures, "trap columns diverging from godce:\n" + "\n".join(failures)
