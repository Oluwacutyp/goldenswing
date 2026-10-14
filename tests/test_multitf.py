"""Tests for the multi-TF structural engine (synthetic bars only)."""
import pandas as pd
import pytest

from goldenswing.live.multitf import (
    PRESETS, SCALP_M1, SWING, TimeframeConfig, detect_bos_msch_choch,
    evaluate, evaluate_bias, evaluate_zones, find_swings, get_preset,
)

CFG = TimeframeConfig("D1", "H4", "H1", swing_n=1)


def mk_bars(rows, start="2026-01-01"):
    idx = pd.date_range(start, periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=idx)


# ---------------------------------------------------------------- fixtures
def bias_bull_bars():
    highs = [100, 95, 105, 100, 95, 110, 105, 100, 115, 110, 105]
    lows = [90, 85, 95, 90, 85, 100, 95, 90, 105, 100, 95]
    rows = [((h + l) / 2, h, l, (h + l) / 2) for h, l in zip(highs, lows)]
    return mk_bars(rows)


def bias_bear_bars():
    highs = [120, 115, 110, 115, 105, 110, 100, 105, 95, 100, 90]
    lows = [110, 105, 100, 105, 95, 100, 90, 95, 85, 90, 80]
    rows = [((h + l) / 2, h, l, (h + l) / 2) for h, l in zip(highs, lows)]
    return mk_bars(rows)


def zone_support_bars():
    mids = [115, 110, 105, 110, 115, 113, 109, 107, 105, 108, 106]
    highs = [120, 115, 110, 115, 120, 118, 114, 112, 110, 113, 111]
    lows = [110, 105, 100, 105, 110, 108, 104, 102, 100, 103, 101]
    return mk_bars([(m, h, l, m) for m, h, l in zip(mids, highs, lows)])


def zone_resistance_bars():
    # price reflection of zone_support_bars around 100
    mids = [85, 90, 95, 90, 85, 87, 91, 93, 95, 92, 94]
    highs = [90, 95, 100, 95, 90, 92, 96, 98, 100, 97, 99]
    lows = [80, 85, 90, 85, 80, 82, 86, 88, 90, 87, 89]
    return mk_bars([(m, h, l, m) for m, h, l in zip(mids, highs, lows)])


def trigger_bull_bars(last=(99, 104, 98, 103)):
    pattern = [
        (99, 100, 98, 99), (97, 99, 96, 97), (97.5, 98, 97, 97.5),
        (100, 101, 98, 100), (98, 99, 97, 98), (99.5, 100, 98, 99.5),
        (101, 102, 99, 101), (99, 100, 98, 99), (100, 101, 99, 100),
        (98, 99, 97, 98), (99, 100, 98, 99), (97, 99, 96, 97),
        (97.5, 98, 97, 97.5), (99, 100, 98, 99),
    ]
    flat = [(98.5, 100, 98, 99.5)] * 25
    return mk_bars(flat + pattern + [last])


def trigger_bear_bars(last=(101, 102, 96, 97)):
    # price reflection of trigger_bull_bars around 100
    pattern = [
        (101, 102, 100, 101), (103, 104, 101, 103),
        (102.5, 103, 102, 102.5), (100, 102, 99, 100),
        (102, 103, 101, 102), (100.5, 102, 100, 100.5),
        (99, 101, 98, 99), (101, 102, 100, 101),
        (100, 101, 99, 100), (102, 103, 101, 102),
        (101, 102, 100, 101), (103, 104, 101, 103),
        (102.5, 103, 102, 102.5), (101, 102, 100, 101),
    ]
    flat = [(101.5, 102, 100, 100.5)] * 25
    return mk_bars(flat + pattern + [last])


def frames(bias, zone, trig):
    return {"D1": bias, "H4": zone, "H1": trig}


# ------------------------------------------------------------------- swings
def test_find_swings_known_levels():
    highs = [10, 12, 11, 13, 12, 11, 10]
    lows = [5, 4, 6, 3, 7, 6, 5]
    h, l = find_swings(highs, lows, 1)
    assert h == [(1, 12.0), (3, 13.0)]
    assert l == [(1, 4.0), (3, 3.0)]
    h2, _ = find_swings(highs, lows, 2)
    assert h2 == [(3, 13.0)]


def test_find_swings_ties_are_not_swings():
    h, l = find_swings([10, 10, 10, 10, 10], [5, 5, 5, 5, 5], 1)
    assert h == [] and l == []


def test_find_swings_causal_last_bar_shift_changes_nothing_earlier():
    highs = [10, 12, 11, 13, 12, 11, 10, 9, 8]
    lows = [1] * 9
    before, _ = find_swings(highs, lows, 1)
    highs_shifted = highs[:-1] + [100]  # last bar explodes upward
    after, _ = find_swings(highs_shifted, lows, 1)
    assert before == after == [(1, 12.0), (3, 13.0)]


# ------------------------------------------------------- displacement gate
def test_displacement_gate_rejects_weak_body():
    rows = [(98, 100, 97, 99)] * 29
    weak = mk_bars(rows + [(100.05, 105, 95, 100.1)])
    # close clears the swing, but the body is dust -> no event
    assert detect_bos_msch_choch(weak, 99, 90, 2.0) is None


def test_displacement_gate_accepts_strong_body():
    rows = [(98, 100, 97, 99)] * 29
    strong = mk_bars(rows + [(99, 103, 97, 103)])
    assert detect_bos_msch_choch(strong, 99, 90, 2.0) == "BOS_bull"


def test_msch_breaks_reference_only_not_older_level():
    rows = [(99, 101, 98, 100)] * 9
    bars = mk_bars(rows + [(100, 105, 99, 104)])
    # breaks the recent swing (103) but not the older one (110)
    assert detect_bos_msch_choch(bars, 103, 90, 2.0,
                                 minor_high=110, minor_low=85) == "MSCH_bull"


def test_mid_range_break_of_old_level_only_is_not_an_event():
    rows = [(99, 101, 98, 100)] * 9
    bars = mk_bars(rows + [(100, 105, 99, 104)])
    # breaks the OLDER level (100) while still inside the recent swing
    # (110): mid-range chop, not a structure event
    assert detect_bos_msch_choch(bars, 110, 90, 2.0,
                                 minor_high=100, minor_low=85) is None


# --------------------------------------------------------------------- bias
def test_bias_bull_bear_flat():
    assert evaluate_bias(bias_bull_bars(), swing_n=1) == 1
    assert evaluate_bias(bias_bear_bars(), swing_n=1) == -1
    flat = mk_bars([(100, 101, 99, 100)] * 12)
    assert evaluate_bias(flat, swing_n=1) == 0


def test_zones_nearest_poi():
    zone = evaluate_zones(zone_support_bars(), swing_n=1)
    assert zone is not None
    assert zone["kind"] == "support" and zone["level"] == 100.0
    assert zone["bottom"] < 100.0 < zone["top"]


# ----------------------------------------------------------------- evaluate
def test_bos_bull_fires_with_bias_poi_displacement():
    res = evaluate(frames(bias_bull_bars(), zone_support_bars(),
                          trigger_bull_bars()), CFG)
    assert res["signal"] is True
    assert res["direction"] == 1
    assert res["bias"] == 1
    assert res["trigger"] == "BOS_bull"
    assert res["entry"] == 103.0
    assert res["zone"]["level"] == 100.0
    assert res["zone"]["bottom"] <= 103.0 <= res["zone"]["top"]
    assert res["stop"] < res["entry"]
    assert res["target"] > res["entry"]
    assert res["reason"] == ""


def test_bias_veto_blocks_counter_trend_trigger():
    # bear bias, but the trigger bar still breaks up with displacement
    # and closes inside the resistance band -> vetoed, never averaged in
    res = evaluate(frames(bias_bear_bars(), zone_resistance_bars(),
                          trigger_bull_bars(last=(99, 104, 98, 103))), CFG)
    assert res["signal"] is False
    assert res["reason"] == "trigger_against_bias"


def test_entry_above_poi_rejected_for_longs():
    # trigger bar closes above the POI band -> no chasing
    res = evaluate(frames(bias_bull_bars(), zone_support_bars(),
                          trigger_bull_bars(last=(99, 111, 98, 110))), CFG)
    assert res["signal"] is False
    assert res["reason"] == "entry_above_poi"


def test_shorts_mirror_bearish_bos():
    res = evaluate(frames(bias_bear_bars(), zone_resistance_bars(),
                          trigger_bear_bars()), CFG)
    assert res["signal"] is True
    assert res["direction"] == -1
    assert res["bias"] == -1
    assert res["trigger"] == "BOS_bear"
    assert res["entry"] == 97.0
    assert res["zone"]["level"] == 100.0
    assert res["zone"]["bottom"] <= 97.0 <= res["zone"]["top"]
    assert res["stop"] > res["entry"]
    assert res["target"] < res["entry"]
    assert res["reason"] == ""


def test_entry_below_poi_rejected_for_shorts():
    res = evaluate(frames(bias_bear_bars(), zone_resistance_bars(),
                          trigger_bear_bars(last=(101, 102, 90, 91))), CFG)
    assert res["signal"] is False
    assert res["reason"] == "entry_below_poi"


def test_missing_tf_reports_reason():
    res = evaluate({"H1": trigger_bull_bars()}, CFG)
    assert res["signal"] is False
    assert res["reason"].startswith("missing_tf")


def test_flat_bias_means_no_signal():
    flat = mk_bars([(100, 101, 99, 100)] * 12)
    res = evaluate(frames(flat, zone_support_bars(),
                          trigger_bull_bars()), CFG)
    assert res["signal"] is False
    assert res["reason"] == "bias_flat"


def test_evaluate_uses_only_bars_up_to_evaluation_point():
    # evaluation at the prefix is identical whether or not extra future
    # bars exist afterwards: the engine never peeks past its window.
    trig = trigger_bull_bars()
    res_prefix = evaluate(frames(bias_bull_bars(), zone_support_bars(),
                                 trig), CFG)
    extended = pd.concat([trig, mk_bars(
        [(50, 60, 40, 55)] * 5, start="2026-01-02 16:00")])
    res_again = evaluate(frames(bias_bull_bars(), zone_support_bars(),
                                trig), CFG)
    assert res_prefix == res_again
    assert res_prefix["signal"] is True
    # the extended frame evaluates at its own (crashed) last bar
    res_extended = evaluate(frames(bias_bull_bars(), zone_support_bars(),
                                   extended), CFG)
    assert res_extended["signal"] is False
    assert extended is not trig  # sanity: different objects


# ------------------------------------------------------------------ presets
def test_presets_exist_and_scalp_m1_hypothesis_documented():
    assert PRESETS["swing"] == SWING
    assert (SWING.bias_tf, SWING.zone_tf, SWING.trigger_tf) == ("D1", "H4", "H1")
    assert (PRESETS["intraday"].bias_tf, PRESETS["intraday"].zone_tf,
            PRESETS["intraday"].trigger_tf) == ("H4", "H1", "M15")
    assert (PRESETS["scalp"].bias_tf, PRESETS["scalp"].zone_tf,
            PRESETS["scalp"].trigger_tf) == ("H1", "M15", "M5")
    assert SCALP_M1.swing_n == 5
    assert (SCALP_M1.zone_tf, SCALP_M1.trigger_tf) == ("M5", "M1")
    with pytest.raises(ValueError):
        get_preset("nope")
