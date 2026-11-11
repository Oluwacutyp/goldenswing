"""TFState momentum + AMD-phase fields (SPEC §3.1 — the §9
ship remainder: "momentum/AMD-phase TFState fields —
build_tf_state carries no such keys").

Pins:
- momentum is a MEASUREMENT from the TF's own bars:
  velocity_atr is the ATR-normalised close velocity over
  VELOCITY_BARS (live/trend_momentum's construction, parity-
  pinned against multitf.atr14), banded from that number;
  a rising series reads positive, a falling one negative, a
  flat one exactly zero / "flat"; displacement is the latest
  bar's body-dominated close or None; divergence is the
  features/divergence family's read or its measured-inactive
  form;
- absent stays absent: no bars / too few bars -> velocity
  and AMD label are None WITH a reason in words, never 0.00,
  never a fabricated neutral, never a defaulted phase;
- AMD phase is STRUCTURAL (never an EMA read): a range-bound
  base -> accumulation; a sweep through the base extreme
  that closes back inside (upthrust / spring) ->
  manipulation; an expansion beyond the base range with a
  displacement-scale move (incl. an ongoing trend) ->
  distribution; a measured frame with no AMD signature ->
  undetermined WITH the measured facts as the reason;
- every TF the builder builds carries both fields, JSON-
  native (the journal is JSONL);
- the /status render (execution/situation_read) carries both
  fields per TF and renders the reason when absent /
  undetermined.
"""
import json

import numpy as np
import pandas as pd
import pytest

from goldenswing.execution.situation_read import render_situation_read
from goldenswing.live import multitf, situation
from goldenswing.live.situation import build_situation, build_tf_state
from goldenswing.live.trend_momentum import VELOCITY_BARS
from tests.test_situation import TF_FREQ, flat_frame, full_stack, \
    stack_of, trend_frame


# ------------------------------------------------------------------
# frame crafting
# ------------------------------------------------------------------
def _frame(closes, *, highs=None, lows=None, opens=None, freq="h"):
    idx = pd.date_range("2026-01-01", periods=len(closes),
                        freq=freq, tz="UTC")
    c = np.array(closes, dtype=float)
    o = np.array(opens, dtype=float) if opens is not None else c - 0.2
    h = np.array(highs, dtype=float) if highs is not None else c + 0.5
    lo = np.array(lows, dtype=float) if lows is not None else c - 0.5
    return pd.DataFrame(
        {"open": o, "high": h, "low": lo, "close": c,
         "volume": 100.0}, index=idx)


def _monotonic(drift=2.0, n=60):
    closes = [2000.0 + drift * i for i in range(n)]
    opens = [c - (0.3 if drift > 0 else -0.3) for c in closes]
    return _frame(closes, highs=[c + 0.7 for c in closes],
                  lows=[c - 0.7 for c in closes], opens=opens)


def _range_frame(n=58):
    """Range-bound oscillation: closes inside 1997–2003, bar
    extremes inside 1992–2008 — a base, on any reading."""
    closes = [2000.0 + 3.0 * np.sin(i * 0.9) for i in range(n)]
    return _frame(closes, highs=[c + 5 for c in closes],
                  lows=[c - 5 for c in closes])


def _sweep_frame(*, side="high"):
    """The range base + ONE recent bar that sweeps through a
    base extreme and closes back inside (the manipulation
    signature: features/phases' sweep-through-the-extreme
    definition), price still inside afterwards."""
    base = _range_frame()
    closes = list(base["close"])
    highs = list(base["high"])
    lows = list(base["low"])
    if side == "high":
        closes[-5] = 2001.0
        highs[-5] = 2016.0  # ~8 beyond the base high, close inside
        lows[-5] = 1999.0
    else:
        closes[-4] = 1999.0
        lows[-4] = 1984.0  # ~8 beyond the base low, close inside
        highs[-4] = 2001.0
    return _frame(closes, highs=highs, lows=lows)


def _round_trip_frame():
    """An inverted-V round trip inside the AMD window: +60
    over 24 bars, -60 back over 24 — too wide for a base, no
    range break at the last close, no sweep: no AMD
    signature."""
    closes = [2000.0 + 2.5 * i for i in range(25)]
    closes += [2060.0 - 2.5 * i for i in range(1, 25)]
    return _frame(closes, highs=[c + 1.5 for c in closes],
                  lows=[c - 1.5 for c in closes])


# ------------------------------------------------------------------
# momentum: measured from bars
# ------------------------------------------------------------------
def test_momentum_velocity_parity_with_the_schema_construction():
    frame = _monotonic(2.0)
    mom = build_tf_state("H1", frame)["momentum"]
    atr = multitf.atr14(frame)
    expected = (float(frame["close"].iloc[-1])
                - float(frame["close"].iloc[-1 - VELOCITY_BARS])) \
        / (VELOCITY_BARS * atr)
    assert mom["velocity_atr"] == pytest.approx(expected)
    assert mom["velocity_atr"] > 0
    assert mom["band"] == "strong_bull"
    assert mom["reason"] is None


def test_momentum_falling_series_reads_negative():
    mom = build_tf_state("H1", _monotonic(-2.0))["momentum"]
    assert mom["velocity_atr"] < 0
    assert mom["band"] == "strong_bear"


def test_momentum_rising_sawtooth_reads_positive_banded():
    mom = build_tf_state("H1", trend_frame(120))["momentum"]
    assert mom["velocity_atr"] > 0
    assert mom["band"] in ("bull", "strong_bull")


def test_momentum_flat_series_is_zero_and_flat():
    mom = build_tf_state("H1", flat_frame(120))["momentum"]
    assert mom["velocity_atr"] == 0.0
    assert mom["band"] == "flat"


def test_momentum_displacement_latest_bar_body_dominated():
    closes = [2000.0] * 40 + [2010.0]
    opens = [2000.0] * 40 + [2000.0]
    frame = _frame(closes, opens=opens,
                   highs=[2000.5] * 40 + [2010.5],
                   lows=[1999.5] * 40 + [1999.5])
    mom = build_tf_state("H1", frame)["momentum"]
    assert mom["displacement"] == "bull"
    # A bar that is NOT body-dominated carries no displacement.
    calm = build_tf_state("H1", flat_frame(60))["momentum"]
    assert calm["displacement"] is None


def test_momentum_divergence_contract():
    # Too short for two confirmed pivots: absent (None) — the
    # mean-reversion family's evidence read degrades to None.
    short = build_tf_state("H1", trend_frame(12))["momentum"]
    assert short["divergence"] is None
    # A long trend frame: the family RAN — a measured-inactive
    # dict (or a live one), never a fabricated active read.
    full = build_tf_state("H1", trend_frame(120))["momentum"]
    div = full["divergence"]
    assert isinstance(div, dict)
    assert set(div) == {"active", "direction", "family"}
    if div["active"]:
        assert div["direction"] in (-1, 1)
        assert div["family"]
    else:
        assert div == {"active": False, "direction": 0,
                       "family": None}


def test_momentum_absent_stays_absent_with_a_reason():
    empty = build_tf_state("H1", None)
    assert empty["momentum"]["velocity_atr"] is None
    assert empty["momentum"]["band"] is None
    assert "no bars" in empty["momentum"]["reason"]
    short = build_tf_state("H1", trend_frame(5))
    assert short["momentum"]["velocity_atr"] is None
    assert "insufficient bars (5)" in short["momentum"]["reason"]


# ------------------------------------------------------------------
# AMD phase: structural classification
# ------------------------------------------------------------------
def test_amd_range_bound_base_is_accumulation():
    phase = build_tf_state("H1", _range_frame())["phase"]
    assert phase["label"] == "accumulation"
    assert phase["direction"] == 0
    assert phase["range_high"] is not None
    assert phase["range_low"] is not None


def test_amd_flat_base_is_accumulation():
    phase = build_tf_state("H1", flat_frame(120))["phase"]
    assert phase["label"] == "accumulation"


def test_amd_sweep_and_retrace_is_manipulation_upthrust():
    phase = build_tf_state("H1", _sweep_frame(side="high"))["phase"]
    assert phase["label"] == "manipulation"
    assert phase["direction"] == -1
    assert "sweep above the base range high" in phase["reason"]
    assert "closed back inside" in phase["reason"]


def test_amd_sweep_and_retrace_is_manipulation_spring():
    phase = build_tf_state("H1", _sweep_frame(side="low"))["phase"]
    assert phase["label"] == "manipulation"
    assert phase["direction"] == 1
    assert "sweep below the base range low" in phase["reason"]


def test_amd_expansion_trend_is_distribution():
    up = build_tf_state("H1", trend_frame(120))["phase"]
    assert up["label"] == "distribution"
    assert up["direction"] == 1
    down = build_tf_state("H1", trend_frame(120, drift=-0.8))["phase"]
    assert down["label"] == "distribution"
    assert down["direction"] == -1
    mono = build_tf_state("H1", _monotonic(2.0))["phase"]
    assert mono["label"] == "distribution"


def test_amd_indeterminate_is_undetermined_with_a_reason():
    phase = build_tf_state("H1", _round_trip_frame())["phase"]
    assert phase["label"] == "undetermined"
    assert phase["direction"] is None
    assert "no AMD signature measured" in phase["reason"]
    # NEVER defaulted to a label the bars did not measure.
    assert phase["label"] != "accumulation"


def test_amd_absent_stays_absent_with_a_reason():
    empty = build_tf_state("H1", None)
    assert empty["phase"]["label"] is None
    assert "no bars" in empty["phase"]["reason"]
    short = build_tf_state("H1", trend_frame(12))
    assert short["phase"]["label"] is None
    assert "insufficient bars (12)" in short["phase"]["reason"]


# ------------------------------------------------------------------
# every TF carries both fields; records stay JSON-native
# ------------------------------------------------------------------
@pytest.mark.parametrize("tf", situation.SITUATION_TFS)
def test_every_tf_state_carries_both_fields(tf):
    state = build_tf_state(tf, trend_frame(120, freq=TF_FREQ[tf]))
    assert state["momentum"]["velocity_atr"] is not None
    assert state["momentum"]["band"] is not None
    assert state["phase"]["label"] in (
        "accumulation", "manipulation", "distribution",
        "undetermined")
    json.dumps(state)  # the journal is JSONL: must not raise


def test_full_situation_carries_both_fields_per_tf():
    built = build_situation(full_stack(), price=2136.5)
    for tf in situation.SITUATION_TFS:
        state = built["per_tf"][tf]
        assert state["momentum"]["velocity_atr"] is not None
        assert state["phase"]["label"] == "distribution"
    json.dumps(built)


# ------------------------------------------------------------------
# the /status render
# ------------------------------------------------------------------
def test_render_carries_momentum_and_phase_per_tf():
    text = render_situation_read(
        build_situation(full_stack(), price=2136.5))
    lines = [ln for ln in text.splitlines()
             if ln.startswith("  H1:")]
    assert len(lines) == 1
    assert "momentum +0.07 bull" in lines[0]
    assert "phase distribution ↑" in lines[0]


def test_render_undetermined_phase_carries_its_reason():
    frames = {tf: trend_frame(120, freq=TF_FREQ[tf])
              for tf in situation.SITUATION_TFS}
    frames["H1"] = _round_trip_frame()
    sit = build_situation(stack_of(**frames), price=2000.0)
    lines = [ln for ln in render_situation_read(sit).splitlines()
             if ln.startswith("  H1:")]
    assert len(lines) == 1
    assert "phase undetermined (no AMD signature measured" in lines[0]


def test_render_unmeasured_fields_say_so_in_words():
    # A ready-length frame with zero range on every bar: no
    # ATR exists, so neither field can be measured — the line
    # says why, and never prints a fabricated number.
    idx = pd.date_range("2026-01-01", periods=60, freq="h",
                        tz="UTC")
    frozen = pd.DataFrame(
        {"open": 2000.0, "high": 2000.0, "low": 2000.0,
         "close": 2000.0, "volume": 100.0}, index=idx)
    frames = {tf: trend_frame(120, freq=TF_FREQ[tf])
              for tf in situation.SITUATION_TFS}
    frames["H1"] = frozen
    sit = build_situation(stack_of(**frames), price=2000.0)
    lines = [ln for ln in render_situation_read(sit).splitlines()
             if ln.startswith("  H1:")]
    assert len(lines) == 1
    assert "momentum unmeasured (no ATR measured" in lines[0]
    assert "phase unmeasured (no ATR measured" in lines[0]
    assert "0.00" not in lines[0]
