"""Regression tests for the DST-aware session -> timeframe mapping.

All timestamps are hand-computed against the zoneinfo tz database:

- Winter (northern hemisphere): EST = UTC-5, GMT = UTC+0, JST = UTC+9.
  The mapping reproduces the repo's legacy UTC buckets exactly:
  Asia 00-07, London 07-12, Overlap 12-16, NY 16-21, Off 21-24 UTC;
  London KZ 07-10 UTC, NY KZ 12-15 UTC.
- US DST 2026: springs forward Sun 2026-03-08 (EST -> EDT, UTC-4);
  falls back Sun 2026-11-01.  UK BST 2026: Sun 2026-03-29 -> Sun
  2026-10-25 (GMT -> BST, UTC+1).  Gap weeks (US on DST, UK not)
  run 2026-03-09..28 and 2026-10-26..31.
"""
from __future__ import annotations

import pandas as pd

from goldenswing.live.multitf import TimeframeConfig, evaluate
from goldenswing.reader.session_timeframes import (
    attach_session_context, compose, mapping_for, session_of,
    tag_sessions,
)

UTC = "UTC"


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


# ------------------------------------------------------- killzone mappings
def test_london_killzone_maps_to_m15():
    # 2026-03-06 08:00 UTC = 03:00 EST (London KZ 02-05 NY), 08:00 GMT
    # (London session: open 07:00 GMT, NY opens 12:00 UTC).
    m = mapping_for(ts("2026-03-06 08:00"))
    assert m.session == "london"
    assert m.killzone == "london_kz"
    assert m.in_killzone
    assert m.trigger_tf == "M15"
    assert m.entry_clock == "M15"
    assert m.bias_tf == "H4" and m.zone_tf == "H1"
    assert m.active_tfs == ("H4", "H1", "M15")


def test_ny_killzone_maps_to_m15():
    # 2026-03-06 13:00 UTC = 08:00 EST (NY KZ 07-10 NY); overlap
    # (NY open 12:00 UTC, London closes 16:00 GMT).
    m = mapping_for(ts("2026-03-06 13:00"))
    assert m.session == "overlap"
    assert m.killzone == "ny_kz"
    assert m.trigger_tf == "M15"
    assert m.entry_clock == "M15"
    assert m.active_tfs == ("H4", "H1", "M15")


def test_killzone_boundaries_are_half_open():
    # NY KZ is [07:00, 10:00) NY wall clock: 12:00 UTC winter is in,
    # 15:00 UTC winter (10:00 EST) is out; London KZ [02:00, 05:00):
    # 07:00 UTC winter (02:00 EST) in, 10:00 UTC (05:00 EST) out.
    assert session_of(ts("2026-03-06 12:00")) == ("overlap", "ny_kz")
    assert session_of(ts("2026-03-06 14:59")) == ("overlap", "ny_kz")
    assert session_of(ts("2026-03-06 15:00")) == ("overlap", None)
    assert session_of(ts("2026-03-06 07:00")) == ("london", "london_kz")
    assert session_of(ts("2026-03-06 09:59")) == ("london", "london_kz")
    assert session_of(ts("2026-03-06 10:00")) == ("london", None)


# ------------------------------------------- off-killzone / off-session
def test_asia_maps_to_m30():
    # 03:00 UTC = 12:00 JST (Asia session), 22:00 EST previous day.
    m = mapping_for(ts("2026-03-06 03:00"))
    assert m.session == "asia"
    assert m.killzone is None
    assert m.trigger_tf == "M30"
    assert m.active_tfs == ("H4", "H1", "M30")


def test_off_session_maps_to_m30():
    # 22:00 UTC = 17:00 EST — after NY close (16:00 EST), before Asia
    # open (09:00 JST = 00:00 UTC).
    m = mapping_for(ts("2026-03-06 22:00"))
    assert m.session == "off"
    assert m.killzone is None
    assert m.trigger_tf == "M30"
    assert m.active_tfs == ("H4", "H1", "M30")


def test_london_non_killzone_maps_to_m30():
    # 11:00 UTC = 06:00 EST — London open, NY not yet open, and 06:00
    # NY is in neither killzone window (02-05, 07-10).
    m = mapping_for(ts("2026-03-06 11:00"))
    assert m.session == "london"
    assert m.killzone is None
    assert m.trigger_tf == "M30"


def test_overlap_non_killzone_maps_to_m30():
    # 15:30 UTC = 10:30 EST — overlap (London still open till 16:00
    # GMT) but past the NY KZ end (10:00 EST).
    m = mapping_for(ts("2026-03-06 15:30"))
    assert m.session == "overlap"
    assert m.killzone is None
    assert m.trigger_tf == "M30"


def test_newyork_non_killzone_maps_to_m30():
    # 18:00 UTC = 13:00 EST — after London close (16:00 GMT), NY open
    # till 16:00 EST, well past the NY KZ.
    m = mapping_for(ts("2026-03-06 18:00"))
    assert m.session == "newyork"
    assert m.killzone is None
    assert m.trigger_tf == "M30"


# ------------------------------------------------------------- DST: US
def test_us_dst_boundary_same_wall_clock_maps_across_change():
    # The NY KZ is 07:00-10:00 NY WALL clock.  Before US DST
    # (2026-03-06, EST) that is 12:00-15:00 UTC; after (2026-03-09,
    # EDT) it is 11:00-14:00 UTC.  The same UTC hour flips:
    # 11:00 UTC is OUT (06:00 EST) before, IN (07:00 EDT) after;
    # 14:00 UTC is IN (09:00 EST) before, OUT (10:00 EDT) after.
    assert session_of(ts("2026-03-06 11:00"))[1] is None
    assert session_of(ts("2026-03-09 11:00")) == ("overlap", "ny_kz")
    assert session_of(ts("2026-03-06 14:00")) == ("overlap", "ny_kz")
    assert session_of(ts("2026-03-09 14:00"))[1] is None
    # London KZ likewise: 09:00 UTC is 04:00 EST (in) before DST,
    # 05:00 EDT (out — end-exclusive) after.
    assert session_of(ts("2026-03-06 09:00"))[1] == "london_kz"
    assert session_of(ts("2026-03-09 09:00"))[1] is None
    # And the trigger follows the killzone, not the UTC hour:
    assert mapping_for(ts("2026-03-09 11:00")).trigger_tf == "M15"
    assert mapping_for(ts("2026-03-09 14:00")).trigger_tf == "M30"


def test_us_dst_fall_back():
    # Sun 2026-11-01: clocks go back EDT -> EST.  11:00 UTC on
    # 2026-10-30 (EDT) is 07:00 EDT — NY KZ; 11:00 UTC on 2026-11-02
    # (EST) is 06:00 EST — not.  12:00 UTC flips the other way.
    assert session_of(ts("2026-10-30 11:00")) == ("overlap", "ny_kz")
    assert session_of(ts("2026-11-02 11:00"))[1] is None
    assert session_of(ts("2026-11-02 12:00")) == ("overlap", "ny_kz")


# ------------------------------------------------------------- DST: UK
def test_uk_dst_boundary_same_utc_hour_changes_session():
    # London opens 07:00 London wall clock: 06:30 UTC is 06:30 GMT
    # (still Asia) on 2026-03-27, but 07:30 BST (London open) on
    # 2026-03-30, after the UK springs forward (Sun 2026-03-29).
    assert session_of(ts("2026-03-27 06:30"))[0] == "asia"
    assert session_of(ts("2026-03-30 06:30"))[0] == "london"
    # London closes 16:00 London: 15:30 UTC is 15:30 GMT (overlap)
    # before, 16:30 BST (newyork — London closed) after.
    assert session_of(ts("2026-03-27 15:30"))[0] == "overlap"
    assert session_of(ts("2026-03-30 15:30"))[0] == "newyork"


def test_gap_week_us_on_dst_uk_not():
    # 2026-03-13: US on EDT, UK still GMT.  NY opens 07:00 EDT =
    # 11:00 UTC, London closes 16:00 GMT = 16:00 UTC: overlap is
    # 11:00-16:00 UTC.  London opens 07:00 GMT = 07:00 UTC, so
    # 06:30 UTC is still Asia.
    assert session_of(ts("2026-03-13 06:30"))[0] == "asia"
    assert session_of(ts("2026-03-13 11:30"))[0] == "overlap"
    assert session_of(ts("2026-03-13 15:30"))[0] == "overlap"


# ------------------------------------------------- never flips bias, never a veto
def test_session_mapping_never_flips_bias():
    # The mapping carries NO direction: bias_tf is H4 in every
    # session/killzone, and the mapping dict has no direction/bias-sign
    # key at all.  A session change can therefore never flip bias.
    samples = [
        "2026-03-06 03:00",  # asia
        "2026-03-06 08:00",  # london KZ
        "2026-03-06 11:00",  # london
        "2026-03-06 13:00",  # overlap NY KZ
        "2026-03-06 15:30",  # overlap
        "2026-03-06 18:00",  # newyork
        "2026-03-06 22:00",  # off
        "2026-03-09 11:00",  # overlap NY KZ, summer (EDT)
        "2026-03-30 06:30",  # london, summer (BST)
    ]
    for s in samples:
        m = mapping_for(ts(s))
        assert m.bias_tf == "H4"
        assert m.zone_tf == "H1"
        d = m.as_dict()
        assert "direction" not in d
        assert "bias" not in d  # no bias-sign field (bias_tf is a TF)
        assert "veto" not in d and "blocked" not in d
        # The trigger clock varies, but only between the two entry
        # clocks — never into a bias/zone clock.
        assert m.trigger_tf in ("M15", "M30")


def test_amd_phase_is_carried_not_changed():
    # Same instant, with and without an AMD phase: session, killzone,
    # and clocks are identical — the session never re-derives phase.
    base = mapping_for(ts("2026-03-06 08:00"))
    for phase in ("accumulation", "manipulation", "expansion_up",
                  "trend_down", "unknown"):
        m = mapping_for(ts("2026-03-06 08:00"), amd_phase=phase)
        assert (m.session, m.killzone, m.trigger_tf) == \
            (base.session, base.killzone, base.trigger_tf)
        assert m.amd_phase == phase
        assert f"amd:{phase}" in m.tags
    # Manipulation inside a killzone earns a descriptive tag only.
    m = mapping_for(ts("2026-03-06 08:00"), amd_phase="manipulation")
    assert "manipulation_in_killzone" in m.tags
    m2 = mapping_for(ts("2026-03-06 03:00"), amd_phase="manipulation")
    assert "manipulation_in_killzone" not in m2.tags


def test_qm_keeps_its_own_m5_clock():
    # QM M5 composes: the session trigger stays M15/M30 (the session's
    # own clock is not rewritten), but the playbook's entry clock
    # stays M5 — session annotates, never overrides QM's clock.
    m = mapping_for(ts("2026-03-06 08:00"), playbook="qm_m5")
    assert m.trigger_tf == "M15"
    assert m.entry_clock == "M5"
    assert m.active_tfs == ("H4", "H1", "M5")
    assert "qm_m5 keeps M5 clock; session is annotation only" in m.tags
    m_off = mapping_for(ts("2026-03-06 03:00"), playbook="qm_m5")
    assert m_off.trigger_tf == "M30"
    assert m_off.entry_clock == "M5"


def test_attach_session_context_is_annotation_only():
    class Sig:
        direction = 1
        score = 60.0
        entry_ref = 4200.0
        invalidation = 4180.0
        targets = [4240.0]
        live_context = {"touch_index": 31}

    sig = Sig()
    out = attach_session_context(sig, ts("2026-03-06 08:00"),
                                 amd_phase="manipulation",
                                 playbook="qm_m5")
    assert out is sig
    assert sig.direction == 1 and sig.score == 60.0
    assert sig.entry_ref == 4200.0 and sig.invalidation == 4180.0
    assert sig.targets == [4240.0]
    assert sig.live_context["touch_index"] == 31  # preserved
    sess = sig.live_context["session"]
    assert sess["session"] == "london"
    assert sess["killzone"] == "london_kz"
    assert sess["entry_clock"] == "M5"  # QM clock, not session trigger
    assert "manipulation_in_killzone" in sess["tags"]


# ------------------------------------------------------------- purity / misc
def test_pure_and_naive_equals_utc():
    a = mapping_for(ts("2026-03-06 08:00"))
    b = mapping_for(ts("2026-03-06 08:00"))
    assert a == b  # pure: same input, same answer, no hidden state
    naive = mapping_for(pd.Timestamp("2026-03-06 08:00"))  # naive == UTC
    assert naive == a
    # compose() is the same mapping in JSON-safe form.
    d = compose(ts("2026-03-06 08:00"), amd_phase="manipulation")
    assert d["session"] == "london" and d["trigger_tf"] == "M15"
    assert d["amd_phase"] == "manipulation"
    assert "manipulation_in_killzone" in d["tags"]
    assert d["bias_tf"] == "H4" and d["active_tfs"] == ["H4", "H1", "M15"]


def test_tag_sessions_index():
    idx = pd.DatetimeIndex(["2026-03-06 03:00", "2026-03-06 08:00",
                            "2026-03-06 13:00"], tz="UTC")
    names, kzs = tag_sessions(idx)
    assert names == ["asia", "london", "overlap"]
    assert kzs == [None, "london_kz", "ny_kz"]


# ------------------------------------------------------- engine wiring
def _frames():
    highs = [100, 95, 105, 100, 95, 110, 105, 100, 115, 110, 105]
    lows = [90, 85, 95, 90, 85, 100, 95, 90, 105, 100, 95]
    bias = pd.DataFrame(
        {"open": [(h + l) / 2 for h, l in zip(highs, lows)],
         "high": highs, "low": lows,
         "close": [(h + l) / 2 for h, l in zip(highs, lows)]},
        index=pd.date_range("2026-01-01", periods=len(highs), freq="h",
                            tz="UTC"))
    zone = pd.DataFrame(
        {"open": [110.0] * 11, "high": [115.0] * 11,
         "low": [105.0] * 11, "close": [110.0] * 11},
        index=pd.date_range("2026-01-01", periods=11, freq="h",
                            tz="UTC"))
    trig = pd.DataFrame(
        {"open": [100.0] * 40, "high": [101.0] * 40,
         "low": [99.0] * 40, "close": [100.0] * 40},
        index=pd.date_range("2026-01-01", periods=40, freq="h",
                            tz="UTC"))
    return {"D1": bias, "H4": zone, "H1": trig}


def test_evaluate_asof_annotates_without_changing_gates():
    cfg = TimeframeConfig("D1", "H4", "H1", swing_n=1)
    plain = evaluate(_frames(), cfg)
    assert "session" not in plain  # no asof: zero behaviour change
    annotated = evaluate(_frames(), cfg, asof=ts("2026-03-06 08:00"))
    # Every gate field identical — the session key is pure annotation.
    for key in plain:
        assert annotated[key] == plain[key]
    sess = annotated["session"]
    assert sess["session"] == "london"
    assert sess["killzone"] == "london_kz"
    assert sess["entry_clock"] == "M15"
    assert sess["active_tfs"] == ["H4", "H1", "M15"]
