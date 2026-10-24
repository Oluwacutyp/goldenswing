"""Census-wiring pass (owner directive, Runtime Truth item 4
remainder): the detector families the Phase 2 census found
implemented-but-unwired — features/patterns chart patterns,
live/patterns premium_discount + breaker/mitigation, features
relstrength, revival engulfing — plus the BUILT named
candlestick taxonomy (live/candle_taxonomy.py), are consumed
on the live MT5 path as RECORDED EVIDENCE on funnel evaluation
records and in the /assess render.

Pins:
- the taxonomy's named patterns fire on their textbook
  geometry (and ONLY there), trend-named where the canon
  requires context, and are causal (truncation-invariant);
- the census consumer's snapshot carries every wired section
  with truthful recorded/unavailable states (incl. the
  positional-index seam the legacy block helpers needed);
- live evaluation records carry evidence.census, their
  assessment markers FLIP to the recorded reads, and the
  rendered assessment shows the census section;
- warmup evaluations record an explicit not_recorded census;
- the wiring is observation-only: decisions are identical
  with the census consumer disabled (no-consumer regression).
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.execution.funnel import assessment_block
from goldenswing.live.candle_taxonomy import (
    PATTERN_NAMES, detect_named_patterns, patterns_at)
from goldenswing.live.census_evidence import (
    census_snapshot, census_unavailable)

from test_funnel_m5 import _fresh_tick
from test_p3_trader import (_FakeScore, _StubPB, _drive_one,
                            _frame, _runner, _sig)


# ----------------------------------------------------------- fixtures

def _rich_h1():
    """Deterministic rich H1: rally to a double top (~2060),
    decline, a hammer at the low, then a bullish engulfing
    pair (mirrors the pass drive fixture)."""
    anchors = [(0, 2000.0), (10, 2060.0), (15, 2040.0),
               (20, 2061.0), (45, 1990.0), (55, 2005.0),
               (70, 1960.0), (79, 1992.0)]
    closes = []
    for (i0, v0), (i1, v1) in zip(anchors, anchors[1:]):
        for i in range(i0, i1):
            closes.append(v0 + (v1 - v0) * (i - i0) / (i1 - i0))
    closes.append(anchors[-1][1])
    rows = []
    prev = closes[0]
    for c in closes:
        o = prev
        rows.append((o, max(o, c) + 0.8, min(o, c) - 0.8, c))
        prev = c
    rows[71] = (1960.0, 1965.0, 1938.0, 1964.0)   # hammer
    rows[72] = (1963.0, 1964.0, 1956.0, 1958.0)   # small bearish
    rows[73] = (1957.0, 1968.0, 1956.5, 1966.0)   # engulfing up
    return _frame(rows)


def _names_at(df, i):
    """Pattern names completing at bar i of df (full detector)."""
    return [p["name"] for p in detect_named_patterns(
        df.iloc[:i + 1], window=1)]


def _drift(n, start, step):
    rows = []
    px = start
    for _ in range(n):
        o, c = px, px + step
        rows.append((o, max(o, c) + 0.6, min(o, c) - 0.6, c))
        px += step
    return rows


# ------------------------------------------------------ taxonomy

def test_taxonomy_hammer_after_decline():
    rows = _drift(16, 126.0, -2.0)          # closes 126 -> 94
    rows.append((94.0, 96.0, 86.0, 95.5))   # long lower wick
    df = _frame(rows)
    hits = [(p["name"], p["direction"])
            for p in detect_named_patterns(df, window=1)]
    assert ("hammer", 1) in hits


def test_taxonomy_hanging_man_same_shape_after_advance():
    rows = _drift(16, 90.0, 1.0)            # closes 90 -> 105
    rows.append((106.0, 107.5, 99.0, 107.0))  # same pin shape
    df = _frame(rows)
    hits = detect_named_patterns(df, window=1)
    assert ("hanging_man", -1) in \
        [(p["name"], p["direction"]) for p in hits]
    assert "hammer" not in [p["name"] for p in hits]


def test_taxonomy_shooting_star_and_inverted_hammer():
    rows = _drift(16, 90.0, 1.0)
    rows.append((106.0, 114.0, 105.5, 108.0))  # long upper wick
    hits = detect_named_patterns(_frame(rows), window=1)
    assert ("shooting_star", -1) in \
        [(p["name"], p["direction"]) for p in hits]
    rows = _drift(16, 110.0, -1.0)
    rows.append((94.0, 102.0, 93.5, 96.0))     # same, in decline
    hits = detect_named_patterns(_frame(rows), window=1)
    assert ("inverted_hammer", 1) in \
        [(p["name"], p["direction"]) for p in hits]


def test_taxonomy_doji_family():
    base = _drift(16, 100.0, 0.0)
    # plain doji: tiny body, one long wick + one short wick
    # (neither subtype share met, long-legged share missed)
    df = _frame(base + [(100.0, 102.8, 99.0, 100.2)])
    assert _names_at(df, len(df) - 1) == ["doji"]
    # dragonfly: body at the top, long lower wick
    df = _frame(base + [(100.0, 100.4, 94.0, 100.2)])
    assert _names_at(df, len(df) - 1) == ["dragonfly_doji"]
    # gravestone: body at the bottom, long upper wick
    df = _frame(base + [(100.0, 106.0, 99.6, 99.8)])
    assert _names_at(df, len(df) - 1) == ["gravestone_doji"]
    # long-legged: both wicks long
    df = _frame(base + [(100.0, 104.0, 96.0, 100.2)])
    assert _names_at(df, len(df) - 1) == ["long_legged_doji"]


def test_taxonomy_engulfing_by_name():
    rows = _drift(10, 100.0, 0.0)
    rows += [(100.0, 100.6, 97.4, 98.0),      # bearish bar
             (97.6, 101.2, 97.4, 100.8)]      # engulfs it upward
    df = _frame(rows)
    assert _names_at(df, len(df) - 1) == ["bullish_engulfing"]
    rows = _drift(10, 100.0, 0.0)
    rows += [(100.0, 102.6, 99.4, 102.0),     # bullish bar
             (102.4, 102.6, 98.8, 99.2)]      # engulfs it downward
    df = _frame(rows)
    assert _names_at(df, len(df) - 1) == ["bearish_engulfing"]


def test_taxonomy_piercing_and_dark_cloud():
    rows = _drift(10, 100.0, 0.0)
    rows += [(100.0, 100.4, 93.6, 94.0),      # strong bearish
             (93.8, 98.4, 93.6, 98.0)]        # opens low, closes
    # past the midpoint (97.0), short of the open (100)
    df = _frame(rows)
    assert "piercing_line" in _names_at(df, len(df) - 1)
    rows = _drift(10, 100.0, 0.0)
    rows += [(100.0, 106.4, 99.6, 106.0),     # strong bullish
             (106.2, 106.4, 101.6, 102.0)]    # opens high, closes
    # below the midpoint (103.0), above the open (100)
    df = _frame(rows)
    assert "dark_cloud_cover" in _names_at(df, len(df) - 1)


def test_taxonomy_morning_and_evening_star():
    rows = _drift(10, 100.0, 0.0)
    rows += [(100.0, 100.5, 93.5, 94.0),      # strong bearish
             (93.8, 94.6, 92.8, 93.4),        # small star bar
             (93.6, 98.5, 93.4, 98.0)]        # confirms past mid
    df = _frame(rows)
    assert "morning_star" in _names_at(df, len(df) - 1)
    rows = _drift(10, 100.0, 0.0)
    rows += [(100.0, 106.5, 99.5, 106.0),     # strong bullish
             (106.2, 107.2, 105.4, 106.6),    # small star bar
             (106.4, 106.6, 101.5, 102.0)]    # confirms past mid
    df = _frame(rows)
    assert "evening_star" in _names_at(df, len(df) - 1)


def test_taxonomy_names_are_from_the_declared_set():
    for p in detect_named_patterns(_rich_h1(), window=80):
        assert p["name"] in PATTERN_NAMES


def test_taxonomy_is_causal_under_truncation():
    """A pattern completing at bar i is a function of bars
    <= i: truncating the frame at i cannot change it."""
    df = _rich_h1()
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    from goldenswing.reader.indicators import atr as atr_fn
    atr_full = atr_fn(df).to_numpy(dtype=float)
    for i in range(2, len(df)):
        full = patterns_at(o, h, l, c, float(atr_full[i]), i)
        sub = df.iloc[:i + 1]
        atr_sub = atr_fn(sub).to_numpy(dtype=float)
        trunc = patterns_at(
            sub["open"].to_numpy(dtype=float),
            sub["high"].to_numpy(dtype=float),
            sub["low"].to_numpy(dtype=float),
            sub["close"].to_numpy(dtype=float),
            float(atr_sub[i]), i)
        assert full == trunc, f"bar {i}: {full} != {trunc}"


# ------------------------------------------------------ consumer

def test_snapshot_rich_fixture_records_every_section():
    rich = _rich_h1()
    snap = census_snapshot(frame=rich, tf="H1", h1_frame=rich,
                             dxy_frame=None)
    assert snap["status"] == "recorded"
    assert snap["frame"] == "H1"
    cp = snap["chart_patterns"]
    assert cp["status"] == "recorded"
    assert any(p["kind"] == "double_top" for p in cp["active"])
    pd_read = snap["premium_discount"]
    assert pd_read["status"] == "recorded"
    assert pd_read["zone"] in ("premium", "discount")
    assert pd_read["eq"] is not None
    assert snap["breaker_blocks"]["status"] == "recorded"
    assert snap["mitigation_blocks"]["status"] == "recorded"
    names = [p["name"] for p in snap["candles"]["taxonomy"]]
    assert "hammer" in names
    assert "bullish_engulfing" in names
    assert snap["candles"]["engulfing"]["direction"] in (-1, 0, 1)
    # No DXY frame in scope: RS unavailable, WITH its reason —
    # never a fabricated zero.
    rs = snap["relative_strength"]
    assert rs["status"] == "unavailable"
    assert rs["reason"]
    assert rs["dollar"]["available"] is False


def test_snapshot_blocks_run_on_datetime_indexed_frames():
    """Regression pin for the seam the wiring needed: the
    legacy block helpers index bars positionally and raise on
    a live datetime-indexed frame (int(Timestamp)); the
    consumer feeds them a positional view of the same bars."""
    rich = _rich_h1()
    assert isinstance(rich.index, pd.DatetimeIndex)
    snap = census_snapshot(frame=rich, tf="H1", h1_frame=rich)
    assert snap["breaker_blocks"]["count"] >= 1
    freshest = snap["breaker_blocks"]["freshest"][0]
    assert freshest["bar_ts"] and "T" not in str(freshest["bar_ts"])


def test_snapshot_relative_strength_with_dxy_peer():
    rich = _rich_h1()
    # DXY rising steadily on the same timestamps.
    dxy_rows = [(100.0 + 0.1 * i, 100.3 + 0.1 * i,
                 99.7 + 0.1 * i, 100.1 + 0.1 * i)
                for i in range(len(rich))]
    dxy = _frame(dxy_rows)
    dxy.index = rich.index
    snap = census_snapshot(frame=rich, tf="H1", h1_frame=rich,
                             dxy_frame=dxy)
    rs = snap["relative_strength"]
    assert rs["status"] == "recorded"
    assert rs["dollar"]["available"] is True
    assert rs["dollar"]["trend"] in (-1, 0, 1)
    assert rs["dollar"]["percentile"] is not None
    assert rs["dollar"]["ratio"] > 0
    # Silver has no live feed: unavailable per-peer, honestly.
    assert rs["silver"]["available"] is False
    # Deterministic: the same inputs give the same snapshot.
    again = census_snapshot(frame=rich, tf="H1", h1_frame=rich,
                            dxy_frame=dxy)
    assert again["relative_strength"] == rs


def test_snapshot_without_frame_is_unavailable():
    snap = census_snapshot(frame=None, tf="H1")
    assert snap["status"] == "unavailable"
    assert snap["reason"]
    nr = census_unavailable("some reason")
    assert nr == {"status": "not_recorded", "reason": "some reason"}


# ------------------------------------------------------ funnel

def test_assessment_markers_flip_with_census():
    rich = _rich_h1()
    snap = census_snapshot(frame=rich, tf="H1", h1_frame=rich,
                             dxy_frame=None)
    block = assessment_block(
        outcome="no_setup", evidence={"census": snap})
    marks = block["capability_markers"]
    assert marks["named_candlestick_taxonomy"].startswith("recorded")
    assert marks["premium_discount"].startswith("recorded")
    assert marks["legacy_smc_block"].startswith("partially_integrated")
    assert marks["chart_patterns"].startswith("recorded")
    assert marks["relative_strength"].startswith(
        "recorded as unavailable")
    assert marks["engulfing"].startswith("recorded")


def test_assessment_markers_without_census_state_the_wiring():
    block = assessment_block(outcome="no_setup", evidence={})
    marks = block["capability_markers"]
    # Post-wiring census facts: integrated as recorded
    # evidence, never the retired absent/not_integrated texts.
    assert marks["named_candlestick_taxonomy"].startswith("integrated")
    assert marks["premium_discount"].startswith("integrated")
    assert marks["legacy_smc_block"].startswith("partially_integrated")
    for key in ("named_candlestick_taxonomy", "premium_discount",
                "legacy_smc_block", "chart_patterns",
                "relative_strength", "engulfing"):
        text = marks[key]
        assert "no live consumer" not in text
        assert not text.startswith("absent")
        assert not text.startswith("not_integrated")


def test_assessment_markers_not_recorded_census():
    block = assessment_block(
        outcome="no_setup",
        evidence={"census": census_unavailable("warmup reason")})
    marks = block["capability_markers"]
    assert marks["premium_discount"].startswith("not_recorded")
    assert "warmup reason" in marks["premium_discount"]


# ------------------------------------------------------ runner e2e

def _trigger_record(runner):
    recs = [r for r in runner.journal.records()
            if r.get("type") == "evaluation"
            and r.get("evaluation_clock") == "h1_trigger"
            and r.get("outcome") == "candidate_emitted"]
    assert recs, "the drive must emit a trigger-path candidate"
    return recs[-1]


def _drive_rich(tmp_path, census_off=False):
    runner = _runner(tmp_path, execute_playbooks=("fvg_midline",))
    rich = _rich_h1()
    runner._stack = SimpleNamespace(frames={"H1": rich})
    runner._score_for = lambda **kw: _FakeScore(70.0, 25.0, 15.0)
    runner._last_tick = _fresh_tick(rich, mid=1992.0)
    if census_off:
        runner._census_for_record = lambda **kw: None
    _drive_one(runner, _StubPB(
        "fvg_midline", _sig(entry=1992.0, stop=1980.0,
                            index=len(rich) - 1)), rich)
    return runner


def test_live_record_carries_census_and_render_shows_it(tmp_path):
    runner = _drive_rich(tmp_path)
    rec = _trigger_record(runner)
    census = rec["evidence"]["census"]
    assert census["status"] == "recorded"
    assert census["chart_patterns"]["status"] == "recorded"
    marks = rec["assessment"]["capability_markers"]
    assert marks["named_candlestick_taxonomy"].startswith("recorded")
    assert marks["premium_discount"].startswith("recorded")
    text = rec["assessment_text"]
    assert "Wired census evidence" in text
    assert "double_top" in text
    assert "hammer" in text
    assert "named_candlestick_taxonomy: recorded" in text


def test_warmup_records_mark_census_not_recorded(tmp_path):
    runner = _runner(tmp_path)
    rich = _rich_h1()
    runner._stack = SimpleNamespace(frames={"H1": rich})
    runner._emit_eval(playbook="fvg_midline",
                      evaluation_clock="h1_close",
                      intended_tf="H1", actual_input_tf="H1",
                      outcome="no_setup", source="warmup_scan")
    rec = [r for r in runner.journal.records()
           if r.get("type") == "evaluation"][-1]
    assert rec["evidence"]["census"]["status"] == "not_recorded"
    assert rec["assessment"]["capability_markers"][
        "premium_discount"].startswith("not_recorded")
    assert "not recorded — warmup/replay" in \
        rec["assessment_text"]


def test_census_wiring_changes_no_decision(tmp_path):
    """No-consumer regression pin: the identical drive with
    the census consumer disabled produces the identical
    decision projection and the identical order count — the
    wiring is observation-only by construction."""
    def projection(runner):
        out = []
        for r in runner.journal.records():
            if r.get("type") != "evaluation":
                continue
            dec = r.get("decision") or {}
            cand = r.get("candidate") or {}
            out.append({
                "playbook": r.get("playbook"),
                "clock": r.get("evaluation_clock"),
                "outcome": r.get("outcome"),
                "reason_code": r.get("reason_code"),
                "candidate": {k: cand.get(k) for k in (
                    "direction", "entry", "stop", "score",
                    "asymmetry_score")},
                "decision": {k: dec.get(k) for k in (
                    "result", "reason_code", "order_result",
                    "execution_eligible", "floor_result",
                    "analytical_verdict", "execution")}})
        return out

    on = _drive_rich(tmp_path / "on")
    off = _drive_rich(tmp_path / "off", census_off=True)
    assert projection(on) == projection(off)
    assert len(on.executor.orders) == len(off.executor.orders) == 1
    # And the off-drive really carried no census (the pin is
    # not vacuous).
    rec_off = _trigger_record(off)
    assert "census" not in (rec_off.get("evidence") or {})
