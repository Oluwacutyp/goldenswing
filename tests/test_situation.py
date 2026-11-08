"""MarketSituation Pass 0 (SPEC §3 + §9 Pass 0; AMENDMENTS-1 §A4.3).

Pins:
- per-TF state extraction is a MEASUREMENT built from the
  canonical organs — bias/event parity with multitf primitives on
  identical bars, so the situation can never disagree with the
  hierarchy's own math;
- NO verdict enters any state field: multitf.evaluate() /
  hierarchy.evaluate_hierarchy() are never consulted (behavioral
  pin via monkeypatched raisers + a source-level pin);
- readiness honesty: no data -> "not_ready", short frame ->
  "collecting", full frame -> "ready"; absent fields stay None;
- keeper cadence: H1 close -> exactly one situation_snapshot;
  other TF closes -> situation_delta ONLY on a real change;
  situation_ids chain via prev_situation_id;
- recovery: a fresh keeper on the same directory wakes with the
  last situation and continues the sequence; reconstruct(id)
  rebuilds what was known at any past id (A4.3);
- forecasts: the outlook is carried verbatim, or None;
- runner wiring: funnel records of a cycle carry the cycle's
  situation_id; with situation_enabled=False nothing is stamped
  and nothing is journaled.
"""
import json

import numpy as np
import pandas as pd
import pytest

from goldenswing.live import multitf, situation
from goldenswing.live.hierarchy import TimeframeStack
from goldenswing.live.situation import (
    DELTA_KIND, SNAPSHOT_KIND, SituationKeeper, build_situation,
    build_tf_state)

TF_FREQ = {"D1": "D", "H4": "4h", "H1": "h", "M30": "30min",
           "M15": "15min", "M5": "5min"}


# ------------------------------------------------------------------
# frame crafting
# ------------------------------------------------------------------
def trend_frame(n=120, *, freq="h", start="2026-01-01", base=2000.0,
                drift=0.8, amp=6.0, period=12):
    """Deterministic sawtooth trend: rising (drift>0) or falling
    base with periodic pullbacks — confirmed HH/HL (or LH/LL)
    swings for find_swings/evaluate_bias to measure."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    closes = np.array([
        base + drift * i
        + amp * (1.0 - abs((i % period) / (period / 2.0) - 1.0))
        for i in range(n)], dtype=float)
    # Highs/lows ride the close path itself: deriving them from
    # open (= prior close) would tie the peak bar's high with the
    # next bar's, and ties are never swings.
    opens = closes - 0.3
    highs = closes + 0.7
    lows = closes - 0.7
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": 100.0}, index=idx)


def flat_frame(n=120, *, freq="h", start="2026-01-01", base=2000.0):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    closes = np.full(n, base)
    return pd.DataFrame(
        {"open": closes, "high": closes + 0.5, "low": closes - 0.5,
         "close": closes, "volume": 100.0}, index=idx)


def stack_of(**frames):
    return TimeframeStack.from_frames(frames)


def full_stack(*, drift=0.8):
    return stack_of(**{tf: trend_frame(120, freq=TF_FREQ[tf],
                                       drift=drift)
                       for tf in situation.SITUATION_TFS})


# ------------------------------------------------------------------
# per-TF state extraction
# ------------------------------------------------------------------
@pytest.mark.parametrize("tf", situation.SITUATION_TFS)
def test_state_parity_with_multitf_primitives(tf):
    """The situation may never disagree with the hierarchy's own
    math on identical bars (SPEC §9 Pass 0 evidence)."""
    frame = trend_frame(120, freq=TF_FREQ[tf])
    state = build_tf_state(tf, frame)
    assert state["bias"] == multitf.evaluate_bias(frame)
    assert state["bias"] == 1  # the crafted trend is bullish
    highs, lows = multitf.find_swings(frame["high"], frame["low"], 3)
    assert state["swings"]["confirmed_high"]["price"] == highs[-1][1]
    assert state["swings"]["confirmed_low"]["price"] == lows[-1][1]
    raw = multitf.detect_bos_msch_choch(
        frame, highs, lows, multitf.atr14(frame),
        trend=state["bias"], minor_high=highs[:-1] or None,
        minor_low=lows[:-1] or None)
    event = state["last_structure_event"]
    if raw is None:
        assert event is None
    else:
        kind, _, side = raw.partition("_")
        assert event["kind"] == kind
        assert event["direction"] == (1 if side == "bull" else -1)


def test_state_bearish_frame():
    frame = trend_frame(120, drift=-0.8)
    state = build_tf_state("H1", frame)
    assert state["bias"] == -1
    assert state["bias"] == multitf.evaluate_bias(frame)


def test_state_readiness_words():
    empty = build_tf_state("H1", None)
    assert empty["readiness"] == "not_ready"
    assert empty["bias"] is None
    assert empty["structure_phase"] is None
    short = build_tf_state("H1", trend_frame(12))
    assert short["readiness"] == "collecting"
    # Structure/direction need the bedrock floor: absent, not faked.
    assert short["structure_phase"] is None
    assert short["direction_word"] is None
    full = build_tf_state("H1", trend_frame(120))
    assert full["readiness"] == "ready"
    assert full["structure_phase"] is not None
    assert full["direction_word"] in ("up", "down", "flat")
    assert full["regime_label"] is not None


def test_state_volatility_is_measured():
    state = build_tf_state("H4", trend_frame(160, freq="4h"))
    vol = state["volatility"]
    assert vol["atr"] == pytest.approx(multitf.atr14(
        trend_frame(160, freq="4h")))
    assert vol["atr_ratio"] is not None
    assert vol["atr_state"] in (
        "contracting", "normal", "elevated", "high", "extreme")
    short = build_tf_state("H4", trend_frame(1, freq="4h"))
    assert short["volatility"] == {
        "atr": None, "atr_ratio": None, "atr_state": None}


def test_state_records_are_json_native():
    state = build_tf_state("M5", trend_frame(120, freq="5min"))
    json.dumps(state)  # must not raise: the journal is JSONL


# ------------------------------------------------------------------
# the no-verdict pin (SPEC §3 D-rule)
# ------------------------------------------------------------------
def test_builder_never_consults_verdicts(monkeypatch):
    """evaluate()/evaluate_hierarchy() are sleeve VERDICTS.  The
    situation is a measurement: building it with both verdict
    functions rigged to raise must change nothing."""
    def _boom(*args, **kwargs):
        raise AssertionError("verdict consulted by the situation")

    monkeypatch.setattr(multitf, "evaluate", _boom)
    from goldenswing.live import hierarchy
    monkeypatch.setattr(hierarchy, "evaluate_hierarchy", _boom)
    monkeypatch.setattr(multitf, "check_vetoes", _boom)
    built = build_situation(full_stack())
    assert built["per_tf"]["H1"]["bias"] == 1


def test_module_source_has_no_verdict_calls():
    import goldenswing.live.situation as mod
    src = open(mod.__file__, encoding="utf-8").read()
    assert "evaluate_hierarchy(" not in src
    assert "multitf.evaluate(" not in src
    assert ".evaluate(" not in src
    assert "check_vetoes" not in src
    assert "check_hard_veto" not in src


# ------------------------------------------------------------------
# forecasts: verbatim or None
# ------------------------------------------------------------------
def test_outlook_carried_verbatim_and_copied():
    outlook = {"method": "empirical", "horizons": {"12": {
        "direction": "UP", "point_price": 2010.5}}}
    built = build_situation(full_stack(), forecast_outlook=outlook)
    carried = built["forecasts"]["outlook"]
    assert carried == outlook
    assert carried is not outlook  # a copy: later mutation of the
    outlook["horizons"]["12"]["point_price"] = 1.0  # source must
    assert carried["horizons"]["12"]["point_price"] == 2010.5


def test_outlook_none_when_absent():
    built = build_situation(full_stack())
    assert built["forecasts"] == {"outlook": None}


def test_build_situation_shape_and_clocks():
    stack = full_stack()
    asof = stack.cursors["H1"]
    built = build_situation(stack, asof=asof, situation_id="X-0001")
    assert built["version"] == 1
    assert built["symbol"] == "XAUUSD"
    assert built["situation_id"] == "X-0001"
    for tf in situation.SITUATION_TFS:
        assert built["clocks"][tf] == str(stack.cursors[tf])
        assert built["per_tf"][tf]["readiness"] == "ready"
    assert built["clocks"]["M1"] is None  # no aux frame supplied
    json.dumps(built)


# ------------------------------------------------------------------
# keeper cadence + journal
# ------------------------------------------------------------------
def _journal_records(journal_dir):
    path = journal_dir / "situations.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_h1_close_journals_exactly_one_snapshot(tmp_path):
    keeper = SituationKeeper(tmp_path)
    stack = full_stack()
    rec = keeper.on_tf_close("H1", stack, asof=stack.cursors["H1"])
    assert rec["kind"] == SNAPSHOT_KIND
    records = _journal_records(tmp_path)
    assert len(records) == 1
    snap = records[0]["situation"]
    # SPEC §9 reconciliation: the snapshot's clocks equal the
    # stack cursors at that close.
    for tf in situation.SITUATION_TFS:
        assert snap["clocks"][tf] == str(stack.cursors[tf])
    assert keeper.latest_id() == snap["situation_id"]
    assert keeper.latest() == snap
    # latest-snapshot file landed, checksummed.
    latest_file = json.loads(
        (tmp_path / "situation_latest.json").read_text(
            encoding="utf-8"))
    assert latest_file["situation"]["situation_id"] == \
        snap["situation_id"]
    assert len(latest_file["checksum"]) == 64


def test_delta_only_on_real_change_and_chain(tmp_path):
    keeper = SituationKeeper(tmp_path)
    stack = full_stack()
    keeper.on_tf_close("H1", stack, asof=stack.cursors["H1"])
    snap_id = keeper.latest_id()
    # Same M5 state rebuilt -> nothing journaled.
    assert keeper.on_tf_close(
        "M5", stack, asof=stack.cursors["M5"]) is None
    assert len(_journal_records(tmp_path)) == 1
    # A genuinely different M5 frame (trend flipped) -> one delta,
    # chained to the snapshot.
    flipped = stack_of(M5=trend_frame(120, freq="5min", drift=-0.8))
    rec = keeper.on_tf_close("M5", flipped,
                             asof=flipped.cursors["M5"])
    assert rec["kind"] == DELTA_KIND
    assert rec["tf"] == "M5"
    assert rec["prev_situation_id"] == snap_id
    assert rec["state"]["bias"] == -1
    assert keeper.latest()["per_tf"]["M5"]["bias"] == -1
    # Unchanged again -> silence.
    assert keeper.on_tf_close(
        "M5", flipped, asof=flipped.cursors["M5"]) is None
    assert len(_journal_records(tmp_path)) == 2


def test_keeper_recovery_and_reconstruction(tmp_path):
    keeper = SituationKeeper(tmp_path)
    stack = full_stack()
    keeper.on_tf_close("H1", stack, asof=stack.cursors["H1"])
    id_a = keeper.latest_id()
    flipped = stack_of(M5=trend_frame(120, freq="5min", drift=-0.8))
    keeper.on_tf_close("M5", flipped, asof=flipped.cursors["M5"])
    id_b = keeper.latest_id()
    keeper.on_tf_close("H1", stack, asof=stack.cursors["H1"])
    id_c = keeper.latest_id()
    assert len({id_a, id_b, id_c}) == 3

    # A4.3: reconstruct exactly what was known at each past id.
    at_a = keeper.reconstruct(id_a)
    assert at_a["situation_id"] == id_a
    assert at_a["per_tf"]["M5"]["bias"] == 1
    at_b = keeper.reconstruct(id_b)
    assert at_b["situation_id"] == id_b
    assert at_b["prev_situation_id"] == id_a
    assert at_b["per_tf"]["M5"]["bias"] == -1  # delta applied
    assert at_b["per_tf"]["H1"]["bias"] == 1   # snapshot carried
    at_c = keeper.reconstruct(id_c)
    assert at_c["per_tf"]["M5"]["bias"] == 1   # H1 snapshot rebuilt
    assert keeper.reconstruct("no-such-id") is None

    # Restart: a fresh keeper on the same directory wakes with
    # the last situation and continues the id sequence.
    woken = SituationKeeper(tmp_path)
    assert woken.latest() == keeper.latest()
    rec = woken.on_tf_close("H1", stack, asof=stack.cursors["H1"])
    assert rec["prev_situation_id"] == id_c
    assert rec["seq"] == 4


def test_keeper_outlook_flows_and_survives_recovery(tmp_path):
    keeper = SituationKeeper(tmp_path)
    stack = full_stack()
    outlook = {"method": "empirical", "horizons": {}}
    keeper.on_tf_close("H1", stack, asof=stack.cursors["H1"],
                       forecast_outlook=outlook)
    assert keeper.latest()["forecasts"]["outlook"] == outlook
    woken = SituationKeeper(tmp_path)
    assert woken.latest()["forecasts"]["outlook"] == outlook
    # No outlook ever supplied -> None, never an exception.
    keeper2 = SituationKeeper(tmp_path / "sub")
    keeper2.on_tf_close("H1", stack, asof=stack.cursors["H1"])
    assert keeper2.latest()["forecasts"]["outlook"] is None


def test_keeper_ignores_non_situation_timeframes(tmp_path):
    keeper = SituationKeeper(tmp_path)
    stack = full_stack()
    assert keeper.on_tf_close("W1", stack, asof=None) is None
    assert keeper.on_tf_close("M1", stack, asof=None) is None
    assert _journal_records(tmp_path) == []


# ------------------------------------------------------------------
# runner wiring
# ------------------------------------------------------------------
def _runner(tmp_path, **cfg_kw):
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        news_pause_enabled=False, **cfg_kw)
    return MT5Runner(cfg)


def _trade_journal_records(tmp_path):
    path = tmp_path / "mt5_journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_runner_stamps_cycle_funnel_records(tmp_path):
    runner = _runner(tmp_path)
    stack = full_stack()
    runner._stack = stack
    tick = {"time": stack.cursors["H1"]}
    runner._situation_on_closes(["H1"], tick)
    snaps = [r for r in _trade_journal_records(tmp_path)
             if r.get("type") == "situation"]
    assert len(snaps) == 1
    sit_id = snaps[0]["situation"]["situation_id"]
    rec = runner._emit_eval(playbook="probe",
                            evaluation_clock="H1",
                            outcome="no_setup")
    assert rec["situation_id"] == sit_id
    evals = [r for r in _trade_journal_records(tmp_path)
             if r.get("type") == "evaluation"]
    assert evals[-1]["situation_id"] == sit_id


def test_runner_no_stamp_before_first_situation(tmp_path):
    runner = _runner(tmp_path)
    rec = runner._emit_eval(playbook="probe",
                            evaluation_clock="H1",
                            outcome="no_setup")
    assert "situation_id" not in rec


def test_runner_situation_disabled_is_silent(tmp_path):
    runner = _runner(tmp_path, situation_enabled=False)
    stack = full_stack()
    runner._stack = stack
    runner._situation_on_closes(["H1"], {"time":
                                         stack.cursors["H1"]})
    assert [r for r in _trade_journal_records(tmp_path)
            if r.get("type") == "situation"] == []
    rec = runner._emit_eval(playbook="probe",
                            evaluation_clock="H1",
                            outcome="no_setup")
    assert "situation_id" not in rec
