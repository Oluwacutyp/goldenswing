"""MarketSituation Pass 1 (SPEC §3.3 + §4.2 + §9 Pass 1;
AMENDMENTS-1 §A1 + §A4).

Pins:
- the §3.3 blocks ride the snapshot: state-derived skeletons
  when the runner supplies nothing (absent named in
  data_quality.missing), verbatim organ blocks when it does;
- relationships ride every assembled situation; a pair-kind
  change journals a delta even when no TF state changed;
- the hierarchy reading appears exactly once per H1 snapshot
  through the runner seam;
- situation_event records fire on structure-event episodes,
  structural breaks, volatility-state changes and liquidity
  pool entries/takes — each anchored to the chain;
- the live block is provisional, carried on records, and never
  contaminates a closed-bar state field.
"""
import json

import numpy as np
import pandas as pd
import pytest

from goldenswing.live import situation
from goldenswing.live.situation import (
    DELTA_KIND, EVENT_KIND, SNAPSHOT_KIND, SituationKeeper,
    build_situation)
from tests.test_situation import (
    TF_FREQ, flat_frame, full_stack, stack_of, trend_frame)


# ------------------------------------------------------------------
# frame crafting (calm trends the displacement gate can read)
# ------------------------------------------------------------------
def calm_frame(n=120, *, freq="5min", drift=0.3, amp=2.0,
               base=2000.0, period=12):
    idx = pd.date_range("2026-01-01", periods=n, freq=freq,
                        tz="UTC")
    closes = np.array([
        base + drift * i
        + amp * (1.0 - abs((i % period) / (period / 2.0) - 1.0))
        for i in range(n)], dtype=float)
    opens = np.concatenate([[closes[0] - 0.1], closes[:-1]])
    return pd.DataFrame(
        {"open": opens, "high": closes + 0.25,
         "low": closes - 0.25, "close": closes, "volume": 100.0},
        index=idx)


def append_bar(frame, close, open_):
    ts = frame.index[-1] + (frame.index[-1] - frame.index[-2])
    row = pd.DataFrame(
        {"open": [open_], "high": [max(open_, close) + 0.3],
         "low": [min(open_, close) - 0.3], "close": [close],
         "volume": [100.0]}, index=[ts])
    return pd.concat([frame, row])


def calm_stack():
    return stack_of(**{tf: calm_frame(120, freq=TF_FREQ[tf])
                       for tf in situation.SITUATION_TFS})


def _journal_records(journal_dir):
    path = journal_dir / "situations.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _events(journal_dir):
    return [r for r in _journal_records(journal_dir)
            if r.get("kind") == EVENT_KIND]


# ------------------------------------------------------------------
# §3.3 blocks on the pure builder
# ------------------------------------------------------------------
def test_blocks_skeleton_when_unsupplied():
    built = build_situation(full_stack())
    assert built["regime"]["bedrock_label_h1"] == \
        built["per_tf"]["H1"]["regime_label"]
    assert built["regime"]["per_tf_labels"]["H1"] == \
        built["per_tf"]["H1"]["regime_label"]
    assert built["regime"]["day"] is None
    assert built["location"] is None
    assert built["activity"] is None
    assert built["cross_asset"] is None
    dq = built["data_quality"]
    assert dq["health"] is None
    assert dq["per_tf_freshness_s"]["H1"] == \
        built["per_tf"]["H1"]["freshness_s"]
    for name in ("location", "activity", "cross_asset"):
        assert name in dq["missing"]
    json.dumps(built)


def test_blocks_supplied_merge_and_verbatim():
    blocks = {
        "regime": {"day": {"day_type": "trend",
                           "adr_state": "fresh"}},
        "location": {"price": 2050.0, "zones_above": [],
                     "zones_below": []},
        "cross_asset": {"dxy": {"available": False},
                        "silver_rs": None, "dxy_rs": None},
        "data_quality": {"health": "healthy"},
    }
    built = build_situation(full_stack(), blocks=blocks)
    # Supplied blocks ride verbatim...
    assert built["location"] == blocks["location"]
    assert built["cross_asset"] == blocks["cross_asset"]
    assert built["data_quality"]["health"] == "healthy"
    # ...while the regime skeleton still fills what the supplier
    # did not repeat, under the supplied day object.
    assert built["regime"]["day"] == {"day_type": "trend",
                                      "adr_state": "fresh"}
    assert built["regime"]["per_tf_labels"]["H1"] is not None
    assert "location" not in built["data_quality"]["missing"]
    assert "activity" in built["data_quality"]["missing"]
    # The caller's dict is never aliased.
    blocks["location"]["price"] = 1.0
    assert built["location"]["price"] == 2050.0


# ------------------------------------------------------------------
# relationships on the keeper cadence
# ------------------------------------------------------------------
def test_snapshot_carries_engine_relationships(tmp_path):
    keeper = SituationKeeper(tmp_path)
    stack = full_stack()  # every TF bullish, trend-phased
    keeper.on_tf_close("H1", stack, asof=stack.cursors["H1"])
    rels = keeper.latest()["relationships"]
    engine = [r for r in rels if r["source"] == "engine"]
    assert len(engine) == 7
    assert all(r["kind"] == "aligned" for r in engine)
    # No hierarchy row: the keeper alone never consults the
    # verdict — the runner supplies the reading.
    assert [r for r in rels if r["source"] == "hierarchy"] == []


def test_relationship_kind_change_journals_delta(tmp_path):
    keeper = SituationKeeper(tmp_path)
    stack = stack_of(
        D1=trend_frame(120, freq="D", drift=0.8),
        H4=trend_frame(120, freq="4h", drift=-0.8),
        H1=trend_frame(120, freq="h", drift=0.8),
        M30=trend_frame(120, freq="30min", drift=0.8),
        M15=trend_frame(120, freq="15min", drift=-0.8),
        M5=trend_frame(120, freq="5min", drift=0.8))
    keeper.on_tf_close("H1", stack, asof=stack.cursors["H1"],
                       price=2090.0)
    kinds = {(r["htf"], r["ltf"]): r["kind"]
             for r in keeper.latest()["relationships"]}
    assert kinds[("D1", "H4")] == "correcting"
    # No TF state changes; price alone breaks D1's defended
    # swing (~2085.7) — the pair flips to conflicting and the
    # change must journal.
    rec = keeper.on_tf_close("M5", stack,
                             asof=stack.cursors["M5"], price=2080.0)
    assert rec is not None and rec["kind"] == DELTA_KIND
    rels = {(r["htf"], r["ltf"]): r["kind"]
            for r in rec["relationships"]}
    assert rels[("D1", "H4")] == "conflicting"
    # Reconstruction at the delta id sees the flipped kind.
    at = keeper.reconstruct(rec["situation_id"])
    kinds_at = {(r["htf"], r["ltf"]): r["kind"]
                for r in at["relationships"]}
    assert kinds_at[("D1", "H4")] == "conflicting"


# ------------------------------------------------------------------
# events (A4.1)
# ------------------------------------------------------------------
def test_structure_event_then_break(tmp_path):
    keeper = SituationKeeper(tmp_path)
    stack = calm_stack()
    keeper.on_tf_close("H1", stack, asof=stack.cursors["H1"])
    assert _events(tmp_path) == []
    base_m5 = calm_frame(120, freq="5min")
    from goldenswing.live.situation import build_tf_state
    hi = build_tf_state("M5", base_m5)["swings"][
        "confirmed_high"]["price"]
    lo = build_tf_state("M5", base_m5)["swings"][
        "confirmed_low"]["price"]
    bos_stack = stack_of(M5=append_bar(base_m5, hi + 4.0, hi - 0.5))
    keeper.on_tf_close("M5", bos_stack,
                       asof=bos_stack.cursors["M5"])
    # (The displacement bar also lifts the ATR regime — a
    # genuine second event family, filtered out here.)
    evs = [e for e in _events(tmp_path)
           if e["event_type"] in ("structure_event",
                                  "structural_break")]
    assert len(evs) == 1
    assert evs[0]["event_type"] == "structure_event"
    assert evs[0]["tf"] == "M5"
    assert evs[0]["detail"]["event"]["kind"] == "BOS"
    assert evs[0]["situation_id"] == keeper.latest_id()
    # A CHoCH against replaces the BOS episode: a structural
    # break of the previous read, not a second ordinary event.
    choch_stack = stack_of(
        M5=append_bar(base_m5, lo - 4.0, lo + 0.5))
    keeper.on_tf_close("M5", choch_stack,
                       asof=choch_stack.cursors["M5"])
    evs = [e for e in _events(tmp_path)
           if e["event_type"] in ("structure_event",
                                  "structural_break")]
    assert len(evs) == 2
    assert evs[1]["event_type"] == "structural_break"
    assert evs[1]["detail"]["event"]["kind"] == "CHoCH"
    assert evs[1]["detail"]["previous"]["kind"] == "BOS"


def test_volatility_change_event(tmp_path):
    keeper = SituationKeeper(tmp_path)
    stack = calm_stack()
    keeper.on_tf_close("H1", stack, asof=stack.cursors["H1"])
    wide = calm_frame(120, freq="5min")
    for _ in range(6):
        nxt = float(wide["close"].iloc[-1]) + 0.5
        ts = wide.index[-1] + (wide.index[-1] - wide.index[-2])
        row = pd.DataFrame(
            {"open": [nxt - 0.3], "high": [nxt + 12.5],
             "low": [nxt - 12.5], "close": [nxt],
             "volume": [100.0]}, index=[ts])
        wide = pd.concat([wide, row])
    wide_stack = stack_of(M5=wide)
    keeper.on_tf_close("M5", wide_stack,
                       asof=wide_stack.cursors["M5"])
    vol = [e for e in _events(tmp_path)
           if e["event_type"] == "volatility_change"]
    assert len(vol) == 1
    assert vol[0]["tf"] == "M5"
    assert vol[0]["detail"]["from"] == "normal"
    assert vol[0]["detail"]["to"] in (
        "elevated", "high", "extreme")


def test_liquidity_pool_entered_then_taken(tmp_path):
    keeper = SituationKeeper(tmp_path)
    stack = calm_stack()
    zone = {"zone_id": "z-up", "bottom": 2100.0, "top": 2110.0,
            "proximal": 2100.0, "distal": 2112.0}
    blocks = {"location": {"price": 2095.0, "zones_above": [zone],
                           "zones_below": []}}
    keeper.on_tf_close("H1", stack, asof=stack.cursors["H1"],
                       blocks=blocks, price=2095.0)
    assert _events(tmp_path) == []
    # Price steps inside the pool's band: entered.  The event
    # alone (no state/kind change) still anchors a delta.
    rec = keeper.on_tf_close("M5", stack,
                             asof=stack.cursors["M5"],
                             blocks=blocks, price=2105.0)
    assert rec is not None and rec.get("trigger") == "event"
    evs = _events(tmp_path)
    assert len(evs) == 1
    assert evs[0]["event_type"] == "liquidity_event"
    assert evs[0]["detail"]["change"] == "entered_pool"
    assert evs[0]["detail"]["zone_id"] == "z-up"
    # The pool leaves the nearest list (the store consumed it);
    # the take is still detected against the previous map.
    empty = {"location": {"price": 2115.0, "zones_above": [],
                          "zones_below": []}}
    keeper.on_tf_close("M5", stack, asof=stack.cursors["M5"],
                       blocks=empty, price=2115.0)
    evs = _events(tmp_path)
    assert len(evs) == 2
    assert evs[1]["detail"]["change"] == "pool_taken"
    assert evs[1]["detail"]["zone_id"] == "z-up"


# ------------------------------------------------------------------
# live block (A4.2): provisional, carried, never contaminating
# ------------------------------------------------------------------
def test_live_block_provisional_and_separate(tmp_path):
    keeper = SituationKeeper(tmp_path)
    stack = full_stack()
    keeper.on_tf_close("H1", stack, asof=stack.cursors["H1"])
    assert keeper.latest()["live"] is None
    before = keeper.latest()
    keeper.note_live(
        2095.0, asof=stack.cursors["H1"],
        zones_inside=[{"zone_id": "z1", "side": "demand",
                       "tf": "H1", "relation": "inside"}],
        positions=[{"ticket": 7, "direction": 1,
                    "entry": 2090.0, "stop": 2080.0,
                    "price": 2095.0, "r": 0.5, "best_r": 0.5}])
    latest = keeper.latest()
    live = latest["live"]
    assert live["provisional"] is True
    assert live["price"] == 2095.0
    assert live["zones"][0]["zone_id"] == "z1"
    assert live["positions"][0]["ticket"] == 7
    # Separation: not one closed-bar field moved.
    assert latest["per_tf"] == before["per_tf"]
    assert latest["relationships"] == before["relationships"]
    for state in latest["per_tf"].values():
        assert "live" not in state
    # The next snapshot carries the block on the record.
    keeper.on_tf_close("H1", stack, asof=stack.cursors["H1"])
    snaps = [r for r in _journal_records(tmp_path)
             if r.get("kind") == SNAPSHOT_KIND]
    assert snaps[-1]["situation"]["live"]["provisional"] is True


# ------------------------------------------------------------------
# runner wiring: blocks + exactly one hierarchy row per snapshot
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


def _snapshots(tmp_path):
    return [r["situation"] for r in _trade_journal_records(tmp_path)
            if r.get("type") == "situation"
            and r.get("kind") == SNAPSHOT_KIND]


def test_runner_snapshot_has_blocks_and_one_hierarchy(tmp_path):
    runner = _runner(tmp_path)
    stack = full_stack()
    runner._stack = stack
    tick = {"time": stack.cursors["H1"], "mid": 2095.0,
            "spread": 0.3}
    runner._situation_on_closes(["H1"], tick)
    runner._situation_on_closes(["H1"], tick)
    snaps = _snapshots(tmp_path)
    assert len(snaps) == 2
    for snap in snaps:
        for key in ("regime", "location", "activity",
                    "cross_asset", "data_quality",
                    "relationships"):
            assert key in snap
        assert snap["location"] is not None
        assert snap["cross_asset"]["silver_rs"] is None
        assert snap["data_quality"]["health"] in (
            "healthy", "degraded", "halt")
        assert snap["regime"]["per_tf_labels"]["H1"] is not None
        rows = [r for r in snap["relationships"]
                if r["source"] == "hierarchy"]
        assert len(rows) == 1
    # Decision records keep their Pass 0 shape: the situation
    # contributes its id stamp and nothing else.
    rec = runner._emit_eval(playbook="probe",
                            evaluation_clock="H1",
                            outcome="no_setup")
    assert rec["situation_id"] == snaps[-1]["situation_id"]


def test_runner_note_live_feeds_the_block(tmp_path):
    runner = _runner(tmp_path)
    stack = full_stack()
    runner._stack = stack
    tick = {"time": stack.cursors["H1"], "mid": 2095.0}
    runner._situation_on_closes(["H1"], tick)
    runner._situation_note_live(tick, 2096.0)
    keeper = runner._situation_keeper
    live = keeper.latest()["live"]
    assert live["provisional"] is True
    assert live["price"] == 2096.0
    assert live["positions"] == []


def test_runner_disabled_stays_silent_with_blocks(tmp_path):
    runner = _runner(tmp_path, situation_enabled=False)
    stack = full_stack()
    runner._stack = stack
    runner._situation_on_closes(
        ["H1"], {"time": stack.cursors["H1"], "mid": 2095.0})
    runner._situation_note_live({"time": stack.cursors["H1"]},
                                2095.0)
    assert _snapshots(tmp_path) == []
    assert getattr(runner, "_situation_keeper", None) in (
        None, False)
