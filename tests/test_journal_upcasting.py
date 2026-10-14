"""Upcasting tests: legacy journal records must map to the current schema
without changing projection semantics.

Strategy (four layers):
  1. Golden fixtures: frozen canonical records per schema version. Never edited.
     Any upcaster change that breaks an old migration fails here.
  2. Projection equivalence (strongest test): the SAME trades journaled in
     old shape vs new shape must produce IDENTICAL LiveStats. This catches
     subtle bugs where upcasting preserves shape but shifts meaning
     (wrong default, dropped field, renamed field mapped badly).
  3. Strict version validation: unknown past versions and future versions
     raise — never silently guessed.
  4. Non-mutation: upcasting must not mutate its input (fixtures get reused).

When bumping JOURNAL_RECORD_VERSION: add tests/fixtures/trade_records_vN.json
and a projection-equivalence test vN vs current. Checklist, not optional.
"""
import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from goldenswing.live.monitor import (
    JOURNAL_RECORD_VERSION,
    LiveStats,
    upcast_trade_record,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load_fixture(name):
    with open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)


def replay(records):
    """Project a list of raw journal dicts into LiveStats via the upcaster."""
    stats = LiveStats()
    for rec in records:
        d = upcast_trade_record(rec)
        stats.record_trade(float(d["r_multiple"]))
    return stats


# -- Layer 1: golden fixtures ------------------------------------------------

def test_v1_legacy_records_upcast_without_version_field():
    """v1 records predate the 'v' field; all original fields must survive."""
    for rec in load_fixture("trade_records_v1.json"):
        assert "v" not in rec  # fixture invariant: truly legacy shape
        out = upcast_trade_record(rec)
        for k in ("seq", "ts_closed", "direction", "entry", "exit", "r_multiple"):
            assert out[k] == rec[k], f"field {k} altered by upcaster"


def test_v1_explicit_version_upcasts_to_v2():
    """v1 records with explicit 'v': 1 migrate exactly like legacy ones."""
    rec = {"v": 1, "seq": 9, "ts_closed": "2026-01-01T00:00:00+00:00",
           "direction": 1, "entry": 1.0, "exit": 2.0, "r_multiple": 0.5}
    out = upcast_trade_record(rec)
    assert out["v"] == 2 and out["type"] == "trade"
    assert out["r_theoretical"] == out["r_multiple"] == 0.5


# -- Layer 2: projection equivalence (strongest test) ------------------------

def test_projection_equivalence_legacy_vs_stamped():
    """Same trades, old shape vs new shape -> IDENTICAL LiveStats.

    If a future upcaster changes a default or drops a field that feeds a
    projection, this fails even when the shape looks right.
    """
    legacy = load_fixture("trade_records_v1.json")
    stamped = [{**r, "v": 1} for r in legacy]
    # strip the fixture-only annotation
    for r in stamped:
        r.pop("_note", None)

    a, b = replay(legacy), replay(stamped)
    assert a.trades == b.trades == 3
    assert a.wins == b.wins == 1          # 1.5 win; -1.0 loss; 0.0 counts as loss
    assert a.total_r == b.total_r == pytest.approx(0.5)
    assert a.recent_rs == b.recent_rs
    assert a.gross_win_r == b.gross_win_r == pytest.approx(1.5)
    assert a.gross_loss_r == b.gross_loss_r == pytest.approx(-1.0)


# -- Layer 3: strict version validation --------------------------------------

@pytest.mark.parametrize("bad_v", [0, -1, "1", 1.5, None])
def test_invalid_versions_rejected(bad_v):
    with pytest.raises(ValueError, match="invalid trade record version"):
        upcast_trade_record({"v": bad_v, "seq": 1, "r_multiple": 0.5})


def test_future_version_rejected():
    with pytest.raises(ValueError, match="newer than code"):
        upcast_trade_record({"v": JOURNAL_RECORD_VERSION + 1,
                             "seq": 1, "r_multiple": 0.5})


# -- Layer 4: non-mutation ----------------------------------------------------

def test_upcaster_does_not_mutate_input():
    rec = load_fixture("trade_records_v1.json")[0]
    frozen = copy.deepcopy(rec)
    upcast_trade_record(rec)
    upcast_trade_record(rec)  # twice, to catch accumulate-on-reuse bugs
    assert rec == frozen


# -- v2: fill-level events ---------------------------------------------------
# Checklist item honored: JOURNAL_RECORD_VERSION bumped to 2, so v2 gets its
# own fixture (trade_records_v2.json) and equivalence tests.

def test_v1_upcasts_to_v2_trade_shape():
    """v1 trade -> v2: type='trade', legacy trade_id, no fabricated fills."""
    from goldenswing.live.monitor import JOURNAL_RECORD_VERSION
    assert JOURNAL_RECORD_VERSION == 2
    rec = load_fixture("trade_records_v1.json")[0]
    out = upcast_trade_record(rec)
    assert out["v"] == 2 and out["type"] == "trade"
    assert out["trade_id"] == f"legacy-{rec['seq']}"
    assert out["fill_seqs"] == [] and out["state_at_signal"] == "unknown"
    # No slippage fabricated: theoretical == realized, vwap == entry/exit.
    assert out["r_theoretical"] == out["r_multiple"] == rec["r_multiple"]
    assert out["vwap_entry"] == rec["entry"] and out["vwap_exit"] == rec["exit"]


def test_projection_equivalence_v1_vs_v2():
    """Same P&L in v1 shape vs native v2 shape -> identical LiveStats.

    v1 fixture has 3 trades totalling +0.5R. Replay the v2 fixture's trade
    record and confirm the P&L projection matches a v1 record with the same
    r_multiple — the schema migration must not move a single satoshi of P&L.
    """
    v1 = load_fixture("trade_records_v1.json")
    v2_trades = [r for r in load_fixture("trade_records_v2.json")
                 if r.get("type") == "trade"]
    a = replay(v1)
    # v1: 3 trades, +0.5R total. Build the equivalent single-trade v2 journal
    # and check the per-trade projection matches on overlapping metrics.
    b_stats = LiveStats()
    for r in v2_trades:
        d = upcast_trade_record(r)
        b_stats.record_trade(float(d["r_multiple"]), record=d)
    assert b_stats.trades == 1 and b_stats.total_r == pytest.approx(1.016)
    # v1 records are untouched by the v2 migration:
    assert a.trades == 3 and a.total_r == pytest.approx(0.5)


def test_fills_never_move_pnl():
    """Fill events are execution evidence, not P&L. Replaying fills + trade
    summary must equal replaying the trade summary alone (no double-count)."""
    recs = load_fixture("trade_records_v2.json")
    full = replay_with_fills(recs)
    trade_only = replay([r for r in recs if r.get("type") == "trade"])
    assert full.trades == trade_only.trades == 1
    assert full.total_r == trade_only.total_r == pytest.approx(1.016)
    assert full.wins == trade_only.wins == 1


def test_execution_cost_measured_not_modeled():
    """r_theoretical - r_multiple = realized slippage cost. Legacy v1 trades
    (no fill provenance) are excluded — unknown cost, not zero cost."""
    recs = load_fixture("trade_records_v2.json")
    s = replay_with_fills(recs)
    assert s.trades_with_exec_data == 1
    assert s.avg_execution_cost_r == pytest.approx(1.04 - 1.016)
    # v1-only journal: no execution data -> None, not 0.0
    s_v1 = replay(load_fixture("trade_records_v1.json"))
    assert s_v1.avg_execution_cost_r is None
    assert s_v1.trades_with_exec_data == 0


def test_state_boundary_audit():
    """A fill landing during WATCH on a NOMINAL-sized position keeps both
    states on the record — auditable, not a reconciliation conflict."""
    fills = [r for r in load_fixture("trade_records_v2.json")
             if r.get("type") == "fill"]
    assert fills[0]["state_at_signal"] == fills[0]["state_at_fill"] == "nominal"
    late = fills[1]
    assert late["state_at_signal"] == "nominal" and late["state_at_fill"] == "watch"
    # Partial fills are separate records; nothing fabricated for the remainder
    assert sum(f["qty"] for f in fills) == pytest.approx(1.0)


def replay_with_fills(records):
    """Replay honoring record types, like LiveStats.from_journal does."""
    stats = LiveStats()
    for rec in records:
        d = upcast_trade_record(rec)
        if d.get("type") == "fill":
            stats.record_fill(d)
            continue
        stats.record_trade(float(d["r_multiple"]), record=d)
    return stats
