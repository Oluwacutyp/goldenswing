"""Judgment Core S3 — outcome reconciliation + self-audit.

Pins: forward outcomes measured honestly (direction, magnitude,
R in the trade's own geometry); calibration bucketed from
outcomes; the Nov 3 failure shape — a material H1 move with zero
candidate evaluations — flagged as a missed opportunity, while an
evaluated bar (even an abstention) is a decision, not a miss; and
thesis outcomes reconciled when the keeper closes a thesis.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from goldenswing.execution.judgment_audit import (
    calibration_add,
    calibration_view,
    forward_outcome,
    missed_opportunity,
)


# ------------------------------------------------------------------
# pure core
# ------------------------------------------------------------------

def test_forward_outcome_signs_and_scales():
    long_hit = forward_outcome(1, 3900.0, 3910.0, 5.0)
    assert long_hit == {"move": 10.0, "move_atr": 2.0, "hit": True}
    short_miss = forward_outcome(-1, 3900.0, 3905.0, 5.0)
    assert short_miss["move"] == -5.0 and short_miss["hit"] is False


def test_calibration_buckets_carry_counts_and_gaps():
    buckets: dict = {}
    calibration_add(buckets, confidence=0.6, expected_r=0.4,
                    realized_r=1.2)
    calibration_add(buckets, confidence=0.6, expected_r=0.4,
                    realized_r=-1.0)
    calibration_add(buckets, confidence=0.1, expected_r=None,
                    realized_r=0.5)
    view = calibration_view(buckets)
    assert view["0.50-0.75"]["n"] == 2
    assert view["0.50-0.75"]["hit_rate"] == 0.5
    assert view["0.50-0.75"]["mean_expected_r"] == 0.4
    assert view["0.50-0.75"]["mean_realized_r"] == 0.1
    assert view["0.00-0.25"]["mean_expected_r"] is None


def test_missed_opportunity_is_the_nov3_shape():
    # London, 2026-11-03: a hard bearish H1 bar, nobody evaluated.
    miss = missed_opportunity(bar_open=3920.0, bar_close=3906.0,
                              atr=8.0, evaluated=False)
    assert miss is not None
    assert miss["direction"] == -1
    assert miss["move_atr"] == 1.75
    # evaluated (even to abstain) = a decision, not a miss
    assert missed_opportunity(bar_open=3920.0, bar_close=3906.0,
                              atr=8.0, evaluated=True) is None
    # a quiet bar is not a missed opportunity
    assert missed_opportunity(bar_open=3920.0, bar_close=3918.0,
                              atr=8.0, evaluated=False) is None


# ------------------------------------------------------------------
# runner wiring
# ------------------------------------------------------------------

class _Journal:
    def __init__(self):
        self.records = []
        self.notes = []

    def append(self, record):
        self.records.append(record)

    def log_note(self, text):
        self.notes.append(str(text))


def _h1(closes, opens=None):
    idx = pd.date_range("2026-11-03T00:00:00Z", periods=len(closes),
                        freq="h")
    opens = opens or closes
    return pd.DataFrame(
        {"open": opens, "high": [max(o, c) + 1 for o, c
                                 in zip(opens, closes)],
         "low": [min(o, c) - 1 for o, c in zip(opens, closes)],
         "close": closes, "volume": [100] * len(closes)}, index=idx)


def _runner(tmp_path):
    from goldenswing.execution.mt5_runner import (
        MT5Runner,
        MT5RunnerConfig,
    )
    from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
    from mt5_fakes import FakeMT5

    feed = MT5Feed(FakeMT5(), MT5FeedConfig(symbol="XAUUSDm"))
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "j.jsonl",
        risk_state_path=tmp_path / "r.json",
        kill_switch_path=tmp_path / "KILL",
        news_pause_enabled=False, max_daily_trades=0,
        max_hourly_trades=0)
    runner = MT5Runner(cfg, feed=feed)
    runner.journal = _Journal()
    return runner


def test_shadow_judgment_reconciles_at_both_horizons(tmp_path):
    runner = _runner(tmp_path)
    runner._last_tick = {"mid": 3900.0, "bid": 3899.9, "ask": 3900.1}
    runner._audit_register_shadow(
        {"decision": "take", "expected_r": 0.5, "confidence": 0.6},
        {"direction": -1, "entry": 3900.0, "stop": 3910.0},
        "2026-11-03T00:30:00Z", playbook="qm_m5")
    # judged 00:30 at 3900 (short, stop 3910): the 02:00 close —
    # first bar at/after the 1h due (01:30) — is 3880 (+2.0R);
    # the 05:00 close (4h due 04:30) is 3875 (+2.5R).
    closes = [3900.0, 3895.0, 3880.0, 3878.0, 3879.0, 3875.0] \
        + [3876.0] * 14
    for n in range(15, len(closes) + 1):
        runner._stack = SimpleNamespace(frames={"H1": _h1(closes[:n])})
        runner._judgment_audit_cycle()
    recs = [r for r in runner.journal.records
            if r.get("type") == "judgment_reconciliation"]
    assert {r["horizon_h"] for r in recs} == {1, 4}
    one_h = next(r for r in recs if r["horizon_h"] == 1)
    assert one_h["hit"] is True
    assert one_h["realized_r"] == 2.0
    st = runner._audit_state()
    assert st["tallies"]["take_n"] == 2
    assert st["tallies"]["take_hits"] == 2
    assert st["pending"] == []


def test_material_bar_without_evaluation_is_flagged(tmp_path):
    runner = _runner(tmp_path)
    closes = [3900.0] * 19 + [3880.0]
    opens = [3900.0] * 19 + [3905.0]
    runner._stack = SimpleNamespace(
        frames={"H1": _h1(closes, opens)})
    runner._judgment_audit_cycle()
    misses = [r for r in runner.journal.records
              if r.get("kind") == "missed_opportunity"]
    assert len(misses) == 1
    assert misses[0]["direction"] == -1


def test_evaluated_bar_is_not_flagged(tmp_path):
    runner = _runner(tmp_path)
    closes = [3900.0] * 19 + [3880.0]
    opens = [3900.0] * 19 + [3905.0]
    frame = _h1(closes, opens)
    runner._audit_mark_evaluated(frame.index[-1])
    runner._stack = SimpleNamespace(frames={"H1": frame})
    runner._judgment_audit_cycle()
    assert not any(r.get("kind") == "missed_opportunity"
                   for r in runner.journal.records)


class _Keeper:
    def __init__(self):
        self._status = {"active": True, "id": "th-1",
                        "scenario": "bearish", "direction": -1,
                        "winner_p": 0.6, "created_ts": 1.0,
                        "updated_ts": 2.0, "status": "active",
                        "plan": None, "closed_total": 0}

    def status(self):
        return dict(self._status)


def test_thesis_outcome_reconciles_on_close(tmp_path):
    runner = _runner(tmp_path)
    keeper = _Keeper()
    runner._thesis_keeper = keeper
    runner._last_tick = {"mid": 3900.0}
    runner._audit_sync_thesis()
    assert "th-1" in runner._audit_state()["thesis_pending"]
    keeper._status = {"active": False, "closed_total": 1}
    runner._last_tick = {"mid": 3880.0}
    runner._audit_sync_thesis()
    outcomes = [r for r in runner.journal.records
                if r.get("kind") == "thesis_outcome"]
    assert len(outcomes) == 1
    assert outcomes[0]["hit"] is True
    assert outcomes[0]["scenario"] == "bearish"
    assert runner._audit_state()["tallies"]["thesis_hits"] == 1
