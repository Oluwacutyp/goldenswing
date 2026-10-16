"""Tests for the trigger-aware playbook boundary.

Trigger direction is immutable authority.  Playbooks inherit it and
validate only tactical pattern / location / score / invalidation /
costs.  ``pattern_absent`` is normal.  No ``direction_mismatch``
appears on this path.  Trigger context is snapshotted per playbook.
All playbooks are evaluated once per trigger fire.
"""
import copy

from goldenswing.playbooks.base import (
    Playbook, PlaybookVerdict, evaluate_all_playbooks,
    snapshot_trigger_context)


class _PatternPlaybook(Playbook):
    name = "test_pattern"
    live = False

    def candidates(self, ctx):
        return []

    def _tactical_pattern(self, bars, direction, trigger_ctx):
        return {"score": 70.0, "invalidation": 4000.0,
                "costs": {"spread": 0.2}, "entry": 4010.0,
                "stop": 4000.0, "target": 4030.0,
                "pattern": "test", "direction": direction}


class _AbsentPlaybook(Playbook):
    name = "test_absent"
    live = False

    def candidates(self, ctx):
        return []


class _OppositePlaybook(Playbook):
    name = "test_opposite"
    live = False

    def candidates(self, ctx):
        return []

    def _tactical_pattern(self, bars, direction, trigger_ctx):
        # Tactical pattern pointing the other way: for THIS trigger
        # it is absent, never a direction_mismatch rejection.
        return {"score": 90.0, "invalidation": 4000.0,
                "costs": {}, "direction": -direction}


class _MutatingPlaybook(Playbook):
    name = "test_mutating"
    live = False

    def candidates(self, ctx):
        return []

    def _tactical_pattern(self, bars, direction, trigger_ctx):
        trigger_ctx["direction"] = -direction  # must not leak out
        trigger_ctx["zone"] = "mutated"
        return None


def test_evaluate_inherits_trigger_direction():
    pb = _PatternPlaybook()
    v = pb.evaluate({}, {"direction": 1, "zone": (4000, 4020)})
    assert isinstance(v, PlaybookVerdict)
    assert v.status == "valid"
    assert v.direction == 1
    assert v.score == 70.0


def test_evaluate_pattern_absent_is_normal():
    pb = _AbsentPlaybook()
    v = pb.evaluate({}, {"direction": -1})
    assert v.status == "pattern_absent"
    assert v.direction == -1
    assert "direction_mismatch" not in v.reason


def test_evaluate_opposite_tactical_is_absent_not_mismatch():
    pb = _OppositePlaybook()
    v = pb.evaluate({}, {"direction": 1})
    assert v.status == "pattern_absent"
    assert v.direction == 1
    assert "direction_mismatch" not in v.reason
    assert "direction_mismatch" not in v.status


def test_evaluate_snapshots_trigger_context_per_playbook():
    ctx = {"direction": 1, "zone": [4000, 4020],
           "invalidation": 3990.0, "ttl_bars": 10}
    original = copy.deepcopy(ctx)
    _MutatingPlaybook().evaluate({}, ctx)
    assert ctx == original  # mutation inside evaluate never leaks
    snap1 = snapshot_trigger_context(ctx)
    snap1["zone"] = "changed"
    snap2 = snapshot_trigger_context(ctx)
    assert snap2["zone"] == [4000, 4020]


def test_evaluate_all_playbooks_once_each():
    pbs = [_PatternPlaybook(), _AbsentPlaybook(), _OppositePlaybook()]
    verdicts = evaluate_all_playbooks(pbs, {}, {"direction": 1})
    assert [v.playbook for v in verdicts] == [
        "test_pattern", "test_absent", "test_opposite"]
    assert [v.status for v in verdicts] == [
        "valid", "pattern_absent", "pattern_absent"]
    # No direction_mismatch anywhere on this path.
    assert not any("direction_mismatch" in (v.status + v.reason)
                   for v in verdicts)


def test_evaluate_no_direction_is_absent():
    pb = _PatternPlaybook()
    v = pb.evaluate({}, {"direction": 0})
    assert v.status == "pattern_absent"


def test_evaluate_missing_invalidation_is_invalid():
    class _NoInv(Playbook):
        name = "test_noinv"
        live = False

        def candidates(self, ctx):
            return []

        def _tactical_pattern(self, bars, direction, trigger_ctx):
            return {"score": 80.0, "costs": {}}

    v = _NoInv().evaluate({}, {"direction": 1})
    assert v.status == "invalid"
