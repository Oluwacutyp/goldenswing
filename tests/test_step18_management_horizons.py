"""Step 18 acceptance tests — declared management horizons
(owner rulings 2026-10-30 §4).

Every setup declares its management horizon (scalp / intraday /
swing) from its thesis — not its entry timeframe — and the six
management fields persist per setup: registry declaration ->
lifecycle store -> funnel records -> tracked position -> close
record.  Policies resolve ONLY to established, tested rule sets
(fixed_bracket_v1 = Step 13's qm_m5 bracket; standard_v1 =
today's standing TickConfig): no new R/ATR numbers exist in
this step, and Step 13's behaviour is pinned unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace

from goldenswing.execution.playbook_registry import (
    MANAGEMENT_HORIZONS, MANAGEMENT_POLICIES, PLAYBOOK_REGISTRY,
    management_horizon_for, management_policy_for, spec_for)
from goldenswing.execution.setup_lifecycle import TrackedSetup
from goldenswing.execution.mt5_runner import management_block_for

from test_step17_revalidation import _runner, _sig


# ---------------------------------------------------------- declarations
def test_every_playbook_declares_a_valid_horizon():
    assert len(PLAYBOOK_REGISTRY) == 18
    for spec in PLAYBOOK_REGISTRY:
        assert spec.management_horizon in MANAGEMENT_HORIZONS, \
            spec.name


def test_horizon_grounding_follows_the_thesis_clock():
    # The native routes declare their own.
    assert spec_for("qm_m5").management_horizon == "scalp"
    assert spec_for("qm_d1").management_horizon == "swing"
    # H1 roster + H4 range family: H4-context intraday theses.
    for name in ("sweep_reversal", "fvg_midline", "inversion",
                 "order_block", "narrative", "range_qm",
                 "range_sweep_reclaim", "range_mean_reversion",
                 "range_compression_breakout"):
        assert spec_for(name).management_horizon == "intraday", name
    # Hot scalp declares through its route.
    assert management_horizon_for("hot_scalp") == "scalp"
    assert management_horizon_for("qm_m5") == "scalp"
    assert management_horizon_for("qm_d1") == "swing"


def test_shadow_playbooks_gain_no_authority_from_declaring():
    for spec in PLAYBOOK_REGISTRY:
        if spec.route == "range_h4_h1":
            assert spec.shadow_evaluation is True


# ------------------------------------------------------------- policies
def test_policy_resolution_uses_only_established_rule_sets():
    assert management_policy_for("qm_m5") == ("fixed_bracket_v1", 1)
    assert management_policy_for("sweep_reversal") == ("standard_v1", 1)
    assert management_policy_for("qm_d1") == ("standard_v1", 1)
    assert management_policy_for("hot_scalp") == ("standard_v1", 1)
    for pid in MANAGEMENT_POLICIES:
        assert pid in ("fixed_bracket_v1", "standard_v1")


def test_management_block_carries_all_six_fields():
    sig = SimpleNamespace(direction=1, entry_ref=2005.0,
                          invalidation=1990.0, targets=[2040.0, 2060.0],
                          atr=10.0)
    block = management_block_for("sweep_reversal", sig)
    assert block == {
        "management_horizon": "intraday",
        "management_policy_id": "standard_v1",
        "management_policy_version": 1,
        "management_context_timeframe": "H1",
        "management_invalidation": 1990.0,
        "management_target": 2040.0,
    }
    qm = management_block_for("qm_m5", sig)
    assert qm["management_horizon"] == "scalp"
    assert qm["management_policy_id"] == "fixed_bracket_v1"
    assert qm["management_context_timeframe"] == "M5"
    hs = management_block_for("hot_scalp", sig)
    assert hs["management_horizon"] == "scalp"
    assert hs["management_policy_id"] == "standard_v1"
    assert hs["management_context_timeframe"] == "M5"
    d1 = management_block_for("qm_d1", sig)
    assert d1["management_horizon"] == "swing"
    assert d1["management_context_timeframe"] == "D1"
    rg = management_block_for("range_qm", sig)
    assert rg["management_context_timeframe"] == "H4"


# ------------------------------------------------------- lifecycle store
def test_tracked_setup_persists_the_management_block():
    block = management_block_for(
        "qm_d1", SimpleNamespace(invalidation=1950.0,
                                 targets=[2150.0]))
    setup = TrackedSetup(setup_id="s1", parent_setup_id="s1",
                         playbook="qm_d1", route="native",
                         direction=1, management=block)
    back = TrackedSetup.from_dict(setup.to_dict())
    assert back.management == block
    # A pre-Step-18 stored record (no management key) loads with
    # the empty default — old stores never break.
    legacy = {k: v for k, v in setup.to_dict().items()
              if k != "management"}
    assert TrackedSetup.from_dict(legacy).management == {}


def test_candidate_dict_carries_the_block(tmp_path):
    runner = _runner(tmp_path)
    sig = _sig(42, 2005.0, 1990.0)
    sig.management = management_block_for("sweep_reversal", sig)
    out = runner._candidate_dict(sig, None, None)
    assert out["management"]["management_horizon"] == "intraday"
    assert out["management"]["management_policy_id"] == "standard_v1"


# ------------------------------------------------------------ positions
def test_position_carries_horizon_policy_and_close_names_them(tmp_path):
    runner = _runner(tmp_path)
    pb = SimpleNamespace(name="sweep_reversal")
    sig = _sig(42, 2005.0, 1990.0)
    sig.management = management_block_for("sweep_reversal", sig)
    stats: dict = {}
    runner._place_live_order(pb, sig, stats)
    assert stats.get("executed") == 1
    pos = runner.tick_manager.position(9001)
    assert pos is not None
    assert pos.management_horizon == "intraday"
    assert pos.management_policy_id == "standard_v1"
    assert runner.tick_manager.is_fixed_bracket(9001) is False
    det: list = []
    runner.deterministic = SimpleNamespace(
        journal_event=lambda kind, data, idempotency_key=None:
        det.append((kind, data)))
    runner._record_close(9001, 1995.0, "stop",
                         {"reconciled_closed": 0})
    closed = [d for kind, d in det if kind == "trade_closed"]
    assert len(closed) == 1
    assert closed[0]["management_policy_id"] == "standard_v1"
    assert closed[0]["management_horizon"] == "intraday"


def test_qm_m5_position_agrees_with_the_fixed_bracket_rule(tmp_path):
    runner = _runner(tmp_path)
    pb = SimpleNamespace(name="qm_m5")
    sig = _sig(42, 2005.0, 1990.0)
    sig.management = management_block_for("qm_m5", sig)
    stats: dict = {}
    runner._place_live_order(pb, sig, stats)
    assert stats.get("executed") == 1
    pos = runner.tick_manager.position(9001)
    assert pos.management_horizon == "scalp"
    assert pos.management_policy_id == "fixed_bracket_v1"
    # The recorded policy AGREES with the manager's own Step-13
    # resolution for the same position — one truth, two surfaces.
    assert runner.tick_manager.is_fixed_bracket(9001) is True
