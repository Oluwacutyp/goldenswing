from goldenswing.live.stacking import (
    StackingConfig, child_entry_id, evaluate_layer_add,
)


def args(**overrides):
    values = dict(
        parent_setup_id="parent-001", mode="pyramid", trigger_id="M1-reclaim-3",
        trigger_direction=1, parent_direction=1, parent_open=True,
        parent_thesis_valid=True, trigger_fresh=True, trigger_independent=True,
        trigger_already_used=False, parent_unrealized_r=0.8, active_layers=1,
        max_layers=3, risk_budget_money=10.0,
        existing_worst_case_risk_money=5.0,
        proposed_worst_case_risk_money=2.0, broker_stop_verified=True,
    )
    values.update(overrides)
    return values


def test_pyramid_requires_favourable_progress_and_shared_budget():
    allowed = evaluate_layer_add(**args())
    assert allowed["eligible"] is True
    assert allowed["combined_worst_case_risk_money"] == 7.0
    assert allowed["execution_authority"] == "research_policy_only"
    refused = evaluate_layer_add(**args(proposed_worst_case_risk_money=6.0))
    assert refused["eligible"] is False
    assert refused["reason_code"] == "stack.parent_budget_exceeded"


def test_same_setup_allows_independent_entry_but_never_averages_down():
    allowed = evaluate_layer_add(**args(mode="same_setup", parent_unrealized_r=0.0))
    assert allowed["eligible"] is True
    refused = evaluate_layer_add(**args(mode="same_setup", parent_unrealized_r=-0.01))
    assert refused["reason_code"] == "stack.averaging_down_refused"


def test_pyramid_needs_favourable_parent_and_fresh_independent_trigger():
    assert evaluate_layer_add(**args(parent_unrealized_r=0.49))["reason_code"] == "stack.not_favourable_enough"
    assert evaluate_layer_add(**args(trigger_fresh=False))["reason_code"] == "stack.trigger_stale"
    assert evaluate_layer_add(**args(trigger_independent=False))["reason_code"] == "stack.correlated_trigger"
    assert evaluate_layer_add(**args(trigger_already_used=True))["reason_code"] == "stack.trigger_duplicate"


def test_layers_fail_closed_on_unknown_risk_or_unverified_stop():
    assert evaluate_layer_add(**args(risk_budget_money=None))["reason_code"] == "stack.risk_unknown"
    assert evaluate_layer_add(**args(broker_stop_verified=False))["reason_code"] == "stack.stop_unverified"
    assert evaluate_layer_add(**args(active_layers=3))["reason_code"] == "stack.layer_limit"


def test_child_identity_is_stable_and_trigger_specific():
    a = child_entry_id("parent-001", "same_setup", "retest-A")
    b = child_entry_id("parent-001", "same_setup", "retest-A")
    c = child_entry_id("parent-001", "same_setup", "retest-B")
    assert a == b
    assert a != c


def test_policy_never_exceeds_three_layers():
    try:
        StackingConfig(max_layers=4)
    except ValueError:
        pass
    else:
        raise AssertionError("stacking config accepted an unsupported fourth layer")
