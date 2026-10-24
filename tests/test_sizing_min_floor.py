"""Minimum-lot floor pins (owner ruling 2026-10-24).

A setup is account-agnostic: size scales with equity, and when
the stop risk exceeds the balance the engine trades the broker
MINIMUM (0.01 lot), never abandons the trade.  The only hard
number is the broker minimum; ``decline_below_min`` and
``risk_pct`` as a refusal gate are dead.

All pins use the gold spec (tick $0.01, tick value $1, so a $10
stop risks $1,000 per 1.0 lot and a $5 stop risks $500) with
C = 1.0 and S = 1.0 isolated — the formula inputs are
risk_money = equity x k, k = 0.01 (proposed coefficient, not a
gate), and lots = max(0.01, floor_down(raw, 0.01)).

These are unit pins on size_for_risk + the journal fields.  The
runner-level wiring (band removal at both order paths, layer
headroom rule, _sizing_detail journal shape, display strings)
is proven by the diff + post-change greps in the implementation
report — the MT5 runner needs a terminal harness and places no
orders in unit tests.  Demo authority stays OFF.
"""

from decimal import Decimal

from goldenswing.execution.risk import gold_spec
from goldenswing.live.sizing import D, VolumeConstraints, size_for_risk

K = Decimal("0.01")


def _risk(equity):
    return Decimal(str(equity)) * K


def test_min_floor_table_small_accounts_trade_minimum():
    spec = gold_spec()
    # $12 at a $10 stop: risk 0.12, raw 0.00012 -> 0.01 min floor.
    r = size_for_risk(_risk(12), 10.0, spec)
    assert r.ok and r.lots == Decimal("0.01") and r.min_override
    assert r.raw_lots == Decimal("0.00012")
    assert r.estimated_stop_exposure == Decimal("10")
    # $24 at a $10 stop: risk 0.24, raw 0.00024 -> 0.01 min floor.
    r = size_for_risk(_risk(24), 10.0, spec)
    assert r.ok and r.lots == Decimal("0.01") and r.min_override
    assert r.raw_lots == Decimal("0.00024")
    assert r.estimated_stop_exposure == Decimal("10")
    # $1,000 at a $10 stop: risk 10, raw exactly 0.01 -> NOT an
    # override (raw == minimum is a normal size, not a floor).
    r = size_for_risk(_risk(1000), 10.0, spec)
    assert r.ok and r.lots == Decimal("0.01") and not r.min_override
    assert r.raw_lots == Decimal("0.01")
    # $10,000 at a $10 stop: risk 100 -> 0.10, no override.
    r = size_for_risk(_risk(10000), 10.0, spec)
    assert r.ok and r.lots == Decimal("0.10") and not r.min_override
    assert r.raw_lots == Decimal("0.10")


def test_min_floor_stop_responsive():
    # Same equity, tighter stop -> bigger size (stop distance in
    # the denominator), still floored at the minimum when small.
    spec = gold_spec()
    wide = size_for_risk(_risk(1000), 10.0, spec)
    tight = size_for_risk(_risk(1000), 5.0, spec)
    assert wide.lots == Decimal("0.01")
    assert tight.lots == Decimal("0.02")
    assert tight.estimated_stop_exposure == Decimal("10")


def test_min_floor_drawdown_boundary_case():
    # The unavoidable consequence, journaled honestly: $12 equity
    # with a $10 stop trades 0.01 lots whose estimated stop
    # exposure is $10 — about 83% of equity.  The 15% drawdown
    # halt stops FURTHER trading after losses; it cannot bound
    # this single trade's exposure.  Nominal stop exposure also
    # cannot guarantee the final loss (gaps/slippage).
    spec = gold_spec()
    r = size_for_risk(_risk(12), 10.0, spec)
    assert r.ok and r.lots == Decimal("0.01") and r.min_override
    assert r.estimated_stop_exposure == Decimal("10")
    assert r.estimated_stop_exposure / Decimal("12") > Decimal("0.8")


def test_min_floor_flag_boundary():
    # min_override is True EXACTLY when raw < minimum.  raw ==
    # minimum and raw > minimum are normal sizes.
    spec = gold_spec()
    below = size_for_risk(Decimal("5"), 10.0, spec)   # raw 0.005
    assert below.min_override is True
    equal = size_for_risk(Decimal("10"), 10.0, spec)  # raw 0.01
    assert equal.min_override is False
    above = size_for_risk(Decimal("29"), 1.0, spec)   # raw 0.29
    assert above.min_override is False
    assert above.lots == Decimal("0.29")  # floored down, not up


def test_min_floor_drag_negative_exactly_on_override():
    # drag = authorized - realized.  On a floor the realized
    # exposure EXCEEDS the authorization, so drag goes negative
    # — the honest arithmetic, flagged by min_override.
    spec = gold_spec()
    r = size_for_risk(Decimal("5"), 10.0, spec)
    assert r.drag == Decimal("-5")
    normal = size_for_risk(Decimal("29"), 1.0, spec)
    assert normal.drag == Decimal("0")  # 0.29 lots x $100 = $29


def test_min_floor_volume_limit_and_nonpositive_unchanged():
    # The floor does NOT touch broker-constraint declines: a
    # VOLUME_LIMIT that cannot fit the minimum still declines,
    # and nonpositive inputs still decline.
    cons = VolumeConstraints(min_volume=D("0.01"),
                             max_volume=D("50"), step=D("0.01"),
                             limit=D("0.05"))
    r = size_for_risk(Decimal("10000"), 10.0, gold_spec(),
                      constraints=cons, open_volume=D("0.05"))
    assert not r.ok and r.verdict == "decline_volume_limit"
    assert r.lots is None and r.min_override is False
    r2 = size_for_risk(Decimal("0"), 10.0, gold_spec())
    assert not r2.ok and r2.verdict == "decline_nonpositive"
    r3 = size_for_risk(Decimal("10"), 0.0, gold_spec())
    assert not r3.ok and r3.verdict == "decline_nonpositive"


def test_min_floor_estimated_exposure_is_realized_risk_alias():
    # Journal terminology: the order-time figure is ESTIMATED
    # stop exposure; actual realized P&L exists only at close.
    # The alias must equal the carried value, never a new number.
    spec = gold_spec()
    r = size_for_risk(Decimal("5"), 10.0, spec)
    assert r.estimated_stop_exposure == r.realized_risk == Decimal("10")
