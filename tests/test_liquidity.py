"""Liquidity pools + sweep events."""
from goldenswing.reader import MarketReader
from conftest import closes_df, make_df


def test_pool_created_at_swing_low():
    # swing low 10.8 at idx2, confirmed idx4 -> sell-side pool at 10.8
    df = closes_df([12, 13, 11, 12, 12.5, 12.2, 12.4])
    states = MarketReader().read(df)
    assert any(abs(p.price - 10.8) < 1e-9 for p in states[5].pools_below)


def test_bullish_sweep():
    # pool at 10.8 (swing low idx2); bar 6 wicks to 10.5 but closes at 12.0
    rows = [
        (11.95, 12.2, 11.8, 12.0),
        (12.95, 13.2, 12.8, 13.0),
        (10.95, 11.2, 10.8, 11.0),
        (11.95, 12.2, 11.8, 12.0),
        (12.45, 12.7, 12.3, 12.5),
        (12.15, 12.4, 12.0, 12.2),
        (11.50, 12.3, 10.5, 12.0),   # sweep bar
        (12.00, 12.4, 11.9, 12.2),
    ]
    states = MarketReader().read(make_df(rows))
    assert states[5].last_sweep is None
    sw = states[6].last_sweep
    assert sw is not None and sw.index == 6
    assert sw.direction == 1                      # swept the lows -> bullish
    assert abs(sw.level - 10.8) < 1e-9
    assert states[6].sweep_age == 0
    # the swept pool is consumed
    assert not any(abs(p.price - 10.8) < 1e-9 for p in states[6].pools_below)


def test_close_through_is_not_a_sweep():
    # bar 6 closes BELOW the pool level -> cleared, not swept
    rows = [
        (11.95, 12.2, 11.8, 12.0),
        (12.95, 13.2, 12.8, 13.0),
        (10.95, 11.2, 10.8, 11.0),
        (11.95, 12.2, 11.8, 12.0),
        (12.45, 12.7, 12.3, 12.5),
        (12.15, 12.4, 12.0, 12.2),
        (11.00, 11.4, 10.2, 10.4),   # closes through the level
        (10.40, 10.8, 10.0, 10.2),
    ]
    states = MarketReader().read(make_df(rows))
    assert states[7].last_sweep is None
