"""Fractal swing detection + BOS/CHoCH structure events."""
from goldenswing.reader import MarketReader
from conftest import closes_df


def test_swing_high_confirmed_with_lag():
    # peak close 13 at idx 3 -> swing high, knowable only at idx 5 (n=2)
    df = closes_df([10, 11, 12, 13, 12, 11, 10, 11, 12])
    states = MarketReader().read(df)
    assert all(abs(s.price - 13.2) > 1e-9
               for s in states[4].recent_swing_highs)
    assert any(abs(s.price - 13.2) < 1e-9 and s.index == 3
               for s in states[5].recent_swing_highs)


def test_swing_low_detection():
    df = closes_df([13, 12, 11, 10, 11, 12, 13])
    states = MarketReader().read(df)
    assert any(abs(s.price - 9.8) < 1e-9 and s.index == 3
               for s in states[5].recent_swing_lows)


def test_bos_up():
    # swing high 13.2 at idx2 (confirmed idx4); close 14 at idx6 breaks it
    df = closes_df([10, 11, 13, 12, 11.5, 12, 14, 15])
    states = MarketReader().read(df)
    assert states[5].last_structure_event is None or \
        states[5].last_structure_event.index < 6
    ev = states[6].last_structure_event
    assert ev is not None and ev.index == 6
    assert ev.kind == "BOS" and ev.direction == 1
    assert states[6].trend == "up"


def test_choch_down_after_uptrend():
    # swing high 13.2 @idx2 (confirmed idx4), swing low 10.8 @idx3
    # (confirmed idx5). Close 14 @idx5 breaks 13.2 -> BOS up.
    # Close 10 @idx7 breaks 10.8 -> CHoCH down.
    df = closes_df([11, 12, 13, 11, 12, 14, 12, 10, 8])
    states = MarketReader().read(df)
    assert states[5].trend == "up"
    ev5 = states[5].last_structure_event
    assert ev5 is not None and ev5.kind == "BOS" and ev5.direction == 1
    ev = states[7].last_structure_event
    assert ev is not None and ev.index == 7
    assert ev.kind == "CHOCH" and ev.direction == -1
    assert states[7].trend == "down"
