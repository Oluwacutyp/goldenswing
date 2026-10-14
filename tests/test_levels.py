"""Fair value gap creation + mitigation."""
from goldenswing.reader import MarketReader
from conftest import make_df


def _fvg_frame():
    # bars 2,3,4 form a bullish FVG: low[4]=102 > high[2]=100 -> gap (100,102)
    return make_df([
        (99.0, 99.5, 98.5, 99.0),     # 0 quiet
        (99.0, 99.6, 98.6, 99.2),     # 1 quiet
        (99.5, 100.0, 99.0, 99.8),    # 2 gap base candle
        (100.0, 103.0, 99.5, 102.8),  # 3 displacement
        (102.8, 104.0, 102.0, 103.5), # 4 gap confirmed here
        (103.5, 104.2, 99.5, 100.5),  # 5 trades down into the gap (low<=100? no, 99.5 <= 100 -> fill)
    ])


def test_bullish_fvg_detected():
    states = MarketReader().read(_fvg_frame())
    assert not any(g.direction == 1 for g in states[3].active_fvgs)
    gaps = [g for g in states[4].active_fvgs if g.direction == 1]
    assert len(gaps) == 1
    assert abs(gaps[0].bottom - 100.0) < 1e-9
    assert abs(gaps[0].top - 102.0) < 1e-9


def test_fvg_mitigated_on_fill():
    states = MarketReader().read(_fvg_frame())
    # bar 5 low = 99.5 <= gap bottom 100 -> fully mitigated, no longer active
    assert not any(g.direction == 1 for g in states[5].active_fvgs)


def test_no_fvg_when_candles_overlap():
    df = make_df([
        (100.0, 100.5, 99.5, 100.1),
        (100.1, 100.6, 99.6, 100.2),
        (100.2, 100.7, 99.7, 100.3),
        (100.3, 100.8, 99.8, 100.4),
    ])
    states = MarketReader().read(df)
    assert all(len(s.active_fvgs) == 0 for s in states)
