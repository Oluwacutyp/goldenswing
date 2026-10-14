"""Classical patterns from confirmed swings: double top + H&S,
formation bars, neckline breaks, invalidation."""
from conftest import closes_df
from goldenswing.features import patterns as pat_mod
from goldenswing.features.config import FeatureConfig
from goldenswing.reader import MarketReader

CFG = FeatureConfig()


def _run(closes):
    df = closes_df(closes, spread=0.3)
    states = MarketReader().read(df)
    return pat_mod.compute(df, states, CFG)


def _zigzag(*points, step=4):
    """Piecewise-linear close path through the given anchor prices."""
    closes = []
    for a, b in zip(points[:-1], points[1:]):
        for k in range(step):
            closes.append(a + (b - a) * k / step)
    closes.append(points[-1])
    return closes


def test_double_top_forms_then_breaks():
    closes = _zigzag(2000, 2010, 2000, 2010, 2004, 1996, 1992)
    out = _run(closes)
    kinds = [(s.event.kind, s.event.status) for s in out if s.event]
    assert ("double_top", "active") in kinds
    assert ("double_top", "broken") in kinds
    formed = next(s.event for s in out
                  if s.event and s.event.kind == "double_top")
    assert formed.direction == -1
    assert formed.neckline < 2005      # the valley between the twins
    assert formed.target < formed.neckline


def test_head_and_shoulders_forms():
    closes = _zigzag(2000, 2008, 2000, 2016, 2001, 2008, 2004)
    out = _run(closes)
    formed = [s.event for s in out
              if s.event and s.event.kind == "head_shoulders"]
    assert formed, "H&S must form on the right shoulder's confirmation"
    hs = formed[0]
    assert hs.direction == -1
    assert hs.extreme > 2014           # the head
    assert hs.neckline <= 2001.5       # lower intervening low


def test_asymmetric_shoulders_do_not_form_hs():
    closes = _zigzag(2000, 2014, 2000, 2016, 2001, 2004, 2002)
    out = _run(closes)
    assert not any(s.event and s.event.kind == "head_shoulders"
                   for s in out)
