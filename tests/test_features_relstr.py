"""Relative strength: ratio trend, graceful degradation."""
import numpy as np

from conftest import make_df
from goldenswing.features import relstrength
from goldenswing.features.config import FeatureConfig

CFG = FeatureConfig()


def _frame(closes, start="2024-01-01"):
    rows, prev = [], closes[0]
    for c in closes:
        rows.append((prev, max(prev, c) + 0.1, min(prev, c) - 0.1, c))
        prev = c
    return make_df(rows, start=start)


def test_rs_trend_when_gold_outruns_silver():
    n = 200
    gold = _frame((2000.0 * 1.002 ** np.arange(n)).tolist())
    silver = _frame((25.0 * 1.0005 ** np.arange(n)).tolist())
    out = relstrength.compute(gold, [None] * n, CFG, {"silver": silver})
    last = out[-1]
    assert last.available and last.silver.available
    assert last.silver.trend == 1
    assert last.score == 1.0
    assert last.silver.percentile > 80.0


def test_rs_missing_peer_degrades_not_invents():
    n = 100
    gold = _frame([2000.0 + i for i in range(n)])
    out = relstrength.compute(gold, [None] * n, CFG, {})
    assert all(not s.available for s in out)
    assert all(s.silver.ratio != s.silver.ratio for s in out)   # NaN


def test_rs_short_peer_history_stays_unavailable_early():
    n = 200
    gold = _frame((2000.0 + np.arange(n)).tolist())
    silver = _frame((25.0 + 0.01 * np.arange(50)).tolist(),
                    start="2024-01-07 06:00")   # starts ~150 bars later
    out = relstrength.compute(gold, [None] * n, CFG, {"silver": silver})
    assert not out[10].silver.available
