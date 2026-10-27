"""Step 6 (2026-10-27): cost-inclusive replay scoring.

Outcomes are bid/ask first-touches, R is measured from the
spread-adjusted entry, expectancy is over resolved candidates
only, and expectancy beats win rate as the headline number.
"""

import pandas as pd

from scripts.score_hot_scalp_replay import score_candidates


def _ticks(rows):
    idx = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.DataFrame({"bid": [r[1] for r in rows],
                         "ask": [r[2] for r in rows]}, index=idx)


def _cand(**kw):
    d = {"trigger_ts": "2026-10-27T00:50:30+00:00", "direction": 1,
         "entry_reference": 100.0, "structural_stop": 99.0,
         "target_reference": 102.0, "spread": 0.0,
         "strategy": "breakout_retest"}
    d.update(kw)
    return d


def test_long_win_scores_full_reward_risk():
    ticks = _ticks([("2026-10-27T00:51:00Z", 101.0, 101.02),
                    ("2026-10-27T00:52:00Z", 102.0, 102.02)])
    result = score_candidates([_cand()], ticks)
    assert result["totals"]["wins"] == 1
    assert result["totals"]["expectancy_r"] == 2.0


def test_long_loss_scores_minus_one():
    ticks = _ticks([("2026-10-27T00:51:00Z", 99.0, 99.02)])
    result = score_candidates([_cand()], ticks)
    assert result["totals"]["losses"] == 1
    assert result["totals"]["expectancy_r"] == -1.0


def test_short_win_uses_ask_touch():
    ticks = _ticks([("2026-10-27T00:51:00Z", 98.5, 98.0)])
    result = score_candidates(
        [_cand(direction=-1, entry_reference=100.0,
               structural_stop=101.0, target_reference=98.0)], ticks)
    assert result["totals"]["wins"] == 1
    assert result["totals"]["expectancy_r"] == 2.0


def test_open_excluded_from_expectancy():
    ticks = _ticks([("2026-10-27T00:51:00Z", 100.5, 100.52)])
    result = score_candidates(
        [_cand(), _cand(trigger_ts="2026-10-27T00:50:31+00:00")], ticks)
    # First candidate: neither level touched in horizon -> open.
    # Second candidate has the same forward window (trigger 1s
    # later) -> also open.  Expectancy stays None, not zero.
    assert result["totals"]["open"] == 2
    assert result["totals"]["expectancy_r"] is None
    assert result["totals"]["win_rate"] is None


def test_spread_shrinks_reward_risk():
    # Spread 0.20: long pays 100.10, so reward is 1.90 over a
    # 1.10 risk — R 1.727, not the mid-price 2.0.
    ticks = _ticks([("2026-10-27T00:51:00Z", 102.0, 102.02)])
    result = score_candidates([_cand(spread=0.20)], ticks)
    assert result["totals"]["expectancy_r"] == round(1.90 / 1.10, 6) \
        or abs(result["totals"]["expectancy_r"] - 1.90 / 1.10) < 1e-9


def test_expectancy_and_per_strategy_breakdown():
    win_ticks = _ticks([("2026-10-27T00:51:00Z", 102.0, 102.02)])
    loss_ticks = _ticks([("2026-10-27T00:51:00Z", 99.0, 99.02)])
    result = score_candidates(
        [_cand(strategy="breakout_retest"),
         _cand(strategy="sweep_reclaim")], win_ticks)
    # Both candidates share the winning ticks: 2 wins, expectancy 2.
    assert result["totals"]["expectancy_r"] == 2.0
    assert result["per_strategy"]["breakout_retest"]["wins"] == 1
    assert result["per_strategy"]["sweep_reclaim"]["wins"] == 1
    result2 = score_candidates([_cand()], loss_ticks)
    assert result2["totals"]["expectancy_r"] == -1.0
