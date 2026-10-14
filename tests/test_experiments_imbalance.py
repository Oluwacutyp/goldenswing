"""Tests for the imbalance (FVG) deep-mining experiment."""
import numpy as np
import pandas as pd
import pytest

from goldenswing.experiments import imbalance as I


def _df(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.004, n)
    close = 2000 * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0.002, 0.001, n)) * close
    o = close * (1 + rng.normal(0, 0.0005, n))
    h = np.maximum(o, close) + spread / 2
    l = np.minimum(o, close) - spread / 2
    v = rng.integers(100, 1000, n).astype(float)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": close,
                         "volume": v})


def _gap_df() -> pd.DataFrame:
    """Deterministic bars containing one known bullish FVG at t=10:
    low[10]=101 > high[8]=98.5.  Background bars live below the gap."""
    n = 60
    o = np.full(n, 98.0)
    h = np.full(n, 98.5)
    l = np.full(n, 97.5)
    c = np.full(n, 98.0)
    # displacement candle 9
    o[9], h[9], l[9], c[9] = 98.0, 103.0, 98.2, 102.8
    # candle 10 leaves the gap: low[10] > high[8]
    o[10], h[10], l[10], c[10] = 102.8, 104.0, 101.0, 103.5
    # bars 11..19 hover above the gap
    for j in range(11, 20):
        o[j], h[j], l[j], c[j] = 103.0, 103.8, 102.0, 103.2
    # candle 20 wick-touches the gap (low <= bottom=98.5? no: bottom is
    # high[8]=98.5 — touch needs low <= 98.5; use a proper touch at 98.0)
    o[20], h[20], l[20], c[20] = 102.5, 102.8, 98.0, 102.0
    v = np.full(n, 500.0)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": v})


def test_detect_bullish_gap_geometry():
    df = _gap_df()
    gaps = I.detect_gaps(df)
    g = next(g for g in gaps if g.created_at == 10)
    assert g.direction == 1
    assert g.bottom == pytest.approx(98.5)    # high[8]
    assert g.top == pytest.approx(101.0)      # low[10]
    assert g.mid == pytest.approx(99.75)


def test_wick_touch_vs_inversion_classification():
    df = _gap_df()
    gaps = I.detect_gaps(df)
    g = next(g for g in gaps if g.created_at == 10)
    lives = I.replay_lives(df, [g])
    lf = lives[0]
    assert lf.first_touch_index == 20
    assert lf.end_kind == "wick"   # close 102 > bottom 98.5


def test_inversion_needs_body_close():
    df = _gap_df()
    df.loc[20, "close"] = 98.0    # close through the gap (below bottom 98.5)
    gaps = I.detect_gaps(df)
    g = next(g for g in gaps if g.created_at == 10)
    lives = I.replay_lives(df, [g])
    assert lives[0].end_kind == "inversion"


def test_detection_is_causal_under_truncation():
    df = _df(400)
    full = I.detect_gaps(df)
    cut = I.detect_gaps(df.iloc[:250])
    # gaps created before bar 240 must be identical in both runs
    full_early = [(g.created_at, g.direction, round(g.top, 6), round(g.bottom, 6))
                  for g in full if g.created_at < 240]
    cut_early = [(g.created_at, g.direction, round(g.top, 6), round(g.bottom, 6))
                 for g in cut if g.created_at < 240]
    assert full_early == cut_early


def test_magnet_stats_shape():
    df = _df(500)
    lives = I.replay_lives(df, I.detect_gaps(df))
    mag = I.magnet_stats(lives)
    assert "fill_within_24" in mag
    hit, n, r = mag["fill_within_24"]
    assert 0.0 <= r <= 1.0 and n == len(lives)


def test_ce_vs_edge_runs():
    df = _gap_df()
    gaps = I.detect_gaps(df)
    lives = I.replay_lives(df, gaps)
    atr = I._atr(df["high"].to_numpy(), df["low"].to_numpy(),
                 df["close"].to_numpy(), 14)
    res = I.ce_vs_edge(df, lives, atr)
    kinds = {e.entry_kind for e in res}
    assert kinds <= {"mid", "edge"}
    # every result carries a finite R
    assert all(np.isfinite(e.r_mult) for e in res)


def test_ifvg_drift_shape():
    df = _df(600)
    lives = I.replay_lives(df, I.detect_gaps(df))
    d = I.ifvg_drift(df, lives)
    assert "n_inversions" in d and "h24" in d
    assert 0.0 <= d["h24"]["hit_rate"] <= 1.0


def test_stacking_runs():
    df = _df(600)
    gaps = I.detect_gaps(df)
    lives = I.replay_lives(df, gaps)
    s = I.stacking_stats(df, gaps, lives)
    assert set(s) <= {"stacked", "single"}
    for b in s.values():
        assert 0.0 <= b["hold_rate"] <= 1.0
