"""Chart engine: all 5 styles render real, non-blank, deterministic
PNGs from a real reader state + signal."""
import io

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest

from goldenswing import charts
from goldenswing.features.engine import FeatureEngine
from goldenswing.golden.detector import GoldenSignal
from goldenswing.reader import MarketReader


def _series(n=420):
    rng = np.random.default_rng(42)
    steps = rng.normal(0.15, 1.0, n).cumsum()
    wave = 6.0 * np.sin(np.arange(n) / 23.0)
    close = 2000.0 + steps + wave
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0.2, 1.2, n)
    low = np.minimum(open_, close) - rng.uniform(0.2, 1.2, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close,
                         "volume": rng.uniform(500, 3000, n)}, index=idx)


@pytest.fixture(scope="module")
def world():
    df = _series()
    states = MarketReader().read(df)
    fsets = FeatureEngine().compute(df, states)
    for s, fs in zip(states, fsets):
        s.features = fs
    st = states[-1]
    sig = GoldenSignal(
        index=st.index, timestamp=st.timestamp, direction=1, score=78.0,
        kind="continuation",
        factors=["trend-aligned continuation (up)",
                 "liquidity sweep of sell-side pool @ 1,995.0, 2 bars ago",
                 "strong displacement 1.4x ATR"],
        entry_zone=(st.close - 2.0, st.close),
        entry_ref=st.close, invalidation=st.close - 9.0,
        targets=[st.close + 12.0, st.close + 25.0], atr=max(st.atr, 1.0))
    return df, states, fsets, st, sig


@pytest.mark.parametrize("style", charts.STYLES)
def test_each_style_renders_real_png(world, style):
    df, _states, fsets, st, sig = world
    png = charts.render(df, st, sig, style=style, features=fsets,
                        forecast=charts.ForecastNote(
                            prob_direction=0.61, rank_pct=88.0))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    img = mpimg.imread(io.BytesIO(png))
    if style == "signal":
        # phone-first portrait
        assert img.shape[0] == 1500 and img.shape[1] == 1200
    else:
        assert img.shape[0] == 1000 and img.shape[1] == 1600
    # non-blank: real ink on the page -> meaningful pixel variance
    assert float(img.std()) > 0.02
    assert len(png) > 20_000


def test_signal_style_deterministic_and_signal_free(world):
    df, _states, fsets, st, sig = world
    a = charts.render(df, st, sig, style="signal")
    b = charts.render(df, st, sig, style="signal")
    assert a == b
    # no-signal variant (market read) also renders
    png = charts.render(df, st, None, style="signal")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 20_000


def test_signal_style_is_the_alert_default():
    from goldenswing.alerts.service import AlertService
    svc = AlertService([], chart_renderer=lambda *a, **k: b"png")
    assert svc.chart_style == "signal"


def test_render_is_deterministic(world):
    df, _states, fsets, st, sig = world
    a = charts.render(df, st, sig, style="blueprint", features=fsets)
    b = charts.render(df, st, sig, style="blueprint", features=fsets)
    assert a == b


def test_render_without_signal_or_features(world):
    df, _states, _fsets, st, _sig = world
    png = charts.render(df, st, None, style="classic")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 10_000


def test_unknown_style_rejected(world):
    df, _states, _fsets, st, sig = world
    with pytest.raises(ValueError):
        charts.render(df, st, sig, style="watercolor")


def test_heikin_ashi_math():
    df = pd.DataFrame(
        {"open": [10.0, 11.0], "high": [12.0, 13.0],
         "low": [9.0, 10.0], "close": [11.0, 12.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="1h", tz="UTC"))
    ha = charts.heikin_ashi(df)
    assert ha["close"].iloc[0] == pytest.approx((10 + 12 + 9 + 11) / 4)
    assert ha["open"].iloc[0] == pytest.approx((10 + 11) / 2)
    assert ha["open"].iloc[1] == pytest.approx(
        (ha["open"].iloc[0] + ha["close"].iloc[0]) / 2)
    assert ha["high"].iloc[1] >= max(ha["open"].iloc[1], ha["close"].iloc[1])


def test_forecast_note_line():
    assert charts.ForecastNote().line() is None
    line = charts.ForecastNote(prob_direction=0.61, rank_pct=88.0).line()
    assert "0.61" in line and "top 12%" in line
