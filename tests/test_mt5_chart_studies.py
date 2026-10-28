"""Chart studies on MT5 signal charts: the four overlays the alert
chart now carries, and the zone-conversion helper behind them.

Studies are caller-supplied data, never invented by the renderer:
nearby cache zones, the feature pipeline's auto-Fib + dealing range,
and killzone session shading from the panels' own timestamps.
"""
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from goldenswing.charts.engine import (  # noqa: E402
    _mt5_edge_labels,
    _mt5_fib,
    _mt5_nearby_lines,
    _mt5_premium_discount,
    _mt5_sessions,
    _mt5_trade_levels,
    render_mt5_signal,
    zones_to_study_dicts,
)

rng = np.random.default_rng(7)


def _frame(n, start, freq):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    walk = 4000.0 + np.cumsum(rng.normal(0, 3, n))
    return pd.DataFrame({
        "open": walk, "high": walk + 4.0, "low": walk - 4.0,
        "close": walk + rng.normal(0, 1, n), "volume": 100.0,
    }, index=idx)


def _frames():
    return (_frame(200, "2026-10-26", "h"),
            _frame(100, "2026-10-20", "4h"),
            _frame(100, "2026-10-28", "15min"))


ZONE = {"top": 4020.0, "bottom": 4000.0, "kind": "support"}


def test_zones_to_study_dicts_nearest_first_capped_and_deduped():
    zones = [SimpleNamespace(level=5000.0, tier="MAJOR", side="resistance"),
             SimpleNamespace(level=4010.0, tier="MINOR", side="support"),
             SimpleNamespace(level=3990.0, tier="MAJOR", side="support"),
             SimpleNamespace(level=4005.0, tier="noise", side="support")]
    # Band [4000, 4008]: 4005 sits inside it -> skipped; 4010 does not.
    band = {"top": 4008.0, "bottom": 4000.0, "kind": "support"}
    out = zones_to_study_dicts(zones, 4015.0, band, cap=2)
    assert [d["level"] for d in out] == [4010.0, 3990.0]
    assert out[0]["tier"] == "MINOR" and out[0]["side"] == "support"


def test_zones_to_study_dicts_tolerates_missing_attrs():
    zones = [SimpleNamespace(level=4010.0),
             SimpleNamespace(level=0.0, tier="MAJOR", side="support"),
             None]
    out = zones_to_study_dicts(zones, 4015.0, None)
    assert [d["level"] for d in out] == [4010.0]
    assert zones_to_study_dicts([], 4015.0, ZONE) == []
    assert zones_to_study_dicts(None, 4015.0, ZONE) == []


def test_render_with_all_studies_is_a_png():
    h1, h4, m15 = _frames()
    png = render_mt5_signal(
        h1, zone=ZONE, entry=4012.0, stop=3995.0, targets=[4040.0],
        direction=1, verdict={"entry_clock": "M15"}, playbook="test",
        score=80.0, h4=h4, entry_frame=m15,
        nearby_zones=[{"level": 4050.0, "tier": "MAJOR",
                       "side": "resistance"},
                      {"level": 3980.0, "tier": "MINOR",
                       "side": "support"}],
        fib_levels=[(0.382, 4030.0), (0.5, 4020.0), (0.618, 4010.0)],
        fib_range=(3990.0, 4060.0), ote_band=(4005.0, 4018.0))
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10_000


def test_render_without_studies_unchanged_back_compat():
    """Old callers pass no studies: same chart as before, no crash."""
    h1, h4, m15 = _frames()
    png = render_mt5_signal(
        h1, zone=ZONE, entry=4012.0, stop=3995.0, targets=[4040.0],
        direction=1, verdict={"entry_clock": "M15"}, playbook="test",
        score=80.0, h4=h4, entry_frame=m15)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10_000


def test_render_partial_studies_do_not_crash():
    h1, _, _ = _frames()
    png = render_mt5_signal(
        h1, zone=None, entry=4012.0, stop=3995.0, direction=-1,
        fib_levels=[(0.5, 4020.0)])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_study_draw_helpers_add_artists():
    fig, ax = plt.subplots()
    before_lines = len(ax.lines)
    _mt5_nearby_lines(ax, 10.0, [{"level": 4010.0, "tier": "MAJOR",
                                  "side": "resistance"}], {"up": "g",
                                                          "down": "r",
                                                          "text": "k"})
    assert len(ax.lines) == before_lines + 1
    before_lines = len(ax.lines)
    _mt5_fib(ax, 10.0, [(0.5, 4020.0)], (4005.0, 4018.0),
             {"text": "k"})
    assert len(ax.lines) == before_lines + 1
    assert len(ax.patches) >= 1  # OTE band span
    plt.close(fig)
    fig, ax = plt.subplots()
    _mt5_premium_discount(ax, -0.6, 10.0, 3990.0, 4060.0, {"text": "k"})
    assert len(ax.patches) >= 2  # premium + discount spans
    plt.close(fig)


def test_clustered_right_edge_labels_do_not_overlap():
    """2026-10-28 qm_m5 chart: trade + study labels sharing the right
    edge must not pile up. Step 7 de-collided trade labels only;
    Fib/zone labels drawn raw at the same edge still collided with
    them (entry 4040.5 / fill 4040.3 / stop 4035.9 / broker 4035.2
    plus Fib and zone levels within a few points). All sets now go
    through one post-draw pass, so no two labels overlap in pixels.
    """
    theme = {"up": "g", "down": "r", "text": "k"}
    fig, ax = plt.subplots()
    ax.set_xlim(-0.6, 22.4)
    ax.set_ylim(4030.0, 4055.0)
    x_hi = 22.4
    trade = _mt5_trade_levels(
        ax, x_hi, entry=4040.5, stop=4035.9, broker_stop=4035.2,
        targets=[4046.2], direction=1, risk=5.3, fill_price=4040.3,
        theme=theme)
    zones = _mt5_nearby_lines(
        ax, x_hi, [{"level": 4042.0, "tier": "MAJOR",
                    "side": "resistance"},
                   {"level": 4037.0, "tier": "MINOR",
                    "side": "support"}],
        theme)
    fibs = _mt5_fib(ax, x_hi, [(0.618, 4041.0), (0.786, 4038.5)],
                    (4048.0, 4050.0), theme)
    fig.canvas.draw()
    before = len(ax.texts)
    _mt5_edge_labels(ax, x_hi, trade + zones + fibs)
    fig.canvas.draw()
    labels = ax.texts[before:]
    assert len(labels) == len(trade) + len(zones) + len(fibs)
    renderer = fig.canvas.get_renderer()
    boxes = sorted((t.get_window_extent(renderer=renderer).y0,
                    t.get_window_extent(renderer=renderer).y1)
                   for t in labels)
    for (_, prev_top), (next_bot, _) in zip(boxes, boxes[1:]):
        assert next_bot >= prev_top - 0.5
    plt.close(fig)


def test_render_with_clustered_levels_is_a_png():
    """Full render path with the clustered Oct 28 shape: no crash,
    valid PNG."""
    h1, h4, m15 = _frames()
    png = render_mt5_signal(
        h1, zone=None, entry=4040.5, stop=4035.9, targets=[4046.2],
        direction=1, verdict={"entry_clock": "M15"}, playbook="qm_m5",
        decision_score=27.0, broker_stop=4035.2, fill_price=4040.3,
        h4=h4, entry_frame=m15,
        nearby_zones=[{"level": 4042.0, "tier": "MAJOR",
                       "side": "resistance"},
                      {"level": 4037.0, "tier": "MINOR",
                       "side": "support"}],
        fib_levels=[(0.618, 4041.0), (0.786, 4038.5)],
        fib_range=(3990.0, 4060.0), ote_band=(4048.0, 4050.0))
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10_000


def test_session_shading_spans_a_weekday():
    """A full weekday of hourly bars always crosses killzones."""
    fig, ax = plt.subplots()
    view = _frame(24, "2026-10-28", "h")
    _mt5_sessions(ax, view, {"text": "k"})
    assert len(ax.patches) >= 1
    plt.close(fig)
