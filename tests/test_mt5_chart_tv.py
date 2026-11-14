"""TradingView Lightweight Charts rung — pins for the renderer
ladder (tv_page + screenshot + the engine dispatcher).

The truth contracts themselves are pinned in
test_mt5_chart_roles.py against the canonical model; this file
pins the MEDIUM: the vendored library, the page config carrying
the same truth into Lightweight Charts, the portrait geometry
of the TV rung, the matplotlib fallback rung's honesty stamp,
and the signal view consuming the same builder.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenswing.charts import engine, screenshot, tv_page  # noqa: E402
from goldenswing.charts.signal_model import (  # noqa: E402
    lifecycle_from_events)
from goldenswing.charts.signal_view import (  # noqa: E402
    build_signal_view_html)

import matplotlib.image as plt  # noqa: E402


def _frame(n, freq, t0, base=4100.0, drift=0.3):
    idx = pd.date_range(t0, periods=n, freq=freq, tz="UTC")
    close = base + np.cumsum(np.full(n, drift)) + \
        np.sin(np.arange(n) * 0.7) * 1.2
    open_ = np.roll(close, 1)
    open_[0] = close[0] - 0.2
    vol = 100.0 + (np.arange(n) % 7) * 13.0
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + 0.5,
         "low": np.minimum(open_, close) - 0.5, "close": close,
         "volume": vol, "real_volume": 0, "spread": 30}, index=idx)


def _lifecycle(**over):
    m5 = _frame(64, "5min", "2026-11-05 09:20", base=4127.0,
                drift=0.12)
    params = dict(
        direction=1, entry_ref=4134.507, structural_stop=4131.469,
        targets=(4136.181, 4136.296, 4138.671),
        entry_zone=(4132.01, 4135.883), state="closed",
        ticket=540274935, volume=0.01, fill_price=4134.437,
        initial_broker_stop=4131.469,
        events=[{"kind": "open", "ts": m5.index[38].timestamp(),
                 "price": 4134.437},
                {"kind": "sl", "ts": m5.index[47].timestamp(),
                 "level": 4135.92, "reason": "profit_lock"}],
        partials=[], exit_price=4135.797,
        exit_ts=m5.index[49].timestamp(),
        exit_reason="SL hit (profit_lock)")
    params.update(over)
    return lifecycle_from_events(**params), m5


def _cfg(page_html: str) -> dict:
    start = page_html.index('<script id="chart-cfg" '
                            'type="application/json">')
    start = page_html.index(">", start) + 1
    end = page_html.index("</script>", start)
    return json.loads(page_html[start:end])


def _roles(lc=None, frames=None, **kw):
    lc = lc if lc is not None else _lifecycle()[0]
    frames = frames if frames is not None else {
        "M5": _lifecycle()[1],
        "H1": _frame(200, "h", "2026-10-26", base=4050.0, drift=0.35)}
    params = dict(frames=frames, lifecycle=lc, instrument="XAUUSDm",
                  playbook="hot_scalp", score_line="Score 79",
                  entry_clock="M5")
    params.update(kw)
    return params


# ------------------------------------------------------------ vendoring
def test_vendored_library_is_pinned_and_licensed():
    lib = tv_page.ASSETS_DIR / "lightweight-charts.standalone.production.js"
    assert lib.exists() and lib.stat().st_size > 100_000
    digest = hashlib.sha256(lib.read_bytes()).hexdigest()
    assert digest == ("bcdca2a528db7c9b386918c99c544bde3eda3f0204"
                      "ec2d23a64411d4cb4686c9")
    assert tv_page.LWC_VERSION == "5.0.8"
    assert "5.0.8" in tv_page.lwc_js()
    lic = (tv_page.ASSETS_DIR / "LICENSE.lightweight-charts.txt")
    assert lic.exists() and "Apache" in lic.read_text()


# ------------------------------------------------------------ page truth
def test_trade_page_config_carries_the_truth():
    lc, _m5 = _lifecycle()
    page = tv_page.build_role_pages(**_roles(lc))[0]
    cfg = _cfg(page)
    titles = [ln["title"] for ln in cfg["lines"]]
    assert any(t.startswith("STOP 4,131.5 (structural)")
               for t in titles)
    assert any(t.startswith("T1 4,136.2 (planned)") for t in titles)
    assert any(t.startswith("FILL 4,134.4") for t in titles)
    # the broker stop history is a stepped series through BOTH
    # recorded levels — never a single line at the latest stop
    stepped = [s for s in cfg["lineSeries"] if s.get("stepped")]
    assert len(stepped) == 1
    values = [p["value"] for p in stepped[0]["points"]]
    assert values[0] == pytest.approx(4131.469)
    assert 4135.92 in values
    # zone band + volume travel with the page
    assert any(b["top"] == pytest.approx(4135.883)
               for b in cfg["bands"])
    assert cfg["volume"], "volume strip data must be present"
    assert any(m.get("text") == "FILL" for m in cfg["markers"])
    assert any(m.get("text") == "EXIT" for m in cfg["markers"])


def test_dense_ladder_top_is_an_edge_tag_not_an_axis_stretch():
    ladder = [4101.3 + 1.5 * i for i in range(12)] + \
             [4200.2 + 8.1 * i for i in range(12)]
    m15 = _frame(96, "15min", "2026-11-04 12:00", base=4110.0,
                 drift=0.25)
    lc = lifecycle_from_events(
        direction=1, entry_ref=4098.3, structural_stop=4084.3,
        targets=tuple(ladder), entry_zone=(4095.9, 4099.7),
        state="open", ticket=540791314, volume=0.01,
        fill_price=4098.3, initial_broker_stop=4084.3,
        events=[{"kind": "open", "ts": m15.index[44].timestamp(),
                 "price": 4098.3}])
    page = tv_page.build_role_pages(
        frames={"M15": m15, "H1": _frame(200, "h", "2026-10-26",
                                         base=4050.0, drift=0.35)},
        lifecycle=lc, instrument="XAUUSDm", playbook="hot_scalp",
        score_line="Score 81", entry_clock="M15")[0]
    cfg = _cfg(page)
    prices = [ln["price"] for ln in cfg["lines"]]
    assert 4289.3 not in prices
    assert cfg["priceRange"][1] < 4200.0
    tags = [t["text"] for t in cfg["edgeTags"]]
    assert any("T24 4,289.3 (planned)" in t for t in tags)
    titles = [ln["title"] for ln in cfg["lines"]]
    assert sum(1 for t in titles if "(planned)" in t) >= 3


def test_unavailable_role_page_is_labeled_not_invented():
    page = tv_page.build_role_pages(**_roles(frames={
        "H1": _frame(60, "h", "2026-11-03")}))[0]
    assert "frame unavailable" in page
    assert "chart-cfg" not in page


# ------------------------------------------------------------ png stamps
def test_png_text_stamp_roundtrip():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
    stamped = screenshot.stamp_png_text(
        buf.getvalue(), {"Renderer": "matplotlib-fallback"})
    assert screenshot.read_png_text(stamped)["Renderer"] == \
        "matplotlib-fallback"


def test_find_browser_shape():
    found = screenshot.find_browser()
    assert found is None or Path(found[0]).exists()


# ------------------------------------------------------------ the rungs
def test_fallback_rung_renders_portrait_and_stamps_honestly(
        monkeypatch):
    monkeypatch.setattr(screenshot, "screenshot_html",
                        lambda *a, **k: None)
    pngs = engine.render_mt5_roles(**_roles())
    assert engine.last_role_render_rungs() == ["matplotlib"] * 3
    assert engine.last_tv_error()
    for png in pngs:
        arr = plt.imread(io.BytesIO(png))
        assert (arr.shape[1], arr.shape[0]) == (tv_page.PAGE_W,
                                                tv_page.PAGE_H)
        meta = screenshot.read_png_text(png)
        assert meta.get("Renderer") == "matplotlib-fallback"


@pytest.mark.skipif(screenshot.ensure_browser() is None,
                    reason="no browser rung on this machine")
def test_tv_rung_renders_portrait_and_stamps_the_rung():
    pngs = engine.render_mt5_roles(**_roles())
    assert engine.last_tv_error() is None
    rungs = engine.last_role_render_rungs()
    assert len(rungs) == 3 and all(
        r != "matplotlib" for r in rungs)
    for png in pngs:
        arr = plt.imread(io.BytesIO(png))
        assert (arr.shape[1], arr.shape[0]) == (tv_page.PAGE_W,
                                                tv_page.PAGE_H)
        meta = screenshot.read_png_text(png)
        assert meta.get("Renderer", "").startswith(
            "lightweight-charts-5.0.8+")


# ------------------------------------------------------------ the study
def test_signal_view_consumes_the_shared_builder():
    html = build_signal_view_html(
        frames={"H1": _frame(60, "h", "2026-11-03"),
                "M5": _frame(60, "5min", "2026-11-05 08:00")},
        entry=4124.49, stop=4110.43,
        targets=[4126.0 + 1.5 * i for i in range(12)]
        + [4250.0 + 6.0 * i for i in range(12)],
        direction=1, playbook="hot_scalp",
        verdict={"entry_clock": "M5", "trigger": "FVG retest"},
        zone={"top": 4135.883, "bottom": 4132.01, "kind": "FVG"},
        nearby_zones=[], fib_levels=[], fib_range=None,
        ote_band=None, fib_info=None, alert_id="MT5-hot_scalp-2008",
        lifecycle=_lifecycle()[0])
    # one builder for PNG and study — no second renderer remains
    assert "GsChart.build" in html
    assert 'id="signal-data"' in html
    assert "getContext('2d')" not in html
    start = html.index('<script id="signal-data" '
                       'type="application/json">')
    start = html.index(">", start) + 1
    payload = json.loads(html[start:html.index("</script>", start)])
    assert payload["meta"]["targets_total"] == 24
    assert payload["meta"]["target_ranks"] == [1, 2, 3, 24]
    # bars carry volume now (6th field) for the TV volume strip
    assert len(payload["tfs"]["M5"]["bars"][0]) == 6


def test_signal_view_m1_rides_last_when_supplied():
    """The Hot Scalp M1 aux frame reaches the signal view's
    timeframe selector when real M1 bars exist (never resampled
    or fabricated), and M1 sits LAST in the selector — the finest
    clock is entry precision, not the lead tab."""
    html = build_signal_view_html(
        frames={"D1": _frame(30, "D", "2026-11-01"),
                "H4": _frame(60, "4h", "2026-11-03"),
                "H1": _frame(60, "h", "2026-11-03"),
                "M15": _frame(60, "15min", "2026-11-05 08:00"),
                "M5": _frame(60, "5min", "2026-11-05 08:00"),
                "M1": _frame(60, "min", "2026-11-05 08:00")},
        entry=4124.49, stop=4110.43,
        targets=[4126.0 + 1.5 * i for i in range(12)],
        direction=1, playbook="hot_scalp",
        verdict={"entry_clock": "M1", "trigger": "sweep reclaim"},
        zone={"top": 4135.883, "bottom": 4132.01, "kind": "FVG"},
        nearby_zones=[], fib_levels=[], fib_range=None,
        ote_band=None, fib_info=None, alert_id="MT5-hot_scalp-2009",
        lifecycle=_lifecycle()[0])
    start = html.index('<script id="signal-data" '
                       'type="application/json">')
    start = html.index(">", start) + 1
    payload = json.loads(html[start:html.index("</script>", start)])
    assert "M1" in payload["tfs"]
    assert list(payload["tfs"]) == ["D1", "H4", "H1", "M15", "M5",
                                    "M1"]
    assert 'data-tf="M1"' in html
    assert html.rindex('data-tf="M1"') > html.rindex('data-tf="M5"')
