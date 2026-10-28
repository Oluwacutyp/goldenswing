"""Step 7 — signal page redesign: one shared model behind the PNG
chart and the HTML page.

The defects from the owner's screenshots (signal MT5-qm_m5-10017):
score showed the sleeve-base placeholder 60, only one stop showed,
the H1 panel was labeled "trigger" on a native route whose trigger
lives on M5, and ENTRY/STOP/T1 labels stacked illegibly at the right
edge. These tests pin the fix at the model seam (shared by both
renderers) and at each renderer's output.
"""
import inspect
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from goldenswing.charts.engine import render_mt5_signal
from goldenswing.charts.signal_model import (
    decollide_label_ys,
    effective_r,
    panel_role,
    score_text,
)
from goldenswing.charts.signal_view import build_signal_view_html
from goldenswing.execution.mt5_runner import MT5Runner

rng = np.random.default_rng(21)


def _frame(n, start, freq):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    walk = 4000.0 + np.cumsum(rng.normal(0, 3, n))
    return pd.DataFrame({
        "open": walk, "high": walk + 4.0, "low": walk - 4.0,
        "close": walk + rng.normal(0, 1, n), "volume": 100.0,
    }, index=idx)


def _payload(html):
    start = html.index('<script id="signal-data" type="application/json">')
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def _html(**kw):
    params = {
        "frames": {"H1": _frame(200, "2026-10-26", "h"),
                   "H4": _frame(100, "2026-10-20", "4h"),
                   "M5": _frame(100, "2026-10-28", "5min")},
        "entry": 4057.9, "stop": 4055.9, "targets": [4062.7],
        "direction": 1, "playbook": "qm_m5",
        "verdict": {"entry_clock": "M5", "bias_d1": 1, "bias_h4": 1,
                    "trigger": "sweep_reversal"},
        "zone": {"top": 4060.0, "bottom": 4050.0, "kind": "support"},
        "nearby_zones": [], "fib_levels": [], "fib_range": None,
        "ote_band": None, "fib_info": None, "alert_id": "MT5-qm_m5-10017",
    }
    params.update(kw)
    return build_signal_view_html(**params)


# --- shared model -------------------------------------------------------

def test_panel_role_native_route_h1_is_context_not_trigger():
    assert panel_role("M5", playbook="qm_m5", entry_clock="M5") == "entry"
    assert panel_role("H1", playbook="qm_m5", entry_clock="M5") == "context"
    assert panel_role("H4", playbook="qm_m5", entry_clock="M5") == "structure"
    assert panel_role("D1", playbook="qm_d1", entry_clock="D1") == "entry"
    assert panel_role("H1", playbook="qm_d1", entry_clock="D1") == "context"


def test_panel_role_h1_route_h1_is_trigger():
    # Real H1-roster shape: the trigger lives on H1, the entry clock
    # is M15 — H1 is the trigger panel, M15 the entry panel.
    assert panel_role("H1", playbook="sweep_reversal",
                      entry_clock="M15") == "trigger"
    assert panel_role("M15", playbook="sweep_reversal",
                      entry_clock="M15") == "entry"
    assert panel_role("H4", playbook="sweep_reversal",
                      entry_clock="M15") == "structure"


def test_score_text_never_defaults_to_placeholder():
    assert score_text(32.71) == "Score 33"
    assert score_text(None) == "Score unavailable"
    assert score_text(None, fallback=80.0) == "Score 80"
    assert score_text(32.71, fallback=80.0) == "Score 33"


def test_effective_r_on_broker_stop_only():
    assert effective_r(entry=100.0, broker_stop=99.0,
                       targets=[102.0]) == 2.0
    assert effective_r(entry=100.0, broker_stop=None,
                       targets=[102.0]) is None
    assert effective_r(entry=100.0, broker_stop=99.0,
                       targets=[]) is None
    assert effective_r(entry=100.0, broker_stop=100.0,
                       targets=[102.0]) is None


def test_decollide_enforces_min_gap_and_preserves_order():
    out = decollide_label_ys([100.0, 105.0, 110.0], min_gap=16.0)
    assert out[1] - out[0] >= 16.0 and out[2] - out[1] >= 16.0
    assert out[0] == 100.0
    out = decollide_label_ys([300.0, 100.0, 200.0], min_gap=16.0)
    assert out == [300.0, 100.0, 200.0]
    out = decollide_label_ys([100.0, 105.0, 110.0], min_gap=16.0,
                             top=120.0)
    assert out[-1] <= 120.0 and out[1] - out[0] >= 16.0


# --- HTML page ----------------------------------------------------------

def test_html_shows_decision_score_and_both_stops():
    html = _html(decision_score=32.71, broker_stop=4054.4)
    p = _payload(html)
    assert p["meta"]["score"] == 32.71
    assert p["meta"]["score_text"] == "Score 33"
    assert p["meta"]["structural_stop"] == 4055.9
    assert p["meta"]["broker_stop"] == 4054.4
    assert p["meta"]["effective_r"] == abs(4062.7 - 4057.9) / abs(4057.9 - 4054.4)
    m5 = p["tfs"]["M5"]["analysis"]
    assert any("Broker SL 4,054.4" in line for line in m5)
    assert any("Structural stop 4,055.9" in line for line in m5)


def test_html_missing_score_and_broker_stop_are_honest_gaps():
    html = _html()  # no decision_score, no broker_stop
    p = _payload(html)
    assert p["meta"]["score"] is None
    assert p["meta"]["score_text"] == "Score unavailable"
    assert p["meta"]["broker_stop"] is None
    assert p["meta"]["effective_r"] is None
    assert "Score unavailable" in html
    assert "Broker SL pending at placement" in html
    assert "Score 60" not in html
    m5 = p["tfs"]["M5"]["analysis"]
    assert any("Broker SL pending at placement" in line for line in m5)


def test_html_buttons_carry_route_roles():
    html = _html()
    assert 'data-tf="M5">M5 · entry' in html
    assert 'data-tf="H1">H1 · context' in html
    assert 'data-tf="H4">H4 · structure' in html


def test_html_self_contained_and_has_theme_and_level_toggles():
    html = _html(decision_score=32.71, broker_stop=4054.4)
    assert "<script src" not in html
    assert "<link" not in html
    assert "https://" not in html
    assert 'id="theme-btn"' in html
    assert 'data-theme="dark"' in html
    assert 'id="ov-levels"' in html
    assert 'id="toggles"' in html
    # Study toggles still present and wired to redraw.
    for ov in ("ov-zone", "ov-nearby", "ov-fib", "ov-prem", "ov-sess"):
        assert f'id="{ov}"' in html


def test_html_legacy_score_kwarg_still_works():
    html = _html(score=80.0)
    assert _payload(html)["meta"]["score"] == 80.0


# --- PNG chart ----------------------------------------------------------

def test_png_renders_with_decision_score_and_broker_stop():
    h1 = _frame(200, "2026-10-26", "h")
    h4 = _frame(100, "2026-10-20", "4h")
    m5 = _frame(100, "2026-10-28", "5min")
    png = render_mt5_signal(
        h1, zone={"top": 4060.0, "bottom": 4050.0, "kind": "support"},
        entry=4057.9, stop=4055.9, targets=[4062.7], direction=1,
        verdict={"entry_clock": "M5", "bias_d1": 1, "bias_h4": 1,
                 "trigger": "sweep_reversal"},
        playbook="qm_m5", decision_score=32.71, broker_stop=4054.4,
        h4=h4, entry_frame=m5)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10_000


def test_png_renders_without_score_or_broker_stop():
    h1 = _frame(200, "2026-10-26", "h")
    png = render_mt5_signal(
        h1, zone=None, entry=4057.9, stop=4055.9, targets=[4062.7],
        direction=1, verdict={"entry_clock": "M15"}, playbook="qm_m5")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_png_close_levels_do_not_crash_label_pass():
    # Entry/stop 0.5 apart on a tight M5 window: the old code stacked
    # the labels; the de-collision pass must still render a PNG.
    h1 = _frame(200, "2026-10-26", "h")
    m5 = _frame(60, "2026-10-28", "5min")
    png = render_mt5_signal(
        h1, zone=None, entry=4057.9, stop=4057.4,
        targets=[4058.4, 4059.0], direction=1,
        verdict={"entry_clock": "M5"}, playbook="qm_m5",
        decision_score=32.71, broker_stop=4057.0,
        entry_frame=m5)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10_000


# --- runner seam ----------------------------------------------------------

def test_runner_builders_accept_step7_kwargs():
    for name in ("_build_chart", "_build_signal_view"):
        params = inspect.signature(getattr(MT5Runner, name)).parameters
        assert "decision_score" in params
        assert "broker_stop" in params
