"""Per-signal HTML terminal view: baked payload, per-timeframe
analysis, self-containment, and injection safety."""
import json

import numpy as np
import pandas as pd

from goldenswing.charts.signal_view import (
    _json_for_html,
    build_signal_view_html,
)

rng = np.random.default_rng(11)


def _frame(n, start, freq):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    walk = 4000.0 + np.cumsum(rng.normal(0, 3, n))
    return pd.DataFrame({
        "open": walk, "high": walk + 4.0, "low": walk - 4.0,
        "close": walk + rng.normal(0, 1, n),
    }, index=idx)


def _payload(html):
    start = html.index('<script id="signal-data" type="application/json">')
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    return json.loads(html[start:end])


VERDICT = {"bias_d1": -1, "bias_h4": 1, "trigger": "sweep_reversal",
           "entry_clock": "M15", "amd_phase": "accumulation",
           "qm": {"status": "ok"}, "volume": {"volume_ratio": 1.4}}
ZONE = {"top": 4020.0, "bottom": 4000.0, "kind": "support"}
STUDIES = {
    "nearby_zones": [{"level": 4050.0, "tier": "MAJOR",
                      "side": "resistance"}],
    "fib_levels": [(0.382, 4030.0), (0.5, 4020.0), (0.618, 4010.0)],
    "fib_range": (3990.0, 4060.0), "ote_band": (4005.0, 4018.0),
    "fib_info": {"nearest_frac": 0.5, "at_fib": True,
                 "in_ote_long": False, "in_ote_short": False,
                 "leg_direction": 1},
}


def _build(frames, verdict=VERDICT, playbook="test", zone=ZONE, **kw):
    params = {**STUDIES, **kw}
    return build_signal_view_html(
        frames=frames, entry=4012.0, stop=3995.0, targets=[4040.0],
        direction=1, playbook=playbook, score=80.0, verdict=verdict,
        zone=zone, alert_id="sig-1", fill_price=None, **params)


def test_buttons_only_for_present_timeframes():
    html = _build({"H1": _frame(200, "2026-10-26", "h"),
                   "H4": _frame(100, "2026-10-20", "4h"),
                   "M15": _frame(100, "2026-10-28", "15min")})
    for tf in ("H4", "H1", "M15"):
        assert f'data-tf="{tf}"' in html
    for tf in ("D1", "M30", "M5"):
        assert f'data-tf="{tf}"' not in html


def test_payload_carries_bars_studies_and_analysis():
    html = _build({"H1": _frame(200, "2026-10-26", "h"),
                   "H4": _frame(100, "2026-10-20", "4h"),
                   "M15": _frame(100, "2026-10-28", "15min")})
    p = _payload(html)
    assert len(p["tfs"]["H1"]["bars"]) == 200
    assert p["zone"] == {"top": 4020.0, "bottom": 4000.0,
                         "kind": "support"}
    assert p["fib"]["levels"][1] == [0.5, 4020.0]
    assert p["fib"]["ote"] == [4005.0, 4018.0]
    h1_lines = " ".join(p["tfs"]["H1"]["analysis"])
    assert "sweep_reversal" in h1_lines
    assert "Dealing range" in h1_lines
    h4_lines = " ".join(p["tfs"]["H4"]["analysis"])
    assert "H4 bias +1" in h4_lines and "4,000.0" in h4_lines
    m15_lines = " ".join(p["tfs"]["M15"]["analysis"])
    assert "Entry timeframe M15" in m15_lines
    assert p["meta"]["verdict_lines"] == ["AMD phase: accumulation",
                                          "QM: ok",
                                          "Volume ratio: 1.4"]


def test_self_contained_no_external_resources():
    html = _build({"H1": _frame(50, "2026-10-26", "h")})
    assert "src=" not in html and "http://" not in html
    assert "https://" not in html


def test_payload_cannot_break_out_of_script_block():
    html = _build({"H1": _frame(50, "2026-10-26", "h")},
                  playbook='</script><img src=x onerror=alert(1)>')
    assert html.count("</script>") == 2  # data block + chart block only
    assert "<img src=x" not in html
    p = _payload(html)
    assert p["meta"]["playbook"].startswith("</script>")


def test_minimal_verdict_states_missing_reads_honestly():
    html = _build({"D1": _frame(30, "2026-09-01", "D"),
                   "H1": _frame(50, "2026-10-26", "h")},
                  verdict={}, zone=None,
                  nearby_zones=[], fib_levels=[], fib_range=None,
                  ote_band=None, fib_info=None)
    p = _payload(html)
    assert "No D1 read recorded." in p["tfs"]["D1"]["analysis"]
    assert "No H1 read recorded." in p["tfs"]["H1"]["analysis"]


def test_no_frames_raises():
    try:
        _build({})
    except ValueError:
        return
    raise AssertionError("expected ValueError for no frames")


def test_bars_capped_and_json_escapes():
    html = _build({"H1": _frame(600, "2026-09-01", "h")})
    assert len(_payload(html)["tfs"]["H1"]["bars"]) == 500
    esc = _json_for_html({"x": "<a>&b"})
    assert "<" not in esc and ">" not in esc and "&" not in esc
    assert json.loads(esc.replace("\\u003c", "<").replace("\\u003e", ">")
                      .replace("\\u0026", "&")) == {"x": "<a>&b"}
