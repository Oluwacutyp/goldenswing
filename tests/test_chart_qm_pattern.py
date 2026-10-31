"""Chart QM-pattern marking (owner order 2026-10-30: "fix the
chart generation").

The alert chart drew every study except the QM pattern itself —
for a QM signal the head / neckline / level / touch appeared
nowhere; the verdict box printed only "QM: qm_confirmed" as
text.  These tests pin:

- the hierarchy annotation carries the touch geometry;
- the runner releases pattern geometry to charts ONLY for a
  confirmed, this-signal QM (the stale-touch guard): native QM
  routes draw their own candidate's pattern; roster signals
  draw only a qm_confirmed annotation's; stale / against-bias /
  absent annotations draw nothing;
- the renderer draws the three levels + touch marker on the
  entry panel, folds the levels into the panel range, and
  routes the labels through the shared de-collision spec shape;
- the HTML signal view carries the same QM overlay.
"""
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from goldenswing.charts import engine as chart_engine  # noqa: E402
from goldenswing.charts.engine import (  # noqa: E402
    _mt5_qm_pattern, render_mt5_signal)
from goldenswing.charts.signal_view import (  # noqa: E402
    build_signal_view_html)
from goldenswing.execution.mt5_runner import (  # noqa: E402
    MT5Runner, MT5RunnerConfig)
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig  # noqa: E402
from goldenswing.live.hierarchy import qm_annotation  # noqa: E402
from mt5_fakes import FakeMT5  # noqa: E402


def _frame(n, freq, base):
    idx = pd.date_range("2026-10-14", periods=n, freq=freq, tz="UTC")
    close = [base + (i % 9) * 0.5 for i in range(n)]
    return pd.DataFrame(
        {"open": close, "high": [c + 0.4 for c in close],
         "low": [c - 0.4 for c in close], "close": close,
         "volume": [100] * n, "real_volume": [0] * n,
         "spread": [30] * n},
        index=idx)


def _runner(tmp_path):
    feed = MT5Feed(FakeMT5(), MT5FeedConfig(symbol="XAUUSDm"))
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "j.jsonl",
        risk_state_path=tmp_path / "r.json",
        kill_switch_path=tmp_path / "KILL",
        news_pause_enabled=False, max_daily_trades=0,
        max_hourly_trades=0)
    return MT5Runner(cfg, feed=feed)


_PATTERN = {"head": 4158.0, "neckline": 4149.0, "level": 4150.0,
            "direction": 1, "touch_ts": None}


# ------------------------------------------------------------------
# annotation geometry
# ------------------------------------------------------------------

def test_annotation_carries_geometry():
    touch = {"direction": 1, "touch_ts": "2026-10-14T09:55:00Z",
             "head": 4036.71, "neckline": 4046.19, "qm_level": 4040.0}
    ann = qm_annotation(touch, 1, "2026-10-14T10:00:00Z")
    assert ann["status"] == "qm_confirmed"
    assert ann["head"] == 4036.71
    assert ann["neckline"] == 4046.19
    assert ann["qm_level"] == 4040.0


def test_annotation_geometry_absent_is_none():
    ann = qm_annotation(None, 1, "2026-10-14T10:00:00Z")
    assert ann["head"] is None and ann["neckline"] is None
    assert ann["qm_level"] is None
    bare = qm_annotation({"direction": 1}, 1, "2026-10-14T10:00:00Z")
    assert bare["head"] is None


# ------------------------------------------------------------------
# the stale-touch guard (runner)
# ------------------------------------------------------------------

def _sig(**over):
    base = dict(direction=-1, entry_ref=4150.0, invalidation=4160.0,
                targets=[4130.0], atr=8.0, live_context={})
    base.update(over)
    return SimpleNamespace(**base)


def test_guard_confirmed_annotation_released(tmp_path):
    r = _runner(tmp_path)
    verdict = {"qm": {"status": "qm_confirmed", "head": 4158.0,
                      "neckline": 4149.0, "qm_level": 4150.5,
                      "direction": -1,
                      "touch_ts": "2026-10-14T10:00:00+00:00"}}
    p = r._qm_pattern_for_chart(_sig(), verdict, "sweep_reversal")
    assert p is not None
    assert p["head"] == 4158.0 and p["neckline"] == 4149.0
    assert p["level"] == 4150.5 and p["direction"] == -1


@pytest.mark.parametrize("status", ["qm_stale", "qm_against_bias",
                                    "qm_absent"])
def test_guard_unconfirmed_annotation_never_drawn(tmp_path, status):
    r = _runner(tmp_path)
    verdict = {"qm": {"status": status, "head": 4158.0,
                      "neckline": 4149.0, "qm_level": 4150.5,
                      "direction": -1, "touch_ts": None}}
    assert r._qm_pattern_for_chart(
        _sig(), verdict, "sweep_reversal") is None
    assert r._qm_pattern_for_chart(_sig(), {}, "sweep_reversal") is None
    assert r._qm_pattern_for_chart(_sig(), None, "sweep_reversal") is None


def test_guard_native_qm_route_draws_its_own_pattern(tmp_path):
    r = _runner(tmp_path)
    sig = _sig(live_context={"head": 4036.71, "neckline": 4046.19,
                             "qm_level": 4040.0})
    p = r._qm_pattern_for_chart(sig, {"qm": {"status": "qm_absent"}},
                                "qm_m5")
    assert p is not None and p["head"] == 4036.71
    bare = _sig(live_context={})
    assert r._qm_pattern_for_chart(bare, {}, "qm_m5") is None


# ------------------------------------------------------------------
# renderer
# ------------------------------------------------------------------

def _verdict():
    return {"bias_d1": -1, "bias_h4": -1, "trigger": "BOS",
            "entry_clock": "M15", "amd_phase": "distribution",
            "qm": {"status": "qm_confirmed"},
            "volume": {"volume_ratio": 1.2}}


def test_render_with_pattern_differs_and_survives():
    h1 = _frame(120, "1h", 4150.0)
    kw = dict(zone=None, entry=4150.0, stop=4160.0, targets=[4130.0],
              direction=-1, verdict=_verdict(), playbook="qm_m5",
              decision_score=61.0,
              h4=_frame(60, "4h", 4150.0),
              entry_frame=_frame(60, "5min", 4150.0),
              entry_clock="M5")
    plain = render_mt5_signal(h1, **kw)
    with_qm = render_mt5_signal(h1, qm_pattern=_PATTERN, **kw)
    # 3 PNGs (owner order 2026-10-31); the pattern marks ONLY the
    # entry panel — H4 and H1 images are byte-identical.
    assert len(plain) == 3 and len(with_qm) == 3
    assert with_qm[0] == plain[0] and with_qm[1] == plain[1]
    assert with_qm[2] != plain[2]
    assert all(p[:4] == b"\x89PNG" for p in with_qm)


def test_three_pngs_even_when_a_frame_is_missing():
    h1 = _frame(120, "1h", 4150.0)
    pngs = render_mt5_signal(
        h1, zone=None, entry=4150.0, stop=4160.0, targets=[4130.0],
        direction=-1, verdict=_verdict(), playbook="qm_m5",
        decision_score=61.0, h4=None, entry_frame=None)
    assert len(pngs) == 3
    assert all(p[:4] == b"\x89PNG" for p in pngs)


class _FakeSender:
    def __init__(self):
        self.photos = []
        self.texts = []

    def send_photo(self, png, caption=None, filename="chart.png"):
        self.photos.append((png, caption or ""))
        return SimpleNamespace(delivered=True, detail="ok")

    def send_text(self, text, silent=False):
        self.texts.append(text)
        return SimpleNamespace(delivered=True, detail="ok")


def test_send_alert_sends_three_photos(tmp_path):
    r = _runner(tmp_path)
    sender = _FakeSender()
    r.senders = [sender]
    r._send_alert("signal text",
                  chart_pngs=[b"H4", b"H1", b"ENTRY"])
    assert [p for p, _ in sender.photos] == [b"H4", b"H1", b"ENTRY"]
    assert sender.photos[0][1] == "signal text"
    assert "H1" in sender.photos[1][1]
    assert "Entry" in sender.photos[2][1]
    assert sender.texts == []


def test_qm_specs_carry_three_labels():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    idx = _frame(60, "15min", 4150.0).index
    specs = _mt5_qm_pattern(ax, 70.0, _PATTERN, idx,
                            chart_engine._THEMES["classic"])
    texts = [s["text"] for s in specs]
    assert any("QM HEAD" in t for t in texts)
    assert any("QM NECKLINE" in t for t in texts)
    assert any("QM LEVEL" in t for t in texts)
    assert all(set(s) == {"value", "text", "color", "bold", "size"}
               for s in specs)
    plt.close(fig)


def test_qm_touch_marker_inside_window():
    import matplotlib.pyplot as plt
    frame = _frame(60, "15min", 4150.0)
    fig, ax = plt.subplots()
    pattern = dict(_PATTERN,
                   touch_ts=frame.index[30].isoformat())
    specs = _mt5_qm_pattern(ax, 70.0, pattern, frame.index,
                            chart_engine._THEMES["classic"])
    texts = [t.get_text() for t in ax.texts]
    assert any("QM TOUCH" in t for t in texts)
    assert len(specs) == 3
    plt.close(fig)


# ------------------------------------------------------------------
# signal view parity
# ------------------------------------------------------------------

def test_signal_view_carries_qm_overlay():
    frames = {"H1": _frame(120, "1h", 4150.0),
              "M15": _frame(60, "15min", 4150.0)}
    html = build_signal_view_html(
        frames=frames, entry=4150.0, stop=4160.0, targets=[4130.0],
        direction=-1, playbook="qm_m5", decision_score=61.0,
        verdict=_verdict(), zone=None, nearby_zones=[],
        fib_levels=[], fib_range=None, ote_band=None, fib_info=None,
        alert_id="t-qm", qm_pattern=_PATTERN)
    assert '"qm": {"head": 4158.0' in html
    assert 'id="ov-qm"' in html
    html_plain = build_signal_view_html(
        frames=frames, entry=4150.0, stop=4160.0, targets=[4130.0],
        direction=-1, playbook="qm_m5", decision_score=61.0,
        verdict=_verdict(), zone=None, nearby_zones=[],
        fib_levels=[], fib_range=None, ote_band=None, fib_info=None,
        alert_id="t-qm2")
    assert '"qm": null' in html_plain
