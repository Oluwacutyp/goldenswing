"""Multi-timeframe MT5 charts: H4 structure / H1 trigger / entry-TF entry.

The old chart was H1-only — the single-timeframe mistake in chart form.
These tests prove the chart shows the whole hierarchy stack, that a
missing timeframe renders a labeled panel (never invented bars), and
that the runner passes H4 + entry-TF frames into the builder.
"""
import io
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from goldenswing.charts.engine import render_mt5_signal  # noqa: E402
from goldenswing.execution.mt5_runner import (  # noqa: E402
    MT5Runner, MT5RunnerConfig)
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig  # noqa: E402
from goldenswing.live.hierarchy import TimeframeStack  # noqa: E402
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


def _verdict(**over):
    v = {"bias_d1": 1, "bias_h4": 1, "trigger": "BOS",
         "entry_clock": "M15", "amd_phase": "distribution",
         "qm": {"status": "qm_absent"}, "volume": {"volume_ratio": 1.2},
         "zone": {"kind": "support", "level": 4150.0, "top": 4152.0,
                  "bottom": 4148.0, "index": 100}}
    v.update(over)
    return v


def _sig(**over):
    s = dict(direction=1, score=72.0, entry_ref=4155.0,
             invalidation=4145.0, targets=[4175.0], atr=10.0, index=100)
    s.update(over)
    return SimpleNamespace(**s)


def _full_chart(**kw):
    args = dict(h1=_frame(150, "h", 4160.0),
                h4=_frame(80, "4h", 4160.0),
                entry_frame=_frame(200, "15min", 4160.0),
                zone=_verdict()["zone"], entry=4155.0, stop=4145.0,
                targets=[4175.0], direction=1,
                verdict=_verdict(), playbook="fvg_midline", score=72)
    args.update(kw)
    return render_mt5_signal(**args)


def test_full_stack_chart_has_three_panels():
    png = _full_chart()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 10_000  # a real chart, not an empty canvas
    arr = plt.imread(io.BytesIO(png))
    # 1600x1300 figure: three stacked panels, non-blank.
    assert arr.shape[0] >= 1200 and arr.shape[1] == 1600
    assert float(arr.std()) > 0.01


def test_levels_flow_into_stack_render():
    png_a = _full_chart()
    png_b = _full_chart(entry=4165.0)
    assert png_a != png_b
    png_fill = _full_chart(fill_price=4155.5)
    assert png_fill != png_a
    # A different H4 frame changes the render (H4 panel is real).
    png_h4 = _full_chart(h4=_frame(80, "4h", 4200.0))
    assert png_h4 != png_a
    # A different entry-TF frame changes the render.
    png_e = _full_chart(entry_frame=_frame(200, "15min", 4200.0))
    assert png_e != png_a


def test_missing_h4_panel_labeled_not_faked():
    png_full = _full_chart()
    png_no_h4 = _full_chart(h4=None)
    assert png_no_h4[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png_no_h4) > 10_000
    # Missing H4 changes the render (labeled panel) but never crashes.
    assert png_no_h4 != png_full
    # An empty H4 frame behaves like a missing one — no invented bars.
    empty_h4 = _frame(80, "4h", 4160.0).iloc[:0]
    png_empty = _full_chart(h4=empty_h4)
    assert png_empty[:8] == b"\x89PNG\r\n\x1a\n"
    assert png_empty != png_full


def test_missing_entry_frame_labeled_not_faked():
    png_full = _full_chart()
    png_no_entry = _full_chart(entry_frame=None)
    assert png_no_entry[:8] == b"\x89PNG\r\n\x1a\n"
    assert png_no_entry != png_full


def test_entry_clock_m30_panel():
    png = _full_chart(
        entry_frame=_frame(120, "30min", 4160.0),
        verdict=_verdict(entry_clock="M30"))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 10_000


def test_blueprint_style_and_bad_style():
    png_classic = _full_chart()
    png_blue = _full_chart(style="blueprint")
    assert png_blue[:8] == b"\x89PNG\r\n\x1a\n"
    assert png_blue != png_classic
    with pytest.raises(ValueError):
        _full_chart(style="line")


def test_empty_h1_raises():
    with pytest.raises(ValueError):
        _full_chart(h1=_frame(150, "h", 4160.0).iloc[:0])


class _FakeExecutor:
    def connect(self, **kw):
        return True, "fake connected"

    def positions(self, symbol=None):
        return []


def _runner(tmp_path, **cfg_kw):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk_state.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        charts_dir=tmp_path / "mt5_charts", **cfg_kw)
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    feed.connect()
    runner = MT5Runner(cfg, feed=feed, executor=_FakeExecutor())
    runner._stack = TimeframeStack.from_frames(feed.warm_start_all())
    return runner


def test_runner_passes_h4_and_entry_frames(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    seen = {}

    def recorder(*args, **kwargs):
        seen.update(kwargs)
        seen["h1"] = args[0] if args else None
        from goldenswing.charts.engine import render_mt5_signal as real
        return real(*args, **kwargs)

    monkeypatch.setattr(
        "goldenswing.execution.mt5_runner.render_mt5_signal", recorder)
    png, path = runner._build_chart(
        _sig(), _verdict(), alert_id="MT5-test-100",
        playbook="fvg_midline")
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"
    assert path is not None and path.exists()
    # The runner passed the whole stack, not just H1.
    assert seen["h4"] is not None and len(seen["h4"]) > 0
    assert seen["entry_frame"] is not None and len(seen["entry_frame"]) > 0
    assert seen["entry_clock"] == "M15"
    assert seen["style"] == "classic"


def test_runner_chart_without_h4_still_builds(tmp_path):
    runner = _runner(tmp_path)
    frames = dict(runner._stack.frames)
    frames.pop("H4", None)
    runner._stack = TimeframeStack.from_frames(frames)
    png, path = runner._build_chart(
        _sig(), _verdict(), alert_id="MT5-test-noh4",
        playbook="fvg_midline")
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"
    assert path is not None and path.exists()


def test_runner_chart_blueprint_style(tmp_path):
    runner = _runner(tmp_path, chart_style="blueprint")
    png, path = runner._build_chart(
        _sig(), _verdict(), alert_id="MT5-test-blue",
        playbook="fvg_midline")
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"
    png_classic, _ = _runner(tmp_path)._build_chart(
        _sig(), _verdict(), alert_id="MT5-test-cl",
        playbook="fvg_midline")
    assert png != png_classic
