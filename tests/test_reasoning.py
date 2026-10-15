"""Dynamic reasoning + study packages for accepted signals.

Reasoning: confidence label + technical factors + macro factors +
risk, every line derived from the signal + verdict — never canned.
Study: the key levels / data behind the signal, JSON-safe, on the
signal record.  Telegram gets the reasoning text + the existing
chart PNG (the visual half of the study).  Shadow signals get no
reasoning (they get no alert).
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from goldenswing.live.reasoning import (
    build_reasoning, build_study, confidence_label)


def _sig(**kw):
    base = dict(direction=1, score=80.0, entry_ref=2000.0,
                invalidation=1990.0, targets=[2030.0], atr=10.0,
                index=100, factors=["FVG tap"],
                timestamp=pd.Timestamp("2026-01-05", tz="UTC"),
                entry_zone=(1990.0, 2010.0), kind="fvg_midline",
                is_golden=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _verdict(**kw):
    base = {"bias_d1": 1, "bias_h4": 1, "trigger": "FVG tap",
            "entry_clock": "M15", "amd_phase": "accumulation",
            "amd_confidence": 0.7, "qm": {"status": "none"},
            "volume": {"volume_ratio": 1.5},
            "zone": {"bottom": 1990.0, "top": 2010.0},
            "dxy_corr": -0.85, "urgency": 78.0,
            "session": {"session": "london", "killzone": "london_kz",
                        "trigger_tf": "M5", "amd_phase": None,
                        "playbook": None, "tags": []}}
    base.update(kw)
    return base


def test_confidence_label_bands():
    assert confidence_label(95) == "Very high"
    assert confidence_label(85) == "High"
    assert confidence_label(75) == "Moderate"
    assert confidence_label(65) == "Marginal"
    assert confidence_label(40) == "Low"


def test_reasoning_derives_every_line_from_signal_and_verdict():
    text = build_reasoning("fvg_midline", _sig(), _verdict(),
                           alignment_label="Golden Swing")
    assert "Why LONG (High conviction, score 80)" in text
    assert "fvg_midline at the POI — entry 2,000.0, stop 1,990.0, " \
        "target 2,030.0 (3.00R)" in text
    assert "Zone 1,990.0–2,010.0 · trigger: FVG tap" in text
    assert "Bias: D1 bullish / H4 bullish" in text
    assert "AMD phase: accumulation (confidence 0.70)" in text
    assert "Volume ratio: 1.50x 20-bar mean" in text
    assert "Alignment: Golden Swing" in text
    assert "Session: london" in text
    assert "Urgency: 78/100" in text
    assert "DXY correlation -0.85 (strong inverse — gold tailwind)" \
        in text
    assert "FVG tap" in text


def test_reasoning_dxy_toxic_positive_and_short():
    text = build_reasoning("sweep_reversal", _sig(direction=-1),
                           _verdict(dxy_corr=0.1))
    assert "Why SHORT" in text
    assert "DXY correlation 0.10 (toxic positive — DXY headwind)" \
        in text


def test_reasoning_missing_fields_drop_lines_never_crash():
    text = build_reasoning("fvg_midline", _sig(), None)
    assert "Why LONG" in text
    assert "Bias: D1 n/a / H4 n/a" in text
    assert "DXY" not in text
    assert "Zone" not in text


def test_study_package_keys_and_r():
    study = build_study("fvg_midline", _sig(), _verdict(),
                        alignment_label="Golden Swing")
    assert study["playbook"] == "fvg_midline"
    assert study["direction"] == 1
    assert study["entry"] == pytest.approx(2000.0)
    assert study["stop"] == pytest.approx(1990.0)
    assert study["targets"] == [2030.0]
    assert study["risk_reward"] == pytest.approx(3.0)
    assert study["zone"] == {"bottom": 1990.0, "top": 2010.0}
    assert study["bias_d1"] == 1 and study["bias_h4"] == 1
    assert study["amd_phase"] == "accumulation"
    assert study["volume_ratio"] == pytest.approx(1.5)
    assert study["dxy_corr"] == pytest.approx(-0.85)
    assert study["urgency"] == pytest.approx(78.0)
    assert study["alignment"] == "Golden Swing"
    assert study["session"] == "london"
    assert study["factors"] == ["FVG tap"]


class _Engine:
    def gate_live_signal(self, pb, sig, ctx, hierarchy_verdict=None,
                         spread_px=None):
        return True, ""

    def score_live_signal(self, pb, sig, ctx, hierarchy_verdict=None):
        return 80.0, ["FVG tap"]


class _Sender:
    def __init__(self):
        self.photos = []
        self.texts = []

    def send_photo(self, png, caption):
        from types import SimpleNamespace as NS

        self.photos.append((png, caption))
        return NS(delivered=True, detail="")

    def send_text(self, text, reply_markup=None):
        from types import SimpleNamespace as NS

        self.texts.append(text)
        return NS(delivered=True, detail="")


class _Exec:
    def symbol_spec(self, symbol):
        from goldenswing.execution.risk import gold_spec

        return gold_spec()

    def positions(self, symbol=None):
        return []

    def place_order(self, request):
        from goldenswing.execution.executor import OrderResult

        return OrderResult(True, ticket=9001,
                           price=request.entry_ref, volume=0.01)

    def cancel_pending(self, ticket):
        from goldenswing.execution.executor import OrderResult

        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        from goldenswing.execution.executor import OrderResult

        return OrderResult(True, ticket=ticket)

    def modify_sl(self, ticket, sl, tp=None):
        from goldenswing.execution.executor import OrderResult

        return OrderResult(True, ticket=ticket)


class _PB:
    name = "fvg_midline"

    def __init__(self, sig):
        self._sig = sig

    def live_signal(self, states, t, df=None):
        return [self._sig]


def test_runner_accepted_signal_carries_reasoning_and_study(tmp_path):
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    from goldenswing.execution.signal_log import signal_records
    from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
    from goldenswing.live.hierarchy import TimeframeStack
    from goldenswing.playbooks.base import EngineContext
    from mt5_fakes import FakeMT5

    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        execute_playbooks=("fvg_midline",),
        golden_swing_confirm_enabled=False)
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    feed.connect()
    sender = _Sender()
    runner = MT5Runner(cfg, feed=feed, executor=_Exec(),
                       senders=[sender])
    h1 = pd.DataFrame(
        {"open": [2000.0] * 60, "high": [2001.0] * 60,
         "low": [1999.0] * 60, "close": [2000.5] * 60,
         "volume": [100] * 60, "real_volume": [0] * 60,
         "spread": [30] * 60},
        index=pd.date_range("2026-01-01", periods=60, freq="h",
                            tz="UTC"))
    runner._stack = TimeframeStack.from_frames({"H1": h1})
    runner._last_tick = {"mid": 2000.0, "spread": 30,
                         "time": pd.Timestamp(
                             "2026-01-05 10:00", tz="UTC"
                         ).to_pydatetime()}
    ctx = EngineContext(df=h1, states=[], years=0.0)
    sig = _sig()
    runner._run_one_playbook(_PB(sig), sig_ctx=(ctx, _Engine()),
                             h1=h1, t=59,
                             verdict=_verdict(),
                             stats={"signals": 0})
    records = signal_records(runner.journal.records())
    accepted = [r for r in records if r["decision"] == "accepted"]
    assert len(accepted) == 1
    rec = accepted[0]
    assert "Why LONG" in rec["reasoning"]
    assert "DXY correlation -0.85" in rec["reasoning"]
    assert rec["study"]["entry"] == pytest.approx(2000.0)
    assert rec["study"]["risk_reward"] == pytest.approx(3.0)
    # The Telegram alert carries the reasoning + the chart PNG.
    # (A second photo follows: the fill alert, after the order
    # placed — the signal alert is photos[0].)
    assert len(sender.photos) == 2
    png, caption = sender.photos[0]
    assert png  # chart attached
    assert "Why LONG" in caption
    assert "DXY correlation -0.85" in caption


def test_runner_shadow_signal_gets_no_reasoning(tmp_path):
    from goldenswing.execution.mt5_runner import (
        MT5Runner, MT5RunnerConfig)
    from goldenswing.execution.signal_log import signal_records
    from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
    from goldenswing.live.hierarchy import TimeframeStack
    from goldenswing.playbooks.base import EngineContext
    from mt5_fakes import FakeMT5

    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        execute_playbooks=(),  # fvg_midline is shadow here
        golden_swing_confirm_enabled=False)
    feed = MT5Feed(MT5FeedConfig(), mt5_module=FakeMT5())
    feed.connect()
    sender = _Sender()
    runner = MT5Runner(cfg, feed=feed, executor=_Exec(),
                       senders=[sender])
    h1 = pd.DataFrame(
        {"open": [2000.0] * 60, "high": [2001.0] * 60,
         "low": [1999.0] * 60, "close": [2000.5] * 60,
         "volume": [100] * 60, "real_volume": [0] * 60,
         "spread": [30] * 60},
        index=pd.date_range("2026-01-01", periods=60, freq="h",
                            tz="UTC"))
    runner._stack = TimeframeStack.from_frames({"H1": h1})
    ctx = EngineContext(df=h1, states=[], years=0.0)
    runner._run_one_playbook(_PB(_sig()), sig_ctx=(ctx, _Engine()),
                             h1=h1, t=59,
                             verdict=_verdict(),
                             stats={"signals": 0})
    records = signal_records(runner.journal.records())
    shadow = [r for r in records if r["decision"] == "shadow"]
    assert len(shadow) == 1
    assert shadow[0]["reasoning"] is None
    assert shadow[0]["study"] is None
    assert sender.photos == [] and sender.texts == []
