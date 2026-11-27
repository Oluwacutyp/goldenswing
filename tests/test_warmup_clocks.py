"""Startup warmup on each playbook's own trigger clock.

Owner correction 2026-11-27 ("29 fucking H1 playbooks?????"):
the old warmup loop drove all 29 playbooks through H1 bars and
printed "warmed" for each — but only 20 trigger on H1 (9 on
M15), and the 11 two-clock family playbooks return nothing from
live_signal without a formation frame, so they were "warmed"
without ever scanning.  These tests pin the fix: each playbook
warms on its registry trigger clock, with its formation frame
threaded, and the report is honest about what actually ran.
"""
from mt5_fakes import FakeMT5
from test_auction_value import _acceptance_frames, _acceptance_pb, _m15_from_h1
from test_auction_value import _situation as _auction_situation
from test_trend_momentum import _happy_frames, _pullback_pb
from test_trend_momentum import _situation as _trend_situation

from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.playbook_registry import (
    AUCTION_SPECS,
    REVERSION_SPECS,
    ROSTER_SPECS,
    SESSION_SPECS,
    TREND_SPECS,
    VOL_SPECS,
    trigger_tf_of,
)
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig
from goldenswing.live.hierarchy import TimeframeStack
from goldenswing.playbooks.auction_value import AuctionValueAcceptancePlaybook
from goldenswing.playbooks.trend_momentum import TrendImpulsePullbackPlaybook

ALL_SPECS = (ROSTER_SPECS + VOL_SPECS + TREND_SPECS + AUCTION_SPECS
             + REVERSION_SPECS + SESSION_SPECS)
TREND_NAMES = {s.name for s in TREND_SPECS}
AUCTION_NAMES = {s.name for s in AUCTION_SPECS}
REVERSION_NAMES = {s.name for s in REVERSION_SPECS}
SESSION_NAMES = {s.name for s in SESSION_SPECS}
TWO_CLOCK_NAMES = (TREND_NAMES | AUCTION_NAMES
                   | REVERSION_NAMES | SESSION_NAMES)


class _FakeExecutor:
    def connect(self, **kw):
        return True, "fake connected"

    def positions(self, symbol=None):
        return []

    def pending_orders(self, symbol=None):
        return []


def _runner(tmp_path):
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        tick_interval=0.001)
    feed = MT5Feed(
        MT5FeedConfig(tick_interval=0.001),
        mt5_module=FakeMT5(tick_script=[
            (2000.0, 2000.30, 0), (2000.10, 2000.40, 1)]))
    return MT5Runner(cfg, feed=feed, executor=_FakeExecutor(),
                     senders=[])


def _stack():
    """One stack for the whole trigger path: the trend happy
    frames (a real impulse + pullback lives in them) with a
    volume column (read_and_featurize resamples volume),
    M15 interpolated from the H1, D1 resampled."""
    h1, h4 = _happy_frames()
    for frame in (h1, h4):
        frame["volume"] = 100.0
    m15 = _m15_from_h1(h1)
    d1 = h1.resample("1d").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna()
    stack = TimeframeStack.from_frames(
        {"H1": h1, "H4": h4, "M15": m15, "D1": d1})
    return stack, {"H1": h1, "H4": h4, "M15": m15, "D1": d1}


def _situation():
    """A keeper snapshot in production shape: the trend read
    (per_tf bias/phase + aligned pair kinds) merged with the
    auction read (location value).  The warmup threads
    whatever the keeper holds; None is the no-snapshot case."""
    trend = _trend_situation()
    auction = _auction_situation()
    merged = {**auction, **trend}
    merged["location"] = auction["location"]
    return merged


def _rows_by_name(report):
    return {row["playbook"]: row for row in report}


# ------------------------------------------- the warmup itself


def test_every_playbook_warms_on_its_own_trigger_clock(
        tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    monkeypatch.setattr(runner, "_current_situation",
                        lambda: _situation())
    stack, frames = _stack()
    runner._stack = stack
    report = runner._warm_playbooks()
    assert len(report) == 29
    assert {row["playbook"] for row in report} == \
        {spec.name for spec in ALL_SPECS}
    by_name = _rows_by_name(report)
    h1_rows = [r for r in report if r["clock"] == "H1"]
    m15_rows = [r for r in report if r["clock"] == "M15"]
    assert len(h1_rows) == 20
    assert len(m15_rows) == 9
    for spec in ALL_SPECS:
        row = by_name[spec.name]
        assert row["clock"] == trigger_tf_of(spec.name), spec.name
        assert row["warmed"] is True, (spec.name, row["reason"])
        assert row["reason"] == ""
        assert row["bars"] == len(frames[row["clock"]]), spec.name
        if spec.name in TREND_NAMES:
            assert row["formation_tf"] == "H4", spec.name
        elif spec.name in (AUCTION_NAMES | REVERSION_NAMES
                           | SESSION_NAMES):
            assert row["formation_tf"] == "H1", spec.name
        else:
            assert row["formation_tf"] is None, spec.name
    # The planted impulse + pullback is in the happy frames:
    # the trend scans find it once their formation is threaded.
    for name in TREND_NAMES:
        assert by_name[name]["signals"] >= 1, name


def test_threading_is_what_makes_the_family_scan_real():
    """The mechanism the warmup relies on, at playbook level:
    without a formation frame the two-clock families return
    nothing on every bar; with it threaded they detect."""
    h1, h4 = _happy_frames()
    pb = TrendImpulsePullbackPlaybook()
    assert [s for t in range(len(h1))
            for s in (pb.live_signal([], t, h1) or [])] == []
    threaded = _pullback_pb(h4)
    found = [s for t in range(len(h1))
             for s in (threaded.live_signal([], t, h1) or [])]
    assert len(found) >= 1

    ah1, am15 = _acceptance_frames()
    pb = AuctionValueAcceptancePlaybook()
    assert [s for t in range(len(am15))
            for s in (pb.live_signal([], t, am15) or [])] == []
    threaded = _acceptance_pb(ah1)
    found = [s for t in range(len(am15))
             for s in (threaded.live_signal([], t, am15) or [])]
    assert len(found) >= 1


def test_single_call_drive_finds_what_per_bar_replay_finds():
    """The warmup drives each playbook with ONE live_signal
    call over its whole trigger frame, not one call per bar:
    a call scans from _processed+1 to t and the detection at
    each bar is causal, so one call must find exactly what
    the per-bar replay finds (with one PreparedFrame build
    instead of one per bar — the per-bar shape rebuilds the
    frame's swings / levels / day context on EVERY bar).
    Proved here on both clocks; the shape is the same for
    every playbook (live_signal's only persistent mutation
    is the _processed cursor)."""
    h1, h4 = _happy_frames()
    for frame in (h1, h4):
        frame["volume"] = 100.0
    per_bar = TrendImpulsePullbackPlaybook()
    per_bar.set_formation(h4, "H4")
    per_bar.set_situation(_trend_situation())
    found_per_bar = [s for t in range(len(h1))
                     for s in (per_bar.live_signal([], t, h1)
                               or [])]
    one_call = TrendImpulsePullbackPlaybook()
    one_call.set_formation(h4, "H4")
    one_call.set_situation(_trend_situation())
    found_one = one_call.live_signal([], len(h1) - 1, h1) or []
    assert [(s.index, s.direction) for s in found_one] == \
        [(s.index, s.direction) for s in found_per_bar]
    assert len(found_one) >= 1
    assert one_call._processed == per_bar._processed \
        == len(h1) - 1

    ah1, am15 = _acceptance_frames()
    per_bar = AuctionValueAcceptancePlaybook()
    per_bar.set_formation(ah1, "H1")
    per_bar.set_situation(_auction_situation())
    found_per_bar = [s for t in range(len(am15))
                     for s in (per_bar.live_signal([], t, am15)
                               or [])]
    one_call = AuctionValueAcceptancePlaybook()
    one_call.set_formation(ah1, "H1")
    one_call.set_situation(_auction_situation())
    found_one = one_call.live_signal([], len(am15) - 1, am15) or []
    assert [(s.index, s.direction) for s in found_one] == \
        [(s.index, s.direction) for s in found_per_bar]
    assert len(found_one) >= 1
    assert one_call._processed == per_bar._processed \
        == len(am15) - 1


def test_warmup_restores_the_live_state(tmp_path, monkeypatch):
    """Warmup-only threading: after the drive the persistent
    instances are back to their live state (no formation
    frame — the families evaluate on the fire path only by
    injection), with the streaming cursor advanced to the
    last trigger bar."""
    runner = _runner(tmp_path)
    monkeypatch.setattr(runner, "_current_situation",
                        lambda: _situation())
    stack, frames = _stack()
    runner._stack = stack
    report = runner._warm_playbooks()
    assert all(row["warmed"] for row in report)
    for pb in runner._get_playbooks():
        if pb.name in TWO_CLOCK_NAMES:
            assert pb._formation_frame is None, pb.name
            assert pb._processed == \
                len(frames[trigger_tf_of(pb.name)]) - 1, pb.name


def test_missing_m15_frame_reports_nine_not_warmed(
        tmp_path, capsys, monkeypatch):
    """A stack without M15: the 9 M15 playbooks are reported
    NOT warmed with the reason named — never printed as
    warmed — while the 20 H1 playbooks warm."""
    runner = _runner(tmp_path)
    monkeypatch.setattr(runner, "_current_situation",
                        lambda: _situation())
    stack, _frames = _stack()
    stack.frames.pop("M15")
    runner._stack = stack
    report = runner._warm_playbooks()
    by_name = _rows_by_name(report)
    for spec in ALL_SPECS:
        row = by_name[spec.name]
        if trigger_tf_of(spec.name) == "M15":
            assert row["warmed"] is False, spec.name
            assert "M15 frame unavailable" in row["reason"]
        else:
            assert row["warmed"] is True, (spec.name, row["reason"])
    out = capsys.readouterr().out
    assert "NOT warmed auction_value_acceptance (M15)" in out
    assert "warmed sweep_reversal (H1, " in out


def test_a_raising_playbook_does_not_kill_warmup(
        tmp_path, monkeypatch):
    """One playbook's scan raising is reported as its own
    not-warmed row; the other 28 still warm and startup
    continues."""
    runner = _runner(tmp_path)
    monkeypatch.setattr(runner, "_current_situation",
                        lambda: _situation())
    stack, _ = _stack()
    runner._stack = stack
    victim = next(pb for pb in runner._get_playbooks()
                  if pb.name == "sweep_reversal")

    def _boom(states, t, df=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(victim, "live_signal", _boom)
    report = runner._warm_playbooks()
    by_name = _rows_by_name(report)
    row = by_name["sweep_reversal"]
    assert row["warmed"] is False
    assert row["reason"].startswith(
        "warmup failed: RuntimeError: boom")
    assert sum(1 for r in report if r["warmed"]) == 28


def test_warmup_never_touches_the_native_paths(
        tmp_path, monkeypatch):
    """The warmup covers exactly the 29 trigger-path
    playbooks: qm_m5 / qm_d1 keep their own startup scans,
    hot_scalp its own M5 engine, arrival_zone the
    resting-limit path — none of them is warmed here, and
    the native scans are never invoked by the warmup."""
    runner = _runner(tmp_path)
    monkeypatch.setattr(runner, "_current_situation",
                        lambda: _situation())
    stack, _ = _stack()
    runner._stack = stack

    def _boom(*args, **kwargs):
        raise AssertionError("native path invoked by warmup")

    monkeypatch.setattr(runner, "_run_qm_m5", _boom)
    monkeypatch.setattr(runner, "_run_qm_d1", _boom)
    report = runner._warm_playbooks()
    names = {row["playbook"] for row in report}
    assert names == {spec.name for spec in ALL_SPECS}
    assert not (names & {"qm_m5", "qm_d1", "hot_scalp",
                         "arrival_zone"})
