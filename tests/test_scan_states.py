"""Scheduled-scan featurized states (playbook audit 2026-11-28).

Three roster playbooks — sweep_reversal, fvg_midline,
trendline_retest — read the featurized MarketState at bar t
(``states[t]``) in live_signal.  The scheduled scan used to pass
``[]`` unconditionally, so all three returned [] on every bar and
could never arm: sweep_reversal 0 setups in 30 days (7/week with
real states), fvg_midline 0 (1/week with real states),
trendline_retest 0.  The runner now featurizes the scan frame
once per clock and threads the states to playbooks that declare
``needs_states``.

What this file pins:
- _scan_states featurizes once per frame per clock (same object
  on repeat calls; a new close re-featurizes);
- _sweep_eval_one passes real states to a needs_states playbook
  and [] to every other playbook;
- the sweep_reversal scan registers a setup on real data (the
  2026-10-07 14:00 UTC H1 close, direction -1, score 84 — one of
  the 7 bars the counterfactual replay found in 7 days).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from test_p3_trader import _frame, _runner

from goldenswing.execution.playbook_registry import PLAYBOOK_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[1]
H1_CSV = REPO_ROOT / "data" / "OANDA_XAU_USD_H1_20241001_20261012.csv"

# The sweep_reversal counterfactual hit (real data): the 400-bar H1
# frame ending at this close fires direction -1, score 84.0.
SWEEP_BAR = pd.Timestamp("2026-10-07 14:00:00", tz="UTC")


def _real_h1_ending(bar: pd.Timestamp, bars: int = 400) -> pd.DataFrame:
    df = pd.read_csv(H1_CSV)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    n = int(df.index.searchsorted(bar, side="right"))
    return df.iloc[max(0, n - bars):n]


def _spec(name: str):
    for spec in PLAYBOOK_REGISTRY:
        if spec.name == name:
            return spec
    raise AssertionError(f"no registry spec for {name}")


def _scan_runner(tmp_path, frame, **cfg_kw):
    runner = _runner(tmp_path, **cfg_kw)
    runner._stack = SimpleNamespace(
        frames={"H1": frame,
                "H4": _frame([(2000.0, 2001.0, 1999.0, 2000.5)]
                             * 40, freq="4h")})
    runner._live_setup = {"bias_d1": 1, "bias_h4": 1,
                          "regime": "trend",
                          "h4_context": "uptrend structure",
                          "size_mult": 1.0}
    tick_time = frame.index[-1] + pd.Timedelta(minutes=90)
    runner._last_tick = {"mid": float(frame["close"].iloc[-1]),
                         "bid": float(frame["close"].iloc[-1]) - 0.1,
                         "ask": float(frame["close"].iloc[-1]) + 0.1,
                         "spread": 0.2, "time": tick_time}
    return runner


def _evals(runner, playbook=None):
    out = [r for r in runner.journal.records()
           if r.get("type") == "evaluation"
           and r.get("kind") == "playbook_eval"]
    if playbook is not None:
        out = [r for r in out if r.get("playbook") == playbook]
    return out


# ------------------------------------------------- _scan_states cache

def test_scan_states_cached_per_frame(tmp_path):
    frame = _real_h1_ending(SWEEP_BAR, bars=60)
    runner = _runner(tmp_path)
    first = runner._scan_states(frame, "H1")
    assert len(first) == len(frame)
    assert runner._scan_states(frame, "H1") is first
    # A new close re-featurizes (different object, new length).
    nxt = pd.concat([frame, frame.iloc[[-1]]])
    nxt.index = list(frame.index) + [
        frame.index[-1] + pd.Timedelta(hours=1)]
    second = runner._scan_states(nxt, "H1")
    assert second is not first
    assert len(second) == len(nxt)
    # A different clock keeps its own entry.
    assert runner._scan_states(frame, "M15") is not first


# --------------------------------------- states threading in the scan

class _RecordingPB:
    """live_signal records the states argument, returns no signal."""

    def __init__(self, needs_states=False):
        self.needs_states = needs_states
        self.seen_states = None

    def live_signal(self, states, t, df=None):
        self.seen_states = states
        return []


def test_sweep_eval_passes_states_only_to_needs_states(tmp_path):
    frame = _real_h1_ending(SWEEP_BAR, bars=60)
    runner = _scan_runner(tmp_path, frame)
    made = {}

    def _factory(needs):
        def _make():
            pb = _RecordingPB(needs_states=needs)
            made[needs] = pb
            return pb
        _make.needs_states = needs
        return _make

    for needs in (True, False):
        spec = SimpleNamespace(
            name="recording", route="h1_roster",
            playbook_class=_factory(needs))
        runner._sweep_eval_one(
            spec, frame, clock="h1_close", intended_tf="H1",
            actual_tf="H1", source="live_close")
    assert len(made[True].seen_states) == len(frame)
    assert made[False].seen_states == []


# --------------------------------- sweep_reversal registers on the scan

def test_sweep_reversal_scan_registers_setup_on_real_data(tmp_path):
    frame = _real_h1_ending(SWEEP_BAR)
    runner = _scan_runner(tmp_path, frame)
    runner._sweep_eval_one(
        _spec("sweep_reversal"), frame,
        clock="h1_close", intended_tf="H1", actual_tf="H1",
        source="live_close")
    active = [s for s in runner._setup_store.active()
              if s.playbook == "sweep_reversal"]
    assert len(active) == 1
    setup = active[0]
    assert setup.direction == -1
    recs = _evals(runner, "sweep_reversal")
    assert len(recs) == 1
    assert recs[0]["reason_code"] == "path.setup_registered"
    assert recs[0]["decision"]["setup_id"] == setup.setup_id


def test_states_gated_playbooks_declare_needs_states():
    for name in ("sweep_reversal", "fvg_midline",
                 "trendline_retest"):
        assert getattr(
            _spec(name).playbook_class, "needs_states", False) is True
    # No other roster playbook needs states (the scan keeps [] for
    # them — featurizing every scan would waste ~0.6s per close).
    for spec in PLAYBOOK_REGISTRY:
        if spec.name not in ("sweep_reversal", "fvg_midline",
                             "trendline_retest"):
            assert getattr(spec.playbook_class, "needs_states",
                           False) is False, spec.name
