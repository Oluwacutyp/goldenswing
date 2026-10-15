"""Tests for the all-timeframe hierarchy (live + backtest, shared code).

The hierarchy: D1 bias -> H4 agreement -> H4 zone -> H1 trigger ->
session entry-clock confirmation.  Hard veto is OFF everywhere.
QM M5 is an annotation leg, never a gate.  Per-TF cursors never
rewind.  Live and backtest share ``evaluate_hierarchy`` — the parity
test proves both paths read the same verdict at the same bar.
"""
import math

import pandas as pd
import pytest

from goldenswing.backtest.mtf_hierarchy import run_hierarchy_backtest
from goldenswing.confluence import ConfluenceEngine
from goldenswing.execution.learning import TradeJournal
from goldenswing.execution.risk import RiskGuards, RiskConfig
from goldenswing.live.hierarchy import (TimeframeStack, evaluate_hierarchy,
                                        verdict_permits)
from goldenswing.live.multitf import atr14, find_swings


# ------------------------------------------------------------------ builders
def _idx(n, start="2026-01-01"):
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


def _frame(prices, start="2026-01-01"):
    """OHLC frame from closes: open = prev close, high/low = close ±0.25."""
    rows = []
    prev = prices[0]
    for c in prices:
        rows.append((prev, c + 0.25, c - 0.25, c))
        prev = c
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=_idx(len(prices), start))


def zigzag(direction, n=40, start=100.0, slope=0.3, amp=1.5,
           start_ts="2026-01-01"):
    """Trending zigzag: strict fractal swings (period-7 sine) on top
    of a linear trend.  ``direction`` +1 rises (bias +1), -1 falls."""
    prices = [start + direction * slope * t
              + amp * math.sin(2 * math.pi * t / 7.0)
              for t in range(n)]
    return _frame(prices, start_ts)


def event_frame(base=100.0, direction=1, n=40, start_ts="2026-01-01"):
    """40 bars whose LAST bar fires a BOS event in ``direction``.

    Prior bars zigzag gently around ``base`` (small bodies, no event);
    the last bar is a displacement bar clearing the most recent swing.
    For bull: last close = base + 1.0; for bear: base - 1.0.
    """
    prices = [base + 0.3 * math.sin(2 * math.pi * t / 7.0)
              for t in range(n - 1)]
    df = _frame(prices, start_ts)
    last_open = df["close"].iloc[-1]
    if direction == 1:
        last = (base - 2.0, base + 1.5, base - 3.0, base + 1.0)
    else:
        last = (base + 2.0, base + 3.0, base - 1.5, base - 1.0)
    last_row = pd.DataFrame([last], columns=["open", "high", "low", "close"],
                            index=_idx(1, start_ts) + pd.Timedelta(hours=n - 1))
    out = pd.concat([df, last_row])
    assert out.index[-1] == _idx(n, start_ts)[-1]
    return out


def _h4_poi(h4):
    """Most recent confirmed H4 swing low (the bull POI) + its band."""
    _h, lows = find_swings(h4["high"], h4["low"], 1)
    level = lows[-1][1]
    half = 0.5 * atr14(h4)
    return level, level - half, level + half


def passing_stack(asof_extra_days=0):
    """A stack where every hierarchy leg passes (swing_n=1).

    D1 bull, H4 bull with a POI, H1 trigger bar closing exactly AT
    the H4 POI with a BOS event, M15 + M30 entry frames with
    confirming events.  The H1 frame is built around the measured H4
    POI so the zone leg passes by construction.
    """
    d1 = zigzag(+1, 40, start=90.0, slope=0.5, amp=3.0)
    h4 = zigzag(+1, 40, start=95.0, slope=0.3, amp=1.5)
    poi, _lo, _hi = _h4_poi(h4)
    # H1: 39 zigzag bars around (poi - 1), then a displacement bar
    # closing exactly at the POI (inside the zone band, BOS_bull).
    h1_prices = [(poi - 1.0) + 0.3 * math.sin(2 * math.pi * t / 7.0)
                 for t in range(39)]
    h1 = _frame(h1_prices)
    last = pd.DataFrame([(poi - 3.0, poi + 0.5, poi - 4.0, poi)],
                        columns=["open", "high", "low", "close"],
                        index=_idx(1) + pd.Timedelta(hours=39))
    h1 = pd.concat([h1, last])
    m15 = event_frame(base=100.0, direction=1)
    m30 = event_frame(base=100.0, direction=1)
    m5 = event_frame(base=100.0, direction=1)
    return TimeframeStack.from_frames(
        {"D1": d1, "H4": h4, "H1": h1, "M15": m15, "M30": m30, "M5": m5})


KILLZONE_ASOF = pd.Timestamp("2026-01-05 13:00", tz="UTC")  # NY 08:00 EST
OFFKZ_ASOF = pd.Timestamp("2026-01-05 03:00", tz="UTC")      # Asia, no KZ


# ---------------------------------------------------------------- full pass
def test_full_pass_killzone_m15():
    v = evaluate_hierarchy(passing_stack(), asof=KILLZONE_ASOF, swing_n=1)
    assert v["allowed"] is True
    assert v["blocked_leg"] == ""
    assert v["direction"] == 1
    assert v["bias_d1"] == 1 and v["bias_h4"] == 1
    assert v["trigger"] == "BOS_bull"
    assert v["entry_clock"] == "M15"
    assert v["session"]["killzone"] == "ny_kz"
    assert v["veto_enabled"] is False
    assert v["qm"]["status"] == "qm_absent"
    assert v["legs"]["entry"]["passed"] is True


def test_full_pass_off_killzone_m30():
    v = evaluate_hierarchy(passing_stack(), asof=OFFKZ_ASOF, swing_n=1)
    assert v["allowed"] is True
    assert v["entry_clock"] == "M30"
    assert v["session"]["killzone"] is None
    assert v["veto_enabled"] is False


def test_session_clock_selection():
    v_kz = evaluate_hierarchy(passing_stack(), asof=KILLZONE_ASOF,
                              swing_n=1)
    v_off = evaluate_hierarchy(passing_stack(), asof=OFFKZ_ASOF,
                               swing_n=1)
    assert v_kz["session"]["entry_clock"] == "M15"  # in killzone
    assert v_off["session"]["entry_clock"] == "M30"  # elsewhere


# ------------------------------------------------------------- hierarchy legs
def test_h4_against_d1_blocks():
    stack = passing_stack()
    stack.frames["H4"] = zigzag(-1, 40, start=110.0, slope=0.3, amp=1.5)
    v = evaluate_hierarchy(stack, asof=KILLZONE_ASOF, swing_n=1)
    assert v["allowed"] is False
    assert v["blocked_leg"] == "h4_agree"
    assert "h4_against_d1" in v["legs"]["h4_agree"]["reason"]
    assert v["veto_enabled"] is False


def test_h4_flat_blocks_no_permission():
    stack = passing_stack()
    flat = pd.DataFrame(
        [(100.0, 100.5, 99.5, 100.0)] * 40,
        columns=["open", "high", "low", "close"], index=_idx(40))
    stack.frames["H4"] = flat
    v = evaluate_hierarchy(stack, asof=KILLZONE_ASOF, swing_n=1)
    assert v["allowed"] is False
    assert v["blocked_leg"] == "h4_agree"
    assert "h4_flat" in v["legs"]["h4_agree"]["reason"]


def test_h1_trigger_without_h4_zone_never_fires():
    stack = passing_stack()
    poi, _lo, _hi = _h4_poi(stack.frames["H4"])
    # H1 trigger bar closes far ABOVE the H4 POI band: chasing, blocked.
    h1 = stack.frames["H1"].copy()
    h1.iloc[-1] = (poi + 7.0, poi + 11.0, poi + 6.0, poi + 10.0)
    stack.frames["H1"] = h1
    v = evaluate_hierarchy(stack, asof=KILLZONE_ASOF, swing_n=1)
    assert v["allowed"] is False
    assert v["blocked_leg"] == "h4_zone"
    assert v["legs"]["h4_zone"]["reason"] == "entry_above_poi"


def test_m5_entry_without_h1_trigger_never_fires():
    stack = passing_stack()
    poi, _lo, _hi = _h4_poi(stack.frames["H4"])
    # H1: pure zigzag centred on the POI — inside the zone, but NO
    # trigger event on the last bar (small body, no displacement).
    # The M15 frame still has its confirming event: it must NOT rescue.
    h1_prices = [poi + 0.3 * math.sin(2 * math.pi * t / 7.0)
                 for t in range(40)]
    stack.frames["H1"] = _frame(h1_prices)
    v = evaluate_hierarchy(stack, asof=KILLZONE_ASOF, swing_n=1)
    assert v["allowed"] is False
    assert v["blocked_leg"] == "h1_trigger"
    assert v["legs"]["h1_trigger"]["reason"] == "no_trigger"


def test_entry_leg_blocks_without_confirmation():
    stack = passing_stack()
    # Entry frames with NO structure event anywhere in the window:
    # flat bars have no swings at all, so no event can fire.
    flat = pd.DataFrame(
        [(100.0, 100.5, 99.5, 100.0)] * 40,
        columns=["open", "high", "low", "close"], index=_idx(40))
    stack.frames["M15"] = flat
    stack.frames["M30"] = flat
    v = evaluate_hierarchy(stack, asof=KILLZONE_ASOF, swing_n=1)
    assert v["allowed"] is False
    assert v["blocked_leg"] == "entry"
    assert "entry_no_confirm" in v["legs"]["entry"]["reason"]


def test_entry_leg_blocks_against_bias():
    stack = passing_stack()
    # M15 fires a BEAR event while the bias is bull: against-bias entry.
    stack.frames["M15"] = event_frame(base=100.0, direction=-1)
    v = evaluate_hierarchy(stack, asof=KILLZONE_ASOF, swing_n=1)
    assert v["allowed"] is False
    assert v["blocked_leg"] == "entry"
    assert "entry_against_bias" in v["legs"]["entry"]["reason"]


def test_entry_leg_confirms_on_earlier_window_bar():
    # The M15 event fires 2 bars before the last bar; the last bars
    # are chop.  The confirmation happened during the trigger H1
    # bar's formation, so the entry leg still passes.
    stack = passing_stack()
    m15 = event_frame(base=100.0, direction=1)
    chop = _frame([100.2, 100.1], start="2026-01-03")
    # Chop bars must carry timestamps AFTER the event frame, and the
    # frame must stay sorted: rebuild with a later index.
    chop.index = m15.index[-1] + pd.Timedelta(hours=1) + pd.to_timedelta(
        [0, 1], unit="h")
    m15 = pd.concat([m15, chop])
    stack.frames["M15"] = m15
    v = evaluate_hierarchy(stack, asof=KILLZONE_ASOF, swing_n=1)
    assert v["allowed"] is True
    assert v["legs"]["entry"]["passed"] is True
    stack = passing_stack()
    flat = pd.DataFrame(
        [(100.0, 100.5, 99.5, 100.0)] * 40,
        columns=["open", "high", "low", "close"], index=_idx(40))
    stack.frames["D1"] = flat
    v = evaluate_hierarchy(stack, asof=KILLZONE_ASOF, swing_n=1)
    assert v["allowed"] is False
    assert v["blocked_leg"] == "d1_bias"


# ---------------------------------------------------------------- QM leg
def test_qm_leg_never_gates():
    # QM against the bias: annotation only, the verdict still allows.
    qm_against = {"direction": -1, "touch_ts": KILLZONE_ASOF,
                  "qm_level": 100.0, "head": 105.0, "neckline": 95.0}
    v = evaluate_hierarchy(passing_stack(), asof=KILLZONE_ASOF,
                           swing_n=1, qm_touch=qm_against)
    assert v["allowed"] is True
    assert v["qm"]["status"] == "qm_against_bias"
    # QM confirming + recent: qm_confirmed, still allowed.
    qm_ok = dict(qm_against, direction=1)
    v2 = evaluate_hierarchy(passing_stack(), asof=KILLZONE_ASOF,
                            swing_n=1, qm_touch=qm_ok)
    assert v2["allowed"] is True
    assert v2["qm"]["status"] == "qm_confirmed"
    # QM confirming but stale (>30 min old): qm_stale, still allowed.
    qm_stale = dict(qm_ok, touch_ts=KILLZONE_ASOF - pd.Timedelta(hours=2))
    v3 = evaluate_hierarchy(passing_stack(), asof=KILLZONE_ASOF,
                            swing_n=1, qm_touch=qm_stale)
    assert v3["allowed"] is True
    assert v3["qm"]["status"] == "qm_stale"


def test_verdict_permits_direction():
    v = evaluate_hierarchy(passing_stack(), asof=KILLZONE_ASOF, swing_n=1)
    assert verdict_permits(v, +1) == (True, "")
    ok, reason = verdict_permits(v, -1)
    assert ok is False and reason.startswith("hierarchy:direction_mismatch")
    blocked = evaluate_hierarchy(passing_stack(), asof=KILLZONE_ASOF,
                                 swing_n=1)
    blocked["allowed"] = False
    blocked["blocked_leg"] = "h4_agree"
    blocked["legs"]["h4_agree"] = {"passed": False,
                                   "reason": "h4_against_d1: test"}
    ok, reason = verdict_permits(blocked, +1)
    assert ok is False and reason.startswith("hierarchy:h4_agree")


# ------------------------------------------------------------------- stack
def test_stack_cursor_never_rewinds():
    stack = TimeframeStack()
    df1 = _frame([100.0 + i * 0.1 for i in range(10)])
    acc, rej = stack.append("H1", df1)
    assert (acc, rej) == (10, 0)
    assert stack.last_ts("H1") == df1.index[-1]
    # Out-of-order batch: older bars are rejected, cursor unmoved.
    df_old = _frame([99.0 + i * 0.1 for i in range(5)],
                    start="2025-12-31 20:00")
    acc, rej = stack.append("H1", df_old)
    assert acc == 0 and rej == 5
    assert stack.last_ts("H1") == df1.index[-1]
    # Duplicate re-delivery: all rejected.
    acc, rej = stack.append("H1", df1)
    assert acc == 0 and rej == 10
    # Forward append: only the new bars land.
    df2 = _frame([101.0 + i * 0.1 for i in range(5)],
                 start="2026-01-01 10:00")
    acc, rej = stack.append("H1", df2)
    assert acc == 5 and rej == 0
    assert stack.last_ts("H1") == df2.index[-1]
    # Mixed batch (some old, some new): old dropped, new accepted.
    mixed = pd.concat([df1.iloc[:3], df2.iloc[:2],
                       _frame([102.0, 102.1], start="2026-01-01 15:00")])
    acc, rej = stack.append("H1", mixed)
    assert acc == 2 and rej == 5
    assert stack.last_ts("H1") == mixed.index[-1]


def test_stack_rejects_in_batch_duplicates():
    stack = TimeframeStack()
    df = _frame([100.0 + i * 0.1 for i in range(5)])
    doubled = pd.concat([df, df])
    acc, rej = stack.append("M5", doubled)
    assert acc == 5 and rej == 0
    assert len(stack.frames["M5"]) == 5


# ------------------------------------------------------- confluence wiring
def _real_ctx_and_signal():
    from goldenswing.backtest.data import load_gold_h1
    from goldenswing.features.engine import read_and_featurize
    from goldenswing.playbooks.base import EngineContext
    from goldenswing.playbooks.fvg_midline import FVGMidlinePlaybook
    df = load_gold_h1().iloc[-3000:].copy()
    states = read_and_featurize(df, None)
    ctx = EngineContext(df=df, states=states, years=0.5,
                        instrument="XAUUSD")
    pb = FVGMidlinePlaybook()
    sig = None
    for t in range(len(df)):
        sigs = pb.live_signal(states, t, df)
        if sigs:
            sig = sigs[0]
            break
    assert sig is not None, "test window needs a real FVG signal"
    return pb, sig, ctx


def test_confluence_gate_enforces_hierarchy(tmp_path):
    pb, sig, ctx = _real_ctx_and_signal()
    engine = ConfluenceEngine(
        playbooks=[pb],
        risk_guards=RiskGuards(tmp_path / "risk.json", RiskConfig()),
        journal=TradeJournal(tmp_path / "journal.jsonl"),
        risk_mode="shadow")
    blocked = {"allowed": False, "blocked_leg": "h4_agree",
               "legs": {"h4_agree": {"passed": False,
                                     "reason": "h4_against_d1: test"}},
               "direction": 1}
    ok, reason = engine.gate_live_signal(
        pb, sig, ctx, hierarchy_verdict=blocked)
    assert ok is False
    assert reason.startswith("hierarchy:h4_agree")
    # An allowing verdict with the signal's own direction does NOT
    # block on hierarchy (later gates may still apply for other
    # reasons, but never with a hierarchy: prefix).
    allowed = {"allowed": True, "blocked_leg": "",
               "legs": {}, "direction": int(sig.direction)}
    ok, reason = engine.gate_live_signal(
        pb, sig, ctx, hierarchy_verdict=allowed)
    assert not reason.startswith("hierarchy:")
    # No verdict (legacy feed): the hierarchy gate is skipped, not faked.
    ok2, reason2 = engine.gate_live_signal(pb, sig, ctx)
    assert not reason2.startswith("hierarchy:")


# ------------------------------------------------- live/backtest parity
def _synthetic_frames(n_h1=120):
    """Long synthetic stack: H1 zigzag (no trigger anywhere), other
    TFs trending.  M15 extends past the H1 window to prove causal
    slicing excludes future bars on both paths."""
    d1 = zigzag(+1, 60, start=90.0, slope=0.5, amp=3.0)
    h4 = zigzag(+1, 60, start=95.0, slope=0.3, amp=1.5)
    poi, _lo, _hi = _h4_poi(h4)
    h1_prices = [poi + 0.3 * math.sin(2 * math.pi * t / 7.0)
                 for t in range(n_h1)]
    h1 = _frame(h1_prices)
    m15 = event_frame(base=100.0, direction=1, n=200,
                      start_ts="2026-01-01")
    m30 = event_frame(base=100.0, direction=1, n=200,
                      start_ts="2026-01-01")
    m5 = event_frame(base=100.0, direction=1, n=200,
                     start_ts="2026-01-01")
    return {"D1": d1, "H4": h4, "H1": h1, "M15": m15, "M30": m30,
            "M5": m5}


def test_live_backtest_parity_on_hierarchy_path():
    frames = _synthetic_frames()
    result = run_hierarchy_backtest(frames, swing_n=1, max_bars=120)
    assert result["evaluated_bars"] > 0
    assert result["veto_enabled"] is False
    # Every verdict the backtest logged must equal a direct live-style
    # evaluation on the causally-sliced prefix frames at the same bar.
    h1 = frames["H1"]
    for vrec in result["verdicts"][::10]:
        i = vrec["i"]
        asof = h1.index[i]
        h1_close = asof + pd.Timedelta(minutes=60)
        prefix = {}
        for tf, frame in frames.items():
            minutes = {"D1": 1440, "H4": 240, "H1": 60, "M15": 15,
                       "M30": 30, "M5": 5}[tf]
            cutoff = h1_close - pd.Timedelta(minutes=minutes)
            prefix[tf] = frame[frame.index <= cutoff]
        direct = evaluate_hierarchy(prefix, asof=asof, swing_n=1)
        assert direct["allowed"] == vrec["allowed"], f"bar {i}"
        assert direct["blocked_leg"] == vrec["blocked_leg"], f"bar {i}"
        assert direct["direction"] == vrec["direction"], f"bar {i}"
        assert direct["entry_clock"] == vrec["entry_clock"], f"bar {i}"
    # Gate counts cover every evaluated bar, exactly once.
    assert sum(result["gate_counts"].values()) == result["evaluated_bars"]
    assert set(result["gate_counts"]) <= {
        "data", "d1_bias", "h4_agree", "h4_zone", "h1_trigger", "entry",
        "passed"}


# ------------------------------------------------------------ multi feed
def test_multi_oanda_feed_interface(monkeypatch):
    import goldenswing.feed.oanda as oanda_mod
    from goldenswing.feed import MultiOandaFeed

    def _candles(base, n=3):
        # Recent timestamps: the feed's staleness check runs on real
        # wall-clock time, so fixed 2026-01-01 bars would read stale.
        now = pd.Timestamp.now(tz="UTC").floor("h")
        return [{"time": (now - pd.Timedelta(hours=n - 1 - i)).isoformat(),
                 "complete": True,
                 "mid": {"o": base, "h": base + 1, "l": base - 1,
                         "c": base + 0.05},
                 "volume": 10} for i in range(n)]

    def _fake_get(path, params, timeout=15.0):
        if path == "/v3/accounts":
            return {"accounts": [{"id": "acct-1"}]}
        if "/pricing" in path:
            # Mid 102.05 == the H1 forming bar's close (102.00 + 0.05):
            # the cross-check must pass for poll_status to be ok.
            return {"prices": [{"bids": [{"price": "102.00"}],
                                "asks": [{"price": "102.10"}],
                                "time": "2026-01-01T03:00:00Z"}]}
        # candles: granularity-specific base so TFs are distinguishable.
        # Keys are OANDA tokens: the D1 feed requests "D" (OANDA's
        # CandlestickGranularity enum has no "D1").
        gran = params.get("granularity", "H1")
        requested.append(gran)
        base = {"D": 100.0, "H4": 101.0, "H1": 102.0, "M15": 103.0,
                "M30": 104.0, "M5": 105.0}[gran]
        return {"candles": _candles(base)}

    requested: list = []
    monkeypatch.setattr(oanda_mod, "_get", _fake_get)
    feed = MultiOandaFeed()
    all_frames = feed.warm_start_all()
    assert set(all_frames) == {"D1", "H4", "H1", "M15", "M30", "M5"}
    # The D1 feed must request OANDA's "D" token, never the invalid "D1".
    assert "D" in requested
    assert "D1" not in requested
    for tf, df in all_frames.items():
        assert len(df) == 3, tf
    # Legacy single-TF interface still serves H1.
    assert len(feed.warm_start()) == 3
    # fetch_new_all respects per-TF cursors (nothing newer -> empty).
    since = {tf: df.index[-1] for tf, df in all_frames.items()}
    assert all(len(df) == 0 for df in
               feed.fetch_new_all(since).values())
    # A stale cursor for one TF returns its bars for that TF only.
    since["H1"] = all_frames["H1"].index[0]
    fresh = feed.fetch_new_all(since)
    assert len(fresh["H1"]) == 2
    assert len(fresh["D1"]) == 0
    assert feed.poll_status()["ok"] is True


# ------------------------------------------------------- runner wiring
class _FakeMultiFeed:
    """Multi-TF fake feed: warm_start_all / fetch_new_all + H1 legacy."""

    def __init__(self, frames, warm):
        self._frames = frames
        self._warm = warm

    def warm_start_all(self):
        return {tf: df.iloc[: self._warm].copy()
                for tf, df in self._frames.items()}

    def fetch_new_all(self, since_by_tf):
        out = {}
        for tf, df in self._frames.items():
            since = (since_by_tf or {}).get(tf)
            out[tf] = df[df.index > since].copy() if since is not None \
                else df.copy()
        return out

    def warm_start(self):
        return self._frames["H1"].iloc[: self._warm].copy()

    def fetch_new(self, since):
        return self._frames["H1"][self._frames["H1"].index > since].copy()

    def poll_status(self):
        return {"ok": True, "reason": None}

    @property
    def poll_interval(self):
        return 0


def test_runner_paper_once_multitf_persists_stack(tmp_path, monkeypatch):
    from goldenswing.backtest.data import load_gold_h1
    from goldenswing.execution.runner import LiveRunner, RunnerConfig

    monkeypatch.delenv("GS_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("GS_TELEGRAM_CHAT_ID", raising=False)
    full = load_gold_h1().iloc[-600:].copy()
    frames = {"H1": full}
    # Other TFs: resampled from the same H1 history (plausible stack,
    # hierarchy may block everything — the test asserts plumbing).
    from goldenswing.backtest.mtf12 import resample_bars
    frames["H4"] = resample_bars(full, "H4")
    frames["D1"] = resample_bars(full, "D1")
    frames["M15"] = full.iloc[::1].copy()
    frames["M30"] = full.iloc[::1].copy()
    frames["M5"] = full.iloc[::1].copy()
    feed = _FakeMultiFeed(frames, warm=590)
    cfg = RunnerConfig(
        journal_path=tmp_path / "journal.jsonl",
        risk_state_path=tmp_path / "risk.json",
        kill_switch_path=tmp_path / "KILL",
        state_dir=tmp_path / "live_state",
        forecast_bundle_path=None)
    store_dir = tmp_path / "paper_state"
    runner = LiveRunner(cfg, senders=[])
    s1 = runner.run_paper_once(feed, state_store_path=store_dir,
                               peers={}, poll_commands=False)
    assert s1["bars"] == 0
    assert "hierarchy_blocked" in s1
    assert (store_dir / "stack.pkl").exists()
    import json as _json
    meta = _json.loads((store_dir / "meta.json").read_text())
    assert "stack_cursors" in meta and "H1" in meta["stack_cursors"]
    # Tick 2: warm resume from the persisted stack, 10 new H1 bars.
    runner2 = LiveRunner(cfg, senders=[])
    s2 = runner2.run_paper_once(feed, state_store_path=store_dir,
                                peers={}, poll_commands=False)
    assert s2["bars"] == 10
    assert "hierarchy_blocked" in s2


def test_runner_hierarchy_required_blocks_without_stack(tmp_path,
                                                        monkeypatch):
    from goldenswing.backtest.data import load_gold_h1
    from goldenswing.execution.runner import LiveRunner, RunnerConfig
    from goldenswing.playbooks.fvg_midline import FVGMidlinePlaybook

    monkeypatch.delenv("GS_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("GS_TELEGRAM_CHAT_ID", raising=False)
    cfg = RunnerConfig(
        journal_path=tmp_path / "journal.jsonl",
        risk_state_path=tmp_path / "risk.json",
        kill_switch_path=tmp_path / "KILL",
        state_dir=tmp_path / "live_state",
        forecast_bundle_path=None,
        hierarchy_required=True)
    runner = LiveRunner(cfg, senders=[])
    df = load_gold_h1().iloc[-3000:].copy()
    # Find a bar where the FVG sleeve fires live.
    from goldenswing.features.engine import read_and_featurize
    from goldenswing.playbooks.base import EngineContext
    states = read_and_featurize(df, {})
    pb = FVGMidlinePlaybook()
    t_hit = None
    for t in range(len(df)):
        if pb.live_signal(states, t, df):
            t_hit = t
            break
    assert t_hit is not None
    engine = runner._get_engine()
    stats = {"bars": 0, "signals": 0, "alerts": 0, "executed": 0,
             "shadow": 0, "gated": 0, "stale_polls": 0,
             "hierarchy_blocked": 0}
    runner._paper_loop_bar(
        df, t_hit, [FVGMidlinePlaybook()], engine, {}, {"fvg_midline"},
        0.0, 0, stats, {}, 0.5, stack=None, qm_touch=None)
    assert stats["hierarchy_blocked"] >= 1
    assert stats["signals"] == 0
