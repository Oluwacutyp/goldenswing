"""Arming-gate fix (owner-approved 2026-10-20): correct-side POI
selection + the continuation arm.

Reproduces the 2026-10-19/20 live geometry from the /status prints:
D1 -1, last price ~4140, the most recent confirmed H4 swing high
STALE below price (4073.2 — the old ``zones_by_side`` pick), H1 ref
4155.7, containment failing against the stale band on every refresh
-> playbooks never evaluated -> zero signals journaled since
deployment.  The old gate had exactly one trend pattern
(pullback-nesting against that single band) and no continuation /
breakdown arm at all.

Scenarios: (a) pullback arms against the NEAREST OVERHEAD supply,
not the stale band; (b) a closed-H1 break of the ref low arms a
continuation setup; (c) the old geometry with no overhead supply at
all -> no arm, honest ``no_h4_poi`` reason; (c2) the pristine bug
itself — the H1 ref nesting in the stale band armed a SHORT anchored
under the market; (d) mirrored LONG cases; (e) one arm per break at
runner level (fires once, next H1 bar does not re-arm the same
break); (b2/b3) invalidation by a close back beyond the level, and a
NEW break after invalidation is a NEW arm.
"""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from goldenswing.execution.executor import OrderResult, Position
from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.live.hierarchy import (TimeframeStack, live_setup,
                                        live_setup_reason)
from goldenswing.live.multitf import (atr14, evaluate_bias, find_swings,
                                      poi_for_direction, zones_by_side)


# ------------------------------------------------------------------ builders
def _frame(closes, idx):
    rows = []
    prev = closes[0]
    for c in closes:
        rows.append((prev, c + 0.25, c - 0.25, c))
        prev = c
    return pd.DataFrame(
        rows, columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex(idx, tz="UTC")).assign(
            volume=100, real_volume=0, spread=30)


def _zigzag(direction, n, start, slope, amp, idx):
    prices = [start + direction * slope * t
              + amp * math.sin(2 * math.pi * t / 7.0)
              for t in range(n)]
    return _frame(prices, idx)


def _skeleton(points, n):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [float(v) for v in np.interp(range(n), xs, ys)]


def _d1_short(idx_end="2026-10-13"):
    idx = pd.date_range(end=idx_end, periods=40, freq="D", tz="UTC")
    d1 = _zigzag(-1, 40, 4300.0, 5.0, 30.0, idx)
    assert evaluate_bias(d1, swing_n=3) == -1
    return d1


def _h4_live_backdrop():
    """H4 closes ending at 4140 with confirmed swing highs overhead
    (4260/4230/4205/4185) AND a stale most-recent swing high at ~4073
    (below price) — the live 2026-10-20 geometry."""
    idx = pd.date_range(end="2026-10-14 04:00", periods=44, freq="4h",
                        tz="UTC")
    closes = _skeleton(
        [(0, 4200), (3, 4260), (6, 4180), (9, 4230), (12, 4160),
         (15, 4205), (18, 4145), (21, 4185), (24, 4045), (27, 4073),
         (30, 4030), (33, 4062), (36, 4098), (39, 4122), (41, 4133),
         (43, 4140)], 44)
    return _frame(closes, idx)


def _h4_stale_only():
    """H4 whose ONLY confirmed swing highs sit below the last close
    (4140): the old selector's band is the whole map."""
    idx = pd.date_range(end="2026-10-14 04:00", periods=44, freq="4h",
                        tz="UTC")
    closes = _skeleton(
        [(0, 4050), (3, 4080), (6, 4045), (9, 4073), (12, 4040),
         (15, 4066), (18, 4090), (21, 4105), (24, 4112), (27, 4118),
         (30, 4124), (33, 4129), (36, 4133), (39, 4136), (41, 4138),
         (43, 4140)], 44)
    return _frame(closes, idx)


def _h1_pullback_into(poi_level, h4):
    """H1 rising zigzag shifted so its last confirmed swing high
    lands exactly on ``poi_level`` (containment by construction)."""
    idx = pd.date_range(end="2026-10-14 07:00", periods=40, freq="h",
                        tz="UTC")
    h1 = _zigzag(+1, 40, 4150.0, 3.0, 15.0, idx)
    highs, _ = find_swings(h1["high"].to_numpy(),
                           h1["low"].to_numpy(), 3)
    shift = poi_level - float(highs[-1][1])
    for col in ("open", "high", "low", "close"):
        h1[col] = h1[col] + shift
    return h1


def _h1_break_closes(drop=(4122.0, 4118.0, 4115.0), extra=()):
    """H1 closes: gentle downtilt sine channel (4 cycles), a rise,
    then ``drop`` closes breaking the last confirmed swing low."""
    closes = [4137.0 - 0.15 * t + 7.0 * math.sin(2 * math.pi * t / 7.0)
              for t in range(31)]
    closes += list(drop)
    closes += list(extra)
    return closes


def _h1_break_frame(drop=(4122.0, 4118.0, 4115.0), extra=()):
    closes = _h1_break_closes(drop, extra)
    idx = pd.date_range(end="2026-10-14 07:00", periods=len(closes),
                        freq="h", tz="UTC")
    return _frame(closes, idx)


def _short_frames(h4, h1):
    return {"D1": _d1_short(), "H4": h4, "H1": h1}


def _mirror(frames, k=8300.0):
    out = {}
    for tf, f in frames.items():
        out[tf] = pd.DataFrame(
            {"open": k - f["close"], "high": k - f["low"],
             "low": k - f["high"], "close": k - f["open"],
             "volume": f["volume"], "real_volume": f["real_volume"],
             "spread": f["spread"]}, index=f.index)
    return out


def _overhead_poi(h4):
    """The F1 pick: freshest confirmed swing high at/above the
    last close (in this descending-overhead geometry also the
    nearest by price)."""
    highs, _ = find_swings(h4["high"].to_numpy(),
                           h4["low"].to_numpy(), 3)
    last_close = float(h4["close"].iloc[-1])
    above = [float(p) for _, p in highs if float(p) >= last_close]
    return above[-1] if above else None


# ------------------------------------------------------------- F1: selection
def test_poi_for_direction_picks_freshest_overhead():
    h4 = _h4_live_backdrop()
    zone = poi_for_direction(h4, -1)
    assert zone is not None
    assert zone["kind"] == "resistance"
    assert zone["level"] == pytest.approx(_overhead_poi(h4))
    # ... and it is NOT the stale most-recent swing the old
    # selector returns.
    stale = zones_by_side(h4, swing_n=3)["resistance"]
    assert stale["level"] < float(h4["close"].iloc[-1])
    assert zone["level"] != pytest.approx(stale["level"])
    # Band built exactly as zones_by_side builds it.
    atr = atr14(h4)
    assert zone["top"] == pytest.approx(zone["level"] + 0.5 * atr)
    assert zone["bottom"] == pytest.approx(zone["level"] - 0.5 * atr)


def test_poi_for_direction_none_when_no_correct_side():
    h4 = _h4_stale_only()
    assert poi_for_direction(h4, -1) is None
    # The old selector still returns the stale below-price band —
    # that asymmetry IS root cause 2.
    assert zones_by_side(h4, swing_n=3)["resistance"] is not None


# ------------------------------------------------------- (a) pullback, fixed
def test_a_pullback_arms_against_overhead_supply():
    h4 = _h4_live_backdrop()
    poi = _overhead_poi(h4)
    assert poi is not None
    h1 = _h1_pullback_into(poi, h4)
    frames = _short_frames(h4, h1)
    setup = live_setup(frames)
    assert setup is not None
    assert setup.get("arm") != "continuation"
    assert setup["direction"] == -1
    assert setup["zone"]["kind"] == "resistance"
    # The armed POI is the overhead supply, NOT the stale band.
    assert setup["zone"]["level"] == pytest.approx(poi)
    assert live_setup_reason(frames) == "armed"


# --------------------------------------------------- (b) continuation, new
def test_b_continuation_arms_on_closed_h1_break():
    h4 = _h4_live_backdrop()
    h1 = _h1_break_frame()
    frames = _short_frames(h4, h1)
    _highs, lows = find_swings(h1["high"].to_numpy(),
                               h1["low"].to_numpy(), 3)
    ref_low = float(lows[-1][1])
    atr_h1 = atr14(h1)
    buf = 0.1 * atr_h1
    # Preconditions: the last close broke ref_low by >= buf, and
    # pullback containment does NOT hold (isolated continuation).
    assert float(h1["close"].iloc[-1]) <= ref_low - buf
    setup = live_setup(frames)
    assert setup is not None
    assert setup["arm"] == "continuation"
    assert setup["regime"] == "trend"
    assert setup["direction"] == -1
    assert setup["bias_d1"] == -1
    assert setup["ref_level"] == pytest.approx(ref_low)
    # Zone = the broken level's band (+/- 0.5 x ATR_H1).
    assert setup["zone"]["level"] == pytest.approx(ref_low)
    assert setup["zone"]["top"] == pytest.approx(ref_low + 0.5 * atr_h1)
    assert setup["zone"]["bottom"] == pytest.approx(
        ref_low - 0.5 * atr_h1)
    # Every field the pullback setup carries is present.
    for key in ("bias_d1", "bias_h4", "h4_context", "direction",
                "zone", "ref_high", "minor_high", "ref_low",
                "minor_low", "ref_level", "atr_h1", "buf", "h4_highs",
                "h4_lows", "swing_n"):
        assert key in setup, key
    assert setup["break_key"]
    assert live_setup_reason(frames) == "armed"


def test_b_no_continuation_on_sub_buffer_drift():
    # Closes creep just past the level but never by >= buf on a
    # closed bar: a drift across is not a break.
    h4 = _h4_live_backdrop()
    probe = _h1_break_frame()
    _h, lows = find_swings(probe["high"].to_numpy(),
                           probe["low"].to_numpy(), 3)
    ref_low = float(lows[-1][1])
    h1 = _h1_break_frame(drop=(ref_low - 0.2, ref_low - 0.3))
    frames = _short_frames(h4, h1)
    setup = live_setup(frames)
    assert setup is None or setup.get("arm") != "continuation"


def test_b2_close_back_beyond_level_invalidates():
    h4 = _h4_live_backdrop()
    h1 = _h1_break_frame(extra=(4130.0,))
    frames = _short_frames(h4, h1)
    setup = live_setup(frames)
    assert setup is None or setup.get("arm") != "continuation"


def test_b3_new_break_after_invalidation_is_new_arm():
    h4 = _h4_live_backdrop()
    base = live_setup(_short_frames(h4, _h1_break_frame()))
    assert base is not None and base["arm"] == "continuation"
    # SAME timeline extended: one close back above the level
    # (invalidation), then a fresh break — a NEW break bar, hence
    # a new break_key and a new arm.
    closes = _h1_break_closes(extra=(4130.0, 4119.0, 4112.0, 4108.0))
    idx = pd.date_range(end="2026-10-14 11:00", periods=len(closes),
                        freq="h", tz="UTC")
    h1 = _frame(closes, idx)
    setup = live_setup(_short_frames(h4, h1))
    assert setup is not None
    assert setup["arm"] == "continuation"
    assert setup["break_key"] != base["break_key"]


# ------------------------------------------- (c) old geometry, honest reason
def test_c_old_geometry_no_arm_honest_reason():
    h4 = _h4_stale_only()
    # H1 ref (last swing high) at the live prints' 4155.7: above
    # every H4 band that exists — with the fixed selector there is
    # no POI on the short side at all.
    idx = pd.date_range(end="2026-10-14 07:00", periods=40, freq="h",
                        tz="UTC")
    h1 = _zigzag(+1, 40, 4150.0, 3.0, 15.0, idx)
    highs, _ = find_swings(h1["high"].to_numpy(),
                           h1["low"].to_numpy(), 3)
    shift = 4155.7 - float(highs[-1][1])
    for col in ("open", "high", "low", "close"):
        h1[col] = h1[col] + shift
    frames = _short_frames(h4, h1)
    assert live_setup(frames) is None
    reason = live_setup_reason(frames)
    assert reason.startswith("no_h4_poi")
    assert "above" in reason  # names the missing side honestly


def test_c2_pristine_bug_ref_nesting_in_stale_band():
    # THE bug: H1's last confirmed swing high sits inside the stale
    # band (fractal lag during the rally).  Pristine code ARMED this
    # — a SHORT whose zone is under the market.  Fixed: no arm.
    h4 = _h4_stale_only()
    stale = zones_by_side(h4, swing_n=3)["resistance"]
    # Segment truncated 3 bars after its last peak (bar 30), so the
    # peak is the frame's last confirmed swing high even after the
    # rally is appended (later zigzag peaks never get their 3
    # confirming bars inside the segment, and the rally is
    # monotonic so it confirms none either).
    idx = pd.date_range(end="2026-10-14 02:00", periods=34, freq="h",
                        tz="UTC")
    h1 = _zigzag(+1, 34, 4150.0, 3.0, 15.0, idx)
    highs, _ = find_swings(h1["high"].to_numpy(),
                           h1["low"].to_numpy(), 3)
    shift = stale["level"] - float(highs[-1][1])
    for col in ("open", "high", "low", "close"):
        h1[col] = h1[col] + shift
    # Rally after the swing high: closes rise well above the stale
    # band with no new confirmed swing high.
    last = float(h1["close"].iloc[-1])
    rally = [last + step for step in
             (15.0, 30.0, 44.0, 57.0, 69.0, 80.0, 90.0, 99.0)]
    ridx = pd.date_range("2026-10-14 03:00", periods=len(rally),
                         freq="h", tz="UTC")
    h1 = pd.concat([h1, _frame(rally, ridx)])
    _rh, _rl = find_swings(h1["high"].to_numpy(),
                           h1["low"].to_numpy(), 3)
    assert float(_rh[-1][1]) == pytest.approx(stale["level"])
    assert float(h1["close"].iloc[-1]) > stale["top"] + 40
    frames = _short_frames(h4, h1)
    assert live_setup(frames) is None
    assert live_setup_reason(frames).startswith("no_h4_poi")


# ------------------------------------------------------------ (d) LONG mirror
def test_d1_long_pullback_arms_against_support_below():
    h4 = _h4_live_backdrop()
    poi = _overhead_poi(h4)
    h1 = _h1_pullback_into(poi, h4)
    frames = _mirror(_short_frames(h4, h1))
    assert evaluate_bias(frames["D1"], swing_n=3) == +1
    setup = live_setup(frames)
    assert setup is not None
    assert setup.get("arm") != "continuation"
    assert setup["direction"] == +1
    assert setup["zone"]["kind"] == "support"
    # Mirrored nearest-below support.
    assert setup["zone"]["level"] == pytest.approx(8300.0 - poi)


def test_d2_long_continuation_arms_on_break_up():
    frames = _mirror(_short_frames(_h4_live_backdrop(),
                                    _h1_break_frame()))
    assert evaluate_bias(frames["D1"], swing_n=3) == +1
    setup = live_setup(frames)
    assert setup is not None
    assert setup["arm"] == "continuation"
    assert setup["direction"] == +1
    _highs, lows = find_swings(frames["H1"]["high"].to_numpy(),
                               frames["H1"]["low"].to_numpy(), 3)
    # The broken level is the mirrored ref low -> ref high.
    assert setup["ref_level"] == pytest.approx(8300.0 - float(
        find_swings(_h1_break_frame()["high"].to_numpy(),
                    _h1_break_frame()["low"].to_numpy(), 3)[1][-1][1]))


# ------------------------------------------------- (e) runner: one arm/break
class FakeExecutor:
    def __init__(self):
        self.orders: list[dict] = []
        self._ticket = 9000
        self._positions: list[Position] = []

    def connect(self, **kw):
        return True, "fake connected"

    def symbol_spec(self, symbol):
        from goldenswing.execution.risk import gold_spec
        return gold_spec()

    def positions(self, symbol=None):
        return [p for p in self._positions
                if symbol is None or p.symbol == symbol]

    def place_order(self, request):
        self._ticket += 1
        self.orders.append({"ticket": self._ticket,
                            "symbol": request.symbol,
                            "direction": request.direction,
                            "volume": request.volume,
                            "sl": request.stop_loss,
                            "tp": request.take_profit})
        self._positions.append(Position(
            ticket=self._ticket, symbol=request.symbol,
            direction=request.direction, volume=request.volume,
            entry_price=request.entry_ref, sl=request.stop_loss,
            tp=request.take_profit, magic=request.magic,
            alert_id=request.alert_id))
        return OrderResult(True, ticket=self._ticket,
                           price=request.entry_ref,
                           volume=request.volume)

    def modify_sl(self, ticket, sl, tp=None):
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        self._positions = [p for p in self._positions
                           if p.ticket != ticket]
        return OrderResult(True, ticket=ticket)


class FakeFeed:
    def fetch_new_all(self, cursors):
        empty = pd.DataFrame(columns=["open", "high", "low", "close",
                                      "volume", "real_volume",
                                      "spread"])
        return {tf: empty for tf in
                ("D1", "H4", "H1", "M15", "M30", "M5")}

    def account_state(self):
        return {"balance": 10000.0, "equity": 10000.0,
                "free_margin": 10000.0}

    def order_margin(self, *, volume, price, direction):
        return 2.0


class StubPlaybook:
    name = "stub"
    session_scope = "any"
    hunting_ground = SimpleNamespace(excluded=[], bonuses={})

    def __init__(self, entry, stop, target):
        self._entry = entry
        self._stop = stop
        self._target = target

    def check_live_signal(self, signal, ctx):
        return True, ""

    def live_signal(self, states, t, df=None):
        return [SimpleNamespace(
            index=t, score=100.0, direction=-1,
            entry_ref=self._entry, invalidation=self._stop,
            targets=[self._target], factors=[], atr=1.0,
            kind="fvg_limit")]


def _tick(when, mid):
    return {"time": pd.Timestamp(when, tz="UTC").to_pydatetime(),
            "mid": mid, "bid": mid, "ask": mid + 0.30,
            "tick_volume": 5, "real_volume": 0, "spread": 30}


def _stats():
    return {"ticks": 0, "bars": 0, "signals": 0, "executed": 0,
            "shadow": 0, "gated": 0, "hierarchy_blocked": 0,
            "tick_actions": 0, "reconciled_closed": 0,
            "triggers": 0, "invalidated": 0}


def _continuation_frames():
    h4 = _h4_live_backdrop()
    h1 = _h1_break_frame()
    _h, lows = find_swings(h1["high"].to_numpy(),
                           h1["low"].to_numpy(), 3)
    level = float(lows[-1][1])
    m15_idx = pd.date_range("2026-10-14 05:00", periods=12,
                            freq="15min", tz="UTC")
    m15 = _frame([level + 2.0 + 1.0 * math.sin(i) for i in range(12)],
                 m15_idx)
    comp_idx = pd.date_range("2026-10-14 08:00", periods=4,
                             freq="15min", tz="UTC")
    comp = _frame([level + 2.0, level + 4.0, level - 16.0,
                   level - 6.0], comp_idx)
    comp.loc[comp_idx[2], ["open", "high", "low", "close"]] = [
        level + 4.0, level + 6.0, level - 20.0, level - 16.0]
    m15 = pd.concat([m15, comp])
    m30_idx = pd.date_range("2026-10-14 00:00", periods=40,
                            freq="30min", tz="UTC")
    m30 = _frame([level + 2.0] * 40, m30_idx)
    m5_idx = pd.date_range("2026-10-14 06:00", periods=60,
                           freq="5min", tz="UTC")
    m5 = _frame([level + 2.0] * 60, m5_idx)
    frames = {"D1": _d1_short(), "H4": h4, "H1": h1, "M15": m15,
              "M30": m30, "M5": m5}
    return frames, level


def test_e_continuation_fires_once_per_break(tmp_path):
    frames, level = _continuation_frames()
    setup = live_setup(frames)
    assert setup is not None and setup["arm"] == "continuation"
    # Retest tick INSIDE the broken level's band, deep enough that
    # the trigger stop (band top + buf) clears the unchanged cost
    # gate (risk >= 2x round-trip cost) in the signal path.
    mid = level - 1.5
    assert setup["zone"]["bottom"] <= mid <= setup["zone"]["top"]
    stub = StubPlaybook(
        entry=mid, stop=setup["zone"]["top"] + setup["buf"],
        target=mid - 2 * (setup["zone"]["top"] + setup["buf"] - mid))
    cfg = MT5RunnerConfig(
        journal_path=tmp_path / "mt5_journal.jsonl",
        risk_state_path=tmp_path / "mt5_risk.json",
        kill_switch_path=tmp_path / "mt5_KILL",
        execute_playbooks=("stub",),
        # sizing: dynamic risk rule (fixed_lots revoked 2026-10-23)
        news_pause_enabled=False,
        golden_swing_confirm_enabled=False)
    runner = MT5Runner(cfg, feed=FakeFeed(), executor=FakeExecutor())
    runner._stack = TimeframeStack.from_frames(frames)
    from goldenswing.live.hierarchy import TF_MINUTES
    for tf, frame in runner._stack.frames.items():
        runner._last_processed[tf] = (
            frame.index[-1] + pd.Timedelta(minutes=TF_MINUTES[tf]))
    runner._refresh_live_setup()
    assert runner._live_setup is not None
    assert runner._live_setup["arm"] == "continuation"
    # /status renders the continuation arm (direction, broken
    # level, arm kind) — no pullback-only field assumptions.
    line = runner.setup_line()
    assert "CONTINUATION" in line
    assert "SHORT" in line
    assert f"{level:.1f}" in line
    runner._playbooks = [stub]
    stats = _stats()
    # Tick back at the broken level: retest touch + M15 confirm ->
    # fires through the standard pending -> signal path -> order.
    runner._evaluate_live_tick(_tick("2026-10-14 08:05", mid), stats)
    assert stats.get("triggers", 0) == 1
    assert len(runner.executor.orders) == 1
    assert runner.executor.orders[0]["direction"] == -1
    # The break is now spent.
    assert setup["break_key"] in runner._continuation_spent
    # Next H1 bar: price still beyond the broken level (same
    # break).  Refresh + another in-band tick must NOT re-arm.
    nxt = pd.DataFrame(
        [(level - 2.0, level + 3.0, level - 8.0, level - 5.0)],
        columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex(["2026-10-14 08:00"], tz="UTC")).assign(
            volume=100, real_volume=0, spread=30)
    runner._stack.append("H1", nxt)
    runner._refresh_live_setup()
    assert runner._live_setup is None
    assert runner._live_setup_reason.startswith("continuation_spent")
    runner._evaluate_live_tick(_tick("2026-10-14 09:05", mid), stats)
    assert stats.get("triggers", 0) == 1
    assert len(runner.executor.orders) == 1
