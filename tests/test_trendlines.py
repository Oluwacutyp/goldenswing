"""Trendline family: formation from confirmed higher lows, lifecycle
(touch -> bounce -> break), dedupe of near-parallel lines, expiry,
and causality of the computed states."""
import numpy as np

from conftest import make_df
from goldenswing.features import FeatureConfig, FeatureEngine
from goldenswing.features.trendlines import compute as tl_compute
from goldenswing.reader import MarketReader
from goldenswing.reader.state import MarketState, SwingPoint

ATR = 10.0
A = SwingPoint(price=100.0, index=10, confirmed_at=12, kind="low")
B = SwingPoint(price=104.0, index=30, confirmed_at=32, kind="low")


def _states(closes, swings, atr=ATR):
    out = []
    for t, c in enumerate(closes):
        recent = [sp for sp in swings if sp.confirmed_at <= t][-8:]
        out.append(MarketState(
            index=t, timestamp=t, close=float(c), atr=atr,
            trend="range", bias=0,
            recent_swing_lows=[sp for sp in recent if sp.kind == "low"],
            recent_swing_highs=[sp for sp in recent if sp.kind == "high"],
        ))
    return out


def _df(closes, overrides=None):
    rows = []
    for t, c in enumerate(closes):
        o, h, l = c - 0.5, c + 1.0, c - 1.0
        if overrides and t in overrides:
            o, h, l = overrides[t]
        rows.append((float(o), float(h), float(l), float(c)))
    return make_df(rows)


def _main_closes():
    closes = [106.0] * 61
    closes[10] = 101.0
    closes[30] = 105.0
    for t in range(31, 61):
        closes[t] = 112.0
    closes[46] = 109.0
    for t in (47, 48, 49):
        closes[t] = 111.0
    closes[50] = 111.0
    closes[55] = 105.0
    for t in range(56, 61):
        closes[t] = 105.0
    return closes


_OVERRIDES = {
    46: (109.5, 110.0, 107.5),   # wick into the line, close back up
    50: (110.0, 112.0, 107.0),   # deeper test, decisive rejection
    55: (110.0, 110.5, 104.5),   # close through the line
}


def test_line_forms_on_second_anchor_confirmation():
    closes = _main_closes()
    states = _states(closes, [A, B])
    out = tl_compute(_df(closes, _OVERRIDES), states, FeatureConfig())
    assert out[31].active == ()            # B not yet confirmed
    live = [ln for ln in out[32].active if ln.direction == 1]
    assert len(live) == 1
    ln = live[0]
    assert ln.anchor1_index == 10 and ln.anchor2_index == 30
    assert ln.created_at == 32 and ln.state == "FORMING"
    assert abs(ln.slope - 0.2) < 1e-9


def test_touch_bounce_break_lifecycle():
    closes = _main_closes()
    states = _states(closes, [A, B])
    out = tl_compute(_df(closes, _OVERRIDES), states, FeatureConfig())

    # bar 46: first retest — a touch; the line becomes ACTIVE (validated)
    ev46 = [e for e in out[46].events if e.direction == 1]
    assert [e.kind for e in ev46] == ["touch"]
    ln46 = [ln for ln in out[46].active if ln.direction == 1][0]
    assert ln46.state == "ACTIVE" and ln46.touches == 1

    # bar 50: decisive rejection — a bounce, touch count 2
    ev50 = [e for e in out[50].events if e.direction == 1]
    assert [e.kind for e in ev50] == ["bounce"]
    ln50 = [ln for ln in out[50].active if ln.direction == 1][0]
    assert ln50.touches == 2

    # bar 55: close through — BROKEN event, line in snapshot as BROKEN,
    # gone the next bar
    ev55 = [e for e in out[55].events if e.direction == 1]
    assert [e.kind for e in ev55] == ["break"]
    ln55 = [ln for ln in out[55].active if ln.direction == 1]
    assert ln55 and ln55[0].state == "BROKEN"
    assert [ln for ln in out[56].active if ln.direction == 1] == []


def test_near_parallel_duplicate_is_suppressed():
    # C sits exactly on the A-B line: pairing A->C must not spawn a
    # second line with the same geometry.
    C = SwingPoint(price=104.8, index=34, confirmed_at=36, kind="low")
    closes = _main_closes()
    states = _states(closes, [A, B, C])
    out = tl_compute(_df(closes, _OVERRIDES), states, FeatureConfig())
    live = [ln for ln in out[40].active if ln.direction == 1]
    assert len(live) == 1


def test_line_expires_when_price_drifts_away():
    closes = _main_closes()
    for t in range(33, 61):
        closes[t] = 200.0                  # > 6 ATR from the line
    states = _states(closes, [A, B])
    out = tl_compute(_df(closes), states, FeatureConfig())
    assert [ln for ln in out[40].active if ln.direction == 1] == []


def test_trendline_states_are_causal():
    # through the full engine: truncating must not change earlier states
    rng = np.random.default_rng(7)
    steps = rng.normal(0, 1.0, 300).cumsum()
    closes = 2000.0 + steps * 3.0
    rows = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c
        rows.append((float(o), float(max(o, c) + abs(rng.normal(0, 0.8))),
                     float(min(o, c) - abs(rng.normal(0, 0.8))), float(c)))
    df = make_df(rows, start="2024-03-01")
    df["volume"] = (800 + 900 * np.abs(rng.normal(0, 1, 300))).tolist()
    cut = 190

    def digest(fs):
        tl = fs.trendlines
        return (
            tuple((ln.direction, ln.anchor1_index, round(ln.slope, 6),
                   ln.touches, ln.state) for ln in tl.active),
            tuple((e.kind, e.direction) for e in tl.events),
        )

    full = FeatureEngine().compute(df, MarketReader().read(df))
    trunc = FeatureEngine().compute(
        df.iloc[:cut], MarketReader().read(df.iloc[:cut]))
    for t in [0, 40, 90, 150, cut - 1]:
        assert digest(full[t]) == digest(trunc[t]), \
            f"trendline state diverged at bar {t}"
