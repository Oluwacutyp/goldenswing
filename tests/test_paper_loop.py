"""Paper loop: live_signal parity with the backtested scanners.

The iron rule for the paper loop is that incremental ``live_signal``
reproduces the batch scanners' signals exactly — otherwise paper
trades would not be the validated setups.  These tests drive each
sleeve's ``live_signal`` bar-by-bar over historical data and compare
against the playbook's own ``candidates()`` (the backtest path).
"""
import pandas as pd
import pytest

from goldenswing.backtest.data import load_gold_h1
from goldenswing.features.config import FeatureConfig
from goldenswing.features.engine import read_and_featurize
from goldenswing.features.trendlines import _Builder
from goldenswing.feed import ReplayFeed
from goldenswing.playbooks.base import EngineContext
from goldenswing.playbooks.fvg_midline import FVGMidlinePlaybook
from goldenswing.playbooks.trendline_retest import TrendlineRetestPlaybook
from goldenswing.reader.reader import MarketReader


@pytest.fixture(scope="module")
def frame():
    df = load_gold_h1().iloc[-3000:].copy()
    return df


@pytest.fixture(scope="module")
def ctx_of(frame):
    states = read_and_featurize(frame, None)
    return EngineContext(df=frame, states=states, years=0.5,
                         instrument="XAUUSD"), states


def _sig_key(sig):
    return (sig.index, sig.direction, round(sig.entry_ref, 6),
            round(sig.invalidation, 6),
            tuple(round(x, 6) for x in (sig.targets or [])),
            round(sig.atr, 6))


# ------------------------------------------------------------- fvg parity
def test_fvg_live_signal_matches_backtest(ctx_of):
    ctx, states = ctx_of
    df = ctx.df
    batch = sorted((_sig_key(c.outcome.signal) for c in
                    FVGMidlinePlaybook().candidates(ctx)))
    assert len(batch) > 10, "test window needs real signal density"
    pb = FVGMidlinePlaybook()
    live = []
    for t in range(len(df)):
        for sig in pb.live_signal(states, t, df):
            live.append(_sig_key(sig))
    live = sorted(live)
    assert live == batch


def test_fvg_live_signal_needs_df(ctx_of):
    # without the bar frame the resting-limit machine cannot run —
    # it must stay silent, never guess
    ctx, states = ctx_of
    pb = FVGMidlinePlaybook()
    assert pb.live_signal(states, 100) == []
    assert pb.live_signal(states, -1, ctx.df) == []


# ------------------------------------------------------- trendline parity
def test_trendline_live_signal_matches_backtest(ctx_of):
    ctx, states = ctx_of
    df = ctx.df
    n = len(df)
    batch = sorted((_sig_key(c.outcome.signal) for c in
                    TrendlineRetestPlaybook().candidates(ctx)))
    assert len(batch) > 5, "test window needs real signal density"
    pb = TrendlineRetestPlaybook()
    live = []
    # interior bars only: the batch scanner's ``t + 1 < n`` guard is a
    # simulation artifact (it needs a forward bar); the live loop
    # deliberately trades the latest closed bar too
    for t in range(n - 1):
        for sig in pb.live_signal(states, t, df):
            live.append(_sig_key(sig))
    assert sorted(live) == batch


def test_trendline_live_signal_last_bar_does_not_crash(ctx_of):
    ctx, states = ctx_of
    df = ctx.df
    pb = TrendlineRetestPlaybook()
    sigs = pb.live_signal(states, len(df) - 1, df)
    assert isinstance(sigs, list)


def test_builder_extend_matches_batch(ctx_of):
    """_Builder.extend keeps the streaming builder identical to a
    batch-built one — the foundation the trendline live_signal stands
    on."""
    ctx, states = ctx_of
    df = ctx.df
    n = len(df)
    ref = _Builder(df, FeatureConfig())
    ref_states = [ref.update(t, s) for t, s in enumerate(states)]
    inc = _Builder(df.iloc[:1000], FeatureConfig())
    for t in range(1000):
        inc.update(t, states[t])
    inc.extend(df)
    for t in range(1000, n):
        ls = inc.update(t, states[t])
        ev_ref = [(e.kind, e.direction, round(e.line_price, 6))
                  for e in ref_states[t].events]
        ev_inc = [(e.kind, e.direction, round(e.line_price, 6))
                  for e in ls.events]
        assert ev_inc == ev_ref, f"event divergence at bar {t}"


# --------------------------------------------------------------- causality
def test_reader_is_causal_for_loop_recompute(frame):
    """The paper loop re-featurizes the whole frame per bar; this must
    give the same states[t] as featurizing the prefix — otherwise the
    parity proven above would not transfer to the loop."""
    full = MarketReader().read(frame)
    for t in (50, 500, 1500, 2500, len(frame) - 1):
        prefix = MarketReader().read(frame.iloc[:t + 1])
        a, b = full[t], prefix[t]
        assert a.close == b.close
        assert (a.atr or 0) == (b.atr or 0)
        assert a.trend == b.trend
        assert a.session == b.session
        assert [p.price for p in a.pools_above] == \
            [p.price for p in b.pools_above]
        ea, eb = a.last_structure_event, b.last_structure_event
        assert (ea.kind if ea else None) == (eb.kind if eb else None)
        assert (ea.index if ea else None) == (eb.index if eb else None)


# ------------------------------------------------------------- replay feed
def test_replay_feed_interface(frame):
    feed = ReplayFeed(frame, warm_bars=2000, chunk_bars=1)
    warm = feed.warm_start()
    assert len(warm) == 2000
    assert warm.index[-1] == frame.index[1999]
    bars = 0
    since = warm.index[-1]
    while not feed.exhausted:
        new = feed.fetch_new(since)
        assert len(new) <= 1
        if len(new):
            assert new.index[0] > since
            since = new.index[-1]
            bars += 1
    assert bars == len(frame) - 2000
    assert feed.poll_status()["ok"]
