"""Supply/demand zones (drop-base-rally) and the AMD phase machine."""
import numpy as np

from conftest import closes_df, make_df
from goldenswing.features import phases, supply_demand
from goldenswing.features.config import FeatureConfig
from goldenswing.reader import MarketReader

CFG = FeatureConfig()


def test_demand_zone_drop_base_rally():
    rows = []
    # drop: 2050 -> 2000
    for k in range(10):
        c = 2050.0 - 5.0 * (k + 1)
        rows.append((c + 5.0, c + 5.5, c - 0.5, c))
    # base: 3 small candles around 2000
    for o, c in ((2000.0, 2000.4), (2000.4, 2000.1), (2000.1, 2000.5)):
        rows.append((o, max(o, c) + 0.3, min(o, c) - 0.3, c))
    # rally departure: big body up
    rows.append((2000.5, 2009.0, 2000.2, 2008.5))
    rows.append((2008.5, 2012.0, 2007.0, 2011.0))
    df = make_df(rows)
    states = MarketReader().read(df)
    out = supply_demand.compute(df, states, CFG)
    dep = out[-2]      # the departure bar
    assert dep.nearest_demand is not None
    assert dep.nearest_demand.direction == 1
    assert dep.nearest_demand.freshness == "fresh"
    assert dep.nearest_demand.created_at == len(rows) - 2
    # zone is the base range; the departure close has already left it
    assert dep.nearest_demand.top <= 2000.8 + 1e-9
    assert dep.nearest_demand.bottom >= 1999.7 - 1e-9
    assert dep.demand_dist_atr > 0.0


def test_zone_becomes_tested_on_return():
    rows = []
    for k in range(10):
        c = 2050.0 - 5.0 * (k + 1)
        rows.append((c + 5.0, c + 5.5, c - 0.5, c))
    for o, c in ((2000.0, 2000.4), (2000.4, 2000.1), (2000.1, 2000.5)):
        rows.append((o, max(o, c) + 0.3, min(o, c) - 0.3, c))
    rows.append((2000.5, 2009.0, 2000.2, 2008.5))   # departure
    rows.append((2008.5, 2010.0, 2006.0, 2009.0))   # away
    rows.append((2009.0, 2009.5, 2000.6, 2005.0))   # back into the zone
    df = make_df(rows)
    states = MarketReader().read(df)
    out = supply_demand.compute(df, states, CFG)
    assert out[-1].nearest_demand is not None
    assert out[-1].nearest_demand.touches >= 1
    assert out[-1].nearest_demand.freshness == "tested"


def test_accumulation_after_decline_then_flat():
    # Shorter base-detection window so the base can form while the
    # prior-move lookback still reaches the decline (with the default
    # 48/48 pairing the decline has always left both windows by the
    # time the range narrows — see FEATURES.md).
    cfg = FeatureConfig(amd_range_bars=20, amd_prior_bars=40)
    closes = [2100.0 - 2.0 * k for k in range(40)]          # decline
    closes += [2020.0 + (0.6 if k % 2 else -0.6)
               for k in range(120)]                          # tight base
    df = closes_df(closes, spread=0.3)
    states = MarketReader().read(df)
    out = phases.compute(df, states, cfg)
    labels = {s.phase for s in out[60:]}
    assert "accumulation" in labels, labels


# ------------------------------------------------------- AMD fallback work
# Owner's note: "How to detect accumulation, manipulation, distribution
# and if it missed it bos liquidity and efficiency." The fallback is
# BOS + recent same-direction sweep + fresh same-direction FVG, all
# three legs required, while in a base. A standing pool alone is NOT
# a leg (measured: 72 noise bars vs 18 sweep-leg bars per 3000 H1).

def _amd_rows():
    """Decline -> tight base (accumulation). Sweep appended by callers."""
    rows = []
    c = 2100.0
    for _ in range(40):
        c -= 2.0
        rows.append((c + 2.0, c + 2.5, c - 0.5, c))
    for k in range(25):
        b = 2020.0 + (0.5 if k % 2 else -0.5)
        rows.append((b, b + 0.6, b - 0.6, b))
    return rows


def _amd_cfg():
    return FeatureConfig(amd_range_bars=20, amd_prior_bars=40)


def test_amd_clean_sequence_manipulation_then_expansion():
    rows = _amd_rows()
    rows.append((2020.0, 2021.0, 2012.0, 2020.5))   # sweep lows, close in
    rows.append((2020.5, 2021.0, 2020.0, 2020.8))   # inside
    rows.append((2020.8, 2035.0, 2020.0, 2034.0))   # displacement exit
    rows.append((2034.0, 2042.0, 2033.0, 2041.0))
    df = make_df(rows)
    out = phases.compute(df, MarketReader().read(df), _amd_cfg())
    seq = [p.phase for p in out[60:]]
    assert seq == ["accumulation"] * 5 + ["manipulation"] * 2 \
        + ["expansion_up"] * 2, seq
    spring = out[65]
    assert spring.event == "spring" and spring.event_direction == 1
    assert spring.via == "direct" and spring.confidence == 0.80
    assert out[67].event == "phase_exit_up"


def test_amd_manipulation_without_distribution_returns_to_base():
    rows = _amd_rows()
    rows.append((2020.0, 2021.0, 2012.0, 2020.5))   # sweep -> manipulation
    for k in range(15):                              # drift back inside
        b = 2020.0 + (0.4 if k % 2 else -0.4)
        rows.append((b, b + 0.5, b - 0.5, b))
    df = make_df(rows)
    out = phases.compute(df, MarketReader().read(df), _amd_cfg())
    labels = {p.phase for p in out[60:]}
    assert "manipulation" in labels
    assert not labels & {"expansion_up", "expansion_down",
                         "trend_up", "trend_down"}, labels
    assert out[-1].phase == "accumulation"


def test_amd_ranging_no_sweep_stays_accumulation():
    closes = [2100.0 - 2.0 * k for k in range(40)]
    closes += [2020.0 + (0.6 if k % 2 else -0.6) for k in range(120)]
    df = closes_df(closes, spread=0.3)
    out = phases.compute(df, MarketReader().read(df), _amd_cfg())
    labels = {p.phase for p in out[60:]}
    assert labels == {"accumulation"}, labels


def test_amd_missed_sweep_fallback_real_data():
    """Real XAUUSD H1 (2026-04-13): BOS + sweep + FVG coincide in a
    base, but the direct check misses (sweep_age 8 > 0). The fallback
    must label manipulation with fallback provenance, and must never
    reach expansion without displacement."""
    from goldenswing.backtest.data import load_gold_h1

    sl = load_gold_h1().iloc[-3000:-3000 + 130].copy()
    states = MarketReader().read(sl)
    out = phases.compute(sl, states, FeatureConfig())
    fb = out[114]
    assert fb.phase == "manipulation"
    assert fb.event == "manipulation_fallback"
    assert fb.event_direction == 1
    assert fb.via == "fallback" and fb.confidence == 0.55
    assert len(fb.evidence) == 3  # BOS + sweep + FVG
    assert "expansion_up" not in {p.phase for p in out[114:]}
    assert "expansion_down" not in {p.phase for p in out[114:]}
    # All three legs required: killing any one leg kills the fallback.
    for kw in (dict(amd_fallback_bos_age=-1),
               dict(amd_fallback_sweep_age=-1),
               dict(amd_fallback_fvg_age=-1)):
        o2 = phases.compute(sl, states, FeatureConfig(**kw))
        assert not [p for p in o2
                    if p.event == "manipulation_fallback"], kw


def test_amd_phase_state_provenance_and_determinism():
    import math

    from goldenswing.features.state import PhaseState

    d = PhaseState()
    assert d.via == "" and d.confidence == 0.0 and d.evidence == ()
    # compute is a pure function of its inputs: same input, same
    # output (no cursor to rewind — the FVG-bug class). NaN-aware
    # compare: dataclass __eq__ treats NaN != NaN.
    from goldenswing.backtest.data import load_gold_h1

    sl = load_gold_h1().iloc[-3000:-3000 + 130].copy()
    states = MarketReader().read(sl)
    o1 = phases.compute(sl, states, FeatureConfig())
    o2 = phases.compute(sl, states, FeatureConfig())
    assert len(o1) == len(o2)
    for a, b in zip(o1, o2):
        assert (a.phase, a.phase_age, a.event, a.event_direction,
                a.via, a.confidence, a.evidence) == \
            (b.phase, b.phase_age, b.event, b.event_direction,
             b.via, b.confidence, b.evidence)
        for x, y in ((a.range_high, b.range_high),
                     (a.range_low, b.range_low)):
            assert (math.isnan(x) and math.isnan(y)) or x == y
