"""Causality across the WHOLE feature layer: truncating the series
must not change any family's output at earlier bars (peers included —
RS alignment is causal by bar-end comparison)."""
import numpy as np

from conftest import make_df
from goldenswing.features import FeatureEngine
from goldenswing.reader import MarketReader


def _series(n=300, seed=11):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1.0, n).cumsum()
    closes = 2000.0 + steps * 3.0
    rows = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c
        rows.append((float(o), float(max(o, c) + abs(rng.normal(0, 0.8))),
                     float(min(o, c) - abs(rng.normal(0, 0.8))), float(c)))
    df = make_df(rows, start="2024-03-01")
    df["volume"] = (800 + 900 * np.abs(rng.normal(0, 1, n))).tolist()
    peer_rows = []
    psteps = rng.normal(0, 1.0, n).cumsum()
    pcloses = 25.0 + psteps * 0.05
    for i, c in enumerate(pcloses):
        o = pcloses[i - 1] if i else c
        peer_rows.append((float(o), float(max(o, c) + 0.05),
                          float(min(o, c) - 0.05), float(c)))
    peer = make_df(peer_rows, start="2024-03-01")
    return df, {"silver": peer}


def _r(x, nd=6):
    return None if x is None or x != x else round(float(x), nd)


def _digest(fs):
    b, f, i, r = fs.bollinger, fs.fib, fs.ichimoku, fs.relstr
    return (
        (_r(b.mid), _r(b.upper), _r(b.pct_b), _r(b.bw_percentile),
         b.squeeze, b.walk, b.rejection),
        (tuple((fr, _r(p)) for fr, p in f.levels), f.leg_direction,
         f.nearest_frac, f.in_ote_long, f.in_ote_short, f.at_fib),
        (_r(i.tenkan), _r(i.kijun), _r(i.cloud_top), i.position,
         i.tk_cross, i.cloud_twist, i.chikou,
         i.d1_position, i.d1_cloud_bullish),
        (r.available, _r(r.silver.ratio), r.silver.trend,
         _r(r.dollar.ratio), r.dollar.trend, _r(r.score)),
        tuple((p.kind, p.status, p.formed_at, _r(p.neckline))
              for p in fs.patterns.active),
        (fs.patterns.event.kind if fs.patterns.event else None),
        ((_r(fs.zones.nearest_supply.top), fs.zones.nearest_supply.touches)
         if fs.zones.nearest_supply else None,
         (_r(fs.zones.nearest_demand.top), fs.zones.nearest_demand.touches)
         if fs.zones.nearest_demand else None,
         fs.zones.in_supply, fs.zones.in_demand),
        (fs.phase.phase, fs.phase.phase_age, fs.phase.event,
         _r(fs.phase.range_high)),
        (fs.volume.available, _r(fs.volume.rvol), fs.volume.climax,
         fs.volume.climax_reversal, fs.volume.dry_up, _r(fs.volume.vwap)),
        (fs.footprints.tags, fs.footprints.institutional_dir,
         fs.footprints.cta_trend, fs.footprints.retail_chase,
         fs.footprints.retail_trap),
        (fs.mtf.h4_trend, fs.mtf.h4_bias, fs.mtf.d1_trend, fs.mtf.d1_bias,
         _r(fs.mtf.h4_swing_high), _r(fs.mtf.d1_swing_low),
         _r(fs.mtf.alignment)),
        (tuple((ln.direction, ln.anchor1_index, _r(ln.slope),
                ln.touches, ln.state) for ln in fs.trendlines.active),
         tuple((e.kind, e.direction) for e in fs.trendlines.events)),
    )


def test_feature_truncation_does_not_change_past():
    df, peers = _series()
    cut = 190
    full = FeatureEngine().compute(df, MarketReader().read(df), peers)
    trunc = FeatureEngine().compute(
        df.iloc[:cut], MarketReader().read(df.iloc[:cut]), peers)
    assert len(full) == len(df) and len(trunc) == cut
    for t in [0, 5, 40, 90, 150, cut - 1]:
        assert _digest(full[t]) == _digest(trunc[t]), \
            f"feature state diverged at bar {t}"
