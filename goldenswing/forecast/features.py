"""Per-bar feature vectors for the forecaster.

Every feature is a pure function of the ``MarketState`` (and its
attached ``FeatureSet``) at bar ``t`` — the reader and the feature
engine are causal by construction (their own truncation tests prove
prefix stability), so a feature row cannot change when the series is
truncated after ``t``.  No full-sample statistics are computed here:
no global means, no global normalisation.  (Fold-local imputation /
scaling happens inside the model pipelines in walkforward.py.)

The detector's own per-bar opinion (Combo A config — the +25.62%
book from VALIDATION.md) is included as three features: score,
direction, signed score.  ``evaluate()`` is used directly (not
``scan()``), so the opinion is the detector's raw view at the bar,
unfiltered by the scan gap rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..golden.detector import DetectorConfig, GoldenSwingDetector
from ..reader.state import MarketState

#: The validated detector configuration (VALIDATION.md): Phase-2
#: flags + combo A families {relstr, volume, phases}.
COMBO_A = frozenset({"relstr", "volume", "phases"})
P2_FLAGS = dict(require_sweep_confirmation=True, daily_veto=True)

_PHASES = ("accumulation", "distribution", "consolidation",
           "manipulation", "expansion_up", "expansion_down",
           "trend_up", "trend_down", "unknown")
_SESSIONS = ("asia", "london", "overlap", "newyork", "off")


def combo_a_config() -> DetectorConfig:
    return DetectorConfig(emit_floor=35.0, golden_threshold=60.0,
                          feature_families=COMBO_A, **P2_FLAGS)


def detector_opinions(states: list[MarketState],
                      detector: GoldenSwingDetector | None = None
                      ) -> pd.DataFrame:
    """The detector's raw per-bar opinion: score (0 when it emits
    nothing at the bar), direction (+1/-1/0), sweep-reversal flag."""
    det = detector or GoldenSwingDetector(combo_a_config())
    n = len(states)
    score = np.zeros(n)
    direction = np.zeros(n, dtype=int)
    is_sweep = np.zeros(n)
    for t, s in enumerate(states):
        sig = det.evaluate(s)
        if sig is not None:
            score[t] = sig.score
            direction[t] = sig.direction
            is_sweep[t] = 1.0 if sig.kind == "sweep_reversal" else 0.0
    return pd.DataFrame({"det_score": score, "det_dir": direction,
                         "det_sweep": is_sweep})


def _age(age: int | None, cap: float) -> float:
    """Normalised age in [0, 1]; unknown age saturates at 1."""
    if age is None:
        return 1.0
    return min(float(age), cap) / cap


def _zone_dist(state: MarketState, bottom: float, top: float) -> float:
    if state.close > top:
        return state.close - top
    if state.close < bottom:
        return bottom - state.close
    return 0.0


def _nearest_dist_atr(state: MarketState, direction: int, which: str,
                      cap: float = 5.0) -> float:
    """Distance in ATRs from close to the nearest FVG/OB of the given
    direction (0 when inside it); ``cap`` when none is active."""
    items = state.active_fvgs if which == "fvg" else state.active_obs
    best = np.inf
    for it in items:
        if it.direction != direction:
            continue
        best = min(best, _zone_dist(state, it.bottom, it.top))
    if not np.isfinite(best) or state.atr <= 0:
        return cap
    return min(best / state.atr, cap)


def build_feature_frame(states: list[MarketState],
                        opinions: pd.DataFrame | None = None
                        ) -> pd.DataFrame:
    """One feature row per state.  Row ``t`` aligns with bar ``t``
    (``states[t].index == t`` — the reader's contract)."""
    if opinions is None:
        opinions = detector_opinions(states)
    rows: list[dict] = []
    for t, s in enumerate(states):
        fs = s.features
        atr = s.atr if s.atr > 0 else np.nan
        r: dict[str, float] = {}

        # ---- reader: structure / trend
        r["bias"] = float(s.bias)
        r["trend_ord"] = {"up": 1.0, "down": -1.0}.get(s.trend, 0.0)
        r["struct_age"] = _age(s.structure_event_age, 150.0)
        ev = s.last_structure_event
        r["last_event_dir"] = float(ev.direction) if ev else 0.0
        r["last_event_choch"] = 1.0 if ev and ev.kind == "CHOCH" else 0.0
        r["trend_age"] = _age(
            None if s.trend_since is None else t - s.trend_since, 300.0)

        # ---- reader: liquidity / sweeps
        r["sweep_age"] = _age(s.sweep_age, 50.0)
        r["sweep_dir"] = float(s.last_sweep.direction) if s.last_sweep else 0.0
        r["pool_above_atr"] = (min((s.pools_above[0].price - s.close) / atr, 6.0)
                               if s.pools_above and atr == atr else 6.0)
        r["pool_below_atr"] = (min((s.close - s.pools_below[0].price) / atr, 6.0)
                               if s.pools_below and atr == atr else 6.0)

        # ---- reader: displacement
        r["bull_disp_3"] = min(s.bull_disp_3, 3.0)
        r["bear_disp_3"] = min(s.bear_disp_3, 3.0)
        r["bull_disp_age"] = _age(s.bull_disp_age, 50.0)
        r["bear_disp_age"] = _age(s.bear_disp_age, 50.0)

        # ---- reader: ICT level proximity
        r["fvg_up_dist"] = _nearest_dist_atr(s, 1, "fvg")
        r["fvg_dn_dist"] = _nearest_dist_atr(s, -1, "fvg")
        r["ob_up_dist"] = _nearest_dist_atr(s, 1, "ob")
        r["ob_dn_dist"] = _nearest_dist_atr(s, -1, "ob")

        # ---- reader: premium/discount + regime
        r["eq_position"] = float(s.eq_position)
        r["zone_premium"] = 1.0 if s.zone == "premium" else 0.0
        r["zone_discount"] = 1.0 if s.zone == "discount" else 0.0
        rng = (s.range_high - s.range_low) / atr \
            if np.isfinite(s.range_high) and atr == atr else np.nan
        r["range_atr"] = min(rng, 30.0) if rng == rng else np.nan
        r["atr_pct"] = s.atr_percentile / 100.0
        r["vol_regime_ord"] = {"compressed": 0.0, "normal": 1.0,
                               "expanded": 2.0}.get(s.vol_regime, 1.0)
        r["momentum"] = float(np.clip(s.momentum, -3.0, 3.0))
        r["body_ratio"] = float(np.clip(s.body_ratio, -2.0, 2.0))

        # ---- reader: higher timeframe + sessions
        r["daily_bias"] = float(s.daily_bias)
        r["daily_trend_ord"] = {"up": 1.0, "down": -1.0}.get(s.daily_trend, 0.0)
        for sess in _SESSIONS:
            r[f"sess_{sess}"] = 1.0 if s.session == sess else 0.0
        r["killzone"] = 1.0 if s.killzone else 0.0
        if np.isfinite(s.session_high) and np.isfinite(s.session_low) \
                and s.session_high > s.session_low:
            r["session_pos"] = float(np.clip(
                (s.close - s.session_low) / (s.session_high - s.session_low),
                -0.5, 1.5))
        else:
            r["session_pos"] = 0.5

        # ---- detector opinion (Combo A)
        r["det_score"] = float(opinions["det_score"].iloc[t])
        r["det_dir"] = float(opinions["det_dir"].iloc[t])
        r["det_signed"] = r["det_score"] * r["det_dir"]
        r["det_sweep"] = float(opinions["det_sweep"].iloc[t])

        if fs is not None:
            # ---- phases (AMD) — one-hot + age + event direction
            ph = fs.phase.phase
            for p in _PHASES:
                r[f"phase_{p}"] = 1.0 if ph == p else 0.0
            if ph not in _PHASES:
                r["phase_unknown"] = 1.0
            r["phase_age"] = _age(fs.phase.phase_age, 200.0)
            r["phase_event_dir"] = float(fs.phase.event_direction)

            # ---- combo A: relative strength
            rs = fs.relstr
            r["rs_available"] = 1.0 if rs.available else 0.0
            r["rs_score"] = float(rs.score) if rs.available else np.nan
            r["rs_silver_pct"] = rs.silver.percentile / 100.0
            r["rs_silver_trend"] = float(rs.silver.trend)
            r["rs_dollar_pct"] = rs.dollar.percentile / 100.0
            r["rs_dollar_trend"] = float(rs.dollar.trend)

            # ---- combo A: volume
            v = fs.volume
            r["vol_available"] = 1.0 if v.available else 0.0
            r["rvol"] = float(np.clip(v.rvol, 0.0, 5.0)) if v.available else np.nan
            r["climax"] = 1.0 if v.climax else 0.0
            r["climax_reversal"] = float(v.climax_reversal)
            r["dry_up"] = 1.0 if v.dry_up else 0.0
            r["vwap_dist_atr"] = (float(np.clip((s.close - v.vwap) / atr, -5.0, 5.0))
                                  if v.available and np.isfinite(v.vwap)
                                  and atr == atr else np.nan)

            # ---- bollinger state
            b = fs.bollinger
            r["bb_pct_b"] = float(np.clip(b.pct_b, -0.5, 1.5)) \
                if np.isfinite(b.pct_b) else np.nan
            r["bb_bw_pct"] = b.bw_percentile / 100.0
            r["bb_squeeze"] = 1.0 if b.squeeze else 0.0
            r["bb_walk"] = float(b.walk)
            r["bb_rejection"] = float(b.rejection)

            # ---- fib / OTE
            f = fs.fib
            r["in_ote_long"] = 1.0 if f.in_ote_long else 0.0
            r["in_ote_short"] = 1.0 if f.in_ote_short else 0.0
            r["at_fib"] = 1.0 if f.at_fib else 0.0
            r["fib_dist_atr"] = (min(f.nearest_dist_atr, 3.0)
                                 if np.isfinite(f.nearest_dist_atr) else 3.0)

            # ---- ichimoku
            i = fs.ichimoku
            r["ichi_pos"] = {"above": 1.0, "below": -1.0}.get(i.position, 0.0)
            r["ichi_tk"] = float(i.tk_cross)
            r["ichi_cloud_bull"] = 1.0 if i.cloud_bullish else 0.0

            # ---- MTF alignment
            m = fs.mtf
            r["mtf_alignment"] = float(m.alignment)
            r["mtf_h4_bias"] = float(m.h4_bias)
            r["mtf_d1_bias"] = float(m.d1_bias)

        rows.append(r)

    X = pd.DataFrame(rows)
    X.index = pd.RangeIndex(len(states))
    return X
