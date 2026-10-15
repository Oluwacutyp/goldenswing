"""GoldenSwing detector.

A Golden Swing is a *big* move that is *backed*: expected expansion of at
least ~2x ATR over the coming ~72 H1 bars (3 days), where the backing is
confluence the reader can verify — structure alignment, a liquidity event
(sweep into a level), displacement out of an ICT level, a supportive
volatility regime, and session timing.  Each bar is scored 0-100 across
those five families plus room-to-run; a move without backing scores low
and is rejected by the gates no matter how large it later turns out to be.
That discrimination is the point of the system.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..reader.state import MarketState


@dataclass
class DetectorConfig:
    emit_floor: float = 35.0        # scan() reports signals at/above this
    golden_threshold: float = 60.0  # score that qualifies as a Golden Swing
    sweep_fresh_bars: int = 8       # sweep stays tradeable this long
    min_gap: int = 3                # bars between emitted signals
    min_room_atr: float = 1.2       # gate: room to nearest opposing pool
    min_stop_atr: float = 0.25      # gate: invalidation distance bounds
    max_stop_atr: float = 3.5
    level_near_atr: float = 0.35    # "price at a level" distance
    # Phase 2 (defaults preserve Phase-1 behaviour; harness2 enables them)
    require_sweep_confirmation: bool = False
    confirm_window: int = 8         # bars after a sweep for confirmation
    daily_veto: bool = False        # block counter-daily-bias trades
    daily_level_tol_atr: float = 1.5  # "at a major daily level" tolerance
    # EMA-stack alignment gate (user's pullback-bot filter): when True,
    # a signal is only emitted if its direction matches the 9/21/50 EMA
    # stack (longs need 9>21>50, shorts need 9<21<50). Mixed stacks
    # emit nothing — stand aside when the trend is unclear.
    require_stack_alignment: bool = False
    # Turtle Soup short: when True, buy-side sweeps with a hard bearish
    # CHoCH are classified as "turtle_soup" (a selective short setup)
    # instead of the generic "sweep_reversal". Default False preserves
    # Phase-2 behavior.
    turtle_soup_short: bool = False
    # Phase 3: feature families to score with (subset of
    # goldenswing.features.FAMILIES).  None = features OFF — evaluate()
    # behaves exactly as Phase 2.  A family only contributes when the
    # state actually carries a FeatureSet (FeatureEngine.attach).
    # Phase 4 adds the context-gated "*_ctx" re-designs and the
    # "trendlines" event family (features/gating.py).
    feature_families: frozenset | None = None
    # Phase 5: phase→playbook routing (goldenswing.modes.routing).
    # Duck-typed: any object with ``cap_for(phase, playbook) ->
    # float | None``.  None (default) = no routing — behaviour is
    # exactly Phases 2-4.  The table is DATA measured from a prior
    # shadow book, never a hardcoded belief.
    route_table: object = None


@dataclass
class GoldenSignal:
    index: int
    timestamp: object
    direction: int                  # +1 long | -1 short
    score: float
    kind: str                       # "sweep_reversal" | "continuation"
    factors: list[str] = field(default_factory=list)
    entry_zone: tuple[float, float] = (0.0, 0.0)
    entry_ref: float = 0.0          # close at signal bar
    invalidation: float = 0.0
    targets: list[float] = field(default_factory=list)  # liquidity pools ahead
    atr: float = 0.0                # ATR at the signal bar
    sweep_index: int | None = None  # bar of the sweep (sweep_reversal only)

    @property
    def is_golden(self) -> bool:
        return self.score >= 60.0


#: Playbook names (Phase 5 routing).  The detector emits two setup
#: ``kind``s; the playbook taxonomy splits "continuation" by *how* the
#: entry rides the trend:
PLAYBOOK_SWEEP_REVERSAL = "sweep_reversal"
PLAYBOOK_TREND_CONTINUATION = "trend_continuation"
PLAYBOOK_BREAKOUT = "breakout"
PLAYBOOKS = (PLAYBOOK_SWEEP_REVERSAL, PLAYBOOK_TREND_CONTINUATION,
             PLAYBOOK_BREAKOUT)

#: A BOS counts as "fresh" for the breakout playbook within this many
#: bars — the same freshness window the structure score uses for its
#: "fresh BOS/CHoCH" factor.
FRESH_BOS_BARS = 5


def playbook_of(state: MarketState, kind: str) -> str:
    """Classify a setup into its playbook from the state that made it.

    - ``sweep_reversal`` setups are their own playbook.
    - Phase-9 sleeve signals carry their own kind and map straight to
      the sleeve (no legacy reclassification — the sleeve IS the
      playbook).
    - A ``continuation`` whose structure event is a FRESH BOS in the
    trade direction (age <= FRESH_BOS_BARS) is a **breakout**: the
    entry rides the break of structure itself.
    - Any other ``continuation`` — a pullback to an OB/FVG inside an
    established trend, or one confirmed by a CHoCH (a character
    change is a turn, not a breakout) — is **trend_continuation**.

    Pure function of (state, kind): the routing-matrix builder and the
    detector's routing cap share this exact definition.
    """
    if kind == "sweep_reversal":
        return PLAYBOOK_SWEEP_REVERSAL
    if kind == "turtle_soup":
        return "turtle_soup"
    if kind == "sr_flip":
        return "sr_flip"
    if kind == "sleeve_choch_fvg":
        return "fvg_midline"
    if kind == "trendline_retest":
        return "trendline_retest"
    if kind == "qm_m5":
        return "qm_m5"
    if kind == "qm_d1":
        return "qm_d1"
    ev = state.last_structure_event
    if ev is not None and ev.kind == "BOS" and ev.direction == state.bias \
            and state.structure_event_age is not None \
            and state.structure_event_age <= FRESH_BOS_BARS:
        return PLAYBOOK_BREAKOUT
    return PLAYBOOK_TREND_CONTINUATION


class GoldenSwingDetector:
    def __init__(self, config: DetectorConfig | None = None):
        self.cfg = config or DetectorConfig()

    # ------------------------------------------------------------------ API
    def evaluate(self, s: MarketState) -> GoldenSignal | None:
        cfg = self.cfg
        atr = s.atr
        if atr <= 0:
            return None

        # Hierarchical mode is implemented as post-filtering (see
        # MTF_CONFLUENCE_FRAMEWORK.md). The detector generates signals
        # normally; the hierarchy is enforced by filtering results.
        # This avoids interactions with scan's min_gap logic.
        setup = self._setup(s)
        if setup is None:
            return None
        direction, kind, anchor_inv, anchor_zone, sweep = setup

        # ---- EMA-stack alignment gate ------------------------------------
        # Signals must trade WITH the 9/21/50 stack, never against it.
        # Mixed stacks (0) emit nothing — the trend is unclear, stand aside.
        if cfg.require_stack_alignment and s.ema_stack != direction:
            return None

        # ---- daily-bias regime veto (Phase 2) ------------------------------
        if cfg.daily_veto and s.daily_bias != 0 and direction == -s.daily_bias:
            if not self._daily_exception(s, direction, kind, sweep):
                return None

        # ---- room to run (gate + score) ----------------------------------
        pools_ahead = s.pools_above if direction == 1 else s.pools_below
        if pools_ahead:
            room_atr = abs(pools_ahead[0].price - s.close) / atr
        else:
            room_atr = 2.5  # no mapped pool: assume moderate room
        if room_atr < cfg.min_room_atr:
            return None

        # ---- invalidation / stop distance (gate) --------------------------
        inv = anchor_inv
        if inv is None:
            inv = self._fallback_invalidation(s, direction)
        stop_atr = abs(s.close - inv) / atr
        if not (cfg.min_stop_atr <= stop_atr <= cfg.max_stop_atr):
            return None

        factors: list[str] = []
        score = 0.0
        score += self._structure_score(s, direction, kind, factors)
        score += self._liquidity_score(s, direction, factors)
        score += self._displacement_score(s, direction, factors)
        score += self._room_score(room_atr, factors)
        score += self._regime_score(s, direction, factors)
        score += self._session_score(s, factors)
        score = float(min(100.0, score))

        # ---- Phase-3 feature families (opt-in; skipped entirely when
        # feature_families is None, preserving Phase 2 exactly) --------
        if cfg.feature_families is not None and s.features is not None:
            score = float(min(100.0, max(0.0, score + self._feature_score(
                s, direction, kind, factors))))
            # Phase 4: the gated MTF veto — both HTFs opposed caps the
            # score below the golden threshold (see features/gating.py)
            cap = self._feature_cap(s, direction)
            if cap is not None and score > cap:
                score = cap
                factors.append(
                    "MTF veto: D1 + H4 both opposed — capped below golden")

        # ---- Phase 5: phase→playbook routing (opt-in; the route
        # table is measured from a shadow book — see modes/routing) --
        if cfg.route_table is not None and s.features is not None:
            cap = cfg.route_table.cap_for(s.features.phase.phase,
                                          playbook_of(s, kind))
            if cap is not None and score > cap:
                score = cap
                factors.append(
                    "phase routing: this playbook is muted in the "
                    "current phase (measured) — capped below golden")

        if score < cfg.emit_floor:
            return None

        zone = anchor_zone if anchor_zone is not None else (
            (s.close - 0.2 * atr, s.close) if direction == 1
            else (s.close, s.close + 0.2 * atr)
        )
        targets = [p.price for p in pools_ahead[:3]]
        return GoldenSignal(
            index=s.index,
            timestamp=s.timestamp,
            direction=direction,
            score=score,
            kind=kind,
            factors=factors,
            entry_zone=(float(zone[0]), float(zone[1])),
            entry_ref=float(s.close),
            invalidation=float(inv),
            targets=[float(x) for x in targets],
            atr=float(s.atr),
            sweep_index=(sweep.index if sweep is not None else None),
        )

    def scan(self, states: list[MarketState]) -> list[GoldenSignal]:
        out: list[GoldenSignal] = []
        last_idx = -10**9
        for s in states:
            sig = self.evaluate(s)
            if sig is None:
                continue
            if sig.index - last_idx < self.cfg.min_gap:
                continue
            out.append(sig)
            last_idx = sig.index
        return out

    # -------------------------------------------------------------- setups
    def _setup(self, s: MarketState):
        """Return (direction, kind, anchor_invalidation, anchor_zone, sweep)
        or None.

        Priority: a liquidity sweep is the strongest backing (stops already
        grabbed = fuel) — but with ``require_sweep_confirmation`` (Phase 2)
        a sweep only becomes a setup once it is CONFIRMED by displacement
        and a structure shift in the reversal direction; an unconfirmed
        sweep falls through to the continuation path.  Otherwise a
        trend-aligned displacement continuation.  Anchors give the
        structural invalidation.
        """
        cfg = self.cfg
        # Turtle Soup short (opt-in): buy-side liquidity swept, then a hard
        # bearish market-structure shift. This is NOT a mirrored sweep
        # reversal — it only fires on buy-side sweeps with confirmed
        # bearish CHoCH. See mining: short-strategies-bearish-setups.
        if cfg.turtle_soup_short:
            ts = self._turtle_soup_short(s)
            if ts is not None:
                return ts
        # sweep reversal
        if s.last_sweep is not None and s.sweep_age is not None:
            window = cfg.confirm_window if cfg.require_sweep_confirmation \
                else cfg.sweep_fresh_bars
            if s.sweep_age <= window:
                d = s.last_sweep.direction
                if not cfg.require_sweep_confirmation \
                        or self._sweep_confirmed(s, d):
                    inv = s.last_sweep.extreme - 0.15 * s.atr if d == 1 \
                        else s.last_sweep.extreme + 0.15 * s.atr
                    return d, "sweep_reversal", inv, None, s.last_sweep
        # trend continuation: displacement in the trend direction
        if s.trend in ("up", "down"):
            d = s.bias
            disp = s.bull_disp_3 if d == 1 else s.bear_disp_3
            if disp >= 0.8:
                ob = s.nearest_ob(d)
                if ob is not None and self._dist_to_zone(s, ob.bottom, ob.top) <= 1.0 * s.atr:
                    inv = ob.bottom - 0.15 * s.atr if d == 1 else ob.top + 0.15 * s.atr
                    return d, "continuation", inv, (ob.bottom, ob.top), None
                g = s.nearest_fvg(d)
                if g is not None and self._dist_to_zone(s, g.bottom, g.top) <= 1.0 * s.atr:
                    inv = g.bottom - 0.15 * s.atr if d == 1 else g.top + 0.15 * s.atr
                    return d, "continuation", inv, (g.bottom, g.top), None
                return d, "continuation", None, None, None
        # S/R flip: price retesting a broken level from the other side.
        # Broken support (direction=-1) -> short when price rallies back
        # into it; broken resistance (direction=+1) -> long on pullback.
        # The flip level IS the invalidation: reclaiming it kills the thesis.
        flip = self._active_flip_retest(s)
        if flip is not None:
            d = flip.direction
            tol = 0.3 * s.atr
            inv = flip.price + 0.3 * s.atr if d == -1 \
                else flip.price - 0.3 * s.atr
            zone = (flip.price - tol, flip.price + tol)
            return d, "sr_flip", inv, zone, None
        return None

    def _active_flip_retest(self, s):
        """Nearest active flip level that price is currently retesting.

        Bearish flip (broken support): price must be at/below the level,
        having rallied back up into it — the level is now resistance.
        Bullish flip (broken resistance): price at/above, pulled back
        into it — now support. Returns the flip or None.
        """
        if not s.active_flips:
            return None
        tol = 0.35 * s.atr
        best, best_dist = None, float("inf")
        for fl in s.active_flips:
            dist = abs(s.close - fl.price)
            if dist > tol:
                continue
            # retest must come from the right side: below a bearish flip
            # level (rally into resistance), above a bullish one
            if fl.direction == -1 and s.close > fl.price + tol:
                continue
            if fl.direction == 1 and s.close < fl.price - tol:
                continue
            if dist < best_dist:
                best, best_dist = fl, dist
        return best

    def _turtle_soup_short(self, s):
        """Turtle Soup short setup: buy-side liquidity swept, then bearish MSS.

        Returns (direction, kind, anchor_invalidation, anchor_zone, sweep)
        or None. Stricter than the generic sweep reversal:
        - sweep must take BUY-side liquidity (direction == -1)
        - requires a bearish CHoCH (not just BOS) after the sweep
        - requires bearish displacement (body >= 1.0x ATR down)
        The stop goes above the swept extreme with a wider buffer —
        gold's upside spikes hunt tight short stops.
        """
        sw = s.last_sweep
        if sw is None or s.sweep_age is None:
            return None
        if sw.direction != -1:  # buy-side sweep only
            return None
        if s.sweep_age > self.cfg.confirm_window:
            return None
        # hard bearish MSS: CHoCH down at/after the sweep
        ev = s.last_structure_event
        if ev is None or ev.kind != "CHOCH" or ev.direction != -1:
            return None
        if ev.index < sw.index:
            return None
        # bearish displacement since the sweep
        if s.bear_disp_age is None or s.bear_disp_age > s.sweep_age:
            return None
        # invalidation: above the swept extreme, wider buffer for shorts
        inv = sw.extreme + 0.30 * s.atr
        zone = (sw.level - 0.2 * s.atr, sw.level + 0.2 * s.atr)
        return -1, "turtle_soup", inv, zone, sw

    def _sweep_confirmed(self, s: MarketState, d: int) -> bool:
        """ICT confirmation for a sweep reversal — BOTH legs required:

        (a) displacement since the sweep bar: a candle body >= 1.0x ATR in
            the reversal direction, or a displacement FVG created by one;
        (b) a structure shift in the reversal direction at/after the sweep:
            a CHoCH/BOS event in direction ``d``, or the trend state itself
            flipping to ``d`` (the CHoCH that flipped it) after the sweep.
        """
        if s.last_sweep is None or s.sweep_age is None:
            return False
        age = s.sweep_age
        disp_age = s.bull_disp_age if d == 1 else s.bear_disp_age
        fvg_age = s.bull_disp_fvg_age if d == 1 else s.bear_disp_fvg_age
        displaced = (disp_age is not None and disp_age <= age) or \
            (fvg_age is not None and fvg_age <= age)
        if not displaced:
            return False
        ev = s.last_structure_event
        if ev is not None and ev.direction == d and ev.index >= s.last_sweep.index:
            return True
        return s.trend_since is not None and s.trend_since >= s.last_sweep.index \
            and s.bias == d

    def _daily_exception(self, s: MarketState, direction: int, kind: str,
                         sweep) -> bool:
        """The only trade allowed against the daily bias: a CONFIRMED sweep
        reversal taken at a major daily level (the swept H1 level or its
        extreme within tolerance of the last confirmed D1 swing)."""
        if kind != "sweep_reversal" or sweep is None:
            return False
        if self.cfg.require_sweep_confirmation \
                and not self._sweep_confirmed(s, direction):
            return False
        level = s.daily_swing_low if direction == 1 else s.daily_swing_high
        if level is None:
            return False
        tol = self.cfg.daily_level_tol_atr * s.atr
        return abs(sweep.level - level) <= tol or abs(sweep.extreme - level) <= tol

    @staticmethod
    def _dist_to_zone(s: MarketState, bottom: float, top: float) -> float:
        if s.close > top:
            return s.close - top
        if s.close < bottom:
            return bottom - s.close
        return 0.0

    def _fallback_invalidation(self, s: MarketState, direction: int) -> float:
        if direction == 1:
            if s.recent_swing_lows:
                return s.recent_swing_lows[-1].price - 0.15 * s.atr
            return s.close - 1.5 * s.atr
        if s.recent_swing_highs:
            return s.recent_swing_highs[-1].price + 0.15 * s.atr
        return s.close + 1.5 * s.atr

    # -------------------------------------------------------------- scoring
    def _structure_score(self, s: MarketState, d: int, kind: str,
                         factors: list[str]) -> float:
        ev = s.last_structure_event
        if kind == "sweep_reversal":
            if ev is not None and s.sweep_age is not None \
                    and ev.index >= (s.last_sweep.index if s.last_sweep else 0) \
                    and ev.direction == d:
                base = 20.0 if ev.kind == "CHOCH" else 17.0
                factors.append(f"{ev.kind} confirms the sweep reversal")
            elif s.trend == ("up" if d == 1 else "down"):
                base = 16.0
                factors.append("trend already flipped with the sweep")
            else:
                base = 12.0
                factors.append("sweep reversal against structure (awaiting CHoCH)")
        else:
            base = 16.0
            factors.append(f"trend-aligned continuation ({s.trend})")
            if ev is not None and ev.direction == d \
                    and s.structure_event_age is not None and s.structure_event_age <= 5:
                base = 18.0
                factors.append(f"fresh {ev.kind} {s.structure_event_age} bars ago")
        if d * s.momentum > 0.5:
            base = min(20.0, base + 2.0)
            factors.append(f"momentum aligned ({s.momentum:+.1f} ATR)")
        return base

    def _liquidity_score(self, s: MarketState, d: int,
                         factors: list[str]) -> float:
        cfg = self.cfg
        if s.last_sweep is not None and s.sweep_age is not None \
                and s.sweep_age <= cfg.sweep_fresh_bars:
            side = "sell-side" if d == 1 else "buy-side"
            factors.append(
                f"liquidity sweep of {side} pool @ {s.last_sweep.level:.1f}, "
                f"{s.sweep_age} bars ago")
            return max(8.0, 20.0 - 2.0 * s.sweep_age)
        # no sweep: credit for reacting at a level
        ob = s.nearest_ob(d)
        if ob is not None and self._dist_to_zone(s, ob.bottom, ob.top) <= cfg.level_near_atr * s.atr:
            factors.append("price reacting at an order block")
            return 12.0
        g = s.nearest_fvg(d)
        if g is not None and self._dist_to_zone(s, g.bottom, g.top) <= cfg.level_near_atr * s.atr:
            factors.append("price reacting at a fair value gap")
            return 12.0
        return 0.0

    def _displacement_score(self, s: MarketState, d: int,
                            factors: list[str]) -> float:
        disp = s.bull_disp_3 if d == 1 else s.bear_disp_3
        if disp >= 1.3:
            factors.append(f"strong displacement {disp:.1f}x ATR")
            return 15.0
        if disp >= 1.0:
            factors.append(f"displacement {disp:.1f}x ATR")
            return 12.0
        if disp >= 0.8:
            return 9.0
        if disp >= 0.5:
            return 5.0
        return 0.0

    def _room_score(self, room_atr: float, factors: list[str]) -> float:
        factors.append(f"room to next liquidity pool: {room_atr:.1f}x ATR")
        if room_atr >= 3.0:
            return 20.0
        if room_atr >= 2.0:
            return 15.0
        if room_atr >= 1.5:
            return 10.0
        return 6.0

    def _regime_score(self, s: MarketState, d: int,
                      factors: list[str]) -> float:
        if s.vol_regime == "compressed":
            base = 14.0
            factors.append(f"volatility compressed (ATR p{s.atr_percentile:.0f}) — coiled")
        elif s.vol_regime == "expanded":
            base = 6.0
            factors.append(f"volatility expanded (ATR p{s.atr_percentile:.0f})")
            if (d == 1 and s.eq_position > 0.85) or (d == -1 and s.eq_position < 0.15):
                factors.append("but price is extended at the range extreme")
                return 0.0
        else:
            base = 10.0
        # premium/discount discipline
        if d == 1 and s.eq_position > 0.75:
            base = max(0.0, base - 4.0)
            factors.append("buying in premium — penalised")
        if d == -1 and s.eq_position < 0.25:
            base = max(0.0, base - 4.0)
            factors.append("selling in discount — penalised")
        if d == 1 and s.zone == "discount":
            factors.append("long from discount")
        if d == -1 and s.zone == "premium":
            factors.append("short from premium")
        return base

    def _session_score(self, s: MarketState, factors: list[str]) -> float:
        if s.killzone:
            factors.append(f"{s.session} killzone")
            return 10.0
        if s.session == "overlap":
            factors.append("London/NY overlap")
            return 8.0
        if s.session in ("london", "newyork"):
            return 7.0
        if s.session == "asia":
            return 2.0
        return 3.0

    # ------------------------------------------------- Phase-3 features
    def _feature_score(self, s: MarketState, d: int, kind: str,
                       factors: list[str]) -> float:
        """Bounded backing adjustments from the enabled feature
        families.  Hand-set priors, deliberately modest next to the
        core families (structure/liquidity/room carry up to 20 each):
        features refine a thesis, they do not create one."""
        fams = self.cfg.feature_families or frozenset()
        fs = s.features
        total = 0.0
        if "bollinger" in fams:
            total += self._feat_bollinger(fs.bollinger, d, factors)
        if "fib" in fams:
            total += self._feat_fib(fs.fib, s, d, factors)
        if "ichimoku" in fams:
            total += self._feat_ichimoku(fs.ichimoku, d, factors)
        if "relstr" in fams:
            total += self._feat_relstr(fs.relstr, d, factors)
        if "patterns" in fams:
            total += self._feat_patterns(fs.patterns, d, factors)
        if "zones" in fams:
            total += self._feat_zones(fs.zones, d, factors)
        if "phases" in fams:
            total += self._feat_phases(fs.phase, d, factors)
        if "footprints" in fams:
            total += self._feat_footprints(fs.footprints, d, factors)
        if "volume" in fams:
            total += self._feat_volume(fs.volume, d, kind, factors)
        if "mtf" in fams:
            total += self._feat_mtf(fs.mtf, d, factors)
        # ---- Phase 4: context-gated re-designs + trendlines.  Each
        # gate is SILENT outside its context (see features/gating.py).
        # Phase 4.5 adds "aoc" (area of concentration, same contract).
        gated = [n for n in ("fib_ctx", "zones_ctx", "ichimoku_ctx",
                             "patterns_ctx", "footprints_ctx", "mtf_ctx",
                             "trendlines", "aoc", "divergence") if n in fams]
        if gated:
            from ..features import gating
            fn = {
                "fib_ctx": gating.fib_gate,
                "zones_ctx": gating.zones_gate,
                "ichimoku_ctx": gating.ichimoku_gate,
                "patterns_ctx": gating.patterns_gate,
                "footprints_ctx": gating.footprints_gate,
                "mtf_ctx": gating.mtf_gate,
                "trendlines": gating.trendlines_score,
                "aoc": gating.aoc_gate,
                "divergence": gating.divergence_gate,
            }
            for name in gated:
                for delta, factor in fn[name](s, fs, d, kind):
                    total += delta
                    factors.append(factor)
        return total

    def _feature_cap(self, s: MarketState, d: int) -> float | None:
        """Phase-4 score cap from the gated MTF veto (None = no cap)."""
        fams = self.cfg.feature_families or frozenset()
        if "mtf_ctx" not in fams or s.features is None:
            return None
        from ..features import gating
        return gating.mtf_cap(s, s.features, d)

    @staticmethod
    def _feat_bollinger(b, d: int, factors: list[str]) -> float:
        out = 0.0
        if b.squeeze:
            out += 3.0
            factors.append(
                f"Bollinger squeeze (bandwidth p{b.bw_percentile:.0f})")
        if b.walk == d:
            out += 4.0
            factors.append("walking the band with the move")
        elif b.walk == -d:
            out -= 4.0
            factors.append("price walking the OPPOSITE band")
        if b.rejection == d:
            out += 4.0
            factors.append("band-rejection wick supports the direction")
        elif b.rejection == -d:
            out -= 3.0
            factors.append("band rejected the move's direction")
        return out

    @staticmethod
    def _feat_fib(f, s: MarketState, d: int, factors: list[str]) -> float:
        out = 0.0
        if (d == 1 and f.in_ote_long) or (d == -1 and f.in_ote_short):
            out += 6.0
            factors.append("entry inside the OTE zone (0.62-0.79 Fib)")
        elif f.at_fib:
            out += 2.0
            factors.append(f"price at Fib {f.nearest_frac:.3f}")
        # a Fib level just ahead caps the early part of the move
        for _frac, price in f.levels:
            ahead = (price - s.close) * d
            if 0 < ahead <= 0.30 * s.atr:
                out -= 3.0
                factors.append(f"Fib {_frac:.3f} caps the move just ahead")
                break
        return out

    @staticmethod
    def _feat_ichimoku(i, d: int, factors: list[str]) -> float:
        out = 0.0
        if i.position == "above":
            if d == 1:
                out += 4.0
                factors.append("above the Ichimoku cloud")
            else:
                out -= 5.0
                factors.append("shorting into cloud support from below")
        elif i.position == "below":
            if d == -1:
                out += 4.0
                factors.append("below the Ichimoku cloud")
            else:
                out -= 5.0
                factors.append("buying into cloud resistance from below")
        elif i.position == "inside":
            out -= 2.0
            factors.append("inside the cloud — no-man's land")
        if i.tk_cross == d:
            out += 3.0
            factors.append("fresh TK cross in the direction")
        if i.cloud_twist == d:
            out += 2.0
            factors.append("cloud twist in the direction")
        if i.chikou == d:
            out += 1.0
        return max(-7.0, min(7.0, out))

    @staticmethod
    def _feat_relstr(r, d: int, factors: list[str]) -> float:
        if not r.available or r.score != r.score:   # unavailable / NaN
            return 0.0
        sd = r.score * d
        if sd >= 0.75:
            factors.append("relative strength confirms (gold leading peers)")
            return 5.0
        if sd >= 0.25:
            factors.append("relative strength leans with the move")
            return 2.0
        if sd <= -0.75:
            factors.append("relative strength DIVERGES (gold lagging peers)")
            return -5.0
        if sd <= -0.25:
            factors.append("relative strength leans against")
            return -2.0
        return 0.0

    @staticmethod
    def _feat_patterns(p, d: int, factors: list[str]) -> float:
        aligned = [x for x in p.active if x.direction == d]
        opposed = [x for x in p.active if x.direction == -d]
        out = 0.0
        if aligned:
            best = aligned[0]
            if any(x.status == "broken" for x in aligned):
                out += 6.0
                factors.append(f"{best.kind} neckline broken in direction")
            else:
                out += 4.0
                factors.append(f"{best.kind} formed, neckline intact")
        if opposed:
            if any(x.status == "broken" for x in opposed):
                out -= 6.0
            else:
                out -= 4.0
            factors.append(f"opposing {opposed[0].kind} active")
        return out

    @staticmethod
    def _feat_zones(z, d: int, factors: list[str]) -> float:
        out = 0.0
        if z.in_demand and z.nearest_demand is not None:
            fresh = z.nearest_demand.freshness == "fresh"
            if d == 1:
                out += 6.0 if fresh else 3.0
                factors.append(
                    f"at a {'fresh' if fresh else 'tested'} demand zone")
            else:
                out -= 4.0
                factors.append("shorting into a demand zone")
        elif z.in_supply and z.nearest_supply is not None:
            fresh = z.nearest_supply.freshness == "fresh"
            if d == -1:
                out += 6.0 if fresh else 3.0
                factors.append(
                    f"at a {'fresh' if fresh else 'tested'} supply zone")
            else:
                out -= 4.0
                factors.append("buying into a supply zone")
        else:
            if z.nearest_demand is not None and d == 1 \
                    and z.demand_dist_atr == z.demand_dist_atr \
                    and z.demand_dist_atr <= 0.35:
                out += 2.0
                factors.append("demand zone just below price")
            if z.nearest_supply is not None and d == -1 \
                    and z.supply_dist_atr == z.supply_dist_atr \
                    and z.supply_dist_atr <= 0.35:
                out += 2.0
                factors.append("supply zone just above price")
        return out

    @staticmethod
    def _feat_phases(p, d: int, factors: list[str]) -> float:
        out = 0.0
        if p.event in ("spring", "upthrust", "manipulation_ssl",
                       "manipulation_bsl") and p.event_direction == d:
            out += 4.0
            factors.append(f"manipulation resolved ({p.event}) — AMD fuel")
        if p.event == "manipulation_fallback" \
                and p.event_direction == d:
            # Inferred (BOS + sweep + FVG), not witnessed: real fuel,
            # scored below the direct sweep's +4.0.
            out += 3.0
            factors.append("manipulation inferred (BOS+sweep+FVG) — AMD fuel")
        if p.phase == "accumulation":
            out += 3.0 if d == 1 else -3.0
            if d == 1:
                factors.append("accumulation base — longs favoured")
        elif p.phase == "distribution":
            out += 3.0 if d == -1 else -3.0
            if d == -1:
                factors.append("distribution base — shorts favoured")
        elif p.phase == "expansion_up":
            out += 2.0 if d == 1 else -2.0
        elif p.phase == "expansion_down":
            out += 2.0 if d == -1 else -2.0
        elif p.phase == "trend_up":
            out += 1.0 if d == 1 else -1.0
        elif p.phase == "trend_down":
            out += 1.0 if d == -1 else -1.0
        return out

    @staticmethod
    def _feat_footprints(f, d: int, factors: list[str]) -> float:
        out = 0.0
        if f.institutional_dir == d:
            out += 7.0
            factors.append("institutional entry signature in direction")
        elif f.institutional_dir == -d:
            out -= 4.0
            factors.append("institutional signature AGAINST the move")
        if f.retail_chase == d:
            out -= 5.0
            factors.append("retail chase character — late, crowded entry")
        if f.retail_trap == d:
            out += 4.0
            factors.append("retail trap — chase crowd trapped, fade fuel")
        if f.cta_trend == d:
            out += 3.0
            factors.append("CTA-character persistent trend flow aligned")
        elif f.cta_trend == -d:
            out -= 3.0
            factors.append("CTA-character flow against the move")
        return max(-8.0, min(8.0, out))

    @staticmethod
    def _feat_volume(v, d: int, kind: str, factors: list[str]) -> float:
        if not v.available:
            return 0.0
        out = 0.0
        if v.climax_reversal == d:
            out += 4.0
            factors.append("volume climax reversal in direction")
        elif v.climax_reversal == -d:
            out -= 3.0
            factors.append("volume climax reversal AGAINST the move")
        if v.rvol == v.rvol:   # finite
            if v.rvol >= 1.5:
                out += 2.0
                factors.append(f"volume confirming (RVOL {v.rvol:.1f})")
            elif v.rvol <= 0.5:
                out -= 3.0
                factors.append(f"dead volume (RVOL {v.rvol:.1f})")
        if v.dry_up and kind == "continuation":
            out -= 2.0
            factors.append("volume dried up under the continuation")
        return out

    @staticmethod
    def _feat_mtf(m, d: int, factors: list[str]) -> float:
        aligned = m.aligned_count(d)
        opposed = m.opposed_count(d)
        out = max(-7.0, min(7.0, 3.0 * aligned - 4.0 * opposed))
        if aligned == 2:
            factors.append("D1 + H4 both aligned (MTF confluence)")
        elif opposed == 2:
            factors.append("D1 + H4 both OPPOSED (MTF conflict)")
        elif aligned == 1 and opposed == 0:
            factors.append("one HTF aligned, other neutral")
        elif opposed == 1 and aligned == 0:
            factors.append("one HTF opposed")
        return out
