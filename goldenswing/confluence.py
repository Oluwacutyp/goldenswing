"""The confluence engine (Phase 9) — the trader's desk.

One AWARE system, multiple entry playbooks.  Each bar the engine:

1. collects candidates from every playbook (each pre-simulated with
   the exact machinery it was validated with — the engine never
   re-simulates and never edits an outcome);
2. ranks them by confluence: playbook base score + context bonuses
   measured by the mining program (AOC backing, divergence presence,
   imbalance-magnet proximity, killzone, HTF bias alignment);
3. takes the best available setup into the live book — one position at
   a time, risk as the final gate;
4. journals everything like a trader's log: what fired, what was
   skipped and why, the full thesis behind every fill.

Live vs shadow: only playbooks with ``live=True`` may enter the live
P&L book.  Shadow playbooks are scored, logged and graded in their own
paper books — promotion to live needs OOS evidence (CONFLUENCE.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .backtest.harness import Outcome, book_metrics
from .contextmap import context_levels
from .execution.learning import TradeJournal
from .execution.risk import GuardDecision, RiskGuards
from .features import gating
from .playbooks.base import Candidate, EngineContext, Playbook


@dataclass
class ConfluenceConfig:
    """Context bonus sizes.  AOC / divergence / imbalance-magnet /
    trap-boost are independent measured findings (their own experiment
    reports), not context-map dimensions — they stay deliberately small
    next to the playbook base scores.  Session / regime / HTF / phase
    bonuses come from each playbook's MEASURED context map
    (``hunting_ground.bonuses``), never from priors.  Trap-boost is
    additionally playbook-gated (``trap_bonuses``): only the playbooks
    it was measured on earn it."""

    bonus_aoc: float = 4.0          # entry backed by an AOC pivot zone
    bonus_divergence: float = 3.0   # any active RSI divergence (measured:
                                    # presence is real, direction is folklore)
    bonus_magnet: float = 2.0       # unfilled FVG within 0.5 ATR (the
                                    # imbalance magnet is weak but real)
    magnet_tol_atr: float = 0.5
    bonus_forecast_agree: float = 2.0  # 1d AND 3d horizons both rank the
                                    # signal's direction top-decile (the
                                    # measured quality dial: gated kept
                                    # +0.28R vs dropped +0.10R; agreement
                                    # is a bonus/sizing input, NEVER a veto)
    bonus_trap_boost: float = 4.0   # direction-aligned GOD-CE trap boost
                                    # (SFP/beyond-FVG/iFVG) active at the
                                    # signal bar — measured for
                                    # trendline_retest only (see
                                    # experiments/GODCE_SALVAGE.md): boost
                                    # subset +0.574R (n=43) vs +0.248R
                                    # overall; whole book +15.40%→+22.04%,
                                    # PF 1.43→2.16 at the signal-bar
                                    # convention; OOS +14.03%→+20.72%.
                                    # Playbook-gated in trap_bonuses().
    bonus_forecast_single: float = 1.0  # exactly one of 1d/3d top-decile
    forecast_top_pct: float = 10.0      # "top decile" definition
    forecast_size_mult: float = 1.25    # position-size multiplier on
                                    # horizon agreement (sizing input only)
    cost_bps: float = 4.0           # broker-cost gate: round-trip cost in
                                    # basis points of entry price (pine
                                    # GOD-CE costBps default)
    min_net_tp1_r: float = 0.6      # broker-cost gate: TP1's R multiple
                                    # net of the cost term must clear this
                                    # (pine GOD-CE minNetRR default)


@dataclass
class ScoredCandidate:
    candidate: Candidate
    confluence: float
    bonuses: list[str] = field(default_factory=list)


def _signal_bonuses(signal_index: int, direction: int, signal,
                    ctx: EngineContext,
                    cfg: ConfluenceConfig) -> tuple[float, list[str]]:
    """Feature bonuses for one signal, from confirmed state at its bar.
    The shared core behind :func:`context_bonuses` (batch) and live
    scoring — one implementation, no divergence."""
    if signal_index >= len(ctx.states):
        return 0.0, []  # no chart state available: no bonuses, no crash
    s = ctx.states[signal_index]
    fs = s.features
    d = direction
    total = 0.0
    notes: list[str] = []

    if fs is not None:
        for delta, factor in gating.aoc_gate(s, fs, d, signal.kind):
            if delta <= 0:
                continue
            already = any("concentration" in f
                          for f in signal.factors)
            if already:
                notes.append("AOC backing already in playbook score "
                             f"({factor})")
            else:
                total += cfg.bonus_aoc
                notes.append(f"+{cfg.bonus_aoc:g} AOC backing: {factor}")
        dv = fs.divergence
        if dv is not None and dv.divergences:
            kinds = ",".join(sorted({x.kind for x in dv.divergences}))
            total += cfg.bonus_divergence
            notes.append(f"+{cfg.bonus_divergence:g} divergence present "
                         f"({kinds}) — energy marker, direction ignored")

    atr = s.atr
    if atr and atr > 0:
        entry_ref = signal.entry_ref or s.close
        for g in s.active_fvgs:
            if abs(g.midpoint - entry_ref) <= cfg.magnet_tol_atr * atr:
                total += cfg.bonus_magnet
                notes.append(
                    f"+{cfg.bonus_magnet:g} imbalance magnet: unfilled FVG "
                    f"midline {g.midpoint:.1f} within "
                    f"{cfg.magnet_tol_atr:g} ATR")
                break

    return total, notes


def context_bonuses(cand: Candidate, ctx: EngineContext,
                    cfg: ConfluenceConfig) -> tuple[float, list[str]]:
    """Confluence bonuses for one candidate, from confirmed state at its
    signal bar.  Every bonus is a measured finding, named in the log:

    - AOC backing (+4): recomputed via the aoc gate.  The anchor's
      detector score ALREADY includes it, so it is not re-added there
      (detected from the signal's factors) — only named.
    - Divergence presence (+3): any active divergence marks energy
      (DIVERGENCE.md: aligned ~= opposing, so direction is ignored).
    - Imbalance-magnet proximity (+2): an unfilled FVG midline within
      0.5 ATR of the signal bar (IMBALANCE.md: weak but real magnet).

    Session / regime / HTF / phase bonuses are NOT here — they come
    from each playbook's measured context map (``map_bonuses``), never
    from hand-set priors.
    """
    return _signal_bonuses(cand.signal_index, cand.direction,
                           cand.outcome.signal, ctx, cfg)


def map_bonuses(cand: Candidate, pb: Playbook,
                ctx: EngineContext) -> tuple[float, list[str]]:
    """Confluence bonuses from the playbook's MEASURED context map.

    A candidate firing in one of its playbook's proven context cells
    earns the map-derived bonus.  This is where session / regime / HTF /
    phase bonuses come from — data, not priors.
    """
    hg = pb.hunting_ground
    if not hg.bonuses:
        return 0.0, []
    levels = context_levels(ctx.states, cand.signal_index, cand.direction)
    total = 0.0
    notes: list[str] = []
    for (dim, level), bonus in hg.bonuses.items():
        if levels.get(dim) == level:
            total += bonus
            notes.append(f"+{bonus:g} proven {dim}={level} (context map)")
    return total, notes


# ------------------------------------------------- GOD-CE trap salvage
# Only the trap-boost survived measurement as a confluence input
# (experiments/GODCE_SALVAGE.md).  The bonus is playbook-gated: it was
# measured for trendline_retest and means nothing elsewhere until
# measured there.

_TRAP_BOOST_PLAYBOOKS = frozenset({"trendline_retest"})


def ensure_trap_features(ctx: EngineContext):
    """The GOD-CE trap-intelligence frame for ``ctx.df``, computed once
    and cached on the context.  The state machines are causal
    (backward-looking), so the frame is valid for both batch scoring
    and live signals on the same history."""
    tf = ctx.trap_features
    if tf is None:
        from goldenswing.features.traps import compute_trap_features
        tf = compute_trap_features(ctx.df)
        ctx.trap_features = tf
    return tf


def trap_bonuses(signal_index: int, direction: int, pb_name: str,
                 ctx: EngineContext,
                 cfg: ConfluenceConfig) -> tuple[float, list[str]]:
    """GOD-CE trap-boost bonus — playbook-gated, signal-bar convention.

    A candidate whose direction-aligned trap boost (SFP / beyond-FVG
    reclaim / iFVG test, ``trap_boost_long`` / ``trap_boost_short``) is
    active at the signal bar earns ``cfg.bonus_trap_boost``.  Measured
    for ``trendline_retest`` only (GODCE_SALVAGE.md): boost subset
    +0.574R (n=43) vs +0.248R overall; whole book +15.40%→+22.04%,
    PF 1.43→2.16; OOS +14.03%→+20.72% (21 trades, PF 3.09).
    """
    notes: list[str] = []
    if pb_name not in _TRAP_BOOST_PLAYBOOKS:
        return 0.0, notes
    try:
        tf = ensure_trap_features(ctx)
    except Exception:
        return 0.0, notes  # trap frame unavailable: no bonus, no crash
    if signal_index < 0 or signal_index >= len(tf):
        return 0.0, notes
    row = tf.iloc[signal_index]
    col = "trap_boost_long" if direction > 0 else "trap_boost_short"
    if bool(row[col]):
        return cfg.bonus_trap_boost, [
            f"+{cfg.bonus_trap_boost:g} trap-boost alignment "
            f"({col} active at signal bar — GODCE_SALVAGE measured)"]
    return 0.0, notes


# horizons (in bars) that carry the measured signal: 1d and 3d.
_FORECAST_HORIZONS = (24, 72)


def forecast_agreement(signal_index: int, direction: int, table,
                       top_pct: float = 10.0) -> tuple[int, list[str]]:
    """How many of the measured horizons (1d, 3d) rank the signal's
    direction in their top ``top_pct`` — 0, 1 or 2.

    Returns (n_agree, notes).  Pure read of the rank table; a missing
    horizon or a NaN rank simply doesn't count.  This is the quality
    dial, not a gate: disagreement changes nothing.
    """
    notes: list[str] = []
    try:
        scores = table.scores
    except AttributeError:
        return 0, []
    n = 0
    for h in _FORECAST_HORIZONS:
        hs = scores.get(h)
        if hs is None:
            continue
        try:
            rank = hs.rank(direction, signal_index)
        except (IndexError, TypeError, ValueError):
            continue
        if math.isfinite(rank) and rank >= 100.0 - top_pct:
            n += 1
            notes.append(f"forecaster {hs.tag} rank top "
                         f"{max(1.0, 100.0 - rank):.0f}%")
    return n, notes


def forecast_bonuses(signal_index: int, direction: int, table,
                     cfg: ConfluenceConfig) -> tuple[float, list[str]]:
    """Confluence bonus from forecaster horizon agreement.

    Agreement (both 1d and 3d top-decile) earns ``bonus_forecast_agree``;
    a single horizon earns ``bonus_forecast_single``; disagreement earns
    nothing and — critically — blocks nothing.  The forecaster is a
    quality dial: the gate test measured kept +0.28R vs dropped +0.10R,
    so agreement nudges score and size, never vetoes.
    """
    if table is None:
        return 0.0, []
    n, notes = forecast_agreement(signal_index, direction, table,
                                  cfg.forecast_top_pct)
    if n >= 2:
        return cfg.bonus_forecast_agree, \
            [f"+{cfg.bonus_forecast_agree:g} forecaster 1d+3d agreement: "
             + ", ".join(notes)]
    if n == 1:
        return cfg.bonus_forecast_single, \
            [f"+{cfg.bonus_forecast_single:g} forecaster single horizon: "
             + ", ".join(notes)]
    return 0.0, []


def cost_gate_ok(signal, entry_price: float, cfg: ConfluenceConfig,
                 playbook: str | None = None,
                 spread_px: float | None = None) -> tuple[bool, str]:
    """Broker-cost entry filter — pine-faithful port of the GOD-CE cost
    gate (``godce/engine.py``, from GOD-CE-SLIM.pine §entries), lifted
    from the engine's fixed TP1 to any candidate's own targets.

    The pine blocks an entry when::

        cost_px = close * costBps / 10000
        risk    = planned stop distance in price
        NOT (risk >= 2 * cost_px
             AND tp1RR - cost_px / risk >= minNetRR)

    This port keeps the two legs and the defaults (``cost_bps=4.0``,
    ``min_net_tp1_r=0.6``, ``tp1RR=1.5``) but generalises the planned
    TP1:

    - ``tp1_r`` is the nearest of ``signal.targets`` lying AHEAD of the
      entry in the trade direction (``(tp1 - entry) * direction / risk``).
    - Fallback when no target lies ahead: ``fvg_midline`` → 2.0 (its
      validated fixed-2R exit — the exit IS the plan there), anything
      else → 1.5 (the pine's configured ``tp1RR`` default).  The
      fallback is documented in the reason string whenever it decides
      the outcome.

    ``playbook`` is only the fallback disambiguator; when omitted the
    fvg_midline sleeve is detected from ``signal.kind`` (``"choch_fvg"``
    in the kind).  ``spread_px`` (live tick spread, price units)
    overrides the ``cost_bps`` estimate: the spread IS the one-way
    cost on a zero-commission account, and the two legs already
    double it for the round trip.  A missing / non-positive spread
    falls back to ``cost_bps``.

    The gate NEVER raises: missing/non-finite entry
    price, missing invalidation, or ``risk <= 0`` all fail closed with
    a named reason.
    """
    try:
        entry = float(entry_price)
    except (TypeError, ValueError):
        return False, "no-entry-price: entry price missing or not a number"
    if not math.isfinite(entry) or entry <= 0:
        return False, "no-entry-price: entry price not a positive number"

    inv = getattr(signal, "invalidation", None)
    try:
        inv = float(inv)
    except (TypeError, ValueError):
        inv = float("nan")
    if not math.isfinite(inv):
        return False, "no-invalidation: signal.invalidation missing"

    d = getattr(signal, "direction", 0)
    if d not in (1, -1):
        return False, f"no-direction: direction={d!r} is not +1/-1"

    cost_px = entry * cfg.cost_bps / 10000.0
    cost_src = f"cost_bps={cfg.cost_bps:g}"
    if spread_px is not None:
        try:
            spread_px = float(spread_px)
        except (TypeError, ValueError):
            spread_px = float("nan")
        if math.isfinite(spread_px) and spread_px > 0:
            cost_px = spread_px
            cost_src = f"live-spread={spread_px:.2f}"
    risk = abs(entry - inv)
    if risk <= 0:
        return False, "risk<=0: entry sits exactly on the invalidation"

    # Leg 1 — the stop must be wide enough that round-trip cost does
    # not dominate the trade (pine: p_risk >= 2 * cost_px).
    if risk < 2.0 * cost_px:
        return False, (
            f"risk-too-small: risk {risk:.2f} < 2×cost "
            f"{2.0 * cost_px:.2f} ({cost_src})")

    # Planned TP1 in R, from the signal's own liquidity pools ahead.
    targets = getattr(signal, "targets", None) or []
    ahead = [t for t in targets
             if isinstance(t, (int, float)) and math.isfinite(t)
             and (t - entry) * d > 0]
    if ahead:
        tp1 = min(ahead, key=lambda t: abs(t - entry))
        tp1_r = (tp1 - entry) * d / risk
        tp1_src = "nearest-target"
    else:
        kind = getattr(signal, "kind", "") or ""
        is_fvg = (playbook == "fvg_midline") or ("choch_fvg" in kind)
        tp1_r = 2.0 if is_fvg else 1.5
        tp1_src = ("fvg-fixed-2R-fallback" if is_fvg
                   else "pine-tp1RR-default")

    # Leg 2 — TP1 net of the cost term must clear the floor (pine:
    # tp1RR - cost_px / risk >= minNetRR).
    cost_r = cost_px / risk
    net_tp1_r = tp1_r - cost_r
    if net_tp1_r < cfg.min_net_tp1_r:
        return False, (
            f"net-tp1-r-too-low: tp1_r {tp1_r:.2f} ({tp1_src}) - "
            f"cost/risk {cost_r:.2f} = {net_tp1_r:.2f} < "
            f"{cfg.min_net_tp1_r:g}")
    return True, ""


@dataclass
class EngineReport:
    live_outcomes: list[Outcome] = field(default_factory=list)
    live_metrics: dict = field(default_factory=dict)
    shadow: dict = field(default_factory=dict)      # playbook -> metrics
    scored: dict = field(default_factory=dict)      # playbook -> [ScoredCandidate]
    risk_log: list[dict] = field(default_factory=list)
    skips: dict = field(default_factory=dict)       # reason -> count
    journal_path: str = ""


class ConfluenceEngine:
    """Ranks playbook candidates by confluence and runs the books.

    ``risk_mode``: "enforce" (live trading — a denied trade is skipped)
    or "shadow" (backtest — every risk decision is computed and logged
    but the historical sequence is never altered, so the anchor canary
    stays exact).
    """

    def __init__(self, playbooks: list[Playbook], risk_guards: RiskGuards,
                 journal: TradeJournal,
                 cfg: ConfluenceConfig | None = None,
                 risk_mode: str = "shadow") -> None:
        assert risk_mode in ("enforce", "shadow")
        self.playbooks = playbooks
        self.risk = risk_guards
        self.journal = journal
        self.cfg = cfg or ConfluenceConfig()
        self.risk_mode = risk_mode

    # ------------------------------------------------------------ scoring
    def score(self, cand: Candidate, ctx: EngineContext,
              pb: Playbook) -> ScoredCandidate:
        bonus, notes = context_bonuses(cand, ctx, self.cfg)
        mb, mnotes = map_bonuses(cand, pb, ctx)
        tb, tnotes = trap_bonuses(cand.signal_index, cand.direction,
                                  pb.name, ctx, self.cfg)
        fb, fnotes = forecast_bonuses(
            cand.signal_index, cand.direction,
            getattr(ctx, "forecast_table", None), self.cfg)
        return ScoredCandidate(candidate=cand,
                               confluence=cand.base_score + bonus + mb + tb + fb,
                               bonuses=notes + mnotes + tnotes + fnotes)

    # --------------------------------------------------- live signals
    def gate_live_signal(self, pb: Playbook, signal,
                         ctx: EngineContext,
                         hierarchy_verdict: dict | None = None,
                         spread_px: float | None = None
                         ) -> tuple[bool, str]:
        """The context gates for a live (un-simulated) signal — the
        same gates the batch path applies, in the same order, plus
        the all-timeframe hierarchy gate (gate 0, first):

        0. hierarchy (when a verdict is supplied): the D1 -> H4 ->
           H1 -> entry-clock chain must permit this signal's
           direction.  A hierarchy-blocked signal never fires — that
           is the correct behaviour, not a regression.  ``None``
           means no stack was available (legacy single-TF feed):
           the gate is skipped, not faked.
        1. session scope (first-class; never fires outside it),
        2. the playbook's required-context tripwire,
        3. the measured hunting-ground exclusions,
        4. the broker-cost gate (``cost_gate_ok``): the planned
           stop must clear twice the cost and TP1 net of cost must
           clear the floor.  ``spread_px`` (live tick spread) is
           the cost when available; without it the ``cost_bps``
           estimate is used (paper loop).
        """
        if hierarchy_verdict is not None:
            from .live.hierarchy import verdict_permits
            ok, reason = verdict_permits(hierarchy_verdict,
                                         signal.direction)
            if not ok:
                return False, reason
        ok, reason = self._session_allowed(pb, signal.index, ctx)
        if not ok:
            return False, reason
        ok, reason = pb.check_live_signal(signal, ctx)
        if not ok:
            return False, f"required-context tripwire: {reason}"
        ok, reason = self._check_hunting_ground_signal(
            pb, signal, ctx)
        if not ok:
            return False, reason
        ok, reason = cost_gate_ok(
            signal, getattr(signal, "entry_ref", None),
            self.cfg, playbook=pb.name, spread_px=spread_px)
        if not ok:
            return False, f"cost_gate: {reason}"
        return True, ""

    def score_live_signal(self, pb: Playbook, signal,
                          ctx: EngineContext,
                          hierarchy_verdict: dict | None = None
                          ) -> tuple[float, list[str]]:
        """Confluence for a live signal: base + feature bonuses + map
        bonuses + forecaster quality dial.  Same arithmetic as the batch
        :meth:`score`.

        When ``hierarchy_verdict`` carries ``volume`` (MT5 bars — see
        ``live/volume.py``), the old bots' bounded volume confirmation
        (``clip((ratio - 1) / 2, -0.5, 0.5)``) is added UNSCALED:
        the scale is the old bots', not a new invented weight.  Volume
        never gates — it only nudges the score.
        """
        bonus, notes = _signal_bonuses(
            signal.index, signal.direction, signal, ctx, self.cfg)
        mb, mnotes = self._map_bonuses_signal(pb, signal, ctx)
        tb, tnotes = trap_bonuses(
            signal.index, signal.direction, pb.name, ctx, self.cfg)
        fb, fnotes = forecast_bonuses(
            signal.index, signal.direction,
            getattr(ctx, "forecast_table", None), self.cfg)
        if hierarchy_verdict is not None:
            from .live.volume import volume_bonus
            vb, vnote = volume_bonus(hierarchy_verdict.get("volume"))
            if vb != 0.0:
                bonus += vb
                notes.append(f"{vb:+g} volume confirmation ({vnote})")
            # DXY correlation (Ghost Sentinel bands, bounded ±0.5):
            # regime health, never a gate.  None = DXY unavailable.
            from .live.dxy import dxy_bonus
            db, dnote = dxy_bonus(hierarchy_verdict.get("dxy_corr"))
            if db != 0.0:
                bonus += db
                notes.append(f"{db:+g} DXY correlation ({dnote})")
        return signal.score + bonus + mb + tb + fb, notes + mnotes + tnotes + fnotes

    @staticmethod
    def _session_allowed(pb: Playbook, signal_index: int,
                         ctx: EngineContext) -> tuple[bool, str]:
        """Session scope is first-class: a session-scoped playbook never
        fires outside its session.  \"any\" fires everywhere."""
        if pb.session_scope == "any":
            return True, ""
        if signal_index >= len(ctx.states):
            return False, "no chart state for session-scope check"
        sess = ctx.states[signal_index].session
        if sess != pb.session_scope:
            return False, (
                f"session_scope={pb.session_scope}: signal fired in "
                f"{sess} session")
        return True, ""

    def _check_session_scope(self, pb: Playbook, cand: Candidate,
                             ctx: EngineContext) -> tuple[bool, str]:
        return self._session_allowed(pb, cand.signal_index, ctx)

    def _check_hunting_ground_signal(self, pb: Playbook, signal,
                                     ctx: EngineContext) -> tuple[bool, str]:
        hg = pb.hunting_ground
        if not hg.excluded:
            return True, ""
        levels = context_levels(ctx.states, signal.index,
                                signal.direction)
        for dim, level in hg.excluded:
            if levels.get(dim) == level:
                return False, (
                    f"outside hunting ground: {dim}={level} is excluded "
                    f"for {pb.name} (measured context map)")
        return True, ""

    def _map_bonuses_signal(self, pb: Playbook, signal,
                            ctx: EngineContext) -> tuple[float, list[str]]:
        hg = pb.hunting_ground
        if not hg.bonuses:
            return 0.0, []
        levels = context_levels(ctx.states, signal.index,
                                signal.direction)
        total = 0.0
        notes: list[str] = []
        for (dim, level), bonus in hg.bonuses.items():
            if levels.get(dim) == level:
                total += bonus
                notes.append(f"+{bonus:g} proven {dim}={level} (context map)")
        return total, notes

    # -------------------------------------------------- context gates
    @staticmethod
    def _check_session_scope(pb: Playbook, cand: Candidate,
                             ctx: EngineContext) -> tuple[bool, str]:
        """Session scope is first-class: a session-scoped playbook never
        fires outside its session.  \"any\" fires everywhere."""
        if pb.session_scope == "any":
            return True, ""
        t = cand.signal_index
        if t >= len(ctx.states):
            return False, "no chart state for session-scope check"
        sess = ctx.states[t].session
        if sess != pb.session_scope:
            return False, (
                f"session_scope={pb.session_scope}: signal fired in "
                f"{sess} session")
        return True, ""

    @staticmethod
    def _check_hunting_ground(pb: Playbook, cand: Candidate,
                              ctx: EngineContext) -> tuple[bool, str]:
        """The measured hunting ground: excluded cells never fire."""
        hg = pb.hunting_ground
        if not hg.excluded:
            return True, ""
        levels = context_levels(ctx.states, cand.signal_index,
                                cand.direction)
        for dim, level in hg.excluded:
            if levels.get(dim) == level:
                return False, (
                    f"outside hunting ground: {dim}={level} is excluded "
                    f"for {pb.name} (measured context map)")
        return True, ""

    def _context_skip(self, rep: EngineReport, ctx: EngineContext,
                      pb: Playbook, sc: ScoredCandidate,
                      reason: str) -> None:
        """A candidate blocked by a context gate — session scope,
        setup tripwire, or hunting ground.  Loud skip: this is the
        trader's discipline (or a system bug), never a silent drop."""
        self._skip(rep, ctx, sc, "context-violation", reason)
        self.journal.log_note(
            "CONTEXT GATE — candidate blocked outside its playbook's "
            "declared context; never booked",
            playbook=pb.name, bar=sc.candidate.signal_index,
            direction=sc.candidate.direction, reason=reason,
            thesis=sc.candidate.thesis)

    # --------------------------------------------------------------- run
    def run(self, ctx: EngineContext) -> EngineReport:
        rep = EngineReport(journal_path=str(self.journal.path))
        scored: dict[str, list[ScoredCandidate]] = {}
        for pb in self.playbooks:
            kept: list[ScoredCandidate] = []
            for c in pb.candidates(ctx):
                # 1. session scope — first-class gate
                ok, reason = self._check_session_scope(pb, c, ctx)
                if not ok:
                    self._context_skip(
                        rep, ctx, pb,
                        ScoredCandidate(candidate=c,
                                        confluence=c.base_score),
                        reason)
                    continue
                # 2. setup-context tripwire — a firing outside the
                #    declared required context is a system bug
                ok, reason = pb.check_context(c, ctx)
                if not ok:
                    self._context_skip(
                        rep, ctx, pb,
                        ScoredCandidate(candidate=c,
                                        confluence=c.base_score),
                        f"SYSTEM BUG CANDIDATE: {reason}")
                    continue
                # 3. hunting ground — measured exclusions never fire
                ok, reason = self._check_hunting_ground(pb, c, ctx)
                if not ok:
                    self._context_skip(
                        rep, ctx, pb,
                        ScoredCandidate(candidate=c,
                                        confluence=c.base_score),
                        reason)
                    continue
                kept.append(self.score(c, ctx, pb))
            scored[pb.name] = kept
        rep.scored = scored

        live_cands = [sc for pb in self.playbooks if pb.live
                      for sc in scored[pb.name]
                      if sc.candidate.base_score >= pb.live_threshold]
        rep.live_outcomes = self._run_live_book(ctx, live_cands, rep)
        rep.live_metrics = book_metrics(rep.live_outcomes, ctx.years)

        for pb in self.playbooks:
            if pb.live:
                continue
            outs = [sc.candidate.outcome for sc in scored[pb.name]]
            book = self._greedy(outs, 0.0)
            rep.shadow[pb.name] = {
                "book": book_metrics(book, ctx.years),
                "n_candidates": len(outs),
            }
            for sc in scored[pb.name]:
                self._journal_shadow(ctx, sc)
        return rep

    # ----------------------------------------------------------- live book
    @staticmethod
    def _greedy(outcomes: list[Outcome],
                threshold: float) -> list[Outcome]:
        """One position at a time, chronological.  Identical to the
        harness's build_book for a single playbook (the anchor canary
        depends on it)."""
        book: list[Outcome] = []
        free_from = -1
        for o in sorted(outcomes, key=lambda x: x.signal.index):
            if o.signal.score < threshold:
                continue
            if o.signal.index < free_from:
                continue
            book.append(o)
            free_from = o.exit_index
        return book

    def _run_live_book(self, ctx: EngineContext,
                       live_cands: list[ScoredCandidate],
                       rep: EngineReport) -> list[Outcome]:
        # Same-bar conflicts (two playbooks, one bar) resolve to the
        # higher confluence — the trader takes the better setup.
        ordered = sorted(live_cands,
                         key=lambda sc: (sc.candidate.signal_index,
                                         -sc.confluence))
        book: list[Outcome] = []
        free_from = -1
        balance = ctx.start_balance
        for sc in ordered:
            cand = sc.candidate
            o = cand.outcome
            if o.signal.score < self._live_threshold_for(cand.playbook):
                continue
            if cand.signal_index < free_from:
                self._skip(rep, ctx, sc, "position-lock",
                           "already in a position from "
                           f"{book[-1].signal.kind if book else '?'}")
                continue
            risk_money = 0.01 * balance
            decision = self.risk.check_new_trade(
                balance=balance, open_risk_money=0.0,
                new_risk_money=risk_money, open_count=0)
            rep.risk_log.append({
                "bar": cand.signal_index, "playbook": cand.playbook,
                "allowed": decision.allowed
                if self.risk_mode == "enforce" else True,
                "risk_would_allow": decision.allowed,
                "reason": decision.reason,
                "mode": self.risk_mode,
            })
            if self.risk_mode == "enforce" and not decision.allowed:
                self._skip(rep, ctx, sc, "risk",
                           f"risk gate denied: {decision.reason}")
                continue
            if self.risk_mode == "shadow" and not decision.allowed:
                self.journal.log_note(
                    "risk would have blocked this trade in live mode",
                    playbook=cand.playbook, bar=cand.signal_index,
                    reason=decision.reason)
            self._journal_open(ctx, sc, risk_money)
            book.append(o)
            free_from = o.exit_index
            pnl_money = o.r_net * risk_money
            balance = balance * (1.0 + o.r_net * 0.01)
            self.risk.record_outcome(pnl_money, balance)
            self._journal_close(ctx, sc, pnl_money)
        return book

    def _live_threshold_for(self, playbook_name: str) -> float:
        for pb in self.playbooks:
            if pb.name == playbook_name:
                return pb.live_threshold
        return 60.0

    # ------------------------------------------------------------ journal
    def _journal_open(self, ctx: EngineContext, sc: ScoredCandidate,
                      risk_money: float) -> None:
        cand, o, sig = sc.candidate, sc.candidate.outcome, \
            sc.candidate.outcome.signal
        s = ctx.states[cand.signal_index] \
            if cand.signal_index < len(ctx.states) else None
        phase = getattr(getattr(s.features, "phase", None), "phase",
                        "unknown") if s is not None and s.features \
            else "unknown"
        entry = ctx.df["open"].to_numpy()[o.entry_index] \
            if o.entry_index < len(ctx.df) else float("nan")
        self.journal.log_open(
            alert_id=f"cfx-{cand.playbook}-{cand.signal_index}",
            instrument=ctx.instrument, mode="swing",
            playbook=cand.playbook, phase=str(phase),
            direction=cand.direction, score=sc.confluence,
            source="confluence-engine", ticket=None, entry=float(entry),
            stop=float(sig.invalidation), volume=0.0,
            risk_money=float(risk_money),
            factors=[cand.thesis, f"base {cand.base_score:.1f}",
                     f"confluence {sc.confluence:.1f}"] + sc.bonuses,
            risk_atr=(abs(float(entry) - float(sig.invalidation)) / s.atr
                      if s is not None and s.atr else None))

    def _journal_close(self, ctx: EngineContext, sc: ScoredCandidate,
                       pnl_money: float) -> None:
        o = sc.candidate.outcome
        exit_px = ctx.df["close"].to_numpy()[o.exit_index] \
            if o.exit_index < len(ctx.df) else float("nan")
        self.journal.log_close(
            alert_id=f"cfx-{sc.candidate.playbook}-"
                     f"{sc.candidate.signal_index}",
            exit_price=float(exit_px), reason=o.exit_reason,
            pnl_money=float(pnl_money), r_net=o.r_net,
            hold_bars=o.hold_bars, partial_taken=o.partial_taken)

    def _journal_shadow(self, ctx: EngineContext,
                        sc: ScoredCandidate) -> None:
        cand = sc.candidate
        self.journal.log_note(
            "shadow candidate graded (paper only — not in live P&L)",
            playbook=cand.playbook, bar=cand.signal_index,
            direction=cand.direction, base_score=cand.base_score,
            confluence=sc.confluence, r_net=cand.r_net,
            exit_reason=cand.outcome.exit_reason,
            thesis=cand.thesis, bonuses=sc.bonuses)

    def _skip(self, rep: EngineReport, ctx: EngineContext,
              sc: ScoredCandidate, reason: str, detail: str) -> None:
        rep.skips[reason] = rep.skips.get(reason, 0) + 1
        self.journal.log_skip(
            alert_id=f"cfx-{sc.candidate.playbook}-"
                     f"{sc.candidate.signal_index}",
            instrument=ctx.instrument, mode="swing",
            direction=sc.candidate.direction, score=sc.confluence,
            playbook=f"{sc.candidate.playbook} (skipped: {reason} — "
                     f"{detail})")
