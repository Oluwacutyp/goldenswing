"""Shared signal-view truth: the ONE model behind the PNG chart and
the HTML page.

Both renderers take the same inputs — the decision score recorded
for the signal (never the sleeve-base placeholder that lives on the
signal object), the structural invalidation AND the broker stop when
one is known — and both derive the same panel roles, the same
effective R, and the same label de-collision from this module. The
renderers differ only in medium (pixels vs DOM); they never differ
in what the trade was.

Honesty rules, enforced here so neither renderer can drift:
- a missing decision score renders as "Score unavailable", never as
  a placeholder number;
- a missing broker stop renders as "pending at placement", never as
  the structural stop relabeled;
- a panel's role comes from the placing route + entry clock, never
  from a hardcoded "H1 is the trigger" label.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_NATIVE_PLAYBOOKS = ("qm_m5", "qm_d1")


def panel_role(tf: str, *, playbook: str, entry_clock: str) -> str:
    """Role of one timeframe panel for this signal's route.

    The entry clock is the entry panel. On native routes (qm_m5 /
    qm_d1) H1 is context, not the trigger — the trigger lives on the
    entry clock. On H1-roster routes H1 is the trigger. H4/D1 are
    structure on every route; anything else is context.
    """
    if tf == entry_clock:
        return "entry"
    if playbook in _NATIVE_PLAYBOOKS:
        return "structure" if tf in ("H4", "D1") else "context"
    if tf == "H1":
        return "trigger"
    if tf in ("H4", "D1"):
        return "structure"
    return "context"


def score_text(decision_score: float | None, *,
               fallback: float | None = None) -> str:
    """Score line for a page: the decision score, or the honest gap."""
    v = decision_score if decision_score is not None else fallback
    if v is None:
        return "Score unavailable"
    return f"Score {v:.0f}"


def effective_r(*, entry: float, broker_stop: float | None,
                targets) -> float | None:
    """Reward/risk to the first target on the BROKER stop.

    None when no broker stop is known yet or no target exists — the
    effective R cannot be stated before the broker stop exists, and
    the structural distance is not a substitute for it.
    """
    if broker_stop is None or not targets:
        return None
    risk = abs(entry - broker_stop)
    if risk <= 0:
        return None
    return abs(float(targets[0]) - entry) / risk


def admit_level_bounds(lo: float, hi: float, values, *,
                       mult: float = 1.6) -> tuple[float, float]:
    """Expand ``(lo, hi)`` to admit level values NEAR the range.

    Admission is tested against the ORIGINAL bounds only. The
    renderer loops this replaces re-tested each candidate against
    the GROWING bounds, so a staircase of levels (a hot_scalp
    target ladder ascending in small steps) ratcheted the axis
    up without bound and crushed the candles into the bottom of
    the panel — the 2026-11-05 live fill charts for ticket
    540791314, where 20+ planned legs walked the axis ~300 points
    above the market. A level farther than ``mult`` x the candle
    span from the original range is drawn (it clips) but never
    admitted to the axis.
    """
    span = max(hi - lo, 1e-9)
    out_lo, out_hi = float(lo), float(hi)
    for v in values:
        if v is None:
            continue
        v = float(v)
        if lo - mult * span <= v <= hi + mult * span:
            out_lo, out_hi = min(out_lo, v), max(out_hi, v)
    return out_lo, out_hi


def chart_targets(targets, *, entry: float, direction: int,
                  cap: int = 3) -> tuple[list[tuple[int, float]], int]:
    """The targets a CHART draws, from the signal's full list.

    A hot_scalp signal carries every planned hierarchy leg (20+
    prices in two clusters on live state). Drawing each as its
    own dashed line tiles the panel into green bands and buries
    the market. The chart draws the first ``cap`` targets in
    trade order (nearest the entry first) plus — when more
    exist — the farthest leg, so the ladder's extent stays
    truthful. Returns ``(shown, total)``: shown is a list of
    ``(rank, price)`` with rank the leg's 1-based position in
    trade order, so a label can name its true rank (``T24`` is
    the ladder top, never a renumbered ``T4``).
    """
    legs = [float(t) for t in (targets or [])
            if t is not None and float(t) == float(t)]
    if direction >= 0:
        legs = sorted(t for t in legs if t > entry) or sorted(legs)
    else:
        legs = sorted((t for t in legs if t < entry), reverse=True) \
            or sorted(legs, reverse=True)
    total = len(legs)
    if total <= cap:
        return [(i + 1, t) for i, t in enumerate(legs)], total
    shown = [(i + 1, legs[i]) for i in range(cap)]
    shown.append((total, legs[-1]))
    return shown, total


# ------------------------------------------------------------------ roles
# The three alert-chart roles (owner spec, 2026-11-05): every alert
# PNG has ONE job. The studies below are the canonical per-role
# data: the PNG role renderers and the HTML signal view both read
# them, so the two mediums can never disagree about the trade.

ROLE_TRADE = "trade"
ROLE_CONTEXT = "context"
ROLE_ENTRY = "entry"
ROLE_ORDER = (ROLE_TRADE, ROLE_CONTEXT, ROLE_ENTRY)
ROLE_TITLES = {
    ROLE_TRADE: "Trade — fill overview",
    ROLE_CONTEXT: "H1 — context",
    ROLE_ENTRY: "Entry timeframe — entry precision",
}

#: The standing overlay law: at most four study families on a
#: role panel. Trade levels (entry / stops / targets / fill) are
#: the trade itself, not overlays; each study below counts the
#: families it asks the renderer to draw.
MAX_ROLE_OVERLAYS = 4


@dataclass(frozen=True)
class StopEvent:
    """One broker-confirmed stop modification, as journaled."""

    ts: float            # epoch seconds (wall clock of the apply)
    level: float
    reason: str = ""     # "profit_lock" | "breakeven" | "trail" | ...


@dataclass(frozen=True)
class PartialExit:
    """One executed partial leg, at its executed price."""

    ts: float            # epoch seconds; 0.0 when only the leg book
                         # survives (adopted positions) — the chart
                         # then lists it in the ledger, unmarked
    price: float
    volume: float
    reason: str = ""


@dataclass(frozen=True)
class LifecycleStudy:
    """Role A truth: the position, planned vs actual, kept apart.

    ``targets`` are PLANNED levels from the signal — the renderer
    labels them "(planned)" and never draws one as an exit. The
    actuals are ``fill_price``, ``stop_events``, ``partials`` and
    the ``exit_*`` fields. ``state`` is "planned" (signal only),
    "open" (filled, running) or "closed".
    """

    direction: int
    entry_ref: float                 # planned entry (signal)
    structural_stop: float           # invalidation — never relabeled
    targets: tuple = ()
    entry_zone: tuple | None = None  # (low, high) the setup was born in
    state: str = "planned"
    ticket: int | None = None
    volume: float | None = None
    fill_price: float | None = None
    fill_ts: float | None = None     # epoch seconds
    initial_broker_stop: float | None = None
    stop_events: tuple = ()          # StopEvent, chronological
    partials: tuple = ()             # PartialExit, chronological
    exit_price: float | None = None
    exit_ts: float | None = None
    exit_reason: str = ""            # display string from the close
                                     # classifier ("SL hit (profit_lock)")

    @property
    def current_broker_stop(self) -> float | None:
        if self.stop_events:
            return float(self.stop_events[-1].level)
        return self.initial_broker_stop

    def overlay_families(self) -> tuple:
        fams = ["entry_zone"] if self.entry_zone else []
        if self.stop_events:
            fams.append("stop_path")
        return tuple(fams)


@dataclass(frozen=True)
class ZoneBand:
    """One relevant structure zone for the context role."""

    kind: str              # "FVG" | "OB"
    top: float
    bottom: float
    direction: int
    created_bar: int       # bar index IN THE CONTEXT FRAME
    age_bars: int          # context-frame bars since creation


@dataclass(frozen=True)
class ContextStudy:
    """Role B truth: the structure the decision actually read."""

    swings: tuple = ()         # (bar, price, label) HH/HL/LH/LL,
                               # bar index in the context frame
    structure_event: dict | None = None  # {kind, direction, level,
                                         #  bar, age_bars}
    zones: tuple = ()          # ZoneBand — relevant only, capped
    pools: tuple = ()          # (price, "above"|"below", strength)
                               # — nearest each side only
    htf_levels: tuple = ()     # (price, source_tf, tier) cache levels
    trigger: str = ""
    bias_d1: int = 0
    bias_h4: int = 0

    def overlay_families(self) -> tuple:
        fams = []
        if self.zones:
            fams.append("zones")
        if self.pools:
            fams.append("pools")
        if self.htf_levels:
            fams.append("htf_levels")
        if self.swings:
            fams.append("swings")
        return tuple(fams)


@dataclass(frozen=True)
class EntryStudy:
    """Role C truth: how the entry was formed.

    ``overlays`` names the indicator families the strategy's own
    recorded evidence used — the renderer draws exactly those and
    no others. Formation / first-touch markers exist only when
    the caller supplies their timestamps; they are never inferred
    from the candles.
    """

    entry_zone: tuple | None = None   # (low, high)
    zone_kind: str = ""
    formation_ts: float | None = None  # epoch seconds
    first_touch_ts: float | None = None
    trigger_label: str = ""
    overlays: tuple = ()      # ("ema_pair",) and/or ("qm",)
    qm_pattern: dict | None = None
    reason_lines: tuple = ()  # recorded entry reason / reaction text

    def overlay_families(self) -> tuple:
        fams = ["entry_zone"] if self.entry_zone else []
        return tuple(fams) + tuple(self.overlays)


def lifecycle_from_events(*, direction: int, entry_ref: float,
                          structural_stop: float, targets=(),
                          entry_zone=None, state: str = "planned",
                          ticket: int | None = None,
                          volume: float | None = None,
                          fill_price: float | None = None,
                          fill_ts: float | None = None,
                          initial_broker_stop: float | None = None,
                          events=(), partials=(),
                          exit_price: float | None = None,
                          exit_ts: float | None = None,
                          exit_reason: str = "") -> LifecycleStudy:
    """Assemble a LifecycleStudy from raw lifecycle records.

    ``events`` are TickManager lifecycle dicts (``{"kind": "sl",
    "ts", "level", "reason"}`` / ``{"kind": "open", ...}``);
    ``partials`` are PartialExit or banked-leg dicts. Anything not
    recorded is left empty — never reconstructed from prices.
    """
    stops = []
    fill_t = fill_ts
    for ev in events or ():
        if not isinstance(ev, dict):
            continue
        if ev.get("kind") == "sl" and ev.get("level") is not None:
            stops.append(StopEvent(ts=float(ev.get("ts") or 0.0),
                                   level=float(ev["level"]),
                                   reason=str(ev.get("reason") or "")))
        elif ev.get("kind") == "open" and fill_t is None \
                and ev.get("ts") is not None:
            fill_t = float(ev["ts"])
    legs = []
    for p in partials or ():
        if isinstance(p, PartialExit):
            legs.append(p)
        elif isinstance(p, dict) and p.get("price") is not None:
            legs.append(PartialExit(
                ts=float(p.get("ts") or 0.0), price=float(p["price"]),
                volume=float(p.get("volume") or 0.0),
                reason=str(p.get("reason") or "")))
    stops.sort(key=lambda e: e.ts)
    legs.sort(key=lambda e: e.ts)
    return LifecycleStudy(
        direction=int(direction), entry_ref=float(entry_ref),
        structural_stop=float(structural_stop),
        targets=tuple(float(t) for t in (targets or ())),
        entry_zone=(tuple(float(x) for x in entry_zone)
                    if entry_zone else None),
        state=str(state), ticket=ticket,
        volume=(float(volume) if volume is not None else None),
        fill_price=(float(fill_price) if fill_price is not None
                    else None),
        fill_ts=fill_t,
        initial_broker_stop=(float(initial_broker_stop)
                             if initial_broker_stop is not None
                             else None),
        stop_events=tuple(stops), partials=tuple(legs),
        exit_price=(float(exit_price) if exit_price is not None
                    else None),
        exit_ts=exit_ts, exit_reason=str(exit_reason or ""))


def context_study_from_state(state, *, entry: float, trigger: str = "",
                             bias_d1: int = 0, bias_h4: int = 0,
                             htf_levels=(),
                             max_zones: int = 4) -> ContextStudy:
    """Build the ContextStudy from a MarketState at the signal bar.

    Relevance is the whole point: of everything the state knows,
    the context panel shows the swings in frame, the LAST
    structure event, the nearest active FVG/OB zones (capped —
    a live accumulated state carries more zones than any panel
    can honestly show), and the nearest liquidity pool on each
    side. Bar indices stay in the state's own frame; the
    renderer offsets them into its window. Swing/zone/event
    indices are causal by construction — the state was built
    only from bars up to its own index, and swing points carry
    their confirmation lag in the reader, never in the chart.
    """
    swings: list[tuple[int, float, str]] = []

    def _labs(points, kind):
        out = []
        prev = None
        for p in sorted(points, key=lambda q: q.index):
            if prev is None:
                lab = "H" if kind == "high" else "L"
            elif kind == "high":
                lab = "HH" if p.price > prev else "LH"
            else:
                lab = "HL" if p.price > prev else "LL"
            out.append((int(p.index), float(p.price), lab))
            prev = p.price
        return out

    swings += _labs(getattr(state, "recent_swing_highs", []) or [],
                    "high")
    swings += _labs(getattr(state, "recent_swing_lows", []) or [],
                    "low")
    ev = getattr(state, "last_structure_event", None)
    event = None
    if ev is not None:
        event = {"kind": str(ev.kind), "direction": int(ev.direction),
                 "level": float(ev.level), "bar": int(ev.index),
                 "age_bars": (int(getattr(state,
                                         "structure_event_age", 0) or 0))}
    zones: list[ZoneBand] = []
    for g in getattr(state, "active_fvgs", []) or []:
        zones.append(ZoneBand(
            kind="FVG", top=float(g.top), bottom=float(g.bottom),
            direction=int(g.direction), created_bar=int(g.created_at),
            age_bars=int(state.index) - int(g.created_at)))
    for ob in getattr(state, "active_obs", []) or []:
        zones.append(ZoneBand(
            kind="OB", top=float(ob.top), bottom=float(ob.bottom),
            direction=int(ob.direction),
            created_bar=int(ob.created_at),
            age_bars=int(state.index) - int(ob.created_at)))
    zones.sort(key=lambda z: abs(0.5 * (z.top + z.bottom) - entry))
    zones = zones[:max_zones]
    pools: list[tuple[float, str, int]] = []
    above = getattr(state, "pools_above", []) or []
    below = getattr(state, "pools_below", []) or []
    if above:
        p = min(above, key=lambda q: q.price)
        pools.append((float(p.price), "above", int(p.strength)))
    if below:
        p = max(below, key=lambda q: q.price)
        pools.append((float(p.price), "below", int(p.strength)))
    return ContextStudy(
        swings=tuple(swings), structure_event=event,
        zones=tuple(zones), pools=tuple(pools),
        htf_levels=tuple((float(lvl), str(tf), str(tier))
                         for lvl, tf, tier in htf_levels),
        trigger=str(trigger or ""), bias_d1=int(bias_d1),
        bias_h4=int(bias_h4))


def label_spec(value: float, text: str, color: str, *, bold: bool = False,
               size: float = 9.0) -> dict:
    """One right-edge label spec — the renderer's shared currency.

    Every label a role panel draws at the price edge is one of
    these dicts, and ALL of them go through the single
    de-collision pass, so no label set can land on another (the
    whole-surface law). Tests assert on these lists directly.
    """
    return {"value": float(value), "text": text, "color": color,
            "bold": bool(bold), "size": float(size)}


def trade_label_specs(lc: LifecycleStudy, *, fmt) -> list[dict]:
    """Role A right-edge labels: planned vs actual, kept distinct.

    Planned targets carry "(planned)" and the ladder-aware ranks
    from :func:`chart_targets`; the structural stop and the
    broker stops are separate labels with separate names, always.
    """
    specs: list[dict] = []
    d = lc.direction
    if lc.fill_price is not None:
        specs.append(label_spec(
            lc.fill_price, f" FILL {fmt(lc.fill_price)}", "#7e57c2",
            bold=True))
    else:
        specs.append(label_spec(
            lc.entry_ref, f" ENTRY {fmt(lc.entry_ref)} (planned)",
            "#b8860b", bold=True))
    if lc.initial_broker_stop is not None:
        specs.append(label_spec(
            lc.initial_broker_stop,
            f" BROKER SL {fmt(lc.initial_broker_stop)} (initial)",
            "#e65100", bold=True))
    cur = lc.current_broker_stop
    if cur is not None and lc.stop_events:
        specs.append(label_spec(
            cur, f" BROKER SL {fmt(cur)} (now)", "#ff7043", bold=True))
    specs.append(label_spec(
        lc.structural_stop,
        f" STOP {fmt(lc.structural_stop)} (structural)", "#d32f2f",
        bold=True))
    shown, _total = chart_targets(lc.targets, entry=lc.entry_ref,
                                  direction=d)
    for rank, t in shown:
        specs.append(label_spec(
            t, f" T{rank} {fmt(t)} (planned)", "#2e7d32"))
    for p in lc.partials:
        specs.append(label_spec(
            p.price,
            f" PARTIAL {p.volume:g} @ {fmt(p.price)}", "#7e57c2"))
    if lc.exit_price is not None:
        specs.append(label_spec(
            lc.exit_price, f" EXIT {fmt(lc.exit_price)}", "#37474f",
            bold=True))
    return specs


def decollide_label_ys(values_px: list[float], *, min_gap: float,
                       top: float | None = None,
                       bottom: float | None = None) -> list[float]:
    """Push stacked label positions apart to at least ``min_gap`` px.

    Positions are pixel coordinates on one axis (either direction
    works — the function only preserves order and enforces gaps).
    Returns positions in the ORIGINAL order. A forward pass pushes
    each label above the previous one; if ``top`` is given and the
    stack overflows it, a backward pass pulls the stack down from
    the top; if ``bottom`` is then violated, a final forward pass
    re-anchors at the bottom (the stack only overflows both bounds
    when more labels than space exist, which the callers avoid by
    capping targets).
    """
    n = len(values_px)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values_px[i])
    adj = [float(v) for v in values_px]
    for k in range(1, n):
        prev, cur = order[k - 1], order[k]
        if adj[cur] < adj[prev] + min_gap:
            adj[cur] = adj[prev] + min_gap
    if top is not None and adj[order[-1]] > top:
        adj[order[-1]] = top
        for k in range(n - 2, -1, -1):
            cur, nxt = order[k], order[k + 1]
            if adj[cur] > adj[nxt] - min_gap:
                adj[cur] = adj[nxt] - min_gap
    if bottom is not None and adj[order[0]] < bottom:
        adj[order[0]] = bottom
        for k in range(1, n):
            prev, cur = order[k - 1], order[k]
            if adj[cur] < adj[prev] + min_gap:
                adj[cur] = adj[prev] + min_gap
    return adj
