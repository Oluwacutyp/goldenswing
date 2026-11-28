"""Binding probe (origination §7, rework 2026-11-27: all 33).

Replays recent real H1/M15/M5 closes through the PRODUCTION
scans + the real trigger-binding path and counts, per playbook:
armed setups, in-direction zone-touch fires, bindings, and the
binding rule that picked each.

What a "fire" is here: an H1 close landing inside an armed
setup's entry zone in the setup's direction — the zone-touch
half of the trigger machine's fire.  The trigger machine's
own POI confirmation is NOT replayed: this probe measures
BINDING given a fire at the touch, not the trigger machine's
fire rate.  The verdict handed to ``_run_signal_path`` is
built from the touching bar + the setup's own geometry
(entry = the close, stop = the setup's invalidation, target
= its first target); the binding selection, the seam, and the
order path are the production code, unmodified.  The roster
close evals run on H1 closes, so roster setups compete for
the one position alongside the families — the binding counts
are measured in the live competition.  The lifecycle is
DRIVEN, not frozen (fill-funnel lane 2026-11-27): every M15
step runs the production tick management + reconcile and
the broker simulation (pendings fill on touch, positions
close on stop/target), so the orders/fills ARE a replay
fill rate — before this, the probe never ran the tick pump,
the first order froze the one-position slot for the rest
of the window, and the probe measured "1 order in 30 days".
Two honest limits stay: the lifecycle resolves on M15 bars
(a wick inside the bar is a touch; the intrabar order of a
stop-vs-target touch in one bar is assumed stop-first) and
the wall-clock cooldowns/caps (entry-gate cooldown, scalp
holding cap) run on wall time, not replay time.

Coverage (rework 2026-11-27): ALL 33 playbooks.  The 29 on
the trigger path go through the fire / binding loop below —
20 on their H1 trigger clock (roster 12 + range 4 + vol 2 +
trend 2) and 9 on their M15 trigger clock (auction 3 +
reversion 2 + session 4); the split is the registry's
(trigger_tf_of), never H1 for all.  The two natives run
their own routes on their own clocks: qm_m5 on M5 closes
(real M5 CSV), qm_d1 on D1 closes (real D1 frame resampled
from H1) — their orders are counted from the executor, not
the fire loop.  The last
two cannot be replayed from these bars and are reported as
such, not faked: hot_scalp needs M1 closes (no M1 CSV on
disk — its live path is proven by tests/test_hot_scalp_*.py
and tests/test_completion02_connections.py) and
arrival_zone is the resting-limit path (its live path is
proven by the arrival-zone tests in
tests/test_completion02_connections.py).  The 26 Nov Hot
Scalp acceptance case is NOT in this probe: the OANDA CSVs
end 2026-10-12 — it is pinned in
tests/test_judgment_core.py::test_26_nov_pullback_sell_reads_buy.

Data: the repo's OANDA XAUUSD H1/M15/M5 CSVs (freshest real
bars on disk; the H1/M15 files end 2026-10-12, M5 starts
2026-04-01).  Window: the last 30 days of H1 closes, with
a 400-bar warmup before them.  No MT5 connection, no live
trading, demo-only (fake executor; orders never leave the
process).

Usage (from the repo root):
    python scripts/family_binding_probe.py [--days N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.execution.playbook_registry import (
    AUCTION_SPECS,
    REVERSION_SPECS,
    ROSTER_SPECS,
    SESSION_SPECS,
    TREND_SPECS,
    VOL_SPECS,
    order_route_for,
)
from goldenswing.execution.risk import gold_spec
from goldenswing.live.hierarchy import TimeframeStack

# The 29 trigger-path playbooks: every spec whose trigger is a
# same-direction fire on its own clock, routed through the
# shared signal path.  ROSTER_SPECS is 16 (12 H1 roster + the 4
# range specs, which ride the H1 roster's scan cadence) — the
# range specs are NOT added twice.
TRIGGER_PATH = ([s.name for s in ROSTER_SPECS]
                + [s.name for s in VOL_SPECS]
                + [s.name for s in TREND_SPECS]
                + [s.name for s in AUCTION_SPECS]
                + [s.name for s in REVERSION_SPECS]
                + [s.name for s in SESSION_SPECS])
TRIGGER_PATH_SET = set(TRIGGER_PATH)
ROUTE_OF = {s.name: s.route for s in (
    list(ROSTER_SPECS) + list(VOL_SPECS)
    + list(TREND_SPECS) + list(AUCTION_SPECS)
    + list(REVERSION_SPECS) + list(SESSION_SPECS))}
# Natives on their own routes (own clocks, own _run_ methods).
NATIVES = ("qm_m5", "qm_d1")
# The two playbooks no real-bars replay on disk can drive.
NOT_REPLAYABLE = {
    "hot_scalp": "needs M1 closes (no M1 CSV on disk)",
    "arrival_zone": "resting-limit path (no resting trigger in replay)",
}
ALL_PLAYBOOKS = TRIGGER_PATH + list(NATIVES) + list(NOT_REPLAYABLE)
# Back-compat names for the old family-only probe.
FAMILY = TRIGGER_PATH
FAMILY_SET = TRIGGER_PATH_SET
FAMILY_OF = ROUTE_OF


class _FakeExecutor:
    """A broker simulator (fill-funnel lane 2026-11-27).

    The old fake filled EVERY order the instant it was placed —
    market and pending alike — and never resolved anything
    after, so the first order froze the one-position slot for
    the rest of the replay and the probe measured "1 order in
    30 days".  This one models the broker: market orders fill
    at once; pending LIMIT/STOP orders REST until a replayed
    bar touches their level; positions close when a bar
    touches their stop or target (stop first when both touch
    — the intrabar order is unknown, so the loss is assumed,
    never the win).  The probe drives it once per M15 step
    via broker_step(); the engine's own _manage_tick /
    _reconcile then record the fills and closes.
    """

    def __init__(self):
        self.orders = []
        self._ticket = 9000
        self._positions = []
        self._pending = {}
        self._closed = {}

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return list(self._positions)

    def pending_orders(self, symbol=None):
        pend = list(self._pending.values())
        if symbol is None:
            return pend
        return [p for p in pend if p.symbol == symbol]

    def closing_deal(self, ticket):
        return self._closed.get(ticket)

    def place_order(self, request):
        from goldenswing.execution.executor import OrderResult, PendingOrder, Position
        self._ticket += 1
        ticket = self._ticket
        self.orders.append({
            "ticket": ticket,
            "volume": request.volume,
            "kind": request.order_kind,
            "alert_id": request.alert_id,
            "targets": list(request.targets or [])})
        if request.order_kind == "market":
            self._positions.append(Position(
                ticket=ticket, symbol=request.symbol,
                direction=request.direction, volume=request.volume,
                entry_price=request.entry_ref or request.order_price,
                sl=request.stop_loss, tp=request.take_profit,
                magic=request.magic, alert_id=request.alert_id))
        else:
            # A pending order RESTS — it is not a position until
            # the market touches its level (broker_step).
            self._pending[ticket] = PendingOrder(
                ticket=ticket, symbol=request.symbol,
                kind=request.order_kind,
                direction=request.direction,
                price=float(request.order_price),
                volume=request.volume,
                sl=request.stop_loss, tp=request.take_profit,
                magic=request.magic, alert_id=request.alert_id)
        return OrderResult(True, ticket=ticket,
                           price=request.entry_ref
                           or request.order_price,
                           volume=request.volume)

    def modify_sl(self, ticket, sl, tp=None):
        from goldenswing.execution.executor import OrderResult
        for p in self._positions:
            if p.ticket == ticket:
                p.sl = sl
                if tp is not None:
                    p.tp = tp
                break
        else:
            pend = self._pending.get(ticket)
            if pend is not None:
                pend.sl = sl
                if tp is not None:
                    pend.tp = tp
        return OrderResult(True, ticket=ticket)

    def cancel_pending(self, ticket):
        from goldenswing.execution.executor import OrderResult
        self._pending.pop(ticket, None)
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        from goldenswing.execution.executor import OrderResult
        for p in list(self._positions):
            if p.ticket != ticket:
                continue
            if volume is None or float(volume) >= float(p.volume):
                self._positions.remove(p)
            else:
                p.volume = float(p.volume) - float(volume)
            return OrderResult(True, ticket=ticket)
        return OrderResult(True, ticket=ticket)

    def broker_step(self, high, low):
        """One replayed bar through the broker: fill touched
        pendings, close SL/TP-touched positions.  Returns
        (filled, closed): filled = [(ticket, kind, price)],
        closed = [(ticket, price, reason)]."""
        from goldenswing.execution.executor import Position
        filled = []
        for ticket, pend in list(self._pending.items()):
            touched = (
                (pend.direction > 0 and pend.kind == "limit"
                 and low <= pend.price)
                or (pend.direction > 0 and pend.kind == "stop"
                    and high >= pend.price)
                or (pend.direction < 0 and pend.kind == "limit"
                    and high >= pend.price)
                or (pend.direction < 0 and pend.kind == "stop"
                    and low <= pend.price))
            if not touched:
                continue
            del self._pending[ticket]
            self._positions.append(Position(
                ticket=ticket, symbol=pend.symbol,
                direction=pend.direction, volume=pend.volume,
                entry_price=float(pend.price),
                sl=pend.sl, tp=pend.tp,
                magic=pend.magic, alert_id=pend.alert_id))
            filled.append((ticket, pend.kind, float(pend.price)))
        closed = []
        for p in list(self._positions):
            sl_hit = (p.sl is not None and (
                (p.direction > 0 and low <= float(p.sl))
                or (p.direction < 0 and high >= float(p.sl))))
            tp_hit = (p.tp is not None and (
                (p.direction > 0 and high >= float(p.tp))
                or (p.direction < 0 and low <= float(p.tp))))
            if sl_hit:
                self._positions.remove(p)
                self._closed[p.ticket] = {
                    "price": float(p.sl), "profit": 0.0,
                    "reason": "stop", "time": None}
                closed.append((p.ticket, float(p.sl), "stop"))
            elif tp_hit:
                self._positions.remove(p)
                self._closed[p.ticket] = {
                    "price": float(p.tp), "profit": 0.0,
                    "reason": "target", "time": None}
                closed.append((p.ticket, float(p.tp), "target"))
        return filled, closed


def _feed():
    return SimpleNamespace(
        account_state=lambda: {"equity": 10000.0,
                               "balance": 10000.0,
                               "free_margin": 10000.0},
        order_margin=lambda *, volume, price, direction:
            float(volume) * 100.0 * float(price) / 500.0)


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / "data" / path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule, label="left",
                       closed="left").agg({
                           "open": "first", "high": "max",
                           "low": "min", "close": "last",
                           "volume": "sum"}).dropna()


def _situation():
    """Kept for reference only: the probe does NOT stub the
    situation — it drives the production situation refresh
    (``_refresh_situation``) so the detectors read the real
    per-TF states (structure_phase, volatility, pair kinds)
    computed from the replayed frames.  A fixed neutral stub
    would suppress detections whose definition IS the
    situation (trend needs a trend phase, reversion a regime
    read).  The only stubbed context is ``_live_setup``'s bias
    (recorded as context evidence, never consulted as
    permission)."""


def _clean_state(tmp: Path) -> None:
    """Delete the runner's persisted state files (see main)."""
    import shutil

    for name in (
            "setup_lifecycle.json", "candidate_instances.json",
            "scenario_book.json", "judgment_audit_state.json",
            "judgment_thesis_state.json", "situations.jsonl",
            "situation_latest.json", "mt5_journal.jsonl",
            "mt5_risk.json", "hot_scalp_setups.db", "zones.db"):
        try:
            (tmp / name).unlink(missing_ok=True)
        except OSError:
            pass
    for name in ("deterministic", "mt5_charts"):
        try:
            shutil.rmtree(tmp / name, ignore_errors=True)
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--frame-bars", type=int, default=400)
    args = ap.parse_args()

    h1_all = _load("OANDA_XAU_USD_H1_20241001_20261012.csv")
    m15_all = _load("OANDA_XAU_USD_M15_20241001_20261012.csv")
    m5_all = _load("OANDA_XAU_USD_M5_20260401_20261012.csv")
    h4_all = _resample(h1_all, "4h")
    d1_all = _resample(h1_all, "1d")
    end = h1_all.index[-1]
    start = end - pd.Timedelta(days=args.days)
    warm = pd.Timedelta(hours=args.frame_bars)
    h1 = h1_all.loc[start - warm:end]
    m15 = m15_all.loc[start - warm:end]
    m5 = m5_all.loc[start - warm:end]
    h4 = h4_all.loc[start - warm:end]
    d1 = d1_all.loc[start - warm:end]
    # Replay steps: the M15 closes inside the window.
    steps = m15.loc[start:end]
    print(f"H1 bars: {len(h1)} (window {start} .. {end}); "
          f"M15 steps: {len(steps)}; M5 bars: {len(m5)}; "
          f"H4: {len(h4)}; D1: {len(d1)}",
          flush=True)

    tmp = REPO_ROOT / "tmp" / "family_binding_probe"
    tmp.mkdir(parents=True, exist_ok=True)
    # Fresh replay every run: the runner restores its persistent
    # state from this dir (setup lifecycle, scenario book,
    # judgment audit/thesis, situations, risk, hot-scalp store,
    # zones) — a second run over the same bars would re-detect
    # the same setup IDs and dedupe them as already-known, so
    # arming collapses (rework 2026-11-27: a reused dir armed
    # 2 setups by close 100 where a fresh dir armed 53).  The
    # output (binding_counts.json) is written at the end and is
    # not among the deleted files.
    _clean_state(tmp)
    cfg = MT5RunnerConfig(
        journal_path=tmp / "mt5_journal.jsonl",
        risk_state_path=tmp / "mt5_risk.json",
        kill_switch_path=tmp / "mt5_KILL",
        news_pause_enabled=False)
    runner = MT5Runner(cfg, feed=_feed(),
                       executor=_FakeExecutor())
    runner._send_alert = lambda *a, **k: None
    runner._build_chart = lambda *a, **k: (None, None)
    runner._live_setup = {"bias_d1": 0, "bias_h4": 0,
                          "regime": "range",
                          "h4_context": "probe (neutral bias)",
                          "size_mult": 1.0}
    # The situation keeper is deliberately NOT stubbed: the
    # first _refresh_situation call builds the real one and it
    # computes per-TF states from the replayed frames.

    armed: dict[str, set] = defaultdict(set)
    fires: Counter = Counter()
    binds: Counter = Counter()
    rules: dict[str, Counter] = defaultdict(Counter)
    orders: Counter = Counter()
    native_orders: Counter = Counter()
    stages: dict[str, Counter] = defaultdict(Counter)
    # FULL FUNNEL (fill-funnel lane 2026-11-27): every refusal
    # counted at the step where it happens, so the funnel
    # armed -> fires -> binds -> orders submitted -> filled is
    # accounted, not asserted.  Merged from the per-fire stats
    # dicts (the signal path's own counters) + the lifecycle
    # terminals + the journal's judgment-abstain reasons.
    funnel: Counter = Counter()
    abstain_reasons: Counter = Counter()
    terminal_outcomes: Counter = Counter()
    first_order: dict = {}
    # Lifecycle accumulators (fill-funnel lane 2026-11-27):
    # the per-step _manage_tick / _reconcile counters, and the
    # broker's own fill/close log.  Kept separate from the
    # fire-loop stats — the lifecycle runs on every M15 step,
    # the fire loop only on H1 closes with a touch.
    lc_stats: Counter = Counter()
    lc_filled: list = []
    lc_closed: list = []
    h1_closes = 0
    t0 = time.time()

    h1_idx = h1.index
    h4_idx = h4.index
    d1_idx = d1.index
    m5_idx = m5.index
    m15_base = int(m15.index.searchsorted(start))
    fb = args.frame_bars

    def _count_orders(before: int, counter: Counter) -> None:
        for o in runner.executor.orders[before:]:
            pb = str(o.get("alert_id", "")).split("-")[1] \
                if str(o.get("alert_id", "")).startswith("MT5-") \
                else "?"
            if pb in TRIGGER_PATH_SET or pb in NATIVES:
                counter[pb] += 1

    for i, ts in enumerate(steps.index):
        # Frames end at the last closed bar of each clock.
        h1_n = int(h1_idx.searchsorted(ts, side="right"))
        h4_n = int(h4_idx.searchsorted(ts, side="right"))
        d1_n = int(d1_idx.searchsorted(ts, side="right"))
        m5_n = int(m5_idx.searchsorted(ts, side="right"))
        m15_pos = m15_base + i
        m15_frame = m15.iloc[max(0, m15_pos - fb + 1):m15_pos + 1]
        m5_frame = m5.iloc[max(0, m5_n - fb):m5_n]
        d1_frame = d1.iloc[max(0, d1_n - fb):d1_n]
        runner._stack = TimeframeStack.from_frames({
            "H1": h1.iloc[max(0, h1_n - fb):h1_n],
            "H4": h4.iloc[max(0, h4_n - fb):h4_n],
            "D1": d1_frame,
            "M15": m15_frame,
            "M5": m5_frame})
        tick = {"mid": float(m15_frame["close"].iloc[-1]),
                "bid": float(m15_frame["close"].iloc[-1]) - 0.1,
                "ask": float(m15_frame["close"].iloc[-1]) + 0.1,
                "spread": 0.2, "time": ts}
        runner._last_tick = tick
        # Lifecycle (fill-funnel lane 2026-11-27): the broker +
        # the tick manager advance on the replayed bars.  The
        # probe previously never ran them — positions never
        # closed and pendings never filled or expired, so the
        # first order froze the one-position slot for the rest
        # of the window and the probe measured "1 order in 30
        # days".  Order mirrors production per tick: manage on
        # the close first, then the broker on the bar range
        # (wick touches the close missed), then reconcile
        # records what the broker did.
        _bar = m15_frame.iloc[-1]
        runner._manage_tick(float(_bar["close"]), lc_stats,
                            bid=tick["bid"], ask=tick["ask"])
        _filled, _closed = runner.executor.broker_step(
            float(_bar["high"]), float(_bar["low"]))
        lc_filled.extend(_filled)
        lc_closed.extend(_closed)
        runner._reconcile(float(_bar["close"]), lc_stats)
        minute = ts.minute
        is_h1 = minute == 45
        is_h4 = is_h1 and ts.hour % 4 == 3
        is_d1 = is_h1 and ts.hour == 23
        appended = ["M15"]
        if is_d1:
            appended.append("D1")
        if is_h4:
            appended.append("H4")
        if is_h1:
            appended.append("H1")
        # Production situation refresh (real per-TF states
        # from the replayed frames; also runs the book cycle
        # on H1 closes) — BEFORE the scans, as in _refresh_bars.
        runner._situation_on_closes(appended, tick)
        # Production cadence (MT5Runner._refresh_bars Pass 7 +
        # the roster close evals + the native routes).
        runner._run_auction_value_family(
            ("H1",), {}, source="live_close")
        runner._run_mean_reversion_family(
            ("M15",), {}, source="live_close")
        runner._run_session_range_family(
            ("H1",), {}, source="live_close")
        runner._run_vol_expansion_family(
            ("M15",), {}, source="live_close")
        # qm_m5 native route on its own clock (M5 closes; for
        # probe purposes it runs on the M5 closes that land on
        # an H1 close — the route still reads real M5 bars).
        if is_h1:
            _nb = len(runner.executor.orders)
            runner._run_qm_m5(m5_frame, {})
            _count_orders(_nb, native_orders)
        if is_d1:
            # qm_d1 native route on the D1 close.
            _nb = len(runner.executor.orders)
            runner._run_qm_d1(d1_frame, {})
            _count_orders(_nb, native_orders)
        if is_h1:
            h1_closes += 1
            runner._run_roster_close_evals({}, source="live_close")
            runner._run_vol_expansion_family(
                ("H1",), {}, source="live_close")
            runner._run_trend_momentum_family(
                ("H4",), {}, source="live_close")
            runner._run_range_family(
                "h1_refinement", {}, source="live_close")
        if is_h4:
            runner._run_vol_expansion_family(
                ("H4",), {}, source="live_close")
            runner._run_trend_momentum_family(
                ("D1",), {}, source="live_close")
            runner._run_range_family(
                "h4_generation", {}, source="live_close")
        # Armed snapshot (distinct setups ever armed).
        for s in runner._setup_store.active():
            if s.playbook in FAMILY_SET:
                armed[s.playbook].add(s.setup_id)
        if not is_h1:
            continue
        # --- fire check at the H1 close ---
        h1_frame = runner._stack.frames["H1"]
        close = float(h1_frame["close"].iloc[-1])
        runner._last_tick = {"mid": close, "bid": close - 0.1,
                             "ask": close + 0.1, "spread": 0.2,
                             "time": ts}
        for direction in (1, -1):
            touching = [s for s in runner._setup_store.active(
                direction=direction, armed_only=True)
                if s.playbook in FAMILY_SET
                and s.entry_zone and len(s.entry_zone) == 2
                and min(s.entry_zone) <= close
                <= max(s.entry_zone)]
            if not touching:
                continue
            for s in touching:
                fires[s.playbook] += 1
            zone = touching[0].entry_zone
            tgt = (touching[0].targets[0]
                   if touching[0].targets else
                   close + 30.0 * direction)
            verdict = {"allowed": True, "bias_d1": 0,
                       "bias_h4": 0, "regime": "range",
                       "direction": direction,
                       "zone": {"top": float(max(zone)),
                                "bottom": float(min(zone)),
                                "level": close},
                       "trigger": "bos", "entry": close,
                       "stop": float(touching[0].invalidation),
                       "target": float(tgt),
                       "risk": abs(close - float(
                           touching[0].invalidation)),
                       "entry_clock": "M30", "session": {},
                       "amd_phase": "accumulation",
                       "qm": {"status": "none"}, "volume": {}}
            before = len(runner.journal.records())
            orders_before = len(runner.executor.orders)
            stats: dict = {"signals": 0, "gated": 0}
            routing: dict = {}
            # Funnel snapshot: holding / pending state BEFORE
            # this fire — the one-position rule routes a held
            # position to layers only, and a working pending
            # refuses the next one.
            holding_before = len(runner.tick_manager.tickets())
            pending_before = len(runner._pending_orders)
            funnel["signal_path_calls"] += 1
            funnel["fires_while_holding" if holding_before
                   else "fires_while_flat"] += 1
            funnel["fires_with_pending" if pending_before
                   else "fires_without_pending"] += 1
            runner._run_signal_path(
                h1_frame, len(h1_frame) - 1, [], 1.0, verdict,
                stats, lifecycle_routing=routing)
            # Merge this fire's own counters (friction, floor,
            # sizing, gates, ineligible, thesis, duplicate,
            # invalidated, executed, pending, ...).
            for key, val in stats.items():
                if isinstance(val, (int, float)):
                    funnel[f"stats.{key}"] += val
            # Judgment takes/abstains, from the journal's own
            # judgment records (a separate record per evaluation:
            # type "judgment", judgment.decision take/abstain).
            # Abstains bucketed by the verdict's own reason text
            # (the condition that abstained).
            for r in runner.journal.records()[before:]:
                if r.get("type") != "judgment":
                    continue
                j = r.get("judgment") or {}
                if j.get("decision") == "take":
                    funnel["judgment_take_total"] += 1
                elif j.get("decision") == "abstain":
                    funnel["judgment_abstain_total"] += 1
                    reason = str(j.get("reason") or "")
                    if "invalid geometry" in reason:
                        bucket = "invalid_geometry"
                    elif "no reads measured" in reason:
                        bucket = "no_reads"
                    elif "split evenly" in reason:
                        bucket = "conflict_tie"
                    elif "fights the measured direction" in reason:
                        bucket = "fights_market"
                    elif "no reaction read confirms it" in reason:
                        bucket = "no_reaction"
                    else:
                        bucket = "other:" + reason[:60]
                    abstain_reasons[bucket] += 1
            # First order: when it landed, in what state.
            if not first_order and \
                    len(runner.executor.orders) > orders_before:
                first_order.update({
                    "h1_close_index": h1_closes,
                    "ts": str(ts),
                    "holding_before": holding_before,
                    "pending_before": pending_before,
                    "playbook": str(
                        runner.executor.orders[orders_before]
                        .get("alert_id", "")),
                    "kind": str(
                        runner.executor.orders[orders_before]
                        .get("kind", ""))})
            # Binding rules, from the journal's own records
            # (kind=trigger_binding, evidence.binding.bound).
            for r in runner.journal.records()[before:]:
                if r.get("kind") != "trigger_binding":
                    continue
                binding = (r.get("evidence") or {}).get(
                    "binding") or {}
                for cand in binding.get("bound", []) or []:
                    pb = cand.get("playbook")
                    if pb in FAMILY_SET:
                        binds[pb] += 1
                        rules[pb][cand.get("rule", "?")] += 1
            runner._apply_lifecycle_routing(routing, direction)
            for pb, res in routing.items():
                if pb not in FAMILY_SET:
                    continue
                # Terminal outcome accounting: every evaluated
                # setup's (result, reason_code, order_result).
                for term in res.get("terminals") or []:
                    terminal_outcomes[(
                        str(term.get("result") or ""),
                        str(term.get("reason_code") or ""),
                        str(term.get("order_result") or ""))] += 1
                st = runner._setup_store.get(
                    res.get("setup_id") or "")
                stages[pb][st.stage if st is not None
                          else "gone"] += 1
            for o in runner.executor.orders[orders_before:]:
                pb = str(o.get("alert_id", "")).split("-")[1] \
                    if str(o.get("alert_id", "")).startswith(
                        "MT5-") else "?"
                if pb in FAMILY_SET:
                    orders[pb] += 1
        runner._upkeep_setups(h1_frame)
        # (The book cycle rides _refresh_situation on H1
        # closes, as in production — no separate call here.)
        if h1_closes % 100 == 0:
            print(f"  {h1_closes} H1 closes, "
                  f"{sum(map(len, armed.values()))} setups armed, "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)

    # One final book cycle so the last close's fills resolve
    # their tickets onto their scenarios.
    runner._book_cycle(["H1"], str(steps.index[-1]))

    # Fills + closes, from the journal's own open/close records
    # (fill-funnel lane 2026-11-27): an "open" record is a new
    # position (market fill or pending fill — both log_open);
    # layer fills ride the held position and carry no open
    # record, so they are not new trades and are excluded here.
    # The close record carries the alert_id, not the playbook —
    # resolved from its MT5-<playbook>-<index> form.
    fills: Counter = Counter()
    closes: Counter = Counter()
    for r in runner.journal.records():
        if r.get("type") == "open":
            _pb = str(r.get("playbook") or "")
            if _pb in ALL_PLAYBOOKS:
                fills[_pb] += 1
        elif r.get("type") == "close":
            _aid = str(r.get("alert_id", ""))
            if _aid.startswith("MT5-"):
                _pb = _aid.split("-")[1]
                if _pb in ALL_PLAYBOOKS:
                    closes[_pb] += 1

    print(f"\nDone: {h1_closes} H1 closes in "
          f"{time.time() - t0:.0f}s; data ends {end}")
    rows = []
    for name in ALL_PLAYBOOKS:
        if name in TRIGGER_PATH_SET:
            route = ROUTE_OF[name]
            n_orders = orders[name]
            note = ""
        elif name in NATIVES:
            route = order_route_for(name) or f"{name}_native"
            n_orders = native_orders[name]
            note = "native route (own clock)"
        else:
            route = order_route_for(name) or name
            n_orders = 0
            note = NOT_REPLAYABLE[name]
        rows.append({
            "playbook": name,
            "route": route,
            "armed": len(armed[name]),
            "fires": fires[name],
            "bindings": binds[name],
            "rules": dict(rules[name]),
            "orders": n_orders,
            "fills": fills[name],
            "closes": closes[name],
            "stages": dict(stages[name]),
            "note": note,
        })
    print(f"{'playbook':38s} {'armed':>5s} {'fires':>5s} "
          f"{'binds':>5s} {'orders':>6s} {'fills':>5s} "
          f"{'closes':>6s}  rules / stages")
    for r in rows:
        print(f"{r['playbook']:38s} {r['armed']:5d} {r['fires']:5d} "
              f"{r['bindings']:5d} {r['orders']:6d} {r['fills']:5d} "
              f"{r['closes']:6d}  "
              f"{r['rules']} {r['stages']}"
              + (f"  [{r['note']}]" if r["note"] else ""))
    out = tmp / "binding_counts.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")
    # FULL FUNNEL (fill-funnel lane 2026-11-27).
    kind_counts: Counter = Counter()
    for o in runner.executor.orders:
        kind_counts[str(o.get("kind", "?"))] += 1
    funnel_doc = {
        "signal_path_calls": funnel["signal_path_calls"],
        "fires_while_flat": funnel["fires_while_flat"],
        "fires_while_holding": funnel["fires_while_holding"],
        "fires_with_pending": funnel["fires_with_pending"],
        "fires_without_pending": funnel["fires_without_pending"],
        "judgment_take_total": funnel["judgment_take_total"],
        "judgment_abstain_total":
            funnel["judgment_abstain_total"],
        "abstain_reasons": dict(abstain_reasons),
        "stats": {k: v for k, v in sorted(funnel.items())
                  if k.startswith("stats.")},
        "terminal_outcomes": {
            " | ".join(key): val
            for key, val in sorted(terminal_outcomes.items())},
        "executor_orders_by_kind": dict(kind_counts),
        "first_order": first_order,
        "tickets_held_at_end": len(runner.tick_manager.tickets()),
        "pending_active_at_end": len(runner._pending_orders),
        # Lifecycle (fill-funnel lane 2026-11-27): the broker +
        # tick-manager counters from the per-step lifecycle
        # driving, the broker's own fill/close log, and the
        # journal's fills/closes (new positions opened/closed).
        "lifecycle_stats": dict(lc_stats),
        "broker_fills": len(lc_filled),
        "broker_closes": len(lc_closed),
        "broker_close_reasons": dict(Counter(
            r for _, _, r in lc_closed)),
        "fills_total": sum(fills.values()),
        "closes_total": sum(closes.values()),
        "fills_by_playbook": {k: v for k, v in sorted(fills.items())
                              if v},
    }
    print("\nFUNNEL:")
    print(json.dumps(funnel_doc, indent=2))
    fout = tmp / "funnel_counts.json"
    fout.write_text(json.dumps(funnel_doc, indent=2))
    print(f"wrote {fout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
