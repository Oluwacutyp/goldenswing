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
are measured in the live competition, and the order counts
are the binding's downstream effect, not a live-competition
fill rate.

Coverage (rework 2026-11-27): ALL 33 playbooks.  The 29 on
the trigger path (roster 12 + range 4 + vol 2 + trend 2 +
auction 3 + reversion 2 + session 4) go through the fire /
binding loop below.  The two natives run their own routes on
their own clocks: qm_m5 on M5 closes (real M5 CSV), qm_d1 on
D1 closes (real D1 frame resampled from H1) — their orders
are counted from the executor, not the fire loop.  The last
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
    """Fill-everything executor (mirrors the test harness)."""

    def __init__(self):
        self.orders = []
        self._ticket = 9000
        self._positions = []

    def symbol_spec(self, symbol):
        return gold_spec()

    def positions(self, symbol=None):
        return list(self._positions)

    def place_order(self, request):
        from goldenswing.execution.executor import OrderResult, Position
        self._ticket += 1
        self.orders.append({
            "ticket": self._ticket,
            "volume": request.volume,
            "kind": request.order_kind,
            "alert_id": request.alert_id,
            "targets": list(request.targets or [])})
        self._positions.append(Position(
            ticket=self._ticket, symbol=request.symbol,
            direction=request.direction, volume=request.volume,
            entry_price=request.entry_ref or request.order_price,
            sl=request.stop_loss, tp=request.take_profit,
            magic=request.magic, alert_id=request.alert_id))
        return OrderResult(True, ticket=self._ticket,
                           price=request.entry_ref
                           or request.order_price,
                           volume=request.volume)

    def modify_sl(self, ticket, sl, tp=None):
        from goldenswing.execution.executor import OrderResult
        return OrderResult(True, ticket=ticket)

    def cancel_pending(self, ticket):
        from goldenswing.execution.executor import OrderResult
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket, volume=None):
        from goldenswing.execution.executor import OrderResult
        return OrderResult(True, ticket=ticket)


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
            runner._run_signal_path(
                h1_frame, len(h1_frame) - 1, [], 1.0, verdict,
                stats, lifecycle_routing=routing)
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
            "stages": dict(stages[name]),
            "note": note,
        })
    print(f"{'playbook':38s} {'armed':>5s} {'fires':>5s} "
          f"{'binds':>5s} {'orders':>6s}  rules / stages")
    for r in rows:
        print(f"{r['playbook']:38s} {r['armed']:5d} {r['fires']:5d} "
              f"{r['bindings']:5d} {r['orders']:6d}  "
              f"{r['rules']} {r['stages']}"
              + (f"  [{r['note']}]" if r["note"] else ""))
    out = tmp / "binding_counts.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
