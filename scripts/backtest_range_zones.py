"""Bar-level backtest for the RANGE regime + structural zones (e855ebb).

WHAT THIS IS: a causal, bar-level harness that walks the OANDA H1
bars once and measures what the range-regime code in ``e855ebb``
actually does.  It is NOT the full tick engine
(``scripts/backtest_full_engine.py``): the tick engine cannot
complete on the range code path — its startup fast-forward calls
each range playbook's ``live_signal`` once per H1 bar, and each
call resamples the whole H1 prefix to H4 plus a full-prefix
``find_swings``/``atr14`` (O(n) per bar, O(n^2) total; ~12k bars).
That is a real performance bug to fix (incremental H4/ATR/swings
in the range path), not a result to fake.  The two O(n^2) bugs in
``zones.py`` (per-call ``find_swings`` in ``_bos``; quadratic
outcome scans in the cluster rule) ARE fixed in this tree, and
this harness verifies the zone seed is fast.

WHAT IT MEASURES (all causal, closed bars only):
- regime mix: TREND / RANGE / HOLD bars from
  ``evaluate_hierarchy`` (D1 flat -> RANGE when a range is defined).
  ``live_setup`` returning None is ARMING (containment / no H4 POI /
  ...), not regime — it is NOT counted as HOLD here.
- range playbook signals + bar-level simulated fills
  (``candidates()`` -> ``simulate_signal_v2``: buffered stop,
  partial at first target >= 1R, trail after +1.5R, 72-bar timeout),
  reported BOTH raw (candidates() is regime-ungated) AND gated to
  signal bars whose hierarchy regime == "range" (live gate 0b).
- sweep vs breakout: ``BoundaryTracker`` events over each RANGE
  stretch (sweep confirmed / breakout confirmed / false breakout).
- zone store: seed time, zone counts by TF/tier/status.
- persisted regime state: save/load round-trip (simulated restart).

WHAT IT DOES NOT MEASURE (not in e855ebb / needs the tick engine):
tick fills, TP1/TP2/TP3 hit rates, opposing-zone TP adjustment,
ATR-dynamic management, min-lot floor, footprint score, STATE 6
management — those live in the sibling narrative work (uncommitted
in the main tree) or need the tick engine.  They are listed as
unmeasured, not estimated.

Usage:
  cd ~/workspace/goldenswing && .venv/bin/python \
      scripts/backtest_range_zones.py [--out DIR] [--max-bars N]
Paper only. No live trading, no network.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "bt_full", str(Path(__file__).resolve().parent / "backtest_full_engine.py"))
bt_full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt_full)

from goldenswing.live.hierarchy import evaluate_hierarchy  # noqa: E402
from goldenswing.live.range import (  # noqa: E402
    BoundaryTracker, detect_range)
from goldenswing.live.regime_state import (  # noqa: E402
    RegimeState, load_regime_state, save_regime_state)
from goldenswing.live.zones import ZoneStore  # noqa: E402
from goldenswing.playbooks.base import EngineContext  # noqa: E402
from goldenswing.playbooks.range_regime import (  # noqa: E402
    RangeCompressionBreakoutPlaybook, RangeMeanReversionPlaybook,
    RangeQMPlaybook, RangeSweepReclaimPlaybook)


def _r_stats(rs: list[float]) -> dict:
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    curve, peak, max_dd = 0.0, 0.0, 0.0
    for r in rs:
        curve += r
        peak = max(peak, curve)
        max_dd = max(max_dd, peak - curve)
    return {
        "trades": len(rs),
        "win_rate": round(len(wins) / len(rs), 4) if rs else 0.0,
        "total_r": round(sum(rs), 2),
        "avg_r": round(sum(rs) / len(rs), 3) if rs else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2)
        if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        "max_drawdown_r": round(max_dd, 2),
        "gross_win_r": round(gross_win, 2),
        "gross_loss_r": round(-gross_loss, 2),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-bars", type=int, default=None,
                    help="limit H1 bars walked (debug only)")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path("data/reports/range_zones_bt")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    missing = bt_full.missing_backtest_data()
    if missing:
        print(f"missing backtest data: {missing}", file=sys.stderr)
        return 2

    h1 = bt_full._with_feed_cols(bt_full._load_csv(
        bt_full.DATA_DIR / "OANDA_XAU_USD_H1_20241001_20261012.csv"))
    m15 = bt_full._with_feed_cols(bt_full._load_csv(
        bt_full.DATA_DIR / "OANDA_XAU_USD_M15_20241001_20261012.csv"))
    m5 = bt_full._with_feed_cols(bt_full._load_csv(
        bt_full.DATA_DIR / "OANDA_XAU_USD_M5_20260401_20261012.csv"))
    frames = {
        "H1": h1,
        "H4": bt_full._resample(h1, "H4"),
        "D1": bt_full._resample(h1, "D1"),
        "W1": bt_full._resample(h1, "W1"),
        "MN1": bt_full._resample(h1, "MN1"),
        "M15": m15,
        "M30": bt_full._resample(m15, "M30"),
        "M5": m5,
    }
    n_h1 = len(h1) if args.max_bars is None else min(args.max_bars, len(h1))
    print(f"window: {h1.index[0]} -> {h1.index[-1]} "
          f"| H1 bars walked: {n_h1} | H4={len(frames['H4'])} "
          f"D1={len(frames['D1'])} W1={len(frames['W1'])} "
          f"MN1={len(frames['MN1'])} M15={len(m15)}", flush=True)

    # --- zone store seed (timed) ---
    t_seed = time.monotonic()
    store = ZoneStore(str(out / "zones.db"))
    store.seed({k: frames[k] for k in ("MN1", "W1", "D1", "H4", "H1")})
    seed_s = time.monotonic() - t_seed
    zones = store.active_zones()
    zone_counts = {
        "active": len(zones),
        "by_tf": dict(Counter(z.tf for z in zones)),
        "by_tier": dict(Counter(z.tier for z in zones)),
        "by_status": dict(Counter(z.status for z in zones)),
    }
    print(f"zone seed: {seed_s:.1f}s, active={len(zones)}, "
          f"by_tier={zone_counts['by_tier']}", flush=True)
    store.close()

    # --- regime walk: evaluate_hierarchy per H1 bar (causal, asof) ---
    # live_setup() returns None for "no armed setup" (containment /
    # no H4 POI / ...) — that is ARMING, not regime.  Counting None
    # as HOLD was a harness bug in run 1.  evaluate_hierarchy is
    # the regime leg: regime = trend (D1 bias != 0) / range (D1
    # flat + defined range) / hold (D1 flat + no defined range).
    # Resampled frames are stamped at their CLOSE (backtest
    # convention); OANDA frames at their OPEN.
    stamps = {"H4": "close", "D1": "close", "W1": "close",
              "MN1": "close", "M30": "close"}
    walk_frames = {tf: frames[tf] for tf in
                   ("H1", "H4", "D1", "W1", "MN1", "M30")}
    regime_counts: Counter = Counter()
    regime_by_bar: list[str] = []
    allowed_counts: Counter = Counter()
    blocked_counts: Counter = Counter()
    bias_d1_counts: Counter = Counter()
    range_stretches: list[tuple[int, int]] = []
    cur_start: int | None = None
    t_walk = time.monotonic()
    for i in range(n_h1):
        try:
            verdict = evaluate_hierarchy(
                walk_frames, asof=h1.index[i], stamps=stamps)
        except Exception as exc:  # noqa: BLE001 - count, don't crash
            regime_counts["evaluate_error"] += 1
            if regime_counts["evaluate_error"] <= 3:
                print(f"evaluate_hierarchy error at bar {i}: {exc}",
                      flush=True)
            verdict = {"regime": "error", "allowed": False,
                       "blocked_leg": "exception", "bias_d1": 0}
        regime = verdict.get("regime", "hold")
        # Early return (insufficient closed bars) leaves the
        # verdict's DEFAULT regime "trend" — not an evaluation.
        # Count it separately, never as trend.
        legs = verdict.get("legs", {}) or {}
        if not (legs.get("data") or {}).get("passed", False):
            regime = "insufficient_data"
        regime_counts[regime] += 1
        regime_by_bar.append(regime)
        allowed_counts[bool(verdict.get("allowed"))] += 1
        blocked_counts[str(verdict.get("blocked_leg", ""))] += 1
        bias_d1_counts[int(verdict.get("bias_d1", 0))] += 1
        if regime == "range":
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                range_stretches.append((cur_start, i - 1))
                cur_start = None
        if (i + 1) % 2000 == 0:
            print(f"  walked {i + 1}/{n_h1} bars "
                  f"({time.monotonic() - t_walk:.0f}s) regimes="
                  f"{dict(regime_counts)}", flush=True)
    if cur_start is not None:
        range_stretches.append((cur_start, n_h1 - 1))
    walk_s = time.monotonic() - t_walk
    print(f"regime walk: {walk_s:.1f}s regimes={dict(regime_counts)} "
          f"range_stretches={len(range_stretches)} "
          f"allowed={dict(allowed_counts)} "
          f"bias_d1={dict(bias_d1_counts)}", flush=True)

    # --- range playbook signals + simulated fills (single scan) ---
    playbooks = [RangeSweepReclaimPlaybook(), RangeQMPlaybook(),
                 RangeMeanReversionPlaybook(),
                 RangeCompressionBreakoutPlaybook()]
    ctx = EngineContext(df=h1.iloc[:n_h1], states=[], years=2.0)
    per_pb: dict[str, dict] = {}
    all_rs: list[float] = []
    exit_reasons: Counter = Counter()
    partials = 0
    hold_bars: list[int] = []
    for pb in playbooks:
        t_pb = time.monotonic()
        cands = pb.candidates(ctx)
        rs = []
        gated_rs = []
        for c in cands:
            oc = c.outcome
            if oc is None:
                continue
            rs.append(float(oc.r_net))
            all_rs.append(float(oc.r_net))
            exit_reasons[str(oc.exit_reason)] += 1
            partials += 1 if getattr(oc, "partial_taken", False) else 0
            hold_bars.append(int(oc.hold_bars))
            # Live gate 0b: range playbooks fire ONLY when the
            # hierarchy verdict regime == "range".  candidates()
            # does not apply that gate, so split raw vs gated here.
            j = int(getattr(getattr(oc, "signal", None), "index", -1))
            if 0 <= j < len(regime_by_bar) \
                    and regime_by_bar[j] == "range":
                gated_rs.append(float(oc.r_net))
        per_pb[pb.name] = {
            "signals": len(cands), "fills": len(rs),
            **_r_stats(rs),
            "exit_reasons": dict(Counter(
                str(c.outcome.exit_reason) for c in cands
                if c.outcome is not None)),
            "partial_taken": sum(1 for c in cands if c.outcome is not None
                                 and getattr(c.outcome, "partial_taken", False)),
            # Regime-gated: what the live engine could actually take.
            "gated_signals": len(gated_rs),
            "gated": _r_stats(gated_rs),
            "scan_s": round(time.monotonic() - t_pb, 1),
        }
        print(f"  {pb.name}: signals={len(cands)} fills={len(rs)} "
              f"total_r={per_pb[pb.name]['total_r']} "
              f"win_rate={per_pb[pb.name]['win_rate']}", flush=True)

    # --- sweep vs breakout: BoundaryTracker over each RANGE stretch ---
    # Causal range_def: H1/H4 sliced to the stretch START (not the
    # stretch end — run 2 used the end = lookahead).  Bars are fed
    # INTERLEAVED in time order (M15 bars of each H1 hour, then the
    # H1 bar) — feeding all M15 first then all H1 broke the
    # tracker's state machine and produced zero events in run 2.
    events: Counter = Counter()
    false_breakouts = 0
    breakouts = 0
    sweeps = 0
    for (a, b) in range_stretches:
        h4_at_start = frames["H4"][frames["H4"].index <= h1.index[a]]
        try:
            rng = detect_range(None, h4_at_start, h1.iloc[:a + 1])
        except Exception:  # noqa: BLE001 - stretch skipped
            continue
        if rng is None:
            continue
        tracker = BoundaryTracker(rng)
        try:
            for i in range(a, b + 1):
                h1_bar = h1.iloc[i:i + 1]
                # M15 bars of THIS H1 hour (stamped at open), fed
                # before the H1 bar closes — time-correct order.
                h1_open = h1.index[i]
                m15_chunk = m15[(m15.index >= h1_open)
                                & (m15.index < h1_open
                                   + pd.Timedelta(minutes=60))]
                evs = tracker.update(h1_bar, m15_chunk)
                for e in evs:
                    kind = str(getattr(e, "kind", e))
                    events[kind] += 1
                    if kind == "false_breakout":
                        false_breakouts += 1
                    elif kind == "breakout_confirmed":
                        breakouts += 1
                    elif kind == "sweep_confirmed":
                        sweeps += 1
        except Exception as exc:  # noqa: BLE001 - stretch skipped
            print(f"tracker error stretch {a}-{b}: {exc}", flush=True)
            continue
    # Diagnostic (NOT runner-faithful): same trackers, but warm-started
    # with 7 days of H1/M15 history before each stretch.  The runner
    # cold-starts fresh trackers (cursors at latest bars); this pass
    # measures how much of the zero-confirmation result is the
    # cold-start rule vs the confirmation rules themselves.
    warm_events: Counter = Counter()
    warm_sweeps = warm_breakouts = warm_false = 0
    for (a, b) in range_stretches:
        h4_at_start = frames["H4"][frames["H4"].index <= h1.index[a]]
        try:
            rng = detect_range(None, h4_at_start, h1.iloc[:a + 1])
        except Exception:  # noqa: BLE001
            continue
        if rng is None:
            continue
        tracker = BoundaryTracker(rng)
        try:
            warm_lo = h1.index[a] - pd.Timedelta(days=7)
            warm_h1 = h1[(h1.index >= warm_lo) & (h1.index <= h1.index[b])]
            warm_m15 = m15[(m15.index >= warm_lo)
                           & (m15.index <= h1.index[b])]
            evs = tracker.update(warm_h1, warm_m15)
            for e in evs:
                kind = str(getattr(e, "kind", e))
                warm_events[kind] += 1
                if kind == "false_breakout":
                    warm_false += 1
                elif kind == "breakout_confirmed":
                    warm_breakouts += 1
                elif kind == "sweep_confirmed":
                    warm_sweeps += 1
        except Exception:  # noqa: BLE001 - stretch skipped
            continue
    print(f"warm-start boundary events (diagnostic): "
          f"{dict(warm_events)}", flush=True)

    # --- persisted regime state: save/load round-trip (sim restart) ---
    st = RegimeState(regime="range", lock_remaining_h1=2,
                     breakout_confirm_count=1, breakout_direction=1,
                     breakout_edge=4200.0,
                     last_range_high=4220.0, last_range_low=4100.0)
    state_path = out / "regime_state_roundtrip.json"
    save_regime_state(state_path, st)
    st2, status, _msg = load_regime_state(state_path)
    state_ok = (status == "ok" and st2 is not None
                and st2.regime == "range"
                and st2.lock_remaining_h1 == 2
                and st2.breakout_confirm_count == 1
                and st2.breakout_edge == 4200.0)
    print(f"regime state round-trip: {'OK' if state_ok else 'FAIL'}",
          flush=True)

    report = {
        "window": {"start": str(h1.index[0]), "end": str(h1.index[-1]),
                   "h1_bars_walked": n_h1,
                   "h4_bars": len(frames["H4"]),
                   "d1_bars": len(frames["D1"]),
                   "w1_bars": len(frames["W1"]),
                   "mn1_bars": len(frames["MN1"])},
        "harness": "bar-level (NOT tick engine). See module docstring.",
        "zone_seed_s": round(seed_s, 1),
        "zone_counts": zone_counts,
        "regime_walk_s": round(walk_s, 1),
        "regime_bars": dict(regime_counts),
        "bias_d1_bars": {str(k): v for k, v in bias_d1_counts.items()},
        "hierarchy_allowed_bars": dict(allowed_counts),
        "hierarchy_blocked_leg": dict(blocked_counts),
        "range_stretches": len(range_stretches),
        "range_stretch_bars": sum(b - a + 1 for a, b in range_stretches),
        "per_playbook": per_pb,
        "all_signals": _r_stats(all_rs),
        "all_signals_gated": {
            "note": "gated = signal bar regime == range (live gate 0b); "
                    "see per_playbook[].gated for per-playbook gated stats",
            "gated_trades": sum(p["gated_signals"]
                                for p in per_pb.values()),
            "gated_total_r": round(sum(p["gated"]["total_r"]
                                       for p in per_pb.values()), 2),
            "gated_win_rate": round(
                sum(p["gated"]["win_rate"] * p["gated"]["trades"]
                    for p in per_pb.values())
                / max(1, sum(p["gated"]["trades"]
                             for p in per_pb.values())), 4),
        },
        "exit_reasons": dict(exit_reasons),
        "partial_taken": partials,
        "avg_hold_bars": round(sum(hold_bars) / len(hold_bars), 1)
        if hold_bars else 0.0,
        "boundary_events": dict(events),
        "boundary_events_warm_start_diagnostic": dict(warm_events),
        "warm_sweep_confirmed": warm_sweeps,
        "warm_breakout_confirmed": warm_breakouts,
        "warm_false_breakout": warm_false,
        "sweep_events": sweeps,
        "breakout_events": breakouts,
        "false_breakout_events": false_breakouts,
        "false_breakout_rate": (
            round(false_breakouts / (breakouts + false_breakouts), 4)
            if breakouts > 0 else None),
        "false_breakout_rate_note": (
            "None when zero breakouts were CONFIRMED (rate undefined); "
            "1 false breakout with 0 confirmed is not a 100% rate."),
        "regime_state_roundtrip_ok": state_ok,
        "not_measured": ["tick fills", "TP1/TP2/TP3 hit rates",
                         "opposing-zone TP adjustment",
                         "ATR-dynamic management", "min-lot floor",
                         "footprint score", "STATE 6 management"],
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
    (out / "results.json").write_text(json.dumps(report, indent=2,
                                                 default=str))
    print(json.dumps(report, indent=2, default=str)[:4000], flush=True)
    print(f"report: {out / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
