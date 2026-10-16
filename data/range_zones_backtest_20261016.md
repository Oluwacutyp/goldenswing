# Range + Zones backtest — 2026-10-16 (bar-level, NOT tick engine)

**Headline: the range regime works (D1 flat → RANGE, not HOLD), but the
live engine would take almost nothing from it.** The H1 pattern
detectors find 38 regime-gated range signals (+1.63R, 44.7% win rate,
n=38 — thin), yet the BoundaryTracker — whose confirmation the live
confluence gate requires for sweep trades — confirmed **zero** sweeps
and **zero** breakouts across the whole window. The sweep playbook and
the tracker disagree; that is an owner decision, not a threshold
re-tune. The full tick-engine backtest is still blocked (see §5).

## 1. What actually ran

- **Harness:** `scripts/backtest_range_zones.py` (new, in this commit).
  Bar-level, causal, closed bars only. NOT the tick engine.
- **Data (real CSVs, NOT 20 bars):** OANDA XAU/USD H1 2024-10-01 →
  2026-10-11 (11,987 H1 bars), M15 47,923 bars, M5. H4 3,241 /
  D1 633 / W1 106 / MN1 25 bars resampled from H1. **Window is ~2
  years, not 3** — the OANDA CSVs span 2024-10-01→2026-10-12. MT5
  3-year data was not available in this environment.
- **Zone seed:** full frames (MN1/W1/D1/H4/H1), 13.7s, 217 active zones
  (MAJOR 203, MINOR 14; MN1 2, W1 12, D1 31, H4 54, H1 118).
- **Regime walk:** `evaluate_hierarchy(asof=bar)` per H1 bar, 117s.
  Resampled frames stamped at CLOSE (backtest convention).
- **Playbooks:** `candidates()` single scan per playbook → simulated
  fills via `simulate_signal_v2` (buffered stop, partial at first
  target ≥1R, trail after +1.5R, 72-bar timeout). Reported BOTH raw
  and gated to bars whose hierarchy regime == "range" (live gate 0b).
- **Boundary:** `BoundaryTracker` per RANGE stretch, range_def at
  stretch START (causal), bars fed interleaved in time order (M15 of
  each H1 hour, then the H1 bar), cold start (runner-faithful: fresh
  tracker counts only bars after creation).
- **Regime state:** save/load round-trip (simulated restart) OK.
- **Elapsed:** 293.5s. Machine-readable: `data/range_zones_bt_20261016_results.json`.

## 2. Regime mix (11,987 H1 bars)

| Regime | Bars | Share |
|---|---|---|
| TREND (D1 bias ≠ 0) | 7,630 | 63.6% |
| HOLD (D1 flat, no defined range) | 3,254 | 27.1% |
| RANGE (D1 flat + defined range) | 898 | 7.5% (76 stretches, avg 11.8 bars) |
| insufficient_data (D1 warmup) | 205 | 1.7% |

D1 bias: +1 on 5,231 bars, −1 on 2,399, 0 on 4,357 (flat 36.3%).
Hierarchy `allowed`: 3/11,987 bars. Dominant blocked leg: `h4_zone`
6,895 bars (57.5%) — the engine waits at H4 POIs by design; then
`d1_bias` 3,254 (flat), `h1_trigger` 1,447, `entry` 183, `data` 205.

## 3. Range playbooks — gated (live gate 0b) vs raw

Gated = signal bar's hierarchy regime == "range". `candidates()` is
regime-ungated, so raw counts overstate what the live engine takes.

| Playbook | Raw signals | Raw R | Gated trades | Gated R | Gated WR | Gated PF |
|---|---|---|---|---|---|---|
| sweep_reclaim | 118 | −7.35 | 16 | +3.50 | 50.0% | 1.43 |
| QM | 37 | +16.88 | 4 | +5.21 | 75.0% | 94.1* |
| mean_reversion | 85 | −6.78 | 17 | −7.68 | 29.4% | 0.38 |
| compression_breakout | 9 | +1.61 | 1 | +0.60 | 100%* | inf* |
| **Total** | **249** | **+4.35** | **38** | **+1.63** | **44.7%** | **1.08** |

\* n=4 / n=1 — meaningless alone, shown for completeness.

Gated totals: 38 trades, win rate 44.7%, total +1.63R, PF 1.08,
max drawdown not computed on gated subset (raw max DD 15.33R).
Exit reasons (raw): stop 141, trail 88, timeout 20. Partial taken
41/249 (16.5%). Avg hold 17.5 H1 bars.

**Weak spot: mean_reversion loses in RANGE regime** (−7.68R, 29.4%
WR, PF 0.38, max DD 9.52R on 17 gated trades). Sweep + QM carry the
gated profit. n=38 total — all of this is thin evidence, not a
verdict.

**Sweep-confirmation gap (the important finding):**
`RangeSweepReclaimPlaybook.requires_sweep_confirmation = True`, so the
live confluence gate fires it ONLY on tracker-confirmed sweeps
(`sweep_confirmed_edges`). The tracker confirmed **0** sweeps in this
window (cold-start, runner-faithful). **Live take would be ~0 sweep
trades, not the 16 gated H1-detector signals.** The H1 detector
(wick + reclaim close) and the tracker (M15 BOS + FVG) disagree.
Either the tracker is too strict or the playbook is too loose —
owner decision, not a threshold re-tune. Same for breakouts: the
tracker confirmed 0 breakouts cold-start.

## 4. Sweep vs breakout (BoundaryTracker, cold-start)

Events over 76 RANGE stretches: `breakout_pending` 11, `sweep_wick` 3,
`false_breakout` 1, `cooldown_started` 1, `sweep_expired` 1.
`sweep_confirmed` **0**, `breakout_confirmed` **0**.
False-breakout rate: **None** (undefined — 0 confirmed breakouts;
1 false with 0 confirmed is not a 100% rate).

Warm-start diagnostic (NOT runner-faithful): feeding 7 days of
history per stretch fabricates 567 `breakout_pending` / 235
`breakout_stalled` / 66 `false_breakout` — bars from before the range
existed counted against the stretch-start range. This confirms why
the runner cold-starts fresh trackers ("replaying history would
fabricate counts"). Warm numbers are NOT true sweep/breakout rates.

## 5. What is still blocked / not measured

- **Full tick-engine backtest: BLOCKED.** The tick engine's startup
  fast-forward calls each range playbook's `live_signal` once per H1
  bar; each call resamples the whole H1 prefix to H4 + full-prefix
  `find_swings`/`atr14` (O(n) per bar, O(n²) total). 2,000-tick probe
  never finished the fast-forward. Fix needed: incremental H4 / ATR
  / swings in the range path (a real refactor, not a 5-minute patch).
  This harness is the honest substitute until then.
- **Performance bugs FIXED in this commit** (zones.py): (1) `_bos`
  recomputed full-prefix swings per zone per bar — memoized per
  transaction + fast path (equivalence test added); (2) cluster rule
  quadratic outcome scans recomputing `key` (a
  `pd.Timestamp().isoformat()` per access) — precomputed sets;
  (3) `key` now lazily cached on the zone (identity fields immutable).
  Zone seed: was >148s for 2,000 H1 + 8,000 M15 bars (test bound 60s,
  FAILED); now 13.7s for the full window. Zone tests 14 passed,
  focused suite 129 passed + 1 pre-existing failure (see below).
- **Pre-existing failure (NOT caused by this change):**
  `test_live_pullback_signal.py::test_live_clock_pullback_reaches_order`
  fails identically on clean `e855ebb` (verified by stashing).
  Trigger fires but no order — likely clock/session-dependent.
  Needs a separate look.
- **Not measured (not in e855ebb / need tick engine):** tick fills,
  TP1/TP2/TP3 hit rates, opposing-zone TP adjustment, ATR-dynamic
  management, min-lot floor, footprint score, STATE 6 management,
  zone-score vs win rate (needs causal per-bar zone snapshots).
  Opposing/ATR/min-lot/footprint/STATE 6 live in the sibling
  narrative work (uncommitted in the main tree), not in `e855ebb`.
- **Not claimed:** no "green", no "ready". 38 gated trades is thin;
  the mean-reversion loss and the sweep-confirmation gap are open
  owner decisions.
