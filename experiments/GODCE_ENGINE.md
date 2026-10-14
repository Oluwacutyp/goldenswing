# GoldenSwing — GOD Conviction Engine (GOD-CE-SLIM.pine port)

**Data:** GCF H1 (gold futures), 11,426 bars, 2024-10-11 → 2026-10-09 (1.99y);
DXY H1 as the macro feed. 4 bps round-trip cost via the engine's own cost
gate (mean 0.07R/trade). 1% risk per trade on a $10k account. Same data the
playbooks were measured on.

**What was ported:** the full engine from `~/workspace/your_files/GOD-CE-SLIM.pine`
(logic-identical to the 4000-line original; only dashboard visuals stripped):
26-factor confluence scoring → buyScore/sellScore 0–100, factor gate
(≥5 distinct factors), dynamic thresholds (Standard profile, 70% base +
regime/counter-bias adjustments), the 5-mode strategy router with 2-bar
dwell, all entry gates (killzone, volume, NFP caution, spike shield, bait,
lure, cost, circuit breaker), and the complete trade state machine
(Hydra 4-phase trail, TP1 1.5R / TP2 3R / TP3 runner in PVS mode, BE logic,
0.5R profit lock, conviction soft/hard invalidation). ASOC self-optimization
is built as a toggle, default OFF — it is in-sample adaptive tuning and gets
its own experiment later.

**Deliberate deviations from the pine (documented, not accidental):**
- All HTF values (`request.security` equivalents: H4 bias/impulse/OB, D1
  bias/pivots/kumo, W1 pivots) use the last **closed** HTF bar. The pine
  reads developing/repainting HTF values intrabar; the closed-bar convention
  is the honest backtest choice.
- NFP caution = first Friday of the month by calendar
  (`weekday==4 and day<=7`). The pine's `dayofweek` convention was suspect;
  this is the corrected version.
- Entries fill at the confirmed bar close; the exit ladder (TP3→SL→TP2
  priority in PVS mode) is evaluated on the entry bar too, matching the
  pine's if/else-if.

**Knobs pre-committed, not fitted:** every parameter is the pine's default
(Standard risk profile). Nothing was tuned against the backtest, in-sample
or out.

---

## Full-sample result — KILLED AS TESTED

| Metric | Value |
|---|---|
| Trades | 611 |
| Net | **−179.8R** (−180% on $10k @1% risk) |
| Gross (zero cost) | **−137.2R** — loses before costs; a fit problem, not a cost problem |
| Win rate | 34.7% |
| Profit factor | 0.31 |
| Avg R | −0.29 |
| MaxDD | 181R |

| Exit reason | n | Mean R |
|---|---|---|
| SL | 426 | −0.58 |
| BE | 181 | +0.32 |
| TP3 (runner) | 4 | +2.55 |
| Conviction invalidation | 0 | — |

**Read:** the exit profile is the tell. TP1 fires (0.75R in the dominant
MEAN-REV mode), the stop jumps to breakeven, and then the trade needs a
2.5–4.5R runner that almost never arrives on H1 (4 TP3 in 611 trades).
Winners are BE scratches (+0.32R); losers are two-thirds of a full R. The
conviction-invalidation exit never fired once — the soft/hard invalidation
layer is decorative on this timeframe. The engine is a machine for turning
small early profits into breakeven stops and letting the rest bleed.

## Walk-forward — 12m expanding train / 3m test (Phase-10 discipline)

| Fold | Window | n | Net R |
|---|---|---|---|
| 0 | 2025-10 → 2026-01 | 67 | −28.2 |
| 1 | 2026-01 → 2026-04 | 73 | −7.6 |
| 2 | 2026-04 → 2026-07 | 68 | −23.9 |
| 3 | 2026-07 → 2026-10 | 100 | −20.5 |
| **OOS total** | | **308** | **−80.2** |

**Read:** negative in all four folds, worst in folds 0 and 2. No fold, no
quarter, no regime rescues it. (GOD-CE v1 has no tunable parameters, so the
"train" half is a formality — the point is the OOS stability check, which it
fails uniformly.)

## The MAP — where it works: nowhere measured

| Session | n | Avg R | Win% |
|---|---|---|---|
| asian | 151 | −0.23 | 37.1 |
| london | 132 | −0.42 | 25.0 |
| newyork | 206 | −0.36 | 35.0 |
| off-hours | 122 | −0.13 | 41.8 |

| Regime | n | Avg R | Win% |
|---|---|---|---|
| RANGING | 342 | −0.32 | 33.6 |
| TRENDING BULL | 150 | −0.21 | 38.0 |
| TRENDING BEAR | 96 | −0.36 | 32.3 |
| REVERSAL | 20 | −0.22 | 40.0 |

| Router mode | n | Avg R | Win% |
|---|---|---|---|
| MEAN-REV | 302 | −0.23 | 34.1 |
| MOMENTUM | 180 | −0.23 | 40.0 |
| BREAKOUT | 117 | **−0.55** | 27.4 |
| POST-SPIKE | 12 | −0.45 | 41.7 |

| Side | n | Avg R | Win% |
|---|---|---|---|
| long | 338 | −0.25 | 37.9 |
| short | 273 | −0.35 | 30.8 |

Every session, every regime, every router mode, both sides, every weekday —
negative. The least-bad cells (off-hours −0.13R, TRENDING BULL −0.21R,
MOMENTUM −0.23R) are still negative with 100+ trades each; there is no
hunting ground to declare. BREAKOUT mode is the worst cell (−0.55R/trade,
27% win): the pattern-breakout entries with −10 threshold offset are
donating.

## Against our measured playbooks

| System | OOS result |
|---|---|
| fvg_midline sleeve | +18.82% (27 trades, thin) |
| trendline_retest sleeve | +14.03% (39 trades, thin) |
| sweep_reversal anchor | −0.28% walk-forward (FAILED) |
| **GOD-CE v1** | **−180% full-sample / −80.2R OOS (FAILED)** |

GOD-CE is not a marginal failure — it is an order of magnitude worse than
the anchor that already failed walk-forward.

---

## Verdict

**Killed as tested** — GC H1, Standard profile, pine defaults, 2024-10 →
2026-10. Not a candidate for demo, not a sleeve, not a confluence input in
its current form. A claim without a context is not a claim, and this engine
has no winning context on the measured data.

## What's still weak / deliberately left out

- **ASOC self-optimization**: ported as `useASOC=False` default. It is
  in-sample adaptive threshold tuning; enabling it here would be fitting.
  It gets its own experiment with a proper train/test split later.
- **Port fidelity risk**: the port was verified section-by-section against
  the slim pine (two real bugs caught and fixed: counter-bias used
  `biasScore==-3` instead of `<=-2`; walk-forward ate 43% of each window to
  warmup). The remaining risk is in the long tail of 26 scoring factors —
  any single mis-ported factor shifts scores a few points. The uniform
  negativity across all modes/sessions argues against a single-factor bug
  flipping the verdict, but a line-by-line audit against live TradingView
  alerts is still the gold standard before any resurrection attempt.
- **The HTF closed-bar convention** deviates from the pine's repainting
  reads. If the pine's edge (if any) lives in intrabar HTF information,
  this port wouldn't capture it — but that edge wouldn't be tradeable
  without lookahead anyway.
- **Ideas worth salvaging** (not the engine): the trap-intelligence layer
  (SFP / fake-breakout / beyond-FVG state machines) as *scoring inputs* to
  our existing playbooks; the priority-stack headline and setup-grade token
  for Telegram alerts (see `docs/mining/sources/godce_visuals.md`); the
  cost gate as a general entry filter.
- **Not attempted**: other symbols/timeframes (the engine is gold-tuned;
  BTC/scalp branches are dead code paths here), other risk profiles, or
  PVS-off (TP2-exit) variants — those are tuning-shaped moves and were
  deliberately not run against this data.

## Reproduce

```bash
cd ~/workspace/goldenswing
python3 -c "
from goldenswing.godce import GodceParams, run_backtest, summarize, walk_forward
P = GodceParams()
trades, df, feat = run_backtest('data/GCF_H1_2y.csv', 'data/DOLLAR_H1_2y.csv', P)
print(summarize(trades))
"
```

Module: `goldenswing/godce/` (`params`, `indicators`, `stateful`, `scoring`,
`router`, `engine`, `backtest`). Tests: `tests/test_godce.py` (12 tests).
Full repo suite: 537 passed, 1 skipped.
