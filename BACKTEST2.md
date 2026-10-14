# GoldenSwing — Backtest Report (Phase 2)

**Data:** GC=F hourly (yfinance), 11,426 bars, 2024-10-11 04:00:00+00:00 → 2026-10-09 20:00:00+00:00 (1.99 years). Same data, same $0.45/oz cost, same 72-bar horizon, same 1% risk as Phase 1.

**What changed (the Phase-1 diagnoses, applied):**
1. Sweep reversals now require CONFIRMATION — displacement (body >= 1.0x ATR or a displacement FVG) AND a structure shift (CHoCH/BOS or trend flip) in the reversal direction, both at/after the sweep bar, within 8 bars — otherwise the setup expires.
2. Daily-bias veto: trades against the D1 trend state (daily fractal structure, EMA50 tilt while ranging) are blocked, except a confirmed sweep reversal at a major daily level.
3. Swing exits: stop buffered 0.25x ATR beyond the structural invalidation, 50% banked at the first liquidity pool (>= 1R away), remainder trails 1.0R behind the best excursion once +1.5R is reached, 72-bar time stop kept. R is measured on the buffered stop.

**Signals simulated (shadow, score ≥ 35):** 467 (Phase 1: 1836)

## Traded book @ score ≥ 60 — Phase 2 vs Phase 1

| Metric | Phase 2 | Phase 1 |
|---|---|---|
| Trades | 199 | 584 |
| Win rate | 43.7% | 25.2% |
| Profit factor | 1.16 | 0.87 |
| Total return | +18.17% | -49.82% |
| Max drawdown | 12.63% | 61.09% |
| Sharpe | 0.68 | -0.91 |
| Avg R | +0.093 | -0.101 |
| Avg hold (bars) | 19 | 11 |

## Threshold comparison (traded book, Phase 2)

| Threshold | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R |
|---|---|---|---|---|---|---|---|
| ≥50 | 224 | 43.3 | 1.14 | +16.71% | 14.40% | 0.61 | +0.078 |
| ≥60 | 199 | 43.7 | 1.16 | +18.17% | 12.63% | 0.68 | +0.093 |
| ≥70 | 147 | 44.2 | 1.20 | +16.43% | 15.58% | 0.70 | +0.113 |

## Per-slice results (shadow book — every signal standalone)

| Slice | P2 n | P2 Win% | P2 Avg R | P2 PF | P1 n | P1 Win% | P1 Avg R | P1 PF |
|---|---|---|---|---|---|---|---|---|
| All signals | 467 | 42.4 | +0.056 | 1.10 | 1836 | 26.5 | -0.086 | — |
| Sweep reversals | 49 | 49.0 | +0.026 | 1.05 | 1551 | 25.0 | -0.113 | — |
| Continuations | 418 | 41.6 | +0.060 | 1.10 | 285 | 34.7 | +0.062 | — |
| Longs | 304 | 45.1 | +0.147 | 1.27 | 949 | 28.9 | -0.026 | — |
| Shorts | 163 | 37.4 | -0.114 | 0.82 | 887 | 23.9 | -0.151 | — |
| Long continuation | 279 | 45.2 | +0.171 | 1.31 | 167 | 38.3 | +0.185 | — |

Phase-2-only shadow slices:

| Slice | n | Win% | Avg R | PF |
|---|---|---|---|---|
| Long sweep reversals | 25 | 44.0 | -0.118 | 0.79 |
| Short sweep reversals | 24 | 54.2 | +0.177 | 1.38 |
| Short continuations | 139 | 34.5 | -0.164 | 0.75 |
| Sweep reversals @ ≥60 | 34 | 50.0 | +0.083 | 1.16 |
| Continuations @ ≥60 | 303 | 43.9 | +0.126 | 1.22 |

## Traded-book slices @ ≥60 (Phase 2)

| Slice | n | Win% | Avg R | PF |
|---|---|---|---|---|
| Sweep reversals | 23 | 47.8 | +0.024 | 1.04 |
| Continuations | 176 | 43.2 | +0.102 | 1.18 |
| Longs | 125 | 44.8 | +0.158 | 1.29 |
| Shorts | 74 | 41.9 | -0.016 | 0.97 |

## Exit behaviour (traded book, Phase 2)

- Exit mix: stop 55.3%, timeout 3.5%, trail 41.2% (Phase 1: stop 73%, target 25%, timeout 2%)
- Partial banked at first pool on 20.1% of trades
- Trades that wicked through the raw invalidation but were NOT stopped out (the ATR buffer doing its job): 13

## Score discrimination (shadow book — the core concept test)

| Score band | Signals | Win% | Avg R | PF |
|---|---|---|---|---|
| 35–50 | 22 | 22.7 | -0.497 | 0.33 |
| 50–60 | 108 | 39.8 | -0.035 | 0.94 |
| 60–75 | 197 | 45.2 | +0.199 | 1.36 |
| 75–101 | 140 | 43.6 | +0.012 | 1.02 |

Phase 1 bands for reference: 35–50 −0.260R · 50–60 −0.084R · 60–75 −0.042R · 75+ −0.120R.

## Long-continuation standalone — real sleeve or bull artifact?

The one Phase-1 slice with a pulse (+0.185R), run as its OWN book (one position at a time among long continuations only), with the Phase-2 exits. Both window halves reported separately — a sleeve that only works in one half is a regime artifact until proven otherwise.

| Book | Trades | Win% | PF | Return | MaxDD | Sharpe | Avg R |
|---|---|---|---|---|---|---|---|
| ≥60 full window | 121 | 42.1 | 1.19 | +12.96% | 11.59% | 0.60 | +0.111 |
| ≥60 first half | 66 | 42.4 | 1.07 | +2.03% | 11.59% | 0.24 | +0.039 |
| ≥60 second half | 55 | 41.8 | 1.34 | +10.71% | 9.35% | 0.94 | +0.197 |
| ≥35 full window | 137 | 43.1 | 1.24 | +18.82% | 10.51% | 0.78 | +0.136 |

Shadow (all long-continuation signals, standalone): full 279 signals, avg R +0.171; first half 153, avg R +0.218; second half 126, avg R +0.114.

## Caveats

- Same single window as Phase 1: one instrument, one strongly bullish 2-year stretch for gold. Nothing here is validated out of sample.
- Daily bars are UTC-calendar aggregates of hourly GC=F bars — an approximation of the futures daily candle.
- Exit redesign changes the meaning of R vs Phase 1 (R is now measured on a wider, buffered stop), so avg-R comparisons across phases are indicative, not exact.
- Score weights remain hand-set priors, not fitted.
## Ablation — which fix did the work?

Same data, entries and exits swapped independently (traded book @ ≥60,
shadow sweep slice):

| Entries | Exits | Trades | PF | Return | Avg R | Sweep avg R |
|---|---|---|---|---|---|---|
| Phase 1 | Phase 1 | 584 | 0.87 | −49.82% | −0.101 | −0.113 (n=1551) |
| Phase 1 | Phase 2 | 528 | 0.89 | −32.82% | −0.067 | −0.120 (n=1554) |
| Phase 2 | Phase 1 | 230 | 1.05 | +5.50% | +0.035 | −0.105 (n=49) |
| Phase 2 | Phase 2 | 199 | 1.16 | +18.17% | +0.093 | +0.026 (n=49) |

Read: **the entry fixes (confirmation + veto) do most of the work** — they
turn the book positive even with the old scalp exits. Better exits on the
old entries still lose (−32.82%): you cannot exit-manage your way out of
noise entries. The exit redesign then roughly triples the return on good
entries (+5.50% → +18.17%) and lengthens avg hold 15 → 19 bars. One
interaction worth noting: confirmed sweeps stay negative under Phase-1
exits (−0.105R) and only turn positive with the swing exits — the tight
structural stop was strangling the very reversals confirmation selected.

## Confirm-window sensitivity

Headline uses 8 bars, picked as the pre-committed midpoint of the 5–10
range, not the best performer. Traded book @ ≥60 is flat across the range
(PF 1.16 at 5, 8 and 10); the sweep slice degrades as the window widens
(shadow avg R: +0.293 @ 5 bars, n=30 · +0.026 @ 8, n=49 · −0.074 @ 10,
n=61) — later confirmations are staler, exactly the Phase-1 failure mode
in miniature.

## Diagnosis (honest read)

1. **The system now makes money on this window — modestly.** +18.17% over
   two years, PF 1.16, max drawdown 12.63% (was 61%). That is a real
   turnaround from −49.82%, and the ablation says it is mostly the entry
   discipline, not exit cosmetics. It is still ONE instrument in ONE
   bullish window, in-sample for the design decisions. Not validated.
2. **Confirmation converted sweeps from the liability into a small,
   roughly break-even sleeve** (1,551 signals → 49; shadow +0.026R, book
   +0.024R). The split is odd and small-n: SHORT confirmed sweeps carry
   it (+0.177R, n=24) while LONG confirmed sweeps still lose (−0.118R,
   n=25) — the mirror image of Phase 1. Do not build a story on 49 trades.
3. **The veto did NOT fix shorts.** Shorts overall remain negative
   (shadow −0.114R), driven by short CONTINUATIONS (−0.164R, n=139) —
   trades the veto *allowed* because the daily state was ranging or
   down-tilted at the time. Fading gold intraday remains a bad idea even
   when the daily filter permits it. The ≥60 book's shorts are ~flat
   (−0.016R), so the score gate, not the veto, is what contains them.
4. **Score discrimination improved where it matters.** Bands now run
   −0.497 / −0.035 / +0.199 / +0.012 — the 60–75 band is genuinely
   positive and the low bands genuinely toxic, a much sharper separation
   than Phase 1. But the 75+ band STILL flattens instead of extending
   (+0.012R): whatever the top scores measure, it is not more edge.
   Treat ≥75 with the same suspicion as Phase 1.
5. **Long continuation: survives the standalone test, with a caveat.**
   Positive in BOTH halves as its own book (H1: +2.03%, PF 1.07, 66
   trades · H2: +10.71%, PF 1.34, 55 trades) and in shadow form in both
   halves (+0.218R / +0.114R). That is more than a one-regime artifact.
   But the first half is thin (PF 1.07 is within noise of break-even),
   the whole window is a gold bull market, and the standalone book
   (+12.96%) actually UNDERPERFORMS the full Phase-2 book (+18.17%) —
   the other slices add value rather than diluting. Verdict: a real
   sleeve, not yet a proven one; it has not seen a gold bear market.
6. **The horizon is closer to design but not there.** Avg hold 19 bars
   (was 11) against a 72-bar design horizon; 55% still exit on stop
   (was 73%), 41% on the trail. The partial only fired on 20.1% of
   trades — most trades never reach a mapped pool ≥1R away before
   resolving. The buffer demonstrably works: 13 book trades wicked
   through the raw invalidation and survived to non-stop exits.

**What remains weak:** shorts (especially short continuation), the
unexplained 75+ score flattening, tiny confirmed-sweep samples, and zero
out-of-sample evidence. Next honest steps: test on a non-bull window or
a second instrument before believing any of it; consider gating short
continuations harder or off; investigate what the 75+ band is actually
selecting for.
