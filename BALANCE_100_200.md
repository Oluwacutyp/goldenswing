# Exness Backtests: $100 and $200 vs $12

Backtest only. $12 demo is UNCHANGED — this is the next phase, not a mid-run change.
Engine: `goldenswing/backtest/exness12.py`, base variant, hard veto ON.
Leverage 1:2000, spread $0.20/oz, commission $0, slippage $0 assumed.
Fixed lot per run. No compounding of size — lot stays fixed, balance compounds.

Data windows: Swing = full 2y H1 (2024-10-01 → 2026-10-12), D1/H4/W1 resampled from H1.
Intraday = full 2y, trigger M15 CSV. Scalp = **Apr–Oct 2026 subset ONLY** (M5 data starts 2026-04-01).
Raw numbers: `BALANCE_100_200_results.json` (18 runs: 3 presets × 3 balances × 2 lots).

## Blow-up room: adverse $ move that kills a FRESH account

Adverse move to blow up = balance / lot_oz − $0.20 spread. Margin at $4,000 gold: 0.01 lot = $2.00, 0.02 lot = $4.00.

| Balance | 0.01 lot room | 0.02 lot room |
|---|---:|---:|
| $12 | **$11.80** | $5.80 |
| $100 | **$99.80** (8.5x $12) | $49.80 (4.2x) |
| $200 | **$199.80** (16.9x) | $99.80 (8.5x) |

$100 at 0.01 has the same blow-up room as $200 at 0.02. Lot scaling eats the buffer exactly as fast as it multiplies profit.

## Swing (D1/H4/H1, full 2y)

| Balance | Lot | Trades | Win% | Final $ | Peak $ | Total $ | MaxDD $ | MaxDD % | PF | Blow-ups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $12 | 0.01 | 87 | 74.7 | 468.45 | 483.53 | +456.45 | 107.84 | 22.3% | 2.01 | 0 |
| $12 | 0.02 | 87 | 74.7 | 924.91 | 955.06 | +912.91 | 215.67 | 22.6% | 2.01 | 0 |
| $100 | 0.01 | 87 | 74.7 | 556.45 | 571.53 | +456.45 | 107.84 | 18.9% | 2.01 | 0 |
| $100 | 0.02 | 87 | 74.7 | 1,012.91 | 1,043.06 | +912.91 | 215.67 | 20.7% | 2.01 | 0 |
| $200 | 0.01 | 87 | 74.7 | 656.45 | 671.53 | +456.45 | 107.84 | 16.1% | 2.01 | 0 |
| $200 | 0.02 | 87 | 74.7 | 1,112.91 | 1,143.06 | +912.91 | 215.67 | 18.9% | 2.01 | 0 |

Swing veto: blocked 912 setups, 12 blocked trades, blocked PnL −$107.56 at 0.01 (−$215.12 at 0.02), blocked win rate 41.7%, 0 blocked blow-ups.
Long +$366.14 / short +$90.32 at 0.01 (doubled at 0.02). Worst single loss −$40.94 at 0.01 (−$81.87 at 0.02). Max losing streak survived: 3.

**Read:** At fixed 0.01 lot, $12 / $100 / $200 give IDENTICAL trades and IDENTICAL $ PnL (+$456.45). Balance changes survival only, not edge.
At $12, the worst single Swing loss (−$40.94) is 3.4x the whole account — survival at $12 depended on never taking that loss at $12.
At $100 it is 41% of starting balance; at $200 it is 20%.

## Intraday (H4/H1/M15, full 2y)

| Balance | Lot | Trades | Win% | Final $ | Peak $ | Total $ | MaxDD $ | MaxDD % | PF | Blow-ups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $12 | 0.01 | 14 | 50.0 | 0.00 | 16.79 | −12.00 | 16.79 | 100% | 0.58 | 1 |
| $12 | 0.02 | 1 | 0.0 | 0.00 | 12.00 | −12.00 | 12.00 | 100% | 0.00 | 1 |
| $100 | 0.01 | 277 | 61.4 | 361.99 | 524.37 | +261.99 | 247.61 | 47.2% | 1.26 | 0 |
| $100 | 0.02 | 80 | 51.2 | 0.00 | 109.58 | −100.00 | 109.58 | 100% | 0.75 | 1 |
| $200 | 0.01 | 277 | 61.4 | 461.99 | 624.37 | +261.99 | 247.61 | 39.7% | 1.26 | 0 |
| $200 | 0.02 | 277 | 61.4 | 723.98 | 1,048.75 | +523.98 | 495.22 | 47.2% | 1.26 | 0 |

Intraday $100/0.01 veto: blocked 5,784 setups, 112 blocked trades, blocked PnL +$376.69, blocked win rate 71.4%, 0 blocked blow-ups.
Intraday $100/0.02 veto: blocked 582 setups, 10 blocked trades, blocked PnL +$48.49, blocked win rate 90.0%.
Long +$210.57 / short +$51.42 at 0.01 ($100 and $200 identical paths). Worst single loss −$142.00 at 0.01 (−$284.00 at 0.02).
Max losing streak survived: 8.

**Read:** $12 Intraday died at 14 trades (censored — not the full strategy). At $100/0.01 it survives to 277 trades and profits, but ugly:
MaxDD 47.2% of peak, one −$142 single loss (after the balance had grown past $100 — at a $100 starting balance that loss size would have killed a fresh account).
**$100 + 0.02 lot BLOWS Intraday** (80 trades, −$100). Scaling lot at $100 kills Intraday.
At $200, 0.02 survives (277 trades, +$523.98) but MaxDD $495.22 (47.2%) and worst loss −$284.

## Scalp (H1/M15/M5, Apr–Oct 2026 subset ONLY)

| Balance | Lot | Trades | Win% | Final $ | Peak $ | Total $ | MaxDD $ | MaxDD % | PF | Blow-ups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $12 | 0.01 | 2 | 50.0 | 0.00 | 19.39 | −12.00 | 19.39 | 100% | 0.38 | 1 |
| $12 | 0.02 | 1 | 0.0 | 0.00 | 12.00 | −12.00 | 12.00 | 100% | 0.00 | 1 |
| $100 | 0.01 | 195 | 68.7 | 445.45 | 450.79 | +345.45 | 33.37 | 7.4% | 1.92 | 0 |
| $100 | 0.02 | 195 | 68.7 | 790.90 | 801.57 | +690.90 | 66.75 | 8.3% | 1.92 | 0 |
| $200 | 0.01 | 195 | 68.7 | 545.45 | 550.79 | +345.45 | 33.37 | 6.1% | 1.92 | 0 |
| $200 | 0.02 | 195 | 68.7 | 890.90 | 901.57 | +690.90 | 66.75 | 7.4% | 1.92 | 0 |

Scalp veto: blocked 6,725 setups, 169 blocked trades, blocked PnL +$129.13 at 0.01 (+$258.25 at 0.02), blocked win rate 63.9%, 0 blocked blow-ups.
Long +$117.77 / short +$227.68 at 0.01 (doubled at 0.02). Shorts carry Scalp. Worst single loss −$25.04 at 0.01 (−$50.08 at 0.02).
Max losing streak survived: 5.

**Read:** $12 Scalp died at 2 trades (censored). At $100/0.01 it survives to 195 trades with the cleanest numbers of any preset:
PF 1.92, MaxDD only 7.4%, worst loss −$25.04. At 0.02 it also survives (+$690.90, MaxDD 8.3%).
BUT: 6-month subset only. No full-2y M5 data. Do not treat Scalp as proven on 2y.

## Does $100–200 fix the fragility?

**Yes, mostly — but it does not fix edge, only survival.**

- $12 → $100 at 0.01: blow-up room 8.5x. Swing, Intraday, Scalp all go from "dies early" to "survives the full window" (Intraday $12 → 14 trades dead; $100 → 277 trades alive. Scalp $12 → 2 trades dead; $100 → 195 trades alive).
- $100 → $200 at 0.01: same $ PnL paths for Swing/Scalp/Intraday (identical trades), but MaxDD% falls (Swing 18.9% → 16.1%, Intraday 47.2% → 39.7%, Scalp 7.4% → 6.1%) and a single worst-case loss is half as deadly.
- $200 is what makes Intraday + 0.02 survive. $100 + 0.02 blows Intraday.
- Intraday at $100/0.01 still has a 47.2% MaxDD and a −$142 worst loss. It survives, but it is not comfortable.

## How many losses in a row can the account survive?

Observed max losing streaks actually survived in the runs: Swing 3, Intraday 8, Scalp 5.
At $12, those same streaks killed the account early (censoring) — the $12 runs never got to show the full streak.
Worst single losses (after balance had grown): Swing −$40.94 (0.01), Intraday −$142.00 (0.01), Scalp −$25.04 (0.01). Double at 0.02.
At $12, ONE worst-case Swing loss (−$40.94) exceeds the whole account. At $100 it is 41% of starting balance. At $200 it is 20%.
At $100/0.01, one worst-case Intraday loss (−$142) exceeds the starting balance — Intraday at $100 survives only because the balance had already grown before that loss hit.

## Veto: safety, not profit (except Swing)

- Swing: veto blocked losers (−$107.56 blocked PnL, 41.7% win). Veto helped profit AND safety.
- Intraday $100/0.01: veto blocked WINNERS (+$376.69 blocked PnL, 71.4% win). Veto cost profit.
- Scalp $100/0.01: veto blocked WINNERS (+$129.13 blocked PnL, 63.9% win). Veto cost profit.
- In all surviving configs veto kept 0 blow-ups. Veto is a safety filter that costs profit on Intraday/Scalp and adds profit on Swing.
- Caveat: no no-veto runs were done at $100/200, so veto-vs-balance credit is unseparated. The survival improvement from $12 → $100 is mostly balance, not veto.

## Recommendation: at $100–200, keep 0.01 lot first

1. **Start $100–200 at 0.01 lot. Do NOT scale to 0.02 on day one.**
2. **Never scale Intraday to 0.02 at $100** — it blows in backtest (80 trades, −$100). Intraday 0.02 needs $200+ and even then MaxDD is 47.2%.
3. **0.02 survives at $200 for all three presets** in backtest, and at $100 for Swing + Scalp. But it doubles MaxDD $ and doubles worst loss. Scale only after forward demo proves 0.01 AND the balance has grown (e.g. balance ≥ $200, or after surviving a full MaxDD cycle) — never scale just because the backtest total is bigger.
4. **$100 is enough for Swing + Scalp at 0.01.** $200 is the safer choice if Intraday is traded, or if 0.02 is ever used.
5. **$12 demo stays $12.** Prove the engine on $12 demo first (Swing only, 0.01). The $100–200 phase starts after that proof, not during it.

## What's still weak

- Backtest is NOT proof. Slippage $0 assumed. POI-mid LIMIT fill assumes intra-bar paths OHLC cannot prove. Real fills will be worse.
- Scalp = Apr–Oct 2026 subset only. No full-2y M5 data. No out-of-sample / walk-forward split for any preset at $100/200.
- No no-veto runs at $100/200 — veto-vs-balance credit unseparated.
- Single-account paths only, no reset-mode runs at $100/200. Blow-up rates under reset not estimated.
- Intraday worst loss −$142 and MaxDD 47.2% at $100/0.01 are uncomfortable even though the run survives.
- Exness leverage restrictions around news/weekends/HMR not modeled. Real margin may be higher in those windows.
- $12 demo repo untouched by this report.
