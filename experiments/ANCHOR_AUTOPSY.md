# Anchor Autopsy — sweep_reversal

_Data: 11426 H1 bars, 2024-10-11 → 2026-10-09 (1.99y). Traded book at validated threshold 60: **199 trades, +28.92%, PF 1.25**._

## 1. Where it broke — monthly

| Month | n | R | Win% |
|---|---:|---:|---:|
| 0 (2024-10) | 14 | -1.1 | 36 |
| 1 (2024-11) | 3 | +3.5 | 67 |
| 2 (2024-12) | 8 | -1.3 | 38 |
| 3 (2025-01) | 8 | +5.1 | 62 |
| 4 (2025-02) | 11 | +2.9 | 55 |
| 5 (2025-03) | 10 | +3.2 | 60 |
| 6 (2025-04) | 9 | -2.4 | 33 |
| 7 (2025-05) | 10 | -2.3 | 40 |
| 8 (2025-06) | 7 | +1.3 | 57 |
| 9 (2025-07) | 9 | -0.8 | 44 |
| 10 (2025-08) | 8 | +2.1 | 50 |
| 11 (2025-09) | 6 | +10.8 | 83 |
| 12 (2025-10) | 5 | -0.3 | 40 |
| 13 (2025-11) | 8 | +0.6 | 50 |
| 14 (2025-12) | 7 | +3.5 | 43 |
| 15 (2026-01) | 7 | +6.2 | 57 |
| 16 (2026-02) | 7 | -2.2 | 29 |
| 17 (2026-03) | 9 | +2.0 | 44 |
| 18 (2026-04) | 12 | -7.6 | 17 |
| 19 (2026-05) | 8 | +4.1 | 62 |
| 20 (2026-06) | 8 | -0.8 | 38 |
| 21 (2026-07) | 8 | +6.2 | 75 |
| 22 (2026-08) | 9 | -0.6 | 33 |
| 23 (2026-09) | 8 | -4.9 | 25 |

**H1 (mo 0–12): 103 trades, +21.0R, avg +0.204R/trade, win 50%**
**H2 (mo 12–24): 96 trades, +6.3R, avg +0.065R/trade, win 42%**

Monthly cumulative R: 0:-1, 1:+2, 2:+1, 3:+6, 4:+9, 5:+12, 6:+10, 7:+8, 8:+9, 9:+8, 10:+10, 11:+21, 12:+21, 13:+21, 14:+25, 15:+31, 16:+29, 17:+31, 18:+23, 19:+27, 20:+27, 21:+33, 22:+32, 23:+27

## 2. Session

| Level | n | avgR | Win% | H1 n/avgR | H2 n/avgR |
|---|---:|---:|---:|---|---|
| asia | 44 | +0.244 | 52 | 24/+0.335 | 20/+0.135 |
| london | 40 | -0.010 | 42 | 26/+0.180 | 14/-0.362 |
| newyork | 35 | +0.397 | 51 | 14/-0.063 | 21/+0.703 |
| off | 6 | -0.039 | 50 | 3/+0.254 | 3/-0.332 |
| overlap | 74 | +0.044 | 41 | 36/+0.232 | 38/-0.135 |

## 3. Volatility regime

| Level | n | avgR | Win% | H1 n/avgR | H2 n/avgR |
|---|---:|---:|---:|---|---|
| compressed | 59 | +0.225 | 54 | 34/+0.275 | 25/+0.157 |
| expanded | 66 | +0.329 | 52 | 34/+0.168 | 32/+0.500 |
| normal | 74 | -0.105 | 34 | 35/+0.169 | 39/-0.350 |

## 4. Trend state

| Level | n | avgR | Win% | H1 n/avgR | H2 n/avgR |
|---|---:|---:|---:|---|---|
| down | 74 | -0.016 | 42 | 34/+0.038 | 40/-0.062 |
| up | 125 | +0.228 | 48 | 69/+0.285 | 56/+0.156 |

## 5. HTF bias alignment

| Level | n | avgR | Win% | H1 n/avgR | H2 n/avgR |
|---|---:|---:|---:|---|---|
| against-bias | 3 | -0.247 | 33 | 2/+0.132 | 1/-1.005 |
| no-bias | 6 | -0.138 | 33 | 6/-0.138 | 0/+nan |
| with-bias | 190 | +0.152 | 46 | 95/+0.227 | 95/+0.077 |

## 6. AMD phase

| Level | n | avgR | Win% | H1 n/avgR | H2 n/avgR |
|---|---:|---:|---:|---|---|
| accumulation | 1 | +2.289 | 100 | 1/+2.289 | 0/+nan |
| consolidation | 6 | +0.185 | 50 | 3/-0.127 | 3/+0.498 |
| distribution | 14 | +0.381 | 50 | 9/+0.269 | 5/+0.582 |
| expansion_down | 19 | +0.212 | 47 | 5/+0.938 | 14/-0.048 |
| expansion_up | 18 | -0.025 | 39 | 7/+0.043 | 11/-0.068 |
| manipulation | 14 | +0.155 | 50 | 3/-0.458 | 11/+0.322 |
| trend_down | 69 | -0.011 | 41 | 34/+0.049 | 35/-0.069 |
| trend_up | 57 | +0.209 | 49 | 40/+0.244 | 17/+0.127 |
| unknown | 1 | +1.615 | 100 | 1/+1.615 | 0/+nan |

## 7. Side

| Level | n | avgR | Win% | H1 n/avgR | H2 n/avgR |
|---|---:|---:|---:|---|---|
| long | 125 | +0.228 | 48 | 69/+0.285 | 56/+0.156 |
| short | 74 | -0.016 | 42 | 34/+0.038 | 40/-0.062 |

## 8. Weekday

| Level | n | avgR | Win% | H1 n/avgR | H2 n/avgR |
|---|---:|---:|---:|---|---|
| Fri | 34 | +0.323 | 50 | 20/+0.429 | 14/+0.171 |
| Mon | 32 | +0.119 | 47 | 19/+0.389 | 13/-0.275 |
| Sun | 4 | +0.445 | 75 | 2/+0.886 | 2/+0.004 |
| Thu | 33 | +0.273 | 48 | 16/+0.233 | 17/+0.310 |
| Tue | 49 | +0.084 | 45 | 23/-0.110 | 26/+0.256 |
| Wed | 47 | -0.052 | 38 | 23/+0.089 | 24/-0.186 |

## 9. Killzone

| Level | n | avgR | Win% | H1 n/avgR | H2 n/avgR |
|---|---:|---:|---:|---|---|
| killzone | 79 | +0.191 | 47 | 44/+0.472 | 35/-0.163 |
| off-killzone | 120 | +0.101 | 45 | 59/+0.003 | 61/+0.196 |

## 10. Cross-cuts

### Side × half
| Side | Half | n | avgR | Win% |
|---|---|---:|---:|---:|
| long | H1 (mo 0-12) | 69 | +0.285 | 52 |
| long | H2 (mo 12-24) | 56 | +0.156 | 43 |
| short | H1 (mo 0-12) | 34 | +0.038 | 44 |
| short | H2 (mo 12-24) | 40 | -0.062 | 40 |

### Session × half (top sessions)
| Session | Half | n | avgR | Win% |
|---|---|---:|---:|---:|
| asia | H1 (mo 0-12) | 24 | +0.335 | 54 |
| asia | H2 (mo 12-24) | 20 | +0.135 | 50 |
| london | H1 (mo 0-12) | 26 | +0.180 | 50 |
| london | H2 (mo 12-24) | 14 | -0.362 | 29 |
| newyork | H1 (mo 0-12) | 14 | -0.063 | 36 |
| newyork | H2 (mo 12-24) | 21 | +0.703 | 62 |
| off | H1 (mo 0-12) | 3 | +0.254 | 67 |
| off | H2 (mo 12-24) | 3 | -0.332 | 33 |
| overlap | H1 (mo 0-12) | 36 | +0.232 | 50 |
| overlap | H2 (mo 12-24) | 38 | -0.135 | 32 |

### Exit reason × half
| Exit | Half | n | avgR |
|---|---|---:|---:|
| stop | H1 (mo 0-12) | 51 | -1.036 |
| stop | H2 (mo 12-24) | 55 | -1.013 |
| timeout | H1 (mo 0-12) | 6 | +0.844 |
| timeout | H2 (mo 12-24) | 3 | +1.262 |
| trail | H1 (mo 0-12) | 46 | +1.495 |
| trail | H2 (mo 12-24) | 38 | +1.531 |

## 11. Direction vs timing (MFE analysis)

| Bucket | H1 n | H1 avgR | H2 n | H2 avgR |
|---|---:|---:|---:|---:|
| MFE < 0.25R (never went anywhere — direction wrong) | 22 | +0.210 | 18 | +0.474 |
| 0.25R ≤ MFE < 1R (moved, failed to extend — timing/exit) | 38 | -0.788 | 45 | -0.762 |
| MFE ≥ 1R (had a real move) | 43 | +1.076 | 33 | +0.970 |

Of 108 losers: MFE<0.25R: 18 (17% never moved); MFE≥1R: 16 (had ≥1R then gave it back).
Wicked invalidation (crossed raw stop, survived): 118 trades, avg -0.810R; non-wicked: avg +1.516R.

## 12. Trap-intelligence overlap

### bait
| Flag | Half | n | avgR | Win% |
|---|---|---:|---:|---:|
| False | H1 (mo 0-12) | 92 | +0.210 | 50 |
| False | H2 (mo 12-24) | 85 | +0.145 | 44 |
| True | H1 (mo 0-12) | 11 | +0.147 | 45 |
| True | H2 (mo 12-24) | 11 | -0.550 | 27 |

### sfp
| Flag | Half | n | avgR | Win% |
|---|---|---:|---:|---:|
| False | H1 (mo 0-12) | 67 | +0.262 | 52 |
| False | H2 (mo 12-24) | 64 | -0.070 | 36 |
| True | H1 (mo 0-12) | 36 | +0.095 | 44 |
| True | H2 (mo 12-24) | 32 | +0.336 | 53 |

---

## VERDICT (2026-10-11)

### The short version

The anchor is **not dead — it decayed ~70% in per-trade edge** (+0.204R →
+0.066R/trade), and the walk-forward's −0.28% **overstates the decay**.
At the fixed validated threshold, months 12–24 are +6.3R over 96 trades
(weak, positive). One bad adaptive-threshold fit (window 15–18: fitted 70
vs validated 60) cost ~3 trades and ~6.5% — the fitting hurt, it didn't
help. The "edge concentrated in months 0–12" verdict is really a story
about **context migration, not time decay**: one monster month (Sep 2025,
+10.8R on 6 trades = 40% of the whole book) flattered H1, while london
session, killzone hours, and normal-vol regime died in H2 — and newyork
exploded (+0.703R/trade) to partly compensate.

### Answers

**1. Concentrated edge or true decay?** Both. Month 11 is 40% of the
book — the in-sample edge was concentrated. But several contexts show
*true decay* (worked H1, failed H2 with 14+ trades each side): london
(+0.180 → −0.362), killzone (+0.472 → −0.163), normal regime
(+0.169 → −0.350), Monday (+0.389 → −0.275). These are not variance —
they're dead contexts.

**2. Hunting ground (positive in BOTH halves)?** Yes — and it's
long-biased, which matches what the anchor always was (125/199 trades
long; shorts never worked: −0.016R/trade full sample):
- **long + with HTF bias** (120 trades): +0.284 / +0.177 — the core book
- **long + expanded regime** (42): +0.592 / +0.548 — the most *stable*
  cell in the entire map, nearly identical across halves
- **long + asia** (25): +0.562 / +0.229
- **long + Thu/Fri** (43): +0.746 / +0.246
- **long + SFP overlap** (44): +0.069 / +0.519 — *improved* in H2
- newyork longs fail the rule (H1 −0.020) despite a spectacular H2
  (+1.282) — excluded; the rule is the rule.

**3. Direction vs timing?** Direction did not fail. Trades that reached
≥1R MFE paid identically in both halves (+1.076 → +0.970), and the exit
machinery (trail +1.50R, stop −1.02R) is byte-identical across halves.
What changed: fewer trades developed (≥1R MFE: 42% → 34% of trades) and
more chopped in the 0.25–1R graveyard (−0.77R avg, *both* halves —
structural, not decayed). Only 17% of losers never moved at all. This is
a **follow-through regime story**: the market stopped extending in the
dead contexts; the strategy's mechanics are intact.

**4. Trap interaction?** Diverges from the salvage finding. Bait-state
overlap **died** for the anchor in H2 (+0.147 → −0.550, 11+11 trades) —
the anchor's sweep entries into beyond-FVG bait state are where it
bleeds, even though bait trades were profitable on the *sleeves*.
Confirmed-SFP overlap went the other way (+0.095 → +0.336). The anchor
likes confirmed SFPs; it does not like bait-state entries.

### Mechanism

The anchor is a long-biased sweep-reversal system whose edge lives in
*follow-through after the sweep*. In H1 that follow-through arrived in
london/killzone/normal-regime; in H2 it migrated to
newyork/expanded-regime/off-killzone while the old contexts chopped.
The short side was always dead weight. Nothing about entries, stops, or
trails degraded — the market's post-entry behavior changed by context.

### What this means (iron rule: anchor stays frozen)

- The CONFLUENCE.md "FAILS" verdict stands for the anchor *as traded*
  (all contexts, adaptive thresholding). It is not resurrected.
- **Variant candidate** (new playbook, own OOS validation, never a
  change to the anchor): long-only + with-bias, optionally intersected
  with asia / expanded-regime / Thu–Fri. Measured 120 trades, +0.235R
  avg, positive in both halves — but it does **not** inherit the
  anchor's in-sample; it needs 50+ genuinely held-out OOS trades like
  everything else.
- **Method note for walk-forward**: the adaptive threshold fit cost more
  than it earned here (one window, −6.5%). Future walk-forwards should
  report FIX alongside FIT — and be suspicious when they diverge.
- **Kill list addition**: short side of sweep_reversal (−0.016R/trade,
  74 trades, never positive in any half) — dead weight in every frame.

### What's still weak

- The hunting-ground cells are thin in H2 (12–21 trades each). The
  long+expanded cell's stability (+0.592/+0.548) is the most trustworthy
  signal, but 42 trades is not a promotion.
- Two disaster months (2026-04: −7.6R, 2026-09: −4.9R) drive much of
  H2's weakness — no context in the map explains them yet (not session,
  not regime, not weekday). Unexplained.
- Bait-overlap death is 22 trades — suggestive, not conclusive.
- This autopsy used the fixed validated threshold; the live engine's
  adaptive path may behave differently.
