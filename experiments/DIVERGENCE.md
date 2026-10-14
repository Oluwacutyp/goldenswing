# GoldenSwing — RSI Divergence as a Feature

**Mined claim (TradingLab scalping video, user-mined 2026-10-11):**
*RSI divergence (price makes a new extreme, RSI doesn't) is an early
sign of a new move. Regular divergence = reversal; hidden divergence =
continuation.* The video uses RSI(9) with auto-labelling. The reader had
RSI-based features (relstr) but **no divergence detection** — a genuinely
new detector-level family.

**Data:** GC=F hourly, 11,426 bars, 2024-10-11 → 2026-10-09 (1.99y).
Same Phase-2 entries/exits/costs as the anchor. Anchor verified first:
Combo A + aoc reproduces **exactly** — 199 trades, PF 1.25, +28.92%,
MaxDD 10.38%, Sharpe 1.00.

**Implementation** (`goldenswing/features/divergence.py`, wired through
`features/state.py`, `config.py`, `engine.py`, scored via
`gating.divergence_gate`, registered in the detector):

- Wilder's RSI (causal; warmup reads 50.0), tested causal by truncation.
- Fractal swing pivots (left=3, right=2): a pivot at bar i is CONFIRMED
  at bar i+2. A divergence is known only from its second pivot's
  confirmation bar — nothing at bar t uses information past t.
- Four types on consecutive confirmed pivots (separation 5–120 bars,
  |ΔRSI| ≥ 2.0 points): bull_reg (price LL / RSI HL), bear_reg (price
  HH / RSI LH), bull_hid (price HL / RSI LL), bear_hid (price LH /
  RSI HL). Active for 30 bars after confirmation, then expires.
- Gate (hand-set priors, ±4 like the other families): a divergence
  aligned with the trade direction backs the entry (+4, type named in
  the factor); an opposing divergence warns against it (−4); silent
  otherwise.

Census (confirmed divergences, 2y): RSI(14) → 59 bull_reg, 102
bear_reg, 94 bull_hid, 78 bear_hid. RSI(9) finds ~35% more of each.

## Role 1 — scored family: KILLED (dilutes decisively)

| Config | Trades | Win% | PF | Return | MaxDD | Sharpe | Shadow60 |
|---|---|---|---|---|---|---|---|
| anchor (Combo A + aoc) | 199 | 45.7 | 1.25 | **+28.92%** | 10.38% | 1.00 | +0.134 |
| + divergence (RSI 14) | 197 | 43.1 | 1.12 | **+12.65%** | 10.37% | 0.51 | +0.108 |
| + divergence (RSI 9) | 197 | 43.1 | 1.09 | **+12.14%** | 11.03% | 0.49 | +0.104 |

The gate fires on 285/467 (RSI14) and 332/467 (RSI9) signals — it is
too chatty, and the ±4 directional adjustments are mostly noise. The
family dilutes the book by >16pp at both RSI lengths. **Killed as a
scored family.** (The module stays wired but unscored; the context
results below are why.)

## Role 2 — entry context: presence is real, direction is folklore

Anchor signals (467) split by divergence state at the signal bar.
RSI(14), |ΔRSI| ≥ 2.0:

| Bucket | n | Avg R | Win% | PF |
|---|---|---|---|---|
| aligned (any type) | 175 | **+0.169** | 45.1 | 1.30 |
| opposing (any type) | 110 | +0.154 | 44.5 | 1.29 |
| **none** | 182 | **−0.111** | 38.5 | 0.82 |

Two findings, both robust across RSI(9)/RSI(14) and |ΔRSI| ∈ {1,2,4}:

1. **Presence beats absence, in both halves.** aligned_any: H1 +0.211 /
   H2 +0.133. none: H1 −0.107 / H2 −0.116. Signals fired in
   divergence-free (dead) markets lose; signals near a recent momentum
   extreme win. The "none" bucket is the robustly bad one.
2. **Direction does not discriminate.** Aligned (+0.169) ≈ opposing
   (+0.154) on the full window, and opposing collapses in H2 (+0.318 →
   −0.016) while aligned holds (+0.211 → +0.133). The practitioner's
   directional story — this divergence points *your* way — is
   folklore. What divergence marks is *energy*, not direction.

### Per type (aligned with signal direction, RSI 14)

| Type | n | Avg R | H1 | H2 | PF | Read |
|---|---|---|---|---|---|---|
| bear_hid | 42 | **+0.471** | +0.684 | +0.352 | 2.02 | **candidate** — positive in both halves and across all 4 param variants (+0.38–0.47R, PF 1.75–2.02) |
| bull_hid | 83 | +0.179 | +0.248 | +0.118 | 1.34 | weak positive, both halves — watchlist |
| bull_reg | 37 | +0.151 | +0.351 | −0.038 | 1.25 | not robust (H2 negative) |
| bear_reg | 23 | −0.243 | −0.004 | −0.692 | 0.64 | **never positive** — regular-bearish divergence does not mark good shorts |

Regular vs hidden do **not** do "opposite jobs" as claimed — the
pattern is lopsided, not symmetric. The one cell that survives is
hidden-bearish backing shorts: shorts with an active bear_hid grade
**+0.471R (n=42)** vs shorts without one **−0.316R (n=121)**. Shorts
are the book's weak side (BACKTEST2); this splits them cleanly — but
n=42 in one bull-market window means OOS before any scoring role.

### Veto check: dropping divergence-free signals

| Book @ ≥60 | Trades | PF | Return | MaxDD | Sharpe |
|---|---|---|---|---|---|
| anchor (all) | 199 | 1.25 | +28.92% | 10.38% | 1.00 |
| any-divergence only | 139 | 1.35 | +28.16% | 9.69% | 1.12 |
| no-divergence only | 91 | 1.11 | +5.33% | 9.13% | 0.33 |

The veto is return-neutral (−0.76pp) with better risk-adjusted numbers
(PF +0.10, Sharpe +0.12, DD −0.69pp, 60 fewer trades) — it does not
clear the adopt bar, so it stays off. But the no-divergence book
(+5.33%, Sharpe 0.33) confirms the context finding: divergence-free
signals are dead weight.

## Verdicts

| # | Claim | Status | Note |
|---|---|---|---|
| 29 | RSI divergence as a scored family (±4 gate) | **killed** | Dilutes +28.92% → +12.65%/+12.14% at RSI 14/9. Too chatty (285–332/467 signals), directional scoring is noise |
| 30 | Divergence presence as entry context | **candidate** | Any-divergence +0.17R vs none −0.11R, both halves, all param variants. Energy marker, not directional |
| 31 | Hidden-bearish divergence backing shorts | **candidate** | +0.38–0.47R, PF 1.75–2.02, both halves, all variants; shorts split +0.47R vs −0.32R. Needs OOS (n=30–52) |
| 32 | Regular = reversal / hidden = continuation (directional story) | **killed** | Aligned ≈ opposing; bear_reg never positive; pattern is lopsided, not symmetric |

## Honest limits

- One instrument (gold H1), one 2-year bullish window, in-sample
  design. The bear_hid cell especially needs out-of-sample before any
  scoring role — n=42 and shorts behave differently in bear markets.
- Pivots are fractal (left=3, right=2); a 2-bar confirmation lag is
  disclosed. Different pivot definitions would find different
  divergences; the rule set was pre-committed, not fitted.
- |ΔRSI| sensitivity (1.0/2.0/4.0) moves the census but not the
  verdicts — reported, not selected.
- `compute()` carries a test-only `rsi_override` injection point so
  fixtures can pin exact RSI values; production never passes it.
