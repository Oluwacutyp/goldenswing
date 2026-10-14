# GoldenSwing — Bollinger trend states, measured

**Data:** GC=F hourly, 11,426 bars, 2024-10-11 → 2026-10-09 (1.99y).
$0.45/oz cost. All states from `goldenswing/features/bollinger.py`
(BB 20, 2σ, squeeze = bandwidth percentile < 20 over 252 bars), causal —
every field at bar t comes from bars ≤ t. All knobs pre-committed.

## Q1 — Squeeze → expansion: KILLED as a directional setup

161 squeeze episodes (maximal runs ≥ 5 bars).

| Test | Result |
|---|---|
| 24-bar \|move\| / ATR after episode end | **3.212** vs 2.846 random-bar baseline (+13%) |
| Breakout drift (first 20-bar-range break, 24 bars, ATR) | +0.228, win 49.3% (n=148) |
| HTF-bias (reader daily_bias) predicts 24-bar sign | 55.3% (n=161) |
| Breakout book (enter break, stop at range extreme ±0.25 ATR, fixed 2R) | 104 trades, PF 1.06, **+2.88%**, Sharpe 0.20, avgR +0.035 |

**Halves kill it:** H1 −0.98% / H2 +3.89%. The breakout direction is a
coin flip (49.3%), and the HTF bias barely beats one (55.3%).

**Read:** the squeeze is REAL as a volatility phenomenon — post-squeeze
moves run ~13% larger than baseline. But direction is not in the bands,
and neither the breakout nor the HTF bias extracts it reliably on gold
H1. Carter's squeeze tells you *when* (volatility is coiled), never
*which way* — the feature docstring already said exactly this
("direction unknown from the bands alone"), and the measurement agrees.

**Verdict:** killed as a directional entry. The squeeze keeps its honest
role as a volatility-context input (expansion due, direction unknown).

## Q2 — Walk-the-band as continuation: CANDIDATE (entry context)

Anchor signals (Combo A + aoc, score ≥ 60) simulated standalone, split by
whether `walk == signal direction` at the signal bar:

| Bucket | n | Avg R | PF | Win% |
|---|---|---|---|---|
| walk == direction | 60 | **+0.265** | **1.47** | 45.0 |
| no walk | 407 | +0.025 | 1.04 | 42.0 |

Halves for the walk bucket: **H1 +0.153R / H2 +0.351R** — positive in
both. This is the strongest entry-context split in the Bollinger family:
a signal firing while price rides the band in the signal's direction
grades ten times the R of one that doesn't.

**Verdict:** candidate as entry context — same shelf as the imbalance
magnet and the T2 pin-violation vote (small, measured, never a trigger).
Needs OOS before it is scored anywhere. n=60 is suggestive, not proof.

## Q3 — Band-rejection wick as reversal: KILLED

1,718 events (wick poked through a band, close back inside). Forward
drift in the rejection direction, in ATR:

| Horizon | Drift | H1 | H2 | Win% (24-bar) |
|---|---|---|---|---|
| 8 bars | −0.101 | — | — | — |
| 12 bars | −0.131 | — | — | — |
| 24 bars | −0.064 | −0.112 | −0.010 | 51.8 |

The drift is NEGATIVE at every horizon — the rejection wick predicts the
wrong way, mildly but consistently. (Win rate 51.8% with negative drift:
small wins, larger losses against.) If the poke-through means anything,
it is a weak *continuation* tell — price pushing through the band tends
to keep pushing, which rhymes with the body-stop finding that wick
extremes get retested rather than respected.

**Verdict:** killed as a reversal signal. Noted as a possible weak
continuation input for the placement file, untested as such.

## Honest limits

- One instrument (gold H1), one 2-year bullish window, in-sample.
- The breakout book uses a fixed-2R target (not pool targets) to isolate
  the squeeze question; a pool-target variant was not tested.
- Walk-the-band n=60: candidate, not a finding — OOS required.
- Rejection drift measured on closes; intrabar paths not modeled.
