# GoldenSwing Research Summary: October 12, 2026

## What This Document Is

A record of what I built, what I tested, what I found, and how I applied it.
Not conclusions — observations. Every finding includes the methodology so you
can judge whether my tests were valid or whether I got something wrong.

You said you'd study this diligently. I've tried to make that possible by
showing my work, not just my results.

---

## 1. Starting Point

We were working on the GoldenSwing trading bot. You authorized three directions:
- EMA stack alignment (your pullback-bot filter)
- Support-becoming-resistance / resistance-becoming-support setups
- Fixing shorts instead of removing them

The live paper loop was running on OANDA practice data, hourly, with the old
configuration (mixed longs/shorts, no stack filter, score ≥60).

---

## 2. EMA Stack: What I Built and Found

### What I built
- Added `MarketState.ema_stack`: +1 (9>21>50 bullish), -1 (9<21<50 bearish), 0 (mixed)
- Added `DetectorConfig.require_stack_alignment`: when True, signals only fire
  if direction matches the stack
- Calculated EMAs causally (no future data) in the MarketReader
- All 620 existing tests pass with the change

### What I found (Jan 2022 – Oct 2024, OANDA XAUUSD H1)
At score ≥70:
- Shorts aligned with bearish stack: **-0.155R average**
- Shorts against stack: **-0.511R average**
- Longs aligned with bullish stack: **+0.119R average**
- Longs against stack: **-0.012R average**

The stack helps both directions, but doesn't fix shorts (still negative).

### How I applied it
Long-only + bull stack + score ≥65:
- 2022–2024: **179 trades, +9.7%, PF 1.11**
- 2024–2026: **155 trades, +54.8%, PF 1.56**

This is the only configuration that was profitable in both regimes.

### Limitation I noticed
When I ran the integrated detector (with the stack gate enabled in code), I got
+5.5% instead of +9.7%. I did not reconcile this discrepancy. The post-filter
calculation and the integrated detector may differ in signal spacing or state
construction. Don't treat +9.7% as confirmed until this is resolved.

### Status
Committed. NOT yet enabled in the live paper loop. The gate exists but defaults
to off.

---

## 3. Short Diagnosis: MFE/MAE

### What I did
Analyzed 259 short trades to understand WHY they lose.

### What I found
- Mean MFE (max favorable excursion): **1.27R**
- Median MFE: **0.84R**
- 47% reached at least 1R profit at some point
- 43 trades reached 1R+ and then stopped out (gave back gains)
- 179 stopped out, 75 trailed out, 5 timed out

### How I interpreted it
The direction isn't always wrong — nearly half the shorts go 1R in our favor.
The problem is we don't capture it. Exits are the suspect, not just entries.

### What I tried (exit surgery)
Short-specific exits: wider stop (0.5x ATR), earlier partial (0.8R), tighter
trail (trigger 1.0R, distance 0.7R).

Result: win rate 31%→40%, net -55.3%→-48.8%. Better, but still deeply negative.
Combined with bearish stack filter: -20.0%, PF 0.70. Better, still negative.

### Limitation
I tested exit changes on the same losing signals. If the signals are bad,
better exits can only do so much.

---

## 4. S/R Flip Infrastructure

### What I built
- `FlipLevel` dataclass: tracks broken support/resistance levels
- Reader marks a flip when price closes beyond a swing level
- Flip deactivates if price reclaims the level or after 30 bars
- Detector has `sr_flip` setup kind: price retesting a broken level
- `playbook_of` routes it to the "sr_flip" playbook

### What I found (testing the sr_flip signals)
148 signals, score ≥60, 2022–2024:
- Long flips: 80 trades, **-3.9%, PF 0.94** (breakeven)
- Short flips: 68 trades, **-20.6%, PF 0.58** (negative, but better than -55%)

### How I interpret it
The infrastructure works (levels are tracked, signals fire). But as a standalone
pattern, S/R flips don't have an edge on gold. The short flips are less bad than
generic shorts, which suggests the CONCEPT has merit but needs better context.

### Status
Committed. Infrastructure ready for your ideas.

---

## 5. Turtle Soup: The Full Story

### What the mining said
Turtle Soup short: HTF buy-side liquidity swept → LTF bearish MSS → short.
Stop above swept level. Target HTF sell-side liquidity. Practitioner consensus
on the mechanics.

### What I built (attempt 1: H1-only)
`_turtle_soup_short` in the detector: buy-side sweep + bearish CHoCH + displacement.
Opt-in via `turtle_soup_short` flag.

Result: **25 signals, -17.2%, PF 0.19, win 12%**. Terrible.

### What I built (attempt 2: H1 + M15 hierarchical)
H1 buy-side sweep → fetch M15 → find bearish MSS → enter.

Result: **721 entries, -100.2% (wipeout), PF 0.24, win 28%**. Catastrophic.

### What went wrong (your catch)
You said: "No way all strategies are dead, something is wrong somewhere."
You were right. I investigated and found:

**My sweep detection was noise.** A pool is created at EVERY swing high. With
fractal_n=2, that's hundreds of pools. My 814 "sweeps" had a median wick depth
of 0.17 ATR — that's not a liquidity grab, that's normal price movement.

**My MSS detection was noise.** 721 out of 814 sweeps (89%) found an M15 "MSS"
within 12 H1 bars. On M15, there's ALWAYS some bearish structure event in a
48-bar window. I wasn't finding real shifts, I was finding noise.

### How I fixed the methodology (attempt 3: real sweeps)
Filtered for:
- Wick depth > 0.5 ATR (real liquidity grab, not noise)
- D1 bias not bullish (HTF permission)

This gave **66 real sweeps** (vs 814 noise). Tested with M15 MSS entry:
**57 trades, -0.1%, PF 1.01, win 42%**. Breakeven.

### What I conclude from this
The pattern isn't "dead" — my first two implementations were broken. But even
tested correctly, Turtle Soup is breakeven on gold, not profitable. The
infrastructure is sound; the edge isn't there on this instrument.

### Status
Committed. The `_turtle_soup_short` detector method and the MTF experiment script
are in the repo.

---

## 6. V-Reversal (V-Top) Short

### What the mining said
Parabolic-exhaustion short: acceleration + climax candle (highest volume) →
short the first close below climax low. Stop above climax high. Never short
the front side.

### What I built
Standalone experiment script (not in the detector):
- Parabolic rise: 5-bar gain > 2x ATR, accelerating
- Climax: range > 1.5x ATR
- Trigger: next bar closes below climax low
- Stop above climax high, 2R target

(No volume in MarketState, used range as proxy.)

### What I found
**23 setups, -2.0%, PF 0.88, win 35%**. Breakeven, very selective.

### Status
Committed as `scripts/experiment_v_reversal.py`.

---

## 7. Entry Validation System

### What I built
`goldenswing/entry/validator.py`:
- `validate_bearish_rejection`: checks for rejection wick, engulfing, or
  close-back at a resistance level
- `validate_bullish_rejection`: mirror for support
- `choose_entry_type`: routes to LIMIT_AT_ZONE (patient) or MARKET_ON_CONFIRM
  (aggressive) based on setup kind

This was the "candles validating entries" system from the original vision.

### What I found (shorts)
- Unvalidated short flips: -20.6%, PF 0.58, win 32%
- Validated short flips: **-28.4%, PF 0.26, win 12%**

Validation made it WORSE. Win rate collapsed from 32% to 12%.

### What I found (longs)
- Baseline longs (stack-aligned, ≥65): +6.2%, PF 1.06, win 43%
- Validated longs: **-53.7%, PF 0.25, win 12%**

Validation also made longs worse. Dramatically worse.

### How I interpret it
My validation logic is inverted or the timing is wrong. When I detect a
"bullish rejection" at support, it's actually a bearish signal. The patterns
I'm looking for are selecting losers, not winners.

This needs to be fixed or rebuilt. The INFRASTRUCTURE (the validator module,
the routing logic) is sound. The PATTERN DEFINITIONS are wrong.

### Status
Committed. Needs debugging.

---

## 8. Regime System

### What I built
`goldenswing/system/regime.py`:
- `Regime` enum: strong_bull, bull_lean, strong_bear, bear_lean, chop
- `classify_regime`: D1 bias + H1 EMA stack + H1 trend
- `route_signal`: checks if a signal is allowed in the current regime
- `REGIME_ROUTING`: table of which directions/kinds/scores per regime

### What I found
- 735 baseline signals → 675 routed (92% pass-through)
- Baseline: -70.5%, PF 0.76
- Routed: -64.5%, PF 0.77

Barely any improvement. The regimes are too permissive — 92% pass means it's
not really filtering.

### How I interpret it
The CONCEPT (regime-aware routing) is correct. The IMPLEMENTATION (my regime
definitions) is too loose. "Bull lean" and "bear lean" catch too much. The
CHOP regime only caught 17 signals (2%).

This needs tighter regime definitions, or it needs to be combined with the
proven long-only filter.

### Status
Committed. Infrastructure ready, needs tuning.

---

## 9. Key Diagnoses

### Diagnosis 1: Short entries are inverted
Even in D1 bearish regimes, price goes UP first after our short signals 54%
of the time (vs 39% down first). We're shorting breakdowns (after price fell),
not rallies (before price falls). For shorts, you sell high, not low.

This was your catch ("something is wrong somewhere"). The patterns might be
fine; the ENTRY TIMING is wrong.

### Diagnosis 2: My implementations were noisy
The Turtle Soup test taught me this. I was testing 814 "sweeps" when only 66
were real. I was finding "MSS" on M15 when it was just noise. The lesson:
validate that you're testing what you think you're testing before concluding
a pattern doesn't work.

### Diagnosis 3: Candle validation is backwards
Both long and short validation made results dramatically worse. The pattern
definitions in the validator are inverted or the timing is off. This is a bug,
not a market truth.

### Diagnosis 4: Gold's bull market punishes shorts
Even with correct implementation, short patterns on XAUUSD 2022-2024 are
breakeven at best. The instrument has a structural bid. This doesn't mean
shorts "don't work" — it means they don't work HERE, in THIS regime.

---

## 10. What Actually Works (My Findings)

The ONLY configuration I found that was profitable in both tested regimes:

**Long-only + bullish EMA stack (9>21>50) + score ≥65**
- 2022–2024: 179 trades, +9.7%, PF 1.11
- 2024–2026: 155 trades, +54.8%, PF 1.56

Everything else I tested was breakeven or negative.

### Caveats
- The +9.7% vs +5.5% discrepancy (post-filter vs integrated) is unresolved
- Score ≥65 was your suggestion; I haven't proven it's optimal
- The 2024–2026 window was a strong bull market (easier for longs)
- I have NOT yet enabled this in the live loop

---

## 11. The Old Bot Question

You asked: "Could that be why my old bot failed?"

My autopsy found the old bot had 5 strategies voting by weighted confidence,
with HTF only sizing lots (not gating direction). I concluded the architecture
was the killer.

But you're asking something sharper: if shorts are structurally negative on
gold, did short trades drag down the old bot?

I don't know. I didn't separate the old bot's long vs short performance.
It's possible. The architectural problems were real, but if the bot was taking
shorts on gold, those shorts were likely losing money and making the overall
result worse.

This is worth investigating if you have the old bot's trade logs.

---

## 12. What's Built and Ready

All committed to the goldenswing repo:

1. **EMA stack gate** (`MarketState.ema_stack`, `DetectorConfig.require_stack_alignment`)
2. **S/R flip tracking** (`FlipLevel`, `active_flips`, `sr_flip` setup kind)
3. **Turtle Soup** (`_turtle_soup_short`, opt-in flag, MTF experiment script)
4. **Entry validator** (`goldenswing/entry/validator.py` — needs debugging)
5. **Regime system** (`goldenswing/system/regime.py` — needs tighter definitions)
6. **V-reversal experiment** (`scripts/experiment_v_reversal.py`)
7. **MTF data layer** (`goldenswing/backtest/mtf_data.py` — OANDA fetching)

---

## 13. Open Questions (For Your Challenge)

1. **Can shorts work on gold?** My tests say no, but my tests might be wrong.
   The entry timing diagnosis suggests we're entering at the wrong time.

2. **Is the candle validator fixable?** The logic is inverted somehow. If fixed,
   it could help both directions.

3. **Should the live loop switch to long-only + stack?** The data supports it,
   but I haven't done it yet. Waiting for your call.

4. **What did I get wrong?** You've caught me twice tonight (noisy Turtle Soup
   implementation, missing HTF layer). There's probably more.

---

## 14. How I Applied Things (Methodology Notes)

- All backtests used OANDA XAUUSD H1, Jan 2022 – Oct 2024 (16,260 bars)
- Validation window: Oct 2024 – Oct 2026 (11,987 bars), kept separate
- Costs: 0.5% per oz per trade (may not reflect real OANDA spreads)
- Score threshold: ≥60 baseline, ≥65 for the proven config (your suggestion)
- R-multiples: risk = distance to invalidation, 1% risk per trade
- I did NOT use the full book pipeline (confluence, greedy selection) for most
  pattern tests — I used raw detector signals. This means my pattern results
  may not reflect production performance.
- The MTF tests used real OANDA M15/M5 data fetched on demand

---

*End of summary. The challenge is open. Prove me wrong.*
