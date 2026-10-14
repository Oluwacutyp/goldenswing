# MQL5 CodeBase Mining — Round 2 (2026-10-11)

Second pass over free CodeBase/Articles with published source, ICT/SMC/liquidity/
price-action families. Round 1 (docs/mql5_mining.md) covered the ICT Structure
Visualizer, Market Intent Engine, ATS Risk Library, FvgGold, SMC_ALL_IN_1, the
Price Action Toolkit trendline parts, and others — none of those are re-mined here.

Rule held throughout: free + source-visible only. Two paid Market products
(SmartSMC Liquidity Sweep, SMC Liquidity Hunter) surfaced in search and were
skipped — descriptions only, unverifiable. No martingale/grid found in this batch.

## 1. SMC Liquidity Sweep Scalper — RanaAli878 (free CodeBase EA)
- URL: https://www.mql5.com/en/code/77094 (published 2026-09-07)
- Source: free CodeBase (login required to download; mechanics below from the
  author's unusually detailed writeup, not from a source read — thresholds for
  the quality gates are not verified).
- Mechanics (paraphrased): tracks the most recent *unswept* swing high/low from
  confirmed closed bars. Sweep = wick past the level + close back through.
  The sweeping candle is treated as an order block. Then it waits a configurable
  number of bars for a confirmation break back through *the sweep candle's body*
  and enters at market. Stop beyond the sweep extreme; fixed R:R take-profit;
  everything ATR-scaled (sweep size, SL buffer, min stop). HTF EMA trend filter
  (sweeps must agree with bigger picture). Optional server-time session window.
  Quality gates before entry: stop distance meaningful vs ATR and vs spread; the
  confirmation candle must show real momentum, not weak drift. Every closed
  trade updates a per-setup-type (bullish/bearish) win-rate scorecard drawn on
  the chart.
- Honesty note: the author publishes a multi-symbol optimization table showing
  mixed results (AUDUSD PF 2.36 on 26 trades; GBPUSD negative) and states plainly
  the EA is early and symbol-dependent. One of the few honest authors in the space.
- What it gives us: a THIRD confirmation definition for the same sweep setup —
  ours is displacement+CHoCH/BOS, the Tsutsumi video's was engulfing close
  (killed, #27), this one's is break-back-through-the-sweep-candle's-body.
  Directly testable A/B. Also: the per-setup win-rate scorecard idea (audit pattern).

## 2. S.M.A.R.T. V3.0 — Session Manipulation & Algorithmic Reaction Tracker (IronHawk Capital LLC, free CodeBase indicator)
- CodeBase ID 77071 (direct page fetch 403'd; mechanics from the author's detailed
  public writeup via robotfx mirror — weights and rules quoted from that writeup).
- What it is: non-trading, closed-bar-only scanner over 3 symbols on M5. A setup is
  only projected when FIVE events occur in order:
  1. sweep of a completed Asian or London session range high/low;
  2. close through a confirmed pre-sweep swing (MSS);
  3. the MSS candle shows sufficient displacement;
  4. a strict 3-candle FVG forms in the same direction;
  5. a LATER closed bar retests the FVG and closes beyond its midpoint
     (the FVG candle cannot confirm its own retest).
- Invalidation: breaching the frozen sweep extreme after MSS kills the sequence.
  Failed/expired observations become "Watch" markers — no score, no projection.
- SMART Quality Score (100 max, only computed after the full sequence):
  session liquidity context 15; sweep depth 10 or 15; confirmed MSS 20;
  displacement strength 5 or 10; FVG quality 15 or 20; later FVG retest 15;
  tick-volume bonus 0 or 5. Author states validated scores run 80–100 and the
  score is confluence-inside-the-model, NOT a win probability.
- Trade projection: entry at the retest-bar close; SL beyond the frozen sweep
  extreme + ATR buffer; TP1 exactly 1.2R; TP2 = opposing session liquidity
  capped at 2.0R; minimum 1.5R of room required for TP2.
- What it gives us: a third published scoring prior (after FvgGold and the Market
  Intent Engine) — directly comparable weights. Two testable deltas vs ours:
  (a) entry = retest bar CLOSE beyond midpoint (we use resting limit AT midline);
  (b) TP2 = opposing session liquidity (we use nearest pool ≥1R). The frozen-sweep-
  extreme invalidation is a crisp rule our sweep playbook lacks.

## 3. FVG + MSS EA — Eugene Mmene / EMcapital (MQL5 Article, full source inline)
- URL: https://www.mql5.com/en/articles/18669 (article text + Fvg_detector.mq5 +
  fvg-mss.mq5 v2.27.2, source read inline)
- FVG detector: 3-candle gap (low[0] > high[2] bullish), min 100 points,
  rectangles extend 10 bars, runs on M5/M15/H1.
- EA entry (from source, function-level): signal = breakout of the 10-bar
  high/low (BuyBrk/SellBrk in code); breakout-strength filter
  |price − broken level| / ATR ≥ dynBrkStr (MinBrkStr = 0.1 ATR); tick-volume
  vs MA filter (bypassable); HTF EMA(20) trend filter on D1/H4 (off by default);
  SL = 2× ATR, TP = 2× SL (fixed 1:2); one position per timeframe-magic;
  1-hour cooldown between signals per TF; news pause via API; free-margin check
  (1.2× required).
- Risk (from source — genuinely interesting): STREAK-ADAPTIVE sizing — if equity
  is below its high-water mark AND loss streak ≥ 2, risk drops to 25% of base
  (floor 0.1%); win streak ≥ 3 raises risk to 125% (cap 5%); resets at the
  recovery target. Per-trade loss capped at $110 (MathMin vs % risk). Daily DD
  2.5% → flatten all + pause to next day. Overall DD 5.5% → flatten + halt.
  Profit target → close all + pause. Trade management: 50% partial at 3R, SL to
  breakeven, then 1.5× ATR trail.
- Prose-only (not in code): the FVG CASCADE entry — HTF FVG → price returns
  into the gap → engulfing candle on the next TF down (ideally near London/NY
  open) → MSS on the next TF down → enter on the breakout or at 60% retracement,
  SL at the opposite end of the breakout. And FVG FAILURE FILTERS: skip
  counter-trend FVGs (never respected); skip when the HTF liquidity target is
  already taken (liquidity purged = no fuel); the sequence must be
  FVG → engulfing → MSS in that order during active sessions. FVGs described
  as propulsion WITH the trend, not reversal magnets.
- What it gives us: (a) the cascade is a new multi-TF entry sequence;
  (b) the "HTF target already taken" failure filter is a cheap context gate;
  (c) streak-adaptive risk + per-trade USD cap + partial-at-3R are risk-layer
  ideas for the infra track. (d) Sloppy bit: hardcoded API key in source —
  not our problem, noted for provenance hygiene.

## 4. Automating Trading Strategies Part 45: Inverse Fair Value Gap — Allan Munene Mutiiria (MQL5 Article, full source inline)
- URL: https://www.mql5.com/en/articles/20361 (FVG_Inverse.mq5, 81.62 KB attached;
  source read inline; backtest section shows placeholder images only — NO
  published numbers, treat as educational)
- Precise 4-state FVG lifecycle (FVGZone struct): Normal → Mitigated → Retraced
  → Inverted. Mitigation = far-side break (bullish gap: prev low < gap low).
  Retracement = a bar overlaps the gap after mitigation. Inversion = CLOSE
  beyond the far side FROM INSIDE — requires bar2's close inside the gap and
  the close beyond the far side, and explicitly NOT the mitigation bar itself.
  Trade: buy on bullish IFVG (bearish gap inverted up), sell on bearish IFVG;
  SL beyond the gap extreme ± fixed points; fixed-point TP; per-FVG trade modes
  (TradeOnce / LimitedTrades / UnlimitedTrades); overlap-ignoring option;
  max 50 tracked FVGs; expired FVGs dropped from tracking (kept on chart);
  points-based trailing with min-profit-to-trail.
- What it gives us: the crispest published IFVG *definition*. Our #25 killed
  IFVG-as-signal (+0.29 ATR drift, noise) — but our definition was looser.
  This strict variant (retracement + from-inside close required) is a
  definitional open variant, not a re-test of the same claim.

## 5. MSNR Key Levels MTF — Vesath (free CodeBase indicator)
- URL: https://www.mql5.com/en/code/76993 (published 2026-09-09)
- Precise body-based level definitions (from the CodeBase description +
  cross-checked against the kayk-pips MSNR spec docs on GitHub):
  A-level = a bullish candle followed by a bearish candle; the level is the
  CLOSE of the bullish candle; acts as resistance. V-level = bearish followed
  by bullish; level = close of the bearish candle; acts as support.
  SBR/RBS: broken support becomes resistance and vice versa. Fresh/unfresh:
  a level is fresh until tested by a wick; a body cross since the last wick
  touch re-freshens it. (Malaysian SNR "Gap Level": close/open gap between two
  same-color candles — from the spec docs, not the indicator text.)
- What it gives us: a genuinely NEW level family — body-close pivots instead of
  extreme pivots. Every pool/zone consumer downstream could be re-anchored.
  Testable as: do body-close levels mark better S/R (bounce/hold rates) than
  extreme-based pools on gold H1?

## 6. Price Action Toolkit Part 27: Liquidity Sweep With MA Filter — MetaQuotes (MQL5 Article, full source)
- URL: https://www.mql5.com/en/articles/18379 (same series as the trendline parts
  mined in round 1; source read inline)
- Precise TWO-BAR micro-sweep booleans (function DetectLiquiditySweep):
  LessStrict bullish = bullish candle AND low < prior low AND close > prior
  OPEN (not prior high — this blocks dojis) AND prior not a doji. Strict =
  same but close > prior HIGH (full reclaim). Bearish mirrors. Optional
  ColorChangeOnly (requires bull-after-bear flip). MA filter: 6 MA types
  (SMA/EMA/LWMA/VWMA/RMA/HMA); keep only sweeps with price on the MA side
  (bullish sweeps above MA, bearish below).
- What it gives us: micro-sweep granularity (prior-BAR extreme, not swing
  level) and the GRADED RECLAIM concept — close-beyond-open vs close-beyond-
  high as two strictness tiers. Our sweep rule is wick-beyond + close-back;
  their reclaim tiers are a quality dimension we don't have. Testable as a
  sweep-quality grade or a micro-trigger.

## Junk / skipped this round
- SmartSMC Liquidity Sweep / SMC Liquidity Hunter (Market products): paid,
  closed source. Their descriptions mention EQH/EQL detection within N pips,
  Fib 50–60% post-CHoCH entries, Asian-range marking — plausible mechanics,
  unverifiable. Noted, not mined.
- Killzone Liquidity Sweep EV (Market product 154855): paid, closed source.
  London/NY killzone + EMA 50/100/200 + pending stop orders — description only.
- ICT Asia Range / ICT Silver Bullet CRT EAs (Market products 141111/141221):
  paid. Judas-swing fade of Asia range; silver-bullet hours (London 3–4am,
  NY 10–11am / 2–3pm ET); TP at opposite range side. Good session mechanics,
  unverifiable — but the session windows are public ICT doctrine anyway.
