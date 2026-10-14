# MQL5 Ecosystem Mining — for GoldenSwing

*Researched 2026-10-10. Sources: mql5.com CodeBase + Articles, GitHub mirrors of MQL5 code, Market pages (descriptions only — no code). Skepticism standard: any performance claim without a reproducible tester report is UNVERIFIED, and several are labelled as such below.*

*What I could not reach: the exact CTrendlineBuilder/CTrendlineManager article the owner referenced did not surface in search — its closest relatives (Price Action Toolkit Parts 60/61, Automating Trading Strategies Parts 25/34) are covered instead and follow the same builder/manager philosophy. MQL5 Market products are closed-source: only their feature descriptions are minable. Deep forum-thread mining was thin — forum search results were mostly product noise; the two forum-adjacent finds worth anything are included.*

---

## 1. XAUUSD / Gold Systems

### FvgGold EA — open-source XAUUSD FVG+OB EA (MIT)
- **URL:** https://github.com/aminu00/fvggold-ea (also on MQL5 Market, product 186778)
- **What:** Full MQL5 EA source for gold: quality-scored Fair Value Gaps with order-block confluence, EMA trend filter, killzone sessions.
- **Mechanism (precise):**
  - Each FVG scored 0–100: **gap size vs ATR 0–30, displacement (impulse-candle body/range ratio) 0–30, HTF alignment (H1 EMA 50/200) 0–20, freshness 0–10, premium/discount position 0–10**, plus **+20 if the FVG overlaps an order block**. Trade only above score 50.
  - OB detection: impulse body ≥ 1.5× ATR within a 50-bar lookback, minimum OB size in points.
  - **Entry is a limit order at the FVG edge + buffer, NOT the midpoint** — buy limit at FVG bottom edge, sell limit at FVG top edge. SL beyond the far FVG edge + buffer.
  - Sessions (GMT): London 07:00–10:00, overlap 12:00–16:00, NY out by 21:00; overlap preferred.
  - Risk: fixed R:R = 1.5 take-profit, **break-even at +0.5× ATR**, daily loss limit, max 1 concurrent trade, fixed lot by default.
- **Claims:** +48.7% over 6 months, 45.3% win rate — on **64 trades**, with parameters labelled "optimized." UNVERIFIED and curve-fit-shaped.
- **Verdict: PORT THE SCORING MODEL.** The component weights are a published prior for grading FVGs — directly comparable to our backing-score factors. Its edge-entry (limit at gap edge) vs our midpoint-entry is a testable difference. Ignore its performance claims entirely.

### RicardoBarato xauusd-mt5-expert-advisor — research-grade gold EA (GPL-3.0)
- **URL:** https://github.com/ricardobarato/xauusd-mt5-expert-advisor
- **What:** Source-only XAUUSD EA built around market-regime classification and risk-controlled execution, with a disciplined Strategy Tester validation methodology and strict separation of source / private presets / published docs.
- **Verdict: STEAL THE ENGINEERING POSTURE, not logic.** It deliberately publishes NO performance metrics — the honest opposite of the Market. Worth an architecture read for the EA bridge repo layout; internals not deeply inspected here.

### XAUUSD Martingale/Grid EA (anti-pattern, named so nobody rebuilds it)
- **URL:** https://github.com/osleyder1985/mql5-format
- **What:** Martingale/grid on XAUUSD M15 — grid orders at fixed steps with increasing lots until the basket closes in profit, mimicking a commercial "Trend AI" product.
- **Verdict: MARKETING JUNK / DANGEROUS.** Martingale on gold is a blow-up schedule, not a strategy. Listed only as a documented anti-pattern.

---

## 2. ICT / SMC Implementations

### ICT Structure Visualizer — sweep/MSS/FVG indicator WITH PUBLISHED HONEST STATS (free CodeBase) ⭐
- **URL:** https://www.mql5.com/en/code/78062
- **Mechanism (precise):**
  - **Liquidity pools are previous-day high/low and Asia-session high/low only** — two pool sources, not every swing.
  - **Sweep** = pool taken and candle closes back inside.
  - **MSS** = opposite structure broken by a **displacement candle** (size threshold in ATR).
  - Entry plan drawn automatically: **entry at FVG midpoint, SL beyond the sweep extreme**, TP at fixed R multiple; same-bar TP/SL collision counts as SL; plans tracked as filled / missed / TP / SL. All objects from **closed bars only, never moved**; every signal optionally logged to CSV at the moment it appears (anti-repaint proof).
  - Volume context labels CFD tick volume / tick delta honestly as tick data, never as order flow.
- **Published results (author's own, published as-is):** XAUUSD M15, Jul 2022–Oct 2026: **1,220 setups, 33.9% win rate, +22R before costs, ≈0R after spread+commission.** EURUSD: +16R before costs, **−75R after**. A sweep-volume filter (RVOL ≥ 1.5) made gold *less* negative and EURUSD *worse* — no consistent improvement. Author's own conclusion: used mechanically, sweep→MSS→FVG has **no edge after costs**, and the winning direction flips year to year with the trend.
- **Verdict: STEAL THE DESIGN + TREAT AS CALIBRATION EVIDENCE.** This independently confirms two GoldenSwing findings: (a) unfiltered mechanical sweeps are noise after costs — our displacement+CHoCH confirmation exists precisely because of this; (b) direction follows the higher-timeframe trend regime year by year — exactly what our D1-bias veto encodes. Also steal: PDH/PDL + Asia H/L as first-class pool sources, and the "log every signal to CSV at signal time" audit pattern.

### Market Intent Engine — five-stage ICT pipeline EA/indicator (MQL5 Article, full code) ⭐
- **URL:** https://www.mql5.com/en/articles/24184
- **What:** A single shared analysis code path that runs as a visual indicator with AutoTrade off and as an EA with it on — "the chart we study is the chart the machine trades."
- **Mechanism (precise):**
  - **Five one-way pipeline stages**, each answering one question: structure → liquidity (where orders cluster) → price behaviour (what price did there) → intent score → decision/trade plan. No stage may rewrite an earlier stage (anti-confirmation-bias by architecture).
  - **Four timeframe roles, all inputs:** macro H4 = bias, primary H1 = traded structure, setup M15 = liquidity/sweeps, entry M5 = confirmation.
  - **Intent score 0–100, weights published and summing to 100:** HTF bias 20, liquidity sweep 20, displacement 18, BOS/CHoCH 14, FVG present 8, FVG retest 8, premium/discount location 7, rejection 5; opposite-side evidence multiplied by a 0.30 conflict penalty. Score ≥45 = WATCH, ≥65 = ACTION.
  - **Hard gates live OUTSIDE the score:** ACTION *requires* a fresh sweep + aligned displacement + aligned BOS (boolean inputs), minimum R:R 1.5, SL buffer 0.35× ATR beyond invalidation, targets closer than 1.0× ATR ignored. Deliberate design point: a weighted score can express degree but a requirement folded into it can be out-voted by points elsewhere.
  - **Swing struct carries `confirmAt`** (the bar index where the swing becomes knowable — the anti-lookahead field) and **separate `broken` vs `swept` booleans** (close-through = absorbed/finished; wick-through + close-back = defended and *more* interesting). 
  - Liquidity stored as **pools with touch counts and a 0–1 strength grade**, not lines; sweeps stored as graded events (excursion size, closed-back flag, follow-through displacement/structure-break flags), not booleans.
- **Verdict: STEAL THE DESIGN — this is GoldenSwing's architectural twin.** Two upgrades to port into our reader: (1) gates-outside-the-score as a formal rule (our Phase-2 mandatory confirmation already behaves this way — formalise it), (2) `confirmAt` + broken≠swept as explicit fields on every swing/pool. The published weights are a second prior to cross-check ours against.

### SMC_ALL_IN_1 EA — OB + BOS + FVG unified EA (MQL5 Article series, full code)
- **URL:** https://www.mql5.com/en/articles/16340
- **Mechanism (precise):**
  - One EA, strategy enum: OB-only / FVG-only / BOS-only / AUTO. **Separate swing streams for OB detection vs BOS detection** (different swing parameters per purpose).
  - **OB entries require Fibonacci validation: the retrace into the OB must reach the 61.8% level** of the impulse leg — a context gate our reader does not currently have (Fib used as OB qualifier, not as a standalone signal).
  - FVGs traded at the **50% equilibrium of the gap** (`FVG_TradeAtEQ`), min gap in points, 20-bar scan window.
  - Per-setup "traded once" state flags prevent duplicate entries on the same OB/BOS.
  - Weakness: fixed-point SL/TP (3500/7500 points) regardless of structure — inferior to invalidation-based stops.
- **Verdict: PORT ONE MECHANIC — Fib-61.8% as an OB entry qualifier** (belongs with our Phase-4 context gating: "Fib where a trader actually uses it"). The rest duplicates our reader with weaker risk logic.

### ICT Sell-Side Sweep Scanner — concrete sweep-candle quality rules (open .mq5)
- **URL:** https://github.com/ms304/metatrader5/blob/main/Ichimoku20260426/ICT_SellSide_Sweep_Scanner.mq5
- **Mechanism:** A sell-side sweep is only flagged when the sweeping candle itself shows rejection quality: **close in the top third of its range (close − low > 0.66 × range), lower wick > 30% of range, and close back above the swept low**, multi-symbol scan.
- **Verdict: PORT THE CANDLE-QUALITY GATE.** Our sweep rule is wick-beyond + close-back; adding close-location (top/bottom third) and wick-share thresholds gives the sweep detector a graded quality dimension the ICT Visualizer's data says it needs.

### SM Radar (Pine Script reference for precise definitions)
- **URL:** https://github.com/CedInvest/sm-radar-pine
- **Mechanism:** OB = last opposing candle before an impulsive move that breaks structure, mitigation status tracked; FVG = 3-candle inefficiency with auto-fill detection; liquidity = equal highs/lows shelves (BSL above EQH, SSL below EQL); swing points carry a structural-significance score.
- **Verdict: INTERESTING REFERENCE.** Not MQL5, but the crispest public rule statements to cross-check our definitions against. Definitions align with our reader; no new mechanics.

### The compiled-only SMC crowd (Market indicators + two GitHub repos)
- Order Block Plus / FVG detectors (paid Market indicators, e.g. https://www.mql5.com/en/market/product/147712) advertise an **OB strength score 0–99 from accumulated tick volume + accompanying FVG size** — a reasonable grading idea (OB strength ∝ displacement volume × gap size), but closed-source, so the formula is unverifiable.
- **BPR (Balanced Price Range)** from the SMC Venom indicator: two opposing FVGs overlapping form a "bridge" zone — an IFVG-adjacent concept our reader lacks. Concept worth a backtest someday; the product itself is a $45 description page.
- https://github.com/generaltradingsarl/smart-money-concepts — genuine MQL5 SMC indicator but ships **.ex5 only, no source** — cannot be audited, therefore cannot be trusted or ported.
- https://github.com/velmopk/smart-money-concepts-indicator-mt5 and the Solasent "scanner" are **C#/.NET desktop simulators wearing MQL5 clothing** (simulated feeds, WinForms UI). 
- **Verdict: MARKETING JUNK (the .NET simulators); UNVERIFIABLE (the .ex5-only and paid indicators).** The OB-strength-by-volume×gap and BPR ideas are noted as hypotheses only.

---

## 3. Trendline / Structure Frameworks

### Price Action Toolkit Part 61 — Slanted Trendlines with 3-Swing Validation (Article, full code) ⭐
- **URL:** https://www.mql5.com/en/articles/21277 (Part 60, objective swing trendlines: https://www.mql5.com/en/articles/21226)
- **Mechanism (precise):**
  - Swings filtered by size: candidate swing must exceed **0.8× ATR** (or a fixed minimum) — noise swings never become anchors.
  - A line needs **two anchor swings to exist and a THIRD swing to be validated**: the third swing must land within **0.15× ATR of the line's price at its time**, and (configurable) must occur *after* the second anchor. Only validated lines can produce signals.
  - Retains only the **most recent ~10 significant swings per side** and the single most significant line per side — deliberate chart discipline, no geometry spam.
  - **Breakout = close-based decisive cross + a points buffer beyond the line** (not a wick touch), recorded with the line's touch count.
- **Verdict: PORT THE VALIDATION MATH into our Phase-4 trendline family.** The 3rd-swing-within-0.15-ATR rule and the ATR-sized swing filter are exactly the rigour our "2 touches to form, 3rd validates" sketch was missing. This is the closest living relative of the owner's CTrendlineBuilder reference.

### Automating Trading Strategies Part 25 & 34 — Least-Squares / R² Trendline Selection
- **URLs:** https://www.mql5.com/en/articles/19077 (least squares), https://www.mql5.com/en/articles/19625 (R² fit)
- **Mechanism:** Instead of two-point lines, fit a line by **least squares across all candidate touch points** and score the fit with **R²**; a line is valid with ≥ MinTouches touches spaced ≥ MinBarSpacing apart; when several candidates exist, keep the one with most touches, tie-broken by best R². Managed lines live in a storage array with an explicit lifecycle: **expiry check → break check (close beyond line price ± penetration tolerance) → remove or signal**; touch detection uses a separate, tighter tolerance than break detection (two-tolerance design).
- **Verdict: STEAL THE DESIGN for line ranking.** Two-point geometry + 3-swing validation (Part 61) for *existence*, R² over touches for *ranking* competing lines, and the two-tolerance (touch vs break) lifecycle for our line state machine.

### TL Method Indicator — trendline strength scoring + confluence gating (open source)
- **URL:** https://github.com/aramic0/tl-method-indicator
- **Mechanism:** Line **strength = count of validated bounces** (touches meeting minimum-distance and ADR/ATR-normalised thresholds over a very long normalisation window). Signals require a conjunction: price within X ADR of a line meeting minimum strength, **no opposing line too close (a room-to-trade veto)**, net directional strength across ALL lines favouring the trade, and optionally a recently broken opposing line.
- **Verdict: PORT THE GATING IDEAS.** Strength-by-bounce-count parallels our pool touch-counts/strength grade, and the opposing-line proximity veto is the trendline dialect of our room-to-run factor — worth expressing in the trendline family's context gates.

### Trend-reversal trendline EA (MQL5 Article, tutorial-grade)
- **URL:** https://www.mql5.com/en/articles/18259
- **Mechanism:** Two-swing ascending/descending lines; trades **bounces** (touch + immediate confirming candle in line direction) and **break-retests** (break, then price returns to the line from the other side and rejects it); SL from bounce-candle extreme or fixed points.
- **Verdict: INTERESTING BUT WEAK.** The break-retest as a distinct event type is worth having in our lifecycle vocabulary; the implementation is beginner-series code with no validation stats.

---

## 4. Execution / Risk Engineering

### ATS Risk Library — five-class risk include with a 78-check self-test (free CodeBase) ⭐
- **URL:** https://www.mql5.com/en/code/78039
- **Mechanism (precise):** A single `.mqh`, **pure guard layer — it never opens, changes or closes a trade.**
  - **CAtsSizer:** lot from stop distance (fixed lot / % balance / % equity / fixed money). **Lots are floored to the volume step, NEVER rounded up** (rounding up to broker minimum is how "1% risk" EAs quietly risk more); if the allowed risk can't buy the minimum lot it returns **0.0 plus a human-readable `LastReason()`**. Optional lot cap + `OrderCalcMargin` free-margin check.
  - **CAtsDayGuard:** daily loss limit as % of day-start balance and/or money (smaller wins), optional daily profit target, max trades/day, configurable day-start server hour, optional inclusion of floating P/L. **Day-start balance and lock state persist in GlobalVariables — a terminal restart cannot reset the limit.**
  - **CAtsDrawdownGuard:** max total loss, static or trailing-on-equity/balance, optional "floor stops at initial balance"; high-water mark persisted.
  - **CAtsExposure:** counts positions (account / symbol / magic) and **sums the actual money lost if every open stop is hit**; positions with NO stop counted separately; `CanAdd(extraRisk)` answers whether one more trade fits position + total-risk limits.
  - **CAtsFilters:** max spread in points, trading window in server time (**windows crossing midnight handled**), Friday cutoff hour.
  - Ships with a self-test EA that prints PASS/FAIL for **78 arithmetic checks** against fixed numbers mimicking gold, EURUSD, a JPY pair, an index and BTC.
- **Verdict: STEAL THE DESIGN WHOLESALE for the MT5 EA bridge.** Our risk layer already specifies these guards (per SENTINEL specs); this is the cleanest public MQL5 expression of them: reason-string refusals, restart-proof guard state, risk-as-money-at-stops accounting, and a self-test harness. Same author's ATS Prop Rules Monitor (CodeBase) adds a "room left on every rule + largest lot that keeps a stop-out inside today's remaining room" panel — the right UX for the bridge dashboard.

### Equity Guardian "Sentinel" — portfolio-level rule engine (open source)
- **URL:** https://github.com/dbgucci1017/mt5-equity-guardian-trailing-pro
- **Mechanism:** Rules are defined **globally over total floating P/L as % of equity**, not per position: e.g. "when net float reaches +2% of equity, move ALL qualifying positions to breakeven," with per-symbol overrides of the trigger. Trailing supports an **acceleration factor** (trail tightens after a second, higher profit threshold) and asymmetric long/short sensitivity. Drawdown governor halts new entries and **scales out the weakest positions first**; time-based curfew tightens winners and trims losers into low-liquidity hours.
- **Verdict: PORT THE PORTFOLIO-LEVEL MECHANISMS.** Per-position management (our Phase-2 exits) misses basket-level states; global-float triggers, weakest-first scale-out ordering, and trail acceleration are concrete additions for the bridge's trade manager.

### Trailing Stop Manager PRO — feature checklist from the paid tier (closed source)
- **URL:** https://www.mql5.com/en/market/product/155307 (free basic version: https://www.mql5.com/en/market/product/177637)
- **Mechanism (as described):** Six trailing modes — fixed pips, ATR, **Chandelier exit**, MA, Parabolic SAR, candle high/low; 3-level cumulative partial closes; **Profit Lock** (once floating profit arms at a threshold, lock a % of the *peak* profit via a stop that can never move backwards); **virtual SL** (stop tracked by the EA, hidden from the broker); **anti-spread filter** (trailing suspends while spread is abnormal); risk presets incl. a funded-account preset; state persisted across restarts.
- **Verdict: STEAL THE CONCEPTS (Profit Lock + anti-spread suspension especially); the product itself is unverifiable paid code.** Profit Lock is a cleaner formulation of our post-+1.5R trail: lock %-of-peak rather than distance-behind-price.

### Aegis Account Protector / Sentinel News Filter — free utility suite + cross-EA pause flag
- **URL:** https://www.mql5.com/en/blogs/post/772365
- **Mechanism:** Account-wide equity guard flattening on DD/daily-loss/margin/consecutive-loss/weekend triggers with a profit-target lock; a trailing manager attachable to any EA's positions by magic number; a **news filter that pauses trading around high-impact events, draws a countdown, and publishes its pause state in a GlobalVariable that other EAs read** before entering.
- **Verdict: STEAL THE INTER-EA FLAG PATTERN.** For our bridge (brain EA + manager EA + visualiser), a GlobalVariable pause bus is the simplest reliable coordination mechanism inside one terminal. Note GoldenSwing's own news work (Devon briefing calendar) can feed the same flag.

---

## 5. God-Tier Design Worth Stealing

1. **Score/gates separation (Market Intent Engine, §2).** Weighted score for quality + boolean hard gates outside the arithmetic for admissibility. Prevents "no liquidity event but scored high anyway" trades. Directly applicable to detector v2.
2. **`confirmAt` + broken≠swept swing state (Market Intent Engine).** Anti-lookahead as a data-structure property, and the close-through vs wick-through distinction as two booleans instead of one flag — the structure engine and liquidity engine draw opposite conclusions from the same candle.
3. **One analysis path, AutoTrade switch at the end (Market Intent Engine).** Indicator-mode and EA-mode share every line of analysis; only order-sending differs. This is the right skeleton for our EA bridge: the MT5 visualiser and the executing EA are the same code.
4. **CSV signal journal at signal time (ICT Structure Visualizer).** Every signal written to file the moment it appears, from closed bars only — repaint claims become checkable, and the journal doubles as our backtest-vs-live reconciliation feed.
5. **Guard library that cannot trade (ATS Risk Library).** Risk as a pure decision layer returning lots + reason strings, with persisted state and a self-test EA. Plus his demo guard's "watch-only by default, enforce only on explicit switch" posture.
6. **Builder/Manager split for chart objects (owner's CTrendline reference; echoed by Parts 25/34/61 storage-array lifecycles).** Creation/validation in the builder; lifecycle (touch/bounce/break/expire/reconcile orphans) in the manager; direction-aware rendering centralised. Our Python trendline family should keep the same state vocabulary (forming → validated → touch/bounce → broken/expired) so the MT5 port is a translation, not a redesign.
7. **Honesty as a feature (ICT Structure Visualizer + ATS author).** Published negative results, costs shown separately, "study tool, not a signal product" labelling. The only two authors in this entire mining pass who published numbers that make their own work look *worse* — and therefore the only performance numbers in this document treated as evidence.

---

## Bottom line for GoldenSwing

- **Port now:** FVG quality-score weights (FvgGold) as a grading prior; sweep-candle quality gate (close-location + wick share); Fib-61.8% as OB qualifier (context gating); 3-swing trendline validation + R² ranking; ATS risk-guard design + Profit Lock + portfolio-level triggers for the EA bridge.
- **Calibration:** the best independent evidence in the ecosystem (ICT Visualizer, 1,220 gold setups) says mechanical ICT ≈ 0R after costs and regime decides the winning direction — our confirmation discipline and daily-bias veto are aimed at exactly the right failure modes; nothing found here contradicts the Phase-2 design.
- **Junk rate:** as expected, most of the MQL5 gold/SMC space is curve-fit EAs, compiled-only "indicators," and .NET simulators in MQL5 clothing. Nothing with a price tag produced verifiable evidence of an edge.
