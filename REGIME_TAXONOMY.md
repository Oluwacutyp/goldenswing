# XAUUSD Full Regime Taxonomy — Bedrock, Not Ceiling

Built 2026-10-13. Brainstorming made concrete, but NOT the ceiling.
Bedrock is the ground everything stands on. If bedrock lies, everything
on top lies. Time was not the constraint — correctness was.

Engine: `goldenswing/live/regime.py` — `classify_regime(bars)` (pure,
OHLC-only). Tests: `tests/test_regime.py` (12 passed).

## 1. What Bedrock Answers

One question only: **"What is the market's current personality?"**

Bedrock does NOT decide entries, does NOT use FVG / Order Block /
Fibonacci / any entry tool, does NOT score trades, does NOT grant trade
permission. Router, context, entry, process score live on top.

## 2. Regime = Direction × Structure × Volatility × Phase × Session

NOT capped at 5. Five was a starter set, not the real system. Every
real combination gets a routing, or an explicit SIT OUT with a reason.
Unlisted combinations default to SIT OUT / Unclear — never a forced label.

| Dimension | Values | How detected (honest, no entry tools) |
|---|---|---|
| Direction | up / down / flat | Last-2 confirmed swings: HH+HL = up, LH+LL = down, else flat. `evaluate_bias`, swing_n=3, confirmed swings only (no lookahead). |
| Structure | clean_trend / range / compression / breakout / reversal / choppy / unclear | Swing structure + width vs ATR + candle character (see §3). |
| Volatility | low / normal / high / spike / unclear | ATR behaviour: short-window ATR (last 40) vs long-window ATR (last 80). Ratio ≥1.25 = high/expanding, ≤0.75 = low/contracting, last-bar range ≥2.5× long ATR or ratio ≥1.80 = spike. |
| Phase (AMD) | accumulation / manipulation / distribution / transition / unclear | Sweep + displacement facts: sweep + displacement = distribution; sweep, no displacement = manipulation; range/compression + low/normal vol = accumulation; else transition. |
| Session | asia / london / ny / overlap / offhours / unknown | Bar timestamp (UTC): 00–06 Asia, 07–11 London, 12–15 Overlap, 16–20 NY, else Off-hours. |

### 2.1 Direction × Structure — the 18 base regimes

Direction comes from swings; structure from swings + width + candles.

| # | Direction | Structure | What it looks like | Base route |
|---|---|---|---|---|
| 1 | up | clean_trend | HH + HL, strong bodies one way | TREND_PULLBACK |
| 2 | down | clean_trend | LH + LL, strong bodies one way | TREND_PULLBACK |
| 3 | flat | clean_trend | (invalid — trend without direction) | SIT OUT |
| 4 | flat | range | Equal highs/lows, bounded oscillation | RANGE_EDGES / SWEEP (see §4) |
| 5 | up/down | range | (invalid — range has no direction) | SIT OUT / Unclear |
| 6 | flat | compression | Tiny width (≤6× ATR), tiny bodies | SIT OUT — wait expansion |
| 7 | flat | breakout | Range width expanding fast + displacement | BREAKOUT_RETEST |
| 8 | up | reversal | Long downtrend + short sweep + displacement up | SWEEP_DISPLACEMENT_RETRACEMENT |
| 9 | down | reversal | Long uptrend + short sweep + displacement down | SWEEP_DISPLACEMENT_RETRACEMENT |
| 10 | flat | reversal | (invalid — reversal needs a direction) | SIT OUT |
| 11 | flat | choppy | Many wicks (≥55%) + many dojis (≥30%), flat | SIT OUT — no core fits |
| 12 | up/down | choppy | (choppy is flat by definition) | SIT OUT |
| 13 | flat | unclear | Short-window structure unreadable | SIT OUT |
| 14 | up/down | unclear | Short/long direction disagree, no reversal proof | SIT OUT |
| 15–18 | any | any + spike vol | News / shock bar | SIT OUT (all) |

Rows 3, 5, 10, 12 are logically invalid combinations — the engine
never emits them; they are listed so the taxonomy is complete and the
SIT OUT is explicit, not assumed.

### 2.2 Volatility modifiers (apply on top of base route)

| Volatility | Meaning | Modifier |
|---|---|---|
| low | ATR contracting (ratio ≤0.75) | Trend + low = SIT OUT (suspicious trap). Range + low = Compression (SIT OUT, wait expansion). Breakout + low = SIT OUT (false break likely). |
| normal | ATR stable (0.75–1.25) | No modifier. Base route stands. |
| high | ATR expanding (ratio ≥1.25) | Trend + high = tradeable (healthy expansion). Range + high + width expanding = Breakout. Reduce size on Expansion core. |
| spike | Last bar ≥2.5× long ATR or ratio ≥1.80 | SIT OUT always. News/shock. Bedrock never grants normal size in a spike. |
| unclear | ATR unmeasurable (too few bars) | SIT OUT. Not enough data. |

### 2.3 Phase (AMD) modifiers

| Phase | Meaning | Modifier |
|---|---|---|
| accumulation | Range/compression, low/normal vol, no sweep | SIT OUT — wait for sweep. |
| manipulation | Sweep, no displacement yet | Trend + manipulation = SIT OUT (wait displacement). Range + manipulation = SWEEP core armed (wait displacement + retracement). Reversal + manipulation = SIT OUT (wait displacement). |
| distribution | Sweep + displacement, or trend + displacement | Tradeable via routed core. This is the only phase that grants entry permission (via router). |
| transition | Mixed / unreadable phase | SIT OUT. Default when evidence conflicts. |
| unclear | Phase unmeasurable | SIT OUT. |

### 2.4 Session modifiers

Session never creates a trade and never vetoes one alone. It adjusts
confidence and size expectations:

| Session | Character on gold | Modifier |
|---|---|---|
| asia (00–06 UTC) | Quiet, range, accumulation. Sweeps form here. | Expect accumulation/compression. Do not trade quiet moves. |
| london (07–11 UTC) | Expansion. Asia sweep often happens here. First expansion. | Trend/pullback and sweep cores most active. |
| overlap (12–15 UTC) | Highest volume. London + NY. Biggest moves. | Expect expansion. Highest slippage risk. |
| ny (16–20 UTC) | Continuation or reversal of London move. | Trend continuation or reversal cores. |
| offhours (21–23 UTC) | Thin, noisy. | Prefer SIT OUT. Low liquidity. |
| unknown | No timestamp. | No session adjustment. |

Session + regime examples:
- Range + Low Vol + Accumulation + Asia = Compression, wait.
- Range + High Vol + Manipulation + London = Sweep core, trap forming.
- Clean Uptrend + Normal Vol + Distribution + London = Trend-pullback core.
- Choppy + High Vol + Manipulation + Transition = SIT OUT, no core fits.
- Compression + Contracting Vol + Accumulation + Asia = Breakout-retest core, wait expansion.

### 2.5 Full routing table (canonical combinations)

"Core" = the strategy family permitted. Entry still requires
Bias → Location → Trigger → LIMIT at POI, with displacement +
retracement. Routing grants PERMISSION, never an entry by itself.

| Regime (Direction/Structure/Vol/Phase) | Route | Tradeable | Why |
|---|---|---|---|
| Up/Down + Clean Trend + Normal/High + Distribution | TREND_PULLBACK | Yes | Clean trend, healthy vol, displacement done |
| Up/Down + Clean Trend + Low + any | SIT OUT | No | Compressed ATR in trend = trap |
| Up/Down + Clean Trend + Spike + any | SIT OUT | No | News/shock |
| Up/Down + Clean Trend + Normal/High + Manipulation | SIT OUT | No | Sweep without displacement yet |
| Flat + Range + Normal + Distribution | SWEEP_DISPLACEMENT_RETRACEMENT | Yes | Range + distribution after sweep |
| Flat + Range + Normal + Manipulation | SWEEP_DISPLACEMENT_RETRACEMENT | Yes (armed) | Wait displacement + retracement |
| Flat + Range + Normal + Accumulation | SIT OUT | No | Wait for sweep |
| Flat + Range + Low + any | Compression → SIT OUT | No | Range + low vol = compression |
| Flat + Range + High + Distribution | BREAKOUT (if width expanding) else SWEEP | Yes | Expanding range = breakout forming |
| Flat + Compression + any + any | SIT OUT | No | Wait expansion |
| Flat + Breakout + Normal/High + Distribution | BREAKOUT_RETEST | Yes | Wait retest, never chase break |
| Flat + Breakout + Low + any | SIT OUT | No | False break likely |
| Up/Down + Reversal + Normal/High + Distribution | SWEEP_DISPLACEMENT_RETRACEMENT | Yes | Counter prior move after displacement |
| Up/Down + Reversal + any + Manipulation | SIT OUT | No | Wait displacement |
| Flat + Choppy + any + any | SIT OUT | No | No core fits |
| Flat + Unclear + any + any | SIT OUT | No | Structure unreadable |
| Any + any + Spike + any | SIT OUT | No | Spike always sits out |
| Any + any + any + Transition (structure unclear) | SIT OUT | No | Mixed evidence |

Any combination not in this table defaults to SIT OUT / Unclear.
That is deliberate: an unlisted combination is evidence the taxonomy
is incomplete, not permission to trade it.

## 3. Bedrock Honesty Rules (do not weaken)

1. **Structure leads.** If swing structure is unclear, regime is
   Unclear. No indicator, candle, or pattern overrides structure.
2. **Two windows.** Short (last 40 bars) = current personality.
   Long (last 80 bars) = real shift or noise. Short/long direction
   disagreement defaults to Unclear — EXCEPT the reversal case
   (long trend + short sweep + short displacement opposite), which
   gets a Reversal label at modest confidence (0.45). Without sweep +
   displacement, disagreement stays Unclear.
3. **Volatility must confirm.** Trend + compressed ATR = suspicious.
   Range + very low ATR = Compression. Sudden ATR expansion after
   quiet = Expansion.
4. **Candle character is the truth serum.** Strong bodies one way
   support Trend. Dojis / small bodies / long opposing wicks argue
   against it. Last 20 bars. Candle shift BEFORE displacement is an
   early warning — confidence must drop, or structure must leave
   Clean Trend.
5. **Permission to say "I don't know."** Transition / Unclear is a
   successful decision, not a failure. Forcing a label when mixed is
   how bedrock starts lying. Sitting out counts as success.
6. **No circular logic.** Regime is readable with FVGs, OBs, Fibs,
   and all entry concepts removed. Entry tools live upstairs.
7. **Early, not late.** First break in structure behaviour, candle
   character change before displacement, ATR starting to expand /
   contract, HTF/LTF divergence — whispers, not shouts. Waiting for
   multiple confirmed swings / completed patterns / aligned indicators
   is late and useless.
8. **Confidence gate.** Computed confidence < 0.35 forces Unclear,
   even if a label was computed. Low confidence = do not trust label.

Working test: "If I can only use structure + recent candle character +
ATR behaviour, can I already say what regime we are in with reasonable
confidence? If not → Unclear, stay out."

## 4. Tools Are Dual-Role (not locked to one job)

The SAME tool is a SENSOR in bedrock and CONFIRMATION in the trigger
stage. Locking a tool to one job is as dumb as locking it to entry only.

| Tool | As sensor (bedrock) | As confirmation (trigger stage) |
|---|---|---|
| Engulfing candles | Many engulfings one way = trending/expansion | Engulfing at POI after sweep = trigger confirmation |
| Pin bar / rejection wick | Frequent long wicks both sides = choppy | Pin bar at pre-marked POI after sweep = rejection confirmation |
| Doji | Many dojis = range/compression/indecision | Doji at POI = pause, wait next candle, never enter alone |
| FVG | FVGs forming constantly = displacement regime | FVG at POI = entry zone on retracement |
| Order Block | OBs forming = displacement regime | OB at POI = entry zone |
| Fibonacci | Fib clusters = confluence zones forming | Fib 62–79% (OTE) inside a core zone = precise entry area |
| Chart patterns (flags, triangles) | Clean flags = trend continuation regime; coiling triangles = compression | Pattern at POI + displacement = context, never standalone entry |
| ATR | Expanding/contracting/stable = volatility regime | Stop-size context, never trigger |
| RSI / Bollinger / ADX / EMA | RSI pinned = trend; oscillating = range; BB width = compression/expansion; EMA separation = trend strength | Context only, never entry trigger |

Rule: a tool becomes entry confirmation ONLY at a pre-marked POI,
AFTER bias + location + displacement, and ONLY if regime routed to
that core. Same tool in the middle of nowhere = noise.

## 5. What Sits on Top of Bedrock (priority order)

1. **Bedrock** (this file + `live/regime.py`) — honest first.
2. **Router** — regime → core mapping (§2.5). Grants permission,
   never an entry.
3. **Context layer** — HTF bias, liquidity pools, Volume Profile,
   session character, confluence quality. Upgrades/downgrades a
   setup the router already allowed. Never creates a trade alone.
4. **Process score** — Did we read regime right? Route right?
   Respect Bias → Location → Trigger → LIMIT? Measured separately
   from P&L. Process-only, never outcome-based (outcome-based
   scoring teaches avoidance of good setups that happened to lose).
5. **Memory** — What works in this regime on gold? What bleeds?
   How long do regimes last? Historical character, not curve-fitting.
   Comes later, once live data exists.
6. **Adaptation** — When regime shifts, system shifts with it
   instead of forcing the old core. Last. Dangerous if bedrock lies.

Veto stays OFF (2026-10-13 backtest: hard veto mostly hurt, removed
winners, did not reduce blow-ups). Bedrock does not implement veto.
Do NOT re-add veto as a bedrock filter.

## 6. Sanity Check on Real Data (not proof)

110 rolling windows (every 100 bars, 200-bar lookback) over
`data/GCF_H1_2y.csv` (11,426 H1 bars):
- Clean Trend up 30, down 35 (59% trend) — high because 2y gold was
  mostly bullish; in-sample observation, not a regime prior.
- Compression 15, Range 18, Choppy 2, Unclear 10.
- Routes: TREND_PULLBACK 63, SWEEP_DISPLACEMENT_RETRACEMENT 16,
  SIT OUT 31 (28% sit-out rate — bedrock says "I don't know" often,
  as designed).
- Last window: flat/range/normal/distribution/ny, conf 0.65,
  route SWEEP_DISPLACEMENT_RETRACEMENT.

This is a smoke test that the engine runs and produces plausible
labels. It is NOT validation, NOT out-of-sample, NOT live proof.

## 7. What's Still Weak

- **Regime detection in real time is murky.** Distinctions obvious
  in hindsight (trend vs big swing in a range, accumulation vs
  quiet chop) are hard live. Bedrock will be wrong sometimes. The
  Unclear default is the mitigation, not a cure.
- **No out-of-sample validation.** No walk-forward split. No
  forward paper. No live proof. The taxonomy is a construction,
  not a validated system.
- **No regime → PnL evidence yet.** We have not measured whether
  routing by this taxonomy actually improves win% / PF / MaxDD vs
  no-routing. That measurement is the next step, not assumed.
- **Reversal detection is early and fragile.** It relies on sweep +
  displacement in the short window. False reversals (sweep +
  displacement that fails) will happen. Confidence 0.45 reflects that.
- **Breakout vs expanding range is ambiguous.** The 55% width rule
  is a construction, not a discovered threshold. Needs sensitivity
  testing.
- **Choppy threshold (wicky ≥55%, doji ≥30%) is a construction.**
  Needs sensitivity testing on real gold data.
- **Session boundaries are approximate.** London 07–11, Overlap
  12–15, NY 16–20 UTC are conventional, not gold-specific. Gold
  may have different effective session edges.
- **Confidence formula is a construction.** Weights (+0.1 agreement,
  +0.15 strong bodies, −0.2 doji/wicky, etc.) are reasoned, not
  fitted. Do NOT treat confidence as a probability.
- **Live paper loop untouched.** Backtest/paper only. No live switch.
- **Veto OFF.** Do not re-enable veto through the back door as a
  "regime filter" without the same before/after evidence veto failed.
