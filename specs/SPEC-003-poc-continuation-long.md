# SPEC-003: POC Continuation — Long

## Conventions

Same as SPEC-001. `T` = last closed bar. Binary gates.

---

## GATE 1 — Macro Permission (D1)

```
IF daily_bias == +1 → TRUE
ELSE → FALSE
```

## GATE 2 — H1 Trend Alignment

```
IF ema_stack_H1 == +1 → TRUE
ELSE → FALSE
```

## GATE 3 — Value Zone Defined (2+ confluence)

```
INPUT:  H1 bars T-100..T-1, tick volume V[1..100]

STEP 3a — Volume Profile:
  vp = VolumeProfile(High[1..100], Low[1..100], Close[1..100], V[1..100])
  poc = vp.poc

STEP 3b — Fibonacci:
  impulse_high = MAX(High[1..50])
  impulse_low  = MIN(Low[1..50])
  move = impulse_high - impulse_low
  IF move < 2.0 * ATR[0] → FALSE (no valid impulse)
  fib_618 = impulse_high - 0.618 * move

STEP 3c — Confluence check:
  poc_near = ABS(poc - fib_618) <= 0.3 * ATR[0]
  IF poc_near → TRUE, zone_mid = (poc + fib_618) / 2
  ELSE → FALSE

OUTPUT: zone_mid, zone_top = zone_mid + 0.3*ATR[0], zone_bottom = zone_mid - 0.3*ATR[0]
```

Rationale: 2-factor confluence required. POC + Fib 61.8% overlapping.
No overlap = no zone = no trade.

## GATE 4 — Price in Zone (at bar T)

```
IF Low[0] <= zone_top AND High[0] >= zone_bottom → TRUE
ELSE → FALSE
```

Rationale: price must be interacting with the zone on bar T.

## GATE 5 — RSI Confirmation (at bar T)

```
rsi_14 = RSI(Close, 14)   (Wilder's, evaluated at T)

STEP 5a — Divergence:
  swing_lows = last 2 fractal swing lows in Low[1..50]
  IF Low at swing_low[2] < Low at swing_low[1]
     AND rsi_14 at swing_low[2] > rsi_14 at swing_low[1]
     → divergence = TRUE

STEP 5b — Exhaustion (fallback):
  was_oversold = MIN(rsi_14[0..10]) < 20.0
  curling_up   = rsi_14[0] > rsi_14[2]
  exhaustion   = was_oversold AND curling_up

IF divergence OR exhaustion → TRUE
ELSE → FALSE
```

## GATE 6 — Rejection Trigger (at bar T)

```
wick = zone_bottom - Low[0]           (how deep into zone)
body = ABS(Close[0] - Open[0])
rejection_wick = wick > 0.3 * ATR[0]
closed_inside  = Close[0] > zone_bottom
bullish_close  = Close[0] > Open[0]

IF rejection_wick AND closed_inside AND bullish_close → TRUE
ELSE → FALSE
```

Rationale: long lower wick into the zone, close back above zone bottom,
bullish body. All binary.

## GATE 7 — Entry, Stop, Invalidation

```
ENTRY:  entry = Close[0]

STOP:   stop = zone_bottom - 1.5 * ATR[0]
        (below entire zone, not just the candle)

INVALIDATION:
  IF Close[i] < zone_bottom for any i >= 0 after entry → EXIT IMMEDIATELY
  (Zone failed. Institutions did not defend.)

TARGETS:
  target_1 = vp.vah          (Value Area High, next resistance)
  target_2 = impulse_high    (origin of the Fib move)
  target_3 = 1.618 extension (impulse_high + 0.618 * move)
```

## POSITION SIZING

```
risk_pct = 1.5 IF (3-factor confluence) ELSE 1.0
(3rd factor = sr_flip level within 0.3 ATR of zone_mid)

Same formula as SPEC-001.
```

---

## EVALUATION ORDER

```
1. GATE 1 (D1) → FALSE = stop
2. GATE 2 (H1 stack) → FALSE = stop
3. GATE 3 (zone 2+ confluence) → FALSE = stop
4. GATE 4 (price in zone) → FALSE = stop
5. GATE 5 (RSI) → FALSE = stop
6. GATE 6 (rejection) → FALSE = stop
7. Compute entry/stop/size → emit
8. Invalidation every bar after entry
```

---

*Status: SPEC. Not yet implemented.*
