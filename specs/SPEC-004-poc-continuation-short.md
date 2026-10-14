# SPEC-004: POC Continuation — Short

## Conventions

Same as SPEC-001. Mirror of SPEC-003.

---

## GATE 1 — Macro Permission (D1)

```
IF daily_bias == -1 → TRUE
ELSE → FALSE
```

## GATE 2 — H1 Trend Alignment

```
IF ema_stack_H1 == -1 → TRUE
ELSE → FALSE
```

## GATE 3 — Value Zone Defined (2+ confluence)

```
vp = VolumeProfile(High[1..100], Low[1..100], Close[1..100], V[1..100])
poc = vp.poc

impulse_high = MAX(High[1..50])
impulse_low  = MIN(Low[1..50])
move = impulse_high - impulse_low
IF move < 2.0 * ATR[0] → FALSE
fib_618 = impulse_low + 0.618 * move   (retracement up into zone)

poc_near = ABS(poc - fib_618) <= 0.3 * ATR[0]
IF poc_near → TRUE, zone_mid = (poc + fib_618) / 2
ELSE → FALSE

zone_top = zone_mid + 0.3*ATR[0], zone_bottom = zone_mid - 0.3*ATR[0]
```

## GATE 4 — Price in Zone (at bar T)

```
IF Low[0] <= zone_top AND High[0] >= zone_bottom → TRUE
ELSE → FALSE
```

## GATE 5 — RSI Confirmation (at bar T)

```
Divergence (bearish):
  swing_highs = last 2 fractal swing highs in High[1..50]
  IF High at sh[2] > High at sh[1]
     AND rsi_14 at sh[2] < rsi_14 at sh[1]
     → divergence = TRUE

Exhaustion:
  was_overbought = MAX(rsi_14[0..10]) > 80.0
  curling_down   = rsi_14[0] < rsi_14[2]

IF divergence OR (was_overbought AND curling_down) → TRUE
ELSE → FALSE
```

## GATE 6 — Rejection Trigger (at bar T)

```
wick = High[0] - zone_top
rejection_wick = wick > 0.3 * ATR[0]
closed_inside  = Close[0] < zone_top
bearish_close  = Close[0] < Open[0]

IF rejection_wick AND closed_inside AND bearish_close → TRUE
ELSE → FALSE
```

## GATE 7 — Entry, Stop, Invalidation

```
ENTRY:  entry = Close[0]
STOP:   stop = zone_top + 1.5 * ATR[0]

INVALIDATION:
  IF Close[i] > zone_top for any i >= 0 after entry → EXIT IMMEDIATELY

TARGETS:
  target_1 = vp.val
  target_2 = impulse_low
  target_3 = 1.618 extension below
```

## POSITION SIZING

Same as SPEC-003 (1.0% base, 1.5% on 3-factor).

---

*Status: SPEC. Mirror of SPEC-003.*
