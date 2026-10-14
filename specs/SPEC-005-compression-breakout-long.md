# SPEC-005: Compression Breakout — Long

## Conventions

Same as SPEC-001. `T` = last closed bar. Binary gates.

---

## GATE 1 — Macro Permission (D1)

```
IF daily_bias == +1 → TRUE
ELSE → FALSE
```

Note: breakout direction must agree with D1. No counter-trend breakouts.

## GATE 2 — Compression Defined

```
INPUT:  High[1..30], Low[1..30], ATR[1..30]

STEP 2a — Range:
  range_high = MAX(High[1..30])
  range_low  = MIN(Low[1..30])
  range_size = range_high - range_low

STEP 2b — ATR contraction:
  atr_now  = ATR[0]
  atr_20ago = ATR[20]
  contracting = atr_now < 0.7 * atr_20ago

STEP 2c — Range tightness:
  tight = range_size < 3.0 * atr_now

STEP 2d — Boundary touches (at least 2 each side):
  upper_touches = COUNT(i in [1..30] WHERE High[i] >= range_high - 0.2*atr_now)
  lower_touches = COUNT(i in [1..30] WHERE Low[i]  <= range_low  + 0.2*atr_now)

IF contracting AND tight AND upper_touches >= 2 AND lower_touches >= 2 → TRUE
ELSE → FALSE

OUTPUT: range_high, range_low
```

Rationale: all four conditions binary. No "looks compressed."

## GATE 3 — Breakout (at bar T)

```
displacement = Close[0] - Open[0]
broke_out    = Close[0] > range_high
strong_body  = displacement > 0.8 * ATR[0]
not_wick     = (High[0] - Close[0]) < 0.3 * ATR[0]

IF broke_out AND strong_body AND not_wick → TRUE
ELSE → FALSE
```

Rationale: close beyond boundary, strong body, minimal upper wick.
A wicky break is not a breakout.

## GATE 4 — Volume Expansion (at bar T)

```
vol_now = V[0]
vol_avg = MEAN(V[1..30])

IF vol_now > 1.5 * vol_avg → TRUE
ELSE → FALSE
```

Rationale: institutional commitment requires volume. No volume = no breakout.

## GATE 5 — Entry, Stop, Invalidation

```
ENTRY:
  Option A (aggressive): entry = Close[0]
  Option B (conservative): wait for retest of range_high, entry on rejection
  DEFAULT: Option A

STOP:   stop = range_high - 0.5 * (range_high - range_low)
        (midpoint of the compression range)

INVALIDATION:
  IF Close[i] < range_high for any i >= 0 after entry → EXIT IMMEDIATELY
  (Re-entry into range = failed breakout. Same rule as lecture.)

TARGETS:
  measured = range_high - range_low
  target_1 = range_high + 1.0 * measured
  target_2 = range_high + 1.5 * measured
```

## POSITION SIZING

```
risk_pct = 1.0 (breakouts are lower win-rate, keep size standard)
Same formula as SPEC-001.
```

---

## EVALUATION ORDER

```
1. GATE 1 (D1) → FALSE = stop
2. GATE 2 (compression) → FALSE = stop
3. GATE 3 (breakout at T) → FALSE = stop
4. GATE 4 (volume) → FALSE = stop
5. Compute entry/stop/size → emit
6. Invalidation every bar after entry
```

---

*Status: SPEC. Not yet implemented.*
