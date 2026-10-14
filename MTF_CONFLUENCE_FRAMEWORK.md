# GoldenSwing Multi-Timeframe Confluence Framework

## Version 1 – Clean Multi-Timeframe Rules (Current Tools)

### 1. Higher-Timeframe Bias (Daily or 4H)

```
function get_htf_bias():
    if Daily/4H EMA stack is bullish (9 > 21 > 50)
        or price is making clear higher highs + higher lows
        → bias = BULLISH

    else if Daily/4H EMA stack is bearish (9 < 21 < 50)
        or price is making clear lower highs + lower lows
        → bias = BEARISH

    else
        → bias = NEUTRAL   // no trade allowed
```

**Rule**: Only allow trades in the direction of `bias`. If `NEUTRAL` → skip everything.

### 2. Intermediate Timeframe Location Filter (H1)

```
function is_valid_location(direction):
    if direction == LONG:
        - H1 EMA stack is bullish (9 > 21 > 50)
        - Price is pulling back into or touching a bullish sr_flip level
        - The pullback has not broken the most recent significant higher-low
        - Current score ≥ 65

    if direction == SHORT:
        - H1 EMA stack is bearish (9 < 21 < 50)
        - Price is pulling back into or touching a bearish sr_flip level
        - The pullback has not broken the most recent significant lower-high
        - Current score ≥ 65

    return True only if all conditions above are met
```

### 3. Entry Trigger

```
function entry_trigger(direction):
    if direction == LONG:
        - Price shows rejection of the sr_flip level
          (close back above, or strong bullish close with rejection wick)
        OR
        - Break of the most recent minor bearish structure on H1/M15

    if direction == SHORT:
        - Opposite conditions
```

### 4. Final Signal Gate

```
function generate_signal():
    bias = get_htf_bias()
    if bias == NEUTRAL:
        return None
    direction = bias
    if not is_valid_location(direction):
        return None
    if not entry_trigger(direction):
        return None
    return Signal(...)
```

### 5. Hard Rules
- Never trade against HTF bias.
- Never trade if H1 EMA stack not aligned.
- Never trade with score < 65.
- Never trade not at valid sr_flip location.
- Risk 0.5–1% per trade.
- Stop beyond structural invalidation.

---

## Version 2 – Professional Multi-Timeframe Rules (Fib + Volume Profile)

### 1. Higher-Timeframe Bias
Same as Version 1.

### 2. Define High-Quality Zones (Confluence Engine)

A zone needs **at least 2** of 3 factors:

```
function is_high_quality_zone(direction, price_area):
    confluence_count = 0

    // 1. Structure / S-R Flip
    if price_area contains valid sr_flip level in correct direction:
        confluence_count += 1

    // 2. Fibonacci
    if price_area overlaps key Fib level (50%, 61.8%, 78.6%):
        confluence_count += 1

    // 3. Volume Profile
    if price_area overlaps POC, HVN, or relevant VAH/VAL:
        confluence_count += 1

    return confluence_count >= 2
```

### 3. Intermediate Timeframe Location Filter (H1)

```
function is_valid_location(direction):
    - H1 EMA stack aligned (or not strongly opposed)
    - Price inside/reacting from high-quality zone (2+ confluence)
    - Pullback hasn't broken HTF structure
    - Score ≥ 60–65
```

### 4. Lower-Timeframe Trigger (15M or 5M)

```
function entry_trigger(direction):
    - Clear rejection of the zone, OR
    - Break of minor counter-trend structure on 15M/5M, OR
    - Displacement candle away from zone in bias direction
```

### 5. Final Signal Gate

```
function generate_signal():
    bias = get_htf_bias()
    if bias == NEUTRAL: return None
    direction = bias
    if not is_valid_location(direction): return None
    if not entry_trigger(direction): return None
    return Signal(
        direction = direction,
        entry = market on trigger or limit in zone,
        stop = beyond confluence zone,
        targets = [next VP level, Fib extension 1.272/1.618, prior structure],
        score = current_score
    )
```

### 6. Hard Rules for Version 2
- HTF bias is absolute boss.
- Minimum 2-factor confluence required.
- Prefer 3-factor when available.
- Stop beyond entire confluence zone.
- Risk 0.5–1%.
- Fallback to Structure + Fib if VP unreliable (still needs 2 factors).

---

## Implementation Roadmap

1. **Stabilize Version 1**: Fix hierarchical mode tuning, resolve +9.7% vs +5.5%
   discrepancy, fix inverted candle validator.
2. **Build Fibonacci module**: Retracement levels from impulse legs, extensions
   for targets.
3. **Build Volume Profile module**: POC, HVN/LVN, Value Area from tick volume.
4. **Build Confluence Engine**: 2+ factor zone scoring.
5. **Upgrade to Version 2**: Layer confluence on top of stable Version 1.

---

*Spec authored 2026-10-12. Version 1 maps to `hierarchical_mode` in detector.
Version 2 requires new `fibonacci` and `volume_profile` modules.*
