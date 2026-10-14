# Sentinel GOD-CONVICTION-STRATEGY.pine — mining notes

**Source:** `~/workspace/sentinel/pine/GOD-CONVICTION-STRATEGY.pine` (514 lines,
MPL-2.0, "© OMNI-GOD ∞"). A `strategy()` backtest twin of a confluence
indicator, written with explicit honesty rules: next-bar-open fills, broker
commission + slippage as inputs, no parameter optimized on the backtest
("Ship defaults only"). **Note:** despite the owner's memory, the script
contains NO Bollinger logic.

## What it is

A 7-family confluence scorer with hard gates, a beyond-FVG state machine,
trap intelligence, a cost gate, a daily circuit breaker, and a fixed-R:R
exit structure. Claims #66–#76 in the ledger.

## Interesting deltas vs GoldenSwing

1. **Bait is penalized, not just absent.** Inside an un-reclaimed gap the
   script doesn't merely withhold zone points — it subtracts 8 (#68). Our
   gating is binary (in/out); we have no negative evidence weights.
2. **The beyond-FVG state machine (#67) is stricter than our gap handling.**
   Entries are blocked while price sits inside a gap that hasn't been
   reclaimed (close back inside after a failure). We track FVG mitigation
   but never gate entries on gap-interior location.
3. **Fixed-R:R exits are a complete philosophy (#71), not a fallback.**
   TP1 1.5R (50% off, stop→BE) / TP2 3R / 1.2-ATR trail, with a
   peak-R-based lock after TP1. Our exits are pool-targeted. The choch_fvg
   sleeve showed fixed-2R beating OB targets — this family of exits is
   not dead on arrival and deserves a head-to-head against Phase-2 exits.
4. **Trap memory (#72).** A sweep + volume spike (or a failed breakout)
   marks a trap bar that stays active for 12 bars and scores toward the
   reversal. Our sweep is an event; theirs is an event with a half-life.
5. **Cost gate (#69) and circuit breaker (#70) belong to infra, not
   strategy.** The cost gate (TP1 must clear round-trip costs net) is a
   cleaner formulation than anything in our risk layer; the max-trades/day
   breaker is a different mechanism from our loss limits.
6. **Crisp zone definitions (#75).** Impulse = body > 1.2×ATR + close
   beyond prior extreme + prior bar opposite color; zone = the impulse
   bar's full range. Crisper than our OB construction; worth an A/B on
   zone quality (bounce/hold rates).

## What to test first (backtest queue)

1. **#71 vs Phase-2 exits** — fixed-R:R exit structure head-to-head on the
   anchor signals. One variable (exits), everything else constant.
2. **#67 bait block** — add "no entries inside un-reclaimed gaps" as a
   gate on the anchor book; measure dilution vs protection.
3. **#72 trap memory** — SFP+volume trap as a scored reversal family.
