# GoldenSwing — Owner Rulings (authoritative)

This file is the tiebreaker. Where any other document, code comment, or
older rule in this repository disagrees with a ruling below, **this file
wins** and the other text is stale. Each ruling names its date and what it
supersedes, so no future pass has to re-derive the owner's decisions — or
silently restore a rule he already revoked.

## 1. Trader sizing doctrine (owner ruling 2026-10-24)

A setup is account-agnostic: size scales with balance/equity. The engine
never abandons a trade merely because the calculated size falls below the
broker minimum.

- **Formula.** `lots = max(VOLUME_MIN, quantize_down((k × E × C × S) /
  (stop_ticks × tick_value), step))`, where `k = cfg.risk_pct` (default
  0.01) is a coefficient, never a gate; `E` is equity; `C` is setup
  confidence; `S` is the setup factor.
- **The only hard number is the broker minimum: 0.01 lot (XAUUSD).**
  A step-quantized size below the minimum is traded AT the minimum,
  flagged `min_override`, with the estimated stop exposure journaled on
  every such record. If the stop risk exceeds the balance, the engine
  still trades the minimum — it does not walk away.
- **No score band.** `score.total` is a ranking total, not confidence.
  There is no score multiplier anywhere in the sizing path (removed
  2026-10-24 at both order paths). `C` is held at ×1.0 pending a real,
  honest confidence field at order time — that field does not exist yet
  and must not be faked from the score.
- **Retired, do not restore.** `decline_below_min` (retired 2026-10-24);
  zero lots when the risk budget cannot buy the broker minimum; `risk_pct`
  as a refusal gate or as a hard range constraint (the old `(0, 0.25]`
  rule); any fixed-lots mode (see §2).
- **Standing guards (unchanged by this ruling).** 15% maximum drawdown
  from peak = the sole account halt (latched); `max_positions` 3;
  `max_consecutive_losses` 6 (latched); friction gate (spread ÷ stop >
  0.10 refuses); `VOLUME_LIMIT` and nonpositive declines stand.
- **Analytical verdicts are identical at every balance.** Balance
  changes size, never the setup's verdict.
- **Demo authority stays OFF. No real-money routing.** The RiskPool
  ceiling stands.

## 2. Fixed-lots mode revoked (owner ruling 2026-10-23, revoking A13)

There is NO fixed-lots mode and no `fixed_lots` field anywhere in the
engine. Dynamic risk sizing (`live/sizing.py`) is the SOLE sizing path;
the legacy float path (`execution/risk.py::size_position`, same risk
rule) remains selectable as `sizing_mode="legacy"` for A/B forensics
only. There is no `--fixed-lots` command-line flag. Any document, table,
or example still showing `--fixed-lots`, a `fixed_lots` config key, or a
fixed-lot demo command is stale — see §5.

## 3. Forecast / ensemble: shadow until validated (owner ruling 2026-10-24)

- The ensemble forecasts BIG MOVES before they happen, horizons 1 day
  and above, from actual market data.
- It stays SHADOW / RESEARCH until validated out-of-sample with a
  locked holdout (baseline-first, chronological walk-forward, no
  exceptions). The judgment core stays shadow-first under the same rule.
- Ensemble outputs are uncalibrated ranking scores. Cite them as RANKS
  ("big-move rank high @1d and @3d"), never as probabilities.
- **No bundled model ships.** The engine carries the training code and
  trains on its own MT5 data on install / first run, retraining as its
  own history grows (owner correction 2026-10-25). No external hourly
  frame is the training source, and no pre-trained artifact is ferried
  between machines.

## 4. Chart studies and terminal (owner direction 2026-10-24, build order)

1. Ensemble live wiring as shadow/research + calibration/validation
   honesty (the main build).
2. Saturday calendar fix (the engine carries its own XAUUSD calendar;
   known-closed periods render `closed_expected`, never a breach;
   feed-vs-calendar disagreement surfaces explicitly).
3. Full chart studies (≤4 real overlays, exactly what the reasoning
   used — zones, Fib, and all) attached to signals/alerts, not just the
   minimal phone chart.
4. Interactive multi-timeframe terminal (switch timeframes like
   TradingView, on the engine's own timestamped data).

The visual layer must never invent evidence the engine never calculated.

## 5. Superseded text (explicit — do not follow these where they disagree)

- `docs/EC2_MT5_RUNBOOK.md`: the `--fixed-lots 0.01` demo command and
  the `fixed_lots` `/config` row — dead flag, dead mode (corrected
  in-repo; the banner on that file points here).
- `SYSTEM_SCOPE_REVIEW.md` §5.1: "answer is 0 lots" for an unaffordable
  broker minimum, and "`risk_pct` must be in (0, 0.25]" — both retired
  by §1 (corrected in-repo; the banner on that file points here).
- `goldenswing/execution/mt5_runner.py` sizing comment (2026-10-23
  wording): the "budget that cannot buy the broker minimum DECLINES"
  clause — corrected in-repo to the §1 floor rule.
- Any score-band sizing, `decline_below_min` verdict, or fixed-lots
  sizing found anywhere else: retired, not an alternative.

## 6. How this file is enforced

`tests/test_owner_rulings.py` pins the load-bearing parts: no
`--fixed-lots` flag on the real CLI, no below-minimum abandonment, no
score parameter in the sizing API, and this file's presence with its
supersession section. The older doctrine pins live in
`tests/test_sizing_min_floor.py` (minimum floor), and balance-independence
plus the 15% drawdown halt are pinned in `tests/test_phase2_fixes.py`.
If a future change breaks one of these pins, the change is wrong, not
the pin — unless the owner issues a new ruling that updates this file.
