# GoldenSwing — Full-Scope System Review

**Date:** 2026-10-13 · **Repo:** `~/workspace/goldenswing` (local only)
**Purpose:** One document the owner can read to review the whole engine:
architecture, every strategy and its exact rules, every sensor, entry /
management / exit, risk, paper loop, journal, Telegram surface — and what
is still weak, unproven, or untested.
**Status of this doc:** descriptive review only. It changes no behaviour.
No performance number in here was re-run for this review; every number is
quoted from the repo module / doc it was read in, and attributed as such.
Where the code was ambiguous or a claim could not be verified from the code
read, that is stated explicitly instead of guessed.

**How this doc was verified.** Each section was grounded by reading the
named module(s) in this repo: `goldenswing/features/` (incl. `config.py`,
`phases.py`, `gating.py`), `goldenswing/playbooks/` (all five),
`goldenswing/live/multitf.py` + `live/regime.py` (bedrock),
`goldenswing/reader/session_timeframes.py`,
`goldenswing/confluence.py`, `goldenswing/execution/` (risk, manager,
paper, mt5, runner, watchdog), `goldenswing/alerts/commands.py`,
`scripts/paper_live.py`, `goldenswing/backtest/` (harness, mtf12,
real_trader, exness12), `goldenswing/godce/`, `goldenswing/modes/`,
`goldenswing/golden/detector.py`, `goldenswing/system/regime.py`,
`SYSTEM.md`, `SPEC.md`, `STRATEGIES_AND_LOGICS.md`, `BACKTEST2.md`.
Cross-checks against `SYSTEM.md` §5/§10/§12 and `CONFLUENCE.md` are
flagged where the docs disagree with code defaults.

---

## 0. One-paragraph summary

GoldenSwing is a single XAUUSD (GC=F / XAU_USD) system with one market
read (reader → feature engine), several entry playbooks ranked by a
confluence engine, one shared exit machine (Phase-2 swing exits), one
risk-guard layer, and three execution paths: backtest, paper loop (live
OANDA H1 feed, paper money — the only live path currently running), and
an MT5 live path that is code-complete but **never run against a real
terminal** (`SYSTEM.md` §10.3). The anchor playbook
(`sweep_reversal`) is in-sample-validated (+28.92%, per its docstring /
`SYSTEM.md`) but **fails walk-forward OOS** (−0.28%, PF 1.01, 94 trades),
so in the paper loop the anchor stays **shadow-only** (alert + journal,
no fill) while the two sleeves `fvg_midline` and `trendline_retest` are
the auto-paper-traded set. Two new pieces landed 2026-10-13 and are
unvalidated: the AMD missed-manipulation fallback, the QM M5 playbook
(shadow-only), and the DST-aware session→timeframe map (tags-only).
The hard veto is **OFF by policy** — but see §5.5 for the important
default-vs-policy gap in the code.

---

## 1. Full system architecture

### 1.1 Pipeline (data → decision → fill → journal)

```
Feed (OANDA v20 H1 XAU/USD, or yfinance GC=F legacy / replay)
  → MarketReader (goldenswing/reader/reader.py)
      structure, liquidity pools, sweeps, FVGs, OBs, displacement,
      daily bias (D1), EMA stack, sessions
  → FeatureEngine (goldenswing/features/engine.py)
      13+ feature families attached per bar (see §3)
  → Detector (goldenswing/golden/detector.py)  [anchor path]
      OR Playbook.live_signal() per playbook    [live incremental path]
  → ConfluenceEngine (goldenswing/confluence.py)
      3 gates (session scope → required-context tripwire →
      hunting-ground exclusions) + score = base + bonuses
  → Risk (goldenswing/execution/risk.py: RiskGuards + size_position)
  → Executor: PaperExecutor (paper) | MT5Executor (live, unproven)
      + ExitStepper (goldenswing/execution/manager.py) — one exit machine
  → TradeJournal (goldenswing/execution/learning.py) + Telegram alerts
      + monitor / divergence checks (goldenswing/live/monitor.py)
```

Key design rules enforced in code:

- **Iron rule (playbooks/__init__.py):** a playbook keeps the EXACT
  entry/exit machinery it was validated with. Mixing a trigger from one
  playbook with exits from another is a new experiment, not a refactor.
- **Context contract (playbooks/base.py):** each playbook declares the
  context where it is genuinely good; a candidate firing outside its
  declared context is skipped as a `context-violation` and journalled
  as a SYSTEM BUG, never traded.
- **Confluence never re-simulates** (base.py): playbooks return
  pre-simulated candidates; the engine only ranks and books.
- **One position at a time** in the batch book; live loop: first-wins
  per bar (SYSTEM.md §6.2). `RiskConfig.max_positions = 3` is the
  guard-layer cap (see §5.1) — the two caps coexist and the tighter
  effective behaviour in the paper loop is first-wins / one-at-a-time;
  this doc could not verify from the runner code alone which cap binds
  first in every live path, so treat "one position" as the paper-loop
  norm and `max_positions=3` as the guard ceiling.

### 1.2 Modules

| Layer | Module(s) | Job |
|---|---|---|
| Feed | `feed/oanda.py`, `feed/poll.py` | OANDA v20 practice H1 (live); yfinance GC=F legacy; ReplayFeed |
| Reader | `reader/reader.py`, `reader/state.py`, `reader/indicators.py` | Per-bar MarketState: swings, BOS/CHoCH, pools, sweeps, FVG, OB, displacement, D1 bias, EMA stack |
| Features | `features/engine.py` + 13 family modules | Bollinger, Fib, Ichimoku, RS, patterns, zones, phases (AMD), footprints, volume, MTF, trendlines, divergence, AOC, traps |
| Detector | `golden/detector.py` | Anchor signal generation + scoring (Phase 2 + Phase 3/4/5 opt-ins) |
| Playbooks | `playbooks/*.py` (5) | Self-contained entry systems (§2) |
| Confluence | `confluence.py`, `contextmap.py` | Ranking, bonuses, 3-gate enforcement, hunting grounds |
| Context / routing | `system/regime.py`, `modes/routing.py`, `modes/config.py` | Regime routing (D1+H1), phase→playbook caps, mode configs |
| Bedrock regime | `live/regime.py` | Direction × Structure × Volatility × Phase × Session (§1.4) |
| Multi-TF engine | `live/multitf.py` | Bias→zone→trigger hierarchy, hard-veto function, session attach |
| Session map | `reader/session_timeframes.py` | DST-aware session/killzone → entry-clock map, tags-only (§2.6) |
| Forecast | `forecast/` | Multi-horizon big-move ML; bonus/sizing input, NEVER a veto |
| GOD-CE | `godce/` | Pine-port conviction engine + trap intelligence (§1.5) |
| Execution | `execution/runner.py`, `executor.py`, `paper.py`, `mt5.py`, `manager.py`, `risk.py`, `watchdog.py`, `state_store.py`, `learning.py`, `reporting.py`, `callbacks.py` | LiveRunner, executors, ExitStepper, guards, watchdog, state, journal |
| Alerts | `alerts/` | Telegram (+inline buttons), email, outbox, narrator, commands |
| Backtest | `backtest/harness.py`, `mtf12.py`, `real_trader.py`, `exness12.py`, `validation.py`, `ablation.py` | Harness v1/v2, MTF, real-trader, Exness $12 model |
| Experiments | `experiments/` + `scripts/experiment_*.py` | One experiment per claim; sleeves import their constants from here |
| V2 confluence | `v2_confluence/` | Zone clustering, FVG tracker, Fib, RSI divergence, volume profile, lecture sizing |

### 1.3 The confluence engine (`confluence.py`)

- **Scoring:** `confluence = base_score + feature bonuses + map bonuses
  + trap-boost + forecast bonuses`. Bonus sizes (`ConfluenceConfig`):
  AOC backing +4.0, divergence +3.0, imbalance magnet (unfilled FVG
  within 0.5 ATR) +2.0, forecast 1d+3d agreement +2.0, single-horizon
  +1.0, trap-boost +4.0 (playbook-gated). Session / regime / HTF /
  phase bonuses come only from each playbook's *measured* context map
  (`hunting_ground.bonuses`), never from priors.
- **Gates (live):** `gate_live_signal` — (1) session scope,
  (2) playbook required-context tripwire, (3) hunting-ground exclusions.
  A gate failure is blocked + journalled, never executed.
- **Risk modes:** `"enforce"` (live/paper loop — a denied trade is
  skipped) vs `"shadow"` (backtest — risk decisions computed + logged
  but the historical sequence is never altered, so the anchor canary
  stays exact).
- **Forecast is never a veto** (`confluence.forecast_bonuses`):
  agreement earns a bonus / sizing input only.
- **Cost gate** exists (`cost_gate_ok`) — this review did not verify
  every call site, so do not assume it gates every live path.

### 1.4 Bedrock regime (`live/regime.py`) — the "personality" layer

Bedrock answers ONE question: "What is the market's current
personality?" It does **not** decide entries, use FVG/OB/Fib/any entry
tool, score trades, or grant trade permission. Router, context, entry,
and process score live on top of it.

- **Inputs:** OHLC bars only + bar timestamp for session. No entry
  tools inside bedrock (honesty rule 6: no circular logic).
- **Windows:** short 40 bars (current personality), long 80 bars (real
  shift vs noise), candle character last 20 bars.
- **Regime = Direction × Structure × Volatility × Phase × Session**
  (a combination, not one of 5 buckets; unlisted combinations default
  to SIT OUT / Unclear — never a forced label):
  - Direction: up / down / flat
  - Structure: clean_trend / range / compression / breakout / reversal
    / choppy / unclear
  - Volatility: low / normal / high / spike / unclear (ATR ratios:
    expanding ≥1.25, contracting ≤0.75, spike ≥1.80, short vs long ATR)
  - Phase (AMD): accumulation / manipulation / distribution /
    transition / unclear
  - Session: asia / london / ny / overlap / offhours / unknown
    (fixed-UTC labeller in this module — see §2.6 for the newer
    DST-aware map that does NOT replace this one)
- **Honesty rules (verbatim from the module):** structure leads; short
  and long windows must agree on direction (disagreement → Transition /
  Unclear); volatility must confirm structure (trend + compressed ATR
  is suspicious; range + very low ATR is Compression, not tradeable
  Range); candle character is the truth serum (strong-body fraction,
  doji fraction — thresholds `STRONG_BODY_MIN=0.55`,
  `DOJI_MAX=0.15`); permission to say "I don't know"; early, not late
  (first break in behaviour, whispers not shouts).
- **Tools are dual-role:** the SAME tool is a SENSOR in bedrock and
  CONFIRMATION in the trigger stage (e.g. engulfing everywhere =
  trending sensor; engulfing at a pre-marked POI after sweep +
  displacement = entry confirmation).
- **Veto stays OFF** in bedrock (module docstring, 2026-10-13
  backtest note). Bedrock does not implement veto.

### 1.5 GOD-CE (GOD Conviction Engine, `godce/`) — Pine port + trap intel

- Faithful port of `GOD-CE-SLIM.pine` (Standard risk profile).
  ASOC self-optimisation deliberately OFF by default.
- **Params (`godce/params.py`):** fast 9 / med 21 / slow 50 / trend 200
  / HMA 50 / ATR 14 / ADX 14 (min 18) / z 20 / vol 20; MACD 12/26/9;
  RSI 14 (OB 70 / OS 30); Ichimoku + H&S on; Wyckoff / vol-div /
  dark-pool on; `baseThresh=70.0`, `minFactors=4`, `slMult=1.0`,
  `trailATR=1.2`, TP 1.5R/3.0R, `riskPct=1.0`, Hydra on,
  move-BE-at-50% on, profit-lock on, BE offset 0.10R, spike shield +
  circuit breaker on, `maxDailyLossR=3.0`, `maxTradesDay=4`,
  `modeDwell=2`, `spikeAtrMult=2.2`, trap engine on, `costBps=4.0`,
  cost gate on, `minNetRR=0.6`, warmup 600 bars, fill-at-close.
- **Router (`godce/router.py`):** candidate mode from a Pine if/else
  chain → FLAT / POST-SPIKE / BREAKOUT / MOMENTUM / MEAN-REV, with a
  2-bar dwell. Per-mode rules set allow-long/short/counter, TP1/2/3,
  SL cap, runner, threshold offset (e.g. MEAN: TP capped 0.75/1.5,
  SL cap 1.5 ATR, offset −5; BRK: TP 1.0/2.0, SL cap 1.5, offset −10).
- **Scoring (`godce/scoring.py`):** 0–100 conviction + distinct-factor
  counts; dynamic threshold = clip(`baseThresh` + adjustments, 55, 92).
- **Status in the live system (important):** GOD-CE is NOT the live
  decision engine. Its live contribution verified in code is the
  **trap-boost bonus** (+4.0, playbook-gated, signal-bar convention)
  via `confluence.trap_bonuses` reading `features/traps.py` (18 boolean
  columns), plus the standalone `godce/backtest.py` harness. Do not
  describe GOD-CE as "the engine that trades" — the playbooks +
  confluence trade; GOD-CE contributes trap intelligence and a
  separate backtest.
- **Trap features (`features/traps.py`):** 18 booleans incl. sweep /
  fake-breakout / inversion / FVG / liquidity-grab patterns; trap
  memory 12 bars, break lookback 6, trap vol mult 1.15.

### 1.6 Modes (`modes/config.py`, `modes/routing.py`, `modes/runner.py`)

One market read, three execution modes. A mode changes only: bar
timeframe, entry filters, exit style/horizon. Costs are identical
($0.45/oz round trip in every mode, on purpose — cost-as-R damage at
small TFs stays visible). Each mode's harness run is independent; no
state shared between modes. `GOLDEN_THRESHOLD = 60.0` in every mode.

| Mode | TF | Period | Killzone-only | Kinds | Min disp | Exit | Horizon |
|---|---|---|---|---|---|---|---|
| Swing | H1 | 2y | No | all | 0 | swing_v2 | 72 bars (~3d) |
| Scalp | M15 | 60d | Yes | all | 0 | nearest_pool | 8 bars (2h) |
| Fast scalp | M5 | 60d | Yes | continuation only | \|disp_3\| ≥ 1.0 ATR | nearest_pool | 6 bars (30m) |

Phase-5 phase→playbook routing (`modes/routing.py`): a route table
measured from a shadow book caps a playbook's score below golden in
phases where it measured badly (`cap_for(phase, playbook)`). The table
is DATA, never a hardcoded belief; when no table is attached, behaviour
is exactly Phases 2–4.

### 1.7 The other regime router (`system/regime.py`)

A second, simpler router (do not confuse with bedrock §1.4): regime
from D1 bias + H1 EMA stack + H1 trend → STRONG_BULL / BULL_LEAN /
STRONG_BEAR / BEAR_LEAN / CHOP. Strong = trade with trend, min score
60, aggressive; lean = same direction only, kinds limited to
{sweep_reversal, continuation, sr_flip}, min score 65, not aggressive;
CHOP = stand aside (allowed directions ∅, min score 100 = impossible).
`hierarchical_filter` is the strict post-filter (D1 assigns direction,
H1 stack must align, score ≥ 65 default) — docstring claims "+7.6%"
proven; this review did not re-verify that figure.

### 1.8 Forecast (`forecast/`)

- Multi-horizon big-move ML: horizons 12 / 24 / 72 / 168 / 360 bars,
  each with its own label/model/evaluation. Label: favourable
  excursion ≥ fav×ATR before adverse excursion ≥ adv×ATR inside the
  window.
- Deployment side (`forecast/api.py`): per-bar ensemble scores + ranks
  per horizon; top-decile citations for alerts.
- Role: **quality dial only** — confluence bonus / sizing input, never
  a veto (see §1.3). The runner loads a bundle via `ForecasterScorer`
  when a bundle path is configured; without a bundle there is no
  forecast input (this review did not verify whether the live paper
  loop currently has a bundle wired — treat forecast as optional,
  not guaranteed live).

---

## 2. Every strategy and its logic

### 2.0 Playbook roster (all five)

| Playbook | File | live flag | live_threshold | Status |
|---|---|---|---|---|
| `sweep_reversal` (anchor) | `playbooks/sweep_reversal.py` | `True` | 60.0 | Live-capable, but SHADOW in the paper loop (see §2.1) |
| `fvg_midline` | `playbooks/fvg_midline.py` | `True` (paper-live) | 0.0 (every fill graded) | Auto-paper-traded (exec_set) |
| `trendline_retest` | `playbooks/trendline_retest.py` | `True` (paper-live) | 0.0 (every retest graded) | Auto-paper-traded (exec_set) |
| `fvg_limit` | `playbooks/fvg_limit.py` | *see note* | — | Signals only; NOT in exec_set (shadow until forward paper proves it) |
| `qm_m5` | `playbooks/qm_m5.py` | `False` (shadow-only) | — | Shadow-only, unvalidated (§2.5) |

Note on `fvg_limit.live`: the playbook module was read for its trade
machine; its exact `live` flag value was not re-confirmed line-by-line
for this review. What IS confirmed from `runner._get_playbooks` is
that it is wired for SIGNALS but explicitly NOT in the auto-paper
`exec_set`. Treat it as shadow in practice regardless of the flag.

All sleeves carry `SLEEVE_BASE_SCORE = 60.0` as a **documented
placeholder, NOT a measured score** (their docstrings say so
explicitly). Only the anchor's base score is a real detector score.

### 2.1 Anchor — `sweep_reversal` (Phase-2 detector, unchanged)

> "The Phase-2 detector, unchanged … This playbook IS the live book;
> its machinery is byte-identical to the validated anchor (+28.92%,
> 199 trades, PF 1.25)." — playbook docstring, quoting BACKTEST2
> in-sample. See §6 for the OOS contradiction.

- **Detector (`golden/detector.py`, `DetectorConfig`):**
  `emit_floor=35.0`, `golden_threshold=60.0`, `sweep_fresh_bars=8`,
  `min_gap=3` bars between emitted signals, `min_room_atr=1.2`
  (room to nearest opposing pool), stop-distance band
  `min_stop_atr=0.25` / `max_stop_atr=3.5`, `level_near_atr=0.35`.
  Anchor config: families `COMBO_A | {"aoc"}`,
  `require_sweep_confirmation=True`, `daily_veto=True`.
- **Setup priority (`_setup`):** liquidity sweep is the strongest
  backing — but with `require_sweep_confirmation`, a sweep only
  becomes a setup once CONFIRMED by displacement + a structure shift
  in the reversal direction within `confirm_window=8` bars; an
  unconfirmed sweep falls through to the continuation path.
  Trend-aligned displacement continuation is the other setup.
- **Gates in `evaluate()`:**
  1. EMA-stack alignment (`require_stack_alignment` when enabled):
     direction must match the 9/21/50 stack; mixed stack → nothing.
  2. Daily-bias veto (`daily_veto`, anchor ON): trades against D1
     bias blocked, except the documented exception — a confirmed sweep
     reversal at a major daily level (`daily_level_tol_atr=1.5`).
     This is the **detector-level D1 veto**, which is ON and is
     distinct from the hard veto (§5.5), which is OFF.
  3. Room gate: `room_atr ≥ 1.2` or no signal.
  4. Stop-distance gate: `0.25 ≤ stop_atr ≤ 3.5` or no signal.
  5. Score: structure + liquidity + displacement + room + regime +
     session + Phase-3 family scores, capped at 100; Phase-4 MTF cap
     (both HTFs opposed → cap 59.0, below golden); Phase-5 route-table
     cap. Below `emit_floor=35` → no signal.
- **Context:** validated on XAUUSD H1 (GC=F, Oct 2024–Oct 2026).
  Session-agnostic (`session_scope="any"`). Hunting ground from the
  measured 199-trade map: no exclusions; bonuses newyork +2.0,
  expanded-vol +2.0, Friday +2.0.
- **Exits:** Phase-2 swing exits (§4.2).
- **Live status:** `live=True` in code, but `LiveRunner` keeps the
  anchor **shadow** in the paper loop ("anchor stays shadow (alert +
  journal, no fill)"). The in-sample/OOS gap (§6) is why.

### 2.2 Sleeve — `fvg_midline` (LuxAlgo sleeve, PAPER-LIVE)

Validated as a candidate sleeve (`experiments/CHoCH_FVG_MID_OB.md`):
**fresh CHoCH → resting limit at the midline of a fresh, untraded FVG
→ fixed-2R exit (+25.88%, PF 1.41, 97 trades, positive in both
halves)** — quoted from the playbook docstring; not re-run here.

- **Entry:** fresh CHoCH (age ≤ `CHOCH_FRESH_BARS = 24` at setup bar),
  FVG paired within `FVG_PAIR_BACK_BARS = 8`, resting limit at FVG
  midline, order expiry `ORDER_EXPIRY_BARS = 24`. Fill = first bar
  whose wick touches the midline.
- **Stop:** far side ± `STOP_BUFFER_ATR = 0.25` ATR. Stop-distance
  band enforced: `MIN_RISK_ATR = 0.25` … `MAX_RISK_ATR = 4.0`.
- **Target/exit:** fixed `FIXED_TP_R = 2.0`. The opposing-OB exit
  variant diluted and is NOT ported.
- **Live gate:** at the fill bar the CHoCH must be ≤
  `MAX_CHOCH_AGE_AT_FILL = 48` bars old (24 fresh + 24 expiry);
  a fill beyond that envelope was never validated. Stop band
  re-verified as tripwire.
- **Instruments:** GC=F, XAUUSD. Base score placeholder 60.0.

### 2.3 Sleeve — `trendline_retest` (PAPER-LIVE)

Claim #31 (Element Of Trading Vol V, user-mined PDFs):
**break of a respected trendline → price spends 2 bars away → first
return touch within 12 bars → enter in the breakout direction,
structural stop beyond the retest extreme, Phase-2 exits. +15.40%,
PF 1.43, 61 trades, −0.01 daily-R correlation with the anchor book**
— quoted from the playbook docstring; not re-run here.

- **Scanner (`experiments/trendline_claims.py`):**
  `TL_TOL_ATR = 0.20` (touch/violation tolerance),
  `RETEST_REARM_BARS = 2` (bars price must spend away before another
  touch registers), `RETEST_WINDOW = 12` (return touch must come
  within 12 bars), one trade per break. A touch needs wick reaching
  the line with close holding the breakout side.
- **Gates (non-score, re-verified as tripwire):** hard daily-bias veto,
  room to the first opposing pool, stop-distance band
  `MIN_STOP_ATR = 0.25` … `MAX_STOP_ATR = 3.5`.
- **Deliberately NOT inherited:** the JeaFx zone-retest kill —
  zones and trendlines are different mechanics.
- **Caveat in its own docstring:** H1 soft (+1.2%) vs H2 strong
  (+14.0%) — regime sensitivity noted, not a gate. 61 trades are too
  thin for session verdicts; hunting ground has no exclusions, no
  proven cells.
- **Exits:** Phase-2 swing exits (§4.2).

### 2.4 Sleeve — `fvg_limit` (limit-at-touch FVG, signals only)

Validated in `scripts/test_fvg_entry_ab.py` on OANDA XAUUSD H1
2024-10→2026-10, all zones both directions, exactly as tested
(quoted from the playbook docstring):

| Entry rule | Result |
|---|---|
| Limit-at-touch, fill needs ≥0.10 ATR edge penetration | +32.0R, PF 1.07 |
| Limit-at-touch, fill needs ≥0.05 ATR edge penetration | +106.1R, PF 1.21 |
| Confirmation entry (wait for rejection) | −71.4R, PF 0.82 |

- **Entry:** resting limit at the NEAR edge (bullish: zone top;
  bearish: zone bottom). Each zone traded at most ONCE (first touch).
  FVGTracker fed closed bars; no repainting, strictly causal.
- **Fill realism:** a touch bar fills ONLY if it penetrates the edge
  by ≥ `MIN_FILL_PENETRATION_ATR = 0.10` ATR (conservative end of the
  validated 0.05–0.10 band). A graze that reverses is a miss, not a
  fill. Fill price = the limit edge.
- **Stop:** far extreme ± `STOP_BUFFER_ATR = 0.25` ATR (catastrophic
  backstop).
- **Invalidation:** close through the zone → exit immediately at
  close.
- **Target:** fixed `TARGET_R = 1.5`. Trades unresolved after
  `MAX_HOLD_BARS = 48` are DROPPED (as in the validation script — no
  timeout exits were booked).
- **Risk:** 1R per trade; results in R multiples.
- **Live-path gap (stated in the docstring):** the paper executor
  fills at next bar's open, not at the limit edge — the loop
  convention. The batch path grades edge fills. The parity test proves
  the SIGNAL sets are identical; the fill-price gap between edge and
  next-open is measured by the live book, not assumed away.
- **OPEN (not part of validation):** the long-only + bull-EMA-stack
  gate from the hybrid engine, and the long/short asymmetry measured
  elsewhere. Adding either is a new experiment.
- **Status:** signals + journal only; NOT in the paper loop's
  `exec_set` until forward paper proves it.

### 2.5 NEW — QM M5 (`playbooks/qm_m5.py`, commit e2752d8, SHADOW-ONLY)

Quasimodo reversal on M5. **No measured validation in this repo — no
backtest, no win rate, no context map.** Base score is a documented
placeholder (60.0), NOT a measured score. `live=False`.

**Pattern (bearish; bullish is the mirror):**

1. HEAD — swing high H0 (the higher high).
2. NECKLINE — pullback swing low L1 (the higher low).
3. QM LEVEL — rally swing high H2 that FAILS below the head
   (the lower high; H2 < H0 by a genuine margin, not an equal high).
   Late buyers are trapped at H2.
4. BREAK — a CLOSE below the neckline L1 (break of structure).
5. RETRACEMENT — price returns to the QM level H2. Entry is the
   resting LIMIT at H2: no retracement, no trade (doctrine: "entry
   AT POI only, never above it").

**Exact parameters (constants in the module):**

| Parameter | Value | Meaning |
|---|---|---|
| `SWING_N` | 3 | Confirmed-fractal pivot lookback (M5 trigger swing_n); a pivot at bar *i* is usable only once bar *i*+3 has closed (no lookahead) |
| `ATR_N` | 14 | ATR window (same formula as `fvg_limit`) |
| `MIN_LEG_ATR` | 0.5 | Each QM leg must span ≥ 0.5 ATR or it is noise, not structure |
| `MIN_FAIL_ATR` | 0.25 | Failed rally must fall short of the head by ≥ 0.25 ATR (a genuine failure, not an equal high) |
| `STOP_BUFFER_ATR` | 0.25 | Stop buffer beyond the head (repo standard structural buffer) |
| `MIN_TARGET_R` | 1.0 | Neckline target must pay ≥ 1.0R or the setup is skipped, not re-targeted |
| `MAX_PATTERN_BARS` | 120 | Head → break must come within 120 bars (~10h on M5, one session) |
| `MAX_RETRACE_BARS` | 48 | Break → first touch of the QM level |
| `MAX_HOLD_BARS` | 48 | Max hold after the touch (`fvg_limit` precedent) |
| `DISPLACEMENT_BODY_ATR` | 1.0 | Break-bar body ≥ 1.0× ATR counts as displacement (Phase-2 rule) |
| `MIN_STOP_ATR` / `MAX_STOP_ATR` | 0.25 / 3.5 | Stop-distance band re-verified by `check_live_signal` |
| `SLEEVE_BASE_SCORE` | 60.0 | Placeholder, NOT measured |

**Entry / stop / target / invalidation:**

- Entry: resting limit at the QM level on the FIRST touch after the
  break (one trade per pattern).
- Stop: head extreme ± 0.25 ATR (raw head = invalidation).
- Invalidation: a CLOSE beyond the head before the touch (pattern
  dead), or a close beyond the head while working the trade.
- Target: the broken neckline level (practitioner target: "the
  structural level that price broke"). Only qualifies if the neckline
  sits ≥ 1.0R from the QM level.
- Simulation order (batch): buffered stop first (harness convention —
  stop wins same-bar ties), then close-beyond-head invalidation at the
  close, then the neckline target. Timeout exits at the close, booked
  (not dropped).

**Confluence TAGS (never gates, never vetoes):**

- `sweep_head` — the head wicked above the prior swing high (the head
  leg itself was the sweep).
- `displacement_break` — the break bar's body is ≥ 1.0× ATR.

**Pivot machinery:** confirmed fractal pivots, `swing_n=3`;
consecutive same-side pivots keep only the more extreme one (MQL5
zig-zag rule); the QM level freezes at the break bar. One state
machine (`_QMEngine`) backs BOTH batch and live paths — parity by
construction.

**Limits / what is NOT claimed:**

- QM is NOT head-and-shoulders: a QM has no second shoulder.
- No M5 live feed yet: the live paper loop is H1-only, so QM has no
  live feed. Promotion to live needs forward paper observation on a
  real M5 feed.
- `VALIDATED_INSTRUMENTS = ("XAUUSD", "GC=F")` is the feed universe,
  NOT a validation claim — QM is unvalidated on every instrument.
- QM keeps its OWN M5 entry clock; the session map annotates but never
  overrides it (§2.6).

### 2.6 NEW — Timezone / session→timeframe mapping (`reader/session_timeframes.py`, commit a33c2b7, TAGS-ONLY)

**Problem it solves.** The repo has THREE fixed-UTC session labellers
(`reader/sessions.py`, `live/regime.py`, `backtest/mtf12.py`), and
`reader/sessions.py` says it outright: "DST ignored — documented
approximation." A fixed UTC hour is wrong twice a year: the ICT
killzones (London 07:00–10:00 UTC, NY 12:00–15:00 UTC in
`STRATEGIES_AND_LOGICS.md` §2.5 / `backtest/real_trader.py`) are the
WINTER values of wall-clock windows. After the US springs forward the
same NY wall clock is one UTC hour earlier, so a fixed-UTC killzone
fires an hour late all summer — and wrong in both directions during
the March / October–November gap weeks when the US and UK are on
different DST states.

**Design decision:** this module is the DST-aware replacement for NEW
code. The three legacy labellers are deliberately NOT changed (tests
pin them; backtest comparability depends on them). Nothing here
re-labels history.

**Session definitions (wall-clock in the exchange's own IANA zone,
`zoneinfo` applies DST — no fixed UTC offset anywhere):**

- Asia opens 09:00 Asia/Tokyo (no DST → always 00:00 UTC).
- London 07:00–16:00 Europe/London.
- New York 07:00–16:00 America/New_York.
- Partition: asia [Tokyo open, London open) → london [London open,
  NY open) → overlap [NY open, London close) → newyork [London close,
  NY close) → off [NY close, Tokyo open). In northern winter this
  reproduces the repo buckets EXACTLY (Asia 00–07, London 07–12,
  Overlap 12–16, NY 16–21, Off 21–24 UTC). In summer every boundary
  shifts with its own clock (e.g. NY open 07:00 EDT = 11:00 UTC,
  London close 16:00 BST = 15:00 UTC → summer overlap 11:00–15:00
  UTC). Gap weeks fall out of the same rule.

**Killzones (ICT, anchored to NY wall clock — the ICT origin):**

- London KZ: 02:00–05:00 America/New_York (winter 07:00–10:00 UTC;
  summer 06:00–09:00 UTC).
- NY KZ: 07:00–10:00 America/New_York (winter 12:00–15:00 UTC;
  summer 11:00–14:00 UTC).
- Killzone is an independent flag, not a session: NY KZ sits at the
  start of the overlap bucket; London KZ sits inside the London bucket
  (02:00 NY == 07:00 London except in gap weeks, when it is off by
  one hour — documented, not fudged).

**Timeframe mapping — selects the ENTRY CLOCK only:**

| Field | Value | Notes |
|---|---|---|
| `bias_tf` | H4 always | Intraday bias clock; session can never flip bias |
| `zone_tf` | H1 always | POIs live on H1 |
| `trigger_tf` | M15 inside a killzone, M30 elsewhere | Peak-liquidity vs range-building clock |
| `active_tfs` | (bias, zone, trigger) | |
| `entry_clock` | trigger, unless playbook owns its clock | QM M5 keeps M5 |

**Contract (standing rules in the module):**

- Tags / context ONLY. NEVER flips bias, NEVER vetoes, NEVER gates
  a trade. The mapping has **no direction field at all**.
- Pure functions, no state, no cursors: same timestamp → same answer.
- Naive timestamps treated as UTC (repo reader convention).
- A clock, not a market calendar: weekends/holidays still map (the
  market is closed, but the clock is not). Do not use it to decide
  whether the market is open.

**Composition with AMD + QM:**

- AMD phase carried through as a tag (`amd:<phase>`), never re-derived
  or changed. Manipulation inside a killzone earns the descriptive
  tag `manipulation_in_killzone` — a tag, never a gate.
- `compose(..., playbook="qm_m5")` reports `playbook_entry_clock="M5"`
  with the note that the session mapping annotates the QM signal but
  never overrides its clock. QM is still session-agnostic until
  measured.

**Wiring:** `live/multitf.py evaluate(asof=)` → `result["session"]`
via `compose(asof)`. A signal that fires without `asof` is identical
apart from the added `"session"` key.

**Limits (explicit):** NO measured session edge — annotation/context
only, **no backtest behind the M15/M30 split**. The M15-in-killzone /
M30-elsewhere choice is a design decision from ICT liquidity
reasoning, not a measured result. Do not quote it as "M15 wins in
killzones" — nothing measured that.

### 2.7 NEW — AMD missed-manipulation fallback (`features/phases.py`, commit cb4ea35)

The AMD phase machine classifies what the market is *doing* (states,
not predictions). The fallback fixes the case where the direct
manipulation check misses.

**Base machine (direct path):**

- A **base** exists when the trailing `amd_range_bars = 48` range is
  narrow (width ≤ `amd_base_width_atr = 5.0` ATR) with little net
  drift (≤ `amd_base_drift_atr = 2.0` ATR). Range extremes freeze at
  base entry.
- Base labelled by the move that led INTO it (`amd_prior_bars = 48`,
  `amd_prior_move_atr = 1.5`): after a decline → **accumulation**;
  after a rise → **distribution**; otherwise **consolidation**.
- **Manipulation (direct):** a liquidity sweep through a base extreme
  THIS bar (`sweep_age == 0`) whose extreme aligns with the frozen
  base extreme (±0.25 ATR): spring in accumulation / upthrust in
  distribution when the swept side matches the base's story
  (otherwise `manipulation_ssl` / `manipulation_bsl`). Confidence
  `amd_direct_confidence = 0.80`. Episode memory
  `amd_manip_bars = 10`.
- **Phase exit → expansion:** close beyond a base extreme by ≥
  `amd_exit_buffer_atr = 0.3` ATR with displacement-strength body
  (`bull_disp_3` / `bear_disp_3 ≥ 1.0`). Confidence 0.70. Expansion
  matures into trend_up / trend_down after
  `amd_expansion_bars = 24`. A fresh base can form out of a trend.
- Manipulation resolves by displacement out of the range (→
  expansion) or by drifting back inside for `amd_manip_bars` (→ back
  to the base label). Base confidence 0.60.

**Missed-manipulation fallback (NEW):**

- **Trigger (all three legs required, while still in a base):**
  1. BOS/CHoCH event with `structure_event_age ≤ amd_fallback_bos_age = 3`
     bars;
  2. recent same-direction sweep with `sweep_age ≤
     amd_fallback_sweep_age = 10` bars;
  3. fresh same-direction FVG, created within
     `amd_fallback_fvg_age = 10` bars.
- **Effect:** event `manipulation_fallback`, `via="fallback"`,
  confidence `amd_fallback_confidence = 0.55` (vs 0.80 direct).
  Same exit logic as direct manipulation — one state machine.
- **Why no pool leg:** a standing pool alone was measured to fire on
  72 noise bars vs 18 for the sweep leg per 3000 real XAUUSD H1 bars
  (pools cluster at base extremes by construction). Pools are NOT a
  fallback leg.
- **Safety:** fallback can never label distribution without
  displacement + exit (it only enters the manipulation state; the
  base label is preserved underneath). While in a fallback
  manipulation episode, if the three legs still hold in the SAME
  direction, the episode refreshes in place (bounded: all three legs
  are aging windows, so the episode still expires) instead of
  dropping back to the base label for one bar and letting the legs
  age out.
- **Status:** implemented + unit-tested (per commit). NOT separately
  backtested as an edge — the fallback's effect on trade outcomes has
  not been measured. Its confidence 0.55 is a judgement parameter,
  not a calibrated probability.

### 2.8 Multi-TF hierarchy engine (`live/multitf.py`) — the structural spine

The SAME structural engine runs fractally on every preset. Rules
(docstring, verified in `_evaluate`):

1. HTF bias ≠ 0 required (last closed swing structure).
2. Trigger direction must equal the bias (counter-trend triggers are
   rejected as `trigger_against_bias`).
3. Entry AT the POI: inside the band, never beyond it
   (`entry_above_poi` / `entry_below_poi` reject).
4. Trigger event = BOS / MSCH / CHoCH on the trigger TF (swings from
   bars[:-1] so the trigger bar can never confirm its own reference
   swing).
5. Stop = adverse structural invalidation + buffer
   (`breakout_atr = 0.1` × ATR). Never tightened.
6. Target = nearest opposing zone-TF pool, 2R floor (if the nearest
   pool pays ≥ 2R, it becomes the target; else target = entry ± 2R).
7. Missing veto TF = fail-open, never a block (when veto enabled).

**Presets:**

| Preset | Bias | Zone | Trigger | swing_n | Hard-veto TF(s) |
|---|---|---|---|---|---|
| SWING | D1 | H4 | H1 | 3 | W1 |
| INTRADAY | H4 | H1 | M15 | 3 | D1 |
| SCALP | H1 | M15 | M5 | 3 | H4, + D1 (second veto) |
| SCALP_M1 | M5 | M5 | M1 | 5 | H1 |

Trigger-bar displacement gate (`TimeframeConfig`):
`displacement_body_mult = 1.5` (body > 1.5× avg body over 20 bars),
`atr_mult = 0.5` (body > 0.5× ATR14), `body_range_min = 0.6`
(body/range ≥ 0.6).

**Hard veto (`check_hard_veto`, defined here):** veto TF bias
(`evaluate_bias`: last two confirmed swings each side) is OPPOSITE
sign to main bias AND both swing legs are strong (legs ≥ 0.25× ATR14
on the veto TF — per `tests/test_hard_veto.py`). Flat / agreeing /
weak-opposite veto TFs pass. `main_bias == 0` → never vetoed.
See §5.5 for the OFF policy vs the code default.

### 2.9 Real-trader backtest (`backtest/real_trader.py`) — the doctrine harness

Replaces the textbook `mtf12` logic (next-bar-open chase, fixed 2R
floor, fixed 72-bar time stop, 4-condition displacement gate,
`amd_phase="unknown"`, rough sessions). All causal (closed bars only):

- HTF veto, never democracy. Bias ≠ 0 required.
- Sweep of liquidity (Asia high/low, PDH/PDL, range high/low) →
  displacement → retracement to FVG/OB → LIMIT fill AT POI.
  Patient limit, never market-chase. Gap out of POI = SKIP.
  No retracement = NO TRADE.
- Displacement = strong body closing beyond swing + FVG created.
  Confirmation, not a 4-condition textbook gate that kills all trades.
- AMD: Accumulation = range, Manipulation = sweep, Distribution =
  expansion after sweep. Only Manipulation → Distribution trades.
- Killzones (fixed-UTC legacy labels): London 07–10 UTC, NY 12–15
  UTC, Asia 00–06 UTC as range. No expansion trades in deep Asia
  unless clear manipulation + displacement.
- Stop = structural invalidation (sweep extreme + buffer) with
  max-stop gate: stop > `MAX_STOP_ATR = 2.5` × ATR14 → REJECT, do not
  take absurd stops.
- Target = opposing liquidity pool (PDH/PDL, Asia high/low, zone-TF
  swing) with ≥ `MIN_TARGET_R = 1.5`. Nearest pool < 1.5R → SKIP.
- Exit = stop (wins ties), target, opposing structural break,
  emergency time stop only (not primary). `EMERGENCY_BARS`:
  M1 300 / M5 200 / M15 120 / H1 100 / H4 60 / D1 30.
  Limit patience: `PATIENCE_BARS = 20` trigger bars.
- Risk (in this harness): $0.12/trade (1% of $12), position = risk /
  stop_distance. Daily loss limit: stop after −2R/day. No compounding.
  Margin at 50× reported separately; infeasible if > balance.
- Results quoted in `STRATEGIES_AND_LOGICS.md` (not re-run here):
  Swing $12 → $424.50 in 2y, 93 trades, 73.1% win, PF 1.79,
  0 blow-ups. Intraday/Scalp blow a single $12. M1 loses.
  **Treat these as doc-quoted, not independently verified by this
  review.**

### 2.10 MTF12 backtest (`backtest/mtf12.py`) — the textbook harness

The older harness the real-trader harness replaced: `evaluate()`-based
signals, stop wins ties, then target, then a 72 trigger-TF-bar time
stop. Cost `$0.45/oz + slippage`, `R_net = R_gross − cost / risk`.
Two ledgers (see §5.2): fractional-unit R accounting + the
`oanda_1unit_feasible` flag. Keep for comparability; do not present
its fractional results as executable.

### 2.11 Exness $12 model (`backtest/exness12.py`) — the small-account model

See §5.2 for full risk rules. Engine: same real-trader logic
(sweep → displacement → retracement → LIMIT at POI), re-run under
Exness Standard MT5 facts instead of OANDA 50:1 / $0.45 cost.

### 2.12 GOD-CE standalone backtest (`godce/backtest.py`)

Pine-faithful backtest of the GOD-CE engine (fill at confirmed bar
close, 600-bar warmup). Separate from the playbook books; used for
GOD-CE validation and trap-intel measurement, not for the live book.

### 2.13 Entry validator (`entry/validator.py`) — candle confirmation

HTF bias → H1 level (S/R flip, OB, pool) → CANDLES validate →
limit-at-zone vs market-on-confirm, LTF refines.

- Bearish rejection at resistance (in strength order):
  1. rejection wick: upper wick > 0.5× ATR, close in lower half,
     level within the upper wick zone;
  2. close-back: wicked above level, closed back below it;
  3. bearish body at resistance (weakest, needs other confluence:
     body > 0.3 ATR, close within 0.3 ATR of level).
  Level must be near the candle (within 0.5 ATR) or no validation.
- Bullish is the mirror (lower wick / close-back / bullish body).
- `choose_entry_type`: S/R flip, OB retest, liquidity retest →
  LIMIT_AT_ZONE; breakout, turtle soup, V-reversal →
  MARKET_ON_CONFIRM; default: validation strength ≥ 0.7 → limit,
  else market-on-confirm.
- **Status note:** this validator is a standalone entry system module.
  This review did not verify that every live playbook path calls it
  — the playbooks own their entry machinery (§1.1 iron rule), so do
  not assume the validator gates live fills.

---

## 3. Every indicator / sensor and what it measures

### 3.1 Reader-level (per-bar `MarketState`, `reader/reader.py`)

| Sensor | What it measures |
|---|---|
| Fractal swings (`fractal_n = 2`) | Confirmed swing highs/lows (bars each side) |
| BOS / CHoCH / structure events | Break of structure / change of character + direction |
| Trend (`trend`) | up / down / range from swing sequence |
| Liquidity pools | Resting-stop clusters above/below (pools above/below) |
| Sweep events | Wick through a pool that closes back inside; direction, extreme, age |
| FVGs | Fair value gaps (3-bar imbalance); active list, direction, age |
| Order blocks | Displacement leg's last opposite candle; mitigation/sweep outcomes |
| Displacement | Body/ATR (`disp_body_atr = 1.0` counts as displacement; `bull_disp_3` / `bear_disp_3` = 3-bar displacement strength) |
| OB displacement leg | `displacement_min = 0.8` body/ATR, `ob_search = 10` bars back |
| EMA stack (9/21/50) | Directional alignment gate (`ema_stack` ∈ {+1, −1, 0 mixed}) |
| ATR + ATR percentile | Volatility + `atr_pct` (window 252) |
| Daily bias (D1) | Daily fractal structure + close-vs-EMA50 tilt while ranging; snapshots use only strictly-prior days (no lookahead); needs `daily_min_days = 10` completed days |
| Sessions | Fixed-UTC legacy labeller (see §2.6 for the newer map) |
| Dealing range | From confirmed swings inside the lookback (premium/discount) |

### 3.2 Feature families (`features/`, defaults in `features/config.py`)

| Family | Module | What it measures / key params |
|---|---|---|
| Bollinger | `bollinger.py` | BB(20, 2.0), %B percentile (window 252), squeeze (pct < 20%), trend states |
| Fibonacci | `fibonacci.py` | Retracement levels, OTE zone; `fib_at_tol_atr = 0.15` |
| Ichimoku | `ichimoku.py` | Tenkan 9 / Kijun 26 / Senkou B 52 / displacement 26; cloud trend, support/resistance, TK alignment; D1 cloud |
| Relative strength | `relstrength.py` | RS vs peers/index: ROC over 24 bars, deadband 0.002, percentile window 252 |
| Classical patterns | `patterns.py` | Double top/bottom, head & shoulders; twin tol 0.30 ATR, shoulder tol 0.50 ATR, shoulder frac 0.35, min head depth 1.0 ATR, min valley 0.5 ATR, max age 150 bars |
| Supply/demand zones | `supply_demand.py` | Base ≤ 3 bars, base body ≤ 0.5 ATR, departure body ≥ 1.1 ATR, 5 context bars, max age 400, max 24 zones |
| AMD phases | `phases.py` | Wyckoff-flavoured state machine (§2.7) |
| Footprints | `footprints.py` | Bank-entry signatures, institution footprints, CTA flow (lookback 10, min 8 closes one VWAP side), retail chase (body ≥ 1.2 ATR, momentum ≥ 2.5 ATR, rvol ≥ 1.5), trap bars 3, inst rvol 1.6, max retrace 0.5 ATR |
| Volume | `volume.py` | RVOL (window 20), climax rvol ≥ 2.0, dry-up ≤ 0.6, delta character |
| MTF | `mtf.py` | HTF (D1/H4) alignment: aligned/opposed counts per direction |
| Trendlines | `trendlines.py` | Validated lines: min anchor sep 5 bars, tol 0.20 ATR, slope 0.002–0.35 ATR/bar, max age 300, max drift 6.0 ATR, max 6 lines/direction, rearm 2 bars |
| Divergence | `divergence.py` | RSI(14) divergence: pivot left 3 / right 2, min sep 5, max sep 120, min ΔRSI 2.0 pts, max age 30 bars |
| AOC | `aoc.py` | Area-of-concentration: cluster band 0.35 ATR, ≥ 2 distinct families (≥ 3 = MAJOR pivot zone), top 8 zones, recency half-life 96 bars (floor 0.5) |
| Traps | `traps.py` | 18 boolean GOD-CE salvage trap columns (see §1.5) |
| Gating | `gating.py` | Context-gated scoring per family (fib / zones / ichimoku / patterns / footprints / MTF / trendlines / AOC / divergence) + MTF cap (both HTFs opposed → 59.0) |

**Feature computation switches** (`FeatureConfig`): all families on by
default (`bollinger, fib, ichimoku, relstr, patterns, zones, phases,
footprints, volume, mtf, trendlines, divergence`).

### 3.3 V2 confluence sensors (`v2_confluence/`)

Zone clustering with same-category + cross-category dedupe (no
double-counting one level), FVG tracker (max age 100 bars, keep 5),
Fib impulse legs + retracement/extension levels, RSI divergence,
volume profile (POC / value area), lecture sizing
(`calculate_position_size`, invalidation check). These feed the
FVG-limit tracker and zone work; the v2 engine is a zone-scoring
layer, not the live decision engine.

---

## 4. Entry / management / exit rules

### 4.1 Entry rules (by path)

| Path | Entry rule |
|---|---|
| Anchor | Detector signal at score ≥ 60 (emit ≥ 35): sweep-confirmed or trend-aligned continuation, all gates §2.1 pass. Live paper loop: shadow only. |
| `fvg_midline` | Resting limit at FVG midline after fresh CHoCH; fill on first wick touch; CHoCH ≤ 48 bars old at fill. |
| `trendline_retest` | Break → 2 bars away → first return touch within 12 bars; enter in breakout direction. |
| `fvg_limit` | Resting limit at near edge; fill only on ≥ 0.10 ATR penetration; one trade per zone, first touch only. |
| `qm_m5` | Resting limit at QM level on first touch after neckline-break close; no retracement = no trade. |
| Multi-TF / real-trader | LIMIT at POI only (sweep → displacement → retracement → limit). Gap out of POI = SKIP. No retracement = NO TRADE. Never market-chase. |
| Paper executor (generic) | Pending orders fill at NEXT BAR'S OPEN ± half spread — the loop convention, NOT the resting-limit edge price (see §4.4 gap). |

### 4.2 Management / exit — Phase-2 swing exits (ONE machine everywhere)

Shared by the backtest harness (`simulate_signal_v2`), the live
stepper (`ExitStepper`), and the paper simulator. Semantics verified
identical in `execution/manager.py`:

- **Stop:** buffered structural stop = raw invalidation ±
  `stop_buffer_atr = 0.25` × ATR. R is measured on the buffered stop.
  Stop wins same-bar ties. Stop checked BEFORE the partial (conservative
  ordering). On MT5 the broker's own stop order is the last line of
  defence between steps.
- **Partial:** `partial_frac = 0.5` banks at the first liquidity pool
  at least `partial_min_r = 1.0`R away.
- **Trail:** once best excursion reaches `trail_trigger_r = 1.5`R,
  the remainder trails `trail_dist_r = 1.0`R behind the best
  excursion. Trail levels are set from COMPLETED bars only — a trail
  set by bar *k* is first active on bar *k*+1 (causal).
- **Time stop:** `horizon = 72` bars (H1 ≈ 3 days) closes what is left.
- **Per-bar order:** stops → partial → trail update → time stop.
- **Wick tracking:** a bar crossing the RAW invalidation sets
  `wicked_inv` (recorded, does not itself exit — the buffered stop
  governs).

**Per-playbook exit overrides (iron rule — these are NOT mixed):**

| Playbook | Exit override |
|---|---|
| `fvg_midline` | Fixed 2R target, NO partial/trail (opposing-OB variant not ported) |
| `fvg_limit` | Fixed 1.5R target; close-through-zone invalidation exits at close; unresolved after 48 bars DROPPED (not booked) |
| `qm_m5` | Neckline target (≥ 1.0R or skip); buffered-stop / head-invalidation / target; timeout booked at close after 48 bars |
| `trendline_retest` | Phase-2 swing exits (no override) |
| `sweep_reversal` | Phase-2 swing exits (no override) |
| Real-trader / MTF12 | Stop / target / opposing structural break / emergency time stop only (see §2.9) |

### 4.3 Exit pricing (paper)

- Paper fills: pending orders at next bar open + direction × half
  spread. Partial closes at the partial level (gapped through → the
  open). Stop closes at the stop level (gapped through → the open).
  Timeout closes at the bar's close. All exits charged the exit half
  of the spread (`_exit_price = raw − direction × half_spread`).
- Cost model (backtest + paper): `COST_PER_OZ = $0.45` round trip
  ($0.225 each side). Exness model uses $0.20/oz spread + $0.00
  commission instead (§5.2) — do not mix the two cost models.

### 4.4 Known fill-model gap (stated in the repo, not hidden)

The FVG sleeves' edges were measured on **resting-limit fills at the
level** (midline / near edge). The paper loop fills at **next bar's
open**. Adverse selection between signal bar and fill bar is not
modelled. The parity test proves the SIGNAL sets identical between
paths; the fill-price gap is measured by the live book, not assumed
away. (`SYSTEM.md` §12 lists this as still weak.)

---

## 5. Risk rules

### 5.1 Guard layer (`execution/risk.py`) — pure guards, never places trades

**Sizing (`size_position`):** lots are FLOORED to the volume step,
never rounded up. If the risk budget cannot buy the broker minimum,
answer is 0 lots + a human-readable reason. `risk_pct` must be in
(0, 0.25]. `gold_spec()`: XAUUSD, tick_value $1.00, tick_size $0.01,
volume_min 0.01, volume_max 50.0, volume_step 0.01.

**Guards (`RiskConfig`, calibrated 2026-10-11 from the anchor book's
loss clusters — "guards protect the TAIL, not the normal"):**

| Guard | Value | Basis in code comment |
|---|---|---|
| `daily_loss_pct` | 3% of day-start balance | Worst in-sample day −2.2% at 1% risk |
| `daily_loss_money` | None (if set, SMALLER wins) | ATS rule |
| `weekly_loss_pct` | 5% of week-start balance | Worst −3.1% |
| `weekly_loss_money` | None | — |
| `max_drawdown_pct` | 15% from peak → latched halt | 1.5× in-sample max 10.38% |
| `max_consecutive_losses` | 6 → latched halt | 3-runs hit 14× in 2y; 6+ hit once — tail only |
| `max_positions` | 3 | Guard ceiling (see §1.1 note on one-at-a-time) |
| `max_total_risk_pct` | 3% money-at-stops, of balance | — |
| `require_stops` | True | An unstopped position blocks new risk |

**Persistence / latching:** guard state is persisted as JSON on disk
(restart can never reset a hit loss limit). Day/week roll over by
calendar date (injectable clock for tests). A tripped halt LATCHES —
stays tripped with its reason until a human clears it via
`RiskGuards.reset_halt`. Corrupt state file → fresh state + recorded
`load_error`, never silent pass.

**Kill switch:** a `KILL` file at `data/KILL` (`RunnerConfig.kill_switch_path`).
`RiskGuards.kill_switch_active()` = the file exists. When active, new
trades are refused. (Operator creates/deletes the file; there is no
Telegram /kill command in `alerts/commands.py` — see §5.6.)

**Runner defaults:** `risk_pct = 0.01` (1%, overridable via
`GS_RISK_PCT`), `paper_balance = 10_000.0` (via `GS_PAPER_BALANCE`).
These are the generic runner defaults; the $12 paper/backtest runs
pass the balance explicitly (see §5.2) — do not quote $10,000 as "the
paper balance" without noting the $12 runs override it.

### 5.2 The $12 models — TWO ledgers, never mixed

There are two different $12 treatments in the repo. Do not conflate
them.

**A. OANDA 1-unit ledger (`live/multitf.py`, `backtest/mtf12.py`,
`backtest/real_trader.py`):**

- `PAPER_START_BALANCE_12 = 12.0`. At 1% risk the budget is **$0.12
  per trade**.
- A $12 account CANNOT execute most swing trades at the OANDA 1-unit
  (1 oz) minimum: a $20 stop means 1-unit risk of $20.00, 167× the
  $0.12 budget.
- Results are reported in TWO ledgers:
  1. **Fractional-unit R accounting** — strategy quality
     (R-multiples, win rate, drawdown). A measurement device, NOT a
     claim OANDA would fill them.
  2. **`oanda_1unit_feasible` flag per trade** — feasible iff the
     risk of ONE ounce (stop distance in dollars) fits inside the
     risk budget. Feasible trades reported as a separate subset.
- Margin reported separately (`margin_50x`), NOT folded into the
  flag: OANDA leverage varies by jurisdiction. At 50:1, 1 oz of gold
  at ~$4000 needs ~$80 margin — a further, separate blocker on $12.
- **Never present fractional-unit results as OANDA-executable.**

**B. Exness $12 model (`backtest/exness12.py`) — the current
small-account model:**

- Exness Standard MT5: max leverage **1:2000**, min spread **0.2**,
  commission **$0.00/side**, min lot **0.01**. (Per the module
  docstring, from the owner's account screenshot.)
- Fixed size **0.01 lot = 1 oz** (1 lot XAUUSD = 100 oz). Lot scaling
  tested separately: 0.02 = 2 oz, 0.05 = 5 oz.
- Leverage changes MARGIN, not PnL per lot. PnL per 0.01 lot stays
  **$1 per $1 gold move** at any leverage. Unlimited leverage only
  increases PnL if lot size scales up — which also scales loss.
- Margin: `notional / leverage`. At ~$4000 gold, 1:2000 → ~$2.00
  margin for 0.01 lot; 1:2 → ~$2000 (so a $12 account CANNOT open
  even 0.01 lot at 1:2 — shown as impossible, not viable); unlimited
  → margin ~$0. Free margin = equity − margin.
- Spread **0.2 interpreted as $0.20/oz per trade** (round-trip cost
  on entry, scales with lot: 0.01 → $0.20, 0.02 → $0.40, 0.05 → $1.00).
  This interpretation is a stated assumption in the module, not a
  broker quote — Exness quotes spread in price points; the $0.20/oz
  mapping should be re-checked against a real Exness fill before
  treating Exness backtest $ results as exact.
- **Stop-out:** Exness Standard margin call 60%, stop-out 0%
  (module states "verified" — this review did not independently
  verify against Exness docs; treat as module-claimed). Margin level
  = equity / used_margin × 100. Stop-out when equity ≤ 0. Negative
  balance protection restores to $0 (assumed, not relied on).
- **No stop-distance gate:** wide stops are NOT rejected; margin is
  the real constraint. A trade whose structural stop is wider than
  margin room will blow the account — that IS the model.
- PnL: `PnL_net = direction × (exit − entry) × oz − spread − commission`,
  one position at a time. If balance < margin at entry, cannot open
  → simulation ends as `cannot_open`.
- **Model correction vs earlier work:** the prior OANDA 50:1 /
  $0.12-risk / R-multiple backtests are the WRONG model for Exness
  fast growth ($12 → $100). The Exness model is timing-based (0.01
  lot, 1:2000, margin-aware), not stop-loss R grinding. $12 adverse
  move = blow-up; $20 favourable = 2–3× account. Timing is everything.
- **Caveat:** Exness may restrict leverage around news, so 1:2000 is
  not guaranteed at CPI/news. The backtest uses constant leverage per
  run; real live leverage may drop in those windows, raising margin.

### 5.3 Max one position (paper loop)

Paper loop: one position at a time (first-wins per bar; batch book:
chronological greedy). The guard layer's `max_positions = 3` is a
ceiling above that norm, not the loop's operating rule.

### 5.4 Watchdog (`execution/watchdog.py`, `live/data_health.py`)

- Feed staleness: halt entries if no new bar within
  `staleness_bars = 3` × timeframe (H1 → 3h). Alert + halt new
  entries; management continues. Auto-recovers on a fresh bar.
- Heartbeat: touched every loop; external watcher alarms if older
  than `heartbeat_stale_s = 300`s. The heartbeat watcher is
  operator-owned (no built-in cron) — `SYSTEM.md` §12.
- Dead-man's: alert text on unclean death via outbox.
- MT5: `mt5.ping()` each loop; on disconnect, alert "MT5 TERMINAL
  LOST" + halt entries until ping succeeds. Empty/garbage
  `copy_rates` (≤ 100 closed bars) → skip iteration, retry next loop.
- Telegram: 3 tries, exponential backoff, 429 honours `retry_after`;
  photo falls back to text; fatal codes (400/401/403/404) not retried.
- Corrupt bar buffer → `load_error`, journaled, cold start (full
  fetch).

### 5.5 Hard veto: OFF by policy — but the code default is ON

This is the single most important policy-vs-code gap in the repo.
Read it carefully:

- **Policy (2026-10-13, owner decision, recorded in `live/regime.py`
  and the task brief):** hard veto OFF. Do NOT enable in live or paper
  trading yet. In-sample it mostly removed winning trades (Swing
  Exness +$43.47 blocked winners; Intraday blocked +21.96R / Exness
  reset +$386.59, incl. PF 3.69 blocked shorts) and did NOT reduce
  blow-up rates anywhere (Intraday 2.4% → 2.5%, Scalp 0.3% → 0.5%
  reset). Only Scalp showed a genuine quality win (higher PF, lower
  MaxDD). The two veto definitions (main repo vs demo repo) differ —
  unify or both off. Treat veto as a failed hypothesis except
  possibly on Scalp. Keep main-bias hierarchy (Swing D1, Intraday H4,
  Scalp H1) without the hard-veto layer.
- **Code fact:** `live/multitf.py` `_evaluate(..., enable_veto: bool =
  True)` and `evaluate(..., enable_veto: bool = True)` — **the
  function default is True**. The backtest CLIs expose `--no-veto`
  (opt-out). `check_hard_veto` / `check_vetoes` remain fully
  implemented and tested (`tests/test_hard_veto.py`).
- **Why OFF still holds in practice:** no caller in the live paper
  loop invokes the multitf veto path — the paper loop goes through
  playbooks + confluence, not `multitf.evaluate`. The veto only bites
  when a caller runs `multitf.evaluate` / the backtest harnesses
  without passing `enable_veto=False`.
- **Action required before any future use:** any new caller of
  `multitf.evaluate`, `real_trader`, `mtf12`, or `exness12` must pass
  `enable_veto=False` (or `--no-veto`) explicitly, or it will silently
  run with the veto ON. Flipping the function default to `False`
  would close this trap; that change is NOT made by this doc.
- **Do not confuse the hard veto with the detector D1 veto:** the
  anchor's `daily_veto=True` (detector-level, §2.1) is ON and is a
  different mechanism (D1 bias blocks counter-trend trades, with the
  confirmed-sweep-at-major-level exception). The hard veto is the
  next-higher-TF strong-disagreement block. The MTF score cap (both
  HTFs opposed → cap 59.0, §3.2) is a third, separate mechanism and
  is also still live in the detector.

### 5.6 Kill switch + halts (operator surface)

- `data/KILL` file = kill switch (new trades refused while present).
- Latched halts (daily/weekly loss, drawdown, consecutive losses) need
  human `reset_halt`.
- **There is NO Telegram /kill, /halt, /update, /pull, /deploy, or
  /restart command** in `alerts/commands.py` (verified: registered
  commands are `/help`, `/status`, `/trades`, `/db`, `/report`,
  `/harvest` only). Halts are cleared by file/process action on the
  host, not from Telegram. The missing TG `/update` (pull) command
  is a known gap recorded in the owner's notes (Devon TG bot gap —
  same class of gap, different bot; do not conflate the two bots).

---

## 6. Paper loop, journal, Telegram

### 6.1 Paper loop (`scripts/paper_live.py` + `execution/runner.py`)

- **Feed:** live OANDA v20 practice H1 XAU/USD (`feed/oanda.py`),
  poll every `poll_seconds = 300`s (5 min). Warm-start history, then
  `fetch_new` for closed bars only (forming bar excluded via OANDA's
  `complete` flag). Live cross-check: forming candle's close vs OANDA
  pricing mid — a mismatch halts live signals (the 2026-10-11
  yfinance-incident shape: phantom GC=F bars voided the first live
  signal). Sanity-gate rejections are journalled.
- **Why OANDA, not yfinance:** the 2026-10-11 incident — yfinance
  printed phantom H1 bars (highs 4235–4242) on a day OANDA did not.
  The legacy `PollingFeed` (yfinance GC=F) is kept for backtest /
  rehearsal only.
- **Modes:** `--once` (one cron tick, then exit — state in
  `data/paper_state/`, designed for hourly cron just after the H1 bar
  closes) or daemon loop. **Never run `--once` concurrently with the
  daemon loop** (both would process the same bars and double-trade).
- **Per bar:** `live_signal` → confluence gate → Telegram alerts →
  risk → PaperExecutor fills → journal + learning. Engine
  `risk_mode = "enforce"` in the paper loop (risk decisions are real).
- **Exec set:** `run_paper_once` defaults to
  `execute_playbooks = ("fvg_midline", "trendline_retest")` — these
  are auto-paper-traded. Every other live playbook (anchor,
  `fvg_limit`, `qm_m5`) appears in candidates / journal / shadow
  book, never in `exec_set`, without a separate explicitly-approved
  step. `min_score` defaults to the runner config; `signal_gap_bars`
  default 0.
- **State:** `StateStore` persists the bar buffer, playbook streaming
  state, paper executor state (open positions keep exit-stepper
  state), and last-processed bar time. On buffer rebase, playbook
  state is rebuilt by cold-start fast-forward and paper positional
  bar indices are shifted by the same delta, so hold-time arithmetic
  is preserved. `GS_COLD_START=1` forces a full cold start.
- **Rehearsal:** `scripts/paper_rehearsal.py` replays history as-if
  live (end-to-end proof); `scripts/paper_report.py` produces
  per-playbook paper OOS stats from the journal.
- **OANDA feed note:** OANDA XAU_USD hours ≈ Sun 22:00 UTC – Fri
  21:00 UTC (`market_closed_now`). The feed is H1-only — this is why
  QM M5 has no live feed (§2.5).

### 6.2 Journal (`execution/learning.py` — `TradeJournal`)

- Append-only JSONL (`data/trade_journal.jsonl` by default). Every
  decision journalled like a trader's log: candidates, confluence
  bonuses, risk decisions, fills, closes, context-violations (as
  SYSTEM BUG candidates), notes.
- `performance().aggregate()` → win rate, avg R, PF, fill rate,
  shadow-fill stats. `LiveStats` tracks zone respect rate, trades,
  wins, total R, PF, execution cost in R, segmented win rates.
- `monitor.py` (`DivergenceMonitor`): compares live stats against a
  baseline with Wilson intervals; segment streaks; downside-only
  "worse" checks. `divergence_api.py` exposes the divergence surface.
- `reconcile.py`: journal/position reconciliation. `journal_lock.py`:
  lock + timeout wrapper for concurrent journal access.
  `shadow_fills.py`: shadow-book fill accounting.
- **Journal seq atomicity:** covered by `tests/test_journal_seq_atomicity.py`
  (sequence assignment under concurrency). This review did not
  re-verify the implementation beyond the test's existence.

### 6.3 Telegram commands (`alerts/commands.py`)

- **One consumer rule:** commands arrive as `message` updates through
  the SAME `getUpdates` poll the button taps use
  (`LiveRunner._poll_telegram`). Never a second poller — two consumers
  race on the update offset and one starves.
- **Owner-only:** every command is answered only when the update's
  `chat.id` matches `GS_TELEGRAM_CHAT_ID`. Anything else — other
  chats, groups, channels, non-text — is ignored SILENTLY: no reply,
  no error, no log line an attacker can probe. Unconfigured owner id
  = nobody (fail closed).
- **A command never breaks the loop:** handler exceptions return a
  failure text, not a crash.
- **Extension contract:** one `@command("/name", "description")`
  function; `/help` picks it up automatically. `CommandContext`
  carries journal, journal_path, paper, feed_status, last_price,
  last_bar_time, measured_oos, reports_dir.

| Command | What it does |
|---|---|
| `/help` | Lists the commands |
| `/status` | Loop health: feed, open positions, today's P&L, last signal |
| `/trades [N]` | Last N paper trades with R multiples (default 5) |
| `/db` | Sends the paper journal database as a file (journal only) |
| `/report` | Sends the latest backtest/paper results as a file (report only) |
| `/harvest [all\|journal\|state\|results]` | One ZIP: database + journal + paper_state + risk_state + measured-OOS / modes / validation + RESULTS.md + manifest.json (sha256 per file). See below. |

**`/harvest` (commit 282b4ca):**

- Scopes: `all` (default) / `journal` (journal files only,
  `*journal*.jsonl`) / `state` (paper_state + risk_state, no
  journal) / `results` (RESULTS.md + manifest + measured-OOS / modes
  / validation).
- `RESULTS.md` is generated into the bundle: trades, win%, total R,
  PF, PnL, per-playbook, today, last 10, open positions.
- `manifest.json` carries sha256 per file.
- Excluded: CSVs / pkl / PNGs / `.venv` / wheels.
- Caps: 20 MB/file, 45 MB ZIP (under Telegram's 50 MB limit), with
  slim fallback. Tests: 31 passed in `test_commands.py` (per the
  commit note; this review did not re-run the suite).
- `/db` (journal only) and `/report` (report only) are unchanged.

**Alert buttons (`alerts/callbacks.py`):** Telegram inline buttons
Execute / Skip / Paper per alert (`AlertAction`), via the same
`getUpdates` poll. Email alerts via `alerts/email.py` (SMTP from env;
mode reporting when unconfigured). Outbox fallback
(`alerts/outbox.py`): when Telegram/email credentials are absent
(`GS_TELEGRAM_BOT_TOKEN`, `GS_TELEGRAM_CHAT_ID`, …), alerts land in
the outbox dir. Narrator (`alerts/narrator.py`) renders trader-style
alert text from signal factors. Charts (`charts/engine.py`): 5 chart
classes/styles incl. phone-first `signal` alert chart, volume drawn,
zones/FVGs/Fib/cloud/swings drawn — "a human can read the whole thesis
off the image" (SPEC.md; chart quality not re-verified by this review).

### 6.4 Live MT5 path (`execution/runner.py run_live`, `execution/mt5.py`)

- `MT5Executor` wraps the official MetaTrader5 Python library:
  `copy_rates` for data, `order_send` for execution. Requires an MT5
  terminal installed, logged in, and ACTIVE on the same machine.
  Windows-only → live deployment = full stack on a Windows host.
- `deviation = 20` default; filling-mode negotiation
  (`_send_with_fillings`); `ping()` for the watchdog; `copy_rates_h1`
  with `len(closed) > 100` guard.
- **Status: code-complete but UNPROVEN. Never run against a real
  terminal.** First runs are DEMO ONLY (`SYSTEM.md` §10.3). Live MT5
  fills are unverified — slippage, requotes, filling modes, and
  broker stop behaviour are all unmeasured.

---

## 7. What's still weak / unproven / untested (brutally honest)

### 7.1 The anchor fails out-of-sample — the elephant

`SYSTEM.md` §5 (walk-forward, expanding 12mo train / 3mo test):

- `sweep_reversal` (anchor, LIVE): **FAILS**. OOS aggregate 94
  trades, **−0.28%, PF 1.01**, Sharpe 0.04, win 39.4%. The +28.92%
  in-sample does not hold up. The edge is concentrated in months 0–12;
  months 12–24 are flat to negative. **Blocker for live trading.**
- `fvg_midline` (SHADOW): SURVIVES (thin) — OOS 27 trades, +18.82%,
  PF 2.54, Sharpe 2.30, win 59.3%. Caveat: the hunting ground
  (trend=down excluded) was derived in-sample; the OOS test is
  optimistic for the filter.
- `trendline_retest` (SHADOW): SURVIVES — OOS 39 trades, +14.03%,
  PF 1.62, Sharpe 1.23, win 46.2%. One negative window (15–18mo,
  −2.49%).
- Promotion checklist (per `CONFLUENCE.md` / `SYSTEM.md` §10) now
  requires: OOS confirmation on 2022–2024 or a second instrument +
  resolution of the anchor's in-sample concentration, before any live
  promotion. The paper loop grows sleeve OOS counts toward a 50-trade
  bar.
- **Consequence:** the "anchor = live, sleeves = shadow" narrative is
  backwards from the evidence. The sleeves are more robust OOS than
  the anchor. The paper loop already reflects this (anchor shadow,
  sleeves executed) — but the playbook `live=True` flag on the anchor
  still invites misreading. Do not trade the anchor live until the
  OOS failure is resolved (investigate the regime break, demote the
  anchor, promote a sleeve, or re-derive on 2022–2024 data).

### 7.2 Placeholder scores everywhere except the anchor

Every sleeve's `SLEEVE_BASE_SCORE = 60.0` is a documented placeholder,
NOT a measured score (their docstrings say so). Confluence ranking
between sleeves therefore ranks placeholders + bonuses, not measured
quality. OOS promotion must calibrate these scores; until then
sleeve-vs-sleeve ranking is ordinal noise with real bonuses on top.

### 7.3 The three 2026-10-13 additions are unvalidated

- **AMD fallback (§2.7):** implemented + unit-tested, but its effect
  on trade outcomes has never been measured. Confidence 0.55 is a
  judgement parameter, not a calibrated probability. A fallback that
  fires more often than the direct check could manufacture
  "manipulation" labels in choppy bases — the expiry/refresh logic
  bounds episodes, but no trade-level measurement exists.
- **QM M5 (§2.5):** zero validation — no backtest, no win rate, no
  context map, no M5 live feed. Every QM parameter (120/48/48 bars,
  0.25/0.5 ATR legs, 1.0R minimum) is a design choice from practitioner
  doctrine + repo precedents, not a measured optimum.
- **Session map (§2.6):** no backtest behind the M15/M30 split. The
  DST correction itself is sound (wall-clock arithmetic), but whether
  M15-in-killzone beats M30-outside — or whether either beats a fixed
  clock — is unmeasured. The legacy fixed-UTC labellers still run in
  the modules that produce backtest numbers, so backtest session
  labels and live session labels can disagree by design.

### 7.4 Fill-model and cost-model gaps

- Paper fills at next-bar-open ± half spread vs resting-limit fills
  at the level (§4.4). Adverse selection unmodelled. The FVG sleeves'
  edges were measured on the limit price; the paper loop does not pay
  the limit price.
- Two cost models coexist ($0.45/oz OANDA vs $0.20/oz + $0 Exness)
  and must never be mixed. The Exness $0.20/oz spread interpretation
  is a stated assumption, not a broker quote — re-check against a real
  Exness fill.
- Exness leverage is assumed constant per run; real leverage may drop
  around news/weekends, raising margin. The stop-out 60%/0% figures
  are module-claimed, not independently verified here.
- No slippage model beyond half-spread in paper; no requote /
  partial-fill / filling-mode behaviour measured (MT5 path unproven).

### 7.5 Live execution is unproven

- MT5 path never run against a real terminal. No live fill, no
  slippage measurement, no broker-stop behaviour, no reconnect-under-
  load test.
- Paper loop proven on replay, not (in the repo's own words) on long
  live-feed operation — first live runs should be watched for feed
  quirks (forming-bar timing, DST/index oddities, weekend gaps).
  The OANDA feed replaced yfinance after a phantom-bar incident; the
  cross-check exists because the feed has lied before.
- Heartbeat watcher is operator-owned (no built-in cron). If the
  operator's watcher dies, nobody notices the loop died.
- Credentials/token in repo history — rotate before any live
  deployment (`SYSTEM.md` §10.4). Status of rotation was not verified
  by this review.

### 7.6 Policy-vs-code traps

- **Hard-veto default ON** (§5.5): any new caller of
  `multitf.evaluate` / the backtest harnesses without
  `enable_veto=False` / `--no-veto` silently runs with the veto ON,
  contradicting the OFF policy.
- **Two veto definitions** (main repo vs demo repo) differ — unify
  or both off. This doc does not resolve which definition is correct.
- **Three session labellers** (fixed-UTC) + one new DST-aware map:
  four sources of "session" truth. Backtest session labels and live
  session labels can disagree.
- **Two regime routers** (bedrock §1.4 + `system/regime.py` §1.7) +
  GOD-CE router + phase routing: four routing layers. Their
  interactions (e.g. bedrock SIT_OUT vs `system/regime` CHOP vs a
  playbook firing anyway) are not documented as one precedence order
  in any single place this review found. If they disagree, the
  playbook/confluence gates are the binding ones in the paper loop —
  but that precedence is inferred from the wiring, not stated in a
  single contract.
- **Forecast bundle wiring** in the live paper loop was not verified
  (see §1.8) — forecast bonuses may be silently absent live.

### 7.7 Thin samples and single-regime history

- Whole validation window is one strongly bullish 2-year stretch for
  gold (BACKTEST2.md). Nothing is validated out of sample on a
  non-bull window or a second instrument (the promotion checklist
  requirement that is still open).
- Sleeve samples are thin: trendline 61 trades in-sample / 39 OOS;
  fvg_midline 97 / 27 OOS. A 27-trade OOS "survival" (PF 2.54) is
  encouraging, not proof.
- FVG-limit A/B: the +106.1R (0.05 penetration) vs +32.0R (0.10)
  gap shows the result is highly sensitive to the fill-penetration
  assumption — and the live loop fills at next-bar-open, i.e. under
  NEITHER tested assumption. The sleeve's live edge is therefore
  strictly less established than its backtest numbers suggest.
- Real-trader $12→$424.50 (Swing) is a doc-quoted backtest on the
  same bullish window, under the OANDA ledger that §5.2 says is the
  wrong model for Exness. Do not quote it as "the bot turns $12 into
  $424" without both caveats.

### 7.8 Gaps in THIS doc (claims not verified)

- The exact `live` flag value of `fvg_limit` was not re-confirmed
  line-by-line; its shadow status rests on the runner's `exec_set`
  wiring, which IS confirmed.
- `run_paper_loop`'s default `execute_playbooks` was read only
  partially (first element `fvg_midline`); the full default is stated
  from `run_paper_once`'s complete signature. If the two defaults
  differ, the `--once` (cron) default governs the deployed loop.
- `cost_gate_ok` call sites, forecast-bundle wiring in the live paper
  loop, and the `max_positions=3` vs one-at-a-time interaction were
  not exhaustively traced; each is flagged inline as unverified
  rather than asserted.
- No backtest was re-run for this review. Every performance number
  is a quote from the repo, not a fresh measurement. `BACKTEST.md`,
  `BACKTEST2.md`, `BALANCE_100_200.md`, `FORECAST.md`,
  `VALIDATION.md`, and the `experiments/` reports contain further
  numbers not reproduced here — this doc covers the live system's
  logic and the headline results, not every experiment's table.
- `tests/test_hard_veto.py` leg-strength threshold (0.25× ATR14) is
  quoted from the test file; the corresponding constant in
  `live/multitf.py` was read as "legs ≥ 0.25× ATR14" via the test
  docstring — if the production constant differs, the test is the
  authority the code was checked against.

---

## 8. What to decide / do next (owner checklist, from the evidence above)

1. **Anchor:** investigate the OOS regime break, demote, or re-derive
   on 2022–2024 — before any live-anchor trade. (SYSTEM.md §10.1)
2. **Sleeves:** grow paper OOS toward the 50-trade bar; calibrate the
   placeholder 60.0 base scores from OOS; confirm on 2022–2024 or a
   second instrument. (SYSTEM.md §10.2)
3. **Hard veto:** flip `multitf.evaluate`'s default to
   `enable_veto=False` (or otherwise close the trap), and unify the
   two veto definitions or delete one. (§5.5)
4. **QM M5:** backtest on M5 history, then give the paper loop a real
   M5 feed, before any promotion discussion. (§2.5)
5. **Session map:** backtest the M15/M30 split (or delete the claim
   that it matters); reconcile the four session labellers. (§2.6)
6. **AMD fallback:** measure its trade-level effect (fires, precision
   vs direct, R impact) before trusting fallback-labelled manipulation
   in any sizing decision. (§2.7)
7. **Fill model:** measure the next-bar-open vs resting-limit fill
   gap on the live book; re-check the Exness $0.20/oz spread mapping
   against a real fill. (§4.4, §5.2)
8. **MT5:** demo-only first run against a real terminal; measure
   slippage / filling modes / broker stops. (§6.4)
9. **Ops:** rotate any exposed credentials; confirm the heartbeat
   watcher exists and is itself watched; add a Telegram /kill or
   document that halts are host-only. (§5.6, §7.5)
10. **Routing precedence:** write down the single precedence order
    across bedrock / system regime / GOD-CE router / phase routing /
    confluence gates. (§7.6)

---

*End of review. No code was modified to produce this document. The
only file this task creates is this one.*
