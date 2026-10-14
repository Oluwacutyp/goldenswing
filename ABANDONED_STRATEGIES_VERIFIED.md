# Abandoned Strategies - Verified (NOT Deleted)

Verified 2026-10-13. No strategy deleted. No module called dead without grep evidence.
Rule: before calling anything dead, check static imports + dynamic imports (importlib / registries / string loading) + tests + scripts + docs + live wiring.

## How to read verdicts

- **LIVE** - In `execution/runner.py::_get_playbooks` or imported by a LIVE playbook.
- **WIRED-BACKTEST-ONLY** - Has test + script runner, not in live loop. Measurement / filter / diagnostic, not a standalone strategy.
- **SHADOW/CANDIDATE** - Valid logic, has test + script, not wired to live. Can revive with veto + Exness timing + forward paper. Not dead.
- **DUPLICATE (partial)** - Overlaps core engine logic, revive only as variant, not as separate strategy.
- **FALSIFIED** - Backtest evidence shows it loses as standalone, with numbers. Does NOT mean its parts are dead.
- **TRULY DEAD** - No imports outside own test/script, no tests, no docs, no live wiring, no logic worth reviving. **None found in this audit.** Every experiment has at least a test + script.
- **FEATURE** - Indicator / context family, not a strategy. Wired via `features/engine.py`, never a standalone trigger.
- **OUT-OF-SCOPE** - Valid for another instrument, not XAUUSD. Needs revalidation before XAU use.

## 1. Playbooks - `goldenswing/playbooks/`

| File | Class | Live flag | In runner `_get_playbooks`? | Verdict | Evidence |
|---|---|---|---|---|---|
| `fvg_midline.py` | FVGMidlinePlaybook | `live=True` | YES | **LIVE** | `runner.py:210-215` instantiates it. `playbook_of` maps to it. |
| `sweep_reversal.py` | SweepReversalPlaybook | `live=True` | YES (anchor, shadow exec) | **LIVE (anchor, shadow exec)** | `runner.py:213`, `execute_playbooks=("fvg_midline","trendline_retest")` excludes sweep from auto-paper, but sweep still signals. Comment: "sweep_reversal anchor, which failed walk-forward" - see §5. |
| `trendline_retest.py` | TrendlineRetestPlaybook | `live=True` | YES | **LIVE** | `runner.py:214`. Imports `trendline_claims` (see §2). |
| `fvg_limit.py` | FVGLimitPlaybook | `live=True` | **NO** - exported in `playbooks/__init__.py` but NOT in runner list | **SHADOW/CANDIDATE** | `playbooks/__init__.py:15` exports it. `runner.py:210-215` does NOT instantiate it. `scripts/test_fvg_limit_parity.py` proves parity. Revive: add to runner `_get_playbooks` + `exec_set`, test live parity, forward paper. Fits Swing/Intraday as patient LIMIT variant. |
| `base.py` | Playbook / EngineContext | - | - | **INFRA** | Contract, not a strategy. |

## 2. Experiments - `goldenswing/experiments/` (all have test + script, none is TRULY DEAD)

| File | Main fn/class | Test | Script | Wired to live? | Real-trader logic? (HTF veto + LIMIT at POI + sweep->disp->retrace + AMD + killzones) | Verdict | Revive path |
|---|---|---|---|---|---|---|---|
| `body_stop.py` | `body_anchor`, `body_invalidation`, `variant_exits` | `test_experiments_body_stop.py` | `scripts/experiment_body_stop.py` | No | Partial: stop rule only, no bias/veto/AMD | **SHADOW/CANDIDATE - stop module, DUPLICATE partial** | Revive as stop rule variant for Swing: sweep extreme + body anchor, with veto + Exness timing. Overlaps structural stop in `exness12.py`. Do not revive as standalone. |
| `bodybreak_confirm.py` | `body_break_trigger`, `scan_variant` | `test_experiments_bodybreak_confirm.py` | `scripts/experiment_bodybreak_confirm.py` | No | Partial: confirmation only | **SHADOW/CANDIDATE - trigger confirmation** | Revive as displacement confirmation variant: body-break = displacement. Fits Scalp (fast confirmation). Needs veto + Exness timing. |
| `bollinger_trends.py` | `squeeze_episodes`, `run_bollinger` | `test_experiments_bollinger_trends.py` | `scripts/experiment_bollinger_trends.py` | No (feature measurement) | No - indicator | **FEATURE, WIRED-BACKTEST-ONLY** | Do NOT revive as standalone strategy. Bollinger is context only. Already in `features/bollinger.py`. |
| `candles.py` | `pin_bar`, `engulfing`, `displacement`, `is_doji` | `test_experiments_candles.py` | `scripts/experiment_candles.py` | Indirect via `engulf_trigger`, `placement` | Partial: candle shapes, no HTF veto | **FEATURE / CONTEXT** | Candle shapes are trigger confirmation, not standalone. `engulfing` used by `engulf_trigger`. `displacement` overlaps core displacement. |
| `choch_fvg.py` | `scan_setups`, `SleeveSetup/Fill` | `test_experiments_choch_fvg.py` | `scripts/experiment_choch_fvg.py` | **YES via `fvg_midline.py:18` import** | Partial: CHoCH->FVG-midline->OB, no veto/AMD in original | **LIVE (via fvg_midline)** | Already live through playbook. Do NOT duplicate. To harden: add veto + AMD to playbook, do not create second copy. |
| `divergence.py` (experiments) | `featurize`, `run_family_test` | `test_experiments_divergence.py` | `scripts/experiment_divergence.py` | No (measurement) - `features/divergence.py` is the wired one via `features/engine.py:17` | No - indicator | **FEATURE, WIRED-BACKTEST-ONLY (experiments) / LIVE FEATURE (features/)** | Do NOT revive as standalone. RSI divergence is context only. |
| `engulf_trigger.py` | `scan_variant`, `_make_signal` | `test_experiments_engulf_trigger.py` | `scripts/experiment_engulf_trigger.py` | No | Partial: engulfing-close trigger for sweep, no veto/AMD | **SHADOW/CANDIDATE - trigger variant** | Revive as trigger variant for Scalp: sweep + engulfing close = trigger, then LIMIT at POI. Needs veto + Exness timing + forward paper. |
| `failure_modes.py` | `diagnose_trades`, `StrategyDiagnosis` | `test_experiments_failure_modes.py` | `scripts/experiment_failure_modes.py` | No (diagnostic) | N/A - diagnostic | **WIRED-BACKTEST-ONLY diagnostic** | Not a strategy. Keep as diagnostic. Use to diagnose losing trades (direction vs timing). |
| `imbalance.py` | `detect_gaps`, `replay_lives` | `test_experiments_imbalance.py` | `scripts/experiment_imbalance.py` | No | Partial: FVG measurement | **SHADOW/CANDIDATE - DUPLICATE partial** | FVG logic already in core engine. Do NOT revive as separate strategy. Use `detect_gaps` as FVG detection audit, not new entry. |
| `marco_sweep.py` | `MarcoHit`, `_make_signal` | `test_experiments_marco_sweep.py` | `scripts/experiment_marco_sweep.py` + `scripts/sensitivity_marco_touches.py` | No | Partial: sweep-entry vs confirmation, no veto/AMD | **SHADOW/CANDIDATE - sweep-entry variant** | Revive as aggressive entry variant: sweep-entry (enter on sweep, no displacement wait) vs confirmation. Fits Scalp only, high risk. Needs veto + Exness timing + forward paper. Prior pillar challenge, not falsified as variant. |
| `min_rr.py` | `qualification`, `verdict_for` | `test_experiments_min_rr.py` | `scripts/experiment_min_rr.py` | No (filter) | N/A - filter | **WIRED-BACKTEST-ONLY filter** | Not a strategy. Minimum R:R filter. In Exness timing model, R is replaced by $ move, so filter must be re-expressed as min $ move >= $1.00 (already in `exness12.py`). Do NOT revive as R filter. |
| `opening_range.py` | `ORSetup`, `ORFill` | `test_experiments_opening_range.py` | `scripts/experiment_opening_range.py` | No | No - QQQ opening range | **OUT-OF-SCOPE / FALSIFIED for XAU** | Valid for QQQ (user-contributed, comment section), not XAUUSD. No XAU evidence. Do NOT revive for gold without full revalidation on XAU M15/M5/M1 + veto + Exness timing. |
| `ote_entry.py` | `OTESetup`, `OTEFill`, `scan_setups` | `test_experiments_ote_entry.py` | `scripts/experiment_ote_entry.py` | No | Partial: OTE Fib zone, no veto/AMD | **SHADOW/CANDIDATE - entry-zone variant** | Revive as POI variant: OTE 62-79% Fib zone = POI. Fits Swing (H4 zones). Needs veto + Exness timing + forward paper. Mirrors `choch_fvg` snapshot logic (see `ote_entry.py:118`). |
| `placement.py` | `_t1/_t2/_t3/_t4_signals`, `_simulate` | `test_experiments_placement.py` | `scripts/experiment_placement.py` | Indirect via `failure_modes.py:351` import | Partial: T1 failed pattern, T2/T3 retest variants | **SHADOW/CANDIDATE - placement logic** | T2/T3 are retest variants, overlap `retest_entry` + `trendline_retest`. Revive T3 (failed zone-retest) as Scalp trap variant with veto + timing. T1 is failed-pattern continuation, needs separate validation. |
| `retest_entry.py` | `RetestSetup`, `RetestFill`, `scan_setups` | `test_experiments_retest_entry.py` | `scripts/experiment_retest_entry.py` | No | Partial: JeaFx retest, failed opposite sweep, no veto/AMD | **SHADOW/CANDIDATE - retest variant, DUPLICATE partial** | Overlaps `trendline_retest` + `placement` T3. Revive as retest confirmation: failed opposite sweep + retest = trigger. Fits Intraday. Needs veto + Exness timing. |
| `trendline_claims.py` | `build_lines`, `index_lines` | `test_experiments_trendline_claims.py` | `scripts/experiment_trendline_claims.py` | **YES via `trendline_retest.py:20` import** | Partial: 3 trendline claims | **LIVE (via trendline_retest)** | Already live through playbook. Do NOT duplicate. |

## 3. GODCE - `goldenswing/godce/`

| File | Role | Live wiring? | Verdict | Evidence |
|---|---|---|---|---|
| `engine.py`, `router.py`, `scoring.py`, `params.py`, `indicators.py`, `stateful.py`, `backtest.py` | GOD Conviction Engine: router, conviction, scoring, Pine parity | **NO live wiring.** Only `tests/test_godce.py` + `tests/test_traps.py:362` (parity check) + `confluence.py` comments referencing `GODCE_SALVAGE.md` | **SHADOW/CANDIDATE - regime/router, not live** | No `from ..godce` / `import godce` in `execution/`, `live/`, `playbooks/` (except tests). `features/traps.py` is godce-independent port. Revive path: use as regime filter (trend/range) with veto + Exness timing, NOT as standalone entry. Needs forward paper. Slim pine waiting for TradingView retry, not live. Do NOT call dead - has tests + parity + docs. |

## 4. Modes - `goldenswing/modes/`

| File | Role | Live wiring? | Verdict |
|---|---|---|---|
| `routing.py`, `runner.py`, `simulate.py`, `config.py`, `report.py`, `data.py` | Mode routing, per-mode simulation, SWING config | **NO live wiring.** Only `scripts/backtest5.py` (modes/report) + `scripts/experiment_choch_fvg.py` (modes/config SWING) | **SHADOW/CANDIDATE - mode infrastructure** | Not live. `SWING` config used in experiments. Revive only if mode routing is needed for multi-preset live. Otherwise keep as backtest infrastructure. |

## 5. Backtest / Ablation - `goldenswing/backtest/`

| File | Role | Verdict |
|---|---|---|
| `harness.py` | GoldenSwing detector harness. Sweep slice: n=1551, win 25.0%, avg_r -0.113 (`harness.py:547`). | **FALSIFIED as standalone anchor** - Sweep-reversal anchor with textbook logic loses. Runner comment confirms: "sweep_reversal anchor, which failed walk-forward". **NOT falsified as part:** Sweep as first step in sweep->displacement->retracement + veto + LIMIT (Exness model) is NOT the same logic. Do NOT conflate. |
| `ablation.py`, `aoc_ablation.py`, `context_ablation.py`, `validation.py` | Phase 3/4/4.5 ablations, winning-combo test | **WIRED-BACKTEST-ONLY infrastructure** - Wired via `validation.py`, `modes/config.py`, `experiments/*`, `scripts/backtest3/4/aoc.py`. Not strategies. Keep. |
| `mtf12.py` | OLD OANDA R model | **SUPERSEDED, not deleted** - Kept for history. Do NOT use for Exness decisions. Replaced by `real_trader.py` + `exness12.py`. |
| `real_trader.py` | Real-trader R model | **WIRED-BACKTEST-ONLY, current R model** - LIMIT at POI, sweep->disp->retrace, AMD, killzones, max-stop 2.5x ATR. |
| `exness12.py` | Exness timing model | **WIRED-BACKTEST-ONLY, current Exness model** - $12, 0.01 lot, 1:2000/unlimited, timing, margin/stop-out. Swing $12->$424.50, 0 blow-ups. |
| `mtf_data.py`, `data.py` | Data fetching | **INFRA** - OANDA range fetch, GC=F fetch. |

## 6. Features - `goldenswing/features/` - Indicators, NOT strategies

All wired via `features/engine.py`. None is a standalone trigger. Use as context only:
- `aoc.py` (area of concentration), `bollinger.py`, `divergence.py` (RSI), `fibonacci.py`, `footprints.py`, `ichimoku.py`, `patterns.py`, `phases.py`, `relstrength.py`, `supply_demand.py`, `traps.py` (godce-independent), `trendlines.py`, `volume.py`, `mtf.py`, `gating.py`, `state.py`, `common.py`, `config.py`.
- Verdict: **FEATURES, LIVE via detector, never triggers.** Do NOT revive any as standalone entry.

## 7. Demo repo - `~/workspace/goldenswing-exness-swing-demo/` - What was carried over vs left behind

Carried over (Swing only):
- Swing logic: D1/H4/H1, sweep->displacement->retracement->LIMIT AT POI, AMD, killzones, Exness $12 / 0.01 / 1:2000 / unlimited.
Left behind (deliberately, not deleted from main repo):
- All experiments (§2), GODCE (§3), modes (§4), OANDA R harness, Intraday/Scalp/M1 presets, live playbooks (§1).
- Reason: Demo is Swing-only proven preset. Intraday/Scalp blew single $12, M1 loses. Do NOT add them to demo until veto + forward paper proves them.

## 8. Revival priority (if user wants to revive)

1. **FVG Limit playbook** - Already live-capable, just add to runner. Lowest effort, highest value. Swing/Intraday.
2. **OTE entry zone** - POI variant for Swing. Needs veto + Exness timing.
3. **Engulf trigger** - Fast trigger for Scalp. Needs veto + timing.
4. **Retest / Placement T3** - Retest confirmation for Intraday. Needs veto + timing.
5. **GODCE router** - Regime filter only, not entry. Needs forward paper.
6. **Marco sweep-entry** - Aggressive, high risk, Scalp only. Needs veto + timing + strict paper.
7. **Body-stop / Bodybreak** - Stop/confirmation variants, not standalone.

Do NOT revive: `opening_range` for XAU (QQQ only), `bollinger` / `divergence` / `ichimoku` as standalone triggers, `min_rr` as R filter in Exness model (use $ move instead).

## What's still weak

- No experiment has been re-run with hard veto (D1 veto for Intraday, H4 veto for Scalp) + Exness timing (0.01 lot, 1:2000/unlimited, $12). Prior experiment numbers are OANDA / textbook / pre-veto, so they cannot be trusted for revival decisions.
- `placement` T1/T2/T3 verdicts are from old harness, not Exness timing.
- GODCE regime filter has no forward paper proof.
- No live forward paper for any revived candidate yet. Revival = backtest with veto + Exness timing + forward paper, in that order.
- Dynamic imports: grep found no `importlib` / `import_module` / `__import__` in `goldenswing/` except `CallbackRegistry` (execution, not strategy registry). No hidden strategy registry found. If a strategy is loaded via string in scripts, it is listed above via script grep.
