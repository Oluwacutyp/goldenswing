# Evidence map — what actually decides, what is only recorded

Owner, 2026-10-26. Code-read from the playbooks, the `h1_trigger`
path (`_run_one_playbook`), and `_census_for_record`. The machine-readable
twin is `goldenswing/execution/evidence_map.py`; coverage of the full
registry is test-pinned, so a new playbook without a map row fails the suite.

## The two rules

1. **Census is never a decision input.** Chart patterns, premium/discount,
   breaker/mitigation blocks, relative strength (DXY/silver), engulfing and
   the named candle taxonomy are recorded on live evaluations only, rendered
   as "recorded, no decision weight", and read by no decision code. A census
   family on an assessment is groundwork, not integration. Legacy SMC is
   partially integrated in exactly the marker's sense: breaker + mitigation
   blocks are recorded from `live/patterns.py` helpers, while the structure /
   order-block / FVG helpers there remain unconsumed legacy duplicates — live
   SMC evidence comes from the playbooks and `live/zones.py`.
2. **A detector's own inputs are its decision inputs.** Every roster detector
   reads only its own timeframe's closed OHLC (plus its own tracker state and,
   where noted, session windows or the causally sliced H4 frame). No roster
   detector reads census families, DXY/silver, or another playbook's output.

## Shared trigger-path inputs (not any one playbook's detector)

- Asymmetry score (`live/asymmetry.py`) — the decision number.
- Hard lines only: declaration gates + manipulation filter.
- Per-style score floor (style/family, not a flat `min_score`).
- Journey/zone context (zone behind the setup, magnet, journey geometry).
- Structural D1 bias — weighed as evidence, **never a veto**.
- Order-path invariants outside any model: friction gate, margin, RiskPool,
  max drawdown / positions / losses.
- Funding events earn a held position's breakeven (management), never an entry.

## Per playbook

| Playbook | Route | Detector reads (decision-bearing) | Order authority |
|---|---|---|---|
| fvg_limit | h1_roster | Own-TF OHLC + FVG tracker; first touch of a holding gap, limit at near edge | h1_trigger streaming only; scans observe |
| fvg_midline | h1_roster | Own-TF OHLC + FVG tracker; fresh CHoCH, resting limit at midline of a fresh untraded FVG | h1_trigger streaming only; scans observe |
| inversion | h1_roster | Own-TF OHLC + own FVG registry; gap formed, close through the whole gap, later retest without reclaim — the retest close IS the entry, no second trigger predicate | h1_trigger streaming only; scans observe |
| judas_swing | h1_roster | Own-TF OHLC + DST-aware session windows; Asia-edge sweep that fails, then the London move | h1_trigger streaming only; scans observe |
| killzone_open | h1_roster | Own-TF OHLC + session blocks; session-range break + retest at a session open | h1_trigger streaming only; scans observe |
| narrative | h1_roster | Institutional chain over frames resampled from the input frame: HTF zone → sweep → displacement → CHoCH/BOS + FVG → entry | h1_trigger streaming only; scans observe |
| order_block | h1_roster | Own-TF OHLC; last opposing candle before a displacement bar | h1_trigger streaming only; scans observe |
| retracement_trap | h1_roster | Own-TF OHLC + swing fractals (n=3); false breakout trap at a confirmed swing level | h1_trigger streaming only; scans observe |
| sweep_reversal | h1_roster | Bar states on the entry frame; sweep → displacement + CHoCH/BOS confirmation | h1_trigger streaming only; scans observe |
| trendline_retest | h1_roster | Own-TF OHLC + trendline state; break, 2 bars away, first return touch within 12 bars | h1_trigger streaming only; scans observe |
| turtle_soup | h1_roster | Own-TF OHLC; failed breakout of a 20-bar extreme | h1_trigger streaming only; scans observe |
| v_reversal | h1_roster | Own-TF OHLC; sharp V spike, fresh 20-bar extreme + capitulation reclaim | h1_trigger streaming only; scans observe |
| range_sweep_reclaim | range_h4_h1 (§G) | H4 range state, causally sliced (only H4 bars closed by the H1 bar) + H1 refinement; sweep reclaimed | NONE in v1 — §G shadow |
| range_qm | range_h4_h1 (§G) | H4 range state, causally sliced + H1 refinement; QM inside the range | NONE in v1 — §G shadow |
| range_mean_reversion | range_h4_h1 (§G) | H4 range state, causally sliced + H1 refinement; fade of a range edge | NONE in v1 — §G shadow |
| range_compression_breakout | range_h4_h1 (§G) | H4 range state, causally sliced + H1 refinement; compression then breakout | NONE in v1 — §G shadow |
| qm_m5 | native M5 | M5 closed bars only; QM head/neckline/level, break of structure, retracement, resting limit at POI; validity window 1 bar; source must be live_close | Native M5 order path |
| qm_d1 | native D1 | D1 closed bars; same QM structure as qm_m5; validity window 1 bar; source must be live_close | Native D1 order path in the runner (see flags) |

## Flags — stated, not resolved by guess

- **narrative:** its chain frames are resampled from the input frame, so they
  carry no future data, but their granularity is derived, not native ticks.
  Do not treat narrative's M15/M5 reads as native M15/M5 bars.
- **sweep_reversal:** the detector docstring still describes a daily-bias veto
  (Phase-2 wording). The shared trigger path weighs D1 as evidence, never a
  veto; whether the detector itself still vetoes internally was not verified
  in this pass.
- **qm_d1:** the playbook docstring labels it SHADOW while the runner wires a
  live order path. The runner is treated as authority; the docstring needs
  reconciling, not silent editing.

## What this map does not claim

- NO-SETUP is a valid outcome, not a broken detector. This map says what each
  playbook *would* consume, not that any of them is profitable or calibrated.
- The interactive chart terminal (multi-timeframe, live inspection of candles,
  overlays, zones and decision evidence) does **not** exist in this repo. What
  exists is `charts/signal_view.py`: a per-signal, frozen-at-signal-time HTML
  page with timeframe switching over capped bars. The terminal stays on the
  deliverables list; this map does not pretend the signal view is it.
- Forecast status is not part of this map. The forecast bundle is a host
  artifact under `data/` (git-ignored, trained from the engine's own MT5
  history); whether a given EC2 box holds a current bundle is answered by its
  `/status`, not by this repo.
