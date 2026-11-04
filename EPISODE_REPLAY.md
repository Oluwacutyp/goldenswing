# GoldenSwing — Episode Replay (WS4 acceptance battery)

Owner-approved missed-move program, workstream 4 of 4: history re-run through the CURRENT engine code (WS1 thesis revision `ff13855`, WS2 Hot Scalp targets `041fbbc`, WS3 H1 handoff `e76b40e`) using only information available at each moment, graded against the live journal (16,509 records, 2026-10-14 -> 2026-11-04). Offline: no orders, no broker, no paper loop. Every stage transition is traceable to a journaled fact or stamped replay-derived.

**Timestamp law** — every forecast is four facts: `forecast_bar_ts`, `forecast_input_cutoff_ts`, `forecast_generated_ts`, `forecast_available_ts`. A forecast counts only from its available time; as-of-bar and operationally-available results are reported separately throughout.

## Program verdicts

| Workstream | Verdict | Evidence |
|---|---|---|
| WS1 thesis revision | **DEMONSTRATED** | replay keeper events: ['thesis_created', 'thesis_child_registered', 'thesis_maintained', 'thesis_invalidated', 'thesis_orphaned', 'thesis_created', 'thesis_maintained']; live journal events: ['thesis_created', 'thesis_maintained', 'thesis_created', 'thesis_created'] (three runs, no revision event of any kind in 16,509 records); invalidation at 2026-11-03 15:00:00+00:00: invalidation observed: price 3944.59 crossed the thesis invalidation price 3944.42 (plan: aligned candidate fvg_limit); orphan cut journaled: thesis_orphaned |
| WS2 Hot Scalp targets | **DEMONSTRATED** | window geometry blocks by direction: {'1': 78, '-1': 9} (of 121 across Nov 3-4); Exhibit A (hs-63dc10ef): live-blocked at net_rr 0.0764 against a spent micro-swing — current engine emits candidate_shadow, plan standard, legs [{'price': 3955.51, 'verdict': 'selected', 'net_rr': 1.4562063250486021}, {'price': 3964.5, 'verdict': 'viable', 'net_rr': 3.03176233205145}]; Exhibit B (hs-ade06913): current engine still blocks it (candidate_blocked), its one true objective pricing 1.1263856423421663 < 1.20 — the floor was not lowered and no target was invented |
| WS3 H1 handoff | **DEMONSTRATED** | both journal scan candidates (live disposition: observed / not_attempted, then silence) are preserved as tracked instances and end in named terminals: {'inversion:2026-11-03 23:00:00+00:00:1:3927.03': 'expired', 'order_block:2026-11-04 05:00:00+00:00:-1:3972.51': 'invalidated'}; orders placed by the scan path: 0 (registration is not authority — routing still requires a trigger fire) |
| Timestamp law / forecast availability | **DEMONSTRATED** | the 17:00Z bar's forecast exists in four timestamps (bar 2026-11-03T17:00:00+00:00, cutoff 2026-11-03T18:00:00+00:00, generated as journaled 2026-11-02 00:00:00+00:00, available 2026-11-03T18:51:56.817346+00:00); not consumable at bar time (visible then: False), consumable from availability (visible at 18:52Z: True), held for 11.13h before the 06:00Z scan; as-of-bar it called the rally (12h run_up 95.35pts) |
| Signal chains (centerpiece) | **ACCOUNTED** | 10 journaled signals: 2 completed chains (sum r_net 0.2608), 3 accepted but never filled in the journal, 5 refused at the gate with reasons the current funnel taxonomy classifies — no chain ends unnamed |
| Signal chains (trend_oct29) | **ACCOUNTED** | 10 journaled signals: 4 completed chains (sum r_net 3.8688), 0 accepted but never filled in the journal, 6 refused at the gate with reasons the current funnel taxonomy classifies — no chain ends unnamed |
| Signal chains (range_oct28) | **ACCOUNTED** | 6 journaled signals: 1 completed chains (sum r_net 0.5021), 0 accepted but never filled in the journal, 5 refused at the gate with reasons the current funnel taxonomy classifies — no chain ends unnamed |

## Episode: The missed move — Nov 3-4, 2026 (the +95pt rally the engine stayed bearish through)

Window: 2026-11-03T12:00:00+00:00 -> 2026-11-04T12:16:43+00:00

Why this episode (journal evidence):
- Ensemble forecast for the 2026-11-03 17:00Z bar: 24h up-rank 85.1485, journaled 2026-11-03T18:51:56Z in restart catch-up (forecast record, D1 cadence, alert_fired).
- Its 12h reconciliation (journaled 2026-11-04T07:00:10Z): run_up +95.35pts, ret +2.395% — the rally was real and the forecast called it.
- Live posture stayed bearish: judgment_thesis records show bearish thesis created 13:00Z, maintained 13:30Z, then re-created fresh at 15:00Z and 16:00Z by restarted runs — never weakened, never invalidated.
- Hot Scalp armed 232 setups in-window and blocked 87 candidates on geometry.net_rr_below_min (121 across Nov 3-4).
- Judgment shadow TOOK the 2026-11-04 06:00Z order_block short (expected_r +0.7287); reconciliations graded it -1.8647R (08:00Z) and -2.7504R (11:00Z).

### The forecast, in four timestamps

| Fact | Value |
|---|---|
| forecast_bar_ts | 2026-11-03T17:00:00+00:00 |
| forecast_input_cutoff_ts | 2026-11-03T18:00:00+00:00 |
| forecast_generated_ts (as journaled) | 2026-11-02 00:00:00+00:00 |
| forecast_available_ts | 2026-11-03T18:51:56.817346+00:00 |
| as-of-bar (hindsight grading) | 24h up-rank 85.1485; realized 12h run_up 95.35pts, ret 2.395% |
| operationally available | from 2026-11-03T18:51:56.817346+00:00; 11.13h before the 06:00Z scan |

### Thesis leg (WS1) — replay events vs live journal

| When | Replay (current code) | Live journal |
|---|---|---|
| Nov 3 cycles | ['thesis_created', 'thesis_child_registered', 'thesis_maintained', 'thesis_invalidated', 'thesis_orphaned', 'thesis_created', 'thesis_maintained'] | thesis_created x3 across restarts, thesis_maintained x1, revision events: none |

### Handoff leg (WS3) — scan candidates

| Setup | Replay terminal | History |
|---|---|---|
| inversion:2026-11-03 23:00:00+00:00:1:3927.03 | **expired** | candidate_detected -> setup_armed -> expired |
| order_block:2026-11-04 05:00:00+00:00:-1:3972.51 | **invalidated** | candidate_detected -> setup_armed -> invalidated |

### Hot Scalp leg (WS2) — exhibits

- Exhibit A (`hs-63dc10ef330c904b42036f37ed9d`): armed 2026-11-04 01:55Z, FVG 3942.517-3947.592, stop 3941.375078; live journal blocked it at 02:02:02Z, geometry.net_rr_below_min, legacy target 3947.646, net_rr 0.076383 — current engine: **candidate_shadow**, plan standard, legs [{'price': 3955.51, 'verdict': 'selected', 'net_rr': 1.4562063250486021}, {'price': 3964.5, 'verdict': 'viable', 'net_rr': 3.03176233205145}], legacy comparison reproduced: {'target': 3947.646, 'net_rr': 0.07638275703179195, 'verdict': 'blocked'}
- Exhibit B (`hs-ade06913864859add85e251814e6`): armed 2026-11-04 07:00Z, FVG 4020.418-4021.269, stop 4019.056827; live journal blocked it at 07:04:00Z, net_rr 0.096269, legacy target 4020.913 inside the zone; its one true objective 4022.522 still prices below the 1.20 floor — current engine: **candidate_blocked**, plan blocked, legs [{'price': 4022.522, 'verdict': 'rejected', 'net_rr': 1.1263856423421663}], legacy comparison reproduced: {'target': 4020.913, 'net_rr': 0.09626893150725543, 'verdict': 'blocked'}

### Stage table

| Time (UTC) | Subject | Stage | Source | Detail |
|---|---|---|---|---|
| 2026-11-03T12:20:01.189768+00:00 | qm_m5 @2026-11-03T12:20 | candidate_detected | journal |  |
| 2026-11-03T12:20:01.189768+00:00 | qm_m5 @2026-11-03T12:20 | trigger_fired | journal |  |
| 2026-11-03T12:21:22.801407+00:00 | qm_m5 @2026-11-03T12:20 | filled | journal |  |
| 2026-11-03T12:22:16.188264+00:00 | qm_m5 @2026-11-03T12:20 | closed | journal | journal chain completed: filled 2026-11-03T12:21:22.801407+00:00, closed 2026-11-03T12:22:16.188264+00:00 at 3911.904 (broker_closed), r_net -1.000111994624179 |
| 2026-11-03 13:00:00+00:00 | thesis thesis-1 | thesis_created | replay-derived | regime direction is down; phase is distribution; qualified flow skews bearish (0 long vs 1 short qualified) |
| 2026-11-03 13:00:00+00:00 | thesis thesis-1 | thesis_child_registered | replay-derived |  |
| 2026-11-03T13:00:23.483410+00:00 | sweep_reversal @2026-11-03T13:00 | candidate_detected | journal |  |
| 2026-11-03T13:00:23.483410+00:00 | sweep_reversal @2026-11-03T13:00 | rejected | journal | live decision 'below_floor' — refusal re-classified by current funnel taxonomy as floor.total (category 7) |
| 2026-11-03T13:00:23.749037+00:00 | fvg_limit @2026-11-03T13:00 | candidate_detected | journal |  |
| 2026-11-03T13:00:23.749037+00:00 | fvg_limit @2026-11-03T13:00 | rejected | journal | live decision 'below_floor' — refusal re-classified by current funnel taxonomy as floor.total (category 7) |
| 2026-11-03T13:00:23.983414+00:00 | inversion @2026-11-03T13:00 | candidate_detected | journal |  |
| 2026-11-03T13:00:23.983414+00:00 | inversion @2026-11-03T13:00 | rejected | journal | live decision 'below_floor' — refusal re-classified by current funnel taxonomy as floor.total (category 7) |
| 2026-11-03 13:30:27+00:00 | thesis thesis-1 | thesis_maintained | replay-derived |  |
| 2026-11-03T13:30:37.004951+00:00 | sweep_reversal @2026-11-03T13:30 | candidate_detected | journal |  |
| 2026-11-03T13:30:37.004951+00:00 | sweep_reversal @2026-11-03T13:30 | rejected | journal | live decision 'below_floor' — refusal re-classified by current funnel taxonomy as floor.total (category 7) |
| 2026-11-03T13:30:37.448957+00:00 | fvg_limit @2026-11-03T13:30 | candidate_detected | journal |  |
| 2026-11-03T13:30:37.448957+00:00 | fvg_limit @2026-11-03T13:30 | rejected | journal | live decision 'below_floor' — refusal re-classified by current funnel taxonomy as floor.total (category 7) |
| 2026-11-03T14:13:36.378923+00:00 | forecast D1 bar 2026-11-03T13:00:00+00:00 | forecast_journaled | journal | bar 2026-11-03T13:00:00+00:00 · inputs through 2026-11-03T14:00:00+00:00 · generated (as journaled) 2026-11-02 00:00:00+00:00 · available 2026-11-03T14:13:36.378923+00:00 · 24h up-rank 39.6040 |
| 2026-11-03T14:13:37.357049+00:00 | forecast H1 bar 2026-11-03T13:00:00+00:00 | forecast_journaled | journal | bar 2026-11-03T13:00:00+00:00 · inputs through 2026-11-03T14:00:00+00:00 · generated (as journaled) 2026-11-03 13:00:00+00:00 · available 2026-11-03T14:13:37.357049+00:00 · 24h up-rank 39.6040 |
| 2026-11-03 15:00:00+00:00 | thesis thesis-1 | thesis_invalidated | replay-derived | invalidation observed: price 3944.59 crossed the thesis invalidation price 3944.42 (plan: aligned candidate fvg_limit) |
| 2026-11-03 15:00:00+00:00 | thesis thesis-1 | thesis_orphaned | replay-derived | invalidation observed: price 3944.59 crossed the thesis invalidation price 3944.42 (plan: aligned candidate fvg_limit) |
| 2026-11-03 15:00:00+00:00 | thesis thesis-2 | thesis_created | replay-derived | regime direction is down; phase is distribution |
| 2026-11-03T15:43:52.077208+00:00 | forecast D1 bar 2026-11-03T14:00:00+00:00 | forecast_journaled | journal | bar 2026-11-03T14:00:00+00:00 · inputs through 2026-11-03T15:00:00+00:00 · generated (as journaled) 2026-11-02 00:00:00+00:00 · available 2026-11-03T15:43:52.077208+00:00 · 24h up-rank 62.3762 |
| 2026-11-03T15:43:53.268294+00:00 | forecast H1 bar 2026-11-03T14:00:00+00:00 | forecast_journaled | journal | bar 2026-11-03T14:00:00+00:00 · inputs through 2026-11-03T15:00:00+00:00 · generated (as journaled) 2026-11-03 14:00:00+00:00 · available 2026-11-03T15:43:53.268294+00:00 · 24h up-rank 62.3762 |
| 2026-11-03 16:00:00+00:00 | thesis thesis-2 | thesis_maintained | replay-derived |  |
| 2026-11-03T16:20:06.927590+00:00 | forecast D1 bar 2026-11-03T15:00:00+00:00 | forecast_journaled | journal | bar 2026-11-03T15:00:00+00:00 · inputs through 2026-11-03T16:00:00+00:00 · generated (as journaled) 2026-11-02 00:00:00+00:00 · available 2026-11-03T16:20:06.927590+00:00 · 24h up-rank 78.2178 |
| 2026-11-03T16:20:07.999348+00:00 | forecast H1 bar 2026-11-03T15:00:00+00:00 | forecast_journaled | journal | bar 2026-11-03T15:00:00+00:00 · inputs through 2026-11-03T16:00:00+00:00 · generated (as journaled) 2026-11-03 15:00:00+00:00 · available 2026-11-03T16:20:07.999348+00:00 · 24h up-rank 78.2178 |
| 2026-11-03T18:51:56.817346+00:00 | forecast D1 bar 2026-11-03T17:00:00+00:00 | forecast_journaled | journal | bar 2026-11-03T17:00:00+00:00 · inputs through 2026-11-03T18:00:00+00:00 · generated (as journaled) 2026-11-02 00:00:00+00:00 · available 2026-11-03T18:51:56.817346+00:00 · 24h up-rank 85.1485 |
| 2026-11-03T18:51:56.817346+00:00 | forecast D1 bar 2026-11-03T17:00:00+00:00 | availability_window | replay-derived | NOT consumable at bar time (journaled in restart catch-up); operationally available from 2026-11-03T18:51:56.817346+00:00 — 11.13h before the 2026-11-04 06:00Z scan |
| 2026-11-03T18:51:58.284263+00:00 | forecast H1 bar 2026-11-03T17:00:00+00:00 | forecast_journaled | journal | bar 2026-11-03T17:00:00+00:00 · inputs through 2026-11-03T18:00:00+00:00 · generated (as journaled) 2026-11-03 17:00:00+00:00 · available 2026-11-03T18:51:58.284263+00:00 · 24h up-rank 85.1485 |
| 2026-11-03T19:15:02.032688+00:00 | qm_m5 @2026-11-03T19:15 | candidate_detected | journal |  |
| 2026-11-03T19:15:02.032688+00:00 | qm_m5 @2026-11-03T19:15 | trigger_fired | journal |  |
| 2026-11-03T19:15:02.032688+00:00 | qm_m5 @2026-11-03T19:15 | execution_evaluated | journal | live decision 'accepted'; the journal carries no open (fill) record for this signal — evaluated, never filled, and no fill is claimed |
| 2026-11-03T19:20:02.388850+00:00 | qm_m5 @2026-11-03T19:20 | candidate_detected | journal |  |
| 2026-11-03T19:20:02.388850+00:00 | qm_m5 @2026-11-03T19:20 | trigger_fired | journal |  |
| 2026-11-03T19:20:02.388850+00:00 | qm_m5 @2026-11-03T19:20 | execution_evaluated | journal | live decision 'accepted'; the journal carries no open (fill) record for this signal — evaluated, never filled, and no fill is claimed |
| 2026-11-03T23:40:01.738970+00:00 | qm_m5 @2026-11-03T23:40 | candidate_detected | journal |  |
| 2026-11-03T23:40:01.738970+00:00 | qm_m5 @2026-11-03T23:40 | trigger_fired | journal |  |
| 2026-11-03T23:56:32.064242+00:00 | qm_m5 @2026-11-03T23:40 | filled | journal |  |
| 2026-11-04T00:00:00+00:00 | gate inversion | gate_check | replay-derived | verdict=independent thesis_id=None — the window's long scan candidate reaches the thesis gate |
| 2026-11-04 00:00:00+00:00 | inversion | candidate_detected | journal | live disposition was not_attempted / path.trigger_not_fired — the setup then vanished from the live chain |
| 2026-11-04 00:00:00+00:00 | inversion:2026-11-03 23:00:00+00:00:1:3927.03 | setup_armed | replay-derived | inversion dir 1 entry 3927.025 stop 3922.81 — store stage after step |
| 2026-11-04T00:00:42.130510+00:00 | forecast D1 bar 2026-11-03T23:00:00+00:00 | forecast_journaled | journal | bar 2026-11-03T23:00:00+00:00 · inputs through 2026-11-04T00:00:00+00:00 · generated (as journaled) 2026-11-03 00:00:00+00:00 · available 2026-11-04T00:00:42.130510+00:00 · 24h up-rank 72.2772 |
| 2026-11-04T00:00:43.171661+00:00 | forecast H1 bar 2026-11-03T23:00:00+00:00 | forecast_journaled | journal | bar 2026-11-03T23:00:00+00:00 · inputs through 2026-11-04T00:00:00+00:00 · generated (as journaled) 2026-11-03 23:00:00+00:00 · available 2026-11-04T00:00:43.171661+00:00 · 24h up-rank 72.2772 |
| 2026-11-04T00:34:48.512822+00:00 | qm_m5 @2026-11-03T23:40 | closed | journal | journal chain completed: filled 2026-11-03T23:56:32.064242+00:00, closed 2026-11-04T00:34:48.512822+00:00 at 3922.911 (broker_closed), r_net 1.26092450758057 |
| 2026-11-04T01:58:40+00:00 | hs-63dc10ef330c904b42036f37ed9d | candidate_shadow | replay-derived | hierarchy [3955.51, 3964.5] · plan standard · legs [(3955.51, 'selected', 1.456), (3964.5, 'viable', 3.032)] |
| 2026-11-04T06:00:00+00:00 | gate order_block | gate_check | replay-derived | verdict=child thesis_id=thesis-2 — the shadow-taken short reaches the thesis gate |
| 2026-11-04 06:00:00+00:00 | order_block | candidate_detected | journal | live disposition was not_attempted / path.trigger_not_fired — the setup then vanished from the live chain |
| 2026-11-04 06:00:00+00:00 | order_block:2026-11-04 05:00:00+00:00:-1:3972.51 | setup_armed | replay-derived | order_block dir -1 entry 3972.506 stop 3999.328 — store stage after step |
| 2026-11-04T07:00:00+00:00 | inversion:2026-11-03 23:00:00+00:00:1:3927.03 | expired | replay-derived | inversion dir 1 entry 3927.025 stop 3922.81 — store stage after step |
| 2026-11-04T07:00:00+00:00 | order_block:2026-11-04 05:00:00+00:00:-1:3972.51 | invalidated | replay-derived | order_block dir -1 entry 3972.506 stop 3999.328 — store stage after step |
| 2026-11-04T07:04:20+00:00 | hs-ade06913864859add85e251814e6 | candidate_blocked | replay-derived | hierarchy [4022.522] · plan blocked · legs [(4022.522, 'rejected', 1.126)] |
| 2026-11-04T11:00:02.363197+00:00 | qm_m5 @2026-11-04T11:00 | candidate_detected | journal |  |
| 2026-11-04T11:00:02.363197+00:00 | qm_m5 @2026-11-04T11:00 | trigger_fired | journal |  |
| 2026-11-04T11:00:02.363197+00:00 | qm_m5 @2026-11-04T11:00 | execution_evaluated | journal | live decision 'accepted'; the journal carries no open (fill) record for this signal — evaluated, never filled, and no fill is claimed |

### Honest residuals

- Assessment-stream discontinuity: the live journal's judgment assessments stop at 16:00Z on Nov 3 (restarts re-created the thesis per run; Nov 4 runs journaled no thesis records at all). WS1's revision machinery can only revise on assessments it receives — in replay, thesis-1 is invalidated at 15:00Z and the same update creates its scenario-only successor (maintained at 16:00Z), which is still the active thesis at the 06:00Z scan; the gate therefore adopts the thesis-aligned order_block short as a child (verdict: child), and what ends that setup is WS3's close-rule invalidation before any trigger fire — not the gate. The fix this points at is keeper persistence / assessment continuity across restarts, which is outside WS1's shipped scope.
- The handoff leg's instances carry thesis_id null: that leg runs the sweep path without judgment cycles, so no keeper adoption resolves there. Thesis-link stamping on scan instances is pinned by WS3's own suite (test_scan_instance_stamped_with_parent_thesis) and the keeper-side adoption is demonstrated in this replay's thesis leg (thesis_child_registered at 13:00Z, cut by thesis_orphaned at 15:00Z).
- Weakening is not exercised by this episode's tape: at the 15:00Z cycle the price fact (3944.595 at/above the thesis's own invalidation price 3944.42) closes the thesis structurally in the same update that carries the erosion (winner_p peak 0.7568 -> 0.5337), and invalidation is checked before revision. The weakening path itself is pinned by WS1's own suite; no recorded assessment in this journal erodes a surviving thesis without also breaking it.
- Routing still needs a trigger fire the journal does not evidence: the 00:00Z inversion long was preserved and its target (3939.345) was crossed by the 02:00Z tape mark (3947.047) while it sat armed, but the window's trigger records are __trigger__ evaluations (25, mostly not_eligible / path.invalidated) and none binds this instance — so under current code it ends expired, unfilled. Preservation and a named end are demonstrated; capture of that long is not claimed.
- The forecast was operationally available for ~11.1h before the 06:00Z scan and the system's posture never moved: E3 keeps forecasts journal-only until live calibration earns weight (top-band lift >= 1.25, >= 20 resolutions), so this is doctrine operating as designed, recorded here as the standing tension the owner's calibration program has to resolve — not a WS1-WS3 defect.
- Hot Scalp repricing is exhibit-deep, not census-deep: the journal's blocked-candidate records carry risk / reward / net_rr but not the M5 structure the current planner prices from, so the other geometry-blocked setups of the window cannot be re-priced without inventing bars. The two exhibits are the episode's journal-exact representatives (one each way); the census (87 blocks in-window, 78 of them longs) is reported as journal fact.

## Episode: Trend handled — Thu Oct 29, 2026 (one-way short day)

Window: 2026-10-29T00:00:00+00:00 -> 2026-10-29T23:59:59+00:00

Why this episode (journal evidence):
- Four qm_m5 short signals accepted (05:15, 09:10, 13:40, 22:55Z) and all four closed positive: +0.5336R, +1.0526R, +0.7982R, +1.4843R (sum +3.8687R).
- Entries step down through the day (4044.594 -> 4035.206 -> 4034.112 -> 4025.566) and exits step down with them — a one-way trend the engine rode repeatedly.
- Signal contexts read down/clean_trend (e.g. 05:15Z signal regime 'down/clean_trend/normal/distribution/asia', posture trend_day); the two counter-trend/long signals of the day were gated.

### Stage table

| Time (UTC) | Subject | Stage | Source | Detail |
|---|---|---|---|---|
| 2026-10-29T00:00:00+00:00 | roster scans | detection_census | journal | 719 playbook evaluations in window; outcomes {'no_setup': 704, 'candidate_emitted': 15} |
| 2026-10-29T00:05:02.834787+00:00 | qm_m5 @2026-10-29T00:05 | candidate_detected | journal |  |
| 2026-10-29T00:05:02.834787+00:00 | qm_m5 @2026-10-29T00:05 | rejected | journal | live decision 'gated' — refusal re-classified by current funnel taxonomy as direction.flip (category 5) |
| 2026-10-29T01:30:14.191500+00:00 | range_sweep_reclaim @2026-10-29T01:30 | candidate_detected | journal |  |
| 2026-10-29T01:30:14.191500+00:00 | range_sweep_reclaim @2026-10-29T01:30 | rejected | journal | live decision 'below_floor' — refusal re-classified by current funnel taxonomy as floor.total (category 7) |
| 2026-10-29T05:15:59.362399+00:00 | qm_m5 @2026-10-29T05:15 | candidate_detected | journal |  |
| 2026-10-29T05:15:59.362399+00:00 | qm_m5 @2026-10-29T05:15 | trigger_fired | journal |  |
| 2026-10-29T05:17:18.728961+00:00 | qm_m5 @2026-10-29T05:15 | filled | journal |  |
| 2026-10-29T05:30:41.966160+00:00 | qm_m5 @2026-10-29T05:15 | closed | journal | journal chain completed: filled 2026-10-29T05:17:18.728961+00:00, closed 2026-10-29T05:30:41.966160+00:00 at 4040.7309999999998 (broker_closed), r_net 0.5336135490205344 |
| 2026-10-29T09:10:03.962631+00:00 | qm_m5 @2026-10-29T09:10 | candidate_detected | journal |  |
| 2026-10-29T09:10:03.962631+00:00 | qm_m5 @2026-10-29T09:10 | trigger_fired | journal |  |
| 2026-10-29T09:11:04.795622+00:00 | qm_m5 @2026-10-29T09:10 | filled | journal |  |
| 2026-10-29T09:55:19.254031+00:00 | qm_m5 @2026-10-29T09:10 | closed | journal | journal chain completed: filled 2026-10-29T09:11:04.795622+00:00, closed 2026-10-29T09:55:19.254031+00:00 at 4029.225 (stop), r_net 1.0526421652745175 |
| 2026-10-29T11:10:04.381574+00:00 | qm_m5 @2026-10-29T11:10 | candidate_detected | journal |  |
| 2026-10-29T11:10:04.381574+00:00 | qm_m5 @2026-10-29T11:10 | rejected | journal | live decision 'gated' — refusal re-classified by current funnel taxonomy as direction.flip (category 5) |
| 2026-10-29T11:55:14.821790+00:00 | qm_m5 @2026-10-29T11:55 | candidate_detected | journal |  |
| 2026-10-29T11:55:14.821790+00:00 | qm_m5 @2026-10-29T11:55 | rejected | journal | live decision 'gated' — refusal re-classified by current funnel taxonomy as direction.flip (category 5) |
| 2026-10-29T13:40:02.952748+00:00 | qm_m5 @2026-10-29T13:40 | candidate_detected | journal |  |
| 2026-10-29T13:40:02.952748+00:00 | qm_m5 @2026-10-29T13:40 | trigger_fired | journal |  |
| 2026-10-29T13:40:44.437524+00:00 | qm_m5 @2026-10-29T13:40 | filled | journal |  |
| 2026-10-29T13:42:58.297519+00:00 | qm_m5 @2026-10-29T13:40 | closed | journal | journal chain completed: filled 2026-10-29T13:40:44.437524+00:00, closed 2026-10-29T13:42:58.297519+00:00 at 4028.26 (broker_closed), r_net 0.7981801088232727 |
| 2026-10-29T20:05:01.965282+00:00 | qm_m5 @2026-10-29T20:05 | candidate_detected | journal |  |
| 2026-10-29T20:05:01.965282+00:00 | qm_m5 @2026-10-29T20:05 | rejected | journal | live decision 'gated' — refusal re-classified by current funnel taxonomy as direction.flip (category 5) |
| 2026-10-29T20:10:01.955913+00:00 | qm_m5 @2026-10-29T20:10 | candidate_detected | journal |  |
| 2026-10-29T20:10:01.955913+00:00 | qm_m5 @2026-10-29T20:10 | rejected | journal | live decision 'gated' — refusal re-classified by current funnel taxonomy as direction.flip (category 5) |
| 2026-10-29T22:55:02.120429+00:00 | qm_m5 @2026-10-29T22:55 | candidate_detected | journal |  |
| 2026-10-29T22:55:02.120429+00:00 | qm_m5 @2026-10-29T22:55 | trigger_fired | journal |  |
| 2026-10-29T22:55:50.078510+00:00 | qm_m5 @2026-10-29T22:55 | filled | journal |  |
| 2026-10-29T23:03:06.892366+00:00 | qm_m5 @2026-10-29T22:55 | closed | journal | journal chain completed: filled 2026-10-29T22:55:50.078510+00:00, closed 2026-10-29T23:03:06.892366+00:00 at 4018.718 (broker_closed), r_net 1.4843164576559502 |

### Honest residuals

- No thesis or forecast records exist for Oct 29 (the Judgment Core went live on EC2 on Nov 3): the thesis and forecast stages are absent from this window's table by evidence, not by omission. WS1/WS2/WS3 verdicts rest on the centerpiece; this episode grades regime behavior of the chain that did exist.

## Episode: Range / chop — Wed Oct 28, 2026 (two-sided rotation, no follow-through)

Window: 2026-10-28T00:00:00+00:00 -> 2026-10-28T23:59:59+00:00

Why this episode (journal evidence):
- Signal entries oscillate inside a ~42pt band all day (4033.147 to 4074.928) in BOTH directions; five of six signals were gated, the one accepted long closed +0.5021R.
- Hot Scalp armed 101 setups split 64 long / 37 short; 57 expired untouched, 35 were geometry-blocked, 8 invalidated — and the sleeve filled nothing.
- The H1 roster scans emitted 24 candidates across the day (inversion 10, fvg_limit 4, retracement_trap 2, others) and the M5/trigger routes 6 more — every emitted candidate was refused downstream (a journal sample refusal: gate.data, touch 11 bars stale) and none became a fill; the day's only fill was the accepted qm_m5 long.

### Stage table

| Time (UTC) | Subject | Stage | Source | Detail |
|---|---|---|---|---|
| 2026-10-28T00:00:00+00:00 | roster scans | detection_census | journal | 720 playbook evaluations in window; outcomes {'no_setup': 690, 'candidate_emitted': 30} |
| 2026-10-28T01:05:02.552695+00:00 | qm_m5 @2026-10-28T01:05 | candidate_detected | journal |  |
| 2026-10-28T01:05:02.552695+00:00 | qm_m5 @2026-10-28T01:05 | rejected | journal | live decision 'gated' — refusal re-classified by current funnel taxonomy as direction.flip (category 5) |
| 2026-10-28T12:20:01.850649+00:00 | qm_m5 @2026-10-28T12:20 | candidate_detected | journal |  |
| 2026-10-28T12:20:01.850649+00:00 | qm_m5 @2026-10-28T12:20 | rejected | journal | live decision 'gated' — refusal re-classified by current funnel taxonomy as direction.flip (category 5) |
| 2026-10-28T14:25:02.233097+00:00 | qm_m5 @2026-10-28T14:25 | candidate_detected | journal |  |
| 2026-10-28T14:25:02.233097+00:00 | qm_m5 @2026-10-28T14:25 | rejected | journal | live decision 'gated' — refusal re-classified by current funnel taxonomy as direction.flip (category 5) |
| 2026-10-28T17:20:01.914577+00:00 | qm_m5 @2026-10-28T17:20 | candidate_detected | journal |  |
| 2026-10-28T17:20:01.914577+00:00 | qm_m5 @2026-10-28T17:20 | trigger_fired | journal |  |
| 2026-10-28T17:20:02.321961+00:00 | qm_m5 @2026-10-28T17:20 | filled | journal |  |
| 2026-10-28T18:00:27.855087+00:00 | qm_m5 @2026-10-28T17:20 | closed | journal | journal chain completed: filled 2026-10-28T17:20:02.321961+00:00, closed 2026-10-28T18:00:27.855087+00:00 at 4042.8900000000003 (broker_closed), r_net 0.5021131892154809 |
| 2026-10-28T20:00:34.990420+00:00 | range_mean_reversion @2026-10-28T20:00 | candidate_detected | journal |  |
| 2026-10-28T20:00:34.990420+00:00 | range_mean_reversion @2026-10-28T20:00 | rejected | journal | live decision 'gated' — refusal re-classified by current funnel taxonomy as direction.flip (category 5) |
| 2026-10-28T20:00:35.271790+00:00 | range_compression_breakout @2026-10-28T20:00 | candidate_detected | journal |  |
| 2026-10-28T20:00:35.271790+00:00 | range_compression_breakout @2026-10-28T20:00 | rejected | journal | live decision 'gated' — refusal re-classified by current funnel taxonomy as gate.location (category 3) |

### Honest residuals

- Range-day selectivity is graded on the journal's own decisions (5 of 6 signals gated, the sleeve's 101 armed setups filling nothing): the replay confirms every chain ends named under the current taxonomy, but does not re-derive the weighed judgments — the confluence engine's inputs (frames) are not in the journal, and re-scoring from journal facts alone would be a parallel decision path, which this program does not build.

## Method notes

- Fixtures: `goldenswing/backtest/episodes/*.json`, extracted from the owner's live journal; each fixture carries its justification with journal timestamps.
- The handoff leg's frames are flat scaffolding anchored on journaled closes (the WS3 fixture pattern): the graded facts are the candidates' journaled identity and their dispositions, which do not depend on bar shape beyond the evaluated close — itself a journal fact.
- Safeguards were inputs, never tuned: the 1.20 net-R/R floor, the friction gate and the min-lot rule are exactly as shipped.
