# MT5 native D1 bars — provenance + data-quality notes
# ====================================================
# Source: /d1 export from the live MT5 engine on EC2
#   (goldenswing/execution/mt5_commands.py `/d1`), run 2026-10-15 ~06:13 UTC.
# Original file: workspace/user/files/mt5-d1-2000bars-20261015-061358_6_y36t.csv
#   (copied verbatim to data/MT5_XAUUSD_D1_native_2000bars_20261015.csv).
# Feed: MT5 copy_rates_from_pos(symbol, TIMEFRAME_D1, 0, 2000), closed bars only.
#   Never resampled. This is the FIRST native D1 dataset in the repo —
#   all earlier Daily QM validation used H1 bars resampled to D1.
#
# DATA QUALITY — read this before using the file:
#   Requested 2000 bars; MT5 returned only 727 (2023-06-26 -> 2026-10-14).
#   The 727 bars are NOT contiguous. Three segments (gap > 3 days breaks a segment):
#     seg 0: 2023-06-26 -> 2023-06-30  (5 bars — fragment, unusable alone)
#     seg 1: 2024-03-31 -> 2026-07-17  (718 bars — main contiguous block, ~2.3y)
#     seg 2: 2026-10-11 -> 2026-10-14  (4 bars — fragment, unusable alone)
#   Holes: 275 days (2023-06-30 -> 2024-03-31) and 86 days (2026-07-17 -> 2026-10-11).
#   The MT5 terminal's D1 history is fragmented; the reason is unknown
#   (terminal history limit / broker history gaps). Do NOT run pattern
#   detection across the holes: a "60-bar" lookback spanning a 275-day
#   hole is meaningless. scripts/validate_qm_d1.py --native validates
#   the longest contiguous segment only.
#
#   121 bars fall on Sundays (dayofweek=6), 0 on Saturdays. This is
#   standard MT5 behaviour: the broker's trading week opens Sunday
#   (short Sunday session candle). Timestamps are UTC midnight.
#   OHLC sanity: 0 violating rows, 0 duplicates, 0 NaNs.
#
#   CONSEQUENCE for the Oct 14 2026 replay: the owner's Oct 14 Daily QM
#   (~4,133-4,140 zone, drop to ~4,104) formed in the weeks BEFORE
#   Oct 11 — inside the 86-day hole. The 4-bar Oct 11-14 fragment cannot
#   support QM detection (formation needs up to 60 bars of lookback).
#   The Oct 14 replay is BLOCKED by the gap, not by the playbook.
#   The Oct 14 bar itself (O 4155.136 H 4170.448 L 4073.195 C 4111.85)
#   is consistent with the owner's description of a drop, but that is
#   not a pattern detection.
#
#   To get a replayable Oct 14 pattern, re-run /d1 on the EC2 terminal
#   after its D1 history is fully downloaded (or export with a date
#   range covering 2026-08 -> 2026-10-14 with no hole).
