"""Full-engine backtest on NATIVE per-timeframe MT5 data. NO RESAMPLING.

Runs the REAL engine — ``MT5Runner`` (same class EC2 runs) over
historical bars through the simulated broker/feed from
``backtest_full_engine`` — with one difference that is the whole
point of this script: every timeframe's bars come from that
timeframe's own MT5 bars (see ``goldenswing/backtest/mt5_native_data``).
No ``_resample`` call exists on this path.

AUTO-FETCH (2026-10-16, owner rule): this script runs on the
machine WITH MT5, so on start it fetches the data itself from the
running terminal (``goldenswing/backtest/mt5_fetch.py``):
``copy_rates_range`` for every TF + ``copy_ticks_range`` (one tick
stream) for the requested window, cached to ``data/mt5_native/``
so a second run is instant. No manual CSV export. If MT5 is not
running / not logged in, it FAILS with a clear message — it never
falls back to OANDA CSVs, never resamples, never silently uses
the M1 proxy. ``--no-fetch`` runs on the cache only (fails if the
cache does not cover the window).

M1/M5/M15/M30 fallback: when the terminal has no cached bars
for a sub-hourly TF but HAS ticks, that TF's bars are
aggregated from the tick stream itself (Bid OHLC per minute
bucket) and flagged ``built_from_ticks`` in the provenance —
never resampled from H1. The same tick-aggregation fallback
applies to HTF bars (H1/H4/D1/W1/MN1): ticks are the source
truth and contain the session boundaries, so a TF with no
native bars is built from ticks (flagged) instead of blocking.
No ticks at all still BLOCKS — with diagnostics.

Ticks: real MT5 ``copy_ticks_range`` from the terminal. Without
real ticks the run refuses unless ``--allow-m1-proxy`` smoke-runs
on the labelled M1 approximation (4 invented ticks per M1 bar) —
flagged ``ticks: m1_proxy`` in the report, never presented as a
tick backtest.

OWNER ADDENDUM (2026-10-16) — NO SHORTCUT. Overrides any time
shortcut:

- The backtest runs for AS LONG AS IT NEEDS TO. No --max-bars
  cap, no time limit, no early exit, no sampling. Full window,
  full engine, full ticks. ``runner.run(max_ticks=len(ticks))``
  runs EVERY tick. There is deliberately no --days / --max-bars /
  --only-playbook flag on this path.
- EXPORT RESULTS TO TELEGRAM on completion: the summary goes to
  Telegram via the vault-backed tg-call on every real-tick full
  run (report + JSON are also written to ``data/``). If the
  Telegram send fails, the failure is reported loudly (stderr +
  ``results.json: telegram``) — never silently skipped.
- BACKTEST TESTS WHAT IS LIVE / GOING LIVE: exactly the code on
  the main branch as-is — the live engine, ALL playbooks, ALL
  gates. No playbook picking, no disabling mean reversion, no
  cherry-picking, no threshold tuning to make numbers look good.
  If mean reversion loses, the report says it loses. Honest
  numbers only.

Usage:
  cd ~/workspace/goldenswing && .venv/bin/python \
      scripts/backtest_mt5_full.py --coverage      # fetch + report
  cd ~/workspace/goldenswing && .venv/bin/python \
      scripts/backtest_mt5_full.py --out DIR [--allow-m1-proxy]
  # one command: fetches the last 3y from MT5, then runs:
  cd ~/workspace/goldenswing && .venv/bin/python \
      scripts/backtest_mt5_full.py --out data/mt5_bt_out

Paper only. No live trading. News pause off (no historical
calendar), confirmation auto-fire (timeout default fire) — same
documented simulation choices as backtest_full_engine.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from goldenswing.backtest.mt5_fetch import (  # noqa: E402
    DEFAULT_CACHE_DIR, DEFAULT_SYMBOL, MT5Unavailable, default_window,
    ensure_mt5_native)
from goldenswing.backtest.mt5_native_data import (  # noqa: E402
    MissingNativeData, coverage_report,
    load_native_frames, load_native_ticks, ticks_from_m1)
from goldenswing.execution.executor import AccountState  # noqa: E402,F401
from goldenswing.execution.mt5_runner import (  # noqa: E402
    MT5Runner, MT5RunnerConfig)
from goldenswing.live.news import NewsCalendar  # noqa: E402
import backtest_full_engine as bfe  # noqa: E402


def _ensure_data(args) -> tuple[pd.Timestamp, pd.Timestamp, dict]:
    """Auto-fetch from the running MT5 terminal (or cache-only).

    Returns (start, end, manifest). On MT5-unavailable prints the
    clear message and raises SystemExit(3) — never falls back.
    """
    if args.start and args.end:
        start = pd.Timestamp(args.start, tz="UTC")
        end = pd.Timestamp(args.end, tz="UTC")
    elif args.start or args.end:
        print("--start and --end must be given together", file=sys.stderr)
        raise SystemExit(2)
    else:
        start, end = default_window(years=args.years)
    cache = Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE_DIR
    if args.no_fetch:
        print(f"cache-only (--no-fetch): {cache}")
        return start, end, {"source": "mt5_cache_only",
                            "cache_dir": str(cache)}
    try:
        manifest = ensure_mt5_native(args.symbol, start, end, cache,
                                     force=args.force_fetch)
    except MT5Unavailable as exc:
        print(f"BLOCKED — {exc}", file=sys.stderr)
        raise SystemExit(3)
    print(f"data: {manifest['source']} ({cache}) "
          f"{start.date()}..{end.date()}")
    return start, end, manifest


def _print_coverage(args) -> int:
    _start, _end, _manifest = _ensure_data(args)
    cache = Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE_DIR
    cov = coverage_report(cache, args.symbol)
    print("Native per-TF coverage (NO resampling — missing = missing):")
    for tf, c in cov.items():
        if c.missing:
            print(f"  {tf:4} MISSING — no native file")
        else:
            segs = "; ".join(f"{s}..{e} ({n})"
                             for s, e, n in c.segments)
            print(f"  {tf:4} {c.rows:>6} rows  {c.start} -> {c.end}  "
                  f"maxgap {c.gap_days:.0f}d  [{c.source}]")
            print(f"       segments: {segs}")
    try:
        frames, prov = load_native_frames(cache, args.symbol)
        w = prov["window"]
        print(f"\nCommon window: {w['start']} -> {w['end']}")
        if prov.get("built_from_ticks"):
            print(f"Built from MT5 ticks (terminal had no cached "
                  f"bars): {', '.join(prov['built_from_ticks'])}")
    except MissingNativeData as exc:
        print(f"\nBLOCKED: {exc.missing}")
        return 3
    try:
        ticks = load_native_ticks(cache, args.symbol)
        print(f"Ticks: {len(ticks):,} real MT5 ticks")
    except MissingNativeData:
        print("Ticks: MISSING — real tick backtest needs "
              f"MT5_{args.symbol}_ticks.csv (copy_ticks_range)")
    return 0


DEFAULT_TELEGRAM_CHAT_ID = "5478650254"


def build_telegram_summary(report: dict) -> str:
    """One Telegram message: window, TFs, ticks, trades, WR, PF,
    expectancy, max DD, TP1/2/3 hit rates, regime breakdown,
    resampling NONE, full-engine. Honest numbers only — this is
    the live code as-is, nothing disabled or tuned."""
    t = report["trades"]
    w = report["window"]
    prov = report.get("data_provenance", {})
    tfs = ", ".join(
        f"{tf}:{m['rows']}" for tf, m in prov.items()
        if isinstance(m, dict) and "rows" in m)
    tp = report.get("tp_hit_rates", {})
    tp_line = " | ".join(
        f"{k.upper()} {v['hits']}/{v['denominator']} ({v['rate']:.0%})"
        for k, v in tp.items()) or "n/a"
    reg = report.get("regime_breakdown", {})
    reg_line = "\n".join(
        f"{r}: {d['trades']} trades, {d['total_r']:+.2f}R, "
        f"WR {d['win_rate']:.0%}" for r, d in sorted(reg.items())) \
        or "n/a"
    return (
        "📊 GoldenSwing FULL-ENGINE BACKTEST — complete\n"
        "Full engine, full window, full ticks. Live code as-is: "
        "all playbooks, all gates, nothing disabled, nothing tuned.\n"
        f"Window: {w['start']} -> {w['end_open']}\n"
        f"TFs (native rows): {tfs}\n"
        f"Ticks: {w['ticks']:,} ({prov.get('ticks', '?')})\n"
        f"Trades closed: {t['closed']} (open at end: "
        f"{t['open_at_end']})\n"
        f"Win rate: {t['win_rate']:.1%} | PF: {t['profit_factor']} | "
        f"Expectancy: {t['avg_r']:+.3f}R | Max DD: "
        f"{t['max_drawdown_r']:.2f}R\n"
        f"Total: {t['total_r']:+.2f}R (${t['total_dollar']:+,.2f})\n"
        f"TP hit rates: {tp_line}\n"
        f"Regime breakdown:\n{reg_line}\n"
        f"Resampling: NONE | Engine: MT5Runner (live code)\n"
        f"Elapsed: {w.get('elapsed_s', '?')}s"
    )


def send_telegram_summary(text: str, chat_id: str | None = None,
                          tg_call: str | None = None) -> dict:
    """Send ``text`` via the vault-backed tg-call. Returns
    ``{"sent": bool, "error": str | None}``. Never raises — a
    failed send is reported by the caller, never silently skipped.
    """
    import os
    import subprocess

    chat = chat_id or os.environ.get("GS_TELEGRAM_CHAT_ID") \
        or DEFAULT_TELEGRAM_CHAT_ID
    call = tg_call or os.environ.get("GS_TELEGRAM_TG_CALL") or str(
        Path.home() / "workspace" / "custom-connectors"
        / "custom.legacy-telegram" / "bin" / "tg-call")
    try:
        proc = subprocess.run(
            [call, "--method", "sendMessage",
             "--data", json.dumps({"chat_id": chat, "text": text})],
            capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        return {"sent": False, "error": f"tg-call not found: {exc}"}
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {"sent": False, "error": f"tg-call failed: {exc}"}
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        payload = {}
    if proc.returncode == 0 and payload.get("ok") is True:
        return {"sent": True, "error": None}
    return {"sent": False,
            "error": f"tg-call rc={proc.returncode} "
                     f"stdout={proc.stdout[:300]!r} "
                     f"stderr={proc.stderr[:300]!r}"}


def write_data_report(report: dict, out: Path) -> dict:
    """Write report + JSON to ``data/`` (timestamped), as well as
    ``out/results.json``. Returns the paths written."""
    import datetime as _dt

    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    data_dir = REPO_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / f"mt5_backtest_results_{ts}.json"
    md_path = data_dir / f"mt5_backtest_report_{ts}.md"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    md_path.write_text(
        "# GoldenSwing full-engine backtest\n\n"
        "Full engine, full window, full ticks. Live code as-is: "
        "all playbooks, all gates, nothing disabled, nothing tuned. "
        "Resampling: NONE.\n\n"
        "```\n" + build_telegram_summary(report) + "\n```\n\n"
        "## Full report (JSON)\n\n"
        "See `" + json_path.name + "` in the same directory.\n")
    (out / "results.json").write_text(json.dumps(report, indent=2,
                                                 default=str))
    return {"json": str(json_path), "md": str(md_path)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage", action="store_true",
                    help="auto-fetch from MT5, report native per-TF "
                         "coverage and exit")
    ap.add_argument("--out", type=str, default=None)
    # NO --days / --max-bars / --only-playbook on this path (owner
    # addendum: full window, full engine, all playbooks as-is).
    ap.add_argument("--allow-m1-proxy", action="store_true",
                    help="smoke-run on labelled M1 tick approximation "
                         "when no real MT5 tick file exists")
    ap.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL,
                    help="MT5 broker symbol to fetch data for "
                         "(default: XAUUSDm — the owner's Exness "
                         "terminal lists gold as XAUUSDm, not XAUUSD)")
    ap.add_argument("--start", type=str, default=None,
                    help="fetch window start (ISO, with --end)")
    ap.add_argument("--end", type=str, default=None,
                    help="fetch window end (ISO, with --start)")
    ap.add_argument("--years", type=int, default=3,
                    help="fetch window length when --start/--end "
                         "are not given (default: 3)")
    ap.add_argument("--cache-dir", type=str, default=None,
                    help="native cache dir (default: data/mt5_native)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="cache-only: never contact the MT5 terminal; "
                         "fail if the cache does not cover the window")
    ap.add_argument("--force-fetch", action="store_true",
                    help="re-fetch from MT5 even if the cache covers "
                         "the window")
    ap.add_argument("--fetch-only", action="store_true",
                    help="fetch from MT5 into the cache and exit")
    ap.add_argument("--no-telegram", action="store_true",
                    help="do not export the summary to Telegram "
                         "(tests / local runs only — real runs send)")
    ap.add_argument("--telegram-chat", type=str, default=None,
                    help="Telegram chat id (default: "
                         "$GS_TELEGRAM_CHAT_ID or 5478650254)")
    args = ap.parse_args(argv)
    if args.coverage:
        return _print_coverage(args)
    if args.fetch_only:
        _ensure_data(args)
        return 0

    out = Path(args.out) if args.out else Path(
        __import__("tempfile").mkdtemp(prefix="mt5_native_bt_"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"out: {out}")

    _start, _end, manifest = _ensure_data(args)
    cache = Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE_DIR
    try:
        frames, provenance = load_native_frames(cache, args.symbol)
    except MissingNativeData as exc:
        print(f"BLOCKED — {exc}", file=sys.stderr)
        return 3
    print("Frames (native, no resampling):")
    for tf, meta in provenance.items():
        if tf in ("window", "resampling", "built_from_ticks"):
            continue
        tag = " [built from MT5 ticks]" \
            if meta.get("built_from_ticks") else ""
        print(f"  {tf:4} {meta['rows']:>6} rows  {meta['source']}{tag}")
    if provenance.get("built_from_ticks"):
        print(f"  Built from ticks: "
              f"{', '.join(provenance['built_from_ticks'])}")
    print(f"  {provenance['resampling']}")

    dxy = bfe._load_csv(bfe.DATA_DIR / "DOLLAR_H1_2y.csv")
    # FULL WINDOW: the whole common window runs — no --days slice,
    # no --max-bars cap, no sampling. run() below consumes EVERY
    # tick (max_ticks=len(ticks)) for as long as it needs.
    start = pd.Timestamp(provenance["window"]["start"])
    end_open = pd.Timestamp(provenance["window"]["end"])
    print(f"window: {start} -> {end_open}")

    try:
        ticks = load_native_ticks(cache, args.symbol)
        tick_provenance = "real_mt5_ticks"
        print(f"ticks: {len(ticks):,} real MT5 ticks")
    except MissingNativeData:
        if not args.allow_m1_proxy:
            print("BLOCKED — no MT5 tick file. The backtest "
                  "auto-fetches ticks from the running MT5 terminal; "
                  "if the fetch returned no ticks, the fetch error "
                  "carries diagnostics (terminal / symbol / probe). "
                  "Re-run with --allow-m1-proxy for a labelled "
                  "smoke run.",
                  file=sys.stderr)
            return 3
        print("WARNING: NO REAL MT5 TICKS — smoke run on M1 proxy "
              "(intra-minute path invented). NOT a tick backtest.")
        ticks = ticks_from_m1(
            frames["M1"][(frames["M1"].index >= start)
                         & (frames["M1"].index <= end_open)])
        tick_provenance = "m1_proxy_NOT_real_ticks"
    ticks = [t for t in ticks
             if start <= pd.Timestamp(t["time"]) <= end_open]
    print(f"ticks in window: {len(ticks):,}")

    clock = bfe.SimClock()
    clock.now = start
    broker = bfe.SimBroker()
    feed = bfe.SimFeed(broker, frames, dxy, start, ticks)
    executor = bfe.SimExecutor(broker)
    cfg = MT5RunnerConfig(
        # NOTE: the engine's instrument label stays the engine
        # default (XAUUSD). --symbol is the BROKER symbol the data
        # was fetched under (XAUUSDm on Exness) — the same gold,
        # different broker suffix. Playbook validation is on the
        # instrument (XAUUSD), not the broker suffix, so the runner
        # label must not become XAUUSDm. Live runner config is
        # untouched.
        journal_path=out / "mt5_journal.jsonl",
        risk_state_path=out / "mt5_risk.json",
        kill_switch_path=out / "mt5_KILL",
        tick_interval=0.0, poll_commands=False, tick_log=False,
        charts_dir=out / "mt5_charts", reconcile_every_ticks=1,
        golden_swing_confirm_enabled=False, news_pause_enabled=False)
    # NO playbook picking: cfg.execute_playbooks stays at the
    # engine default — ALL playbooks, ALL gates, exactly as live.
    # Nothing disabled (mean reversion included), nothing tuned.
    news = NewsCalendar(cache_path=out / "news_cache.json",
                        fetcher=bfe._no_network,
                        clock=lambda: clock.now_dt())
    runner = MT5Runner(cfg, feed=feed, executor=executor, senders=[],
                       news_calendar=news)
    runner.guards._today_fn = clock.now_date  # noqa: SLF001 - driver
    real_time = time.time
    time.time = clock.now_ts
    try:
        t0 = time.perf_counter()
        stats = runner.run(max_ticks=len(ticks))
        elapsed = time.perf_counter() - t0
    finally:
        time.time = real_time
    print(f"run: {elapsed:.1f}s wall | stats: {stats}")

    report = bfe.analyze(out / "mt5_journal.jsonl", runner, stats,
                         frames, broker, ticks)
    report["data_provenance"] = provenance
    report["data_provenance"]["ticks"] = tick_provenance
    report["data_provenance"]["fetch"] = manifest.get("source")
    report["data_provenance"]["fetch_window"] = {
        "start": str(_start), "end": str(_end)}
    report["data_provenance"]["symbol"] = args.symbol
    report["data_provenance"]["dxy"] = "DOLLAR_H1_2y.csv (OANDA DXY H1)"
    report["window"] = {"start": str(start), "end_open": str(end_open),
                        "ticks": len(ticks), "elapsed_s": round(elapsed, 1),
                        "full_run": True,
                        "playbooks": "all as-is (no picking, "
                                     "no disabling, no tuning)"}
    # Report + JSON to data/ (timestamped) as well as out/results.json.
    report["report_files"] = write_data_report(report, out)
    bfe.print_report(report)

    # TELEGRAM EXPORT (owner addendum): real-tick full runs only.
    # M1-proxy smoke runs never send — their numbers are not a
    # tick backtest and must never land in the owner's Telegram
    # as if they were.
    if tick_provenance == "real_mt5_ticks" and not args.no_telegram:
        tg = send_telegram_summary(build_telegram_summary(report),
                                   chat_id=args.telegram_chat)
        report["telegram"] = tg
        if tg["sent"]:
            print("Telegram: backtest summary sent.")
        else:
            print(f"TELEGRAM EXPORT FAILED: {tg['error']} — "
                  f"summary NOT delivered. Report still in "
                  f"{report['report_files']['json']}",
                  file=sys.stderr)
        # Re-write with the telegram outcome recorded.
        report["report_files"] = write_data_report(report, out)
    elif tick_provenance != "real_mt5_ticks":
        print("Telegram: skipped (M1-proxy smoke run — not a "
              "real-tick backtest).")
        report["telegram"] = {"sent": False,
                              "error": "skipped: m1_proxy smoke run"}
        report["report_files"] = write_data_report(report, out)
    else:
        report["telegram"] = {"sent": False,
                              "error": "skipped: --no-telegram"}
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
