"""Start the GoldenSwing MT5 engine (MT5 = source of truth).

Windows EC2: MT5 terminal installed, logged in (demo), active.
The engine attaches to the terminal as-is — no credentials needed
when the terminal is already logged in.  Pass MT5 credentials only
when the terminal is NOT logged in (env GS_MT5_LOGIN /
GS_MT5_PASSWORD / GS_MT5_SERVER; never on the command line).

DEMO-ONLY by default: the runner refuses a REAL MT5 account unless
--allow-live is passed explicitly.

Telegram: this engine OWNS getUpdates by default (command polling
ON) — /status /report /db /trades /harvest /ticks are answered from
the MT5 journal/state.  The OANDA loop's poller must be OFF at
cutover (GS_TELEGRAM_POLL=0) — one getUpdates consumer only.  While
the OANDA loop still polls (overlap window), start with
--no-poll-commands.

Every processed tick is logged to per-day CSVs
(data/mt5_ticks_YYYYMMDD.csv) exactly as seen; every signal and fill
gets a chart PNG (data/mt5_charts/) attached to its Telegram alert.

Examples:
  python scripts/run_mt5.py                 # demo, runs forever
  python scripts/run_mt5.py --max-ticks 1000  # rehearsal: 1000 ticks
  python scripts/run_mt5.py --fixed-lots 0.01  # $12-style demo sizing
  python scripts/run_mt5.py --no-poll-commands  # overlap: OANDA still polls
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goldenswing.execution.mt5_runner import MT5Runner, MT5RunnerConfig
from goldenswing.feed.mt5 import MT5Feed, MT5FeedConfig


def _resolve_symbol(args_symbol: str | None, environ) -> tuple[str, bool]:
    """CLI/env -> (symbol, explicit).

    Explicit = passed on the command line OR set via GS_SYMBOL.
    Only a non-explicit (default) symbol gets auto-detect fallback.
    """
    explicit = args_symbol is not None or "GS_SYMBOL" in environ
    symbol = args_symbol or environ.get("GS_SYMBOL") or "XAUUSD"
    return symbol, explicit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default=None,
                    help="broker gold symbol (default: XAUUSD with "
                         "auto-detect fallback; explicit --symbol / "
                         "GS_SYMBOL always wins, never auto-detected)")
    ap.add_argument("--max-ticks", type=int, default=None)
    ap.add_argument("--fixed-lots", type=float, default=None)
    ap.add_argument("--risk-pct", type=float, default=None)
    ap.add_argument("--tick-interval", type=float, default=0.25)
    ap.add_argument("--allow-live", action="store_true",
                    help="allow trading a REAL MT5 account (demo-only by "
                         "default — do not pass this without thinking)")
    ap.add_argument("--journal", type=str, default=None)
    ap.add_argument("--poll-commands", dest="poll_commands",
                    action="store_true", default=True,
                    help="own Telegram getUpdates (default ON)")
    ap.add_argument("--no-poll-commands", dest="poll_commands",
                    action="store_false",
                    help="do NOT poll Telegram commands — overlap window "
                         "while the OANDA loop still polls")
    ap.add_argument("--no-tick-log", dest="tick_log",
                    action="store_false", default=True,
                    help="do NOT log per-day tick CSVs")
    args = ap.parse_args(argv)

    login = os.environ.get("GS_MT5_LOGIN")
    # Explicit symbol = passed on the command line OR set via GS_SYMBOL.
    # Only a non-explicit (default) symbol gets auto-detect fallback.
    symbol, symbol_explicit = _resolve_symbol(args.symbol, os.environ)
    feed_cfg = MT5FeedConfig(
        symbol=symbol,
        symbol_explicit=symbol_explicit,
        login=int(login) if login else None,
        password=os.environ.get("GS_MT5_PASSWORD"),
        server=os.environ.get("GS_MT5_SERVER"),
        tick_interval=args.tick_interval)
    cfg = MT5RunnerConfig.from_env(
        symbol=symbol,
        fixed_lots=args.fixed_lots,
        allow_live=args.allow_live,
        tick_interval=args.tick_interval)
    if args.risk_pct is not None:
        cfg.risk_pct = args.risk_pct
    if args.journal:
        cfg.journal_path = Path(args.journal)
    cfg.poll_commands = args.poll_commands
    cfg.tick_log = args.tick_log

    senders = []
    if os.environ.get("GS_TELEGRAM_CHAT_ID"):
        from goldenswing.alerts.telegram import TelegramSender
        senders = [TelegramSender.from_env()]
    runner = MT5Runner(cfg, feed=MT5Feed(feed_cfg), senders=senders)
    # An explicit --fixed-lots wins over the persisted Telegram
    # config (data/mt5_config.json) the runner just loaded, and is
    # itself persisted for the next start.
    if args.fixed_lots is not None:
        runner.set_live_config(
            {"field": "fixed_lots", "value": args.fixed_lots})
    stats = runner.run(max_ticks=args.max_ticks)
    print(f"MT5 engine stopped: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
