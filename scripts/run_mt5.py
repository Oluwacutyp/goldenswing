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
  python scripts/run_mt5.py --risk-pct 0.01   # dynamic risk sizing rule
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


def _dotenv_path() -> Path:
    """Repo-root .env path (computed from __file__ at call time so
    tests can monkeypatch __file__)."""
    return Path(__file__).resolve().parent.parent / ".env"


def _load_dotenv() -> int:
    """Load GoldenSwing env vars from a repo-root .env (if present).

    Fresh PowerShell sessions lose $env:GS_* exports — without them
    the engine starts with NO Telegram sender and every ping +
    command reply is silently dropped.  A .env next to the repo
    restores them.  Never overrides vars already in the environment
    (explicit env wins); only GS_*/MT5_* keys are loaded, nothing
    else from the file is trusted.  Returns the number loaded.
    """
    env_path = _dotenv_path()
    if not env_path.is_file():
        return 0
    loaded = 0
    try:
        # utf-8-sig: Windows Notepad saves .env as UTF-8 WITH BOM by
        # default — a bare utf-8 read leaves "\ufeff" on the first
        # line's key, which then fails the GS_/MT5_ prefix check and
        # is silently skipped.  The BOM bug meant a .env whose FIRST
        # line held a Telegram key loaded nothing for that key, and
        # the engine reported "not configured" even though .env had
        # it.  utf-8-sig strips the BOM when present and behaves
        # like utf-8 otherwise.
        text = env_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, _, val = line.partition("=")
        key = key.strip()
        if not (key.startswith("GS_") or key.startswith("MT5_")):
            continue
        if key in os.environ:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ[key] = val
        loaded += 1
    if loaded:
        print(f"MT5 startup: loaded {loaded} var(s) from {env_path} "
              "(explicit env wins)", flush=True)
    return loaded


def _missing_telegram_vars(environ=None) -> list[str]:
    """Which Telegram vars are absent from the process environment."""
    env = os.environ if environ is None else environ
    return [v for v in ("GS_TELEGRAM_BOT_TOKEN", "GS_TELEGRAM_CHAT_ID")
            if not env.get(v)]


def _telegram_not_configured_text(*, dotenv_path: Path,
                                  dotenv_existed: bool,
                                  dotenv_loaded: int,
                                  missing: list[str]) -> str:
    """Accurate 'not configured' warning.

    The old message stated a flat persistent fact ("not configured")
    which contradicted the owner's evidence: notifications + commands
    worked yesterday.  The truth is session-scoped — $env: exports
    live only in the PowerShell session that set them — so the
    message must say WHERE it looked, WHAT is missing, and WHY
    yesterday worked while today does not.  Never used when a
    sender exists (callers check that first).
    """
    if dotenv_existed:
        dotenv_status = (
            f"found, loaded {dotenv_loaded} var(s), but the Telegram "
            "keys are still missing"
            if dotenv_loaded else
            "found, but it holds no loadable GS_TELEGRAM_* keys")
    else:
        dotenv_status = "not found"
    return (
        "MT5 startup: WARNING: Telegram sender NOT configured in "
        "THIS session — startup pings + command replies will NOT "
        "reach Telegram.\n"
        "MT5 startup: checked process environment + repo-root .env "
        f"at {dotenv_path} ({dotenv_status}).\n"
        "MT5 startup: missing: " + ", ".join(missing) + ".\n"
        "MT5 startup: why yesterday worked: $env:GS_TELEGRAM_* "
        "exports live only in the PowerShell session that set "
        "them — a fresh PowerShell starts without them. "
        "Yesterday's session had them exported; this session "
        "does not.\n"
        "MT5 startup: fix: put GS_TELEGRAM_BOT_TOKEN and "
        f"GS_TELEGRAM_CHAT_ID in {dotenv_path} and restart — "
        ".env persists across sessions.")


def _telegram_outbox_text() -> str:
    """Sender exists but cannot reach Telegram (outbox-only).

    Happens when GS_TELEGRAM_CHAT_ID is set but the bot token is
    missing: a sender IS built, so no 'not configured' warning
    fires, yet every send reports delivered=False.  Without this
    the engine looks configured while staying mute — the opposite
    false claim to the not-configured one.
    """
    return (
        "MT5 startup: WARNING: Telegram sender is outbox-only "
        "(GS_TELEGRAM_CHAT_ID is set but GS_TELEGRAM_BOT_TOKEN "
        "is missing) — pings will NOT reach Telegram. Set "
        "GS_TELEGRAM_BOT_TOKEN in the environment (or a "
        "repo-root .env) and restart.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default=None,
                    help="broker gold symbol (default: XAUUSD with "
                         "auto-detect fallback; explicit --symbol / "
                         "GS_SYMBOL always wins, never auto-detected)")
    ap.add_argument("--terminal-path", default=None,
                    help="pin initialize(path=...) to the running "
                         "terminal's terminal64.exe (default: "
                         "MT5_TERMINAL_PATH env, else bare "
                         "initialize() — which can attach to a "
                         "different MT5 install than the one on "
                         "screen)")
    ap.add_argument("--max-ticks", type=int, default=None)
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

    dotenv_path = _dotenv_path()
    dotenv_existed = dotenv_path.is_file()
    dotenv_loaded = _load_dotenv()
    login = os.environ.get("GS_MT5_LOGIN")
    # Explicit symbol = passed on the command line OR set via GS_SYMBOL.
    # Only a non-explicit (default) symbol gets auto-detect fallback.
    symbol, symbol_explicit = _resolve_symbol(args.symbol, os.environ)
    terminal_path = args.terminal_path \
        or os.environ.get("MT5_TERMINAL_PATH") or None
    feed_cfg = MT5FeedConfig(
        symbol=symbol,
        symbol_explicit=symbol_explicit,
        login=int(login) if login else None,
        password=os.environ.get("GS_MT5_PASSWORD"),
        server=os.environ.get("GS_MT5_SERVER"),
        terminal_path=terminal_path,
        tick_interval=args.tick_interval)
    cfg = MT5RunnerConfig.from_env(
        symbol=symbol,
        allow_live=args.allow_live,
        tick_interval=args.tick_interval)
    cfg.terminal_path = terminal_path
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
    if not senders:
        # LOUD, not journal-only — and ACCURATE: the message must
        # name where it looked, what is missing, and why yesterday
        # worked while today does not (session-scoped $env:
        # exports).  A flat "not configured" contradicts the
        # owner's evidence that notifications + commands worked
        # yesterday.  (The runner prints the same warning again
        # at startup — belt + suspenders, one line each.)
        print(_telegram_not_configured_text(
            dotenv_path=dotenv_path,
            dotenv_existed=dotenv_existed,
            dotenv_loaded=dotenv_loaded,
            missing=_missing_telegram_vars()), flush=True)
    elif all(getattr(s, "mode", "live") == "outbox" for s in senders):
        # Sender exists but cannot reach Telegram (no token) —
        # never claim "configured" while staying mute.
        print(_telegram_outbox_text(), flush=True)
    runner = MT5Runner(cfg, feed=MT5Feed(feed_cfg), senders=senders)
    stats = runner.run(max_ticks=args.max_ticks)
    print(f"MT5 engine stopped: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
