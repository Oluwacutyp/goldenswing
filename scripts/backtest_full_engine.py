"""Full-engine backtest: the WHOLE GoldenSwing MT5 engine as ONE system.

Runs the real ``MT5Runner`` (same class the EC2 engine runs) over
historical bars through a simulated broker/feed.  This is NOT a
per-playbook backtest stitched together, and it is NOT the
``mtf_hierarchy`` verdict audit (which evaluates the hierarchy verdict
per H1 bar without playbooks, gates, orders, or management).  Here the
live signal path runs end to end: playbook signals -> hierarchy /
confluence gate -> order selection -> tick management -> exits.

METHOD (what is real engine code, what is simulation)
-----------------------------------------------------
REAL engine code: MT5Runner, all 12 playbooks, hierarchy/live_setup,
confluence gate, RiskGuards, cost gate, margin + shield, entry gates,
MacroTrend sizing, tick manager (partials / profit-lock / Chandelier /
panic / time exit), pending lifecycle + score decay, DXY context,
reasoning + study packages, journal.  The runner cannot tell it is
not live: it reads ticks, bars, account state, and fills from the
simulated feed/executor through the same interfaces the live MT5
feed/executor implement.

SIMULATION (documented, not hidden):
- Bars: OANDA H1/M15/M5 CSVs (same files the hierarchy audit used);
  H4/D1/W1/MN1 resampled from H1, M30 from M15, ALL open-stamped
  (the MT5-native convention the live feed serves).  DXY H1 from
  data/DOLLAR_H1_2y.csv via the feed's bars_for.
- Ticks: 4 per M5 bar, at +0s (open), +120s (high), +240s (low),
  +299s (close).  Intrabar path is approximated in fixed OHLC order:
  a level inside the bar is crossed at its high/low tick; if both
  SL and TP sit inside one M5 bar, the TP-side tick comes first in
  tick order.  Trigger detection is therefore M5-close granularity
  at best (a level touched mid-M5-bar but not at its high/low tick
  is missed -- impossible by construction, since high/low ticks ARE
  the bar extremes).
- Fills: market orders fill at the current tick mid (zero slippage
  model); pending LIMIT/STOP fill at their level when the tick price
  crosses it.  Exits book at the FIRST TICK beyond the level (the
  runner records the close at the tick price, not the level) --
  the standard tick-sim convention, and the same rule for SL (via
  the tick manager) and TP (via the sim's broker-side TP + reconcile).
- Spread: constant 0.20 (the Exness demo minimum).  The cost gate
  therefore sees a fixed spread; real spread varies.
- Account: $10,000 starting balance, 1% risk/trade (engine default),
  leverage 500 for margin math.  No broker stop-out simulation --
  the runner's margin check + shield are the binding constraint.
- Wall clock: the engine's wall-clock logic (entry-gate cooldowns /
  dedup TTL, RiskGuards day/week windows) runs on SIM time:
  time.time is patched to the current tick time and the guards'
  today function is pointed at the sim date.  Confirm timeouts use
  time.monotonic (unpatchable) -- see the confirmation handling
  below.  Datetime.now (journal record labels) stays real time.
- Confirmation: Golden Swing confirmation is DISABLED in this run
  (golden_swing_confirm_enabled=False).  That is exactly the
  configured timeout default (timeout action "fire" = auto-fire on
  timeout) with the 300s delay collapsed to zero -- a missed tap
  fires the order, per config.  The delay itself is not simulated.
- News pause: DISABLED (news_pause_enabled=False).  The news module
  fetches ForexFactory "this week"; applying this week's calendar
  to Apr-Oct bars would be wrong, and no historical calendar source
  exists.  The news feature is unit-tested separately; its fetcher
  is wired to raise here so any accidental fetch fails loudly.
- Ladder: OFF (engine default, opt-in).  Scalp cap: OFF (engine
  default -- the playbook audit found no live scalp playbook).
- Golden Swing: UNTESTABLE on this window (see the report) -- W1
  needs 200 weekly bars (~3.8y) and MN1 200 monthly bars (~16.7y)
  for EMA200; the CSVs hold ~104 W1 / ~24 MN1 bars, so W1/MN1 bias
  is NEUTRAL throughout and alignment never reaches Golden Swing.
  Sizing therefore runs at 1.0x / 0.5x (conflict) only.
- Reconcile every tick (reconcile_every_ticks=1, not the live 20):
  exits are recorded at the first tick beyond the level instead of
  up to 25 sim-minutes late.  Live reconcile runs every 20 ticks
  (~5s real time); the backtest equivalent is every tick.

DATA WINDOW: production = intersection of the gold frames
(2026-04-01 -> 2026-10-12); warmup = everything before
(2024-10-01 -> 2026-04-01).  Same OANDA files as the audit.

ACCOUNTING: journal close records book the REMAINDER's per-unit R
(partials bank R that never appears in a close record), so the
report reconstructs per-ticket total R/$ fraction-weighted.
Max drawdown attributes each ticket's full R to its close time
(partials banked earlier in reality).
total_R = sum(partial_R * partial_vol/total_vol) + close_R *
(remaining_vol/total_vol), and likewise for $.  Open-at-end
positions are mark-to-market (not counted in totals).

Usage:
  cd ~/workspace/goldenswing && .venv/bin/python scripts/backtest_full_engine.py [--max-ticks N] [--out DIR] [--days N] [--only-playbook NAME] [--notify-telegram]
Paper only.  No live trading, no network (news fetcher raises) — except
``--notify-telegram``, which sends the result to Telegram on completion.

--days N: backtest only the most recent N days of the dataset
  (default: whole window).  Warmup bars before the window are still
  served, so indicators are not cold-started.  Telegram /backtest
  defaults to 30 days, capped at 90.
--only-playbook NAME: run with only that executing playbook enabled
  (one of sweep_reversal, fvg_midline, trendline_retest, fvg_limit,
  inversion, killzone_open, judas_swing).  Shadow playbooks
  (turtle_soup, v_reversal, retracement_trap, order_block, qm_m5,
  qm_d1) never order, so running one alone is refused.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from goldenswing.execution.executor import (  # noqa: E402
    AccountState, OrderResult, PendingOrder, Position)
from goldenswing.execution.mt5_runner import (  # noqa: E402
    MT5Runner, MT5RunnerConfig)
from goldenswing.execution.risk import gold_spec  # noqa: E402
from goldenswing.live.macrotrend import compute_macrotrend  # noqa: E402
from goldenswing.live.news import NewsCalendar  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
START_BALANCE = 10_000.0

#: Backtest feed files. These ship with the repo (force-added despite
#: ``data/*.csv`` in .gitignore) so an EC2 ``git pull`` has the feed.
#: Never run the backtest without them — see missing_backtest_data().
BACKTEST_CSVS = (
    "OANDA_XAU_USD_H1_20241001_20261012.csv",
    "OANDA_XAU_USD_M15_20241001_20261012.csv",
    "OANDA_XAU_USD_M5_20260401_20261012.csv",
    "DOLLAR_H1_2y.csv",
)

#: Distinct exit code for missing feed data (2 = arg validation).
BACKTEST_MISSING_DATA_EXIT = 3

BACKTEST_DATA_MISSING_MSG = (
    "Backtest data missing: {missing}. On EC2: git pull origin main "
    "(CSVs ship with the repo), or run /harvest. "
    "Do not run backtest without feed.")


def missing_backtest_data(data_dir: Path | None = None) -> list[str]:
    """Names of backtest CSVs absent from ``data_dir`` (default DATA_DIR)."""
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    return [name for name in BACKTEST_CSVS if not (base / name).is_file()]
LEVERAGE = 500.0
SPREAD = 0.20
CONTRACT = 100.0  # 1 lot = 100 oz


# ------------------------------------------------------- telegram notify
# Second-terminal runs (e.g. a 365-day backtest) finish long after the
# owner stopped watching the terminal.  ``--notify-telegram`` sends the
# Septorch-style summary + results.json via the Bot API when the run
# completes — success or failure.  Stdlib urllib only, no new deps.
# Credentials come from the same env the live runner uses; the token is
# never printed, never logged, and never embedded in an error string
# (Bot API URLs contain the token, so error paths report status +
# description only).  Notify failure never marks the backtest failed.
_TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
_TELEGRAM_MESSAGE_LIMIT = 4096
_TELEGRAM_DOCUMENT_LIMIT_BYTES = 50_000_000  # Bot API sendDocument limit
_NOTIFY_ENV_MISSING = ("Telegram notify skipped: GS_TELEGRAM_BOT_TOKEN / "
                       "GS_TELEGRAM_CHAT_ID not set")

#: Last run's notify context, for the main() failure path.
_LAST_NOTIFY: dict = {}


def _progress_bar(rate: float, width: int = 10) -> str:
    filled = max(0, min(width, round(float(rate) * width)))
    return "▓" * filled + "░" * (width - filled)


def format_notify_message(report: dict | None, out_dir, *, error=None,
                          error_tail: str = "") -> str:
    """Septorch-style Telegram summary for a finished backtest."""
    if error is not None or report is None:
        lines = ["❌ Backtest failed", "",
                 f"📁 out: `{out_dir}`"]
        if error is not None:
            lines.append(f"error: `{str(error)[:300]}`")
        if error_tail:
            lines += ["", "last lines:", error_tail[-1500:]]
        lines += ["",
                  "The run itself errored, not the engine. "
                  "Re-run with /backtest or the script."]
        return "\n".join(lines)
    window = report.get("window", {}) or {}
    trades = report.get("trades", {}) or {}
    per_pb = report.get("per_playbook", {}) or {}
    closed = int(trades.get("closed", 0))
    win_rate = float(trades.get("win_rate", 0.0))
    total_r = float(trades.get("total_r", 0.0))
    total_dollar = float(trades.get("total_dollar", 0.0))
    try:
        pf_f = float(trades.get("profit_factor", 0.0))
    except (TypeError, ValueError):
        pf_f = 0.0
    pf_txt = "∞" if pf_f == float("inf") else f"{pf_f:.2f}"
    pb_txt = window.get("only_playbook") or "all executing"
    lines = [
        "📊 Backtest done — whole engine",
        "",
        f"period: `{window.get('start', '?')}` → "
        f"`{window.get('end_open', '?')}` · "
        f"days `{window.get('days_requested', '?')}` · "
        f"playbooks `{pb_txt}` · "
        f"ticks `{int(window.get('ticks', 0)):,}`",
        "",
        "📈 Trades",
        f"closed: `{closed}` · "
        f"open at end: `{trades.get('open_at_end', 0)}` "
        f"(unrealized `{float(trades.get('open_at_end_unrealized_r', 0.0)):+.2f}R`, "
        f"not in totals)",
        f"win rate: {_progress_bar(win_rate)} {win_rate:.1%}",
        f"total: `{total_r:+.2f}R` · `${total_dollar:+,.2f}` · "
        f"avg `{float(trades.get('avg_r', 0.0)):+.3f}R` · "
        f"PF `{pf_txt}` · MaxDD `{float(trades.get('max_drawdown_r', 0.0)):.2f}R`",
        "",
        "🎯 Per-Playbook",
    ]
    if per_pb:
        for name in sorted(per_pb):
            d = per_pb[name] or {}
            lines.append(
                f"  🎯 `{name}`: {d.get('trades', 0)} trades, "
                f"`{float(d.get('total_r', 0.0)):+.2f}R` · "
                f"`${float(d.get('total_dollar', 0.0)):+,.2f}`")
    else:
        lines.append("  ➖ (none) — no closed trades")
    lines += [
        "",
        f"📁 out: `{out_dir}`",
        "➡️ Next: drop results.json back to Muse for review",
        "",
        "⚠️ Limits: sim ticks, zero slippage, spread 0.20 constant, "
        "resampled HTF, Golden Swing untestable, news pause off.",
    ]
    return "\n".join(lines)


def _telegram_post(token: str, method: str, *, json_payload: dict | None = None,
                   raw_body: bytes | None = None,
                   content_type: str | None = None,
                   opener=None, timeout: float = 30.0) -> tuple[bool, str]:
    """One Bot API POST.  Returns (ok, detail) — never includes the
    token (it lives in the URL) in any error string."""
    url = _TELEGRAM_API.format(token=token, method=method)
    if json_payload is not None:
        body = json.dumps(json_payload).encode("utf-8")
        ctype = "application/json"
    else:
        body = raw_body or b""
        ctype = content_type or "application/octet-stream"
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": ctype},
                                 method="POST")
    opener = opener or urllib.request.build_opener()
    try:
        with opener.open(req, timeout=timeout) as resp:
            parsed = json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            parsed = json.loads(exc.read().decode("utf-8") or "{}")
        except Exception:
            parsed = {}
        return False, (f"Telegram API {exc.code}: "
                       f"{str(parsed.get('description', ''))[:200]}")
    except Exception as exc:  # noqa: BLE001 — network down, timeout, DNS
        return False, f"{type(exc).__name__}: {exc}"
    if parsed.get("ok"):
        mid = (parsed.get("result") or {}).get("message_id")
        return True, f"ok, message_id={mid}"
    return False, (f"Telegram API: "
                   f"{str(parsed.get('description', ''))[:200]}")


def send_telegram_text(token: str, chat_id, text: str, *,
                        opener=None) -> tuple[bool, str]:
    """sendMessage, truncated to Telegram's 4096-char limit."""
    if len(text) > _TELEGRAM_MESSAGE_LIMIT:
        text = text[:_TELEGRAM_MESSAGE_LIMIT - 1] + "…"
    return _telegram_post(token, "sendMessage",
                          json_payload={"chat_id": chat_id, "text": text,
                                        "disable_notification": False},
                          opener=opener)


def send_telegram_document(token: str, chat_id, path, caption: str, *,
                           opener=None,
                           limit_bytes: int = _TELEGRAM_DOCUMENT_LIMIT_BYTES
                           ) -> tuple[bool, str]:
    """sendDocument via stdlib multipart.  Files over the Bot API limit
    are refused up front with an honest message, never a doomed upload."""
    p = Path(path)
    if not p.exists():
        return False, f"document not found: {p.name}"
    size = p.stat().st_size
    if size > limit_bytes:
        return False, (f"{p.name} is {size / 1e6:.1f}MB — over Telegram's "
                       f"50MB bot document limit")
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []

    def _field(name: str, value: str) -> None:
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"'
                     f"\r\n\r\n".encode())
        parts.append(f"{value}\r\n".encode())

    _field("chat_id", str(chat_id))
    _field("caption", caption[:1023] + "…" if len(caption) > 1024 else caption)
    _field("disable_notification", "true")
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="document"; '
        f'filename="{p.name}"\r\n'.encode())
    parts.append(b"Content-Type: application/json\r\n\r\n")
    parts.append(p.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return _telegram_post(
        token, "sendDocument", raw_body=b"".join(parts),
        content_type=f"multipart/form-data; boundary={boundary}",
        opener=opener)


def notify_telegram(report: dict | None, out_dir, *, error=None,
                    error_tail: str = "", opener=None) -> str:
    """Send the backtest result to Telegram.  Never raises; returns a
    one-line status.  Notify failure never marks the backtest failed."""
    try:
        token = os.environ.get("GS_TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("GS_TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return _NOTIFY_ENV_MISSING
        text = format_notify_message(report, out_dir, error=error,
                                     error_tail=error_tail)
        doc_note = ""
        results = Path(out_dir) / "results.json"
        if error is None and report is not None and results.exists():
            size = results.stat().st_size
            if size > _TELEGRAM_DOCUMENT_LIMIT_BYTES:
                doc_note = (
                    f"\n📁 results.json is {size / 1e6:.1f}MB — over "
                    f"Telegram's 50MB bot document limit, on EC2 at "
                    f"`{results}`.")
        ok, detail = send_telegram_text(token, chat_id, text + doc_note,
                                      opener=opener)
        if not ok:
            return f"Telegram notify failed (sendMessage): {detail}"
        if error is not None or report is None:
            return f"Telegram notify sent (failure report): {detail}"
        if not results.exists():
            return ("Telegram notify sent (summary only, no "
                    f"results.json): {detail}")
        if doc_note:
            return ("Telegram notify sent (summary only, results.json "
                    f"too large): {detail}")
        days = (report.get("window", {}) or {}).get("days_requested", "?")
        closed = (report.get("trades", {}) or {}).get("closed", 0)
        caption = (f"📊 Backtest results — {days} days — {closed} closed "
                   f"trades — drop back to Muse for review")
        ok2, detail2 = send_telegram_document(token, chat_id, results,
                                               caption, opener=opener)
        if not ok2:
            return f"Telegram notify sent (summary), document failed: {detail2}"
        return (f"Telegram notify sent (summary + results.json): {detail2}")
    except Exception as exc:  # noqa: BLE001 — notify must never crash
        return f"Telegram notify failed: {type(exc).__name__}: {exc}"



# ------------------------------------------------------------------ clock
class SimClock:
    """Mutable sim time; time.time is patched to it during the run."""

    def __init__(self) -> None:
        self.now = pd.Timestamp("2026-04-01", tz="UTC")

    def now_ts(self) -> float:
        return float(self.now.timestamp())

    def now_dt(self) -> datetime:
        return self.now.to_pydatetime()

    def now_date(self) -> date:
        return self.now.date()


# ------------------------------------------------------------------- data
def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = "ts" if "ts" in df.columns else "Datetime"
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]]


def _resample_open(df: pd.DataFrame, key: pd.DatetimeIndex) -> pd.DataFrame:
    tmp = df.copy()
    tmp["__key"] = key
    agg = tmp.groupby("__key").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"))
    agg.index = pd.DatetimeIndex(agg.index, tz="UTC")
    agg = agg.sort_index()
    agg["real_volume"] = 0.0
    agg["spread"] = SPREAD
    return agg


def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    idx = df.index
    if tf == "H4":
        key = idx.floor("4h")
    elif tf == "D1":
        key = idx.floor("D")
    elif tf == "M30":
        key = idx.floor("30min")
    elif tf == "W1":
        key = idx.normalize() - pd.to_timedelta(idx.weekday, unit="D")
    elif tf == "MN1":
        key = pd.DatetimeIndex(
            [t.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
             for t in idx], tz="UTC")
    else:
        raise ValueError(f"no resample rule for {tf}")
    return _resample_open(df, key)


def _with_feed_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["real_volume"] = 0.0
    out["spread"] = SPREAD
    return out


# ------------------------------------------------------------------ broker
class SimBroker:
    """Simulated broker: positions, resting pendings, margin, equity.

    TP closes are broker-side (the tick manager has no TP action);
    SL closes are left to the tick manager's stop_hit.  Pending fills
    and TP closes settle lazily on every positions()/pending_orders()
    / place_order() / account_state() call, against the last tick
    price -- the runner only ever observes the broker through those
    calls, so settlement timing matches observation timing.
    """

    def __init__(self, *, balance: float = START_BALANCE,
                 leverage: float = LEVERAGE) -> None:
        self.balance = float(balance)
        self.leverage = float(leverage)
        self.positions: dict[int, Position] = {}
        self.pendings: dict[int, PendingOrder] = {}
        self.last_price: float = 0.0
        self.last_time: pd.Timestamp | None = None
        self._ticket = 1000
        self.tp_closes: list[dict] = []

    def _next_ticket(self) -> int:
        self._ticket += 1
        return self._ticket

    # ------------------------------------------------------------ settle
    def settle(self) -> None:
        if self.last_price <= 0:
            return
        price = self.last_price
        # Pending fills: price crossed the trigger level.
        for ticket, po in list(self.pendings.items()):
            if po.kind == "limit":
                hit = price <= po.price if po.direction > 0 \
                    else price >= po.price
            else:  # stop
                hit = price >= po.price if po.direction > 0 \
                    else price <= po.price
            if not hit:
                continue
            del self.pendings[ticket]
            self.positions[ticket] = Position(
                ticket=ticket, symbol=po.symbol, direction=po.direction,
                volume=po.volume, entry_price=float(po.price),
                sl=po.sl, tp=po.tp, magic=po.magic,
                alert_id=po.alert_id, open_time=self.last_time,
                risk_money=0.0)
        # Broker-side TP closes (SL is the tick manager's job).
        for ticket, pos in list(self.positions.items()):
            if pos.tp is None:
                continue
            hit = price >= pos.tp if pos.direction > 0 \
                else price <= pos.tp
            if not hit:
                continue
            del self.positions[ticket]
            pnl = pos.direction * (pos.tp - pos.entry_price) \
                * pos.volume * CONTRACT
            self.balance += pnl
            self.tp_closes.append({"ticket": ticket, "tp": pos.tp,
                                    "price": price})

    # ------------------------------------------------------------ account
    def account(self) -> dict:
        self.settle()
        floating = 0.0
        used = 0.0
        for pos in self.positions.values():
            floating += pos.direction * (self.last_price - pos.entry_price) \
                * pos.volume * CONTRACT
            used += pos.volume * CONTRACT * pos.entry_price / self.leverage
        equity = self.balance + floating
        return {"trade_mode": 0, "balance": self.balance,
                "equity": equity, "margin": used,
                "free_margin": equity - used, "login": 0,
                "currency": "USD"}

    def margin_for(self, volume: float, price: float) -> float:
        return volume * CONTRACT * price / self.leverage

# ------------------------------------------------------------------- feed
class SimFeed:
    """MT5Feed-shaped simulated feed over precomputed frames."""

    def __init__(self, broker: SimBroker, frames: dict[str, pd.DataFrame],
                 dxy: pd.DataFrame, window_start: pd.Timestamp,
                 ticks: list[dict]) -> None:
        self.broker = broker
        self.frames = frames
        self.dxy = dxy
        self.window_start = window_start
        self._ticks = ticks
        self._tick_idx = 0
        self.cfg = SimpleNamespace(symbol="XAUUSD", symbol_explicit=True)
        self._sim_now = window_start

    # ------------------------------------------------------------ iface
    def connect(self):
        return True, "sim connected"

    def account_state(self) -> dict:
        return self.broker.account()

    def warm_start_all(self) -> dict[str, pd.DataFrame]:
        out = {}
        for tf, frame in self.frames.items():
            out[tf] = frame[frame.index < self.window_start].copy()
        return out

    def fetch_new_all(self, cursors) -> dict[str, pd.DataFrame]:
        # Vectorized closed-bar filter (mirror of MT5Feed._closed_only).
        # A per-timestamp Python loop here cost ~0.7s/call (12M calls
        # over 200 ticks) and would make the full run take ~40h.
        out = {}
        now = self._sim_now
        for tf, frame in self.frames.items():
            cur = (cursors or {}).get(tf)
            sub = frame if cur is None else frame[frame.index > cur]
            if len(sub) == 0:
                out[tf] = sub
                continue
            if tf == "W1":
                monday = now.normalize() - pd.Timedelta(days=now.weekday())
                out[tf] = sub[sub.index < monday].copy()
            elif tf == "MN1":
                month_start = now.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0,
                    nanosecond=0)
                out[tf] = sub[sub.index < month_start].copy()
            else:
                tf_min = {"M1": 1, "M5": 5, "M15": 15, "M30": 30,
                          "H1": 60, "H4": 240, "D1": 1440}[tf]
                cutoff = now - pd.Timedelta(minutes=tf_min)
                out[tf] = sub[sub.index <= cutoff].copy()
        return out

    def bars_for(self, symbol: str, tf: str, count: int) -> pd.DataFrame:
        if symbol in ("DXYm", "DXY"):
            frame = self.dxy
        elif symbol in ("XAUUSD",):
            frame = self.frames.get("H1")
        else:
            raise RuntimeError(
                f"symbol_select({symbol}) failed — not carried")
        if frame is None:
            raise RuntimeError(f"no frame for {symbol}")
        sub = frame[frame.index <= self._sim_now]
        return sub.tail(int(count)).copy()

    def tick(self) -> dict | None:
        if self._tick_idx >= len(self._ticks):
            return None
        tick = self._ticks[self._tick_idx]
        self._tick_idx += 1
        self._sim_now = pd.Timestamp(tick["time"])
        self.broker.last_price = float(tick["mid"])
        self.broker.last_time = self._sim_now
        return tick

    def order_margin(self, *, volume: float, price: float,
                     direction: int) -> float | None:
        return self.broker.margin_for(float(volume), float(price))

    def poll_status(self) -> dict:
        return {"ok": True, "reason": None}

    def market_open(self) -> tuple[bool, str]:
        """COMEX gold hours (same rule as the paper feed's
        market_closed_now): Sun 22:00 UTC - Fri 21:00 UTC, daily
        break 21:00-22:00 UTC."""
        now = self._sim_now
        wd, hr = now.weekday(), now.hour
        if wd == 5:
            return False, "Saturday"
        if wd == 6 and hr < 22:
            return False, "Sunday before open"
        if wd == 4 and hr >= 21:
            return False, "Friday after close"
        if hr == 21:
            return False, "daily break"
        return True, ""


# --------------------------------------------------------------- executor
class SimExecutor:
    """Executor-shaped simulated broker (same contract as Executor)."""

    def __init__(self, broker: SimBroker) -> None:
        self.broker = broker
        self.spec = gold_spec()

    def connect(self, **kw):
        return True, "sim connected"

    def account_state(self) -> AccountState:
        acct = self.broker.account()
        return AccountState(balance=acct["balance"], equity=acct["equity"],
                            free_margin=acct["free_margin"])

    def symbol_spec(self, symbol: str):
        return self.spec

    def positions(self, symbol=None):
        self.broker.settle()
        return [p for p in self.broker.positions.values()
                if symbol is None or p.symbol == symbol]

    def pending_orders(self, symbol=None):
        self.broker.settle()
        return [p for p in self.broker.pendings.values()
                if symbol is None or p.symbol == symbol]

    def place_order(self, request) -> OrderResult:
        self.broker.settle()
        ticket = self.broker._next_ticket()
        if request.order_kind == "market":
            fill = self.broker.last_price
            self.broker.positions[ticket] = Position(
                ticket=ticket, symbol=request.symbol,
                direction=int(request.direction),
                volume=float(request.volume), entry_price=float(fill),
                sl=float(request.stop_loss),
                tp=float(request.take_profit)
                if request.take_profit else None,
                magic=int(request.magic), alert_id=request.alert_id,
                open_time=self.broker.last_time, risk_money=0.0)
            return OrderResult(True, ticket=ticket, price=float(fill),
                               volume=float(request.volume))
        # Pending LIMIT / STOP: rest at the order price.
        price = float(request.order_price) or float(request.entry_ref)
        self.broker.pendings[ticket] = PendingOrder(
            ticket=ticket, symbol=request.symbol,
            kind=str(request.order_kind),
            direction=int(request.direction), price=price,
            volume=float(request.volume),
            sl=float(request.stop_loss),
            tp=float(request.take_profit)
            if request.take_profit else None,
            magic=int(request.magic), alert_id=request.alert_id)
        return OrderResult(True, ticket=ticket, price=price,
                           volume=float(request.volume))

    def modify_sl(self, ticket: int, sl: float,
                  tp: float | None = None) -> OrderResult:
        pos = self.broker.positions.get(ticket)
        if pos is None:
            return OrderResult(False, ticket=ticket,
                               message="unknown ticket")
        pos.sl = float(sl)
        if tp is not None:
            pos.tp = float(tp)
        return OrderResult(True, ticket=ticket)

    def close_position(self, ticket: int,
                       volume: float | None = None) -> OrderResult:
        pos = self.broker.positions.get(ticket)
        if pos is None:
            # Idempotent: already closed (e.g. TP settled) -- the
            # runner's stop_hit path checks positions() first, so
            # this is a race guard, not a fabricated success.
            return OrderResult(True, ticket=ticket,
                               message="already closed")
        vol = float(volume) if volume is not None else pos.volume
        vol = min(vol, pos.volume)
        pnl = pos.direction * (self.broker.last_price - pos.entry_price) \
            * vol * CONTRACT
        self.broker.balance += pnl
        pos.volume -= vol
        if pos.volume <= 1e-9:
            del self.broker.positions[ticket]
        return OrderResult(True, ticket=ticket)

    def close_all(self, symbol=None):
        out = []
        for ticket in list(self.broker.positions):
            out.append(self.close_position(ticket))
        return out

    def cancel_pending(self, ticket: int) -> OrderResult:
        if ticket in self.broker.pendings:
            del self.broker.pendings[ticket]
            return OrderResult(True, ticket=ticket)
        return OrderResult(False, ticket=ticket,
                           message="already filled or unknown")

# --------------------------------------------------------------------- run
def _build_ticks(m5: pd.DataFrame, start: pd.Timestamp,
                 end_open: pd.Timestamp) -> list[dict]:
    ticks: list[dict] = []
    sub = m5[(m5.index >= start) & (m5.index <= end_open)]
    for ts, row in sub.iterrows():
        for off_s, price in ((0, row["open"]), (120, row["high"]),
                             (240, row["low"]), (299, row["close"])):
            t = ts + pd.Timedelta(seconds=int(off_s))
            mid = float(price)
            ticks.append({"bid": mid - SPREAD / 2, "ask": mid + SPREAD / 2,
                          "mid": mid, "spread": SPREAD, "time": t,
                          "tick_volume": 1.0, "real_volume": 0.0})
    # Flush tick: closes the last M5 bars + lets H1 closes fire.
    last_close = float(sub["close"].iloc[-1])
    flush_t = end_open + pd.Timedelta(minutes=5)
    ticks.append({"bid": last_close - SPREAD / 2,
                  "ask": last_close + SPREAD / 2, "mid": last_close,
                  "spread": SPREAD, "time": flush_t,
                  "tick_volume": 1.0, "real_volume": 0.0})
    return ticks


def _no_network(url, timeout):  # noqa: D103 - proves no news fetch
    raise AssertionError(
        "backtest must not touch the network (news fetch attempted)")


def _main_impl(argv: list[str] | None = None) -> int:
    _LAST_NOTIFY.clear()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-ticks", type=int, default=None)
    ap.add_argument("--out", type=str, default=None,
                    help="output dir (default: fresh temp dir)")
    ap.add_argument("--days", type=int, default=None,
                    help="backtest only the most recent N days "
                         "(warmup bars still served)")
    ap.add_argument("--only-playbook", type=str, default=None,
                    help="run with only this executing playbook enabled")
    ap.add_argument("--notify-telegram", action="store_true",
                    help="send the Septorch-style summary + results.json "
                         "to Telegram on completion (success or failure) "
                         "via GS_TELEGRAM_BOT_TOKEN / GS_TELEGRAM_CHAT_ID. "
                         "For second-terminal runs; the /backtest command "
                         "already delivers via ctx.deliver, so it does not "
                         "pass this flag.")
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else Path(
        __import__("tempfile").mkdtemp(prefix="full_engine_bt_"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"out: {out}")
    _LAST_NOTIFY.update({"notify": bool(args.notify_telegram), "out": out})

    # ------------------------------------------------------------ feed check
    # Fail gracefully with a clear message, never a pandas traceback
    # as the first signal (EC2 pulls once lacked these: data/*.csv
    # is gitignored, the CSVs ship force-added).
    missing = missing_backtest_data()
    if missing:
        msg = BACKTEST_DATA_MISSING_MSG.format(missing=", ".join(missing))
        print(msg, file=sys.stderr)
        if args.notify_telegram:
            print(notify_telegram(None, out, error=RuntimeError(msg)))
        return BACKTEST_MISSING_DATA_EXIT

    # ------------------------------------------------------------ data
    h1 = _with_feed_cols(_load_csv(DATA_DIR / "OANDA_XAU_USD_H1_20241001_20261012.csv"))
    m15 = _with_feed_cols(_load_csv(DATA_DIR / "OANDA_XAU_USD_M15_20241001_20261012.csv"))
    m5 = _with_feed_cols(_load_csv(DATA_DIR / "OANDA_XAU_USD_M5_20260401_20261012.csv"))
    dxy = _load_csv(DATA_DIR / "DOLLAR_H1_2y.csv")
    frames: dict[str, pd.DataFrame] = {
        "H1": h1,
        "H4": _resample(h1, "H4"),
        "D1": _resample(h1, "D1"),
        "W1": _resample(h1, "W1"),
        "MN1": _resample(h1, "MN1"),
        "M15": m15,
        "M30": _resample(m15, "M30"),
        "M5": m5,
    }
    start = max(f.index[0] for f in frames.values())
    end_open = min(f.index[-1] for f in frames.values())
    if args.days is not None:
        if args.days <= 0:
            print("--days must be positive", file=sys.stderr)
            return 2
        # Recent window: warmup bars before `start` are still served
        # by warm_start_all, so indicators are not cold-started.
        start = max(start, end_open - pd.Timedelta(days=args.days))
    only_playbook = args.only_playbook
    if only_playbook is not None:
        _executing = ("sweep_reversal", "fvg_midline", "trendline_retest",
                      "fvg_limit", "inversion", "killzone_open",
                      "judas_swing")
        if only_playbook not in _executing:
            print(f"--only-playbook must be one of {', '.join(_executing)}; "
                  f"got {only_playbook!r} (shadow playbooks never order)",
                  file=sys.stderr)
            return 2
    print(f"window: {start} -> {end_open} "
          f"(production H1 bars: "
          f"{int(((h1.index >= start) & (h1.index <= end_open)).sum())})")
    print(f"W1 bars in dataset: {len(frames['W1'])} "
          f"(EMA200 needs 200) | MN1 bars: {len(frames['MN1'])}")
    print(f"DXY H1: {dxy.index[0]} -> {dxy.index[-1]} ({len(dxy)} bars)")

    ticks = _build_ticks(m5, start, end_open)
    if args.max_ticks:
        ticks = ticks[: args.max_ticks]
    print(f"ticks: {len(ticks):,}")

    # ------------------------------------------------------------ sim
    clock = SimClock()
    clock.now = start
    broker = SimBroker()
    feed = SimFeed(broker, frames, dxy, start, ticks)
    executor = SimExecutor(broker)

    cfg = MT5RunnerConfig(
        journal_path=out / "mt5_journal.jsonl",
        risk_state_path=out / "mt5_risk.json",
        kill_switch_path=out / "mt5_KILL",
        tick_interval=0.0,
        poll_commands=False,
        tick_log=False,
        charts_dir=out / "mt5_charts",
        reconcile_every_ticks=1,
        golden_swing_confirm_enabled=False,  # auto-fire (see docstring)
        news_pause_enabled=False,  # no historical calendar (see docstring)
    )
    if only_playbook is not None:
        cfg.execute_playbooks = (only_playbook,)
    news = NewsCalendar(cache_path=out / "news_cache.json",
                        fetcher=_no_network,
                        clock=lambda: clock.now_dt())
    runner = MT5Runner(cfg, feed=feed, executor=executor, senders=[],
                       news_calendar=news)
    # RiskGuards day/week windows follow SIM time, not wall time.
    runner.guards._today_fn = clock.now_date  # noqa: SLF001 - driver only

    real_time = time.time
    time.time = clock.now_ts
    try:
        t0 = __import__("time").perf_counter()
        stats = runner.run(max_ticks=len(ticks))
        elapsed = __import__("time").perf_counter() - t0
    finally:
        time.time = real_time
    print(f"run: {elapsed:.1f}s wall | stats: {stats}")

    # -------------------------------------------------------- analysis
    report = analyze(out / "mt5_journal.jsonl", runner, stats,
                     frames, broker, ticks)
    report["window"] = {"start": str(start), "end_open": str(end_open),
                        "ticks": len(ticks), "elapsed_s": round(elapsed, 1),
                        "days_requested": args.days,
                        "only_playbook": only_playbook}
    report["config"] = {
        "balance": START_BALANCE, "risk_pct": cfg.risk_pct,
        "leverage": LEVERAGE, "spread": SPREAD,
        "confirm": "disabled=auto-fire (timeout default fire)",
        "news_pause": "disabled (no historical calendar)",
        "reconcile_every_ticks": 1,
        "ladder": False, "scalp_cap": "off (no scalp playbook)",
    }
    (out / "results.json").write_text(json.dumps(report, indent=2,
                                                 default=str))
    print_report(report)
    if args.notify_telegram:
        print(notify_telegram(report, out))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point: run the backtest, Telegram-notifying on failure too.

    Arg-validation failures (return 2) never notify — no run happened.
    Any other exception notifies with the error + traceback tail, then
    re-raises so the exit code still reflects the failure.
    """
    try:
        return _main_impl(argv)
    except Exception as exc:  # noqa: BLE001 — notify, then re-raise
        ctx = _LAST_NOTIFY
        if ctx.get("notify"):
            tail = "\n".join(traceback.format_exc().splitlines()[-15:])
            print(notify_telegram(None, ctx.get("out", "?"), error=exc,
                                  error_tail=tail))
        raise


# ---------------------------------------------------------------- analysis
_PARTIAL_RE = re.compile(
    r"mt5 tick: ticket (\d+) partial close ([\d.]+) lots at ([\d.]+)")
_SL_RE = re.compile(r"mt5 tick: ticket (\d+) SL -> ([\d.]+) \((\w+)\)")
_PEND_CANCEL_RE = re.compile(
    r"mt5 pending cancelled: ticket (\d+) (\w+) @ ([\d.]+) \((.*)\)")
_CLOSE_NOTE_RE = re.compile(
    r"mt5 close: ticket (\d+) (.+?) at ([\d.]+) \(([+-][\d.]+)R\)")


def analyze(journal_path: Path, runner, stats: dict,
            frames: dict[str, pd.DataFrame], broker: SimBroker,
            ticks: list[dict]) -> dict:
    records = [json.loads(line) for line in
               journal_path.read_text().splitlines() if line.strip()]
    opens = [r for r in records if r.get("type") == "open"]
    closes = [r for r in records if r.get("type") == "close"]
    notes = [r for r in records if r.get("type") == "note"]
    signals = [r for r in records if r.get("type") == "signal"]

    open_by_ticket: dict[int, list[dict]] = defaultdict(list)
    for o in opens:
        if o.get("ticket") is not None:
            open_by_ticket[int(o["ticket"])].append(o)
    close_by_ticket: dict[int, dict] = {}
    for c in closes:
        if c.get("ticket") is not None:
            close_by_ticket[int(c["ticket"])] = c  # last wins; 1 expected

    # Partials per ticket: (price, volume).
    partials: dict[int, list[tuple[float, float]]] = defaultdict(list)
    sl_moves: Counter = Counter()
    for n in notes:
        text = n.get("text", "")
        m = _PARTIAL_RE.search(text)
        if m:
            partials[int(m.group(1))].append(
                (float(m.group(3)), float(m.group(2))))
        m = _SL_RE.search(text)
        if m:
            sl_moves[m.group(3)] += 1

    # Per-ticket fraction-weighted R and $ (see module docstring).
    tickets = []
    for ticket, o_list in open_by_ticket.items():
        base = o_list[0]
        total_vol = sum(float(o["volume"]) for o in o_list)
        risk = abs(float(base["entry"]) - float(base["stop"]))
        direction = int(base["direction"])
        entry = float(base["entry"])
        ticket_r = 0.0
        ticket_dollar = 0.0
        remaining = total_vol
        for price, vol in partials.get(ticket, []):
            pr = direction * (price - entry) / risk if risk > 0 else 0.0
            ticket_r += pr * (vol / total_vol)
            ticket_dollar += direction * (price - entry) * vol * CONTRACT
            remaining -= vol
        close = close_by_ticket.get(ticket)
        closed_r = None
        if close is not None:
            closed_r = float(close["r_net"])
            ticket_r += closed_r * (max(remaining, 0.0) / total_vol)
            ticket_dollar += float(close["pnl_money"])
        tickets.append({
            "ticket": ticket, "playbook": base["playbook"],
            "direction": direction, "entry": entry,
            "volume": total_vol, "total_r": ticket_r,
            "total_dollar": ticket_dollar,
            "closed": close is not None,
            "close_reason": close["reason"] if close else None,
            "layer_opens": len(o_list) - 1,
        })

    closed_t = [t for t in tickets if t["closed"]]
    open_end = [t for t in tickets if not t["closed"]]
    last_price = float(ticks[-1]["mid"])
    # Unrealized R for open-at-end (mark-to-market, not in totals).
    for t in open_end:
        o = open_by_ticket[t["ticket"]][0]
        risk = abs(float(o["entry"]) - float(o["stop"]))
        t["unrealized_r"] = t["direction"] * (last_price - t["entry"]) \
            / risk if risk > 0 else 0.0

    rs = [t["total_r"] for t in closed_t]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    # Max drawdown on the cumulative-R curve (CLOSE order).  Each
    # ticket's full R is attributed to its close; partial R was
    # banked earlier in reality, so this is a close-time DD.
    close_order = [int(c["ticket"]) for c in closes
                   if c.get("ticket") is not None]
    by_ticket = {t["ticket"]: t for t in closed_t}
    curve, peak, max_dd = 0.0, 0.0, 0.0
    for ticket in close_order:
        t = by_ticket.get(ticket)
        if t is None:
            continue
        curve += t["total_r"]
        peak = max(peak, curve)
        max_dd = max(max_dd, peak - curve)

    per_pb: dict[str, dict] = defaultdict(
        lambda: {"trades": 0, "total_r": 0.0, "total_dollar": 0.0})
    for t in closed_t:
        d = per_pb[t["playbook"]]
        d["trades"] += 1
        d["total_r"] += t["total_r"]
        d["total_dollar"] += t["total_dollar"]

    # Close reasons + displays.
    close_reasons = Counter(t["close_reason"] for t in closed_t)
    displays = Counter()
    for n in notes:
        m = _CLOSE_NOTE_RE.search(n.get("text", ""))
        if m:
            displays[m.group(2).strip()] += 1

    # Pending cancels by reason.
    pend_cancel = Counter()
    pend_decay = 0
    for n in notes:
        m = _PEND_CANCEL_RE.search(n.get("text", ""))
        if not m:
            continue
        reason = m.group(4)
        if "engine_stop" in reason:
            pend_cancel["engine_stop_cleanup"] += 1
        elif "decay" in reason.lower():
            pend_cancel["score_decay"] += 1
            pend_decay += 1
        elif "expired" in reason.lower():
            pend_cancel["expired"] += 1
        elif "invalidated" in reason.lower():
            pend_cancel["invalidated"] += 1
        else:
            pend_cancel["other"] += 1

    # Signal records.
    sig_decisions = Counter(
        (s.get("playbook"), s.get("decision")) for s in signals)
    sig_reasons = Counter(s.get("reason", "") for s in signals
                          if s.get("decision") == "gated")

    # TP1/TP2/TP3 hit rates (owner addendum 2026-10-16): the journal
    # carries no explicit "TP" close reason — TPs are banked as
    # ordered partial closes (reason="partial"). The 1st partial on
    # a ticket IS its TP1, the 2nd its TP2, the 3rd its TP3 (the
    # engine banks TPs in order). Hit rate = tickets with >=k
    # partials / closed tickets. Open-at-end tickets are excluded
    # from the denominator (their TPs may still be pending) — the
    # denominator is stated, not hidden.
    tp_hit_rates: dict[str, dict] = {}
    for k in (1, 2, 3):
        hits = sum(1 for t in closed_t
                   if len(partials.get(t["ticket"], [])) >= k)
        tp_hit_rates[f"tp{k}"] = {
            "hits": hits, "denominator": len(closed_t),
            "rate": round(hits / len(closed_t), 4) if closed_t else 0.0}

    # Regime breakdown (owner addendum): regime is NOT journalled
    # per trade, so it is derived from the playbook family —
    # range_* -> RANGE (compression breakout is the boundary
    # moment -> BREAKOUT), narrative -> NARRATIVE (its regime is
    # not journalled per trade, so it gets its own bucket rather
    # than being mislabelled TREND), everything else -> TREND.
    # The per-playbook table below is the precise breakdown; this
    # is the regime roll-up of it.
    def _regime_of(playbook: str) -> str:
        if "narrative" in playbook:
            return "NARRATIVE"
        if playbook == "range_compression_breakout":
            return "BREAKOUT"
        if playbook.startswith("range_"):
            return "RANGE"
        return "TREND"

    regime_breakdown: dict[str, dict] = {}
    for t in closed_t:
        reg = _regime_of(t["playbook"])
        d = regime_breakdown.setdefault(
            reg, {"trades": 0, "total_r": 0.0, "wins": 0})
        d["trades"] += 1
        d["total_r"] += t["total_r"]
        d["wins"] += 1 if t["total_r"] > 0 else 0
    for d in regime_breakdown.values():
        d["total_r"] = round(d["total_r"], 2)
        d["win_rate"] = round(d["wins"] / d["trades"], 4) \
            if d["trades"] else 0.0

    # Golden Swing: impossible on this window (W1/MN1 < 200 bars)?
    w1_bars = len(frames["W1"])
    mn1_bars = len(frames["MN1"])
    golden_notes = sum(
        1 for n in notes if "golden" in n.get("text", "").lower())

    return {
        "run_stats": dict(stats),
        "blocked_stats": runner.blocked_stats(),
        "trades": {
            "closed": len(closed_t),
            "open_at_end": len(open_end),
            "open_at_end_unrealized_r": round(
                sum(t.get("unrealized_r", 0.0) for t in open_end), 2),
            "win_rate": round(len(wins) / len(rs), 4) if rs else 0.0,
            "total_r": round(sum(rs), 2),
            "total_dollar": round(
                sum(t["total_dollar"] for t in closed_t), 2),
            "avg_r": round(sum(rs) / len(rs), 3) if rs else 0.0,
            "profit_factor": round(gross_win / gross_loss, 2)
            if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
            "max_drawdown_r": round(max_dd, 2),
            "gross_win_r": round(gross_win, 2),
            "gross_loss_r": round(-gross_loss, 2),
            "close_reasons": dict(close_reasons),
            "close_displays": dict(displays),
            "layer_opens": sum(t["layer_opens"] for t in tickets),
        },
        "tp_hit_rates": tp_hit_rates,
        "regime_breakdown": regime_breakdown,
        "per_playbook": {k: {"trades": v["trades"],
                             "total_r": round(v["total_r"], 2),
                             "total_dollar": round(v["total_dollar"], 2)}
                         for k, v in sorted(per_pb.items())},
        "signals": {
            "records": len(signals),
            "by_playbook_decision": {
                f"{pb}:{dec}": c for (pb, dec), c in
                sorted(sig_decisions.items())},
            "gated_reasons_top": dict(sig_reasons.most_common(15)),
        },
        "management": {
            "partials": sum(len(v) for v in partials.values()),
            "sl_moves_by_reason": dict(sl_moves),
            "profit_locks": sl_moves.get("profit_lock", 0),
            "breakevens": sl_moves.get("breakeven", 0),
            "panic_closes": close_reasons.get("panic", 0),
            "timeout_closes": close_reasons.get("timeout", 0),
        },
        "pendings": {
            "placed": stats.get("pending", 0),
            "filled": stats.get("pending_filled", 0),
            "cancelled": stats.get("pending_cancelled", 0),
            "cancelled_by_reason": dict(pend_cancel),
            "score_decay_cancels": pend_decay,
        },
        "golden_swing": {
            "trades": 0,
            "note": ("W1/MN1 EMA200 needs 200 bars; dataset holds "
                     f"{w1_bars} W1 / {mn1_bars} MN1 bars, so W1/MN1 "
                     "bias is NEUTRAL throughout and alignment can "
                     "never reach Golden Swing on this window."),
            "journal_golden_notes": golden_notes,
        },
        "journal_counts": {"opens": len(opens), "closes": len(closes),
                           "notes": len(notes), "signals": len(signals)},
        "broker_end": broker.account(),
    }


def print_report(report: dict) -> None:
    t = report["trades"]
    print("\n================ FULL-ENGINE BACKTEST ================")
    print(f"window: {report['window']['start']} -> "
          f"{report['window']['end_open']} | "
          f"ticks {report['window']['ticks']:,}")
    print(f"run stats: {report['run_stats']}")
    print(f"\nTRADES closed={t['closed']} open_at_end={t['open_at_end']} "
          f"(unrealized {t['open_at_end_unrealized_r']:+.2f}R, not in totals)")
    print(f"win rate {t['win_rate']:.1%} | total {t['total_r']:+.2f}R "
          f"(${t['total_dollar']:+,.2f}) | avg {t['avg_r']:+.3f}R | "
          f"PF {t['profit_factor']} | max DD {t['max_drawdown_r']:.2f}R")
    print(f"close reasons: {t['close_reasons']}")
    print(f"TP HIT RATES (partials in order, / closed trades): "
          + ", ".join(
              f"{k.upper()} {v['hits']}/{v['denominator']} "
              f"({v['rate']:.1%})"
              for k, v in report.get("tp_hit_rates", {}).items()))
    print(f"REGIME BREAKDOWN (by playbook family): "
          + ", ".join(
              f"{reg} {d['trades']} trades {d['total_r']:+.2f}R "
              f"WR {d['win_rate']:.1%}"
              for reg, d in report.get(
                  "regime_breakdown", {}).items()))
    print(f"close displays: {t['close_displays']}")
    print(f"layer opens: {t['layer_opens']}")
    print("\nPER-PLAYBOOK (closed trades):")
    for pb, d in report["per_playbook"].items():
        print(f"  {pb}: {d['trades']} trades, {d['total_r']:+.2f}R, "
              f"${d['total_dollar']:+,.2f}")
    print(f"\nBLOCKED (runner counters): {report['blocked_stats']}")
    print(f"\nSIGNAL RECORDS: {report['signals']['records']}")
    for k, c in report["signals"]["by_playbook_decision"].items():
        print(f"  {k}: {c}")
    print(f"gated reasons: {report['signals']['gated_reasons_top']}")
    print(f"\nMANAGEMENT: {report['management']}")
    print(f"PENDINGS: {report['pendings']}")
    print(f"GOLDEN SWING: {report['golden_swing']}")
    print(f"JOURNAL: {report['journal_counts']}")
    print(f"BROKER END: {report['broker_end']}")
    print("=====================================================\n")


if __name__ == "__main__":
    sys.exit(main())
