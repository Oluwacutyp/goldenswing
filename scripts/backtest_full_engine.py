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
  cd ~/workspace/goldenswing && .venv/bin/python scripts/backtest_full_engine.py [--max-ticks N] [--out DIR] [--days N] [--only-playbook NAME]
Paper only.  No live trading, no network (news fetcher raises).

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
import re
import sys
import time
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
LEVERAGE = 500.0
SPREAD = 0.20
CONTRACT = 100.0  # 1 lot = 100 oz


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-ticks", type=int, default=None)
    ap.add_argument("--out", type=str, default=None,
                    help="output dir (default: fresh temp dir)")
    ap.add_argument("--days", type=int, default=None,
                    help="backtest only the most recent N days "
                         "(warmup bars still served)")
    ap.add_argument("--only-playbook", type=str, default=None,
                    help="run with only this executing playbook enabled")
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else Path(
        __import__("tempfile").mkdtemp(prefix="full_engine_bt_"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"out: {out}")

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
    return 0


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
