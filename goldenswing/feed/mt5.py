"""MT5 feed adapter — MetaTrader5 is the source of truth for EVERYTHING.

Prices, bars, ticks, volumes, spread, account state, open positions:
all of it comes from the MT5 terminal.  There is NO OANDA cross-check
and no second source of truth in this path — the owner's correction
(2026-10-14): OANDA 5-minute polling cannot manage live trades (5
minutes is enough to miss a breakeven or an entry), and OANDA/pandas
cannot provide tick volume, real volume, or spread.  OANDA remains
only for the local paper loops (``scripts/paper_live.py``,
``scripts/paper_live_exness.py``) until the EC2 MT5 engine takes over.

Plumbing follows the owner's previous bots
(``myxauusdbot__4__1_gn20.py`` ``MT5Connection``) — terminal
initialize -> login -> symbol_select, ``copy_rates_from_pos`` with
retries + exponential backoff, ``tick_volume`` renamed to ``volume``,
``symbol_info_tick`` for the fresh tick, ``positions_get`` /
``account_info`` for live state.  Their old failure was strategy
voting (five strategies cancelling each other), not this plumbing.

Two MT5 realities this adapter handles honestly:

- The Python API is pull-based: there is no tick callback.  The
  :class:`TickPump` reads ``symbol_info_tick`` at tick cadence
  (default 0.25s) and delivers every NEW tick; management fires on
  the crossing tick, never on the next bar or poll cycle.  That is
  the same pattern the old bots used.
- MT5 bar timestamps are SERVER time, not UTC.  The adapter
  calibrates the server->UTC offset from a fresh tick on connect
  (``tick.time - time.time()``) and stamps every bar in UTC, so the
  hierarchy's causal slicing and the session killzones (New York
  wall time) read the right clock.  Override with
  ``utc_offset_hours`` when the broker's offset is known.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..live.hierarchy import STACK_TFS, TF_MINUTES

#: MT5 account trade modes (ACCOUNT_TRADE_MODE_*).
TRADE_MODE_DEMO = 0
TRADE_MODE_CONTEST = 1
TRADE_MODE_REAL = 2

#: Warm-start bar counts per timeframe (matches the OANDA feed's).
_WARM_BARS: dict[str, int] = {
    "D1": 2000, "H4": 2000, "H1": 2000, "M15": 2000, "M30": 2000,
    "M5": 2000,
}


class MT5Unavailable(RuntimeError):
    """The MetaTrader5 library or terminal cannot be reached."""


@dataclass
class MT5FeedConfig:
    symbol: str = "XAUUSD"
    granularities: tuple[str, ...] = STACK_TFS
    #: Credentials: None means attach to the already-logged-in terminal
    #: (the normal EC2 case — the owner logs the terminal in by hand).
    #: When set, they come from env (GS_MT5_LOGIN / GS_MT5_PASSWORD /
    #: GS_MT5_SERVER) via scripts/run_mt5.py — never hardcoded.
    login: int | None = None
    password: str | None = None
    server: str | None = None
    terminal_path: str | None = None
    #: Server->UTC offset hours. None = auto-calibrate from a fresh tick.
    utc_offset_hours: float | None = None
    max_connect_attempts: int = 5
    max_fetch_retries: int = 3
    tick_interval: float = 0.25
    warm_bars: dict[str, int] = field(
        default_factory=lambda: dict(_WARM_BARS))


class MT5Feed:
    """All-timeframe MT5 feed: bars + ticks + account + positions.

    Interface mirrors :class:`goldenswing.feed.oanda.MultiOandaFeed`
    (``warm_start_all`` / ``fetch_new_all`` / ``poll_status``) so the
    stack code works unchanged — only the data source differs.
    """

    def __init__(self, cfg: MT5FeedConfig | None = None,
                 mt5_module=None) -> None:
        self.cfg = cfg or MT5FeedConfig()
        self._injected = mt5_module
        self._mt5 = None
        self._utc_offset_seconds: float = 0.0
        self.connected = False

    # ------------------------------------------------------------- library
    def _lib(self):
        if self._mt5 is not None:
            return self._mt5
        if self._injected is not None:
            self._mt5 = self._injected
            return self._mt5
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise MT5Unavailable(
                "the MetaTrader5 Python package is not importable here — "
                "live MT5 feed requires a Windows host with the package "
                "installed and an MT5 terminal logged in and active "
                f"(import error: {exc})") from exc
        self._mt5 = mt5
        return mt5

    def available(self) -> bool:
        """True iff the library imports AND the terminal is connected."""
        try:
            mt5 = self._lib()
            info = mt5.terminal_info()
            return info is not None and bool(getattr(info, "connected", False))
        except Exception:
            return False

    # ------------------------------------------------------------ connect
    def connect(self) -> tuple[bool, str]:
        """initialize -> login -> symbol_select, old-bot retry pattern.

        With no credentials it attaches to the terminal as-is (the
        normal case).  Calibrates the server->UTC offset from a fresh
        tick so bars are stamped in UTC.
        """
        mt5 = self._lib()
        last_err = ""
        for attempt in range(self.cfg.max_connect_attempts):
            kwargs: dict = {}
            if self.cfg.terminal_path:
                kwargs["path"] = self.cfg.terminal_path
            if self.cfg.login is not None:
                kwargs.update(login=int(self.cfg.login),
                              password=self.cfg.password,
                              server=self.cfg.server)
            if mt5.initialize(**kwargs):
                info = mt5.terminal_info()
                if info is not None and getattr(info, "connected", False):
                    if not mt5.symbol_select(self.cfg.symbol, True):
                        last_err = f"symbol_select({self.cfg.symbol}) failed"
                        mt5.shutdown()
                        time.sleep(2 ** attempt)
                        continue
                    self.connected = True
                    self._calibrate_utc_offset()
                    return True, "MT5 connected"
                last_err = "terminal not connected to a broker server"
            else:
                last_err = f"initialize failed: {mt5.last_error()}"
            time.sleep(2 ** attempt)
        self.connected = False
        return False, f"MT5 connect failed after {self.cfg.max_connect_attempts} attempts: {last_err}"

    def _calibrate_utc_offset(self) -> None:
        """Server->UTC offset from a fresh tick (seconds, server - UTC).

        A tick's ``time`` is server time; the difference to local UTC
        now is the offset (modulo sub-second latency, rounded to the
        minute — brokers sit on whole-minute offsets).
        """
        if self.cfg.utc_offset_hours is not None:
            self._utc_offset_seconds = float(self.cfg.utc_offset_hours) * 3600.0
            return
        try:
            mt5 = self._lib()
            tick = mt5.symbol_info_tick(self.cfg.symbol)
            if tick is not None and getattr(tick, "time", None):
                raw = float(tick.time) - time.time()
                # Round to the nearest minute: latency noise is seconds.
                self._utc_offset_seconds = round(raw / 60.0) * 60.0
        except Exception:
            self._utc_offset_seconds = 0.0

    def shutdown(self) -> None:
        if self._mt5 is not None and self._injected is None:
            self._mt5.shutdown()
        self.connected = False

    # --------------------------------------------------------------- bars
    def _tf_const(self, tf: str) -> int:
        mt5 = self._lib()
        const = getattr(mt5, f"TIMEFRAME_{tf}", None)
        if const is None:
            raise MT5Unavailable(f"MT5 has no TIMEFRAME_{tf}")
        return int(const)

    def _to_utc_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Server-time ``time`` column -> UTC DatetimeIndex."""
        stamps = pd.to_datetime(df["time"], unit="s", utc=True)
        if self._utc_offset_seconds:
            stamps = stamps - pd.Timedelta(seconds=self._utc_offset_seconds)
        return df.assign(_ts=stamps).set_index("_ts").sort_index()

    def _frame(self, rates) -> pd.DataFrame:
        """MT5 rates array -> OHLC frame (UTC index).

        ``volume`` is MT5 ``tick_volume`` (the old bots' convention);
        ``real_volume`` is carried alongside when the terminal reports
        it (often 0 for metals — callers must check, never assume).
        ``spread`` is in points, as reported per bar.
        """
        df = pd.DataFrame(rates)
        if "tick_volume" in df.columns:
            df = df.rename(columns={"tick_volume": "volume"})
        if "volume" not in df.columns:
            df["volume"] = 0
        if "real_volume" not in df.columns:
            df["real_volume"] = 0
        if "spread" not in df.columns:
            df["spread"] = 0
        df = self._to_utc_index(df)
        return df[["open", "high", "low", "close", "volume",
                   "real_volume", "spread"]]

    def bars(self, tf: str, count: int) -> pd.DataFrame:
        """Latest ``count`` closed+forming bars for ``tf`` (UTC index).

        Retries with the old bots' backoff (0.5 * 2**(attempt-1)).
        Callers trade on closed bars only — the last row is the
        still-forming bar, exactly like the OANDA feed's contract.
        """
        mt5 = self._lib()
        last_err: object = None
        for attempt in range(1, self.cfg.max_fetch_retries + 1):
            rates = mt5.copy_rates_from_pos(
                self.cfg.symbol, self._tf_const(tf), 0, int(count))
            if rates is not None and len(rates) > 0:
                return self._frame(rates)
            last_err = mt5.last_error()
            if attempt < self.cfg.max_fetch_retries:
                time.sleep(0.5 * (2 ** (attempt - 1)))
        raise MT5Unavailable(
            f"copy_rates_from_pos({self.cfg.symbol} {tf}) failed: {last_err}")

    def warm_start_all(self) -> dict[str, pd.DataFrame]:
        """Full CLOSED history per TF (forming bar dropped).

        The stack's forward-only cursor is keyed on bar opens: if the
        forming bar entered the stack, its later closed version
        (same open) would be rejected as a duplicate and the stack
        would keep the forming snapshot forever.  The OANDA feed
        avoids this with ``complete_only=True``; this method does
        the same.  Use :meth:`bars` for the raw frame (forming bar
        included) when you need it.
        """
        now = datetime.now(timezone.utc)
        return {tf: self._closed_only(self.bars(tf, self.cfg.warm_bars.get(tf, 2000)), tf, now)
                for tf in self.cfg.granularities}

    def fetch_new_all(self, since_by_tf: dict | None = None
                      ) -> dict[str, pd.DataFrame]:
        """New CLOSED bars per TF (forming bar dropped — see above)."""
        since_by_tf = since_by_tf or {}
        now = datetime.now(timezone.utc)
        out: dict[str, pd.DataFrame] = {}
        for tf in self.cfg.granularities:
            frame = self.bars(tf, 500)
            since = since_by_tf.get(tf)
            if since is not None:
                frame = frame[frame.index > since]
            out[tf] = self._closed_only(frame, tf, now)
        return out

    @staticmethod
    def _closed_only(frame: pd.DataFrame, tf: str,
                     now: datetime) -> pd.DataFrame:
        """Drop the still-forming bar: open + TF minutes must be <= now."""
        if frame is None or len(frame) == 0:
            return frame
        cutoff = pd.Timestamp(now) - pd.Timedelta(minutes=TF_MINUTES[tf])
        return frame[frame.index <= cutoff]

    def poll_status(self) -> dict:
        try:
            mt5 = self._lib()
            info = mt5.terminal_info()
            if info is None or not getattr(info, "connected", False):
                return {"ok": False, "reason": "terminal not connected"}
            return {"ok": True, "reason": None}
        except Exception as exc:
            return {"ok": False, "reason": f"MT5 poll failed: {exc}"}

    # --------------------------------------------------------------- ticks
    def tick(self) -> dict | None:
        """Fresh tick from the terminal (old-bot ``get_fresh_tick``).

        Carries ``tick_volume`` (``symbol_info_tick.volume``) and
        ``real_volume`` (``volume_real``) alongside bid/ask — the
        tick-CSV logger records them exactly as seen.  Older fakes
        without volume attributes get 0, never a crash.
        """
        mt5 = self._lib()
        t = mt5.symbol_info_tick(self.cfg.symbol)
        if t is None:
            return None
        bid = float(t.bid)
        ask = float(t.ask)
        ts = datetime.fromtimestamp(float(t.time), tz=timezone.utc)
        if self._utc_offset_seconds:
            ts = ts - timedelta(seconds=self._utc_offset_seconds)
        return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2.0,
                "spread": ask - bid, "time": ts,
                "tick_volume": float(getattr(t, "volume", 0) or 0),
                "real_volume": float(getattr(t, "volume_real", 0) or 0)}

    def ticks_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Tick history via copy_ticks_range (COPY_TICKS_ALL)."""
        mt5 = self._lib()
        ticks = mt5.copy_ticks_range(self.cfg.symbol, start, end,
                                     mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return pd.DataFrame(
                columns=["time", "bid", "ask", "volume", "flags"])
        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        if self._utc_offset_seconds:
            df["time"] = df["time"] - pd.Timedelta(
                seconds=self._utc_offset_seconds)
        return df

    # ------------------------------------------------- account / positions
    def account_state(self) -> dict:
        """Account state, straight from the terminal — MT5 is the
        source of truth for balance / equity / margin / trade mode."""
        mt5 = self._lib()
        info = mt5.account_info()
        if info is None:
            raise MT5Unavailable(f"account_info failed: {mt5.last_error()}")
        return {
            "balance": float(info.balance),
            "equity": float(info.equity),
            "margin": float(getattr(info, "margin", 0.0)),
            "free_margin": float(getattr(info, "margin_free", 0.0)),
            "currency": str(getattr(info, "currency", "")),
            "login": int(getattr(info, "login", 0) or 0),
            "leverage": int(getattr(info, "leverage", 0) or 0),
            "trade_mode": int(getattr(info, "trade_mode", TRADE_MODE_DEMO)),
        }

    def open_positions(self, symbol: str | None = None) -> list[dict]:
        """Open positions, straight from the terminal — MT5 is the
        source of truth for what is actually open."""
        mt5 = self._lib()
        raw = mt5.positions_get(
            symbol=symbol) if symbol else mt5.positions_get()
        if raw is None:
            raise MT5Unavailable(f"positions_get failed: {mt5.last_error()}")
        out = []
        for p in raw:
            out.append({
                "ticket": int(p.ticket),
                "symbol": str(p.symbol),
                "direction": 1 if int(p.type) == 0 else -1,
                "volume": float(p.volume),
                "entry_price": float(p.price_open),
                "sl": float(p.sl) if p.sl else None,
                "tp": float(p.tp) if p.tp else None,
                "magic": int(getattr(p, "magic", 0)),
                "comment": str(getattr(p, "comment", "") or ""),
                "open_time": getattr(p, "time", None),
            })
        return out

    # ------------------------------------------------------ bar boundaries
    @staticmethod
    def closed_bar_close(tf: str, tick_time) -> pd.Timestamp:
        """Close of the latest bar that had CLOSED by ``tick_time``.

        MT5 bars align to UTC boundaries: an H1 bar opens on the hour
        and closes on the next hour, so the latest closed H1 close is
        ``tick_time`` floored to the hour.  D1 floors to midnight UTC.
        The runner compares this against its per-TF last-processed
        close to detect new closed bars — from tick time, never from
        a poll timer.
        """
        ts = pd.Timestamp(tick_time)
        if tf == "D1":
            return ts.floor("D")
        minutes = TF_MINUTES[tf]
        return ts.floor(f"{int(minutes)}min")


class TickPump:
    """Delivers every NEW tick from the terminal at tick cadence.

    The MT5 Python API has no push callback, so the pump reads
    ``symbol_info_tick`` in a tight loop (default 0.25s) and calls
    ``on_tick`` for each tick that is new (later timestamp, or same
    timestamp with a changed bid/ask).  Management subscribed to the
    pump fires on the crossing tick — never on the next bar or poll.
    """

    def __init__(self, feed: MT5Feed, *,
                 interval: float | None = None) -> None:
        self.feed = feed
        self.interval = interval if interval is not None \
            else feed.cfg.tick_interval
        self._stopped = False
        self.delivered = 0

    def stop(self) -> None:
        self._stopped = True

    def run(self, on_tick, *, max_ticks: int | None = None) -> int:
        """Pump ticks into ``on_tick(tick)`` until stopped / capped.

        Returns the number of ticks delivered.  A tick that is
        identical to the previous one (same time, same bid/ask) is
        not delivered twice.  ``on_tick`` raising :class:`StopIteration`
        stops the pump cleanly.
        """
        last: dict | None = None
        while not self._stopped:
            if max_ticks is not None and self.delivered >= max_ticks:
                break
            tick = self.feed.tick()
            if tick is None:
                time.sleep(self.interval)
                continue
            if last is not None and tick["time"] == last["time"] \
                    and tick["bid"] == last["bid"] \
                    and tick["ask"] == last["ask"]:
                time.sleep(self.interval)
                continue
            last = tick
            try:
                on_tick(tick)
            except StopIteration:
                break
            self.delivered += 1
            time.sleep(self.interval)
        return self.delivered
