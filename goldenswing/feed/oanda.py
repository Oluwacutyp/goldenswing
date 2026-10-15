"""OANDA v20 spot XAU/USD H1 feed for the paper loop.

Replaces the yfinance GC=F feed after the 2026-10-11 incident, when
yfinance printed phantom H1 bars (highs 4235-4242) on a day OANDA
spot never broke 4217.39. The phantom bars voided the first live
signal (a shadow long entered at 4233.5 on prices that never
existed). Lesson: free aggregated futures data can't be trusted
without a sanity gate.

Same interface as :class:`PollingFeed` — ``warm_start``,
``fetch_new``, ``poll_status``, ``poll_interval`` — so the runner
needs no changes. Two deliberate differences:

- Bars come from OANDA's practice API (``api-fxpractice.oanda.com``),
  which serves the same real market candles as live accounts. The
  instrument is spot XAU/USD, not GC=F futures; the few dollars of
  basis are immaterial to H1 signal logic, and spot matches what the
  owner sees on TradingView.
- Every batch passes a sanity gate before it reaches the loop:
  structurally impossible bars (high < low, non-positive prices)
  and implausible jumps (range > 6x recent ATR, close gap > 2% vs
  the previous close) are dropped and counted. Too many rejections
  marks the feed unhealthy instead of trading on garbage.

Auth: the practice API token lives in ``.oanda_token`` at the repo
root (mode 600, never logged, never committed). The feed reads it
once and caches it; ``OANDA_TOKEN_FILE`` env overrides the path.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------- config
API_HOST = "api-fxpractice.oanda.com"
INSTRUMENT = "XAU_USD"

#: Internal TF names -> OANDA CandlestickGranularity tokens. OANDA's
#: enum (https://developer.oanda.com/rest-live-v20/instrument-df/) is
#: S5 S10 S15 S30 M1 M2 M4 M5 M10 M15 M30 H1 H2 H3 H4 H6 H8 H12 D W M
#: — there is NO "D1", "W1", or "MN1". Passing "D1" verbatim made
#: OANDA reject every D1 request, so the D1 feed never succeeded once
#: while M5-H4 (valid tokens) worked through identical code. The
#: mapping applies ONLY at the OANDA request boundary: internal names
#: (STACK_TFS, MT5 TIMEFRAME_ constants) are untouched.
OANDA_GRANULARITY: dict[str, str] = {
    "D1": "D",    # daily
    "W1": "W",    # weekly
    "MN1": "M",   # monthly
}


def to_oanda_granularity(tf: str) -> str:
    """Translate an internal TF name to OANDA's granularity token.

    Tokens already in OANDA's enum pass through unchanged; only the
    internal D1/W1/MN1 spellings need translation.
    """
    return OANDA_GRANULARITY.get(tf, tf)


@dataclass
class OandaFeedConfig:
    instrument: str = INSTRUMENT
    granularity: str = "H1"
    #: warm-start history depth (one request; OANDA caps at 5000)
    warm_bars: int = 2000
    #: seconds between polls for fresh bars
    poll_seconds: float = 300.0
    #: a poll is stale if no fetch succeeded within this window
    stale_poll_minutes: float = 30.0
    #: bars are stale if the newest closed bar is older than this
    stale_bar_hours: float = 8.0
    #: skip the bar-staleness check while the market is closed
    weekend_grace: bool = True
    #: max backoff between retries after consecutive fetch failures
    max_backoff_seconds: float = 900.0
    #: sanity gate: bar range beyond this multiple of recent ATR is dropped
    max_range_atr_mult: float = 8.0
    #: sanity gate: close-to-close gap beyond this ATR multiple is dropped
    max_gap_atr_mult: float = 8.0
    #: live cross-check: forming candle's close vs OANDA pricing mid
    #: diverging beyond this fraction marks the feed unhealthy. Both
    #: are OANDA's own "right now", seconds apart, so any real
    #: divergence means the candles endpoint is lying (the 2026-10-11
    #: yfinance incident shape).
    max_price_diverge_frac: float = 0.002
    #: sanity gate: this many rejected bars in one batch = unhealthy feed
    max_rejections_per_batch: int = 3
    #: run the pricing cross-check on this feed (the multi-TF feed
    #: cross-checks on H1 only, so the other TFs set this False and
    #: skip 5 redundant pricing calls per tick).
    cross_check: bool = True


# ------------------------------------------------------------------ auth
_TOKEN_CACHE: str | None = None


def _token_path() -> Path:
    import os

    env = os.environ.get("OANDA_TOKEN_FILE")
    if env:
        return Path(env)
    # repo root = three levels above this file (feed/oanda.py)
    return Path(__file__).resolve().parents[2] / ".oanda_token"


def _token() -> str:
    global _TOKEN_CACHE
    if _TOKEN_CACHE is None:
        p = _token_path()
        _TOKEN_CACHE = p.read_text().strip()
        if not _TOKEN_CACHE:
            raise RuntimeError(f"OANDA token file is empty: {p}")
    return _TOKEN_CACHE


def _get(path: str, params: dict, timeout: float = 15.0) -> dict:
    """GET against the OANDA v20 practice API. Raises on HTTP error."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"https://{API_HOST}{path}?{query}"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {_token()}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"OANDA HTTP {exc.code}: {body}") from exc


# --------------------------------------------------------------- fetching
def normalize_oanda_candles(payload: dict,
                            complete_only: bool = True) -> pd.DataFrame:
    """OANDA candles JSON -> load_gold_h1 conventions.

    The forming bar is excluded precisely via OANDA's ``complete``
    flag (not by dropping the last row blindly), unless
    ``complete_only=False``. Columns are lowercase OHLCV, UTC
    DatetimeIndex, sorted, deduplicated.
    """
    rows = []
    for c in payload.get("candles", []):
        if complete_only and not c.get("complete"):
            continue
        m = c.get("mid") or {}
        try:
            rows.append({
                "ts": pd.Timestamp(c["time"], tz="UTC"),
                "open": float(m["o"]),
                "high": float(m["h"]),
                "low": float(m["l"]),
                "close": float(m["c"]),
                "volume": int(c.get("volume", 0)),
            })
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["open", "high", "low", "close", "volume"]]


def fetch_oanda(instrument: str = INSTRUMENT, granularity: str = "H1",
                count: int = 500, price: str = "M",
                complete_only: bool = True) -> pd.DataFrame:
    """Latest ``count`` H1 bars from OANDA (mid prices).

    With ``complete_only=True`` (default) the still-forming bar is
    excluded precisely via OANDA's ``complete`` flag. With False, the
    forming bar is included as the last row (used by the pricing
    cross-check, never by the loop).
    """
    payload = _get(
        f"/v3/instruments/{instrument}/candles",
        {"granularity": to_oanda_granularity(granularity),
         "count": count, "price": price},
    )
    return normalize_oanda_candles(payload, complete_only=complete_only)


def fetch_pricing_mid(instrument: str = INSTRUMENT,
                      account_id: str | None = None) -> tuple[float, str]:
    """Current mid price from OANDA's pricing endpoint.

    Returns (mid, timestamp). This is the independent cross-check
    against the candles endpoint: if the latest complete candle's
    close disagrees with live pricing beyond tolerance, the candles
    feed is lying (the 2026-10-11 yfinance incident shape).
    """
    if account_id is None:
        accounts = _get("/v3/accounts", {})
        try:
            account_id = accounts["accounts"][0]["id"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError("no OANDA practice accounts found") from exc
    payload = _get(
        f"/v3/accounts/{account_id}/pricing",
        {"instruments": instrument},
    )
    try:
        p = payload["prices"][0]
        bid = float(p["bids"][0]["price"])
        ask = float(p["asks"][0]["price"])
        return (bid + ask) / 2.0, p["time"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"unparseable pricing response: {payload}") \
            from exc


def sanity_gate(df: pd.DataFrame, ref: pd.DataFrame | None,
                cfg: OandaFeedConfig) -> tuple[pd.DataFrame, int, list[str]]:
    """Drop implausible bars; return (clean, n_rejected, reasons).

    ``ref`` is recent history used for the ATR context (may be None
    on the very first fetch, in which case only structural checks
    run). Structural impossibilities always reject; statistical
    outliers reject only when we have context.
    """
    if len(df) == 0:
        return df, 0, []
    reasons: list[str] = []
    keep = pd.Series(True, index=df.index)

    # structural: high>=low>=0, high>=open/close, low<=open/close
    bad_struct = (
        (df["high"] < df["low"])
        | (df["low"] <= 0)
        | (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
    )
    if bad_struct.any():
        reasons.append(
            f"{int(bad_struct.sum())} structurally impossible bar(s)")
    keep &= ~bad_struct

    # statistical: range and close gap vs recent ATR (expressed in ATR
    # multiples so the gate adapts to volatility; only true garbage
    # trips it — real news spikes stay under 8x ATR on H1)
    if ref is not None and len(ref) >= 15:
        tr = pd.concat([
            ref["high"] - ref["low"],
            (ref["high"] - ref["close"].shift(1)).abs(),
            (ref["low"] - ref["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        if pd.notna(atr) and atr > 0:
            work = df[keep]
            bad_range = (work["high"] - work["low"]) \
                > cfg.max_range_atr_mult * atr
            # sequential close gaps: each bar judged against its own
            # predecessor, not against a single distant reference
            bad_gap = pd.Series(False, index=work.index)
            for ts in work.index:
                i = df.index.get_loc(ts)
                pc = df["close"].iloc[i - 1] if i > 0 \
                    else ref["close"].iloc[-1]
                if abs(df.at[ts, "close"] - pc) \
                        > cfg.max_gap_atr_mult * atr:
                    bad_gap.at[ts] = True
            bad = bad_range | bad_gap
            if bad.any():
                reasons.append(
                    f"{int(bad.sum())} implausible bar(s) "
                    f"(range or gap >{cfg.max_range_atr_mult}xATR)")
                keep &= ~bad.reindex(df.index, fill_value=False)

    n_rejected = int((~keep).sum())
    return df[keep].copy(), n_rejected, reasons


# ---------------------------------------------------------- market hours
def market_closed_now(now: datetime | None = None) -> bool:
    """Rough OANDA XAU_USD hours: Sun 22:00 UTC - Fri 21:00 UTC.

    Used only for the staleness grace - never for trading decisions.
    """
    now = now or datetime.now(timezone.utc)
    wd, hr = now.weekday(), now.hour
    if wd == 5:                       # Saturday
        return True
    if wd == 6 and hr < 22:            # Sunday before the open
        return True
    if wd == 4 and hr >= 21:           # Friday after the close
        return True
    return False


# ------------------------------------------------------------- live feed
class OandaFeed:
    """Live H1 XAU/USD feed over the OANDA v20 practice API.

    Same interface as :class:`PollingFeed` (``warm_start``,
    ``fetch_new``, ``poll_status``, ``poll_interval``). Fetch
    failures never raise to the loop - they back off and report
    through ``poll_status``. Sanity-gate rejections are journaled
    via the status reason so bad data is loud, not silent.
    """

    def __init__(self, cfg: OandaFeedConfig | None = None) -> None:
        self.cfg = cfg or OandaFeedConfig()
        self._failures = 0
        self._last_poll_ok: float | None = None
        self._last_bar_time: pd.Timestamp | None = None
        self._last_error: str | None = None
        self._gate_rejections = 0
        self._gate_note: str | None = None
        self._ref: pd.DataFrame | None = None  # ATR context
        self._account_id: str | None = None
        self._xcheck_note: str | None = None

    # ------------------------------------------------------------ iface
    @property
    def poll_interval(self) -> float:
        if self._failures:
            backoff = self.cfg.poll_seconds * (2 ** self._failures)
            return min(backoff, self.cfg.max_backoff_seconds)
        return self.cfg.poll_seconds

    def warm_start(self) -> pd.DataFrame:
        try:
            df = fetch_oanda(self.cfg.instrument, self.cfg.granularity,
                             count=self.cfg.warm_bars)
            df, rej, why = sanity_gate(df, None, self.cfg)
            self._note_gate(rej, why)
            self._ref = df
            self._note_success(df)
        except Exception as exc:
            self._note_failure(str(exc))
            df = pd.DataFrame(columns=["open", "high", "low", "close",
                                       "volume"])
        return df

    def fetch_new(self, since: pd.Timestamp) -> pd.DataFrame:
        try:
            df = fetch_oanda(self.cfg.instrument, self.cfg.granularity,
                             count=500)
            # gate the bars the loop hasn't seen yet (overlap was
            # gated when new; OANDA revises recent candles and
            # re-gating revisions is noise, not signal)
            candidate = df[df.index > since]
            gated, rej, why = sanity_gate(candidate, self._ref, self.cfg)
            self._note_gate(rej, why)
            if rej > self.cfg.max_rejections_per_batch:
                # data looks poisoned: treat the whole batch as failed
                self._note_failure(
                    f"sanity gate rejected {rej} bars: "
                    + "; ".join(why))
                return pd.DataFrame(columns=["open", "high", "low",
                                             "close", "volume"])
            if self.cfg.cross_check:
                self._cross_check()
            if len(gated):
                self._ref = pd.concat([self._ref, gated]).iloc[-2000:] \
                    if self._ref is not None else gated
            self._note_success(df)
        except Exception as exc:
            self._note_failure(str(exc))
            return pd.DataFrame(columns=["open", "high", "low", "close",
                                         "volume"])
        return gated

    def poll_status(self) -> dict:
        now = time.time()
        if self._last_poll_ok is None:
            # Never hide the failure: before the first success the
            # real error is the ONLY diagnostic, so surface it always.
            # (Previously this returned a bare "no successful poll yet"
            # and swallowed _last_error, making the D1 failure
            # undiagnosable for its whole lifetime.)
            reason = "no successful poll yet"
            if self._last_error:
                reason += f" (last error: {self._last_error})"
            return {"ok": False, "reason": reason}
        idle = (now - self._last_poll_ok) / 60.0
        if idle > self.cfg.stale_poll_minutes:
            return {"ok": False,
                    "reason": f"no successful poll for {idle:.0f} min "
                              f"(last error: {self._last_error})"}
        if self._gate_note and self._gate_rejections:
            return {"ok": False,
                    "reason": f"sanity gate: {self._gate_note}"}
        if self._xcheck_note:
            return {"ok": False, "reason": self._xcheck_note}
        if self._last_bar_time is not None and not (
                self.cfg.weekend_grace and market_closed_now()):
            age_h = (datetime.now(timezone.utc)
                     - self._last_bar_time.to_pydatetime()) \
                .total_seconds() / 3600.0
            if age_h > self.cfg.stale_bar_hours:
                return {"ok": False,
                        "reason": f"newest closed bar is {age_h:.1f}h old"}
        return {"ok": True, "reason": None,
                "consecutive_failures": self._failures}

    # ---------------------------------------------------------- helpers
    def _note_success(self, df: pd.DataFrame) -> None:
        self._failures = 0
        self._last_error = None
        self._last_poll_ok = time.time()
        if len(df):
            self._last_bar_time = df.index[-1]

    def _note_failure(self, error: str) -> None:
        self._failures += 1
        self._last_error = error

    def _note_gate(self, rejected: int, reasons: list[str]) -> None:
        self._gate_rejections = rejected
        self._gate_note = "; ".join(reasons) if reasons else None

    def _cross_check(self) -> None:
        """Forming candle close vs live pricing mid.

        Both are OANDA's own "right now", seconds apart: the forming
        H1 bar's latest price against the pricing endpoint's mid. Any
        real divergence means the candles endpoint is lying (the
        2026-10-11 yfinance incident shape) — halt instead of trading
        on it. A failed cross-check (pricing endpoint down) is noted
        but does not halt; the sanity gate already ran.
        """
        self._xcheck_note = None
        try:
            if self._account_id is None:
                self._account_id = self._discover_account()
            mid, _ts = fetch_pricing_mid(self.cfg.instrument,
                                         self._account_id)
            forming = fetch_oanda(self.cfg.instrument,
                                  self.cfg.granularity,
                                  count=2, complete_only=False)
            if len(forming) == 0:
                return
            fclose = float(forming["close"].iloc[-1])
        except Exception:
            return  # pricing/candles down: gate already ran, don't pile on
        div = abs(mid - fclose) / fclose
        if div > self.cfg.max_price_diverge_frac:
            self._xcheck_note = (
                f"pricing cross-check failed: forming bar {fclose:.1f} "
                f"vs live mid {mid:.1f} ({div:.2%} apart)")

    def _discover_account(self) -> str:
        accounts = _get("/v3/accounts", {})
        return accounts["accounts"][0]["id"]


# -------------------------------------------------- multi-timeframe feed
#: The hierarchy's timeframe stack: D1 bias -> H4 zones -> H1 trigger
#: -> M15/M30 entry clock -> M5 QM leg.  The multi feed serves all of
#: them; the loop evaluates one tick over the whole stack.
STACK_TFS: tuple[str, ...] = ("D1", "H4", "H1", "M15", "M30", "M5")

#: Warm-start depth per TF (OANDA caps one request at 5000 bars).
_WARM_BARS: dict[str, int] = {
    "D1": 600, "H4": 1200, "H1": 2000, "M15": 2500, "M30": 2500,
    "M5": 3000,
}

#: A bar is stale if the newest closed bar is older than this, per TF.
#: H4/D1 bars legitimately sit for hours (a D1 bar can be ~24h old);
#: the single-TF 8h default would flag them stale every day.
_STALE_BAR_HOURS: dict[str, float] = {
    "D1": 36.0, "H4": 12.0, "H1": 8.0, "M15": 3.0, "M30": 4.0, "M5": 2.0,
}


@dataclass
class MultiOandaFeedConfig:
    instrument: str = INSTRUMENT
    granularities: tuple[str, ...] = STACK_TFS
    poll_seconds: float = 300.0
    stale_poll_minutes: float = 30.0
    max_backoff_seconds: float = 900.0
    max_range_atr_mult: float = 8.0
    max_gap_atr_mult: float = 8.0
    max_price_diverge_frac: float = 0.002
    max_rejections_per_batch: int = 3


class MultiOandaFeed:
    """All-timeframe OANDA feed: one :class:`OandaFeed` per granularity.

    The live loop's hierarchy needs D1/H4/H1/M15/M30/M5 together —
    never H1 alone.  Each TF keeps its own sanity gate, ATR context,
    failure backoff, and staleness clock (H4/D1 bars are not "stale"
    at 8h).  The pricing cross-check runs on H1 only.

    Interface (superset of the single-TF feed):

    - ``warm_start_all() -> {tf: DataFrame}`` — full history per TF.
    - ``fetch_new_all(since_by_tf) -> {tf: DataFrame}`` — closed bars
      with ``index > since_by_tf[tf]`` per TF (a TF missing from
      ``since_by_tf`` returns its latest 500 bars; the stack's
      forward-only append dedupes them).
    - ``warm_start()`` / ``fetch_new(since)`` — H1 only, the legacy
      single-TF interface, so old callers keep working.
    - ``poll_status()`` — ok iff EVERY TF feed is ok; the reason
      names the first failing TF.
    - ``poll_interval`` — the H1 cadence (the loop is H1-driven; M5
      bars accumulate between ticks).
    """

    def __init__(self, cfg: MultiOandaFeedConfig | None = None) -> None:
        self.cfg = cfg or MultiOandaFeedConfig()
        self.feeds: dict[str, OandaFeed] = {}
        for tf in self.cfg.granularities:
            self.feeds[tf] = OandaFeed(OandaFeedConfig(
                instrument=self.cfg.instrument, granularity=tf,
                warm_bars=_WARM_BARS.get(tf, 2000),
                poll_seconds=self.cfg.poll_seconds,
                stale_poll_minutes=self.cfg.stale_poll_minutes,
                stale_bar_hours=_STALE_BAR_HOURS.get(tf, 8.0),
                max_backoff_seconds=self.cfg.max_backoff_seconds,
                max_range_atr_mult=self.cfg.max_range_atr_mult,
                max_gap_atr_mult=self.cfg.max_gap_atr_mult,
                max_price_diverge_frac=self.cfg.max_price_diverge_frac,
                max_rejections_per_batch=self.cfg.max_rejections_per_batch,
                cross_check=(tf == "H1")))

    # ------------------------------------------------------------ iface
    @property
    def poll_interval(self) -> float:
        h1 = self.feeds.get("H1")
        return h1.poll_interval if h1 is not None \
            else self.cfg.poll_seconds

    def warm_start_all(self) -> dict[str, pd.DataFrame]:
        return {tf: feed.warm_start()
                for tf, feed in self.feeds.items()}

    def fetch_new_all(self, since_by_tf: dict | None = None
                      ) -> dict[str, pd.DataFrame]:
        since_by_tf = since_by_tf or {}
        out: dict[str, pd.DataFrame] = {}
        for tf, feed in self.feeds.items():
            since = since_by_tf.get(tf)
            if since is None:
                # No cursor for this TF: return the latest 500 bars;
                # the stack's forward-only append dedupes what it
                # already holds, so this can never rewind it.
                since = pd.Timestamp("1970-01-01", tz="UTC")
            out[tf] = feed.fetch_new(since)
        return out

    def warm_start(self) -> pd.DataFrame:
        return self.feeds["H1"].warm_start()

    def fetch_new(self, since: pd.Timestamp) -> pd.DataFrame:
        return self.feeds["H1"].fetch_new(since)

    def poll_status(self) -> dict:
        for tf, feed in self.feeds.items():
            status = feed.poll_status()
            if not status.get("ok", True):
                return {"ok": False,
                        "reason": f"{tf}: {status.get('reason')}",
                        "tf": tf}
        return {"ok": True, "reason": None}
