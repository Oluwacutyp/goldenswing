"""Interactive terminal server: the engine serves its own live view.

Stdlib only (``http.server.ThreadingHTTPServer``) — no new dependency
for a read-only page. The server is transport only: every fact comes
from the provider (the runner's live stack + latest-signal studies +
calendar read). The server never decides, never orders, never holds
market state of its own.

Routes (GET only — read-only is enforced, not promised):

- ``GET /`` — the terminal page (``charts.terminal``).
- ``GET /api/frames?tf=M5`` — capped bars + session spans + studies
  as data for one timeframe.
- ``GET /api/status`` — server time, market-open per the engine
  calendar, and per-timeframe last-bar age + stale flags.

Anything else: POST/PUT/DELETE/PATCH answer 405, unknown paths 404,
unknown timeframes 400. There is no order, command, or control
endpoint anywhere in this module — a client cannot trade through it.

Provider protocol (duck-typed, implemented by the runner):

- ``frames(tf)`` -> frame with a DatetimeIndex and OHLC columns,
  or None when the stack has no bars for that timeframe.
- ``studies()`` -> dict (``signal``, ``signal_time``, ``zone``,
  ``nearby``, ``fib``) from the latest signal, or None when no
  signal has been charted yet.
- ``status()`` -> dict (``server_time`` ISO, ``market_open`` bool,
  ``tfs`` mapping tf -> ``last_bar`` ISO or None).

Freshness rule (mirrors /status semantics): a timeframe is stale
iff the market is open AND its last closed bar is older than twice
its interval. Market closed reads CLOSED, never stale — closed is
not a breach.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlparse

from ..reader import session_timeframes as sessions
from .terminal import build_terminal_page

TF_INTERVALS = {"M5": 300, "M15": 900, "M30": 1800,
                "H1": 3600, "H4": 14400, "D1": 86400}
TF_ORDER = ("M5", "M15", "M30", "H1", "H4", "D1")
_BARS_CAP = 500


def bars_to_rows(frame) -> list:
    """Frame -> [iso, open, high, low, close] rows, newest capped."""
    rows = []
    for ts, row in frame.iloc[-_BARS_CAP:].iterrows():
        try:
            iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            rows.append([iso, float(row["open"]), float(row["high"]),
                         float(row["low"]), float(row["close"])])
        except Exception:
            continue
    return rows


def session_spans(frame) -> list:
    """Killzone runs over the capped window, as [start_i, end_i]."""
    view = frame.iloc[-_BARS_CAP:]
    spans = []
    run_start = None
    n = len(view)
    for i, ts in enumerate(view.index):
        try:
            _name, kz = sessions.session_of(ts)
        except Exception:
            kz = None
        if kz and run_start is None:
            run_start = i
        if (not kz or i == n - 1) and run_start is not None:
            run_end = i if not kz else i + 1
            spans.append([run_start, run_end])
            run_start = None
    return spans


def _parse_iso(value: str | None):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def compute_status(*, server_time: str, market_open: bool,
                   last_bars: dict, now: datetime | None = None) -> dict:
    """Freshness payload: per-TF last bar, age, and stale flag."""
    now = now or datetime.now(timezone.utc)
    tfs = {}
    for tf in TF_ORDER:
        last = last_bars.get(tf)
        last_dt = _parse_iso(last)
        age = (now - last_dt).total_seconds() if last_dt else None
        stale = bool(market_open and age is not None
                     and age > 2 * TF_INTERVALS[tf])
        tfs[tf] = {"last_bar": last, "age_s": age, "stale": stale}
    return {"server_time": server_time, "market_open": market_open,
            "tfs": tfs}


def make_handler(provider, instrument: str):
    """Build the request handler class bound to one provider."""

    class TerminalHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep the engine log clean
            pass

        def _send_json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, code: int, body: str) -> None:
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 - stdlib handler name
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/":
                self._send_html(200, build_terminal_page(
                    instrument=instrument))
                return
            if path == "/api/frames":
                tf = (parse_qs(parsed.query).get("tf") or [None])[0]
                if tf is None:
                    self._send_json(400, {"error": "tf is required"})
                    return
                if tf not in TF_INTERVALS:
                    self._send_json(400, {"error": f"unknown tf {tf!r}"})
                    return
                try:
                    frame = provider.frames(tf)
                except Exception as exc:
                    self._send_json(500, {"error": f"provider: {exc}"})
                    return
                if frame is None or len(frame) == 0:
                    self._send_json(404, {"error": f"no bars for {tf}"})
                    return
                try:
                    studies = provider.studies()
                except Exception:
                    studies = None
                self._send_json(200, {
                    "tf": tf, "bars": bars_to_rows(frame),
                    "sessions": session_spans(frame),
                    "studies": studies})
                return
            if path == "/api/status":
                try:
                    raw = provider.status()
                except Exception as exc:
                    self._send_json(500, {"error": f"provider: {exc}"})
                    return
                payload = compute_status(
                    server_time=raw.get("server_time", ""),
                    market_open=bool(raw.get("market_open", False)),
                    last_bars={tf: (raw.get("tfs") or {}).get(tf)
                               for tf in TF_ORDER})
                payload["instrument"] = instrument
                self._send_json(200, payload)
                return
            self._send_json(404, {"error": "unknown path"})

        def _method_not_allowed(self) -> None:
            self._send_json(405, {"error": "terminal is read-only"})

        do_POST = _method_not_allowed  # noqa: N802
        do_PUT = _method_not_allowed  # noqa: N802
        do_DELETE = _method_not_allowed  # noqa: N802
        do_PATCH = _method_not_allowed  # noqa: N802

    return TerminalHandler


class TerminalServer:
    """A terminal server bound to one provider, localhost-first."""

    def __init__(self, provider, *, host: str = "127.0.0.1",
                 port: int = 8899, instrument: str = "XAUUSD"):
        self._httpd = ThreadingHTTPServer(
            (host, port), make_handler(provider, instrument))
        self._httpd.daemon_threads = True
        self.host = host
        self.port = self._httpd.server_address[1]
        self._thread: Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> str:
        """Start serving in a daemon thread; idempotent."""
        if self.running:
            return self.url
        self._thread = Thread(target=self._httpd.serve_forever,
                              kwargs={"poll_interval": 0.25},
                              daemon=True)
        self._thread.start()
        return self.url

    def stop(self) -> None:
        """Stop serving; safe to call when never started."""
        if self._thread is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread = None
