"""Chart workspace: structure measurement, terminal server, page,
PNG roles and TV context — all on measured structure, never
synthesised.  The 15-bar frame is hand-traced (see the comments):
its swings, break, sweeps and FVG are asserted exactly."""
import http.client
import io
import json
import threading
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402

from goldenswing.charts import structure as st  # noqa: E402
from goldenswing.charts.engine import (  # noqa: E402
    _render_mt5_roles_mpl, render_mt5_roles)
from goldenswing.charts.signal_model import (  # noqa: E402
    lifecycle_from_events)
from goldenswing.charts.terminal import build_terminal_page  # noqa: E402
from goldenswing.charts.terminal_server import (  # noqa: E402
    TerminalServer)
from goldenswing.charts.tv_page import _context_cfg  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"


def _ohlc(close, up, dn, t0, freq="1h"):
    """Frame from closes + wicks; open is the previous close."""
    close = np.asarray(close, dtype=float)
    up = np.asarray(up, dtype=float)
    dn = np.asarray(dn, dtype=float)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    idx = pd.date_range(t0, periods=len(close), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": open_,
         "high": np.maximum(open_, close) + up,
         "low": np.minimum(open_, close) - dn,
         "close": close,
         "volume": np.full(len(close), 150.0)},
        index=idx)


# Hand-traced 15-bar frame (closes/up/dn as in the module docstring
# reasoning): swings HIGH@2=103.0, LOW@5=98.5, HIGH@7=104.0,
# LOW@11=97.5; labels H/L/HH/LL; breaks of 98.5 at bar 11 (failed
# at 12), of 103.0 and 104.0 at bar 13; sweeps above 103.0 at
# bars 5 and 8 (wick 0.5) and below 98.5 at bar 12 (wick 1.0);
# bearish FVG @10 (101.5/100.5) mitigated @13, bullish FVG @14
# (103.5/100.5) still open.
_TRACED_CLOSE = [100, 101, 102, 101, 100, 99, 101, 103, 102, 100,
                 99, 98, 100, 105, 104]
_TRACED_UP = [0.5, 0.5, 1.0, 0.5, 0.5, 3.5, 0.5, 1.0, 0.5, 0.5,
              0.5, 0.5, 0.5, 1.0, 0.5]
_TRACED_DN = [0.5] * 15


@pytest.fixture(scope="module")
def traced():
    return _ohlc(_TRACED_CLOSE, _TRACED_UP, _TRACED_DN,
                 "2026-11-20 00:00")


@pytest.fixture(scope="module")
def sessions_frame():
    """Thu 2026-11-19 + Fri 2026-11-20 + Mon 2026-11-23 (H1)."""
    rng = np.random.default_rng(7)
    n = 24 + 24 + 13
    steps = rng.normal(0.0, 1.5, n).cumsum()
    close = 4100.0 + steps
    idx = pd.to_datetime(
        [pd.Timestamp("2026-11-19 00:00", tz="UTC")
         + pd.Timedelta(hours=i) for i in range(24)]
        + [pd.Timestamp("2026-11-20 00:00", tz="UTC")
           + pd.Timedelta(hours=i) for i in range(24)]
        + [pd.Timestamp("2026-11-23 00:00", tz="UTC")
           + pd.Timedelta(hours=i) for i in range(13)])
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame(
        {"open": open_,
         "high": np.maximum(open_, close)
                 + rng.uniform(0.2, 1.0, n),
         "low": np.minimum(open_, close)
                - rng.uniform(0.2, 1.0, n),
         "close": close,
         "volume": rng.uniform(500, 3000, n)},
        index=idx)


# ------------------------------------------------------------ swings
def test_swing_points_hand_traced(traced):
    pts = st.swing_points(traced)
    got = [(p.index, p.price, p.kind) for p in pts]
    assert got == [(2, 103.0, "high"), (5, 98.5, "low"),
                   (7, 104.0, "high"), (11, 97.5, "low")]


def test_swing_labels_hand_traced(traced):
    labs = st.label_swings(st.swing_points(traced))
    assert [(s["bar"], s["label"]) for s in labs] == [
        (2, "H"), (5, "L"), (7, "HH"), (11, "LL")]


def test_swing_labels_agree_with_engine(traced):
    from goldenswing.charts.engine import _swing_labels
    from goldenswing.reader.state import SwingPoint as RSP
    pts = st.swing_points(traced)
    # The reader's SwingPoint needs confirmed_at: the bar where
    # the fractal became knowable — index + right (right=2 here).
    hi = [RSP(index=p.index, price=p.price, kind="high",
              confirmed_at=p.index + 2)
          for p in pts if p.kind == "high"]
    lo = [RSP(index=p.index, price=p.price, kind="low",
              confirmed_at=p.index + 2)
          for p in pts if p.kind == "low"]
    assert [l for _i, _p, l in _swing_labels(hi, "high")] == \
        [s["label"] for s in st.label_swings(pts)
         if s["kind"] == "high"]
    assert [l for _i, _p, l in _swing_labels(lo, "low")] == \
        [s["label"] for s in st.label_swings(pts)
         if s["kind"] == "low"]


def test_swings_never_use_unconfirmed_bars(traced):
    # A swing needs `right`=2 bars after it: the last two bars
    # can never hold a swing, whatever the prices do.
    pts = st.swing_points(traced)
    assert all(p.index <= len(traced) - 3 for p in pts)


# ---------------------------------------------------- breaks / liquidity
def test_break_hand_traced(traced):
    brs = st.detect_breaks(traced, st.swing_points(traced))
    assert [(b["kind"], b["bar"], b["level"], b["direction"])
            for b in brs] == [
        ("break", 11, 98.5, -1), ("failed_break", 12, 98.5, -1),
        ("break", 13, 103.0, 1), ("break", 13, 104.0, 1)]


def test_failed_break_is_a_second_linked_event():
    close = [100, 101, 102, 101, 100, 99, 101, 103, 102, 100,
             99, 98, 100, 105, 103]
    df = _ohlc(close, _TRACED_UP, _TRACED_DN, "2026-11-20 00:00")
    brs = st.detect_breaks(df, st.swing_points(df))
    assert ("break", 13, 104.0) in [
        (b["kind"], b["bar"], b["level"]) for b in brs]
    assert ("failed_break", 14, 104.0) in [
        (b["kind"], b["bar"], b["level"]) for b in brs]


def test_liquidity_sweeps_hand_traced(traced):
    sw = st.detect_liquidity(traced, st.swing_points(traced))
    assert [(s["bar"], s["side"], s["level"], s["wick"]) for s in sw] \
        == [(5, "above", 103.0, 0.5), (8, "above", 103.0, 0.5),
            (12, "below", 98.5, 1.0)]


# ------------------------------------------------------------------ FVGs
def test_fvgs_hand_traced(traced):
    gaps = st.detect_fvgs(traced)
    assert [(g["direction"], g["created_bar"], g["top"],
             g["bottom"], g["status"], g["mitigated_bar"])
            for g in gaps] == [
        (-1, 10, 101.5, 100.5, "mitigated", 13),
        (1, 14, 103.5, 100.5, "open", None)]


# --------------------------------------------------------- session levels
def test_session_levels_named_and_exact(sessions_frame):
    lvls = st.session_levels(sessions_frame, count=2)
    assert [(l["date"], l["weekday"], l["label_high"],
             l["label_low"]) for l in lvls] == [
        ("2026-11-19", "Thursday", "Thursday H", "Thursday L"),
        ("2026-11-20", "Friday", "Friday H", "Friday L")]
    thu = sessions_frame.iloc[0:24]
    fri = sessions_frame.iloc[24:48]
    assert lvls[0]["high"] == pytest.approx(thu["high"].max())
    assert lvls[0]["low"] == pytest.approx(thu["low"].min())
    assert lvls[1]["high"] == pytest.approx(fri["high"].max())
    assert lvls[1]["low"] == pytest.approx(fri["low"].min())
    assert lvls[0]["high_bar"] == int(thu["high"].argmax())
    assert lvls[1]["low_bar"] == 24 + int(fri["low"].argmin())


def test_session_levels_never_include_the_forming_day(sessions_frame):
    # The Monday in progress is never a "completed" level.
    lvls = st.session_levels(sessions_frame, count=5)
    assert all(l["date"] != "2026-11-23" for l in lvls)


def test_session_range_band(sessions_frame):
    band = st.session_range_band(sessions_frame)
    fri = sessions_frame.iloc[24:48]
    mon = sessions_frame.iloc[48:]
    assert band["prior"]["label"] == "Friday range"
    assert band["prior"]["top"] == pytest.approx(fri["high"].max())
    assert band["prior"]["bottom"] == pytest.approx(fri["low"].min())
    assert band["current"]["developing"] is True
    assert band["current"]["top"] == pytest.approx(mon["high"].max())


def test_expected_path_targets_the_prior_low(sessions_frame):
    ws = st.measure_workspace(sessions_frame)
    path = ws["expected_path"]
    assert path is not None
    assert path["target"] == "prior_low"
    assert path["target_label"] == "Friday L"
    fri_low = sessions_frame.iloc[24:48]["low"].min()
    assert path["target_price"] == pytest.approx(fri_low)
    assert path["marker"] == "$"
    # Points run in time order — the A-jay zigzag shape: the recent
    # swings inside the current session, then the last close, then
    # the projected target.  The close is second-to-last, never first.
    pts = path["points"]
    bars = [p["bar"] for p in pts if p["bar"] is not None]
    assert bars == sorted(bars)
    assert pts[-2]["kind"] == "close"
    assert pts[-2]["price"] == pytest.approx(
        sessions_frame["close"].iloc[-1])
    assert pts[-1]["projected"] is True
    assert pts[-1]["bar"] is None
    assert pts[-1]["price"] == pytest.approx(fri_low)
    mids = pts[:-2]
    assert len(mids) >= 1
    assert all(p["kind"] == "swing" for p in mids)
    assert all(p["projected"] is False for p in mids)


def test_expected_path_missing_without_a_session(traced):
    assert st.expected_path(traced, []) is None
    ws = st.measure_workspace(traced)
    assert ws["expected_path"] is None
    assert any(m["what"] == "expected_path"
               for m in ws["missing"])


def test_measure_workspace_is_causal(sessions_frame):
    # A prefix ending mid-Friday knows Thursday only, and no
    # swing past the confirmation lag.
    pre = sessions_frame.iloc[:30]
    ws = st.measure_workspace(pre)
    assert [l["date"] for l in ws["session_levels"]] == ["2026-11-19"]
    assert all(s["bar"] <= 27 for s in ws["swings"])
    assert ws["as_of"] == str(pre.index[-1].isoformat())
    assert ws["bars"] == 30


# ------------------------------------------------------ volatility regime
def test_volatility_expansion_and_contraction():
    n = 70
    idx = pd.date_range("2026-11-01", periods=n, freq="1h", tz="UTC")
    base = np.full(n, 4100.0)
    small = np.concatenate([np.full(60, 0.5), np.full(10, 5.0)])
    big = np.concatenate([np.full(60, 5.0), np.full(10, 0.5)])
    # The run is the trailing run of the CURRENT regime label, and
    # Wilder's ATR does not flip at once: expansion crosses 1.5x on
    # the first big bar (run 10, from bar 60), but contraction needs
    # the ATR to decay below 0.7x the median — atr_k = 1 + 9*(13/14)^k
    # reaches 7.0 only at k=6, i.e. bar 65 (run 5, not 10).
    for amp, want, run, start in (
            (small, "expansion", 10, 60),
            (big, "contraction", 5, 65)):
        df = pd.DataFrame(
            {"open": base, "high": base + amp, "low": base - amp,
             "close": base, "volume": np.full(n, 100.0)}, index=idx)
        vol = st.volatility_regime(df)
        assert vol["regime"] == want
        assert vol["run_bars"] == run
        assert vol["run_start_bar"] == start


# ------------------------------------------------------------- real data
def test_measure_workspace_on_real_xauusd_bars():
    h1 = pd.read_csv(DATA / "OANDA_XAU_USD_H1_20241001_20261012.csv",
                     parse_dates=["ts"], index_col="ts")
    ws = st.measure_workspace(h1)
    assert ws["bars"] == len(h1) > 10000
    assert ws["session_levels"], "real bars must level sessions"
    for lvl in ws["session_levels"]:
        want = pd.Timestamp(lvl["date"]).strftime("%A")
        assert lvl["weekday"] == want
        assert lvl["label_high"] == f"{want} H"
    assert ws["expected_path"] is not None
    assert ws["swings"] and ws["fvgs"]
    assert not any(m["what"] == "session_levels"
                   for m in ws["missing"])
    assert not any(m["what"] == "volume"
                   for m in ws["missing"])
    # JSON-safe: the server ships this dict verbatim.
    json.dumps(ws)


def test_mtf_state_is_per_timeframe():
    h1 = pd.read_csv(DATA / "OANDA_XAU_USD_H1_20241001_20261012.csv",
                     parse_dates=["ts"], index_col="ts")
    m5 = pd.read_csv(DATA / "OANDA_XAU_USD_M5_20260401_20261012.csv",
                     parse_dates=["ts"], index_col="ts")
    a, b = st.mtf_state(h1, tf="H1"), st.mtf_state(m5, tf="M5")
    assert a["available"] and b["available"]
    assert a["trend"] in ("up", "down", "mixed", "flat", "unknown")
    assert b["trend"] in ("up", "down", "mixed", "flat", "unknown")
    assert a["regime"] in ("expansion", "contraction", "normal")
    assert a["bars"] != b["bars"]
    assert st.mtf_state(pd.DataFrame(), tf="M1")["available"] is False


# --------------------------------------------------------- terminal page
def test_terminal_page_is_light_blue_black_with_workspace_toggles():
    page = build_terminal_page(instrument="XAUUSD")
    assert 'data-theme="light"' in page
    assert "Dark theme" in page
    assert "'#1a5fd7'" in page and "'#111111'" in page
    for t in ("ov-slevels", "ov-srange", "ov-swings", "ov-breaks",
              "ov-fvg", "ov-liq", "ov-vol", "ov-path", "ov-sess",
              "ov-nearby", "ov-fib", "ov-prem", "ov-zone"):
        assert f'id="{t}"' in page, t
    assert 'id="jump"' in page and 'id="mtf"' in page
    assert "expected path" in page.lower()
    assert "not a forecast" in page


# ------------------------------------------------------- terminal server
class _Provider:
    def __init__(self, frames):
        self._frames = frames

    def frames(self, tf):
        return self._frames.get(tf)

    def studies(self):
        return {"signal": "sig-1", "signal_time": "2026-11-20",
                "zone": None, "nearby": [], "fib": None}

    def status(self):
        return {"server_time": "2026-11-23T12:00:00+00:00",
                "market_open": True,
                "tfs": {"H1": "2026-11-23T11:00:00+00:00",
                        "M5": "2026-11-23T11:55:00+00:00"}}


@pytest.fixture(scope="module")
def server(sessions_frame):
    m5_idx = pd.date_range("2026-11-23 00:00", periods=200,
                           freq="5min", tz="UTC")
    rng = np.random.default_rng(3)
    c = 4100.0 + rng.normal(0, 1.0, 200).cumsum()
    m5 = pd.DataFrame(
        {"open": c, "high": c + 0.6, "low": c - 0.6, "close": c,
         "volume": np.full(200, 100.0)}, index=m5_idx)
    srv = TerminalServer(_Provider({"H1": sessions_frame, "M5": m5}),
                         host="127.0.0.1", port=0,
                         instrument="XAUUSD")
    url = srv.start()
    yield srv, url
    srv.stop()


def _port(url):
    return int(url.rsplit(":", 1)[1].rstrip("/"))


def _get(url, path):
    conn = http.client.HTTPConnection("127.0.0.1", _port(url),
                                     timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


def test_server_frames_carry_workspace_structure(server):
    srv, url = server
    code, body = _get(url, "/api/frames?tf=H1")
    assert code == 200
    d = json.loads(body)
    assert d["tf"] == "H1"
    assert len(d["bars"][0]) == 6, "volume rides as 6th element"
    stt = d["structure"]
    assert [l["label_high"] for l in stt["session_levels"]] == \
        ["Thursday H", "Friday H"]
    assert stt["session_range"]["prior"]["label"] == "Friday range"
    assert stt["expected_path"]["target_label"] == "Friday L"
    assert stt["swings"] and stt["fvgs"] is not None
    assert "volume" not in [m["what"] for m in stt["missing"]]


def test_server_frames_before_is_historical(server):
    srv, url = server
    # urlencode: the '+' in the timezone offset must ride as %2B,
    # or parse_qs reads it as a space and the date is malformed.
    from urllib.parse import urlencode
    q = urlencode({"tf": "H1",
                   "before": "2026-11-20T12:00:00+00:00"})
    code, body = _get(url, "/api/frames?" + q)
    assert code == 200
    d = json.loads(body)
    assert d["structure"]["as_of"] <= "2026-11-20T12:00:00+00:00"
    assert [l["date"] for l in d["structure"]["session_levels"]] \
        == ["2026-11-19"], "Friday had not completed yet"
    code, _ = _get(url, "/api/frames?tf=H1&before=not-a-date")
    assert code == 400
    code, _ = _get(url, "/api/frames?tf=BAD")
    assert code == 400
    code, _ = _get(url, "/api/frames")
    assert code == 400


def test_server_frames_structure_coords_match_bars():
    """Bar-indexed overlays must reference the bars the page got.

    The structure used to be measured on the provider's full frame
    while the page only receives the last 500 bars, so every swing,
    break, sweep, FVG and path point sat outside the drawn window
    and none of them ever rendered — the live screenshot proved it
    (only price-based levels showed).  The frame here is the real
    OANDA H1 file (11,987 bars), far past the 500-bar cap, so this
    fails unless server and page share bar coordinates.
    """
    h1 = pd.read_csv(DATA / "OANDA_XAU_USD_H1_20241001_20261012.csv",
                     parse_dates=["ts"], index_col="ts")
    assert len(h1) > 500
    srv = TerminalServer(_Provider({"H1": h1}), host="127.0.0.1",
                         port=0, instrument="XAUUSD")
    url = srv.start()
    try:
        code, body = _get(url, "/api/frames?tf=H1")
        assert code == 200
        d = json.loads(body)
        n = len(d["bars"])
        assert n == 500
        s = d["structure"]
        assert s["bars"] == n
        for sw in s["swings"]:
            assert 0 <= sw["bar"] < n
        assert s["swings"], "real data must yield swings"
        for b in s["breaks"]:
            assert 0 <= b["bar"] < n
        for q in s["liquidity"]:
            assert 0 <= q["bar"] < n
        for g in s["fvgs"]:
            assert 0 <= g["created_bar"] < n
            if g["mitigated_bar"] is not None:
                assert 0 <= g["mitigated_bar"] < n
        if s["volatility"]["run_start_bar"] is not None:
            assert 0 <= s["volatility"]["run_start_bar"] < n
        if s["expected_path"]:
            for p in s["expected_path"]["points"]:
                if p["bar"] is not None:
                    assert 0 <= p["bar"] < n
    finally:
        srv.stop()


def test_server_mtf_is_measured_per_timeframe(server):
    srv, url = server
    code, body = _get(url, "/api/mtf")
    assert code == 200
    d = json.loads(body)
    assert d["tfs"]["H1"]["available"] is True
    assert d["tfs"]["M5"]["available"] is True
    assert d["tfs"]["M1"]["available"] is False
    assert d["tfs"]["H1"]["trend"] in (
        "up", "down", "mixed", "flat", "unknown")
    assert d["tfs"]["H1"]["regime"] in (
        "expansion", "contraction", "normal")


def test_server_page_and_methods(server):
    srv, url = server
    code, body = _get(url, "/")
    assert code == 200
    assert b'data-theme="light"' in body
    assert b"Session levels" in body
    conn = http.client.HTTPConnection(
        "127.0.0.1", _port(url), timeout=10)
    conn.request("POST", "/")
    assert conn.getresponse().status == 405
    conn.close()
    code, _ = _get(url, "/nope")
    assert code == 404


# ------------------------------------------------------------ PNG / TV
def _lifecycle():
    return lifecycle_from_events(
        direction=1, entry_ref=4100.0, structural_stop=4090.0,
        targets=[4110.0, 4125.0])


def test_mpl_roles_draw_workspace_and_are_deterministic(
        sessions_frame):
    ws = st.measure_workspace(sessions_frame)
    frames = {"H1": sessions_frame, "M15": sessions_frame.iloc[-60:]}
    kw = dict(frames=frames, lifecycle=_lifecycle(), context=None,
              workspace=ws)
    a = _render_mt5_roles_mpl(**kw)
    b = _render_mt5_roles_mpl(**kw)
    assert len(a) == 3
    assert a == b, "matplotlib rung is deterministic"
    for png in a:
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        img = mpimg.imread(io.BytesIO(png))
        assert float(img.std()) > 0.02
        assert len(png) > 10_000
    plain = _render_mt5_roles_mpl(
        frames=frames, lifecycle=_lifecycle(), context=None)
    assert plain != a, "workspace overlays change the context PNG"


def test_roles_ladder_accepts_workspace(sessions_frame):
    ws = st.measure_workspace(sessions_frame)
    frames = {"H1": sessions_frame, "M15": sessions_frame.iloc[-60:]}
    pngs = render_mt5_roles(frames=frames, lifecycle=_lifecycle(),
                            context=None, workspace=ws)
    assert len(pngs) == 3
    for png in pngs:
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(png) > 10_000


def test_tv_context_cfg_draws_workspace(sessions_frame):
    ws = st.measure_workspace(sessions_frame)
    cfg = _context_cfg(sessions_frame, None, _lifecycle(),
                       workspace=ws)
    titles = [ln["title"] for ln in cfg["lines"]]
    assert any("Friday H" in t for t in titles)
    assert any("Thursday L" in t for t in titles)
    assert any(b.get("fill") == "rgba(37,99,235,0.18)"
               for b in cfg["bands"])
    assert any(ls.get("points") for ls in cfg["lineSeries"]), \
        "expected path rides as a line series"
    assert any("Friday L" in line for line in cfg["ledger"])
    # No workspace passed: the cfg measures it from the frame.
    cfg2 = _context_cfg(sessions_frame, None, _lifecycle())
    assert any("Friday H" in ln["title"]
               for ln in cfg2["lines"])
