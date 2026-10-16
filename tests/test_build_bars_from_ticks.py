"""Tick-aggregation fallback: sub-hourly bars built from MT5 ticks.

Owner screenshot (2026-10-16): backtest BLOCKED because the terminal
had no cached M1 bars for 3y. M1 bars are redundant when ticks are
present — ticks are the source truth, M1 is just their aggregation.
Requiring cached M1 bars was the bug, not missing data.
"""
import numpy as np
import pandas as pd
import pytest

from goldenswing.backtest import mt5_fetch as mf
from goldenswing.backtest.mt5_native_data import (
    MissingNativeData, build_bars_from_ticks, load_native_frames)


def _tick_df():
    # 3 minutes of ticks: min 0 -> 3 ticks, min 1 -> 2 ticks, min 2 -> 1.
    base = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    rows = [
        (0, 100.0, 100.2, 2.0),
        (20, 101.0, 101.2, 2.0),
        (50, 99.0, 99.2, 2.0),
        (70, 102.0, 102.2, 2.0),
        (110, 103.0, 103.2, 2.0),
        (130, 105.0, 105.2, 2.0),
    ]
    return pd.DataFrame({
        "time": [base + pd.Timedelta(seconds=s) for s, _, _, _ in rows],
        "bid": [b for _, b, _, _ in rows],
        "ask": [a for _, _, a, _ in rows],
        "volume": [v for _, _, _, v in rows],
    })


def test_build_m1_matches_hand_computed_ohlc():
    bars = build_bars_from_ticks(_tick_df(), "M1")
    assert len(bars) == 3
    # Minute 0: open 100 (first bid), high 101, low 99, close 99 (last).
    assert bars["open"].iloc[0] == 100.0
    assert bars["high"].iloc[0] == 101.0
    assert bars["low"].iloc[0] == 99.0
    assert bars["close"].iloc[0] == 99.0
    assert bars["volume"].iloc[0] == 6.0  # 3 ticks x 2.0
    assert bars["spread"].iloc[0] == pytest.approx(0.2)
    # Minute 1: open 102, high 103, low 102, close 103.
    assert bars["open"].iloc[1] == 102.0
    assert bars["close"].iloc[1] == 103.0
    # Minute 2: single tick -> OHLC all equal.
    assert bars["open"].iloc[2] == bars["close"].iloc[2] == 105.0


def test_build_m5_aggregates_across_minutes():
    bars = build_bars_from_ticks(_tick_df(), "M5")
    assert len(bars) == 1  # all ticks inside one 5-min bucket
    assert bars["open"].iloc[0] == 100.0
    assert bars["high"].iloc[0] == 105.0
    assert bars["low"].iloc[0] == 99.0
    assert bars["close"].iloc[0] == 105.0
    assert bars["volume"].iloc[0] == 12.0


def test_build_htf_from_ticks():
    # HTF bars ARE buildable from ticks now (2026-10-16 fix): ticks
    # are the source truth and contain the session boundaries.
    # H1: ticks across two hours -> two hourly bars.
    bars = build_bars_from_ticks(_tick_df(), "H1")
    assert len(bars) == 1  # all 6 ticks inside one UTC hour
    assert bars["open"].iloc[0] == 100.0
    assert bars["high"].iloc[0] == 105.0
    assert bars["low"].iloc[0] == 99.0
    assert bars["close"].iloc[0] == 105.0
    # D1: same ticks -> one daily bar.
    bars_d1 = build_bars_from_ticks(_tick_df(), "D1")
    assert len(bars_d1) == 1
    assert bars_d1["open"].iloc[0] == 100.0
    assert bars_d1["close"].iloc[0] == 105.0
    # H4 / W1 / MN1: one bucket for this tiny tick set.
    for tf in ("H4", "W1", "MN1"):
        bars_tf = build_bars_from_ticks(_tick_df(), tf)
        assert len(bars_tf) == 1
        assert bars_tf["open"].iloc[0] == 100.0


def test_build_unknown_tf_still_rejected():
    with pytest.raises(ValueError):
        build_bars_from_ticks(_tick_df(), "M99")


def test_build_h1_buckets_split_on_hour_boundary():
    base = pd.Timestamp("2026-01-01 00:55", tz="UTC")
    df = pd.DataFrame({
        "time": [base, base + pd.Timedelta(minutes=5),
                 base + pd.Timedelta(minutes=10)],
        "bid": [100.0, 101.0, 102.0],
        "ask": [100.2, 101.2, 102.2],
        "volume": [1.0, 1.0, 1.0]})
    bars = build_bars_from_ticks(df, "H1")
    assert len(bars) == 2
    assert bars.index[0] == pd.Timestamp("2026-01-01 00:00", tz="UTC")
    assert bars.index[1] == pd.Timestamp("2026-01-01 01:00", tz="UTC")
    assert bars["close"].iloc[0] == 100.0
    assert bars["open"].iloc[1] == 101.0
    assert bars["close"].iloc[1] == 102.0


def test_build_d1_respects_server_offset():
    # Server at UTC+2: server midnight = 22:00 UTC. A tick at
    # 23:00 UTC (01:00 server) belongs to the server day that
    # started 22:00 UTC, NOT the UTC day starting 00:00.
    base = pd.Timestamp("2026-01-01 23:00", tz="UTC")
    df = pd.DataFrame({
        "time": [base, base + pd.Timedelta(hours=2)],
        "bid": [100.0, 101.0],
        "ask": [100.2, 101.2],
        "volume": [1.0, 1.0]})
    bars_utc = build_bars_from_ticks(df, "D1",
                                     server_utc_offset_hours=0.0)
    assert len(bars_utc) == 2  # UTC days split at 00:00
    bars_srv = build_bars_from_ticks(df, "D1",
                                     server_utc_offset_hours=2.0)
    assert len(bars_srv) == 1  # same server day
    assert bars_srv.index[0] == pd.Timestamp(
        "2026-01-01 22:00", tz="UTC")


def _native_tf_file(tmp_path, tf, idx):
    df = pd.DataFrame(
        {"ts": idx, "open": 100.0, "high": 110.0, "low": 90.0,
         "close": 100.0, "volume": 10.0})
    df.to_csv(tmp_path / f"MT5_XAUUSDm_{tf}_native.csv", index=False)


def _ticks_file(tmp_path, start, end):
    idx = pd.date_range(start, end, freq="min", tz="UTC")
    df = pd.DataFrame({
        "time": idx,
        "bid": 100.0 + np.arange(len(idx)) * 0.01,
        "ask": 100.2 + np.arange(len(idx)) * 0.01,
        "last": 100.1 + np.arange(len(idx)) * 0.01,
        "volume": 1.0, "flags": 0})
    df.to_csv(tmp_path / "MT5_XAUUSDm_ticks.csv", index=False)
    return idx


def test_loader_builds_missing_m1_from_ticks(tmp_path):
    # Every TF native EXCEPT M1, plus a ticks file -> loads fine,
    # M1 flagged built_from_ticks, M1 bars match tick aggregation.
    idx = pd.date_range("2024-10-01", "2024-11-10", periods=40, tz="UTC")
    for tf in ("M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"):
        _native_tf_file(tmp_path, tf, idx)
    _ticks_file(tmp_path, "2024-10-01", "2024-11-10")
    frames, prov = load_native_frames(tmp_path)
    assert "M1" in frames
    assert prov["M1"]["built_from_ticks"] is True
    assert "built_from_ticks" in prov["M1"]["source"]
    assert "M1" in prov["built_from_ticks"]
    assert prov["resampling"].startswith("NONE")
    # M1 bars came from ticks (per-minute), not from H1 bars.
    assert len(frames["M1"]) > len(frames["H1"])


def test_loader_missing_m1_and_missing_ticks_still_blocks(tmp_path):
    # No M1 file AND no ticks file -> still MissingNativeData naming M1.
    idx = pd.date_range("2024-10-01", "2024-11-10", periods=40, tz="UTC")
    for tf in ("M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"):
        _native_tf_file(tmp_path, tf, idx)
    with pytest.raises(MissingNativeData) as exc:
        load_native_frames(tmp_path)
    assert "M1" in exc.value.missing


def test_loader_builds_missing_htf_from_ticks(tmp_path):
    # D1 missing + ticks present -> D1 is built from ticks now
    # (2026-10-16 fix): ticks contain the session boundaries, so
    # blocking on missing native HTF history was the bug.
    idx = pd.date_range("2024-10-01", "2024-11-10", periods=40, tz="UTC")
    for tf in ("M1", "M5", "M15", "M30", "H1", "H4", "W1", "MN1"):
        _native_tf_file(tmp_path, tf, idx)
    _ticks_file(tmp_path, "2024-10-01", "2024-11-10")
    frames, prov = load_native_frames(tmp_path)
    assert "D1" in frames
    assert prov["D1"]["built_from_ticks"] is True
    assert "D1" in prov["built_from_ticks"]


def test_loader_missing_htf_and_missing_ticks_still_blocks(tmp_path):
    # D1 missing AND no ticks file -> still MissingNativeData.
    idx = pd.date_range("2024-10-01", "2024-11-10", periods=40, tz="UTC")
    for tf in ("M1", "M5", "M15", "M30", "H1", "H4", "W1", "MN1"):
        _native_tf_file(tmp_path, tf, idx)
    with pytest.raises(MissingNativeData) as exc:
        load_native_frames(tmp_path)
    assert "D1" in exc.value.missing


class _FakeMT5MissingM1:
    """Fake MT5: native bars for every TF except M1; ticks present."""

    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388
    TIMEFRAME_D1 = 16408
    TIMEFRAME_W1 = 32769
    TIMEFRAME_MN1 = 49153
    COPY_TICKS_ALL = 3

    def __init__(self, missing_tfs=()):
        self.missing_tfs = set(missing_tfs)

    def initialize(self):
        return True

    def shutdown(self):
        pass

    def symbol_select(self, symbol, enable):
        return True

    def copy_rates_range(self, symbol, tf, start, end):
        tf_name = {v: k for k, v in {
            "M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 16385,
            "H4": 16388, "D1": 16408, "W1": 32769,
            "MN1": 49153}.items()}.get(tf)
        if tf_name in self.missing_tfs:
            return None
        times = pd.date_range(start, end, freq="h", tz="UTC")
        n = len(times)
        base = 4000.0 + np.arange(n) * 0.1
        arr = np.zeros(n, dtype=[("time", "i8"), ("open", "f8"),
                                 ("high", "f8"), ("low", "f8"),
                                 ("close", "f8"), ("tick_volume", "i8"),
                                 ("spread", "i8"), ("real_volume", "i8")])
        arr["time"] = (times.astype("int64") // 10 ** 9).to_numpy()
        arr["open"] = base
        arr["high"] = base + 1.0
        arr["low"] = base - 1.0
        arr["close"] = base + 0.5
        arr["tick_volume"] = 10
        arr["spread"] = 20
        arr["real_volume"] = 0
        return arr

    def copy_ticks_range(self, symbol, start, end, flags):
        times = pd.date_range(start, end, freq="min", tz="UTC")
        n = len(times)
        arr = np.zeros(n, dtype=[("time", "i8"), ("bid", "f8"),
                                 ("ask", "f8"), ("last", "f8"),
                                 ("volume", "f8"), ("flags", "i8")])
        arr["time"] = (times.astype("int64") // 10 ** 9).to_numpy()
        arr["bid"] = 4000.0
        arr["ask"] = 4000.2
        arr["last"] = 4000.1
        arr["volume"] = 1.0
        arr["flags"] = 0
        return arr


def test_fetch_builds_missing_m1_from_ticks(tmp_path):
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-03", tz="UTC")
    fake = _FakeMT5MissingM1(missing_tfs=("M1",))
    manifest = mf.fetch_mt5_native("XAUUSDm", start, end, tmp_path,
                                   mt5_module=fake)
    # No block: M1 bars aggregated from ticks, cached as from_ticks.
    derived = tmp_path / "MT5_XAUUSDm_M1_from_ticks.csv"
    assert derived.exists()
    assert manifest["tfs"]["M1"]["source"] == "built_from_ticks"
    assert manifest["built_from_ticks"] == ["M1"]
    # Cache covers the window (derived file counts as M1 coverage).
    assert mf._cache_covers(tmp_path, "XAUUSDm", start, end) is True
    # Loader reads the derived file and flags it.
    frames, prov = load_native_frames(tmp_path, symbol="XAUUSDm")
    assert prov["M1"]["built_from_ticks"] is True


def test_fetch_builds_missing_h1_from_ticks(tmp_path):
    # H1 native missing + ticks present -> H1 built from ticks,
    # no block (2026-10-16 fix: manual download did not fix it, H1 still empty).
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-03", tz="UTC")
    fake = _FakeMT5MissingM1(missing_tfs=("H1",))
    manifest = mf.fetch_mt5_native("XAUUSDm", start, end, tmp_path,
                                   mt5_module=fake)
    derived = tmp_path / "MT5_XAUUSDm_H1_from_ticks.csv"
    assert derived.exists()
    assert manifest["tfs"]["H1"]["source"] == "built_from_ticks"
    assert "H1" in manifest["built_from_ticks"]
    assert mf._cache_covers(tmp_path, "XAUUSDm", start, end) is True


def test_fetch_missing_h1_and_no_ticks_blocks_with_diagnostics(tmp_path):
    # H1 native missing AND no ticks -> blocks, and the error
    # carries diagnostics (not a bare "download history" message).
    class _NoTicks(_FakeMT5MissingM1):
        def copy_ticks_range(self, symbol, start, end, flags):
            return None

        def terminal_info(self):
            class _TI:
                connected = True
                trade_allowed = True
                server = "Exness-MT5Trial10"
                build = 5000
                name = "MetaTrader 5"
                company = "Exness"
                path = "C:/MT5"
            return _TI()

        def symbol_info(self, symbol):
            class _SI:
                visible = True
                select = True
                trade_mode = 0
                digits = 2
                point = 0.01
                description = "Gold vs US Dollar"
            return _SI()

        def last_error(self):
            return (1, "no data")

    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-03", tz="UTC")
    with pytest.raises(mf.MT5Unavailable) as exc:
        mf.fetch_mt5_native("XAUUSDm", start, end, tmp_path,
                            mt5_module=_NoTicks(missing_tfs=("H1",)))
    msg = str(exc.value)
    assert "diagnostics" in msg
    assert "Exness-MT5Trial10" in msg
    assert "probe_H1_last_30d_rows" in msg


def test_fetch_no_ticks_still_blocks(tmp_path):
    class _NoTicks(_FakeMT5MissingM1):
        def copy_ticks_range(self, symbol, start, end, flags):
            return None

    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-03", tz="UTC")
    with pytest.raises(mf.MT5Unavailable) as exc:
        mf.fetch_mt5_native("XAUUSDm", start, end, tmp_path,
                            mt5_module=_NoTicks())
    assert "no ticks" in str(exc.value).lower()


def test_no_resample_in_tick_aggregation_code():
    import inspect

    import goldenswing.backtest.mt5_fetch as fetch_mod
    import goldenswing.backtest.mt5_native_data as native_mod

    for mod in (fetch_mod, native_mod):
        assert ".resample(" not in inspect.getsource(mod)
    assert ".resample(" not in inspect.getsource(
        build_bars_from_ticks)


def test_diagnose_reports_terminal_symbol_probe(tmp_path):
    # Fake MT5 that answers diagnostics + returns rows only for the
    # 30-day probe window (simulating a depth-limited broker).
    import datetime as _dt

    class _DiagMT5(_FakeMT5MissingM1):
        def terminal_info(self):
            class _TI:
                connected = True
                trade_allowed = True
                server = "Exness-MT5Trial10"
                build = 5000
                name = "MetaTrader 5"
                company = "Exness"
                path = "C:/MT5"
            return _TI()

        def symbol_info(self, symbol):
            class _SI:
                visible = True
                select = True
                trade_mode = 0
                digits = 2
                point = 0.01
                description = "Gold"
            return _SI()

        def last_error(self):
            return (0, "ok")

        def copy_rates_range(self, symbol, tf, start, end):
            # Probe window (last 30d) returns rows; 3y window empty.
            span = (pd.Timestamp(end) - pd.Timestamp(start)).days
            if span <= 31:
                return super().copy_rates_range(symbol, tf, start, end)
            return None

    fake = _DiagMT5()
    start = pd.Timestamp("2023-10-17", tz="UTC")
    end = pd.Timestamp("2026-10-16", tz="UTC")
    diag = mf.diagnose_mt5(fake, "XAUUSDm", start, end)
    assert diag["terminal"]["connected"] == "True"
    assert diag["terminal"]["server"] == "Exness-MT5Trial10"
    assert diag["symbol_info"]["visible"] == "True"
    assert diag["probe_h1_last_30d"]["rows"] > 0  # probe finds data
    assert "diagnostics" in mf._diag_summary(diag)


def test_copy_rates_naive_retry(tmp_path):
    # Terminal that returns nothing for tz-aware datetimes but rows
    # for naive (server-time) datetimes -> retry rescues the fetch.
    import datetime as _dt
    import numpy as np

    class _NaiveOnly(_FakeMT5MissingM1):
        def copy_rates_range(self, symbol, tf, start, end):
            if getattr(start, "tzinfo", None) is not None:
                return None
            return super().copy_rates_range(symbol, tf, start, end)

    fake = _NaiveOnly()
    rates, mode = mf._copy_rates(
        fake, "XAUUSDm", fake.TIMEFRAME_H1,
        _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc),
        _dt.datetime(2026, 1, 3, tzinfo=_dt.timezone.utc))
    assert mode == "naive_server_time"
    assert rates is not None and len(rates) > 0


def test_fetch_records_datetime_mode_and_requested_range(tmp_path):
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-03", tz="UTC")
    fake = _FakeMT5MissingM1()
    manifest = mf.fetch_mt5_native("XAUUSDm", start, end, tmp_path,
                                   mt5_module=fake)
    assert manifest["datetime_modes"]["H1"] == "tz_aware"
    assert manifest["tfs"]["H1"]["requested"] == "2026-01-01..2026-01-03"
    assert "returned" in manifest["tfs"]["H1"]


def test_diagnose_includes_terminal_path_account_and_from_pos(tmp_path):
    # Owner evidence (2026-10-16): chart shows H1 back to Jun 2023,
    # yet copy_rates_range returned 0. Diagnostics must show WHICH
    # terminal Python connected to (path / data path / account)
    # and whether from_pos fetch works at all.
    class _FullDiag(_FakeMT5MissingM1):
        def terminal_info(self):
            class _TI:
                connected = True
                trade_allowed = True
                server = "Exness-MT5Trial10"
                build = 5000
                name = "MetaTrader 5"
                company = "Exness"
                path = r"C:\Program Files\Exness MT5\terminal64.exe"
                data_path = r"C:\Users\Admin\AppData\Roaming\MetaQuotes\Terminal\ABC"
            return _TI()

        def account_info(self):
            class _AI:
                login = 81800936
                server = "Exness-MT5Trial10"
                name = "Demo"
                currency = "USD"
                company = "Exness"
            return _AI()

        def copy_rates_range(self, symbol, tf, start, end):
            return None  # range returns nothing (the reported bug)

        def copy_rates_from_pos(self, symbol, tf, start_pos, count):
            times = pd.date_range("2026-10-01", periods=count,
                                  freq="h", tz="UTC")
            arr = np.zeros(count, dtype=[("time", "i8"), ("open", "f8"),
                                         ("high", "f8"), ("low", "f8"),
                                         ("close", "f8"),
                                         ("tick_volume", "i8"),
                                         ("spread", "i8"),
                                         ("real_volume", "i8")])
            arr["time"] = (times.astype("int64") // 10 ** 9).to_numpy()
            arr["open"] = 4000.0
            arr["high"] = 4001.0
            arr["low"] = 3999.0
            arr["close"] = 4000.5
            return arr

    fake = _FullDiag()
    start = pd.Timestamp("2023-10-17", tz="UTC")
    end = pd.Timestamp("2026-10-16", tz="UTC")
    diag = mf.diagnose_mt5(fake, "XAUUSDm", start, end)
    assert diag["terminal"]["path"] == (
        r"C:\Program Files\Exness MT5\terminal64.exe")
    assert diag["terminal"]["data_path"].endswith("ABC")
    assert diag["account"]["login"] == "81800936"
    assert diag["account"]["server"] == "Exness-MT5Trial10"
    # Range probe empty, from_pos probe finds bars -> date-param /
    # terminal-window bug, NOT a history limit.
    assert diag["probe_h1_last_30d"]["rows"] == 0
    assert diag["probe_from_pos_h1"]["rows"] == 100
    summary = mf._diag_summary(diag)
    assert "terminal_path=" in summary
    assert "account_login=81800936" in summary
    assert "probe_from_pos_H1_rows=100" in summary


def test_initialize_with_path_passes_path(tmp_path):
    # --terminal-path / MT5_TERMINAL_PATH must reach initialize(path=...).
    seen: dict = {}

    class _PathFake(_FakeMT5MissingM1):
        def initialize(self, path=None, **kwargs):
            seen["path"] = path
            return True

    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-03", tz="UTC")
    mf.fetch_mt5_native("XAUUSDm", start, end, tmp_path,
                        mt5_module=_PathFake(),
                        terminal_path=r"C:\MT5\terminal64.exe")
    assert seen["path"] == r"C:\MT5\terminal64.exe"


def test_terminal_path_env_var(tmp_path, monkeypatch):
    seen: dict = {}

    class _PathFake(_FakeMT5MissingM1):
        def initialize(self, path=None, **kwargs):
            seen["path"] = path
            return True

    monkeypatch.setenv("MT5_TERMINAL_PATH", r"C:\MT5\terminal64.exe")
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-03", tz="UTC")
    manifest = mf.fetch_mt5_native("XAUUSDm", start, end, tmp_path,
                                   mt5_module=_PathFake())
    assert seen["path"] == r"C:\MT5\terminal64.exe"
    assert manifest["terminal_path"] == r"C:\MT5\terminal64.exe"
