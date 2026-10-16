"""Auto-fetch from MT5: cache, fail-closed, no fallback, no resample."""
import pathlib

import numpy as np
import pandas as pd
import pytest

from goldenswing.backtest import mt5_fetch as mf


class _FakeMT5:
    """Minimal MetaTrader5 stand-in: native bars + ticks per TF."""

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

    def __init__(self, *, fail_init=False):
        self.fail_init = fail_init
        self.calls = {"rates": 0, "ticks": 0}

    def initialize(self):
        return not self.fail_init

    def shutdown(self):
        pass

    def symbol_select(self, symbol, enable):
        return True

    def copy_rates_range(self, symbol, tf, start, end):
        self.calls["rates"] += 1
        times = pd.date_range(start, end, freq="h", tz="UTC")
        if len(times) == 0:
            times = pd.DatetimeIndex([pd.Timestamp(start, tz="UTC")])
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
        self.calls["ticks"] += 1
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


def _window():
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-03", tz="UTC")
    return start, end


def test_fetch_writes_all_tfs_and_ticks(tmp_path):
    start, end = _window()
    fake = _FakeMT5()
    manifest = mf.fetch_mt5_native("XAUUSD", start, end, tmp_path,
                                   mt5_module=fake)
    assert manifest["source"] == "mt5_terminal_live_fetch"
    assert manifest["resampling"] == "NONE"
    for tf in mf.REQUIRED_TFS:
        assert (tmp_path / f"MT5_XAUUSD_{tf}_native.csv").exists()
    assert (tmp_path / "MT5_XAUUSD_ticks.csv").exists()
    assert fake.calls["rates"] == len(mf.REQUIRED_TFS)
    assert manifest["ticks"]["rows"] > 0


def test_ensure_uses_cache_second_run(tmp_path):
    start, end = _window()
    fake = _FakeMT5()
    first = mf.ensure_mt5_native("XAUUSD", start, end, tmp_path,
                                 mt5_module=fake)
    assert first["source"] == "mt5_terminal_live_fetch"
    calls_after_first = dict(fake.calls)
    second = mf.ensure_mt5_native("XAUUSD", start, end, tmp_path,
                                  mt5_module=fake)
    assert second["source"] == "mt5_cache"
    assert fake.calls == calls_after_first  # no terminal contact


def test_initialize_failure_fails_with_clear_message(tmp_path):
    start, end = _window()
    fake = _FakeMT5(fail_init=True)
    with pytest.raises(mf.MT5Unavailable) as excinfo:
        mf.fetch_mt5_native("XAUUSD", start, end, tmp_path,
                            mt5_module=fake)
    assert "MT5 terminal not running" in str(excinfo.value)


def test_missing_package_fails_closed(tmp_path, monkeypatch):
    import importlib as _il
    start, end = _window()
    monkeypatch.setattr(
        _il, "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("no mt5")))
    with pytest.raises(mf.MT5Unavailable):
        mf.fetch_mt5_native("XAUUSD", start, end, tmp_path)


def test_no_resample_call_in_fetch_module():
    src = pathlib.Path(mf.__file__).read_text()
    assert ".resample(" not in src
    # No OANDA data file / import — the word may appear in prose
    # ("no OANDA fallback"), but no OANDA source is ever read.
    assert "OANDA_XAU" not in src
    assert "oanda" not in src.lower().split("no oanda fallback")[0]


def test_cache_covers_rejects_short_window(tmp_path):
    start, end = _window()
    fake = _FakeMT5()
    mf.fetch_mt5_native("XAUUSD", start, end, tmp_path, mt5_module=fake)
    # Ask for a WIDER window than the cache holds -> must not claim cover.
    wider_end = pd.Timestamp("2026-02-01", tz="UTC")
    assert mf._cache_covers(tmp_path, "XAUUSD", start, wider_end) is False


def test_default_symbol_is_xauusdm():
    # Owner's terminal is Exness: gold is XAUUSDm, plain XAUUSD
    # does not exist there.
    assert mf.DEFAULT_SYMBOL == "XAUUSDm"


def test_fetch_filenames_carry_symbol(tmp_path):
    # Cache filenames must use the symbol: MT5_XAUUSDm_* for the
    # default Exness symbol, and a different symbol gets its own
    # files (caches never mix across broker symbols).
    start, end = _window()
    fake = _FakeMT5()
    mf.fetch_mt5_native("XAUUSDm", start, end, tmp_path,
                        mt5_module=fake)
    for tf in mf.REQUIRED_TFS:
        assert (tmp_path / f"MT5_XAUUSDm_{tf}_native.csv").exists()
    assert (tmp_path / "MT5_XAUUSDm_ticks.csv").exists()
    assert mf._cache_covers(tmp_path, "XAUUSDm", start, end) is True
    # Same bars do NOT satisfy a different symbol's cache.
    assert mf._cache_covers(tmp_path, "XAUUSD", start, end) is False

"""Per-chunk tick log + naive retry + stale 1-day cache rejection.

Owner evidence (2026-10-16): the backtest ran on ~74,579 ticks
(~1 day) with no visibility into WHY the 3y tick request came
back with one day. The fetch must log per-chunk rows so a
depth-limited broker (only the last chunk has rows) is
distinguishable from a fetch failure (all chunks empty).
"""


def _empty_ticks():
    return np.zeros(0, dtype=[("time", "i8"), ("bid", "f8"),
                              ("ask", "f8"), ("last", "f8"),
                              ("volume", "f8"), ("flags", "i8")])


class _RecentOnlyTicks(_FakeMT5):
    """Broker/terminal returns ticks only for the recent chunk."""

    def copy_ticks_range(self, symbol, start, end, flags):
        self.calls["ticks"] += 1
        _ts = pd.Timestamp(start)
        if _ts.tzinfo is None:
            _ts = _ts.tz_localize("UTC")
        else:
            _ts = _ts.tz_convert("UTC")
        if _ts < pd.Timestamp("2026-01-07", tz="UTC"):
            return _empty_ticks()
        return super().copy_ticks_range(symbol, start, end, flags)


def test_fetch_records_per_chunk_tick_log(tmp_path):
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-10", tz="UTC")
    fake = _RecentOnlyTicks()
    manifest = mf.fetch_mt5_native("XAUUSDm", start, end, tmp_path,
                                   mt5_module=fake, tick_chunk_days=3)
    chunks = manifest["tick_chunks"]
    assert len(chunks) == 3  # 01-04, 04-07, 07-10
    assert chunks[0]["rows"] == 0
    assert chunks[1]["rows"] == 0
    assert chunks[2]["rows"] > 0
    assert manifest["tick_fetch"]["chunks_requested"] == 3
    assert manifest["tick_fetch"]["chunks_with_data"] == 1
    # Tick window starts where the broker actually had ticks.
    assert manifest["tick_window"]["start"].startswith("2026-01-07")


class _TzAwareEmptyTicks(_FakeMT5):
    """Terminal build that returns nothing for tz-aware datetimes."""

    def copy_ticks_range(self, symbol, start, end, flags):
        self.calls["ticks"] += 1
        if getattr(start, "tzinfo", None) is not None:
            return _empty_ticks()
        return super().copy_ticks_range(symbol, start, end, flags)


def test_copy_ticks_naive_retry(tmp_path):
    fake = _TzAwareEmptyTicks()
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-02", tz="UTC")
    ticks, mode = mf._copy_ticks_range(fake, "XAUUSDm",
                                      start.to_pydatetime(),
                                      end.to_pydatetime())
    assert mode == "naive_server_time"
    assert ticks is not None and len(ticks) > 0


def test_ensure_does_not_reuse_one_day_stale_ticks(tmp_path):
    # Owner's exact trap: a 1-day ticks cache + TF files covering
    # that 1 day. ensure() must NOT treat it as "covers tick
    # window" for a 3y request — the overlap is 1 day < 30d, so
    # it must re-fetch from the terminal.
    tick_idx = pd.date_range("2026-10-15", "2026-10-16", freq="min",
                             tz="UTC")
    pd.DataFrame({"time": tick_idx, "bid": 4000.0, "ask": 4000.2,
                  "last": 4000.1, "volume": 1.0,
                  "flags": 0}).to_csv(
        tmp_path / "MT5_XAUUSDm_ticks.csv", index=False)
    for tf in mf.REQUIRED_TFS:
        idx = pd.date_range("2026-10-15", "2026-10-16", freq="h",
                            tz="UTC")
        pd.DataFrame({"time": idx, "open": 4000.0, "high": 4001.0,
                      "low": 3999.0, "close": 4000.5,
                      "tick_volume": 10, "spread": 20,
                      "real_volume": 0}).to_csv(
            tmp_path / f"MT5_XAUUSDm_{tf}_native.csv", index=False)
    start = pd.Timestamp("2023-10-17", tz="UTC")
    end = pd.Timestamp("2026-10-16", tz="UTC")
    fake = _FakeMT5()
    result = mf.ensure_mt5_native("XAUUSDm", start, end, tmp_path,
                                  mt5_module=fake)
    assert result["source"] == "mt5_terminal_live_fetch"
    assert fake.calls["ticks"] > 0
