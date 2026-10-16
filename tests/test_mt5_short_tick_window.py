"""Crash + short-tick-window guard for scripts/backtest_mt5_full.py.

Owner evidence (2026-10-16): after pulling 12ca475 the run
printed frames (MN1 1 row, D1 3 rows, M1 1394 rows, 74,579
ticks) then crashed:
  AttributeError: 'str' object has no attribute 'get'
at the provenance print loop — provenance["tick_coverage"] is
a STRING, not a dict. And the 74,579 ticks were ~1 DAY, not the
~10 months of tick history assumed — a 1-day run must never be
reported as a 3y backtest.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "backtest_mt5_full",
    Path(__file__).resolve().parent.parent
    / "scripts" / "backtest_mt5_full.py")
_mod = importlib.util.module_from_spec(SPEC)
sys.modules["backtest_mt5_full"] = _mod
SPEC.loader.exec_module(_mod)


def _provenance(tick_start="2026-10-15", tick_end="2026-10-16"):
    return {
        "MN1": {"source": "built_from_ticks", "rows": 1,
                "built_from_ticks": True},
        "D1": {"source": "built_from_ticks", "rows": 3,
               "built_from_ticks": True},
        "M1": {"source": "built_from_ticks", "rows": 1394,
               "built_from_ticks": True},
        "window": {"start": tick_start, "end": tick_end},
        "tick_window": {"start": tick_start, "end": tick_end,
                        "rows": 74579, "source": "ticks"},
        # The crash: a STRING in the provenance mapping.
        "tick_coverage": "ticks available 2026-10-15..2026-10-16",
        "resampling": "NONE",
        "built_from_ticks": ["MN1", "D1", "M1"],
    }


def test_print_frames_survives_string_provenance(capsys):
    # Exact owner crash shape: tick_coverage is a str. Must not raise.
    _mod._print_frames(_provenance())
    out = capsys.readouterr().out
    assert "MN1" in out and "1394" in out
    # tick_window dict must NOT print as a fake TF row.
    assert "tick_window" not in out
    assert "tick_coverage" not in out


def test_check_tick_window_blocks_short_window():
    prov = _provenance()
    err = _mod._check_tick_window(prov, {"tick_chunks": []},
                                 allow_short=False)
    assert err is not None
    assert "BLOCKED" in err
    assert "1.0 days" in err or "1 day" in err
    assert "--allow-short-tick-window" in err


def test_check_tick_window_includes_chunk_log():
    prov = _provenance()
    manifest = {"tick_chunks": [
        {"chunk_start": "2026-01-01", "chunk_end": "2026-04-01",
         "rows": 0, "mode": "empty"},
        {"chunk_start": "2026-10-15", "chunk_end": "2026-10-16",
         "rows": 74579, "mode": "tz_aware"}]}
    err = _mod._check_tick_window(prov, manifest, allow_short=False)
    assert "74579" in err
    assert "2026-01-01" in err


def test_check_tick_window_allows_long_and_explicit_short():
    long_prov = _provenance(tick_start="2026-01-01",
                            tick_end="2026-10-16")
    assert _mod._check_tick_window(long_prov, {}, False) is None
    short_prov = _provenance()
    assert _mod._check_tick_window(short_prov, {}, True) is None


def test_check_tick_window_skips_native_only():
    assert _mod._check_tick_window({"window": {}}, {}, False) is None
