"""Paper-shadow runner: journals to a SEPARATE file, never the live journal."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_mtf12_shadow import run_shadow


def _bars(tmp_path):
    # Rising then falling H1 bars so the engine has structure to read.
    # 400 bars is enough for the 30-bar warmup + swings.
    idx = pd.date_range("2026-01-01", periods=400, freq="h", tz="UTC")
    import math
    close = [4000.0 + 30.0 * math.sin(i / 25.0) + i * 0.05 for i in range(400)]
    df = pd.DataFrame({
        "ts": idx,
        "open": close, "high": [c + 2.0 for c in close],
        "low": [c - 2.0 for c in close], "close": close,
        "volume": [100] * 400,
    })
    p = tmp_path / "bars.csv"
    df.to_csv(p, index=False)
    return p


def test_shadow_journals_separate_file_only(tmp_path):
    # Slice the real OANDA CSV around the known 2025-03-11 swing long
    # so the replay produces >=1 trade without a 35s full-file run.
    import pandas as pd
    real = pd.read_csv("data/OANDA_XAU_USD_H1_20241001_20261012.csv",
                       parse_dates=["ts"])
    # From file start so HTF resampling has full warmup for the March trade.
    sl = real[(real["ts"] >= "2024-10-01") & (real["ts"] < "2025-04-01")]
    csv_path = tmp_path / "slice.csv"
    sl.to_csv(csv_path, index=False)
    shadow = tmp_path / "shadow.jsonl"
    live = tmp_path / "live.jsonl"
    live.write_text('{"type": "trade", "trade_id": "t-1"}\n')
    summary = run_shadow(csv_path=csv_path, preset="swing",
                         balance=12.0, journal_path=shadow)
    assert summary["balance"] == 12.0
    assert summary["journal_path"] == str(shadow)
    assert summary["trades"] >= 1
    # Live journal untouched.
    assert live.read_text() == '{"type": "trade", "trade_id": "t-1"}\n'
    # Shadow journal: one record per trade, $12 fields present.
    lines = [json.loads(x) for x in shadow.read_text().splitlines() if x.strip()]
    assert len(lines) == summary["trades"]
    for rec in lines:
        assert rec["type"] == "mtf12_shadow_paper"
        assert rec["start_balance"] == 12.0
        assert rec["risk_money"] == 0.12
        assert "oanda_1unit_feasible" in rec


def test_shadow_unknown_preset_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        run_shadow(csv_path=_bars(tmp_path), preset="nope",
                   balance=12.0, journal_path=tmp_path / "s.jsonl")
