import pandas as pd
import pytest


def make_df(rows, start="2024-01-01"):
    """rows: list of (open, high, low, close) tuples -> hourly UTC frame."""
    idx = pd.date_range(start, periods=len(rows), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1000.0] * len(rows),
        },
        index=idx,
    )


def closes_df(closes, spread=0.2, start="2024-01-01"):
    """Build a frame from closes with symmetric high/low padding."""
    rows = [(c - 0.05, c + spread, c - spread, c) for c in closes]
    return make_df(rows, start=start)


@pytest.fixture
def df_from_closes():
    return closes_df
