"""Session tagging in UTC (DST ignored — documented approximation).

ICT killzones (converted from EST): London 07:00-10:00 UTC,
New York 12:00-15:00 UTC.
"""
from __future__ import annotations

import pandas as pd


def session_of(ts: pd.Timestamp) -> tuple[str, bool]:
    """Return (session_name, in_killzone) for a UTC timestamp."""
    h = ts.hour
    if 7 <= h < 10 or 12 <= h < 15:
        kz = True
    else:
        kz = False
    if h < 7:
        name = "asia"
    elif h < 12:
        name = "london"
    elif h < 16:
        name = "overlap"      # London still open + New York open
    elif h < 21:
        name = "newyork"
    else:
        name = "off"
    return name, kz


def tag_sessions(index: pd.DatetimeIndex) -> tuple[list[str], list[bool]]:
    names: list[str] = []
    kzs: list[bool] = []
    for ts in index:
        n, k = session_of(ts)
        names.append(n)
        kzs.append(k)
    return names, kzs
