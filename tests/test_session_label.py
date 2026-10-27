import pandas as pd

from goldenswing.live.patterns import session_label


def test_session_label_uses_dst_aware_ny_wall_clock_and_silver_bullet_is_reachable():
    # After the 2026 US DST fallback, 14:00 UTC is 09:00 EST and 15:00 UTC is 10:00 EST.
    ny_kz = session_label(pd.Timestamp("2026-11-02T14:00:00Z"))
    silver = session_label(pd.Timestamp("2026-11-02T15:00:00Z"))
    outside = session_label(pd.Timestamp("2026-11-02T15:30:00Z"))
    assert ny_kz["killzone"] == "ny_killzone"
    assert silver["killzone"] == "silver_bullet"
    assert outside["killzone"] is None
