"""AlertService — signal -> chart -> alert -> senders, with dedup.

Dedup rule: the same (instrument, direction) does not re-alert within
``cooldown_bars`` bars of the last alert, and the same alert_id never
alerts twice.  This is the spam guard that stands between a detector
that can fire on consecutive bars and a phone that would buzz all day.
"""
from __future__ import annotations

from .. import charts
from .model import Alert, build_alert


class AlertService:
    def __init__(self, senders: list, cooldown_bars: int = 24,
                 chart_style: str = "signal",
                 chart_renderer=None):
        if chart_style not in charts.STYLES:
            raise ValueError(f"unknown chart style {chart_style!r}")
        self.senders = list(senders)
        self.cooldown_bars = int(cooldown_bars)
        self.chart_style = chart_style
        # injectable for tests; default is the real chart engine
        self._render = chart_renderer or charts.render
        self._last_alert_bar: dict[tuple[str, int], int] = {}
        self._seen_ids: set[str] = set()

    def is_duplicate(self, instrument: str, direction: int,
                     bar_index: int, alert_id: str) -> bool:
        if alert_id in self._seen_ids:
            return True
        last = self._last_alert_bar.get((instrument, direction))
        return last is not None and 0 <= bar_index - last < self.cooldown_bars

    def process(self, df, state, signal, *, instrument: str = "XAUUSD",
                mode: str = "swing", features=None,
                forecast=None, chart_style: str | None = None) -> Alert | None:
        """Build and dispatch an alert for ``signal`` at ``state``.

        Returns None when the dedup/cooldown suppresses it.  The
        cooldown is checked before any rendering, so a suppressed
        signal costs nothing.
        """
        from .model import make_alert_id
        alert_id = make_alert_id(instrument, signal.index, signal.direction)
        if self.is_duplicate(instrument, signal.direction, signal.index,
                              alert_id):
            return None
        png = self._render(df, state, signal,
                           style=chart_style or self.chart_style,
                           features=features, forecast=forecast,
                           instrument=instrument, mode=mode)
        forecast_line = forecast.line() if forecast is not None else None
        alert = build_alert(signal, instrument=instrument, mode=mode,
                            chart_png=png, forecast_note=forecast_line)
        alert.send_results = [sender.send(alert) for sender in self.senders]
        self._last_alert_bar[(instrument, signal.direction)] = signal.index
        self._seen_ids.add(alert.alert_id)
        return alert
