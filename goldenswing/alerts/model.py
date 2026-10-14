"""The Alert model: one signal, its chart, and its full text thesis."""
from __future__ import annotations

from dataclasses import dataclass, field

from .narrator import narrate


@dataclass
class SendResult:
    """Outcome of one sender attempting one alert.  ``delivered`` is
    only True when the message actually left the machine — outbox
    writes report ``delivered=False`` with mode ``"outbox"``."""

    channel: str            # "telegram" | "email"
    delivered: bool
    mode: str               # "live" | "outbox"
    detail: str = ""


@dataclass
class Alert:
    alert_id: str
    instrument: str
    mode: str                       # swing | scalp | fast_scalp
    direction: int                  # +1 long | -1 short
    kind: str                       # sweep_reversal | continuation
    score: float
    is_golden: bool
    entry_ref: float
    entry_zone: tuple[float, float]
    invalidation: float
    targets: list[float]
    factors: list[str]
    timestamp: object               # signal bar timestamp
    bar_index: int
    atr: float = 0.0
    forecast_note: str | None = None
    chart_png: bytes = b""
    text: str = ""
    send_results: list[SendResult] = field(default_factory=list)

    @property
    def direction_label(self) -> str:
        return "LONG" if self.direction == 1 else "SHORT"

    def risk_distance(self) -> float:
        return abs(self.entry_ref - self.invalidation)


def make_alert_id(instrument: str, bar_index: int, direction: int) -> str:
    """Compact, callback-safe id (Telegram callback_data is capped at
    64 bytes): e.g. ``XAU50412L``."""
    tag = "".join(ch for ch in instrument.upper() if ch.isalnum())[:3]
    return f"{tag}{bar_index}{'L' if direction == 1 else 'S'}"


def build_alert_text(a: Alert) -> str:
    """The full thesis as plain text — Telegram caption / email body."""
    icon = "🟢" if a.direction == 1 else "🔴"
    arrow = "▲" if a.direction == 1 else "▼"
    ts = a.timestamp
    ts_txt = ts.strftime("%Y-%m-%d %H:%M UTC") if hasattr(ts, "strftime") \
        else str(ts)
    star = " ★ GOLDEN SWING" if a.is_golden else ""
    lines = [
        f"{icon} {arrow} {a.direction_label} {a.instrument} — "
        f"score {a.score:.0f}{star}",
        f"Mode: {a.mode} · Setup: {a.kind} · {ts_txt}",
        "",
        f"Entry: {a.entry_ref:,.1f} "
        f"(zone {a.entry_zone[0]:,.1f}–{a.entry_zone[1]:,.1f})",
        f"Stop (invalidation): {a.invalidation:,.1f}",
    ]
    if a.targets:
        lines.append("Targets: " + " · ".join(
            f"T{j + 1} {t:,.1f}" for j, t in enumerate(a.targets)))
    if a.atr > 0:
        lines.append(
            f"Risk distance: {a.risk_distance():.1f} "
            f"({a.risk_distance() / a.atr:.2f}x ATR)")
    lines.append("")
    # The narrator is the story; the Backing list below is the evidence.
    narration = narrate(a)
    if narration:
        lines.append(narration)
        lines.append("")
    lines.append("Backing:")
    for f in a.factors:
        lines.append(f"• {f}")
    if a.forecast_note:
        lines.append("")
        lines.append(a.forecast_note)
    return "\n".join(lines)


def build_alert(signal, *, instrument: str = "XAUUSD", mode: str = "swing",
                chart_png: bytes = b"",
                forecast_note: str | None = None) -> Alert:
    """Assemble an Alert from a GoldenSignal (text built eagerly)."""
    a = Alert(
        alert_id=make_alert_id(instrument, signal.index, signal.direction),
        instrument=instrument,
        mode=mode,
        direction=signal.direction,
        kind=signal.kind,
        score=signal.score,
        is_golden=signal.is_golden,
        entry_ref=signal.entry_ref,
        entry_zone=tuple(signal.entry_zone),
        invalidation=signal.invalidation,
        targets=list(signal.targets),
        factors=list(signal.factors),
        timestamp=signal.timestamp,
        bar_index=signal.index,
        atr=signal.atr,
        forecast_note=forecast_note,
        chart_png=chart_png,
    )
    a.text = build_alert_text(a)
    return a
