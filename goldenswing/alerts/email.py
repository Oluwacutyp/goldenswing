"""Email sender — smtplib, chart PNG attached, outbox fallback.

Configuration (environment):

- ``GS_SMTP_HOST`` / ``GS_SMTP_PORT`` (default 587, STARTTLS;
  port 465 uses implicit SSL)
- ``GS_SMTP_USER`` / ``GS_SMTP_PASSWORD`` (login, optional for
  unauthenticated relays)
- ``GS_ALERT_EMAIL_TO`` — recipient
- ``GS_ALERT_OUTBOX``   — outbox dir for offline mode (optional)

Live only when host AND recipient are set; otherwise outbox mode.
"""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

from . import outbox
from .model import Alert, SendResult


class EmailSender:
    def __init__(self, host: str | None = None, port: int | None = None,
                 user: str | None = None, password: str | None = None,
                 recipient: str | None = None,
                 outbox_dir: str | Path | None = None,
                 timeout: float = 30.0):
        self.host = host or os.environ.get("GS_SMTP_HOST") or None
        self.port = port or int(os.environ.get("GS_SMTP_PORT", "587"))
        self.user = user or os.environ.get("GS_SMTP_USER") or None
        self.password = password or os.environ.get(
            "GS_SMTP_PASSWORD") or None
        self.recipient = recipient or os.environ.get(
            "GS_ALERT_EMAIL_TO") or None
        self.outbox_dir = outbox.resolve_outbox(outbox_dir)
        self.timeout = timeout

    @classmethod
    def from_env(cls, **kw) -> "EmailSender":
        return cls(**kw)

    @property
    def mode(self) -> str:
        return "live" if (self.host and self.recipient) else "outbox"

    def build_message(self, alert: Alert) -> EmailMessage:
        arrow = "▲ LONG" if alert.direction == 1 else "▼ SHORT"
        star = " ★ GOLDEN" if alert.is_golden else ""
        msg = EmailMessage()
        msg["Subject"] = (f"[GoldenSwing] {arrow} {alert.instrument} "
                          f"score {alert.score:.0f}{star} — "
                          f"{alert.mode}")
        msg["From"] = self.user or "goldenswing@localhost"
        msg["To"] = self.recipient or ""
        msg.set_content(alert.text)
        if alert.chart_png:
            msg.add_attachment(alert.chart_png, maintype="image",
                               subtype="png",
                               filename=f"{alert.alert_id}.png")
        return msg

    def send(self, alert: Alert) -> SendResult:
        if self.mode == "outbox":
            paths = outbox.write_alert(self.outbox_dir, alert, "email")
            return SendResult(
                "email", False, "outbox",
                "no GS_SMTP_HOST/GS_ALERT_EMAIL_TO — wrote "
                + ", ".join(p.name for p in paths))
        try:
            msg = self.build_message(alert)
            if self.port == 465:
                smtp = smtplib.SMTP_SSL(self.host, self.port,
                                        timeout=self.timeout)
            else:
                smtp = smtplib.SMTP(self.host, self.port,
                                    timeout=self.timeout)
            with smtp:
                if self.port != 465:
                    smtp.starttls()
                if self.user and self.password:
                    smtp.login(self.user, self.password)
                smtp.send_message(msg)
            return SendResult("email", True, "live",
                              f"sent to {self.recipient} via {self.host}")
        except Exception as exc:  # report, don't fake
            return SendResult("email", False, "live",
                              f"send failed: {type(exc).__name__}: {exc}")
