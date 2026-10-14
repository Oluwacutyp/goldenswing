"""GoldenSwing alert service.

Turns GoldenSignals into study-able alerts — chart PNG + full text
thesis — delivered via Telegram (inline Execute/Paper/Skip buttons)
and email.  Without credentials both senders run in OFFLINE OUTBOX
mode: the alert is written to a directory, never faked as sent.
"""
from .callbacks import (AlertAction, CallbackHandler, CallbackRegistry,
                        CallbackResult, make_callback_data,
                        parse_callback_data)
from .email import EmailSender
from .model import Alert, SendResult, build_alert, build_alert_text
from .positions import PositionNotifier
from .service import AlertService
from .summary import build_daily_summary
from .telegram import ErrorDeduper, TelegramSender

__all__ = [
    "Alert", "AlertAction", "AlertService", "CallbackHandler",
    "CallbackRegistry", "CallbackResult", "EmailSender", "ErrorDeduper",
    "PositionNotifier", "SendResult",
    "TelegramSender", "build_alert", "build_alert_text",
    "build_daily_summary",
    "make_callback_data", "parse_callback_data",
]
