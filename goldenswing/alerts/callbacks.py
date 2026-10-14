"""Telegram inline-button callbacks — the interface the EXECUTION
phase will implement.

Button flow
-----------
A Telegram alert carries three inline buttons::

    [✅ Execute] [📄 Paper] [⏭ Skip]

Each button's ``callback_data`` is ``gs:<action>:<alert_id>`` (built by
:func:`make_callback_data`, parsed by :func:`parse_callback_data`).
When a user taps, the bot process receives a ``callback_query``; the
host application (the live runner, Phase 8+) is responsible for
polling ``getUpdates`` and feeding ``callback_query.data`` into
:meth:`CallbackRegistry.dispatch` together with the alerts it has
sent (keyed by ``alert_id``).

A handler is any object implementing the :class:`CallbackHandler`
protocol — ``handle(action, alert) -> str`` returning the reply text to
show the user.  The execution phase registers ONE handler that maps:

- ``EXECUTE`` -> place the trade via the MetaTrader5 Python executor
  (SPEC: execution is the MT5 python library, demo or real by login),
- ``PAPER``   -> record the trade in the paper book,
- ``SKIP``    -> log the skip (feeds the learning loop).

This module deliberately ships NO default handler and NO fake
execution: dispatching with no handler registered returns
``handled=False`` with the reason, so a button tap can never silently
pretend a trade happened.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, runtime_checkable

CALLBACK_PREFIX = "gs"


class AlertAction(str, Enum):
    EXECUTE = "execute"
    PAPER = "paper"
    SKIP = "skip"


def make_callback_data(action: AlertAction, alert_id: str) -> str:
    data = f"{CALLBACK_PREFIX}:{AlertAction(action).value}:{alert_id}"
    if len(data.encode("utf-8")) > 64:
        raise ValueError(f"callback_data too long for Telegram: {data!r}")
    return data


def parse_callback_data(data: str) -> tuple[AlertAction, str]:
    parts = data.split(":", 2)
    if len(parts) != 3 or parts[0] != CALLBACK_PREFIX:
        raise ValueError(f"not a GoldenSwing callback: {data!r}")
    try:
        action = AlertAction(parts[1])
    except ValueError:
        raise ValueError(f"unknown callback action: {parts[1]!r}") from None
    if not parts[2]:
        raise ValueError("callback carries no alert_id")
    return action, parts[2]


@runtime_checkable
class CallbackHandler(Protocol):
    """Implemented by the execution phase (see module docstring)."""

    def handle(self, action: AlertAction, alert) -> str:
        """Act on the tap; return the user-visible reply text."""
        ...


@dataclass
class CallbackResult:
    handled: bool
    action: AlertAction | None = None
    alert_id: str | None = None
    message: str = ""


class CallbackRegistry:
    """Local registry: one handler, fed by the host's update loop."""

    def __init__(self) -> None:
        self._handler: CallbackHandler | None = None

    def register(self, handler: CallbackHandler) -> None:
        if not isinstance(handler, CallbackHandler):
            raise TypeError("handler must implement CallbackHandler "
                            "(handle(action, alert) -> str)")
        self._handler = handler

    @property
    def has_handler(self) -> bool:
        return self._handler is not None

    def dispatch(self, data: str, alerts: Mapping[str, object]) -> CallbackResult:
        """Route one raw ``callback_query.data`` string.

        Never pretends: parse errors, unknown alerts and a missing
        handler all come back ``handled=False`` with the reason.
        """
        try:
            action, alert_id = parse_callback_data(data)
        except ValueError as exc:
            return CallbackResult(False, message=str(exc))
        alert = alerts.get(alert_id)
        if alert is None:
            return CallbackResult(
                False, action, alert_id,
                f"unknown alert {alert_id!r} (expired or not ours)")
        if self._handler is None:
            return CallbackResult(
                False, action, alert_id,
                "no callback handler registered — the execution phase "
                "is not wired yet; nothing was executed")
        reply = self._handler.handle(action, alert)
        return CallbackResult(True, action, alert_id, reply)
