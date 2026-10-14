"""Position-lifecycle notifications for the live manager.

A real desk narrates its positions: fills, partial banks, stop moves to
breakeven / profit-lock, and closes with PnL + R.  ``_LiveManager`` calls
into :class:`PositionNotifier`; the notifier formats and sends.  Routine
chatter is silent (``disable_notification``); closes buzz — the outcome
is what the human is waiting for.  Mined from the v18 bot's
``notify_trade_update`` (per-action semantics), minus its ML confetti.
"""
from __future__ import annotations

from .telegram import TelegramSender


def _fmt(x: float) -> str:
    return f"{x:,.1f}"


class PositionNotifier:
    """Formats position events and sends them via a TelegramSender.

    ``sender`` may be None (dry-run / tests) — events are then formatted
    and returned but not sent, so the manager never depends on Telegram
    being up to manage the position.
    """

    def __init__(self, sender: TelegramSender | None = None):
        self.sender = sender

    def _send(self, text: str, *, silent: bool) -> str:
        if self.sender is None:
            return text
        res = self.sender.send_text(text, silent=silent)
        return text if res.delivered else f"{text}\n[send failed: {res.detail}]"

    # ------------------------------------------------------------------ #
    def on_fill(self, *, ticket: int, alert_id: str, direction: int,
                fill_price: float, volume: float, stop: float,
                instrument: str = "XAUUSD") -> str:
        arrow = "▲ LONG" if direction == 1 else "▼ SHORT"
        text = (
            f"🎯 FILLED {arrow} {instrument}\n"
            f"ticket {ticket} · alert {alert_id}\n"
            f"entry {_fmt(fill_price)} · {volume:.2f} lots\n"
            f"stop {_fmt(stop)}")
        return self._send(text, silent=True)

    def on_partial(self, *, ticket: int, direction: int,
                   fill_price: float, volume_closed: float,
                   pnl_money: float, instrument: str = "XAUUSD") -> str:
        arrow = "▲" if direction == 1 else "▼"
        text = (
            f"🔔 PARTIAL {arrow} {instrument} (ticket {ticket})\n"
            f"banked {volume_closed:.2f} lots @ {_fmt(fill_price)} · "
            f"PnL {pnl_money:+.2f}")
        return self._send(text, silent=True)

    def on_breakeven(self, *, ticket: int, direction: int,
                     instrument: str = "XAUUSD") -> str:
        arrow = "▲" if direction == 1 else "▼"
        text = (f"🛡️ BREAKEVEN {arrow} {instrument} "
                f"(ticket {ticket})\nstop moved to entry — risk off")
        return self._send(text, silent=True)

    def on_trail(self, *, ticket: int, direction: int, new_stop: float,
                 locked_r: float, instrument: str = "XAUUSD") -> str:
        arrow = "▲" if direction == 1 else "▼"
        text = (f"🔒 TRAIL {arrow} {instrument} (ticket {ticket})\n"
                f"stop → {_fmt(new_stop)} · locking {locked_r:.2f}R")
        return self._send(text, silent=True)

    def on_close(self, *, ticket: int, direction: int, exit_price: float,
                 reason: str, pnl_money: float, r_net: float,
                 hold_bars: int, instrument: str = "XAUUSD") -> str:
        arrow = "▲" if direction == 1 else "▼"
        icon = "💰" if pnl_money >= 0 else "💸"
        reason_txt = {"stop": "stopped", "trail": "trailed stop",
                      "timeout": "time stop"}.get(reason, reason)
        text = (
            f"{icon} CLOSED {arrow} {instrument} (ticket {ticket})\n"
            f"{reason_txt} @ {_fmt(exit_price)} · "
            f"PnL {pnl_money:+.2f} · {r_net:+.2f}R · held {hold_bars} bars")
        # closes buzz — the outcome is what the human waits for
        return self._send(text, silent=False)

    def on_broker_close(self, *, ticket: int, alert_id: str,
                        instrument: str = "XAUUSD") -> str:
        text = (
            f"⚠️ POSITION GONE {instrument} (ticket {ticket}, "
            f"alert {alert_id})\nclosed broker-side (SL/TP in the terminal) "
            f"— outcome is in the terminal history, not the journal")
        return self._send(text, silent=False)
