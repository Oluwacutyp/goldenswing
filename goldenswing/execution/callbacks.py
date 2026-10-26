"""ExecutionCallbackHandler — the alerts CallbackHandler, implemented.

A Telegram button tap flows:

    alert -> tap -> this handler -> risk pipeline -> result text

- EXECUTE -> size (never-round-up sizer) -> guard check (kill
  switch, latched halts, loss limits, exposure) -> place via the
  live executor (MT5Executor on the Windows host) -> journal "open".
- PAPER   -> the same sizing, placed on the PaperExecutor (no guard
  gate: it is simulation, and its record is labelled source="paper"
  so live and paper statistics never mix).
- SKIP    -> journal "skip" (the setup, judged and declined — data
  for the learning loop).

Every refusal is returned as the reply text with its reason: a guard
saying no is a result, not an error, and it is journalled as a note.
Context the Alert object does not carry (market phase at entry,
playbook, forecaster probability/rank) is supplied by whoever built
the alert via :meth:`register_context` — the live runner does this
for every alert it sends.
"""
from __future__ import annotations

from ..alerts.callbacks import AlertAction
from .executor import Executor, OrderRequest
from .learning import TradeJournal
from .risk import RiskGuards, SizingResult, SymbolSpec, size_position
from ..live.sizing import size_for_risk

#: alert.kind -> playbook when no richer context was registered.
_KIND_TO_PLAYBOOK = {
    "sweep_reversal": "sweep_reversal",
    "continuation": "trend_continuation",
}

DEFAULT_MAGIC = 260101


class ExecutionCallbackHandler:
    def __init__(self, *, executor: Executor, paper_executor: Executor,
                 guards: RiskGuards, spec: SymbolSpec,
                 journal: TradeJournal, risk_pct: float = 0.01,
                 magic: int = DEFAULT_MAGIC,
                 fixed_lots: float | None = None):
        self.executor = executor
        self.paper_executor = paper_executor
        self.guards = guards
        self.spec = spec
        self.journal = journal
        self.risk_pct = float(risk_pct)
        self.magic = int(magic)
        # Fixed-size mode (Exness $12 paper loop): every paper trade
        # at exactly ``fixed_lots`` (0.01), ignoring risk_pct AND the
        # forecaster sizing dial. None = the normal risk_pct sizer.
        # PAPER SIMULATION ONLY: _place ignores fixed lots for live
        # (execute) orders (journal-noted) and sizes them with the
        # dynamic rule. The executor underneath still enforces margin
        # (Exness) / its own rules.
        self.fixed_lots = None if fixed_lots is None else float(fixed_lots)
        self._context: dict[str, dict] = {}

    # ------------------------------------------------------------- context
    def register_context(self, alert_id: str, *, phase: str | None = None,
                         playbook: str | None = None,
                         forecaster_prob: float | None = None,
                         forecaster_rank: float | None = None,
                         forecast_size_mult: float = 1.0) -> None:
        self._context[alert_id] = {
            "phase": phase, "playbook": playbook,
            "forecaster_prob": forecaster_prob,
            "forecaster_rank": forecaster_rank,
            "forecast_size_mult": float(forecast_size_mult or 1.0),
        }

    def _context_for(self, alert) -> dict:
        ctx = self._context.get(alert.alert_id, {})
        return {
            "phase": ctx.get("phase") or "unknown",
            "playbook": ctx.get("playbook")
                        or _KIND_TO_PLAYBOOK.get(alert.kind, "unknown"),
            "forecaster_prob": ctx.get("forecaster_prob"),
            "forecaster_rank": ctx.get("forecaster_rank"),
            "forecast_size_mult": ctx.get("forecast_size_mult", 1.0),
        }

    # ------------------------------------------------------ CallbackHandler
    def handle(self, action: AlertAction, alert) -> str:
        action = AlertAction(action)
        if action is AlertAction.SKIP:
            ctx = self._context_for(alert)
            self.journal.log_skip(
                alert_id=alert.alert_id, instrument=alert.instrument,
                mode=alert.mode, direction=alert.direction,
                score=alert.score, phase=ctx["phase"],
                playbook=ctx["playbook"])
            return (f"⏭ Skipped {alert.direction_label} {alert.instrument} "
                    f"(score {alert.score:.0f}) — logged for learning.")
        if action is AlertAction.EXECUTE:
            return self._place(self.executor, alert, source="execute",
                               check_guards=True)
        if action is AlertAction.PAPER:
            return self._place(self.paper_executor, alert, source="paper",
                               check_guards=False)
        return f"unknown action {action!r}"  # pragma: no cover

    # ------------------------------------------------------------- pipeline
    def _place(self, executor: Executor, alert, *, source: str,
               check_guards: bool) -> str:
        label = "EXECUTE" if source == "execute" else "PAPER"
        risk_distance = alert.risk_distance()
        if risk_distance <= 0:
            return self._refuse(alert, label,
                                "alert carries no valid stop distance")
        acct = executor.account_state()
        ctx = self._context_for(alert)
        if source == "execute":
            # ENFORCEMENT (owner rulings): a live order NEVER uses
            # fixed lots or the retired legacy sizer — only the
            # live dynamic rule, which floors below-minimum sizes
            # at the broker minimum instead of abandoning. A stale
            # fixed-lots knob is IGNORED (journal-noted), never
            # applied and never a reason to miss the trade. Paper
            # simulation keeps its own models in the branches below.
            # The forecaster dial stays a scalar (never a veto).
            if self.fixed_lots is not None:
                self.journal.log_note(
                    f"EXECUTE: fixed lots {self.fixed_lots} ignored — "
                    "live orders use dynamic sizing",
                    alert_id=alert.alert_id, instrument=alert.instrument,
                    action="execute")
            size_mult = ctx.get("forecast_size_mult", 1.0) or 1.0
            result = size_for_risk(
                float(acct.balance) * float(self.risk_pct)
                * float(size_mult),
                float(risk_distance), self.spec)
            if not result.ok:
                return self._refuse(alert, label, result.verdict)
            sizing = SizingResult(float(result.lots),
                                  float(result.realized_risk), "")
        elif self.fixed_lots is not None:
            # Fixed-size mode: no risk_pct math, no forecaster dial.
            # Floor to the volume step (never round up); the broker
            # minimum still applies.
            step = self.spec.volume_step if self.spec.volume_step > 0 \
                else self.spec.volume_min
            lots = int(self.fixed_lots / step + 1e-9) * step
            lots = round(lots, 10)
            if lots < self.spec.volume_min:
                return self._refuse(
                    alert, label,
                    f"fixed lots {self.fixed_lots} floors below the "
                    f"broker minimum {self.spec.volume_min}")
            sizing = SizingResult(
                lots, self.spec.money_for_move(risk_distance, lots), "")
            size_mult = 1.0
        else:
            # the forecaster dial as sizing input: horizon agreement
            # scales risk up modestly; disagreement leaves sizing
            # untouched (never a veto — a low rank never blocks or
            # shrinks the trade).
            size_mult = ctx.get("forecast_size_mult", 1.0) or 1.0
            sizing = size_position(balance=acct.balance,
                                   risk_pct=self.risk_pct * size_mult,
                                   entry=alert.entry_ref, stop=alert.invalidation,
                                   spec=self.spec)
            if not sizing.ok:
                return self._refuse(alert, label, sizing.reason)
        if check_guards:
            open_risk, unstopped = executor.open_risk(self.spec)
            decision = self.guards.check_new_trade(
                balance=acct.balance, open_risk_money=open_risk,
                new_risk_money=sizing.risk_money,
                open_count=len(executor.positions()),
                unstopped_positions=unstopped)
            if not decision.allowed:
                return self._refuse(alert, label, decision.reason)
        request = OrderRequest(
            symbol=alert.instrument, direction=alert.direction,
            volume=sizing.lots, stop_loss=alert.invalidation,
            entry_ref=alert.entry_ref, targets=list(alert.targets),
            atr=alert.atr, risk_money=sizing.risk_money,
            alert_id=alert.alert_id, bar_index=alert.bar_index,
            magic=self.magic,
            comment=f"GS {alert.alert_id} ({source})")
        result = executor.place_order(request)
        if not result.ok:
            return self._refuse(alert, label,
                                f"executor refused: {result.message}")
        self.journal.log_open(
            alert_id=alert.alert_id, instrument=alert.instrument,
            mode=alert.mode, playbook=ctx["playbook"], phase=ctx["phase"],
            direction=alert.direction, score=alert.score, source=source,
            ticket=result.ticket, entry=alert.entry_ref,
            stop=alert.invalidation, volume=sizing.lots,
            risk_money=sizing.risk_money, factors=alert.factors,
            forecaster_prob=ctx["forecaster_prob"],
            forecaster_rank=ctx["forecaster_rank"],
            risk_atr=(risk_distance / alert.atr) if alert.atr > 0 else None)
        icon = "✅" if source == "execute" else "📄"
        dial = f" (forecaster 1d+3d agree ×{size_mult:.2f})" \
            if size_mult != 1.0 else ""
        return (f"{icon} {label} {alert.direction_label} {alert.instrument}: "
                f"{sizing.lots:.2f} lots @ ~{alert.entry_ref:,.1f}, "
                f"stop {alert.invalidation:,.1f}, "
                f"risk {sizing.risk_money:,.0f} {acct.currency}{dial} "
                f"(ticket {result.ticket}).")

    def _refuse(self, alert, label: str, reason: str) -> str:
        self.journal.log_note(
            f"{label} refused: {reason}", alert_id=alert.alert_id,
            instrument=alert.instrument, action=label.lower())
        return f"⛔ {label} {alert.instrument} refused — {reason}"
