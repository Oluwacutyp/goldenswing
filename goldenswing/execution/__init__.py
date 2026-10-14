"""GoldenSwing execution layer (Phase 8).

Risk guards that can refuse but never trade (persisted, latching),
a never-round-up lot sizer, two executors behind one interface —
PaperExecutor (bar-driven simulation, works anywhere) and
MT5Executor (the official MetaTrader5 Python library on a Windows
host with an active terminal; imported lazily) — the Phase-2 exit
stepper shared by both, the append-only trade journal + performance
book (the learning loop), the alerts CallbackHandler implementation,
and the live runner.

Nothing in this package imports MetaTrader5 at module level.
"""
from .callbacks import ExecutionCallbackHandler
from .executor import (AccountState, Executor, OrderRequest, OrderResult,
                       Position)
from .learning import PerformanceBook, TradeJournal, export_training_data
from .manager import ExitEvent, ExitStepper, TrackedPosition
from .mt5 import MT5Executor, MT5Unavailable
from .paper import CloseRecord, PaperExecutor
from .risk import (GuardDecision, RiskConfig, RiskGuards, SizingResult,
                   SymbolSpec, gold_spec, size_position)
from .runner import ForecasterScorer, LiveRunner, RunnerConfig

__all__ = [
    "AccountState", "CloseRecord", "ExecutionCallbackHandler", "Executor",
    "ExitEvent", "ExitStepper", "ForecasterScorer", "GuardDecision",
    "LiveRunner", "MT5Executor", "MT5Unavailable", "OrderRequest",
    "OrderResult", "PaperExecutor", "PerformanceBook", "Position",
    "RiskConfig", "RiskGuards", "RunnerConfig", "SizingResult",
    "SymbolSpec", "TradeJournal", "TrackedPosition", "export_training_data",
    "gold_spec", "size_position",
]
