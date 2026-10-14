"""MT5 runner — the live engine on MT5 as the single source of truth.

Tick pump drives everything; there is NO poll timer anywhere in
this path:

- Each NEW tick -> :class:`TickManager` re-evaluates every open
  position; breakeven / trailing / partial actions execute via the
  MT5 executor ON THAT TICK (``modify_sl`` / partial close).  A
  breakeven is never up to 5 minutes late again.
- Tick time crossing a bar boundary -> that TF's bars refresh from
  MT5 (``copy_rates``), the forward-only stack appends them, and a
  new closed H1 bar runs the SAME signal path as the paper loop:
  featurize -> playbooks ``live_signal`` -> confluence gate with
  the hierarchy verdict (D1 -> H4 -> H1 -> entry clock, hard veto
  OFF) -> score (with the MT5 volume bonus) -> setup-dependent
  order via MT5Executor (market at the level, pending LIMIT at the
  POI, or pending STOP beyond the breakout level — never
  always-market) -> TickManager tracking from the fill tick on.
- Positions are reconciled against ``positions_get`` (MT5 is the
  source of truth for what is actually open): a position the broker
  closed (stop / TP / manual) drops out of tracking; a position
  with our magic that appeared (restart) is adopted.

Safety:

- DEMO-ONLY by default: the runner refuses to start against a REAL
  MT5 account (``trade_mode == 2``) unless ``allow_live=True`` is
  passed explicitly.  The $12 demo / EC2 deploy runs on demo.
- One position at a time (same as the paper loop).
- Kill switch file stops the loop (checked every tick batch).
- Telegram command polling is ON by default (``poll_commands=True``):
  this engine OWNS ``getUpdates``.  The OANDA loop's poller must be
  OFF at cutover (``GS_TELEGRAM_POLL=0``) — one getUpdates consumer
  only, or the two pollers steal each other's updates.  While the
  OANDA loop still runs (overlap window), start this engine with
  ``--no-poll-commands``.
- Journal + risk state live under ``data/mt5_...`` — never the
  OANDA loop's ``data/paper_state/`` / ``trade_journal.jsonl`` and
  never the Exness loop's ``data/exness12/``.
- Every processed tick is logged to per-day CSVs
  (``data/mt5_ticks_YYYYMMDD.csv``) next to the journal, exactly as
  seen.  Every signal alert and every fill generates a chart PNG
  (``data/mt5_charts/<alert_id>[ -fill].png``) with the H4 zone,
  entry, stop, targets, and hierarchy verdict.

RS peer features (silver/dollar) are OFF on this path
(``peers=None`` — "RS unavailable", never zeros): the peer loader
would otherwise call yfinance, a second live source.  Fetching peer
bars from MT5 is a follow-up; nothing in the signal path depends
on peers.
"""
from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..alerts.commands import CommandContext, is_owner_callback
from ..backtest.data import REPO_ROOT
from ..charts.engine import render_mt5_signal
from ..confluence import ConfluenceEngine
from ..features.engine import read_and_featurize
from ..feed.tick_log import TickLogger
from ..live.hierarchy import (LEG_ORDER, TimeframeStack, TF_MINUTES,
                              check_live_entry, check_live_trigger,
                              evaluate_hierarchy, live_setup,
                              qm_annotation)
from ..live.manipulation import chase_blocked
from ..live.volume import volume_features
from ..reader.session_timeframes import compose as _compose_session
from ..playbooks import (FVGLimitPlaybook, FVGMidlinePlaybook,
                         InversionPlaybook, JudasSwingPlaybook,
                         KillzoneOpenPlaybook, OrderBlockPlaybook,
                         QMM5Playbook, RetracementTrapPlaybook,
                         SweepReversalPlaybook, TrendlineRetestPlaybook,
                         TurtleSoupPlaybook, VReversalPlaybook)
from ..playbooks.base import EngineContext
from .executor import OrderRequest, PendingOrder
from .learning import TradeJournal
from .mt5 import MT5Executor, select_order_type
from .mt5_commands import (mt5_confirm_keyboard, mt5_control_keyboard,
                           mt5_router, parse_mt5_callback)
from .mt5_config import LiveConfigStore
from .risk import RiskConfig, RiskGuards, size_position
from .tick_manager import TickAction, TickManager
from ..feed.mt5 import TRADE_MODE_REAL, MT5Feed, TickPump

DATA_DIR = REPO_ROOT / "data"


@dataclass
class MT5RunnerConfig:
    symbol: str = "XAUUSD"
    risk_pct: float = 0.01
    fixed_lots: float | None = None  # when set, every trade uses it
    magic: int = 260101
    execute_playbooks: tuple = ("sweep_reversal", "fvg_midline",
                                "trendline_retest", "fvg_limit",
                                "inversion", "killzone_open",
                                "judas_swing")
    min_score: float = 60.0
    signal_gap_bars: int = 0
    journal_path: Path = DATA_DIR / "mt5_journal.jsonl"
    risk_state_path: Path = DATA_DIR / "mt5_risk_state.json"
    kill_switch_path: Path = DATA_DIR / "mt5_KILL"
    hierarchy_required: bool = True
    tick_interval: float = 0.25
    allow_live: bool = False  # True ONLY for an explicit real-money run
    max_positions: int = 1
    reconcile_every_ticks: int = 20
    # Telegram command polling: ON by default — this engine OWNS
    # getUpdates.  Pass poll_commands=False (CLI --no-poll-commands)
    # only while the OANDA loop still polls (overlap window).
    poll_commands: bool = True
    poll_interval_seconds: float = 30.0
    # Tick CSV logging (per-day CSVs next to the journal).
    tick_log: bool = True
    charts_dir: Path | None = None  # default: <journal parent>/mt5_charts
    chart_style: str = "classic"  # MT5 chart class: classic | blueprint
    # Pending-order expiry: a pending LIMIT / STOP that has not
    # filled after this many H1 bar closes is cancelled (the setup
    # went stale — a real trader does not leave a limit resting at
    # a dead zone).  Session-end expiry is NOT used: a pending
    # placed late in a session must not die on a clock boundary
    # before the pullback has had a chance to arrive.
    pending_expiry_bars: int = 4
    # Layering / scale-in (one position, several layers): a new
    # confirming setup in the same direction adds a layer only when
    # the position is already in profit by this much (R, measured on
    # the base layer's risk) and the TOTAL risk across all layers
    # stays within this multiple of the base layer's risk money.
    # At most this many layers including the base.
    layer_min_profit_r: float = 0.5
    layer_max_total_risk_mult: float = 2.0
    layer_max_layers: int = 3
    # Live Telegram control (see mt5_config.py — persisted in
    # data/mt5_config.json, changeable via /set + Confirm):
    manipulation_filter: bool = True   # liquidity-test entry filter
    disabled_playbooks: tuple = ()      # playbook names switched off
    paused: bool = False                # pause new signals, not management

    @classmethod
    def from_env(cls, **overrides) -> "MT5RunnerConfig":
        cfg = cls(
            symbol=os.environ.get("GS_SYMBOL", "XAUUSD"),
            risk_pct=float(os.environ.get("GS_RISK_PCT", "0.01")),
            fixed_lots=float(os.environ["GS_FIXED_LOTS"])
            if os.environ.get("GS_FIXED_LOTS") else None,
            allow_live=os.environ.get("GS_MT5_ALLOW_LIVE", "0") == "1",
            poll_commands=os.environ.get("GS_MT5_POLL_COMMANDS", "1") != "0",
            poll_interval_seconds=float(
                os.environ.get("GS_MT5_POLL_INTERVAL", "30.0")),
            pending_expiry_bars=int(
                os.environ.get("GS_MT5_PENDING_EXPIRY_BARS", "4")),
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg


@dataclass
class PendingOrderState:
    """One pending LIMIT / STOP the engine is responsible for.

    The broker holds the order; this record is the engine's memory
    of WHY it exists, so invalidation / expiry / bias-flip cancels
    can fire without re-deriving the setup.  ``zone`` is the H4 POI
    band the setup was built on (None for a pending adopted at
    restart, where only the broker SL is known).  ``placed_bar`` is
    the forming-H1-bar index at placement; ``expiry_bar`` is the
    first H1-close index at which the pending dies unfilled.
    """

    ticket: int
    kind: str                       # "limit" | "stop"
    direction: int                  # +1 buy | -1 sell
    level: float                    # pending price (the POI entry level)
    volume: float
    sl: float                       # buffered broker stop
    tp: float | None
    alert_id: str
    playbook: str
    zone: dict | None
    invalidation: float             # raw structural invalidation
    placed_bar: int
    expiry_bar: int
    track_request: object = None    # OrderRequest for TickManager on fill
    sig: object = None              # playbook signal (fill chart/alert)
    verdict: dict | None = None     # hierarchy verdict (fill chart)
    is_layer: bool = False          # pending fills onto the held position


class MT5Runner:
    """Tick-driven live loop on MT5 feed + MT5 executor."""

    def __init__(self, config: MT5RunnerConfig | None = None, *,
                 feed: MT5Feed | None = None,
                 executor: MT5Executor | None = None,
                 senders: list | None = None,
                 tick_manager: TickManager | None = None) -> None:
        self.cfg = config or MT5RunnerConfig()
        self.feed = feed
        self.executor = executor
        self.senders = senders or []
        self.journal = TradeJournal(self.cfg.journal_path)
        self.guards = RiskGuards(self.cfg.risk_state_path, RiskConfig(),
                                 kill_switch_path=self.cfg.kill_switch_path)
        self.tick_manager = tick_manager or TickManager()
        self._playbooks: list | None = None
        self._engine: ConfluenceEngine | None = None
        self._stack: TimeframeStack | None = None
        self._qm_touch: dict | None = None
        # Live evaluation clock (tick-cross trigger, entry-TF entry):
        # the forming H1 bar tracked from ticks, the cached trigger
        # levels from closed bars, and the pending (fired, still
        # unconfirmed) trigger awaiting entry confirmation.
        self._forming: dict | None = None
        self._live_setup: dict | None = None
        self._pending: dict | None = None
        # Broker pending orders (LIMIT / STOP) this engine placed or
        # adopted: ticket -> PendingOrderState.  Distinct from
        # ``self._pending`` (the fired, unconfirmed trigger of the
        # tick-clock machinery) — a pending ORDER is a broker order;
        # a pending TRIGGER is pre-signal state.
        self._pending_orders: dict[int, PendingOrderState] = {}
        self._last_processed: dict[str, pd.Timestamp] = {}
        self._last_sig_bar: dict[str, int] = {}
        self._tick_count = 0
        self._last_tick: dict | None = None
        self._stopped = False
        # Startup ping: exactly one Telegram "engine started" message
        # per process start (see _send_startup_ping).  Guarded so a
        # second run() call on the same runner never double-pings.
        self._startup_ping_sent = False
        # Per-day tick CSV logger (next to the journal; None = disabled).
        self._tick_logger = TickLogger(self.cfg.journal_path.parent) \
            if self.cfg.tick_log else None
        self._charts_dir = self.cfg.charts_dir or (
            self.cfg.journal_path.parent / "mt5_charts")
        self._charts_dir.mkdir(parents=True, exist_ok=True)
        # Telegram command polling (ON by default — this engine owns
        # getUpdates; see MT5RunnerConfig.poll_commands).
        self._tg_offset = 0
        self._tg_synced = False
        self._last_poll = 0.0
        self._command_router = None
        # Blocked-leg stats (why no trades — surfaced by /status):
        # cheap in-memory counters + journal notes on every block.
        # A restart resets the counters; the journal keeps the notes.
        self._blocked_leg_counts: dict[str, int] = {}
        self._blocked_last: str = ""
        self._blocked_total: int = 0
        self._signal_counts: dict[str, int] = {}
        self._manipulation_blocked: int = 0
        # Pending Confirm taps: token -> {"action", "arg"}.
        self._confirm_tokens: dict[str, dict] = {}
        # Persisted live config (data/mt5_config.json): when the file
        # exists it wins over the constructed defaults — /set changes
        # survive a restart because the next process loads this file.
        self._config_store = LiveConfigStore(
            self.cfg.journal_path.parent / "mt5_config.json")
        if self._config_store.path.exists():
            self._config_store.apply_to(self.cfg)

    # ------------------------------------------------------------ wiring
    def _get_playbooks(self) -> list:
        """Every playbook the engine knows: the 7 executing ones plus
        the 5 shadow ones.  Instantiation alone never orders — the
        ``auto = pb.name in execute_playbooks`` gate at the order step
        decides, and shadow playbooks only ever get evaluated +
        logged (see _run_signal_path).  A playbook missing from this
        list can never fire, executing or shadow."""
        if self._playbooks is None:
            self._playbooks = [SweepReversalPlaybook(),
                               FVGMidlinePlaybook(),
                               TrendlineRetestPlaybook(),
                               FVGLimitPlaybook(),
                               InversionPlaybook(),
                               KillzoneOpenPlaybook(),
                               JudasSwingPlaybook(),
                               TurtleSoupPlaybook(),
                               VReversalPlaybook(),
                               RetracementTrapPlaybook(),
                               OrderBlockPlaybook(),
                               QMM5Playbook()]
        return self._playbooks

    # ------------------------------------------------- blocked-leg stats
    def _record_blocked(self, reason: str, playbook: str | None = None
                        ) -> None:
        """Count one blocked verdict by leg (cheap counters only —
        journal notes are written by the caller where they matter)."""
        if reason.startswith("hierarchy:"):
            parts = reason.split(":")
            leg = parts[1] if len(parts) > 1 and parts[1] else "hierarchy"
        elif "manipulation" in reason:
            leg = "manipulation_filter"
        elif reason.startswith("paused"):
            leg = "paused"
        elif "session_scope" in reason:
            leg = "session"
        elif "tripwire" in reason:
            leg = "required_context"
        elif "hunting ground" in reason:
            leg = "hunting_ground"
        elif reason.startswith("order:"):
            leg = "order_selection"
        elif reason.startswith("min_score"):
            leg = "min_score"
        elif reason.startswith("playbook_error"):
            leg = "playbook_error"
        else:
            leg = "other"
        self._blocked_leg_counts[leg] = \
            self._blocked_leg_counts.get(leg, 0) + 1
        self._blocked_total += 1
        self._blocked_last = \
            f"{playbook or '?'}: {reason}"[:200]

    def blocked_stats(self) -> dict:
        """Snapshot for /status: why no trades."""
        return {"legs": dict(self._blocked_leg_counts),
                "last": self._blocked_last,
                "total": self._blocked_total,
                "manipulation_blocked": self._manipulation_blocked,
                "signal_counts": dict(self._signal_counts)}

    def _get_engine(self) -> ConfluenceEngine:
        if self._engine is None:
            self._engine = ConfluenceEngine(
                playbooks=self._get_playbooks(),
                risk_guards=self.guards, journal=self.journal,
                risk_mode="enforce")
        return self._engine

    # --------------------------------------------------------------- loop
    def run(self, *, max_ticks: int | None = None) -> dict:
        """Run the tick pump until stopped / capped.  Returns stats."""
        feed = self.feed or MT5Feed()
        executor = self.executor or MT5Executor()
        ok, msg = feed.connect()
        if not ok:
            raise RuntimeError(f"MT5 runner: feed connect failed: {msg}")
        feed_msg = msg  # executor.connect() below reuses `msg`
        # The feed is the source of truth for the broker's gold symbol:
        # auto-detect may have resolved XAUUSD -> XAUUSDm.  Orders,
        # positions, and specs below must use the resolved symbol, so
        # adopt it when the feed was not explicitly configured (an
        # explicit feed symbol already matches the runner's).
        if not feed.cfg.symbol_explicit \
                and feed.cfg.symbol != self.cfg.symbol:
            self.cfg.symbol = feed.cfg.symbol
        acct = feed.account_state()
        if acct.get("trade_mode") == TRADE_MODE_REAL \
                and not self.cfg.allow_live:
            raise RuntimeError(
                "MT5 runner: refusing to trade a REAL MT5 account "
                "(trade_mode=2) without allow_live=True — demo only.")
        if Path(self.cfg.kill_switch_path).exists():
            raise RuntimeError(
                f"MT5 runner: kill switch present at "
                f"{self.cfg.kill_switch_path} — refusing to start.")
        ok, msg = executor.connect()
        if not ok:
            raise RuntimeError(f"MT5 runner: executor connect failed: {msg}")

        # Warm start: full closed history per TF -> forward-only stack.
        self._stack = TimeframeStack.from_frames(feed.warm_start_all())
        h1 = self._stack.frames.get("H1")
        if h1 is None or len(h1) == 0:
            raise RuntimeError("MT5 runner: warm start returned no H1 bars")
        for tf, frame in self._stack.frames.items():
            if len(frame):
                # Last-processed CLOSE (= last closed bar's open + TF).
                # Comparing closes to closes: a close-to-open mixup
                # would refresh on every tick.
                self._last_processed[tf] = (
                    frame.index[-1] + pd.Timedelta(minutes=TF_MINUTES[tf]))
        self._qm_touch = self._advance_qm(self._stack.frames.get("M5"))
        # Trigger levels from the closed warm-start bars.  The H1
        # close is NOT the signal clock anymore — this setup is what
        # the tick clock checks every tick (see _evaluate_live_tick).
        self._live_setup = live_setup(self._stack)
        self.feed = feed
        self.executor = executor
        self.journal.log_note(
            "mt5 runner: started "
            f"(symbol={self.cfg.symbol}) "
            f"({', '.join(f'{tf}:{len(self._stack.frames.get(tf, []))}' for tf in self._stack.frames)}) "
            f"account login={acct.get('login')} equity={acct.get('equity')} "
            f"trade_mode={acct.get('trade_mode')}"
            + (f" connect: {feed_msg}" if feed_msg != "MT5 connected" else ""))

        # Fast-forward playbook streaming state over history; past
        # signals are discarded — only the latest close trades.
        states = read_and_featurize(h1, None)
        for pb in self._get_playbooks():
            for t2 in range(len(h1)):
                pb.live_signal(states, t2, h1)
        # Adopt positions the broker already holds with our magic
        # (restart sync — MT5 is the source of truth for open trades).
        self._adopt_existing(executor)
        # Adopt pending orders the broker still holds with our magic:
        # a restart must not orphan a resting limit/stop.
        self._adopt_existing_pending(executor)

        # Startup ping: ONE Telegram "engine started" message, after
        # feed connect + runner start, independent of command polling
        # (it fires even with --no-poll-commands).  Never raises.
        self._send_startup_ping(acct)

        stats: dict = {"ticks": 0, "bars": 0, "signals": 0,
                       "executed": 0, "shadow": 0, "gated": 0,
                       "hierarchy_blocked": 0, "tick_actions": 0,
                       "reconciled_closed": 0, "triggers": 0,
                       "invalidated": 0, "pending": 0,
                       "pending_filled": 0, "pending_cancelled": 0}
        pump = TickPump(feed, interval=self.cfg.tick_interval)
        try:
            pump.run(self._on_tick(stats), max_ticks=max_ticks)
        finally:
            # Engine stop: cancel every pending order.  A resting
            # limit/stop left on the broker would fill later with no
            # management — a stale pending is never left behind.
            for ticket in list(self._pending_orders):
                self._cancel_pending(ticket, "engine_stop", stats)
            self.journal.log_note(
                f"mt5 runner: stopped: {stats['ticks']} ticks, "
                f"{stats['bars']} H1 bars, {stats['signals']} signals, "
                f"{stats['executed']} executed, {stats['shadow']} shadow, "
                f"{stats['hierarchy_blocked']} hierarchy-blocked, "
                f"{stats['triggers']} triggers, "
                f"{stats['invalidated']} invalidated, "
                f"{stats['pending']} pending, "
                f"{stats['pending_filled']} pending-filled, "
                f"{stats['pending_cancelled']} pending-cancelled")
        return {**stats, "journal": str(self.cfg.journal_path)}

    def stop(self) -> None:
        self._stopped = True

    # -------------------------------------------------------------- ticks
    def _on_tick(self, stats: dict):
        """TickPump callback: log, manage, reconcile, refresh, poll."""
        def _handle(tick: dict) -> None:
            if self._stopped or Path(self.cfg.kill_switch_path).exists():
                raise StopIteration
            stats["ticks"] += 1
            self._tick_count += 1
            self._last_tick = tick
            # Tick CSV: exactly as seen, never breaks the loop.
            if self._tick_logger is not None:
                try:
                    self._tick_logger.log(tick)
                except Exception as exc:
                    self.journal.log_note(
                        f"mt5: tick log failed (loop continues): {exc}")
            price = tick["mid"]
            self._manage_tick(price, stats)
            # Pending LIMIT / STOP invalidation is tick-level (the
            # zone breaking must kill a resting order now, not on
            # the next reconcile): price beyond the adverse zone
            # edge or the invalidation level cancels the order.
            self._check_pending_orders(price, stats)
            if self._tick_count % self.cfg.reconcile_every_ticks == 0:
                self._reconcile(price, stats)
            self._refresh_bars(tick, stats)
            # Live evaluation clock: trigger on tick-cross, entry on
            # entry-TF closes.  H1 closes never gate evaluation.
            self._evaluate_live_tick(tick, stats)
            self._poll_telegram_commands()
        return _handle

    def _manage_tick(self, price: float, stats: dict) -> None:
        """TickManager actions for every tracked position, executed now."""
        for ticket in list(self.tick_manager.tickets()):
            for action in self.tick_manager.on_tick(ticket, price):
                stats["tick_actions"] += 1
                if action.kind == "modify_sl":
                    res = self.executor.modify_sl(ticket, action.level)
                    if res.ok:
                        self.tick_manager.apply(ticket, action)
                        self.journal.log_note(
                            f"mt5 tick: ticket {ticket} SL -> "
                            f"{action.level:.2f} ({action.reason})")
                    else:
                        self.journal.log_note(
                            f"mt5 tick: ticket {ticket} SL move FAILED "
                            f"({action.reason}): {res.message} — retry next tick")
                elif action.kind == "partial":
                    res = self.executor.close_position(ticket, action.volume)
                    if res.ok:
                        self.tick_manager.apply(ticket, action)
                        self.journal.log_note(
                            f"mt5 tick: ticket {ticket} partial close "
                            f"{action.volume} lots at {price:.2f}")
                    else:
                        self.journal.log_note(
                            f"mt5 tick: ticket {ticket} partial FAILED: "
                            f"{res.message} — retry next tick")
                elif action.kind == "layer_stop":
                    # Added layer's own SL crossed: close just that
                    # leg, the base rides.
                    res = self.executor.close_position(ticket, action.volume)
                    if res.ok:
                        self.tick_manager.apply(ticket, action)
                        self.journal.log_note(
                            f"mt5 tick: ticket {ticket} layer stop "
                            f"{action.volume} lots at {price:.2f} "
                            f"(layer SL {action.level:.2f})")
                    else:
                        self.journal.log_note(
                            f"mt5 tick: ticket {ticket} layer stop FAILED: "
                            f"{res.message} — retry next tick")
                elif action.kind == "stop_hit":
                    # Broker SL is primary; only close here if the
                    # position is STILL open (safety net).
                    still = self.executor.positions()
                    if any(p.ticket == ticket for p in still):
                        res = self.executor.close_position(ticket)
                        if res.ok:
                            self._record_close(ticket, price, "stop",
                                               stats)
                    else:
                        self._record_close(ticket, price, "stop", stats)

    def _reconcile(self, price: float, stats: dict) -> None:
        """MT5 positions_get is the truth: drop broker-closed, adopt ours."""
        try:
            live = {p.ticket: p for p in self.executor.positions()}
        except Exception as exc:
            self.journal.log_note(f"mt5 reconcile: positions failed: {exc}")
            return
        for ticket in list(self.tick_manager.tickets()):
            if ticket not in live:
                self._record_close(ticket, price, "broker_closed", stats)
        # Positions that match a tracked pending order are NOT adopted
        # here: _reconcile_pending below handles them as FILLS (full
        # open record + fill chart), not as bare adoptions — adopting
        # first would double-handle them and lose the signal context.
        pending_alerts = {s.alert_id for s in self._pending_orders.values()}
        for ticket, pos in live.items():
            if ticket not in self.tick_manager.tickets() \
                    and pos.magic == self.cfg.magic \
                    and pos.alert_id not in pending_alerts:
                self.tick_manager.adopt(
                    ticket=ticket, symbol=pos.symbol,
                    direction=pos.direction, volume=pos.volume,
                    entry_price=pos.entry_price, sl=pos.sl,
                    alert_id=pos.alert_id)
                self.journal.log_note(
                    f"mt5 reconcile: adopted ticket {ticket} "
                    f"({'+' if pos.direction > 0 else '-'}1 "
                    f"@{pos.entry_price:.2f})")
        self._reconcile_pending(live, stats)

    def _reconcile_pending(self, live_positions: dict, stats: dict) -> None:
        """Pending orders vs the broker: fills become positions,
        externally-cancelled orders are dropped, broker pendings
        with our magic are adopted.  A pending that left
        ``orders_get`` is NEVER assumed cancelled — a position with
        the same ``alert_id`` proves it filled; only the absence of
        both proves an external cancel."""
        try:
            live_pending = {o.ticket: o for o in
                            self.executor.pending_orders(self.cfg.symbol)}
        except Exception as exc:
            self.journal.log_note(
                f"mt5 reconcile: pending orders failed: {exc}")
            return
        for ticket in list(self._pending_orders):
            state = self._pending_orders.get(ticket)
            if state is None or ticket in live_pending:
                continue  # still resting on the broker
            matched = [p for p in live_positions.values()
                       if p.alert_id == state.alert_id
                       and p.magic == self.cfg.magic]
            if matched:
                self._record_pending_fill(state, matched[0], stats)
            else:
                del self._pending_orders[ticket]
                stats["pending_cancelled"] = stats.get(
                    "pending_cancelled", 0) + 1
                self.journal.log_note(
                    f"mt5 pending gone: ticket {ticket} "
                    f"{state.kind} @ {state.level:.2f} — cancelled "
                    f"externally / broker expiry, no fill")
        for ticket, order in live_pending.items():
            if ticket not in self._pending_orders \
                    and order.magic == self.cfg.magic:
                self._track_pending_order(
                    order, playbook="", zone=None,
                    invalidation=order.sl or order.price,
                    placed_bar=self._current_h1_bar(),
                    sig=None, verdict=None, track_request=None)
                self.journal.log_note(
                    f"mt5 reconcile: adopted pending {order.kind} "
                    f"ticket {ticket} @ {order.price:.2f}")

    def _record_close(self, ticket: int, price: float, reason: str,
                      stats: dict) -> None:
        pos = self.tick_manager.position(ticket)
        if pos is None:
            return
        d = pos.direction
        pnl_r = d * (price - pos.entry_price) / pos.risk_price \
            if pos.risk_price > 0 else 0.0
        pnl_money = 0.0
        try:
            spec = self.executor.symbol_spec(self.cfg.symbol)
            pnl_money = d * spec.money_for_move(
                abs(price - pos.entry_price), pos.volume)
        except Exception:
            pass
        self.journal.log_close(
            alert_id=pos.alert_id, ticket=ticket, exit_price=price,
            reason=reason, pnl_money=float(pnl_money), r_net=float(pnl_r),
            hold_bars=0, partial_taken=pos.partial_done)
        self.journal.log_note(
            f"mt5 close: ticket {ticket} {reason} at {price:.2f} "
            f"({pnl_r:+.2f}R)")
        self.tick_manager.remove(ticket)
        stats["reconciled_closed"] += 1

    # --------------------------------------------------------------- bars
    def _refresh_bars(self, tick: dict, stats: dict) -> None:
        """Tick time crossing a bar boundary -> refresh that TF's bars.

        A TF close refreshes ONLY that TF's leg (bias / zones /
        trigger levels / entry closes) — it never gates evaluation.
        The trigger fires on tick-cross; the entry leg on entry-TF
        closes (see _evaluate_live_tick).  No poll timer: the tick
        itself is the clock.

        Ordering on a tick that closes several TFs at once (an H1
        close coinciding with an entry-TF close is the common case):
        bars append first, then the pending entry gets its
        confirmation shot against the freshly closed entry bars —
        the entry bar that closes WITH its H1 bar still belongs to
        that bar — and only then does the H1 close expire an
        unconfirmed pending.  A single fetch serves the whole tick
        (one snapshot), and the setup is recomputed once, not once
        per TF.  Phase order is fixed, never dict-iteration order.
        """
        if self._stack is None:
            return
        tick_time = pd.Timestamp(tick["time"])
        cursors = {t: self._stack.cursors.get(t)
                   for t in self._stack.frames}
        # Phase 1: which TFs closed a bar on this tick?
        closed_tfs = []
        for tf in list(self._stack.frames):
            closed = MT5Feed.closed_bar_close(tf, tick_time)
            if closed > self._last_processed.get(tf, closed):
                closed_tfs.append(tf)
        if not closed_tfs:
            return
        fresh = self.feed.fetch_new_all(cursors)
        # Phase 2: append every newly closed bar; refresh legs.
        appended: list[str] = []
        h1_accepted = 0
        for tf in closed_tfs:
            new = fresh.get(tf)
            if new is None or not len(new):
                continue
            accepted, _rej = self._stack.append(tf, new)
            if not accepted:
                continue
            appended.append(tf)
            self._last_processed[tf] = (
                self._stack.cursors[tf]
                + pd.Timedelta(minutes=TF_MINUTES[tf]))
            if tf == "M5":
                self._qm_touch = self._advance_qm(
                    self._stack.frames.get("M5"))
            if tf == "H1":
                h1_accepted += accepted
        if any(tf in ("D1", "H4", "H1") for tf in appended):
            self._live_setup = live_setup(self._stack)
        # Phase 3: the pending entry confirms on entry-TF closes
        # BEFORE any H1-close expiry — the entry bar that closes
        # with the H1 bar still belongs to that bar.
        if self._pending is not None and \
                self._pending["entry_clock"] in appended:
            self._evaluate_pending_entry(tick, stats)
        # Phase 4: H1 close — time stop, pending expiry, forming reset.
        if "H1" in appended:
            stats["bars"] = stats.get("bars", 0) + h1_accepted
            self._on_h1_close(tick, stats)
        # Phase 5: D1/H4 close flipped the bias — the pending
        # trigger's premise is gone (only if still pending).
        if any(tf in ("D1", "H4") for tf in appended):
            self._check_pending_bias(stats)

    # ------------------------------------------- live evaluation clock
    # The H1 close is NOT the signal clock.  The trigger fires when
    # the tick crosses the trigger level; the entry leg evaluates on
    # entry-TF closes; H1/H4/D1 closes only refresh their legs.  The
    # hierarchy legs and gate formulas are unchanged (see
    # live/hierarchy.py::live_setup / check_live_trigger /
    # check_live_entry) — only the EVALUATION CLOCK changed.

    def _on_h1_close(self, tick: dict, stats: dict) -> None:
        """An H1 close: time stop + pending expiry + forming reset.

        The close refreshes legs (done in _refresh_bars before this
        call) — it never gates evaluation.
        """
        h1 = self._stack.frames["H1"]
        last_bar_index = len(h1) - 1
        # Time stop for tracked positions (bar-defined rule).
        for ticket in list(self.tick_manager.tickets()):
            for action in self.tick_manager.on_bar_close(
                    ticket, last_bar_index):
                res = self.executor.close_position(ticket, action.volume)
                if res.ok:
                    self._record_close(
                        ticket,
                        self._last_tick["mid"] if self._last_tick else 0.0,
                        "timeout", stats)
        # Pending-order expiry: unfilled past expiry_bar dies — the
        # setup went stale (a real trader does not leave a limit
        # resting at a dead zone).
        for ticket, state in list(self._pending_orders.items()):
            if last_bar_index >= state.expiry_bar:
                self._cancel_pending(
                    ticket,
                    f"expired: no fill after "
                    f"{self.cfg.pending_expiry_bars} H1 bars", stats)
        # A pending trigger belongs to the H1 bar that just closed:
        # without entry confirmation it is invalidated by
        # non-confirmation — logged, never erased.
        if self._pending is not None:
            self._invalidate_pending("h1_close_without_entry_confirm",
                                     stats)
        # The new forming H1 bar starts with this tick (the tick that
        # crossed the boundary belongs to the new bar).  Volume
        # starts at zero: _evaluate_live_tick owns accumulation, so
        # the boundary tick is counted exactly once (there).
        mid = float(tick["mid"])
        self._forming = {
            "open": mid, "high": mid, "low": mid,
            "open_time": pd.Timestamp(tick["time"]).floor("h"),
            "volume": 0.0, "real_volume": 0.0,
            "spread": tick.get("spread", 0)}

    def _check_pending_bias(self, stats: dict) -> None:
        """D1/H4 close flipped the bias (or un-armed the setup): the
        pending trigger's premise is gone — invalidate, never erase.
        Pending ORDERS die the same way: a bias flip / un-armed
        setup cancels every resting order whose premise is gone."""
        setup = self._live_setup
        if self._pending is not None:
            if setup is None \
                    or setup["direction"] != self._pending["direction"]:
                self._invalidate_pending("bias_flipped", stats)
        if self._pending_orders:
            if setup is None:
                for ticket in list(self._pending_orders):
                    self._cancel_pending(
                        ticket, "bias_flipped: setup unarmed", stats)
            else:
                for ticket, state in list(
                        self._pending_orders.items()):
                    if state.direction != setup["direction"]:
                        self._cancel_pending(
                            ticket, "bias_flipped", stats)

    def _invalidate_pending(self, reason: str, stats: dict) -> None:
        """Drop a pending trigger: logged, never erased.

        Non-confirmation is not repainting — the fired trigger stays
        in the journal; only the pending state is cleared.
        """
        pend = self._pending
        if pend is None:
            return
        self._pending = None
        stats["invalidated"] = stats.get("invalidated", 0) + 1
        self.journal.log_note(
            f"mt5: trigger invalidated ({reason}): {pend['trigger']} "
            f"{'+' if pend['direction'] > 0 else '-'}1 "
            f"@ {pend['entry']:.2f} — entry never confirmed, "
            f"setup logged, never erased")

    def _evaluate_live_tick(self, tick: dict, stats: dict) -> None:
        """Live evaluation clock: trigger on tick-cross, entry on
        entry-TF closes.  H1 closes never gate evaluation."""
        if self._stack is None:
            return
        mid = float(tick["mid"])
        tick_time = pd.Timestamp(tick["time"])
        bar_open = tick_time.floor("h")
        forming = self._forming
        if forming is None or forming["open_time"] != bar_open:
            forming = {"open": mid, "high": mid, "low": mid,
                       "open_time": bar_open, "volume": 0.0,
                       "real_volume": 0.0, "spread": 0}
            self._forming = forming
        else:
            forming["high"] = max(forming["high"], mid)
            forming["low"] = min(forming["low"], mid)
        # The forming bar's volume accumulates from ticks (never a
        # synthetic zero): the live frame's forming row carries real
        # accumulated tick volume.  The forming bar is market-data
        # tracking: it updates even while a position is open, so the
        # intrabar high/low survive a hold; only the setup logic
        # below is gated on holding.
        forming["volume"] = forming.get("volume", 0.0) \
            + float(tick.get("tick_volume", 0) or 0)
        forming["real_volume"] = forming.get("real_volume", 0.0) \
            + float(tick.get("real_volume", 0) or 0)
        forming["spread"] = tick.get("spread", forming.get("spread", 0))
        # NOTE (layering): setup logic runs while holding — a confirming
        # setup in the same direction becomes a LAYER, not a second
        # position.  The one-position rule lives in _run_signal_path
        # (new positions only when flat; layers only when holding).
        if self._pending is not None:
            # Reversal: the break no longer holds — invalidate
            # (logged, never erased).  The entry leg simply does not
            # confirm.
            pend = self._pending
            if (pend["direction"] > 0 and mid <= pend["level"]) or \
                    (pend["direction"] < 0 and mid >= pend["level"]):
                self._invalidate_pending("reversal", stats)
            return
        setup = self._live_setup
        if setup is None:
            return  # legs 1-2 not armed
        # Cheap pre-filter: only a price past the trigger level AND
        # at the POI band can fire — the full check runs rarely.
        if setup["direction"] > 0 and mid <= setup["ref_high"]:
            return
        if setup["direction"] < 0 and mid >= setup["ref_low"]:
            return
        zone = setup["zone"]
        if not (zone["bottom"] <= mid <= zone["top"]):
            return
        trig = check_live_trigger(
            setup, open_=forming["open"], high=forming["high"],
            low=forming["low"], close=mid,
            closed_h1=self._stack.frames["H1"])
        if trig is None:
            return
        # Trigger fired on the tick — logged, never erased.  The
        # entry leg decides whether it confirms (on this tick's
        # already-closed composing entry bars, then on each
        # entry-TF close).
        session = _compose_session(tick_time)
        self._pending = {**trig, "fired_ts": tick_time,
                         "h1_open": bar_open,
                         "entry_clock": session["entry_clock"]}
        stats["triggers"] = stats.get("triggers", 0) + 1
        self.journal.log_note(
            f"mt5: trigger fired {trig['trigger']} "
            f"{'+' if trig['direction'] > 0 else '-'}1 @ {mid:.2f} "
            f"(H1 bar {bar_open}) — awaiting "
            f"{session['entry_clock']} entry")
        self._evaluate_pending_entry(tick, stats)

    def _evaluate_pending_entry(self, tick: dict, stats: dict) -> None:
        """Entry leg on entry-TF closes (or immediately on trigger
        fire, against this tick's already-closed composing bars).

        A confirmed entry runs the signal path and consumes the
        pending trigger.  Non-confirmation leaves the pending
        trigger standing until the H1 close expires it, a reversal
        invalidates it, or a bias flip kills it — the fired trigger
        is logged, never erased.
        """
        pend = self._pending
        if pend is None or self._stack is None:
            return
        # NOTE (layering): entry confirmation runs while holding — a
        # confirmed entry in the same direction becomes a layer
        # (_run_signal_path branches), never a second position.
        entry_frame = self._stack.frames.get(pend["entry_clock"])
        event = check_live_entry(entry_frame, pend["direction"],
                                 swing_n=3, since=pend["h1_open"],
                                 entry_clock=pend["entry_clock"])
        if event is None or event != pend["direction"]:
            return  # no entry event yet, or against-bias: not confirmed
        self._pending = None
        self._run_live_signal(pend, tick, stats)

    def _run_live_signal(self, pend: dict, tick: dict,
                         stats: dict) -> None:
        """Entry confirmed: run the signal path on the live frame.

        The live frame is the closed H1 bars PLUS the forming bar
        (live values from ticks).  The verdict is built from the
        fired trigger + the confirm tick — same shape as the
        closed-bar verdict — so the playbooks, confluence gate, and
        order path are unchanged.
        """
        # NOTE (layering): no holding guard here — _run_signal_path
        # branches to layers when holding, new positions when flat.
        h1_closed = self._stack.frames["H1"]
        forming = self._forming or {
            "open": float(tick["mid"]), "high": float(tick["mid"]),
            "low": float(tick["mid"]),
            "open_time": pd.Timestamp(tick["time"]).floor("h"),
            "volume": 0.0, "real_volume": 0.0, "spread": 0}
        row = pd.DataFrame([{
            "open": float(forming["open"]),
            "high": float(forming["high"]),
            "low": float(forming["low"]),
            "close": float(tick["mid"]),
            "volume": float(forming.get("volume", 0.0)),
            "real_volume": float(forming.get("real_volume", 0.0)),
            "spread": forming.get("spread", 0)}],
            index=pd.DatetimeIndex([forming["open_time"]], tz="UTC"))
        h1 = pd.concat([h1_closed, row])
        states = read_and_featurize(h1, None)
        t = len(h1) - 1
        years = (h1.index[-1] - h1.index[0]).total_seconds() \
            / 86400.0 / 365.25
        verdict = self._live_verdict(pend, tick, h1_closed, states, t)
        self._run_signal_path(h1, t, states, years, verdict, stats)

    def _live_verdict(self, pend: dict, tick: dict,
                      h1_closed: pd.DataFrame, states, t: int) -> dict:
        """Live verdict: same shape as the closed-bar verdict.

        Legs 1-4 come from the fired trigger (closed-bar levels +
        tick-cross); leg 5 from the confirmed entry-TF event.  Entry
        is the confirm tick's mid (the actual market entry),
        stop/target from the trigger computation, risk re-measured
        from the confirm price.  AMD / QM / volume annotations as in
        :meth:`_verdict_for_bar`.
        """
        direction = int(pend["direction"])
        state = states[t] if 0 <= t < len(states) else None
        feats = getattr(state, "features", None) \
            if state is not None else None
        ph = getattr(feats, "phase", None) if feats is not None else None
        amd_phase = getattr(ph, "phase", None) if ph is not None else None
        amd_via = getattr(ph, "via", None) if ph is not None else None
        amd_conf = getattr(ph, "confidence", None) \
            if ph is not None else None
        if amd_phase is None:
            amd_phase = "unknown"
        session = _compose_session(pd.Timestamp(tick["time"]),
                                   amd_phase=amd_phase)
        mid = float(tick["mid"])
        stop = float(pend["stop"])
        legs = {leg: {"passed": True, "reason": ""} for leg in LEG_ORDER}
        legs["d1_bias"]["reason"] = f"D1 bias {direction:+d} (closed bars)"
        legs["h4_agree"]["reason"] = f"H4 agrees {direction:+d} (closed)"
        legs["h4_zone"]["reason"] = "live price at H4 POI (tick-cross)"
        legs["h1_trigger"]["reason"] = f"tick-cross {pend['trigger']}"
        legs["entry"]["reason"] = \
            f"{pend['entry_clock']} event confirmed"
        return {
            "allowed": True, "blocked_leg": "", "legs": legs,
            "bias_d1": direction, "bias_h4": direction,
            "direction": direction, "zone": pend["zone"],
            "trigger": pend["trigger"], "entry": mid,
            "stop": stop, "target": float(pend["target"]),
            "risk": abs(mid - stop),
            "session": session, "entry_clock": session["entry_clock"],
            "amd_phase": amd_phase, "amd_via": amd_via,
            "amd_confidence": amd_conf,
            "qm": qm_annotation(self._qm_touch, direction,
                                pend["h1_open"]),
            "veto_enabled": False, "asof": pend["h1_open"].isoformat(),
            "volume": volume_features(h1_closed),
        }

    def _run_signal_path(self, h1: pd.DataFrame, t: int,
                         states, years: float, verdict: dict,
                         stats: dict) -> None:
        """Verdict -> playbooks -> gate -> order (shared signal path).

        The verdict may come from a closed H1 bar (legacy) or from a
        tick-cross trigger + confirmed entry (live clock) — the path
        after the verdict is identical: same playbooks, same
        confluence gate, same hierarchy, manipulation filter
        (entry selection: no chasing a failed liquidity test),
        setup-dependent order (market / pending limit / pending
        stop), position tracked by the TickManager from the fill
        tick on.  When holding, a confirming same-direction signal
        becomes a LAYER (_try_layer), never a second position.
        """
        # One position at a time is enforced by branching below:
        # flat -> new position, holding -> layer only.  There is no
        # early return here: the signal path must run while holding
        # so layers can be evaluated.
        # Paused (Telegram Pause): evaluate nothing into orders — the
        # confirmed entry is dropped, tick management of the held
        # position continues elsewhere.  The block is counted so
        # /status can answer "why no trades".
        if self.cfg.paused:
            self._record_blocked(
                "paused: engine paused — confirmed entry dropped")
            self.journal.log_note(
                "mt5: signal path paused — confirmed entry dropped, "
                "no order")
            return
        ctx = EngineContext(df=h1, states=states, years=years,
                            instrument=self.cfg.symbol)
        engine = self._get_engine()
        for pb in self._get_playbooks():
            if pb.name in (self.cfg.disabled_playbooks or ()):
                continue  # switched off via /set — not evaluated
            try:
                self._run_one_playbook(pb, sig_ctx=(ctx, engine),
                                       h1=h1, t=t, verdict=verdict,
                                       stats=stats)
            except Exception as exc:
                # A crashing playbook never kills the signal path
                # (shadow playbooks in particular only ran behind
                # the live-skip before this wiring fix).
                self._record_blocked(
                    f"playbook_error: {type(exc).__name__}: {exc}",
                    pb.name)
                self.journal.log_note(
                    f"mt5: {pb.name} signal path failed: "
                    f"{type(exc).__name__}: {exc}")

    def _run_one_playbook(self, pb, *, sig_ctx, h1, t, verdict, stats
                          ) -> None:
        """One playbook through gate -> score -> (order | shadow log).

        Shadow playbooks (not in ``execute_playbooks``) are
        EVALUATED + LOGGED + COUNTED exactly like executing ones —
        but they never reach chart/alert/order: the ``auto`` gate
        below is the only thing that may order, and it is False for
        them.  Instantiation never orders by itself.
        """
        ctx, engine = sig_ctx
        # Positional, as before: playbook live_signal signatures name
        # the frame parameter differently (df / h1).
        sigs = pb.live_signal(ctx.states, t, h1) or []
        for sig in sigs:
            if sig.score < self.cfg.min_score:
                self._record_blocked(
                    f"min_score: score {sig.score:.0f} < "
                    f"{self.cfg.min_score:.0f}", pb.name)
                continue
            if sig.index - self._last_sig_bar.get(pb.name, -10 ** 9) \
                    < self.cfg.signal_gap_bars:
                continue
            ok, reason = engine.gate_live_signal(
                pb, sig, ctx, hierarchy_verdict=verdict)
            if not ok:
                stats["gated"] += 1
                if reason.startswith("hierarchy:"):
                    stats["hierarchy_blocked"] = stats.get(
                        "hierarchy_blocked", 0) + 1
                self._record_blocked(reason, pb.name)
                self.journal.log_note(
                    f"mt5: {pb.name} signal gated: {reason}")
                continue
            conf, notes = engine.score_live_signal(
                pb, sig, ctx, hierarchy_verdict=verdict)
            self._last_sig_bar[pb.name] = sig.index
            stats["signals"] += 1
            self._signal_counts[pb.name] = \
                self._signal_counts.get(pb.name, 0) + 1
            auto = pb.name in self.cfg.execute_playbooks
            tag = "MT5" if auto else "SHADOW"
            alert_id = f"MT5-{pb.name}-{sig.index}"
            self.journal.log_note(
                f"mt5 [{tag}] {pb.name} "
                f"{'+' if sig.direction > 0 else '-'}1 "
                f"score={sig.score:.0f} conf={conf:.0f} "
                f"entry~{sig.entry_ref:.1f} stop~{sig.invalidation:.1f} "
                f"factors={'+'.join(notes) if notes else '-'} "
                f"hierarchy: D1 {verdict['bias_d1']:+d} / H4 "
                f"{verdict['bias_h4']:+d} / {verdict['trigger']} / "
                f"{verdict['entry_clock']} amd:{verdict.get('amd_phase')} "
                f"qm:{verdict.get('qm', {}).get('status')} "
                f"vol:{(verdict.get('volume') or {}).get('volume_ratio')}")
            if not auto:
                # Shadow: evaluated + logged + counted — never
                # charted, alerted, or ordered.
                continue
            # Chart: H4 zone + entry/stop/targets + verdict, saved
            # next to the journal and attached to the alert.
            png, _path = self._build_chart(
                sig, verdict, alert_id=alert_id, playbook=pb.name)
            self._send_alert(
                f"MT5 [{tag}] {pb.name} "
                f"{'+' if sig.direction > 0 else '-'}1 "
                f"entry~{sig.entry_ref:.1f} stop~{sig.invalidation:.1f} "
                f"conf={conf:.0f}",
                chart_png=png)
            # Manipulation / liquidity-test filter (entry
            # selection, NOT a veto): a signal chasing past
            # a level whose most recent test failed is
            # skipped — the engine waits for the retest /
            # real leg.  Retest entries and reversals pass.
            # Off via /set manipulation_filter off.
            if self.cfg.manipulation_filter:
                h1_closed = (self._stack.frames.get("H1")
                             if self._stack is not None else None)
                blocked, why = chase_blocked(
                    h1_closed, int(sig.direction),
                    float(sig.entry_ref), float(sig.atr))
                if blocked:
                    stats["manipulation_blocked"] = stats.get(
                        "manipulation_blocked", 0) + 1
                    self._manipulation_blocked += 1
                    self._record_blocked(
                        f"manipulation: {why}", pb.name)
                    self.journal.log_note(
                        f"mt5: {pb.name} signal manipulation-blocked: "
                        f"{why}")
                    continue
            if self.tick_manager.tickets():
                self._try_layer(pb, sig, stats, verdict=verdict,
                                entry_bar=t)
            else:
                self._place_live_order(pb, sig, stats,
                                       verdict=verdict,
                                       entry_bar=t)

    # ------------------------------------------------------------- layering
    # Scale-in onto ONE position: a confirming setup in the same
    # direction adds a layer (own entry / volume / SL) while the
    # position is in profit, with total risk capped.  The broker nets
    # the layer deal into the one position (same symbol / direction /
    # magic); the engine tracks the legs in TickManager.layers.
    def _try_layer(self, pb, sig, stats: dict,
                   verdict: dict | None = None,
                   entry_bar: int | None = None) -> None:
        """A gated signal while holding: layer or skip, never a
        second position."""
        tickets = self.tick_manager.tickets()
        if len(tickets) != 1:
            self.journal.log_note(
                f"mt5: layer skipped {pb.name}: not exactly one "
                f"position held")
            return
        ticket = tickets[0]
        pos = self.tick_manager.position(ticket)
        if pos is None:
            return
        if int(sig.direction) != int(pos.direction):
            self.journal.log_note(
                f"mt5: layer skipped {pb.name}: signal against the "
                f"held position (no flip)")
            return
        layers = self.tick_manager.layers(ticket)
        if len(layers) >= self.cfg.layer_max_layers:
            self.journal.log_note(
                f"mt5: layer skipped {pb.name}: max layers "
                f"({self.cfg.layer_max_layers}) reached")
            return
        # In profit?  R measured on the base layer's risk.
        mid = float((self._last_tick or {}).get("mid")
                    or (verdict or {}).get("entry")
                    or sig.entry_ref)
        unreal_r = int(pos.direction) * (mid - pos.entry_price) \
            / pos.risk_price if pos.risk_price > 0 else 0.0
        if unreal_r < self.cfg.layer_min_profit_r:
            self.journal.log_note(
                f"mt5: layer skipped {pb.name}: position +{unreal_r:.2f}R "
                f"< +{self.cfg.layer_min_profit_r:.2f}R required")
            return
        # Total risk cap: base risk money x max_mult, across all layers.
        try:
            spec = self.executor.symbol_spec(self.cfg.symbol)
        except Exception:
            self.journal.log_note(f"mt5: layer skipped {pb.name}: "
                                  f"no symbol spec")
            return
        base = layers[0]
        base_risk = spec.money_for_move(
            abs(base.entry - base.sl), base.volume)
        current = sum(spec.money_for_move(abs(ly.entry - ly.sl),
                                          ly.volume)
                    for ly in layers)
        budget = self.cfg.layer_max_total_risk_mult * base_risk - current
        if budget <= 0:
            self.journal.log_note(
                f"mt5: layer skipped {pb.name}: total risk cap "
                f"({self.cfg.layer_max_total_risk_mult}x base) reached")
            return
        # The layer's own stop (buffered, like a base order).
        stop0 = TickManager.planned_stop(
            OrderRequest(symbol=self.cfg.symbol, direction=sig.direction,
                         volume=0.0, stop_loss=sig.invalidation,
                         atr=sig.atr))
        risk_per_lot = spec.money_for_move(
            abs(float(sig.entry_ref) - stop0), 1.0)
        if risk_per_lot <= 0:
            self.journal.log_note(
                f"mt5: layer skipped {pb.name}: zero risk per lot")
            return
        # Size: same sizing as a base order, floored to 0.01 and
        # capped by the remaining risk budget.
        if self.cfg.fixed_lots is not None:
            lots = float(self.cfg.fixed_lots)
        else:
            acct = self.feed.account_state()
            sizing = size_position(balance=acct["equity"],
                                   risk_pct=self.cfg.risk_pct,
                                   entry=sig.entry_ref, stop=stop0,
                                   spec=spec)
            if sizing.lots <= 0:
                self.journal.log_note(
                    f"mt5: layer sizing refused {pb.name}: "
                    f"{sizing.reason}")
                return
            lots = sizing.lots
        lots = min(lots, int(budget / risk_per_lot / 0.01) * 0.01)
        if lots < 0.01:
            self.journal.log_note(
                f"mt5: layer skipped {pb.name}: risk cap leaves "
                f"{lots:.2f} lots (< 0.01)")
            return
        self._place_live_order(pb, sig, stats, verdict=verdict,
                               entry_bar=entry_bar, is_layer=True,
                               layer_lots=float(lots))

    def _record_layer_fill(self, pb_name: str, sig, *, fill: float,
                           lots: float, stop0: float, stats: dict) -> None:
        """A layer order filled: append the leg, move the broker SL
        to the base breakeven (only ever TIGHTENS), journal + alert.
        No new open record — the layer is a leg of the one position,
        not a second trade (the position's close record settles the
        whole stack).
        """
        tickets = self.tick_manager.tickets()
        if len(tickets) != 1:
            self.journal.log_note(
                f"mt5: layer fill dropped {pb_name}: no position held")
            return
        ticket = tickets[0]
        pos = self.tick_manager.position(ticket)
        if pos is None:
            return
        layer = self.tick_manager.add_layer(
            ticket, entry=float(fill), volume=float(lots), sl=float(stop0),
            alert_id=f"MT5-{pb_name}-{sig.index}-layer",
            playbook=pb_name)
        if layer is None:
            return
        stats["layers"] = stats.get("layers", 0) + 1
        stats["layer_fills"] = stats.get("layer_fills", 0) + 1
        layers = self.tick_manager.layers(ticket)
        # Move the broker SL to the base breakeven (safe) — only if
        # it tightens the current broker SL, never looser.
        d = int(pos.direction)
        eff = self.tick_manager.effective_stop(ticket)
        level = float(pos.entry_price)
        moved = False
        if eff is not None and ((d == 1 and level > eff)
                                or (d == -1 and level < eff)):
            res = self.executor.modify_sl(ticket, level)
            if res.ok:
                self.tick_manager.apply(
                    ticket, TickAction("modify_sl", level=level,
                                       reason="breakeven"))
                moved = True
            else:
                self.journal.log_note(
                    f"mt5 layer: broker SL move to breakeven FAILED "
                    f"({res.message}) — retry next tick")
        self.journal.log_note(
            f"mt5 layer: ticket {ticket} {pb_name} "
            f"{'+' if sig.direction > 0 else '-'}1 {lots} lots @ "
            f"{fill:.2f} SL {stop0:.2f} "
            f"({len(layers)} layers)"
            + (f" broker SL -> breakeven {level:.2f}" if moved else ""))
        self._send_alert(
            f"MT5 LAYER {pb_name} {'+' if sig.direction > 0 else '-'}1 "
            f"{lots} lots @ {fill:.2f} SL {stop0:.2f} "
            f"({len(layers)} layers, ticket {ticket})"
            + (f" — broker SL -> breakeven {level:.2f}" if moved else ""))

    def trim_top_layer(self) -> str:
        """Manual trim (/trim): close the top (most recent) added
        layer early while the base rides.  The first mutating
        Telegram command — it executes one broker partial close,
        synchronously, inside the poll loop."""
        tickets = self.tick_manager.tickets()
        if not tickets:
            return "no open position — nothing to trim"
        ticket = tickets[0]
        action = self.tick_manager.close_top_layer(ticket)
        if action is None:
            return "only the base layer — nothing to trim"
        res = self.executor.close_position(ticket, action.volume)
        if not res.ok:
            return f"trim FAILED: {res.message} — retry /trim"
        self.tick_manager.apply(ticket, action)
        remaining = len(self.tick_manager.layers(ticket))
        self.journal.log_note(
            f"mt5 trim: ticket {ticket} top layer closed "
            f"({action.volume} lots); {remaining} layer(s) remain")
        return (f"trimmed top layer ({action.volume} lots); "
                f"{remaining} layer(s) remain")

    # ------------------------------------------- telegram control surface
    # Mutating actions (pause / resume / trim / cancel pending / every
    # /set) run ONLY through request_confirm -> confirm_pending: the
    # button / command stages a token, the owner taps Confirm, and
    # only then does anything act.  Read-only commands never touch
    # these.  Live-trading permission is NOT here and never will be:
    # there is no key, command, or button for it.

    def get_live_config(self) -> dict:
        """Safe live config only (see mt5_config.snapshot_from)."""
        return self._config_store.snapshot_from(self.cfg)

    def _persist_live_config(self) -> None:
        self._config_store.save(self.get_live_config())

    def reload_live_config(self) -> None:
        """Re-read data/mt5_config.json and apply (e.g. hand-edited)."""
        self._config_store.apply_to(self.cfg)

    def set_live_config(self, change: dict) -> tuple[bool, str]:
        """Apply one parsed /set change (from mt5_config.parse_set):
        validate already happened — this applies + persists +
        journals.  Returns (ok, message)."""
        field = change.get("field")
        value = change.get("value")
        if field == "fixed_lots":
            self.cfg.fixed_lots = value
        elif field == "max_layers":
            self.cfg.layer_max_layers = int(value)
        elif field == "manipulation_filter":
            self.cfg.manipulation_filter = bool(value)
        elif field == "pending_expiry_bars":
            self.cfg.pending_expiry_bars = int(value)
        elif field == "playbook":
            name = change.get("name")
            disabled = set(self.cfg.disabled_playbooks or ())
            if value:
                disabled.discard(name)
            else:
                disabled.add(name)
            self.cfg.disabled_playbooks = tuple(sorted(disabled))
        else:
            return False, f"unknown config change {field!r} — not applied"
        self._persist_live_config()
        from .mt5_config import describe_change
        label = describe_change(change)
        self.journal.log_note(f"mt5 config: {label} (via Telegram)")
        return True, f"applied: {label}"

    def pause_engine(self) -> str:
        if self.cfg.paused:
            return "engine already paused"
        self.cfg.paused = True
        self._persist_live_config()
        self.journal.log_note("mt5: engine PAUSED via Telegram "
                              "(new signals blocked, management continues)")
        return ("engine PAUSED — new signals blocked; tick management "
                "(breakeven / trail / partial) continues")

    def resume_engine(self) -> str:
        if not self.cfg.paused:
            return "engine already running"
        self.cfg.paused = False
        self._persist_live_config()
        self.journal.log_note("mt5: engine RESUMED via Telegram")
        return "engine RESUMED — new signals live again"

    def cancel_all_pending(self) -> str:
        tickets = list(self._pending_orders)
        if not tickets:
            return "no pending orders — nothing to cancel"
        stats: dict = {}
        done = 0
        for ticket in tickets:
            if self._cancel_pending(
                    ticket, "telegram: cancel all pending", stats):
                done += 1
        return (f"cancelled {done}/{len(tickets)} pending order(s)"
                + ("" if done == len(tickets)
                   else " — retry /pending; a failed cancel retries "
                        "next tick"))

    def pending_orders_snapshot(self) -> list[dict]:
        """Resting pendings for /status / /pending (no secrets)."""
        out = []
        for state in self._pending_orders.values():
            out.append({"ticket": state.ticket, "kind": state.kind,
                        "direction": state.direction,
                        "level": state.level, "volume": state.volume,
                        "sl": state.sl, "playbook": state.playbook,
                        "expiry_bar": state.expiry_bar,
                        "alert_id": state.alert_id})
        return out

    # ------------------------------------------------- confirm handshake
    def request_confirm(self, action: str, arg=None) -> str:
        """Stage a mutating action; returns the Confirm token."""
        token = secrets.token_hex(4)
        self._confirm_tokens[token] = {"action": action, "arg": arg}
        return token

    def confirm_pending(self, token: str) -> tuple[bool, str]:
        """Execute a staged action (Confirm tap).  The token is
        single-use: consumed whether the action succeeds or not."""
        staged = self._confirm_tokens.pop(token, None)
        if staged is None:
            return False, ("confirmation expired — nothing acted. "
                           "Send the command again.")
        action, arg = staged["action"], staged.get("arg")
        try:
            if action == "pause":
                return True, self.pause_engine()
            if action == "resume":
                return True, self.resume_engine()
            if action == "trim":
                return True, self.trim_top_layer()
            if action == "cancel_pending":
                return True, self.cancel_all_pending()
            if action == "set":
                return self.set_live_config(arg or {})
            return False, f"unknown staged action {action!r} — not acted"
        except Exception as exc:
            return False, (f"{action} FAILED: {type(exc).__name__}: {exc}")

    def cancel_confirm(self, token: str) -> str:
        if self._confirm_tokens.pop(token, None) is None:
            return "nothing to cancel (confirmation already expired)"
        return "cancelled — nothing changed"

    # -------------------------------------------------------------- orders
    def _place_live_order(self, pb, sig, stats: dict,
                          verdict: dict | None = None,
                          entry_bar: int | None = None,
                          is_layer: bool = False,
                          layer_lots: float | None = None) -> None:
        """Setup-dependent order selection + placement.

        The order type is chosen by where price is vs the entry level
        when the signal fires (see ``select_order_type``): at the
        level with the trigger confirmed -> MARKET at the tick;
        entry behind price (pullback) -> PENDING LIMIT at the POI;
        entry ahead of price (breakout) -> PENDING STOP beyond the
        level; zone failed -> no order at all.  A filled pending
        order becomes a TickManager position exactly like a market
        fill; an unfilled one is tracked until it fills, cancels,
        or expires.

        ``entry_bar`` is the bar index the position entered on (the
        forming bar's index on the live clock); when omitted it
        defaults to the last closed H1 bar (legacy closed-bar path).
        ``is_layer``: the order adds a layer to the held position
        (fill -> ``_record_layer_fill``) instead of opening a new
        one; ``layer_lots`` is the pre-capped layer size from
        ``_try_layer``.
        """
        executor = self.executor
        spec = executor.symbol_spec(self.cfg.symbol)
        acct = self.feed.account_state()
        stop0 = TickManager.planned_stop(
            OrderRequest(symbol=self.cfg.symbol, direction=sig.direction,
                         volume=0.0, stop_loss=sig.invalidation,
                         atr=sig.atr))
        if is_layer:
            # Layer size was pre-capped by _try_layer (total risk cap).
            lots = float(layer_lots)
        elif self.cfg.fixed_lots is not None:
            lots = float(self.cfg.fixed_lots)
        else:
            sizing = size_position(balance=acct["equity"],
                                   risk_pct=self.cfg.risk_pct,
                                   entry=sig.entry_ref, stop=stop0,
                                   spec=spec)
            lots = sizing.lots
            if lots <= 0:
                self.journal.log_note(
                    f"mt5: sizing refused {pb.name}: {sizing.reason}")
                return
        # Selection: where is price vs the entry level NOW?
        price = float((self._last_tick or {}).get("mid")
                      or (verdict or {}).get("entry")
                      or sig.entry_ref)
        kind, level, reason = select_order_type(
            direction=int(sig.direction), price=price,
            entry_ref=float(sig.entry_ref),
            zone=(verdict or {}).get("zone"), atr=float(sig.atr))
        request = OrderRequest(
            symbol=self.cfg.symbol, direction=int(sig.direction),
            volume=lots, stop_loss=float(stop0),
            entry_ref=float(sig.entry_ref),
            take_profit=float(sig.targets[0]) if sig.targets else None,
            targets=[float(x) for x in (sig.targets or [])],
            atr=float(sig.atr), risk_money=0.0,
            alert_id=f"MT5-{pb.name}-{sig.index}",
            bar_index=int(sig.index), magic=self.cfg.magic,
            order_kind=kind,
            order_price=float(level) if kind != "market" else 0.0)
        # The stepper measures R on the BUFFERED stop: pass the
        # invalidation-view request for tracking so the buffer is not
        # applied twice (broker SL is already stop0).
        track_request = OrderRequest(
            symbol=self.cfg.symbol, direction=int(sig.direction),
            volume=lots, stop_loss=float(sig.invalidation),
            entry_ref=float(sig.entry_ref),
            targets=[float(x) for x in (sig.targets or [])],
            atr=float(sig.atr), bar_index=int(sig.index),
            alert_id=request.alert_id, magic=self.cfg.magic)
        if kind == "invalid":
            stats["invalidated"] = stats.get("invalidated", 0) + 1
            self._record_blocked(f"order:invalid: {reason}", pb.name)
            self.journal.log_note(
                f"mt5: order invalidated {pb.name}: {reason} — "
                f"no order placed")
            return
        if kind != "market":
            self._place_pending_order(
                pb, sig, verdict, kind, level, request, track_request,
                stats, entry_bar, is_layer=is_layer)
            return
        res = executor.place_order(request)
        if not res.ok:
            self.journal.log_note(
                f"mt5: order FAILED {pb.name}: {res.message}")
            return
        fill = res.price or sig.entry_ref
        if is_layer:
            self._record_layer_fill(pb.name, sig, fill=fill, lots=lots,
                                    stop0=stop0, stats=stats)
            return
        h1 = self._stack.frames["H1"]
        self.tick_manager.open(ticket=int(res.ticket),
                               request=track_request, fill_price=fill,
                               entry_bar=len(h1) - 1 if entry_bar is None
                               else entry_bar, broker_sl=stop0)
        stats["executed"] = stats.get("executed", 0) + 1
        # Open record: without it, /trades /report /harvest cannot see
        # this fill (they join close -> open by alert_id).  Stop here is
        # the invalidation the R accounting tracks on; the broker SL
        # is the buffered stop0 (see the fill note below).
        try:
            risk_money = spec.money_for_move(
                abs(fill - float(sig.invalidation)), lots)
        except Exception:
            risk_money = 0.0
        self.journal.log_open(
            alert_id=request.alert_id, instrument=self.cfg.symbol,
            mode="MT5", playbook=pb.name, phase="live",
            direction=int(sig.direction), score=float(sig.score),
            source="mt5", ticket=int(res.ticket), entry=float(fill),
            stop=float(sig.invalidation), volume=float(lots),
            risk_money=float(risk_money),
            factors=list(getattr(sig, "factors", []) or []))
        self.journal.log_note(
            f"mt5 fill: ticket {res.ticket} {pb.name} "
            f"{'+' if sig.direction > 0 else '-'}1 {lots} lots @ {fill:.2f} "
            f"SL {stop0:.2f}")
        # Fill chart: same levels + the actual fill price, saved
        # alongside the signal chart and attached to the fill alert.
        png, _path = self._build_chart(
            sig, verdict, alert_id=request.alert_id, playbook=pb.name,
            fill_price=fill, suffix="-fill")
        self._send_alert(
            f"MT5 FILL {pb.name} {'+' if sig.direction > 0 else '-'}1 "
            f"{lots} lots @ {fill:.2f} SL {stop0:.2f} "
            f"(ticket {res.ticket})",
            chart_png=png)

    # ------------------------------------------------------- pending orders
    # A pending LIMIT / STOP is a broker order waiting for the market:
    # the market comes to it (limit) or breaks through it (stop).
    # The engine tracks every pending it places or adopts until it
    # fills, cancels, or expires — NEVER a stale pending.

    def _current_h1_bar(self) -> int:
        """Forming-H1-bar index (len of the closed H1 frame)."""
        if self._stack is None:
            return 0
        h1 = self._stack.frames.get("H1")
        return len(h1) if h1 is not None else 0

    def _track_pending_order(self, order: PendingOrder, *,
                             playbook: str, zone: dict | None,
                             invalidation: float, placed_bar: int,
                             sig, verdict, track_request,
                             is_layer: bool = False) -> PendingOrderState:
        """Register a broker pending order in engine tracking."""
        state = PendingOrderState(
            ticket=int(order.ticket), kind=str(order.kind),
            direction=int(order.direction), level=float(order.price),
            volume=float(order.volume),
            sl=float(order.sl) if order.sl is not None
            else float(invalidation),
            tp=float(order.tp) if order.tp is not None else None,
            alert_id=str(order.alert_id), playbook=str(playbook),
            zone=zone, invalidation=float(invalidation),
            placed_bar=int(placed_bar),
            expiry_bar=int(placed_bar) + self.cfg.pending_expiry_bars,
            track_request=track_request, sig=sig, verdict=verdict,
            is_layer=bool(is_layer))
        self._pending_orders[state.ticket] = state
        return state

    def _place_pending_order(self, pb, sig, verdict, kind: str,
                             level: float, request: OrderRequest,
                             track_request: OrderRequest,
                             stats: dict,
                             entry_bar: int | None,
                             is_layer: bool = False) -> None:
        """Place a pending LIMIT / STOP and start tracking it."""
        # One-position rule: a pending order COUNTS — never a second
        # pending while one pending is active, and never a base
        # pending while a position is active.  A pending LAYER is the
        # exception: it rides on the held position.
        if (not is_layer and self.tick_manager.tickets()) \
                or self._pending_orders:
            self.journal.log_note(
                f"mt5: pending {kind} NOT placed {pb.name}: already "
                f"holding/pending")
            return
        res = self.executor.place_order(request)
        if not res.ok:
            self.journal.log_note(
                f"mt5: pending {kind} FAILED {pb.name}: {res.message}")
            return
        placed_bar = entry_bar if entry_bar is not None \
            else self._current_h1_bar()
        order = PendingOrder(
            ticket=int(res.ticket), symbol=self.cfg.symbol, kind=kind,
            direction=int(sig.direction), price=float(level),
            volume=float(request.volume), sl=float(request.stop_loss),
            tp=float(request.take_profit)
            if request.take_profit else None,
            magic=self.cfg.magic, alert_id=request.alert_id)
        state = self._track_pending_order(
            order, playbook=pb.name, zone=(verdict or {}).get("zone"),
            invalidation=float(sig.invalidation),
            placed_bar=placed_bar, sig=sig, verdict=verdict,
            track_request=track_request, is_layer=is_layer)
        stats["pending"] = stats.get("pending", 0) + 1
        self.journal.log_note(
            f"mt5 pending: {kind.upper()} {pb.name} "
            f"{'+' if sig.direction > 0 else '-'}1 "
            f"{request.volume} lots @ {level:.2f} "
            f"(ticket {res.ticket}) SL {request.stop_loss:.2f} "
            f"expires H1 bar {state.expiry_bar}")
        # Pending chart + alert, marked PENDING: the entry line IS
        # the pending level (sig.entry_ref), the fill line comes
        # only when the market reaches it.
        png, _path = self._build_chart(
            sig, verdict, alert_id=request.alert_id, playbook=pb.name,
            suffix="-pending")
        self._send_alert(
            f"MT5 PENDING {kind.upper()} {pb.name} "
            f"{'+' if sig.direction > 0 else '-'}1 "
            f"{request.volume} lots @ {level:.2f} "
            f"SL {request.stop_loss:.2f} (ticket {res.ticket})",
            chart_png=png)

    def _check_pending_orders(self, price: float, stats: dict) -> None:
        """Tick-level pending invalidation: price beyond the adverse
        zone edge or the invalidation level kills the resting order
        NOW (the zone failed — the setup's premise is gone)."""
        for ticket, state in list(self._pending_orders.items()):
            zone = state.zone
            if state.direction > 0:
                if zone is not None and price < float(zone["bottom"]):
                    self._cancel_pending(
                        ticket,
                        f"invalidated: price {price:.2f} below POI "
                        f"bottom {float(zone['bottom']):.2f}", stats)
                elif price <= state.invalidation:
                    self._cancel_pending(
                        ticket,
                        f"invalidated: price {price:.2f} through "
                        f"invalidation {state.invalidation:.2f}", stats)
            else:
                if zone is not None and price > float(zone["top"]):
                    self._cancel_pending(
                        ticket,
                        f"invalidated: price {price:.2f} above POI "
                        f"top {float(zone['top']):.2f}", stats)
                elif price >= state.invalidation:
                    self._cancel_pending(
                        ticket,
                        f"invalidated: price {price:.2f} through "
                        f"invalidation {state.invalidation:.2f}", stats)

    def _cancel_pending(self, ticket: int, reason: str,
                        stats: dict) -> bool:
        """Cancel one tracked pending order.  A failed cancel is
        retried on the next tick (same philosophy as failed SL
        moves) — EXCEPT when the order already filled between the
        check and the cancel, in which case the fill is recorded
        instead (a position with the same alert_id proves it)."""
        state = self._pending_orders.get(ticket)
        if state is None:
            return False
        res = self.executor.cancel_pending(ticket)
        if res.ok:
            del self._pending_orders[ticket]
            stats["pending_cancelled"] = stats.get(
                "pending_cancelled", 0) + 1
            self.journal.log_note(
                f"mt5 pending cancelled: ticket {ticket} "
                f"{state.kind} @ {state.level:.2f} ({reason})")
            return True
        # Cancel failed: did it fill in the meantime?
        try:
            live = self.executor.positions()
        except Exception:
            live = []
        matched = [p for p in live
                   if p.alert_id == state.alert_id
                   and p.magic == self.cfg.magic]
        if matched:
            self._record_pending_fill(state, matched[0], stats)
            return True
        self.journal.log_note(
            f"mt5 pending cancel FAILED ticket {ticket} ({reason}): "
            f"{res.message} — retry next tick")
        return False

    def _record_pending_fill(self, state: PendingOrderState,
                             pos, stats: dict) -> None:
        """A pending order filled: TickManager tracking starts,
        exactly like a market fill (same open record, same fill
        chart/alert — the entry is the broker's fill price).  A
        pending LAYER fills onto the held position instead (the
        layer entry is the pending price — the deal price); if the
        position closed before the pending filled, the layer falls
        back to opening a fresh base position.
        """
        self._pending_orders.pop(state.ticket, None)
        if state.is_layer and self.tick_manager.tickets():
            self._record_layer_fill(
                state.playbook or "pending", state.sig,
                fill=float(state.level), lots=float(state.volume),
                stop0=float(state.sl), stats=stats)
            return
        track = state.track_request
        if track is None:
            track = OrderRequest(
                symbol=self.cfg.symbol, direction=int(state.direction),
                volume=float(state.volume),
                stop_loss=float(state.invalidation),
                entry_ref=float(state.level),
                alert_id=state.alert_id, magic=self.cfg.magic)
        self.tick_manager.open(ticket=int(pos.ticket), request=track,
                               fill_price=float(pos.entry_price),
                               entry_bar=int(state.placed_bar),
                               broker_sl=float(state.sl))
        stats["executed"] = stats.get("executed", 0) + 1
        stats["pending_filled"] = stats.get("pending_filled", 0) + 1
        sig = getattr(state, "sig", None)
        try:
            spec = self.executor.symbol_spec(self.cfg.symbol)
            risk_money = spec.money_for_move(
                abs(float(pos.entry_price) - float(state.invalidation)),
                float(state.volume))
        except Exception:
            risk_money = 0.0
        self.journal.log_open(
            alert_id=state.alert_id, instrument=self.cfg.symbol,
            mode="MT5", playbook=state.playbook or "pending",
            phase="live", direction=int(state.direction),
            score=float(getattr(sig, "score", 0.0) or 0.0)
            if sig is not None else 0.0,
            source="mt5", ticket=int(pos.ticket),
            entry=float(pos.entry_price),
            stop=float(state.invalidation),
            volume=float(state.volume), risk_money=float(risk_money),
            factors=list(getattr(sig, "factors", []) or [])
            if sig is not None else [])
        self.journal.log_note(
            f"mt5 fill: ticket {pos.ticket} "
            f"{state.playbook or 'pending'} "
            f"{'+' if state.direction > 0 else '-'}1 "
            f"{state.volume} lots @ {float(pos.entry_price):.2f} "
            f"SL {state.sl:.2f} (PENDING {state.kind.upper()} filled)")
        # Fill chart when the signal is known (placed this run);
        # adopted pendings (restart) have no signal — text only.
        png = None
        if sig is not None and getattr(state, "verdict", None) is not None:
            png, _path = self._build_chart(
                sig, state.verdict, alert_id=state.alert_id,
                playbook=state.playbook, fill_price=float(pos.entry_price),
                suffix="-fill")
        self._send_alert(
            f"MT5 FILL {state.playbook or 'pending'} "
            f"{'+' if state.direction > 0 else '-'}1 "
            f"{state.volume} lots @ {float(pos.entry_price):.2f} "
            f"SL {state.sl:.2f} (ticket {pos.ticket})",
            chart_png=png)

    def _adopt_existing_pending(self, executor: MT5Executor) -> None:
        """Adopt broker pending orders with our magic (restart sync)."""
        try:
            live = executor.pending_orders(self.cfg.symbol)
        except Exception as exc:
            self.journal.log_note(
                f"mt5: startup pending orders failed: {exc}")
            return
        for order in live:
            if order.magic != self.cfg.magic:
                continue
            if order.ticket in self._pending_orders:
                continue
            self._track_pending_order(
                order, playbook="", zone=None,
                invalidation=order.sl or order.price,
                placed_bar=self._current_h1_bar(),
                sig=None, verdict=None, track_request=None)
            self.journal.log_note(
                f"mt5: adopted pending {order.kind} ticket "
                f"{order.ticket} (@{order.price:.2f} "
                f"SL {order.sl})")

    def _adopt_existing(self, executor: MT5Executor) -> None:
        """Adopt our-magic positions already open at startup."""
        try:
            live = executor.positions(self.cfg.symbol)
        except Exception as exc:
            self.journal.log_note(f"mt5: startup positions failed: {exc}")
            return
        for pos in live:
            if pos.magic != self.cfg.magic:
                continue
            self.tick_manager.adopt(
                ticket=pos.ticket, symbol=pos.symbol,
                direction=pos.direction, volume=pos.volume,
                entry_price=pos.entry_price, sl=pos.sl,
                alert_id=pos.alert_id)
            self.journal.log_note(
                f"mt5: adopted open ticket {pos.ticket} "
                f"(@{pos.entry_price:.2f} SL {pos.sl})")

    # ------------------------------------------------------------ helpers
    def _verdict_for_bar(self, h1: pd.DataFrame, t: int,
                         states) -> dict | None:
        """Hierarchy verdict at H1 bar ``t`` + MT5 volume annotation."""
        try:
            amd_phase = amd_via = amd_conf = None
            st = states[t] if 0 <= t < len(states) else None
            feats = getattr(st, "features", None) if st is not None else None
            ph = getattr(feats, "phase", None) if feats is not None else None
            if ph is not None:
                amd_phase = getattr(ph, "phase", None) or "unknown"
                amd_via = getattr(ph, "via", None)
                amd_conf = getattr(ph, "confidence", None)
            verdict = evaluate_hierarchy(
                self._stack, asof=h1.index[t], amd_phase=amd_phase,
                amd_via=amd_via, amd_confidence=amd_conf,
                qm_touch=self._qm_touch, swing_n=3)
            # MT5 volume on the trigger frame (closed bars only):
            # the causal H1 slice, so the forming bar never counts.
            sliced = h1[h1.index <= h1.index[t]]
            verdict["volume"] = volume_features(sliced)
            return verdict
        except Exception as exc:
            self.journal.log_note(
                f"mt5: hierarchy evaluation failed at bar {t}: {exc}")
            return None

    def _advance_qm(self, m5_df) -> dict | None:
        """Latest QM M5 touch (annotation only — same as paper loop)."""
        try:
            if m5_df is None or len(m5_df) < 30:
                return None
            pb = QMM5Playbook()
            ctx = EngineContext(df=m5_df, states=[], years=0.0,
                                instrument=self.cfg.symbol)
            cands = pb.candidates(ctx)
            if not cands:
                return None
            latest = max(cands, key=lambda c: c.signal_index)
            sig = latest.outcome.signal
            lc = dict(getattr(sig, "live_context", None) or {})
            return {"direction": int(latest.direction),
                    "touch_ts": m5_df.index[latest.signal_index],
                    "touch_index": int(latest.signal_index),
                    "qm_level": lc.get("qm_level"),
                    "head": lc.get("head"),
                    "neckline": lc.get("neckline"),
                    "tags": lc.get("tags", [])}
        except Exception as exc:
            self.journal.log_note(
                f"mt5: QM M5 scan failed (annotation only): {exc}")
            return None

    # ------------------------------------------------- alerts + charts
    def _build_chart(self, sig, verdict: dict | None, *,
                     alert_id: str, playbook: str = "",
                     fill_price: float | None = None,
                     suffix: str = "") -> tuple[bytes | None, Path | None]:
        """Chart PNG for a signal/fill: the WHOLE hierarchy stack
        (H4 structure / H1 trigger / entry-TF entry) + levels + verdict.

        Never breaks the signal path: on any failure returns
        (None, None) and logs a note; the text alert still goes out.
        """
        try:
            frames = self._stack.frames if self._stack else {}
            h1 = frames.get("H1")
            if h1 is None or len(h1) == 0:
                return None, None
            zone = (verdict or {}).get("zone")
            entry_clock = (verdict or {}).get("entry_clock") or "M15"
            png = render_mt5_signal(
                h1, zone=zone, entry=float(sig.entry_ref),
                stop=float(sig.invalidation),
                targets=[float(x) for x in (sig.targets or [])],
                direction=int(sig.direction), verdict=verdict,
                playbook=playbook,
                score=float(getattr(sig, "score", 0.0) or 0.0),
                instrument=self.cfg.symbol, fill_price=fill_price,
                h4=frames.get("H4"),
                entry_frame=frames.get(entry_clock),
                entry_clock=entry_clock,
                style=self.cfg.chart_style)
            path = self._charts_dir / f"{alert_id}{suffix}.png"
            path.write_bytes(png)
            return png, path
        except Exception as exc:
            self.journal.log_note(
                f"mt5: chart build failed for {alert_id}: {exc}")
            return None, None

    def _send_startup_ping(self, acct: dict) -> None:
        """One Telegram 'engine started' ping per process start.

        Fires after feed connect + runner start (feed, executor,
        warm start, adoption all done).  Sending is INDEPENDENT of
        command polling (getUpdates): the ping goes out even with
        --no-poll-commands — the owner must be able to tell the
        engine is alive while the OANDA loop still owns getUpdates.
        Never raises: no senders configured -> journal note only;
        a send failure is journaled and start continues.
        """
        if self._startup_ping_sent:
            return
        self._startup_ping_sent = True
        if not self.senders:
            self.journal.log_note(
                "mt5: startup ping skipped (no Telegram sender "
                "configured)")
            return
        try:
            mode = acct.get("trade_mode")
            mode_txt = ("demo" if mode == 0 else
                        "live" if mode == 2 else f"mode={mode}")
            executing = [p for p in self.cfg.execute_playbooks
                         if p not in self.cfg.disabled_playbooks]
            state = "PAUSED" if self.cfg.paused else "RUNNING"
            polling = ("ON" if self.cfg.poll_commands else
                       "OFF (overlap — OANDA loop owns getUpdates)")
            text = (
                "MT5 engine started\n"
                f"symbol: {self.cfg.symbol}\n"
                f"trade_mode: {mode} ({mode_txt})\n"
                f"equity: {acct.get('equity')}\n"
                f"playbooks: {', '.join(executing) if executing else '(none)'}\n"
                f"state: {state}\n"
                f"command polling: {polling}"
            )
            self._send_alert(text)
        except Exception as exc:
            self.journal.log_note(f"mt5: startup ping failed: {exc}")

    def _send_alert(self, text: str, *,
                    chart_png: bytes | None = None) -> None:
        """Send a text alert, with the chart photo when one is attached.

        Sender contract (in preference order): ``send_photo(png,
        caption)`` (TelegramSender), ``send_text(text)``, legacy
        ``send(text)`` duck-type.  A sender failure is journaled,
        never raised into the tick loop.
        """
        for sender in self.senders:
            try:
                if chart_png and hasattr(sender, "send_photo"):
                    res = sender.send_photo(chart_png, text)
                    if not getattr(res, "delivered", False):
                        self.journal.log_note(
                            f"mt5: alert photo send failed: {res.detail}")
                elif hasattr(sender, "send_text"):
                    res = sender.send_text(text)
                    if not getattr(res, "delivered", False):
                        self.journal.log_note(
                            f"mt5: alert send failed: {res.detail}")
                else:
                    sender.send(text)
            except Exception as exc:
                self.journal.log_note(f"mt5: alert send failed: {exc}")

    # ------------------------------------------------- telegram commands
    def _get_command_router(self):
        """Owner-only MT5 command router (lazy; MT5 registry)."""
        if self._command_router is None:
            self._command_router = mt5_router(
                owner_chat_id=os.environ.get("GS_TELEGRAM_CHAT_ID"))
        return self._command_router

    def _command_context(self) -> CommandContext:
        """The read-mostly world one MT5 command is allowed to see."""
        live_positions: list[dict] = []
        try:
            if self.executor is not None:
                for p in self.executor.positions(self.cfg.symbol):
                    ticket = getattr(p, "ticket", None)
                    layers = []
                    try:
                        for ly in self.tick_manager.layers(ticket):
                            layers.append({
                                "entry": getattr(ly, "entry", 0),
                                "volume": getattr(ly, "volume", 0),
                                "sl": getattr(ly, "sl", 0),
                                "playbook": getattr(ly, "playbook", ""),
                            })
                    except Exception:
                        layers = []
                    live_positions.append({
                        "ticket": ticket,
                        "symbol": getattr(p, "symbol", self.cfg.symbol),
                        "direction": getattr(p, "direction", 0),
                        "entry_price": getattr(p, "entry_price", 0),
                        "sl": getattr(p, "sl", 0),
                        "volume": getattr(p, "volume", 0),
                        "alert_id": getattr(p, "alert_id", ""),
                        "playbook": "",
                        "layers": layers,
                    })
        except Exception:
            live_positions = []
        h1 = self._stack.frames.get("H1") if self._stack else None
        last_bar = None
        if h1 is not None and len(h1):
            last_bar = h1.index[-1]
        return CommandContext(
            journal=self.journal,
            journal_path=self.cfg.journal_path,
            paper=None,
            feed_status=(self.feed.poll_status() if self.feed else None),
            last_price=(self._last_tick or {}).get("mid"),
            last_bar_time=(last_bar.isoformat()
                           if last_bar is not None
                           and hasattr(last_bar, "isoformat")
                           else last_bar),
            reports_dir=self.cfg.journal_path.parent / "reports",
            feed=self.feed,
            live_positions=live_positions,
            trim_callback=self.trim_top_layer,
            mt5_control=self,
            pending_orders=self.pending_orders_snapshot(),
        )

    def _reply_command(self, result) -> None:
        """Send one command result back (document or text)."""
        if result is None:
            return
        for sender in self.senders:
            try:
                if result.document is not None:
                    if hasattr(sender, "send_document"):
                        sender.send_document(result.document,
                                             caption=result.caption)
                    else:
                        sender.send(f"{result.caption or ''} "
                                    f"({result.document})")
                elif result.text is not None:
                    if hasattr(sender, "send_text"):
                        try:
                            sender.send_text(
                                result.text, silent=True,
                                reply_markup=result.reply_markup)
                        except TypeError:
                            # Legacy sender without reply_markup support.
                            sender.send_text(result.text, silent=True)
                    else:
                        sender.send(result.text)
            except Exception as exc:
                self.journal.log_note(
                    f"mt5: command reply failed: {exc}")

    def _tg_api(self, method: str, payload: dict) -> dict:
        """One Bot API call; returns the parsed JSON or {}.

        Transport: env token when configured, otherwise the telegram
        skill's tg-call CLI (vault-backed).  No token, no CLI, no poll.
        """
        import json as _json
        import subprocess as _sp

        token = os.environ.get("GS_TELEGRAM_BOT_TOKEN")
        for cand in ("~/workspace/custom-connectors/custom.legacy-telegram/bin/tg-call",
                     "~/workspace/skills/telegram/bin/tg-call"):
            tg_call = os.path.expanduser(cand)
            if os.path.isfile(tg_call):
                break
        else:
            tg_call = ""
        use_skill = not token and bool(tg_call)
        if not token and not use_skill:
            return {}
        try:
            if token:
                import requests
                url = f"https://api.telegram.org/bot{token}/{method}"
                if method == "getUpdates":
                    r = requests.get(url, params=payload, timeout=10)
                else:
                    r = requests.post(url, data=payload, timeout=10)
                return r.json()
            proc = _sp.run(
                [tg_call, "--method", method,
                 "--data", _json.dumps(payload)],
                capture_output=True, text=True, timeout=40)
            return _json.loads(proc.stdout or "{}")
        except Exception:
            return {}

    def _poll_telegram_commands(self) -> None:
        """Poll getUpdates and answer owner commands (MT5 registry).

        ON by default: this engine OWNS getUpdates.  The OANDA loop's
        poller must be OFF at cutover (GS_TELEGRAM_POLL=0) — one
        consumer only.  First poll only syncs the offset (drains the
        backlog without answering stale commands); later polls answer.
        A poll failure is silent — the tick loop never stops for it.
        """
        if not self.cfg.poll_commands:
            return
        now = time.monotonic()
        if now - self._last_poll < self.cfg.poll_interval_seconds:
            return
        self._last_poll = now
        if not self.senders:
            return  # nowhere to answer; stay quiet
        try:
            updates = self._tg_api(
                "getUpdates",
                {"offset": self._tg_offset, "timeout": 0}).get("result", [])
        except Exception:
            return
        if not self._tg_synced:
            # Drain the backlog: advance past everything seen so far
            # WITHOUT answering (they predate this engine's ownership).
            for upd in updates:
                self._tg_offset = int(upd["update_id"]) + 1
            if updates:
                self._tg_api("getUpdates",
                             {"offset": self._tg_offset, "timeout": 0})
            self._tg_synced = True
            return
        router = self._get_command_router()
        ctx = self._command_context()
        for upd in updates:
            self._tg_offset = int(upd["update_id"]) + 1
            if upd.get("callback_query") is not None:
                # Button taps share THIS poller (one getUpdates
                # consumer only — never a second poller).
                self._handle_callback_query(upd)
                continue
            try:
                result, _chat_id = router.dispatch(upd, ctx)
            except Exception as exc:
                self.journal.log_note(
                    f"mt5: command dispatch failed: {exc}")
                continue
            if result is not None:
                self._reply_command(result)
        if updates:
            # Confirm the drained offset server-side.
            self._tg_api("getUpdates",
                         {"offset": self._tg_offset, "timeout": 0})

    # -------------------------------------------------- callback_query
    def _handle_callback_query(self, upd: dict) -> None:
        """One button tap: owner-gated, answered, acted on.

        Flow: a tap on Pause / Resume / Trim / Cancel pending only
        STAGES a confirmation (a "confirm?" message with Confirm /
        Cancel buttons) — the action runs only on the Confirm tap.
        Refresh / Config re-render the message in place.  A tap from
        anyone but the owner is refused and acts on nothing.
        """
        import json as _json

        query = upd.get("callback_query") or {}
        query_id = query.get("id", "")
        router = self._get_command_router()

        def _answer(text: str) -> None:
            try:
                self._tg_api("answerCallbackQuery",
                             {"callback_query_id": query_id,
                              "text": (text or "")[:200]})
            except Exception:
                pass

        if not is_owner_callback(upd, router.owner_chat_id):
            _answer("not authorized")
            return
        parsed = parse_mt5_callback(query.get("data", ""))
        if parsed is None:
            _answer("")  # not ours (e.g. a legacy alert button) — ignore
            return
        action, arg = parsed
        msg = query.get("message") or {}
        chat = msg.get("chat") or {}
        chat_id = chat.get("id") or query.get("from", {}).get("id")
        message_id = msg.get("message_id")
        try:
            if action in ("pause", "resume", "trim", "cancel_pending"):
                labels = {"pause": "⏸ Pause engine",
                          "resume": "▶ Resume engine",
                          "trim": "✂️ Trim top layer",
                          "cancel_pending": "❌ Cancel ALL pending orders"}
                token = self.request_confirm(action)
                self._tg_api("sendMessage", {
                    "chat_id": chat_id,
                    "text": f"{labels[action]} — tap Confirm to act, "
                            f"Cancel to abort.",
                    "reply_markup": _json.dumps(
                        mt5_confirm_keyboard(token))})
                _answer(f"Confirm to {action}…")
                return
            if action == "confirm":
                ok, text = self.confirm_pending(arg or "")
                if message_id is not None:
                    self._tg_api("editMessageText", {
                        "chat_id": chat_id, "message_id": message_id,
                        "text": text,
                        "reply_markup": _json.dumps(
                            mt5_control_keyboard())})
                _answer(text)
                return
            if action == "cancel":
                text = self.cancel_confirm(arg or "")
                if message_id is not None:
                    self._tg_api("editMessageText", {
                        "chat_id": chat_id, "message_id": message_id,
                        "text": text})
                _answer(text)
                return
            if action in ("refresh", "status"):
                from .mt5_commands import _mt5_status
                result = _mt5_status(self._command_context(), [])
                if message_id is not None:
                    self._tg_api("editMessageText", {
                        "chat_id": chat_id, "message_id": message_id,
                        "text": result.text or "",
                        "reply_markup": _json.dumps(
                            mt5_control_keyboard())})
                _answer("refreshed")
                return
            if action == "config":
                from .mt5_commands import _mt5_config
                result = _mt5_config(self._command_context(), [])
                if message_id is not None:
                    self._tg_api("editMessageText", {
                        "chat_id": chat_id, "message_id": message_id,
                        "text": result.text or "",
                        "reply_markup": _json.dumps(
                            mt5_control_keyboard())})
                _answer("config")
                return
            _answer("unknown button")
        except Exception as exc:
            self.journal.log_note(
                f"mt5: callback handling failed: {exc}")
            _answer("button failed — try the command again")
