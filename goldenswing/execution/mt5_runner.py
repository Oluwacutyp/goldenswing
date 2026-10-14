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
  OFF) -> score (with the MT5 volume bonus) -> market order via
  MT5Executor -> TickManager tracking.
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
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..alerts.commands import CommandContext
from ..backtest.data import REPO_ROOT
from ..charts.engine import render_mt5_signal
from ..confluence import ConfluenceEngine
from ..features.engine import read_and_featurize
from ..feed.tick_log import TickLogger
from ..live.hierarchy import TF_MINUTES, TimeframeStack, evaluate_hierarchy
from ..live.volume import volume_features
from ..playbooks import (FVGLimitPlaybook, FVGMidlinePlaybook,
                         QMM5Playbook, SweepReversalPlaybook,
                         TrendlineRetestPlaybook)
from ..playbooks.base import EngineContext
from .executor import OrderRequest
from .learning import TradeJournal
from .mt5 import MT5Executor
from .mt5_commands import mt5_router
from .risk import RiskConfig, RiskGuards, size_position
from .tick_manager import TickManager
from ..feed.mt5 import TRADE_MODE_REAL, MT5Feed, TickPump

DATA_DIR = REPO_ROOT / "data"


@dataclass
class MT5RunnerConfig:
    symbol: str = "XAUUSD"
    risk_pct: float = 0.01
    fixed_lots: float | None = None  # when set, every trade uses it
    magic: int = 260101
    execute_playbooks: tuple = ("fvg_midline", "trendline_retest")
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
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg


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
        self._last_processed: dict[str, pd.Timestamp] = {}
        self._last_sig_bar: dict[str, int] = {}
        self._tick_count = 0
        self._last_tick: dict | None = None
        self._stopped = False
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

    # ------------------------------------------------------------ wiring
    def _get_playbooks(self) -> list:
        if self._playbooks is None:
            self._playbooks = [SweepReversalPlaybook(),
                               FVGMidlinePlaybook(),
                               TrendlineRetestPlaybook(),
                               FVGLimitPlaybook(),
                               QMM5Playbook()]
        return self._playbooks

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
        self.feed = feed
        self.executor = executor
        self.journal.log_note(
            "mt5 runner: started "
            f"({', '.join(f'{tf}:{len(self._stack.frames.get(tf, []))}' for tf in self._stack.frames)}) "
            f"account login={acct.get('login')} equity={acct.get('equity')} "
            f"trade_mode={acct.get('trade_mode')}")

        # Fast-forward playbook streaming state over history; past
        # signals are discarded — only the latest close trades.
        states = read_and_featurize(h1, None)
        for pb in self._get_playbooks():
            for t2 in range(len(h1)):
                pb.live_signal(states, t2, h1)
        # Adopt positions the broker already holds with our magic
        # (restart sync — MT5 is the source of truth for open trades).
        self._adopt_existing(executor)

        stats: dict = {"ticks": 0, "bars": 0, "signals": 0,
                       "executed": 0, "shadow": 0, "gated": 0,
                       "hierarchy_blocked": 0, "tick_actions": 0,
                       "reconciled_closed": 0}
        pump = TickPump(feed, interval=self.cfg.tick_interval)
        try:
            pump.run(self._on_tick(stats), max_ticks=max_ticks)
        finally:
            self.journal.log_note(
                f"mt5 runner: stopped: {stats['ticks']} ticks, "
                f"{stats['bars']} H1 bars, {stats['signals']} signals, "
                f"{stats['executed']} executed, {stats['shadow']} shadow, "
                f"{stats['hierarchy_blocked']} hierarchy-blocked")
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
            if self._tick_count % self.cfg.reconcile_every_ticks == 0:
                self._reconcile(price, stats)
            self._refresh_bars(tick, stats)
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
        for ticket, pos in live.items():
            if ticket not in self.tick_manager.tickets() \
                    and pos.magic == self.cfg.magic:
                self.tick_manager.adopt(
                    ticket=ticket, symbol=pos.symbol,
                    direction=pos.direction, volume=pos.volume,
                    entry_price=pos.entry_price, sl=pos.sl,
                    alert_id=pos.alert_id)
                self.journal.log_note(
                    f"mt5 reconcile: adopted ticket {ticket} "
                    f"({'+' if pos.direction > 0 else '-'}1 "
                    f"@{pos.entry_price:.2f})")

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

        New closed H1 bars run the signal path.  No poll timer: the
        tick itself is the clock.
        """
        if self._stack is None:
            return
        tick_time = pd.Timestamp(tick["time"])
        cursors = {t: self._stack.cursors.get(t)
                   for t in self._stack.frames}
        for tf in list(self._stack.frames):
            closed = MT5Feed.closed_bar_close(tf, tick_time)
            if closed <= self._last_processed.get(tf, closed):
                continue
            fresh = self.feed.fetch_new_all(cursors)
            new = fresh.get(tf)
            if new is not None and len(new):
                accepted, _rej = self._stack.append(tf, new)
                if accepted:
                    self._last_processed[tf] = (
                        self._stack.cursors[tf]
                        + pd.Timedelta(minutes=TF_MINUTES[tf]))
                    if tf == "M5":
                        self._qm_touch = self._advance_qm(
                            self._stack.frames.get("M5"))
                    if tf == "H1":
                        self._process_new_h1(new, stats)

    def _process_new_h1(self, new: pd.DataFrame, stats: dict) -> None:
        """Each new closed H1 bar through the signal path."""
        h1 = self._stack.frames["H1"]
        states = read_and_featurize(h1, None)
        years = (h1.index[-1] - h1.index[0]).total_seconds() \
            / 86400.0 / 365.25
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
        for t in range(len(h1) - len(new), len(h1)):
            stats["bars"] += 1
            self._process_closed_bar(h1, t, states, years, stats)

    def _process_closed_bar(self, h1: pd.DataFrame, t: int,
                            states, years: float, stats: dict) -> None:
        """One closed H1 bar: verdict -> playbooks -> gate -> order.

        Mirrors the paper loop's ``_paper_loop_bar`` — same playbooks,
        same confluence gate, same hierarchy — but fills go to the
        MT5 executor at market, and the position is tracked by the
        TickManager from the fill tick on.
        """
        if self.tick_manager.tickets():
            return  # one position at a time, like the paper loop
        verdict = self._verdict_for_bar(h1, t, states)
        if verdict is None:
            self.journal.log_note(
                "mt5: hierarchy evaluation failed — all signals blocked")
            return
        ctx = EngineContext(df=h1, states=states, years=years,
                            instrument=self.cfg.symbol)
        engine = self._get_engine()
        for pb in self._get_playbooks():
            if not pb.live:
                continue
            sigs = pb.live_signal(states, t, h1) or []
            for sig in sigs:
                if sig.score < self.cfg.min_score:
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
                    self.journal.log_note(
                        f"mt5: {pb.name} signal gated: {reason}")
                    continue
                conf, notes = engine.score_live_signal(
                    pb, sig, ctx, hierarchy_verdict=verdict)
                self._last_sig_bar[pb.name] = sig.index
                stats["signals"] += 1
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
                if auto:
                    self._place_live_order(pb, sig, stats, verdict=verdict)

    # -------------------------------------------------------------- orders
    def _place_live_order(self, pb, sig, stats: dict,
                          verdict: dict | None = None) -> None:
        """Market order at the current tick + TickManager tracking."""
        executor = self.executor
        spec = executor.symbol_spec(self.cfg.symbol)
        acct = self.feed.account_state()
        stop0 = TickManager.planned_stop(
            OrderRequest(symbol=self.cfg.symbol, direction=sig.direction,
                         volume=0.0, stop_loss=sig.invalidation,
                         atr=sig.atr))
        if self.cfg.fixed_lots is not None:
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
        request = OrderRequest(
            symbol=self.cfg.symbol, direction=int(sig.direction),
            volume=lots, stop_loss=float(stop0),
            entry_ref=float(sig.entry_ref),
            take_profit=float(sig.targets[0]) if sig.targets else None,
            targets=[float(x) for x in (sig.targets or [])],
            atr=float(sig.atr), risk_money=0.0,
            alert_id=f"MT5-{pb.name}-{sig.index}",
            bar_index=int(sig.index), magic=self.cfg.magic)
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
        res = executor.place_order(request)
        if not res.ok:
            self.journal.log_note(
                f"mt5: order FAILED {pb.name}: {res.message}")
            return
        fill = res.price or sig.entry_ref
        h1 = self._stack.frames["H1"]
        self.tick_manager.open(ticket=int(res.ticket),
                               request=track_request, fill_price=fill,
                               entry_bar=len(h1) - 1, broker_sl=stop0)
        stats["executed"] += 1
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
                    live_positions.append({
                        "ticket": getattr(p, "ticket", None),
                        "symbol": getattr(p, "symbol", self.cfg.symbol),
                        "direction": getattr(p, "direction", 0),
                        "entry_price": getattr(p, "entry_price", 0),
                        "sl": getattr(p, "sl", 0),
                        "volume": getattr(p, "volume", 0),
                        "alert_id": getattr(p, "alert_id", ""),
                        "playbook": "",
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
