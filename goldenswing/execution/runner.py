"""The live runner — one loop, two hosts.

``run_bars`` (works anywhere, incl. this build sandbox): replays
bars through the full production path — reader -> features ->
Combo-A detector -> alert service (charts + outbox) -> callback
handler -> risk pipeline -> PaperExecutor -> journal + learning
book.  This is the end-to-end proof of the loop.

``run_live`` (Windows host only): the same loop fed by the
MetaTrader5 library — bars via ``copy_rates_from_pos``, EXECUTE via
MT5Executor (trading whichever account the terminal is logged into,
demo or real), PAPER via the PaperExecutor fed with the same bars,
Telegram ``getUpdates`` polling feeding button taps into the
CallbackRegistry, and open MT5 positions managed bar-by-bar with
the Phase-2 exit stepper (partial closes, trail via SL moves, time
stop).  It cannot run without the library + an active terminal and
says so, plainly, when they are missing.

Env config (all optional): GS_SYMBOL, GS_MODE, GS_AUTO_EXECUTE,
GS_RISK_PCT, GS_PAPER_BALANCE, GS_JOURNAL, GS_RISK_STATE,
GS_KILL_SWITCH, GS_FORECAST_BUNDLE, GS_MT5_PATH / GS_MT5_LOGIN /
GS_MT5_PASSWORD / GS_MT5_SERVER, plus the Phase-7 alert env
(GS_TELEGRAM_*, GS_SMTP_*).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..alerts import (AlertService, CallbackRegistry, EmailSender,
                      TelegramSender, make_callback_data)
from ..alerts.callbacks import AlertAction
from ..alerts.positions import PositionNotifier
from ..alerts.summary import build_daily_summary
from ..alerts.telegram import ErrorDeduper
from ..backtest.data import REPO_ROOT, load_gold_h1
from ..backtest.harness import ExitConfig
from ..charts import ForecastNote
from ..features import FeatureEngine
from ..forecast.features import combo_a_config
from ..golden.detector import GoldenSwingDetector, playbook_of
from ..live.hierarchy import TimeframeStack, evaluate_hierarchy
from ..reader.reader import MarketReader
from .callbacks import ExecutionCallbackHandler
from .executor import Executor, OrderRequest
from .learning import TradeJournal
from .manager import ExitStepper
from .mt5 import MT5Executor, MT5Unavailable
from .paper import CloseRecord, PaperExecutor
from .risk import RiskConfig, RiskGuards, gold_spec

DATA_DIR = REPO_ROOT / "data"


# ------------------------------------------------------------- forecaster
class ForecasterScorer:
    """Scores bars with a saved Phase-6 bundle (``scripts/
    train_forecast.py`` artefact).  Construction takes the bundle
    dict directly so tests can inject fake models; :meth:`try_load`
    is the honest loader — every failure mode returns (None, reason).
    """

    def __init__(self, bundle: dict, states):
        from ..forecast.features import build_feature_frame
        X = build_feature_frame(states)
        cols = list(bundle["features"])
        for c in cols:
            if c not in X.columns:
                X[c] = np.nan
        Xv = X[cols].to_numpy(dtype=float)
        self.probs: dict[str, np.ndarray] = {}
        for dname in ("up", "dn"):
            models = bundle["models"][dname]
            ps = [np.asarray(m.predict_proba(Xv), dtype=float)[:, 1]
                  for m in models.values()]
            self.probs[dname] = np.mean(ps, axis=0) if ps else \
                np.full(len(Xv), np.nan)

    @classmethod
    def try_load(cls, path, states) -> tuple["ForecasterScorer | None", str]:
        if path is None:
            return None, "no forecaster bundle configured"
        path = Path(path)
        if not path.exists():
            return None, f"no forecaster bundle at {path}"
        try:
            import joblib
        except ImportError:
            return None, ("bundle exists but joblib/scikit-learn are not "
                          "installed on this host — detector-only mode")
        try:
            bundle = joblib.load(path)
            if not isinstance(bundle, dict) or "models" not in bundle \
                    or "features" not in bundle:
                return None, f"bundle at {path} has an unexpected shape"
            return cls(bundle, states), f"forecaster bundle loaded ({path})"
        except Exception as exc:  # unpickle needs sklearn; say so
            return None, f"forecaster bundle could not be loaded: {exc}"

    def note_for(self, signal) -> ForecastNote | None:
        dname = "up" if signal.direction == 1 else "dn"
        series = self.probs[dname]
        t = signal.index
        if t >= len(series) or not np.isfinite(series[t]):
            return None
        p = float(series[t])
        finite = series[np.isfinite(series)]
        rank = float((finite <= p).mean() * 100.0) if len(finite) else 0.0
        word = "up" if signal.direction == 1 else "down"
        return ForecastNote(
            prob_direction=p, rank_pct=rank,
            text=f"ensemble 3-day big-{word}-move prob")


def _load_scorer(bundle_path, states):
    """Multi-horizon scorer when the bundle has horizons (the quality
    dial: 1d/3d rank tables), else the legacy single-horizon scorer.
    Returns (scorer_or_None, reason) — never raises."""
    from ..forecast.api import MultiHorizonScorer
    scorer, why = MultiHorizonScorer.try_load(bundle_path, states)
    if scorer is not None:
        return scorer, why
    return ForecasterScorer.try_load(bundle_path, states)


# ----------------------------------------------------------------- config
@dataclass
class RunnerConfig:
    instrument: str = "XAUUSD"
    mode: str = "swing"
    min_score: float = 60.0
    auto_execute: bool = False
    risk_pct: float = 0.01
    paper_balance: float = 10_000.0
    journal_path: Path = DATA_DIR / "trade_journal.jsonl"
    risk_state_path: Path = DATA_DIR / "risk_state.json"
    kill_switch_path: Path = DATA_DIR / "KILL"
    forecast_bundle_path: Path | None = DATA_DIR / "forecast_models.joblib"
    chart_style: str = "signal"     # phone-first alert chart (Phase 11)
    risk: RiskConfig = field(default_factory=RiskConfig)
    mt5_module: object = None        # injection point (tests)
    use_confluence: bool = True      # Phase 10: the confluence engine is
                                     # the default path; the legacy detector
                                     # is a fallback flag, not the default.
    state_dir: Path = DATA_DIR / "live_state"  # watchdog heartbeat,
                                     # dead-man's outbox, bar buffer
    min_gap_bars: int = 0   # min bars between live signals (live loop)
    fixed_lots: float | None = None  # when set, the callback handler
                                     # sizes EVERY trade at this fixed
                                     # volume (Exness $12 loop: 0.01),
                                     # ignoring risk_pct + forecaster
                                     # dial — no R-grinding, no scaling.
    hierarchy_required: bool = False  # when True, a signal fires ONLY
                                     # with a hierarchy verdict (the
                                     # all-timeframe D1->H4->H1->entry
                                     # chain).  A missing stack blocks
                                     # every signal instead of falling
                                     # back to H1-only trading.  The
                                     # production paper loop
                                     # (scripts/paper_live.py) sets
                                     # this True; rehearsal / legacy
                                     # single-TF feeds leave it False
                                     # and run ungated but loud.

    @classmethod
    def from_env(cls, **overrides) -> "RunnerConfig":
        cfg = cls(
            instrument=os.environ.get("GS_SYMBOL", "XAUUSD"),
            mode=os.environ.get("GS_MODE", "swing"),
            auto_execute=os.environ.get("GS_AUTO_EXECUTE", "0") == "1",
            risk_pct=float(os.environ.get("GS_RISK_PCT", "0.01")),
            paper_balance=float(os.environ.get("GS_PAPER_BALANCE", "10000")),
            forecast_bundle_path=Path(os.environ["GS_FORECAST_BUNDLE"])
            if os.environ.get("GS_FORECAST_BUNDLE")
            else DATA_DIR / "forecast_models.joblib",
            use_confluence=os.environ.get("GS_USE_CONFLUENCE", "1") == "1",
            fixed_lots=float(os.environ["GS_FIXED_LOTS"])
            if os.environ.get("GS_FIXED_LOTS") else None,
            hierarchy_required=os.environ.get(
                "GS_HIERARCHY_REQUIRED", "0") == "1",
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg


# ------------------------------------------------------------------ runner
class LiveRunner:
    def __init__(self, config: RunnerConfig | None = None, *,
                 detector: GoldenSwingDetector | None = None,
                 senders: list | None = None,
                 playbooks: list | None = None,
                 paper_executor: Executor | None = None):
        self.cfg = config or RunnerConfig()
        self.spec = gold_spec()
        self.journal = TradeJournal(self.cfg.journal_path)
        self.guards = RiskGuards(self.cfg.risk_state_path, self.cfg.risk,
                                 kill_switch_path=self.cfg.kill_switch_path)
        # Injection point for the Exness loop: an ExnessPaperExecutor
        # (margin + stop-out model) instead of the default OANDA-model
        # PaperExecutor. Default path unchanged.
        self.paper = paper_executor or PaperExecutor(
            self.spec, balance=self.cfg.paper_balance)
        self.paper.on_close = self._on_paper_close
        self.detector = detector or GoldenSwingDetector(combo_a_config())
        self.service = AlertService(
            senders=senders if senders is not None
            else [TelegramSender(), EmailSender()],
            cooldown_bars=24, chart_style=self.cfg.chart_style)
        self.registry = CallbackRegistry()
        self.handler: ExecutionCallbackHandler | None = None
        self.live_executor: Executor | None = None
        self._alerts: dict[str, object] = {}
        self.forecaster_note = "forecaster not loaded"
        self._error_deduper = ErrorDeduper()  # watchdog/error flood guard
        # All-timeframe hierarchy state (live loop): the per-TF stack
        # + the latest QM M5 touch.  Built when the feed is
        # multi-TF; None on legacy single-TF feeds (ungated, loud).
        self._stack: TimeframeStack | None = None
        self._qm_touch: dict | None = None
        # Telegram command surface (owner-only; shares the getUpdates
        # poller with the button taps — one consumer, never two).
        self._command_router = None
        self._measured_oos = None
        # Last-known loop health for /status.
        self._feed_status: dict = {}
        self._last_bar_time = None
        self._last_price: float | None = None
        # Phase 10: the confluence engine is the default path.
        self._playbooks = playbooks  # lazy: built on first use
        self._engine = None          # lazy: built on first use

    def _get_playbooks(self) -> list:
        """Phase-9 playbooks + revived FVG-Limit (shadow signals only).

        FVG-Limit is wired for SIGNALS (it appears in candidates and
        the journal) but is NOT in the auto-paper ``exec_set`` — it
        stays shadow until forward paper proves it. Adding it to
        ``exec_set`` is a separate, explicitly-approved step.
        """
        if self._playbooks is None:
            from ..playbooks import (FVGLimitPlaybook,
                                     FVGMidlinePlaybook,
                                     QMM5Playbook,
                                     SweepReversalPlaybook,
                                     TrendlineRetestPlaybook)
            self._playbooks = [SweepReversalPlaybook(),
                               FVGMidlinePlaybook(),
                               TrendlineRetestPlaybook(),
                               FVGLimitPlaybook(),
                               # QM M5: shadow-only (live=False) — it
                               # appears in candidates/the journal/the
                               # shadow book, never in exec_set.
                               QMM5Playbook()]
        return self._playbooks

    def _get_engine(self):
        """The confluence engine (built once, reused)."""
        if self._engine is None:
            from ..confluence import ConfluenceEngine
            self._engine = ConfluenceEngine(
                playbooks=self._get_playbooks(),
                risk_guards=self.guards,
                journal=self.journal,
                risk_mode="shadow")
        return self._engine

    # ------------------------------------------------------------ wiring
    def _build_handler(self, live_executor: Executor) -> None:
        self.live_executor = live_executor
        self.handler = ExecutionCallbackHandler(
            executor=live_executor, paper_executor=self.paper,
            guards=self.guards, spec=self.spec, journal=self.journal,
            risk_pct=self.cfg.risk_pct, fixed_lots=self.cfg.fixed_lots)
        self.registry.register(self.handler)

    def _on_paper_close(self, record: CloseRecord) -> None:
        self.journal.log_close(
            alert_id=record.alert_id, ticket=record.ticket,
            exit_price=record.exit_price, reason=record.reason,
            pnl_money=record.pnl_money, r_net=record.r_net,
            hold_bars=record.hold_bars, partial_taken=record.partial_taken)
        # Paper closes feed the guards only when paper IS the execute
        # executor (dry mode); on the live host paper is a side book
        # and must not halt real trading.
        if self.live_executor is self.paper:
            self.guards.record_outcome(
                record.pnl_money, self.paper.account_state().balance)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _phase_of(state) -> str:
        feats = getattr(state, "features", None)
        if feats is not None and getattr(feats, "phase", None) is not None:
            return str(feats.phase.phase)
        return "unknown"

    # -------------------------------------------- all-timeframe hierarchy
    @staticmethod
    def _is_multitf_feed(feed) -> bool:
        """A feed serves the whole stack iff it has the multi-TF
        interface (``warm_start_all`` / ``fetch_new_all``)."""
        return hasattr(feed, "warm_start_all") and hasattr(
            feed, "fetch_new_all")

    def _build_stack_from_warm(self, feed) -> TimeframeStack:
        """Cold-start stack: full history per TF from the feed."""
        return TimeframeStack.from_frames(feed.warm_start_all())

    def _refresh_stack(self, feed, stack: TimeframeStack
                       ) -> dict[str, tuple[int, int]]:
        """Fetch every TF's new bars into the stack.  Returns
        {tf: (accepted, rejected)} — rejected > 0 means out-of-order /
        duplicate bars arrived and were dropped, never merged."""
        new_by_tf = feed.fetch_new_all(dict(stack.cursors))
        out: dict[str, tuple[int, int]] = {}
        for tf, df_new in new_by_tf.items():
            out[tf] = stack.append(tf, df_new)
        return out

    def _advance_qm(self, m5_df) -> dict | None:
        """Latest QM M5 touch on the M5 frame (annotation only).

        A FRESH QM playbook scans the whole M5 frame each tick — no
        positional cursor to rewind, no persisted state to go stale.
        The touch is the M5 entry leg inside the hierarchy verdict;
        it never fires on its own and never gates a sleeve signal.
        Failures return None (annotation missing, never a crash).
        """
        try:
            if m5_df is None or len(m5_df) < 30:
                return None
            from ..playbooks.base import EngineContext
            from ..playbooks.qm_m5 import QMM5Playbook
            pb = QMM5Playbook()
            ctx = EngineContext(df=m5_df, states=[], years=0.0,
                                instrument=self.cfg.instrument)
            cands = pb.candidates(ctx)
            if not cands:
                return None
            latest = max(cands, key=lambda c: c.signal_index)
            sig = latest.outcome.signal
            lc = dict(getattr(sig, "live_context", None) or {})
            touch = {"direction": int(latest.direction),
                     "touch_ts": m5_df.index[latest.signal_index],
                     "touch_index": int(latest.signal_index),
                     "qm_level": lc.get("qm_level"),
                     "head": lc.get("head"),
                     "neckline": lc.get("neckline"),
                     "tags": lc.get("tags", [])}
            return touch
        except Exception as exc:
            self.journal.log_note(
                f"hierarchy: QM M5 scan failed (annotation only): {exc}")
            return None

    def _hierarchy_verdict_for_bar(self, df, t: int, states,
                                   stack: TimeframeStack,
                                   qm_touch: dict | None) -> dict | None:
        """Hierarchy verdict at H1 bar ``t`` (closed bars only).

        AMD phase comes from the bar's own feature state (tag, never
        a veto).  Evaluation failures return None and are journalled
        loudly — never silently treated as permission.
        """
        try:
            amd_phase = amd_via = amd_conf = None
            st = states[t] if 0 <= t < len(states) else None
            feats = getattr(st, "features", None) if st is not None \
                else None
            ph = getattr(feats, "phase", None) if feats is not None \
                else None
            if ph is not None:
                amd_phase = getattr(ph, "phase", None) or "unknown"
                amd_via = getattr(ph, "via", None)
                amd_conf = getattr(ph, "confidence", None)
            return evaluate_hierarchy(
                stack, asof=df.index[t], amd_phase=amd_phase,
                amd_via=amd_via, amd_confidence=amd_conf,
                qm_touch=qm_touch, swing_n=3)
        except Exception as exc:
            self.journal.log_note(
                f"hierarchy: evaluation failed at bar {t}: {exc}")
            return None

    def _emit_alert(self, df, state, signal, fsets, scorer,
                    engine=None, auto_dispatch: bool | None = None) -> object | None:
        """Build and send one alert; auto-dispatch EXECUTE when enabled.

        ``auto_dispatch`` overrides ``cfg.auto_execute`` for this alert —
        the paper loop uses it to paper-trade the sleeves while the
        anchor stays shadow (alert + journal, no fill).
        """
        note = scorer.note_for(signal) if scorer is not None else None
        alert = self.service.process(
            df, state, signal, instrument=self.cfg.instrument,
            mode=self.cfg.mode, features=fsets, forecast=note)
        if alert is None:
            return None
        self._alerts[alert.alert_id] = alert
        prob = rank = None
        if note is not None:
            prob, rank = note.prob_direction, note.rank_pct
        # horizon agreement -> sizing input (never a veto): the dial
        size_mult = 1.0
        table = scorer.table() if scorer is not None \
            and hasattr(scorer, "table") else None
        if table is not None:
            from ..confluence import forecast_agreement
            n_agree, _ = forecast_agreement(
                signal.index, signal.direction, table)
            if n_agree >= 2 and engine is not None:
                size_mult = engine.cfg.forecast_size_mult
        self.handler.register_context(
            alert.alert_id, phase=self._phase_of(state),
            playbook=playbook_of(state, signal.kind),
            forecaster_prob=prob, forecaster_rank=rank,
            forecast_size_mult=size_mult)
        if auto_dispatch if auto_dispatch is not None \
                else self.cfg.auto_execute:
            self.registry.dispatch(
                make_callback_data(AlertAction.EXECUTE, alert.alert_id),
                self._alerts)
        return alert

    @staticmethod
    def _bar_of(df: pd.DataFrame, t: int) -> dict:
        row = df.iloc[t]
        return {"index": t, "time": df.index[t], "open": float(row["open"]),
                "high": float(row["high"]), "low": float(row["low"]),
                "close": float(row["close"])}

    # -------------------------------------------------------------- dry run
    def run_bars(self, df: pd.DataFrame | None = None, *,
                 max_bars: int | None = None,
                 peers: dict | None = None) -> dict:
        """The end-to-end loop over historical bars + PaperExecutor.

        Phase 10: the default path is reader -> playbooks -> confluence
        engine -> alerts -> risk -> PaperExecutor -> journal.  The
        legacy Combo-A detector path is available via
        ``use_confluence=False`` (fallback flag, not the default).
        """
        if not self.cfg.use_confluence:
            return self._run_bars_legacy(df, max_bars=max_bars,
                                         peers=peers)
        if df is None:
            df = load_gold_h1()
        if max_bars is not None:
            df = df.iloc[:max_bars]
        self._build_handler(self.paper)

        # reader -> features (Phase-9 path, not the legacy split)
        from ..backtest.data import load_peers
        from ..features.engine import read_and_featurize
        from ..playbooks.base import EngineContext
        if peers is None:
            peers = load_peers()  # the canary lesson: relstr needs its peers
        states = read_and_featurize(df, peers)
        years = (df.index[-1] - df.index[0]).total_seconds() / 86400.0 / 365.25
        ctx = EngineContext(df=df, states=states, years=years,
                            instrument=self.cfg.instrument)

        # playbooks -> confluence (batch run gives the canary + signals)
        engine = self._get_engine()
        rep = engine.run(ctx)
        m = rep.live_metrics
        # CANARY: on the full dataset, the anchor book reproduces exactly
        # through the new runner.  (Partial runs skip the canary.)
        if max_bars is None:
            assert m["trades"] == 199, f"CANARY FAILED: {m['trades']} != 199"
            assert abs(m["return_pct"] - 28.92) < 0.01, \
                f"CANARY FAILED: return {m['return_pct']:.2f} != 28.92"
            assert abs(m["pf"] - 1.25) < 0.01, \
                f"CANARY FAILED: PF {m['pf']:.2f} != 1.25"

        # signals by bar: live playbooks, at/above their live threshold.
        # (The rehearsal simulates the LIVE decision process — the paper
        # executor manages positions — not a replay of the greedy book.)
        signals: dict[int, list] = {}
        for pb in self._get_playbooks():
            if not pb.live:
                continue
            for sc in rep.scored.get(pb.name, []):
                c = sc.candidate
                if c.base_score >= pb.live_threshold:
                    signals.setdefault(c.signal_index, []).append(
                        (pb, sc, c.outcome.signal))

        scorer, why = _load_scorer(
            self.cfg.forecast_bundle_path, states)
        self.forecaster_note = why

        # rehearsal: bar-by-bar through alert -> risk -> executor -> journal
        n_alerts = n_exec = 0
        for t in range(len(df)):
            self.paper.advance(self._bar_of(df, t))
            for pb, sc, sig in signals.get(t, []):
                if sig.score < self.cfg.min_score:
                    continue
                if self.paper.positions():
                    continue  # one position at a time, like live
                alert = self._emit_alert(df, states[t], sig, None, scorer,
                                           engine)
                if alert is None:
                    continue
                n_alerts += 1
                if self.cfg.auto_execute:
                    # risk -> executor (via the callback handler path)
                    self.registry.dispatch(alert.alert_id, "EXECUTE")
                    n_exec += 1
        perf = self.journal.performance().aggregate()
        return {
            "bars": len(df), "signals": sum(len(v) for v in signals.values()),
            "alerts": n_alerts, "executed": n_exec,
            "paper_balance": self.paper.account_state().balance,
            "paper_open": len(self.paper.positions()),
            "journal": str(self.cfg.journal_path),
            "performance": perf, "forecaster": self.forecaster_note,
            "canary": {"trades": m["trades"],
                       "return_pct": m["return_pct"], "pf": m["pf"]},
        }

    # ------------------------------------------------------- paper loop
    def run_paper_loop(self, feed, *,
                       execute_playbooks: tuple = ("fvg_midline",
                                                   "trendline_retest"),
                       max_bars: int | None = None,
                       min_score: float | None = None,
                       signal_gap_bars: int = 0,
                       peers: dict | None = None,
                       poll_commands: bool = True) -> dict:
        """The paper loop: live feed -> live_signal -> confluence rank ->
        alerts -> paper fills -> journal.  Paper money only — no MT5, no
        live-order code paths.

        ``feed`` is a ``PollingFeed`` (live, yfinance GC=F) or a
        ``ReplayFeed`` (rehearsal); both expose ``warm_start``,
        ``fetch_new``, ``poll_status`` and ``poll_interval``.
        ``execute_playbooks`` are auto-paper-traded; any other live
        playbook (the sweep_reversal anchor, which failed walk-forward)
        runs shadow — its signals are journaled, never filled.
        ``max_bars`` caps processed new bars (the rehearsal's stop
        condition; None = run forever).
        ``poll_commands=False`` skips Telegram command polling: the
        side-by-side Exness loop must never compete with the live
        loop for the shared getUpdates poller (one consumer only).
        """
        import time as _time

        from ..alerts.positions import PositionNotifier
        from ..backtest.data import load_peers
        from ..features.engine import read_and_featurize
        from ..playbooks.base import EngineContext

        self._build_handler(self.paper)
        if peers is None:
            peers = load_peers()
        engine = self._get_engine()
        prev_risk_mode = engine.risk_mode
        engine.risk_mode = "enforce"   # paper loop: risk decisions are real
        playbooks = [pb for pb in self._get_playbooks() if pb.live]
        exec_set = set(execute_playbooks)
        score_floor = self.cfg.min_score if min_score is None else min_score

        df = feed.warm_start()
        if len(df) == 0:
            raise RuntimeError("paper loop: feed warm start returned no bars")
        # Multi-TF stack (the hierarchy): full history per TF from the
        # feed, plus the latest QM M5 touch.  Legacy single-TF feeds
        # keep stack=None (ungated, loud — see _paper_loop_bar).
        stack = None
        qm_touch = None
        if self._is_multitf_feed(feed):
            stack = self._build_stack_from_warm(feed)
            df = stack.frames.get("H1", df)
            qm_touch = self._advance_qm(stack.frames.get("M5"))
            self.journal.log_note(
                "paper loop: multi-TF stack live "
                f"({', '.join(f'{tf}:{len(stack.frames.get(tf, []))}' for tf in stack.frames)})")
        # Fast-forward: rebuild each playbook's streaming state over
        # history.  Signals for past bars are discarded — the loop only
        # trades the latest close from here on.
        states = read_and_featurize(df, peers)
        for pb in playbooks:
            for t2 in range(len(df)):
                pb.live_signal(states, t2, df)

        # Paper fill/close narration (outbox files in dry mode).
        prev_hooks = self._wire_paper_notifier()

        stats = {"bars": 0, "signals": 0, "alerts": 0, "executed": 0,
                 "shadow": 0, "gated": 0, "stale_polls": 0,
                 "hierarchy_blocked": 0}
        last_time = df.index[-1]
        last_sig_bar: dict[str, int] = {}
        try:
            while max_bars is None or stats["bars"] < max_bars:
                if poll_commands:
                    self._poll_telegram()   # owner commands share the loop
                if stack is not None:
                    counts = self._refresh_stack(feed, stack)
                    new = stack.frames.get("H1", df).loc[
                        lambda d: d.index > last_time] \
                        if len(stack.frames.get("H1", [])) else df.iloc[0:0]
                    if counts.get("M5", (0, 0))[0] > 0:
                        qm_touch = self._advance_qm(
                            stack.frames.get("M5"))
                    rejected = {tf: r for tf, (_a, r) in counts.items()
                                if r > 0}
                    if rejected:
                        self.journal.log_note(
                            "paper loop: stack rejected out-of-order / "
                            f"duplicate bars (dropped, never merged): "
                            f"{rejected}")
                else:
                    new = feed.fetch_new(since=last_time)
                self._feed_status = feed.poll_status()
                if len(new) == 0:
                    status = self._feed_status
                    if not status.get("ok", True):
                        stats["stale_polls"] += 1
                        self.journal.log_note(
                            "paper loop: feed stale: "
                            f"{status.get('reason')}")
                    _time.sleep(feed.poll_interval)
                    continue
                fresh = new.index[new.index > last_time]
                if len(fresh) == 0:
                    _time.sleep(feed.poll_interval)
                    continue
                df = pd.concat([df, new.loc[fresh]]).sort_index()
                years = (df.index[-1] - df.index[0]).total_seconds() \
                    / 86400.0 / 365.25
                for t in range(len(df) - len(fresh), len(df)):
                    if max_bars is not None and stats["bars"] >= max_bars:
                        break
                    stats["bars"] += 1
                    self._paper_loop_bar(
                        df, t, playbooks, engine, peers, exec_set,
                        score_floor, signal_gap_bars, stats, last_sig_bar,
                        years, stack=stack, qm_touch=qm_touch)
                    last_time = df.index[t]
        finally:
            engine.risk_mode = prev_risk_mode
            self._unwire_paper_notifier(prev_hooks)
            self.journal.log_note(
                f"paper loop stopped: {stats['bars']} bars, "
                f"{stats['signals']} signals, {stats['alerts']} alerts, "
                f"{stats['executed']} paper fills, {stats['shadow']} shadow, "
                f"{stats['hierarchy_blocked']} hierarchy-blocked")
        perf = self.journal.performance().aggregate()
        return {**stats,
                "paper_balance": self.paper.account_state().balance,
                "paper_open": len(self.paper.positions()),
                "journal": str(self.cfg.journal_path),
                "performance": perf,
                "forecaster": self.forecaster_note}

    # ------------------------------------------------- paper notifier hooks
    def _wire_paper_notifier(self) -> tuple:
        """Narrate paper fills/closes (outbox files in dry mode).

        Returns the previous hooks for :meth:`_unwire_paper_notifier`.
        """
        notifier = PositionNotifier(self._telegram_sender())
        prev_on_fill = self.paper.on_fill
        prev_on_close = self.paper.on_close

        def _notify_fill(pos) -> None:
            try:
                notifier.on_fill(
                    ticket=pos.ticket, alert_id=pos.alert_id,
                    direction=pos.direction, fill_price=pos.entry_price,
                    volume=pos.volume, stop=pos.sl,
                    instrument=self.cfg.instrument)
            finally:
                if prev_on_fill is not None:
                    prev_on_fill(pos)

        def _notify_close(record) -> None:
            try:
                notifier.on_close(
                    ticket=record.ticket, direction=record.direction,
                    exit_price=record.exit_price, reason=record.reason,
                    pnl_money=record.pnl_money, r_net=record.r_net,
                    hold_bars=record.hold_bars,
                    instrument=self.cfg.instrument)
            finally:
                if prev_on_close is not None:
                    prev_on_close(record)

        self.paper.on_fill = _notify_fill
        self.paper.on_close = _notify_close
        return prev_on_fill, prev_on_close

    def _unwire_paper_notifier(self, prev: tuple) -> None:
        self.paper.on_fill, self.paper.on_close = prev

    # ------------------------------------------------------- paper --once
    def run_paper_once(self, feed, *,
                       execute_playbooks: tuple = ("fvg_midline",
                                                   "trendline_retest"),
                       state_store_path=None,
                       min_score: float | None = None,
                       signal_gap_bars: int = 0,
                       peers: dict | None = None,
                       poll_commands: bool = True) -> dict:
        """One cron tick: process new closed bars since the last tick.

        Designed for ``scripts/paper_live.py --once`` from cron (hourly,
        just after the H1 bar closes).  State survives across ticks in
        the StateStore: the bar buffer, playbook streaming state, paper
        executor state (open positions keep their exit-stepper state),
        and last-processed bar time.  ``GS_COLD_START=1`` forces a full
        cold start (ignores the store).

        When the buffer would exceed ``StateStore.buffer_bars``, it is
        rebased to the tail: playbook state is rebuilt by the proven
        cold-start fast-forward and the paper executor's positional bar
        indices are shifted by the same delta, so hold-time arithmetic
        is preserved.

        Do NOT run --once concurrently with the daemon paper loop (or
        two cron ticks overlapping): both would process the same bars
        and double-trade.  Pick one.
        """
        import os as _os
        from pathlib import Path as _Path

        from ..backtest.data import load_peers
        from ..features.engine import read_and_featurize
        from .state_store import StateStore

        store_path = _Path(state_store_path) if state_store_path \
            else _Path(self.cfg.journal_path).parent / "paper_state"
        store = StateStore(store_path)

        self._build_handler(self.paper)
        if peers is None:
            peers = load_peers()
        engine = self._get_engine()
        prev_risk_mode = engine.risk_mode
        engine.risk_mode = "enforce"   # paper loop: risk decisions are real
        playbooks = [pb for pb in self._get_playbooks() if pb.live]
        exec_set = set(execute_playbooks)
        score_floor = self.cfg.min_score if min_score is None else min_score

        stats = {"bars": 0, "signals": 0, "alerts": 0, "executed": 0,
                 "shadow": 0, "gated": 0, "stale_polls": 0,
                 "hierarchy_blocked": 0}
        prev_hooks = self._wire_paper_notifier()
        # All-timeframe stack (multi-TF feeds only).  Legacy feeds
        # keep stack=None (ungated, loud — unless hierarchy_required).
        stack: TimeframeStack | None = None
        qm_touch: dict | None = None
        multitf = self._is_multitf_feed(feed)
        try:
            if poll_commands:
                self._poll_telegram()   # answer owner commands each tick

            cold = _os.environ.get("GS_COLD_START") == "1"
            if cold:
                store.clear()
                self.journal.log_note(
                    "paper --once: GS_COLD_START=1, bar buffer wiped")
            buffered = None if cold else store.load_bars()
            meta = {} if cold else store.load_meta()
            if store.load_error:
                self.journal.log_note(f"paper --once: {store.load_error}")
                buffered, meta = None, {}

            last_sig_bar: dict[str, int] = {}
            if buffered is None or len(buffered) == 0:
                # Cold start: full history + fast-forward (the proven
                # path from run_paper_loop).
                if multitf:
                    stack = self._build_stack_from_warm(feed)
                    df = stack.frames.get("H1")
                    qm_touch = self._advance_qm(stack.frames.get("M5"))
                else:
                    df = feed.warm_start()
                if df is None or len(df) == 0:
                    raise RuntimeError(
                        "paper --once: feed warm start returned no bars")
                states = read_and_featurize(df, peers)
                for pb in playbooks:
                    for t2 in range(len(df)):
                        pb.live_signal(states, t2, df)
                last_time = df.index[-1]
            else:
                # Warm resume: only what is new since the buffer end.
                for pb in playbooks:
                    pb.import_state(store.load_playbook(pb.name))
                self.paper.import_state(store.load_paper())
                buf_end = buffered.index[-1]
                if multitf:
                    # Restore the stack (forward-compatible: a
                    # pre-MTF store has no stack.pkl — start from
                    # empty cursors at the buffer end and let the
                    # fetches fill the gap; the forward-only append
                    # self-heals bars the first fetch misses).
                    saved = store.load_stack()
                    if saved is not None:
                        cursors = {}
                        for tf, iso in (meta.get("stack_cursors")
                                        or {}).items():
                            try:
                                cursors[tf] = pd.Timestamp(iso)
                            except Exception:
                                continue
                        stack = TimeframeStack(
                            frames=saved, cursors=cursors)
                    else:
                        self.journal.log_note(
                            "paper --once: no MTF stack in store "
                            "(pre-MTF store): rebuilding from fetches")
                        stack = TimeframeStack(
                            frames={},
                            cursors={tf: buf_end for tf in
                                     ("D1", "H4", "H1", "M15", "M30",
                                      "M5")})
                    counts = self._refresh_stack(feed, stack)
                    new = stack.frames.get("H1", buffered).loc[
                        lambda d: d.index > buf_end] \
                        if stack.frames.get("H1") is not None \
                        else buffered.iloc[0:0]
                    if counts.get("M5", (0, 0))[0] > 0:
                        qm_touch = self._advance_qm(
                            stack.frames.get("M5"))
                    rejected = {tf: r for tf, (_a, r) in counts.items()
                                if r > 0}
                    if rejected:
                        self.journal.log_note(
                            "paper --once: stack rejected out-of-order / "
                            f"duplicate bars (dropped, never merged): "
                            f"{rejected}")
                else:
                    new = feed.fetch_new(since=buf_end)
                self._feed_status = feed.poll_status()
                if not self._feed_status.get("ok", True):
                    stats["stale_polls"] += 1
                    self.journal.log_note(
                        "paper --once: feed stale: "
                        f"{self._feed_status.get('reason')}")
                fresh_idx = new.index[new.index > buf_end] \
                    if len(new) else new.index[:0]
                df = buffered
                if len(fresh_idx):
                    df = pd.concat(
                        [buffered, new.loc[fresh_idx]]).sort_index()
                    df = df[~df.index.duplicated(keep="last")]
                    overflow = len(df) - store.buffer_bars
                    if overflow > 0:
                        # Rebase: tail the buffer, rebuild playbook
                        # streaming state by the cold fast-forward,
                        # shift the paper executor's positional indices
                        # by the same delta.
                        df = df.tail(store.buffer_bars)
                        self.paper.shift_bar_indices(-overflow)
                        # Replay ONLY the already-processed prefix.
                        # Bars with timestamp > meta last_bar_time are
                        # FRESH and must be left for the _paper_loop_bar
                        # pass below: replay ignores live_signal return
                        # values, so replaying the tail would DISCARD
                        # its signals, and the fresh loop's re-call
                        # would then rewind playbook cursors (FVG
                        # crashed every tick on idx=1998 vs last=1999).
                        meta_last = pd.Timestamp(meta["last_bar_time"]) \
                            if meta.get("last_bar_time") else buf_end
                        n_fresh = int((df.index > meta_last).sum())
                        replay_end = max(0, len(df) - n_fresh)
                        states = read_and_featurize(df, peers)
                        for pb in playbooks:
                            pb.import_state({})
                            for t2 in range(replay_end):
                                pb.live_signal(states, t2, df)
                # Last PROCESSED bar (timestamps — robust to rebasing).
                last_time = pd.Timestamp(meta["last_bar_time"]) \
                    if meta.get("last_bar_time") else buf_end
                if meta.get("last_sig_bar"):
                    last_sig_bar = {k: int(v) for k, v in
                                    meta["last_sig_bar"].items()}

            fresh = df.index[df.index > last_time] \
                if len(df) else df.index[:0]
            if len(fresh):
                years = (df.index[-1] - df.index[0]).total_seconds() \
                    / 86400.0 / 365.25
                for t in range(len(df) - len(fresh), len(df)):
                    stats["bars"] += 1
                    self._paper_loop_bar(
                        df, t, playbooks, engine, peers, exec_set,
                        score_floor, signal_gap_bars, stats, last_sig_bar,
                        years, stack=stack, qm_touch=qm_touch)
                    last_time = df.index[t]
                self._last_bar_time = last_time
                try:
                    self._last_price = float(df["close"].iloc[-1])
                except Exception:
                    pass

            # Persist everything the next tick needs.
            store.save_bars(df)
            meta_out = {
                "last_bar_time": last_time,
                "last_sig_bar": last_sig_bar,
                "instrument": self.cfg.instrument,
            }
            if stack is not None:
                store.save_stack(stack.frames)
                meta_out["stack_cursors"] = {
                    tf: ts.isoformat()
                    for tf, ts in stack.cursors.items() if ts is not None}
            store.save_meta(meta_out)
            for pb in playbooks:
                store.save_playbook(pb.name, pb.export_state())
            store.save_paper(self.paper.export_state())
            self.journal.log_note(
                f"paper --once: {stats['bars']} new bars, "
                f"{stats['signals']} signals, {stats['alerts']} alerts, "
                f"{stats['executed']} paper fills, {stats['shadow']} shadow, "
                f"{stats['hierarchy_blocked']} hierarchy-blocked")
        finally:
            engine.risk_mode = prev_risk_mode
            self._unwire_paper_notifier(prev_hooks)
        perf = self.journal.performance().aggregate()
        return {**stats,
                "paper_balance": self.paper.account_state().balance,
                "paper_open": len(self.paper.positions()),
                "journal": str(self.cfg.journal_path),
                "state_store": str(store_path),
                "feed_status": dict(self._feed_status),
                "performance": perf}

    def _log_below_floor(self, below_floor: list[tuple[str, float]],
                         floor: float, bar_t, bar_ts, *,
                         prefix: str) -> None:
        """One compact below-floor record per bar, no matter how many
        playbooks emitted below the floor on it.  Shared by the paper
        loop and the live loop — a below-floor emission is a real
        evaluation outcome in both, never silent in either."""
        if not below_floor:
            return
        self.journal.log_note(
            f"{prefix}: below score floor "
            + ", ".join(f"{name} score={score:.0f}"
                        for name, score in below_floor)
            + f" (floor={floor:.0f}, bar={bar_t} {bar_ts})")

    def _paper_loop_bar(self, df, t: int, playbooks, engine, peers,
                        exec_set: set, score_floor: float,
                        signal_gap_bars: int, stats: dict,
                        last_sig_bar: dict, years: float,
                        stack: TimeframeStack | None = None,
                        qm_touch: dict | None = None) -> None:
        """One new closed bar through the paper pipeline.

        ``stack`` is the all-timeframe stack (multi-TF feeds only):
        when present, every signal must pass the hierarchy verdict
        (D1 -> H4 -> H1 -> entry clock) via the confluence gate
        before it can alert or fill.  When ``stack`` is None and
        ``hierarchy_required`` is set, every signal is blocked
        instead of falling back to H1-only trading.
        """
        from ..features.engine import read_and_featurize
        from ..playbooks.base import EngineContext

        bar = self._bar_of(df, t)
        # Full re-featurize each bar: O(n) per bar at H1 cadence, and it
        # guarantees states[t] is bit-identical to the batch path the
        # parity tests prove (no incremental-featurization drift).
        states = read_and_featurize(df, peers)
        self.paper.advance(bar)
        ctx = EngineContext(df=df, states=states, years=years,
                            instrument=self.cfg.instrument)
        # Hierarchy verdict, once per bar (not per playbook): the
        # whole stack evaluated at this bar's close, causally sliced.
        verdict = None
        if stack is not None:
            verdict = self._hierarchy_verdict_for_bar(
                df, t, states, stack, qm_touch)
        hierarchy_unavailable = stack is None \
            and self.cfg.hierarchy_required
        if self.paper.positions():
            return  # one position at a time, like live
        below_floor: list[tuple[str, float]] = []
        for pb in playbooks:
            sigs = pb.live_signal(states, t, df) or []
            for sig in sigs:
                if sig.score < score_floor:
                    # Never silent: a below-floor emission is a real
                    # evaluation outcome. Collected and logged once
                    # per bar below (aggregated, not per-signal spam).
                    below_floor.append((pb.name, sig.score))
                    continue
                if sig.index - last_sig_bar.get(pb.name, -10 ** 9) \
                        < signal_gap_bars:
                    continue
                if hierarchy_unavailable:
                    stats["gated"] += 1
                    stats["hierarchy_blocked"] = stats.get(
                        "hierarchy_blocked", 0) + 1
                    self.journal.log_note(
                        "paper loop: hierarchy REQUIRED but no stack "
                        f"(single-TF feed): {pb.name} signal blocked, "
                        "no H1-only fallback")
                    continue
                ok, reason = engine.gate_live_signal(
                    pb, sig, ctx, hierarchy_verdict=verdict)
                if not ok:
                    stats["gated"] += 1
                    if reason.startswith("hierarchy:"):
                        stats["hierarchy_blocked"] = stats.get(
                            "hierarchy_blocked", 0) + 1
                    self.journal.log_note(
                        f"paper loop: {pb.name} signal gated: {reason}")
                    continue
                scorer, why = _load_scorer(
                    self.cfg.forecast_bundle_path, states) \
                    if self.cfg.forecast_bundle_path is not None \
                    else (None, "no forecaster bundle configured")
                self.forecaster_note = why
                table = scorer.table() \
                    if scorer is not None \
                    and hasattr(scorer, "table") else None
                ctx.forecast_table = table  # the quality dial, never a veto
                conf, notes = engine.score_live_signal(pb, sig, ctx)
                auto = pb.name in exec_set
                alert = self._emit_alert(df, states[t], sig, None, scorer,
                                         engine, auto_dispatch=auto)
                if alert is None:
                    continue
                last_sig_bar[pb.name] = sig.index
                stats["signals"] += 1
                stats["alerts"] += 1
                tag = "PAPER" if auto else "SHADOW"
                if auto:
                    stats["executed"] += 1
                else:
                    stats["shadow"] += 1
                self.journal.log_note(
                    f"paper loop [{tag}] {pb.name} "
                    f"{'+' if sig.direction > 0 else '-'}1 "
                    f"score={sig.score:.0f} conf={conf:.0f} "
                    f"entry~{sig.entry_ref:.1f} stop~{sig.invalidation:.1f} "
                    f"factors={'+'.join(notes) if notes else '-'}"
                    + (f" hierarchy: D1 {verdict['bias_d1']:+d} / H4 "
                       f"{verdict['bias_h4']:+d} / "
                       f"{verdict['trigger']} / "
                       f"{verdict['entry_clock']} "
                       f"amd:{verdict.get('amd_phase')} "
                       f"qm:{verdict.get('qm', {}).get('status')}"
                       if verdict is not None else
                       " hierarchy: unavailable (legacy feed)"))
        self._log_below_floor(below_floor, score_floor, t, df.index[t],
                              prefix="paper loop")

    def _run_bars_legacy(self, df: pd.DataFrame | None = None, *,
                         max_bars: int | None = None,
                         peers: dict | None = None) -> dict:
        """The pre-Phase-10 Combo-A detector path (fallback flag)."""
        if df is None:
            df = load_gold_h1()
        if max_bars is not None:
            df = df.iloc[:max_bars]
        self._build_handler(self.paper)
        states = MarketReader().read(df)
        try:
            fsets = FeatureEngine().compute(df, states, peers)
            for s, fs in zip(states, fsets):
                s.features = fs
        except Exception as exc:  # features degrade; detector falls back
            fsets = None
            self.journal.log_note(f"feature attach failed in runner: {exc}")
        scorer, why = _load_scorer(
            self.cfg.forecast_bundle_path, states)
        self.forecaster_note = why

        signals = {s.index: s for s in self.detector.scan(states)}
        n_alerts = 0
        for t in range(len(df)):
            self.paper.advance(self._bar_of(df, t))
            sig = signals.get(t)
            if sig is None or sig.score < self.cfg.min_score:
                continue
            if self._emit_alert(df, states[t], sig, fsets, scorer) is not None:
                n_alerts += 1
        perf = self.journal.performance().aggregate()
        return {
            "bars": len(df), "signals": len(signals), "alerts": n_alerts,
            "paper_balance": self.paper.account_state().balance,
            "paper_open": len(self.paper.positions()),
            "journal": str(self.cfg.journal_path),
            "performance": perf, "forecaster": self.forecaster_note,
            "path": "legacy-detector",
        }

    # -------------------------------------------------------------- live run
    def run_live(self, *, poll_seconds: int = 20, bars_count: int = 1500,
                 max_iterations: int | None = None) -> None:
        """The Windows-host loop.  Raises MT5Unavailable with the
        precise blocker when the library/terminal is not there.

        Phase 10: the default path runs playbooks -> confluence engine
        (with feed watchdog + warm-resume state).  The legacy Combo-A
        detector path is available via ``use_confluence=False``.
        """
        if not self.cfg.use_confluence:
            return self._run_live_legacy(poll_seconds=poll_seconds,
                                         bars_count=bars_count,
                                         max_iterations=max_iterations)
        return self._run_live_confluence(
            poll_seconds=poll_seconds, bars_count=bars_count,
            max_iterations=max_iterations)

    def _run_live_confluence(self, *, poll_seconds: int = 20,
                             bars_count: int = 1500,
                             max_iterations: int | None = None) -> None:
        """Live loop: MT5 bars -> StateStore -> playbooks -> confluence
        engine -> alerts -> risk -> executor -> journal, guarded by the
        feed watchdog."""
        from .state_store import StateStore
        from .watchdog import FeedWatchdog, WatchdogConfig
        from ..features.engine import read_and_featurize
        from ..playbooks.base import EngineContext

        mt5 = MT5Executor(mt5_module=self.cfg.mt5_module)
        ok, msg = mt5.connect(
            path=os.environ.get("GS_MT5_PATH"),
            login=int(os.environ["GS_MT5_LOGIN"])
            if os.environ.get("GS_MT5_LOGIN") else None,
            password=os.environ.get("GS_MT5_PASSWORD"),
            server=os.environ.get("GS_MT5_SERVER"))
        if not ok:
            raise MT5Unavailable(msg)
        self.journal.log_note(f"live runner connected: {msg}")
        self._build_handler(mt5)

        # watchdog: staleness + heartbeat + dead-man's
        state_dir = Path(self.cfg.state_dir)
        watchdog = FeedWatchdog(
            WatchdogConfig(
                heartbeat_path=state_dir / "heartbeat.txt",
                deadman_outbox=state_dir / "outbox" / "deadman.log"),
            on_alert=self._watchdog_alert)
        if watchdog.check_previous_death():
            self.journal.log_note("watchdog: previous loop died uncleanly")
        watchdog.arm_deadman()

        # warm resume: bar buffer from the store, else cold start.
        # GS_COLD_START=1 forces a fresh cold start (ignores the buffer).
        store = StateStore(state_dir / "bars")
        if os.environ.get("GS_COLD_START") == "1":
            store.clear()
            self.journal.log_note(
                "GS_COLD_START=1: bar buffer wiped, forcing cold start")
        if store.load_error:
            self.journal.log_note(f"state store: {store.load_error}")
        buffered = store.load_bars()
        meta = store.load_meta()
        last_bar_time = None
        # position-lifecycle narration (fills/partials/stops/closes)
        notifier = PositionNotifier(self._telegram_sender())
        manager = _LiveManager(self, mt5, notifier=notifier)
        # daily summary: first one fires at the next UTC day rollover
        from datetime import datetime, timezone
        summary_day = datetime.now(timezone.utc).date().isoformat()
        n_alerts_sent = 0
        if meta.get("last_bar_time"):
            try:
                from datetime import datetime
                last_bar_time = datetime.fromisoformat(
                    meta["last_bar_time"])
            except Exception:
                pass
        for pb in self._get_playbooks():
            pb.import_state(store.load_playbook(pb.name))

        engine = self._get_engine()
        engine.risk_mode = "enforce"  # live: risk enforces, not shadow
        last_signal_bar = -10 ** 9
        iteration = 0
        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                self._poll_telegram()
                watchdog.beat()

                conn_ok, conn_msg = mt5.ping()
                if not conn_ok:
                    watchdog._alert(f"MT5 TERMINAL LOST: {conn_msg} - "
                                    f"entries halted until reconnect")
                    self.journal.log_note(f"watchdog: {conn_msg}")
                    if max_iterations is None:
                        time.sleep(poll_seconds)
                    continue

                fresh = mt5.copy_rates_h1(self.cfg.instrument, bars_count)
                if buffered is not None and len(buffered):
                    new = fresh[fresh.index > buffered.index[-1]]
                    df = pd.concat([buffered, new]).tail(store.buffer_bars)
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                else:
                    df = fresh  # cold start: full fetch
                buffered = df
                closed = df.iloc[:-1]  # drop the forming bar
                bar_time = closed.index[-1]
                if bar_time == last_bar_time or len(closed) <= 100:
                    if max_iterations is None:
                        time.sleep(poll_seconds)
                    continue

                watchdog.note_bar(bar_time)
                if not watchdog.check_staleness():
                    self.journal.log_note(
                        "watchdog: feed stale - entries halted")
                last_bar_time = bar_time
                t = len(closed) - 1

                states = read_and_featurize(closed, None)
                ctx = EngineContext(df=closed, states=states, years=0.0,
                                    instrument=self.cfg.instrument)
                bar = self._bar_of(closed, t)
                self.paper.advance(bar)
                manager.on_bar(bar, closed)

                if watchdog.entries_allowed:
                    below_floor: list[tuple[str, float]] = []
                    for pb in self._get_playbooks():
                        if not pb.live:
                            continue
                        sigs = pb.live_signal(states, t, closed) or []
                        for sig in sigs:
                            if sig.score < self.cfg.min_score:
                                # Never silent: same treatment the paper
                                # loop got (bugs phase) — collected and
                                # logged once per bar below.
                                below_floor.append((pb.name, sig.score))
                                continue
                            if sig.index - last_signal_bar \
                                    < self.cfg.min_gap_bars:
                                continue
                            ok, reason = engine.gate_live_signal(pb, sig, ctx)
                            if not ok:
                                self.journal.log_note(
                                    f"context gate blocked {pb.name}: {reason}")
                                continue
                            scorer = self._live_scorer(states)
                            table = scorer.table() \
                                if scorer is not None \
                                and hasattr(scorer, "table") else None
                            ctx.forecast_table = table  # the quality dial
                            conf, notes = engine.score_live_signal(pb, sig, ctx)
                            last_signal_bar = sig.index
                            alert = self._emit_alert(closed, states[t], sig,
                                                     None, scorer, engine)
                            if alert is not None:
                                n_alerts_sent += 1
                                self.journal.log_note(
                                    f"live signal {pb.name} {sig.direction} "
                                    f"score={sig.score:.0f} conf={conf:.0f}")
                    self._log_below_floor(
                        below_floor, self.cfg.min_score, t,
                        closed.index[t], prefix="live loop")

                store.save_bars(closed)
                store.save_meta({
                    "last_bar_time": bar_time,
                    "last_signal_bar": last_signal_bar,
                    "instrument": self.cfg.instrument,
                })
                for pb in self._get_playbooks():
                    store.save_playbook(pb.name, pb.export_state())

                # daily summary at UTC day rollover (silent report)
                today = datetime.now(timezone.utc).date().isoformat()
                if today != summary_day:
                    summary_day = today
                    self._send_daily_summary(mt5, n_alerts_sent)
                    n_alerts_sent = 0

                if max_iterations is None:
                    time.sleep(poll_seconds)
        finally:
            watchdog.disarm_deadman("loop exited")
            self.journal.log_note("live loop exited cleanly")

    def _watchdog_alert(self, text: str) -> None:
        """Watchdog alerts: journal + best-effort Telegram via the
        hardened sender (retry/backoff), with flood protection — a
        stuck condition alerts once per 10 minutes, not every loop."""
        self.journal.log_note(f"WATCHDOG: {text}")
        if self._error_deduper.should_send(f"watchdog: {text}"):
            sender = self._telegram_sender()
            if sender is not None and sender.mode == "live":
                sender.send_text(f"GoldenSwing watchdog: {text}",
                                 silent=False)

    def _send_daily_summary(self, mt5, n_alerts_sent: int) -> None:
        """Build and send the day's summary (silent).  Numbers come
        from the journal — the audited record, not a side counter."""
        from datetime import datetime, timezone
        try:
            records = self.journal.records()
        except Exception:
            records = []
        gst = self.guards.state
        balance = None
        drawdown = None
        open_pos = 0
        try:
            acct = mt5.account_state()
            balance = acct.balance
            peak = gst.get("peak_balance") or balance
            drawdown = max(0.0, (peak - balance) / peak) if peak else 0.0
            open_pos = len(mt5.positions())
        except Exception:
            pass
        text = build_daily_summary(
            day=datetime.now(timezone.utc).date().isoformat(),
            records=records, halted=bool(gst.get("halted")),
            halt_reason=str(gst.get("halt_reason") or ""),
            balance=balance, drawdown_pct=drawdown,
            open_positions=open_pos, alerts_sent=n_alerts_sent)
        self.journal.log_note("daily summary sent", summary=text)
        sender = self._telegram_sender()
        if sender is not None and sender.mode == "live":
            sender.send_text(text, silent=True)

    def _telegram_sender(self):
        """The live TelegramSender from the alert service, if any."""
        for s in self.service.senders:
            if isinstance(s, TelegramSender):
                return s
        return None

    def _run_live_legacy(self, *, poll_seconds: int = 20,
                         bars_count: int = 1500,
                         max_iterations: int | None = None) -> None:
        """The pre-Phase-10 live loop (Combo-A detector, no watchdog)."""
        mt5 = MT5Executor(mt5_module=self.cfg.mt5_module)
        ok, msg = mt5.connect(
            path=os.environ.get("GS_MT5_PATH"),
            login=int(os.environ["GS_MT5_LOGIN"])
            if os.environ.get("GS_MT5_LOGIN") else None,
            password=os.environ.get("GS_MT5_PASSWORD"),
            server=os.environ.get("GS_MT5_SERVER"))
        if not ok:
            raise MT5Unavailable(msg)
        self.journal.log_note(f"live runner connected: {msg}")
        self._build_handler(mt5)
        manager = _LiveManager(self, mt5)

        last_bar_time = None
        last_signal_bar = -10 ** 9
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            self._poll_telegram()
            df = mt5.copy_rates_h1(self.cfg.instrument, bars_count)
            closed = df.iloc[:-1]                    # drop the forming bar
            bar_time = closed.index[-1]
            if bar_time != last_bar_time and len(closed) > 100:
                last_bar_time = bar_time
                t = len(closed) - 1
                states = MarketReader().read(closed)
                try:
                    fsets = FeatureEngine().compute(closed, states, None)
                    for s, fs in zip(states, fsets):
                        s.features = fs
                except Exception:
                    fsets = None
                bar = self._bar_of(closed, t)
                self.paper.advance(bar)
                manager.on_bar(bar, closed)
                state = states[t]
                sig = self.detector.evaluate(state)
                if sig is not None and sig.score >= self.cfg.min_score \
                        and sig.index - last_signal_bar \
                        >= self.detector.cfg.min_gap:
                    last_signal_bar = sig.index
                    scorer = self._live_scorer(states)
                    self._emit_alert(closed, state, sig, fsets, scorer)
            if max_iterations is None:
                time.sleep(poll_seconds)

    _scorer_cache: tuple | None = None

    def _live_scorer(self, states):
        """Load the forecaster bundle once per process (live loop).

        Prefers the multi-horizon scorer (1d/3d rank tables — the
        quality dial); falls back to the legacy single-horizon scorer
        when the bundle is the old shape.  Both expose ``note_for``;
        only the multi-horizon one exposes ``table``.
        """
        if self._scorer_cache is None:
            self._scorer_cache = _load_scorer(
                self.cfg.forecast_bundle_path, states)
            self.forecaster_note = self._scorer_cache[1]
        return self._scorer_cache[0]

    # ------------------------------------------------------ telegram polling
    _tg_offset: int = 0

    def _get_command_router(self):
        """The owner-only command router (lazy; shares the poller)."""
        if self._command_router is None:
            from ..alerts.commands import CommandRouter
            self._command_router = CommandRouter(
                owner_chat_id=os.environ.get("GS_TELEGRAM_CHAT_ID"))
        return self._command_router

    def _measured_oos_rows(self) -> list:
        if self._measured_oos is None:
            from .reporting import load_measured_oos
            self._measured_oos = load_measured_oos()
        return self._measured_oos

    def _command_context(self):
        """The read-mostly world one command is allowed to see."""
        from pathlib import Path as _Path

        from ..alerts.commands import CommandContext
        return CommandContext(
            journal=self.journal,
            journal_path=_Path(self.cfg.journal_path),
            paper=self.paper,
            feed_status=dict(self._feed_status),
            last_price=self._last_price,
            last_bar_time=(self._last_bar_time.isoformat()
                           if self._last_bar_time is not None
                           and hasattr(self._last_bar_time, "isoformat")
                           else self._last_bar_time),
            measured_oos=self._measured_oos_rows(),
            reports_dir=_Path(self.cfg.journal_path).parent / "reports",
        )

    def _dispatch_command(self, update: dict) -> None:
        """Answer one owner command (a no-op for everyone else)."""
        router = self._get_command_router()
        result, _chat_id = router.dispatch(update, self._command_context())
        if result is None:
            return                       # not a command / not the owner
        sender = self._telegram_sender()
        if sender is None:
            return                       # nowhere to answer; stay quiet
        try:
            if result.document is not None:
                sender.send_document(result.document,
                                     caption=result.caption)
            elif result.text is not None:
                sender.send_text(result.text, silent=True)
        except Exception:
            pass                         # answering never breaks the loop

    def _poll_telegram(self) -> None:
        """Feed button taps AND owner commands from getUpdates into
        their registries, answering each.

        ONE consumer for both: callback_query updates go to the
        CallbackRegistry, message updates starting with "/" go to the
        CommandRouter.  A second poller would race on the update offset
        and starve one of the two — never add one.

        CUTOVER: when the MT5 engine takes over, set
        ``GS_TELEGRAM_POLL=0`` in this loop's environment — the MT5
        engine owns getUpdates then, and this poller stays silent.

        Transport: env token when configured, otherwise the telegram
        skill's tg-call CLI (vault-backed).  No token, no CLI, no poll.
        """
        import json as _json
        import subprocess as _sp

        if os.environ.get("GS_TELEGRAM_POLL", "1") == "0":
            return  # cutover: the MT5 engine owns getUpdates now
        token = os.environ.get("GS_TELEGRAM_BOT_TOKEN")
        tg_call = os.path.expanduser(
            "~/workspace/skills/telegram/bin/tg-call")
        use_skill = not token and os.path.isfile(tg_call)

        def _api(method: str, payload: dict):
            """One Bot API call; returns the parsed JSON or {}."""
            if token:
                import requests
                url = f"https://api.telegram.org/bot{token}/{method}"
                try:
                    if method == "getUpdates":
                        r = requests.get(url, params=payload, timeout=10)
                    else:
                        r = requests.post(url, data=payload, timeout=10)
                    return r.json()
                except Exception:
                    return {}
            try:
                proc = _sp.run(
                    [tg_call, "--method", method,
                     "--data", _json.dumps(payload)],
                    capture_output=True, text=True, timeout=40)
                return _json.loads(proc.stdout or "{}")
            except Exception:
                return {}

        if not token and not use_skill:
            return
        updates = _api("getUpdates", {"offset": self._tg_offset,
                                      "timeout": 0}).get("result", [])
        for upd in updates:
            self._tg_offset = int(upd["update_id"]) + 1
            query = upd.get("callback_query")
            if query:
                result = self.registry.dispatch(query.get("data", ""),
                                                self._alerts)
                _api("answerCallbackQuery",
                     {"callback_query_id": query["id"],
                      "text": result.message[:200]})
                continue
            # Owner commands share this poller (one consumer, never two).
            self._dispatch_command(upd)
        if updates:
            # Confirm the drained offset server-side.  Without this,
            # Telegram keeps every update unconfirmed and the next
            # --once tick (fresh process, offset reset to 0) re-fetches
            # and re-answers the last 24h of owner commands every hour.
            _api("getUpdates", {"offset": self._tg_offset, "timeout": 0})


# ------------------------------------------------------- live MT5 manager
class _LiveManager:
    """Bar-by-bar Phase-2 exit management for real MT5 positions.

    Positions opened by this runner (auto-execute) or adopted from
    the session's alerts are tracked with the shared ExitStepper:
    partials become partial closes, trail advances become SL moves,
    stop/timeout become full closes.  The broker-side SL placed with
    the order remains the protection of last resort between bars.
    PnL for the journal is computed from the close price the broker
    reports (or the bar close when it reports none).
    """

    def __init__(self, runner: LiveRunner, mt5: MT5Executor,
                 notifier=None):
        self.runner = runner
        self.mt5 = mt5
        self.stepper = ExitStepper(ExitConfig())
        self.tracked: dict[int, object] = {}
        self._banked: dict[int, float] = {}   # partial PnL per ticket
        self.notifier = notifier              # PositionNotifier | None
        self._last_trail_r: dict[int, float] = {}

    def _notify(self, method: str, **kw):
        if self.notifier is None:
            return
        try:
            getattr(self.notifier, method)(**kw)
        except Exception as exc:  # notifying never breaks management
            self.runner.journal.log_note(
                f"position notify {method} failed: {exc}")

    def _adopt(self, position, bar_index: int) -> None:
        alert = self.runner._alerts.get(position.alert_id)
        if alert is None:
            return          # not ours to manage beyond its broker SL
        req = OrderRequest(
            symbol=position.symbol, direction=position.direction,
            volume=position.volume, stop_loss=alert.invalidation,
            entry_ref=alert.entry_ref, targets=list(alert.targets),
            atr=alert.atr, alert_id=alert.alert_id,
            bar_index=alert.bar_index)
        self.tracked[position.ticket] = self.stepper.open(
            ticket=position.ticket, request=req,
            fill_price=position.entry_price, entry_bar=bar_index)
        pos = self.tracked[position.ticket]
        self._notify(
            "on_fill", ticket=position.ticket, alert_id=position.alert_id,
            direction=position.direction, fill_price=position.entry_price,
            volume=position.volume, stop=pos.active_stop,
            instrument=position.symbol)

    def on_bar(self, bar: dict, df: pd.DataFrame) -> None:
        bar_index = int(bar["index"])
        positions = {p.ticket: p for p in self.mt5.positions()
                     if p.magic == self.mt5.magic}
        for ticket, p in positions.items():
            if ticket not in self.tracked:
                self._adopt(p, bar_index)
        for ticket in list(self.tracked):
            pos = self.tracked[ticket]
            live = positions.get(ticket)
            if live is None:                    # closed by broker SL/TP
                del self.tracked[ticket]
                self._banked.pop(ticket, None)
                self._last_trail_r.pop(ticket, None)
                self._notify("on_broker_close", ticket=ticket,
                             alert_id=pos.alert_id,
                             instrument=pos.symbol)
                self.runner.journal.log_note(
                    "tracked MT5 position vanished (broker-side close, "
                    "e.g. the SL placed with the order) — its outcome is "
                    "in the terminal history, not the journal",
                    ticket=ticket, alert_id=pos.alert_id)
                continue
            prev_stop = pos.active_stop
            for ev in self.stepper.step(pos, bar, bar_index):
                if ev.kind == "partial":
                    res = self.mt5.close_position(ticket, ev.volume)
                    fill = res.price if res.price > 0 else (ev.level or 0.0)
                    spec = self.runner.spec
                    pnl = pos.direction * (fill - pos.entry_price) \
                        / spec.tick_size * spec.tick_value * ev.volume
                    self._banked[ticket] = self._banked.get(ticket, 0.0) \
                        + pnl
                    self.stepper.apply(pos, ev)
                    self._notify(
                        "on_partial", ticket=ticket,
                        direction=pos.direction, fill_price=fill,
                        volume_closed=ev.volume, pnl_money=pnl,
                        instrument=pos.symbol)
                else:
                    res = self.mt5.close_position(ticket)
                    self._journal_close(pos, ev, res, bar)
                    self._last_trail_r.pop(ticket, None)
                    del self.tracked[ticket]
                    break
            else:
                if ticket in self.tracked and pos.active_stop != prev_stop \
                        and not pos.closed:
                    self.mt5.modify_sl(ticket, pos.active_stop)
                    self._notify_stop_move(pos, prev_stop)

    def _notify_stop_move(self, pos, prev_stop: float) -> None:
        """Narrate stop moves: breakeven crosses always, trail locks when
        they bank another half-R (per-ticket throttle — a trail that
        creeps every bar is not worth a buzz every bar)."""
        if self.notifier is None or pos.risk_price <= 0:
            return
        d = pos.direction
        new_stop = pos.active_stop
        crossed_be = (d == 1 and prev_stop < pos.entry_price <= new_stop) \
            or (d == -1 and prev_stop > pos.entry_price >= new_stop)
        if crossed_be:
            self._last_trail_r[pos.ticket] = 0.0
            self._notify("on_breakeven", ticket=pos.ticket, direction=d,
                         instrument=pos.symbol)
            return
        locked_r = d * (new_stop - pos.entry_price) / pos.risk_price
        last = self._last_trail_r.get(pos.ticket, float("-inf"))
        if locked_r - last >= 0.5:
            self._last_trail_r[pos.ticket] = locked_r
            self._notify("on_trail", ticket=pos.ticket, direction=d,
                         new_stop=new_stop, locked_r=locked_r,
                         instrument=pos.symbol)

    def _journal_close(self, pos, ev, res, bar: dict) -> None:
        spec = self.runner.spec
        exit_price = res.price if res and res.price > 0 else float(bar["close"])
        final = pos.direction * (exit_price - pos.entry_price) \
            / spec.tick_size * spec.tick_value * ev.volume
        pnl = self._banked.pop(pos.ticket, 0.0) + final
        risk_money = spec.money_for_move(pos.risk_price, pos.orig_volume)
        r_net = pnl / risk_money if risk_money > 0 else 0.0
        self.runner.journal.log_close(
            alert_id=pos.alert_id,
            ticket=pos.ticket, exit_price=exit_price, reason=ev.reason,
            pnl_money=pnl,
            r_net=r_net,
            hold_bars=int(bar["index"]) - pos.signal_bar,
            partial_taken=pos.partial_done)
        self._notify(
            "on_close", ticket=pos.ticket, direction=pos.direction,
            exit_price=exit_price, reason=ev.reason, pnl_money=pnl,
            r_net=r_net, hold_bars=int(bar["index"]) - pos.signal_bar,
            instrument=pos.symbol)
        acct = self.mt5.account_state()
        self.runner.guards.record_outcome(pnl, acct.balance)


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="GoldenSwing live runner")
    ap.add_argument("--live", action="store_true",
                    help="run against MT5 (Windows host, active terminal)")
    ap.add_argument("--auto", action="store_true",
                    help="auto-execute golden signals (GS_AUTO_EXECUTE)")
    ap.add_argument("--max-bars", type=int, default=None)
    ap.add_argument("--symbol", type=str, default=None)
    ap.add_argument("--balance", type=float, default=None)
    ap.add_argument("--min-score", type=float, default=None)
    args = ap.parse_args(argv)

    overrides = {}
    if args.symbol:
        overrides["instrument"] = args.symbol
    if args.balance:
        overrides["paper_balance"] = args.balance
    if args.min_score:
        overrides["min_score"] = args.min_score
    if args.auto:
        overrides["auto_execute"] = True
    cfg = RunnerConfig.from_env(**overrides)
    runner = LiveRunner(cfg)

    if args.live:
        runner.run_live()
        return
    # Dry mode is offline BY DESIGN (same posture as Phase 7's demo):
    # strip live-send credentials so alerts land in the outbox.
    for var in ("GS_TELEGRAM_BOT_TOKEN", "GS_TELEGRAM_CHAT_ID",
                "GS_SMTP_HOST", "GS_ALERT_EMAIL_TO"):
        os.environ.pop(var, None)
    runner = LiveRunner(cfg)
    summary = runner.run_bars(max_bars=args.max_bars)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
