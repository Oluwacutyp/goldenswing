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

import hashlib
import json
import os
import math
import secrets
import subprocess
import time
import uuid
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..alerts.commands import CommandContext, is_owner_callback
from ..backtest.data import REPO_ROOT
from ..charts.engine import render_mt5_signal, zones_to_study_dicts
from ..charts.signal_view import build_signal_view_html
from ..charts.terminal_server import TerminalServer
from ..confluence import ConfluenceEngine
from ..features.engine import read_and_featurize
from ..feed.tick_log import TickLogger
from ..live.asymmetry import (STYLE_FLOORS, ScoreContext, floor_for,
                              score_asymmetry)
from ..live.atr_regime import (atr_regime as _atr_regime,
                               baseline_atr as _baseline_atr,
                               scaled_thresholds as _scaled_thresholds)
from ..live.dxy import DXYTracker
from ..live import day_context as _day_context
from ..live import posture as _posture_mod
from ..live.funding import (SMT_PLAYBOOKS as _SMT_PLAYBOOKS,
                            FundingReport, detect_funding_events,
                            smt_funding_event)
from ..live.health import HealthMonitor
from ..live.management_profiles import (
    drift_triggered as _drift_triggered,
    elapsed_h1_bars as _mgmt_elapsed_h1,
    mgmt_clock_due as _mgmt_clock_due,
    profile_for as _mgmt_profile_for, progress_due as _progress_due,
    tp_ladder as _tp_ladder)
from ..live.news import preprint_action as _preprint_action
from ..live.smt import detect_smt as _detect_smt
from ..live.opposing import risk_tier as _zone_risk_tier, \
    tier_spec as _tier_spec
from ..live.sizing import RiskPool, size_for_risk
from ..forecast.mt5 import MT5Forecaster
from ..live.urgency import urgency_score
from ..live.volume_profile import volume_profile
from ..live.hot_scalp import HotScalpConfig, HotScalpResearchEngine
from ..live.multitf import atr14 as _atr14
from ..live.evidence import (BASE_CATEGORY_WEIGHTS, EvidenceItem, score_evidence)
from ..live.stacking import evaluate_layer_add as _evaluate_layer_add
from ..live.patterns import (detect_sweep as _ev_detect_sweep,
                             liquidity_pools as _ev_liquidity_pools,
                             order_blocks as _ev_order_blocks,
                             premium_discount as _ev_premium_discount,
                             session_label as _ev_session_label,
                             structure_event as _ev_structure_event,
                             trend_bias as _ev_trend_bias,
                             unfilled_fvgs as _ev_unfilled_fvgs,
                             volume_confirm as _ev_volume_confirm)
from ..live.candle_taxonomy import detect_named_patterns as _ev_named_candles
from ..live.setup_store import SetupStore, SetupStoreError
from ..live.hierarchy import (LEG_ORDER, TimeframeStack, TF_MINUTES,
                              check_live_trigger,
                              check_zone_touch, evaluate_hierarchy,
                              live_setup, live_setup_hold_state,
                              live_setup_reason, log_feed_error,
                              log_hold_state, qm_annotation,
                              resolve_setup_for_touch, zones_by_side)
from ..live.exit import partial_r, realized_r
from ..live.multitf import atr14
from ..live.range import RANGE_SIZE_MULT, BoundaryTracker, RangeDef
from ..live.regime_state import (RegimeState, load_regime_state,
                                 save_regime_state)
from ..live.zones import (TF_ORDER as _TF_ORDER, ZoneCache,
                          ZoneStore, find_clusters)
from ..live.macrotrend import (MacroTrendResult, compute_macrotrend)
from ..live.narrative import (ENTRY as _NARR_ENTRY,
                              NarrativeState, advance as advance_narrative,
                              amd_for_state, finalize_exit as
                              _finalize_narrative_exit, load_narrative_state,
                              mark_entered as _mark_narrative_entered,
                              save_narrative_state)
from ..live import market_hours as _market_hours
from ..live.manipulation import chase_blocked
from ..live.news import NewsCalendar
from ..live.reasoning import build_reasoning, build_study
from ..live.regime import classify_regime
from ..live.volume import volume_features
from ..reader.session_timeframes import (
    compose as _compose_session, session_of)
from ..playbooks import (FVGLimitPlaybook, FVGMidlinePlaybook,
                         InversionPlaybook, JudasSwingPlaybook,
                         KillzoneOpenPlaybook, NarrativePlaybook,
                         OrderBlockPlaybook, QMD1Playbook, QMM5Playbook,
                         RangeCompressionBreakoutPlaybook,
                         RangeMeanReversionPlaybook, RangeQMPlaybook,
                         RangeSweepReclaimPlaybook,
                         RetracementTrapPlaybook,
                         SweepReversalPlaybook, TrendlineRetestPlaybook,
                         TurtleSoupPlaybook, VReversalPlaybook)
from ..playbooks.base import EngineContext
from .entry_gates import (
    DEFAULT_SESSION_MULTIPLIERS, EntryGate, EntryGateConfig,
    is_friday_close, session_multiplier, signal_fingerprint)
from .executor import OrderRequest, PendingOrder
from ..engine.runtime import DeterministicRuntime
from .learning import TradeJournal
from .mt5 import MT5Executor, build_ladder, select_order_type
from .funnel import (assessment_block as _assessment_block,
                     classify_reason as _funnel_classify,
                     decision_bucket as _decision_bucket,
                     terminal_decision as _terminal_decision,
                     verdict_blocks as _verdict_blocks)
from .assessment_render import (is_shadow_record as
                                _is_shadow_record,
                                render_assessment as
                                _render_assessment)
from . import candidate_lifecycle as _lifecycle
from . import journal_audit as _journal_audit
from . import setup_lifecycle as _setup_lc
from .setup_lifecycle import (SetupLifecycleStore as _SetupStore,
                              TrackedSetup as _TrackedSetup)
from .friction_gate import (
    FRICTION_REASON_CODE as _FRICTION_CODE,
    assess_entry_friction as _assess_friction)
from .signal_log import log_signal
from .mt5_commands import (mt5_back_keyboard, mt5_confirm_keyboard,
                           mt5_control_keyboard, mt5_nav_keyboard,
                           mt5_router, parse_mt5_callback)
from .mt5_config import LiveConfigStore, unknown_disabled_names
from .playbook_registry import (FLOOR_AUTHORITY as _FLOOR_AUTHORITY,
                                PLAYBOOK_REGISTRY, RANGE_SPECS,
                                ROSTER_SPECS, ROSTER_SPECS_H1,
                                decay_exempt as _decay_exempt,
                                native_tf_of as _native_tf)
from .risk import RiskConfig, RiskGuards
from .tick_manager import (DEFAULT_TIME_EXIT_BARS, TickAction, TickConfig,
                          TickManager, sl_level_valid)
from ..feed.mt5 import TRADE_MODE_REAL, MT5Feed, TickPump

DATA_DIR = REPO_ROOT / "data"

# Volume tolerance (lots) for fill comparisons: a broker fill
# within this of the pending volume counts as full — float dust
# must not manufacture a phantom partial fill.
_VOL_EPS = 1e-9

_log = logging.getLogger(__name__)


def _session_multipliers_from_env() -> dict:
    """GS_SESSION_MULTIPLIERS is a JSON object {session: mult}; bad
    JSON or a non-object fails closed to the defaults (never a
    half-applied map)."""
    raw = os.environ.get("GS_SESSION_MULTIPLIERS", "")
    if not raw:
        return dict(DEFAULT_SESSION_MULTIPLIERS)
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return dict(DEFAULT_SESSION_MULTIPLIERS)
        out = dict(DEFAULT_SESSION_MULTIPLIERS)
        for key, value in parsed.items():
            out[str(key)] = float(value)
        return out
    except (ValueError, TypeError):
        return dict(DEFAULT_SESSION_MULTIPLIERS)


def _time_exit_bars_from_env() -> dict | None:
    """GS_TIME_EXIT_BARS is a JSON object {structure: bars}; bad JSON
    or a non-object fails closed to the engine-derived default map
    (DEFAULT_TIME_EXIT_BARS).  A valid env JSON replaces the whole
    map."""
    raw = os.environ.get("GS_TIME_EXIT_BARS", "")
    if not raw:
        return dict(DEFAULT_TIME_EXIT_BARS)
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return dict(DEFAULT_TIME_EXIT_BARS)
        return {str(key): int(value) for key, value in parsed.items()}
    except (ValueError, TypeError):
        return dict(DEFAULT_TIME_EXIT_BARS)


def _tick_now_utc(last_tick: dict | None) -> datetime | None:
    """The tick's own timestamp as an aware UTC datetime (the engine's
    clock is the tick clock).  None when the tick carries no usable
    time — callers fall back to the wall clock."""
    try:
        ts = pd.Timestamp((last_tick or {}).get("time"))
        if ts is None or pd.isna(ts):
            return None
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC").to_pydatetime()
    except Exception:
        return None


@dataclass
class MT5RunnerConfig:
    symbol: str = "XAUUSD"
    risk_pct: float = 0.01
    magic: int = 260101
    # A14 UNSHADOW (owner addendum 2026-10-21): the shadow tier is
    # abolished on the MT5 path — a log-only signal can never be
    # confirmed, and the demo engine IS the shadow.  None = ALL
    # playbooks evaluated on the H1 path execute through the same
    # decision flow (score floors, RiskPool, hard lines).  An
    # explicit tuple remains possible as a config override.
    # Capital protection is NOT this list: it is the RiskPool
    # ceiling + StyleBook per-style caps + the RiskGuards
    # max-drawdown halt (spec §11 A14).  The per-playbook kill switch is
    # ``disabled_playbooks`` below (default: all enabled).
    execute_playbooks: tuple | None = None
    min_score: float = 60.0
    # --- Confluence rework (spec v2, arrival pass) ------------------
    # Sizing (owner ruling 2026-10-23, REVOKING A13's fixed-lots
    # override; floor rule per owner ruling 2026-10-24; legacy
    # fence per owner rulings enforcement 2026-10-25): dynamic
    # risk sizing is the SOLE sizing path for orders.
    # "risk" = sizes computed from risk money in Decimal
    # (live/sizing.py), ledgered per position in a RiskPool; the
    # legacy float path (risk.py sizer) remains ONLY for
    # evaluation/forensics — every order path refuses a non-risk
    # sizing mode with a journal note instead of placing an order.
    # There is NO fixed-lots mode and no fixed_lots field: a size
    # always derives from (k x equity x C x S) / (stop ticks x
    # tick value), step-quantized down; a step-quantized size below
    # the broker minimum FLOORS at the minimum with min_override
    # (trade, never abandon); sizes quantize down, never round up.
    sizing_mode: str = "risk"
    # A6: resting limits at FRESH MAJOR POIs, placed ahead of
    # arrival at the zone's proximal edge.
    resting_limit_enabled: bool = True
    # §5 PullbackWatch: H1 bars a runaway watch survives awaiting
    # the pullback before it lapses.
    pullback_watch_bars: int = 12
    signal_gap_bars: int = 0
    journal_path: Path = DATA_DIR / "mt5_journal.jsonl"
    # Deterministic engine: when True the runner dual-writes every
    # decision-critical frame to goldenswing/engine DeterministicJournal
    # (chained, fenced, idempotent) alongside the legacy JSONL audit
    # journal.  Legacy journal is never removed; deterministic journal
    # is the authority for seq / chain / fencing / idempotency.
    use_deterministic_engine: bool = True
    # FUNNEL (§1): decision-funnel instrumentation — one evaluation
    # record per playbook per evaluation clock in the journal,
    # including no-signal outcomes.  OBSERVATION-ONLY: the records
    # never change a decision (test-pinned on/off identity).  The
    # flag exists so a test (or a forensic A/B) can prove that.
    funnel_enabled: bool = True
    deterministic_dir: Path | None = None  # default: <journal parent>/deterministic
    risk_state_path: Path = DATA_DIR / "mt5_risk_state.json"
    kill_switch_path: Path = DATA_DIR / "mt5_KILL"
    hierarchy_required: bool = True
    tick_interval: float = 0.25
    allow_live: bool = False  # True ONLY for an explicit real-money run
    #: Terminal exe to pin initialize(path=...) to (feed + executor).
    #: None = MT5_TERMINAL_PATH env, else bare initialize() — which
    #: can attach to a different MT5 install than the one on screen
    #: (the 2026-10-16 backtest failure).  Set via --terminal-path.
    terminal_path: str | None = None
    # Interactive terminal (step 8): the engine serves its own live
    # chart page. Localhost-only by default — RDP + localhost on the
    # box reaches it with zero new network exposure. A port opening
    # or tunnel is a host network change, never a default. (Not to
    # be confused with terminal_path above, which is the MT5 exe.)
    terminal_host: str = "127.0.0.1"
    terminal_port: int = 8899
    max_positions: int = 1
    reconcile_every_ticks: int = 20
    # Telegram command polling: ON by default — this engine OWNS
    # getUpdates.  Pass poll_commands=False (CLI --no-poll-commands)
    # only while the OANDA loop still polls (overlap window).
    poll_commands: bool = True
    poll_interval_seconds: float = 30.0
    # Tick CSV logging (per-day CSVs next to the journal).
    tick_log: bool = True
    # Tick-CSV retention (UTC days): day files older than this are
    # pruned on day rollover and at startup — EXCEPT every file from
    # the earliest open position's open day onward, which is never
    # evicted while that position is open.  0 = keep only today +
    # open-position days.
    tick_csv_retain_days: int = 30
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
    # Margin checks (owner's-bot floor): every order must clear the
    # broker-computed margin + a buffer, and a Golden Swing base
    # order / any layer must leave the post-order free margin at or
    # above the shield (free/equity).  Config-file / env only —
    # deliberately NOT /set-editable.
    margin_buffer_pct: float = 5.0
    margin_shield_pct: float = 40.0
    # Alignment is a SIZING + CONFIRMATION layer, never a hierarchy
    # gate leg.  Sizing: Golden Swing (D1/W1/MN1 agree) multiplies
    # base-order size by 1.5; D1/W1 conflict multiplies by 0.5.
    # Confirmation: a Golden Swing base order WAITS for the owner's
    # Confirm tap (REJECT cancels; timeout fires or cancels per
    # golden_swing_confirm_timeout_action).  Layers are NOT gated
    # (they scale via the layer risk budget).  Config-file / env
    # only — deliberately NOT /set-editable.
    golden_swing_sizing_enabled: bool = True
    golden_swing_confirm_enabled: bool = True
    golden_swing_confirm_timeout_sec: float = 300.0
    # Timeout action for an unanswered Golden Swing confirm:
    # "fire" places the staged order (unattended trading continues),
    # "cancel" drops it.  Default "fire": a missed tap must not
    # silently kill the engine's trading while the owner sleeps.
    golden_swing_confirm_timeout_action: str = "fire"
    # News pause + news-adaptive sizing (owner's-bot floor): no NEW
    # entries inside a high-impact window; size blends down in the
    # 90-min post-news cooldown.  Management is never paused.
    # Config-file / env only — deliberately NOT /set-editable.
    news_pause_enabled: bool = True
    # --- P3 trader build (spec v3 §12) ------------------------------
    # Breakeven doctrine (S9/F13): "earned" — breakeven arms on a
    # funding/structure event or TP1 bank, price touch demoted to
    # the backstop.  "touch" restores the legacy +1R rule.
    breakeven_mode: str = "earned"
    breakeven_backstop_r: float = 2.0
    # S5 pre-print position rule action for intraday exposure
    # approaching an EXTREME print: "be" (stop to breakeven) |
    # "trim" (bank half) | "be_trim" (both).
    news_preprint_action: str = "be"
    # S11 per-style daily trade ceiling (churn brake): a style
    # whose playbooks have opened this many trades today stops
    # opening more today.  0 = no ceiling.
    max_daily_trades_per_style: int = 12
    # Journey trading (G): counter-journey trades need a nested
    # zone AND at least this much room (R multiples of the stop)
    # to the opposing magnet.
    counter_journey_min_room_r: float = 3.0
    # Entry gates (owner's-bot floor): per-day / per-hour trade caps,
    # cooldown after a close, fingerprint dedup.  In-memory per
    # process run (a restart resets them — see entry_gates.py).
    # Config-file / env only — deliberately NOT /set-editable.
    entry_gates_enabled: bool = True
    max_daily_trades: int = 0   # 0 = unlimited (owner's default)
    max_hourly_trades: int = 0  # 0 = unlimited (owner's default)
    loss_cooldown_sec: float = 300.0  # owner's 300s
    win_cooldown_sec: float = 0.0
    fingerprint_ttl_sec: float = 3600.0
    # Session sizing + Friday close (owner's-bot floor): base-order
    # size multiplied by the session multiplier (DST-aware sessions);
    # no new base entries Friday at/after friday_close_hour UTC.
    # friday_close_action: "hold" (positions ride the weekend, the
    # broker SL guards them) or "close" (all positions + pendings
    # closed at the Friday hour).  Config-file / env only.
    session_multipliers: dict = field(
        default_factory=lambda: dict(DEFAULT_SESSION_MULTIPLIERS))
    friday_close_hour: int = 20  # owner's friday_close_hour
    friday_close_action: str = "hold"
    # Scalp holding cap (owner's-bot floor): positions whose playbook
    # is in scalp_playbooks close after scalp_cap_minutes of
    # wall-clock hold.  Default empty — DELIBERATELY, from a
    # playbook-by-playbook audit: every live playbook is H1-evaluated
    # with 1.5R structural targets and 48–72-bar horizons (intraday /
    # swing, not scalps — a 15-min cap would cut them before their
    # structure can resolve).  qm_m5 trades live on its native M5
    # route (funnel §2) and stays out of the default list on the
    # same merits: its hold horizon is pattern-defined (the QM
    # trade expires 48 M5 bars after the touch), so a 15-minute
    # cap would cut its positions before their structure resolves.
    # Tagging it WOULD cap its positions — the empty default is a
    # real choice, not inert config.  Config-file / env only.
    scalp_playbooks: tuple = ()
    scalp_cap_minutes: float = 15.0  # owner's scalping_max_holding
    # The first Hot Scalp implementation is a distinct, persistent RESEARCH
    # sleeve. It journals candidates from M1 reactions + M5 structure + M15
    # context, but deliberately has no order-dispatch path.
    hot_scalp_shadow_enabled: bool = True
    hot_scalp_db_path: Path | None = None
    # Step 6 promotion switch — DEFAULT ON per owner order
    # 2026-10-27 ("I meant live"; "I don't have time to be
    # configuring"): pulling the code enables live Hot Scalp
    # authority with zero host configuration.  Still
    # config-file/env-overridable (set GS_HOT_SCALP_LIVE=0
    # or the config flag False to disable) and still NOT
    # /set-editable — parse_set allowlists only max_layers,
    # manipulation_filter, pending_expiry_bars, playbook.*.
    # While OFF, the promotion method is never called from
    # the live loop, and it refuses even when called
    # directly (defense in depth).
    hot_scalp_live_enabled: bool = True
    # DXY correlation (owner's-bot floor): the XAUUSD vs DXYm
    # correlation is confluence context (Ghost Sentinel bands,
    # bounded ±0.5) — never a gate leg.  A dead DXY feed degrades
    # to unavailable, never fails the gold path.  Config-file /
    # env only — deliberately NOT /set-editable.
    dxy_enabled: bool = True
    dxy_symbol: str = "DXYm"  # owner's bot correlation_symbol
    # Pending-order score decay (owner's-bot PendingOrderTracker):    # on each H1 close the originating playbook is re-evaluated; a
    # pending whose setup no longer emits a same-direction signal at
    # this level scoring at/above the decay floor accrues a decay
    # bar, and at score_decay_bars it is cancelled.  Floor None =
    # min_score (the floor that would have blocked it at birth).
    # Config-file / env only — deliberately NOT /set-editable.
    score_decay_enabled: bool = True
    score_decay_floor: float | None = None
    score_decay_bars: int = 2
    # Armed-state TTL (owner rule, 2026-10-15): an armed zone
    # touch disarms as stale after armed_ttl_bars ENTRY-TF bars
    # without confirmation.  Counted in entry-TF bars (the state
    # machine runs on the entry clock), not wall clock.  The
    # number 2 mirrors score_decay_bars (the codebase's existing
    # "two bars of no-confirmation = stale" rule), counted on the
    # entry clock instead of H1 closes.  On M15 it binds after
    # 2 closes (30 min); on M30 it coincides with the H1-close
    # expiry.  Config-file / env only.
    armed_ttl_bars: int = 2
    # Multi-zone limit ladder (owner's-bot floor): a LIMIT order's
    # size staggers across rungs from the playbook level to the
    # adverse zone edge; only rung 0 is placed, the next rung
    # advances on expiry.  Default OFF: it changes fill behavior
    # (smaller first rung), so it is opt-in.  Config-file / env
    # only — deliberately NOT /set-editable.
    ladder_enabled: bool = False
    ladder_rungs: int = 3
    ladder_split: tuple = (0.5, 0.3, 0.2)
    # structure.  Default DEFAULT_TIME_EXIT_BARS (engine-derived:
    # 72-bar swing horizon for clean_trend/unclear, shorter for
    # faster-resolving structures — see tick_manager).  Env JSON
    # (GS_TIME_EXIT_BARS) replaces the whole map.  Config-file /
    # env only.
    time_exit_bars: dict | None = field(
        default_factory=lambda: dict(DEFAULT_TIME_EXIT_BARS))

    @classmethod
    def from_env(cls, **overrides) -> "MT5RunnerConfig":
        cfg = cls(
            symbol=os.environ.get("GS_SYMBOL", "XAUUSD"),
            risk_pct=float(os.environ.get("GS_RISK_PCT", "0.01")),
            allow_live=os.environ.get("GS_MT5_ALLOW_LIVE", "0") == "1",
            poll_commands=os.environ.get("GS_MT5_POLL_COMMANDS", "1") != "0",
            terminal_host=os.environ.get("GS_TERMINAL_HOST", "127.0.0.1"),
            terminal_port=int(os.environ.get("GS_TERMINAL_PORT", "8899")),
            poll_interval_seconds=float(
                os.environ.get("GS_MT5_POLL_INTERVAL", "30.0")),
            pending_expiry_bars=int(
                os.environ.get("GS_MT5_PENDING_EXPIRY_BARS", "4")),
            golden_swing_sizing_enabled=os.environ.get(
                "GS_GOLDEN_SWING_SIZING", "1") != "0",
            golden_swing_confirm_enabled=os.environ.get(
                "GS_GOLDEN_SWING_CONFIRM", "1") != "0",
            golden_swing_confirm_timeout_sec=float(os.environ.get(
                "GS_GOLDEN_SWING_CONFIRM_TIMEOUT_SEC", "300")),
            golden_swing_confirm_timeout_action=os.environ.get(
                "GS_GOLDEN_SWING_CONFIRM_TIMEOUT_ACTION", "fire"),
            news_pause_enabled=os.environ.get(
                "GS_NEWS_PAUSE", "1") != "0",
            entry_gates_enabled=os.environ.get(
                "GS_ENTRY_GATES", "1") != "0",
            max_daily_trades=int(os.environ.get(
                "GS_MAX_DAILY_TRADES", "0")),
            max_hourly_trades=int(os.environ.get(
                "GS_MAX_HOURLY_TRADES", "0")),
            loss_cooldown_sec=float(os.environ.get(
                "GS_LOSS_COOLDOWN_SEC", "300")),
            win_cooldown_sec=float(os.environ.get(
                "GS_WIN_COOLDOWN_SEC", "0")),
            fingerprint_ttl_sec=float(os.environ.get(
                "GS_FINGERPRINT_TTL_SEC", "3600")),
            session_multipliers=_session_multipliers_from_env(),
            friday_close_hour=int(os.environ.get(
                "GS_FRIDAY_CLOSE_HOUR", "20")),
            friday_close_action=os.environ.get(
                "GS_FRIDAY_CLOSE_ACTION", "hold"),
            scalp_playbooks=tuple(
                p.strip() for p in os.environ.get(
                    "GS_SCALP_PLAYBOOKS", "").split(",") if p.strip()),
            scalp_cap_minutes=float(os.environ.get(
                "GS_SCALP_CAP_MINUTES", "15")),
            time_exit_bars=_time_exit_bars_from_env(),
            margin_buffer_pct=float(os.environ.get(
                "GS_MT5_MARGIN_BUFFER_PCT", "5.0")),
            margin_shield_pct=float(os.environ.get(
                "GS_MT5_MARGIN_SHIELD_PCT", "40.0")),
            dxy_enabled=os.environ.get("GS_DXY_ENABLED", "1") != "0",
            dxy_symbol=os.environ.get("GS_DXY_SYMBOL", "DXYm"),
            score_decay_enabled=os.environ.get(
                "GS_SCORE_DECAY_ENABLED", "1") != "0",
            score_decay_floor=(
                float(os.environ["GS_SCORE_DECAY_FLOOR"])
                if os.environ.get("GS_SCORE_DECAY_FLOOR") else None),
            score_decay_bars=int(os.environ.get(
                "GS_SCORE_DECAY_BARS", "2")),
            armed_ttl_bars=int(os.environ.get(
                "GS_ARMED_TTL_BARS", "2")),
            ladder_enabled=os.environ.get(
                "GS_LADDER_ENABLED", "0") != "0",
            ladder_rungs=int(os.environ.get("GS_LADDER_RUNGS", "3")),
            ladder_split=tuple(
                float(x) for x in os.environ.get(
                    "GS_LADDER_SPLIT", "0.5,0.3,0.2").split(",")
                if x.strip()),
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def __post_init__(self) -> None:
        # A bad timeout action (hand-edited config) must not
        # silently change order behavior: coerce to "fire".
        if self.golden_swing_confirm_timeout_action not in ("fire",
                                                             "cancel"):
            self.golden_swing_confirm_timeout_action = "fire"
        # Same for the Friday-close action: unknown -> "hold".
        if self.friday_close_action not in ("hold", "close"):
            self.friday_close_action = "hold"
        env_shadow = os.environ.get("GS_HOT_SCALP_SHADOW")
        if env_shadow is not None:
            self.hot_scalp_shadow_enabled = env_shadow.strip().lower() in (
                "1", "true", "yes", "on")
        env_live = os.environ.get("GS_HOT_SCALP_LIVE")
        if env_live is not None:
            self.hot_scalp_live_enabled = env_live.strip().lower() in (
                "1", "true", "yes", "on")
        if self.hot_scalp_db_path is None:
            self.hot_scalp_db_path = self.journal_path.parent / "hot_scalp_setups.db"


def execution_intent_for(pb_name: str, sig=None) -> dict:
    """Step 15 (owner rulings 2026-10-30 §2): a candidate's DECLARED
    execution intent — which order family its entry thesis calls for.

    Returns {"family": "limit" | "market", "basis": str}.  The
    registry is the declaration of record for roster playbooks
    (PlaybookSpec.entry_thesis); the routing taxonomy names the
    rest: a ``breakout`` playbook/signal rides the break itself,
    hot_scalp is market-shape only by its v1 design, and
    breakout-shaped sleeve kinds are displacement entries.
    Everything else — every pullback / mitigation / retracement /
    return-to-zone premise, declared or not — is the limit family:
    the entry rests at the setup's level and a miss is the thesis
    fulfilled, never a chase.  Pure: no state, no I/O.
    """
    name = str(pb_name or "")
    kind = str(getattr(sig, "kind", "") or "")
    if name == "breakout" or kind == "breakout":
        return {"family": "market",
                "basis": "confirmed breakout — the entry rides "
                         "the break of structure itself"}
    if kind in ("wedge_breakout", "squeeze_breakout"):
        return {"family": "market",
                "basis": f"{kind}: displacement breakout entry"}
    if name == "hot_scalp":
        return {"family": "market",
                "basis": "hot scalp v1 is market-shape only"}
    for spec in PLAYBOOK_REGISTRY:
        if spec.name == name:
            if spec.entry_thesis == "market":
                return {"family": "market",
                        "basis": f"{name} declares a market thesis "
                                 "(immediate reaction / displacement "
                                 "continuation — entry on the "
                                 "trigger, per the playbook's own "
                                 "definition)"}
            return {"family": "limit",
                    "basis": f"{name} declares a limit thesis "
                             "(pullback / mitigation / "
                             "return-to-zone — the entry rests at "
                             "the level)"}
    return {"family": "limit",
            "basis": "default limit thesis (pullback / "
                     "return-to-zone)"}


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

    Step 4 (2026-10-27) route-aware fields: ``premise`` records
    WHICH setup the pending belongs to — "h1" for pendings under
    the H1 trigger premise (``_live_setup``) and "native" for
    pendings placed by a native-clock route (qm_m5 / qm_d1), which
    ``_live_setup`` never arms.  ``entry_clock`` is the playbook's
    generation clock (registry ``native_tf``, "H1" fallback for
    adopted / unknown playbooks); ``entry_placed_bar`` and
    ``entry_expiry_bar`` are the placement / expiry pair counted
    in ENTRY-CLOCK bars, so a native pending is governed on its
    own clock.  For H1 pendings entry_* == placed / expiry by
    construction, so H1 behaviour is unchanged.
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
    premise: str = "h1"             # "h1" | "native" (step 4)
    entry_clock: str = "H1"         # playbook generation clock (step 4)
    entry_placed_bar: int = 0       # forming-bar index on entry clock
    entry_expiry_bar: int = 0       # first entry-clock close index at
                                    # which the pending dies unfilled
    track_request: object = None    # OrderRequest for TickManager on fill
    sig: object = None              # playbook signal (fill chart/alert)
    verdict: dict | None = None     # hierarchy verdict (fill chart)
    is_layer: bool = False          # pending fills onto the held position
    # Score-decay (owner's-bot PendingOrderTracker): the DECISION
    # score at placement (Asymmetry total when the Score object
    # exists — never the sleeve-base placeholder, step 2
    # 2026-10-27) + consecutive H1 closes on which a re-evaluation
    # found no same-direction signal at this level scoring at/above
    # the decay floor.  At score_decay_bars the pending is cancelled
    # (the setup died while the order rested — a real trader does
    # not leave a limit at a dead setup).
    placed_score: float = 0.0
    decay_bars: int = 0
    # Limit ladder: remaining rungs (levels + lots) after the live
    # rung.  On expiry the next rung advances; on fill the rest are
    # discarded (further entries go through layers, not the ladder).
    ladder_levels: list = field(default_factory=list)
    ladder_lots: list = field(default_factory=list)


def _telegram_missing_vars(environ=None) -> list[str]:
    """Which Telegram vars are absent from the process environment."""
    env = os.environ if environ is None else environ
    return [v for v in ("GS_TELEGRAM_BOT_TOKEN", "GS_TELEGRAM_CHAT_ID")
            if not env.get(v)]


def _repo_dotenv_path() -> Path:
    """Repo-root .env path (goldenswing/execution/ -> repo root)."""
    return Path(__file__).resolve().parent.parent.parent / ".env"


def _telegram_startup_warnings(senders: list) -> list[str]:
    """Accurate Telegram startup warnings for this session.

    Returns [] when at least one sender can actually deliver
    (live/skill mode, or an unknown duck-type sender assumed
    deliverable) — 'not configured' must NEVER print when a
    sender exists.  Two warning cases:

    * no senders: name what is missing, where it was looked for,
      and why yesterday worked while today does not ($env:
      exports are per-PowerShell-session; yesterday's session
      had them exported, a fresh one does not).  A flat
      'not configured' contradicts the owner's evidence that
      notifications + commands worked yesterday.
    * senders exist but all outbox-only (chat id set, no token):
      the engine would look configured while staying mute —
      the opposite false claim.
    """
    if not senders:
        missing = _telegram_missing_vars()
        dotenv_path = _repo_dotenv_path()
        return [(
            "MT5 startup: WARNING: no Telegram sender configured "
            "in THIS session — startup pings + command replies "
            "will NOT reach Telegram. Missing: "
            + ", ".join(missing) + ". Checked process environment "
            "(scripts/run_mt5.py also loads a repo-root .env at "
            f"{dotenv_path} before this point). $env:GS_TELEGRAM_* "
            "exports live only in the PowerShell session that set "
            "them — yesterday's session had them exported, a "
            "fresh one does not. Put GS_TELEGRAM_BOT_TOKEN and "
            f"GS_TELEGRAM_CHAT_ID in {dotenv_path} and restart — "
            ".env persists across sessions.")]
    if all(getattr(s, "mode", "live") == "outbox" for s in senders):
        return [(
            "MT5 startup: WARNING: Telegram sender is outbox-only "
            "(GS_TELEGRAM_CHAT_ID is set but GS_TELEGRAM_BOT_TOKEN "
            "is missing) — pings will NOT reach Telegram. Set "
            "GS_TELEGRAM_BOT_TOKEN in the environment (or a "
            "repo-root .env) and restart.")]
    return []


def _safe_atr(frame) -> float:
    """Best-effort finite ATR for observational evidence only."""
    try:
        from ..live.multitf import atr14
        value = float(atr14(frame))
        return value if math.isfinite(value) and value > 0 else 0.0
    except Exception:
        return 0.0


class MT5Runner:
    """Tick-driven live loop on MT5 feed + MT5 executor."""

    def __init__(self, config: MT5RunnerConfig | None = None, *,
                 feed: MT5Feed | None = None,
                 executor: MT5Executor | None = None,
                 senders: list | None = None,
                 tick_manager: TickManager | None = None,
                 news_calendar: NewsCalendar | None = None,
                 entry_gate: EntryGate | None = None) -> None:
        self.cfg = config or MT5RunnerConfig()
        self.feed = feed
        self.executor = executor
        self.senders = senders or []
        # Telegram failure visibility: a stuck sender must never be
        # silent (journal-only) and must never flood the console —
        # _note_tg_failure prints each distinct failure at most once
        # per 600s (same key window as alerts.telegram.ErrorDeduper).
        self._tg_fail_last: dict[str, float] = {}
        # Telegram-unavailable warning (no token, no CLI): logged +
        # printed once per process start, never every 30s poll.
        self._tg_unavailable_logged = False
        self.journal = TradeJournal(self.cfg.journal_path)
        # Step 8: latest signal studies for the interactive terminal
        # (candles are live from the stack; studies are labeled with
        # the signal they came from, never recomputed from a guess).
        self._latest_studies: dict | None = None
        self._terminal_server: TerminalServer | None = None
        # SL stand-down notes already journalled, keyed
        # (ticket, reason, level): a stood-down level notes once
        # per side-invalid window, never once per tick.
        self._sl_stood_down: set = set()
        # Setup lifecycle (owner rulings 2026-10-30 §1): scan-detected
        # roster setups persist across restarts so a later trigger
        # fire can route them to execution evaluation.  A store that
        # cannot be read starts empty with a loud note — the journal
        # still holds every detection event, so nothing vanishes
        # silently.
        self._setup_store: _SetupStore | None = None
        try:
            _lc_path = getattr(self.cfg, "setup_lifecycle_path",
                               None) or (
                self.cfg.journal_path.parent
                / "setup_lifecycle.json")
            self._setup_store = _SetupStore(_lc_path)
        except Exception as _lc_exc:
            self.journal.log_note(
                "setup_lifecycle: persistent setup store "
                f"unavailable ({_lc_exc}) — scan setups will not "
                "persist this run")
        # Hot Scalp is a separate, persistent research sleeve. Setup storage
        # lives beside the journal so two runner processes on this EC2 host
        # share identity and atomic trigger claims. Failure disables this
        # sleeve; it never falls back to in-memory ownership or order dispatch.
        self._hot_scalp: HotScalpResearchEngine | None = None
        self._hot_scalp_last_degraded_key: str | None = None
        self._hot_scalp_instance_id = uuid.uuid4().hex
        if self.cfg.hot_scalp_shadow_enabled:
            try:
                _hs_store = SetupStore(self.cfg.hot_scalp_db_path)
                self._hot_scalp = HotScalpResearchEngine(
                    _hs_store, HotScalpConfig(enabled=True),
                    instance_id=self._hot_scalp_instance_id)
            except Exception as _hs_exc:
                self.journal.log_note(
                    "hot_scalp: shadow sleeve disabled because shared setup "
                    f"storage is unavailable: {_hs_exc}")
        # Forecast service (owner rulings §3): forecasts + alerts only.
        # The engine trains itself: untrained at warm start, or stale
        # by 500+ new H1 bars at a D1 close, it trains in the
        # background from its own MT5 history — no command, no owner
        # step. Until a bundle exists, scoring reports "untrained"
        # and no forecast alert fires. Order paths never read this.
        _bundle = os.environ.get("GS_FORECAST_BUNDLE") or (
            DATA_DIR / "forecast_models.joblib")
        self._forecaster = MT5Forecaster(bundle_path=_bundle)
        self._forecast_last_scored_bar = None
        self._forecast_no_d1_noted = False
        # Deterministic engine runtime (bridge).  Best-effort: if it
        # fails to initialise, the legacy journal path continues and
        # the failure is noted — the live loop never dies on it.
        self.deterministic = None
        self._det_error: str | None = None
        if getattr(self.cfg, "use_deterministic_engine", True):
            try:
                det_dir = self.cfg.deterministic_dir or (
                    self.cfg.journal_path.parent / "deterministic")
                self.deterministic = DeterministicRuntime(det_dir)
            except Exception as exc:  # noqa: BLE001 - fallback is intentional
                self._det_error = str(exc)
        # RiskGuards ENFORCED on the live path (previously constructed
        # but never consulted here).  max_total_risk_pct=0 disables
        # the money-at-stops exposure cap — on the Exness MT5 loop
        # margin + equity stop-out are the real constraint, not R
        # (same documented reason the paper loop sets it to 0 in
        # risk.py).  The margin check in _place_live_order is the
        # binding pre-trade constraint; the guards supply the
        # max-drawdown halt (the sole account-level hard halt —
        # owner ruling 2026-10-23: daily/weekly caps removed),
        # the consecutive-loss halt, the max position cap and the
        # unstopped-position refusal.
        self.guards = RiskGuards(
            self.cfg.risk_state_path,
            RiskConfig(max_total_risk_pct=0.0),
            kill_switch_path=self.cfg.kill_switch_path)
        self.tick_manager = tick_manager or TickManager(
            tick=TickConfig(
                scalp_cap_minutes=float(self.cfg.scalp_cap_minutes),
                time_exit_bars=self.cfg.time_exit_bars,
                breakeven_mode=str(self.cfg.breakeven_mode),
                breakeven_backstop_r=float(
                    self.cfg.breakeven_backstop_r)))
        # News calendar: the engine's single news source (fetch on
        # the entry path only — never per tick, never blocking).
        self.news = news_calendar or NewsCalendar(
            cache_path=DATA_DIR / "news_cache.json")
        # Entry gates: caps / cooldowns / dedup for new base entries.
        self.entry_gate = entry_gate or EntryGate(EntryGateConfig(
            max_daily_trades=int(self.cfg.max_daily_trades),
            max_hourly_trades=int(self.cfg.max_hourly_trades),
            loss_cooldown_sec=float(self.cfg.loss_cooldown_sec),
            win_cooldown_sec=float(self.cfg.win_cooldown_sec),
            fingerprint_ttl_sec=float(self.cfg.fingerprint_ttl_sec)))
        self._playbooks: list | None = None
        self._engine: ConfluenceEngine | None = None
        self._stack: TimeframeStack | None = None
        # M1 is an auxiliary trigger stream for Hot Scalp, deliberately kept
        # outside the hierarchy stack so it cannot become a context vote.
        self._hot_scalp_aux_frames: dict[str, pd.DataFrame] = {}
        self._hot_scalp_last_processed_m1: pd.Timestamp | None = None
        self._hot_scalp_m1_refresh_error_key: str | None = None
        self._qm_touch: dict | None = None
        # Daily QM: last touch index already journal-logged
        # (the D1 scan logs each touch once, never re-logs history).
        self._qm_d1_last_logged: int = -1
        # FUNNEL §2: the native M5 route's state — its dedupe
        # cursor (the roster used to feed this playbook H1 bars;
        # the route now owns its evaluation, see _run_qm_m5), its
        # latest-touch annotation state, and the kill-switch
        # record latch (one not_eligible record per disable, not
        # one per M5 close).
        self._qm_m5_last_logged: int = -1
        self._qm_m5_last_touch: dict | None = None
        self._qm_m5_disabled_noted: bool = False
        # Phase 2 (P0-1): the durable seen-instance set for the
        # native routes — instance key -> terminal disposition.
        # The in-memory cursors above reset on restart, which is
        # exactly how the EC2 warm start re-routed nine days of
        # historical touches; this set is loaded from (and saved
        # to) a JSON sidecar beside the journal so a setup
        # instance can never be routed twice across restarts,
        # rescans or repeated evaluations.  Friction refusals are
        # NOT recorded (RULING 6: the setup stays alive).
        self._seen_instances: dict[str, str] | None = None
        self._seen_instances_path = (
            self.cfg.journal_path.parent / "candidate_instances.json")
        # Live evaluation clock (zone-touch arm, family confirm):
        # the forming H1 bar tracked from ticks, the cached trigger
        # levels from closed bars, and the pending (armed touch,
        # still unconfirmed) awaiting POI confirmation.
        self._forming: dict | None = None
        self._live_setup: dict | None = None
        # Why the live setup is / is not armed (live_setup_reason):
        # /status must be able to say WHY no setup is armed instead
        # of showing "blocked legs: none" for a runner that is
        # merely waiting (2026-10-16 screenshot).
        self._live_setup_reason: str = "not evaluated yet"
        # d1_flat HOLD state (D1BiasState) — the pending swings /
        # bars-remaining / flip requirements behind a flat bias.
        # A flat bias is a HOLD, never a feed error; this rides
        # alongside _live_setup so /status can show it.
        self._live_hold_state = None
        # Last HOLD line logged — HOLD is logged only when its
        # description changes (a new pending extreme, a bar
        # confirming, a flip requirement changing), not on every
        # TF-close refresh, or a flat week would spam the log.
        self._last_hold_logged: str | None = None
        self._pending: dict | None = None
        # Fired-window guard (trigger redesign): the H1 bar open
        # time in which a trigger last fired.  The H1 bar IS the
        # trigger's window (the pending expires at the H1 close),
        # and the live setup is fixed within it — so a second fire
        # in the same H1 bar is necessarily the same setup firing
        # twice on the same composing bars (one trigger per
        # window).  The arm path skips re-arm while
        # ``bar_open == self._fired_h1_bar``; the next H1 bar is a
        # fresh window.  Without this, every tick in the zone
        # re-arms and re-fires on the same already-closed
        # composing bars (trigger storm: dozens of fires per bar).
        self._fired_h1_bar = None
        # Stale-zone re-arm block (armed TTL, owner rule): after a
        # stale disarm, (direction, zone bottom, zone top) of the
        # dead zone.  Re-arm needs price to EXIT that zone and
        # re-enter — no immediate re-arm while still inside, or
        # the TTL is meaningless.  Cleared on exit, or when the
        # setup's zone changes (a different POI is a fresh setup).
        self._stale_zone: tuple | None = None
        # Invalidated-zone re-arm block (Close Rule, owner rule):
        # after a Close-Rule invalidation, (direction, zone bottom,
        # zone top) of the FAILED zone.  Unlike the stale block,
        # exit + re-entry does NOT revive a failed zone — the
        # block lifts only when the setup's zone changes (a
        # different POI is a different setup).  Monitoring for
        # the failed setup is terminated.
        self._invalidated_zone: tuple | None = None
        # Range boundary machine (sweep vs breakout, live/range.py)
        # + persisted hysteresis clock (live/regime_state.py).
        # The tracker is fed on every TF-close refresh; its state
        # is persisted atomically on every H1 close BEFORE any
        # order decision using the new counts.
        self._boundary: BoundaryTracker | None = None
        self._boundary_snapshot: dict = {}
        # breakout_id already fired on the tick clock (one fire per
        # confirmed breakout — a confirmed breakout fires once, it
        # never re-fires on later ticks).
        self._breakout_fired_id: int = 0
        # Continuation break_keys already fired (arming-gate fix
        # F2: one arm per break — a break that fired never re-arms
        # while price stays beyond the broken level; the stateless
        # live_setup would otherwise re-issue the same setup on
        # every refresh).  Spent at fire time in _run_live_signal;
        # enforced in _refresh_live_setup (suppression) and in the
        # tick clock (backstop).  In-memory like _breakout_fired_id.
        self._continuation_spent: set[str] = set()
        # Persisted regime state: journal-adjacent
        # (<journal dir>/regime_state.json), never /tmp.
        self._regime_state_path = \
            self.cfg.journal_path.parent / "regime_state.json"
        self._regime_state_loaded = False
        self._regime_state_status = "not loaded"
        self._regime_state_saved_at: str | None = None
        # Regime pause (checksum-mismatch decision tree — see
        # _handle_corrupt_regime_state): None | "observation" |
        # "paused_new" | "paused_complete".  "observation" = corrupt
        # state rebuilt, no new orders until 2 clean H1 bars.
        # "paused_new" = open risk exists: no new orders, management
        # of open positions continues.  "paused_complete" = position
        # state itself untrusted: no orders AND no management,
        # await owner /resume.  NEVER silently continued.
        self._regime_pause: str | None = None
        self._regime_pause_reason: str = ""
        self._regime_observation_remaining: int = 0
        self._regime_stable_bars: int = 0
        self._regime_stable_signature: str | None = None
        # Last PendingOrderAudit summary (see
        # _audit_pending_orders): {"cancelled": {reason: n},
        # "kept": n, "filled": n, "cancel_failed": n} + timestamp.
        # Shown on /status (boundary_line) until the next audit.
        self._last_pending_audit: dict | None = None
        self._last_pending_audit_at: str | None = None
        # Multi-day structural zones (live/zones.py): the durable
        # ZoneStore (SQLite, journal-adjacent) + the in-memory
        # ZoneCache (execution path — the tick loop bisects the
        # cache, never SQLite).  The store opens LAZILY on the
        # first _update_zones (many tests construct a runner
        # without ever refreshing zones).
        self._structural_zones: list = []
        self._zone_store: object | None = None
        self._zone_cache = ZoneCache()
        self._zone_store_failed = False
        self._zones_seeded = False
        self._last_zone_conflict_sig: tuple | None = None
        self._zone_cluster_keys: set = set()
        self._nearby_zones: list = []
        self._near_major_zone: object | None = None
        self._last_zone_atr: float = 0.0
        # Broker pending orders (LIMIT / STOP) this engine placed or
        # adopted: ticket -> PendingOrderState.  Distinct from
        # ``self._pending`` (the armed zone touch of the
        # tick-clock machinery) — a pending ORDER is a broker order;
        # a pending TRIGGER is pre-signal state.
        self._pending_orders: dict[int, PendingOrderState] = {}
        # Narrative chain state per live ticket (CHUNK 3): the
        # MANAGEMENT state built at fill by _narrative_fill_plan.
        # Only narrative trades get an entry; _record_close pops
        # it and runs finalize_exit (STATE 7).  Never consulted
        # for non-narrative tickets.
        self._narrative_states: dict[int, NarrativeState] = {}
        # XD-4: last seen closed-frame length per slow clock,
        # for gating the horizon-timeout evaluation to that
        # clock's closes (D1/H4 positions only).
        self._timeout_seen_bars: dict[str, int] = {}
        # Confluence rework (arrival pass): per-ticket RiskPool
        # ledgers (A13), the latest Asymmetry Score breakdown per
        # playbook (journal/status), funding-event keys already
        # consumed per ticket, the PullbackWatch state (§5), the
        # resting-limit dedupe key (A6) and the last sizing detail
        # per alert_id (authorized vs realized risk, §6).
        self._pools: dict[int, RiskPool] = {}
        self._last_scores: dict[str, dict] = {}
        self._last_score: dict | None = None
        self._funded_events: dict[int, set] = {}
        # Step 5: layer trigger identities already consumed by a
        # filled layer, per parent ticket (stacking policy
        # trigger_duplicate input).  Marked on fill, never on
        # placement attempt — a rejected order must not burn its
        # trigger.
        self._layer_trigger_ids: dict[int, set] = {}
        # P3: PullbackWatch is MULTI-SLOT — several parked setups,
        # each with its own zone/expiry/invalidation (was a single
        # dict pre-P3; _pullback_watch property below keeps the
        # legacy read for status).
        self._pullback_watches: list[dict] = []
        self._resting_limit_key: str | None = None
        self._sizing_detail: dict[str, dict] = {}
        # --- P3 scenario state (spec v3 §12) ---------------------
        # Refreshed on every TF-close setup refresh: bedrock (the
        # ONE regime brain), the day object, the composed posture,
        # the NY acceptance checkpoint, the SMT read.
        self._bedrock = None
        self._day_ctx = None
        self._posture = None
        self._ny_checkpoint: str | None = None
        self._ny_checkpoint_day: str = ""
        self._smt_read = None
        self._smt_unavailable_key: str = ""
        self._last_clock = None
        # Rolling spread samples (dead-window proof, S3/S4).
        self._spread_samples: list[float] = []
        self._spread_ratio: float | None = None
        # Per-signal decision context (alert_id -> ctx) captured
        # at acceptance, consumed at fill into the per-position
        # management context (ticket -> ctx).
        self._pending_signal_ctx: dict | None = None
        self._signal_ctx: dict[str, dict] = {}
        self._position_ctx: dict[int, dict] = {}
        # S11 per-style daily open counts (churn ceilings).
        self._style_opens_today: dict[str, int] = {}
        self._style_day: str = ""
        # Funding events carried by a pending LAYER order, consumed
        # when the layer fills (ticket -> [FundingEvent]).
        self._pending_funding: dict[int, list] = {}
        self._last_processed: dict[str, pd.Timestamp] = {}
        self._last_sig_bar: dict[str, int] = {}
        self._tick_count = 0
        self._last_tick: dict | None = None
        # UTC date of the last tick seen (rollover recap clock).
        self._last_tick_date = None
        # UTC date the Friday-close action last fired (once per day).
        self._friday_close_done_date = None
        self._stopped = False
        # Startup ping: exactly one Telegram "engine started" message
        # per process start (see _send_startup_ping).  Guarded so a
        # second run() call on the same runner never double-pings.
        self._startup_ping_sent = False
        # Per-day tick CSV logger (next to the journal; None = disabled).
        self._tick_logger = TickLogger(
            self.cfg.journal_path.parent,
            retain_days=self.cfg.tick_csv_retain_days) \
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
        # FUNNEL (§1): per-run identity + instrumentation state.
        # run_id stamps every evaluation record so one process
        # run's funnel can be queried end-to-end; the config
        # fingerprint pins WHICH effective config produced it
        # (provenance is a journal fact now, not a memory).
        self._run_id: str = uuid.uuid4().hex[:12]
        self._build_commit_cache: str | None = None
        self._config_fingerprint_cache: str | None = None
        self._funnel_seq: int = 0
        self._funnel_counts: dict[str, dict] = {}
        self._funnel_refusals: dict[str, int] = {}
        self._funnel_trigger: dict | None = None
        # Pending Confirm taps: token -> {"action", "arg"}.
        self._confirm_tokens: dict[str, dict] = {}
        # MacroTrend alignment (D1/W1/MN1): recomputed at warm start
        # and whenever a D1/W1/MN1 bar closes.  Sizing + Golden
        # Swing confirmation read this — never a hierarchy leg.
        self._alignment: MacroTrendResult = MacroTrendResult()
        # DXY correlation (owner's-bot floor): refreshed on H1
        # closes; confluence context only, never a gate.  A dead
        # DXY feed degrades to unavailable — the gold path never
        # depends on it.
        self._dxy = DXYTracker(
            self.feed, symbol=self.cfg.dxy_symbol,
            enabled=self.cfg.dxy_enabled)
        self._dxy_last_available: bool | None = None
        # Execution health (tick freshness / order latency /
        # slippage): observed, never gating.  Breach journaling is
        # new-breach-only (a stuck breach gets one note).
        self._health = HealthMonitor()
        self._health_last_breaches: set[str] = set()
        # Runtime Truth item 1: when the health monitor's market
        # state was last refreshed from the feed (monotonic clock;
        # the refresh is cached — symbol_info is a terminal call).
        self._health_market_ts: float = 0.0
        # Staged Golden Swing orders awaiting a Confirm tap:
        # order key -> {playbook, sig, verdict, entry_bar}.  The
        # token in _confirm_tokens (action "place_order") is the
        # single-use handshake; this dict is the payload.  Both are
        # popped together on confirm / reject / timeout.
        self._order_confirms: dict[str, dict] = {}
        # Persisted live config (data/mt5_config.json): when the file
        # exists it wins over the constructed defaults — /set changes
        # survive a restart because the next process loads this file.
        self._config_store = LiveConfigStore(
            self.cfg.journal_path.parent / "mt5_config.json")
        if self._config_store.path.exists():
            self._config_store.apply_to(self.cfg)
            # FUNNEL §3: a persisted kill-switch entry naming no
            # real playbook is dropped by _clean — correctly, but
            # NEVER silently (a dropped entry once hid four
            # unkillable range playbooks).  Journal what was lost.
            try:
                _raw_cfg = json.loads(
                    self._config_store.path.read_text())
                _dropped = unknown_disabled_names(_raw_cfg)
                if _dropped:
                    self.journal.log_note(
                        "mt5: persisted config dropped unknown "
                        "disabled_playbooks entries (not real "
                        f"playbooks): {', '.join(_dropped)}")
            except Exception:
                pass
        # Common evidence is attached to the signal logger so every existing
        # MT5 playbook gets the same explainable, non-gating research snapshot.
        # It cannot alter a verdict or authorize an order.
        self.journal.shared_evidence_provider = self._build_shared_signal_evidence

    # ------------------------------------------------------------ wiring
    def _get_playbooks(self) -> list:
        """Every playbook the H1-clocked signal path knows.

        A14 (owner addendum 2026-10-21): there is no shadow tier
        on this path any more.  Every playbook here is evaluated
        AND eligible to execute through the same decision flow
        (Asymmetry Score floors, RiskPool, hard lines) — see
        :meth:`_executes`.  The only opt-out is the per-playbook
        kill switch (``disabled_playbooks``).  Capital protection
        is the RiskPool ceiling + StyleBook caps + RiskGuards
        daily cap, not a log-only tier.

        Native-clock playbooks are deliberately NOT here: the H1
        path drives ``live_signal`` with H1 bars, and running M5
        or D1 QM logic on H1 bars would emit H1 patterns
        mislabelled as M5 / Daily QM.  ``qm_m5`` lives in
        :meth:`_run_qm_m5` (real M5 frame, every M5 close) and
        ``qm_d1`` in :meth:`_run_qm_d1` (real D1 frame, every D1
        close) — one playbook, one clock, one route.

        FUNNEL §3: the roster is DERIVED from the one registry
        (``playbook_registry.ROSTER_SPECS``) — the same registry
        the config whitelist, /playbooks and /set validation are
        derived from, so the lists cannot drift apart again."""
        if self._playbooks is None:
            self._playbooks = [spec.playbook_class()
                               for spec in ROSTER_SPECS]
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
        elif reason.startswith("risk_guard"):
            leg = "risk_guard"
        elif reason.startswith("margin_shield"):
            leg = "margin_shield"
        elif reason.startswith("margin"):
            leg = "margin"
        elif reason.startswith("market_closed"):
            leg = "market_closed"
        elif reason.startswith("risk_friction"):
            # Ruling-6 entry friction gate — its own leg, NOT the
            # hard-lines cost_gate leg (separate mechanisms).
            leg = "friction_gate"
        elif reason.startswith("cost_gate"):
            leg = "cost_gate"
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

    # ------------------------------------------------ FUNNEL (§1)
    # Decision-funnel instrumentation: one evaluation record per
    # playbook per evaluation clock, INCLUDING no-signal outcomes,
    # in the journal (type "evaluation").  OBSERVATION-ONLY: the
    # emit path never raises into a decision and never changes one
    # (cfg.funnel_enabled=False reproduces base behaviour exactly;
    # the identity is test-pinned).

    def _build_commit(self) -> str:
        """Best-effort build identity for the manifest/records:
        GS_COMMIT env, else the checkout's git HEAD, else
        "unknown".  Never raises (a runtime without git still
        journals its funnel; the commit is then honestly absent)."""
        if self._build_commit_cache is not None:
            return self._build_commit_cache
        commit = os.environ.get("GS_COMMIT") or ""
        if not commit:
            try:
                out = subprocess.run(
                    ["git", "-C", str(REPO_ROOT), "rev-parse",
                     "--short", "HEAD"], capture_output=True,
                    text=True, timeout=3)
                if out.returncode == 0:
                    commit = out.stdout.strip()
            except Exception:
                commit = ""
        self._build_commit_cache = commit or "unknown"
        return self._build_commit_cache

    def _config_fingerprint(self) -> str:
        """Hash of the EFFECTIVE decision config (post /set store):
        the provenance stamp on every funnel record, so "which
        config produced this decision" is a journal fact."""
        if self._config_fingerprint_cache is not None:
            return self._config_fingerprint_cache
        cfg = self.cfg
        payload = {
            "symbol": cfg.symbol,
            "execute_playbooks": cfg.execute_playbooks,
            "disabled_playbooks": sorted(
                cfg.disabled_playbooks or ()),
            "min_score": cfg.min_score,
            "sizing_mode": cfg.sizing_mode,
            "risk_pct": cfg.risk_pct,
            "paused": cfg.paused,
            "manipulation_filter": cfg.manipulation_filter,
            "news_pause_enabled": cfg.news_pause_enabled,
            "entry_gates_enabled": cfg.entry_gates_enabled,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        self._config_fingerprint_cache = hashlib.sha256(
            raw.encode()).hexdigest()[:16]
        return self._config_fingerprint_cache

    def _evaluation_now(self):
        """The evaluation clock for journal timestamps: the tick
        clock when the engine has a tick, else the wall clock.
        The same basis the QM routes use for freshness — a record
        without a timestamp rendered "evaluated not recorded",
        which made warmup scans and live closes indistinguishable
        in /assess (owner, 2026-10-26).  Never raises."""
        now_ref = None
        try:
            tick = getattr(self, "_last_tick", None)
            if tick and tick.get("time") is not None:
                now_ref = pd.Timestamp(tick["time"])
        except Exception:
            now_ref = None
        if now_ref is None:
            now_ref = pd.Timestamp.now(tz="UTC")
        if now_ref.tzinfo is None:
            now_ref = now_ref.tz_localize("UTC")
        return now_ref

    def _emit_eval(self, *, playbook: str, evaluation_clock: str,
                   intended_tf: str | None = None,
                   actual_input_tf: str | None = None,
                   last_bar_ts=None, data_freshness_s=None,
                   outcome: str = "no_setup", reason_code: str = "",
                   reason: str = "", candidate: dict | None = None,
                   decision: dict | None = None,
                   evidence: dict | None = None,
                   kind: str = "playbook_eval",
                   trigger_stage: str | None = None,
                   source: str | None = None,
                   assessment: dict | None = None) -> dict | None:
        """Append ONE funnel evaluation record + update the
        in-memory /status counters.  Never raises; returns the
        record (or None when the funnel is disabled).

        Phase 2: the decision block is normalized to TERMINAL
        semantics before it is journaled (an analytical acceptance
        the order path refused terminates as a refusal, with the
        analytical verdict preserved — see
        funnel.terminal_decision), and ``source`` records where
        the evaluation came from (live close / warmup scan /
        trigger path) — provenance is a journal fact, not an
        inference."""
        self._funnel_seq += 1
        if not getattr(self.cfg, "funnel_enabled", True):
            return None
        # Census-wiring pass: the previously unwired detector
        # families (chart patterns, premium/discount, breaker/
        # mitigation, relative strength, engulfing, named
        # candlestick taxonomy) are read causally off the frames
        # in scope and recorded on the record as evidence.
        # Observation-only: this never feeds a decision, and a
        # failure degrades to no snapshot (the assessment then
        # marks the wiring without a per-record read).
        census = self._census_for_record(
            intended_tf=intended_tf,
            actual_input_tf=actual_input_tf, source=source)
        if census is not None:
            evidence = {**(evidence or {}), "census": census}
        if decision is not None:
            decision = _terminal_decision(decision)
        if decision is not None and "reason_category" not in decision:
            from .funnel import category_for_code
            decision = dict(decision)
            decision["reason_category"] = category_for_code(
                decision.get("reason_code") or reason_code,
                decision.get("result"))
        # Step 3: capture the evaluation instant ONCE — the record's
        # ts and its evaluation-session tag must describe the same
        # instant (a second wall-clock read could straddle a
        # session boundary).
        _eval_now = self._evaluation_now()
        record = {
            "type": "evaluation", "kind": kind,
            # Journal auditability (owner, 2026-10-26): every
            # record carries its own evaluation timestamp — the
            # renderer reads this first, so no record can ever
            # again render "evaluated not recorded".
            "ts": str(_eval_now),
            "run_id": self._run_id, "commit": self._build_commit(),
            "config_fingerprint": self._config_fingerprint(),
            "playbook": playbook,
            "intended_tf": intended_tf,
            "actual_input_tf": actual_input_tf,
            "last_bar_ts": (str(last_bar_ts)
                            if last_bar_ts is not None else None),
            "data_freshness_s": data_freshness_s,
            "evaluation_clock": evaluation_clock,
            "outcome": outcome,
            "reason_code": reason_code or None,
            "reason": reason or "",
            "candidate": candidate,
            "decision": decision,
            "evidence": evidence,
            "trigger_stage": trigger_stage,
            "evaluation_source": source,
        }
        # Step 3: session provenance — the session of the candidate
        # bar AND of the evaluation instant, from the one DST-aware
        # authority (reader.session_timeframes).  Session detection
        # is not market-open detection: these fields say which
        # session a decision belongs to, never whether the broker
        # accepts orders.
        def _sess(ts):
            if ts is None:
                return None
            try:
                _s, _kz = session_of(pd.Timestamp(ts))
                return {"session": _s, "killzone": _kz}
            except Exception:
                return None
        record["session"] = {
            "candidate_bar": _sess(last_bar_ts),
            "evaluation": _sess(_eval_now),
        }
        # Amendment 3 (owner, 2026-10-23): every evaluation record
        # carries a renderable MARKET ASSESSMENT — structure read,
        # HTF context, itemized evidence (detector + source-bar
        # ts, correlated items grouped), supporting vs
        # contradictory lists, competing scenarios, the verdict
        # in the owner's vocabulary and a plain-language reason —
        # with absent/stale/not-integrated capabilities MARKED,
        # never silently omitted.  A caller-supplied ``assessment``
        # overlays richer route-specific fields on the derived
        # shape.  The presentation layer is a later milestone;
        # these records are what it will render from.
        try:
            record["assessment"] = _assessment_block(
                outcome=outcome, decision=decision,
                evidence=evidence, candidate=candidate,
                reason=reason, reason_code=reason_code or None,
                last_bar_ts=last_bar_ts,
                data_freshness_s=data_freshness_s,
                overlay=assessment)
            record["reason_text"] = \
                record["assessment"]["reason_text"]
        except Exception:
            record["assessment"] = None
            record["reason_text"] = reason or ""
        # Visible market-reading milestone (owner, 2026-10-23):
        # the RENDERED assessment rides on the record itself, so
        # the journal is human-readable evidence — the owner can
        # read what the engine saw without a separate tool.  The
        # renderer prints only what the record carries (absent
        # sections render as explicit not-produced lines).
        try:
            record["assessment_text"] = _render_assessment(
                record, symbol=getattr(self.cfg, "symbol", None))
        except Exception:
            record["assessment_text"] = None
        try:
            self.journal.append(record)
        except Exception:
            pass  # audit is best-effort; a journal failure must
            # never break the signal path (log_signal's contract).
        # In-memory counters for /status (restart resets them; the
        # journal is the durable record).
        if kind == "trigger_event":
            self._funnel_trigger = {
                "stage": trigger_stage, "reason_code": reason_code,
                "reason": reason}
        else:
            slot = self._funnel_counts.setdefault(playbook, {
                "evaluations": 0, "no_setup": 0, "candidates": 0,
                "refused": 0, "ordered": 0, "ineligible": 0,
                "no_order": 0, "unresolved": 0,
                "ineligible_codes": {}, "qualified": 0})
            slot["evaluations"] += 1
            if outcome == "no_setup":
                slot["no_setup"] += 1
            elif outcome == "candidate_emitted":
                slot["candidates"] += 1
                dec = decision or {}
                # ONE terminal bucket per candidate (funnel.
                # decision_bucket): candidates reconcile as
                # ordered + refused + ineligible + no_order +
                # unresolved — the Phase 1 counter read only
                # decision.result, so the 23 order-path refusals
                # of run cb9b3fb522b7 rendered unaccounted.
                bucket = _decision_bucket(dec)
                if bucket == "refused":
                    slot["refused"] += 1
                    code = dec.get("reason_code") or reason_code
                    if code:
                        self._funnel_refusals[code] = \
                            self._funnel_refusals.get(code, 0) + 1
                elif bucket == "ordered":
                    slot["ordered"] += 1
                elif bucket == "ineligible":
                    slot["ineligible"] += 1
                    # Runtime Truth item 2: an ineligible count
                    # with its reasons NOWHERE was the EC2
                    # discrepancy (31/31 + 10/10 "ineligible",
                    # blocked legs empty).  The eligibility code
                    # rides the decision (execution layer) or the
                    # record; count it per playbook so /status can
                    # name WHY every candidate was ineligible.
                    code = dec.get("reason_code") or reason_code \
                        or "no_code_recorded"
                    codes = slot["ineligible_codes"]
                    codes[code] = codes.get(code, 0) + 1
                elif bucket == "no_order":
                    slot["no_order"] += 1
                else:
                    slot["unresolved"] += 1
                # Analytical layer (two-layer verdict law): the
                # strategy layer's own verdict, counted separately
                # from the terminal bucket.
                _av = dec.get("analytical_verdict") or {}
                if isinstance(_av, dict) \
                        and _av.get("verdict") == "qualified":
                    slot["qualified"] += 1
        return record

    def _census_for_record(self, *, intended_tf, actual_input_tf,
                           source) -> dict | None:
        """The census-evidence snapshot for ONE evaluation
        record (live/census_evidence.py), cached per frame
        state so the ~20 records of one bar share one read.

        Live evaluations (live_close / trigger path / native
        routes) record the snapshot off the stack frames as
        held NOW; warmup/replay evaluations explicitly record
        NOTHING (their evaluated bar is historical — a
        last-bar read stamped on them would misdescribe the
        evidence), and records with no frame in scope say so.
        Never raises; never a decision input."""
        try:
            from ..live.census_evidence import (
                census_snapshot, census_unavailable)
            if source == "warmup_scan":
                return census_unavailable(
                    "warmup/replay evaluation — census "
                    "evidence is recorded on live evaluations "
                    "only")
            stack = getattr(self, "_stack", None)
            frames = getattr(stack, "frames", None) or {}
            tf = actual_input_tf or intended_tf or "H1"
            frame = frames.get(tf)
            if frame is None:
                frame = frames.get("H1")
                tf = "H1"
            if frame is None or not len(frame):
                return census_unavailable(
                    "no live frame in scope at emit time")
            h1_frame = frames.get("H1")
            dxy = getattr(self, "_dxy", None)
            dxy_frame = getattr(dxy, "frame", None) \
                if dxy is not None else None

            def _key(fr):
                if fr is None or not len(fr):
                    return None
                return (len(fr), str(fr.index[-1]))

            key = (tf, _key(frame), _key(h1_frame),
                   _key(dxy_frame))
            cache = getattr(self, "_census_cache", None)
            if cache is None:
                cache = self._census_cache = {}
            if key in cache:
                return cache[key]
            snap = census_snapshot(
                frame=frame, tf=tf, h1_frame=h1_frame,
                dxy_frame=dxy_frame)
            if len(cache) > 64:
                cache.clear()
            cache[key] = snap
            return snap
        except Exception:
            return None

    # --------------------------------- candidate instances (P0-1)
    def _load_seen_instances(self) -> dict[str, str]:
        """The durable seen-instance set (lazy, fail-open to an
        empty set only when the sidecar does not exist yet; a
        corrupt sidecar is journaled and treated as empty — the
        order-path eligibility backstop still refuses any
        candidate that is not affirmatively eligible)."""
        if self._seen_instances is not None:
            return self._seen_instances
        seen: dict[str, str] = {}
        try:
            if self._seen_instances_path.exists():
                raw = json.loads(
                    self._seen_instances_path.read_text(
                        encoding="utf-8"))
                if isinstance(raw, dict):
                    seen = {str(k): str(v)
                            for k, v in raw.items()}
        except Exception as exc:
            try:
                self.journal.log_note(
                    f"mt5: candidate instance store unreadable "
                    f"({exc}) — starting empty; eligibility "
                    "backstops still apply")
            except Exception:
                pass
        self._seen_instances = seen
        return seen

    def _mark_instance_seen(self, key: str, disposition: str) -> None:
        """Record one setup instance's terminal disposition and
        persist it (bounded: the oldest entries fall off past
        10,000 — a setup older than 10,000 routed instances is
        far outside any validity window anyway)."""
        seen = self._load_seen_instances()
        if key in seen:
            return
        seen[key] = disposition
        while len(seen) > 10000:
            seen.pop(next(iter(seen)))
        try:
            self._seen_instances_path.parent.mkdir(
                parents=True, exist_ok=True)
            tmp = self._seen_instances_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(seen), encoding="utf-8")
            tmp.replace(self._seen_instances_path)
        except Exception:
            pass  # best-effort durability; the in-memory set and
            # the cursors still dedupe within this process.

    def _announce_assessment(self, record: dict | None) -> None:
        """Send one LIVE terminal evaluation's rendered market
        assessment to the owner (visible market-reading
        milestone): when a live evaluation produces a signal or
        watch verdict, the outbound signal surface carries the
        assessment itself — not a bare score.  Guards keep this
        honest and quiet:

        - only signal / watch verdicts (rejected / expired /
          no-setup stay journal-only);
        - only LIVE sources (live_close / trigger_path) —
          warmup, catch-up and replay records never announce
          (the EC2 burst class);
        - shadow records never announce (no order authority —
          they are grading records, not signals), and the
          scheduled scans are never routed here at all (their
          records are journal-only by call site);
        - sent through the existing _send_alert path — no new
          sender, no new poller.  Never raises into the tick
          loop (same contract as _send_alert's callers)."""
        try:
            if not record:
                return
            if record.get("evaluation_source") not in (
                    _lifecycle.SOURCE_LIVE_CLOSE,
                    _lifecycle.SOURCE_TRIGGER_PATH):
                return
            if _is_shadow_record(record):
                return
            assessment = record.get("assessment") or {}
            if assessment.get("verdict") not in ("signal", "watch"):
                return
            text = record.get("assessment_text")
            if not text:
                text = _render_assessment(
                    record,
                    symbol=getattr(self.cfg, "symbol", None))
            if text:
                self._send_alert(text)
        except Exception:
            pass  # an announcement failure must never break
            # the signal path (log_signal's contract).

    def _emit_trigger_event(self, stage: str, reason_code: str,
                            reason: str) -> None:
        """One trigger-lifecycle funnel record (armed / fired /
        expired / invalidated) — the disposition trail whose
        absence from /status hid arm #2's fate (watch status #6)."""
        self._emit_eval(
            playbook="__trigger__", kind="trigger_event",
            evaluation_clock="h1_trigger", intended_tf="H1",
            actual_input_tf="H1", outcome="candidate_emitted"
            if stage == "fired" else "not_eligible",
            reason_code=reason_code, reason=reason,
            trigger_stage=stage)

    # ------------------------------------------- setup lifecycle (§1)
    def _journal_setup_event(self, setup, stage: str,
                             detail: str = "") -> None:
        """One setup-lifecycle funnel record (kind=setup_event).

        The chain candidate_detected -> setup_armed ->
        trigger_fired -> execution_evaluated -> order_submitted
        -> filled / rejected / expired / invalidated, every link
        carrying the stable setup ID and parent setup ID (owner
        rulings 2026-10-30 §1: a fired trigger — and a detected
        setup — must never disappear between analysis and
        execution)."""
        self._emit_eval(
            playbook=setup.playbook, kind="setup_event",
            evaluation_clock="setup_lifecycle",
            intended_tf=setup.trigger_tf,
            actual_input_tf=setup.trigger_tf,
            outcome=("candidate_emitted"
                     if stage in (_setup_lc.STAGE_DETECTED,
                                  _setup_lc.STAGE_ARMED,
                                  _setup_lc.STAGE_TRIGGER_FIRED,
                                  _setup_lc.STAGE_SUBMITTED,
                                  _setup_lc.STAGE_FILLED)
                     else "not_eligible"),
            reason_code="", reason=detail, trigger_stage=stage,
            candidate={"direction": int(setup.direction),
                       "entry": float(setup.entry_ref),
                       "stop": float(setup.invalidation),
                       "targets": [float(x) for x in setup.targets]},
            evidence={"setup_id": setup.setup_id,
                      "parent_setup_id": setup.parent_setup_id,
                      "lifecycle_stage": stage,
                      "route": setup.route,
                      "trigger_kind": setup.trigger_kind,
                      "validity_bars": int(setup.validity_bars),
                      "expiry_ts": setup.expiry_ts,
                      "formation_ts": setup.formation_ts,
                      "touch_ts": setup.touch_ts},
            source="setup_lifecycle")

    def _transition_setup(self, setup, stage: str,
                          detail: str = "", **kw) -> None:
        """Move a tracked setup and journal the transition."""
        if self._setup_store is None:
            return
        ev = self._setup_store.transition(
            setup.setup_id, stage, ts=str(self._evaluation_now()),
            detail=detail, **kw)
        if ev is not None:
            self._journal_setup_event(setup, stage, detail)

    def _rebuild_setup_signal(self, setup, t: int):
        """The detector's candidate, rebuilt from its persisted
        fields — the persisted setup is evaluated at fire time,
        not re-detected fresh (a fresh-only filter is exactly
        what starved the Oct 30 scan setups)."""
        from ..golden.detector import GoldenSignal
        sig = GoldenSignal(
            index=int(t), timestamp=setup.touch_ts,
            direction=int(setup.direction),
            score=float(setup.score), kind=str(setup.kind or ""),
            factors=list(setup.factors),
            entry_zone=(tuple(setup.entry_zone)
                        if setup.entry_zone else (0.0, 0.0)),
            entry_ref=float(setup.entry_ref),
            invalidation=float(setup.invalidation),
            targets=[float(x) for x in setup.targets],
            atr=float(setup.atr),
            sweep_index=setup.sweep_index)
        if setup.live_context:
            sig.live_context = dict(setup.live_context)
        return sig

    def _setup_provenance(self, setup, h1, t: int) -> dict:
        """Provenance for a lifecycle-routed candidate: the
        candidate's origin is the live-close scan that detected
        it (eligibility is computed from THAT origin and the
        touch's age at fire time); this record's evaluation
        source is the trigger path that routed it."""
        touch_age = None
        if setup.touch_ts is not None and 0 <= t < len(h1):
            try:
                touch_age = int(
                    (pd.Timestamp(h1.index[t])
                     - pd.Timestamp(setup.touch_ts))
                    / pd.Timedelta(hours=1))
            except Exception:
                touch_age = None
        eligible, code, _why = _lifecycle.assess_eligibility(
            source=_lifecycle.SOURCE_LIVE_CLOSE,
            touch_age_bars=touch_age,
            validity_window_bars=int(setup.validity_bars))
        return _lifecycle.provenance_block(
            key=setup.setup_id, formation_ts=setup.formation_ts,
            touch_ts=setup.touch_ts, created_ts=setup.created_ts,
            evaluated_bar_ts=(h1.index[t] if 0 <= t < len(h1)
                              else None),
            evaluation_ts=self._evaluation_now(),
            source=_lifecycle.SOURCE_TRIGGER_PATH,
            touch_age_bars=touch_age,
            validity_window_bars=int(setup.validity_bars),
            eligible=eligible, eligibility_code=code)

    _SETUP_TF_DELTAS = {
        "M5": pd.Timedelta(minutes=5),
        "M15": pd.Timedelta(minutes=15),
        "M30": pd.Timedelta(minutes=30),
        "H1": pd.Timedelta(hours=1),
        "H4": pd.Timedelta(hours=4),
        "D1": pd.Timedelta(days=1)}

    def _register_scan_setup(self, spec, sig, prov, *, frame, t,
                             formation_ts, eval_now,
                             intended_tf) -> None:
        """Persist one scan-detected candidate as a tracked
        setup (owner rulings 2026-10-30 §1).  Detection happens
        at the level on a live close, so the setup arms
        immediately; its declared trigger is a same-direction
        h1_trigger fire inside its validity window."""
        clean_ctx, _dropped = _setup_lc.sanitize_context(
            getattr(sig, "live_context", None))
        touch_ts = frame.index[t]
        delta = self._SETUP_TF_DELTAS.get(
            str(intended_tf), pd.Timedelta(hours=1))
        setup = _TrackedSetup(
            setup_id=prov["instance_key"],
            parent_setup_id=prov["instance_key"],
            playbook=spec.name, route=spec.route,
            direction=int(sig.direction),
            kind=str(getattr(sig, "kind", "") or ""),
            score=float(getattr(sig, "score", 0.0) or 0.0),
            factors=list(getattr(sig, "factors", None) or []),
            entry_zone=[float(x) for x in (
                getattr(sig, "entry_zone", (0.0, 0.0))
                or (0.0, 0.0))],
            entry_ref=float(sig.entry_ref),
            invalidation=float(sig.invalidation),
            targets=[float(x) for x in (sig.targets or [])],
            atr=float(getattr(sig, "atr", 0.0) or 0.0),
            sweep_index=getattr(sig, "sweep_index", None),
            live_context=clean_ctx,
            formation_ts=(str(formation_ts)
                          if formation_ts is not None else None),
            touch_ts=str(touch_ts), created_ts=str(eval_now),
            trigger_kind=str(getattr(
                spec, "trigger_kind", "h1_trigger_fire")),
            trigger_tf=str(intended_tf),
            validity_bars=int(getattr(spec, "validity_bars", 6)),
            expiry_ts=str(touch_ts + delta
                          * int(getattr(spec, "validity_bars",
                                        6))))
        if self._setup_store.register(setup, ts=str(eval_now)):
            self._journal_setup_event(
                setup, _setup_lc.STAGE_DETECTED,
                f"scan-detected {setup.playbook} "
                f"{setup.direction:+d} @ {setup.entry_ref:.2f} — "
                "registered for trigger routing (the scan itself "
                "holds no order authority)")
            self._transition_setup(
                setup, _setup_lc.STAGE_ARMED,
                f"armed: trigger {setup.trigger_kind} "
                f"(same-direction h1_trigger fire) within "
                f"{setup.validity_bars} {setup.trigger_tf} bars; "
                f"invalidation {setup.invalidation:.2f}; "
                f"expiry {setup.expiry_ts}")

    def _upkeep_setups(self, h1) -> None:
        """Per-H1-close lifecycle upkeep: a tracked setup whose
        invalidation level has been CLOSED beyond, or whose
        validity window has lapsed, leaves the active set
        through a journaled terminal transition — never a
        silent drop.  Runs on every roster close-eval cycle,
        including the warm-start cycle (setups persisted across
        a restart meet their expiry here, journaled)."""
        if self._setup_store is None or h1 is None or not len(h1):
            return
        last_bar = h1.index[-1]
        close = float(h1.iloc[-1]["close"])
        for setup in self._setup_store.active(armed_only=True):
            if setup.expiry_ts is not None:
                try:
                    if pd.Timestamp(last_bar) >= pd.Timestamp(
                            setup.expiry_ts):
                        self._transition_setup(
                            setup, _setup_lc.STAGE_EXPIRED,
                            f"validity window lapsed: touch "
                            f"{setup.touch_ts}, expiry "
                            f"{setup.expiry_ts}, last closed bar "
                            f"{last_bar}")
                        continue
                except Exception:
                    pass
            inv = float(setup.invalidation)
            if (int(setup.direction) > 0 and close < inv) or \
                    (int(setup.direction) < 0 and close > inv):
                self._transition_setup(
                    setup, _setup_lc.STAGE_INVALIDATED,
                    f"close rule: {setup.trigger_tf} close "
                    f"{close:.2f} beyond invalidation {inv:.2f}")

    @staticmethod
    def _candidate_dict(sig, breakdown: dict | None = None,
                        provenance: dict | None = None) -> dict:
        """The candidate block of a funnel record, from a signal.

        Phase 2 (P0-2): TWO scores, separately named.  The
        playbook's own ``sig.score`` is the SLEEVE BASE score —
        for the QM sleeves a documented placeholder (60.0) that
        ranks sleeves against each other; it is NOT a measured
        quality score and never a floor input.  The score the
        DECISION uses is the Asymmetry total, which rides in
        ``score_components["total"]`` when a breakdown exists.
        Phase 1 records carried the sleeve base as the headline
        ``score`` (60.0 flat on all 32 burst candidates while the
        component totals ranged 31.5–48.1) — a constant wearing
        a computed value's name.  Now: ``score`` is the Asymmetry
        total when a breakdown exists (the actual computed value
        it claims to be — no constant, no clamp-to-floor), else
        the sleeve base; both are always present under their own
        names.  Floor authority per route:
        playbook_registry.FLOOR_AUTHORITY.

        ``provenance`` (candidate_lifecycle) is attached verbatim
        when the emitting route computed it.
        """
        sleeve_base = float(getattr(sig, "score", 0.0) or 0.0)
        asymmetry = None
        if breakdown is not None and breakdown.get("total") is not None:
            try:
                asymmetry = float(breakdown["total"])
            except (TypeError, ValueError):
                asymmetry = None
        out = {
            "direction": int(sig.direction),
            "entry": float(sig.entry_ref),
            "stop": float(sig.invalidation),
            "targets": [float(x) for x in (sig.targets or [])],
            "score": (asymmetry if asymmetry is not None
                      else sleeve_base),
            "asymmetry_score": asymmetry,
            "sleeve_base_score": sleeve_base,
            "score_components": breakdown,
            "factors": list(getattr(sig, "factors", []) or []),
            "index": int(getattr(sig, "index", -1)),
        }
        if provenance is not None:
            out["provenance"] = provenance
        # Step 15: the candidate's declared execution intent rides
        # every funnel record whose route stamped it on the signal.
        _intent = getattr(sig, "execution_intent", None)
        if _intent is not None:
            out["execution_intent"] = dict(_intent)
        return out

    def _journal_manifest(self) -> None:
        """FUNNEL §4.6 / directive §2: the runtime manifest,
        journaled once per run — commit, tree state, config
        fingerprint, the registry roster with each playbook's
        native clock + route, and the doctrine flags.  "Which
        build, which config" is never carried in anyone's head
        again (the R4 provenance failure)."""
        dirty = None
        try:
            out = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "status",
                 "--porcelain"], capture_output=True, text=True,
                timeout=3)
            if out.returncode == 0:
                dirty = bool(out.stdout.strip())
        except Exception:
            dirty = None
        try:
            self.journal.append({
                "type": "manifest", "run_id": self._run_id,
                "commit": self._build_commit(), "dirty_tree": dirty,
                "config_fingerprint": self._config_fingerprint(),
                "symbol": self.cfg.symbol,
                "feed": type(getattr(self, "feed", None)).__name__
                if getattr(self, "feed", None) is not None else None,
                "roster": [{"name": s.name, "clock": s.native_tf,
                            "route": s.route,
                            "refinement_clock": s.refinement_tf,
                            "shadow_evaluation":
                                s.shadow_evaluation}
                           for s in PLAYBOOK_REGISTRY],
                "doctrine": {
                    "execute_playbooks": self.cfg.execute_playbooks,
                    "disabled_playbooks": sorted(
                        self.cfg.disabled_playbooks or ()),
                    "sizing_mode": self.cfg.sizing_mode,
                    # Owner ruling 2026-10-23: dynamic risk
                    # sizing is the sole path (the A13 fixed_lots
                    # override is revoked and the field is gone).
                    # The manifest prints the EFFECTIVE mode + the
                    # configured risk rule, so what sized the
                    # trades is a journal fact, never an implication.
                    "sizing_mode_effective": self._sizing_mode(),
                    "risk_pct": self.cfg.risk_pct,
                    "min_score": self.cfg.min_score,
                    "floor_authority": dict(_FLOOR_AUTHORITY),
                }})
        except Exception:
            pass

    def funnel_totals(self) -> dict:
        """Run-level funnel totals (Runtime Truth item 2): every
        candidate counted exactly once in a terminal bucket, the
        ineligible reason codes aggregated, and the analytical
        qualified count kept separate (two-layer verdict law).
        The /status blocked-legs section cross-reads this so the
        two sections can never tell different stories again."""
        totals = {"candidates": 0, "ordered": 0, "refused": 0,
                  "ineligible": 0, "no_order": 0, "unresolved": 0,
                  "qualified": 0, "ineligible_codes": {}}
        for c in self._funnel_counts.values():
            for key in ("candidates", "ordered", "refused",
                        "ineligible", "no_order", "unresolved",
                        "qualified"):
                totals[key] += int(c.get(key, 0) or 0)
            for code, n in (c.get("ineligible_codes")
                            or {}).items():
                totals["ineligible_codes"][code] = \
                    totals["ineligible_codes"].get(code, 0) + n
        return totals

    def funnel_line(self) -> str:
        """The /status funnel section: provenance + the armed
        plan's last disposition + per-playbook evaluation counts
        with top refusal codes.  Guarded like the other *_line
        surfaces (old runners lack it)."""
        lines = [
            f"run `{self._run_id}` · build `{self._build_commit()}`"
            f" · config `{self._config_fingerprint()}`"]
        trig = self._funnel_trigger
        if trig:
            lines.append(
                f"trigger: last disposition {trig['stage']}"
                + (f" ({trig['reason_code']})"
                   if trig.get("reason_code") else "")
                + (f" — {trig['reason']}" if trig.get("reason")
                   else ""))
        else:
            lines.append("trigger: no lifecycle event yet this run")
        if not self._funnel_counts:
            lines.append("evaluations: none recorded yet this run "
                         "(playbooks evaluate on their clocks — "
                         "a zero here is a fact, see the journal)")
        for name in sorted(self._funnel_counts):
            c = self._funnel_counts[name]
            # Phase 2: ONE reconciling view — candidates must
            # equal ordered + refused + ineligible + no-order (+
            # unresolved, printed when nonzero: it is a defect
            # signal, never silently absorbed).
            line = (
                f"  `{name}`: eval {c['evaluations']} · "
                f"no-setup {c['no_setup']} · "
                f"candidates {c['candidates']} · "
                f"ordered {c['ordered']} · "
                f"refused {c['refused']} · "
                f"ineligible {c.get('ineligible', 0)}")
            if c.get("no_order"):
                line += f" · no-order {c['no_order']}"
            if c.get("unresolved"):
                line += f" · UNRESOLVED {c['unresolved']}"
            lines.append(line)
            # Runtime Truth item 2: the ineligible count carries
            # its reason codes inline — a bare "ineligible N"
            # beside an empty blocked-legs list was the EC2
            # discrepancy this pass closes.
            codes = c.get("ineligible_codes") or {}
            if codes:
                top = sorted(codes.items(),
                             key=lambda kv: -kv[1])[:3]
                lines.append(
                    "    ineligible reasons: " + " · ".join(
                        f"{code} ×{n}" for code, n in top))
        _not_evaluated = [s.name for s in PLAYBOOK_REGISTRY
                          if s.name not in self._funnel_counts]
        if _not_evaluated:
            lines.append(
                "not evaluated this run: "
                + ", ".join(f"`{n}`" for n in _not_evaluated))
        if self._funnel_refusals:
            top = sorted(self._funnel_refusals.items(),
                         key=lambda kv: -kv[1])[:5]
            lines.append("top refusals: " + " · ".join(
                f"{code} ×{n}" for code, n in top))
        # Runtime Truth item 2: the run-level reconciliation —
        # candidates must equal the sum of the terminal buckets,
        # printed in the directive's classes (submitted =
        # ordered, risk-refused = refused) with the analytical
        # qualified count beside them, never blended in.
        totals = self.funnel_totals()
        if totals["candidates"]:
            lines.append(
                f"reconcile: candidates {totals['candidates']} = "
                f"submitted {totals['ordered']} + risk-refused "
                f"{totals['refused']} + ineligible "
                f"{totals['ineligible']} + no-order "
                f"{totals['no_order']} + unresolved "
                f"{totals['unresolved']} · analytically qualified "
                f"{totals['qualified']}")
        return "\n".join(lines)

    # -------------------------------------------- deterministic engine
    def _det_journal(self, kind: str, data: dict,
                     *, idempotency_key: str | None = None) -> dict | None:
        """Dual-write one decision-critical frame to the deterministic
        journal.  Never raises, never breaks the live loop."""
        if self.deterministic is None:
            return None
        try:
            return self.deterministic.journal_event(
                kind, data, idempotency_key=idempotency_key)
        except Exception as exc:  # noqa: BLE001 - dual-write is best-effort
            try:
                self.journal.log_note(f"mt5: deterministic journal failed: {exc}")
            except Exception:
                pass
            return None

    def deterministic_status(self) -> dict:
        if self.deterministic is None:
            return {"enabled": False, "error": self._det_error}
        return {"enabled": True, **self.deterministic.status()}

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
        # Pin the terminal for both halves: the feed's cfg wins,
        # else the runner cfg (CLI --terminal-path / env), else the
        # feed's own MT5_TERMINAL_PATH fallback inside connect().
        if self.cfg.terminal_path \
                and getattr(feed.cfg, "terminal_path", None) is None:
            feed.cfg.terminal_path = self.cfg.terminal_path
        self._send_startup_progress(
            "MT5 engine starting…\n"
            f"symbol: {self.cfg.symbol}")
        _tg_warnings = _telegram_startup_warnings(self.senders)
        if _tg_warnings:
            # LOUD, not journal-only: with no sender configured (or
            # an outbox-only sender) every TG ping + command reply
            # is silently dropped.  The warning must be ACCURATE
            # (where it looked, what is missing, why yesterday
            # worked) — a flat "not configured" contradicts the
            # owner's evidence that notifications + commands worked
            # yesterday.  The engine still starts — market work
            # does not depend on Telegram — but the owner must know
            # at a glance why the bot is quiet.
            for line in _tg_warnings:
                print(line, flush=True)
            if not self.senders:
                self.journal.log_note(
                    "mt5: startup with no Telegram sender configured "
                    "(GS_TELEGRAM_BOT_TOKEN / GS_TELEGRAM_CHAT_ID "
                    "unset)")
        try:
            ok, msg = feed.connect()
            if not ok:
                # Feed failure on the FEED_ERROR channel (ERROR), before
                # raising — a connect failure is a feed problem, and it
                # must never be mistaken for a flat-bias HOLD.
                log_feed_error(_log, symbol=self.cfg.symbol, reason=msg)
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
            ok, msg = executor.connect(path=self.cfg.terminal_path)
            if not ok:
                raise RuntimeError(f"MT5 runner: executor connect failed: {msg}")

            # Warm start: full closed history per TF -> forward-only stack.
            # Feed health is reported on the FEED_ERROR channel (ERROR)
            # when poll_status says not-ok — a stale / disconnected
            # feed is a feed problem, distinct from a flat-bias HOLD.
            self.report_feed_status(feed.poll_status())
            # M1 is an auxiliary trigger stream for Hot Scalp, NOT a new
            # higher-timeframe hierarchy leg. It is fetched only while the
            # research sleeve is enabled.
            if hasattr(feed, "cfg") and hasattr(feed.cfg, "auxiliary_tfs"):
                _aux = tuple(getattr(feed.cfg, "auxiliary_tfs", ()) or ())
                if self._hot_scalp is not None:
                    feed.cfg.auxiliary_tfs = tuple(dict.fromkeys(_aux + ("M1",)))
                else:
                    feed.cfg.auxiliary_tfs = tuple(x for x in _aux if x != "M1")
            _warm_frames = feed.warm_start_all()
            if self._hot_scalp is not None:
                _m1 = _warm_frames.pop("M1", None)
                if _m1 is not None and len(_m1):
                    self._hot_scalp_aux_frames["M1"] = _m1.tail(3000)
                    self._hot_scalp_last_processed_m1 = (
                        _m1.index[-1] + pd.Timedelta(minutes=1))
            else:
                _warm_frames.pop("M1", None)
            self._stack = TimeframeStack.from_frames(_warm_frames)
            h1 = self._stack.frames.get("H1")
            if h1 is None or len(h1) == 0:
                raise RuntimeError("MT5 runner: warm start returned no H1 bars")
            # Forecast self-training, stated once at startup (no eager
            # bundle load for scoring — a multi-hundred-MB file is read
            # only when the first D1 close scores, and cached after
            # that; the training check itself reads the bundle only
            # when one exists, and never blocks the tick loop — the
            # train runs on a daemon thread).
            _thread, _note = self._forecaster.maybe_train(
                self._stack.frames)
            self.journal.log_note(f"forecast: {_note}")
            for tf, frame in self._stack.frames.items():
                if len(frame):
                    # Last-processed CLOSE (= last closed bar's open + TF).
                    # Comparing closes to closes: a close-to-open mixup
                    # would refresh on every tick.
                    self._last_processed[tf] = (
                        frame.index[-1] + pd.Timedelta(minutes=TF_MINUTES[tf]))
            self._send_startup_progress(
                "MT5 engine: history fetched ("
                + ", ".join(f"{tf}:{len(self._stack.frames.get(tf, []))}"
                            for tf in self._stack.frames)
                + ")")
            # (The warm QM annotation scan that used to sit here
            # is gone: the native M5 route below — run once at
            # warm start with the run stats — feeds the same
            # _qm_touch annotation state from true evaluation.
            # Funnel §2: no parallel scanner.)
            # Trigger levels from the closed warm-start bars.  The H1
            # close is NOT the signal clock anymore — this setup is what
            # the tick clock checks every tick (see _evaluate_live_tick).
            print("MT5 startup: building zones…", flush=True)
            self._refresh_live_setup()
            # MacroTrend alignment from the warm-start frames (D1/W1/MN1
            # closed bars).  Sizing + Golden Swing confirmation read this.
            self._refresh_alignment()
            self._send_startup_progress(
                "MT5 engine: zones built "
                f"({len(self._structural_zones)} zones)")
            self.feed = feed
            self.executor = executor
            # Deterministic lifecycle: REPLAY -> RECONCILING -> LIVE.
            # Replay re-reads committed frames with effects suppressed;
            # reconcile with no unresolved outbox flips to LIVE.  Failure
            # here never blocks the legacy loop — it is noted, not fatal.
            if self.deterministic is not None:
                try:
                    self.deterministic.begin_live()
                except Exception as exc:  # noqa: BLE001 - fallback is legacy
                    self.journal.log_note(
                        f"mt5: deterministic begin_live failed: {exc}")
            self._det_journal("runner_started", {
                "symbol": self.cfg.symbol,
                "login": acct.get("login"),
                "trade_mode": acct.get("trade_mode")})
            self.journal.log_note(
                "mt5 runner: started "
                f"(symbol={self.cfg.symbol}) "
                f"({', '.join(f'{tf}:{len(self._stack.frames.get(tf, []))}' for tf in self._stack.frames)}) "
                f"account login={acct.get('login')} equity={acct.get('equity')} "
                f"trade_mode={acct.get('trade_mode')} "
                f"macro: {self._alignment.describe()}"
                + (f" connect: {feed_msg}" if feed_msg != "MT5 connected" else ""))
            # FUNNEL §4.6: the runtime manifest (commit, config
            # fingerprint, registry roster + clocks) is journaled
            # once per run, before the first decision can land.
            self._journal_manifest()

            # Fast-forward playbook streaming state over history; past
            # signals are discarded — only the latest close trades.
            # Range playbooks get the native H4 frame (no resample).
            states = read_and_featurize(h1, None)
            _playbooks = self._get_playbooks()
            print(f"MT5 startup: warming {len(_playbooks)} "
                  "playbooks over history…", flush=True)
            for pb in _playbooks:
                if hasattr(pb, "set_h4_frame"):
                    pb.set_h4_frame(
                        self._stack.frames.get("H4"))
                _pb_t0 = time.perf_counter()
                for t2 in range(len(h1)):
                    pb.live_signal(states, t2, h1)
                print(f"MT5 startup: warmed {pb.name} "
                      f"in {time.perf_counter() - _pb_t0:.1f}s", flush=True)
            self._send_startup_progress(
                "MT5 engine: playbooks warmed "
                f"({len(_playbooks)} playbooks)")
            # Adopt positions the broker already holds with our magic
            # (restart sync — MT5 is the source of truth for open trades).
            self._adopt_existing(executor)
            # Adopt pending orders the broker still holds with our magic:
            # a restart must not orphan a resting limit/stop.
            self._adopt_existing_pending(executor)

            # Bound the tick warehouse now that positions are known:
            # aged day files go, open-position days never do (see
            # _prune_tick_csvs).  Never raises into startup.
            self._prune_tick_csvs()

            # Startup ping: ONE Telegram "engine started" message, after
            # feed connect + runner start, independent of command polling
            # (it fires even with --no-poll-commands).  Never raises.
            self._send_startup_ping(acct)
            # Console confirmation that startup COMPLETED: without it
            # the console goes blank after "playbooks warmed" and the
            # owner cannot tell a started engine from a stalled one.
            # (The TG ping above is Telegram-only; this is console-only.)
            print(
                "MT5 startup: engine started — tick pump running; "
                "commands polled every "
                f"{self.cfg.poll_interval_seconds:.0f}s even with no "
                "ticks (market closed => no new ticks, polling still "
                "runs)",
                flush=True)
        except Exception as _startup_exc:
            self._send_startup_progress(
                f"MT5 engine FAILED to start: {_startup_exc}")
            raise

        stats: dict = {"ticks": 0, "bars": 0, "signals": 0,
                       "executed": 0, "shadow": 0, "gated": 0,
                       "hierarchy_blocked": 0, "tick_actions": 0,
                       "reconciled_closed": 0, "triggers": 0,
                       "invalidated": 0, "pending": 0,
                       "pending_filled": 0, "pending_cancelled": 0,
                       "qm_d1_signals": 0, "qm_m5_signals": 0}
        # Daily QM (P3 real order path): evaluate the D1 frame
        # once at warm start (the latest touch, if any, is
        # evaluated here; later touches on their D1 close by
        # _refresh_bars).  Phase 2: the warm-start scan is
        # source="warmup_scan" — its candidates are journaled
        # with full provenance and are categorically
        # execution-ineligible (candidate_lifecycle); they can
        # never reach the order path.
        self._run_qm_d1(self._stack.frames.get("D1"), stats,
                        source="warmup_scan")
        # M5 QM (funnel §2 native route): same shape — evaluate
        # the M5 frame once at warm start (also feeds the
        # _qm_touch annotation state); later touches on their
        # M5 close by _refresh_bars.
        self._run_qm_m5(self._stack.frames.get("M5"), stats,
                        source="warmup_scan")
        # Phase 2 (P0-3): the scheduled roster evaluations run
        # once at warm start too — every playbook's evaluation
        # (or its honest not_eligible reason) is on record from
        # the run's first minutes, not only after its clock or
        # the trigger happens to fire.
        self._run_roster_close_evals(stats, source="warmup_scan")
        self._run_range_family("h4_generation", stats,
                               source="warmup_scan")
        self._run_range_family("h1_refinement", stats,
                               source="warmup_scan")
        pump = TickPump(feed, interval=self.cfg.tick_interval)
        try:
            # on_idle: command polling MUST NOT depend on new ticks.
            # With the market closed the terminal returns the same
            # stale tick forever, TickPump dedupes it, and _on_tick
            # never fires — so a poller that only runs inside
            # _on_tick never fires either and the engine goes deaf
            # to /status.  The idle hook runs the poller on every
            # spin; _poll_telegram_commands' own _last_poll gate
            # keeps it to once per poll_interval_seconds.
            pump.run(self._on_tick(stats), max_ticks=max_ticks,
                     on_idle=lambda: self._poll_telegram_commands(stats))
        finally:
            # Engine stop: cancel every pending order.  A resting
            # limit/stop left on the broker would fill later with no
            # management — a stale pending is never left behind.
            for ticket in list(self._pending_orders):
                self._cancel_pending(ticket, "engine_stop", stats)
            self._det_journal("runner_stopped", dict(stats))
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
            # P3 spread state (S3/S4): rolling sample book; the
            # ratio vs the session norm is the dead-window /
            # spike proof — measured, never clock-assumed.
            try:
                sp = float(tick.get("spread") or 0.0)
                if sp > 0:
                    self._spread_samples.append(sp)
                    if len(self._spread_samples) > 600:
                        self._spread_samples = \
                            self._spread_samples[-600:]
                    if len(self._spread_samples) >= 20:
                        srt = sorted(self._spread_samples)
                        med = srt[len(srt) // 2]
                        if med > 0:
                            self._spread_ratio = sp / med
            except Exception:
                pass
            self._health.note_tick(tick.get("time"))
            # Day rollover -> the honest recap for the day/week that
            # just ended (wins AND losses).  Never breaks the loop.
            _day_before = self._last_tick_date
            self._maybe_send_rollover_recap(tick)
            if (_day_before is not None
                    and self._last_tick_date is not None
                    and self._last_tick_date != _day_before):
                # New UTC day: bound the tick warehouse (aged day
                # files go; open-position days never do).
                self._prune_tick_csvs()
            # Friday close (action "close" only): existing positions
            # + pendings are closed at the Friday hour, once per day.
            # Default action "hold" is a no-op here.  A COMPLETE
            # regime pause stops even this: position state is
            # untrusted, so nothing may act on it.
            if self._regime_pause != "paused_complete":
                self._maybe_friday_close(tick, stats)
            # Tick CSV: exactly as seen, never breaks the loop.
            if self._tick_logger is not None:
                try:
                    self._tick_logger.log(tick)
                except Exception as exc:
                    self.journal.log_note(
                        f"mt5: tick log failed (loop continues): {exc}")
            price = tick["mid"]
            # Tick management (breakeven / trail / partial) keeps
            # running under "paused_new" / "observation" — it reads
            # position state, not regime_state.  Only a COMPLETE
            # pause (position state itself untrusted) stops it.
            if self._regime_pause != "paused_complete":
                self._manage_tick(price, stats,
                                  bid=tick.get("bid"),
                                  ask=tick.get("ask"))
            if self._tick_count % self.cfg.reconcile_every_ticks == 0:
                if self._regime_pause != "paused_complete":
                    self._reconcile(price, stats)
            self._refresh_bars(tick, stats)
            # Forecast alerts (owner rulings §3): scored on D1 closes
            # — a days-ahead forecast is daily news, not hourly noise.
            # The H1 frame stays the model's input; only the cadence
            # changed. Forecasts never touch orders — this call reads
            # bars and sends an alert, nothing else.
            self._maybe_forecast_alert()
            # Live evaluation clock: touch arms, family confirms on
            # entry-TF closes.  H1 closes never gate evaluation.
            self._evaluate_live_tick(tick, stats)
            # Golden Swing confirm timeouts are wall-clock: sweep
            # every tick (a tap processed after timeout is late —
            # the token is already gone, "confirmation expired").
            self._sweep_order_confirms(stats)
            self._poll_telegram_commands(stats)
        return _handle

    def _sl_margin(self, bid: float, ask: float) -> float:
        """Minimum distance a proposed SL must keep from the quote.

        Broker stops/freeze levels (in points x point size) when
        the executor reports them; else one spread when a real
        quote is available; else 0 (legacy mid-only calls — the
        side check itself still applies there).
        """
        try:
            cons = self.executor.sl_constraints(self.cfg.symbol)
        except Exception:
            cons = None
        if cons:
            try:
                point = float(cons.get("point") or 0.0)
                pts = max(int(cons.get("stops_level_points") or 0),
                          int(cons.get("freeze_level_points") or 0))
                if point > 0 and pts > 0:
                    return pts * point
            except Exception:
                pass
        if ask > bid:
            return float(ask) - float(bid)
        return 0.0

    def _manage_tick(self, price: float, stats: dict,
                     bid: float | None = None,
                     ask: float | None = None) -> None:
        """TickManager actions for every tracked position, executed now.

        Every executed management action that changes the position
        gets a Telegram follow-up (breakeven / partial / layer stop)
        — the disciplined caller's update — on top of the journal
        note.  Trail SL moves stay journal-only: they fire on
        nearly every favourable tick and would spam the channel.

        SL moves are broker-aware before they are sent: a proposed
        stop on the wrong side of the quote, or inside the
        broker's stops/freeze distance, is stood down (noted once
        per window, retried when price recovers) instead of
        resubmitted every tick — the Oct 27 journal shows 52
        identical "Invalid stops" for one ticket.  A broker
        rejection is bounded to 3 submissions per side-valid
        window, and a "no position" report stands the ticket down
        for reconcile instead of retrying a closed trade.
        """
        bid = float(bid) if bid is not None else float(price)
        ask = float(ask) if ask is not None else float(price)
        for ticket in list(self.tick_manager.tickets()):
            for action in self.tick_manager.on_tick(ticket, price):
                stats["tick_actions"] += 1
                pos = self.tick_manager.position(ticket)
                pb = self._playbook_from_alert_id(
                    pos.alert_id if pos is not None else "")
                if action.kind == "modify_sl":
                    level = float(action.level)
                    if self.tick_manager.sl_blocked(
                            ticket, action.reason, level):
                        # Window exhausted; already journalled at
                        # block time.  Silent until the window
                        # re-opens (quote went side-invalid).
                        continue
                    direction = int(pos.direction) \
                        if pos is not None else 0
                    margin = self._sl_margin(bid, ask)
                    if direction != 0 and not sl_level_valid(
                            direction, level, bid, ask, margin):
                        # Side-invalid: close the failure window
                        # (a fresh budget when price recovers) and
                        # note the stand-down once per window.
                        self.tick_manager.reset_sl_window(
                            ticket, action.reason, level)
                        key = (ticket, action.reason, level)
                        if key not in self._sl_stood_down:
                            self._sl_stood_down.add(key)
                            self.journal.log_note(
                                f"mt5 tick: ticket {ticket} SL move "
                                f"STOOD DOWN ({action.reason}): "
                                f"proposed {level:.2f} not valid at "
                                f"bid {bid:.2f}/ask {ask:.2f} "
                                f"(margin {margin:.2f}) — retry when "
                                f"price recovers")
                        continue
                    self._sl_stood_down.discard(
                        (ticket, action.reason, level))
                    res = self.executor.modify_sl(ticket, level)
                    if res.ok:
                        self.tick_manager.apply(ticket, action)
                        # A13: the pool's base layer rolls with the
                        # broker stop (refund net of reserve).
                        if pos is not None:
                            self._pool_sync_stop(
                                ticket, pos.alert_id,
                                float(action.level))
                        self.journal.log_note(
                            f"mt5 tick: ticket {ticket} SL -> "
                            f"{action.level:.2f} ({action.reason})")
                        if action.reason == "breakeven":
                            self._send_alert(
                                f"MT5 BREAKEVEN {pb} ticket {ticket}: "
                                f"SL -> entry {action.level:.2f} — "
                                f"risk off, trade is now free")
                            # P3 (J): the BE event joins the
                            # structured record, WITH its earning
                            # evidence (funding kind / tp1 banked /
                            # backstop) — F13's audit trail.
                            if pos is not None:
                                _be = self.tick_manager.be_state(
                                    ticket)
                                log_signal(
                                    self.journal,
                                    signal_id=str(pos.alert_id),
                                    playbook=pb,
                                    direction=int(pos.direction),
                                    score=float(
                                        (self._position_ctx.get(
                                            int(ticket)) or {}).get(
                                            "entry_total") or 0.0),
                                    decision="management",
                                    reason=(f"be_earned: "
                                            f"{_be.get('by')}"
                                            if _be else
                                            "be_earned: plan"),
                                    context=self._decision_context())
                        elif action.reason == "profit_lock":
                            self._send_alert(
                                f"MT5 PROFIT LOCK {pb} ticket {ticket}: "
                                f"SL -> {action.level:.2f} — profit "
                                f"locked in, trade cannot lose")
                    else:
                        if str(res.message).startswith("no position"):
                            # Broker-closed between ticks: stand
                            # down, never retry.  The close record
                            # itself comes from _reconcile on its
                            # normal cadence — this path must not
                            # invent or duplicate it.
                            if self.tick_manager.mark_broker_gone(
                                    ticket):
                                self.journal.log_note(
                                    f"mt5 tick: ticket {ticket} SL "
                                    f"move FAILED ({action.reason}): "
                                    f"broker has no such position — "
                                    f"tracking stood down, reconcile "
                                    f"will close it")
                        else:
                            blocked = self.tick_manager.note_sl_failure(
                                ticket, action.reason, level)
                            cur_sl = self.tick_manager.effective_stop(
                                ticket)
                            spread = (ask - bid) \
                                if ask > bid else 0.0
                            self.journal.log_note(
                                f"mt5 tick: ticket {ticket} SL move "
                                f"FAILED ({action.reason}): attempted "
                                f"{level:.2f} vs broker SL "
                                f"{cur_sl if cur_sl is not None else float('nan'):.2f} "
                                f"at bid {bid:.2f}/ask {ask:.2f} "
                                f"spread {spread:.2f} "
                                f"margin {margin:.2f} "
                                f"retcode {res.retcode}: "
                                f"{res.message}"
                                + (" — level blocked after 3 "
                                   "rejections in this window"
                                   if blocked else ""))
                elif action.kind == "partial":
                    res = self.executor.close_position(ticket, action.volume)
                    if res.ok:
                        # PNL-PARTIAL: the leg's money is booked NOW
                        # (day/week via record_partial + the leg
                        # stored on the position for the close
                        # total).  Before this, banked legs never
                        # entered any book — the close booked the
                        # remainder only and the banked profit was
                        # invisible.  A FAILED partial books
                        # nothing (below) — retry next tick.
                        leg = 0.0
                        if pos is not None:
                            leg = self._book_partial_leg(
                                ticket, pos, volume=action.volume,
                                price=price, entry=pos.entry_price,
                                stop=pos.stop0)
                        self.tick_manager.apply(ticket, action)
                        self.journal.log_note(
                            f"mt5 tick: ticket {ticket} partial close "
                            f"{action.volume} lots at {price:.2f} "
                            f"(banked ${leg:+.2f})")
                        self._send_alert(
                            f"MT5 PARTIAL {pb} ticket {ticket}: banked "
                            f"{action.volume} lots @ {price:.2f} "
                            f"(remainder rides)")
                    else:
                        self.journal.log_note(
                            f"mt5 tick: ticket {ticket} partial FAILED: "
                            f"{res.message} — retry next tick")
                elif action.kind == "layer_stop":
                    # Added layer's own SL crossed: close just that
                    # leg, the base rides.  The leg's money is
                    # measured against the LAYER's entry (looked up
                    # before apply pops it), not the base entry.
                    res = self.executor.close_position(ticket, action.volume)
                    if res.ok:
                        leg = 0.0
                        if pos is not None:
                            layer = next(
                                (ly for ly in
                                 self.tick_manager.layers(ticket)
                                 if ly.volume == action.volume
                                 and (action.level is None
                                      or ly.sl == action.level)),
                                None)
                            leg = self._book_partial_leg(
                                ticket, pos, volume=action.volume,
                                price=price,
                                entry=(layer.entry if layer is not None
                                       else pos.entry_price),
                                stop=(layer.sl if layer is not None
                                      else pos.stop0))
                        self.tick_manager.apply(ticket, action)
                        self.journal.log_note(
                            f"mt5 tick: ticket {ticket} layer stop "
                            f"{action.volume} lots at {price:.2f} "
                            f"(layer SL {action.level:.2f}, "
                            f"banked ${leg:+.2f})")
                        self._send_alert(
                            f"MT5 LAYER STOP {pb} ticket {ticket}: "
                            f"closed {action.volume} lots @ {price:.2f} "
                            f"(layer SL {action.level:.2f}, base rides)")
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
                elif action.kind == "panic":
                    # Panic-candle emergency close: adverse excursion
                    # past 3x ATR with severity > 1.5 — close the
                    # whole position NOW at market (the broker SL is
                    # too far to help).
                    still = self.executor.positions()
                    if any(p.ticket == ticket for p in still):
                        res = self.executor.close_position(ticket)
                        if res.ok:
                            self._record_close(ticket, price, "panic",
                                               stats)
                            self._send_alert(
                                f"MT5 PANIC CLOSE {pb} ticket {ticket}: "
                                f"adverse excursion past panic "
                                f"threshold — closed @ {price:.2f}")
                    else:
                        self._record_close(ticket, price, "panic", stats)
                elif action.kind == "scalp_cap":
                    # Scalp holding cap: wall-clock hold past the cap
                    # — close the whole position NOW.
                    still = self.executor.positions()
                    if any(p.ticket == ticket for p in still):
                        res = self.executor.close_position(ticket)
                        if res.ok:
                            self._record_close(ticket, price,
                                               "scalp_cap", stats)
                            self._send_alert(
                                f"MT5 SCALP CAP {pb} ticket {ticket}: "
                                f"held past the scalp cap — closed "
                                f"@ {price:.2f}")
                    else:
                        self._record_close(ticket, price, "scalp_cap",
                                           stats)

    def _broker_close_price(self, ticket: int,
                            fallback: float) -> tuple[float, str | None]:
        """Step 12d: price a vanished position's exit from the
        broker's closing DEAL when history has it — the tick
        snapshot mis-recorded exits by up to 0.64R (2026-10-28:
        a TP fill at 4046.19 journalled as 4042.89).  Returns
        (price, provenance note); the snapshot stays the fallback
        (older executor, history lag) and the note says which
        source priced the exit, so misses stay countable."""
        getter = getattr(self.executor, "closing_deal", None)
        if getter is None:
            return fallback, None
        try:
            deal = getter(int(ticket))
        except Exception:
            return fallback, None
        if not deal or not float(deal.get("price") or 0.0):
            return fallback, None
        price = float(deal["price"])
        return price, (
            f"mt5 close: ticket {ticket} exit priced from the "
            f"broker deal ({deal.get('reason') or 'deal'} @ "
            f"{price:.2f}); tick snapshot was {fallback:.2f}")

    def _reconcile(self, price: float, stats: dict) -> None:
        """MT5 positions_get is the truth: drop broker-closed, adopt ours."""
        try:
            live = {p.ticket: p for p in self.executor.positions()}
        except Exception as exc:
            self.journal.log_note(f"mt5 reconcile: positions failed: {exc}")
            return
        for ticket in list(self.tick_manager.tickets()):
            if ticket not in live:
                close_price, deal_note = self._broker_close_price(
                    ticket, price)
                if deal_note:
                    self.journal.log_note(deal_note)
                self._record_close(ticket, close_price,
                                   "broker_closed", stats)
        # Positions that match a tracked pending order are NOT adopted
        # here: _reconcile_pending below handles them as FILLS (full
        # open record + fill chart), not as bare adoptions — adopting
        # first would double-handle them and lose the signal context.
        pending_alerts = {s.alert_id for s in self._pending_orders.values()}
        # A broker position sharing an alert_id with a HELD position
        # is the remainder of a partial fill (hedging: a second
        # ticket for the same order) — never adopt it as a second
        # position; the remainder fill lands on the held position
        # via _record_pending_fill.
        held_alerts = set()
        for _t in self.tick_manager.tickets():
            _p = self.tick_manager.position(_t)
            if _p is not None and not _p.closed:
                held_alerts.add(_p.alert_id)
        for ticket, pos in live.items():
            if ticket not in self.tick_manager.tickets() \
                    and pos.magic == self.cfg.magic \
                    and pos.alert_id not in pending_alerts \
                    and pos.alert_id not in held_alerts:
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
        both proves an external cancel.

        Fill detection is by broker VOLUME for the setup (live
        volume minus already-tracked volume), not by the pending
        leaving ``orders_get``: a PARTIAL fill leaves the
        remainder resting, so "pending still there" must not
        hide the fill.  A netting remainder fill grows the
        already-tracked position (same ticket); a hedging one
        appears as a new ticket — both are the same fill event
        (the volume not yet tracked).
        """
        try:
            live_pending = {o.ticket: o for o in
                            self.executor.pending_orders(self.cfg.symbol)}
        except Exception as exc:
            self.journal.log_note(
                f"mt5 reconcile: pending orders failed: {exc}")
            return
        for ticket in list(self._pending_orders):
            state = self._pending_orders.get(ticket)
            if state is None:
                continue
            still_resting = ticket in live_pending
            mine = [p for p in live_positions.values()
                    if p.alert_id == state.alert_id
                    and p.magic == self.cfg.magic]
            if not mine:
                if not still_resting:
                    del self._pending_orders[ticket]
                    stats["pending_cancelled"] = stats.get(
                        "pending_cancelled", 0) + 1
                    self.journal.log_note(
                        f"mt5 pending gone: ticket {ticket} "
                        f"{state.kind} @ {state.level:.2f} — cancelled "
                        f"externally / broker expiry, no fill")
                continue
            # New fill volume = live volume for this setup minus
            # already-tracked volume for it.  Zero means nothing
            # new (the fill was already recorded, or the pending
            # is genuinely still resting with no fill).
            tracked_vol = 0.0
            for _t in self.tick_manager.tickets():
                _pp = self.tick_manager.position(_t)
                if _pp is not None and not _pp.closed \
                        and _pp.alert_id == state.alert_id:
                    tracked_vol += float(_pp.volume)
            live_vol = sum(float(p.volume) for p in mine)
            if live_vol - tracked_vol <= _VOL_EPS:
                continue  # nothing new
            # Representative position for the fill event: the
            # untracked one if any (first fill / hedging), else
            # the grown one (netting remainder / layer).
            untracked = [p for p in mine
                         if p.ticket not in self.tick_manager.tickets()]
            pos = untracked[0] if untracked else mine[0]
            self._record_pending_fill(state, pos, stats)
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

    @staticmethod
    def _playbook_from_alert_id(alert_id: str) -> str:
        """Playbook out of ``MT5-<playbook>-<index>`` (``?`` when the
        alert id does not follow the shape — follow-up text only,
        never a gate)."""
        if alert_id.startswith("MT5-") and "-" in alert_id[4:]:
            return alert_id[4:].rsplit("-", 1)[0] or "?"
        return "?"

    def _classify_close(self, pos, price: float, reason: str) -> str:
        """What the close MEANS, for the journal note + Telegram
        follow-up (the structured journal ``reason`` keeps the raw
        rule reason — ``stop`` / ``broker_closed`` / ``timeout`` —
        so downstream joins never change).

        - ``timeout`` stays ``timeout`` (the rule that fired), even
          when the exit is in profit.
        - The safety-net ``stop`` is an SL hit; when the stop that
          got hit is the TRAIL level or the BREAKEVEN level, it says
          so (a breakeven stop is risk-off, not a loss).
        - ``broker_closed`` is the broker's TP or SL (or a manual
          close): classify by price against the signal's first
          target and the effective broker stop.  Between the two,
          it is an unexplained broker close — reported as such,
          never dressed up as a TP.
        """
        d = pos.direction
        if reason == "timeout":
            return "timeout"
        if reason == "stop":
            return self._stop_label(pos)
        if reason == "broker_closed":
            targets = list(getattr(pos, "targets", None) or [])
            if targets:
                tp = targets[0]
                if (d == 1 and price >= tp) or \
                        (d == -1 and price <= tp):
                    return "TP hit"
            eff = self.tick_manager.effective_stop(pos.ticket)
            if eff is not None and ((d == 1 and price <= eff) or
                                    (d == -1 and price >= eff)):
                return self._stop_label(pos)
            return "broker_closed"
        return reason

    def _stop_label(self, pos) -> str:
        """Which stop got hit: breakeven (risk-off), trail, or the
        structural stop."""
        if abs((self.tick_manager.effective_stop(pos.ticket) or 0.0)
               - pos.entry_price) < 1e-9 and pos.risk_price > 0:
            return "SL hit (breakeven)"
        if pos.stop_is_trail:
            return "SL hit (trail)"
        return "SL hit"

    def _narrative_fill_plan(self, sig, fill_price: float):
        """Build the narrative MANAGEMENT state for a fill.

        Only narrative signals (``kind == "narrative"``) get a plan.
        The chain is rebuilt from the live frames at fill time
        (``advance`` from a fresh state — the same evaluation the
        NarrativePlaybook ran when it emitted the signal).
        REWORK (spec §3.2): if the chain is no longer in ENTRY at
        fill time the plan is STILL built — from the fill, via
        ``mark_entered`` — because the fill was score-authorized
        and the chain's ENTRY precondition was a gating artifact.
        ``mark_entered`` locks the SL at fill and builds the
        differentiated management plan (clean/late, tier,
        opposing, ATR regime).

        Returns ``(state, kwargs)`` where ``kwargs`` are the
        ``tick_manager.open`` narrative kwargs; ``(None, {})``
        when no plan applies.  Fail-open: any failure journals a
        note and the position runs on TickConfig defaults.
        """
        try:
            if getattr(sig, "kind", "") != "narrative":
                return None, {}
            if self._stack is None:
                return None, {}
            frames = dict(self._stack.frames or {})
            if not frames:
                return None, {}
            state = advance_narrative(NarrativeState(), frames)
            if state.state != _NARR_ENTRY:
                # REWORK (spec §3.2, following the CODE): the
                # per-position state write is re-pointed at
                # plan_management via mark_entered.  Under the
                # rework a narrative fill is score-authorized, so
                # the chain may have moved past ENTRY (or never
                # completed) by fill time — refusing the plan
                # would leave a live position on TickConfig
                # defaults for a gating artifact that no longer
                # exists.  The runner attests the fill: the state
                # is set to ENTRY with the SIGNAL's direction and
                # the one real plan builder runs unchanged (tier /
                # opposing / ATR adjustments included).
                self.journal.log_note(
                    "mt5 narrative: chain not ENTRY at fill "
                    f"(state {state.state}) — management plan "
                    "built from the fill itself (score-authorized, "
                    "spec §3.2 re-point)")
                state.direction = int(sig.direction)
                state.state = _NARR_ENTRY
            state = _mark_narrative_entered(
                state, entry_price=float(fill_price),
                stop=float(sig.invalidation),
                targets=[float(x) for x in (sig.targets or [])])
            plan = dict(getattr(state, "management", None) or {})
            if not plan:
                return None, {}
            confirm = getattr(state, "confirm", None) or {}
            fvg = confirm.get("fvg") if isinstance(confirm, dict) else None
            kwargs = {
                "narrative_plan": plan,
                "narrative_atr": float(
                    getattr(state, "atr_value", 0.0) or 0.0),
                "narrative_choch_extreme": getattr(
                    state, "invalidation", None),
                "narrative_fvg": dict(fvg) if isinstance(fvg, dict)
                else None,
            }
            return state, kwargs
        except Exception as exc:
            try:
                self.journal.log_note(
                    f"mt5 narrative: fill-plan build failed ({exc}) — "
                    "position runs on TickConfig defaults")
            except Exception:
                pass
            return None, {}

    def _signed_leg_pnl(self, *, direction: int, price: float,
                        entry: float, volume: float) -> float:
        """Signed money of one leg: money_for_move abs()es the
        distance, so sign it by hand — a long below entry (or a
        short above) is a LOSS.  Booking the unsigned value would
        count every losing leg as a win.  0.0 when the spec is
        unreadable (fail-open: the volume bookkeeping still
        happens; the money gap is noted by the caller path)."""
        try:
            spec = self.executor.symbol_spec(self.cfg.symbol)
            pnl = spec.money_for_move(price - entry, volume)
            if direction * (price - entry) < 0:
                pnl = -pnl
            return float(pnl)
        except Exception:
            return 0.0

    def _book_partial_leg(self, ticket: int, pos, *, volume: float,
                          price: float, entry: float,
                          stop: float) -> float:
        """Book one executed partial leg: its money into day/week
        (RiskGuards record_partial — NOT a closed trade, so
        trades_today / the loss streak are untouched) and onto
        the position for the close total.  Never breaks the
        caller: a booking failure is noted, the volume reduction
        still applies.  Returns the leg money."""
        leg = self._signed_leg_pnl(direction=pos.direction, price=price,
                                   entry=entry, volume=volume)
        try:
            self.tick_manager.bank_leg(
                ticket, volume=volume, price=price, entry=entry,
                stop=stop, pnl_money=leg)
        except Exception as exc:
            self.journal.log_note(
                f"mt5: leg store failed for ticket {ticket}: {exc}")
        try:
            balance = float(self.feed.account_state().get("balance", 0.0)) \
                if self.feed is not None else 0.0
            if balance > 0:
                self.guards.record_partial(float(leg), balance)
        except Exception as exc:
            self.journal.log_note(
                f"mt5: partial P&L not booked for ticket {ticket}: "
                f"{exc}")
        return leg

    def _record_close(self, ticket: int, price: float, reason: str,
                      stats: dict) -> None:
        pos = self.tick_manager.position(ticket)
        if pos is None:
            return
        d = pos.direction
        pnl_r = d * (price - pos.entry_price) / pos.risk_price \
            if pos.risk_price > 0 else 0.0
        # Remainder leg's money (pos.volume is the volume LEFT
        # after every partial reduced it).  The trade total adds
        # the banked legs — PNL-PARTIAL: before this fix the
        # close booked the remainder only and the banked legs
        # vanished from every book.
        remainder_pnl = 0.0
        try:
            spec = self.executor.symbol_spec(self.cfg.symbol)
            # money_for_move abs()es the distance, so sign it by
            # hand: a long below entry (or a short above) is a LOSS.
            # Booking the unsigned value would count every losing
            # long as a win — in the journal, the recaps, and the
            # RiskGuards outcome feed.
            remainder_pnl = spec.money_for_move(
                price - pos.entry_price, pos.volume)
            if d * (price - pos.entry_price) < 0:
                remainder_pnl = -remainder_pnl
        except Exception:
            pass
        pnl_money = float(getattr(pos, "banked_pnl", 0.0) or 0.0) \
            + float(remainder_pnl)
        # Volume-weighted total R (live/exit.realized_r — the same
        # formula the paper path uses): banked fractions at their
        # leg R plus the remainder fraction at the final R.
        # Fractions are taken against the FINAL orig_volume (legs
        # + remainder sum to it exactly), so the sum can never
        # exceed 1.0.  Any failure falls back to the remainder R
        # — the money total above stays exact regardless.
        total_r = float(pnl_r)
        partials: list = []
        try:
            legs = getattr(pos, "banked_legs", None) or []
            orig = float(getattr(pos, "orig_volume", 0.0) or 0.0)
            if orig > 0 and legs:
                partials = [{
                    "fraction": float(leg["volume"]) / float(orig),
                    "r": float(partial_r(
                        entry_price=float(leg["entry"]),
                        stop=float(leg["stop"]), direction=d,
                        exit_price=float(leg["price"])))}
                    for leg in legs]
                total_r = float(realized_r(partials, float(pnl_r)))
        except Exception:
            total_r = float(pnl_r)
            partials = []
        # Classify BEFORE remove(): the classifier reads the live
        # effective stop from the TickManager.
        display = self._classify_close(pos, price, reason)
        self.journal.log_close(
            alert_id=pos.alert_id, ticket=ticket, exit_price=price,
            reason=reason, pnl_money=float(pnl_money), r_net=float(total_r),
            hold_bars=0, partial_taken=pos.partial_done)
        self._det_journal("trade_closed", {
            "alert_id": pos.alert_id, "ticket": ticket,
            "exit": price, "reason": reason,
            "pnl_money": float(pnl_money), "r_net": float(total_r)},
            idempotency_key=f"close-{pos.alert_id}-{ticket}")
        self.journal.log_note(
            f"mt5 close: ticket {ticket} {display} at {price:.2f} "
            f"({total_r:+.2f}R)")
        # Telegram follow-up — the disciplined caller's update: every
        # close is reported, wins AND losses, with the R and the
        # reason (a loss is never silently journalled).
        pb = self._playbook_from_alert_id(pos.alert_id)
        self._send_alert(
            f"MT5 CLOSE {pb} {'+' if d > 0 else '-'}1 "
            f"{float(getattr(pos, 'orig_volume', 0.0) or pos.volume)} lots entry {pos.entry_price:.2f} -> exit "
            f"{price:.2f} — {display} {total_r:+.2f}R "
            f"(ticket {ticket})")
        # Feed the closed trade's outcome to RiskGuards: the
        # daily/weekly/drawdown/consecutive-loss halts only work if
        # outcomes are booked.  Never breaks the close path.
        # PNL-PARTIAL: the close books the REMAINDER only — the
        # banked legs already entered day/week at partial time
        # (record_partial).  Booking the total here would double
        # -count every banked leg.  The journal close record above
        # carries the trade TOTAL (banked + remainder, as the
        # paper CloseRecord does); the ledger increment here is
        # the remainder, so both surfaces reconcile to the total.
        try:
            balance = float(self.feed.account_state().get("balance", 0.0))
            if balance > 0:
                self.guards.record_outcome(float(remainder_pnl), balance)
        except Exception as exc:
            self.journal.log_note(
                f"mt5: risk outcome not booked for ticket {ticket}: "
                f"{exc}")
        # Feed the close to the entry gates: the loss/win cooldown
        # starts at the close.  Never breaks the close path.
        # PNL-PARTIAL: the cooldown keys on the TRADE's total R,
        # not the remainder leg's — a trade that banked a win then
        # stopped the remainder at a smaller loss is a WIN.
        try:
            self.entry_gate.record_close(time.time(), total_r > 0)
        except Exception:
            pass
        # Narrative STATE 7: a narrative trade's close finalizes the
        # chain (journal entry + reset to IDLE).  Fail-open: a
        # finalize failure never breaks the close path.  Partials
        # never reach here (they go through tick_manager.apply),
        # so this is the full close only.
        _narr_state = self._narrative_states.pop(ticket, None)
        # Rework: close out the position's RiskPool ledger — its
        # final state joins the close evidence (A13/A14).
        # getattr-guarded like everything else on this fail-open
        # path: a partially constructed runner must never crash a
        # close.
        _pools = getattr(self, "_pools", None)
        _pool = _pools.pop(ticket, None) if _pools is not None \
            else None
        _funded = getattr(self, "_funded_events", None)
        if _funded is not None:
            _funded.pop(ticket, None)
        # P3: the management context dies with the position.
        _pctx = getattr(self, "_position_ctx", None)
        if _pctx is not None:
            _pctx.pop(ticket, None)
        if _pool is not None:
            self.journal.log_note(
                f"mt5 pool closed: ticket {ticket} budget "
                f"{_pool.budget} consumed at close "
                f"{_pool.consumption()} ({pb} {total_r:+.2f}R)")
        if _narr_state is not None:
            try:
                _idle, _entry = _finalize_narrative_exit(
                    _narr_state, reason=reason, final_r=float(pnl_r),
                    partials=partials or None,
                    pnl_dollars=float(pnl_money), exit_price=float(price))
                self.journal.log_note(
                    f"mt5 narrative STATE 7 EXIT: "
                    f"{_entry.get('exit_reason', reason)} "
                    f"R={_entry.get('realized_r', total_r):+.2f} "
                    f"(ticket {ticket})")
            except Exception as exc:
                self.journal.log_note(
                    f"mt5 narrative finalize failed for ticket "
                    f"{ticket}: {exc}")
        self.tick_manager.remove(ticket)
        stats["reconciled_closed"] += 1

    def _maybe_friday_close(self, tick: dict, stats: dict) -> None:
        """Friday-close action "close": at/after the Friday hour, close
        every open position + cancel every pending order, once per
        UTC date.  Action "hold" (default) never reaches here.
        Never breaks the tick loop."""
        if self.cfg.friday_close_action != "close":
            return
        try:
            now = _tick_now_utc(tick)
            if now is None:
                return
            day = now.date()
            if day == self._friday_close_done_date:
                return
            if not is_friday_close(now, self.cfg.friday_close_hour):
                return
            self._friday_close_done_date = day
            price = float(tick.get("mid") or 0.0)
            for ticket in list(self.tick_manager.tickets()):
                try:
                    res = self.executor.close_position(ticket)
                    if res.ok:
                        self._record_close(ticket, price,
                                           "friday_close", stats)
                except Exception as exc:
                    self.journal.log_note(
                        f"mt5 friday close: ticket {ticket} failed: "
                        f"{exc}")
            for ticket in list(self._pending_orders):
                self._cancel_pending(ticket, "friday_close", stats)
            self.journal.log_note(
                "mt5 friday close: positions + pendings closed for "
                "the weekend")
        except Exception as exc:
            self.journal.log_note(
                f"mt5 friday close failed (loop continues): {exc}")

    def _prune_tick_csvs(self) -> list[str]:
        """Bound the tick-CSV warehouse (rollover + startup hook).

        Deletes day files older than ``tick_csv_retain_days``,
        except every file from the earliest open position's open
        day onward — a live position's ticks are never evicted
        while it is open.  Fail-closed: an open position with an
        unknown open time (adopted positions carry
        ``opened_at=None``) skips pruning entirely rather than
        evicting on a guessed span; the skip is journaled.
        Never raises into the tick loop or startup.
        """
        if self._tick_logger is None:
            return []
        try:
            keep_since = None
            for ticket in self.tick_manager.tickets():
                pos = self.tick_manager.position(ticket)
                if pos is None:
                    continue
                opened = getattr(pos, "opened_at", None)
                if opened is None:
                    self.journal.log_note(
                        "mt5: tick-CSV prune SKIPPED — open position "
                        f"(ticket {ticket}) has unknown open time "
                        "(adopted); no day files evicted")
                    return []
                from datetime import datetime, timezone
                day = datetime.fromtimestamp(
                    float(opened), tz=timezone.utc).strftime("%Y%m%d")
                keep_since = day if keep_since is None \
                    else min(keep_since, day)
            deleted = self._tick_logger.prune(
                keep_since_day=keep_since)
            if deleted:
                self.journal.log_note(
                    "mt5: tick-CSV prune deleted "
                    f"{len(deleted)} aged day file(s): "
                    f"{', '.join(deleted)}")
            return deleted
        except Exception as exc:
            try:
                self.journal.log_note(
                    f"mt5: tick-CSV prune failed (loop continues): "
                    f"{exc}")
            except Exception:
                pass
            return []

    def _maybe_send_rollover_recap(self, tick: dict) -> None:
        """UTC day rollover -> Telegram recap(s) for what just ended.

        On the first tick of a new UTC day: the DAILY recap for the
        day that just ended.  When the new day is a Monday, the
        WEEKLY recap for the Mon–Sun week that just ended too.
        Both report wins AND losses (see reporting.py) — the recap
        never hides losers.  No senders configured -> nothing sent
        (and no journal read).  Never raises into the tick loop; a
        recap failure is one journal note.
        """
        if not self.senders:
            # Still advance the clock: a sender-less run must not
            # "roll over" the same day twice if senders appear later.
            try:
                ts = pd.Timestamp(tick["time"])
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                self._last_tick_date = ts.tz_convert("UTC").date()
            except Exception:
                pass
            return
        try:
            ts = pd.Timestamp(tick["time"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            day = ts.tz_convert("UTC").date()
        except Exception:
            return
        prev = self._last_tick_date
        self._last_tick_date = day
        if prev is None or prev == day:
            return  # first tick, or same day: no rollover
        try:
            from .reporting import (build_daily_recap,
                                    build_weekly_recap)
            from datetime import timedelta as _td
            self._send_alert(build_daily_recap(self.journal, prev))
            if day.weekday() == 0:  # Monday: the week just ended
                self._send_alert(build_weekly_recap(
                    self.journal, prev - _td(days=6)))
        except Exception as exc:
            self.journal.log_note(
                f"mt5: rollover recap failed: {exc}")

    # --------------------------------------------------------------- bars
    def forecast_line(self) -> str:
        """One /status Forecast block from real runtime values (never
        hard-coded): readiness, version, training time, coverage,
        validation, last evaluation, last error, alert state."""
        try:
            frame = self._stack.frames.get("H1") \
                if self._stack is not None else None
            st = self._forecaster.status_dict(
                h1_bars=len(frame) if frame is not None else None)
        except Exception as exc:  # noqa: BLE001 — status must not die
            return f"forecast: status unavailable ({exc})"
        if not st["ready"]:
            lines = [f"🔮 forecast: NOT READY — {st['reason']}"]
            if st["training_running"]:
                lines.append("  training: running in the background")
            elif st["last_train_result"] is not None:
                ok, note = st["last_train_result"]
                lines.append(
                    f"  last training: {'ok' if ok else 'FAILED'} — {note}")
            if st["last_error"]:
                lines.append(f"  last error: {st['last_error']}")
            lines.append("  alerts: idle — no valid bundle")
            return "\n".join(lines)
        lines = ["🔮 forecast: READY — alerts live on 1d/3d crossings"]
        lines.append(
            f"  model: {st['version']} ({st['feature_set']}) · "
            f"{st['symbol']} · trained {st['trained_at']} through "
            f"{st['trained_through']} ({st['train_bars']} usable MTF rows)")
        h1 = st["h1_bars"]
        lines.append(
            f"  coverage: {h1} H1 bars in stack"
            if h1 is not None else "  coverage: H1 stack unavailable")
        val = st["validation"]
        if isinstance(val, dict) and "error" not in val:
            parts = []
            for h in (24, 72, 168, 360):
                entry = val.get(h, {})
                au, ad = entry.get("auc_up"), entry.get("auc_dn")
                label = {24: "1d", 72: "3d", 168: "7d", 360: "15d"}[h]
                if au is None and ad is None:
                    parts.append(f"{label} n/a")
                else:
                    parts.append(f"{label} up {au} / dn {ad}")
            lines.append(
                "  validation (chronological holdout AUC): "
                + " · ".join(parts))
        elif isinstance(val, dict):
            lines.append(f"  validation: unavailable — {val['error']}")
        else:
            lines.append("  validation: none recorded (old bundle)")
        last = st["last_evaluation"]
        if last is not None:
            r = last["ranks"]
            parts = []
            for h in (24, 72):
                if h in r:
                    parts.append(
                        f"{ {24: '1d', 72: '3d'}[h] } "
                        f"up {r[h]['up']:.0f} / dn {r[h]['dn']:.0f}")
            lines.append(
                f"  last evaluation: {last['bar_time']} — "
                + (" · ".join(parts) if parts else "no gating ranks"))
        else:
            lines.append("  last evaluation: none yet")
        if st["training_running"]:
            lines.append("  training: running in the background")
        if st["last_error"]:
            lines.append(f"  last error: {st['last_error']}")
        return "\n".join(lines)

    def forecast_train_now(self) -> str:
        """Operator-triggered retrain (/forecast_train): starts
        background training when the bundle needs it, else reports
        the truthful state. Never blocks the tick loop. Auto-bootstrap
        remains the primary path — this command is the operator's
        explicit lever, not the only way a fresh install gets a model."""
        try:
            frames = self._stack.frames if self._stack is not None else None
            h1 = frames.get("H1") if isinstance(frames, dict) else None
            if h1 is None or len(h1) == 0:
                return ("forecast_train: no H1 frame available — "
                        "engine not warm-started?")
            _thread, note = self._forecaster.maybe_train(frames)
            return f"forecast_train: {note}"
        except Exception as exc:  # noqa: BLE001 — command honesty
            return f"forecast_train failed: {exc}"

    def _maybe_forecast_alert(self) -> None:
        """Score the ensemble on a newly closed D1 bar; alert a new
        threshold crossing. Self-training rides along: a finished
        background train journals here (every tick, so the outcome
        surfaces within a tick of finishing), and a stale bundle
        retrains on the D1 close.

        Runs at most one scoring per closed D1 bar (cursor change
        detection). The stack's frames are scored — M5/M15/H1/H4/D1
        closed bars as-of aligned (see forecast/mtf.py) — but the
        D1 close sets the cadence: a 1-day+ forecast re-alerting
        every hour while its rank sits high would be spam, not
        signal. Any failure is journaled and never raised into the
        tick loop — a forecast problem must never stop evaluation,
        management, or orders.
        """
        try:
            # Finished-train outcome: journaled on the tick thread
            # (the worker thread only records it, never journals).
            result = self._forecaster._train_result
            if result is not None:
                self._forecaster._train_result = None
                ok, note = result
                self.journal.log_note(
                    f"forecast training {'complete' if ok else 'FAILED'}: "
                    f"{note}")
            d1 = self._stack.frames.get("D1")
            if d1 is None or len(d1) == 0:
                if not self._forecast_no_d1_noted:
                    self._forecast_no_d1_noted = True
                    self.journal.log_note(
                        "forecast: no D1 frame — forecast alerts idle")
                return
            day = d1.index[-1]
            if day == self._forecast_last_scored_bar:
                return
            self._forecast_last_scored_bar = day
            frames = self._stack.frames
            if not frames.get("H1") or len(frames["H1"]) < 50:
                return
            thread, note = self._forecaster.maybe_train(frames)
            if thread is not None:
                self.journal.log_note(f"forecast: {note}")
            text, note = self._forecaster.maybe_alert(frames)
            if text is not None:
                self.journal.log_note(
                    f"forecast alert @ {day}: {note}")
                self._send_alert(text)
        except Exception as exc:  # noqa: BLE001 — tick-loop guard
            self.journal.log_note(
                f"forecast alert failed (loop continues): {exc}")

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
        # Refresh the separate M1 trigger stream on its own minute clock.
        # It is deliberately not inserted into TimeframeStack: that stack is
        # reserved for hierarchy/context frames. A missed/failed M1 refresh
        # simply makes Hot Scalp fail closed on freshness; it cannot disturb
        # the main strategy clock.
        if self._hot_scalp is not None:
            m1_close = MT5Feed.closed_bar_close("M1", tick_time)
            if (self._hot_scalp_last_processed_m1 is None
                    or m1_close > self._hot_scalp_last_processed_m1):
                try:
                    _m1_current = self._hot_scalp_aux_frames.get("M1")
                    _m1_cursor = (_m1_current.index[-1]
                                  if _m1_current is not None and len(_m1_current)
                                  else None)
                    _m1_new = self.feed.fetch_new_all(
                        {"M1": _m1_cursor}, tfs=("M1",)).get("M1")
                    if _m1_new is not None and len(_m1_new):
                        _m1_merged = pd.concat([_m1_current, _m1_new]) \
                            if _m1_current is not None and len(_m1_current) \
                            else _m1_new
                        _m1_merged = _m1_merged.sort_index()
                        _m1_merged = _m1_merged[~_m1_merged.index.duplicated(keep="last")]
                        self._hot_scalp_aux_frames["M1"] = _m1_merged.tail(3000)
                        self._hot_scalp_last_processed_m1 = (
                            _m1_merged.index[-1] + pd.Timedelta(minutes=1))
                    self._hot_scalp_m1_refresh_error_key = None
                except Exception as _m1_exc:
                    _error_key = f"{m1_close}:{type(_m1_exc).__name__}:{_m1_exc}"
                    if _error_key != self._hot_scalp_m1_refresh_error_key:
                        _log.warning("hot_scalp M1 refresh failed; shadow scan will fail closed: %s", _m1_exc)
                        self._hot_scalp_m1_refresh_error_key = _error_key
        if not closed_tfs:
            return
        # Logged (info): this fetch is the live engine's normal
        # refresh path, NOT a second warm start.  fetch_new_all only
        # fetches bars since the cursor and appends nothing new when
        # the tick was stale — a routine refresh, not a re-fetch of
        # history.  (Before this log, a refresh's W1/MN1 short-history
        # warnings looked like a duplicate warm start; those
        # warnings are now warm-start-only via warn_short=False.)
        _log.info("mt5 refresh: TFs closed: %s — fetching new bars",
                  closed_tfs)
        try:
            fresh = self.feed.fetch_new_all(cursors, tfs=tuple(closed_tfs))
        except TypeError as _fetch_type_error:
            # Legacy test doubles / externally supplied feeds may still expose
            # the pre-filter signature; keep that adapter explicit.
            if "tfs" not in str(_fetch_type_error) and "keyword" not in str(_fetch_type_error):
                raise
            fresh = self.feed.fetch_new_all(cursors)
        # Phase 2: append every newly closed bar; refresh legs.
        appended: list[str] = []
        appended_counts: dict[str, int] = {}
        h1_accepted = 0
        for tf in closed_tfs:
            new = fresh.get(tf)
            if new is None or not len(new):
                continue
            accepted, _rej = self._stack.append(tf, new)
            if not accepted:
                continue
            appended.append(tf)
            appended_counts[tf] = accepted
            self._last_processed[tf] = (
                self._stack.cursors[tf]
                + pd.Timedelta(minutes=TF_MINUTES[tf]))
            if tf == "M5":
                # Native M5 route (funnel §2): qm_m5 evaluates on
                # the REAL M5 frame at every M5 close and can
                # reach the shared order path.  Also refreshes the
                # annotation state (_qm_touch) from the same
                # evaluation — the annotation-only _advance_qm
                # scanner is gone.
                self._run_qm_m5(self._stack.frames.get("M5"),
                                stats)
            if tf == "D1":
                self._run_qm_d1(
                    self._stack.frames.get("D1"), stats)
            if tf == "H1":
                h1_accepted += accepted
        # MacroTrend alignment refreshes ONLY on D1/W1/MN1 closes
        # (EMA200 bias moves only when a macro bar closes).  It never
        # refreshes hierarchy legs — alignment is sizing +
        # confirmation, never a gate leg.
        if any(tf in ("D1", "W1", "MN1") for tf in appended):
            self._refresh_alignment()
        # DXY correlation refreshes on H1 closes (its window is H1
        # bars).  Confluence context only — never a hierarchy leg.
        if "H1" in appended:
            self._refresh_dxy()
        if any(tf in ("D1", "H4", "H1") for tf in appended):
            self._refresh_live_setup()
        # Phase 2 (P0-3): scheduled evaluations on their own
        # clocks, independent of the generic H1 trigger — an H1
        # close evaluates the H1 roster + range refinement; an
        # H4 close evaluates range scenario generation.  (The
        # native M5/D1 routes fired inside the append loop
        # above, on their own closes, with the default
        # live_close source.)
        if "H1" in appended:
            self._run_roster_close_evals(stats,
                                         source="live_close")
            self._run_range_family("h1_refinement", stats,
                                   source="live_close")
        if "H4" in appended:
            self._run_range_family("h4_generation", stats,
                                   source="live_close")
        # Phase 3: entry-TF closes — the Close Rule FIRST (a bar
        # closing beyond the invalidation level kills the setup
        # instantly, before any confirmation on the same close),
        # then the confirm shot, then the armed TTL.  The entry
        # bar that closes with the H1 bar still belongs to that
        # bar — confirmation is evaluated BEFORE any H1-close
        # expiry.
        self._check_close_rule(tick, appended, appended_counts, stats)
        if self._pending is not None and \
                self._pending["entry_clock"] in appended:
            pend = self._pending
            # Confirm shot FIRST: a confirming close fires, and
            # confirm always wins over TTL on the same bar.
            self._evaluate_pending_entry(tick, stats)
            # Armed TTL (owner rule, 2026-10-15): if still armed,
            # these entry bars closed without confirmation.
            # At armed_ttl_bars the touch is stale (price lingered
            # without reacting: absorbed, edge decayed) and disarms
            # as ``stale``, distinct from exit/invalidation.
            # Re-arm needs exit + re-entry (see _stale_zone).
            if self._pending is pend and pend.get("armed"):
                pend["armed_bars"] = pend.get("armed_bars", 0) + \
                    appended_counts.get(pend["entry_clock"], 0)
                if pend["armed_bars"] >= self.cfg.armed_ttl_bars:
                    self._disarm_stale(stats)
        # XD-4: management-clock inheritance — an M5/D1 close
        # runs management for positions on that clock NOW, so
        # the H1-close pass below (gated by new-bar-since-last
        # -pass) does not double-check them.
        _mgmt_clocks = tuple(tf for tf in ("M5", "D1")
                             if tf in appended)
        if _mgmt_clocks:
            try:
                self._manage_positions_bar_close(
                    stats, clocks=_mgmt_clocks)
            except Exception as exc:
                _log.warning("mt5 bar-close management (M5/D1) "
                             "failed: %s", exc)
        # Phase 4: H1 close — time stop, pending expiry, forming reset.
        if "H1" in appended:
            stats["bars"] = stats.get("bars", 0) + h1_accepted
            self._on_h1_close(tick, stats)
            # Post-rebuild observation / stability clock: one clean
            # H1 close of observation or stability counting.
            self._advance_regime_pause()
        # Phase 5: D1/H4 close flipped the bias — the pending
        # trigger's premise is gone (only if still pending).
        if any(tf in ("D1", "H4") for tf in appended):
            self._check_pending_bias(stats)

    # ------------------------------------------- MacroTrend alignment
    def _refresh_alignment(self) -> None:
        """Recompute D1/W1/MN1 alignment from the stack's closed bars.

        Journals only when the alignment label changes (a macro-bar
        close that flips bias) or at warm start — not on every tick.
        Alignment is sizing + confirmation only; it never touches
        hierarchy legs.
        """
        if self._stack is None:
            return
        new = compute_macrotrend(self._stack.frames)
        old = self._alignment
        self._alignment = new
        if new.alignment_label != old.alignment_label \
                or new.conflict != old.conflict:
            self.journal.log_note(
                f"mt5: macro alignment -> {new.describe()}")

    def _refresh_health_market(self) -> None:
        """Runtime Truth item 1: feed the health monitor the
        broker's market state, from the SAME check the order path
        gates on (:meth:`_market_open`) — one market truth for
        execution and for status.  Cached 30s (the check is a
        terminal call).  A feed whose market state cannot be read
        leaves the state UNKNOWN (None): tick staleness then
        keeps its fail-loud breach reading rather than being
        excused by a guess."""
        now = time.monotonic()
        if now - self._health_market_ts < 30.0 \
                and self._health.market_open_state is not None:
            return
        self._health_market_ts = now
        fn = getattr(self.feed, "market_open", None) \
            if self.feed is not None else None
        if fn is None:
            self._health.note_market_open(None)
            return
        try:
            is_open, _reason = fn()
            self._health.note_market_open(bool(is_open))
        except Exception:
            self._health.note_market_open(None)

    def feed_truth(self) -> dict:
        """The reconciled feed picture for /status (Runtime Truth
        item 1): market state + tick state + tick age from the
        health monitor — the Feed line and the Health line are
        rendered from this one source, so they cannot disagree
        (``feed: OK`` beside ``BREACH: tick_stale`` on a closed
        market was the defect)."""
        self._refresh_health_market()
        return {"market_open": self._health.market_open_state,
                "tick_state": self._health.tick_state(),
                "tick_age_s": self._health.tick_age_sec()}

    def _journal_new_breaches(self) -> None:
        """Journal health breaches that were not breaching before —
        a stuck breach gets one note, not one per order."""
        self._refresh_health_market()
        current = self._health.breaches()
        new = current - self._health_last_breaches
        self._health_last_breaches = current
        for key in sorted(new):
            self.journal.log_note(
                f"mt5: health breach: {key} — {self._health.describe()}")

    def health_line(self) -> str:
        """One-line execution health for /status (see mt5_commands)."""
        self._refresh_health_market()
        return self._health.describe()

    def _refresh_dxy(self) -> None:
        """Recompute XAUUSD/DXY correlation from the stack's H1 bars.

        Journals only when availability flips (a DXY outage is
        worth one note, not one per H1 close).  The correlation is
        confluence context — it never touches hierarchy legs.
        """
        if self._stack is None:
            return
        state = self._dxy.refresh(self._stack.frames.get("H1"))
        if self._dxy_last_available is None \
                or state.available != self._dxy_last_available:
            self.journal.log_note(f"mt5: DXY -> {state.describe()}")
        self._dxy_last_available = state.available

    def dxy_line(self) -> str:
        """One-line DXY state for /status (see mt5_commands)."""
        return self._dxy.state.describe()

    def _refresh_live_setup(self) -> None:
        """Recompute the cached trigger levels + the reason string.

        Single place both call sites (warm start, TF-close refresh)
        go through, so ``_live_setup`` and ``_live_setup_reason``
        can never disagree.
        """
        if self._stack is None:
            self._live_setup = None
            self._live_setup_reason = "no stack"
            self._live_hold_state = None
            return
        self._live_setup = live_setup(self._stack)
        self._live_setup_reason = live_setup_reason(self._stack)
        # Continuation one-arm-per-break: a break that already
        # fired is spent — the stateless setup re-arms it on every
        # refresh while price stays beyond the broken level.
        # Suppress it (with the honest reason for /status); the
        # tick clock checks the spent set too, as a backstop for
        # ticks evaluated before this refresh.
        if self._live_setup is not None \
                and self._live_setup.get("arm") == "continuation" \
                and str(self._live_setup.get("break_key") or "") \
                in self._continuation_spent:
            self._live_setup_reason = (
                "continuation_spent: break of "
                f"{float(self._live_setup['ref_level']):.1f} "
                "already fired (one arm per break)")
            self._live_setup = None
        # HOLD state: computed on every refresh (cheap, closed bars
        # only) so /status + logs always describe the CURRENT flat
        # state, not a stale one.  Logged at INFO as HOLD when the
        # reason is d1_flat — never ERROR, never FEED_ERROR.
        self._live_hold_state = live_setup_hold_state(self._stack)
        if self._live_setup_reason.startswith("d1_flat") \
                and self._live_hold_state is not None:
            line = self._live_hold_state.describe()
            if line != self._last_hold_logged:
                self._last_hold_logged = line
                log_hold_state(_log, self._live_hold_state)
        # Boundary machine + persisted hysteresis clock (runs on
        # every TF-close refresh, before any order decision on the
        # new bars — see _update_boundary).
        self._update_boundary()
        # Multi-day structural zones (historical WHERE).
        self._update_zones()
        # P3 scenario state: bedrock + day + posture + SMT (the
        # decision flow's context — refreshed BEFORE any order
        # decision on the new bars, like the boundary machine).
        self._refresh_scenario()

    # --------------------------------------- P3 scenario refresh
    def _refresh_scenario(self) -> None:
        """Recompute bedrock / day / posture / SMT from the stack.

        Bedrock (live/regime.py five-axis) is the ONE regime
        brain of the decision flow (S1): its classification is
        computed here, on the H1 frame classify_regime is defined
        on, and composed with the day object + session + loss
        ladder into the posture every decision journals.  Failures
        degrade to the previous state with a warning — a context
        refresh must never kill the trading loop.
        """
        try:
            if self._stack is None:
                return
            frames = self._stack.frames
            h1 = frames.get("H1")
            if h1 is None or not len(h1):
                return
            self._bedrock = classify_regime(h1)
            prev_hist = self._day_ctx.day_type_history \
                if self._day_ctx is not None else None
            self._day_ctx = _day_context.compute_day_context(
                frames, prior_history=prev_hist)
            self._refresh_ny_checkpoint(frames)
            self._refresh_smt(frames)
            self._posture = self._compose_posture()
        except Exception as exc:
            _log.warning("mt5 scenario refresh failed (previous "
                         "context kept): %s", exc)

    def _refresh_ny_checkpoint(self, frames) -> None:
        """Evaluate the NY acceptance checkpoint once per day, on
        the first COMPLETED NY-session H1 bar (S3)."""
        d = self._day_ctx
        if d is None or self._ny_checkpoint_day == d.day_key:
            return
        h1 = frames.get("H1")
        if h1 is None or not len(h1):
            return
        days = _day_context.group_days(h1)
        cur = days.get(max(days.keys())) if days else None
        if cur is None or not len(cur):
            return
        ny_bars = [(ts, r) for ts, r in cur.iterrows()
                   if session_of(ts)[0] in ("newyork", "overlap")]
        if len(ny_bars) < 1:
            return
        # The first NY bar must be COMPLETE: it is not the last
        # bar of the frame.
        first_ts, first = ny_bars[0]
        if first_ts >= h1.index[-1]:
            return
        verdict = _posture_mod.ny_checkpoint(
            d, ny_close=float(first["close"]),
            ny_low=float(first["low"]), ny_high=float(first["high"]))
        self._ny_checkpoint = verdict
        self._ny_checkpoint_day = d.day_key
        if verdict is not None:
            self.journal.log_note(
                f"mt5 NY checkpoint: London leg {verdict} — "
                "acceptance read at the NY open (S3); posture "
                "follows the read")

    def _refresh_smt(self, frames) -> None:
        """SMT read from the DXY tracker's already-fetched H1
        frame (S7).  Unavailability is journaled ONCE per reason
        — a broken correlation link is a fact in the record,
        never a silent zero."""
        tracker = getattr(self, "_dxy", None)
        gold = frames.get("H1")
        dxy_frame = None
        corr = None
        try:
            if tracker is not None:
                dxy_frame = getattr(tracker, "frame", None)
                st = getattr(tracker, "state", None)
                # Only a LIVE correlation read may arm SMT; a
                # stale one reads as unavailable (never zero).
                if st is not None and st.available:
                    corr = st.corr
        except Exception:
            dxy_frame, corr = None, None
        # Phase 2 census fix: align the frames on bar timestamps
        # before the event read — detect_smt indexes the DXY
        # arrays positionally, so unaligned frames (a DXY fetch
        # with a missing/extra bar) would compare DIFFERENT bars
        # as if they were the same moment.  Unalignable frames
        # read as SMT-unavailable, never as a fabricated event.
        if gold is not None and dxy_frame is not None:
            from ..live.smt import SmtRead, align_on_timestamps
            gold, dxy_frame = align_on_timestamps(gold, dxy_frame)
            if gold is None:
                read = SmtRead(
                    available=False, corr=corr,
                    reason="SMT unavailable: gold/DXY frames "
                           "could not be aligned on timestamps")
                self._smt_read = read
                if read.reason != self._smt_unavailable_key:
                    self._smt_unavailable_key = read.reason
                    self.journal.log_note(f"mt5 SMT: {read.reason}")
                return
        read = _detect_smt(gold, dxy_frame, corr=corr)
        self._smt_read = read
        if not read.available:
            key = read.reason
            if key != self._smt_unavailable_key:
                self._smt_unavailable_key = key
                self.journal.log_note(f"mt5 SMT: {read.reason}")
        else:
            self._smt_unavailable_key = ""
            if read.event is not None:
                self.journal.log_note(
                    f"mt5 SMT event: {read.event.kind} — "
                    f"{read.event.detail}")

    def _compose_posture(self):
        """Compose the live posture from bedrock + day + session
        + ladder + news state (posture.compose_posture)."""
        now = None
        if self._last_tick is not None:
            now = self._last_tick.get("time")
        if now is None and self._stack is not None:
            h1 = self._stack.frames.get("H1")
            if h1 is not None and len(h1):
                now = h1.index[-1]
        if now is None:
            now = pd.Timestamp.now(tz="UTC")
        sess = _posture_mod.session_posture(
            now, spread_ratio=self._spread_ratio)
        st = {}
        try:
            st = self.guards.state()
        except Exception:
            st = {}
        # Per-style opens reset on the NY trading day.
        day_key = self._day_ctx.day_key if self._day_ctx else ""
        if day_key and day_key != self._style_day:
            self._style_day = day_key
            self._style_opens_today = {}
        ladder = _posture_mod.loss_ladder(
            int(st.get("consecutive_losses", 0) or 0),
            style_counts=dict(self._style_opens_today),
            style_ceiling=int(self.cfg.max_daily_trades_per_style))
        # Killzone cooldown: the ladder's cooldown holds only
        # until the next killzone — if we ARE in one, it has
        # already lifted.
        if ladder.cooldown:
            try:
                from ..live.patterns import session_label
                if session_label(now).get("in_killzone"):
                    ladder = _posture_mod.LadderState(
                        consecutive_losses=ladder.consecutive_losses,
                        layer_cap_delta=ladder.layer_cap_delta,
                        traffic_cap=ladder.traffic_cap,
                        cooldown=False,
                        style_blocked=ladder.style_blocked)
            except Exception:
                pass
        news_block = False
        if self.cfg.news_pause_enabled:
            try:
                news_block = bool(
                    self.news.execution_state(
                        self._tick_datetime()).paused)
            except Exception:
                news_block = False
        post = _posture_mod.compose_posture(
            bedrock=self._bedrock, day=self._day_ctx, session=sess,
            ladder=ladder, news_block=news_block,
            checkpoint=self._ny_checkpoint)
        if not post.block_entries:
            post.size_scalar = float(post.size_scalar) \
                * float(sess.size_scalar)
        return post

    def _tick_datetime(self):
        """Aware datetime of the last tick (news clock input)."""
        t = (self._last_tick or {}).get("time")
        if t is not None:
            ts = pd.Timestamp(t)
            return ts.to_pydatetime() if ts.tzinfo is not None \
                else ts.tz_localize("UTC").to_pydatetime()
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)

    def _build_shared_signal_evidence(self, *, direction: int, playbook: str,
                                      entry: float | None, bar_ts=None,
                                      context: dict | None = None) -> dict:
        """Build the common evidence snapshot for any emitted MT5 signal.

        This is audit/research metadata only. Missing measurements are neutral,
        correlated evidence shares one contribution budget, and the result is
        never consumed by the execution path. Main playbooks remain governed by
        their current decision/risk logic until this measure is calibrated on
        chronological, cost-aware replay.
        """
        try:
            direction = 1 if int(direction) > 0 else -1 if int(direction) < 0 else 0
            if direction == 0:
                return {"status": "unavailable", "reason_code": "direction.invalid",
                        "execution_gate": False}
            stack = getattr(self, "_stack", None)
            frames = dict(getattr(stack, "frames", {}) or {}) if stack is not None else {}
            m5 = frames.get("M5")
            m15 = frames.get("M15")
            aux = getattr(self, "_hot_scalp_aux_frames", {}) or {}
            m1 = aux.get("M1")
            # This is an observational layer called on signal emissions, not a
            # history backfill. Bound every detector to recent causal windows so
            # a large MT5 cache cannot stall the tick-driven execution loop.
            if m5 is not None and len(m5) > 600:
                m5 = m5.tail(600)
            if m15 is not None and len(m15) > 600:
                m15 = m15.tail(600)
            if m1 is not None and len(m1) > 300:
                m1 = m1.tail(300)
            if m5 is None or not len(m5) or m15 is None or not len(m15):
                return {"status": "unavailable", "reason_code": "data.stack_not_ready",
                        "playbook": str(playbook), "execution_gate": False,
                        "data_timestamps": {tf: str(df.index[-1])
                                             for tf, df in (("M5", m5), ("M15", m15))
                                             if df is not None and len(df)}}

            def _utc(value):
                try:
                    ts = pd.Timestamp(value)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    else:
                        ts = ts.tz_convert("UTC")
                    return ts
                except Exception:
                    return None

            tick = getattr(self, "_last_tick", None) or {}
            now = _utc(tick.get("time") or bar_ts or m5.index[-1])
            if now is None:
                return {"status": "unavailable", "reason_code": "clock.invalid",
                        "playbook": str(playbook), "execution_gate": False}
            observed_price = None
            try:
                if entry is not None and math.isfinite(float(entry)) and float(entry) > 0:
                    observed_price = float(entry)
            except (TypeError, ValueError):
                pass
            if observed_price is None:
                try:
                    bid, ask = float(tick.get("bid")), float(tick.get("ask"))
                    observed_price = (bid + ask) / 2 if bid > 0 and ask >= bid else None
                except (TypeError, ValueError):
                    observed_price = None
            if observed_price is None:
                observed_price = float(m5["close"].iloc[-1])

            items: list[EvidenceItem] = []
            timestamps = {"M5": _utc(m5.index[-1]), "M15": _utc(m15.index[-1])}
            if m1 is not None and len(m1):
                timestamps["M1"] = _utc(m1.index[-1])

            def _item(key, label, source, observed_at, timeframe, category,
                      group, stance="neutral", strength=0.0, available=True,
                      detail=""):
                ts = _utc(observed_at)
                items.append(EvidenceItem(
                    key=str(key), label=str(label), source=str(source),
                    observed_at=ts.isoformat() if ts is not None else "unknown",
                    timeframe=str(timeframe), category=str(category),
                    correlation_group=str(group), stance=stance,
                    strength=float(strength), available=bool(available),
                    detail=str(detail)))

            # M15 context bias.
            m15_ctx = _ev_trend_bias(m15)
            cdir = int(m15_ctx.get("bias", 0) or 0)
            _item(f"m15_bias:{timestamps['M15']}:{direction}",
                  "M15 swing-structure context", "patterns.trend_bias",
                  m15.index[-1], "M15", "higher_context",
                  f"m15-context:{timestamps['M15']}",
                  "supports" if cdir == direction else
                  "contradicts" if cdir == -direction else "neutral",
                  1.0 if cdir else 0.0, detail=m15_ctx.get("detail", "bias unavailable"))

            # M5 swing bias and last-bar event are intentionally one correlated
            # evidence group: two labels derived from the same local structure
            # must not count as two independent confirmations.
            m5_ctx = _ev_trend_bias(m5)
            bdir = int(m5_ctx.get("bias", 0) or 0)
            group_m5 = f"m5-structure:{timestamps['M5']}"
            _item(f"m5_bias:{timestamps['M5']}:{direction}",
                  "M5 swing structure", "patterns.trend_bias", m5.index[-1],
                  "M5", "structure", group_m5,
                  "supports" if bdir == direction else
                  "contradicts" if bdir == -direction else "neutral",
                  0.8 if bdir else 0.0, detail=m5_ctx.get("detail", "bias unavailable"))
            try:
                event = _ev_structure_event(m5, trend=cdir)
            except Exception as exc:
                event = {"present": False, "detail": f"detector error: {type(exc).__name__}"}
            edir = int(event.get("direction", 0) or 0)
            _item(f"m5_event:{timestamps['M5']}:{event.get('event', 'none')}:{direction}",
                  "M5 BOS/CHoCH/MSB event", "patterns.structure_event",
                  m5.index[-1], "M5", "structure", group_m5,
                  "supports" if event.get("present") and edir == direction else
                  "contradicts" if event.get("present") and edir == -direction else "neutral",
                  min(1.0, float(event.get("score", 0.0) or 0.0) / 75.0),
                  detail=event.get("detail", "no confirmed event"))

            # Nearest unfilled FVG around the actual candidate price. If it is
            # part of the latest M5 impulse, it shares that event's correlation
            # group rather than increasing the score independently.
            try:
                fvgs = _ev_unfilled_fvgs(m5, lookback=120)
                near = [f for f in fvgs if float(f["bottom"]) <= observed_price + 0.5 * max(_safe_atr(m5), 0.0)
                        and float(f["top"]) >= observed_price - 0.5 * max(_safe_atr(m5), 0.0)]
                if near:
                    fvg = min(near, key=lambda f: abs(float(f.get("ce", (float(f["top"])+float(f["bottom"]))/2)) - observed_price))
                    fdir = int(fvg.get("direction", 0) or 0)
                    fts = _utc(fvg.get("ts"))
                    evts = _utc(m5.index[-1])
                    same_impulse = fts is not None and evts is not None and abs((evts - fts).total_seconds()) <= 2 * 5 * 60
                    fgroup = f"m5-structure:{timestamps['M5']}" if same_impulse else f"fvg:{fts.isoformat() if fts is not None else 'unknown'}"
                    _item(f"fvg:{fts}:{fdir}", "nearby unfilled fair-value gap",
                          "patterns.unfilled_fvgs", fvg.get("ts"), "M5", "imbalance", fgroup,
                          "supports" if fdir == direction else
                          "contradicts" if fdir == -direction else "neutral",
                          0.7, detail=f"FVG {fvg.get('bottom')}–{fvg.get('top')}; filled={fvg.get('filled')}")
            except Exception:
                # Unknown FVG schema is neutral; don't manufacture support.
                pass

            # Liquidity is a conditional event, not an automatic vote. First find
            # a measured nearby pool, then require wick-through + close-back-inside
            # from the detector. The sweep direction is the trade direction AFTER
            # the sweep, and shares its reaction group with a candle pattern on the
            # exact same bar so one rejection event isn't counted twice.
            try:
                atr5 = _safe_atr(m5)
                pools = _ev_liquidity_pools(m5, lookback=240)
                nearby_pools = [p for p in pools
                                if abs(float(p.get("level", 0.0)) - observed_price)
                                <= max(0.0, atr5 * 0.75)] if atr5 > 0 else []
                if nearby_pools:
                    pool = min(nearby_pools,
                               key=lambda p: abs(float(p["level"]) - observed_price))
                    sweep = _ev_detect_sweep(m5, float(pool["level"]),
                                             direction=direction, atr=atr5)
                    if sweep.get("present"):
                        sweep_ts = _utc(sweep.get("ts")) or timestamps["M5"]
                        _item(f"liquidity-sweep:{sweep_ts}:{direction}",
                              "confirmed liquidity sweep and close-back-inside",
                              "patterns.liquidity_pools + patterns.detect_sweep",
                              sweep_ts, "M5", "reaction",
                              f"m5-reaction:{sweep_ts.isoformat() if sweep_ts is not None else 'unknown'}",
                              "supports", min(1.0, float(sweep.get("score", 70.0)) / 100.0),
                              detail=str(sweep.get("detail", "confirmed sweep"))
                              + f"; pool={pool.get('side')} level={pool.get('level')}")
            except Exception:
                # A missing pool/sweep detector is not a negative signal or support.
                pass

            # Recent named candle patterns can explain the final reaction. Only
            # the freshest two closed bars are relevant; age/meaning are recorded,
            # and neutral dojis never become directional support.
            try:
                for candle in _ev_named_candles(m5, window=3):
                    if int(candle.get("bars_ago", 999)) > 1:
                        continue
                    candle_dir = int(candle.get("direction", 0) or 0)
                    candle_ts = _utc(candle.get("bar_ts")) or timestamps["M5"]
                    c_stance = ("supports" if candle_dir == direction else
                                "contradicts" if candle_dir == -direction else "neutral")
                    _item(f"candle:{candle.get('name')}:{candle_ts}:{direction}",
                          f"recent candle pattern: {candle.get('name')}",
                          "live.candle_taxonomy.detect_named_patterns", candle_ts,
                          "M5", "reaction",
                          f"m5-reaction:{candle_ts.isoformat() if candle_ts is not None else 'unknown'}",
                          c_stance, 0.55 if c_stance != "neutral" else 0.0,
                          detail=f"bars_ago={candle.get('bars_ago')}; direction={candle_dir}")
            except Exception:
                pass

            # A nearby unmitigated order block is a location read, not a standalone
            # trigger. It shares the location category's cap with premium/discount.
            try:
                atr5 = _safe_atr(m5)
                blocks = [b for b in _ev_order_blocks(m5, lookback=60)
                          if not bool(b.get("mitigated"))
                          and float(b.get("bottom", 0.0)) - 0.25 * atr5 <= observed_price
                          <= float(b.get("top", 0.0)) + 0.25 * atr5] if atr5 > 0 else []
                if blocks:
                    block = min(blocks, key=lambda b: abs(
                        float(b.get("level", (float(b["top"])+float(b["bottom"]))/2)) - observed_price))
                    block_dir = int(block.get("direction", 0) or 0)
                    block_ts = _utc(block.get("ts"))
                    _item(f"order-block:{block_ts}:{block_dir}:{direction}",
                          "nearby unmitigated order block", "patterns.order_blocks",
                          block_ts, "M5", "location",
                          f"order-block:{block_ts.isoformat() if block_ts is not None else 'unknown'}",
                          "supports" if block_dir == direction else
                          "contradicts" if block_dir == -direction else "neutral",
                          0.65, detail=f"OB {block.get('bottom')}–{block.get('top')}; kind={block.get('kind')}")
            except Exception:
                pass

            # Premium/discount only scores when the detector supplies a schema
            # we understand. Unknown output remains explicitly neutral.
            try:
                loc = _ev_premium_discount(m15)
                zone = str(loc.get("zone", loc.get("position", loc.get("location", "")))).lower()
                loc_stance = "neutral"
                if zone in ("discount", "below_eq", "below equilibrium"):
                    loc_stance = "supports" if direction > 0 else "contradicts"
                elif zone in ("premium", "above_eq", "above equilibrium"):
                    loc_stance = "supports" if direction < 0 else "contradicts"
                _item(f"location:{timestamps['M15']}:{direction}",
                      "M15 premium/discount location", "patterns.premium_discount",
                      m15.index[-1], "M15", "location", f"range-location:{timestamps['M15']}",
                      loc_stance, 0.7 if loc_stance != "neutral" else 0.0,
                      detail=loc.get("detail", f"location={zone or 'unknown'}"))
            except Exception:
                pass

            # Tick volume is directional; a surge opposite the proposed trade
            # contradicts it instead of automatically voting in favour.
            try:
                volume_frame = m5
                if "volume" not in volume_frame.columns and "tick_volume" in volume_frame.columns:
                    volume_frame = volume_frame.copy()
                    volume_frame["volume"] = volume_frame["tick_volume"]
                vol = _ev_volume_confirm(volume_frame)
                if vol.get("present"):
                    last_dir = (1 if float(m5["close"].iloc[-1]) > float(m5["open"].iloc[-1])
                                else -1 if float(m5["close"].iloc[-1]) < float(m5["open"].iloc[-1]) else 0)
                    stance = ("supports" if last_dir == direction else
                              "contradicts" if last_dir == -direction else "neutral")
                    _item(f"volume:{timestamps['M5']}", "direction-aware M5 volume expansion",
                          "patterns.volume_confirm + M5 candle direction", m5.index[-1], "M5",
                          "activity", f"volume:{timestamps['M5']}", stance,
                          min(1.0, float(vol.get("ratio", 1.5) or 1.5) / 2.0),
                          detail=str(vol.get("detail", "volume expansion")) + f"; candle_direction={last_dir}")
            except Exception:
                pass

            # EMA 9/21 is a small momentum observation, never a trigger or gate.
            try:
                if m1 is not None and len(m1) >= 22:
                    closes = m1["close"].astype(float)
                    fast = closes.ewm(span=9, adjust=False).mean().iloc[-1]
                    slow = closes.ewm(span=21, adjust=False).mean().iloc[-1]
                    gap = float(fast - slow)
                    atr = _safe_atr(m1)
                    normalized = abs(gap) / atr if atr > 0 else 0.0
                    stance = ("neutral" if normalized < 0.05 else
                              "supports" if direction * gap > 0 else "contradicts")
                    _item(f"ema9x21:{timestamps.get('M1')}:{direction}",
                          "M1 EMA(9)/EMA(21) alignment", "pandas.ewm EMA(9)/EMA(21)",
                          m1.index[-1], "M1", "momentum", f"m1-momentum:{timestamps.get('M1')}",
                          stance, min(1.0, normalized / 0.25),
                          detail=f"EMA gap={gap:.6f}, normalized gap={normalized:.3f}")
                else:
                    _item(f"ema9x21:missing:{direction}", "M1 EMA momentum unavailable",
                          "pandas.ewm EMA(9)/EMA(21)", now, "M1", "momentum", "m1-momentum:missing",
                          available=False, detail="M1 auxiliary stream not ready")
            except Exception:
                _item(f"ema9x21:error:{direction}", "M1 EMA momentum unavailable",
                      "pandas.ewm EMA(9)/EMA(21)", now, "M1", "momentum", "m1-momentum:missing",
                      available=False, detail="M1 EMA measurement failed")

            # Record the true wall/session state even outside a killzone. Only a
            # named active window is supportive; ordinary time is neutral.
            session = _ev_session_label(now)
            in_window = bool(session.get("in_killzone"))
            _item(f"session:{now.date()}:{session.get('session')}:{session.get('killzone')}",
                  "session / killzone context", "patterns.session_label", now,
                  "clock", "timing", f"session-window:{now.date()}:{session.get('session')}:{session.get('killzone')}",
                  "supports" if in_window else "neutral", 0.5 if in_window else 0.0,
                  detail=str(session.get("detail", "session unavailable")))

            # Transaction costs are context on existing paths (their own friction
            # gates remain authoritative). Missing quotes mean unavailable.
            try:
                bid, ask = float(tick.get("bid")), float(tick.get("ask"))
                spread = ask - bid if bid > 0 and ask >= bid else None
            except (TypeError, ValueError):
                spread = None
            atr1 = _safe_atr(m1) if m1 is not None and len(m1) else 0.0
            if spread is not None and atr1 > 0:
                ratio = spread / atr1
                stance = "supports" if ratio <= 0.20 else "contradicts"
                strength = min(1.0, max(0.0, 1.0 - ratio / 0.20))
                _item(f"spread:{now.floor('min')}", "live spread relative to M1 ATR",
                      "tick.bid/ask + live M1 ATR", now, "tick/M1", "execution_cost",
                      f"spread-window:{now.floor('min')}", stance, strength,
                      detail=f"spread/ATR={ratio:.4f}")
            else:
                _item(f"spread:missing:{now.floor('min')}", "live spread/ATR unavailable",
                      "tick.bid/ask + live M1 ATR", now, "tick/M1", "execution_cost",
                      f"spread-window:{now.floor('min')}", available=False,
                      detail="quote or M1 ATR unavailable")

            result = score_evidence(items, BASE_CATEGORY_WEIGHTS)
            data_ts = {tf: (ts.isoformat() if ts is not None else None)
                       for tf, ts in timestamps.items()}
            ages = {tf: round(max(0.0, (now - ts).total_seconds()), 3)
                    if ts is not None else None for tf, ts in timestamps.items()}
            return {
                "status": "measured", "playbook": str(playbook),
                "direction": direction, "observed_price": observed_price,
                "observed_at": now.isoformat(), "candidate_bar_ts": str(bar_ts) if bar_ts is not None else None,
                "score": result.score, "score_is_calibrated_probability": False,
                "method": result.method, "execution_gate": False,
                "category_weights": dict(BASE_CATEGORY_WEIGHTS),
                "evidence_score": result.to_dict(),
                "evidence": [item.to_dict() for item in items],
                "data_timestamps": data_ts, "data_age_seconds": ages,
                "session": session, "context": dict(context or {}),
            }
        except Exception as exc:
            return {"status": "unavailable", "reason_code": "shared_evidence.build_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "playbook": str(playbook), "execution_gate": False}

    def _decision_context(self, extra: dict | None = None) -> dict:
        """The J journaling context for one decision: regime
        label + posture, day type, ADR state, session, SMT,
        trigger clock.  Every decision record carries it."""
        d = self._day_ctx
        post = self._posture
        ctx = {
            "regime": (self._bedrock.label()
                       if self._bedrock is not None else None),
            "posture": (post.name if post is not None else None),
            "posture_reason": (post.reason
                               if post is not None else None),
            "day_type": (d.day_type if d is not None else None),
            "adr_pct": (round(d.adr_consumed_pct, 1)
                        if d is not None else None),
            "adr_state": (d.adr_state if d is not None else None),
            "session": (post.session.name
                        if post is not None and post.session
                        else None),
            "ny_checkpoint": self._ny_checkpoint,
            "smt": (self._smt_read.event.kind
                    if self._smt_read is not None
                    and self._smt_read.event is not None else None),
            "trigger_tf": (self._last_clock.trigger_tf
                           if self._last_clock is not None
                           else None),
        }
        if extra:
            ctx.update(extra)
        return ctx

    def _log_management(self, ticket: int, pos, reason: str,
                        note: str) -> None:
        """Journal one position-management event (P3 J): a text
        note AND a structured management record carrying the
        scenario context — the per-playbook demo record is the
        confirmation evidence (A14), management included."""
        self.journal.log_note(note)
        try:
            log_signal(
                self.journal, signal_id=str(pos.alert_id),
                playbook=self._playbook_from_alert_id(pos.alert_id),
                direction=int(pos.direction),
                score=float((self._position_ctx.get(int(ticket))
                             or {}).get("entry_total") or 0.0),
                decision="management", reason=reason,
                context=self._decision_context())
        except Exception:
            pass

    def _runner_partial(self, ticket: int, pos, frac: float,
                        reason: str, stats: dict,
                        price: float) -> bool:
        """Execute + book a runner-side partial (bar-close
        management).  Volume is quantized DOWN to the spec step
        and never below the broker minimum on the remainder —
        a partial that would strand a sub-minimum remainder
        closes the whole position instead (journaled as such)."""
        try:
            spec = self.executor.symbol_spec(self.cfg.symbol)
            step = float(getattr(spec, "volume_step", 0.01) or 0.01)
            minv = float(getattr(spec, "min_volume", 0.01) or 0.01)
        except Exception:
            step, minv = 0.01, 0.01
        want = float(pos.volume) * float(frac)
        vol = int(want / step) * step if step > 0 else want
        remainder = float(pos.volume) - vol
        if vol < minv or remainder < minv:
            res = self.executor.close_position(int(ticket))
            if not res.ok:
                return False
            self._log_management(
                ticket, pos, reason,
                f"mt5 {reason}: ticket {ticket} closing in full "
                f"({pos.volume:.2f} lots — a {frac:.0%} partial "
                "would strand a sub-minimum remainder)")
            self._record_close(int(ticket), float(price),
                               reason, stats)
            return True
        res = self.executor.close_position(int(ticket), volume=vol)
        if not res.ok:
            self.journal.log_note(
                f"mt5 {reason}: partial close FAILED for ticket "
                f"{ticket} ({res.message})")
            return False
        # PNL-PARTIAL: book the leg now (base entry — management
        # trims are position-level, consistent with the remainder
        # convention in _record_close), then reduce the volume.
        leg = self._book_partial_leg(
            ticket, pos, volume=vol, price=float(price),
            entry=pos.entry_price, stop=pos.stop0)
        self.tick_manager.book_external_partial(int(ticket), vol)
        self._log_management(
            ticket, pos, reason,
            f"mt5 {reason}: ticket {ticket} banked "
            f"{vol:.2f} lots ({frac:.0%} of the position) @ "
            f"{price:.2f} (${leg:+.2f})")
        return True

    def _day_levels(self) -> list:
        """Location magnets from the day object: unswept session
        extremes + prior day/week extremes + naked POCs (S3)."""
        d = self._day_ctx
        if d is None:
            return []
        out: list[float] = []
        for v in (d.session_levels or {}).values():
            out.append(float(v))
        if d.prior:
            out += [float(d.prior["high"]), float(d.prior["low"]),
                    float(d.prior["poc"])]
        if d.prior_week:
            out += [float(d.prior_week["high"]),
                    float(d.prior_week["low"])]
        out += [float(x) for x in (d.naked_pocs or [])]
        return out

    # --------------------------------------- boundary machine + state
    def _update_boundary(self) -> None:
        """Feed the sweep-vs-breakout machine, persist its clock.

        Runs at the end of every TF-close refresh — i.e. BEFORE the
        confirm / order phases of that close, so every order
        decision reads counts that are already persisted.  Order:

        1. Load the persisted state once (warm start): verified ->
           restore the tracker exactly; corrupt/missing -> the
           checksum-mismatch decision tree (never trade on
           compromised memory, never fabricate counts).
        2. Feed newly closed H1 + M15 bars into the tracker
           (idempotent: bars at/before the persisted last-processed
           close are skipped).
        3. If the tracker confirmed a breakout, override the live
           setup with the BREAKOUT_TREND setup (D1 still flat, so
           it is NOT a D1 TREND — trend playbooks arm in the
           breakout direction).
        4. Atomically persist the new counts.  A persist failure
           is a WARNING, never a crash — but it is never silent.

        A persist/load failure must never break the engine loop:
        everything here is wrapped, and failures degrade to
        in-memory-only with a logged WARNING.
        """
        if self._stack is None:
            return
        try:
            h1 = self._stack.frames.get("H1")
            m15 = self._stack.frames.get("M15")
            if h1 is None or m15 is None or not len(h1) \
                    or not len(m15):
                return
            setup = self._live_setup
            rng_def = None
            if setup is not None and setup.get("regime") == "range" \
                    and setup.get("range"):
                try:
                    rng_def = RangeDef.from_dict(setup["range"])
                except Exception:
                    rng_def = None
            tracker = self._boundary
            if tracker is not None and rng_def is None:
                # No defined range from live_setup.  Keep the
                # tracker alive only while it is mid-breakout
                # (BREAKOUT_TREND, or a beyond-1/2 close pending):
                # a breakout needs its 2nd close + BOS even after
                # detect_range stops seeing the range.  Otherwise
                # the range dissolved -> discard, HOLD resumes.
                if tracker.regime != "breakout_trend" \
                        and not tracker.breakout_pending:
                    self._boundary = None
                    self._boundary_snapshot = {}
                    tracker = None
                else:
                    rng_def = tracker.range_def
            if tracker is not None and rng_def is not None \
                    and tracker.regime == "range" \
                    and not tracker.breakout_pending \
                    and (rng_def.high, rng_def.low) != \
                    (tracker.range_def.high, tracker.range_def.low):
                # New range (edges moved) -> fresh machine.
                tracker = None
                self._boundary = None
                self._boundary_snapshot = {}
            if tracker is None and rng_def is not None:
                tracker = self._restore_or_fresh_tracker(
                    rng_def, h1, m15)
                self._boundary = tracker
            if tracker is None:
                # No range, no tracker: persist the plain regime
                # (TREND / HOLD) so a restart still sees it.
                self._persist_regime_state(None)
                return
            events = tracker.update(h1, m15)
            for ev in events:
                if ev.kind in ("breakout_confirmed",
                               "false_breakout", "regime_reverted"):
                    self.journal.log_note(
                        f"mt5 boundary: {ev.kind} ({ev.edge}) — "
                        f"{ev.detail}")
                else:
                    _log.info("mt5 boundary: %s (%s) — %s",
                              ev.kind, ev.edge, ev.detail)
            self._boundary_snapshot = tracker.snapshot()
            if tracker.regime == "breakout_trend":
                bias_h4 = 0
                if setup is not None:
                    try:
                        bias_h4 = int(setup.get("bias_h4", 0))
                    except Exception:
                        bias_h4 = 0
                self._live_setup = self._breakout_setup(
                    tracker, bias_h4=bias_h4)
                self._live_setup_reason = "armed_breakout_trend"
            self._persist_regime_state(tracker)
        except Exception as exc:
            _log.warning("mt5 boundary update failed (in-memory "
                         "only this refresh): %s", exc)

    def _restore_or_fresh_tracker(self, rng_def: RangeDef,
                                 h1, m15) -> BoundaryTracker:
        """First-use tracker: restore from disk, or rebuild fresh.

        Runs once per process (``_regime_state_loaded``).  A
        verified state restores the hysteresis clock EXACTLY.
        A CORRUPT state goes through the checksum-mismatch
        decision tree (pause vs conservative fallback) — never a
        silent zero, never fabricated counts.  A MISSING file is
        a first boot, not corruption: fresh tracker, no pause.

        A fresh tracker ALWAYS starts its cursors at the latest
        closed bars: it counts only bars that close AFTER it is
        created.  Replaying history into a fresh machine would
        fabricate counts (a beyond-close from before the range
        existed, a wick from an old regime) — the rebuild rule
        is lock = 0, confirm = 0, cooldown = 0, full stop.
        """
        fresh = None
        if not self._regime_state_loaded:
            self._regime_state_loaded = True
            try:
                state, status, msg = load_regime_state(
                    self._regime_state_path)
            except Exception as exc:
                state, status, msg = None, "corrupt", str(exc)
            if status == "ok" and state is not None:
                if state.regime in ("range", "breakout_trend") \
                        and state.last_range_high is not None \
                        and state.last_range_low is not None:
                    tracker = BoundaryTracker(rng_def)
                    tracker.apply_regime_state(state)
                    self._regime_state_status = (
                        f"restored: {state.regime.upper()}, lock "
                        f"{state.lock_remaining_h1} H1 remaining, "
                        f"breakout confirm "
                        f"{state.breakout_confirm_count}/2")
                    _log.info("mt5 regime state restored: %s",
                              self._regime_state_status)
                    self.journal.log_note(
                        "mt5: regime state restored: "
                        f"{self._regime_state_status}")
                    return tracker
                # Verified state, but TREND/HOLD (no tracker) or no
                # range stored: nothing to restore, fresh tracker
                # for the current range.
                self._regime_state_status = (
                    f"verified ({state.regime}), fresh tracker")
                fresh = BoundaryTracker(rng_def)
            elif status == "corrupt":
                # Checksum mismatch: decision tree (pause vs
                # conservative fallback).  Never trade on
                # compromised memory.
                self._handle_corrupt_regime_state(status, msg)
                fresh = BoundaryTracker(rng_def)
            else:  # missing: first boot, not corruption
                self._regime_state_status = (
                    "no regime state file yet — fresh tracker")
                fresh = BoundaryTracker(rng_def)
        else:
            fresh = BoundaryTracker(rng_def)
        # Fresh machine counts only NEW bars (see docstring).
        try:
            if h1 is not None and len(h1):
                fresh.last_h1_ts = h1.index[-1]
            if m15 is not None and len(m15):
                fresh.last_m15_ts = m15.index[-1]
        except Exception:
            pass
        return fresh

    def _handle_corrupt_regime_state(self, status: str,
                                     msg: str) -> None:
        """Checksum-mismatch decision tree.  NEVER trade on
        compromised memory, NEVER fabricate counts, NEVER silently
        continue.  The branch depends on OPEN RISK, not one
        blanket rule:

        * Position state untrustworthy (journal corrupt / MT5
          positions unreadable) -> PAUSE COMPLETELY: no new
          orders, no management, CRITICAL + Telegram notify,
          await owner /resume.  The only full-pause case.
        * Open positions OR pending orders exist -> PAUSE NEW
          EXECUTION, KEEP MANAGING EXISTING RISK: no new orders,
          no arming, no regime switch acted on; tick management
          (breakeven / trailing / partials / SL/TP / expiry /
          margin shield) keeps working — it reads position
          state (journal + MT5 positions), NOT regime_state.
          Rebuild in background; after 2 clean H1 bars + no
          contradiction with open risk -> auto-resume; if the
          rebuild contradicts open risk, stay paused until owner
          /resume or the rebuild stabilises for 4 H1 bars.
        * No open risk -> CONSERVATIVE REBUILD: re-derive the
          regime from bars (live_setup already did), lock = 0,
          confirm = 0, cooldown = 0; no new orders until 2 H1 bars
          have closed cleanly post-rebuild (observation window);
          then resume.
        """
        # Any armed (unconfirmed) touch was armed on untrusted
        # memory: drop it, journaled, never erased.
        if self._pending is not None:
            self._pending = None
        trustworthy, why = self._position_state_trustworthy()
        if not trustworthy:
            self._regime_pause = "paused_complete"
            self._regime_pause_reason = why
            self._regime_state_status = (
                f"CORRUPT ({status}): {msg} — position state "
                f"untrusted ({why})")
            _log.critical(
                "mt5 regime_state corrupt, PAUSED COMPLETELY: %s; "
                "position state untrusted (%s)", msg, why)
            self.journal.log_note(
                "CRITICAL mt5: regime_state corrupt, PAUSED "
                f"COMPLETELY — {msg}; position state untrusted "
                f"({why}); no new orders, no management; manual "
                "/resume needed")
            try:
                self._send_alert(
                    "🚨 MT5 PAUSED, state corrupt, manual check "
                    "needed — regime_state checksum mismatch and "
                    f"position state untrusted ({why}). No new "
                    "orders, no management. Send /resume after "
                    "checking.")
            except Exception as exc:
                _log.warning("mt5 corrupt-state Telegram notify "
                             "failed: %s", exc)
            return
        if self._open_risk_exists():
            self._regime_pause = "paused_new"
            self._regime_pause_reason = (
                "open positions / pending orders exist")
            self._regime_observation_remaining = 2
            self._regime_stable_bars = 0
            self._regime_stable_signature = None
            self._regime_state_status = (
                f"CORRUPT ({status}): {msg} — new execution PAUSED, "
                f"managing open positions only")
            _log.error(
                "mt5 regime_state corrupt, NEW execution PAUSED, "
                "managing open positions only, awaiting rebuild: %s",
                msg)
            self.journal.log_note(
                "ERROR mt5: regime_state corrupt, NEW execution "
                "PAUSED, managing open positions only, awaiting "
                f"rebuild ({msg})")
            return
        self._regime_pause = "observation"
        self._regime_pause_reason = "no open risk — conservative rebuild"
        self._regime_observation_remaining = 2
        self._regime_state_status = (
            f"CORRUPT ({status}): {msg} — rebuilt, observation "
            f"window 2 H1 bars")
        _log.warning(
            "mt5 regime_state corrupt, rebuilt from bars, lock "
            "reset conservative: %s", msg)
        self.journal.log_note(
            "WARNING mt5: regime_state corrupt, rebuilt from "
            "bars, lock reset conservative — no new orders until "
            f"2 H1 bars close cleanly ({msg})")

    def _position_state_trustworthy(self) -> tuple[bool, str]:
        """Is the POSITION state (not the regime state) readable?

        Journal: every line must parse as JSON (``records()``
        raises on a corrupt line).  MT5: ``executor.positions()``
        must return.  Called only on the corrupt-regime path.
        """
        try:
            self.journal.records()
        except Exception as exc:
            return False, f"journal corrupt: {exc}"
        if self.executor is None:
            return False, "no executor — positions unreadable"
        try:
            self.executor.positions()
        except Exception as exc:
            return False, f"MT5 positions unreadable: {exc}"
        return True, ""

    def _open_risk_exists(self) -> bool:
        """Open positions (tracked) OR resting broker pending orders."""
        try:
            if self.tick_manager is not None \
                    and self.tick_manager.tickets():
                return True
        except Exception:
            pass
        return bool(self._pending_orders)

    def _rebuild_contradicts_open(self) -> bool:
        """Does the rebuilt regime argue against the open risk?

        Rebuilt TREND / BREAKOUT_TREND whose direction opposes an
        open position, or a rebuilt RANGE with a sweep confirmed
        on the edge that would trade against an open position
        (open LONG + SHORT sweep armed on the high edge, or the
        mirror).  A RANGE with no sweep confirmed, or HOLD, argues
        nothing — no contradiction.  Never auto-trade against open
        risk on rebuilt memory.
        """
        try:
            from .reporting import open_positions
            opens = open_positions(self.journal)
        except Exception:
            return True  # unreadable opens: stay paused (safe side)
        if not opens:
            return False
        open_dirs = {int(o.get("direction") or 0) for o in opens}
        setup = self._live_setup or {}
        regime = str(setup.get("regime", ""))
        if regime in ("trend", "breakout_trend"):
            direction = int(setup.get("direction", 0) or 0)
            if direction != 0 and -direction in open_dirs:
                return True
            return False
        if regime == "range":
            confirmed = set(
                (self._boundary_snapshot or {}).get(
                    "sweep_confirmed_edges") or [])
            # Sweep on the high edge trades SHORT; on low, LONG.
            if "high" in confirmed and +1 in open_dirs:
                return True
            if "low" in confirmed and -1 in open_dirs:
                return True
        return False

    def _regime_stable_signature(self) -> str:
        """Fingerprint of the rebuilt regime for stability counting."""
        setup = self._live_setup or {}
        confirmed = sorted(
            (self._boundary_snapshot or {}).get(
                "sweep_confirmed_edges") or [])
        rng = setup.get("range") or {}
        return (f"{setup.get('regime')}:{setup.get('direction')}:"
                f"{','.join(confirmed)}:"
                f"{rng.get('high')}-{rng.get('low')}")

    def _advance_regime_pause(self) -> None:
        """One clean H1 close of post-rebuild observation / stability.

        Called from the H1-close phase of ``_refresh_bars``.
        Observation (no open risk): 2 clean H1 bars -> resume.
        Paused-new (open risk): after the 2-bar observation, no
        contradiction -> auto-resume; contradiction -> stay paused
        until owner /resume or 4 H1 bars of stable rebuilt regime.
        A fresh open position appearing mid-observation escalates
        observation -> paused_new (the rebuild continues, the
        contradiction check now applies).
        """
        mode = self._regime_pause
        if mode is None:
            return
        if mode == "paused_complete":
            return  # only owner /resume lifts a complete pause
        if mode == "observation" and self._open_risk_exists():
            mode = "paused_new"
            self._regime_pause = "paused_new"
            self._regime_pause_reason = (
                "open risk appeared during observation")
            self._regime_stable_bars = 0
            self._regime_stable_signature = None
            _log.error(
                "mt5 regime pause escalated observation -> "
                "paused_new: open risk appeared during rebuild")
            self.journal.log_note(
                "ERROR mt5: regime pause escalated: open risk "
                "appeared during rebuild observation — new "
                "execution stays PAUSED, managing open positions")
        if self._regime_observation_remaining > 0:
            self._regime_observation_remaining -= 1
            if self._regime_observation_remaining > 0:
                return
            # Observation window complete.
            if mode == "observation":
                # PendingOrderAudit BEFORE any new orders.
                self._audit_pending_orders()
                self._regime_pause = None
                self._regime_pause_reason = ""
                self._regime_state_status = (
                    "re-established, resuming")
                _log.info("mt5 regime state re-established, "
                          "resuming")
                self.journal.log_note(
                    "mt5: regime state re-established, resuming")
                return
            # paused_new: contradiction check decides.
            if not self._rebuild_contradicts_open():
                # PendingOrderAudit BEFORE any new orders.
                self._audit_pending_orders()
                self._regime_pause = None
                self._regime_pause_reason = ""
                self._regime_state_status = (
                    "re-established, resuming (open risk kept)")
                _log.info("mt5 regime state re-established, "
                          "resuming (open risk kept, no "
                          "contradiction)")
                self.journal.log_note(
                    "mt5: regime state re-established, resuming "
                    "(open risk kept, rebuilt regime does not "
                    "contradict it)")
                return
            _log.error(
                "mt5 rebuilt regime contradicts open risk — "
                "staying PAUSED on new orders until /resume or "
                "4 stable H1 bars")
            self.journal.log_note(
                "ERROR mt5: rebuilt regime contradicts open risk "
                "— staying PAUSED on new orders until /resume or "
                "4 stable H1 bars")
            return
        # Post-observation paused_new with contradiction: stability.
        if mode == "paused_new":
            sig = self._regime_stable_signature()
            if sig == self._regime_stable_signature:
                self._regime_stable_bars += 1
            else:
                self._regime_stable_signature = sig
                self._regime_stable_bars = 1
            if self._regime_stable_bars >= 4:
                # PendingOrderAudit BEFORE any new orders.
                self._audit_pending_orders()
                self._regime_pause = None
                self._regime_pause_reason = ""
                self._regime_stable_bars = 0
                self._regime_state_status = (
                    "re-established, resuming (4 stable H1 bars)")
                _log.info("mt5 regime state re-established after "
                          "4 stable H1 bars, resuming")
                self.journal.log_note(
                    "mt5: regime state re-established after 4 "
                    "stable H1 bars, resuming")

    def _breakout_setup(self, tracker: BoundaryTracker, *,
                        bias_h4: int = 0) -> dict:
        """Live-setup dict for a confirmed breakout (BREAKOUT_TREND).

        D1 is still flat, so this is NOT a D1 TREND: ``bias_d1``
        stays 0, and the trend playbooks arm in the breakout
        direction.  The zone is the broken edge as support /
        resistance (retest level); size is the tracker's
        (0.5x until the retest holds, then 1.0x).
        """
        rng = tracker.range_def
        edge = rng.edge_zone(tracker.breakout_edge or "high")
        return {
            "regime": "breakout_trend", "bias_d1": 0,
            "bias_h4": int(bias_h4), "h4_context": "breakout_trend",
            "direction": int(tracker.breakout_direction),
            "zone": edge, "edges": rng.edges(),
            "range": rng.to_dict(),
            "trigger_edge": tracker.breakout_edge,
            "ref_level": float(edge["level"]),
            "atr_h1": float(rng.atr),
            "buf": float(0.5 * rng.atr),
            "h4_highs": [], "h4_lows": [], "swing_n": 3,
            "size_mult": float(tracker.size_mult),
        }

    def _persist_regime_state(
            self, tracker: BoundaryTracker | None) -> None:
        """Atomically persist the regime / hysteresis clock.

        The runner's regime (TREND / RANGE / BREAKOUT_TREND /
        HOLD) rides on the state; the tracker's counters ride
        when a tracker exists.  While a regime pause is active
        (observation / paused_new / paused_complete) NOTHING is
        persisted: the corrupt file stays corrupt on disk, so a
        crash mid-pause re-runs the decision tree on the next
        boot instead of laundering the rebuild as verified
        state.  Never raises — a persist failure is a WARNING,
        never a crash.
        """
        if self._regime_pause is not None:
            return  # paused: do not launder the rebuild as verified
        try:
            import datetime as _dt
            now = _dt.datetime.now(_dt.timezone.utc).strftime(
                "%H:%M:%S")
            if tracker is not None:
                state = tracker.to_regime_state()
            else:
                state = RegimeState()
            setup = self._live_setup
            if setup is not None:
                state.regime = str(setup.get("regime", "hold"))
                if setup.get("range"):
                    rng = setup["range"]
                    state.last_range_high = float(rng["high"])
                    state.last_range_low = float(rng["low"])
                    state.last_range_mid = float(rng["mid"])
                    state.last_range_atr = float(rng["atr"])
            else:
                state.regime = "hold"
            save_regime_state(self._regime_state_path, state,
                              saved_at=now)
            self._regime_state_saved_at = now
            if not self._regime_state_status.startswith("CORRUPT"):
                self._regime_state_status = f"persisted {now}"
        except Exception as exc:
            _log.warning("mt5 regime state persist failed "
                         "(in-memory only): %s", exc)

    def _get_zone_store(self) -> ZoneStore | None:
        """Lazy-open the persistent zone store (journal-adjacent
        ``zones.db``).  None when unavailable — zones are then
        skipped, never fatal.  A failed open is remembered (no
        retry spam every refresh)."""
        if self._zone_store_failed:
            return None
        if self._zone_store is None:
            try:
                path = self.cfg.journal_path.parent / "zones.db"
                path.parent.mkdir(parents=True, exist_ok=True)
                self._zone_store = ZoneStore(path)
            except Exception as exc:
                _log.warning("mt5 zone store unavailable: %s", exc)
                self._zone_store_failed = True
                return None
        return self._zone_store  # type: ignore[return-value]

    def _update_zones(self) -> None:
        """Refresh the multi-day structural zone book — ONE atomic
        transaction per refresh (see live/zones.py).

        Snapshot the stack's closed-bar frames -> store.apply()
        (compute outcomes from that snapshot, persist ALL of them
        in one SQLite transaction) -> swap the in-memory cache
        (version++, copy-on-write).  A compute/persist failure
        keeps the OLD cache — never partial.  Journal mitigated /
        broken zones; log cluster conflicts near price (deduped);
        annotate the live setup with top MAJOR zones + the MAJOR
        zone the armed POI coincides with (``major_zone``), the
        same-direction cluster flag, or ``zone_conflict`` (which
        clears ``major_zone`` — the engine waits for immediate
        confirm, never picks historical alone).

        Zones never trigger a trade alone: they are the WHERE.
        Arming still needs the POI touch + the confirm family
        (the WHEN).  A broken zone does not switch the regime by
        itself — the BoundaryTracker owns regime switches.
        Never raises: a zone failure must not break the refresh.
        """
        if self._stack is None:
            return
        try:
            store = self._get_zone_store()
            if store is None:
                return
            frames = self._stack.frames
            # Idempotency key: max closed-bar timestamp across
            # frames.  A refresh with no new bars replays to [].
            tick_id = None
            for frame in frames.values():
                if frame is not None and len(frame):
                    ts = str(frame.index[-1])
                    if tick_id is None or ts > tick_id:
                        tick_id = ts
            if tick_id is None:
                return
            if not self._zones_seeded:
                outcomes = store.seed(frames)
                self._zones_seeded = True
            else:
                outcomes = store.apply(frames, tick_id)
            zones = store.active_zones()
            self._zone_cache.swap(zones, outcomes, tick_id,
                                   reason="bar close")
            self._structural_zones = zones
            # H1 ATR for the tick-path cache radii (once per
            # refresh, never per tick).
            try:
                h1 = frames.get("H1")
                if h1 is not None and len(h1):
                    self._last_zone_atr = float(atr14(h1))
            except Exception:
                pass
            # Journal mitigated / broken; info-log the rest.
            for oc in outcomes:
                if oc.outcome in ("MITIGATED", "BROKEN"):
                    self.journal.log_note(
                        f"mt5 zone {oc.outcome}: {oc.detail}")
                elif oc.outcome in ("EXPIRED", "FADED", "ACTIVATED"):
                    _log.info("mt5 zone %s: %s", oc.outcome, oc.detail)
            # Clusters + conflict-near-price logging (deduped).
            clusters = find_clusters(zones)
            self._zone_cluster_keys = {
                z.key for cl in clusters
                if cl["same_direction"] for z in cl["zones"]}
            last = self._last_tick or {}
            mid = last.get("mid")
            atr = self._last_zone_atr
            if mid and atr > 0:
                for cl in clusters:
                    if not cl["conflict"]:
                        continue
                    if not any(abs(z.level - float(mid)) <= 0.5 * atr
                               for z in cl["zones"]):
                        continue
                    sig = tuple(sorted(
                        (z.tf, round(z.level, 1)) for z in cl["zones"]))
                    if sig == self._last_zone_conflict_sig:
                        continue
                    self._last_zone_conflict_sig = sig
                    slow = min(cl["zones"],
                               key=lambda z: _TF_ORDER.index(z.tf)
                               if z.tf in _TF_ORDER else 99)
                    fast = max(cl["zones"],
                               key=lambda z: _TF_ORDER.index(z.tf)
                               if z.tf in _TF_ORDER else -1)
                    _log.info(
                        "zone conflict: multi-year %s %.1f vs fresh "
                        "%s %.1f, awaiting immediate confirm",
                        slow.tf, slow.level, fast.tf, fast.level)
            # Annotate the live setup.
            setup = self._live_setup
            if setup is not None:
                majors = [z.to_dict() for z in
                          sorted((z for z in zones
                                  if z.tier == "MAJOR"),
                                 key=lambda z: -z.score)[:3]]
                setup["structural_zones"] = majors
                setup["major_zone"] = self._setup_major_zone(
                    setup, majors)
                setup["zone_conflict"] = None
                levels: list[float] = []
                zone = setup.get("zone") or {}
                if zone.get("level") is not None:
                    levels.append(float(zone["level"]))
                for edge in setup.get("edges") or []:
                    if edge.get("level") is not None:
                        levels.append(float(edge["level"]))
                if levels and atr > 0:
                    tol = 0.5 * atr
                    for cl in clusters:
                        if not any(
                                abs(z.level - lv) <= tol
                                for z in cl["zones"] for lv in levels):
                            continue
                        if cl["conflict"]:
                            setup["zone_conflict"] = {
                                "levels": [float(z.level)
                                           for z in cl["zones"]],
                                "sides": sorted({z.side
                                                 for z in cl["zones"]}),
                            }
                            setup["major_zone"] = None
                        elif setup.get("major_zone") is not None:
                            mz = setup["major_zone"]
                            if any(abs(z.level - float(mz["level"]))
                                   < 1e-9 for z in cl["zones"]):
                                mz["cluster"] = True
                                mz["cluster_score"] = \
                                    cl["combined_score"]
        except Exception as exc:
            _log.warning("mt5 structural zones update failed: %s",
                         exc)

    def _tick_zone_cache(self, mid: float) -> None:
        """Per-tick nearby-zone query (in-memory bisect, NO SQLite).

        Updates the /status nearby list + the nearest MAJOR zone
        within 0.5 ATR (immediate-proximity context).  Never
        raises, never blocks on the store.
        """
        try:
            atr = self._last_zone_atr
            if not atr or atr <= 0:
                return
            _v, nearby = self._zone_cache.nearby(
                float(mid), atr, radius_atr=2.0)
            self._nearby_zones = nearby
            _v2, majors = self._zone_cache.major_nearby(
                float(mid), atr, radius_atr=0.5)
            self._near_major_zone = majors[0] if majors else None
        except Exception:
            pass

    @staticmethod
    def _setup_major_zone(setup: dict, majors: list) -> dict | None:
        """The MAJOR zone the armed POI coincides with, if any.

        Coincidence = MAJOR level within 0.5x ATR of the setup's
        POI level (trend) or either range edge (range).  None
        means the armed POI is not a MAJOR structural zone —
        the setup still arms (fresh POIs have no touches yet by
        construction); the score prioritises, it does not gate.
        """
        try:
            atr = float(setup.get("atr_h1", 0.0) or 0.0)
            tol = 0.5 * atr if atr > 0 else 0.0
            levels: list[float] = []
            zone = setup.get("zone") or {}
            if zone.get("level") is not None:
                levels.append(float(zone["level"]))
            for edge in setup.get("edges") or []:
                if edge.get("level") is not None:
                    levels.append(float(edge["level"]))
            for major in majors:
                if any(abs(float(major["level"]) - lv) <= max(tol, 1e-9)
                       for lv in levels):
                    return major
        except Exception:
            pass
        return None

    def zones_line(self) -> str:
        """Cache version + top MAJOR structural zones + distance to
        nearest, for /status (see mt5_commands).  Cache line:
        ``Cache v1234, last update H1 close 14:00, zones applied:
        D1 BROKEN 4227.5, H1 MITIGATED 4210.0, no partial`` and
        ``Zone cache: 342 active zones, 3 nearby, last refresh H1
        close, query p95 0.2ms``.
        """
        stats = self._zone_cache.stats()
        lines = [
            f"Cache v{stats['version']}, last update "
            f"{stats['last_refresh_reason']} "
            f"{stats['last_refresh_ts'][:16] if stats['last_refresh_ts'] else '---'}, "
            f"zones applied: {stats['last_outcomes'] or 'no changes'}, "
            f"no partial",
            f"Zone cache: {stats['n_zones']} active zones, "
            f"{len(self._nearby_zones)} nearby, last refresh "
            f"{stats['last_refresh_reason']}, query p95 "
            f"{stats['p95_ms']:.2f}ms",
        ]
        majors = [z for z in self._structural_zones
                  if z.status == "active" and z.tier == "MAJOR"][:3]
        if not majors:
            minors = [z for z in self._structural_zones
                      if z.status == "active"][:1]
            if minors:
                lines.append(
                    "zones: no MAJOR structural zones — top "
                    f"MINOR: {minors[0].describe()}")
            else:
                lines.append(
                    "zones: no structural zones (insufficient bars)")
            return "\n".join(lines)
        for z in majors:
            lines.append(
                f"zones: {z.describe(cluster=z.key in self._zone_cluster_keys)}")
        last = self._last_tick or {}
        mid = last.get("mid")
        if mid:
            actives = [z for z in self._structural_zones
                       if z.status == "active"]
            if actives:
                nearest = min(actives,
                              key=lambda z: abs(z.level - mid))
                lines.append(
                    f"nearest zone {nearest.level:.1f} "
                    f"(${abs(mid - nearest.level):.1f} away)")
        return "\n".join(lines)

    def market_read_line(self) -> str:
        """Standing market read for /status (see market_read).

        Inputs are the live zone cache + D1 hold state only: the
        trend phrase comes from confirmed swings, the key zones
        from active zone bands, the scenario triggers from those
        bands.  Nothing here copies an analyst's read.
        """
        from goldenswing.execution.market_read import render_market_read
        last = getattr(self, "_last_tick", None) or {}
        price = last.get("mid")
        hold_state = getattr(self, "_live_hold_state", None)
        hold = hold_state.to_dict() if hold_state is not None else None
        zones = []
        for z in getattr(self, "_structural_zones", None) or []:
            if z.status != "active":
                continue
            zones.append({
                "tf": z.tf, "side": z.side, "level": float(z.level),
                "top": float(z.top), "bottom": float(z.bottom),
                "score": float(z.score), "tier": z.tier,
                "touches": int(z.touches),
            })
        return render_market_read(price=price, hold=hold, zones=zones)

    def report_feed_status(self, feed_status: dict | None) -> None:
        """Log a feed failure on the SEPARATE FEED_ERROR channel.

        Call with the ``feed.poll_status()`` dict.  ``ok=False`` logs
        one ERROR ``FEED_ERROR`` line (terminal disconnected, symbol
        unselected, fetch failed, stale bars).  ``ok=True`` logs
        nothing — a healthy feed is not news, and a flat bias on a
        healthy feed is a HOLD (see ``_refresh_live_setup``), never
        a feed error.  The two channels are never conflated.
        """
        if not feed_status or feed_status.get("ok", True):
            return
        log_feed_error(
            _log, symbol=self.cfg.symbol,
            reason=str(feed_status.get("reason") or "feed not ok"))

    def setup_line(self) -> str:
        """One-line live-setup state for /status (see mt5_commands).

        Armed: direction + H4 POI band + H4 context + distance
        from the last tick to the zone.  Not armed: the reason
        (d1_flat / no_h4_poi / containment / …) — the answer to
        \"why no trades\" when no playbook has been reached yet.
        """
        setup = self._live_setup
        if setup is None:
            # d1_flat with no defined range renders as HOLD with
            # pending swings / bars-remaining / flip requirements /
            # earliest flip — never as a feed error, never as a
            # bare "none".
            if self._live_setup_reason.startswith("d1_flat") \
                    and self._live_hold_state is not None:
                line = self._live_hold_state.describe()
                last = self._last_tick or {}
                mid = last.get("mid")
                if mid:
                    line += f" · last price {mid:.1f}"
                return line
            return (f"setup: none — {self._live_setup_reason}"
                    + self._score_pool_lines())
        if setup.get("regime") == "range":
            # RANGE regime: range band + touches + mid + nearest
            # edge + distance to it — never "HOLD d1_flat".
            rng = RangeDef.from_dict(setup["range"])
            line = (f"setup: ARMED RANGE · range "
                    f"{rng.low:.1f}–{rng.high:.1f} "
                    f"(width {rng.width:.1f} = "
                    f"{rng.width / rng.atr:.1f}x ATR, touches "
                    f"H{rng.touches_high}/L{rng.touches_low}, "
                    f"mid {rng.mid:.1f}"
                    + (", compressed" if rng.compressed else "") + ")")
            last = self._last_tick or {}
            mid = last.get("mid")
            if mid:
                if mid < rng.low:
                    line += (f" · last price {mid:.1f} "
                             f"(${rng.low - mid:.1f} below range)")
                elif mid > rng.high:
                    line += (f" · last price {mid:.1f} "
                             f"(${mid - rng.high:.1f} above range)")
                else:
                    d_low = abs(mid - rng.low)
                    d_high = abs(mid - rng.high)
                    if d_low <= d_high:
                        line += (f" · last price {mid:.1f} "
                                 f"(${d_low:.1f} from low edge — "
                                 f"LONG side armed)")
                    else:
                        line += (f" · last price {mid:.1f} "
                                 f"(${d_high:.1f} from high edge — "
                                 f"SHORT side armed)")
            return line
        if setup.get("regime") == "breakout_trend":
            # BREAKOUT_TREND: D1 still flat, range edge broken —
            # trend playbooks arm in the breakout direction.
            direction = "LONG" if setup["direction"] > 0 else "SHORT"
            edge = setup["zone"]
            snap = self._boundary_snapshot or {}
            lock = snap.get("lock_remaining_h1", 0)
            size = float(setup.get("size_mult", 0.5) or 0.5)
            retest = "held" if snap.get("retest_held") else "pending"
            line = (f"setup: ARMED BREAKOUT_TREND {direction} · "
                    f"broke {edge['side']} edge {edge['level']:.1f} "
                    f"(retest {retest}, size {size:.1f}x, "
                    f"lock {lock} more H1 bars)")
            last = self._last_tick or {}
            mid = last.get("mid")
            if mid:
                line += (f" · last price {mid:.1f} "
                         f"(${abs(mid - edge['level']):.1f} from "
                         f"broken edge)")
            return line
        if setup.get("arm") == "continuation":
            # CONTINUATION arm: a closed H1 bar broke the reference
            # level; the zone is the broken level's retest band,
            # not an H4 POI — render it as what it is (F3).
            direction = "LONG" if setup["direction"] > 0 else "SHORT"
            zone = setup["zone"]
            line = (f"setup: ARMED {direction} CONTINUATION · "
                    f"broke H1 ref {setup['ref_level']:.1f} "
                    f"(break {setup.get('break_ts', '')}) · "
                    f"retest band {zone['bottom']:.1f}–"
                    f"{zone['top']:.1f} (H4 "
                    f"{setup.get('h4_context', '')})")
            last = self._last_tick or {}
            mid = last.get("mid")
            if mid:
                line += (f" · last price {mid:.1f} "
                         f"(${abs(mid - setup['ref_level']):.1f} "
                         "from broken level)")
            return line
        zone = setup["zone"]
        direction = "LONG" if setup["direction"] > 0 else "SHORT"
        context = setup.get("h4_context", "")
        line = (f"setup: ARMED {direction} · H4 POI "
                f"{zone['bottom']:.1f}–{zone['top']:.1f} "
                f"(H4 {context}) · H1 ref {setup['ref_level']:.1f}")
        last = self._last_tick or {}
        mid = last.get("mid")
        if mid:
            if mid < zone["bottom"]:
                line += f" · last price {mid:.1f} (${zone['bottom'] - mid:.1f} below zone)"
            elif mid > zone["top"]:
                line += f" · last price {mid:.1f} (${mid - zone['top']:.1f} above zone)"
            else:
                line += f" · last price {mid:.1f} INSIDE zone (armed, awaiting POI confirm)"
        line += self._score_pool_lines()
        return line

    def _score_pool_lines(self) -> str:
        """Rework /status additions (spec §6.2): the live setup's
        latest Asymmetry Score breakdown + the held position's
        RiskPool state + any active PullbackWatch.  Appended to
        the setup line — every pre-existing status field is
        untouched."""
        out = ""
        b = self._last_score
        if b:
            def _p(name):
                return float((b.get(name) or {}).get("points", 0.0))
            out += (f"\nscore: {float(b.get('total', 0.0)):.0f}/100 "
                    f"[structure {_p('structure'):.0f}/30 · "
                    f"efficiency {_p('efficiency'):.0f}/25 · "
                    f"pullback {_p('pullback'):.0f}/15 · "
                    f"location {_p('location'):.0f}/15 · "
                    f"session {_p('session'):.0f}/10 · "
                    f"macro {_p('macro'):.0f}/5]"
                    + (" · FLIPPED" if b.get("flipped") else ""))
        for ticket, pool in self._pools.items():
            st = pool.state()
            out += (f"\npool: ticket {ticket} budget "
                    f"${st['budget']:.2f} · consumed "
                    f"${st['consumed']:.2f} · headroom "
                    f"${st['headroom']:.2f} · "
                    f"{len(st['layers'])} layer(s) booked")
        for w in self._pullback_watches:
            zone = w["zone"]
            left = int(self.cfg.pullback_watch_bars) \
                - (self._current_h1_bar() - int(w["bar0"]))
            out += (f"\nwatch: {w['playbook']} "
                    f"{'+' if w['direction'] > 0 else '-'}1 "
                    f"pullback watch on {zone.get('tf', '?')} zone "
                    f"{float(zone['bottom']):.1f}–"
                    f"{float(zone['top']):.1f} · {max(left, 0)} H1 "
                    "bars left")
        return out

    def day_line(self) -> str:
        """P3 /status day line (spec v3 §12 J): the day object +
        posture in one compact line — day type, ADR consumption,
        IB state, session posture, NY checkpoint, SMT."""
        d = self._day_ctx
        if d is None:
            return "day: — (no frames yet)"
        parts = [f"day: {d.day_type}"]
        if d.adr > 0:
            if d.adr_days < 10:
                # Disarmed gauge (stub D1 sample, or a thin sample
                # under the standing >=10-day guard): never print a
                # pct no decision may price.  The numeric adr_pct
                # stays in the journal context / summary dict
                # untouched — only this human-facing string changes.
                # (The magnitude leg of _adr_gauge_valid needs a
                # signal ATR the day line does not carry, so the
                # render gates on the sample guard alone.)
                parts.append("ADR — (disarmed)")
            else:
                parts.append(f"ADR {d.adr_consumed_pct:.0f}% "
                             f"({d.adr_state})")
        if d.ib_width > 0:
            parts.append(
                f"IB {'set' if d.ib_complete else 'forming'} "
                f"{d.ib_low:.0f}–{d.ib_high:.0f}")
        if self._posture is not None:
            parts.append(f"posture {self._posture.name}")
        sess = _posture_mod.session_posture(
            self._last_tick.get("time") if self._last_tick
            else pd.Timestamp.now(tz="UTC"),
            spread_ratio=self._spread_ratio)
        parts.append(f"session {sess.name}")
        if self._ny_checkpoint:
            parts.append(f"NY {self._ny_checkpoint}")
        if d.gap_state != "none":
            parts.append(f"gap {d.gap_state}")
        read = self._smt_read
        if read is not None:
            if read.event is not None:
                parts.append(f"SMT {read.event.kind}")
            elif not read.available:
                parts.append("SMT n/a (corr broken)")
        return " · ".join(parts)

    def boundary_line(self) -> str:
        """Edge-boundary + persisted-state lines for /status.

        Shows the sweep-vs-breakout boundary state (wick awaiting
        reclaim / breakout closes / retest / lock / cooldown), the
        persisted hysteresis clock (last saved, lock, confirm
        count), any active regime pause (never hidden), the state
        file path, and the last pending-audit summary.  Multi-line
        (one fact per line); /status joins lines with newlines.
        """
        out: list[str] = []
        snap = self._boundary_snapshot or {}
        # Regime pause states are NEVER hidden.
        if self._regime_pause == "observation":
            done = 2 - max(0, self._regime_observation_remaining)
            out.append(
                "Regime state: CORRUPT, rebuilt, new execution "
                f"PAUSED for 2 H1 bars ({done}/2)")
        elif self._regime_pause == "paused_new":
            out.append(
                "Regime state: CORRUPT, rebuilt — Managing open "
                "positions only (new execution PAUSED)")
            if self._regime_observation_remaining <= 0:
                try:
                    contra = self._rebuild_contradicts_open()
                except Exception:
                    contra = False
                if contra:
                    out.append(
                        "rebuild contradicts open risk — /resume "
                        "or 4 stable H1 bars to resume "
                        f"({self._regime_stable_bars}/4)")
        elif self._regime_pause == "paused_complete":
            out.append(
                "Regime state: CORRUPT — PAUSED COMPLETE, manual "
                "/resume needed (position state untrusted)")
        elif snap:
            rng = snap.get("range") or {}
            atr = float(rng.get("atr") or 0.0)
            edge_states = snap.get("edge_states") or {}
            for side in ("high", "low"):
                level = rng.get(side)
                if level is None:
                    continue
                st = (edge_states.get(side) or {}).get("state",
                                                      "inside")
                cooling = (snap.get("cooldown_remaining_h1") or {}
                           ).get(side, 0)
                if cooling and cooling > 0:
                    text = f"cooling ({cooling} H1 left)"
                elif st == "wick":
                    text = ("last sweep: wick beyond, awaiting "
                            "reclaim close")
                elif st == "sweep_confirmed":
                    text = "sweep confirmed, armed"
                elif st == "beyond1":
                    text = ("Breakout: close beyond 1/2, awaiting "
                            "2nd close")
                elif st == "beyond2_no_bos":
                    text = ("Breakout: close beyond 2/2, awaiting "
                            "M15 BOS")
                elif st == "awaiting_retest":
                    text = "Breakout confirmed, awaiting retest"
                elif st == "retest_held":
                    text = "Retest held"
                elif st == "breakout_no_retest":
                    text = "Breakout, no retest (0.5x)"
                else:
                    text = "inside"
                out.append(
                    f"Edge {side.upper()} {float(level):.1f}, "
                    f"buffer ±{0.25 * atr:.1f} · {text}")
            lock = int(snap.get("lock_remaining_h1", 0) or 0)
            if lock > 0:
                regime = str(snap.get("regime", "")).upper()
                out.append(
                    f"Regime locked {regime} for {lock} more "
                    f"H1 bars")
            edge = snap.get("breakout_edge")
            confirm = 0
            if edge:
                confirm = int((edge_states.get(edge) or {}).get(
                    "beyond_closes", 0) or 0)
            out.append(
                f"Regime state: {self._regime_state_status}, "
                f"lock {lock} H1, confirm {confirm}/2")
        else:
            out.append(
                f"Regime state: {self._regime_state_status}")
        try:
            import os as _os
            disp = _os.path.relpath(str(self._regime_state_path))
        except Exception:
            disp = str(self._regime_state_path)
        out.append(f"State file: {disp}")
        audit = self._last_pending_audit
        if audit is not None:
            cancelled = audit.get("cancelled") or {}
            total_cancelled = sum(cancelled.values())
            breakdown = ", ".join(
                f"{n} {k}" for k, n in sorted(cancelled.items())
                if n > 0) or "none"
            out.append(
                "Pending audit on resume: "
                f"{total_cancelled} cancelled ({breakdown}), "
                f"{audit.get('kept', 0)} kept, "
                f"{audit.get('filled', 0)} filled during downtime")
        return "\n".join(out)

    def stack_line(self) -> str:
        """One-line per-TF last closed bar for /status.

        The old /status showed only the H1 last bar, which read as
        \"feed is H1-only\" — the stack carries D1/H4/H1/M15/M30/M5
        (+ W1/MN1 macro).  This line proves it per TF.
        """
        if self._stack is None:
            return "bars: no stack"
        parts = []
        for tf in ("D1", "H4", "H1", "M15", "M30", "M5"):
            frame = self._stack.frames.get(tf)
            if frame is None or len(frame) == 0:
                parts.append(f"{tf}: —")
            else:
                parts.append(f"{tf}: {frame.index[-1]:%m-%d %H:%M} "
                             f"×{len(frame)}")
        return "bars (last closed, UTC): " + " · ".join(parts)

    def freshness_line(self) -> str:
        """Per-TF feed freshness for /status — the audit's point 1
        made computable. For each stack TF: age of its last closed
        bar vs the engine's UTC clock, judged against the XAUUSD
        calendar (live/market_hours.py). Market open + age beyond
        2x the TF interval + 5 min grace -> STALE (fail loud, never
        a bare OK). Market closed -> CLOSED-EXPECTED, not stale.
        No frame -> UNKNOWN. Bar times use the engine's stored UTC
        convention (same as stack_line); if the broker's server
        clock disagrees with that convention, the STALE flag is what
        surfaces the disagreement instead of hiding it.
        """
        from datetime import datetime, timezone

        from ..live.market_hours import xauusd_market_open

        if self._stack is None:
            return "freshness: no stack"
        now = datetime.now(timezone.utc)
        market_open = xauusd_market_open(now)
        intervals = {"D1": 1440, "H4": 240, "H1": 60,
                     "M15": 15, "M30": 30, "M5": 5}
        parts = []
        for tf in ("D1", "H4", "H1", "M15", "M30", "M5"):
            frame = self._stack.frames.get(tf)
            if frame is None or len(frame) == 0:
                parts.append(f"{tf}: UNKNOWN")
                continue
            last = frame.index[-1]
            try:
                last_utc = last.tz_localize("UTC") \
                    if last.tzinfo is None else last.tz_convert("UTC")
                age_min = (now - last_utc.to_pydatetime()).total_seconds() / 60
            except Exception:
                parts.append(f"{tf}: UNKNOWN")
                continue
            if age_min < 60:
                age_txt = f"{age_min:.0f}m"
            elif age_min < 2880:
                age_txt = f"{age_min / 60:.1f}h"
            else:
                age_txt = f"{age_min / 1440:.1f}d"
            if not market_open:
                state = "CLOSED-EXPECTED"
            elif age_min <= 2 * intervals[tf] + 5:
                state = "fresh"
            else:
                state = "STALE ⚠️"
            parts.append(f"{tf}: {age_txt} {state}")
        market_txt = "OPEN" if market_open else "CLOSED (weekly close)"
        return (f"freshness (market {market_txt}, bar times UTC "
                f"convention): " + " · ".join(parts))

    def alignment_line(self) -> str:
        """One-line alignment for /status (see mt5_commands)."""
        return f"macro alignment: {self._alignment.describe()}"

    # ------------------------------------------- live evaluation clock
    # The H1 close is NOT the signal clock.  A touch arms the
    # pending when the tick sits AT the H4 POI band; the
    # confirmation family fires on entry-TF closes; H1/H4/D1
    # closes only refresh their legs.  The hierarchy legs and gate
    # formulas are unchanged (see
    # live/hierarchy.py::live_setup / check_zone_touch /
    # check_live_trigger) — only the TRIGGER changed.

    def _pending_current_score(self, state, playbooks: dict,
                                states, t: int, h1
                                ) -> tuple[float | None, str | None]:
        """Re-score one pending against CURRENT bars.

        Re-runs the originating playbook's ``live_signal`` on the
        current H1 frame: the current score is the score of a
        same-direction signal whose entry sits at this pending's
        level (within 0.5x its ATR).  Returns ``(score, crash)``:

        - ``(score, None)`` — re-evaluation ran; ``score`` is None
          when no such signal exists (the setup is gone) or the
          playbook is unknown (the setup cannot be re-confirmed).
        - ``(None, error)`` — the re-evaluation CRASHED.  A crash is
          NOT "gone": callers must fail open on it (decay leaves
          the pending alone; the audit keeps it and lets the normal
          machinery re-check).  Never raises.
        """
        try:
            pb = playbooks.get(state.playbook)
            if pb is None:
                return None, None
            for sig in pb.live_signal(states, t, h1) or []:
                if int(sig.direction) != state.direction:
                    continue
                atr = float(getattr(sig, "atr", 0.0) or 0.0)
                tol = 0.5 * atr if atr > 0 else 0.0
                if abs(float(sig.entry_ref) - state.level) \
                        <= max(tol, 1e-9):
                    return float(sig.score), None
        except Exception as exc:
            self.journal.log_note(
                f"mt5: pending re-score failed for ticket "
                f"{state.ticket} (treated as crash, not gone): {exc}")
            return None, str(exc)
        return None, None

    def _check_pending_score_decay(self, stats: dict) -> None:
        """Score-decay cancellation (owner's-bot PendingOrderTracker):
        on each H1 close, re-evaluate the originating playbook; a
        pending whose setup no longer emits a same-direction signal
        at this level scoring at/above the decay floor accrues a
        decay bar, and at ``score_decay_bars`` it is cancelled.

        A re-evaluation crash never cancels: the pending is left
        alone and the failure is journalled (fail-open — decay is
        an optimisation, not a safety rule; invalidation/expiry/
        bias-flip still guard a truly dead setup).
        """
        if not self.cfg.score_decay_enabled \
                or not self._pending_orders \
                or self._stack is None:
            return
        h1 = self._stack.frames.get("H1")
        if h1 is None or len(h1) == 0:
            return
        floor = self.cfg.score_decay_floor \
            if self.cfg.score_decay_floor is not None \
            else self.cfg.min_score
        try:
            states = read_and_featurize(h1, None)
        except Exception as exc:
            self.journal.log_note(
                f"mt5: score-decay check skipped (featurize failed): "
                f"{exc}")
            return
        t = len(h1) - 1
        playbooks = {pb.name: pb for pb in self._get_playbooks()}
        for ticket, state in list(self._pending_orders.items()):
            if state.premise == "native":
                # Step 4: H1-clock decay never applies to a native
                # pending — the H1 roster cannot re-score it (natives
                # are not in _get_playbooks), so without this skip
                # every native pending reads "gone" and dies within
                # score_decay_bars H1 closes.  Step 2 exempted only
                # the no-floor route (qm_m5); qm_d1 carries a floor
                # and still died the same way — this skip covers
                # ALL native premises.  A native-frame re-emission
                # decay was investigated and REJECTED: the QM
                # live_signal is touch-only and cursor-based (it
                # returns signals touched on bar t, never re-emits
                # a processed bar), so absence of a re-emission
                # proves nothing about a resting limit — decay by
                # re-emission would kill every QM pending within
                # score_decay_bars native closes.  Native pendings
                # are guarded by native expiry + invalidation
                # instead (see _check_native_pendings).
                self.journal.log_note(
                    f"mt5: pending ticket {ticket} decay skipped "
                    f"({state.playbook} native premise — H1 decay "
                    f"does not apply; native expiry/invalidation "
                    f"guard it)")
                continue
            if _decay_exempt(state.playbook):
                # Step 2: this route carries NO score floor, so the
                # legacy-floor decay must not touch it.
                self.journal.log_note(
                    f"mt5: pending ticket {ticket} decay skipped "
                    f"({state.playbook} carries no score floor — "
                    f"legacy-floor decay does not apply)")
                continue
            try:
                current, crash = self._pending_current_score(
                    state, playbooks, states, t, h1)
                if crash is not None:
                    # Fail-open: a re-evaluation crash never cancels.
                    self.journal.log_note(
                        f"mt5: score-decay check failed for ticket "
                        f"{ticket} (pending left alone): {crash}")
                    continue
                if current is not None and current >= floor:
                    if state.decay_bars:
                        self.journal.log_note(
                            f"mt5: pending ticket {ticket} score "
                            f"recovered ({current:.0f} >= floor "
                            f"{floor:.0f}) — decay reset")
                    state.decay_bars = 0
                    continue
                state.decay_bars += 1
                if state.decay_bars >= self.cfg.score_decay_bars:
                    self._cancel_pending(
                        ticket,
                        f"score_decay: setup re-scored "
                        f"{current if current is not None else 'gone'} "
                        f"vs floor {floor:.0f} for "
                        f"{state.decay_bars} H1 closes "
                        f"(placed score {state.placed_score:.0f})",
                        stats)
                else:
                    self.journal.log_note(
                        f"mt5: pending ticket {ticket} decay "
                        f"{state.decay_bars}/"
                        f"{self.cfg.score_decay_bars} "
                        f"(re-score "
                        f"{current if current is not None else 'gone'} "
                        f"vs floor {floor:.0f})")
            except Exception as exc:
                self.journal.log_note(
                    f"mt5: score-decay check failed for ticket "
                    f"{ticket} (pending left alone): {exc}")

    # ------------------------------------------- PendingOrderAudit
    def _audit_pending_orders(self, stats: dict | None = None
                              ) -> dict:
        """Audit resting pending orders on resume from a regime pause.

        Stale pendings NEVER auto-resume: the audit runs BEFORE
        any new order after a rebuild (auto-resume after the
        observation window / stable bars, or owner /resume), and
        every transition out of a regime pause goes through it.

        1. Reconcile: MT5 live pendings + recorded pendings.
           Filled during downtime (a broker position with this
           pending's alert_id + magic) -> journal "pending filled
           during downtime, reconciled", counted, NEVER
           double-placed (the fill is left for the normal
           reconcile to record into the position book).
           Broker-cancelled (neither resting nor filled) ->
           dropped, counted ``broker_cancelled``.
        2. Re-validate each survivor against CURRENT state (never
           pre-pause state): expiry (``pending_expiry_bars``),
           POI still valid (price not through POI + buffer, POI
           level still in the current H4 zones), distance
           (entry vs price > 2.0x ATR14, or price already through
           the entry — would fill instantly at a bad price),
           regime still supports the direction, score /
           manipulation re-evaluated on current bars.
        3. Cancel stale via MT5 + journal with reason codes
           (``stale_poi`` / ``expired`` / ``distance`` /
           ``regime_flip`` / ``score_drop`` /
           ``filled_during_downtime``) + Telegram per cancel.
           Keep valid only if ALL checks pass, logged "Pending X
           kept on resume, still valid".  NEVER reprice: the
           pending price IS the POI — if the POI moved, cancel +
           re-arm a fresh setup (the normal tick clock re-arms
           from current bars on subsequent ticks; the audit
           itself never places orders or re-arms the same price
           just because it existed pre-pause).

        Returns the summary dict (also stored on
        ``_last_pending_audit`` for /status).  Never raises: a
        failed audit step leaves the pending alone and is
        journalled (the normal per-tick machinery re-checks it).
        """
        stats = stats if stats is not None else {}
        summary: dict = {"cancelled": {}, "kept": 0, "filled": 0,
                         "cancel_failed": 0}
        try:
            # --- 1. Reconcile: live MT5 vs recorded pendings.
            try:
                live_pending = {
                    o.ticket: o for o in
                    self.executor.pending_orders(self.cfg.symbol)
                } if self.executor is not None else {}
            except Exception as exc:
                live_pending = {}
                self.journal.log_note(
                    "mt5 audit: live pending fetch failed "
                    f"({exc}) — validating recorded pendings only")
            try:
                live_positions = list(self.executor.positions()) \
                    if self.executor is not None else []
            except Exception as exc:
                live_positions = []
                self.journal.log_note(
                    "mt5 audit: live positions fetch failed "
                    f"({exc}) — fill detection skipped")
            for ticket, state in list(self._pending_orders.items()):
                filled = any(
                    getattr(p, "alert_id", None) == state.alert_id
                    and getattr(p, "magic", None) == self.cfg.magic
                    for p in live_positions)
                resting = ticket in live_pending
                if filled:
                    summary["filled"] += 1
                    self.journal.log_note(
                        f"mt5: pending ticket {ticket} filled "
                        f"during downtime, reconciled — no "
                        f"double-place (position kept, normal "
                        f"reconcile records the fill)")
                    continue  # left tracked: normal reconcile
                    # records the fill into the position book
                if not resting:
                    del self._pending_orders[ticket]
                    summary["cancelled"]["broker_cancelled"] = \
                        summary["cancelled"].get(
                            "broker_cancelled", 0) + 1
                    self.journal.log_note(
                        f"mt5: pending ticket {ticket} gone — "
                        f"cancelled externally / broker expiry, "
                        f"no fill (audit)")
            # --- 2. Re-validate survivors against CURRENT state.
            h1 = self._stack.frames.get("H1") \
                if self._stack is not None else None
            price = (self._last_tick or {}).get("mid")
            try:
                atr = float(atr14(h1)) \
                    if h1 is not None and len(h1) else 0.0
            except Exception:
                atr = 0.0
            for ticket, state in list(self._pending_orders.items()):
                # Already-filled pendings (step 1) skip validation:
                # they are positions now, not pendings.
                if any(getattr(p, "alert_id", None)
                       == state.alert_id
                       and getattr(p, "magic", None)
                       == self.cfg.magic for p in live_positions):
                    continue
                reason = self._audit_pending_reason(
                    state, h1, price, atr)
                if reason is None:
                    summary["kept"] += 1
                    self.journal.log_note(
                        f"mt5: Pending {ticket} kept on resume, "
                        f"still valid ({state.kind} @ "
                        f"{state.level:.2f})")
                    continue
                code, detail = reason
                ok = self._cancel_pending(
                    ticket, f"resume_audit:{code}: {detail}", stats)
                if ok:
                    summary["cancelled"][code] = \
                        summary["cancelled"].get(code, 0) + 1
                    self.journal.log_note(
                        f"mt5: pending ticket {ticket} cancelled "
                        f"on resume: {code} — {detail}")
                    try:
                        self._send_alert(
                            f"Pending {ticket} cancelled on "
                            f"resume: {code} — {detail}")
                    except Exception:
                        pass
                else:
                    summary["cancel_failed"] += 1
        except Exception as exc:
            self.journal.log_note(
                f"mt5: pending audit failed ({exc}) — pendings "
                f"left alone, normal machinery re-checks")
        import datetime as _dt
        self._last_pending_audit = summary
        self._last_pending_audit_at = _dt.datetime.now(
            _dt.timezone.utc).strftime("%H:%M:%S")
        cancelled = summary["cancelled"]
        self.journal.log_note(
            "mt5: pending audit on resume: "
            f"{sum(cancelled.values())} cancelled "
            f"({', '.join(f'{n} {k}' for k, n in sorted(cancelled.items()) if n > 0) or 'none'}), "
            f"{summary['kept']} kept, "
            f"{summary['filled']} filled during downtime")
        return summary

    def _audit_pending_reason(self, state, h1, price, atr
                              ) -> tuple[str, str] | None:
        """First failing re-validation for one pending, or None (keep).

        Check order (first failure wins the reason): expiry ->
        POI -> distance/through -> regime -> score/manipulation.
        """
        # Expiry: pending_expiry_bars (config) elapsed during
        # downtime -> stale.  Step 4: a native pending is audited
        # on its own clock when that frame is available, and the
        # H1 check is then SKIPPED for it — otherwise the H1 pair
        # would still kill a daily pending after 4h at audit time,
        # which is exactly the units bug entry-clock expiry fixes.
        # The H1 pair is the fallback only when the native frame
        # is missing (or the pending is H1 / adopted).
        _native_audited = False
        if getattr(state, "entry_clock", "H1") != "H1" \
                and self._stack is not None:
            _nframe = self._stack.frames.get(state.entry_clock)
            if _nframe is not None and len(_nframe):
                _native_audited = True
                _nidx = len(_nframe) - 1
                if _nidx >= state.entry_expiry_bar:
                    return ("expired",
                            f"entry_expiry_bar "
                            f"{state.entry_expiry_bar} passed on "
                            f"{state.entry_clock} (now {_nidx})")
        if not _native_audited:
            last_bar_index = len(h1) - 1 \
                if h1 is not None and len(h1) else 0
            if last_bar_index >= state.expiry_bar:
                return ("expired",
                        f"expiry_bar {state.expiry_bar} passed "
                        f"(now {last_bar_index})")
        # POI still valid?  Price through the POI + buffer kills
        # it; a POI level that no longer exists in the current H4
        # zones is a dead setup (never reprice silently).
        zone = state.zone
        if zone is None:
            return ("stale_poi",
                    "unverifiable POI (adopted pending, no zone)")
        buf = 0.5 * atr if atr > 0 else 0.0
        if price is not None:
            if state.direction > 0 \
                    and price < float(zone["bottom"]) - buf:
                return ("stale_poi",
                        f"price {float(price):.2f} through POI "
                        f"support {float(zone['bottom']):.2f} "
                        f"+ buffer")
            if state.direction < 0 \
                    and price > float(zone["top"]) + buf:
                return ("stale_poi",
                        f"price {float(price):.2f} through POI "
                        f"resistance {float(zone['top']):.2f} "
                        f"+ buffer")
        try:
            h4 = self._stack.frames.get("H4") \
                if self._stack is not None else None
            if h4 is not None and len(h4):
                zones = zones_by_side(h4, swing_n=3)
                want = "support" if state.direction > 0 \
                    else "resistance"
                cur = zones.get(want)
                tol = 0.5 * atr if atr > 0 else 0.0
                if cur is None or abs(
                        float(cur["level"])
                        - float(zone["level"])) > max(tol, 1e-9):
                    return ("stale_poi",
                            f"POI level {float(zone['level']):.2f} "
                            f"no longer in current H4 {want} zones")
        except Exception:
            pass  # zone recompute failure: fail-open on this check
        # Distance: entry too far from current price (setup gone),
        # or price already through the entry (would fill
        # instantly at a bad price).
        if price is not None and atr > 0:
            if abs(float(price) - state.level) > 2.0 * atr:
                return ("distance",
                        f"entry {state.level:.2f} vs price "
                        f"{float(price):.2f} > 2.0x ATR")
            through = (
                (state.direction > 0 and state.kind == "limit"
                 and price <= state.level)
                or (state.direction > 0 and state.kind == "stop"
                    and price >= state.level)
                or (state.direction < 0 and state.kind == "limit"
                    and price >= state.level)
                or (state.direction < 0 and state.kind == "stop"
                    and price <= state.level))
            if through:
                return ("distance",
                        f"price {float(price):.2f} already "
                        f"through entry {state.level:.2f} — would "
                        f"fill instantly at a bad price")
        # Regime still supports the direction?
        setup = self._live_setup or {}
        regime = str(setup.get("regime", ""))
        if not regime or regime == "hold":
            return ("regime_flip",
                    "no live setup post-rebuild (HOLD)")
        if regime in ("trend", "breakout_trend"):
            if int(setup.get("direction", 0) or 0) != state.direction:
                return ("regime_flip",
                        f"rebuilt {regime} direction "
                        f"{setup.get('direction')} vs pending "
                        f"{state.direction:+d}")
        elif regime == "range":
            confirmed = set(
                (self._boundary_snapshot or {}).get(
                    "sweep_confirmed_edges") or [])
            if state.direction > 0 and "high" in confirmed:
                return ("regime_flip",
                        "rebuilt RANGE has SHORT sweep armed, "
                        "pending is LONG")
            if state.direction < 0 and "low" in confirmed:
                return ("regime_flip",
                        "rebuilt RANGE has LONG sweep armed, "
                        "pending is SHORT")
        # Score / manipulation re-evaluated on current bars.
        if h1 is not None and len(h1):
            floor = self.cfg.score_decay_floor \
                if self.cfg.score_decay_floor is not None \
                else self.cfg.min_score
            try:
                states = read_and_featurize(h1, None)
            except Exception:
                states = None
            if states is not None:
                current, crash = self._pending_current_score(
                    state,
                    {pb.name: pb for pb in self._get_playbooks()},
                    states, len(h1) - 1, h1)
                if crash is None and (current is None
                                      or current < floor):
                    return ("score_drop",
                            f"re-scored {current} vs floor "
                            f"{floor:.0f}")
            if price is not None and atr > 0:
                blocked, why = chase_blocked(
                    h1, state.direction, state.level, atr)
                if blocked:
                    return ("score_drop",
                            f"manipulation filter ON: {why}")
        return None

    # --------------------------------------- P3 bar-close management
    def _manage_positions_bar_close(self, stats: dict,
                                    clocks: tuple | None = None
                                    ) -> None:
        """Manage open positions like a trader, on bar closes.

        XD-4: a position is managed on its OWN clock's closes,
        not on every H1 close.  ``clocks`` narrows the pass to
        positions whose management clock is in the tuple (the
        M5/D1 refresh invocation); ``None`` (the H1-close pass)
        considers every position but still skips any whose
        clock has closed no new bar since its last pass — so a
        D1 swing is no longer drift-checked 24 times a day and
        an M5 scalp is checked on M5 closes, not hourly.
        Elapsed time is measured in management-clock bars and
        converted to H1 bars for the (H1-denominated) profile
        windows, so H1 positions behave byte-identically and
        threshold trigger TIMES are unchanged — only the check
        cadence moves to the owning clock.  Tick-level stop
        enforcement is untouched (it never lived here).
        """
        h1 = self._stack.frames.get("H1") if self._stack else None
        if h1 is None or not len(h1):
            return
        last_close = float(h1["close"].iloc[-1])
        atr_h1 = float(atr14(h1)) if len(h1) else 0.0
        mid = float((self._last_tick or {}).get("mid")
                    or last_close)
        for ticket in list(self.tick_manager.tickets()):
            pos = self.tick_manager.position(ticket)
            ctx = self._position_ctx.get(int(ticket))
            if pos is None or pos.closed or ctx is None:
                continue
            # XD-4 gate: this pass runs the position only on its
            # own clock's closes.
            clock_tf = str(ctx.get("management_clock")
                           or ctx.get("clock_tf") or "H1")
            if clocks is not None and clock_tf not in clocks:
                continue
            now_mgmt = self._mgmt_bar_now(clock_tf)
            if not _mgmt_clock_due(
                    clock_tf=clock_tf,
                    last_mgmt_bar=ctx.get("last_mgmt_bar"),
                    current_mgmt_bar=now_mgmt):
                continue
            ctx["last_mgmt_bar"] = now_mgmt
            profile = ctx["profile"]
            d = int(pos.direction)
            # XD-4 elapsed: management-clock bars -> H1 bars.
            # H1 positions: identical to the old cur_bar -
            # entry_bar (entry_mgmt_bar == entry_bar, now ==
            # cur_bar); M5/D1: wall-time-faithful conversion.
            elapsed = _mgmt_elapsed_h1(
                clock_tf=clock_tf,
                entry_mgmt_bar=int(ctx.get(
                    "entry_mgmt_bar", ctx["entry_bar"])),
                current_mgmt_bar=now_mgmt)
            # (1) defended stop: close through the defended pivot.
            defended = ctx.get("defended_stop")
            if defended is not None and atr_h1 > 0:
                if (last_close - float(defended)) * d \
                        < -0.1 * atr_h1:
                    res = self.executor.close_position(int(ticket))
                    if res.ok:
                        self._log_management(
                            ticket, pos, "defended_stop",
                            f"mt5 defended stop: ticket {ticket} "
                            f"H1 close {last_close:.2f} through "
                            f"the defended pivot {float(defended):.2f}"
                            " — the pullback thesis is done")
                        self._record_close(int(ticket), mid,
                                           "defended_stop", stats)
                    continue
            # (2) pre-print rule (S5).
            if self._preprint_for_position(ticket, pos, ctx, mid,
                                            atr_h1, stats):
                if self.tick_manager.position(ticket) is None \
                        or pos.closed:
                    continue
            # (3) progress stop (S10): trim once, exit on a
            # second full window of no progress.
            if _progress_due(profile, elapsed_h1_bars=elapsed,
                             best_r=float(pos.best_r)):
                if not ctx["progress_trimmed"]:
                    ctx["progress_trimmed"] = True
                    ctx["progress_trim_bar"] = elapsed
                    self._runner_partial(
                        ticket, pos, 0.5, "progress_trim", stats,
                        mid)
                    continue
                if ctx["progress_trim_bar"] is not None and \
                        elapsed - int(ctx["progress_trim_bar"]) \
                        >= profile.progress_h1_bars:
                    res = self.executor.close_position(int(ticket))
                    if res.ok:
                        self._log_management(
                            ticket, pos, "progress_exit",
                            f"mt5 progress exit: ticket {ticket} "
                            f"still < {profile.progress_floor_r}R "
                            f"after {elapsed} H1 bars — the tape "
                            "is not paying for this thesis (S10)")
                        self._record_close(int(ticket), mid,
                                           "progress_exit", stats)
                    continue
            # (4) drift exit (S10): re-score the position's own
            # direction; decay below the entry floor − margin,
            # with price NOT through invalidation, trims once and
            # exits on persistence.
            if ctx.get("entry_total") is not None:
                try:
                    now_score = self._score_for(
                        direction=d, verdict=None, price=mid,
                        zone_d=ctx.get("zone_d"), atr=atr_h1,
                        style_key=ctx["style_key"],
                        playbook=ctx["playbook"])
                    through = (last_close
                               - float(ctx["invalidation"])) * d < 0
                    if _drift_triggered(
                            profile, entry_floor=float(ctx["floor"]),
                            current_total=float(now_score.total),
                            through_invalidation=bool(through)):
                        if not ctx["drift_trimmed"]:
                            ctx["drift_trimmed"] = True
                            self._runner_partial(
                                ticket, pos, 0.5, "thesis_drift",
                                stats, mid)
                        else:
                            res = self.executor.close_position(
                                int(ticket))
                            if res.ok:
                                self._log_management(
                                    ticket, pos, "thesis_drift",
                                    f"mt5 drift exit: ticket "
                                    f"{ticket} re-score "
                                    f"{now_score.total:.0f} < entry "
                                    f"floor {ctx['floor']:.0f} − "
                                    f"{profile.drift_margin:.0f} — "
                                    "the premise decayed (S10)")
                                self._record_close(
                                    int(ticket), mid,
                                    "thesis_drift", stats)
                        continue
                except Exception as exc:
                    _log.warning("mt5 drift read failed for "
                                 "ticket %s: %s", ticket, exc)
            # (5) bank into the opposing zone (A8): price reaches
            # the magnet — take half off at the level the ladder
            # named, once.
            prox = ctx.get("opposing_prox")
            if prox is not None and not ctx["zone_banked"] \
                    and atr_h1 > 0:
                if abs(mid - float(prox)) <= 0.25 * atr_h1 \
                        or (mid - float(prox)) * d > 0:
                    ctx["zone_banked"] = True
                    self._runner_partial(
                        ticket, pos, 0.5, "tp_zone", stats, mid)
                    continue
            # (6) ADR exhaustion bank (S8): the day's fuel is
            # spent and the position is a runner in profit — bank
            # half into the exhaustion rather than gifting it back.
            day = self._day_ctx
            if day is not None and not ctx["adr_banked"] \
                    and day.adr_days >= 10 \
                    and day.adr_consumed_pct >= 100.0 \
                    and float(pos.best_r) >= 1.0:
                ctx["adr_banked"] = True
                self._runner_partial(
                    ticket, pos, 0.5, "adr_bank", stats, mid)
                continue

    def _preprint_for_position(self, ticket, pos, ctx, mid,
                               atr_h1, stats) -> bool:
        """S5 pre-print rule for ONE position.  Returns True when
        it acted.  Into an EXTREME title whose window opens within
        the lead: intraday exposure is secured (stop to breakeven
        and/or a trim, per config); a swing position whose stop
        already sits beyond 2x the event's typical range is
        exempt — journaled either way, once per event."""
        if not self.cfg.news_pause_enabled:
            return False
        try:
            nxt = self.news.next_event(self._tick_datetime())
        except Exception:
            return False
        if nxt is None:
            return False
        ev, mins = nxt
        from ..live.news import _is_extreme as _is_extreme_title
        if not _is_extreme_title(ev.title) \
                or mins > self.news.PREPRINT_LEAD_MIN \
                or mins < -45.0:
            return False
        key = f"{ev.title}@{ev.time.isoformat()}"
        if ctx.get("preprint_key") == key:
            return False
        ctx["preprint_key"] = key
        # Typical event range proxy: 3x H1 ATR (a red print's
        # first-hour range on gold runs ~2-4 H1 ATRs; documented
        # proxy — the calendar carries no per-event range).
        event_range = 3.0 * atr_h1 if atr_h1 > 0 else 0.0
        eff_stop = self.tick_manager.effective_stop(int(ticket)) \
            or float(ctx["invalidation"])
        stop_distance = abs(float(pos.entry_price)
                            - float(eff_stop))
        action = _preprint_action(
            style_family=("swing" if ctx["style_key"] == "swing"
                          else "intraday"),
            stop_distance=stop_distance,
            event_range=event_range)
        if action == "exempt":
            self._log_management(
                ticket, pos, "pre_print_exempt",
                f"mt5 pre-print: ticket {ticket} EXEMPT from "
                f"securing into {ev.title} — swing stop "
                f"{stop_distance:.1f} beyond 2x the event range "
                f"({event_range:.1f}) (S5)")
            return True
        mode = str(self.cfg.news_preprint_action or "be")
        acted = False
        if mode in ("trim", "be_trim"):
            acted = self._runner_partial(
                ticket, pos, 0.5, "pre_print_trim", stats, mid) \
                or acted
        if mode in ("be", "be_trim") and \
                self.tick_manager.is_fixed_bracket(int(ticket)):
            # Step 13: fixed-bracket positions keep their placement
            # stop into the print — no breakeven securing.  Journaled
            # (never silent) via the management log.
            self._log_management(
                ticket, pos, "pre_print_be_skipped",
                f"mt5 pre-print: ticket {ticket} stop NOT secured "
                f"to breakeven into {ev.title} — fixed-bracket "
                f"playbook (step 13)")
        elif mode in ("be", "be_trim"):
            # Stop to breakeven via the ordinary action path (the
            # tick manager books it, the pool syncs in _manage_tick
            # style below).
            from .tick_manager import TickAction as _TA
            act = _TA("modify_sl", level=float(pos.entry_price),
                      reason="breakeven")
            res = self.executor.modify_sl(int(ticket),
                                          float(pos.entry_price))
            if res.ok:
                self.tick_manager.apply(int(ticket), act)
                pool = self._pools.get(int(ticket))
                if pool is not None:
                    try:
                        pool.move_stop(str(pos.alert_id),
                                       float(pos.entry_price))
                    except Exception:
                        pass
                self._log_management(
                    ticket, pos, "pre_print_be",
                    f"mt5 pre-print: ticket {ticket} stop secured "
                    f"to breakeven into {ev.title} (window opens "
                    f"in {mins:.0f}m) (S5)")
                acted = True
        return acted

    def _on_h1_close(self, tick: dict, stats: dict) -> None:
        """An H1 close: time stop + pending expiry + forming reset.

        The close refreshes legs (done in _refresh_bars before this
        call) — it never gates evaluation.
        """
        h1 = self._stack.frames["H1"]
        last_bar_index = len(h1) - 1
        # Regime-dependent time exit: bedrock regime of the H1 frame
        # sets each position's horizon (classification failure falls
        # back to the fixed horizon — never breaks the close).
        regime_structure = None
        try:
            regime_structure = classify_regime(h1).structure
        except Exception as exc:
            self.journal.log_note(
                f"mt5: regime classify failed, fixed horizon: {exc}")
        # Time stop for tracked positions (bar-defined rule).
        # XD-4: slow-clock positions (D1/H4) get their horizon
        # timeout evaluated on their OWN clock's closes only —
        # the H1-bar horizon VALUE is unchanged, only the
        # evaluation cadence moves.  Due-ness is per CLOCK per
        # close (all D1 tickets share one D1 evaluation per
        # D1 close), never per ticket.  Fast clocks evaluate
        # as today (M5-close timeout evaluation needs
        # M5-indexed signal bars in the tick manager — named
        # XD remainder).
        _timeout_due: dict[str, bool] = {}
        for _clk in ("D1", "H4"):
            _tframe = (self._stack.frames.get(_clk)
                       if self._stack else None)
            _tlen = len(_tframe) if _tframe is not None else 0
            _timeout_due[_clk] = (
                self._timeout_seen_bars.get(_clk) != _tlen)
            if _timeout_due[_clk]:
                self._timeout_seen_bars[_clk] = _tlen
        for ticket in list(self.tick_manager.tickets()):
            _tctx = self._position_ctx.get(int(ticket))
            _tclock = str((_tctx or {}).get("management_clock")
                          or (_tctx or {}).get("clock_tf")
                          or "H1")
            if _tclock in ("D1", "H4") \
                    and not _timeout_due.get(_tclock, True):
                continue
            for action in self.tick_manager.on_bar_close(
                    ticket, last_bar_index, regime_structure):
                res = self.executor.close_position(ticket, action.volume)
                if res.ok:
                    self._record_close(
                        ticket,
                        self._last_tick["mid"] if self._last_tick else 0.0,
                        "timeout", stats)
        # P3 bar-close management (S5/S8/S10 + defended stops):
        # progress / drift / pre-print / zone + ADR banking.
        # Runs AFTER the time stop (a timed-out position is gone)
        # and never breaks the close.
        try:
            self._manage_positions_bar_close(stats)
        except Exception as exc:
            _log.warning("mt5 bar-close management failed: %s",
                         exc)
        # Pending-order expiry: unfilled past expiry_bar dies — the
        # setup went stale (a real trader does not leave a limit
        # resting at a dead zone).
        # Score decay runs BEFORE expiry: a setup that died on
        # re-evaluation is cancelled as score_decay (more
        # informative than expired), and decay at 2 bars fires
        # before the 4-bar expiry anyway.
        self._check_pending_score_decay(stats)
        for ticket, state in list(self._pending_orders.items()):
            if state.premise == "native":
                # Step 4: native pendings expire on their own clock
                # (_check_native_pendings, called from the native
                # routes) — the H1-close loop must not kill them on
                # H1 units, or entry-clock expiry is dead code.
                continue
            if last_bar_index >= state.expiry_bar:
                if state.ladder_levels and state.ladder_lots:
                    # Ladder: this rung expired — cancel it and
                    # advance the next rung (one live pending at a
                    # time).  Score decay already ran first: a dead
                    # setup is cancelled as score_decay, never
                    # advanced.  A failed cancel keeps the rung
                    # (do not place a second pending).
                    next_level = state.ladder_levels.pop(0)
                    next_lots = state.ladder_lots.pop(0)
                    if self._cancel_pending(
                            ticket,
                            f"expired: no fill after "
                            f"{self.cfg.pending_expiry_bars} H1 bars "
                            f"— ladder advancing to {next_level:.2f}",
                            stats):
                        self._place_ladder_rung(
                            state, next_level, next_lots,
                            last_bar_index, stats)
                else:
                    self._cancel_pending(
                        ticket,
                        f"expired: no fill after "
                        f"{self.cfg.pending_expiry_bars} H1 bars",
                        stats)
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

    def _check_native_pendings(self, clock: str,
                               stats: dict) -> None:
        """Native-clock pending guards (step 4), run on native closes.

        Called from the native routes (``_run_qm_m5`` / ``_run_qm_d1``)
        before candidate evaluation, so an expired native pending
        never blocks the one-position rule.  Two guards, both
        counted on the entry clock:

        * expiry — the closed entry-clock index reached
          ``entry_expiry_bar`` (entry-clock bars since placement,
          wall duration preserved from ``pending_expiry_bars``);
        * invalidation — any closed entry-clock bar since placement
          traded through the structural invalidation.  In
          continuous trading a fill precedes such a breach (price
          must cross the entry to reach the invalidation), so a
          still-resting pending past its invalidation is a dead
          setup (gap / fill miss), never a live one.

        A guard failure never crashes the route: it is journalled
        and the pending is left for the next close.  H1 pendings
        are untouched here — the H1-close loop governs them.
        """
        if self._stack is None:
            return
        frame = self._stack.frames.get(clock)
        if frame is None or len(frame) == 0:
            return
        idx = len(frame) - 1
        for ticket, state in list(self._pending_orders.items()):
            if state.entry_clock != clock:
                continue
            try:
                if idx >= state.entry_expiry_bar:
                    self._cancel_pending(
                        ticket,
                        f"expired: no fill after "
                        f"{state.entry_expiry_bar - state.entry_placed_bar} "
                        f"{clock} bars",
                        stats)
                    continue
                inv = float(state.invalidation or 0.0)
                if inv and "high" in frame.columns \
                        and "low" in frame.columns:
                    try:
                        bars = frame.iloc[state.entry_placed_bar:]
                    except Exception:
                        bars = frame.iloc[0:0]
                    if len(bars):
                        if state.direction > 0:
                            breached = bool((bars["low"] < inv).any())
                        else:
                            breached = bool((bars["high"] > inv).any())
                        if breached:
                            self._cancel_pending(
                                ticket,
                                f"invalidated: price through "
                                f"invalidation {inv:.2f} on {clock} "
                                f"close",
                                stats)
            except Exception as exc:
                self.journal.log_note(
                    f"mt5: native pending check failed for ticket "
                    f"{ticket} (pending left alone): {exc}")

    def _check_pending_bias(self, stats: dict) -> None:
        """D1/H4 close flipped the bias (or un-armed the setup): the
        pending trigger's premise is gone — invalidate, never erase.
        Pending ORDERS die the same way: a bias flip / un-armed
        setup cancels every resting order whose premise is gone.

        Step 4 premise scope: this check governs ONLY H1-premise
        orders.  ``_live_setup`` is armed solely by the H1 trigger
        path — native routes never arm it — so applying this check
        to a native pending cancels it on every D1/H4 close for a
        premise it never had (2026-10-26: qm_m5 ticket 514827226
        cancelled 12:00:27 "bias_flipped: setup unarmed" five
        minutes after placement, never filled).  Native pendings
        are skipped here and guarded on their own clock (native
        expiry / invalidation, step 4).  The setup-None reason is
        renamed ``setup_unarmed``: it is not a bias flip, it is an
        un-armed H1 premise, and the old wording hid that
        distinction in the journal.
        """
        setup = self._live_setup
        if self._pending is not None:
            if setup is None \
                    or setup["direction"] != self._pending["direction"]:
                self._invalidate_pending("bias_flipped", stats)
        if self._pending_orders:
            if setup is None:
                for ticket, state in list(
                        self._pending_orders.items()):
                    if state.premise != "h1":
                        self.journal.log_note(
                            f"mt5: bias check skipped ticket {ticket} "
                            f"({state.playbook}): native premise — "
                            f"_live_setup never arms native routes")
                        continue
                    self._cancel_pending(
                        ticket, "setup_unarmed", stats)
            else:
                for ticket, state in list(
                        self._pending_orders.items()):
                    if state.premise != "h1":
                        self.journal.log_note(
                            f"mt5: bias check skipped ticket {ticket} "
                            f"({state.playbook}): native premise — "
                            f"_live_setup never arms native routes")
                        continue
                    if state.direction != setup["direction"]:
                        self._cancel_pending(
                            ticket, "bias_flipped", stats)

    def _invalidate_pending(self, reason: str, stats: dict) -> None:
        """Drop a pending trigger: logged, never erased.

        Non-confirmation is not repainting — the armed touch / fired
        trigger stays in the journal; only the pending state is
        cleared.
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
        # FUNNEL §1.1: the disposition is a funnel record too —
        # an armed plan's fate must be queryable, not prose-only.
        _stage, _code = {
            "stale": ("expired", "path.trigger_not_fired"),
            "h1_close_without_entry_confirm": (
                "expired", "path.trigger_not_fired"),
            "invalidation": ("invalidated",
                             "path.invalidated_close_rule"),
            "bias_flipped": ("invalidated", "path.bias_flip"),
        }.get(reason, ("invalidated", "path.invalidated"))
        self._emit_trigger_event(
            _stage, _code,
            f"{reason}: {pend['trigger']} "
            f"{'+' if pend['direction'] > 0 else '-'}1 "
            f"@ {pend['entry']:.2f}")

    def _disarm_stale(self, stats: dict) -> None:
        """TTL expiry: the armed touch lingered without confirmation.

        Owner rule (2026-10-15): a zone price lingers in without
        reacting is being absorbed — its edge decays, it doesn't
        grow; a confirmation firing hours later would trigger on
        stale context.  Journaled as ``stale`` (distinct from
        exit/invalidation), and re-arm is blocked until price
        EXITS this zone and re-enters.
        """
        pend = self._pending
        if pend is None or not pend.get("armed"):
            return
        zone = pend["zone"]
        self._stale_zone = (pend["direction"], zone["bottom"],
                            zone["top"])
        self._invalidate_pending("stale", stats)

    def _invalidate_close_rule(self, stats: dict) -> None:
        """Close-Rule invalidation of the armed touch: an entry-TF
        bar CLOSED beyond the invalidation level — the setup's
        premise is gone (the zone failed).  Terminates monitoring
        for THIS setup: the zone is blocked until a different POI
        appears (see _invalidated_zone) — unlike the stale block,
        exit + re-entry does NOT revive a failed zone.  Zero
        hesitation: no waiting for another bar, no hoping for a
        bounce.
        """
        pend = self._pending
        if pend is None or not pend.get("armed"):
            return
        zone = pend["zone"]
        self._invalidated_zone = (pend["direction"], zone["bottom"],
                                  zone["top"])
        # Boundary cooldown: a Close-Rule invalidation on a range
        # edge cools that edge for 2 H1 bars (BoundaryTracker).
        if self._boundary is not None:
            try:
                setup = pend.get("setup") or {}
                if setup.get("regime") == "range":
                    ev = self._boundary.note_invalidation(
                        (zone or {}).get("side"))
                    if ev is not None:
                        _log.info("mt5 boundary: %s (%s) — %s",
                                  ev.kind, ev.edge, ev.detail)
            except Exception:
                pass
        self._invalidate_pending("invalidation", stats)

    @staticmethod
    def _entry_clock_for(verdict, tick_time) -> str:
        """Entry clock for a pre-entry entity: the verdict's clock
        when known, else the session clock at this tick (the state
        machine runs on the entry clock)."""
        clock = (verdict or {}).get("entry_clock")
        if clock:
            return str(clock)
        return _compose_session(pd.Timestamp(tick_time))["entry_clock"]

    def _drop_staged_confirm(self, key: str, payload: dict,
                             stats: dict) -> None:
        """Drop a staged Golden Swing confirm on Close-Rule
        invalidation: the confirmed setup died before entry, so
        the order NEVER places.  Both the payload and its Confirm
        token die (a tap on the dead token reports expired)."""
        self._order_confirms.pop(key, None)
        for token, staged in list(self._confirm_tokens.items()):
            if staged.get("action") == "place_order" \
                    and staged.get("arg") == key:
                self._confirm_tokens.pop(token, None)
        sig = payload.get("sig")
        self.journal.log_note(
            "mt5: Golden Swing order invalidated — setup failed "
            "the Close Rule before entry, no order placed "
            f"({getattr(sig, 'direction', '?'):+} "
            f"invalidation {float(getattr(sig, 'invalidation', 0.0) or 0.0):.2f})")

    def _check_close_rule(self, tick: dict, appended: list[str],
                          appended_counts: dict[str, int],
                          stats: dict) -> None:
        """The Close Rule (owner, 2026-10-15): on EVERY entry-TF
        bar close, exactly one check — the bar's ``close`` vs the
        invalidation level.  A WICK beyond the level does NOT
        invalidate (the expected stop-hunt); a CLOSE beyond
        invalidates instantly.  Applies to every pre-entry entity:
        the armed touch, resting pending orders, staged Golden
        Swing confirms.  Once a position is open the stop loss
        owns it — never touched here.  This is the ONE
        invalidation condition: no tick-level, no second
        condition.
        """
        if self._stack is None:
            return
        tick_time = pd.Timestamp(tick["time"])
        for tf in appended:
            frame = self._stack.frames.get(tf)
            count = appended_counts.get(tf, 0)
            if frame is None or count <= 0 or len(frame) < count:
                continue
            closes = frame["close"].iloc[-count:]
            # Armed touch: close beyond its invalidation level.
            pend = self._pending
            if pend is not None and pend.get("armed") \
                    and pend.get("entry_clock") == tf:
                inv = pend.get("invalidation_level")
                if inv is not None:
                    for close in closes:
                        if (pend["direction"] > 0
                                and close < inv) or \
                                (pend["direction"] < 0
                                 and close > inv):
                            self._invalidate_close_rule(stats)
                            break
            # Resting pending orders: close beyond the order's
            # (playbook) invalidation — cancel immediately.
            for ticket, state in list(self._pending_orders.items()):
                if self._entry_clock_for(state.verdict,
                                         tick_time) != tf:
                    continue
                inv = state.invalidation
                for close in closes:
                    if (state.direction > 0 and close < inv) or \
                            (state.direction < 0 and close > inv):
                        self._cancel_pending(
                            ticket,
                            f"invalidated: {tf} close {close:.2f} "
                            f"beyond invalidation {inv:.2f}", stats)
                        break
            # Staged Golden Swing confirms (confirmed, not yet
            # placed): close beyond the signal's invalidation —
            # the order never places.
            for key, payload in list(self._order_confirms.items()):
                sig = payload.get("sig")
                if sig is None:
                    continue
                if self._entry_clock_for(payload.get("verdict"),
                                         tick_time) != tf:
                    continue
                inv = float(getattr(sig, "invalidation", 0.0) or 0.0)
                for close in closes:
                    if (int(sig.direction) > 0 and close < inv) or \
                            (int(sig.direction) < 0 and close > inv):
                        self._drop_staged_confirm(key, payload, stats)
                        break

    # --------------------------------------- PullbackWatch (§5)
    #: P3: PullbackWatch slot ceiling — several setups may park
    #: at once, but the book stays readable.
    _WATCH_SLOTS = 6

    def _enter_pullback_watch(self, pb, sig, verdict, zone_d,
                              score, funding) -> None:
        """Park a runaway setup in PullbackWatch (spec §5): the
        DO NOT CHASE reset is gone on the arrival path — the setup
        is not dismantled, the watch awaits the pullback for up to
        ``pullback_watch_bars`` H1 bars, dying early only on a
        close through the zone's far edge (the idea is dead, not
        merely late).  P3: MULTI-SLOT — each parked setup carries
        its own zone, expiry and invalidation; a re-print of the
        same setup refreshes its slot instead of stacking."""
        key = (pb.name, int(sig.direction),
               round(float(zone_d.get("level", 0.0)), 1))
        for w in self._pullback_watches:
            if w["key"] == key:
                w["bar0"] = self._current_h1_bar()
                w["score_total"] = float(score.total)
                return
        if len(self._pullback_watches) >= self._WATCH_SLOTS:
            self.journal.log_note(
                f"mt5 watch: {pb.name} runaway NOT parked — all "
                f"{self._WATCH_SLOTS} watch slots are held")
            return
        self._pullback_watches.append({
            "key": key,
            "direction": int(sig.direction),
            "zone": dict(zone_d),
            "playbook": pb.name,
            "bar0": self._current_h1_bar(),
            "score_total": float(score.total),
        })
        self.journal.log_note(
            f"mt5 watch: {pb.name} "
            f"{'+' if sig.direction > 0 else '-'}1 runaway from "
            f"{zone_d.get('tf', '?')} zone "
            f"{float(zone_d['bottom']):.1f}-{float(zone_d['top']):.1f} "
            f"— PullbackWatch armed for "
            f"{self.cfg.pullback_watch_bars} H1 bars (asym "
            f"{score.total:.0f}); the setup is parked, not reset")

    def _maintain_pullback_watch(self, mid: float) -> None:
        """Per-tick watch upkeep, per slot: pullback arrival
        clears that slot (the live setup re-arms through the
        normal trigger machinery — nothing was dismantled); a
        close through the far edge kills that slot; the bar
        budget lapses it.  A filled position clears every slot —
        the watches were alternatives to being in."""
        if not self._pullback_watches:
            return
        if self.tick_manager.tickets():
            self._pullback_watches = []
            self.journal.log_note(
                "mt5 watch: cleared — a position is open, the "
                "watches are moot")
            return
        survivors = []
        for w in self._pullback_watches:
            zone = w["zone"]
            d = int(w["direction"])
            if float(zone["bottom"]) <= mid <= float(zone["top"]):
                self.journal.log_note(
                    f"mt5 watch: pullback arrived — price back "
                    f"inside the {zone.get('tf', '?')} zone "
                    f"{float(zone['bottom']):.1f}-"
                    f"{float(zone['top']):.1f} ({w['playbook']}); "
                    "watch cleared, the armed setup owns the "
                    "entry now")
                continue
            # Invalidation: an H1 CLOSE beyond the far edge + buf.
            dead = False
            try:
                h1 = (self._stack.frames.get("H1")
                      if self._stack is not None else None)
                atr = float(zone.get("atr")
                            or self._last_zone_atr or 0.0)
                if h1 is not None and len(h1) and atr > 0:
                    last_close = float(h1["close"].iloc[-1])
                    buf = 0.25 * atr
                    dead = (last_close < float(zone["bottom"]) - buf) \
                        if d > 0 else \
                        (last_close > float(zone["top"]) + buf)
            except Exception:
                dead = False
            if dead:
                self.journal.log_note(
                    f"mt5 watch: KILLED ({w['playbook']}) — H1 "
                    "close through the zone's far edge; the idea "
                    "is dead, not late (§5 probe-death rule)")
                continue
            if self._current_h1_bar() - int(w["bar0"]) \
                    > int(self.cfg.pullback_watch_bars):
                self.journal.log_note(
                    f"mt5 watch: lapsed ({w['playbook']}) after "
                    f"{self.cfg.pullback_watch_bars} H1 bars "
                    "without a pullback")
                continue
            survivors.append(w)
        self._pullback_watches = survivors

    # --------------------------------------- resting limit (A6)
    def _maybe_place_resting_limit(self, tick: dict,
                                   stats: dict) -> None:
        """A6: when the armed POI is a FRESH MAJOR zone, rest a
        limit at its proximal edge AHEAD of arrival — subject to
        the score floor, the RiskPool budget logic (placement-time
        sizing), RiskGuards and the margin check.  Journaled with
        trigger_source=resting_limit, playbook 'arrival_zone' (the
        decision flow is the actor; no playbook fired)."""
        if not self.cfg.resting_limit_enabled:
            return
        if self.cfg.paused or self._regime_pause is not None:
            return
        # P3 posture hook (S1): resting zone interest survives
        # postures that shelve NEW mid-range entries (compression
        # explicitly keeps it live) — only a posture that closes
        # resting interest itself suppresses placement.
        if self._posture is not None \
                and not self._posture.resting_interest_live:
            return
        # A resting order that filled / cancelled / expired frees
        # its dedupe key: a later identical setup may rest again.
        if self._resting_limit_key is not None \
                and not self._pending_orders \
                and not self.tick_manager.tickets():
            self._resting_limit_key = None
        if self.tick_manager.tickets() or self._pending_orders:
            return
        if self._pending is not None:
            return  # a touch is already armed — arrival path owns it
        setup = self._live_setup
        if not setup or setup.get("regime") not in (None, "trend"):
            return
        if setup.get("arm") == "continuation":
            return  # retest bands are not graded POIs
        direction = int(setup.get("direction", 0) or 0)
        if direction == 0:
            return
        major = setup.get("major_zone")
        if not major:
            return
        if str(major.get("lifecycle") or "") != "fresh" \
                or str(major.get("tier") or "") != "MAJOR":
            return
        side = str(major.get("side", ""))
        if (direction > 0) != (side == "support"):
            return
        mid = float(tick["mid"])
        atr = float(setup.get("atr_h1") or self._last_zone_atr or 0.0)
        if atr <= 0:
            return
        prox = float(major.get("proximal")
                     or (major["top"] if direction > 0
                         else major["bottom"]))
        distal = float(major.get("distal")
                       or (major["bottom"] if direction > 0
                           else major["top"]))
        # Ahead of arrival only: approach side, within 2 ATR,
        # not yet at/inside the band, not through it.
        if direction > 0:
            if mid <= prox or mid - prox > 2.0 * atr:
                return
        else:
            if mid >= prox or prox - mid > 2.0 * atr:
                return
        key = f"{major.get('zone_id') or major.get('level')}:{direction}"
        if self._resting_limit_key == key:
            return
        zone_d = dict(major)
        # The score prices the TRADE, whose entry is the proximal
        # edge — not the current tick (evaluating location at the
        # tick would make any approach beyond 0.5 ATR
        # mathematically unauthorizable, a dead feature).
        score = self._score_for(direction=direction, verdict=None,
                                price=prox, zone_d=zone_d, atr=atr)
        fl = floor_for("ZONE_PATH")
        if score.total < float(fl["floor"]) \
                or score.component("location") < float(
                    fl.get("min_location", 0.0)):
            if getattr(self, "_resting_skip_key", None) != key:
                self._resting_skip_key = key
                self.journal.log_note(
                    f"mt5 resting limit: NOT placed at fresh MAJOR "
                    f"{major.get('tf', '?')} {prox:.1f} — asymmetry "
                    f"{score.total:.0f} below the zone-path floor "
                    f"({fl['floor']:.0f} + location "
                    f"{fl.get('min_location', 0.0):.0f})")
            return
        buf = float(setup.get("buf") or 0.25 * atr)
        invalidation = distal - buf if direction > 0 \
            else distal + buf
        entry = prox
        risk = abs(entry - invalidation)
        if risk <= 0:
            return
        opp = self._opposing_dict(direction)
        target = float(opp["level"]) if opp else \
            entry + direction * 2.0 * risk
        try:
            spec = self.executor.symbol_spec(self.cfg.symbol)
            acct = self.feed.account_state()
        except Exception:
            return
        stop0 = TickManager.planned_stop(
            OrderRequest(symbol=self.cfg.symbol, direction=direction,
                         volume=0.0, stop_loss=invalidation, atr=atr))
        bar = self._current_h1_bar()
        alert_id = f"MT5-arrival_zone-{bar}"
        detail = None
        if self._sizing_mode() == "risk":
            # Same formula as the base-order risk path (equity x
            # k x C x traffic tier), C held at x1.0 — no honest
            # confidence field exists at order time; the old
            # score band is REMOVED (owner ruling 2026-10-24).
            # Below-minimum floors to the broker minimum with
            # min_override flagged (see size_for_risk).
            _tier, tier_mult, _ceil = self._traffic(zone_d)
            _k = float(self.cfg.risk_pct)
            _confidence_mult = 1.0  # unresolved — see base path
            risk_money = float(acct.get("equity", 0.0)) \
                * _k * _confidence_mult * tier_mult
            _stop_dist = abs(entry - stop0)
            result = size_for_risk(
                risk_money, _stop_dist, spec)
            if not result.ok:
                self.journal.log_note(
                    f"mt5 resting limit: DECLINED sizing "
                    f"{result.verdict} at fresh MAJOR "
                    f"{major.get('tf', '?')} {prox:.1f} "
                    "(volume-limit / nonpositive refusal only — "
                    "below-minimum now floors to the minimum)")
                return
            lots = float(result.lots)
            detail = {"authorized_risk": float(result.risk_money),
                      "realized_risk": float(result.realized_risk),
                      "estimated_stop_exposure":
                          float(result.estimated_stop_exposure),
                      "drag": float(result.drag),
                      "raw_lots": float(result.raw_lots),
                      "lots": lots,
                      "min_override": bool(result.min_override),
                      "sizing_coefficient": _k,
                      "confidence_mult": _confidence_mult,
                      "scalars": {
                          "traffic_tier": _tier,
                          "traffic_mult": float(tier_mult)},
                      "spec": {
                          "symbol": self.cfg.symbol,
                          "tick_size": float(spec.tick_size),
                          "tick_value": float(spec.tick_value),
                          "volume_min": float(spec.volume_min),
                          "volume_step": float(spec.volume_step),
                          "volume_max": float(spec.volume_max)},
                      "stop_distance": float(_stop_dist)}
        else:
            # ENFORCEMENT (owner rulings): a non-risk sizing mode may
            # evaluate but NEVER places an order. A stale config
            # selecting legacy mode refuses here, loudly, instead of
            # silently restoring retired behavior.
            self.journal.log_note(
                "mt5 resting limit REFUSED: sizing_mode is not "
                "'risk' — legacy sizing cannot place orders "
                "(evaluation/forensics only)")
            return
        from types import SimpleNamespace
        pb_stub = SimpleNamespace(name="arrival_zone")
        sig_stub = SimpleNamespace(
            direction=direction, entry_ref=entry,
            invalidation=invalidation, targets=[target], atr=atr,
            index=bar, score=float(score.total),
            factors=["resting_limit"], kind="")
        if not self._check_risk_guards(pb_stub, sig_stub, lots,
                                       stop0, acct):
            return
        if not self._check_margin(pb_stub, sig_stub, lots, False,
                                  acct):
            return
        request = OrderRequest(
            symbol=self.cfg.symbol, direction=direction,
            volume=lots, stop_loss=float(stop0),
            entry_ref=entry, take_profit=target,
            targets=[target], atr=atr, risk_money=0.0,
            alert_id=alert_id, bar_index=bar,
            magic=self.cfg.magic, order_kind="limit",
            order_price=entry)
        track_request = OrderRequest(
            symbol=self.cfg.symbol, direction=direction,
            volume=lots, stop_loss=float(invalidation),
            entry_ref=entry, targets=[target], atr=atr,
            bar_index=bar, alert_id=alert_id, magic=self.cfg.magic)
        verdict_stub = {
            "zone": zone_d, "direction": direction,
            "trigger": "resting_limit",
            "bias_d1": setup.get("bias_d1"),
            "bias_h4": setup.get("bias_h4"),
            "entry_clock": "H1", "amd_phase": None, "qm": {},
            "volume": None, "dxy_corr": None, "urgency": None}
        placed = self._place_pending_order(
            pb_stub, sig_stub, verdict_stub, "limit", entry,
            request, track_request, stats, bar)
        if placed:
            self._resting_limit_key = key
            # P3: park the decision context so the FILL registers
            # a full management context (arrival_zone, zone path).
            self._signal_ctx[alert_id] = {
                "playbook": "arrival_zone",
                "style_key": self._style_key_for(
                    "arrival_zone", zone_d),
                "style_bucket": self._style_bucket(
                    self._style_key_for("arrival_zone", zone_d)),
                "family": "zone", "floor": float(fl["floor"]),
                "path": "ZONE_PATH",
                "entry_total": float(score.total),
                "zone_d": dict(zone_d), "clock_tf": "H1",
                "defended_stop": None}
            if detail is not None:
                self._sizing_detail[alert_id] = detail
            self.journal.log_note(
                f"mt5 resting limit: LIMIT "
                f"{'+' if direction > 0 else '-'}1 {lots} lots @ "
                f"{entry:.2f} (proximal edge of fresh MAJOR "
                f"{major.get('tf', '?')} zone "
                f"{float(major['bottom']):.1f}-"
                f"{float(major['top']):.1f}) SL {stop0:.2f} — "
                f"asym {score.total:.0f}, source resting_limit")
            log_signal(
                self.journal, signal_id=alert_id,
                playbook="arrival_zone", direction=direction,
                score=float(score.total), decision="accepted",
                reason="resting limit at fresh MAJOR proximal edge "
                       "(A6)",
                entry=entry, stop=invalidation, targets=[target],
                score_breakdown=score.breakdown(),
                trigger_source="resting_limit",
                context=self._decision_context())

    def _run_hot_scalp_shadow(self, tick: dict) -> list[dict]:
        """Observe the separate Hot Scalp sleeve without granting order authority.

        Returns the engine's events (step 6: the live loop feeds
        candidate_shadow events to the flag-gated promotion path;
        returning them changes no existing caller, which ignored
        the previous None).
        """
        engine = getattr(self, "_hot_scalp", None)
        stack = getattr(self, "_stack", None)
        if engine is None or stack is None:
            return []
        try:
            _scalp_frames = dict(stack.frames or {})
            _scalp_frames["M1"] = self._hot_scalp_aux_frames.get("M1")
            events = engine.on_tick(
                symbol=str(getattr(self.cfg, "symbol", "XAUUSD")),
                tick=tick, frames=_scalp_frames, now=tick.get("time"))
        except Exception as exc:  # safety boundary: research cannot break main path
            events = [{"event": "scan_degraded",
                       "reason_code": "hot_scalp.runner_adapter_error",
                       "reason": f"{type(exc).__name__}: {exc}"}]
        for event in events:
            name = str(event.get("event", "unknown"))
            # A stale feed is a state, not a per-tick event. Report once per
            # unchanged M1 close; reset after the sleeve recovers.
            if name == "scan_degraded":
                m1 = self._hot_scalp_aux_frames.get("M1")
                key = f"{event.get('reason_code')}:{m1.index[-1] if m1 is not None and len(m1) else 'missing'}"
                if key == self._hot_scalp_last_degraded_key:
                    continue
                self._hot_scalp_last_degraded_key = key
            else:
                self._hot_scalp_last_degraded_key = None
            record = {
                "type": "hot_scalp_event",
                "run_id": getattr(self, "_run_id", None),
                "commit": self._build_commit(),
                "timestamp": str(tick.get("time")),
                "symbol": str(getattr(self.cfg, "symbol", "XAUUSD")),
                "event": name,
                "reason_code": event.get("reason_code"),
                "execution_authority": "shadow_only",
                "research_only": True,
                "details": event,
            }
            try:
                self.journal.append(record)
            except Exception as exc:
                # If journaling fails, this sleeve remains non-executable.
                _log.error("hot_scalp: journal write failed: %s", exc)
            if name in ("setup_armed", "candidate_shadow", "candidate_blocked",
                        "setup_invalidated", "setup_expired", "scan_degraded"):
                try:
                    self.journal.log_note(
                        f"hot_scalp {name}: "
                        f"{event.get('reason_code') or event.get('setup', {}).get('strategy', '')} "
                        f"{event.get('setup_id') or event.get('setup', {}).get('setup_id', '')} "
                        f"— shadow only")
                except Exception:
                    pass
        return events

    def _promote_hot_scalp_candidate(self, candidate: dict, tick: dict,
                                     stats: dict) -> bool:
        """Promote one shadow candidate to the SHARED order path (step 6).

        Gated on ``hot_scalp_live_enabled`` (default ON per owner
        order 2026-10-27; config-file/env can disable it); the flag
        can never be flipped from chat.
        Every refusal is journalled with a reason code, and the
        claim's fate follows the refusal kind:

        - transient refusals (paused, spread widened, order-path
          refusal with nothing placed) resolve the claim back to
          "touched" via an authoritative False — the setup stays
          live and may re-trigger;
        - disagreement/ambiguity (broker shows an untracked magic
          position, broker lookup failed, fill recorded but broker
          does not confirm it) LEAVES the claim claimed — fail
          closed; the setup expires on its TTL if nothing resolves
          it, and V2 reconciliation remains the only release.
        - success resolves to "order_reconciled" only after the
          broker itself shows the new magic position.

        v1 scope, stated honestly: flat-only (a candidate while
        holding is refused — scalp layering needs the parent
        identity mapping between hot-scalp setup IDs and live
        alert IDs, which does not exist yet) and market-shape-only
        (a candidate whose promotion tick would select a pending
        order is refused — pending-fill claim resolution is not
        built yet, and a pending that later fills must never leave
        its claim dangling).
        """
        sid = str(candidate.get("setup_id") or "")
        engine = getattr(self, "_hot_scalp", None)
        if not self.cfg.hot_scalp_live_enabled:
            self.journal.log_note(
                f"mt5: hot_scalp promotion refused {sid}: "
                f"live authority OFF (hot_scalp_live_enabled=False)")
            return False
        if engine is None:
            return False
        claim = candidate.get("claim") or {}
        if str(claim.get("claimant") or "") != str(engine.instance_id):
            self.journal.log_note(
                f"mt5: hot_scalp promotion refused {sid}: claim owned "
                f"by another process ({claim.get('claimant')})")
            return False

        def _resolve(exists: bool) -> bool:
            try:
                return bool(engine.store.resolve_claim_after_broker_reconciliation(
                    setup_id=sid,
                    fencing_token=int(claim.get("fencing_token") or 0),
                    broker_order_exists=bool(exists), now=tick.get("time")))
            except Exception as exc:
                self.journal.log_note(
                    f"mt5: hot_scalp claim resolution failed {sid}: {exc}")
                return False

        if self.cfg.paused:
            self.journal.log_note(
                f"mt5: hot_scalp promotion refused {sid}: runner paused")
            _resolve(False)
            return False
        # Flat-only v1: a candidate while holding is not a layer
        # yet — the parent-identity mapping does not exist.
        if self.tick_manager.tickets():
            self.journal.log_note(
                f"mt5: hot_scalp promotion refused {sid}: holding — "
                f"scalp layering not yet promoted (flat-only v1)")
            _resolve(False)
            return False
        # V2 pre-submit broker check: the broker must agree we are
        # flat before any new risk is submitted.
        try:
            broker_before = [p for p in self.executor.positions()
                             if int(getattr(p, "magic", -1))
                             == int(self.cfg.magic)]
        except Exception as exc:
            self.journal.log_note(
                f"mt5: hot_scalp promotion withheld {sid}: broker "
                f"lookup failed ({exc}) — claim left claimed")
            return False
        if broker_before:
            self.journal.log_note(
                f"mt5: hot_scalp promotion refused {sid}: broker "
                f"shows an untracked magic position — claim left claimed")
            return False
        # Cost-envelope re-check at the promotion tick: the spread
        # that was tight at trigger time may have widened since.
        try:
            bid, ask = float(tick.get("bid")), float(tick.get("ask"))
            spread = ask - bid
            m1 = (getattr(self, "_hot_scalp_aux_frames", {}) or {}).get("M1")
            atr1 = float(_atr14(m1) or 0.0) if m1 is not None \
                and len(m1) >= 15 else 0.0
        except (TypeError, ValueError):
            bid, ask, spread, atr1 = 0.0, 0.0, None, 0.0
        if spread is None or spread < 0 or atr1 <= 0 \
                or spread > engine.cfg.max_spread_atr_m1 * atr1:
            self.journal.log_note(
                f"mt5: hot_scalp promotion refused {sid}: cost "
                f"envelope unverifiable or exceeded at promotion "
                f"(spread={spread}, m1_atr={atr1})")
            _resolve(False)
            return False
        # Market-shape-only v1: promotion runs on the trigger tick,
        # where price IS the entry, so the shared path must select
        # MARKET.  Anything else (limit/stop/invalid) is refused
        # rather than leaving a pending whose fill could never
        # resolve this claim.
        direction = 1 if int(candidate.get("direction", 0)) > 0 else -1
        entry_ref = float(candidate.get("entry_reference") or 0.0)
        mid = float(tick.get("mid") or entry_ref)
        setup_row = None
        try:
            for row in engine.store.active():
                if str(row.get("setup_id")) == sid:
                    setup_row = row
                    break
        except Exception:
            setup_row = None
        setup = (setup_row or {}).get("payload") or {}
        atr_m5 = 0.0
        try:
            atr_m5 = float(setup.get("atr_m5") or 0.0)
        except (TypeError, ValueError):
            atr_m5 = 0.0
        kind, _level, _why = select_order_type(
            direction=direction, price=mid, entry_ref=entry_ref,
            atr=atr_m5)
        if kind != "market":
            self.journal.log_note(
                f"mt5: hot_scalp promotion refused {sid}: order "
                f"shape is {kind}, not market (market-shape-only v1)")
            _resolve(False)
            return False
        sig = SimpleNamespace(
            direction=direction, entry_ref=entry_ref,
            invalidation=float(candidate.get("structural_stop") or 0.0),
            targets=[float(candidate.get("target_reference") or 0.0)],
            atr=atr_m5, index=self._current_h1_bar(),
            score=float(candidate.get("score") or 0.0),
            execution_eligible=True)
        # Step 15: stamp the declared execution intent (market
        # family — hot scalp v1 is market-shape only).
        sig.execution_intent = execution_intent_for("hot_scalp", sig)
        pb = SimpleNamespace(name="hot_scalp")
        tracked_before = set(self.tick_manager.tickets())
        self._last_tick = dict(tick)
        # score=None on purpose: _place_live_order's score parameter
        # must be a Score object (it reads score.total); passing a
        # float crashes its fill journal.  The candidate score
        # travels on sig.score, which the order path uses when
        # score is None.
        self._place_live_order(
            pb, sig, stats, verdict={"entry": entry_ref}, score=None)
        tracked_new = set(self.tick_manager.tickets()) - tracked_before
        if not tracked_new:
            # Nothing placed: the shared path refused it (gated and
            # journalled there).  The setup stays live.
            _resolve(False)
            return False
        try:
            broker_after = [p for p in self.executor.positions()
                            if int(getattr(p, "magic", -1))
                            == int(self.cfg.magic)]
        except Exception as exc:
            self.journal.log_note(
                f"mt5: hot_scalp fill recorded {sid} but broker "
                f"confirmation lookup failed ({exc}) — claim left claimed")
            return False
        if not broker_after:
            self.journal.log_note(
                f"mt5: hot_scalp fill recorded {sid} but broker "
                f"shows no new magic position — claim left claimed")
            return False
        _resolve(True)
        self.journal.log_note(
            f"mt5: hot_scalp promoted {sid}: order reconciled at "
            f"broker (claim resolved)")
        return True

    def _evaluate_live_tick(self, tick: dict, stats: dict) -> None:
        """Live evaluation clock: touch arms, family confirms, entry
        on entry-TF closes.  H1 closes never gate evaluation."""
        if self._stack is None:
            return
        # Independent Hot Scalp research: it runs on every live tick even while
        # the main sleeve is holding/paused. Its only output is a journaled
        # shadow candidate; it cannot submit orders or alter existing gates.
        # Step 6: when (and only when) the operator has enabled live
        # Hot Scalp authority on the host, this tick's shadow
        # candidates also flow to the flag-gated promotion path.
        _hs_events = self._run_hot_scalp_shadow(tick)
        if self.cfg.hot_scalp_live_enabled:
            for _event in _hs_events or []:
                if _event.get("event") == "candidate_shadow":
                    # Safety boundary: a promotion failure must
                    # never break the tick loop (the shared order
                    # path can raise after a fill, e.g. on a
                    # journal error — the claim then stays claimed,
                    # fail closed, and the loop continues).
                    try:
                        self._promote_hot_scalp_candidate(
                            _event, tick, stats)
                    except Exception as exc:
                        self.journal.log_note(
                            f"mt5: hot_scalp promotion error "
                            f"{_event.get('setup_id')}: "
                            f"{type(exc).__name__}: {exc}")
        mid = float(tick["mid"])
        # Zone cache: per-tick nearby query (in-memory bisect, no
        # SQLite) — runs even while paused / holding, so /status
        # nearby zones stay fresh on every tick.
        self._tick_zone_cache(mid)
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
            # Armed touch: survives only while price stays in/near
            # the zone (zone ± buf).  Ran away the other side
            # without confirming = the moment passed (left_zone):
            # the pending dies (logged, never erased) — a later
            # re-entry is a NEW touch, not a stale arm.  The far
            # side (price through the invalidation level) is NOT
            # checked here: the Close Rule says a WICK beyond the
            # level does not invalidate — only an entry-TF bar
            # CLOSE beyond does (see _check_close_rule, run on
            # entry closes in _refresh_bars phase 3).  A tick
            # through the level is a wick (the expected stop-hunt),
            # not an invalidation.  This is the sequential state
            # machine: confirmation must arrive while price is
            # still in/near the zone, not after it left.
            # (Confirmation on an entry close is evaluated before
            # this block runs — see _refresh_bars phase 3 — so a
            # confirming close wins over a disarm on the same
            # tick.)
            pend = self._pending
            buf = pend["setup"]["buf"]
            hi = pend["zone"]["top"] + buf
            if (pend["direction"] > 0 and mid >= hi) or \
                    (pend["direction"] < 0 and
                     mid <= pend["zone"]["bottom"] - buf):
                self._invalidate_pending("left_zone", stats)
            return
        # Regime pause (corrupt-state rebuild): no arming, no
        # breakout fire, no new setups — the rebuild must finish
        # (2 clean H1 bars) or the owner must /resume first.
        # Tick management runs elsewhere and is unaffected.
        if self._regime_pause is not None:
            return
        # Rework per-tick upkeep: PullbackWatch maintenance (§5)
        # + the A6 resting-limit check (fresh MAJOR POI ahead of
        # arrival).  Both are cheap guards first, no-ops unless
        # their state calls for action.
        self._maintain_pullback_watch(mid)
        self._maybe_place_resting_limit(tick, stats)
        # Stale re-arm block: if a previous touch disarmed as stale,
        # price must have EXITED that zone before any re-arm.  The
        # clear runs here (before the setup-None return) so an exit
        # lifts the block even while no setup is armed.
        if self._stale_zone is not None:
            _d, _b, _t = self._stale_zone
            if not (_b <= mid <= _t):
                self._stale_zone = None  # exited: re-arm allowed
        setup = self._live_setup
        if setup is None:
            return  # legs 1-2 not armed
        # RANGE regime: a range setup carries direction 0 and two
        # edge bands.  A tick AT an edge resolves to that edge's
        # per-edge setup (direction toward mid, zone = edge band)
        # so the stale / invalidated / pending machinery below —
        # which reads setup["direction"] / setup["zone"] — needs no
        # range special case.  A tick touching NEITHER edge leaves
        # the setup unresolved: only a compression breakout (below)
        # can fire from there.
        if setup.get("regime") == "range" \
                and check_zone_touch(setup, mid=mid):
            setup = resolve_setup_for_touch(setup, mid=mid)
        # Stale re-arm block (continued): if the block is still set,
        # price is still inside the dead zone.  Re-arm only on a
        # DIFFERENT zone (a different POI is a fresh setup) — the
        # same zone re-arms only after exit + re-entry.  Skipped
        # while a range setup is unresolved (no single zone yet).
        if self._stale_zone is not None and setup.get("zone") is not None:
            _d, _b, _t = self._stale_zone
            if (_d, _b, _t) == (setup["direction"],
                                setup["zone"]["bottom"],
                                setup["zone"]["top"]):
                return  # still inside the stale zone: no re-arm
            self._stale_zone = None  # zone changed: fresh setup
        # Invalidated-zone block (Close Rule): a zone that failed
        # the Close Rule is TERMINATED — exit + re-entry does not
        # revive it.  Re-arm only on a DIFFERENT zone (a different
        # POI is a different setup).
        if self._invalidated_zone is not None \
                and setup.get("zone") is not None:
            _d, _b, _t = self._invalidated_zone
            if (_d, _b, _t) == (setup["direction"],
                                setup["zone"]["bottom"],
                                setup["zone"]["top"]):
                return  # zone failed: terminated, no re-arm
            self._invalidated_zone = None  # zone changed: fresh setup
        # Continuation spent-break block: this break already fired
        # (one arm per break).  _refresh_live_setup suppresses the
        # setup at the next TF close; this guards ticks evaluated
        # against the pre-refresh setup within the same cycle.
        if setup.get("arm") == "continuation" \
                and str(setup.get("break_key") or "") \
                in self._continuation_spent:
            return
        # One trigger per H1-bar window: this bar already fired —
        # the setup is fixed within the bar, so a second fire here
        # would be the same setup re-firing on the same composing
        # bars (trigger storm).  The next H1 bar is a fresh window.
        if bar_open == self._fired_h1_bar:
            return
        # BREAKOUT_TREND fire: the boundary machine confirmed the
        # breakout (2 H1 closes beyond + M15 BOS); the tick clock
        # fires the outward trade ONCE per confirmed breakout
        # (breakout_id).  The compression-breakout playbook
        # re-verifies the pattern on the H1 frame in the signal
        # path; the tracker is the gate, the playbook is the
        # scorer.  A single close beyond never fires (see
        # BoundaryTracker); the old 1-close trigger is gone.
        if setup.get("regime") == "breakout_trend" \
                and self._boundary is not None:
            tracker = self._boundary
            if tracker.breakout_id != self._breakout_fired_id:
                self._breakout_fired_id = tracker.breakout_id
                fired = self._breakout_fired(setup, tick, bar_open)
                if fired is not None:
                    self._fired_h1_bar = bar_open
                    stats["triggers"] = stats.get("triggers", 0) + 1
                    self.journal.log_note(
                        f"mt5: trigger fired range_breakout "
                        f"{'+' if fired['direction'] > 0 else '-'}1 "
                        f"@ {fired['entry']:.2f} (H1 bar {bar_open}) — "
                        f"tracker-confirmed breakout (2 closes + BOS)")
                    self._emit_trigger_event(
                        "fired", "",
                        f"range_breakout "
                        f"{'+' if fired['direction'] > 0 else '-'}1 "
                        f"@ {fired['entry']:.2f} — tracker-confirmed "
                        "breakout (2 closes + BOS)")
                    self._run_live_signal(fired, tick, stats)
                    return
        # TRIGGER, REDESIGNED (2026-10-15): a tick AT the POI band
        # arms the pending; the confirmation family decides whether
        # it fires (see _evaluate_pending_entry).  The old
        # pre-filter demanded price PAST the H1 swing level AND
        # inside the zone — geometrically near-impossible (0/573
        # armed windows in the full-engine backtest) and against
        # the doctrine (entries AT the POI, never above it).
        if not check_zone_touch(setup, mid=mid):
            return
        session = _compose_session(tick_time)
        direction = int(setup["direction"])
        zone = setup["zone"]
        # The armed pending's invalidation level is the far boundary
        # of the zone (beyond the sweep extreme): a price through
        # it on a TICK is a wick (the expected stop-hunt, not an
        # invalidation); only an entry-TF bar CLOSE beyond it
        # invalidates (the Close Rule — see _check_close_rule).
        # A sweep inside the buffer survives (sweep_reclaim
        # needs it).
        invalidation_level = float(zone["bottom"] - setup["buf"]) \
            if direction > 0 else \
            float(zone["top"] + setup["buf"])
        self._pending = {
            "armed": True, "trigger": "zone_touch",
            "direction": direction, "entry": mid, "zone": zone,
            "entry_clock": session["entry_clock"],
            "h1_open": bar_open, "touch_ts": tick_time,
            "invalidation_level": invalidation_level,
            "setup": setup,
            # Armed TTL: entry-TF bars closed without confirmation
            # (see _refresh_bars phase 3 / _disarm_stale).
            "armed_bars": 0,
        }
        self._emit_trigger_event(
            "armed", "",
            f"zone_touch {'+' if direction > 0 else '-'}1 "
            f"@ {mid:.2f} (entry clock {session['entry_clock']})")
        self._evaluate_pending_entry(tick, stats)

    def _breakout_fired(self, setup: dict, tick: dict,
                       bar_open) -> dict | None:
        """Fired dict for a tracker-confirmed breakout.

        The boundary machine already confirmed it (2 H1 closes
        beyond + M15 BOS); this only packages the outward trade
        (direction outward, entry = confirm tick, stop just inside
        the edge, target = measured move edge +/- width) in the
        same shape as a confirmed pending trigger, so
        :meth:`_run_live_signal` needs no special case.  The
        breakout playbook re-verifies the pattern on the H1 frame
        in the signal path.
        """
        if self._stack is None:
            return None
        try:
            rng = RangeDef.from_dict(setup["range"])
        except Exception:
            return None
        direction = int(setup.get("direction", 0) or 0)
        if direction not in (1, -1):
            return None
        mid = float(tick["mid"])
        buf = float(setup.get("buf") or 0.5 * rng.atr)
        edge = setup.get("zone") or rng.edge_zone(
            "high" if direction > 0 else "low")
        if direction > 0:
            stop = float(rng.high - buf)
            target = float(rng.high + rng.width)
        else:
            stop = float(rng.low + buf)
            target = float(rng.low - rng.width)
        risk = abs(mid - stop)
        if risk <= 0:
            return None
        session = _compose_session(pd.Timestamp(tick["time"]))
        return {"trigger": "range_breakout",
                "confirm_shapes": ["range_breakout"],
                "direction": direction, "entry": float(mid),
                "stop": stop, "target": target, "risk": float(risk),
                "level": stop, "zone": edge,
                "fired_ts": pd.Timestamp(tick["time"]),
                "h1_open": bar_open,
                "entry_clock": session["entry_clock"],
                "setup": setup}

    def _evaluate_pending_entry(self, tick: dict, stats: dict) -> None:
        """Confirm an armed zone touch with the POI confirmation family.

        Runs on the arming tick (against this tick's already-closed
        composing entry bars) and on each entry-TF close.  A
        confirmed touch FIRES the trigger — the pending is consumed
        and the signal path runs.  Non-confirmation leaves the
        pending armed until the H1 close expires it (the window
        lapses), the armed TTL disarms it as stale (entry-TF bars
        without confirmation — see _refresh_bars phase 3), the
        Close Rule invalidates it (an entry-TF bar CLOSED beyond
        the invalidation level — a tick through the level is a
        wick, not an invalidation), price leaving the zone disarms
        it, or a bias flip kills it — the touch is logged, never
        erased.
        """
        pend = self._pending
        if pend is None or self._stack is None:
            return
        if not pend.get("armed"):
            return  # only armed pendings await confirmation
        # NOTE (layering): entry confirmation runs while holding — a
        # confirmed entry in the same direction becomes a layer
        # (_run_signal_path branches), never a second position.
        entry_frame = self._stack.frames.get(pend["entry_clock"])
        trig = check_live_trigger(
            pend["setup"], entry_frame=entry_frame,
            h1_open=pend["h1_open"], close=float(tick["mid"]))
        if trig is None:
            return  # no confirming shape yet: pending stands
        self._pending = None
        # One trigger per H1-bar window (see _fired_h1_bar): the
        # window is spent once it fires.
        self._fired_h1_bar = pend["h1_open"]
        fired = {**trig, "fired_ts": pd.Timestamp(tick["time"]),
                 "h1_open": pend["h1_open"],
                 "entry_clock": pend["entry_clock"],
                 # The arming setup rides along so _live_verdict can
                 # report the MEASURED H4 bias / context (a pullback
                 # verdict must not claim "H4 agrees" — see below).
                 "setup": pend["setup"]}
        stats["triggers"] = stats.get("triggers", 0) + 1
        self.journal.log_note(
            f"mt5: trigger fired {trig['trigger']} "
            f"{'+' if trig['direction'] > 0 else '-'}1 "
            f"@ {trig['entry']:.2f} (H1 bar {pend['h1_open']}) — "
            f"zone touch + {pend['entry_clock']} POI confirm "
            f"({','.join(trig['confirm_shapes'])})")
        self._emit_trigger_event(
            "fired", "",
            f"{trig['trigger']} "
            f"{'+' if trig['direction'] > 0 else '-'}1 "
            f"@ {trig['entry']:.2f} — "
            f"{','.join(trig['confirm_shapes'])}")
        self._run_live_signal(fired, tick, stats)

    def _run_live_signal(self, pend: dict, tick: dict,
                         stats: dict) -> None:
        """Entry confirmed: run the signal path on CLOSED bars.

        XD-6: the forming bar is reaction context, labelled —
        never a generator input.  The old code concatenated the
        still-forming H1 bar onto the closed frame and ran the
        playbooks, features and ATR over it, so live evaluated
        on a bar a closed-bar backtest would never see (a
        train/serve skew).  Now the playbooks, features and
        score read the closed H1 frame only; the forming row
        rides the verdict labelled ``forming: True`` plus one
        journal note, and the confirm tick's mid is still the
        entry price.  A setup that needs the forming bar's data
        confirms on the next closed bar, exactly as in
        backtest — that delay is the correction, not a loss.
        """
        # One arm per break (arming-gate fix F2): a continuation
        # setup that reaches the signal path has FIRED — spend its
        # break so neither the tick clock nor the next setup
        # refresh can arm the same break again.  (Breakout fires
        # carry no ``arm`` key and are spent via _breakout_fired_id
        # instead.)
        fired_setup = pend.get("setup") or {}
        if fired_setup.get("arm") == "continuation" \
                and fired_setup.get("break_key"):
            key = str(fired_setup["break_key"])
            if key not in self._continuation_spent:
                self._continuation_spent.add(key)
                self.journal.log_note(
                    "mt5: continuation break spent (one arm per "
                    f"break): {int(fired_setup.get('direction', 0)):+d} "
                    f"break of "
                    f"{float(fired_setup.get('ref_level', 0.0)):.2f} "
                    f"at {fired_setup.get('break_ts')}")
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
        # XD-6: no closed bars, no signal path — a forming bar
        # alone is not a setup (the old code ran the path on
        # the single forming row).
        if not len(h1_closed):
            self.journal.log_note(
                "mt5: live signal dropped — no closed H1 bars; "
                "forming bar alone is not generator input (XD-6)")
            return
        # XD-6: generators read the CLOSED frame only.  The
        # forming row is reaction context, labelled, carried on
        # the verdict — never appended to the generator frame.
        states = read_and_featurize(h1_closed, None)
        t = len(h1_closed) - 1
        years = (h1_closed.index[-1] - h1_closed.index[0]) \
            .total_seconds() / 86400.0 / 365.25
        verdict = self._live_verdict(pend, tick, h1_closed, states, t)
        forming_record = {
            "open": float(forming["open"]),
            "high": float(forming["high"]),
            "low": float(forming["low"]),
            "close": float(tick["mid"]),
            "volume": float(forming.get("volume", 0.0)),
            "real_volume": float(forming.get("real_volume", 0.0)),
            "spread": forming.get("spread", 0),
            "open_time": str(forming["open_time"]),
            "forming": True}
        verdict["forming"] = forming_record
        self.journal.log_note(
            "mt5: XD-6 live signal — generators on closed H1 "
            f"bars only (last closed {h1_closed.index[-1]}); "
            f"forming bar {forming['open_time']} labelled "
            "reaction-only")
        # Setup lifecycle (owner rulings 2026-10-30 §1): the fire
        # gets a guaranteed disposition — after the signal path
        # runs, an execution_evaluated trigger record states what
        # the fire produced (records emitted, lifecycle setups
        # routed and their outcomes), including when the path
        # raises.  A fired trigger must never disappear between
        # analysis and execution again (the 2026-10-30 03:30
        # hole).
        _seq0 = self._funnel_seq
        _routing: dict = {}
        _fire_dir = int(verdict.get("direction")
                        or pend.get("direction") or 0)
        try:
            self._run_signal_path(h1_closed, t, states, years,
                                  verdict, stats,
                                  forming=forming_record,
                                  lifecycle_routing=_routing)
        except Exception:
            self._emit_trigger_event(
                "execution_evaluated", "path.exception",
                f"fire {_fire_dir:+d}: signal path raised — "
                f"routed setups so far: {sorted(_routing)}")
            raise
        _frags = self._apply_lifecycle_routing(_routing, _fire_dir)
        _armed_waiting = 0
        if self._setup_store is not None and _fire_dir in (1, -1):
            _armed_waiting = len(self._setup_store.active(
                direction=_fire_dir, armed_only=True))
        self._emit_trigger_event(
            "execution_evaluated", "",
            f"fire {_fire_dir:+d}: signal path emitted "
            f"{self._funnel_seq - _seq0} funnel records; "
            "lifecycle routing: "
            f"{'; '.join(_frags) if _frags else 'no armed setups routed'}; "
            f"armed setups remaining in direction: "
            f"{_armed_waiting}")

    def _apply_lifecycle_routing(self, routing: dict,
                                 direction: int) -> list[str]:
        """Advance routed setups' lifecycles from one fire's
        routing results (owner rulings 2026-10-30 §1).  Returns
        one summary fragment per routed setup for the fire's
        disposition record.  A setup that was evaluated but
        neither ordered nor refused (parked in PullbackWatch,
        execute-list narrowed, layer evaluation with no order)
        returns to armed — it stays live for a later fire inside
        its validity window; the shared order path's
        one-position / pending guards remain the final word
        against duplicates."""
        out: list[str] = []
        if self._setup_store is None:
            return out
        for _name, res in routing.items():
            setup = self._setup_store.get(res.get("setup_id") or "")
            if setup is None or not setup.armed:
                continue
            status = res.get("status")
            if status == "skipped_disabled":
                out.append(
                    f"{setup.setup_id}: not routed (playbook "
                    "disabled; stays armed)")
                continue
            self._transition_setup(
                setup, _setup_lc.STAGE_TRIGGER_FIRED,
                f"h1_trigger fired {direction:+d} — routed to "
                "execution evaluation")
            if status == "exception":
                self._transition_setup(
                    setup, _setup_lc.STAGE_EVALUATED,
                    "evaluation raised — see the playbook's "
                    "exception record")
                self._transition_setup(
                    setup, _setup_lc.STAGE_ARMED,
                    "re-armed after evaluation exception (still "
                    "within validity)")
                out.append(
                    f"{setup.setup_id}: evaluation exception; "
                    "re-armed")
                continue
            terms = res.get("terminals") or []
            if not terms:
                self._transition_setup(
                    setup, _setup_lc.STAGE_EVALUATED,
                    "no terminal decision produced")
                self._transition_setup(
                    setup, _setup_lc.STAGE_ARMED,
                    "re-armed after evaluation with no decision "
                    "(still within validity)")
                out.append(
                    f"{setup.setup_id}: evaluated, no decision; "
                    "re-armed")
                continue
            term = terms[0]
            result = term.get("result")
            ores = term.get("order_result")
            code = term.get("reason_code") or ""
            detail = (f"decision {result}"
                      + (f" ({code})" if code else "")
                      + (f"; order {ores}" if ores else ""))
            self._transition_setup(
                setup, _setup_lc.STAGE_EVALUATED, detail)
            if ores in ("filled", "layer_filled"):
                self._transition_setup(
                    setup, _setup_lc.STAGE_FILLED, detail,
                    alert_id=term.get("alert_id"))
                out.append(f"{setup.setup_id}: filled")
            elif ores in ("placed_pending", "layer_pending"):
                self._transition_setup(
                    setup, _setup_lc.STAGE_SUBMITTED, detail,
                    alert_id=term.get("alert_id"))
                out.append(
                    f"{setup.setup_id}: order submitted ({ores})")
            elif result == "refused" or ores in (
                    "refused_in_order_path", "rejected"):
                self._transition_setup(
                    setup, _setup_lc.STAGE_REJECTED, detail,
                    alert_id=term.get("alert_id"))
                out.append(
                    f"{setup.setup_id}: rejected "
                    f"({code or result})")
            else:
                self._transition_setup(
                    setup, _setup_lc.STAGE_ARMED,
                    f"re-armed after evaluation ({detail}; "
                    "still within validity)")
                out.append(
                    f"{setup.setup_id}: evaluated ({result}); "
                    "re-armed")
        return out

    def _live_verdict(self, pend: dict, tick: dict,
                      h1_closed: pd.DataFrame, states, t: int) -> dict:
        """Live verdict: same shape as the closed-bar verdict.

        Legs 1-4 come from the fired trigger (closed-bar levels +
        zone touch); leg 5 from the confirmed POI shape on the
        entry clock.  Entry is the confirm tick's mid (the actual
        market entry), stop/target from the trigger computation,
        risk re-measured from the confirm price.  AMD / QM / volume
        annotations as in :meth:`_verdict_for_bar`.
        """
        direction = int(pend["direction"])
        # Measured H4 bias from the arming setup — NEVER assumed
        # equal to the direction.  The old code hardcoded
        # ``bias_h4 = direction`` and logged "H4 agrees", which was
        # false on every pullback verdict (H4 +1, direction -1)
        # and hid the 2026-10-16 root cause in the journal.
        # Missing setup (legacy callers) falls back to the
        # direction, exactly as before.
        setup = pend.get("setup") or {}
        bias_h4 = int(setup.get("bias_h4", direction))
        h4_context = str(setup.get("h4_context", "") or "")
        # Regime rides on the verdict: "trend" (default), "range"
        # (D1 flat + defined range), and the range def itself.
        # Range verdicts carry bias_d1 == 0 (the D1 bias really is
        # flat) and size_mult 0.5 (choppy regime = half size).
        regime = str(setup.get("regime", "trend"))
        bias_d1 = int(setup.get("bias_d1", direction))
        size_mult = float(setup.get("size_mult", 1.0) or 1.0)
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
        if regime == "range":
            legs["d1_bias"]["reason"] = (
                "RANGE regime (D1 flat -> range edge)")
        elif regime == "breakout_trend":
            legs["d1_bias"]["reason"] = (
                "BREAKOUT_TREND (D1 flat, range edge broken — "
                "trend playbooks armed in breakout direction)")
        else:
            legs["d1_bias"]["reason"] = \
                f"D1 bias {direction:+d} (closed bars)"
        legs["h4_agree"]["reason"] = (
            f"H4 {h4_context or 'context'} {bias_h4:+d} (closed) — "
            f"POI supplier, direction stays D1 {direction:+d}")
        if setup.get("arm") == "continuation":
            # The zone here is the broken H1 level's retest band,
            # NOT an H4 POI — the verdict must not claim otherwise.
            legs["h4_zone"]["reason"] = (
                "live price at broken H1 level (continuation "
                "retest)")
            legs["h1_trigger"]["reason"] = (
                f"continuation retest + {pend['trigger']} confirm")
        else:
            legs["h4_zone"]["reason"] = \
                "live price at H4 POI (zone touch)"
            legs["h1_trigger"]["reason"] = \
                f"zone touch + {pend['trigger']} confirm"
        legs["entry"]["reason"] = \
            f"{pend['entry_clock']} POI confirm"
        return {
            "allowed": True, "blocked_leg": "", "legs": legs,
            "bias_d1": bias_d1, "bias_h4": bias_h4,
            "h4_context": h4_context,
            "regime": regime, "range": setup.get("range"),
            "size_mult": size_mult,
            # Trade-style marker (hierarchy F2): the arming
            # setup's arm rides the verdict so the per-style
            # floor in _run_one_playbook routes a continuation
            # trade to its own floor row.  The setup dict is the
            # only place this marker exists on the tick path
            # (live_setup sets arm == "continuation"); without
            # this copy the verdict reaching the floor had lost
            # it.  Pullback / range setups carry no arm (None)
            # and route exactly as before.
            "arm": setup.get("arm"),
            # Boundary-gate fields (confluence gate): sweep
            # confirmations + edge cooldowns from the persisted
            # boundary machine.  A wick alone never passes the
            # sweep gate; a cooling edge never passes the
            # cooldown gate.
            "sweep_confirmed_edges": list(
                (self._boundary_snapshot or {}).get(
                    "sweep_confirmed_edges") or []),
            "cooldown_edges": list(
                (self._boundary_snapshot or {}).get(
                    "cooldown_edges") or []),
            # Multi-day structural zones (historical WHERE):
            # top MAJOR zones + the MAJOR zone the armed POI
            # coincides with, if any.  Context only — zones
            # never trigger a trade alone.
            "structural_zones": list(
                (setup.get("structural_zones") or [])[:3]),
            "major_zone": setup.get("major_zone"),
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
            # DXY correlation: confluence context, never a gate.
            "dxy_corr": (self._dxy.state.corr
                         if self._dxy.state.available else None),
            # Signal-time urgency (dispatch context, never a gate).
            "urgency": self._urgency_for(
                mid, tick.get("time"), pend["zone"], h1_closed),
        }

    def _urgency_for(self, price, timestamp, zone, frame) -> float | None:
        """Signal-time urgency for a verdict (record + reasoning).
        Never raises — None on any failure (verdict annotation must
        not break the verdict).  Placement-time urgency is computed
        separately in _place_live_order (dispatch reading)."""
        try:
            return urgency_score(
                float(price), session=session_of(timestamp)[0],
                zone=zone, frame=frame)
        except Exception:
            return None

    # --------------------------------------- rework decision helpers
    def _executes(self, pb_name: str) -> bool:
        """A14: is this playbook eligible to execute on the MT5
        path?  Default (execute_playbooks None) = every evaluated
        playbook.  An explicit config tuple narrows it; the
        disabled list is the kill switch and is checked by the
        caller before evaluation."""
        if self.cfg.execute_playbooks is None:
            return True
        return pb_name in self.cfg.execute_playbooks

    def _sizing_mode(self) -> str:
        """Effective sizing mode (owner ruling 2026-10-23): dynamic
        risk sizing is the ONLY sizing family for orders — the
        configured mode selects between the Decimal RiskPool path
        ("risk") and the legacy float path for evaluation/forensics
        only; every order path refuses a non-risk mode (see the
        enforcement notes at the order sites). There is no
        fixed-lots mode (the A13 override is revoked, the field
        is gone)."""
        return str(self.cfg.sizing_mode or "risk")

    def _zone_dict_for(self, verdict: dict | None,
                       direction: int) -> dict | None:
        """The graded structural zone behind this verdict, as a
        dict (zones.to_dict shape, incl. lifecycle A2): the
        setup's coincident MAJOR zone when the refresh annotated
        one; else the nearest structural zone overlapping the
        trigger band; else the trigger band itself, graded
        'tested' (a retest band is definitionally tested) so the
        score still reads a zone context."""
        if not verdict:
            return None
        major = verdict.get("major_zone")
        if major:
            return dict(major)
        zone = verdict.get("zone") or {}
        top, bottom = zone.get("top"), zone.get("bottom")
        if top is None or bottom is None:
            return None
        best = None
        best_d = None
        for z in self._structural_zones or []:
            try:
                d = min(abs(float(z.top) - float(top)),
                        abs(float(z.bottom) - float(bottom)))
            except Exception:
                continue
            if best_d is None or d < best_d:
                best, best_d = z, d
        atr = float(verdict.get("risk") or 0.0)
        if best is not None and (best_d is not None
                                 and (atr <= 0 or best_d <= atr)):
            return best.to_dict()
        out = dict(zone)
        out.setdefault("tier", "MINOR")
        out.setdefault("lifecycle", "tested")
        out.setdefault("touches", 1)
        out.setdefault("status", "active")
        out.setdefault("score", 40.0)
        # This band is the hierarchy's own ARMED entry level (legs
        # 1-2 marked it; the trigger fired there).  It carries no
        # ZoneStore grade, so no tier is invented for it — the
        # marker tells the Location component to credit the
        # at-the-level share in full (the setup armed HERE) while
        # curve + draw carry the quality discrimination.
        out["armed_band"] = True
        return out

    def _opposing_dict(self, direction: int) -> dict | None:
        """Nearest opposing structural zone (the draw magnet)."""
        try:
            from ..live.opposing import opposing_zone_for
            zones = [z.to_dict() for z in self._structural_zones or []]
            price = float((self._last_tick or {}).get("mid") or 0.0)
            if not zones or price <= 0:
                return None
            opp = opposing_zone_for(zones, price, direction)
            return dict(opp) if opp else None
        except Exception:
            return None

    def _adr_gauge_valid(self, atr: float) -> bool:
        """The ADR fuel gauge is a measurement only when the
        day object's ADR actually measures THIS market: enough
        D1 days behind it (>= 10, the standing guard) AND a
        magnitude commensurate with the intraday frame — a mean
        daily range below the signal frame's own ATR is a scale
        break between the D1 series and the intraday frames
        (a day's range contains its hours' ranges), not a spent
        day.  On a broken gauge the input reverts to absent
        (P1 behaviour — never invented, never a block): the P3
        regression was a 0.5-point ADR against a 41-point day
        reading 8349% consumed and de-rating every continuation
        score's session block to the floor factor."""
        d = self._day_ctx
        if d is None or d.adr_days < 10:
            return False
        adr = float(d.adr or 0.0)
        if adr <= 0:
            return False
        try:
            a = float(atr or 0.0)
        except (TypeError, ValueError):
            a = 0.0
        return a <= 0 or adr >= a

    def _score_for(self, *, direction: int, verdict: dict | None,
                   price: float, zone_d: dict | None,
                   atr: float, style_key: str = "intraday",
                   playbook: str = "") -> object:
        """Asymmetry Score for one candidate (spec §2).  Frames
        are parameters, not premises: P3 floats the scoring frame
        per style via the S12 clock contract (volatility +
        structure clarity) — at NORMAL vol the desk's defaults
        reproduce the P1 reads exactly (H1 path on H1).  The
        chosen clock is stashed on ``self._last_clock`` and
        journaled per decision (``trigger_tf_used``).

        At-zone candidates (price inside the zone band ± 0.5 ATR)
        are scored with allow_flip=False: the approach into a
        zone closes through minor trigger-frame pivots on the
        way in — arrival, not a thesis flip; the zone's far edge
        governs invalidation there (spec §5)."""
        frames = (self._stack.frames if self._stack is not None
                  else {}) or {}
        at_zone = False
        if zone_d is not None and atr and float(atr) > 0:
            try:
                at_zone = float(zone_d["bottom"]) - 0.5 * float(atr) \
                    <= float(price) \
                    <= float(zone_d["top"]) + 0.5 * float(atr)
            except Exception:
                at_zone = False
        # The dealing range for the Location curve is the range of
        # the trade's CONTEXT — the timeframe that owns the zone
        # (an H4 POI trade is located in the H4 range; measuring it
        # against D1 would grade a different trade).  Fallback:
        # D1, then the scoring frame.
        range_frame = None
        zone_tf = str((zone_d or {}).get("tf") or "")
        if not zone_tf and verdict is not None \
                and verdict.get("h4_context"):
            # The verdict's own provenance: its POI is the
            # hierarchy's H4 zone (h4_context is only set when
            # the H4 leg built the setup).
            zone_tf = "H4"
        if zone_tf and frames.get(zone_tf) is not None:
            range_frame = frames.get(zone_tf)
        elif frames.get("D1") is not None:
            range_frame = frames.get("D1")
        else:
            range_frame = frames.get("H1")
        # --- P3 clock float (S12) --------------------------------
        clock = self._clock_for_signal(style_key, frames)
        self._last_clock = clock
        score_frame = frames.get(clock.score_tf)
        if score_frame is None:
            score_frame = frames.get("H1")
        frame_atr = float(atr or 0.0)
        if clock.score_tf != "H1" and score_frame is not None \
                and len(score_frame):
            try:
                fa = float(atr14(score_frame))
                if fa > 0:
                    frame_atr = fa
            except Exception:
                pass
        # --- P3 score inputs --------------------------------------
        smt_dir = 0
        if self._smt_read is not None \
                and self._smt_read.event is not None:
            smt_dir = int(self._smt_read.event.direction)
        from ..live.funding import SMT_PLAYBOOKS as _SMT_PBS
        ctx = ScoreContext(
            direction=int(direction),
            frame=score_frame,
            atr=frame_atr, price=float(price),
            style="", zone=zone_d,
            opposing_zone=self._opposing_dict(direction),
            range_frame=range_frame,
            verdict=verdict,
            dxy_corr=(verdict or {}).get("dxy_corr"),
            smt_direction=smt_dir,
            smt_context=("reversal" if playbook in _SMT_PBS
                         else "trend"),
            adr_consumed_pct=(self._day_ctx.adr_consumed_pct
                              if self._adr_gauge_valid(atr)
                              else None),
            adr_applies=(_posture_mod.family_of(playbook)
                         == "continuation"),
            day_levels=self._day_levels())
        return score_asymmetry(ctx, allow_flip=not at_zone)

    def _clock_for_signal(self, style_key: str, frames: dict):
        """The S12 floated clock for one style: volatility regime
        from the H4 ATR baseline + per-TF structure clarity."""
        vol = "NORMAL"
        try:
            h4 = frames.get("H4")
            atr = float(atr14(h4)) if h4 is not None and len(h4) \
                else 0.0
            if atr > 0:
                reg = _atr_regime(atr, _baseline_atr(h4))
                vol = str(reg.get("regime", "NORMAL"))
                ratio = float(reg.get("ratio", 1.0) or 1.0)
                if ratio <= 0.75:
                    vol = "LOW"
        except Exception:
            vol = "NORMAL"
        clarity = {}
        if vol == "NORMAL":
            for tf in ("M5", "M15", "H1", "H4"):
                f = frames.get(tf)
                if f is not None and len(f) >= 24:
                    clarity[tf] = _posture_mod.structure_clarity(f)
        return _posture_mod.clock_for(
            style_key, vol_regime=vol, clarity=clarity or None)

    def _style_key_for(self, pb_name: str,
                       zone_d: dict | None) -> str:
        """Clock/style bucket for a signal: qm_d1 runs the D1
        clock; scalps (qm_m5 + scalp-named) the scalp clocks;
        big-zone trades swing clocks; the rest intraday."""
        if pb_name == "qm_d1":
            return "qm_d1"
        if pb_name == "qm_m5" or "scalp" in str(pb_name):
            return "scalp"
        zone_tf = str((zone_d or {}).get("tf") or "")
        if zone_tf in ("W1", "MN1", "D1"):
            return "swing"
        return "intraday"

    @staticmethod
    def _style_bucket(style_key: str) -> str:
        """StyleBook bucket name for one clock key (churn
        ceilings + position context)."""
        return {"swing": "MACRO_SWING", "qm_d1": "MACRO_SWING",
                "intraday": "INTRADAY",
                "scalp": "MICRO_SCALP"}.get(style_key, "INTRADAY")

    def _floor_ok(self, fl: dict, score) -> bool:
        """One floor row against one score: total + component
        minimums + the Path B combined Structure+Efficiency
        term where the row declares it (P3)."""
        ok = score.total >= float(fl["floor"]) \
            and score.component("location") >= float(
                fl.get("min_location", 0.0)) \
            and score.component("structure") >= float(
                fl.get("min_structure", 0.0)) \
            and score.component("efficiency") >= float(
                fl.get("min_efficiency", 0.0)) \
            and score.component("pullback") >= float(
                fl.get("min_pullback", 0.0))
        mse = fl.get("min_structure_efficiency")
        if mse is not None:
            ok = ok and (score.component("structure")
                         + score.component("efficiency")) \
                >= float(mse)
        return ok

    def _zone_dicts(self) -> list:
        """The graded zone book as dicts (decision-flow reads)."""
        out = []
        for z in (self._structural_zones or []):
            try:
                out.append(z.to_dict())
            except Exception:
                continue
        return out

    def _zone_tp_targets(self, sig, zone_d) -> list:
        """A8 zone-anchored TP targets for one accepted signal:
        the graded zones AHEAD of entry (excluding the entry zone
        itself) priced through tp_ladder.  Empty when no zone is
        ahead — the playbook's own targets stand."""
        try:
            entry = float(sig.entry_ref)
            stop = float(sig.invalidation)
            d = int(sig.direction)
            own = float((zone_d or {}).get("level", -1e18))
            ahead = []
            for z in self._zone_dicts():
                if z.get("status", "active") != "active":
                    continue
                if abs(float(z.get("level", 0.0)) - own) < 1e-9:
                    continue  # the entry zone is not a target
                prox = float(z.get("proximal")
                             or z.get("level") or 0.0)
                if (prox - entry) * d > 0:
                    ahead.append(z)
            if not ahead:
                return []
            ladder = _tp_ladder(entry=entry, stop=stop,
                                direction=d, zones_ahead=ahead)
            targets = [float(ladder[k]) for k in
                       ("tp1", "tp2", "tp3")
                       if ladder.get(k) is not None]
            if targets:
                self.journal.log_note(
                    "mt5 tp ladder: " + "; ".join(ladder["log"]))
            return targets
        except Exception:
            return []

    def _journey_context(self, *, direction: int, price: float,
                         stop_dist: float, zone_d: dict | None,
                         atr: float) -> dict:
        """Journey geometry for an OPEN-SPACE signal (G): the
        map is never empty between zones — the nearest graded
        magnet ahead is the journey's draw.  With-journey trades
        ride toward it (room < 1.5R is a geometry refusal: the
        target is inside the noise).  Counter-journey trades
        (the nearer draw is BEHIND) are refused unless a nested
        faster-TF zone holds price AND the room to the magnet
        clears the configured minimum — the refusal is journaled
        with its reason, never silent."""
        res: dict = {"state": "at_zone" if zone_d is not None
                     else "open", "magnet": None,
                     "magnet_dist_atr": None, "room_r": None,
                     "refusal": None, "nested": False}
        if zone_d is not None:
            return res
        zones = [z for z in self._zone_dicts()
                 if z.get("status", "active") == "active"
                 and z.get("tier") in ("MAJOR", "MINOR")]

        def _prox(z):
            return float(z.get("proximal") or z.get("level")
                         or 0.0)

        def _inside(z):
            b, t = z.get("bottom"), z.get("top")
            return b is not None and t is not None \
                and float(b) <= price <= float(t)

        # A zone whose band CONTAINS price is the holder, not the
        # magnet — the journey's draw is the next zone beyond it.
        ahead = [z for z in zones
                 if (_prox(z) - price) * direction > 0
                 and not _inside(z)]
        behind = [z for z in zones
                  if (_prox(z) - price) * direction < 0]
        if not ahead:
            return res
        magnet = min(ahead, key=lambda z: abs(_prox(z) - price))
        mdist = abs(_prox(magnet) - price)
        res["magnet"] = magnet
        res["magnet_dist_atr"] = (mdist / atr) if atr > 0 else None
        res["room_r"] = (mdist / stop_dist) \
            if stop_dist > 0 else None
        bdist = min((abs(_prox(z) - price) for z in behind),
                    default=None)
        need = float(self.cfg.counter_journey_min_room_r)
        if bdist is not None and bdist < mdist:
            res["state"] = "counter"
            try:
                mag_idx = _TF_ORDER.index(str(magnet.get("tf")))
            except ValueError:
                mag_idx = 0
            nested = False
            for z in zones:
                try:
                    zi = _TF_ORDER.index(str(z.get("tf")))
                except ValueError:
                    continue
                if zi > mag_idx and float(z.get("bottom", 0.0)) \
                        - 0.5 * atr <= price \
                        <= float(z.get("top", 0.0)) + 0.5 * atr:
                    nested = True
                    break
            res["nested"] = nested
            if not nested:
                res["refusal"] = (
                    "counter_journey: the nearer draw is behind "
                    "the trade and no nested faster-TF zone holds "
                    "price — counter-journey needs a nested zone "
                    f"+ >= {need:.0f}R room to the magnet (G)")
            elif res["room_r"] is not None \
                    and res["room_r"] < need:
                res["refusal"] = (
                    f"counter_journey: room to the magnet "
                    f"{res['room_r']:.1f}R < {need:.0f}R minimum "
                    "— refused (G)")
        else:
            res["state"] = "with"
            if res["room_r"] is not None and res["room_r"] < 1.5:
                res["refusal"] = (
                    f"journey_room: magnet only "
                    f"{res['room_r']:.1f}R away — the target "
                    "sits inside the noise (geometry, G)")
        return res

    def _funding_report(self, verdict: dict | None,
                        zone_d: dict | None):
        """Funding-event detectors over the entry frame (spec
        §4.2): the chain evaluators reused, never gated on.  The
        entry frame follows the S12 floated funding clock for the
        verdict's style (M15 at NORMAL vol — the narrative
        chain's convention; H1 when vol promotes); ATR stays the
        H4 reference with regime-scaled thresholds."""
        frames = (self._stack.frames if self._stack is not None
                  else {}) or {}
        entry_clock = str((verdict or {}).get("entry_clock")
                          or "M15")
        style = "scalp" if entry_clock in ("M5", "M1") \
            else "intraday"
        clock = self._clock_for_signal(style, frames)
        entry = frames.get(clock.funding_tf)
        if entry is None:
            entry = frames.get("M15")
        if entry is None:
            entry = frames.get("H1")
        h4 = frames.get("H4")
        atr = float(atr14(h4)) if h4 is not None and len(h4) else 0.0
        if atr <= 0:
            atr = float((verdict or {}).get("risk") or 0.0)
        try:
            reg = _atr_regime(atr, _baseline_atr(h4))
            th = _scaled_thresholds(str(reg.get("regime", "NORMAL")))
        except Exception:
            th = {"disp_min_atr": 1.5, "volume_spike_min": 1.5}
        direction = int((verdict or {}).get("direction", 0) or 0)
        return detect_funding_events(
            entry, direction=direction, atr=atr, zone=zone_d,
            disp_min_atr=float(th["disp_min_atr"]),
            volume_spike_min=float(th["volume_spike_min"]))

    def _traffic(self, zone_d: dict | None) -> tuple[str, float, int]:
        """Traffic-light scalar (spec §4.4): (tier, size_mult,
        layer_ceiling) from the armed zone's corrected grade.
        Trims size/layers/TPs — never vetoes.  No zone grade ->
        no trim (the scalar prices a barrier; without one there
        is nothing to price)."""
        if not zone_d:
            tier, mult, ceiling = "YELLOW", 1.0, 99
        else:
            tier = _zone_risk_tier(zone_d)
            spec = _tier_spec(tier)
            ceiling = {"GREEN": 3, "YELLOW": 2, "RED": 1}.get(
                tier, 2)
            mult = float(spec.get("size_mult", 0.75))
        # S11 loss ladder: >= 2 consecutive losses caps the
        # traffic light at YELLOW (size + layer ceiling follow
        # the tier down).  A cap only ever trims — never vetoes.
        ladder = self._posture.ladder if self._posture else None
        if ladder is not None and ladder.traffic_cap == "YELLOW" \
                and tier == "GREEN":
            tier = "YELLOW"
            mult = min(mult, float(_tier_spec("YELLOW").get(
                "size_mult", 0.75)))
            ceiling = min(ceiling, 2)
        return tier, mult, ceiling

    def _register_position_ctx(self, ticket: int, *,
                               alert_id: str, entry: float,
                               stop: float, direction: int) -> dict:
        """P3: build the per-position management context at fill.

        Consumes the decision context parked at placement (the
        golden-confirm path may not carry one — defaults are then
        derived from the playbook identity, never invented
        scores).  The context is what bar-close management
        (progress / drift / pre-print / zone bank / ADR bank /
        defended stop) reads: entry score + floor, style, the
        S9 management profile, the opposing magnet, one-shot
        flags.  Also books the S11 per-style daily open count.
        """
        parked = self._signal_ctx.pop(alert_id, None) or {}
        pb_name = parked.get("playbook") \
            or self._playbook_from_alert_id(alert_id)
        zone_d = parked.get("zone_d")
        style_key = parked.get("style_key") \
            or self._style_key_for(pb_name, zone_d)
        bucket = parked.get("style_bucket") \
            or self._style_bucket(style_key)
        narr = self._narrative_states.get(int(ticket))
        clock_tf = parked.get("clock_tf") or "H1"
        profile_style = "scalp" if bucket == "MICRO_SCALP" else (
            "v_reversal" if pb_name == "v_reversal" else style_key)
        profile = _mgmt_profile_for(
            style=profile_style,
            philosophy=(self._posture.target_philosophy
                        if self._posture is not None else "trend"),
            late_confirm=bool(getattr(narr, "late_confirm", False)),
            high_vol=bool(getattr(narr, "high_vol", False))
            or (self._bedrock is not None
                and str(self._bedrock.volatility) in
                ("high", "spike")),
            clock_tf=clock_tf)
        opp = self._opposing_dict(int(direction))
        ctx = {
            "ticket": int(ticket), "alert_id": alert_id,
            "playbook": pb_name, "style_key": style_key,
            "style_bucket": bucket,
            "family": parked.get("family")
            or _posture_mod.family_of(pb_name),
            "floor": float(parked.get("floor", 50.0)),
            "entry_total": parked.get("entry_total"),
            "entry_bar": self._current_h1_bar(),
            "clock_tf": clock_tf,
            # XD-4: the position's management clock, inherited
            # from its origin route (M5 scalp / D1 swing / H1
            # roster / floated trigger clock).  entry_mgmt_bar
            # is the forming-bar index on that clock at fill
            # (last closed index + 1, the _current_h1_bar
            # convention); last_mgmt_bar starts at the last
            # closed index so the entry bar's own close is due.
            "management_clock": clock_tf,
            "entry_mgmt_bar": self._mgmt_bar_now(clock_tf) + 1,
            "last_mgmt_bar": self._mgmt_bar_now(clock_tf),
            "profile": profile,
            "defended_stop": parked.get("defended_stop"),
            "invalidation": float(stop),
            "direction": int(direction),
            "zone_d": zone_d,
            "opposing_prox": (float(opp["proximal"])
                              if opp and opp.get("proximal")
                              is not None else None),
            "progress_trimmed": False, "progress_trim_bar": None,
            "drift_trimmed": False, "preprint_key": None,
            "zone_banked": False, "adr_banked": False,
        }
        self._position_ctx[int(ticket)] = ctx
        # S11 churn bookkeeping: opens count against the style's
        # daily ceiling (reset on the NY trading day).
        day_key = self._day_ctx.day_key if self._day_ctx else ""
        if day_key and day_key != self._style_day:
            self._style_day = day_key
            self._style_opens_today = {}
        self._style_opens_today[bucket] = \
            self._style_opens_today.get(bucket, 0) + 1
        return ctx

    def _pool_for_open(self, ticket: int, *, lots: float,
                       entry: float, stop: float, direction: int,
                       alert_id: str) -> RiskPool | None:
        """Create + book the RiskPool for a freshly opened base
        position (A13).  Budget = the legacy total-risk envelope
        (equity x risk_pct x layer cap multiple) so the rework's
        ceiling is exactly the old one; distribution inside it is
        evidence-driven.  Fixed mode books the owner's explicit
        lots against an envelope at least as large (the pool is
        the ledger there, RiskGuards remain the gate).

        ``stop`` MUST be the buffered broker stop (state.sl /
        stop0), never the structural invalidation — step 2: the
        pool books true exposure, matching sizing and close R."""
        # P3: every base fill registers its management context
        # (independent of the pool ledger's fate below).
        self._register_position_ctx(
            ticket, alert_id=alert_id, entry=entry, stop=stop,
            direction=direction)
        if ticket in self._pools:
            return self._pools[ticket]
        try:
            spec = self.executor.symbol_spec(self.cfg.symbol)
            acct = self.feed.account_state()
            equity = float(acct.get("equity", 0.0) or 0.0)
        except Exception:
            return None
        if equity <= 0:
            return None
        budget = equity * float(self.cfg.risk_pct) \
            * float(self.cfg.layer_max_total_risk_mult)
        pool = RiskPool(budget)
        if not pool.authorize(alert_id, lots, entry, stop,
                              int(direction), spec):
            # The order already filled (the pool is created at
            # fill time) — the ledger must book reality, so widen
            # the budget to the layer's realized risk and journal
            # it.  It means placement sized past the envelope —
            # visible, never silent.
            try:
                realized = float(spec.money_for_move(
                    abs(float(entry) - float(stop)), float(lots)))
            except Exception:
                realized = budget
            # Step 2: a widening is an EXCEPTION, named as one.  The
            # routine case after broker-stop booking is the broker
            # minimum flooring above the budget (min_override) —
            # the owner ruling says the trade still happens, but
            # the ledger must say WHY it exceeds the envelope.
            _min_floor = bool(
                (self._sizing_detail.get(alert_id) or {}).get(
                    "min_override"))
            self.journal.log_note(
                f"mt5 pool: base layer risk {realized:.2f} exceeds "
                f"the position budget {budget:.2f} for ticket "
                f"{ticket} ({self._sizing_mode()} mode) — ledger "
                "widened to fit the filled order, headroom 0"
                + (" — exception: broker-minimum floor "
                   "(min_override): the broker minimum sizes above "
                   "the budget, so the filled order cannot fit it"
                   if _min_floor else
                   " — exception: placement sized past the envelope"))
            pool = RiskPool(max(budget, realized))
            pool.authorize(alert_id, lots, entry, stop,
                           int(direction), spec)
        self._pools[ticket] = pool
        return pool

    def _pool_book_layer(self, ticket: int, *, lots: float,
                         entry: float, stop: float, direction: int,
                         alert_id: str) -> bool:
        """Book one filled layer into the position's pool.
        Returns the authorize verdict (False = over the ceiling;
        the caller journals — a filled order is never un-filled)."""
        pool = self._pools.get(ticket)
        if pool is None:
            return True
        try:
            spec = self.executor.symbol_spec(self.cfg.symbol)
        except Exception:
            return True
        return pool.authorize(alert_id, lots, entry, stop,
                              int(direction), spec)

    def _pool_sync_stop(self, ticket: int, layer_key: str,
                        new_stop: float) -> None:
        """Roll a pool layer's stop (tighten-only; widening is
        ignored — pool accounting never loosens)."""
        pool = self._pools.get(ticket)
        if pool is None:
            return
        try:
            pool.move_stop(layer_key, new_stop)
        except Exception:
            pass

    def _run_signal_path(self, h1: pd.DataFrame, t: int,
                         states, years: float, verdict: dict,
                         stats: dict,
                         forming: dict | None = None,
                         lifecycle_routing: dict | None = None
                         ) -> None:
        """Verdict -> playbooks -> gate -> order (shared signal path).

        The verdict may come from a closed H1 bar (legacy) or from a
        zone touch + POI confirmation (live clock) — the path
        after the verdict is identical: same playbooks, same
        confluence gate, same hierarchy, manipulation filter
        (entry selection: no chasing a failed liquidity test),
        setup-dependent order (market / pending limit / pending
        stop), position tracked by the TickManager from the fill
        tick on.  When holding, a confirming same-direction signal
        becomes a LAYER (_try_layer), never a second position.

        XD-6 contract: ``h1`` is CLOSED bars only.  ``forming``
        is reaction context (the still-open bar, labelled
        ``forming: True`` on the verdict) — this path MUST NOT
        append it to the frame or hand it to any generator or
        scorer.  A caller that needs forming-bar data for
        reaction measurement reads ``verdict["forming"]``.
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
            self._emit_eval(
                playbook="__signal_path__",
                evaluation_clock="h1_trigger", intended_tf="H1",
                outcome="not_eligible", reason_code="path.paused",
                reason="engine paused — confirmed entry dropped; "
                       "no playbook was evaluated")
            return
        # Regime pause (corrupt-state rebuild): evaluate nothing
        # into orders — the confirmed entry is dropped, tick
        # management of the held position continues elsewhere.
        if self._regime_pause is not None:
            self._record_blocked(
                f"paused: regime pause ({self._regime_pause}) — "
                f"confirmed entry dropped")
            self.journal.log_note(
                "mt5: signal path regime-paused — confirmed entry "
                "dropped, no order")
            self._emit_eval(
                playbook="__signal_path__",
                evaluation_clock="h1_trigger", intended_tf="H1",
                outcome="not_eligible", reason_code="path.paused",
                reason=f"regime pause ({self._regime_pause}) — "
                       "no playbook was evaluated")
            return
        # Market-closed guard (signal processing): the tick pump
        # delivers nothing while the market is closed, so this is
        # defence-in-depth — it blocks the path if a tick ever
        # arrives while the symbol is disabled / close-only.
        open_ok, open_reason = self._market_open()
        if not open_ok:
            self._record_blocked(f"market_closed: {open_reason}")
            self.journal.log_note(
                "mt5: signal path market-closed — confirmed entry "
                f"dropped, no order ({open_reason})")
            self._emit_eval(
                playbook="__signal_path__",
                evaluation_clock="h1_trigger", intended_tf="H1",
                outcome="not_eligible",
                reason_code="path.market_closed",
                reason=f"market closed: {open_reason} — "
                       "no playbook was evaluated")
            return
        ctx = EngineContext(df=h1, states=states, years=years,
                            instrument=self.cfg.symbol)
        engine = self._get_engine()
        # Rework decision context, computed ONCE per verdict (not
        # per playbook): the graded zone behind the setup + the
        # funding-event report (chain evaluators as detectors).
        zone_d = self._zone_dict_for(
            verdict, int((verdict or {}).get("direction", 0) or 0))
        funding = None
        try:
            funding = self._funding_report(verdict, zone_d)
        except Exception as exc:
            self.journal.log_note(
                f"mt5: funding detectors failed ({exc}) — "
                "decisions proceed on score alone")
        # P3 (S9/F13): funding events EARN the held position's
        # breakeven — a structure/displacement/sweep print in the
        # position's direction is the desk's evidence that the
        # stop may move; the +1R price touch alone no longer
        # moves it (backstop excepted, in the tick manager).
        try:
            _tickets = self.tick_manager.tickets()
            if _tickets and funding is not None and funding.events:
                _pos = self.tick_manager.position(_tickets[0])
                if _pos is not None and int(_pos.direction) == int(
                        (verdict or {}).get("direction", 0) or 0):
                    _via = "funding:" + ",".join(sorted(
                        {e.kind for e in funding.events}))
                    if self.tick_manager.mark_be_earned(
                            _tickets[0], _via):
                        self.journal.log_note(
                            f"mt5: breakeven EARNED for ticket "
                            f"{_tickets[0]} ({_via}) — the stop "
                            "moves on the next tick (S9)")
                        log_signal(
                            self.journal,
                            signal_id=str(_pos.alert_id),
                            playbook=self._playbook_from_alert_id(
                                _pos.alert_id),
                            direction=int(_pos.direction),
                            score=float((self._position_ctx.get(
                                _tickets[0]) or {}).get(
                                "entry_total") or 0.0),
                            decision="management",
                            reason=f"be_earned: {_via}",
                            context=self._decision_context())
        except Exception:
            pass
        # Setup lifecycle routing (owner rulings 2026-10-30 §1):
        # armed scan-detected setups in the fired direction are
        # evaluated in this pass from their PERSISTED candidates
        # (oldest first, one per playbook per fire — the store
        # owns identity).  Expired-but-not-yet-upkept setups are
        # skipped here and expire on the next close's upkeep.
        _lc_candidates: dict = {}
        if lifecycle_routing is not None \
                and self._setup_store is not None:
            _lc_dir = int((verdict or {}).get("direction", 0) or 0)
            if _lc_dir in (1, -1) and t < len(h1):
                for _s in self._setup_store.active(
                        direction=_lc_dir, armed_only=True):
                    if _s.playbook in _lc_candidates:
                        continue
                    if _s.expiry_ts is not None:
                        try:
                            if pd.Timestamp(_s.expiry_ts) \
                                    <= h1.index[t]:
                                continue
                        except Exception:
                            pass
                    _lc_candidates[_s.playbook] = _s
        for pb in self._get_playbooks():
            if pb.name in (self.cfg.disabled_playbooks or ()):
                # Kill switch (/set) — not evaluated (A14).  FUNNEL:
                # the skip is a record, not a silence.
                self._emit_eval(
                    playbook=pb.name, evaluation_clock="h1_trigger",
                    intended_tf=_native_tf(pb.name),
                    outcome="not_eligible",
                    reason_code="path.kill_switch",
                    reason="disabled via /set kill switch — "
                           "not evaluated")
                if lifecycle_routing is not None \
                        and pb.name in _lc_candidates:
                    lifecycle_routing[pb.name] = {
                        "status": "skipped_disabled",
                        "setup_id":
                            _lc_candidates[pb.name].setup_id,
                        "terminals": []}
                continue
            try:
                _inj = _lc_candidates.get(pb.name)
                _terms = self._run_one_playbook(
                    pb, sig_ctx=(ctx, engine),
                    h1=h1, t=t, verdict=verdict,
                    stats=stats, zone_d=zone_d,
                    funding=funding,
                    injected_sig=(self._rebuild_setup_signal(_inj, t)
                                  if _inj is not None else None),
                    injected_provenance=(
                        self._setup_provenance(_inj, h1, t)
                        if _inj is not None else None),
                    injected_setup_id=(_inj.setup_id
                                       if _inj is not None else None),
                    injected_parent_setup_id=(
                        _inj.parent_setup_id
                        if _inj is not None else None))
                if lifecycle_routing is not None \
                        and _inj is not None:
                    lifecycle_routing[pb.name] = {
                        "status": "evaluated",
                        "setup_id": _inj.setup_id,
                        "terminals": _terms or []}
            except Exception as exc:
                # A crashing playbook never kills the signal path.
                self._record_blocked(
                    f"playbook_error: {type(exc).__name__}: {exc}",
                    pb.name)
                self.journal.log_note(
                    f"mt5: {pb.name} signal path failed: "
                    f"{type(exc).__name__}: {exc}")
                self._emit_eval(
                    playbook=pb.name, evaluation_clock="h1_trigger",
                    intended_tf=_native_tf(pb.name),
                    actual_input_tf="H1",
                    last_bar_ts=h1.index[t] if t < len(h1) else None,
                    outcome="exception",
                    reason_code="path.exception",
                    reason=f"{type(exc).__name__}: {exc}")
                if lifecycle_routing is not None \
                        and pb.name in _lc_candidates:
                    lifecycle_routing[pb.name] = {
                        "status": "exception",
                        "setup_id":
                            _lc_candidates[pb.name].setup_id,
                        "terminals": []}

    def _run_one_playbook(self, pb, *, sig_ctx, h1, t, verdict, stats,
                          zone_d=None, funding=None,
                          injected_sig=None, injected_provenance=None,
                          injected_setup_id=None,
                          injected_parent_setup_id=None):
        """One playbook through the REWORK decision flow (spec §3).

        Per signal: Asymmetry Score (live/asymmetry.py) -> hard
        lines only (declaration gates via the engine's
        hard_line_check + the manipulation filter) -> per-style
        score floor (the flat min_score and the hierarchy/chain
        permission gates are GONE on this path: the verdict feeds
        the Macro block + traffic scalar, the chain evaluators
        feed funding events) -> order / layer.  A14: every
        evaluated playbook can execute — there is no shadow
        branch; a signal is authorized, floored, gated by a hard
        line, parked in PullbackWatch, or the playbook is off.
        """
        ctx, engine = sig_ctx
        # FUNNEL §1.1: seq snapshot — if this evaluation produces
        # no per-signal record by the end, it was a no_setup
        # evaluation and gets one (evaluated-and-empty must be
        # distinguishable from never-called).
        _funnel_seq0 = self._funnel_seq
        # Setup lifecycle (owner rulings 2026-10-30 §1): when a
        # persisted setup is routed into this evaluation, its
        # terminal decisions are collected and returned so the
        # caller can advance the setup's lifecycle.  Normal
        # (fresh-detection) calls ignore the return value.
        injected_terminals: list = []
        # Range playbooks need the native H4 frame (threaded in
        # here — no H4-resample fallback anywhere on this path).
        if hasattr(pb, "set_h4_frame"):
            pb.set_h4_frame(self._stack.frames.get("H4"))
        # Positional, as before: playbook live_signal signatures name
        # the frame parameter differently (df / h1).
        if injected_sig is not None:
            # Lifecycle routing: the persisted candidate IS the
            # signal — no fresh re-detection, no signal-gap
            # dedupe (the lifecycle store owns this setup's
            # identity and dedupe).
            sigs = [injected_sig]
        else:
            sigs = pb.live_signal(ctx.states, t, h1) or []
        for sig in sigs:
            alert_id = f"MT5-{pb.name}-{sig.index}"
            if injected_sig is None and \
                    sig.index - self._last_sig_bar.get(
                        pb.name, -10 ** 9) \
                    < self.cfg.signal_gap_bars:
                continue
            # Step 15: stamp the candidate's declared execution
            # intent — every funnel record for this candidate
            # carries it (_candidate_dict reads the attribute) and
            # the shared order path enforces it at selection.
            sig.execution_intent = execution_intent_for(pb.name, sig)
            # ---- P3 style/family + journey pre-pass -------------
            # The style bucket picks the floated clocks + the
            # open-space floor row; the family is the posture unit.
            # With NO zone context the trade is open-space: the
            # journey map (zones both sides) decides with/counter
            # -journey geometry, and a scalp whose magnet sits
            # within reach is a scalp-INTO-zone — the approached
            # zone becomes its scoring context.
            style_key = self._style_key_for(pb.name, zone_d)
            family = _posture_mod.family_of(
                pb.name, str(getattr(pb, "regime_scope", "") or ""))
            atr_h1 = float(atr14(h1)) if len(h1) else 0.0
            _atr = atr_h1 or float(getattr(sig, "atr", 0.0) or 0.0)
            stop_dist = abs(float(sig.entry_ref)
                            - float(sig.invalidation)) or _atr
            scalp_into_zone = False
            journey = self._journey_context(
                direction=int(sig.direction),
                price=float(sig.entry_ref), stop_dist=stop_dist,
                zone_d=zone_d, atr=_atr)
            if zone_d is None and style_key == "scalp" \
                    and journey.get("magnet") is not None \
                    and journey.get("magnet_dist_atr") is not None \
                    and journey["magnet_dist_atr"] <= 1.5:
                zone_d = journey["magnet"]
                scalp_into_zone = True
                # The scalp's geometry IS the magnet: the generic
                # journey-room rule does not refuse it (its floor
                # + defended stop govern instead).
                journey["refusal"] = None
                journey["state"] = "scalp_into_zone"
            # ---- Asymmetry Score (the decision number) ----------
            score = self._score_for(
                direction=int(sig.direction), verdict=verdict,
                price=float(sig.entry_ref), zone_d=zone_d,
                atr=_atr, style_key=style_key, playbook=pb.name)
            breakdown = score.breakdown()
            if self._last_clock is not None:
                breakdown["frame_tf"] = self._last_clock.score_tf
                breakdown["trigger_tf"] = self._last_clock.trigger_tf
                breakdown["clock_reason"] = self._last_clock.reason
            if self._posture is not None:
                breakdown["posture"] = self._posture.name
            if self._day_ctx is not None:
                breakdown["day_type"] = self._day_ctx.day_type
            self._last_scores[pb.name] = breakdown
            self._last_score = breakdown

            def _emit_sig(*, result, reason_code="", reason="",
                          floor_result=None, execution_eligible=None,
                          order_result=None, selected_action=None,
                          rejected_alternatives=None):
                """FUNNEL §1.1: this signal's terminal decision
                record (mirrors the log_signal beside it).

                Burn-down #1: the record carries the full
                assessment objects the native routes produce —
                the evidence the decider actually weighed for
                THIS candidate (signal factors, the asymmetry
                score's stance, HTF hierarchy agreement, zone
                context, journey geometry, funding events), with
                correlation groups, plus the two competing
                scenarios.  Built from closure state at emit
                time; observation-only — no value here is read
                by any decision."""
                if injected_sig is not None:
                    injected_terminals.append(
                        {"result": result,
                         "reason_code": reason_code or "",
                         "order_result": order_result,
                         "alert_id": alert_id})
                dec = {"result": result,
                       "reason_code": reason_code or None,
                       "reason": reason}
                if floor_result is not None:
                    dec["floor_result"] = floor_result
                if execution_eligible is not None:
                    dec["execution_eligible"] = execution_eligible
                if order_result is not None:
                    dec["order_result"] = order_result
                # Two verdict layers, distinct (owner ruling):
                # analytical qualification vs risk-layer
                # eligibility, derived from this record's own
                # terminal fields.
                _vb_eligible = (
                    order_result in ("filled", "placed_pending",
                                     "layer_filled", "layer_pending",
                                     "layer_no_order")
                    if order_result is not None
                    else (bool(execution_eligible)
                          if execution_eligible is not None
                          else result == "accepted"))
                dec.update(_verdict_blocks(
                    analytical=("qualified" if result == "accepted"
                                else "not_qualified"),
                    analytical_code=(None if result == "accepted"
                                     else (reason_code or None)),
                    eligible=_vb_eligible,
                    execution_code=(
                        (reason_code or None)
                        if order_result in (
                            "refused_in_order_path", "rejected")
                        else None),
                    outcome=order_result or "not_attempted"))
                dec.setdefault(
                    "floor_authority",
                    _FLOOR_AUTHORITY["h1_roster"])
                # --- assessment objects (burn-down #1) --------
                _dir = int(sig.direction)
                _entry = float(sig.entry_ref)
                _stop = float(sig.invalidation)
                _sup: list[str] = []
                _con: list[str] = []
                _groups: dict[str, list[str]] = {}

                def _note(bucket, group, text):
                    bucket.append(text)
                    _groups.setdefault(group, []).append(text)

                for _f in (getattr(sig, "factors", None)
                           or []):
                    _note(_sup, f"{pb.name}_signal",
                          f"signal factor: {_f}")
                if bool(score.flipped):
                    _note(_con, "asymmetry_score",
                          "asymmetry score flipped: the weighed "
                          "structure funds the opposite side")
                else:
                    _note(_sup, "asymmetry_score",
                          f"asymmetry score {score.total:.0f} "
                          f"for the {_dir:+d} side")
                _bias_d1 = (verdict or {}).get("bias_d1")
                if _bias_d1:
                    if int(_bias_d1) == _dir:
                        _note(_sup, "htf_structure",
                              f"with structural D1 bias "
                              f"{int(_bias_d1):+d}")
                    else:
                        _note(_con, "htf_structure",
                              f"against structural D1 bias "
                              f"{int(_bias_d1):+d} (weighed as "
                              "evidence, not a veto)")
                if zone_d is not None:
                    _lc = str(zone_d.get("lifecycle") or "")
                    _zt = "zone context: "
                    if _lc:
                        _zt += f"{_lc} "
                    _zt += (f"zone [{zone_d.get('bottom')}, "
                            f"{zone_d.get('top')}]")
                    _note(_con if _lc == "used_up" else _sup,
                          "zone_context", _zt)
                if journey.get("refusal"):
                    _note(_con, "journey_geometry",
                          f"journey geometry: "
                          f"{journey['refusal']}")
                if funding is not None:
                    for _e in (funding.events or []):
                        _note(_sup, "funding_events",
                              f"funding event: {_e.kind}"
                              + (f" — {_e.detail}"
                                 if getattr(_e, "detail", "")
                                 else ""))
                    if funding.runaway:
                        _note(_con, "funding_events",
                              "funding runaway: confirmation "
                              "said price ran from the zone")
                _observed = {
                    "entry": _entry, "stop": _stop,
                    "stop_dist": stop_dist,
                    "bias_d1": (verdict or {}).get("bias_d1"),
                    "bias_h4": (verdict or {}).get("bias_h4")}
                if t < len(h1):
                    try:
                        _observed["session_bucket"] = \
                            session_of(h1.index[t])[0]
                    except Exception:
                        pass
                _inferred = {
                    "score_total": float(score.total),
                    "score_components": breakdown,
                    "score_flipped": bool(score.flipped),
                    "regime": (verdict or {}).get("regime"),
                    "posture": (self._posture.name
                                if self._posture is not None
                                else None),
                    "alignment": getattr(
                        self._alignment, "alignment_label",
                        None),
                    "journey_state": journey.get("state"),
                    "room_r": journey.get("room_r")}
                dec["competing_scenarios"] = [
                    {"scenario": f"take the {_dir:+d} "
                                 f"{pb.name} setup at "
                                 f"{_entry:.2f}",
                     "status": (
                         "selected" if result == "accepted"
                         else "parked — PullbackWatch armed (§5)"
                         if result == "watch"
                         else f"not taken — {result}"
                         + (f" ({reason_code})"
                            if reason_code else ""))},
                    {"scenario": f"take the {-_dir:+d} side",
                     "status": (
                         "funded by the weighed evidence — the "
                         "asymmetry score flipped to this side"
                         if bool(score.flipped) else
                         "not funded — no opposite-side "
                         "candidate from this playbook at this "
                         "bar; the score did not flip")}]
                if selected_action:
                    dec["selected_action"] = selected_action
                if rejected_alternatives:
                    dec["rejected_alternatives"] = list(
                        rejected_alternatives)
                _ev = {"route": "h1_trigger",
                       "hierarchy": hier_rec,
                       "observed": _observed,
                       "inferred": _inferred,
                       "supporting": _sup,
                       "contradictory": _con,
                       "evidence_groups": _groups,
                       "independent_evidence_groups":
                           sorted(_groups)}
                if injected_sig is not None:
                    # The decision record joins the setup's
                    # lifecycle chain by ID (owner rulings
                    # 2026-10-30 §1).
                    _ev["setup_id"] = injected_setup_id
                    _ev["parent_setup_id"] = \
                        injected_parent_setup_id
                    _ev["lifecycle_routed"] = True
                return self._emit_eval(
                    playbook=pb.name,
                    evaluation_clock="h1_trigger",
                    intended_tf=_native_tf(pb.name) or "H1",
                    actual_input_tf="H1",
                    last_bar_ts=h1.index[t] if t < len(h1) else None,
                    outcome="candidate_emitted",
                    reason_code=reason_code, reason=reason,
                    candidate=self._candidate_dict(
                        sig, breakdown,
                        injected_provenance
                        if injected_sig is not None else None),
                    decision=dec,
                    # Phase 2 (addendum B/C wiring): the HTF
                    # hierarchy context was computed for every
                    # roster candidate and carried on the SIGNAL
                    # record, but never reached the FUNNEL
                    # record — HTF disagreement is now explicit
                    # evidence there too, never a silent factor.
                    # Burn-down #1 completes the object: observed
                    # geometry + bias/session, the inferred
                    # structure read (score/regime/posture/
                    # alignment/journey), itemized supporting vs
                    # contradictory evidence with correlation
                    # groups — the same shape the native routes
                    # journal.  (Built as _ev above so a
                    # lifecycle-routed record can join its
                    # setup's chain by ID.)
                    evidence=_ev,
                    source="trigger_path")

            hier_rec = {
                "bias_d1": (verdict or {}).get("bias_d1"),
                "bias_h4": (verdict or {}).get("bias_h4"),
                "trigger": (verdict or {}).get("trigger"),
                "entry_clock": (verdict or {}).get("entry_clock"),
            }
            if score.flipped:
                # The score object is for the OTHER side: it can
                # never fund this direction (spec §2.1 flip rule).
                reason = ("asymmetry_flip: close through the last "
                          "defended pivot — structure funds the "
                          "opposite side only")
                self._record_blocked(reason, pb.name)
                self.journal.log_note(
                    f"mt5: {pb.name} signal refused: {reason}")
                log_signal(
                    self.journal, signal_id=alert_id, playbook=pb.name,
                    direction=int(sig.direction), score=float(sig.score),
                    decision="gated", reason=reason,
                    bar_ts=h1.index[t] if t < len(h1) else None,
                    entry=float(sig.entry_ref),
                    stop=float(sig.invalidation),
                    targets=[float(x) for x in (sig.targets or [])],
                    factors=list(getattr(sig, "factors", []) or []),
                    hierarchy=hier_rec, score_breakdown=breakdown,
                    context=self._decision_context())
                _emit_sig(result="refused",
                          reason_code="direction.flip", reason=reason)
                continue
            if zone_d is not None \
                    and str(zone_d.get("lifecycle") or "") == "used_up":
                # A2: a used-up zone is abandoned — not traded
                # even on confirmation.
                reason = ("zone_used_up: armed zone lifecycle is "
                          "used_up (touches consume; A2 abandon)")
                self._record_blocked(reason, pb.name)
                self.journal.log_note(
                    f"mt5: {pb.name} signal refused: {reason}")
                log_signal(
                    self.journal, signal_id=alert_id, playbook=pb.name,
                    direction=int(sig.direction), score=float(sig.score),
                    decision="gated", reason=reason,
                    bar_ts=h1.index[t] if t < len(h1) else None,
                    entry=float(sig.entry_ref),
                    stop=float(sig.invalidation),
                    targets=[float(x) for x in (sig.targets or [])],
                    factors=list(getattr(sig, "factors", []) or []),
                    hierarchy=hier_rec, score_breakdown=breakdown,
                    context=self._decision_context())
                _emit_sig(result="refused",
                          reason_code="gate.location", reason=reason)
                continue
            # ---- hard lines: declarations + cost ----------------
            # The real engine exposes hard_line_check (rework).
            # Pre-rework engine doubles (older test stubs) fall
            # back to gate_live_signal with the chain requirement
            # waived — production always takes the first branch.
            _hl = getattr(engine, "hard_line_check", None)
            if _hl is not None:
                ok, reason = _hl(
                    pb, sig, ctx,
                    spread_px=(self._last_tick or {}).get("spread"))
            else:
                try:
                    ok, reason = engine.gate_live_signal(
                        pb, sig, ctx, hierarchy_verdict=verdict,
                        spread_px=(self._last_tick or {}).get(
                            "spread"),
                        chain_required=False)
                except TypeError:
                    ok, reason = engine.gate_live_signal(
                        pb, sig, ctx, hierarchy_verdict=verdict,
                        spread_px=(self._last_tick or {}).get(
                            "spread"))
            if not ok:
                stats["gated"] += 1
                self._record_blocked(reason, pb.name)
                self.journal.log_note(
                    f"mt5: {pb.name} signal gated: {reason}")
                log_signal(
                    self.journal, signal_id=alert_id, playbook=pb.name,
                    direction=int(sig.direction), score=float(sig.score),
                    decision="gated", reason=reason,
                    bar_ts=h1.index[t] if t < len(h1) else None,
                    entry=float(sig.entry_ref),
                    stop=float(sig.invalidation),
                    targets=[float(x) for x in (sig.targets or [])],
                    factors=list(getattr(sig, "factors", []) or []),
                    hierarchy=hier_rec, score_breakdown=breakdown,
                    context=self._decision_context())
                _emit_sig(result="refused",
                          reason_code=_funnel_classify(reason),
                          reason=reason)
                continue
            # ---- P3 posture gate (S1/S3/S11/S12) ----------------
            # The composed scenario posture rules BEFORE the floor
            # (eligibility precedes quality): entry-blocking
            # postures (spike / dead window / loss cooldown),
            # family shelvings, per-style churn ceilings, and the
            # floated clock's stand-aside.  Every refusal is a
            # JOURNALED posture with its reason — never silent.
            post = self._posture
            if post is not None:
                # bedrock.direction speaks "up"/"down"/"flat"
                # (regime.py) — normalize through posture.dir_int,
                # never a raw int() (int("flat") crashed the tick
                # signal path: the exception was swallowed by
                # _run_signal_path and the fired trigger died
                # journaled only as "signal path failed").
                _trend_dir = _posture_mod.dir_int(
                    getattr(self._bedrock, "direction", 0)) \
                    if self._bedrock is not None else 0
                _block = None
                if post.block_entries:
                    _block = f"posture {post.name}: {post.reason}"
                else:
                    _shelved = post.shelved_reason(
                        family, direction=int(sig.direction),
                        trend_direction=_trend_dir)
                    if _shelved:
                        _block = _shelved
                    elif post.ladder is not None and \
                            self._style_bucket(style_key) in \
                            post.ladder.style_blocked:
                        _block = (
                            f"posture churn ceiling: style "
                            f"{self._style_bucket(style_key)} hit "
                            f"its daily ceiling of "
                            f"{self.cfg.max_daily_trades_per_style} "
                            "opens (S11 — the style is done for "
                            "today, not the engine)")
                    elif self._last_clock is not None \
                            and self._last_clock.stand_aside:
                        _block = f"clock float: {self._last_clock.reason}"
                if _block is not None:
                    stats["gated"] = stats.get("gated", 0) + 1
                    self._record_blocked(_block, pb.name)
                    self.journal.log_note(
                        f"mt5: {pb.name} signal posture-gated: "
                        f"{_block}")
                    log_signal(
                        self.journal, signal_id=alert_id,
                        playbook=pb.name,
                        direction=int(sig.direction),
                        score=float(sig.score), decision="gated",
                        reason=_block,
                        bar_ts=h1.index[t] if t < len(h1) else None,
                        entry=float(sig.entry_ref),
                        stop=float(sig.invalidation),
                        targets=[float(x) for x in
                                 (sig.targets or [])],
                        factors=list(getattr(sig, "factors", [])
                                     or []),
                        hierarchy=hier_rec,
                        score_breakdown=breakdown,
                        context=self._decision_context())
                    _emit_sig(result="refused",
                              reason_code="regime.scope", reason=_block)
                    continue
            # ---- P3 SMT funding leg (S7) -------------------------
            # A live SMT event backing a REVERSAL playbook at a
            # level joins this signal's funding report (colour
            # +5) — composed per playbook, the shared verdict
            # report is never mutated.
            if funding is not None and self._smt_read is not None:
                _smt_ev = smt_funding_event(
                    self._smt_read.event, playbook=pb.name,
                    direction=int(sig.direction),
                    at_level=zone_d is not None)
                if _smt_ev is not None:
                    funding = FundingReport(
                        events=[*funding.events, _smt_ev],
                        runaway=funding.runaway,
                        confirm_decision=funding.confirm_decision,
                        logs=funding.logs)
            # ---- per-style floor (was: flat min_score) ----------
            # The floor is defined for zone-path trades (P1 is the
            # arrival pass).  A signal with NO zone context is not
            # an arrival-path signal — production verdicts always
            # carry the setup's zone, so this branch is a stated
            # scope boundary, journaled when it happens.
            #
            # Routing (by the trade's actual style, in priority
            # order): (1) RANGE — verdict regime, exactly as
            # before; range setups never carry an arm and are
            # mutually exclusive with the continuation arm.
            # (2) CONTINUATION — the hierarchy F2 arm: the verdict's
            # ``arm`` marker (threaded by _live_verdict from the
            # arming setup dict — live_setup is its only source;
            # the closed-bar verdict from evaluate_hierarchy has
            # no arm concept, so no continuation can arise there),
            # a verdict ``style`` marker, a nested verdict
            # ``setup`` dict's arm/style, or a playbook-declared
            # ``style`` naming the continuation row.  Playbooks
            # today declare only regime_scope (base.py), never a
            # STYLE_FLOORS style, so that last check is inert until
            # one does.  (3) A playbook/verdict style that names a
            # STYLE_FLOORS row verbatim.  (4) ZONE_PATH default —
            # the P1 arrival path, whose Location >= 12 condition
            # is a zone-arrival trade-TYPE condition and stays
            # exactly as-is for arrival signals.
            _v = verdict or {}
            _nested = _v.get("setup") if isinstance(
                _v.get("setup"), dict) else {}
            _arm = str(_v.get("arm") or _nested.get("arm") or "")
            _style = str(_v.get("style") or _nested.get("style")
                         or getattr(pb, "style", "") or "")
            floor_exempt = False
            if str(_v.get("regime", "")) == "range":
                path = "RANGE"
            elif _arm == "continuation" \
                    or _style in ("continuation",
                                  "INTRADAY_CONTINUATION"):
                path = "INTRADAY_CONTINUATION"
            elif _style in STYLE_FLOORS:
                path = _style
            elif scalp_into_zone:
                path = "MICRO_SCALP_OPEN"
            elif zone_d is None:
                # P3 OPEN SPACE (G): a DECLARED open-space trade
                # floors on its style row (Path B: total +
                # component minimums + Structure+Efficiency
                # >= 35), and the journey geometry below rules
                # the map between zones.  Declaration is a
                # positive marker: an open-space style bucket
                # (swing / scalp / qm_d1 — the map below) or the
                # scenario verdict's declared trend regime (the
                # §3.3 mid-range continuation).  A signal with NO
                # zone context and NO open-space declaration is
                # not an open-space trade — it is the residue the
                # P1 scope boundary covered ("not an arrival-path
                # signal; hard lines only"), and defaulting it
                # onto the continuation row floored trades the
                # desk never classified (the P3 regression: the
                # Golden Swing ceremony, the manipulation hard
                # line and reasoning attachment all sit downstream
                # of this check).  The boundary is restored for
                # the undeclared residue only; every declared
                # open-space trade keeps its Path B floor.
                if style_key in ("swing", "scalp", "qm_d1") \
                        or str(_v.get("regime", "")) == "trend":
                    path = {"swing": "MACRO_SWING_OPEN",
                            "qm_d1": "MACRO_SWING_OPEN",
                            "scalp": "MICRO_SCALP_OPEN"}.get(
                                style_key, "INTRADAY_CONTINUATION")
                else:
                    path = "ZONE_PATH"
                    floor_exempt = True
            else:
                path = "ZONE_PATH"
            fl = floor_for(path)
            floor_ok = True if floor_exempt else self._floor_ok(
                fl, score)
            if floor_exempt:
                self.journal.log_note(
                    f"mt5: {pb.name} signal has no zone context "
                    "— asymmetry floor not applied (not an "
                    "arrival-path signal); hard lines only")
            elif zone_d is None:
                self.journal.log_note(
                    f"mt5: {pb.name} open-space signal — floored "
                    f"on {path} (Path B), journey "
                    f"{journey.get('state')}"
                    + (f", room {journey['room_r']:.1f}R to the "
                       "magnet" if journey.get("room_r")
                       is not None else ""))
            # Journey geometry refusal (G): with-journey room or
            # counter-journey rules — journaled with the reason,
            # binding on layers too (geometry does not care which
            # leg it is).
            if journey.get("refusal"):
                stats["gated"] = stats.get("gated", 0) + 1
                self._record_blocked(journey["refusal"], pb.name)
                self.journal.log_note(
                    f"mt5: {pb.name} signal journey-refused: "
                    f"{journey['refusal']}")
                log_signal(
                    self.journal, signal_id=alert_id,
                    playbook=pb.name, direction=int(sig.direction),
                    score=float(sig.score), decision="gated",
                    reason=journey["refusal"],
                    bar_ts=h1.index[t] if t < len(h1) else None,
                    entry=float(sig.entry_ref),
                    stop=float(sig.invalidation),
                    targets=[float(x) for x in (sig.targets or [])],
                    factors=list(getattr(sig, "factors", []) or []),
                    hierarchy=hier_rec, score_breakdown=breakdown,
                    context=self._decision_context())
                _emit_sig(result="refused",
                          reason_code="gate.location",
                          reason=journey["refusal"])
                continue
            holding = bool(self.tick_manager.tickets())
            if not floor_ok and not holding:
                reason = (
                    f"asymmetry_floor: total {score.total:.0f} < "
                    f"{fl['floor']:.0f} or component minimum unmet "
                    f"(location {score.component('location'):.1f} "
                    f">= {fl.get('min_location', 0.0):.0f} required "
                    f"on {path})")
                # §5 PullbackWatch: a runaway (confirmation said
                # price ran) parks the setup instead of resetting
                # it — the watch awaits the pullback.
                if funding is not None and funding.runaway \
                        and zone_d is not None:
                    self._enter_pullback_watch(
                        pb, sig, verdict, zone_d, score, funding)
                    log_signal(
                        self.journal, signal_id=alert_id,
                        playbook=pb.name,
                        direction=int(sig.direction),
                        score=float(sig.score), decision="watch",
                        reason="runaway: price ran from the zone — "
                               "PullbackWatch armed (§5)",
                        bar_ts=h1.index[t] if t < len(h1) else None,
                        entry=float(sig.entry_ref),
                        stop=float(sig.invalidation),
                        targets=[float(x) for x in (sig.targets or [])],
                        factors=list(getattr(sig, "factors", [])
                                     or []),
                        hierarchy=hier_rec, score_breakdown=breakdown,
                        trigger_source="arrival_score",
                        context=self._decision_context())
                    _emit_sig(
                        result="watch",
                        reason_code="path.pullback_watch",
                        reason="runaway: price ran from the zone — "
                               "PullbackWatch armed (§5)",
                        floor_result={
                            "floor": float(fl["floor"]),
                            "total": float(score.total), "path": path},
                        selected_action="park the setup in "
                                        "PullbackWatch (§5) and "
                                        "await the pullback — do "
                                        "not chase the runaway")
                    continue
                self._record_blocked(
                    f"min_score: {reason}", pb.name)
                stats["below_floor"] = stats.get("below_floor", 0) + 1
                log_signal(
                    self.journal, signal_id=alert_id, playbook=pb.name,
                    direction=int(sig.direction), score=float(sig.score),
                    decision="below_floor", reason=reason,
                    bar_ts=h1.index[t] if t < len(h1) else None,
                    entry=float(sig.entry_ref),
                    stop=float(sig.invalidation),
                    targets=[float(x) for x in (sig.targets or [])],
                    factors=list(getattr(sig, "factors", []) or []),
                    hierarchy=hier_rec, score_breakdown=breakdown,
                    context=self._decision_context())
                _emit_sig(
                    result="refused",
                    reason_code=("floor.total"
                                 if float(score.total)
                                 < float(fl["floor"])
                                 else "floor.component"),
                    reason=reason,
                    floor_result={"floor": float(fl["floor"]),
                                  "total": float(score.total),
                                  "path": path})
                continue
            conf, notes = engine.score_live_signal(
                pb, sig, ctx, hierarchy_verdict=verdict)
            trigger_source = "funding_layer" if (
                holding and funding is not None
                and funding.events) else "arrival_score"
            # S5 label (P3): a reversal entry inside the
            # post-print reaction window, at a level, is the
            # desk's spike-fade shape — the record says so.
            if trigger_source == "arrival_score" \
                    and pb.name in _SMT_PLAYBOOKS \
                    and zone_d is not None \
                    and self.cfg.news_pause_enabled:
                try:
                    if self.news.reaction_state(
                            self._tick_datetime()) is not None:
                        trigger_source = "post_print_fade"
                except Exception:
                    pass
            self._last_sig_bar[pb.name] = sig.index
            stats["signals"] += 1
            self._signal_counts[pb.name] = \
                self._signal_counts.get(pb.name, 0) + 1
            auto = self._executes(pb.name)
            tag = "MT5"
            # Dynamic reasoning + study package (accepted signals
            # only): the why + the key levels/data behind it — on
            # the signal record and in the Telegram alert (with the
            # chart PNG, the visual half of the study).
            reasoning = study = None
            if auto:
                reasoning = build_reasoning(
                    pb.name, sig, verdict,
                    alignment_label=getattr(
                        self._alignment, "alignment_label", None))
                study = build_study(
                    pb.name, sig, verdict,
                    alignment_label=getattr(
                        self._alignment, "alignment_label", None))
            self.journal.log_note(
                f"mt5 [{tag}] {pb.name} "
                f"{'+' if sig.direction > 0 else '-'}1 "
                f"score={sig.score:.0f} asym={score.total:.0f} "
                f"conf={conf:.0f} "
                f"entry~{sig.entry_ref:.1f} stop~{sig.invalidation:.1f} "
                f"factors={'+'.join(notes) if notes else '-'} "
                f"hierarchy: D1 {verdict['bias_d1']:+d} / H4 "
                f"{verdict['bias_h4']:+d} / {verdict['trigger']} / "
                f"{verdict['entry_clock']} amd:{verdict.get('amd_phase')} "
                f"qm:{verdict.get('qm', {}).get('status')} "
                f"vol:{(verdict.get('volume') or {}).get('volume_ratio')}")
            log_signal(
                self.journal, signal_id=alert_id, playbook=pb.name,
                direction=int(sig.direction), score=float(sig.score),
                decision="accepted" if auto else "shadow",
                confluence=float(conf),
                bar_ts=h1.index[t] if t < len(h1) else None,
                entry=float(sig.entry_ref),
                stop=float(sig.invalidation),
                targets=[float(x) for x in (sig.targets or [])],
                factors=list(notes),
                dxy_corr=(verdict or {}).get("dxy_corr"),
                urgency=(verdict or {}).get("urgency"),
                reasoning=reasoning,
                study=study,
                hierarchy=hier_rec,
                score_breakdown=breakdown,
                trigger_source=trigger_source,
                context=self._decision_context())
            if not auto:
                # Config-narrowed execute list (A14 override): the
                # signal was evaluated and logged; the owner's
                # explicit tuple keeps it from ordering.
                _emit_sig(result="not_executable",
                          reason_code="path.execute_list",
                          reason="config execute_playbooks tuple "
                                 "excludes this playbook "
                                 "(A14 override)",
                          execution_eligible=False,
                          order_result="not_attempted",
                          selected_action="journal the evaluated "
                                          "signal only — the "
                                          "owner's execute list "
                                          "keeps it from ordering "
                                          "(A14 override)")
                continue
            # Chart: H4 zone + entry/stop/targets + verdict, saved
            # next to the journal and attached to the alert.
            # Step 7: the journal score here IS sig.score on this
            # path (log_signal above records score=float(sig.score)),
            # so passing it explicitly is the journal score, not a
            # placeholder; no broker stop exists pre-order.
            png, _path = self._build_chart(
                sig, verdict, alert_id=alert_id, playbook=pb.name,
                decision_score=float(sig.score))
            _, view_path = self._build_signal_view(
                sig, verdict, alert_id=alert_id, playbook=pb.name,
                decision_score=float(sig.score))
            self._send_alert(
                f"MT5 [{tag}] {pb.name} "
                f"{'+' if sig.direction > 0 else '-'}1 "
                f"entry~{sig.entry_ref:.1f} stop~{sig.invalidation:.1f} "
                f"conf={conf:.0f}"
                + (f"\n\n{reasoning}" if reasoning else ""),
                chart_png=png, html_path=view_path)
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
                    log_signal(
                        self.journal, signal_id=alert_id,
                        playbook=pb.name,
                        direction=int(sig.direction),
                        score=float(sig.score),
                        decision="manipulation_blocked", reason=why,
                        confluence=float(conf),
                        bar_ts=h1.index[t] if t < len(h1) else None,
                        entry=float(sig.entry_ref),
                        stop=float(sig.invalidation),
                        targets=[float(x) for x in (sig.targets or [])],
                        factors=list(notes),
                        context=self._decision_context())
                    _emit_sig(result="refused",
                              reason_code="path.manipulation_blocked",
                              reason=why, execution_eligible=True,
                              order_result="not_attempted")
                    continue
            # FUNNEL §1.1: snapshot the order-path counters so the
            # terminal record can state the order outcome from
            # observed effects (observation-only — reads, never
            # steers).
            _pre_order = (stats.get("executed", 0),
                          len(self._pending_orders),
                          stats.get("invalidated", 0),
                          stats.get("gate_friction", 0),
                          stats.get("sizing_declined", 0))
            _branch = ("layer" if self.tick_manager.tickets()
                       else "order")
            if self.tick_manager.tickets():
                self._try_layer(pb, sig, stats, verdict=verdict,
                                entry_bar=t, score=score,
                                funding=funding, zone_d=zone_d)
            elif self.cfg.golden_swing_confirm_enabled \
                    and self._alignment.golden_swing:
                _branch = "confirm"
                # Golden Swing base order: the order WAITS for the
                # owner's Confirm tap (REJECT cancels; timeout
                # fires/cancels per config).  Nothing is placed
                # here.  Layers never come here — a held position
                # routes to _try_layer above.
                self._stage_order_confirm(pb, sig, stats,
                                          verdict=verdict, entry_bar=t)
            else:
                # P3: park the decision context for the fill +
                # price the A8 zone TP ladder for the order.
                _sig_ctx = {
                    "playbook": pb.name, "style_key": style_key,
                    "style_bucket": self._style_bucket(style_key),
                    "family": family, "floor": float(fl["floor"]),
                    "path": path,
                    "entry_total": float(score.total),
                    "zone_d": (dict(zone_d) if zone_d else None),
                    "clock_tf": (self._last_clock.trigger_tf
                                 if self._last_clock else "H1"),
                    "defended_stop": (
                        float(sig.invalidation) if path in (
                            "INTRADAY_CONTINUATION",
                            "MACRO_SWING_OPEN", "MICRO_SCALP_OPEN")
                        else None),
                }
                _ladder_targets = self._zone_tp_targets(
                    sig, zone_d)
                if _ladder_targets:
                    _sig_ctx["targets"] = _ladder_targets
                self._place_live_order(
                    pb, sig, stats, verdict=verdict,
                    entry_bar=t, score=score, zone_d=zone_d,
                    funding=funding, signal_ctx=_sig_ctx)
            # FUNNEL §1.1: the accepted signal's terminal record —
            # order outcome read from the counters snapshotted
            # before dispatch (filled / pending placed / rejected
            # inside the order path / staged for confirm / layer
            # produced no order).  The refusal detail itself is
            # journaled by the order path, as before.
            if _branch == "confirm":
                _ores = "staged_confirm"
            elif stats.get("executed", 0) > _pre_order[0]:
                _ores = "filled"
            elif len(self._pending_orders) > _pre_order[1]:
                _ores = "placed_pending"
            elif stats.get("invalidated", 0) > _pre_order[2]:
                _ores = "rejected"
            elif _branch == "layer":
                _ores = "layer_no_order"
            else:
                _ores = "refused_in_order_path"
            _terminal_record = _emit_sig(
                result="accepted", execution_eligible=True,
                order_result=_ores,
                reason_code=(
                    _FRICTION_CODE
                    if _ores == "refused_in_order_path"
                    and stats.get("gate_friction", 0) > _pre_order[3]
                    else ("gate.solvency"
                          if _ores == "refused_in_order_path"
                          and stats.get("sizing_declined", 0)
                          > _pre_order[4] else None)),
                floor_result={"floor": float(fl["floor"]),
                              "total": float(score.total),
                              "path": path},
                reason=f"accepted on {path}; order outcome: {_ores}",
                selected_action=(
                    "stage the base order for the owner's "
                    "Confirm tap (Golden Swing ceremony)"
                    if _branch == "confirm" else
                    "layer onto the held position through the "
                    "funding ladder (_try_layer)"
                    if _branch == "layer" else
                    "submit to the shared order path "
                    "(setup-dependent order type inside "
                    "_place_live_order)"),
                rejected_alternatives=[
                    "take the opposite side — rejected: no "
                    "opposite-side candidate from this playbook "
                    "at this bar; the asymmetry score did not "
                    "flip",
                    "abstain and wait for a later bar — "
                    "rejected: the setup qualified on this "
                    "bar's weighed evidence (score, floor and "
                    "hard facts all passed)"])
            # Visible market reading: a live signal/watch verdict
            # announces its rendered assessment (the helper's
            # guards keep refusals, scans and shadow records
            # journal-only).
            self._announce_assessment(_terminal_record)
        # FUNNEL §1.1: no per-signal record was emitted for this
        # evaluation — zero signals, or every signal was skipped
        # by the signal-gap dedupe before reaching a decision.
        # Evaluated-and-empty is a record, not a silence.
        if self._funnel_seq == _funnel_seq0:
            # Burn-down #1: the evaluated-and-empty record
            # carries the structure/HTF snapshot that was
            # present on the fired trigger and states why no
            # scenario qualified — it no longer renders as
            # "not produced" on every section.
            _fb_ts = h1.index[t] if t < len(h1) else None
            _fb_ev = {"route": "h1_trigger",
                      "hierarchy": {
                          "bias_d1": (verdict or {}).get(
                              "bias_d1"),
                          "bias_h4": (verdict or {}).get(
                              "bias_h4"),
                          "trigger": (verdict or {}).get(
                              "trigger"),
                          "entry_clock": (verdict or {}).get(
                              "entry_clock")}}
            _fb_ev.update(self._desk_context_evidence(
                last_bar_ts=_fb_ts))
            _fb_regime = (verdict or {}).get("regime")
            if _fb_regime is not None:
                _fb_inf = dict(_fb_ev.get("inferred") or {})
                _fb_inf.setdefault("regime", _fb_regime)
                _fb_ev["inferred"] = _fb_inf
            self._emit_eval(
                playbook=pb.name, evaluation_clock="h1_trigger",
                intended_tf=_native_tf(pb.name) or "H1",
                actual_input_tf="H1",
                last_bar_ts=_fb_ts,
                outcome="no_setup",
                reason="evaluated on the fired trigger — no "
                       "candidate reached a decision",
                evidence=_fb_ev,
                assessment={"competing_scenarios": [
                    {"scenario": f"a {pb.name} candidate on "
                                 "the fired trigger",
                     "status": "not formed — the playbook "
                               "emitted no signal at this bar, "
                               "or every signal was skipped by "
                               "the signal-gap dedupe before "
                               "reaching a decision"}]})
        return (injected_terminals
                if injected_sig is not None else None)

    # ------------------------------------------------------------- layering
    # Scale-in onto ONE position: a confirming setup in the same
    # direction adds a layer (own entry / volume / SL) while the
    # position is in profit, with total risk capped.  The broker nets
    # the layer deal into the one position (same symbol / direction /
    # magic); the engine tracks the legs in TickManager.layers.
    def _layer_broker_state(self, ticket: int, pos):
        """Fresh broker snapshot for a layer decision (step 5).

        Returns (snapshot, fail_reason): snapshot carries the broker
        parent position (ticket + magic match), broker entry/SL,
        broker volume (netting ticket plus same-alert hedging
        tickets when the tracked alert_id is non-empty), and the
        tracked legs.  ``fail_reason`` is set only when the broker
        read itself failed — unknown broker state fails closed
        before any policy input is built.  A missing parent, a
        direction mismatch, or a volume disagreement is NOT a read
        failure: it is carried in the snapshot so the policy (or
        the pre-gate note) can name it precisely.
        """
        try:
            live = self.executor.positions()
        except Exception as exc:
            return None, f"broker positions unreadable: {exc}"
        try:
            parent = None
            broker_volume = 0.0
            for p in live:
                if int(getattr(p, "ticket", -1)) == int(ticket) \
                        and int(getattr(p, "magic", -1)) \
                        == int(self.cfg.magic):
                    parent = p
            if parent is not None:
                broker_volume = float(parent.volume)
                alert = str(getattr(pos, "alert_id", "") or "")
                if alert:
                    for p in live:
                        if int(getattr(p, "ticket", -1)) == int(ticket):
                            continue
                        if int(getattr(p, "magic", -1)) \
                                != int(self.cfg.magic):
                            continue
                        if str(getattr(p, "alert_id", "") or "") != alert:
                            continue
                        if int(getattr(p, "direction", 0)) \
                                != int(pos.direction):
                            continue
                        broker_volume += float(p.volume)
            return {
                "parent": parent,
                "broker_entry": (float(parent.entry_price)
                                 if parent is not None else None),
                "broker_sl": (float(parent.sl)
                              if parent is not None
                              and parent.sl is not None else None),
                "broker_volume": broker_volume,
            }, None
        except Exception as exc:
            return None, f"broker snapshot malformed: {exc}"

    def _layer_policy_gate(self, *, ticket: int, pos, layers,
                           snapshot, mid: float, lots: float,
                           stop0: float, mode: str, trigger_key: str,
                           trigger_independent: bool, max_layers: int,
                           budget: float, spec, pb, sig,
                           stats: dict) -> bool:
        """Stacking-policy gate between sizing and placement (step 5).

        The release decision (funding / in-profit) is the trigger;
        this gate is the broker-reconciled check on top of it: the
        parent must be broker-confirmed, the broker SL verified,
        the parent not losing, the layer count and the shared
        budget respected — all measured on broker stops, never on
        tracked stops alone.  Returns True to place, False to skip
        (already journalled, with the policy's reason code).
        """
        parent = snapshot.get("parent") if snapshot else None
        broker_sl = snapshot.get("broker_sl") if snapshot else None
        broker_entry = snapshot.get("broker_entry") if snapshot else None
        sl_verified = (
            parent is not None and broker_sl is not None
            and broker_entry is not None and broker_sl > 0
            and ((int(pos.direction) > 0
                  and broker_sl < broker_entry)
                 or (int(pos.direction) < 0
                     and broker_sl > broker_entry)))
        if sl_verified:
            dist = abs(broker_entry - broker_sl)
            unreal_r = (int(pos.direction) * (float(mid) - broker_entry)
                        / dist) if dist > 0 else None
        else:
            unreal_r = None
        if snapshot is not None:
            # A missing parent is the policy's parent_not_open, not
            # a volume disagreement — the volume pre-gate only fires
            # when the broker DOES hold the parent at another size.
            volume_ok = (parent is None or abs(
                float(snapshot["broker_volume"])
                - float(pos.volume)) <= _VOL_EPS)
        else:
            volume_ok = False
        if not volume_ok:
            self.journal.log_note(
                f"mt5: layer skipped {pb.name}: broker volume "
                f"{snapshot.get('broker_volume') if snapshot else None} "
                f"disagrees with tracked {float(pos.volume):.2f} — "
                f"layer refused until reconcile")
            return False
        existing = 0.0
        for ly in layers:
            existing += spec.money_for_move(
                abs(float(ly.entry) - broker_sl), float(ly.volume)) \
                if sl_verified else 0.0
        proposed = spec.money_for_move(
            abs(float(sig.entry_ref) - float(stop0)), float(lots))
        decision = _evaluate_layer_add(
            parent_setup_id=str(getattr(pos, "alert_id", "")
                                or f"ticket-{ticket}"),
            mode=mode, trigger_id=trigger_key,
            trigger_direction=int(sig.direction),
            parent_direction=int(pos.direction),
            parent_open=parent is not None,
            parent_thesis_valid=parent is not None,
            trigger_fresh=True,
            trigger_independent=bool(trigger_independent),
            trigger_already_used=(
                trigger_key in self._layer_trigger_ids.get(ticket, set())),
            parent_unrealized_r=unreal_r,
            active_layers=len(layers),
            max_layers=min(int(max_layers), 3),
            risk_budget_money=(float(budget) if budget is not None
                               else None),
            existing_worst_case_risk_money=existing,
            proposed_worst_case_risk_money=proposed,
            broker_stop_verified=bool(sl_verified))
        if not decision.get("eligible"):
            self.journal.log_note(
                f"mt5: layer refused {pb.name} "
                f"({decision.get('reason_code')}): "
                f"{decision.get('reason')}")
            log_signal(
                self.journal,
                signal_id=f"MT5-{pb.name}-{getattr(sig, 'index', '?')}-layer",
                playbook=pb.name, direction=int(sig.direction),
                score=float(getattr(sig, "score", 0.0) or 0.0),
                decision="gated",
                reason=f"stacking: {decision.get('reason_code')}",
                entry=float(sig.entry_ref), stop=float(stop0),
                trigger_source="stacking_policy")
            return False
        self.journal.log_note(
            f"mt5: layer policy pass {pb.name} "
            f"({decision.get('reason_code')}, child "
            f"{decision.get('child_entry_id')}, combined risk "
            f"{decision.get('combined_worst_case_risk_money')} of "
            f"budget {decision.get('shared_risk_budget_money')})")
        return True

    def _try_layer(self, pb, sig, stats: dict,
                   verdict: dict | None = None,
                   entry_bar: int | None = None,
                   score=None, funding=None, zone_d=None) -> None:
        """A signal while holding: layer or skip, never a second
        position.  REWORK (spec §4): layers are released by the
        funding ladder — a NEW funding event (structure shift /
        displacement / CHoCH-MSB / FVG / sweep / absorption) with
        the stop at breakeven, or the legacy in-profit path — and
        sized from the RiskPool's headroom in risk mode.  The
        legacy fallback path sizes by the same dynamic risk rule
        (owner ruling 2026-10-23: no fixed-lots sizing exists)."""
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
        # Traffic-light scalar trims the layer CEILING (§4.4) —
        # it never vetoes the position itself.  P3 adds the
        # posture cap (coil: 1) and the S11 loss-ladder delta
        # (>= 2 consecutive losses: ceiling -1) to the same min.
        _tier, _smult, ceiling = self._traffic(zone_d)
        max_layers = min(int(self.cfg.layer_max_layers), ceiling)
        post = self._posture
        if post is not None:
            if post.layer_ceiling_cap is not None:
                max_layers = min(max_layers,
                                 int(post.layer_ceiling_cap))
            if post.ladder is not None:
                max_layers = max(
                    1, max_layers + int(post.ladder.layer_cap_delta))
        if len(layers) >= max_layers:
            self.journal.log_note(
                f"mt5: layer skipped {pb.name}: layer ceiling "
                f"{max_layers} reached (config "
                f"{self.cfg.layer_max_layers}, traffic {_tier})")
            return
        # Step 5: broker-reconciled snapshot BEFORE sizing — a layer
        # decision on tracked state alone is how a broker-closed or
        # partially-closed parent keeps stacking.  A failed broker
        # read fails closed: unknown broker state never authorizes
        # new risk.
        snapshot, snapshot_fail = self._layer_broker_state(ticket, pos)
        if snapshot is None:
            self.journal.log_note(
                f"mt5: layer skipped {pb.name}: {snapshot_fail} — "
                f"unknown broker state fails closed")
            return
        trigger_key = f"MT5-{pb.name}-{getattr(sig, 'index', '?')}"
        # Funding events not yet consumed by this position.
        consumed = self._funded_events.setdefault(ticket, set())
        new_events = [e for e in (funding.events if funding else [])
                      if e.key not in consumed]
        # In profit?  R measured on the base layer's risk.
        mid = float((self._last_tick or {}).get("mid")
                    or (verdict or {}).get("entry")
                    or sig.entry_ref)
        unreal_r = int(pos.direction) * (mid - pos.entry_price) \
            / pos.risk_price if pos.risk_price > 0 else 0.0
        # Stop at breakeven or better — the funding ladder's
        # precondition (spec §4.2).
        eff_stop = self.tick_manager.effective_stop(ticket)
        d = int(pos.direction)
        be_secured = eff_stop is not None and (
            (d > 0 and eff_stop >= pos.entry_price - 1e-9)
            or (d < 0 and eff_stop <= pos.entry_price + 1e-9))
        event_funded = bool(new_events) and be_secured
        if not event_funded \
                and unreal_r < self.cfg.layer_min_profit_r:
            self.journal.log_note(
                f"mt5: layer skipped {pb.name}: position "
                f"+{unreal_r:.2f}R < +{self.cfg.layer_min_profit_r:.2f}R "
                "required and no funding event with the stop at "
                "breakeven")
            return
        # Step 5: trigger independence for the policy gate — a
        # funding release is independent by its fresh event; a
        # legacy release is independent when it is not a
        # re-emission of the parent's own signal.
        trigger_independent = (
            bool(new_events) if event_funded
            else trigger_key != str(getattr(pos, "alert_id", "") or ""))
        # Total risk cap: base risk money x max_mult, across all layers.
        try:
            spec = self.executor.symbol_spec(self.cfg.symbol)
        except Exception:
            self.journal.log_note(f"mt5: layer skipped {pb.name}: "
                                  f"no symbol spec")
            return
        # The layer's own stop (buffered, like a base order).
        stop0 = TickManager.planned_stop(
            OrderRequest(symbol=self.cfg.symbol, direction=sig.direction,
                         volume=0.0, stop_loss=sig.invalidation,
                         atr=sig.atr))
        pool = self._pools.get(ticket)
        if self._sizing_mode() == "risk" and pool is not None:
            # ---- RiskPool path (A13): size from headroom --------
            headroom = pool.headroom()
            if headroom <= 0:
                self.journal.log_note(
                    f"mt5: layer skipped {pb.name}: RiskPool "
                    f"headroom exhausted (consumed "
                    f"{pool.consumption()} of {pool.budget})")
                return
            result = size_for_risk(
                headroom, abs(float(sig.entry_ref) - stop0), spec)
            if result.min_override:
                # Layer-headroom refusal (2026-10-24 ruling):
                # layers are OPTIONAL additions — headroom that
                # cannot fit the broker minimum does NOT floor
                # to minimum lots over the ceiling.  This is a
                # distinct reason from a base-order sizing
                # decline (base orders DO take the floor); the
                # base position is unaffected.
                self.journal.log_note(
                    f"mt5: layer DECLINED {pb.name}: headroom "
                    f"{headroom} over stop distance "
                    f"{abs(float(sig.entry_ref) - stop0):.2f} gives "
                    f"raw {result.raw_lots} lots, below the broker "
                    "minimum — layer skipped, ceiling preserved")
                log_signal(
                    self.journal,
                    signal_id=f"MT5-{pb.name}-{sig.index}-layer",
                    playbook=pb.name, direction=int(sig.direction),
                    score=float(sig.score), decision="gated",
                    reason="layer_headroom: headroom cannot fit "
                           "minimum lots",
                    entry=float(sig.entry_ref), stop=float(stop0),
                    score_breakdown=(score.breakdown()
                                     if score is not None else None),
                    trigger_source="funding_layer",
                    pool=pool.state())
                return
            if not result.ok:
                self.journal.log_note(
                    f"mt5: layer DECLINED {pb.name}: sizing "
                    f"{result.verdict} — headroom {headroom} over "
                    f"stop distance "
                    f"{abs(float(sig.entry_ref) - stop0):.2f} gives "
                    f"raw {result.raw_lots} lots")
                log_signal(
                    self.journal,
                    signal_id=f"MT5-{pb.name}-{sig.index}-layer",
                    playbook=pb.name, direction=int(sig.direction),
                    score=float(sig.score), decision="gated",
                    reason=f"sizing_decline: {result.verdict}",
                    entry=float(sig.entry_ref), stop=float(stop0),
                    score_breakdown=(score.breakdown()
                                     if score is not None else None),
                    trigger_source="funding_layer",
                    pool=pool.state())
                return
            self._sizing_detail[
                f"MT5-{pb.name}-{sig.index}-layer"] = {
                    "authorized_risk": float(result.risk_money),
                    "realized_risk": float(result.realized_risk),
                    "estimated_stop_exposure":
                        float(result.estimated_stop_exposure),
                    "drag": float(result.drag),
                    "raw_lots": float(result.raw_lots),
                    "lots": float(result.lots),
                    "min_override": False,
                    "headroom": float(headroom)}
            # Step 5: stacking-policy gate on broker-reconciled
            # state (shared budget = pool budget, fixed at open).
            if not self._layer_policy_gate(
                    ticket=ticket, pos=pos, layers=layers,
                    snapshot=snapshot, mid=mid,
                    lots=float(result.lots), stop0=float(stop0),
                    mode="pyramid" if event_funded else "same_setup",
                    trigger_key=trigger_key,
                    trigger_independent=trigger_independent,
                    max_layers=max_layers, budget=float(pool.budget),
                    spec=spec, pb=pb, sig=sig, stats=stats):
                return
            self._place_live_order(
                pb, sig, stats, verdict=verdict,
                entry_bar=entry_bar, is_layer=True,
                layer_lots=float(result.lots),
                funding_events=new_events,
                score=score, zone_d=zone_d)
            return
        # ---- legacy fallback arithmetic (dynamic sizing) --------
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
        risk_per_lot = spec.money_for_move(
            abs(float(sig.entry_ref) - stop0), 1.0)
        if risk_per_lot <= 0:
            self.journal.log_note(
                f"mt5: layer skipped {pb.name}: zero risk per lot")
            return
        # Size: dynamic risk sizing (the same risk rule as a base
        # order), floored to 0.01 and capped by the remaining risk
        # budget.
        acct = self.feed.account_state()
        # Layer fallback sizing uses the live dynamic rule (the same
        # risk rule as a base order): below-minimum floors to 0.01,
        # then the budget cap below may still cut it under 0.01 and
        # skip (ceiling preserved). The retired legacy sizer must
        # never size a layer order (owner rulings).
        result = size_for_risk(
            float(acct["equity"]) * float(self.cfg.risk_pct),
            abs(float(sig.entry_ref) - stop0), spec)
        if not result.ok:
            self.journal.log_note(
                f"mt5: layer sizing refused {pb.name}: "
                f"{result.verdict}")
            return
        lots = float(result.lots)
        lots = min(lots, int(budget / risk_per_lot / 0.01) * 0.01)
        if lots < 0.01:
            self.journal.log_note(
                f"mt5: layer skipped {pb.name}: risk cap leaves "
                f"{lots:.2f} lots (< 0.01)")
            return
        # Step 5: stacking-policy gate (legacy budget = mult x
        # ORIGINAL base risk, so a breakeven stop roll frees
        # budget instead of collapsing it to zero).
        _base_leg = layers[0]
        _base_risk = spec.money_for_move(
            abs(float(_base_leg.entry) - float(_base_leg.sl)),
            float(_base_leg.volume))
        if not self._layer_policy_gate(
                ticket=ticket, pos=pos, layers=layers,
                snapshot=snapshot, mid=mid, lots=float(lots),
                stop0=float(stop0),
                mode="pyramid" if event_funded else "same_setup",
                trigger_key=trigger_key,
                trigger_independent=trigger_independent,
                max_layers=max_layers,
                budget=float(self.cfg.layer_max_total_risk_mult
                             * _base_risk),
                spec=spec, pb=pb, sig=sig, stats=stats):
            return
        self._place_live_order(pb, sig, stats, verdict=verdict,
                               entry_bar=entry_bar, is_layer=True,
                               layer_lots=float(lots),
                               funding_events=new_events,
                               score=score, zone_d=zone_d)

    def _record_layer_fill(self, pb_name: str, sig, *, fill: float,
                           lots: float, stop0: float, stats: dict,
                           funding_events: list | None = None) -> None:
        """A layer order filled: append the leg, move the broker SL
        to the base breakeven (only ever TIGHTENS), journal + alert.
        No new open record — the layer is a leg of the one position,
        not a second trade (the position's close record settles the
        whole stack).

        REWORK (spec §4.2): the layer is booked into the RiskPool;
        the funding events that released it are marked consumed,
        and the roll target is the tightest of (base breakeven,
        the events' invalidation levels) that still tightens the
        current stop and stays on the safe side of the mid.
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
        # Step 5: the trigger that produced this layer is consumed
        # (stacking policy trigger_duplicate input).  Marked on
        # fill, never on placement attempt.
        self._layer_trigger_ids.setdefault(ticket, set()).add(
            f"MT5-{pb_name}-{getattr(sig, 'index', '?')}")
        layers = self.tick_manager.layers(ticket)
        # RiskPool booking + funding-event consumption (A13/§4.2).
        if not self._pool_book_layer(
                ticket, lots=float(lots), entry=float(fill),
                stop=float(stop0), direction=int(pos.direction),
                alert_id=layer.alert_id):
            self.journal.log_note(
                f"mt5 pool: layer {layer.alert_id} on ticket "
                f"{ticket} exceeds the pool ceiling — booked by "
                "the broker already; consumption is over budget "
                "and no further layer can fund until stops roll")
        if funding_events:
            consumed = self._funded_events.setdefault(ticket, set())
            for e in funding_events:
                consumed.add(e.key)
        # Move the broker SL to the roll target (base breakeven,
        # or a funding event's tighter invalidation) — only if it
        # tightens the current broker SL, never looser.
        d = int(pos.direction)
        eff = self.tick_manager.effective_stop(ticket)
        level = float(pos.entry_price)
        roll_note = "breakeven"
        mid = float((self._last_tick or {}).get("mid") or fill)
        for e in (funding_events or []):
            inv = getattr(e, "invalidation", None)
            if inv is None:
                continue
            inv = float(inv)
            if d > 0 and level < inv < mid:
                level, roll_note = inv, f"funding {e.kind}"
            elif d < 0 and mid < inv < level:
                level, roll_note = inv, f"funding {e.kind}"
        moved = False
        if eff is not None and self.tick_manager.is_fixed_bracket(
                int(ticket)) and ((d == 1 and level > eff)
                                  or (d == -1 and level < eff)):
            # Step 13: a layer fill does NOT roll a fixed-bracket
            # position's stop — the placement bracket stands.
            self.journal.log_note(
                f"mt5 layer: ticket {ticket} stop roll to "
                f"{roll_note} SKIPPED — fixed-bracket playbook "
                f"(step 13)")
        elif eff is not None and ((d == 1 and level > eff)
                                  or (d == -1 and level < eff)):
            res = self.executor.modify_sl(ticket, level)
            if res.ok:
                self.tick_manager.apply(
                    ticket, TickAction("modify_sl", level=level,
                                       reason="breakeven"))
                moved = True
                # The pool's base layer rolls with the broker stop
                # (refund net of the friction reserve, §4/A13).
                self._pool_sync_stop(ticket, pos.alert_id, level)
            else:
                self.journal.log_note(
                    f"mt5 layer: broker SL move to breakeven FAILED "
                    f"({res.message}) — retry next tick")
        self.journal.log_note(
            f"mt5 layer: ticket {ticket} {pb_name} "
            f"{'+' if sig.direction > 0 else '-'}1 {lots} lots @ "
            f"{fill:.2f} SL {stop0:.2f} "
            f"({len(layers)} layers)"
            + (f" broker SL -> {roll_note} {level:.2f}" if moved else ""))
        self._send_alert(
            f"MT5 LAYER {pb_name} {'+' if sig.direction > 0 else '-'}1 "
            f"{lots} lots @ {fill:.2f} SL {stop0:.2f} "
            f"({len(layers)} layers, ticket {ticket})"
            + (f" — broker SL -> {roll_note} {level:.2f}"
               if moved else ""))

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
        # PNL-PARTIAL: the trimmed layer's money is measured
        # against the TOP layer's entry (looked up before apply
        # pops it), at the last tick's mid — the market price
        # this broker close fills near.  (action.level is the
        # layer's STOP, never the execution price.)  With no tick
        # seen yet there is no honest price: the leg is left
        # unbooked and noted, never booked at a guessed price.
        pos = self.tick_manager.position(ticket)
        layers = self.tick_manager.layers(ticket)
        top = layers[-1] if len(layers) > 1 else None
        last = self._last_tick or {}
        if pos is not None and last.get("mid"):
            self._book_partial_leg(
                ticket, pos, volume=action.volume,
                price=float(last["mid"]),
                entry=top.entry if top is not None else pos.entry_price,
                stop=top.sl if top is not None else pos.stop0)
        else:
            self.journal.log_note(
                f"mt5 trim: ticket {ticket} top layer closed with "
                f"no tick price seen — leg P&L left unbooked")
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
        if field == "max_layers":
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
        """Owner /resume: clear the Telegram pause AND any regime
        pause.  A regime pause is never cleared silently: every
        transition out of it runs the PendingOrderAudit first
        (stale pendings never auto-resume).  A COMPLETE pause
        (position state untrusted) re-verifies position state
        first — if still untrusted it stays paused — then audits
        and drops to the 2-bar observation window.
        """
        msgs: list[str] = []
        if self.cfg.paused:
            self.cfg.paused = False
            self._persist_live_config()
            self.journal.log_note("mt5: engine RESUMED via Telegram")
            msgs.append("engine RESUMED — new signals live again")
        if self._regime_pause == "paused_complete":
            trustworthy, why = self._position_state_trustworthy()
            if not trustworthy:
                return ("regime pause NOT lifted — position state "
                        f"still untrusted ({why}); check + /resume "
                        "again")
            self._audit_pending_orders()
            self._regime_pause = "observation"
            self._regime_pause_reason = (
                "owner /resume after complete pause")
            self._regime_observation_remaining = 2
            self._regime_stable_bars = 0
            self._regime_stable_signature = None
            self.journal.log_note(
                "mt5: regime pause lifted by owner /resume — "
                "observation window 2 H1 bars")
            msgs.append("regime pause -> observation (2 H1 bars, "
                        "pending audit done)")
        elif self._regime_pause in ("paused_new", "observation"):
            self._audit_pending_orders()
            self._regime_pause = None
            self._regime_pause_reason = ""
            self._regime_stable_bars = 0
            self._regime_stable_signature = None
            self.journal.log_note(
                "mt5: regime pause cleared by owner /resume "
                "(pending audit done)")
            msgs.append("regime pause cleared by owner — pending "
                        "audit done")
        if not msgs:
            return "engine already running"
        return "; ".join(msgs)

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
                        "alert_id": state.alert_id,
                        "ladder_remaining": len(state.ladder_levels)})
        return out

    # ------------------------------------------------- confirm handshake
    def request_confirm(self, action: str, arg=None) -> str:
        """Stage a mutating action; returns the Confirm token."""
        token = secrets.token_hex(4)
        # created_at (monotonic) lets the order-confirm sweep expire
        # place_order tokens; control tokens never expire.
        self._confirm_tokens[token] = {
            "action": action, "arg": arg,
            "created_at": time.monotonic()}
        return token

    def confirm_pending(self, token: str,
                        stats: dict | None = None) -> tuple[bool, str]:
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
            if action == "place_order":
                return self._execute_order_confirm(
                    arg or "", stats if stats is not None else {})
            return False, f"unknown staged action {action!r} — not acted"
        except Exception as exc:
            return False, (f"{action} FAILED: {type(exc).__name__}: {exc}")

    def cancel_confirm(self, token: str) -> str:
        staged = self._confirm_tokens.pop(token, None)
        if staged is None:
            return "nothing to cancel (confirmation already expired)"
        if staged.get("action") == "place_order":
            # A rejected Golden Swing order dies here: drop the
            # payload so it can never fire later.  Nothing was
            # placed — the order waited for the tap.
            self._order_confirms.pop(staged.get("arg") or "", None)
            self.journal.log_note(
                "mt5: Golden Swing order REJECTED — no order placed")
            return "rejected — no order placed"
        return "cancelled — nothing changed"

    def _apply_alignment_multiplier(self, lots: float, spec,
                                    pb_name: str) -> float:
        """MacroTrend size multiplier on a base-order size.

        Golden Swing -> 1.5x, D1/W1 conflict -> 0.5x, else 1.0x
        (no-op).  Re-steps DOWN to the broker volume step (never
        rounds up), floors at the broker minimum, caps at the
        maximum.  No-op when sizing is disabled or the multiplier
        is 1.0 — the common path is untouched.
        """
    def _apply_size_multiplier(self, lots: float, spec,
                               mult: float, note: str) -> float:
        """Multiply a base-order size, re-stepping DOWN to the broker
        volume step (never rounds up), floored at the broker minimum,
        capped at the maximum.  No-op at 1.0x."""
        if mult == 1.0:
            return lots
        step = spec.volume_step if spec.volume_step > 0 \
            else spec.volume_min
        scaled = math.floor(lots * mult / step + 1e-9) * step
        scaled = round(min(max(scaled, spec.volume_min),
                           spec.volume_max), 10)
        self.journal.log_note(
            f"mt5: sizing {note}: {lots:.2f} lots x {mult:g} -> "
            f"{scaled:.2f} lots")
        return scaled

    def _apply_alignment_multiplier(self, lots: float, spec,
                                    pb_name: str) -> float:
        """MacroTrend size multiplier on a base-order size.

        Golden Swing -> 1.5x, D1/W1 conflict -> 0.5x, else 1.0x
        (no-op).  No-op when sizing is disabled or the multiplier
        is 1.0 — the common path is untouched.
        """
        if not self.cfg.golden_swing_sizing_enabled:
            return lots
        mult = float(self._alignment.size_multiplier)
        if mult == 1.0:
            return lots
        return self._apply_size_multiplier(
            lots, spec, mult,
            f"alignment {pb_name} ({self._alignment.alignment_label})")

    # ------------------------------------------- Golden Swing confirm
    # A Golden Swing base order WAITS for the owner's Confirm tap.
    # Unlike the owner's bot (order fired first, tap changed
    # nothing), nothing is placed until Confirm, timeout, or
    # Reject.  The existing confirm handshake carries it:
    # request_confirm token + mt5_confirm_keyboard + the same
    # callback parsing; only the staged action is new.
    def _stage_order_confirm(self, pb, sig, stats: dict,
                             verdict: dict | None,
                             entry_bar: int | None) -> None:
        """Stage a Golden Swing base order: Telegram Confirm/Reject
        buttons, order held.  Layers never come here (they route
        through _try_layer at fire time)."""
        key = secrets.token_hex(4)
        self._order_confirms[key] = {
            "pb": pb, "sig": sig, "verdict": verdict,
            "entry_bar": entry_bar}
        token = self.request_confirm("place_order", key)
        timeout = self.cfg.golden_swing_confirm_timeout_sec
        action = self.cfg.golden_swing_confirm_timeout_action
        self.journal.log_note(
            f"mt5: Golden Swing {pb.name} "
            f"{'+' if sig.direction > 0 else '-'}1 staged — awaiting "
            f"Confirm tap (timeout {timeout:.0f}s, then {action})")
        self._send_alert(
            f"🎯 GOLDEN SWING {pb.name} "
            f"{'+' if sig.direction > 0 else '-'}1\n"
            f"entry~{sig.entry_ref:.1f} stop~{sig.invalidation:.1f} "
            f"score={sig.score:.0f} size x{self._alignment.size_multiplier:g}\n"
            f"Tap Confirm to place, Reject to cancel. "
            f"Timeout {timeout:.0f}s → {action}."
            + (f"\n\n{build_reasoning(pb.name, sig, verdict, alignment_label=getattr(self._alignment, 'alignment_label', None))}"),
            reply_markup=mt5_confirm_keyboard(token))

    def _execute_order_confirm(self, key: str,
                               stats: dict) -> tuple[bool, str]:
        """Place a staged Golden Swing order (Confirm tap or
        timeout-fire).  Fire-time revalidation: paused engine
        refuses; a gone/flipped setup refuses; a held position
        routes to _try_layer (one-position rule) instead of a
        second position.  Placement itself re-checks price vs
        entry (select_order_type) so a stale setup cannot fill
        at a dead level."""
        payload = self._order_confirms.pop(key, None)
        if payload is None:
            return False, ("confirmation expired — nothing acted. "
                           "Send the command again.")
        pb, sig = payload["pb"], payload["sig"]
        if self.cfg.paused:
            self.journal.log_note(
                f"mt5: Golden Swing {pb.name} confirm refused — "
                "engine paused, no order placed")
            return False, "engine paused — order not placed"
        setup = self._live_setup
        if setup is None or int(setup["direction"]) != int(sig.direction):
            self.journal.log_note(
                f"mt5: Golden Swing {pb.name} confirm refused — "
                "setup gone/flipped, no order placed")
            return False, "setup gone — order not placed"
        if self.tick_manager.tickets():
            held = self.tick_manager.tickets()
            if len(held) != 1:
                return False, "position state unclear — order not placed"
            pos = self.tick_manager.position(held[0])
            if pos is not None and int(pos.direction) != int(sig.direction):
                self.journal.log_note(
                    f"mt5: Golden Swing {pb.name} confirm refused — "
                    "opposite position held, no order placed")
                return False, "opposite position held — order not placed"
            self._try_layer(pb, sig, stats,
                            verdict=payload.get("verdict"),
                            entry_bar=payload.get("entry_bar"))
            return True, "confirmed — evaluated as layer; see journal"
        self._place_live_order(pb, sig, stats,
                               verdict=payload.get("verdict"),
                               entry_bar=payload.get("entry_bar"))
        return True, "confirmed — order placed"

    def _sweep_order_confirms(self, stats: dict) -> None:
        """Expire unanswered Golden Swing confirms: timeout -> fire
        (place, revalidated) or cancel (drop), per config.  Runs
        every tick from _on_tick — the timeout is wall-clock, not
        bar-clock, so it must not wait for a bar close."""
        timeout = float(self.cfg.golden_swing_confirm_timeout_sec)
        action = self.cfg.golden_swing_confirm_timeout_action
        now = time.monotonic()
        for token, staged in list(self._confirm_tokens.items()):
            if staged.get("action") != "place_order":
                continue
            created = staged.get("created_at", now)
            if now - created < timeout:
                continue
            self._confirm_tokens.pop(token, None)
            key = staged.get("arg") or ""
            if action == "cancel":
                self._order_confirms.pop(key, None)
                self.journal.log_note(
                    "mt5: Golden Swing order confirm timed out — "
                    "cancelled per config, no order placed")
                self._send_alert(
                    "Golden Swing order timed out — cancelled "
                    "(no Confirm tap).")
            else:  # "fire" (also the fallback for a bad config value)
                ok, text = self._execute_order_confirm(key, stats)
                self.journal.log_note(
                    "mt5: Golden Swing order confirm timed out — "
                    f"auto-fire: {text}")

    # ------------------------------------------------- safety wiring
    # RiskGuards, margin + shield, market-closed: the live path
    # enforces all of them at order time.  A refusal is journalled
    # + counted by leg (see _record_blocked) — never silent.
    def _market_open(self) -> tuple[bool, str]:
        """Broker accepting new orders?  Feeds without market_open()
        are legacy — treated as open (the real MT5Feed implements
        it, and a missing method must not change order behaviour
        for a feed that never claimed the check).

        Step 3: broker-open AGREEMENT — the broker flag alone does
        not open the market.  Broker-open while the engine's weekly
        calendar says closed is a stale flag inside the weekly close
        and refuses (fail closed); broker-closed while the calendar
        is open (a real holiday the weekly calendar does not know)
        stays closed via the broker flag.  If the calendar itself
        cannot answer, the broker flag (measured from the terminal)
        stands.  Session detection is a separate question and never
        enters this guard.
        """
        fn = getattr(self.feed, "market_open", None)
        if fn is None:
            return True, ""
        try:
            ok, reason = fn()
        except Exception as exc:
            return False, f"market_open check failed: {exc}"
        if ok:
            try:
                if not _market_hours.xauusd_market_open(
                        datetime.now(timezone.utc)):
                    return False, ("calendar disagreement: broker "
                                   "says open, calendar says weekly "
                                   "close — refusing (stale broker flag)")
            except Exception:
                pass
        return ok, reason

    def _open_risk_state(self, spec) -> tuple[float, int]:
        """(open risk money, unstopped position count) across the
        held book — the RiskGuards inputs, computed once and
        shared by the order path and the decision-stage solvency
        pre-check so both read the SAME book state."""
        open_risk = 0.0
        unstopped = 0
        for ticket in self.tick_manager.tickets():
            pos = self.tick_manager.position(ticket)
            if pos is None:
                continue
            # risk_price <= 0 = no measurable stop (e.g. adopted
            # without an SL): exposure unknowable, counts unstopped.
            if pos.risk_price <= 0:
                unstopped += 1
            else:
                open_risk += spec.money_for_move(
                    abs(pos.entry_price - pos.risk_price), pos.volume)
        return open_risk, unstopped

    def _check_risk_guards(self, pb, sig, lots: float, stop0: float,
                           acct: dict) -> bool:
        """RiskGuards on the live path: may this order open?"""
        try:
            spec = self.executor.symbol_spec(self.cfg.symbol)
        except Exception as exc:
            self.journal.log_note(
                f"mt5: order refused {pb.name}: risk guard check "
                f"failed, no symbol spec ({exc})")
            return False
        open_risk, unstopped = self._open_risk_state(spec)
        new_risk = spec.money_for_move(
            abs(float(sig.entry_ref) - stop0), lots)
        decision = self.guards.check_new_trade(
            balance=float(acct.get("balance", 0.0)),
            open_risk_money=open_risk, new_risk_money=new_risk,
            open_count=len(self.tick_manager.tickets()),
            unstopped_positions=unstopped)
        if decision.allowed:
            return True
        self._record_blocked(f"risk_guard: {decision.reason}", pb.name)
        self.journal.log_note(
            f"mt5: order refused {pb.name}: risk guard — "
            f"{decision.reason}")
        return False

    def _decision_solvency(self, pb, sig, score=None
                           ) -> tuple[bool, str]:
        """Phase 2 (P1-1): decision-stage solvency pre-check.

        The EC2 burst showed the seam: 23 candidates were
        analytically "accepted" at decision and only refused
        inside the order path by RiskGuards (daily loss room
        0.36 vs trade risks 4.45–13.84) — the decision stage
        never asked the question the order path was guaranteed
        to answer NO.  This pre-check asks it, with the SAME
        accounting: the planned stop (TickManager.planned_stop,
        the order path's own helper), the intended size under
        the EFFECTIVE sizing mode, the live book's open risk,
        and the real RiskGuards.check_new_trade — so a candidate
        the room would refuse is refused AT DECISION with the
        guard's exact reason (gate.solvency), and the order path
        remains the authoritative backstop (it re-checks
        everything, including margin, at placement time).

        Sizing truth (owner ruling 2026-10-23): dynamic risk
        sizing is the sole path — the intended size is derived
        from the risk budget, so the candidate's worst-case new
        risk IS the budget itself (equity x risk_pct x score
        band, pre-multiplier): if the BUDGET cannot fit, no
        size the order path could compute fits either.  No
        silent fallback exists: invalid risk sizing declines
        inside the order path (sizing_declined) and is never
        rescued by a fixed lot count (the mode no longer
        exists).

        Fail closed: if the check itself cannot run (no spec,
        no account state), the candidate is refused at decision
        with the reason — an unanswerable solvency question is
        not a pass.  Returns (ok, reason).
        """
        try:
            spec = self.executor.symbol_spec(self.cfg.symbol)
            acct = self.feed.account_state()
            stop0 = TickManager.planned_stop(
                OrderRequest(symbol=self.cfg.symbol,
                             direction=int(sig.direction),
                             volume=0.0,
                             stop_loss=float(sig.invalidation),
                             atr=float(sig.atr or 0.0)))
            total = float(score.total) \
                if score is not None else 75.0
            band = 1.0 if total >= 75.0 else (
                0.75 if total >= 60.0 else 0.5)
            new_risk = float(acct.get("equity", 0.0)) \
                * float(self.cfg.risk_pct) * band
            sizing_desc = (f"risk budget {new_risk:.2f} "
                           f"(band {band})")
            open_risk, unstopped = self._open_risk_state(spec)
            decision = self.guards.check_new_trade(
                balance=float(acct.get("balance", 0.0)),
                open_risk_money=open_risk,
                new_risk_money=new_risk,
                open_count=len(self.tick_manager.tickets()),
                unstopped_positions=unstopped)
            if decision.allowed:
                return True, ""
            return False, (
                f"{decision.reason} [decision-stage solvency: "
                f"{sizing_desc}, planned stop {stop0:.2f}, "
                f"new risk {new_risk:.2f}, open risk "
                f"{open_risk:.2f}, balance "
                f"{float(acct.get('balance', 0.0)):.2f}]")
        except Exception as exc:
            return False, (
                f"decision-stage solvency check could not run "
                f"({type(exc).__name__}: {exc}) — fail closed, "
                "no order")

    def _check_margin(self, pb, sig, lots: float, is_layer: bool,
                      acct: dict) -> bool:
        """Margin + swing margin shield before any order placement.

        Margin: the broker-computed requirement (order_calc_margin)
        plus the configured buffer must fit inside free margin —
        the owner's-bot formula (lots × contract × price / leverage,
        +5%) with the terminal's own number instead of a hand-rolled
        one.  Shield: a Golden Swing base order or any layer must
        leave the POST-order free margin at or above the shield
        fraction of equity (owner's bot: 40%), so a layer can never
        push the account into margin-call territory.  A margin
        number the terminal cannot compute refuses the order —
        never a guess.
        """
        free = float(acct.get("free_margin", 0.0))
        equity = float(acct.get("equity", 0.0))
        try:
            margin_req = self.feed.order_margin(
                volume=float(lots), price=float(sig.entry_ref),
                direction=int(sig.direction))
        except Exception as exc:
            self.journal.log_note(
                f"mt5: order refused {pb.name}: margin check failed "
                f"({exc}) — no order without a margin number")
            return False
        if margin_req is None:
            self.journal.log_note(
                f"mt5: order refused {pb.name}: terminal could not "
                "compute margin — no order without a margin number")
            return False
        buffered = margin_req * (1.0 + self.cfg.margin_buffer_pct / 100.0)
        if buffered > free:
            self._record_blocked(
                f"margin: required {buffered:.2f} (buffered) > free "
                f"{free:.2f}", pb.name)
            self.journal.log_note(
                f"mt5: order refused {pb.name}: margin — required "
                f"{buffered:.2f} (buffered) > free margin {free:.2f}")
            return False
        if is_layer or self._alignment.golden_swing:
            post_free = free - margin_req
            shield = self.cfg.margin_shield_pct / 100.0
            if equity <= 0 or post_free / equity < shield:
                self._record_blocked(
                    f"margin_shield: post-order free {post_free:.2f} "
                    f"/ equity {equity:.2f} < "
                    f"{self.cfg.margin_shield_pct:g}%", pb.name)
                self.journal.log_note(
                    f"mt5: order refused {pb.name}: margin shield — "
                    f"post-order free {post_free:.2f} / equity "
                    f"{equity:.2f} < {self.cfg.margin_shield_pct:g}%")
                return False
        return True

    # -------------------------------------------------------------- orders
    def _entry_friction_veto(self, sig, planned_stop: float,
                             *, point: float | None = None
                             ) -> str | None:
        """RULING 6 friction gate reading for one proposed order
        (see execution/friction_gate.py).  Returns the refusal
        detail — measured spread, stop distance, ratio, timestamp,
        symbol/feed — when the entry is refused, else None.

        Spread sources, in order: the live tick (ask-bid when the
        quote carries both sides, else the tick's ``spread``
        field — the feed contract: a price distance, feed/mt5.py
        builds it as ask - bid); when no tick exists yet (warm
        start), the newest bar's ``spread`` column — MT5 bars
        carry spread in POINTS, normalized through the
        instrument's point size (the XAUUSDm spec: 0.01), never
        compared raw against a price distance.  When NO source
        can measure the spread, the assessment fails closed.
        Never raises.
        """
        tick = self._last_tick or {}
        try:
            spread_price = None
            bid, ask = tick.get("bid"), tick.get("ask")
            try:
                if bid is not None and ask is not None \
                        and float(ask) > float(bid) > 0:
                    spread_price = float(ask) - float(bid)
            except (TypeError, ValueError):
                spread_price = None
            if spread_price is None:
                raw = tick.get("spread")
                try:
                    spread_price = float(raw) \
                        if raw is not None else None
                except (TypeError, ValueError):
                    spread_price = None
            spread_points = None
            if spread_price is None:
                frames = (self._stack.frames
                          if self._stack is not None else {}) or {}
                for _tf in ("H1", "M5", "M15", "H4", "D1"):
                    _fr = frames.get(_tf)
                    _cols = getattr(_fr, "columns", None)
                    if _fr is not None and _cols is not None \
                            and len(_fr) and "spread" in _cols:
                        try:
                            spread_points = float(
                                _fr["spread"].iloc[-1])
                        except (TypeError, ValueError):
                            spread_points = None
                        break
            verdict = _assess_friction(
                entry=float(sig.entry_ref),
                stop=float(planned_stop),
                spread_price=spread_price,
                spread_points=spread_points, point=point)
        except Exception as exc:
            # The gate NEVER raises into the order path: a broken
            # reading is an unmeasurable spread — fail closed.
            verdict = None
            _veto = (f"{_FRICTION_CODE}: friction gate reading "
                     f"failed ({type(exc).__name__}: {exc}) — "
                     "fail closed, no order "
                     f"[ts={tick.get('time')} "
                     f"symbol={self.cfg.symbol} "
                     f"feed={type(self.feed).__name__}]")
            return _veto
        if not verdict.refused:
            return None
        return (f"{_FRICTION_CODE}: {verdict.detail} "
                f"[spread={verdict.spread_price} "
                f"stop_distance={verdict.stop_distance} "
                f"ratio={verdict.ratio} ts={tick.get('time')} "
                f"symbol={self.cfg.symbol} "
                f"feed={type(self.feed).__name__}]")

    def _place_live_order(self, pb, sig, stats: dict,
                          verdict: dict | None = None,
                          entry_bar: int | None = None,
                          is_layer: bool = False,
                          layer_lots: float | None = None,
                          funding_events: list | None = None,
                          score=None, zone_d: dict | None = None,
                          funding=None,
                          signal_ctx: dict | None = None,
                          intent: dict | None = None
                          ) -> None:
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
        # Phase 2 (P0-1) ELIGIBILITY BACKSTOP: a candidate
        # stamped execution-ineligible at creation (warmup /
        # catch-up / replay provenance, expired validity window,
        # unestablishable age — see candidate_lifecycle) is
        # refused HERE, at the shared order boundary, whatever
        # caller produced it.  The native routes check before
        # dispatch; this check makes the refusal structural, so
        # no future caller can route around the lifecycle rules.
        # Signals that were never stamped (the trigger path's own
        # candidates, management layers on held positions) carry
        # no stamp and are unaffected.
        if getattr(sig, "execution_eligible", True) is False:
            _code = str(getattr(sig, "ineligible_reason_code", None)
                        or "path.warmup_ineligible")
            stats[f"ineligible::{_code}"] = stats.get(
                f"ineligible::{_code}", 0) + 1
            self._record_blocked(
                f"candidate_ineligible: {_code} — provenance / "
                "validity forbids execution (candidate_lifecycle)",
                pb.name)
            self.journal.log_note(
                f"mt5: order refused {pb.name}: candidate is "
                f"execution-ineligible ({_code}) — refused at "
                "the order boundary (Phase 2 backstop)")
            return
        # Market-closed guard (order placement): never place while
        # the broker is not accepting new orders — covers
        # confirm-fired orders and layers, not just the signal path.
        open_ok, open_reason = self._market_open()
        if not open_ok:
            self.journal.log_note(
                f"mt5: order refused {pb.name}: market closed "
                f"({open_reason})")
            return
        # Step 15 duplicate guard (rulings 2026-10-30 §2): the SAME
        # parent thesis expressed by two playbooks must not create
        # duplicate orders merely because they prefer different
        # execution methods.  A thesis twin = a working PENDING in
        # the same direction at the same entry level (within the
        # selector's at-level tolerance, 0.25 x ATR): one playbook
        # rests a limit there, another reaches for a market fill at
        # the same level — the second order is refused.  Different
        # levels are different setups: the desk legitimately works
        # an inversion pending and a judas/turtle market entry on
        # the same frame (pinned by the telegram-control suite),
        # and the one-PENDING rule in _place_pending_order still
        # caps resting orders at one.
        if not is_layer and self._pending_orders:
            _tol = 0.25 * float(getattr(sig, "atr", 0.0) or 0.0)
            _twin = None
            for _w in self._pending_orders.values():
                if int(_w.direction) == int(sig.direction) and \
                        abs(float(_w.level)
                            - float(sig.entry_ref)) <= _tol:
                    _twin = _w
                    break
            if _twin is not None:
                stats["duplicate_working_order"] = stats.get(
                    "duplicate_working_order", 0) + 1
                self._record_blocked(
                    f"duplicate working order: {_twin.playbook} "
                    f"{_twin.kind} pending (ticket {_twin.ticket}) "
                    f"already working the same thesis — direction "
                    f"{int(sig.direction):+d} at level "
                    f"{float(_twin.level):.2f} vs candidate entry "
                    f"{float(sig.entry_ref):.2f}; {pb.name} refused "
                    "(execution method is not a second order)",
                    pb.name)
                self.journal.log_note(
                    f"mt5: order refused {pb.name}: already "
                    f"holding/pending — duplicate working order "
                    f"({_twin.playbook} {_twin.kind} pending ticket "
                    f"{_twin.ticket} at the same level "
                    f"{float(_twin.level):.2f})")
                return
        # News pause (base orders only): no NEW entries inside a
        # high-impact window.  Layers are management of a held
        # position, never paused.  Confirm-fired orders re-check
        # here (the wait may have crossed into a window).
        news_mult = 1.0
        if not is_layer and self.cfg.news_pause_enabled:
            state = self.news.execution_state()
            if state.paused:
                stats["gate_news"] = stats.get("gate_news", 0) + 1
                self.journal.log_note(
                    f"mt5: order refused {pb.name}: news pause "
                    f"({state.regime} {state.event_title})")
                return
            news_mult = state.size_mult
            # S5 post-print REACTION state (P3): for ~30 min after
            # an EXTREME print the desk trades the REACTION, not
            # the clock — an entry needs the first post-print
            # structure (displacement / structure shift in the
            # funding report).  Once structure prints, the size
            # blend keys off IT (full mult), not the cooldown
            # clock.  A fade AT a graded level off the spike's
            # termination is the sanctioned exception shape — it
            # still needs its displacement leg, which the fade's
            # own funding carries when real.
            _reaction = self.news.reaction_state(
                self._tick_datetime())
            if _reaction is not None:
                _has_structure = funding is not None and any(
                    e.kind in ("structure_shift", "displacement")
                    for e in funding.events)
                if not _has_structure:
                    stats["gate_reaction"] = stats.get(
                        "gate_reaction", 0) + 1
                    self.journal.log_note(
                        f"mt5: order refused {pb.name}: post-print "
                        f"reaction state ({_reaction.title}) — no "
                        "first displacement / structure print yet; "
                        "the desk trades the reaction, it does "
                        "not front-run it")
                    return
                news_mult = 1.0
                self.journal.log_note(
                    f"mt5: post-print reaction entry {pb.name}: "
                    f"structure printed after {_reaction.title} "
                    "— size blend keys off the structure, not "
                    "the clock (S5)")
        # Entry gates (base orders only): caps / cooldown /
        # fingerprint dedup.  Layers are management of a held
        # position, never gated.  The fingerprint is recorded only
        # on a successful placement (below).
        fingerprint = ""
        if not is_layer and self.cfg.entry_gates_enabled:
            fingerprint = signal_fingerprint(
                pb.name, int(sig.direction), float(sig.entry_ref),
                float(sig.invalidation))
            ok, gate_reason = self.entry_gate.check_new_entry(
                time.time(), fingerprint)
            if not ok:
                stats[f"gate_{gate_reason}"] = stats.get(
                    f"gate_{gate_reason}", 0) + 1
                self.journal.log_note(
                    f"mt5: order refused {pb.name}: entry gate "
                    f"{gate_reason}")
                return
        # Friday close (base orders only): no NEW swing entries late
        # Friday.  Layers are management, never blocked.  The clock
        # is the tick clock, falling back to the wall clock.
        now_utc = _tick_now_utc(self._last_tick) \
            or datetime.now(timezone.utc)
        if not is_layer and is_friday_close(
                now_utc, self.cfg.friday_close_hour):
            stats["gate_friday"] = stats.get("gate_friday", 0) + 1
            self.journal.log_note(
                f"mt5: order refused {pb.name}: friday close "
                f"(no new entries Fri >= "
                f"{self.cfg.friday_close_hour}:00 UTC)")
            return
        executor = self.executor
        spec = executor.symbol_spec(self.cfg.symbol)
        acct = self.feed.account_state()
        stop0 = TickManager.planned_stop(
            OrderRequest(symbol=self.cfg.symbol, direction=sig.direction,
                         volume=0.0, stop_loss=sig.invalidation,
                         atr=sig.atr))
        # RULING 6 friction gate (owner 2026-10-23): hard entry
        # refusal when spread / stop_distance > 0.10, measured on
        # THIS proposed order (entry level vs the buffered broker
        # stop stop0).  Sits at the risk/execution boundary,
        # OUTSIDE any judgment/scoring — no score, floor, regime
        # or bias reading reaches past it or is reached by it.
        # The refusal does NOT invalidate or consume the setup:
        # the armed plan stays armed, and the native routes hold
        # their evaluation cursor so the candidate is re-evaluated
        # when friction normalizes.  Separate mechanism from the
        # hard-lines cost gate (setup quality, inside the engine)
        # and from RiskPool's friction reserve (sizing) — the
        # journaled code is risk.friction (taxonomy category 8).
        _fric_detail = self._entry_friction_veto(
            sig, stop0, point=spec.point)
        if _fric_detail is not None:
            stats["gate_friction"] = stats.get(
                "gate_friction", 0) + 1
            self._record_blocked(
                f"risk_friction: {_fric_detail}", pb.name)
            self.journal.log_note(
                f"mt5: order refused {pb.name}: {_fric_detail}")
            return
        if is_layer:
            # Layer size was pre-capped by _try_layer (total risk cap).
            lots = float(layer_lots)
        elif self._sizing_mode() == "risk":
            # ---- A13 risk sizing (Decimal, quantized once) ------
            # Risk money = equity x k x C x S, where k is the
            # sizing coefficient (cfg.risk_pct — a proposed
            # coefficient, NOT a refusal gate), C is the
            # confidence multiplier, and S is the product of the
            # kept non-gate scalars (traffic tier, verdict
            # size_mult, news, posture, alignment, session),
            # each applied to the RISK so lot quantization
            # happens exactly once, in Decimal.
            # C is held at x1.0: no honest confidence field
            # exists at order time (score.total is a ranking
            # total, not confidence — the old score band is
            # REMOVED, owner ruling 2026-10-24).  The confidence
            # definition is an open design task; until it lands,
            # the formula carries C as a hook, never a fake.
            _tier, tier_mult, _ceil = self._traffic(zone_d)
            _size_mult = float(
                (verdict or {}).get("size_mult", 1.0) or 1.0)
            _posture_mult = float(self._posture.size_scalar) \
                if self._posture is not None else 1.0
            _align_mult = float(self._alignment.size_multiplier) \
                if self.cfg.golden_swing_sizing_enabled else 1.0
            _session_mult = session_multiplier(
                now_utc, self.cfg.session_multipliers)
            _k = float(self.cfg.risk_pct)
            _confidence_mult = 1.0  # unresolved — see above
            risk_money = float(acct["equity"]) \
                * _k * _confidence_mult * tier_mult
            risk_money *= _size_mult
            risk_money *= float(news_mult)
            # P3 posture scalar (S1/S3): the scenario's size
            # expression — trend/rotation/choppy/session posture
            # priced into the risk, never a gate.
            risk_money *= _posture_mult
            risk_money *= _align_mult
            risk_money *= _session_mult
            _stop_dist = abs(float(sig.entry_ref) - stop0)
            result = size_for_risk(
                risk_money, _stop_dist, spec)
            if not result.ok:
                # FUNNEL §1: a counter for the decline, so the
                # evaluation record can classify this refusal as
                # solvency (gate.solvency) instead of an unnamed
                # order-path refusal.  Observation-only.
                # Remaining declines here are broker/input
                # constraints only (volume limit, nonpositive
                # inputs) — below-minimum no longer declines:
                # it floors to the broker minimum (2026-10-24).
                stats["sizing_declined"] = stats.get(
                    "sizing_declined", 0) + 1
                self.journal.log_note(
                    f"mt5: order DECLINED {pb.name}: sizing "
                    f"{result.verdict} — risk {risk_money:.2f} over "
                    f"stop distance "
                    f"{_stop_dist:.2f} gives "
                    f"raw {result.raw_lots} lots")
                log_signal(
                    self.journal,
                    signal_id=f"MT5-{pb.name}-{sig.index}",
                    playbook=pb.name, direction=int(sig.direction),
                    score=float(sig.score), decision="gated",
                    reason=f"sizing_decline: {result.verdict}",
                    entry=float(sig.entry_ref), stop=float(stop0),
                    targets=[float(x) for x in (sig.targets or [])],
                    score_breakdown=(score.breakdown()
                                     if score is not None else None),
                    trigger_source="arrival_score")
                return
            lots = float(result.lots)
            self._sizing_detail[f"MT5-{pb.name}-{sig.index}"] = {
                "authorized_risk": float(result.risk_money),
                "realized_risk": float(result.realized_risk),
                # Estimated stop exposure at order time (NOT the
                # trade's final realized P&L, known only at close).
                "estimated_stop_exposure":
                    float(result.estimated_stop_exposure),
                "drag": float(result.drag),
                "raw_lots": float(result.raw_lots),
                "lots": lots,
                # Min-floor flag (2026-10-24 ruling): True when
                # the raw size fell below the broker minimum and
                # the order was sized UP to the minimum.
                "min_override": bool(result.min_override),
                # Formula inputs, each journaled separately so
                # the record reconstructs the size exactly.
                "sizing_coefficient": _k,
                "confidence_mult": _confidence_mult,
                "scalars": {
                    "traffic_tier": _tier,
                    "traffic_mult": float(tier_mult),
                    "size_mult": _size_mult,
                    "size_mult_regime":
                        (verdict or {}).get("regime", "trend"),
                    "news_mult": float(news_mult),
                    "posture_mult": _posture_mult,
                    "alignment_mult": _align_mult,
                    "session_mult": float(_session_mult)},
                "spec": {
                    "symbol": self.cfg.symbol,
                    "tick_size": float(spec.tick_size),
                    "tick_value": float(spec.tick_value),
                    "volume_min": float(spec.volume_min),
                    "volume_step": float(spec.volume_step),
                    "volume_max": float(spec.volume_max)},
                "stop_distance": float(_stop_dist)}
        else:
            # ENFORCEMENT (owner rulings): a non-risk sizing mode may
            # evaluate but NEVER places an order. A stale config
            # selecting legacy mode refuses here, loudly, instead of
            # silently restoring retired behavior.
            self.journal.log_note(
                f"mt5: order REFUSED {pb.name}: sizing_mode is not "
                "'risk' — legacy sizing cannot place orders "
                "(evaluation/forensics only)")
            return
        # Safety gates (order time, never silent): RiskGuards, then
        # margin + swing margin shield.  Both must pass — a refusal
        # is journalled + counted by leg.
        if not self._check_risk_guards(pb, sig, lots, stop0, acct):
            return
        if not self._check_margin(pb, sig, lots, is_layer, acct):
            return
        # Selection: where is price vs the entry level NOW?
        price = float((self._last_tick or {}).get("mid")
                      or (verdict or {}).get("entry")
                      or sig.entry_ref)
        # Placement-time urgency + volume profile (dispatch
        # context): urgency from the live tick (session + ATR
        # velocity + zone distance), profile from the H1 closed
        # bars.  Signal-time urgency (verdict annotation, for the
        # record + reasoning) is computed in the verdict builders;
        # this is the dispatch-time reading, which may differ if
        # the tick moved since the verdict.
        _urgency = None
        _profile = None
        try:
            _h1 = self._stack.frames.get("H1") \
                if self._stack is not None else None
            _session = session_of(now_utc)[0]
            _urgency = urgency_score(
                price, session=_session,
                zone=(verdict or {}).get("zone"), frame=_h1)
            _profile = volume_profile(_h1)
        except Exception:
            _urgency, _profile = None, None
        # Step 15: the candidate's DECLARED execution intent steers
        # selection (the caller's resolved intent wins; the signal's
        # stamped intent is next; the resolver is the fallback).
        # Layers are position management, not candidates — no intent,
        # legacy geometry.
        _intent = intent
        if _intent is None and not is_layer:
            _intent = (getattr(sig, "execution_intent", None)
                       or execution_intent_for(pb.name, sig))
        kind, level, reason = select_order_type(
            direction=int(sig.direction), price=price,
            entry_ref=float(sig.entry_ref),
            zone=(verdict or {}).get("zone"), atr=float(sig.atr),
            urgency=_urgency, profile=_profile,
            intent_family=(_intent or {}).get("family"))
        if _intent is not None:
            self.journal.log_note(
                f"mt5: execution intent {pb.name}: "
                f"family={_intent.get('family')} — "
                f"{_intent.get('basis')}; selected {kind}: {reason}")
        # Limit ladder (opt-in): stagger the LIMIT across rungs
        # from the playbook level to the adverse zone edge.  Rung 0
        # is placed now (replaces the single-pending level + lots);
        # the rest advance on expiry.  Layers never ladder (they
        # scale through the layer risk budget).
        _ladder_levels: list = []
        _ladder_lots: list = []
        if kind == "limit" and self.cfg.ladder_enabled \
                and not is_layer:
            _rungs = build_ladder(
                direction=int(sig.direction),
                entry_ref=float(sig.entry_ref),
                zone=(verdict or {}).get("zone"), lots=lots,
                rungs=self.cfg.ladder_rungs,
                split=self.cfg.ladder_split)
            if _rungs:
                level = float(_rungs[0][0])
                lots = float(_rungs[0][1])
                _ladder_levels = [float(r[0]) for r in _rungs[1:]]
                _ladder_lots = [float(r[1]) for r in _rungs[1:]]
                self.journal.log_note(
                    f"mt5: ladder {pb.name}: rung 0 {level:.2f} "
                    f"({lots:.2f} lots), "
                    f"{len(_ladder_levels)} rungs remaining")
        # P3: the A8 zone-anchored TP ladder (computed at signal
        # acceptance) replaces the playbook's raw targets when it
        # priced rungs — the opposing zone IS the target map.
        _targets = [float(x) for x in (sig.targets or [])]
        if signal_ctx is not None and signal_ctx.get("targets"):
            _targets = [float(x) for x in signal_ctx["targets"]]
            self.journal.log_note(
                f"mt5: {pb.name} targets from the zone TP "
                f"ladder: {[round(x, 2) for x in _targets]}")
        request = OrderRequest(
            symbol=self.cfg.symbol, direction=int(sig.direction),
            volume=lots, stop_loss=float(stop0),
            entry_ref=float(sig.entry_ref),
            take_profit=float(_targets[0]) if _targets else None,
            targets=list(_targets),
            atr=float(sig.atr), risk_money=0.0,
            alert_id=f"MT5-{pb.name}-{sig.index}",
            bar_index=int(sig.index), magic=self.cfg.magic,
            order_kind=kind,
            order_price=float(level) if kind != "market" else 0.0)
        # P3: park the decision context against the alert — the
        # fill (market now, pending later) consumes it into the
        # per-position management context.
        if signal_ctx is not None and not is_layer:
            self._signal_ctx[request.alert_id] = dict(signal_ctx)
        # The stepper measures R on the BUFFERED stop: pass the
        # invalidation-view request for tracking so the buffer is not
        # applied twice (broker SL is already stop0).
        track_request = OrderRequest(
            symbol=self.cfg.symbol, direction=int(sig.direction),
            volume=lots, stop_loss=float(sig.invalidation),
            entry_ref=float(sig.entry_ref),
            targets=list(_targets),
            atr=float(sig.atr), bar_index=int(sig.index),
            alert_id=request.alert_id, magic=self.cfg.magic,
            is_scalp=pb.name in self.cfg.scalp_playbooks)
        if kind == "invalid":
            stats["invalidated"] = stats.get("invalidated", 0) + 1
            self._record_blocked(f"order:invalid: {reason}", pb.name)
            self.journal.log_note(
                f"mt5: order invalidated {pb.name}: {reason} — "
                f"no order placed")
            log_signal(
                self.journal, signal_id=request.alert_id,
                playbook=pb.name, direction=int(sig.direction),
                score=float(sig.score), decision="invalidated",
                reason=reason,
                entry=float(sig.entry_ref),
                stop=float(sig.invalidation),
                targets=[float(x) for x in (sig.targets or [])],
                factors=list(getattr(sig, "factors", []) or []),
                dxy_corr=(verdict or {}).get("dxy_corr"),
                urgency=(verdict or {}).get("urgency"))
            return
        if kind != "market":
            placed = self._place_pending_order(
                pb, sig, verdict, kind, level, request, track_request,
                stats, entry_bar, is_layer=is_layer,
                ladder_levels=_ladder_levels,
                ladder_lots=_ladder_lots,
                funding_events=funding_events, score=score)
            if placed and fingerprint:
                self.entry_gate.record_open(time.time(), fingerprint)
            return
        _t0 = time.monotonic()
        res = executor.place_order(request)
        self._health.note_order(time.monotonic() - _t0)
        self._journal_new_breaches()
        if not res.ok:
            self.journal.log_note(
                f"mt5: order FAILED {pb.name}: {res.message}")
            return
        fill = res.price or sig.entry_ref
        # Slippage: adverse fill vs the tick mid the order aimed at
        # (long filled above the aim, short below it), in ATR units.
        self._health.note_fill(
            int(sig.direction) * (float(fill) - float(price)),
            float(sig.atr))
        if is_layer:
            self._record_layer_fill(pb.name, sig, fill=fill, lots=lots,
                                    stop0=stop0, stats=stats,
                                    funding_events=funding_events)
            return
        if fingerprint:
            self.entry_gate.record_open(time.time(), fingerprint)
        _narr_state, _narr_kwargs = self._narrative_fill_plan(sig, fill)
        h1 = self._stack.frames["H1"]
        self.tick_manager.open(ticket=int(res.ticket),
                               request=track_request, fill_price=fill,
                               entry_bar=len(h1) - 1 if entry_bar is None
                               else entry_bar, broker_sl=stop0,
                               playbook=pb.name,
                               **_narr_kwargs)
        if _narr_state is not None:
            self._narrative_states[int(res.ticket)] = _narr_state
        # A13: open the position's RiskPool ledger on the fill.
        # Step 2: the pool books the BROKER stop (stop0), never the
        # structural invalidation — sizing, close R and the tick
        # manager all measure on the broker stop, and booking the
        # tighter structural stop understated true exposure
        # (2026-10-26: pool booked 2.28, broker exposure 3.27).
        self._pool_for_open(
            int(res.ticket), lots=float(lots), entry=float(fill),
            stop=float(stop0),
            direction=int(sig.direction),
            alert_id=request.alert_id)
        stats["executed"] = stats.get("executed", 0) + 1
        # Open record: without it, /trades /report /harvest cannot see
        # this fill (they join close -> open by alert_id).  Two stops
        # ride the record, each under its own name: ``stop`` is the
        # structural invalidation (analysis), ``broker_sl`` is the
        # buffered stop0 that owns the trade — and close R divides
        # by the broker distance, so risk_money is broker-based too.
        try:
            risk_money = spec.money_for_move(
                abs(fill - float(stop0)), lots)
        except Exception:
            risk_money = 0.0
        self.journal.log_open(
            alert_id=request.alert_id, instrument=self.cfg.symbol,
            mode="MT5", playbook=pb.name, phase="live",
            direction=int(sig.direction),
            score=(float(score.total)
                   if score is not None else float(sig.score)),
            source="mt5", ticket=int(res.ticket), entry=float(fill),
            stop=float(sig.invalidation), broker_sl=float(stop0),
            volume=float(lots),
            risk_money=float(risk_money),
            factors=list(getattr(sig, "factors", []) or []),
            intended_entry=float(sig.entry_ref),
            slippage=(int(sig.direction)
                      * (float(fill) - float(sig.entry_ref))))
        self.journal.log_note(
            f"mt5 fill: ticket {res.ticket} {pb.name} "
            f"{'+' if sig.direction > 0 else '-'}1 {lots} lots @ {fill:.2f} "
            f"SL {stop0:.2f}")
        _sd = self._sizing_detail.get(request.alert_id)
        if _sd is not None:
            _pool = self._pools.get(int(res.ticket))
            self.journal.log_note(
                f"mt5 sizing: {pb.name} authorized risk "
                f"${_sd['authorized_risk']:.2f} -> estimated stop "
                f"exposure "
                f"${_sd.get('estimated_stop_exposure', _sd['realized_risk']):.2f} "
                f"(quantization drag "
                f"${_sd['drag']:.2f}, raw {_sd['raw_lots']:.4f} "
                f"lots"
                f"{' · MIN-LOT FLOOR' if _sd.get('min_override') else ''}) · pool "
                f"{_pool.state() if _pool is not None else 'n/a'}")
        # Fill chart: same levels + the actual fill price, saved
        # alongside the signal chart and attached to the fill alert.
        # Step 7: decision score + broker stop mirror the fill record
        # logged just above (score.total or sig.score; stop0).
        png, _path = self._build_chart(
            sig, verdict, alert_id=request.alert_id, playbook=pb.name,
            fill_price=fill, suffix="-fill",
            decision_score=(float(score.total)
                            if score is not None else float(sig.score)),
            broker_stop=float(stop0))
        _, view_path = self._build_signal_view(
            sig, verdict, alert_id=request.alert_id, playbook=pb.name,
            fill_price=fill, suffix="-fill",
            decision_score=(float(score.total)
                            if score is not None else float(sig.score)),
            broker_stop=float(stop0))
        self._send_alert(
            f"MT5 FILL {pb.name} {'+' if sig.direction > 0 else '-'}1 "
            f"{lots} lots @ {fill:.2f} SL {stop0:.2f} "
            f"(ticket {res.ticket})",
            chart_png=png, html_path=view_path)

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

    def _mgmt_bar_now(self, clock_tf: str) -> int:
        """XD-4: last closed bar INDEX on the given management
        clock (len of that clock's closed frame, minus one).
        A clock with no frame falls back to the H1 index —
        fail-open, today's behaviour — never a skip."""
        frames = self._stack.frames if self._stack is not None \
            else {}
        frame = frames.get(clock_tf) if frames else None
        if frame is not None and len(frame):
            return len(frame) - 1
        h1 = frames.get("H1") if frames else None
        if h1 is not None and len(h1):
            return len(h1) - 1
        return 0

    def _track_pending_order(self, order: PendingOrder, *,
                             playbook: str, zone: dict | None,
                             invalidation: float, placed_bar: int,
                             sig, verdict, track_request,
                             is_layer: bool = False,
                             ladder_levels: list | None = None,
                             ladder_lots: list | None = None,
                             decision_score: float | None = None
                             ) -> PendingOrderState:
        """Register a broker pending order in engine tracking.

        Step 4: premise and entry clock are derived from the
        playbook name via the registry (``native_tf``), at this
        single tracking point, so every caller — H1 path, native
        routes, layers, restart adoption — classifies the pending
        the same way without new arguments threaded through the
        order path.  A native pending's entry window preserves the
        H1 window's WALL duration (``pending_expiry_bars`` H1 bars)
        counted in entry-clock bars (ceil): qm_m5 gets 48 M5 bars
        (4h, same duration as before, finer granularity); qm_d1
        gets 1 D1 bar (its route cannot be evaluated more finely
        than daily closes — the old 4h H1 kill of a daily setup
        was the units bug).  The entry-placed index is the forming
        index on the entry clock (len of that clock's frame), the
        same convention as ``_current_h1_bar``.
        """
        entry_clock = _native_tf(playbook) or "H1"
        premise = "native" if entry_clock != "H1" else "h1"
        if premise == "native" and self._stack is not None:
            _frame = self._stack.frames.get(entry_clock)
            entry_placed = len(_frame) if _frame is not None \
                else int(placed_bar)
        else:
            entry_placed = int(placed_bar)
        _minutes = {"M5": 5, "D1": 1440, "H1": 60}.get(entry_clock, 60)
        _count = (int(self.cfg.pending_expiry_bars) * 60
                  + _minutes - 1) // _minutes
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
            premise=premise, entry_clock=entry_clock,
            entry_placed_bar=int(entry_placed),
            entry_expiry_bar=int(entry_placed) + max(1, _count),
            track_request=track_request, sig=sig, verdict=verdict,
            is_layer=bool(is_layer),
            placed_score=(float(decision_score)
                          if decision_score is not None
                          else float(getattr(sig, "score", 0.0) or 0.0)),
            ladder_levels=list(ladder_levels or []),
            ladder_lots=list(ladder_lots or []))
        self._pending_orders[state.ticket] = state
        return state

    def _place_ladder_rung(self, state: PendingOrderState,
                           level: float, lots: float,
                           entry_bar: int, stats: dict) -> bool:
        """Advance a ladder: place the next rung after the previous
        one expired.  Rebuilds the request from the state's own
        fields (same direction / SL / TP / alert_id); the sig and
        verdict ride along so fill handling and reasoning see the
        originating setup.  Returns False when the playbook is gone
        or the broker refuses — the ladder dies, no retry."""
        sig = getattr(state, "sig", None)
        if sig is None:
            return False
        playbooks = {pb.name: pb for pb in self._get_playbooks()}
        pb = playbooks.get(state.playbook)
        if pb is None:
            self.journal.log_note(
                f"mt5: ladder rung NOT placed: playbook "
                f"{state.playbook} unknown")
            return False
        request = OrderRequest(
            symbol=self.cfg.symbol, direction=int(state.direction),
            volume=float(lots), order_kind="limit",
            order_price=float(level), stop_loss=float(state.sl),
            take_profit=float(state.tp) if state.tp is not None
            else None, magic=self.cfg.magic,
            alert_id=state.alert_id)
        track_request = OrderRequest(
            symbol=self.cfg.symbol, direction=int(state.direction),
            volume=float(lots), order_kind="limit",
            order_price=float(level), stop_loss=float(state.sl),
            take_profit=float(state.tp) if state.tp is not None
            else None, magic=self.cfg.magic,
            alert_id=state.alert_id)
        return self._place_pending_order(
            pb, sig, state.verdict, "limit", float(level), request,
            track_request, stats, entry_bar, is_layer=state.is_layer,
            ladder_levels=list(state.ladder_levels),
            ladder_lots=list(state.ladder_lots))

    def _place_pending_order(self, pb, sig, verdict, kind: str,
                             level: float, request: OrderRequest,
                             track_request: OrderRequest,
                             stats: dict,
                             entry_bar: int | None,
                             is_layer: bool = False,
                             ladder_levels: list | None = None,
                             ladder_lots: list | None = None,
                             funding_events: list | None = None,
                             score=None
                             ) -> bool:
        """Place a pending LIMIT / STOP and start tracking it.
        Returns True when the broker accepted the order."""
        # One-position rule: a pending order COUNTS — never a second
        # pending while one pending is active, and never a base
        # pending while a position is active.  A pending LAYER is the
        # exception: it rides on the held position.
        if (not is_layer and self.tick_manager.tickets()) \
                or self._pending_orders:
            self.journal.log_note(
                f"mt5: pending {kind} NOT placed {pb.name}: already "
                f"holding/pending")
            return False
        _t0 = time.monotonic()
        res = self.executor.place_order(request)
        self._health.note_order(time.monotonic() - _t0)
        self._journal_new_breaches()
        if not res.ok:
            self.journal.log_note(
                f"mt5: pending {kind} FAILED {pb.name}: {res.message}")
            return False
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
            track_request=track_request, is_layer=is_layer,
            ladder_levels=ladder_levels, ladder_lots=ladder_lots,
            decision_score=(float(score.total)
                            if score is not None else None))
        if is_layer and funding_events:
            self._pending_funding[state.ticket] = list(funding_events)
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
        # Step 7: decision score + broker stop mirror the pending
        # record (score.total or None; the pending order's stop).
        png, _path = self._build_chart(
            sig, verdict, alert_id=request.alert_id, playbook=pb.name,
            suffix="-pending",
            decision_score=(float(score.total)
                            if score is not None else None),
            broker_stop=float(request.stop_loss))
        _, view_path = self._build_signal_view(
            sig, verdict, alert_id=request.alert_id, playbook=pb.name,
            suffix="-pending",
            decision_score=(float(score.total)
                            if score is not None else None),
            broker_stop=float(request.stop_loss))
        self._send_alert(
            f"MT5 PENDING {kind.upper()} {pb.name} "
            f"{'+' if sig.direction > 0 else '-'}1 "
            f"{request.volume} lots @ {level:.2f} "
            f"SL {request.stop_loss:.2f} (ticket {res.ticket})",
            chart_png=png, html_path=view_path)
        return True

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
            # Setup lifecycle (owner rulings 2026-10-30 §1): a
            # routed setup's cancelled order resolves its
            # order_submitted stage — expiry-shaped cancels are
            # terminal; other cancels return the setup to armed
            # inside its validity window.
            if self._setup_store is not None:
                _lcs = self._setup_store.by_alert_id(
                    state.alert_id)
                if _lcs is not None and \
                        _lcs.stage == _setup_lc.STAGE_SUBMITTED:
                    if "expir" in str(reason).lower():
                        self._transition_setup(
                            _lcs, _setup_lc.STAGE_EXPIRED,
                            f"pending cancelled: {reason}")
                    else:
                        self._transition_setup(
                            _lcs, _setup_lc.STAGE_ARMED,
                            f"pending cancelled ({reason}) — "
                            "setup re-armed within validity")
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
            # Prefer the untracked position (a hedging remainder
            # fill is a NEW ticket; the already-tracked base
            # position would read as zero new volume).
            untracked = [p for p in matched
                         if p.ticket not in self.tick_manager.tickets()]
            self._record_pending_fill(
                state, untracked[0] if untracked else matched[0], stats)
            return True
        self.journal.log_note(
            f"mt5 pending cancel FAILED ticket {ticket} ({reason}): "
            f"{res.message} — retry next tick")
        return False

    def _position_for_alert(self, alert_id: str):
        """Held OPEN TickManager position with this alert_id, if any.
        Closed positions don't count: a remainder filling after the
        base closed opens fresh (the broker filled a live order —
        it must be tracked, not dropped)."""
        for ticket in self.tick_manager.tickets():
            pos = self.tick_manager.position(ticket)
            if pos is not None and not pos.closed \
                    and pos.alert_id == alert_id:
                return pos
        return None

    def _fill_volume(self, state: PendingOrderState, pos) -> float:
        """Volume of THIS fill event (not the position's total).

        Netting (``pos`` already tracked, grown by the fill) ->
        the growth.  Hedging / first fill (``pos`` untracked) ->
        the position's volume.  A ``pos`` without a volume (direct
        test calls) -> the state's volume (a full fill).
        """
        _pos_vol = getattr(pos, "volume", None)
        if _pos_vol is None:
            return float(state.volume)
        if int(pos.ticket) in self.tick_manager.tickets():
            tracked = self.tick_manager.position(int(pos.ticket))
            base_vol = float(tracked.volume) \
                if tracked is not None else 0.0
            return max(0.0, float(_pos_vol) - base_vol)
        return float(_pos_vol)

    def _fill_track_request(self, state: PendingOrderState,
                            volume: float, fill_bar: int):
        """Placement-time track request re-pointed at the fill:
        the management clocks start AT THE FILL (time-exit bars
        from the fill bar — not the placement signal bar), and the
        tracked volume is the filled volume.  SL/TP (invalidation
        + targets) are the placement-time values from the signal —
        attached unchanged, never recomputed from the fill price.
        """
        track = state.track_request
        if track is None:
            return OrderRequest(
                symbol=self.cfg.symbol, direction=int(state.direction),
                volume=float(volume),
                stop_loss=float(state.invalidation),
                entry_ref=float(state.level),
                alert_id=state.alert_id, magic=self.cfg.magic,
                bar_index=fill_bar)
        return replace(track, volume=float(volume),
                       bar_index=fill_bar)

    def _record_pending_fill(self, state: PendingOrderState,
                             pos, stats: dict) -> None:
        """A pending order filled — the fill event drives CONFIRMED
        -> ACTIVE (reconcile matches the broker position to the
        pending by alert_id; never price polling).

        SL/TP are the PLACEMENT-time values from the playbook's
        signal (SL = invalidation + buffer via planned_stop, TP =
        the signal's targets) — attached unchanged, never
        recomputed from the fill price.  A resting LIMIT fill is
        at-or-better by construction, so it cannot slip against
        you — the pre-computed R holds.

        On fill: state -> ACTIVE; fill price/time/volume recorded
        (open record + fill note); the filled order leaves pending
        tracking (fully filled: nothing remains pending); the
        management clocks start AT THE FILL (time-exit bars from
        the fill bar, holding cap from the fill time); handoff to
        the tick manager.  From this moment the SL owns
        invalidation — pre-entry invalidation no longer applies.

        Partial fill: state -> ACTIVE with the PARTIAL volume;
        the unfilled remainder stays pending (reduced volume) —
        not a full fill, not a failure.  When the remainder later
        fills it lands on the held position as an added leg (same
        setup, same alert_id), never a second position.

        A pending LAYER fills onto the held position instead (the
        layer entry is the pending price — the deal price); if the
        position closed before the pending filled, the layer falls
        back to opening a fresh base position.
        """
        # Setup lifecycle (owner rulings 2026-10-30 §1): a routed
        # setup whose pending filled resolves to FILLED.
        if self._setup_store is not None:
            _lcs = self._setup_store.by_alert_id(state.alert_id)
            if _lcs is not None and \
                    _lcs.stage == _setup_lc.STAGE_SUBMITTED:
                self._transition_setup(
                    _lcs, _setup_lc.STAGE_FILLED,
                    f"pending filled: ticket "
                    f"{getattr(pos, 'ticket', '?')}",
                    ticket=int(getattr(pos, "ticket", 0) or 0)
                    or None)
        # Slippage: adverse fill vs the pending level (long filled
        # above it, short below it), in ATR units.  Adopted pendings
        # (restart) have no signal — no ATR, no sample.
        _sig = getattr(state, "sig", None)
        if _sig is not None:
            self._health.note_fill(
                int(state.direction)
                * (float(pos.entry_price) - float(state.level)),
                float(getattr(_sig, "atr", 0.0) or 0.0))
            self._journal_new_breaches()
        _pos_vol = getattr(pos, "volume", None)
        filled_vol = self._fill_volume(state, pos)
        fill_bar = self._current_h1_bar()
        # --- layer fill (onto a held position) -------------------
        if state.is_layer and self.tick_manager.tickets():
            if len(self.tick_manager.tickets()) == 1:
                if filled_vol < state.volume - _VOL_EPS:
                    # Partial layer fill: the leg lands partially,
                    # the remainder stays pending.  No breakeven
                    # move yet — that fires once per layer, when the
                    # remainder completes it (see _record_layer_fill).
                    self.tick_manager.add_layer(
                        self.tick_manager.tickets()[0],
                        entry=float(pos.entry_price),
                        volume=filled_vol, sl=float(state.sl),
                        alert_id=state.alert_id,
                        playbook=state.playbook)
                    # Step 5: partial fills consume their trigger
                    # too — the layer exists, even partially.
                    _psig = getattr(state, "sig", None)
                    self._layer_trigger_ids.setdefault(
                        self.tick_manager.tickets()[0], set()).add(
                        f"MT5-{state.playbook or 'pending'}-"
                        f"{getattr(_psig, 'index', '?')}")
                    self._pool_book_layer(
                        self.tick_manager.tickets()[0],
                        lots=float(filled_vol),
                        entry=float(pos.entry_price),
                        stop=float(state.sl),
                        direction=int(state.direction),
                        alert_id=f"{state.alert_id}#partial")
                    state.volume = state.volume - filled_vol
                    stats["pending_partial"] = stats.get(
                        "pending_partial", 0) + 1
                    self.journal.log_note(
                        f"mt5 partial layer fill: ticket {pos.ticket} "
                        f"{state.playbook or 'pending'} "
                        f"{'+' if state.direction > 0 else '-'}1 "
                        f"{filled_vol} lots @ "
                        f"{float(pos.entry_price):.2f} "
                        f"({state.volume} lots remain pending)")
                    return
                self._pending_orders.pop(state.ticket, None)
                self._record_layer_fill(
                    state.playbook or "pending", state.sig,
                    fill=float(state.level), lots=float(state.volume),
                    stop0=float(state.sl), stats=stats,
                    funding_events=self._pending_funding.pop(
                        state.ticket, None))
                return
            # No single held position: drop (same as a full layer
            # fill with no position — see _record_layer_fill).
            self._pending_orders.pop(state.ticket, None)
            self._record_layer_fill(
                state.playbook or "pending", state.sig,
                fill=float(state.level), lots=float(state.volume),
                stop0=float(state.sl), stats=stats,
                funding_events=self._pending_funding.pop(
                    state.ticket, None))
            return
        # --- remainder fill (a partial fill's second leg) ---------
        # ``filled_vol`` is already the NEW volume (the growth for
        # netting, the new position's volume for hedging) — it
        # lands on the held position as an added leg.
        held = self._position_for_alert(state.alert_id)
        if held is not None:
            added = filled_vol
            self._pending_orders.pop(state.ticket, None)
            stats["pending_filled"] = stats.get("pending_filled", 0) + 1
            if added > _VOL_EPS:
                self.tick_manager.add_layer(
                    held.ticket, entry=float(pos.entry_price),
                    volume=added, sl=float(state.sl),
                    alert_id=state.alert_id, playbook=state.playbook)
                self._pool_book_layer(
                    held.ticket, lots=float(added),
                    entry=float(pos.entry_price),
                    stop=float(state.sl),
                    direction=int(state.direction),
                    alert_id=f"{state.alert_id}#remainder")
                self.journal.log_note(
                    f"mt5 fill: ticket {pos.ticket} "
                    f"{state.playbook or 'pending'} "
                    f"{'+' if state.direction > 0 else '-'}1 "
                    f"{added} lots @ {float(pos.entry_price):.2f} "
                    f"SL {state.sl:.2f} (PENDING {state.kind.upper()} "
                    f"remainder filled — position now "
                    f"{float(held.volume):.4g} lots)")
            else:
                self.journal.log_note(
                    f"mt5 fill: ticket {pos.ticket} "
                    f"{state.playbook or 'pending'} remainder "
                    f"already accounted for (no volume added)")
            return
        # --- base fill (full, or the first partial leg) -----------
        if filled_vol < state.volume - _VOL_EPS:
            # Partial: ACTIVE with the partial volume; the
            # remainder stays pending (reduced volume).
            track = self._fill_track_request(state, filled_vol,
                                             fill_bar)
            _narr_state, _narr_kwargs = self._narrative_fill_plan(
                getattr(state, "sig", None), float(pos.entry_price))
            self.tick_manager.open(
                ticket=int(pos.ticket), request=track,
                fill_price=float(pos.entry_price),
                entry_bar=fill_bar, broker_sl=float(state.sl),
                playbook=state.playbook,
                **_narr_kwargs)
            if _narr_state is not None:
                self._narrative_states[int(pos.ticket)] = _narr_state
            self._pool_for_open(
                int(pos.ticket), lots=float(filled_vol),
                entry=float(pos.entry_price),
                stop=float(state.sl),
                direction=int(state.direction),
                alert_id=state.alert_id)
            state.volume = state.volume - filled_vol
            stats["executed"] = stats.get("executed", 0) + 1
            stats["pending_filled"] = stats.get("pending_filled", 0) + 1
            stats["pending_partial"] = stats.get("pending_partial", 0) + 1
            self._journal_fill_open(state, pos, filled_vol, stats,
                                    partial=True)
            return
        # Full fill: the order leaves pending tracking entirely.
        self._pending_orders.pop(state.ticket, None)
        track = self._fill_track_request(state, float(state.volume),
                                         fill_bar)
        _narr_state, _narr_kwargs = self._narrative_fill_plan(
            getattr(state, "sig", None), float(pos.entry_price))
        self.tick_manager.open(ticket=int(pos.ticket), request=track,
                               fill_price=float(pos.entry_price),
                               entry_bar=fill_bar,
                               broker_sl=float(state.sl),
                               playbook=state.playbook,
                               **_narr_kwargs)
        if _narr_state is not None:
            self._narrative_states[int(pos.ticket)] = _narr_state
        self._pool_for_open(
            int(pos.ticket), lots=float(state.volume),
            entry=float(pos.entry_price),
            stop=float(state.sl),
            direction=int(state.direction),
            alert_id=state.alert_id)
        stats["executed"] = stats.get("executed", 0) + 1
        stats["pending_filled"] = stats.get("pending_filled", 0) + 1
        self._journal_fill_open(state, pos, float(state.volume), stats,
                                partial=False)

    def _journal_fill_open(self, state: PendingOrderState, pos,
                           volume: float, stats: dict,
                           *, partial: bool) -> None:
        """Open record + fill note/alert for a base fill (full or
        partial).  Same open record as a market fill — the entry
        is the broker's fill price, the volume is the filled volume.
        Step 2: the record carries BOTH stops under their own names
        (``stop`` = structural invalidation, ``broker_sl`` = the
        broker stop that owns the trade) and the DECISION score
        (state.placed_score, set at placement from the Score object
        — never the sleeve-base placeholder that used to ride here
        as a flat 60.0 while the signal record said 32.71).
        risk_money is broker-distance exposure, matching sizing
        and close R."""
        sig = getattr(state, "sig", None)
        try:
            spec = self.executor.symbol_spec(self.cfg.symbol)
            risk_money = spec.money_for_move(
                abs(float(pos.entry_price) - float(state.sl)),
                float(volume))
        except Exception:
            risk_money = 0.0
        _decision_score = float(state.placed_score or 0.0)
        if not _decision_score and sig is not None:
            _decision_score = float(getattr(sig, "score", 0.0) or 0.0)
        self.journal.log_open(
            alert_id=state.alert_id, instrument=self.cfg.symbol,
            mode="MT5", playbook=state.playbook or "pending",
            phase="live", direction=int(state.direction),
            score=float(_decision_score),
            source="mt5", ticket=int(pos.ticket),
            entry=float(pos.entry_price),
            stop=float(state.invalidation),
            broker_sl=float(state.sl),
            volume=float(volume), risk_money=float(risk_money),
            factors=list(getattr(sig, "factors", []) or [])
            if sig is not None else [],
            intended_entry=float(state.level),
            slippage=(int(state.direction)
                      * (float(pos.entry_price) - float(state.level))))
        self._det_journal("trade_opened", {
            "alert_id": state.alert_id, "ticket": int(pos.ticket),
            "entry": float(pos.entry_price),
            "playbook": state.playbook or "pending",
            "direction": int(state.direction),
            "volume": float(volume)},
            idempotency_key=f"open-{state.alert_id}-{int(pos.ticket)}")
        self.journal.log_note(
            f"mt5 fill: ticket {pos.ticket} "
            f"{state.playbook or 'pending'} "
            f"{'+' if state.direction > 0 else '-'}1 "
            f"{volume} lots @ {float(pos.entry_price):.2f} "
            f"SL {state.sl:.2f} (PENDING {state.kind.upper()} filled"
            + (" — PARTIAL, "
               f"{state.volume} lots remain pending)" if partial
               else ")"))
        _sd = self._sizing_detail.get(state.alert_id)
        if _sd is not None:
            _pool = self._pools.get(int(pos.ticket))
            self.journal.log_note(
                f"mt5 sizing: {state.playbook or 'pending'} "
                f"authorized risk ${_sd['authorized_risk']:.2f} -> "
                f"estimated stop exposure "
                f"${_sd.get('estimated_stop_exposure', _sd['realized_risk']):.2f} "
                f"(quantization drag ${_sd['drag']:.2f}, raw "
                f"{_sd['raw_lots']:.4f} lots"
                f"{' · MIN-LOT FLOOR' if _sd.get('min_override') else ''}) · pool "
                f"{_pool.state() if _pool is not None else 'n/a'}")
        # Fill chart when the signal is known (placed this run);
        # adopted pendings (restart) have no signal — text only.
        # Step 7: decision score is the recorded placed_score, or
        # unavailable when none was recorded (never the sleeve
        # placeholder); broker stop is the filled position's SL.
        png = None
        view_path = None
        if sig is not None and getattr(state, "verdict", None) is not None:
            png, _path = self._build_chart(
                sig, state.verdict, alert_id=state.alert_id,
                playbook=state.playbook, fill_price=float(pos.entry_price),
                suffix="-fill",
                decision_score=(float(state.placed_score)
                                if state.placed_score else None),
                broker_stop=float(state.sl))
            _, view_path = self._build_signal_view(
                sig, state.verdict, alert_id=state.alert_id,
                playbook=state.playbook,
                fill_price=float(pos.entry_price), suffix="-fill",
                decision_score=(float(state.placed_score)
                                if state.placed_score else None),
                broker_stop=float(state.sl))
        self._send_alert(
            f"MT5 FILL {state.playbook or 'pending'} "
            f"{'+' if state.direction > 0 else '-'}1 "
            f"{volume} lots @ {float(pos.entry_price):.2f} "
            f"SL {state.sl:.2f} (ticket {pos.ticket})"
            + (" — PARTIAL" if partial else ""),
            chart_png=png, html_path=view_path)

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
            # DXY correlation: confluence context, never a gate.
            # None = DXY unavailable (scorer treats it as 0.0).
            verdict["dxy_corr"] = (
                self._dxy.state.corr
                if self._dxy.state.available else None)
            # Signal-time urgency (dispatch context, never a gate).
            verdict["urgency"] = self._urgency_for(
                h1["close"].iloc[t], h1.index[t],
                verdict.get("zone"), sliced)
            return verdict
        except Exception as exc:
            self.journal.log_note(
                f"mt5: hierarchy evaluation failed at bar {t}: {exc}")
            return None

    # FUNNEL §2: how old the newest closed M5 bar may be before
    # the data itself is a FACT refusal (gate.data) — three missed
    # M5 bars while the market is open.  Freshness is measured
    # against the tick clock, never assumed.
    _QM_M5_STALE_AFTER_S = 900.0

    def _run_qm_m5(self, m5_df, stats: dict | None = None, *,
                   source: str = _lifecycle.SOURCE_LIVE_CLOSE
                   ) -> list[dict]:
        """qm_m5 REAL order path on its own M5 clock (funnel §2 —
        closes the verified timeframe-wiring defect).

        Phase 2 (P0-1): every candidate carries lifecycle
        provenance (formation / touch / creation / evaluated-bar
        / evaluation timestamps, source, instance key — see
        candidate_lifecycle) and is execution-eligible ONLY when
        discovered on a live close inside its validity window.
        Warmup scans (``source="warmup_scan"``), restart catch-up
        and replays journal their candidates as analytical
        records and NEVER reach the order path; a durable
        seen-instance set makes routing idempotent across
        restarts.  This closes the EC2 burst: 32 historical
        touches re-routed as live orders on one stale bar.

        Before this route existed, qm_m5 sat in the H1 roster and
        the signal loop fed it the H1 frame: its M5 pattern
        windows, swing lookbacks and hold periods ran on 60-minute
        bars, while the only true-M5 scan (the old ``_advance_qm``)
        was annotation-only and never reached an order.  This
        route clones the repo's proven native-clock precedent
        (:meth:`_run_qm_d1`): it runs on the real M5 frame at warm
        start and on every M5 close (see :meth:`_refresh_bars`),
        each fresh touch (deduped by ``_qm_m5_last_logged``) goes
        through :meth:`_execute_qm_m5_candidate`, and every
        evaluation — including no-touch closes — writes its
        funnel record.

        The annotation consumers of the old scanner
        (``qm_annotation`` in the verdict, ``qm_touch`` in
        ``evaluate_hierarchy``) are re-fed from this route's
        latest-touch state — same dict shape — so their behaviour
        is unchanged except that the annotation now comes from
        the real evaluation.  There is no second QM scan left:
        one playbook, one clock, one route.  Kill switch:
        ``/set playbook.qm_m5 off``.
        """
        stats = stats if stats is not None else {}
        if "qm_m5" in (self.cfg.disabled_playbooks or ()):
            if not self._qm_m5_disabled_noted:
                self._qm_m5_disabled_noted = True
                self._emit_eval(
                    playbook="qm_m5", evaluation_clock="m5_close",
                    intended_tf="M5",
                    last_bar_ts=(m5_df.index[-1]
                                 if m5_df is not None and len(m5_df)
                                 else None),
                    outcome="not_eligible",
                    reason_code="path.kill_switch",
                    reason="disabled via /set kill switch — "
                           "not evaluated")
            return []
        self._qm_m5_disabled_noted = False
        try:
            if m5_df is None or len(m5_df) < 30:
                self._emit_eval(
                    playbook="qm_m5", evaluation_clock="m5_close",
                    intended_tf="M5", outcome="not_eligible",
                    reason_code="gate.data",
                    reason="M5 frame unavailable or shorter than "
                           "30 bars — not evaluated")
                return []
            # Step 4: native-clock pending guards run first, on
            # every M5 close, before new candidates are evaluated.
            self._check_native_pendings("M5", stats)
            last_bar_ts = m5_df.index[-1]
            # Data freshness against the tick clock (the engine's
            # clock); the wall clock is only a bare-test fallback.
            now_ref = None
            if self._last_tick and \
                    self._last_tick.get("time") is not None:
                try:
                    now_ref = pd.Timestamp(self._last_tick["time"])
                except Exception:
                    now_ref = None
            if now_ref is None:
                now_ref = pd.Timestamp.now(tz="UTC")
            if now_ref.tzinfo is None:
                now_ref = now_ref.tz_localize("UTC")
            last_close = last_bar_ts + pd.Timedelta(minutes=5)
            freshness_s = max(
                0.0, (now_ref - last_close).total_seconds())
            pb = QMM5Playbook()
            ctx = EngineContext(df=m5_df, states=[], years=0.0,
                                instrument=self.cfg.symbol)
            out: list[dict] = []
            seen = self._load_seen_instances()
            last_index = len(m5_df) - 1
            for cand in pb.candidates(ctx):
                # The annotation state follows the LATEST touch,
                # evaluated for an order or not (the old
                # annotation scanner's contract, now fed by the
                # real route).
                lc0 = dict(getattr(cand.outcome.signal,
                                   "live_context", None) or {})
                latest = {"direction": int(cand.direction),
                          "touch_ts":
                              m5_df.index[cand.signal_index],
                          "touch_index": int(cand.signal_index),
                          "qm_level": lc0.get("qm_level"),
                          "head": lc0.get("head"),
                          "neckline": lc0.get("neckline"),
                          "tags": lc0.get("tags", [])}
                self._qm_m5_last_touch = latest
                self._qm_touch = latest
                if cand.signal_index <= self._qm_m5_last_logged:
                    continue  # already evaluated on an earlier scan
                sig = cand.outcome.signal
                # Step 15: stamp the declared execution intent
                # (QM = limit thesis: entry at the QM level).
                sig.execution_intent = execution_intent_for(
                    "qm_m5", sig)
                lc = dict(getattr(sig, "live_context", None) or {})
                touch = {"direction": int(cand.direction),
                         "touch_ts": m5_df.index[cand.signal_index],
                         "touch_index": int(cand.signal_index),
                         "qm_level": lc.get("qm_level"),
                         "head": lc.get("head"),
                         "neckline": lc.get("neckline"),
                         "tags": lc.get("tags", []),
                         "r_net": float(cand.outcome.r_net),
                         "exit_reason": cand.outcome.exit_reason}
                out.append(touch)
                # ---- Phase 2 lifecycle: provenance + eligibility
                # + idempotency.  The instance key is time-based
                # (touch timestamp), so it is stable across frame
                # rebuilds and restarts; the durable seen set is
                # what makes "the same setup can never be routed
                # twice" true across processes, not just scans.
                touch_ts = touch["touch_ts"]
                touch_age = last_index - int(cand.signal_index)
                _p0 = lc.get("p0_index")
                formation_ts = (
                    m5_df.index[int(_p0)]
                    if _p0 is not None
                    and 0 <= int(_p0) < len(m5_df) else None)
                level = lc.get("qm_level")
                key = _lifecycle.instance_key(
                    "qm_m5", pattern_ts=touch_ts,
                    direction=int(cand.direction),
                    level=(float(level) if level is not None
                           else float(sig.entry_ref)))
                if key in seen:
                    # Already terminally disposed in an earlier
                    # run/scan: consume silently (its record
                    # exists in that run's journal) and advance
                    # the cursor past it.
                    self._qm_m5_last_logged = max(
                        self._qm_m5_last_logged,
                        int(cand.signal_index))
                    continue
                eligible, elig_code, elig_reason = \
                    _lifecycle.assess_eligibility(
                        source=source, touch_age_bars=touch_age,
                        validity_window_bars=(
                            _lifecycle.QM_VALIDITY_WINDOW_BARS))
                prov = _lifecycle.provenance_block(
                    key=key, formation_ts=formation_ts,
                    touch_ts=touch_ts, created_ts=now_ref,
                    evaluated_bar_ts=last_bar_ts,
                    evaluation_ts=now_ref, source=source,
                    touch_age_bars=touch_age,
                    validity_window_bars=(
                        _lifecycle.QM_VALIDITY_WINDOW_BARS),
                    eligible=eligible,
                    eligibility_code=elig_code)
                if not eligible:
                    # Analytical record only — NEVER the order
                    # path.  Warmup/catch-up/replay provenance is
                    # "not_executable" (ineligible bucket); an
                    # expired or unageable LIVE candidate is a
                    # refusal on a data fact (gate.data).
                    _warm = (elig_code ==
                             _lifecycle.CODE_WARMUP_INELIGIBLE)
                    self._emit_eval(
                        playbook="qm_m5",
                        evaluation_clock="m5_close",
                        intended_tf="M5", actual_input_tf="M5",
                        last_bar_ts=last_bar_ts,
                        data_freshness_s=freshness_s,
                        outcome="candidate_emitted",
                        reason_code=elig_code, reason=elig_reason,
                        candidate=self._candidate_dict(sig, None,
                                                       prov),
                        decision={
                            "result": ("not_executable" if _warm
                                       else "refused"),
                            "reason_code": elig_code,
                            "reason": elig_reason,
                            "execution_eligible": False,
                            "order_result": "not_attempted",
                            **_verdict_blocks(
                                analytical="not_evaluated",
                                analytical_code=None,
                                eligible=False,
                                execution_code=elig_code)},
                        evidence={"route": "qm_m5_native",
                                  "provenance": prov},
                        source=source)
                    self.journal.log_note(
                        f"mt5 qm_m5 "
                        f"{'+' if touch['direction'] > 0 else '-'}1 "
                        f"touch {touch_ts} — NOT routed: "
                        f"{elig_reason}")
                    self._mark_instance_seen(
                        key, elig_code or "ineligible")
                    seen[key] = elig_code or "ineligible"
                    self._qm_m5_last_logged = max(
                        self._qm_m5_last_logged,
                        int(cand.signal_index))
                    continue
                self.journal.log_note(
                    f"mt5 qm_m5 "
                    f"{'+' if touch['direction'] > 0 else '-'}1 "
                    f"touch {touch['touch_ts']} "
                    f"QM {touch['qm_level']:.1f} head "
                    f"{touch['head']:.1f} neckline "
                    f"{touch['neckline']:.1f} — evaluating for a "
                    "live order on the M5 clock (funnel §2 "
                    "native route)"
                    + (f" tags:{','.join(touch['tags'])}"
                       if touch["tags"] else ""))
                _outcome_code = None
                try:
                    _outcome_code = self._execute_qm_m5_candidate(
                        pb, sig, m5_df, stats,
                        freshness_s=freshness_s,
                        last_bar_ts=last_bar_ts,
                        provenance=prov)
                except Exception as exc:
                    self.journal.log_note(
                        f"mt5: qm_m5 order path failed for the "
                        f"{touch['touch_ts']} touch: {exc}")
                    self._emit_eval(
                        playbook="qm_m5",
                        evaluation_clock="m5_close",
                        intended_tf="M5", actual_input_tf="M5",
                        last_bar_ts=last_bar_ts,
                        data_freshness_s=freshness_s,
                        outcome="exception",
                        reason_code="path.exception",
                        reason=f"{type(exc).__name__}: {exc}",
                        candidate=self._candidate_dict(sig, None,
                                                       prov),
                        source=source)
                    self._mark_instance_seen(key, "path.exception")
                    seen[key] = "path.exception"
                if _outcome_code == _FRICTION_CODE:
                    # RULING 6: a friction refusal does NOT
                    # consume the setup — the dedupe cursor holds
                    # BELOW this touch, so the next M5 close
                    # re-evaluates it (the arm-stays-alive pattern,
                    # applied to the native route's cursor).
                    self.journal.log_note(
                        f"mt5 qm_m5: friction refusal at "
                        f"{touch['touch_ts']} — setup stays alive, "
                        "will re-evaluate on the next M5 close")
                else:
                    self._qm_m5_last_logged = max(
                        self._qm_m5_last_logged,
                        int(cand.signal_index))
                    # Terminal disposition: this instance can
                    # never be routed again, in this process or
                    # any later one (durable seen set).
                    if key not in seen:
                        self._mark_instance_seen(
                            key, _outcome_code or "terminal")
                        seen[key] = _outcome_code or "terminal"
            if out:
                stats["qm_m5_signals"] = stats.get(
                    "qm_m5_signals", 0) + len(out)
            if not out:
                # Evaluated on the M5 close, no fresh touch — the
                # record that distinguishes "evaluated, nothing
                # there" from "never called" for this playbook.
                self._emit_eval(
                    playbook="qm_m5", evaluation_clock="m5_close",
                    intended_tf="M5", actual_input_tf="M5",
                    last_bar_ts=last_bar_ts,
                    data_freshness_s=freshness_s,
                    outcome="no_setup",
                    reason="evaluated on the M5 frame at M5 "
                           "close — no fresh QM touch",
                    source=source)
            return out
        except Exception as exc:
            self.journal.log_note(f"mt5: QM M5 scan failed: {exc}")
            return []

    def _order_outcome(self, stats: dict, pre: tuple, *,
                       holding: bool) -> tuple[str, str | None]:
        """Read the shared order path's outcome from observed
        counter effects (observation-only): (order_result,
        reason_code).  The order path journals its own refusal
        detail; this only classifies the outcome for the funnel
        record — filled / pending placed / order invalidated /
        refused at a named order-path gate (news, friday, entry
        gates, RiskGuards, margin)."""
        (executed0, pending0, invalidated0, gates0, legs0,
         sizing0) = pre
        ineligible0 = gates0.get("__ineligible__", {})
        # Phase 2 backstop refusals classify FIRST: an ineligible
        # candidate refused at the order boundary is a refusal
        # with its lifecycle code, never a phantom fill.
        for key, val in stats.items():
            if key.startswith("ineligible::") \
                    and val > ineligible0.get(key, 0):
                return "refused_in_order_path", key.split("::", 1)[1]
        if stats.get("executed", 0) > executed0:
            return ("layer_filled" if holding else "filled"), None
        if len(self._pending_orders) > pending0:
            return ("layer_pending" if holding else "placed_pending"), None
        if stats.get("invalidated", 0) > invalidated0:
            return "rejected", "order.invalid_geometry"
        # Ruling-6 friction refusal is checked before the generic
        # gate scan: its code is risk.friction (category 8), and
        # the M5 route keys its setup-alive reevaluation off it.
        if stats.get("gate_friction", 0) > gates0.get(
                "gate_friction", 0):
            return "refused_in_order_path", _FRICTION_CODE
        if stats.get("sizing_declined", 0) > sizing0:
            return "refused_in_order_path", "gate.solvency"
        for key, val in stats.items():
            if key.startswith("gate_") and val > gates0.get(key, 0):
                code = {"gate_news": "regime.news",
                        "gate_friday": "regime.session"}.get(
                            key, "gate.solvency")
                return "refused_in_order_path", code
        legs = self._blocked_leg_counts
        for leg in ("risk_guard", "margin", "margin_shield"):
            if legs.get(leg, 0) > legs0.get(leg, 0):
                return "refused_in_order_path", "gate.solvency"
        if holding:
            return "layer_no_order", None
        return "refused_in_order_path", None

    def _order_snapshot(self, stats: dict) -> tuple:
        """Counter snapshot for :meth:`_order_outcome`."""
        gates = {k: v for k, v in stats.items()
                 if k.startswith("gate_")}
        gates["__ineligible__"] = {
            k: v for k, v in stats.items()
            if k.startswith("ineligible::")}
        return (stats.get("executed", 0), len(self._pending_orders),
                stats.get("invalidated", 0), gates,
                dict(self._blocked_leg_counts),
                stats.get("sizing_declined", 0))

    def _execute_qm_m5_candidate(self, pb, sig, m5_df, stats,
                                 *, freshness_s: float,
                                 last_bar_ts,
                                 provenance: dict | None = None
                                 ) -> str | None:
        """One fresh qm_m5 touch through the JUDGMENT-shaped
        decision step (owner steering 2026-10-23 + directive §5C:
        "a neural trading engine", not a gate stack).

        The candidate is NOT routed through the roster's layered
        veto chain (arm -> touch -> confirm -> floor -> regime).
        Instead:

        1. EVIDENCE is assembled and weighed: the Asymmetry Score
           computed on the M5 candidate's own context, the QM
           pattern's reaction evidence (sweep_head = rejection of
           the raid at the head; displacement_break = acceptance
           by closes through the neckline), HTF context (structural
           D1/H4 bias, regime, posture, alignment), journey
           geometry, and friction — all recorded on the funnel
           record as observed facts vs inferred readings, with
           supporting AND contradictory observations named.  HTF
           conditions are CONTEXTUAL EVIDENCE on this route —
           weighed, never silent vetoes (gates-vs-scores law).
           The one weighed refusal is the score's own flip: the
           candidate's evidence, weighed, funds the opposite side.
        2. HARD STOPS remain only for trade-existence FACTS:
           invalid stop geometry, stale data, and the shared hard
           lines (the playbook's own fact re-verification, the
           measured cost/friction bound) — binary, because they
           are facts, not opinions.
        3. An accepted candidate is handed to the SHARED sizing /
           order machinery (_place_live_order / _try_layer) with
           its evidence attached in the signal context: RiskGuards,
           margin, one-position rule and in-trade management apply
           exactly as for every other candidate.  Risk is carried
           by position size and management, per doctrine — the
           route does not pre-kill the candidate for failing a
           confirmation checklist, and there is NO score floor on
           this route (the roster's floors are untouched; they
           simply are not this route's contract).

        Every outcome writes the full §7-shaped funnel record:
        evidence vector, hard-risk checks with exact vetoes,
        selected action + rejected alternatives, input timestamps
        — the judgment is reconstructible from its record, never
        a bare gate name and never a post-hoc story.

        Returns the order path's reason code (None on a clean
        fill/placement): the route keys the RULING 6 setup-alive
        rule off it — a ``risk.friction`` outcome holds the
        dedupe cursor so the touch is re-evaluated when friction
        normalizes; every other outcome consumes the touch.
        """
        direction = int(sig.direction)
        entry = float(sig.entry_ref)
        stop = float(sig.invalidation)
        atr_m5 = float(sig.atr or atr14(m5_df))
        alert_id = f"MT5-qm_m5-{sig.index}"
        lc = dict(getattr(sig, "live_context", None) or {})
        # Phase 2: this route only receives candidates the
        # lifecycle assessed as execution-eligible; stamp the
        # verdict on the signal so the shared order path's
        # backstop can see it (an explicit False anywhere
        # upstream refuses the order at the boundary).
        sig.execution_eligible = True
        _src = (provenance or {}).get("source")
        stop_dist = abs(entry - stop)
        tick = self._last_tick or {}
        spread_px = (float(tick["ask"]) - float(tick["bid"])) \
            if tick.get("ask") and tick.get("bid") else 0.0
        friction_ratio = (spread_px / stop_dist) \
            if stop_dist > 0 else None

        # HTF context: attached as evidence (the desk's current
        # structural read), NOT consulted as permission.
        setup = self._live_setup or {}
        verdict = {
            "bias_d1": int(setup.get("bias_d1", 0) or 0),
            "bias_h4": int(setup.get("bias_h4", 0) or 0),
            "h4_context": str(setup.get("h4_context", "") or ""),
            "regime": str(setup.get("regime", "trend") or "trend"),
            "direction": direction, "zone": None,
            "major_zone": None, "trigger": "qm_m5",
            "entry_clock": "M5", "amd_phase": "unknown",
            "qm": {"status": "qm_m5"}, "volume": {},
            "dxy_corr": (self._dxy.state.corr
                         if self._dxy is not None
                         and self._dxy.state is not None else None),
            "size_mult": float(setup.get("size_mult", 1.0) or 1.0),
        }
        score = self._score_for(
            direction=direction, verdict=verdict, price=entry,
            zone_d=None, atr=atr_m5, style_key="scalp",
            playbook="qm_m5")
        breakdown = dict(score.breakdown())
        # The score was computed FOR this candidate on its own
        # clock — the breakdown must not claim the H1 clock float.
        breakdown["frame_tf"] = "M5"
        breakdown["trigger_tf"] = "M5"
        breakdown["clock_reason"] = "qm_m5 native M5 route (§2)"
        if self._posture is not None:
            breakdown["posture"] = self._posture.name
        journey = self._journey_context(
            direction=direction, price=entry, stop_dist=stop_dist,
            zone_d=None, atr=atr_m5)
        post = self._posture
        tags = list(lc.get("tags") or [])
        supporting: list[str] = []
        contradictory: list[str] = []
        # Phase 2 (addendum C): CORRELATION GROUPING — every
        # evidence item also records its ORIGIN GROUP.  Items
        # from the same event (sweep_head and displacement_break
        # are two readings of THIS pattern's one reaction) share
        # a group and are counted ONCE as independent evidence:
        # correlated observations must not stack as independent
        # confirmations.  The flat lists stay (existing
        # consumers); the groups make the correlation structure
        # explicit on the record.
        _groups: dict[str, list[str]] = {}

        def _note(bucket: list, group: str, text: str) -> None:
            bucket.append(text)
            _groups.setdefault(group, []).append(text)

        if "sweep_head" in tags:
            _note(supporting, "qm_pattern_event",
                  "sweep_head: the head leg swept the prior swing "
                  "— rejection evidence at the extreme")
        if "displacement_break" in tags:
            _note(supporting, "qm_pattern_event",
                  "displacement_break: the neckline break displaced "
                  "— acceptance by closes through the level")
        bias_d1 = int(verdict["bias_d1"])
        if bias_d1 and bias_d1 == direction:
            _note(supporting, "htf_structure",
                  f"with structural D1 bias {bias_d1:+d}")
        elif bias_d1:
            _note(contradictory, "htf_structure",
                  f"against structural D1 bias {bias_d1:+d} "
                  "(weighed as evidence, not a veto)")
        if bool(score.flipped):
            _note(contradictory, "asymmetry_score",
                  "asymmetry score flipped: the weighed structure "
                  "funds the opposite side")
        if journey.get("refusal"):
            _note(contradictory, "journey_geometry",
                  f"journey geometry: {journey['refusal']}")
        _sess_bucket, _sess_kz = session_of(last_bar_ts)
        evidence = {
            "route": "qm_m5_native",
            "observed": {
                "tags": tags,
                "qm_level": lc.get("qm_level"),
                "head": lc.get("head"),
                "neckline": lc.get("neckline"),
                "target_r": lc.get("target_r"),
                "entry": entry, "stop": stop,
                "stop_dist": stop_dist,
                "spread_px": spread_px,
                "spread_over_stop": friction_ratio,
                "bias_d1": verdict["bias_d1"],
                "bias_h4": verdict["bias_h4"],
                # RULING 1: the session bucket is RECORDED for
                # later per-session evaluation — session_scope
                # stays "any" for v1; the bucket is evidence,
                # never a gate on this route.
                "session_bucket": _sess_bucket,
                "killzone": _sess_kz,
                "last_bar_ts": str(last_bar_ts),
                "freshness_s": freshness_s,
                "tick_time": (str(tick.get("time"))
                              if tick.get("time") is not None
                              else None),
            },
            "inferred": {
                "score_total": float(score.total),
                "score_components": breakdown,
                "score_flipped": bool(score.flipped),
                "regime": verdict["regime"],
                "posture": post.name if post is not None else None,
                "alignment": getattr(self._alignment,
                                     "alignment_label", None),
                "journey_state": journey.get("state"),
                "room_r": journey.get("room_r"),
            },
            "supporting": supporting,
            "contradictory": contradictory,
            # Correlation structure (addendum C): the groups and
            # the count of INDEPENDENT evidence groups — two
            # readings of one pattern event count once.
            "evidence_groups": _groups,
            "independent_evidence_groups": sorted(_groups),
            # Phase 2: the candidate's lifecycle provenance rides
            # the evidence too — formation/touch/creation times,
            # source, instance key, eligibility verdict.
            "provenance": provenance,
        }
        # COMPETING SCENARIOS (addendum C): both sides coexist on
        # the record until the evidence resolves them — the
        # candidate's side, and the opposite side whose funding
        # is exactly what a score flip reports.  Neither is
        # suppressed; the decision states which was selected and
        # why (the weighed evidence / the flip refusal).
        scenarios = [
            {"scenario": f"take the {direction:+d} QM touch at "
                         f"{entry:.2f}",
             "status": "selected"},
            {"scenario": f"take the {-direction:+d} side (the "
                         "flip side)",
             "status": ("funded by the weighed evidence — score "
                        "flipped" if bool(score.flipped) else
                        "not funded by the weighed evidence")},
        ]
        candidate = self._candidate_dict(sig, breakdown, provenance)
        targets = [float(x) for x in (sig.targets or [])]
        factors = list(getattr(sig, "factors", []) or [])

        def _refuse(code: str, reason: str,
                    hard_checks: list,
                    analytical_qualified: bool = False) -> None:
            # Competing scenarios on a refusal: this side was
            # NOT taken; on a flip the other side is what the
            # weighed evidence selected.
            _sc = [dict(s) for s in scenarios]
            _sc[0]["status"] = f"not taken — refused ({code})"
            if code == "direction.flip":
                _sc[1]["status"] = ("selected by the weighed "
                                    "evidence (score flip)")
            # Phase 2 (P0-2): the signal record's score is the
            # ASYMMETRY total — the computed decision score.  The
            # sleeve base (sig.score, a 60.0 placeholder) is
            # recorded under its own name in the candidate block,
            # never as the headline.
            log_signal(
                self.journal, signal_id=alert_id, playbook="qm_m5",
                direction=direction, score=float(score.total),
                decision="gated", reason=reason, bar_ts=last_bar_ts,
                entry=entry, stop=stop, targets=targets,
                factors=factors, score_breakdown=breakdown,
                trigger_source="qm_m5",
                context=self._decision_context(
                    {"route": "qm_m5_native", "evidence": evidence}))
            self._record_blocked(reason, "qm_m5")
            self._emit_eval(
                playbook="qm_m5", evaluation_clock="m5_close",
                intended_tf="M5", actual_input_tf="M5",
                last_bar_ts=last_bar_ts,
                data_freshness_s=freshness_s,
                outcome="candidate_emitted", reason_code=code,
                reason=reason, candidate=candidate,
                decision={"result": "refused", "reason_code": code,
                          "reason": reason,
                          "execution_eligible": False,
                          "order_result": "not_attempted",
                          "hard_risk_checks": hard_checks,
                          "reasoning_mode": "standalone",
                          "floor_authority":
                              _FLOOR_AUTHORITY["native:qm_m5"],
                          "competing_scenarios": _sc,
                          # Two layers, distinct (owner ruling):
                          # an analytical refusal says
                          # not_qualified with the strategy code;
                          # the decision-stage solvency refusal
                          # says QUALIFIED analytically and
                          # refused by the risk layer — the EC2
                          # burst's 23, now readable at a glance.
                          **_verdict_blocks(
                              analytical=(
                                  "qualified"
                                  if analytical_qualified
                                  else "not_qualified"),
                              analytical_code=(
                                  None if analytical_qualified
                                  else code),
                              eligible=False,
                              execution_code=(
                                  code if analytical_qualified
                                  else None))},
                evidence=evidence, source=_src)

        hard_checks: list[dict] = []
        # FACT 1 — stop geometry: a stop not beyond the entry in
        # the trade's direction is not a trade at all.
        geom_ok = stop_dist > 0 and (entry - stop) * direction > 0
        hard_checks.append({
            "check": "stop_geometry",
            "result": "pass" if geom_ok else "VETO",
            "detail": f"entry {entry:.2f} stop {stop:.2f} "
                      f"dist {stop_dist:.2f} direction "
                      f"{direction:+d}"})
        if not geom_ok:
            _refuse(
                "gate.location",
                f"invalid stop geometry: stop {stop:.2f} is not "
                f"beyond entry {entry:.2f} for a {direction:+d} "
                "trade — no valid stop, no trade",
                hard_checks)
            return
        # FACT 2 — data freshness: no trading on dead data.
        market_ok, _mkt_reason = self._market_open()
        stale = freshness_s > self._QM_M5_STALE_AFTER_S and market_ok
        hard_checks.append({
            "check": "data_freshness",
            "result": "VETO" if stale else "pass",
            "detail": f"newest M5 bar closed {freshness_s:.0f}s "
                      f"ago (bound {self._QM_M5_STALE_AFTER_S:.0f}s"
                      f"; market "
                      f"{'open' if market_ok else 'closed'})"})
        if stale:
            _refuse(
                "gate.data",
                f"stale M5 data: newest closed bar is "
                f"{freshness_s:.0f}s old (bound "
                f"{self._QM_M5_STALE_AFTER_S:.0f}s) — no trade "
                "on dead data",
                hard_checks)
            return
        # WEIGHED refusal — the score's own flip.  Not a checklist
        # item: the candidate's evidence, weighed, funds the
        # opposite side; the record carries the full vector so
        # the judgment is auditable as a judgment.
        if bool(score.flipped):
            _refuse(
                "direction.flip",
                "weighed judgment: the asymmetry evidence is "
                "flipped — structure funds the opposite side "
                "only; this candidate's own evidence contradicts "
                "it",
                hard_checks)
            return
        # FACT 3 — the shared hard lines: the playbook's own fact
        # re-verification (instrument universe, stop band, target
        # qualification), measured hunting-ground exclusions,
        # session scope (qm_m5 declares "any"), and the broker
        # cost/friction bound.  Facts, never readings.
        engine = self._get_engine()
        ectx = EngineContext(df=m5_df, states=[], years=0.0,
                             instrument=self.cfg.symbol)
        _hl = getattr(engine, "hard_line_check", None)
        if callable(_hl):
            ok, reason = _hl(pb, sig, ectx, spread_px)
        else:
            ok, reason = engine.gate_live_signal(
                pb, sig, ectx, hierarchy_verdict=verdict,
                spread_px=spread_px)
        hard_checks.append({
            "check": "hard_lines",
            "result": "pass" if ok else "VETO",
            "detail": reason or "session scope / setup facts / "
                                "hunting ground / cost gate"})
        if not ok:
            _refuse(_funnel_classify(reason), reason, hard_checks)
            return
        # Execute-list narrowing (owner config, A14 override):
        # evaluated + recorded, never ordered.
        if not self._executes("qm_m5"):
            _why = ("config execute_playbooks tuple excludes "
                    "qm_m5 (A14 override)")
            log_signal(
                self.journal, signal_id=alert_id, playbook="qm_m5",
                direction=direction, score=float(score.total),
                decision="shadow", reason=_why, bar_ts=last_bar_ts,
                entry=entry, stop=stop, targets=targets,
                factors=factors, score_breakdown=breakdown,
                trigger_source="qm_m5",
                context=self._decision_context(
                    {"route": "qm_m5_native", "evidence": evidence}))
            self._emit_eval(
                playbook="qm_m5", evaluation_clock="m5_close",
                intended_tf="M5", actual_input_tf="M5",
                last_bar_ts=last_bar_ts,
                data_freshness_s=freshness_s,
                outcome="candidate_emitted",
                reason_code="path.execute_list", reason=_why,
                candidate=candidate,
                decision={"result": "not_executable",
                          "reason_code": "path.execute_list",
                          "reason": _why,
                          "execution_eligible": False,
                          "order_result": "not_attempted",
                          "hard_risk_checks": hard_checks,
                          "reasoning_mode": "standalone",
                          "floor_authority":
                              _FLOOR_AUTHORITY["native:qm_m5"],
                          "competing_scenarios": [
                              {**scenarios[0],
                               "status": "not taken — execute-list "
                                         "narrowing (owner config)"},
                              scenarios[1]],
                          **_verdict_blocks(
                              analytical="qualified",
                              eligible=False,
                              execution_code="path.execute_list")},
                evidence=evidence, source=_src)
            return
        # Phase 2 (P1-1) DECISION-STAGE SOLVENCY: the same
        # RiskGuards accounting the order path enforces,
        # asked BEFORE dispatch — a candidate the guards would
        # refuse is refused here, at decision, with the guard's
        # exact reason (the EC2 burst's 23 "accepted"-then-
        # refused candidates are this check's reason to exist).
        # The order path re-checks everything at placement and
        # remains the authoritative backstop.
        _solv_ok, _solv_reason = self._decision_solvency(
            pb, sig, score)
        hard_checks.append({
            "check": "decision_solvency (RiskGuards, same "
                     "accounting as the order path)",
            "result": "pass" if _solv_ok else "VETO",
            "detail": _solv_reason or "solvent under the "
                                      "effective sizing mode"})
        if not _solv_ok:
            _refuse("gate.solvency",
                    f"decision-stage solvency: {_solv_reason}",
                    hard_checks, analytical_qualified=True)
            return "gate.solvency"
        # ACCEPT — the weighed judgment, stated with its evidence,
        # then the shared order machinery carries the risk.
        judgment = (
            f"weighed judgment: score {score.total:.0f} on M5 "
            f"evidence (supporting: "
            f"{'; '.join(supporting) if supporting else 'none special'}"
            + (f"; contradictory: {'; '.join(contradictory)}"
               if contradictory else "")
            + ") — hard facts pass; handed to the shared "
              "sizing/management path with evidence attached")
        self.journal.log_note(
            f"mt5 qm_m5 {'+' if direction > 0 else '-'}1 {judgment}")
        log_signal(
            self.journal, signal_id=alert_id, playbook="qm_m5",
            direction=direction, score=float(score.total),
            decision="accepted", reason=judgment,
            bar_ts=last_bar_ts, entry=entry, stop=stop,
            targets=targets, factors=factors,
            score_breakdown=breakdown, trigger_source="qm_m5",
            context=self._decision_context(
                {"route": "qm_m5_native", "evidence": evidence}))
        self._last_scores["qm_m5"] = breakdown
        self._last_score = breakdown
        sig_ctx = {
            "playbook": "qm_m5", "style_key": "scalp",
            "style_bucket": "MICRO_SCALP",
            "family": _posture_mod.family_of(
                "qm_m5", str(getattr(pb, "regime_scope", "") or "")),
            # No score floor on the judgment route (the roster's
            # floors are untouched — they are not this route's
            # contract).  0.0 keeps ctx["floor"] numeric for the
            # management consumers that read it.
            "floor": 0.0, "floor_path": "QM_M5_NATIVE (no floor)",
            "path": "QM_M5_NATIVE", "route": "qm_m5_native",
            "entry_total": float(score.total), "zone_d": None,
            "clock_tf": "M5", "defended_stop": stop,
            "evidence": evidence,
        }
        pre = self._order_snapshot(stats)
        holding = bool(self.tick_manager.tickets())
        if holding:
            self._try_layer(pb, sig, stats, verdict=verdict,
                            entry_bar=None, score=score,
                            funding=None, zone_d=None)
        else:
            self._place_live_order(
                pb, sig, stats, verdict=verdict, entry_bar=None,
                score=score, zone_d=None, funding=None,
                signal_ctx=sig_ctx)
        _ores, _ocode = self._order_outcome(stats, pre,
                                            holding=holding)
        hard_checks.append({
            "check": "shared_order_path (risk guards / margin / "
                     "one-position / ruling-6 friction gate)",
            "result": ("pass" if _ores in (
                "filled", "placed_pending", "layer_filled",
                "layer_pending") else "refused (see order_result)"),
            "detail": f"shared order path outcome: {_ores}"
                      + (f" ({_ocode})" if _ocode else "")})
        _rec = self._emit_eval(
            playbook="qm_m5", evaluation_clock="m5_close",
            intended_tf="M5", actual_input_tf="M5",
            last_bar_ts=last_bar_ts, data_freshness_s=freshness_s,
            outcome="candidate_emitted",
            reason=judgment + f"; order outcome: {_ores}",
            candidate=candidate,
            decision={
                "result": "accepted", "reason_code": _ocode,
                "reason": judgment,
                "execution_eligible": True, "order_result": _ores,
                "hard_risk_checks": hard_checks,
                "reasoning_mode": "standalone",
                "floor_authority": _FLOOR_AUTHORITY["native:qm_m5"],
                "competing_scenarios": scenarios,
                **_verdict_blocks(
                    analytical="qualified",
                    eligible=_ores in (
                        "filled", "placed_pending", "layer_filled",
                        "layer_pending", "layer_no_order"),
                    execution_code=_ocode, outcome=_ores),
                "selected_action": "submit to the shared order "
                                   "path (order type by price vs "
                                   "the QM level, inside "
                                   "_place_live_order)",
                "rejected_alternatives": [
                    "wait for a generic H1 trigger — rejected: "
                    "the M5 route is independently clocked by "
                    "contract (directive §4); the H1 trigger is "
                    "not this playbook's entry condition",
                    "abstain on HTF disagreement — rejected: HTF "
                    "context is weighed evidence on this route, "
                    "not a veto (gates-vs-scores law)",
                ]},
            evidence=evidence, source=_src)
        self._announce_assessment(_rec)
        return _ocode

    def _run_qm_d1(self, d1_df, stats: dict | None = None, *,
                   source: str = _lifecycle.SOURCE_LIVE_CLOSE
                   ) -> list[dict]:
        """qm_d1 REAL order path on its own D1 clock (P3 — closes
        the P1-flagged exception; A14: the demo IS the shadow).

        Runs on the real D1 frame at warm start and on every D1
        close (see :meth:`_refresh_bars`) — NOT through the
        H1-clocked signal path, which would run D1 logic on H1
        bars.  Each fresh touch (deduped by ``_qm_d1_last_logged``)
        goes through the desk's ordinary machinery denominated in
        D1: Asymmetry Score on the D1 frame (clock D1/D1/D1) ->
        hard lines -> MACRO_SWING_OPEN floor (Path B) -> posture
        -> journey geometry -> order or layer through
        ``_place_live_order`` with the signal's own D1 stop
        (beyond the QM head).  The Golden Swing Confirm ceremony
        is NOT used: it validates against the H1 live setup, which
        a D1-clock signal cannot satisfy — the order still passes
        RiskGuards, margin, news and posture.  The playbook's
        batch-engine flag is ``live=True`` since the owner directive
        of 2026-10-22 (the batch book executes too; the MT5 demo
        book always did, exactly like qm_m5).  Kill switch:
        ``/set playbook.qm_d1 off``.
        """
        if "qm_d1" in (self.cfg.disabled_playbooks or ()):
            self._emit_eval(
                playbook="qm_d1", evaluation_clock="d1_close",
                intended_tf="D1",
                last_bar_ts=(d1_df.index[-1]
                             if d1_df is not None and len(d1_df)
                             else None),
                outcome="not_eligible",
                reason_code="path.kill_switch",
                reason="disabled via /set kill switch — "
                       "not evaluated")
            return []
        try:
            if d1_df is None or len(d1_df) < 30:
                self._emit_eval(
                    playbook="qm_d1", evaluation_clock="d1_close",
                    intended_tf="D1", outcome="not_eligible",
                    reason_code="gate.data",
                    reason="D1 frame unavailable or shorter than "
                           "30 bars — not evaluated")
                return []
            # Step 4: native-clock pending guards run first, on
            # every D1 close, before new candidates are evaluated.
            self._check_native_pendings("D1", stats)
            pb = QMD1Playbook()
            ctx = EngineContext(df=d1_df, states=[], years=0.0,
                                instrument=self.cfg.symbol)
            out: list[dict] = []
            seen = self._load_seen_instances()
            last_index = len(d1_df) - 1
            # Freshness against the tick clock, same basis as the
            # M5 route (recorded truthfully; the D1 route has no
            # stale-data veto of its own — candidate AGE is the
            # lifecycle gate below).
            _now_ref = None
            if self._last_tick and \
                    self._last_tick.get("time") is not None:
                try:
                    _now_ref = pd.Timestamp(self._last_tick["time"])
                except Exception:
                    _now_ref = None
            if _now_ref is None:
                _now_ref = pd.Timestamp.now(tz="UTC")
            if _now_ref.tzinfo is None:
                _now_ref = _now_ref.tz_localize("UTC")
            _freshness_s = max(
                0.0, (_now_ref - (d1_df.index[-1]
                                  + pd.Timedelta(days=1))
                      ).total_seconds())
            for cand in pb.candidates(ctx):
                if cand.signal_index <= self._qm_d1_last_logged:
                    continue  # already evaluated on an earlier scan
                self._qm_d1_last_logged = max(
                    self._qm_d1_last_logged, int(cand.signal_index))
                sig = cand.outcome.signal
                # Step 15: stamp the declared execution intent
                # (QM = limit thesis: entry at the QM level).
                sig.execution_intent = execution_intent_for(
                    "qm_d1", sig)
                lc = dict(getattr(sig, "live_context", None) or {})
                touch = {"direction": int(cand.direction),
                         "touch_ts": d1_df.index[cand.signal_index],
                         "touch_index": int(cand.signal_index),
                         "qm_level": lc.get("qm_level"),
                         "head": lc.get("head"),
                         "neckline": lc.get("neckline"),
                         "tags": lc.get("tags", []),
                         "r_net": float(cand.outcome.r_net),
                         "exit_reason": cand.outcome.exit_reason}
                out.append(touch)
                # Phase 2 lifecycle (P0-1): provenance +
                # eligibility + idempotency, same contract as
                # the M5 route — warmup/catch-up/replay D1
                # candidates are analytical only.
                touch_ts = touch["touch_ts"]
                touch_age = last_index - int(cand.signal_index)
                _p0 = lc.get("p0_index")
                formation_ts = (
                    d1_df.index[int(_p0)]
                    if _p0 is not None
                    and 0 <= int(_p0) < len(d1_df) else None)
                _level = lc.get("qm_level")
                key = _lifecycle.instance_key(
                    "qm_d1", pattern_ts=touch_ts,
                    direction=int(cand.direction),
                    level=(float(_level) if _level is not None
                           else float(sig.entry_ref)))
                if key in seen:
                    continue  # terminally disposed in an earlier
                    # run — its record lives in that journal.
                eligible, elig_code, elig_reason = \
                    _lifecycle.assess_eligibility(
                        source=source, touch_age_bars=touch_age,
                        validity_window_bars=(
                            _lifecycle.QM_VALIDITY_WINDOW_BARS))
                prov = _lifecycle.provenance_block(
                    key=key, formation_ts=formation_ts,
                    touch_ts=touch_ts, created_ts=_now_ref,
                    evaluated_bar_ts=d1_df.index[-1],
                    evaluation_ts=_now_ref, source=source,
                    touch_age_bars=touch_age,
                    validity_window_bars=(
                        _lifecycle.QM_VALIDITY_WINDOW_BARS),
                    eligible=eligible, eligibility_code=elig_code)
                if not eligible:
                    _warm = (elig_code ==
                             _lifecycle.CODE_WARMUP_INELIGIBLE)
                    self._emit_eval(
                        playbook="qm_d1",
                        evaluation_clock="d1_close",
                        intended_tf="D1", actual_input_tf="D1",
                        last_bar_ts=touch_ts,
                        data_freshness_s=_freshness_s,
                        outcome="candidate_emitted",
                        reason_code=elig_code, reason=elig_reason,
                        candidate=self._candidate_dict(sig, None,
                                                       prov),
                        decision={
                            "result": ("not_executable" if _warm
                                       else "refused"),
                            "reason_code": elig_code,
                            "reason": elig_reason,
                            "execution_eligible": False,
                            "order_result": "not_attempted",
                            **_verdict_blocks(
                                analytical="not_evaluated",
                                eligible=False,
                                execution_code=elig_code)},
                        evidence={"route": "qm_d1_native",
                                  "provenance": prov},
                        source=source)
                    self.journal.log_note(
                        f"mt5 qm_d1 "
                        f"{'+' if touch['direction'] > 0 else '-'}1 "
                        f"touch {touch_ts.date()} — NOT routed: "
                        f"{elig_reason}")
                    self._mark_instance_seen(
                        key, elig_code or "ineligible")
                    seen[key] = elig_code or "ineligible"
                    continue
                self.journal.log_note(
                    f"mt5 qm_d1 "
                    f"{'+' if touch['direction'] > 0 else '-'}1 "
                    f"touch {touch['touch_ts'].date()} "
                    f"QM {touch['qm_level']:.1f} head "
                    f"{touch['head']:.1f} neckline "
                    f"{touch['neckline']:.1f} — evaluating for a "
                    "live order on the D1 clock (P3/A14)"
                    + (f" tags:{','.join(touch['tags'])}"
                       if touch["tags"] else ""))
                try:
                    self._execute_qm_d1_candidate(
                        pb, sig, d1_df, ctx, stats, provenance=prov)
                except Exception as exc:
                    self.journal.log_note(
                        f"mt5: qm_d1 order path failed for the "
                        f"{touch['touch_ts'].date()} touch: {exc}")
                    # Phase 2: the failure is a funnel record,
                    # not only a note (the M5 route already did
                    # this; D1 silently dropped it).
                    self._emit_eval(
                        playbook="qm_d1",
                        evaluation_clock="d1_close",
                        intended_tf="D1", actual_input_tf="D1",
                        last_bar_ts=touch_ts,
                        data_freshness_s=_freshness_s,
                        outcome="exception",
                        reason_code="path.exception",
                        reason=f"{type(exc).__name__}: {exc}",
                        candidate=self._candidate_dict(sig, None,
                                                       prov),
                        source=source)
                self._mark_instance_seen(key, "terminal")
                seen[key] = "terminal"
            if out and stats is not None:
                stats["qm_d1_signals"] = stats.get(
                    "qm_d1_signals", 0) + len(out)
            if not out:
                # FUNNEL §1.1: evaluated on the D1 close, no fresh
                # touch — a record, not a silence.
                self._emit_eval(
                    playbook="qm_d1", evaluation_clock="d1_close",
                    intended_tf="D1", actual_input_tf="D1",
                    last_bar_ts=d1_df.index[-1],
                    outcome="no_setup",
                    reason="evaluated on the D1 frame at D1 "
                           "close — no fresh QM touch",
                    source=source)
            return out
        except Exception as exc:
            self.journal.log_note(
                f"mt5: QM D1 scan failed: {exc}")
            return []

    def _execute_qm_d1_candidate(self, pb, sig, d1_df, ctx,
                                 stats, provenance: dict | None = None
                                 ) -> None:
        """One fresh qm_d1 touch through the desk machinery."""
        stats = stats if stats is not None else {}
        engine = self._get_engine()
        # Phase 2: eligibility stamp for the order-path backstop
        # (this route only receives eligible candidates).
        sig.execution_eligible = True
        _src = (provenance or {}).get("source")
        direction = int(sig.direction)
        entry = float(sig.entry_ref)
        stop = float(sig.invalidation)
        atr_d1 = float(sig.atr or atr14(d1_df))
        alert_id = f"MT5-qm_d1-{sig.index}"
        verdict = {
            "bias_d1": direction, "bias_h4": 0, "regime": "trend",
            "direction": direction, "zone": None,
            "major_zone": None, "trigger": "qm_d1",
            "entry_clock": "D1", "amd_phase": "unknown",
            "qm": {"status": "qm_d1"}, "volume": {},
            "dxy_corr": (self._dxy.state.corr
                         if self._dxy is not None
                         and self._dxy.state is not None else None),
            "size_mult": 1.0,
        }
        score = self._score_for(
            direction=direction, verdict=verdict, price=entry,
            zone_d=None, atr=atr_d1, style_key="qm_d1",
            playbook="qm_d1")
        breakdown = dict(score.breakdown())
        if self._last_clock is not None:
            breakdown["frame_tf"] = self._last_clock.score_tf
            breakdown["trigger_tf"] = self._last_clock.trigger_tf
            breakdown["clock_reason"] = self._last_clock.reason
        breakdown["posture"] = (self._posture.name
                                if self._posture else "open")
        targets = [float(x) for x in (sig.targets or [])]
        factors = list(getattr(sig, "factors", []) or [])
        bar_ts = d1_df.index[sig.index] \
            if sig.index < len(d1_df) else None

        def _journal(decision: str, reason: str,
                     funnel_code: str | None = None,
                     analytical_qualified: bool = False) -> None:
            # Phase 2 (P0-2): signal-record score = the Asymmetry
            # total (the computed decision score); the sleeve
            # base rides under its own name in the candidate
            # block — see _candidate_dict.
            log_signal(
                self.journal, signal_id=alert_id, playbook="qm_d1",
                direction=direction, score=float(score.total),
                decision=decision, reason=reason, bar_ts=bar_ts,
                entry=entry, stop=stop, targets=targets,
                factors=factors, score_breakdown=breakdown,
                trigger_source="qm_d1",
                context=self._decision_context())
            # FUNNEL §1.1: the refusal half of the funnel record
            # (the accepted record is emitted after the order
            # dispatch below, when the order outcome is known).
            if funnel_code is not None:
                self._emit_eval(
                    playbook="qm_d1", evaluation_clock="d1_close",
                    intended_tf="D1", actual_input_tf="D1",
                    last_bar_ts=bar_ts, outcome="candidate_emitted",
                    reason_code=funnel_code, reason=reason,
                    candidate=self._candidate_dict(
                        sig, breakdown, provenance),
                    decision={"result": "refused",
                              "reason_code": funnel_code,
                              "reason": reason,
                              "execution_eligible": False,
                              "order_result": "not_attempted",
                              "reasoning_mode": "standalone",
                              "floor_authority":
                                  _FLOOR_AUTHORITY["native:qm_d1"],
                              **_verdict_blocks(
                                  analytical=(
                                      "qualified"
                                      if analytical_qualified
                                      else "not_qualified"),
                                  analytical_code=(
                                      None if analytical_qualified
                                      else funnel_code),
                                  eligible=False,
                                  execution_code=(
                                      funnel_code
                                      if analytical_qualified
                                      else None))},
                    evidence={"route": "qm_d1_native",
                              "provenance": provenance},
                    source=_src)

        # Hard lines (spread / hunting ground / session / the
        # playbook's own live check on the D1 frame).
        tick = self._last_tick or {}
        spread_px = (float(tick["ask"]) - float(tick["bid"])) \
            if tick.get("ask") and tick.get("bid") else 0.0
        _hl = getattr(engine, "hard_line_check", None)
        if callable(_hl):
            ok, reason = _hl(pb, sig, ctx, spread_px)
        else:
            ok, reason = engine.gate_live_signal(
                pb, sig, ctx, hierarchy_verdict=verdict,
                spread_px=spread_px)
        if not ok:
            stats["hard_blocked"] = stats.get("hard_blocked", 0) + 1
            self._record_blocked(reason, "qm_d1")
            _journal("gated", reason, _funnel_classify(reason))
            return
        # Posture (block / shelve) — qm_d1 is the 'zone' family.
        post = self._posture
        if post is not None:
            why = None
            if post.block_entries:
                why = f"posture_block: {post.name} — {post.reason}"
            else:
                shed = post.shelved_reason(
                    "zone", direction,
                    self._bedrock.direction
                    if self._bedrock is not None else 0)
                if shed:
                    why = (f"posture_shelved: {shed} "
                           f"({post.name})")
            if why:
                stats["gated"] = stats.get("gated", 0) + 1
                self._record_blocked(why, "qm_d1")
                _journal("gated", why, "regime.scope")
                return
        # Journey geometry in open space (D1-denominated).
        journey = self._journey_context(
            direction=direction, price=entry,
            stop_dist=abs(entry - stop), zone_d=None, atr=atr_d1)
        if journey.get("refusal"):
            stats["gated"] = stats.get("gated", 0) + 1
            self._record_blocked(journey["refusal"], "qm_d1")
            _journal("gated", journey["refusal"], "gate.location")
            return
        # MACRO_SWING_OPEN floor (Path B).
        fl = floor_for("MACRO_SWING_OPEN")
        if not self._floor_ok(fl, score):
            stats["below_floor"] = stats.get("below_floor", 0) + 1
            _journal(
                "below_floor",
                f"asymmetry_floor: total {score.total:.1f} < "
                f"{fl['floor']:.0f} or component minimum unmet "
                f"(location {score.component('location'):.1f} "
                f">= {fl.get('min_location', 0):.0f} required on "
                "MACRO_SWING_OPEN)",
                "floor.total" if float(score.total)
                < float(fl["floor"]) else "floor.component")
            return
        # Phase 2 (P1-1) DECISION-STAGE SOLVENCY: same
        # RiskGuards accounting as the order path, asked before
        # dispatch (see _decision_solvency).  The order path
        # remains the authoritative backstop.
        _solv_ok, _solv_reason = self._decision_solvency(
            pb, sig, score)
        if not _solv_ok:
            stats["gated"] = stats.get("gated", 0) + 1
            self._record_blocked(
                f"decision-stage solvency: {_solv_reason}", "qm_d1")
            _journal("gated",
                     f"decision-stage solvency: {_solv_reason}",
                     "gate.solvency", analytical_qualified=True)
            return
        _journal("accepted",
                 f"asymmetry {score.total:.0f} clears "
                 "MACRO_SWING_OPEN (Path B, D1 clock)")
        sig_ctx = {
            "playbook": "qm_d1", "style_key": "qm_d1",
            "style_bucket": "MACRO_SWING", "family": "zone",
            "floor": float(fl["floor"]), "path": "MACRO_SWING_OPEN",
            "entry_total": float(score.total), "zone_d": None,
            "clock_tf": "D1",
            "defended_stop": stop,
        }
        # FUNNEL §1.1: order-outcome snapshot (observation-only).
        _pre_order = (stats.get("executed", 0),
                      len(self._pending_orders),
                      stats.get("invalidated", 0),
                      stats.get("gate_friction", 0),
                      stats.get("sizing_declined", 0))
        _holding = bool(self.tick_manager.tickets())
        if _holding:
            self._try_layer(pb, sig, stats, verdict=verdict,
                            entry_bar=None, score=score,
                            funding=None, zone_d=None)
        else:
            self._place_live_order(
                pb, sig, stats, verdict=verdict, entry_bar=None,
                score=score, zone_d=None, funding=None,
                signal_ctx=sig_ctx)
        if stats.get("executed", 0) > _pre_order[0]:
            _ores = "filled"
        elif len(self._pending_orders) > _pre_order[1]:
            _ores = "placed_pending"
        elif stats.get("invalidated", 0) > _pre_order[2]:
            _ores = "rejected"
        elif _holding:
            _ores = "layer_no_order"
        else:
            _ores = "refused_in_order_path"
        _rec = self._emit_eval(
            playbook="qm_d1", evaluation_clock="d1_close",
            intended_tf="D1", actual_input_tf="D1",
            last_bar_ts=bar_ts, outcome="candidate_emitted",
            reason=f"accepted on MACRO_SWING_OPEN; order "
                   f"outcome: {_ores}",
            candidate=self._candidate_dict(sig, breakdown,
                                           provenance),
            decision={"result": "accepted",
                      "reason_code": (
                          _FRICTION_CODE
                          if _ores == "refused_in_order_path"
                          and stats.get("gate_friction", 0)
                          > _pre_order[3]
                          else ("gate.solvency"
                                if _ores == "refused_in_order_path"
                                and stats.get("sizing_declined", 0)
                                > _pre_order[4] else None)),
                      "reason": f"order outcome: {_ores}",
                      "floor_result": {
                          "floor": float(fl["floor"]),
                          "total": float(score.total),
                          "path": "MACRO_SWING_OPEN"},
                      "reasoning_mode": "standalone",
                      "floor_authority":
                          _FLOOR_AUTHORITY["native:qm_d1"],
                      **_verdict_blocks(
                          analytical="qualified",
                          eligible=_ores in (
                              "filled", "placed_pending",
                              "layer_no_order"),
                          execution_code=(
                              _FRICTION_CODE
                              if _ores == "refused_in_order_path"
                              and stats.get("gate_friction", 0)
                              > _pre_order[3]
                              else ("gate.solvency"
                                    if _ores
                                    == "refused_in_order_path"
                                    and stats.get(
                                        "sizing_declined", 0)
                                    > _pre_order[4] else None)),
                          outcome=_ores),
                      "execution_eligible": True,
                      "order_result": _ores},
            evidence={"route": "qm_d1_native",
                      "provenance": provenance},
            source=_src)
        self._announce_assessment(_rec)

    # --------------------------- scheduled evaluations (P0-3)
    # Phase 2: EVERY registered playbook evaluates on its agreed
    # clock and journals the evaluation — the EC2 journal held 49
    # records, all qm_m5, because the 12 pure-H1 roster playbooks
    # only ever evaluated inside the trigger lifecycle and the
    # range family had no scheduled route at all.  These sweeps
    # are EVALUATIONS (scans): they journal no_setup / candidate
    # (observed) / not_eligible / exception records on schedule,
    # independent of the generic H1 trigger.  Order authority is
    # unchanged: the trigger path for the roster, the native
    # routes for qm_m5/qm_d1, and NO authority for the §G range
    # evaluation routes (shadow until the owner promotes them).

    def _desk_context_evidence(self, *,
                               last_bar_ts=None) -> dict:
        """The desk's current causal context as assessment
        evidence (burn-down #1): the structure / HTF snapshot
        the native routes stamp on their records, read from the
        SAME live state — the refreshed setup dict (D1/H4 bias,
        regime), the composed posture, the alignment read, the
        session bucket of the evaluated bar — for evaluation
        routes that weigh no candidate of their own (scheduled
        scans, kill-switch skips, the trigger path's
        evaluated-and-empty fallback), so their records state
        what structure was present instead of "not produced".
        Observation-only: values are copied onto the record,
        never consulted by any decision.  None-safe throughout:
        an unpopulated snapshot yields an empty dict and the
        record keeps its honest absent markers."""
        setup = self._live_setup or {}
        observed: dict = {}
        for key in ("bias_d1", "bias_h4"):
            if setup.get(key) is not None:
                observed[key] = setup.get(key)
        if last_bar_ts is not None:
            try:
                observed["session_bucket"] = \
                    session_of(last_bar_ts)[0]
            except Exception:
                pass
        inferred: dict = {}
        if setup.get("regime") is not None:
            inferred["regime"] = setup.get("regime")
        post = self._posture
        if post is not None and getattr(post, "name", None):
            inferred["posture"] = post.name
        alignment = getattr(self._alignment,
                            "alignment_label", None)
        if alignment is not None:
            inferred["alignment"] = alignment
        out: dict = {}
        if observed:
            out["observed"] = observed
        if inferred:
            out["inferred"] = inferred
        return out

    def _sweep_blocked_reason(self) -> tuple[str, str] | None:
        """Path-level blocks that make a scheduled scan not an
        evaluation at all (kill states are per-playbook and are
        checked by the caller)."""
        if getattr(self.cfg, "paused", False):
            return ("path.paused",
                    "engine paused via /set — scheduled scan "
                    "not evaluated")
        try:
            open_ok, open_reason = self._market_open()
        except Exception:
            open_ok, open_reason = True, ""
        if not open_ok:
            return ("path.market_closed",
                    f"market closed ({open_reason}) — scheduled "
                    "scan not evaluated")
        return None

    def _sweep_eval_one(self, spec, frame, *, clock: str,
                        intended_tf: str, actual_tf: str,
                        source: str, shadow: bool,
                        extra_evidence: dict | None = None
                        ) -> None:
        """One scheduled-scan evaluation for one playbook.

        A FRESH playbook instance scans only the closing bar:
        the roster instances carry streaming cursors owned by the
        trigger lifecycle, and letting a scan advance them would
        eat the trigger path's signals.  Signals found are
        journaled as OBSERVED candidates — evaluated, truthfully
        ineligible for execution from this route (the trigger
        path / native route remains the order authority; the §G
        range routes are shadow)."""
        name = spec.name
        last_bar_ts = (frame.index[-1]
                       if frame is not None and len(frame) else None)
        # Journal auditability (owner, 2026-10-26): the sweep
        # used to emit records with no freshness measurement at
        # all, so /assess could not demonstrate feed age for any
        # roster/range record.  Freshness here is the QM routes'
        # basis — seconds from the last closed bar's close to
        # the evaluation clock — computed once per scan and
        # carried on every record the scan emits, evaluated or
        # not (a feed-age measurement is not a structure read,
        # so it does not break the not-evaluated absent-marking
        # rule below).
        _eval_now = self._evaluation_now()
        _fresh_s = _journal_audit.freshness_seconds(
            _eval_now, last_bar_ts, intended_tf)
        base = dict(playbook=name, evaluation_clock=clock,
                    intended_tf=intended_tf, actual_input_tf=actual_tf,
                    last_bar_ts=last_bar_ts, source=source,
                    data_freshness_s=_fresh_s)
        # Burn-down #1: every EVALUATED scheduled-scan record
        # (no-setup and candidate outcomes below) carries the
        # desk's structure/HTF snapshot, so /assess shows what
        # the market looked like when this playbook was scanned,
        # not "not produced".  The not-evaluated exits above
        # (kill switch / paused / market closed / no frame)
        # deliberately keep NO evidence block — an evaluation
        # that never ran must not wear a structure read
        # (Amendment 3 absent-marking, test-pinned).
        context = self._desk_context_evidence(
            last_bar_ts=last_bar_ts)
        scan_evidence = {"evaluation_kind": "scheduled_scan",
                         "route": spec.route,
                         "shadow": bool(shadow), **context}
        if name in (self.cfg.disabled_playbooks or ()):
            self._emit_eval(
                **base, outcome="not_eligible",
                reason_code="path.kill_switch",
                reason="disabled via /set kill switch — scheduled "
                       "scan not evaluated")
            return
        blocked = self._sweep_blocked_reason()
        if blocked is not None:
            self._emit_eval(**base, outcome="not_eligible",
                            reason_code=blocked[0],
                            reason=blocked[1])
            return
        if frame is None or len(frame) < 30:
            self._emit_eval(
                **base, outcome="not_eligible",
                reason_code="gate.data",
                reason=f"{intended_tf} frame unavailable or "
                       "shorter than 30 bars — scheduled scan "
                       "not evaluated")
            return
        t = len(frame) - 1
        try:
            pb = spec.playbook_class()
            if hasattr(pb, "set_h4_frame"):
                pb.set_h4_frame(
                    self._stack.frames.get("H4")
                    if self._stack is not None else None)
            sigs = pb.live_signal([], t, frame) or []
            fresh = [s for s in sigs
                     if int(getattr(s, "index", -1)) == t]
        except Exception as exc:
            self._emit_eval(
                **base, outcome="exception",
                reason_code="path.exception",
                reason=f"scheduled scan failed: "
                       f"{type(exc).__name__}: {exc}")
            return
        evidence = dict(scan_evidence)
        if extra_evidence:
            evidence.update(extra_evidence)
        if not fresh:
            # No scenario formed: the record says what structure
            # was present (the context snapshot above) AND why
            # no scenario qualified — the detector emitted no
            # signal on the closing bar (burn-down #1).
            self._emit_eval(
                **base, outcome="no_setup",
                reason="scheduled scan on the closing bar — no "
                       "signal", evidence=evidence,
                assessment={"competing_scenarios": [
                    {"scenario": f"a {name} candidate at the "
                                 "closing bar",
                     "status": "not formed — the playbook's "
                               "detector emitted no signal on "
                               "the closing bar; no scenario "
                               "qualified"}]})
            return
        code = ("path.shadow_route" if shadow
                else "path.trigger_not_fired")
        why = ("§G shadow evaluation route: the candidate is "
               "journaled for grading and holds NO order "
               "authority in v1 (owner §G ruling)"
               if shadow else
               "scheduled scan found a candidate on the closing "
               "bar; this scan route holds no order authority "
               "and does not run the h1_trigger streaming "
               "evaluation — recorded as observed, never routed "
               "from the scan (see Trigger decision)")
        for sig in fresh:
            # Journal auditability (owner, 2026-10-26): the old
            # provenance stamped the BAR time as created AND
            # evaluated, so /assess printed the bar time as the
            # evaluation time.  Creation/evaluation are the scan
            # clock (_eval_now); formation is the pattern's own
            # bar when the detector records it (inversion-style
            # live_context indices mapped through the frame).
            _lc = getattr(sig, "live_context", None) or {}
            _formation_ts = None
            if isinstance(_lc, dict):
                for _fk in ("zone_formed_idx", "formed_idx",
                            "formation_idx"):
                    _fi = _lc.get(_fk)
                    if isinstance(_fi, int) and not isinstance(
                            _fi, bool) and 0 <= _fi < len(frame):
                        _formation_ts = frame.index[_fi]
                        break
            prov = _lifecycle.provenance_block(
                key=_lifecycle.instance_key(
                    name, pattern_ts=frame.index[t],
                    direction=int(sig.direction),
                    level=float(sig.entry_ref)),
                formation_ts=_formation_ts, touch_ts=frame.index[t],
                created_ts=_eval_now,
                evaluated_bar_ts=last_bar_ts,
                evaluation_ts=_eval_now, source=source,
                touch_age_bars=0, validity_window_bars=0,
                eligible=False, eligibility_code=code)
            # Burn-down #1: an observed candidate carries the
            # same assessment objects the native routes produce
            # — itemized evidence with correlation groups (the
            # detector's own factors; the candidate's stance vs
            # the structural D1 bias, weighed as context) and
            # the two competing scenarios (this side observed,
            # the other side unfunded by this bar).
            _dir = int(sig.direction)
            _entry = float(sig.entry_ref)
            _stop = float(sig.invalidation)
            _sup: list[str] = []
            _con: list[str] = []
            _groups: dict[str, list[str]] = {}

            def _note(bucket, group, text):
                bucket.append(text)
                _groups.setdefault(group, []).append(text)

            for _f in (getattr(sig, "factors", None) or []):
                _note(_sup, f"{name}_signal",
                      f"signal factor: {_f}")
            _d1 = (self._live_setup or {}).get("bias_d1")
            if _d1:
                _suffix = (" (context only — §G: D1 never "
                           "vetoes)" if shadow else "")
                if int(_d1) == _dir:
                    _note(_sup, "htf_structure",
                          f"with structural D1 bias "
                          f"{int(_d1):+d}{_suffix}")
                else:
                    _note(_con, "htf_structure",
                          f"against structural D1 bias "
                          f"{int(_d1):+d}{_suffix}")
            _obs = dict(evidence.get("observed") or {})
            _obs.update({"entry": _entry, "stop": _stop,
                         "stop_dist": abs(_entry - _stop)})
            self._emit_eval(
                **base, outcome="candidate_emitted",
                reason_code=code, reason=why,
                candidate=self._candidate_dict(sig, None, prov),
                decision={"result": ("shadow" if shadow
                                     else "observed"),
                          "reason_code": code, "reason": why,
                          "execution_eligible": False,
                          "order_result": "not_attempted",
                          "competing_scenarios": [
                              {"scenario": f"take the {_dir:+d} "
                                           f"{name} setup at "
                                           f"{_entry:.2f}",
                               "status": (
                                   "observed, not routed — §G "
                                   "shadow evaluation: journaled "
                                   "for grading, this route "
                                   "holds no order authority"
                                   if shadow else
                                   "observed, not routed — "
                                   "order authority belongs to "
                                   "the trigger path, which did "
                                   "not fire")},
                              {"scenario": f"take the "
                                           f"{-_dir:+d} side",
                               "status": "not funded — no "
                                         "opposite-side signal "
                                         "at the closing bar"}],
                          **_verdict_blocks(
                              analytical="not_evaluated",
                              eligible=False,
                              execution_code=code)},
                evidence={**evidence, "observed": _obs,
                          "supporting": _sup,
                          "contradictory": _con,
                          "evidence_groups": _groups,
                          "independent_evidence_groups":
                              sorted(_groups),
                          "provenance": prov,
                          # Journal auditability (owner,
                          # 2026-10-26): the complete trigger
                          # account — route authority, source
                          # eligibility, the scan's explicit
                          # non-evaluation of the trigger path,
                          # and the detector's own facts.
                          "trigger_decision":
                              _journal_audit.trigger_decision_block(
                                  route=spec.route, source=source,
                                  reason_code=code, shadow=shadow,
                                  detector_facts=
                                  _journal_audit.detector_facts(
                                      sig, frame))})
            # Setup lifecycle (owner rulings 2026-10-30 §1): a
            # non-shadow roster candidate is REGISTERED, not
            # just journaled — it persists with its trigger,
            # invalidation and expiry facts so a later trigger
            # fire can route it to execution evaluation.  The
            # scan itself still never orders; shadow (§G)
            # candidates are never registered.
            if not shadow and self._setup_store is not None \
                    and source == _lifecycle.SOURCE_LIVE_CLOSE:
                try:
                    self._register_scan_setup(
                        spec, sig, prov, frame=frame, t=t,
                        formation_ts=_formation_ts,
                        eval_now=_eval_now,
                        intended_tf=intended_tf)
                except Exception as _reg_exc:
                    self.journal.log_note(
                        f"setup_lifecycle: registration failed "
                        f"for {name}: {_reg_exc}")

    def _run_roster_close_evals(self, stats: dict | None = None,
                                *, source: str) -> None:
        """The 12 pure-H1 roster playbooks evaluate on every H1
        close (and once at warm start), independent of the
        generic H1 trigger lifecycle."""
        h1 = (self._stack.frames.get("H1")
              if self._stack is not None else None)
        for spec in ROSTER_SPECS_H1:
            try:
                self._sweep_eval_one(
                    spec, h1, clock="h1_close", intended_tf="H1",
                    actual_tf="H1", source=source, shadow=False)
            except Exception as exc:
                self.journal.log_note(
                    f"mt5: roster close eval failed for "
                    f"{spec.name}: {exc}")
        # Setup lifecycle upkeep (owner rulings 2026-10-30 §1):
        # expiry / close-rule invalidation for tracked setups.
        self._upkeep_setups(h1)

    def _run_range_family(self, stage: str,
                          stats: dict | None = None, *,
                          source: str) -> None:
        """The range family under its §G contract.

        ``stage="h4_generation"``: on an H4 close, each range
        playbook generates scenarios — the range detector's state
        derives exclusively from H4 closes (causally sliced by
        the playbook itself); the scan runs on the H1 timebase
        the playbooks are built on, at the freshest H1 bar.
        ``stage="h1_refinement"``: on an H1 close, the same scan
        refines entries against the current H4 context.  D1 bias
        is recorded as CONTEXT evidence only — it is never
        consulted as permission (D1 never vetoes; §G).  Both
        stages are shadow: they journal, they never order.
        """
        h1 = (self._stack.frames.get("H1")
              if self._stack is not None else None)
        setup = self._live_setup or {}
        ctx_evidence = {
            "stage": stage,
            "d1_bias_context": setup.get("bias_d1"),
            "d1_role": "context only — recorded, never consulted "
                       "as permission (owner §G: D1 never vetoes)",
        }
        if stage == "h4_generation":
            clock, itf, atf = "h4_generation", "H4", "H4"
        else:
            clock, itf, atf = "h1_refinement", "H1", "H1"
        for spec in RANGE_SPECS:
            try:
                self._sweep_eval_one(
                    spec, h1, clock=clock, intended_tf=itf,
                    actual_tf=atf, source=source, shadow=True,
                    extra_evidence=ctx_evidence)
            except Exception as exc:
                self.journal.log_note(
                    f"mt5: range {stage} eval failed for "
                    f"{spec.name}: {exc}")

    # ------------------------------------------------- alerts + charts
    def _studies_for_chart(self, h1, entry: float, atr: float,
                           zone) -> dict:
        """Studies shared by the PNG chart and the HTML signal view:
        nearby cache zones, the feature pipeline's auto-Fib + dealing
        range on H1, and the Fib read (nearest level, at-Fib, OTE).
        Any study that fails to compute comes back None — never
        invented, never breaking the alert."""
        nearby = []
        try:
            _v, cache_zones = self._zone_cache.nearby(entry, atr)
            nearby = zones_to_study_dicts(cache_zones, entry, zone)
        except Exception:
            nearby = []
        fib_levels = fib_range = ote_band = None
        fib_info = None
        try:
            states = read_and_featurize(h1, None)
            last = states[-1]
            fib = last.features.fib
            if fib.levels:
                fib_levels = [(float(f), float(p)) for f, p in fib.levels]
                if (math.isfinite(fib.ote_low)
                        and math.isfinite(fib.ote_high)):
                    ote_band = (float(fib.ote_low), float(fib.ote_high))
                nf = fib.nearest_frac
                fib_info = {
                    "nearest_frac": (float(nf)
                                     if math.isfinite(nf) else None),
                    "at_fib": bool(fib.at_fib),
                    "in_ote_long": bool(fib.in_ote_long),
                    "in_ote_short": bool(fib.in_ote_short),
                    "leg_direction": int(fib.leg_direction),
                }
            if (math.isfinite(last.range_high)
                    and math.isfinite(last.range_low)
                    and last.range_high > last.range_low):
                fib_range = (float(last.range_low),
                             float(last.range_high))
        except Exception:
            fib_levels = fib_range = ote_band = None
            fib_info = None
        return {"nearby": nearby, "fib_levels": fib_levels,
                "fib_range": fib_range, "ote_band": ote_band,
                "fib_info": fib_info}

    def _build_chart(self, sig, verdict: dict | None, *,
                     alert_id: str, playbook: str = "",
                     fill_price: float | None = None,
                     suffix: str = "",
                     decision_score: float | None = None,
                     broker_stop: float | None = None) -> tuple[bytes | None, Path | None]:
        """Chart PNG for a signal/fill: the WHOLE hierarchy stack
        (H4 structure / H1 trigger / entry-TF entry) + levels + verdict
        + the studies the reasoning used: nearby cache zones, the
        feature pipeline's auto-Fib + dealing range on H1, and
        killzone session shading.

        Step 7: the chart shows the DECISION score passed by the
        caller (never the signal object's sleeve placeholder) and
        BOTH stops — structural invalidation plus the broker stop
        when the caller knows one. A caller that passes neither gets
        "Score unavailable" / "Broker SL pending at placement".

        Studies are best-effort: any study that fails to compute is
        left off, never invented. Never breaks the signal path: on
        any failure returns (None, None) and logs a note; the text
        alert still goes out.
        """
        try:
            frames = self._stack.frames if self._stack else {}
            h1 = frames.get("H1")
            if h1 is None or len(h1) == 0:
                return None, None
            zone = (verdict or {}).get("zone")
            entry_clock = (verdict or {}).get("entry_clock") or "M15"
            entry = float(sig.entry_ref)
            atr = float(getattr(sig, "atr", 0.0) or 0.0)
            st = self._studies_for_chart(h1, entry, atr, zone)
            # Step 8: cache this signal's studies for the terminal.
            # The terminal labels them with this signal id + time;
            # it never recomputes studies from a guessed entry/ATR.
            self._latest_studies = {
                "signal": alert_id,
                "signal_time": datetime.now(timezone.utc).isoformat(),
                "zone": ({"top": float(zone["top"]),
                          "bottom": float(zone["bottom"]),
                          "kind": str(zone.get("kind", ""))}
                         if zone and zone.get("top")
                         and zone.get("bottom") else None),
                "nearby": st["nearby"],
                "fib": {
                    "levels": [[float(f), float(p)]
                               for f, p in (st["fib_levels"] or [])],
                    "ote": ([float(st["ote_band"][0]),
                             float(st["ote_band"][1])]
                            if st["ote_band"] is not None else None),
                    "range": ([float(st["fib_range"][0]),
                               float(st["fib_range"][1])]
                              if st["fib_range"] is not None else None),
                },
            }
            png = render_mt5_signal(
                h1, zone=zone, entry=entry,
                stop=float(sig.invalidation),
                targets=[float(x) for x in (sig.targets or [])],
                direction=int(sig.direction), verdict=verdict,
                playbook=playbook,
                decision_score=decision_score,
                broker_stop=broker_stop,
                instrument=self.cfg.symbol, fill_price=fill_price,
                h4=frames.get("H4"),
                entry_frame=frames.get(entry_clock),
                entry_clock=entry_clock,
                style=self.cfg.chart_style,
                nearby_zones=st["nearby"], fib_levels=st["fib_levels"],
                fib_range=st["fib_range"], ote_band=st["ote_band"])
            path = self._charts_dir / f"{alert_id}{suffix}.png"
            path.write_bytes(png)
            return png, path
        except Exception as exc:
            self.journal.log_note(
                f"mt5: chart build failed for {alert_id}: {exc}")
            return None, None

    def _build_signal_view(self, sig, verdict: dict | None, *,
                           alert_id: str, playbook: str = "",
                           fill_price: float | None = None,
                           suffix: str = "",
                           decision_score: float | None = None,
                           broker_stop: float | None = None) -> tuple[str | None, Path | None]:
        """Self-contained HTML terminal view for a signal: one page,
        baked at signal time, with per-timeframe bars + analysis,
        trade levels, and the same studies as the PNG chart. Saved
        next to the chart and attached to the alert as a document.

        Step 7: same truth as the PNG — the caller's decision score
        (never the sleeve placeholder) and both stops.

        Never breaks the signal path: on any failure returns
        (None, None) and logs a note.
        """
        try:
            frames = self._stack.frames if self._stack else {}
            h1 = frames.get("H1")
            if h1 is None or len(h1) == 0:
                return None, None
            zone = (verdict or {}).get("zone")
            entry = float(sig.entry_ref)
            atr = float(getattr(sig, "atr", 0.0) or 0.0)
            st = self._studies_for_chart(h1, entry, atr, zone)
            html_str = build_signal_view_html(
                frames=frames, entry=entry,
                stop=float(sig.invalidation),
                targets=[float(x) for x in (sig.targets or [])],
                direction=int(sig.direction), playbook=playbook,
                decision_score=decision_score,
                broker_stop=broker_stop,
                verdict=verdict, zone=zone,
                nearby_zones=st["nearby"], fib_levels=st["fib_levels"],
                fib_range=st["fib_range"], ote_band=st["ote_band"],
                fib_info=st["fib_info"], alert_id=alert_id,
                instrument=self.cfg.symbol, fill_price=fill_price,
                signal_time=datetime.now(timezone.utc).isoformat())
            path = self._charts_dir / f"{alert_id}{suffix}.html"
            path.write_text(html_str, encoding="utf-8")
            return html_str, path
        except Exception as exc:
            self.journal.log_note(
                f"mt5: signal view build failed for {alert_id}: {exc}")
            return None, None

    def terminal_provider(self):
        """Adapter: live stack + latest studies -> terminal provider."""
        runner = self

        class _TerminalProvider:
            def frames(self, tf):
                stack = runner._stack
                if stack is None:
                    return None
                return stack.frames.get(tf)

            def studies(self):
                return runner._latest_studies

            def status(self):
                from ..live.market_hours import xauusd_market_open
                stack = runner._stack
                tfs: dict[str, str | None] = {}
                if stack is not None:
                    for tf, frame in stack.frames.items():
                        if frame is not None and len(frame) > 0:
                            idx = frame.index[-1]
                            tfs[tf] = (idx.isoformat()
                                       if hasattr(idx, "isoformat")
                                       else str(idx))
                        else:
                            tfs[tf] = None
                return {
                    "server_time": datetime.now(timezone.utc).isoformat(),
                    "market_open": bool(xauusd_market_open()),
                    "tfs": tfs}

        return _TerminalProvider()

    def start_terminal_server(self) -> str:
        """Start the interactive terminal server; idempotent.

        Localhost-only by default (cfg.terminal_host) — reachable
        via RDP + localhost on the box, with zero new network
        exposure. A second call returns the running URL.
        """
        srv = self._terminal_server
        if srv is not None and srv.running:
            return srv.url
        self._terminal_server = TerminalServer(
            self.terminal_provider(), host=self.cfg.terminal_host,
            port=self.cfg.terminal_port, instrument=self.cfg.symbol)
        return self._terminal_server.start()

    def terminal_status(self) -> dict:
        """Terminal server state for the /terminal command."""
        srv = self._terminal_server
        return {
            "running": bool(srv is not None and srv.running),
            "url": srv.url if srv is not None else None,
            "host": self.cfg.terminal_host,
            "port": self.cfg.terminal_port}

    def _send_startup_progress(self, text: str) -> None:
        """One Telegram + console progress line during startup.

        The owner watches Telegram, not the console: every startup
        stage (starting / history / zones / playbooks / failed)
        reports itself, so a slow-but-healthy start never looks
        like a hang.  Never raises: no senders configured ->
        console only; a send failure is journaled and start
        continues.  Zone building has NO timeout — a slow zone
        build is reported, never cut short.
        """
        print(f"MT5 startup: {text}", flush=True)
        if not self.senders:
            return
        try:
            self._send_alert(text)
        except Exception as exc:  # noqa: BLE001 - never break start
            self._note_tg_failure(
                f"startup progress send failed: {exc}")

    def _send_startup_ping(self, acct: dict) -> None:
        """One Telegram 'engine started' ping per process start.

        Fires after feed connect + runner start (feed, executor,
        warm start, adoption all done).  Sending is INDEPENDENT of
        command polling (getUpdates): the ping goes out even with
        --no-poll-commands — the owner must be able to tell the
        engine is alive while the OANDA loop still owns getUpdates.
        Never raises: no senders configured -> console warning +
        journal note; a send failure is journaled + console-printed
        (via _send_alert / _note_tg_failure) and start continues.
        """
        if self._startup_ping_sent:
            return
        self._startup_ping_sent = True
        if not self.senders:
            # Accurate, not flat: name the missing vars + why
            # yesterday worked (session-scoped $env: exports).
            # _telegram_startup_warnings([]) returns exactly this
            # one line; never printed when a sender exists.
            for line in _telegram_startup_warnings([]):
                print(line, flush=True)
            self.journal.log_note(
                "mt5: startup ping skipped (no Telegram sender "
                "configured)")
            return
        try:
            mode = acct.get("trade_mode")
            mode_txt = ("demo" if mode == 0 else
                        "live" if mode == 2 else f"mode={mode}")
            # PRODUCTION FIX (gs-clock pass): execute_playbooks
            # defaults to None (= every evaluated playbook, see
            # _executes) — iterating it directly raised TypeError
            # ('NoneType' object is not iterable), the send failed,
            # and the startup ping NEVER fired on a default config
            # (the EC2 startup-ping bug: silent, no crash).  None
            # now resolves to the roster names, exactly as _executes
            # treats it; an explicit tuple still narrows it.
            if self.cfg.execute_playbooks is None:
                _executing_names = [pb.name
                                    for pb in self._get_playbooks()]
            else:
                _executing_names = list(self.cfg.execute_playbooks)
            executing = [p for p in _executing_names
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
            self._note_tg_failure(f"startup ping failed: {exc}")

    def _note_tg_failure(self, detail: str) -> None:
        """Journal + deduped console error for a Telegram send failure.

        A send failure must never be silent (journal-only) — the
        owner watches the console and would otherwise never know
        the bot is mute.  It must also never flood the console: the
        same failure prints at most once per 600s (same key window
        as alerts.telegram.ErrorDeduper).  Always journaled.
        """
        self.journal.log_note(f"mt5: alert send failed: {detail}")
        key = (detail or "")[:80]
        now = time.monotonic()
        last = self._tg_fail_last.get(key)
        if last is not None and now - last < 600:
            return
        self._tg_fail_last[key] = now
        print(f"MT5 Telegram send FAILED: {detail}", flush=True)

    def _send_alert(self, text: str, *,
                    chart_png: bytes | None = None,
                    html_path: Path | None = None,
                    reply_markup: dict | None = None) -> None:
        """Send a text alert, with the chart photo when one is attached.

        Sender contract (in preference order): ``send_photo(png,
        caption)`` (TelegramSender), ``send_text(text)``, legacy
        ``send(text)`` duck-type.  A sender failure is journaled +
        deduped-console-printed (see _note_tg_failure), never raised
        into the tick loop.  ``reply_markup`` (inline keyboard) is
        passed through when the sender supports it —
        used by the Golden Swing Confirm request.  ``html_path`` (the
        per-signal HTML view) goes out as a document after the alert,
        on senders that support it; a document failure is journaled
        the same way and never breaks the alert.
        """
        for sender in self.senders:
            try:
                if chart_png and hasattr(sender, "send_photo"):
                    res = sender.send_photo(chart_png, text)
                    if not getattr(res, "delivered", False):
                        self._note_tg_failure(
                            f"alert photo send failed: {res.detail}")
                elif hasattr(sender, "send_text"):
                    try:
                        res = sender.send_text(
                            text, reply_markup=reply_markup)
                    except TypeError:
                        # Legacy sender without reply_markup support.
                        res = sender.send_text(text)
                    if not getattr(res, "delivered", False):
                        self._note_tg_failure(
                            f"alert send failed: {res.detail}")
                else:
                    sender.send(text)
                if html_path is not None and hasattr(sender,
                                                     "send_document"):
                    try:
                        res = sender.send_document(
                            html_path,
                            caption="Signal view — open in a browser, "
                                    "switch timeframes")
                    except TypeError:
                        # Legacy sender without caption support.
                        res = sender.send_document(html_path)
                    if not getattr(res, "delivered", False):
                        self._note_tg_failure(
                            f"alert view send failed: {res.detail}")
            except Exception as exc:
                self._note_tg_failure(f"alert send failed: {exc}")

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
            # Background /backtest delivery: the finished report is
            # sent through the same reply path as sync commands.
            deliver=self._reply_command,
        )

    def _reply_command(self, result) -> None:
        """Send one command result back (document or text)."""
        if result is None:
            return
        for sender in self.senders:
            try:
                if result.document is not None:
                    if hasattr(sender, "send_document"):
                        try:
                            sender.send_document(
                                result.document,
                                caption=result.caption,
                                reply_markup=result.reply_markup)
                        except TypeError:
                            # Legacy sender without reply_markup support.
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

    def _telegram_available(self) -> bool:
        """Can we call the Telegram Bot API at all?

        Polling uses the env token / skill CLI (see _tg_api), NOT
        the alert senders: a missing GS_TELEGRAM_CHAT_ID disables
        alert sends (send_text needs a chat id) but must NOT
        disable command polling when a token is present — replies
        go to the chat id from the incoming update.  Mirrors
        _tg_api's transport check: env token, else tg-call CLI.

        An all-outbox sender list does NOT count as available:
        outbox mode means chat id set but no token (and no skill
        CLI) — polling would silently return nothing.  In that
        case fall through to the token/CLI check so the caller
        gets the honest 'polling disabled' warning instead of a
        silent dead poller.
        """
        if self.senders:
            if any(getattr(s, "mode", "live") != "outbox"
                   for s in self.senders):
                return True
        if os.environ.get("GS_TELEGRAM_BOT_TOKEN"):
            return True
        for cand in ("~/workspace/custom-connectors/"
                     "custom.legacy-telegram/bin/tg-call",
                     "~/workspace/skills/telegram/bin/tg-call"):
            if os.path.isfile(os.path.expanduser(cand)):
                return True
        return False

    def _poll_telegram_commands(self, stats: dict | None = None) -> None:
        """Poll getUpdates and answer owner commands (MT5 registry).

        ON by default: this engine OWNS getUpdates.  The OANDA loop's
        poller must be OFF at cutover (GS_TELEGRAM_POLL=0) — one
        consumer only.  First poll only syncs the offset (drains the
        backlog without answering stale commands); later polls answer.
        A poll failure is silent — the tick loop never stops for it —
        EXCEPT a missing transport (no token, no CLI), which is
        logged + printed once per process start.  ``stats`` (this
        tick's counters) is threaded to Confirm taps so a deferred
        Golden Swing order's execution counts on the tick that
        confirmed it.
        """
        if not self.cfg.poll_commands:
            return
        now = time.monotonic()
        if now - self._last_poll < self.cfg.poll_interval_seconds:
            return
        self._last_poll = now
        if not self._telegram_available():
            # No transport at all (no token, no CLI): polling cannot
            # work.  Logged + printed ONCE per process start, never
            # every 30s poll — a missing token must be visible, not
            # silent, and not spam.
            if not self._tg_unavailable_logged:
                self._tg_unavailable_logged = True
                print(
                    "MT5 startup: WARNING: Telegram polling "
                    "disabled — no bot token "
                    "(GS_TELEGRAM_BOT_TOKEN) and no tg-call CLI.",
                    flush=True)
                self.journal.log_note(
                    "mt5: Telegram polling disabled (no token, "
                    "no tg-call CLI)")
            return
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
                self._handle_callback_query(upd, stats)
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
    def _handle_callback_query(self, upd: dict,
                               stats: dict | None = None) -> None:
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
                ok, text = self.confirm_pending(
                    arg or "", stats if stats is not None else {})
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
                            result.reply_markup or mt5_control_keyboard())})
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
                            result.reply_markup or mt5_control_keyboard())})
                _answer("config")
                return
            if action in ("assess_all", "health", "forecast_train"):
                # Read-only navigation: re-render in place for text,
                # send as a new message for documents (a document
                # cannot edit a message in place).
                from .mt5_commands import (
                    _mt5_assess, _mt5_forecast_train, _mt5_health)
                _nav = {
                    "assess_all": lambda: _mt5_assess(
                        self._command_context(), ["all"]),
                    "health": lambda: _mt5_health(
                        self._command_context(), []),
                    "forecast_train": lambda: _mt5_forecast_train(
                        self._command_context(), []),
                }[action]
                result = _nav()
                if result.document is not None:
                    self._reply_command(result)
                elif message_id is not None:
                    self._tg_api("editMessageText", {
                        "chat_id": chat_id, "message_id": message_id,
                        "text": result.text or "",
                        "reply_markup": _json.dumps(
                            result.reply_markup or mt5_nav_keyboard())})
                _answer(action.replace("_", " "))
                return
            if action == "cmd":
                # Generic next-button routing: the button payload
                # is the typed command itself (``mt5:cmd:/pending``).
                # Dispatch through the same registry typed
                # commands use; re-render in place for text,
                # new message for documents (cannot edit in
                # place).  Results without a keyboard fall back
                # to a bare Back row so navigation never strands.
                cmd_text = (arg or "").strip()
                if not cmd_text.startswith("/"):
                    _answer("unknown button")
                    return
                parts = cmd_text[1:].split()
                if not parts:
                    _answer("unknown button")
                    return
                from .mt5_commands import MT5_COMMANDS as _MT5_REG
                entry = _MT5_REG.get("/" + parts[0])
                if entry is None:
                    _answer("unknown button")
                    return
                result = entry[1](self._command_context(), parts[1:])
                if result.document is not None:
                    self._reply_command(result)
                elif message_id is not None:
                    self._tg_api("editMessageText", {
                        "chat_id": chat_id, "message_id": message_id,
                        "text": result.text or "",
                        "reply_markup": _json.dumps(
                            result.reply_markup or mt5_back_keyboard())})
                _answer(parts[0].replace("_", " "))
                return
            _answer("unknown button")
        except Exception as exc:
            self.journal.log_note(
                f"mt5: callback handling failed: {exc}")
            _answer("button failed — try the command again")
