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

import json
import os
import math
import secrets
import time
from dataclasses import dataclass, field, replace
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..alerts.commands import CommandContext, is_owner_callback
from ..backtest.data import REPO_ROOT
from ..charts.engine import render_mt5_signal
from ..confluence import ConfluenceEngine
from ..features.engine import read_and_featurize
from ..feed.tick_log import TickLogger
from ..live.asymmetry import (STYLE_FLOORS, ScoreContext, floor_for,
                              score_asymmetry)
from ..live.atr_regime import (atr_regime as _atr_regime,
                               baseline_atr as _baseline_atr,
                               scaled_thresholds as _scaled_thresholds)
from ..live.dxy import DXYTracker
from ..live.funding import detect_funding_events
from ..live.health import HealthMonitor
from ..live.opposing import risk_tier as _zone_risk_tier, \
    tier_spec as _tier_spec
from ..live.sizing import RiskPool, size_for_risk
from ..live.urgency import urgency_score
from ..live.volume_profile import volume_profile
from ..live.hierarchy import (LEG_ORDER, TimeframeStack, TF_MINUTES,
                              check_live_trigger,
                              check_zone_touch, evaluate_hierarchy,
                              live_setup, live_setup_hold_state,
                              live_setup_reason, log_feed_error,
                              log_hold_state, qm_annotation,
                              resolve_setup_for_touch, zones_by_side)
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
from .signal_log import log_signal
from .mt5_commands import (mt5_confirm_keyboard, mt5_control_keyboard,
                           mt5_router, parse_mt5_callback)
from .mt5_config import LiveConfigStore
from .risk import RiskConfig, RiskGuards, size_position
from .tick_manager import (DEFAULT_TIME_EXIT_BARS, TickAction, TickConfig,
                          TickManager)
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
    fixed_lots: float | None = None  # when set, every trade uses it
    magic: int = 260101
    # A14 UNSHADOW (owner addendum 2026-10-21): the shadow tier is
    # abolished on the MT5 path — a log-only signal can never be
    # confirmed, and the demo engine IS the shadow.  None = ALL
    # playbooks evaluated on the H1 path execute through the same
    # decision flow (score floors, RiskPool, hard lines).  An
    # explicit tuple remains possible as a config override.
    # Capital protection is NOT this list: it is the RiskPool
    # ceiling + StyleBook per-style caps + the RiskGuards daily
    # cap (spec §11 A14).  The per-playbook kill switch is
    # ``disabled_playbooks`` below (default: all enabled).
    execute_playbooks: tuple | None = None
    min_score: float = 60.0
    # --- Confluence rework (spec v2, arrival pass) ------------------
    # Sizing mode (A13): "risk" = sizes computed from risk money in
    # Decimal (live/sizing.py), ledgered per position in a
    # RiskPool; risk is the MT5-path default.  "fixed" is the
    # legacy behaviour and is selected by setting fixed_lots (the
    # EC2 --fixed-lots flag), which always wins when present.
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
    # structure can resolve).  qm_m5 is the only M5-frame playbook
    # but it is shadow-only and never opens a position, so tagging it
    # would be inert config implying a behavior that cannot happen.
    # No live playbook qualifies as a scalp.  Config-file / env only.
    scalp_playbooks: tuple = ()
    scalp_cap_minutes: float = 15.0  # owner's scalping_max_holding
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
            fixed_lots=float(os.environ["GS_FIXED_LOTS"])
            if os.environ.get("GS_FIXED_LOTS") else None,
            allow_live=os.environ.get("GS_MT5_ALLOW_LIVE", "0") == "1",
            poll_commands=os.environ.get("GS_MT5_POLL_COMMANDS", "1") != "0",
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
    # Score-decay (owner's-bot PendingOrderTracker): the signal score
    # at placement + consecutive H1 closes on which a re-evaluation
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
        # daily/weekly/drawdown/consecutive-loss halts, the max
        # position cap and the unstopped-position refusal.
        self.guards = RiskGuards(
            self.cfg.risk_state_path,
            RiskConfig(max_total_risk_pct=0.0),
            kill_switch_path=self.cfg.kill_switch_path)
        self.tick_manager = tick_manager or TickManager(
            tick=TickConfig(
                scalp_cap_minutes=float(self.cfg.scalp_cap_minutes),
                time_exit_bars=self.cfg.time_exit_bars))
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
        self._qm_touch: dict | None = None
        # Daily QM shadow: last touch index already journal-logged
        # (the D1 scan logs each touch once, never re-logs history).
        self._qm_d1_last_logged: int = -1
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
        self._pullback_watch: dict | None = None
        self._resting_limit_key: str | None = None
        self._sizing_detail: dict[str, dict] = {}
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

        ``qm_d1`` is deliberately NOT here: the H1 path drives
        ``live_signal`` with H1 bars, and running D1 QM logic on H1
        bars would emit H1 patterns mislabelled as Daily QM.  The
        Daily QM scan lives in :meth:`_scan_qm_d1`, which evaluates
        it on the real D1 frame on every D1 close (see
        :meth:`_refresh_bars`) — it is outside A14's scope (not an
        H1-path playbook; no D1-clock order path exists yet)."""
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
                               QMM5Playbook(),
                               RangeSweepReclaimPlaybook(),
                               RangeQMPlaybook(),
                               RangeMeanReversionPlaybook(),
                               RangeCompressionBreakoutPlaybook(),
                               NarrativePlaybook()]
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
            self._send_startup_progress(
                "MT5 engine: history fetched ("
                + ", ".join(f"{tf}:{len(self._stack.frames.get(tf, []))}"
                            for tf in self._stack.frames)
                + ")")
            self._qm_touch = self._advance_qm(self._stack.frames.get("M5"))
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
                       "qm_d1_shadow": 0}
        # Daily QM shadow: evaluate the D1 frame once at warm start
        # (the latest touch, if any, is logged here; later touches
        # are logged on their D1 close by _refresh_bars).
        self._scan_qm_d1(self._stack.frames.get("D1"), stats)
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
            self._health.note_tick(tick.get("time"))
            # Day rollover -> the honest recap for the day/week that
            # just ended (wins AND losses).  Never breaks the loop.
            self._maybe_send_rollover_recap(tick)
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
                self._manage_tick(price, stats)
            if self._tick_count % self.cfg.reconcile_every_ticks == 0:
                if self._regime_pause != "paused_complete":
                    self._reconcile(price, stats)
            self._refresh_bars(tick, stats)
            # Live evaluation clock: touch arms, family confirms on
            # entry-TF closes.  H1 closes never gate evaluation.
            self._evaluate_live_tick(tick, stats)
            # Golden Swing confirm timeouts are wall-clock: sweep
            # every tick (a tap processed after timeout is late —
            # the token is already gone, "confirmation expired").
            self._sweep_order_confirms(stats)
            self._poll_telegram_commands(stats)
        return _handle

    def _manage_tick(self, price: float, stats: dict) -> None:
        """TickManager actions for every tracked position, executed now.

        Every executed management action that changes the position
        gets a Telegram follow-up (breakeven / partial / layer stop)
        — the disciplined caller's update — on top of the journal
        note.  Trail SL moves stay journal-only: they fire on
        nearly every favourable tick and would spam the channel.
        """
        for ticket in list(self.tick_manager.tickets()):
            for action in self.tick_manager.on_tick(ticket, price):
                stats["tick_actions"] += 1
                pos = self.tick_manager.position(ticket)
                pb = self._playbook_from_alert_id(
                    pos.alert_id if pos is not None else "")
                if action.kind == "modify_sl":
                    res = self.executor.modify_sl(ticket, action.level)
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
                        elif action.reason == "profit_lock":
                            self._send_alert(
                                f"MT5 PROFIT LOCK {pb} ticket {ticket}: "
                                f"SL -> {action.level:.2f} — profit "
                                f"locked in, trade cannot lose")
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
                    # leg, the base rides.
                    res = self.executor.close_position(ticket, action.volume)
                    if res.ok:
                        self.tick_manager.apply(ticket, action)
                        self.journal.log_note(
                            f"mt5 tick: ticket {ticket} layer stop "
                            f"{action.volume} lots at {price:.2f} "
                            f"(layer SL {action.level:.2f})")
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
            # money_for_move abs()es the distance, so sign it by
            # hand: a long below entry (or a short above) is a LOSS.
            # Booking the unsigned value would count every losing
            # long as a win — in the journal, the recaps, and the
            # RiskGuards outcome feed.
            pnl_money = spec.money_for_move(
                price - pos.entry_price, pos.volume)
            if d * (price - pos.entry_price) < 0:
                pnl_money = -pnl_money
        except Exception:
            pass
        # Classify BEFORE remove(): the classifier reads the live
        # effective stop from the TickManager.
        display = self._classify_close(pos, price, reason)
        self.journal.log_close(
            alert_id=pos.alert_id, ticket=ticket, exit_price=price,
            reason=reason, pnl_money=float(pnl_money), r_net=float(pnl_r),
            hold_bars=0, partial_taken=pos.partial_done)
        self._det_journal("trade_closed", {
            "alert_id": pos.alert_id, "ticket": ticket,
            "exit": price, "reason": reason,
            "pnl_money": float(pnl_money), "r_net": float(pnl_r)},
            idempotency_key=f"close-{pos.alert_id}-{ticket}")
        self.journal.log_note(
            f"mt5 close: ticket {ticket} {display} at {price:.2f} "
            f"({pnl_r:+.2f}R)")
        # Telegram follow-up — the disciplined caller's update: every
        # close is reported, wins AND losses, with the R and the
        # reason (a loss is never silently journalled).
        pb = self._playbook_from_alert_id(pos.alert_id)
        self._send_alert(
            f"MT5 CLOSE {pb} {'+' if d > 0 else '-'}1 "
            f"{pos.volume} lots entry {pos.entry_price:.2f} -> exit "
            f"{price:.2f} — {display} {pnl_r:+.2f}R "
            f"(ticket {ticket})")
        # Feed the closed trade's outcome to RiskGuards: the
        # daily/weekly/drawdown/consecutive-loss halts only work if
        # outcomes are booked.  Never breaks the close path.
        # (Partial closes are not separately booked — the full
        # close books the remainder volume's P&L, matching the
        # journal's close record.)
        try:
            balance = float(self.feed.account_state().get("balance", 0.0))
            if balance > 0:
                self.guards.record_outcome(float(pnl_money), balance)
        except Exception as exc:
            self.journal.log_note(
                f"mt5: risk outcome not booked for ticket {ticket}: "
                f"{exc}")
        # Feed the close to the entry gates: the loss/win cooldown
        # starts at the close.  Never breaks the close path.
        try:
            self.entry_gate.record_close(time.time(), pnl_r > 0)
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
        if _pool is not None:
            self.journal.log_note(
                f"mt5 pool closed: ticket {ticket} budget "
                f"{_pool.budget} consumed at close "
                f"{_pool.consumption()} ({pb} {pnl_r:+.2f}R)")
        if _narr_state is not None:
            try:
                _idle, _entry = _finalize_narrative_exit(
                    _narr_state, reason=reason, final_r=float(pnl_r),
                    pnl_dollars=float(pnl_money), exit_price=float(price))
                self.journal.log_note(
                    f"mt5 narrative STATE 7 EXIT: "
                    f"{_entry.get('exit_reason', reason)} "
                    f"R={_entry.get('realized_r', pnl_r):+.2f} "
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
        # Logged (info): this fetch is the live engine's normal
        # refresh path, NOT a second warm start.  fetch_new_all only
        # fetches bars since the cursor and appends nothing new when
        # the tick was stale — a routine refresh, not a re-fetch of
        # history.  (Before this log, a refresh's W1/MN1 short-history
        # warnings looked like a duplicate warm start; those
        # warnings are now warm-start-only via warn_short=False.)
        _log.info("mt5 refresh: TFs closed: %s — fetching new bars",
                  closed_tfs)
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
                self._qm_touch = self._advance_qm(
                    self._stack.frames.get("M5"))
            if tf == "D1":
                self._scan_qm_d1(
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

    def _journal_new_breaches(self) -> None:
        """Journal health breaches that were not breaching before —
        a stuck breach gets one note, not one per order."""
        current = self._health.breaches()
        new = current - self._health_last_breaches
        self._health_last_breaches = current
        for key in sorted(new):
            self.journal.log_note(
                f"mt5: health breach: {key} — {self._health.describe()}")

    def health_line(self) -> str:
        """One-line execution health for /status (see mt5_commands)."""
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
        w = self._pullback_watch
        if w is not None:
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
        # downtime -> stale.
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
        for ticket in list(self.tick_manager.tickets()):
            for action in self.tick_manager.on_bar_close(
                    ticket, last_bar_index, regime_structure):
                res = self.executor.close_position(ticket, action.volume)
                if res.ok:
                    self._record_close(
                        ticket,
                        self._last_tick["mid"] if self._last_tick else 0.0,
                        "timeout", stats)
        # Pending-order expiry: unfilled past expiry_bar dies — the
        # setup went stale (a real trader does not leave a limit
        # resting at a dead zone).
        # Score decay runs BEFORE expiry: a setup that died on
        # re-evaluation is cancelled as score_decay (more
        # informative than expired), and decay at 2 bars fires
        # before the 4-bar expiry anyway.
        self._check_pending_score_decay(stats)
        for ticket, state in list(self._pending_orders.items()):
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
    def _enter_pullback_watch(self, pb, sig, verdict, zone_d,
                              score, funding) -> None:
        """Park a runaway setup in PullbackWatch (spec §5): the
        DO NOT CHASE reset is gone on the arrival path — the setup
        is not dismantled, the watch awaits the pullback for up to
        ``pullback_watch_bars`` H1 bars, dying early only on a
        close through the zone's far edge (the idea is dead, not
        merely late)."""
        if self._pullback_watch is not None:
            return  # one watch at a time; the first setup owns it
        self._pullback_watch = {
            "direction": int(sig.direction),
            "zone": dict(zone_d),
            "playbook": pb.name,
            "bar0": self._current_h1_bar(),
            "score_total": float(score.total),
        }
        self.journal.log_note(
            f"mt5 watch: {pb.name} "
            f"{'+' if sig.direction > 0 else '-'}1 runaway from "
            f"{zone_d.get('tf', '?')} zone "
            f"{float(zone_d['bottom']):.1f}-{float(zone_d['top']):.1f} "
            f"— PullbackWatch armed for "
            f"{self.cfg.pullback_watch_bars} H1 bars (asym "
            f"{score.total:.0f}); the setup is parked, not reset")

    def _maintain_pullback_watch(self, mid: float) -> None:
        """Per-tick watch upkeep: pullback arrival clears the
        watch (the live setup re-arms through the normal trigger
        machinery — nothing was dismantled); a close through the
        far edge kills it; the bar budget lapses it."""
        w = self._pullback_watch
        if w is None:
            return
        zone = w["zone"]
        d = int(w["direction"])
        if self.tick_manager.tickets():
            self._pullback_watch = None
            self.journal.log_note(
                "mt5 watch: cleared — a position is open, the "
                "watch is moot")
            return
        if float(zone["bottom"]) <= mid <= float(zone["top"]):
            self._pullback_watch = None
            self.journal.log_note(
                f"mt5 watch: pullback arrived — price back inside "
                f"the {zone.get('tf', '?')} zone "
                f"{float(zone['bottom']):.1f}-{float(zone['top']):.1f}; "
                "watch cleared, the armed setup owns the entry now")
            return
        # Invalidation: an H1 CLOSE beyond the far edge + buffer.
        try:
            h1 = (self._stack.frames.get("H1")
                  if self._stack is not None else None)
            atr = float(zone.get("atr") or self._last_zone_atr or 0.0)
            if h1 is not None and len(h1) and atr > 0:
                last_close = float(h1["close"].iloc[-1])
                buf = 0.25 * atr
                dead = (last_close < float(zone["bottom"]) - buf) \
                    if d > 0 else \
                    (last_close > float(zone["top"]) + buf)
                if dead:
                    self._pullback_watch = None
                    self.journal.log_note(
                        "mt5 watch: KILLED — H1 close through the "
                        "zone's far edge; the idea is dead, not "
                        "late (§5 probe-death rule)")
                    return
        except Exception:
            pass
        if self._current_h1_bar() - int(w["bar0"]) \
                > int(self.cfg.pullback_watch_bars):
            self._pullback_watch = None
            self.journal.log_note(
                f"mt5 watch: lapsed after "
                f"{self.cfg.pullback_watch_bars} H1 bars without a "
                "pullback")

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
            total = float(score.total)
            band = 1.0 if total >= 75.0 else (
                0.75 if total >= 60.0 else 0.5)
            _tier, tier_mult, _ceil = self._traffic(zone_d)
            risk_money = float(acct.get("equity", 0.0)) \
                * float(self.cfg.risk_pct) * band * tier_mult
            result = size_for_risk(
                risk_money, abs(entry - stop0), spec)
            if not result.ok:
                self.journal.log_note(
                    f"mt5 resting limit: DECLINED sizing "
                    f"{result.verdict} at fresh MAJOR "
                    f"{major.get('tf', '?')} {prox:.1f} (never "
                    "rounded up)")
                return
            lots = float(result.lots)
            detail = {"authorized_risk": float(result.risk_money),
                      "realized_risk": float(result.realized_risk),
                      "drag": float(result.drag),
                      "raw_lots": float(result.raw_lots),
                      "lots": lots}
        elif self.cfg.fixed_lots is not None:
            lots = float(self.cfg.fixed_lots)
        else:
            sizing = size_position(
                balance=float(acct.get("equity", 0.0)),
                risk_pct=self.cfg.risk_pct, entry=entry, stop=stop0,
                spec=spec)
            if sizing.lots <= 0:
                return
            lots = float(sizing.lots)
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
                trigger_source="resting_limit")

    def _evaluate_live_tick(self, tick: dict, stats: dict) -> None:
        """Live evaluation clock: touch arms, family confirms, entry
        on entry-TF closes.  H1 closes never gate evaluation."""
        if self._stack is None:
            return
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
        self._run_live_signal(fired, tick, stats)

    def _run_live_signal(self, pend: dict, tick: dict,
                         stats: dict) -> None:
        """Entry confirmed: run the signal path on the live frame.

        The live frame is the closed H1 bars PLUS the forming bar
        (live values from ticks).  The verdict is built from the
        fired trigger + the confirm tick — same shape as the
        closed-bar verdict — so the playbooks, confluence gate, and
        order path are unchanged.
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
        """Effective sizing mode (A13): an explicit fixed_lots
        always wins (the EC2 --fixed-lots flag is the owner's
        deliberate override); otherwise the configured mode."""
        if self.cfg.fixed_lots is not None:
            return "fixed"
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

    def _score_for(self, *, direction: int, verdict: dict | None,
                   price: float, zone_d: dict | None,
                   atr: float) -> object:
        """Asymmetry Score for one candidate (spec §2).  Frames
        are parameters, not premises: the H1 path scores on the
        H1 frame (the path's clock) with the D1 frame as the
        dealing range when the stack carries one — P3 floats
        these per style.

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
        ctx = ScoreContext(
            direction=int(direction),
            frame=frames.get("H1"),
            atr=float(atr or 0.0), price=float(price),
            style="", zone=zone_d,
            opposing_zone=self._opposing_dict(direction),
            range_frame=range_frame,
            verdict=verdict,
            dxy_corr=(verdict or {}).get("dxy_corr"))
        return score_asymmetry(ctx, allow_flip=not at_zone)

    def _funding_report(self, verdict: dict | None,
                        zone_d: dict | None):
        """Funding-event detectors over the entry frame (spec
        §4.2): the chain evaluators reused, never gated on.  The
        entry frame + ATR follow the narrative chain's own
        convention (M15 bars, H4 ATR, regime-scaled thresholds)."""
        frames = (self._stack.frames if self._stack is not None
                  else {}) or {}
        entry = frames.get("M15")
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
            return "YELLOW", 1.0, 99
        tier = _zone_risk_tier(zone_d)
        spec = _tier_spec(tier)
        ceiling = {"GREEN": 3, "YELLOW": 2, "RED": 1}.get(tier, 2)
        return tier, float(spec.get("size_mult", 0.75)), ceiling

    def _pool_for_open(self, ticket: int, *, lots: float,
                       entry: float, stop: float, direction: int,
                       alert_id: str) -> RiskPool | None:
        """Create + book the RiskPool for a freshly opened base
        position (A13).  Budget = the legacy total-risk envelope
        (equity x risk_pct x layer cap multiple) so the rework's
        ceiling is exactly the old one; distribution inside it is
        evidence-driven.  Fixed mode books the owner's explicit
        lots against an envelope at least as large (the pool is
        the ledger there, RiskGuards remain the gate)."""
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
            # it.  In fixed mode this is expected (owner-explicit
            # lots; RiskGuards gated the order).  In risk mode it
            # means placement sized past the envelope — visible,
            # never silent.
            try:
                realized = float(spec.money_for_move(
                    abs(float(entry) - float(stop)), float(lots)))
            except Exception:
                realized = budget
            self.journal.log_note(
                f"mt5 pool: base layer risk {realized:.2f} exceeds "
                f"the position budget {budget:.2f} for ticket "
                f"{ticket} ({self._sizing_mode()} mode) — ledger "
                "widened to fit the filled order, headroom 0")
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
                         stats: dict) -> None:
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
        for pb in self._get_playbooks():
            if pb.name in (self.cfg.disabled_playbooks or ()):
                continue  # kill switch (/set) — not evaluated (A14)
            try:
                self._run_one_playbook(pb, sig_ctx=(ctx, engine),
                                       h1=h1, t=t, verdict=verdict,
                                       stats=stats, zone_d=zone_d,
                                       funding=funding)
            except Exception as exc:
                # A crashing playbook never kills the signal path.
                self._record_blocked(
                    f"playbook_error: {type(exc).__name__}: {exc}",
                    pb.name)
                self.journal.log_note(
                    f"mt5: {pb.name} signal path failed: "
                    f"{type(exc).__name__}: {exc}")

    def _run_one_playbook(self, pb, *, sig_ctx, h1, t, verdict, stats,
                          zone_d=None, funding=None) -> None:
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
        # Range playbooks need the native H4 frame (threaded in
        # here — no H4-resample fallback anywhere on this path).
        if hasattr(pb, "set_h4_frame"):
            pb.set_h4_frame(self._stack.frames.get("H4"))
        # Positional, as before: playbook live_signal signatures name
        # the frame parameter differently (df / h1).
        sigs = pb.live_signal(ctx.states, t, h1) or []
        for sig in sigs:
            alert_id = f"MT5-{pb.name}-{sig.index}"
            if sig.index - self._last_sig_bar.get(pb.name, -10 ** 9) \
                    < self.cfg.signal_gap_bars:
                continue
            # ---- Asymmetry Score (the decision number) ----------
            atr_h1 = float(atr14(h1)) if len(h1) else 0.0
            score = self._score_for(
                direction=int(sig.direction), verdict=verdict,
                price=float(sig.entry_ref), zone_d=zone_d,
                atr=atr_h1 or float(getattr(sig, "atr", 0.0) or 0.0))
            breakdown = score.breakdown()
            self._last_scores[pb.name] = breakdown
            self._last_score = breakdown
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
                    hierarchy=hier_rec, score_breakdown=breakdown)
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
                    hierarchy=hier_rec, score_breakdown=breakdown)
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
                    hierarchy=hier_rec, score_breakdown=breakdown)
                continue
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
            if str(_v.get("regime", "")) == "range":
                path = "RANGE"
            elif _arm == "continuation" \
                    or _style in ("continuation",
                                  "INTRADAY_CONTINUATION"):
                path = "INTRADAY_CONTINUATION"
            elif _style in STYLE_FLOORS:
                path = _style
            else:
                path = "ZONE_PATH"
            fl = floor_for(path)
            if zone_d is None:
                floor_ok = True
                self.journal.log_note(
                    f"mt5: {pb.name} signal has no zone context — "
                    "asymmetry floor not applied (not an "
                    "arrival-path signal); hard lines only")
            else:
                floor_ok = score.total >= float(fl["floor"]) \
                    and score.component("location") >= float(
                        fl.get("min_location", 0.0)) \
                    and score.component("structure") >= float(
                        fl.get("min_structure", 0.0)) \
                    and score.component("efficiency") >= float(
                        fl.get("min_efficiency", 0.0)) \
                    and score.component("pullback") >= float(
                        fl.get("min_pullback", 0.0))
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
                        trigger_source="arrival_score")
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
                    hierarchy=hier_rec, score_breakdown=breakdown)
                continue
            conf, notes = engine.score_live_signal(
                pb, sig, ctx, hierarchy_verdict=verdict)
            trigger_source = "funding_layer" if (
                holding and funding is not None
                and funding.events) else "arrival_score"
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
                trigger_source=trigger_source)
            if not auto:
                # Config-narrowed execute list (A14 override): the
                # signal was evaluated and logged; the owner's
                # explicit tuple keeps it from ordering.
                continue
            # Chart: H4 zone + entry/stop/targets + verdict, saved
            # next to the journal and attached to the alert.
            png, _path = self._build_chart(
                sig, verdict, alert_id=alert_id, playbook=pb.name)
            self._send_alert(
                f"MT5 [{tag}] {pb.name} "
                f"{'+' if sig.direction > 0 else '-'}1 "
                f"entry~{sig.entry_ref:.1f} stop~{sig.invalidation:.1f} "
                f"conf={conf:.0f}"
                + (f"\n\n{reasoning}" if reasoning else ""),
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
                        factors=list(notes))
                    continue
            if self.tick_manager.tickets():
                self._try_layer(pb, sig, stats, verdict=verdict,
                                entry_bar=t, score=score,
                                funding=funding, zone_d=zone_d)
            elif self.cfg.golden_swing_confirm_enabled \
                    and self._alignment.golden_swing:
                # Golden Swing base order: the order WAITS for the
                # owner's Confirm tap (REJECT cancels; timeout
                # fires/cancels per config).  Nothing is placed
                # here.  Layers never come here — a held position
                # routes to _try_layer above.
                self._stage_order_confirm(pb, sig, stats,
                                          verdict=verdict, entry_bar=t)
            else:
                self._place_live_order(pb, sig, stats,
                                       verdict=verdict,
                                       entry_bar=t, score=score,
                                       zone_d=zone_d)

    # ------------------------------------------------------------- layering
    # Scale-in onto ONE position: a confirming setup in the same
    # direction adds a layer (own entry / volume / SL) while the
    # position is in profit, with total risk capped.  The broker nets
    # the layer deal into the one position (same symbol / direction /
    # magic); the engine tracks the legs in TickManager.layers.
    def _try_layer(self, pb, sig, stats: dict,
                   verdict: dict | None = None,
                   entry_bar: int | None = None,
                   score=None, funding=None, zone_d=None) -> None:
        """A signal while holding: layer or skip, never a second
        position.  REWORK (spec §4): layers are released by the
        funding ladder — a NEW funding event (structure shift /
        displacement / CHoCH-MSB / FVG / sweep / absorption) with
        the stop at breakeven, or the legacy in-profit path — and
        sized from the RiskPool's headroom in risk mode.  Fixed
        mode keeps the legacy arithmetic, unchanged."""
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
        # it never vetoes the position itself.
        _tier, _smult, ceiling = self._traffic(zone_d)
        max_layers = min(int(self.cfg.layer_max_layers), ceiling)
        if len(layers) >= max_layers:
            self.journal.log_note(
                f"mt5: layer skipped {pb.name}: layer ceiling "
                f"{max_layers} reached (config "
                f"{self.cfg.layer_max_layers}, traffic {_tier})")
            return
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
            if not result.ok:
                self.journal.log_note(
                    f"mt5: layer DECLINED {pb.name}: sizing "
                    f"{result.verdict} — headroom {headroom} over "
                    f"stop distance "
                    f"{abs(float(sig.entry_ref) - stop0):.2f} gives "
                    f"raw {result.raw_lots} lots, below the broker "
                    "minimum after step flooring (never rounded up)")
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
                    "drag": float(result.drag),
                    "raw_lots": float(result.raw_lots)}
            self._place_live_order(
                pb, sig, stats, verdict=verdict,
                entry_bar=entry_bar, is_layer=True,
                layer_lots=float(result.lots),
                funding_events=new_events,
                score=score, zone_d=zone_d)
            return
        # ---- legacy / fixed-mode arithmetic (unchanged) ---------
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
        if eff is not None and ((d == 1 and level > eff)
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
        for a feed that never claimed the check)."""
        fn = getattr(self.feed, "market_open", None)
        if fn is None:
            return True, ""
        try:
            return fn()
        except Exception as exc:
            return False, f"market_open check failed: {exc}"

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
    def _place_live_order(self, pb, sig, stats: dict,
                          verdict: dict | None = None,
                          entry_bar: int | None = None,
                          is_layer: bool = False,
                          layer_lots: float | None = None,
                          funding_events: list | None = None,
                          score=None, zone_d: dict | None = None
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
        # Market-closed guard (order placement): never place while
        # the broker is not accepting new orders — covers
        # confirm-fired orders and layers, not just the signal path.
        open_ok, open_reason = self._market_open()
        if not open_ok:
            self.journal.log_note(
                f"mt5: order refused {pb.name}: market closed "
                f"({open_reason})")
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
        if is_layer:
            # Layer size was pre-capped by _try_layer (total risk cap).
            lots = float(layer_lots)
        elif self.cfg.fixed_lots is not None:
            lots = float(self.cfg.fixed_lots)
        elif self._sizing_mode() == "risk":
            # ---- A13 risk sizing (Decimal, quantized once) ------
            # Risk money = equity x risk_pct x score band x traffic
            # scalar x the legacy size multipliers (alignment /
            # regime / news / session applied to the RISK, so the
            # lot quantization happens exactly once, in Decimal).
            total = float(score.total) if score is not None else 75.0
            band = 1.0 if total >= 75.0 else (
                0.75 if total >= 60.0 else 0.5)
            _tier, tier_mult, _ceil = self._traffic(zone_d)
            risk_money = float(acct["equity"]) \
                * float(self.cfg.risk_pct) * band * tier_mult
            risk_money *= float(
                (verdict or {}).get("size_mult", 1.0) or 1.0)
            risk_money *= float(news_mult)
            if self.cfg.golden_swing_sizing_enabled:
                risk_money *= float(self._alignment.size_multiplier)
            risk_money *= session_multiplier(
                now_utc, self.cfg.session_multipliers)
            result = size_for_risk(
                risk_money, abs(float(sig.entry_ref) - stop0), spec)
            if not result.ok:
                self.journal.log_note(
                    f"mt5: order DECLINED {pb.name}: sizing "
                    f"{result.verdict} — risk {risk_money:.2f} over "
                    f"stop distance "
                    f"{abs(float(sig.entry_ref) - stop0):.2f} gives "
                    f"raw {result.raw_lots} lots (below broker "
                    "minimum after step flooring; never rounded up)")
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
                "drag": float(result.drag),
                "raw_lots": float(result.raw_lots),
                "lots": lots}
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
            # MacroTrend alignment sizing (base orders only): Golden
            # Swing 1.5x, D1/W1 conflict 0.5x.  fixed_lots is an
            # explicit override and is never multiplied; layers are
            # NOT multiplied here — they scale via the layer risk
            # budget, which derives from the (multiplied) base risk.
            lots = self._apply_alignment_multiplier(lots, spec, pb.name)
            # Regime sizing (base orders only): RANGE regime trades
            # at 0.5x (choppy regime = smaller).  Same no-multiply
            # rules as alignment: fixed_lots is an explicit
            # override and is never multiplied; layers scale via
            # the layer risk budget, which derives from the
            # (multiplied) base risk.
            lots = self._apply_size_multiplier(
                lots, spec,
                float((verdict or {}).get("size_mult", 1.0) or 1.0),
                f"regime {(verdict or {}).get('regime', 'trend')} "
                f"{pb.name}")
            # News-adaptive sizing: shrink entries in the post-news
            # cooldown blend (1.0 far from news).  Same no-multiply
            # rules as alignment (risk-% base orders only).
            lots = self._apply_size_multiplier(
                lots, spec, news_mult, f"news {pb.name}")
            # Session sizing: the session multiplier for the tick
            # clock (killzone sessions are the overlap bucket,
            # off-session is reduced).  Same no-multiply rules.
            lots = self._apply_size_multiplier(
                lots, spec,
                session_multiplier(now_utc, self.cfg.session_multipliers),
                f"session {pb.name}")
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
        kind, level, reason = select_order_type(
            direction=int(sig.direction), price=price,
            entry_ref=float(sig.entry_ref),
            zone=(verdict or {}).get("zone"), atr=float(sig.atr),
            urgency=_urgency, profile=_profile)
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
                funding_events=funding_events)
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
                               **_narr_kwargs)
        if _narr_state is not None:
            self._narrative_states[int(res.ticket)] = _narr_state
        # A13: open the position's RiskPool ledger on the fill.
        self._pool_for_open(
            int(res.ticket), lots=float(lots), entry=float(fill),
            stop=float(sig.invalidation),
            direction=int(sig.direction),
            alert_id=request.alert_id)
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
        _sd = self._sizing_detail.get(request.alert_id)
        if _sd is not None:
            _pool = self._pools.get(int(res.ticket))
            self.journal.log_note(
                f"mt5 sizing: {pb.name} authorized risk "
                f"${_sd['authorized_risk']:.2f} -> realized "
                f"${_sd['realized_risk']:.2f} (quantization drag "
                f"${_sd['drag']:.2f}, raw {_sd['raw_lots']:.4f} "
                f"lots) · pool "
                f"{_pool.state() if _pool is not None else 'n/a'}")
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
                             is_layer: bool = False,
                             ladder_levels: list | None = None,
                             ladder_lots: list | None = None
                             ) -> PendingOrderState:
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
            is_layer=bool(is_layer),
            placed_score=float(getattr(sig, "score", 0.0) or 0.0),
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
                             funding_events: list | None = None
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
            ladder_levels=ladder_levels, ladder_lots=ladder_lots)
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
        png, _path = self._build_chart(
            sig, verdict, alert_id=request.alert_id, playbook=pb.name,
            suffix="-pending")
        self._send_alert(
            f"MT5 PENDING {kind.upper()} {pb.name} "
            f"{'+' if sig.direction > 0 else '-'}1 "
            f"{request.volume} lots @ {level:.2f} "
            f"SL {request.stop_loss:.2f} (ticket {res.ticket})",
            chart_png=png)
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
                **_narr_kwargs)
            if _narr_state is not None:
                self._narrative_states[int(pos.ticket)] = _narr_state
            self._pool_for_open(
                int(pos.ticket), lots=float(filled_vol),
                entry=float(pos.entry_price),
                stop=float(state.invalidation),
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
                               **_narr_kwargs)
        if _narr_state is not None:
            self._narrative_states[int(pos.ticket)] = _narr_state
        self._pool_for_open(
            int(pos.ticket), lots=float(state.volume),
            entry=float(pos.entry_price),
            stop=float(state.invalidation),
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
        is the broker's fill price, the stop is the placement-time
        invalidation, the volume is the filled volume."""
        sig = getattr(state, "sig", None)
        try:
            spec = self.executor.symbol_spec(self.cfg.symbol)
            risk_money = spec.money_for_move(
                abs(float(pos.entry_price) - float(state.invalidation)),
                float(volume))
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
            volume=float(volume), risk_money=float(risk_money),
            factors=list(getattr(sig, "factors", []) or [])
            if sig is not None else [])
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
                f"realized ${_sd['realized_risk']:.2f} "
                f"(quantization drag ${_sd['drag']:.2f}, raw "
                f"{_sd['raw_lots']:.4f} lots) · pool "
                f"{_pool.state() if _pool is not None else 'n/a'}")
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
            f"{volume} lots @ {float(pos.entry_price):.2f} "
            f"SL {state.sl:.2f} (ticket {pos.ticket})"
            + (" — PARTIAL" if partial else ""),
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

    def _scan_qm_d1(self, d1_df, stats: dict | None = None) -> list[dict]:
        """Daily QM SHADOW scan on the D1 frame: evaluated + logged,
        never ordered, never charted, never alerted.

        Runs on the real D1 frame at warm start and on every D1
        close (see :meth:`_refresh_bars`) — NOT through the
        H1-clocked signal path, which would run D1 logic on H1 bars.
        Each touch is journal-logged exactly once (tracked by
        ``_qm_d1_last_logged``); the scan itself is the shadow book
        for this playbook.  Switched off via
        ``/set playbook.qm_d1 off`` like any other shadow playbook.
        Failures return [] (annotation missing, never a crash).
        """
        if "qm_d1" in (self.cfg.disabled_playbooks or ()):
            return []
        try:
            if d1_df is None or len(d1_df) < 30:
                return []
            pb = QMD1Playbook()
            ctx = EngineContext(df=d1_df, states=[], years=0.0,
                                instrument=self.cfg.symbol)
            out: list[dict] = []
            for cand in pb.candidates(ctx):
                if cand.signal_index <= self._qm_d1_last_logged:
                    continue  # already logged on an earlier scan
                sig = cand.outcome.signal
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
                self.journal.log_note(
                    f"mt5 [SHADOW] qm_d1 "
                    f"{'+' if touch['direction'] > 0 else '-'}1 "
                    f"touch {touch['touch_ts'].date()} "
                    f"QM {touch['qm_level']:.1f} head "
                    f"{touch['head']:.1f} neckline "
                    f"{touch['neckline']:.1f} "
                    f"-> {touch['r_net']:+.2f}R "
                    f"({touch['exit_reason']})"
                    + (f" tags:{','.join(touch['tags'])}"
                       if touch["tags"] else ""))
            if out:
                self._qm_d1_last_logged = max(
                    t["touch_index"] for t in out)
                if stats is not None:
                    stats["qm_d1_shadow"] = stats.get(
                        "qm_d1_shadow", 0) + len(out)
            return out
        except Exception as exc:
            self.journal.log_note(
                f"mt5: QM D1 shadow scan failed (never ordered): {exc}")
            return []

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
                    reply_markup: dict | None = None) -> None:
        """Send a text alert, with the chart photo when one is attached.

        Sender contract (in preference order): ``send_photo(png,
        caption)`` (TelegramSender), ``send_text(text)``, legacy
        ``send(text)`` duck-type.  A sender failure is journaled +
        deduped-console-printed (see _note_tg_failure), never raised
        into the tick loop.  ``reply_markup`` (inline keyboard) is
        passed through when the sender supports it —
        used by the Golden Swing Confirm request.
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
