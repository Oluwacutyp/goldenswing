"""Persisted live config for the MT5 engine (``data/mt5_config.json``).

The Telegram ``/set`` surface may change ONLY the safe keys listed
here.  Everything else — live-trading permission, tokens,
credentials, paths, code — is not a key at all: ``parse_set`` rejects
it, and the error message never echoes the supplied value (a
``/set token <value>`` attempt must not print the value back).

File format (all keys optional; missing keys fall back to defaults)::

    {"fixed_lots": null | 0.01-1.0, "max_layers": 0-3,
     "manipulation_filter": true | false,
     "pending_expiry_bars": 1-24,
     "disabled_playbooks": ["<name>", ...], "paused": true | false}

The runner loads this file at startup (when it exists) and
``/set`` applies + persists in-process — the file is the durable
record, the runner's config is the live one.  A corrupt file falls
back to defaults rather than refusing to start.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

#: Playbooks the engine may ORDER from (validated set).
#: Executing playbooks: evaluated AND eligible to order (subject to
#: the per-playbook kill switch and every live gate — hierarchy,
#: confluence floors, regime, news).  Owner directive 2026-10-22:
#: the whole roster executes on the demo; backtests map where a
#: setup historically worked, they never gate this list.  Forward
#: live-demo performance is the verdict.
#:
#: FUNNEL PATCH §3: these lists are DERIVED from the one playbook
#: registry (``playbook_registry.PLAYBOOK_REGISTRY``) — the runner's
#: roster, the native-clock routes and this manageable set can no
#: longer drift apart.  The registry carries all 18 playbooks,
#: including the four range playbooks an earlier hand-maintained
#: copy of this list silently omitted (they could not be killed
#: via /set at all, and _clean dropped their persisted kill
#: entries without a trace).
from .playbook_registry import PLAYBOOK_NAMES as _REGISTRY_NAMES

EXECUTE_PLAYBOOK_NAMES: tuple[str, ...] = _REGISTRY_NAMES
#: Shadow playbooks: evaluated + logged, never ordered.  EMPTY since
#: the owner directive of 2026-10-22 (the roster is live; a playbook
#: leaves execution via the kill switch, not a shadow tier).
SHADOW_PLAYBOOK_NAMES: tuple[str, ...] = ()
ALL_PLAYBOOK_NAMES: tuple[str, ...] = (
    EXECUTE_PLAYBOOK_NAMES + SHADOW_PLAYBOOK_NAMES)

FIXED_LOTS_MIN = 0.01
FIXED_LOTS_MAX = 1.0
MAX_LAYERS_MIN = 0
MAX_LAYERS_MAX = 3
PENDING_EXPIRY_MIN = 1
PENDING_EXPIRY_MAX = 24

#: Keys ``/set`` accepts (``playbook.<name>`` handled separately).
SAFE_SCALAR_KEYS = ("fixed_lots", "max_layers", "manipulation_filter",
                    "pending_expiry_bars")


def defaults() -> dict:
    """Default live config (fixed_lots None = risk-% sizing)."""
    return {"fixed_lots": None, "max_layers": 3,
            "manipulation_filter": True, "pending_expiry_bars": 4,
            "disabled_playbooks": [], "paused": False}


def _clean(values: dict) -> dict:
    """Merge stored values over defaults, dropping unknown keys."""
    out = defaults()
    if not isinstance(values, dict):
        return out
    if values.get("fixed_lots") is None:
        out["fixed_lots"] = None
    else:
        try:
            lots = float(values.get("fixed_lots"))
        except (TypeError, ValueError):
            lots = None
        if lots is not None and FIXED_LOTS_MIN <= lots <= FIXED_LOTS_MAX:
            out["fixed_lots"] = lots
    for key in ("max_layers", "pending_expiry_bars"):
        try:
            val = int(values.get(key))
        except (TypeError, ValueError):
            continue
        lo, hi = ((MAX_LAYERS_MIN, MAX_LAYERS_MAX)
                  if key == "max_layers"
                  else (PENDING_EXPIRY_MIN, PENDING_EXPIRY_MAX))
        if lo <= val <= hi:
            out[key] = val
    if isinstance(values.get("manipulation_filter"), bool):
        out["manipulation_filter"] = values["manipulation_filter"]
    if isinstance(values.get("paused"), bool):
        out["paused"] = values["paused"]
    disabled = values.get("disabled_playbooks")
    if isinstance(disabled, list):
        out["disabled_playbooks"] = sorted(
            {str(n) for n in disabled if str(n) in ALL_PLAYBOOK_NAMES})
    return out


def unknown_disabled_names(values) -> list[str]:
    """Persisted ``disabled_playbooks`` entries that are NOT real
    playbooks — the entries :func:`_clean` will drop.  Dropping is
    correct (a typo must not invent a playbook), but it must never
    be silent again: the runner journals this list at startup so a
    vanished kill-switch entry is a visible fact, not a mystery."""
    if not isinstance(values, dict):
        return []
    disabled = values.get("disabled_playbooks")
    if not isinstance(disabled, list):
        return []
    return sorted({str(n) for n in disabled
                   if str(n) not in ALL_PLAYBOOK_NAMES})


def _parse_bool(raw: str) -> bool | None:
    low = raw.strip().lower()
    if low in ("on", "true", "1", "yes"):
        return True
    if low in ("off", "false", "0", "no"):
        return False
    return None


def parse_set(key: str, raw_value: str) -> tuple[dict | None, str | None]:
    """Validate one ``/set <key> <value>``.  Returns (change, error).

    ``change`` is a dict the runner applies (``{"field": ..., "value":
    ...}`` or ``{"field": "playbook", "name": ..., "value": bool}``).
    ``error`` never echoes the supplied value.
    """
    key = (key or "").strip().lower()
    if key in ("pause", "paused"):
        return None, ("pause is not a /set key — use the Pause / Resume "
                      "buttons (confirm required).")
    if key in ("allow_live", "live", "allow-live", "trading",
               "live_trading"):
        return None, ("live-trading permission can never be changed "
                      "from Telegram.")
    if key.startswith("playbook."):
        name = key.split(".", 1)[1]
        if name not in ALL_PLAYBOOK_NAMES:
            return None, (f"unknown playbook {name!r} — /config lists "
                          f"the playbooks that exist.")
        enabled = _parse_bool(raw_value)
        if enabled is None:
            return None, (f"playbook.{name} takes on/off.")
        return {"field": "playbook", "name": name, "value": enabled}, None
    if key not in SAFE_SCALAR_KEYS:
        return None, (f"{key!r} is not a safe /set key.  Safe keys: "
                      f"fixed_lots, max_layers, manipulation_filter, "
                      f"pending_expiry_bars, playbook.<name>.")
    if key == "fixed_lots":
        low = raw_value.strip().lower()
        if low in ("auto", "none", "off"):
            return {"field": "fixed_lots", "value": None}, None
        try:
            lots = float(raw_value)
        except ValueError:
            return None, "fixed_lots takes a number (0.01–1.0) or auto."
        if not FIXED_LOTS_MIN <= lots <= FIXED_LOTS_MAX:
            return None, (f"fixed_lots out of range "
                          f"({FIXED_LOTS_MIN}–{FIXED_LOTS_MAX}).")
        return {"field": "fixed_lots", "value": lots}, None
    if key == "max_layers":
        try:
            val = int(raw_value)
        except ValueError:
            return None, "max_layers takes a whole number (0–3)."
        if not MAX_LAYERS_MIN <= val <= MAX_LAYERS_MAX:
            return None, (f"max_layers out of range "
                          f"({MAX_LAYERS_MIN}–{MAX_LAYERS_MAX}).")
        return {"field": "max_layers", "value": val}, None
    if key == "manipulation_filter":
        enabled = _parse_bool(raw_value)
        if enabled is None:
            return None, "manipulation_filter takes on/off."
        return {"field": "manipulation_filter", "value": enabled}, None
    if key == "pending_expiry_bars":
        try:
            val = int(raw_value)
        except ValueError:
            return None, "pending_expiry_bars takes a whole number (1–24)."
        if not PENDING_EXPIRY_MIN <= val <= PENDING_EXPIRY_MAX:
            return None, (f"pending_expiry_bars out of range "
                          f"({PENDING_EXPIRY_MIN}–{PENDING_EXPIRY_MAX}).")
        return {"field": "pending_expiry_bars", "value": val}, None
    return None, f"{key!r} is not a safe /set key."  # unreachable


def describe_change(change: dict) -> str:
    """Human-readable form of a parsed change (parsed values only)."""
    if change.get("field") == "playbook":
        return (f"playbook.{change['name']} = "
                f"{'on' if change['value'] else 'off'}")
    if change.get("field") == "fixed_lots":
        return ("fixed_lots = auto (risk-% sizing)" if change["value"] is None
                else f"fixed_lots = {change['value']}")
    return f"{change.get('field')} = {change.get('value')}"


class LiveConfigStore:
    """Atomic JSON store for the MT5 live config."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict:
        try:
            return _clean(json.loads(self.path.read_text()))
        except (OSError, json.JSONDecodeError):
            return defaults()

    def save(self, values: dict) -> None:
        clean = _clean(values)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".",
                                   dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(clean, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ---------------------------------------------------------- runner glue
    def apply_to(self, cfg) -> None:
        """Apply persisted values onto an ``MT5RunnerConfig``."""
        values = self.load()
        cfg.fixed_lots = values["fixed_lots"]
        cfg.layer_max_layers = values["max_layers"]
        cfg.manipulation_filter = values["manipulation_filter"]
        cfg.pending_expiry_bars = values["pending_expiry_bars"]
        cfg.disabled_playbooks = tuple(values["disabled_playbooks"])
        cfg.paused = values["paused"]

    def snapshot_from(self, cfg) -> dict:
        """Current live values from an ``MT5RunnerConfig``."""
        disabled = set(getattr(cfg, "disabled_playbooks", ()) or ())
        return {
            "fixed_lots": getattr(cfg, "fixed_lots", None),
            "max_layers": getattr(cfg, "layer_max_layers", 3),
            "manipulation_filter": getattr(
                cfg, "manipulation_filter", True),
            "pending_expiry_bars": getattr(
                cfg, "pending_expiry_bars", 4),
            "disabled_playbooks": sorted(
                n for n in disabled if n in ALL_PLAYBOOK_NAMES),
            "paused": bool(getattr(cfg, "paused", False)),
            "playbooks": {name: name not in disabled
                          for name in ALL_PLAYBOOK_NAMES},
        }
