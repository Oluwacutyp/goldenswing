"""Learning loop — the append-only trade journal and what reads it.

Every decision the system makes leaves a record the moment it
happens (the ICT Visualizer's audit pattern: a journal written at
signal time makes repaint claims checkable and doubles as the
backtest-vs-live reconciliation feed):

- ``open``  — a trade was placed (Execute or Paper), with the full
  context: score, backing factors, forecaster probability/rank,
  mode, playbook, and the market PHASE at entry.
- ``close`` — the trade ended: exit, reason, money P/L, R.
- ``skip``  — the owner tapped Skip: the setup existed, was judged,
  and was not taken.  Skips are training data too.
- ``note``  — refusals and other facts worth remembering.

:class:`PerformanceBook` joins opens to closes and aggregates per
(phase x playbook x mode) — the measured effectiveness matrix the
routing layer and future forecaster retrains consume.  The journal
is JSONL: one object per line, append-only, human-readable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TradeJournal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ writing
    def append(self, record: dict) -> dict:
        record = {"ts": _now_iso(), **record}
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return record

    def log_open(self, *, alert_id: str, instrument: str, mode: str,
                 playbook: str, phase: str, direction: int, score: float,
                 source: str, ticket: int | None, entry: float,
                 stop: float, volume: float, risk_money: float,
                 factors: list | tuple = (),
                 forecaster_prob: float | None = None,
                 forecaster_rank: float | None = None,
                 risk_atr: float | None = None,
                 intended_entry: float | None = None,
                 slippage: float | None = None,
                 broker_sl: float | None = None) -> dict:
        # Friction audit (owner directive 2026-10-22): every open
        # record carries its stop distance (|entry - stop|, price
        # units) so R geometry is auditable from the journal alone;
        # execution sites that know the intended entry also record
        # it plus the signed slippage (price units, adverse-positive:
        # direction * (fill - intended)) so M5-scale friction can be
        # measured against the stop it eats into.
        record = {
            "type": "open", "alert_id": alert_id, "instrument": instrument,
            "mode": mode, "playbook": playbook, "phase": phase,
            "direction": direction, "score": score, "source": source,
            "ticket": ticket, "entry": entry, "stop": stop,
            "stop_distance": abs(float(entry) - float(stop)),
            "volume": volume, "risk_money": risk_money,
            "factors": list(factors),
            "forecaster_prob": forecaster_prob,
            "forecaster_rank": forecaster_rank,
            "features": {
                "score": score, "direction": direction, "phase": phase,
                "playbook": playbook, "mode": mode,
                "forecaster_prob": forecaster_prob,
                "forecaster_rank": forecaster_rank,
                "risk_atr": risk_atr,
                "n_factors": len(factors),
            },
        }
        if broker_sl is not None:
            # Step 2: the broker stop that owns the trade, kept
            # distinct from the structural ``stop`` — close R divides
            # by this distance, so it must be on the record.
            record["broker_sl"] = float(broker_sl)
            record["broker_stop_distance"] = abs(
                float(entry) - float(broker_sl))
        if intended_entry is not None:
            record["intended_entry"] = float(intended_entry)
        if slippage is not None:
            record["slippage"] = float(slippage)
        return self.append(record)

    def log_close(self, *, alert_id: str, ticket: int | None = None,
                  exit_price: float, reason: str, pnl_money: float,
                  r_net: float, hold_bars: int = 0,
                  partial_taken: bool = False) -> dict:
        return self.append({
            "type": "close", "alert_id": alert_id, "ticket": ticket,
            "exit": exit_price, "reason": reason, "pnl_money": pnl_money,
            "r_net": r_net, "hold_bars": hold_bars,
            "partial_taken": partial_taken,
        })

    def log_skip(self, *, alert_id: str, instrument: str = "", mode: str = "",
                 direction: int = 0, score: float = 0.0,
                 phase: str = "unknown", playbook: str = "unknown") -> dict:
        return self.append({
            "type": "skip", "alert_id": alert_id, "instrument": instrument,
            "mode": mode, "direction": direction, "score": score,
            "phase": phase, "playbook": playbook,
        })

    def log_note(self, text: str, **context) -> dict:
        return self.append({"type": "note", "text": text, **context})

    # ------------------------------------------------------------ reading
    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def performance(self) -> "PerformanceBook":
        return PerformanceBook(self.records())


# ------------------------------------------------------------- aggregation
def _stats(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"trades": 0, "win_rate": 0.0, "total_r": 0.0, "avg_r": 0.0,
                "pf": 0.0, "total_pnl": 0.0}
    rs = [r["r_net"] for r in rows]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trades": n,
        "win_rate": 100.0 * len(wins) / n,
        "total_r": float(sum(rs)),
        "avg_r": float(sum(rs) / n),
        "pf": float(gross_win / gross_loss) if gross_loss > 0
              else float("inf"),
        "total_pnl": float(sum(r["pnl_money"] for r in rows)),
    }


class PerformanceBook:
    """Closed trades joined from journal records, with aggregates."""

    def __init__(self, records: list[dict]):
        opens = {r["alert_id"]: r for r in records if r.get("type") == "open"}
        self.rows: list[dict] = []
        for r in records:
            if r.get("type") != "close":
                continue
            o = opens.get(r["alert_id"], {})
            self.rows.append({
                "alert_id": r["alert_id"],
                "phase": o.get("phase", "unknown"),
                "playbook": o.get("playbook", "unknown"),
                "mode": o.get("mode", "unknown"),
                "source": o.get("source", "unknown"),
                "direction": o.get("direction", 0),
                "score": o.get("score", 0.0),
                "r_net": r["r_net"], "pnl_money": r["pnl_money"],
                "reason": r.get("reason", ""),
                "hold_bars": r.get("hold_bars", 0),
            })
        self.skips = [r for r in records if r.get("type") == "skip"]

    def aggregate(self) -> dict:
        groups: dict[str, list[dict]] = {}
        for row in self.rows:
            key = f"{row['phase']} × {row['playbook']} × {row['mode']}"
            groups.setdefault(key, []).append(row)
        return {
            "overall": _stats(self.rows),
            "by_phase_playbook_mode": {k: _stats(v)
                                       for k, v in sorted(groups.items())},
            "skips": len(self.skips),
        }


# -------------------------------------------------------- retraining data
def export_training_data(records: list[dict], path: str | Path) -> int:
    """Write (features -> outcome) rows for closed trades: the data a
    forecaster retrain consumes.  Label = 1 when the trade made money,
    with the realised R kept alongside — ranking and sizing models
    want the R, classifiers want the label.  Returns rows written."""
    book = PerformanceBook(records)
    opens = {r["alert_id"]: r for r in records if r.get("type") == "open"}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for row in book.rows:
            o = opens.get(row["alert_id"], {})
            features = dict(o.get("features") or {})
            if not features:
                features = {"score": row["score"],
                            "direction": row["direction"],
                            "phase": row["phase"],
                            "playbook": row["playbook"],
                            "mode": row["mode"]}
            fh.write(json.dumps({
                "alert_id": row["alert_id"], "features": features,
                "label": 1 if row["r_net"] > 0 else 0,
                "r_net": row["r_net"], "pnl_money": row["pnl_money"],
            }, default=str) + "\n")
            n += 1
    return n
