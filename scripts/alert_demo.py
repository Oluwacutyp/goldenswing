#!/usr/bin/env python3
"""Phase-7 demo: real signals -> charts -> outbox alerts.

Runs the production detector config (Combo A {relstr, volume, phases}
+ Phase-2 flags) over the cached 2y GC=F H1 series, then:

- renders the best Golden signal in all 4 chart styles -> samples/
- pushes 3 Golden signals through the AlertService with BOTH senders
  forced to OFFLINE OUTBOX mode (credentials are stripped from the
  environment for this process — nothing is sent anywhere)

Run:  python3 scripts/alert_demo.py
"""
import os
import sys
from pathlib import Path

# The demo is offline BY DESIGN: strip any ambient alert credentials
# so the senders fall back to the outbox instead of really sending.
for var in ("GS_TELEGRAM_BOT_TOKEN", "GS_TELEGRAM_CHAT_ID",
            "GS_SMTP_HOST", "GS_ALERT_EMAIL_TO"):
    os.environ.pop(var, None)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from goldenswing import charts  # noqa: E402
from goldenswing.alerts import AlertService, EmailSender, TelegramSender  # noqa: E402
from goldenswing.backtest.data import load_gold_h1, load_peers  # noqa: E402
from goldenswing.backtest.validation import COMBO_A  # noqa: E402
from goldenswing.backtest.ablation import P2_FLAGS  # noqa: E402
from goldenswing.features.engine import FeatureEngine  # noqa: E402
from goldenswing.golden.detector import DetectorConfig, GoldenSwingDetector  # noqa: E402
from goldenswing.reader.reader import MarketReader  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "samples"
OUTBOX = SAMPLES / "outbox"


def forecast_note_for(signal, probs: pd.DataFrame | None) -> charts.ForecastNote | None:
    """Ensemble context from the Phase-6 OOF probabilities, if the
    signal bar has an out-of-fold row."""
    if probs is None or signal.index not in probs.index:
        return None
    col = "ens_up" if signal.direction == 1 else "ens_dn"
    series = probs[col].dropna()
    if len(series) == 0:
        return None
    p = float(probs.loc[signal.index, col])
    rank = float((series <= p).mean() * 100.0)
    word = "up" if signal.direction == 1 else "down"
    return charts.ForecastNote(
        prob_direction=p, rank_pct=rank,
        text=f"ensemble 3-day big-{word}-move prob")


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    df = load_gold_h1()
    peers = load_peers()
    print(f"bars: {len(df)}  {df.index[0]} .. {df.index[-1]}")

    states = MarketReader().read(df)
    engine = FeatureEngine()
    fsets = engine.compute(df, states, peers)
    for s, fs in zip(states, fsets):
        s.features = fs

    detector = GoldenSwingDetector(DetectorConfig(
        emit_floor=35.0, golden_threshold=60.0,
        feature_families=COMBO_A, **P2_FLAGS))
    signals = detector.scan(states)
    golden = [s for s in signals if s.is_golden]
    print(f"signals: {len(signals)}  golden: {len(golden)}")
    if not golden:
        print("no golden signals — demo cannot proceed")
        return

    probs = None
    probs_path = REPO / "data" / "forecast_oof_probs.csv"
    if probs_path.exists():
        probs = pd.read_csv(probs_path, index_col=0)

    # ---- 4 style samples from the highest-scoring golden signal -----
    best = max(golden, key=lambda s: s.score)
    print(f"best signal: bar {best.index} {best.timestamp} "
          f"dir {best.direction:+d} score {best.score:.0f} ({best.kind})")
    for style in charts.STYLES:
        path = SAMPLES / f"chart_{style}.png"
        charts.render_to_file(
            path, df, states[best.index], best, style=style,
            features=fsets, forecast=forecast_note_for(best, probs),
            mode="swing")
        print(f"  sample [{style}]: {path} ({path.stat().st_size} bytes)")

    # ---- 3 alerts through the service (outbox mode) ------------------
    picks = [golden[int(f * (len(golden) - 1))] for f in (0.2, 0.55, 0.9)]
    picks = sorted({s.index: s for s in picks}.values(),
                   key=lambda s: s.index)
    service = AlertService(
        senders=[TelegramSender(outbox_dir=OUTBOX),
                 EmailSender(outbox_dir=OUTBOX)],
        cooldown_bars=24)  # default chart_style="signal" (phone-first)
    for sig in picks:
        alert = service.process(
            df, states[sig.index], sig, instrument="XAUUSD", mode="swing",
            features=fsets, forecast=forecast_note_for(sig, probs))
        if alert is None:
            print(f"  alert bar {sig.index}: SUPPRESSED by cooldown")
            continue
        modes = ", ".join(f"{r.channel}={r.mode}"
                          f"({'sent' if r.delivered else 'outbox'})"
                          for r in alert.send_results)
        print(f"  alert {alert.alert_id} bar {sig.index} "
              f"score {sig.score:.0f}: {modes}")
    print(f"\noutbox contents: {sorted(p.name for p in OUTBOX.iterdir())}")


if __name__ == "__main__":
    main()
