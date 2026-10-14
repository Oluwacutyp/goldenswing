"""Live gold H1 bar feed for the paper loop.

Implementations behind one interface (``warm_start``, ``fetch_new``,
``poll_status``, ``poll_interval``):

- :class:`OandaFeed` — polls OANDA v20 practice API for spot XAU/USD
  H1 bars. The live feed since the 2026-10-11 yfinance incident
  (phantom GC=F bars voided the first live signal). Every batch
  passes a sanity gate before reaching the loop.
- :class:`MultiOandaFeed` — the all-timeframe feed (D1/H4/H1/M15/M30/
  M5) behind the real-trader hierarchy; one OandaFeed per TF plus
  ``warm_start_all`` / ``fetch_new_all``.  The paper loop uses this.
- :class:`PollingFeed` — legacy yfinance GC=F feed. Kept for
  reference; not used by the live loop.
- :class:`ReplayFeed` — replays a historical frame bar-by-bar through
  the same interface; the rehearsal harness and the feed unit tests
  use it to prove the loop without touching the network.
"""
from .oanda import (MultiOandaFeed, MultiOandaFeedConfig, OandaFeed,
                    OandaFeedConfig, STACK_TFS)
from .poll import FeedConfig, PollingFeed, ReplayFeed

__all__ = ["FeedConfig", "PollingFeed", "ReplayFeed",
           "OandaFeed", "OandaFeedConfig",
           "MultiOandaFeed", "MultiOandaFeedConfig", "STACK_TFS"]
