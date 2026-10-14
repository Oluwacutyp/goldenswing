"""ExnessPaperExecutor — the $12 Exness-model paper broker.

Paper only: same signal pipeline as the live loop, but every fill is
priced and margined like an Exness Standard MT5 $12 account, NOT like
the $10,000 OANDA-model :class:`PaperExecutor` it extends:

- Starting balance $12.00, fixed 0.01 lot = 1 oz (sizing is fixed in
  the callback handler via ``RunnerConfig.fixed_lots`` — never the
  1%-risk sizer; see :mod:`goldenswing.execution.callbacks`).
- Spread $0.20/oz round-trip, commission $0.00 (vs $0.45/oz OANDA).
- Margin = notional / leverage, leverage 1:2000 (configurable).
  A trade that cannot meet margin is REFUSED (``insufficient margin``),
  never opened on credit.
- Stop-out at equity <= 0 (Exness Standard stop-out 0%): the position
  is closed at the stop-out price and the balance floors at $0.00
  (negative balance protection). Margin call at 60% is a WARNING
  only — it never closes a position, so this executor does not
  enforce it.

The margin / stop-out formulas are the same pure math as
:mod:`goldenswing.backtest.exness12` (``margin_required`` /
``stop_out_price`` there); they are repeated here rather than imported
because importing that module would drag the whole backtest +
``live.revival`` chain into the live loop. A parity test in
``tests/test_exness_paper.py`` proves the two implementations agree.

HONESTY LIMITS (read before trusting a CPI result from this loop):

- Constant leverage per run. Exness can restrict / reduce leverage
  around news / weekends / high-margin-requirement periods. A live
  $12 trade on the CPI spike could face higher margin + stop-out
  beyond what this executor models. The runner script records the
  assumed leverage in the Exness journal on every start so the
  result is read with its assumption attached.
- The feed is OANDA spot XAU as a proxy for Exness spot (same proxy
  the ``exness12`` backtest uses); basis is immaterial for timing
  logic, but fills are still simulated, never real.
- Stop-out is checked on H1 bar extremes (the loop's bar pump). A
  real account stops out tick-by-tick; intra-bar paths the H1 bar
  cannot see may stop a real trade out earlier or later.
"""
from __future__ import annotations

import math

from ..backtest.harness import ExitConfig
from .executor import AccountState, OrderRequest, OrderResult
from .paper import PaperExecutor, _Meta
from .risk import SymbolSpec, gold_spec

# Exness Standard MT5 specs (owner screenshot + verified stop-out).
START_BALANCE_12 = 12.00
LOT_OZ = 1.0  # 0.01 lot XAUUSD = 1 oz (1 lot = 100 oz)
LEVERAGE = 2000.0
SPREAD_PER_OZ = 0.20  # 0.2 interpreted as $0.20/oz round-trip
COMMISSION_PER_TRADE = 0.00
MARGIN_CALL_LEVEL = 60.0  # % margin level, warning only, never closes
STOP_OUT_LEVEL = 0.0  # % margin level, equity <= 0 closes
OZ_PER_LOT = 100.0


def margin_required(entry: float, lot_oz: float = LOT_OZ,
                    leverage: float = LEVERAGE) -> float:
    """Margin = notional / leverage. Notional = entry * oz.

    Unlimited leverage (inf) -> margin ~$0. 1:2 -> ~$2000 at $4000
    gold, so a $12 account CANNOT open even 0.01 lot at 1:2.
    """
    if leverage is None or (isinstance(leverage, float)
                            and math.isinf(leverage)):
        return 0.0
    if leverage <= 0:
        raise ValueError("leverage must be positive or inf for unlimited")
    return float(entry * lot_oz / leverage)


def stop_out_price(entry: float, direction: int, balance_before: float,
                   spread: float = SPREAD_PER_OZ,
                   lot_oz: float = LOT_OZ) -> float:
    """Price at which equity hits $0 (stop-out 0%).

    Equity(t) = balance_before + direction*(price-entry)*oz - spread_total,
    where spread_total = spread_per_oz * oz. Set = 0 and solve.
    For 1 oz: long stop-out = entry - (balance - spread).
    """
    spread_total = float(spread * lot_oz)
    room = float(balance_before - spread_total)
    if direction == 1:
        return float(entry - room / lot_oz)
    return float(entry + room / lot_oz)


class ExnessPaperExecutor(PaperExecutor):
    """PaperExecutor priced/margined as an Exness $12 account.

    Inherits the fill-at-next-bar-open + ExitStepper management from
    :class:`PaperExecutor`, with three Exness differences:

    1. ``spread_cost`` defaults to $0.20/oz (not $0.45/oz OANDA).
    2. :meth:`place_order` refuses trades the account cannot margin
       (``insufficient margin``) or cannot cover the spread for.
    3. :meth:`advance` checks equity stop-out BEFORE the exit stepper
       on every bar (stop-out wins over the structural stop, matching
       the ``exness12`` backtest's conservative ordering), and floors
       the balance at $0.00 on a blow-up.
    """

    def __init__(self, spec: SymbolSpec | None = None, *,
                 balance: float = START_BALANCE_12,
                 spread_cost: float = SPREAD_PER_OZ,
                 leverage: float = LEVERAGE,
                 exits: ExitConfig | None = None,
                 on_close=None, on_fill=None):
        if leverage is None or (isinstance(leverage, float)
                                and math.isinf(leverage)):
            self._leverage = float("inf")
        elif leverage <= 0:
            raise ValueError(
                f"leverage must be positive or inf (got {leverage})")
        else:
            self._leverage = float(leverage)
        super().__init__(spec or gold_spec(), balance=balance,
                         spread_cost=spread_cost, exits=exits,
                         on_close=on_close, on_fill=on_fill)

    # ------------------------------------------------------------ views
    @property
    def leverage(self) -> float:
        return self._leverage

    def margin_for(self, entry: float, volume_lots: float) -> float:
        """Margin for ``volume_lots`` at ``entry`` under this leverage."""
        return margin_required(entry, lot_oz=volume_lots * OZ_PER_LOT,
                               leverage=self._leverage)

    def account_state(self) -> AccountState:
        """Balance + equity + free margin (equity - used margin).

        The base class reports free_margin = equity; here free margin
        subtracts the margin actually used by open positions, because
        that is the number a margin call / stop-out is judged against.
        """
        state = super().account_state()
        used = 0.0
        for pos in self._tracked.values():
            used += self.margin_for(pos.entry_price, pos.volume)
        return AccountState(balance=state.balance, equity=state.equity,
                            free_margin=state.equity - used,
                            currency=state.currency)

    # -------------------------------------------------------- Executor API
    def place_order(self, request: OrderRequest) -> OrderResult:
        if self._balance <= 0:
            return OrderResult(
                False, message="account blown (balance <= 0) — "
                               "Exness stop-out already hit, no new risk")
        oz = float(request.volume) * OZ_PER_LOT
        spread_total = float(self._half_spread * 2.0 * oz)
        if self._balance <= spread_total:
            return OrderResult(
                False, message="insufficient_for_spread: balance "
                               f"{self._balance:.2f} <= spread "
                               f"{spread_total:.2f}")
        entry_ref = float(request.entry_ref or 0.0)
        if entry_ref > 0:
            margin = self.margin_for(entry_ref, request.volume)
            equity = self.account_state().equity
            if equity < margin:
                return OrderResult(
                    False, message="insufficient margin: equity "
                                   f"{equity:.2f} < margin {margin:.2f} "
                                   f"(entry {entry_ref:.1f}, "
                                   f"{request.volume:.2f} lots, "
                                   f"leverage 1:{self._leverage:.0f})")
        return super().place_order(request)

    # ------------------------------------------------------------ bar pump
    def advance(self, bar: dict) -> list[dict]:
        """Base fill/step pump + equity stop-out before the stepper.

        Mirrors :meth:`PaperExecutor.advance` (fills at the open,
        manual closes at the open, stepper management) with one
        addition: for every open position, if the bar's adverse
        extreme reaches the equity stop-out price, the position is
        closed at that price (or the open, if gapped through) with
        reason ``stop_out_blowup`` and the balance floors at $0.00.
        Stop-out is checked BEFORE the stepper so it always wins
        over the structural stop on the same bar.
        """
        idx = int(bar["index"])
        o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))
        self._last_close[self._spec.name] = c
        events: list[dict] = []

        # 1. manual closes at the open (base behaviour)
        for ticket, volume in self._pending_closes:
            pos = self._tracked.get(ticket)
            if pos is None:
                continue
            raw = o
            self._close_volume(pos, volume, self._exit_price(raw, pos.direction),
                               "manual", idx, events)
        self._pending_closes.clear()

        # 2. pending orders fill at the open (base behaviour)
        for ticket, req in self._pending_orders:
            fill = o + req.direction * self._half_spread
            pos = self._stepper.open(ticket=ticket, request=req,
                                     fill_price=fill, entry_bar=idx)
            self._tracked[ticket] = pos
            self._meta[ticket] = _Meta(
                alert_id=req.alert_id, magic=req.magic, open_time=bar.get("time"),
                initial_risk_money=self._money(pos.risk_price, pos.volume))
            events.append({"kind": "fill", "ticket": ticket,
                           "price": fill, "volume": pos.volume})
            if self.on_fill is not None:
                self.on_fill([p for p in self.positions()
                              if p.ticket == ticket][0])
        self._pending_orders.clear()

        # 3. stop-out first, then the stepper, per open position
        for ticket in list(self._tracked):
            pos = self._tracked[ticket]
            so_price = self._stop_out_price(pos)
            if so_price is not None:
                d = pos.direction
                breached = (d == 1 and l <= so_price) or \
                           (d == -1 and h >= so_price)
                if breached:
                    raw = so_price
                    if (d == 1 and o < so_price) or \
                            (d == -1 and o > so_price):
                        raw = o  # gapped through: take the open
                    price = self._exit_price(raw, d)
                    self._close_volume(pos, None, price, "stop_out_blowup",
                                       idx, events)
                    # Negative balance protection: never below $0.
                    if self._balance < 0:
                        self._balance = 0.0
                    continue
            for ev in self._stepper.step(pos, bar, idx):
                self._price_event(pos, ev, bar, events)
        return events

    def _stop_out_price(self, pos) -> float | None:
        """Raw stop-out price for an open position.

        Closing the remaining volume at this raw price takes the
        account to exactly $0: the fill already paid half the spread
        (entry price embeds it), so only the exit half remains —
        ``raw = entry - d*(balance - half_spread*oz)/oz``. Uses the
        CURRENT balance (banked partials included) and remaining
        volume, so a position that banked profit stops out later.
        """
        oz = float(pos.volume) * OZ_PER_LOT
        if oz <= 0:
            return None
        return float(pos.entry_price - pos.direction
                     * (self._balance - self._half_spread * oz) / oz)
