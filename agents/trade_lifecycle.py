"""
trade_lifecycle.py - 5-Stage Automated Trade Lifecycle State Machine

Manages an active trade bar-by-bar (both in live WebSocket trading and in
backtest.py) using one shared implementation:
- STATE 0 (`INITIAL`): 100% size open, SL at -1.0R
- STATE 1 (`RETEST_SHIELD` at +1.3R): Close 30% at +1.3R; move remaining 70% SL to -0.5R
- STATE 2 (`PROFIT_LOCK` at +2.2R): Close 30% at +2.2R; move remaining 40% SL to +0.20R
- STATE 3 (`CHANDELIER_RUNNER` after +2.2R): Trail remaining 40% with ratchet Chandelier Exit
- STATE 4 (`TIME_STOP`): Exit at market if still in STATE 0 after N candles without reaching +0.6R
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("TradeLifecycle")

TAKER_FEE_PER_SIDE = 0.0005  # 0.05% Binance Futures taker fee
SLIPPAGE_ROUNDTRIP = 0.0002  # 0.02% slippage reserve (total 0.12% round-trip)

STATE_INITIAL = "INITIAL"
STATE_RETEST_SHIELD = "RETEST_SHIELD"
STATE_PROFIT_LOCK = "PROFIT_LOCK"
STATE_CHANDELIER_RUNNER = "CHANDELIER_RUNNER"
STATE_TIME_STOP = "TIME_STOP"


@dataclass
class TradePosition:
    trade_id: str
    side: str  # "BUY" | "SELL" | "LONG" | "SHORT"
    entry_price: float
    initial_sl: float
    notional_usd: float
    timeframe: str = "15m"
    risk_usd: float = 0.0
    horizon: str = "INTRADAY"
    regime_mode: str = "EXPANSION"
    time_stop_candles: Optional[int] = None
    state: str = STATE_INITIAL
    remaining_fraction: float = 1.0
    current_sl: float = field(init=False)
    r_distance: float = field(init=False)
    highest_high: float = field(init=False)
    lowest_low: float = field(init=False)
    candles_held: int = 0
    candles_to_tp1: Optional[int] = None
    reached_06r: bool = False
    intermediate_ratchet_hit: bool = False
    mfe_r: float = 0.0
    mae_r: float = 0.0
    realized_r: float = 0.0
    realized_gross_usd: float = 0.0
    fees_paid_usd: float = 0.0
    closed: bool = False
    exit_reason: str = ""
    exit_time: str = ""

    def __post_init__(self) -> None:
        self.side = "LONG" if str(self.side).upper() in ("BUY", "LONG") else "SHORT"
        self.regime_mode = str(self.regime_mode).upper() if self.regime_mode else "EXPANSION"
        self.current_sl = float(self.initial_sl)
        self.r_distance = max(abs(float(self.entry_price) - float(self.initial_sl)), 1e-8)
        self.highest_high = float(self.entry_price)
        self.lowest_low = float(self.entry_price)
        # Entry taker fee + half slippage reserve
        self.fees_paid_usd += self.notional_usd * (TAKER_FEE_PER_SIDE + SLIPPAGE_ROUNDTRIP / 2.0)

    @property
    def is_long(self) -> bool:
        return self.side == "LONG"

    def price_at_r(self, r_mult: float) -> float:
        return self.entry_price + r_mult * self.r_distance if self.is_long else self.entry_price - r_mult * self.r_distance

    def r_at_price(self, price: float) -> float:
        return (price - self.entry_price) / self.r_distance if self.is_long else (self.entry_price - price) / self.r_distance

    def _close_fraction(self, fraction: float, exit_price: float) -> None:
        frac = min(fraction, self.remaining_fraction)
        if frac <= 0:
            return
        r_mult = self.r_at_price(exit_price)
        self.realized_r += frac * r_mult
        pct_return = ((exit_price - self.entry_price) / self.entry_price) if self.is_long else ((self.entry_price - exit_price) / self.entry_price)
        self.realized_gross_usd += (self.notional_usd * frac) * pct_return
        self.fees_paid_usd += (self.notional_usd * frac) * (TAKER_FEE_PER_SIDE + SLIPPAGE_ROUNDTRIP / 2.0)
        self.remaining_fraction = max(0.0, round(self.remaining_fraction - frac, 6))

    def step_candle(
        self,
        high: float,
        low: float,
        close: float,
        atr_14: float,
        candle_time: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Advance the lifecycle state machine by one closed candle.
        Returns a list of state-transition / exit event dicts for Telegram & logging.
        """
        if self.closed:
            return []

        events: List[Dict[str, Any]] = []
        self.candles_held += 1
        self.highest_high = max(self.highest_high, float(high))
        self.lowest_low = min(self.lowest_low, float(low))

        # Update MFE / MAE in R
        fav_r = self.r_at_price(high if self.is_long else low)
        adv_r = self.r_at_price(low if self.is_long else high)
        self.mfe_r = max(self.mfe_r, fav_r)
        self.mae_r = min(self.mae_r, adv_r)
        threshold_r = 0.5 if self.regime_mode == "ROTATIONAL" else 0.6
        if fav_r >= threshold_r:
            self.reached_06r = True

        # 1. Protective Stop Check first (conservative intrabar ordering)
        sl_hit = (low <= self.current_sl) if self.is_long else (high >= self.current_sl)
        if sl_hit:
            reason_map = {
                STATE_INITIAL: "INITIAL_SL",
                STATE_RETEST_SHIELD: "RETEST_SHIELD_SL",
                STATE_PROFIT_LOCK: "PROFIT_LOCK_SL",
                STATE_CHANDELIER_RUNNER: "CHANDELIER_TRAIL_STOP",
            }
            exit_reason = reason_map.get(self.state, "STOP_LOSS")
            self._close_fraction(self.remaining_fraction, self.current_sl)
            self.closed = True
            self.exit_reason = exit_reason
            self.exit_time = candle_time
            events.append(self._build_event("CLOSED", exit_reason, self.current_sl))
            return events

        if self.regime_mode == "ROTATIONAL":
            # Mode A: "ROTATIONAL" (20 <= ADX < 26 — Quick Harvest, No Runner Give-back)
            # STATE 1 at +1.1R: Close 50% of position. Move SL of remaining 50% to +0.15R
            tp1_price = self.price_at_r(1.1)
            tp1_hit = (high >= tp1_price) if self.is_long else (low <= tp1_price)
            if self.state == STATE_INITIAL and tp1_hit:
                self._close_fraction(0.50, tp1_price)
                self.candles_to_tp1 = self.candles_held
                self.current_sl = self.price_at_r(0.15)
                self.state = STATE_RETEST_SHIELD
                events.append(self._build_event("TRANSITION", "TP1_RETEST_SHIELD", tp1_price))

            # STATE 2 at +1.8R: Close remaining 50% (100% full exit at target, no runner)
            tp2_price = self.price_at_r(1.8)
            tp2_hit = (high >= tp2_price) if self.is_long else (low <= tp2_price)
            if self.state == STATE_RETEST_SHIELD and tp2_hit:
                self._close_fraction(self.remaining_fraction, tp2_price)
                self.state = STATE_PROFIT_LOCK
                self.closed = True
                self.exit_reason = "TP2_PROFIT_LOCK"
                self.exit_time = candle_time
                events.append(self._build_event("CLOSED", "TP2_PROFIT_LOCK", tp2_price))

        else:
            # Mode B: "EXPANSION" (26 <= ADX <= 35 — Fat-Tail Trend Runner)
            # STATE 1 at +1.4R: Close 35%. Move SL of remaining 65% to -0.20R
            tp1_price = self.price_at_r(1.4)
            tp1_hit = (high >= tp1_price) if self.is_long else (low <= tp1_price)
            if self.state == STATE_INITIAL and tp1_hit:
                self._close_fraction(0.35, tp1_price)
                self.candles_to_tp1 = self.candles_held
                self.current_sl = self.price_at_r(-0.20)
                self.state = STATE_RETEST_SHIELD
                events.append(self._build_event("TRANSITION", "TP1_RETEST_SHIELD", tp1_price))

            # Intermediate Ratchet at +1.7R: Move SL of remaining 65% to +0.40R
            ratchet_price = self.price_at_r(1.7)
            ratchet_hit = (high >= ratchet_price) if self.is_long else (low <= ratchet_price)
            if self.state == STATE_RETEST_SHIELD and not self.intermediate_ratchet_hit and ratchet_hit:
                self.intermediate_ratchet_hit = True
                ratchet_sl = self.price_at_r(0.40)
                if self.is_long:
                    self.current_sl = max(self.current_sl, ratchet_sl)
                else:
                    self.current_sl = min(self.current_sl, ratchet_sl)
                events.append(self._build_event("TRANSITION", "INTERMEDIATE_RATCHET_LOCK", ratchet_price))

            # STATE 2 at +2.4R: Close 30% (leaving 35% runner). Move SL to +1.00R and trail with Chandelier
            tp2_price = self.price_at_r(2.4)
            tp2_hit = (high >= tp2_price) if self.is_long else (low <= tp2_price)
            if self.state == STATE_RETEST_SHIELD and tp2_hit:
                self._close_fraction(0.30, tp2_price)
                self.current_sl = self.price_at_r(1.00)
                self.state = STATE_PROFIT_LOCK
                events.append(self._build_event("TRANSITION", "TP2_PROFIT_LOCK", tp2_price))
                self.state = STATE_CHANDELIER_RUNNER

            # STATE 3: Ratchet Chandelier Exit on candle close for remaining 35%
            if self.state == STATE_CHANDELIER_RUNNER and atr_14 > 0 and not self.closed:
                if self.is_long:
                    chandelier = self.highest_high - 2.2 * float(atr_14)
                    if chandelier > self.current_sl and chandelier < close:
                        self.current_sl = round(chandelier, 2)
                else:
                    chandelier = self.lowest_low + 2.2 * float(atr_14)
                    if chandelier < self.current_sl and chandelier > close:
                        self.current_sl = round(chandelier, 2)

        # 5. STATE 4: TIME_STOP (Horizon-adaptive)
        if self.time_stop_candles is not None and self.time_stop_candles > 0:
            max_stale = self.time_stop_candles
        elif self.timeframe.lower() == "5m" or str(self.horizon).upper() == "SCALP":
            max_stale = 20
        elif self.timeframe.lower() == "1h" or str(self.horizon).upper() == "SWING":
            max_stale = 12
        else:
            max_stale = 16
        if self.state == STATE_INITIAL and not self.reached_06r and self.candles_held >= max_stale:
            self._close_fraction(self.remaining_fraction, float(close))
            self.state = STATE_TIME_STOP
            self.closed = True
            self.exit_reason = "TIME_STOP"
            self.exit_time = candle_time
            events.append(self._build_event("CLOSED", "TIME_STOP", float(close)))

        return events

    def force_close(self, price: float, reason: str = "BACKTEST_END", candle_time: str = "") -> Dict[str, Any]:
        """Close any open remainder (e.g. end of backtest window)."""
        if not self.closed and self.remaining_fraction > 0:
            self._close_fraction(self.remaining_fraction, float(price))
            self.closed = True
            self.exit_reason = reason
            self.exit_time = candle_time
        return self.to_lifecycle_summary()

    def _build_event(self, kind: str, reason: str, price: float) -> Dict[str, Any]:
        return {
            "kind": kind,
            "trade_id": self.trade_id,
            "side": self.side,
            "state": self.state,
            "reason": reason,
            "price": round(float(price), 2),
            "new_sl": round(float(self.current_sl), 2),
            "remaining_pct": round(self.remaining_fraction * 100.0, 1),
            "net_pnl_usd": round(self.realized_gross_usd - self.fees_paid_usd, 2),
            "net_r_multiple": self._net_r(),
        }

    def _net_r(self) -> float:
        if self.risk_usd > 0:
            return round((self.realized_gross_usd - self.fees_paid_usd) / self.risk_usd, 4)
        fee_r = (self.fees_paid_usd / (self.notional_usd * (self.r_distance / self.entry_price))) if self.notional_usd > 0 else 0.0
        return round(self.realized_r - fee_r, 4)

    def to_lifecycle_summary(self) -> Dict[str, Any]:
        net_pnl = round(self.realized_gross_usd - self.fees_paid_usd, 2)
        return {
            "trade_horizon": self.horizon,
            "exit_time": self.exit_time,
            "exit_state": self.state,
            "exit_reason": self.exit_reason or self.state,
            "mfe_r": round(self.mfe_r, 4),
            "mae_r": round(self.mae_r, 4),
            "candles_to_tp1": self.candles_to_tp1,
            "total_candles_held": self.candles_held,
            "net_pnl_usd": net_pnl,
            "net_r_multiple": self._net_r(),
            "fees_paid_usd": round(self.fees_paid_usd, 2),
        }

