"""
futures_context.py - Pluggable Binance USDT-M Futures Derivatives Context (100% free CCXT)

Provides data Spot candles cannot show:
- Funding Rate (crowded longs / shorts)
- Mark Price vs Index Price basis
- Open Interest

Only called AFTER a BUY/SELL technical setup exists (inside risk_arbiter), so
it never adds API noise on HOLD candles. Any network failure returns
{"available": False} so the pipeline is never blocked.
"""

import logging
from typing import Any, Dict, Optional, Protocol

logger = logging.getLogger("FuturesContext")


def to_futures_symbol(symbol: str) -> str:
    """'BTC/USDT' -> 'BTC/USDT:USDT' (CCXT unified linear perpetual symbol)."""
    if ":" in symbol:
        return symbol
    quote = symbol.split("/")[-1]
    return f"{symbol}:{quote}"


class DerivativesProvider(Protocol):
    def get_context(self, symbol: str) -> Dict[str, Any]:
        ...


class CCXTBinanceDerivativesProvider:
    """Free public Binance USDT-M Futures data via ccxt.binanceusdm (no API key needed)."""

    def __init__(self, exchange: Optional[Any] = None, timeout_ms: int = 8000):
        self._exchange = exchange
        self._timeout_ms = timeout_ms

    def _get_exchange(self):
        if self._exchange is None:
            import ccxt
            self._exchange = ccxt.binanceusdm({"enableRateLimit": True, "timeout": self._timeout_ms})
        return self._exchange

    def get_context(self, symbol: str = "BTC/USDT") -> Dict[str, Any]:
        fut_symbol = to_futures_symbol(symbol)
        context: Dict[str, Any] = {"available": False, "symbol": fut_symbol}
        try:
            ex = self._get_exchange()
            fr = ex.fetch_funding_rate(fut_symbol)
            mark = float(fr.get("markPrice") or 0.0)
            index = float(fr.get("indexPrice") or 0.0)
            context.update({
                "available": True,
                "funding_rate": float(fr.get("fundingRate") or 0.0),
                "mark_price": mark,
                "index_price": index,
                "basis_pct": ((mark - index) / index * 100.0) if index > 0 else 0.0,
            })
        except Exception as e:
            logger.info(f"Funding rate unavailable ({type(e).__name__}); continuing without derivatives data.")
            return context

        try:
            oi = ex.fetch_open_interest(fut_symbol)
            context["open_interest"] = float(oi.get("openInterestAmount") or oi.get("baseVolume") or 0.0)
        except Exception as e:
            logger.info(f"Open interest unavailable ({type(e).__name__}).")
        return context


class NullDerivativesProvider:
    """Offline / backtest provider: derivatives data unavailable (neutral scoring)."""

    def get_context(self, symbol: str = "BTC/USDT") -> Dict[str, Any]:
        return {"available": False, "symbol": to_futures_symbol(symbol)}


_DEFAULT_PROVIDER: Optional[DerivativesProvider] = None


def fetch_futures_market_context(symbol: str = "BTC/USDT") -> Dict[str, Any]:
    """Module-level helper used by risk_arbiter (easy to patch in tests)."""
    global _DEFAULT_PROVIDER
    if _DEFAULT_PROVIDER is None:
        _DEFAULT_PROVIDER = CCXTBinanceDerivativesProvider()
    return _DEFAULT_PROVIDER.get_context(symbol)


def evaluate_funding_risk(direction: str, funding_rate: float, extreme_threshold: float) -> Dict[str, Any]:
    """
    Crowded-trade filter:
    - LONG blocked when funding >= +extreme (longs over-leveraged -> long squeeze risk)
    - SHORT blocked when funding <= -extreme (shorts crowded -> short squeeze risk)
    """
    if direction == "BUY" and funding_rate >= extreme_threshold:
        return {"blocked": True, "reason": f"Funding {funding_rate * 100:+.4f}% too positive (crowded longs)"}
    if direction == "SELL" and funding_rate <= -extreme_threshold:
        return {"blocked": True, "reason": f"Funding {funding_rate * 100:+.4f}% too negative (crowded shorts)"}
    return {"blocked": False, "reason": ""}
