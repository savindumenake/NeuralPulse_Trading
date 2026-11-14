from typing import Dict, Any, NotRequired, TypedDict

import pandas as pd


class TradingState(TypedDict):
    # Market & Asset info
    symbol: str
    current_price: float

    # Technical Agent outputs
    technical_signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    indicators: Dict[str, Any]

    # Sentiment Agent outputs
    sentiment_score: float
    sentiment_summary: NotRequired[str]

    # Risk Arbiter & Execution outputs
    final_decision: str
    position_size: float
    stop_loss: float
    capital_allocation: float
    execution_status: str
    telegram_sent: NotRequired[bool]

    # Optional inputs for the multi-timeframe technical agent
    ohlcv_1d: NotRequired[pd.DataFrame]
    ohlcv_4h: NotRequired[pd.DataFrame]
    ohlcv_1h: NotRequired[pd.DataFrame]
    ohlcv_15m: NotRequired[pd.DataFrame]
    ohlcv_5m: NotRequired[pd.DataFrame]
    ohlcv_ml: NotRequired[pd.DataFrame]
    trade_horizon: NotRequired[str]  # 'SCALP', 'INTRADAY', 'SWING'
    technical_reason: NotRequired[str]
