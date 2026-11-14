"""
execution_agent.py - Signal Broadcast & Paper Execution Agent (Phase 4, Node 4)

Triggered only when a high-conviction trade is approved by both Technical
and Fundamental/Risk gates:
1. Formats a clean VIP Trader Signal Card (Entry, SL, TP1, TP2, TP3, Confirmations).
2. Prints a prominent Console Alert (with audio bell) when a trade triggers.
3. Broadcasts the signal to the configured Telegram Channel / Chat (with anti-spam cooldown).
4. Logs the paper trade to data/paper_trades.csv.
"""

import os
import csv
import logging
from datetime import datetime, timezone
from typing import Optional

from agents.state import TradingState
from agents.trade_memory import DEFAULT_DB_PATH, TradeMemory, infer_market_session
from agents.telegram_notifier import (
    format_trade_signal_message,
    send_telegram_signal,
    is_in_cooldown,
    mark_signal_sent,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ExecutionAgent")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TRADE_LOG_PATH = os.path.join(DATA_DIR, "paper_trades.csv")

TRADE_LOG_FIELDS = [
    "timestamp",
    "symbol",
    "direction",
    "entry_price",
    "stop_loss",
    "take_profit",
    "position_size_btc",
    "capital_allocated_usd",
    "sentiment_score",
    "confidence",
    "technical_reason",
    "status",
]


def _ensure_trade_log_exists():
    """Create trade log CSV with headers if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(TRADE_LOG_PATH):
        with open(TRADE_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
            writer.writeheader()


def _log_trade(state: TradingState):
    """Append a single trade record to the CSV log."""
    _ensure_trade_log_exists()

    indicators = state.get("indicators", {}) or {}
    trade_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": state.get("symbol", "BTC/USDT"),
        "direction": state.get("final_decision", "HOLD"),
        "entry_price": state.get("current_price", 0.0),
        "stop_loss": state.get("stop_loss", 0.0),
        "take_profit": float(indicators.get("take_profit", 0.0)),
        "position_size_btc": state.get("position_size", 0.0),
        "capital_allocated_usd": state.get("capital_allocation", 0.0),
        "sentiment_score": state.get("sentiment_score", 0.0),
        "confidence": state.get("confidence", 0.0),
        "technical_reason": state.get("technical_reason", ""),
        "status": "PAPER_EXECUTED",
    }

    with open(TRADE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
        writer.writerow(trade_record)

    return trade_record


def execution_agent(state: TradingState, memory: Optional[TradeMemory] = None) -> TradingState:
    """
    Execution & Alert Agent — Node 4 of the LangGraph pipeline.
    Only alerts and logs when final_decision is BUY or SELL.
    """
    decision = state.get("final_decision", "HOLD")

    if decision in ("BUY", "SELL"):
        _log_trade(state)
        state["execution_status"] = "PAPER_EXECUTED"

        indicators = dict(state.get("indicators") or {})
        db = memory if memory is not None else TradeMemory(DEFAULT_DB_PATH)
        now_iso = datetime.now(timezone.utc).isoformat()
        patterns = indicators.get("detected_patterns") or []
        pattern_str = (
            indicators.get("chart_pattern_1h")
            if indicators.get("chart_pattern_1h") and indicators.get("chart_pattern_1h") != "NO_CLEAR_PATTERN"
            else (patterns[0] if patterns else "NONE")
        )
        trade_id = db.log_entry({
            "entry_time": now_iso,
            "side": "LONG" if decision == "BUY" else "SHORT",
            "timeframe": "15m",
            "market_session": infer_market_session(),
            "confluence_score": float(indicators.get("sniper_score", indicators.get("score", 0.0))),
            "xgb_confidence": float(indicators.get("ml_confidence", 0.0)),
            "top_shap_features": indicators.get("top_shap_features", []),
            "pattern_detected": pattern_str,
            "atr_pct": float(indicators.get("atr_pct", 0.0)),
            "adx_14": float(indicators.get("adx_14", 0.0)),
            "rsi_14": float(indicators.get("rsi_14", 50.0)),
            "volume_ratio": float(indicators.get("volume_ratio", 0.0)),
            "funding_rate": float(indicators.get("funding_rate", 0.0)),
            "news_sentiment_score": float(state.get("sentiment_score", 0.0)),
            "entry_price": float(state.get("current_price", 0.0)),
            "initial_sl": float(state.get("stop_loss", 0.0)),
            "risk_pct": float(indicators.get("risk_pct", 1.0)) / 100.0,
            "account_equity_before": float(indicators.get("account_equity", db.get_account_equity())),
        })
        indicators["trade_id"] = trade_id
        indicators["cumulative_metrics"] = db.get_cumulative_metrics()
        state["indicators"] = indicators

        signal_card = format_trade_signal_message(state)
        print("\a")  # Terminal alert bell
        print("\n" + signal_card + "\n")

        if not is_in_cooldown(decision):
            sent = send_telegram_signal(signal_card)
            state["telegram_sent"] = sent
            if sent:
                mark_signal_sent(decision, float(state.get("current_price", 0.0)))
        else:
            state["telegram_sent"] = False
            logger.info(f"Duplicate {decision} signal suppressed by cooldown timer.")
    else:
        state["execution_status"] = "SKIPPED"
        state["telegram_sent"] = False

    return state
