"""
main.py - NeuralPulse Pro-Trader Signal & Live Monitoring Entry Point

Silently monitors live multi-timeframe BTC/USDT candles and ONLY alerts
(Console + Telegram Channel + Paper Trade Log) when high-conviction
Technical Confluence AND Fundamental News Sentiment align for a trade.

Usage:
    python main.py --live                  # Live WebSocket watch (alerts only on valid trades)
    python main.py --live --trigger 1h     # Live WebSocket watch on 1H candle closes
    python main.py --test-telegram         # Send a test VIP Signal Card to your Telegram Channel
    python main.py                         # Run a single check right now
"""

import os
import sys
import time
import asyncio
import argparse
import logging
from datetime import datetime, timezone

import ccxt
import pandas as pd
from dotenv import load_dotenv

from agents.state import TradingState
from agents.telegram_notifier import (
    format_lifecycle_event_message,
    format_trade_signal_message,
    send_telegram_signal,
)
from agents.trade_lifecycle import TradePosition
from agents.trade_memory import DEFAULT_DB_PATH, TradeMemory
from graph_builder import build_trading_graph
from live_feed import stream_binance_klines

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Main")

SYMBOL = "BTC/USDT"
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")


def fetch_multi_timeframe_data(exchange: ccxt.Exchange) -> dict:
    """
    Fetch OHLCV data for all 3 required timeframes from the exchange.
    Drops the currently forming (unclosed) candle so pattern, volume, and
    indicator checks run strictly on confirmed closed candles.

    Returns dict with keys: ohlcv_1d, ohlcv_1h, ohlcv_15m as DataFrames.
    """
    timeframes = {
        "ohlcv_1d": ("1d", 211),   # 210 closed daily candles for regime detection
        "ohlcv_4h": ("4h", 101),   # 100 closed 4h candles for HTF EMA trend alignment
        "ohlcv_1h": ("1h", 201),   # 200 closed hourly candles for S/R & patterns
        "ohlcv_15m": ("15m", 61),  # 60 closed fifteen-minute candles for entry triggers
        "ohlcv_5m": ("5m", 61),    # 60 closed five-minute candles for scalp triggers
    }

    now_ms = exchange.milliseconds()
    result = {}
    for key, (tf, limit) in timeframes.items():
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol=SYMBOL,
                timeframe=tf,
                limit=limit,
            )
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            tf_ms = exchange.parse_timeframe(tf) * 1000
            if not df.empty and (now_ms < int(df["timestamp"].iloc[-1]) + tf_ms):
                df = df.iloc[:-1].reset_index(drop=True)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            result[key] = df
        except Exception as e:
            logger.error(f"Failed to fetch {key}: {e}")
            result[key] = pd.DataFrame()

    return result


_ACTIVE_POSITIONS: list = []


def _manage_active_positions(data: dict, trigger_tf: str = "15m") -> None:
    """Advance any open TradePosition on new closed candle and process 4h/12h autopsies."""
    df_entry = data.get(f"ohlcv_{trigger_tf}")
    if df_entry is None or df_entry.empty:
        df_entry = data.get("ohlcv_15m")
    if df_entry is None or df_entry.empty:
        return

    latest = df_entry.iloc[-1]
    c_high = float(latest["high"])
    c_low = float(latest["low"])
    c_close = float(latest["close"])
    c_time = str(latest.get("datetime", datetime.now(timezone.utc).isoformat()))

    atr_s = df_entry.ta.atr(length=14) if len(df_entry) >= 15 else None
    atr_14 = float(atr_s.dropna().iloc[-1]) if atr_s is not None and not atr_s.dropna().empty else 0.0

    db = TradeMemory(DEFAULT_DB_PATH)
    db.process_due_autopsies(current_price=c_close, current_time=c_time, window_high=c_high, window_low=c_low)

    still_open = []
    for pos in _ACTIVE_POSITIONS:
        events = pos.step_candle(high=c_high, low=c_low, close=c_close, atr_14=atr_14, candle_time=c_time)
        for ev in events:
            if pos.closed:
                db.log_exit(pos.trade_id, pos.to_lifecycle_summary())
            metrics = db.get_cumulative_metrics()
            msg = format_lifecycle_event_message(ev, metrics)
            print("\n" + msg + "\n")
            send_telegram_signal(msg)
        if not pos.closed:
            still_open.append(pos)
    _ACTIVE_POSITIONS[:] = still_open


def run_pipeline_with_data(graph, data: dict, quiet_on_hold: bool = True, trigger_tf: str = "15m"):
    """
    Execute one pipeline check.
    When `quiet_on_hold=True`, stays silent (1 status line) unless a real BUY/SELL
    trade is approved. When a trade IS approved, `execution_agent` prints the
    VIP Signal Card and sends it to Telegram.
    """
    _manage_active_positions(data, trigger_tf=trigger_tf)

    initial_state = TradingState(
        symbol=SYMBOL,
        current_price=0.0,
        technical_signal="HOLD",
        confidence=0.0,
        indicators={},
        sentiment_score=0.0,
        final_decision="HOLD",
        position_size=0.0,
        stop_loss=0.0,
        capital_allocation=0.0,
        execution_status="PENDING",
        ohlcv_1d=data.get("ohlcv_1d", pd.DataFrame()),
        ohlcv_4h=data.get("ohlcv_4h", pd.DataFrame()),
        ohlcv_1h=data.get("ohlcv_1h", pd.DataFrame()),
        ohlcv_15m=data.get("ohlcv_15m", pd.DataFrame()),
        ohlcv_5m=data.get("ohlcv_5m", pd.DataFrame()),
    )

    result = graph.invoke(initial_state)
    decision = result.get("final_decision", "HOLD")
    price = float(result.get("current_price", 0.0))
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if decision in ("BUY", "SELL"):
        ind = result.get("indicators") or {}
        if ind.get("trade_id"):
            _ACTIVE_POSITIONS.append(TradePosition(
                trade_id=str(ind["trade_id"]),
                side=decision,
                entry_price=price,
                initial_sl=float(result.get("stop_loss", 0.0)),
                notional_usd=float(ind.get("notional_value_usd", 0.0)),
                timeframe=str(ind.get("execution_timeframe", trigger_tf)),
                horizon=str(ind.get("trade_horizon", "INTRADAY")),
                time_stop_candles=ind.get("time_stop_candles"),
                risk_usd=float(ind.get("risk_amount_usd", 0.0)),
            ))
    elif decision == "HOLD":
        if quiet_on_hold:
            print(
                f"  [{now_str}] ⏳ Watching {SYMBOL} (${price:,.2f}) — "
                f"Waiting for high-conviction trade setup...",
                flush=True,
            )
        else:
            reason = result.get("technical_reason", "Waiting for confirmations")
            print(f"[{now_str}] {SYMBOL} @ ${price:,.2f} -> No Trade Setup ({reason})")
    return result


def run_pipeline_once(graph, exchange: ccxt.Exchange, quiet_on_hold: bool = False):
    """Fetch latest closed candles via REST and execute one pipeline check."""
    data = fetch_multi_timeframe_data(exchange)
    return run_pipeline_with_data(graph, data, quiet_on_hold=quiet_on_hold)


def send_test_telegram_alert():
    """Send a sample Binance Futures VIP Signal Card to verify Telegram Bot & Channel setup."""
    sample_state = TradingState(
        symbol="BTC/USDT",
        current_price=91180.0,
        technical_signal="BUY",
        confidence=0.85,
        indicators={
            "score": 85.0,
            "required_threshold": 60.0,
            "take_profit_1": 91863.75,
            "take_profit": 92091.80,
            "take_profit_2": 92091.80,
            "take_profit_3": 92547.70,
            "market_regime": "STRONG_BULLISH",
            "detected_patterns": ["BULLISH_PIN_BAR", "BULLISH_LIQUIDITY_SWEEP", "ASCENDING_TRIANGLE_BREAKOUT"],
            "market_type": "FUTURES",
            "margin_mode": "ISOLATED",
            "recommended_leverage": 10,
            "notional_value_usd": 40000.0,
            "margin_required_usd": 4000.0,
            "risk_amount_usd": 200.0,
            "risk_pct": 2.0,
            "est_liquidation_price": 82517.90,
        },
        sentiment_score=0.42,
        sentiment_summary="Bullish Fundamental News",
        final_decision="BUY",
        position_size=0.4387,
        stop_loss=90724.10,
        capital_allocation=4000.0,
        execution_status="PAPER_EXECUTED",
        technical_reason=(
            "1H Bullish Structure/Pattern: ASCENDING_TRIANGLE_BREAKOUT; "
            "Bullish price action: BULLISH_PIN_BAR; "
            "Liquidity Sweep Stop-Hunt Reversal (BULLISH_LIQUIDITY_SWEEP); "
            "Volume Spike Confirmed (2.4x Average); "
            "1H Momentum Aligned (BULLISH, RSI 58.4); "
            "XGBoost ML BUY Confirmation (Confidence: 62.0%)"
        ),
    )
    msg = format_trade_signal_message(sample_state)
    print("\n" + msg + "\n")
    ok = send_telegram_signal(msg)
    if ok:
        print("✅ Test signal successfully sent to your Telegram Channel/Chat!")
    else:
        print("⚠️ Telegram not configured or failed. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")


def main():
    parser = argparse.ArgumentParser(description="NeuralPulse Pro-Trader Signal Pipeline")
    parser.add_argument(
        "--live", action="store_true",
        help="Monitor live Binance candles via WebSocket and alert ONLY when a trade setup triggers"
    )
    parser.add_argument(
        "--trigger", type=str, default="15m", choices=["15m", "1h", "1d"],
        help="Candle close timeframe that triggers analysis (default: 15m)"
    )
    parser.add_argument("--loop", action="store_true", help="Run continuously on REST interval")
    parser.add_argument(
        "--interval", type=int, default=15,
        help="Interval in minutes between REST loop checks (default: 15)"
    )
    parser.add_argument(
        "--test-telegram", action="store_true",
        help="Send a sample VIP Signal Card to test your Telegram Channel configuration"
    )
    args = parser.parse_args()

    if args.test_telegram:
        send_test_telegram_alert()
        return

    exchange_class = getattr(ccxt, EXCHANGE_ID, None)
    if exchange_class is None:
        raise ValueError(f"Unsupported exchange ID: {EXCHANGE_ID}")
    exchange = exchange_class({"enableRateLimit": True, "timeout": 30000})

    graph = build_trading_graph()

    if args.live:
        print("🚀 NeuralPulse Live Pro-Trader Scanner Started (Silent Watch Mode)")
        print("   Alerts & Telegram Signals will fire ONLY when a confirmed trade setup appears.\n")
        try:
            asyncio.run(
                stream_binance_klines(
                    symbol=SYMBOL,
                    on_candle_close=lambda data: run_pipeline_with_data(
                        graph, data, quiet_on_hold=True, trigger_tf=args.trigger
                    ),
                    trigger_timeframe=args.trigger,
                    exchange=exchange,
                )
            )
        except KeyboardInterrupt:
            print("\n🛑 Live scanner stopped by user.")
            sys.exit(0)
    elif args.loop:
        print(f"🔄 NeuralPulse Scanner running every {args.interval} minutes (Silent Watch Mode)...")
        while True:
            try:
                run_pipeline_once(graph, exchange, quiet_on_hold=True)
                time.sleep(args.interval * 60)
            except KeyboardInterrupt:
                print("\n🛑 Scanner stopped by user.")
                sys.exit(0)
            except Exception as e:
                logger.error(f"Pipeline error: {e}. Retrying in 60 seconds...")
                time.sleep(60)
    else:
        run_pipeline_once(graph, exchange, quiet_on_hold=False)


if __name__ == "__main__":
    main()
