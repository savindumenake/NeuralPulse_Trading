"""
data_loader.py - Data Ingestion and Feature Engineering Pipeline
Part of NeuralPulse Trading (Phase 1)

Fetches historical OHLCV data for BTC/USDT via CCXT (Binance)
and computes stationary technical indicator features using pandas_ta.
Adheres strictly to zero look-ahead bias and stationary feature requirements.
"""

import os
import argparse
import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import ccxt
import pandas_ta as ta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataLoader")

DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1h"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def get_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    """Initialize exchange instance with rate limiting enabled."""
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Exchange '{exchange_id}' not supported by CCXT.")
    exchange = exchange_class({
        "enableRateLimit": True,
        "timeout": 30000,
    })
    return exchange


def fetch_historical_ohlcv(
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    limit: int = 1500,
    exchange_id: str = "binance"
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from CCXT exchange with pagination.

    Parameters:
        symbol: Trading pair, e.g. 'BTC/USDT'
        timeframe: Candle interval, e.g. '1h'
        limit: Number of candles to fetch
        exchange_id: CCXT exchange identifier

    Returns:
        pd.DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    exchange = get_exchange(exchange_id)
    logger.info(f"Connecting to {exchange_id} to fetch {limit} candles for {symbol} ({timeframe})...")

    all_ohlcv = []
    fetch_limit = min(limit, 1000)
    
    # Calculate since timestamp if limit > 1000
    ms_per_candle = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - (limit * ms_per_candle)

    current_since = since
    while len(all_ohlcv) < limit:
        batch_limit = min(limit - len(all_ohlcv), fetch_limit)
        try:
            batch = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since,
                limit=batch_limit
            )
        except Exception as e:
            logger.warning(f"Error fetching from {exchange_id}: {e}. Retrying with fallback...")
            # Fallback to fetching latest candles if 'since' pagination hits edge
            batch = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=batch_limit)

        if not batch:
            break

        all_ohlcv.extend(batch)
        current_since = batch[-1][0] + ms_per_candle

        if len(batch) < batch_limit:
            break

    # Deduplicate and sort by timestamp
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    # Truncate to requested limit if exceeded
    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)

    logger.info(f"Successfully fetched {len(df)} candles from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}.")
    return df


def calculate_technical_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Computes technical indicator features strictly preserving stationarity
    and preventing look-ahead bias.

    Stationarity rules applied:
    - Raw price levels are NEVER used as model inputs.
    - EMAs are transformed into percentage distance from current close price.
    - MACD and ATR are normalized by the close price.
    - Return series and ratios are bounded and stationary.

    Parameters:
        df: DataFrame with ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']
        is_training: If True, adds future shifted target labels without leaking to features.

    Returns:
        pd.DataFrame enriched with stationary technical indicators and targets.
    """
    df = df.copy()

    # 1. Price Returns & Stationary Ratios
    df["return_1h"] = df["close"].pct_change()
    df["return_4h"] = df["close"].pct_change(4)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    df["co_spread"] = (df["close"] - df["open"]) / df["open"]
    df["vol_change"] = df["volume"].pct_change()
    df["volatility_10"] = df["return_1h"].rolling(window=10).std()

    # 2. RSI (14 period)
    df["rsi_14"] = df.ta.rsi(close=df["close"], length=14)
    # Normalized RSI centered around 0 (range roughly -0.5 to 0.5)
    df["rsi_14_norm"] = (df["rsi_14"] - 50.0) / 100.0

    # 3. MACD (12, 26, 9)
    macd = df.ta.macd(close=df["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        # Columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        # Normalize MACD values relative to close price for stationarity
        df["macd_line_pct"] = macd["MACD_12_26_9"] / df["close"]
        df["macd_signal_pct"] = macd["MACDs_12_26_9"] / df["close"]
        df["macd_hist_pct"] = macd["MACDh_12_26_9"] / df["close"]

    # 4. EMAs (20 and 50) - Normalized to relative distances
    ema_20 = df.ta.ema(close=df["close"], length=20)
    ema_50 = df.ta.ema(close=df["close"], length=50)
    df["ema_20_dist"] = (df["close"] - ema_20) / ema_20
    df["ema_50_dist"] = (df["close"] - ema_50) / ema_50
    df["ema_cross_20_50"] = (ema_20 - ema_50) / ema_50

    # Raw EMAs kept for reference/logging only, not fed as raw features
    df["raw_ema_20"] = ema_20
    df["raw_ema_50"] = ema_50

    # 5. ATR (14 period) for Risk Arbiter position sizing and Stop Loss
    atr_14 = df.ta.atr(high=df["high"], low=df["low"], close=df["close"], length=14)
    df["raw_atr_14"] = atr_14
    df["atr_pct"] = atr_14 / df["close"]  # Stationary version for ML

    # 6. Target Variable Creation (Guarded against Look-Ahead Bias)
    if is_training:
        # Candle t predicts directional return from t to t+1
        # Feature row t ONLY contains past information up to candle t.
        # Future return is strictly calculated from close(t) to close(t+1):
        future_return = (df["close"].shift(-1) - df["close"]) / df["close"]
        df["future_return"] = future_return

        # Binary classification target: 1 = BULLISH (Up), 0 = BEARISH (Down / Flat)
        # Using a small threshold (0.0) or (0.0005)
        df["target"] = (future_return > 0.0).astype(int)

    return df


def get_feature_column_names() -> list:
    """Return the list of strictly stationary feature columns used for XGBoost."""
    return [
        "return_1h",
        "return_4h",
        "log_return",
        "hl_spread",
        "co_spread",
        "vol_change",
        "volatility_10",
        "rsi_14_norm",
        "macd_line_pct",
        "macd_signal_pct",
        "macd_hist_pct",
        "ema_20_dist",
        "ema_50_dist",
        "ema_cross_20_50",
        "atr_pct",
    ]


def prepare_training_dataset(
    limit: int = 2000,
    save_to_csv: bool = True
) -> Tuple[pd.DataFrame, list]:
    """
    Downloads historical data, calculates features, applies dropna,
    and returns cleaned stationary dataset ready for XGBoost training.
    """
    raw_df = fetch_historical_ohlcv(limit=limit)
    featured_df = calculate_technical_features(raw_df, is_training=True)

    feature_cols = get_feature_column_names()
    
    # Drop rows with NaN caused by rolling windows and future shift
    clean_df = featured_df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    if save_to_csv:
        os.makedirs(DATA_DIR, exist_ok=True)
        raw_path = os.path.join(DATA_DIR, "btc_usdt_1h_raw.csv")
        feat_path = os.path.join(DATA_DIR, "btc_usdt_1h_features.csv")
        raw_df.to_csv(raw_path, index=False)
        clean_df.to_csv(feat_path, index=False)
        logger.info(f"Saved raw data to {raw_path}")
        logger.info(f"Saved cleaned features to {feat_path} (Shape: {clean_df.shape})")

    return clean_df, feature_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralPulse Trading - Data Loader and Feature Engineering")
    parser.add_argument("--limit", type=int, default=1500, help="Number of candles to fetch")
    parser.add_argument("--save", action="store_true", default=True, help="Save to CSV in data/ directory")
    args = parser.parse_args()

    print("=" * 60)
    print("NeuralPulse Trading - Phase 1: Environment & Data Pipeline")
    print("=" * 60)

    clean_data, features = prepare_training_dataset(limit=args.limit, save_to_csv=args.save)
    print(f"\nFeature extraction complete!")
    print(f"Total valid samples: {len(clean_data)}")
    print(f"Feature columns ({len(features)}): {features}")
    print("\nSample feature row (latest):")
    print(clean_data[features + ["target", "datetime"]].tail(1).T)
