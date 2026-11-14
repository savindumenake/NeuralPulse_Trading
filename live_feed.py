"""
live_feed.py - Real-Time Binance WebSocket Candle Streamer

Maintains live OHLCV buffers for 1D, 1H, and 15m timeframes via Binance
WebSocket streams. Automatically triggers the trading pipeline whenever
a confirmed candle closes, and keeps data/btc_usdt_1h_raw.csv up to date.
"""

import os
import json
import asyncio
import logging
from typing import Callable, Dict, Optional

import ccxt
import pandas as pd
import websockets

logger = logging.getLogger("LiveFeed")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_1H_CSV = os.path.join(DATA_DIR, "btc_usdt_1h_raw.csv")

DEFAULT_LIMITS = {
    "1d": 220,
    "4h": 120,
    "1h": 220,
    "15m": 100,
    "5m": 100,
}


class LiveCandleStore:
    """In-memory multi-timeframe OHLCV store updated by live WebSocket klines."""

    def __init__(
        self,
        limits: Optional[Dict[str, int]] = None,
        save_1h_csv_path: Optional[str] = RAW_1H_CSV,
    ):
        self.limits = limits or dict(DEFAULT_LIMITS)
        self.save_1h_csv_path = save_1h_csv_path
        self.frames: Dict[str, pd.DataFrame] = {tf: pd.DataFrame() for tf in self.limits}
        self.live_candles: Dict[str, dict] = {}

    def bootstrap(self, exchange: ccxt.Exchange, symbol: str = "BTC/USDT") -> None:
        """Fetch initial confirmed closed candles via REST before starting the stream."""
        now_ms = exchange.milliseconds()
        for tf, limit in self.limits.items():
            fetch_count = limit + 1
            ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=tf, limit=fetch_count)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            tf_ms = exchange.parse_timeframe(tf) * 1000
            if not df.empty and (now_ms < int(df["timestamp"].iloc[-1]) + tf_ms):
                df = df.iloc[:-1].reset_index(drop=True)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            self.frames[tf] = df.tail(limit).reset_index(drop=True)
            logger.info(f"Bootstrapped {tf}: {len(self.frames[tf])} closed candles.")

    def apply_kline_event(self, timeframe: str, kline: dict) -> bool:
        """
        Process a Binance kline payload (`data['k']`).

        Updates the live forming candle state and, when `k['x']` is True (closed),
        appends/updates the confirmed candle DataFrame.

        Returns True if a candle just closed, False otherwise.
        """
        if timeframe not in self.frames:
            return False

        row = {
            "timestamp": int(kline["t"]),
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
        }
        self.live_candles[timeframe] = row

        if not bool(kline.get("x", False)):
            return False

        row["datetime"] = pd.to_datetime(row["timestamp"], unit="ms", utc=True)
        existing = self.frames[timeframe]
        new_df = pd.DataFrame([row])

        if existing.empty:
            updated = new_df
        else:
            updated = pd.concat([existing, new_df], ignore_index=True)
            updated = (
                updated.drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

        max_rows = self.limits.get(timeframe, 200)
        if len(updated) > max_rows:
            updated = updated.tail(max_rows).reset_index(drop=True)
        self.frames[timeframe] = updated

        if timeframe == "1h" and self.save_1h_csv_path:
            self._append_1h_csv(row)

        logger.info(
            f"Closed {timeframe} candle @ {row['datetime']} | "
            f"Close: ${row['close']:.2f} | Vol: {row['volume']:.2f}"
        )
        return True

    def _append_1h_csv(self, row: dict) -> None:
        """Keep data/btc_usdt_1h_raw.csv synchronized with newly closed 1H candles."""
        try:
            os.makedirs(os.path.dirname(self.save_1h_csv_path), exist_ok=True)
            new_row = pd.DataFrame([row])
            if os.path.exists(self.save_1h_csv_path):
                existing = pd.read_csv(self.save_1h_csv_path)
                combined = pd.concat([existing, new_row], ignore_index=True)
                combined = (
                    combined.drop_duplicates(subset=["timestamp"], keep="last")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
            else:
                combined = new_row
            combined.to_csv(self.save_1h_csv_path, index=False)
        except Exception as e:
            logger.warning(f"Could not update 1H CSV: {e}")

    def get_pipeline_data(self) -> Dict[str, pd.DataFrame]:
        """Return copies of confirmed closed DataFrames formatted for TradingState."""
        return {
            "ohlcv_1d": self.frames.get("1d", pd.DataFrame()).copy(),
            "ohlcv_4h": self.frames.get("4h", pd.DataFrame()).copy(),
            "ohlcv_1h": self.frames.get("1h", pd.DataFrame()).copy(),
            "ohlcv_15m": self.frames.get("15m", pd.DataFrame()).copy(),
            "ohlcv_5m": self.frames.get("5m", pd.DataFrame()).copy(),
        }


def build_binance_stream_url(symbol: str = "BTC/USDT", timeframes: tuple = ("5m", "15m", "1h", "4h", "1d")) -> str:
    """Build Binance combined WebSocket stream URL for multi-timeframe klines."""
    stream_symbol = symbol.replace("/", "").lower()
    streams = "/".join(f"{stream_symbol}@kline_{tf}" for tf in timeframes)
    return f"wss://stream.binance.com:9443/stream?streams={streams}"


async def stream_binance_klines(
    symbol: str,
    on_candle_close: Callable[[Dict[str, pd.DataFrame]], None],
    trigger_timeframe: str = "15m",
    exchange: Optional[ccxt.Exchange] = None,
    save_1h_csv_path: Optional[str] = RAW_1H_CSV,
) -> None:
    """
    Connect to Binance WebSocket and run `on_candle_close` automatically
    whenever `trigger_timeframe` (default '15m') closes.
    """
    if exchange is None:
        exchange = ccxt.binance({"enableRateLimit": True, "timeout": 30000})

    store = LiveCandleStore(save_1h_csv_path=save_1h_csv_path)
    store.bootstrap(exchange, symbol=symbol)

    # Initial check on latest confirmed candles
    on_candle_close(store.get_pipeline_data())

    ws_url = build_binance_stream_url(symbol=symbol, timeframes=tuple(store.limits.keys()))
    backoff = 2

    while True:
        try:
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
                print(f"🟢 Live Binance Stream Connected ({symbol} — {trigger_timeframe} close trigger). Watching market...")
                backoff = 2

                async for raw_msg in ws:
                    payload = json.loads(raw_msg)
                    data = payload.get("data", {})
                    kline = data.get("k")
                    if not kline:
                        continue

                    tf = kline.get("i")
                    closed = store.apply_kline_event(tf, kline)

                    is_trigger = (
                        (tf == trigger_timeframe)
                        or (trigger_timeframe.lower() == "all" and tf in ("5m", "15m", "1h"))
                        or (tf in [t.strip() for t in trigger_timeframe.split(",")])
                    )
                    if closed and is_trigger:
                        on_candle_close(store.get_pipeline_data())

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"WebSocket disconnected ({e}). Reconnecting in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
            try:
                store.bootstrap(exchange, symbol=symbol)
            except Exception as boot_err:
                logger.warning(f"Resync bootstrap failed: {boot_err}")

