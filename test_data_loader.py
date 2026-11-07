"""
test_data_loader.py - Unit test suite for Phase 1 Data Pipeline
"""

import unittest
import numpy as np
import pandas as pd
from data_loader import calculate_technical_features, get_feature_column_names


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Generate synthetic OHLCV data (100 candles)
        np.random.seed(42)
        n = 100
        timestamps = pd.date_range("2026-01-01", periods=n, freq="1h")
        
        # Random walk price
        returns = np.random.normal(0, 0.01, size=n)
        prices = 50000.0 * np.exp(np.cumsum(returns))
        highs = prices * (1 + np.abs(np.random.normal(0, 0.005, size=n)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.005, size=n)))
        opens = (highs + lows) / 2
        closes = prices
        volumes = np.random.uniform(10, 100, size=n)

        self.df = pd.DataFrame({
            "timestamp": timestamps.astype("int64") // 10**6,
            "datetime": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        })

    def test_feature_generation(self):
        featured = calculate_technical_features(self.df, is_training=True)
        feature_cols = get_feature_column_names()

        # Check all required features exist
        for col in feature_cols:
            self.assertIn(col, featured.columns, f"Missing feature: {col}")

        # Check target exists
        self.assertIn("target", featured.columns)

    def test_stationarity_no_raw_price_levels(self):
        feature_cols = get_feature_column_names()
        # Ensure no raw price level columns are in feature set
        forbidden = ["open", "high", "low", "close", "raw_ema_20", "raw_ema_50", "raw_atr_14"]
        for col in forbidden:
            self.assertNotIn(col, feature_cols, f"Forbidden non-stationary column in feature set: {col}")

    def test_no_lookahead_bias(self):
        featured = calculate_technical_features(self.df, is_training=True)
        
        # target[t] should be 1 if close[t+1] > close[t], else 0
        valid_rows = featured.dropna(subset=["target", "close"]).iloc[:-1]
        for i in range(len(valid_rows) - 1):
            curr_close = valid_rows["close"].iloc[i]
            next_close = valid_rows["close"].iloc[i + 1]
            expected_target = 1 if next_close > curr_close else 0
            self.assertEqual(
                valid_rows["target"].iloc[i],
                expected_target,
                f"Mismatch in target alignment at index {i}"
            )


if __name__ == "__main__":
    unittest.main()
