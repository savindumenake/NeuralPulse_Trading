import os
import csv
import importlib
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from agents.state import TradingState
from agents.technical_agent import TechnicalAgent, technical_analyst_agent
from agents.market_structure import (
    detect_candlestick_signal,
    detect_basic_patterns,
    detect_liquidity_sweep,
    analyze_htf_momentum_alignment,
)
from agents.risk_arbiter import risk_arbiter_agent
from graph_builder import build_trading_graph
from main import fetch_multi_timeframe_data
from live_feed import LiveCandleStore, build_binance_stream_url
from agents.telegram_notifier import (
    format_lifecycle_event_message,
    format_trade_signal_message,
    send_telegram_signal,
    reset_signal_cooldown,
    is_in_cooldown,
    mark_signal_sent,
)
from agents.trade_lifecycle import (
    STATE_CHANDELIER_RUNNER,
    STATE_INITIAL,
    STATE_PROFIT_LOCK,
    STATE_RETEST_SHIELD,
    STATE_TIME_STOP,
    TradePosition,
)
from agents.trade_memory import TradeMemory
from agents.sentiment_agent import analyze_headlines_vader
from analyze_performance import generate_diagnostic_report

execution_module = importlib.import_module("agents.execution_agent")
sentiment_module = importlib.import_module("agents.sentiment_agent")


class TestTechnicalAgent(unittest.TestCase):
    def make_frame(self, rows: int, close: float = 100.0) -> pd.DataFrame:
        return pd.DataFrame({
            "open": [close] * rows,
            "high": [close + 1] * rows,
            "low": [close - 1] * rows,
            "close": [close] * rows,
            "volume": [10.0] * rows,
        })

    def make_state(self) -> TradingState:
        return TradingState(
            symbol="BTC/USDT",
            current_price=0.0,
            technical_signal="HOLD",
            confidence=0.0,
            indicators={},
            sentiment_score=0.0,
            final_decision="HOLD",
            position_size=0.0,
            stop_loss=0.0,
            execution_status="PENDING",
        )

    def test_support_proximity_does_not_accept_resistance_above_price(self):
        agent = TechnicalAgent()

        is_near_support, support = agent._check_proximity(100.0, [100.2])

        self.assertFalse(is_near_support)
        self.assertIsNone(support)

    def test_missing_timeframe_data_returns_hold_in_state_adapter(self):
        state = self.make_state()

        result = technical_analyst_agent(state)

        self.assertIs(result, state)
        self.assertEqual(state["technical_signal"], "HOLD")
        self.assertIn("1D timeframe data is missing", state["technical_reason"])

    @patch("agents.technical_agent.calculate_dynamic_risk", return_value={
        "atr_value": 0.67,
        "recommended_sl_distance": 1.0,
    })
    @patch("agents.technical_agent.detect_candlestick_signal", return_value="BULLISH_ENGULFING")
    @patch("agents.technical_agent.analyze_volume_spike", return_value={"is_spike": True, "ratio": 2.0})
    @patch("agents.technical_agent.find_angled_trendlines", return_value={
        "slope": 0.1,
        "strength_r2": 0.9,
        "support_price": 99.8,
    })
    @patch("agents.technical_agent.find_support_resistance_zones", return_value={
        "support": [100.0],
        "resistance": [101.0],
    })
    @patch("agents.technical_agent.detect_market_regime", return_value="RANGING")
    def test_ranging_market_can_buy_at_maximum_valid_score(self, *_mocks):
        agent = TechnicalAgent()
        decision = agent.analyze_multi_timeframe(
            self.make_frame(200),
            self.make_frame(30),
            self.make_frame(20),
        )

        self.assertEqual(decision["signal"], "BUY")
        self.assertEqual(decision["score"], 80)
        self.assertEqual(decision["required_threshold"], 80)
        self.assertEqual(decision["nearest_support_zone"], 100.0)
        self.assertEqual(decision["recommended_stop_loss"], 99.0)

    @patch("agents.technical_agent.calculate_dynamic_risk", return_value={
        "atr_value": 0.67,
        "recommended_sl_distance": 1.0,
    })
    @patch("agents.technical_agent.detect_candlestick_signal", return_value="BEARISH_ENGULFING")
    @patch("agents.technical_agent.analyze_volume_spike", return_value={"is_spike": True, "ratio": 2.0})
    @patch("agents.technical_agent.find_angled_resistance_trendline", return_value=None)
    @patch("agents.technical_agent.find_angled_trendlines", return_value=None)
    @patch("agents.technical_agent.find_support_resistance_zones", return_value={
        "support": [99.0],
        "resistance": [100.0],
    })
    @patch("agents.technical_agent.detect_market_regime", return_value="STRONG_BEARISH")
    def test_bearish_market_can_sell_with_stop_above_entry(self, *_mocks):
        agent = TechnicalAgent()
        decision = agent.analyze_multi_timeframe(
            self.make_frame(200),
            self.make_frame(30),
            self.make_frame(20),
        )

        self.assertEqual(decision["signal"], "SELL")
        self.assertEqual(decision["score"], 60)
        self.assertEqual(decision["nearest_resistance_zone"], 100.0)
        self.assertEqual(decision["recommended_stop_loss"], 101.0)

    @patch("agents.technical_agent.calculate_dynamic_risk", return_value={
        "atr_value": float("nan"),
        "recommended_sl_distance": float("nan"),
    })
    @patch("agents.technical_agent.detect_candlestick_signal", return_value="BULLISH_ENGULFING")
    @patch("agents.technical_agent.analyze_volume_spike", return_value={"is_spike": True, "ratio": 2.0})
    @patch("agents.technical_agent.find_angled_trendlines", return_value=None)
    @patch("agents.technical_agent.find_support_resistance_zones", return_value={
        "support": [100.0],
        "resistance": [101.0],
    })
    @patch("agents.technical_agent.detect_market_regime", return_value="STRONG_BULLISH")
    def test_invalid_atr_blocks_buy(self, *_mocks):
        agent = TechnicalAgent()
        decision = agent.analyze_multi_timeframe(
            self.make_frame(200),
            self.make_frame(30),
            self.make_frame(20),
        )

        self.assertEqual(decision["signal"], "HOLD")
        self.assertIn("ATR stop-loss distance", decision["reason"])

    def test_crypto_continuous_bullish_and_bearish_engulfing_detected(self):
        # In 24/7 crypto markets, current['open'] == prev['close']
        bullish_df = pd.DataFrame({
            "open": [100.0, 99.0],
            "high": [100.2, 101.2],
            "low": [98.8, 98.9],
            "close": [99.0, 100.5],
            "volume": [10.0, 20.0],
        })
        self.assertEqual(detect_candlestick_signal(bullish_df), "BULLISH_ENGULFING")

        bearish_df = pd.DataFrame({
            "open": [100.0, 101.0],
            "high": [101.2, 101.1],
            "low": [99.8, 98.8],
            "close": [101.0, 99.5],
            "volume": [10.0, 20.0],
        })
        self.assertEqual(detect_candlestick_signal(bearish_df), "BEARISH_ENGULFING")

    def test_chart_pattern_confluences_enable_ranging_buy_without_angled_trendline(self):
        with (
            patch("agents.technical_agent.calculate_dynamic_risk", return_value={
                "atr_value": 0.67, "recommended_sl_distance": 1.0
            }),
            patch("agents.technical_agent.detect_candlestick_signal", return_value="BULLISH_PIN_BAR"),
            patch("agents.technical_agent.analyze_volume_spike", return_value={"is_spike": True, "ratio": 2.0}),
            patch("agents.technical_agent.find_angled_trendlines", return_value=None),
            patch("agents.technical_agent.detect_basic_patterns", return_value="DOUBLE_BOTTOM_BREAKOUT"),
            patch("agents.technical_agent.find_support_resistance_zones", return_value={
                "support": [100.0], "resistance": [105.0]
            }),
            patch("agents.technical_agent.detect_market_regime", return_value="RANGING"),
        ):
            agent = TechnicalAgent()
            decision = agent.analyze_multi_timeframe(
                self.make_frame(200),
                self.make_frame(30),
                self.make_frame(20),
            )
            self.assertEqual(decision["signal"], "BUY")
            self.assertEqual(decision["score"], 80)
            self.assertIn("DOUBLE_BOTTOM_BREAKOUT", decision["detected_patterns"])
            self.assertEqual(decision["recommended_take_profit"], 102.0)

    def test_xgb_model_confluence_adds_points_when_aligned(self):
        with (
            patch("agents.technical_agent.calculate_dynamic_risk", return_value={
                "atr_value": 0.67, "recommended_sl_distance": 1.0
            }),
            patch("agents.technical_agent.detect_candlestick_signal", return_value="BULLISH_PIN_BAR"),
            patch("agents.technical_agent.analyze_volume_spike", return_value={"is_spike": True, "ratio": 2.0}),
            patch("agents.technical_agent.find_angled_trendlines", return_value=None),
            patch("agents.technical_agent.detect_basic_patterns", return_value="NO_CLEAR_PATTERN"),
            patch("agents.technical_agent.find_support_resistance_zones", return_value={
                "support": [100.0], "resistance": [105.0]
            }),
            patch("agents.technical_agent.detect_market_regime", return_value="TRANSITION"),
            patch.object(TechnicalAgent, "_predict_xgboost_signal", return_value={
                "signal": "BUY", "prob_up": 0.62, "confidence": 0.62
            }),
        ):
            agent = TechnicalAgent()
            decision = agent.analyze_multi_timeframe(
                self.make_frame(200),
                self.make_frame(30),
                self.make_frame(20),
            )
            self.assertEqual(decision["signal"], "BUY")
            self.assertEqual(decision["score"], 75)
            self.assertEqual(decision["ml_signal"], "BUY")


class TestPipelineAgents(unittest.TestCase):
    def make_state(self, **overrides) -> TradingState:
        state = TradingState(
            symbol="BTC/USDT",
            current_price=50000.0,
            technical_signal="HOLD",
            confidence=0.8,
            indicators={
                "score": 85.0,
                "required_threshold": 60.0,
                "take_profit": 52000.0,
                "ml_signal": "BUY",
                "ml_confidence": 0.78,
                "adx_14_override": 32.0,
                "volume_ratio_override": 2.1,
                "trend_1h_override": "BUY",
                "trend_4h_override": "BUY",
                "detected_patterns": ["BULLISH_ENGULFING", "BULLISH_BREAKOUT"],
            },
            sentiment_score=0.2,
            final_decision="HOLD",
            position_size=0.0,
            stop_loss=49000.0,
            capital_allocation=0.0,
            execution_status="PENDING",
        )
        state.update(overrides)
        return state

    def test_risk_arbiter_approves_buy_when_sentiment_above_threshold(self):
        mem = TradeMemory(":memory:", initial_equity=10000.0)
        state = self.make_state(technical_signal="BUY", sentiment_score=0.1, current_price=50000.0, stop_loss=49000.0)
        with patch("agents.risk_arbiter.fetch_futures_market_context", return_value={"available": False}):
            result = risk_arbiter_agent(state, memory=mem)

        self.assertEqual(result["final_decision"], "BUY")
        self.assertGreater(result["position_size"], 0.0)
        self.assertGreater(result["capital_allocation"], 0.0)
        self.assertIn(result["indicators"]["trade_tier"], ("A_GRADE", "S_TIER"))

    def test_risk_arbiter_rejects_buy_when_sentiment_too_negative(self):
        mem = TradeMemory(":memory:", initial_equity=10000.0)
        state = self.make_state(technical_signal="BUY", sentiment_score=-0.10, current_price=50000.0, stop_loss=49000.0)
        with patch("agents.risk_arbiter.fetch_futures_market_context", return_value={"available": False}):
            result = risk_arbiter_agent(state, memory=mem)

        self.assertEqual(result["final_decision"], "HOLD")
        self.assertEqual(result["position_size"], 0.0)
        self.assertEqual(result["capital_allocation"], 0.0)

    def test_risk_arbiter_approves_sell_when_bearish(self):
        mem = TradeMemory(":memory:", initial_equity=10000.0)
        ind = dict(self.make_state()["indicators"])
        ind.update({"ml_signal": "SELL", "ml_confidence": 0.78, "trend_1h_override": "SELL", "trend_4h_override": "SELL"})
        state = self.make_state(technical_signal="SELL", sentiment_score=-0.3, current_price=50000.0, stop_loss=51000.0, indicators=ind)
        with patch("agents.risk_arbiter.fetch_futures_market_context", return_value={"available": False}):
            result = risk_arbiter_agent(state, memory=mem)

        self.assertEqual(result["final_decision"], "SELL")
        self.assertGreater(result["position_size"], 0.0)

    def test_execution_agent_logs_trade_to_csv_and_skips_hold(self):
        reset_signal_cooldown()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "paper_trades.csv")
            db_path = os.path.join(tmpdir, "mem.db")
            mem = TradeMemory(db_path=db_path, initial_equity=10000.0)
            with (
                patch.object(execution_module, "DATA_DIR", tmpdir),
                patch.object(execution_module, "TRADE_LOG_PATH", log_path),
                patch.object(execution_module, "DEFAULT_DB_PATH", db_path),
                patch.object(execution_module, "send_telegram_signal", return_value=True) as mock_tg,
            ):
                hold_result = execution_module.execution_agent(self.make_state(final_decision="HOLD"))
                self.assertEqual(hold_result["execution_status"], "SKIPPED")
                self.assertFalse(os.path.exists(log_path))
                mock_tg.assert_not_called()

                buy_state = self.make_state(
                    final_decision="BUY",
                    position_size=0.1,
                    capital_allocation=5000.0,
                )
                buy_result = execution_module.execution_agent(buy_state)
                self.assertEqual(buy_result["execution_status"], "PAPER_EXECUTED")
                self.assertTrue(buy_result["telegram_sent"])
                self.assertTrue(os.path.exists(log_path))
                mock_tg.assert_called_once()

                with open(log_path, "r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["direction"], "BUY")
                self.assertEqual(rows[0]["take_profit"], "52000.0")

    def test_sentiment_agent_parses_votes_and_handles_missing_key(self):
        with patch.object(sentiment_module, "CRYPTOPANIC_API_KEY", ""):
            res = sentiment_module.sentiment_analyst_agent(self.make_state())
            self.assertEqual(res["sentiment_score"], 0.0)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"votes": {"positive": 8, "important": 2, "negative": 1, "toxic": 1}},
            ]
        }
        mock_resp.raise_for_status.return_value = None
        with (
            patch.object(sentiment_module, "CRYPTOPANIC_API_KEY", "test_token"),
            patch("agents.sentiment_agent.requests.get", return_value=mock_resp),
        ):
            res = sentiment_module.sentiment_analyst_agent(self.make_state())
            self.assertEqual(res["sentiment_score"], 0.67)

    def test_full_langgraph_pipeline_end_to_end(self):
        reset_signal_cooldown()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "paper_trades.csv")
            db_path = os.path.join(tmpdir, "pipeline_mem.db")
            with (
                patch.object(execution_module, "DATA_DIR", tmpdir),
                patch.object(execution_module, "TRADE_LOG_PATH", log_path),
                patch.object(execution_module, "DEFAULT_DB_PATH", db_path),
                patch("agents.risk_arbiter.DEFAULT_DB_PATH", db_path),
                patch("agents.risk_arbiter.fetch_futures_market_context", return_value={"available": False}),
                patch.object(execution_module, "send_telegram_signal", return_value=True),
                patch.object(sentiment_module, "CRYPTOPANIC_API_KEY", ""),
                patch("agents.technical_agent.calculate_dynamic_risk", return_value={
                    "atr_value": 100.0, "recommended_sl_distance": 150.0
                }),
                patch("agents.technical_agent.detect_candlestick_signal", return_value="BULLISH_ENGULFING"),
                patch("agents.technical_agent.analyze_volume_spike", return_value={"is_spike": True, "ratio": 2.1}),
                patch("agents.technical_agent.find_angled_trendlines", return_value=None),
                patch("agents.technical_agent.detect_basic_patterns", return_value="BULLISH_BREAKOUT"),
                patch("agents.technical_agent.find_support_resistance_zones", return_value={
                    "support": [50000.0], "resistance": [51000.0]
                }),
                patch("agents.technical_agent.detect_market_regime", return_value="STRONG_BULLISH"),
                patch.object(TechnicalAgent, "_predict_xgboost_signal", return_value={
                    "signal": "BUY", "prob_up": 0.78, "confidence": 0.78, "top_features": []
                }),
            ):
                graph = build_trading_graph()
                frame_1d = pd.DataFrame({
                    "open": [50000.0] * 200, "high": [50100.0] * 200,
                    "low": [49900.0] * 200, "close": [50000.0] * 200, "volume": [100.0] * 200,
                })
                frame_1h = pd.DataFrame({
                    "open": [50000.0] * 30, "high": [50100.0] * 30,
                    "low": [49900.0] * 30, "close": [50000.0] * 30, "volume": [100.0] * 30,
                })
                frame_15m = pd.DataFrame({
                    "open": [50000.0] * 20, "high": [50100.0] * 20,
                    "low": [49900.0] * 20, "close": [50000.0] * 20, "volume": [100.0] * 20,
                })
                initial_state = TradingState(
                    symbol="BTC/USDT",
                    current_price=0.0,
                    technical_signal="HOLD",
                    confidence=0.0,
                    indicators={"adx_14_override": 32.0, "volume_ratio_override": 2.1, "trend_1h_override": "BUY", "trend_4h_override": "BUY"},
                    sentiment_score=0.0,
                    final_decision="HOLD",
                    position_size=0.0,
                    stop_loss=0.0,
                    capital_allocation=0.0,
                    execution_status="PENDING",
                    ohlcv_1d=frame_1d,
                    ohlcv_1h=frame_1h,
                    ohlcv_15m=frame_15m,
                )
                result = graph.invoke(initial_state)
                self.assertEqual(result["technical_signal"], "BUY")
                self.assertEqual(result["final_decision"], "BUY")
                self.assertEqual(result["execution_status"], "PAPER_EXECUTED")
                self.assertTrue(os.path.exists(log_path))

    def test_fetch_multi_timeframe_data_drops_unclosed_live_candle(self):
        mock_exchange = MagicMock()
        # Current time = 3,600,000 + 60,000 (1 minute into the second 1h candle)
        mock_exchange.milliseconds.return_value = 3_660_000
        mock_exchange.parse_timeframe.side_effect = lambda tf: {"1d": 86400, "4h": 14400, "1h": 3600, "15m": 900}[tf]
        # Candle 0 at ts=0 (closed by ts=3_660_000 for 1h/15m), Candle 1 at ts=3_600_000 (unclosed for all tfs)
        mock_exchange.fetch_ohlcv.return_value = [
            [0, 100.0, 105.0, 95.0, 102.0, 10.0],
            [3_600_000, 102.0, 103.0, 101.0, 102.5, 1.0],
        ]

        data = fetch_multi_timeframe_data(mock_exchange)
        self.assertEqual(len(data["ohlcv_1h"]), 1)
        self.assertEqual(len(data["ohlcv_15m"]), 1)
        self.assertEqual(data["ohlcv_1h"]["timestamp"].iloc[-1], 0)

    def test_live_candle_store_updates_only_on_closed_candles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "raw_1h.csv")
            store = LiveCandleStore(limits={"1d": 5, "1h": 5, "15m": 5}, save_1h_csv_path=csv_path)

            # Forming (unclosed) 15m candle should not append to closed frames
            closed = store.apply_kline_event("15m", {
                "t": 1_000_000, "o": "90000", "h": "90200", "l": "89900", "c": "90100", "v": "12.5", "x": False
            })
            self.assertFalse(closed)
            self.assertTrue(store.frames["15m"].empty)
            self.assertEqual(store.live_candles["15m"]["close"], 90100.0)

            # Closed 15m candle should append to confirmed frame
            closed = store.apply_kline_event("15m", {
                "t": 1_000_000, "o": "90000", "h": "90300", "l": "89900", "c": "90250", "v": "25.0", "x": True
            })
            self.assertTrue(closed)
            self.assertEqual(len(store.frames["15m"]), 1)
            self.assertEqual(float(store.frames["15m"]["close"].iloc[-1]), 90250.0)

            # Closed 1H candle should also sync to CSV
            closed_1h = store.apply_kline_event("1h", {
                "t": 3_600_000, "o": "90000", "h": "90500", "l": "89800", "c": "90400", "v": "150.0", "x": True
            })
            self.assertTrue(closed_1h)
            self.assertTrue(os.path.exists(csv_path))
            saved_df = pd.read_csv(csv_path)
            self.assertEqual(len(saved_df), 1)
            self.assertEqual(float(saved_df["close"].iloc[0]), 90400.0)

    def test_build_binance_stream_url(self):
        url = build_binance_stream_url("BTC/USDT", ("15m", "1h", "1d"))
        self.assertEqual(
            url,
            "wss://stream.binance.com:9443/stream?streams=btcusdt@kline_15m/btcusdt@kline_1h/btcusdt@kline_1d",
        )

    def test_conditional_graph_skips_sentiment_when_technical_is_hold(self):
        with patch("graph_builder.sentiment_analyst_agent") as mock_sentiment:
            graph = build_trading_graph()
            initial_state = self.make_state()  # Missing OHLCV -> tech_agent outputs HOLD
            result = graph.invoke(initial_state)
            self.assertEqual(result["technical_signal"], "HOLD")
            self.assertEqual(result["final_decision"], "HOLD")
            self.assertEqual(result["execution_status"], "SKIPPED")
            mock_sentiment.assert_not_called()

    def test_telegram_signal_formatting_and_cooldown(self):
        reset_signal_cooldown()
        state = self.make_state(
            final_decision="BUY",
            current_price=91000.0,
            stop_loss=90500.0,
            position_size=0.4,
            capital_allocation=5000.0,
            technical_reason="ASCENDING_TRIANGLE_BREAKOUT; BULLISH_PIN_BAR",
        )
        msg = format_trade_signal_message(state)
        self.assertIn("NEURALPULSE VIP TRADE SIGNAL", msg)
        self.assertIn("🟢 BUY / LONG", msg)
        self.assertIn("Perpetual (Futures)", msg)
        self.assertIn("ISOLATED", msg)
        self.assertIn("$91,000.00", msg)
        self.assertIn("Take Profit 1", msg)
        self.assertIn("Take Profit 2", msg)
        self.assertIn("Take Profit 3", msg)

        mock_post = MagicMock()
        mock_post.ok = True
        with patch("agents.telegram_notifier.requests.post", return_value=mock_post):
            sent = send_telegram_signal(msg, bot_token="123:ABC", chat_id="@test_channel")
            self.assertTrue(sent)

        self.assertFalse(is_in_cooldown("BUY", current_ts=1000.0))
        mark_signal_sent("BUY", 91000.0, current_ts=1000.0)
        self.assertTrue(is_in_cooldown("BUY", current_ts=1100.0))
        self.assertFalse(is_in_cooldown("SELL", current_ts=1100.0))

        self.assertGreater(analyze_headlines_vader(["Bitcoin surges to record high after strong ETF inflows"]), 0.0)

    def test_liquidity_sweep_and_htf_momentum_detection(self):
        # Need at least lookback (24) + 2 = 26 rows; final bar sweeps low to 98.3 then closes above 99 at 100.2
        rows = 28
        opens = [100.0] * (rows - 1) + [99.6]
        highs = [102.0] * (rows - 1) + [100.5]
        lows = [99.0] * (rows - 1) + [98.3]
        closes = [100.5 + (i % 2) * 0.2 for i in range(rows - 1)] + [100.2]
        df_bull_sweep = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes, "volume": [100.0] * rows
        })
        self.assertEqual(detect_liquidity_sweep(df_bull_sweep), "BULLISH_LIQUIDITY_SWEEP")

        # Bearish sweep above 102.0 prior high followed by close back below 102.0
        df_bear_sweep = pd.DataFrame({
            "open": [100.0] * (rows - 1) + [101.6],
            "high": [102.0] * (rows - 1) + [102.8],
            "low": [99.0] * (rows - 1) + [100.9],
            "close": [100.5 + (i % 2) * 0.2 for i in range(rows - 1)] + [101.0],
            "volume": [100.0] * rows,
        })
        self.assertEqual(detect_liquidity_sweep(df_bear_sweep), "BEARISH_LIQUIDITY_SWEEP")

        # 1H uptrend bias check
        uptrend_1h = pd.DataFrame({
            "open": [90000 + i * 50 for i in range(60)],
            "high": [90080 + i * 50 for i in range(60)],
            "low": [89920 + i * 50 for i in range(60)],
            "close": [90020 + i * 50 for i in range(60)],
            "volume": [500.0] * 60,
        })
        self.assertEqual(analyze_htf_momentum_alignment(uptrend_1h)["bias"], "BULLISH")

    def test_technical_agent_min_sl_floor_and_opposing_candle_penalty(self):
        # Tight ATR of 10.0 on $50,000 BTC (0.02%) should be raised to MIN_SL_PCT 0.5% ($250.0)
        with (
            patch("agents.technical_agent.calculate_dynamic_risk", return_value={
                "atr_value": 6.67, "recommended_sl_distance": 10.0
            }),
            patch("agents.technical_agent.detect_candlestick_signal", return_value="BULLISH_ENGULFING"),
            patch("agents.technical_agent.analyze_volume_spike", return_value={"is_spike": True, "ratio": 2.0}),
            patch("agents.technical_agent.find_angled_trendlines", return_value=None),
            patch("agents.technical_agent.find_support_resistance_zones", return_value={
                "support": [50000.0], "resistance": [51000.0]
            }),
            patch("agents.technical_agent.detect_market_regime", return_value="STRONG_BULLISH"),
        ):
            agent = TechnicalAgent()
            decision = agent.analyze_multi_timeframe(
                pd.DataFrame({"open": [50000.0]*200, "high": [50100.0]*200, "low": [49900.0]*200, "close": [50000.0]*200, "volume": [100.0]*200}),
                pd.DataFrame({"open": [50000.0]*30, "high": [50100.0]*30, "low": [49900.0]*30, "close": [50000.0]*30, "volume": [100.0]*30}),
                pd.DataFrame({"open": [50000.0]*20, "high": [50100.0]*20, "low": [49900.0]*20, "close": [50000.0]*20, "volume": [100.0]*20}),
            )
            self.assertEqual(decision["signal"], "BUY")
            self.assertEqual(decision["recommended_stop_loss"], 49750.0)
            self.assertEqual(decision["recommended_take_profit_1"], 50375.0)
            self.assertEqual(decision["recommended_take_profit"], 50500.0)
            self.assertEqual(decision["recommended_take_profit_3"], 50750.0)

        # Opposing BEARISH_ENGULFING during BUY trend penalizes score so trade stays HOLD
        with (
            patch("agents.technical_agent.calculate_dynamic_risk", return_value={
                "atr_value": 100.0, "recommended_sl_distance": 300.0
            }),
            patch("agents.technical_agent.detect_candlestick_signal", return_value="BEARISH_ENGULFING"),
            patch("agents.technical_agent.analyze_volume_spike", return_value={"is_spike": True, "ratio": 2.0}),
            patch("agents.technical_agent.find_angled_trendlines", return_value=None),
            patch("agents.technical_agent.find_support_resistance_zones", return_value={
                "support": [50000.0], "resistance": [51000.0]
            }),
            patch("agents.technical_agent.detect_market_regime", return_value="STRONG_BULLISH"),
        ):
            agent = TechnicalAgent()
            decision = agent.analyze_multi_timeframe(
                pd.DataFrame({"open": [50000.0]*200, "high": [50100.0]*200, "low": [49900.0]*200, "close": [50000.0]*200, "volume": [100.0]*200}),
                pd.DataFrame({"open": [50000.0]*30, "high": [50100.0]*30, "low": [49900.0]*30, "close": [50000.0]*30, "volume": [100.0]*30}),
                pd.DataFrame({"open": [50000.0]*20, "high": [50100.0]*20, "low": [49900.0]*20, "close": [50000.0]*20, "volume": [100.0]*20}),
            )
            self.assertEqual(decision["signal"], "HOLD")
            self.assertEqual(decision["score"], 15)

    def test_risk_arbiter_computes_futures_leverage_margin_and_liquidation(self):
        mem = TradeMemory(":memory:", initial_equity=10000.0)
        with patch("agents.risk_arbiter.fetch_futures_market_context", return_value={"available": False}):
            # Wide 5% SL should automatically scale leverage down to 8x (below default 10x cap)
            wide_sl_state = self.make_state(
                technical_signal="BUY",
                sentiment_score=0.1,
                current_price=50000.0,
                stop_loss=47500.0,
            )
            res = risk_arbiter_agent(wide_sl_state, memory=mem)
            self.assertEqual(res["final_decision"], "BUY")
            ind = res["indicators"]
            self.assertEqual(ind["market_type"], "FUTURES")
            self.assertEqual(ind["margin_mode"], "ISOLATED")
            self.assertEqual(ind["recommended_leverage"], 8)
            self.assertLess(ind["est_liquidation_price"], res["stop_loss"])
            self.assertGreater(ind["notional_value_usd"], ind["margin_required_usd"])

            # Tight 0.5% SL on SHORT should use up to max 10x leverage with liquidation safely above SL
            ind_s = dict(self.make_state()["indicators"])
            ind_s.update({"ml_signal": "SELL", "ml_confidence": 0.78, "trend_1h_override": "SELL", "trend_4h_override": "SELL"})
            short_state = self.make_state(
                technical_signal="SELL",
                sentiment_score=-0.2,
                current_price=50000.0,
                stop_loss=50250.0,
                indicators=ind_s,
            )
            res_short = risk_arbiter_agent(short_state, memory=mem)
            self.assertEqual(res_short["final_decision"], "SELL")
            self.assertEqual(res_short["indicators"]["recommended_leverage"], 10)
            self.assertGreater(res_short["indicators"]["est_liquidation_price"], res_short["stop_loss"])

    def test_sniper_hard_regime_filters_and_drawdown_shield(self):
        mem = TradeMemory(":memory:", initial_equity=10000.0)
        with patch("agents.risk_arbiter.fetch_futures_market_context", return_value={"available": False}):
            # 1. ADX outside 20-35 sweet spot -> Auto-Reject
            low_adx = self.make_state(technical_signal="BUY")
            low_adx["indicators"]["adx_14_override"] = 18.0
            res_low_adx = risk_arbiter_agent(low_adx, memory=mem)
            self.assertEqual(res_low_adx["final_decision"], "HOLD")
            self.assertEqual(res_low_adx["indicators"]["rejection_reason"], "Rejected: ADX outside 20-35 sweet spot")

            high_adx = self.make_state(technical_signal="BUY")
            high_adx["indicators"]["adx_14_override"] = 38.0
            res_high_adx = risk_arbiter_agent(high_adx, memory=mem)
            self.assertEqual(res_high_adx["final_decision"], "HOLD")
            self.assertEqual(res_high_adx["indicators"]["rejection_reason"], "Rejected: ADX outside 20-35 sweet spot")

            # 2. Volume < 1.35x -> Auto-Reject
            low_vol = self.make_state(technical_signal="BUY")
            low_vol["indicators"]["volume_ratio_override"] = 1.2
            self.assertEqual(risk_arbiter_agent(low_vol, memory=mem)["final_decision"], "HOLD")

            # 3. 1H vs 4H EMA conflict -> Auto-Reject
            conflict = self.make_state(technical_signal="BUY")
            conflict["indicators"]["trend_4h_override"] = "SELL"
            self.assertEqual(risk_arbiter_agent(conflict, memory=mem)["final_decision"], "HOLD")

            # 4. ML confidence < 65% -> Auto-Reject
            low_ml = self.make_state(technical_signal="BUY")
            low_ml["indicators"]["ml_confidence"] = 0.60
            self.assertEqual(risk_arbiter_agent(low_ml, memory=mem)["final_decision"], "HOLD")

            # 5. Mandatory Structural Pattern Gate (No NONE/NEUTRAL/empty patterns)
            for bad_pat in ([], ["NONE"], ["NEUTRAL"], ["NO_PATTERN"], ["N/A"]):
                no_pat = self.make_state(technical_signal="BUY")
                no_pat["indicators"]["detected_patterns"] = bad_pat
                res_pat = risk_arbiter_agent(no_pat, memory=mem)
                self.assertEqual(res_pat["final_decision"], "HOLD")
                self.assertEqual(
                    res_pat["indicators"]["rejection_reason"],
                    "Rejected: No valid price action pattern detected (NONE)",
                )

            # 6. Anti-Chasing / Overextended Trigger Candle (> 1.8x ATR)
            overext = self.make_state(technical_signal="BUY")
            overext["indicators"]["atr_14_override"] = 100.0
            overext["indicators"]["candle_range_override"] = 185.0
            res_ext = risk_arbiter_agent(overext, memory=mem)
            self.assertEqual(res_ext["final_decision"], "HOLD")
            self.assertEqual(
                res_ext["indicators"]["rejection_reason"],
                "Rejected: Trigger candle overextended (> 1.8x ATR)",
            )

            # 6b. EMA20 Overextension (> 1.3x ATR)
            ema_ext = self.make_state(technical_signal="BUY", current_price=50000.0)
            ema_ext["indicators"]["atr_14_override"] = 100.0
            ema_ext["indicators"]["ema_20_override"] = 49800.0  # abs(50000 - 49800) = 200 > 130
            res_ema = risk_arbiter_agent(ema_ext, memory=mem)
            self.assertEqual(res_ema["final_decision"], "HOLD")
            self.assertEqual(res_ema["indicators"]["rejection_reason"], "Rejected: Price > 1.3x ATR from EMA20")

            # 6c. Tag Regime Mode: ROTATIONAL (20 <= ADX < 26) vs EXPANSION (26 <= ADX <= 35)
            rot_state = self.make_state(technical_signal="BUY")
            rot_state["indicators"]["adx_14_override"] = 23.0
            res_rot = risk_arbiter_agent(rot_state, memory=mem)
            self.assertEqual(res_rot["final_decision"], "BUY")
            self.assertEqual(res_rot["indicators"]["regime_mode"], "ROTATIONAL")

            exp_state = self.make_state(technical_signal="BUY")
            exp_state["indicators"]["adx_14_override"] = 30.0
            res_exp = risk_arbiter_agent(exp_state, memory=mem)
            self.assertEqual(res_exp["final_decision"], "BUY")
            self.assertEqual(res_exp["indicators"]["regime_mode"], "EXPANSION")

            # 7. Anti-Exhaustion RSI Filter (Short < 34 oversold bear trap, Long > 70 overbought bull trap)
            oversold_short = self.make_state(technical_signal="SELL", sentiment_score=-0.20)
            oversold_short["indicators"]["rsi_14_override"] = 30.0
            oversold_short["indicators"]["trend_1h_override"] = "SELL"
            oversold_short["indicators"]["trend_4h_override"] = "SELL"
            res_short = risk_arbiter_agent(oversold_short, memory=mem)
            self.assertEqual(res_short["final_decision"], "HOLD")
            self.assertIn("oversold / bear-trap risk", res_short["indicators"]["rejection_reason"])

            overbought_long = self.make_state(technical_signal="BUY", sentiment_score=0.20)
            overbought_long["indicators"]["rsi_14_override"] = 75.0
            res_long = risk_arbiter_agent(overbought_long, memory=mem)
            self.assertEqual(res_long["final_decision"], "HOLD")
            self.assertIn("overbought / bull-trap risk", res_long["indicators"]["rejection_reason"])

            # 8. Drawdown Shield: 3 consecutive losses cut risk % in half
            for k in range(3):
                tid = mem.log_entry({
                    "entry_time": f"2026-11-10T0{k}:00:00+00:00",
                    "side": "LONG", "entry_price": 50000.0, "initial_sl": 49500.0, "risk_pct": 0.01,
                })
                mem.log_exit(tid, {
                    "exit_time": f"2026-11-10T0{k}:30:00+00:00",
                    "exit_state": "INITIAL", "exit_reason": "INITIAL_SL",
                    "mfe_r": 0.1, "mae_r": -1.0, "total_candles_held": 2,
                    "net_pnl_usd": -100.0, "net_r_multiple": -1.0, "fees_paid_usd": 5.0,
                })
            self.assertTrue(mem.is_drawdown_shield_active(3))
            shielded = risk_arbiter_agent(self.make_state(technical_signal="BUY"), memory=mem)
            self.assertEqual(shielded["final_decision"], "BUY")
            self.assertTrue(shielded["indicators"]["drawdown_shield_active"])
            self.assertIn(shielded["indicators"]["risk_pct"], (0.5, 1.0))

            # 9. Fee-to-ATR Viability Gate: Reject if TP1 distance < 3 * 0.0012 (0.36%)
            tight_tp1 = self.make_state(technical_signal="BUY")
            tight_tp1["indicators"]["tp1_distance_override"] = 0.0020
            res_tight = risk_arbiter_agent(tight_tp1, memory=mem)
            self.assertEqual(res_tight["final_decision"], "HOLD")
            self.assertIn("Fee-to-ATR Viability Gate failed", res_tight["indicators"]["rejection_reason"])

    def test_multi_horizon_risk_parameters_and_execution(self):
        mem = TradeMemory(":memory:", initial_equity=10000.0)
        with patch("agents.risk_arbiter.fetch_futures_market_context", return_value={"available": False}):
            # SCALP: SL floor 0.35%, max leverage 15x
            scalp_state = self.make_state(technical_signal="BUY", trade_horizon="SCALP")
            scalp_res = risk_arbiter_agent(scalp_state, memory=mem)
            self.assertEqual(scalp_res["final_decision"], "BUY")
            self.assertEqual(scalp_res["indicators"]["trade_horizon"], "SCALP")
            self.assertEqual(scalp_res["indicators"]["execution_timeframe"], "5m")
            self.assertEqual(scalp_res["indicators"]["recommended_leverage"], 15)
            self.assertEqual(scalp_res["indicators"]["horizon_sl_floor_pct"], 0.0035)

            # SWING: SL floor 0.80%, max leverage 5x
            swing_state = self.make_state(technical_signal="BUY", trade_horizon="SWING")
            swing_res = risk_arbiter_agent(swing_state, memory=mem)
            self.assertEqual(swing_res["final_decision"], "BUY")
            self.assertEqual(swing_res["indicators"]["trade_horizon"], "SWING")
            self.assertEqual(swing_res["indicators"]["execution_timeframe"], "1h")
            self.assertEqual(swing_res["indicators"]["recommended_leverage"], 5)
            self.assertEqual(swing_res["indicators"]["horizon_sl_floor_pct"], 0.0080)

            # Adaptive TIME_STOP: SCALP uses 20 candles (5m), SWING uses 12 candles (1h)
            scalp_pos = TradePosition(
                trade_id="scalp_01", side="BUY", entry_price=50000.0, initial_sl=49800.0,
                notional_usd=5000.0, horizon="SCALP", timeframe="5m", risk_usd=100.0,
            )
            for k in range(19):
                scalp_pos.step_candle(high=50050.0, low=49950.0, close=50000.0, atr_14=20.0, candle_time=f"2026-11-12T00:{k:02d}:00Z")
            self.assertFalse(scalp_pos.closed)
            scalp_pos.step_candle(high=50050.0, low=49950.0, close=50000.0, atr_14=20.0, candle_time="2026-11-12T00:20:00Z")
            self.assertTrue(scalp_pos.closed)
            self.assertEqual(scalp_pos.state, STATE_TIME_STOP)

            swing_pos = TradePosition(
                trade_id="swing_01", side="BUY", entry_price=50000.0, initial_sl=49000.0,
                notional_usd=5000.0, horizon="SWING", timeframe="1h", risk_usd=100.0,
            )
            for k in range(11):
                swing_pos.step_candle(high=50100.0, low=49900.0, close=50000.0, atr_14=100.0, candle_time=f"2026-11-12T{k:02d}:00:00Z")
            self.assertFalse(swing_pos.closed)
            swing_pos.step_candle(high=50100.0, low=49900.0, close=50000.0, atr_14=100.0, candle_time="2026-11-12T12:00:00Z")
            self.assertTrue(swing_pos.closed)
            self.assertEqual(swing_pos.state, STATE_TIME_STOP)

    def test_five_stage_trade_lifecycle_and_autopsy(self):
        # 1. Mode B: EXPANSION (+1.4R / +1.7R Ratchet / +2.4R / Chandelier Runner)
        pos = TradePosition(
            trade_id="t001", side="BUY", entry_price=50000.0, initial_sl=49500.0,
            notional_usd=10000.0, timeframe="15m", risk_usd=100.0, regime_mode="EXPANSION",
        )
        self.assertEqual(pos.state, STATE_INITIAL)
        # Hit +1.4R (50700)
        ev1 = pos.step_candle(high=50750.0, low=49900.0, close=50700.0, atr_14=150.0, candle_time="2026-11-12T01:00:00+00:00")
        self.assertEqual(pos.state, STATE_RETEST_SHIELD)
        self.assertEqual(pos.current_sl, 49900.0)  # -0.20R (50000 - 0.20*500)
        self.assertAlmostEqual(pos.remaining_fraction, 0.65)
        self.assertTrue(any(e["reason"] == "TP1_RETEST_SHIELD" for e in ev1))

        # Intermediate Ratchet at +1.7R (50850) -> Ratchet SL of remaining 65% to +0.40R (50200)
        ev_ratchet = pos.step_candle(high=50900.0, low=50400.0, close=50850.0, atr_14=150.0, candle_time="2026-11-12T01:10:00+00:00")
        self.assertEqual(pos.state, STATE_RETEST_SHIELD)
        self.assertEqual(pos.current_sl, 50200.0)  # +0.40R (50000 + 0.40*500)
        self.assertAlmostEqual(pos.remaining_fraction, 0.65)
        self.assertTrue(any(e["reason"] == "INTERMEDIATE_RATCHET_LOCK" for e in ev_ratchet))

        # Hit +2.4R (51200) -> transitions to PROFIT_LOCK (+1.00R = 50500) & CHANDELIER_RUNNER
        ev2 = pos.step_candle(high=51300.0, low=50600.0, close=51250.0, atr_14=150.0, candle_time="2026-11-12T01:15:00+00:00")
        self.assertEqual(pos.state, STATE_CHANDELIER_RUNNER)
        self.assertAlmostEqual(pos.remaining_fraction, 0.35)
        self.assertGreaterEqual(pos.current_sl, 50500.0)
        self.assertTrue(any(e["reason"] == "TP2_PROFIT_LOCK" for e in ev2))

        # Chandelier trail hit
        ev3 = pos.step_candle(high=51250.0, low=50000.0, close=50050.0, atr_14=150.0, candle_time="2026-11-12T01:30:00+00:00")
        self.assertTrue(pos.closed)
        self.assertEqual(pos.exit_reason, "CHANDELIER_TRAIL_STOP")
        self.assertIn("TRADE CLOSED", format_lifecycle_event_message(ev3[0]))

        # 1b. Mode A: ROTATIONAL (+1.1R / +1.8R full exit)
        rot_pos = TradePosition(
            trade_id="rot01", side="BUY", entry_price=50000.0, initial_sl=49500.0,
            notional_usd=10000.0, timeframe="15m", risk_usd=100.0, regime_mode="ROTATIONAL",
        )
        # Hit +1.1R (50550) -> Close 50%, SL to +0.15R (50075)
        ev_r1 = rot_pos.step_candle(high=50600.0, low=49900.0, close=50550.0, atr_14=150.0, candle_time="2026-11-12T02:00:00+00:00")
        self.assertEqual(rot_pos.state, STATE_RETEST_SHIELD)
        self.assertEqual(rot_pos.current_sl, 50075.0)  # +0.15R (50000 + 0.15*500)
        self.assertAlmostEqual(rot_pos.remaining_fraction, 0.50)

        # Hit +1.8R (50900) -> Close remaining 50% (100% full exit at target, no runner)
        ev_r2 = rot_pos.step_candle(high=50950.0, low=50400.0, close=50900.0, atr_14=150.0, candle_time="2026-11-12T02:15:00+00:00")
        self.assertTrue(rot_pos.closed)
        self.assertEqual(rot_pos.state, STATE_PROFIT_LOCK)
        self.assertEqual(rot_pos.remaining_fraction, 0.0)

        # 2. TIME_STOP after 16 candles on 15m without reaching +0.6R (EXPANSION)
        stale = TradePosition(
            trade_id="t002", side="BUY", entry_price=50000.0, initial_sl=49500.0,
            notional_usd=10000.0, timeframe="15m", risk_usd=100.0, regime_mode="EXPANSION",
        )
        for k in range(16):
            stale.step_candle(high=50200.0, low=49800.0, close=50050.0, atr_14=150.0, candle_time=f"2026-11-12T02:{k:02d}:00+00:00")
        self.assertTrue(stale.closed)
        self.assertEqual(stale.state, STATE_TIME_STOP)

        # 3. SQLite Post-Exit Autopsy (INITIAL_SL loss later hits +1.5R -> was_stop_hunt = 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "autopsy.db")
            mem = TradeMemory(db_path=db_path, initial_equity=10000.0)
            tid = mem.log_entry({
                "entry_time": "2026-11-12T00:00:00+00:00",
                "side": "LONG", "entry_price": 50000.0, "initial_sl": 49500.0, "risk_pct": 0.01,
            })
            mem.log_exit(tid, {
                "exit_time": "2026-11-12T01:00:00+00:00",
                "exit_state": STATE_INITIAL, "exit_reason": "INITIAL_SL",
                "mfe_r": 0.1, "mae_r": -1.0, "total_candles_held": 3,
                "net_pnl_usd": -105.0, "net_r_multiple": -1.05, "fees_paid_usd": 12.0,
            })
            mem.process_due_autopsies(current_price=50800.0, current_time="2026-11-12T05:30:00+00:00", window_high=50800.0, window_low=49600.0)
            mem.process_due_autopsies(current_price=50900.0, current_time="2026-11-12T13:30:00+00:00", window_high=50900.0, window_low=50100.0)
            report_text = generate_diagnostic_report(db_path)
            self.assertIn("Stop-Hunts (later hit +1.5R): 1 (100.0%)", report_text)


if __name__ == "__main__":
    unittest.main()