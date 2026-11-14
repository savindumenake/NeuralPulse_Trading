"""
backtest.py - Sniper Mode Historical Backtester (5-Stage Lifecycle + TradeMemory)

Replays historical BTC/USDT 15m candles from data/btc_usdt_15m_raw.csv through:
1. TechnicalAgent (1D/1H/15m) with dynamic multi-timeframe aggregation
2. Sniper RiskArbiter (Hard Regime Filters + Anti-Exhaustion RSI + 0-100 Confluence + Wick-Proof SL + Equity Sizing)
3. 5-Stage Automated Trade Lifecycle State Machine (`TradePosition` with 16 candles TIME_STOP on 15m)
4. SQLite `TradeMemory` logging + 4h/12h Stop-Hunt Autopsies (`data/backtest_memory.db`)

Usage:
    python backtest.py
"""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import xgboost as xgb

from agents.risk_arbiter import risk_arbiter_agent
from agents.state import TradingState
from agents.technical_agent import TechnicalAgent, technical_analyst_agent
from agents.trade_lifecycle import TradePosition
from agents.trade_memory import DATA_DIR, TradeMemory, infer_market_session
from data_loader import calculate_technical_features, get_feature_column_names

DATA_PATH = os.path.join(DATA_DIR, "btc_usdt_15m_raw.csv")
BACKTEST_DB_PATH = os.path.join(DATA_DIR, "backtest_memory.db")


def _train_15m_xgboost_model(df_history: pd.DataFrame) -> xgb.Booster | None:
    """Train/calibrate an XGBoost model on historical 15m closed candles."""
    try:
        featured = calculate_technical_features(df_history, is_training=True)
        feature_cols = get_feature_column_names()
        valid = featured.dropna(subset=feature_cols + ["target"])
        if len(valid) < 100:
            return None
        X = valid[feature_cols]
        y = valid["target"]
        clf = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
        )
        clf.fit(X, y)
        return clf.get_booster()
    except Exception as e:
        print(f"[Backtest] Warning during 15m XGBoost training: {e}")
        return None


def _build_proxy_1d(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    """Build proxy 1D daily bars calibrated for 15m candles (200/800 bars = 50h/200h)."""
    fast_span = min(idx, 200)
    slow_span = min(idx, 800)
    ema_fast = float(df["close"].iloc[idx - fast_span:idx + 1].mean())
    ema_slow = float(df["close"].iloc[idx - slow_span:idx + 1].mean())
    if ema_fast > ema_slow * 1.005:
        return pd.DataFrame({
            "open": np.linspace(50000, 90000, 200),
            "high": np.linspace(50500, 90500, 200),
            "low": np.linspace(49500, 89500, 200),
            "close": np.linspace(50000, 90000, 200),
            "volume": [1000.0] * 200,
        })
    if ema_fast < ema_slow * 0.995:
        return pd.DataFrame({
            "open": np.linspace(90000, 50000, 200),
            "high": np.linspace(90500, 50500, 200),
            "low": np.linspace(89500, 49500, 200),
            "close": np.linspace(90000, 50000, 200),
            "volume": [1000.0] * 200,
        })
    return pd.DataFrame({
        "open": [70000.0] * 200,
        "high": [70500.0] * 200,
        "low": [69500.0] * 200,
        "close": [70000.0] * 200,
        "volume": [1000.0] * 200,
    })


def run_backtest(db_path: str = BACKTEST_DB_PATH, initial_capital: float = 10000.0) -> dict:
    """
    Scan historical 15m BTC data with Sniper Mode gating, 5-stage TradePosition
    management, and TradeMemory SQLite persistence.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Historical 15m data not found at {DATA_PATH}")
    if db_path != ":memory:" and os.path.exists(db_path):
        os.remove(db_path)

    df = pd.read_csv(DATA_PATH)
    if "datetime" not in df.columns and "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    df["dt_parsed"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("dt_parsed").reset_index(drop=True)

    atr_series = df.ta.atr(length=14)
    memory = TradeMemory(db_path=db_path, initial_equity=initial_capital)

    active_pos: TradePosition | None = None
    trades = []

    # Need at least 1,000 candles for 15m ML training and MTF structure
    start_idx = 1000

    print(f"\n[Backtest] Fitting initial 15m XGBoost model on first {start_idx} candles...")
    initial_booster = _train_15m_xgboost_model(df.iloc[:start_idx])
    if initial_booster is not None:
        TechnicalAgent.set_booster(initial_booster)
        print("[Backtest] 15m XGBoost model successfully fitted and loaded.")

    with patch("agents.risk_arbiter.fetch_futures_market_context", return_value={"available": False}):
        for i in range(start_idx, len(df)):
            # Walk-forward model recalibration every 1,000 candles
            if (i - start_idx) > 0 and (i - start_idx) % 1000 == 0:
                wf_booster = _train_15m_xgboost_model(df.iloc[:i])
                if wf_booster is not None:
                    TechnicalAgent.set_booster(wf_booster)

            row = df.iloc[i]
            bar_time = str(row["datetime"])
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_close = float(row["close"])
            bar_atr = float(atr_series.iloc[i]) if atr_series is not None and pd.notna(atr_series.iloc[i]) else 0.0

            memory.process_due_autopsies(
                current_price=bar_close,
                current_time=bar_time,
                window_high=bar_high,
                window_low=bar_low,
            )

            if active_pos is not None:
                active_pos.step_candle(
                    high=bar_high,
                    low=bar_low,
                    close=bar_close,
                    atr_14=bar_atr,
                    candle_time=bar_time,
                )
                if active_pos.closed:
                    summary = active_pos.to_lifecycle_summary()
                    equity_after = memory.log_exit(active_pos.trade_id, summary)
                    trades.append({
                        "trade_id": active_pos.trade_id,
                        "datetime": bar_time,
                        "direction": active_pos.side,
                        "entry": active_pos.entry_price,
                        "sl": active_pos.initial_sl,
                        "exit_state": summary["exit_state"],
                        "outcome": summary["exit_reason"],
                        "r_multiple": summary["net_r_multiple"],
                        "net_pnl_usd": summary["net_pnl_usd"],
                        "equity": equity_after,
                    })
                    active_pos = None
                continue

            if i >= len(df) - 16:
                continue

            # Zero-lookahead slice: history up to candle i (inclusive)
            sub_history = df.iloc[:i + 1]
            window_15m = sub_history.tail(120).copy().reset_index(drop=True)

            # Resample historical 15m bars to 1H and 4H strictly up to candle i
            sub_dt_indexed = sub_history.set_index("dt_parsed")
            window_1h = (
                sub_dt_indexed.resample("1h")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
                .reset_index()
                .tail(180)
                .reset_index(drop=True)
            )
            window_4h = (
                sub_dt_indexed.resample("4h")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
                .reset_index()
                .tail(100)
                .reset_index(drop=True)
            )

            if len(window_1h) < 60:
                continue

            proxy_1d = _build_proxy_1d(df, i)

            state = TradingState(
                symbol="BTC/USDT",
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
                ohlcv_1d=proxy_1d,
                ohlcv_1h=window_1h,
                ohlcv_15m=window_15m,
                ohlcv_4h=window_4h,
                ohlcv_ml=window_15m,
            )
            state = technical_analyst_agent(state)
            if state.get("technical_signal") not in ("BUY", "SELL"):
                continue

            state = risk_arbiter_agent(state, memory=memory)
            if state.get("final_decision") not in ("BUY", "SELL"):
                continue

            ind = state["indicators"]
            horizon_val = str(ind.get("trade_horizon") or state.get("trade_horizon") or "INTRADAY")
            trade_id = memory.log_entry({
                "entry_time": bar_time,
                "side": "LONG" if state["final_decision"] == "BUY" else "SHORT",
                "timeframe": "15m",
                "trade_horizon": horizon_val,
                "market_session": infer_market_session(bar_time),
                "confluence_score": float(ind.get("sniper_score", 0.0)),
                "xgb_confidence": float(ind.get("ml_confidence", 0.0)),
                "top_shap_features": ind.get("top_shap_features", []),
                "pattern_detected": (ind.get("detected_patterns") or ["NONE"])[0],
                "atr_pct": float(ind.get("atr_pct", 0.0)),
                "adx_14": float(ind.get("adx_14", 0.0)),
                "rsi_14": float(ind.get("rsi_14", 50.0)),
                "volume_ratio": float(ind.get("volume_ratio", 0.0)),
                "funding_rate": float(ind.get("funding_rate", 0.0)),
                "news_sentiment_score": float(state.get("sentiment_score", 0.0)),
                "entry_price": float(state["current_price"]),
                "initial_sl": float(state["stop_loss"]),
                "risk_pct": float(ind.get("risk_pct", 1.0)) / 100.0,
                "account_equity_before": float(ind.get("account_equity", initial_capital)),
            })
            active_pos = TradePosition(
                trade_id=trade_id,
                side=state["final_decision"],
                entry_price=float(state["current_price"]),
                initial_sl=float(state["stop_loss"]),
                notional_usd=float(ind["notional_value_usd"]),
                timeframe="15m",
                horizon=horizon_val,
                regime_mode=ind.get("regime_mode", "EXPANSION"),
                risk_usd=float(ind["risk_amount_usd"]),
            )

    if active_pos is not None and not active_pos.closed:
        last_row = df.iloc[-1]
        summary = active_pos.force_close(float(last_row["close"]), candle_time=str(last_row["datetime"]))
        memory.log_exit(active_pos.trade_id, summary)

    metrics = memory.get_cumulative_metrics()
    return {
        "total_candles": len(df),
        "total_signals": metrics["total_trades"],
        "wins": metrics["wins"],
        "losses": metrics["losses"],
        "win_rate_pct": metrics["win_rate"],
        "total_r_multiple": metrics["net_r"],
        "net_pnl_usd": metrics["net_pnl_usd"],
        "final_equity": metrics["account_equity"],
        "signals": trades,
    }


if __name__ == "__main__":
    print("=" * 72)
    print("  NeuralPulse Sniper Mode Backtester (15m) — 5-Stage Lifecycle & TradeMemory")
    print("=" * 72)
    report = run_backtest()
    print(f"\n📊 Candles Scanned:     {report['total_candles']}")
    print(f"🎯 Sniper Trades Fired: {report['total_signals']}")
    print(f"✅ Winning Trades:      {report['wins']}")
    print(f"❌ Losing Trades:       {report['losses']}")
    print(f"🏆 Win Rate:            {report['win_rate_pct']:.1f}%")
    print(f"💰 Net Profit (in R):   {report['total_r_multiple']:+.2f}R (${report['net_pnl_usd']:+,.2f})")
    print(f"🏦 Final Equity:        ${report['final_equity']:,.2f}\n")
    if report["signals"]:
        print("Trades Sample:")
        print("-" * 72)
        for s in report["signals"][-10:]:
            print(
                f"  [{s['datetime']}] {s['direction']:5s} @ ${s['entry']:,.2f} | "
                f"SL: ${s['sl']:,.2f} | {s['outcome']} ({s['r_multiple']:+.2f}R, ${s['net_pnl_usd']:+.2f})"
            )
