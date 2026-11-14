"""
risk_arbiter.py - Sniper Mode Confluence Engine, Dynamic Equity & Position Sizer (Node 3)

Implements:
1. Hard Regime Filters: ADX_14 >= 22, Breakout Volume >= 1.35x SMA20, 1H & 4H EMA alignment.
2. Weighted 0-100 Sniper Confluence Score:
   - Technical Structure & Pattern Quality (0-35)
   - XGBoost ML Confidence (0-35; requires >= 65% aligned confidence)
   - Volume & Volatility Alignment (0-15)
   - Fundamental/Sentiment & Derivatives Score (0-15)
3. Execution Gate & Drawdown Shield:
   - < 78: NO_TRADE (HOLD)
   - 78-87: A_GRADE -> 1.0% equity risk (0.5% under Drawdown Shield)
   - 88-100: S_TIER -> 2.0% equity risk (1.0% under Drawdown Shield)
4. Wick-Proof Stop-Loss & Fee-Adjusted Futures Sizing (40% max margin cap).
"""

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401
from dotenv import load_dotenv

from agents.futures_context import evaluate_funding_risk, fetch_futures_market_context
from agents.trade_memory import DEFAULT_DB_PATH, TradeMemory
from agents.state import TradingState

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RiskArbiter")

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000.0"))
REQUIRE_POSITIVE_FUNDAMENTALS = os.getenv("REQUIRE_POSITIVE_FUNDAMENTALS", "false").lower() == "true"
TRADING_MARKET_TYPE = os.getenv("TRADING_MARKET_TYPE", "FUTURES").strip().upper()
MAX_FUTURES_LEVERAGE = max(1, int(os.getenv("MAX_FUTURES_LEVERAGE", "15")))
EXTREME_FUNDING_RATE = float(os.getenv("EXTREME_FUNDING_RATE", "0.0005"))
ROUND_TRIP_FEE_SLIPPAGE = 0.0012  # 0.05% taker/side + 0.02% slippage
MAX_MARGIN_EQUITY_PCT = 0.40
SNIPER_MIN_SCORE = 78.0
SNIPER_S_TIER_SCORE = 88.0
MIN_ML_CONFIDENCE = 0.65

HORIZON_CONFIGS = {
    "SCALP": {
        "sl_floor_pct": 0.0035,        # 0.35% minimum SL distance for 5m scalps
        "max_leverage": min(MAX_FUTURES_LEVERAGE, 15), # up to 15x safe leverage
        "time_stop_candles": 20,       # 20 candles of 5m = 100 min
        "timeframe": "5m",
    },
    "INTRADAY": {
        "sl_floor_pct": 0.0050,        # 0.50% minimum SL distance for 15m intraday
        "max_leverage": min(MAX_FUTURES_LEVERAGE, 10), # up to 10x safe leverage
        "time_stop_candles": 16,       # 16 candles of 15m = 4 hours
        "timeframe": "15m",
    },
    "SWING": {
        "sl_floor_pct": 0.0080,        # 0.80% minimum SL distance for 1h/4h swing
        "max_leverage": min(MAX_FUTURES_LEVERAGE, 5),  # up to 5x safe leverage
        "time_stop_candles": 12,       # 12 candles of 1h = 12 hours
        "timeframe": "1h",
    },
}


def _resample_4h_from_1h(df_1h: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Build 4H OHLCV bars from 1H data when an explicit 4H frame is not provided."""
    if df_1h is None or len(df_1h) < 40:
        return None
    n_bars = len(df_1h) // 4
    trimmed = df_1h.iloc[len(df_1h) - n_bars * 4:].reset_index(drop=True)
    groups = trimmed.index // 4
    return pd.DataFrame({
        "open": trimmed["open"].groupby(groups).first().values,
        "high": trimmed["high"].groupby(groups).max().values,
        "low": trimmed["low"].groupby(groups).min().values,
        "close": trimmed["close"].groupby(groups).last().values,
        "volume": trimmed["volume"].groupby(groups).sum().values,
    })


def _ema_trend_direction(df: Optional[pd.DataFrame]) -> str:
    """Return 'BUY', 'SELL', or 'NEUTRAL' from EMA20 vs EMA50."""
    if df is None or len(df) < 20 or np.std(df["close"].values) == 0:
        return "NEUTRAL"
    closes = df["close"]
    ema20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
    latest = float(closes.iloc[-1])
    if ema20 > ema50 and latest >= ema50:
        return "BUY"
    if ema20 < ema50 and latest <= ema50:
        return "SELL"
    return "NEUTRAL"


def compute_regime_metrics(state: TradingState) -> Dict[str, Any]:
    """Extract ADX_14, volume ratio, ATR_14, RSI_14, and 1H/4H EMA alignment."""
    horizon = str(state.get("trade_horizon") or state.get("indicators", {}).get("trade_horizon", "INTRADAY")).upper()
    if horizon == "SCALP" and state.get("ohlcv_5m") is not None and not state["ohlcv_5m"].empty:
        df_entry = state.get("ohlcv_5m")
    elif horizon == "SWING" and state.get("ohlcv_1h") is not None and not state["ohlcv_1h"].empty:
        df_entry = state.get("ohlcv_1h")
    else:
        df_entry = state.get("ohlcv_15m")
    df_1h = state.get("ohlcv_1h")
    df_4h = state.get("ohlcv_4h") if state.get("ohlcv_4h") is not None else _resample_4h_from_1h(df_1h)

    vol_df = df_entry if (df_entry is not None and len(df_entry) >= 20) else df_1h
    adx_df = df_1h if (df_1h is not None and len(df_1h) >= 28) else df_entry

    adx_14 = 0.0
    if adx_df is not None and len(adx_df) >= 28 and np.std(adx_df["close"].values) > 0:
        adx_tbl = adx_df.ta.adx(length=14)
        if adx_tbl is not None and "ADX_14" in adx_tbl and not adx_tbl["ADX_14"].dropna().empty:
            adx_14 = float(adx_tbl["ADX_14"].dropna().iloc[-1])

    volume_ratio = 0.0
    if vol_df is not None and len(vol_df) >= 20:
        avg_vol = float(vol_df["volume"].rolling(20).mean().iloc[-1])
        cur_vol = float(vol_df["volume"].iloc[-1])
        volume_ratio = (cur_vol / avg_vol) if avg_vol > 0 else 0.0

    atr_df = df_entry if (df_entry is not None and len(df_entry) >= 15) else df_1h
    atr_14 = 0.0
    rsi_14 = 50.0
    candle_range = 0.0
    if atr_df is not None and not atr_df.empty:
        candle_range = abs(float(atr_df["high"].iloc[-1]) - float(atr_df["low"].iloc[-1]))
    if atr_df is not None and len(atr_df) >= 15 and np.std(atr_df["close"].values) > 0:
        atr_s = atr_df.ta.atr(length=14)
        if atr_s is not None and not atr_s.dropna().empty:
            atr_14 = float(atr_s.dropna().iloc[-1])
        rsi_s = atr_df.ta.rsi(length=14)
        if rsi_s is not None and not rsi_s.dropna().empty:
            rsi_14 = float(rsi_s.dropna().iloc[-1])

    ema_df = df_entry if (df_entry is not None and len(df_entry) >= 20) else df_1h
    ema_20 = 0.0
    close_price = 0.0
    if ema_df is not None and len(ema_df) >= 20 and np.std(ema_df["close"].values) > 0:
        ema_20 = float(ema_df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        close_price = float(ema_df["close"].iloc[-1])
    elif ema_df is not None and not ema_df.empty:
        close_price = float(ema_df["close"].iloc[-1])

    return {
        "adx_14": round(adx_14, 2),
        "volume_ratio": round(volume_ratio, 2),
        "atr_14": round(atr_14, 2),
        "candle_range": round(candle_range, 4),
        "rsi_14": round(rsi_14, 2),
        "ema_20": round(ema_20, 2),
        "close": round(close_price, 2),
        "trend_1h": _ema_trend_direction(df_1h),
        "trend_4h": _ema_trend_direction(df_4h),
    }


def calculate_wick_proof_sl(
    direction: str,
    entry: float,
    atr_14: float,
    df_entry: Optional[pd.DataFrame],
    fallback_sl: float,
    swing_lookback: int = 12,
    sl_floor_pct: float = 0.005,
) -> float:
    """
    Wick-Proof Stop-Loss:
    - LONG:  min(recent_swing_low, entry - 1.5 * ATR_14) - (0.2 * ATR_14)
    - SHORT: max(recent_swing_high, entry + 1.5 * ATR_14) + (0.2 * ATR_14)
    Enforces horizon-adaptive SL floor (sl_floor_pct).
    """
    if atr_14 <= 0 or not np.isfinite(atr_14):
        return round(float(fallback_sl), 2)

    if df_entry is not None and len(df_entry) >= 3:
        recent = df_entry.tail(swing_lookback)
        swing_low = float(recent["low"].min())
        swing_high = float(recent["high"].max())
    else:
        swing_low = entry - 1.5 * atr_14
        swing_high = entry + 1.5 * atr_14

    min_dist = entry * sl_floor_pct if entry > 0 else 0.0

    if direction == "BUY":
        raw_sl = min(swing_low, entry - 1.5 * atr_14) - (0.2 * atr_14)
        sl = min(raw_sl, entry - min_dist) if min_dist > 0 else raw_sl
        return round(max(0.01, sl), 2)
    raw_sl = max(swing_high, entry + 1.5 * atr_14) + (0.2 * atr_14)
    sl = max(raw_sl, entry + min_dist) if min_dist > 0 else raw_sl
    return round(sl, 2)


def evaluate_sniper_confluence(
    state: TradingState,
    regime_metrics: Dict[str, Any],
    derivatives: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate Hard Regime Filters + 0-100 Weighted Sniper Confluence Score.
    Returns dict with `approved`, `score`, `tier` ('A_GRADE'|'S_TIER'|'NO_TRADE'), and `reason`.
    """
    direction = state.get("technical_signal", "HOLD")
    if direction not in ("BUY", "SELL"):
        return {"approved": False, "score": 0.0, "tier": "NO_TRADE", "reason": "No technical setup"}

    indicators = state.get("indicators") or {}
    sentiment = float(state.get("sentiment_score", 0.0))

    # Sentiment hard gate
    buy_sentiment_ok = (sentiment >= 0.05) if REQUIRE_POSITIVE_FUNDAMENTALS else (sentiment > -0.10)
    sell_sentiment_ok = (sentiment <= -0.05) if REQUIRE_POSITIVE_FUNDAMENTALS else (sentiment < 0.10)
    if (direction == "BUY" and not buy_sentiment_ok) or (direction == "SELL" and not sell_sentiment_ok):
        return {
            "approved": False,
            "score": 0.0,
            "tier": "NO_TRADE",
            "reason": f"Sentiment {sentiment:+.2f} opposes {direction}",
        }

    # Hard Regime Filter 1: ADX Sweet-Spot (20.0 <= adx_14 <= 35.0)
    adx_14 = float(indicators.get("adx_14_override", regime_metrics["adx_14"]))
    if adx_14 < 20.0 or adx_14 > 35.0:
        return {
            "approved": False,
            "score": 0.0,
            "tier": "NO_TRADE",
            "reason": "Rejected: ADX outside 20-35 sweet spot",
        }

    # Hard Regime Filter 2: Breakout Volume >= 1.35x SMA20
    vol_ratio = float(indicators.get("volume_ratio_override", regime_metrics["volume_ratio"]))
    if vol_ratio < 1.35:
        return {
            "approved": False,
            "score": 0.0,
            "tier": "NO_TRADE",
            "reason": f"Hard Regime Reject: Volume {vol_ratio:.2f}x < 1.35x average",
        }

    # Hard Regime Filter 3: 1H and 4H EMA trends must not conflict (and must support direction)
    t1h = indicators.get("trend_1h_override", regime_metrics["trend_1h"])
    t4h = indicators.get("trend_4h_override", regime_metrics["trend_4h"])
    if t1h != "NEUTRAL" and t4h != "NEUTRAL" and t1h != t4h:
        return {
            "approved": False,
            "score": 0.0,
            "tier": "NO_TRADE",
            "reason": f"Hard Regime Reject: 1H ({t1h}) and 4H ({t4h}) EMA trends conflict",
        }
    if (t1h != "NEUTRAL" and t1h != direction) or (t4h != "NEUTRAL" and t4h != direction):
        return {
            "approved": False,
            "score": 0.0,
            "tier": "NO_TRADE",
            "reason": f"Hard Regime Reject: HTF EMA trend ({t1h}/{t4h}) opposes {direction}",
        }

    # Hard Regime Filter 4: Mandatory Structural Pattern Gate (No NONE/NEUTRAL patterns)
    invalid_patterns = {"", "NONE", "NEUTRAL", "NO_PATTERN", "NO_CLEAR_PATTERN", "N/A"}
    valid_patterns = [
        str(p).strip().upper()
        for p in (indicators.get("detected_patterns") or [])
        if str(p).strip().upper() not in invalid_patterns
    ]
    explicit_pattern = str(indicators.get("pattern_detected") or indicators.get("pattern") or "").strip().upper()
    if explicit_pattern and explicit_pattern not in invalid_patterns and explicit_pattern not in valid_patterns:
        valid_patterns.append(explicit_pattern)
    if not valid_patterns:
        return {
            "approved": False,
            "score": 0.0,
            "tier": "NO_TRADE",
            "reason": "Rejected: No valid price action pattern detected (NONE)",
        }

    # Hard Regime Filter 5: Anti-Chasing / Overextended Trigger Candle (> 1.8x ATR)
    atr_14 = float(indicators.get("atr_14_override", regime_metrics.get("atr_14", 0.0)))
    candle_range = float(indicators.get("candle_range_override", regime_metrics.get("candle_range", 0.0)))
    if atr_14 > 0 and candle_range > (1.8 * atr_14):
        return {
            "approved": False,
            "score": 0.0,
            "tier": "NO_TRADE",
            "reason": "Rejected: Trigger candle overextended (> 1.8x ATR)",
        }

    # Hard Regime Filter 5b: EMA20 Overextension (Price > 1.3x ATR from EMA20)
    cur_close = float(indicators.get("close_override") or state.get("current_price") or regime_metrics.get("close", 0.0))
    ema_20 = float(indicators.get("ema_20_override") or regime_metrics.get("ema_20", 0.0))
    if atr_14 > 0 and ema_20 > 0 and cur_close > 0 and abs(cur_close - ema_20) > (1.3 * atr_14):
        return {
            "approved": False,
            "score": 0.0,
            "tier": "NO_TRADE",
            "reason": "Rejected: Price > 1.3x ATR from EMA20",
        }

    # Hard Regime Filter 6: Anti-Exhaustion RSI Filter (Prevent Selling the Bottom / Buying the Top)
    rsi_14 = float(indicators.get("rsi_14_override", regime_metrics.get("rsi_14", 50.0)))
    if direction == "SELL" and rsi_14 < 34.0:
        return {
            "approved": False,
            "score": 0.0,
            "tier": "NO_TRADE",
            "reason": f"Rejected: Short setup oversold / bear-trap risk (RSI_14={rsi_14:.1f} < 34)",
        }
    if direction == "BUY" and rsi_14 > 70.0:
        return {
            "approved": False,
            "score": 0.0,
            "tier": "NO_TRADE",
            "reason": f"Rejected: Long setup overbought / bull-trap risk (RSI_14={rsi_14:.1f} > 70)",
        }

    # Hard Regime Filter 7: Fee-to-ATR Viability Gate (TP1 return must be >= 3x Round-Trip Fees)
    tp1_price = float(indicators.get("take_profit_1") or 0.0)
    current_entry = float(state.get("current_price", 0.0))
    min_viable_ratio = 3.0 * ROUND_TRIP_FEE_SLIPPAGE  # 3 * 0.0012 = 0.0036 (0.36%)
    override_tp1_dist = indicators.get("tp1_distance_override")
    if override_tp1_dist is not None:
        if float(override_tp1_dist) < min_viable_ratio:
            return {
                "approved": False,
                "score": 0.0,
                "tier": "NO_TRADE",
                "reason": (
                    f"Rejected: Fee-to-ATR Viability Gate failed "
                    f"(TP1 distance {float(override_tp1_dist):.2%} < 3x round-trip fee {min_viable_ratio:.2%})"
                ),
            }
    elif current_entry > 0 and tp1_price > 0:
        tp1_dist_pct = abs(tp1_price - current_entry) / current_entry
        if tp1_dist_pct < min_viable_ratio:
            return {
                "approved": False,
                "score": 0.0,
                "tier": "NO_TRADE",
                "reason": (
                    f"Rejected: Fee-to-ATR Viability Gate failed "
                    f"(TP1 distance {tp1_dist_pct:.2%} < 3x round-trip fee {min_viable_ratio:.2%})"
                ),
            }

    # Hard ML Filter: require aligned XGBoost confidence >= 65%
    ml_signal = str(indicators.get("ml_signal", "NEUTRAL")).upper()
    ml_conf = float(indicators.get("ml_confidence", 0.0))
    if ml_signal != direction or ml_conf < MIN_ML_CONFIDENCE:
        return {
            "approved": False,
            "score": 0.0,
            "tier": "NO_TRADE",
            "reason": (
                f"ML Gate Reject: ML={ml_signal} ({ml_conf:.2%}) "
                f"vs Tech={direction} (need aligned >= {MIN_ML_CONFIDENCE:.0%})"
            ),
        }

    # Hard Funding Squeeze Filter when live derivatives data is available
    if derivatives.get("available"):
        fr_check = evaluate_funding_risk(direction, float(derivatives.get("funding_rate", 0.0)), EXTREME_FUNDING_RATE)
        if fr_check["blocked"]:
            return {"approved": False, "score": 0.0, "tier": "NO_TRADE", "reason": fr_check["reason"]}

    # ==========================================
    # Weighted Confluence Score (0-100)
    # ==========================================
    # 1. Technical Structure & Pattern Quality (0-35)
    tech_raw = float(indicators.get("score", 70.0))
    patterns = indicators.get("detected_patterns") or []
    structure_pts = min(35.0, 22.0 + max(0.0, min(tech_raw, 100.0) - 60.0) / 40.0 * 9.0 + min(len(patterns), 2) * 2.0)

    # 2. XGBoost ML Confidence (0-35; 0.65 -> 27, 0.80+ -> 35)
    ml_pts = min(35.0, 27.0 + max(0.0, ml_conf - MIN_ML_CONFIDENCE) / 0.15 * 8.0)

    # 3. Volume & Volatility Alignment (0-15)
    vol_pts = min(9.0, 6.0 + max(0.0, vol_ratio - 1.35) * 3.0)
    adx_pts = min(6.0, 3.5 + max(0.0, adx_14 - 22.0) * 0.20)
    vol_vol_pts = round(vol_pts + adx_pts, 2)

    # 4. Fundamental/Sentiment & Derivatives Score (0-15)
    aligned_sent = sentiment if direction == "BUY" else -sentiment
    sent_pts = max(0.0, min(8.0, 4.0 + aligned_sent * 4.0))
    if derivatives.get("available"):
        fr = float(derivatives.get("funding_rate", 0.0))
        fr_aligned = (-fr if direction == "BUY" else fr)
        deriv_pts = max(2.0, min(7.0, 5.0 + fr_aligned * 2000.0))
    else:
        deriv_pts = 5.0  # Neutral baseline when offline
    fund_deriv_pts = round(sent_pts + deriv_pts, 2)

    total_score = round(min(100.0, structure_pts + ml_pts + vol_vol_pts + fund_deriv_pts), 1)
    if total_score < SNIPER_MIN_SCORE:
        return {
            "approved": False,
            "score": total_score,
            "tier": "NO_TRADE",
            "reason": f"Sniper Score {total_score:.1f}/100 < {SNIPER_MIN_SCORE:.0f} threshold",
        }

    tier = "S_TIER" if total_score >= SNIPER_S_TIER_SCORE else "A_GRADE"
    regime_mode = "ROTATIONAL" if (20.0 <= adx_14 < 26.0) else "EXPANSION"
    return {
        "approved": True,
        "score": total_score,
        "tier": tier,
        "regime_mode": regime_mode,
        "breakdown": {
            "structure": round(structure_pts, 1),
            "ml": round(ml_pts, 1),
            "volume_volatility": vol_vol_pts,
            "sentiment_derivatives": fund_deriv_pts,
        },
        "reason": f"{tier} Sniper Setup ({total_score:.1f}/100) [{regime_mode}]",
    }


def _reject(state: TradingState, reason: str, score: float = 0.0) -> TradingState:
    logger.info(f"Trade REJECTED: {reason}")
    indicators = dict(state.get("indicators") or {})
    indicators["sniper_score"] = score
    indicators["trade_tier"] = "NO_TRADE"
    indicators["rejection_reason"] = reason
    state["indicators"] = indicators
    state["final_decision"] = "HOLD"
    state["position_size"] = 0.0
    state["capital_allocation"] = 0.0
    return state


def risk_arbiter_agent(
    state: TradingState,
    memory: Optional[TradeMemory] = None,
) -> TradingState:
    """
    Sniper Confluence Gate + Dynamic Equity & Wick-Proof Position Sizer.
    """
    tech_signal = state.get("technical_signal", "HOLD")
    current_price = float(state.get("current_price", 0.0))
    fallback_sl = float(state.get("stop_loss", 0.0))

    if tech_signal not in ("BUY", "SELL") or current_price <= 0:
        return _reject(state, "Missing valid BUY/SELL signal or price.")

    db = memory if memory is not None else TradeMemory(DEFAULT_DB_PATH, initial_equity=INITIAL_CAPITAL)
    regime_metrics = compute_regime_metrics(state)
    derivatives = (
        fetch_futures_market_context(state.get("symbol", "BTC/USDT"))
        if TRADING_MARKET_TYPE == "FUTURES"
        else {"available": False}
    )

    sniper = evaluate_sniper_confluence(state, regime_metrics, derivatives)
    if not sniper["approved"]:
        return _reject(state, sniper["reason"], score=sniper["score"])

    horizon = str(state.get("trade_horizon") or state.get("indicators", {}).get("trade_horizon", "INTRADAY")).upper()
    h_cfg = HORIZON_CONFIGS.get(horizon, HORIZON_CONFIGS["INTRADAY"])
    horizon_max_lev = h_cfg["max_leverage"]
    horizon_sl_floor = h_cfg["sl_floor_pct"]

    regime_mode = sniper.get("regime_mode", "ROTATIONAL" if (20.0 <= regime_metrics["adx_14"] < 26.0) else "EXPANSION")

    # Dynamic equity & Drawdown Shield
    account_equity = db.get_account_equity()
    shield_active = db.is_drawdown_shield_active(3)
    base_risk_pct = 0.02 if sniper["tier"] == "S_TIER" else 0.01
    risk_pct = base_risk_pct * 0.5 if shield_active else base_risk_pct

    # Wick-Proof SL
    df_entry = (
        state.get("ohlcv_5m") if horizon == "SCALP" and state.get("ohlcv_5m") is not None and not state["ohlcv_5m"].empty
        else (state.get("ohlcv_1h") if horizon == "SWING" and state.get("ohlcv_1h") is not None and not state["ohlcv_1h"].empty else state.get("ohlcv_15m"))
    )
    initial_sl = calculate_wick_proof_sl(
        direction=tech_signal,
        entry=current_price,
        atr_14=regime_metrics["atr_14"],
        df_entry=df_entry,
        fallback_sl=fallback_sl,
        sl_floor_pct=horizon_sl_floor,
    )
    sl_distance = abs(current_price - initial_sl)
    if initial_sl <= 0 or sl_distance <= 0:
        return _reject(state, "Invalid Wick-Proof stop-loss distance.", score=sniper["score"])

    sl_ratio = sl_distance / current_price
    effective_sl_pct = sl_ratio + ROUND_TRIP_FEE_SLIPPAGE
    risk_usd = account_equity * risk_pct
    raw_notional = risk_usd / effective_sl_pct

    if TRADING_MARKET_TYPE == "FUTURES":
        safe_leverage = max(1, min(horizon_max_lev, int(0.40 / max(sl_ratio, 1e-6))))
        max_margin_usd = account_equity * MAX_MARGIN_EQUITY_PCT
        max_notional = max_margin_usd * safe_leverage
        notional_value = min(raw_notional, max_notional)
        margin_required = notional_value / safe_leverage
        position_size = notional_value / current_price
        margin_mode = "ISOLATED"
        est_liq_price = (
            current_price * max(0.0, 1.0 - (1.0 / safe_leverage) + 0.005)
            if tech_signal == "BUY"
            else current_price * (1.0 + (1.0 / safe_leverage) - 0.005)
        )
    else:
        safe_leverage = 1
        notional_value = min(raw_notional, account_equity * MAX_MARGIN_EQUITY_PCT)
        margin_required = notional_value
        position_size = notional_value / current_price
        margin_mode = "SPOT"
        est_liq_price = 0.0

    # 5-stage lifecycle price targets in R
    sign = 1.0 if tech_signal == "BUY" else -1.0
    if regime_mode == "ROTATIONAL":
        tp1_price = round(current_price + sign * 1.1 * sl_distance, 2)
        tp2_price = round(current_price + sign * 1.8 * sl_distance, 2)
        tp3_price = round(current_price + sign * 1.8 * sl_distance, 2)
    else:
        tp1_price = round(current_price + sign * 1.4 * sl_distance, 2)
        tp2_price = round(current_price + sign * 2.4 * sl_distance, 2)
        tp3_price = round(current_price + sign * 3.0 * sl_distance, 2)

    indicators = dict(state.get("indicators") or {})
    indicators.update({
        "trade_horizon": horizon,
        "regime_mode": regime_mode,
        "horizon_max_leverage": horizon_max_lev,
        "horizon_sl_floor_pct": horizon_sl_floor,
        "time_stop_candles": h_cfg["time_stop_candles"],
        "execution_timeframe": h_cfg["timeframe"],
        "market_type": TRADING_MARKET_TYPE,
        "margin_mode": margin_mode,
        "recommended_leverage": safe_leverage,
        "notional_value_usd": round(notional_value, 2),
        "margin_required_usd": round(margin_required, 2),
        "risk_amount_usd": round(risk_usd, 2),
        "risk_pct": round(risk_pct * 100.0, 2),
        "sl_pct": round(sl_ratio * 100.0, 3),
        "effective_sl_pct": round(effective_sl_pct * 100.0, 3),
        "est_liquidation_price": round(est_liq_price, 2),
        "sniper_score": sniper["score"],
        "trade_tier": sniper["tier"],
        "sniper_breakdown": sniper["breakdown"],
        "drawdown_shield_active": shield_active,
        "account_equity": round(account_equity, 2),
        "adx_14": regime_metrics["adx_14"],
        "rsi_14": regime_metrics["rsi_14"],
        "atr_14": regime_metrics["atr_14"],
        "atr_pct": round((regime_metrics["atr_14"] / current_price * 100.0) if current_price > 0 else 0.0, 3),
        "volume_ratio": regime_metrics["volume_ratio"],
        "funding_rate": float(derivatives.get("funding_rate", 0.0)),
        "open_interest": float(derivatives.get("open_interest", 0.0)),
        "take_profit_1": tp1_price,
        "take_profit": tp2_price,
        "take_profit_2": tp2_price,
        "take_profit_3": tp3_price,
    })
    state["indicators"] = indicators
    state["final_decision"] = tech_signal
    state["position_size"] = round(position_size, 8)
    state["capital_allocation"] = round(margin_required, 2)
    state["stop_loss"] = round(initial_sl, 2)
    state["confidence"] = round(sniper["score"] / 100.0, 2)

    logger.info(
        f"{tech_signal} APPROVED [{sniper['tier']} {sniper['score']}/100 | "
        f"Risk {risk_pct*100:.1f}%{' SHIELD' if shield_active else ''}] | "
        f"Entry: ${current_price:,.2f} | Wick-SL: ${initial_sl:,.2f} (-{sl_ratio*100:.2f}%) | "
        f"Lev: {safe_leverage}x | Margin: ${margin_required:,.2f} / Equity: ${account_equity:,.2f}"
    )
    return state
