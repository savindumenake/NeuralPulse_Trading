import os
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

from agents.state import TradingState

# ගණිතමය මොළයෙන් (market_structure) අවශ්‍ය ආයුධ ගෙන්වා ගැනීම
from agents.market_structure import (
    detect_market_regime,
    find_support_resistance_zones,
    find_angled_trendlines,
    find_angled_resistance_trendline,
    find_curved_support,
    detect_basic_patterns,
    detect_candlestick_signal,
    detect_liquidity_sweep,
    analyze_volume_spike,
    calculate_dynamic_risk,
    analyze_momentum_indicators,
    analyze_htf_momentum_alignment,
)


HORIZON_SL_FLOORS = {
    "SCALP": 0.0035,    # 0.35% minimum SL floor for 5m scalps
    "INTRADAY": 0.0050, # 0.50% minimum SL floor for 15m intraday
    "SWING": 0.0080,    # 0.80% minimum SL floor for 1h/4h swing
}


class TechnicalAgent:
    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}
    _xgb_booster = None
    _xgb_load_attempted = False

    def __init__(self):
        # සාමාන්‍ය වෙළඳපොළකදී Trade එකක් ගන්න අවශ්‍ය මූලික ලකුණු සීමාව (Base Threshold)
        self.base_threshold = 70

    def _detect_macro_regime(self, df_macro: pd.DataFrame) -> str:
        """Evaluate macro market regime (1D or 1H for SCALP)."""
        if df_macro is None or df_macro.empty:
            return "INSUFFICIENT_DATA"
        if len(df_macro) >= 200:
            return detect_market_regime(df_macro)
        if len(df_macro) < 20:
            return "INSUFFICIENT_DATA"
        closes = df_macro["close"]
        ema20 = closes.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = closes.ewm(span=50, adjust=False).mean().iloc[-1]
        cur = closes.iloc[-1]
        if ema20 > ema50 and cur >= ema50:
            return "STRONG_BULLISH"
        elif ema20 < ema50 and cur <= ema50:
            return "STRONG_BEARISH"
        return "RANGING"

    @classmethod
    def set_booster(cls, booster) -> None:
        """Explicitly set an in-memory XGBoost booster (e.g. calibrated on 15m candles)."""
        cls._xgb_booster = booster
        cls._xgb_load_attempted = True

    @classmethod
    def _get_xgb_booster(cls):
        """Lazy-load trained XGBoost model (xgb_model.json) once."""
        if not cls._xgb_load_attempted:
            cls._xgb_load_attempted = True
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "xgb_model.json",
            )
            if os.path.exists(model_path):
                try:
                    import xgboost as xgb
                    booster = xgb.Booster()
                    booster.load_model(model_path)
                    cls._xgb_booster = booster
                except Exception:
                    cls._xgb_booster = None
        return cls._xgb_booster

    def _predict_xgboost_signal(self, df_1h: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the trained Phase 2 XGBoost model on 1H stationary features.
        Returns NEUTRAL on short/synthetic frames (< 60 candles).
        """
        default_res = {"signal": "NEUTRAL", "prob_up": 0.5, "confidence": 0.0}
        if df_1h is None or len(df_1h) < 60 or np.std(df_1h["close"].values) == 0:
            return default_res

        booster = self._get_xgb_booster()
        if booster is None:
            return default_res

        try:
            import xgboost as xgb
            from data_loader import calculate_technical_features, get_feature_column_names

            featured = calculate_technical_features(df_1h, is_training=False)
            feature_cols = get_feature_column_names()
            latest_row = featured[feature_cols].iloc[[-1]].replace([np.inf, -np.inf], np.nan)
            if latest_row.isna().any(axis=1).iloc[0]:
                return default_res

            dmat = xgb.DMatrix(latest_row, feature_names=feature_cols)
            prob_up = float(booster.predict(dmat)[0])

            # Free native TreeSHAP contributions (no external `shap` dependency at runtime)
            top_features = []
            try:
                contribs = booster.predict(dmat, pred_contribs=True)[0][:-1]  # last column = bias
                order = np.argsort(np.abs(contribs))[::-1][:3]
                top_features = [
                    {"feature": feature_cols[i], "shap": round(float(contribs[i]), 4)} for i in order
                ]
            except Exception:
                top_features = []

            if prob_up >= 0.55:
                res = {"signal": "BUY", "prob_up": round(prob_up, 4), "confidence": round(prob_up, 4)}
            elif prob_up <= 0.45:
                res = {"signal": "SELL", "prob_up": round(prob_up, 4), "confidence": round(1.0 - prob_up, 4)}
            else:
                res = {"signal": "NEUTRAL", "prob_up": round(prob_up, 4), "confidence": round(max(prob_up, 1.0 - prob_up), 4)}
            res["top_features"] = top_features
            return res
        except Exception:
            return default_res

    def _check_proximity(self, current_price: float, zones: list, tolerance_pct: float = 0.005) -> tuple:
        """Return the nearest support below price only when it is close enough."""
        supports_below_price = [zone for zone in zones if zone <= current_price]
        if not supports_below_price:
            return False, None

        nearest_support = max(supports_below_price)
        is_near = (current_price - nearest_support) / current_price <= tolerance_pct
        return is_near, nearest_support if is_near else None

    def _check_resistance_proximity(self, current_price: float, zones: list, tolerance_pct: float = 0.005) -> tuple:
        """Return the nearest resistance above price only when it is close enough."""
        resistances_above_price = [zone for zone in zones if zone >= current_price]
        if not resistances_above_price:
            return False, None

        nearest_resistance = min(resistances_above_price)
        is_near = (nearest_resistance - current_price) / current_price <= tolerance_pct
        return is_near, nearest_resistance if is_near else None

    def _validate_frame(self, frame: Any, name: str, minimum_rows: int) -> str | None:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return f"{name} timeframe data is missing or empty."
        missing_columns = self.REQUIRED_COLUMNS.difference(frame.columns)
        if missing_columns:
            return f"{name} timeframe is missing columns: {', '.join(sorted(missing_columns))}."
        if len(frame) < minimum_rows:
            return f"{name} timeframe needs at least {minimum_rows} candles; got {len(frame)}."

        try:
            values = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
        except (TypeError, ValueError):
            return f"{name} timeframe OHLCV values must be numeric."
        if not np.isfinite(values).all():
            return f"{name} timeframe contains missing or non-finite OHLCV values."

        opens, highs, lows, closes, volumes = values.T
        if (
            np.any(opens <= 0)
            or np.any(highs < np.maximum(opens, closes))
            or np.any(lows > np.minimum(opens, closes))
            or np.any(highs < lows)
            or np.any(volumes < 0)
        ):
            return f"{name} timeframe contains invalid OHLCV candles."
        return None

    def analyze_horizon(
        self,
        df_macro: pd.DataFrame,
        df_structure: pd.DataFrame,
        df_trigger: pd.DataFrame,
        horizon: str = "INTRADAY",
        df_ml: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Multi-Horizon Top-Down Confluence Analysis:
        - SCALP:    df_macro=1H, df_structure=15m, df_trigger=5m (SL Floor: 0.35%)
        - INTRADAY: df_macro=1D, df_structure=1H,  df_trigger=15m (SL Floor: 0.50%)
        - SWING:    df_macro=1D, df_structure=4H,  df_trigger=1H (SL Floor: 0.80%)
        """
        horizon = str(horizon).upper()
        if horizon not in HORIZON_SL_FLOORS:
            horizon = "INTRADAY"

        macro_name = "1H" if horizon == "SCALP" else "1D"
        struct_name = "15m" if horizon == "SCALP" else ("4H" if horizon == "SWING" else "1H")
        trigger_name = "5m" if horizon == "SCALP" else ("1H" if horizon == "SWING" else "15m")

        min_macro_rows = 20 if horizon == "SCALP" else 200
        min_struct_rows = 21 if horizon in ("INTRADAY", "SWING") else 20
        frame_requirements = (
            (df_macro, macro_name, min_macro_rows),
            (df_structure, struct_name, min_struct_rows),
            (df_trigger, trigger_name, 20),
        )
        for frame, name, minimum_rows in frame_requirements:
            validation_error = self._validate_frame(frame, name, minimum_rows)
            if validation_error:
                return {
                    "signal": "HOLD",
                    "score": 0,
                    "reason": validation_error,
                    "trade_horizon": horizon,
                }

        current_price = float(df_trigger["close"].iloc[-1])
        regime = self._detect_macro_regime(df_macro)

        current_threshold = self.base_threshold
        direction = "BUY"
        if regime == "STRONG_BULLISH":
            current_threshold = 60
        elif regime == "STRONG_BEARISH":
            direction = "SELL"
            current_threshold = 60
        elif regime == "RANGING":
            current_threshold = 80
        elif regime == "INSUFFICIENT_DATA":
            return {
                "signal": "HOLD",
                "score": 0,
                "required_threshold": current_threshold,
                "market_regime": regime,
                "entry_price": current_price,
                "reason": f"Market is {regime}. No trade is allowed.",
                "trade_horizon": horizon,
            }

        # Key Levels & Patterns on Structure Frame
        zones_struct = find_support_resistance_zones(df_structure)
        trendline_struct = find_angled_trendlines(df_structure)
        chart_pattern_struct = detect_basic_patterns(df_structure)

        bullish_breakout_patterns = {
            "DOUBLE_BOTTOM_BREAKOUT",
            "BULLISH_BREAKOUT",
            "ASCENDING_TRIANGLE_BREAKOUT",
            "SYMMETRICAL_TRIANGLE_BREAKOUT",
        }
        bearish_breakout_patterns = {
            "DOUBLE_TOP_BREAKDOWN",
            "BEARISH_BREAKDOWN",
            "DESCENDING_TRIANGLE_BREAKDOWN",
            "SYMMETRICAL_TRIANGLE_BREAKDOWN",
        }
        bullish_patterns = bullish_breakout_patterns | {
            "DOUBLE_BOTTOM_FORMING",
            "BULLISH_HH_HL_STRUCTURE",
        }
        bearish_patterns = bearish_breakout_patterns | {
            "DOUBLE_TOP_FORMING",
            "BEARISH_LH_LL_STRUCTURE",
        }

        needs_both_zone_types = regime in {"RANGING", "TRANSITION"}
        missing_required_zones = (
            not zones_struct["support"]
            or not zones_struct["resistance"] if needs_both_zone_types
            else not zones_struct["support"] if direction == "BUY"
            else not zones_struct["resistance"]
        )
        if missing_required_zones:
            return {
                "signal": "HOLD",
                "score": 0,
                "required_threshold": current_threshold,
                "market_regime": regime,
                "entry_price": current_price,
                "reason": f"Required confirmed {struct_name} {direction.lower()} zones were not found.",
                "trade_horizon": horizon,
            }

        # Action Trigger on Trigger Frame
        is_near_support, nearest_support = self._check_proximity(current_price, zones_struct["support"])
        is_near_resistance, nearest_resistance = self._check_resistance_proximity(
            current_price, zones_struct["resistance"]
        )

        sweep_signal = detect_liquidity_sweep(df_trigger)
        if not is_near_support and sweep_signal == "BULLISH_LIQUIDITY_SWEEP" and zones_struct["support"]:
            sweep_candidates = [
                float(z) for z in zones_struct["support"]
                if abs(current_price - float(z)) / current_price <= 0.005
            ]
            if sweep_candidates:
                nearest_support = min(sweep_candidates, key=lambda z: abs(current_price - z))
                is_near_support = True

        if not is_near_resistance and sweep_signal == "BEARISH_LIQUIDITY_SWEEP" and zones_struct["resistance"]:
            sweep_candidates = [
                float(z) for z in zones_struct["resistance"]
                if abs(current_price - float(z)) / current_price <= 0.005
            ]
            if sweep_candidates:
                nearest_resistance = min(sweep_candidates, key=lambda z: abs(current_price - z))
                is_near_resistance = True

        if not is_near_support and chart_pattern_struct in bullish_patterns:
            is_near_support, nearest_support = self._check_proximity(
                current_price, zones_struct["support"], tolerance_pct=0.015
            )
            if not is_near_support and chart_pattern_struct in bullish_breakout_patterns and zones_struct["support"]:
                levels_below = [
                    float(z) for z in (zones_struct["support"] + zones_struct.get("resistance", []))
                    if z <= current_price
                ]
                nearest_support = max(levels_below) if levels_below else float(min(zones_struct["support"]))
                is_near_support = True

        if not is_near_resistance and chart_pattern_struct in bearish_patterns:
            is_near_resistance, nearest_resistance = self._check_resistance_proximity(
                current_price, zones_struct["resistance"], tolerance_pct=0.015
            )
            if not is_near_resistance and chart_pattern_struct in bearish_breakout_patterns and zones_struct["resistance"]:
                levels_above = [
                    float(z) for z in (zones_struct["resistance"] + zones_struct.get("support", []))
                    if z >= current_price
                ]
                nearest_resistance = min(levels_above) if levels_above else float(max(zones_struct["resistance"]))
                is_near_resistance = True

        if regime in {"RANGING", "TRANSITION"}:
            if is_near_support and not is_near_resistance:
                direction = "BUY"
            elif is_near_resistance and not is_near_support:
                direction = "SELL"
            elif is_near_support and is_near_resistance:
                if chart_pattern_struct in bearish_patterns:
                    direction = "SELL"
                elif chart_pattern_struct in bullish_patterns:
                    direction = "BUY"
                else:
                    return {
                        "signal": "HOLD", "score": 0, "required_threshold": current_threshold,
                        "market_regime": regime, "entry_price": current_price,
                        "reason": f"Price is not near a single, clear {struct_name} support or resistance zone.",
                        "trade_horizon": horizon,
                    }
            else:
                return {
                    "signal": "HOLD", "score": 0, "required_threshold": current_threshold,
                    "market_regime": regime, "entry_price": current_price,
                    "reason": f"Price is not near a single, clear {struct_name} support or resistance zone.",
                    "trade_horizon": horizon,
                }

        is_near_trade_level = is_near_support if direction == "BUY" else is_near_resistance
        nearest_zone = nearest_support if direction == "BUY" else nearest_resistance
        if not is_near_trade_level:
            return {
                "signal": "HOLD", "score": 0, "required_threshold": current_threshold,
                "market_regime": regime, "entry_price": current_price,
                "reason": f"Price is not near confirmed {struct_name} {direction.lower()} trade levels.",
                "trade_horizon": horizon,
            }

        score = 0
        reasons = []
        detected_patterns = []

        # A. Volume Confirmation (on trigger frame)
        vol_info = analyze_volume_spike(df_trigger)
        if vol_info["is_spike"]:
            score += 30
            reasons.append(f"Volume Spike Confirmed ({vol_info['ratio']}x Average)")

        # B. Micro-Price Action (on trigger frame)
        candle_signal = detect_candlestick_signal(df_trigger)
        expected_candle_direction = "BULLISH" if direction == "BUY" else "BEARISH"
        opposing_candle_direction = "BEARISH" if direction == "BUY" else "BULLISH"
        if expected_candle_direction in candle_signal:
            score += 30
            detected_patterns.append(candle_signal)
            reasons.append(f"{expected_candle_direction.title()} price action: {candle_signal}")
        elif opposing_candle_direction in candle_signal:
            score -= 15
            reasons.append(f"Warning: opposing {trigger_name} candle ({candle_signal})")

        # B2. Institutional Liquidity Sweep
        expected_sweep = "BULLISH_LIQUIDITY_SWEEP" if direction == "BUY" else "BEARISH_LIQUIDITY_SWEEP"
        if sweep_signal == expected_sweep:
            score += 20
            detected_patterns.append(sweep_signal)
            reasons.append(f"Liquidity Sweep Stop-Hunt Reversal ({sweep_signal})")

        # C. Trendline Confluence (on structure frame)
        trade_trendline = (
            trendline_struct if direction == "BUY"
            else find_angled_resistance_trendline(df_structure)
        )
        trendline_price_key = "support_price" if direction == "BUY" else "resistance_price"
        trendline_near_price = False
        if trade_trendline:
            trendline_price = trade_trendline[trendline_price_key]
            trendline_near_price = (
                trendline_price <= current_price
                if direction == "BUY"
                else trendline_price >= current_price
            ) and abs(current_price - trendline_price) / current_price <= 0.005
        if (
            trade_trendline
            and ((direction == "BUY" and trade_trendline["slope"] > 0)
                 or (direction == "SELL" and trade_trendline["slope"] < 0))
            and trendline_near_price
        ):
            score += 20
            trendline_type = "support" if direction == "BUY" else "resistance"
            reasons.append(
                f"Near directional {struct_name} {trendline_type} trendline "
                f"(Strength: {trade_trendline['strength_r2'] * 100:.1f}%)"
            )
        elif direction == "BUY" and find_curved_support(df_structure) is not None:
            score += 10
            reasons.append(f"Curved parabolic {struct_name} support structure active")

        # D. Chart Pattern & Breakout Confluence (on structure frame)
        if direction == "BUY" and chart_pattern_struct in bullish_patterns:
            score += 20
            detected_patterns.append(chart_pattern_struct)
            reasons.append(f"{struct_name} Bullish Structure/Pattern: {chart_pattern_struct}")
        elif direction == "SELL" and chart_pattern_struct in bearish_patterns:
            score += 20
            detected_patterns.append(chart_pattern_struct)
            reasons.append(f"{struct_name} Bearish Structure/Pattern: {chart_pattern_struct}")
        elif direction == "BUY" and chart_pattern_struct in bearish_breakout_patterns:
            score -= 15
            reasons.append(f"Warning: counter-trend {struct_name} breakdown ({chart_pattern_struct})")
        elif direction == "SELL" and chart_pattern_struct in bullish_breakout_patterns:
            score -= 15
            reasons.append(f"Warning: counter-trend {struct_name} breakout ({chart_pattern_struct})")

        # D2. Higher-Timeframe Momentum Alignment
        htf_momentum = analyze_htf_momentum_alignment(df_structure)
        expected_htf_bias = "BULLISH" if direction == "BUY" else "BEARISH"
        if htf_momentum["bias"] == expected_htf_bias:
            score += 10
            reasons.append(f"{struct_name} Momentum Aligned ({htf_momentum['bias']}, RSI {htf_momentum['rsi_1h']:.1f})")

        # E. Momentum Indicators (on trigger frame)
        momentum = analyze_momentum_indicators(df_trigger)
        if direction == "BUY" and momentum["bullish_score"] > 0:
            score += momentum["bullish_score"]
            reasons.extend(momentum["bullish_reasons"])
        elif direction == "SELL" and momentum["bearish_score"] > 0:
            score += momentum["bearish_score"]
            reasons.extend(momentum["bearish_reasons"])

        # F. XGBoost ML Model Confluence
        target_ml_df = df_ml if df_ml is not None and not df_ml.empty else df_structure
        xgb_info = self._predict_xgboost_signal(target_ml_df)
        if xgb_info["signal"] == direction:
            score += 15
            reasons.append(
                f"XGBoost ML {direction} Confirmation (Confidence: {xgb_info['confidence'] * 100:.1f}%)"
            )

        score = max(0, min(score, 100))

        # STEP 5: Risk Management (ATR + Horizon-Adaptive SL Floor)
        risk_info = calculate_dynamic_risk(df_trigger)
        stop_loss_distance = risk_info["recommended_sl_distance"]
        if not np.isfinite(stop_loss_distance) or stop_loss_distance <= 0:
            return {
                "signal": "HOLD", "score": score, "required_threshold": current_threshold,
                "market_regime": regime, "entry_price": current_price,
                "reason": "ATR stop-loss distance is unavailable or invalid.",
                "trade_horizon": horizon,
            }

        horizon_floor = HORIZON_SL_FLOORS.get(horizon, 0.0050)
        min_sl_pct = float(os.getenv("MIN_SL_PCT", str(horizon_floor)))
        min_sl_pct = max(min_sl_pct, horizon_floor)
        if current_price > 0 and min_sl_pct > 0:
            stop_loss_distance = max(stop_loss_distance, round(current_price * min_sl_pct, 2))

        # STEP 6: Final Decision Maker
        if score >= current_threshold:
            stop_loss = (
                current_price - stop_loss_distance
                if direction == "BUY"
                else current_price + stop_loss_distance
            )
            take_profit_1 = (
                current_price + (stop_loss_distance * 1.5)
                if direction == "BUY"
                else current_price - (stop_loss_distance * 1.5)
            )
            take_profit = (
                current_price + (stop_loss_distance * 2.0)
                if direction == "BUY"
                else current_price - (stop_loss_distance * 2.0)
            )
            take_profit_3 = (
                current_price + (stop_loss_distance * 3.0)
                if direction == "BUY"
                else current_price - (stop_loss_distance * 3.0)
            )
            if stop_loss <= 0 or take_profit <= 0 or take_profit_3 <= 0:
                return {
                    "signal": "HOLD", "score": score, "required_threshold": current_threshold,
                    "market_regime": regime, "entry_price": current_price,
                    "reason": "Calculated stop-loss or take-profit price is invalid.",
                    "trade_horizon": horizon,
                }
            sl_pct = round((stop_loss_distance / current_price) * 100.0, 3) if current_price > 0 else 0.0
            return {
                "signal": direction,
                "score": score,
                "required_threshold": current_threshold,
                "market_regime": regime,
                "entry_price": current_price,
                "nearest_support_zone" if direction == "BUY" else "nearest_resistance_zone": nearest_zone,
                "recommended_stop_loss": round(stop_loss, 2),
                "recommended_take_profit_1": round(take_profit_1, 2),
                "recommended_take_profit": round(take_profit, 2),
                "recommended_take_profit_3": round(take_profit_3, 2),
                "sl_pct": sl_pct,
                "risk_reward_ratio": "1:2.0",
                "atr_value": risk_info["atr_value"],
                "ml_signal": xgb_info["signal"],
                "ml_confidence": xgb_info["confidence"],
                "top_shap_features": xgb_info.get("top_features", []),
                "chart_pattern_1h": chart_pattern_struct,
                "candle_signal": candle_signal,
                "sweep_signal": sweep_signal,
                "detected_patterns": detected_patterns,
                "reasons_for_trade": reasons,
                "trade_horizon": horizon,
                "horizon_sl_floor": horizon_floor,
                "execution_timeframe": trigger_name,
            }
        else:
            return {
                "signal": "HOLD",
                "score": score,
                "required_threshold": current_threshold,
                "market_regime": regime,
                "entry_price": current_price,
                "ml_signal": xgb_info["signal"],
                "ml_confidence": xgb_info["confidence"],
                "detected_patterns": detected_patterns,
                "trade_horizon": horizon,
                "reason": f"Failed Confluence Check (Score {score}/{current_threshold}). Waiting for more confirmations.",
            }

    def analyze_multi_timeframe(self, df_1d: pd.DataFrame, df_1h: pd.DataFrame, df_15m: pd.DataFrame) -> Dict[str, Any]:
        """Backward-compatible entry point for Intraday 1D -> 1H -> 15m analysis."""
        return self.analyze_horizon(df_macro=df_1d, df_structure=df_1h, df_trigger=df_15m, horizon="INTRADAY")


def technical_analyst_agent(state: TradingState) -> TradingState:
    """Adapt the multi-horizon multi-timeframe analysis to the shared agent state."""
    agent = TechnicalAgent()

    ml_frame = state.get("ohlcv_ml")
    if ml_frame is None or ml_frame.empty:
        ml_frame = state.get("ohlcv_1h")

    # 1. If state explicitly requests a trade_horizon
    requested_horizon = state.get("trade_horizon")
    if requested_horizon:
        h = requested_horizon.upper()
        if h == "SCALP":
            decision = agent.analyze_horizon(
                df_macro=state.get("ohlcv_1h"),
                df_structure=state.get("ohlcv_15m"),
                df_trigger=state.get("ohlcv_5m"),
                horizon="SCALP",
                df_ml=ml_frame,
            )
        elif h == "SWING":
            decision = agent.analyze_horizon(
                df_macro=state.get("ohlcv_1d"),
                df_structure=state.get("ohlcv_4h"),
                df_trigger=state.get("ohlcv_1h"),
                horizon="SWING",
                df_ml=ml_frame,
            )
        else:
            decision = agent.analyze_horizon(
                df_macro=state.get("ohlcv_1d"),
                df_structure=state.get("ohlcv_1h"),
                df_trigger=state.get("ohlcv_15m"),
                horizon="INTRADAY",
                df_ml=ml_frame,
            )
    else:
        # 2. Multi-Horizon Scan: Check available horizons and pick the highest conviction valid trade
        candidates = []

        # Check INTRADAY (1D -> 1H -> 15m)
        if state.get("ohlcv_1d") is not None and state.get("ohlcv_1h") is not None and state.get("ohlcv_15m") is not None:
            dec_intra = agent.analyze_horizon(
                df_macro=state.get("ohlcv_1d"),
                df_structure=state.get("ohlcv_1h"),
                df_trigger=state.get("ohlcv_15m"),
                horizon="INTRADAY",
                df_ml=ml_frame,
            )
            candidates.append(dec_intra)

        # Check SCALP (1H -> 15m -> 5m)
        if (
            state.get("ohlcv_1h") is not None
            and state.get("ohlcv_15m") is not None
            and state.get("ohlcv_5m") is not None
            and not state.get("ohlcv_5m").empty
        ):
            dec_scalp = agent.analyze_horizon(
                df_macro=state.get("ohlcv_1h"),
                df_structure=state.get("ohlcv_15m"),
                df_trigger=state.get("ohlcv_5m"),
                horizon="SCALP",
                df_ml=ml_frame,
            )
            candidates.append(dec_scalp)

        # Check SWING (1D -> 4H -> 1H)
        if (
            state.get("ohlcv_1d") is not None
            and state.get("ohlcv_4h") is not None
            and state.get("ohlcv_1h") is not None
            and not state.get("ohlcv_4h").empty
        ):
            dec_swing = agent.analyze_horizon(
                df_macro=state.get("ohlcv_1d"),
                df_structure=state.get("ohlcv_4h"),
                df_trigger=state.get("ohlcv_1h"),
                horizon="SWING",
                df_ml=ml_frame,
            )
            candidates.append(dec_swing)

        if not candidates:
            decision = agent.analyze_multi_timeframe(
                state.get("ohlcv_1d"),
                state.get("ohlcv_1h"),
                state.get("ohlcv_15m"),
            )
        else:
            valid_signals = [c for c in candidates if c.get("signal") in ("BUY", "SELL")]
            if valid_signals:
                decision = max(valid_signals, key=lambda c: float(c.get("score", 0)))
            else:
                decision = candidates[0]

    score = float(decision.get("score", 0))
    threshold = float(decision.get("required_threshold", 0))
    state["technical_signal"] = decision["signal"]
    state["confidence"] = round(score / 100.0, 2)
    existing_indicators = dict(state.get("indicators") or {})
    state["indicators"] = {
        **existing_indicators,
        "score": score,
        "required_threshold": threshold,
        "take_profit_1": float(decision.get("recommended_take_profit_1", 0.0)),
        "take_profit": float(decision.get("recommended_take_profit", 0.0)),
        "take_profit_2": float(decision.get("recommended_take_profit", 0.0)),
        "take_profit_3": float(decision.get("recommended_take_profit_3", 0.0)),
        "sl_pct": float(decision.get("sl_pct", 0.0)),
        "risk_reward_ratio": decision.get("risk_reward_ratio", "1:2.0"),
        "atr_value": float(decision.get("atr_value", 0.0)),
        "ml_signal": decision.get("ml_signal", "NEUTRAL"),
        "ml_confidence": float(decision.get("ml_confidence", 0.0)),
        "top_shap_features": decision.get("top_shap_features", []),
        "chart_pattern_1h": decision.get("chart_pattern_1h", "NO_CLEAR_PATTERN"),
        "candle_signal": decision.get("candle_signal", "NEUTRAL"),
        "sweep_signal": decision.get("sweep_signal", "NEUTRAL"),
        "market_regime": decision.get("market_regime", "UNKNOWN"),
        "detected_patterns": decision.get("detected_patterns", []),
        "trade_horizon": decision.get("trade_horizon", "INTRADAY"),
        "horizon_sl_floor": float(decision.get("horizon_sl_floor", 0.0050)),
        "execution_timeframe": decision.get("execution_timeframe", "15m"),
    }
    state["trade_horizon"] = decision.get("trade_horizon", "INTRADAY")
    state["technical_reason"] = decision.get("reason") or "; ".join(decision.get("reasons_for_trade", []))
    state["current_price"] = float(decision.get("entry_price", state.get("current_price", 0.0)))
    state["stop_loss"] = float(decision.get("recommended_stop_loss", 0.0))
    if decision["signal"] == "HOLD":
        state["final_decision"] = "HOLD"
        state["position_size"] = 0.0
        state["capital_allocation"] = 0.0
        state["execution_status"] = "SKIPPED"
    return state


# ==========================================
# Testing the Agent (මෙය පරීක්ෂා කිරීම සඳහා)
# ==========================================
if __name__ == "__main__":
    import json

    print("[Testing] Top-Down Multi-Timeframe Technical Agent...\n")
    np.random.seed(42)

    # 1. 1D Data (Strong Bullish Trend)
    df_1d = pd.DataFrame({"close": np.linspace(50000, 95000, 200)})
    df_1d["high"] = df_1d["close"] + 500
    df_1d["low"] = df_1d["close"] - 500
    df_1d["open"] = df_1d["close"] - 200
    df_1d["volume"] = 1000.0

    # 2. 1H Data (With valid 'open' column and clear swing waves around 91,000)
    t = np.linspace(0, 8 * np.pi, 200)
    close_1h = 91500 + 500 * np.sin(t) + np.linspace(-200, 200, 200)
    df_1h = pd.DataFrame({
        "open": close_1h - 50,
        "high": close_1h + 150,
        "low": close_1h - 150,
        "close": close_1h,
        "volume": np.random.uniform(400, 600, 200),
    })

    # 3. 15m Data (Entry Trigger near 1H support zone with Bullish Pin Bar + Volume Spike)
    df_15m = pd.DataFrame({
        "open": np.linspace(90900, 91100, 20),
        "high": np.linspace(91000, 91250, 20),
        "low": np.linspace(90800, 90950, 20),
        "close": np.linspace(90950, 91180, 20),
        "volume": np.random.uniform(20, 40, 20),
    })
    # Latest 15m candle: Strong Bullish Pin Bar + Volume Spike at support
    df_15m.at[19, "open"] = 91120.0
    df_15m.at[19, "close"] = 91180.0
    df_15m.at[19, "high"] = 91200.0
    df_15m.at[19, "low"] = 90750.0
    df_15m.at[19, "volume"] = 180.0

    agent = TechnicalAgent()
    decision = agent.analyze_multi_timeframe(df_1d, df_1h, df_15m)
    print(json.dumps(decision, indent=4))
