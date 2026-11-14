import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.signal import argrelextrema
from scipy.stats import linregress
from sklearn.cluster import KMeans
import warnings

# අනවශ්‍ය Warnings සැඟවීම සඳහා
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# Data Pipeline Fail-Safes (ආරක්ෂිත දැල්)
# ==========================================

def interpolate_data_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """ API Timeouts නිසා ඇතිවන හිඩැස් Forward-Fill මඟින් පුරවයි. """
    return df.ffill().bfill()

def remove_outliers_iqr(data: np.ndarray, multiplier: float = 2.5) -> np.ndarray:
    """ Scam Wicks සහ Flash Crashes ඉවත් කිරීම (Strong Trends ආරක්ෂා වන පරිදි multiplier=2.5). """
    if len(data) < 4:
        return data
    q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q3 - q1
    if iqr == 0:
        return data
    lower_bound, upper_bound = q1 - (multiplier * iqr), q3 + (multiplier * iqr)
    filtered = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered if len(filtered) > 0 else data

def filter_confirmed_extrema(extrema_indices: np.ndarray, total_length: int, order: int = 10) -> np.ndarray:
    """ Repainting වළක්වයි: අනාගතයට දත්ත ඇති ලක්ෂ්‍ය පමණක් තහවුරු කරයි. """
    if len(extrema_indices) == 0:
        return extrema_indices
    return np.array([idx for idx in extrema_indices if idx <= (total_length - 1 - order)])

# ==========================================
# 1. Market Regime Detection (දිශාව සහ ප්‍රබලත්වය)
# ==========================================
def detect_market_regime(df_daily: pd.DataFrame) -> str:
    df_daily = interpolate_data_gaps(df_daily.copy())
    if len(df_daily) < 200:
        return "INSUFFICIENT_DATA"

    df_daily['sma_200'] = df_daily['close'].rolling(window=200).mean().bfill()
    df_daily['ema_50'] = df_daily['close'].ewm(span=50, adjust=False).mean()

    adx = df_daily.ta.adx(length=14)
    df_daily['adx'] = adx['ADX_14'].fillna(0.0) if adx is not None and 'ADX_14' in adx else 0.0

    latest_close = df_daily['close'].iloc[-1]
    latest_sma200 = df_daily['sma_200'].iloc[-1]
    latest_ema50 = df_daily['ema_50'].iloc[-1]
    lookback_idx = -4 if len(df_daily) >= 4 else -2
    prev_ema50 = df_daily['ema_50'].iloc[lookback_idx]
    latest_adx = df_daily['adx'].iloc[-1]

    if latest_adx < 20.0:
        return "RANGING"
    elif latest_close > latest_sma200 and latest_ema50 > prev_ema50 and latest_adx >= 25.0:
        return "STRONG_BULLISH"
    elif latest_close < latest_sma200 and latest_ema50 < prev_ema50 and latest_adx >= 25.0:
        return "STRONG_BEARISH"
    else:
        return "TRANSITION"

# ==========================================
# 2. Support & Resistance Zones (තිරස්, කෝණික, වක්‍ර)
# ==========================================
def _extract_confirmed_swings(df_1h: pd.DataFrame, primary_order: int = 10) -> tuple:
    """Confirmed swing highs සහ swing lows ලබා ගනී (අවශ්‍ය නම් order=5 fallback සහිතව)."""
    total_len = len(df_1h)
    highs = df_1h['high'].values
    lows = df_1h['low'].values

    raw_max = argrelextrema(highs, np.greater_equal, order=primary_order)[0]
    raw_min = argrelextrema(lows, np.less_equal, order=primary_order)[0]

    local_max = filter_confirmed_extrema(raw_max, total_len, primary_order)
    local_min = filter_confirmed_extrema(raw_min, total_len, primary_order)

    # Flat series එකකදී සියලුම ලක්ෂ්‍ය extrema වීම වැළැක්වීම
    if np.std(highs) == 0:
        local_max = np.array([], dtype=int)
    if np.std(lows) == 0:
        local_min = np.array([], dtype=int)

    if (len(local_max) == 0 or len(local_min) == 0) and total_len >= 15 and primary_order > 3:
        fallback_order = max(3, primary_order // 2)
        if len(local_max) == 0 and np.std(highs) > 0:
            raw_max = argrelextrema(highs, np.greater, order=fallback_order)[0]
            local_max = filter_confirmed_extrema(raw_max, total_len, fallback_order)
        if len(local_min) == 0 and np.std(lows) > 0:
            raw_min = argrelextrema(lows, np.less, order=fallback_order)[0]
            local_min = filter_confirmed_extrema(raw_min, total_len, fallback_order)

    return local_max, local_min

def find_horizontal_zones(df_1h: pd.DataFrame, num_clusters: int = 5) -> list:
    local_max, local_min = _extract_confirmed_swings(df_1h, primary_order=10)
    if len(local_max) == 0 or len(local_min) == 0:
        return []

    high_prices = df_1h['high'].iloc[local_max].values
    low_prices = df_1h['low'].iloc[local_min].values

    all_extremes = np.concatenate((high_prices, low_prices))
    all_extremes = remove_outliers_iqr(all_extremes)

    if len(all_extremes) < num_clusters or np.std(all_extremes) == 0:
        return []

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(all_extremes.reshape(-1, 1))
    return [round(float(zone), 2) for zone in np.sort(kmeans.cluster_centers_.flatten())]

def _cluster_price_levels(prices: np.ndarray, num_clusters: int) -> list:
    prices = remove_outliers_iqr(np.asarray(prices, dtype=float))
    if len(prices) == 0:
        return []

    cluster_count = min(num_clusters, len(np.unique(prices)))
    if cluster_count == 1:
        return [round(float(np.mean(prices)), 2)]

    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    centers = kmeans.fit(prices.reshape(-1, 1)).cluster_centers_.flatten()
    return [round(float(level), 2) for level in np.sort(centers)]

def find_support_resistance_zones(df_1h: pd.DataFrame, num_clusters: int = 5) -> dict:
    """Cluster confirmed swing lows as support and swing highs as resistance."""
    local_max, local_min = _extract_confirmed_swings(df_1h, primary_order=10)

    resistance = _cluster_price_levels(df_1h['high'].iloc[local_max].values, num_clusters) if len(local_max) > 0 else []
    support = _cluster_price_levels(df_1h['low'].iloc[local_min].values, num_clusters) if len(local_min) > 0 else []
    return {"support": support, "resistance": resistance}

def _fit_angled_trendline(df_1h: pd.DataFrame, is_support: bool) -> dict | None:
    """Fit a validated directional linear trendline from confirmed swing extrema."""
    if df_1h is None or len(df_1h) < 15 or np.std(df_1h['close'].values) == 0:
        return None

    local_max, local_min = _extract_confirmed_swings(df_1h, primary_order=10)
    extrema = local_min if is_support else local_max
    if len(extrema) < 3:
        return None

    recent = extrema[-4:] if len(extrema) >= 4 else extrema[-3:]
    col = 'low' if is_support else 'high'
    slope, intercept, r_value, _, _ = linregress(recent, df_1h[col].iloc[recent].values)
    r2 = float(r_value ** 2)
    min_r2 = 0.85 if len(recent) >= 4 else 0.90
    if r2 <= min_r2:
        return None
    if is_support and slope <= 0:
        return None
    if not is_support and slope >= 0:
        return None

    projected_price = float(slope * (len(df_1h) - 1) + intercept)
    if not np.isfinite(projected_price) or projected_price <= 0:
        return None

    price_key = "support_price" if is_support else "resistance_price"
    return {
        "slope": round(float(slope), 4),
        "strength_r2": round(r2, 4),
        price_key: round(projected_price, 2),
    }


def find_angled_trendlines(df_1h: pd.DataFrame):
    return _fit_angled_trendline(df_1h, is_support=True)


def find_angled_resistance_trendline(df_1h: pd.DataFrame):
    return _fit_angled_trendline(df_1h, is_support=False)

def find_curved_support(df_1h: pd.DataFrame):
    raw_min = argrelextrema(df_1h['low'].values, np.less, order=10)[0]
    local_min = filter_confirmed_extrema(raw_min, len(df_1h), 10)

    if len(local_min) >= 5:
        recent_min = local_min[-5:]
        coeffs = np.polyfit(range(len(recent_min)), df_1h['low'].iloc[recent_min].values, 2)
        if abs(coeffs[0]) > 1000:
            return None
        return [round(float(c), 4) for c in coeffs]
    return None

# ==========================================
# 3. Chart Patterns & Market Structure (Double Top/Bottom, Triangles, HH/HL, Breakouts)
# ==========================================
def detect_basic_patterns(df_1h: pd.DataFrame, tolerance: float = 0.015) -> str:
    """
    Chart Patterns සහ Market Structure හඳුනා ගනී (Fresh Breakout Verification සහිතව):
    - DOUBLE_TOP_BREAKDOWN / DOUBLE_BOTTOM_BREAKOUT
    - DOUBLE_TOP_FORMING / DOUBLE_BOTTOM_FORMING
    - ASCENDING_TRIANGLE_BREAKOUT / DESCENDING_TRIANGLE_BREAKDOWN
    - SYMMETRICAL_TRIANGLE_BREAKOUT / SYMMETRICAL_TRIANGLE_BREAKDOWN
    - BULLISH_BREAKOUT / BEARISH_BREAKDOWN
    - BULLISH_HH_HL_STRUCTURE / BEARISH_LH_LL_STRUCTURE
    """
    if len(df_1h) < 20 or np.std(df_1h['close'].values) == 0:
        return "NO_CLEAR_PATTERN"

    local_max, local_min = _extract_confirmed_swings(df_1h, primary_order=10)
    latest_close = float(df_1h['close'].iloc[-1])
    latest_open = float(df_1h['open'].iloc[-1])
    recent_closes = df_1h['close'].iloc[-4:-1].values if len(df_1h) >= 4 else df_1h['close'].iloc[:-1].values

    latest_high = float(df_1h['high'].iloc[-1])
    latest_low = float(df_1h['low'].iloc[-1])
    prev_close = float(df_1h['close'].iloc[-2]) if len(df_1h) >= 2 else latest_close
    latest_body = abs(latest_close - latest_open)

    # 1. Double Top Check (Neckline strictly between Peak 1 & Peak 2 + Fresh Breakdown/Retest check)
    if len(local_max) >= 2:
        idx1, idx2 = int(local_max[-2]), int(local_max[-1])
        p1, p2 = float(df_1h['high'].iloc[idx1]), float(df_1h['high'].iloc[idx2])
        if idx2 > idx1 + 2 and p1 > 0 and abs(p1 - p2) / p1 <= tolerance:
            neckline = float(df_1h['low'].iloc[idx1:idx2 + 1].min())
            was_recently_above = len(recent_closes) > 0 and bool(np.max(recent_closes) >= neckline * 0.998)
            near_neckline = neckline > 0 and abs(latest_close - neckline) / neckline <= 0.02
            if latest_close < neckline and was_recently_above and near_neckline:
                return "DOUBLE_TOP_BREAKDOWN"
            # Forming only when price shows fresh rejection near Peak 2 (within last 8 bars)
            upper_wick = latest_high - max(latest_open, latest_close)
            has_bearish_rejection = (upper_wick >= latest_body * 0.5) or (latest_close < prev_close)
            if (
                (len(df_1h) - 1 - idx2) <= 8
                and latest_close >= neckline
                and abs(latest_close - p2) / p2 <= tolerance
                and latest_close < latest_open
                and has_bearish_rejection
            ):
                return "DOUBLE_TOP_FORMING"

    # 2. Double Bottom Check (Neckline strictly between Trough 1 & Trough 2 + Fresh Breakout/Retest check)
    if len(local_min) >= 2:
        idx1, idx2 = int(local_min[-2]), int(local_min[-1])
        t1, t2 = float(df_1h['low'].iloc[idx1]), float(df_1h['low'].iloc[idx2])
        if idx2 > idx1 + 2 and t1 > 0 and abs(t1 - t2) / t1 <= tolerance:
            neckline = float(df_1h['high'].iloc[idx1:idx2 + 1].max())
            was_recently_below = len(recent_closes) > 0 and bool(np.min(recent_closes) <= neckline * 1.002)
            near_neckline = neckline > 0 and abs(latest_close - neckline) / neckline <= 0.02
            if latest_close > neckline and was_recently_below and near_neckline:
                return "DOUBLE_BOTTOM_BREAKOUT"
            lower_wick = min(latest_open, latest_close) - latest_low
            has_bullish_rejection = (lower_wick >= latest_body * 0.5) or (latest_close > prev_close)
            if (
                (len(df_1h) - 1 - idx2) <= 8
                and latest_close <= neckline
                and abs(latest_close - t2) / t2 <= tolerance
                and latest_close > latest_open
                and has_bullish_rejection
            ):
                return "DOUBLE_BOTTOM_FORMING"

    # 3. Triangle Breakouts, Swing Breakouts & HH/HL or LH/LL Swing Structure
    if len(local_max) >= 2 and len(local_min) >= 2:
        prev_high, last_high = float(df_1h['high'].iloc[local_max[-2]]), float(df_1h['high'].iloc[local_max[-1]])
        prev_low, last_low = float(df_1h['low'].iloc[local_min[-2]]), float(df_1h['low'].iloc[local_min[-1]])

        fresh_break_up = (
            latest_close > last_high * 1.0005
            and len(recent_closes) > 0
            and bool(np.min(recent_closes) <= last_high * 1.002)
            and abs(latest_close - last_high) / last_high <= 0.025
        )
        fresh_break_down = (
            latest_close < last_low * 0.9995
            and len(recent_closes) > 0
            and bool(np.max(recent_closes) >= last_low * 0.998)
            and abs(latest_close - last_low) / last_low <= 0.025
        )

        flat_highs = prev_high > 0 and abs(last_high - prev_high) / prev_high <= tolerance
        flat_lows = prev_low > 0 and abs(last_low - prev_low) / prev_low <= tolerance
        rising_lows = last_low > prev_low * 1.002
        falling_highs = last_high < prev_high * 0.998

        # Ascending / Descending / Symmetrical Triangle Breakouts
        if fresh_break_up and flat_highs and rising_lows:
            return "ASCENDING_TRIANGLE_BREAKOUT"
        if fresh_break_down and flat_lows and falling_highs:
            return "DESCENDING_TRIANGLE_BREAKDOWN"
        if fresh_break_up and falling_highs and rising_lows:
            return "SYMMETRICAL_TRIANGLE_BREAKOUT"
        if fresh_break_down and falling_highs and rising_lows:
            return "SYMMETRICAL_TRIANGLE_BREAKDOWN"

        # Horizontal / Swing Structure Breakouts
        if fresh_break_up and last_low >= prev_low * 0.995:
            return "BULLISH_BREAKOUT"
        if fresh_break_down and last_high <= prev_high * 1.005:
            return "BEARISH_BREAKDOWN"

        # Active Trend Structure (HH/HL or LH/LL)
        if last_high > prev_high and last_low > prev_low and latest_close > last_low:
            return "BULLISH_HH_HL_STRUCTURE"
        if last_high < prev_high and last_low < prev_low and latest_close < last_high:
            return "BEARISH_LH_LL_STRUCTURE"

    return "NO_CLEAR_PATTERN"

# ==========================================
# 4. Micro-Confirmations (Candlestick Patterns)
# ==========================================
def detect_candlestick_signal(df_1h: pd.DataFrame) -> str:
    """
    24/7 Crypto Market එකට ගැලපෙන ලෙස නිවැරදි කළ Candlestick Pattern Detector:
    - BULLISH_PIN_BAR / BEARISH_PIN_BAR
    - BULLISH_HAMMER / BEARISH_SHOOTING_STAR
    - BULLISH_ENGULFING / BEARISH_ENGULFING (Crypto continuous open/close fix සහිතව)
    - BULLISH_MORNING_STAR / BEARISH_EVENING_STAR
    - BULLISH_MOMENTUM_CANDLE / BEARISH_MOMENTUM_CANDLE
    """
    if len(df_1h) < 2:
        return "NEUTRAL"

    current, prev = df_1h.iloc[-1], df_1h.iloc[-2]
    c_open, c_high, c_low, c_close = float(current['open']), float(current['high']), float(current['low']), float(current['close'])
    p_open, p_high, p_low, p_close = float(prev['open']), float(prev['high']), float(prev['low']), float(prev['close'])

    body = abs(c_open - c_close)
    prev_body = abs(p_open - p_close)
    total_length = c_high - c_low

    if total_length <= 0:
        return "NEUTRAL"

    lower_wick = min(c_open, c_close) - c_low
    upper_wick = c_high - max(c_open, c_close)

    # 1. Pin Bars (Strong rejection wicks >= 60% of candle range)
    if lower_wick > (total_length * 0.6) and body < (total_length * 0.3):
        return "BULLISH_PIN_BAR"
    if upper_wick > (total_length * 0.6) and body < (total_length * 0.3):
        return "BEARISH_PIN_BAR"

    # 2. Crypto-Aware Engulfing (Handles current_open == prev_close in 24/7 crypto markets)
    is_curr_bullish = c_close > c_open
    is_curr_bearish = c_close < c_open
    is_prev_bullish = p_close > p_open
    is_prev_bearish = p_close < p_open

    if (
        is_curr_bullish
        and is_prev_bearish
        and prev_body > 0
        and c_close >= p_open
        and c_open <= p_close * 1.001
        and body >= prev_body * 1.05
    ):
        return "BULLISH_ENGULFING"

    if (
        is_curr_bearish
        and is_prev_bullish
        and prev_body > 0
        and c_close <= p_open
        and c_open >= p_close * 0.999
        and body >= prev_body * 1.05
    ):
        return "BEARISH_ENGULFING"

    # 3. Hammer & Shooting Star (Wick >= 50% of range, directional close)
    if lower_wick >= (total_length * 0.5) and upper_wick <= (total_length * 0.15) and is_curr_bullish:
        return "BULLISH_HAMMER"
    if upper_wick >= (total_length * 0.5) and lower_wick <= (total_length * 0.15) and is_curr_bearish:
        return "BEARISH_SHOOTING_STAR"

    # 4. 3-Candle Morning Star & Evening Star
    if len(df_1h) >= 3:
        first = df_1h.iloc[-3]
        f_open, f_close = float(first['open']), float(first['close'])
        f_body = abs(f_open - f_close)
        f_midpoint = (f_open + f_close) / 2.0

        # Morning Star: Big red -> small indecision body -> strong green closing above first midpoint
        if f_close < f_open and f_body > 0 and prev_body < (f_body * 0.4) and is_curr_bullish and c_close > f_midpoint:
            return "BULLISH_MORNING_STAR"
        # Evening Star: Big green -> small indecision body -> strong red closing below first midpoint
        if f_close > f_open and f_body > 0 and prev_body < (f_body * 0.4) and is_curr_bearish and c_close < f_midpoint:
            return "BEARISH_EVENING_STAR"

    # 5. Strong Momentum Candle (Marubozu - body >= 75% of range & larger than recent average range)
    avg_range = float((df_1h['high'] - df_1h['low']).tail(10).mean())
    if avg_range > 0 and total_length >= avg_range * 1.3 and body >= total_length * 0.75:
        return "BULLISH_MOMENTUM_CANDLE" if is_curr_bullish else "BEARISH_MOMENTUM_CANDLE"

    return "NEUTRAL"

# ==========================================
# 5. Volume Spread Analysis (VSA)
# ==========================================
def analyze_volume_spike(df_1h: pd.DataFrame, multiplier: float = 1.5) -> dict:
    df_temp = df_1h.copy()
    df_temp['vol_sma_20'] = df_temp['volume'].rolling(window=20).mean().bfill()

    current_vol, avg_vol = float(df_temp['volume'].iloc[-1]), float(df_temp['vol_sma_20'].iloc[-1])
    is_spike = bool(current_vol > (avg_vol * multiplier)) if avg_vol > 0 else False
    return {"is_spike": is_spike, "ratio": round(current_vol / avg_vol, 2) if avg_vol > 0 else 0.0}

def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 24, max_sweep_pct: float = 0.015) -> str:
    """
    Detect institutional Swing Failure / Liquidity Sweep reversals on closed candles:
    - BULLISH_LIQUIDITY_SWEEP: Wick sweeps below prior swing low and closes back above it.
    - BEARISH_LIQUIDITY_SWEEP: Wick sweeps above prior swing high and closes back below it.
    """
    if df is None or len(df) < lookback + 2 or np.std(df["close"].values) == 0:
        return "NEUTRAL"

    prev_window = df.iloc[-(lookback + 1):-1]
    prior_low = float(prev_window["low"].min())
    prior_high = float(prev_window["high"].max())
    if prior_low <= 0 or prior_high <= 0:
        return "NEUTRAL"

    curr = df.iloc[-1]
    c_open = float(curr["open"])
    c_high = float(curr["high"])
    c_low = float(curr["low"])
    c_close = float(curr["close"])
    body = abs(c_open - c_close)
    lower_wick = min(c_open, c_close) - c_low
    upper_wick = c_high - max(c_open, c_close)

    # Bullish stop-hunt below prior low followed by strong reclaim above prior low
    if (
        c_low < prior_low
        and c_close > prior_low
        and (prior_low - c_low) / prior_low <= max_sweep_pct
        and (c_close >= c_open or lower_wick > body * 1.2)
    ):
        return "BULLISH_LIQUIDITY_SWEEP"

    # Bearish stop-hunt above prior high followed by strong rejection below prior high
    if (
        c_high > prior_high
        and c_close < prior_high
        and (c_high - prior_high) / prior_high <= max_sweep_pct
        and (c_close <= c_open or upper_wick > body * 1.2)
    ):
        return "BEARISH_LIQUIDITY_SWEEP"

    return "NEUTRAL"


# ==========================================
# 6. Dynamic Volatility & ATR (Risk Management)
# ==========================================
def calculate_dynamic_risk(df_1h: pd.DataFrame, atr_multiplier: float = 1.5, min_sl_pct: float = 0.0) -> dict:
    atr = df_1h.ta.atr(length=14)
    current_atr = float(atr.iloc[-1]) if atr is not None and not atr.empty else 0.0
    sl_distance = current_atr * atr_multiplier
    if np.isfinite(current_atr) and current_atr > 0 and min_sl_pct > 0 and "close" in df_1h and len(df_1h) > 0:
        latest_close = float(df_1h["close"].iloc[-1])
        if np.isfinite(latest_close) and latest_close > 0:
            sl_distance = max(sl_distance, latest_close * min_sl_pct)
    return {"atr_value": round(current_atr, 2), "recommended_sl_distance": round(sl_distance, 2)}


def analyze_htf_momentum_alignment(df_1h: pd.DataFrame) -> dict:
    """Check 1H EMA 20/50 trend & RSI bias so lower-timeframe setups align with higher-timeframe flow."""
    result = {"bias": "NEUTRAL", "rsi_1h": 50.0, "ema_20": 0.0, "ema_50": 0.0}
    if df_1h is None or len(df_1h) < 30 or np.std(df_1h["close"].values) == 0:
        return result

    closes = df_1h["close"]
    latest_close = float(closes.iloc[-1])
    ema20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
    rsi_series = df_1h.ta.rsi(close=closes, length=14)
    rsi_1h = float(rsi_series.dropna().iloc[-1]) if rsi_series is not None and not rsi_series.dropna().empty else 50.0

    result["rsi_1h"] = round(rsi_1h, 2)
    result["ema_20"] = round(ema20, 2)
    result["ema_50"] = round(ema50, 2)

    if latest_close > ema20 and (ema20 >= ema50 or 48.0 <= rsi_1h <= 72.0):
        result["bias"] = "BULLISH"
    elif latest_close < ema20 and (ema20 <= ema50 or 28.0 <= rsi_1h <= 52.0):
        result["bias"] = "BEARISH"
    return result

# ==========================================
# 7. Momentum & Indicator Confluence (RSI, MACD, EMA 20/50)
# ==========================================
def analyze_momentum_indicators(df: pd.DataFrame) -> dict:
    """
    RSI (14), MACD (12, 26, 9), සහ EMA (20/50) පරීක්ෂා කර
    BUY හෝ SELL සඳහා අමතර තාක්ෂණික තහවුරු කිරීම් (Confluence Points) ලබා දෙයි.
    """
    result = {
        "rsi": 50.0,
        "macd_hist": 0.0,
        "ema_20": 0.0,
        "ema_50": 0.0,
        "bullish_score": 0,
        "bearish_score": 0,
        "bullish_reasons": [],
        "bearish_reasons": [],
    }
    if df is None or len(df) < 15 or np.std(df["close"].values) == 0:
        return result

    closes = df["close"]
    latest_close = float(closes.iloc[-1])

    # 1. RSI (14)
    rsi_series = df.ta.rsi(close=closes, length=14)
    if rsi_series is not None and not rsi_series.dropna().empty:
        latest_rsi = float(rsi_series.dropna().iloc[-1])
        result["rsi"] = round(latest_rsi, 2)
        if latest_rsi <= 38.0:
            result["bullish_score"] += 10
            result["bullish_reasons"].append(f"RSI Oversold Bounce Zone ({latest_rsi:.1f})")
        elif 50.0 < latest_rsi < 68.0:
            result["bullish_score"] += 10
            result["bullish_reasons"].append(f"RSI Bullish Momentum ({latest_rsi:.1f})")

        if latest_rsi >= 62.0:
            result["bearish_score"] += 10
            result["bearish_reasons"].append(f"RSI Overbought Rejection Zone ({latest_rsi:.1f})")
        elif 32.0 < latest_rsi < 50.0:
            result["bearish_score"] += 10
            result["bearish_reasons"].append(f"RSI Bearish Momentum ({latest_rsi:.1f})")

    # 2. EMA 20 & EMA 50 Trend Alignment
    ema20 = closes.ewm(span=20, adjust=False).mean()
    ema50 = closes.ewm(span=50, adjust=False).mean()
    latest_ema20 = float(ema20.iloc[-1])
    latest_ema50 = float(ema50.iloc[-1])
    result["ema_20"] = round(latest_ema20, 2)
    result["ema_50"] = round(latest_ema50, 2)

    if latest_close > latest_ema20 and latest_ema20 >= latest_ema50:
        result["bullish_score"] += 10
        result["bullish_reasons"].append("EMA 20 > EMA 50 Bullish Alignment")
    elif latest_close < latest_ema20 and latest_ema20 <= latest_ema50:
        result["bearish_score"] += 10
        result["bearish_reasons"].append("EMA 20 < EMA 50 Bearish Alignment")

    # 3. MACD (12, 26, 9) when enough rows exist
    if len(df) >= 35:
        macd = df.ta.macd(close=closes, fast=12, slow=26, signal=9)
        if macd is not None and "MACDh_12_26_9" in macd and not macd["MACDh_12_26_9"].dropna().empty:
            hist = macd["MACDh_12_26_9"].dropna()
            latest_hist = float(hist.iloc[-1])
            prev_hist = float(hist.iloc[-2]) if len(hist) >= 2 else 0.0
            result["macd_hist"] = round(latest_hist, 4)
            if latest_hist > 0 or (latest_hist > prev_hist and prev_hist < 0):
                result["bullish_score"] = min(20, result["bullish_score"] + 10)
                result["bullish_reasons"].append("MACD Bullish Momentum")
            if latest_hist < 0 or (latest_hist < prev_hist and prev_hist > 0):
                result["bearish_score"] = min(20, result["bearish_score"] + 10)
                result["bearish_reasons"].append("MACD Bearish Momentum")

    return result

# ==========================================
# කේතය පරීක්ෂා කිරීම (Testing)
# ==========================================
if __name__ == "__main__":
    import os
    print("[Testing] Ultimate Market Structure Core (Fail-Safes Active)...")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, "data", "btc_usdt_1h_raw.csv")

    try:
        df = pd.read_csv(data_path)

        print(f"\n📊 1. Market Regime: {detect_market_regime(df)}")
        print(f"🧱 2. Main Support/Resistance Zones: {find_horizontal_zones(df[-500:])}")
        print(f"📐 3. Chart Pattern Detected: {detect_basic_patterns(df[-100:])}")
        print(f"🕯️ 4. Latest Candle Action: {detect_candlestick_signal(df)}")

        vol_info = analyze_volume_spike(df)
        print(f"📈 5. Volume Spike Detected: {vol_info['is_spike']} (x{vol_info['ratio']} of Average)")

        risk_info = calculate_dynamic_risk(df)
        print(f"🛡️ 6. Dynamic Risk: ATR=${risk_info['atr_value']} | Safe SL Distance=${risk_info['recommended_sl_distance']}")

    except FileNotFoundError:
        print("\n[!] Data file not found. Ensure Phase 1 data is in the 'data' folder.")
