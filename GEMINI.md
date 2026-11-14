# NeuralPulse Trading — Pro-Trader Automated Signal & Quant Pipeline

## 1. Project Overview & Core Goal
This project automates the workflow of a disciplined, institutional-grade Bitcoin (`BTC/USDT`) futures trader using a **Multi-Agent State-Machine Architecture** (LangGraph).

Operating in **High-Selectivity "Sniper Mode"** (aiming for 2 to 4 high-conviction trades per week), the system **silently monitors live multi-timeframe market structure across 3 Execution Horizons (`SCALP`, `INTRADAY`, `SWING`) via Binance WebSockets (`5m`, `15m`, `1h`, `4h`, `1d`)** and **only triggers an alert & Telegram channel signal** when:
1. **High-Conviction Technical Confluence** aligns (Market Regime + Key S/R Zone / S/R Flip / Sweep + Chart Breakout + Candlestick Trigger + Liquidity Sweep + Volume Spike + Multi-Timeframe Momentum + Conflict Trap Filter + XGBoost ML), **AND**
2. **Fundamental / News Sentiment & Derivatives** support the trade direction, **AND**
3. **7 Hard Regime, Anti-Chasing, Anti-Exhaustion, and Fee-Viability Filters** all pass.

**Target Market:** `BTC/USDT` Perpetual Futures (`BTCUSDT`)  
**Execution Horizons:**
- ⚡ **SCALP:** `1H` (Macro Regime) → `15m` (Structure & S/R) → `5m` (Trigger & Entry)
- 🏹 **INTRADAY:** `1D` (Macro Regime) → `1H` (Structure & S/R) → `15m` (Trigger & Entry)
- 🌊 **SWING:** `1D` (Macro Regime) → `4H` (Structure & S/R) → `1H` (Trigger & Entry)

**Output Mode:** Live Trader Alert + **Binance Futures Telegram VIP Signal Card** + SQLite Trade Memory + Paper Trade CSV Ledger.

---

## 2. Core Technology Stack
- **Live Data & Streaming:** `ccxt` (Binance REST bootstrap & derivatives funding), `websockets` (Real-time Binance Kline Streams across `5m`, `15m`, `1h`, `4h`, `1d`), `pandas`, `pandas_ta`.
- **Machine Learning:** `xgboost`, `scikit-learn`, `shap` (Stationary 1H tabular directional probability with Shapley feature attributions).
- **Fundamental & News Sentiment:** CryptoPanic API + **NLTK VADER NLP** (`SentimentIntensityAnalyzer`) + FinBERT pluggable interface + RSS fallback (`CoinTelegraph` / `CoinDesk`).
- **Orchestration:** `langgraph` (`StateGraph` with **Conditional Routing**).
- **Trade Lifecycle & Execution:** 5-Stage Automated State Machine (`TradePosition`) + Dynamic Equity Tracking + Horizon-Adaptive Time Stops.
- **Trade Memory & Quant Diagnostics:** Zero-cost local SQLite engine (`TradeMemory` in `data/trade_memory.db` / `data/backtest_memory.db`) + `analyze_performance.py`.
- **Signal Delivery:** Telegram Bot API (`telegram_notifier.py` with VIP Signal Cards, automated lifecycle stage broadcasts, and cumulative performance metrics).

---

## 3. Multi-Agent Architecture (Conditional LangGraph Flow)
The pipeline strictly uses `langgraph` (`StateGraph`) with a **conditional gate** after `tech_agent` so sentiment APIs, risk models, and alerts are only invoked when a valid technical trade setup exists:

```
START ──► tech_agent ──┬── (HOLD / No Setup) ──────────────────────────► END (Silent Watch)
                       │
                       └── (BUY or SELL Setup!) ──► sentiment_agent ──► risk_arbiter ──► execution_agent ──► END
```

### 1. State Definition (`TradingState` in `agents/state.py`):
Shared TypedDict holding:
- Market context: `symbol`, `current_price`, `ohlcv_5m`, `ohlcv_15m`, `ohlcv_1h`, `ohlcv_4h`, `ohlcv_1d`, `trade_horizon` (`SCALP`, `INTRADAY`, `SWING`).
- Signals: `technical_signal`, `confidence`, `technical_reason`, `sentiment_score`, `sentiment_summary`, `final_decision`.
- Sizing & Risk indicators: `score`, `required_threshold`, `sniper_score`, `tier`, `drawdown_shield_active`, `take_profit_1`, `take_profit_2`, `take_profit_3`, `risk_reward_ratio`, `sl_pct`, `market_type`, `margin_mode`, `recommended_leverage`, `notional_value_usd`, `margin_required_usd`, `est_liquidation_price`, `detected_patterns`, `market_regime`, `execution_timeframe`, `horizon_sl_floor_pct`.
- Execution: `position_size`, `stop_loss`, `capital_allocation`, `execution_status`, `telegram_sent`.

### 2. Node 1: Technical Analyst Agent (`tech_agent` in `agents/technical_agent.py` & `agents/market_structure.py`)
- **3-Tier Multi-Horizon Scanner:**
  - Automatically evaluates all three horizons (`SCALP`, `INTRADAY`, `SWING`) when multi-timeframe candle data is available, selecting the trade setup with the highest conviction confluence score.
  - Supports explicit horizon targeting via `trade_horizon` in state.
- **Macro Regime Evaluation:**
  - Evaluates macro trend (`STRONG_BULLISH`, `STRONG_BEARISH`, `RANGING`, `TRANSITION`) on `1D` (200 bars for Intraday/Swing) or `1H` (20+ bars fast regime for Scalp).
- **Structure & Trigger Confluence:**
  - Maps structure timeframe KMeans S/R zones, directional angled/curved trendlines, HTF momentum (EMA 20/50 + RSI), and chart patterns (`DOUBLE_BOTTOM_BREAKOUT`, `DOUBLE_TOP_BREAKDOWN`, `ASCENDING_TRIANGLE_BREAKOUT`, `DESCENDING_TRIANGLE_BREAKDOWN`, `SYMMETRICAL_TRIANGLE_BREAKOUT`, `BULLISH_BREAKOUT`, `BULLISH_HH_HL_STRUCTURE`, `BEARISH_LH_LL_STRUCTURE`).
  - Evaluates trigger timeframe entries (Pin Bar, Crypto Engulfing, Hammer, Shooting Star, Morning/Evening Star, Liquidity Sweep Reversal `BULLISH_LIQUIDITY_SWEEP` / `BEARISH_LIQUIDITY_SWEEP`, Volume Spike, RSI/MACD/EMA alignment, counter-signal trap penalty) + XGBoost ML probability + Horizon-adaptive SL protection floor (`0.35%` Scalp, `0.50%` Intraday, `0.80%` Swing).
- Outputs `BUY`, `SELL`, or `HOLD`. If `HOLD`, the graph exits silently.

### 3. Node 2: Fundamental & Sentiment Agent (`sentiment_agent` in `agents/sentiment_agent.py`)
- Triggered **only** when `tech_agent` finds a `BUY` or `SELL` setup.
- Combines CryptoPanic community votes + pluggable NLP (`SENTIMENT_ENGINE=finbert|vader`) + RSS headline fallback to output `sentiment_score` (`-1.0` to `1.0`) and `sentiment_summary`.

### 4. Node 3: Sniper Risk Arbiter & Position Sizer (`risk_arbiter` in `agents/risk_arbiter.py`, `agents/futures_context.py`)
- **Hard Regime Filters (Auto-Reject to `HOLD` if any fail):**
  1. `ADX Sweet-Spot (20.0 <= ADX_14 <= 35.0)`: Eliminates both choppy consolidation chop (<20) and late-trend blow-off exhaustion (>35).
  2. `Breakout Volume >= 1.35x SMA20`: Rejects low-volume fakeouts.
  3. `1H` & `4H` EMA Trend Alignment: Both higher-timeframe trends must not conflict and must align with the trade direction.
  4. `Mandatory Structural Pattern Gate`: Rejects any trade if price action pattern is missing, empty, or `NONE`/`NEUTRAL`.
  5. `Anti-Chasing / Overextended Candle Filter`: Rejects setups where trigger candle range `> 1.8 * ATR_14`.
  6. `EMA20 Overextension Filter`: Rejects setups where `abs(close - ema_20) > 1.3 * ATR_14` (prevents entering far away from the mean).
  7. `Anti-Exhaustion RSI Filter`: Rejects `SHORT` if `RSI_14 < 34.0` (bear-trap / oversold bottom) and `LONG` if `RSI_14 > 70.0` (bull-trap / overbought top).
  8. `Fee-to-ATR Viability Gate`: Rejects setups where expected TP1 distance `< 3x round-trip fee + slippage` (`tp1_pct < 3 * 0.0012 = 0.0036` or `0.36%`), preventing commission erosion especially on tight 5m scalp candles.
  - `Hard ML Gate`: Requires aligned XGBoost directional prediction with confidence `>= 65%`.
  - `Funding Rate Squeeze Filter`: Blocks LONGs during crowded positive funding spikes and SHORTs during crowded negative funding spikes.
- **Weighted Confluence Scoring Engine (0–100):**
  - Technical Structure & Pattern Quality: `0–35 pts`
  - XGBoost ML Directional Confidence: `0–35 pts`
  - Volume & Volatility Alignment: `0–15 pts`
  - Fundamental Sentiment & Derivatives Score: `0–15 pts`
- **Execution Gate & Drawdown Shield:**
  - Score `< 78`: `NO_TRADE` (Skip).
  - Score `78–87`: `A_GRADE` trade -> Risk `1.0%` of account equity (`0.5%` under Drawdown Shield).
  - Score `88–100`: `S_TIER` sniper trade -> Risk `2.0%` of account equity (`1.0%` under Drawdown Shield).
  - **Drawdown Shield:** Automatically cuts risk in half if the last 3 closed trades were losses, until a winning trade resets the shield.
- **Horizon-Adaptive Risk, Leverage & Wick-Proof Stop-Loss:**
  - **Horizon Parameters:**
    | Horizon | Execution TF | SL Floor | Max Safe Leverage | Time Stop |
    | :--- | :--- | :--- | :--- | :--- |
    | **SCALP** | `5m` | `0.35%` (`0.0035`) | Up to `15x` | 20 candles (100 min) |
    | **INTRADAY** | `15m` | `0.50%` (`0.0050`) | Up to `10x` | 16 candles (4 hours) |
    | **SWING** | `1H` | `0.80%` (`0.0080`) | Up to `5x` | 12 candles (12 hours) |
  - **Wick-Proof SL:**
    - `LONG`: `min(recent_swing_low, entry - 1.5 * ATR_14) - (0.2 * ATR_14)` (clamped by horizon `sl_floor_pct`).
    - `SHORT`: `max(recent_swing_high, entry + 1.5 * ATR_14) + (0.2 * ATR_14)` (clamped by horizon `sl_floor_pct`).
  - **Dynamic Equity & Futures Sizing:**
    - Incorporates round-trip fees and slippage (`+0.12%`).
    - Caps maximum margin used at `40%` of account equity.
    - Calculates safe isolated leverage ensuring liquidation price is at least `2.5x` the stop-loss distance away (clamped to the horizon max leverage).

### 5. Node 4: Execution, Two-Sigma Regime Lifecycle & Trade Memory (`execution_agent.py`, `trade_lifecycle.py`, `trade_memory.py`)
- **Two-Sigma Regime-Adaptive Lifecycle State Machine:**
  - **Mode A: `ROTATIONAL` (`20 <= ADX < 26` — Quick Harvest, No Runner Give-back):**
    - `STATE 1` at **`+1.1R`**: Close **50%** of position. Move SL of remaining 50% to **`+0.15R`** (fee-covered breakeven).
    - `STATE 2` at **`+1.8R`**: Close remaining **50%** (100% full exit at target, no runner).
    - `TIME_STOP`: 16 candles on 15m without hitting `+0.5R`.
  - **Mode B: `EXPANSION` (`26 <= ADX <= 35` — Fat-Tail Trend Runner):**
    - `STATE 1` at **`+1.4R`**: Close **35%**. Move SL of remaining 65% to **`-0.20R`**.
    - `Intermediate Ratchet` at **`+1.7R`**: Move SL of remaining 65% to **`+0.40R`**.
    - `STATE 2` at **`+2.4R`**: Close **30%** (leaving 35% runner). Move SL to **`+1.00R`** and trail remaining 35% with `2.2 * ATR_14` Chandelier Exit (`STATE 3`).
    - `TIME_STOP`: 16 candles on 15m without hitting `+0.6R`.
- **Trade Memory SQLite Database (`data/trade_memory.db` / `data/backtest_memory.db`):**
  - `trades_context`: Records entry conditions, `trade_horizon` (`SCALP`/`INTRADAY`/`SWING`), session (Asia/London/NY), ADX, RSI, volume ratio, XGBoost confidence, top SHAP features, and initial risk parameters.
  - `trades_lifecycle`: Records exit time, exit state, MFE (Maximum Favorable Excursion), MAE (Maximum Adverse Excursion), holding duration, net PnL, fees, and updated equity.
  - `post_exit_autopsy`: Automated 4h and 12h post-exit checks for `INITIAL_SL` losses to detect stop-hunts (`was_stop_hunt = 1` if price subsequently hit `+1.5R`).
- **Performance Diagnostics (`analyze_performance.py`):**
  - Generates comprehensive quantitative diagnostic reports querying SQLite by pattern, market session, ADX regime, trade horizon, MFE/MAE distributions, and stop-hunt frequencies.

---

## 4. Live Multi-Stream & Backtest Engine
- **Live WebSocket Multi-Stream (`live_feed.py` & `main.py`):**
  - Connects to Binance combined stream (`wss://stream.binance.com:9443/stream?streams=btcusdt@kline_5m/btcusdt@kline_15m/btcusdt@kline_1h/btcusdt@kline_4h/btcusdt@kline_1d`).
  - Ingests closed candles across all 5 timeframes in real time into thread-safe rolling buffers.
  - Triggers pipeline evaluation on candle close events, matching the active horizons.
- **VIP Signal Delivery (`telegram_notifier.py`):**
  - Broadcasts institutional Binance Futures VIP Signal Cards with Horizon Badges:
    - ⚡ `SCALP (5m Trigger | 1H/15m MTF)`
    - 🏹 `INTRADAY (15m Trigger | 1D/1H MTF)`
    - 🌊 `SWING (1H Trigger | 1D/4H MTF)`
  - Automates real-time broadcast of all lifecycle events (`RETEST_SHIELD`, `PROFIT_LOCK`, `CHANDELIER_TRAIL_STOP`, `TIME_STOP`).
- **Backtesting & Simulation (`backtest.py`):**
  - Replays candle-by-candle on true closed candles (`data/btc_usdt_15m_raw.csv`).
  - Zero-lookahead dynamic resampling into higher-timeframe structures.

---

## 5. Strict Engineering Guardrails
- **Avoid Look-Ahead Bias:** Closed candles only; extrema confirmed with `filter_confirmed_extrema`; future returns shifted `shift(-1)` only during offline model training.
- **Stationarity:** Only percentage returns, normalized oscillators, and relative distances feed `xgb_model.json`.
- **Zero-Noise Live Monitoring:** Never spam Telegram or logs on `HOLD` candles. Silent monitoring by default; alerts fire strictly on confirmed entries and automated lifecycle stage transitions.
- **Capital Preservation First:** Drawdown Shield, fee-adjusted sizing, Fee-to-ATR Viability Gate, wick-proof buffers, and tight isolated margin liquidation protections are strictly non-negotiable.
