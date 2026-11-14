# NeuralPulse Trading 📈

An institutional-grade, multi-agent quantitative trading & **Binance Futures VIP Signal** pipeline for cryptocurrency markets (`BTC/USDT`). Operating in **High-Selectivity Sniper Mode**, this project automates multi-horizon market structure monitoring across 3 execution horizons (**SCALP**, **INTRADAY**, **SWING**) via live Binance WebSockets (`5m`, `15m`, `1h`, `4h`, `1d`), Machine Learning (XGBoost + SHAP), NLP news sentiment (CryptoPanic + NLTK VADER / FinBERT), and conditional AI Agent orchestration (LangGraph) with automated 5-stage trade lifecycle management and zero-cost SQLite quant trade memory.

## 🏗️ Architecture

The system uses a conditional **LangGraph `StateGraph`** pipeline so fundamental news APIs, risk engines, and Telegram broadcasts only fire when a high-conviction technical trade setup appears:

```
START ──► tech_agent ──┬── (HOLD / No Setup) ──────────────────────────► END (Silent Watch)
                       │
                       └── (BUY or SELL Setup!) ──► sentiment_agent ──► risk_arbiter ──► execution_agent ──► END
```

### 🎯 3-Tier Multi-Horizon Execution

The pipeline continuously evaluates 3 execution horizons, picking the highest-conviction setup across all timeframes:
- ⚡ **SCALP:** `1H` (Macro Regime) → `15m` (Structure & S/R) → `5m` (Trigger & Entry) | SL Floor: `0.35%` | Max Leverage: `15x` | Time Stop: 20 candles (100 min)
- 🏹 **INTRADAY:** `1D` (Macro Regime) → `1H` (Structure & S/R) → `15m` (Trigger & Entry) | SL Floor: `0.50%` | Max Leverage: `10x` | Time Stop: 16 candles (4 hours)
- 🌊 **SWING:** `1D` (Macro Regime) → `4H` (Structure & S/R) → `1H` (Trigger & Entry) | SL Floor: `0.80%` | Max Leverage: `5x` | Time Stop: 12 candles (12 hours)

### 🛡️ 7 Hard Regime & Viability Filters
Automated auto-rejection to `HOLD` if any filter fails:
1. `ADX_14 >= 22`: Filters out choppy, sideways chop.
2. `Breakout Volume >= 1.5x SMA20`: Rejects low-volume fakeouts.
3. `1H` & `4H` EMA Trend Alignment: Ensures higher-timeframe trends support the trade direction.
4. `Mandatory Structural Pattern Gate`: Rejects any trade lacking confirmed chart patterns.
5. `Anti-Chasing / Overextended Candle Filter`: Blocks entries where candle range `> 1.8 * ATR_14`.
6. `Anti-Exhaustion RSI Filter`: Rejects shorts if `RSI_14 < 34` and longs if `RSI_14 > 70`.
7. `Fee-to-ATR Viability Gate`: Rejects setups where expected TP1 distance `< 3x round-trip fee + slippage` (`0.36%`), protecting against fee attrition especially on 5m scalps.
- `Hard ML Gate`: Aligned XGBoost prediction with `>= 65%` confidence.
- `Funding Rate Squeeze Filter`: Blocks crowded funding spike entries.

### 🔄 5-Stage Automated Trade Lifecycle
- **STATE 0 (`INITIAL`):** 100% size open, SL at `-1.0R`.
- **STATE 1 (`RETEST_SHIELD` at `+1.3R`):** Close **30%** at `+1.3R`. Move SL to **`-0.5R`** (mathematically risk-free net position: `+0.39R - 0.35R = +0.04R`).
- **STATE 2 (`PROFIT_LOCK` at `+2.2R`):** Close **30%** at `+2.2R`. Move SL to **`+0.20R`** (fee-adjusted breakeven+).
- **STATE 3 (`CHANDELIER_RUNNER`):** Trail the remaining **40%** via ratchet Chandelier Exit (`2.2 * ATR_14`).
- **STATE 4 (`TIME_STOP`):** Horizon-adaptive candle threshold (`20` on 5m, `16` on 15m, `12` on 1H) without reaching `+0.6R` triggers market exit to recycle capital.

---

## 💻 Tech Stack

* **Language:** Python 3.12
* **Streaming & REST Data:** `ccxt`, `websockets`, `pandas`, `pandas-ta`, `numpy`, `scipy`
* **Machine Learning & XAI:** `xgboost`, `scikit-learn`, `shap`, `matplotlib`
* **Sentiment Analysis:** CryptoPanic API, `nltk` VADER (`SentimentIntensityAnalyzer`), RSS (`CoinTelegraph` / `CoinDesk`)
* **Orchestration:** `langgraph`, `langchain-core`
* **Database & Memory:** Zero-cost SQLite (`TradeMemory` in `data/trade_memory.db` / `data/backtest_memory.db`)

---

## ⚙️ How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Add your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
   ```
3. **Run the pipeline:**
   ```bash
   # Test Telegram Futures VIP Signal Card delivery
   python main.py --test-telegram

   # Real-time Binance WebSocket multi-stream (5m/15m/1h/4h/1d)
   python main.py --live

   # Windows double-click launcher
   start_live.bat

   # Historical BTC/USDT Sniper backtest (15m execution with 1H/4H MTF context)
   python backtest.py

   # Quantitative diagnostic report from SQLite database
   python analyze_performance.py --db data/backtest_memory.db

   # Single REST cycle bootstrap
   python main.py
   ```
4. **Run unit tests:**
   ```bash
   python -m unittest test_data_loader test_agents -v
   ```
