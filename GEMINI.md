# Crypto Quant Trading Pipeline - Agentic Workflow Instructions

## 1. Project Overview
This project builds a fully automated, deterministic cryptocurrency quantitative trading pipeline using a **Multi-Agent State-Machine Architecture** (LangGraph). The goal is to predict short-term BTC/USDT price direction and execute paper trades based on a combination of Technical Data, Machine Learning, and Sentiment Analysis.

**Target Market:** BTC/USDT (1-Hour or 4-Hour Timeframe)
**Execution Mode:** Paper Trading / Virtual Balance only (Version 1).

## 2. Core Technology Stack
- **Data & Features:** `ccxt` (Binance Public API), `pandas`, `pandas_ta`, `requests`.
- **Machine Learning:** `xgboost`, `scikit-learn` (for tabular feature classification).
- **Sentiment Analysis:** `transformers` (HuggingFace `ProsusAI/finbert`) or basic VADER via CryptoPanic RSS.
- **Orchestration:** `langgraph` (StateGraph) for deterministic agent routing.

## 3. Multi-Agent Architecture (LangGraph Nodes)
The pipeline must strictly follow this directed graph flow. Do NOT use AutoGen or CrewAI.

1. **State Definition (`TradingState`):** A shared TypedDict containing `current_price`, `technical_features`, `tech_signal`, `sentiment_score`, `final_decision`, and `capital_allocation`.
2. **Node 1: Data & Technical Agent (`tech_agent`)**
   - Fetches OHLCV data using `ccxt`.
   - Calculates RSI, MACD, and EMA 20/50.
   - Runs the XGBoost classification model to output a `tech_signal` (BULLISH/BEARISH/NEUTRAL) and `confidence`.
3. **Node 2: Sentiment Agent (`sentiment_agent`)**
   - Fetches recent BTC headlines (e.g., via CryptoPanic or NewsAPI).
   - Generates a `sentiment_score` (-1.0 to 1.0).
4. **Node 3: Risk Arbiter Agent (`risk_arbiter`)**
   - A hard-coded logic node.
   - Rule: Only approve "BUY" if `tech_signal` is BULLISH and `sentiment_score` > -0.10.
   - Calculates Position Size and Stop Loss based on ATR (Average True Range).
5. **Node 4: Execution Agent (`execution_agent`)**
   - Logs the trade to a local CSV/SQLite database (Paper Trading).

## 4. Strict Engineering Guardrails (DO NOT IGNORE)
- **Avoid Look-Ahead Bias:** When creating features for training XGBoost, ensure feature `t` does not include information from candle `t+1`. Shift target variables appropriately.
- **Stationarity:** Do not feed raw prices (e.g., $68,000) into the XGBoost model. Use percentage returns (`pct_change`) or log returns.
- **Modularity:** Separate concerns into different files (e.g., `data_loader.py`, `models.py`, `graph_builder.py`, `main.py`).

## 5. Implementation Phases (Step-by-Step for the Agent)
When the user asks to begin, follow these phases sequentially:

- **Phase 1: Environment & Data Loading:** Set up `requirements.txt`. Write a script to download historical BTC/USDT data via `ccxt` and calculate basic `pandas_ta` features.
- **Phase 2: ML Model Training:** Create a script to train a simple XGBoost classifier on historical data and save the model (`xgb_model.json`).
- **Phase 3: LangGraph Construction:** Build the `StateGraph`, define the Agent Nodes (functions), and connect the edges.
- **Phase 4: Execution & Logging:** Implement the paper trading loop that runs every hour and logs results.
