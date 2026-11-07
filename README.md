# NeuralPulse Trading 📈

An institutional-grade, multi-agent quantitative trading pipeline for cryptocurrency markets. This project leverages Machine Learning (XGBoost) and deterministic AI Agent orchestration (LangGraph) to analyze market data, predict price movements, and execute risk-adjusted trading strategies.

## 🏗️ Architecture

The system is designed with a **Multi-Agent State-Machine Architecture** to ensure determinism, strict risk isolation, and auditability—key requirements for quantitative finance.

*   **Data Ingestion Pipeline:** Connects to Binance via `ccxt` to fetch historical OHLCV data.
*   **Feature Engineering:** Computes strictly stationary technical indicators (RSI, normalized MACD, relative EMAs, Volatility) using `pandas_ta`, actively guarding against look-ahead bias.
*   **Predictive Engine (XGBoost):** A lightweight, fast classifier trained on numerical data to predict localized market direction.
*   **Agent Orchestration (LangGraph):** *[Upcoming in Phase 3]* Coordinates specialized agents (Technical Analyst, Sentiment Analyst, and Risk Arbiter) to evaluate signals and make final trading decisions.

## 🚀 Current Progress

*   ✅ **Phase 1: Data Pipeline** - Completed. Robust data loader with pagination, error handling, and stationary feature extraction.
*   ✅ **Phase 2: Model Training** - Completed. XGBoost model trained and evaluated, serialized for production use (`xgb_model.json`).
*   ⏳ **Phase 3: Agent Orchestration** - In progress. Implementing LangGraph state machines for multi-agent decision-making.

## 💻 Tech Stack & Environment

*   **Language:** Python 3.12.7
*   **Core Libraries:** `xgboost`, `pandas`, `pandas-ta`, `scikit-learn`
*   **Exchange API:** `ccxt` (Binance)
*   **AI/Orchestration:** `langgraph`, `langchain-core`

## ⚙️ How to Run (Locally)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/NeuralPulse_Trading.git](https://github.com/YourUsername/NeuralPulse_Trading.git)
   cd NeuralPulse_Trading