# 🥇 Agentic Gold Price Prediction System

An **AI-powered gold price forecasting system** that predicts next-day and intra-day (24-hour) MCX gold prices in INR/10g. It combines a **3-model stacking ML ensemble** with **LLM-generated market intelligence** from 8 specialist agents to deliver probabilistic forecasts and actionable buy/sell/hold recommendations for MCX traders, retail investors, and jewellery businesses.

### ML Ensemble Architecture (2-Layer Stacking)

| Layer | Model | Role |
|-------|-------|------|
| **Layer 1 – Base Learners** | XGBoost | Non-linear gradient boosting on 16 time-series features |
| **Layer 1 – Base Learners** | LightGBM | Leaf-wise gradient boosting (complementary to XGBoost) |
| **Layer 1 – Base Learners** | Ridge Regression | Linear baseline — captures trends the tree models may underweight |
| **Layer 2 – Meta-Learner** | Ridge Regression | Learns the optimal blend of the three base predictions → final point estimate |

All three Layer 1 models independently predict on the same raw feature set. The Layer 2 Ridge meta-learner is trained on the out-of-fold (validation set) predictions of the three base models, so it learns how to weight their outputs optimally — it does **not** see the raw features directly.

**Confidence bands:** XGBoost quantile regression (α = 0.05 / 0.95) provides a 90 % prediction interval around the stacked forecast.

**Explainability:** SHAP TreeExplainer on the XGBoost base model delivers per-feature contributions for every prediction.

**Macro signals used as features:** USD/INR exchange rate, DXY (US Dollar Index), crude oil (CL=F), US real interest rates, CPI (inflation), India VIX, gold ETF flows, and intra-day session patterns extracted from 90 days of COMEX (GC=F) hourly candles.

### LLM Agent Layer

Eight specialist agents (geopolitics, macro economics, ETF flows, oil & energy, sentiment, technical analysis, historical patterns, trend analysis) each analyse domain-specific live data using GPT-4o and produce structured signals (outlook, confidence, prediction bias). A "Chief Strategist" meta-LLM synthesises all eight reports, and the resulting composite signal adjusts the ML ensemble's point forecast post-prediction.

**Target users:** MCX intraday traders, retail gold investors, and jewellery businesses hedging procurement costs.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     STREAMLIT DASHBOARD                        │
│               (app.py – live-updating UI)                      │
├────────────────────────────────────────────────────────────────┤
│                    PREDICTION ENGINE                           │
│          (scheduling, caching, accuracy tracking)              │
├────────────────────────────────────────────────────────────────┤
│   ML ENSEMBLE (2-layer stacking)          LLM AGENT LAYER      │
│  ┌──────────────────────────────┐   ┌───────────────────────┐  │
│  │  Layer 1 – Base Learners     │   │  8 Specialist Agents  │  │
│  │  • XGBoost (gradient boost)  │   │  (GPT-4o per agent)   │  │
│  │  • LightGBM (leaf-wise)      │   │  Geopolitics, Macro,  │  │
│  │  • Ridge (linear baseline)   │   │  ETF, Oil, Sentiment, │  │
│  ├──────────────────────────────┤   │  Technical, Historical│  │
│  │  Layer 2 – Meta-Learner      │   │  Trend Analysis       │  │
│  │  • Ridge blends base preds   │   └──────────┬────────────┘  │
│  ├──────────────────────────────┤              │ composite      │
│  │  Confidence Bands            │              │ signal         │
│  │  • XGBoost quantile (5/95%)  │◄─────────────┘                │
│  ├──────────────────────────────┤                               │
│  │  Explainability              │                               │
│  │  • SHAP (XGBoost base model) │                               │
│  └──────────────────────────────┘                               │
├────────────────────────────────────────────────────────────────┤
│              DATA FETCHERS  (Yahoo Finance, FRED, NewsAPI)     │
│   COMEX GC=F hourly · MCX GOLD.NS · USD/INR · DXY · VIX ·     │
│   Crude Oil · US rates · CPI · Indian Gold ETFs                │
└────────────────────────────────────────────────────────────────┘
```

## 8 Specialist Agents

| # | Agent | Responsibility | Data Sources |
|---|-------|---------------|-------------|
| 1 | 🌍 **Geopolitics** | Wars, sanctions, BRICS, central bank reserves | NewsAPI, RSS |
| 2 | 📈 **Trend Analysis** | Price trends, moving averages, momentum | Yahoo Finance |
| 3 | 💰 **ETF Flows** | Gold ETF inflows/outflows, institutional demand | Yahoo Finance (GLD, IAU, etc.) |
| 4 | 🏦 **Macro Economics** | Interest rates, CPI, USD, M2, debt | FRED API |
| 5 | 🛢️ **Oil & Energy** | Oil prices, OPEC, gold-oil ratio | Yahoo Finance, NewsAPI |
| 6 | 😨 **Sentiment** | VIX, fear/greed, news tone | Yahoo Finance, NewsAPI |
| 7 | 📊 **Technical** | RSI, MACD, Bollinger, support/resistance | Yahoo Finance (computed) |
| 8 | 📜 **Historical** | Seasonal patterns, YoY, drawdowns, cycles | Yahoo Finance |

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Azure OpenAI resource (endpoint + API key + deployment name)
- (Optional) NewsAPI key, FRED API key

### 2. Install

```bash
cd "gold price predictor llm"
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 3. Configure

```bash
copy .env.example .env
# Edit .env and add your API keys
```

**Minimum required:** `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`

Optional but recommended:
- `NEWS_API_KEY` – enables geopolitics + sentiment agents' news analysis
- `FRED_API_KEY` – enables macro-economics agent's FRED data

### 4. Run the Dashboard

```bash
streamlit run app.py
```

Then click **"Generate New Prediction"** in the sidebar.

### 5. CLI Mode

```bash
# Single prediction with formatted output
python run.py --once

# Single prediction as JSON
python run.py --once --json

# Daemon mode (auto-refreshes every 30 min)
python run.py --daemon
```

## How It Works

1. **Data Collection:** Market data fetchers pull COMEX (GC=F) hourly candles, MCX daily prices (GOLD.NS), USD/INR, DXY, crude oil, VIX, Indian gold ETF flows, and US macro indicators (Fed rate, CPI, real rates) from Yahoo Finance and FRED.
2. **Feature Engineering:** 16 time-series features per bar — lagged prices, rolling statistics, and intra-day session patterns — are built from the last 90 days of hourly COMEX data.
3. **ML Ensemble Training (2-layer stacking):**
   - **Layer 1** – three base learners (XGBoost, LightGBM, Ridge) each train independently on 80 % of the feature matrix.
   - **Quantile models** – two additional XGBoost regressors (α = 0.05, 0.95) train for the 90 % prediction interval.
   - **Layer 2 (meta-learner)** – a Ridge model is trained on the out-of-fold predictions of the three base learners (the remaining 20 % validation set), learning the optimal blend.
4. **LLM Agent Analysis:** Eight specialist agents run in parallel threads; each uses GPT-4o to analyse its domain data and produce a structured report with outlook, confidence, and prediction bias.
5. **Meta-Reasoning:** A "Chief Strategist" LLM call synthesises all 8 agent reports into a composite directional signal and executive summary.
6. **Prediction:** For each of the next 24 hours, all three base learners produce a prediction; the meta-learner blends them into a point estimate; the LLM composite signal applies a post-prediction adjustment; session-pattern shaping produces a realistic intra-day price curve in INR/10g.
7. **Explainability:** SHAP values on the XGBoost base model are computed once per prediction cycle, showing which features drove the forecast.
8. **Live Dashboard:** Streamlit renders hourly forecasts, confidence bands, SHAP charts, agent breakdowns, and accuracy tracking with auto-refresh.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | – | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | – | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o-prod` | Deployment name |
| `TEMPERATURE` | `0.0` | Lower = more deterministic LLM output |
| `REFRESH_INTERVAL_MINUTES` | `30` | Auto-refresh cycle |
| `PREDICTION_HOURS` | `24` | Intra-day forecast horizon (hours) |
| `PLAN_REFRESH_HOURS` | `6` | How often to re-train the ML ensemble |
| `HISTORICAL_LOOKBACK_DAYS` | `365` | How far back to fetch data for agents |

## Project Structure

```
gold price predictor llm/
├── app.py                          # Streamlit dashboard
├── run.py                          # CLI runner
├── requirements.txt
├── .env.example
├── .streamlit/config.toml          # Dashboard theme
├── src/
│   ├── config.py                   # Central configuration
│   ├── orchestrator.py             # Agent coordinator + meta-LLM
│   ├── prediction_engine.py        # Scheduling, caching, Prophet
│   ├── agents/
│   │   ├── base_agent.py           # Abstract base class
│   │   ├── geopolitics_agent.py    # 🌍 Geopolitics
│   │   ├── trend_analysis_agent.py # 📈 Trend Analysis
│   │   ├── etf_flow_agent.py       # 💰 ETF Flows
│   │   ├── macro_economics_agent.py# 🏦 Macro Economics
│   │   ├── oil_energy_agent.py     # 🛢️ Oil & Energy
│   │   ├── sentiment_agent.py      # 😨 Sentiment
│   │   ├── technical_agent.py      # 📊 Technical Analysis
│   │   └── historical_pattern_agent.py # 📜 Historical Patterns
│   └── data_fetchers/
│       ├── market_data.py          # Yahoo Finance
│       ├── news_data.py            # NewsAPI + RSS
│       ├── macro_data.py           # FRED API
│       └── etf_data.py             # Gold ETF & miners
├── data/cache/                     # Prediction cache
└── logs/                           # Application logs
```

## Cost Estimate

Each prediction cycle makes ~9 LLM calls (8 agents + 1 meta-reasoning). With GPT-4o via Azure OpenAI:
- ~$0.05–0.15 per cycle depending on data volume
- At 30-min refresh: ~$2.40–7.20/day

Use a smaller deployment (e.g. `gpt-4o-mini`) in your Azure deployment to reduce costs by ~10x.

## Disclaimer

⚠️ **This is an educational/research project. It is NOT financial advice.** Gold prices are inherently unpredictable. Always consult a licensed financial advisor before making investment decisions.
