# 🥇 Agentic Gold Price Prediction System

A **multi-agent AI system** that predicts gold prices over a 7-day rolling window by synthesising insights from 8 specialist agents across geopolitics, macroeconomics, technical analysis, sentiment, and more.

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                     │
│              (app.py – live-updating UI)                   │
├───────────────────────────────────────────────────────────┤
│                   PREDICTION ENGINE                        │
│        (scheduling, caching, Prophet baseline)             │
├───────────────────────────────────────────────────────────┤
│                     ORCHESTRATOR                           │
│       (parallel agent execution + meta-reasoning)          │
├────────┬────────┬────────┬────────┬────────┬──────────────┤
│ Geo-   │ Trend  │  ETF   │ Macro  │  Oil   │  Sentiment   │
│politics│Analysis│ Flows  │ Econ   │Energy  │   Agent      │
├────────┼────────┼────────┴────────┴────────┼──────────────┤
│Techni- │Histori-│                          │              │
│  cal   │  cal   │     DATA FETCHERS        │   LLM (GPT)  │
│Analysis│Pattern │  (Yahoo, FRED, News)     │              │
└────────┴────────┴──────────────────────────┴──────────────┘
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
- OpenAI API key (GPT-4o recommended)
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

**Minimum required:** `OPENAI_API_KEY`

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

1. **Data Collection:** Each agent fetches domain-specific data (prices, news, macro indicators, etc.)
2. **Agent Analysis:** Each agent uses GPT-4o to analyse its data and produce a structured report with:
   - Outlook (bullish/bearish/neutral)
   - Confidence score (0–1)
   - Impact score (how relevant this factor is right now)
   - Prediction bias (−1 to +1)
   - Key factors and supporting reasoning
3. **Orchestration:** All 8 agents run in parallel threads
4. **Meta-Reasoning:** A senior "Chief Strategist" LLM call synthesises all agent reports into:
   - 7-day price predictions with confidence bands
   - Executive summary, bull/bear cases
   - Risk factors
5. **Statistical Baseline:** Facebook Prophet provides an independent statistical forecast as a sanity check
6. **Live Dashboard:** Streamlit renders everything with auto-refresh capability

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_MODEL` | `gpt-4o` | LLM model to use |
| `TEMPERATURE` | `0.2` | Lower = more deterministic |
| `REFRESH_INTERVAL_MINUTES` | `30` | Auto-refresh cycle |
| `PREDICTION_DAYS` | `7` | Forecast horizon |
| `HISTORICAL_LOOKBACK_DAYS` | `365` | How far back to fetch data |

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

Each prediction cycle makes ~9 LLM calls (8 agents + 1 meta-reasoning). With GPT-4o:
- ~$0.05–0.15 per cycle depending on data volume
- At 30-min refresh: ~$2.40–7.20/day

Use `gpt-4o-mini` in `.env` to reduce costs by ~10x.

## Disclaimer

⚠️ **This is an educational/research project. It is NOT financial advice.** Gold prices are inherently unpredictable. Always consult a licensed financial advisor before making investment decisions.
