"""
Central configuration for the Gold Price Predictor system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Read from env vars first, fall back to Streamlit secrets (for Cloud)."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# ── Paths ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = ROOT_DIR / "logs"

for d in [DATA_DIR, CACHE_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ─────────────────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT = _get_secret("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = _get_secret("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = _get_secret("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-prod")
AZURE_OPENAI_API_VERSION = _get_secret("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
NEWS_API_KEY = _get_secret("NEWS_API_KEY")
FRED_API_KEY = _get_secret("FRED_API_KEY")

# ── LLM Settings ────────────────────────────────────────────────────
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

# ── System Settings ─────────────────────────────────────────────────
REFRESH_INTERVAL_MINUTES = int(os.getenv("REFRESH_INTERVAL_MINUTES", "30"))
FORECAST_GRANULARITY = os.getenv("FORECAST_GRANULARITY", "hourly")
PREDICTION_HOURS = int(os.getenv("PREDICTION_HOURS", "24"))
PLAN_REFRESH_HOURS = int(os.getenv("PLAN_REFRESH_HOURS", "1"))
ENABLE_XGBOOST_CORRECTION = os.getenv("ENABLE_XGBOOST_CORRECTION", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
XGBOOST_BLEND_WEIGHT = float(os.getenv("XGBOOST_BLEND_WEIGHT", "0.35"))
HISTORICAL_LOOKBACK_DAYS = 365

# ── Ticker Symbols ──────────────────────────────────────────────────
GOLD_TICKER = "GC=F"           # Gold Futures
SILVER_TICKER = "SI=F"         # Silver Futures
OIL_TICKER = "CL=F"           # Crude Oil Futures
DXY_TICKER = "DX-Y.NYB"       # US Dollar Index
SP500_TICKER = "^GSPC"        # S&P 500
TREASURY_10Y = "^TNX"         # 10-Year Treasury Yield
VIX_TICKER = "^VIX"           # Volatility Index

GOLD_ETF_TICKERS = ["GLD", "IAU", "SGOL", "BAR", "OUNZ"]
GOLD_MINER_TICKERS = ["GDX", "GDXJ", "NEM", "GOLD", "AEM"]

# ── FRED Series IDs ─────────────────────────────────────────────────
FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "real_interest_rate": "REAINTRATREARAT10Y",
    "us_debt": "GFDEBTN",
    "m2_money_supply": "M2SL",
    "inflation_expectation": "T5YIE",
}
