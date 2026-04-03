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
XGBOOST_BLEND_WEIGHT = float(os.getenv("XGBOOST_BLEND_WEIGHT", "0.45"))
HISTORICAL_LOOKBACK_DAYS = 365

# ── Ticker Symbols ──────────────────────────────────────────────────
GOLD_TICKER = "GC=F"           # COMEX Gold Futures (base reference)
MCX_GOLD_TICKER = "GOLD.NS"   # MCX Gold via NSE (INR-denominated proxy)
SILVER_TICKER = "SI=F"         # Silver Futures
OIL_TICKER = "CL=F"           # Crude Oil Futures
DXY_TICKER = "DX-Y.NYB"       # US Dollar Index (affects INR gold)
USDINR_TICKER = "INR=X"       # USD/INR exchange rate
NIFTY_TICKER = "^NSEI"        # Nifty 50 Index
INDIA_VIX_TICKER = "^INDIAVIX" # India VIX
SENSEX_TICKER = "^BSESN"      # BSE Sensex
TREASURY_10Y = "^TNX"         # US 10-Year Treasury (global reference)

GOLD_ETF_TICKERS = [
    "GOLDBEES.NS",       # Nippon India ETF Gold BeES
    "ICICIGOLD.NS",      # ICICI Prud Gold ETF
    "SBIGETF.NS",        # SBI Gold ETF
    "HDFCGOLD.NS",       # HDFC Gold ETF
    "KOTAKGOLD.NS",      # Kotak Gold ETF
    "TATAGOLD.NS",       # Tata Gold ETF
]
GOLD_FUND_TICKERS = [
    "0P0001BALK.BO",     # SBI Gold Mutual Fund
    "0P0000XVLE.BO",     # HDFC Gold Mutual Fund
    "0P0001BAH4.BO",     # Axis Gold Mutual Fund
    "0P0001BAL8.BO",     # Kotak Gold Mutual Fund
    "0P0001BAL4.BO",     # Nippon India Gold Mutual Fund
    "0P0001BAII.BO",     # ICICI Prudential Gold Mutual Fund
]
GOLD_MINER_TICKERS = ["TITAN.NS", "TBZ.NS", "RAJESHEXPO.NS"]

# ── FRED Series IDs (global macro reference – supplements India-specific data) ──
FRED_SERIES = {
    "us_fed_funds_rate": "FEDFUNDS",          # US Fed rate (global spillover)
    "us_cpi": "CPIAUCSL",                     # US CPI (global inflation ref)
    "us_real_interest_rate": "REAINTRATREARAT10Y",
    "us_m2_money_supply": "M2SL",
    "us_inflation_expectation": "T5YIE",
}
