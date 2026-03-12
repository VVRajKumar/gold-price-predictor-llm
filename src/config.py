"""
Central configuration for the Gold Price Predictor system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = ROOT_DIR / "logs"

for d in [DATA_DIR, CACHE_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ─────────────────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-prod")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ── LLM Settings ────────────────────────────────────────────────────
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# ── System Settings ─────────────────────────────────────────────────
REFRESH_INTERVAL_MINUTES = int(os.getenv("REFRESH_INTERVAL_MINUTES", "30"))
PREDICTION_DAYS = 7
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
    "gold_fixing_price": "GOLDAMGBD228NLBM",
}
