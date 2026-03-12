"""Agents package – each module is one specialist agent."""

from .base_agent import BaseAgent
from .geopolitics_agent import GeopoliticsAgent
from .trend_analysis_agent import TrendAnalysisAgent
from .etf_flow_agent import ETFFlowAgent
from .macro_economics_agent import MacroEconomicsAgent
from .oil_energy_agent import OilEnergyAgent
from .sentiment_agent import SentimentAgent
from .technical_agent import TechnicalAnalysisAgent
from .historical_pattern_agent import HistoricalPatternAgent

__all__ = [
    "BaseAgent",
    "GeopoliticsAgent",
    "TrendAnalysisAgent",
    "ETFFlowAgent",
    "MacroEconomicsAgent",
    "OilEnergyAgent",
    "SentimentAgent",
    "TechnicalAnalysisAgent",
    "HistoricalPatternAgent",
]
