"""Agents package – each module is one specialist agent."""

from src.agents.base_agent import BaseAgent
from src.agents.geopolitics_agent import GeopoliticsAgent
from src.agents.trend_analysis_agent import TrendAnalysisAgent
from src.agents.etf_flow_agent import ETFFlowAgent
from src.agents.macro_economics_agent import MacroEconomicsAgent
from src.agents.oil_energy_agent import OilEnergyAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.technical_agent import TechnicalAnalysisAgent
from src.agents.historical_pattern_agent import HistoricalPatternAgent

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
