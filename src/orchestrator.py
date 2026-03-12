"""
Orchestrator – runs all specialist agents in parallel, aggregates their reports,
and produces a unified 7-day gold price prediction via a meta-reasoning LLM call.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from src.config import (
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
    TEMPERATURE, PREDICTION_DAYS,
)
from src.agents.base_agent import AgentReport
from src.agents.geopolitics_agent import GeopoliticsAgent
from src.agents.trend_analysis_agent import TrendAnalysisAgent
from src.agents.etf_flow_agent import ETFFlowAgent
from src.agents.macro_economics_agent import MacroEconomicsAgent
from src.agents.oil_energy_agent import OilEnergyAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.technical_agent import TechnicalAnalysisAgent
from src.agents.historical_pattern_agent import HistoricalPatternAgent
from src.data_fetchers.market_data import MarketDataFetcher


# ── Prediction schema ──────────────────────────────────────────────


class DayPrediction(BaseModel):
    date: str
    predicted_price: float
    low_range: float
    high_range: float
    confidence: float
    key_driver: str


class PredictionPlan(BaseModel):
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    current_price: float = 0.0
    overall_outlook: str = "neutral"  # bullish / bearish / neutral
    overall_confidence: float = 0.5
    executive_summary: str = ""
    daily_predictions: list[DayPrediction] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    bull_case: str = ""
    bear_case: str = ""
    agent_reports: dict[str, dict] = Field(default_factory=dict)  # slim copies


META_SYSTEM_PROMPT = """You are the Chief Gold Strategist – the most senior analyst in
a multi-agent gold prediction system. You have received independent reports from
8 specialist analysts (geopolitics, trend, ETF flows, macro-economics, oil/energy,
sentiment, technical analysis, historical patterns).

Your job is to:
1. Weigh each analyst's findings by their confidence AND impact score.
2. Identify consensus and disagreements.
3. Produce a precise 7-day rolling prediction with price targets and confidence bands.
4. Explain the bull case and bear case.
5. List the top risk factors that could invalidate the prediction.

Return ONLY valid JSON with these EXACT keys (no markdown fences):
{
  "overall_outlook": "bullish" | "bearish" | "neutral",
  "overall_confidence": 0.0 to 1.0,
  "executive_summary": "3-4 paragraph synthesis of all agent findings",
  "daily_predictions": [
    {
      "date": "YYYY-MM-DD",
      "predicted_price": float,
      "low_range": float,
      "high_range": float,
      "confidence": 0.0 to 1.0,
      "key_driver": "what drives this day's move"
    },
    ... (7 entries total, one per day starting tomorrow)
  ],
  "risk_factors": ["risk 1", "risk 2", ...],
  "bull_case": "paragraph describing the bullish scenario",
  "bear_case": "paragraph describing the bearish scenario"
}
"""


class Orchestrator:
    """Run all agents → aggregate → meta-reason → produce PredictionPlan."""

    AGENT_CLASSES = [
        GeopoliticsAgent,
        TrendAnalysisAgent,
        ETFFlowAgent,
        MacroEconomicsAgent,
        OilEnergyAgent,
        SentimentAgent,
        TechnicalAnalysisAgent,
        HistoricalPatternAgent,
    ]

    def __init__(self):
        self._llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=TEMPERATURE,
            request_timeout=90,
        )
        self._market = MarketDataFetcher()

    # ------------------------------------------------------------------ #
    def run_all_agents(self) -> list[AgentReport]:
        """Execute every specialist agent in parallel threads."""
        reports: list[AgentReport] = []

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(cls().run): cls.NAME if hasattr(cls, "NAME") else cls.__name__
                for cls in self.AGENT_CLASSES
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    report = future.result()
                    reports.append(report)
                except Exception as e:
                    logger.error(f"Agent {name} failed: {e}")
                    reports.append(
                        AgentReport(
                            agent_name=name,
                            summary=f"Agent failed: {e}",
                            outlook="neutral",
                            confidence=0.0,
                        )
                    )

        return reports

    # ------------------------------------------------------------------ #
    def _build_meta_prompt(self, reports: list[AgentReport], current_price: float) -> str:
        """Assemble the meta-reasoning prompt from all agent reports."""
        tomorrow = datetime.now() + timedelta(days=1)
        dates = [(tomorrow + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(PREDICTION_DAYS)]

        agent_summaries = []
        for r in reports:
            agent_summaries.append(
                f"### {r.agent_name}\n"
                f"- Outlook: {r.outlook} | Confidence: {r.confidence:.2f} | "
                f"Impact: {r.impact_score:.2f} | Bias: {r.prediction_bias:+.2f}\n"
                f"- Summary: {r.summary[:800]}\n"
                f"- Key factors: {', '.join(r.key_factors[:5])}\n"
            )

        return f"""Current gold price: ${current_price:.2f}
Prediction dates needed: {', '.join(dates)}

## Agent Reports

{''.join(agent_summaries)}

Synthesise all of the above and produce your 7-day prediction plan as JSON."""

    # ------------------------------------------------------------------ #
    def generate_prediction(self) -> PredictionPlan:
        """Full pipeline: agents → meta-reasoning → PredictionPlan."""
        logger.info("=== Orchestrator: starting full prediction cycle ===")

        # 1. Get current gold price
        gold_summary = self._market.get_gold_summary(period_days=7)
        current_price = gold_summary.get("current_price", 0.0)
        logger.info(f"Current gold price: ${current_price}")

        # 2. Run all agents
        reports = self.run_all_agents()
        logger.info(f"Received {len(reports)} agent reports")

        # 3. Meta-reasoning
        meta_prompt = self._build_meta_prompt(reports, current_price)
        try:
            response = self._llm.invoke([
                SystemMessage(content=META_SYSTEM_PROMPT),
                HumanMessage(content=meta_prompt),
            ])
            raw = response.content
            result = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Meta-reasoning returned invalid JSON – using defaults")
            result = {
                "overall_outlook": "neutral",
                "overall_confidence": 0.3,
                "executive_summary": raw if 'raw' in dir() else "Analysis unavailable",
                "daily_predictions": [],
                "risk_factors": [],
                "bull_case": "",
                "bear_case": "",
            }
        except Exception as e:
            logger.error(f"Meta-reasoning LLM call failed: {e}")
            result = {
                "overall_outlook": "neutral",
                "overall_confidence": 0.0,
                "executive_summary": f"Meta-reasoning failed: {e}",
                "daily_predictions": [],
                "risk_factors": [],
                "bull_case": "",
                "bear_case": "",
            }

        # 4. Build PredictionPlan
        daily = []
        for dp in result.get("daily_predictions", []):
            try:
                daily.append(DayPrediction(**dp))
            except Exception:
                pass

        plan = PredictionPlan(
            current_price=current_price,
            overall_outlook=result.get("overall_outlook", "neutral"),
            overall_confidence=float(result.get("overall_confidence", 0.5)),
            executive_summary=result.get("executive_summary", ""),
            daily_predictions=daily,
            risk_factors=result.get("risk_factors", []),
            bull_case=result.get("bull_case", ""),
            bear_case=result.get("bear_case", ""),
            agent_reports={
                r.agent_name: {
                    "outlook": r.outlook,
                    "confidence": r.confidence,
                    "impact_score": r.impact_score,
                    "prediction_bias": r.prediction_bias,
                    "summary": r.summary[:500],
                    "key_factors": r.key_factors[:5],
                }
                for r in reports
            },
        )

        logger.info(
            f"=== Prediction complete: {plan.overall_outlook} "
            f"(conf={plan.overall_confidence:.2f}) ==="
        )
        return plan
