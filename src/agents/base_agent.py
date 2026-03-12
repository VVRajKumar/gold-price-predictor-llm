"""
Base agent class – provides LLM integration and a standard interface.
Every specialist agent inherits from this.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from ..config import (
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
    TEMPERATURE,
)


class AgentReport(BaseModel):
    """Standardised output every agent returns."""

    agent_name: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    summary: str = ""
    outlook: str = ""                 # bullish / bearish / neutral
    confidence: float = 0.5           # 0-1 scale
    impact_score: float = 0.5         # how much this factor affects gold (0-1)
    key_factors: list[str] = Field(default_factory=list)
    data_points: dict[str, Any] = Field(default_factory=dict)
    prediction_bias: float = 0.0      # -1 (very bearish) to +1 (very bullish)
    raw_llm_response: str = ""


class BaseAgent(ABC):
    """Abstract base class for every specialist agent."""

    NAME: str = "base_agent"
    SYSTEM_PROMPT: str = "You are a financial analyst."

    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=TEMPERATURE,
            request_timeout=60,
        )

    # ------------------------------------------------------------------ #
    def _ask_llm(self, user_prompt: str) -> str:
        """Send a system+user message pair to the LLM and return the text."""
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"[{self.NAME}] LLM call failed: {e}")
            return f"ERROR: {e}"

    # ------------------------------------------------------------------ #
    @abstractmethod
    def gather_data(self) -> dict[str, Any]:
        """Fetch the data this agent needs."""
        ...

    @abstractmethod
    def analyse(self, data: dict[str, Any]) -> AgentReport:
        """Run analysis on the gathered data and return a report."""
        ...

    # ------------------------------------------------------------------ #
    def run(self) -> AgentReport:
        """Convenience: gather → analyse in one call."""
        logger.info(f"[{self.NAME}] Starting run …")
        data = self.gather_data()
        report = self.analyse(data)
        logger.info(
            f"[{self.NAME}] Done – outlook={report.outlook}, "
            f"confidence={report.confidence:.2f}, bias={report.prediction_bias:+.2f}"
        )
        return report
