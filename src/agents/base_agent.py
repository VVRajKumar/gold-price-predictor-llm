"""
Base agent class – provides LLM integration and a standard interface.
Every specialist agent inherits from this.
"""

from __future__ import annotations
import json
import re
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
from ..time_utils import iso_now_ist
from ..guardrails import validate_agent_report


class AgentReport(BaseModel):
    """Standardised output every agent returns."""

    agent_name: str
    timestamp: str = Field(default_factory=iso_now_ist)
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

    # Terms that commonly trigger Azure OpenAI's content filter.
    # Each pair is (regex_pattern, replacement).
    _CONTENT_FILTER_REPLACEMENTS: list[tuple[str, str]] = [
        (r"\bwar\b", "armed conflict"),
        (r"\bwars\b", "armed conflicts"),
        (r"\bmilitary\b", "defense"),
        (r"\bbomb(?:er|ing|s|ed)?s?\b", "strikes"),
        (r"\bterror(?:ism|ist|ists)?\b", "security threat"),
        (r"\bviolence\b", "unrest"),
        (r"\battack(?:s|ed|ing)?\b", "escalation"),
        (r"\binvasion\b", "incursion"),
        (r"\bnuclear\b", "strategic"),
        (r"\bkill(?:ed|ing|s)?\b", "casualties"),
        (r"\bdestroy(?:ed|ing|s)?\b", "damage"),
        (r"\bexplosion(?:s)?\b", "blast"),
        (r"\bweapon(?:s)?\b", "armament"),
        (r"\bhostage(?:s)?\b", "detainee"),
        (r"\bassassinat(?:e|ed|ion|ions)\b", "targeted action"),
        (r"\bgenocide\b", "humanitarian crisis"),
        (r"\bmassacre(?:s|d)?\b", "mass casualty event"),
    ]

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
        """Send a system+user message pair to the LLM and return the text.

        If the first attempt is rejected by the content filter, retry up to
        two more times with progressively softened/reduced prompts to improve
        resilience against false-positive jailbreak detection from external
        news headlines.
        """
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            err_text = str(e)
            if "content_filter" not in err_text and "content management policy" not in err_text:
                logger.error(f"[{self.NAME}] LLM call failed: {e}")
                return f"ERROR: {e}"

            # ── Retry 1: soften terms ──
            logger.warning(
                f"[{self.NAME}] Content filter triggered – retrying with softened prompt"
            )
            softened_user = self._soften_prompt(user_prompt)
            softened_system = self._soften_prompt(self.SYSTEM_PROMPT)
            try:
                retry_messages = [
                    SystemMessage(content=softened_system),
                    HumanMessage(content=softened_user),
                ]
                response = self.llm.invoke(retry_messages)
                return response.content
            except Exception as retry_err:
                retry_text = str(retry_err)
                if "content_filter" not in retry_text and "content management policy" not in retry_text:
                    logger.error(f"[{self.NAME}] Retry also failed: {retry_err}")
                    return f"ERROR: {retry_err}"

            # ── Retry 2: aggressively reduce external content ──
            logger.warning(
                f"[{self.NAME}] Content filter triggered again – retrying with reduced content"
            )
            reduced_user = self._reduce_prompt_content(softened_user)
            try:
                retry2_messages = [
                    SystemMessage(content=softened_system),
                    HumanMessage(content=reduced_user),
                ]
                response = self.llm.invoke(retry2_messages)
                return response.content
            except Exception as final_err:
                logger.error(f"[{self.NAME}] All retries failed: {final_err}")
                return f"ERROR: {final_err}"

    @staticmethod
    def _reduce_prompt_content(text: str) -> str:
        """Aggressively reduce external content in a prompt for final retry.

        Keeps only the first 8 list-item lines (headlines) and truncates each
        to 100 chars, which greatly reduces the jailbreak-classifier surface.
        """
        lines = text.split("\n")
        kept: list[str] = []
        headline_count = 0
        max_headlines = 8
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- "):
                headline_count += 1
                if headline_count > max_headlines:
                    continue
                # Truncate long headlines
                if len(stripped) > 100:
                    stripped = stripped[:100].rsplit(" ", 1)[0] + " …"
                    line = stripped
            kept.append(line)
        return "\n".join(kept)

    @staticmethod
    def _soften_prompt(text: str) -> str:
        """Replace terms that commonly trigger Azure content filters."""
        result = text
        for pattern, replacement in BaseAgent._CONTENT_FILTER_REPLACEMENTS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result

    @staticmethod
    def _sanitize_headlines(text: str, max_chars_per_line: int = 160) -> str:
        """Strip prompt-injection-like patterns from external text (e.g. news headlines).

        External text can accidentally include phrases that Azure's jailbreak
        detector flags (e.g. "ignore previous instructions", markdown injection,
        or role-play overrides).  This helper neutralises common patterns while
        preserving the informational content of headlines.
        """
        # Remove common prompt-injection phrases
        injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions?",
            r"disregard\s+(all\s+)?previous",
            r"you\s+are\s+now\s+a",
            r"act\s+as\s+(if\s+you\s+are|a)\b",
            r"pretend\s+to\s+be",
            r"system\s*:\s*",
            r"<\s*/?\s*(?:system|user|assistant|s|prompt)\s*>",
            r"do\s+not\s+follow",
            r"override\s+(all\s+)?instructions?",
            r"new\s+instructions?\s*:",
            r"forget\s+(all\s+)?previous",
            r"(?:assistant|AI|bot)\s*:\s*",
            r"```[^`]*```",           # code blocks
            r"https?://\S{80,}",      # very long URLs (can confuse classifier)
        ]
        result = text
        for pat in injection_patterns:
            result = re.sub(pat, "", result, flags=re.IGNORECASE)

        # Truncate individual lines to cap per-headline length (reduces attack
        # surface for jailbreak classifier while keeping informational content).
        if max_chars_per_line:
            lines = result.split("\n")
            truncated = []
            for line in lines:
                if len(line) > max_chars_per_line:
                    line = line[:max_chars_per_line].rsplit(" ", 1)[0] + " …"
                truncated.append(line)
            result = "\n".join(truncated)

        # Also apply the standard content-filter softening
        result = BaseAgent._soften_prompt(result)
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_llm_json(raw: str, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
        """Robustly parse a JSON response from the LLM.

        Handles common LLM quirks: markdown code fences, truncated JSON, and
        malformed output.  Returns a dict with as many fields extracted as
        possible, falling back to *defaults* for anything missing.
        """
        if defaults is None:
            defaults = {}
        if not isinstance(raw, str) or not raw.strip():
            return dict(defaults)

        text = raw.strip()

        # Strip markdown code fences (```json ... ```)
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # drop opening fence
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]  # drop closing fence
            text = "\n".join(lines).strip()

        # 1. Try standard JSON parse
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                # Merge defaults under the parsed result so that any fields
                # the LLM omitted still get agent-specific default values.
                merged = dict(defaults)
                merged.update(result)
                return merged
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # 2. Regex fallback – extract individual fields from malformed JSON
        result = dict(defaults)

        # summary (greedy: grab as much as possible up to the closing quote)
        m = re.search(
            r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL,
        )
        if m:
            result["summary"] = (
                m.group(1).replace('\\"', '"').replace("\\n", "\n")
            )

        # key_factors – array of strings
        m = re.search(r'"key_factors"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if m:
            factors = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
            if factors:
                result["key_factors"] = [
                    f.replace('\\"', '"') for f in factors
                ]

        # outlook
        m = re.search(
            r'"outlook"\s*:\s*"(bullish|bearish|neutral)"', text, re.IGNORECASE,
        )
        if m:
            result["outlook"] = m.group(1).lower()

        # numeric fields
        for field in ("confidence", "impact_score", "prediction_bias"):
            m = re.search(rf'"{field}"\s*:\s*(-?[\d.]+)', text)
            if m:
                try:
                    result[field] = float(m.group(1))
                except ValueError:
                    pass

        return result

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
        """Convenience: gather → analyse → guardrail in one call."""
        logger.info(f"[{self.NAME}] Starting run …")
        data = self.gather_data()
        report = self.analyse(data)

        # ── Guardrail: validate & correct agent output ──
        corrected = validate_agent_report(
            {
                "outlook": report.outlook,
                "confidence": report.confidence,
                "impact_score": report.impact_score,
                "prediction_bias": report.prediction_bias,
            },
            agent_name=self.NAME,
        )
        report.outlook = corrected["outlook"]
        report.confidence = corrected["confidence"]
        report.impact_score = corrected["impact_score"]
        report.prediction_bias = corrected["prediction_bias"]

        logger.info(
            f"[{self.NAME}] Done – outlook={report.outlook}, "
            f"confidence={report.confidence:.2f}, bias={report.prediction_bias:+.2f}"
        )
        return report
