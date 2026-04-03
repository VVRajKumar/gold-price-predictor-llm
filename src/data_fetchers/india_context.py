"""
India-specific macro context – RBI repo rate, CPI, import duty,
festival calendar, and monsoon season awareness.

These data points have no free real-time API.  Values are updated
manually when the RBI / government announces changes (typically a few
times per year).  The module also provides a calendar-driven view of
upcoming festivals and seasonal demand patterns so LLM agents can
reason about them.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

# ── Static macro context (update when RBI / govt announces changes) ──

_RBI_REPO_RATE = {
    "rate_pct": 6.00,
    "effective_date": "2025-04-09",
    "direction": "cutting",              # cutting | holding | hiking
    "last_change_bps": -25,              # basis points of last move
    "next_review": "2025-06-06",
    "notes": "RBI cut 25 bps in Apr-2025 amid slowing growth; stance 'accommodative'.",
}

_INDIA_CPI = {
    "latest_pct": 3.61,
    "reference_month": "2025-02",
    "trend": "falling",                  # rising | stable | falling
    "rbi_target_pct": 4.0,
    "rbi_tolerance_band": "2-6%",
    "notes": "Feb-2025 CPI at 3.61%, below RBI target. Food inflation easing.",
}

_GOLD_IMPORT_DUTY = {
    "total_duty_pct": 6.0,               # Basic customs + AIDC
    "basic_customs_pct": 5.0,
    "aidc_pct": 1.0,                     # Agriculture Infrastructure Development Cess
    "gst_pct": 3.0,                      # GST on gold
    "effective_date": "2024-07-23",
    "notes": "Duty slashed from 15% to 6% in Union Budget Jul-2024.",
}

# ── Indian festival / seasonal demand calendar ──────────────────────
# Each entry: (month, day_start, day_end, name, demand_impact)
# demand_impact: "very_high" | "high" | "moderate"
# Dates are approximate and shift yearly – good enough for LLM context.

_FESTIVAL_CALENDAR: list[dict[str, Any]] = [
    {"month": 1,  "name": "Makar Sankranti / Pongal",    "demand": "moderate",  "days": "13-15"},
    {"month": 2,  "name": "Valentine's week gifting",     "demand": "moderate",  "days": "7-14"},
    {"month": 3,  "name": "Ugadi / Gudi Padwa",          "demand": "moderate",  "days": "20-31"},
    {"month": 4,  "name": "Akshaya Tritiya",             "demand": "very_high", "days": "mid-Apr to early-May"},
    {"month": 4,  "name": "Ram Navami",                  "demand": "moderate",  "days": "early-Apr"},
    {"month": 5,  "name": "Wedding season (late spring)", "demand": "high",     "days": "1-31"},
    {"month": 8,  "name": "Raksha Bandhan",              "demand": "moderate",  "days": "mid-Aug"},
    {"month": 8,  "name": "Onam (Kerala)",               "demand": "high",      "days": "late-Aug to early-Sep"},
    {"month": 9,  "name": "Ganesh Chaturthi",            "demand": "moderate",  "days": "early-Sep"},
    {"month": 9,  "name": "Navratri begins",             "demand": "high",      "days": "late-Sep to mid-Oct"},
    {"month": 10, "name": "Dussehra",                    "demand": "high",      "days": "early-Oct"},
    {"month": 10, "name": "Dhanteras",                   "demand": "very_high", "days": "late-Oct"},
    {"month": 10, "name": "Diwali",                      "demand": "very_high", "days": "late-Oct to early-Nov"},
    {"month": 11, "name": "Wedding season (peak)",       "demand": "very_high", "days": "Nov-Feb"},
    {"month": 12, "name": "Christmas / Year-end",        "demand": "moderate",  "days": "25-31"},
]

_MONSOON_MONTHS = {6, 7, 8, 9}  # Jun–Sep: SW monsoon


# ── Public helpers ──────────────────────────────────────────────────

def get_india_macro_context() -> dict[str, Any]:
    """Return a dict of India-specific macro context for LLM agents."""
    return {
        "rbi_repo_rate": _RBI_REPO_RATE,
        "india_cpi": _INDIA_CPI,
        "gold_import_duty": _GOLD_IMPORT_DUTY,
    }


def get_festival_context(ref_date: date | None = None) -> dict[str, Any]:
    """Return festival/season context for the given (or current) date."""
    today = ref_date or date.today()
    month = today.month

    # Festivals in current month and next month
    upcoming = [
        f for f in _FESTIVAL_CALENDAR
        if f["month"] == month or f["month"] == (month % 12) + 1
    ]

    # Monsoon flag
    is_monsoon = month in _MONSOON_MONTHS

    # Wedding season: Oct–Feb
    is_wedding_season = month in {10, 11, 12, 1, 2}

    # Seasonal demand level
    if any(f["demand"] == "very_high" for f in upcoming if f["month"] == month):
        seasonal_demand = "very_high"
    elif any(f["demand"] == "high" for f in upcoming if f["month"] == month):
        seasonal_demand = "high"
    elif is_wedding_season:
        seasonal_demand = "high"
    elif is_monsoon:
        seasonal_demand = "low"          # rural buyers focused on agriculture
    else:
        seasonal_demand = "moderate"

    return {
        "current_month": today.strftime("%B %Y"),
        "upcoming_festivals": upcoming,
        "is_monsoon_season": is_monsoon,
        "monsoon_note": (
            "SW monsoon active – rural gold demand typically subdued as "
            "farmers prioritise agriculture spending."
            if is_monsoon else None
        ),
        "is_wedding_season": is_wedding_season,
        "seasonal_demand_level": seasonal_demand,
    }
