from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo

    INDIA_TZ = ZoneInfo("Asia/Kolkata")
except Exception:
    INDIA_TZ = timezone(timedelta(hours=5, minutes=30))

# IST offset from UTC (5 hours 30 minutes)
IST_OFFSET = timedelta(hours=5, minutes=30)


def now_ist() -> datetime:
    return datetime.now(INDIA_TZ)


def iso_now_ist() -> str:
    return now_ist().isoformat()


def parse_iso_to_ist(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=INDIA_TZ)
    return dt.astimezone(INDIA_TZ)


# ── 6-hour prediction slot helpers ───────────────────────────────────
# Predictions are aligned to fixed 6-hour slots: 00:00, 06:00, 12:00, 18:00 IST.

SLOT_HOURS = [0, 6, 12, 18]


def current_slot_ist() -> datetime:
    """Return the start of the current 6-hour slot in IST."""
    now = now_ist()
    candidates = [h for h in SLOT_HOURS if h <= now.hour]
    slot_hour = candidates[-1] if candidates else 0
    return now.replace(hour=slot_hour, minute=0, second=0, microsecond=0)


def next_slot_ist() -> datetime:
    """Return the start of the next 6-hour slot in IST."""
    now = now_ist()
    future_hours = [h for h in SLOT_HOURS if h > now.hour]
    if future_hours:
        next_hour = future_hours[0]
        return now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
    # Next slot is 00:00 tomorrow
    tomorrow = now + timedelta(days=1)
    return tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)


def slot_id(dt: datetime) -> str:
    """Convert a datetime to a slot identifier string like '2026-04-10T06:00'."""
    candidates = [h for h in SLOT_HOURS if h <= dt.hour]
    slot_hour = candidates[-1] if candidates else 0
    slot_dt = dt.replace(hour=slot_hour, minute=0, second=0, microsecond=0)
    return slot_dt.strftime("%Y-%m-%dT%H:%M")
