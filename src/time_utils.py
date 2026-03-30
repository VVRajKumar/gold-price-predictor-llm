from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo

    INDIA_TZ = ZoneInfo("Asia/Kolkata")
except Exception:
    INDIA_TZ = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> datetime:
    return datetime.now(INDIA_TZ)


def iso_now_ist() -> str:
    return now_ist().isoformat()


def parse_iso_to_ist(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=INDIA_TZ)
    return dt.astimezone(INDIA_TZ)
