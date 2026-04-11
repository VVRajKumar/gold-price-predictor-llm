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


# ── Market open/close helpers ────────────────────────────────────────
# Gold futures (COMEX GC=F) trade Sunday 6 PM ET → Friday 5 PM ET.
# In IST that is roughly Monday 3:30 AM → Saturday 3:30 AM.
# MCX Gold trades Monday–Friday.
# For simplicity we treat the market as **closed** on Saturday and Sunday IST.
# This avoids generating stale predictions when prices do not move.


def is_market_open(dt: datetime | None = None) -> bool:
    """Return True if the gold market is expected to be open at *dt* (IST).

    Uses a conservative weekday check: Saturday (5) and Sunday (6) IST are
    considered closed.  The caller can still force a prediction via the
    manual "Generate" button.
    """
    if dt is None:
        dt = now_ist()
    # Monday=0 … Friday=4, Saturday=5, Sunday=6
    return dt.weekday() < 5


def next_market_open_ist(dt: datetime | None = None) -> datetime:
    """Return the start of the next market-open day (Monday 00:00 IST).

    This is intended to be called when the market is closed (weekend).
    From Saturday it returns Monday; from Sunday it returns Monday.
    """
    if dt is None:
        dt = now_ist()
    # Advance to the next Monday (weekday 0)
    days_ahead = (7 - dt.weekday()) % 7  # Saturday→2, Sunday→1
    if days_ahead == 0:
        # Already Monday; if called on a weekday, advance to next Monday.
        days_ahead = 7
    next_monday = dt + timedelta(days=days_ahead)
    return next_monday.replace(hour=0, minute=0, second=0, microsecond=0)
