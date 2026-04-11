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


# ── Market open/close helpers (IST only) ─────────────────────────────
# MCX Gold trades Monday–Friday, approximately 09:00–23:30 IST.
# All market-hour logic uses IST exclusively — no other timezone involved.
# Saturday and Sunday IST are always closed.

# MCX Gold market hours (IST)
MCX_MARKET_OPEN_HOUR = 9    # Monday–Friday market opens at 09:00 IST
MCX_MARKET_CLOSE_HOUR = 23  # Friday market closes at ~23:30 IST


def is_market_open(dt: datetime | None = None) -> bool:
    """Return True if the gold market is expected to be open at *dt* (IST).

    Uses MCX Gold hours in IST:
      - Saturday (5) and Sunday (6): always closed.
      - Monday before 09:00 IST: closed (extends the weekend flatline).
      - Monday 09:00 – Friday 23:30 IST: open.
    The caller can still force a prediction via the manual "Generate" button.

    This is the logical inverse of :func:`is_market_closed_ist` for the hours
    both functions cover (weekends + Monday pre-market).
    """
    if dt is None:
        dt = now_ist()
    # Saturday / Sunday — always closed
    if dt.weekday() >= 5:
        return False
    # Monday before MCX opens at 09:00 IST — still part of the weekend close
    if dt.weekday() == 0 and dt.hour < MCX_MARKET_OPEN_HOUR:
        return False
    return True


def is_market_closed_ist(dt: datetime) -> bool:
    """Return True if the predicted hour at *dt* (IST) falls outside MCX trading.

    Used to decide whether a prediction hour should be flatlined.
    Closed when:
      - Saturday or Sunday (any hour)
      - Monday before 09:00 IST (pre-market, extends the weekend flatline)
    Friday closes at 23:30 IST, but since predictions are at whole hours only,
    the 23:00 candle is the last partially active hour and is NOT flatlined.
    Saturday 00:00 onward is flatlined.
    Note: *dt* is expected to be in IST (naive or aware).  The function only
    inspects weekday() and hour, so a naive IST datetime works correctly.
    """
    wd = dt.weekday()  # Mon=0 … Sun=6
    if wd >= 5:  # Saturday / Sunday
        return True
    # Monday before MCX market open (09:00 IST) — extends the weekend flatline.
    # Tue–Fri pre-market hours are NOT flatlined because global gold (COMEX)
    # can still move during those hours and the ML model trains on them.
    if wd == 0 and dt.hour < MCX_MARKET_OPEN_HOUR:
        return True
    return False


def next_market_open_ist(dt: datetime | None = None) -> datetime:
    """Return the next MCX market-open time: Monday 09:00 IST.

    This is intended to be called when the market is closed (weekend).
    From Saturday it returns Monday 09:00; from Sunday it returns Monday 09:00.
    """
    if dt is None:
        dt = now_ist()
    # Advance to the next Monday (weekday 0)
    days_ahead = (7 - dt.weekday()) % 7  # Saturday→2, Sunday→1
    if days_ahead == 0:
        # Already Monday; if called on a weekday, advance to next Monday.
        days_ahead = 7
    next_monday = dt + timedelta(days=days_ahead)
    return next_monday.replace(
        hour=MCX_MARKET_OPEN_HOUR, minute=0, second=0, microsecond=0
    )
