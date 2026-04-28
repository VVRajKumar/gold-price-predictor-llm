"""
Chart utilities shared by the main dashboard (app.py) and the Prediction
Archive page.
"""

from __future__ import annotations

from datetime import datetime

# Gaps wider than this many hours trigger a line break in Plotly traces.
# 6 hours comfortably covers any intra-day gap while catching the ~65-hour
# weekend closure (Friday 23:00 → Monday 09:00 IST).
_MAX_CONTIGUOUS_GAP_HOURS = 6

# Known offline windows: list of (start, end) naive datetimes in IST.
# Gaps whose entire span falls within one of these windows are treated as
# contiguous — the chart connects the last data point before the outage
# directly to the first data point after it, with no line break.
_KNOWN_OFFLINE_WINDOWS: list[tuple[datetime, datetime]] = [
    # Apr 27 09:00 IST → Apr 28 11:00 IST – no predictions were generated
    (datetime(2026, 4, 27, 9, 0, 0), datetime(2026, 4, 28, 11, 0, 0)),
]


def is_in_offline_window(dt: datetime) -> bool:
    """Return True when *dt* falls inside any known offline window.

    Used to filter out data points (including synthetic gap-fill rows) that
    belong to a period when no predictions were generated, so those hours are
    fully excluded from all chart traces.  Both the start and end of the
    window are treated as naive IST timestamps; timezone info is stripped
    before comparison.
    """
    def _naive(d: datetime) -> datetime:
        return d.replace(tzinfo=None) if d.tzinfo is not None else d

    dtn = _naive(dt)
    for win_start, win_end in _KNOWN_OFFLINE_WINDOWS:
        if win_start <= dtn < win_end:
            return True
    return False


def _gap_spans_offline_window(t1: datetime, t2: datetime) -> bool:
    """Return True when the gap [t1, t2] is *entirely explained* by known offline windows.

    A gap is considered explained only when the duration NOT covered by any
    offline window is within the normal contiguous-gap threshold.  This
    prevents a weekend gap from being bridged simply because it happens to
    overlap with an offline window — only gaps whose remaining (non-offline)
    portion is ≤ _MAX_CONTIGUOUS_GAP_HOURS are bridged.

    Both timestamps are normalised to naive (no timezone) before comparison.
    """
    def _naive(dt: datetime) -> datetime:
        return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt

    t1n = _naive(t1)
    t2n = _naive(t2)
    total_gap_hours = (t2n - t1n).total_seconds() / 3600

    # Sum up how many hours of this gap fall inside any offline window.
    offline_overlap_hours = 0.0
    for win_start, win_end in _KNOWN_OFFLINE_WINDOWS:
        overlap_start = max(t1n, win_start)
        overlap_end = min(t2n, win_end)
        if overlap_end > overlap_start:
            offline_overlap_hours += (overlap_end - overlap_start).total_seconds() / 3600

    # Bridge only when the non-offline remainder is within the normal threshold.
    remaining_gap = total_gap_hours - offline_overlap_hours
    return remaining_gap <= _MAX_CONTIGUOUS_GAP_HOURS


def break_at_gaps(dates, *value_cols):
    """Insert ``None`` gap-breakers so Plotly line traces don't draw
    misleading diagonals across weekend market closures.

    When only market-open hours are plotted, consecutive data points
    may jump from Friday evening to Monday morning.  Plotly draws a
    straight line between them unless there is a ``None`` in the data.
    This helper detects such jumps and inserts ``None`` values.

    Gaps that span a :data:`_KNOWN_OFFLINE_WINDOWS` entry are bridged
    (no ``None`` inserted) so the chart stays connected across planned
    maintenance or outage periods.

    Parameters
    ----------
    dates : sequence of datetime
        Sorted timestamps (pd.Series, list, or array).
    *value_cols : sequence
        One or more aligned value columns (y-data for the traces).

    Returns
    -------
    tuple of lists
        ``(x_out, y1_out, y2_out, …)`` with ``None`` sentinels at gaps.
    """
    dates_list = list(dates)
    val_lists = [list(v) for v in value_cols]
    x_out: list = []
    y_outs: list[list] = [[] for _ in val_lists]

    for i in range(len(dates_list)):
        if i > 0:
            gap = (dates_list[i] - dates_list[i - 1]).total_seconds() / 3600
            if gap > _MAX_CONTIGUOUS_GAP_HOURS and not _gap_spans_offline_window(
                dates_list[i - 1], dates_list[i]
            ):
                x_out.append(None)
                for y in y_outs:
                    y.append(None)
        x_out.append(dates_list[i])
        for j, vl in enumerate(val_lists):
            y_outs[j].append(vl[i])

    return (x_out, *y_outs)


def split_into_segments(dates, *value_cols):
    """Split time-series data into contiguous segments at large gaps.

    Unlike :func:`break_at_gaps` which inserts ``None`` sentinels,
    this function returns a list of separate segments.  This is needed
    for ``fill="tonexty"`` band traces where ``None`` gaps cause
    jagged / spiky fill artefacts in Plotly.

    Gaps that span a :data:`_KNOWN_OFFLINE_WINDOWS` entry are bridged
    (kept in the same segment) so the prediction band stays continuous
    across planned maintenance or outage periods.

    Parameters
    ----------
    dates : sequence of datetime
        Sorted timestamps (pd.Series, list, or array).
    *value_cols : sequence
        One or more aligned value columns (y-data for the traces).

    Returns
    -------
    list[tuple[list, ...]]
        Each element is ``(x_seg, y1_seg, y2_seg, …)`` for one
        contiguous segment.
    """
    dates_list = list(dates)
    val_lists = [list(v) for v in value_cols]

    if not dates_list:
        return []

    segments: list[tuple[list, ...]] = []
    seg_x: list = [dates_list[0]]
    seg_ys: list[list] = [[vl[0]] for vl in val_lists]

    for i in range(1, len(dates_list)):
        gap = (dates_list[i] - dates_list[i - 1]).total_seconds() / 3600
        if gap > _MAX_CONTIGUOUS_GAP_HOURS and not _gap_spans_offline_window(
            dates_list[i - 1], dates_list[i]
        ):
            segments.append((seg_x, *seg_ys))
            seg_x = []
            seg_ys = [[] for _ in val_lists]
        seg_x.append(dates_list[i])
        for j, vl in enumerate(val_lists):
            seg_ys[j].append(vl[i])

    # Don't forget the last segment
    if seg_x:
        segments.append((seg_x, *seg_ys))

    return segments
