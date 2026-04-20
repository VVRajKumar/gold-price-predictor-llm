"""
Chart utilities shared by the main dashboard (app.py) and the Prediction
Archive page.
"""

from __future__ import annotations

# Gaps wider than this many hours trigger a line break in Plotly traces.
# 6 hours comfortably covers any intra-day gap while catching the ~65-hour
# weekend closure (Friday 23:00 → Monday 09:00 IST).
_MAX_CONTIGUOUS_GAP_HOURS = 6


def break_at_gaps(dates, *value_cols):
    """Insert ``None`` gap-breakers so Plotly line traces don't draw
    misleading diagonals across weekend market closures.

    When only market-open hours are plotted, consecutive data points
    may jump from Friday evening to Monday morning.  Plotly draws a
    straight line between them unless there is a ``None`` in the data.
    This helper detects such jumps and inserts ``None`` values.

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
            if gap > _MAX_CONTIGUOUS_GAP_HOURS:
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
        if gap > _MAX_CONTIGUOUS_GAP_HOURS:
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
