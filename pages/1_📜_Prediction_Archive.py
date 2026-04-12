"""
Prediction Archive – full historical log of all predicted vs actual gold prices.

Unlike the main dashboard's accuracy scorecard (limited to the last 72 hours),
this page displays every evaluated prediction ever made, providing a long-term
performance record for showcasing the system at scale.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ── Fix imports (same pattern as app.py) ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_expected = ROOT / "src" / "__init__.py"
if "src" in sys.modules:
    _loaded = getattr(sys.modules["src"], "__file__", None)
    if _loaded is None or Path(_loaded).resolve() != _expected.resolve():
        for _k in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
            del sys.modules[_k]

try:
    from src.prediction_engine import PredictionEngine
    from src.time_utils import now_ist, parse_iso_to_ist, is_market_closed_ist
    from src.accuracy_tracker import compute_accuracy_score, _DIR_NEUTRAL_PCT
except (KeyError, ImportError, AttributeError):
    for _k in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
        del sys.modules[_k]
    from src.prediction_engine import PredictionEngine
    from src.time_utils import now_ist, parse_iso_to_ist, is_market_closed_ist
    from src.accuracy_tracker import compute_accuracy_score, _DIR_NEUTRAL_PCT

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediction Archive – Gold Price Predictor",
    page_icon="📜",
    layout="wide",
)

st.markdown("""
<style>
    .stApp {background-color: #0e1117;}

    /* ── Sidebar polish ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1320 0%, #151a2e 50%, #0f1320 100%);
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(99, 110, 140, 0.2);
        margin: 0.6rem 0;
    }
    section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 0.88rem;
        transition: background 0.2s;
    }
    section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover {
        background: rgba(255, 217, 61, 0.08);
    }
    section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] {
        background: rgba(255, 217, 61, 0.12);
        border-left: 3px solid #ffd93d;
    }
</style>
""", unsafe_allow_html=True)


# ── Engine (shared with main app via cache) ──────────────────────────
@st.cache_resource
def get_engine() -> PredictionEngine:
    return PredictionEngine()


engine = get_engine()
accuracy_tracker = engine.get_accuracy_tracker()

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """<div style="text-align:center;padding:14px 0 10px 0;">
        <div style="display:inline-flex;align-items:center;justify-content:center;
                    width:52px;height:52px;border-radius:14px;
                    background:linear-gradient(135deg,#ffd93d 0%,#f59e0b 100%);
                    box-shadow:0 4px 14px rgba(255,217,61,0.25);margin-bottom:8px;">
            <span style="font-size:1.6rem;line-height:1;">🥇</span>
        </div><br>
        <span style="font-size:1.25rem;font-weight:700;letter-spacing:0.3px;
                     color:#f0f0f5;">Gold Predictor</span><br>
        <span style="font-size:0.75rem;color:#7c8db5;letter-spacing:0.5px;">
            Prediction Archive</span>
        </div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Navigation ────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:0.7rem;font-weight:600;color:#7c8db5;text-transform:uppercase;'
        'letter-spacing:1.2px;margin-bottom:4px;">Navigation</div>',
        unsafe_allow_html=True,
    )
    st.page_link("app.py", label="🏠 Back to Home", icon=None)

    st.divider()
    st.markdown(
        f'<div style="text-align:center;font-size:0.7rem;color:#4a5568;">'
        f'v1.0 · Updated {now_ist().strftime("%H:%M")}</div>',
        unsafe_allow_html=True,
    )

# ── Header ───────────────────────────────────────────────────────────
st.title("📜 Prediction Archive")
st.caption(
    "Complete historical log of all predicted vs actual Indian gold prices (₹/10g). "
    "This archive is never trimmed — it grows with every evaluated prediction."
)

# ── Load archive data ────────────────────────────────────────────────
archive = accuracy_tracker.get_prediction_archive()

if not archive:
    st.info(
        "🗂️ **No archived predictions yet.** The archive is populated automatically "
        "as predictions are evaluated against actual market data. Generate a prediction "
        "from the main dashboard and check back after those hours have passed."
    )
    st.stop()

# ── Helper: compute directional accuracy from a sorted DataFrame ─────
def _compute_directional_accuracy(data: pd.DataFrame) -> float:
    """Compute directional accuracy from consecutive hourly predictions.

    Returns the percentage of hours where the predicted direction of
    price movement matched the actual direction.  When the actual price
    moved less than the neutral zone threshold, the market is considered
    flat and counted as correct regardless of prediction.
    """
    _NEUTRAL_PCT = _DIR_NEUTRAL_PCT
    if len(data) < 2 or "predicted" not in data.columns or "actual" not in data.columns:
        return 50.0
    sorted_df = data.sort_values("date").reset_index(drop=True)
    correct = 0
    total = 0
    for i in range(1, len(sorted_df)):
        prev_actual = sorted_df.loc[i - 1, "actual"]
        curr_predicted = sorted_df.loc[i, "predicted"]
        curr_actual = sorted_df.loc[i, "actual"]
        if prev_actual <= 0:
            continue
        # Only compare consecutive hours (gap ≤ 2 hours)
        gap = (sorted_df.loc[i, "date"] - sorted_df.loc[i - 1, "date"]).total_seconds()
        if gap > 2 * 3600:
            continue
        actual_move_pct = abs(curr_actual - prev_actual) / prev_actual * 100
        total += 1
        if actual_move_pct < _NEUTRAL_PCT:
            correct += 1  # flat market – auto-correct
        else:
            pred_dir = 1 if curr_predicted >= prev_actual else -1
            actual_dir = 1 if curr_actual >= prev_actual else -1
            if pred_dir == actual_dir:
                correct += 1
    return round(correct / total * 100, 1) if total > 0 else 50.0


# Convert to DataFrame
df = pd.DataFrame(archive)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df = df.sort_values("date")

# Deduplicate: for same hour, keep the entry from the most recently generated plan
if "plan_generated_at" in df.columns:
    df["_gen_at"] = pd.to_datetime(df["plan_generated_at"], errors="coerce")
    df = df.sort_values(["date", "_gen_at"], ascending=[True, False])
    df = df.drop_duplicates(subset="date", keep="first")
    df = df.drop(columns=["_gen_at"])
    df = df.sort_values("date")

# Exclude market-closed hours (weekends + Monday pre-market) from all
# metrics, tables, and summaries — these predictions are trivially
# correct (predicted == actual) and would inflate accuracy numbers.
df = df[~df["date"].apply(lambda dt: is_market_closed_ist(dt.to_pydatetime()))].copy()

if df.empty:
    st.info("No market-hours predictions available yet.")
    st.stop()

# ── Summary Metrics ──────────────────────────────────────────────────
st.divider()
total_hours = len(df)
date_range_start = df["date"].min()
date_range_end = df["date"].max()
total_days = (date_range_end - date_range_start).days + 1
unique_dates = df["date"].dt.date.nunique()

avg_mape = df["pct_error"].mean() if "pct_error" in df.columns else 0
band_hit = (df["within_band"].sum() / len(df) * 100) if "within_band" in df.columns else 0
avg_error = df["error"].abs().mean() if "error" in df.columns else 0
dir_accuracy = _compute_directional_accuracy(df)
overall_score = compute_accuracy_score(avg_mape, band_hit, dir_accuracy)

m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
with m1:
    st.metric("📊 Total Hours", f"{total_hours}")
with m2:
    st.metric("📅 Days Covered", f"{unique_dates}")
with m3:
    mape_icon = "🟢" if avg_mape < 2 else ("🟡" if avg_mape < 5 else "🔴")
    st.metric(f"{mape_icon} Avg MAPE", f"{avg_mape:.2f}%")
with m4:
    st.metric("📏 Avg MAE", f"₹{avg_error:,.2f}")
with m5:
    hit_icon = "🟢" if band_hit >= 80 else ("🟡" if band_hit >= 60 else "🔴")
    st.metric(f"{hit_icon} Band Hit Rate", f"{band_hit:.1f}%")
with m6:
    dir_icon = "🟢" if dir_accuracy >= 80 else ("🟡" if dir_accuracy >= 60 else "🔴")
    st.metric(f"{dir_icon} Direction", f"{dir_accuracy:.1f}%")
with m7:
    score_icon = "🟢" if overall_score >= 75 else ("🟡" if overall_score >= 50 else "🔴")
    st.metric(f"{score_icon} Accuracy Score", f"{overall_score:.1f}")
with m8:
    st.metric("🗓️ Date Range", f"{date_range_start.strftime('%b %d')} – {date_range_end.strftime('%b %d')}")

st.caption(
    "**MAPE** = Mean Absolute Percentage Error · "
    "**MAE** = Mean Absolute Error in ₹ · "
    "**Band Hit** = % of hours within predicted range · "
    "**Direction** = % of hours with correct price direction · "
    "**Accuracy Score** = Composite (0–100) combining MAPE 35%, Band Hit 30%, Direction 35%"
)

# ── Filters ──────────────────────────────────────────────────────────
st.divider()

col_filter1, col_filter2 = st.columns(2)
with col_filter1:
    date_options = ["All Time", "Last 24 Hours", "Last 3 Days", "Last 7 Days", "Last 14 Days", "Last 30 Days"]
    time_filter = st.selectbox("📅 Time Range", date_options, index=0)
with col_filter2:
    show_band_misses_only = st.checkbox("Show only band misses (outside predicted range)", value=False)

# Apply filters
filtered_df = df.copy()
now = now_ist().replace(tzinfo=None)
if time_filter == "Last 24 Hours":
    filtered_df = filtered_df[filtered_df["date"] >= now - timedelta(hours=24)]
elif time_filter == "Last 3 Days":
    filtered_df = filtered_df[filtered_df["date"] >= now - timedelta(days=3)]
elif time_filter == "Last 7 Days":
    filtered_df = filtered_df[filtered_df["date"] >= now - timedelta(days=7)]
elif time_filter == "Last 14 Days":
    filtered_df = filtered_df[filtered_df["date"] >= now - timedelta(days=14)]
elif time_filter == "Last 30 Days":
    filtered_df = filtered_df[filtered_df["date"] >= now - timedelta(days=30)]

if show_band_misses_only and "within_band" in filtered_df.columns:
    filtered_df = filtered_df[~filtered_df["within_band"]]

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# ── Predicted vs Actual Chart (full archive) ─────────────────────────
_heading_map = {
    "All Time": "📈 Predicted vs Actual – Full History",
    "Last 24 Hours": "📈 Predicted vs Actual – Last 24 Hours",
    "Last 3 Days": "📈 Predicted vs Actual – Last 3 Days",
    "Last 7 Days": "📈 Predicted vs Actual – Last 7 Days",
    "Last 14 Days": "📈 Predicted vs Actual – Last 14 Days",
    "Last 30 Days": "📈 Predicted vs Actual – Last 30 Days",
}
st.subheader(_heading_map.get(time_filter, "📈 Predicted vs Actual – Full History"))

# Forward-fill NaN gaps so lines stay connected
for _col in ("predicted", "actual", "high_range", "low_range"):
    if _col in filtered_df.columns:
        filtered_df[_col] = filtered_df[_col].ffill()

fig = go.Figure()

# Prediction band
if "high_range" in filtered_df.columns and "low_range" in filtered_df.columns:
    fig.add_trace(go.Scatter(
        x=filtered_df["date"], y=filtered_df["high_range"],
        mode="lines", name="Upper Band",
        line=dict(width=0), showlegend=False,
        connectgaps=True,
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df["date"], y=filtered_df["low_range"],
        mode="lines", name="Prediction Range",
        fill="tonexty", fillcolor="rgba(0,212,170,0.10)",
        line=dict(width=0),
        connectgaps=True,
    ))

# Predicted line — solid green
fig.add_trace(go.Scatter(
    x=filtered_df["date"], y=filtered_df["predicted"],
    mode="lines+markers", name="Predicted",
    line=dict(color="#00d4aa", width=2),
    marker=dict(size=5, symbol="diamond"),
    connectgaps=True,
))

# Actual line — solid yellow
fig.add_trace(go.Scatter(
    x=filtered_df["date"], y=filtered_df["actual"],
    mode="lines+markers", name="Actual",
    line=dict(color="#ffd93d", width=2),
    marker=dict(size=5),
    connectgaps=True,
))

# Highlight band misses
if "within_band" in filtered_df.columns:
    outside = filtered_df[~filtered_df["within_band"]]
    if not outside.empty:
        fig.add_trace(go.Scatter(
            x=outside["date"], y=outside["actual"],
            mode="markers", name="Outside Band ✗",
            marker=dict(size=10, color="#ff6b6b", symbol="x"),
        ))

# Smart y-axis scaling based on actual prices
_actual_vals = filtered_df["actual"].dropna()
if not _actual_vals.empty:
    _y_center = _actual_vals.median()
    _y_iqr = _actual_vals.quantile(0.75) - _actual_vals.quantile(0.25)
    _y_spread = max(_y_iqr * 3, _y_center * 0.05)
    _y_range = [max(0, (_y_center - _y_spread) * 0.98), (_y_center + _y_spread) * 1.02]
else:
    _y_range = None

fig.update_layout(
    template="plotly_dark",
    height=550,
    yaxis_title="Price (₹/10g)",
    xaxis_title="Time (IST)",
    yaxis=dict(range=_y_range) if _y_range else {},
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── Daily Summary Table ──────────────────────────────────────────────
st.subheader("📅 Daily Performance Summary")

daily_summary = filtered_df.copy()
daily_summary["day"] = daily_summary["date"].dt.date

# Compute directional accuracy per day
daily_dir_acc: dict = {}
for day_val, day_group in daily_summary.groupby("day"):
    daily_dir_acc[day_val] = _compute_directional_accuracy(day_group)

daily_grouped = daily_summary.groupby("day").agg(
    hours_evaluated=("date", "count"),
    avg_mape=("pct_error", "mean"),
    avg_mae=("error", lambda x: x.abs().mean()),
    band_hits=("within_band", "sum"),
    total=("within_band", "count"),
).reset_index()
daily_grouped["band_hit_rate"] = (daily_grouped["band_hits"] / daily_grouped["total"] * 100).round(1)
daily_grouped["direction_accuracy"] = daily_grouped["day"].map(daily_dir_acc)
daily_grouped["accuracy_score"] = daily_grouped.apply(
    lambda r: compute_accuracy_score(r["avg_mape"], r["band_hit_rate"], r["direction_accuracy"]),
    axis=1,
)
daily_grouped = daily_grouped.sort_values("day", ascending=False)

display_daily = daily_grouped[["day", "hours_evaluated", "avg_mape", "avg_mae", "band_hit_rate", "direction_accuracy", "accuracy_score"]].rename(columns={
    "day": "Date",
    "hours_evaluated": "Hours",
    "avg_mape": "Avg MAPE (%)",
    "avg_mae": "Avg MAE (₹)",
    "band_hit_rate": "Band Hit Rate (%)",
    "direction_accuracy": "Direction (%)",
    "accuracy_score": "Score",
})

st.dataframe(
    display_daily.style.format({
        "Avg MAPE (%)": "{:.2f}",
        "Avg MAE (₹)": "₹{:,.2f}",
        "Band Hit Rate (%)": "{:.1f}",
        "Direction (%)": "{:.1f}",
        "Score": "{:.1f}",
    }),
    use_container_width=True,
    hide_index=True,
)

# ── Full Hourly Detail Table ─────────────────────────────────────────
with st.expander("📋 Full Hourly Detail", expanded=False):
    detail_cols = ["date", "predicted", "actual", "low_range", "high_range",
                   "error", "pct_error", "within_band"]
    available_cols = [c for c in detail_cols if c in filtered_df.columns]
    detail_df = filtered_df[available_cols].copy()
    detail_df = detail_df.sort_values("date", ascending=False)
    detail_df = detail_df.rename(columns={
        "date": "Hour (IST)",
        "predicted": "Predicted (₹)",
        "actual": "Actual (₹)",
        "low_range": "Low Band (₹)",
        "high_range": "High Band (₹)",
        "error": "Error (₹)",
        "pct_error": "Error (%)",
        "within_band": "In Range",
    })
    st.dataframe(
        detail_df.style.format({
            "Predicted (₹)": "₹{:,.2f}",
            "Actual (₹)": "₹{:,.2f}",
            "Low Band (₹)": "₹{:,.2f}",
            "High Band (₹)": "₹{:,.2f}",
            "Error (₹)": "{:+,.2f}",
            "Error (%)": "{:.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
        height=500,
    )

# ── Footer ───────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Archive contains **{total_hours}** evaluated hourly predictions across "
    f"**{unique_dates}** days. Data is auto-updated when the accuracy tracker "
    f"evaluates new predictions (every 2 hours)."
)
st.caption(
    "⚠️ **Disclaimer:** This is an AI-powered prediction system for educational/research "
    "purposes only. It is NOT financial advice."
)
