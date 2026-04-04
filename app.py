"""
Streamlit Dashboard – live-updating gold price prediction dashboard.
Run with:  streamlit run app.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Fix imports ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Streamlit Cloud: repos live at /mount/src/<name>/ which can shadow our 'src' pkg
_expected = ROOT / "src" / "__init__.py"
if "src" in sys.modules:
    _loaded = getattr(sys.modules["src"], "__file__", None)
    if _loaded is None or Path(_loaded).resolve() != _expected.resolve():
        for _k in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
            del sys.modules[_k]

from src.prediction_engine import PredictionEngine
from src.data_fetchers.market_data import MarketDataFetcher
from src.time_utils import now_ist, parse_iso_to_ist

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Gold Price Predictor – Agentic AI",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {background-color: #0e1117;}
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 20px; margin: 8px 0;
        border: 1px solid #2d3748;
    }
    .bullish {color: #00d4aa; font-weight: bold;}
    .bearish {color: #ff6b6b; font-weight: bold;}
    .neutral {color: #ffd93d; font-weight: bold;}
    .agent-card {
        background: #1a1a2e; border-radius: 10px; padding: 15px;
        margin: 5px 0; border-left: 4px solid #4a5568;
    }
    .agent-card.bullish-border {border-left-color: #00d4aa;}
    .agent-card.bearish-border {border-left-color: #ff6b6b;}
    .agent-card.neutral-border {border-left-color: #ffd93d;}
</style>
""", unsafe_allow_html=True)


# ── Session state & engine ───────────────────────────────────────────
@st.cache_resource
def get_engine() -> PredictionEngine:
    return PredictionEngine()


@st.cache_resource
def get_market() -> MarketDataFetcher:
    return MarketDataFetcher()


engine = get_engine()
market = get_market()
accuracy_tracker = engine.get_accuracy_tracker()

# Streamlit can sometimes keep a stale cached engine object across hot reloads.
# If weekly API is missing or market fetcher lacks INR helpers, rebuild once.
if not hasattr(engine, "ensure_weekly_prediction") or not hasattr(market, "convert_usd_to_inr"):
    st.cache_resource.clear()
    engine = get_engine()
    market = get_market()
    accuracy_tracker = engine.get_accuracy_tracker()


def outlook_color(outlook: str) -> str:
    return {"bullish": "#00d4aa", "bearish": "#ff6b6b"}.get(outlook, "#ffd93d")


def outlook_emoji(outlook: str) -> str:
    return {"bullish": "🟢", "bearish": "🔴"}.get(outlook, "🟡")


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🥇 Gold Predictor")
    st.caption("Multi-Agent AI System")

    view_mode = "Live Prediction"

    st.divider()

    if st.button("🔄 Generate New Prediction", width="stretch", type="primary"):
        with st.spinner("Running 8 specialist agents … this takes 1-2 minutes"):
            plan = engine.generate()
        st.success("Prediction updated!")
        st.rerun()

    st.divider()
    st.markdown("### Agent Roster")
    agent_names = [
        "🌍 Geopolitics",
        "📈 Trend Analysis",
        "💰 ETF Flows",
        "🏦 Macro Economics",
        "🛢️ Oil & Energy",
        "😨 Market Sentiment",
        "📊 Technical Analysis",
        "📜 Historical Patterns",
    ]
    for a in agent_names:
        st.markdown(f"- {a}")

    st.divider()
    st.caption(f"v1.0 · Updated {now_ist().strftime('%H:%M')}")


# ════════════════════════════════════════════════════════════════════
st.title("🥇 Indian Gold Price Prediction System (₹/10g)")
st.caption("ML Ensemble (XGBoost + LightGBM + Ridge) · LLM for narrative only · SHAP explainability")

if view_mode == "Weekly Archive":
    st.subheader("🗂️ Weekly Prediction Archive")
    weekly_archive = engine.get_weekly_archive()

    if not weekly_archive:
        st.info("No completed weekly predictions archived yet.")
        st.stop()

    rows = []
    for item in weekly_archive:
        plan_dict = item.get("plan", {})
        rows.append({
            "Week": item.get("week_id", ""),
            "Generated": plan_dict.get("generated_at", "")[:16],
            "Outlook": str(plan_dict.get("overall_outlook", "")).upper(),
            "Confidence": f"{float(plan_dict.get('overall_confidence', 0)):.0%}",
            "Anchor Price": f"₹{float(plan_dict.get('current_price', 0)):,.2f}",
        })

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    for item in weekly_archive:
        plan_dict = item.get("plan", {})
        week_id = item.get("week_id", "Unknown Week")
        outlook = str(plan_dict.get("overall_outlook", "neutral")).upper()
        with st.expander(f"{week_id} — {outlook}"):
            st.markdown(plan_dict.get("executive_summary", "No summary available."))

            daily = plan_dict.get("daily_predictions", [])
            if daily:
                adf = pd.DataFrame(daily)
                if "date" in adf.columns:
                    adf["date"] = pd.to_datetime(adf["date"])
                st.dataframe(
                    adf[["date", "predicted_price", "low_range", "high_range", "confidence", "key_driver"]],
                    width="stretch",
                    hide_index=True,
                )
    st.stop()

# MAIN DASHBOARD
# ════════════════════════════════════════════════════════════════════
if hasattr(engine, "ensure_weekly_prediction"):
    plan = engine.ensure_weekly_prediction()
else:
    plan = engine.get_current_plan()

# Always keep the live OHLC chart visible.
st.subheader("🕯️ Live Indian Gold OHLC (10D) – INR/10g")
gold_df = market.fetch_ticker("GC=F", period_days=10, interval="1h")
if not gold_df.empty:
    if isinstance(gold_df.index, pd.DatetimeIndex):
        latest_ts = gold_df.index.max()
        cutoff_ts = latest_ts - pd.Timedelta(days=10)
        gold_df = gold_df[gold_df.index >= cutoff_ts]

    # Convert USD/oz to INR/10g using time-aligned daily FX rates
    gold_df = market.convert_usd_to_inr(gold_df, period_days=10)

    range_start = gold_df.index.min()
    range_end = gold_df.index.max()

    fig_ohlc = go.Figure()
    fig_ohlc.add_trace(go.Candlestick(
        x=gold_df.index,
        open=gold_df["Open"],
        high=gold_df["High"],
        low=gold_df["Low"],
        close=gold_df["Close"],
        name="Gold (INR/10g)",
    ))
    fig_ohlc.update_layout(
        title="Indian Gold – Last 10 Days (₹ per 10 grams)",
        template="plotly_dark",
        height=450,
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            fixedrange=True,
            range=[range_start, range_end + pd.Timedelta(hours=2)],
            rangebreaks=[dict(bounds=["sat", "mon"])],
        ),
        yaxis=dict(fixedrange=True, title="Price (₹/10g)"),
    )
    st.plotly_chart(
        fig_ohlc,
        width="stretch",
        config={
            "scrollZoom": False,
            "doubleClick": False,
            "displayModeBar": False,
        },
    )
    if isinstance(range_start, pd.Timestamp) and isinstance(range_end, pd.Timestamp):
        st.caption(
            f"Plotted range: {range_start.strftime('%Y-%m-%d')} to {range_end.strftime('%Y-%m-%d')} "
            f"({len(gold_df)} hourly sessions in last 10 calendar days)"
        )
else:
    st.warning("Live gold OHLC data is temporarily unavailable.")

if plan is None:
    st.info("No prediction yet. Click **Generate New Prediction** in the sidebar to start.")
    st.stop()

# ── Top Metrics Row ──────────────────────────────────────────────────
import math as _math
_live_fx = market.get_usdinr_rate()
_valid_price = _math.isfinite(plan.current_price) and plan.current_price > 0
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Current Price", f"₹{plan.current_price:,.2f}" if _valid_price else "N/A")
with c2:
    color = outlook_color(plan.overall_outlook)
    st.metric("Outlook", f"{outlook_emoji(plan.overall_outlook)} {plan.overall_outlook.upper()}")
with c3:
    st.metric("Confidence", f"{plan.overall_confidence:.0%}")
with c4:
    if plan.daily_predictions and _valid_price:
        horizon_target = plan.daily_predictions[-1]
        delta = horizon_target.predicted_price - plan.current_price
        st.metric("24-Hour Target", f"₹{horizon_target.predicted_price:,.2f}", f"₹{delta:+,.2f}")
    else:
        st.metric("24-Hour Target", "N/A")
with c5:
    st.metric("USD/INR Rate", f"₹{_live_fx:.2f}" if _math.isfinite(_live_fx) else "N/A")
with c6:
    try:
        st.metric("Last Updated", parse_iso_to_ist(plan.generated_at).strftime("%H:%M %b %d"))
    except Exception:
        st.metric("Last Updated", plan.generated_at)

# ── Quick Accuracy Badge (visible to all visitors) ───────────────────
_quick_agg = accuracy_tracker.get_aggregate_stats()
if _quick_agg and _quick_agg["total_predictions_evaluated"] > 0:
    _mape = _quick_agg["overall_mape"]
    _hit = _quick_agg["overall_band_hit_rate"]
    _dir = _quick_agg["avg_directional_accuracy"]
    _n = _quick_agg["total_predictions_evaluated"]
    _mape_icon = "🟢" if _mape < 2 else ("🟡" if _mape < 5 else "🔴")
    _hit_icon = "🟢" if _hit >= 70 else ("🟡" if _hit >= 50 else "🔴")
    _dir_icon = "🟢" if _dir >= 60 else ("🟡" if _dir >= 50 else "🔴")
    st.markdown(
        f"""<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
        border-radius:12px;padding:14px 20px;margin:10px 0;
        border:1px solid #2d3748;display:flex;justify-content:space-around;
        flex-wrap:wrap;gap:10px;text-align:center;">
        <span>{_mape_icon} <b>MAPE</b> {_mape:.1f}%</span>
        <span>{_hit_icon} <b>Band Hit</b> {_hit:.0f}%</span>
        <span>{_dir_icon} <b>Direction</b> {_dir:.0f}%</span>
        <span>📊 <b>{_n}</b> hours scored</span>
        </div>""",
        unsafe_allow_html=True,
    )

st.divider()

# ── Executive Summary ────────────────────────────────────────────────
with st.expander("📋 Executive Summary", expanded=True):
    _summary = plan.executive_summary
    # Guard: if LLM returned raw JSON instead of prose, show it cleanly
    if _summary and (_summary.strip().startswith("{") or _summary.strip().startswith("[")):
        st.code(_summary, language="json")
    else:
        st.markdown(_summary or "No executive summary available.")

# ── Hourly Prediction Chart ──────────────────────────────────────────
st.subheader("🕐 Next 24-Hour Price Prediction")

if plan.daily_predictions:
    pred_df = pd.DataFrame([dp.model_dump() for dp in plan.daily_predictions])
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    # Reuse the already-converted INR data from the OHLC chart above
    # (avoids a second fetch+convert which can double-convert on Cloud)
    gold_recent = gold_df.copy() if not gold_df.empty else pd.DataFrame()

    fig = go.Figure()

    # Historical prices (already in INR/10g from the OHLC section)
    if not gold_recent.empty:
        close_series = pd.to_numeric(gold_recent["Close"], errors="coerce").dropna()
        # Reindex hourly so missing bars do not break the timeline.
        if not close_series.empty:
            chart_end = close_series.index.max()
            pred_start = pred_df["date"].min() if not pred_df.empty else chart_end
            if pd.notna(pred_start):
                chart_end = max(chart_end, pred_start - pd.Timedelta(hours=1))

            daily_idx = pd.date_range(
                start=close_series.index.min(),
                end=chart_end,
                freq="h",
            )
            close_series = close_series.reindex(daily_idx).ffill()

        fig.add_trace(go.Scatter(
            x=close_series.index, y=close_series.values,
            mode="lines", name="Actual",
            line=dict(color="#ffd93d", width=2),
        ))

    # Prediction band
    fig.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df["high_range"],
        mode="lines", name="Upper Band",
        line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df["low_range"],
        mode="lines", name="Prediction Range",
        fill="tonexty", fillcolor="rgba(0,212,170,0.15)",
        line=dict(width=0),
    ))

    # Prediction line
    fig.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df["predicted_price"],
        mode="lines+markers", name="Predicted",
        line=dict(color="#00d4aa", width=3),
        marker=dict(size=10, symbol="circle"),
    ))

    fig.update_layout(
        template="plotly_dark", height=500,
        yaxis_title="Price (₹/10g)",
        xaxis_title="Time",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    # Daily breakdown table
    st.dataframe(
        pred_df[["date", "predicted_price", "low_range", "high_range", "confidence", "key_driver"]]
        .rename(columns={
            "date": "Date",
            "predicted_price": "Predicted (₹)",
            "low_range": "Low (₹)",
            "high_range": "High (₹)",
            "confidence": "Confidence",
            "key_driver": "Key Driver",
        })
        .style.format({
            "Predicted (₹)": "₹{:,.2f}",
            "Low (₹)": "₹{:,.2f}",
            "High (₹)": "₹{:,.2f}",
            "Confidence": "{:.0%}",
        }),
        width="stretch",
        hide_index=True,
    )
else:
    st.warning("No daily predictions available in this plan.")

st.divider()

# ── ML Model Transparency ────────────────────────────────────────────
st.subheader("🧠 How the Prediction is Made")
st.caption(
    "Your gold price forecast is powered by **3 ML models working together** "
    "(not AI text generation). The LLM only writes the summary text — "
    "every price number comes from trained mathematical models."
)

_ml_info = plan.agent_reports.get("_ml_ensemble", {})
_shap = _ml_info.get("shap") if isinstance(_ml_info, dict) else None

if _shap:
    # ── Row 1: How It Works (3 cards) ────────────────────────────────
    st.markdown("#### How It Works")
    hw1, hw2, hw3 = st.columns(3)
    with hw1:
        st.markdown("""
        <div class="metric-card">
        <h4>📊 Step 1: Data Collection</h4>
        <p>We gather <b>90 days of hourly gold prices</b> plus live signals
        from 8 specialist agents (sentiment, geopolitics, macro, technical,
        ETF flows, oil, seasonal patterns, trend).</p>
        </div>
        """, unsafe_allow_html=True)
    with hw2:
        st.markdown(f"""
        <div class="metric-card">
        <h4>🤖 Step 2: Triple Model Prediction</h4>
        <p>Three independent ML models each analyze <b>{_shap.get('total_features', 24)} features</b>
        and make their own forecast. A 4th model combines them into
        one final prediction — reducing any single model's mistakes.</p>
        </div>
        """, unsafe_allow_html=True)
    with hw3:
        st.markdown("""
        <div class="metric-card">
        <h4>🎯 Step 3: Confidence Bands</h4>
        <p>Instead of one number, we give a <b>range</b> — there's an 80%
        probability the real price lands inside the Low–High band.
        The model also learns from its past mistakes to self-correct.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Row 2: What Drives the Prediction ────────────────────────────
    st.markdown("#### What Drives the Prediction")
    st.caption(
        "This chart shows which inputs matter most to the model. "
        "Taller bars = stronger influence on the predicted price."
    )

    _feat_imp = _shap.get("feature_importance", [])
    if _feat_imp:
        shap_df = pd.DataFrame(_feat_imp[:12])
        # Color code: price features green, agent signals cyan, time features gray
        _price_feats = {"Price 1 Hour Ago", "Price 2 Hours Ago", "Price 3 Hours Ago",
                        "Price 6 Hours Ago", "Price 12 Hours Ago", "Price 24 Hours Ago",
                        "6-Hour Average Price", "12-Hour Average Price", "24-Hour Average Price",
                        "1-Hour Price Change %", "6-Hour Price Change %", "12-Hour Volatility"}
        _agent_feats = {"Market Sentiment", "Geopolitical Risk", "Macro-Economic Outlook",
                        "Technical Analysis Signal", "ETF Fund Flows", "Oil & Energy Impact",
                        "Seasonal Pattern", "Trend Strength"}
        colors = []
        for _, row in shap_df.iterrows():
            if row["feature"] in _price_feats:
                colors.append("#00d4aa")   # green - price features
            elif row["feature"] in _agent_feats:
                colors.append("#4ecdc4")   # teal - agent signals
            else:
                colors.append("#636e72")   # gray - time features

        fig_shap = go.Figure(go.Bar(
            x=shap_df["importance"],
            y=shap_df["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Influence Score: %{x:.1f}<extra></extra>",
        ))
        fig_shap.update_layout(
            template="plotly_dark",
            height=420,
            yaxis=dict(autorange="reversed", title=""),
            xaxis=dict(title="Influence Score (higher = more impact)"),
            showlegend=False,
            margin=dict(l=10, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # Legend
        st.markdown(
            '<span style="color:#00d4aa">●</span> Price History &nbsp;&nbsp; '
            '<span style="color:#4ecdc4">●</span> Agent Intelligence Signals &nbsp;&nbsp; '
            '<span style="color:#636e72">●</span> Time Patterns',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Row 3: Hour-by-Hour Drivers ──────────────────────────────────
    _hourly_shap = _shap.get("hourly_drivers", [])
    if _hourly_shap:
        st.markdown("#### Hour-by-Hour: What Moves Each Prediction")
        st.caption(
            "For each upcoming hour, these are the top 3 factors pushing the price "
            "up (↑ green) or down (↓ red)."
        )
        # Show in a 3-column grid
        cols_per_row = 3
        for row_start in range(0, min(12, len(_hourly_shap)), cols_per_row):
            cols = st.columns(cols_per_row)
            for ci, hd in enumerate(_hourly_shap[row_start:row_start + cols_per_row]):
                with cols[ci]:
                    drivers_html = ""
                    for d in hd.get("drivers", []):
                        if isinstance(d, dict):
                            color = "#00d4aa" if d["value"] > 0 else "#e74c3c"
                            arrow = d["direction"]
                            drivers_html += (
                                f'<div style="margin:4px 0;">'
                                f'<span style="color:{color};font-weight:bold">{arrow}</span> '
                                f'{d["name"]} '
                                f'<span style="color:{color}">({d["value"]:+.1f})</span>'
                                f'</div>'
                            )
                        else:
                            # fallback for old string format
                            drivers_html += f'<div style="margin:4px 0;">{d}</div>'
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<h4 style="margin-bottom:8px">Hour {hd["hour"]}</h4>'
                        f'{drivers_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
else:
    st.info("🔄 ML model explainability will appear after the first prediction cycle completes.")

st.divider()

# ── Agent Reports (Intelligence Signals) ─────────────────────────────
st.subheader("🤖 Agent Intelligence Reports")
st.caption(
    "Agents gather market intelligence and extract structured signals "
    "that are fed as **numeric features** into the ML ensemble. "
    "They do NOT generate price predictions."
)

if plan.agent_reports:
    # Agent confidence comparison chart
    agent_data = []
    for name, report in plan.agent_reports.items():
        if name.startswith("_"):
            continue  # skip internal entries like _ml_ensemble
        if isinstance(report, dict) and "outlook" in report:
            agent_data.append({
                "Agent": name.replace("_", " ").title(),
                "Confidence": report.get("confidence", 0),
                "Impact": report.get("impact_score", 0),
                "Bias": report.get("prediction_bias", 0),
                "Outlook": report.get("outlook", "neutral"),
            })

    if agent_data:
        adf = pd.DataFrame(agent_data)

        col1, col2 = st.columns(2)

        with col1:
            fig_conf = px.bar(
                adf, x="Agent", y="Confidence", color="Outlook",
                color_discrete_map={"bullish": "#00d4aa", "bearish": "#ff6b6b", "neutral": "#ffd93d"},
                title="Agent Confidence Levels",
            )
            fig_conf.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_conf, width="stretch")

        with col2:
            fig_bias = px.bar(
                adf, x="Agent", y="Bias", color="Outlook",
                color_discrete_map={"bullish": "#00d4aa", "bearish": "#ff6b6b", "neutral": "#ffd93d"},
                title="Agent Prediction Bias (-1 Bearish → +1 Bullish)",
            )
            fig_bias.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_bias, width="stretch")

    # Individual agent cards
    for name, report in plan.agent_reports.items():
        if name.startswith("_"):
            continue
        if not isinstance(report, dict) or "outlook" not in report:
            continue

        outlook = report.get("outlook", "neutral")
        emoji = outlook_emoji(outlook)
        border_class = f"{outlook}-border"

        with st.expander(f"{emoji} {name.replace('_', ' ').title()} — {outlook.upper()}"):
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Confidence", f"{report.get('confidence', 0):.0%}")
            mc2.metric("Impact Score", f"{report.get('impact_score', 0):.0%}")
            mc3.metric("Bias", f"{report.get('prediction_bias', 0):+.2f}")

            st.markdown(report.get("summary", "No summary available."))

            factors = report.get("key_factors", [])
            if factors:
                st.markdown("**Key Factors:** " + " · ".join(factors))

st.divider()

# ── Bull / Bear Cases ────────────────────────────────────────────────
st.subheader("⚖️ Bull vs Bear Case")
b1, b2 = st.columns(2)
with b1:
    st.markdown(f"""
    <div style="background:#0a2e1a; border-radius:10px; padding:20px; border-left:4px solid #00d4aa;">
    <h4 style="color:#00d4aa;">🐂 Bull Case</h4>
    <p>{plan.bull_case or 'Not available'}</p>
    </div>
    """, unsafe_allow_html=True)

with b2:
    st.markdown(f"""
    <div style="background:#2e0a0a; border-radius:10px; padding:20px; border-left:4px solid #ff6b6b;">
    <h4 style="color:#ff6b6b;">🐻 Bear Case</h4>
    <p>{plan.bear_case or 'Not available'}</p>
    </div>
    """, unsafe_allow_html=True)

# ── Risk Factors ─────────────────────────────────────────────────────
if plan.risk_factors:
    st.subheader("⚠️ Risk Factors")
    for i, risk in enumerate(plan.risk_factors, 1):
        st.markdown(f"**{i}.** {risk}")

is_streamlit_cloud = str(ROOT).replace("\\", "/").startswith("/mount/src/")
st.divider()

# ════════════════════════════════════════════════════════════════════
# PREDICTION ACCURACY SCORECARD
# ════════════════════════════════════════════════════════════════════
history = engine.get_plan_history()
st.subheader("🎯 Prediction Accuracy Scorecard")
# Auto-evaluate all stored plans against latest market data
stored_plans = accuracy_tracker.get_stored_plans()

# Ensure all history plans are stored for tracking
import json as _json
if history:
    for h in history:
        plan_dict = _json.loads(h.model_dump_json())
        accuracy_tracker.store_plan(plan_dict)

# Also store the current plan
accuracy_tracker.store_plan(_json.loads(plan.model_dump_json()))

# Re-evaluate all stored plans (picks up any new hour closes)
accuracy_tracker.refresh_all()

all_evals = accuracy_tracker.get_all_evaluations()
agg_stats = accuracy_tracker.get_aggregate_stats()

if agg_stats and agg_stats["total_predictions_evaluated"] > 0:
    # Show when accuracy was last auto-updated
    latest_eval = accuracy_tracker.get_latest_evaluation()
    if latest_eval:
        eval_time = latest_eval.get("evaluated_at", "")
        try:
            eval_dt = parse_iso_to_ist(eval_time)
            st.caption(f"🔄 Auto-updated: {eval_dt.strftime('%H:%M %b %d, %Y')} "
                       f"· {latest_eval['days_evaluated']}/{latest_eval['days_total']} hours scored "
                       f"· Background check every 1h")
        except Exception:
            pass

if agg_stats and agg_stats["total_predictions_evaluated"] > 0:
    # ── Aggregate Metrics Row ────────────────────────────────────
    st.markdown("#### Overall Accuracy (All Past Predictions)")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        mape = agg_stats["overall_mape"]
        mape_color = "🟢" if mape < 2 else ("🟡" if mape < 5 else "🔴")
        st.metric(f"{mape_color} MAPE", f"{mape:.1f}%")
    with m2:
        st.metric("📏 MAE", f"₹{agg_stats['overall_mae']:,.2f}")
    with m3:
        hit = agg_stats["overall_band_hit_rate"]
        hit_color = "🟢" if hit >= 70 else ("🟡" if hit >= 50 else "🔴")
        st.metric(f"{hit_color} Band Hit Rate", f"{hit:.0f}%")
    with m4:
        da = agg_stats["avg_directional_accuracy"]
        da_color = "🟢" if da >= 60 else ("🟡" if da >= 50 else "🔴")
        st.metric(f"{da_color} Direction Accuracy", f"{da:.0f}%")
    with m5:
        st.metric("📊 Hours Evaluated", f"{agg_stats['total_predictions_evaluated']}")

    st.caption(
        "**MAPE** = Mean Absolute Percentage Error (lower is better) · "
        "**MAE** = Mean Absolute Error in ₹ · "
        "**Band Hit Rate** = % of hours actual price fell within predicted range · "
        "**Direction** = % of hours predicted direction matched actual"
    )

    # ── Predicted vs Actual Chart ────────────────────────────────
    all_daily = []
    for ev in all_evals:
        for d in ev.get("daily_results", []):
            all_daily.append(d)

    if all_daily:
        acc_df = pd.DataFrame(all_daily)
        acc_df["date"] = pd.to_datetime(acc_df["date"])
        acc_df = acc_df.sort_values("date").drop_duplicates(subset="date", keep="last")

        fig_acc = go.Figure()

        # Prediction band
        fig_acc.add_trace(go.Scatter(
                x=acc_df["date"], y=acc_df["high_range"],
                mode="lines", name="Upper Band",
                line=dict(width=0), showlegend=False,
        ))
        fig_acc.add_trace(go.Scatter(
                x=acc_df["date"], y=acc_df["low_range"],
                mode="lines", name="Prediction Range",
                fill="tonexty", fillcolor="rgba(0,212,170,0.12)",
                line=dict(width=0),
        ))

        # Predicted line
        fig_acc.add_trace(go.Scatter(
                x=acc_df["date"], y=acc_df["predicted"],
                mode="lines+markers", name="Predicted",
                line=dict(color="#00d4aa", width=2, dash="dash"),
                marker=dict(size=7, symbol="diamond"),
        ))

        # Actual line
        fig_acc.add_trace(go.Scatter(
                x=acc_df["date"], y=acc_df["actual"],
                mode="lines+markers", name="Actual",
                line=dict(color="#ffd93d", width=2),
                marker=dict(size=7),
        ))

        # Color markers for within/outside band
        outside = acc_df[~acc_df["within_band"]]
        if not outside.empty:
            fig_acc.add_trace(go.Scatter(
                    x=outside["date"], y=outside["actual"],
                    mode="markers", name="Outside Band ✗",
                    marker=dict(size=12, color="#ff6b6b", symbol="x"),
            ))

        fig_acc.update_layout(
                title="Predicted vs Actual Indian Gold Price (₹/10g)",
                template="plotly_dark", height=450,
                yaxis_title="Price (₹/10g)", xaxis_title="Time",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode="x unified",
        )
        st.plotly_chart(fig_acc, width="stretch")

        # ── Hourly Accuracy Table ────────────────────────────────
        with st.expander("📋 Hourly Accuracy Breakdown", expanded=False):
            display_df = acc_df[["date", "predicted", "actual", "low_range",
                                 "high_range", "error", "pct_error", "within_band"]].copy()
            display_df = display_df.rename(columns={
                "date": "Hour",
                "predicted": "Predicted (₹)",
                "actual": "Actual (₹)",
                "low_range": "Low (₹)",
                "high_range": "High (₹)",
                "error": "Error (₹)",
                "pct_error": "Error (%)",
                "within_band": "In Range",
            })
            st.dataframe(
                display_df.style.format({
                    "Predicted (₹)": "₹{:,.2f}",
                    "Actual (₹)": "₹{:,.2f}",
                    "Low (₹)": "₹{:,.2f}",
                    "High (₹)": "₹{:,.2f}",
                    "Error (₹)": "{:+,.2f}",
                    "Error (%)": "{:.2f}%",
                }),
                width="stretch",
                hide_index=True,
            )

        # ── Per-Plan Accuracy History ────────────────────────────────
        if len(all_evals) > 1:
            with st.expander("📈 Accuracy Trend Across Predictions"):
                trend_data = []
                for ev in all_evals:
                    try:
                        gen_dt = parse_iso_to_ist(ev["plan_generated_at"])
                    except Exception:
                        gen_dt = pd.to_datetime(ev.get("plan_generated_at", ""), errors="coerce")
                    trend_data.append({
                        "Generated At": gen_dt,
                        "Hours Checked": ev["days_evaluated"],
                        "MAE (₹)": ev["mae"],
                        "MAPE (%)": ev["mape"],
                        "Band Hit (%)": ev["band_hit_rate"],
                        "Direction (%)": ev["directional_accuracy"],
                    })
                trend_df = pd.DataFrame(trend_data)
                trend_df = trend_df.dropna(subset=["Generated At"]).sort_values("Generated At")
                trend_df["Generated At Display"] = trend_df["Generated At"].dt.strftime("%Y-%m-%d %H:%M:%S")

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=trend_df["Generated At"], y=trend_df["MAPE (%)"],
                    mode="lines+markers", name="MAPE %",
                    line=dict(color="#ff6b6b"),
                ))
                fig_trend.add_trace(go.Scatter(
                    x=trend_df["Generated At"], y=trend_df["Band Hit (%)"],
                    mode="lines+markers", name="Band Hit %",
                    line=dict(color="#00d4aa"),
                ))
                fig_trend.add_trace(go.Scatter(
                    x=trend_df["Generated At"], y=trend_df["Direction (%)"],
                    mode="lines+markers", name="Direction %",
                    line=dict(color="#ffd93d"),
                ))
                fig_trend.update_layout(
                    title="Accuracy Trend Over Time",
                    template="plotly_dark", height=350,
                    yaxis_title="Percentage",
                    xaxis=dict(type="date", tickformat="%b %d\n%H:%M"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_trend, width="stretch")

                st.dataframe(
                    trend_df[["Generated At Display", "Hours Checked", "MAE (₹)", "MAPE (%)", "Band Hit (%)", "Direction (%)"]]
                    .rename(columns={"Generated At Display": "Generated At"}),
                    width="stretch",
                    hide_index=True,
                )

    else:
        # Show pending status with details
        _n_plans = len(accuracy_tracker.get_stored_plans())
        _latest_data = gold_df.index.max() if not gold_df.empty else None
        _pending_msg = (
            f"⏳ **Evaluation pending.** {_n_plans} prediction(s) stored and waiting "
            f"for actual market data to compare against.\n\n"
        )
        if _latest_data is not None:
            _pending_msg += (
                f"Latest market data: **{_latest_data.strftime('%Y-%m-%d %H:%M')} UTC**. "
                f"Predictions will be scored once newer candles arrive "
                f"(market may be closed for weekend/holiday).\n\n"
            )
        _pending_msg += "The system **auto-checks every 1 hour** in the background."
        st.info(_pending_msg)

else:
    st.info(
        "📍 **No accuracy data yet.** Accuracy scoring requires at least one "
        "prediction where predicted hours are now in the past. Generate a "
        "prediction and check back after those hours to see how accurate the "
        "system was!\n\n"
        "The system **auto-checks every 1 hour** in the background."
    )

# ── Prediction Generation History ────────────────────────────────────
if len(history) > 1:
    st.divider()
    st.subheader("📊 Prediction History")
    hist_data = []
    for h in history[-20:]:
        hist_data.append({
            "generated_at": h.generated_at,
            "price": h.current_price,
            "outlook": h.overall_outlook,
            "confidence": h.overall_confidence,
        })
    hdf = pd.DataFrame(hist_data)
    hdf["generated_at"] = pd.to_datetime(hdf["generated_at"])

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hdf["generated_at"], y=hdf["price"],
        mode="lines+markers", name="Gold Price at Prediction Time",
        line=dict(color="#ffd93d"),
    ))
    fig_hist.add_trace(go.Bar(
        x=hdf["generated_at"], y=hdf["confidence"],
        name="Confidence", yaxis="y2", opacity=0.3,
        marker_color=[outlook_color(o) for o in hdf["outlook"]],
    ))
    fig_hist.update_layout(
        template="plotly_dark", height=350,
        yaxis=dict(title="Gold Price (₹/10g)"),
        yaxis2=dict(title="Confidence", overlaying="y", side="right", range=[0, 1]),
    )
    st.plotly_chart(fig_hist, width="stretch")

# ── Footer ───────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer:** This is an AI-powered prediction system for educational/research "
    "purposes only. It is NOT financial advice. Gold prices are inherently unpredictable. "
    "Always consult a licensed financial advisor before making investment decisions."
)
