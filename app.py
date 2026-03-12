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

from src.prediction_engine import PredictionEngine
from src.data_fetchers.market_data import MarketDataFetcher

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Price Predictor – Agentic AI",
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
    st.divider()

    if st.button("🔄 Generate New Prediction", use_container_width=True, type="primary"):
        with st.spinner("Running 8 specialist agents … this takes 1-2 minutes"):
            plan = engine.generate()
        st.success("Prediction updated!")
        st.rerun()

    st.divider()

    # Auto-refresh toggle
    auto_refresh = st.toggle("Auto-refresh", value=False)
    if auto_refresh:
        from streamlit_autorefresh import st_autorefresh
        interval = st.slider("Refresh interval (min)", 15, 120, 30)
        st_autorefresh(interval=interval * 60_000, key="auto_refresh")

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
    st.caption(f"v1.0 · Updated {datetime.now().strftime('%H:%M')}")


# ════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ════════════════════════════════════════════════════════════════════
plan = engine.get_current_plan()

st.title("🥇 Agentic Gold Price Prediction System")

if plan is None:
    st.info("No prediction yet. Click **Generate New Prediction** in the sidebar to start.")
    # Show live gold chart as a teaser
    gold_df = market.fetch_ticker("GC=F", period_days=90)
    if not gold_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=gold_df.index, open=gold_df["Open"], high=gold_df["High"],
            low=gold_df["Low"], close=gold_df["Close"], name="Gold (GC=F)",
        ))
        fig.update_layout(
            title="Gold Futures – Last 90 Days",
            template="plotly_dark", height=500,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    st.stop()

# ── Top Metrics Row ──────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Current Price", f"${plan.current_price:,.2f}")
with c2:
    color = outlook_color(plan.overall_outlook)
    st.metric("Outlook", f"{outlook_emoji(plan.overall_outlook)} {plan.overall_outlook.upper()}")
with c3:
    st.metric("Confidence", f"{plan.overall_confidence:.0%}")
with c4:
    if plan.daily_predictions:
        day7 = plan.daily_predictions[-1]
        delta = day7.predicted_price - plan.current_price
        st.metric("7-Day Target", f"${day7.predicted_price:,.2f}", f"{delta:+,.2f}")
    else:
        st.metric("7-Day Target", "N/A")
with c5:
    st.metric("Last Updated", datetime.fromisoformat(plan.generated_at).strftime("%H:%M %b %d"))

st.divider()

# ── Executive Summary ────────────────────────────────────────────────
with st.expander("📋 Executive Summary", expanded=True):
    st.markdown(plan.executive_summary)

# ── 7-Day Prediction Chart ──────────────────────────────────────────
st.subheader("📅 7-Day Price Prediction")

if plan.daily_predictions:
    pred_df = pd.DataFrame([dp.model_dump() for dp in plan.daily_predictions])
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    # Also get recent actuals
    gold_recent = market.fetch_ticker("GC=F", period_days=30)

    fig = go.Figure()

    # Historical prices
    if not gold_recent.empty:
        fig.add_trace(go.Scatter(
            x=gold_recent.index, y=gold_recent["Close"],
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
        marker=dict(size=8),
    ))

    fig.update_layout(
        template="plotly_dark", height=500,
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Daily breakdown table
    st.dataframe(
        pred_df[["date", "predicted_price", "low_range", "high_range", "confidence", "key_driver"]]
        .rename(columns={
            "date": "Date",
            "predicted_price": "Predicted ($)",
            "low_range": "Low ($)",
            "high_range": "High ($)",
            "confidence": "Confidence",
            "key_driver": "Key Driver",
        })
        .style.format({
            "Predicted ($)": "${:,.2f}",
            "Low ($)": "${:,.2f}",
            "High ($)": "${:,.2f}",
            "Confidence": "{:.0%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning("No daily predictions available in this plan.")

st.divider()

# ── Agent Reports ────────────────────────────────────────────────────
st.subheader("🤖 Agent Reports")

if plan.agent_reports:
    # Agent confidence comparison chart
    agent_data = []
    for name, report in plan.agent_reports.items():
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
            st.plotly_chart(fig_conf, use_container_width=True)

        with col2:
            fig_bias = px.bar(
                adf, x="Agent", y="Bias", color="Outlook",
                color_discrete_map={"bullish": "#00d4aa", "bearish": "#ff6b6b", "neutral": "#ffd93d"},
                title="Agent Prediction Bias (-1 Bearish → +1 Bullish)",
            )
            fig_bias.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_bias, use_container_width=True)

    # Individual agent cards
    for name, report in plan.agent_reports.items():
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

st.divider()

# ── Historical Prediction Accuracy ──────────────────────────────────
history = engine.get_plan_history()
if len(history) > 1:
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
        yaxis=dict(title="Gold Price ($)"),
        yaxis2=dict(title="Confidence", overlaying="y", side="right", range=[0, 1]),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer:** This is an AI-powered prediction system for educational/research "
    "purposes only. It is NOT financial advice. Gold prices are inherently unpredictable. "
    "Always consult a licensed financial advisor before making investment decisions."
)
