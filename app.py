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
    st.markdown("### 🎯 Accuracy Auto-Check")
    if st.button("🔍 Check Accuracy Now", use_container_width=True):
        with st.spinner("Fetching latest market data & comparing..."):
            n = accuracy_tracker.refresh_all()
        st.success(f"Checked! {n} plan(s) had new data.")
        st.rerun()

    last = accuracy_tracker.last_checked
    if last:
        try:
            last_dt = datetime.fromisoformat(last)
            st.caption(f"Last checked: {last_dt.strftime('%H:%M %b %d')}")
        except Exception:
            st.caption(f"Last checked: {last}")
    else:
        st.caption("Not checked yet — will auto-check in background")

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

# ════════════════════════════════════════════════════════════════════
# PREDICTION ACCURACY SCORECARD
# ════════════════════════════════════════════════════════════════════
st.subheader("🎯 Prediction Accuracy Scorecard")

# Auto-evaluate all stored plans against latest market data
history = engine.get_plan_history()
stored_plans = accuracy_tracker.get_stored_plans()

# Ensure all history plans are stored for tracking
import json as _json
if history:
    for h in history:
        plan_dict = _json.loads(h.model_dump_json())
        accuracy_tracker.store_plan(plan_dict)

# Also store the current plan
accuracy_tracker.store_plan(_json.loads(plan.model_dump_json()))

# Re-evaluate all stored plans (picks up any new day closes)
accuracy_tracker.refresh_all()

all_evals = accuracy_tracker.get_all_evaluations()
agg_stats = accuracy_tracker.get_aggregate_stats()

if agg_stats and agg_stats["total_predictions_evaluated"] > 0:
    # Show when accuracy was last auto-updated
    latest_eval = accuracy_tracker.get_latest_evaluation()
    if latest_eval:
        eval_time = latest_eval.get("evaluated_at", "")
        try:
            eval_dt = datetime.fromisoformat(eval_time)
            st.caption(f"🔄 Auto-updated: {eval_dt.strftime('%H:%M %b %d, %Y')} "
                       f"· {latest_eval['days_evaluated']}/{latest_eval['days_total']} days scored "
                       f"· Background check every 6h")
        except Exception:
            pass

if agg_stats and agg_stats["total_predictions_evaluated"] > 0:

    if agg_stats and agg_stats["total_predictions_evaluated"] > 0:
        # ── Aggregate Metrics Row ────────────────────────────────────
        st.markdown("#### Overall Accuracy (All Past Predictions)")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            mape = agg_stats["overall_mape"]
            mape_color = "🟢" if mape < 2 else ("🟡" if mape < 5 else "🔴")
            st.metric(f"{mape_color} MAPE", f"{mape:.1f}%")
        with m2:
            st.metric("📏 MAE", f"${agg_stats['overall_mae']:,.2f}")
        with m3:
            hit = agg_stats["overall_band_hit_rate"]
            hit_color = "🟢" if hit >= 70 else ("🟡" if hit >= 50 else "🔴")
            st.metric(f"{hit_color} Band Hit Rate", f"{hit:.0f}%")
        with m4:
            da = agg_stats["avg_directional_accuracy"]
            da_color = "🟢" if da >= 60 else ("🟡" if da >= 50 else "🔴")
            st.metric(f"{da_color} Direction Accuracy", f"{da:.0f}%")
        with m5:
            st.metric("📊 Days Evaluated", f"{agg_stats['total_predictions_evaluated']}")

        st.caption(
            "**MAPE** = Mean Absolute Percentage Error (lower is better) · "
            "**MAE** = Mean Absolute Error in $ · "
            "**Band Hit Rate** = % of days actual price fell within predicted range · "
            "**Direction** = % of days predicted direction matched actual"
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
                title="Predicted vs Actual Gold Price",
                template="plotly_dark", height=450,
                yaxis_title="Price (USD)", xaxis_title="Date",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode="x unified",
            )
            st.plotly_chart(fig_acc, use_container_width=True)

            # ── Daily Accuracy Table ─────────────────────────────────
            with st.expander("📋 Daily Accuracy Breakdown", expanded=False):
                display_df = acc_df[["date", "predicted", "actual", "low_range",
                                     "high_range", "error", "pct_error", "within_band"]].copy()
                display_df = display_df.rename(columns={
                    "date": "Date",
                    "predicted": "Predicted ($)",
                    "actual": "Actual ($)",
                    "low_range": "Low ($)",
                    "high_range": "High ($)",
                    "error": "Error ($)",
                    "pct_error": "Error (%)",
                    "within_band": "In Range",
                })
                st.dataframe(
                    display_df.style.format({
                        "Predicted ($)": "${:,.2f}",
                        "Actual ($)": "${:,.2f}",
                        "Low ($)": "${:,.2f}",
                        "High ($)": "${:,.2f}",
                        "Error ($)": "{:+,.2f}",
                        "Error (%)": "{:.2f}%",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

        # ── Per-Plan Accuracy History ────────────────────────────────
        if len(all_evals) > 1:
            with st.expander("📈 Accuracy Trend Across Predictions"):
                trend_data = []
                for ev in all_evals:
                    trend_data.append({
                        "Generated At": ev["plan_generated_at"][:16],
                        "Days Checked": ev["days_evaluated"],
                        "MAE ($)": ev["mae"],
                        "MAPE (%)": ev["mape"],
                        "Band Hit (%)": ev["band_hit_rate"],
                        "Direction (%)": ev["directional_accuracy"],
                    })
                trend_df = pd.DataFrame(trend_data)

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
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_trend, use_container_width=True)

                st.dataframe(trend_df, use_container_width=True, hide_index=True)

    else:
        st.info(
            "📍 **No accuracy data yet.** Accuracy scoring requires at least one "
            "prediction where the predicted dates are now in the past. Generate a "
            "prediction and check back after those dates to see how accurate the "
            "system was!\n\n"
            "The system **auto-checks every 6 hours** in the background, or click "
            "**Check Accuracy Now** in the sidebar."
        )

else:
    st.info(
        "📍 **No accuracy data yet.** Accuracy scoring requires at least one "
        "prediction where the predicted dates are now in the past. Generate a "
        "prediction and check back after those dates to see how accurate the "
        "system was!\n\n"
        "The system **auto-checks every 6 hours** in the background, or click "
        "**Check Accuracy Now** in the sidebar."
    )

st.divider()
# ── Prediction Generation History ────────────────────────────────────
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
