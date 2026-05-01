"""
generate_architecture.py
Run: python generate_architecture.py
Output: architecture.png (~3200×1800 px, 100 DPI)

Generates a corrected system architecture diagram for the Agentic Gold Price
Prediction System, matching the dark yellow-on-black theme of the project.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Colour palette ────────────────────────────────────────────────────────────
BG       = "#1a1a1a"   # canvas background
CARD     = "#2d2d2d"   # panel / box background
CARD2    = "#383838"   # slightly lighter card (nested boxes)
GOLD     = "#FFD700"   # primary accent – yellow/gold
GOLD2    = "#FFC300"   # secondary accent
WHITE    = "#FFFFFF"
GREY     = "#AAAAAA"
RED      = "#FF6B6B"
GREEN    = "#6BFF9E"
BLUE     = "#6BB5FF"
PURPLE   = "#C678DD"
ORANGE   = "#E5C07B"
TEAL     = "#56B6C2"

# ── Canvas setup ──────────────────────────────────────────────────────────────
FIG_W, FIG_H = 32, 18   # inches → at 100 DPI = 3200×1800
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ── Helper functions ──────────────────────────────────────────────────────────

def rounded_box(x, y, w, h, fc=CARD, ec=GOLD, lw=1.5, radius=0.35, zorder=2, alpha=1.0):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
        alpha=alpha,
    )
    ax.add_patch(box)
    return box


def label(x, y, txt, size=8, color=WHITE, weight="normal", ha="center", va="center", zorder=5):
    ax.text(x, y, txt, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=zorder,
            fontfamily="DejaVu Sans")


def stage_header(x, y, w, h, number, title, subtitle="", icon=""):
    """Draw a stage panel header band."""
    rounded_box(x, y + h - 1.1, w, 1.1, fc=GOLD, ec=GOLD, lw=0, radius=0.3, zorder=3)
    label(x + w / 2, y + h - 0.55, f"{icon} Stage {number}: {title}",
          size=11, color="#1a1a1a", weight="bold")
    if subtitle:
        label(x + w / 2, y + h - 0.55 - 0.38, subtitle, size=7.5, color="#1a1a1a")


def arrow(x1, y1, x2, y2, color=GOLD, lw=2, style="->", zorder=6,
          connectionstyle="arc3,rad=0.0"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            connectionstyle=connectionstyle,
        ),
        zorder=zorder,
    )


def bullet_lines(x, y, lines, size=7.4, color=WHITE, dy=0.34, bullet="•",
                 bullet_color=GOLD, indent=0.25):
    """Draw a vertical list of bullet-point lines."""
    for i, line in enumerate(lines):
        cy = y - i * dy
        label(x, cy, bullet, size=size, color=bullet_color, ha="left", va="center")
        label(x + indent, cy, line, size=size, color=color, ha="left", va="center")


def sub_box(x, y, w, h, title, lines, title_color=GOLD, fc=CARD2,
            ec=GOLD2, lw=1.0, title_size=7.8, body_size=7.2, dy=0.31):
    """Draw a labelled sub-box with bullet content."""
    rounded_box(x, y, w, h, fc=fc, ec=ec, lw=lw, radius=0.2, zorder=3)
    label(x + w / 2, y + h - 0.22, title, size=title_size,
          color=title_color, weight="bold")
    top = y + h - 0.52
    bullet_lines(x + 0.18, top, lines, size=body_size, dy=dy)


# ─────────────────────────────────────────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────────────────────────────────────────
MARGIN   = 0.35
FOOTER_H = 0.65
HEADER_H = 0.90
BODY_TOP = FIG_H - MARGIN - HEADER_H   # y-top of stage panels
BODY_BOT = MARGIN + FOOTER_H + 0.15    # y-bottom of stage panels
BODY_H   = BODY_TOP - BODY_BOT         # total height of stage panels

N_STAGES = 5
GAP      = 0.18                        # gap between stage panels
TOTAL_W  = FIG_W - 2 * MARGIN
SW       = (TOTAL_W - (N_STAGES - 1) * GAP) / N_STAGES  # stage width
SX = [MARGIN + i * (SW + GAP) for i in range(N_STAGES)]  # stage left-x coords
SY = BODY_BOT                          # stage bottom-y

# ─────────────────────────────────────────────────────────────────────────────
# Top title bar
# ─────────────────────────────────────────────────────────────────────────────
rounded_box(MARGIN, FIG_H - MARGIN - HEADER_H, TOTAL_W, HEADER_H,
            fc="#111111", ec=GOLD, lw=2, radius=0.3, zorder=2)
label(FIG_W / 2, FIG_H - MARGIN - HEADER_H / 2,
      "🥇  Agentic Gold Price Prediction System  —  Corrected System Architecture",
      size=15, color=GOLD, weight="bold")

# ─────────────────────────────────────────────────────────────────────────────
# Footer bar
# ─────────────────────────────────────────────────────────────────────────────
rounded_box(MARGIN, MARGIN, TOTAL_W, FOOTER_H,
            fc="#111111", ec=GOLD, lw=1.5, radius=0.25, zorder=2)
label(FIG_W / 2, MARGIN + FOOTER_H / 2,
      # Update frequency from src/config.py REFRESH_INTERVAL_MINUTES = 30
      "MCX Trading Hours: 09:15–17:30 IST (Mon–Fri)  |  "
      "Update Frequency: Every 30 minutes  |  Total Latency: ~50 s",
      size=9, color=GREY)

# ─────────────────────────────────────────────────────────────────────────────
# Stage backgrounds
# ─────────────────────────────────────────────────────────────────────────────
for i in range(N_STAGES):
    rounded_box(SX[i], SY, SW, BODY_H, fc="#222222", ec=GOLD, lw=2,
                radius=0.4, zorder=1)

# ─────────────────────────────────────────────────────────────────────────────
# Inter-stage arrows
# ─────────────────────────────────────────────────────────────────────────────
ARROW_Y = SY + BODY_H / 2
for i in range(N_STAGES - 1):
    x1 = SX[i] + SW + 0.02
    x2 = SX[i + 1] - 0.02
    arrow(x1, ARROW_Y, x2, ARROW_Y, color=GOLD, lw=3)

# ─────────────────────────────────────────────────────────────────────────────
# ── STAGE 1 – Data Ingestion ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
s = 0
stage_header(SX[s], SY, SW, BODY_H, 1, "Data Ingestion", icon="📥")

inner_x = SX[s] + 0.22
inner_w = SW - 0.44
top_y   = SY + BODY_H - 1.25   # below header band

# ── Market Data ──
bh = 1.75
by = top_y - bh - 0.05
sub_box(inner_x, by, inner_w, bh,
        "📊 Market Data (Yahoo Finance)",
        [
            "COMEX GC=F  ← ML training & hourly prediction (primary)",
            "MCX GOLD.NS  ← UI display price only (daily)",
            "USD/INR  ← real-time FX conversion",
            "DX-Y.NYB  ← US Dollar Index (macro/FX)",
            "^TNX  ← US 10Y Treasury yield",
        ],
        title_color=GOLD, dy=0.305)

# ── ETF & Oil Data ──
bh2 = 1.60
by2 = by - bh2 - 0.18
sub_box(inner_x, by2, inner_w, bh2,
        "💰 ETF & Oil Data (Yahoo Finance)",
        [
            "GOLDBEES.NS — Nippon India ETF Gold BeES",
            "HDFCGOLD.NS — HDFC Gold ETF",
            "TATAGOLD.NS — Tata Gold ETF",
            "Indian Gold Mutual Funds (0P0000XVLE.BO, ...)",
            "CL=F — Crude Oil / WTI",
        ],
        title_color=TEAL, dy=0.29)

# ── US Macro (FRED) ──
bh3 = 1.22
by3 = by2 - bh3 - 0.18
sub_box(inner_x, by3, inner_w, bh3,
        "🏦 US Macro — FRED API",
        [
            "FEDFUNDS — US Fed Funds Rate",
            "CPIAUCSL — US CPI",
            "REAINTRATREARAT10Y — US Real Interest Rate",
        ],
        title_color=BLUE, dy=0.30)

# ── India Macro ──
bh4 = 1.55
by4 = by3 - bh4 - 0.18
sub_box(inner_x, by4, inner_w, bh4,
        "🇮🇳 India Macro — india_context.py",
        [
            "RBI Repo Rate (6.00 %)",
            "India CPI (3.61 %)",
            "Gold Import Duty (6 % total)",
            "India 10Y Bond Yield (IN10Y=RR live)",
            "Festival Calendar & Seasonal Demand",
        ],
        title_color=ORANGE, dy=0.28)

# ── News ──
bh5 = 0.90
by5 = by4 - bh5 - 0.18
sub_box(inner_x, by5, inner_w, bh5,
        "📰 News & Sentiment",
        [
            "NewsAPI — headlines & keyword scoring",
            "RSS feeds — geopolitics, BRICS, sanctions",
        ],
        title_color=PURPLE, dy=0.29)

# ─────────────────────────────────────────────────────────────────────────────
# ── STAGE 2 – Agent Intelligence ─────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
s = 1
stage_header(SX[s], SY, SW, BODY_H, 2, "Agent Intelligence", icon="🤖")

inner_x2 = SX[s] + 0.22
inner_w2  = SW - 0.44
top_y2    = SY + BODY_H - 1.25

agents = [
    ("🌍", "Geopolitics",    "Wars, sanctions, BRICS, central bank reserves",       RED),
    ("📈", "Trend Analysis", "Price trends, moving averages, momentum",              TEAL),
    ("💰", "ETF Flows",      "Gold ETF inflows/outflows, institutional demand",      GOLD2),
    ("🏦", "Macro Econ",     "Interest rates, CPI, USD, M2 (US + India)",            BLUE),
    ("🛢️", "Oil & Energy",   "Oil prices, OPEC, gold-oil ratio",                    ORANGE),
    ("😨", "Sentiment",      "VIX, fear/greed index, news tone",                     PURPLE),
    ("📊", "Technical",      "RSI, MACD, Bollinger, support/resistance",             GREEN),
    ("📜", "Historical",     "Seasonal patterns, YoY, drawdowns, cycles",            TEAL),
]

ag_h = (BODY_H - 1.45) / len(agents) - 0.12
ag_y  = top_y2 - 0.08

for icon, name, desc, col in agents:
    rounded_box(inner_x2, ag_y - ag_h, inner_w2, ag_h,
                fc=CARD2, ec=col, lw=1.2, radius=0.2, zorder=3)
    label(inner_x2 + 0.22, ag_y - ag_h / 2 + 0.10,
          f"{icon} {name}", size=7.8, color=col, weight="bold",
          ha="left", va="center")
    label(inner_x2 + 0.22, ag_y - ag_h / 2 - 0.12,
          desc, size=6.8, color=GREY, ha="left", va="center")
    ag_y -= ag_h + 0.12

# LLM label
lx = inner_x2 + inner_w2 / 2
ly = ag_y - 0.28
label(lx, ly, "Each agent calls GPT-4o for structured JSON analysis",
      size=7.2, color=GREY, ha="center")
label(lx, ly - 0.32, "Output: {outlook, confidence, prediction_bias, key_factors}",
      size=6.8, color=GREY, ha="center")

# ─────────────────────────────────────────────────────────────────────────────
# ── STAGE 3 – Feature Engineering ────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
s = 2
stage_header(SX[s], SY, SW, BODY_H, 3, "Feature Engineering", icon="⚙️")

inner_x3 = SX[s] + 0.22
inner_w3  = SW - 0.44
top_y3    = SY + BODY_H - 1.25

# 16 time-series features box
feat_h = 3.70
feat_y = top_y3 - feat_h - 0.05
sub_box(inner_x3, feat_y, inner_w3, feat_h,
        "📐 16 Time-Series Features → ML Input Vector",
        [
            "── Price Lags (6 features) ──",
            "lag_1, lag_2, lag_3, lag_6, lag_12, lag_24",
            "── Rolling Means (3 features) ──",
            "roll_6, roll_12, roll_24",
            "── Returns & Volatility (3 features) ──",
            "ret_1h, ret_6h, vol_12",
            "── Time Cycles (4 features) ──",
            "hour_sin, hour_cos, dow_sin, dow_cos",
        ],
        title_color=GREEN, fc=CARD2, ec=GREEN, dy=0.38)

# Agent signals – separate path (post-prediction)
sig_h = 2.50
sig_y = feat_y - sig_h - 0.30
sub_box(inner_x3, sig_y, inner_w3, sig_h,
        "🧠 8 Agent Signals → Post-Prediction Adjustment",
        [
            "sentiment_score",
            "geopolitical_risk",
            "macro_outlook",
            "technical_signal",
            "etf_flow_signal",
            "oil_energy_signal",
            "historical_seasonal",
            "overall_bias",
        ],
        title_color=PURPLE, fc="#2a2040", ec=PURPLE, dy=0.255)

# Note underneath
note_y = sig_y - 0.32
label(inner_x3 + inner_w3 / 2, note_y,
      "⚠  Agent signals are NOT ML inputs — applied after",
      size=7.0, color=PURPLE, ha="center")
label(inner_x3 + inner_w3 / 2, note_y - 0.28,
      "prediction as a ±1.5 % directional multiplier",
      size=7.0, color=PURPLE, ha="center")

# ─────────────────────────────────────────────────────────────────────────────
# ── STAGE 4 – ML Ensemble ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
s = 3
stage_header(SX[s], SY, SW, BODY_H, 4, "ML Ensemble & Stacking", icon="🔬")

inner_x4 = SX[s] + 0.22
inner_w4  = SW - 0.44
top_y4    = SY + BODY_H - 1.25

# Base learners
base_h = 0.80
base_labels = [
    ("XGBoost", "Gradient Boosted Trees (base learner)", RED),
    ("LightGBM", "Gradient Boosted Trees (base learner)", GREEN),
    ("Ridge",    "Linear Baseline (base learner)",        BLUE),
]
base_y = top_y4 - 0.08
bw = (inner_w4 - 0.30) / 3

for i, (name, desc, col) in enumerate(base_labels):
    bx = inner_x4 + i * (bw + 0.15)
    rounded_box(bx, base_y - base_h, bw, base_h,
                fc=CARD2, ec=col, lw=1.3, radius=0.18, zorder=3)
    label(bx + bw / 2, base_y - base_h / 2 + 0.10,
          name, size=8.0, color=col, weight="bold", ha="center")
    label(bx + bw / 2, base_y - base_h / 2 - 0.13,
          desc, size=6.2, color=GREY, ha="center")

# Arrow down to meta-learner
meta_arrow_x = inner_x4 + inner_w4 / 2
meta_top = base_y - base_h
meta_h   = 0.90
meta_y   = meta_top - 0.50 - meta_h
arrow(meta_arrow_x, meta_top, meta_arrow_x, meta_y + meta_h + 0.02,
      color=GOLD, lw=2)

# Ridge meta-learner box
rounded_box(inner_x4 + 0.30, meta_y, inner_w4 - 0.60, meta_h,
            fc="#1f2d1f", ec=GOLD, lw=2.0, radius=0.22, zorder=4)
label(inner_x4 + inner_w4 / 2, meta_y + meta_h / 2 + 0.12,
      "Ridge Meta-Learner", size=9, color=GOLD, weight="bold")
label(inner_x4 + inner_w4 / 2, meta_y + meta_h / 2 - 0.15,
      "Stacks 3 base predictions — weights learned from validation data",
      size=6.8, color=GREY)

# Quantile bands
qb_h = 0.70
qb_y = meta_y - 0.40 - qb_h
rounded_box(inner_x4 + 0.30, qb_y, inner_w4 - 0.60, qb_h,
            fc=CARD2, ec=TEAL, lw=1.2, radius=0.18, zorder=3)
label(inner_x4 + inner_w4 / 2, qb_y + qb_h / 2 + 0.08,
      "📉 Quantile Bands (XGBoost)", size=7.8, color=TEAL, weight="bold")
label(inner_x4 + inner_w4 / 2, qb_y + qb_h / 2 - 0.15,
      "5th–95th percentile uncertainty envelope", size=6.8, color=GREY)
arrow(meta_arrow_x, meta_y, meta_arrow_x, qb_y + qb_h + 0.02,
      color=TEAL, lw=1.5)

# Agent-signal adjustment arrow (comes from Stage 3 sig box, feeds into here)
adj_h = 0.70
adj_y = qb_y - 0.35 - adj_h
rounded_box(inner_x4 + 0.30, adj_y, inner_w4 - 0.60, adj_h,
            fc="#2a2040", ec=PURPLE, lw=1.5, radius=0.18, zorder=3)
label(inner_x4 + inner_w4 / 2, adj_y + adj_h / 2 + 0.08,
      "🧠 Agent Signal Adjustment", size=7.8, color=PURPLE, weight="bold")
label(inner_x4 + inner_w4 / 2, adj_y + adj_h / 2 - 0.15,
      "±1.5 % directional multiplier applied post-prediction",
      size=6.8, color=GREY)
arrow(meta_arrow_x, qb_y, meta_arrow_x, adj_y + adj_h + 0.02,
      color=PURPLE, lw=1.5)

# ResidualLearner
rl_h = 0.80
rl_y  = adj_y - 0.35 - rl_h
rounded_box(inner_x4 + 0.20, rl_y, inner_w4 - 0.40, rl_h,
            fc="#1f2020", ec=ORANGE, lw=1.8, radius=0.20, zorder=4)
label(inner_x4 + inner_w4 / 2, rl_y + rl_h / 2 + 0.12,
      "🔧 Residual Correction Layer (residual_learner.py)",
      size=7.8, color=ORANGE, weight="bold")
label(inner_x4 + inner_w4 / 2, rl_y + rl_h / 2 - 0.15,
      "Slot-specific bias correction · dynamic per-agent weight adjustment",
      size=6.5, color=GREY)
arrow(meta_arrow_x, adj_y, meta_arrow_x, rl_y + rl_h + 0.02,
      color=ORANGE, lw=1.5)

# Guardrails
gr_h = 0.75
gr_y  = rl_y - 0.30 - gr_h
rounded_box(inner_x4 + 0.20, gr_y, inner_w4 - 0.40, gr_h,
            fc="#201010", ec=RED, lw=1.8, radius=0.20, zorder=4)
label(inner_x4 + inner_w4 / 2, gr_y + gr_h / 2 + 0.10,
      "🛡 Guardrails Validation Gate (guardrails.py)",
      size=7.8, color=RED, weight="bold")
label(inner_x4 + inner_w4 / 2, gr_y + gr_h / 2 - 0.15,
      "Validates predictions — blocks out-of-range outputs before Stage 5",
      size=6.5, color=GREY)
arrow(meta_arrow_x, rl_y, meta_arrow_x, gr_y + gr_h + 0.02, color=RED, lw=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# ── STAGE 5 – Output & Feedback ──────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
s = 4
stage_header(SX[s], SY, SW, BODY_H, 5, "Output & Feedback", icon="📤")

inner_x5 = SX[s] + 0.22
inner_w5  = SW - 0.44
top_y5    = SY + BODY_H - 1.25

# Streamlit Dashboard
dash_h = 1.45
dash_y = top_y5 - dash_h - 0.05
sub_box(inner_x5, dash_y, inner_w5, dash_h,
        "🖥 Streamlit Dashboard (app.py)",
        [
            "INR/10 g price — current + 24-hour forecast",
            "Confidence bands (5th–95th percentile)",
            "SHAP explainability chart",
            "8-agent report cards",
        ],
        title_color=GOLD, dy=0.28)

# Narrator
nar_h = 0.95
nar_y = dash_y - nar_h - 0.22
sub_box(inner_x5, nar_y, inner_w5, nar_h,
        "✍ Narrator LLM (narrator.py)",
        [
            "GPT-4o writes prose analysis (never generates prices)",
            "Executive summary · Bull/Bear cases · Risk factors",
        ],
        title_color=PURPLE, dy=0.30)

# Live Price Tracking
lp_h = 0.88
lp_y  = nar_y - lp_h - 0.22
sub_box(inner_x5, lp_y, inner_w5, lp_h,
        "📡 Live Price Tracking",
        [
            "30-minute refresh cycle (REFRESH_INTERVAL_MINUTES = 30)",
            "MCX GOLD.NS — daily display price (UI)",
        ],
        title_color=TEAL, dy=0.30)

# Accuracy Tracker
at_h = 0.90
at_y  = lp_y - at_h - 0.22
sub_box(inner_x5, at_y, inner_w5, at_h,
        "📊 Accuracy Tracker (accuracy_tracker.py)",
        [
            "Logs predicted vs actual prices every cycle",
            "Feeds prediction errors → ResidualLearner (Stage 4)",
        ],
        title_color=GREEN, dy=0.30)

# Feedback arrow: Accuracy Tracker → Residual Learner
# (from Stage 5 bottom-left to Stage 4 right side of ResidualLearner)
fb_x1 = SX[s] + 0.12                    # left edge of stage 5
fb_y1 = at_y + at_h / 2                  # mid-height of accuracy tracker
fb_x2 = SX[s - 1] + SW + 0.02           # right edge of stage 4 (≈ arrow arrival)
fb_y2 = rl_y + rl_h / 2                  # mid-height of residual learner

# Draw a curved feedback arrow below the panels
arrow(fb_x1, fb_y1, fb_x2, fb_y2,
      color=GREEN, lw=2.0, style="<-",
      connectionstyle="arc3,rad=-0.28")
label((fb_x1 + fb_x2) / 2, fb_y1 - 0.45,
      "feedback loop", size=7.0, color=GREEN, ha="center")

# Prediction archive / cloud
arch_h = 0.80
arch_y  = at_y - arch_h - 0.22
sub_box(inner_x5, arch_y, inner_w5, arch_h,
        "☁ Prediction Archive (cloud_storage.py)",
        [
            "S3 / local storage — model checkpoints & predictions",
        ],
        title_color=GREY, dy=0.30)

# ─────────────────────────────────────────────────────────────────────────────
# ── Agent signal dashed arrow: Stage 3 → Stage 4 ─────────────────────────────
# (illustrate that agent signals bypass ML and go directly to adjustment layer)
# ─────────────────────────────────────────────────────────────────────────────
sig_mid_x = SX[2] + SW / 2
sig_mid_y  = sig_y + sig_h / 2

adj_mid_x  = SX[3] + SW / 2
adj_mid_y  = adj_y + adj_h / 2

ax.annotate(
    "", xy=(adj_mid_x - inner_w4 / 2 + 0.15, adj_mid_y),
    xytext=(sig_mid_x + inner_w3 / 2, sig_mid_y),
    arrowprops=dict(
        arrowstyle="->",
        color=PURPLE,
        lw=1.8,
        linestyle="dashed",
        connectionstyle="arc3,rad=-0.22",
    ),
    zorder=7,
)
label((sig_mid_x + adj_mid_x) / 2 + 0.3,
      (sig_mid_y + adj_mid_y) / 2 - 0.65,
      "post-prediction\nadjustment path",
      size=6.5, color=PURPLE, ha="center")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0)
plt.savefig("architecture.png", dpi=100, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print("architecture.png saved successfully.")
