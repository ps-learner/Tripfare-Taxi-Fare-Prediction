# === FILE: app.py ===
# ============================================================
# Project : TripFare — Predicting Urban Taxi Fare with ML
# File    : app.py
# Purpose : 5-page Streamlit application for interactive EDA,
#           model performance review, fare prediction, and
#           trip mapping.  Uses taxi-themed dark UI.
# Author  : TripFare Project
# ============================================================

import json
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import pytz
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Optional import for map page ────────────────────────────
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────

# Colour palette — taxi-themed dark dashboard
# These values mirror the CSS variables injected below so that
# both HTML/CSS elements and Plotly charts share one palette.
BG_DARK     = "#0A0A14"
BG_CARD     = "#111827"
GOLD        = "#F7C948"
RED         = "#E94560"
GREEN       = "#2ECC71"
BLUE        = "#3B82F6"
MUTED       = "#8B9BAD"
WHITE       = "#F0F4F8"

# Plotly chart layout defaults — applied to every chart via
# a helper function to ensure visual consistency across pages.
PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_CARD,
    plot_bgcolor =BG_CARD,
    font         =dict(color=WHITE, family="DM Sans, Inter, sans-serif"),
    title_font   =dict(color=GOLD,  size=15),
    legend       =dict(bgcolor=BG_DARK, bordercolor=MUTED, borderwidth=1),
    xaxis        =dict(gridcolor="#1F2937", zerolinecolor="#1F2937"),
    yaxis        =dict(gridcolor="#1F2937", zerolinecolor="#1F2937"),
    margin       =dict(l=40, r=20, t=50, b=40),
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title       = "🚕 TripFare — Taxi Fare Predictor",
    page_icon        = "🚕",
    layout           = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS INJECTION
# Purpose : Override Streamlit's default white/purple theme
#           with the taxi-yellow-on-dark palette.
# Using st.markdown with unsafe_allow_html because Streamlit's
# built-in theming doesn't support per-element colour overrides.
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600&display=swap');

  /* ── Global background ── */
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stApp"] {{
      background-color : {BG_DARK} !important;
      color            : {WHITE};
      font-family      : 'DM Sans', sans-serif;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
      background-color : {BG_CARD} !important;
      border-right     : 1px solid #1F2937;
  }}
  [data-testid="stSidebar"] * {{
      color : {WHITE} !important;
  }}

  /* ── Headers ── */
  h1, h2, h3 {{
      font-family : 'IBM Plex Mono', monospace !important;
      color       : {GOLD} !important;
  }}

  /* ── Metric cards ── */
  [data-testid="stMetric"] {{
      background    : {BG_CARD};
      border        : 1px solid #1F2937;
      border-radius : 12px;
      padding       : 16px 20px;
  }}
  [data-testid="stMetricLabel"] {{ color: {MUTED} !important; font-size: 0.82rem; }}
  [data-testid="stMetricValue"] {{ color: {GOLD}  !important; font-size: 1.7rem; font-weight: 700; }}
  [data-testid="stMetricDelta"] {{ color: {GREEN} !important; }}

  /* ── Buttons ── */
  .stButton > button {{
      background    : linear-gradient(135deg, {GOLD}, #E0A800);
      color         : {BG_DARK};
      font-weight   : 700;
      border        : none;
      border-radius : 8px;
      padding       : 10px 28px;
      font-size     : 1rem;
      transition    : opacity 0.2s;
  }}
  .stButton > button:hover {{ opacity: 0.85; }}

  /* ── Tabs ── */
  button[data-baseweb="tab"] {{
      color            : {MUTED} !important;
      background-color : {BG_CARD} !important;
      border-bottom    : 2px solid transparent;
  }}
  button[data-baseweb="tab"][aria-selected="true"] {{
      color        : {GOLD}  !important;
      border-bottom: 2px solid {GOLD} !important;
  }}

  /* ── DataFrames / Tables ── */
  .stDataFrame, .stTable {{
      background : {BG_CARD} !important;
  }}

  /* ── Input widgets ── */
  input, select, textarea {{
      background-color : {BG_CARD}  !important;
      color            : {WHITE}    !important;
      border           : 1px solid #374151 !important;
      border-radius    : 6px !important;
  }}

  /* ── Prediction result card ── */
  .fare-card {{
      background    : linear-gradient(135deg, #1a2240, #111827);
      border        : 2px solid {GOLD};
      border-radius : 16px;
      padding       : 28px 32px;
      text-align    : center;
      box-shadow    : 0 4px 24px rgba(247, 201, 72, 0.15);
  }}
  .fare-value {{
      font-family : 'IBM Plex Mono', monospace;
      font-size   : 3.5rem;
      font-weight : 700;
      color       : {GOLD};
      letter-spacing: 2px;
  }}
  .fare-label {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 8px; }}

  /* ── Hero banner ── */
  .hero-banner {{
      background    : linear-gradient(135deg, #0A0A14 0%, #1a1040 50%, #0A0A14 100%);
      border        : 1px solid #1F2937;
      border-radius : 16px;
      padding       : 40px;
      text-align    : center;
      margin-bottom : 28px;
  }}

  /* ── Taxi animation ── */
  @keyframes taxi-drive {{
      0%   {{ transform: translateX(-60px); }}
      100% {{ transform: translateX(60px);  }}
  }}
  .taxi-animate {{
      display   : inline-block;
      animation : taxi-drive 2s ease-in-out infinite alternate;
      font-size : 3rem;
  }}

  /* ── Pipeline step cards ── */
  .pipeline-step {{
      background    : {BG_CARD};
      border-left   : 4px solid {GOLD};
      border-radius : 8px;
      padding       : 14px 18px;
      margin-bottom : 10px;
  }}
  .step-number {{
      color       : {GOLD};
      font-family : 'IBM Plex Mono', monospace;
      font-weight : 700;
      font-size   : 1.1rem;
  }}

  /* ── Badge (best model) ── */
  .best-badge {{
      background    : linear-gradient(135deg, {GOLD}, #E0A800);
      color         : {BG_DARK};
      font-weight   : 700;
      border-radius : 20px;
      padding       : 4px 14px;
      font-size     : 0.85rem;
  }}

  /* ── Divider ── */
  hr {{ border-color: #1F2937; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Load all model artefacts from disk and cache them in memory.
    Using st.cache_resource so artefacts are loaded only once
    per Streamlit server session — avoids repeated disk I/O.

    Returns
    -------
    tuple : (best_model, scaler, selected_features,
              label_encoders, meta_dict)
            Returns (None, …) if artefacts are missing.
    """
    try:
        best_model        = joblib.load("models/best_model.pkl")
        scaler            = joblib.load("models/scaler.pkl")
        selected_features = joblib.load("models/selected_features.pkl")
        label_encoders    = joblib.load("models/label_encoders.pkl")
        with open("models/meta.json") as f:
            meta = json.load(f)
        return best_model, scaler, selected_features, label_encoders, meta
    except FileNotFoundError:
        return None, None, None, None, None

@st.cache_data(show_spinner=False)
def load_data():
    """
    Load the processed CSV from disk (data/taxi_data.csv).
    Returns an empty DataFrame with a warning if file missing.
    """
    path = "data/taxi_data.csv"
    if os.path.exists(path):
        return pd.read_csv(path, nrows=50_000)   # 50K rows for fast UI
    return pd.DataFrame()

def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """
    Apply the global PLOTLY_LAYOUT dict to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure

    Returns
    -------
    go.Figure  — same figure with theme applied.
    """
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

def haversine_scalar(lat1: float, lon1: float,
                     lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance (miles) between two points.
    Scalar version for use in Streamlit prediction form.

    Parameters
    ----------
    lat1, lon1 : float — Pickup latitude / longitude.
    lat2, lon2 : float — Dropoff latitude / longitude.

    Returns
    -------
    float  — Distance in miles.
    """
    R    = 3958.8
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a    = (np.sin(dlat/2)**2
            + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
            * np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))

def compute_bearing_scalar(lat1: float, lon1: float,
                            lat2: float, lon2: float) -> float:
    """
    Compute compass bearing (0–360°) from pickup to dropoff.

    Parameters
    ----------
    lat1/lon1, lat2/lon2 : float — Coordinates in decimal degrees.

    Returns
    -------
    float  — Bearing in degrees clockwise from North.
    """
    lat1r = np.radians(lat1);  lat2r = np.radians(lat2)
    dlon  = np.radians(lon2 - lon1)
    x     = np.sin(dlon) * np.cos(lat2r)
    y     = (np.cos(lat1r) * np.sin(lat2r)
             - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon))
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

# ─────────────────────────────────────────────────────────────
# LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────
best_model, scaler, selected_features, label_encoders, meta = load_models()
df_raw = load_data()

# ── Session state initialisation ────────────────────────────
# Using st.session_state to persist lat/lon values so
# coordinates entered on the Predict page automatically
# pre-fill the Trip Map page — improves UX flow.
for key, default in [
    ("pickup_lat",  40.7128), ("pickup_lon", -74.0059),
    ("dropoff_lat", 40.7580), ("dropoff_lon", -73.9855),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown(f"""
<div style="text-align:center; padding: 20px 0 10px 0;">
  <span style="font-family:'IBM Plex Mono',monospace;
               font-size:2rem; color:{GOLD};">🚕 TripFare</span><br/>
  <span style="color:{MUTED}; font-size:0.78rem;">
    Urban Taxi Fare Prediction
  </span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    options=["🏠  Home", "📊  EDA Dashboard",
             "🤖  Model Performance", "🎯  Predict Fare", "📍  Trip Map"],
    label_visibility="collapsed"
)

# ── Sidebar model status card ────────────────────────────────
st.sidebar.markdown("---")
if meta:
    st.sidebar.markdown(f"""
    <div style="background:{BG_DARK}; border:1px solid #1F2937;
                border-radius:10px; padding:14px; font-size:0.8rem;">
      <div style="color:{MUTED};">Best Model</div>
      <div style="color:{GOLD}; font-weight:700; font-family:'IBM Plex Mono',monospace;">
        {meta['best_model_name']}
      </div>
      <div style="color:{MUTED}; margin-top:8px;">
        R² &nbsp;<span style="color:{GREEN};">{meta['test_r2']:.4f}</span> &nbsp;|&nbsp;
        RMSE <span style="color:{RED};">${meta['test_rmse']:.2f}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.warning("⚠️ Models not found. Run model_training.py first.")

# ═════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════
if page == "🏠  Home":

    # ── Hero Banner ──────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-banner">
      <div class="taxi-animate">🚕</div>
      <h1 style="font-size:2.8rem; margin:12px 0 6px 0;">TripFare</h1>
      <p style="color:{MUTED}; font-size:1.1rem; margin:0;">
        Predicting Urban Taxi Fares with Machine Learning
      </p>
      <div style="margin-top:16px;">
        <span style="background:#1a2240; border:1px solid {GOLD};
                     border-radius:20px; padding:4px 14px; font-size:0.82rem;
                     color:{GOLD}; margin:0 4px;">
          Domain: Urban Transport Analytics
        </span>
        <span style="background:#1a2240; border:1px solid {GREEN};
                     border-radius:20px; padding:4px 14px; font-size:0.82rem;
                     color:{GREEN}; margin:0 4px;">
          Problem: Supervised Regression
        </span>
        <span style="background:#1a2240; border:1px solid {BLUE};
                     border-radius:20px; padding:4px 14px; font-size:0.82rem;
                     color:{BLUE}; margin:0 4px;">
          Target: total_amount ($)
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Metric Cards ─────────────────────────────────────
    # Four at-a-glance numbers that summarise the project scope.
    c1, c2, c3, c4 = st.columns(4)

    total_trips  = f"{len(df_raw):,}" if not df_raw.empty else "N/A"
    avg_fare     = f"${df_raw['total_amount'].mean():.2f}" if not df_raw.empty else "N/A"
    best_r2      = f"{meta['test_r2']:.4f}" if meta else "N/A"
    avg_dist_val = df_raw['trip_distance'].mean() if (
        not df_raw.empty and 'trip_distance' in df_raw.columns
    ) else None
    avg_dist = f"{avg_dist_val:.2f} mi" if avg_dist_val else "N/A"

    c1.metric("🚖 Total Trips Analysed", total_trips)
    c2.metric("💵 Average Fare",         avg_fare)
    c3.metric("🏆 Best Model R²",        best_r2)
    c4.metric("📍 Avg Trip Distance",    avg_dist)

    st.markdown("---")

    # ── Dataset Overview ─────────────────────────────────────
    col_info, col_pipeline = st.columns([1, 1])

    with col_info:
        st.markdown(f"### 📋 Dataset Overview")
        if not df_raw.empty:
            overview = pd.DataFrame({
                "Property" : ["Shape (rows × cols)", "Numeric columns",
                              "Categorical columns", "Missing values"],
                "Value"    : [
                    str(df_raw.shape),
                    str(df_raw.select_dtypes(include="number").shape[1]),
                    str(df_raw.select_dtypes(include="object").shape[1]),
                    str(df_raw.isnull().sum().sum()),
                ]
            })
            st.dataframe(overview, hide_index=True, use_container_width=True)
        else:
            st.info("Run `model_training.py` to generate the dataset.")

    with col_pipeline:
        st.markdown(f"### ⚙️ How It Works")
        steps = [
            ("01", "Data Collection",     "Download NYC taxi CSV via gdown"),
            ("02", "Feature Engineering", "Haversine distance, time flags, speed"),
            ("03", "EDA & Transformation","Outlier removal, log transform, encode"),
            ("04", "Model Training",       "8 models + hyperparameter tuning → save"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
              <span class="step-number">STEP {num}</span>
              &nbsp;&nbsp;<strong>{title}</strong>
              <div style="color:{MUTED}; font-size:0.85rem; margin-top:4px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Column Descriptions Table ─────────────────────────────
    st.markdown("### 🗂️ Dataset Column Guide")
    cols_desc = {
        "VendorID"              : "Taxi provider ID (1 or 2)",
        "pickup / dropoff"      : "GPS coordinates for trip start/end",
        "passenger_count"       : "Number of passengers (1–6)",
        "tpep_pickup_datetime"  : "Trip start timestamp (UTC → converted to EDT)",
        "fare_amount"           : "Base metered fare (excluded from model — leakage)",
        "extra"                 : "Rush-hour / overnight surcharge",
        "mta_tax"               : "MTA flat tax ($0.50)",
        "tip_amount"            : "Passenger tip",
        "tolls_amount"          : "Bridge/tunnel tolls",
        "improvement_surcharge" : "Flat surcharge ($0.30)",
        "total_amount"          : "🎯 TARGET — sum of all fare components",
        "payment_type"          : "1=Credit, 2=Cash, 3=No charge, 4=Dispute",
        "RatecodeID"            : "1=Standard, 2=JFK, 3=Newark, 4=Nassau, 5=Negotiated",
    }
    st.dataframe(
        pd.DataFrame(list(cols_desc.items()), columns=["Column", "Description"]),
        hide_index=True, use_container_width=True
    )

# ═════════════════════════════════════════════════════════════
# PAGE 2 — EDA DASHBOARD
# ═════════════════════════════════════════════════════════════
elif page == "📊  EDA Dashboard":

    st.markdown(f"# 📊 Exploratory Data Analysis")
    st.markdown(f"<p style='color:{MUTED};'>Interactive charts — hover, zoom, and filter to explore the data.</p>",
                unsafe_allow_html=True)

    if df_raw.empty:
        st.warning("No data found. Please run `model_training.py` first.")
        st.stop()

    # ── Sidebar Filters ───────────────────────────────────────
    # Filters update all charts on this page simultaneously
    # using Streamlit's reactive re-run mechanism.
    st.sidebar.markdown("### 🔧 EDA Filters")
    hour_range = st.sidebar.slider("Pickup Hour Range", 0, 23, (0, 23))
    weekday_options = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    sel_weekdays = st.sidebar.multiselect(
        "Weekday Filter", weekday_options, default=weekday_options
    )
    pax_max = st.sidebar.slider("Max Passenger Count", 1, 6, 6)

    # Apply filters to create the EDA working dataframe
    df_eda = df_raw.copy()
    if "pickup_hour" in df_eda.columns:
        df_eda = df_eda[df_eda["pickup_hour"].between(*hour_range)]
    if "pickup_weekday" in df_eda.columns:
        df_eda = df_eda[df_eda["pickup_weekday"].isin(sel_weekdays)]
    if "passenger_count" in df_eda.columns:
        df_eda = df_eda[df_eda["passenger_count"] <= pax_max]

    st.markdown(f"<small style='color:{MUTED};'>Filtered dataset: {len(df_eda):,} trips</small>",
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Distributions", "💵 Fare Analysis",
        "🕒 Time Patterns", "🔎 Outliers & Correlation"
    ])

    # ── TAB 1: Distributions ──────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Histogram of total_amount — shows right-skew clearly
            if "total_amount" in df_eda.columns:
                fig = px.histogram(
                    df_eda[df_eda["total_amount"].between(0, 100)],
                    x="total_amount", nbins=80,
                    title="Fare Amount Distribution",
                    color_discrete_sequence=[GOLD]
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 Right-skewed fare distribution; log transform used during training.")

        with col2:
            if "trip_distance" in df_eda.columns:
                fig = px.histogram(
                    df_eda[df_eda["trip_distance"].between(0, 25)],
                    x="trip_distance", nbins=80,
                    title="Trip Distance Distribution",
                    color_discrete_sequence=[GREEN]
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 Most trips are under 5 miles — short urban hops dominate.")

        if "trip_duration_min" in df_eda.columns:
            fig = px.histogram(
                df_eda[df_eda["trip_duration_min"].between(0, 90)],
                x="trip_duration_min", nbins=80,
                title="Trip Duration Distribution (minutes)",
                color_discrete_sequence=[BLUE]
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 Median trip duration ~12 min; very long durations suggest outliers.")

    # ── TAB 2: Fare Analysis ──────────────────────────────────
    with tab2:
        if "trip_distance" in df_eda.columns and "total_amount" in df_eda.columns:
            # Sample to keep scatter fast and browser-friendly
            sample_df = df_eda[
                df_eda["trip_distance"].between(0, 30) &
                df_eda["total_amount"].between(0, 100)
            ].sample