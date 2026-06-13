# # === FILE: app.py ===
# # ============================================================
# # Project : TripFare — Predicting Urban Taxi Fare with ML
# # File    : app.py
# # Purpose : 5-page Streamlit application for interactive EDA,
# #           model performance review, fare prediction, and
# #           trip mapping.  Uses taxi-themed dark UI.
# # Author  : TripFare Project
# # ============================================================

# import json
# import os
# import warnings
# import numpy as np
# import pandas as pd
# import joblib
# import pytz
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import streamlit as st

# # ── Optional import for map page ────────────────────────────
# try:
#     import folium
#     from streamlit_folium import st_folium
#     FOLIUM_AVAILABLE = True
# except ImportError:
#     FOLIUM_AVAILABLE = False

# warnings.filterwarnings("ignore")

# # ─────────────────────────────────────────────────────────────
# # GLOBAL CONSTANTS & CONFIG
# # ─────────────────────────────────────────────────────────────

# # Colour palette — taxi-themed dark dashboard
# # These values mirror the CSS variables injected below so that
# # both HTML/CSS elements and Plotly charts share one palette.
# BG_DARK     = "#0A0A14"
# BG_CARD     = "#111827"
# GOLD        = "#F7C948"
# RED         = "#E94560"
# GREEN       = "#2ECC71"
# BLUE        = "#3B82F6"
# MUTED       = "#8B9BAD"
# WHITE       = "#F0F4F8"

# # Plotly chart layout defaults — applied to every chart via
# # a helper function to ensure visual consistency across pages.
# PLOTLY_LAYOUT = dict(
#     paper_bgcolor=BG_CARD,
#     plot_bgcolor =BG_CARD,
#     font         =dict(color=WHITE, family="DM Sans, Inter, sans-serif"),
#     title_font   =dict(color=GOLD,  size=15),
#     legend       =dict(bgcolor=BG_DARK, bordercolor=MUTED, borderwidth=1),
#     xaxis        =dict(gridcolor="#1F2937", zerolinecolor="#1F2937"),
#     yaxis        =dict(gridcolor="#1F2937", zerolinecolor="#1F2937"),
#     margin       =dict(l=40, r=20, t=50, b=40),
# )

# # ─────────────────────────────────────────────────────────────
# # PAGE CONFIGURATION — must be the very first Streamlit call
# # ─────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title       = "TripFare — Taxi Fare Predictor",
#     page_icon        = "🚕",
#     layout           = "wide",
#     initial_sidebar_state = "expanded",
# )

# # ─────────────────────────────────────────────────────────────
# # CUSTOM CSS INJECTION
# # Purpose : Override Streamlit's default white/purple theme
# #           with the taxi-yellow-on-dark palette.
# # Using st.markdown with unsafe_allow_html because Streamlit's
# # built-in theming doesn't support per-element colour overrides.
# # ─────────────────────────────────────────────────────────────
# st.markdown(f"""
# <style>
#   /* ── Google Fonts ── */
#   @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600&display=swap');

#   /* ── Global background ── */
#   html, body, [data-testid="stAppViewContainer"],
#   [data-testid="stApp"] {{
#       background-color : {BG_DARK} !important;
#       color            : {WHITE};
#       font-family      : 'DM Sans', sans-serif;
#   }}

#   /* ── Sidebar ── */
#   [data-testid="stSidebar"] {{
#       background-color : {BG_CARD} !important;
#       border-right     : 1px solid #1F2937;
#   }}
#   [data-testid="stSidebar"] * {{
#       color : {WHITE} !important;
#   }}

#   /* ── Headers ── */
#   h1, h2, h3 {{
#       font-family : 'IBM Plex Mono', monospace !important;
#       color       : {GOLD} !important;
#   }}

#   /* ── Metric cards ── */
#   [data-testid="stMetric"] {{
#       background    : {BG_CARD};
#       border        : 1px solid #1F2937;
#       border-radius : 12px;
#       padding       : 16px 20px;
#   }}
#   [data-testid="stMetricLabel"] {{ color: {MUTED} !important; font-size: 0.82rem; }}
#   [data-testid="stMetricValue"] {{ color: {GOLD}  !important; font-size: 1.7rem; font-weight: 700; }}
#   [data-testid="stMetricDelta"] {{ color: {GREEN} !important; }}

#   /* ── Buttons ── */
#   .stButton > button {{
#       background    : linear-gradient(135deg, {GOLD}, #E0A800);
#       color         : {BG_DARK};
#       font-weight   : 700;
#       border        : none;
#       border-radius : 8px;
#       padding       : 10px 28px;
#       font-size     : 1rem;
#       transition    : opacity 0.2s;
#   }}
#   .stButton > button:hover {{ opacity: 0.85; }}

#   /* ── Tabs ── */
#   button[data-baseweb="tab"] {{
#       color            : {MUTED} !important;
#       background-color : {BG_CARD} !important;
#       border-bottom    : 2px solid transparent;
#   }}
#   button[data-baseweb="tab"][aria-selected="true"] {{
#       color        : {GOLD}  !important;
#       border-bottom: 2px solid {GOLD} !important;
#   }}

#   /* ── DataFrames / Tables ── */
#   .stDataFrame, .stTable {{
#       background : {BG_CARD} !important;
#   }}

#   /* ── Input widgets ── */
#   input, select, textarea {{
#       background-color : {BG_CARD}  !important;
#       color            : {WHITE}    !important;
#       border           : 1px solid #374151 !important;
#       border-radius    : 6px !important;
#   }}

#   /* ── Prediction result card ── */
#   .fare-card {{
#       background    : linear-gradient(135deg, #1a2240, #111827);
#       border        : 2px solid {GOLD};
#       border-radius : 16px;
#       padding       : 28px 32px;
#       text-align    : center;
#       box-shadow    : 0 4px 24px rgba(247, 201, 72, 0.15);
#   }}
#   .fare-value {{
#       font-family : 'IBM Plex Mono', monospace;
#       font-size   : 3.5rem;
#       font-weight : 700;
#       color       : {GOLD};
#       letter-spacing: 2px;
#   }}
#   .fare-label {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 8px; }}

#   /* ── Hero banner ── */
#   .hero-banner {{
#       background    : linear-gradient(135deg, #0A0A14 0%, #1a1040 50%, #0A0A14 100%);
#       border        : 1px solid #1F2937;
#       border-radius : 16px;
#       padding       : 40px;
#       text-align    : center;
#       margin-bottom : 28px;
#   }}

#   /* ── Taxi animation ── */
#   @keyframes taxi-drive {{
#       0%   {{ transform: translateX(-60px); }}
#       100% {{ transform: translateX(60px);  }}
#   }}
#   .taxi-animate {{
#       display   : inline-block;
#       animation : taxi-drive 2s ease-in-out infinite alternate;
#       font-size : 3rem;
#   }}

#   /* ── Pipeline step cards ── */
#   .pipeline-step {{
#       background    : {BG_CARD};
#       border-left   : 4px solid {GOLD};
#       border-radius : 8px;
#       padding       : 14px 18px;
#       margin-bottom : 10px;
#   }}
#   .step-number {{
#       color       : {GOLD};
#       font-family : 'IBM Plex Mono', monospace;
#       font-weight : 700;
#       font-size   : 1.1rem;
#   }}

#   /* ── Badge (best model) ── */
#   .best-badge {{
#       background    : linear-gradient(135deg, {GOLD}, #E0A800);
#       color         : {BG_DARK};
#       font-weight   : 700;
#       border-radius : 20px;
#       padding       : 4px 14px;
#       font-size     : 0.85rem;
#   }}

#   /* ── Divider ── */
#   hr {{ border-color: #1F2937; }}
# </style>
# """, unsafe_allow_html=True)

# # ─────────────────────────────────────────────────────────────
# # HELPER FUNCTIONS
# # ─────────────────────────────────────────────────────────────

# @st.cache_resource(show_spinner=False)
# def load_models():
#     """
#     Load all model artefacts from disk and cache them in memory.
#     Using st.cache_resource so artefacts are loaded only once
#     per Streamlit server session — avoids repeated disk I/O.

#     Returns
#     -------
#     tuple : (best_model, scaler, selected_features,
#               label_encoders, meta_dict)
#             Returns (None, …) if artefacts are missing.
#     """
#     try:
#         best_model        = joblib.load("models/best_model.pkl")
#         scaler            = joblib.load("models/scaler.pkl")
#         selected_features = joblib.load("models/selected_features.pkl")
#         label_encoders    = joblib.load("models/label_encoders.pkl")
#         with open("models/meta.json") as f:
#             meta = json.load(f)
#         return best_model, scaler, selected_features, label_encoders, meta
#     except FileNotFoundError:
#         return None, None, None, None, None

# @st.cache_data(show_spinner=False)
# def load_data():
#     """
#     Load the processed CSV from disk (data/taxi_data.csv).
#     Returns an empty DataFrame with a warning if file missing.
#     """

#     path = "data/taxi_data.csv"
#     if os.path.exists(path):
#         return pd.read_csv(path, nrows=50_000)   # 50K rows for fast UI
#     return pd.DataFrame()

# def apply_plotly_theme(fig: go.Figure) -> go.Figure:
#     """
#     Apply the global PLOTLY_LAYOUT dict to a Plotly figure.

#     Parameters
#     ----------
#     fig : go.Figure

#     Returns
#     -------
#     go.Figure  — same figure with theme applied.
#     """
#     fig.update_layout(**PLOTLY_LAYOUT)
#     return fig

# def haversine_scalar(lat1: float, lon1: float,
#                      lat2: float, lon2: float) -> float:
#     """
#     Compute great-circle distance (miles) between two points.
#     Scalar version for use in Streamlit prediction form.

#     Parameters
#     ----------
#     lat1, lon1 : float — Pickup latitude / longitude.
#     lat2, lon2 : float — Dropoff latitude / longitude.

#     Returns
#     -------
#     float  — Distance in miles.
#     """
#     R    = 3958.8
#     dlat = np.radians(lat2 - lat1)
#     dlon = np.radians(lon2 - lon1)
#     a    = (np.sin(dlat/2)**2
#             + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
#             * np.sin(dlon/2)**2)
#     return R * 2 * np.arcsin(np.sqrt(a))

# def compute_bearing_scalar(lat1: float, lon1: float,
#                             lat2: float, lon2: float) -> float:
#     """
#     Compute compass bearing (0–360°) from pickup to dropoff.

#     Parameters
#     ----------
#     lat1/lon1, lat2/lon2 : float — Coordinates in decimal degrees.

#     Returns
#     -------
#     float  — Bearing in degrees clockwise from North.
#     """
#     lat1r = np.radians(lat1);  lat2r = np.radians(lat2)
#     dlon  = np.radians(lon2 - lon1)
#     x     = np.sin(dlon) * np.cos(lat2r)
#     y     = (np.cos(lat1r) * np.sin(lat2r)
#              - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon))
#     return (np.degrees(np.arctan2(x, y)) + 360) % 360

# # ─────────────────────────────────────────────────────────────
# # LOAD ARTEFACTS
# # ─────────────────────────────────────────────────────────────
# best_model, scaler, selected_features, label_encoders, meta = load_models()
# df_raw = load_data()

# # ── Session state initialisation ────────────────────────────
# # Using st.session_state to persist lat/lon values so
# # coordinates entered on the Predict page automatically
# # pre-fill the Trip Map page — improves UX flow.
# for key, default in [
#     ("pickup_lat",  40.7128), ("pickup_lon", -74.0059),
#     ("dropoff_lat", 40.7580), ("dropoff_lon", -73.9855),
# ]:
#     if key not in st.session_state:
#         st.session_state[key] = default

# # ─────────────────────────────────────────────────────────────
# # SIDEBAR NAVIGATION
# # ─────────────────────────────────────────────────────────────
# st.sidebar.markdown(f"""
# <div style="text-align:center; padding: 20px 0 10px 0;">
#   <span style="font-family:'IBM Plex Mono',monospace;
#                font-size:2rem; color:{GOLD};">🚕 TripFare</span><br/>
#   <span style="color:{MUTED}; font-size:0.78rem;">
#     Urban Taxi Fare Prediction
#   </span>
# </div>
# """, unsafe_allow_html=True)

# st.sidebar.markdown("---")

# page = st.sidebar.radio(
#     "Navigation",
#     options=["🏠  Home", "📊  EDA Dashboard",
#              "🤖  Model Performance", "🎯  Predict Fare", "📍  Trip Map"],
#     label_visibility="collapsed"
# )

# # ── Sidebar model status card ────────────────────────────────
# st.sidebar.markdown("---")
# if meta:
#     st.sidebar.markdown(f"""
#     <div style="background:{BG_DARK}; border:1px solid #1F2937;
#                 border-radius:10px; padding:14px; font-size:0.8rem;">
#       <div style="color:{MUTED};">Best Model</div>
#       <div style="color:{GOLD}; font-weight:700; font-family:'IBM Plex Mono',monospace;">
#         {meta['best_model_name']}
#       </div>
#       <div style="color:{MUTED}; margin-top:8px;">
#         R² &nbsp;<span style="color:{GREEN};">{meta['test_r2']:.4f}</span> &nbsp;|&nbsp;
#         RMSE <span style="color:{RED};">${meta['test_rmse']:.2f}</span>
#       </div>
#     </div>
#     """, unsafe_allow_html=True)
# else:
#     st.sidebar.warning("⚠️ Models not found. Run model_training.py first.")

# # ═════════════════════════════════════════════════════════════
# # PAGE 1 — HOME
# # ═════════════════════════════════════════════════════════════
# if page == "🏠  Home":

#     # ── Hero Banner ──────────────────────────────────────────
#     st.markdown(f"""
#     <div class="hero-banner">
#       <div class="taxi-animate">🚕</div>
#       <h1 style="font-size:2.8rem; margin:12px 0 6px 0;">TripFare</h1>
#       <p style="color:{MUTED}; font-size:1.1rem; margin:0;">
#         Predicting Urban Taxi Fares with Machine Learning
#       </p>
#       <div style="margin-top:16px;">
#         <span style="background:#1a2240; border:1px solid {GOLD};
#                      border-radius:20px; padding:4px 14px; font-size:0.82rem;
#                      color:{GOLD}; margin:0 4px;">
#           Domain: Urban Transport Analytics
#         </span>
#         <span style="background:#1a2240; border:1px solid {GREEN};
#                      border-radius:20px; padding:4px 14px; font-size:0.82rem;
#                      color:{GREEN}; margin:0 4px;">
#           Problem: Supervised Regression
#         </span>
#         <span style="background:#1a2240; border:1px solid {BLUE};
#                      border-radius:20px; padding:4px 14px; font-size:0.82rem;
#                      color:{BLUE}; margin:0 4px;">
#           Target: total_amount ($)
#         </span>
#       </div>
#     </div>
#     """, unsafe_allow_html=True)

#     # ── KPI Metric Cards ─────────────────────────────────────
#     # Four at-a-glance numbers that summarise the project scope.
#     c1, c2, c3, c4 = st.columns(4)

#     total_trips  = f"{len(df_raw):,}" if not df_raw.empty else "N/A"
#     avg_fare     = f"${df_raw['total_amount'].mean():.2f}" if not df_raw.empty else "N/A"
#     best_r2      = f"{meta['test_r2']:.4f}" if meta else "N/A"
#     avg_dist_val = df_raw['trip_distance'].mean() if (
#         not df_raw.empty and 'trip_distance' in df_raw.columns
#     ) else None
#     avg_dist = f"{avg_dist_val:.2f} mi" if avg_dist_val else "N/A"

#     c1.metric("🚖 Total Trips Analysed", total_trips)
#     c2.metric("💵 Average Fare",         avg_fare)
#     c3.metric("🏆 Best Model R²",        best_r2)
#     c4.metric("📍 Avg Trip Distance",    avg_dist)

#     st.markdown("---")

#     # ── Dataset Overview ─────────────────────────────────────
#     col_info, col_pipeline = st.columns([1, 1])

#     with col_info:
#         st.markdown(f"### 📋 Dataset Overview")
#         if not df_raw.empty:
#             overview = pd.DataFrame({
#                 "Property" : ["Shape (rows × cols)", "Numeric columns",
#                               "Categorical columns", "Missing values"],
#                 "Value"    : [
#                     str(df_raw.shape),
#                     str(df_raw.select_dtypes(include="number").shape[1]),
#                     str(df_raw.select_dtypes(include="object").shape[1]),
#                     str(df_raw.isnull().sum().sum()),
#                 ]
#             })
#             st.dataframe(overview, hide_index=True, use_container_width=True)
#         else:
#             st.info("Run `model_training.py` to generate the dataset.")

#     with col_pipeline:
#         st.markdown(f"### ⚙️ How It Works")
#         steps = [
#             ("01", "Data Collection",     "Download NYC taxi CSV via gdown"),
#             ("02", "Feature Engineering", "Haversine distance, time flags, speed"),
#             ("03", "EDA & Transformation","Outlier removal, log transform, encode"),
#             ("04", "Model Training",       "8 models + hyperparameter tuning → save"),
#         ]
#         for num, title, desc in steps:
#             st.markdown(f"""
#             <div class="pipeline-step">
#               <span class="step-number">STEP {num}</span>
#               &nbsp;&nbsp;<strong>{title}</strong>
#               <div style="color:{MUTED}; font-size:0.85rem; margin-top:4px;">{desc}</div>
#             </div>
#             """, unsafe_allow_html=True)

#     st.markdown("---")

#     # ── Column Descriptions Table ─────────────────────────────
#     st.markdown("### 🗂️ Dataset Column Guide")
#     cols_desc = {
#         "VendorID"              : "Taxi provider ID (1 or 2)",
#         "pickup / dropoff"      : "GPS coordinates for trip start/end",
#         "passenger_count"       : "Number of passengers (1–6)",
#         "tpep_pickup_datetime"  : "Trip start timestamp (UTC → converted to EDT)",
#         "fare_amount"           : "Base metered fare (excluded from model — leakage)",
#         "extra"                 : "Rush-hour / overnight surcharge",
#         "mta_tax"               : "MTA flat tax ($0.50)",
#         "tip_amount"            : "Passenger tip",
#         "tolls_amount"          : "Bridge/tunnel tolls",
#         "improvement_surcharge" : "Flat surcharge ($0.30)",
#         "total_amount"          : "🎯 TARGET — sum of all fare components",
#         "payment_type"          : "1=Credit, 2=Cash, 3=No charge, 4=Dispute",
#         "RatecodeID"            : "1=Standard, 2=JFK, 3=Newark, 4=Nassau, 5=Negotiated",
#     }
#     st.dataframe(
#         pd.DataFrame(list(cols_desc.items()), columns=["Column", "Description"]),
#         hide_index=True, use_container_width=True
#     )

# # ═════════════════════════════════════════════════════════════
# # PAGE 2 — EDA DASHBOARD
# # ═════════════════════════════════════════════════════════════
# elif page == "📊  EDA Dashboard":

#     st.markdown(f"# 📊 Exploratory Data Analysis")
#     st.markdown(f"<p style='color:{MUTED};'>Interactive charts — hover, zoom, and filter to explore the data.</p>",
#                 unsafe_allow_html=True)

#     if df_raw.empty:
#         st.warning("No data found. Please run `model_training.py` first.")
#         st.stop()

#     # ── Sidebar Filters ───────────────────────────────────────
#     # Filters update all charts on this page simultaneously
#     # using Streamlit's reactive re-run mechanism.
#     st.sidebar.markdown("### 🔧 EDA Filters")
#     hour_range = st.sidebar.slider("Pickup Hour Range", 0, 23, (0, 23))
#     weekday_options = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
#     sel_weekdays = st.sidebar.multiselect(
#         "Weekday Filter", weekday_options, default=weekday_options
#     )
#     pax_max = st.sidebar.slider("Max Passenger Count", 1, 6, 6)

#     # Apply filters to create the EDA working dataframe
#     df_eda = df_raw.copy()
#     if "pickup_hour" in df_eda.columns:
#         df_eda = df_eda[df_eda["pickup_hour"].between(*hour_range)]
#     if "pickup_weekday" in df_eda.columns:
#         df_eda = df_eda[df_eda["pickup_weekday"].isin(sel_weekdays)]
#     if "passenger_count" in df_eda.columns:
#         df_eda = df_eda[df_eda["passenger_count"] <= pax_max]

#     st.markdown(f"<small style='color:{MUTED};'>Filtered dataset: {len(df_eda):,} trips</small>",
#                 unsafe_allow_html=True)

#     tab1, tab2, tab3, tab4 = st.tabs([
#         "📈 Distributions", "💵 Fare Analysis",
#         "🕒 Time Patterns", "🔎 Outliers & Correlation"
#     ])

#     # ── TAB 1: Distributions ──────────────────────────────────
#     with tab1:
#         col1, col2 = st.columns(2)

#         with col1:
#             # Histogram of total_amount — shows right-skew clearly
#             if "total_amount" in df_eda.columns:
#                 fig = px.histogram(
#                     df_eda[df_eda["total_amount"].between(0, 100)],
#                     x="total_amount", nbins=80,
#                     title="Fare Amount Distribution",
#                     color_discrete_sequence=[GOLD]
#                 )
#                 apply_plotly_theme(fig)
#                 st.plotly_chart(fig, use_container_width=True)
#                 st.caption("💡 Right-skewed fare distribution; log transform used during training.")

#         with col2:
#             if "trip_distance" in df_eda.columns:
#                 fig = px.histogram(
#                     df_eda[df_eda["trip_distance"].between(0, 25)],
#                     x="trip_distance", nbins=80,
#                     title="Trip Distance Distribution",
#                     color_discrete_sequence=[GREEN]
#                 )
#                 apply_plotly_theme(fig)
#                 st.plotly_chart(fig, use_container_width=True)
#                 st.caption("💡 Most trips are under 5 miles — short urban hops dominate.")

#         if "trip_duration_min" in df_eda.columns:
#             fig = px.histogram(
#                 df_eda[df_eda["trip_duration_min"].between(0, 90)],
#                 x="trip_duration_min", nbins=80,
#                 title="Trip Duration Distribution (minutes)",
#                 color_discrete_sequence=[BLUE]
#             )
#             apply_plotly_theme(fig)
#             st.plotly_chart(fig, use_container_width=True)
#             st.caption("💡 Median trip duration ~12 min; very long durations suggest outliers.")

#     # ── TAB 2: Fare Analysis ──────────────────────────────────
#     with tab2:
#         if "trip_distance" in df_eda.columns and "total_amount" in df_eda.columns:
#             # Sample to keep scatter fast and browser-friendly
#             sample_df = df_eda[
#                 df_eda["trip_distance"].between(0, 30) &
#                 df_eda["total_amount"].between(0, 100)
#             ].sample









































import json
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

warnings.filterwarnings("ignore")

# ============================================================
# THEME
# ============================================================
BG_DARK = "#0A0A14"
BG_CARD = "#111827"
BG_PANEL = "#0F172A"
GOLD = "#F7C948"
RED = "#E94560"
GREEN = "#2ECC71"
BLUE = "#3B82F6"
PURPLE = "#8B5CF6"
MUTED = "#94A3B8"
WHITE = "#F8FAFC"
GRID = "#243041"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_CARD,
    font=dict(color=WHITE, family="Inter, DM Sans, sans-serif"),
    title_font=dict(color=GOLD, size=16),
    legend=dict(
        bgcolor=BG_PANEL,
        bordercolor="#334155",
        borderwidth=1,
        font=dict(color=WHITE)
    ),
    margin=dict(l=40, r=20, t=55, b=40),
)

st.set_page_config(
    page_title="🚕 TripFare — Taxi Fare Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
        background: {BG_DARK} !important;
        color: {WHITE} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    [data-testid="stHeader"] {{
        background: rgba(10, 10, 20, 0.75);
    }}

    [data-testid="stSidebar"] {{
        background: {BG_CARD} !important;
        border-right: 1px solid #1F2937;
    }}

    [data-testid="stSidebar"] * {{
        color: {WHITE} !important;
    }}

    h1, h2, h3 {{
        color: {GOLD} !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }}

    [data-testid="stMetric"] {{
        background: linear-gradient(180deg, {BG_CARD}, {BG_PANEL});
        border: 1px solid #243041;
        border-radius: 14px;
        padding: 12px 16px;
    }}

    [data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {GOLD} !important;
    }}

    .stButton > button {{
        background: linear-gradient(135deg, {GOLD}, #E0A800);
        color: #111827 !important;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.6rem 1.2rem;
    }}

    .stButton > button:hover {{
        filter: brightness(0.96);
    }}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stDateInput > div > div,
    .stNumberInput > div > div {{
        background: {BG_CARD} !important;
        color: {WHITE} !important;
        border-color: #334155 !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: {BG_CARD};
        color: {MUTED};
        border-radius: 8px 8px 0 0;
        padding: 10px 14px;
    }}

    .stTabs [aria-selected="true"] {{
        color: {GOLD} !important;
        border-bottom: 2px solid {GOLD} !important;
    }}

    .hero {{
        background: linear-gradient(135deg, #0B1020 0%, #1E1B4B 50%, #0B1020 100%);
        border: 1px solid #243041;
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 20px;
    }}

    .hero h1 {{
        margin: 0;
        font-size: 2.4rem;
    }}

    .subtle {{
        color: {MUTED};
    }}

    .fare-card {{
        background: linear-gradient(135deg, #111827, #172554);
        border: 2px solid {GOLD};
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(247, 201, 72, 0.12);
    }}

    .fare-label {{
        color: {MUTED};
        font-size: 0.95rem;
        margin-bottom: 6px;
    }}

    .fare-value {{
        font-size: 3rem;
        font-weight: 800;
        color: {GOLD};
        font-family: 'IBM Plex Mono', monospace;
    }}

    .badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 8px;
        margin-top: 8px;
    }}

    .badge-gold {{
        background: rgba(247, 201, 72, 0.12);
        color: {GOLD};
        border: 1px solid rgba(247, 201, 72, 0.35);
    }}

    .badge-blue {{
        background: rgba(59, 130, 246, 0.12);
        color: {BLUE};
        border: 1px solid rgba(59, 130, 246, 0.35);
    }}

    .badge-green {{
        background: rgba(46, 204, 113, 0.12);
        color: {GREEN};
        border: 1px solid rgba(46, 204, 113, 0.35);
    }}

    hr {{
        border-color: #243041;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================
def apply_plotly_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zerolinecolor=GRID)
    return fig


def safe_sample(df, n=5000):
    if df is None or df.empty:
        return df
    n = min(n, len(df))
    if n <= 0:
        return df
    return df.sample(n=n, random_state=42)


def haversine_scalar(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(a))


def compute_bearing_scalar(lat1, lon1, lat2, lon2):
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def ensure_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def derive_time_features(df):
    candidate_pickup_cols = [
        "tpep_pickup_datetime",
        "pickup_datetime",
        "lpep_pickup_datetime",
    ]
    pickup_col = next((c for c in candidate_pickup_cols if c in df.columns), None)

    if pickup_col is not None:
        df = ensure_datetime(df, pickup_col)
        s = df[pickup_col]
        df["pickup_hour"] = s.dt.hour
        df["pickup_day"] = s.dt.day
        df["pickup_month"] = s.dt.month
        df["pickup_weekday_num"] = s.dt.weekday
        weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        df["pickup_weekday"] = df["pickup_weekday_num"].map(weekday_map)
        df["is_weekend"] = df["pickup_weekday_num"].isin([5, 6]).astype(int)
        df["pickup_part_of_day"] = pd.cut(
            df["pickup_hour"],
            bins=[-1, 5, 11, 17, 21, 24],
            labels=["Night", "Morning", "Afternoon", "Evening", "Late Night"],
            ordered=False,
        )

    candidate_drop_cols = [
        "tpep_dropoff_datetime",
        "dropoff_datetime",
        "lpep_dropoff_datetime",
    ]
    drop_col = next((c for c in candidate_drop_cols if c in df.columns), None)

    if pickup_col is not None and drop_col is not None:
        df = ensure_datetime(df, drop_col)
        duration = (df[drop_col] - df[pickup_col]).dt.total_seconds() / 60.0
        df["trip_duration_min"] = duration

    return df


def derive_geo_features(df):
    pickup_lat_cols = ["pickup_latitude", "PULatitude"]
    pickup_lon_cols = ["pickup_longitude", "PULongitude"]
    drop_lat_cols = ["dropoff_latitude", "DOLatitude"]
    drop_lon_cols = ["dropoff_longitude", "DOLongitude"]

    plat = next((c for c in pickup_lat_cols if c in df.columns), None)
    plon = next((c for c in pickup_lon_cols if c in df.columns), None)
    dlat = next((c for c in drop_lat_cols if c in df.columns), None)
    dlon = next((c for c in drop_lon_cols if c in df.columns), None)

    if plat and plon and dlat and dlon:
        mask = df[[plat, plon, dlat, dlon]].notna().all(axis=1)
        sub = df.loc[mask, [plat, plon, dlat, dlon]].copy()

        if "trip_distance_haversine" not in df.columns:
            R = 3958.8
            dlat_r = np.radians(sub[dlat] - sub[plat])
            dlon_r = np.radians(sub[dlon] - sub[plon])
            a = (
                np.sin(dlat_r / 2) ** 2
                + np.cos(np.radians(sub[plat]))
                * np.cos(np.radians(sub[dlat]))
                * np.sin(dlon_r / 2) ** 2
            )
            hav = R * 2 * np.arcsin(np.sqrt(a))
            df.loc[mask, "trip_distance_haversine"] = hav

        if "bearing" not in df.columns:
            lat1r = np.radians(sub[plat])
            lat2r = np.radians(sub[dlat])
            dlon_r = np.radians(sub[dlon] - sub[plon])
            x = np.sin(dlon_r) * np.cos(lat2r)
            y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon_r)
            df.loc[mask, "bearing"] = (np.degrees(np.arctan2(x, y)) + 360) % 360

    if "trip_distance" not in df.columns and "trip_distance_haversine" in df.columns:
        df["trip_distance"] = df["trip_distance_haversine"]

    if "trip_duration_min" in df.columns and "trip_distance" in df.columns:
        duration_hr = df["trip_duration_min"] / 60.0
        df["avg_speed_mph"] = np.where(duration_hr > 0, df["trip_distance"] / duration_hr, np.nan)

    return df


def clean_for_eda(df):
    data = df.copy()
    data = derive_time_features(data)
    data = derive_geo_features(data)

    for col in ["total_amount", "trip_distance", "trip_duration_min", "passenger_count"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


@st.cache_resource(show_spinner=False)
def load_models():
    try:
        best_model = joblib.load("models/best_model.pkl")
    except Exception:
        best_model = None

    try:
        scaler = joblib.load("models/scaler.pkl")
    except Exception:
        scaler = None

    try:
        selected_features = joblib.load("models/selected_features.pkl")
    except Exception:
        selected_features = None

    try:
        label_encoders = joblib.load("models/label_encoders.pkl")
    except Exception:
        label_encoders = None

    try:
        with open("models/meta.json", "r") as f:
            meta = json.load(f)
    except Exception:
        meta = None

    return best_model, scaler, selected_features, label_encoders, meta


@st.cache_data(show_spinner=False)
def load_data():
    candidate_paths = [
        "data/taxi_data.csv",
        "data/processed_taxi_data.csv",
        "taxi_data.csv",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            df = pd.read_csv(path, nrows=50000)
            return clean_for_eda(df)
    return pd.DataFrame()


def get_meta_value(meta, keys, default=None):
    if meta is None:
        return default
    for key in keys:
        if key in meta:
            return meta[key]
    return default


def encode_value(label_encoders, col, val):
    if label_encoders is None:
        return val
    if col not in label_encoders:
        return val
    enc = label_encoders[col]
    classes = list(enc.classes_)
    if val not in classes:
        return classes.index(classes[0]) if len(classes) > 0 else 0
    return enc.transform([val])[0]


def prepare_prediction_input(input_dict, selected_features, scaler=None, label_encoders=None):
    row = pd.DataFrame([input_dict])

    for col in row.columns:
        if row[col].dtype == "object":
            if label_encoders is not None and col in label_encoders:
                try:
                    row[col] = encode_value(label_encoders, col, row[col].iloc[0])
                except Exception:
                    row[col] = 0

    if selected_features is not None:
        for col in selected_features:
            if col not in row.columns:
                row[col] = 0
        row = row[selected_features]

    for col in row.columns:
        row[col] = pd.to_numeric(row[col], errors="coerce").fillna(0)

    if scaler is not None:
        try:
            arr = scaler.transform(row)
            row = pd.DataFrame(arr, columns=row.columns)
        except Exception:
            pass

    return row


def infer_model_table(meta):
    if meta is None:
        return pd.DataFrame()

    possible_keys = ["all_model_results","model_results", "results", "metrics", "all_model_metrics"]
    payload = None
    for k in possible_keys:
        if k in meta:
            payload = meta[k]
            break

    if payload is None:
        keys = [k for k in meta.keys() if isinstance(meta.get(k), dict)]
        rows = []
        for k in keys:
            d = meta[k]
            row = {"Model": k}
            for mk in ["train_r2", "test_r2", "rmse", "test_rmse", "mae", "test_mae"]:
                if mk in d:
                    row[mk] = d[mk]
            if len(row) > 1:
                rows.append(row)
        return pd.DataFrame(rows)

    if isinstance(payload, dict):
        rows = []
        for model_name, vals in payload.items():
            row = {"Model": model_name}
            if isinstance(vals, dict):
                row.update(vals)
            rows.append(row)
        return pd.DataFrame(rows)

    if isinstance(payload, list):
        return pd.DataFrame(payload)

    return pd.DataFrame()


best_model, scaler, selected_features, label_encoders, meta = load_models()
df_raw = load_data()

for key, default in [
    ("pickup_lat", 40.7128),
    ("pickup_lon", -74.0060),
    ("dropoff_lat", 40.7580),
    ("dropoff_lon", -73.9855),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(
    f"""
    <div style="text-align:center; padding: 8px 0 4px 0;">
        <div style="font-size:2rem;">🚕</div>
        <div style="font-family:'IBM Plex Mono', monospace; color:{GOLD}; font-size:1.6rem; font-weight:700;">TripFare</div>
        <div style="color:{MUTED}; font-size:0.85rem;">Theme-based Taxi Fare Dashboard</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 EDA Playground", "🤖 Model Performance", "🎯 Predict Fare", "🗺️ Trip Map"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
if meta:
    best_name = get_meta_value(meta, ["best_model_name", "best_model"], "Available")
    best_r2 = get_meta_value(meta, ["test_r2", "best_test_r2"], None)
    best_rmse = get_meta_value(meta, ["test_rmse", "rmse", "best_rmse"], None)

    st.sidebar.markdown(
        f"""
        <div style="background:{BG_PANEL}; border:1px solid #243041; border-radius:12px; padding:14px;">
            <div style="color:{MUTED}; font-size:0.8rem;">Best Model</div>
            <div style="color:{GOLD}; font-weight:800; font-size:1.05rem;">{best_name}</div>
            <div style="margin-top:8px; color:{WHITE}; font-size:0.82rem;">
                R²: <span style="color:{GREEN};">{best_r2 if best_r2 is not None else "N/A"}</span><br>
                RMSE: <span style="color:{RED};">{f"${best_rmse:.2f}" if isinstance(best_rmse, (int, float)) else "N/A"}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.info("Model metadata not found.")

# ============================================================
# PAGE: HOME
# ============================================================
if page == "🏠 Home":
    st.markdown(
        f"""
        <div class="hero">
            <h1>TripFare Dashboard</h1>
            <p class="subtle" style="font-size:1.05rem; margin-top:8px;">
                A complete Streamlit app for EDA, model review, fare prediction, and trip mapping.
            </p>
            <div style="margin-top:10px;">
                <span class="badge badge-gold">Urban Mobility</span>
                <span class="badge badge-blue">Regression ML</span>
                <span class="badge badge-green">Interactive Analytics</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    total_trips = f"{len(df_raw):,}" if not df_raw.empty else "N/A"
    avg_fare = f"${df_raw['total_amount'].mean():.2f}" if (not df_raw.empty and "total_amount" in df_raw.columns) else "N/A"
    avg_dist = f"{df_raw['trip_distance'].mean():.2f} mi" if (not df_raw.empty and "trip_distance" in df_raw.columns) else "N/A"
    best_r2 = get_meta_value(meta, ["test_r2", "best_test_r2"], "N/A")

    c1.metric("Trips Loaded", total_trips)
    c2.metric("Average Fare", avg_fare)
    c3.metric("Average Distance", avg_dist)
    c4.metric("Best Test R²", best_r2)

    st.markdown("---")

    a, b = st.columns([1.2, 1])

    with a:
        st.subheader("Dataset Snapshot")
        if df_raw.empty:
            st.warning("No dataset found in the data folder.")
        else:
            summary = pd.DataFrame({
                "Property": ["Rows", "Columns", "Numeric features", "Missing values"],
                "Value": [
                    len(df_raw),
                    df_raw.shape[1],
                    df_raw.select_dtypes(include=np.number).shape[1],
                    int(df_raw.isna().sum().sum()),
                ],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.dataframe(df_raw.head(8), use_container_width=True)

    with b:
        st.subheader("App Modules")
        st.markdown(
            """
            - **EDA Playground**: distributions, fare analysis, time patterns, outliers, correlations.
            - **Model Performance**: metrics table, model comparison charts, feature summary if available.
            - **Predict Fare**: interactive feature form and instant fare estimate.
            - **Trip Map**: pickup/dropoff visualization and route context.
            """
        )

# ============================================================
# PAGE: EDA
# ============================================================
elif page == "📊 EDA Playground":
    st.title("📊 EDA Playground")

    if df_raw.empty:
        st.warning("No data found. Put taxi_data.csv in the data folder.")
        st.stop()

    st.sidebar.markdown("### EDA Filters")

    hour_min, hour_max = 0, 23
    if "pickup_hour" in df_raw.columns and df_raw["pickup_hour"].notna().any():
        hour_min = int(df_raw["pickup_hour"].dropna().min())
        hour_max = int(df_raw["pickup_hour"].dropna().max())

    hour_range = st.sidebar.slider("Pickup Hour Range", 0, 23, (hour_min, hour_max))

    weekday_options = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sel_weekdays = st.sidebar.multiselect("Weekdays", weekday_options, default=weekday_options)

    max_pax_default = int(df_raw["passenger_count"].dropna().max()) if ("passenger_count" in df_raw.columns and df_raw["passenger_count"].notna().any()) else 6
    max_pax_default = max(1, min(max_pax_default, 10))
    pax_max = st.sidebar.slider("Max Passenger Count", 1, 10, max_pax_default)

    df_eda = df_raw.copy()

    if "pickup_hour" in df_eda.columns:
        df_eda = df_eda[df_eda["pickup_hour"].between(hour_range[0], hour_range[1], inclusive="both")]

    if "pickup_weekday" in df_eda.columns:
        df_eda = df_eda[df_eda["pickup_weekday"].isin(sel_weekdays)]

    if "passenger_count" in df_eda.columns:
        df_eda = df_eda[pd.to_numeric(df_eda["passenger_count"], errors="coerce").fillna(0) <= pax_max]

    st.caption(f"Filtered trips: {len(df_eda):,}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Distributions", "💵 Fare Analysis", "🕒 Time Patterns", "🔎 Outliers & Correlation"]
    )

    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            if "total_amount" in df_eda.columns:
                d = df_eda[df_eda["total_amount"].between(0, 150, inclusive="both")]
                fig = px.histogram(
                    d,
                    x="total_amount",
                    nbins=60,
                    title="Total Fare Distribution",
                    color_discrete_sequence=[GOLD],
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "trip_distance" in df_eda.columns:
                d = df_eda[df_eda["trip_distance"].between(0, 30, inclusive="both")]
                fig = px.histogram(
                    d,
                    x="trip_distance",
                    nbins=60,
                    title="Trip Distance Distribution",
                    color_discrete_sequence=[GREEN],
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            if "trip_duration_min" in df_eda.columns:
                d = df_eda[df_eda["trip_duration_min"].between(0, 120, inclusive="both")]
                fig = px.histogram(
                    d,
                    x="trip_duration_min",
                    nbins=60,
                    title="Trip Duration Distribution",
                    color_discrete_sequence=[BLUE],
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        with c4:
            if "passenger_count" in df_eda.columns:
                d = df_eda.copy()
                fig = px.histogram(
                    d,
                    x="passenger_count",
                    title="Passenger Count Distribution",
                    color_discrete_sequence=[PURPLE],
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            if {"trip_distance", "total_amount"}.issubset(df_eda.columns):
                d = df_eda[
                    df_eda["trip_distance"].between(0, 30, inclusive="both")
                    & df_eda["total_amount"].between(0, 150, inclusive="both")
                ].copy()
                d = safe_sample(d, 5000)
                fig = px.scatter(
                    d,
                    x="trip_distance",
                    y="total_amount",
                    opacity=0.55,
                    trendline="ols",
                    title="Fare vs Trip Distance",
                    color_discrete_sequence=[GOLD],
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if {"passenger_count", "total_amount"}.issubset(df_eda.columns):
                d = df_eda[df_eda["total_amount"].between(0, 150, inclusive="both")]
                fig = px.box(
                    d,
                    x="passenger_count",
                    y="total_amount",
                    title="Fare by Passenger Count",
                    color_discrete_sequence=[BLUE],
                    points=False,
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            if {"payment_type", "total_amount"}.issubset(df_eda.columns):
                d = (
                    df_eda.groupby("payment_type", dropna=False)["total_amount"]
                    .mean()
                    .reset_index()
                    .sort_values("total_amount", ascending=False)
                )
                fig = px.bar(
                    d,
                    x="payment_type",
                    y="total_amount",
                    title="Average Fare by Payment Type",
                    color="total_amount",
                    color_continuous_scale="YlOrBr",
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        with c4:
            if {"VendorID", "total_amount"}.issubset(df_eda.columns):
                d = (
                    df_eda.groupby("VendorID", dropna=False)["total_amount"]
                    .mean()
                    .reset_index()
                    .sort_values("total_amount", ascending=False)
                )
                fig = px.bar(
                    d,
                    x="VendorID",
                    y="total_amount",
                    title="Average Fare by Vendor",
                    color="total_amount",
                    color_continuous_scale="Blues",
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)

        with c1:
            if {"pickup_hour", "total_amount"}.issubset(df_eda.columns):
                d = (
                    df_eda.groupby("pickup_hour")["total_amount"]
                    .agg(["mean", "count"])
                    .reset_index()
                )
                fig = px.line(
                    d,
                    x="pickup_hour",
                    y="mean",
                    markers=True,
                    title="Average Fare by Pickup Hour",
                )
                fig.update_traces(line_color=GOLD)
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if {"pickup_hour"}.issubset(df_eda.columns):
                d = df_eda.groupby("pickup_hour").size().reset_index(name="trips")
                fig = px.bar(
                    d,
                    x="pickup_hour",
                    y="trips",
                    title="Trip Volume by Pickup Hour",
                    color="trips",
                    color_continuous_scale="Tealgrn",
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            if {"pickup_weekday", "total_amount"}.issubset(df_eda.columns):
                order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                d = (
                    df_eda.groupby("pickup_weekday")["total_amount"]
                    .mean()
                    .reindex(order)
                    .reset_index()
                )
                fig = px.bar(
                    d,
                    x="pickup_weekday",
                    y="total_amount",
                    title="Average Fare by Weekday",
                    color="total_amount",
                    color_continuous_scale="Sunset",
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        with c4:
            if {"pickup_weekday", "pickup_hour", "total_amount"}.issubset(df_eda.columns):
                order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                d = (
                    df_eda.pivot_table(
                        index="pickup_weekday",
                        columns="pickup_hour",
                        values="total_amount",
                        aggfunc="mean",
                    )
                    .reindex(order)
                )
                fig = px.imshow(
                    d,
                    aspect="auto",
                    color_continuous_scale="YlOrBr",
                    title="Fare Heatmap: Weekday vs Pickup Hour",
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)

        with c1:
            if {"trip_distance", "total_amount"}.issubset(df_eda.columns):
                d = df_eda[
                    df_eda["trip_distance"].between(0, 40, inclusive="both")
                    & df_eda["total_amount"].between(0, 200, inclusive="both")
                ]
                fig = px.box(
                    d,
                    y="total_amount",
                    title="Fare Outlier View",
                    color_discrete_sequence=[RED],
                    points="suspectedoutliers",
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            numeric_cols = df_eda.select_dtypes(include=np.number).columns.tolist()
            corr_cols = [c for c in numeric_cols if c in [
                "total_amount", "trip_distance", "trip_duration_min", "passenger_count",
                "pickup_hour", "pickup_month", "is_weekend", "avg_speed_mph", "bearing"
            ]]
            corr_cols = [c for c in corr_cols if df_eda[c].notna().sum() > 3]

            if len(corr_cols) >= 2:
                corr = df_eda[corr_cols].corr(numeric_only=True)
                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="Correlation Matrix",
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        if {"trip_distance", "trip_duration_min", "avg_speed_mph"}.issubset(df_eda.columns):
            d = df_eda[
                df_eda["trip_distance"].between(0, 30, inclusive="both")
                & df_eda["trip_duration_min"].between(0, 120, inclusive="both")
                & df_eda["avg_speed_mph"].between(0, 80, inclusive="both")
            ]
            d = safe_sample(d, 4000)
            fig = px.scatter(
                d,
                x="trip_duration_min",
                y="trip_distance",
                color="avg_speed_mph",
                title="Distance vs Duration colored by Average Speed",
                color_continuous_scale="Turbo",
                opacity=0.6,
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    if meta is None:
        st.warning("meta.json not found. Add training metadata in models/meta.json.")
        st.stop()

    best_name = get_meta_value(meta, ["best_model_name", "best_model"], "N/A")
    test_r2 = get_meta_value(meta, ["test_r2", "best_test_r2"], None)
    test_rmse = get_meta_value(meta, ["test_rmse", "rmse", "best_rmse"], None)
    test_mae = get_meta_value(meta, ["test_mae", "mae", "best_mae"], None)
    train_r2 = get_meta_value(meta, ["train_r2"], None)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_name)
    c2.metric("Test R²", f"{test_r2:.4f}" if isinstance(test_r2, (float, int)) else "N/A")
    c3.metric("Test RMSE", f"${test_rmse:.2f}" if isinstance(test_rmse, (float, int)) else "N/A")
    c4.metric("Test MAE", f"${test_mae:.2f}" if isinstance(test_mae, (float, int)) else "N/A")

    st.markdown("---")

    model_df = infer_model_table(meta)

    if not model_df.empty:
        st.subheader("Model Metrics Table")
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        candidate_r2_cols = [c for c in model_df.columns if "r2" in c.lower()]
        candidate_rmse_cols = [c for c in model_df.columns if "rmse" in c.lower()]
        candidate_mae_cols = [c for c in model_df.columns if "mae" in c.lower()]

        c1, c2 = st.columns(2)

        with c1:
            if "Model" in model_df.columns and candidate_r2_cols:
                r2_col = candidate_r2_cols[-1]
                fig = px.bar(
                    model_df.sort_values(r2_col, ascending=False),
                    x="Model",
                    y=r2_col,
                    color=r2_col,
                    title=f"Model Comparison by {r2_col}",
                    color_continuous_scale="Viridis",
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "Model" in model_df.columns and candidate_rmse_cols:
                rmse_col = candidate_rmse_cols[-1]
                fig = px.bar(
                    model_df.sort_values(rmse_col, ascending=True),
                    x="Model",
                    y=rmse_col,
                    color=rmse_col,
                    title=f"Model Comparison by {rmse_col}",
                    color_continuous_scale="Reds",
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        if "Model" in model_df.columns and candidate_r2_cols and candidate_rmse_cols:
            r2_col = candidate_r2_cols[-1]
            rmse_col = candidate_rmse_cols[-1]
            fig = px.scatter(
                model_df,
                x=rmse_col,
                y=r2_col,
                text="Model",
                size_max=20,
                title="R² vs RMSE",
                color="Model",
            )
            fig.update_traces(textposition="top center")
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Structured model comparison data not found in meta.json.")

    if selected_features is not None:
        st.subheader("Selected Features")
        feat_df = pd.DataFrame({"Feature": list(selected_features)})
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ============================================================
# PAGE: PREDICT FARE
# ============================================================
elif page == "🎯 Predict Fare":
    st.title("🎯 Predict Fare")

    if best_model is None:
        st.warning("best_model.pkl not found in models folder.")
        st.stop()

    st.markdown("Enter trip details to estimate taxi fare.")

    left, right = st.columns([1.2, 1])

    with left:
        c1, c2, c3 = st.columns(3)
        with c1:
            passenger_count = st.number_input("Passenger Count", min_value=1, max_value=8, value=1, step=1)
        with c2:
            vendor_id = st.selectbox("Vendor ID", [1, 2], index=0)
        with c3:
            payment_type = st.selectbox("Payment Type", [1, 2, 3, 4], index=0)

        st.markdown("### Pickup / Dropoff")
        c4, c5 = st.columns(2)

        with c4:
            pickup_lat = st.number_input("Pickup Latitude", value=float(st.session_state["pickup_lat"]), format="%.6f")
            pickup_lon = st.number_input("Pickup Longitude", value=float(st.session_state["pickup_lon"]), format="%.6f")

        with c5:
            dropoff_lat = st.number_input("Dropoff Latitude", value=float(st.session_state["dropoff_lat"]), format="%.6f")
            dropoff_lon = st.number_input("Dropoff Longitude", value=float(st.session_state["dropoff_lon"]), format="%.6f")

        st.session_state["pickup_lat"] = pickup_lat
        st.session_state["pickup_lon"] = pickup_lon
        st.session_state["dropoff_lat"] = dropoff_lat
        st.session_state["dropoff_lon"] = dropoff_lon

        c6, c7 = st.columns(2)
        with c6:
            pickup_date = st.date_input("Pickup Date", value=datetime.now().date())
        with c7:
            pickup_time = st.time_input("Pickup Time", value=datetime.now().time())

        dt = datetime.combine(pickup_date, pickup_time)
        pickup_hour = dt.hour
        pickup_month = dt.month
        pickup_weekday_num = dt.weekday()
        is_weekend = int(pickup_weekday_num >= 5)

        trip_distance = haversine_scalar(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
        bearing = compute_bearing_scalar(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

        st.markdown("### Engineered Features")
        e1, e2, e3 = st.columns(3)
        e1.metric("Estimated Distance", f"{trip_distance:.2f} mi")
        e2.metric("Bearing", f"{bearing:.1f}°")
        e3.metric("Pickup Hour", pickup_hour)

        predict_btn = st.button("Predict Fare", use_container_width=True)

    with right:
        st.markdown(
            """
            <div class="fare-card">
                <div class="fare-label">Prediction output will appear here</div>
                <div class="fare-value">--</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if predict_btn:
        input_dict = {
            "VendorID": vendor_id,
            "passenger_count": passenger_count,
            "payment_type": payment_type,
            "pickup_latitude": pickup_lat,
            "pickup_longitude": pickup_lon,
            "dropoff_latitude": dropoff_lat,
            "dropoff_longitude": dropoff_lon,
            "trip_distance": trip_distance,
            "trip_distance_haversine": trip_distance,
            "bearing": bearing,
            "pickup_hour": pickup_hour,
            "pickup_month": pickup_month,
            "pickup_weekday_num": pickup_weekday_num,
            "is_weekend": is_weekend,
        }

        X = prepare_prediction_input(
            input_dict=input_dict,
            selected_features=selected_features,
            scaler=scaler,
            label_encoders=label_encoders,
        )

        try:
            pred = float(best_model.predict(X)[0])
            pred = max(0.0, pred)

            st.markdown("---")
            st.markdown(
                f"""
                <div class="fare-card">
                    <div class="fare-label">Predicted Taxi Fare</div>
                    <div class="fare-value">${pred:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            breakdown = pd.DataFrame({
                "Feature": ["Trip distance", "Pickup hour", "Weekend", "Passenger count", "Payment type"],
                "Value": [round(trip_distance, 2), pickup_hour, is_weekend, passenger_count, payment_type]
            })
            st.dataframe(breakdown, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ============================================================
# PAGE: MAP
# ============================================================
elif page == "🗺️ Trip Map":
    st.title("🗺️ Trip Map")

    pickup_lat = float(st.session_state["pickup_lat"])
    pickup_lon = float(st.session_state["pickup_lon"])
    dropoff_lat = float(st.session_state["dropoff_lat"])
    dropoff_lon = float(st.session_state["dropoff_lon"])

    center_lat = (pickup_lat + dropoff_lat) / 2
    center_lon = (pickup_lon + dropoff_lon) / 2
    trip_distance = haversine_scalar(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    c1, c2, c3 = st.columns(3)
    c1.metric("Pickup", f"{pickup_lat:.4f}, {pickup_lon:.4f}")
    c2.metric("Dropoff", f"{dropoff_lat:.4f}, {dropoff_lon:.4f}")
    c3.metric("Estimated Distance", f"{trip_distance:.2f} mi")

    if FOLIUM_AVAILABLE:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB dark_matter")
        folium.Marker(
            [pickup_lat, pickup_lon],
            tooltip="Pickup",
            popup="Pickup Location",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(m)
        folium.Marker(
            [dropoff_lat, dropoff_lon],
            tooltip="Dropoff",
            popup="Dropoff Location",
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(m)

        folium.PolyLine(
            locations=[[pickup_lat, pickup_lon], [dropoff_lat, dropoff_lon]],
            color="#F7C948",
            weight=5,
            opacity=0.85,
        ).add_to(m)

        st_folium(m, width=None, height=520)
    else:
        st.info("folium/streamlit-folium not installed, showing Plotly fallback map.")

        map_df = pd.DataFrame({
            "lat": [pickup_lat, dropoff_lat],
            "lon": [pickup_lon, dropoff_lon],
            "point": ["Pickup", "Dropoff"],
            "size": [18, 18],
        })

        fig = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            color="point",
            size="size",
            zoom=11,
            height=520,
            title="Trip Points",
            color_discrete_map={"Pickup": GREEN, "Dropoff": RED},
        )
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            paper_bgcolor=BG_CARD,
            plot_bgcolor=BG_CARD,
            font=dict(color=WHITE),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        fig.add_trace(
            go.Scattermapbox(
                lat=[pickup_lat, dropoff_lat],
                lon=[pickup_lon, dropoff_lon],
                mode="lines",
                line=dict(width=4, color=GOLD),
                name="Route Line",
            )
        )
        st.plotly_chart(fig, use_container_width=True)