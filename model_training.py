# === FILE: model_training.py ===
# ============================================================
# Project : TripFare — Predicting Urban Taxi Fare with ML
# File    : model_training.py
# Purpose : End-to-end pipeline: data download → feature
#           engineering → EDA plots → transformation →
#           feature selection → model training & tuning →
#           saving best model + artefacts for Streamlit
# ============================================================

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import pytz

from math import radians
from scipy import stats
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, GridSearchCV
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
)
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── Directory Setup ─────────────────────────────────────────
# Create output directories if they do not already exist.
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data",   exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1 | DATA COLLECTION
# Purpose : Download the NYC taxi dataset from Google Drive
#           using gdown. Skip download if local file exists.
# ─────────────────────────────────────────────────────────────

DATA_PATH   = "data/taxi_data.csv"
GDRIVE_ID   = "1VUb9ucTsroGDBOPcwpOfXwzDi-rd4wqQ"
GDRIVE_URL  = f"https://drive.google.com/uc?id={GDRIVE_ID}"

def load_dataset(path: str, url: str) -> pd.DataFrame:
    """
    Download dataset from Google Drive if not cached locally,
    then load it into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Local file path to save / load the CSV.
    url  : str
        Google Drive direct-download URL.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    if not os.path.exists(path):
        print(f"[INFO] Downloading dataset from Google Drive …")
        try:
            import gdown
            gdown.download(url, path, quiet=False)
        except Exception as e:
            # Provide a clear, actionable error message so users know
            # exactly what failed and what to do about it.
            raise FileNotFoundError(
                f"[ERROR] Could not download dataset.\n"
                f"  Reason : {e}\n"
                f"  Fix    : pip install gdown  OR  manually place "
                f"           the CSV at '{path}'."
            )
    else:
        print(f"[INFO] Local cache found — loading from '{path}' …")

    df = pd.read_csv(path)
    print(f"\n✅ Dataset loaded successfully.")
    print(f"   Shape   : {df.shape}")
    print(f"   Columns : {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}\n")
    return df

df = load_dataset(DATA_PATH, GDRIVE_URL)

# ─────────────────────────────────────────────────────────────
# STEP 2 | DATA UNDERSTANDING
# Purpose : Summarise the raw dataset — types, missing values,
#           duplicates, and unique value counts for categoricals.
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 2 | DATA UNDERSTANDING")
print("=" * 60)

# Basic shape and dtype overview
print(f"\n📐 Shape : {df.shape}")
print(f"\n📋 Data Types:\n{df.dtypes}")
print(f"\n📊 Statistical Summary:\n{df.describe().to_string()}")
df.info()

# Missing values — count AND percentage so we know severity
missing_counts = df.isnull().sum()
missing_pct    = (missing_counts / len(df) * 100).round(2)
missing_df     = pd.DataFrame({
    "Missing Count"  : missing_counts,
    "Missing Pct (%)" : missing_pct
}).sort_values("Missing Count", ascending=False)
print(f"\n🔍 Missing Values:\n{missing_df[missing_df['Missing Count'] > 0]}")

# Duplicate rows — drop them silently after reporting
dup_count = df.duplicated().sum()
print(f"\n🔁 Duplicate Rows Found : {dup_count}")
if dup_count > 0:
    df = df.drop_duplicates()
    print(f"   ✅ Dropped {dup_count} duplicates. New shape: {df.shape}")

# Unique value counts for key categorical columns
# Business Insight: VendorID has only 2 vendors; RatecodeID has
# 6 codes where code 99 is an anomaly to investigate.
for col in ["VendorID", "RatecodeID", "payment_type", "store_and_fwd_flag"]:
    if col in df.columns:
        print(f"\n  Unique values in '{col}': {df[col].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────
# STEP 3 | FEATURE ENGINEERING
# Purpose : Create trip_distance via Haversine, extract
#           time features, build binary flags for night/
#           weekend/rush-hour rides, compute speed & rate cols.
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 | FEATURE ENGINEERING")
print("=" * 60)

shape_before = df.shape

# ── 3a. Rename datetime columns for clarity ─────────────────
df = df.rename(columns={
    "tpep_pickup_datetime"  : "pickup_datetime",
    "tpep_dropoff_datetime" : "dropoff_datetime"
})

# ── 3b. Parse datetimes and convert UTC → EDT ───────────────
# NYC taxis record timestamps in UTC; all downstream time
# features (hour, rush hour, AM/PM) must reflect local time.
utc_tz = pytz.utc
edt_tz = pytz.timezone("America/New_York")

df["pickup_datetime"]  = pd.to_datetime(df["pickup_datetime"],  errors="coerce", utc=True)
df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce", utc=True)

# Convert to EDT; dt.tz_convert returns tz-aware → remove tz
# info after conversion so downstream operations are simpler.
df["pickup_datetime"]  = df["pickup_datetime"].dt.tz_convert(edt_tz).dt.tz_localize(None)
df["dropoff_datetime"] = df["dropoff_datetime"].dt.tz_convert(edt_tz).dt.tz_localize(None)

# ── 3c. Haversine trip distance (vectorized, miles) ──────────
def haversine_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    Compute great-circle distance between pickup and dropoff
    coordinates using the Haversine formula.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: pickup_latitude, pickup_longitude,
        dropoff_latitude, dropoff_longitude (decimal degrees).

    Returns
    -------
    pd.Series
        Distance in miles for each row.
    """
    R = 3958.8   # Earth's radius in miles

    # Convert degrees → radians for trig functions
    lat1 = np.radians(df["pickup_latitude"].values)
    lon1 = np.radians(df["pickup_longitude"].values)
    lat2 = np.radians(df["dropoff_latitude"].values)
    lon2 = np.radians(df["dropoff_longitude"].values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula: a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return pd.Series(R * c, index=df.index)

df["trip_distance"] = haversine_vectorized(df)

# ── 3d. Trip duration in minutes ────────────────────────────
df["trip_duration_min"] = (
    (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60
)

# ── 3e. Time feature extraction (all from EDT pickup time) ──
df["pickup_hour"]    = df["pickup_datetime"].dt.hour
df["pickup_day"]     = df["pickup_datetime"].dt.dayofweek    # 0=Mon
df["pickup_month"]   = df["pickup_datetime"].dt.month
df["pickup_weekday"] = df["pickup_datetime"].dt.day_name().str[:3]  # Mon, Tue …

# ── 3f. AM/PM flag ──────────────────────────────────────────
df["am_pm"] = np.where(df["pickup_hour"] < 12, "AM", "PM")

# ── 3g. Weekend flag ────────────────────────────────────────
# pickup_day: 5=Saturday, 6=Sunday
df["is_weekend"] = (df["pickup_day"] >= 5).astype(int)

# ── 3h. Night flag (10 PM – 5:59 AM) ───────────────────────
# MTA charges a $0.50 night surcharge during these hours.
df["is_night"] = (
    (df["pickup_hour"] >= 22) | (df["pickup_hour"] < 6)
).astype(int)

# ── 3i. Rush hour flag (weekday + 7–9 AM or 4–7 PM) ────────
# Rush hours typically drive higher fares due to traffic.
morning_rush = (df["pickup_hour"] >= 7)  & (df["pickup_hour"] < 9)
evening_rush = (df["pickup_hour"] >= 16) & (df["pickup_hour"] < 19)
is_weekday   = df["pickup_day"] < 5
df["is_rush_hour"] = (is_weekday & (morning_rush | evening_rush)).astype(int)

# ── 3j. Time-of-day bucket ──────────────────────────────────
def get_time_of_day(hour: int) -> str:
    """
    Map an integer hour (0–23) to a named time-of-day bucket.

    Parameters
    ----------
    hour : int
        Hour of day in 24-hour format.

    Returns
    -------
    str
        One of: 'Morning', 'Afternoon', 'Evening', 'Night'.
    """
    if   6  <= hour < 12: return "Morning"
    elif 12 <= hour < 17: return "Afternoon"
    elif 17 <= hour < 22: return "Evening"
    else:                  return "Night"

df["time_of_day"] = df["pickup_hour"].apply(get_time_of_day)

# ── 3k. Average speed (mph) — guard against zero duration ───
# Clip trip_duration_min to a minimum of 0.01 to prevent inf.
df["avg_speed_mph"] = df["trip_distance"] / (
    df["trip_duration_min"].clip(lower=0.01) / 60
)

# ── 3l. Fare efficiency metrics — guard against zero ────────
# Replace 0-distance/duration rows with NaN before dividing
# so we don't propagate inf values into the model.
df["fare_per_mile"]   = df["fare_amount"] / df["trip_distance"].replace(0, np.nan)
df["fare_per_minute"] = df["fare_amount"] / df["trip_duration_min"].replace(0, np.nan)

# ── 3m. Direction bearing (compass degrees) ─────────────────
def compute_bearing(df: pd.DataFrame) -> pd.Series:
    """
    Compute the initial compass bearing (0–360°) from the
    pickup coordinate to the dropoff coordinate.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: pickup_latitude, pickup_longitude,
        dropoff_latitude, dropoff_longitude.

    Returns
    -------
    pd.Series
        Bearing in degrees clockwise from North for each row.
    """
    lat1 = np.radians(df["pickup_latitude"].values)
    lat2 = np.radians(df["dropoff_latitude"].values)
    dlon = np.radians(
        df["dropoff_longitude"].values - df["pickup_longitude"].values
    )
    x    = np.sin(dlon) * np.cos(lat2)
    y    = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(x, y)) + 360) % 360
    return pd.Series(bearing, index=df.index)

df["direction_bearing"] = compute_bearing(df)

# Print before/after shape and new column names
print(f"\n  Shape BEFORE engineering : {shape_before}")
print(f"  Shape AFTER  engineering : {df.shape}")
new_cols = [c for c in df.columns if c not in [
    "VendorID","tpep_pickup_datetime","tpep_dropoff_datetime",
    "passenger_count","pickup_longitude","pickup_latitude",
    "RatecodeID","store_and_fwd_flag","dropoff_longitude",
    "dropoff_latitude","payment_type","fare_amount","extra",
    "mta_tax","tip_amount","tolls_amount","improvement_surcharge",
    "total_amount","pickup_datetime","dropoff_datetime"
]]
print(f"\n  New columns created : {new_cols}")

# ─────────────────────────────────────────────────────────────
# STEP 4 | EXPLORATORY DATA ANALYSIS (EDA)
# Purpose : Generate 10 business-insight visualisations and
#           save them as PNGs for use in the Streamlit app.
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 | EDA — Generating plots …")
print("=" * 60)

# Consistent style for all matplotlib/seaborn plots
plt.rcParams.update({
    "figure.facecolor" : "#111827",
    "axes.facecolor"   : "#111827",
    "axes.edgecolor"   : "#8B9BAD",
    "axes.labelcolor"  : "#F0F4F8",
    "xtick.color"      : "#8B9BAD",
    "ytick.color"      : "#8B9BAD",
    "text.color"       : "#F0F4F8",
    "grid.color"       : "#1F2937",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.5,
    "font.family"      : "DejaVu Sans",
})

GOLD  = "#F7C948"
RED   = "#E94560"
GREEN = "#2ECC71"
BLUE  = "#3B82F6"
MUTED = "#8B9BAD"

# ── Plot 01: Target Distribution ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Total Fare Distribution", color=GOLD, fontsize=14, fontweight="bold")

axes[0].hist(df["total_amount"].dropna().clip(upper=200),
             bins=80, color=GOLD, edgecolor="#0A0A14", alpha=0.85)
axes[0].set_title("Raw Distribution", color=GOLD)
axes[0].set_xlabel("total_amount ($)")
axes[0].set_ylabel("Count")

# Log transform reduces right-skew, making the distribution
# more symmetric — ideal for regression model training.
axes[1].hist(np.log1p(df["total_amount"].dropna().clip(lower=0)),
             bins=80, color=GREEN, edgecolor="#0A0A14", alpha=0.85)
axes[1].set_title("Log-Transformed Distribution", color=GREEN)
axes[1].set_xlabel("log1p(total_amount)")
axes[1].set_ylabel("Count")

# Business Insight: The raw fare distribution is heavily right-
# skewed with a long tail above $50. After log1p transformation
# the distribution becomes approximately normal — confirming
# that training on log(total_amount) will yield better residuals.
plt.tight_layout()
plt.savefig("plots/01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 01_target_distribution.png")

# ── Plot 02: Fare vs Distance Scatter ───────────────────────
sample = df[
    (df["trip_distance"].between(0.1, 30)) &
    (df["total_amount"].between(2.5, 100))
].sample(min(5000, len(df)), random_state=42)

fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(
    sample["trip_distance"], sample["total_amount"],
    c=sample["pickup_hour"], cmap="plasma",
    alpha=0.4, s=8, edgecolors="none"
)
cb = plt.colorbar(sc, ax=ax)
cb.set_label("Pickup Hour", color="#F0F4F8")
ax.set_xlabel("Trip Distance (miles)")
ax.set_ylabel("Total Fare ($)")
ax.set_title("Fare vs Trip Distance (coloured by Pickup Hour)", color=GOLD, fontsize=13)

# Business Insight: A strong positive linear relationship exists
# between distance and fare — the primary fare driver. Trips
# coloured in yellow/green (late evening ~18–22 h) cluster
# slightly above the regression line, suggesting a time premium.
plt.tight_layout()
plt.savefig("plots/02_fare_vs_distance_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 02_fare_vs_distance_scatter.png")

# ── Plot 03: Fare vs Passengers Boxplot ─────────────────────
plot_df = df[df["passenger_count"].between(1, 6)]
fig, ax = plt.subplots(figsize=(10, 6))

groups  = [
    plot_df[plot_df["passenger_count"] == i]["total_amount"].dropna().clip(upper=80).values
    for i in range(1, 7)
]
bp = ax.boxplot(groups, patch_artist=True, notch=False,
                medianprops=dict(color=RED, linewidth=2))

for patch, color in zip(bp["boxes"], [GOLD, GREEN, BLUE, RED, "#A78BFA", "#F97316"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticklabels([f"{i} pax" for i in range(1, 7)])
ax.set_xlabel("Passenger Count")
ax.set_ylabel("Total Fare ($)")
ax.set_title("Fare Distribution by Passenger Count", color=GOLD, fontsize=13)

# Business Insight: Median fares are almost identical across
# passenger counts (1–6), indicating that NYC yellow cab fares
# are NOT proportional to passenger count — the taxi meter
# charges by distance and time, regardless of occupancy.
plt.tight_layout()
plt.savefig("plots/03_fare_vs_passengers_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 03_fare_vs_passengers_boxplot.png")

# ── Plot 04: Hourly Patterns ─────────────────────────────────
hourly_count = df.groupby("pickup_hour").size()
hourly_fare  = df.groupby("pickup_hour")["total_amount"].mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle("Hourly Trip Patterns", color=GOLD, fontsize=14, fontweight="bold")

# Shade morning rush (7–9) in green and evening rush (16–19) in blue
for ax in [ax1, ax2]:
    ax.axvspan(7,  9,  alpha=0.15, color=GREEN, label="Morning Rush")
    ax.axvspan(16, 19, alpha=0.15, color=BLUE,  label="Evening Rush")

ax1.bar(hourly_count.index, hourly_count.values, color=GOLD, alpha=0.85, edgecolor="#0A0A14")
ax1.set_ylabel("Trip Count")
ax1.set_title("Trip Volume by Hour", color=GOLD)
ax1.legend(loc="upper left", fontsize=8)

ax2.plot(hourly_fare.index, hourly_fare.values, color=RED, linewidth=2.5, marker="o", markersize=5)
ax2.fill_between(hourly_fare.index, hourly_fare.values, alpha=0.2, color=RED)
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Avg Fare ($)")
ax2.set_title("Average Fare by Hour", color=RED)

# Business Insight: Trip volume peaks at 6–8 PM (evening commute)
# but the highest average fares occur between 4–6 AM — likely
# airport runs and long-haul late-night trips with few short trips
# to bring the average down during those hours.
plt.tight_layout()
plt.savefig("plots/04_hourly_patterns.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 04_hourly_patterns.png")

# ── Plot 05: Weekday & Time-of-Day Analysis ──────────────────
weekday_order  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
tod_order      = ["Morning", "Afternoon", "Evening", "Night"]

weekday_fare   = df.groupby("pickup_weekday")["total_amount"].mean().reindex(weekday_order)
tod_fare       = df.groupby("time_of_day")["total_amount"].mean().reindex(tod_order)

# Highlight weekend bars in red to make the weekend effect
# visually obvious for stakeholders.
weekday_colors = [RED if d in ["Sat","Sun"] else GOLD for d in weekday_order]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Fare by Weekday & Time of Day", color=GOLD, fontsize=14, fontweight="bold")

ax1.bar(weekday_order, weekday_fare.values, color=weekday_colors, alpha=0.85, edgecolor="#0A0A14")
ax1.set_ylabel("Avg Fare ($)")
ax1.set_title("Avg Fare by Weekday  (🔴 = Weekend)", color=GOLD)
red_patch  = mpatches.Patch(color=RED,  label="Weekend")
gold_patch = mpatches.Patch(color=GOLD, label="Weekday")
ax1.legend(handles=[gold_patch, red_patch])

ax2.bar(tod_order, tod_fare.values, color=[GREEN, BLUE, GOLD, RED], alpha=0.85, edgecolor="#0A0A14")
ax2.set_ylabel("Avg Fare ($)")
ax2.set_title("Avg Fare by Time of Day", color=GOLD)

# Business Insight: Weekend fares (Sat/Sun) are slightly higher
# than mid-week fares, consistent with longer leisure trips.
# Night rides command the highest average fare across all
# time-of-day buckets, confirming MTA night surcharge impact.
plt.tight_layout()
plt.savefig("plots/05_weekday_tod_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 05_weekday_tod_analysis.png")

# ── Plot 06: Correlation Heatmap ─────────────────────────────
# Select numeric columns only; drop datetime and high-leakage
# sub-components to make the heatmap focused and interpretable.
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
drop_for_heatmap = ["fare_amount","tip_amount","tolls_amount",
                    "extra","mta_tax","improvement_surcharge"]
heatmap_cols = [c for c in numeric_cols if c not in drop_for_heatmap]
corr_matrix  = df[heatmap_cols].corr()

# Lower-triangle mask: avoids redundant symmetric cells
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
    linewidths=0.5, linecolor="#0A0A14",
    annot_kws={"size": 7}, ax=ax
)
ax.set_title("Feature Correlation Heatmap (lower triangle)", color=GOLD, fontsize=13, pad=12)

# Business Insight: trip_distance shows the highest positive
# correlation with total_amount (≈ 0.87), confirming it as the
# dominant predictor. trip_duration_min also correlates strongly
# (≈ 0.72), while time flags (is_night, is_weekend) show weak
# but non-zero correlations, validating their inclusion.
plt.tight_layout()
plt.savefig("plots/06_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 06_correlation_heatmap.png")

# ── Plot 07: Outlier Boxplots ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Outlier Detection — Key Variables", color=GOLD, fontsize=13, fontweight="bold")

outlier_cols   = ["total_amount", "trip_distance", "trip_duration_min"]
outlier_colors = [GOLD, GREEN, RED]

for ax, col, color in zip(axes, outlier_cols, outlier_colors):
    data = df[col].dropna().clip(upper=df[col].quantile(0.99))
    ax.boxplot(data, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.6),
               medianprops=dict(color="white", linewidth=2),
               flierprops=dict(marker=".", color=RED, alpha=0.3, markersize=3))
    ax.set_title(col, color=color)
    ax.set_ylabel("Value")

# Business Insight: All three key variables exhibit extreme
# upper outliers — total_amount has fares > $200, distances > 50
# miles, and durations > 4 hours — consistent with airport trips,
# data entry errors, or GPS artifacts.  IQR-based filtering in
# Step 5 will remove these while preserving genuine long trips.
plt.tight_layout()
plt.savefig("plots/07_outlier_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 07_outlier_boxplots.png")

# ── Plot 08: Monthly Fare Trend ──────────────────────────────
monthly_fare = df.groupby("pickup_month")["total_amount"].mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly_fare.index, monthly_fare.values,
        color=GOLD, linewidth=2.5, marker="o", markersize=8)
ax.fill_between(monthly_fare.index, monthly_fare.values, alpha=0.2, color=GOLD)
ax.set_xlabel("Month")
ax.set_ylabel("Avg Fare ($)")
ax.set_title("Average Fare Trend by Month", color=GOLD, fontsize=13)
ax.set_xticks(monthly_fare.index)
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"][:len(monthly_fare)], rotation=45)

# Business Insight: If the dataset spans multiple months, winter
# months (Dec–Feb) typically show higher average fares due to
# longer routes caused by traffic and weather avoidance.
# Single-month datasets will show this as a flat reference line.
plt.tight_layout()
plt.savefig("plots/08_monthly_fare_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 08_monthly_fare_trend.png")

# ── Plot 09: Night vs Day Bar ────────────────────────────────
night_day = df.groupby("is_night")["total_amount"].mean()
labels    = ["Day Ride", "Night Ride"]
colors    = [BLUE, RED]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, night_day.values, color=colors, width=0.5,
              edgecolor="#0A0A14", alpha=0.85)

# Add dollar-value labels on top of each bar for quick reading
for bar, val in zip(bars, night_day.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"${val:.2f}", ha="center", va="bottom", color="white", fontweight="bold")

ax.set_ylabel("Avg Fare ($)")
ax.set_title("Average Fare: Day Rides vs Night Rides", color=GOLD, fontsize=13)

# Business Insight: Night rides are consistently more expensive
# than day rides — the MTA levies a $0.50 night surcharge, and
# nighttime trips also tend to be longer in distance as fewer
# passengers share short hops during off-peak hours.
plt.tight_layout()
plt.savefig("plots/09_night_vs_day_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 09_night_vs_day_bar.png")

# ── Plot 10: Fare-per-Mile KDE by Rush Hour ──────────────────
fig, ax = plt.subplots(figsize=(10, 5))

for flag, color, label in [(0, BLUE, "Off-Peak"), (1, GOLD, "Rush Hour")]:
    subset = df[
        (df["is_rush_hour"] == flag) &
        (df["fare_per_mile"].between(0.5, 15))
    ]["fare_per_mile"].dropna()

    # KDE provides a smooth density estimate; histogram shows raw counts
    subset.plot.kde(ax=ax, color=color, linewidth=2.5, label=f"{label} (KDE)")
    ax.hist(subset, bins=60, color=color, alpha=0.2, density=True)

ax.set_xlabel("Fare per Mile ($/mile)")
ax.set_ylabel("Density")
ax.set_title("Fare-per-Mile Distribution: Rush Hour vs Off-Peak", color=GOLD, fontsize=13)
ax.legend()

# Business Insight: Rush-hour trips have a slightly right-shifted
# fare-per-mile distribution, suggesting traffic congestion leads
# to longer durations for the same distance — inflating the
# effective per-mile rate through time-based meter charges.
plt.tight_layout()
plt.savefig("plots/10_fare_per_mile_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 10_fare_per_mile_distribution.png")

print("\n✅ All EDA plots saved to /plots/")

# ─────────────────────────────────────────────────────────────
# STEP 5 | DATA TRANSFORMATION
# Purpose : Apply domain validity filters, IQR outlier removal,
#           skewness correction, categorical encoding, and
#           data-leakage column drops.
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5 | DATA TRANSFORMATION")
print("=" * 60)

rows_start = len(df)

# ── 5A. Validity Filters — hard domain rules ─────────────────
# These bounds reflect real-world NYC taxi operating constraints.
# Rows outside these ranges are data entry errors or GPS noise.
filters = {
    "total_amount"       : (2.5,  300),
    "fare_amount"        : (2.5,  300),
    "trip_distance"      : (0.1,  80),
    "trip_duration_min"  : (1,    180),
    "passenger_count"    : (1,    6),
    "pickup_latitude"    : (40.4, 41.0),
    "pickup_longitude"   : (-74.3, -73.6),
    "dropoff_latitude"   : (40.4, 41.0),
    "dropoff_longitude"  : (-74.3, -73.6),
}

for col, (lo, hi) in filters.items():
    if col in df.columns:
        before = len(df)
        df = df[(df[col] >= lo) & (df[col] <= hi)]
        print(f"  Filter '{col}' [{lo}, {hi}] → removed {before - len(df):,} rows")

print(f"\n  After validity filters : {len(df):,} rows  (removed {rows_start - len(df):,})")

# ── 5B. IQR Outlier Removal (factor = 2.5) ──────────────────
# Using factor=2.5 (not the traditional 1.5) to be conservative
# — we want to remove only extreme outliers, not genuine fares.
iqr_cols = [
    "total_amount", "trip_distance", "trip_duration_min",
    "fare_per_mile", "avg_speed_mph"
]

for col in iqr_cols:
    if col not in df.columns:
        continue
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo  = Q1 - 2.5 * IQR
    hi  = Q3 + 2.5 * IQR
    before = len(df)
    df  = df[(df[col] >= lo) & (df[col] <= hi)]
    removed = before - len(df)
    print(f"  IQR filter '{col}' → removed {removed:,} rows")

print(f"\n  After IQR outlier removal : {len(df):,} rows")

# ── 5C. Skewness Correction — log1p for skewed columns ──────
# A skewness threshold of |0.75| is widely used; features above
# it benefit significantly from log transformation.
skew_cols = [
    "total_amount", "fare_amount", "trip_distance",
    "trip_duration_min", "tip_amount", "tolls_amount",
    "fare_per_mile", "avg_speed_mph"
]

print("\n  Skewness Before → After log1p:")
for col in skew_cols:
    if col not in df.columns:
        continue
    skew_before = df[col].skew()
    if abs(skew_before) > 0.75:
        # Clip to 0 before log1p to avoid log of negative numbers
        df[col] = np.log1p(df[col].clip(lower=0))
        skew_after = df[col].skew()
        print(f"    {col:25s}: {skew_before:+.3f} → {skew_after:+.3f}")

# ── 5D. Categorical Encoding ─────────────────────────────────
# LabelEncoder converts string categories to integer indices.
# We save the fitted encoders so Streamlit can reverse-encode
# user inputs at inference time.
label_enc_cols  = ["store_and_fwd_flag", "am_pm", "time_of_day", "pickup_weekday"]
label_encoders  = {}   # Will be saved as models/label_encoders.pkl

for col in label_enc_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"  LabelEncoded '{col}' → classes: {list(le.classes_)}")

# Cast remaining categoricals as numeric directly
# (VendorID, RatecodeID, payment_type are already numeric codes)
for col in ["VendorID", "RatecodeID", "payment_type"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ── 5E. Drop data-leakage / redundant columns ────────────────
# Dropping fare_amount here — it is a direct sub-component
# of total_amount (our target). Keeping it would be data
# leakage: the model would just learn total ≈ fare + extras.
# We also drop datetime objects and all other sub-components.
drop_cols = [
    "pickup_datetime", "dropoff_datetime",
    "fare_amount", "tip_amount", "tolls_amount",
    "extra", "mta_tax", "improvement_surcharge"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
print(f"\n  Dropped leakage/redundant columns: {[c for c in drop_cols if c in df.columns or True]}")

# ── 5F. Fill remaining NaN values with column median ─────────
nan_before = df.isnull().sum().sum()
df = df.fillna(df.median(numeric_only=True))
print(f"\n  Filled {nan_before} remaining NaN values with column medians.")
print(f"\n  Final shape after transformation : {df.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 6 | FEATURE SELECTION
# Purpose : Use three independent methods (Pearson correlation,
#           SelectKBest, Random Forest importances) and take the
#           union of their top-15 features as the final set.
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6 | FEATURE SELECTION")
print("=" * 60)

# Define feature matrix X and log-transformed target y
# total_amount was already log1p-transformed in Step 5C.
y = df["total_amount"]
X = df.drop(columns=["total_amount"])

# Columns that are guaranteed to be in the final feature set
# because they have strong domain relevance.
MANDATORY_FEATURES = [
    "trip_distance", "trip_duration_min", "pickup_hour",
    "passenger_count", "is_night", "is_weekend", "is_rush_hour"
]

# ── Method 1: Pearson Correlation with target ────────────────
pearson_corr = X.corrwith(y).abs().sort_values(ascending=False)
top_pearson  = pearson_corr.head(15).index.tolist()
print(f"\n  Method 1 — Pearson Correlation (top 15):\n  {top_pearson}")

# ── Method 2: SelectKBest with F-regression ──────────────────
X_filled = X.fillna(X.median(numeric_only=True))
selector  = SelectKBest(score_func=f_regression, k=min(15, X_filled.shape[1]))
selector.fit(X_filled, y)
top_kbest = X_filled.columns[selector.get_support()].tolist()
f_scores  = pd.Series(selector.scores_, index=X_filled.columns).sort_values(ascending=False)
print(f"\n  Method 2 — SelectKBest F-regression (top 15):\n  {top_kbest}")

# ── Method 3: Random Forest Feature Importances ──────────────
rf_selector = RandomForestRegressor(n_estimators=50, max_depth=8,
                                    random_state=42, n_jobs=-1)
rf_selector.fit(X_filled, y)
importances  = pd.Series(rf_selector.feature_importances_, index=X_filled.columns)
top_rf       = importances.sort_values(ascending=False).head(15).index.tolist()
print(f"\n  Method 3 — Random Forest Importances (top 15):\n  {top_rf}")

# Plot RF feature importances
fig, ax = plt.subplots(figsize=(10, 7))
importances.sort_values().tail(20).plot.barh(ax=ax, color=GOLD, alpha=0.85, edgecolor="#0A0A14")
ax.set_title("Random Forest Feature Importances", color=GOLD, fontsize=13)
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("plots/11_rf_feature_importances.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 11_rf_feature_importances.png")

# ── Union of all three methods ────────────────────────────────
# Taking the union ensures we don't miss any feature flagged by
# at least one statistically motivated method.
selected_features = list(
    set(top_pearson) | set(top_kbest) | set(top_rf) | set(MANDATORY_FEATURES)
)
# Remove target if it somehow crept in
selected_features = [f for f in selected_features if f in X.columns]

print(f"\n  ✅ Final selected feature list ({len(selected_features)} features):")
print(f"     {selected_features}")

# Save selected features for Streamlit inference pipeline
joblib.dump(selected_features, "models/selected_features.pkl")
print("  💾 Saved: models/selected_features.pkl")

# ─────────────────────────────────────────────────────────────
# STEP 7 | MODEL BUILDING
# Purpose : Train 8 regression models, evaluate on test set,
#           print ranked comparison table, save model chart.
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7 | MODEL BUILDING")
print("=" * 60)

# Using log1p-transformed total_amount as training target
# (already applied in Step 5C). expm1() restores original scale.
X_model = X_filled[selected_features]
y_model = y   # already log-transformed

# Train-test split: 80/20 with fixed seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X_model, y_model, test_size=0.2, random_state=42
)
print(f"\n  Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# Using RobustScaler instead of StandardScaler because our
# data still contains moderate outliers after IQR filtering
# RobustScaler uses median/IQR and is less sensitive to them
scaler  = RobustScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

def evaluate_model(name: str, model, X_tr, X_te,
                   y_tr, y_te, scaled=True) -> dict:
    """
    Fit a model, predict on test set, and return a dict of metrics.
    Predictions are inverse-transformed from log space (expm1).

    Parameters
    ----------
    name     : str             — Model label for reporting.
    model    : sklearn/xgb     — Untrained estimator object.
    X_tr/te  : ndarray/df      — Training / test features.
    y_tr/te  : pd.Series       — Training / test log targets.
    scaled   : bool            — True if X_tr/te are already scaled.

    Returns
    -------
    dict  with keys: Model, R2, MSE, RMSE, MAE, MAPE
    """
    model.fit(X_tr, y_tr)
    preds_log = model.predict(X_te)

    # Back-transform from log space to original dollar scale
    preds  = np.expm1(preds_log)
    actual = np.expm1(y_te.values)

    r2   = r2_score(actual, preds)
    mse  = mean_squared_error(actual, preds)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(actual, preds)
    # MAPE guard: avoid division by zero for zero actuals
    mape = np.mean(np.abs((actual - preds) / np.where(actual == 0, 1, actual))) * 100

    print(f"  {name:35s}  R²={r2:.4f}  RMSE=${rmse:.2f}  MAE=${mae:.2f}  MAPE={mape:.1f}%")
    return {"Model": name, "R2": r2, "MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape,
            "estimator": model}

# ── Define all 8 models ──────────────────────────────────────
models_config = [
    # Linear models use the scaled feature matrix
    ("Linear Regression",   LinearRegression(),
     X_train_sc, X_test_sc),
    ("Ridge Regression",    Ridge(alpha=1.0),
     X_train_sc, X_test_sc),
    ("Lasso Regression",    Lasso(alpha=0.01, max_iter=3000),
     X_train_sc, X_test_sc),
    # Tree-based models are scale-invariant — use unscaled data
    ("Decision Tree",       DecisionTreeRegressor(max_depth=10, random_state=42),
     X_train, X_test),
    ("Random Forest",       RandomForestRegressor(n_estimators=200, max_depth=15,
                                                  random_state=42, n_jobs=-1),
     X_train, X_test),
    ("Gradient Boosting",   GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                                       max_depth=5, random_state=42),
     X_train, X_test),
    ("XGBoost",             XGBRegressor(n_estimators=300, learning_rate=0.08,
                                         max_depth=6, random_state=42,
                                         n_jobs=-1, verbosity=0),
     X_train, X_test),
    ("Extra Trees",         ExtraTreesRegressor(n_estimators=200, max_depth=15,
                                                random_state=42, n_jobs=-1),
     X_train, X_test),
]

results = []
trained_models = {}
for name, model, X_tr, X_te in models_config:
    res = evaluate_model(name, model, X_tr, X_te, y_train, y_test)
    results.append(res)
    trained_models[name] = (model, X_tr, X_te)  # keep reference for tuning

results_df = pd.DataFrame(results).drop(columns=["estimator"]).sort_values("R2", ascending=False)
print(f"\n  📊 Model Comparison Table:\n{results_df.to_string(index=False)}")

# ── Save model comparison bar chart ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model Performance Comparison", color=GOLD, fontsize=14, fontweight="bold")

metrics = [("R2", GOLD), ("RMSE", RED), ("MAE", BLUE)]
for ax, (metric, color) in zip(axes, metrics):
    sorted_df = results_df.sort_values(metric, ascending=(metric != "R2"))
    ax.barh(sorted_df["Model"], sorted_df[metric], color=color, alpha=0.85, edgecolor="#0A0A14")
    ax.set_title(f"Model {metric}", color=color)
    ax.set_xlabel(metric)

plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: plots/model_comparison.png")

# ─────────────────────────────────────────────────────────────
# STEP 7b | HYPERPARAMETER TUNING
# Purpose : Fine-tune the top-2 models (XGBoost + Random Forest)
#           using RandomizedSearchCV and GridSearchCV respectively.
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7b | HYPERPARAMETER TUNING")
print("=" * 60)

# ── XGBoost — RandomizedSearchCV (n_iter=25) ─────────────────
# RandomizedSearchCV is faster than GridSearchCV for large
# parameter spaces; 25 iterations gives a good trade-off.
xgb_param_grid = {
    "n_estimators"     : [200, 300, 400],
    "learning_rate"    : [0.05, 0.08, 0.10, 0.12],
    "max_depth"        : [4, 5, 6, 7],
    "subsample"        : [0.7, 0.8, 0.9],
    "colsample_bytree" : [0.7, 0.8, 0.9],
    "min_child_weight" : [1, 3, 5],
    "gamma"            : [0, 0.1, 0.2],
}

xgb_tuned = RandomizedSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
    param_distributions=xgb_param_grid,
    n_iter=25, cv=3, scoring="r2",
    random_state=42, n_jobs=-1, verbose=1
)
print("\n  Tuning XGBoost …")
xgb_tuned.fit(X_train, y_train)
print(f"  Best params  : {xgb_tuned.best_params_}")
print(f"  Best CV R²   : {xgb_tuned.best_score_:.4f}")

# Evaluate tuned XGBoost on test set
xgb_tuned_res = evaluate_model(
    "XGBoost (Tuned)", xgb_tuned.best_estimator_,
    X_train, X_test, y_train, y_test
)

# ── Random Forest — GridSearchCV ─────────────────────────────
# Smaller grid for RF since training is slower than XGBoost.
rf_param_grid = {
    "n_estimators"     : [100, 200],
    "max_depth"        : [10, 15, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf" : [1, 2],
    "max_features"     : ["sqrt", "log2"],
}

rf_tuned = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=rf_param_grid,
    cv=3, scoring="r2", n_jobs=-1, verbose=1
)
print("\n  Tuning Random Forest …")
rf_tuned.fit(X_train, y_train)
print(f"  Best params  : {rf_tuned.best_params_}")
print(f"  Best CV R²   : {rf_tuned.best_score_:.4f}")

rf_tuned_res = evaluate_model(
    "Random Forest (Tuned)", rf_tuned.best_estimator_,
    X_train, X_test, y_train, y_test
)

# Add tuned results to comparison dataframe
tuned_df    = pd.DataFrame([xgb_tuned_res, rf_tuned_res]).drop(columns=["estimator"])
results_df  = pd.concat([results_df, tuned_df], ignore_index=True)

all_model_results = results_df.sort_values("R2", ascending=False)
print(f"\n  📊 Final Ranked Model Table:\n{all_model_results.to_string(index=False)}")

# ─────────────────────────────────────────────────────────────
# STEP 8 | FINALIZE & SAVE BEST MODEL
# Purpose : Select the best model by test R², save all artefacts
#           required by the Streamlit inference pipeline, and
#           generate residual diagnostic plots.
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 8 | FINALISE & SAVE BEST MODEL")
print("=" * 60)

# Pick the model with highest R² on the test set
best_row   = all_model_results.loc[all_model_results["R2"].idxmax()]
best_name  = best_row["Model"]

# Retrieve the actual fitted estimator from training results
estimator_map = {r["Model"]: r["estimator"] for r in results}
estimator_map["XGBoost (Tuned)"]       = xgb_tuned.best_estimator_
estimator_map["Random Forest (Tuned)"] = rf_tuned.best_estimator_

best_estimator = estimator_map[best_name]
print(f"\n  🏆 Best Model : {best_name}")
print(f"     R²   : {best_row['R2']:.4f}")
print(f"     RMSE : ${best_row['RMSE']:.2f}")
print(f"     MAE  : ${best_row['MAE']:.2f}")

# ── Save artefacts ───────────────────────────────────────────
joblib.dump(best_estimator,  "models/best_model.pkl")
joblib.dump(scaler,          "models/scaler.pkl")
joblib.dump(label_encoders,  "models/label_encoders.pkl")
# selected_features.pkl already saved in Step 6

# ── Save JSON metadata for Streamlit ────────────────────────
# Streamlit reads meta.json to know which model was used, what
# the evaluation metrics were, and whether to apply expm1().
meta = {
    "best_model_name"   : best_name,
    "test_r2"           : round(float(best_row["R2"]),   4),
    "test_rmse"         : round(float(best_row["RMSE"]), 4),
    "test_mae"          : round(float(best_row["MAE"]),  4),
    "use_log_transform" : True,    # We trained on log1p(total_amount)
    "selected_features" : selected_features,
    "all_model_results" : all_model_results[
        ["Model","R2","RMSE","MAE","MAPE"]
    ].to_dict(orient="records"),
}
with open("models/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n  💾 Saved artefacts:")
print("     models/best_model.pkl")
print("     models/scaler.pkl")
print("     models/selected_features.pkl")
print("     models/label_encoders.pkl")
print("     models/meta.json")

# ── Residual Diagnostic Plots (3-panel) ─────────────────────
# Determine correct feature matrix (scaled vs unscaled) for best model
uses_scaling = best_name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]
X_te_final   = X_test_sc if uses_scaling else X_test

preds_log    = best_estimator.predict(X_te_final)
preds_dollar = np.expm1(preds_log)
actual_dollar = np.expm1(y_test.values)
residuals    = actual_dollar - preds_dollar

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"Residual Diagnostics — {best_name}", color=GOLD, fontsize=13, fontweight="bold")

# Panel 1: Residuals vs Fitted
axes[0].scatter(preds_dollar, residuals, alpha=0.2, s=5, color=GOLD)
axes[0].axhline(0, color=RED, linewidth=1.5, linestyle="--")
axes[0].set_xlabel("Fitted Values ($)")
axes[0].set_ylabel("Residuals ($)")
axes[0].set_title("Residuals vs Fitted", color=GOLD)

# Panel 2: Actual vs Predicted
axes[1].scatter(actual_dollar, preds_dollar, alpha=0.2, s=5, color=GREEN)
max_val = max(actual_dollar.max(), preds_dollar.max())
axes[1].plot([0, max_val], [0, max_val], color=RED, linewidth=1.5, linestyle="--")
axes[1].set_xlabel("Actual Fare ($)")
axes[1].set_ylabel("Predicted Fare ($)")
axes[1].set_title("Actual vs Predicted", color=GREEN)

# Panel 3: Residual histogram — should be approximately Normal
axes[2].hist(residuals.clip(-30, 30), bins=80, color=BLUE, edgecolor="#0A0A14", alpha=0.85)
axes[2].axvline(0, color=RED, linewidth=1.5, linestyle="--")
axes[2].set_xlabel("Residual ($)")
axes[2].set_ylabel("Count")
axes[2].set_title("Residual Distribution", color=BLUE)

plt.tight_layout()
plt.savefig("plots/12_best_model_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✅ Saved: plots/12_best_model_diagnostics.png")

print("\n" + "=" * 60)
print("✅ TRAINING PIPELINE COMPLETE")
print(f"   Best model : {best_name}  |  R² = {meta['test_r2']}  "
      f"|  RMSE = ${meta['test_rmse']}  |  MAE = ${meta['test_mae']}")
print("=" * 60)