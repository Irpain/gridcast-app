import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import timedelta

# 1) Page config — the very first Streamlit call
st.set_page_config(
    page_title="GridCast — 24 h Raw‐Load Forecast",
    layout="wide",
)

# 2) Load your trained raw‐load pipeline
@st.cache_resource
def load_pipeline(path="gridcast_raw_pipeline.pkl"):
    return joblib.load(path)

pipe = load_pipeline()

# 3) Fetch Madrid weather (forecast vs archive) — replace your existing version with this:
@st.cache_data
def fetch_madrid_weather(forecast_date: pd.Timestamp) -> pd.DataFrame:
    """
    For strictly past dates (forecast_date < today) → archive API (uses snowfall_sum).
    For today/future            → forecast API (uses snowfall), with automatic fallback.
    """
    date_str  = forecast_date.strftime("%Y-%m-%d")
    today_utc = pd.Timestamp.utcnow().date()

    # build URLs + parameter templates
    archive_url  = "https://archive-api.open-meteo.com/v1/archive"
    forecast_url = "https://api.open-meteo.com/v1/forecast"

    # which endpoint + which snowfall field?
    if forecast_date.date() < today_utc:
        endpoint = archive_url
        snowfall = "snowfall_sum"
    else:
        endpoint = forecast_url
        snowfall = "snowfall"

    hourly_vars = [
        "temperature_2m",
        "relativehumidity_2m",
        "pressure_msl",
        "cloudcover",
        "windspeed_10m",
        "winddirection_10m",
        "precipitation",
        snowfall,
    ]

    params = {
        "latitude":   40.4168,
        "longitude": -3.7038,
        "hourly":    ",".join(hourly_vars),
        "start_date": date_str,
        "end_date":   date_str,
        "timezone":   "UTC",
    }

    # try the chosen endpoint, fallback archive→forecast if necessary
    r = requests.get(endpoint, params=params)
    if endpoint is archive_url and r.status_code != 200:
        # archive sometimes rejects certain days → fall back to forecast
        r = requests.get(forecast_url, params={
            **params,
            "hourly": params["hourly"].replace("snowfall_sum", "snowfall")
        })
    r.raise_for_status()

    data = r.json().get("hourly", {})
    df   = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    # rename to match your training data
    df = df.rename(columns={
        "temperature_2m":      "temp",
        "relativehumidity_2m": "humidity",
        "pressure_msl":        "pressure",
        "cloudcover":          "clouds_all",
        "windspeed_10m":       "wind_speed",
        "winddirection_10m":   "wind_deg",
        "precipitation":       "rain_1h",
        "snowfall":            "snow_1h",
        "snowfall_sum":        "snow_3h",
    })

    # if we only got snow_1h, roll it into snow_3h
    if "snow_1h" in df.columns and "snow_3h" not in df.columns:
        df["snow_3h"] = df["snow_1h"].rolling(3, min_periods=1).sum()

    # temp min/max and rolling rain sum (always needed)
    df["temp_min"] = df["temp"]
    df["temp_max"] = df["temp"]
    df["rain_3h"]  = df["rain_1h"].rolling(3, min_periods=1).sum()

    # ensure we return exactly your training weather columns
    for c in pipe["weather_cols"]:
        if c not in df.columns:
            df[c] = 0.0

    return df[pipe["weather_cols"]]

# 4) Raw‐feature builder exactly as in training
def build_raw_features(
    df_hist: pd.DataFrame,
    weather_df: pd.DataFrame,
    lags: list[int],
    rolls: list[int],
) -> pd.DataFrame:
    idx   = weather_df.index
    feats = pd.DataFrame(index=idx)

    # lag features
    for lag in lags:
        feats[f"lag_{lag}"] = df_hist["load_MW"].shift(lag).reindex(idx)

    # rolling stats
    for r in rolls:
        rolled = df_hist["load_MW"].shift(1).rolling(r)
        feats[f"roll_mean_{r}"] = rolled.mean().reindex(idx)
        feats[f"roll_std_{r}"]  = rolled.std().reindex(idx)

    # calendar
    feats["hour"]  = idx.hour
    feats["dow"]   = idx.dayofweek
    feats["month"] = idx.month

    # join weather
    feats = feats.join(weather_df, how="left")
    return feats.fillna(0)

# 5) UI — upload yesterday’s actuals
st.title("🔌 GridCast — 24 h Raw‐Load Forecast")
st.markdown("""
Upload **“Yesterday’s actuals”** CSV  
(≥ 168 h of `load_MW`, hourly, with a datetime column).
We’ll auto‑fetch Madrid’s weather and predict the next 24 h.
""")

hist_file = st.file_uploader("Upload historical actuals CSV", type="csv")
if not hist_file:
    st.info("Please upload yesterday’s actuals CSV.")
    st.stop()

# read & validate
hist_df = (
    pd.read_csv(hist_file, parse_dates=[0], index_col=0)
      .asfreq("h")
)
if "load_MW" not in hist_df.columns:
    st.error("CSV must contain a `load_MW` column.")
    st.stop()

# 6) determine forecast window
last_ts       = hist_df.index.max()
forecast_start = last_ts + pd.Timedelta(hours=1)
forecast_idx   = pd.date_range(forecast_start, periods=24, freq="h")
forecast_date  = forecast_start.date()

# 7) check history ≥ 168 h
min_req  = max(pipe["lags"] + pipe["rolls"])
earliest = forecast_start - pd.Timedelta(hours=min_req)
have_from = hist_df.index.min()
if have_from > earliest:
    st.error(
      f"Need at least {min_req} h of history before {forecast_start} (so your file must start on or before {earliest}).\n"
      f"You only have data from {have_from} onward."
    )
    st.stop()

# 8) fetch weather
weather_df = fetch_madrid_weather(pd.to_datetime(forecast_date))

# preview
st.subheader("Inputs Preview")
st.dataframe(pd.concat([
    hist_df.tail(3)[["load_MW"]],
    pd.DataFrame(index=forecast_idx)  # empty rows for forecast
], axis=0).assign(_type=lambda df: 
    np.where(df.index <= last_ts, "actual", "– forecast –")
).head(10))

# 9) build features & predict
feats = build_raw_features(
    df_hist   = hist_df.reindex(forecast_idx.union(hist_df.index)),
    weather_df= weather_df.reindex(forecast_idx),
    lags      = pipe["lags"],
    rolls     = pipe["rolls"],
)

# reorder
feats = feats[pipe["feature_cols"]]

X_scaled = pipe["scaler"].transform(feats.values)
pred      = pipe["lgbm"].predict(X_scaled)

out = pd.DataFrame({"forecast_MW": pred}, index=forecast_idx)

# 10) display
st.subheader(f"🔮 Forecast for next 24 h starting {forecast_start}")
st.line_chart(out["forecast_MW"])
st.subheader("Forecast Table")
st.dataframe(out)