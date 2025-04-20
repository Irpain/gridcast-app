# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1) Page config (first call)
st.set_page_config(
    page_title="GridCast — 24 h Energy Demand Forecast",
    layout="wide",
)

# 2) Load pipeline
@st.cache_resource
def load_pipeline(path="gridcast_pipeline.pkl"):
    return joblib.load(path)
pipe = load_pipeline()

# 3) Feature‐builder (unchanged)
def build_residual_features(iso_df, weather_df, df_full, lags, rolls):
    feats = pd.DataFrame(index=iso_df.index)
    for lag in lags:
        feats[f"lag_{lag}"] = df_full["load_MW"].shift(lag).reindex(feats.index)
    for r in rolls:
        rolled = df_full["load_MW"].shift(1).rolling(r)
        feats[f"roll_mean_{r}"] = rolled.mean().reindex(feats.index)
        feats[f"roll_std_{r}"]  = rolled.std().reindex(feats.index)
    feats["hour"]  = feats.index.hour
    feats["dow"]   = feats.index.dayofweek
    feats["month"] = feats.index.month
    feats = feats.join(weather_df, how="left")
    return feats.fillna(0)

# 4) UI
st.title("🔌 GridCast — 24 h Energy Demand Forecast")
st.markdown("Use the date picker to choose your forecast horizon, then upload both CSVs:")

# → new: let the user pick the **start** of the 24 h window
start_date = st.sidebar.date_input(
    "Forecast start date",
    value=pd.Timestamp.now().floor("D").date()
)
# We'll build an hourly index from start_date 00:00 → +23:00
t0 = pd.Timestamp(start_date)
t1 = t0 + pd.Timedelta(hours=23)

iso_file     = st.file_uploader("ISO 24 h baseline forecast CSV",   type="csv")
weather_file = st.file_uploader("Weather forecast CSV (next 24 h)", type="csv")

if iso_file and weather_file:
    # — read uploads
    iso_df     = pd.read_csv(iso_file,    parse_dates=[0], index_col=0).asfreq("h")
    weather_df = pd.read_csv(weather_file, parse_dates=[0], index_col=0).asfreq("h")

    # — slice to the 24 h window the user selected
    iso_df     = iso_df.loc[t0 : t1]
    weather_df = weather_df.loc[t0 : t1]

    # — sanity checks
    if iso_df.shape[0] != 24 or weather_df.shape[0] != 24:
        st.error("Your CSVs must cover exactly 24 hourly rows from "
                 f"{t0} to {t1} (inclusive).")
        st.stop()

    if "load_forecast" not in iso_df.columns:
        st.error("ISO file must include a `load_forecast` column.")
        st.stop()

    # — ensure all weather cols exist
    for c in pipe["weather_cols"]:
        if c not in weather_df.columns:
            st.warning(f"Filling missing weather column `{c}` with zeros.")
            weather_df[c] = 0.0
    weather_df = weather_df[pipe["weather_cols"]]

    # — preview
    st.subheader("Uploaded Data Preview")
    st.dataframe(pd.concat([iso_df[["load_forecast"]], weather_df], axis=1))

    # — build features & predict
    feats = build_residual_features(
        iso_df[["load_forecast"]], weather_df,
        pipe["df_full"], pipe["lags"], pipe["rolls"]
    )

    # — drop any extras & check missing
    feats = feats[pipe["feature_cols"]]
    if set(pipe["feature_cols"]) - set(feats.columns):
        st.error("Feature mismatch after slicing. Aborting.")
        st.stop()

    X_raw    = feats.values
    X_scaled = pipe["scaler_r"].transform(X_raw)
    res_pred = pipe["lgbm_resid"].predict(X_scaled)
    final_fc = iso_df["load_forecast"].values + res_pred

    # — show results
    result = pd.DataFrame({"forecast_MW": final_fc}, index=feats.index)
    st.subheader("24 h Demand Forecast")
    st.line_chart(result)
    st.subheader("Forecast Table")
    st.dataframe(result)

else:
    st.info("Please upload **both** CSV files.")