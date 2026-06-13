from pathlib import Path
import pickle

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "odisha_flood_resource_demand.csv"
MODEL_PATH = ROOT / "models" / "ration_demand_model.joblib"
REPORT_DIR = ROOT / "reports"


st.set_page_config(page_title="NGO Relief Demand Forecast", layout="wide")
st.title("Flood Relief Kit Demand Forecast")
st.caption("Coastal Odisha demo for NGO resource pre-positioning")


@st.cache_data
def load_data():
    history = pd.read_csv(DATA_PATH, parse_dates=["date"])
    forecast = pd.read_csv(REPORT_DIR / "forecast_next_30_days.csv", parse_dates=["date"])
    accuracy = pd.read_csv(REPORT_DIR / "accuracy_report.csv")
    importance = pd.read_csv(REPORT_DIR / "feature_importance.csv")
    return history, forecast, accuracy, importance


history, forecast, accuracy, importance = load_data()

with st.sidebar:
    st.header("Planning Controls")
    block = st.selectbox("Block", sorted(forecast["block"].unique()))
    horizon = st.slider("Forecast horizon", min_value=7, max_value=30, value=14, step=1)
    buffer_pct = st.slider("Safety buffer", min_value=0, max_value=40, value=15, step=5)

selected_forecast = forecast[forecast["block"] == block].head(horizon)
selected_history = history[history["block"] == block].tail(120)

total_kits = int(selected_forecast["dry_ration_kits_needed"].sum())
buffered_kits = int(round(total_kits * (1 + buffer_pct / 100)))
peak_day = selected_forecast.loc[selected_forecast["dry_ration_kits_needed"].idxmax()]
model_mape = float(accuracy.loc[accuracy["metric"] == "MAPE", "value"].iloc[0])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Base kits needed", f"{total_kits:,}")
col2.metric("With buffer", f"{buffered_kits:,}")
col3.metric("Peak demand day", peak_day["date"].strftime("%d %b %Y"))
col4.metric("Model MAPE", f"{model_mape:.1f}%")

decision = "PRE-POSITION NOW" if selected_forecast["forecast_upper"].sum() > total_kits * 1.15 else "MONITOR"
st.subheader(decision)
st.write(
    f"For {block}, prepare **{buffered_kits:,} dry ration kits** for the next {horizon} days. "
    "The shaded forecast band is a practical uncertainty range for planning stock buffers."
)

chart = px.line(
    selected_forecast,
    x="date",
    y=["forecast_lower", "dry_ration_kits_needed", "forecast_upper"],
    markers=True,
    labels={"value": "Kits", "date": "Date", "variable": "Forecast"},
    title=f"{horizon}-Day Demand Forecast: {block}",
)
st.plotly_chart(chart, use_container_width=True)

left, right = st.columns(2)
with left:
    hist_chart = px.line(
        selected_history,
        x="date",
        y=["rainfall_mm", "dry_ration_kits_needed"],
        title="Recent Rainfall and Kit Demand",
        labels={"value": "Value", "date": "Date", "variable": "Signal"},
    )
    st.plotly_chart(hist_chart, use_container_width=True)

with right:
    top_features = importance.head(10)
    feature_chart = px.bar(
        top_features.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Top Model Drivers",
    )
    st.plotly_chart(feature_chart, use_container_width=True)

st.subheader("Allocation Table")
allocation = selected_forecast[
    ["date", "district", "block", "rainfall_mm", "river_level_m", "flood_incidents", "dry_ration_kits_needed", "forecast_lower", "forecast_upper"]
].copy()
allocation["date"] = allocation["date"].dt.strftime("%Y-%m-%d")
st.dataframe(allocation, use_container_width=True, hide_index=True)
