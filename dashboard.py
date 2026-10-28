"""Optional Streamlit dashboard: streamlit run dashboard.py"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from smartwatt.data import generate_synthetic, load_uci
from smartwatt.evaluate import run_once

st.title("SmartWatt: electricity anomaly detection")
UCI = "data_uci/household_power_consumption.txt"


@st.cache_data(show_spinner="Loading real UCI data...")
def real_data(seed):
    return load_uci(UCI, seed=seed)


@st.cache_data(show_spinner="Running detectors...")
def run(df):
    return run_once(df)


sources = ["Synthetic (with drift)", "Upload CSV"]
if os.path.exists(UCI):
    sources.insert(1, "Real UCI household (injected anomalies)")
src = st.sidebar.radio("Data", sources)
if src.startswith("Real"):
    df = real_data(int(st.sidebar.number_input("Injection seed", 0, 999, 0)))
elif src.startswith("Synthetic"):
    seed = st.sidebar.number_input("Seed", 0, 999, 0)
    df = generate_synthetic(seed=int(seed))
else:
    up = st.sidebar.file_uploader("CSV with timestamp and consumption columns")
    if up is None:
        st.info("Upload a CSV, or switch to synthetic data.")
        st.stop()
    df = pd.read_csv(up, parse_dates=["timestamp"])
    if "is_anomaly" not in df:
        df["is_anomaly"] = -1

table, feat, preds, cut = run(df)
method = st.sidebar.selectbox("Detector", list(preds))
if table is not None:
    st.subheader("Test-set metrics")
    st.dataframe(table.style.format("{:.3f}"))

feat = feat.assign(flagged=preds[method])
days = st.sidebar.slider("Days to show", 3, 60, 14)
view = feat.iloc[cut:cut + days * 24].set_index("timestamp")
st.subheader(f"Consumption and detections ({method})")
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(view.index, view["consumption"], lw=0.8, color="gray", label="consumption")
truth = view[view["is_anomaly"] == 1]
ax.scatter(truth.index, truth["consumption"], s=90, facecolors="none",
           edgecolors="green", linewidths=1.4, label="true anomaly")
hit = view[view["flagged"] == 1]
ax.scatter(hit.index, hit["consumption"], s=18, color="red", label="flagged")
ax.set_ylabel("consumption"); ax.legend(loc="upper left", fontsize=8)
st.pyplot(fig)
st.write(f"Flagged {int(view['flagged'].sum())} readings in this window.")
st.dataframe(view.loc[view["flagged"] == 1, ["consumption", "is_anomaly"]])
