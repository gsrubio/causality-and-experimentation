import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import norm

st.set_page_config(page_title="Sample Size & Runtime Explorer", layout="wide")
st.title("📐 Sample Size & Runtime Explorer")

# --- Sidebar Inputs ---
st.sidebar.header("📊 Input Parameters")
daily_traffic = st.sidebar.number_input("Daily Traffic (Total)", min_value=100, value=10000, step=100)
baseline_cvr = st.sidebar.number_input("Baseline CVR (%)", min_value=0.1, max_value=100.0, value=2.0, step=0.1) / 100
mde = st.sidebar.number_input("Minimum Detectable Effect (%)", min_value=0.1, max_value=100.0, value=10.0, step=0.1) / 100
test_type = st.sidebar.radio("Test Type", ["Two-sided", "One-sided"])
conf_level = st.sidebar.slider("Confidence Level (1 - α)", 0.80, 0.99, 0.95, 0.01)
power = st.sidebar.slider("Power (1 - β)", 0.5, 0.99, 0.8, 0.01)
#true_positive_rate = st.sidebar.slider("True Positive Rate (Simulated Ground Truth)", 0.0, 1.0, 0.5, 0.01)
#true_lift = st.sidebar.number_input("True Lift for Positives (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1) / 100

# --- Calculations ---
alpha = 1 - conf_level
z_alpha = norm.ppf(1 - alpha / (2 if test_type == "Two-sided" else 1))
z_beta = norm.ppf(power)

p1 = baseline_cvr
p2 = baseline_cvr * (1 + mde)
pooled_var = p1 * (1 - p1) + p2 * (1 - p2)
delta = abs(p2 - p1)
sample_size_per_variant = int(((z_alpha + z_beta)**2 * pooled_var) / (delta**2))
runtime_days = int(np.ceil((2 * sample_size_per_variant) / daily_traffic))

# --- Output Table ---
st.subheader("📅 Sample Size & Runtime Estimate")
result_df = pd.DataFrame({
    "Sample Size per Variant": [sample_size_per_variant],
    "Estimated Runtime (Days)": [runtime_days]
})
st.dataframe(result_df, use_container_width=True, hide_index=True)

# --- Chart 4: Runtime vs Power by Test Type & Confidence ---
runtime_by_power_type_conf = []
test_types = ["One-sided", "Two-sided"]
special_conf_levels = [0.80, 0.90]
powers = np.arange(0.5, 0.96, 0.05)

for t_type in test_types:
    for cl in special_conf_levels:
        z_alpha_cl = norm.ppf(1 - (1 - cl) / (2 if t_type == "Two-sided" else 1))
        for pwr in powers:
            z_beta_pwr = norm.ppf(pwr)
            ss = ((z_alpha_cl + z_beta_pwr) ** 2 * pooled_var) / (delta ** 2)
            rt = (2 * ss) / daily_traffic

            label = f"{t_type} - {int(cl * 100)}%"
            runtime_by_power_type_conf.append({
                "Power": pwr,
                "Sample Size": int(ss),
                "Runtime (Days)": float(f"{rt:.2f}"),  # ensure numerical type for plotting
                "Tail - Conf. level": label
            })

st.subheader("🔍 Runtime vs Power by Test Type & Confidence")
df4 = pd.DataFrame(runtime_by_power_type_conf)
fig4 = px.line(
    df4,
    x="Power",
    y="Runtime (Days)",
    color="Tail - Conf. level",
    markers=True,
    hover_data=["Sample Size"]
)
st.plotly_chart(fig4, use_container_width=True)
