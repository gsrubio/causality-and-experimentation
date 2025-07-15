import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest

st.set_page_config(page_title="A/B Test Runtime & Power Simulator", layout="wide")
st.title("🔬 A/B Test Runtime & Power Simulator")

# --- Sidebar Inputs for Sample Size Calculation ---
st.sidebar.header("🔢 Runtime Calculation Parameters")
confidence_level = st.sidebar.slider("Confidence Level (1 - α)", 0.80, 0.99, 0.95, 0.01)
test_type = st.sidebar.radio("Test Type", ["Two-sided", "One-sided"])
power = st.sidebar.slider("Power (1 - β)", 0.5, 0.99, 0.8, 0.01)
daily_traffic = st.sidebar.number_input("Daily Traffic (Total)", min_value=100, value=10000, step=100)
baseline_cvr = st.sidebar.number_input("Baseline CVR (%)", min_value=0.1, max_value=100.0, value=2.0, step=0.1) / 100
mde = st.sidebar.number_input("Minimum Detectable Effect (%)", min_value=0.1, max_value=100.0, value=10.0, step=0.1) / 100

# --- Z-values ---
alpha = 1 - confidence_level
z_alpha = norm.ppf(1 - alpha / (2 if test_type == "Two-sided" else 1))
z_beta = norm.ppf(power)

# --- Sample Size Formula ---
p1 = baseline_cvr
p2 = baseline_cvr * (1 + mde)
pooled_var = p1 * (1 - p1) + p2 * (1 - p2)
delta = abs(p2 - p1)
sample_size_per_variant = int(((z_alpha + z_beta)**2 * pooled_var) / (delta**2))
runtime_days = int(np.ceil((2 * sample_size_per_variant) / daily_traffic))

# --- Sample Size Output ---
st.markdown("#### 📊 Sample Size & Runtime Estimate")
sample_summary = {
    "🧪 Sample Size per Variant": f"{sample_size_per_variant:,}",
    "📆 Runtime (Days)": f"{runtime_days:,}"
}
st.dataframe(pd.DataFrame.from_dict(sample_summary, orient="index").T, use_container_width=True, hide_index=True)

# --- Simulation Inputs ---
st.sidebar.header("🧪 Simulation Parameters")
sim_runtime = st.sidebar.number_input("Simulation Runtime", min_value=1, value=runtime_days, step=1)
sim_sample_size = sim_runtime * (daily_traffic/2)
true_lift = st.sidebar.number_input("True Lift (%)", min_value=-10.0, max_value=100.0, value=10.0, step=0.1) / 100
n_simulations = 10000
alt = 'two-sided' if test_type == "Two-sided" else 'larger'

if st.sidebar.button("Run Simulations"):

    df = pd.DataFrame(columns=["obs_lift", "stat_sig"])
    p_variant = baseline_cvr * (1 + true_lift)

    for _ in range(n_simulations):
        conv_a = np.random.binomial(sim_sample_size, baseline_cvr)
        conv_b = np.random.binomial(sim_sample_size, p_variant)
        rate_a = conv_a / sim_sample_size
        rate_b = conv_b / sim_sample_size

        z_stat, p_value = proportions_ztest([conv_b, conv_a], [sim_sample_size, sim_sample_size], alternative=alt)
        stat_sig = int(p_value < alpha)
        obs_lift = (rate_b / rate_a) - 1

        df.loc[len(df)] = [obs_lift, stat_sig]

    sig_and_positive = ((df["stat_sig"] == 1) & (df["obs_lift"] > 0)).sum()
    sig_and_negative = ((df["stat_sig"] == 1) & (df["obs_lift"] < 0)).sum()
    total_sig = int(df["stat_sig"].sum())

    if true_lift > 0:
        observed_power = sig_and_positive / n_simulations
        power_summary = {
            "🔁 Simulations Ran": f"{n_simulations:,}",
            "✅ Statsig Simulations": f"{total_sig:,}",
            "📈 Statsig Simulations with Positive Lift": f"{sig_and_positive:,}",
            "⚡ Observed Power": f"{observed_power:.1%}"
        }
        st.markdown("#### 📈 Power Simulation Summary")
        st.dataframe(pd.DataFrame.from_dict(power_summary, orient="index").T, use_container_width=True, hide_index=True)

    elif test_type == "One-sided":
        false_positive_rate = total_sig / n_simulations
        false_positive_winners = sig_and_positive / n_simulations
        fp_summary = {
            "🔁 Simulations Ran": f"{n_simulations:,}",
            "✅ Statsig Simulations": f"{total_sig:,}",
            "📈 Statsig Simulations with Positive Lift": f"{sig_and_positive:,}",
            "🚨 False Alarm Rate": f"{false_positive_rate:.1%}",
            "🚨 False 'Winners'": f"{false_positive_winners:.1%}"
        }
        st.markdown("#### 🚨 False Positive Summary")
        st.dataframe(pd.DataFrame.from_dict(fp_summary, orient="index").T, use_container_width=True, hide_index=True)

    
    elif test_type == "Two-sided" and true_lift == 0:
        false_positive_rate = total_sig / n_simulations
        false_positive_winners = sig_and_positive / n_simulations
        false_negative_winners = sig_and_negative / n_simulations
        power_summary = {
            "🔁 Simulations Ran": f"{n_simulations:,}",
            "✅ Statsig Simulations": f"{total_sig:,}",
            "📈 Statsig Simulations with Positive Lift": f"{sig_and_positive:,}",
            "🚨 False Alarm Rate": f"{false_positive_rate:.1%}",
            "🚨 False 'Winners'": f"{false_positive_winners:.1%}",
            "🚨 False 'Losers'": f"{false_negative_winners:.1%}"
        }
        st.markdown("#### 📈 Power Simulation Summary")
        st.dataframe(pd.DataFrame.from_dict(power_summary, orient="index").T, use_container_width=True, hide_index=True)

    
    else:
        observed_power = sig_and_negative / n_simulations
        fp_summary = {
            "🔁 Simulations Ran": f"{n_simulations:,}",
            "✅ Statsig Simulations": f"{total_sig:,}",
            "📉 Statsig Simulations with Negative Lift": f"{sig_and_negative:,}",
            "⚡ Observed Power": f"{observed_power:.1%}"
        }
        st.markdown("#### 🚨 False Positive Summary")
        st.dataframe(pd.DataFrame.from_dict(fp_summary, orient="index").T, use_container_width=True, hide_index=True)

# --- Histogram Visualization ---
    st.markdown("#### 🔍 Distribution of Observed Lifts")
    df["obs_lift_pct"] = df["obs_lift"] * 100  # for hover formatting

    fig = px.histogram(
        df,
        x="obs_lift",
        color="stat_sig",
        nbins=50,
        template="plotly_white",
        color_discrete_map={1: ("blue" if true_lift > 0 else "red"), 0: "gray"},
        hover_data={"obs_lift_pct": ':.1f'},
        labels={"obs_lift": "Observed Lift", "stat_sig": "Stat Sig"}
    )
    fig.update_layout(
        xaxis_tickformat=".0%",
        hovermode="x unified"
    )

    if true_lift > 0:
        # Light gray for MDE
        fig.add_vline(
            x=mde,
            line_dash="dash",
            line_color="#808080",
            annotation_text=f"MDE: {mde:.1%}",
            annotation_position="top right"
        )
        # Dark gray for true lift
        fig.add_vline(
            x=true_lift,
            line_dash="dashdot",
            line_color="#444444",
            annotation_text=f"True Lift: {true_lift:.1%}",
            annotation_position="top left"
        )
    else: 
        # Light gray for MDE
       fig.add_vline(
            x=true_lift,
            line_dash="dashdot",
            line_color="#444444",
            annotation_text=f"True Lift: {true_lift:.1%}",
            annotation_position="top left"
       )

    st.plotly_chart(fig)
