import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm


# --- Sidebar: Metric Selection & Common Inputs ---
st.sidebar.header("Simulation Parameters")
metric = st.sidebar.selectbox("📊 Select the metric:", ["CVR (Conversion Rate)", "Sales per Visitor"])

n_obs = st.sidebar.number_input("Sample Size per Variant", min_value=100, max_value=1000000, value=10000, step=1000)
lift = st.sidebar.number_input("True Lift (%)", min_value=0.001, max_value=0.5, value=0.015, step=0.0001, format="%.3f")  # Relative lift
observed_lift = st.sidebar.number_input("Observed Lift (for comparison)", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.3f")
alpha = st.sidebar.slider("One-Sided Significance Level (α)", 0.01, 0.1, 0.05, 0.01)

n_tests = 10000

# --- Metric-specific inputs ---
if metric == "CVR (Conversion Rate)":
    conv_control = st.sidebar.number_input("Baseline CVR", min_value=0.001, max_value=0.5, value=0.015, step=0.0001, format="%.3f")
    conv_variant = conv_control * (1 + lift)
elif metric == "Sales per Visitor":
    mean_control = st.sidebar.number_input("Control Sales/Visitor", value=2.25)
    std_dev = st.sidebar.number_input("Standard Deviation", value=19.5)
    mean_variant = mean_control * (1 + lift)

# --- Run Simulation ---
if st.sidebar.button("Run Simulation"):

    df = pd.DataFrame(columns=('n', 'obs_effect', 'conf_level', 'stat_sig', 'obs_control', 'obs_variant'))

    for _ in range(n_tests):
        if metric == "CVR (Conversion Rate)":
            success_A = np.random.binomial(n_obs, conv_control)
            success_B = np.random.binomial(n_obs, conv_variant)

            obs_control = success_A / n_obs
            obs_variant = success_B / n_obs

            # Z-test for proportions
            z_stat, p_value = proportions_ztest([success_B, success_A], [n_obs, n_obs], alternative='larger')

        elif metric == "Sales per Visitor":
            group_A = np.random.normal(loc=mean_control, scale=std_dev, size=n_obs)
            group_B = np.random.normal(loc=mean_variant, scale=std_dev, size=n_obs)

            obs_control = group_A.mean()
            obs_variant = group_B.mean()

            # T-test for independent samples (1-sided, equal variance assumed)
            se = np.sqrt((std_dev**2 / n_obs) * 2)
            t_stat = (obs_variant - obs_control) / se
            p_value = 1 - norm.cdf(t_stat)

        obs_effect = (obs_variant / obs_control) - 1
        stat_sig = 1 if p_value < alpha else 0
        conf_level = 1 - p_value

        df.loc[len(df)] = [n_obs, obs_effect, conf_level, stat_sig, obs_control, obs_variant]

    # --- Summary ---
    pct_gte_observed = (df["obs_effect"] >= observed_lift).mean()
    summary = {
        "Total Tests": f"{df['n'].count():,}",
        "Significant Tests": f"{int(df['stat_sig'].sum()):,}",
        "Observed Power": f"{df['stat_sig'].mean():.1%}",
        "True Lift": f"{lift:.1%}",
        f"Tests ≥ {observed_lift:.1%} Observed Lift": f"{pct_gte_observed:.1%}",
    }

    st.subheader(f"📊 Summary: {metric}")
    st.dataframe(pd.DataFrame([summary]))

    # --- Histogram: Observed Lifts ---
    st.subheader("Distribution of Observed Lifts")
    fig1 = px.histogram(
        df,
        x="obs_effect",
        color='stat_sig',
        color_discrete_map={0: "red", 1: "blue"},
        hover_data=df.columns,
        template='plotly_dark',
        height=300
    )
    fig1.add_vline(
        x=df["obs_effect"].mean(),
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Mean: {df['obs_effect'].mean():.2%}"
    )
    fig1.update_layout(xaxis=dict(tickformat=".0%", title="Observed Lift"))
    st.plotly_chart(fig1)

    # --- Histogram: Confidence Levels ---
    st.subheader("Distribution of Confidence Levels")
    fig2 = px.histogram(
        df,
        x="conf_level",
        color='stat_sig',
        color_discrete_map={0: "red", 1: "blue"},
        nbins=10,
        height=300
    )
    fig2.update_traces(
        xbins=dict(start=0, end=1.001, size=(1.0000001 - 0) / 10)
    )
    st.plotly_chart(fig2)
