import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from statsmodels.stats.proportion import proportions_ztest

st.title("P-Values Simulation App")
st.write("Simulate A/B tests and visualize the distribution of observed effects and p-values.")

# Sidebar inputs
st.sidebar.header("Simulation Parameters")
n_tests = st.sidebar.number_input("Number of A/B tests", min_value=100, max_value=50000, value=10000, step=100)
n_obs = st.sidebar.number_input("Sample Size per Variant", min_value=100, max_value=1000000, value=250000, step=1000)
conv_control = st.sidebar.number_input("Baseline CVR (Control)", min_value=0.0001, max_value=0.5, value=0.0145, step=0.0001, format="%.4f")
effect = st.sidebar.number_input("True Effect (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.3f")
alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.5, 0.2, 0.01)
test_type = st.sidebar.radio("Test Type", ["two-sided", "larger"])

conv_variant = conv_control * (1 + effect)

if st.sidebar.button("Run Simulation"):
    np.random.seed(84)
    df = pd.DataFrame(columns=("n", "stat_sig", "obs_effect", "p_value"))
    for _ in range(n_tests):
        # Control group
        success_A = np.random.binomial(n_obs, conv_control)
        trials_A = n_obs
        # Variant group
        success_B = np.random.binomial(n_obs, conv_variant)
        trials_B = n_obs
        # Hypothesis test
        count = np.array([success_B, success_A])
        nobs = np.array([trials_B, trials_A])
        z_stat, p_value = proportions_ztest(count, nobs, alternative=test_type)
        stat_sig = 1 if p_value < alpha else 0
        obs_effect = (success_B / success_A) - 1 if success_A > 0 else np.nan
        record = [n_obs, stat_sig, obs_effect, p_value]
        df.loc[len(df)] = record

    # Summary
    summary = {
        "Total Tests": f"{df['n'].count():,}",
        "Significant Tests": f"{int(df['stat_sig'].sum()):,}",
        "Power": f"{df['stat_sig'].mean():.2%}",
        "Avg Observed Effect": f"{df['obs_effect'].mean():.2%}",
    }
    st.subheader("📊 Summary of Simulated A/B Tests")
    st.dataframe(pd.DataFrame([summary]))

    # Histogram of observed effects
    st.subheader("Distribution of Observed Effects")
    fig1 = px.histogram(
        df,
        x="obs_effect",
        nbins=50,
        template="plotly_dark",
        height=300
    )
    fig1.add_vline(
        x=effect,
        line_dash="dash",
        line_color="red",
        annotation_text=f"True Effect: {effect:.2%}",
        annotation_position="top"
    )
    fig1.update_layout(xaxis=dict(tickformat=".0%", title="Observed Effect"))
    st.plotly_chart(fig1)

    # Histogram of p-values
    st.subheader("Distribution of P-values")
    fig2 = px.histogram(
        df,
        x="p_value",
        nbins=50,
        template="plotly_dark",
        height=300
    )
    fig2.add_vline(
        x=alpha,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Alpha: {alpha:.2f}",
        annotation_position="top"
    )
    fig2.update_layout(xaxis=dict(title="P-value"))
    st.plotly_chart(fig2)

    # Scatter plot: Observed Effect vs P-value
    st.subheader("Observed Effect vs P-value")
    fig3 = px.scatter(
        df,
        x="obs_effect",
        y="p_value",
        template="plotly_dark",
        height=400,
        opacity=0.7
    )
    fig3.add_hline(
        y=alpha,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Alpha: {alpha:.2f}",
        annotation_position="top left"
    )
    fig3.add_vline(
        x=effect,
        line_dash="dash",
        line_color="red",
        annotation_text=f"True Effect: {effect:.2%}",
        annotation_position="top right"
    )
    fig3.update_layout(
        xaxis=dict(tickformat=".0%", title="Observed Effect"),
        yaxis=dict(title="P-value", range=[0, 1])
    )
    st.plotly_chart(fig3)

    # Scatter plot: Observed Effect vs Confidence
    st.subheader("Observed Effect vs Confidence (1 - p/2)")
    df["confidence"] = 1 - (df["p_value"] / 2)
    confidence_at_alpha = 1 - (alpha / 2)
    fig4 = px.scatter(
        df,
        x="obs_effect",
        y="confidence",
        template="plotly_dark",
        height=400,
        opacity=0.7
    )
    fig4.add_hline(
        y=confidence_at_alpha,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Conf. at α: {confidence_at_alpha:.2f}",
        annotation_position="top left"
    )
    fig4.add_vline(
        x=effect,
        line_dash="dash",
        line_color="red",
        annotation_text=f"True Effect: {effect:.2%}",
        annotation_position="top right"
    )
    fig4.update_layout(
        xaxis=dict(tickformat=".0%", title="Observed Effect"),
        yaxis=dict(title="Confidence (1 - p/2)", range=[0, 1.1])
    )
    st.plotly_chart(fig4)